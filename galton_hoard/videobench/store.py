"""Video-bank runs and items in Galton's database (tables ``vb_runs`` and ``vb_items``, migration 11)."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Iterable, Optional

from ..db import Database
from ..errors import GaltonError
from ..messages import recognise
from ..util import new_id

RUN_JSON = ("prompts", "settings", "progress", "summary")
RUN_FIELDS = ("label", "state", "model", "jury", "prompts", "settings", "progress", "total", "done", "source", "caller", "error", "cancel", "summary", "started_ts", "finished_ts")
ITEM_JSON = ("static", "render", "auto", "jury")
ITEM_FIELDS = ("state", "outcome", "reason", "detail", "html_chars", "tokens_in", "tokens_out", "latency_ms", "static", "render", "auto", "jury", "jury_mode", "mean", "folder")
ACTIVE = ("queued", "running")
TERMINAL = ("done", "failed", "cancelled")
_DEFAULTS: dict[str, Any] = {"prompts": [], "settings": {}, "progress": {}, "summary": {}, "static": {}, "render": {}, "auto": {}, "jury": {}}


def _enc(value: Any) -> Any:
    if isinstance(value, bool):
        return int(value)
    return value if isinstance(value, (str, int, float)) or value is None else json.dumps(value, ensure_ascii=False, default=str)


def _loads(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return default if value is None else value
    try:
        return json.loads(value)
    except ValueError:
        return default


def _run(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    data = dict(row)
    for key in RUN_JSON:
        data[key] = _loads(data.get(key), _DEFAULTS[key])
    data["cancel"] = bool(data.get("cancel"))
    data["error"] = recognise(data.get("error"))
    data["label"] = recognise(data.get("label"), prefix="label_")
    return data


def _item(row: Any) -> dict[str, Any]:
    data = dict(row)
    for key in ITEM_JSON:
        data[key] = _loads(data.get(key), _DEFAULTS[key])
    data["detail"] = recognise(data.get("detail"))
    return data


class VideoStore:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db, self.clock = db, clock

    # ------------------------------------------------------------------ runs
    def create_run(self, **fields: Any) -> dict[str, Any]:
        rid = new_id("vb")
        data = {k: _enc(v) for k, v in fields.items() if k in RUN_FIELDS}
        cols = ["id", "created_ts", *data]
        self.db.execute(f"INSERT INTO vb_runs({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [rid, self.clock(), *data.values()])
        return self.run(rid)

    def update_run(self, rid: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in RUN_FIELDS}
        if data:
            self.db.execute(f"UPDATE vb_runs SET {', '.join(f'{k} = ?' for k in data)} WHERE id = ?", [*data.values(), rid])
        return self.run(rid)

    def find_run(self, ref: str) -> Optional[dict[str, Any]]:
        ref = (ref or "").strip()
        return _run(self.db.one("SELECT * FROM vb_runs WHERE id = ?", (ref,)))

    def run(self, ref: str) -> dict[str, Any]:
        row = self.find_run(ref)
        if row is None:
            raise GaltonError("not_found", "vb_no_run", id=ref)
        return row

    def runs(self, *, states: Optional[Iterable[str]] = None, limit: int = 30) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM vb_runs WHERE 1=1", []
        if states:
            states = list(states)
            sql += f" AND state IN ({', '.join('?' for _ in states)})"
            params.extend(states)
        sql += " ORDER BY created_ts DESC, id DESC LIMIT ?"
        params.append(max(1, min(int(limit), 500)))
        return [_run(r) for r in self.db.query(sql, params)]

    def delete_run(self, rid: str) -> None:
        self.db.execute("DELETE FROM vb_items WHERE run_id = ?", (rid,))
        self.db.execute("DELETE FROM vb_runs WHERE id = ?", (rid,))

    # ------------------------------------------------------------------ items
    def upsert_item(self, rid: str, prompt_id: str, **fields: Any) -> dict[str, Any]:
        data = {k: _enc(v) for k, v in fields.items() if k in ITEM_FIELDS}
        if "detail" in data and data["detail"] is not None:
            data["detail"] = str(data["detail"])[:600]
        cols = ["run_id", "prompt_id", "ts", *data]
        updates = ", ".join(f"{k} = excluded.{k}" for k in data) + (", " if data else "") + "ts = excluded.ts"
        self.db.execute(f"INSERT INTO vb_items({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)}) ON CONFLICT(run_id, prompt_id) DO UPDATE SET {updates}",
                        [rid, prompt_id, self.clock(), *data.values()])
        return self.item(rid, prompt_id)

    def item(self, rid: str, prompt_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM vb_items WHERE run_id = ? AND prompt_id = ?", (rid, prompt_id))
        if row is None:
            raise GaltonError("not_found", "vb_no_item", id=rid, prompt=prompt_id)
        return _item(row)

    def items(self, rid: str) -> list[dict[str, Any]]:
        return [_item(r) for r in self.db.query("SELECT * FROM vb_items WHERE run_id = ? ORDER BY id", (rid,))]
