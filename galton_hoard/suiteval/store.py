"""Suit evaluations and their records in Galton's database (tables ``se_evals`` and ``se_records``, migration 12)."""
from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from ..db import Database

RECORD_JSON = ("checks", "inputs")


def _loads(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return default if value is None else value
    try:
        return json.loads(value)
    except ValueError:
        return default


def _eval(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    d["definition"] = _loads(d.get("definition"), {})
    d["exported"] = _loads(d.get("exported"), {})
    return d


def _record(row: Any) -> dict[str, Any]:
    d = dict(row)
    for k in RECORD_JSON:
        d[k] = _loads(d.get(k), [] if k == "checks" else {})
    d["success"] = None if d["success"] is None else bool(d["success"])
    return d


class SuitEvalStore:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db, self.clock = db, clock

    def put_eval(self, eval_id: str, suit: str, name: str, definition: dict[str, Any]) -> dict[str, Any]:
        now = self.clock()
        with self.db.tx() as c:
            if c.execute("SELECT 1 FROM se_evals WHERE id=?", (eval_id,)).fetchone():
                c.execute("UPDATE se_evals SET suit=?, name=?, definition=?, updated_ts=? WHERE id=?",
                          (suit, name, json.dumps(definition, ensure_ascii=False), now, eval_id))
            else:
                c.execute("INSERT INTO se_evals(id, suit, name, definition, created_ts, updated_ts) VALUES (?,?,?,?,?,?)",
                          (eval_id, suit, name, json.dumps(definition, ensure_ascii=False), now, now))
        return self.eval(eval_id)

    def eval(self, eval_id: str) -> Optional[dict[str, Any]]:
        return _eval(self.db.one("SELECT * FROM se_evals WHERE id=?", (eval_id,)))

    def evals(self, suit: str = "") -> list[dict[str, Any]]:
        rows = self.db.query("SELECT * FROM se_evals WHERE suit=? ORDER BY updated_ts DESC" if suit else "SELECT * FROM se_evals ORDER BY updated_ts DESC",
                             (suit,) if suit else ())
        return [_eval(r) for r in rows]

    def set_exported(self, eval_id: str, info: dict[str, Any]) -> None:
        with self.db.tx() as c:
            c.execute("UPDATE se_evals SET exported=? WHERE id=?", (json.dumps(info, ensure_ascii=False), eval_id))

    def delete_eval(self, eval_id: str) -> None:
        with self.db.tx() as c:
            c.execute("DELETE FROM se_records WHERE eval_id=?", (eval_id,))
            c.execute("DELETE FROM se_evals WHERE id=?", (eval_id,))

    def record(self, eval_id: str, case_id: str, arm: str, run: int) -> Optional[dict[str, Any]]:
        row = self.db.one("SELECT * FROM se_records WHERE eval_id=? AND case_id=? AND arm=? AND run=?", (eval_id, case_id, arm, run))
        return _record(row) if row else None

    def records(self, eval_id: str) -> list[dict[str, Any]]:
        return [_record(r) for r in self.db.query("SELECT * FROM se_records WHERE eval_id=? ORDER BY case_id, arm, run", (eval_id,))]

    def count_for_case(self, eval_id: str, case_id: str) -> int:
        return self.db.one("SELECT COUNT(*) AS n FROM se_records WHERE eval_id=? AND case_id=?", (eval_id, case_id))["n"]

    def put_record(self, eval_id: str, case_id: str, arm: str, run: int, fields: dict[str, Any]) -> dict[str, Any]:
        now = self.clock()
        data = {k: (json.dumps(v, ensure_ascii=False, default=str) if k in RECORD_JSON else v) for k, v in fields.items()}
        if "success" in data and data["success"] is not None:
            data["success"] = int(bool(data["success"]))
        with self.db.tx() as c:
            if c.execute("SELECT 1 FROM se_records WHERE eval_id=? AND case_id=? AND arm=? AND run=?", (eval_id, case_id, arm, run)).fetchone():
                sets = {**data, "ts": now}
                c.execute(f"UPDATE se_records SET {', '.join(f'{k}=?' for k in sets)} WHERE eval_id=? AND case_id=? AND arm=? AND run=?",
                          (*sets.values(), eval_id, case_id, arm, run))
            else:
                cols = {"eval_id": eval_id, "case_id": case_id, "arm": arm, "run": run, **data, "ts": now}
                c.execute(f"INSERT INTO se_records({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})", tuple(cols.values()))
        return self.record(eval_id, case_id, arm, run)
