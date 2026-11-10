"""Case sets, cases, runs and per-case results of the RAG pipeline bench in the evaluation database (tables ``rb_*``, migration 13)."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

from ..db import Database
from ..util import new_id

ACTIVE = ("queued", "running")
TERMINAL = ("done", "failed", "cancelled")
CASE_STATES = ("proposed", "accepted", "rejected")
RUN_JSON = ("pipeline", "settings", "summary")
RESULT_JSON = ("trace", "metrics", "verdict")


def _loads(value: Any, default: Any) -> Any:
    if not isinstance(value, str):
        return default if value is None else value
    try:
        return json.loads(value)
    except ValueError:
        return default


def _dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _set(row: Any, *, corpus: bool = True) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    docs = _loads(d.get("corpus"), [])
    d["documents"] = [{"id": x.get("id"), "name": x.get("name"), "chars": len(x.get("text") or "")} for x in docs]
    if corpus:
        d["corpus"] = docs
    else:
        d.pop("corpus", None)
    return d


def _case(row: Any) -> dict[str, Any]:
    d = dict(row)
    d["passages"] = _loads(d.get("passages"), [])
    d["answerable"] = bool(d["answerable"])
    return d


def _run(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    for k in RUN_JSON:
        d[k] = _loads(d.get(k), {})
    d["cancel"] = bool(d["cancel"])
    return d


def _result(row: Any) -> dict[str, Any]:
    d = dict(row)
    for k in RESULT_JSON:
        d[k] = _loads(d.get(k), {})
    d["ok"] = None if d["ok"] is None else bool(d["ok"])
    return d


class RagStore:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db, self.clock = db, clock

    # ------------------------------------------------------------------ sets
    def create_set(self, name: str, description: str, corpus: list[dict[str, Any]]) -> dict[str, Any]:
        sid, now = new_id("rs", self.clock()), self.clock()
        with self.db.tx() as c:
            c.execute("INSERT INTO rb_sets(id, name, description, corpus, created_ts, updated_ts) VALUES (?,?,?,?,?,?)", (sid, name, description, _dumps(corpus), now, now))
        return self.set(sid)

    def update_set(self, sid: str, **fields: Any) -> dict[str, Any]:
        data = {k: (_dumps(v) if k == "corpus" else v) for k, v in fields.items()}
        data["updated_ts"] = self.clock()
        with self.db.tx() as c:
            c.execute(f"UPDATE rb_sets SET {', '.join(f'{k}=?' for k in data)} WHERE id=?", (*data.values(), sid))
        return self.set(sid)

    def set(self, sid: str, *, corpus: bool = True) -> Optional[dict[str, Any]]:
        return _set(self.db.one("SELECT * FROM rb_sets WHERE id=?", (sid,)), corpus=corpus)

    def find_set(self, ref: str) -> Optional[dict[str, Any]]:
        found = self.set(ref)
        if found is None:
            row = self.db.one("SELECT * FROM rb_sets WHERE name=? COLLATE NOCASE ORDER BY updated_ts DESC LIMIT 1", (ref,))
            found = _set(row)
        return found

    def sets(self) -> list[dict[str, Any]]:
        return [_set(r, corpus=False) for r in self.db.query("SELECT * FROM rb_sets ORDER BY updated_ts DESC")]

    def delete_set(self, sid: str) -> None:
        with self.db.tx() as c:
            c.execute("DELETE FROM rb_results WHERE run_id IN (SELECT id FROM rb_runs WHERE set_id=?)", (sid,))
            c.execute("DELETE FROM rb_runs WHERE set_id=?", (sid,))
            c.execute("DELETE FROM rb_cases WHERE set_id=?", (sid,))
            c.execute("DELETE FROM rb_sets WHERE id=?", (sid,))

    # ------------------------------------------------------------------ cases
    def add_case(self, sid: str, *, question: str, expected: str = "", passages: Optional[list[dict[str, Any]]] = None, answerable: bool = True, require: str = "all",
                 status: str = "accepted", origin: str = "manual", notes: str = "") -> dict[str, Any]:
        cid, now = new_id("rk", self.clock()), self.clock()
        with self.db.tx() as c:
            position = c.execute("SELECT COALESCE(MAX(position), 0) + 1 AS n FROM rb_cases WHERE set_id=?", (sid,)).fetchone()["n"]
            c.execute("INSERT INTO rb_cases(id, set_id, position, question, expected, passages, answerable, require, status, origin, notes, created_ts, updated_ts) "
                      "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", (cid, sid, position, question, expected, _dumps(passages or []), int(answerable), require, status, origin, notes, now, now))
            c.execute("UPDATE rb_sets SET updated_ts=? WHERE id=?", (now, sid))
        return self.case(cid)

    def update_case(self, cid: str, **fields: Any) -> dict[str, Any]:
        data = {k: (_dumps(v) if k == "passages" else int(v) if k == "answerable" else v) for k, v in fields.items()}
        data["updated_ts"] = self.clock()
        with self.db.tx() as c:
            c.execute(f"UPDATE rb_cases SET {', '.join(f'{k}=?' for k in data)} WHERE id=?", (*data.values(), cid))
        return self.case(cid)

    def case(self, cid: str) -> Optional[dict[str, Any]]:
        row = self.db.one("SELECT * FROM rb_cases WHERE id=?", (cid,))
        return _case(row) if row else None

    def cases(self, sid: str, status: str = "") -> list[dict[str, Any]]:
        if status:
            rows = self.db.query("SELECT * FROM rb_cases WHERE set_id=? AND status=? ORDER BY position, id", (sid, status))
        else:
            rows = self.db.query("SELECT * FROM rb_cases WHERE set_id=? ORDER BY position, id", (sid,))
        return [_case(r) for r in rows]

    def case_counts(self, sid: str) -> dict[str, int]:
        rows = self.db.query("SELECT status, COUNT(*) AS n FROM rb_cases WHERE set_id=? GROUP BY status", (sid,))
        counts = {s: 0 for s in CASE_STATES}
        counts.update({r["status"]: r["n"] for r in rows})
        return counts

    def delete_case(self, cid: str) -> None:
        with self.db.tx() as c:
            c.execute("DELETE FROM rb_cases WHERE id=?", (cid,))

    # ------------------------------------------------------------------ runs
    def create_run(self, *, set_id: str, set_name: str, label: str, pipeline: dict[str, Any], settings: dict[str, Any], total: int, source: str = "", caller: str = "") -> dict[str, Any]:
        rid, now = new_id("rr", self.clock()), self.clock()
        with self.db.tx() as c:
            c.execute("INSERT INTO rb_runs(id, set_id, set_name, label, state, pipeline, settings, total, source, caller, created_ts) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (rid, set_id, set_name, label, "queued", _dumps(pipeline), _dumps(settings), total, source, caller, now))
        return self.run(rid)

    def update_run(self, rid: str, **fields: Any) -> dict[str, Any]:
        data = {k: (_dumps(v) if k in RUN_JSON else int(v) if k == "cancel" else str(v) if k == "error" else v) for k, v in fields.items()}
        with self.db.tx() as c:
            c.execute(f"UPDATE rb_runs SET {', '.join(f'{k}=?' for k in data)} WHERE id=?", (*data.values(), rid))
        return self.run(rid)

    def run(self, rid: str) -> Optional[dict[str, Any]]:
        return _run(self.db.one("SELECT * FROM rb_runs WHERE id=?", (rid,)))

    def runs(self, *, set_id: str = "", states: tuple[str, ...] = (), limit: int = 50) -> list[dict[str, Any]]:
        where, args = [], []
        if set_id:
            where.append("set_id=?")
            args.append(set_id)
        if states:
            where.append(f"state IN ({', '.join('?' * len(states))})")
            args.extend(states)
        sql = "SELECT * FROM rb_runs" + (" WHERE " + " AND ".join(where) if where else "") + " ORDER BY created_ts DESC LIMIT ?"
        return [_run(r) for r in self.db.query(sql, (*args, limit))]

    def delete_run(self, rid: str) -> None:
        with self.db.tx() as c:
            c.execute("DELETE FROM rb_results WHERE run_id=?", (rid,))
            c.execute("DELETE FROM rb_runs WHERE id=?", (rid,))

    # ------------------------------------------------------------------ results
    def put_result(self, rid: str, case_id: str, *, position: int, question: str, trace: dict[str, Any], metrics: dict[str, Any], verdict: dict[str, Any],
                   ok: Optional[bool], piece: str, error: str = "") -> None:
        with self.db.tx() as c:
            c.execute("DELETE FROM rb_results WHERE run_id=? AND case_id=?", (rid, case_id))
            c.execute("INSERT INTO rb_results(run_id, case_id, position, question, trace, metrics, verdict, ok, piece, error, ts) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                      (rid, case_id, position, question, _dumps(trace), _dumps(metrics), _dumps(verdict), None if ok is None else int(ok), piece, error, self.clock()))

    def results(self, rid: str) -> list[dict[str, Any]]:
        return [_result(r) for r in self.db.query("SELECT * FROM rb_results WHERE run_id=? ORDER BY position, id", (rid,))]

    def result(self, rid: str, case_id: str) -> Optional[dict[str, Any]]:
        row = self.db.one("SELECT * FROM rb_results WHERE run_id=? AND case_id=?", (rid, case_id))
        return _result(row) if row else None
