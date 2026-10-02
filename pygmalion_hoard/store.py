"""Typed access to the database: datasets and their versions, artifacts (the lineage), jobs and runs."""

from __future__ import annotations

import time
from typing import Any, Callable, Iterable, Optional

from .db import Database
from .errors import PygmalionError
from .util import dumps, loads, new_id

ARTIFACT_KINDS = ("base", "adapter", "merged", "gguf", "imatrix", "ollama", "ctx_variant")
JOB_STATES = ("queued", "waiting_gpu", "running", "done", "failed", "cancelled", "interrupted")
ACTIVE_STATES = ("queued", "waiting_gpu", "running")


def _dataset(row: Any) -> Optional[dict[str, Any]]:
    return dict(row) if row is not None else None


def _version(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    d["splits"] = loads(d.get("splits"), {})
    d["recipe"] = loads(d.get("recipe"), {})
    d["stats"] = loads(d.get("stats"), {})
    return d


def _artifact(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    for key, default in (("parents", []), ("recipe", {}), ("metrics", {}), ("published", [])):
        d[key] = loads(d.get(key), default)
    d["pinned"] = bool(d.get("pinned"))
    return d


def _job(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    d = dict(row)
    d["params"] = loads(d.get("params"), {})
    d["then"] = loads(d.pop("then_steps", "[]"), [])
    d["progress"] = loads(d.get("progress"), {})
    d["result"] = loads(d.get("result"), {})
    d["vram"] = loads(d.get("vram"), {})
    d["gpus"] = loads(d.get("gpus"), [])
    d["cancel_requested"] = bool(d.get("cancel_requested"))
    return d


class Store:
    def __init__(self, db: Database, clock: Callable[[], float] = time.time):
        self.db = db
        self.clock = clock

    # ------------------------------------------------------------------ datasets
    def create_dataset(self, name: str, kind: str, description: str = "") -> dict[str, Any]:
        if self.db.one("SELECT 1 FROM datasets WHERE name = ?", (name,)):
            raise PygmalionError("conflict", "dataset_exists", name=name)
        did, now = new_id("ds", self.clock()), self.clock()
        self.db.execute("INSERT INTO datasets(id, name, description, kind, created_ts, updated_ts) VALUES (?, ?, ?, ?, ?, ?)",
                        (did, name, description, kind, now, now))
        return self.dataset(did)

    def dataset(self, ref: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM datasets WHERE id = ? OR name = ?", ((ref or "").strip(), (ref or "").strip()))
        if row is None:
            raise PygmalionError("not_found", "no_dataset", ref=ref)
        return _dataset(row)

    def datasets(self) -> list[dict[str, Any]]:
        return [_dataset(r) for r in self.db.query("SELECT * FROM datasets ORDER BY updated_ts DESC")]

    def update_dataset(self, did: str, **fields: Any) -> dict[str, Any]:
        allowed = {k: v for k, v in fields.items() if k in ("name", "description", "notes", "kind")}
        if allowed:
            sets = ", ".join(f"{k} = ?" for k in allowed)
            self.db.execute(f"UPDATE datasets SET {sets}, updated_ts = ? WHERE id = ?", [*allowed.values(), self.clock(), did])
        return self.dataset(did)

    def delete_dataset(self, did: str) -> None:
        self.db.execute("DELETE FROM datasets WHERE id = ?", (did,))

    def next_version_number(self, did: str) -> int:
        row = self.db.one("SELECT MAX(n) AS n FROM dataset_versions WHERE dataset_id = ?", (did,))
        return int(row["n"] or 0) + 1

    def add_version(self, did: str, n: int, path: str, sha256: str, *, kind: str, records: int, usable: int, chars: int, tokens: int,
                    splits: dict[str, Any], recipe: dict[str, Any], stats: dict[str, Any], parent: Optional[str] = None) -> dict[str, Any]:
        vid = new_id("dv", self.clock())
        self.db.execute(
            "INSERT INTO dataset_versions(id, dataset_id, n, path, sha256, kind, records, usable, chars, tokens, splits, recipe, stats, parent, created_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (vid, did, n, path, sha256, kind, records, usable, chars, tokens, dumps(splits), dumps(recipe), dumps(stats), parent, self.clock()))
        self.db.execute("UPDATE datasets SET updated_ts = ?, kind = ? WHERE id = ?", (self.clock(), kind, did))
        return self.version(vid)

    def version(self, ref: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM dataset_versions WHERE id = ?", ((ref or "").strip(),))
        if row is None:
            raise PygmalionError("not_found", "no_dataset_version", ref=ref)
        return _version(row)

    def find_version(self, dataset_ref: str, n: Optional[int] = None) -> dict[str, Any]:
        """A version by id, or the dataset's version ``n`` (latest when None). ``dataset_ref`` may be a dataset id/name or a version id."""
        ref = (dataset_ref or "").strip()
        row = self.db.one("SELECT * FROM dataset_versions WHERE id = ?", (ref,))
        if row is not None:
            return _version(row)
        ds = self.dataset(ref)
        if n is None:
            row = self.db.one("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY n DESC LIMIT 1", (ds["id"],))
        else:
            row = self.db.one("SELECT * FROM dataset_versions WHERE dataset_id = ? AND n = ?", (ds["id"], int(n)))
        if row is None:
            raise PygmalionError("not_found", "dataset_no_version" if n is not None else "dataset_no_versions", name=ds["name"], n=n)
        return _version(row)

    def versions(self, did: str) -> list[dict[str, Any]]:
        return [_version(r) for r in self.db.query("SELECT * FROM dataset_versions WHERE dataset_id = ? ORDER BY n", (did,))]

    # ------------------------------------------------------------------ artifacts
    def create_artifact(self, kind: str, name: str, *, path: str = "", size: int = 0, parents: Optional[list[str]] = None,
                        dataset_version: Optional[str] = None, job_id: Optional[str] = None, recipe: Optional[dict[str, Any]] = None,
                        metrics: Optional[dict[str, Any]] = None, notes: str = "", artifact_id: Optional[str] = None) -> dict[str, Any]:
        if kind not in ARTIFACT_KINDS:
            raise PygmalionError("invalid", "artifact_kind_unknown", kind=kind, options=list(ARTIFACT_KINDS))
        aid, now = artifact_id or new_id("a", self.clock()), self.clock()
        self.db.execute(
            "INSERT INTO artifacts(id, kind, name, path, size, parents, dataset_version, job_id, recipe, metrics, notes, created_ts, updated_ts) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (aid, kind, name, path, size, dumps(parents or []), dataset_version, job_id, dumps(recipe or {}), dumps(metrics or {}), notes, now, now))
        return self.artifact(aid)

    def artifact(self, ref: str) -> dict[str, Any]:
        ref = (ref or "").strip()
        row = self.db.one("SELECT * FROM artifacts WHERE id = ?", (ref,))
        if row is None:
            row = self.db.one("SELECT * FROM artifacts WHERE name = ? ORDER BY created_ts DESC LIMIT 1", (ref,))
        if row is None:
            raise PygmalionError("not_found", "no_artifact", ref=ref)
        return _artifact(row)

    def find_artifact_by_path(self, path: str) -> Optional[dict[str, Any]]:
        return _artifact(self.db.one("SELECT * FROM artifacts WHERE path = ? ORDER BY created_ts DESC LIMIT 1", (path,)))

    def find_artifact_by_name(self, kind: str, name: str) -> Optional[dict[str, Any]]:
        return _artifact(self.db.one("SELECT * FROM artifacts WHERE kind = ? AND name = ? ORDER BY created_ts DESC LIMIT 1", (kind, name)))

    def artifacts(self, *, kind: str = "", text: str = "", pinned: Optional[bool] = None, limit: int = 500) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM artifacts WHERE 1=1", []
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if text:
            sql += " AND (name LIKE ? OR notes LIKE ? OR id LIKE ?)"
            params.extend([f"%{text}%"] * 3)
        if pinned is not None:
            sql += " AND pinned = ?"
            params.append(int(pinned))
        sql += " ORDER BY created_ts DESC LIMIT ?"
        params.append(max(1, min(int(limit), 5000)))
        return [_artifact(r) for r in self.db.query(sql, params)]

    def update_artifact(self, aid: str, **fields: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in fields.items():
            if key in ("name", "notes", "path", "size", "pinned"):
                data[key] = int(bool(value)) if key == "pinned" else value
            elif key in ("metrics", "recipe", "published", "parents"):
                data[key] = dumps(value)
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE artifacts SET {sets}, updated_ts = ? WHERE id = ?", [*data.values(), self.clock(), aid])
        return self.artifact(aid)

    def merge_metrics(self, aid: str, key: str, value: Any) -> dict[str, Any]:
        """Set one key of an artifact's metrics. Read and write are one step under the database lock, so two jobs that add different
        measurements to the same artifact never overwrite each other."""
        with self.db.lock:
            metrics = dict(self.artifact(aid)["metrics"])
            metrics[key] = value
            return self.update_artifact(aid, metrics=metrics)

    def delete_artifact(self, aid: str) -> None:
        self.db.execute("DELETE FROM artifacts WHERE id = ?", (aid,))

    def children(self, aid: str) -> list[dict[str, Any]]:
        return [a for a in self.artifacts(limit=5000) if aid in a["parents"]]

    # ------------------------------------------------------------------ jobs
    def create_job(self, kind: str, lane: str, params: dict[str, Any], *, title: str = "", then: Optional[list[dict[str, Any]]] = None,
                   pipeline_id: Optional[str] = None, step: int = 0, out_artifact: Optional[str] = None, job_dir: str = "") -> dict[str, Any]:
        jid = new_id("j", self.clock())
        self.db.execute(
            "INSERT INTO jobs(id, kind, title, state, lane, params, then_steps, pipeline_id, step, created_ts, job_dir, out_artifact) "
            "VALUES (?, ?, ?, 'queued', ?, ?, ?, ?, ?, ?, ?, ?)",
            (jid, kind, title, lane, dumps(params), dumps(then or []), pipeline_id or jid, step, self.clock(), job_dir, out_artifact))
        return self.job(jid)

    def job(self, jid: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM jobs WHERE id = ?", ((jid or "").strip(),))
        if row is None:
            raise PygmalionError("not_found", "no_job", id=jid)
        return _job(row)

    def jobs(self, *, states: Optional[Iterable[str]] = None, kind: str = "", lane: str = "", pipeline_id: str = "", limit: int = 200,
             oldest_first: bool = False) -> list[dict[str, Any]]:
        sql, params = "SELECT * FROM jobs WHERE 1=1", []
        if states:
            states = list(states)
            sql += f" AND state IN ({', '.join('?' for _ in states)})"
            params.extend(states)
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if lane:
            sql += " AND lane = ?"
            params.append(lane)
        if pipeline_id:
            sql += " AND pipeline_id = ?"
            params.append(pipeline_id)
        sql += f" ORDER BY created_ts {'ASC' if oldest_first else 'DESC'}, id {'ASC' if oldest_first else 'DESC'} LIMIT ?"
        params.append(max(1, min(int(limit), 2000)))
        return [_job(r) for r in self.db.query(sql, params)]

    def update_job(self, jid: str, **fields: Any) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for key, value in fields.items():
            if key in ("params", "progress", "result", "vram", "gpus"):
                data[key] = dumps(value)
            elif key == "then":
                data["then_steps"] = dumps(value)
            elif key == "cancel_requested":
                data[key] = int(bool(value))
            elif key in ("state", "title", "started_ts", "finished_ts", "error", "hint", "queue_position", "attempts", "job_dir", "out_artifact", "lane"):
                data[key] = value
        if data:
            sets = ", ".join(f"{k} = ?" for k in data)
            self.db.execute(f"UPDATE jobs SET {sets} WHERE id = ?", [*data.values(), jid])
        return self.job(jid)

    def delete_job(self, jid: str) -> None:
        self.db.execute("DELETE FROM jobs WHERE id = ?", (jid,))

    def next_queued(self, lane: str) -> Optional[dict[str, Any]]:
        return _job(self.db.one("SELECT * FROM jobs WHERE state = 'queued' AND lane = ? ORDER BY created_ts ASC, id ASC LIMIT 1", (lane,)))

    def claim_job(self, job_id: str) -> bool:
        """Move a queued job to running in one statement; False when it is no longer queued (cancelled in the meantime)."""
        return self.db.execute("UPDATE jobs SET state = 'running' WHERE id = ? AND state = 'queued'", (job_id,)).rowcount == 1

    def mark_interrupted(self) -> int:
        """At start: whatever was running when the app stopped is interrupted (and can be resumed)."""
        return self.db.execute(
            "UPDATE jobs SET state = 'interrupted', finished_ts = ?, queue_position = NULL WHERE state IN ('running', 'waiting_gpu')",
            (self.clock(),)).rowcount

    # ------------------------------------------------------------------ runs and counts
    def add_run(self, kind: str, ref: str, ok: bool, duration_ms: int, detail: str) -> None:
        self.db.execute("INSERT INTO runs(ts, kind, ref, ok, duration_ms, detail) VALUES (?, ?, ?, ?, ?, ?)",
                        (self.clock(), kind, ref, int(ok), duration_ms, detail[:500]))
        self.db.execute("DELETE FROM runs WHERE id IN (SELECT id FROM runs ORDER BY id DESC LIMIT -1 OFFSET 2000)")

    def runs(self, kind: str = "", limit: int = 50) -> list[dict[str, Any]]:
        if kind:
            rows = self.db.query("SELECT * FROM runs WHERE kind = ? ORDER BY id DESC LIMIT ?", (kind, limit))
        else:
            rows = self.db.query("SELECT * FROM runs ORDER BY id DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def counts(self) -> dict[str, int]:
        one = lambda sql, p=(): int((self.db.one(sql, p) or [0])[0] or 0)  # noqa: E731
        return {
            "datasets": one("SELECT COUNT(*) FROM datasets"),
            "artifacts": one("SELECT COUNT(*) FROM artifacts"),
            "bases": one("SELECT COUNT(*) FROM artifacts WHERE kind = 'base'"),
            "jobs_active": one("SELECT COUNT(*) FROM jobs WHERE state IN ('queued', 'waiting_gpu', 'running')"),
            "jobs_running": one("SELECT COUNT(*) FROM jobs WHERE state = 'running'"),
            "jobs_failed": one("SELECT COUNT(*) FROM jobs WHERE state IN ('failed', 'interrupted')"),
            "jobs": one("SELECT COUNT(*) FROM jobs"),
        }
