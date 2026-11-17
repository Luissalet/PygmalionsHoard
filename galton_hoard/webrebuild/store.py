"""Runs of the website-rebuild bench, one JSON file per run (``<root>/<run>/run.json``) with its items inside, written atomically."""
from __future__ import annotations

import json
import shutil
import threading
import uuid
from pathlib import Path
from typing import Any, Callable, Optional

from ..errors import GaltonError
from ..hoard_link.atomic import write_json_atomic

ACTIVE = ("queued", "running")
TERMINAL = ("done", "failed", "cancelled")


class RebuildStore:
    def __init__(self, root: Callable[[], Path], clock: Callable[[], float]):
        self._root, self.clock = root, clock
        self.lock = threading.RLock()

    @property
    def root(self) -> Path:
        return Path(self._root())

    def _file(self, rid: str) -> Path:
        if not rid or not rid.replace("-", "").isalnum():
            raise GaltonError("not_found", "wr_no_run", id=str(rid)[:40])
        return self.root / rid / "run.json"

    def _write(self, run: dict[str, Any]) -> None:
        path = self._file(run["id"])
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(path, run)

    def create_run(self, **fields: Any) -> dict[str, Any]:
        rid = "wr-" + uuid.uuid4().hex[:10]
        now = self.clock()
        run = {"id": rid, "created_ts": now, "started_ts": None, "finished_ts": None, "state": "queued", "cancel": False, "error": "", "summary": None,
               "items": {}, **fields}
        with self.lock:
            self._write(run)
        return run

    def run(self, rid: str) -> dict[str, Any]:
        path = self._file(rid)
        with self.lock:
            if not path.is_file():
                raise GaltonError("not_found", "wr_no_run", id=rid)
            return json.loads(path.read_text(encoding="utf-8"))

    def runs(self, *, limit: int = 50, states: Optional[tuple[str, ...]] = None) -> list[dict[str, Any]]:
        out = []
        if not self.root.is_dir():
            return out
        for folder in self.root.iterdir():
            f = folder / "run.json"
            if not f.is_file():
                continue
            try:
                run = json.loads(f.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                continue
            if states and run.get("state") not in states:
                continue
            out.append(run)
        out.sort(key=lambda r: r.get("created_ts") or 0, reverse=True)
        return out[:limit]

    def update_run(self, rid: str, **fields: Any) -> dict[str, Any]:
        with self.lock:
            run = self.run(rid)
            run.update(fields)
            self._write(run)
            return run

    @staticmethod
    def key(task: str, slot: str) -> str:
        return f"{task}/{slot}"

    def upsert_item(self, rid: str, task: str, slot: str, **fields: Any) -> dict[str, Any]:
        with self.lock:
            run = self.run(rid)
            k = self.key(task, slot)
            item = run["items"].get(k) or {"task": task, "slot": slot, "state": "queued"}
            item.update(fields)
            run["items"][k] = item
            self._write(run)
            return item

    def item(self, rid: str, task: str, slot: str) -> dict[str, Any]:
        run = self.run(rid)
        item = run["items"].get(self.key(task, slot))
        if item is None:
            raise GaltonError("not_found", "wr_no_item", task=task, slot=slot)
        return item

    def items(self, rid: str) -> list[dict[str, Any]]:
        return list(self.run(rid)["items"].values())

    def delete(self, rid: str) -> None:
        path = self._file(rid)
        with self.lock:
            if not path.is_file():
                raise GaltonError("not_found", "wr_no_run", id=rid)
            shutil.rmtree(path.parent, ignore_errors=True)
