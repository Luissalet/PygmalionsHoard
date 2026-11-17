"""The public part of WebsiteBench (Apache-2.0), kept in Galton's data folder: its release index, licence, notice and, per task, the
instruction, the task file and the case manifest (how many hidden cases of each kind and tier the task has).

The hidden cases and the reference sites run in the benchmark's own containers: a full run goes through the agent bench (``websitebench``
in its catalogue, Docker and the trial runner). What is imported here lets Galton list the 70 tasks with their size and difficulty and
read what each one asks, without running anything.
"""
from __future__ import annotations

import json
import re
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Optional

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    tomllib = None  # type: ignore[assignment]

from ..errors import GaltonError
from ..hoard_link.atomic import write_bytes_atomic, write_json_atomic

REPO = "https://github.com/TIGER-AI-Lab/WebsiteBench"
RAW = "https://raw.githubusercontent.com/TIGER-AI-Lab/WebsiteBench/{rev}/{path}"
INDEX = "src/websitebench/websitebench-v1.json"
TASK_FILES = ("task.toml", "instruction.md", "tests/fixtures/case-manifest.json")
TASK_ID = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,80}$")
CREDIT = ("WebsiteBench, by the WebsiteBench Authors (TIGER-AI-Lab), " + REPO + ". Task definitions and verifier under the Apache License 2.0; "
          "the reference websites reproduce third-party sites and their content belongs to its owners (non-commercial research only).")

Fetch = Callable[[str], bytes]


def http_fetch(url: str, timeout: float = 30.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "galton-hoard"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:   # noqa: S310 - fixed https host
        return resp.read()


def folder(data_root: Path) -> Path:
    return Path(data_root) / "websitebench"


def _case_summary(raw: bytes) -> dict[str, Any]:
    data = json.loads(raw.decode("utf-8"))
    cases = data.get("cases") or []
    return {"cases": len(cases), "by_kind": dict(Counter(str(c.get("kind") or "?") for c in cases)),
            "by_tier": dict(Counter(str(c.get("tier") or "?") for c in cases)), "status": data.get("status", ""), "profile": data.get("suite_profile", "")}


def _task_meta(raw: bytes) -> dict[str, Any]:
    if tomllib is None:
        return {}
    data = tomllib.loads(raw.decode("utf-8"))
    task, meta, agent = data.get("task") or {}, data.get("metadata") or {}, data.get("agent") or {}
    env = data.get("environment") or {}
    return {"description": task.get("description", ""), "keywords": task.get("keywords", []), "difficulty": meta.get("difficulty", ""),
            "task_type": meta.get("task_type", ""), "agent_timeout_s": agent.get("timeout_sec"), "cpus": env.get("cpus"), "memory_mb": env.get("memory_mb")}


def import_public(data_root: Path, *, tasks: Optional[list[str]] = None, every: bool = False, rev: str = "main", fetch: Fetch = http_fetch) -> dict[str, Any]:
    """Download the release index, licence and notice, and the public files of the tasks asked for (``every`` for all of them)."""
    dest = folder(data_root)
    dest.mkdir(parents=True, exist_ok=True)
    try:
        index_raw = fetch(RAW.format(rev=rev, path=INDEX))
        index = json.loads(index_raw.decode("utf-8"))
        licence, notice = fetch(RAW.format(rev=rev, path="LICENSE")), fetch(RAW.format(rev=rev, path="NOTICE"))
    except (OSError, ValueError) as exc:
        raise GaltonError("unavailable", "wr_wb_fetch", error=str(exc)[:200]) from exc
    if index.get("schema") != "websitebench.release-manifest.v1":
        raise GaltonError("invalid", "wr_wb_schema", schema=str(index.get("schema"))[:80])
    write_bytes_atomic(dest / "websitebench-v1.json", index_raw)
    write_bytes_atomic(dest / "LICENSE", licence)
    write_bytes_atomic(dest / "NOTICE", notice)
    (dest / "CREDITS.txt").write_text(CREDIT + "\n", encoding="utf-8")
    known = sorted((index.get("tasks") or {}).keys())
    wanted = known if every else [t for t in (tasks or [])]
    unknown = [t for t in wanted if t not in known or not TASK_ID.match(t)]
    if unknown:
        raise GaltonError("not_found", "wr_wb_no_task", id=", ".join(unknown[:5]))
    imported, failed = [], []
    for tid in wanted:
        tdir = dest / "tasks" / tid
        try:
            files = {name: fetch(RAW.format(rev=rev, path=f"tasks/{tid}/{name}")) for name in TASK_FILES}
        except OSError as exc:
            failed.append({"task": tid, "error": str(exc)[:200]})
            continue
        for name, raw in files.items():
            write_bytes_atomic(tdir / name, raw)
        try:
            summary = {**_task_meta(files["task.toml"]), **_case_summary(files["tests/fixtures/case-manifest.json"])}
        except (ValueError, UnicodeDecodeError) as exc:
            failed.append({"task": tid, "error": f"unreadable task files: {exc}"[:200]})
            continue
        write_json_atomic(tdir / "summary.json", summary)
        imported.append(tid)
    return {"folder": str(dest), "version": index.get("version"), "task_count": index.get("task_count"), "imported": imported, "failed": failed,
            "credit": CREDIT}


def listing(data_root: Path, *, query: str = "") -> dict[str, Any]:
    """What is in the data folder: every task of the index, with its summary when its files were imported."""
    dest = folder(data_root)
    idx = dest / "websitebench-v1.json"
    if not idx.is_file():
        return {"imported": False, "folder": str(dest), "tasks": [], "repository": REPO, "credit": CREDIT}
    index = json.loads(idx.read_text(encoding="utf-8"))
    q = query.strip().lower()
    rows = []
    for tid, entry in sorted((index.get("tasks") or {}).items()):
        if q and q not in tid:
            continue
        summary_file = dest / "tasks" / tid / "summary.json"
        summary = json.loads(summary_file.read_text(encoding="utf-8")) if summary_file.is_file() else None
        rows.append({"id": tid, "files": entry.get("file_count"), "details": summary})
    return {"imported": True, "folder": str(dest), "version": index.get("version"), "task_count": index.get("task_count"),
            "hf_dataset": index.get("hf_dataset"), "hub_dataset": index.get("hub_dataset"), "tasks": rows, "repository": REPO, "credit": CREDIT,
            "with_details": sum(1 for r in rows if r["details"])}


def instruction(data_root: Path, tid: str) -> dict[str, Any]:
    if not TASK_ID.match(tid or ""):
        raise GaltonError("not_found", "wr_wb_no_task", id=str(tid)[:80])
    tdir = folder(data_root) / "tasks" / tid
    path = tdir / "instruction.md"
    if not path.is_file():
        raise GaltonError("not_found", "wr_wb_not_imported", id=tid)
    text = re.sub(r"^<!--.*?-->\s*", "", path.read_text(encoding="utf-8"), flags=re.S)
    summary = json.loads((tdir / "summary.json").read_text(encoding="utf-8")) if (tdir / "summary.json").is_file() else {}
    return {"id": tid, "instruction": text, "details": summary, "credit": CREDIT}
