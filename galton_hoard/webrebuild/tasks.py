"""The sites of the website-rebuild bench: small local references that ship with the app, and any site given by URL.

A local task is a folder under ``sites/`` served on 127.0.0.1 while a run needs it, with its cases in ``sites/tasks.json``. A URL task is a
reference that already runs somewhere reachable (a site of the network, a reference started from an imported benchmark task); its cases are
given in the call, or one picture of each listed route is taken.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from ..errors import GaltonError
from ..hoard_link import webfidelity as wf

SITES = Path(__file__).resolve().parent / "sites"
WIDTHS = (1280, 390)
HEIGHTS = {1280: 800, 390: 844}
MAX_TASKS = 12


@lru_cache(maxsize=1)
def builtin() -> tuple[dict[str, Any], ...]:
    data = json.loads((SITES / "tasks.json").read_text(encoding="utf-8"))
    out = []
    for t in data["tasks"]:
        out.append({**t, "kind": "local", "folder": str(SITES / t["id"]), "cases": [wf.validate_case(c) for c in t["cases"]]})
    return tuple(out)


def by_id() -> dict[str, dict[str, Any]]:
    return {t["id"]: t for t in builtin()}


def height_for(width: int) -> int:
    return HEIGHTS.get(int(width), 844 if int(width) < 700 else 800)


def public(task: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    checkpoints = sum(1 for c in task["cases"] for s in c["steps"] if "checkpoint" in s)
    checks = sum(1 for c in task["cases"] for s in c["steps"] if any(k in s for k in wf.CHECKS))
    out = {"id": task["id"], "title": task.get("title") or task["id"], "summary": task.get("summary", ""), "level": task.get("level", ""), "kind": task["kind"],
           "cases": len(task["cases"]), "checkpoints": checkpoints, "checks": checks, "url": task.get("url", "")}
    if full:
        out["case_list"] = task["cases"]
    return out


def url_task(spec: dict[str, Any], index: int) -> dict[str, Any]:
    """A reference given by URL: ``{url, title?, routes?: ["/", "/a"], cases?: [...]}``."""
    url = str(spec.get("url") or "").strip().rstrip("/")
    if not re.match(r"^https?://[^\s/]+", url):
        raise GaltonError("invalid", "wr_bad_url", url=url[:200])
    cases = spec.get("cases")
    if cases:
        try:
            cases = [wf.validate_case(c) for c in cases]
        except ValueError as exc:
            raise GaltonError("invalid", "wr_bad_case", error=str(exc)) from exc
    else:
        routes = [str(r).strip() or "/" for r in (spec.get("routes") or ["/"])][:12]
        cases = [wf.validate_case({"id": f"ruta-{n}", "steps": [{"goto": r}, {"checkpoint": f"ruta-{n}"}]}) for n, r in enumerate(routes, 1)]
    return {"id": f"url-{index}", "title": str(spec.get("title") or url)[:120], "summary": url, "level": "", "kind": "url", "url": url, "cases": cases}


def resolve(ids: Optional[list[str]], urls: Optional[list[dict[str, Any]]] = None) -> list[dict[str, Any]]:
    known = by_id()
    chosen: list[dict[str, Any]] = []
    for i in [str(x).strip() for x in (ids or []) if str(x).strip()]:
        if i not in known:
            raise GaltonError("not_found", "wr_no_task", id=i)
        chosen.append(dict(known[i]))
    for n, spec in enumerate(urls or [], 1):
        chosen.append(url_task(spec, n))
    if not chosen:
        raise GaltonError("invalid", "wr_no_tasks")
    if len(chosen) > MAX_TASKS:
        raise GaltonError("invalid", "wr_too_many_tasks", n=len(chosen), max=MAX_TASKS)
    return chosen


def intro(task: dict[str, Any]) -> str:
    """The first message of the model: what to rebuild and where it starts."""
    routes = sorted({s["goto"] for c in task["cases"] for s in c["steps"] if "goto" in s})
    return (f"Reconstruye el sitio de referencia «{task.get('title') or task['id']}». Empieza por la ruta / con browse; "
            f"las rutas que se comprobarán incluyen: {', '.join(routes) or '/'}. Míralo a 1280 px y a 390 px, explora sus estados "
            "(pulsa sus botones, rellena sus formularios, sigue sus enlaces), escribe tus ficheros y termina con finish.")
