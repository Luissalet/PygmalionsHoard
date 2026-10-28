"""The briefs: the prompts of a public list of animations people asked a model to write as code (``data/prompts.json``).

Filter used to build the file: the creator published the whole prompt (not an excerpt) and it has more than 400 characters. ``standalone`` marks the briefs
that do not lean on a file, a link or an image the creator had and we do not (an attached video, a local path, a screenshot, generated images); those are the
ones a run picks by default. Each brief keeps its author and the post it came from: the list is MIT, the videos belong to their authors.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

DATA = Path(__file__).resolve().parent / "data" / "prompts.json"


def _load() -> dict[str, Any]:
    return json.loads(DATA.read_text(encoding="utf-8"))


_FILE = _load()
SOURCE = _FILE["source"]
FILTER = _FILE["filter"]
PROMPTS: list[dict[str, Any]] = _FILE["prompts"]
PROMPTS_BY_ID = {p["id"]: p for p in PROMPTS}
CATEGORIES = sorted({p["category"] for p in PROMPTS})
TAGS = sorted({t for p in PROMPTS for t in p["tech_tags"]})
EXCERPT = 220


def public(p: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    """What the tools and the screen show of a brief: the provenance always, the whole prompt only when asked."""
    prompt = p["prompt"]
    out = {"id": p["id"], "category": p["category"], "tech_tags": p["tech_tags"], "standalone": p["standalone"], "chars": p["chars"], "author": p["author"],
           "author_url": p["author_url"], "post_url": p["post_url"], "excerpt": " ".join(prompt.split())[:EXCERPT]}
    if full:
        out["prompt"] = prompt
    return out


def select(ids: Iterable[str] = (), *, category: str = "", tag: str = "", standalone: bool | None = True, count: int = 0, seed: int = 0, query: str = "") -> list[dict[str, Any]]:
    """The briefs a request names (``ids``) or a deterministic sample of the ones that match the filters: the same ``seed`` gives the same briefs."""
    ids = [i for i in ids if i]
    if ids:
        return [PROMPTS_BY_ID[i] for i in ids if i in PROMPTS_BY_ID]
    q = query.strip().lower()
    rows = [p for p in PROMPTS if (not category or p["category"] == category) and (not tag or tag in p["tech_tags"]) and (standalone is None or p["standalone"] == standalone)
            and (not q or q in p["prompt"].lower() or q in p["id"])]
    if count and count < len(rows):
        rows = sorted(rows, key=lambda p: hashlib.sha256(f"{seed}:{p['id']}".encode()).hexdigest())[:count]
    return rows
