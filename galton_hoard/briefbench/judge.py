"""The judge: a session that did not write the answer scores it on four dimensions, without knowing which model wrote it.

The scale is the one of the video bank's jury (5 competent, 6 decent, 7 solid professional, 8 high-end, 9 shortlist, 10 the best there is; most first answers deserve
5 to 7). The judge is a model: its scores are a judgement, not a measurement, and a model that judges its own answer tends to be kinder to it: the run says which model
judged and whether it looked at the pictures or only at the automatic facts and the text.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

DIMENSIONS = ("fidelity", "craft", "polish", "technical")
RUBRIC = {
    "fidelity": "it does everything the brief asks, in the form it asks for, and nothing that contradicts it",
    "craft": "the quality of the idea and its execution: composition and design for a page, reasoning and structure for a text",
    "polish": "finish and detail: no rough edges, placeholders, clutter, filler or generic choices",
    "technical": "it works: a page renders, runs without errors and moves where it should; a text is correct, consistent and checkable",
}

SYSTEM = """You are a demanding reviewer. You did NOT write the answer you are about to judge, and you do not know which model did. Judge it strictly and precisely: kindness is useless, specificity is everything.

You get the brief, the facts measured automatically (do not argue with them) and either the pictures of the result or the result itself. Score each dimension from 1 to 10: 5 = competent student work; 6 = decent; 7 = solid professional; 8 = high-end; 9 = shortlist for an award; 10 = the best there is. Most first answers deserve 5 to 7. Do not inflate. A blank, broken or empty answer scores 1 to 3 on every dimension it fails.
Dimensions:
""" + "\n".join(f"- {d}: {RUBRIC[d]}" for d in DIMENSIONS) + """

Reply with one JSON object and nothing else:
{"scores": {"fidelity": n, "craft": n, "polish": n, "technical": n}, "summary": "two sentences", "problems": ["the most important faults, concrete and observable"]}"""


def digest(auto: dict[str, Any], render: dict[str, Any], static: dict[str, Any]) -> str:
    if not render and not auto:
        return "Text answer: no automatic checks beyond its length."
    return (f"Rendered page. Blank: {auto.get('blank')}. Moves by itself between two pictures {render.get('motion_ms', '?')} ms apart: {auto.get('moves')} (change {auto.get('motion')}). "
            f"Page errors: {len(render.get('page_errors') or [])}; console errors: {len(render.get('console_errors') or [])}; blocked network requests: {len(render.get('blocked_requests') or [])}. "
            f"Canvases: {render.get('canvases')}; SVGs: {render.get('svgs')}; visible text characters: {render.get('text_chars')}; page size {render.get('page_width')}x{render.get('page_height')}. "
            f"HTML size: {static.get('chars')} characters." + (" First errors: " + " | ".join((render.get("page_errors") or [])[:3]) if render.get("page_errors") else ""))


def judge_messages(brief: dict[str, Any], answer: str, auto: dict[str, Any], render: dict[str, Any], static: dict[str, Any], *, images: bool) -> list[dict[str, Any]]:
    user = "BRIEF:\n" + brief["prompt"].strip()[:6000]
    if brief.get("criteria"):
        user += "\n\nEXTRA CRITERIA from the person who asked (weigh them in fidelity):\n" + brief["criteria"][:1500]
    user += "\n\nAUTOMATIC FACTS:\n" + digest(auto, render, static)
    if brief["kind"] == "text":
        user += "\n\nANSWER:\n" + answer[:12000]
    elif images:
        user += "\n\nThe pictures follow: the page after it settled, then the same page a moment later."
    else:
        user += "\n\nYou cannot see the page. Judge from the facts and the source below, and say so in the summary; be careful with what you cannot know.\n\nPAGE SOURCE (cut):\n" + answer[:12000]
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_JSON = re.compile(r"\{.*\}", re.S)


def parse_judge(text: str) -> Optional[dict[str, Any]]:
    """The scores of a judge reply, or None when it holds none. Scores are clamped to 1..10; a missing dimension stays out."""
    text = (text or "").strip()
    for candidate in (text, *(m.group(0) for m in [_JSON.search(text)] if m)):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("scores"), dict):
            continue
        scores: dict[str, float] = {}
        for d in DIMENSIONS:
            v = obj["scores"].get(d)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            scores[d] = float(min(10, max(1, v)))
        if not scores:
            continue
        return {"scores": scores, "mean": round(sum(scores.values()) / len(scores), 2), "dimensions_scored": len(scores), "summary": str(obj.get("summary", ""))[:600],
                "problems": [str(c)[:300] for c in (obj.get("problems") or [])][:8]}
    return None
