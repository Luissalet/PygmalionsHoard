"""The jury: a second session, which did not write the page, scores the render on seven dimensions.

The rubric is the one of an open collection of motion-design styles (MIT, see ``provenance/videobench``): seven dimensions scored 1 to 10 with a stated scale
(5 competent student work, 6 decent template, 7 solid professional, 8 high-end agency, 9 award shortlist, 10 best in the world; most first renders deserve 5 to 7),
a list of fixes ordered by impact and a list of critical issues. Two changes for a silent HTML animation: ``style_fidelity`` is fidelity to the brief and the style
it names, and ``sound_sync`` is not scored (the contract has no sound), so it is left out of the mean.

The jury is a model: its scores are a judgement, not a measurement, and a model that scores a page it wrote itself tends to be kinder to it. The run says which
model judged and whether it looked at the images or only at the automatic checks and the text of the page.
"""
from __future__ import annotations

import json
import re
from typing import Any, Optional

DIMENSIONS = ("style_fidelity", "concept_wow", "motion_craft", "design_typography", "finish_texture", "sound_sync", "technical")
SCORED = tuple(d for d in DIMENSIONS if d != "sound_sync")
RUBRIC = {
    "style_fidelity": "fidelity to the brief: it shows what was asked, in the style it names, with every feature the brief lists",
    "concept_wow": "a clear idea, a hook in the first half second, an unmistakable hero moment, a satisfying last frame",
    "motion_craft": "easing, spacing, overlap, anticipation and follow-through, arcs, transitions; no dead or jittery motion",
    "design_typography": "composition, hierarchy, grid, type setting, palette, negative space",
    "finish_texture": "lighting, texture, glow and detail that suit the style; no banding, aliasing or artefacts",
    "sound_sync": "not scored: the animation is silent by contract",
    "technical": "the page does what the contract says: it renders, seek is deterministic, nothing is clipped, no glitches, no fallback-font tofu",
}

SYSTEM = """You are a senior motion-design jury member (think of an editor of a motion-design journal or a creative director at a top studio). You did NOT make this animation. Judge it harshly and precisely so it can reach award-level quality. Kindness is useless; specificity is everything.

You get the brief it was made from, a contact sheet of 12 frames taken at even times (each labelled with its time) and two larger frames when images are available, the text visible on the page at each time, and automatic checks that are facts (do not argue with them). The animation is a silent HTML page of 1280x720.

Score each dimension from 1 to 10: 5 = competent student work; 6 = decent template level; 7 = solid professional; 8 = high-end agency work; 9 = award shortlist; 10 = best in the world. Most first renders deserve 5 to 7. Do not inflate. A page that is blank, static or broken scores 1 to 3 on every dimension it fails.
Dimensions:
""" + "\n".join(f"- {d}: {RUBRIC[d]}" for d in SCORED) + """

Then list 3 to 6 fixes ordered by impact: the time, what is wrong (observable), why it matters and a concrete way to fix it. At least one must raise the wow factor. List as critical anything that reads as generic or template, fallback fonts or tofu, and any failure of the contract.

Reply with one JSON object and nothing else:
{"scores": {"style_fidelity": n, "concept_wow": n, "motion_craft": n, "design_typography": n, "finish_texture": n, "technical": n}, "summary": "two sentences", "critical": ["..."], "fixes": [{"t": "2.5s", "problem": "...", "fix": "..."}]}"""


def digest(auto: dict[str, Any], render: dict[str, Any]) -> str:
    """The facts the jury gets next to the pictures (and instead of them for a jury that cannot see)."""
    times, texts, lum, motion = render.get("times") or [], render.get("texts") or [], auto.get("luminance") or [], auto.get("motion") or []
    lines = [f"Length {render.get('duration')} s; {len(times)} frames. The page defines window.seek: {render.get('has_seek')}. seek is deterministic (same time, same picture): {render.get('deterministic')}.",
             f"Blank frames: {auto.get('blank_frames')}. Mean change between consecutive frames (0 to 1): {auto.get('motion_mean')}. Still steps: {auto.get('still_steps')}. "
             f"Change from first to last frame: {auto.get('first_last_change')} (a loop ends where it began); largest change of any frame from the first: {auto.get('max_change_from_first')}. Page errors: {len(render.get('page_errors') or [])}; blocked network requests: {len(render.get('blocked_requests') or [])}."]
    if render.get("page_errors"):
        lines.append("Errors: " + " | ".join(render["page_errors"][:3]))
    for i, t in enumerate(times):
        change = f" change from the previous frame {motion[i - 1]}" if i and i - 1 < len(motion) else ""
        lines.append(f"t={t}s luminance {lum[i] if i < len(lum) else '?'}{change}; visible text: {texts[i] if i < len(texts) and texts[i] else '(none)'}")
    return "\n".join(lines)


def jury_messages(brief: str, auto: dict[str, Any], render: dict[str, Any], *, images: bool, html: str = "") -> list[dict[str, Any]]:
    user = "BRIEF:\n" + brief.strip()[:6000] + "\n\nAUTOMATIC CHECKS (facts):\n" + digest(auto, render)
    user += "\n\nThe pictures follow: the contact sheet first, then two larger frames." if images else (
        "\n\nYou cannot see the frames. Judge from the checks and the text above and from the page source below, and say so in the summary; be careful with the dimensions you cannot know.\n\n"
        "PAGE SOURCE (cut):\n" + html[:12000])
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


_JSON = re.compile(r"\{.*\}", re.S)


def parse_jury(text: str) -> Optional[dict[str, Any]]:
    """The scores of a jury reply, or None when it holds none. Scores are clamped to 1..10; a missing dimension stays out."""
    text = (text or "").strip()
    for candidate in (text, *(m.group(0) for m in [_JSON.search(text)] if m)):
        try:
            obj = json.loads(candidate)
        except ValueError:
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("scores"), dict):
            continue
        scores: dict[str, float] = {}
        for d in SCORED:
            v = obj["scores"].get(d)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            scores[d] = float(min(10, max(1, v)))
        if not scores:
            continue
        fixes = [{"t": str(f.get("t", ""))[:40], "problem": str(f.get("problem", ""))[:400], "fix": str(f.get("fix", ""))[:400]} for f in (obj.get("fixes") or []) if isinstance(f, dict)][:8]
        return {"scores": scores, "mean": round(sum(scores.values()) / len(scores), 2), "dimensions_scored": len(scores), "summary": str(obj.get("summary", ""))[:600],
                "critical": [str(c)[:300] for c in (obj.get("critical") or [])][:8], "fixes": fixes}
    return None
