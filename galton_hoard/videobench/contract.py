"""What the model is asked for, and how its reply is read.

The contract is the one of the HTML animations of the video editor: a single page that defines ``window.seek(t)`` (``t`` in seconds) and draws exactly the
frame at that time, so a renderer can take any frame in any order. ``window.DURATION`` tells how long it is.
"""
from __future__ import annotations

import re
from typing import Any

WIDTH, HEIGHT = 1280, 720
DEFAULT_DURATION = 10
MAX_DURATION = 30
MIN_HTML = 200

SYSTEM = f"""You write animations as code. Reply with one self-contained HTML document and nothing else (one ```html block).

The contract, so that the page can be rendered frame by frame:
- One file. No network: no CDN, no external script, stylesheet, font, image or video; everything inline. Use system fonts only. If the brief names a library, write by hand what you need from it.
- The stage is {WIDTH}x{HEIGHT} pixels (body margin 0, overflow hidden, the stage fills the viewport).
- Define window.DURATION: the length in seconds (the length the brief asks for, at most {MAX_DURATION}; {DEFAULT_DURATION} when it names none).
- Define window.seek(t): it draws exactly the frame at time t (seconds, from 0 to DURATION) and returns. It is deterministic: the same t gives the same picture whatever was drawn before and whatever the clock says. Do not read Date.now, performance.now or the real time to decide what is on screen; do not use unseeded Math.random (write a seeded generator).
- Optionally play it when the page opens: a requestAnimationFrame loop that calls seek(elapsed seconds), looping at DURATION. The renderer only needs seek.
- The animation is silent. Make everything the brief asks for that a silent {WIDTH}x{HEIGHT} animation can show; leave out what needs a file, an account or a device you do not have."""


def build_messages(brief: str) -> list[dict[str, Any]]:
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Here is the brief:\n\n" + brief.strip()}]


_FENCE = re.compile(r"```(?:html|HTML)?[ \t]*\n(.*?)```", re.S)
_DOC = re.compile(r"(<!doctype html.*?</html>|<html.*?</html>)", re.S | re.I)


def extract_html(text: str) -> str:
    """The HTML document in a reply: the biggest fenced block that looks like one, else the document itself, else an unfinished one (the model ran out of tokens)."""
    text = text or ""
    blocks = [b.strip() for b in _FENCE.findall(text)]
    blocks = [b for b in blocks if "<" in b and len(b) >= MIN_HTML]
    if blocks:
        return max(blocks, key=len)
    m = _DOC.search(text)
    if m:
        return m.group(1).strip()
    start = re.search(r"<!doctype html|<html", text, re.I)
    if start:                                           # a reply cut off before the closing tag is still a page the renderer can try
        return text[start.start():].replace("```", "").strip()
    return ""


_EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//|@import\s+(?:url\()?["']?\s*(?:https?:)?//|url\(\s*["']?\s*(?:https?:)?//|\b(?:fetch|XMLHttpRequest|WebSocket|importScripts)\s*\(|\bimport\s*\(\s*["']https?:""", re.I)


def static_check(html: str) -> dict[str, Any]:
    """What can be said about the page without running it."""
    return {"chars": len(html), "defines_seek": bool(re.search(r"window\s*\.\s*seek\s*=|window\s*\[\s*['\"]seek['\"]\s*\]\s*=|function\s+seek\s*\(", html)),
            "defines_duration": bool(re.search(r"DURATION\s*=", html)), "external_references": len(_EXTERNAL.findall(html)),
            "reads_clock": bool(re.search(r"Date\.now|performance\.now|new Date\(", html)), "unseeded_random": bool(re.search(r"Math\.random\s*\(", html))}
