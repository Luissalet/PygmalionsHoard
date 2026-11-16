"""What each model is asked for and how its reply is read.

A ``page`` brief is answered with one self-contained HTML document: it is rendered in a browser that blocks the network, so the page may not reach for a CDN. A library the
person keeps in the ``briefbench.libraries`` folder can be used with ``<script src="lib/<file>"></script>``: the file is put into the page when it is rendered or played, so
the page the model wrote stays exactly as written. A ``text`` brief is answered as it comes.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional

from ..videobench.contract import extract_html

WIDTH, HEIGHT = 1280, 720

SYSTEM_PAGE = f"""You build a web page from a brief. Reply with one self-contained HTML document and nothing else (one ```html block).

- One file; plain HTML, CSS and JavaScript. No network: no CDN, no external script, stylesheet, font, image or video. Use system fonts and draw with CSS, SVG or canvas.
- The page is shown at {WIDTH}x{HEIGHT} pixels. Everything the brief asks for must be visible and working without any interaction first: whatever should move, moves by itself as soon as the page opens, and the first screen already looks finished.
- Make the best, most complete version of what the brief asks for. Do not explain, apologise or leave placeholders.@@LIBRARIES@@"""

SYSTEM_TEXT = ("You answer a brief. Give the answer itself, complete and correct, in the language of the brief, and nothing before or after it that is not part of the answer. "
               "Respect any length limit the brief states.")


def library_file(folder: str, name: str) -> Optional[str]:
    """The file in the libraries folder that holds the library ``name`` (``<name>.min.js`` first, then ``<name>.js``), or None."""
    if not folder or not re.fullmatch(r"[A-Za-z0-9_.-]{1,60}", name or ""):
        return None
    base = Path(folder)
    for candidate in (f"{name}.min.js", f"{name}.js"):
        if (base / candidate).is_file():
            return candidate
    return None


def build_messages(brief: dict[str, Any], libraries: dict[str, str]) -> list[dict[str, Any]]:
    """``libraries`` maps the library names the brief named to the file that holds them (only the ones that exist)."""
    if brief["kind"] == "text":
        return [{"role": "system", "content": SYSTEM_TEXT}, {"role": "user", "content": brief["prompt"].strip()}]
    lines = ""
    if libraries:
        lines = "\n- These libraries are available offline; load each one with exactly this tag and use it as usual:\n" + "\n".join(
            f'  <script src="lib/{file}"></script>   ({name})' for name, file in libraries.items())
    missing = [n for n in (brief.get("libraries") or []) if n not in libraries]
    if missing:
        lines += "\n- " + ", ".join(missing) + " is not available: write by hand what you need (canvas, SVG, CSS or WebGL)."
    return [{"role": "system", "content": SYSTEM_PAGE.replace("@@LIBRARIES@@", lines)}, {"role": "user", "content": "Here is the brief:\n\n" + brief["prompt"].strip()}]


_LIB_TAG = re.compile(r"""<script\b[^>]*\bsrc\s*=\s*["'](?:\./)?lib/([A-Za-z0-9_.-]+\.js)["'][^>]*>\s*</script\s*>""", re.I)


def inline_libraries(html: str, folder: str) -> str:
    """The page with each ``<script src="lib/x.js">`` replaced by the file's code (only files that exist in the libraries folder); everything else as it is."""
    if not folder:
        return html

    def put(m: "re.Match[str]") -> str:
        path = Path(folder) / m.group(1)
        if not path.is_file():
            return m.group(0)
        code = path.read_text(encoding="utf-8", errors="replace").replace("</script", "<\\/script")
        return f"<script>{code}\n</script>"
    return _LIB_TAG.sub(put, html)


_EXTERNAL = re.compile(r"""(?:src|href)\s*=\s*["']\s*(?:https?:)?//|@import\s+(?:url\()?["']?\s*(?:https?:)?//|url\(\s*["']?\s*(?:https?:)?//|\b(?:fetch|XMLHttpRequest|WebSocket|importScripts)\s*\(""", re.I)


def extract_answer(text: str, kind: str) -> str:
    """The deliverable in a reply: the HTML document of a page brief (empty when there is none), the whole reply of a text brief."""
    return extract_html(text) if kind == "page" else (text or "").strip()


def static_check(html: str) -> dict[str, Any]:
    """What can be said about a page without running it."""
    return {"chars": len(html), "external_references": len(_EXTERNAL.findall(html)), "uses_canvas": "<canvas" in html.lower(), "uses_webgl": bool(re.search(r"webgl|THREE\.", html)),
            "loads_library": bool(_LIB_TAG.search(html))}
