"""Files of the video bank for the screen: the contact sheet and frames of an item, and its page to play in a sandboxed frame."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .deps import services

router = APIRouter()
SAFE_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300", "Cross-Origin-Resource-Policy": "same-origin"}
# The page is written by a model: it runs in a sandbox with no origin, no network and no navigation, whatever it contains.
PLAY_HEADERS = {**SAFE_HEADERS, "Cache-Control": "no-store", "X-Hoard-Raw": "1",     # the Pygmalion shell leaves a page marked like this as it is (no bar, no title change)
                "Content-Security-Policy": "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src data: blob:"}
#: Runs first in the page: lets the screen pause the page's own animation loop and draw a chosen time, by message.
CONTROLLER = ("<script>(function(){var raf=window.requestAnimationFrame,paused=false;"
              "window.requestAnimationFrame=function(f){return paused?0:raf.call(window,f)};"
              "addEventListener('message',function(e){var m=e.data||{};if(m.pause)paused=true;"
              "if(typeof m.seek==='number'&&typeof window.seek==='function'){paused=true;window.seek(m.seek)}});"
              "addEventListener('load',function(){parent.postMessage({videobench:'ready',duration:typeof window.DURATION==='number'?window.DURATION:null},'*')})})();</script>")


def _same_origin(request: Request) -> None:
    if request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
        raise HTTPException(403, "Files are only served to the app itself.")


@router.get("/api/videobench/{run}/{prompt}/play.html")
def play(request: Request, run: str, prompt: str):
    _same_origin(request)
    path = services(request).videobench.file_path(run, prompt, "index.html")
    html = path.read_text(encoding="utf-8", errors="replace")
    lower = html.lower()
    head = lower.find("<head")
    at = html.find(">", head) + 1 if head >= 0 and html.find(">", head) >= 0 else 0
    return Response((html[:at] + CONTROLLER + html[at:]).encode("utf-8"), media_type="text/html; charset=utf-8", headers=PLAY_HEADERS)


@router.get("/api/videobench/{run}/{prompt}/{name:path}")
def item_file(request: Request, run: str, prompt: str, name: str):
    _same_origin(request)
    path = services(request).videobench.file_path(run, prompt, name)
    if name == "index.html":
        return FileResponse(path, media_type="text/plain; charset=utf-8", headers=PLAY_HEADERS, filename=f"{prompt}.html")
    return FileResponse(path, media_type="image/png", headers=SAFE_HEADERS)
