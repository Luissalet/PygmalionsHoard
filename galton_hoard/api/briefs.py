"""Files of the brief bench for the screen: the comparison grid of a brief, the pictures and text of an answer, and its page to play in a sandboxed frame."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse, Response

from .deps import services

router = APIRouter()
SAFE_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=60", "Cross-Origin-Resource-Policy": "same-origin"}
# The page is written by a model: it runs in a sandbox with no origin, no network and no navigation, whatever it contains.
PLAY_HEADERS = {**SAFE_HEADERS, "Cache-Control": "no-store", "X-Hoard-Raw": "1",     # the Pygmalion shell leaves a page marked like this as it is (no bar, no title change)
                "Content-Security-Policy": "sandbox allow-scripts; default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data: blob:; font-src data:; media-src data: blob:"}


def _same_origin(request: Request) -> None:
    if request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
        raise HTTPException(403, "Files are only served to the app itself.")


@router.get("/api/briefbench/{run}/{brief}/grid.png")
def grid(request: Request, run: str, brief: str):
    _same_origin(request)
    return FileResponse(services(request).briefbench.grid_path(run, brief), media_type="image/png", headers={**SAFE_HEADERS, "Cache-Control": "no-store"})


@router.get("/api/briefbench/{run}/{brief}/{slot}/play.html")
def play(request: Request, run: str, brief: str, slot: str):
    _same_origin(request)
    html = services(request).briefbench.page_html(run, brief, slot)
    return Response(html.encode("utf-8"), media_type="text/html; charset=utf-8", headers=PLAY_HEADERS)


@router.get("/api/briefbench/{run}/{brief}/{slot}/{name:path}")
def item_file(request: Request, run: str, brief: str, slot: str, name: str):
    _same_origin(request)
    path = services(request).briefbench.file_path(run, brief, slot, name)
    if name == "index.html":
        return FileResponse(path, media_type="text/plain; charset=utf-8", headers=PLAY_HEADERS, filename=f"{brief}-{slot}.html")
    if name == "answer.txt":
        return FileResponse(path, media_type="text/plain; charset=utf-8", headers=SAFE_HEADERS)
    return FileResponse(path, media_type="image/png", headers=SAFE_HEADERS)
