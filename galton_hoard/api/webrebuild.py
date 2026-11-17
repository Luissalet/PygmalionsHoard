"""Pictures of the website-rebuild bench for the screen: reference, candidate and diff pictures of every checkpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

from .deps import services

router = APIRouter()
SAFE_HEADERS = {"X-Content-Type-Options": "nosniff", "Cache-Control": "private, max-age=300", "Cross-Origin-Resource-Policy": "same-origin"}


@router.get("/api/webrebuild/{run}/{site}/{slot}/{name}")
def picture(request: Request, run: str, site: str, slot: str, name: str):
    if request.headers.get("sec-fetch-site") not in (None, "same-origin", "none"):
        raise HTTPException(403, "Files are only served to the app itself.")
    return FileResponse(services(request).webrebuild.file_path(run, site, slot, name), media_type="image/png", headers=SAFE_HEADERS)
