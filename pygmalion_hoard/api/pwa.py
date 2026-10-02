"""PWA manifest and service worker: makes the app installable on the phone home screen.

The service worker caches only the built ``/assets/`` files (their names carry a content hash, so cache-first is safe) and its cache is named
after the build: the hash of ``index.html``, which lists those files. A new build therefore starts a new cache and the activation of its worker
deletes every older one. The page itself (``/``, ``index.html``), the icons, the manifest and the worker are never cached by it: they always go to
the network, so an old index or an old icon cannot outlive a new build. Every ``/api/`` request goes to the network too. No offline page.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from fastapi import APIRouter, Response

from .. import __version__

router = APIRouter()
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_MANIFEST = {
    "name": "Pygmalion's Hoard",
    "short_name": "Pygmalion",
    "start_url": "/",
    "display": "standalone",
    "background_color": "#1d1417",
    "theme_color": "#1d1417",
    "lang": "es",
    "icons": [
        {"src": "/icon-192.png", "sizes": "192x192", "type": "image/png", "purpose": "any"},
        {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png", "purpose": "any maskable"},
    ],
}


def build_id() -> str:
    """Identifies the built client: the hash of its ``index.html`` (which names every hashed asset), else the app version."""
    try:
        return hashlib.sha256((STATIC_DIR / "index.html").read_bytes()).hexdigest()[:12]
    except OSError:
        return f"v{__version__}"


def _service_worker() -> str:
    cache_name = json.dumps(f"pygmalion-hoard-{build_id()}")
    return f"""// Minimal service worker: installability plus a cache-first strategy for the built /assets/ files only. The cache is named after the build.
// The page, the icons, the manifest, this worker and every /api/ request always go to the network.
const CACHE_NAME = {cache_name};

self.addEventListener("install", () => {{
  self.skipWaiting();
}});

self.addEventListener("activate", (event) => {{
  event.waitUntil(
    (async () => {{
      const names = await caches.keys();
      await Promise.all(names.filter((name) => name !== CACHE_NAME).map((name) => caches.delete(name)));
      await self.clients.claim();
    }})(),
  );
}});

self.addEventListener("fetch", (event) => {{
  const request = event.request;
  if (request.method !== "GET") return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  if (request.mode === "navigate") return; // the page is never served from a cache
  if (!url.pathname.startsWith("/assets/")) return; // index.html, icons, manifest, sw.js, /api/: normal browser handling (network)

  event.respondWith(
    (async () => {{
      const cache = await caches.open(CACHE_NAME);
      const cached = await cache.match(request);
      if (cached) return cached;
      const response = await fetch(request);
      if (response.ok) cache.put(request, response.clone());
      return response;
    }})(),
  );
}});
"""


@router.get("/manifest.webmanifest", include_in_schema=False)
def manifest() -> Response:
    return Response(content=json.dumps(_MANIFEST), media_type="application/manifest+json", headers={"Cache-Control": "no-cache"})


@router.get("/sw.js", include_in_schema=False)
def service_worker() -> Response:
    return Response(content=_service_worker(), media_type="application/javascript", headers={"Service-Worker-Allowed": "/", "Cache-Control": "no-cache"})
