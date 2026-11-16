"""Rendering a page of the brief bench: the browser runs in another process (``shot_worker.py``); this module starts it and measures the pictures it leaves.

The browser is a Python that has Playwright (``briefbench.browser_python``, else the one of the video bank, else the Python of the app when it has it, else the first
``python`` / ``py`` on the PATH that has it). The app itself does not need Playwright: it only reads the PNG files.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from ..errors import GaltonError
from ..videobench.render import browser_python as _browser_python
from .contract import HEIGHT, WIDTH

WORKER = Path(__file__).resolve().parent / "shot_worker.py"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
THUMB = (480, 270)


def browser_python(configured: str = "", fallback: str = "") -> str:
    return _browser_python((configured or fallback or "").strip())


def render(html_path: Path, out_dir: Path, *, python: str, settle_ms: int, motion_ms: int, timeout_s: float = 30.0, total_timeout_s: float = 120.0) -> dict[str, Any]:
    """Run the worker. Returns its report (``ok`` false with an ``error`` when the page did not render); raises only when there is no browser to run."""
    if not python:
        raise GaltonError("unavailable", "bk_no_browser")
    cfg = {"html": str(html_path), "out": str(out_dir), "width": WIDTH, "height": HEIGHT, "settle_ms": settle_ms, "motion_ms": motion_ms, "timeout_s": timeout_s}
    try:
        done = subprocess.run([python, str(WORKER)], input=json.dumps(cfg), capture_output=True, text=True, encoding="utf-8", timeout=total_timeout_s, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the page did not finish rendering within {int(total_timeout_s)} s"}
    except OSError as exc:
        raise GaltonError("unavailable", "bk_browser_start", python=python, error=str(exc)) from exc
    lines = [ln for ln in done.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"ok": False, "error": (done.stderr or "the renderer printed nothing").strip()[-400:]}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"ok": False, "error": "the renderer answered with something that is not JSON"}


def _small(path: Path, size: tuple[int, int] = (160, 90)):
    from PIL import Image
    import numpy as np
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(size), dtype="float32")


def measure(folder: Path, files: list[str]) -> dict[str, Any]:
    """What the pictures say without a model: is the first one blank, how much it changes in the second one and how colourful it is."""
    import numpy as np
    first = _small(folder / files[0])
    gray = first.mean(axis=2)
    blank = bool(gray.std() < 2.0)
    out: dict[str, Any] = {"blank": blank, "not_blank": not blank, "luminance": round(float(gray.mean() / 255.0), 3),
                           "colourfulness": round(float(np.abs(first[..., 0] - first[..., 1]).mean() + np.abs(0.5 * (first[..., 0] + first[..., 1]) - first[..., 2]).mean()) / 255.0, 4)}
    if len(files) > 1:
        change = float(np.abs(gray - _small(folder / files[1]).mean(axis=2)).mean() / 255.0)
        out.update(motion=round(change, 5), moves=change > 0.002)
    else:
        out.update(motion=None, moves=None)
    return out


def thumbnail(folder: Path, name: str = "shot.png", out: str = "thumb.png") -> Path:
    from PIL import Image
    with Image.open(folder / name) as im:
        im.convert("RGB").resize(THUMB).save(folder / out, "PNG", optimize=True)
    return folder / out


def png_bytes(path: Path, *, width: int = 0) -> bytes:
    import io
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if width and im.width > width:
            im = im.resize((width, round(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue()
