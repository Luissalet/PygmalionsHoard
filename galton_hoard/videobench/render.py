"""Rendering an animation: a headless browser in another process, the frames, the contact sheet and the automatic checks.

The browser runs in a Python that has Playwright (the setting ``videobench.browser_python``, else the Python of the app when it has it, else the first ``python`` /
``py`` on the PATH that has it). The app itself does not need Playwright: it only reads the PNG files the worker writes.
"""
from __future__ import annotations

import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Optional

from ..errors import GaltonError
from .contract import HEIGHT, WIDTH

WORKER = Path(__file__).resolve().parent / "render_worker.py"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
SHEET_COLUMNS, THUMB = 4, (480, 270)


def _has_playwright(python: str) -> bool:
    try:
        done = subprocess.run([python, "-c", "import playwright"], capture_output=True, timeout=30, creationflags=_NO_WINDOW)
        return done.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


_cache: dict[str, str] = {}


def browser_python(configured: str = "") -> str:
    """The Python that renders. A configured path is trusted (a wrong one fails with its own message when used); otherwise the candidates are tried once."""
    configured = (configured or "").strip()
    if configured:
        return configured
    if "found" in _cache:
        return _cache["found"]
    candidates = [sys.executable, shutil.which("python") or "", shutil.which("python3") or "", shutil.which("py") or ""]
    for c in dict.fromkeys(x for x in candidates if x):
        if _has_playwright(c):
            _cache["found"] = c
            return c
    return ""


def render(html_path: Path, out_dir: Path, *, python: str, frames: int, duration_cap: float, default_duration: float, timeout_s: float = 20.0,
           total_timeout_s: float = 240.0) -> dict[str, Any]:
    """Run the worker. Returns its report (``ok`` false with an ``error`` when the page did not render); raises only when there is no browser to run."""
    if not python:
        raise GaltonError("unavailable", "vb_no_browser")
    cfg = {"html": str(html_path), "out": str(out_dir), "width": WIDTH, "height": HEIGHT, "frames": frames, "duration_cap": duration_cap, "default_duration": default_duration,
           "timeout_s": timeout_s}
    try:
        done = subprocess.run([python, str(WORKER)], input=json.dumps(cfg), capture_output=True, text=True, encoding="utf-8", timeout=total_timeout_s, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"the page did not finish rendering within {int(total_timeout_s)} s"}
    except OSError as exc:
        raise GaltonError("unavailable", "vb_browser_start", python=python, error=str(exc)) from exc
    lines = [ln for ln in done.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        return {"ok": False, "error": (done.stderr or "the renderer printed nothing").strip()[-400:]}
    try:
        return json.loads(lines[-1])
    except ValueError:
        return {"ok": False, "error": "the renderer answered with something that is not JSON"}


# ------------------------------------------------------------------------------------------------- what the frames show
def _small(path: Path, size: tuple[int, int] = (160, 90)):
    from PIL import Image
    import numpy as np
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize(size), dtype="float32")


def measure(frames_dir: Path, files: list[str], times: list[float]) -> dict[str, Any]:
    """Automatic measures of the frames (no model): blank frames, how much the picture changes, still stretches, how far any frame gets from the first (a loop ends where it began, so the last frame is not the measure)."""
    import numpy as np
    pics = [_small(frames_dir / f) for f in files]
    gray = [p.mean(axis=2) for p in pics]
    blank = [bool(g.std() < 2.0) for g in gray]
    diffs = [float(np.abs(a - b).mean() / 255.0) for a, b in zip(gray, gray[1:])]
    still = sum(1 for d in diffs if d < 0.002)
    first_last = float(np.abs(gray[0] - gray[-1]).mean() / 255.0) if len(gray) > 1 else 0.0
    from_first = max((float(np.abs(gray[0] - g).mean() / 255.0) for g in gray[1:]), default=0.0)      # a perfect loop ends where it began, but it still left the first frame
    lum = [float(g.mean() / 255.0) for g in gray]
    colour = [float(np.abs(p[..., 0] - p[..., 1]).mean() + np.abs(0.5 * (p[..., 0] + p[..., 1]) - p[..., 2]).mean()) / 255.0 for p in pics]
    return {"blank_frames": sum(blank), "motion_mean": round(sum(diffs) / len(diffs), 5) if diffs else 0.0, "motion": [round(d, 5) for d in diffs], "still_steps": still,
            "first_last_change": round(first_last, 5), "max_change_from_first": round(from_first, 5), "luminance": [round(x, 3) for x in lum], "colourfulness_mean": round(sum(colour) / len(colour), 4),
            "moves": bool(diffs) and sum(diffs) / len(diffs) > 0.004 and from_first > 0.01, "not_blank": sum(blank) <= len(blank) // 3}


def contact_sheet(frames_dir: Path, files: list[str], times: list[float], out: Path) -> Path:
    """The frames in a grid, each with its time, as one PNG (4 columns of 480x270)."""
    from PIL import Image, ImageDraw
    rows = (len(files) + SHEET_COLUMNS - 1) // SHEET_COLUMNS
    sheet = Image.new("RGB", (THUMB[0] * SHEET_COLUMNS, THUMB[1] * rows), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    for i, (f, t) in enumerate(zip(files, times)):
        with Image.open(frames_dir / f) as im:
            thumb = im.convert("RGB").resize(THUMB)
        x, y = (i % SHEET_COLUMNS) * THUMB[0], (i // SHEET_COLUMNS) * THUMB[1]
        sheet.paste(thumb, (x, y))
        label = f"{t:.1f}s"
        draw.rectangle([x + 4, y + 4, x + 12 + 7 * len(label), y + 20], fill=(0, 0, 0))
        draw.text((x + 8, y + 6), label, fill=(255, 255, 255))
        draw.rectangle([x, y, x + THUMB[0] - 1, y + THUMB[1] - 1], outline=(40, 40, 40))
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "PNG", optimize=True)
    return out


def png_bytes(path: Path, *, width: Optional[int] = None) -> bytes:
    """A PNG file as bytes, optionally scaled down to ``width`` (the jury gets the contact sheet whole and two key frames smaller)."""
    from PIL import Image
    with Image.open(path) as im:
        im = im.convert("RGB")
        if width and im.width > width:
            im = im.resize((width, round(im.height * width / im.width)))
        buf = io.BytesIO()
        im.save(buf, "PNG", optimize=True)
        return buf.getvalue()
