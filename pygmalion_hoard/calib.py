"""Calibration text for the importance matrix: the user's own text (a dataset version) or the bundled Spanish and English sample."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from .datasets.formats import content_text, status_of
from .errors import PygmalionError
from .util import sha256_text

BUNDLED = Path(__file__).resolve().parent / "calib" / "es-en-general.txt"
MIN_CHARS = 20_000


def tag_of(spec: Optional[dict[str, Any]]) -> str:
    """The short identity of a calibration source: the same request always gives the same tag, so two importance matrices with equal tags were
    computed on the same text (the cached calibration file is named after it)."""
    return sha256_text(json.dumps(spec or {"source": "bundled"}, sort_keys=True))[:12]


def bundled_text() -> str:
    try:
        return BUNDLED.read_text(encoding="utf-8")
    except OSError as exc:
        raise PygmalionError("not_found", "calib_missing") from exc


def text_from_records(records: list[dict[str, Any]], max_chars: int = 400_000) -> str:
    """Records joined by blank lines. Chat turns keep their content only, as prose; the importance matrix needs text that looks like use."""
    parts, size = [], 0
    for rec in records:
        if status_of(rec) != "ok":
            continue
        piece = content_text(rec).strip()
        if not piece:
            continue
        parts.append(piece)
        size += len(piece) + 2
        if size >= max_chars:
            break
    return "\n\n".join(parts)


def write_calibration(dest: Path, records: Optional[list[dict[str, Any]]] = None, max_chars: int = 400_000) -> dict[str, Any]:
    """Write the calibration file and say where the text came from. A dataset that is too short is padded with the bundled text."""
    source, text = "bundled", ""
    if records:
        text = text_from_records(records, max_chars)
        source = "dataset"
        if len(text) < MIN_CHARS:
            text = (text + "\n\n" + bundled_text()).strip()
            source = "dataset+bundled"
    else:
        text = bundled_text()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(text, encoding="utf-8")
    return {"path": str(dest), "chars": len(text), "source": source}
