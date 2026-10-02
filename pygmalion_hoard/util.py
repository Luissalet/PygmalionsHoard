"""Small pure helpers: identifiers, slugs, sizes, hashing, atomic JSON files."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .hoard_link import atomic, ids
from .hoard_link.atomic import read_json  # noqa: F401  (re-exported: a missing, empty or corrupt file gives the default)
from .hoard_link.text import fold, sha256_file, sha256_text, slugify  # noqa: F401  (re-exported: the shared folding, hashing and slug rules)

CHARS_PER_TOKEN = 3.6


def new_id(prefix: str) -> str:
    """``<prefix>_<ULID>`` in lowercase (shared ``ids``): sortable by creation time, strictly increasing inside the process and safe in file and
    model names. The older ``<prefix>_<12 characters>`` ids still resolve everywhere an id is looked up."""
    return f"{prefix}_{ids.new_ulid().lower()}"


def slug(text: str, limit: int = 40) -> str:
    """Lowercase ASCII identifier made of letters, digits and single hyphens (the shared slug rules; ``model`` when nothing is left)."""
    return slugify(text, max_len=limit, fallback="model")


def est_tokens(chars: int) -> int:
    """Approximate token count of ``chars`` characters of mixed Spanish and English text."""
    return int(chars / CHARS_PER_TOKEN + 0.5)


def human_bytes(n: float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def clamp_text(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def loads(value: Any, default: Any) -> Any:
    """JSON text from the database to a Python value; anything unreadable becomes ``default``."""
    if not isinstance(value, str):
        return value if value is not None else default
    try:
        return json.loads(value or json.dumps(default))
    except ValueError:
        return default


def dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def write_json_atomic(path: Path, data: Any) -> None:
    """Write JSON next to its destination and rename it over (retrying while Windows has the file open): a reader never sees half a file."""
    atomic.write_json_atomic(path, data, indent=2, ensure_ascii=False)


def dir_size(path: Path) -> int:
    """Total bytes of the files below ``path`` (a file gives its own size)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                continue
    return total


def tail_lines(path: Path, n: int = 40, max_bytes: int = 64_000) -> list[str]:
    """The last ``n`` lines of a text file, reading only its end."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read()
    except OSError:
        return []
    lines = data.decode("utf-8", "replace").splitlines()
    return lines[-n:]


def repo_dirname(repo_id: str) -> str:
    """Hugging Face repo id to the folder name used in the work dir: ``org/name`` becomes ``org--name``."""
    return repo_id.strip().strip("/").replace("/", "--")
