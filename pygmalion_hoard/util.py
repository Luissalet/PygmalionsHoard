"""Small pure helpers: identifiers, slugs, sizes, hashing, atomic JSON files."""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import time
import unicodedata
from pathlib import Path
from typing import Any, Optional

from .hoard_link import atomic
from .hoard_link.atomic import read_json  # noqa: F401  (re-exported: a missing, empty or corrupt file gives the default)

_ID_ALPHABET = "0123456789abcdefghjkmnpqrstvwxyz"
CHARS_PER_TOKEN = 3.6


def new_id(prefix: str, now: Optional[float] = None) -> str:
    """``<prefix>_<time><random>``: sortable by creation time, unique enough for a local database."""
    millis = int((now if now is not None else time.time()) * 1000)
    stamp = ""
    for _ in range(7):
        millis, rest = divmod(millis, 32)
        stamp = _ID_ALPHABET[rest] + stamp
    tail = "".join(secrets.choice(_ID_ALPHABET) for _ in range(5))
    return f"{prefix}_{stamp}{tail}"


def fold(text: str) -> str:
    """Lowercase without accents."""
    return "".join(c for c in unicodedata.normalize("NFD", text or "") if unicodedata.category(c) != "Mn").lower()


def slug(text: str, limit: int = 40) -> str:
    """Lowercase ASCII identifier made of letters, digits and single hyphens."""
    out = re.sub(r"[^a-z0-9]+", "-", fold(text)).strip("-")
    return out[:limit].strip("-") or "model"


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


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
