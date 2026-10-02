"""Which architectures ``convert_hf_to_gguf.py`` knows, read from the sources themselves.

A converter registers itself with ``@ModelBase.register("QwenForCausalLM", "QwenForCausalLM2", ...)`` (older versions ``@Model.register``),
with one or several names, on one line or many. In the old layout every converter lives in ``convert_hf_to_gguf.py``; in the new one the
script imports them from a ``conversion`` package that sits next to it (``<src>/conversion/*.py``, possibly with subfolders), and the
registrations are there. Reading the names from the text means no import and no dependency on the version installed.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterator, Optional

from .messages import text
from .util import read_json

_REGISTER = re.compile(r"@\s*(?:ModelBase|Model)\s*\.\s*register\s*\(")
PACKAGE_DIR = "conversion"
_cache: dict[tuple, frozenset[str]] = {}


def _call_body(text: str, start: int) -> str:
    """The text of a call from just after its opening parenthesis to the matching one, skipping strings and comments."""
    quote = ""
    depth = 1
    i = start
    while i < len(text):
        ch = text[i]
        if quote:
            if ch == "\\":
                i += 1
            elif ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch == "#":
            newline = text.find("\n", i)
            i = len(text) if newline < 0 else newline
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return text[start:i]
        i += 1
    return text[start:]


def _names_in(text: str) -> set[str]:
    found: set[str] = set()
    for match in _REGISTER.finditer(text):
        body = _call_body(text, match.end())
        body = re.sub(r"#[^\n]*", "", body)
        found.update(re.findall(r"""["']([^"'\s]+)["']""", body))
    return found


def conversion_sources(script: Path) -> Iterator[Path]:
    """The script and every ``.py`` file of the ``conversion`` package next to it (recursively), in a stable order."""
    yield script
    package = script.parent / PACKAGE_DIR
    if package.is_dir():
        yield from sorted(p for p in package.rglob("*.py") if p.is_file())


def registered_architectures(script: str | Path) -> frozenset[str]:
    path = Path(script)
    try:
        files = list(conversion_sources(path))
        key = tuple((str(f), f.stat().st_mtime_ns, f.stat().st_size) for f in files)
    except OSError:
        return frozenset()
    if key in _cache:
        return _cache[key]
    names: set[str] = set()
    for f in files:
        try:
            names.update(_names_in(f.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    _cache[key] = frozenset(names)
    return _cache[key]


def model_architectures(model_dir: str | Path) -> list[str]:
    """``architectures`` from the model's config.json (the first entry of a nested text config when the top one is missing)."""
    config = read_json(Path(model_dir) / "config.json", {}) or {}
    names = config.get("architectures")
    if not names and isinstance(config.get("text_config"), dict):
        names = config["text_config"].get("architectures")
    return [str(n) for n in (names or [])]


def convertible(model_dir: str | Path, script: Optional[str | Path]) -> dict[str, Any]:
    """``{known: True|False|None, architecture, note}``: None when the script is not there to ask."""
    arches = model_architectures(model_dir)
    arch = arches[0] if arches else ""
    if not script or not Path(script).is_file():
        return {"known": None, "architecture": arch, "note": text("convert_note_no_script")}
    registered = registered_architectures(script)
    if not registered:
        return {"known": None, "architecture": arch, "note": text("convert_note_no_arches")}
    if not arch:
        return {"known": None, "architecture": "", "note": text("convert_note_no_arch")}
    if arch in registered:
        return {"known": True, "architecture": arch, "note": ""}
    return {"known": False, "architecture": arch, "note": text("convert_note_unregistered", arch=arch)}
