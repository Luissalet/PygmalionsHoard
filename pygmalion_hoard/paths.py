"""Which folders and files the app may read on behalf of the user (or of the assistant)."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path, PurePath
from typing import Iterable, Optional, Sequence

from .messages import text

SYSTEM_PARTS_WIN = {"windows", "program files", "program files (x86)", "programdata", "$recycle.bin", "system volume information"}
SYSTEM_PARTS_POSIX = {"etc", "usr", "bin", "sbin", "lib", "lib64", "proc", "sys", "dev", "boot", "root", "var", "run", "opt", "snap"}
SECRET_NAMES = re.compile(r"(^\.env($|\.)|mcp-token|^id_(rsa|ed25519|ecdsa|dsa)|\.pem$|\.key$|\.p12$|\.pfx$|\.kdbx$|credentials|secrets?\.|\.ssh$|\.htpasswd)", re.I)
HIDDEN_PARTS = {".git", ".ssh", ".gnupg", ".aws", ".config", "appdata", "node_modules", "__pycache__"}


def hidden_names(parts: Iterable[str]) -> set[str]:
    """The configuration-like names (``.git``, ``AppData``, ``node_modules``...) among folder names; case-insensitive."""
    return {x.lower() for x in parts} & HIDDEN_PARTS


def hidden_parts(directory: PurePath, temp: Optional[PurePath] = None) -> set[str]:
    """Configuration-like folder names among the parts of ``directory`` (a folder, never a file). Pure: it only looks at the
    parts, so it behaves the same with a ``PureWindowsPath`` (drive letter, backslashes, case-insensitive names) as with a
    POSIX path. Inside the system temp folder ``temp`` (which lives under AppData on Windows) only the parts below it count."""
    parts: Sequence[str] = directory.parts
    if temp is not None:
        try:
            parts = directory.relative_to(temp).parts
        except ValueError:
            pass
    return hidden_names(parts)


def _temp_dir() -> Optional[Path]:
    try:
        return Path(tempfile.gettempdir()).resolve()
    except (OSError, RuntimeError):
        return None


def _hidden_in(directory: Path) -> set[str]:
    return hidden_parts(directory, _temp_dir())


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(Path(parent).resolve())
        return True
    except (ValueError, OSError):
        return False


def _is_root(path: Path) -> bool:
    return path == path.parent or len(path.parts) <= 1


def _system(p: Path, data_dir: Optional[Path]) -> bool:
    parts = {x.lower() for x in p.parts}
    if parts & SYSTEM_PARTS_WIN and (p.drive or os.name == "nt"):
        return True
    if os.name != "nt" and len(p.parts) > 1 and p.parts[1].lower() in SYSTEM_PARTS_POSIX and not (data_dir and _inside(p, data_dir)):
        # /var/tmp and the like are fine for tests; the real system trees are not
        if p.parts[1].lower() != "var" or len(p.parts) < 3 or p.parts[2].lower() not in ("tmp", "folders"):
            return True
    return False


def unsafe_folder(path: str | Path, data_dir: Optional[Path] = None, *, allow_data_subdir: Optional[Path] = None) -> str:
    """Empty when the folder may be read; otherwise the reason it must not. ``allow_data_subdir`` exempts one folder of the app's own data."""
    try:
        p = Path(path).expanduser()
    except (OSError, ValueError, RuntimeError):
        return text("path_invalid")
    if not p.is_absolute():
        return text("path_relative")
    try:
        p = p.resolve()
    except OSError:
        return text("path_unresolved")
    if not p.is_dir():
        return text("folder_missing")
    if _is_root(p):
        return text("path_root")
    try:
        home = Path.home().resolve()
    except (OSError, RuntimeError):
        home = None
    if home is not None and p == home:
        return text("path_home")
    if _system(p, data_dir):
        return text("path_system")
    if data_dir is not None and _inside(p, data_dir) and not (allow_data_subdir and _inside(p, allow_data_subdir)):
        return text("path_own_data_folder")
    if _hidden_in(p):
        return text("path_hidden")
    return ""


def unsafe_file(path: str | Path, data_dir: Optional[Path] = None, *, allow_data_subdir: Optional[Path] = None) -> str:
    """Empty when the file may be read; otherwise the reason."""
    try:
        p = Path(path).expanduser()
    except (OSError, ValueError, RuntimeError):
        return text("path_invalid")
    if not p.is_absolute():
        return text("path_relative")
    try:
        p = p.resolve()
    except OSError:
        return text("path_unresolved")
    if not p.is_file():
        return text("file_missing")
    if SECRET_NAMES.search(p.name):
        return text("path_credentials")
    if _system(p, data_dir):
        return text("path_system")
    if data_dir is not None and _inside(p, data_dir) and not (allow_data_subdir and _inside(p, allow_data_subdir)):
        return text("path_own_data_file")
    if _hidden_in(p.parent):
        return text("path_hidden")
    return ""
