"""Which folders and files the app may read on behalf of the user (or of the assistant).

The rules are the shared ones of Hoard Link (``hoard_link.paths``); this module keeps the names the app and its tests call and words the reasons
with the app's own message catalogue, so the interface still translates them."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from .hoard_link import paths as shared
from .hoard_link.paths import HIDDEN_PARTS, SECRET_NAMES, clean_user_path  # noqa: F401
from .messages import text

#: The shared reason for a code, in English, back to the code: the shared functions answer with text, the catalogue is keyed by code.
_CODE_OF = {entry["en"]: code for code, entry in shared.REASONS.items()}
#: Codes the catalogue words with another entry (the interface has no text of its own for them).
_KEY = {"path_contains_own_data": "path_own_data_folder", "path_is_file": "folder_missing"}


def _worded(reason: Optional[str], *, file: bool = False) -> str:
    """"" when the path may be used, else the reason as the catalogue's coded text."""
    if not reason:
        return ""
    code = _CODE_OF.get(reason, "path_invalid")
    key = _KEY.get(code, code)
    if file and key == "path_own_data_folder":
        key = "path_own_data_file"
    return text(key)


def hidden_names(parts: Iterable[str]) -> set[str]:
    """The configuration-like names (``.git``, ``AppData``, ``node_modules``...) among folder names; case-insensitive."""
    return {x.lower() for x in parts} & HIDDEN_PARTS


def unsafe_folder(path: str | Path, data_dir: Optional[Path] = None, *, allow_data_subdir: Optional[Path] = None) -> str:
    """Empty when the folder may be read; otherwise the reason it must not. ``allow_data_subdir`` exempts one folder of the app's own data."""
    return _worded(shared.unsafe_folder(path, data_dir=data_dir, allow_data_subdir=allow_data_subdir, lang="en"))


def unsafe_file(path: str | Path, data_dir: Optional[Path] = None, *, allow_data_subdir: Optional[Path] = None) -> str:
    """Empty when the file may be read; otherwise the reason."""
    return _worded(shared.unsafe_file(path, data_dir=data_dir, allow_data_subdir=allow_data_subdir, lang="en"), file=True)
