"""Atomic writes for the worker scripts.

The workers run as plain scripts in the trainer's own Python, so they cannot import the app package. This loads the shared Hoard Link ``atomic.py``
(standard library only) from the vendored copy next to them by its path. When that copy is not there (a worker folder copied elsewhere) the same
calls fall back to a bare ``os.replace`` with a short retry, which is what the workers did before.
"""

from __future__ import annotations

import importlib.util
import os
import time
from pathlib import Path
from typing import Any, Optional


def _shared() -> Optional[Any]:
    path = Path(__file__).resolve().parent.parent / "hoard_link" / "atomic.py"
    try:
        spec = importlib.util.spec_from_file_location("_hoard_link_atomic", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:  # noqa: BLE001 - a missing or unreadable copy must not stop a training run
        return None


_atomic = _shared()


def replace_with_retry(src: Any, dst: Any) -> None:
    """``os.replace`` that waits out a Windows sharing violation (a file or folder another program holds for a moment)."""
    if _atomic is not None:
        _atomic.replace_with_retry(src, dst)
        return
    for attempt in range(40):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(min(0.25, 0.05 * (1 + attempt)))


def write_text_atomic(path: Any, text: str) -> None:
    """Write ``text`` next to ``path`` and replace it, so a reader (the job view, a resume) never sees half a file."""
    if _atomic is not None:
        _atomic.write_text_atomic(path, text)
        return
    target = Path(path)
    tmp = target.with_name(target.name + f".{os.getpid()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    replace_with_retry(tmp, target)
