"""Program output made fit for a log: terminal escape sequences removed and progress redrawn in place collapsed to its last state.

Programs such as ``ollama create`` and ``llama-quantize`` draw a spinner or a progress bar by rewriting one terminal line again and again
(``\\r`` plus ANSI cursor and mode sequences). Read line by line that is one log line per frame; stored as it is, the log tail is a wall of
``ESC[?2026h`` and braille characters. ``clean_line`` strips the sequences, ``Collapser`` decides when a line replaces the one before it and
``clean_lines`` does both for lines already stored (old logs are cleaned when they are read)."""

from __future__ import annotations

import re
from typing import Iterable, Optional

CSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
OSC = re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)?")
OTHER_ESC = re.compile(r"\x1b[@-Z\\-_]|\x1b[()][A-Za-z0-9]|\x1b[=>78]")
CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")
SPINNER = re.compile("[⠀-⣿]")                       # braille dots: the spinner frames of Go and Node programs
PERCENT = re.compile(r"\d+(?:[.,]\d+)?\s*%")
BAR = re.compile(r"[▀-▟─-╿■-◿]+")  # block, box and shape characters used to draw bars
KEY_LEN = 18


def strip_ansi(value: str) -> str:
    """``value`` without terminal escape sequences (CSI, OSC and the short ones) or other control characters."""
    value = OSC.sub("", value)
    value = CSI.sub("", value)
    value = OTHER_ESC.sub("", value)
    return CONTROL.sub("", value)


def clean_line(raw: str) -> tuple[str, bool]:
    """``(text, is_progress)`` for one output line. A carriage return inside the line keeps only what was drawn last. ``is_progress``: the line
    is a frame of a spinner or a bar (it was drawn with escape codes, has a spinner glyph, a bar or a percentage), so the next frame of the same activity replaces it."""
    if "\r" in raw:
        parts = [p for p in raw.split("\r") if strip_ansi(p).strip()]
        raw = parts[-1] if parts else ""
        redrawn = True
    else:
        redrawn = False
    text = strip_ansi(raw)
    progress = redrawn or "\x1b" in raw or bool(SPINNER.search(text)) or bool(BAR.search(text)) or bool(PERCENT.search(text))
    text = SPINNER.sub("", text)
    return re.sub(r"[ \t]{2,}", " ", text).strip(), progress


def activity_key(text: str) -> str:
    """What identifies the activity a progress frame belongs to: its leading letters, ignoring numbers, bars and units."""
    return re.sub(r"[^A-Za-z]+", "", text).lower()[:KEY_LEN]


class Collapser:
    """Decides, line by line, whether a cleaned line is new or the next frame of the line before it (``feed`` returns ``(text, replace)``;
    ``text`` is None for a line that was only escape codes)."""

    def __init__(self) -> None:
        self._key: Optional[str] = None
        self._progress = False

    def feed(self, raw: str) -> tuple[Optional[str], bool]:
        text, progress = clean_line(raw)
        if not text:
            return (None, False) if raw.strip() else ("", False)
        key = activity_key(text)
        replace = bool(self._progress and key and key == self._key)
        self._key, self._progress = key, progress
        return text, replace


def clean_lines(lines: Iterable[str]) -> list[str]:
    """Lines as stored by an older version: escape sequences stripped and progress frames collapsed to the last one."""
    out: list[str] = []
    collapser = Collapser()
    for raw in lines:
        text, replace = collapser.feed(raw)
        if text is None:
            continue
        if replace and out:
            out[-1] = text
        else:
            out.append(text)
    return out
