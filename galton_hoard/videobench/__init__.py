"""Video bank: briefs for animations written as code, rendered in a browser and scored by a jury.

The model is asked for one self-contained HTML document with ``window.seek(t)`` (the contract of the HTML animations of the video editor: the page draws the
exact frame at time ``t``). The page is rendered headless to frames, a contact sheet is made, automatic checks measure what can be measured without a model
(does it render, does it move, is ``seek`` deterministic, does it reach for the network) and a jury, a second session that did not write the page, scores it on
the rubric of seven dimensions. The briefs come from a public list of prompts people shared for such animations: the list is MIT, every video belongs to its
author, and the author and the post are kept with each brief.
"""
from __future__ import annotations

from .contract import DEFAULT_DURATION, HEIGHT, WIDTH, build_messages, extract_html, static_check
from .jury import DIMENSIONS, RUBRIC, jury_messages, parse_jury
from .prompts import CATEGORIES, PROMPTS, PROMPTS_BY_ID, public, select

__all__ = ["CATEGORIES", "DEFAULT_DURATION", "DIMENSIONS", "HEIGHT", "PROMPTS", "PROMPTS_BY_ID", "RUBRIC", "WIDTH", "build_messages", "extract_html", "jury_messages",
           "parse_jury", "public", "select", "static_check"]
