"""Brief bench: the same brief to several models, side by side, with what each one took and cost.

A brief is a page (one HTML document, rendered in a browser) or a text. Every model answers it; the app measures time, tokens and the cost those tokens would have at the price the
person gives, renders the pages, and an optional judge that did not write them scores every answer without knowing who wrote it. The result is a grid of renders and a table of figures.
"""
from __future__ import annotations

from .briefs import BRIEFS, BY_ID, CATEGORIES, KINDS, custom, public
from .contract import build_messages, extract_answer, inline_libraries, library_file, static_check
from .judge import DIMENSIONS, RUBRIC, judge_messages, parse_judge

__all__ = ["BRIEFS", "BY_ID", "CATEGORIES", "DIMENSIONS", "KINDS", "RUBRIC", "build_messages", "custom", "extract_answer", "inline_libraries", "judge_messages", "library_file",
           "parse_judge", "public", "static_check"]
