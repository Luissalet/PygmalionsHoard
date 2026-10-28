"""The world a language-bench trial acts on: the shop and the family tools of one language, plus the calculator."""

from __future__ import annotations

from typing import Any

from ..harnessbench.world import ToolResult, ToolSpec, World
from .family import FamilyTools
from .shop import ShopTools

CALC = {"en": ("Evaluate an arithmetic expression. Supports + - * / // % ** parentheses and sqrt(), abs(), round().", "The expression, for example (3 + 4) * 2."),
        "es": ("Evalúa una expresión aritmética. Admite + - * / // % **, paréntesis y sqrt(), abs(), round().", "La expresión, por ejemplo (3 + 4) * 2.")}


def calculator_spec(lang: str) -> ToolSpec:
    desc, arg = CALC[lang]
    return ToolSpec("calculator", desc, {"type": "object", "properties": {"expression": {"type": "string", "description": arg}}, "required": ["expression"]})


class LangWorld(ShopTools, FamilyTools, World):
    """Shop and household tools in one language. ``ToolResult.empty`` is set when a search found nothing."""

    def __init__(self, lang: str):
        World.__init__(self)
        self.lang = lang
        self.cart: dict[str, int] = {}
        self.notes: dict[str, str] = {}
        self._empty = False

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        self._empty = False
        out = World.run(self, name, args)
        out.empty = bool(self._empty and out.ok)
        return out
