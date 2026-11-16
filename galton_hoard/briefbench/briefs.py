"""The briefs that ship with the bench: short, self-contained requests whose answers differ enough between models to be worth putting side by side.

A brief has an ``id``, a ``title``, a ``kind`` (``page``: the answer is one HTML document that gets rendered; ``text``: the answer is read as it is), the ``prompt``
and, for a page, the libraries it may lean on (``libraries``: names of files the person keeps in the ``briefbench.libraries`` folder). Anything else is a
brief the person writes in the call: the catalogue is a starting point, not a limit.
"""
from __future__ import annotations

from typing import Any, Optional

KINDS = ("page", "text")

BRIEFS: list[dict[str, Any]] = [
    {"id": "dragon-castle", "title": "Dragon over a castle", "kind": "page", "category": "3d", "libraries": ["three"],
     "prompt": "Build a 3D scene with Three.js: a dragon circles above a stone castle on a cliff at dusk. The castle has towers, walls and lit windows; the dragon flaps its wings and "
               "follows a smooth flight path around the towers; there are a few clouds, stars coming out and a slow camera that orbits the scene. Everything is made from code (primitives, "
               "no model files). It must start by itself and look good without any interaction."},
    {"id": "solar-orrery", "title": "Solar system orrery", "kind": "page", "category": "canvas", "libraries": [],
     "prompt": "Draw an animated orrery of the solar system on a canvas: the sun, the eight planets with orbits at plausible relative speeds, Saturn's rings, the moon around the earth, "
               "a faint starfield and a label that follows each planet. A slider changes the speed of time. Use only the canvas and plain JavaScript."},
    {"id": "pricing-page", "title": "Pricing page", "kind": "page", "category": "ui", "libraries": [],
     "prompt": "Design a pricing page for a made-up project-management tool with three plans and a monthly/yearly switch that changes the prices. Include a comparison table, a short FAQ "
               "with expandable answers and a footer. It has to look professional, work on a phone width and be usable with the keyboard. Plain HTML, CSS and JavaScript."},
    {"id": "svg-dashboard", "title": "Sales dashboard", "kind": "page", "category": "dataviz", "libraries": [],
     "prompt": "Make a sales dashboard in one page with invented data for twelve months: a line chart of revenue, a bar chart by region, a donut of the product mix and four headline numbers "
               "with their change against the previous month. Draw the charts as inline SVG by hand, with tooltips on hover and a clean, consistent palette."},
    {"id": "snake-game", "title": "Snake game", "kind": "page", "category": "interactive", "libraries": [],
     "prompt": "Write the game Snake in one page: arrow keys and WASD, a score, a best score kept while the page is open, speed that grows as the snake grows, walls that kill, a game-over "
               "screen with a restart key and a pause key. It should look polished. Before any key is pressed, the snake already moves on its own in a demo mode so the page is never static."},
    {"id": "explain-recursion", "title": "Explain recursion (Spanish)", "kind": "text", "category": "writing", "libraries": [],
     "prompt": "Explica la recursividad a una persona que sabe programar un poco pero nunca la ha entendido. Usa un ejemplo con código en Python, uno de los fallos típicos y una forma "
               "de comprobar que una función recursiva termina. Máximo 350 palabras."},
    {"id": "find-the-bug", "title": "Find the bug", "kind": "text", "category": "code", "libraries": [],
     "prompt": "This function should return the n-th Fibonacci number modulo m, but it is wrong for some inputs. Say exactly which inputs fail and why, then give a corrected version.\n\n"
               "def fib_mod(n, m):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, (a + b) % m\n    return a\n\n"
               "Check your claim by reasoning on at least two concrete inputs, including m = 1 and n = 0."},
]
BY_ID = {b["id"]: b for b in BRIEFS}
CATEGORIES = sorted({b["category"] for b in BRIEFS})


def public(b: dict[str, Any], *, full: bool = False) -> dict[str, Any]:
    out = {"id": b["id"], "title": b["title"], "kind": b["kind"], "category": b["category"], "libraries": list(b.get("libraries") or []),
           "excerpt": " ".join(b["prompt"].split())[:200]}
    if full:
        out["prompt"] = b["prompt"]
    return out


def custom(prompt: str, *, title: str = "", kind: str = "page", libraries: Optional[list[str]] = None, criteria: str = "", index: int = 1) -> dict[str, Any]:
    """A brief the person wrote in the call: its id is ``own-<n>`` so that several can be given at once."""
    return {"id": f"own-{index}", "title": (title or " ".join(prompt.split())[:60]).strip(), "kind": kind, "category": "own", "libraries": list(libraries or []),
            "prompt": prompt.strip(), "criteria": criteria.strip()}
