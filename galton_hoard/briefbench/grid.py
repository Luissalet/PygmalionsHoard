"""The comparison grid of one brief: one cell per model with its result (the render of a page, the start of a text, or why there is none) and, under it, what it cost.

One PNG so that an assistant or a person can see every answer at once. The screen builds its own grid from the same data; this file is what the tools hand out.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any, Optional

from .shot import THUMB

LABEL_H = 64
PAD = 8
BACKGROUND, CELL, INK, MUTED, BAD = (18, 18, 20), (30, 30, 34), (235, 235, 238), (150, 150, 160), (224, 108, 117)


def _font(size: int):
    from PIL import ImageFont
    for name in ("arial.ttf", "DejaVuSans.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def figures(item: dict[str, Any], currency: str = "USD") -> str:
    """The numbers under a cell: time, tokens, cost, score."""
    parts = []
    if item.get("latency_ms") is not None:
        parts.append(f"{item['latency_ms'] / 1000:.1f} s")
    if item.get("tokens_out") is not None:
        parts.append(f"{item.get('tokens_in') or 0}→{item['tokens_out']} tok")
    parts.append(f"{item['cost']:.4f} {currency}" if item.get("cost") is not None else "cost ?")
    if item.get("mean") is not None:
        parts.append(f"score {item['mean']:.1f}")
    return " · ".join(parts)


def compose(items: list[dict[str, Any]], title: str, out: Path, *, currency: str = "USD", columns: int = 3) -> Path:
    """Write the grid of ``items`` (the rows of a brief, one per model) to ``out``."""
    from PIL import Image, ImageDraw
    n = max(1, len(items))
    cols = max(1, min(columns, n))
    rows = (n + cols - 1) // cols
    head = 36
    cw, ch = THUMB[0] + 2 * PAD, THUMB[1] + LABEL_H + 2 * PAD
    sheet = Image.new("RGB", (cols * cw, head + rows * ch), BACKGROUND)
    draw = ImageDraw.Draw(sheet)
    draw.text((PAD + 4, 8), title[:90], fill=INK, font=_font(18))
    small, tiny = _font(14), _font(12)
    for i, item in enumerate(items):
        x, y = (i % cols) * cw, head + (i // cols) * ch
        draw.rectangle([x + 2, y + 2, x + cw - 3, y + ch - 3], fill=CELL)
        box = (x + PAD, y + PAD)
        shot: Optional[Path] = Path(item["folder"]) / "shot.png" if item.get("folder") else None
        if shot is not None and shot.is_file():
            with Image.open(shot) as im:
                sheet.paste(im.convert("RGB").resize(THUMB), box)
        else:
            draw.rectangle([box[0], box[1], box[0] + THUMB[0], box[1] + THUMB[1]], fill=(12, 12, 14))
            body = ""
            answer = Path(item["folder"]) / "answer.txt" if item.get("folder") else None
            if answer is not None and answer.is_file():
                body = answer.read_text(encoding="utf-8", errors="replace")
            shown = body if body else (item.get("reason") or item.get("outcome") or "no result") + (": " + item["detail"] if item.get("detail") else "")
            lines = textwrap.wrap(" ".join(str(shown).split()), 62)[:15]
            draw.multiline_text((box[0] + 8, box[1] + 8), "\n".join(lines), fill=INK if body else BAD, font=tiny, spacing=2)
        ty = y + PAD + THUMB[1] + 6
        draw.text((x + PAD, ty), str(item.get("model_name") or item.get("slot"))[:46], fill=INK, font=small)
        draw.text((x + PAD, ty + 20), figures(item, currency)[:70], fill=MUTED, font=tiny)
        flag = "" if item.get("outcome") == "done" else (item.get("reason") or item.get("outcome") or "")
        if flag:
            draw.text((x + PAD, ty + 36), str(flag)[:60], fill=BAD, font=tiny)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, "PNG", optimize=True)
    return out
