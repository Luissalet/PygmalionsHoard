"""Personal identifiers in training text: Spanish ID numbers, IBAN, card numbers, phone numbers and e-mail addresses.

Same approach as the other Hoard apps: patterns plus a checksum where there is one (IBAN mod 97, Luhn for cards), so an invoice
number is not mistaken for a card. ``mask`` replaces each hit with a neutral token the model can learn to ignore.
"""

from __future__ import annotations

import re
from typing import Any

DNI = re.compile(r"\b\d{8}[A-HJ-NP-TV-Z]\b")
NIE = re.compile(r"\b[XYZ]\d{7}[A-Z]\b")
IBAN = re.compile(r"\b[A-Z]{2}\d{2}(?:[ \-]?[A-Z0-9]{4}){3,7}(?:[ \-]?[A-Z0-9]{1,4})?\b")
CARD = re.compile(r"(?<![\w])(?:\d[ \-]?){12,18}\d(?![\w])")
EMAIL = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")
PHONE_LABELLED = re.compile(r"(?i)(tel(?:[eé]fono)?\.?|tfno\.?|m[oó]vil|phone|mobile|contacto|whatsapp)(\s*[:.]?\s*)((?:\+|00)?[\d][\d .\-]{7,16}\d)")
PHONE_PREFIXED = re.compile(r"(?<![\w+])(?:\+34|0034)[ .\-]?[6-9]\d{2}[ .\-]?\d{3}[ .\-]?\d{3}\b")
PHONE_SPANISH = re.compile(r"(?<![\w+\-])[6-9]\d{2}[ .\-]?\d{3}[ .\-]?\d{3}(?![\w\-])")

TOKENS = {"dni": "<DNI>", "nie": "<NIE>", "iban": "<IBAN>", "card": "<CARD>", "email": "<EMAIL>", "phone": "<PHONE>"}


def luhn(digits: str) -> bool:
    total, flip = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if flip:
            d *= 2
            if d > 9:
                d -= 9
        total += d
        flip = not flip
    return total % 10 == 0


def iban_ok(raw: str) -> bool:
    iban = re.sub(r"[ \-]", "", raw).upper()
    if not 15 <= len(iban) <= 34:
        return False
    moved = iban[4:] + iban[:4]
    try:
        number = int("".join(str(int(c, 36)) for c in moved))
    except ValueError:
        return False
    return number % 97 == 1


def scan(text: str) -> list[dict[str, Any]]:
    """Every hit as ``{kind, start, end, text}``, sorted by position, without overlaps."""
    hits: list[dict[str, Any]] = []

    def add(kind: str, m: re.Match, group: int = 0) -> None:
        hits.append({"kind": kind, "start": m.start(group), "end": m.end(group), "text": m.group(group)})

    for m in IBAN.finditer(text):
        if iban_ok(m.group(0)):
            add("iban", m)
    for m in DNI.finditer(text):
        add("dni", m)
    for m in NIE.finditer(text):
        add("nie", m)
    for m in EMAIL.finditer(text):
        add("email", m)
    for m in CARD.finditer(text):
        digits = re.sub(r"\D", "", m.group(0))
        if 13 <= len(digits) <= 19 and luhn(digits):
            add("card", m)
    for m in PHONE_LABELLED.finditer(text):
        add("phone", m, 3)
    for m in PHONE_PREFIXED.finditer(text):
        add("phone", m)
    for m in PHONE_SPANISH.finditer(text):
        add("phone", m)
    hits.sort(key=lambda h: (h["start"], -(h["end"] - h["start"])))
    clean, end = [], -1
    for h in hits:
        if h["start"] >= end:
            clean.append(h)
            end = h["end"]
    return clean


def mask(text: str) -> str:
    """Replace each hit with its token. Idempotent: the tokens match no pattern."""
    hits = scan(text)
    if not hits:
        return text
    out, cursor = [], 0
    for h in hits:
        out.append(text[cursor:h["start"]])
        out.append(TOKENS[h["kind"]])
        cursor = h["end"]
    out.append(text[cursor:])
    return "".join(out)


def counts(text: str) -> dict[str, int]:
    found: dict[str, int] = {}
    for h in scan(text):
        found[h["kind"]] = found.get(h["kind"], 0) + 1
    return found
