"""Personal identifiers in training text: Spanish ID numbers, IBAN, card numbers, phone numbers and e-mail addresses.

The detection is the shared one of Hoard Link (``hoard_link.idcheck.scan_pii``): a value counts only when its checksum is right (DNI and NIE
letters, the CIF control character, IBAN mod 97 with the country's length, Luhn for cards), so an invoice number is not mistaken for a card. This
module keeps the app's wire shape (lowercase kinds, ``{kind, start, end, text}``) and its mask tokens: ``mask`` replaces each hit with a neutral
token the model can learn to ignore.
"""

from __future__ import annotations

from typing import Any

from ..hoard_link import idcheck

#: The shared kind to the one the app has always reported (the interface and the saved statistics use them).
KIND = {"DNI": "dni", "NIE": "nie", "CIF": "cif", "IBAN": "iban", "CARD": "card", "EMAIL": "email", "PHONE_ES": "phone"}
TOKENS = {kind: f"<{kind.upper()}>" for kind in KIND.values()}


def scan(text: str) -> list[dict[str, Any]]:
    """Every hit as ``{kind, start, end, text}``, sorted by position, without overlaps."""
    return [{"kind": KIND[h.kind], "start": h.start, "end": h.end, "text": h.value} for h in idcheck.scan_pii(text)]


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
