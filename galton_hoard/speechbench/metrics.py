"""Word and character error rates against an independent reference, with the alignment counts behind them.

Normalisation before comparing: lower case, punctuation out, whitespace collapsed; ``accents=False`` also folds accents
(«canción» = «cancion»). Numbers are compared as written (a reference «18» and a hypothesis «dieciocho» count as an error)."""

from __future__ import annotations

import re
import unicodedata
from typing import Any

_PUNCT = re.compile(r"[^\w\s']|_", re.UNICODE)
_SPACE = re.compile(r"\s+")


def normalise(text: str, *, accents: bool = True) -> str:
    text = unicodedata.normalize("NFC", str(text or "")).lower().replace("’", "'")
    text = _PUNCT.sub(" ", text)
    if not accents:
        text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return _SPACE.sub(" ", text).strip()


def align(ref: list[str], hyp: list[str]) -> dict[str, int]:
    """Levenshtein alignment counts: substitutions, deletions (in the reference, missing) and insertions (extra)."""
    cols = len(hyp) + 1
    prev = [(j, 0, 0, j) for j in range(cols)]  # (cost, subs, dels, ins) per cell of the previous row
    for i in range(1, len(ref) + 1):
        cur = [(i, 0, i, 0)]
        for j in range(1, cols):
            if ref[i - 1] == hyp[j - 1]:
                cur.append(prev[j - 1])
                continue
            sub, dele, ins = prev[j - 1], prev[j], cur[j - 1]
            best = min((sub[0], 0), (dele[0], 1), (ins[0], 2))[1]
            if best == 0:
                cur.append((sub[0] + 1, sub[1] + 1, sub[2], sub[3]))
            elif best == 1:
                cur.append((dele[0] + 1, dele[1], dele[2] + 1, dele[3]))
            else:
                cur.append((ins[0] + 1, ins[1], ins[2], ins[3] + 1))
        prev = cur
    cost, subs, dels, ins = prev[-1]
    return {"errors": cost, "substitutions": subs, "deletions": dels, "insertions": ins, "reference": len(ref)}


def score(reference: str, hypothesis: str, *, accents: bool = True) -> dict[str, Any]:
    """WER and CER of one hypothesis (None when the reference is empty)."""
    ref, hyp = normalise(reference, accents=accents), normalise(hypothesis, accents=accents)
    words = align(ref.split(), hyp.split())
    chars = align(list(ref.replace(" ", "")), list(hyp.replace(" ", "")))
    return {"wer": round(words["errors"] / words["reference"], 4) if words["reference"] else None,
            "cer": round(chars["errors"] / chars["reference"], 4) if chars["reference"] else None,
            "words": words, "chars": {k: chars[k] for k in ("errors", "reference")},
            "reference_norm": ref, "hypothesis_norm": hyp}


def corpus_rates(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Corpus-level WER/CER: total errors over total reference words (not the mean of per-file rates)."""
    scored = [i for i in items if i.get("score") and i["score"].get("wer") is not None]
    words = sum(i["score"]["words"]["reference"] for i in scored)
    chars = sum(i["score"]["chars"]["reference"] for i in scored)
    out: dict[str, Any] = {"files": len(scored), "reference_words": words,
                           "wer": round(sum(i["score"]["words"]["errors"] for i in scored) / words, 4) if words else None,
                           "cer": round(sum(i["score"]["chars"]["errors"] for i in scored) / chars, 4) if chars else None}
    for key in ("substitutions", "deletions", "insertions"):
        out[key] = sum(i["score"]["words"][key] for i in scored)
    return out
