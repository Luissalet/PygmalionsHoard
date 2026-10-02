"""Dataset operations. Each one takes a list of records and returns the kept records plus a count of what it dropped or changed.
Pure Python, no third-party code.

* exact duplicates: normalised text hash
* near duplicates: a bottom-k MinHash sketch over character 5-grams, candidates from an index of the smallest hashes, the pair
  confirmed by the estimated Jaccard similarity (default threshold 0.85)
* length and language filters (a stop-word detector for Spanish and English)
* personal data: scan, mask or drop
* train/evaluation split: seeded, deterministic
* statistics
"""

from __future__ import annotations

import math
import random
import re
import zlib
from typing import Any, Callable, Optional

from ..util import est_tokens, fold, sha256_text
from . import pii
from .formats import content_text, map_text, meta_of, record_text, status_of, with_meta

SKETCH_K = 128
INDEX_KEYS = 6
MAX_POSTINGS = 60

STOP_ES = frozenset("el la los las un una unos unas de del al y o u que en es se no por con para su sus lo le les mi tu como más pero muy ya si sí está están "
                    "ser son fue ha han hay este esta esto estos estas ese esa eso también cuando donde porque entre sobre desde hasta me te nos yo "
                    "usted ustedes ellos ellas nuestro vuestro hacer tiene tengo puede puedo qué cómo cuál".split())
STOP_EN = frozenset("the a an and or of to in is are was were be been being it its this that these those for on with as at by from not but "
                    "have has had do does did will would can could should may might you your we our they their he she his her i my me what "
                    "which who how when where why there here about into than then so if".split())
_WORD = re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ']+")


# ------------------------------------------------------------------ language
def detect_language(text: str) -> tuple[str, float]:
    """``("es" | "en" | "unknown", confidence)`` from the share of stop words and Spanish-only letters."""
    words = [w.lower() for w in _WORD.findall(text or "")]
    if len(words) < 4:
        return "unknown", 0.0
    es = sum(1 for w in words if w in STOP_ES)
    en = sum(1 for w in words if w in STOP_EN)
    es += 2 * len(re.findall(r"[ñ¿¡]", text)) + 0.3 * len(re.findall(r"[áéíóú]", text))
    total = es + en
    if total < 2 or total / len(words) < 0.08:
        return "unknown", 0.0
    lang = "es" if es >= en else "en"
    return lang, round(max(es, en) / total, 3)


# ------------------------------------------------------------------ duplicates
def normalise(text: str) -> str:
    return " ".join(fold(text).split())


def exact_key(rec: dict[str, Any]) -> str:
    return sha256_text(normalise(content_text(rec)))


def dedupe_exact(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    seen: set[str] = set()
    kept = []
    for rec in records:
        key = exact_key(rec)
        if key in seen:
            continue
        seen.add(key)
        kept.append(rec)
    return kept, len(records) - len(kept)


def shingle_hashes(text: str, n: int = 5) -> set[int]:
    norm = normalise(text)
    if not norm:
        return set()
    grams = {norm} if len(norm) <= n else {norm[i:i + n] for i in range(len(norm) - n + 1)}
    out = set()
    for g in grams:
        b = g.encode("utf-8")
        out.add((zlib.crc32(b) << 32) | zlib.crc32(b, 0x9E3779B9))
    return out


def sketch(text: str, k: int = SKETCH_K) -> list[int]:
    """The ``k`` smallest shingle hashes: a bottom-k MinHash sketch (the whole set when the text is short)."""
    return sorted(shingle_hashes(text))[:k]


def jaccard_estimate(a: list[int], b: list[int], k: int = SKETCH_K) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    union = sorted(sa | sb)[:k]
    both = sum(1 for h in union if h in sa and h in sb)
    return both / len(union)


def dedupe_near(records: list[dict[str, Any]], threshold: float = 0.85) -> tuple[list[dict[str, Any]], int]:
    """Drop every record that is at least ``threshold`` similar to one kept before it."""
    kept: list[dict[str, Any]] = []
    sketches: list[list[int]] = []
    index: dict[int, list[int]] = {}
    for rec in records:
        sk = sketch(content_text(rec))
        duplicate = False
        seen: set[int] = set()
        for key in sk[:INDEX_KEYS]:
            for other in index.get(key, ())[-MAX_POSTINGS:]:
                if other in seen:
                    continue
                seen.add(other)
                if jaccard_estimate(sk, sketches[other]) >= threshold:
                    duplicate = True
                    break
            if duplicate:
                break
        if duplicate:
            continue
        position = len(kept)
        kept.append(rec)
        sketches.append(sk)
        for key in sk[:INDEX_KEYS]:
            index.setdefault(key, []).append(position)
    return kept, len(records) - len(kept)


# ------------------------------------------------------------------ filters
def filter_length(records: list[dict[str, Any]], min_chars: int = 0, max_chars: int = 0) -> tuple[list[dict[str, Any]], int]:
    kept = []
    for rec in records:
        n = len(content_text(rec))
        if n < min_chars or (max_chars and n > max_chars):
            continue
        kept.append(rec)
    return kept, len(records) - len(kept)


def filter_language(records: list[dict[str, Any]], keep: list[str]) -> tuple[list[dict[str, Any]], int]:
    """Keep records in the wanted languages; text whose language cannot be told ("unknown") is kept."""
    wanted = {k.lower() for k in keep}
    kept = []
    for rec in records:
        lang, _ = detect_language(content_text(rec))
        if lang == "unknown" or lang in wanted:
            kept.append(rec)
    return kept, len(records) - len(kept)


def pii_scan(records: list[dict[str, Any]]) -> dict[str, Any]:
    totals: dict[str, int] = {}
    affected = 0
    for rec in records:
        found = pii.counts(record_text(rec))
        if found:
            affected += 1
            for kind, n in found.items():
                totals[kind] = totals.get(kind, 0) + n
    return {"records_with_pii": affected, "by_kind": totals}


def pii_apply(records: list[dict[str, Any]], mode: str = "mask") -> tuple[list[dict[str, Any]], int]:
    """``mask`` replaces identifiers by tokens; ``drop`` removes the records that contain any. Returns (records, changed)."""
    out, changed = [], 0
    for rec in records:
        if not pii.scan(record_text(rec)):
            out.append(rec)
            continue
        changed += 1
        if mode == "drop":
            continue
        out.append(map_text(rec, pii.mask))
    return out, changed


# ------------------------------------------------------------------ split
def make_split(usable: list[int], eval_pct: float = 5.0, min_eval: int = 20, seed: int = 42) -> dict[str, Any]:
    """Pick the evaluation indices out of ``usable`` (indices of the records that may be trained on).

    The size is ``eval_pct`` % of the records but at least ``min_eval``, and never more than half of them, so a small dataset
    still has something to train on. The choice depends only on the seed and the list: the same input gives the same split."""
    n = len(usable)
    size = 0 if n < 2 or eval_pct <= 0 and min_eval <= 0 else max(int(round(n * eval_pct / 100.0)), min_eval)
    size = min(size, n // 2)
    chosen = sorted(random.Random(seed).sample(usable, size)) if size else []
    return {"eval": chosen, "seed": seed, "eval_pct": eval_pct, "min_eval": min_eval}


def split_indices(usable: list[int], splits: dict[str, Any]) -> tuple[list[int], list[int]]:
    held = set(splits.get("eval") or [])
    return [i for i in usable if i not in held], [i for i in usable if i in held]


# ------------------------------------------------------------------ pipeline
def run_operations(records: list[dict[str, Any]], operations: list[dict[str, Any]],
                   defaults: Optional[dict[str, Any]] = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Apply the operations in order. Returns the records and a report ``[{op, before, after, changed}]``."""
    defaults = defaults or {}
    report = []
    for spec in operations:
        op = str(spec.get("op", "")).strip()
        before = len(records)
        changed = 0
        if op == "dedupe_exact":
            records, changed = dedupe_exact(records)
        elif op == "dedupe_near":
            records, changed = dedupe_near(records, float(spec.get("threshold", defaults.get("near_dup", 0.85))))
        elif op == "length":
            records, changed = filter_length(records, int(spec.get("min_chars", 0)), int(spec.get("max_chars", 0)))
        elif op == "language":
            records, changed = filter_language(records, list(spec.get("keep") or ["es", "en"]))
        elif op == "pii":
            records, changed = pii_apply(records, str(spec.get("mode", "mask")))
        else:
            raise ValueError(f"unknown operation {op!r}; use dedupe_exact, dedupe_near, length, language or pii")
        report.append({"op": op, "before": before, "after": len(records), "changed": changed})
    return records, report


# ------------------------------------------------------------------ statistics
HIST_EDGES = (100, 300, 1000, 3000, 10000)


def histogram(lengths: list[int]) -> list[dict[str, Any]]:
    labels = ["<100", "100-300", "300-1k", "1k-3k", "3k-10k", ">10k"]
    counts = [0] * len(labels)
    for n in lengths:
        index = sum(1 for edge in HIST_EDGES if n >= edge)
        counts[index] += 1
    return [{"bucket": label, "count": c} for label, c in zip(labels, counts)]


def stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    lengths, roles, langs = [], {}, {"es": 0, "en": 0, "unknown": 0}
    status = {"ok": 0, "pending": 0, "rejected": 0}
    synthetic = chars = turns = 0
    for rec in records:
        text = content_text(rec)
        lengths.append(len(text))
        chars += len(text)
        st = status_of(rec)
        status[st] += 1
        if meta_of(rec).get("synthetic"):
            synthetic += 1
        if "messages" in rec:
            turns += len(rec["messages"])
            for m in rec["messages"]:
                roles[m["role"]] = roles.get(m["role"], 0) + 1
        elif "prompt" in rec:
            roles["user"] = roles.get("user", 0) + 1
            roles["assistant"] = roles.get("assistant", 0) + 1
            turns += 2
        lang, _ = detect_language(text)
        langs[lang] += 1
    n = len(records)
    ordered = sorted(lengths)
    return {"records": n, "usable": status["ok"], "pending": status["pending"], "rejected": status["rejected"], "chars": chars,
            "tokens": est_tokens(chars), "synthetic": synthetic, "roles": roles, "avg_turns": round(turns / n, 2) if n and turns else 0,
            "languages": langs, "histogram": histogram(lengths), "median_chars": ordered[n // 2] if n else 0,
            "p95_chars": ordered[min(n - 1, int(n * 0.95))] if n else 0, "max_chars": ordered[-1] if n else 0}


def preview_text(rec: dict[str, Any], limit: int = 240) -> str:
    text = " ".join(content_text(rec).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
