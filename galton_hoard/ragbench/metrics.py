"""Measures of a RAG pipeline, one piece at a time. Everything here is arithmetic on text: no model, no network.

* Retrieval: which retrieved passages are the reference passages of a case (``hits``), and from them recall@k, MRR and nDCG@k.
* Reranking: the same measures before and after the reranker, and whether it rescued or demoted the passage the case needs.
* Generation without a judge: how much of the answer the passages support (``support``), how much of the expected answer it contains
  (``answer_coverage``) and whether it declines to answer (``abstains``). A judge model replaces these when one is available (``judge.py``).

A retrieved passage is the reference passage when they share an id, or when the reference text is inside it, or when it holds most of the
reference's content words (``threshold``), so the same case set serves pipelines that cut their documents in different sizes. A reference
with no text but a document name matches any passage of that document.
"""

from __future__ import annotations

import math
import re
from typing import Any, Optional, Sequence

from ..util import fold

STOP = frozenset("""a an the of and or to in on for with without by at from is are was were be been it its this that these those as but not no
de del la las el los un una unos unas y o u e en con sin por para al lo se su sus que es son fue era ha han como mas pero ni si ya le les me mi te
tu nos he hay muy tambien entre sobre cuando donde cual cuales quien cuanto cuantos cuanta cuantas what which who when where how does do did""".split())
_WORD = re.compile(r"\w+", re.UNICODE)
_SENTENCE = re.compile(r"(?<=[.!?…])\s+|\n+")
MATCH_THRESHOLD = 0.7
SUPPORT_THRESHOLD = 0.6
SHORT_REFERENCE = 5                 # a reference with fewer content words than this must be contained whole

_ABSTAIN = re.compile(
    r"\bno lo se\b|\bno se (?:menciona|indica|especifica|dice|encuentra|proporciona|sabe|puede (?:saber|determinar|responder))\b"
    r"|\bno (?:aparece|figura|consta)\b|\bno (?:hay|tengo|dispongo de|cuento con|existe) (?:la |esa |ninguna |suficiente )?(?:informacion|datos|respuesta|mencion)\b"
    r"|\bno puedo (?:responder|contestar|determinar|saberlo|decirlo)\b|\bno (?:lo |la )?(?:dicen|dice|mencionan|menciona|indican|indica|contienen|contiene|recogen|recoge)\b"
    r"|\bi (?:do not|don't|cannot|can't|can not) (?:know|answer|tell|determine|find)\b|\b(?:is|are|was) not (?:mentioned|stated|provided|specified|available|given|found)\b"
    r"|\bno (?:information|mention|answer)\b|\b(?:does|do) not (?:say|mention|state|contain|provide|specify|include)\b|\bunable to (?:answer|determine)\b"
    r"|\binsufficient (?:information|context)\b|\bnot enough (?:information|context)\b")


def norm(text: Any) -> str:
    """Accents and case removed, every run of whitespace as one space."""
    return re.sub(r"\s+", " ", fold(str(text or "")).casefold()).strip()


def words(text: Any) -> list[str]:
    return _WORD.findall(fold(str(text or "")).casefold())


def _stem(token: str) -> str:
    return token[:-1] if len(token) > 3 and token.endswith("s") and not token.endswith("ss") else token


def content(text: Any) -> list[str]:
    """The words that carry meaning: stop words of Spanish and English out, a plural ``s`` dropped, numbers kept."""
    return [_stem(w) for w in words(text) if w not in STOP and (len(w) > 1 or w.isdigit())]


def coverage(reference: Any, text: Any) -> float:
    """Share of the content words of ``reference`` that appear in ``text`` (0 when the reference has none)."""
    ref = set(content(reference))
    if not ref:
        return 0.0
    return len(ref & set(content(text))) / len(ref)


def doc_key(name: Any) -> str:
    """A document name without folder or extension, folded: ``D:\\docs\\Manual.md`` and ``manual`` are the same document."""
    base = re.split(r"[\\/]", str(name or "").strip())[-1]
    return norm(re.sub(r"\.[A-Za-z0-9]{1,5}$", "", base))


def passage_matches(ref: dict[str, Any], passage: dict[str, Any], threshold: float = MATCH_THRESHOLD) -> bool:
    rid, pid = str(ref.get("id") or "").strip(), str(passage.get("id") or "").strip()
    if rid and pid and rid == pid:
        return True
    rtext, ptext = norm(ref.get("text")), norm(passage.get("text"))
    if rtext:
        if not ptext:
            return False
        if rtext in ptext:
            return True
        if len(set(content(rtext))) < SHORT_REFERENCE:
            return False
        return coverage(rtext, ptext) >= threshold
    rdoc = doc_key(ref.get("doc"))
    return bool(rdoc) and rdoc == doc_key(passage.get("doc"))


def hits(refs: Sequence[dict[str, Any]], passages: Sequence[dict[str, Any]], threshold: float = MATCH_THRESHOLD) -> list[list[int]]:
    """For every passage, in order, the indices of the reference passages it is."""
    return [[i for i, ref in enumerate(refs) if passage_matches(ref, p, threshold)] for p in passages]


def first_rank(found: Sequence[Sequence[int]]) -> Optional[int]:
    """1-based position of the first passage that is a reference passage."""
    return next((i for i, h in enumerate(found, 1) if h), None)


def recall_at(found: Sequence[Sequence[int]], n_refs: int, k: int) -> Optional[float]:
    if n_refs <= 0:
        return None
    seen = {i for h in found[:max(0, k)] for i in h}
    return len(seen) / n_refs


def mrr(found: Sequence[Sequence[int]]) -> float:
    rank = first_rank(found)
    return 1.0 / rank if rank else 0.0


def ndcg_at(found: Sequence[Sequence[int]], n_refs: int, k: int) -> Optional[float]:
    """Binary gains: a passage counts when it brings a reference passage no earlier passage brought (a repeated one adds nothing)."""
    if n_refs <= 0 or k <= 0:
        return None
    seen: set[int] = set()
    dcg = 0.0
    for position, h in enumerate(found[:k]):
        new = set(h) - seen
        if new:
            dcg += 1.0 / math.log2(position + 2)
            seen |= new
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, n_refs)))
    return dcg / ideal if ideal else None


def satisfied(found: Sequence[Sequence[int]], n_refs: int, k: Optional[int] = None, require: str = "all") -> bool:
    """Do the first ``k`` passages (all of them when ``k`` is None) bring what the case needs: every reference passage, or any of them."""
    if n_refs <= 0:
        return False
    top = found if k is None else found[:max(0, k)]
    seen = {i for h in top for i in h}
    return len(seen) >= n_refs if require != "any" else bool(seen)


def retrieval_metrics(found: Sequence[Sequence[int]], n_refs: int, ks: Sequence[int]) -> dict[str, Any]:
    rank = first_rank(found)
    return {"first_rank": rank, "mrr": round(mrr(found), 4),
            "recall": {str(k): _round(recall_at(found, n_refs, k)) for k in ks},
            "ndcg": {str(k): _round(ndcg_at(found, n_refs, k)) for k in ks}}


def _round(value: Optional[float]) -> Optional[float]:
    return None if value is None else round(value, 4)


# ------------------------------------------------------------------------------------------------ generation, without a judge
def abstains(answer: Any) -> bool:
    """Does the answer decline (it says it does not know or that the passages do not say)? A long answer that also answers is not an abstention."""
    text = norm(answer)
    if not text:
        return False
    return bool(_ABSTAIN.search(text)) and len(text.split()) <= 60


def sentences(text: Any) -> list[str]:
    return [s.strip() for s in _SENTENCE.split(str(text or "")) if s.strip()]


def support(answer: Any, passages: Sequence[Any]) -> Optional[float]:
    """Share of the answer's statements whose content words are in the passages. None for an empty answer; 1 for an answer that states nothing
    (an abstention). A statement with a number the passages do not have is not supported."""
    if not str(answer or "").strip():
        return None
    pool: set[str] = set()
    for p in passages:
        pool |= set(content(p.get("text") if isinstance(p, dict) else p))
    claims = [s for s in sentences(answer) if len(set(content(s))) >= 2 and not abstains(s)]
    if not claims:
        return 1.0
    good = 0
    for claim in claims:
        tokens = set(content(claim))
        numbers = {t for t in tokens if t.isdigit()}
        if numbers - pool:
            continue
        if len(tokens & pool) / len(tokens) >= SUPPORT_THRESHOLD:
            good += 1
    return good / len(claims)


def answer_coverage(answer: Any, expected: Any) -> Optional[float]:
    """Share of the expected answer's content words that the answer has. None when there is no expected answer."""
    if not set(content(expected)):
        return None
    return coverage(expected, answer)


def mean(values: Sequence[Optional[float]]) -> Optional[float]:
    data = [float(v) for v in values if v is not None]
    return round(sum(data) / len(data), 4) if data else None
