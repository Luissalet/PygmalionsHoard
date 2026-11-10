"""Cases proposed by a model from the documents of a case set, so the set does not have to be written by hand.

For a fragment of a document the model writes questions a person could ask without having seen it, the short answer, and the sentence of
the fragment that supports it, copied word for word. The quote is checked against the fragment: when it is there it becomes the reference
passage of the case (short, so pipelines that cut the documents in other sizes still match it); when the model paraphrased it, the closest
sentence of the fragment is used, and when nothing is close the whole fragment is. Every generated case is ``proposed`` until a person
accepts it. Questions the corpus cannot answer can be asked for too; nothing can check that the rest of the corpus does not answer them,
which is one more reason a person reviews them.
"""

from __future__ import annotations

import random
import re
from typing import Any, Callable, Sequence

from ..checkers.textutil import extract_json, split_reasoning
from ..hoard_link.docs.chunking import chunk_text
from ..util import fold
from . import metrics as M

MIN_FRAGMENT_WORDS = 20
QUOTE_CLOSE = 0.8

ANSWERABLE_SYSTEM = (
    "You write evaluation questions for a document retrieval system. From the PASSAGE write {n} question(s) whose answer the passage states. "
    "Each question must stand alone: a person who never saw the passage could ask it, so name what it asks about and never say 'the passage', 'the text' or "
    "'the document'. Prefer facts that only this passage gives (a figure, a name, a date, a rule). For each question give the short answer and the one "
    "sentence of the passage that supports it, copied word for word. Write question and answer in the language of the passage. "
    'Reply ONLY with JSON {{"cases": [{{"question": "...", "answer": "...", "quote": "..."}}]}}.')
UNANSWERABLE_SYSTEM = (
    "You write evaluation questions for a document retrieval system. From the PASSAGE write {n} question(s) on the same subject that sound answerable "
    "but whose answer the passage does NOT give: a detail it never states (a date, a figure, a name, a reason). Not general knowledge either: only the "
    "missing document could answer. Each question must stand alone and never say 'the passage', 'the text' or 'the document'. Write in the language of "
    'the passage. Reply ONLY with JSON {{"cases": [{{"question": "..."}}]}}.')


def fragments(corpus: Sequence[dict[str, Any]], chunk_chars: int = 900) -> list[dict[str, Any]]:
    out = []
    for doc in corpus:
        for chunk in chunk_text(doc.get("text") or "", size=chunk_chars, overlap=0, min_tail=min(200, chunk_chars // 3)):
            text = chunk.text.strip()
            if len(M.content(text)) >= MIN_FRAGMENT_WORDS:
                out.append({"doc": doc.get("name") or doc.get("id") or "", "ordinal": chunk.ordinal, "text": text})
    return out


def pick(rows: Sequence[dict[str, Any]], n: int, seed: int = 0) -> list[dict[str, Any]]:
    """``n`` fragments spread over the documents: one from each in turn, in an order fixed by the seed."""
    rng = random.Random(seed)
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_doc.setdefault(row["doc"], []).append(row)
    for group in by_doc.values():
        rng.shuffle(group)
    docs = sorted(by_doc)
    rng.shuffle(docs)
    out: list[dict[str, Any]] = []
    while len(out) < n and any(by_doc[d] for d in docs):
        for d in docs:
            if by_doc[d] and len(out) < n:
                out.append(by_doc[d].pop())
    return out


def reference_for(quote: str, fragment: str) -> tuple[str, str]:
    """The reference passage of a generated case and how it was found: ``quote`` (verbatim), ``sentence`` (closest sentence) or ``fragment``."""
    quote = " ".join(str(quote or "").split())
    if quote and len(M.content(quote)) >= 3:
        # found whatever the accents, the case and the line breaks; what is kept is the fragment's own text, not the model's copy
        found = re.search(r"\s+".join(re.escape(word) for word in fold(quote).split()), fold(fragment))
        if found:
            return fragment[found.start():found.end()], "quote"
    best, score = "", 0.0
    for sentence in M.sentences(fragment):
        s = M.coverage(quote, sentence) if quote else 0.0
        if s > score:
            best, score = sentence, s
    if best and score >= QUOTE_CLOSE and len(M.content(best)) >= 3:
        return best, "sentence"
    return fragment, "fragment"


def _cases_of(reply: str) -> list[dict[str, Any]]:
    found = extract_json(split_reasoning(reply)[0], expect="object")
    if not found or not isinstance(found[0], dict):
        return []
    rows = found[0].get("cases")
    return [r for r in rows if isinstance(r, dict)] if isinstance(rows, list) else []


def propose(corpus: Sequence[dict[str, Any]], ask: Callable[[list[dict[str, str]]], str], *, n: int = 10, per_fragment: int = 1, unanswerable: int = 0,
            chunk_chars: int = 900, seed: int = 0, known: Sequence[str] = (), cancel: Callable[[], bool] = lambda: False) -> dict[str, Any]:
    """Ask the model for ``n`` answerable cases (and ``unanswerable`` ones). Returns ``{cases, fragments, asked, discarded}``; ``known`` are questions
    the set already has, which are not proposed again."""
    rows = fragments(corpus, chunk_chars)
    if rows and n > len(rows) * per_fragment:                              # a small corpus: more questions per fragment, up to three
        per_fragment = min(3, -(-n // len(rows)))
    seen = {M.norm(q) for q in known}
    cases: list[dict[str, Any]] = []
    discarded = {"duplicate": 0, "incomplete": 0, "unreadable": 0}
    asked = 0

    def run(chosen: Sequence[dict[str, Any]], want: int, answerable: bool) -> None:
        nonlocal asked
        made = 0
        for row in chosen:
            if made >= want or cancel():
                return
            system = (ANSWERABLE_SYSTEM if answerable else UNANSWERABLE_SYSTEM).format(n=min(per_fragment, want - made))
            reply = ask([{"role": "system", "content": system}, {"role": "user", "content": f"DOCUMENT: {row['doc']}\nPASSAGE:\n<<<\n{row['text']}\n>>>"}])
            asked += 1
            proposed = _cases_of(reply)
            if not proposed:
                discarded["unreadable"] += 1
                continue
            for item in proposed[:per_fragment]:
                question = " ".join(str(item.get("question") or "").split())
                answer = " ".join(str(item.get("answer") or "").split())
                if not question or (answerable and not answer):
                    discarded["incomplete"] += 1
                    continue
                if M.norm(question) in seen:
                    discarded["duplicate"] += 1
                    continue
                seen.add(M.norm(question))
                if answerable:
                    text, how = reference_for(str(item.get("quote") or ""), row["text"])
                    cases.append({"question": question[:1000], "expected": answer[:2000], "answerable": True, "passages": [{"doc": row["doc"], "text": text}],
                                  "notes": f"{row['doc']} · {how}"})
                else:
                    cases.append({"question": question[:1000], "expected": "", "answerable": False, "passages": [], "notes": f"{row['doc']} · unanswerable"})
                made += 1
                if made >= want:
                    break

    # more fragments than cases are picked, so a fragment the model cannot use does not leave the set short
    run(pick(rows, max(n, 1) * 2, seed), n, True)
    if unanswerable > 0:
        run(pick(rows, unanswerable * 2, seed + 1), unanswerable, False)
    return {"cases": cases, "fragments": len(rows), "asked": asked, "discarded": discarded}
