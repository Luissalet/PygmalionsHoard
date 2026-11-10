"""Which piece of the pipeline made a case fail.

A case goes through three pieces in order, and the first one that loses what the case needs is the cause:

* ``retriever``: the reference passage is nowhere in what was retrieved (cause ``not_retrieved``), or it was retrieved below the cut and
  the pipeline has no reranker, so the order is the retriever's own (cause ``cut_off``).
* ``ranking``: it was retrieved but is not among the passages handed to the generator. ``demoted`` when it was inside the cut in the
  retriever's own order and the reranker (or the pipeline's own filter) pushed it out; ``cut_off`` when it was below the cut and the
  reranker did not lift it.
* ``generator``: it had the passage and still answered wrong: ``abstained`` (said it could not answer), ``invented`` (statements the
  passages do not support) or ``incomplete`` (supported, but not the expected answer). On a case with no answer in the corpus the only
  failure is ``did_not_abstain``. An empty reply is ``no_answer``.

A pipeline that reports no passages (only its answer) cannot be split: its failures are ``unknown``. What the generator did when the
passage never reached it is kept as a note (``generator_note``), because it says whether the pipeline at least knows when it does not know.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from . import metrics as M

PIECES = ("retriever", "ranking", "generator", "unknown")
CAUSES = ("not_retrieved", "cut_off", "demoted", "abstained", "invented", "incomplete", "did_not_abstain", "no_answer", "no_trace", "error")


def answer_verdict(gen: dict[str, Any], *, faithful_min: float, coverage_min: float) -> tuple[bool, str]:
    """Is the answer right for an answerable case, and if not, how it is wrong."""
    if gen.get("empty"):
        return False, "no_answer"
    if gen.get("abstained"):
        return False, "abstained"
    faithful, covered = gen.get("faithfulness"), gen.get("coverage")
    if faithful is not None and faithful < faithful_min:
        return False, "invented"
    if covered is not None and covered < coverage_min:
        return False, "incomplete"
    return True, ""


def judge_case(*, answerable: bool, n_refs: int, traced: bool, found_retrieved: Sequence[Sequence[int]], found_ranked: Sequence[Sequence[int]],
               found_context: Sequence[Sequence[int]], cut: int, gen: dict[str, Any], require: str = "all", faithful_min: float = 0.6,
               coverage_min: float = 0.6, error: str = "", reranked: bool = False) -> dict[str, Any]:
    """The verdict of one case: ``{ok, piece, cause, retrieved_ok, context_ok, answer_ok, generator_note, rescued, demoted}``.

    ``cut`` is how many passages the generator received (the size of the context); ``found_*`` come from ``metrics.hits``."""
    out: dict[str, Any] = {"ok": False, "piece": "", "cause": "", "retrieved_ok": None, "context_ok": None, "answer_ok": None, "generator_note": "",
                           "rescued": False, "demoted": False}
    if error:
        return {**out, "piece": "unknown", "cause": "error"}
    if not answerable:
        ok = bool(gen.get("abstained"))
        return {**out, "ok": ok, "answer_ok": ok, "piece": "" if ok else "generator", "cause": "" if ok else ("no_answer" if gen.get("empty") else "did_not_abstain")}
    answer_ok, wrong = answer_verdict(gen, faithful_min=faithful_min, coverage_min=coverage_min)
    out["answer_ok"] = answer_ok
    if not traced or n_refs <= 0:
        # nothing to compare the passages with: only the answer can be judged
        covered = gen.get("coverage")
        ok = not gen.get("empty") and not gen.get("abstained") and (covered is None or covered >= coverage_min)
        return {**out, "ok": ok, "answer_ok": ok, "piece": "" if ok else "unknown", "cause": "" if ok else "no_trace"}
    retrieved_ok = M.satisfied(found_retrieved, n_refs, None, require)
    context_ok = M.satisfied(found_context, n_refs, None, require)
    before_cut_ok = M.satisfied(found_retrieved, n_refs, cut, require)
    out.update(retrieved_ok=retrieved_ok, context_ok=context_ok, rescued=context_ok and not before_cut_ok, demoted=before_cut_ok and not context_ok)
    if context_ok and answer_ok:
        return {**out, "ok": True}
    if not context_ok:
        if gen.get("abstained"):
            out["generator_note"] = "abstained_rightly"
        elif answer_ok or (gen.get("coverage") is not None and gen["coverage"] >= coverage_min):
            out["generator_note"] = "answered_without_passage"
        elif not gen.get("empty"):
            out["generator_note"] = "invented"
        if not retrieved_ok:
            return {**out, "piece": "retriever", "cause": "not_retrieved"}
        if before_cut_ok:
            return {**out, "piece": "ranking", "cause": "demoted"}
        return {**out, "piece": "ranking" if reranked else "retriever", "cause": "cut_off"}      # with no reranker the order is the retriever's own
    return {**out, "piece": "generator", "cause": wrong}


def summarise(verdicts: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Failures by piece and by cause, with the share of all failures each piece has."""
    failed = [v for v in verdicts if not v.get("ok")]
    by_piece = {p: sum(1 for v in failed if v.get("piece") == p) for p in PIECES}
    by_cause: dict[str, int] = {}
    for v in failed:
        by_cause[v.get("cause") or "error"] = by_cause.get(v.get("cause") or "error", 0) + 1
    total = len(failed)
    share = {p: (round(n / total, 4) if total else None) for p, n in by_piece.items()}
    main: Optional[str] = max((p for p in PIECES if by_piece[p]), key=lambda p: by_piece[p], default=None)
    return {"cases": len(verdicts), "passed": len(verdicts) - total, "failed": total, "by_piece": by_piece, "share": share, "by_cause": by_cause, "main": main,
            "generator_notes": {note: sum(1 for v in verdicts if v.get("generator_note") == note)
                                for note in ("abstained_rightly", "answered_without_passage", "invented") if any(v.get("generator_note") == note for v in verdicts)}}
