"""Decide between two arms of a suit evaluation from what a person cares about.

The order is fixed: first the deliverables that passed their acceptance checks (paired by case), then the minutes a person spent
correcting them, the defects and the total time. Tokens are reported next to the decision but never decide it: an arm that only saves
tokens is not adopted. Cases used to build the candidate are already out of the score (``stats.excluded_cases``), so the decision never
rests on them. With fewer than ``MIN_CASES`` cases compared the answer is ``insufficient``.

Decisions: ``adopt`` (the challenger is better where it matters), ``keep`` (it is not), ``tradeoff`` (better in one human measure and
worse in another: a person weighs it) and ``insufficient``."""

from __future__ import annotations

from typing import Any, Optional

from .. import messages as msgs

MIN_CASES = 3
# metric, absolute threshold, relative threshold (share of the incumbent's mean); a difference matters when it reaches the larger one
HUMAN = (("correction_minutes", 1.0, 0.10), ("defects", 0.5, 0.0), ("time_s", 1.0, 0.10))
TOKEN_REL = 0.10
QUESTIONS = (("candidate", "current"), ("current", "generic"))
DECISIONS = ("adopt", "keep", "tradeoff", "insufficient")


def _effect(diff: Optional[float], base: Optional[float], abs_th: float, rel_th: float) -> str:
    """``better`` / ``worse`` / ``same`` for a lower-is-better measure, ``unmeasured`` without paired values."""
    if diff is None:
        return "unmeasured"
    threshold = max(abs_th, rel_th * abs(base or 0.0))
    if abs(diff) < threshold:
        return "same"
    return "better" if diff < 0 else "worse"


def decide(score: dict[str, Any], challenger: str, incumbent: str) -> dict[str, Any]:
    """The decision for «should ``challenger`` replace ``incumbent``?» from an aggregated score (``stats.aggregate``)."""
    pair = score["paired"].get(f"{challenger}_vs_{incumbent}") or {}
    arms = score["aggregate"]
    n, wins, losses = pair.get("cases", 0), pair.get("better", 0), pair.get("worse", 0)
    basis: list[dict[str, Any]] = []
    reasons: list[msgs.CodedText] = []
    an, bn = msgs.text(f"arm_{challenger}"), msgs.text(f"arm_{incumbent}")
    for name, abs_th, rel_th in HUMAN:
        d = (pair.get("metric_diff") or {}).get(name) or {}
        diff = d.get("mean") if d.get("n") else None
        base = (arms.get(incumbent, {}).get(name) or {}).get("mean")
        basis.append({"metric": name, "diff": diff, "n": d.get("n", 0), "incumbent_mean": base,
                      "challenger_mean": (arms.get(challenger, {}).get(name) or {}).get("mean"), "effect": _effect(diff, base, abs_th, rel_th)})
    t = (pair.get("metric_diff") or {}).get("tokens") or {}
    t_diff = t.get("mean") if t.get("n") else None
    t_base = (arms.get(incumbent, {}).get("tokens") or {}).get("mean")
    tokens = {"diff": t_diff, "n": t.get("n", 0), "effect": _effect(t_diff, t_base, 1.0, TOKEN_REL), "decides": False}
    out = {"challenger": challenger, "incumbent": incumbent, "cases": n, "success": {"better": wins, "worse": losses, "same": n - wins - losses,
           "sign_test_p": pair.get("sign_test_p")}, "basis": basis, "tokens": tokens, "min_cases": MIN_CASES}
    if n < MIN_CASES:
        reasons.append(msgs.text("sv_insufficient", a=an, b=bn, n=n, min=MIN_CASES))
        return {**out, "decision": "insufficient", "reasons": reasons}
    better = [b["metric"] for b in basis if b["effect"] == "better"]
    worse = [b["metric"] for b in basis if b["effect"] == "worse"]
    for b in basis:
        if b["effect"] in ("better", "worse"):
            reasons.append(msgs.text(f"sv_{b['effect']}_{b['metric']}", a=an, b=bn, diff=round(abs(b["diff"]), 2), n=b["n"]))
    if wins != losses:
        key = "sv_success_better" if wins > losses else "sv_success_worse"
        reasons.insert(0, msgs.text(key, a=an, b=bn, wins=wins, losses=losses, n=n))
        decision = "adopt" if wins > losses else "keep"
        p = pair.get("sign_test_p")
        if p is not None and p > 0.05:
            reasons.append(msgs.text("sv_weak", n=n, p=p))
    else:
        reasons.insert(0, msgs.text("sv_success_same", a=an, b=bn, n=n))
        if better and not worse:
            decision = "adopt"
        elif worse and not better:
            decision = "keep"
        elif better and worse:
            decision = "tradeoff"
            reasons.append(msgs.text("sv_tradeoff"))
        else:
            decision = "keep"
            reasons.append(msgs.text("sv_tokens_only" if tokens["effect"] in ("better", "worse") else "sv_no_difference",
                                     diff=f"{t_diff:+.0f}" if t_diff is not None else "0"))
    if tokens["effect"] in ("better", "worse") and not any(r.key == "sv_tokens_only" for r in reasons):
        reasons.append(msgs.text("sv_tokens_note", diff=f"{t_diff:+.0f}"))
    return {**out, "decision": decision, "reasons": reasons}


def verdicts(score: dict[str, Any]) -> dict[str, Any]:
    """Both questions a suit evaluation answers: does the candidate beat the current suit, and does the suit beat plain tools?"""
    return {f"{a}_vs_{b}": decide(score, a, b) for a, b in QUESTIONS}
