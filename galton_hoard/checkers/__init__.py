"""Checkers: pure functions ``(spec, output, context) -> {score, passed, detail, unavailable?}``.

``run_checker`` dispatches on ``spec["type"]`` and never raises: a checker that crashes returns a failed verdict that says so.
``all`` and ``any`` combine other checkers (a RAG case checks the citations and the fact in one go).
"""

from __future__ import annotations

from typing import Any, Callable

from .basic import check_choice, check_contains, check_exact, check_none, check_regex
from .code import check_python_tests
from .constraints import check_constraints
from .ifmt import check_ifmt
from .judge import check_family, check_judge
from .jsoncheck import check_json, check_tool_call
from .mathcheck import check_math
from .numbers import check_number
from .rag import check_citations, check_needle
from .ifmt import REPLY_PARSERS
from .ragtriad import check_answer_relevance, check_faithfulness, check_similar, parse_claims_reply
from .types import CheckContext, ModelOutput, bad_spec, verdict

#: The faithfulness checker sends its own messages to the judge and reads a claim list back.
REPLY_PARSERS.setdefault("claims", parse_claims_reply)

Checker = Callable[[dict[str, Any], ModelOutput, CheckContext], dict[str, Any]]

CHECKERS: dict[str, Checker] = {
    "exact": check_exact,
    "contains": check_contains,
    "regex": check_regex,
    "choice": check_choice,
    "number": check_number,
    "math_equiv": check_math,
    "json": check_json,
    "tool_call": check_tool_call,
    "python_tests": check_python_tests,
    "constraints": check_constraints,
    "needle": check_needle,
    "citations": check_citations,
    "judge": check_judge,
    "family": check_family,
    "ifmt": check_ifmt,
    "faithfulness": check_faithfulness,
    "answer_relevance": check_answer_relevance,
    "similar": check_similar,
    "none": check_none,
}
COMBINATORS = ("all", "any")
#: Checkers whose verdict depends on a model or another service, so the reference answer of a case cannot be verified offline.
NON_DETERMINISTIC = frozenset({"judge", "family", "ifmt", "faithfulness", "answer_relevance", "none"})
#: Checkers that ask the judge model (see ``judging.has_judge``).
JUDGED = frozenset({"judge", "faithfulness", "answer_relevance"})

#: One line per checker type, for the catalogue (``checkers_list``) and the case editor.
CHECKER_HELP: dict[str, str] = {
    "exact": "the answer equals `expected` (normalised)",
    "contains": "the answer contains `all` / `any` of the strings and none of `none`",
    "regex": "the answer matches `pattern` / `patterns` (`forbid` to refuse a match)",
    "choice": "the chosen option equals `answer` among `options`",
    "number": "the number in the answer is `expected` within `tolerance` / `abs` / `rel`",
    "math_equiv": "the expression is mathematically equal to `expected`",
    "json": "the JSON in the answer matches `expected` or follows `schema` / `rules`",
    "tool_call": "the model made the `expected` tool calls with matching arguments",
    "python_tests": "the code passes the hidden `tests` (calling `entry`)",
    "constraints": "format constraints (length, words, case, bullet count…)",
    "needle": "a planted value (`needle`/`needles`) appears in the answer",
    "citations": "the answer cites the `correct` passages ([p3])",
    "judge": "the judge model grades the answer against `rubric` (0-10)",
    "family": "another Hoard app verifies it (`app`, `tool`, `arguments`)",
    "ifmt": "IFMTBench translation obeying glossary, style and format",
    "faithfulness": "every claim of the answer is supported by `context` (judge, claim by claim; hallucination check)",
    "answer_relevance": "the answer addresses the question (judge, 0-10), not whether it is true",
    "similar": "close to `reference` without a model: `method` rouge_l | token_f1 | chars, `threshold`",
    "none": "not scored (the answer is only stored)",
    "all": "every check in `checks` passes (weighted score)",
    "any": "at least one check in `checks` passes",
}


def checker_types() -> list[str]:
    return [*CHECKERS, *COMBINATORS]


def _combine(kind: str, spec: dict[str, Any], output: ModelOutput, ctx: CheckContext) -> dict[str, Any]:
    checks = spec.get("checks")
    if not isinstance(checks, list) or not checks:
        return bad_spec(f"{kind} needs a non-empty `checks` list")
    weights = spec.get("weights") or [1.0] * len(checks)
    if len(weights) != len(checks):
        return bad_spec(f"{kind}: `weights` must have one entry per check")
    parts = [(run_checker(c, output, ctx), float(w)) for c, w in zip(checks, weights)]
    usable = [(r, w) for r, w in parts if not r.get("unavailable")]
    detail = {"checks": [r for r, _ in parts]}
    if not usable:
        return {**verdict(0.0, False, detail, unavailable=True)}
    if kind == "all":
        total = sum(w for _, w in usable) or 1.0
        score = sum(r["score"] * w for r, w in usable) / total
        return verdict(score, all(r["passed"] for r, _ in usable), detail)
    best = max(usable, key=lambda rw: rw[0]["score"])[0]
    return verdict(best["score"], any(r["passed"] for r, _ in usable), detail)


def run_checker(spec: dict[str, Any], output: ModelOutput, ctx: CheckContext | None = None) -> dict[str, Any]:
    """Check ``output`` against the checker ``spec``. Never raises."""
    ctx = ctx or CheckContext()
    kind = (spec or {}).get("type")
    try:
        if kind in COMBINATORS:
            return _combine(kind, spec, output, ctx)
        func = CHECKERS.get(kind or "")
        if func is None:
            return bad_spec(f"unknown checker type {kind!r}")
        return func(spec, output, ctx)
    except Exception as exc:  # noqa: BLE001 — a bug in a checker must not take a whole run down
        return verdict(0.0, False, {"error": f"checker {kind} crashed: {type(exc).__name__}: {exc}"})


def is_deterministic(spec: dict[str, Any]) -> bool:
    """Can the reference answer of a case with this checker be verified without a model or another app?"""
    kind = (spec or {}).get("type")
    if kind in COMBINATORS:
        return all(is_deterministic(c) for c in spec.get("checks", []))
    return kind not in NON_DETERMINISTIC


__all__ = ["CHECKERS", "CHECKER_HELP", "COMBINATORS", "JUDGED", "CheckContext", "ModelOutput", "checker_types", "is_deterministic", "run_checker"]
