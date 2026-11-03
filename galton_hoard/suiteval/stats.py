"""Aggregation of a suit evaluation: per arm with spread, paired arm comparisons, and the cases left out of the score.

A case used to build the candidate improvement (marked ``used_for_building`` or listed in ``candidate.built_from_cases``) is excluded from
the score of every arm: judging an improvement on the case it was tuned on measures memory, not skill. Its records stay visible and a warning
says so. Several runs of the same case and arm are averaged into one value per case before arms are summarised, so a case counts once."""

from __future__ import annotations

import math
from statistics import mean, median, stdev
from typing import Any, Iterable, Optional

from .. import messages as msgs

ARMS = ("generic", "current", "candidate")
PAIRS = (("current", "generic"), ("candidate", "current"), ("candidate", "generic"))
METRICS = (("total_time_s", "time_s"), ("correction_minutes", "correction_minutes"), ("defects", "defects"), ("tokens", "tokens"))
LOW_N = 3


def wilson(successes: float, n: int, z: float = 1.96) -> Optional[list[float]]:
    if n <= 0:
        return None
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def summarize(values: Iterable[Optional[float]]) -> dict[str, Any]:
    """n, mean, spread (sample standard deviation, min, max, median). Missing values are not counted; with no values everything is ``None``."""
    v = [float(x) for x in values if x is not None]
    if not v:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None, "median": None, "sum": None}
    return {"n": len(v), "mean": round(mean(v), 4), "sd": round(stdev(v), 4) if len(v) > 1 else None, "min": round(min(v), 4),
            "max": round(max(v), 4), "median": round(median(v), 4), "sum": round(sum(v), 4)}


def sign_test(wins: int, losses: int) -> Optional[float]:
    """Exact two-sided sign test on the cases where the arms differ."""
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return round(min(1.0, 2 * tail), 4)


def excluded_cases(definition: dict[str, Any]) -> dict[str, str]:
    out = {c["id"]: "used_for_building" for c in definition.get("cases", []) if c.get("used_for_building")}
    for cid in (definition.get("candidate") or {}).get("built_from_cases") or []:
        out.setdefault(cid, "used_for_building")
    return out


def collapse(records: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    """One value per (case, arm): the mean over its runs; ``success`` is the share of runs that passed, ``None`` while every run waits for a judge."""
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in records:
        groups.setdefault((r["case_id"], r["arm"]), []).append(r)
    out = {}
    for key, rows in groups.items():
        done = [1.0 if r["success"] else 0.0 for r in rows if r["success"] is not None]
        item: dict[str, Any] = {"runs": len(rows), "success": round(mean(done), 4) if done else None, "pending": len(rows) - len(done)}
        for src, name in METRICS:
            vals = [r[src] for r in rows if r.get(src) is not None]
            item[name] = round(mean(vals), 4) if vals else None
        out[key] = item
    return out


def aggregate(definition: dict[str, Any], records: list[dict[str, Any]]) -> dict[str, Any]:
    excluded = excluded_cases(definition)
    cases = definition.get("cases", [])
    by_key = collapse(records)
    scored_ids = [c["id"] for c in cases if c["id"] not in excluded]
    arms: dict[str, Any] = {}
    for arm in ARMS:
        rows = {cid: by_key[(cid, arm)] for cid in scored_ids if (cid, arm) in by_key}
        done = {cid: r for cid, r in rows.items() if r["success"] is not None}
        succ = sum(r["success"] for r in done.values())
        arms[arm] = {
            "cases": len(done), "pending": len(rows) - len(done), "missing": len(scored_ids) - len(rows),
            "success_rate": round(succ / len(done), 4) if done else None, "success_ci95": wilson(succ, len(done)),
            **{name: summarize(r[name] for r in done.values()) for _src, name in METRICS},
            "runs": sum(r["runs"] for r in rows.values()),
        }
    paired = {}
    for a, b in PAIRS:
        shared = [cid for cid in scored_ids if by_key.get((cid, a), {}).get("success") is not None and by_key.get((cid, b), {}).get("success") is not None]
        wins = sum(by_key[(c, a)]["success"] > by_key[(c, b)]["success"] for c in shared)
        losses = sum(by_key[(c, a)]["success"] < by_key[(c, b)]["success"] for c in shared)
        diff = {name: summarize((by_key[(c, a)][name] - by_key[(c, b)][name]) if by_key[(c, a)][name] is not None and by_key[(c, b)][name] is not None else None
                                for c in shared) for _src, name in METRICS}
        paired[f"{a}_vs_{b}"] = {"cases": len(shared), "better": wins, "worse": losses, "same": len(shared) - wins - losses,
                                 "success_diff": summarize(by_key[(c, a)]["success"] - by_key[(c, b)]["success"] for c in shared), "sign_test_p": sign_test(wins, losses),
                                 "metric_diff": diff}
    scored_any = [cid for cid in scored_ids if any(by_key.get((cid, arm), {}).get("success") is not None for arm in ARMS)]
    per_case = []
    for c in cases:
        per_case.append({"case": c["id"], "title": c.get("title", ""), "excluded": excluded.get(c["id"], ""),
                         "arms": {arm: by_key[(c["id"], arm)] for arm in ARMS if (c["id"], arm) in by_key}})
    return {"cases_total": len(cases), "cases_scored": len(scored_any), "excluded": [{"case": cid, "reason": why} for cid, why in excluded.items()],
            "aggregate": arms, "paired": paired, "per_case": per_case, "warnings": warnings(definition, records, arms, excluded, len(scored_any))}


def warnings(definition: dict[str, Any], records: list[dict[str, Any]], arms: dict[str, Any], excluded: dict[str, str], scored: int) -> list[msgs.CodedText]:
    out: list[msgs.CodedText] = []
    if excluded:
        ran = sorted({r["case_id"] for r in records if r["case_id"] in excluded and r["arm"] == "candidate"})
        out.append(msgs.text("se_warn_excluded", cases=", ".join(sorted(excluded))))
        if ran:
            out.append(msgs.text("se_warn_tuned", cases=", ".join(ran)))
    elif any(r["arm"] == "candidate" for r in records):
        out.append(msgs.text("se_warn_no_build_cases"))
    if scored == 0:
        out.append(msgs.text("se_warn_no_scored"))
    for arm in ARMS:
        a = arms[arm]
        if 0 < a["cases"] < LOW_N:
            out.append(msgs.text("se_warn_low_n", arm=arm, n=a["cases"]))
        if a["pending"]:
            out.append(msgs.text("se_warn_pending", arm=arm, n=a["pending"]))
    return out
