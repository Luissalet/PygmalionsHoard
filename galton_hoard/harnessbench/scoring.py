"""Turn trials into the model x harness matrix: one cell per pair, margins, and a short automatic conclusion.

A trial whose verifier never ran (``outcome == "none"``: timeout, network, server error) is not a zero. It is left out of every rate and every
denominator and counted apart in the cell (``none_n`` and ``none_reasons``); a cell with nothing left shows no number at all. The conclusion compares
two spreads of pass@1 over the cells that have a number: how much a model moves when only the harness changes, and how much the models differ
under one harness.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Optional

from ..langbench.judge import INCONSISTENT_BELOW
from ..rulebench.scoring import pass_at_k, pass_hat_k
from ..stats import wilson
from .store import column_id

WEAK_BELOW = 20             # fewer valid trials than this per cell: the conclusion says the evidence is weak
SENSITIVE_POINTS = 15.0     # a model whose best and worst harness differ by this many points depends on the harness
SIMILAR_POINTS = 2.0        # the two spreads differ by less than this: no clear difference
LOW_PASS = 0.5              # a model whose best harness stays under this fails whatever the harness


def _mean(values: Iterable[Optional[float]], digits: int = 3) -> Optional[float]:
    items = [float(v) for v in values if v is not None]
    return round(sum(items) / len(items), digits) if items else None


def _round(x: Optional[float], digits: int = 4) -> Optional[float]:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(x, digits)


def _language_metrics(valid: list[dict[str, Any]]) -> dict[str, Any]:
    """The extra figures of a language-bench cell: consistency of the replies, empty searches and tool arguments in another language than the data."""
    mine = [t for t in valid if t.get("lang")]
    judged = [t["lang_consistency"] for t in mine if t.get("lang_consistency") is not None]
    searches = sum(t.get("search_calls") or 0 for t in mine)
    empties = sum(t.get("empty_searches") or 0 for t in mine)
    args = sum(t.get("arg_calls") or 0 for t in mine)
    wrong = sum(t.get("arg_mismatch") or 0 for t in mine)
    return {"lang": mine[0]["lang"], "lang_consistency": _mean(judged, 4), "lang_judged": len(judged),
            "inconsistent_rate": _round(sum(1 for x in judged if x < INCONSISTENT_BELOW) / len(judged)) if judged else None,
            "search_calls_mean": _mean((t.get("search_calls") or 0 for t in mine), 2), "empty_searches": empties, "search_calls": searches,
            "empty_search_rate": _round(empties / searches) if searches else None,
            "empty_search_trial_rate": _round(sum(1 for t in mine if (t.get("empty_searches") or 0) > 0) / len(mine)),
            "arg_calls": args, "arg_mismatch": wrong, "arg_mismatch_rate": _round(wrong / args) if args else None}


def cell_stats(trials: list[dict[str, Any]], k: int = 1) -> dict[str, Any]:
    """Every metric of one (model, harness) cell. Rates use the trials that ran (not ``none``) as denominator."""
    valid = [t for t in trials if t["outcome"] != "none"]
    nones = [t for t in trials if t["outcome"] == "none"]
    n = len(valid)
    passed = sum(1 for t in valid if t["outcome"] == "pass")
    counts = Counter(t["outcome"] for t in trials)
    out: dict[str, Any] = {"trials": len(trials), "valid": n, "none_n": len(nones), "none_reasons": dict(Counter((t.get("detail") or t.get("reason") or "?")[:80] for t in nones)),
                           "passed": passed, "counts": {o: counts.get(o, 0) for o in ("pass", "fail_answer", "fail_format", "none")}}
    out["pass_at_1"] = _round(passed / n) if n else None
    out["ci"] = [_round(x) for x in wilson(passed, n)] if n else None
    out["format_fail_rate"] = _round(counts.get("fail_format", 0) / n) if n else None
    out["answer_fail_rate"] = _round(counts.get("fail_answer", 0) / n) if n else None
    own = [t for t in valid if t.get("format_failures") is not None]          # command-line agents do not expose their calls
    out["format_error_rate"] = _round(sum(1 for t in own if t["format_failures"] > 0) / len(own)) if own else None
    out["extra_calls_mean"] = _mean((t["extra_calls"] for t in own), 3)
    out["extra_rate"] = _round(sum(1 for t in own if (t["extra_calls"] or 0) > 0) / len(own)) if own else None
    out["turns_mean"] = _mean((t.get("turns") for t in valid), 2)
    out["calls_mean"] = _mean((t.get("calls") for t in valid), 2)
    out["tokens_out_mean"] = _mean((t.get("tokens_out") for t in valid), 1)
    out["tokens_in_mean"] = _mean((t.get("tokens_in") for t in valid), 1)
    out["latency_ms_mean"] = _mean((t.get("latency_ms") for t in valid), 1)
    if k > 1 and n:
        by_task: dict[str, list[bool]] = {}
        for t in valid:
            by_task.setdefault(t["task_id"], []).append(t["outcome"] == "pass")
        at_k, hat_k = [], []
        for results in by_task.values():
            size, good = len(results), sum(results)
            kk = min(k, size)
            at_k.append(pass_at_k(size, good, kk))
            hat_k.append(pass_hat_k(size, good, kk))
        out["pass_at_k"] = _mean((x for x in at_k if not math.isnan(x)), 4)
        out["pass_hat_k"] = _mean((x for x in hat_k if not math.isnan(x)), 4)
    if any(t.get("lang") for t in valid):
        out.update(_language_metrics(valid))
    return out


def build_matrix(trials: list[dict[str, Any]], model_ids: list[str], harnesses: list[str], k: int = 1) -> dict[str, Any]:
    """``harnesses`` are the column ids: a harness id, or ``harness@lang`` for the trials of a language-bench matrix."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for t in trials:
        grouped.setdefault((t["contestant_id"], column_id(t["harness"], t.get("lang") or "")), []).append(t)
    cells = {m: {h: cell_stats(grouped.get((m, h), []), k) for h in harnesses} for m in model_ids}
    by_model = {m: _mean((c["pass_at_1"] for c in cells[m].values()), 4) for m in model_ids}
    by_harness = {h: _mean((cells[m][h]["pass_at_1"] for m in model_ids), 4) for h in harnesses}
    return {"cells": cells, "by_model": by_model, "by_harness": by_harness}


def _pts(x: float) -> str:
    return f"{x:.1f}"


def conclude(matrix: dict[str, Any], names: dict[str, str], harnesses: list[str]) -> dict[str, Any]:
    """The short automatic reading of a matrix. ``names`` maps a model id to the name shown."""
    cells = matrix["cells"]
    ids = list(cells)
    value = lambda m, h: cells[m][h]["pass_at_1"]  # noqa: E731
    per_model: dict[str, dict[str, Any]] = {}
    model_spreads, harness_spreads = [], []
    for m in ids:
        have = [(value(m, h), h) for h in harnesses if value(m, h) is not None]
        if not have:
            per_model[m] = {"name": names.get(m, m), "best": None, "verdict": "no_data"}
            continue
        order = {h: i for i, h in enumerate(harnesses)}
        ranked = sorted(have, key=lambda vh: (-vh[0], cells[m][vh[1]]["format_fail_rate"] or 0.0, cells[m][vh[1]]["tokens_out_mean"] or 0.0, order[vh[1]]))
        best, worst = ranked[0], min(have, key=lambda vh: (vh[0], -order[vh[1]]))
        spread = (best[0] - worst[0]) * 100 if len(have) > 1 else None
        fails = [(cells[m][h]["counts"]["fail_format"], cells[m][h]["counts"]["fail_answer"]) for h in harnesses]
        total_fail = sum(f + a for f, a in fails)
        second = ranked[1] if len(ranked) > 1 else None
        clear = bool(second and cells[m][best[1]]["ci"][0] > cells[m][second[1]]["ci"][1])
        if best[0] < LOW_PASS:
            verdict = "weak_everywhere"
        elif spread is not None and spread >= SENSITIVE_POINTS:
            verdict = "harness_sensitive"
        else:
            verdict = "stable"
        per_model[m] = {"name": names.get(m, m), "best": best[1], "best_pass": best[0], "worst": worst[1] if len(have) > 1 else None, "worst_pass": worst[0] if len(have) > 1 else None,
                        "spread_points": _round(spread, 1), "second": second[1] if second else None, "margin_points": _round((best[0] - second[0]) * 100, 1) if second else None,
                        "best_is_clear": clear, "format_share_of_failures": _round(sum(f for f, _ in fails) / total_fail) if total_fail else None, "verdict": verdict}
        if spread is not None:
            model_spreads.append(spread)
    for h in harnesses:
        have = [value(m, h) for m in ids if value(m, h) is not None]
        if len(have) > 1:
            harness_spreads.append((max(have) - min(have)) * 100)
    x = round(sum(model_spreads) / len(model_spreads), 1) if model_spreads else None       # same model, other harness
    y = round(sum(harness_spreads) / len(harness_spreads), 1) if harness_spreads else None  # same harness, other model
    if x is None or y is None:
        verdict = "not_comparable"
    elif x - y > SIMILAR_POINTS:
        verdict = "harness_larger"
    elif y - x > SIMILAR_POINTS:
        verdict = "model_larger"
    else:
        verdict = "similar"
    valid_counts = [c["valid"] for row in cells.values() for c in row.values() if c["trials"]]
    min_valid = min(valid_counts) if valid_counts else 0
    weak = min_valid < WEAK_BELOW
    none_total = sum(c["none_n"] for row in cells.values() for c in row.values())
    best_by_model = {names.get(m, m): d["best"] for m, d in per_model.items() if d.get("best")}
    out = {"verdict": verdict, "harness_spread_points": x, "model_spread_points": y, "harness_spread_max": _round(max(model_spreads), 1) if model_spreads else None,
           "model_spread_max": _round(max(harness_spreads), 1) if harness_spreads else None, "best_harness_by_model": best_by_model, "per_model": per_model,
           "min_valid_per_cell": min_valid, "weak": weak, "none_total": none_total}
    out["summary"] = {"en": _text("en", out, names), "es": _text("es", out, names)}
    return out


def _text(lang: str, c: dict[str, Any], names: dict[str, str]) -> str:
    es = lang == "es"
    x, y = c["harness_spread_points"], c["model_spread_points"]
    parts: list[str] = []
    if c["verdict"] == "not_comparable":
        parts.append("Con un solo modelo o un solo arnés no se pueden comparar las dos diferencias." if es else "With a single model or a single harness the two spreads cannot be compared.")
        if x is not None:
            parts.append(f"Cambiar de arnés mueve el pass@1 de un mismo modelo {_pts(x)} puntos de media." if es else f"Changing the harness moves a model's pass@1 by {_pts(x)} points on average.")
        if y is not None:
            parts.append(f"Cambiar de modelo con el mismo arnés lo mueve {_pts(y)} puntos de media." if es else f"Changing the model under one harness moves it by {_pts(y)} points on average.")
    else:
        word_es = {"harness_larger": "mayor", "model_larger": "menor", "similar": "parecida"}[c["verdict"]]
        word_en = {"harness_larger": "larger", "model_larger": "smaller", "similar": "similar"}[c["verdict"]]
        if es:
            parts.append(f"La diferencia entre arneses ({_pts(x)} puntos de pass@1 para un mismo modelo) es {word_es} que la diferencia entre modelos ({_pts(y)} puntos con un mismo arnés).")
        else:
            parts.append(f"The difference between harnesses ({_pts(x)} pass@1 points for the same model) is {word_en} than the difference between models ({_pts(y)} points under the same harness).")
    for m, d in c["per_model"].items():
        if d.get("best") is None:
            continue
        name = d["name"]
        if d["worst"] is None:
            parts.append(f"{name}: solo un arnés medido ({d['best']}, {round(d['best_pass'] * 100)} %)." if es else f"{name}: only one harness measured ({d['best']}, {round(d['best_pass'] * 100)} %).")
            continue
        base = f"{name}: {d['best']} {round(d['best_pass'] * 100)} %, {d['worst']} {round(d['worst_pass'] * 100)} %"
        tail = {"harness_sensitive": ("depende del arnés", "depends on the harness"), "weak_everywhere": ("falla con cualquier arnés", "fails whatever the harness"),
                "stable": ("estable entre arneses", "stable across harnesses")}[d["verdict"]]
        parts.append(f"{base}: {tail[0] if es else tail[1]}.")
    if c["none_total"]:
        parts.append(f"{c['none_total']} pruebas sin resultado (infraestructura) quedan fuera de los porcentajes." if es else f"{c['none_total']} trials without a result (infrastructure) are left out of the percentages.")
    if c["weak"]:
        parts.append(f"Evidencia débil: solo {c['min_valid_per_cell']} pruebas válidas en la celda con menos." if es else f"Weak evidence: only {c['min_valid_per_cell']} valid trials in the smallest cell.")
    return " ".join(parts)


# ------------------------------------------------------------------------------------------------- the language axis
def _spread(rows: list[list[Optional[float]]]) -> Optional[float]:
    spreads = [(max(v) - min(v)) * 100 for v in ([x for x in row if x is not None] for row in rows) if len(v) > 1]
    return round(sum(spreads) / len(spreads), 1) if spreads else None


def _paired_calls(trials: list[dict[str, Any]], model: str, harness: str, lang: str, base: str) -> Optional[float]:
    """Tool calls in ``lang`` divided by tool calls in the base language, over the (task, sample) pairs that ran in both."""
    def keyed(code: str) -> dict[tuple[str, int], int]:
        return {(t["task_id"], t["sample"]): t["calls"] for t in trials if t["contestant_id"] == model and t["harness"] == harness and t.get("lang") == code
                and t["outcome"] != "none" and t.get("calls") is not None}
    a, b = keyed(lang), keyed(base)
    common = a.keys() & b.keys()
    den = sum(b[c] for c in common)
    return round(sum(a[c] for c in common) / den, 3) if common and den else None


def language_report(trials: list[dict[str, Any]], matrix: dict[str, Any], model_ids: list[str], harnesses: list[str], langs: list[str], names: dict[str, str]) -> dict[str, Any]:
    """The language axis of a matrix: per model and harness, each language against the base one, and the size of the language effect against the harness and model effects."""
    base = "en" if "en" in langs else langs[0]
    cells = matrix["cells"]
    pairs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for m in model_ids:
        pairs[m] = {}
        for h in harnesses:
            row: dict[str, dict[str, Any]] = {}
            b = cells[m][column_id(h, base)]
            for lang in langs:
                c = cells[m][column_id(h, lang)]
                row[lang] = {"pass_at_1": c["pass_at_1"], "valid": c["valid"], "pass_hat_k": c.get("pass_hat_k"), "lang_consistency": c.get("lang_consistency"),
                             "inconsistent_rate": c.get("inconsistent_rate"), "empty_search_rate": c.get("empty_search_rate"), "empty_search_trial_rate": c.get("empty_search_trial_rate"),
                             "arg_mismatch_rate": c.get("arg_mismatch_rate"), "calls_mean": c.get("calls_mean"), "search_calls_mean": c.get("search_calls_mean"),
                             "pass_delta_points": None if lang == base or c["pass_at_1"] is None or b["pass_at_1"] is None else round((c["pass_at_1"] - b["pass_at_1"]) * 100, 1),
                             "calls_vs_base": None if lang == base else _paired_calls(trials, m, h, lang, base)}
            pairs[m][h] = row
    by_lang: dict[str, dict[str, Any]] = {}
    for lang in langs:
        cs = [cells[m][column_id(h, lang)] for m in model_ids for h in harnesses]
        valid = [c for c in cs if c["trials"] and c["pass_at_1"] is not None]
        total_search = sum(c.get("search_calls") or 0 for c in cs)
        total_args = sum(c.get("arg_calls") or 0 for c in cs)
        ratios = [pairs[m][h][lang]["calls_vs_base"] for m in model_ids for h in harnesses if lang != base and pairs[m][h][lang]["calls_vs_base"] is not None]
        deltas = [pairs[m][h][lang]["pass_delta_points"] for m in model_ids for h in harnesses if pairs[m][h][lang]["pass_delta_points"] is not None]
        by_lang[lang] = {"pass_at_1": _mean((c["pass_at_1"] for c in valid), 4), "lang_consistency": _mean((c.get("lang_consistency") for c in cs), 4),
                         "empty_search_rate": _round(sum(c.get("empty_searches") or 0 for c in cs) / total_search) if total_search else None,
                         "arg_mismatch_rate": _round(sum(c.get("arg_mismatch") or 0 for c in cs) / total_args) if total_args else None,
                         "calls_vs_base": _mean(ratios, 3), "pass_delta_points": _mean(deltas, 1)}
    effects = {"language_points": _spread([[cells[m][column_id(h, lang)]["pass_at_1"] for lang in langs] for m in model_ids for h in harnesses]),
               "harness_points": _spread([[cells[m][column_id(h, lang)]["pass_at_1"] for h in harnesses] for m in model_ids for lang in langs]),
               "model_points": _spread([[cells[m][column_id(h, lang)]["pass_at_1"] for m in model_ids] for h in harnesses for lang in langs])}
    ranked = sorted(((v, k) for k, v in (("language", effects["language_points"]), ("harness", effects["harness_points"]), ("model", effects["model_points"])) if v is not None), reverse=True)
    if len(ranked) < 2:
        effects["verdict"] = "not_comparable"
    elif ranked[0][0] - ranked[1][0] <= SIMILAR_POINTS:
        effects["verdict"] = "similar"
    else:
        effects["verdict"] = f"{ranked[0][1]}_larger"
    out = {"langs": list(langs), "base": base, "pairs": pairs, "by_lang": by_lang, "effects": effects}
    out["summary"] = {"en": _language_text("en", out), "es": _language_text("es", out)}
    return out


def _language_text(lang: str, rep: dict[str, Any]) -> str:
    es, base, e = lang == "es", rep["base"], rep["effects"]
    parts: list[str] = []
    lp, hp, mp = e["language_points"], e["harness_points"], e["model_points"]
    if lp is not None:
        parts.append(f"Cambiar de idioma (mismo modelo y arnés) mueve el pass@1 {_pts(lp)} puntos de media; cambiar de arnés, {'—' if hp is None else _pts(hp)}; cambiar de modelo, {'—' if mp is None else _pts(mp)}." if es
                     else f"Changing the language (same model and harness) moves pass@1 by {_pts(lp)} points on average; changing the harness, {'—' if hp is None else _pts(hp)}; changing the model, {'—' if mp is None else _pts(mp)}.")
    for code, d in rep["by_lang"].items():
        if code == base:
            continue
        bits_es, bits_en = [], []
        if d["calls_vs_base"] is not None:
            bits_es.append(f"{d['calls_vs_base']:.2f} veces las llamadas de {base}")
            bits_en.append(f"{d['calls_vs_base']:.2f} times the tool calls of {base}")
        if d["empty_search_rate"] is not None:
            bits_es.append(f"{round(d['empty_search_rate'] * 100)} % de búsquedas vacías")
            bits_en.append(f"{round(d['empty_search_rate'] * 100)} % empty searches")
        if d["arg_mismatch_rate"] is not None:
            bits_es.append(f"{round(d['arg_mismatch_rate'] * 100)} % de argumentos en otro idioma que los datos")
            bits_en.append(f"{round(d['arg_mismatch_rate'] * 100)} % of arguments in another language than the data")
        if d["lang_consistency"] is not None:
            bits_es.append(f"consistencia de idioma {round(d['lang_consistency'] * 100)} %")
            bits_en.append(f"language consistency {round(d['lang_consistency'] * 100)} %")
        if bits_en:
            parts.append(f"En {code}: " + ", ".join(bits_es) + "." if es else f"In {code}: " + ", ".join(bits_en) + ".")
    return " ".join(parts)
