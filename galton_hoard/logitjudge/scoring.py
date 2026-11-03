"""From the log-probabilities of the option tokens to an answer: chosen option, distribution, expected score, and the difference between two runs."""

from __future__ import annotations

import math
from typing import Any, Optional

#: below this share of probability on the option labels the answer is flagged: the model wanted to write something else
LOW_MASS = 0.5


def softmax_over(logprobs: dict[str, Optional[float]]) -> tuple[dict[str, float], float]:
    """Normalise over the labels that were read. Returns (probabilities summing to 1, the mass the labels held before normalising)."""
    known = {k: v for k, v in logprobs.items() if v is not None and math.isfinite(v)}
    if not known:
        return {k: 0.0 for k in logprobs}, 0.0
    top = max(known.values())
    weights = {k: math.exp(v - top) for k, v in known.items()}
    total = sum(weights.values())
    mass = sum(math.exp(v) for v in known.values())
    return {k: (weights.get(k, 0.0) / total) for k in logprobs}, min(1.0, mass)


def answer_for(question: dict[str, Any], logprobs: dict[str, Optional[float]]) -> dict[str, Any]:
    """The answer to one question. A choice gives ``chosen`` (label and text) and ``probs``; a score also gives ``expected`` (probability-weighted)."""
    probs, mass = softmax_over({label: logprobs.get(label) for label in question["labels"]})
    chosen = max(question["labels"], key=lambda label: probs[label])
    entropy = -sum(p * math.log(p) for p in probs.values() if p > 0)
    out: dict[str, Any] = {"chosen": chosen, "confidence": round(probs[chosen], 6), "probs": {k: round(v, 6) for k, v in probs.items()},
                           "mass": round(mass, 6), "entropy": round(entropy, 6),
                           "logprobs": {k: (None if v is None else round(v, 5)) for k, v in logprobs.items() if k in probs}}
    if question["type"] == "choice":
        out["text"] = question["options"][question["labels"].index(chosen)]
    else:
        out["expected"] = round(sum(p * int(label) for label, p in probs.items()), 6)
        out["value"] = int(chosen)
    if mass < LOW_MASS:
        out["low_mass"] = True
    return out


def _mean(values: list[float]) -> Optional[float]:
    return sum(values) / len(values) if values else None


def aggregate(rubric: list[dict[str, Any]], items: list[dict[str, Any]]) -> dict[str, Any]:
    """Per question over all items: mean expected score (score) or mean probability and share of items choosing each option (choice)."""
    out: dict[str, Any] = {}
    for q in rubric:
        answers = [it["answers"][q["id"]] for it in items if q["id"] in it.get("answers", {})]
        if not answers:
            continue
        row: dict[str, Any] = {"n": len(answers), "type": q["type"], "mean_mass": round(_mean([a["mass"] for a in answers]) or 0.0, 6)}
        if q["type"] == "score":
            row["mean_expected"] = round(_mean([a["expected"] for a in answers]) or 0.0, 6)
        else:
            row["mean_probs"] = {label: round(_mean([a["probs"][label] for a in answers]) or 0.0, 6) for label in q["labels"]}
            row["chosen_share"] = {label: round(sum(1 for a in answers if a["chosen"] == label) / len(answers), 6) for label in q["labels"]}
        out[q["id"]] = row
    return out


def deltas(rubric: list[dict[str, Any]], base: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    """``other`` minus ``base`` (two result sets). Score: change of the mean expected score. Choice: change of each option's mean probability.
    ``flips`` counts the items where the chosen answer changed."""
    out: dict[str, Any] = {}
    base_items = {it["id"]: it for it in base["items"]}
    for q in rubric:
        a, b = base["aggregate"].get(q["id"]), other["aggregate"].get(q["id"])
        if not a or not b:
            continue
        flips = 0
        for it in other["items"]:
            before = base_items.get(it["id"], {}).get("answers", {}).get(q["id"])
            after = it["answers"].get(q["id"])
            if before and after and before["chosen"] != after["chosen"]:
                flips += 1
        row: dict[str, Any] = {"type": q["type"], "flips": flips, "n": min(a["n"], b["n"])}
        if q["type"] == "score":
            row["expected"] = {"base": a["mean_expected"], "other": b["mean_expected"], "delta": round(b["mean_expected"] - a["mean_expected"], 6)}
        else:
            row["probs"] = {label: {"base": a["mean_probs"][label], "other": b["mean_probs"][label], "delta": round(b["mean_probs"][label] - a["mean_probs"][label], 6)}
                            for label in q["labels"]}
        out[q["id"]] = row
    return out
