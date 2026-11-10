"""The RAG-triad checkers (faithfulness, answer_relevance) and the model-free similarity checker."""

from __future__ import annotations

import json

import pytest

from galton_hoard.checkers import CHECKER_HELP, JUDGED, NON_DETERMINISTIC, checker_types, is_deterministic, run_checker
from galton_hoard.checkers.judge import build_judge_messages, parse_judge_reply
from galton_hoard.checkers.ragtriad import char_ratio, parse_claims_reply, rouge_l, token_f1
from galton_hoard.checkers.types import CheckContext, ModelOutput
from galton_hoard.judging import has_judge


def _judge_from(text: str):
    """A judge that answers ``text`` and records what it was asked (parsed the way the real judge parses it)."""
    seen: list[dict] = []

    def ask(request):
        seen.append(request)
        reply = parse_judge_reply(text, request.get("parse", ""))
        return {**reply, "judge": "fake"}
    return ask, seen


CLAIMS = json.dumps({"claims": [
    {"claim": "La ciudad se fundó en 1850", "supported": True, "why": "lo dice el pasaje 1"},
    {"claim": "Tiene un millón de habitantes", "supported": False, "why": "no aparece"},
]})


def test_faithfulness_scores_the_share_of_supported_claims_and_lists_the_rest():
    ask, seen = _judge_from(CLAIMS)
    ctx = CheckContext(case={"prompt_text": "¿Cuándo se fundó?"}, judge=ask)
    out = run_checker({"type": "faithfulness", "context": ["La ciudad se fundó en 1850.", "Es costera."]}, ModelOutput(text="Se fundó en 1850 y tiene un millón de habitantes."), ctx)
    assert out["score"] == 0.5 and out["passed"] is False
    assert out["detail"]["supported"] == 1 and out["detail"]["claims"] == 2
    assert out["detail"]["unsupported"] == ["Tiene un millón de habitantes"]
    request = seen[0]
    assert request["parse"] == "claims" and "[1] La ciudad se fundó en 1850." in request["reference"]
    assert build_judge_messages(request)[1]["content"].startswith("CONTEXTO:")
    passing = run_checker({"type": "faithfulness", "context": "x", "pass_score": 0.5}, ModelOutput(text="y"), ctx)
    assert passing["passed"] is True


def test_faithfulness_falls_back_to_the_case_context_then_the_prompt():
    ask, seen = _judge_from(json.dumps({"claims": []}))
    run_checker({"type": "faithfulness"}, ModelOutput(text="Hola"), CheckContext(case={"context": "pasaje del caso", "prompt_text": "p"}, judge=ask))
    run_checker({"type": "faithfulness"}, ModelOutput(text="Hola"), CheckContext(case={"prompt_text": "pasajes en el enunciado"}, judge=ask))
    assert seen[0]["reference"] == "pasaje del caso" and seen[1]["reference"] == "pasajes en el enunciado"


def test_faithfulness_without_a_judge_or_with_a_bad_reply_is_unavailable_not_failed():
    out = run_checker({"type": "faithfulness", "context": "c"}, ModelOutput(text="a"), CheckContext())
    assert out["unavailable"] is True
    ask, _ = _judge_from("no json here")
    bad = run_checker({"type": "faithfulness", "context": "c"}, ModelOutput(text="a"), CheckContext(judge=ask))
    assert bad["unavailable"] is True
    empty = run_checker({"type": "faithfulness"}, ModelOutput(text="a"), CheckContext(judge=ask))
    assert "invalid checker" in empty["detail"]["error"]


def test_claims_parser():
    assert parse_claims_reply(CLAIMS)["score"] == 5.0
    assert parse_claims_reply('{"claims": []}') == {"score": 10.0, "reasons": "no factual claims", "claims": []}
    assert "error" in parse_claims_reply("{}")
    assert parse_judge_reply(CLAIMS, "claims")["score"] == 5.0, "registered with the judge's reply parsers"


def test_answer_relevance_uses_a_fixed_rubric():
    ask, seen = _judge_from('{"score": 8, "reasons": "responde"}')
    out = run_checker({"type": "answer_relevance"}, ModelOutput(text="Mañana a las nueve."), CheckContext(case={"prompt_text": "¿Cuándo abre?"}, judge=ask))
    assert out["score"] == 0.8 and out["passed"] is True
    assert seen[0]["prompt"] == "¿Cuándo abre?" and "atiende" in seen[0]["rubric"]
    assert run_checker({"type": "answer_relevance"}, ModelOutput(text="x"), CheckContext())["unavailable"] is True


@pytest.mark.parametrize("method", ["rouge_l", "token_f1", "chars"])
def test_similar_methods(method):
    ref = "El tratado se firmó en Viena en 1815."
    close = run_checker({"type": "similar", "method": method, "reference": ref, "threshold": 0.5}, ModelOutput(text="El tratado se firmó en Viena, en 1815."), CheckContext())
    far = run_checker({"type": "similar", "method": method, "reference": ref, "threshold": 0.5}, ModelOutput(text="No lo sé."), CheckContext())
    assert close["passed"] is True and far["passed"] is False and close["score"] > far["score"]


def test_similar_takes_the_best_of_several_references_and_the_case_reference():
    out = run_checker({"type": "similar", "reference": ["algo distinto", "rojo verde azul"]}, ModelOutput(text="rojo verde azul"), CheckContext())
    assert out["score"] == 1.0 and out["detail"]["best_reference"] == 1
    from_case = run_checker({"type": "similar"}, ModelOutput(text="hola mundo"), CheckContext(case={"reference": "hola mundo"}))
    assert from_case["passed"] is True
    assert "invalid checker" in run_checker({"type": "similar"}, ModelOutput(text="x"), CheckContext())["detail"]["error"]
    assert "invalid checker" in run_checker({"type": "similar", "reference": "a", "method": "bleu"}, ModelOutput(text="x"), CheckContext())["detail"]["error"]


def test_metric_values():
    assert rouge_l("a b c d", "a b c d") == 1.0 and rouge_l("", "a") == 0.0
    assert token_f1("a a b", "a b b") == pytest.approx(2 / 3)
    assert 0.9 < char_ratio("Hola mundo", "hola  mundo") <= 1.0


def test_catalogue_flags():
    assert {"faithfulness", "answer_relevance", "similar"} <= set(checker_types())
    assert set(CHECKER_HELP) == set(checker_types())
    assert {"faithfulness", "answer_relevance"} <= JUDGED and {"faithfulness", "answer_relevance"} <= NON_DETERMINISTIC
    assert is_deterministic({"type": "similar"}) and not is_deterministic({"type": "faithfulness"})
    assert has_judge({"type": "all", "checks": [{"type": "similar"}, {"type": "faithfulness"}]})
    assert not has_judge({"type": "similar"})
