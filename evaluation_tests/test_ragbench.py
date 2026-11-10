"""The RAG pipeline bench: the measures of each piece on hand-made lists, the attribution of every kind of failure, the three pipelines
(built-in, a Faustus and a generic one behind mock HTTP), case generation with a scripted model, runs with pieces broken on purpose and the
comparison of two runs. Everything is invented data with a known outcome."""

from __future__ import annotations

import json
import math
import threading

import httpx
import pytest

from evaluation_tests.conftest import tool
from evaluation_tests.helpers import Clock, build_services, json_response, mock_client_factory
from galton_hoard.errors import GaltonError
from galton_hoard.ragbench import attribution as A, generate as G, metrics as M, pipelines as P
from galton_hoard.ragbench.judge import GenerationJudge, lexical
from galton_hoard.ragbench.service import normalise_settings

DOCS = [
    {"name": "faro.md", "text": "El faro de Punta Brava se encendió por primera vez en 1887. Su torre mide 42 metros y la luz alcanza 23 millas náuticas.\n\n"
                                "El último farero, Anselmo Rivas, se jubiló en 1994, cuando la linterna se automatizó."},
    {"name": "molino.md", "text": "El molino de Aldea Vieja muele trigo desde 1702. Tiene tres muelas de granito y lo mueve el agua del río Cerval.\n\n"
                                  "La cooperativa restauró el molino de Aldea Vieja en 2011 con un presupuesto de 180000 euros."},
    {"name": "puente.md", "text": "El puente de hierro de Riosa tiene 96 metros de largo y fue diseñado por la ingeniera Clara Montes. Se inauguró en 1931.\n\n"
                                  "El puente de Riosa soporta un máximo de 12 toneladas por vehículo y se pinta cada ocho años."},
]
CASES = [
    {"question": "¿En qué año se encendió por primera vez el faro de Punta Brava?", "expected": "En 1887.",
     "passages": [{"text": "El faro de Punta Brava se encendió por primera vez en 1887."}]},
    {"question": "¿Quién diseñó el puente de hierro de Riosa?", "expected": "La ingeniera Clara Montes.", "passages": [{"text": "fue diseñado por la ingeniera Clara Montes"}]},
    {"question": "¿Con qué presupuesto se restauró el molino de Aldea Vieja?", "expected": "180000 euros.",
     "passages": [{"doc": "molino.md", "text": "La cooperativa restauró el molino de Aldea Vieja en 2011 con un presupuesto de 180000 euros."}]},
    {"question": "¿Cuántos visitantes recibe el faro de Punta Brava al año?", "answerable": False},
]
SMALL = {"kind": "builtin", "generator": "extractive", "chunk_chars": 130, "chunk_overlap": 0, "depth": 6}
LEX = {"judge_mode": "lexical", "parallel": 1}


def make_set(svc, cases=CASES, docs=DOCS, name="obras"):
    return tool(svc, "rag_set_save", name=name, documents=docs, cases=cases)["set"]["id"]


def run_now(svc, sid, pipeline, **kw):
    out = svc.ragbench.run_start(set=sid, pipeline=pipeline, settings=kw.pop("settings", LEX), start_thread=False, **kw)
    return tool(svc, "rag_run_result", run=out["run"]["id"])


def add_server(svc, name, responder):
    svc.fake_world.register(name, responder)
    return svc.store.create_contestant(key=f"server:test:{name}", kind="server", name=name, url="http://fake.invalid:9000", api="openai", model=name, provider="openai_compat",
                                       aliases=[name], remote=False, remote_ok=False, enabled=True, source="manual")


def user_of(req):
    return next((str(m.get("content", "")) for m in reversed(req.messages) if m.get("role") == "user"), "")


def reader(req):
    """A generator that answers from the passages it is given and declines when the fact is not there."""
    passages, _, question = user_of(req).partition("PREGUNTA:")
    for needle, asked, answer in (("primera vez en 1887", "encendió", "Se encendió en 1887."), ("ingeniera Clara Montes", "diseñó", "Lo diseñó la ingeniera Clara Montes."),
                                  ("presupuesto de 180000 euros", "presupuesto", "Con un presupuesto de 180000 euros.")):
        if needle in passages and asked in question:
            return answer
    return P.NO_ANSWER


# ------------------------------------------------------------------ retrieval measures
def test_retrieval_measures_on_a_known_ranking():
    refs = [{"id": "a"}, {"id": "b"}]
    found = M.hits(refs, [{"id": "x"}, {"id": "a"}, {"id": "y"}, {"id": "b"}, {"id": "a"}])
    assert found == [[], [0], [], [1], [0]]
    assert M.first_rank(found) == 2 and M.mrr(found) == 0.5
    assert [M.recall_at(found, 2, k) for k in (1, 2, 4)] == [0.0, 0.5, 1.0]
    ideal = 1 + 1 / math.log2(3)
    assert M.ndcg_at(found, 2, 5) == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)      # the repeated passage adds nothing
    assert M.ndcg_at(M.hits(refs, [{"id": "a"}, {"id": "b"}]), 2, 2) == pytest.approx(1.0)
    assert M.satisfied(found, 2, 2) is False and M.satisfied(found, 2, 2, "any") is True and M.satisfied(found, 2) is True
    assert M.recall_at(found, 0, 3) is None and M.mrr([[], []]) == 0.0


def test_a_passage_is_the_reference_by_id_containment_coverage_or_document():
    ref = {"text": "La cooperativa restauró el molino en 2011 con un presupuesto de 180000 euros."}
    assert M.passage_matches(ref, {"text": "Historia. LA COOPERATIVA restauro el molino en 2011 con un presupuesto de 180000 euros. Fin."})      # contained, accents and case apart
    assert M.passage_matches(ref, {"text": "En 2011 la cooperativa restauró aquel molino: 180000 euros de presupuesto y dos años de obra."})   # most content words
    assert not M.passage_matches(ref, {"text": "El molino muele trigo desde 1702 con tres muelas."})
    assert not M.passage_matches({"text": "en 1887"}, {"text": "El año fue 1887, en verano."})            # a short reference must be contained whole
    assert M.passage_matches({"text": "en 1887"}, {"text": "Se encendió en 1887."})
    assert M.passage_matches({"doc": "D:\\docs\\Molino.md"}, {"doc": "molino", "text": "cualquier fragmento"})
    assert not M.passage_matches({"doc": "molino.md"}, {"doc": "faro.md", "text": "x"})
    assert M.passage_matches({"id": "d2#1", "text": "otra cosa"}, {"id": "d2#1", "text": "sin relación"})


def test_generation_measures_without_a_judge():
    passages = [{"text": "El faro de Punta Brava se encendió por primera vez en 1887. Su torre mide 42 metros."}]
    assert M.support("El faro se encendió en 1887.", passages) == 1.0
    assert M.support("El faro se encendió en 1901 por orden del rey.", passages) == 0.0           # a number the passages do not have
    assert M.support("El faro se encendió en 1887. Lo pintó un pirata llamado Barbazul.", passages) == 0.5
    assert M.support(P.NO_ANSWER, passages) == 1.0 and M.support("", passages) is None
    assert M.answer_coverage("Se encendió en 1887.", "En 1887.") == 1.0 and M.answer_coverage("No consta.", "En 1887.") == 0.0
    assert M.answer_coverage("algo", "") is None
    assert M.abstains("No lo sé: los pasajes no lo dicen.") and M.abstains("The passage does not mention it.") and not M.abstains("Fue en 1887.")
    gen = lexical("q", passages, "En 1887.", "Fue en 1887.")
    assert gen["mode"] == "lexical" and gen["coverage"] == 1.0 and gen["abstained"] is False and gen["empty"] is False


# ------------------------------------------------------------------ attribution
def verdict(retrieved, context, gen, *, cut=2, ranked=None, answerable=True, n_refs=1, traced=True, **kw):
    return A.judge_case(answerable=answerable, n_refs=n_refs, traced=traced, found_retrieved=retrieved, found_ranked=ranked or retrieved, found_context=context, cut=cut, gen=gen,
                        reranked=ranked is not None, **kw)


GOOD = {"faithfulness": 1.0, "coverage": 1.0, "abstained": False}


def test_every_failure_goes_to_the_piece_that_lost_the_case():
    hit, miss = [0], []
    assert verdict([hit, miss], [hit, miss], GOOD) == {**verdict([hit, miss], [hit, miss], GOOD), "ok": True, "piece": "", "cause": ""}
    v = verdict([miss, miss, miss], [miss, miss], {"faithfulness": 1.0, "coverage": 0.0, "abstained": True})
    assert (v["piece"], v["cause"], v["generator_note"]) == ("retriever", "not_retrieved", "abstained_rightly")
    v = verdict([miss, miss, hit], [miss, miss], {"faithfulness": 0.1, "coverage": 0.0, "abstained": False})
    assert (v["piece"], v["cause"], v["generator_note"]) == ("retriever", "cut_off", "invented")       # no reranker: the order is the retriever's own
    v = verdict([miss, miss, hit], [miss, miss], GOOD, ranked=[miss, miss, hit])
    assert (v["piece"], v["cause"]) == ("ranking", "cut_off")                                          # a reranker that did not lift it
    v = verdict([hit, miss, miss], [miss, miss], GOOD, ranked=[miss, miss, hit])
    assert (v["piece"], v["cause"], v["demoted"], v["generator_note"]) == ("ranking", "demoted", True, "answered_without_passage")
    v = verdict([miss, miss, hit], [hit, miss], GOOD, ranked=[hit, miss, miss])
    assert v["ok"] and v["rescued"] is True
    for gen, cause in (({"faithfulness": 1.0, "coverage": 0.0, "abstained": True}, "abstained"), ({"faithfulness": 0.2, "coverage": 0.9, "abstained": False}, "invented"),
                       ({"faithfulness": 1.0, "coverage": 0.3, "abstained": False}, "incomplete"), ({"empty": True, "faithfulness": None, "coverage": 0.0}, "no_answer")):
        v = verdict([hit, miss], [hit, miss], gen)
        assert (v["piece"], v["cause"]) == ("generator", cause), cause
    assert verdict([], [], {"abstained": True}, answerable=False)["ok"] is True
    v = verdict([], [], {"abstained": False, "faithfulness": 0.0}, answerable=False)
    assert (v["piece"], v["cause"]) == ("generator", "did_not_abstain")
    v = verdict([], [], {"faithfulness": 1.0, "coverage": 0.1, "abstained": False}, traced=False)
    assert (v["piece"], v["cause"]) == ("unknown", "no_trace")
    assert verdict([], [], GOOD, traced=False)["ok"] is True
    assert verdict([], [], {}, error="boom")["cause"] == "error"
    v = verdict([[0], [1]], [[0], []], GOOD, n_refs=2)
    assert (v["piece"], v["cause"]) == ("ranking", "demoted") and verdict([[0], [1]], [[0], []], GOOD, n_refs=2, require="any")["ok"]


def test_the_summary_gives_the_share_of_each_piece():
    verdicts = [{"ok": True}] * 4 + [{"ok": False, "piece": "retriever", "cause": "not_retrieved", "generator_note": "abstained_rightly"}] * 3 + \
               [{"ok": False, "piece": "generator", "cause": "invented"}]
    s = A.summarise(verdicts)
    assert (s["cases"], s["passed"], s["failed"], s["main"]) == (8, 4, 4, "retriever")
    assert s["share"]["retriever"] == 0.75 and s["share"]["generator"] == 0.25 and s["by_cause"] == {"not_retrieved": 3, "invented": 1}
    assert s["generator_notes"] == {"abstained_rightly": 3}
    assert A.summarise([{"ok": True}])["main"] is None and A.summarise([{"ok": True}])["share"]["retriever"] is None


# ------------------------------------------------------------------ the pieces of the built-in pipeline
def test_index_retrieves_reranks_and_the_broken_pieces_are_repeatable():
    index = P.Index([{"id": f"d{i}", **d} for i, d in enumerate(DOCS, 1)], chunk_chars=130, chunk_overlap=0)
    assert len(index.passages) >= 5
    top = index.retrieve("¿En qué año se encendió por primera vez el faro de Punta Brava?", "bm25", 3)
    assert "1887" in top[0]["text"] and top[0]["doc"] == "faro.md" and top[0]["score"] > 0
    assert index.retrieve("q", "none") == []
    assert index.retrieve("q", "random", 2, seed=3) == index.retrieve("q", "random", 2, seed=3)
    assert P.rerank("q", top, "none") is None and P.rerank("q", top, "reverse") == top[::-1]
    lifted = P.rerank("presupuesto restauró molino", [{"id": "a", "doc": "", "text": "Nada que ver."}, {"id": "b", "doc": "", "text": "Restauró el molino con un presupuesto."}], "overlap")
    assert [p["id"] for p in lifted] == ["b", "a"]
    scored = P.rerank("q", top, "model", rate=lambda q, text: 1.0 if "farero" in text else 0.0)
    assert "farero" in scored[0]["text"] or all("farero" not in p["text"] for p in top)
    assert P.extractive("¿Cuánto mide la torre?", [{"text": "Su torre mide 42 metros. Otra frase."}]) == "Su torre mide 42 metros."
    assert P.extractive("¿Quién ganó la liga?", [{"text": "Su torre mide 42 metros."}]) == P.NO_ANSWER


def test_pipeline_descriptions_are_validated():
    p = P.normalise({"kind": "builtin", "model": "m", "top_k": 2})
    assert p["generator"] == "model" and p["retriever"] == "bm25" and p["name"] == "bm25 · 600/80 · top 2 · model"
    assert P.normalise({"generator": "extractive"})["kind"] == "builtin"
    for bad, key in ((["x"], "rb_pipeline_shape"), ({"kind": "magic"}, "rb_pipeline_option"), ({"generator": "model"}, "rb_pipeline_needs_model"),
                     ({"reranker": "model", "generator": "extractive"}, "rb_pipeline_needs_model"), ({"kind": "http", "url": "ftp://x"}, "rb_pipeline_url"),
                     ({"kind": "faustus", "url": "http://f", "answer": "model"}, "rb_pipeline_needs_model"), ({"depth": 0, "generator": "extractive"}, "rb_pipeline_field")):
        with pytest.raises(GaltonError) as err:
            P.normalise(bad)
        assert err.value.key == key, bad
    shown = P.public(P.normalise({"kind": "faustus", "url": "http://f/", "token": "secret"}))
    assert "token" not in shown and shown["token_set"] is True and shown["url"] == "http://f"
    assert P.fill({"q": "{question}", "n": "{k}", "nested": ["about {question}"]}, {"question": "why", "k": 4}) == {"q": "why", "n": 4, "nested": ["about why"]}
    with pytest.raises(GaltonError):
        normalise_settings({"ks": [0]})
    assert normalise_settings({"ks": [5, 1, 5]})["ks"] == [1, 5] and normalise_settings(None)["judge_mode"] == "auto"


# ------------------------------------------------------------------ runs over the built-in pipeline
def test_a_run_scores_each_piece_and_passes_what_it_should(svc):
    sid = make_set(svc)
    add_server(svc, "lector", reader)
    out = run_now(svc, sid, {**SMALL, "generator": "model", "model": "lector", "top_k": 2})
    s = out["summary"]
    assert (s["cases"], s["passed"], s["pass_rate"]) == (4, 4, 1.0) and s["attribution"]["failed"] == 0 and s["attribution"]["main"] is None
    assert s["retrieval"]["cases"] == 3 and s["retrieval"]["recall"]["3"] == 1.0 and s["retrieval"]["mrr"] == 1.0 and s["retrieval"]["hit_rate"] == 1.0
    assert s["rerank"]["applied"] is False and s["rerank"]["mrr"]["delta"] == 0.0
    assert s["generation"]["with_passage"] == {"cases": 3, "right": 3, "rate": 1.0} and s["generation"]["abstention"]["unanswerable"] == {"cases": 1, "abstained": 1, "rate": 1.0}
    assert s["generation"]["judge_modes"] == {"lexical": 4} and any(w["key"] == "rb_warn_few_cases" for w in s["warnings"])
    one = tool(svc, "rag_run_result", run=out["run"]["id"], case=out["cases"][0]["case"])["case"]
    assert one["retrieved"][0]["hit"] is True and one["context"][0]["hit"] is True and one["expected"] == "En 1887." and one["metrics"]["retrieval"]["first_rank"] == 1


def test_a_retriever_broken_on_purpose_is_blamed_for_the_failures(svc):
    sid = make_set(svc)
    add_server(svc, "lector", reader)
    out = run_now(svc, sid, {**SMALL, "retriever": "none", "generator": "model", "model": "lector"})
    a = out["summary"]["attribution"]
    assert a["by_piece"] == {"retriever": 3, "ranking": 0, "generator": 0, "unknown": 0} and a["share"]["retriever"] == 1.0 and a["main"] == "retriever"
    assert a["by_cause"] == {"not_retrieved": 3} and a["generator_notes"] == {"abstained_rightly": 3}
    assert out["summary"]["retrieval"]["recall"]["10"] == 0.0 and out["summary"]["generation"]["abstention"]["without_passage"]["rate"] == 1.0
    assert out["summary"]["passed"] == 1                                                     # the unanswerable case: the generator rightly declines
    noisy = [{"name": f"relleno-{i}.md", "text": f"El almacén número {i} guarda sacos de sal, cuerdas y faroles viejos desde hace décadas en la calle {i}."} for i in range(30)]
    big = make_set(svc, docs=DOCS + noisy, name="con ruido")
    lost = run_now(svc, big, {**SMALL, "retriever": "random", "depth": 2, "top_k": 2, "generator": "model", "model": "lector"})["summary"]
    assert lost["attribution"]["main"] == "retriever" and lost["attribution"]["share"]["retriever"] >= 0.66 and lost["retrieval"]["recall"]["10"] <= 0.34


def test_a_reranker_that_reverses_the_order_is_blamed_as_ranking(svc):
    sid = make_set(svc)
    add_server(svc, "lector", reader)
    out = run_now(svc, sid, {**SMALL, "reranker": "reverse", "top_k": 1, "generator": "model", "model": "lector"})
    s = out["summary"]
    assert s["attribution"]["by_piece"]["ranking"] == 3 and s["attribution"]["by_cause"] == {"demoted": 3} and s["attribution"]["main"] == "ranking"
    assert s["rerank"]["applied"] is True and s["rerank"]["demoted"] == 3 and s["rerank"]["mrr"]["delta"] < 0 and s["rerank"]["recall"]["after"] == 0.0
    assert s["retrieval"]["recall"]["1"] == 1.0                                              # the retriever did its part
    rescued = run_now(svc, sid, {**SMALL, "retriever": "bm25", "reranker": "overlap", "top_k": 1, "generator": "model", "model": "lector"})["summary"]
    assert rescued["attribution"]["by_piece"]["ranking"] == 0 and rescued["rerank"]["applied"] is True


def test_a_generator_that_ignores_or_invents_is_blamed_as_generator(svc):
    sid = make_set(svc)
    add_server(svc, "inventor", lambda req: "Lo construyó el pirata Barbazul en 1650 con oro de Sumatra.")
    add_server(svc, "mudo", lambda req: P.NO_ANSWER)
    inv = run_now(svc, sid, {**SMALL, "generator": "model", "model": "inventor"})["summary"]["attribution"]
    assert inv["by_piece"]["generator"] == 4 and inv["by_cause"] == {"invented": 3, "did_not_abstain": 1} and inv["main"] == "generator"
    shy = run_now(svc, sid, {**SMALL, "generator": "model", "model": "mudo"})["summary"]
    assert shy["attribution"]["by_cause"] == {"abstained": 3} and shy["generation"]["abstention"]["with_passage"]["rate"] == 1.0 and shy["passed"] == 1
    blind = run_now(svc, sid, {**SMALL, "generator": "no_context", "model": "mudo"})["summary"]
    assert blind["attribution"]["by_piece"]["generator"] == 3


def test_cases_are_reviewed_before_they_count_and_runs_need_them(svc):
    sid = make_set(svc, cases=[])
    with pytest.raises(GaltonError) as err:
        svc.ragbench.run_start(set=sid, pipeline=SMALL, settings=LEX, start_thread=False)
    assert err.value.key == "rb_no_cases"
    made = tool(svc, "rag_case_save", set=sid, question=CASES[0]["question"], expected="En 1887.", passages=CASES[0]["passages"], status="proposed")
    assert made["status"] == "proposed" and tool(svc, "rag_sets")["sets"][0]["cases"] == {"proposed": 1, "accepted": 0, "rejected": 0}
    assert run_now(svc, sid, SMALL, include_proposed=True)["summary"]["cases"] == 1
    assert tool(svc, "rag_cases_review", set=sid, status="accepted", all_proposed=True)["changed"] == 1
    edited = tool(svc, "rag_case_save", id=made["id"], expected="En el año 1887.")
    assert edited["expected"] == "En el año 1887." and edited["question"] == CASES[0]["question"] and edited["status"] == "accepted"
    for bad, key in (({"set": sid, "question": ""}, "rb_case_question"), ({"set": sid, "question": "¿Y?"}, "rb_case_empty"), ({"id": "rk_none", "question": "x"}, "rb_no_case")):
        with pytest.raises(GaltonError) as err:
            tool(svc, "rag_case_save", **bad)
        assert err.value.key == key
    nothing = tool(svc, "rag_case_save", set=sid, question="¿Cuántos visitantes?", answerable=False, expected="da igual")
    assert nothing["answerable"] is False and nothing["expected"] == "" and nothing["passages"] == []
    with pytest.raises(GaltonError):
        tool(svc, "rag_case_remove", id=nothing["id"])
    assert tool(svc, "rag_case_remove", id=nothing["id"], confirm=True)["deleted"] == nothing["id"]


# ------------------------------------------------------------------ cases from a corpus
def test_a_model_proposes_cases_with_their_source_passage(svc):
    sid = make_set(svc, cases=[])

    def writer(req):
        text = user_of(req)
        if "NOT give" in req.messages[0]["content"]:
            return json.dumps({"cases": [{"question": "¿Cuánto pesa la linterna del faro de Punta Brava?"}]})
        if "faro" in text:
            return 'Claro:\n```json\n{"cases": [{"question": "¿Cuántos metros mide la torre del faro de Punta Brava?", "answer": "42 metros", "quote": "Su torre mide 42 metros y la luz alcanza 23 millas náuticas."}]}\n```'
        if "molino" in text:
            return json.dumps({"cases": [{"question": "¿Qué río mueve el molino de Aldea Vieja?", "answer": "El río Cerval", "quote": "lo mueve el agua del rio Cerval"}]})   # paraphrased: no accent
        return "no sé hacer esto"
    add_server(svc, "redactor", writer)
    out = tool(svc, "rag_cases_generate", set=sid, model="redactor", n=3, unanswerable=1, chunk_chars=400)
    assert out["proposed"] == 3 and out["discarded"]["unreadable"] >= 1 and out["counts"]["proposed"] == 3
    by_q = {c["question"]: c for c in out["cases"]}
    tower = by_q["¿Cuántos metros mide la torre del faro de Punta Brava?"]
    assert tower["status"] == "proposed" and tower["origin"] == "generated" and tower["expected"] == "42 metros"
    assert tower["passages"] == [{"doc": "faro.md", "text": "Su torre mide 42 metros y la luz alcanza 23 millas náuticas."}]
    river = by_q["¿Qué río mueve el molino de Aldea Vieja?"]
    assert "río Cerval" in river["passages"][0]["text"] and river["passages"][0]["doc"] == "molino.md"
    assert by_q["¿Cuánto pesa la linterna del faro de Punta Brava?"]["answerable"] is False
    again = tool(svc, "rag_cases_generate", set=sid, model="redactor", n=3, chunk_chars=400)
    assert again["proposed"] == 0 and again["discarded"]["duplicate"] >= 2                   # the same questions are not proposed twice
    assert tool(svc, "rag_cases_review", set=sid, status="rejected", ids=[river["id"]])["cases"]["rejected"] == 1
    tool(svc, "rag_cases_review", set=sid, status="accepted", all_proposed=True)
    assert run_now(svc, sid, SMALL)["summary"]["cases"] == 2
    assert G.reference_for("frase que no está", "Uno dos tres cuatro. Cinco seis siete ocho.") == ("Uno dos tres cuatro. Cinco seis siete ocho.", "fragment")


# ------------------------------------------------------------------ judges
def test_the_model_judge_and_the_logit_judge_score_the_answer(tmp_path):
    def judge_model(req):
        text = user_of(req)
        return json.dumps({"faithfulness": 2 if "Barbazul" in text else 10, "coverage": 0 if "Barbazul" in text else 9, "abstains": False, "unsupported": "Barbazul"})

    def http(request):
        body = json.loads(request.content)
        user = body["messages"][-1]["content"]
        invented = "Barbazul" in user
        if "supported by the PASSAGES" in user:
            probs = {"0": 0.7, "1": 0.2, "9": 0.05} if invented else {"9": 0.8, "8": 0.15}
        elif "EXPECTED ANSWER does the ANSWER contain" in user:
            probs = {"0": 0.9} if invented else {"9": 0.9}
        else:
            probs = {"A": 0.95, "B": 0.03}
        top = [{"token": k, "logprob": math.log(v)} for k, v in probs.items()]
        return json_response({"choices": [{"message": {"content": "A"}, "logprobs": {"content": [{"token": "A", "logprob": -0.1, "top_logprobs": top}]}}]})

    svc = build_services(tmp_path, clock=Clock(), client_factory=mock_client_factory(http))
    try:
        sid = make_set(svc, cases=CASES[:2])
        add_server(svc, "inventor", lambda req: "Lo construyó el pirata Barbazul en 1650.")
        add_server(svc, "lector", reader)
        add_server(svc, "juez", judge_model)
        written = run_now(svc, sid, {**SMALL, "generator": "model", "model": "inventor"}, judge="juez", settings={"judge_mode": "model"})
        assert written["summary"]["generation"]["judge_modes"] == {"model": 2} and written["summary"]["generation"]["faithfulness"] == 0.2
        assert written["summary"]["attribution"]["by_cause"] == {"invented": 2} and written["summary"]["generation"]["judge"] == "juez"
        logit = run_now(svc, sid, {**SMALL, "generator": "model", "model": "lector"}, judge={"kind": "server", "url": "http://127.0.0.1:9999", "model": "tiny"}, settings={"judge_mode": "auto"})
        gen = logit["summary"]["generation"]
        assert gen["judge_modes"] == {"logit": 2} and gen["faithfulness"] > 0.9 and gen["coverage"] == 1.0 and logit["summary"]["passed"] == 2
        bad = run_now(svc, sid, {**SMALL, "generator": "model", "model": "inventor"}, judge={"kind": "server", "url": "http://127.0.0.1:9999", "model": "tiny"}, settings={"judge_mode": "logit"})
        assert bad["summary"]["generation"]["faithfulness"] < 0.2 and bad["summary"]["attribution"]["by_cause"] == {"invented": 2}
        with pytest.raises(GaltonError) as err:
            svc.ragbench.run_start(set=sid, pipeline=SMALL, settings={"judge_mode": "logit"}, start_thread=False)
        assert err.value.key == "rb_judge_needs_model"
    finally:
        svc.db.close()


def test_a_server_without_logprobs_falls_back_and_an_unreadable_verdict_is_one_lexical_case():
    class NoLogits:
        def score(self, client, messages, labels):
            raise GaltonError("unsupported", "lj_no_logprobs", name="x")
    judge = GenerationJudge("auto", scorer=NoLogits(), client=None, ask=lambda messages: '{"faithfulness": 10, "coverage": 10, "abstains": false}', name="j")
    out = judge.judge("q", [{"text": "p"}], "e", "a")
    assert out["mode"] == "model" and judge.mode == "model" and judge.notes == ["lj_no_logprobs"]
    mute = GenerationJudge("auto", ask=lambda messages: "no tengo opinión", name="j")
    assert mute.judge("q", [{"text": "Fue en 1887."}], "En 1887.", "Fue en 1887.")["mode"] == "lexical" and mute.mode == "model"
    with pytest.raises(GaltonError):
        GenerationJudge("model", ask=lambda messages: "nada", name="j").judge("q", [], "e", "a")
    assert GenerationJudge("lexical").judge("q", [], "e", "")["empty"] is True


# ------------------------------------------------------------------ pipelines behind HTTP
def test_a_faustus_and_a_generic_http_pipeline_are_read_through_their_adapters(tmp_path):
    seen = []

    def http(request: httpx.Request):
        body = json.loads(request.content) if request.content else {}
        seen.append((request.method, str(request.url), body, request.headers.get("authorization", "")))
        if request.url.path == "/api/rag/answer":
            if "visitantes" in body["query"]:
                return json_response({"results": [], "answer": "Los documentos no lo dicen.", "model": "principal", "threshold": 0.35})
            rows = [{"id": "c1", "text": "Relleno sobre almacenes de sal.", "filename": "otro.md", "similarity": 0.52, "used": True},
                    {"id": "c2", "text": "El faro de Punta Brava se encendió por primera vez en 1887.", "filename": "faro.md", "similarity": 0.31, "used": False}]
            return json_response({"results": rows, "answer": "No lo sé: los documentos no lo dicen.", "model": "principal", "threshold": 0.35})
        if request.url.path == "/api/rag/search":
            return json_response({"results": [{"id": "c2", "text": "El faro de Punta Brava se encendió por primera vez en 1887.", "filename": "faro.md", "similarity": 0.8, "used": True}]})
        if request.url.path == "/query":
            if body.get("q", "").startswith("¿Quién"):
                return json_response({"detail": "index is rebuilding"}, 503)
            return json_response({"data": {"hits": [{"content": "El faro de Punta Brava se encendió por primera vez en 1887.", "metadata": {"source": "faro.md"}, "relevance": 0.9}],
                                           "reply": "Se encendió en 1887."}})
        return json_response({"answer": "Se encendió en 1887."})

    svc = build_services(tmp_path, clock=Clock(), client_factory=mock_client_factory(http))
    try:
        sid = make_set(svc, cases=[CASES[0], CASES[3]], docs=[])
        faustus = run_now(svc, sid, {"kind": "faustus", "url": "http://faustus.invalid:7000", "token": "tk", "depth": 8})
        row = faustus["cases"][0]
        assert (row["piece"], row["cause"], row["generator_note"]) == ("retriever", "cut_off", "abstained_rightly") and faustus["cases"][1]["ok"] is True
        assert faustus["run"]["pipeline"]["token_set"] is True and "tk" not in json.dumps(faustus["run"])
        assert seen[0][1].endswith("/api/rag/answer") and seen[0][2] == {"query": CASES[0]["question"], "k": 8, "source": "documents"} and seen[0][3] == "Bearer tk"
        add_server(svc, "lector", reader)
        local = svc.ragbench.probe(pipeline={"kind": "faustus", "url": "http://faustus.invalid:7000", "answer": "model", "model": "lector"}, question=CASES[0]["question"])
        assert local["answer"] == "Se encendió en 1887." and local["context"][0]["doc"] == "faro.md"
        generic = {"kind": "http", "url": "http://rag.invalid/query", "body": {"q": "{question}", "n": "{k}"}, "passages_path": "data.hits", "answer_path": "data.reply", "depth": 4}
        probe = tool(svc, "rag_probe", pipeline=generic, question=CASES[0]["question"])
        assert probe["traced"] is True and probe["retrieved"][0] == {"id": "1", "doc": "faro.md", "text": "El faro de Punta Brava se encendió por primera vez en 1887.", "score": 0.9}
        assert probe["answer"] == "Se encendió en 1887." and seen[-1][2] == {"q": CASES[0]["question"], "n": 4}
        two = make_set(svc, cases=CASES[:2], docs=[], name="dos")
        out = run_now(svc, two, generic)
        assert out["cases"][0]["ok"] is True and out["cases"][1]["cause"] == "error" and "503" in out["cases"][1]["error"] and out["summary"]["errors"] == 1
        blind = run_now(svc, sid, {"kind": "http", "url": "http://rag.invalid/answer-only"})
        assert blind["cases"][0]["ok"] is True and any(w["key"] == "rb_warn_untraced" for w in blind["summary"]["warnings"])
        with pytest.raises(GaltonError) as err:
            tool(svc, "rag_probe", pipeline={"kind": "http", "url": "http://rag.invalid/query", "answer_path": "nowhere"}, question="x")
        assert err.value.key == "rb_pipeline_no_answer"
        with pytest.raises(GaltonError) as err:
            tool(svc, "rag_probe", pipeline=SMALL, question="x", set=sid)
        assert err.value.key == "rb_no_corpus"
    finally:
        svc.db.close()


def test_a_pipeline_that_is_down_fails_the_run_instead_of_every_case(tmp_path):
    def down(request):
        raise httpx.ConnectError("refused")
    svc = build_services(tmp_path, clock=Clock(), client_factory=mock_client_factory(down))
    try:
        sid = make_set(svc, cases=[{**CASES[0], "question": f"{CASES[0]['question']} ({i})"} for i in range(8)], docs=[])
        out = run_now(svc, sid, {"kind": "http", "url": "http://rag.invalid/query"})
        assert out["run"]["state"] == "failed" and "did not answer" in out["run"]["error"] and len(out["cases"]) < 8
    finally:
        svc.db.close()


# ------------------------------------------------------------------ comparison
def test_two_runs_are_compared_case_by_case(svc):
    sid = make_set(svc)
    add_server(svc, "lector", reader)
    base = run_now(svc, sid, {**SMALL, "reranker": "reverse", "top_k": 1, "generator": "model", "model": "lector"})["run"]["id"]
    better = run_now(svc, sid, {**SMALL, "top_k": 2, "generator": "model", "model": "lector"})["run"]["id"]
    out = tool(svc, "rag_compare", a=base, b=better)
    assert (out["shared"], out["improved"], out["regressed"], out["same"]) == (4, 3, 0, 1) and out["verdict"] == "b_better_weak" and out["sign_test_p"] == 0.25
    assert out["differs"] == {"reranker": {"a": "reverse", "b": "none"}, "top_k": {"a": 1, "b": 2}}
    assert out["pieces"]["pass_rate"] == {"a": 0.25, "b": 1.0, "delta": 0.75} and out["pieces"]["rerank_recall_at_cut"]["delta"] == 1.0
    assert out["attribution"] == {"a": {"retriever": 0, "ranking": 3, "generator": 0, "unknown": 0}, "b": {"retriever": 0, "ranking": 0, "generator": 0, "unknown": 0}}
    assert out["moves"] == {"ranking>ok": 3} and [c["change"] for c in out["cases"]] == ["improved", "improved", "improved", "same_ok"]
    back = tool(svc, "rag_compare", a=better, b=base)
    assert back["regressed"] == 3 and back["verdict"] == "a_better_weak"
    other = make_set(svc, cases=CASES[:1], name="otro")                                     # another set: the same question is the same case
    third = run_now(svc, other, SMALL)["run"]["id"]
    assert tool(svc, "rag_compare", a=better, b=third)["shared"] == 1


# ------------------------------------------------------------------ tools, REST and threads
def test_sets_runs_and_deletes_through_the_tools(svc, tmp_path):
    folder = tmp_path / "corpus"
    folder.mkdir()
    (folder / "faro.md").write_text(DOCS[0]["text"], encoding="utf-8")
    (folder / "ignorado.bin").write_bytes(b"\x00\x01")
    made = tool(svc, "rag_set_save", name="desde carpeta", paths=[str(folder)], cases=[CASES[0]])
    sid = made["set"]["id"]
    assert [d["name"] for d in made["set"]["documents"]] == ["faro.md"] and made["set"]["cases"]["accepted"] == 1
    more = tool(svc, "rag_set_save", id=sid, documents=[DOCS[1], "Un texto suelto sobre el puerto y sus grúas."])
    assert [d["name"] for d in more["set"]["documents"]] == ["faro.md", "molino.md", "doc-2"]
    assert tool(svc, "rag_set_get", set="desde carpeta", with_text=True)["set"]["corpus"][0]["text"].startswith("El faro")
    for bad, key in (({"name": ""}, "rb_set_name"), ({"name": "x", "paths": ["relative/path"]}, "path_not_absolute"), ({"id": "rs_none", "documents": ["x"]}, "rb_no_set")):
        with pytest.raises(GaltonError) as err:
            tool(svc, "rag_set_save", **bad)
        assert err.value.key == key
    started = tool(svc, "rag_run_start", set=sid, pipeline=SMALL, judge_mode="lexical", wait_s=20)
    rid = started["run"]["id"]
    assert started["run"]["state"] == "done" and started["run"]["head"]["passed"] == 1 and "case_ids" not in started["run"]["settings"]
    assert tool(svc, "rag_run_status", run=rid)["run"]["done"] == 1 and tool(svc, "rag_run_list", set=sid)["count"] == 1
    assert tool(svc, "rag_run_result", run=rid, only_failed=True)["cases"] == []
    assert tool(svc, "rag_run_cancel", run=rid)["cancelled"] is False
    assert svc.ragbench.stop_all(5) == [] and not [t for t in threading.enumerate() if t.name.startswith("ragbench-") and t.is_alive()]
    with pytest.raises(GaltonError) as err:
        tool(svc, "rag_run_delete", run=rid)
    assert err.value.code == "confirm_required"
    assert tool(svc, "rag_run_delete", run=rid, confirm=True)["deleted"] == rid
    with pytest.raises(GaltonError) as err:
        tool(svc, "rag_run_result", run=rid)
    assert err.value.key == "rb_no_run"
    with pytest.raises(GaltonError):
        tool(svc, "rag_set_delete", set=sid)
    assert tool(svc, "rag_set_delete", set=sid, confirm=True)["deleted"] == sid and tool(svc, "rag_sets")["count"] == 0


def test_a_run_left_running_by_a_stopped_process_is_closed(svc):
    sid = make_set(svc)
    run = svc.ragbench.store.create_run(set_id=sid, set_name="obras", label="x", pipeline={}, settings=normalise_settings(LEX), total=4)
    svc.ragbench.store.update_run(run["id"], state="running")
    svc.ragbench.abandon_stale()
    assert svc.ragbench.store.run(run["id"])["state"] == "failed"
    assert tool(svc, "rag_run_cancel", run=run["id"])["cancelled"] is False


def test_rest_routes_call_the_same_handlers(client):
    made = client.post("/api/rag/sets", json={"name": "rest", "documents": DOCS, "cases": CASES}).json()
    sid = made["set"]["id"]
    assert client.get("/api/rag/sets").json()["count"] == 1 and len(client.get(f"/api/rag/sets/{sid}", params={"status": "accepted"}).json()["cases"]) == 4
    probe = client.post("/api/rag/probe", json={"set": sid, "pipeline": SMALL, "question": CASES[0]["question"]}).json()
    assert "1887" in probe["answer"]
    run = client.post("/api/rag/runs", json={"set": sid, "pipeline": SMALL, "judge_mode": "lexical", "wait_s": 20}).json()["run"]
    assert run["state"] == "done"
    assert client.get(f"/api/rag/runs/{run['id']}").json()["summary"]["cases"] == 4 and client.get(f"/api/rag/runs/{run['id']}/status").json()["run"]["done"] == 4
    assert client.get("/api/rag/runs", params={"set": sid}).json()["count"] == 1
    assert client.get("/api/rag/compare", params={"a": run["id"], "b": run["id"]}).json()["verdict"] == "no_clear_difference"
    case = client.post(f"/api/rag/sets/{sid}/cases", json={"question": "¿Cuánto mide la torre del faro?", "expected": "42 metros", "passages": [{"text": "Su torre mide 42 metros"}]}).json()
    assert client.post(f"/api/rag/sets/{sid}/review", json={"status": "rejected", "ids": [case["id"]]}).json()["cases"]["rejected"] == 1
    assert client.delete(f"/api/rag/cases/{case['id']}").status_code == 400 and client.delete(f"/api/rag/cases/{case['id']}", params={"confirm": True}).status_code == 200
    assert client.post(f"/api/rag/runs/{run['id']}/cancel").json()["cancelled"] is False
    assert client.delete(f"/api/rag/runs/{run['id']}", params={"confirm": True}).json()["deleted"] == run["id"]
    assert client.delete(f"/api/rag/sets/{sid}", params={"confirm": True}).json()["deleted"] == sid
    missing = client.get("/api/rag/sets/rs_none")
    assert missing.status_code == 404 and missing.json()["key"] == "rb_no_set"
