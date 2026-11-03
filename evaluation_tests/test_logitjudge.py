"""The logit judge: rubric, probability maths, the server backend (mock HTTP), the Hugging Face backend (fake worker result) and the worker's own logic."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest

from evaluation_tests.conftest import tool
from evaluation_tests.helpers import Clock, build_services, json_response, mock_client_factory
from galton_hoard.errors import GaltonError
from galton_hoard.logitjudge import rubric as R, scoring as S
from galton_hoard.logitjudge.scorers import label_of, labels_from_top, top_of_reply

RUBRIC = [
    {"id": "tone", "type": "choice", "instructions": "What is the tone?", "options": ["formal", "casual", "rude"]},
    {"id": "quality", "type": "score", "instructions": "How good is the text?", "min": 0, "max": 9},
]
ITEMS = [{"id": "a", "context": "Dear sir, we regret to inform you."}, {"id": "b", "context": "hey whats up lol"}]
SERVER = {"kind": "server", "url": "http://127.0.0.1:9999", "model": "tiny"}


def lp(p: float) -> float:
    return math.log(p)


def top(**probs: float) -> list[dict]:
    return [{"token": k, "logprob": lp(v)} for k, v in probs.items()]


def reply(top_list, *, field="content"):
    return {"choices": [{"message": {"content": "A"}, "logprobs": {field: [{"token": "A", "logprob": -0.1, "top_logprobs": top_list}]}}]}


def responder(request):
    """A server that answers 'formal' for the first item and 'casual' for the second, and a score near 7 or 2."""
    body = json.loads(request.content)
    user = body["messages"][-1]["content"]
    formal = "Dear sir" in user
    if "What is the tone?" in user:
        probs = {"A": 0.7, " B": 0.1, "C": 0.05} if formal else {"A": 0.1, "B": 0.75, "C": 0.1}
    else:
        probs = {"7": 0.6, "8": 0.2, "6": 0.1, "0": 0.05} if formal else {"2": 0.5, "1": 0.3, "3": 0.1, "9": 0.05}
    return json_response(reply(top(**probs)))


@pytest.fixture
def lsvc(tmp_path):
    service = build_services(tmp_path, clock=Clock(), client_factory=mock_client_factory(responder))
    yield service
    service.db.close()


# ------------------------------------------------------------------ rubric and maths
def test_rubric_gives_labels_and_rejects_bad_input():
    rubric = R.normalise_rubric(RUBRIC)
    assert rubric[0]["labels"] == ["A", "B", "C"] and rubric[1]["labels"] == [str(i) for i in range(10)]
    for bad, key in (([], "lj_rubric_empty"), ([{"id": "x", "type": "choice", "instructions": "?", "options": ["only"]}], "lj_options_needed"),
                     ([{"id": "x", "type": "score", "instructions": "?", "min": 0, "max": 12}], "lj_scale"),
                     ([{"id": "x", "type": "essay", "instructions": "?"}], "lj_question_type"),
                     ([RUBRIC[0], RUBRIC[0]], "lj_question_id")):
        with pytest.raises(GaltonError) as err:
            R.normalise_rubric(bad)
        assert err.value.key == key
    messages = R.build_messages("ctx", rubric[0])
    assert "A) formal" in messages[1]["content"] and "ctx" in messages[1]["content"]


def test_labels_from_top_sums_spellings_and_marks_missing():
    out = labels_from_top(top(A=0.4) + [{"token": " A", "logprob": lp(0.2)}, {"token": "ĠB", "logprob": lp(0.1)}], ["A", "B", "C"])
    assert math.exp(out["A"]) == pytest.approx(0.6) and math.exp(out["B"]) == pytest.approx(0.1) and out["C"] is None
    assert label_of("▁C") == "C"


def test_answer_for_choice_and_score():
    rubric = R.normalise_rubric(RUBRIC)
    choice = S.answer_for(rubric[0], {"A": lp(0.6), "B": lp(0.2), "C": None})
    assert choice["chosen"] == "A" and choice["text"] == "formal" and choice["probs"]["A"] == pytest.approx(0.75) and choice["mass"] == pytest.approx(0.8)
    score = S.answer_for(rubric[1], {str(v): (lp(0.5) if v in (2, 4) else None) for v in range(10)})
    assert score["expected"] == pytest.approx(3.0) and score["value"] in (2, 4)
    assert S.answer_for(rubric[0], {"A": lp(0.01), "B": lp(0.01), "C": lp(0.01)})["low_mass"] is True


def test_top_of_reply_variants():
    assert top_of_reply(reply(top(A=0.5))) == top(A=0.5) or len(top_of_reply(reply(top(A=0.5)))) == 1
    legacy = {"choices": [{"text": "A", "logprobs": {"top_logprobs": [{"A": -0.2, "B": -2.0}]}}]}
    assert {e["token"] for e in top_of_reply(legacy)} == {"A", "B"}
    assert top_of_reply({"choices": [{"message": {"content": "A"}, "logprobs": None}]}) is None
    assert top_of_reply({"nothing": 1}) is None


# ------------------------------------------------------------------ the server backend
def test_server_run_reads_probabilities_and_is_stored_as_a_run(lsvc):
    out = tool(lsvc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[SERVER], detail=True)
    assert out["run"].startswith("lj_") and len(out["sets"]) == 1 and out["compare"] == []
    first, second = out["sets"][0]["items"]
    assert first["answers"]["tone"]["chosen"] == "A" and first["answers"]["tone"]["text"] == "formal"
    assert second["answers"]["tone"]["chosen"] == "B"
    assert first["answers"]["quality"]["expected"] > 6 and second["answers"]["quality"]["expected"] < 3
    agg = out["sets"][0]["aggregate"]
    assert agg["quality"]["n"] == 2 and sum(agg["tone"]["mean_probs"].values()) == pytest.approx(1.0)
    slim = tool(lsvc, "logit_judge_result", run=out["run"])
    assert "items" not in slim["sets"][0] and slim["run"]["state"] == "done"
    assert tool(lsvc, "logit_judge_list")["runs"][0]["id"] == out["run"]
    assert any(r["id"] == out["run"] and r["source"] == "logit_judge" for r in lsvc.store.runs(limit=10))
    assert tool(lsvc, "runs_list")["runs"][0]["id"] == out["run"]


def test_two_servers_are_compared_by_deltas(lsvc):
    out = tool(lsvc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[SERVER, {**SERVER, "url": "http://127.0.0.1:9998", "model": "tiny2"}])
    assert len(out["sets"]) == 2 and len(out["compare"]) == 1
    assert out["compare"][0]["deltas"]["quality"]["expected"]["delta"] == pytest.approx(0.0)


def test_dry_run_checks_without_asking(lsvc):
    out = tool(lsvc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[SERVER], dry_run=True)
    assert out["requests_per_set"] == 4 and out["sets"] == ["tiny"]
    assert lsvc.store.runs(source="logit_judge") == []


def test_server_without_logprobs_is_reported(tmp_path):
    svc = build_services(tmp_path, clock=Clock(), client_factory=mock_client_factory(lambda r: json_response({"choices": [{"message": {"content": "A"}}]})))
    try:
        with pytest.raises(GaltonError) as err:
            tool(svc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[SERVER])
        assert err.value.key == "lj_no_logprobs" and "ignores the logprobs request" in err.value.hint
        assert svc.store.runs(source="logit_judge")[0]["state"] == "failed"
        probe = tool(svc, "logit_judge_probe", targets=[SERVER])["servers"][0]
        assert probe["supported"] is False and probe["problem"]["key"] == "lj_no_logprobs"
    finally:
        svc.db.close()


def test_probe_of_a_good_server_and_retry_without_template_flag(tmp_path):
    seen = []

    def handler(request):
        body = json.loads(request.content)
        seen.append("chat_template_kwargs" in body)
        if "chat_template_kwargs" in body:
            return json_response({"error": "unknown field"}, 400)
        return json_response(reply(top(A=0.6, B=0.3)))

    svc = build_services(tmp_path, clock=Clock(), client_factory=mock_client_factory(handler))
    try:
        probe = tool(svc, "logit_judge_probe", targets=[SERVER])["servers"][0]
        assert probe["supported"] is True and seen == [True, False]
    finally:
        svc.db.close()


def test_remote_server_needs_permission_and_labels_must_appear(lsvc):
    with pytest.raises(GaltonError) as err:
        tool(lsvc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[{"kind": "server", "url": "http://192.0.2.10:8080", "model": "far"}])
    assert err.value.code == "remote_not_allowed"
    from galton_hoard.logitjudge.scorers import ServerScorer
    client = mock_client_factory(lambda r: json_response(reply(top(Z=0.9))))()
    with pytest.raises(GaltonError) as err:
        ServerScorer({"name": "n", "url": "http://127.0.0.1:1", "model": "m"}).score(client, [{"role": "user", "content": "?"}], ["A", "B"])
    assert err.value.key == "lj_labels_not_in_top"


# ------------------------------------------------------------------ the Hugging Face backend
def test_hf_base_vs_adapter_deltas(lsvc, tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "lora"
    base.mkdir(), adapter.mkdir()
    (base / "config.json").write_text("{}"), (adapter / "adapter_config.json").write_text("{}")
    calls = []

    def fake_worker(args):
        calls.append(args)
        assert args["variants"] == ["base", "adapter"] and args["adapter_path"] == str(adapter)
        out = {"base": {}, "adapter": {}}
        for r in args["requests"]:
            labels = r["labels"]
            for variant, favoured in (("base", labels[0]), ("adapter", labels[-1])):
                out[variant][r["key"]] = {label: lp(0.7 if label == favoured else 0.3 / (len(labels) - 1)) for label in labels}
        return out

    lsvc.logit.hf_runner = fake_worker
    out = tool(lsvc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[{"kind": "hf", "base_path": str(base), "adapter_path": str(adapter), "name": "tiny-lm"}], detail=True)
    assert len(calls) == 1, "the worker loads the model once for both variants"
    assert [s["label"] for s in out["sets"]] == ["tiny-lm (base)", "tiny-lm (+ lora)"]
    deltas = out["compare"][0]["deltas"]
    assert deltas["quality"]["expected"]["delta"] > 4 and deltas["tone"]["flips"] == 2
    assert deltas["tone"]["probs"]["C"]["delta"] > 0.5 > 0 > deltas["tone"]["probs"]["A"]["delta"]


def test_hf_paths_are_checked(lsvc, tmp_path):
    for target, key in (({"kind": "hf", "base_path": "relative"}, "lj_hf_path"), ({"kind": "hf", "base_path": str(tmp_path)}, "lj_hf_missing")):
        with pytest.raises(GaltonError) as err:
            tool(lsvc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[target])
        assert err.value.key == key


def test_hf_without_python_says_so(lsvc, tmp_path, monkeypatch):
    import pygmalion_hoard.settings as pyg_settings
    monkeypatch.setattr(pyg_settings, "WIN_PYTHON", str(tmp_path / "no-python.exe"))
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text("{}")
    with pytest.raises(GaltonError) as err:
        tool(lsvc, "logit_judge_run", rubric=RUBRIC, items=ITEMS, targets=[{"kind": "hf", "base_path": str(base)}])
    assert err.value.key == "lj_no_python"


# ------------------------------------------------------------------ the worker
def load_worker():
    path = Path(__file__).resolve().parent.parent / "pygmalion_hoard" / "workers" / "logit_judge.py"
    spec = importlib.util.spec_from_file_location("logit_judge_worker", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    chat_template = "x"
    vocab = {"A": 1, " A": 2, "B": 3, "0": 4, "1": 5}

    def encode(self, text, add_special_tokens=False):
        return [self.vocab[text]] if text in self.vocab else [90, 91]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, **kw):
        return "|".join(m["content"] for m in messages)


class FakeEngine:
    tokenizer = FakeTokenizer()

    def __init__(self):
        self.entered = []

    def variant(self, name):
        import contextlib
        self.entered.append(name)
        return contextlib.nullcontext()

    def last_logprobs(self, texts, ids):
        return [{i: lp(0.1 * (n + 1) / len(ids)) for i in ids} for n, _ in enumerate(texts)]


def test_worker_resolves_label_tokens_and_folds_spellings():
    w = load_worker()
    ids = w.candidate_ids(FakeTokenizer(), ["A", "B"])
    assert ids == {"A": [1, 2], "B": [3]}
    folded = w.fold(ids, {1: lp(0.2), 2: lp(0.1), 3: lp(0.4)})
    assert math.exp(folded["A"]) == pytest.approx(0.3) and math.exp(folded["B"]) == pytest.approx(0.4)
    assert w.fold({"A": [1]}, {}) == {"A": None}
    with pytest.raises(w.C.WorkerError):
        w.candidate_ids(FakeTokenizer(), ["ZZ"])


def test_worker_scores_every_request_for_every_variant(capsys):
    w = load_worker()
    engine = FakeEngine()
    args = {"variants": ["base", "adapter"], "adapter_path": "x", "batch": 2,
            "requests": [{"key": f"k{i}", "messages": [{"role": "user", "content": f"q{i}"}], "labels": ["A", "B"]} for i in range(3)]}
    w.main(args, engine=engine)
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
    result = next(e for e in lines if e["event"] == "result")["data"]
    assert engine.entered == ["base", "adapter"] and set(result) == {"base", "adapter"} and set(result["base"]) == {"k0", "k1", "k2"}
    assert set(result["base"]["k0"]) == {"A", "B"} and sum(1 for e in lines if e["event"] == "progress") == 4
