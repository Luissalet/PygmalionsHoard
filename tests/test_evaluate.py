"""Evaluation against the parent with Galton: the hub path, the direct fallback, polling, cancel, verdicts and the long-context needle table.

``FakeGalton`` answers with the shapes Galton's own tools return (run card under ``run``, ``progress.pct``, ``verdict`` of a compare)."""

import json
import threading

import httpx
import pytest

from conftest import call
from gguf_builder import build_gguf
from pygmalion_hoard import evaluate as E
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.evaluate import Galton


class FakeGalton:
    def __init__(self, *, verdict="better", states=("running", "done"), contestants=("m_parent", "m_child"), rows=None, fail_tool=None):
        self.calls = []
        self.verdict = verdict
        self.states = list(states)
        self.contestants = list(contestants)
        self.rows = rows or []
        self.fail_tool = fail_tool
        self.polled = 0

    def card(self, state, pct):
        return {"id": "r_1", "state": state, "error": "boom" if state == "failed" else "", "progress": {"done": 1, "total": 2, "pct": pct},
                "contestants": [{"id": c, "name": c} for c in self.contestants], "suites": []}

    def handle(self, tool, args):
        self.calls.append((tool, args))
        if tool == self.fail_tool:
            raise PygmalionError("galton_unavailable", "down")
        if tool == "galton_overview":
            return {"summary": "ok"}
        if tool == "run_start":
            return {"run": self.card("queued", 0.0), "warnings": []}
        if tool == "run_status":
            state = self.states[min(self.polled, len(self.states) - 1)]
            self.polled += 1
            return {"run": self.card(state, 100.0 if state == "done" else 40.0)}
        if tool == "run_cancel":
            return {"cancelled": True}
        if tool == "compare":
            return {"scope": "suites", "n": 24, "verdict": self.verdict, "p_value": 0.012, "diff": 0.081, "diff_ci": [0.02, 0.14], "wins": 12, "losses": 4,
                    "ties": 8, "warnings": ["Only 24 cases"], "sentence": "x", "per_case": [{"case_id": "c1"}]}
        if tool == "run_results":
            return {"run": "r_1", "results": self.rows, "count": len(self.rows)}
        raise AssertionError(f"unexpected tool {tool}")

    def family_call(self, app, tool, args=None, timeout=60.0):
        assert app == "galton"
        try:
            return {"ok": True, "app": app, "tool": tool, "status": 200, "result": self.handle(tool, args or {})}
        except PygmalionError as exc:
            return {"ok": False, "app": app, "tool": tool, "status": 502, "error": exc.message}

    def args_of(self, tool):
        return [a for t, a in self.calls if t == tool]


@pytest.fixture
def pair(ctx, tmp_path):
    """A parent GGUF and the child made from it."""
    parent_file = build_gguf(tmp_path / "parent.gguf", context=2048, size=4096)
    child_file = build_gguf(tmp_path / "child.gguf", context=2048, size=4096)
    store = ctx.svc.store
    parent = store.create_artifact("gguf", "parent-f16", path=str(parent_file), size=4096)
    child = store.create_artifact("gguf", "child-f16", path=str(child_file), size=4096, parents=[parent["id"]])
    return parent, child


def attach(ctx, fake, **kw):
    ctx.svc.galton.family_call = fake.family_call
    ctx.svc.galton.offline = False
    ctx.svc.galton.sleep = lambda s: None
    return fake


def run(ctx, child, **kw):
    return ctx.svc.evaluator.run(child["id"], **kw)


# ------------------------------------------------------------------------------------------------ pure helpers
def test_verdicts_are_normalised():
    assert [E.normalise_verdict(v) for v in ("better", "Worse", "no clear difference", "no_clear_difference", "no-data", "???", None)] == \
        ["better", "worse", "no_clear_difference", "no_clear_difference", "no_data", "unknown", "unknown"]


def test_first_key_looks_into_wrappers():
    assert E.first_key({"run": {"id": "r_9"}}, "id") == "r_9"
    assert E.first_key({"a": 1}, "b") is None and E.first_key("text", "a") is None
    assert E.first_key({"id": "", "run": {"id": "x"}}, "id") == "x"


def test_suites_follow_the_intent_the_dataset_or_the_context():
    assert E.pick_suites() == ["escritura-es", "instrucciones"]
    assert E.pick_suites(intent="code") == ["codigo-python"]
    assert E.pick_suites(suites=["a", "b"], intent="code") == ["a", "b"]
    assert E.pick_suites(context_variant=True) == ["contexto-largo"]
    assert E.pick_suites(dataset_kind="other") == ["razonamiento", "instrucciones", "escritura-es"]
    with pytest.raises(PygmalionError) as exc:
        E.pick_suites(intent="astrology")
    assert exc.value.code == "invalid" and "style" in exc.value.hint


def test_contestant_specs_only_for_gguf_and_ollama():
    assert E.contestant_spec({"kind": "gguf", "path": "/m/a.gguf", "name": "a"}) == {"kind": "gguf", "path": "/m/a.gguf", "name": "a"}
    assert E.contestant_spec({"kind": "ollama", "path": "pyg-a:v1", "name": "pyg-a:v1"}) == {"kind": "ollama", "model": "pyg-a:v1", "name": "pyg-a:v1"}
    with pytest.raises(PygmalionError) as exc:
        E.contestant_spec({"kind": "adapter", "path": "/x", "name": "ad"})
    assert "convert_start" in exc.value.hint


def test_summarise_compare_keeps_what_galton_calls_things():
    raw = {"run": 1, "verdict": "worse", "p_value": 0.03, "n": 40, "diff": -0.1, "diff_ci": [-0.2, -0.01], "wins": 3, "losses": 20, "ties": 17,
           "warnings": ["w"], "scope": "suite", "per_case": [1, 2, 3]}
    out = E.summarise_compare(raw)
    assert out == {"verdict": "worse", "p_value": 0.03, "n": 40, "warnings": ["w"], "diff": -0.1, "diff_ci": [-0.2, -0.01], "wins": 3, "losses": 20, "ties": 17, "scope": "suite"}
    assert E.summarise_compare({})["verdict"] == "unknown" and E.summarise_compare({})["warnings"] == []


# ------------------------------------------------------------------------------------------------ plan
def test_plan_picks_the_nearest_gguf_ancestor(ctx, pair):
    parent, child = pair
    plan = ctx.svc.evaluator.plan(child["id"])
    assert plan["parent"]["id"] == parent["id"] and plan["suites"] == ["escritura-es", "instrucciones"] and plan["context_variant"] is False


def test_plan_without_a_reference_explains_what_to_pass(ctx):
    lone = ctx.svc.store.create_artifact("gguf", "lone", path="/m/x.gguf")
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.evaluator.plan(lone["id"])
    assert exc.value.code == "not_found" and "against" in exc.value.hint


def test_plan_refuses_an_adapter_and_accepts_an_explicit_reference(ctx, pair, base_model):
    parent, child = pair
    with pytest.raises(PygmalionError):
        ctx.svc.evaluator.plan(base_model["id"])
    other = ctx.svc.store.create_artifact("gguf", "other", path="/m/o.gguf")
    assert ctx.svc.evaluator.plan(child["id"], against=other["id"])["parent"]["id"] == other["id"]


def test_plan_uses_the_context_suite_below_a_context_variant(ctx, pair, base_model):
    parent, child = pair
    variant = ctx.svc.store.create_artifact("ctx_variant", "v", path=str(base_model["path"]), parents=[base_model["id"]])
    gg = ctx.svc.store.create_artifact("gguf", "v-gguf", path=child["path"], parents=[variant["id"]])
    plan = ctx.svc.evaluator.plan(gg["id"], against=parent["id"])
    assert plan["context_variant"] is True and plan["suites"] == ["contexto-largo"]


# ------------------------------------------------------------------------------------------------ run through the hub
def test_a_run_through_the_hub_stores_the_verdict_in_the_lineage(ctx, pair):
    parent, child = pair
    fake = attach(ctx, FakeGalton())
    record = run(ctx, child, intent="style")
    assert record["verdict"] == "better" and record["promote_suggested"] is True and record["run"] == "r_1" and record["parent"] == parent["id"]
    assert record["diff"] == 0.081 and record["n"] == 24 and record["warnings"] == ["Only 24 cases"]
    start = fake.args_of("run_start")[0]
    assert start["suites"] == ["escritura-es", "instrucciones"] and start["settings"] == {"temperature": 0, "repeats": 1}
    assert [s["kind"] for s in start["contestants"]] == ["gguf", "gguf"] and start["contestants"][0]["path"] == parent["path"] and start["contestants"][1]["path"] == child["path"]
    assert fake.args_of("run_status") == [{"run": "r_1"}, {"run": "r_1"}]
    stored = ctx.svc.store.artifact(child["id"])["metrics"]["galton"]
    assert stored["verdict"] == "better" and stored["suites"] == start["suites"]
    assert ctx.svc.lineage.card(ctx.svc.store.artifact(child["id"]))["verdict"] == "better"


def test_compare_asks_for_the_new_model_against_its_parent(ctx, pair):
    parent, child = pair
    fake = attach(ctx, FakeGalton(contestants=("m_parent", "m_child")))
    run(ctx, child, intent="style")
    assert fake.args_of("compare") == [{"a": "m_child", "b": "m_parent", "include_stale": False}]


def test_compare_is_limited_to_the_suite_only_when_there_is_one(ctx, pair):
    _parent, child = pair
    fake = attach(ctx, FakeGalton())
    run(ctx, child, suites=["codigo-python"])
    assert fake.args_of("compare")[0]["suite"] == "codigo-python"
    fake2 = attach(ctx, FakeGalton())
    run(ctx, child, suites=["a", "b"])
    assert "suite" not in fake2.args_of("compare")[0]


def test_a_worse_result_is_not_suggested_for_promotion(ctx, pair):
    _parent, child = pair
    attach(ctx, FakeGalton(verdict="worse"))
    record = run(ctx, child)
    assert record["verdict"] == "worse" and record["promote_suggested"] is False


def test_progress_is_reported_from_galtons_percentage(ctx, pair):
    _parent, child = pair
    attach(ctx, FakeGalton(states=("running", "running", "done")))
    seen = []
    run(ctx, child, progress=lambda **kw: seen.append((kw["step"], kw["state"])))
    assert seen == [(40.0, "running"), (40.0, "running"), (100.0, "done")]


def test_a_failed_run_is_an_error_with_galtons_message(ctx, pair):
    _parent, child = pair
    attach(ctx, FakeGalton(states=("failed",)))
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "failed" in exc.value.message and "boom" in exc.value.message


def test_a_run_that_never_finishes_times_out(ctx, pair):
    _parent, child = pair
    attach(ctx, FakeGalton(states=("running",)))
    now = {"t": 0.0}

    def clock():
        now["t"] += 1000.0
        return now["t"]
    ctx.svc.galton.clock = clock
    ctx.svc.settings.set({"galton.timeout_s": 30})
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "did not finish" in exc.value.message and "galton.timeout_s" in exc.value.hint


def test_cancel_stops_the_galton_run_too(ctx, pair):
    _parent, child = pair
    fake = attach(ctx, FakeGalton(states=("running",)))
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child, cancel=cancel)
    assert exc.value.message == "Cancelled." and fake.args_of("run_cancel") == [{"run": "r_1"}]


def test_a_run_with_the_wrong_number_of_models_is_refused(ctx, pair):
    _parent, child = pair
    attach(ctx, FakeGalton(contestants=("only_one",)))
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "1 models instead of 2" in exc.value.message


def test_the_hub_refusing_the_arguments_is_a_clear_error(ctx, pair):
    _parent, child = pair

    def refuse(app, tool, args=None, timeout=60.0):
        return {"ok": False, "status": 422, "error": "Unknown suites: nope"}
    ctx.svc.galton.family_call = refuse
    ctx.svc.galton.offline = False
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child, suites=["nope"])
    assert "refused run_start" in exc.value.message and "Unknown suites" in exc.value.message


# ------------------------------------------------------------------------------------------------ the long-context table
def test_context_variant_runs_set_the_window_and_build_the_needle_table(ctx, base_model, tmp_path):
    parent = ctx.svc.store.create_artifact("gguf", "base-f16", path=str(build_gguf(tmp_path / "p.gguf")), size=1, parents=[base_model["id"]])
    variant = ctx.svc.store.create_artifact("ctx_variant", "v4x", path=base_model["path"], parents=[base_model["id"]])
    child = ctx.svc.store.create_artifact("gguf", "v4x-f16", path=str(build_gguf(tmp_path / "c.gguf", context=8192)), size=1, parents=[variant["id"]],
                                          metrics={"gguf": {"context_length": 8192}})
    rows = [{"contestant": c, "title": f"Aguja a {k}k tokens {i}", "score": s} for k, per in ((4, ((1.0, 1.0), (1.0, 0.0))), (8, ((0.0, 1.0),)))
            for i, (a, b) in enumerate(per) for c, s in (("m_parent", a), ("m_child", b))]
    fake = attach(ctx, FakeGalton(rows=rows))
    record = ctx.svc.evaluator.run(child["id"], against=parent["id"])
    assert fake.args_of("run_start")[0]["settings"]["context"] == 8192 and fake.args_of("run_start")[0]["suites"] == ["contexto-largo"]
    assert fake.args_of("run_results") == [{"run": "r_1", "suite": "contexto-largo", "include_output": False, "limit": 500}]
    assert record["needle"] == {"4k": {"m_parent": 1.0, "m_child": 0.5}, "8k": {"m_parent": 0.0, "m_child": 1.0}}
    assert record["contestants"] == {"parent": "m_parent", "child": "m_child"}


def test_a_missing_needle_table_does_not_fail_the_evaluation(ctx, base_model, tmp_path):
    variant = ctx.svc.store.create_artifact("ctx_variant", "v", path=base_model["path"], parents=[base_model["id"]])
    child = ctx.svc.store.create_artifact("gguf", "v-gguf", path=str(build_gguf(tmp_path / "c.gguf")), parents=[variant["id"]])
    parent = ctx.svc.store.create_artifact("gguf", "b-gguf", path=str(build_gguf(tmp_path / "b.gguf")))
    attach(ctx, FakeGalton(fail_tool="run_results"))
    record = ctx.svc.evaluator.run(child["id"], against=parent["id"])
    assert record["needle"] is None and record["verdict"] == "better"


# ------------------------------------------------------------------------------------------------ direct fallback
def galton_server(calls, answer=None, status=200):
    fake = answer or FakeGalton()

    def handler(request):
        calls.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
        body = json.loads(request.content)
        if status != 200:
            return httpx.Response(status, json={"error": "token rejected"})
        return httpx.Response(200, json=fake.handle(body["name"], body["arguments"]))
    return httpx.MockTransport(handler)


def test_direct_fallback_uses_galtons_port_and_token(ctx, pair, tmp_path):
    _parent, child = pair
    token = tmp_path / "galton-token"
    token.write_text("secret-from-galton\n", encoding="utf-8")
    ctx.svc.settings.set({"galton.token_file": str(token), "galton.url": "http://127.0.0.1:5999"})
    calls = []
    ctx.svc.galton.transport = galton_server(calls)
    ctx.svc.galton.offline = False
    ctx.svc.galton.family_call = None
    ctx.svc.galton.sleep = lambda s: None
    record = run(ctx, child)
    assert record["verdict"] == "better"
    assert {c[0] for c in calls} == {"http://127.0.0.1:5999/api/agent/call"} and {c[1] for c in calls} == {"Bearer secret-from-galton"}
    assert [c[2]["name"] for c in calls][:2] == ["run_start", "run_status"] and all(c[2]["caller"] == "pygmalion" for c in calls)


def test_direct_fallback_when_the_hub_has_no_answer(ctx, pair, tmp_path):
    _parent, child = pair
    token = tmp_path / "t"
    token.write_text("tok", encoding="utf-8")
    ctx.svc.settings.set({"galton.token_file": str(token)})
    calls = []
    ctx.svc.galton.transport = galton_server(calls)
    ctx.svc.galton.offline = False
    ctx.svc.galton.family_call = lambda *a, **k: {"ok": False, "status": 502, "error": "hub down"}
    ctx.svc.galton.sleep = lambda s: None
    assert run(ctx, child)["verdict"] == "better" and calls


def test_direct_call_without_a_token_says_which_setting_to_fill(ctx, pair):
    _parent, child = pair
    ctx.svc.galton.offline = False
    ctx.svc.galton.family_call = None
    ctx.svc.galton._registry_token_file = lambda: ""
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert exc.value.code == "galton_unavailable" and "galton.token_file" in exc.value.hint


def test_direct_call_reports_galtons_own_rejection(ctx, pair, tmp_path):
    _parent, child = pair
    token = tmp_path / "t"
    token.write_text("bad", encoding="utf-8")
    ctx.svc.settings.set({"galton.token_file": str(token)})
    ctx.svc.galton.transport = galton_server([], status=401)
    ctx.svc.galton.offline = False
    ctx.svc.galton.family_call = None
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "401" in exc.value.message and "token rejected" in exc.value.message


def test_direct_call_when_galton_is_not_running(ctx, pair, tmp_path):
    _parent, child = pair
    token = tmp_path / "t"
    token.write_text("x", encoding="utf-8")
    ctx.svc.settings.set({"galton.token_file": str(token)})

    def boom(request):
        raise httpx.ConnectError("refused")
    ctx.svc.galton.transport = httpx.MockTransport(boom)
    ctx.svc.galton.offline = False
    ctx.svc.galton.family_call = None
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "Cannot reach Galton" in exc.value.message and "Start Galton" in exc.value.hint


def test_the_token_file_is_found_through_the_hub_registry(ctx, tmp_path, monkeypatch):
    token = tmp_path / "found-token"
    token.write_text("from-registry", encoding="utf-8")
    snapshot = {"apps": [{"id": "kafka", "token_file": "/x"}, {"id": "galton", "token_file": str(token), "data_dir": str(tmp_path)}]}
    monkeypatch.setattr(E._hubclient, "fetch", lambda url, **kw: (200, snapshot))
    ctx.svc.galton.offline = False
    assert ctx.svc.galton.token() == "from-registry"
    monkeypatch.setattr(E._hubclient, "fetch", lambda url, **kw: (None, None))
    assert ctx.svc.galton.token() == "from-registry"         # the path found once is kept while the hub is away
    token.unlink()
    monkeypatch.setattr(E, "SIBLING_TOKEN", tmp_path / "nowhere" / "mcp-token")
    assert ctx.svc.galton.token() == ""


def test_the_registry_falls_back_to_the_apps_data_folder(ctx, tmp_path, monkeypatch):
    (tmp_path / "gdata").mkdir()
    (tmp_path / "gdata" / "mcp-token").write_text("by-folder", encoding="utf-8")
    monkeypatch.setattr(E._hubclient, "fetch", lambda url, **kw: (200, [{"app": "galton", "data_dir": str(tmp_path / "gdata")}]))
    ctx.svc.galton.offline = False
    assert ctx.svc.galton.token() == "by-folder"


# ------------------------------------------------------------------------------------------------ availability and tools
def test_availability_through_the_hub_then_directly_then_offline(ctx, tmp_path):
    g = ctx.svc.galton
    assert g.available() == {"ok": False, "via": None, "detail": "offline"}
    g.offline = False
    g.family_call = FakeGalton().family_call
    assert g.available() == {"ok": True, "via": "hub", "detail": ""}
    token = tmp_path / "t"
    token.write_text("tok", encoding="utf-8")
    ctx.svc.settings.set({"galton.token_file": str(token)})
    g.family_call = None
    g.transport = galton_server([])
    assert g.available() == {"ok": True, "via": "direct", "detail": ""}
    g.transport = galton_server([], status=500)
    out = g.available()
    assert out["ok"] is False and "500" in out["detail"]


def test_evaluate_start_queues_a_job_and_the_runner_finishes_it(ctx, pair):
    parent, child = pair
    attach(ctx, FakeGalton())
    answer = call(ctx.svc, "evaluate_start", artifact=child["id"], intent="code")
    assert {k: answer["plan"][k] for k in ("child", "parent", "suites", "intent", "dataset")} == \
        {"child": "child-f16", "parent": "parent-f16", "suites": ["codigo-python"], "intent": "code", "dataset": None}
    assert answer["plan"]["reference"]["mode"] == "parent" and answer["plan"]["reference"]["ready"] is True, "no training behind it: its f16 parent is the reference"
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["state"] == "done" and job["result"]["verdict"] == "better" and job["lane"] == "gpu"
    assert not ctx.leases.requests, "evaluation holds no lease: Galton leases what its servers need"


def test_evaluate_start_refuses_before_queueing_when_there_is_nothing_to_compare(ctx):
    lone = ctx.svc.store.create_artifact("gguf", "lone", path="/m/x.gguf")
    with pytest.raises(PygmalionError):
        call(ctx.svc, "evaluate_start", artifact=lone["id"])
    assert not ctx.svc.store.jobs(kind="evaluate")


def test_a_model_galton_could_not_run_is_an_error_not_a_verdict(ctx, pair):
    _parent, child = pair
    fake = FakeGalton()
    original = fake.card

    def card(state, pct):
        out = original(state, pct)
        if state == "done":
            out["contestants"][1].update(state="failed", error="No allowed GPU has 18.2 GB free.")
        return out
    fake.card = card
    attach(ctx, fake)
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "child" in exc.value.message and "18.2 GB" in exc.value.message and not fake.args_of("compare")


def test_no_data_is_not_suggested_for_promotion(ctx, pair):
    _parent, child = pair
    attach(ctx, FakeGalton(verdict="no data"))
    record = run(ctx, child)
    assert record["verdict"] == "no_data" and record["promote_suggested"] is False


# ------------------------------------------------------------------------------------------------ the dataset's own held-out records
class DatasetGalton(FakeGalton):
    """A Galton that also keeps suites: ``suites_list``, ``suite_create``, ``cases_import`` and ``suite_remove``, and a judge setting."""

    def __init__(self, *, judge="qwen3:8b", existing=None, imported=None, **kw):
        super().__init__(**kw)
        self.judge = judge
        self.suites = dict(existing or {})
        self.imported = imported
        self.rows = []

    def handle(self, tool, args):
        if tool == "galton_status":
            self.calls.append((tool, args))
            return {"judge": self.judge, "settings": {"judge.contestant": self.judge}}
        if tool == "suites_list":
            self.calls.append((tool, args))
            return {"suites": list(self.suites.values())}
        if tool == "suite_create":
            self.calls.append((tool, args))
            self.suites[args["name"]] = {"id": "s_" + args["name"], "name": args["name"], "description": args["description"], "cases": 0}
            return {"id": "s_" + args["name"], "name": args["name"]}
        if tool == "cases_import":
            self.calls.append((tool, args))
            rows = [json.loads(x) for x in args["text"].splitlines()]
            self.rows = rows
            added = len(rows) if self.imported is None else self.imported
            for card in self.suites.values():
                if card["id"] == args["suite"]:
                    card["cases"] = added
            return {"added": added, "errors": [] if added else [{"line": 1, "error": "bad"}]}
        if tool == "suite_remove":
            self.calls.append((tool, args))
            self.suites = {k: v for k, v in self.suites.items() if v["id"] != args["suite"]}
            return {"removed": True}
        return super().handle(tool, args)


def trained_child(ctx, tmp_path, records, kind=None, name="Prueba"):
    """A child GGUF whose training dataset (version 1) holds out records, and its parent."""
    spec = {"name": name, "sources": [{"type": "jsonl", "text": "\n".join(json.dumps(r, ensure_ascii=False) for r in records)}]}
    if kind:
        spec["kind"] = kind
    version = ctx.svc.datasets.build(spec)["version"]
    parent_file = build_gguf(tmp_path / "p.gguf", context=2048, size=4096)
    child_file = build_gguf(tmp_path / "c.gguf", context=2048, size=4096)
    store = ctx.svc.store
    parent = store.create_artifact("gguf", "parent-f16", path=str(parent_file), size=4096)
    child = store.create_artifact("gguf", "child-f16", path=str(child_file), size=4096, parents=[parent["id"]], dataset_version=version["id"])
    return parent, child, version


def qa(n):
    return [{"messages": [{"role": "user", "content": f"Pregunta {i}"}, {"role": "assistant", "content": f"Respuesta {i}"}]} for i in range(n)]


def test_a_chat_record_becomes_a_case_with_a_judge_and_the_reference():
    row = E.record_case({"messages": [{"role": "system", "content": "Sé breve"}, {"role": "user", "content": "¿Capital de Francia?"},
                                      {"role": "assistant", "content": "París"}]}, 3)
    assert row["prompt"] == "¿Capital de Francia?" and row["system"] == "Sé breve" and row["expected"] == "París"
    assert row["checker"] == {"type": "judge", "rubric": E.RUBRIC, "reference": "París"} and row["title"].startswith("eval-003")


def test_a_longer_exchange_keeps_its_turns_and_asks_up_to_the_last_answer():
    row = E.record_case({"messages": [{"role": "user", "content": "Hola"}, {"role": "assistant", "content": "Buenas"},
                                      {"role": "user", "content": "¿Y 2+2?"}, {"role": "assistant", "content": "4"}]}, 1)
    assert row["prompt"]["text"] == "¿Y 2+2?" and len(row["prompt"]["messages"]) == 3 and row["expected"] == "4"


def test_instruction_and_text_records_become_cases_and_empty_ones_do_not():
    assert E.record_case({"prompt": "Di hola", "response": "Hola"}, 1)["expected"] == "Hola"
    body = "Era una mañana tranquila en el pueblo y todos los vecinos salieron a la plaza para celebrar la fiesta mayor."
    row = E.record_case({"text": body}, 2)
    assert row["prompt"].startswith(E.CONTINUE_PROMPT) and row["checker"]["rubric"] == E.RUBRIC_TEXT
    assert row["expected"] and body.endswith(row["expected"])
    assert E.record_case({"text": "corto"}, 1) is None
    assert E.record_case({"messages": [{"role": "user", "content": "x"}]}, 1) is None
    assert E.record_case({"unrelated": 1}, 1) is None


def test_eval_rows_stop_at_the_limit_and_count_what_was_skipped():
    rows, skipped = E.eval_rows([{"unrelated": 1}, *qa(5)], 3)
    assert len(rows) == 3 and skipped == 1


def test_the_plan_of_a_child_with_held_out_records_is_the_dataset_intent(ctx, tmp_path):
    _parent, child, version = trained_child(ctx, tmp_path, qa(60))
    plan = ctx.svc.evaluator.plan(child["id"])
    assert plan["intent"] == "dataset" and plan["suites"] == ["pyg-prueba-v1-eval", "rapida"] and plan["dataset"]["records"] == version["eval"]
    only = ctx.svc.evaluator.plan(child["id"], regression=False)
    assert only["suites"] == ["pyg-prueba-v1-eval"] and only["regression"] is False
    explicit = ctx.svc.evaluator.plan(child["id"], suites=["codigo-python"])
    assert explicit["intent"] == "" and explicit["suites"] == ["codigo-python"]


def test_the_dataset_intent_needs_a_dataset_with_held_out_records(ctx, pair):
    _parent, child = pair
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.evaluator.plan(child["id"], intent="dataset")
    assert exc.value.code == "invalid" and "held-out" in exc.value.message


def test_a_dataset_evaluation_builds_the_suite_runs_both_models_and_stores_both_verdicts(ctx, tmp_path):
    parent, child, version = trained_child(ctx, tmp_path, qa(60))
    fake = attach(ctx, DatasetGalton())
    record = run(ctx, child)
    create = fake.args_of("suite_create")[0]
    assert create["name"] == "pyg-prueba-v1-eval" and f"sha256:{version['sha256']}:{version['eval']}" in create["description"]
    assert fake.args_of("cases_import")[0]["format"] == "jsonl" and len(fake.rows) == version["eval"]
    assert all(r["checker"]["type"] == "judge" and r["checker"]["reference"] == r["expected"] for r in fake.rows)
    start = fake.args_of("run_start")[0]
    assert start["suites"] == ["s_pyg-prueba-v1-eval", "rapida"] and len(start["contestants"]) == 2
    assert start["settings"]["effort"] == "off"            # the records answer directly: no thinking budget spent on them
    compares = fake.args_of("compare")
    assert [c.get("suite") for c in compares] == ["s_pyg-prueba-v1-eval", "rapida"]
    assert record["intent"] == "dataset" and record["verdict"] == "better" and record["regression"]["suite"] == "rapida" and record["regression"]["verdict"] == "better"
    assert record["dataset"]["cases"] == version["eval"] and record["dataset"]["created"] is True and record["promote_suggested"] is True
    assert ctx.svc.store.artifact(child["id"])["metrics"]["galton"]["dataset"]["suite"] == "pyg-prueba-v1-eval"


def test_the_suite_is_reused_for_the_same_dataset_version(ctx, tmp_path):
    _parent, child, version = trained_child(ctx, tmp_path, qa(60))
    existing = {"pyg-prueba-v1-eval": {"id": "s_old", "name": "pyg-prueba-v1-eval", "cases": version["eval"],
                                       "description": f"Pygmalion: x. sha256:{version['sha256']}:{version['eval']}"}}
    fake = attach(ctx, DatasetGalton(existing=existing))
    record = run(ctx, child)
    assert not fake.args_of("suite_create") and not fake.args_of("cases_import")
    assert fake.args_of("run_start")[0]["suites"][0] == "s_old" and record["dataset"]["created"] is False


def test_a_suite_of_another_version_with_the_same_name_is_not_overwritten(ctx, tmp_path):
    _parent, child, version = trained_child(ctx, tmp_path, qa(60))
    existing = {"pyg-prueba-v1-eval": {"id": "s_mine", "name": "pyg-prueba-v1-eval", "cases": 7, "description": "the user's own suite"}}
    fake = attach(ctx, DatasetGalton(existing=existing))
    run(ctx, child)
    assert not fake.args_of("suite_remove")
    assert fake.args_of("suite_create")[0]["name"] == f"pyg-prueba-v1-eval-{version['sha256'][:8]}"


def test_an_incomplete_suite_built_from_this_version_is_replaced(ctx, tmp_path):
    _parent, child, version = trained_child(ctx, tmp_path, qa(60))
    existing = {"pyg-prueba-v1-eval": {"id": "s_half", "name": "pyg-prueba-v1-eval", "cases": 2,
                                       "description": f"Pygmalion: x. sha256:{version['sha256']}:{version['eval']}"}}
    fake = attach(ctx, DatasetGalton(existing=existing))
    run(ctx, child)
    assert fake.args_of("suite_remove")[0]["suite"] == "s_half" and fake.args_of("suite_create")


def test_without_a_judge_model_the_evaluation_stops_early_with_a_hint(ctx, tmp_path):
    _parent, child, _version = trained_child(ctx, tmp_path, qa(60))
    fake = attach(ctx, DatasetGalton(judge=""))
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "judge" in exc.value.message.lower() and "judge.contestant" in (exc.value.hint or "")
    assert not fake.args_of("suite_create") and not fake.args_of("run_start")


def test_an_import_that_adds_nothing_is_an_error(ctx, tmp_path):
    _parent, child, _version = trained_child(ctx, tmp_path, qa(60))
    attach(ctx, DatasetGalton(imported=0))
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "imported no case" in exc.value.message


def test_answers_the_judge_never_graded_are_not_a_verdict(ctx, tmp_path):
    _parent, child, _version = trained_child(ctx, tmp_path, qa(60))
    fake = attach(ctx, DatasetGalton(verdict="no data"))
    original = fake.card

    def card(state, pct):
        out = original(state, pct)
        out["pending_judge"] = 48
        return out
    fake.card = card
    with pytest.raises(PygmalionError) as exc:
        run(ctx, child)
    assert "48" in exc.value.message


def test_a_regression_on_the_general_suite_blocks_the_promotion_suggestion(ctx, tmp_path):
    _parent, child, _version = trained_child(ctx, tmp_path, qa(60))
    fake = attach(ctx, DatasetGalton())
    original = fake.handle

    def handle(tool, args):
        out = original(tool, args)
        if tool == "compare" and args.get("suite") == "rapida":
            out = {**out, "verdict": "worse"}
        return out
    fake.handle = handle
    record = run(ctx, child)
    assert record["verdict"] == "better" and record["regression"]["verdict"] == "worse" and record["promote_suggested"] is False


def test_the_number_of_cases_is_capped_by_the_setting_and_says_so(ctx, tmp_path):
    _parent, child, version = trained_child(ctx, tmp_path, qa(400))
    ctx.svc.settings.set({"galton.eval_cases": 5})
    fake = attach(ctx, DatasetGalton())
    record = run(ctx, child)
    assert len(fake.rows) == 5 and record["dataset"]["cases"] == 5 and version["eval"] > 5
    assert any("5" in str(w) for w in record["warnings"])


def test_a_text_dataset_is_measured_by_continuations(ctx, tmp_path):
    texts = [{"text": f"Párrafo número {i} con bastante contenido para ser contado como texto corrido y para poder partirlo en dos mitades."} for i in range(60)]
    _parent, child, _version = trained_child(ctx, tmp_path, texts, name="Textos")
    fake = attach(ctx, DatasetGalton())
    run(ctx, child)
    assert all(r["prompt"].startswith(E.CONTINUE_PROMPT) and r["checker"]["rubric"] == E.RUBRIC_TEXT for r in fake.rows)


def test_evaluate_start_by_dataset_returns_the_plan_with_the_dataset(ctx, tmp_path):
    _parent, child, _version = trained_child(ctx, tmp_path, qa(60))
    attach(ctx, DatasetGalton())
    answer = call(ctx.svc, "evaluate_start", artifact=child["id"])
    assert answer["plan"]["intent"] == "dataset" and answer["plan"]["dataset"]["dataset"] == "Prueba"
    ctx.svc.jobs.run_until_idle()
    assert ctx.svc.store.job(answer["job"]["id"])["state"] == "done"


def test_the_token_falls_back_to_galtons_folder_next_to_this_one(monkeypatch, tmp_path):
    from pygmalion_hoard import evaluate as ev

    token = tmp_path / "Galton's Hoard" / "data" / "mcp-token"
    token.parent.mkdir(parents=True)
    token.write_text("abc\n", encoding="utf-8")
    monkeypatch.setattr(ev, "SIBLING_TOKEN", token)

    class S:
        def get(self, key):
            return ""
    g = ev.Galton(S(), offline=True)            # offline: the hub registry is not asked
    assert g.token() == "abc"
