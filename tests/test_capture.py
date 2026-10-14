"""The capture proxy: forwarding to fake llama.cpp and vLLM servers, the prefix tree of calls, loss masks, exports balanced by harness and the tools."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from conftest import build_services
from fake_servers import SPECIAL, FakeLlama, FakeVllm, Routed
from pygmalion_hoard.agent_tools import TOOLS_BY_NAME, call_tool, uncapped
from pygmalion_hoard.capture import tree as T
from pygmalion_hoard.capture.proxy import ProxyError, identify, sse_events
from pygmalion_hoard.capture.service import flatten_messages, parse_when
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.main import create_app

LLAMA = "http://llama.test:8093"
VLLM = "http://vllm.test:8002"
USER = {"role": "user", "content": "Hello there"}
SYSTEM = {"role": "system", "content": "Be brief"}


@pytest.fixture
def cap(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    monkeypatch.setenv("HOARD_HOME", str(tmp_path / "hoard-home"))
    monkeypatch.setenv("PYGMALION_OFFLINE", "1")
    routed = Routed()
    routed.hosts = {"llama.test": FakeLlama(), "vllm.test": FakeVllm()}
    c = build_services(tmp_path, capture_transport=routed)
    app = create_app(c.svc.config, services=c.svc)
    with TestClient(app, base_url="http://127.0.0.1") as client:
        ns = SimpleNamespace(client=client, svc=c.svc, llama=routed.hosts["llama.test"], vllm=routed.hosts["vllm.test"], routed=routed)

        def start(url=LLAMA, **kw):
            with uncapped():
                return call_tool(c.svc, "capture_proxy", {"action": "start", "upstream_url": url, **kw})

        def chat(messages, path="/capture/rollout-a/v1/chat/completions", **body):
            return client.post(path, json={"model": "m", "messages": messages, **body})

        def tool(name, **arguments):
            with uncapped():
                return call_tool(c.svc, name, arguments)

        ns.start, ns.chat, ns.tool = start, chat, tool
        yield ns


def converse(cap, session="rollout-a", turns=3, harness=None):
    """A harness conversation: each call carries the whole history, answers re-rendered as the template would."""
    path = f"/capture/{harness}/{session}/v1/chat/completions" if harness else f"/capture/{session}/v1/chat/completions"
    history = [SYSTEM, USER]
    for n in range(turns):
        cap.llama.say(f"Answer number {n}")
        r = cap.chat(history, path=path)
        assert r.status_code == 200, r.text
        history = history + [r.json()["choices"][0]["message"], {"role": "user", "content": f"Follow up {n}"}]
    return history


# ------------------------------------------------------------------------------------------------- the pure tree
def node(i, prompt, completion, parent=None, seq=None, harness="h", session="s", reward=None, lps=None):
    return {"id": i, "parent": parent, "prompt_ids": prompt, "completion_ids": completion, "logprobs": lps or [-1.0] * len(completion), "seq": seq or int(i[1:]),
            "harness": harness, "session": session, "reward": reward, "outcome": None, "ts": 1.0 + int(i[1:]), "model": "m", "sampling": {}}


def test_find_parent_takes_the_deepest_exact_prefix():
    a = [1, 2, 3, 4]
    b = a + [9, 8, 7]
    cands = [{"id": "a", "full_len": 4, "full_sha": T.fingerprint(a), "seq": 1}, {"id": "b", "full_len": 7, "full_sha": T.fingerprint(b), "seq": 2},
             {"id": "x", "full_len": 5, "full_sha": T.fingerprint([1, 2, 3, 4, 6]), "seq": 3}]
    assert T.find_parent(b + [5], cands)["id"] == "b"
    assert T.find_parent(a + [5, 5], cands)["id"] == "a"          # x differs at its last token
    assert T.find_parent([1, 2, 3], cands) is None
    assert T.find_parent([], cands) is None


def test_a_chain_is_one_sequence_with_the_sampled_tokens_unmasked():
    n1 = node("n1", [1, 2, 3], [10, 11])
    n2 = node("n2", [1, 2, 3, 10, 11, 4, 5], [12], parent="n1")
    n3 = node("n3", [1, 2, 3, 10, 11, 4, 5, 12, 6], [13, 14], parent="n2")
    rows = T.sequences([n1, n2, n3])
    assert len(rows) == 1
    row = rows[0]
    assert row["input_ids"] == [1, 2, 3, 10, 11, 4, 5, 12, 6, 13, 14]
    assert row["mask"] == [0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1]
    assert row["nodes"] == ["n1", "n2", "n3"] and row["loss_tokens"] == 5 and row["leaf"] == "n3"
    assert row["logprobs"][3] == -1.0 and row["logprobs"][0] is None
    sft = T.sft_row(row)
    assert sft["labels"] == [-100, -100, -100, 10, 11, -100, -100, 12, -100, 13, 14]
    assert T.rl_row(row)["loss_mask"] == row["mask"]


def test_branches_give_one_row_each_and_shared_tokens_train_once():
    root = node("n1", [1, 2], [10, 11])
    left = node("n2", [1, 2, 10, 11, 3], [20], parent="n1")
    right = node("n3", [1, 2, 10, 11, 4], [30, 31], parent="n1")
    rows = T.sequences([root, left, right])
    assert [r["leaf"] for r in rows] == ["n2", "n3"]
    assert rows[0]["mask"] == [0, 0, 1, 1, 0, 1]
    assert rows[1]["mask"] == [0, 0, 0, 0, 0, 1, 1]            # the root's tokens are context in the second path
    again = T.sequences([root, left, right], dedupe_shared=False)
    assert again[1]["mask"] == [0, 0, 1, 1, 0, 1, 1]
    last = T.sequences([root, left, right], loss_on="last")
    assert [sum(r["mask"]) for r in last] == [1, 2]


def test_a_call_that_continues_nothing_is_its_own_root_and_empty_completions_are_skipped():
    a = node("n1", [1, 2], [10])
    b = node("n2", [7, 8], [11])                                   # unrelated prompt
    c = node("n3", [5, 5], [])                                     # nothing sampled
    rows, skipped = T.sequences_report([a, b, c])
    assert sorted(r["leaf"] for r in rows) == ["n1", "n2"] and skipped["no_loss_tokens"] == 1


def test_an_excluded_parent_leaves_the_child_as_a_root_with_its_whole_prompt():
    child = node("n2", [1, 2, 10, 3], [20], parent="n1")
    rows = T.sequences([child])
    assert rows[0]["input_ids"] == [1, 2, 10, 3, 20] and rows[0]["mask"] == [0, 0, 0, 0, 1]


def test_balance_by_harness_modes_and_report():
    rows = [{"harness": "a", "loss_tokens": 2, "leaf": f"a{i}"} for i in range(6)] + [{"harness": "b", "loss_tokens": 3, "leaf": f"b{i}"} for i in range(2)]
    kept, report = T.balance_by_harness(rows, "min", seed=1)
    assert sorted(r["harness"] for r in kept) == ["a", "a", "b", "b"]
    assert report["a"] == {"available": 6, "kept": 2, "loss_tokens_available": 12, "loss_tokens_kept": 4}
    assert report["b"]["kept"] == 2
    assert T.balance_by_harness(rows, "min", seed=1)[0] == kept                       # deterministic
    capped, rep = T.balance_by_harness(rows, 3)
    assert rep["a"]["kept"] == 3 and rep["b"]["kept"] == 2
    all_rows, rep = T.balance_by_harness(rows, None)
    assert len(all_rows) == 8 and rep["a"]["kept"] == 6


# ------------------------------------------------------------------------------------------------- llama.cpp
def test_the_proxy_is_stopped_until_started_and_says_so_in_the_api_format(cap):
    r = cap.chat([USER])
    assert r.status_code == 503 and r.json()["error"]["type"] == "capture_stopped"
    st = cap.tool("capture_proxy")
    assert st["enabled"] is False and st["base_url"].endswith("/capture/v1")
    with pytest.raises(PygmalionError):
        cap.tool("capture_proxy", action="start")                  # no upstream yet
    cap.start()
    assert cap.client.get("/capture/v1/models").json()["data"][0]["id"] == "fake-model"


def test_a_conversation_becomes_a_chain_with_verified_ids(cap):
    cap.start()
    converse(cap, turns=3)
    view = cap.tool("capture_session", session="rollout-a")
    nodes = view["nodes"]
    assert [n["parent"] for n in nodes] == [None, nodes[0]["id"], nodes[1]["id"]]
    assert [n["depth"] for n in nodes] == [0, 1, 2]
    assert all(n["verified"] and n["prompt_ids_source"] == "apply-template+tokenize" for n in nodes), [n["verify_note"] for n in nodes]
    assert view["session"]["calls"] == 3 and view["session"]["roots"] == 1 and view["session"]["leaves"] == 1
    assert nodes[1]["input"] == "Follow up 0" and nodes[1]["output"] == "Answer number 1" and nodes[1]["lp_mean"] < 0
    full = cap.tool("capture_session", session="rollout-a", node=nodes[2]["id"], max_tokens=4)["node"]
    assert full["parent"] == nodes[1]["id"] and len(full["tokens"]) == 4 and full["tokens"][0]["token"] is not None
    assert len(full["messages"]) == 6 and full["messages"][0]["role"] == "system" and full["response"]["content"] == "Answer number 2"
    assert full["new_prompt_tokens"] < full["prompt_tokens"] and full["lowest"][0]["logprob"] <= full["tokens"][0]["logprob"]
    assert cap.llama.calls.count("POST /v1/chat/completions") == 3 and "POST /apply-template" in cap.llama.calls


def test_the_client_sees_a_plain_answer_without_the_extra_fields(cap):
    cap.start()
    cap.llama.say("Hi")
    r = cap.chat([USER])
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Hi" and body["choices"][0]["logprobs"] is None
    assert "prompt_token_ids" not in body and r.headers["x-capture-session"] == "rollout-a" and r.headers["x-capture-verified"] == "1"
    sent = cap.llama.requests[0]
    assert sent["logprobs"] is True and sent["top_logprobs"] == 1 and sent["stream"] is False


def test_a_client_that_asked_for_logprobs_gets_them(cap):
    cap.start()
    cap.llama.say("Hi")
    r = cap.chat([USER], logprobs=True, top_logprobs=3)
    assert r.json()["choices"][0]["logprobs"]["content"][0]["logprob"] < 0
    assert cap.llama.requests[0]["top_logprobs"] == 3


def test_a_streaming_client_gets_events_from_the_complete_answer(cap):
    cap.start()
    cap.llama.say("Streamed text")
    r = cap.chat([USER], stream=True, stream_options={"include_usage": True})
    assert r.headers["content-type"].startswith("text/event-stream")
    events = [line[6:] for line in r.text.split("\n\n") if line.startswith("data: ")]
    assert events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks if c["choices"]) == "Streamed text"
    assert [c["choices"][0]["finish_reason"] for c in chunks if c["choices"]][-1] == "stop" and chunks[-1]["usage"]["prompt_tokens"] > 0
    assert cap.llama.requests[0]["stream"] is False and "stream_options" not in cap.llama.requests[0]
    assert cap.tool("capture_sessions")["sessions"][0]["calls"] == 1


def test_streamed_tool_calls_keep_their_shape(cap):
    cap.start()
    cap.llama.say("", tool_calls=[("lookup", {"q": "x"})])
    r = cap.chat([USER], stream=True, tools=[{"type": "function", "function": {"name": "lookup", "parameters": {}}}])
    chunks = [json.loads(line[6:]) for line in r.text.split("\n\n") if line.startswith("data: {")]
    calls = [c["choices"][0]["delta"]["tool_calls"][0] for c in chunks if c["choices"] and c["choices"][0]["delta"].get("tool_calls")]
    assert calls[0]["function"]["name"] == "lookup" and json.loads(calls[0]["function"]["arguments"]) == {"q": "x"}
    assert chunks[-1]["choices"][0]["finish_reason"] == "tool_calls"
    node = cap.tool("capture_sessions")
    assert node["sessions"][0]["unverified"] == 0


def test_tool_calls_chain_and_tools_are_stored_once(cap):
    cap.start()
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    cap.llama.say("", tool_calls=[("lookup", {"q": "x"})])
    first = cap.chat([USER], tools=tools).json()["choices"][0]["message"]
    history = [USER, first, {"role": "tool", "tool_call_id": "call_0", "content": "result"}]
    cap.llama.say("Done")
    cap.chat(history, tools=tools)
    nodes = cap.tool("capture_session", session="rollout-a")["nodes"]
    assert nodes[1]["parent"] == nodes[0]["id"] and all(n["verified"] for n in nodes)
    stored = cap.svc.capture_store.node(nodes[1]["id"])
    assert stored["tools"] == tools and [m["role"] for m in stored["messages"]] == ["user", "assistant", "tool"]
    raw = cap.svc.db.one("SELECT tools, msg_skip, messages FROM capture_nodes WHERE id = ?", (nodes[1]["id"],))
    assert raw["tools"] is None and raw["msg_skip"] == 2 and len(json.loads(raw["messages"])) == 1      # only what is new is stored


def test_tools_the_server_cannot_render_make_the_call_unverified_and_the_export_skips_it(cap):
    cap.llama.jinja = False
    cap.start()
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    cap.llama.say("Plain")
    cap.chat([USER], tools=tools)
    node = cap.tool("capture_session", session="rollout-a")["nodes"][0]
    assert node["verified"] is False and "server counted" in node["verify_note"]
    with pytest.raises(PygmalionError):
        cap.tool("capture_export")                                             # nothing verified, nothing to export
    dry = cap.tool("capture_export", dry_run=True)
    assert dry["rows"] == 0 and dry["report"]["unverified_excluded"] == 1
    inc = cap.tool("capture_export", dry_run=True, include_unverified=True)
    assert inc["rows"] == 1


def test_a_bos_added_by_the_tokenizer_is_found_by_the_count_check(cap):
    cap.llama.special_bos = True
    cap.start()
    cap.llama.say("Hi")
    cap.chat([USER])
    node = cap.tool("capture_session", session="rollout-a")["nodes"][0]
    assert node["verified"]
    full = cap.svc.capture_store.node(node["id"])
    assert full["prompt_ids"][0] == 0                                          # the variant with the BOS was the one that matched usage


def test_the_missing_end_of_turn_token_is_appended_so_the_chain_holds(cap):
    cap.llama.report_eos = False
    cap.start()
    history = converse(cap, turns=2)
    nodes = cap.tool("capture_session", session="rollout-a")["nodes"]
    assert nodes[1]["parent"] == nodes[0]["id"] and all(n["eos_appended"] and n["verified"] for n in nodes)
    full = cap.svc.capture_store.node(nodes[0]["id"])
    assert full["completion_ids"][-1] == SPECIAL["<|im_end|>"] and full["logprobs"][-1] is None
    assert len(history) == 6


def test_a_branch_of_the_same_conversation_gives_a_second_row(cap):
    cap.start()
    history = converse(cap, turns=1)
    for variant in ("A", "B"):
        cap.llama.say(f"Variant {variant}")
        cap.chat(history)
    sessions = cap.tool("capture_sessions")["sessions"]
    assert sessions[0]["calls"] == 3 and sessions[0]["leaves"] == 2 and sessions[0]["roots"] == 1
    dry = cap.tool("capture_export", dry_run=True)
    assert dry["rows"] == 2


# ------------------------------------------------------------------------------------------------- vLLM
def test_vllm_gives_the_ids_directly(cap):
    cap.start(VLLM)
    cap.vllm.say("Glm answer")
    r = cap.chat([USER])
    body = r.json()
    assert body["choices"][0]["message"]["content"] == "Glm answer" and "prompt_token_ids" not in body and "token_ids" not in body["choices"][0]
    sent = cap.vllm.requests[0]
    assert sent["return_token_ids"] is True and sent["return_tokens_as_token_ids"] is True and sent["logprobs"] is True
    node = cap.tool("capture_session", session="rollout-a")["nodes"][0]
    assert node["verified"] and node["prompt_ids_source"] == "upstream" and node["model"] == "fake-glm"
    assert cap.vllm.calls.count("GET /version") == 1 and "POST /apply-template" not in cap.vllm.calls


def test_vllm_conversation_chain_and_rl_export(cap, tmp_path):
    cap.start(VLLM)
    history = [USER]
    for n in range(2):
        cap.vllm.say(f"Step {n}")
        history = history + [cap.chat(history, path="/capture/glm/ep1/v1/chat/completions").json()["choices"][0]["message"], {"role": "user", "content": f"Next {n}"}]
    nodes = cap.tool("capture_session", session="ep1")["nodes"]
    assert nodes[1]["parent"] == nodes[0]["id"]
    out = cap.tool("capture_export", harness=["glm"])
    rl = [json.loads(line) for line in open(out["files"]["rl"], encoding="utf-8")]
    assert len(rl) == 1 and rl[0]["harness"] == "glm" and rl[0]["reward"] is None
    ids, mask, lps = rl[0]["input_ids"], rl[0]["loss_mask"], rl[0]["logprobs"]
    assert len(ids) == len(mask) == len(lps) and sum(mask) == out["loss_tokens"]
    assert all((lp is not None) == bool(m) for lp, m in zip(lps, mask)) and ids[-1] == SPECIAL["<|im_end|>"]
    sft = [json.loads(line) for line in open(out["files"]["sft"], encoding="utf-8")]
    assert [x if m else -100 for x, m in zip(sft[0]["input_ids"], mask)] == sft[0]["labels"]


def test_auto_detection_tells_the_two_servers_apart(cap):
    cap.start(LLAMA)
    assert cap.svc.capture_proxy.detect("http://llama.test:8093")[0] == "llama_server"
    assert cap.svc.capture_proxy.detect("http://vllm.test:8002")[0] == "vllm"
    assert cap.svc.capture_proxy.detect("http://nowhere.test:1")[0] == "unreachable"
    status = cap.tool("capture_proxy", action="check")
    assert status["upstream"]["detected"] == "llama_server" and status["upstream"]["models"] == ["fake-model"]


# ------------------------------------------------------------------------------------------------- faithful sampling and errors
def test_faithful_sampling_neutralises_the_truncating_parameters(cap):
    cap.start(top_logprobs=1)
    cap.llama.say("a")
    cap.chat([USER], top_p=0.9, top_k=20, temperature=0.7)
    assert cap.llama.requests[-1]["top_p"] == 0.9 and cap.llama.requests[-1]["top_k"] == 20      # untouched by default
    cap.tool("capture_proxy", action="configure", faithful_sampling=True)
    cap.llama.say("b")
    cap.chat([USER], top_p=0.9, top_k=20, temperature=0.7)
    sent = cap.llama.requests[-1]
    assert (sent["top_k"], sent["top_p"], sent["min_p"], sent["temperature"]) == (0, 1.0, 0.0, 0.7)
    cap.start(VLLM)
    cap.vllm.say("c")
    cap.chat([USER], top_p=0.9, top_k=20, min_p=0.1)
    assert not {"top_p", "top_k", "min_p"} & set(cap.vllm.requests[-1])
    node = cap.tool("capture_sessions")["sessions"][0]
    assert node["calls"] == 3


def test_an_unreachable_upstream_and_an_upstream_error_reach_the_client_in_the_api_format(cap):
    cap.start("http://down.test:1")
    r = cap.chat([USER])
    assert r.status_code == 502 and r.json()["error"]["type"] == "upstream_unreachable"
    cap.start(LLAMA)
    cap.llama.fail_with = 400
    r = cap.chat([USER])
    assert r.status_code == 400 and r.json()["error"]["message"] == "boom"
    assert cap.tool("capture_sessions")["sessions"] == []                                 # nothing recorded for a failed call
    assert cap.chat([]).status_code in (200, 400)
    assert cap.client.post("/capture/v1/chat/completions", content=b"nope", headers={"content-type": "application/json"}).status_code == 400


def test_session_and_harness_come_from_the_url_the_headers_or_the_key(cap):
    class H(dict):
        get = lambda self, k, d=None: dict.get(self, k, d)  # noqa: E731
    assert identify("", "", H()) == ("", "default")
    assert identify("faustus", "s1", H()) == ("faustus", "s1")
    assert identify("", "", H({"x-capture-session": "s2", "x-capture-harness": "cowork"})) == ("cowork", "s2")
    assert identify("", "", H({"authorization": "Bearer faustus:ep7"})) == ("faustus", "ep7")
    assert identify("", "", H({"authorization": "Bearer ep8"})) == ("", "ep8")
    assert identify("", "", H({"authorization": "Bearer sk-no-key-required"})) == ("", "default")
    assert identify("bad name!", "s/1", H()) == ("", "default")
    cap.start()
    cap.llama.say("x")
    cap.client.post("/capture/v1/chat/completions", json={"messages": [USER]}, headers={"Authorization": "Bearer hoardA:keyed", "X-Capture-Harness": "ignored-for-key"})
    cap.llama.say("y")
    cap.client.post("/capture/v1/chat/completions", json={"messages": [USER]}, headers={"X-Capture-Session": "viaheader"})
    sessions = {s["id"]: s for s in cap.tool("capture_sessions")["sessions"]}
    assert set(sessions) == {"keyed", "viaheader"} and sessions["keyed"]["harness"] == "ignored-for-key"


# ------------------------------------------------------------------------------------------------- outcomes, exports, datasets
def test_outcome_filters_dates_and_the_dry_run(cap):
    cap.start()
    converse(cap, session="win", turns=2, harness="faustus")
    converse(cap, session="lose", turns=2, harness="faustus")
    converse(cap, session="open", turns=2, harness="cowork")
    cap.tool("capture_outcome", session="win", reward=1.0, outcome="success")
    cap.tool("capture_outcome", session="lose", reward=0.0, outcome="failure", note="tests failed")
    sess = {s["id"]: s for s in cap.tool("capture_sessions")["sessions"]}
    assert sess["win"]["reward"] == 1.0 and sess["open"]["reward"] is None and sess["lose"]["outcome"] == "failure"
    assert cap.tool("capture_export", dry_run=True)["rows"] == 3
    assert cap.tool("capture_export", dry_run=True, require_reward=True)["rows"] == 2
    assert cap.tool("capture_export", dry_run=True, min_reward=0.5)["rows"] == 1
    assert cap.tool("capture_export", dry_run=True, harness=["cowork"])["rows"] == 1
    assert cap.tool("capture_export", dry_run=True, sessions=["win"])["rows"] == 1
    assert cap.tool("capture_export", dry_run=True, since="2999-01-01")["rows"] == 0
    assert cap.tool("capture_export", dry_run=True, until="2999-01-01T00:00:00Z")["rows"] == 3
    one = cap.tool("capture_session", session="win")["nodes"][-1]["id"]
    cap.tool("capture_outcome", session="win", node=one, reward=0.25)
    assert cap.svc.capture_store.node(one)["reward"] == 0.25
    cleared = cap.tool("capture_outcome", session="win", clear=True)
    assert cleared["session"]["reward"] is None
    with pytest.raises(PygmalionError):
        cap.tool("capture_outcome", session="nope", reward=1)
    with pytest.raises(PygmalionError):
        cap.tool("capture_export", dry_run=True, since="not a date")


def test_balance_by_harness_in_the_export_and_its_report(cap):
    cap.start()
    for i in range(4):
        converse(cap, session=f"f{i}", turns=1, harness="faustus")
    for i in range(2):
        converse(cap, session=f"c{i}", turns=1, harness="cowork")
    plain = cap.tool("capture_export", dry_run=True)
    assert plain["per_harness"]["faustus"]["available"] == 4 and plain["rows"] == 6
    balanced = cap.tool("capture_export", dry_run=True, balance_by_harness="min")
    assert balanced["per_harness"]["faustus"] == {**balanced["per_harness"]["faustus"], "available": 4, "kept": 2} and balanced["rows"] == 4
    capped = cap.tool("capture_export", dry_run=True, balance_by_harness=3)
    assert capped["rows"] == 5
    out = cap.tool("capture_export", balance_by_harness="min", formats=["sft"])
    rows = [json.loads(line) for line in open(out["files"]["sft"], encoding="utf-8")]
    assert sorted(r["harness"] for r in rows) == ["cowork", "cowork", "faustus", "faustus"] and "rl" not in out["files"]
    manifest = json.load(open(out["manifest"], encoding="utf-8"))
    assert manifest["per_harness"]["cowork"]["kept"] == 2


def test_the_export_registers_a_dataset_the_trainer_can_use(cap):
    cap.start()
    tools = [{"type": "function", "function": {"name": "lookup", "parameters": {}}}]
    cap.llama.say("", tool_calls=[("lookup", {"q": "x"})])
    first = cap.chat([SYSTEM, USER], tools=tools, path="/capture/faustus/r1/v1/chat/completions").json()["choices"][0]["message"]
    cap.llama.say("The answer is 42")
    cap.chat([SYSTEM, USER, first, {"role": "tool", "tool_call_id": "call_0", "content": "42"}], tools=tools, path="/capture/faustus/r1/v1/chat/completions")
    cap.tool("capture_outcome", session="r1", reward=1.0, outcome="success")
    with pytest.raises(PygmalionError):
        cap.tool("capture_export", formats=["dataset"])                         # a dataset needs a name
    out = cap.tool("capture_export", dataset_name="Captured agent", seed=3)
    assert set(out["files"]) == {"sft", "rl"} | set() and out["dataset"]["name"] == "Captured agent" and out["version"]["records"] == 1
    listed = {d["name"]: d for d in cap.tool("datasets_list")["datasets"]}
    assert "Captured agent" in listed
    records = cap.tool("dataset_records", dataset="Captured agent")["records"]
    assert len(records) == 1
    version = cap.svc.store.find_version("Captured agent")
    rec = cap.svc.datasets.read_records(version)[0]
    roles = [m["role"] for m in rec["messages"]]
    assert roles == ["system", "user", "assistant", "user", "assistant"]
    assert "<tool_call>" in rec["messages"][2]["content"] and rec["messages"][3]["content"].startswith("Tool result:") and rec["messages"][-1]["content"] == "The answer is 42"
    assert rec["_meta"]["harness"] == "faustus" and rec["_meta"]["reward"] == 1.0 and rec["_meta"]["source"] == "capture"
    again = cap.tool("capture_export", dataset="Captured agent", formats=["dataset"])
    assert again["version"]["n"] == 2
    dry = cap.tool("capture_export", dataset_name="x", dry_run=True)
    assert dry["dry_run"] is True and dry["rows"] == 1


def test_flatten_merges_and_translates_roles():
    flat = flatten_messages([{"role": "developer", "content": "d"}, {"role": "user", "content": [{"type": "text", "text": "a"}]}, {"role": "user", "content": "b"},
                             {"role": "assistant", "content": None, "tool_calls": [{"function": {"name": "f", "arguments": "{\"x\":1}"}}]},
                             {"role": "tool", "content": "r"}, {"role": "weird", "content": "z"}])
    assert [m["role"] for m in flat] == ["system", "user", "assistant", "user"]
    assert flat[1]["content"] == "a\n\nb" and '"name": "f"' in flat[2]["content"] and flat[3]["content"] == "Tool result:\nr"


def test_parse_when_accepts_numbers_and_iso():
    assert parse_when(5) == 5.0 and parse_when("5") == 5.0 and parse_when("2026-01-01T00:00:00Z") == parse_when("2026-01-01") and parse_when("") is None


# ------------------------------------------------------------------------------------------------- management
def test_delete_needs_confirmation_and_cascades(cap):
    cap.start()
    converse(cap, session="gone", turns=2)
    converse(cap, session="stay", turns=1)
    with pytest.raises(PygmalionError):
        cap.tool("capture_delete", sessions=["gone"])
    assert cap.tool("capture_delete", sessions=["gone"], confirm=True)["deleted"] == 1
    assert [s["id"] for s in cap.tool("capture_sessions")["sessions"]] == ["stay"]
    assert cap.svc.capture_store.counts()["calls"] == 1
    with pytest.raises(PygmalionError):
        cap.tool("capture_delete", confirm=True)
    assert cap.tool("capture_delete", all=True, confirm=True)["deleted"] == 1
    with pytest.raises(PygmalionError):
        cap.tool("capture_session", session="gone")


def test_rest_routes_wrap_the_tools(cap):
    cap.start()
    converse(cap, session="web", turns=1, harness="faustus")
    c = cap.client
    assert c.get("/api/capture").json()["enabled"] is True
    assert c.get("/api/capture/sessions", params={"harness": "faustus"}).json()["sessions"][0]["id"] == "web"
    tree = c.get("/api/capture/sessions/web").json()
    assert tree["session"]["calls"] == 1
    assert c.post("/api/capture/sessions/web/outcome", json={"reward": 0.5, "outcome": "partial"}).json()["session"]["reward"] == 0.5
    exp = c.post("/api/capture/export", json={"arguments": {"dry_run": True}}).json()
    assert exp["rows"] == 1 and exp["dry_run"] is True
    assert c.delete("/api/capture/sessions/web").status_code == 400
    assert c.delete("/api/capture/sessions/web", params={"confirm": "true"}).json()["deleted"] == 1
    assert c.post("/api/capture/proxy", json={"action": "stop"}).json()["enabled"] is False
    assert c.post("/api/ui/call", json={"name": "capture_sessions", "arguments": {}}).json()["sessions"] == []


def test_the_tools_are_in_the_catalogue_with_short_first_lines():
    names = [n for n in TOOLS_BY_NAME if n.startswith("capture_")]
    assert names == ["capture_proxy", "capture_sessions", "capture_session", "capture_outcome", "capture_export", "capture_delete"]
    for n in names:
        assert len(TOOLS_BY_NAME[n].description.splitlines()[0]) <= 110


def test_a_session_with_many_turns_costs_only_the_new_tokens(cap):
    cap.start()
    converse(cap, session="long", turns=6)
    rows = cap.svc.db.query("SELECT prompt_len, parent_len, prompt_delta FROM capture_nodes WHERE session_id = 'long' ORDER BY seq")
    for earlier, later in zip(rows, rows[1:]):
        assert later["parent_len"] > earlier["prompt_len"] and len(json.loads(later["prompt_delta"])) < later["prompt_len"]
    nodes = cap.svc.capture_store.nodes("long")
    for n in nodes[1:]:
        assert n["prompt_ids"][: len(nodes[0]["prompt_ids"])] == nodes[0]["prompt_ids"]


def test_sse_events_survive_an_answer_without_text():
    events = list(sse_events({"id": "x", "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": None}, "finish_reason": "length"}]}))
    assert events[-1] == b"data: [DONE]\n\n" and b'"finish_reason":"length"' in events[-2]


def test_a_proxy_error_has_the_api_shape():
    err = ProxyError(503, "x", "capture_stopped")
    assert err.body() == {"error": {"message": "x", "type": "capture_stopped", "code": "capture_stopped"}}
