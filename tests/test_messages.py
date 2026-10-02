"""Everything the UI shows from the backend is a catalogue key plus parameters (messages.py), and the client has both languages for each."""

import ast
import json
import re
from pathlib import Path

import pytest

from pygmalion_hoard import messages as M
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.messages import ERRORS, TEXTS, CodedText, error_text, fields_of, fill, hint_text, localise, recognise, text, wire

ROOT = Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "pygmalion_hoard"
MSGS = (ROOT / "client" / "src" / "msgs.js").read_text(encoding="utf-8")
CLIENT = {k: json.loads(v) for k, v in re.findall(r'^  ((?:err|hint|msg)_[a-z0-9_]+): (\[.*\]),$', MSGS, re.M)}


def holes(template: str) -> list[str]:
    return sorted(re.findall(r"(?<!\{)\{([a-z_][a-z0-9_]*)\}", template.replace("{{", "\0\0")))


# ------------------------------------------------------------------------------------------------ the two tables
def test_keys_are_plain_unique_and_do_not_collide():
    assert not set(ERRORS) & set(TEXTS)
    for key in [*ERRORS, *TEXTS]:
        assert M.KEY.match(key) and not key.startswith(("hint_", "err_", "msg_")), key


def test_every_catalogue_entry_is_in_the_client_in_both_languages_with_the_same_placeholders():
    wanted = {}
    for key, (message, hint) in ERRORS.items():
        wanted[f"err_{key}"] = message
        if hint:
            wanted[f"hint_{key}"] = hint
    wanted.update({f"msg_{key}": sentence for key, sentence in TEXTS.items()})
    assert set(CLIENT) == set(wanted), (sorted(set(CLIENT) ^ set(wanted))[:10])
    for key, english in wanted.items():
        es, en = CLIENT[key]
        assert es.strip() and en.strip(), key
        assert en == english, f"{key}: the English of the client differs from the backend's"
        assert holes(es) == holes(en) == holes(english), key


def test_templates_use_bare_placeholders_only():
    """No format specs or positional fields: the client fills `{name}` without Python."""
    for key, template in [*((k, v[0]) for k, v in ERRORS.items()), *((k, v[1]) for k, v in ERRORS.items()), *TEXTS.items()]:
        rest = re.sub(r"\{[a-z_][a-z0-9_]*\}", "", template.replace("{{", "").replace("}}", ""))
        assert "{" not in rest and "}" not in rest, key


def test_every_template_formats_without_leftover_placeholders():
    for key, (message, hint) in ERRORS.items():
        for template in (message, hint):
            filled = fill(template, {name: "x" for name in fields_of(template)})
            assert not re.search(r"(?<!\{)\{[a-z_]+\}", filled), key
    for key, sentence in TEXTS.items():
        assert not re.search(r"(?<!\{)\{[a-z_]+\}", fill(sentence, {name: "x" for name in fields_of(sentence)})), key


# ------------------------------------------------------------------------------------------------ the call sites
def _calls():
    for path in sorted(PACKAGE.rglob("*.py")):
        if "hoard_link" in path.parts or path.name == "messages.py":
            continue
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                yield path, node


def _name(node: ast.Call) -> str:
    f = node.func
    return f.id if isinstance(f, ast.Name) else f.attr if isinstance(f, ast.Attribute) else ""


def _literal(node: ast.AST):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def test_every_call_site_names_a_catalogue_key_and_gives_its_parameters():
    seen = 0
    for path, node in _calls():
        name = _name(node)
        if name in ("text", "msg"):
            table, index = TEXTS, 0
        elif name in ("error_text", "hint_text"):
            table, index = ERRORS, 0
        elif name in ("PygmalionError",) and len(node.args) >= 2:
            table, index = ERRORS, 1
        elif name == "WorkerError" and node.args:
            table, index = ERRORS, 0
        else:
            continue
        if len(node.args) <= index:
            continue
        key = _literal(node.args[index])
        if key is None:
            continue                      # a computed key (f"{prefix}_failed") or a message from a library
        if name == "PygmalionError" and " " in key:
            pytest.fail(f"{path.name}:{node.lineno} raises PygmalionError with a free sentence: {key!r}")
        if name in ("WorkerError", "PygmalionError") and key not in table and " " in key:
            continue
        assert key in table, f"{path.name}:{node.lineno}: {key!r} is not in the catalogue"
        template = table[key] if isinstance(table[key], str) else " ".join(table[key])
        needed = fields_of(template)
        if any(k.arg is None for k in node.keywords):
            continue
        given = {k.arg for k in node.keywords}
        if name == "WorkerError" and len(node.args) > 1:
            given.add("hint")
        if name in ("hint_text",):
            needed = fields_of(table[key][1])
        elif name in ("PygmalionError", "error_text", "WorkerError"):
            needed = fields_of(table[key][0]) | fields_of(table[key][1])
        missing = needed - given
        assert not missing, f"{path.name}:{node.lineno}: {key} lacks {sorted(missing)}"
        seen += 1
    assert seen > 150


def test_the_computed_keys_exist():
    for prefix in ("galton_{role}_failed",):
        for role in ("parent", "child"):
            assert prefix.format(role=role) in ERRORS
    for prefix in ("convert", "imatrix", "quantize", "perplexity"):
        assert f"{prefix}_failed" in ERRORS and f"{prefix}_failed_code" in ERRORS


def test_worker_keys_are_in_the_catalogue():
    for path in sorted((PACKAGE / "workers").glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call) and _name(node) == "WorkerError" and node.args and _literal(node.args[0]) and "_" in _literal(node.args[0]):
                assert _literal(node.args[0]) in ERRORS or " " in _literal(node.args[0]), (path.name, _literal(node.args[0]))


def test_an_unknown_key_is_a_programming_error_not_a_message():
    with pytest.raises(KeyError):
        PygmalionError("invalid", "no_such_key_at_all")
    with pytest.raises(KeyError):
        text("no_such_key_at_all")


# ------------------------------------------------------------------------------------------------ what an error looks like
def test_an_error_carries_key_params_message_and_hint():
    e = PygmalionError("not_found", "no_job", id="j_1")
    body = e.to_dict()
    assert body["key"] == "no_job" and body["params"] == {"id": "j_1"} and body["error"] == "No job j_1." == e.message and body["code"] == "not_found"
    assert body["hint"] == ERRORS["no_job"][1] and isinstance(body["hint"], CodedText) and body["hint"].key == "hint_no_job"


def test_a_free_text_error_has_no_key():
    e = PygmalionError("failed", "the library said so", "do this")
    assert e.key == "" and "key" not in e.to_dict() and e.to_dict()["hint"] == "do this"


def test_params_named_like_the_constructors_arguments_are_fine():
    e = PygmalionError("failed", "worker_exit", name="x", code=2, tail="t")
    assert e.params["code"] == 2 and e.status == 500
    assert PygmalionError("failed", "galton_status", status=401, detail="d").params["status"] == 401


# ------------------------------------------------------------------------------------------------ the UI gets keys, the assistant gets sentences
def test_wire_turns_coded_texts_into_items_and_leaves_the_rest():
    c = text("plan_few_records", n=7)
    assert isinstance(c, str) and c == "Only 7 training records: expect overfitting; 200 or more is a safer start."
    assert wire({"w": [c, "plain"], "n": 3}) == {"w": [{"key": "plan_few_records", "params": {"n": 7}, "text": str(c)}, "plain"], "n": 3}


def test_nested_messages_are_items_too():
    inner = text("plan_no_fit")
    outer = text("next_step_failed", reason=inner)
    item = wire(outer)
    assert item["params"]["reason"]["key"] == "plan_no_fit"


def test_the_ui_route_sends_items_and_the_agent_route_the_plain_sentence(client, base_model, dataset):
    ui = client.post("/api/ui/call", json={"name": "train_plan", "arguments": {"base": base_model["id"], "dataset": dataset["id"]}}).json()
    agent = client.post("/api/agent/call", json={"name": "train_plan", "arguments": {"base": base_model["id"], "dataset": dataset["id"]}}, headers=client.bearer).json()
    assert ui["warnings"] and all(isinstance(w, dict) and w["key"] and w["text"] for w in ui["warnings"])
    assert all(isinstance(w, str) for w in agent["warnings"]) and [w["text"] for w in ui["warnings"]] == agent["warnings"]
    assert ui["time"]["note"]["key"] == "time_note" and agent["time"]["note"] == str(text("time_note"))
    assert ui["vram"]["formula"]["key"].startswith("mem_formula") and isinstance(agent["vram"]["formula"], str)


def test_errors_reach_the_ui_with_key_params_and_a_translatable_hint_and_the_agent_with_plain_text(client):
    ui = client.post("/api/ui/call", json={"name": "job_get", "arguments": {"job": "j_nope"}})
    agent = client.post("/api/agent/call", json={"name": "job_get", "arguments": {"job": "j_nope"}}, headers=client.bearer)
    assert ui.status_code == agent.status_code == 404
    u, a = ui.json(), agent.json()
    assert u["key"] == a["key"] == "no_job" and u["params"] == {"id": "j_nope"}
    assert u["hint"] == {"key": "hint_no_job", "params": {"id": "j_nope"}, "text": ERRORS["no_job"][1]} and a["hint"] == ERRORS["no_job"][1]
    unknown = client.post("/api/ui/call", json={"name": "no_such_tool"})
    assert unknown.status_code == 404 and unknown.json()["key"] == "unknown_tool"


def test_a_stored_job_error_is_recognised_for_the_ui(client, ctx):
    job = ctx.svc.store.create_job("convert", "cpu", {}, title=str(text("title_convert")))
    ctx.svc.store.update_job(job["id"], state="failed", error=str(error_text("cancelled")), hint=ERRORS["no_job"][1])
    ui = client.post("/api/ui/call", json={"name": "job_get", "arguments": {"job": job["id"]}}).json()
    assert ui["error"]["key"] == "cancelled" and ui["title"]["key"] == "title_convert" and ui["hint"]["key"] == "hint_no_job"
    agent = client.post("/api/agent/call", json={"name": "job_get", "arguments": {"job": job["id"]}}, headers=client.bearer).json()
    assert agent["error"] == "Cancelled." and agent["title"] == "convert"


# ------------------------------------------------------------------------------------------------ recognising stored sentences
def test_every_sentence_is_recognised_back_to_its_entry():
    ambiguous = 0
    for key, template in TEXTS.items():
        if not re.sub(r"\{[a-z_]+\}", "", template).strip():
            continue
        sentence = fill(template, {name: "7" for name in fields_of(template)})
        got = recognise(sentence)
        if getattr(got, "key", "") != key:
            assert getattr(got, "key", "") and TEXTS.get(got.key) == template or ERRORS.get(got.key, ("",))[0] == template, (key, getattr(got, "key", None))
            ambiguous += 1
    assert ambiguous < 5


def test_error_messages_and_hints_are_recognised_too():
    got = recognise("No job j_9.")
    assert got.key == "no_job" and got.params == {"id": "j_9"}
    assert recognise(ERRORS["no_job"][1]).key == "hint_no_job"


def test_a_sentence_that_is_only_a_placeholder_never_matches_everything():
    assert recognise("some library error text") == "some library error text" and not getattr(recognise("some library error text"), "key", "")
    assert not hasattr(recognise("Anything at all goes here"), "key") or recognise("Anything at all goes here").key == ""


def test_nested_sentences_inside_parameters_are_recognised():
    got = recognise(str(text("next_step_failed", reason=text("cancelled_waiting_gpu") if "cancelled_waiting_gpu" in TEXTS else error_text("cancelled_waiting_gpu"))))
    assert got.key == "next_step_failed" and getattr(got.params["reason"], "key", "") == "cancelled_waiting_gpu"


def test_localise_touches_message_fields_only_and_never_user_records():
    sentence = str(text("plan_no_fit"))
    out = localise({"warnings": [sentence], "records": [{"title": sentence}], "preview": sentence, "other": sentence, "params": {"error": sentence}})
    assert out["warnings"][0].key == "plan_no_fit"
    assert out["records"][0]["title"] == sentence and not hasattr(out["records"][0]["title"], "key")
    assert not hasattr(out["other"], "key") and not hasattr(out["params"]["error"], "key") and not hasattr(out["preview"], "key")


# ------------------------------------------------------------------------------------------------ the worker's errors reach the app as keys
def test_a_worker_error_names_its_catalogue_entry_and_keeps_the_english_text():
    from pygmalion_hoard.workers import _common as C
    e = C.WorkerError("config_unreadable", path="/m")
    assert e.key == "config_unreadable" and e.params == {"path": "/m"} and e.msg == "/m: config.json is missing or unreadable." and e.hint
    free = C.WorkerError("Some library text", "a hint")
    assert free.key == "" and free.msg == "Some library text" and free.hint == "a hint"
    assert C.WorkerError("hf_read_failed", "custom hint", repo="a/b", detail="x").hint == "custom hint"


def test_the_workers_tips_are_catalogue_texts():
    from pygmalion_hoard.workers import _common as C
    assert C.hint_for("CUDA out of memory. Tried to allocate") == TEXTS["tip_oom"]
    assert C.hint_for("No module named 'peft'") == "Install it in the trainer environment: pip install peft."
    assert recognise(C.hint_for("No module named 'peft'")).key == "tip_module"
