"""Modelfiles, tag names, llama.cpp backends and the rules that keep publishing from touching what it did not create."""

import json
import socket
from pathlib import Path

import pytest

from conftest import call, needs_posix
from gguf_builder import build_gguf
from pygmalion_hoard import publish as P
from pygmalion_hoard.errors import PygmalionError


@pytest.fixture
def gguf(ctx, tmp_path):
    path = build_gguf(tmp_path / "models" / "my model.gguf", context=4096, size=4096)
    return ctx.svc.store.create_artifact("gguf", "my-model-f16", path=str(path), size=path.stat().st_size)


def backends(ctx):
    path = Path(ctx.tmp) / "hoard-home" / "backends.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"commands": []}


# ------------------------------------------------------------------------------------------------ names and Modelfile
@pytest.mark.parametrize("name,tag,expected", [
    ("Casa Abuela", "v1", "pyg-casa-abuela:v1"), ("pyg-ya-tiene", "latest", "pyg-ya-tiene:latest"), ("Qwen3 4B_x", "", "pyg-qwen3-4b-x:latest"),
    ("ñandú", "Q4_K_M", "pyg-nandu:q4-k-m"), ("a" * 200, "t", "pyg-" + "a" * 56 + ":t"),
])
def test_ollama_names_are_always_prefixed_and_lowercase(name, tag, expected):
    assert P.ollama_name(name, tag) == expected
    assert P.OLLAMA_NAME_RE.match(expected)


def test_ollama_name_falls_back_to_a_neutral_name_for_symbols_only():
    assert P.ollama_name("***", "!!") == "pyg-model:model"
    assert P.ollama_name("") == "pyg-model:latest"


def test_modelfile_quotes_paths_with_spaces_and_uses_forward_slashes():
    assert P.modelfile("C:\\Users\\me\\my model.gguf").splitlines()[0] == 'FROM "C:/Users/me/my model.gguf"'
    assert P.modelfile("/m/a.gguf") == "FROM /m/a.gguf\n"


def test_modelfile_options_in_order():
    text = P.modelfile("/m/a.gguf", adapter="/m/lora.gguf", num_ctx=8192, system="Eres amable.", template="{{ .Prompt }}", parameters={"temperature": 0.7, "stop": ["<a>", "<b>"]})
    lines = text.splitlines()
    assert lines[:3] == ["FROM /m/a.gguf", "ADAPTER /m/lora.gguf", "PARAMETER num_ctx 8192"]
    assert "PARAMETER temperature 0.7" in lines and lines.count("PARAMETER stop <a>") == 1 and "PARAMETER stop <b>" in lines
    assert 'TEMPLATE """{{ .Prompt }}"""' in text and 'SYSTEM """Eres amable."""' in text


def test_modelfile_cannot_be_broken_out_of_by_the_prompt_text():
    text = P.modelfile("/m/a.gguf", system='x""" \nPARAMETER num_ctx 1\nFROM /etc/passwd')
    assert text.count('"""') == 2, "the triple quotes inside the text are escaped"
    with pytest.raises(PygmalionError):
        P.modelfile("/m/a.gguf", parameters={"bad key\nFROM x": 1})


def test_port_free_detects_a_listener():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        assert P.port_free(s.getsockname()[1]) is False
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        free = s.getsockname()[1]
    assert P.port_free(free) is True


# ------------------------------------------------------------------------------------------------ plans
def test_ollama_plan_reads_the_gguf_and_adds_notes(ctx, gguf):
    plan = ctx.svc.publisher.ollama_plan(gguf["id"], "Mi Modelo", "v2")
    assert plan["tag"] == "pyg-mi-modelo:v2" and plan["num_ctx"] == 4096 and 'my model.gguf"' in plan["modelfile"]
    assert plan["notes"] == [] and plan["summary"]["context_length"] == 4096


def test_ollama_plan_notes_a_missing_template_and_rope_scaling(ctx, tmp_path):
    path = build_gguf(tmp_path / "r.gguf", context=8192, chat_template=False, rope={"scaling.type": "yarn", "scaling.factor": 4.0, "scaling.original_context_length": 2048})
    art = ctx.svc.store.create_artifact("gguf", "r", path=str(path), size=1)
    plan = ctx.svc.publisher.ollama_plan(art["id"])
    assert any("yarn" in n for n in plan["notes"]) and any("no chat template" in n for n in plan["notes"])
    assert not any("no chat template" in n for n in ctx.svc.publisher.ollama_plan(art["id"], template="{{ .Prompt }}")["notes"])


def test_ollama_plan_needs_an_existing_gguf_file(ctx):
    ghost = ctx.svc.store.create_artifact("gguf", "ghost", path="/nowhere/x.gguf")
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.publisher.ollama_plan(ghost["id"])
    assert exc.value.code == "not_found"
    adapter = ctx.svc.store.create_artifact("adapter", "ad")
    with pytest.raises(PygmalionError):
        ctx.svc.publisher.ollama_plan(adapter["id"])


def test_publish_ollama_without_ollama_installed(ctx, gguf):
    ctx.svc.settings.set({"publish.ollama_exe": str(ctx.tmp / "no-ollama")})
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.publisher.publish_ollama(gguf["id"])
    assert exc.value.code == "tool_missing" and "publish.ollama_exe" in exc.value.hint


@needs_posix
def test_republishing_the_same_tag_updates_the_one_artifact(ctx, gguf):
    first = ctx.svc.publisher.publish_ollama(gguf["id"], name="x", tag="v1")
    second = ctx.svc.publisher.publish_ollama(gguf["id"], name="x", tag="v1", num_ctx=2048)
    assert first["artifact"] == second["artifact"] and len(ctx.svc.store.artifacts(kind="ollama")) == 1
    assert "num_ctx 2048" in ctx.svc.store.artifact(second["artifact"])["metrics"]["modelfile"]
    assert len(ctx.svc.store.artifact(gguf["id"])["published"]) == 1


@needs_posix
def test_an_adapter_gguf_becomes_the_adapter_line(ctx, gguf, tmp_path):
    lora = build_gguf(tmp_path / "lora.gguf", size=2048)
    adapter = ctx.svc.store.create_artifact("gguf", "lora-gguf", path=str(lora), size=1)
    out = ctx.svc.publisher.publish_ollama(gguf["id"], name="con-lora", adapter=adapter["id"])
    assert f"ADAPTER {lora}" in out["modelfile"]
    assert set(ctx.svc.store.artifact(out["artifact"])["parents"]) == {gguf["id"], adapter["id"]}


# ------------------------------------------------------------------------------------------------ llama.cpp backends
@needs_posix
def test_backend_entry_binds_one_allowed_gpu_and_loopback(ctx, gguf):
    entry = ctx.svc.publisher.backend_entry(gguf, backend_id="pyg-x", port=8100, ctx=4096, ngl=99, gpu=3, extra=["--rope-scaling", "yarn"])
    argv = entry["argv"]
    assert argv[0].endswith("llama-server") and argv[argv.index("--host") + 1] == "127.0.0.1" and argv[argv.index("--port") + 1] == "8100"
    assert argv[argv.index("-m") + 1] == gguf["path"] and argv[-2:] == ["--rope-scaling", "yarn"] and "--jinja" in argv
    assert entry["env"] == {"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "3"} and entry["health"] == "http://127.0.0.1:8100/health" and entry["capabilities"] == ["llm"]


@needs_posix
def test_backend_entry_without_a_server_binary(ctx, gguf):
    ctx.svc.settings.set({"llama.bin_dir": str(ctx.tmp / "empty")})
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.publisher.backend_entry(gguf, backend_id="x", port=8100, ctx=1, ngl=1, gpu=None, extra=[])
    assert exc.value.code == "tool_missing"


@needs_posix
def test_ports_are_taken_in_order_skipping_used_and_busy_ones(ctx, gguf):
    pub = ctx.svc.publisher
    start = ctx.svc.settings.int("publish.llama_port_start")
    pub.is_free = lambda port: port != start + 1
    first = pub.publish_llama(gguf["id"], name="uno")
    second = pub.publish_llama(gguf["id"], name="dos")
    assert first["port"] == start and second["port"] == start + 2, "the busy port start+1 is skipped, the one already in backends.json too"
    assert [c["id"] for c in backends(ctx)["commands"]] == ["pyg-uno", "pyg-dos"]


@needs_posix
def test_publishing_again_replaces_the_backend_instead_of_duplicating(ctx, gguf):
    pub = ctx.svc.publisher
    pub.publish_llama(gguf["id"], name="uno", ctx=2048)
    pub.publish_llama(gguf["id"], name="uno", ctx=4096)
    commands = backends(ctx)["commands"]
    assert len(commands) == 1 and commands[0]["argv"][commands[0]["argv"].index("-c") + 1] == "4096"


@needs_posix
def test_default_context_is_capped_and_gpu_is_the_first_allowed(ctx, tmp_path):
    big = build_gguf(tmp_path / "big.gguf", context=131072)
    art = ctx.svc.store.create_artifact("gguf", "big", path=str(big), size=1)
    entry = ctx.svc.publisher.publish_llama(art["id"])["entry"]
    assert entry["argv"][entry["argv"].index("-c") + 1] == "32768" and entry["env"]["CUDA_VISIBLE_DEVICES"] == "2"


@needs_posix
def test_a_reserved_gpu_is_refused(ctx, gguf):
    for gpu in (0, 1):
        with pytest.raises(PygmalionError) as exc:
            ctx.svc.publisher.publish_llama(gguf["id"], gpu=gpu)
        assert exc.value.code == "forbidden"
    assert backends(ctx)["commands"] == []


@needs_posix
def test_the_backend_id_and_extra_arguments_are_kept_safe(ctx, gguf):
    out = ctx.svc.publisher.publish_llama(gguf["id"], name="Con Espacios; rm -rf /", extra_args=["--x", 3])
    assert P.BACKEND_ID_RE.match(out["id"]) and ";" not in out["id"] and out["entry"]["argv"][-2:] == ["--x", "3"]


# ------------------------------------------------------------------------------------------------ unpublish
def test_unpublish_something_that_is_not_published(ctx, gguf):
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.publisher.unpublish(gguf["id"])
    assert exc.value.code == "not_found"


@needs_posix
def test_unpublish_only_removes_what_matches(ctx, gguf):
    pub = ctx.svc.publisher
    pub.publish_llama(gguf["id"], name="uno")
    pub.publish_ollama(gguf["id"], name="uno", tag="v1")
    with pytest.raises(PygmalionError):
        pub.unpublish(gguf["id"], target="llama", name="otro")
    out = pub.unpublish(gguf["id"], target="llama")
    assert out["removed"] == [{"type": "llama", "name": "pyg-uno"}] and backends(ctx)["commands"] == []
    assert [p["type"] for p in ctx.svc.store.artifact(gguf["id"])["published"]] == ["ollama"]


@needs_posix
def test_unpublishing_the_ollama_artifact_itself(ctx, gguf):
    out = ctx.svc.publisher.publish_ollama(gguf["id"], name="uno", tag="v1")
    ctx.svc.publisher.unpublish(out["artifact"])
    assert ctx.svc.store.artifact(gguf["id"])["published"] == []
    assert "Unpublished" in ctx.svc.store.artifact(out["artifact"])["notes"]
    assert "unpublished_ts" in ctx.svc.store.artifact(out["artifact"])["metrics"]


@needs_posix
def test_a_tag_not_made_by_this_app_is_never_removed(ctx, gguf):
    ctx.svc.lineage.mark_published(gguf["id"], {"type": "ollama", "name": "llama3:8b"})
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.publisher.unpublish(gguf["id"])
    assert exc.value.code == "forbidden"
    log = Path(ctx.tmp) / "ollama-calls.jsonl"
    assert not log.exists() or "rm" not in log.read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ the context written into the Modelfile
def long_context(ctx, tmp_path, context=262144, name="long"):
    path = build_gguf(tmp_path / f"{name}.gguf", context=context, size=4096)
    return ctx.svc.store.create_artifact("gguf", name, path=str(path), size=4096)


def test_the_modelfile_does_not_ask_for_the_whole_trained_context(ctx, tmp_path):
    art = long_context(ctx, tmp_path)
    plan = ctx.svc.publisher.ollama_plan(art["id"])
    assert plan["num_ctx"] == 8192 and "PARAMETER num_ctx 8192" in plan["modelfile"] and "262144" not in plan["modelfile"]
    assert plan["summary"]["context_length"] == 262144 and plan["notes"] == []


def test_the_default_follows_the_setting_but_never_goes_above_the_trained_context(ctx, tmp_path):
    ctx.svc.settings.set({"publish.num_ctx": 16384})
    assert ctx.svc.publisher.ollama_plan(long_context(ctx, tmp_path)["id"])["num_ctx"] == 16384
    short = long_context(ctx, tmp_path, context=4096, name="short")
    assert ctx.svc.publisher.ollama_plan(short["id"])["num_ctx"] == 4096


def test_a_context_the_caller_asks_for_is_kept_and_noted_when_above_the_trained_one(ctx, tmp_path):
    art = long_context(ctx, tmp_path, context=4096)
    assert ctx.svc.publisher.ollama_plan(art["id"], num_ctx=2048)["num_ctx"] == 2048
    plan = ctx.svc.publisher.ollama_plan(art["id"], num_ctx=8192)
    assert plan["num_ctx"] == 8192 and plan["notes"]


def test_a_context_variant_is_published_at_the_length_it_was_built_for(ctx, tmp_path):
    base = long_context(ctx, tmp_path, context=32768, name="plain")
    path = build_gguf(tmp_path / "variant.gguf", context=131072, size=4096)
    variant = ctx.svc.store.create_artifact("ctx_variant", "v128k", path=str(tmp_path / "x"), parents=[base["id"]])
    child = ctx.svc.store.create_artifact("gguf", "v128k-gguf", path=str(path), size=4096, parents=[variant["id"]])
    assert ctx.svc.publisher.ollama_plan(child["id"])["num_ctx"] == 131072
