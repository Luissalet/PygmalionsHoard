"""The Hugging Face client (mock transport), the environment check, secrets, settings and the dashboard."""

import json
import os
import stat
import sys

import httpx
import pytest

from conftest import build_services
from models import make_model
from pygmalion_hoard import envcheck, hf
from pygmalion_hoard.errors import PygmalionError


def client(handler, **kw):
    return hf.HubClient(transport=httpx.MockTransport(handler), **kw)


# ------------------------------------------------------------------------------------------------ hub client
@pytest.mark.parametrize("repo,ok", [("Qwen/Qwen3-4B", True), ("a.b/c_d-e", True), ("noslash", False), ("a/b/c", False), ("../x/y", False), ("a b/c", False), ("", False)])
def test_repo_ids(repo, ok):
    if ok:
        assert hf.check_repo(repo) == repo
    else:
        with pytest.raises(PygmalionError):
            hf.check_repo(repo)


def test_search_sends_the_filters_and_shapes_the_cards():
    seen = {}

    def handler(request):
        seen.update(path=request.url.path, params=dict(request.url.params), expand=request.url.params.get_list("expand[]"), auth=request.headers.get("authorization"))
        return httpx.Response(200, json=[
            {"id": "acme/a", "downloads": 5, "likes": 2, "pipeline_tag": "text-generation", "gated": "manual", "tags": ["license:apache-2.0", "x"],
             "config": {"architectures": ["LlamaForCausalLM"]}, "safetensors": {"total": 4_000_000_000, "parameters": {"BF16": 4_000_000_000}}, "lastModified": "2026-01-01"},
            {"modelId": "acme/b", "tags": ["safetensors"]}, "junk"])
    cards = client(handler, token=lambda: "tok123").search("  qwen  ", sort="likes", limit=500)
    assert seen["path"] == "/api/models" and seen["params"]["search"] == "qwen" and seen["params"]["limit"] == "50" and seen["params"]["sort"] == "likes"
    assert seen["params"]["pipeline_tag"] == "text-generation" and "safetensors" in seen["expand"] and seen["auth"] == "Bearer tok123"
    assert cards[0] == {"repo_id": "acme/a", "downloads": 5, "likes": 2, "pipeline": "text-generation", "gated": True, "license": "apache-2.0",
                        "architecture": "LlamaForCausalLM", "params": 4_000_000_000, "approx_bytes": 8_000_000_000, "has_safetensors": True,
                        "updated": "2026-01-01", "private": False}
    assert cards[1]["repo_id"] == "acme/b" and cards[1]["has_safetensors"] is True and len(cards) == 2


def test_search_validates_its_options():
    c = client(lambda r: httpx.Response(200, json=[]))
    with pytest.raises(PygmalionError):
        c.search("x", pipeline="audio")
    with pytest.raises(PygmalionError):
        c.search("x", sort="random")
    assert c.search("x", pipeline="") == []


@pytest.mark.parametrize("status,code", [(404, "not_found"), (401, "forbidden"), (403, "forbidden"), (500, "failed")])
def test_http_errors_become_clear_errors(status, code):
    with pytest.raises(PygmalionError) as exc:
        client(lambda r: httpx.Response(status)).info("acme/x")
    assert exc.value.code == code and exc.value.hint


def test_unreadable_answers_and_network_failures():
    with pytest.raises(PygmalionError) as exc:
        client(lambda r: httpx.Response(200, text="<html>")).info("acme/x")
    assert exc.value.code == "failed"

    def boom(request):
        raise httpx.ConnectError("no route")
    with pytest.raises(PygmalionError) as exc:
        client(boom).info("acme/x")
    assert exc.value.code == "offline" and "local models still work" in exc.value.hint


def test_offline_mode_never_touches_the_network():
    c = hf.HubClient(offline=True, transport=httpx.MockTransport(lambda r: pytest.fail("network used")))
    with pytest.raises(PygmalionError) as exc:
        c.search("x")
    assert exc.value.code == "offline"


def test_info_counts_what_a_download_would_fetch():
    files = [{"rfilename": "model-00001-of-00002.safetensors", "size": 1000}, {"rfilename": "model-00002-of-00002.safetensors", "size": 2000},
             {"rfilename": "config.json", "size": 100}, {"rfilename": "pytorch_model.bin", "size": 99999}, {"rfilename": "modeling_x.py", "size": 5},
             {"rfilename": "original/consolidated.00.pth", "size": 88888}]
    info = client(lambda r: httpx.Response(200, json={"id": "acme/m", "sha": "abc", "gated": False, "siblings": files})).info("acme/m")
    assert info["weights"] == 2 and info["files"] == 6 and info["has_python_files"] is True and info["sha"] == "abc"
    assert 3100 <= info["download_bytes"] < 90000, "the .bin and .pth copies are not downloaded"


# ------------------------------------------------------------------------------------------------ the local scan
def test_local_cards_report_what_each_model_supports(tmp_path, ctx):
    make_model(tmp_path / "hf" / "acme--one", seed=1)
    make_model(tmp_path / "hf" / "acme--odd", arch="MysteryForCausalLM", seed=2)
    (tmp_path / "hf" / "not-a-model").mkdir()
    script = ctx.svc.work.src_script("convert_hf_to_gguf.py")
    cards = hf.scan_local(tmp_path / "hf", script, {"FakeForCausalLM"})
    by = {c["name"]: c for c in cards}
    assert set(by) == {"acme--one", "acme--odd"}
    one, odd = by["acme--one"], by["acme--odd"]
    assert one["repo_id"] == "acme/one" and one["architecture"] == "FakeForCausalLM" and one["trainable"] is True and one["convertible"] is True
    assert odd["trainable"] is False and odd["convertible"] is False and "MysteryForCausalLM" in odd["convert_note"]
    assert one["layers"] == 2 and one["context"] == 2048 and one["moe"] is False and one["multimodal"] is False and one["size"] > 0
    assert hf.scan_local(tmp_path / "missing") == []
    assert hf.scan_local(tmp_path / "hf")[0]["trainable"] is None, "unknown until the trainer environment has been asked"


# ------------------------------------------------------------------------------------------------ environment check
def test_env_check_with_a_healthy_machine(ctx):
    out = ctx.svc.env.check()
    assert out["ok"] is True and out["problems"] == [] and out["trainer"]["torch"]["cuda_available"] is True
    assert all(out["capabilities"].values()), out["capabilities"]
    assert set(out["llama_binaries"]) == {"llama-quantize", "llama-imatrix", "llama-perplexity", "llama-server"} and out["ollama"]
    assert ctx.svc.env.last() is out and "FakeForCausalLM" in ctx.svc.env.trainable_architectures()


def test_env_check_names_every_gap_and_its_fix(ctx):
    ctx.svc.settings.set({"llama.bin_dir": str(ctx.tmp / "none"), "llama.src_dir": str(ctx.tmp / "none"), "publish.ollama_exe": str(ctx.tmp / "none")})
    out = ctx.svc.env.check()
    what = {p["what"]: p for p in out["problems"]}
    assert {"llama-quantize", "llama-imatrix", "llama-perplexity", "llama-server", "convert_hf_to_gguf.py", "convert_lora_to_gguf.py", "ollama"} <= set(what)
    assert what["ollama"].get("optional") is True and "git clone" in what["convert_hf_to_gguf.py"]["fix"]
    assert out["ok"] is False and out["capabilities"]["quantize"] is False and out["capabilities"]["convert"] is False and out["capabilities"]["publish_ollama"] is False
    assert out["capabilities"]["train"] is True


def test_env_check_without_a_trainer_environment(ctx):
    ctx.svc.settings.set({"env.python": ""})
    ctx.svc.work.settings.db.set_setting("env.python", "")
    out = ctx.svc.env.check()
    trainer = next(p for p in out["problems"] if p["what"] == "trainer environment")
    assert out["ok"] is False and out["capabilities"]["train"] is False and ("venv" in trainer["fix"] or "pip install" in trainer["fix"])


def test_probe_problems_are_listed_with_the_right_fix(ctx):
    fake = lambda argv, **kw: (0, json.dumps({"event": "result", "data": {"libs": {"transformers": "5.0"}, "problems": [
        {"lib": "torch", "problem": "CUDA build missing"}, {"lib": "peft", "problem": "not installed"}], "torch": {"cuda_available": False}}}) + "\n", "")
    ctx.svc.env.capture = fake
    out = ctx.svc.env.check()
    fixes = {p["what"]: p["fix"] for p in out["problems"]}
    assert "download.pytorch.org" in fixes["torch"] and "transformers peft trl" in fixes["peft"]
    assert out["capabilities"]["train"] is False and out["capabilities"]["qlora"] is False


def test_probe_errors_are_reported_not_raised(ctx):
    ctx.svc.env.capture = lambda argv, **kw: (1, json.dumps({"event": "error", "msg": "No module named 'torch'", "hint": "pip install torch"}) + "\n", "")
    assert ctx.svc.env.probe() == {"ok": False, "error": "No module named 'torch'", "hint": "pip install torch"}
    ctx.svc.env.capture = lambda argv, **kw: (1, "", "Traceback ... boom")
    out = ctx.svc.env.probe()
    assert out["ok"] is False and "boom" in out["error"]
    ctx.svc.settings.set({"env.python": str(ctx.tmp / "nope")})
    assert "does not exist" in ctx.svc.env.probe()["error"]


def test_fix_commands_quote_the_interpreter():
    cmds = envcheck.fix_commands("/opt/my env/bin/python")
    assert cmds["libs"].startswith('"/opt/my env/bin/python" -m pip install') and "cu128" in cmds["torch"]


# ------------------------------------------------------------------------------------------------ secrets and the dashboard
def test_the_token_file_is_private_and_the_token_is_used_for_downloads(ctx):
    ctx.svc.set_secret("hf.token", "  hf_secret_value_1  ")
    path = ctx.svc.config.secrets_path
    assert path.read_text(encoding="utf-8") == "PYGMALION_HF_TOKEN=hf_secret_value_1\n"
    if os.name != "nt":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert ctx.svc.config.secret("HF_TOKEN") == "hf_secret_value_1" and ctx.svc.secrets_status()["hf.token"] == {"configured": True, "source": "file"}
    assert ctx.svc.hub.token() == "hf_secret_value_1"
    ctx.svc.set_secret("hf.token", "")
    assert ctx.svc.secrets_status()["hf.token"]["configured"] is False and path.read_text(encoding="utf-8") == ""


def test_the_environment_beats_the_file_and_unknown_secrets_are_refused(ctx, monkeypatch):
    ctx.svc.set_secret("hf.token", "from-file-123")
    monkeypatch.setenv("PYGMALION_HF_TOKEN", "from-env-456")
    assert ctx.svc.config.secret("HF_TOKEN") == "from-env-456" and ctx.svc.secrets_status()["hf.token"]["source"] == "env"
    with pytest.raises(PygmalionError):
        ctx.svc.set_secret("openai.key", "x")


def test_the_dashboard_summarises_the_studio(ctx, base_model, dataset):
    d = ctx.svc.dashboard()
    assert d["counts"]["bases"] == 1 and d["counts"]["datasets"] == 1 and d["counts"]["artifacts"] == 1
    assert d["gpus"]["allowed"] == [2, 3] and {"env", "galton", "jobs", "failed", "artifacts", "storage"} <= set(d)
    assert d["env"] == {"checked": False}
    ctx.svc.store.create_artifact("gguf", "g", path="/m/g.gguf", size=5)
    ctx.svc.env.check()
    again = ctx.svc.dashboard()
    assert again["counts"]["artifacts"] == 2, "counts and jobs are always live"
    assert again["env"]["checked"] is True and again["env"]["ok"] is True, "an environment check shows in the very next dashboard"
    assert ctx.svc.dashboard()["env"] is ctx.svc.dashboard()["env"], "between checks the summary is cached"


def test_health_counts_follow_the_store(ctx, base_model):
    counts = ctx.svc.counts()
    assert counts["artifacts"] == 1 and counts["bases"] == 1 and counts["datasets"] == 0


def test_the_dashboard_follows_an_environment_check_made_through_the_tool(ctx):
    from conftest import call
    assert ctx.svc.dashboard()["env"] == {"checked": False}
    call(ctx.svc, "env_check", fresh=True)
    assert ctx.svc.dashboard()["env"]["checked"] is True


def test_dashboard_counts_read_the_same_folders_as_bases_list(ctx):
    """Two models on disk that nobody listed yet: the dashboard counts them, and bases_list returns the same two (no duplicates)."""
    from conftest import call
    from models import make_model
    make_model(ctx.work / "hf" / "acme--one", seed=1)
    make_model(ctx.work / "hf" / "acme--two", seed=2)
    (ctx.work / "hf" / "notes").mkdir()                              # not a model folder
    assert ctx.svc.dashboard()["counts"]["bases"] == 2
    assert ctx.svc.dashboard()["counts"]["artifacts"] == 2
    listed = call(ctx.svc, "bases_list")["bases"]
    assert len(listed) == 2 and ctx.svc.counts()["bases"] == 2 == ctx.svc.counts()["artifacts"]
    assert {c["artifact"] for c in listed} == {a["id"] for a in ctx.svc.store.artifacts(kind="base")}


def test_dashboard_bases_drop_when_a_folder_is_removed(ctx):
    import shutil
    from models import make_model
    folder = make_model(ctx.work / "hf" / "acme--gone", seed=3)
    assert ctx.svc.counts()["bases"] == 1
    shutil.rmtree(folder)
    assert ctx.svc.counts()["bases"] == 0


def make_fake_torch(folder, *, gpus=4):
    """An installable stand-in for torch: its CUDA sees ``gpus`` devices unless CUDA_VISIBLE_DEVICES restricts them (as the real one does)."""
    (folder / "torch").mkdir(parents=True)
    (folder / "torch-2.9.0.dist-info").mkdir()
    (folder / "torch-2.9.0.dist-info" / "METADATA").write_text("Metadata-Version: 2.1\nName: torch\nVersion: 2.9.0\n", encoding="utf-8")
    (folder / "torch" / "__init__.py").write_text(
        "import os\nfrom types import SimpleNamespace\n__version__ = '2.9.0'\nversion = SimpleNamespace(cuda='12.8')\n"
        "def _n():\n"
        "    v = os.environ.get('CUDA_VISIBLE_DEVICES')\n"
        "    return %d if v is None else len([x for x in v.split(',') if x.strip()])\n"
        "class cuda:\n"
        "    is_available = staticmethod(lambda: _n() > 0)\n"
        "    device_count = staticmethod(_n)\n"
        "    order = staticmethod(lambda: os.environ.get('CUDA_DEVICE_ORDER'))\n"
        "    get_device_properties = staticmethod(lambda i: SimpleNamespace(name='GPU %%d' %% i, total_memory=16 * 1024 ** 3, major=12, minor=0))\n" % gpus,
        encoding="utf-8")
    return folder


def test_the_probe_lists_every_gpu_even_when_the_app_was_started_with_a_restriction(ctx, monkeypatch):
    """The real probe worker in a real subprocess: the lease environment (no GPU) and an inherited restriction must not reach it."""
    from pygmalion_hoard.workdir import PACKAGE_WORKERS
    fake = make_fake_torch(ctx.tmp / "site")
    monkeypatch.setenv("PYTHONPATH", str(fake))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")                    # whoever started the app hid the GPUs
    ctx.svc.work.workers_dir = PACKAGE_WORKERS
    probe = ctx.svc.env.probe()
    assert probe["ok"] is True
    torch = probe["torch"]
    assert torch["cuda_available"] is True and torch["device_count"] == 4
    assert [(d["index"], d["name"], d["total_mb"]) for d in torch["devices"]] == [(i, f"GPU {i}", 16384) for i in range(4)]


def test_the_probe_environment_has_pci_order_and_no_gpu_restriction(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    env = envcheck.probe_environment()
    assert "CUDA_VISIBLE_DEVICES" not in env and env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


def test_the_lease_environment_still_applies_to_jobs_only():
    from pygmalion_hoard import procs
    assert procs.build_env(gpus=[2, 3])["CUDA_VISIBLE_DEVICES"] == "2,3"
    assert procs.build_env(gpus=[])["CUDA_VISIBLE_DEVICES"] == ""


def test_env_check_passes_the_probe_environment_to_the_capture(ctx, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seen = {}

    def fake(argv, **kw):
        seen.update(kw["env"])
        return 0, json.dumps({"event": "result", "data": {"libs": {}, "problems": [], "torch": {"cuda_available": True, "devices": []}}}) + "\n", ""

    ctx.svc.env.capture = fake
    ctx.svc.env.check()
    assert "CUDA_VISIBLE_DEVICES" not in seen and seen["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"
