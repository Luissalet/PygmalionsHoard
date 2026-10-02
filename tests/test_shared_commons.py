"""What comes from Hoard Link: the one-instance start, the stable token, the shared bridge and agent router, one error envelope, JSON settings next to
old plain-text rows, the secrets of .env that are never exported, the launcher script."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from pygmalion_hoard import config as config_module
from pygmalion_hoard.agent_tools import TOOLS
from pygmalion_hoard.config import Config
from pygmalion_hoard.hoard_link import net

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def running_app(tmp_path):
    """The real app (``python -m pygmalion_hoard``) in a child process on a free port."""
    port = net.free_port()
    env = {**os.environ, "PYGMALION_DATA_DIR": str(tmp_path / "data"), "PYGMALION_PORT": str(port), "PYGMALION_OFFLINE": "1", "PYGMALION_SCHEDULER": "0",
           "HOARD_NO_BROWSER": "1", "HOARD_HUB_AUTOSTART": "0", "PYTHONPATH": str(ROOT)}
    command = [sys.executable, "-m", "pygmalion_hoard"]
    child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        assert net.wait_healthy(f"http://127.0.0.1:{port}", "pygmalion-hoard", timeout=40), "the app did not start"
        yield {"port": port, "env": env, "command": command, "data": tmp_path / "data"}
    finally:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()


def test_second_start_exits_cleanly_and_keeps_the_token(running_app):
    token_file = running_app["data"] / "mcp-token"
    token = token_file.read_text(encoding="utf-8")
    second = subprocess.run(running_app["command"], cwd=ROOT, env=running_app["env"], capture_output=True, text=True, timeout=40)
    assert second.returncode == 0 and "already running" in second.stdout
    assert token_file.read_text(encoding="utf-8") == token and len(token) >= 32
    health = httpx.get(f"http://127.0.0.1:{running_app['port']}/api/health", trust_env=False).json()
    assert health["service"] == "pygmalion-hoard" and health["dataDirConfigured"] is True and health["offline"] is True and "hoard_link" in health
    assert {"counts", "scheduler"} <= set(health)
    assert (running_app["data"] / "logs").is_dir()                                   # the shared launcher logs to <data>/logs
    assert (running_app["data"] / "url").read_text(encoding="utf-8") == f"http://127.0.0.1:{running_app['port']}"


def test_the_shared_bridge_lists_and_calls_the_tools_of_the_running_app(running_app, monkeypatch):
    from pygmalion_hoard.hoard_link.bridge import CatalogBridge

    monkeypatch.setenv("PYGMALION_DATA_DIR", str(running_app["data"]))
    monkeypatch.setenv("PYGMALION_URL", f"http://127.0.0.1:{running_app['port']}")
    monkeypatch.setenv("PYGMALION_BRIDGE_AUTOSTART", "0")
    bridge = CatalogBridge(app="pygmalion", service="pygmalion-hoard", package="pygmalion_hoard", default_port=5202, data_dir_env="PYGMALION_DATA_DIR",
                           title="Pygmalion's Hoard", root=str(ROOT / "mcp_server.py"), default_timeout=90.0)

    async def go():
        tools = await bridge.tools()
        settings = await bridge.call("settings_get", {})
        missing = await bridge.call("dataset_get", {"dataset": "nada"})
        return tools, settings, missing

    tools, settings, missing = asyncio.run(go())
    assert len(tools) == len(TOOLS)
    assert not settings.is_error and "values" in settings.body or "settings" in settings.body or not settings.is_error
    assert missing.is_error and missing.body["code"] == "not_found" and missing.body["hint"] and missing.body["key"]


def test_the_agent_router_answers_every_failure_as_json(client):
    call = lambda body, **kw: client.post("/api/agent/call", headers=client.bearer, json=body, **kw)  # noqa: E731
    assert client.post("/api/agent/call", json={"name": "settings_get"}).status_code == 401
    unknown = call({"name": "no_such_tool"})
    assert unknown.status_code == 404 and unknown.json()["code"] == "unknown_tool"
    bad = call({"name": "hf_search", "arguments": {"limit": 9999}})
    assert bad.status_code == 400 and bad.json()["code"] == "invalid_arguments" and bad.json()["issues"]
    missing = call({"name": "dataset_get", "arguments": {"dataset": "nada"}})
    assert missing.status_code == 404 and missing.json()["code"] == "not_found" and missing.json()["key"] and missing.json()["hint"]
    assert client.get("/api/agent/tools").json()["tools"][0]["name"] == TOOLS[0].name


def test_agent_results_are_capped_but_the_ui_gets_everything(client):
    from pygmalion_hoard.agent_tools import call_tool, uncapped

    svc = client.svc
    for i in range(80):
        svc.store.create_artifact("adapter", f"adapter-{i:03d}", notes="n" * 600)
    capped = call_tool(svc, "artifacts_list", {"limit": 200})
    with uncapped():
        full = call_tool(svc, "artifacts_list", {"limit": 200})
    assert len(full["artifacts"]) == 80 and "truncated" not in full
    assert "truncated" in capped and capped["truncated"]["original_lengths"]["artifacts"] == 80 and len(capped["artifacts"]) < 80


def test_old_settings_rows_written_as_plain_text_still_work(ctx):
    svc = ctx.svc
    # what the previous version wrote: the text 1, a number, a list of GPU numbers and a path
    svc.db.execute("INSERT OR REPLACE INTO settings(key, value) VALUES ('train.rank', '16'), ('train.dropout', '0.05'), ('gpus.allowed', '0,1'), "
                   "('publish.ollama_exe', '/opt/ollama/ollama')")
    assert svc.settings.get("train.rank") == "16" and svc.settings.int("train.rank") == 16
    assert svc.settings.float("train.dropout") == 0.05
    assert svc.settings.intlist("gpus.allowed") == [0, 1]
    assert svc.settings.get("publish.ollama_exe") == "/opt/ollama/ollama"
    svc.settings.set({"train.rank": 32, "gpus.allowed": [0]}, confirm_reserved=True)
    assert svc.settings.int("train.rank") == 32 and svc.settings.intlist("gpus.allowed") == [0]


def test_the_token_and_url_files_are_stable_and_atomic(tmp_path):
    from conftest import build_services

    a = build_services(tmp_path / "a")
    token = a.svc.token
    a.svc.stop()
    b = build_services(tmp_path / "a")
    try:
        assert b.svc.token == token and (a.data / "mcp-token").read_text() == token and len(token) >= 32
        assert (a.data / "url").read_text().startswith("http://127.0.0.1:")
        assert sorted(p.name for p in a.data.glob("*.tmp*")) == []
    finally:
        b.svc.stop()


def test_the_env_file_is_read_by_the_shared_parser_and_never_exported(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text('# comment\nexport PYGMALION_HF_TOKEN="hf_abc"\nPYGMALION_OTHER=x # inline\n', encoding="utf-8")
    monkeypatch.delenv("PYGMALION_HF_TOKEN", raising=False)
    monkeypatch.delenv("PYGMALION_OTHER", raising=False)
    values = config_module.load_dotenv(env_file)
    assert values["PYGMALION_HF_TOKEN"] == "hf_abc" and values["PYGMALION_OTHER"] == "x"
    assert "PYGMALION_HF_TOKEN" not in os.environ and "PYGMALION_OTHER" not in os.environ   # a trainer child would inherit them otherwise
    monkeypatch.setenv("PYGMALION_OTHER", "real")
    assert config_module.load_dotenv(env_file)["PYGMALION_OTHER"] == "x" and os.environ["PYGMALION_OTHER"] == "real"


def test_ports_and_timeouts_out_of_range_fall_back_to_the_defaults(monkeypatch, tmp_path):
    monkeypatch.setenv("PYGMALION_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PYGMALION_PORT", "99999")
    monkeypatch.setenv("PYGMALION_HTTP_TIMEOUT_S", "9999")
    monkeypatch.delenv("PORT", raising=False)
    config = Config.from_env()
    assert config.port == config_module.DEFAULT_PORT == 5202 and config.http_timeout_s == 25.0 and config.data_dir_configured is True
    monkeypatch.setenv("PYGMALION_PORT", "5300")
    monkeypatch.setenv("PYGMALION_HTTP_TIMEOUT_S", "30")
    config = Config.from_env()
    assert config.port == 5300 and config.http_timeout_s == 30.0
    assert config.db_path == tmp_path / "pygmalion.db" and config.token_path == tmp_path / "mcp-token" and config.logs_dir == tmp_path / "logs"


def test_the_launcher_script_imports_the_shared_net_helpers():
    import importlib.util

    spec = importlib.util.spec_from_file_location("pygmalion_launch_script", ROOT / "scripts" / "launch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # a deleted helper module would fail here, not on the user's double click
    assert module.SERVICE == "pygmalion-hoard" and callable(module.main)


# ---------------------------------------------------------------- atomic files
def test_json_files_are_written_atomically_and_survive_a_file_held_by_a_reader(tmp_path, monkeypatch):
    import time

    from pygmalion_hoard import util
    from pygmalion_hoard.hoard_link import atomic

    target = tmp_path / "x" / "state.json"
    util.write_json_atomic(target, {"a": 1})
    assert target.read_text(encoding="utf-8") == '{\n  "a": 1\n}\n'
    real, calls = os.replace, {"n": 0}

    def flaky(src, dst):                      # Windows: the destination is open in a reader for a moment
        calls["n"] += 1
        if calls["n"] < 3:
            raise PermissionError(13, "in use")
        return real(src, dst)

    monkeypatch.setattr(atomic.os, "replace", flaky)
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    util.write_json_atomic(target, {"a": 2})
    assert calls["n"] == 3 and util.read_json(target) == {"a": 2}
    assert [p.name for p in target.parent.iterdir()] == ["state.json"]          # no temp file left


def test_a_secret_is_written_atomically_and_owner_only(ctx):
    svc = ctx.svc
    svc.set_secret("hf.token", "hf_abcdefghijklmnop")
    path = svc.config.secrets_path
    assert "hf_abcdefghijklmnop" in path.read_text(encoding="utf-8")
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o600
    svc.set_secret("hf.token", "")
    assert "hf_abcdefghijklmnop" not in path.read_text(encoding="utf-8")
    assert [p.name for p in path.parent.iterdir() if p.name.endswith(".tmp")] == []


def test_a_dataset_version_file_is_replaced_whole_and_a_failed_write_leaves_no_temp_file(ctx, monkeypatch):
    from pygmalion_hoard.hoard_link import atomic

    from helpers import chat_records

    def build(name):
        return ctx.svc.datasets.build({"name": name, "kind": "chat", "sources": [{"type": "jsonl", "text": "\n".join(json.dumps(r) for r in chat_records(30))}]})

    ds = ctx.svc.datasets
    out = build("Atomico")
    folder = ds.dir / ctx.svc.store.version(out["version"]["id"])["dataset_id"]
    assert [p.name for p in folder.iterdir() if p.name.endswith(".tmp")] == [] and (folder / "v1.jsonl").is_file()

    def boom(src, dst, **kw):
        raise OSError("disk full")

    monkeypatch.setattr(atomic, "replace_with_retry", boom)
    with pytest.raises(OSError):
        build("Roto")
    assert [p for p in ds.dir.rglob("*") if p.name.endswith(".tmp")] == []


def test_the_workers_write_their_files_through_the_shared_atomic_module(tmp_path, monkeypatch):
    import models  # noqa: F401  (puts the workers folder on sys.path)
    import _atomic
    import train_lora

    assert _atomic._atomic is not None                      # the vendored copy next to the workers was found
    state = tmp_path / "ck" / "training_state.json"
    state.parent.mkdir()
    train_lora.write_state(state, {"step": 3})
    assert json.loads(state.read_text(encoding="utf-8")) == {"step": 3}
    assert [p.name for p in state.parent.iterdir()] == ["training_state.json"]


def test_the_worker_atomic_helpers_fall_back_when_the_shared_copy_is_missing(tmp_path, monkeypatch):
    import models  # noqa: F401
    import _atomic

    monkeypatch.setattr(_atomic, "_atomic", None)
    target = tmp_path / "f.json"
    _atomic.write_text_atomic(target, "{}")
    other = tmp_path / "g.json"
    other.write_text("old", encoding="utf-8")
    _atomic.replace_with_retry(target, other)
    assert other.read_text(encoding="utf-8") == "{}" and not target.exists()


# ---------------------------------------------------------------- child processes
def test_children_run_with_utf8_and_the_pci_gpu_order(monkeypatch):
    from pygmalion_hoard import procs

    monkeypatch.delenv("PYTHONUNBUFFERED", raising=False)
    env = procs.build_env({"X": 1}, gpus=[2, 0])
    assert env["PYTHONUTF8"] == "1" and env["PYTHONIOENCODING"] == "utf-8" and env["PYTHONUNBUFFERED"] == "1"
    assert env["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID" and env["CUDA_VISIBLE_DEVICES"] == "2,0" and env["X"] == "1"
    assert procs.build_env(gpus=[])["CUDA_VISIBLE_DEVICES"] == ""


@pytest.mark.skipif(os.name == "nt", reason="uses sh")
def test_kill_tree_stops_the_grandchildren_too(tmp_path):
    import time

    from pygmalion_hoard import procs
    from pygmalion_hoard.hoard_link import proc as hl_proc

    flag = tmp_path / "grandchild.pid"
    child = hl_proc.popen(["sh", "-c", f"sleep 60 & echo $! > {flag}; wait"])
    deadline = time.monotonic() + 10
    while not flag.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    grandchild = int(flag.read_text().strip())
    assert hl_proc.pid_alive(grandchild)
    procs.kill_tree(child)
    deadline = time.monotonic() + 10
    while hl_proc.pid_alive(grandchild) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert child.poll() is not None and not hl_proc.pid_alive(grandchild)


def test_run_capture_reports_a_timeout_and_a_missing_program_without_raising(tmp_path):
    from pygmalion_hoard import procs

    code, out, err = procs.run_capture([sys.executable, "-c", "print('hola')"])
    assert code == 0 and out.strip() == "hola"
    code, out, err = procs.run_capture([sys.executable, "-c", "import time; time.sleep(30)"], timeout_s=0.5)
    assert code is None and "timed out" in err
    code, out, err = procs.run_capture([str(tmp_path / "no-such-program")])
    assert code is None and err
