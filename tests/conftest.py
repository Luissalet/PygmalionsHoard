"""Fixtures: a fully wired Services object with a fake trainer environment, fake llama.cpp programs, a fake GPU inventory and no network."""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pygmalion_hoard import procs, settings as settings_module, workdir  # noqa: E402
from pygmalion_hoard.config import Config  # noqa: E402
from pygmalion_hoard.services import Services  # noqa: E402

import helpers  # noqa: E402
from helpers import FakeLeaseFactory, chat_records, fake_inventory, make_toolchain  # noqa: E402
from models import make_model  # noqa: E402

FAKE_WORKERS = Path(__file__).resolve().parent / "fake_workers"
needs_posix = pytest.mark.skipif(not helpers.EXECUTABLE_TOOLS, reason=helpers.POSIX_ONLY)

# Environment variables that change what the app finds on the machine: a test that wants one sets it itself.
HOST_ENV_PREFIXES = ("PYGMALION_", "GALTON_", "HOARD_", "OLLAMA_", "HF_", "HUGGING", "TRANSFORMERS_", "CUDA_")
HOST_ENV_NAMES = ("LOCALAPPDATA", "APPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMDATA", "PORT", "PORT_STRICT")


@pytest.fixture(autouse=True)
def hermetic_host(tmp_path_factory, monkeypatch):
    """No test depends on the computer it runs on. The real defaults stay in the product (the Windows work folder, the trainer's venv,
    llama.cpp, Ollama in its install folders, programs on the PATH, OLLAMA_MODELS...) and are looked up in the same functions as
    always; this fixture only points those lookups at a place where nothing exists, unless a test builds something there."""
    nowhere = tmp_path_factory.getbasetemp() / "no-such-host"          # never created
    for name in list(os.environ):
        if name.upper().startswith(HOST_ENV_PREFIXES) or name.upper() in HOST_ENV_NAMES:
            monkeypatch.delenv(name)
    for name in ("LOCALAPPDATA", "APPDATA", "ProgramFiles", "ProgramFiles(x86)", "ProgramData"):
        monkeypatch.setenv(name, str(nowhere / name))
    monkeypatch.setenv("HOARD_HUB_AUTOSTART", "0")                     # never start a real hub
    monkeypatch.setattr(settings_module, "WIN_WORK", str(nowhere / "pygmalion"))
    monkeypatch.setattr(settings_module, "WIN_PYTHON", str(nowhere / "pygmalion" / "venv" / "Scripts" / "python.exe"))
    monkeypatch.setattr(settings_module, "WIN_LLAMA", str(nowhere / "llama.cpp"))
    monkeypatch.setattr(workdir, "OLLAMA_SYSTEM_PATHS", ())
    monkeypatch.setattr(procs, "which", lambda name: None)


def build_services(tmp_path: Path, *, offline: bool = True, with_tools: bool = True, **kwargs: Any) -> SimpleNamespace:
    """Services wired for tests. Returns a namespace: ``svc``, the tool folders, the lease factory and the work folder."""
    data = tmp_path / "data"
    work = tmp_path / "work"
    config = Config(data_dir=data, offline=offline, scheduler=False, workers_dir=FAKE_WORKERS, data_dir_configured=True)
    leases = FakeLeaseFactory()
    inventory = kwargs.pop("inventory", None)
    kwargs.setdefault("probe", lambda: inventory if inventory is not None else fake_inventory())
    kwargs.setdefault("lease_factory", leases)
    kwargs.setdefault("hub_status", lambda: None)
    svc = Services(config, **kwargs)
    tools = make_toolchain(tmp_path / "tools") if with_tools else {"bin": tmp_path / "tools" / "bin", "src": tmp_path / "tools" / "src",
                                                                    "ollama": tmp_path / "tools" / "bin" / f"ollama{workdir.EXE}", "ollama_log": tmp_path / "ollama.jsonl"}
    svc.settings.set({"env.python": sys.executable, "paths.work": str(work), "llama.bin_dir": str(tools["bin"]), "llama.src_dir": str(tools["src"]),
                      "publish.ollama_exe": str(tools["ollama"])})
    svc.work.ensure()
    return SimpleNamespace(svc=svc, tools=tools, leases=leases, work=work, tmp=tmp_path, data=data)


@pytest.fixture
def ctx(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_OLLAMA_LOG", str(tmp_path / "ollama-calls.jsonl"))
    monkeypatch.setenv("HOARD_HOME", str(tmp_path / "hoard-home"))
    monkeypatch.setenv("PYGMALION_OFFLINE", "1")
    c = build_services(tmp_path)
    yield c
    c.svc.stop()


@pytest.fixture
def svc(ctx):
    return ctx.svc


@pytest.fixture
def base_model(ctx):
    """A tiny base model registered as an artifact."""
    folder = make_model(ctx.work / "hf" / "acme--tiny-base", seed=1)
    return ctx.svc.lineage.ensure_base(folder, "acme/tiny-base")


@pytest.fixture
def dataset(ctx):
    """A small chat dataset (60 records) built through the real service."""
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in chat_records(60))
    out = ctx.svc.datasets.build({"name": "Casa de la abuela", "sources": [{"type": "jsonl", "text": text}]})
    return out["version"]


def call(svc: Services, name: str, /, **arguments: Any) -> Any:
    from pygmalion_hoard.agent_tools import call_tool, uncapped
    with uncapped():
        return call_tool(svc, name, arguments)


def finish(svc: Services, answer: dict) -> dict:
    """Run a queued job (and the pipeline behind it) to the end inline and return the first job."""
    first = answer["job"]["id"]
    svc.jobs.run_until_idle()
    return svc.store.job(first)


@pytest.fixture
def client(ctx):
    """The real app over the wired services, with the MCP bearer token ready."""
    from fastapi.testclient import TestClient
    from pygmalion_hoard.main import create_app

    app = create_app(ctx.svc.config, services=ctx.svc)
    with TestClient(app, base_url="http://127.0.0.1") as c:
        c.svc = ctx.svc
        c.ctx = ctx
        c.bearer = {"Authorization": f"Bearer {ctx.svc.token}"}
        yield c


# ------------------------------------------------------------------------------------------------- no thread outlives its test
#: threads the app and its vendored link start; every one of them must have ended when the test that started it is over. A daemon thread that is
#: still running when the interpreter shuts down can be killed in the middle of a write or of a call into a C extension, and the whole run
#: then ends with a fatal error although every test passed.
APP_THREAD_PREFIXES = ("pygmalion-", "hoard-")
SETTLE_S = 5.0


def lingering_threads() -> list[threading.Thread]:
    return [t for t in threading.enumerate() if t is not threading.main_thread() and t.name.startswith(APP_THREAD_PREFIXES) and t.is_alive()]


@pytest.fixture(autouse=True)
def no_lingering_threads():
    """After each test give the threads of the app a moment to end (an event worker finishing its last call, a lease helper noticing that it
    was abandoned), then fail the test that left one running. Services, job managers and lease helpers are stopped by the fixtures."""
    before = {id(t) for t in lingering_threads()}
    yield
    deadline = time.monotonic() + SETTLE_S
    while time.monotonic() < deadline:
        left = [t for t in lingering_threads() if id(t) not in before]
        if not left:
            return
        time.sleep(0.05)
    left = [t for t in lingering_threads() if id(t) not in before]
    if left:
        for t in left:                                    # do not let one leaking test fail all the ones after it
            t.join(0.01)
        pytest.fail("threads still running after the test: " + ", ".join(f"{t.name} (daemon={t.daemon})" for t in left))
