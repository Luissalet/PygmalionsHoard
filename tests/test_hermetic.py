"""The tests do not depend on the computer they run on, and the real defaults still work: the machine's defaults (the Windows work folder,
the trainer's venv, llama.cpp, Ollama in its install folders, the PATH, PYGMALION_*/OLLAMA_*/HF_* variables) are neutralised by the
``hermetic_host`` fixture in conftest, and these tests check both halves: nothing of the host is visible, and each default is still honoured
when something really is there."""

import os
from pathlib import Path

import pytest

from pygmalion_hoard import procs, settings as settings_module, workdir
from pygmalion_hoard.db import Database
from pygmalion_hoard.settings import Settings
from pygmalion_hoard.workdir import EXE, Work


@pytest.fixture
def fresh(tmp_path):
    """A settings store with nothing configured, as on a first start."""
    db = Database(tmp_path / "fresh.db")
    yield lambda windows, env=None: Settings(db, tmp_path / "data", env={} if env is None else env, platform_is_windows=windows)
    db.close()


# ------------------------------------------------------------------------------------------------ nothing of the host is visible
def test_no_variable_of_the_host_reaches_the_tests():
    leaked = [k for k in os.environ if k.upper().startswith(("PYGMALION_", "GALTON_", "OLLAMA_", "HF_", "HUGGING", "TRANSFORMERS_", "CUDA_"))]
    assert leaked == []
    assert not Path(os.environ["LOCALAPPDATA"]).exists() and os.environ["HOARD_HUB_AUTOSTART"] == "0"


def test_no_program_of_the_host_is_found():
    assert procs.which("ollama") is None and procs.which("llama-server") is None and procs.which("nvidia-smi") is None
    assert workdir.OLLAMA_SYSTEM_PATHS == ()


def test_no_default_folder_of_the_host_is_used(tmp_path, fresh):
    for windows in (True, False):
        assert fresh(windows).default_work() == str(tmp_path / "data" / "work")
    assert not Path(settings_module.WIN_WORK).exists() and not Path(settings_module.WIN_PYTHON).exists() and not Path(settings_module.WIN_LLAMA).exists()
    assert fresh(True).default_python() == ""


def test_a_fresh_install_has_nothing_configured_by_the_machine(ctx, tmp_path):
    from conftest import build_services
    c = build_services(tmp_path / "fresh-install", with_tools=False)
    try:
        c.svc.settings.set({"env.python": "", "llama.bin_dir": "", "llama.src_dir": "", "publish.ollama_exe": ""})
        w = c.svc.work
        assert w.python() == "" and not w.python_ok() and w.llama_bin("llama-quantize") is None and w.ollama_exe() is None
        assert not Path(w.settings.get("llama.src_dir")).exists()
    finally:
        c.svc.stop()


# ------------------------------------------------------------------------------------------------ the real defaults still apply
def test_the_windows_work_folder_and_trainer_venv_are_the_defaults_when_they_exist(tmp_path, monkeypatch, fresh):
    work = tmp_path / "studio" / "pygmalion"
    python = work / "venv" / "Scripts" / "python.exe"
    monkeypatch.setattr(settings_module, "WIN_WORK", str(work))
    monkeypatch.setattr(settings_module, "WIN_PYTHON", str(python))
    s = fresh(True)
    assert s.get("paths.work") == str(tmp_path / "data" / "work") and s.get("env.python") == "", "nothing there yet"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    assert s.get("paths.work") == str(work) and s.get("env.python") == str(python)
    assert fresh(False).get("paths.work") == str(tmp_path / "data" / "work"), "the Windows folders are for Windows only"
    s.set({"paths.work": str(tmp_path / "elsewhere")})
    assert s.get("paths.work") == str(tmp_path / "elsewhere"), "a saved setting beats the default"


def test_the_trainer_python_can_come_from_the_environment(fresh):
    assert fresh(False, {"PYGMALION_ENV_PYTHON": " /opt/trainer/bin/python "}).get("env.python") == "/opt/trainer/bin/python"
    assert fresh(False, {"PYGMALION_GALTON_TOKEN_FILE": "/x/mcp-token"}).get("galton.token_file") == "/x/mcp-token"


def test_the_llama_folder_default_is_the_windows_one_on_windows_only(tmp_path, monkeypatch, fresh):
    monkeypatch.setattr(settings_module, "WIN_LLAMA", str(tmp_path / "llama.cpp"))
    assert fresh(True).get("llama.bin_dir") == str(tmp_path / "llama.cpp") and fresh(False).get("llama.bin_dir") == ""


def test_llama_programs_are_found_in_the_folder_then_on_the_path(tmp_path, monkeypatch, fresh):
    s = fresh(False)
    w = Work(s, tmp_path / "workers")
    assert w.llama_bin("llama-quantize") is None
    monkeypatch.setattr(procs, "which", lambda name: str(tmp_path / "pathbin" / name) if name == "llama-quantize" else None)
    assert w.llama_bin("llama-quantize") == str(tmp_path / "pathbin" / "llama-quantize")
    folder = tmp_path / "bin"
    folder.mkdir()
    (folder / f"llama-quantize{EXE}").write_bytes(b"")
    s.set({"llama.bin_dir": str(folder)})
    assert w.llama_bin("llama-quantize") == str(folder / f"llama-quantize{EXE}"), "the configured folder comes first"
    assert w.llama_bin("llama-imatrix") is None


def test_ollama_is_found_where_it_installs_itself(tmp_path, monkeypatch, fresh):
    s = fresh(False)
    w = Work(s, tmp_path / "workers")
    assert w.ollama_exe() is None
    on_path = tmp_path / "onpath" / "ollama"
    on_path.parent.mkdir()
    on_path.write_bytes(b"")
    monkeypatch.setattr(procs, "which", lambda name: str(on_path) if name == "ollama" else None)
    assert w.ollama_exe() == str(on_path)
    monkeypatch.setattr(procs, "which", lambda name: None)
    system = tmp_path / "usr-local" / "ollama"
    system.parent.mkdir()
    system.write_bytes(b"")
    monkeypatch.setattr(workdir, "OLLAMA_SYSTEM_PATHS", (str(tmp_path / "nowhere" / "ollama"), str(system)))
    monkeypatch.setattr(workdir, "IS_WIN", False)
    assert w.ollama_exe() == str(system), "the Linux and macOS install folders"
    configured = tmp_path / "mine" / "ollama"
    configured.parent.mkdir()
    configured.write_bytes(b"")
    s.set({"publish.ollama_exe": str(configured)})
    assert w.ollama_exe() == str(configured), "what the user configured comes first"


def test_ollama_is_found_in_the_windows_install_folder(tmp_path, monkeypatch, fresh):
    local = tmp_path / "Local"
    exe = local / "Programs" / "Ollama" / "ollama.exe"
    monkeypatch.setenv("LOCALAPPDATA", str(local))
    monkeypatch.setattr(workdir, "IS_WIN", True)
    w = Work(fresh(True), tmp_path / "workers")
    assert w.ollama_exe() is None
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"")
    assert w.ollama_exe() == str(exe)
