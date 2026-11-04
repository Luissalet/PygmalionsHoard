"""Speech engines the bench can run, and how each one is run.

* ``faster-whisper:<size>`` (tiny, base, small, medium, large-v3, large-v3-turbo): ``whisper_worker.py`` in a Python that has
  faster-whisper (setting ``speechbench.whisper_python``, GALTON_WHISPER_PYTHON, this app's own Python, then the venvs of the
  family's apps as the hub lists them), on the CPU with int8 unless asked otherwise, one warm model for the whole corpus.
* ``cmd:<name>``: a command saved with ``speechbench_engine_save``; ``{audio}`` and ``{language}`` are replaced, it runs once
  per file and prints plain text or JSON ``{"text", "first_text_s"?}``. Load time is inside each file's time."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..hoard_link import proc

WHISPER_SIZES = ("tiny", "base", "small", "medium", "large-v3", "large-v3-turbo")
WORKER = Path(__file__).with_name("whisper_worker.py")
NAME = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,39}$")
_HAS_WHISPER: dict[str, bool] = {}


def has_faster_whisper(python: str) -> bool:
    if python not in _HAS_WHISPER:
        try:
            out = proc.run([python, "-I", "-c", "import importlib.util as u; print(bool(u.find_spec('faster_whisper')))"], timeout=30)
            _HAS_WHISPER[python] = out.returncode == 0 and out.stdout.strip() == "True"
        except (OSError, subprocess.TimeoutExpired):
            _HAS_WHISPER[python] = False
    return _HAS_WHISPER[python]


def family_pythons(hub_apps: Callable[[], list[dict[str, Any]]]) -> list[str]:
    """The venv Pythons of the family's apps (the hub's app list), first those that run one already."""
    found: list[str] = []
    try:
        apps = hub_apps() or []
    except Exception:  # noqa: BLE001 - hub down: nothing found there
        apps = []
    for app in apps:
        launch = (app.get("launch") or {}).get("executable") or ""
        if launch.lower().endswith("python.exe") or launch.endswith("/python"):
            found.append(str(Path(launch)))
        folder = app.get("folder")
        for venv in ("venv", ".venv"):
            exe = Path(folder or "", venv, "Scripts" if os.name == "nt" else "bin", "python.exe" if os.name == "nt" else "python")
            if folder and exe.is_file():
                found.append(str(exe))
    return list(dict.fromkeys(found))


def find_whisper_python(configured: str = "", hub_apps: Optional[Callable[[], list[dict[str, Any]]]] = None) -> Optional[str]:
    candidates = [configured, os.getenv("GALTON_WHISPER_PYTHON", ""), sys.executable]
    if hub_apps is not None:
        candidates += family_pythons(hub_apps)
    for exe in dict.fromkeys(c.strip().strip('"') for c in candidates if c and c.strip()):
        if Path(exe).is_file() and has_faster_whisper(exe):
            return exe
    return None


def parse_engine(spec: str, commands: dict[str, dict[str, Any]]) -> dict[str, Any]:
    spec = spec.strip()
    if spec.startswith("faster-whisper"):
        size = spec.split(":", 1)[1] if ":" in spec else "small"
        if size not in WHISPER_SIZES:
            raise ValueError(f"Unknown Whisper size {size!r}. Use: {', '.join(WHISPER_SIZES)}.")
        return {"id": f"faster-whisper:{size}", "kind": "faster-whisper", "size": size}
    if spec.startswith("cmd:"):
        name = spec[4:]
        if name not in commands:
            raise ValueError(f"No command engine named {name!r}; save it with speechbench_engine_save.")
        return {"id": spec, "kind": "command", **commands[name]}
    raise ValueError(f"Unknown engine {spec!r}: use faster-whisper:<size> or cmd:<name>.")


def run_whisper(python: str, engine: dict[str, Any], items: list[dict[str, Any]], *, language: Optional[str], device: str,
                compute: str, models_dir: str, on_event: Callable[[dict[str, Any]], None], cancel: Any) -> None:
    """One warm faster-whisper process for every file; each JSON line it prints goes to ``on_event``."""
    spec = {"size": engine["size"], "device": device, "compute": compute, "language": language, "models_dir": models_dir,
            "files": [{"id": i["id"], "audio": i["audio"]} for i in items]}
    child = proc.popen([python, "-I", str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       text=True, encoding="utf-8", errors="replace", env=proc.build_env(), low_priority=True)
    try:
        child.stdin.write(json.dumps(spec) + "\n")
        child.stdin.close()
        for line in child.stdout:
            if cancel is not None and cancel.is_set():
                proc.kill_tree(child)
                return
            line = line.strip()
            if line.startswith("{"):
                try:
                    on_event(json.loads(line))
                except ValueError:
                    continue
        code = child.wait(timeout=60)
        if code != 0:
            on_event({"event": "failed", "error": proc.tail_lines(child.stderr.read() if child.stderr else "")})
    finally:
        if child.poll() is None:
            proc.kill_tree(child)


def _command_argv(engine: dict[str, Any], audio: str, language: Optional[str]) -> list[str]:
    return [str(part).replace("{audio}", audio).replace("{language}", language or "") for part in engine["command"]]


def run_command(engine: dict[str, Any], item: dict[str, Any], *, language: Optional[str], cancel: Any = None) -> dict[str, Any]:
    """Run a command engine on one file: plain text, or JSON with ``text`` (and ``first_text_s`` if it streams)."""
    started = time.perf_counter()
    try:
        out = proc.run(_command_argv(engine, item["audio"], language), timeout=float(engine.get("timeout_s") or 600),
                       cwd=engine.get("cwd") or None, env=proc.build_env(), cancel=cancel)
    except subprocess.TimeoutExpired:
        return {"error": f"took longer than {engine.get('timeout_s') or 600} s", "decode_s": round(time.perf_counter() - started, 3)}
    except OSError as error:
        return {"error": f"could not start: {error}"}
    elapsed = round(time.perf_counter() - started, 3)
    if out.returncode != 0:
        return {"error": f"exit {out.returncode}: {proc.tail_lines(out.stderr or out.stdout, 4)}", "decode_s": elapsed}
    text, first = (out.stdout or "").strip(), None
    candidates = [text] + [ln for ln in reversed(text.splitlines()) if ln.strip().startswith("{")]
    for raw in candidates:
        try:
            data = json.loads(raw)
        except ValueError:
            continue
        if isinstance(data, dict) and "text" in data:
            text, first = str(data["text"]), data.get("first_text_s")
            break
    return {"text": text, "first_text_s": first, "decode_s": elapsed}


def save_command(commands: dict[str, dict[str, Any]], name: str, command: list[str], timeout_s: float = 600,
                 cwd: str = "", note: str = "") -> dict[str, Any]:
    if not NAME.match(name):
        raise ValueError("Name: lower case letters, digits, '.', '_' or '-', up to 40 characters.")
    if not command or not any("{audio}" in str(p) for p in command):
        raise ValueError("The command must contain {audio} where the audio file goes.")
    commands[name] = {"name": name, "command": [str(p) for p in command], "timeout_s": float(timeout_s), "cwd": cwd, "note": note}
    return commands[name]
