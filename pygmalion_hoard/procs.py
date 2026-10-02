"""Every external command goes through here: argv lists (never shell strings), a working folder, an environment with
``CUDA_VISIBLE_DEVICES``, a reader thread that writes each line to the job log, and a way to stop the whole process tree.

Starting, stopping and killing the process tree is the shared ``hoard_link.proc``: on Windows the child gets its own process group and no
console window and a stop request is a ``CTRL_BREAK_EVENT`` first and ``taskkill /T /F`` after the grace period; on Linux it is ``SIGTERM`` to the
process group, then ``SIGKILL``. A child started without a console never receives the Ctrl+Break, so the polite request is also a file: with
``stop_file`` the child is told its path (``PYGMALION_STOP_FILE``) and the file is created on a stop request; the workers watch
it and save a checkpoint before they exit.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

from .hoard_link import proc as hl_proc
from .hoard_link.proc import request_stop  # noqa: F401  (CTRL_BREAK_EVENT on Windows, SIGTERM to the group elsewhere: the child may save a checkpoint)
from .logclean import Collapser

TAIL_LINES = 60


@dataclass
class ProcResult:
    returncode: Optional[int]
    cancelled: bool = False
    timed_out: bool = False
    duration_s: float = 0.0
    pid: Optional[int] = None
    tail: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.cancelled and not self.timed_out


def which(name: str) -> Optional[str]:
    """A program on the PATH or in its usual install folders (the one place the app asks; tests replace it so the machine running them does not matter)."""
    return hl_proc.which(name)


STOP_FILE_ENV = "PYGMALION_STOP_FILE"
#: seconds ``run_streaming`` waits for the thread that reads the output once the program has ended (then it kills the program's group)
READER_JOIN_S = 5.0


def build_env(extra: Optional[dict[str, str]] = None, gpus: Optional[Sequence[int]] = None) -> dict[str, str]:
    """The parent environment plus ``extra``. ``gpus`` sets ``CUDA_VISIBLE_DEVICES`` (an empty list hides every GPU).

    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` always: CUDA numbers GPUs fastest-first by default, while nvidia-smi, the hub and the
    allowed list number them by PCI bus; without it ``CUDA_VISIBLE_DEVICES=2`` can be one of the owner's GPUs."""
    env = hl_proc.build_env()                       # a copy of the environment with UTF-8 forced for Python children
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.update({k: str(v) for k, v in (extra or {}).items()})
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if gpus is not None:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    return env


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop the process and everything it started, at once."""
    if proc.poll() is not None:
        # The program ended but something it started may still hold its output open: on POSIX that group can still be addressed (the shared
        # kill_tree treats an exited leader as "already gone" and leaves the group alone).
        if os.name != "nt":
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        return
    hl_proc.kill_tree(proc, grace_s=0)
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass


def stop_process(proc: subprocess.Popen, grace_s: float = 15.0, stop_file: Optional[Path] = None) -> None:
    """Graceful request (the stop file, then the signal), wait up to ``grace_s``, then kill the tree."""
    if proc.poll() is not None:
        return
    if stop_file is not None:
        try:
            Path(stop_file).write_text("stop\n", encoding="utf-8")
        except OSError:
            pass
    request_stop(proc)
    deadline = time.monotonic() + max(0.0, grace_s)
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(0.1)
    kill_tree(proc)
    try:
        proc.wait(10)
    except subprocess.TimeoutExpired:
        pass


def redact(line: str, secrets: Sequence[str]) -> str:
    for secret in secrets:
        if secret and len(secret) >= 6:
            line = line.replace(secret, "***")
    return line


def _write_log(log, line: str, replace: bool, last: dict) -> None:
    """Write ``line`` to the log; when it is the next frame of the previous line, rewrite that line instead of adding one. Nothing is rewritten
    when somebody else wrote after it (the file no longer ends where this reader left it)."""
    log.flush()
    size = os.fstat(log.fileno()).st_size
    if replace and last["start"] is not None and size == last["end"]:
        log.truncate(last["start"])
        log.seek(0, os.SEEK_END)
        size = last["start"]
    log.write(line + "\n")
    log.flush()
    last["start"], last["end"] = size, os.fstat(log.fileno()).st_size


def run_streaming(argv: Sequence[str], *, cwd: Optional[str | Path] = None, env: Optional[dict[str, str]] = None,
                  log_path: Optional[Path] = None, on_line: Optional[Callable[[str], None]] = None,
                  cancel: Optional[threading.Event] = None, timeout_s: Optional[float] = None, grace_s: float = 15.0,
                  secrets: Sequence[str] = (), on_start: Optional[Callable[[subprocess.Popen], None]] = None,
                  stop_file: Optional[Path] = None) -> ProcResult:
    """Run ``argv`` to the end, feeding every output line (stdout and stderr merged) to the log file and ``on_line``.

    Returns when the process exits, ``cancel`` is set (graceful stop, then kill) or ``timeout_s`` passes. Never raises for a
    failing command: look at ``result.returncode``. A command that cannot be started raises OSError. ``stop_file``: see the
    module docstring (a stale one from an earlier stop is removed first, or the child would stop at once)."""
    started = time.monotonic()
    if stop_file is not None:
        try:
            Path(stop_file).unlink()
        except OSError:
            pass
        env = dict(env or build_env())
        env[STOP_FILE_ENV] = str(stop_file)
    log = None
    if log_path is not None:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log = open(log_path, "a", encoding="utf-8", errors="replace", buffering=1)
        log.write(f"$ {' '.join(redact(str(a), secrets) for a in argv)}\n")
    proc = hl_proc.popen([str(a) for a in argv], cwd=str(cwd) if cwd else None, env=env or build_env(), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True, bufsize=1)
    if on_start:
        on_start(proc)
    tail: list[str] = []

    def reader() -> None:
        assert proc.stdout is not None
        collapser = Collapser()
        last = {"start": None, "end": None}              # where the last line written by this reader begins and ends, to redraw it in place
        for raw in proc.stdout:
            shown, replace = collapser.feed(raw.rstrip("\r\n"))
            if shown is None:                            # only escape codes: nothing to show
                continue
            line = redact(shown, secrets)
            if replace and tail:
                tail[-1] = line
            else:
                tail.append(line)
                del tail[:-TAIL_LINES]
            if log is not None:
                try:
                    _write_log(log, line, replace, last)
                except (OSError, ValueError):
                    pass
            if on_line is not None:
                try:
                    on_line(line)
                except Exception:  # noqa: BLE001 — a bad parser must not stop the reader
                    pass

    thread = threading.Thread(target=reader, name="pygmalion-proc-reader", daemon=True)
    thread.start()
    cancelled = timed_out = False
    try:
        while True:
            try:
                proc.wait(timeout=0.2)
                break
            except subprocess.TimeoutExpired:
                pass
            if cancel is not None and cancel.is_set():
                cancelled = True
                stop_process(proc, grace_s, stop_file)
                break
            if timeout_s is not None and time.monotonic() - started > timeout_s:
                timed_out = True
                stop_process(proc, min(grace_s, 5.0), stop_file)
                break
    finally:
        if proc.poll() is None:
            kill_tree(proc)
        thread.join(timeout=READER_JOIN_S)
        if thread.is_alive():                 # a grandchild still holds the pipe open: kill the group so the reader sees the end
            kill_tree(proc)
            thread.join(timeout=2.0)
        if log is not None:
            log.close()
    return ProcResult(proc.returncode, cancelled, timed_out, time.monotonic() - started, proc.pid, list(tail))


def run_capture(argv: Sequence[str], *, cwd: Optional[str | Path] = None, env: Optional[dict[str, str]] = None,
                timeout_s: float = 30.0) -> tuple[Optional[int], str, str]:
    """A short command whose output is wanted as text: ``(returncode, stdout, stderr)``; ``(None, "", reason)`` when it
    cannot be run or times out."""
    try:
        done = hl_proc.run([str(a) for a in argv], cwd=str(cwd) if cwd else None, env=env or build_env(), timeout=timeout_s)
    except subprocess.TimeoutExpired:
        return None, "", f"timed out after {timeout_s:g} s"
    except (OSError, ValueError) as exc:
        return None, "", str(exc)
    return done.returncode, done.stdout or "", done.stderr or ""
