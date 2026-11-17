"""The app's side of the browser worker: start it with a Python that has Playwright, send it commands, read its answers, stop it.

The app itself never imports Playwright. A command that gets no answer within its time stops the worker; the next command starts a new one.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Any, Optional

from ..errors import GaltonError

WORKER = Path(__file__).resolve().parent / "browser_worker.py"
FIDELITY = Path(__file__).resolve().parents[1] / "hoard_link" / "webfidelity.py"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


class Browser:
    def __init__(self, python: str, *, start_timeout_s: float = 60.0):
        if not python:
            raise GaltonError("unavailable", "bk_no_browser")
        self.python, self.start_timeout_s = python, start_timeout_s
        self.proc: Optional[subprocess.Popen] = None
        self.lines: "queue.Queue[Optional[str]]" = queue.Queue()
        self.reader: Optional[threading.Thread] = None

    def _start(self) -> None:
        try:
            self.proc = subprocess.Popen([self.python, str(WORKER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                         text=True, encoding="utf-8", creationflags=_NO_WINDOW)
        except OSError as exc:
            raise GaltonError("unavailable", "bk_browser_start", python=self.python, error=str(exc)) from exc
        self.lines = queue.Queue()
        proc, lines = self.proc, self.lines

        def read() -> None:
            for line in proc.stdout:          # type: ignore[union-attr]
                lines.put(line)
            lines.put(None)
        self.reader = threading.Thread(target=read, name="webrebuild-browser-reader", daemon=True)
        self.reader.start()
        self._send({"fidelity": str(FIDELITY)})
        ready = self._read(self.start_timeout_s)
        if not ready.get("ok"):
            self.close()
            raise GaltonError("unavailable", "wr_browser_failed", error=str(ready.get("error") or "")[:300])

    def _send(self, obj: dict[str, Any]) -> None:
        assert self.proc is not None and self.proc.stdin is not None
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def _read(self, timeout_s: float) -> dict[str, Any]:
        while True:
            try:
                line = self.lines.get(timeout=timeout_s)
            except queue.Empty:
                self.close()
                return {"ok": False, "error": f"the browser did not answer within {int(timeout_s)} s"}
            if line is None:
                self.close()
                return {"ok": False, "error": "the browser stopped"}
            if line.strip().startswith("{"):
                try:
                    return json.loads(line)
                except ValueError:
                    continue

    def call(self, cmd: dict[str, Any], timeout_s: float = 60.0) -> dict[str, Any]:
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        try:
            self._send(cmd)
        except (OSError, ValueError) as exc:
            self.close()
            return {"ok": False, "error": f"the browser stopped: {exc}"}
        return self._read(timeout_s)

    def close(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin is not None:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(10)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        if proc.poll() is None:
            proc.kill()
            proc.wait(5)
        if self.reader is not None:
            self.reader.join(5)
            self.reader = None

    def __enter__(self) -> "Browser":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
