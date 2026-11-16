"""Command-line agents as contestants: any program that can be told the brief and prints its answer.

They are written by the person in the setting ``briefbench.cli``, one line each: ``name = command and arguments``; a line that starts with ``#`` is switched off. Nothing is
configured by default. The command runs without a shell in a scratch folder of the run, with these placeholders replaced in its arguments and exported as environment variables:

    {prompt}       the brief text                                  BRIEFBENCH_PROMPT
    {prompt_file}  a text file that holds the same text            BRIEFBENCH_PROMPT_FILE
    {workdir}      the scratch folder                              BRIEFBENCH_WORKDIR

What it prints is the answer; for a page brief, an ``index.html`` it leaves in the folder is the answer when its output holds no HTML document. The agent counts its own tokens
and bills its own session, so the bench cannot see either: tokens and cost of these answers are unknown (None), never zero. Its time is the wall time of the command.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Callable, Optional

from ..errors import GaltonError

ID_PREFIX = "cli:"
_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def parse(lines: list[str]) -> dict[str, list[str]]:
    """``{name: argv}`` of the active lines of the setting. A line that is not ``name = command`` is ignored."""
    out: dict[str, list[str]] = {}
    for raw in lines or []:
        line = str(raw).strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, command = line.partition("=")
        name = name.strip()
        try:
            argv = _split(command.strip())
        except ValueError:
            continue
        if name and argv:
            out[name] = argv
    return out


def _split(command: str) -> list[str]:
    if os.name == "nt":
        return [p[1:-1] if len(p) > 1 and p[0] == p[-1] and p[0] in "\"'" else p for p in shlex.split(command, posix=False)]
    return shlex.split(command)


def is_cli(ref: Any) -> bool:
    return isinstance(ref, str) and ref.startswith(ID_PREFIX)


def lookup(lines: list[str], ref: str) -> tuple[str, list[str]]:
    name = ref[len(ID_PREFIX):].strip()
    table = parse(lines)
    if name not in table:
        raise GaltonError("not_found", "bk_no_cli", name=name)
    return name, table[name]


def run(argv: list[str], prompt: str, workdir: Path, *, timeout_s: float, cancel: Callable[[], bool]) -> dict[str, Any]:
    """Run one command for one brief. Returns ``{"text", "wall_ms", "error", "exit"}``; ``error`` is empty when it ended by itself with exit code 0."""
    workdir.mkdir(parents=True, exist_ok=True)
    prompt_file = workdir / "brief.txt"
    prompt_file.write_text(prompt, encoding="utf-8")
    subs = {"{prompt}": prompt, "{prompt_file}": str(prompt_file), "{workdir}": str(workdir)}
    command = []
    for a in argv:
        for k, v in subs.items():
            a = a.replace(k, v)
        command.append(a)
    env = {**os.environ, "BRIEFBENCH_PROMPT": prompt, "BRIEFBENCH_PROMPT_FILE": str(prompt_file), "BRIEFBENCH_WORKDIR": str(workdir)}
    started = time.monotonic()
    try:
        proc = subprocess.Popen(command, cwd=str(workdir), env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                                errors="replace", creationflags=_NO_WINDOW)
    except OSError as exc:
        return {"text": "", "wall_ms": 0.0, "error": f"{type(exc).__name__}: {exc}"[:300], "exit": None}
    out = err = ""
    error = ""
    while True:
        try:
            out, err = proc.communicate(timeout=0.5)
            break
        except subprocess.TimeoutExpired:
            if cancel():
                proc.kill()
                out, err = proc.communicate()
                error = "cancelled"
                break
            if time.monotonic() - started > timeout_s:
                proc.kill()
                out, err = proc.communicate()
                error = f"no answer within {int(timeout_s)} s"
                break
    wall = (time.monotonic() - started) * 1000
    if not error and proc.returncode not in (0, None):
        error = ((err or "").strip()[-300:] or f"exit code {proc.returncode}")
    return {"text": out or "", "wall_ms": wall, "error": error, "exit": proc.returncode}


def collected(workdir: Path, text: str, kind: str) -> str:
    """The reply: the output of the command, or the page it left in its folder when the output holds no document."""
    if kind == "page" and "<html" not in text.lower() and "<!doctype" not in text.lower():
        page: Optional[Path] = workdir / "index.html"
        if page.is_file():
            return "```html\n" + page.read_text(encoding="utf-8", errors="replace") + "\n```"
    return text
