"""What every worker shares: the line protocol, argument loading, stop signals and a few pure helpers. Standard library only.

Protocol: each line on stdout is one JSON object with an ``event``:

    {"event": "progress", "step": 10, "total": 200, "loss": 1.23, "lr": 0.0002, "tokens_per_s": 900, "eta_s": 300, "gpu_mem_mb": 11000}
    {"event": "log", "msg": "..."}
    {"event": "eval", "step": 50, "eval_loss": 1.1}
    {"event": "artifact", "path": "...", "kind": "adapter"}
    {"event": "result", "data": {...}}
    {"event": "done"}   or   {"event": "error", "msg": "...", "hint": "...", "key": "...", "params": {...}}

``key`` and ``params`` appear when the failure is an entry of the app's message catalogue (``messages.ERRORS``): the app words it again
and the UI translates it.

Anything else on stdout (library chatter) is ignored by the app and kept in the job log.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

STOP = threading.Event()
EXIT_STOPPED = 143


_MESSAGES: Any = None


def messages() -> Any:
    """The app's ``messages.py`` (standard library only), loaded by path because the worker runs under another interpreter; None if missing."""
    global _MESSAGES
    if _MESSAGES is None:
        try:
            path = Path(__file__).resolve().parent.parent / "messages.py"
            spec = importlib.util.spec_from_file_location("pygmalion_messages", str(path))
            module = importlib.util.module_from_spec(spec)           # type: ignore[arg-type]
            sys.modules["pygmalion_messages"] = module
            spec.loader.exec_module(module)                          # type: ignore[union-attr]
            _MESSAGES = module
        except Exception:  # noqa: BLE001 — the worker still reports its failures, in plain English
            _MESSAGES = False
    return _MESSAGES or None


class WorkerError(Exception):
    """A failure with a message for the person and a hint on how to fix it.

    ``WorkerError("config_unreadable", path=p)`` takes the message and hint of that entry of ``messages.ERRORS`` and keeps ``key`` and
    ``params`` so the app can word it in the language of the UI. ``WorkerError("free text", "hint")`` is for a text that cannot be known
    in advance; ``hint=`` overrides the catalogue's hint (a more specific one)."""

    def __init__(self, msg: str, hint: str = "", **params: Any):
        self.key = ""
        self.params: dict[str, Any] = {}
        module = messages()
        if module is not None and msg in module.ERRORS:
            template, hint_template = module.ERRORS[msg]
            used = module.fields_of(template) | module.fields_of(hint_template)
            self.key = msg
            self.params = {k: module.scalar(v) for k, v in params.items() if k in used}
            msg, hint = module.fill(template, self.params), hint or module.fill(hint_template, self.params)
        super().__init__(msg)
        self.msg = msg
        self.hint = hint


def emit(event: str, **fields: Any) -> None:
    sys.stdout.write(json.dumps({"event": event, **fields}, ensure_ascii=False, default=str) + "\n")
    sys.stdout.flush()


def log(msg: str) -> None:
    emit("log", msg=msg)


STOP_FILE_ENV = "PYGMALION_STOP_FILE"


def install_stop_handlers() -> None:
    """SIGTERM and Ctrl+C everywhere; Ctrl+Break on Windows (what the app sends to the worker's process group); and the stop file
    the app names in ``PYGMALION_STOP_FILE``, which is the request that actually arrives on Windows: a worker started without a
    console window never receives Ctrl+Break."""
    def handler(_signum: int, _frame: Any) -> None:
        STOP.set()

    for name in ("SIGTERM", "SIGINT", "SIGBREAK"):
        number = getattr(signal, name, None)
        if number is not None:
            try:
                signal.signal(number, handler)
            except (ValueError, OSError):
                pass
    path = os.environ.get(STOP_FILE_ENV, "").strip()
    if path:
        watch_stop_file(Path(path))


def watch_stop_file(path: Path, every_s: float = 0.5) -> threading.Thread:
    """Set ``STOP`` as soon as ``path`` exists (checked every half second in a daemon thread)."""
    def loop() -> None:
        while not STOP.is_set():
            if path.exists():
                STOP.set()
                return
            time.sleep(every_s)

    thread = threading.Thread(target=loop, name="stop-file", daemon=True)
    thread.start()
    return thread


def load_args(argv: Optional[Sequence[str]] = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", required=True, help="path to the job's args.json")
    ns = parser.parse_args(argv)
    try:
        return json.loads(Path(ns.args).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise WorkerError("args_unreadable", path=ns.args, detail=exc) from exc


HINTS = (
    (re.compile(r"out of memory|CUDA error: out of memory|OutOfMemoryError", re.I), "tip_oom"),
    (re.compile(r"No module named '?(\w+)", re.I), "tip_module"),
    (re.compile(r"gated repo|401 Client Error|403 Client Error|Access to model .* is restricted", re.I), "tip_gated"),
    (re.compile(r"No space left|disk full|not enough space", re.I), "tip_disk"),
    (re.compile(r"CUDA.*not available|no CUDA-capable|Torch not compiled with CUDA", re.I), "tip_cuda"),
    (re.compile(r"Unrecognized configuration class|does not recognize this architecture|model type .* not supported", re.I), "tip_arch"),
)


def hint_for(text: str) -> str:
    """What to do about a library's error text: the sentence of the matching ``tip_*`` entry of ``messages.TEXTS`` (empty if none matches)."""
    module = messages()
    for pattern, key in HINTS:
        match = pattern.search(text)
        if match and module is not None:
            return module.fill(module.TEXTS[key], {"module": match.group(1)} if match.groups() else {})
    return ""


def run_worker(main: Callable[[dict[str, Any]], None], argv: Optional[Sequence[str]] = None) -> int:
    """Run ``main(args)`` with stop handlers installed, turning any failure into an ``error`` event."""
    install_stop_handlers()
    try:
        main(load_args(argv))
    except WorkerError as exc:
        emit("error", msg=exc.msg, hint=exc.hint or hint_for(exc.msg), key=exc.key, params=exc.params)
        return 1
    except KeyboardInterrupt:
        emit("log", msg="Stopped.")
        return EXIT_STOPPED
    except BaseException as exc:  # noqa: BLE001 — the app needs the reason, not a traceback on a closed pipe
        text = f"{type(exc).__name__}: {exc}"
        emit("error", msg=text[:600], hint=hint_for(text), trace=traceback.format_exc()[-1500:])
        return 1
    if STOP.is_set():
        return EXIT_STOPPED
    emit("done")
    return 0


def import_by_path(name: str, path: Path) -> Any:
    """Import a pure module from the package folder without importing the package (the worker runs under another interpreter)."""
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise WorkerError("module_unloadable", path=path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def package_module(name: str) -> Any:
    """``chatmask`` or ``merge_math``: the file one folder above the workers."""
    return import_by_path(name, Path(__file__).resolve().parent.parent / f"{name}.py")


# ------------------------------------------------------------------ pure helpers used by the trainer and tested without torch
def lr_at(step: int, total: int, base_lr: float, warmup_fraction: float = 0.03) -> float:
    """Linear warm-up for ``warmup_fraction`` of the steps, then a cosine decay to zero. ``step`` counts optimizer steps from 0."""
    total = max(1, total)
    warm = max(1, int(round(total * warmup_fraction))) if warmup_fraction > 0 else 0
    if warm and step < warm:
        return base_lr * (step + 1) / warm
    progress = (step - warm) / max(1, total - warm)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))


def epoch_order(n: int, epoch: int, seed: int) -> list[int]:
    """The order in which an epoch visits ``n`` examples: depends only on the seed and the epoch, so a resumed run continues exactly."""
    import random
    order = list(range(n))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return order


VISION_WORDS = ("vision", "visual", "image", "audio", "speech", "multi_modal", "mm_projector", "projector", "patch_embed", "pixtral")
SKIP_NAMES = ("lm_head", "embed", "router", "norm")


def select_target_names(linear_names: Sequence[str], requested: Optional[Sequence[str]] = None) -> list[str]:
    """Which linear layers to adapt: all of the text model's projections by default (not the vision tower, the language-model
    head, embeddings, MoE routers or individual experts), or only those whose last name part is in ``requested``."""
    wanted = {r.strip() for r in (requested or []) if r and r.strip() and r.strip().lower() != "all"}
    out = []
    for name in linear_names:
        low = name.lower()
        last = name.rsplit(".", 1)[-1]
        if any(w in low for w in VISION_WORDS) or ".experts." in low or low.endswith(".gate") or last in ("gate",):
            continue
        if any(s in last.lower() for s in SKIP_NAMES):
            continue
        if wanted and last not in wanted:
            continue
        out.append(name)
    return out


def names_regex(names: Sequence[str]) -> str:
    """A regular expression PEFT accepts as ``target_modules``: it must match the whole module name, so no look-alike in another
    part of the model can be adapted by accident."""
    return r"^(?:" + "|".join(re.escape(n) for n in names) + r")$"


def checkpoint_steps(root: Path) -> list[int]:
    """Steps of the checkpoints under ``root``/checkpoint-<step>, oldest first."""
    steps = []
    if root.is_dir():
        for child in root.iterdir():
            m = re.fullmatch(r"checkpoint-(\d+)", child.name)
            if m and child.is_dir() and (child / "training_state.json").is_file():
                steps.append(int(m.group(1)))
    return sorted(steps)


def prune_checkpoints(root: Path, keep: int = 2) -> list[int]:
    import shutil
    steps = checkpoint_steps(root)
    removed = steps[:-keep] if keep > 0 else steps
    for step in removed:
        shutil.rmtree(root / f"checkpoint-{step}", ignore_errors=True)
    return removed


def eta_seconds(step: int, total: int, elapsed_s: float) -> int:
    return int(elapsed_s / step * (total - step)) if step > 0 else 0
