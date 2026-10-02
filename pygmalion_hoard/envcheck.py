"""Is this computer ready? Runs the probe in the trainer environment and checks the llama.cpp programs, the conversion scripts and
Ollama. Reports exactly what is missing and the command that fixes it."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

from .messages import ERRORS, TEXTS, error_text, hint_text, text
from .procs import build_env, run_capture
from .workdir import CONVERT_SCRIPTS, LLAMA_BINARIES, Work

TORCH_INDEX = "https://download.pytorch.org/whl/cu128"
PIP_LIBS = "transformers peft trl accelerate datasets bitsandbytes safetensors huggingface_hub gguf"


def fix_commands(python: str) -> dict[str, str]:
    """The commands that repair each kind of gap. ``python`` is the trainer's interpreter (or the one to create)."""
    py = python or "python"
    return {
        "create_env": "python -m venv <work>/venv",
        "torch": f'"{py}" -m pip install torch --index-url {TORCH_INDEX}',
        "libs": f'"{py}" -m pip install -U {PIP_LIBS}',
        "llama_src": "git clone --depth 1 https://github.com/ggml-org/llama.cpp <work>/llama.cpp-src",
        "llama_src_deps": f'"{py}" -m pip install -r <work>/llama.cpp-src/requirements/requirements-convert_hf_to_gguf.txt',
        "llama_bin": text("env_fix_llama_bin"),
        "ollama": text("env_fix_ollama"),
    }


def probe_environment() -> dict[str, str]:
    """The environment of the probe: PCI bus numbering and every GPU visible. The lease (``CUDA_VISIBLE_DEVICES``) governs jobs only; a
    probe that hid the GPUs would report none, and so would one that inherited a restriction from whoever started the app."""
    env = build_env()
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


class EnvChecker:
    def __init__(self, work: Work, *, capture: Callable[..., tuple[Optional[int], str, str]] = run_capture, clock: Callable[[], float] = time.time):
        self.work = work
        self.capture = capture
        self.clock = clock
        self._last: Optional[dict[str, Any]] = None

    def probe(self, model: Optional[str] = None, timeout_s: float = 120.0) -> dict[str, Any]:
        """Run ``workers/probe_env.py`` in the trainer environment. ``{ok: False, error}`` when it cannot run."""
        python = self.work.python()
        if not python:
            return self.failure("env_python_unset")
        if not Path(python).is_file():
            return self.failure("env_python_missing", python=python)
        args = {"work_dir": str(self.work.root)}
        if model:
            args["model"] = model
        args_path = self.work.root / "probe-args.json"
        try:
            args_path.parent.mkdir(parents=True, exist_ok=True)
            args_path.write_text(json.dumps(args), encoding="utf-8")
        except OSError as exc:
            return self.failure("work_unwritable", detail=exc)
        code, out, err = self.capture([python, str(self.work.worker("probe_env")), "--args", str(args_path)], env=probe_environment(), timeout_s=timeout_s)
        data = None
        for line in out.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if isinstance(event, dict) and event.get("event") == "result":
                data = event.get("data")
            elif isinstance(event, dict) and event.get("event") == "error":
                if event.get("key") in ERRORS:
                    return self.failure(event["key"], **(event.get("params") if isinstance(event.get("params"), dict) else {}))
                return {"ok": False, "error": event.get("msg", "probe failed"), "hint": event.get("hint", "")}
        if data is None:
            detail = (err or out).strip()[-400:]
            return self.failure("probe_failed", detail=detail) if detail else self.failure("probe_silent")
        return {"ok": True, **data}

    @staticmethod
    def failure(key: str, **params: Any) -> dict[str, Any]:
        """``{ok: False, error, hint}`` for the catalogue error ``key``: both are coded, so the UI words them in its language."""
        return {"ok": False, "error": error_text(key, **params), "hint": hint_text(key, **params)}

    def check(self, model: Optional[str] = None) -> dict[str, Any]:
        python = self.work.python()
        probe = self.probe(model)
        fixes = fix_commands(python)
        problems: list[dict[str, str]] = []
        if not probe.get("ok"):
            problems.append({"what": text("env_trainer"), "problem": probe.get("error") or text("env_unavailable"), "fix": fixes["create_env"] if not python else fixes["libs"]})
        else:
            for item in probe.get("problems", []):
                lib = item["lib"]
                fix = fixes["torch"] if lib == "torch" else fixes["libs"]
                problems.append({"what": lib, "problem": text(item["key"], **(item.get("params") or {})) if item.get("key") in TEXTS else item["problem"], "fix": fix})
        binaries = {name: self.work.llama_bin(name) for name in LLAMA_BINARIES}
        for name, found in binaries.items():
            if not found:
                problems.append({"what": name, "problem": text("env_not_found"), "fix": fixes["llama_bin"]})
        scripts = {name: self.work.src_script(name) for name in CONVERT_SCRIPTS}
        for name, found in scripts.items():
            if not found:
                problems.append({"what": name, "problem": text("env_not_in_src"), "fix": fixes["llama_src"]})
        ollama = self.work.ollama_exe()
        if not ollama:
            problems.append({"what": "ollama", "problem": text("env_ollama_missing"), "fix": fixes["ollama"], "optional": True})
        required = [p for p in problems if not p.get("optional")]
        result = {"ok": not required, "checked_ts": self.clock(), "trainer": probe, "python": python, "work_dir": str(self.work.root),
                  "llama_binaries": binaries, "convert_scripts": scripts, "ollama": ollama, "problems": problems, "fixes": fixes,
                  "capabilities": self.capabilities(probe, binaries, scripts, ollama)}
        self._last = result
        return result

    @staticmethod
    def capabilities(probe: dict[str, Any], binaries: dict[str, Optional[str]], scripts: dict[str, Optional[str]], ollama: Optional[str]) -> dict[str, bool]:
        torch = probe.get("torch") or {}
        gpu = bool(torch.get("cuda_available"))
        libs = probe.get("libs") or {}
        train_ready = bool(probe.get("ok") and gpu and libs.get("peft") and libs.get("transformers"))
        return {
            "download": bool(probe.get("ok") and libs.get("huggingface_hub")),
            "train": train_ready,
            "qlora": bool(train_ready and (probe.get("bitsandbytes") or {}).get("ok")),
            "merge": bool(probe.get("ok")),
            "convert": bool(probe.get("ok") and scripts.get("convert_hf_to_gguf.py")),
            "quantize": bool(binaries.get("llama-quantize")),
            "imatrix": bool(binaries.get("llama-imatrix")),
            "perplexity": bool(binaries.get("llama-perplexity")),
            "publish_ollama": bool(ollama),
            "publish_llama": bool(binaries.get("llama-server")),
        }

    def last(self) -> Optional[dict[str, Any]]:
        return self._last

    def trainable_architectures(self) -> Optional[set[str]]:
        """Architectures the trainer's transformers can load, from the last probe (None until a probe has run)."""
        names = ((self._last or {}).get("trainer") or {}).get("trainable_architectures")
        return set(names) if names else None
