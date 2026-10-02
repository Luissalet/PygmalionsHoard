"""The settings registry: every key the UI and the assistants may change, with its type, default and limits."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from .errors import PygmalionError

IS_WIN = sys.platform.startswith("win")
WIN_WORK = r"D:\LocalAI\pygmalion"
WIN_PYTHON = WIN_WORK + r"\venv\Scripts\python.exe"
WIN_LLAMA = r"D:\LocalAI\llama.cpp"
RESERVED_DEFAULT = (0, 1)
ALLOWED_DEFAULT = (2, 3)


@dataclass(frozen=True)
class Spec:
    kind: str                       # str | int | float | bool | choice | intlist | path
    default: str = ""
    choices: tuple[str, ...] = ()
    low: Optional[float] = None
    high: Optional[float] = None
    group: str = "general"
    doc: str = ""


def _s(kind: str, default: Any = "", group: str = "general", doc: str = "", choices: tuple[str, ...] = (),
       low: Optional[float] = None, high: Optional[float] = None) -> Spec:
    return Spec(kind, str(default), choices, low, high, group, doc)


SPECS: dict[str, Spec] = {
    "ui.language": _s("choice", "es", choices=("es", "en")),
    "scheduler.paused": _s("bool", 0, "jobs", "Stop starting queued jobs."),
    "gpus.allowed": _s("intlist", "2,3", "gpus", "GPU indices the apps may use. Anything else is never touched."),
    "gpus.reserved": _s("intlist", "0,1", "gpus", "GPUs that belong to the owner of the computer. Allowing one needs a confirmation."),
    "env.python": _s("path", "", "paths", "Python of the trainer environment (torch, transformers, peft, bitsandbytes, trl)."),
    "paths.work": _s("path", "", "paths", "Work folder: hf/, outputs/, jobs/, calib/."),
    "llama.bin_dir": _s("path", "", "paths", "Folder with llama-quantize, llama-imatrix, llama-perplexity and llama-server (empty: the PATH; on Windows D:\\LocalAI\\llama.cpp first)."),
    "llama.src_dir": _s("path", "", "paths", "llama.cpp sources: convert_hf_to_gguf.py and convert_lora_to_gguf.py."),
    "merge.device": _s("choice", "cpu", "paths", "Where adapters are merged into a base model.", ("cpu", "cuda")),
    "train.method": _s("choice", "qlora", "defaults", choices=("qlora", "lora")),
    "train.rank": _s("int", 16, "defaults", low=1, high=512),
    "train.alpha": _s("int", 32, "defaults", low=1, high=1024),
    "train.dropout": _s("float", 0.05, "defaults", low=0, high=0.9),
    "train.lr": _s("float", 0.0002, "defaults", low=1e-7, high=1.0),
    "train.seq_len": _s("int", 2048, "defaults", low=64, high=262144),
    "train.batch": _s("int", 1, "defaults", low=1, high=256),
    "train.grad_accum": _s("int", 16, "defaults", low=1, high=4096),
    "train.epochs": _s("float", 1, "defaults", low=0.01, high=100),
    "train.max_steps": _s("int", 0, "defaults", "0 means: run the epochs.", low=0, high=10_000_000),
    "train.warmup": _s("float", 0.03, "defaults", "Fraction of the steps spent warming up.", low=0, high=0.5),
    "train.weight_decay": _s("float", 0.0, "defaults", low=0, high=1),
    "train.eval_every": _s("int", 50, "defaults", low=0, high=1_000_000),
    "train.save_every": _s("int", 100, "defaults", low=0, high=1_000_000),
    "train.log_every": _s("int", 5, "defaults", low=1, high=10_000),
    "train.seed": _s("int", 42, "defaults", low=0, high=2**31 - 1),
    "train.eval_split": _s("float", 5, "defaults", "Percentage of the records kept for evaluation.", low=0, high=50),
    "train.eval_min": _s("int", 20, "defaults", "Minimum number of evaluation records.", low=0, high=100000),
    "dataset.chunk_chars": _s("int", 1500, "defaults", "Length of the pieces text files are cut into.", low=200, high=20000),
    "dataset.near_dup": _s("float", 0.85, "defaults", low=0.3, high=1),
    "quant.imatrix_chunks": _s("int", 100, "defaults", low=1, high=100000),
    "quant.ppl_chunks": _s("int", 20, "defaults", low=1, high=100000),
    "quant.ppl_ctx": _s("int", 2048, "defaults", low=128, high=262144),
    "teacher.model": _s("str", "", "teacher", "Model name to prefer for synthetic data (empty: whichever the family resolves)."),
    "teacher.max_items": _s("int", 200, "teacher", low=1, high=100000),
    "teacher.effort": _s("choice", "off", "teacher", choices=("off", "low", "medium", "high")),
    "galton.url": _s("str", "http://127.0.0.1:5201", "galton", "Where Galton's Hoard listens (direct fallback)."),
    "galton.token_file": _s("path", "", "galton", "Galton's mcp-token (direct fallback when the family hub is not available)."),
    "galton.timeout_s": _s("int", 7200, "galton", "How long an evaluation may run before the job gives up.", low=30, high=604800),
    "galton.eval_cases": _s("int", 100, "galton", "Most held-out records turned into cases of the dataset suite (the judge grades each answer).", low=5, high=2000),
    "publish.num_ctx": _s("int", 8192, "publish", "Context (num_ctx) written into an Ollama Modelfile unless the publish asks for another; never above the trained context.", low=256, high=2_000_000),
    "publish.llama_port_start": _s("int", 8100, "publish", "First port tried for published llama.cpp servers.", low=1024, high=65000),
    "publish.ollama_exe": _s("path", "", "publish", "Ollama executable (empty: PATH or the default install folder)."),
    "jobs.stop_grace_s": _s("int", 30, "jobs", "Seconds a worker gets to save a checkpoint after a stop request.", low=1, high=600),
}

SECRET_KEYS = ("hf.token",)


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def parse_intlist(raw: Any) -> list[int]:
    if isinstance(raw, (list, tuple)):
        parts = [str(x) for x in raw]
    else:
        parts = str(raw or "").replace(";", ",").replace(" ", ",").split(",")
    out: list[int] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if not part.isdigit():
            raise PygmalionError("invalid", "not_gpu_index", part=part)
        if int(part) not in out:
            out.append(int(part))
    return out


def normalize(key: str, value: Any) -> str:
    """Validate ``value`` for ``key`` and return its stored text. Raises PygmalionError."""
    spec = SPECS.get(key)
    if spec is None:
        raise PygmalionError("invalid", "setting_unknown", key=key, options=list(SPECS))
    if isinstance(value, bool):
        value = "1" if value else "0"
    text = ("" if value is None else (",".join(str(x) for x in value) if isinstance(value, (list, tuple)) else str(value))).strip()
    if spec.kind == "choice" and text not in spec.choices:
        raise PygmalionError("invalid", "setting_choice", key=key, options=list(spec.choices))
    if spec.kind == "bool":
        if text.lower() not in ("0", "1", "true", "false", "yes", "no", "on", "off"):
            raise PygmalionError("invalid", "setting_bool", key=key)
        return "1" if _truthy(text) else "0"
    if spec.kind in ("int", "float"):
        try:
            number = float(text)
        except ValueError as exc:
            raise PygmalionError("invalid", "setting_number", key=key) from exc
        if spec.low is not None and number < spec.low or spec.high is not None and number > spec.high:
            raise PygmalionError("invalid", "setting_range", key=key, low=spec.low, high=spec.high)
        return str(int(number)) if spec.kind == "int" else repr(number)
    if spec.kind == "intlist":
        return ",".join(str(i) for i in parse_intlist(text))
    return text


class Settings:
    """Reads and writes settings in the database, applying defaults that depend on the machine."""

    def __init__(self, db: Any, data_dir: Path, env: Optional[dict[str, str]] = None, platform_is_windows: bool = IS_WIN):
        self.db = db
        self.data_dir = Path(data_dir)
        self.env = os.environ if env is None else env
        self.windows = platform_is_windows

    # -- defaults that depend on the machine --------------------------------------------------------------------
    def default_work(self) -> str:
        if self.windows and Path(WIN_WORK).is_dir():
            return WIN_WORK
        return str(self.data_dir / "work")

    def default_python(self) -> str:
        if self.windows and Path(WIN_PYTHON).is_file():
            return WIN_PYTHON
        return (self.env.get("PYGMALION_ENV_PYTHON") or "").strip()

    def default_llama_bin(self) -> str:
        return WIN_LLAMA if self.windows else ""

    def default_src(self) -> str:
        return str(Path(self.get("paths.work")) / "llama.cpp-src")

    def default_galton_token(self) -> str:
        return (self.env.get("PYGMALION_GALTON_TOKEN_FILE") or "").strip()

    def get(self, key: str) -> str:
        value = self.db.get_setting(key, None)
        if value not in (None, ""):
            return str(value)
        if key == "paths.work":
            return self.default_work()
        if key == "env.python":
            return self.default_python()
        if key == "llama.bin_dir":
            return self.default_llama_bin()
        if key == "llama.src_dir":
            return self.default_src()
        if key == "galton.token_file":
            return self.default_galton_token()
        spec = SPECS.get(key)
        return spec.default if spec else ""

    def int(self, key: str) -> int:
        return int(float(self.get(key) or 0))

    def float(self, key: str) -> float:
        return float(self.get(key) or 0)

    def bool(self, key: str) -> bool:
        return _truthy(self.get(key))

    def intlist(self, key: str) -> list[int]:
        return parse_intlist(self.get(key))

    def all(self) -> dict[str, str]:
        return {key: self.get(key) for key in SPECS}

    def set(self, values: dict[str, Any], confirm_reserved: bool = False) -> dict[str, str]:
        cleaned = {key: normalize(key, value) for key, value in values.items()}
        if "gpus.allowed" in cleaned:
            reserved = set(parse_intlist(cleaned.get("gpus.reserved", self.get("gpus.reserved"))))
            asked = set(parse_intlist(cleaned["gpus.allowed"]))
            touching = sorted(asked & reserved)
            if touching and not confirm_reserved:
                raise PygmalionError("confirm_required", "gpu_reserved", gpus=touching, reserved=touching)
            if not asked:
                raise PygmalionError("invalid", "no_gpu_allowed_setting")
        for key, value in cleaned.items():
            self.db.set_setting(key, value)
        return self.all()

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for key, spec in SPECS.items():
            out.setdefault(spec.group, []).append(key)
        return out

    def train_defaults(self) -> dict[str, Any]:
        return {"method": self.get("train.method"), "rank": self.int("train.rank"), "alpha": self.int("train.alpha"),
                "dropout": self.float("train.dropout"), "lr": self.float("train.lr"), "seq_len": self.int("train.seq_len"),
                "batch": self.int("train.batch"), "grad_accum": self.int("train.grad_accum"), "epochs": self.float("train.epochs"),
                "max_steps": self.int("train.max_steps"), "warmup": self.float("train.warmup"),
                "weight_decay": self.float("train.weight_decay"), "eval_every": self.int("train.eval_every"),
                "save_every": self.int("train.save_every"), "log_every": self.int("train.log_every"), "seed": self.int("train.seed")}
