"""Where things live on disk and which external programs are configured: the work folder, the trainer's Python, the llama.cpp
binaries and conversion scripts, Ollama. Nothing here runs a program; it only resolves paths from the settings."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

from . import procs
from .settings import Settings
from .util import repo_dirname

IS_WIN = sys.platform.startswith("win")
EXE = ".exe" if IS_WIN else ""
PACKAGE_WORKERS = Path(__file__).resolve().parent / "workers"
LLAMA_BINARIES = ("llama-quantize", "llama-imatrix", "llama-perplexity", "llama-server")
CONVERT_SCRIPTS = ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py")
OLLAMA_SYSTEM_PATHS = ("/usr/local/bin/ollama", "/usr/bin/ollama")   # where Linux and macOS installs put it


class Work:
    def __init__(self, settings: Settings, workers_dir: Path):
        self.settings = settings
        self.workers_dir = Path(workers_dir)

    # -- folders ------------------------------------------------------------------------------------------------
    @property
    def root(self) -> Path:
        return Path(self.settings.get("paths.work"))

    @property
    def hf_dir(self) -> Path:
        return self.root / "hf"

    @property
    def outputs_dir(self) -> Path:
        return self.root / "outputs"

    @property
    def jobs_dir(self) -> Path:
        return self.root / "jobs"

    @property
    def calib_dir(self) -> Path:
        return self.root / "calib"

    def ensure(self) -> None:
        for folder in (self.root, self.hf_dir, self.outputs_dir, self.jobs_dir, self.calib_dir):
            folder.mkdir(parents=True, exist_ok=True)

    def base_dir(self, repo_id: str) -> Path:
        return self.hf_dir / repo_dirname(repo_id)

    def output_dir(self, artifact_id: str) -> Path:
        return self.outputs_dir / artifact_id

    def job_dir(self, job_id: str) -> Path:
        return self.jobs_dir / job_id

    # -- programs ------------------------------------------------------------------------------------------------
    def python(self) -> str:
        return self.settings.get("env.python").strip()

    def python_ok(self) -> bool:
        python = self.python()
        return bool(python) and Path(python).is_file()

    def worker(self, name: str) -> Path:
        """``workers_dir/<name>.py``; a replacement folder (tests) may hold only some workers, the package's own cover the rest."""
        candidate = self.workers_dir / f"{name}.py"
        return candidate if candidate.is_file() else PACKAGE_WORKERS / f"{name}.py"

    def llama_bin(self, name: str) -> Optional[str]:
        folder = self.settings.get("llama.bin_dir").strip()
        if folder:
            candidate = Path(folder) / f"{name}{EXE}"
            if candidate.is_file():
                return str(candidate)
        return procs.which(name)

    def src_script(self, name: str) -> Optional[str]:
        folder = self.settings.get("llama.src_dir").strip()
        if folder:
            candidate = Path(folder) / name
            if candidate.is_file():
                return str(candidate)
        return None

    def ollama_exe(self) -> Optional[str]:
        configured = self.settings.get("publish.ollama_exe").strip()
        candidates = [configured, procs.which("ollama")]
        if IS_WIN and os.environ.get("LOCALAPPDATA"):
            candidates.append(str(Path(os.environ["LOCALAPPDATA"]) / "Programs" / "Ollama" / "ollama.exe"))
        else:
            candidates += list(OLLAMA_SYSTEM_PATHS)
        for candidate in candidates:
            if candidate and Path(candidate).is_file():
                return str(candidate)
        return None
