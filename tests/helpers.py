"""Shared test helpers: fake GPU inventory, a fake lease, and the fake external tools."""

from __future__ import annotations

import os
import stat
import sys
import threading
from pathlib import Path
from typing import Any, Optional

HERE = Path(__file__).resolve().parent
IS_WIN = sys.platform.startswith("win")
POSIX_ONLY = "the fake llama.cpp programs are POSIX scripts"
# The fake llama.cpp / Ollama programs are scripts with a shebang, so they run only where the OS honours it. Elsewhere (Windows)
# the toolchain holds inert placeholders: enough for everything that only looks programs up, and tests that run them are skipped.
# PYG_TEST_PLACEHOLDER_TOOLS=1 gives a POSIX machine the same toolchain, to check the Windows behaviour of the suite from Linux.
EXECUTABLE_TOOLS = not IS_WIN and os.environ.get("PYG_TEST_PLACEHOLDER_TOOLS") != "1"


def fake_inventory(*, free: Optional[dict[int, int]] = None) -> list[dict[str, Any]]:
    """The machine of the spec: GPU 0 (12 GB) and 1-3 (16 GB). ``free`` overrides the free MB of some GPUs."""
    cards = [(0, "NVIDIA GeForce RTX 4070 Ti", 12282), (1, "NVIDIA GeForce RTX 5060 Ti", 16311), (2, "NVIDIA GeForce RTX 5060 Ti", 16311),
             (3, "NVIDIA GeForce RTX 5060 Ti", 16311)]
    out = []
    for index, name, total in cards:
        free_mb = (free or {}).get(index, total - 400)
        out.append({"index": index, "name": name, "total_mb": total, "used_mb": total - free_mb, "free_mb": free_mb})
    return out


class FakeLease:
    def __init__(self, factory: "FakeLeaseFactory", kwargs: dict[str, Any]):
        self.factory = factory
        self.kwargs = kwargs
        self.gpu = kwargs.get("gpu") if isinstance(kwargs.get("gpu"), int) else None
        self.via = "hub"
        self.info: dict[str, Any] = {}
        self.warning = None
        self.released = False

    def acquire(self) -> "FakeLease":
        self.factory.wait_for_open()
        self.factory.acquired.append(self.kwargs)
        return self

    def release(self) -> None:
        self.released = True
        self.factory.released.append(self.kwargs)


class FakeLeaseFactory:
    """Stands in for ``hoard_link.lease``: records every request; ``block()`` makes ``acquire`` wait until ``open()``."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.acquired: list[dict[str, Any]] = []
        self.released: list[dict[str, Any]] = []
        self.gate = threading.Event()
        self.gate.set()

    def block(self) -> None:
        self.gate.clear()

    def open(self) -> None:
        self.gate.set()

    def wait_for_open(self) -> None:
        self.gate.wait(30)

    def __call__(self, **kwargs: Any) -> FakeLease:
        self.requests.append(kwargs)
        return FakeLease(self, kwargs)


def install_tool(path: Path, template: str) -> Path:
    """An executable script that runs the template with the interpreter running the tests."""
    body = (HERE / "fake_tools" / template).read_text(encoding="utf-8")
    path.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def make_toolchain(root: Path, *, executable: bool = EXECUTABLE_TOOLS) -> dict[str, Path]:
    """``bin/`` with llama-quantize, llama-imatrix, llama-perplexity, llama-server and ollama (the product's own file names, so with
    ``.exe`` on Windows); ``src/`` with the fake conversion scripts, which are plain Python and work everywhere.

    ``executable=False`` writes inert placeholders instead of runnable fakes."""
    from pygmalion_hoard.workdir import EXE

    bin_dir, src_dir = root / "bin", root / "src"
    bin_dir.mkdir(parents=True, exist_ok=True)
    src_dir.mkdir(parents=True, exist_ok=True)
    for name, template in (("llama-quantize", "exec_quantize.py"), ("llama-imatrix", "exec_imatrix.py"), ("llama-perplexity", "exec_perplexity.py"),
                           ("llama-server", "exec_server.py"), ("ollama", "exec_ollama.py")):
        if executable:
            install_tool(bin_dir / f"{name}{EXE}", template)
        else:
            (bin_dir / f"{name}{EXE}").write_bytes(b"placeholder: not a runnable program\n")
    for name in ("convert_hf_to_gguf.py", "convert_lora_to_gguf.py"):
        body = (HERE / "fake_tools" / name).read_text(encoding="utf-8")
        (src_dir / name).write_text(f"import sys\nsys.path.insert(0, {str(HERE)!r})\n" + body, encoding="utf-8")
    return {"bin": bin_dir, "src": src_dir, "ollama": bin_dir / f"ollama{EXE}", "ollama_log": root / "ollama-calls.jsonl"}


def chat_records(n: int, topic: str = "casa") -> list[dict[str, Any]]:
    return [{"messages": [{"role": "user", "content": f"Pregunta número {i} sobre la {topic} de la abuela."},
                          {"role": "assistant", "content": f"Respuesta {i}: la {topic} tiene muchas historias que contar, sobre todo la número {i}."}]} for i in range(n)]
