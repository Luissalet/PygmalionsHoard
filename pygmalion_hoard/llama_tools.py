"""Command lines and output parsers for the llama.cpp programs: the conversion scripts, ``llama-imatrix``, ``llama-quantize`` and
``llama-perplexity``. Pure functions; running them is ``procs.run_streaming``'s job."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence

from .errors import PygmalionError
from .messages import text

# type -> (bits per weight on average, needs an importance matrix to be any good, note)
QUANT_TYPES: dict[str, tuple[float, bool, str]] = {
    "Q8_0": (8.5, False, "Almost lossless; about half the size of f16."),
    "Q6_K": (6.56, False, "Very close to the original."),
    "Q5_K_M": (5.69, False, "Good balance for 16 GB cards."),
    "Q4_K_M": (4.85, False, "The usual choice: small with little loss."),
    "IQ4_XS": (4.3, True, "Smaller than Q4_K_M; use an importance matrix."),
    "Q3_K_M": (3.9, False, "Noticeable loss; for models that would not fit otherwise."),
    "IQ3_M": (3.66, True, "Very small; poor without an importance matrix."),
}
OUT_TYPES = ("f16", "bf16", "q8_0")
IMATRIX_RECOMMENDED = tuple(t for t, (_b, need, _n) in QUANT_TYPES.items() if need)


def check_quant_types(types: Sequence[str], imatrix: bool) -> tuple[list[str], list[str]]:
    """The normalised list of types and the warnings (an IQ type without an importance matrix)."""
    out, warnings = [], []
    for raw in types:
        t = str(raw).strip().upper()
        if t not in QUANT_TYPES:
            raise PygmalionError("invalid", "quant_type_unknown", type=raw, options=list(QUANT_TYPES))
        if t not in out:
            out.append(t)
        if QUANT_TYPES[t][1] and not imatrix:
            warnings.append(text("quant_needs_imatrix", type=t))
    if not out:
        raise PygmalionError("invalid", "quant_type_needed", options=list(QUANT_TYPES))
    return out, warnings


def expected_size(params: int, qtype: str) -> int:
    """Bytes of a quantized file: parameters x average bits per weight / 8 (plus a little for metadata)."""
    bits = QUANT_TYPES[qtype][0] if qtype in QUANT_TYPES else 16.0
    return int(params * bits / 8 * 1.01)


# ------------------------------------------------------------------ command lines
def convert_argv(python: str, script: str, hf_dir: str, outfile: str, outtype: str) -> list[str]:
    if outtype not in OUT_TYPES:
        raise PygmalionError("invalid", "outtype_unknown", outtype=outtype, options=list(OUT_TYPES))
    return [python, script, hf_dir, "--outfile", outfile, "--outtype", outtype]


def convert_lora_argv(python: str, script: str, base_dir: str, adapter_dir: str, outfile: str, outtype: str = "f16") -> list[str]:
    return [python, script, "--base", base_dir, adapter_dir, "--outfile", outfile, "--outtype", outtype]


def imatrix_argv(exe: str, model: str, calib: str, out: str, chunks: int, ngl: int = 99) -> list[str]:
    return [exe, "-m", model, "-f", calib, "-o", out, "-ngl", str(ngl), "--chunks", str(int(chunks))]


def quantize_argv(exe: str, src: str, dst: str, qtype: str, imatrix: Optional[str] = None) -> list[str]:
    argv = [exe]
    if imatrix:
        argv += ["--imatrix", imatrix]
    return argv + [src, dst, qtype]


def perplexity_argv(exe: str, model: str, text: str, ctx: int, chunks: int, ngl: int = 99) -> list[str]:
    return [exe, "-m", model, "-f", text, "-c", str(int(ctx)), "--chunks", str(int(chunks)), "-ngl", str(ngl)]


# ------------------------------------------------------------------ parsers (each takes one output line)
_CONVERT_TENSOR = re.compile(r"hf-to-gguf:\s*(?P<name>[\w.\-]+),\s*torch\.\S+\s*-->\s*(?P<to>\S+)")
_IMATRIX_TOTAL = re.compile(r"computing over (\d+) chunks")
_IMATRIX_CHUNK = re.compile(r"\[(\d+)\](-?[\d.]+)")
_QUANT_STEP = re.compile(r"\[\s*(\d+)\s*/\s*(\d+)\s*\]")
_SIZES = re.compile(r"(model size|quant size)\s*=\s*([\d.]+)\s*(MiB|MB|GiB|GB)", re.I)
_PPL = re.compile(r"Final estimate:\s*PPL\s*=\s*([\d.]+)\s*\+/-\s*([\d.]+)")
_PPL_RUNNING = re.compile(r"\[(\d+)\]\s*([\d.]+)")


def parse_convert_line(line: str) -> Optional[str]:
    """The tensor name of a ``hf-to-gguf`` progress line, or None."""
    m = _CONVERT_TENSOR.search(line)
    return m.group("name") if m else None


def parse_imatrix_line(line: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = _IMATRIX_TOTAL.search(line)
    if m:
        out["total"] = int(m.group(1))
    chunks = _IMATRIX_CHUNK.findall(line)
    if chunks and "PPL" not in line and "computing" not in line:
        out["step"] = max(int(c[0]) for c in chunks)
    return out


def parse_quantize_line(line: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    m = _QUANT_STEP.search(line)
    if m and ("converting" in line or "size =" in line or "type =" in line):
        out["step"], out["total"] = int(m.group(1)), int(m.group(2))
    for kind, value, unit in _SIZES.findall(line):
        factor = {"mib": 1 << 20, "mb": 1 << 20, "gib": 1 << 30, "gb": 1 << 30}[unit.lower()]
        out["model_bytes" if kind.lower() == "model size" else "quant_bytes"] = int(float(value) * factor)
    return out


def parse_perplexity_line(line: str) -> dict[str, Any]:
    m = _PPL.search(line)
    if m:
        return {"ppl": float(m.group(1)), "ppl_error": float(m.group(2)), "final": True}
    # llama-perplexity prints the running values on one line ("[1]5.12,[2]5.34,[3]...") that grows as it goes: the last is the newest
    found = _PPL_RUNNING.findall(line) if line.strip().startswith("[") else []
    if found:
        chunk, value = found[-1]
        return {"chunk": int(chunk), "ppl_running": float(value)}
    return {}


def gguf_name(base: str, label: str, suffix: str = "") -> str:
    """``<base>-<label>.gguf`` with characters safe on Windows and Linux."""
    clean = re.sub(r"-{2,}", "-", re.sub(r"[^\w.\-]+", "-", f"{base}-{label}{suffix}")).strip("-")
    return clean + ".gguf"


def is_sharded_gguf(path: str | Path) -> bool:
    return bool(re.search(r"-\d{5}-of-\d{5}\.gguf$", str(path)))
