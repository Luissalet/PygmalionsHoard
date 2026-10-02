"""Reading and writing safetensors one tensor at a time, with numpy only (no torch, no safetensors package).

A safetensors file is 8 bytes (header length, little endian), a JSON header ``{name: {dtype, shape, data_offsets}}`` and the raw
tensor bytes. Reading seeks to one tensor and returns it; writing needs the header first, so the plan (names, dtypes, shapes) is
fixed before the bytes are produced one by one. That is what lets a merge run over 27B-parameter models without ever holding more
than a few tensors in memory.

bfloat16 has no numpy type: it is carried as uint16 and converted to float32 by shifting (exact), and back with round-to-nearest-even.
"""

from __future__ import annotations

import json
import os
import re
import struct
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

ITEMSIZE = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1, "U16": 2, "U32": 4, "U64": 8,
            "F8_E4M3": 1, "F8_E5M2": 1}
FLOAT_DTYPES = ("F64", "F32", "F16", "BF16")
MAX_HEADER = 100 * 1024 * 1024


class StioError(Exception):
    pass


def read_header(path: str | Path) -> tuple[dict[str, Any], int]:
    """``(header without __metadata__, offset where the data starts)``."""
    with open(path, "rb") as fh:
        raw = fh.read(8)
        if len(raw) != 8:
            raise StioError(f"{path}: too short to be a safetensors file")
        (size,) = struct.unpack("<Q", raw)
        if size > MAX_HEADER:
            raise StioError(f"{path}: the header size is implausible")
        header = json.loads(fh.read(size).decode("utf-8"))
    header.pop("__metadata__", None)
    return header, 8 + size


def tensor_nbytes(info: dict[str, Any]) -> int:
    n = 1
    for d in info["shape"]:
        n *= d
    return n * ITEMSIZE[info["dtype"]]


def read_raw(path: str | Path, info: dict[str, Any], data_start: int) -> bytes:
    start, end = info["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(data_start + start)
        data = fh.read(end - start)
    if len(data) != end - start:
        raise StioError(f"{path}: truncated tensor data")
    return data


def to_float32(raw: bytes, dtype: str, shape: list[int]):
    import numpy as np
    if dtype == "F32":
        return np.frombuffer(raw, dtype="<f4").reshape(shape).copy()
    if dtype == "F16":
        return np.frombuffer(raw, dtype="<f2").astype(np.float32).reshape(shape)
    if dtype == "F64":
        return np.frombuffer(raw, dtype="<f8").astype(np.float32).reshape(shape)
    if dtype == "BF16":
        bits = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        return bits.view(np.float32).reshape(shape)
    raise StioError(f"{dtype} is not a floating point type this tool can merge")


def from_float32(array, dtype: str) -> bytes:
    import numpy as np
    a = np.ascontiguousarray(array, dtype=np.float32)
    if dtype == "F32":
        return a.astype("<f4").tobytes()
    if dtype == "F16":
        return a.astype("<f2").tobytes()
    if dtype == "F64":
        return a.astype("<f8").tobytes()
    if dtype == "BF16":
        bits = a.view(np.uint32)
        finite = np.isfinite(a)
        rounding = ((bits >> 16) & 1) + np.uint32(0x7FFF)           # round to nearest, ties to even
        rounded = ((bits + rounding) >> 16).astype(np.uint16)
        truncated = (bits >> 16).astype(np.uint16)                   # inf and NaN keep their pattern
        return np.where(finite, rounded, truncated).astype("<u2").tobytes()
    raise StioError(f"cannot write {dtype} from float32")


def weight_files(model_dir: str | Path) -> list[Path]:
    """The safetensors shards of a model folder, in index order when an index exists."""
    root = Path(model_dir)
    index = root / "model.safetensors.index.json"
    if index.is_file():
        try:
            names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
            files = [root / n for n in names]
            if all(f.is_file() for f in files):
                return files
        except (ValueError, KeyError, OSError):
            pass
    return sorted(root.glob("*.safetensors"))


def model_tensors(model_dir: str | Path) -> dict[str, dict[str, Any]]:
    """``{tensor name: {dtype, shape, file, data_start, data_offsets}}`` across all shards (headers only)."""
    out: dict[str, dict[str, Any]] = {}
    files = weight_files(model_dir)
    if not files:
        raise StioError(f"{model_dir}: no .safetensors files")
    for f in files:
        header, start = read_header(f)
        for name, info in header.items():
            if name in out:
                raise StioError(f"{model_dir}: tensor {name} appears in two shards")
            out[name] = {**info, "file": str(f), "data_start": start}
    return out


def read_tensor(entry: dict[str, Any]):
    return to_float32(read_raw(entry["file"], entry, entry["data_start"]), entry["dtype"], entry["shape"])


def check_compatible(models: list[str | Path]) -> dict[str, Any]:
    """Compare the tensors of several models: same names and shapes are required, different dtypes are reported.

    Returns ``{ok, models, tensors, problems: [...], dtype_differences: [...]}``; ``problems`` is empty when the merge can start."""
    problems: list[str] = []
    dtype_notes: list[str] = []
    tensors = [model_tensors(m) for m in models]
    first = tensors[0]
    for idx, other in enumerate(tensors[1:], start=1):
        only_first = sorted(set(first) - set(other))
        only_other = sorted(set(other) - set(first))
        if only_first:
            problems.append(f"model {idx + 1} lacks {len(only_first)} tensors of the first (e.g. {only_first[0]})")
        if only_other:
            problems.append(f"model {idx + 1} has {len(only_other)} tensors the first lacks (e.g. {only_other[0]})")
        for name in sorted(set(first) & set(other)):
            if first[name]["shape"] != other[name]["shape"]:
                problems.append(f"{name}: shape {first[name]['shape']} in the first, {other[name]['shape']} in model {idx + 1}")
                if len(problems) >= 12:
                    break
            if first[name]["dtype"] != other[name]["dtype"] and len(dtype_notes) < 5:
                dtype_notes.append(f"{name}: {first[name]['dtype']} vs {other[name]['dtype']} (cast to the first model's)")
    return {"ok": not problems, "models": len(models), "tensors": len(first), "problems": problems[:12], "dtype_differences": dtype_notes,
            "bytes": sum(tensor_nbytes(i) for i in first.values())}


def plan_shards(entries: list[tuple[str, str, list[int]]], max_bytes: int) -> list[list[tuple[str, str, list[int]]]]:
    """Group ``(name, dtype, shape)`` entries into shards of at most ``max_bytes`` (a tensor bigger than that gets its own)."""
    shards, current, size = [], [], 0
    for entry in entries:
        n = tensor_nbytes({"dtype": entry[1], "shape": entry[2]})
        if current and size + n > max_bytes:
            shards.append(current)
            current, size = [], 0
        current.append(entry)
        size += n
    if current:
        shards.append(current)
    return shards


def write_shard(path: str | Path, entries: list[tuple[str, str, list[int]]], produce: Callable[[str], bytes],
                metadata: Optional[dict[str, str]] = None) -> None:
    """Write one safetensors file: the header from ``entries``, then ``produce(name)`` for each tensor in order."""
    header: dict[str, Any] = {"__metadata__": metadata or {"format": "pt"}}
    offset = 0
    for name, dtype, shape in entries:
        n = tensor_nbytes({"dtype": dtype, "shape": shape})
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + n]}
        offset += n
    blob = json.dumps(header, separators=(",", ":")).encode("utf-8")
    blob += b" " * ((8 - len(blob) % 8) % 8)                    # the format asks for an 8-byte aligned data start
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for name, dtype, shape in entries:
            data = produce(name)
            expected = tensor_nbytes({"dtype": dtype, "shape": shape})
            if len(data) != expected:
                raise StioError(f"{name}: produced {len(data)} bytes, expected {expected}")
            fh.write(data)
    os.replace(tmp, path)


def write_index(root: str | Path, weight_map: dict[str, str], total_size: int) -> None:
    Path(root, "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2), encoding="utf-8")


SKIP_WHEN_COPYING = re.compile(r"(\.safetensors$|^model\.safetensors\.index\.json$|\.bin$|\.pt$|\.pth$|\.gguf$|^\.cache$)")


def copy_non_weights(src: str | Path, dst: str | Path) -> list[str]:
    """Copy config, tokenizer and template files (everything that is not weights) from the first model."""
    import shutil
    copied = []
    for item in Path(src).iterdir():
        if item.is_file() and not SKIP_WHEN_COPYING.search(item.name):
            shutil.copy2(item, Path(dst) / item.name)
            copied.append(item.name)
    return copied
