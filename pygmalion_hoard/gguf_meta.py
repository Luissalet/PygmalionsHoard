"""Read the metadata of a GGUF file without loading it: header and key/value pairs only, never tensor data. Pure Python.

The format: ``GGUF`` magic, version (2 or 3), tensor count, key/value count, then the pairs. Large arrays (the tokenizer's
vocabulary) are skipped and reported by length.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Any, BinaryIO, Optional

from .errors import PygmalionError

MAGIC = b"GGUF"
T_U8, T_I8, T_U16, T_I16, T_U32, T_I32, T_F32, T_BOOL, T_STR, T_ARR, T_U64, T_I64, T_F64 = range(13)
_FIXED = {T_U8: ("<B", 1), T_I8: ("<b", 1), T_U16: ("<H", 2), T_I16: ("<h", 2), T_U32: ("<I", 4), T_I32: ("<i", 4),
          T_F32: ("<f", 4), T_BOOL: ("<?", 1), T_U64: ("<Q", 8), T_I64: ("<q", 8), T_F64: ("<d", 8)}
MAX_STRING = 1 << 24
KEPT_ARRAY_ITEMS = 16

FILE_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M",
              13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS",
              21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S", 25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S",
              29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M", 32: "BF16"}


class GgufError(PygmalionError):
    def __init__(self, key: str, **params: Any):
        super().__init__("invalid", key, **params)


def _read(fh: BinaryIO, n: int) -> bytes:
    data = fh.read(n)
    if len(data) != n:
        raise GgufError("gguf_truncated")
    return data


def _string(fh: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read(fh, 8))
    if length > MAX_STRING:
        raise GgufError("gguf_string_long")
    return _read(fh, length).decode("utf-8", "replace")


def _value(fh: BinaryIO, kind: int, keep: bool = True) -> Any:
    if kind in _FIXED:
        fmt, size = _FIXED[kind]
        return struct.unpack(fmt, _read(fh, size))[0]
    if kind == T_STR:
        return _string(fh)
    if kind == T_ARR:
        (item_kind,) = struct.unpack("<I", _read(fh, 4))
        (count,) = struct.unpack("<Q", _read(fh, 8))
        if item_kind in _FIXED:
            size = _FIXED[item_kind][1]
            head = min(count, KEPT_ARRAY_ITEMS)
            items = [struct.unpack(_FIXED[item_kind][0], _read(fh, size))[0] for _ in range(head)]
            fh.seek((count - head) * size, os.SEEK_CUR)
        else:
            items = []
            for index in range(count):
                item = _value(fh, item_kind)
                if index < KEPT_ARRAY_ITEMS:
                    items.append(item)
        return {"_array": True, "length": count, "items": items}
    raise GgufError("gguf_value_type", kind=kind)


def read_metadata(path: str | Path) -> dict[str, Any]:
    """All key/value pairs plus ``_tensor_count``, ``_version`` and ``_file_size``."""
    path = Path(path)
    try:
        size = path.stat().st_size
        fh = open(path, "rb", buffering=1 << 20)
    except OSError as exc:
        raise PygmalionError("not_found", "file_unreadable", path=path, detail=exc.strerror or str(exc)) from exc
    with fh:
        if _read(fh, 4) != MAGIC:
            raise GgufError("gguf_magic")
        (version,) = struct.unpack("<I", _read(fh, 4))
        if version not in (2, 3):
            raise GgufError("gguf_version", version=version)
        tensor_count, kv_count = struct.unpack("<QQ", _read(fh, 16))
        if kv_count > 100_000:
            raise GgufError("gguf_count")
        meta: dict[str, Any] = {}
        for _ in range(kv_count):
            key = _string(fh)
            (kind,) = struct.unpack("<I", _read(fh, 4))
            meta[key] = _value(fh, kind)
    meta["_tensor_count"] = tensor_count
    meta["_version"] = version
    meta["_file_size"] = size
    return meta


def _num(meta: dict[str, Any], key: str) -> Optional[int]:
    value = meta.get(key)
    if isinstance(value, dict) and value.get("_array"):
        items = value.get("items") or []
        return int(max(items)) if items else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def summarize(meta: dict[str, Any]) -> dict[str, Any]:
    """The numbers the apps care about, found under ``<architecture>.*``."""
    arch = str(meta.get("general.architecture") or "")
    get = lambda suffix: _num(meta, f"{arch}.{suffix}")  # noqa: E731
    heads, kv_heads = get("attention.head_count"), get("attention.head_count_kv")
    embedding = get("embedding_length")
    head_dim = get("attention.key_length") or (embedding // heads if embedding and heads else None)
    file_type = _num(meta, "general.file_type")
    vocab = meta.get("tokenizer.ggml.tokens")
    return {
        "architecture": arch, "name": meta.get("general.name") or "", "context_length": get("context_length"),
        "block_count": get("block_count"), "embedding_length": embedding, "head_count": heads, "head_count_kv": kv_heads or heads,
        "head_dim": head_dim, "full_attention_interval": get("full_attention_interval"),
        "file_type": FILE_TYPES.get(file_type, str(file_type) if file_type is not None else ""),
        "has_chat_template": bool(meta.get("tokenizer.chat_template")), "tensor_count": meta.get("_tensor_count"),
        "vocab_size": vocab.get("length") if isinstance(vocab, dict) else None, "file_size": meta.get("_file_size"),
        "rope_scaling_type": meta.get(f"{arch}.rope.scaling.type"), "rope_scaling_factor": meta.get(f"{arch}.rope.scaling.factor"),
        "rope_original_context": get("rope.scaling.original_context_length"),
    }


def kv_cache_bytes(summary: dict[str, Any], context: int, bytes_per_value: int = 2) -> Optional[int]:
    """K and V for ``context`` tokens. Hybrid models with linear attention only keep a cache for the full-attention layers."""
    layers, kv_heads, head_dim = summary.get("block_count"), summary.get("head_count_kv"), summary.get("head_dim")
    if not (layers and kv_heads and head_dim):
        return None
    interval = summary.get("full_attention_interval")
    attention_layers = max(1, layers // interval) if interval and interval > 1 else layers
    return int(2 * attention_layers * kv_heads * head_dim * context * bytes_per_value)


def fit_table(summary: dict[str, Any], contexts: list[int], free_mb: Optional[int] = None, bytes_per_value: int = 2,
              overhead_mb: int = 600) -> list[dict[str, Any]]:
    """For each context: KV cache and total memory (file + cache + headroom), and whether it fits in ``free_mb``."""
    weights_mb = (summary.get("file_size") or 0) / 1048576
    rows = []
    for context in contexts:
        kv = kv_cache_bytes(summary, context, bytes_per_value)
        kv_mb = None if kv is None else kv / 1048576
        total = None if kv_mb is None else weights_mb + kv_mb + overhead_mb
        rows.append({"context": context, "kv_mb": None if kv_mb is None else round(kv_mb), "total_mb": None if total is None else round(total),
                     "fits": None if total is None or free_mb is None else total <= free_mb})
    return rows


def max_trained_context(summary: dict[str, Any]) -> Optional[int]:
    """The context the weights were trained for: the original length when rope scaling stretches the advertised one."""
    return summary.get("rope_original_context") or summary.get("context_length")
