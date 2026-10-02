"""Build a small but valid GGUF file (header and key/value pairs, no tensors) for tests."""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Optional

T_U32, T_F32, T_BOOL, T_STR, T_ARR, T_U64 = 4, 6, 7, 8, 9, 10


def _s(text: str) -> bytes:
    raw = text.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _kv(key: str, value: Any) -> bytes:
    out = _s(key)
    if isinstance(value, bool):
        return out + struct.pack("<IB", T_BOOL, int(value))
    if isinstance(value, int):
        return out + struct.pack("<II", T_U32, value)
    if isinstance(value, float):
        return out + struct.pack("<If", T_F32, value)
    if isinstance(value, str):
        return out + struct.pack("<I", T_STR) + _s(value)
    if isinstance(value, list):
        body = b"".join(_s(x) for x in value)
        return out + struct.pack("<IIQ", T_ARR, T_STR, len(value)) + body
    raise TypeError(type(value))


def build_gguf(path: str | Path, *, arch: str = "fakearch", context: int = 4096, blocks: int = 4, heads: int = 8, kv_heads: int = 2,
               embedding: int = 256, file_type: int = 1, chat_template: bool = True, rope: Optional[dict[str, Any]] = None,
               size: int = 0, tokens: Optional[list[str]] = None, name: str = "fake") -> Path:
    pairs: list[tuple[str, Any]] = [
        ("general.architecture", arch), ("general.name", name), ("general.file_type", file_type),
        (f"{arch}.context_length", context), (f"{arch}.block_count", blocks), (f"{arch}.embedding_length", embedding),
        (f"{arch}.attention.head_count", heads), (f"{arch}.attention.head_count_kv", kv_heads),
        ("tokenizer.ggml.tokens", tokens or ["<s>", "</s>", "a", "b"]),
    ]
    if chat_template:
        pairs.append(("tokenizer.chat_template", "{{ messages }}"))
    for key, value in (rope or {}).items():
        pairs.append((f"{arch}.rope.{key}", value))
    body = b"".join(_kv(k, v) for k, v in pairs)
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(pairs)) + body
    if size > len(data):
        data += b"\0" * (size - len(data))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path
