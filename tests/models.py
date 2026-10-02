"""Tiny Hugging Face model folders for tests: a config and safetensors written with the app's own streaming writer."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pygmalion_hoard" / "workers"))
import _stio as S  # noqa: E402


def tensor_names(layers: int) -> list[tuple[str, tuple[int, ...]]]:
    return [("model.embed_tokens.weight", (32, 8))] + [
        item for i in range(layers) for item in (
            (f"model.layers.{i}.self_attn.q_proj.weight", (8, 8)), (f"model.layers.{i}.self_attn.k_proj.weight", (4, 8)),
            (f"model.layers.{i}.self_attn.v_proj.weight", (4, 8)), (f"model.layers.{i}.self_attn.o_proj.weight", (8, 8)),
            (f"model.layers.{i}.mlp.up_proj.weight", (16, 8)), (f"model.layers.{i}.input_layernorm.weight", (8,)))
    ] + [("lm_head.weight", (32, 8))]


def make_model(folder: Path, *, layers: int = 2, seed: int = 0, dtype: str = "F32", arch: str = "FakeForCausalLM", shard_bytes: int = 0,
               context: int = 2048, extra_config: dict | None = None) -> Path:
    """Write ``config.json``, a tokenizer stub and ``model.safetensors`` (several shards when ``shard_bytes`` is small)."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arrays = {name: rng.standard_normal(shape).astype("<f4") for name, shape in tensor_names(layers)}
    config = {"architectures": [arch], "model_type": "fake", "hidden_size": 8, "num_hidden_layers": layers, "num_attention_heads": 2,
              "num_key_value_heads": 1, "intermediate_size": 16, "vocab_size": 32, "max_position_embeddings": context, "torch_dtype": "float32",
              "rope_theta": 10000.0, **(extra_config or {})}
    (folder / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (folder / "tokenizer.json").write_text('{"version": "1.0"}', encoding="utf-8")
    (folder / "tokenizer_config.json").write_text('{"chat_template": "{{ messages }}"}', encoding="utf-8")
    entries = [(n, dtype, list(a.shape)) for n, a in arrays.items()]
    shards = S.plan_shards(entries, shard_bytes) if shard_bytes else [entries]
    weight_map = {}
    for number, group in enumerate(shards, start=1):
        fname = f"model-{number:05d}-of-{len(shards):05d}.safetensors" if len(shards) > 1 else "model.safetensors"
        S.write_shard(folder / fname, group, lambda n: S.from_float32(arrays[n], dtype))
        for n, _d, _s in group:
            weight_map[n] = fname
    if len(shards) > 1:
        S.write_index(folder, weight_map, sum(S.tensor_nbytes({"dtype": d, "shape": s}) for _, d, s in entries))
    return folder


def read_all(folder: Path) -> dict[str, np.ndarray]:
    return {name: S.read_tensor(entry) for name, entry in S.model_tensors(folder).items()}


def make_adapter(folder: Path, base: Path, *, rank: int = 2, alpha: int = 4, seed: int = 0, modules=("q_proj", "v_proj"), prefix: str = "base_model.model.",
                 extra_config: dict | None = None, scale_b: float = 0.1) -> Path:
    """A PEFT-style LoRA folder (adapter_config.json + adapter_model.safetensors) for the tiny base."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    base_tensors = S.model_tensors(base)
    arrays = {}
    for name, info in base_tensors.items():
        if not name.endswith(".weight") or not any(f".{m}." in name for m in modules):
            continue
        out_dim, in_dim = info["shape"]
        stem = prefix + name[: -len(".weight")]
        arrays[f"{stem}.lora_A.weight"] = rng.standard_normal((rank, in_dim)).astype("<f4")
        arrays[f"{stem}.lora_B.weight"] = (rng.standard_normal((out_dim, rank)) * scale_b).astype("<f4")
    entries = [(n, "F32", list(a.shape)) for n, a in arrays.items()]
    S.write_shard(folder / "adapter_model.safetensors", entries, lambda n: S.from_float32(arrays[n], "F32"))
    config = {"peft_type": "LORA", "r": rank, "lora_alpha": alpha, "base_model_name_or_path": str(base), "target_modules": list(modules),
              **(extra_config or {})}
    (folder / "adapter_config.json").write_text(json.dumps(config), encoding="utf-8")
    return folder
