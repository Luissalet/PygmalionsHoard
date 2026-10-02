"""Fold a LoRA adapter into its base model and write a plain Hugging Face folder (safetensors shards of 4 GB, tokenizer, config).

Arguments (args.json): ``base_path``, ``adapter_path``, ``output_dir``, ``device`` (``cpu`` by default, so no VRAM is needed),
``shard_gb`` (default 4), ``mode`` (``auto``, ``stream`` or ``transformers``).

Two ways, picked automatically:

* ``stream``: the adapter's tensor names are matched to the base model's, and ``W + scale * B @ A`` is computed one tensor at a time
  in float32 (numpy) while the base is read shard by shard. Memory use is one tensor, the output keeps every file of the base
  (config, tokenizer, untouched tensors) exactly, and no model class has to exist in the installed transformers.
* ``transformers``: load the base on the chosen device, apply the adapter with PEFT and call ``merge_and_unload``. Used when the
  adapter has DoRA, extra saved modules or names that cannot be matched to the base's tensors.

The adapter is untrusted input: only its safetensors file and JSON config are read; a pickle-based ``adapter_model.bin`` is refused.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import _stio as S  # noqa: E402

PREFIX = "base_model.model."
A_SUFFIX, B_SUFFIX = ".lora_A.weight", ".lora_B.weight"


def read_adapter_config(adapter_dir: Path) -> dict[str, Any]:
    try:
        return json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise C.WorkerError("adapter_config_unreadable", path=adapter_dir) from exc


def lora_scale(config: dict[str, Any]) -> float:
    rank = max(1, int(config.get("r", 8)))
    alpha = float(config.get("lora_alpha", rank))
    return alpha / math.sqrt(rank) if config.get("use_rslora") else alpha / rank


def streamable(config: dict[str, Any], adapter_names: list[str]) -> tuple[bool, str]:
    """Whether the stream path can apply this adapter, and why not when it cannot."""
    if config.get("use_dora"):
        return False, "the adapter uses DoRA"
    if config.get("modules_to_save") or config.get("trainable_token_indices"):
        return False, "the adapter saves extra modules"
    if config.get("rank_pattern") or config.get("alpha_pattern"):
        return False, "the adapter has per-module ranks or alphas (one scale does not fit every layer)"
    if config.get("fan_in_fan_out"):
        return False, "the adapter stores transposed (fan_in_fan_out) weights"
    other = [n for n in adapter_names if not (n.endswith(A_SUFFIX) or n.endswith(B_SUFFIX))]
    if other:
        return False, f"the adapter has tensors that are not LoRA matrices (e.g. {other[0]})"
    return True, ""


def resolve_name(adapter_key: str, base_names: set[str], suffix_index: dict[str, list[str]]) -> Optional[str]:
    """The base tensor a ``...lora_A.weight`` key refers to, or None. Tries the exact name, the nested-language-model spelling
    differences, and finally a unique match on the part from ``layers.`` onwards."""
    name = adapter_key[len(PREFIX):] if adapter_key.startswith(PREFIX) else adapter_key
    name = name[: -len(A_SUFFIX)] if name.endswith(A_SUFFIX) else name[: -len(B_SUFFIX)]
    target = name + ".weight"
    candidates = [target, target.replace(".language_model.", "."), target.replace("model.", "model.language_model.", 1)]
    for candidate in candidates:
        if candidate in base_names:
            return candidate
    at = target.find("layers.")
    if at != -1:
        found = suffix_index.get(target[at:], [])
        if len(found) == 1:
            return found[0]
    return None


def build_suffix_index(base_names: set[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for n in base_names:
        at = n.find("layers.")
        if at != -1:
            index.setdefault(n[at:], []).append(n)
    return index


def pair_up(adapter: dict[str, Any], base_names: set[str]) -> tuple[dict[str, tuple[str, str]], list[str]]:
    """``{base tensor: (lora_A key, lora_B key)}`` and the adapter modules that match nothing."""
    suffixes = build_suffix_index(base_names)
    pairs: dict[str, tuple[str, str]] = {}
    unmatched: list[str] = []
    for key in sorted(adapter):
        if not key.endswith(A_SUFFIX):
            continue
        b_key = key[: -len(A_SUFFIX)] + B_SUFFIX
        target = resolve_name(key, base_names, suffixes)
        if target is None or b_key not in adapter:
            unmatched.append(key)
            continue
        pairs[target] = (key, b_key)
    return pairs, unmatched


class ShapeMisfit(C.WorkerError):
    """The adapter's matrices do not fit the base tensors on disk: another base model, or a model stored with fused or split
    projections that only the library's loader puts back together."""

    def __init__(self, name: str, lora_b: Any, lora_a: Any, weight: Any):
        super().__init__("adapter_shape_misfit", name=name, lora_b=str(list(lora_b)), lora_a=str(list(lora_a)), weight=str(list(weight)))


def shape_mismatch(pairs: dict[str, tuple[str, str]], base: dict[str, Any], adapter: dict[str, Any]) -> tuple:
    """Empty when every ``B @ A`` has the shape of the base tensor it is added to; else the first mismatch as ``(tensor, B, A, weight)``."""
    for target, (a_key, b_key) in pairs.items():
        weight, lora_a, lora_b = list(base[target]["shape"]), list(adapter[a_key]["shape"]), list(adapter[b_key]["shape"])
        if len(weight) != 2 or len(lora_a) != 2 or len(lora_b) != 2 or lora_b[0] != weight[0] or lora_a[1] != weight[1] or lora_b[1] != lora_a[0]:
            return (target, lora_b, lora_a, weight)
    return ()


def merge_stream(base_dir: Path, adapter_dir: Path, out: Path, config: dict[str, Any], shard_gb: float) -> Optional[dict[str, Any]]:
    """Returns the summary, or None when this adapter cannot be merged by streaming (the caller falls back)."""
    adapter = S.model_tensors(adapter_dir)
    ok, why = streamable(config, list(adapter))
    if not ok:
        C.log(f"Streaming merge not possible: {why}.")
        return None
    base = S.model_tensors(base_dir)
    pairs, unmatched = pair_up(adapter, set(base))
    if unmatched or not pairs:
        C.log(f"Streaming merge not possible: {len(unmatched)} adapter modules do not match the base tensors" + (f" (e.g. {unmatched[0]})." if unmatched else "."))
        return None
    misfit = shape_mismatch(pairs, base, adapter)
    if misfit:
        # checked before anything is written, so nothing half-merged is left behind
        raise ShapeMisfit(*misfit)
    scale = lora_scale(config)
    entries = [(n, i["dtype"], list(i["shape"])) for n, i in base.items()]
    out.mkdir(parents=True, exist_ok=True)
    shards = S.plan_shards(entries, int(shard_gb * (1 << 30)))
    total, done, started = len(entries), 0, time.monotonic()
    weight_map: dict[str, str] = {}
    changed = 0

    def produce(name: str) -> bytes:
        nonlocal done, changed
        if C.STOP.is_set():
            raise C.WorkerError("merge_stopped")
        info = base[name]
        if name in pairs and info["dtype"] in S.FLOAT_DTYPES:
            a_key, b_key = pairs[name]
            weight = S.read_tensor(info)
            lora_a, lora_b = S.read_tensor(adapter[a_key]), S.read_tensor(adapter[b_key])
            if lora_b.shape[0] != weight.shape[0] or lora_a.shape[1] != weight.shape[1]:
                raise ShapeMisfit(name, lora_b.shape, lora_a.shape, weight.shape)
            data = S.from_float32(weight + scale * (lora_b @ lora_a), info["dtype"])
            changed += 1
        else:
            data = S.read_raw(info["file"], info, info["data_start"])
        done += 1
        if done % 10 == 0 or done == total:
            C.emit("progress", step=done, total=total, eta_s=C.eta_seconds(done, total, time.monotonic() - started))
        return data

    for number, group in enumerate(shards, start=1):
        fname = f"model-{number:05d}-of-{len(shards):05d}.safetensors" if len(shards) > 1 else "model.safetensors"
        S.write_shard(out / fname, group, produce)
        for name, _d, _s in group:
            weight_map[name] = fname
    if len(shards) > 1:
        S.write_index(out, weight_map, sum(S.tensor_nbytes({"dtype": d, "shape": s}) for _, d, s in entries))
    copied = S.copy_non_weights(base_dir, out)
    return {"mode": "stream", "tensors": total, "merged_tensors": changed, "shards": len(shards), "scale": scale, "copied": copied}


def merge_transformers(base_dir: Path, adapter_dir: Path, out: Path, device: str, shard_gb: float) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise C.WorkerError("merge_cuda_missing")
    C.log(f"Loading the base on {device} in bfloat16.")
    kwargs = {"device_map": {"": device}}
    try:
        model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        model = AutoModelForCausalLM.from_pretrained(str(base_dir), torch_dtype=torch.bfloat16, **kwargs)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    C.log("Merging the adapter.")
    merged = model.merge_and_unload()
    out.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out), max_shard_size=f"{int(shard_gb)}GB", safe_serialization=True)
    AutoTokenizer.from_pretrained(str(base_dir)).save_pretrained(str(out))
    for extra in ("chat_template.jinja", "generation_config.json", "preprocessor_config.json"):
        src = base_dir / extra
        if src.is_file() and not (out / extra).exists():
            (out / extra).write_bytes(src.read_bytes())
    base_arch = json.loads((base_dir / "config.json").read_text(encoding="utf-8")).get("architectures")
    new_arch = json.loads((out / "config.json").read_text(encoding="utf-8")).get("architectures")
    if base_arch != new_arch:
        C.log(f"Note: the architecture changed from {base_arch} to {new_arch}; check that the GGUF converter knows it.")
    return {"mode": "transformers", "device": device}


def merge(args: dict[str, Any]) -> dict[str, Any]:
    base_dir, adapter_dir, out = Path(args["base_path"]), Path(args["adapter_path"]), Path(args["output_dir"])
    if not (base_dir / "config.json").is_file():
        raise C.WorkerError("merge_base_invalid", path=base_dir)
    if not list(adapter_dir.glob("adapter_model.safetensors")):
        raise C.WorkerError("adapter_weights_missing", path=adapter_dir)
    config = read_adapter_config(adapter_dir)
    mode = args.get("mode", "auto")
    shard_gb = float(args.get("shard_gb", 4))
    summary = None
    misfit: Optional[ShapeMisfit] = None
    if mode in ("auto", "stream"):
        try:
            summary = merge_stream(base_dir, adapter_dir, out, config, shard_gb)
        except ShapeMisfit as exc:
            if mode == "stream":
                raise
            misfit = exc
            C.log(f"Streaming merge not possible: {exc.msg}; trying the library's loader.")
        if summary is None and misfit is None and mode == "stream":
            raise C.WorkerError("adapter_not_streamable")
    if summary is None:
        try:
            summary = merge_transformers(base_dir, adapter_dir, out, str(args.get("device", "cpu")), shard_gb)
        except Exception as exc:  # noqa: BLE001 — with a shape misfit, that is the reason worth reporting
            if misfit is not None:
                raise misfit from exc
            raise
    summary.update({"base_path": str(base_dir), "adapter_path": str(adapter_dir), "rank": config.get("r"), "alpha": config.get("lora_alpha")})
    C.emit("artifact", path=str(out), kind="merged")
    C.emit("result", data=summary)
    return summary


def main(args: dict[str, Any]) -> None:
    merge(args)


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
