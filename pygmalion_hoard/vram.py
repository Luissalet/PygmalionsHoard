"""Memory and time estimates for a training run, from the model's config.json.

The formulas are shown in the interface next to the numbers, so a surprising estimate can be checked by hand (the time model is described
above ``THROUGHPUT_CONSTANT``):

    weights      = params x 0.55 bytes (QLoRA, 4-bit with double quantization) or x 2 bytes (LoRA, bf16);
                   with QLoRA the embeddings and the output head are not quantized and stay at 2 bytes
    lora         = lora_params x 16 bytes (fp32 weight + gradient + two Adam moments)
    checkpoints  = batch x seq x hidden x layers x 2 bytes x 2   (gradient checkpointing keeps one tensor per layer, k = 2)
    working set  = batch x seq x (3 x intermediate + 8 x hidden) x 2 bytes   (one layer being recomputed)
    logits       = batch x seq x vocab x 4 bytes x 2   (float32 logits and their gradient)
    overhead     = 700 MB (CUDA context, allocator) + 8 % of the total
"""

from __future__ import annotations

from typing import Any, Optional

from .messages import text

QLORA_BYTES_PER_PARAM = 0.55
LORA_BYTES_PER_PARAM = 2.0
LORA_STATE_BYTES = 16
CHECKPOINT_K = 2
OVERHEAD_FIXED_MB = 700
OVERHEAD_FRACTION = 0.08
MB = 1048576

# Time model. Rough constants, not a benchmark: they give the right order of magnitude before a run, and the first steps of the run replace
# them with the measured speed (the job's ``tokens_per_s`` and ETA).
#
#     throughput   = THROUGHPUT_CONSTANT / billions of parameters   (tokens per second, QLoRA on the reference card)
#                    x card scale (the card's sustained bf16 TFLOPS / the reference card's)
#                    x method speed (plain LoRA, bf16 without dequantizing, is 1.6x faster than QLoRA); at most MAX_TOKENS_PER_S
#     tokens       = records processed x the average tokens of a record (at most seq_len) x PADDING (batches are padded to their longest record)
#     time         = start-up (load, quantize, tokenize: STARTUP_S + STARTUP_S_PER_B per billion parameters)
#                    + tokens / throughput + steps x STEP_OVERHEAD_S (optimizer step, clipping, logging)
THROUGHPUT_CONSTANT = 1600.0        # tokens/s x billions of parameters: QLoRA on a 16 GB 50-series card (RTX 5060 Ti class), gradient checkpointing on
REFERENCE_TFLOPS = 90.0             # the sustained bf16 TFLOPS of that reference card
METHOD_SPEED = {"qlora": 1.0, "lora": 1.6}
MAX_TOKENS_PER_S = 8000.0           # a very small model is bound by kernel launches, not by its size
PADDING = 1.15
STARTUP_S = 20.0
STARTUP_S_PER_B = 8.0
STEP_OVERHEAD_S = 0.4
# bf16 TFLOPS that a card sustains on this kind of work, by a fragment of its name (rough: only their ratio to the reference card is used)
GPU_TFLOPS = (("5060 ti", 90.0), ("4070 ti", 80.0), ("5090", 200.0), ("4090", 160.0), ("3090", 70.0), ("4060", 60.0))
DEFAULT_TFLOPS = 60.0


def text_config(config: dict[str, Any]) -> dict[str, Any]:
    """The language model's section of a (possibly multimodal) config."""
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) and nested.get("hidden_size") else config


def _first(config: dict[str, Any], *names: str, default: Optional[float] = None) -> Optional[float]:
    for name in names:
        value = config.get(name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value
    return default


def dims(config: dict[str, Any]) -> dict[str, float]:
    c = text_config(config)
    hidden = _first(c, "hidden_size", "n_embd", "d_model", default=0) or 0
    layers = _first(c, "num_hidden_layers", "n_layer", "num_layers", default=0) or 0
    heads = _first(c, "num_attention_heads", "n_head", default=0) or 0
    kv_heads = _first(c, "num_key_value_heads", default=heads) or heads
    head_dim = _first(c, "head_dim", default=(hidden / heads if heads else 0)) or 0
    inter = _first(c, "intermediate_size", "ffn_hidden_size", "n_inner", default=4 * hidden) or 0
    experts = _first(c, "num_experts", "n_routed_experts", "num_local_experts", default=0) or 0
    moe_inter = _first(c, "moe_intermediate_size", default=inter) or inter
    shared = _first(c, "n_shared_experts", "num_shared_experts", default=0) or 0
    vocab = _first(c, "vocab_size", default=0) or 0
    return {"hidden": hidden, "layers": layers, "heads": heads, "kv_heads": kv_heads, "head_dim": head_dim, "inter": inter,
            "experts": experts, "moe_inter": moe_inter, "shared": shared, "vocab": vocab, "tied": bool(c.get("tie_word_embeddings", False))}


def estimate_params(config: dict[str, Any]) -> int:
    """Parameter count from the shapes (dense or mixture of experts). Within a few percent of the real count for common decoders."""
    d = dims(config)
    h, layers = d["hidden"], d["layers"]
    if not (h and layers):
        return 0
    qdim, kvdim = d["heads"] * d["head_dim"], d["kv_heads"] * d["head_dim"]
    attention = h * qdim * 2 + h * kvdim * 2
    if d["experts"]:
        mlp = d["experts"] * 3 * h * d["moe_inter"] + h * d["experts"] + d["shared"] * 3 * h * d["moe_inter"]
    else:
        mlp = 3 * h * d["inter"]
    embeddings = d["vocab"] * h * (1 if d["tied"] else 2)
    return int(layers * (attention + mlp) + embeddings)


def lora_params(config: dict[str, Any], rank: int) -> int:
    """LoRA parameters on every attention projection and on the dense MLP (experts are not adapted)."""
    d = dims(config)
    h, layers, inter = d["hidden"], d["layers"], d["inter"]
    if not (h and layers):
        return 0
    qdim, kvdim = d["heads"] * d["head_dim"], d["kv_heads"] * d["head_dim"]
    per_layer = rank * ((h + qdim) + 2 * (h + kvdim) + (qdim + h))
    if not d["experts"]:
        per_layer += rank * (2 * (h + inter) + (inter + h))
    return int(layers * per_layer)


def embedding_params(config: dict[str, Any]) -> int:
    """Input embeddings plus the output head (one matrix when they are tied)."""
    d = dims(config)
    return int(d["vocab"] * d["hidden"] * (1 if d["tied"] else 2))


def estimate_training_memory(config: dict[str, Any], *, method: str = "qlora", rank: int = 16, batch: int = 1, seq_len: int = 2048,
                             params: Optional[int] = None) -> dict[str, Any]:
    d = dims(config)
    n = params or estimate_params(config)
    method = "lora" if method == "lora" else "qlora"
    if method == "qlora":
        # bitsandbytes quantizes the linear layers only: a 150k-250k vocabulary keeps billions of parameters in bf16
        kept = min(n, embedding_params(config))
        weights = (n - kept) * QLORA_BYTES_PER_PARAM + kept * LORA_BYTES_PER_PARAM
    else:
        weights = n * LORA_BYTES_PER_PARAM
    lp = lora_params(config, rank)
    lora = lp * LORA_STATE_BYTES
    checkpoints = batch * seq_len * d["hidden"] * d["layers"] * 2 * CHECKPOINT_K
    working = batch * seq_len * (3 * d["inter"] + 8 * d["hidden"]) * 2
    logits = batch * seq_len * d["vocab"] * 4 * 2
    subtotal = weights + lora + checkpoints + working + logits
    overhead = OVERHEAD_FIXED_MB * MB + subtotal * OVERHEAD_FRACTION
    total = subtotal + overhead
    parts = {"weights_mb": weights / MB, "lora_mb": lora / MB, "checkpoints_mb": checkpoints / MB, "working_mb": working / MB,
             "logits_mb": logits / MB, "overhead_mb": overhead / MB}
    formula = text("mem_formula_qlora" if method == "qlora" else "mem_formula_lora", params_b=round(n / 1e9, 2), kept_b=round(kept / 1e9, 2) if method == "qlora" else 0,
                   q_bytes=QLORA_BYTES_PER_PARAM, l_bytes=LORA_BYTES_PER_PARAM, weights_mb=round(parts["weights_mb"]), lora_m=round(lp / 1e6, 1),
                   state_bytes=LORA_STATE_BYTES, lora_mb=round(parts["lora_mb"]), batch=batch, seq_len=seq_len, hidden=int(d["hidden"]), layers=int(d["layers"]),
                   k=CHECKPOINT_K, checkpoints_mb=round(parts["checkpoints_mb"]), working_mb=round(parts["working_mb"]), vocab=int(d["vocab"]),
                   logits_mb=round(parts["logits_mb"]), fixed_mb=OVERHEAD_FIXED_MB, pct=int(OVERHEAD_FRACTION * 100), overhead_mb=round(parts["overhead_mb"]))
    return {"method": method, "params": n, "params_b": round(n / 1e9, 3), "lora_params": lp, "total_mb": round(total / MB),
            **{k: round(v) for k, v in parts.items()}, "formula": formula, "batch": batch, "seq_len": seq_len, "rank": rank}


def fit_on_gpus(total_mb: int, gpus: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each candidate GPU (``index``, ``total_mb``, ``free_mb``): does the run fit in what is free now, or in the card at all."""
    out = []
    for g in gpus:
        out.append({"gpu": g["index"], "name": g.get("name", ""), "total_mb": g.get("total_mb"), "free_mb": g.get("free_mb"),
                    "fits_free": g.get("free_mb") is not None and total_mb <= g["free_mb"],
                    "fits_total": g.get("total_mb") is not None and total_mb <= g["total_mb"]})
    return out


def pick_gpus(total_mb: int, gpus: list[dict[str, Any]]) -> dict[str, Any]:
    """The fewest allowed GPUs that can hold the run. One card when any is big enough (the one with the most free memory first),
    otherwise the smallest set whose combined capacity is enough (the model is then split across them)."""
    known = [g for g in gpus if g.get("total_mb")]
    if not known:
        return {"gpus": [], "fits": None, "reason": text("pick_no_inventory")}
    single = [g for g in known if total_mb <= g["total_mb"]]
    if single:
        best = max(single, key=lambda g: (g.get("free_mb") or 0))
        return {"gpus": [best["index"]], "fits": True, "reason": "", "waits": total_mb > (best.get("free_mb") or 0)}
    ordered = sorted(known, key=lambda g: g["total_mb"], reverse=True)
    chosen, capacity = [], 0
    for g in ordered:
        chosen.append(g)
        capacity += g["total_mb"]
        if capacity >= total_mb * 1.05:
            return {"gpus": sorted(x["index"] for x in chosen), "fits": True, "split": True, "reason": "",
                    "waits": sum(x.get("free_mb") or 0 for x in chosen) < total_mb}
    return {"gpus": [], "fits": False, "reason": text("pick_no_fit", total_mb=total_mb, capacity_mb=capacity)}


def card_tflops(gpu_name: str = "") -> float:
    for fragment, value in GPU_TFLOPS:
        if fragment in gpu_name.lower():
            return value
    return DEFAULT_TFLOPS


def throughput(params: int, method: str, gpu_name: str = "") -> float:
    """Tokens per second the model of ``params`` parameters trains at on the card named ``gpu_name`` (the model above)."""
    if not params:
        return 0.0
    speed = THROUGHPUT_CONSTANT / (params / 1e9) * (card_tflops(gpu_name) / REFERENCE_TFLOPS) * METHOD_SPEED["lora" if method == "lora" else "qlora"]
    return min(speed, MAX_TOKENS_PER_S)


def estimate_time(params: int, tokens_per_epoch: int, epochs: float, method: str, gpu_name: str = "", steps: int = 0) -> dict[str, Any]:
    """Wall-clock estimate from the tokens to process (``tokens_per_epoch`` x ``epochs``) and the throughput model above."""
    method = "lora" if method == "lora" else "qlora"
    tokens = int(tokens_per_epoch * max(epochs, 0.01))
    rate = throughput(params, method, gpu_name)
    params_b = params / 1e9
    startup = STARTUP_S + STARTUP_S_PER_B * params_b
    seconds = startup + (tokens * PADDING / rate if rate else 0.0) + steps * STEP_OVERHEAD_S
    tflops = card_tflops(gpu_name)
    formula = text("time_formula", startup_s=round(startup), tokens=tokens, padding=PADDING, tok_s=int(rate), steps=steps, step_s=STEP_OVERHEAD_S,
                   const=int(THROUGHPUT_CONSTANT), params_b=round(params_b, 3), scale=round(tflops / REFERENCE_TFLOPS, 2),
                   speed=METHOD_SPEED[method], method=method, cap=int(MAX_TOKENS_PER_S))
    return {"seconds": int(seconds), "tokens_per_s": int(rate), "tokens": tokens, "startup_s": round(startup), "steps": steps, "assumed_tflops": tflops,
            "gpu_scale": round(tflops / REFERENCE_TFLOPS, 3), "method_speed": METHOD_SPEED[method], "source": "estimate", "formula": formula,
            "note": text("time_note")}
