"""Write a copy of a Hugging Face model folder whose config asks for a longer context with YaRN rope scaling.

Arguments: ``model_path``, ``output_dir``, ``factor`` (2, 4, 8 or any number above 1), ``original_max_position`` (default: the model's
own ``max_position_embeddings``), ``link`` (``hardlink`` or ``copy``; hard links fall back to a copy when the file system refuses them).

The weights are not touched, so the variant costs no disk space with hard links. YaRN extends the usable window without
training; quality at the far end is something to measure (the long-context suite of the test bench does that).
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def text_section(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("text_config")
    return nested if isinstance(nested, dict) and ("max_position_embeddings" in nested or "hidden_size" in nested) else config


def rope_layer_report(section: dict[str, Any]) -> dict[str, Any]:
    """Which layers use RoPE. Hybrid models mix full attention (RoPE) with linear-attention or state-space layers (no RoPE to scale)."""
    layers = int(section.get("num_hidden_layers") or section.get("n_layer") or 0)
    types = section.get("layer_types")
    interval = section.get("full_attention_interval")
    if isinstance(types, list) and types:
        full = sum(1 for t in types if str(t) in ("full_attention", "attention", "global_attention"))
        sliding = sum(1 for t in types if "sliding" in str(t))
        other = len(types) - full - sliding
        note = ""
        if other:
            note = f"{other} of {len(types)} layers are not full attention: only the {full + sliding} attention layers use RoPE, so only their positions are scaled."
        elif sliding:
            note = f"{sliding} layers use sliding-window attention; YaRN scales their RoPE too, but their window is unchanged."
        return {"layers": len(types), "rope_layers": full + sliding, "full_attention": full, "sliding_attention": sliding, "other": other,
                "hybrid": bool(other), "note": note}
    if isinstance(interval, int) and interval > 1 and layers:
        full = layers // interval
        return {"layers": layers, "rope_layers": full, "full_attention": full, "sliding_attention": 0, "other": layers - full, "hybrid": True,
                "note": f"A hybrid model: one layer in {interval} is full attention ({full} of {layers}); only those layers use RoPE and are scaled."}
    return {"layers": layers, "rope_layers": layers, "full_attention": layers, "sliding_attention": 0, "other": 0, "hybrid": False, "note": ""}


YARN_KEYS = ("type", "rope_type", "factor", "original_max_position_embeddings")


def _rope_targets(params: dict[str, Any]) -> list[dict[str, Any]]:
    """The dicts of ``rope_parameters`` that RoPE settings live in: the dict itself, or (when it is nested by layer type, as
    transformers 5 writes it for models with sliding-window layers) the full-attention entry, else every entry."""
    nested = {k: v for k, v in params.items() if isinstance(v, dict)}
    if nested and len(nested) == len(params):
        return [nested["full_attention"]] if isinstance(nested.get("full_attention"), dict) else list(nested.values())
    return [params]


def apply_yarn(config: dict[str, Any], factor: float, original: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """The config with YaRN scaling set (in the text config when it is nested) and a report of what changed.

    Newer configs (transformers 5) keep every RoPE setting in ``rope_parameters`` (``rope_theta``, ``partial_rotary_factor``,
    ``mrope_section``...). There the YaRN keys are added to that dict and no ``rope_scaling`` is written: transformers takes a
    ``rope_scaling`` key *instead of* ``rope_parameters``, so a partial one would silently drop the theta and the rotary
    fraction and break the model. Older configs get ``rope_scaling`` as before."""
    if factor <= 1:
        raise C.WorkerError("ctx_factor_low")
    new = copy.deepcopy(config)
    section = text_section(new)
    current = section.get("max_position_embeddings")
    existing = section.get("rope_scaling") if isinstance(section.get("rope_scaling"), dict) else {}
    params = section.get("rope_parameters") if isinstance(section.get("rope_parameters"), dict) else None
    targets = _rope_targets(params) if params is not None else []
    previous = next((t.get("original_max_position_embeddings") for t in targets if t.get("rope_type") == "yarn"), None)
    base_ctx = int(original or existing.get("original_max_position_embeddings") or previous or current or 0)
    if base_ctx <= 0:
        raise C.WorkerError("ctx_no_max_position")
    yarn = {"rope_type": "yarn", "factor": float(factor), "original_max_position_embeddings": base_ctx}
    extra = {k: v for k, v in existing.items() if k not in YARN_KEYS}   # e.g. beta_fast / beta_slow tuned by the model's authors
    if targets:
        for target in targets:
            target.update({**extra, **yarn})
        section.pop("rope_scaling", None)
    else:
        section["rope_scaling"] = {**extra, **yarn}
    section["max_position_embeddings"] = int(base_ctx * factor)
    if section is not new and "max_position_embeddings" in new:
        new["max_position_embeddings"] = int(base_ctx * factor)
    report = rope_layer_report(section)
    report.update({"factor": float(factor), "original_max_position_embeddings": base_ctx, "new_max_position_embeddings": int(base_ctx * factor),
                   "nested_text_config": section is not new, "rope_parameters_updated": bool(targets)})
    return new, report


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if mode != "copy":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def extend(args: dict[str, Any]) -> dict[str, Any]:
    src, out = Path(args["model_path"]), Path(args["output_dir"])
    config_path = src / "config.json"
    if not config_path.is_file():
        raise C.WorkerError("config_unreadable", path=src)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    new, report = apply_yarn(config, float(args.get("factor", 4)), int(args["original_max_position"]) if args.get("original_max_position") else None)
    out.mkdir(parents=True, exist_ok=True)
    methods = {"hardlink": 0, "copy": 0}
    for item in sorted(src.iterdir()):
        if item.name == "config.json" or item.name.startswith(".") or not item.is_file():
            continue
        methods[link_or_copy(item, out / item.name, str(args.get("link", "hardlink")))] += 1
    (out / "config.json").write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
    report["files"] = methods
    C.emit("artifact", path=str(out), kind="ctx_variant")
    C.emit("result", data=report)
    return report


def main(args: dict[str, Any]) -> None:
    extend(args)


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
