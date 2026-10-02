"""Report what the trainer environment can do: versions, CUDA, device names, bitsandbytes, free disk. Optionally check one model.

Arguments (all optional): ``work_dir`` (free disk is measured there), ``model`` (a folder: can transformers load its architecture?).
"""

from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

LIBS = ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes", "safetensors", "huggingface_hub", "gguf", "numpy")
MINIMUM = {"transformers": 5}


def version_of(name: str) -> str | None:
    try:
        return md.version(name.replace("_", "-"))
    except md.PackageNotFoundError:
        try:
            return md.version(name)
        except md.PackageNotFoundError:
            return None


def _first(*attempts):
    """The first of several ways to read a value that does not raise (drivers and torch builds differ), else None."""
    for attempt in attempts:
        try:
            value = attempt()
        except Exception:  # noqa: BLE001
            continue
        if value is not None:
            return value
    return None


def list_devices(torch: Any) -> list[dict[str, Any]]:
    """Every CUDA device torch can see, as ``{index, name, total_mb, capability, bf16}``. One device that cannot be read in full still
    appears (with what could be read and an ``error``): the list is never shorter than ``torch.cuda.device_count()``."""
    devices: list[dict[str, Any]] = []
    for index in range(int(torch.cuda.device_count())):
        props = _first(lambda: torch.cuda.get_device_properties(index))
        name = _first(lambda: props.name, lambda: torch.cuda.get_device_name(index))
        total = _first(lambda: props.total_memory, lambda: torch.cuda.mem_get_info(index)[1])
        capability = _first(lambda: f"{props.major}.{props.minor}", lambda: "%d.%d" % tuple(torch.cuda.get_device_capability(index)))
        bf16 = _first(lambda: int(str(capability).split(".")[0]) >= 8)          # bfloat16 in hardware from Ampere (compute capability 8.0)
        entry: dict[str, Any] = {"index": index, "name": str(name) if name is not None else None,
                                 "total_mb": int(total / 1048576) if total is not None else None, "capability": capability, "bf16": bf16}
        if name is None or total is None:
            entry["error"] = "the driver did not report the name or the memory of this device"
        devices.append(entry)
    return devices


def probe(args: dict[str, Any]) -> dict[str, Any]:
    # Same numbering as nvidia-smi and the hub (PCI bus order); it must be set before CUDA starts, i.e. before torch is used.
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    info: dict[str, Any] = {"python": platform.python_version(), "platform": platform.platform(), "executable": sys.executable, "libs": {}}
    for lib in LIBS:
        info["libs"][lib] = version_of(lib)
    torch_info: dict[str, Any] = {"installed": info["libs"]["torch"] is not None, "cuda_available": False, "devices": []}
    if torch_info["installed"]:
        try:
            torch = importlib.import_module("torch")
            torch_info["version"] = torch.__version__
            torch_info["cuda_build"] = getattr(torch.version, "cuda", None)
            torch_info["cuda_available"] = bool(torch.cuda.is_available())
            if torch_info["cuda_available"]:
                torch_info["devices"] = list_devices(torch)
                torch_info["device_count"] = len(torch_info["devices"])
        except Exception as exc:  # noqa: BLE001
            torch_info["error"] = f"{type(exc).__name__}: {exc}"
    info["torch"] = torch_info
    bnb: dict[str, Any] = {"installed": info["libs"]["bitsandbytes"] is not None, "ok": False}
    if bnb["installed"]:
        try:
            importlib.import_module("bitsandbytes")
            bnb["ok"] = True
        except Exception as exc:  # noqa: BLE001
            bnb["error"] = f"{type(exc).__name__}: {str(exc)[:200]}"
    info["bitsandbytes"] = bnb
    problems = []
    for lib in LIBS[:-1]:
        if info["libs"][lib] is None:
            problems.append({"lib": lib, "problem": "missing", "key": "env_lib_missing"})
    tv = info["libs"]["transformers"]
    if tv and int(tv.split(".")[0]) < MINIMUM["transformers"]:
        problems.append({"lib": "transformers", "problem": f"version {tv} is older than {MINIMUM['transformers']}", "key": "env_lib_old",
                         "params": {"version": tv, "minimum": MINIMUM["transformers"]}})
    if torch_info["installed"] and not torch_info["cuda_available"]:
        problems.append({"lib": "torch", "problem": "no CUDA device is visible (CPU-only build, or no driver)", "key": "env_no_cuda"})
    if bnb["installed"] and not bnb["ok"]:
        problems.append({"lib": "bitsandbytes", "problem": bnb.get("error", "does not import")})
    info["problems"] = problems
    info["trainable_architectures"] = trainable_architectures() if info["libs"]["transformers"] else []
    work = Path(args.get("work_dir") or ".")
    try:
        target = work if work.exists() else work.parent
        usage = shutil.disk_usage(target)
        info["disk"] = {"path": str(work), "free_gb": round(usage.free / 1e9, 1), "total_gb": round(usage.total / 1e9, 1)}
    except OSError:
        info["disk"] = {"path": str(work), "free_gb": None}
    if args.get("model"):
        info["model"] = check_model(str(args["model"]))
    return info


def _class_names(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    return [str(x) for x in value] if isinstance(value, (list, tuple)) else []


def causal_class_names(causal: dict[str, Any], image_text: dict[str, Any]) -> set[str]:
    """Class names the trainer can load with ``AutoModelForCausalLM``: the causal-LM table, plus the multimodal class of a model
    type that table also knows (its checkpoints carry ``...ForConditionalGeneration`` and load through the text model). A
    multimodal type the causal table does not know is not trainable here, however its class is named."""
    names: set[str] = set()
    for value in causal.values():
        names.update(_class_names(value))
    for model_type, value in image_text.items():
        if model_type in causal:
            names.update(_class_names(value))
    return names


def trainable_architectures() -> list[str]:
    """Class names transformers can load as a causal language model (``architectures`` in a config.json is one of these)."""
    try:
        from transformers.models.auto import modeling_auto as auto
        causal = dict(getattr(auto, "MODEL_FOR_CAUSAL_LM_MAPPING_NAMES", {}) or {})
        image_text = dict(getattr(auto, "MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES", {}) or {})
    except Exception:  # noqa: BLE001 — the table moves between versions; the per-model check is the fallback
        return []
    return sorted(causal_class_names(causal, image_text))


def check_model(path: str) -> dict[str, Any]:
    """Can this transformers version load the folder's architecture as a causal language model?"""
    out: dict[str, Any] = {"path": path, "trainable": False}
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(path, local_files_only=True)
        out["model_type"] = config.model_type
        out["architectures"] = list(getattr(config, "architectures", None) or [])
        try:
            from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING, MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
            known = type(config) in MODEL_FOR_CAUSAL_LM_MAPPING
            out["trainable"] = bool(known)
            out["image_text_only"] = (not known) and type(config) in MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING
        except Exception:  # noqa: BLE001 — the mapping names move between versions
            out["trainable"] = None
            out["note"] = "Could not consult the architecture table of this transformers version."
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    return out


def main(args: dict[str, Any]) -> None:
    data = probe(args)
    C.emit("result", data=data)


if __name__ == "__main__":
    sys.exit(C.run_worker(main) if "--args" in sys.argv else (print(json.dumps(probe({}), indent=2)) or 0))
