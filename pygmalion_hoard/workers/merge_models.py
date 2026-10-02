"""Merge models of the same architecture tensor by tensor, streaming over the safetensors shards.

Arguments (args.json): ``models`` (folders), ``output_dir``, ``method`` (linear | slerp | ties | dare), ``weights``, ``base`` (folder; ties and
dare), ``t`` and ``t_map`` (slerp: a number, and optional ``{regex: t}`` overrides by tensor name), ``density``, ``lam``, ``seed``,
``consensus`` (dare: linear | ties), ``normalize``, ``shard_gb`` (default 4).

Never loads a whole model: each output tensor is read from every input, merged by ``merge_math`` in float32 and cast back to the first
model's dtype before the next one is read. Non-floating tensors (indices, masks) are copied from the first model.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402
import _stio as S  # noqa: E402


def slerp_t(name: str, t: float, t_map: dict[str, float]) -> float:
    for pattern, value in (t_map or {}).items():
        if re.search(pattern, name):
            return float(value)
    return float(t)


def merge(args: dict[str, Any], progress=lambda **kw: None) -> dict[str, Any]:
    math = C.package_module("merge_math")
    models = [str(m) for m in args["models"]]
    method = args.get("method", "linear")
    out = Path(args["output_dir"])
    base_dir = args.get("base")
    if method not in math.METHODS:
        raise C.WorkerError("merge_method_unknown", method=method, options=list(math.METHODS))
    if method == "slerp" and len(models) != 2:
        raise C.WorkerError("merge_slerp_two")
    if method in ("ties", "dare") and not base_dir:
        raise C.WorkerError("merge_needs_base", method=method)
    if len(models) < 2:
        raise C.WorkerError("merge_needs_two")

    check = S.check_compatible([*models, *([base_dir] if base_dir else [])])
    if not check["ok"]:
        raise C.WorkerError("models_incompatible", problems="; ".join(check["problems"][:3]))
    for note in check["dtype_differences"]:
        C.log("dtype difference: " + note)
    tensors = [S.model_tensors(m) for m in models]
    base_tensors = S.model_tensors(base_dir) if base_dir else None
    first = tensors[0]
    weights = args.get("weights") or None
    entries = [(name, info["dtype"], list(info["shape"])) for name, info in first.items()]
    out.mkdir(parents=True, exist_ok=True)
    shards = S.plan_shards(entries, int(float(args.get("shard_gb", 4)) * (1 << 30)))
    total = len(entries)
    done = 0
    started = time.monotonic()
    weight_map: dict[str, str] = {}
    total_size = sum(S.tensor_nbytes({"dtype": d, "shape": s}) for _, d, s in entries)

    def produce(name: str) -> bytes:
        nonlocal done
        if C.STOP.is_set():
            raise C.WorkerError("merge_stopped")
        info = first[name]
        if info["dtype"] not in S.FLOAT_DTYPES:
            data = S.read_raw(info["file"], info, info["data_start"])
        else:
            arrays = [S.read_tensor(t[name]) for t in tensors]
            base = S.read_tensor(base_tensors[name]) if base_tensors else None
            merged = math.merge_tensor(method, arrays, base=base, weights=weights, t=slerp_t(name, args.get("t", 0.5), args.get("t_map") or {}),
                                       density=float(args.get("density", 0.5)), lam=float(args.get("lam", 1.0)), seed=int(args.get("seed", 0)),
                                       name=name, consensus=args.get("consensus", "linear"), normalize=bool(args.get("normalize", True)))
            data = S.from_float32(merged, info["dtype"])
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
        S.write_index(out, weight_map, total_size)
    copied = S.copy_non_weights(models[0], out)
    summary = {"method": method, "models": models, "base": base_dir, "tensors": total, "shards": len(shards), "bytes": total_size, "copied": copied,
               "params": {k: args.get(k) for k in ("weights", "t", "t_map", "density", "lam", "seed", "consensus", "normalize")}}
    C.emit("artifact", path=str(out), kind="merged")
    C.emit("result", data=summary)
    return summary


def main(args: dict[str, Any]) -> None:
    merge(args)


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
