"""Stand-in for workers/train_lora.py: same arguments and the same JSON-lines protocol, no torch.

Behaviour is steered by environment variables so a test does not need a different arguments file:
FAKE_STEPS (default: max_steps or 12), FAKE_DELAY seconds per step (default 0.01), FAKE_FAIL_AT step that raises "CUDA out of memory"."""

import json
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pygmalion_hoard" / "workers"))
import _common as C  # noqa: E402
import _stio as S  # noqa: E402


def save_checkpoint(root: Path, step: int) -> None:
    folder = root / f"checkpoint-{step}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "training_state.json").write_text(json.dumps({"step": step}), encoding="utf-8")
    C.prune_checkpoints(root, 2)


def main(args):
    out = Path(args["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    root = out / "checkpoints"
    total = int(os.environ.get("FAKE_STEPS") or args.get("max_steps") or 12)
    delay = float(os.environ.get("FAKE_DELAY", "0.01"))
    fail_at = int(os.environ.get("FAKE_FAIL_AT", "0"))
    save_every = int(args.get("save_every") or 4)
    start = 0
    if args.get("resume"):
        steps = C.checkpoint_steps(root)
        if steps:
            start = steps[-1]
            C.log(f"Resumed from checkpoint-{start}.")
    (out / "env.json").write_text(json.dumps({"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"), "args": args, "start": start}), encoding="utf-8")
    t0 = time.time()
    for step in range(start + 1, total + 1):
        if C.STOP.is_set():
            save_checkpoint(root, step - 1 if step > 1 else 1)
            C.log("Stop requested: checkpoint saved.")
            return
        time.sleep(delay)
        loss = round(2.0 / (1 + 0.2 * step), 4)
        C.emit("progress", step=step, total=total, loss=loss, lr=args.get("lr"), tokens_per_s=1500, avg_tokens_per_s=1450, eta_s=C.eta_seconds(step - start, total - start, time.time() - t0),
               gpu_mem_mb=9000)
        if step % 5 == 0:
            C.emit("eval", step=step, eval_loss=round(loss + 0.1, 4))
        if fail_at and step == fail_at:
            raise RuntimeError("CUDA out of memory (simulated)")
        if step % save_every == 0 and step < total:
            save_checkpoint(root, step)
    rank = int(args.get("rank", 8))
    (out / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": int(args.get("alpha", 16)), "peft_type": "LORA", "target_modules": ["q_proj"]}), encoding="utf-8")
    name = "base_model.model.model.layers.0.self_attn.q_proj"
    import numpy as np
    arrays = {f"{name}.lora_A.weight": np.full((rank, 8), 0.01, "<f4"), f"{name}.lora_B.weight": np.full((8, rank), 0.02, "<f4")}
    entries = [(n, "F32", list(a.shape)) for n, a in arrays.items()]
    S.write_shard(out / "adapter_model.safetensors", entries, lambda n: S.from_float32(arrays[n], "F32"))
    shutil.rmtree(root, ignore_errors=True)
    summary = {"steps": total, "final_loss": loss, "best_loss": loss, "final_eval_loss": loss + 0.1, "best_eval_loss": loss + 0.1, "tokens_seen": total * 1500,
               "time_s": round(time.time() - t0, 2), "peak_gpu_mb": 9000, "method": args.get("method")}
    (out / "training_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    C.emit("artifact", path=str(out), kind="adapter")
    C.emit("result", data=summary)


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
