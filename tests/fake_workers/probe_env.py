"""Stand-in for workers/probe_env.py: a healthy trainer environment with two GPUs."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pygmalion_hoard" / "workers"))
import _common as C  # noqa: E402


def main(args):
    libs = {k: "1.0" for k in ("torch", "transformers", "peft", "trl", "accelerate", "datasets", "bitsandbytes", "safetensors", "huggingface_hub", "gguf", "numpy")}
    libs["transformers"] = "5.1.0"
    C.emit("result", data={"python": "3.13", "executable": sys.executable, "libs": libs, "problems": [], "bitsandbytes": {"installed": True, "ok": True},
                           "torch": {"installed": True, "version": "2.9.0+cu128", "cuda_available": True,
                                     "devices": [{"index": 0, "name": "Fake GPU", "total_mb": 16000, "capability": "12.0", "bf16": True}]},
                           "disk": {"path": str(args.get("work_dir")), "free_gb": 500.0}, "trainable_architectures": ["FakeForCausalLM", "LlamaForCausalLM"]})


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
