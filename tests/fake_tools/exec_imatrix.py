import os
import sys
from pathlib import Path

args = sys.argv[1:]
opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2) if args[i].startswith("-")}
if os.environ.get("FAKE_IMATRIX_FAIL"):
    print("CUDA error: out of memory", flush=True)
    sys.exit(1)
chunks = int(opt.get("--chunks", 4))
print(f"compute_imatrix: computing over {chunks} chunks, n_ctx=512, batch_size=2048, n_seq=4", flush=True)
vals = []
for i in range(1, chunks + 1):
    vals.append(f"[{i}]{6.0 - 0.1 * i:.4f}")
    print(",".join(vals) + ",", flush=True)
text = Path(opt["-f"]).read_text(encoding="utf-8")
Path(opt["-o"]).write_text(f"IMATRIX cuda={os.environ.get('CUDA_VISIBLE_DEVICES')} chars={len(text)} model={Path(opt['-m']).name}\n", encoding="utf-8")
print("save_imatrix: stored collected data", flush=True)
