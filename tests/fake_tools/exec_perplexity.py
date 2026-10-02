import os
import sys

args = sys.argv[1:]
opt = {args[i]: args[i + 1] for i in range(0, len(args) - 1, 2) if args[i].startswith("-")}
chunks = int(opt.get("--chunks", 3))
for i in range(1, chunks + 1):
    print(f"[{i}]{7.5 - 0.05 * i:.4f}", flush=True)
print(f"Final estimate: PPL = {os.environ.get('FAKE_PPL', '7.1234')} +/- 0.04321", flush=True)
