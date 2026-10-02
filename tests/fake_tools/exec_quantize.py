import os
import sys
from pathlib import Path

argv = sys.argv[1:]
matrix = None
if argv and argv[0] == "--imatrix":
    matrix, argv = argv[1], argv[2:]
src, dst, qtype = Path(argv[0]), Path(argv[1]), argv[2]
if os.environ.get("FAKE_QUANT_FAIL"):
    print("llama_model_quantize: failed to quantize", flush=True)
    sys.exit(1)
ratio = {"Q8_0": 0.55, "Q6_K": 0.42, "Q5_K_M": 0.36, "Q4_K_M": 0.3, "IQ4_XS": 0.27, "Q3_K_M": 0.24, "IQ3_M": 0.22}[qtype]
data = src.read_bytes()
for i in range(1, 5):
    print(f"[{i:4d}/{4:4d}]            blk.{i}.attn_q.weight - [  256,   256,     1,     1], type =    f16, converting to {qtype.lower()} .. size =     0.12 MiB ->     0.04 MiB", flush=True)
dst.write_bytes(data[: max(int(len(data) * ratio), 3500)] + (b"imatrix" if matrix else b""))
print(f"llama_model_quantize_internal: model size  = {len(data) / 1048576:8.2f} MB", flush=True)
print(f"llama_model_quantize_internal: quant size  = {dst.stat().st_size / 1048576:8.2f} MB", flush=True)
