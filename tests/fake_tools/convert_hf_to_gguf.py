"""Fake conversion script: writes a small valid GGUF and prints the same tensor lines the real one does.

# Architectures it claims to know, in the same decorator form the real script uses (convert_detect reads them from the text):
# @ModelBase.register("FakeForCausalLM", "LlamaForCausalLM")
"""

import argparse
import json
import os
import struct
import sys
from pathlib import Path

from gguf_builder import build_gguf  # noqa: E402


def tensor_names(folder: Path):
    names = []
    for f in sorted(folder.glob("*.safetensors")):
        with open(f, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        names += [(k, v["shape"]) for k, v in header.items() if k != "__metadata__"]
    return names


parser = argparse.ArgumentParser()
parser.add_argument("model")
parser.add_argument("--outfile", required=True)
parser.add_argument("--outtype", default="f16")
ns = parser.parse_args()
if os.environ.get("FAKE_CONVERT_FAIL"):
    print("ERROR:hf-to-gguf:Model FakeForCausalLM is not supported", flush=True)
    sys.exit(1)
folder = Path(ns.model)
config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
for name, shape in tensor_names(folder):
    print(f"INFO:hf-to-gguf:{name},\ttorch.float32 --> F16, shape = {{{', '.join(map(str, shape))}}}", flush=True)
rope = None
scaling = config.get("rope_scaling")
if scaling and scaling.get("rope_type") == "yarn":
    rope = {"scaling.type": "yarn", "scaling.factor": float(scaling["factor"]), "scaling.original_context_length": int(scaling["original_max_position_embeddings"])}
file_type = {"f16": 1, "bf16": 32, "q8_0": 7}[ns.outtype]
size = int(sum(f.stat().st_size for f in folder.glob("*.safetensors")) / 2) + 4096
build_gguf(ns.outfile, arch="llama", context=int(config["max_position_embeddings"]), blocks=int(config["num_hidden_layers"]), heads=int(config["num_attention_heads"]),
           kv_heads=int(config["num_key_value_heads"]), embedding=int(config["hidden_size"]), file_type=file_type, rope=rope, size=size)
print(f"INFO:hf-to-gguf:Model successfully exported to {ns.outfile}", flush=True)
