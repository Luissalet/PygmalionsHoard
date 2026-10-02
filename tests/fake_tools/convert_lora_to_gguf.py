"""Fake LoRA conversion script (also registers @ModelBase.register("FakeForCausalLM"))."""

import argparse
import sys
from pathlib import Path

from gguf_builder import build_gguf  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--base", required=True)
parser.add_argument("adapter")
parser.add_argument("--outfile", required=True)
parser.add_argument("--outtype", default="f16")
ns = parser.parse_args()
print("INFO:lora-to-gguf:converting adapter", flush=True)
build_gguf(ns.outfile, arch="llama", size=2048, name="adapter")
