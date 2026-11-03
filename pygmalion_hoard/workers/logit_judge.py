"""Read the probability a Hugging Face model (with or without a LoRA adapter) gives to each option label as the next token.

Arguments (args.json): ``base_path``, ``adapter_path`` (optional), ``variants`` (``["base"]``, ``["adapter"]`` or both, in the order to report),
``requests`` = ``[{key, messages, labels}]``, ``device`` (``auto`` | ``cpu``), ``batch`` (default 8).

Result: ``{variant: {key: {label: log-probability over the whole vocabulary}}}``. The chat template of the model turns ``messages`` into the
prompt (with the generation prompt appended), the forward pass runs once per batch of prompts and only the last position is read.
A label is the sum, in probability, of its single-token spellings (``A`` and `` A``). The adapter is applied with PEFT from its safetensors
file only; a pickle file is refused. With both variants the base model is loaded once and the adapter is switched off for the base pass.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402


def candidate_ids(tokenizer: Any, labels: list[str]) -> dict[str, list[int]]:
    """For each label the ids of its single-token spellings. A label that is never one token cannot be read with one forward pass."""
    out: dict[str, list[int]] = {}
    for label in labels:
        ids: list[int] = []
        for spelling in (label, " " + label):
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) == 1 and encoded[0] not in ids:
                ids.append(int(encoded[0]))
        if not ids:
            raise C.WorkerError(f"The label '{label}' is not a single token for this tokenizer.", "Use another model or fewer options.")
        out[label] = ids
    return out


def fold(ids_by_label: dict[str, list[int]], logprob_by_id: dict[int, float]) -> dict[str, Optional[float]]:
    """Sum the probability of each label's token ids and return its log-probability."""
    out: dict[str, Optional[float]] = {}
    for label, ids in ids_by_label.items():
        values = [logprob_by_id[i] for i in ids if i in logprob_by_id]
        if not values:
            out[label] = None
            continue
        top = max(values)
        out[label] = top + math.log(sum(math.exp(v - top) for v in values))
    return out


def prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return "\n\n".join(m["content"] for m in messages) + "\n\nAnswer:"


def run_requests(engine: Any, args: dict[str, Any]) -> dict[str, Any]:
    """The job, with the model behind ``engine`` (``tokenizer``, ``variant(name)`` context manager, ``last_logprobs(texts, ids)``)."""
    requests = args.get("requests") or []
    if not requests:
        raise C.WorkerError("No questions were given.")
    all_labels = sorted({label for r in requests for label in r["labels"]})
    ids_by_label = candidate_ids(engine.tokenizer, all_labels)
    wanted_ids = sorted({i for ids in ids_by_label.values() for i in ids})
    texts = [prompt_text(engine.tokenizer, r["messages"]) for r in requests]
    batch = max(1, int(args.get("batch") or 8))
    result: dict[str, Any] = {}
    variants = args.get("variants") or ["base"]
    for number, variant in enumerate(variants):
        rows: list[dict[int, float]] = []
        with engine.variant(variant):
            for start in range(0, len(texts), batch):
                if C.STOP.is_set():
                    raise KeyboardInterrupt
                rows += engine.last_logprobs(texts[start:start + batch], wanted_ids)
                C.emit("progress", step=number * len(texts) + min(len(texts), start + batch), total=len(texts) * len(variants))
        result[variant] = {r["key"]: fold({lb: ids_by_label[lb] for lb in r["labels"]}, row) for r, row in zip(requests, rows)}
    return result


class TorchEngine:
    def __init__(self, base_path: str, adapter_path: str, device: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(base_path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda" if (device != "cpu" and torch.cuda.is_available()) else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        try:
            model = AutoModelForCausalLM.from_pretrained(base_path, dtype=dtype)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(base_path, torch_dtype=dtype)
        self.has_adapter = bool(adapter_path)
        if adapter_path:
            adapter = Path(adapter_path)
            if not (adapter / "adapter_model.safetensors").is_file():
                raise C.WorkerError("The adapter has no adapter_model.safetensors; pickle files are not read.", "Re-save the adapter as safetensors.")
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, str(adapter), use_safetensors=True)
        self.model = model.to(self.device).eval()
        C.log(f"Loaded {Path(base_path).name} on {self.device}" + (f" with adapter {Path(adapter_path).name}" if adapter_path else ""))

    def variant(self, name: str):
        import contextlib
        if name == "base" and self.has_adapter:
            return self.model.disable_adapter()
        return contextlib.nullcontext()

    def last_logprobs(self, texts: list[str], ids: list[int]) -> list[dict[int, float]]:
        torch = self.torch
        enc = self.tokenizer(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        with torch.inference_mode():
            logits = self.model(**enc).logits[:, -1, :].float()
            lse = torch.logsumexp(logits, dim=-1)
            picked = logits[:, ids] - lse[:, None]
        return [dict(zip(ids, row)) for row in picked.cpu().tolist()]


def main(args: dict[str, Any], engine: Any = None) -> None:
    variants = args.get("variants") or ["base"]
    if "adapter" in variants and not args.get("adapter_path"):
        raise C.WorkerError("The adapter variant needs adapter_path.")
    if engine is None:
        engine = TorchEngine(str(args.get("base_path") or ""), str(args.get("adapter_path") or "") if "adapter" in variants else "", str(args.get("device") or "auto"))
    elif not hasattr(engine, "variant"):
        raise C.WorkerError("Bad engine.")
    C.emit("result", data=run_requests(engine, args))


if __name__ == "__main__":
    sys.exit(C.run_worker(main))
