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
    try:
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
    except Exception:  # noqa: BLE001 — a tokenizer without a readable vocabulary is read through encode() only
        vocab = {}
    for label in labels:
        ids: list[int] = []
        for spelling in (label, " " + label):
            encoded = tokenizer.encode(spelling, add_special_tokens=False)
            if len(encoded) == 1 and int(encoded[0]) not in ids:
                ids.append(int(encoded[0]))
        # sentencepiece encode() adds the word-start mark, so the bare token (and the byte-level spellings) are looked up in the vocabulary
        for piece in (label, "▁" + label, "Ġ" + label):
            token_id = vocab.get(piece) if isinstance(vocab, dict) else None
            if token_id is not None and int(token_id) not in ids:
                ids.append(int(token_id))
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


def _templated(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _without_system(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    """The system text folded into the first user message, for chat templates that refuse a system role."""
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    rest = [dict(m) for m in messages if m["role"] != "system"]
    if system and rest and rest[0]["role"] == "user":
        rest[0]["content"] = f"{system}\n\n{rest[0]['content']}"
    elif system:
        rest.insert(0, {"role": "user", "content": system})
    return rest


def prompt_text(tokenizer: Any, messages: list[dict[str, str]]) -> str:
    if getattr(tokenizer, "chat_template", None):
        try:
            return _templated(tokenizer, messages)
        except Exception:  # noqa: BLE001 — e.g. a template that raises on the system role
            if not any(m["role"] == "system" for m in messages):
                raise
            return _templated(tokenizer, _without_system(messages))
    # no template: the text is tokenised without special tokens, so the BOS a model expects is written here
    bos = getattr(tokenizer, "bos_token", None) if getattr(tokenizer, "add_bos_token", False) else None
    return (bos or "") + "\n\n".join(m["content"] for m in messages) + "\n\nAnswer:"


def forward_last(model: Any, enc: Any) -> Any:
    """The forward pass asking only for the logits of the last position (a [batch, length, vocabulary] tensor of a long prompt does not fit);
    older transformers versions that do not know the argument get the plain call."""
    for kwargs in ({"logits_to_keep": 1}, {"num_logits_to_keep": 1}):
        try:
            return model(**enc, **kwargs)
        except TypeError as exc:
            if next(iter(kwargs)) not in str(exc):
                raise
    return model(**enc)


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
            logits = forward_last(self.model, enc).logits[:, -1, :].float()
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
