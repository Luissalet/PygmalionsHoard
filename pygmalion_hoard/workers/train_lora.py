"""LoRA / QLoRA fine-tuning of a causal language model with a plain training loop.

Why a loop and not a trainer class: the loss mask, the batching, the checkpoint format and the resume rule are the parts that matter
here, and they are short to write and independent of the trainer libraries' API changes. The libraries used are the stable core:
``AutoModelForCausalLM``, ``BitsAndBytesConfig``, ``peft.LoraConfig`` / ``get_peft_model`` and ``torch`` itself.

Arguments (``args.json``):

    base_path        folder of the base model (Hugging Face format)
    output_dir       where the adapter and ``training_summary.json`` are written
    train_path       JSONL of the training split (chat, instruction or text records, mixed is fine)
    eval_path        JSONL of the evaluation split (optional)
    method           ``qlora`` (4-bit NF4, double quantisation, bf16 compute) or ``lora`` (bf16 weights)
    rank, alpha, dropout, target_modules (list of last names, or ["all"]), lr, weight_decay, max_grad_norm, warmup
    epochs, max_steps (0 = by epochs), batch_size, grad_accum, seq_len, eval_every, eval_max_examples, save_every
    train_on         ``assistant`` (every assistant turn) or ``last`` (only the final one)
    gradient_checkpointing, seed, resume (bool), trust_remote_code (default false), keep_checkpoints (default false)

Loss is computed on assistant tokens only for chat records, on the response only for prompt/response records without a chat
template, and on every token for raw text (packed to ``seq_len``). One optimizer step is ``grad_accum`` micro-batches of
``batch_size`` examples and the loss of a step is averaged over its labelled tokens, so the batch layout does not change the maths.
"""

from __future__ import annotations

import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

IGNORE = -100


# ------------------------------------------------------------------ pure part (no torch): records -> examples -> steps -> batches
def record_kind(record: dict[str, Any], has_chat_template: bool) -> str:
    """``chat``, ``completion`` (prompt/response without a template) or ``text``; ``""`` when the record is none of them."""
    if isinstance(record.get("messages"), list) and record["messages"]:
        return "chat"
    if "prompt" in record and "response" in record:
        return "chat" if has_chat_template else "completion"
    if isinstance(record.get("text"), str) and record["text"].strip():
        return "text"
    return ""


def read_jsonl(path: Optional[str]) -> list[dict[str, Any]]:
    if not path:
        return []
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


class Tokenization:
    """The three functions the label mask needs, built from any object that behaves like a Hugging Face tokenizer."""

    def __init__(self, tok: Any):
        self.tok = tok
        self.has_template = bool(getattr(tok, "chat_template", None))
        self.eos_id = getattr(tok, "eos_token_id", None)
        pad = getattr(tok, "pad_token_id", None)
        self.pad_id = pad if pad is not None else (self.eos_id if self.eos_id is not None else 0)

    def render(self, messages: list[dict[str, Any]], add_generation_prompt: bool) -> str:
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)

    def encode(self, text: str) -> tuple[list[int], Optional[list[tuple[int, int]]]]:
        try:
            enc = self.tok(text, add_special_tokens=False, return_offsets_mapping=True)
            return list(enc["input_ids"]), [tuple(o) for o in enc["offset_mapping"]]
        except (NotImplementedError, ValueError, TypeError, KeyError):
            return list(self.tok(text, add_special_tokens=False)["input_ids"]), None

    def plain(self, text: str) -> list[int]:
        return list(self.tok(text, add_special_tokens=False)["input_ids"])


def build_examples(records: Sequence[dict[str, Any]], tk: Tokenization, seq_len: int, train_on: str = "assistant",
                   chatmask: Any = None) -> dict[str, Any]:
    """Tokenised examples ``{"input_ids", "labels"}`` plus counts. ``chatmask`` is the pure module (loaded by path in the worker)."""
    cm = chatmask or C.package_module("chatmask")
    examples: list[dict[str, Any]] = []
    texts: list[list[int]] = []
    skipped = {"empty": 0, "no_label": 0, "error": 0}
    first_error = ""
    modes: dict[str, int] = {}
    for record in records:
        kind = record_kind(record, tk.has_template)
        if not kind:
            skipped["empty"] += 1
            continue
        try:
            if kind == "chat":
                messages = record["messages"] if "messages" in record else cm.instruction_to_messages(record)
                built = cm.build_chat_example(messages, tk.render, tk.encode, max_len=seq_len, train_on=train_on)
            elif kind == "completion":
                built = cm.build_completion_example(str(record["prompt"]) + "\n\n", str(record["response"]), tk.plain, tk.eos_id, max_len=seq_len)
            else:
                texts.append(tk.plain(record["text"]))
                continue
        except Exception as exc:  # noqa: BLE001 — a template that rejects one conversation must not stop the whole run
            skipped["error"] += 1
            first_error = first_error or f"{type(exc).__name__}: {exc}"
            continue
        if built["trainable"] <= 0:
            skipped["no_label"] += 1
            continue
        modes[built["mode"]] = modes.get(built["mode"], 0) + 1
        examples.append({"input_ids": built["input_ids"], "labels": built["labels"]})
    packed = cm.pack_sequences(texts, seq_len, tk.eos_id)
    for chunk in packed:
        examples.append({"input_ids": chunk, "labels": list(chunk)})
    if packed:
        modes["packed"] = len(packed)
    tokens = sum(len(e["input_ids"]) for e in examples)
    labelled = sum(1 for e in examples for x in e["labels"] if x != IGNORE)
    return {"examples": examples, "skipped": skipped, "first_error": first_error, "modes": modes, "tokens": tokens, "labelled_tokens": labelled}


def plan_steps(n_examples: int, batch_size: int, grad_accum: int, epochs: float, max_steps: int) -> tuple[int, int]:
    """``(total optimizer steps, steps per epoch)``."""
    per_step = max(1, batch_size * grad_accum)
    per_epoch = max(1, math.ceil(n_examples / per_step))
    if max_steps and max_steps > 0:
        return int(max_steps), per_epoch
    return max(1, int(math.ceil(per_epoch * max(epochs, 0.01)))), per_epoch


def step_indices(step: int, n_examples: int, batch_size: int, grad_accum: int, per_epoch: int, seed: int) -> list[list[int]]:
    """The micro-batches (lists of example indices) of optimizer step ``step``: a pure function of the step, so a resume is exact."""
    per_step = batch_size * grad_accum
    epoch, within = divmod(step, per_epoch)
    order = C.epoch_order(n_examples, epoch, seed)
    chosen = order[within * per_step:(within + 1) * per_step]
    return [chosen[i:i + batch_size] for i in range(0, len(chosen), batch_size) if chosen[i:i + batch_size]]


def collate(examples: Sequence[dict[str, Any]], pad_id: int, multiple: int = 8) -> dict[str, list[list[int]]]:
    """Right-pad to the longest example (rounded up to ``multiple``). Pure lists; the loop turns them into tensors."""
    longest = max(len(e["input_ids"]) for e in examples)
    width = ((longest + multiple - 1) // multiple) * multiple
    ids, labels, mask = [], [], []
    for e in examples:
        pad = width - len(e["input_ids"])
        ids.append(list(e["input_ids"]) + [pad_id] * pad)
        labels.append(list(e["labels"]) + [IGNORE] * pad)
        mask.append([1] * len(e["input_ids"]) + [0] * pad)
    return {"input_ids": ids, "labels": labels, "attention_mask": mask}


def shifted_label_count(labels: Sequence[Sequence[int]]) -> int:
    """How many positions the shifted loss will score (labels from the second token on)."""
    return sum(1 for row in labels for x in row[1:] if x != IGNORE)


def scored_in_step(examples: Sequence[dict[str, Any]], micro: Sequence[Sequence[int]]) -> int:
    """Labelled positions in all the micro-batches of one optimizer step: the divisor that makes the step loss a per-token mean."""
    return shifted_label_count([examples[i]["labels"] for chunk in micro for i in chunk])


# ------------------------------------------------------------------ checkpoints
def latest_checkpoint(root: Path) -> Optional[Path]:
    steps = C.checkpoint_steps(root)
    return root / f"checkpoint-{steps[-1]}" if steps else None


def write_state(path: Path, state: dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ torch part
def load_model(args: dict[str, Any], torch: Any):
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    kwargs: dict[str, Any] = {"trust_remote_code": bool(args.get("trust_remote_code", False))}
    if args.get("attn_implementation"):
        kwargs["attn_implementation"] = args["attn_implementation"]
    visible = torch.cuda.device_count()
    if visible == 0:
        raise C.WorkerError("train_no_cuda")
    kwargs["device_map"] = {"": 0} if visible == 1 else "auto"
    if args.get("method", "qlora") == "qlora":
        try:
            import bitsandbytes  # noqa: F401
        except ImportError as exc:
            raise C.WorkerError("bnb_missing") from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    try:
        return AutoModelForCausalLM.from_pretrained(args["base_path"], dtype=torch.bfloat16, **kwargs)
    except TypeError as exc:
        if "dtype" not in str(exc):
            raise
        return AutoModelForCausalLM.from_pretrained(args["base_path"], torch_dtype=torch.bfloat16, **kwargs)  # transformers 4


def linear_module_names(model: Any, torch: Any) -> list[str]:
    return [name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)]


def prepare_for_training(model: Any, checkpointing: bool) -> Any:
    """Freeze the base model and switch on gradient checkpointing (non-reentrant).

    Deliberately not ``peft.prepare_model_for_kbit_training``: besides this it casts every bf16 parameter that is not 4-bit to
    float32, and on a 4-bit model those are the embeddings and the language-model head. With a 150k-250k vocabulary that is
    several GB more on a 16 GB card for no gain here (the forward runs under bf16 autocast and the loss is taken in float32)."""
    for _name, param in model.named_parameters():
        param.requires_grad = False
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    model.config.use_cache = False
    return model


def attach_lora(model: Any, args: dict[str, Any], torch: Any):
    from peft import LoraConfig, get_peft_model
    names = C.select_target_names(linear_module_names(model, torch), args.get("target_modules"))
    if not names:
        raise C.WorkerError("no_target_layers")
    model = prepare_for_training(model, bool(args.get("gradient_checkpointing", True)))
    cfg = LoraConfig(r=int(args.get("rank", 16)), lora_alpha=int(args.get("alpha", 32)), lora_dropout=float(args.get("dropout", 0.05)),
                     target_modules=C.names_regex(names), bias="none", task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, cfg)
    return peft_model, len(names)


def make_optimizer(model: Any, args: dict[str, Any], torch: Any):
    params = [p for p in model.parameters() if p.requires_grad]
    lr = float(args.get("lr", 2e-4))
    wd = float(args.get("weight_decay", 0.0))
    if args.get("method", "qlora") == "qlora":
        import bitsandbytes as bnb
        return bnb.optim.PagedAdamW8bit(params, lr=lr, weight_decay=wd)
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def save_checkpoint(model: Any, optimizer: Any, root: Path, state: dict[str, Any], torch: Any, keep: int = 2) -> Path:
    final = root / f"checkpoint-{state['step']}"
    tmp = root / f"checkpoint-{state['step']}.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(tmp))
    torch.save(optimizer.state_dict(), tmp / "optimizer.pt")
    write_state(tmp / "training_state.json", state)  # written last: a folder without it is not a valid checkpoint
    shutil.rmtree(final, ignore_errors=True)
    os.replace(tmp, final)
    C.prune_checkpoints(root, keep)
    return final


def restore_checkpoint(model: Any, optimizer: Any, ckpt: Path, torch: Any) -> dict[str, Any]:
    from peft import set_peft_model_state_dict
    state = json.loads((ckpt / "training_state.json").read_text(encoding="utf-8"))
    weights = ckpt / "adapter_model.safetensors"
    if weights.is_file():
        from safetensors.torch import load_file
        tensors = load_file(str(weights))
    else:
        tensors = torch.load(str(ckpt / "adapter_model.bin"), map_location="cpu")
    set_peft_model_state_dict(model, tensors)
    opt = ckpt / "optimizer.pt"
    if opt.is_file():
        optimizer.load_state_dict(torch.load(str(opt), map_location="cpu"))
    return state


def to_device(batch: dict[str, list[list[int]]], torch: Any, device: Any, extra_zero_token_types: bool) -> dict[str, Any]:
    out = {k: torch.tensor(v, dtype=torch.long, device=device) for k, v in batch.items()}
    if extra_zero_token_types:
        out["token_type_ids"] = torch.zeros_like(out["input_ids"])
    return out


def token_loss_sum(logits: Any, labels: Any, torch: Any) -> Any:
    """Sum of the cross-entropy over the labelled positions only, in float32 (selecting first keeps the large vocab matrix small)."""
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    keep = shift_labels != IGNORE
    if not bool(keep.any()):
        return logits.sum() * 0.0
    picked = shift_logits[keep].float()
    return torch.nn.functional.cross_entropy(picked, shift_labels[keep], reduction="sum")


def evaluate(model: Any, examples: Sequence[dict[str, Any]], args: dict[str, Any], pad_id: int, torch: Any, device: Any, tt: bool) -> float:
    model.eval()
    batch_size = max(1, int(args.get("batch_size", 1)))
    total, count = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            chunk = examples[i:i + batch_size]
            batch = to_device(collate(chunk, pad_id), torch, device, tt)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            total += float(token_loss_sum(logits, batch["labels"], torch).item())
            count += shifted_label_count(batch["labels"].tolist())
            if C.STOP.is_set():
                break
    model.train()
    return total / count if count else float("nan")


def train(args: dict[str, Any]) -> None:
    import torch
    from transformers import AutoTokenizer, set_seed

    seed = int(args.get("seed", 42))
    set_seed(seed)
    out_dir = Path(args["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_root = out_dir / "checkpoints"
    seq_len = int(args.get("seq_len", 2048))
    batch_size = max(1, int(args.get("batch_size", 1)))
    grad_accum = max(1, int(args.get("grad_accum", 16)))

    tok = AutoTokenizer.from_pretrained(args["base_path"], trust_remote_code=bool(args.get("trust_remote_code", False)))
    tk = Tokenization(tok)
    train_set = build_examples(read_jsonl(args["train_path"]), tk, seq_len, args.get("train_on", "assistant"))
    eval_set = build_examples(read_jsonl(args.get("eval_path")), tk, seq_len, args.get("train_on", "assistant"))
    examples, eval_examples = train_set["examples"], eval_set["examples"][: int(args.get("eval_max_examples", 200))]
    C.log(f"Training examples: {len(examples)} ({train_set['modes']}), {train_set['tokens']} tokens, {train_set['labelled_tokens']} labelled; "
          f"skipped {train_set['skipped']}; eval examples: {len(eval_examples)}.")
    if train_set["first_error"]:
        C.log(f"First example error: {train_set['first_error']}")
    if not examples:
        raise C.WorkerError("no_label_tokens")

    total_steps, per_epoch = plan_steps(len(examples), batch_size, grad_accum, float(args.get("epochs", 1)), int(args.get("max_steps", 0)))
    model = load_model(args, torch)
    model, n_target = attach_lora(model, args, torch)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    C.log(f"LoRA on {n_target} linear layers, {trainable} trainable parameters, {total_steps} optimizer steps ({per_epoch} per epoch).")
    optimizer = make_optimizer(model, args, torch)
    device = next(p for p in model.parameters()).device
    needs_token_types = "gemma3" in str(getattr(model.config, "model_type", "")).lower()

    step, tokens_seen, elapsed_before = 0, 0, 0.0
    losses: list[float] = []
    best_eval: Optional[float] = None
    if args.get("resume"):
        ckpt = latest_checkpoint(ckpt_root)
        if ckpt is not None:
            state = restore_checkpoint(model, optimizer, ckpt, torch)
            step, tokens_seen = int(state["step"]), int(state.get("tokens_seen", 0))
            elapsed_before = float(state.get("elapsed_s", 0.0))
            losses = list(state.get("losses", []))
            best_eval = state.get("best_eval")
            C.log(f"Resumed from {ckpt.name}.")
        else:
            C.log("No checkpoint to resume from; starting over.")

    start_step = step
    tokens_at_start = tokens_seen
    warmup = float(args.get("warmup", 0.03))
    clip = float(args.get("max_grad_norm", 1.0))
    eval_every = int(args.get("eval_every", 0))
    save_every = int(args.get("save_every", 0))
    model.train()
    torch.cuda.reset_peak_memory_stats()
    loop_started = time.time()
    stopped = False

    def checkpoint() -> None:
        save_checkpoint(model, optimizer, ckpt_root, {
            "step": step, "tokens_seen": tokens_seen, "elapsed_s": elapsed_before + time.time() - loop_started,
            "losses": losses[-200:], "best_eval": best_eval}, torch)

    while step < total_steps:
        if C.STOP.is_set():
            stopped = True
            break
        micro = step_indices(step, len(examples), batch_size, grad_accum, per_epoch, seed)
        denominator = max(1, scored_in_step(examples, micro))
        lr = C.lr_at(step, total_steps, float(args.get("lr", 2e-4)), warmup)
        for group in optimizer.param_groups:
            group["lr"] = lr
        step_loss = 0.0
        step_tokens = 0
        t0 = time.time()
        interrupted = False
        for chunk in micro:
            if C.STOP.is_set():          # one optimizer step can take minutes: do not make a stop request wait for it
                interrupted = True
                break
            batch = to_device(collate([examples[i] for i in chunk], tk.pad_id), torch, device, needs_token_types)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
            loss_sum = token_loss_sum(logits, batch["labels"], torch)
            (loss_sum / denominator).backward()
            step_loss += float(loss_sum.item()) / denominator
            step_tokens += int(batch["attention_mask"].sum().item())
            del logits, loss_sum
        if interrupted:                  # the half-accumulated step is dropped; the checkpoint is the last complete step
            optimizer.zero_grad(set_to_none=True)
            stopped = True
            break
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1
        tokens_seen += step_tokens
        losses.append(round(step_loss, 5))
        dt = max(1e-6, time.time() - t0)
        C.emit("progress", step=step, total=total_steps, loss=round(step_loss, 5), lr=lr, tokens_per_s=int(step_tokens / dt),
               avg_tokens_per_s=int((tokens_seen - tokens_at_start) / max(1e-6, time.time() - loop_started)),
               eta_s=C.eta_seconds(step - start_step, total_steps - start_step, time.time() - loop_started),
               gpu_mem_mb=int(torch.cuda.max_memory_allocated() / 1048576), epoch=round(step / per_epoch, 3))
        if eval_every and eval_examples and step % eval_every == 0 and step < total_steps:
            value = evaluate(model, eval_examples, args, tk.pad_id, torch, device, needs_token_types)
            C.emit("eval", step=step, eval_loss=round(value, 5))
            if best_eval is None or value < best_eval:
                best_eval = value
        if save_every and step % save_every == 0 and step < total_steps:
            checkpoint()
            C.log(f"Checkpoint saved at step {step}.")

    if stopped:
        checkpoint()
        C.log(f"Stop requested: checkpoint saved at step {step}.")
        return

    final_eval = None
    if eval_examples:
        final_eval = evaluate(model, eval_examples, args, tk.pad_id, torch, device, needs_token_types)
        C.emit("eval", step=step, eval_loss=round(final_eval, 5))
        if best_eval is None or final_eval < best_eval:
            best_eval = final_eval
    model.save_pretrained(str(out_dir))
    tail = losses[-10:]
    summary = {
        "base_path": args["base_path"], "method": args.get("method", "qlora"), "rank": int(args.get("rank", 16)),
        "alpha": int(args.get("alpha", 32)), "steps": step, "epochs_done": round(step / per_epoch, 3), "examples": len(examples),
        "final_loss": round(sum(tail) / len(tail), 5) if tail else None, "best_loss": min(losses) if losses else None,
        "final_eval_loss": None if final_eval is None else round(final_eval, 5), "best_eval_loss": None if best_eval is None else round(best_eval, 5),
        "tokens_seen": tokens_seen, "trainable_params": trainable, "target_layers": n_target,
        "time_s": round(elapsed_before + time.time() - loop_started, 1),
        "peak_gpu_mb": int(torch.cuda.max_memory_allocated() / 1048576), "seed": seed, "label_modes": train_set["modes"],
        "skipped": train_set["skipped"],
    }
    write_state(out_dir / "training_summary.json", summary)
    if not args.get("keep_checkpoints"):
        shutil.rmtree(ckpt_root, ignore_errors=True)
    C.emit("artifact", path=str(out_dir), kind="adapter")
    C.emit("result", data=summary)


def main() -> int:
    return C.run_worker(train)


if __name__ == "__main__":
    sys.exit(main())
