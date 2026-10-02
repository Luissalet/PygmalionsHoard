"""The trainer's torch-free part: records to examples, step planning, batching and checkpoints."""

import json
from pathlib import Path

import pytest

import models  # noqa: F401
import train_lora as T


class FakeTok:
    """Character-level tokenizer with a ChatML-like template."""

    def __init__(self, template=True, offsets=True, pad=None):
        self.chat_template = "x" if template else None
        self.eos_token_id = 1
        self.pad_token_id = pad
        self.offsets = offsets

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        out = "".join(f"<{m['role']}>{m['content']}</s>" for m in messages)
        return out + ("<assistant>" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False, return_offsets_mapping=False):
        ids = [ord(c) + 2 for c in text]
        if return_offsets_mapping:
            if not self.offsets:
                raise NotImplementedError("slow tokenizer")
            return {"input_ids": ids, "offset_mapping": [(i, i + 1) for i in range(len(text))]}
        return {"input_ids": ids}


CHAT = {"messages": [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "buenas tardes"}]}
PAIR = {"prompt": "dime algo", "response": "algo"}
TEXT = {"text": "x" * 40}


def test_record_kind():
    assert T.record_kind(CHAT, True) == "chat" and T.record_kind(PAIR, True) == "chat" and T.record_kind(PAIR, False) == "completion"
    assert T.record_kind(TEXT, True) == "text" and T.record_kind({"text": "  "}, True) == "" and T.record_kind({}, True) == ""


def test_read_jsonl_skips_blank_and_broken_lines(tmp_path):
    p = tmp_path / "d.jsonl"
    p.write_text('{"a": 1}\n\nnot json\n[1]\n{"b": 2}\n', encoding="utf-8")
    assert T.read_jsonl(str(p)) == [{"a": 1}, {"b": 2}] and T.read_jsonl(None) == []


def test_tokenization_pad_falls_back_to_eos():
    assert T.Tokenization(FakeTok()).pad_id == 1 and T.Tokenization(FakeTok(pad=7)).pad_id == 7


def test_chat_examples_label_only_the_answer():
    r = T.build_examples([CHAT], T.Tokenization(FakeTok()), 512)
    ex = r["examples"][0]
    assert "".join(chr(l - 2) for l in ex["labels"] if l != -100) == "buenas tardes</s>"
    assert r["modes"] == {"prefix": 1} and r["labelled_tokens"] == len("buenas tardes</s>")


def test_completion_examples_for_models_without_a_template():
    r = T.build_examples([PAIR], T.Tokenization(FakeTok(template=False)), 512)
    ex = r["examples"][0]
    assert r["modes"] == {"completion": 1} and ex["labels"][-1] == 1 and ex["labels"][0] == -100


def test_text_records_are_packed():
    r = T.build_examples([TEXT, TEXT, TEXT], T.Tokenization(FakeTok()), 32)
    assert r["modes"]["packed"] >= 3 and all(e["labels"] == e["input_ids"] for e in r["examples"])


def test_examples_without_a_label_or_with_errors_are_counted():
    long_prompt = {"messages": [{"role": "user", "content": "q" * 200}, {"role": "assistant", "content": "a"}]}
    r = T.build_examples([long_prompt, {"nothing": 1}], T.Tokenization(FakeTok()), 50)
    assert r["skipped"]["no_label"] == 1 and r["skipped"]["empty"] == 1 and not r["examples"]


def test_a_failing_template_is_skipped_not_fatal():
    class Bad(FakeTok):
        def apply_chat_template(self, *a, **k):
            raise ValueError("template says no")
    r = T.build_examples([CHAT, CHAT], T.Tokenization(Bad()), 100)
    assert r["skipped"]["error"] == 2 and "template says no" in r["first_error"] and not r["examples"]


def test_tokenizers_without_offsets_still_work():
    r = T.build_examples([CHAT], T.Tokenization(FakeTok(offsets=False)), 512)
    assert r["labelled_tokens"] == len("buenas tardes</s>")


@pytest.mark.parametrize("n,b,g,epochs,max_steps,expected", [(100, 2, 4, 1.0, 0, (13, 13)), (100, 2, 4, 2.0, 0, (26, 13)), (100, 2, 4, 1.0, 5, (5, 13)),
                                                              (3, 8, 8, 0.0, 0, (1, 1))])
def test_plan_steps(n, b, g, epochs, max_steps, expected):
    assert T.plan_steps(n, b, g, epochs, max_steps) == expected


def test_every_example_is_visited_once_per_epoch():
    n, b, g = 23, 2, 3
    total, per_epoch = T.plan_steps(n, b, g, 1.0, 0)
    seen = [i for step in range(per_epoch) for micro in T.step_indices(step, n, b, g, per_epoch, 5) for i in micro]
    assert sorted(seen) == list(range(n))


def test_step_indices_are_pure_so_a_resume_is_exact():
    a = T.step_indices(7, 50, 2, 2, 13, 1)
    assert a == T.step_indices(7, 50, 2, 2, 13, 1) and a != T.step_indices(7, 50, 2, 2, 13, 2)


def test_second_epoch_uses_another_order():
    first = [i for m in T.step_indices(0, 40, 2, 2, 10, 1) for i in m]
    second = [i for m in T.step_indices(10, 40, 2, 2, 10, 1) for i in m]
    assert first != second


def test_collate_pads_to_a_multiple():
    out = T.collate([{"input_ids": [5, 6, 7], "labels": [-100, 6, 7]}, {"input_ids": [5], "labels": [5]}], pad_id=0, multiple=4)
    assert [len(r) for r in out["input_ids"]] == [4, 4]
    assert out["attention_mask"] == [[1, 1, 1, 0], [1, 0, 0, 0]] and out["labels"][1] == [5, -100, -100, -100]


def test_scored_positions_exclude_the_first_token_and_masked_ones():
    assert T.shifted_label_count([[7, 8, -100, 9]]) == 2
    ex = [{"labels": [1, 2, 3]}, {"labels": [-100, -100, 4]}]
    assert T.scored_in_step(ex, [[0], [1]]) == 3


def test_latest_checkpoint_and_state_roundtrip(tmp_path):
    assert T.latest_checkpoint(tmp_path) is None
    for s in (3, 12):
        d = tmp_path / f"checkpoint-{s}"
        d.mkdir()
        T.write_state(d / "training_state.json", {"step": s})
    assert T.latest_checkpoint(tmp_path).name == "checkpoint-12"
    assert json.loads((tmp_path / "checkpoint-12" / "training_state.json").read_text(encoding="utf-8")) == {"step": 12}
    assert not list(tmp_path.glob("**/*.tmp"))


class _Param:
    def __init__(self, dtype):
        self.requires_grad = True
        self.dtype = dtype


class _Model:
    """Duck-typed stand-in for a 4-bit model: records what preparing it for training does."""

    def __init__(self):
        self.params = {"model.embed_tokens.weight": _Param("bf16"), "model.layers.0.q_proj.weight": _Param("4bit"), "lm_head.weight": _Param("bf16")}
        self.calls = []
        self.config = type("Cfg", (), {"use_cache": True})()

    def named_parameters(self):
        return list(self.params.items())

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.calls.append(("checkpointing", gradient_checkpointing_kwargs))

    def enable_input_require_grads(self):
        self.calls.append(("input_grads", None))


def test_preparing_a_4bit_model_freezes_it_without_upcasting_embeddings_or_head():
    model = T.prepare_for_training(_Model(), checkpointing=True)
    assert all(not p.requires_grad for p in model.params.values())
    assert [p.dtype for p in model.params.values()] == ["bf16", "4bit", "bf16"], "no float32 copy of the embeddings or the head"
    assert ("checkpointing", {"use_reentrant": False}) in model.calls and model.config.use_cache is False
    plain = T.prepare_for_training(_Model(), checkpointing=False)
    assert plain.calls == [] and plain.config.use_cache is False
