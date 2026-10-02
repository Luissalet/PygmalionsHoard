"""Pure helpers every worker shares, and the line protocol."""

import json
import sys
from pathlib import Path

import pytest

import models  # noqa: F401
import _common as C

ROOT = Path(__file__).resolve().parent.parent


def test_lr_schedule_warms_up_then_decays_to_zero():
    lrs = [C.lr_at(s, 100, 1e-3, 0.1) for s in range(100)]
    assert lrs[0] < lrs[9] <= 1e-3 + 1e-12
    assert max(lrs) == pytest.approx(1e-3, rel=1e-6)
    assert lrs[-1] < 1e-5 and all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))


def test_lr_without_warmup_starts_at_the_base_rate():
    assert C.lr_at(0, 50, 2e-4, 0.0) == pytest.approx(2e-4)


def test_lr_with_a_single_step_is_defined():
    assert C.lr_at(0, 1, 1e-3, 0.03) > 0


def test_epoch_order_is_a_deterministic_permutation():
    a, b = C.epoch_order(20, 0, 7), C.epoch_order(20, 0, 7)
    assert a == b and sorted(a) == list(range(20))
    assert C.epoch_order(20, 1, 7) != a and C.epoch_order(20, 0, 8) != a


NAMES = ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj",
         "model.layers.0.mlp.experts.3.up_proj", "model.layers.0.mlp.gate", "model.layers.0.block_sparse_moe.router", "lm_head", "model.embed_tokens",
         "model.vision_tower.layers.0.q_proj", "model.layers.0.input_layernorm", "model.visual.blocks.0.attn.qkv"]


def test_target_selection_defaults_to_the_text_projections():
    got = C.select_target_names(NAMES)
    assert got == ["model.layers.0.self_attn.q_proj", "model.layers.0.self_attn.k_proj", "model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj"]


def test_target_selection_honours_a_request():
    assert C.select_target_names(NAMES, ["q_proj"]) == ["model.layers.0.self_attn.q_proj"]
    assert C.select_target_names(NAMES, ["all"]) == C.select_target_names(NAMES)


def test_names_regex_matches_whole_names_only():
    import re
    rx = re.compile(C.names_regex(["model.layers.0.self_attn.q_proj"]))
    assert rx.fullmatch("model.layers.0.self_attn.q_proj")
    assert not rx.search("x.model.layers.0.self_attn.q_proj") and not rx.search("model.layers.0.self_attn.q_proj2")
    assert not rx.search("modelXlayers.0.self_attn.q_proj")


def test_checkpoint_steps_and_pruning(tmp_path):
    for s in (10, 20, 30, 5):
        d = tmp_path / f"checkpoint-{s}"
        d.mkdir()
        (d / "training_state.json").write_text("{}", encoding="utf-8")
    (tmp_path / "checkpoint-40").mkdir()            # no state file: an interrupted save, ignored
    (tmp_path / "other").mkdir()
    assert C.checkpoint_steps(tmp_path) == [5, 10, 20, 30]
    assert C.prune_checkpoints(tmp_path, keep=2) == [5, 10]
    assert C.checkpoint_steps(tmp_path) == [20, 30]


def test_eta():
    assert C.eta_seconds(10, 100, 20.0) == 180 and C.eta_seconds(0, 100, 5.0) == 0


@pytest.mark.parametrize("text,fragment", [("CUDA out of memory. Tried to allocate", "ran out of memory"), ("No module named 'peft'", "pip install peft"),
                                           ("401 Client Error: gated repo", "gated"), ("No space left on device", "disk is full"),
                                           ("Torch not compiled with CUDA enabled", "CUDA"), ("The checkpoint you are trying to load has model type `x` which Transformers does not recognize this architecture", "Update transformers")])
def test_hints(text, fragment):
    assert fragment in C.hint_for(text)


def test_hint_for_unknown_text_is_empty():
    assert C.hint_for("something else") == ""


def run_worker(tmp_path, script, args):
    import subprocess
    (tmp_path / "args.json").write_text(json.dumps(args), encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "pygmalion_hoard" / "workers" / script), "--args", str(tmp_path / "args.json")], capture_output=True, text=True, timeout=120)
    events = [json.loads(line) for line in out.stdout.splitlines() if line.startswith("{")]
    return out.returncode, events


def test_worker_reports_errors_as_events(tmp_path):
    code, events = run_worker(tmp_path, "ctx_extend.py", {"model_path": str(tmp_path / "missing"), "output_dir": str(tmp_path / "o")})
    assert code == 1 and events[-1]["event"] == "error" and "config.json" in events[-1]["msg"]


def test_worker_with_unreadable_args(tmp_path):
    import subprocess
    out = subprocess.run([sys.executable, str(ROOT / "pygmalion_hoard" / "workers" / "ctx_extend.py"), "--args", str(tmp_path / "none.json")], capture_output=True, text=True)
    assert out.returncode == 1 and '"event": "error"' in out.stdout


def test_package_module_loads_the_pure_files():
    assert C.package_module("merge_math").METHODS and C.package_module("chatmask").IGNORE_INDEX == -100


def test_only_multimodal_types_the_causal_table_knows_count_as_trainable():
    import probe_env as P
    causal = {"llama": "LlamaForCausalLM", "qwen3_5": "Qwen3_5ForCausalLM", "gemma3": ("Gemma3ForConditionalGeneration",)}
    image_text = {"qwen3_5": "Qwen3_5ForConditionalGeneration", "llava": "LlavaForConditionalGeneration", "gemma3": "Gemma3ForConditionalGeneration"}
    names = P.causal_class_names(causal, image_text)
    assert {"LlamaForCausalLM", "Qwen3_5ForCausalLM", "Qwen3_5ForConditionalGeneration", "Gemma3ForConditionalGeneration"} <= names
    assert "LlavaForConditionalGeneration" not in names, "AutoModelForCausalLM cannot load it: do not promise training"


# ------------------------------------------------------------------ the environment probe lists every GPU
class FakeCuda:
    """A torch.cuda with four devices; ``broken`` indexes fail on some queries the way a driver can."""

    def __init__(self, count=4, broken=()):
        self.count, self.broken = count, set(broken)

    def is_available(self):
        return self.count > 0

    def device_count(self):
        return self.count

    def get_device_properties(self, index):
        if index in self.broken:
            raise RuntimeError("CUDA error: invalid device ordinal")
        from types import SimpleNamespace
        return SimpleNamespace(name=["RTX 4070 Ti", "RTX 5060 Ti", "RTX 5060 Ti", "RTX 5060 Ti"][index], total_memory=[12, 16, 16, 16][index] * 1024 ** 3,
                               major=8 if index == 0 else 12, minor=9 if index == 0 else 0)

    def get_device_name(self, index):
        return f"Fallback name {index}"

    def mem_get_info(self, index):
        return (1, 8 * 1024 ** 3)

    def get_device_capability(self, index):
        return (12, 0)

    def is_bf16_supported(self):
        raise RuntimeError("bf16 query is not available on this driver")


def fake_torch(**kwargs):
    from types import SimpleNamespace
    return SimpleNamespace(__version__="2.9.0+cu128", version=SimpleNamespace(cuda="12.8"), cuda=FakeCuda(**kwargs))


def test_every_device_is_listed_with_index_name_and_memory():
    import probe_env as P
    devices = P.list_devices(fake_torch())
    assert [d["index"] for d in devices] == [0, 1, 2, 3]
    assert [d["total_mb"] for d in devices] == [12288, 16384, 16384, 16384] and devices[1]["name"] == "RTX 5060 Ti"
    assert devices[0]["capability"] == "8.9" and devices[0]["bf16"] is True and "error" not in devices[0]


def test_a_device_that_cannot_be_read_in_full_does_not_empty_the_list():
    """The failure seen on the real machine: one failing query used to drop every device while CUDA was reported as available."""
    import probe_env as P
    devices = P.list_devices(fake_torch(broken=[2]))
    assert len(devices) == 4
    assert devices[2]["name"] == "Fallback name 2" and devices[2]["total_mb"] == 8192 and devices[2]["capability"] == "12.0"
    assert devices[3]["name"] == "RTX 5060 Ti"


def test_the_probe_reports_devices_when_cuda_is_available(monkeypatch):
    import probe_env as P
    monkeypatch.setitem(sys.modules, "torch", fake_torch(broken=[1]))
    monkeypatch.setattr(P, "version_of", lambda name: None if name == "transformers" else "1.0")
    monkeypatch.setattr(P, "trainable_architectures", lambda: [])
    info = P.probe({})
    assert info["torch"]["cuda_available"] is True and info["torch"]["device_count"] == 4 and len(info["torch"]["devices"]) == 4
    assert [d["index"] for d in info["torch"]["devices"]] == [0, 1, 2, 3]
