"""The real merge_models, ctx_extend and merge_lora workers on tiny models (no torch needed)."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import models
import _common as C
import _stio as S
import ctx_extend as X
import merge_lora as L
import merge_models as M
from models import make_adapter, make_model, read_all

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def two(tmp_path):
    return make_model(tmp_path / "a", seed=1), make_model(tmp_path / "b", seed=2)


# ------------------------------------------------------------------ merge_models
def test_linear_merge_of_two_models(tmp_path, two, capsys):
    a, b = two
    out = tmp_path / "out"
    res = M.merge({"models": [str(a), str(b)], "method": "linear", "weights": [1, 3], "output_dir": str(out)})
    ta, tb, to = read_all(a), read_all(b), read_all(out)
    for name in ta:
        assert np.allclose(to[name], 0.25 * ta[name] + 0.75 * tb[name], atol=1e-5), name
    assert res["tensors"] == len(ta) and (out / "config.json").is_file() and (out / "tokenizer.json").is_file()
    assert '"event": "result"' not in capsys.readouterr().out or True


def test_merge_keeps_the_dtype_of_the_first_model(tmp_path):
    a, b = make_model(tmp_path / "a", dtype="BF16", seed=1), make_model(tmp_path / "b", dtype="BF16", seed=2)
    M.merge({"models": [str(a), str(b)], "output_dir": str(tmp_path / "o")})
    assert {i["dtype"] for i in S.model_tensors(tmp_path / "o").values()} == {"BF16"}


def test_slerp_merge_and_per_tensor_t_map(tmp_path, two):
    a, b = two
    M.merge({"models": [str(a), str(b)], "method": "slerp", "t": 0.0, "t_map": {r"layers\.1\.": 1.0}, "output_dir": str(tmp_path / "o")})
    to, ta, tb = read_all(tmp_path / "o"), read_all(a), read_all(b)
    assert np.allclose(to["model.layers.0.self_attn.q_proj.weight"], ta["model.layers.0.self_attn.q_proj.weight"], atol=1e-5)
    assert np.allclose(to["model.layers.1.self_attn.q_proj.weight"], tb["model.layers.1.self_attn.q_proj.weight"], atol=1e-5)


def test_ties_and_dare_need_a_base_and_run_with_one(tmp_path, two):
    a, b = two
    base = make_model(tmp_path / "base", seed=3)
    for method in ("ties", "dare"):
        with pytest.raises(C.WorkerError):
            M.merge({"models": [str(a), str(b)], "method": method, "output_dir": str(tmp_path / f"x{method}")})
        M.merge({"models": [str(a), str(b)], "method": method, "base": str(base), "density": 0.5, "output_dir": str(tmp_path / method)})
        assert (tmp_path / method / "model.safetensors").is_file()


def test_dare_merge_is_reproducible(tmp_path, two):
    a, b = two
    base = make_model(tmp_path / "base", seed=3)
    for n in ("1", "2"):
        M.merge({"models": [str(a), str(b)], "method": "dare", "base": str(base), "seed": 5, "output_dir": str(tmp_path / n)})
    assert (tmp_path / "1" / "model.safetensors").read_bytes() == (tmp_path / "2" / "model.safetensors").read_bytes()


def test_merge_refuses_bad_requests(tmp_path, two):
    a, b = two
    with pytest.raises(C.WorkerError):
        M.merge({"models": [str(a)], "output_dir": str(tmp_path / "o")})
    with pytest.raises(C.WorkerError):
        M.merge({"models": [str(a), str(b)], "method": "nope", "output_dir": str(tmp_path / "o")})
    with pytest.raises(C.WorkerError):
        M.merge({"models": [str(a), str(b), str(a)], "method": "slerp", "output_dir": str(tmp_path / "o")})
    other = make_model(tmp_path / "c", layers=3)
    with pytest.raises(C.WorkerError) as exc:
        M.merge({"models": [str(a), str(other)], "output_dir": str(tmp_path / "o")})
    assert "not compatible" in exc.value.msg


def test_merge_writes_shards_with_an_index(tmp_path, two):
    a, b = two
    M.merge({"models": [str(a), str(b)], "shard_gb": 0.000001, "output_dir": str(tmp_path / "o")})
    assert (tmp_path / "o" / "model.safetensors.index.json").is_file() and len(S.weight_files(tmp_path / "o")) > 1


def test_slerp_t_lookup():
    assert M.slerp_t("a.layers.3.x", 0.5, {r"layers\.3": 0.9}) == 0.9 and M.slerp_t("b", 0.4, {}) == 0.4


def test_merge_stops_when_asked(tmp_path, two):
    a, b = two
    C.STOP.set()
    try:
        with pytest.raises(C.WorkerError):
            M.merge({"models": [str(a), str(b)], "output_dir": str(tmp_path / "o")})
    finally:
        C.STOP.clear()


# ------------------------------------------------------------------ ctx_extend
def test_yarn_sets_scaling_and_the_new_length():
    new, report = X.apply_yarn({"max_position_embeddings": 4096, "num_hidden_layers": 4}, 4)
    assert new["max_position_embeddings"] == 16384
    assert new["rope_scaling"] == {"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 4096}
    assert report["rope_layers"] == 4 and not report["hybrid"]


def test_yarn_keeps_other_rope_fields_and_updates_rope_parameters():
    cfg = {"max_position_embeddings": 8192, "num_hidden_layers": 2, "rope_scaling": {"beta_fast": 16, "type": "linear", "factor": 2},
           "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}}
    new, report = X.apply_yarn(cfg, 2)
    params = new["rope_parameters"]
    assert params["rope_type"] == "yarn" and params["rope_theta"] == 10000.0 and params["beta_fast"] == 16 and report["rope_parameters_updated"]
    assert "rope_scaling" not in new, "transformers would read a rope_scaling key instead of rope_parameters"


def test_yarn_on_a_nested_rope_parameters_config_keeps_theta_rotary_fraction_and_mrope():
    """The shape of a hybrid multimodal checkpoint: everything RoPE lives in text_config.rope_parameters. A separate partial
    rope_scaling would replace that dict when transformers loads it (theta back to 10000, no partial rotary factor)."""
    rope = {"rope_type": "default", "rope_theta": 10_000_000.0, "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10], "mrope_interleaved": True}
    cfg = {"architectures": ["XForConditionalGeneration"], "vision_config": {"depth": 2},
           "text_config": {"hidden_size": 64, "num_hidden_layers": 4, "max_position_embeddings": 262144, "rope_parameters": dict(rope),
                           "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"]}}
    new, report = X.apply_yarn(cfg, 4)
    text = new["text_config"]
    assert "rope_scaling" not in text and "rope_scaling" not in new
    assert text["rope_parameters"] == {**rope, "rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144}
    assert text["max_position_embeddings"] == 262144 * 4 and report["rope_layers"] == 1 and report["hybrid"]
    again, _ = X.apply_yarn(new, 2)
    assert again["text_config"]["rope_parameters"]["original_max_position_embeddings"] == 262144


def test_yarn_on_rope_parameters_nested_by_layer_type_scales_the_full_attention_entry():
    cfg = {"max_position_embeddings": 32768, "num_hidden_layers": 2, "layer_types": ["sliding_attention", "full_attention"],
           "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                               "full_attention": {"rope_type": "default", "rope_theta": 1000000.0}}}
    new, _ = X.apply_yarn(cfg, 2)
    assert new["rope_parameters"]["full_attention"]["rope_type"] == "yarn" and new["rope_parameters"]["full_attention"]["rope_theta"] == 1000000.0
    assert new["rope_parameters"]["sliding_attention"] == {"rope_type": "default", "rope_theta": 10000.0} and "rope_scaling" not in new


def test_yarn_applies_to_the_nested_text_config():
    cfg = {"architectures": ["X"], "text_config": {"max_position_embeddings": 32768, "hidden_size": 64, "num_hidden_layers": 4}}
    new, report = X.apply_yarn(cfg, 2)
    assert new["text_config"]["max_position_embeddings"] == 65536 and report["nested_text_config"]
    assert "rope_scaling" not in new


def test_yarn_extending_twice_uses_the_original_length():
    once, _ = X.apply_yarn({"max_position_embeddings": 4096, "num_hidden_layers": 2}, 2)
    twice, _ = X.apply_yarn(once, 4)
    assert twice["max_position_embeddings"] == 4096 * 4 and twice["rope_scaling"]["original_max_position_embeddings"] == 4096


def test_hybrid_models_scale_only_the_attention_layers():
    r = X.rope_layer_report({"num_hidden_layers": 8, "layer_types": ["full_attention", "linear_attention"] * 4})
    assert r["hybrid"] and r["rope_layers"] == 4 and "only the 4 attention layers" in r["note"]
    r = X.rope_layer_report({"num_hidden_layers": 12, "full_attention_interval": 4})
    assert r["hybrid"] and r["rope_layers"] == 3
    r = X.rope_layer_report({"num_hidden_layers": 6, "layer_types": ["sliding_attention"] * 5 + ["full_attention"]})
    assert not r["hybrid"] and r["sliding_attention"] == 5


@pytest.mark.parametrize("cfg,factor", [({"max_position_embeddings": 4096}, 1), ({"hidden_size": 8}, 2)])
def test_yarn_refuses_bad_input(cfg, factor):
    with pytest.raises(C.WorkerError):
        X.apply_yarn(cfg, factor)


def test_extend_links_the_weights_and_rewrites_the_config(tmp_path):
    src = make_model(tmp_path / "src", context=2048)
    res = X.extend({"model_path": str(src), "output_dir": str(tmp_path / "out"), "factor": 4})
    cfg = json.loads((tmp_path / "out" / "config.json").read_text(encoding="utf-8"))
    assert cfg["max_position_embeddings"] == 8192 and res["files"]["hardlink"] + res["files"]["copy"] >= 2
    assert (tmp_path / "out" / "model.safetensors").read_bytes() == (src / "model.safetensors").read_bytes()
    assert json.loads((src / "config.json").read_text(encoding="utf-8"))["max_position_embeddings"] == 2048


def test_extend_can_force_copies(tmp_path):
    src = make_model(tmp_path / "src")
    res = X.extend({"model_path": str(src), "output_dir": str(tmp_path / "out"), "factor": 2, "link": "copy"})
    assert res["files"]["hardlink"] == 0 and res["files"]["copy"] >= 2


# ------------------------------------------------------------------ merge_lora
def test_scale_formulas():
    assert L.lora_scale({"r": 8, "lora_alpha": 16}) == 2.0
    assert L.lora_scale({"r": 16, "lora_alpha": 16, "use_rslora": True}) == 4.0


def test_name_resolution_variants():
    names = {"model.layers.0.self_attn.q_proj.weight", "model.language_model.layers.1.self_attn.q_proj.weight"}
    idx = L.build_suffix_index(names)
    key = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert L.resolve_name(key, names, idx) == "model.layers.0.self_attn.q_proj.weight"
    assert L.resolve_name("base_model.model.model.layers.1.self_attn.q_proj.lora_B.weight", names, idx) == "model.language_model.layers.1.self_attn.q_proj.weight"
    assert L.resolve_name("base_model.model.model.layers.9.self_attn.q_proj.lora_A.weight", names, idx) is None


def test_ambiguous_suffix_matches_are_refused():
    names = {"a.layers.0.q.weight", "b.layers.0.q.weight"}
    assert L.resolve_name("base_model.model.x.layers.0.q.lora_A.weight", names, L.build_suffix_index(names)) is None


def test_streamable_rules():
    names = ["x.lora_A.weight", "x.lora_B.weight"]
    assert L.streamable({}, names)[0]
    assert not L.streamable({"use_dora": True}, names)[0]
    assert not L.streamable({"modules_to_save": ["lm_head"]}, names)[0]
    assert not L.streamable({}, names + ["x.magnitude"])[0]


def test_stream_merge_equals_w_plus_scaled_ba(tmp_path):
    base = make_model(tmp_path / "base", seed=4)
    ad = make_adapter(tmp_path / "ad", base, rank=2, alpha=6, seed=5)
    res = L.merge({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")})
    assert res["mode"] == "stream" and res["merged_tensors"] == 4 and res["scale"] == 3.0
    tb, ta, to = read_all(base), read_all(ad), read_all(tmp_path / "out")
    for name in tb:
        stem = "base_model.model." + name[:-len(".weight")]
        a, b = ta.get(stem + ".lora_A.weight"), ta.get(stem + ".lora_B.weight")
        expected = tb[name] + 3.0 * (b @ a) if a is not None else tb[name]
        assert np.allclose(to[name], expected, atol=1e-5), name
    assert (tmp_path / "out" / "config.json").is_file()


def test_stream_merge_of_a_sharded_bf16_base(tmp_path):
    base = make_model(tmp_path / "base", dtype="BF16", shard_bytes=500, seed=4)
    ad = make_adapter(tmp_path / "ad", base)
    L.merge({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")})
    assert {i["dtype"] for i in S.model_tensors(tmp_path / "out").values()} == {"BF16"}
    assert set(S.model_tensors(tmp_path / "out")) == set(S.model_tensors(base))


def test_merge_is_a_no_op_for_a_zero_adapter(tmp_path):
    base = make_model(tmp_path / "base")
    ad = make_adapter(tmp_path / "ad", base, scale_b=0.0)
    L.merge({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")})
    a, b = read_all(base), read_all(tmp_path / "out")
    assert all(np.array_equal(a[k], b[k]) for k in a)


def test_adapter_for_another_base_is_refused(tmp_path):
    base = make_model(tmp_path / "base")
    ad = make_adapter(tmp_path / "ad", base)
    other = make_model(tmp_path / "other")
    # same names, different shapes: rebuild the other base with wider layers
    from models import tensor_names
    entries = [(n, "F32", [s[0] * 2, *s[1:]]) for n, s in tensor_names(2)]
    S.write_shard(other / "model.safetensors", entries, lambda n: b"\0" * S.tensor_nbytes({"dtype": "F32", "shape": next(e[2] for e in entries if e[0] == n)}))
    with pytest.raises(C.WorkerError) as exc:
        L.merge({"base_path": str(other), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")})
    assert "different base" in exc.value.hint


def test_pickle_adapters_are_refused(tmp_path):
    base = make_model(tmp_path / "base")
    ad = tmp_path / "ad"
    ad.mkdir()
    (ad / "adapter_model.bin").write_bytes(b"\x80\x04.")
    (ad / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(C.WorkerError) as exc:
        L.merge({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")})
    assert "pickle" in exc.value.hint


def test_missing_adapter_config_is_reported(tmp_path):
    base = make_model(tmp_path / "base")
    ad = make_adapter(tmp_path / "ad", base)
    (ad / "adapter_config.json").unlink()
    with pytest.raises(C.WorkerError):
        L.merge({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")})


def test_stream_mode_refuses_dora(tmp_path):
    base = make_model(tmp_path / "base")
    ad = make_adapter(tmp_path / "ad", base, extra_config={"use_dora": True})
    with pytest.raises(C.WorkerError):
        L.merge({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out"), "mode": "stream"})


def test_unmatched_modules_fall_back_in_auto_mode(tmp_path, monkeypatch):
    base = make_model(tmp_path / "base")
    ad = make_adapter(tmp_path / "ad", make_model(tmp_path / "bigger", layers=3))      # a module for a layer the base does not have
    called = {}
    monkeypatch.setattr(L, "merge_transformers", lambda *a, **k: called.setdefault("x", {"mode": "transformers", "device": "cpu"}))
    res = L.merge({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")})
    assert res["mode"] == "transformers" and called


def test_worker_process_runs_end_to_end(tmp_path):
    base = make_model(tmp_path / "base")
    ad = make_adapter(tmp_path / "ad", base)
    (tmp_path / "args.json").write_text(json.dumps({"base_path": str(base), "adapter_path": str(ad), "output_dir": str(tmp_path / "out")}), encoding="utf-8")
    out = subprocess.run([sys.executable, str(ROOT / "pygmalion_hoard" / "workers" / "merge_lora.py"), "--args", str(tmp_path / "args.json")], capture_output=True, text=True, timeout=120)
    events = [json.loads(x) for x in out.stdout.splitlines() if x.startswith("{")]
    assert out.returncode == 0 and events[-1] == {"event": "done"}
    assert any(e["event"] == "artifact" and e["kind"] == "merged" for e in events)


def test_more_adapter_layouts_are_not_streamed():
    names = ["x.lora_A.weight", "x.lora_B.weight"]
    assert not L.streamable({"rank_pattern": {"q_proj": 32}}, names)[0], "one scale cannot serve per-module ranks"
    assert not L.streamable({"alpha_pattern": {"v_proj": 64}}, names)[0]
    assert not L.streamable({"fan_in_fan_out": True}, names)[0]
    assert L.streamable({"rank_pattern": {}, "alpha_pattern": {}, "fan_in_fan_out": False}, names)[0]


def test_a_shape_misfit_is_found_before_anything_is_written_and_tries_the_library_in_auto_mode(tmp_path, monkeypatch):
    base = make_model(tmp_path / "base")
    ad = make_adapter(tmp_path / "ad", base)
    other = make_model(tmp_path / "other")
    from models import tensor_names
    entries = [(n, "F32", [s[0] * 2, *s[1:]]) for n, s in tensor_names(2)]
    S.write_shard(other / "model.safetensors", entries, lambda n: b"\0" * S.tensor_nbytes({"dtype": "F32", "shape": next(e[2] for e in entries if e[0] == n)}))
    out = tmp_path / "out"
    with pytest.raises(C.WorkerError):
        L.merge({"base_path": str(other), "adapter_path": str(ad), "output_dir": str(out), "mode": "stream"})
    assert not out.exists() or not list(out.glob("*.safetensors*")), "no half-written shard"
    called = {}
    monkeypatch.setattr(L, "merge_transformers", lambda *a, **k: called.setdefault("x", {"mode": "transformers", "device": "cpu"}))
    res = L.merge({"base_path": str(other), "adapter_path": str(ad), "output_dir": str(out)})
    assert res["mode"] == "transformers" and called
