"""The training plan, the recipe pipelines that follow a stage, and the checks the operations make before queueing anything."""

import json

import pytest

from conftest import call
from models import make_model
from pygmalion_hoard.errors import PygmalionError

BIG = {"hidden_size": 4096, "num_hidden_layers": 32, "num_attention_heads": 32, "num_key_value_heads": 8, "intermediate_size": 14336,
       "vocab_size": 128256, "max_position_embeddings": 131072}
HUGE = {"hidden_size": 8192, "num_hidden_layers": 80, "num_attention_heads": 64, "num_key_value_heads": 8, "intermediate_size": 28672,
        "vocab_size": 128256, "max_position_embeddings": 131072}


@pytest.fixture
def big_base(ctx):
    return ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--big-8b", extra_config=BIG), "acme/big-8b")


def plan(ctx, base, dataset, **overrides):
    return ctx.svc.planner.train_plan(base["id"], dataset["id"], None, overrides)


# ------------------------------------------------------------------------------------------------ train_plan
def test_plan_uses_the_defaults_and_fits_a_tiny_model(ctx, base_model, dataset):
    p = plan(ctx, base_model, dataset)
    assert p["params"]["method"] == "qlora" and p["params"]["rank"] == 16 and p["params"]["seq_len"] == 2048
    assert p["pick"]["fits"] is True and set(p["pick"]["gpus"]) <= {2, 3} and p["allowed_gpus"] == [2, 3]
    assert p["dataset"]["kind"] == "chat" and p["dataset"]["records"] == 60 and p["steps"]["total"] >= 1
    assert p["vram"]["total_mb"] > 0 and p["time"] and not [w for w in p["warnings"] if "lowered" in w]


def test_plan_counts_steps_from_the_batch_the_accumulation_and_the_epochs(ctx, base_model, dataset):
    p = plan(ctx, base_model, dataset, batch=2, grad_accum=4, epochs=3)
    n_train = p["dataset"]["train"]
    assert p["steps"]["per_epoch"] == -(-n_train // 8) and p["steps"]["total"] == p["steps"]["per_epoch"] * 3
    assert plan(ctx, base_model, dataset, max_steps=7)["steps"]["total"] == 7


def test_plan_warns_about_tiny_runs(ctx, base_model, dataset):
    warnings = plan(ctx, base_model, dataset)["warnings"]
    assert any("Fewer than 20 optimizer steps" in w for w in warnings) and any("Only" in w and "training records" in w for w in warnings)


def test_plan_lowers_seq_len_until_a_big_model_fits_but_never_when_asked(ctx, big_base, dataset):
    auto = plan(ctx, big_base, dataset, batch=4)
    assert auto["params"]["seq_len"] < 4096 or auto["vram"]["total_mb"] <= 16311 * 0.92
    assert auto["pick"]["fits"] is True
    pinned = plan(ctx, big_base, dataset, batch=4, seq_len=8192)
    assert pinned["params"]["seq_len"] == 8192 and not [w for w in pinned["warnings"] if "lowered" in w]


def test_plan_limits_the_default_sequence_to_the_models_own_context(ctx, dataset):
    small = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--short", context=1024), "acme/short")
    assert plan(ctx, small, dataset)["params"]["seq_len"] == 1024


def test_plan_says_when_nothing_fits(ctx, dataset):
    huge = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--huge", extra_config=HUGE), "acme/huge")
    p = plan(ctx, huge, dataset, method="lora", seq_len=4096)
    assert p["pick"]["fits"] is False and any("does not fit" in w or "Plain LoRA" in w for w in p["warnings"])


def test_plan_rejects_a_bad_method_and_an_undescribed_config(ctx, base_model, dataset, tmp_path):
    with pytest.raises(PygmalionError):
        plan(ctx, base_model, dataset, method="full")
    empty = ctx.svc.lineage.ensure_base(make_model(tmp_path / "e", extra_config={"hidden_size": 0, "num_hidden_layers": 0}), "e")
    with pytest.raises(PygmalionError) as exc:
        plan(ctx, empty, dataset)
    assert exc.value.code == "invalid" and "memory cannot be estimated" in exc.value.message


def test_plan_ignores_unknown_overrides_and_none_values(ctx, base_model, dataset):
    p = plan(ctx, base_model, dataset, evil="x", rank=None, alpha=8)
    assert "evil" not in p["params"] and p["params"]["rank"] == 16 and p["params"]["alpha"] == 8


def test_plan_without_a_gpu_inventory_says_so(tmp_path, base_model):
    from conftest import build_services
    c = build_services(tmp_path / "nogpu", inventory=[])
    try:
        folder = make_model(c.work / "hf" / "acme--x", seed=1)
        base = c.svc.lineage.ensure_base(folder, "acme/x")
        ds = c.svc.datasets.build({"name": "d", "sources": [{"type": "jsonl", "text": '{"messages":[{"role":"user","content":"hola"},{"role":"assistant","content":"adios"}]}'}]})["version"]
        p = c.svc.planner.train_plan(base["id"], ds["id"])
        assert any("No GPU inventory" in w for w in p["warnings"])
    finally:
        c.svc.stop()


def test_train_params_are_explicit_and_reproducible(ctx, base_model, dataset):
    params = ctx.svc.planner.train_params(base_model["id"], dataset["id"], None, {"rank": 8}, "mi-adaptador")
    assert params["base"] == base_model["id"] and params["dataset"] == dataset["id"] and params["rank"] == 8 and params["name"] == "mi-adaptador"
    assert {"method", "lr", "seq_len", "batch", "grad_accum", "epochs", "train_on", "target_modules"} <= set(params)


# ------------------------------------------------------------------------------------------------ train_start checks
def test_train_start_needs_the_trainer_environment(ctx, base_model, dataset):
    ctx.svc.settings.set({"env.python": str(ctx.tmp / "missing-python")})
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "train_start", base=base_model["id"], dataset=dataset["id"])
    assert exc.value.code == "env_missing" and "env.python" in exc.value.hint


def test_train_start_refuses_what_does_not_fit_unless_forced(ctx, dataset):
    huge = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--huge2", extra_config=HUGE), "acme/huge2")
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "train_start", base=huge["id"], dataset=dataset["id"], method="lora", seq_len=4096)
    assert exc.value.code == "gpu_unavailable" and "force=true" in exc.value.hint
    queued = call(ctx.svc, "train_start", base=huge["id"], dataset=dataset["id"], method="lora", seq_len=4096, force=True)
    assert queued["state"] == "queued"


def test_train_start_with_after_builds_the_pipeline_with_references(ctx, base_model, dataset):
    answer = call(ctx.svc, "train_start", base=base_model["id"], dataset=dataset["id"], name="x",
                  after={"quantize": ["Q4_K_M"], "imatrix": True, "publish": {"target": "ollama"}, "evaluate": {"intent": "style"}})
    # the parent's GGUF (the baseline) is built first so the evaluation compares like with like
    assert answer["steps"] == ["convert", "imatrix", "quantize", "train", "merge_lora", "convert", "imatrix", "quantize", "publish", "evaluate"]
    first = ctx.svc.store.job(answer["job"]["id"])
    assert first["kind"] == "convert" and first["params"]["model"] == base_model["id"] and first["params"]["_as"] == "baseline_gguf"
    rest = first["then"]
    train = rest[2]
    assert train["kind"] == "train" and train["params"]["base"] == base_model["id"] and train["params"]["name"] == "x"
    assert rest[3]["params"]["adapter"] == "$adapter" and rest[3]["params"]["base"] == base_model["id"]
    assert rest[-1]["params"]["against"] == "$baseline" and rest[-1]["params"]["artifact"] == "$quant:Q4_K_M"


def test_train_start_without_evaluation_starts_with_the_training(ctx, base_model, dataset):
    answer = call(ctx.svc, "train_start", base=base_model["id"], dataset=dataset["id"], after={"merge": True, "quantize": ["Q4_K_M"]})
    assert answer["steps"] == ["train", "merge_lora", "convert", "quantize"]
    assert ctx.svc.store.job(answer["job"]["id"])["kind"] == "train"


# ------------------------------------------------------------------------------------------------ pipelines
def test_normalise_after_decides_what_is_needed(ctx):
    n = ctx.svc.planner.normalise_after
    assert n(None)["convert"] is False and n({"quantize": ["Q4_K_M"]})["convert"] is True
    assert n({"publish": True})["convert"] is True and n({"imatrix": True})["imatrix"] is False, "an importance matrix without a quantization is pointless"
    assert n({"convert": "bf16"})["outtype"] == "bf16" and n({})["outtype"] == "f16"
    assert n({"quantize": ["q4_k_m", "Q4_K_M"], "imatrix": True})["quantize"] == ["Q4_K_M"]


def test_normalise_after_warns_about_iq_types_without_a_matrix(ctx):
    out = ctx.svc.planner.normalise_after({"quantize": ["IQ3_M"], "imatrix": False})
    assert out["warnings"] and "importance matrix" in out["warnings"][0]
    assert ctx.svc.planner.normalise_after({"quantize": ["IQ3_M"], "imatrix": True})["warnings"] == []


def test_normalise_after_rejects_unknown_quantization_types(ctx):
    with pytest.raises(PygmalionError):
        ctx.svc.planner.normalise_after({"quantize": ["Q9_Z"]})


def test_post_steps_from_a_folder(ctx):
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=None, name="mio",
                                       after={"quantize": ["Q4_K_M", "Q8_0"], "imatrix": True, "perplexity": True, "publish": {"name": "mio", "tag": "v2"},
                                              "evaluate": {"intent": "code"}})
    assert [s["kind"] for s in steps] == ["convert", "imatrix", "quantize", "quantize", "perplexity", "perplexity", "perplexity", "publish", "evaluate"]
    ppl = [s["params"] for s in steps if s["kind"] == "perplexity"]
    assert [p["gguf"] for p in ppl] == ["$gguf", "$quant:Q4_K_M", "$quant:Q8_0"], "the unquantized file and every quantization"
    assert len({(json.dumps(p["text"]), p["ctx"], p["chunks"]) for p in ppl}) == 1, "all measured on the same text, context and chunks"
    by = {(s["kind"], s["params"].get("type")): s["params"] for s in steps if s["kind"] != "perplexity"}
    assert by[("convert", None)]["model"] == "$merged" and by[("imatrix", None)]["gguf"] == "$gguf"
    assert by[("quantize", "Q4_K_M")]["imatrix"] == "$imatrix" and by[("quantize", "Q8_0")]["gguf"] == "$gguf"
    assert by[("publish", None)]["gguf"] == "$quant:Q4_K_M" and by[("publish", None)]["tag"] == "v2"
    assert by[("evaluate", None)]["artifact"] == "$quant:Q4_K_M" and by[("evaluate", None)]["intent"] == "code"


def test_post_steps_without_quantization_use_the_converted_file(ctx):
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=None, name="n", after={"publish": True, "evaluate": True})
    assert [s["kind"] for s in steps] == ["convert", "publish", "evaluate"]
    assert steps[1]["params"]["gguf"] == "$gguf" and steps[2]["params"]["artifact"] == "$gguf"


def test_post_steps_from_an_existing_gguf_skip_the_conversion(ctx):
    steps = ctx.svc.planner.post_steps(hf=None, gguf="a_123", baseline_model=None, name="n", after={"quantize": ["Q5_K_M"], "imatrix": False})
    assert [s["kind"] for s in steps] == ["quantize"] and steps[0]["params"]["gguf"] == "a_123" and steps[0]["params"]["imatrix"] is None


def test_post_steps_with_nothing_to_do(ctx):
    assert ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=None, name="n", after={}) == []


def test_baseline_steps_are_added_before_the_new_ones(ctx, base_model):
    after = {"quantize": ["Q4_K_M"], "imatrix": True, "evaluate": {"intent": "style"}}
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=base_model["id"], after=after, name="n")
    full = ctx.svc.planner.with_baseline(steps, base_model["id"], after)
    assert [s["kind"] for s in full[:3]] == ["convert", "imatrix", "quantize"] and full[0]["params"]["model"] == base_model["id"]
    assert full[2]["params"]["_as"] == "baseline" and full[-1]["kind"] == "evaluate" and full[-1]["params"]["against"] == "$baseline"
    assert len(full) == len(steps) + 3


def test_baseline_is_skipped_when_against_is_given_or_refused(ctx, base_model):
    after = {"quantize": ["Q4_K_M"], "evaluate": {"against": "a_xyz"}}
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=base_model["id"], after=after, name="n")
    assert ctx.svc.planner.with_baseline(steps, base_model["id"], after) == steps
    after = {"quantize": ["Q4_K_M"], "evaluate": {"baseline": False}}
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=base_model["id"], after=after, name="n")
    assert ctx.svc.planner.with_baseline(steps, base_model["id"], after) == steps
    assert ctx.svc.planner.with_baseline(steps, base_model["id"], {"quantize": ["Q4_K_M"]}) == steps


def test_an_existing_baseline_is_reused_not_rebuilt(ctx, base_model):
    s = ctx.svc.store
    (ctx.tmp / "f16.gguf").write_bytes(b"x")
    (ctx.tmp / "q4.gguf").write_bytes(b"x")
    f16 = s.create_artifact("gguf", "tiny-f16", path=str(ctx.tmp / "f16.gguf"), parents=[base_model["id"]])
    q4 = s.create_artifact("gguf", "tiny-q4", path=str(ctx.tmp / "q4.gguf"), parents=[f16["id"]], recipe={"params": {"qtype": "Q4_K_M"}}, metrics={"quant": "Q4_K_M"})
    refs = ctx.svc.planner.references
    assert refs.find(base_model["id"], {"qtype": "", "outtype": "f16"})["id"] == f16["id"]
    assert refs.find(base_model["id"], {"qtype": "q4_k_m", "outtype": "f16"})["id"] == q4["id"]
    assert refs.find(base_model["id"], {"qtype": "Q8_0", "outtype": "f16"}) is None
    after = {"quantize": ["Q4_K_M"], "evaluate": True}
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=base_model["id"], after=after, name="n")
    full = ctx.svc.planner.with_baseline(steps, base_model["id"], after)
    assert len(full) == len(steps) and full[-1]["params"]["against"] == q4["id"]


def test_a_baseline_is_rebuilt_when_its_file_is_gone_but_the_unquantized_file_is_reused(ctx, base_model):
    s = ctx.svc.store
    (ctx.tmp / "f16.gguf").write_bytes(b"x")
    f16 = s.create_artifact("gguf", "tiny-f16", path=str(ctx.tmp / "f16.gguf"), parents=[base_model["id"]])
    s.create_artifact("gguf", "tiny-q4", path=str(ctx.tmp / "gone.gguf"), parents=[f16["id"]], recipe={"params": {"qtype": "Q4_K_M"}}, metrics={"quant": "Q4_K_M"})
    after = {"quantize": ["Q4_K_M"], "evaluate": True}
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=base_model["id"], after=after, name="n")
    full = ctx.svc.planner.with_baseline(steps, base_model["id"], after)
    assert [x["kind"] for x in full[:1]] == ["quantize"] and full[0]["params"]["gguf"] == f16["id"] and full[-1]["params"]["against"] == "$baseline"


def test_a_lora_gguf_is_never_taken_as_a_baseline(ctx, base_model):
    ctx.svc.store.create_artifact("gguf", "lora-gguf", path="/m/l.gguf", parents=[base_model["id"]], recipe={"params": {"adapter": True}})
    assert ctx.svc.planner.references.find(base_model["id"], {"qtype": "", "outtype": "f16"}, must_exist=False) is None


# ------------------------------------------------------------------------------------------------ operations
def test_merge_lora_needs_an_adapter_and_a_base(ctx, base_model):
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "merge_lora_start", base=base_model["id"], adapter=base_model["id"])
    assert exc.value.code == "invalid" and "adapter" in exc.value.message


def test_merge_check_reports_compatibility(ctx, base_model):
    other = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--same", seed=4), "acme/same")
    deep = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--deeper", layers=3), "acme/deeper")
    ok = call(ctx.svc, "merge_check", models=[base_model["id"], other["id"]])
    assert ok["ok"] is True and [m["name"] for m in ok["models"]] == ["acme/tiny-base", "acme/same"]
    bad = call(ctx.svc, "merge_check", models=[base_model["id"], deep["id"]])
    assert bad["ok"] is False and bad["problems"]


def test_ctx_extend_by_target_length_computes_the_factor(ctx, base_model):
    answer = call(ctx.svc, "ctx_extend_start", model=base_model["id"], target_length=8192)
    assert answer["factor"] == 4.0 and answer["original"] == 2048 and answer["new_context"] == 8192


def test_ctx_extend_with_evaluation_defaults_to_the_context_suite(ctx, base_model):
    answer = call(ctx.svc, "ctx_extend_start", model=base_model["id"], factor=2, after={"evaluate": True})
    assert answer["steps"] == ["convert", "ctx_extend", "convert", "evaluate"], "baseline conversion of the original, then the variant and its GGUF"
    first = ctx.svc.store.job(answer["job"]["id"])
    ev = first["then"][-1]["params"]
    assert ev["intent"] == "context" and ev["artifact"] == "$gguf" and ev["against"] == "$baseline"
    assert first["then"][0]["params"]["model"] == base_model["id"] and first["then"][1]["params"]["model"] == "$ctx_variant"


def test_quantize_needs_a_source_and_warns_about_iq_without_a_matrix(ctx, base_model):
    with pytest.raises(PygmalionError):
        call(ctx.svc, "quantize_start", types=["Q4_K_M"])
    answer = call(ctx.svc, "quantize_start", model=base_model["id"], types=["IQ3_M"], imatrix=False)
    assert answer["warnings"] and answer["steps"] == ["convert", "quantize"]


def test_quantize_start_from_a_gguf_artifact_only_queues_quantization(ctx):
    gguf = ctx.svc.store.create_artifact("gguf", "x-f16", path="/m/x.gguf", size=10)
    answer = call(ctx.svc, "quantize_start", gguf=gguf["id"], types=["Q4_K_M", "Q8_0"], imatrix=True, perplexity=True)
    assert answer["steps"] == ["imatrix", "quantize", "quantize", "perplexity", "perplexity", "perplexity"]
    with pytest.raises(PygmalionError):
        call(ctx.svc, "quantize_start", gguf=ctx.svc.store.create_artifact("adapter", "ad")["id"])


def test_convert_start_validates_the_kind_of_what_it_gets(ctx, base_model):
    with pytest.raises(PygmalionError):
        call(ctx.svc, "convert_start", model=ctx.svc.store.create_artifact("gguf", "g", path="/m/g.gguf")["id"])
    answer = call(ctx.svc, "convert_start", model=base_model["id"], after={"quantize": ["Q4_K_M"], "imatrix": False}, outtype="bf16")
    assert answer["steps"] == ["convert", "quantize"]
    assert ctx.svc.store.job(answer["job"]["id"])["params"]["outtype"] == "bf16"


def test_jobs_list_filters_by_kind_and_state(ctx, base_model):
    call(ctx.svc, "ctx_extend_start", model=base_model["id"], factor=2)
    assert len(call(ctx.svc, "jobs_list", kind="ctx_extend")["jobs"]) == 1
    assert call(ctx.svc, "jobs_list", states=["done"])["jobs"] == []
    with pytest.raises(PygmalionError):
        call(ctx.svc, "jobs_list", kind="teleport")


def test_ctx_fit_tabulates_the_kv_cache_for_the_allowed_gpus(ctx, tmp_path):
    from gguf_builder import build_gguf
    path = build_gguf(tmp_path / "m.gguf", context=8192, blocks=32, heads=32, kv_heads=8, embedding=4096, size=2_000_000)
    gguf = ctx.svc.store.create_artifact("gguf", "m", path=str(path), size=path.stat().st_size)
    out = call(ctx.svc, "ctx_fit", gguf=gguf["id"], contexts=[4096, 131072])
    assert out["total_mb"] == 16311 and [r["context"] for r in out["rows"]] == [4096, 131072]
    assert out["rows"][0]["kv_mb"] < out["rows"][1]["kv_mb"] and out["rows"][0]["fits"] is True and out["rows"][1]["fits"] is False
    only = call(ctx.svc, "ctx_fit", gguf=gguf["id"], contexts=[4096], gpu=3)
    assert only["total_mb"] == 16311
    assert [r["context"] for r in call(ctx.svc, "ctx_fit", gguf=gguf["id"])["rows"]][0] == 4096


def test_base_download_reports_the_size_before_asking_for_confirmation(ctx):
    import httpx

    def hub(request):
        assert request.url.path == "/api/models/acme/dl"
        return httpx.Response(200, json={"id": "acme/dl", "gated": False, "tags": ["license:apache-2.0"], "config": {"architectures": ["LlamaForCausalLM"]},
                                         "siblings": [{"rfilename": "model.safetensors", "size": 4_000_000}, {"rfilename": "config.json", "size": 500},
                                                      {"rfilename": "model.bin", "size": 9_000_000}]})
    ctx.svc.hub.transport = httpx.MockTransport(hub)
    ctx.svc.hub.offline = False
    out = call(ctx.svc, "base_download", repo_id="acme/dl")
    assert out["started"] is False and out["license"] == "apache-2.0" and out["weights"] == 1 and out["download_bytes"] == 4_000_500
    assert "confirm=true" in out["hint"]
    started = call(ctx.svc, "base_download", repo_id="acme/dl", confirm=True)
    assert started["started"] is True and started["job"]["kind"] == "download"
    ctx.svc.jobs.run_until_idle()
    assert ctx.svc.store.job(started["job"]["id"])["state"] == "done"
    assert call(ctx.svc, "bases_list")["bases"][0]["repo_id"] == "acme/dl"


def test_base_download_refuses_repositories_without_safetensors_and_bad_names(ctx):
    import httpx
    ctx.svc.hub.transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"id": "acme/pk", "siblings": [{"rfilename": "pytorch_model.bin", "size": 5}]}))
    ctx.svc.hub.offline = False
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "base_download", repo_id="acme/pk")
    assert exc.value.code == "unsupported"
    with pytest.raises(PygmalionError):
        call(ctx.svc, "base_download", repo_id="no-slash-here")


# ------------------------------------------------------------------------------------------------ the time estimate
QWEN_06B = {"hidden_size": 1024, "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
            "intermediate_size": 3072, "vocab_size": 151936, "tie_word_embeddings": True, "max_position_embeddings": 32768}


def test_plan_time_for_a_small_qlora_counts_the_tokens_of_the_whole_run(ctx):
    """The measured defect: '9 s - 27 steps' for a QLoRA of a 0.6B model, 140 records x 3 epochs, seq 768."""
    import json
    base = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--small-0.6b", extra_config=QWEN_06B), "acme/small-0.6b")
    long_answer = "la casa de la abuela guarda muchas historias " * 40                       # about 450 tokens a record
    records = [{"messages": [{"role": "user", "content": f"Cuéntame la historia {i}."}, {"role": "assistant", "content": f"{i}: {long_answer}"}]} for i in range(140)]
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in records)
    version = ctx.svc.datasets.build({"name": "Ciento cuarenta", "sources": [{"type": "jsonl", "text": text}], "split": {"eval_pct": 0}})["version"]
    p = ctx.svc.planner.train_plan(base["id"], version["id"], None, {"method": "qlora", "seq_len": 768, "epochs": 3, "batch": 2, "grad_accum": 8})
    steps = p["steps"]["total"]
    assert steps == 3 * -(-p["dataset"]["train"] // 16) and steps >= 24
    t = p["time"]
    per_record = version["tokens"] / version["usable"]
    assert t["tokens"] == pytest.approx(steps * 16 * min(per_record, 768), rel=0.02)          # the records the steps visit, not the dataset once
    assert t["tokens_per_s"] == pytest.approx(1600 / (p["vram"]["params_b"]), rel=0.15) or t["tokens_per_s"] == 8000
    assert t["seconds"] >= t["startup_s"] + steps * 0.4 and t["seconds"] > 60          # minutes, not the 9 s that was shown
    assert t["steps"] == steps and t["source"] == "estimate" and t["formula"].key == "time_formula"


def test_plan_time_grows_with_the_epochs_and_shrinks_with_plain_lora(ctx, dataset):
    base = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--mid-0.6b", extra_config=QWEN_06B), "acme/mid-0.6b")
    one = plan(ctx, base, dataset, epochs=1, batch=1, grad_accum=2)["time"]
    five = plan(ctx, base, dataset, epochs=5, batch=1, grad_accum=2)["time"]
    assert five["tokens"] == pytest.approx(5 * one["tokens"], rel=0.05) and five["seconds"] > one["seconds"]
    lora = plan(ctx, base, dataset, epochs=5, batch=1, grad_accum=2, method="lora")["time"]
    assert lora["tokens_per_s"] == pytest.approx(1.6 * five["tokens_per_s"], rel=0.01) and lora["method_speed"] == 1.6 and lora["seconds"] < five["seconds"]


def test_plan_time_is_in_max_steps_mode_the_steps_times_the_records_per_step(ctx, base_model, dataset):
    t = plan(ctx, base_model, dataset, max_steps=10, batch=2, grad_accum=4)["time"]
    avg = min(ctx.svc.store.find_version(dataset["id"])["tokens"] / dataset["usable"], 2048)
    assert t["steps"] == 10 and t["tokens"] == pytest.approx(10 * 8 * avg, rel=0.05)


def test_train_params_carry_the_estimate_for_the_job_view_but_not_for_the_worker(ctx, base_model, dataset):
    params = ctx.svc.planner.train_params(base_model["id"], dataset["id"], None, {})
    assert set(params["_estimate"]) == {"seconds", "tokens_per_s", "tokens"} and params["_estimate"]["seconds"] > 0


def test_perplexity_of_the_base_is_measured_the_same_way_as_the_new_files(ctx, base_model):
    after = {"quantize": ["Q4_K_M"], "imatrix": True, "perplexity": True, "evaluate": {"intent": "style"}}
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=base_model["id"], after=after, name="n")
    full = ctx.svc.planner.with_baseline(steps, base_model["id"], after)
    ppl = [s["params"] for s in full if s["kind"] == "perplexity"]
    assert [p["gguf"] for p in ppl] == ["$baseline_gguf", "$baseline", "$gguf", "$quant:Q4_K_M"], "the base's f16 and quantization, then the new ones"
    assert len({(json.dumps(p["text"]), p["ctx"], p["chunks"]) for p in ppl}) == 1
    order = [s["kind"] for s in full]
    assert order.index("perplexity") > order.index("quantize"), "a file is measured only after it is built"


def test_no_perplexity_step_for_the_base_when_perplexity_is_off(ctx, base_model):
    after = {"quantize": ["Q4_K_M"], "evaluate": True}
    steps = ctx.svc.planner.post_steps(hf="$merged", gguf=None, baseline_model=base_model["id"], after=after, name="n")
    assert "perplexity" not in [s["kind"] for s in ctx.svc.planner.with_baseline(steps, base_model["id"], after)]
