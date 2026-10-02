"""The job runners end to end: fake trainer workers and fake llama.cpp programs, real merge/context workers and real lineage."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import call, finish, needs_posix
from models import make_model, read_all
from pygmalion_hoard import procs
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.jobs import JobManager


def read_env(art):
    return json.loads((Path(art["path"]) / "env.json").read_text(encoding="utf-8"))


def train(ctx, base_model, dataset, **extra):
    args = {"base": base_model["id"], "dataset": dataset["id"], "max_steps": 8, "grad_accum": 2, "save_every": 4, **extra}
    return call(ctx.svc, "train_start", **args)


def log_of(ctx, job_id):
    return (ctx.svc.work.job_dir(job_id) / "job.log").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ training
def test_training_makes_an_adapter_with_its_lineage(ctx, base_model, dataset):
    answer = train(ctx, base_model, dataset, name="abuela")
    job = finish(ctx.svc, answer)
    assert job["state"] == "done", job["error"]
    adapter = ctx.svc.store.artifact(job["out_artifact"])
    assert adapter["kind"] == "adapter" and adapter["name"] == "abuela" and adapter["parents"] == [base_model["id"]]
    assert adapter["dataset_version"] == dataset["id"]
    assert adapter["metrics"]["training"]["steps"] == 8 and adapter["metrics"]["vram_estimate"]["total_mb"] > 0
    assert (Path(adapter["path"]) / "adapter_model.safetensors").is_file()
    assert job["progress"]["pct"] == 100.0 and job["vram"]["total_mb"] > 0


def test_a_training_job_keeps_the_plans_estimate_and_reports_the_measured_speed(ctx, base_model, dataset):
    """While it runs the estimate is shown; the worker's measured tokens/s and real ETA replace it; when done the elapsed time sits next to the estimate."""
    answer = train(ctx, base_model, dataset)
    job = ctx.svc.store.job(answer["job"]["id"])
    queued = ctx.svc.jobs.view(job)["timing"]
    assert queued["source"] == "estimate" and queued["eta_s"] == job["params"]["_estimate"]["seconds"] > 0
    finished = finish(ctx.svc, answer)
    assert finished["state"] == "done"
    assert finished["progress"]["avg_tokens_per_s"] == 1450 and finished["progress"]["tokens_per_s"] == 1500
    timing = ctx.svc.jobs.view(finished)["timing"]
    assert timing["source"] == "done" and timing["tokens_per_s"] == 1450 and timing["estimated_s"] == job["params"]["_estimate"]["seconds"]
    assert "_estimate" not in ctx.svc.jobs.view(finished, detail=True)["params"]


def test_training_gets_an_allowed_gpu_through_the_lease_and_runs_offline(ctx, base_model, dataset):
    job = finish(ctx.svc, train(ctx, base_model, dataset))
    env = read_env(ctx.svc.store.artifact(job["out_artifact"]))
    assert set(env["CUDA_VISIBLE_DEVICES"].split(",")) <= {"2", "3"} and env["HF_HUB_OFFLINE"] == "1"
    assert ctx.leases.requests and ctx.leases.released, "the lease was requested and given back"
    assert all(r.get("gpu") in (2, 3) or r.get("gpu") is None for r in ctx.leases.requests)


def test_training_passes_the_planned_parameters_to_the_worker(ctx, base_model, dataset):
    job = finish(ctx.svc, train(ctx, base_model, dataset, rank=4, alpha=8, lr=0.0001, seq_len=512))
    args = read_env(ctx.svc.store.artifact(job["out_artifact"]))["args"]
    assert (args["rank"], args["alpha"], args["lr"], args["seq_len"], args["method"]) == (4, 8, 0.0001, 512, "qlora")
    assert args["resume"] is False and Path(args["train_path"]).is_file() and args["train_on"] == "assistant"


def test_training_failure_keeps_the_hint_releases_the_lease_and_resumes(ctx, base_model, dataset, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_AT", "6")
    job = finish(ctx.svc, train(ctx, base_model, dataset, max_steps=10))
    assert job["state"] == "failed" and "out of memory" in job["error"].lower() and "batch" in job["hint"].lower()
    assert len(ctx.leases.released) == len(ctx.leases.acquired) >= 1
    monkeypatch.delenv("FAKE_FAIL_AT")
    call(ctx.svc, "job_resume", job=job["id"])
    ctx.svc.jobs.run_until_idle()
    again = ctx.svc.store.job(job["id"])
    assert again["state"] == "done" and again["attempts"] == 2
    env = read_env(ctx.svc.store.artifact(again["out_artifact"]))
    assert env["start"] == 4 and env["args"]["resume"] is True


@pytest.mark.parametrize("signal_arrives", [True, False], ids=["with-signal", "stop-file-only"])
def test_cancelling_a_training_run_saves_a_checkpoint_and_it_can_continue(ctx, base_model, dataset, monkeypatch, signal_arrives):
    """``stop-file-only`` is the Windows case: a worker started without a console never receives Ctrl+Break, so only the stop file asks it to stop."""
    monkeypatch.setenv("FAKE_DELAY", "0.15")
    if not signal_arrives:
        monkeypatch.setattr(procs, "request_stop", lambda proc: None)
    live = JobManager(ctx.svc.jobs.deps, ctx.svc.jobs.runners, enabled=True)
    live.start()
    try:
        answer = call(ctx.svc, "train_start", base=base_model["id"], dataset=dataset["id"], max_steps=300, save_every=100)
        job_id = answer["job"]["id"]
        for _ in range(200):
            if ctx.svc.store.job(job_id)["state"] == "running" and ctx.svc.store.job(job_id)["progress"].get("step", 0) >= 2:
                break
            time.sleep(0.05)
        live.cancel(job_id)
        final = live.wait(job_id, timeout=20)
    finally:
        live.stop()
    assert final["state"] == "cancelled"
    assert list(ctx.work.rglob("training_state.json")), "a checkpoint was written before the worker stopped"
    assert ctx.leases.released


def test_a_dataset_without_training_records_is_refused(ctx, base_model):
    empty = ctx.svc.datasets.build({"name": "vacio", "sources": [{"type": "jsonl", "text": json.dumps({"messages": [{"role": "user", "content": "hola"},
                                                                                                                      {"role": "assistant", "content": "adiós"}]})}]})["version"]
    ctx.svc.datasets.review(empty["id"], reject=[0])
    job = finish(ctx.svc, call(ctx.svc, "train_start", base=base_model["id"], dataset="vacio", max_steps=4))
    assert job["state"] == "failed" and "no training records" in job["error"]


# ------------------------------------------------------------------------------------------------ merges and context
def test_train_then_merge_folds_the_adapter_into_the_base(ctx, base_model, dataset):
    answer = train(ctx, base_model, dataset, rank=4, alpha=8, after={"merge": True})
    ctx.svc.jobs.run_until_idle()
    jobs = ctx.svc.store.jobs(pipeline_id=answer["job"]["pipeline_id"], oldest_first=True)
    assert [j["kind"] for j in jobs] == ["train", "merge_lora"] and all(j["state"] == "done" for j in jobs), [(j["kind"], j["error"]) for j in jobs]
    adapter, merged = (ctx.svc.store.artifact(j["out_artifact"]) for j in jobs)
    assert merged["kind"] == "merged" and set(merged["parents"]) == {base_model["id"], adapter["id"]} and merged["dataset_version"] == dataset["id"]
    before, after = read_all(Path(base_model["path"])), read_all(Path(merged["path"]))
    name = "model.layers.0.self_attn.q_proj.weight"
    delta = 2.0 * (4 * 0.02 * 0.01)          # scale alpha/r = 2; B and A are constant matrices of 0.02 and 0.01
    np.testing.assert_allclose(after[name] - before[name], delta, atol=1e-6)
    np.testing.assert_array_equal(after["model.layers.1.self_attn.q_proj.weight"], before["model.layers.1.self_attn.q_proj.weight"])
    assert merged["metrics"]["merge"]["mode"] == "stream" and (Path(merged["path"]) / "config.json").is_file()


def test_merge_models_averages_two_models(ctx, base_model):
    other = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--other", seed=2), "acme/other")
    answer = call(ctx.svc, "merge_models_start", models=[base_model["id"], other["id"]], method="linear", weights=[0.5, 0.5], name="mezcla")
    job = finish(ctx.svc, answer)
    assert job["state"] == "done", job["error"]
    merged = ctx.svc.store.artifact(job["out_artifact"])
    assert merged["kind"] == "merged" and merged["name"] == "mezcla" and set(merged["parents"]) == {base_model["id"], other["id"]}
    a, b, m = (read_all(Path(x["path"])) for x in (base_model, other, merged))
    name = "model.layers.1.mlp.up_proj.weight"
    np.testing.assert_allclose(m[name], (a[name] + b[name]) / 2, atol=1e-6)


def test_merge_models_refuses_incompatible_shapes(ctx, base_model):
    big = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--deep", seed=3, layers=3), "acme/deep")
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "merge_models_start", models=[base_model["id"], big["id"]])
    assert exc.value.code == "invalid" and "not compatible" in exc.value.message


def test_context_extension_writes_yarn_and_keeps_the_weights(ctx, base_model):
    job = finish(ctx.svc, call(ctx.svc, "ctx_extend_start", model=base_model["id"], factor=4))
    assert job["state"] == "done", job["error"]
    variant = ctx.svc.store.artifact(job["out_artifact"])
    assert variant["kind"] == "ctx_variant" and variant["parents"] == [base_model["id"]]
    config = json.loads((Path(variant["path"]) / "config.json").read_text(encoding="utf-8"))
    scaling = config.get("rope_scaling") or config.get("rope_parameters")
    assert scaling["factor"] == 4 and config["max_position_embeddings"] == 2048 * 4
    np.testing.assert_array_equal(read_all(Path(variant["path"]))["lm_head.weight"], read_all(Path(base_model["path"]))["lm_head.weight"])


def test_context_extension_needs_a_factor_above_one(ctx, base_model):
    with pytest.raises(Exception):      # the argument model already requires a factor above 1
        call(ctx.svc, "ctx_extend_start", model=base_model["id"], factor=1.0)


# ------------------------------------------------------------------------------------------------ download
def test_download_registers_the_base_and_never_logs_the_token(ctx):
    token = "hf_secrettoken123456"
    ctx.svc.set_secret("hf.token", token)
    job = ctx.svc.jobs.submit("download", {"repo_id": "acme/fresh-model"}, title="download")
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(job["id"])
    assert job["state"] == "done", job["error"]
    base = ctx.svc.store.artifact(job["out_artifact"])
    assert base["kind"] == "base" and base["name"] == "acme/fresh-model" and (Path(base["path"]) / "config.json").is_file()
    assert (Path(base["path"]) / "token-seen.txt").read_text(encoding="utf-8") == token, "the worker received the token"
    text = log_of(ctx, job["id"])
    assert token not in text and "using token ***" in text
    assert token not in json.dumps(ctx.svc.store.job(job["id"]), default=str)


def test_download_failure_explains_the_gated_repository(ctx, monkeypatch):
    monkeypatch.setenv("FAKE_DOWNLOAD_FAIL", "1")
    job = ctx.svc.jobs.submit("download", {"repo_id": "acme/private"})
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(job["id"])
    assert job["state"] == "failed" and "gated" in job["hint"].lower()


def test_download_rejects_a_bad_repository_name(ctx):
    job = ctx.svc.jobs.submit("download", {"repo_id": "../../etc/passwd"})
    ctx.svc.jobs.run_until_idle()
    assert ctx.svc.store.job(job["id"])["state"] == "failed"


# ------------------------------------------------------------------------------------------------ GGUF
@needs_posix
def test_convert_makes_a_gguf_with_metadata_and_parent(ctx, base_model):
    job = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"], outtype="f16"))
    assert job["state"] == "done", job["error"]
    gguf = ctx.svc.store.artifact(job["out_artifact"])
    assert gguf["kind"] == "gguf" and gguf["parents"] == [base_model["id"]] and gguf["path"].endswith(".gguf")
    assert gguf["metrics"]["gguf"]["context_length"] == 2048 and gguf["metrics"]["gguf"]["block_count"] == 2
    assert gguf["recipe"]["params"]["outtype"] == "f16"


@needs_posix
def test_convert_fails_clearly_when_the_converter_fails(ctx, base_model, monkeypatch):
    monkeypatch.setenv("FAKE_CONVERT_FAIL", "1")
    job = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))
    assert job["state"] == "failed" and "conversion failed" in job["error"] and "not supported" in job["error"]


@needs_posix
def test_convert_refuses_an_architecture_the_converter_does_not_know(ctx):
    odd = ctx.svc.lineage.ensure_base(make_model(ctx.work / "hf" / "acme--odd", arch="MysteryForCausalLM"), "acme/odd")
    job = finish(ctx.svc, call(ctx.svc, "convert_start", model=odd["id"]))
    assert job["state"] == "failed" and "MysteryForCausalLM" in job["error"] and "llama.cpp" in job["hint"]


@needs_posix
def test_convert_reports_a_missing_script(ctx, base_model):
    ctx.svc.settings.set({"llama.src_dir": str(ctx.tmp / "nowhere")})
    job = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))
    assert job["state"] == "failed" and "convert_hf_to_gguf.py" in job["error"] and "llama.src_dir" in job["hint"]


@needs_posix
def test_convert_a_lora_adapter_to_a_gguf_adapter(ctx, base_model, dataset):
    adapter = ctx.svc.store.artifact(finish(ctx.svc, train(ctx, base_model, dataset))["out_artifact"])
    job = finish(ctx.svc, call(ctx.svc, "convert_start", adapter=adapter["id"]))
    assert job["state"] == "done", job["error"]
    out = ctx.svc.store.artifact(job["out_artifact"])
    assert out["kind"] == "gguf" and out["parents"][0] == adapter["id"] and "lora" in out["name"].lower()


@needs_posix
def test_a_context_variant_gguf_notes_missing_rope_scaling(ctx, base_model):
    variant = finish(ctx.svc, call(ctx.svc, "ctx_extend_start", model=base_model["id"], factor=2))["out_artifact"]
    job = finish(ctx.svc, call(ctx.svc, "convert_start", model=variant))
    gguf = ctx.svc.store.artifact(job["out_artifact"])
    assert gguf["metrics"]["gguf"]["rope_scaling_type"] == "yarn" and "note" not in gguf["metrics"]


@needs_posix
def test_quantize_pipeline_with_importance_matrix_keeps_lineage(ctx, base_model):
    answer = call(ctx.svc, "quantize_start", model=base_model["id"], types=["Q4_K_M", "Q8_0"], imatrix=True)
    ctx.svc.jobs.run_until_idle()
    jobs = ctx.svc.store.jobs(pipeline_id=answer["job"]["pipeline_id"], oldest_first=True)
    assert [j["kind"] for j in jobs] == ["convert", "imatrix", "quantize", "quantize"] and all(j["state"] == "done" for j in jobs), [(j["kind"], j["error"]) for j in jobs]
    f16, matrix, q4, q8 = (ctx.svc.store.artifact(j["out_artifact"]) for j in jobs)
    assert matrix["kind"] == "imatrix" and matrix["parents"] == [f16["id"]]
    assert q4["parents"] == [f16["id"], matrix["id"]] and q4["metrics"]["quant"] == "Q4_K_M" and q4["metrics"]["imatrix"] is True
    assert q8["metrics"]["quant"] == "Q8_0" and q4["size"] < f16["size"]
    assert "IMATRIX" in Path(matrix["path"]).read_text(encoding="utf-8")
    assert "cuda=2" in Path(matrix["path"]).read_text(encoding="utf-8") or "cuda=3" in Path(matrix["path"]).read_text(encoding="utf-8")


@needs_posix
def test_imatrix_calibrated_on_the_users_own_dataset(ctx, base_model, dataset):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    answer = call(ctx.svc, "quantize_start", gguf=gguf, types=["Q4_K_M"], imatrix=True, calibration={"source": "dataset", "dataset": dataset["id"]})
    ctx.svc.jobs.run_until_idle()
    jobs = ctx.svc.store.jobs(pipeline_id=answer["job"]["pipeline_id"], oldest_first=True)
    matrix = ctx.svc.store.artifact(jobs[0]["out_artifact"])
    assert matrix["metrics"]["calibration"]["source"] == "dataset+bundled"      # 60 short records are padded with the bundled text


@needs_posix
def test_failing_imatrix_releases_the_gpu_and_explains(ctx, base_model, monkeypatch):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    monkeypatch.setenv("FAKE_IMATRIX_FAIL", "1")
    answer = call(ctx.svc, "quantize_start", gguf=gguf, types=["Q4_K_M"], imatrix=True)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.jobs(pipeline_id=answer["job"]["pipeline_id"], oldest_first=True)[0]
    assert job["state"] == "failed" and "imatrix failed" in job["error"] and "memory" in job["hint"]
    assert len(ctx.leases.released) == len(ctx.leases.acquired) >= 1


@needs_posix
def test_failing_quantize_leaves_no_artifact(ctx, base_model, monkeypatch):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    monkeypatch.setenv("FAKE_QUANT_FAIL", "1")
    before = len(ctx.svc.store.artifacts(kind="gguf"))
    answer = call(ctx.svc, "quantize_start", gguf=gguf, types=["Q5_K_M"], imatrix=False)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["state"] == "failed" and "llama-quantize failed" in job["error"]
    assert len(ctx.svc.store.artifacts(kind="gguf")) == before


@needs_posix
def test_perplexity_is_stored_on_the_artifact(ctx, base_model, monkeypatch):
    monkeypatch.setenv("FAKE_PPL", "9.5")
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    job = finish(ctx.svc, call(ctx.svc, "perplexity_start", gguf=gguf, chunks=3))
    assert job["state"] == "done", job["error"]
    assert job["result"]["value"] == 9.5 and job["result"]["error"] == pytest.approx(0.04321)
    assert ctx.svc.store.artifact(gguf)["metrics"]["ppl"]["value"] == 9.5


@needs_posix
def test_tools_missing_is_reported_with_the_setting_to_change(ctx, base_model):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    ctx.svc.settings.set({"llama.bin_dir": str(ctx.tmp / "empty")})
    answer = call(ctx.svc, "quantize_start", gguf=gguf, types=["Q4_K_M"], imatrix=False)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["state"] == "failed" and "llama-quantize was not found" in job["error"] and "llama.bin_dir" in job["hint"]


# ------------------------------------------------------------------------------------------------ publishing
def ollama_calls(ctx):
    path = Path(ctx.tools["ollama_log"]) if Path(ctx.tools["ollama_log"]).exists() else ctx.tmp / "ollama-calls.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


@needs_posix
def test_publish_to_ollama_creates_a_prefixed_tag_and_unpublish_removes_it(ctx, base_model):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    answer = call(ctx.svc, "publish_ollama", gguf=gguf, name="Casa Abuela", tag="v1", num_ctx=4096, wait_s=30)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["state"] == "done", job["error"]
    pub = ctx.svc.store.artifact(job["result"]["ollama"]["artifact"])
    assert pub["kind"] == "ollama" and pub["name"] == "pyg-casa-abuela:v1" and pub["parents"] == [gguf]
    create = [c for c in ollama_calls(ctx) if c["cmd"] == "create"][0]
    assert create["name"] == "pyg-casa-abuela:v1" and "PARAMETER num_ctx 4096" in create["modelfile"] and create["modelfile"].startswith("FROM ")
    assert ctx.svc.store.artifact(gguf)["published"][0]["name"] == "pyg-casa-abuela:v1"
    call(ctx.svc, "unpublish", artifact=gguf, confirm=True)
    assert [c["name"] for c in ollama_calls(ctx) if c["cmd"] == "rm"] == ["pyg-casa-abuela:v1"]
    assert ctx.svc.store.artifact(gguf)["published"] == []


@needs_posix
def test_unpublish_needs_confirmation(ctx, base_model):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    call(ctx.svc, "publish_ollama", gguf=gguf, name="x", wait_s=30)
    ctx.svc.jobs.run_until_idle()
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "unpublish", artifact=gguf)
    assert exc.value.code == "confirm_required"


@needs_posix
def test_ollama_failure_is_reported_and_nothing_is_marked_published(ctx, base_model, monkeypatch):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    monkeypatch.setenv("FAKE_OLLAMA_FAIL", "1")
    answer = call(ctx.svc, "publish_ollama", gguf=gguf, name="roto", wait_s=30)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["state"] == "failed" and "ollama create failed" in job["error"]
    assert ctx.svc.store.artifact(gguf)["published"] == [] and not ctx.svc.store.artifacts(kind="ollama")


@needs_posix
def test_publish_to_llama_writes_a_hub_backend_that_is_never_started(ctx, base_model):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    answer = call(ctx.svc, "publish_llama", gguf=gguf, name="Abuela", ctx=8192, wait_s=30)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["state"] == "done", job["error"]
    entry = job["result"]["llama"]["entry"]
    assert entry["id"] == "pyg-abuela" and entry["env"]["CUDA_VISIBLE_DEVICES"] in ("2", "3") and "-c" in entry["argv"] and entry["argv"][entry["argv"].index("-c") + 1] == "8192"
    assert entry["health"].endswith("/health") and job["result"]["llama"]["port"] >= ctx.svc.settings.int("publish.llama_port_start")
    backends = json.loads((Path(ctx.tmp) / "hoard-home" / "backends.json").read_text(encoding="utf-8"))
    assert [c["id"] for c in backends["commands"]] == ["pyg-abuela"]
    call(ctx.svc, "unpublish", artifact=gguf, target="llama", confirm=True)
    assert json.loads((Path(ctx.tmp) / "hoard-home" / "backends.json").read_text(encoding="utf-8"))["commands"] == []


@needs_posix
def test_publish_llama_refuses_a_gpu_that_is_not_allowed(ctx, base_model):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    answer = call(ctx.svc, "publish_llama", gguf=gguf, gpu=0, wait_s=30)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["state"] == "failed" and "not in the allowed list" in job["error"]


@needs_posix
def test_the_full_recipe_from_one_training_call(ctx, base_model, dataset):
    answer = train(ctx, base_model, dataset, name="receta", after={"merge": True, "quantize": ["Q4_K_M"], "imatrix": True, "publish": {"target": "ollama", "name": "receta"}})
    ctx.svc.jobs.run_until_idle()
    jobs = ctx.svc.store.jobs(pipeline_id=answer["job"]["pipeline_id"], oldest_first=True)
    assert [j["kind"] for j in jobs] == ["train", "merge_lora", "convert", "imatrix", "quantize", "publish"], [(j["kind"], j["state"], j["error"]) for j in jobs]
    assert all(j["state"] == "done" for j in jobs), [(j["kind"], j["error"]) for j in jobs]
    tag = ctx.svc.store.artifacts(kind="ollama")[0]
    chain = [a["kind"] for a in ctx.svc.lineage.ancestors(tag["id"])]
    for kind in ("gguf", "imatrix", "merged", "adapter", "base"):
        assert kind in chain
    recipe = call(ctx.svc, "artifact_recipe", artifact=tag["id"])
    assert recipe["steps"][0]["kind"] == "base" and recipe["steps"][-1]["kind"] == "ollama"
    assert dataset["id"] in json.dumps(recipe)


def test_a_resumed_pipeline_keeps_going_after_the_failed_step(ctx, base_model, dataset, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_AT", "3")
    answer = train(ctx, base_model, dataset, after={"merge": True}, max_steps=6)
    ctx.svc.jobs.run_until_idle()
    first = ctx.svc.store.job(answer["job"]["id"])
    assert first["state"] == "failed" and len(ctx.svc.store.jobs(pipeline_id=first["pipeline_id"])) == 1
    monkeypatch.delenv("FAKE_FAIL_AT")
    call(ctx.svc, "job_resume", job=first["id"])
    ctx.svc.jobs.run_until_idle()
    assert [j["state"] for j in ctx.svc.store.jobs(pipeline_id=first["pipeline_id"], oldest_first=True)] == ["done", "done"]


@needs_posix
def test_every_quantization_and_the_unquantized_file_get_a_perplexity_with_its_error(ctx, base_model, monkeypatch):
    monkeypatch.setenv("FAKE_PPL", "8.25")
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    answer = call(ctx.svc, "quantize_start", gguf=gguf, types=["Q4_K_M", "Q8_0"], imatrix=False, perplexity=True)
    ctx.svc.jobs.run_until_idle()
    jobs = ctx.svc.store.jobs(pipeline_id=answer["job"]["pipeline_id"], oldest_first=True)
    assert [j["kind"] for j in jobs] == ["quantize", "quantize", "perplexity", "perplexity", "perplexity"] and all(j["state"] == "done" for j in jobs), [j["error"] for j in jobs]
    measured = [a for a in ctx.svc.store.artifacts(kind="gguf") if (a["metrics"].get("ppl") or {}).get("value") == 8.25]
    assert len(measured) == 3, "the f16 file and both quantizations"
    one = measured[0]["metrics"]["ppl"]
    assert one["error"] == pytest.approx(0.04321) and one["ctx"] and one["chunks"] and (one.get("text") or one.get("text_tag"))
    assert len({(m["metrics"]["ppl"]["ctx"], m["metrics"]["ppl"]["chunks"]) for m in measured}) == 1, "measured the same way, so the numbers compare"
    rows = ctx.svc.lineage.ppl_family(ctx.svc.store.artifact(measured[0]["id"]))
    assert len(rows) == 3 and any("ppl_cmp" in r for r in rows)


@needs_posix
def test_a_measurement_that_is_not_kept_is_an_error_not_a_success(ctx, base_model, monkeypatch):
    monkeypatch.setenv("FAKE_PPL", "9.5")
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    monkeypatch.setattr(ctx.svc.lineage, "add_metrics", lambda *a, **k: None)
    job = finish(ctx.svc, call(ctx.svc, "perplexity_start", gguf=gguf, chunks=3))
    assert job["state"] == "failed" and "not" in job["error"].lower()
