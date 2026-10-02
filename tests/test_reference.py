"""What an evaluation compares with: the base model the training started from, in the same quantization as the result (and with an importance
matrix on the same calibration text), built and kept in the lineage when it does not exist; the parent file only for a plain quantization,
a context extension compares with the model before it, and an explicit ``against`` always wins.

The lineages here are written by hand in the shapes the runners write them (``tests/test_runners.py`` checks those); the pipelines run with the
fake llama.cpp programs and a fake Galton."""

import json

import pytest

from conftest import call, needs_posix
from pygmalion_hoard import calib
from pygmalion_hoard.errors import PygmalionError
from test_evaluate import DatasetGalton, FakeGalton, attach, qa

BUNDLED = {"source": "bundled"}


def make_file(ctx, name):
    path = ctx.tmp / "files" / name
    path.parent.mkdir(exist_ok=True)
    path.write_bytes(b"gguf")
    return str(path)


def matrix_of(ctx, f16, name, spec=BUNDLED, chunks=8, dataset_version=None):
    return ctx.svc.store.create_artifact(
        "imatrix", name, path=make_file(ctx, name + ".dat"), parents=[f16["id"]], dataset_version=dataset_version,
        recipe={"job_kind": "imatrix", "params": {"gguf": f16["name"], "calibration": spec, "chunks": chunks}},
        metrics={"calibration": {"source": spec.get("source"), "tag": calib.tag_of(spec)}, "chunks": chunks})


def quantized(ctx, f16, name, qtype="Q4_K_M", matrix=None, dataset_version=None):
    return ctx.svc.store.create_artifact(
        "gguf", name, path=make_file(ctx, name + ".gguf"), parents=[f16["id"]] + ([matrix["id"]] if matrix else []), dataset_version=dataset_version,
        recipe={"job_kind": "quantize", "params": {"gguf": f16["name"], "qtype": qtype}, "qtype": qtype, "imatrix": matrix["id"] if matrix else None},
        metrics={"quant": qtype, "imatrix": bool(matrix)})


def unquantized(ctx, model, name, outtype="f16", dataset_version=None):
    return ctx.svc.store.create_artifact("gguf", name, path=make_file(ctx, name + ".gguf"), parents=[model["id"]], dataset_version=dataset_version,
                                         recipe={"job_kind": "convert", "params": {"outtype": outtype}})


def dataset_version(ctx, name="Casa"):
    try:
        return ctx.svc.store.find_version(name, None)
    except PygmalionError:
        pass
    spec = {"name": name, "sources": [{"type": "jsonl", "text": "\n".join(json.dumps(r, ensure_ascii=False) for r in qa(60))}]}
    return ctx.svc.datasets.build(spec)["version"]


class Tuned:
    """base -> adapter -> merged -> f16 -> [importance matrix] -> quantized, as a training pipeline leaves them."""


def tuned(ctx, base, *, qtype="Q4_K_M", spec=BUNDLED, chunks=8, trained=True, version=None):
    t = Tuned()
    store = ctx.svc.store
    vid = (version or dataset_version(ctx))["id"] if trained else None
    if trained:
        t.adapter = store.create_artifact("adapter", "lora", path=make_file(ctx, "lora"), parents=[base["id"]], dataset_version=vid)
        t.merged = store.create_artifact("merged", "tuned", path=base["path"], parents=[base["id"], t.adapter["id"]], dataset_version=vid)
        t.f16 = unquantized(ctx, t.merged, "tuned-f16", dataset_version=vid)
    else:
        t.adapter = t.merged = None
        t.f16 = unquantized(ctx, base, "plain-f16")
    t.matrix = matrix_of(ctx, t.f16, "tuned-imatrix", spec, chunks, vid) if spec and qtype else None
    t.child = quantized(ctx, t.f16, "tuned-q4", qtype, t.matrix, vid) if qtype else t.f16
    return t


def plan(ctx, child, **kw):
    return ctx.svc.evaluator.plan(child["id"], **kw)["reference"]


# ------------------------------------------------------------------------------------------------ the rule
@pytest.mark.parametrize("intent", ["dataset", "style", "writing", "code", "general", "smoke"])
def test_intents_that_ask_whether_the_training_helped_compare_with_the_base_at_the_same_quantization(ctx, base_model, intent):
    t = tuned(ctx, base_model)
    ref = plan(ctx, t.child, intent=intent)
    assert ref["mode"] == "base" and ref["origin"]["id"] == base_model["id"] and ref["quant"] == "Q4_K_M" and ref["ready"] is False
    assert [s["kind"] for s in ref["steps"]] == ["convert", "imatrix", "quantize"], "nothing of the base exists yet: its f16, its matrix and the Q4_K_M"
    assert ref["model"] == "acme/tiny-base" and ref["message"].startswith("Compared with the base model acme/tiny-base at Q4_K_M")


def test_the_default_of_a_trained_result_is_the_base_too(ctx, base_model):
    t = tuned(ctx, base_model)
    plain = ctx.svc.evaluator.plan(t.child["id"], suites=["rapida"])
    assert plain["reference"]["mode"] == "base" and plain["parent"] is None
    assert ctx.svc.evaluator.plan(t.child["id"])["reference"]["mode"] == "base"


def test_the_base_steps_repeat_the_results_quantization_matrix_and_calibration(ctx, base_model):
    spec = {"source": "dataset", "dataset": "Casa", "n": 1}
    t = tuned(ctx, base_model, qtype="Q5_K_M", spec=spec, chunks=12)
    ref = plan(ctx, t.child, intent="code")
    convert, imatrix, quantize = ref["steps"]
    assert convert["params"] == {"model": base_model["id"], "outtype": "f16", "_as": "baseline_gguf"}
    assert imatrix["params"] == {"gguf": "$baseline_gguf", "calibration": spec, "chunks": 12, "_as": "baseline_imatrix"}
    assert quantize["params"] == {"gguf": "$baseline_gguf", "type": "Q5_K_M", "imatrix": "$baseline_imatrix", "_as": "baseline"}
    assert ref["ref"] == "$baseline" and ref["quant"] == "Q5_K_M" and ref["notes"], "the plan says the matrix uses the same calibration text"


def test_a_result_without_a_matrix_gets_a_reference_without_one(ctx, base_model):
    t = tuned(ctx, base_model, spec=None)
    ref = plan(ctx, t.child, intent="style")
    assert [s["kind"] for s in ref["steps"]] == ["convert", "quantize"] and ref["steps"][1]["params"]["imatrix"] is None and not ref["notes"]


def test_an_unquantized_result_is_compared_with_the_bases_unquantized_file(ctx, base_model):
    t = tuned(ctx, base_model, qtype=None, spec=None)
    ref = plan(ctx, t.child, intent="general")
    assert ref["mode"] == "base" and ref["quant"] == "F16" and [s["kind"] for s in ref["steps"]] == ["convert"] and ref["steps"][0]["params"]["_as"] == "baseline"
    assert ref["ref"] == "$baseline"


def test_the_outtype_of_the_results_f16_is_kept(ctx, base_model):
    store = ctx.svc.store
    t = tuned(ctx, base_model, qtype=None, spec=None)
    store.update_artifact(t.f16["id"], recipe={"job_kind": "convert", "params": {"outtype": "bf16"}})
    ref = plan(ctx, t.f16, intent="general")
    assert ref["quant"] == "BF16" and ref["steps"][0]["params"]["outtype"] == "bf16"


def test_a_plain_quantization_keeps_comparing_with_its_f16_parent(ctx, base_model):
    t = tuned(ctx, base_model, trained=False)
    for intent in ("", "quant", "style", "code"):
        ref = plan(ctx, t.child, intent=intent)
        assert ref["mode"] == "parent" and ref["artifact"]["id"] == t.f16["id"] and ref["ready"] and not ref["steps"] and ref["intent"] == (intent or "quant")
    assert ctx.svc.evaluator.plan(t.child["id"])["suites"] == ["escritura-es", "instrucciones"], "the suites of an untrained result did not change"
    assert ctx.svc.evaluator.plan(t.child["id"], intent="quant")["suites"] == ["rapida", "razonamiento"]


def test_the_quant_intent_on_a_trained_result_is_the_old_behaviour_on_request(ctx, base_model):
    t = tuned(ctx, base_model)
    ref = plan(ctx, t.child, intent="quant")
    assert ref["mode"] == "parent" and ref["artifact"]["id"] == t.f16["id"] and not ref["steps"]


def test_an_explicit_against_always_wins(ctx, base_model):
    t = tuned(ctx, base_model)
    other = ctx.svc.store.create_artifact("gguf", "other", path=make_file(ctx, "other.gguf"))
    for intent in ("", "style", "dataset", "context", "quant"):
        ref = plan(ctx, t.child, intent=intent, against=other["id"]) if intent != "dataset" else plan(ctx, t.child, intent="style", against=other["id"])
        assert ref["mode"] == "explicit" and ref["artifact"]["id"] == other["id"] and not ref["steps"] and ref["ref"] == other["id"]


def test_a_context_extension_is_compared_with_the_model_before_it(ctx, base_model):
    store = ctx.svc.store
    variant = store.create_artifact("ctx_variant", "v4x", path=base_model["path"], parents=[base_model["id"]])
    f16 = unquantized(ctx, variant, "v4x-f16")
    q4 = quantized(ctx, f16, "v4x-q4", matrix=matrix_of(ctx, f16, "v4x-imatrix"))
    for intent in ("", "context"):
        ref = plan(ctx, q4, intent=intent)
        assert ref["mode"] == "context" and ref["origin"]["id"] == base_model["id"] and [s["kind"] for s in ref["steps"]] == ["convert", "imatrix", "quantize"]
        assert ref["message"].startswith("Compared with acme/tiny-base without the context extension, at Q4_K_M")
    assert ctx.svc.evaluator.plan(q4["id"])["suites"] == ["contexto-largo"]


def test_a_trained_result_whose_base_is_not_in_the_lineage_falls_back_with_a_warning(ctx, tmp_path):
    version = dataset_version(ctx)
    f16 = ctx.svc.store.create_artifact("gguf", "orphan-f16", path=make_file(ctx, "o.gguf"), dataset_version=version["id"])
    child = quantized(ctx, f16, "orphan-q4", dataset_version=version["id"])
    ref = plan(ctx, child, intent="style")
    assert ref["mode"] == "parent" and ref["artifact"]["id"] == f16["id"] and [w.key for w in ref["warnings"]] == ["eval_ref_unknown_base"]


def test_a_published_tag_takes_the_quantization_of_the_file_it_was_made_from(ctx, base_model):
    t = tuned(ctx, base_model)
    tag = ctx.svc.store.create_artifact("ollama", "pyg-tuned:v1", path="pyg-tuned:v1", parents=[t.child["id"]], dataset_version=t.child["dataset_version"])
    ref = plan(ctx, tag, intent="style")
    assert ref["mode"] == "base" and ref["quant"] == "Q4_K_M" and ref["imatrix"]["tag"] == calib.tag_of(BUNDLED)


def test_a_model_trained_twice_is_compared_with_the_model_the_last_training_started_from(ctx, base_model):
    first = tuned(ctx, base_model, qtype=None, spec=None)
    store = ctx.svc.store
    second_adapter = store.create_artifact("adapter", "lora2", path=make_file(ctx, "lora2"), parents=[first.merged["id"]], dataset_version=first.merged["dataset_version"])
    merged2 = store.create_artifact("merged", "tuned2", path=base_model["path"], parents=[first.merged["id"], second_adapter["id"]], dataset_version=first.merged["dataset_version"])
    f16 = unquantized(ctx, merged2, "tuned2-f16", dataset_version=first.merged["dataset_version"])
    child = quantized(ctx, f16, "tuned2-q4", dataset_version=first.merged["dataset_version"])
    assert plan(ctx, child, intent="style")["origin"]["id"] == first.merged["id"]


# ------------------------------------------------------------------------------------------------ what already exists is reused
def base_files(ctx, base, *, matrix_spec=BUNDLED, qtype="Q4_K_M", with_quant=True, with_matrix=True):
    f16 = unquantized(ctx, base, "tiny-f16")
    matrix = matrix_of(ctx, f16, "tiny-imatrix", matrix_spec) if with_matrix else None
    quant = quantized(ctx, f16, "tiny-q4", qtype, matrix) if with_quant else None
    return f16, matrix, quant


def test_the_bases_own_file_with_the_same_quantization_and_matrix_is_the_reference_and_nothing_is_built(ctx, base_model):
    t = tuned(ctx, base_model)
    _f16, _matrix, quant = base_files(ctx, base_model)
    ref = plan(ctx, t.child, intent="style")
    assert ref["ready"] and ref["artifact"]["id"] == quant["id"] and ref["ref"] == quant["id"] and not ref["steps"]
    assert ref["message"].startswith("Compared with the base model acme/tiny-base at Q4_K_M.")


def test_only_the_missing_steps_are_queued(ctx, base_model):
    t = tuned(ctx, base_model)
    f16, matrix, _ = base_files(ctx, base_model, with_quant=False)
    ref = plan(ctx, t.child, intent="style")
    assert [s["kind"] for s in ref["steps"]] == ["quantize"] and ref["steps"][0]["params"]["gguf"] == f16["id"] and ref["steps"][0]["params"]["imatrix"] == matrix["id"]
    ctx.svc.store.delete_artifact(matrix["id"])
    ref = plan(ctx, t.child, intent="style")
    assert [s["kind"] for s in ref["steps"]] == ["imatrix", "quantize"] and ref["steps"][0]["params"]["gguf"] == f16["id"]


def test_a_matrix_from_another_calibration_text_is_not_taken_as_the_same(ctx, base_model):
    spec = {"source": "dataset", "dataset": "Casa", "n": 1}
    t = tuned(ctx, base_model, spec=spec)
    f16, _matrix, quant = base_files(ctx, base_model, matrix_spec=BUNDLED)
    ref = plan(ctx, t.child, intent="style")
    assert ref["artifact"] is None and [s["kind"] for s in ref["steps"]] == ["imatrix", "quantize"], "the bundled matrix is another calibration: a new one on the dataset's text"
    assert ref["steps"][0]["params"]["calibration"] == spec and ref["steps"][0]["params"]["gguf"] == f16["id"] and quant["id"] != ref["ref"]


def test_a_quantization_made_without_a_matrix_is_not_the_reference_of_one_made_with_it(ctx, base_model):
    t = tuned(ctx, base_model)
    f16 = unquantized(ctx, base_model, "tiny-f16")
    quantized(ctx, f16, "tiny-q4-plain")
    assert [s["kind"] for s in plan(ctx, t.child, intent="style")["steps"]] == ["imatrix", "quantize"]
    plain = tuned(ctx, base_model, spec=None)
    assert plan(ctx, plain.child, intent="style")["ready"] is True, "...and the other way round it is the reference"


def test_a_file_that_is_gone_from_the_disk_is_built_again(ctx, base_model):
    t = tuned(ctx, base_model)
    _f16, _matrix, quant = base_files(ctx, base_model)
    import os
    os.remove(quant["path"])
    ref = plan(ctx, t.child, intent="style")
    assert ref["artifact"] is None and [s["kind"] for s in ref["steps"]] == ["quantize"]


def test_files_with_a_training_or_a_lora_are_never_the_bases_own(ctx, base_model):
    t = tuned(ctx, base_model)
    plan_ = plan(ctx, t.child, intent="style")
    assert [s["kind"] for s in plan_["steps"]] == ["convert", "imatrix", "quantize"], "the result's own f16 hangs below the merged model, not below the base"
    ctx.svc.store.create_artifact("gguf", "lora-gguf", path=make_file(ctx, "l.gguf"), parents=[base_model["id"]], recipe={"params": {"adapter": True}})
    ctx.svc.store.create_artifact("gguf", "trained", path=make_file(ctx, "t.gguf"), parents=[base_model["id"]], dataset_version=t.child["dataset_version"])
    assert [s["kind"] for s in plan(ctx, t.child, intent="style")["steps"]] == ["convert", "imatrix", "quantize"]


# ------------------------------------------------------------------------------------------------ evaluate_plan: what the interface shows
def test_evaluate_plan_is_read_only_and_says_what_will_be_used(ctx, base_model):
    t = tuned(ctx, base_model)
    before = len(ctx.svc.store.jobs())
    out = call(ctx.svc, "evaluate_plan", artifact=t.child["id"], intent="code")
    assert out["reference"]["mode"] == "base" and out["reference"]["model"] == "acme/tiny-base" and out["reference"]["quant"] == "Q4_K_M"
    assert out["reference"]["ready"] is False and out["reference"]["steps"] == ["convert", "imatrix", "quantize"] and out["suites"] == ["codigo-python"]
    assert out["reference"]["message"].key == "eval_ref_base_build" and out["reference"]["imatrix"] is True
    assert len(ctx.svc.store.jobs()) == before and not [a for a in ctx.svc.store.children(base_model["id"]) if a["kind"] == "gguf"]
    json.dumps(out)


def test_the_message_for_the_interface_is_a_catalogue_key_in_both_languages(ctx, base_model):
    from pygmalion_hoard.messages import wire
    t = tuned(ctx, base_model)
    shown = wire(call(ctx.svc, "evaluate_plan", artifact=t.child["id"], intent="style"))["reference"]["message"]
    assert shown["key"] == "eval_ref_base_build" and shown["params"] == {"model": "acme/tiny-base", "quant": "Q4_K_M"}


def test_artifact_get_carries_the_evaluate_plan_of_a_gguf_only(ctx, base_model):
    t = tuned(ctx, base_model)
    assert call(ctx.svc, "artifact_get", artifact=t.child["id"])["evaluate_plan"]["reference"]["mode"] == "base"
    assert "evaluate_plan" not in call(ctx.svc, "artifact_get", artifact=base_model["id"])
    lone = ctx.svc.store.create_artifact("gguf", "lone", path=make_file(ctx, "lone.gguf"))
    assert call(ctx.svc, "artifact_get", artifact=lone["id"])["evaluate_plan"]["error"]


def test_evaluate_plan_refuses_a_kind_that_cannot_be_evaluated(ctx, base_model):
    with pytest.raises(PygmalionError):
        call(ctx.svc, "evaluate_plan", artifact=base_model["id"])


# ------------------------------------------------------------------------------------------------ evaluate_start: the pipeline
@needs_posix
def test_evaluate_start_prepares_the_reference_then_evaluates_and_the_next_one_reuses_it(ctx, base_model):
    t = tuned(ctx, base_model)
    fake = attach(ctx, FakeGalton())
    answer = call(ctx.svc, "evaluate_start", artifact=t.child["id"], intent="style")
    assert answer["steps"] == ["convert", "imatrix", "quantize", "evaluate"] and answer["plan"]["reference"]["mode"] == "base"
    ctx.svc.jobs.run_until_idle()
    jobs = ctx.svc.store.jobs(pipeline_id=answer["job"]["pipeline_id"], oldest_first=True)
    assert [j["kind"] for j in jobs] == ["convert", "imatrix", "quantize", "evaluate"] and all(j["state"] == "done" for j in jobs), [(j["kind"], j["error"]) for j in jobs]
    f16, matrix, quant, _ev = (ctx.svc.store.artifact(j["out_artifact"]) if j["out_artifact"] else None for j in jobs)
    # in the lineage, below the base model, written like the result
    assert f16["parents"] == [base_model["id"]] and f16["dataset_version"] is None and matrix["parents"] == [f16["id"]]
    assert quant["parents"] == [f16["id"], matrix["id"]] and quant["metrics"]["quant"] == "Q4_K_M" and quant["metrics"]["imatrix"] is True
    assert matrix["metrics"]["calibration"]["tag"] == t.matrix["metrics"]["calibration"]["tag"] and matrix["metrics"]["chunks"] == 8
    # Galton ran the base's Q4_K_M against the result's, and the verdict says what it rests on
    start = fake.args_of("run_start")[0]
    assert [c["path"] for c in start["contestants"]] == [quant["path"], t.child["path"]]
    record = ctx.svc.store.artifact(t.child["id"])["metrics"]["galton"]
    assert record["parent"] == quant["id"] and record["reference"] == {"mode": "base", "model": "acme/tiny-base", "quant": "Q4_K_M", "imatrix": True}
    # the next evaluation finds it ready: one job, no steps
    again = call(ctx.svc, "evaluate_start", artifact=t.child["id"], intent="code")
    assert "steps" not in again and again["plan"]["reference"]["ready"] is True and again["plan"]["reference"]["artifact"]["id"] == quant["id"]
    ctx.svc.jobs.run_until_idle()
    assert ctx.svc.store.job(again["job"]["id"])["params"]["against"] == quant["id"]
    assert len([a for a in ctx.svc.store.children(base_model["id"]) if a["kind"] == "gguf"]) == 1


@needs_posix
def test_evaluate_start_by_dataset_compares_with_the_base_and_keeps_the_regression_check(ctx, base_model):
    t = tuned(ctx, base_model)
    fake = attach(ctx, DatasetGalton())
    answer = call(ctx.svc, "evaluate_start", artifact=t.child["id"])
    assert answer["steps"][-1] == "evaluate" and answer["plan"]["intent"] == "dataset" and answer["plan"]["reference"]["mode"] == "base"
    ctx.svc.jobs.run_until_idle()
    record = ctx.svc.store.artifact(t.child["id"])["metrics"]["galton"]
    assert record["intent"] == "dataset" and record["reference"]["mode"] == "base" and record["regression"]["suite"] == "rapida"
    first, second = fake.args_of("run_start")[0]["contestants"]
    assert first["kind"] == "gguf" and first["path"] != second["path"] and second["path"] == t.child["path"]


def test_evaluate_start_with_against_queues_one_job_and_builds_nothing(ctx, base_model):
    t = tuned(ctx, base_model)
    attach(ctx, FakeGalton())
    answer = call(ctx.svc, "evaluate_start", artifact=t.child["id"], against=t.f16["id"], intent="style")
    assert "steps" not in answer and answer["plan"]["reference"]["mode"] == "explicit"
    job = ctx.svc.store.job(answer["job"]["id"])
    assert job["kind"] == "evaluate" and job["params"]["against"] == t.f16["id"]
    ctx.svc.jobs.run_until_idle()
    assert ctx.svc.store.artifact(t.child["id"])["metrics"]["galton"]["reference"]["mode"] == "explicit"


def test_evaluate_start_needs_the_trainer_environment_only_when_it_has_to_convert(ctx, base_model):
    t = tuned(ctx, base_model)
    ctx.svc.settings.set({"env.python": str(ctx.tmp / "missing-python")})
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "evaluate_start", artifact=t.child["id"], intent="style")
    assert exc.value.code == "env_missing" and not ctx.svc.store.jobs(kind="evaluate")
    base_files(ctx, base_model, with_quant=False, with_matrix=False)
    queued = call(ctx.svc, "evaluate_start", artifact=t.child["id"], intent="style")        # only imatrix and quantize remain: no trainer needed
    assert queued["steps"] == ["imatrix", "quantize", "evaluate"]


def test_evaluate_start_names_a_base_folder_that_is_gone(ctx, base_model):
    import shutil
    t = tuned(ctx, base_model)
    shutil.rmtree(base_model["path"])
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "evaluate_start", artifact=t.child["id"], intent="style")
    assert exc.value.code == "not_found" and "acme/tiny-base" in exc.value.message and not ctx.svc.store.jobs(kind="evaluate")


def test_the_job_itself_refuses_a_missing_reference_instead_of_picking_another(ctx, base_model):
    t = tuned(ctx, base_model)
    attach(ctx, FakeGalton())
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.evaluator.run(t.child["id"], intent="style")
    assert exc.value.code == "not_found" and "acme/tiny-base" in exc.value.message and "Q4_K_M" in exc.value.message and "evaluate_start" in exc.value.hint


def test_the_job_uses_the_base_file_when_it_exists_and_records_which_reference(ctx, base_model):
    t = tuned(ctx, base_model)
    _f16, _matrix, quant = base_files(ctx, base_model)
    attach(ctx, FakeGalton())
    record = ctx.svc.evaluator.run(t.child["id"], intent="style")
    assert record["parent"] == quant["id"] and record["reference"] == {"mode": "base", "model": "acme/tiny-base", "quant": "Q4_K_M", "imatrix": True}


def test_a_warning_about_an_unknown_base_travels_with_the_verdict(ctx):
    version = dataset_version(ctx)
    f16 = ctx.svc.store.create_artifact("gguf", "orphan-f16", path=make_file(ctx, "o.gguf"), dataset_version=version["id"])
    child = quantized(ctx, f16, "orphan-q4", dataset_version=version["id"])
    attach(ctx, DatasetGalton())
    record = ctx.svc.evaluator.run(child["id"])
    assert any(getattr(w, "key", "") == "eval_ref_unknown_base" for w in record["warnings"]) and record["reference"]["mode"] == "parent"


# ------------------------------------------------------------------------------------------------ the pipelines that end in an evaluation
def test_convert_start_with_an_evaluation_builds_the_base_reference_of_a_fine_tuned_model(ctx, base_model):
    t = tuned(ctx, base_model)
    answer = call(ctx.svc, "convert_start", model=t.merged["id"], after={"quantize": ["Q4_K_M"], "imatrix": True, "evaluate": {"intent": "style"}})
    assert answer["steps"] == ["convert", "imatrix", "quantize", "convert", "imatrix", "quantize", "evaluate"]
    first = ctx.svc.store.job(answer["job"]["id"])
    assert first["params"]["model"] == base_model["id"] and first["then"][-1]["params"]["against"] == "$baseline"
    plain = call(ctx.svc, "convert_start", model=base_model["id"], after={"quantize": ["Q4_K_M"], "evaluate": True})
    assert plain["steps"] == ["convert", "quantize", "evaluate"], "an untouched base model has no other model to be measured against"


def test_the_pipeline_of_a_training_reuses_a_matching_base_file(ctx, base_model, dataset):
    _f16, _matrix, quant = base_files(ctx, base_model)
    answer = call(ctx.svc, "train_start", base=base_model["id"], dataset=dataset["id"], name="x",
                  after={"quantize": ["Q4_K_M"], "imatrix": True, "evaluate": {"intent": "style"}})
    assert answer["steps"] == ["train", "merge_lora", "convert", "imatrix", "quantize", "evaluate"]
    assert ctx.svc.store.job(answer["job"]["id"])["then"][-1]["params"]["against"] == quant["id"]
    other = call(ctx.svc, "train_start", base=base_model["id"], dataset=dataset["id"], name="y",
                 after={"quantize": ["Q4_K_M"], "imatrix": True, "calibration": {"source": "dataset", "dataset": dataset["id"]}, "evaluate": {"intent": "style"}})
    assert other["steps"][:2] == ["imatrix", "quantize"], "another calibration text: a new matrix on the existing f16"
