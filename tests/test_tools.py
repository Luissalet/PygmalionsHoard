"""Every tool of the catalogue is called through ``call_tool`` on a fully wired studio, in the order a person would use them."""

import json

import httpx
import pytest

from conftest import needs_posix
from gguf_builder import build_gguf
from helpers import chat_records
from pygmalion_hoard.agent_tools import TOOLS, TOOLS_BY_NAME, call_tool, tool_catalog, uncapped
from pygmalion_hoard.errors import PygmalionError


class Runner:
    def __init__(self, ctx):
        self.ctx = ctx
        self.called = []

    def __call__(self, tool, /, **arguments):
        self.called.append(tool)
        with uncapped():
            result = call_tool(self.ctx.svc, tool, arguments)
        assert isinstance(result, dict)
        json.dumps(result)                                   # every answer must be plain JSON
        return result

    def wait(self):
        self.ctx.svc.jobs.run_until_idle()


def test_the_catalogue_is_well_formed():
    names = [t.name for t in TOOLS]
    assert len(names) == len(set(names)) == 44
    for entry in tool_catalog():
        assert entry["description"].splitlines()[0] and entry["inputSchema"]["type"] == "object"
        assert set(entry["annotations"]) == {"readOnlyHint", "destructiveHint", "idempotentHint", "openWorldHint"}
    deleting = {"dataset_delete", "job_delete", "artifact_delete", "unpublish"}
    for name in deleting:
        assert "confirm" in TOOLS_BY_NAME[name].input_model.model_fields


def test_unknown_tool_and_bad_arguments():
    with pytest.raises(KeyError):
        call_tool(None, "no_such_tool", {})
    with pytest.raises(ValueError):
        call_tool(None, "train_plan", {"base": "x"})          # dataset is required


@needs_posix
def test_every_tool_works_in_a_realistic_session(ctx, base_model, tmp_path):
    t = Runner(ctx)
    # --- look around
    ov = t("pygmalion_overview")
    assert ov["allowed_gpus"] == [2, 3] and ov["next_steps"] and "bases" in ov["summary"]
    env = t("env_check")
    assert env["trainer"]["ok"] is True and env["capabilities"]["train"] is True and env["capabilities"]["quantize"] is True and env["ok"] is True
    assert t("env_check", fresh=False)["checked_ts"] >= env["checked_ts"]
    gpus = t("gpu_status")
    assert [g["index"] for g in gpus["gpus"]] == [0, 1, 2, 3] and gpus["allowed"] == [2, 3]
    settings = t("settings_get")
    assert settings["values"]["train.rank"] in (16, "16") and "hf.token" in settings["secrets"]
    t("settings_set", values={"train.rank": 8})
    assert t("settings_get")["values"]["train.rank"] in (8, "8")
    t("settings_set", values={"train.rank": 16})
    assert t("secret_set", value="hf_exampletoken1234")["hf.token"] == {"configured": True, "source": "file"}
    assert "hf_exampletoken1234" not in json.dumps(t("settings_get"))
    t("secret_set", value="")

    # --- models
    bases = t("bases_list")
    assert [b["name"] for b in bases["bases"]] == ["acme--tiny-base"] and bases["bases"][0]["convertible"] is True
    assert t("bases_list", deep=True)["bases"]
    got = t("base_get", base=base_model["id"])
    assert got["artifact"]["id"] == base_model["id"] and got["estimate"]["total_mb"] > 0
    ctx.svc.hub.transport = httpx.MockTransport(lambda r: httpx.Response(200, json=(
        [{"id": "acme/found", "downloads": 10, "tags": ["license:mit"], "safetensors": {"total": 1_000_000}}] if r.url.path == "/api/models" else
        {"id": "acme/found", "siblings": [{"rfilename": "model.safetensors", "size": 1000}, {"rfilename": "config.json", "size": 10}]})))
    ctx.svc.hub.offline = False
    found = t("hf_search", query="found")
    assert found["results"][0]["repo_id"] == "acme/found" and found["results"][0]["license"] == "mit"
    assert t("base_download", repo_id="acme/found")["started"] is False

    # --- datasets
    text = "\n".join(json.dumps(r, ensure_ascii=False) for r in chat_records(60))
    assert t("dataset_preview_source", source={"type": "jsonl", "text": text}, limit=2)["count"] == 60
    made = t("dataset_create", name="Familia", sources=[{"type": "jsonl", "text": text}], wait_s=0)
    t.wait()
    assert t("datasets_list")["datasets"][0]["name"] == "Familia"
    version = ctx.svc.store.job(made["job"]["id"])["result"]["version"]
    assert t("dataset_get", dataset="Familia")["version"]["records"] == 60
    assert t("dataset_records", dataset="Familia", limit=3)["total"] == 60
    assert t("dataset_review", dataset="Familia", reject=[0])["version"]["n"] == 2
    assert t("dataset_apply", dataset="Familia", operations=[{"op": "dedupe_exact"}])["version"]["n"] == 3

    # --- training and what follows
    plan = t("train_plan", base=base_model["id"], dataset="Familia", rank=4, alpha=8)
    assert plan["pick"]["fits"] is True and plan["params"]["rank"] == 4
    started = t("train_start", base=base_model["id"], dataset="Familia", rank=4, alpha=8, max_steps=6, grad_accum=2, save_every=3, name="mi-lora")
    t.wait()
    adapter = ctx.svc.store.artifact(ctx.svc.store.job(started["job"]["id"])["out_artifact"])
    assert adapter["kind"] == "adapter" and adapter["name"] == "mi-lora"
    merged_answer = t("merge_lora_start", base=base_model["id"], adapter=adapter["id"], name="fusion")
    t.wait()
    merged = ctx.svc.store.artifact(ctx.svc.store.job(merged_answer["job"]["id"])["out_artifact"])
    assert merged["kind"] == "merged"
    assert t("merge_check", models=[base_model["id"], merged["id"]])["ok"] is True
    mm = t("merge_models_start", models=[base_model["id"], merged["id"]], method="slerp", t=0.3, name="slerp")
    t.wait()
    assert ctx.svc.store.job(mm["job"]["id"])["state"] == "done"
    cx = t("ctx_extend_start", model=base_model["id"], factor=2, name="largo")
    t.wait()
    variant = ctx.svc.store.artifact(ctx.svc.store.job(cx["job"]["id"])["out_artifact"])
    assert variant["kind"] == "ctx_variant"
    conv = t("convert_start", model=merged["id"], outtype="f16", name="fusion-f16")
    t.wait()
    gguf = ctx.svc.store.artifact(ctx.svc.store.job(conv["job"]["id"])["out_artifact"])
    assert gguf["kind"] == "gguf"
    q = t("quantize_start", gguf=gguf["id"], types=["Q4_K_M"], imatrix=True, perplexity=False, name="fusion")
    t.wait()
    quant = [a for a in ctx.svc.store.artifacts(kind="gguf") if a["metrics"].get("quant") == "Q4_K_M"][0]
    assert q["steps"] == ["imatrix", "quantize"]
    ppl = t("perplexity_start", gguf=quant["id"], chunks=2)
    t.wait()
    assert ctx.svc.store.artifact(quant["id"])["metrics"]["ppl"]["value"] > 0 and ppl["state"] in ("queued", "done")
    path = build_gguf(tmp_path / "fit.gguf", context=8192, blocks=32, heads=32, kv_heads=8, embedding=4096)
    fit_art = ctx.svc.store.create_artifact("gguf", "fit", path=str(path), size=100)
    assert t("ctx_fit", gguf=fit_art["id"], contexts=[4096, 8192])["rows"][1]["context"] == 8192

    # --- jobs
    listed = t("jobs_list", limit=100)
    assert len(listed["jobs"]) >= 8
    first = listed["jobs"][-1]["id"]
    job = t("job_get", job=first)
    assert job["state"] == "done" and "log_tail" in job and job["pipeline"]
    queued = t("convert_start", model=variant["id"])
    cancelled = t("job_cancel", job=queued["job"]["id"])
    assert cancelled["job"]["state"] == "cancelled"
    assert t("job_resume", job=queued["job"]["id"])["job"]["state"] == "queued"
    t.wait()
    with pytest.raises(PygmalionError):
        t("job_delete", job=first)
    assert t("job_delete", job=first, confirm=True)["deleted"] == first

    # --- artifacts and lineage
    arts = t("artifacts_list", kind="gguf")
    assert len(arts["artifacts"]) >= 3 and arts["storage"]["total"] > 0
    assert t("artifact_get", artifact=quant["id"])["comparable_parent"]["id"] == gguf["id"]
    recipe = t("artifact_recipe", artifact=quant["id"])
    assert recipe["steps"][0]["kind"] == "base"
    assert t("artifact_update", artifact=merged["id"], notes="la buena", pinned=True)["artifact"]["pinned"]
    graph = t("lineage_graph", root=quant["id"])
    assert {n["kind"] for n in graph["nodes"]} >= {"base", "adapter", "merged", "gguf", "imatrix"} and graph["edges"]

    # --- evaluation (Galton answers through a stand-in)
    from test_evaluate import FakeGalton
    fake = FakeGalton()
    ctx.svc.galton.family_call = fake.family_call
    ctx.svc.galton.offline = False
    ctx.svc.galton.sleep = lambda s: None
    ev = t("evaluate_start", artifact=quant["id"], against=gguf["id"], intent="smoke")
    t.wait()
    assert ctx.svc.store.job(ev["job"]["id"])["result"]["verdict"] == "better"
    # trained from the base model: without `against` the reference is the base's own Q4_K_M, prepared first
    plan = t("evaluate_plan", artifact=quant["id"], intent="smoke")
    assert plan["reference"]["mode"] == "base" and plan["reference"]["model"] == "acme/tiny-base" and plan["reference"]["steps"] == ["convert", "imatrix", "quantize"]
    ev = t("evaluate_start", artifact=quant["id"], intent="smoke")
    assert ev["steps"] == ["convert", "imatrix", "quantize", "evaluate"]
    t.wait()
    done = ctx.svc.store.jobs(pipeline_id=ev["job"]["pipeline_id"], oldest_first=True)
    assert [j["state"] for j in done] == ["done"] * 4 and done[-1]["result"]["reference"]["mode"] == "base"
    assert t("evaluate_plan", artifact=quant["id"], intent="smoke")["reference"]["ready"] is True

    # --- publishing
    pub = t("publish_ollama", gguf=quant["id"], name="mi modelo", tag="v1", wait_s=30)
    t.wait()
    assert ctx.svc.store.job(pub["job"]["id"])["result"]["ollama"]["tag"] == "pyg-mi-modelo:v1"
    llama = t("publish_llama", gguf=quant["id"], name="mi modelo", ctx=4096, wait_s=30)
    t.wait()
    assert ctx.svc.store.job(llama["job"]["id"])["result"]["llama"]["id"] == "pyg-mi-modelo"
    with pytest.raises(PygmalionError):
        t("unpublish", artifact=quant["id"])
    assert len(t("unpublish", artifact=quant["id"], confirm=True)["removed"]) == 2

    # --- cleaning up
    assert t("artifact_delete", artifact=variant["id"], delete_files=True, confirm=True)["files_removed"] is True
    assert t("dataset_delete", dataset="Familia", confirm=True)["deleted"] == "Familia"

    never = set(TOOLS_BY_NAME) - set(t.called)
    assert not never, f"tools never called by this test: {sorted(never)}"


def test_agent_results_are_capped_and_say_so(ctx, base_model):
    for i in range(400):
        ctx.svc.store.create_artifact("gguf", f"m{i:04d}", path=f"/m/{i}.gguf", size=1, notes="n" * 50)
    capped = call_tool(ctx.svc, "artifacts_list", {"limit": 500})
    assert len(capped["artifacts"]) < 400 and capped["truncated"]["hint"]
    assert len(json.dumps(capped)) <= 20_000
    with uncapped():
        assert len(call_tool(ctx.svc, "artifacts_list", {"limit": 500})["artifacts"]) == 401
