"""Lanes, pipelines, cancel, resume, interrupted jobs and progress."""

import json
import threading
import time

import pytest

from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.jobs import JobCancelled, JobManager, resolve_refs


@pytest.fixture
def jm(ctx):
    """The app's job manager with simple runners; the queue is run inline unless a test starts the lanes."""
    manager = ctx.svc.jobs
    manager.log = []

    def make(kind):
        def runner(c):
            manager.log.append((kind, dict(c.params)))
            if c.params.get("fail"):
                raise PygmalionError("failed", "it broke", "try this")
            if c.params.get("crash"):
                raise ZeroDivisionError("oops")
            if c.params.get("slow"):
                for _ in range(200):
                    c.check_cancel()
                    time.sleep(0.02)
            art = ctx.svc.store.create_artifact(c.params.get("artifact_kind", "merged"), c.params.get("name", f"{kind}-{c.id}"), parents=c.params.get("parents") or [],
                                                job_id=c.id, metrics=c.params.get("metrics"), recipe=c.params.get("recipe"))
            c.set_output(art["id"])
            return {"made": art["id"]}
        return runner

    from pygmalion_hoard.jobs import KINDS
    manager.runners = {k: make(k) for k in KINDS}
    return manager


def test_lanes_by_kind(jm):
    assert [jm.lane_for(k) for k in ("train", "imatrix", "perplexity", "evaluate")] == ["gpu"] * 4
    assert [jm.lane_for(k) for k in ("download", "dataset_build", "merge_models", "convert", "quantize", "ctx_extend", "publish")] == ["cpu"] * 7
    assert jm.lane_for("merge_lora", {}) == "cpu" and jm.lane_for("merge_lora", {"device": "cuda"}) == "gpu"


def test_unknown_kinds_are_refused(jm):
    with pytest.raises(PygmalionError):
        jm.submit("teleport", {})
    with pytest.raises(PygmalionError):
        jm.submit("train", {}, then=[{"kind": "teleport"}])
    with pytest.raises(PygmalionError):
        jm.pipeline([])


def test_a_job_runs_and_is_recorded(ctx, jm):
    job = jm.submit("convert", {"name": "x"}, title="Convert")
    assert job["state"] == "queued" and (ctx.svc.work.job_dir(job["id"])).is_dir()
    assert jm.run_until_idle() == 1
    done = ctx.svc.store.job(job["id"])
    assert done["state"] == "done" and done["out_artifact"] and done["progress"]["pct"] == 100.0 and done["attempts"] == 1
    assert ctx.svc.store.runs("convert")[0]["ok"] == 1 and "== done" in (ctx.svc.work.job_dir(job["id"]) / "job.log").read_text(encoding="utf-8")


def test_failure_is_recorded_with_its_hint_and_stops_the_pipeline(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {"fail": True}}, {"kind": "quantize", "params": {}}])
    jm.run_until_idle()
    job = ctx.svc.store.job(first["id"])
    assert job["state"] == "failed" and job["error"] == "it broke" and job["hint"] == "try this"
    assert len(ctx.svc.store.jobs(pipeline_id=first["pipeline_id"])) == 1


def test_a_crashing_runner_does_not_stop_the_queue(ctx, jm):
    a = jm.submit("convert", {"crash": True})
    b = jm.submit("convert", {})
    jm.run_until_idle()
    assert ctx.svc.store.job(a["id"])["state"] == "failed" and "ZeroDivisionError" in ctx.svc.store.job(a["id"])["error"]
    assert ctx.svc.store.job(b["id"])["state"] == "done"


def test_pipeline_steps_run_in_order_and_resolve_references(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {"artifact_kind": "gguf", "name": "f16"}},
                         {"kind": "quantize", "params": {"parents": ["$gguf"], "artifact_kind": "gguf", "name": "q", "metrics": {"quant": "Q4_K_M"}}},
                         {"kind": "publish", "params": {"parents": ["$prev", "$gguf", "$quant:Q4_K_M"], "artifact_kind": "ollama", "name": "pub"}}])
    assert jm.run_until_idle() == 3
    jobs = ctx.svc.store.jobs(pipeline_id=first["pipeline_id"], oldest_first=True)
    assert [j["kind"] for j in jobs] == ["convert", "quantize", "publish"] and [j["step"] for j in jobs] == [0, 1, 2]
    f16, quant, pub = (ctx.svc.store.artifact(j["out_artifact"]) for j in jobs)
    assert quant["parents"] == [f16["id"]] and pub["parents"] == [quant["id"], f16["id"], quant["id"]]


def test_an_unresolvable_reference_stops_the_pipeline_cleanly(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {}}, {"kind": "quantize", "params": {"x": "$adapter"}}])
    jm.run_until_idle()
    job = ctx.svc.store.job(first["id"])
    assert job["state"] == "done" and "could not start" in job["error"]
    assert len(ctx.svc.store.jobs(pipeline_id=first["pipeline_id"])) == 1


def test_resolve_refs_only_touches_reference_shaped_strings():
    outputs = {"adapter": "a_1", "quant:Q4_K_M": "a_2"}
    got = resolve_refs({"a": "$adapter", "b": ["$quant:Q4_K_M", "$5 dollars", "plain", 3], "c": {"d": "$adapter"}}, outputs)
    assert got == {"a": "a_1", "b": ["a_2", "$5 dollars", "plain", 3], "c": {"d": "a_1"}}
    with pytest.raises(PygmalionError) as exc:
        resolve_refs("$merged", outputs)
    assert "$adapter" in exc.value.hint


def test_alias_names_from_the_as_parameter(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {"_as": "baseline", "artifact_kind": "gguf"}}, {"kind": "quantize", "params": {"parents": ["$baseline"]}}])
    jm.run_until_idle()
    outs = jm.pipeline_outputs(first["pipeline_id"])
    assert outs["baseline"] == outs["gguf"] and outs["prev"] == outs["merged"] != outs["baseline"]


def test_cancel_a_queued_job(ctx, jm):
    job = jm.submit("convert", {})
    assert jm.cancel(job["id"])["state"] == "cancelled"
    assert jm.run_until_idle() == 0 and jm.log == []
    with pytest.raises(PygmalionError) as exc:
        jm.cancel(job["id"])
    assert exc.value.code == "conflict"


def test_cancel_a_running_job_through_the_lane_threads(ctx, jm):
    live = JobManager(jm.deps, jm.runners, enabled=True)
    live.start()
    try:
        job = live.submit("convert", {"slow": True})
        for _ in range(100):
            if ctx.svc.store.job(job["id"])["state"] == "running":
                break
            time.sleep(0.05)
        assert ctx.svc.store.job(job["id"])["state"] == "running"
        live.cancel(job["id"])
        final = live.wait(job["id"], timeout=10)
        assert final["state"] == "cancelled"
    finally:
        live.stop()


def test_a_cancelled_pipeline_step_does_not_spawn_the_next(ctx, jm):
    live = JobManager(jm.deps, jm.runners, enabled=True)
    live.start()
    try:
        first = live.pipeline([{"kind": "convert", "params": {"slow": True}}, {"kind": "quantize", "params": {}}])
        time.sleep(0.4)
        live.cancel(first["id"])
        live.wait(first["id"], timeout=10)
        assert len(ctx.svc.store.jobs(pipeline_id=first["pipeline_id"])) == 1
    finally:
        live.stop()


def test_different_lanes_run_at_the_same_time_and_one_lane_one_job(ctx, jm):
    running = {"max_cpu": 0, "now_cpu": 0, "overlap": False, "gpu_on": False}
    lock = threading.Lock()

    def cpu(c):
        with lock:
            running["now_cpu"] += 1
            running["max_cpu"] = max(running["max_cpu"], running["now_cpu"])
            if running["gpu_on"]:
                running["overlap"] = True
        time.sleep(0.3)
        with lock:
            running["now_cpu"] -= 1
        return {}

    def gpu(c):
        with lock:
            running["gpu_on"] = True
        time.sleep(0.5)
        with lock:
            running["gpu_on"] = False
        return {}

    live = JobManager(jm.deps, {**jm.runners, "convert": cpu, "train": gpu}, enabled=True)
    live.start()
    try:
        a, b = live.submit("convert", {}), live.submit("convert", {})
        t = live.submit("train", {})
        for j in (a, b, t):
            live.wait(j["id"], timeout=15)
    finally:
        live.stop()
    assert running["max_cpu"] == 1 and running["overlap"]


def test_interrupted_on_restart_and_resume(ctx, jm):
    job = ctx.svc.store.create_job("train", "gpu", {})
    ctx.svc.store.update_job(job["id"], state="running")
    assert JobManager(jm.deps, jm.runners, enabled=False).start() == 1
    assert ctx.svc.store.job(job["id"])["state"] == "interrupted"
    assert jm.view(ctx.svc.store.job(job["id"]))["resumable"] is True
    jm.resume(job["id"])
    assert ctx.svc.store.job(job["id"])["state"] == "queued"
    jm.run_until_idle()
    assert ctx.svc.store.job(job["id"])["state"] == "done" and ctx.svc.store.job(job["id"])["attempts"] == 1


def test_only_stopped_jobs_resume(ctx, jm):
    job = jm.submit("convert", {})
    with pytest.raises(PygmalionError):
        jm.resume(job["id"])
    jm.run_until_idle()
    with pytest.raises(PygmalionError):
        jm.resume(job["id"])


def test_resuming_a_failed_job_in_a_pipeline_continues_the_pipeline(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {"fail": True}}, {"kind": "quantize", "params": {}}])
    jm.run_until_idle()
    ctx.svc.store.update_job(first["id"], params={})
    jm.resume(first["id"])
    jm.run_until_idle()
    assert [j["state"] for j in ctx.svc.store.jobs(pipeline_id=first["pipeline_id"], oldest_first=True)] == ["done", "done"]
    assert ctx.svc.store.job(first["id"])["attempts"] == 2


def test_delete_needs_a_finished_job_and_removes_its_folder(ctx, jm):
    job = jm.submit("convert", {})
    with pytest.raises(PygmalionError):
        jm.delete(job["id"])
    jm.run_until_idle()
    folder = ctx.svc.work.job_dir(job["id"])
    out = ctx.svc.store.job(job["id"])["out_artifact"]
    assert folder.is_dir()
    assert jm.delete(job["id"])["artifacts_kept"] and not folder.exists()
    assert ctx.svc.store.artifact(out)


def test_progress_is_merged_throttled_and_charted(ctx, jm):
    seen = {}

    def runner(c):
        for step in range(1, 6):
            c.progress(step=step, total=5, loss=2.0 / step, lr=1e-4)
        c.progress(force=True, eval_loss=0.9, step=5)
        seen["pct"] = c._progress["pct"]
        return {}

    jm.runners["train"] = runner
    job = jm.submit("train", {})
    jm.run_until_idle()
    curve = jm.curve(job["id"])
    assert seen["pct"] == 100.0 and len(curve["train"]) == 5 and curve["train"][0] == [1, 2.0] and curve["eval"] == [[5, 0.9]]
    assert curve["min_loss"] == pytest.approx(0.4) and curve["last_loss"] == pytest.approx(0.4)


def test_curve_is_downsampled(ctx, jm):
    job = jm.submit("train", {})
    folder = ctx.svc.work.job_dir(job["id"])
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "progress.jsonl").write_text("\n".join(json.dumps({"step": i, "loss": 1.0 / (i + 1)}) for i in range(5000)) + "\nbroken\n", encoding="utf-8")
    c = jm.curve(job["id"], max_points=100)
    assert len(c["train"]) <= 101 and c["points"] == 5000 and c["train"][-1][0] == 4999


def test_view_hides_internal_parameters_and_lists_the_pipeline(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {"x": 1, "_out_id": "secret"}}, {"kind": "quantize", "params": {}}])
    view = jm.view(ctx.svc.store.job(first["id"]), detail=True)
    assert view["params"] == {"x": 1} and view["next"] == ["quantize"] and view["pipeline"][0]["kind"] == "convert"


def test_reserve_output_id_is_stable_across_attempts(ctx, jm):
    ids = []

    def runner(c):
        ids.append(c.reserve_output_id())
        if len(ids) == 1:
            raise PygmalionError("failed", "first try fails")
        return {}

    jm.runners["convert"] = runner
    job = jm.submit("convert", {})
    jm.run_until_idle()
    jm.resume(job["id"])
    jm.run_until_idle()
    assert len(ids) == 2 and ids[0] == ids[1]


def test_status_reports_lanes(ctx, jm):
    jm.submit("train", {})
    jm.submit("convert", {})
    jm.submit("convert", {})
    st = jm.status()
    assert st["lanes"]["gpu"]["queued"] == 1 and st["lanes"]["cpu"]["queued"] == 2


def test_events_are_emitted(ctx, jm):
    events = []
    jm.emit = lambda t, d: events.append(t)
    jm.submit("convert", {})
    jm.run_until_idle()
    assert events == ["pygmalion.job.queued", "pygmalion.job.started", "pygmalion.job.done"]


def test_a_paused_scheduler_starts_nothing(ctx, jm):
    live = JobManager(jm.deps, jm.runners, enabled=True, paused=lambda: True)
    live.start()
    try:
        job = live.submit("convert", {})
        time.sleep(1.5)
        assert ctx.svc.store.job(job["id"])["state"] == "queued"
    finally:
        live.stop()


def test_job_cancelled_exception_cancels(ctx, jm):
    def runner(c):
        raise JobCancelled()

    jm.runners["convert"] = runner
    job = jm.submit("convert", {})
    jm.run_until_idle()
    assert ctx.svc.store.job(job["id"])["state"] == "cancelled"


def test_a_cancel_that_lands_between_picking_and_starting_a_job_wins(ctx, jm):
    job = jm.submit("convert", {"name": "x"})
    picked = ctx.svc.store.next_queued("cpu")          # the lane has read the job...
    jm.cancel(job["id"])                                # ...and the person cancels it before it starts
    jm._execute(picked)
    assert ctx.svc.store.job(job["id"])["state"] == "cancelled" and jm.log == []


# ------------------------------------------------------------------------------------------------ timing: the estimate, then the measured speed
def _train_job(jm, **estimate):
    return jm.submit("train", {"_estimate": {"seconds": 120, "tokens_per_s": 2500, "tokens": 190000, **estimate}})


def test_a_queued_job_shows_the_planners_estimate(ctx, jm):
    job = _train_job(jm)
    t = jm.view(ctx.svc.store.job(job["id"]))["timing"]
    assert t["source"] == "estimate" and t["eta_s"] == 120 and t["tokens_per_s"] == 2500 and t["estimated_s"] == 120


def test_a_running_job_replaces_the_estimate_with_the_measured_speed_and_the_real_eta(ctx, jm):
    job = _train_job(jm)
    store = ctx.svc.store
    store.update_job(job["id"], state="running", started_ts=1000.0, progress={"step": 0, "total": 27})
    assert jm.view(store.job(job["id"]))["timing"]["source"] == "estimate"                  # nothing measured yet
    store.update_job(job["id"], progress={"step": 3, "total": 27, "tokens_per_s": 1000, "avg_tokens_per_s": 1400, "eta_s": 310})
    t = jm.view(store.job(job["id"]))["timing"]
    assert t == {"source": "measured", "tokens_per_s": 1400, "eta_s": 310, "estimated_s": 120, "estimated_tokens_per_s": 2500}


def test_a_finished_job_reports_how_long_it_took_against_the_estimate(ctx, jm):
    job = _train_job(jm)
    store = ctx.svc.store
    store.update_job(job["id"], state="done", started_ts=1000.0, finished_ts=1250.0, progress={"step": 27, "total": 27, "avg_tokens_per_s": 1300})
    t = jm.view(store.job(job["id"]))["timing"]
    assert t["source"] == "done" and t["elapsed_s"] == 250 and t["estimated_s"] == 120 and t["tokens_per_s"] == 1300


def test_a_job_without_an_estimate_has_no_timing_until_done(ctx, jm):
    job = jm.submit("convert", {})
    assert jm.view(ctx.svc.store.job(job["id"]))["timing"] is None


# ------------------------------------------------------------------------------------------------ the chain of steps and the scheduler's own state
def test_the_pipeline_lists_every_step_including_the_ones_not_created_yet(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {}, "title": "Convert"}, {"kind": "quantize", "params": {}, "title": "Quantize"},
                         {"kind": "perplexity", "params": {}, "title": "Measure"}])
    steps = jm.pipeline_steps(first)
    assert [(s["kind"], s["state"], s["id"] is None) for s in steps] == [("convert", "queued", False), ("quantize", "planned", True), ("perplexity", "planned", True)]
    assert [s["step"] for s in steps] == [0, 1, 2]
    jm.run_until_idle()
    done = jm.pipeline_steps(ctx.svc.store.job(first["id"]))
    assert [(s["kind"], s["state"]) for s in done] == [("convert", "done"), ("quantize", "done"), ("perplexity", "done")] and all(s["id"] for s in done)


def test_a_failed_step_leaves_the_rest_listed_as_planned_no_more(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {"fail": True}}, {"kind": "quantize", "params": {}}])
    jm.run_until_idle()
    steps = jm.pipeline_steps(ctx.svc.store.job(first["id"]))
    assert [(s["kind"], s["state"]) for s in steps][0] == ("convert", "failed") and steps[0]["error"] is True


def test_the_job_detail_carries_the_whole_chain(ctx, jm):
    first = jm.pipeline([{"kind": "convert", "params": {}}, {"kind": "quantize", "params": {}}])
    detail = jm.view(ctx.svc.store.job(first["id"]), detail=True)
    assert [s["state"] for s in detail["pipeline"]] == ["queued", "planned"]


def test_the_scheduler_reports_the_running_job_only_while_the_database_says_so(ctx, jm):
    job = jm.submit("convert", {})
    jm._running["cpu"] = job["id"]
    assert jm.status()["lanes"]["cpu"]["current"] is not None
    ctx.svc.store.update_job(job["id"], state="done")
    assert jm.status()["lanes"]["cpu"]["current"] is None, "a lane whose job already ended is idle, whatever its thread remembers"
    jm._running["cpu"] = "j_gone"
    assert jm.status()["lanes"]["cpu"]["current"] is None


def test_a_row_stuck_in_running_is_closed_as_interrupted_when_nothing_runs_it(ctx, jm):
    live = JobManager(jm.deps, jm.runners, enabled=True, paused=lambda: True)     # paused: a lane must not pick the row up between the two writes below
    live.start()
    try:
        stuck = ctx.svc.store.create_job("convert", "cpu", {})
        ctx.svc.store.update_job(stuck["id"], state="running")
        status = live.status()
        assert ctx.svc.store.job(stuck["id"])["state"] == "interrupted" and status["lanes"]["cpu"]["current"] is None
        assert ctx.svc.store.job(stuck["id"])["error"]
    finally:
        live.stop()


def test_a_job_this_process_runs_is_not_reaped(ctx, jm):
    live = JobManager(jm.deps, jm.runners, enabled=True)
    live.start()
    try:
        job = live.submit("convert", {"slow": True})
        for _ in range(100):
            if ctx.svc.store.job(job["id"])["state"] == "running":
                break
            time.sleep(0.05)
        assert live.reap_stale() == 0 and ctx.svc.store.job(job["id"])["state"] == "running"
        live.cancel(job["id"])
        live.wait(job["id"], 10)
    finally:
        live.stop()


def test_a_lane_keeps_working_after_a_bookkeeping_error(ctx, jm):
    live = JobManager(jm.deps, jm.runners, enabled=True)
    original = live.store.update_job
    calls = {"n": 0}

    def flaky(job_id, **fields):
        if fields.get("state") == "done" and calls["n"] == 0:
            calls["n"] += 1
            raise RuntimeError("database is locked")
        return original(job_id, **fields)
    live.store.update_job = flaky
    live.start()
    try:
        first = live.submit("convert", {})
        live.wait(first["id"], 10)
        second = live.submit("convert", {})
        assert live.wait(second["id"], 10)["state"] == "done"
        assert ctx.svc.store.job(first["id"])["state"] not in ("running", "queued"), "the broken job was closed, not left running"
        assert live.status()["lanes"]["cpu"]["current"] is None
    finally:
        live.store.update_job = original
        live.stop()
