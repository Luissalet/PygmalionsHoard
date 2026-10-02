"""Canonical job events for the family hub: pygmalion.job.queued|started|progress|done|failed|cancelled."""

import time

import pytest

from conftest import call, finish, needs_posix
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.jobevents import JobEvents, canonical_kind, published_model
from pygmalion_hoard.jobs import KINDS


@pytest.fixture
def bus(ctx):
    events = []
    ctx.svc.jobs.emit = lambda t, d: events.append((t, dict(d)))
    return events


def of(bus, name):
    return [d for t, d in bus if t == f"pygmalion.job.{name}"]


def runners(ctx, **special):
    def make(kind):
        def run(c):
            if kind in special:
                return special[kind](c)
            return {}
        return run
    ctx.svc.jobs.runners = {k: make(k) for k in KINDS}


def test_kinds_are_canonical():
    assert [canonical_kind(k) for k in ("train", "merge_lora", "merge_models", "convert", "quantize", "publish", "imatrix")] == \
        ["train", "merge", "merge", "convert", "quantize", "publish", "imatrix"]
    assert published_model({"ollama": {"tag": "pyg-x:v1"}, "llama": {"id": "pyg-x"}}) == "pyg-x:v1"
    assert published_model({"llama": {"id": "pyg-x"}}) == "pyg-x" and published_model({}) == "" and published_model(None) == ""


def test_the_lifecycle_has_the_canonical_fields(ctx, bus):
    runners(ctx)
    job = ctx.svc.jobs.submit("merge_models", {}, title="Mezcla")
    ctx.svc.jobs.run_until_idle()
    assert [t for t, _ in bus] == ["pygmalion.job.queued", "pygmalion.job.started", "pygmalion.job.done"]
    queued, started, done = of(bus, "queued")[0], of(bus, "started")[0], of(bus, "done")[0]
    for data in (queued, started, done):
        assert data["job_id"] == job["id"] and data["title"] == "Mezcla" and data["kind"] == "merge" and data["job_kind"] == "merge_models"
        assert data["url"] == f"http://127.0.0.1:{ctx.svc.config.port}/#/trabajos/{job['id']}"
    assert queued["progress"] == 0.0 and done["progress"] == 1.0 and "model" not in done and "error" not in done


def test_a_failed_job_carries_the_error_and_high_level_fields(ctx, bus):
    def broken(c):
        raise PygmalionError("failed", "it broke", "try this")
    runners(ctx, convert=broken)
    ctx.svc.jobs.submit("convert", {})
    ctx.svc.jobs.run_until_idle()
    failed = of(bus, "failed")[0]
    assert failed["error"] == "it broke" and failed["kind"] == "convert" and "progress" not in failed
    assert of(bus, "done") == []


def test_cancelled_while_queued_and_while_running(ctx, bus):
    runners(ctx)
    queued = ctx.svc.jobs.submit("convert", {})
    ctx.svc.jobs.cancel(queued["id"])
    assert [d["job_id"] for d in of(bus, "cancelled")] == [queued["id"]] and of(bus, "started") == []

    def slow(c):
        for _ in range(200):
            c.check_cancel()
            time.sleep(0.01)
            if c.id and not c.cancel.is_set():
                c.cancel.set()                 # ask for the cancel ourselves, as the user would
    runners(ctx, convert=slow)
    bus.clear()
    ctx.svc.jobs.submit("convert", {})
    ctx.svc.jobs.run_until_idle()
    assert len(of(bus, "cancelled")) == 1 and of(bus, "done") == []


def test_progress_events_are_throttled_and_carry_the_gpu(ctx, bus):
    ctx.svc.jobs.events.min_interval_s = 0.0

    def train(c):
        c.job["gpus"] = [3]
        c.progress(step=5, total=10, eta_s=40)
        c.progress(step=10, total=10)
        return {}
    runners(ctx, train=train)
    ctx.svc.jobs.submit("train", {})
    ctx.svc.jobs.run_until_idle()
    progress = of(bus, "progress")
    assert [p["progress"] for p in progress] == [0.5, 1.0] and progress[0]["eta_s"] == 40 and progress[0]["gpu"] == 3
    # with the default interval a burst sends one event at most
    bus.clear()
    ctx.svc.jobs.events.min_interval_s = 60.0

    def burst(c):
        for step in range(1, 6):
            c.progress(step=step, total=5)
        return {}
    runners(ctx, train=burst)
    ctx.svc.jobs.submit("train", {})
    ctx.svc.jobs.run_until_idle()
    assert of(bus, "progress") == []


def test_progress_without_a_percentage_sends_nothing():
    sent = []
    events = JobEvents(lambda t, d: sent.append(t), min_interval_s=0.0)
    events.progress({"id": "j", "kind": "train"}, {"message": "loading"})
    assert sent == []


def test_a_failing_emit_never_reaches_the_job(ctx):
    runners(ctx)

    def boom(*_):
        raise RuntimeError("bus down")
    ctx.svc.jobs.emit = boom
    job = ctx.svc.jobs.submit("convert", {})
    ctx.svc.jobs.run_until_idle()
    assert ctx.svc.store.job(job["id"])["state"] == "done"


def test_a_job_found_running_at_start_is_reported_failed(ctx, bus):
    runners(ctx)
    job = ctx.svc.jobs.submit("train", {})
    ctx.svc.store.update_job(job["id"], state="running")
    bus.clear()
    ctx.svc.jobs.start()
    failed = of(bus, "failed")
    assert len(failed) == 1 and failed[0]["job_id"] == job["id"] and "stopped" in failed[0]["error"]


def test_a_publish_job_names_the_published_model(ctx, bus):
    runners(ctx, publish=lambda c: {"ollama": {"artifact": "a_1", "tag": "pyg-casa-abuela:v1"}, "llama": {"id": "pyg-casa-abuela"}})
    ctx.svc.jobs.submit("publish", {"gguf": "x"})
    ctx.svc.jobs.run_until_idle()
    done = of(bus, "done")[0]
    assert done["kind"] == "publish" and done["model"] == "pyg-casa-abuela:v1"


@needs_posix
def test_a_real_publish_to_ollama_reports_the_tag_the_hub_rule_needs(ctx, base_model, bus):
    gguf = finish(ctx.svc, call(ctx.svc, "convert_start", model=base_model["id"]))["out_artifact"]
    bus.clear()
    answer = call(ctx.svc, "publish_ollama", gguf=gguf, name="Casa Abuela", tag="v1", num_ctx=4096, wait_s=30)
    ctx.svc.jobs.run_until_idle()
    assert ctx.svc.store.job(answer["job"]["id"])["state"] == "done"
    done = of(bus, "done")[0]
    assert done["kind"] == "publish" and done["model"] == "pyg-casa-abuela:v1" and done["title"]
    assert [t for t, _ in bus][:2] == ["pygmalion.job.queued", "pygmalion.job.started"]
