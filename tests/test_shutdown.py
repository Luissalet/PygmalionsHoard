"""Stopping the app leaves nothing running: lane threads, the event worker, lease helpers and output readers all end, so the interpreter never
shuts down with a thread in the middle of a call (which ends a run with a fatal error although every test passed)."""

import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conftest import APP_THREAD_PREFIXES, lingering_threads, needs_posix
from pygmalion_hoard import procs
from pygmalion_hoard.config import Config
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.events import EventPump
from pygmalion_hoard.gpus import GpuManager
from pygmalion_hoard.hoard_link import family
from pygmalion_hoard.jobs import JobManager
from pygmalion_hoard.services import Services

from helpers import FakeLeaseFactory, fake_inventory

ROOT = Path(__file__).resolve().parents[1]


def threads_now() -> set[threading.Thread]:
    return {t for t in threading.enumerate() if t is not threading.main_thread()}


def wait_until(check, seconds=5.0):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if check():
            return True
        time.sleep(0.02)
    return check()


@pytest.fixture
def bus(monkeypatch):
    """The family bus as the app configures it, with the HTTP call replaced by a list."""
    posted = []
    monkeypatch.setitem(family._state, "app", "pygmalion")
    monkeypatch.setitem(family._state, "enabled", True)
    monkeypatch.setattr(family, "_post", lambda path, body, timeout: (posted.append(body) or (200, {"ok": True})))
    return posted


def live_services(tmp_path, n):
    config = Config(data_dir=tmp_path / f"data{n}", offline=True, scheduler=True, workers_dir=ROOT / "tests" / "fake_workers", data_dir_configured=True)
    svc = Services(config, probe=lambda: fake_inventory(), lease_factory=FakeLeaseFactory(), hub_status=lambda: None)
    svc.jobs.runners["convert"] = lambda c: {"done": True}
    return svc


# ------------------------------------------------------------------------------------------------- the whole app, again and again
def test_starting_and_stopping_the_services_again_and_again_leaves_no_thread(tmp_path, bus):
    before = threads_now()
    for n in range(6):
        svc = live_services(tmp_path, n)
        svc.start()
        assert svc.jobs.alive(), "the lane threads run"
        job = svc.jobs.submit("convert", {}, title="x")
        assert svc.jobs.wait(job["id"], timeout=10)["state"] == "done"
        assert svc.events.alive(), "an event was queued for the bus, so the one worker is running"
        assert svc.stop() == []
        assert not svc.jobs.alive() and not svc.events.alive()
        assert svc.stop() == [], "stopping twice is harmless"
        left = [t for t in threads_now() - before if t.is_alive()]
        assert not left, f"round {n}: {[t.name for t in left]}"
    assert any(b["type"] == "pygmalion.job.done" for b in bus), "the events did reach the bus"
    assert not [t for t in threads_now() if not t.daemon and t not in before], "no non-daemon thread was left either"
    assert not [t for t in lingering_threads() if t.name.startswith(APP_THREAD_PREFIXES)]


def test_the_lifespan_of_the_app_stops_the_services_and_their_threads(tmp_path, bus):
    from fastapi.testclient import TestClient
    from pygmalion_hoard.main import create_app
    before = threads_now()
    svc = live_services(tmp_path, 0)
    with TestClient(create_app(svc.config, services=svc), base_url="http://127.0.0.1"):
        assert svc.jobs.alive()
        job = svc.jobs.submit("convert", {})
        svc.jobs.wait(job["id"], timeout=10)
    assert not svc.jobs.alive() and svc._stopped
    assert not [t for t in threads_now() - before if t.is_alive()]


def test_the_app_code_never_loads_numpy(tmp_path):
    """A daemon thread in the middle of a numpy call at exit was a suspect: the app does not import numpy at all (the merge arithmetic runs in the
    trainer's worker processes)."""
    code = "import sys; import pygmalion_hoard.main, pygmalion_hoard.services, pygmalion_hoard.runners, pygmalion_hoard.jobs; print('numpy' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], cwd=str(ROOT), capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------------------------------------- the event worker
def test_events_go_through_one_worker_in_order_and_close_ends_it():
    seen = []
    pump = EventPump(lambda t, d: seen.append((t, d["n"])))
    for n in range(5):
        assert pump.emit("e", {"n": n})
    assert wait_until(lambda: len(seen) == 5) and [n for _t, n in seen] == [0, 1, 2, 3, 4]
    names = {t.name for t in threading.enumerate() if t.name == "pygmalion-events"}
    assert len(names) == 1 and pump.close() is True and not pump.alive()
    assert pump.emit("late", {"n": 9}) is False and pump.dropped == 1, "after close events are dropped, never sent from a new thread"


def test_close_drops_what_waits_and_waits_for_the_event_in_flight_only():
    release, started, seen = threading.Event(), threading.Event(), []

    def slow(t, d):
        started.set()
        release.wait(5)
        seen.append(d["n"])
    pump = EventPump(slow)
    pump.emit("e", {"n": 1})
    assert started.wait(5)
    for n in (2, 3, 4):
        pump.emit("e", {"n": n})
    threading.Timer(0.2, release.set).start()
    assert pump.close(timeout=5) is True
    assert seen == [1] and pump.dropped == 3


def test_close_gives_up_on_an_event_that_never_returns_and_says_so():
    release = threading.Event()
    pump = EventPump(lambda t, d: release.wait(10))
    pump.emit("e", {})
    time.sleep(0.1)
    started = time.monotonic()
    assert pump.close(timeout=0.3) is False and time.monotonic() - started < 3
    release.set()
    assert wait_until(lambda: not pump.alive())


def test_nothing_is_queued_when_nobody_listens():
    pump = EventPump(lambda t, d: None, wanted=lambda: False)
    assert pump.emit("e", {}) is False and not pump.alive()


def test_a_sender_that_fails_does_not_end_the_worker():
    seen = []

    def send(t, d):
        if d["n"] == 1:
            raise RuntimeError("hub down")
        seen.append(d["n"])
    pump = EventPump(send)
    for n in (1, 2):
        pump.emit("e", {"n": n})
    assert wait_until(lambda: seen == [2]) and pump.dropped == 1
    assert pump.close()


# ------------------------------------------------------------------------------------------------- the job lanes
def test_stop_joins_the_lanes_within_one_deadline_and_names_the_ones_that_would_not_end(ctx):
    release = threading.Event()
    jm = JobManager(ctx.svc.jobs.deps, {**ctx.svc.jobs.runners, "convert": lambda c: release.wait(10) and {}}, enabled=True)
    jm.start()
    job = jm.submit("convert", {})
    assert wait_until(lambda: ctx.svc.store.job(job["id"])["state"] == "running")
    started = time.monotonic()
    late = jm.stop(timeout=0.4)
    assert late == ["pygmalion-lane-cpu"] and time.monotonic() - started < 3, "one deadline for all the lanes, not one per lane"
    release.set()
    assert wait_until(lambda: not jm.alive())
    assert jm.stop() == [] and not jm._threads


def test_a_job_that_is_queued_when_the_app_stops_is_not_started_by_a_lane_that_is_leaving(ctx):
    jm = ctx.svc.jobs
    ran = []
    jm.runners["convert"] = lambda c: ran.append(c.id) or {}
    job = jm.submit("convert", {})
    jm.stop()
    jm._execute(ctx.svc.store.job(job["id"]))
    assert ran == [] and ctx.svc.store.job(job["id"])["state"] == "queued", "claimed after the stop, a job would never be cancelled"


# ------------------------------------------------------------------------------------------------- lease helpers
def test_close_gives_up_the_lease_requests_that_wait_and_ends_their_helper_threads(ctx):
    gate = threading.Event()
    released = []

    class Held:
        gpu, via, warning, info = 2, "hub", None, {"position": 1}

        def __init__(self, **kw):
            pass

        def acquire(self):
            gate.wait(10)

        def release(self):
            released.append(True)

    gpus = GpuManager(ctx.svc.settings, probe=lambda: fake_inventory(), lease_factory=Held, hub_status=lambda: None)
    errors = []

    def asker():
        try:
            gpus.acquire(1000, "test")
        except PygmalionError as exc:
            errors.append(exc)
    thread = threading.Thread(target=asker, name="asker")
    thread.start()
    assert wait_until(lambda: gpus._helpers)
    helper_names = [t.name for t in gpus._helpers]
    assert helper_names == ["pygmalion-lease"]
    assert gpus.close(timeout=0.3) == ["pygmalion-lease"], "the helper is blocked in the hub's long wait: it is named, not hidden"
    thread.join(5)
    assert errors and errors[0].message, "the caller that was waiting is released with a clear error"
    gate.set()
    assert wait_until(lambda: not gpus._helpers) and released, "a lease that is granted after the stop is given back at once"
    assert gpus.close() == []


# ------------------------------------------------------------------------------------------------- output readers
@needs_posix
def test_a_program_that_leaves_a_child_holding_its_output_does_not_leave_the_reader_running(tmp_path, monkeypatch):
    monkeypatch.setattr(procs, "READER_JOIN_S", 0.3)
    before = threads_now()
    started = time.monotonic()
    result = procs.run_streaming(["sh", "-c", "sleep 30 & echo hello"], cwd=tmp_path)
    assert result.returncode == 0 and "hello" in result.tail
    assert time.monotonic() - started < 10, "did not wait for the child that holds the pipe"
    assert wait_until(lambda: not [t for t in threads_now() - before if t.is_alive()]), "the reader ended because the group was killed"
