"""The database, the store, settings validation and the GPU manager."""

import threading
import time

import pytest

from conftest import build_services
from helpers import FakeLeaseFactory, fake_inventory
from pygmalion_hoard.db import Database
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.settings import Settings, normalize, parse_intlist
from pygmalion_hoard.store import Store


@pytest.fixture
def store(tmp_path):
    db = Database(tmp_path / "x.db")
    yield Store(db)
    db.close()


# ------------------------------------------------------------------ db
def test_database_migrates_and_is_idempotent(tmp_path):
    db = Database(tmp_path / "d.db")
    version = db.schema_version
    db.migrate()
    assert db.schema_version == version >= 1
    db.close()
    assert Database(tmp_path / "d.db").schema_version == version


def test_database_uses_wal_and_settings_table(tmp_path):
    db = Database(tmp_path / "d.db")
    assert db.one("PRAGMA journal_mode")[0] == "wal"
    db.set_setting("a", "1")
    db.set_setting("a", "2")
    assert db.get_setting("a") == "2" and db.get_setting("zz", "d") == "d"


def test_transaction_rolls_back_on_error(tmp_path):
    db = Database(tmp_path / "d.db")
    with pytest.raises(RuntimeError):
        with db.transaction():
            db.execute("INSERT INTO settings(key, value) VALUES ('k', 'v')")
            raise RuntimeError("no")
    assert db.get_setting("k") is None


# ------------------------------------------------------------------ store
def test_dataset_names_are_unique_and_found_by_name_or_id(store):
    ds = store.create_dataset("Casa", "chat")
    assert store.dataset("Casa")["id"] == ds["id"] == store.dataset(ds["id"])["id"]
    with pytest.raises(PygmalionError) as exc:
        store.create_dataset("Casa", "chat")
    assert exc.value.code == "conflict"
    with pytest.raises(PygmalionError):
        store.dataset("nada")


def test_versions_are_numbered_and_found(store):
    ds = store.create_dataset("D", "chat")
    for n in (1, 2):
        assert store.next_version_number(ds["id"]) == n
        store.add_version(ds["id"], n, f"/x/v{n}", "h", kind="chat", records=1, usable=1, chars=1, tokens=1, splits={}, recipe={}, stats={})
    assert store.find_version("D")["n"] == 2 and store.find_version("D", 1)["n"] == 1
    assert store.find_version(store.find_version("D", 1)["id"])["n"] == 1
    with pytest.raises(PygmalionError):
        store.find_version("D", 9)
    store.delete_dataset(ds["id"])
    assert store.versions(ds["id"]) == []


def test_artifacts_roundtrip_filters_and_children(store):
    a = store.create_artifact("base", "acme/x", path="/p", size=5, metrics={"ppl": 1.5})
    b = store.create_artifact("adapter", "ad", parents=[a["id"]], recipe={"rank": 8})
    assert store.artifact("ad")["id"] == b["id"] and b["recipe"] == {"rank": 8} and a["metrics"] == {"ppl": 1.5}
    assert [x["id"] for x in store.children(a["id"])] == [b["id"]]
    assert [x["name"] for x in store.artifacts(kind="adapter")] == ["ad"] and store.artifacts(text="acme")[0]["id"] == a["id"]
    store.update_artifact(b["id"], pinned=True, notes="n")
    assert [x["id"] for x in store.artifacts(pinned=True)] == [b["id"]]
    assert store.find_artifact_by_path("/p")["id"] == a["id"] and store.find_artifact_by_name("base", "acme/x")["id"] == a["id"]
    store.delete_artifact(b["id"])
    assert store.find_artifact_by_name("adapter", "ad") is None


def test_unknown_artifact_kind_is_rejected(store):
    with pytest.raises(PygmalionError):
        store.create_artifact("spaceship", "x")


def test_jobs_lifecycle_and_ordering(store):
    j1 = store.create_job("train", "gpu", {"a": 1})
    j2 = store.create_job("convert", "cpu", {}, then=[{"kind": "quantize"}], pipeline_id=j1["id"], step=1)
    assert j1["state"] == "queued" and j2["then"] == [{"kind": "quantize"}] and j2["pipeline_id"] == j1["id"]
    assert store.next_queued("gpu")["id"] == j1["id"] and store.next_queued("cpu")["id"] == j2["id"]
    store.update_job(j1["id"], state="running", progress={"step": 3}, cancel_requested=True)
    got = store.job(j1["id"])
    assert got["progress"] == {"step": 3} and got["cancel_requested"] and store.next_queued("gpu") is None
    assert [j["id"] for j in store.jobs(pipeline_id=j1["id"], oldest_first=True)] == [j1["id"], j2["id"]]
    assert store.counts()["jobs_running"] == 1 and store.counts()["jobs_active"] == 2


def test_mark_interrupted_only_touches_running_jobs(store):
    a, b, c = (store.create_job("train", "gpu", {}) for _ in range(3))
    store.update_job(a["id"], state="running")
    store.update_job(b["id"], state="waiting_gpu")
    assert store.mark_interrupted() == 2
    assert [store.job(x["id"])["state"] for x in (a, b, c)] == ["interrupted", "interrupted", "queued"]


def test_runs_are_kept_in_order(store):
    for i in range(5):
        store.add_run("tool", f"t{i}", i % 2 == 0, i, "d")
    assert [r["ref"] for r in store.runs(limit=2)] == ["t4", "t3"] and len(store.runs("tool")) == 5


# ------------------------------------------------------------------ settings
def test_parse_intlist():
    assert parse_intlist("2, 3;3 4") == [2, 3, 4] and parse_intlist([1, "2"]) == [1, 2] and parse_intlist("") == []
    with pytest.raises(PygmalionError):
        parse_intlist("a")


@pytest.mark.parametrize("key,value", [("train.rank", 0), ("train.rank", "x"), ("train.method", "full"), ("scheduler.paused", "maybe"), ("nope", 1),
                                       ("train.dropout", 2), ("ui.language", "fr")])
def test_invalid_settings_are_rejected(key, value):
    with pytest.raises(PygmalionError):
        normalize(key, value)


def test_normalize_formats():
    assert normalize("train.rank", "16.0") == "16" and normalize("scheduler.paused", True) == "1" and normalize("gpus.allowed", [3, 2, 3]) == "3,2"


def test_settings_defaults_and_overrides(ctx):
    s = ctx.svc.settings
    assert s.get("galton.url") == "http://127.0.0.1:5201" and s.int("train.rank") == 16 and s.bool("scheduler.paused") is False
    s.set({"train.rank": 8})
    assert s.int("train.rank") == 8 and s.train_defaults()["rank"] == 8
    assert set(s.groups()) >= {"gpus", "paths", "defaults", "galton", "publish"}


def test_reserved_gpus_need_confirmation(ctx):
    s = ctx.svc.settings
    with pytest.raises(PygmalionError) as exc:
        s.set({"gpus.allowed": "1,2"})
    assert exc.value.code == "confirm_required" and s.intlist("gpus.allowed") == [2, 3]
    s.set({"gpus.allowed": "1,2"}, confirm_reserved=True)
    assert s.intlist("gpus.allowed") == [1, 2]


def test_allowed_gpus_cannot_be_emptied(ctx):
    with pytest.raises(PygmalionError):
        ctx.svc.settings.set({"gpus.allowed": ""})


def test_a_failed_batch_changes_nothing(ctx):
    s = ctx.svc.settings
    with pytest.raises(PygmalionError):
        s.set({"train.rank": 4, "train.alpha": "bad"})
    assert s.int("train.rank") == 16


# ------------------------------------------------------------------ gpus
def test_inventory_marks_allowed_and_reserved(ctx):
    inv = {g["index"]: g for g in ctx.svc.gpus.inventory()}
    assert inv[0]["reserved"] and not inv[0]["allowed"] and inv[2]["allowed"] and not inv[2]["reserved"]
    assert [g["index"] for g in ctx.svc.gpus.allowed_inventory()] == [2, 3]


def test_status_has_a_note_without_gpus(tmp_path):
    c = build_services(tmp_path, inventory=[])
    try:
        st = c.svc.gpus.status()
        assert st["inventory"] is False and "nvidia-smi" in st["note"]
    finally:
        c.svc.stop()


def test_plan_picks_the_allowed_card_with_most_free_memory(tmp_path):
    c = build_services(tmp_path, inventory=fake_inventory(free={2: 3000, 3: 14000}))
    try:
        plan = c.svc.gpus.plan(10000)
        assert plan["pick"]["gpus"] == [3] and plan["warning"] == ""
        assert c.svc.gpus.plan(60000)["pick"]["fits"] is False
        assert "wait" in c.svc.gpus.plan(15000)["warning"]
    finally:
        c.svc.stop()


def test_acquire_only_asks_for_allowed_gpus(ctx):
    grant = ctx.svc.gpus.acquire(8000, "test")
    assert grant.gpus and set(grant.gpus) <= {2, 3}
    assert all(r["gpu"] in (2, 3) for r in ctx.leases.requests) and ctx.leases.requests[0]["owner"] == "pygmalion"
    grant.release()
    assert len(ctx.leases.released) == len(ctx.leases.acquired)


def test_a_run_that_does_not_fit_is_refused_before_asking_the_hub(ctx):
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.gpus.acquire(90000, "test")
    assert exc.value.code == "gpu_unavailable" and ctx.leases.requests == []


def test_a_big_run_is_split_over_both_allowed_gpus(ctx):
    grant = ctx.svc.gpus.acquire(24000, "test")
    assert sorted(grant.gpus) == [2, 3] and {r["gpu"] for r in ctx.leases.requests} == {2, 3}
    grant.release()


def test_cancel_while_waiting_leaves_without_a_lease(ctx):
    ctx.leases.block()
    cancel = threading.Event()
    seen = []
    threading.Timer(0.3, cancel.set).start()
    start = time.time()
    with pytest.raises(PygmalionError):
        ctx.svc.gpus.acquire(8000, "test", cancel=cancel, on_wait=seen.append)
    assert time.time() - start < 5
    ctx.leases.open()
    time.sleep(0.2)
    assert len(ctx.leases.released) == len(ctx.leases.acquired)


def test_grant_release_is_idempotent(ctx):
    grant = ctx.svc.gpus.acquire(1000, "t")
    grant.release()
    grant.release()
    assert len(ctx.leases.released) == 1
