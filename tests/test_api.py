"""HTTP surface: health, status, dashboard, UI and agent bridges, PWA, static fallback."""

from __future__ import annotations

import hashlib
import json

from conftest import needs_posix
from pygmalion_hoard import SERVICE, __version__
from pygmalion_hoard.main import STATIC_DIR


def ui(client, name, **arguments):
    return client.post("/api/ui/call", json={"name": name, "arguments": arguments})


def test_health_is_cheap_and_names_the_service(client):
    r = client.get("/api/health")
    body = r.json()
    assert r.status_code == 200 and body["service"] == SERVICE == "pygmalion-hoard" and body["version"] == __version__
    assert body["offline"] is True and set(body["counts"]) >= {"bases", "datasets", "artifacts"} and "hoard_link" in body


def test_status_and_dashboard(client):
    assert client.get("/api/status").status_code == 200
    d = client.get("/api/dashboard").json()
    assert {"env", "gpus", "jobs", "artifacts", "counts", "storage"} <= set(d)
    assert [g["index"] for g in d["gpus"]["gpus"]] == [0, 1, 2, 3] and d["gpus"]["allowed"] == [2, 3]


def test_ui_call_runs_a_tool_and_reports_its_errors(client):
    ov = ui(client, "pygmalion_overview").json()
    assert "summary" in ov and ov["allowed_gpus"] == [2, 3]
    r = ui(client, "artifact_get", artifact="nope")
    assert r.status_code == 404 and r.json()["code"] == "not_found" and r.json()["hint"]
    r = ui(client, "dataset_delete", dataset="x")
    assert r.status_code == 400 and r.json()["code"] in ("confirm_required", "not_found")
    assert client.post("/api/ui/call", json={"name": "no_such_tool"}).json()["error"].startswith("Unknown tool")
    assert "train_start" in client.get("/api/ui/tools").json()["tools"]


def test_ui_call_validates_arguments(client):
    r = ui(client, "train_plan", base="b", dataset="d", rank=0)
    assert r.status_code == 400 and "rank" in r.json()["error"]
    r = client.post("/api/ui/call", json={"arguments": {}})
    assert r.status_code == 400


def test_ui_results_are_not_capped_but_the_agent_ones_are(client, base_model):
    for i in range(300):
        client.svc.store.create_artifact("gguf", f"model-{i:03d}" + "x" * 80, path=f"/m/{i}.gguf", size=1)
    full = ui(client, "artifacts_list", kind="gguf", limit=300).json()
    assert len(full["artifacts"]) == 300 and "truncated" not in full
    capped = client.post("/api/agent/call", json={"name": "artifacts_list", "arguments": {"kind": "gguf", "limit": 300}}, headers=client.bearer).json()
    assert len(capped["artifacts"]) < 300 and capped["truncated"]["original_lengths"]


def test_agent_needs_the_token(client):
    assert client.post("/api/agent/call", json={"name": "pygmalion_overview"}).status_code == 401
    assert client.post("/api/agent/call", json={"name": "pygmalion_overview"}, headers={"Authorization": "Bearer nope"}).status_code == 401
    r = client.post("/api/agent/call", json={"name": "pygmalion_overview"}, headers=client.bearer)
    assert r.status_code == 200 and "summary" in r.json()


def test_agent_tool_listing_has_instructions_and_short_first_lines(client):
    body = client.get("/api/agent/tools").json()
    assert "Pygmalion" in body["instructions"] and len(body["tools"]) == 44
    assert all(len(t["description"].splitlines()[0]) <= 110 and t["inputSchema"]["type"] == "object" for t in body["tools"])
    names = {t["name"] for t in body["tools"]}
    assert {"train_start", "quantize_start", "evaluate_start", "publish_ollama", "lineage_graph"} <= names


def test_agent_unknown_tool_and_bad_arguments(client):
    assert client.post("/api/agent/call", json={"name": "no_such"}, headers=client.bearer).status_code == 404
    r = client.post("/api/agent/call", json={"name": "base_get", "arguments": {}}, headers=client.bearer)
    assert r.status_code == 400 and "base" in r.json()["error"]
    r = client.post("/api/agent/call", json={"name": "artifact_get", "arguments": {"artifact": "zz"}}, headers=client.bearer)
    assert r.status_code == 404 and r.json()["code"] == "not_found"


def test_guard_rejects_cross_site_calls(client):
    r = ui(client, "pygmalion_overview")
    assert r.status_code == 200
    r = client.post("/api/ui/call", json={"name": "pygmalion_overview"}, headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"})
    assert r.status_code == 403


def test_settings_roundtrip_through_the_ui(client):
    got = ui(client, "settings_set", values={"train.rank": 32}).json()
    assert str(got["values"]["train.rank"]) == "32" and "specs" in got
    assert str(ui(client, "settings_get").json()["values"]["train.rank"]) == "32"
    r = ui(client, "settings_set", values={"train.rank": 100000})
    assert r.status_code == 400
    r = ui(client, "settings_set", values={"gpus.allowed": "0,2"})
    assert r.status_code == 403 or r.status_code == 400
    assert str(ui(client, "settings_get").json()["values"]["gpus.allowed"]).replace(" ", "") in ("2,3", "[2,3]")


def test_the_token_is_stored_write_only(client):
    out = ui(client, "secret_set", value="hf_abcdefghijklmnop").json()
    assert "hf_abcdefghijklmnop" not in json.dumps(out)
    assert "hf_abcdefghijklmnop" not in json.dumps(ui(client, "settings_get").json())
    assert "hf_abcdefghijklmnop" not in json.dumps(client.get("/api/status").json())
    secrets_file = client.svc.config.secrets_path
    assert secrets_file.is_file() and "hf_abcdefghijklmnop" in secrets_file.read_text(encoding="utf-8")
    ui(client, "secret_set", value="")
    assert "hf_abcdefghijklmnop" not in secrets_file.read_text(encoding="utf-8")


@needs_posix
def test_job_flow_over_http(client, base_model):
    r = ui(client, "ctx_extend_start", model=base_model["id"], factor=2, wait_s=0).json()
    job_id = r["job"]["id"]
    client.svc.jobs.run_until_idle()
    got = ui(client, "job_get", job=job_id).json()
    assert got["state"] == "done" and got["log_tail"] and got["pipeline"][0]["kind"] == "ctx_extend"
    listed = ui(client, "jobs_list", states=["done"]).json()
    assert [j["id"] for j in listed["jobs"]] == [job_id]


def test_pwa_manifest_service_worker_and_static_fallback(client):
    m = client.get("/manifest.webmanifest").json()
    assert m["name"] == "Pygmalion's Hoard" and m["theme_color"].startswith("#")
    sw = client.get("/sw.js")
    assert sw.status_code == 200 and "/api/" in sw.text and _build_id() in sw.text and sw.headers["cache-control"] == "no-cache"
    assert client.get("/api/does-not-exist").status_code == 404
    page = client.get("/somewhere/inside")
    assert page.status_code in (200, 503)
    assert client.get("/../../etc/passwd").status_code in (200, 404, 503)


def test_the_brand_icon_is_served_as_a_png(client):
    r = client.get("/icon-192.png")
    assert r.status_code == 200 and r.headers["content-type"].startswith("image/png") and r.content[:8] == b"\x89PNG\r\n\x1a\n"


def _build_id():
    """What names the worker's cache: the hash of the built index.html (the version when there is no build)."""
    try:
        return hashlib.sha256((STATIC_DIR / "index.html").read_bytes()).hexdigest()[:12]
    except OSError:
        return __version__


def test_the_service_worker_cache_is_named_after_the_build_and_never_holds_the_icons(client, tmp_path):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from pygmalion_hoard.hoard_link.service import install_pwa

    first = client.get("/sw.js").text
    assert f'"pygmalion-hoard-{_build_id()}"' in first
    # a new build (another index.html) is another cache name, and the worker deletes every cache that is not its own
    fake_static = tmp_path / "static"
    fake_static.mkdir()
    (fake_static / "index.html").write_text("<script src='/assets/other.js'></script>", encoding="utf-8")
    app = FastAPI()
    install_pwa(app, name="Pygmalion's Hoard", short_name="Pygmalion", theme="#1d1417", background="#1d1417", cache="pygmalion-hoard", static_dir=fake_static)
    second = TestClient(app).get("/sw.js").text
    assert second.split('"pygmalion-hoard-')[1].split('"')[0] != first.split('"pygmalion-hoard-')[1].split('"')[0]
    assert "caches.delete(name)" in second and 'request.mode === "navigate"' in second and 'startsWith("/assets/")' in second
    # the page is fetched from the network first (a new build is never hidden behind an old index); the icons are never put in a cache by the worker
    assert "icon-" not in second


def test_the_page_and_icons_are_revalidated_and_the_hashed_assets_are_immutable(client):
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert client.get("/some/route").headers["cache-control"] == "no-cache"
    assert client.get("/icon-192.png").headers["cache-control"] == "no-cache"
    assert client.get("/favicon.ico").headers["cache-control"] == "no-cache"
    assert client.get("/manifest.webmanifest").headers["cache-control"] == "no-cache"
    from pygmalion_hoard.main import STATIC_DIR
    assets = sorted((STATIC_DIR / "assets").glob("*.js"))
    if assets:
        assert "immutable" in client.get(f"/assets/{assets[0].name}").headers["cache-control"]
