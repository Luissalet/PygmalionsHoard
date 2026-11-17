"""Website-rebuild bench: the reference sites and their cases, the model's tool loop, scoring with normalized SSIM and checks, the zones that
differ, the run with several models, the public WebsiteBench files, the container benchmark entry and (with a browser) the real worker."""

from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.request
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evaluation_tests.conftest import tool
from evaluation_tests.helpers import add_gguf, build_services
from galton_hoard.agentbench import harbor, spec
from galton_hoard.backends import Completion
from galton_hoard.briefbench import shot as shots
from galton_hoard.errors import GaltonError
from galton_hoard.hoard_link import webfidelity as wf
from galton_hoard.main import create_app
from galton_hoard.webrebuild import tasks as tasks_lib
from galton_hoard.webrebuild import websitebench
from galton_hoard.webrebuild.agent import RebuildAgent, safe_path, snapshot_text, tool_specs
from galton_hoard.webrebuild.service import normalise_rebuild_settings

SITES = tasks_lib.SITES


# ---------------------------------------------------------------- the reference sites
def test_every_reference_site_has_files_and_valid_cases_with_checks_and_pictures():
    sites = tasks_lib.builtin()
    assert [t["id"] for t in sites] == ["ficha", "precios", "tareas", "blog"]
    for t in sites:
        assert (SITES / t["id"] / "index.html").is_file()
        shown = tasks_lib.public(t)
        assert shown["cases"] >= 3 and shown["checkpoints"] >= 2 and shown["checks"] >= 3 and shown["kind"] == "local"
        for case in t["cases"]:
            for step in case["steps"]:
                if "goto" in step and step["goto"] != "/":
                    assert (SITES / t["id"] / step["goto"].lstrip("/")).is_file()
    assert "Volveré dentro de diez días" in (SITES / "blog" / "articulo-1.html").read_text(encoding="utf-8")


def test_sites_given_by_url_get_cases_or_one_picture_per_route():
    t = tasks_lib.url_task({"url": "http://127.0.0.1:9/", "routes": ["/", "/b"]}, 1)
    assert t["id"] == "url-1" and t["url"] == "http://127.0.0.1:9" and [c["steps"] for c in t["cases"]] == [
        [{"goto": "/"}, {"checkpoint": "ruta-1"}], [{"goto": "/b"}, {"checkpoint": "ruta-2"}]]
    with pytest.raises(GaltonError) as e:
        tasks_lib.url_task({"url": "file:///etc"}, 1)
    assert e.value.key == "wr_bad_url"
    with pytest.raises(GaltonError) as e:
        tasks_lib.url_task({"url": "http://x", "cases": [{"id": "a", "steps": [{"fly": 1}]}]}, 1)
    assert e.value.key == "wr_bad_case"
    with pytest.raises(GaltonError) as e:
        tasks_lib.resolve(["nope"])
    assert e.value.key == "wr_no_task"
    with pytest.raises(GaltonError) as e:
        tasks_lib.resolve([])
    assert e.value.key == "wr_no_tasks"
    assert "/articulo-2.html" in tasks_lib.intro(tasks_lib.by_id()["blog"])


def test_settings_have_bounds():
    rs = normalise_rebuild_settings({})
    assert rs["max_steps"] == 30 and rs["threshold"] == 0.99 and rs["widths"] == [1280, 390] and rs["max_tokens"] == 12000
    assert normalise_rebuild_settings({"widths": [390], "threshold": 0.9, "max_steps": 8})["widths"] == [390]
    for bad in ({"max_steps": 2}, {"threshold": 0.2}, {"widths": [100]}, {"widths": [1, 2, 3, 4]}):
        with pytest.raises(GaltonError):
            normalise_rebuild_settings(bad)


# ---------------------------------------------------------------- the model's tool loop
SNAP = {"title": "T", "url": "/", "page": {"width": 1280, "height": 800, "bg": "rgb(255, 255, 255)", "color": "rgb(0, 0, 0)", "font": "Arial"},
        "items": [{"d": 1, "tag": "h1", "sel": "h1", "x": 10, "y": 20, "w": 300, "h": 40, "text": "Hola", "font": "700 32px/40px Arial", "color": "rgb(0, 0, 0)"},
                  {"d": 1, "tag": "a", "sel": "a.more", "x": 10, "y": 80, "w": 60, "h": 20, "text": "Más", "href": "b.html"}]}


def scripted(*turns):
    """A chat that answers with the given turns in order (a dict of tool calls, a text, or a Completion)."""
    sent = []
    queue = list(turns)

    def chat(messages, tools):
        sent.append(messages)
        turn = queue.pop(0) if queue else {"finish": {}}
        if isinstance(turn, Completion):
            return turn
        if isinstance(turn, str):
            return Completion(text=turn, prompt_tokens=10, completion_tokens=5, latency_ms=100)
        return Completion(tool_calls=[{"name": n, "arguments": a} for n, a in turn.items()], prompt_tokens=100, completion_tokens=50, latency_ms=200)
    return chat, sent


def test_snapshot_text_and_tool_specs():
    text = snapshot_text(SNAP)
    assert 'h1 h1 @10,20 300x40 "Hola" font=700 32px/40px Arial' in text and "href=b.html" in text and text.startswith("title: T")
    assert snapshot_text({}) and len(snapshot_text({**SNAP, "items": SNAP["items"] * 400}, limit=500)) < 600
    names = [t["function"]["name"] for t in tool_specs()]
    assert names == ["browse", "click", "fill", "press", "back", "write_file", "read_file", "list_files", "preview", "finish"]


def test_paths_are_kept_inside_the_site(tmp_path):
    assert safe_path(tmp_path, "css/a.css") == tmp_path / "css" / "a.css"
    for bad in ("../x.html", "a.exe", "", "/../../x.css", "a b.html"):
        assert safe_path(tmp_path, bad) is None


def test_the_agent_explores_writes_previews_and_finishes(tmp_path):
    explored = []

    def explore(which, width, action):
        explored.append((which, width, action))
        return {"ok": True, "snapshot": SNAP}
    chat, sent = scripted({"browse": {"path": "/", "width": 390}}, {"click": {"target": "text=Más"}},
                          {"write_file": {"path": "index.html", "content": "<h1>Hola</h1>"}, "write_file ": {}},
                          {"write_file": {"path": "../evil.html", "content": "x"}},
                          {"preview": {"path": "/"}}, {"list_files": {}}, {"read_file": {"path": "index.html"}}, {"finish": {"note": "listo"}})
    res = RebuildAgent(chat, explore, tmp_path / "site").run("rebuild it")
    assert res.stop == "finished" and res.note == "listo" and res.files == ["index.html"] and res.steps == 8
    assert (tmp_path / "site" / "index.html").read_text(encoding="utf-8") == "<h1>Hola</h1>" and not (tmp_path / "evil.html").exists()
    assert explored == [("reference", 390, {"goto": "/"}), ("reference", 1280, {"click": "text=Más"}), ("candidate", 1280, {"goto": "/"})]
    assert res.tokens_in == 800 and res.tokens_out == 400 and res.latency_ms == 1600
    last = sent[-1]
    tool_msgs = [m for m in last if m["role"] == "tool"]
    assert any("ruta no válida" in m["content"] for m in tool_msgs) and any(m["content"] == "<h1>Hola</h1>" for m in tool_msgs)
    assert any(m.get("tool_calls") for m in last if m["role"] == "assistant")
    assert [m for m in last if m["role"] == "tool"][0]["content"].endswith("[resultado antiguo omitido]")      # old snapshots are cut, recent ones kept


def test_a_model_that_writes_the_page_in_the_text_keeps_it_and_one_that_never_calls_tools_stops(tmp_path):
    chat, _ = scripted("Aquí está:\n```html\n<!doctype html><html><body>hola</body></html>\n```", "ya está")
    res = RebuildAgent(chat, lambda *a: {}, tmp_path / "s").run("x")
    assert res.stop == "no_tools" and res.files == ["index.html"] and "hola" in (tmp_path / "s" / "index.html").read_text(encoding="utf-8")
    broken, _ = scripted(Completion(error="connection refused"))
    res = RebuildAgent(broken, lambda *a: {}, tmp_path / "t").run("x")
    assert res.stop == "server_error" and "refused" in res.error and res.files == []
    endless, _ = scripted(*[{"list_files": {}}] * 10)
    res = RebuildAgent(endless, lambda *a: {}, tmp_path / "u", max_steps=4).run("x")
    assert res.stop == "steps" and res.steps == 4


# ---------------------------------------------------------------- a run with a fake browser
class FakeBrowser:
    """Reads the page a case visits over HTTP and draws a picture from its bytes: the same page draws the same picture.
    Checks of text pass when the page's source holds the text; every other check passes."""

    def __init__(self, python=""):
        self.calls = []

    def _page(self, base, route):
        try:
            with urllib.request.urlopen(base.rstrip("/") + (route if route != "/" else "/index.html"), timeout=5) as r:
                return r.read()
        except OSError:
            return b""

    def _draw(self, html: bytes, path: Path, width: int):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (width, 200), (255, 255, 255))
        d = ImageDraw.Draw(img)
        digest = hashlib.sha256(html.replace(b"\r\n", b"\n")).digest()  # a checkout may turn LF into CRLF
        for i in range(0, 24, 4):
            x, y = digest[i] * width // 256, digest[i + 1] * 200 // 256
            d.rectangle([x, y, x + 60, y + 40], fill=(digest[i + 2], digest[i + 3], 90))
        img.save(path)

    def call(self, cmd, timeout_s=60):
        self.calls.append(cmd["cmd"])
        if cmd["cmd"] == "explore":
            return {"ok": True, "snapshot": SNAP}
        if cmd["cmd"] != "case":
            return {"ok": True}
        out = Path(cmd["out"])
        out.mkdir(parents=True, exist_ok=True)
        route, html, checks, shots_ = "/", b"", [], []
        for n, step in enumerate(cmd["case"]["steps"], 1):
            if "goto" in step:
                route = step["goto"]
                html = self._page(cmd["base"], route)
            elif "expect_text" in step:
                ok = step["expect_text"].encode("utf-8") in html
                checks.append({"step": n, "kind": "expect_text", "ok": ok, "detail": "" if ok else "missing"})
            elif "checkpoint" in step:
                name = f"{cmd['case']['id']}--{step['checkpoint']}--{cmd['width']}.png"
                self._draw(html, out / name, int(cmd["width"]))
                shots_.append({"name": step["checkpoint"], "route": route, "width": cmd["width"], "key": wf.checkpoint_key(route, cmd["width"], step["checkpoint"]),
                               "file": str(out / name)})
        return {"id": cmd["case"]["id"], "checks": checks, "checkpoints": shots_, "error": ""}

    def close(self):
        self.calls.append("close")


def builder(copy_of: str):
    """A model that browses once, then writes the reference's own files (a perfect copy) or a poor page, then finishes."""
    def respond(req):
        done_tools = [m for m in req.messages if m["role"] == "tool"]
        if not done_tools:
            return {"text": "", "tool_calls": [{"name": "browse", "arguments": {"path": "/"}}]}
        if len(done_tools) == 1:
            if copy_of:
                calls = [{"name": "write_file", "arguments": {"path": p.name, "content": p.read_text(encoding="utf-8")}} for p in sorted((SITES / copy_of).iterdir())]
            else:
                calls = [{"name": "write_file", "arguments": {"path": "index.html", "content": "<!doctype html><html><body>hola</body></html>"}}]
            return {"text": "", "tool_calls": calls}
        return {"text": "", "tool_calls": [{"name": "finish", "arguments": {"note": "hecho"}}]}
    return respond


@pytest.fixture
def svc(tmp_path):
    s = build_services(tmp_path)
    s.settings.set_many({"webrebuild.browser_python": "fake-python"})
    s.webrebuild.browser_factory = FakeBrowser
    yield s
    s.db.close()


def test_two_models_rebuild_two_sites_scored_with_ssim_checks_cost_and_diff_pictures(svc):
    add_gguf(svc, "copista", responder=lambda req: builder(req_site(req))(req), tps=60.0)
    add_gguf(svc, "vago", responder=builder(""), tps=30.0)
    out = tool(svc, "webrebuild_start", models=["copista", "vago"], sites=["ficha", "blog"], prices={"copista": {"input": 1, "output": 2}}, wait_s=60)
    run = out["run"]
    assert run["state"] == "done" and run["done"] == run["total"] == 4 and run["widths"] == [1280, 390]
    res = tool(svc, "webrebuild_result")
    rows = {(t["id"], r["model_name"]): r for t in res["tasks"] for r in t["rows"]}
    good, bad = rows[("ficha", "copista")], rows[("ficha", "vago")]
    # the fake browser runs no scripts: the cases that need a click fail on the reference itself and are left out, with a warning
    assert good["outcome"] == "done" and good["score"] == 1.0 and good["passed"] == good["cases"] == 2 and good["visual_mean"] == 1.0
    ficha = next(t for t in res["tasks"] if t["id"] == "ficha")
    assert ficha["reference_cases"] == 6 and ficha["reference_valid"] == 2
    assert any("ficha-cesta@1280" in w for w in svc.web_store.run(run["id"])["progress"]["warnings"])
    assert bad["score"] == 0.0 and bad["passed"] == 0 and bad["functional_rate"] < 1 and bad["visual_mean"] < 0.99
    assert good["cost"] == round((good["tokens_in"] * 1 + good["tokens_out"] * 2) / 1e6, 6) and bad["cost"] is None
    assert good["steps"] == 3 and good["stop"] == "finished" and good["build_s"] is not None
    assert rows[("blog", "copista")]["score"] == 1.0
    s = res["summary"]
    assert s["best"] == "copista" and {m["model"]: m["sites"] for m in s["models"]} == {"copista": 2, "vago": 2}
    assert next(m for m in s["models"] if m["model"] == "vago")["cost"] is None
    item = tool(svc, "webrebuild_item", site="ficha", model="vago")
    case = next(c for c in item["case_results"] if c["case"] == "ficha-inicio" and c["width"] == 1280)
    cp = case["checkpoints"][0]
    assert not case["passed"] and case["failed_checks"] and cp["score"] < 0.99 and cp["regions"] and cp["diff_url"].endswith(".png")
    assert cp["reference_url"].startswith(f"/api/webrebuild/{run['id']}/ficha/reference/")
    code = tool(svc, "webrebuild_code", site="ficha", model="m2")
    assert code["files"] == ["index.html"] and "hola" in code["content"]
    assert tool(svc, "webrebuild_list")["count"] == 1 and tool(svc, "webrebuild_status")["run"]["id"] == run["id"]
    with TestClient(create_app(svc.config, services=svc), base_url="http://127.0.0.1") as c:
        r = c.get(cp["diff_url"])
        assert r.status_code == 200 and r.headers["content-type"] == "image/png" and r.content[:4] == b"\x89PNG"
        assert c.get(cp["reference_url"]).status_code == 200
        assert c.get(f"/api/webrebuild/{run['id']}/ficha/m2/..%2Frun.json").status_code == 404
        assert c.get(cp["diff_url"], headers={"sec-fetch-site": "cross-site"}).status_code == 403


def req_site(req) -> str:
    intro = next(m["content"] for m in req.messages if m["role"] == "user")
    return "blog" if "Blog" in intro else "ficha"


def test_a_server_that_does_not_answer_leaves_the_sites_as_none(svc):
    add_gguf(svc, "caido", responder=lambda req: {"text": "", "error": "connection refused"})
    tool(svc, "webrebuild_start", models=["caido"], sites=["ficha", "precios", "tareas"], wait_s=60)
    rows = [r for t in tool(svc, "webrebuild_result")["tasks"] for r in t["rows"]]
    assert all(r["outcome"] == "none" for r in rows) and {r["reason"] for r in rows} == {"server_error"}
    assert tool(svc, "webrebuild_result")["summary"]["models"][0]["sites"] == 0


def test_plan_cancel_delete_and_no_browser(svc, monkeypatch):
    add_gguf(svc, "copista", responder=builder("ficha"))
    plan = tool(svc, "webrebuild_start", models=["copista"], sites=["ficha"], urls=[{"url": "http://127.0.0.1:9", "routes": ["/"]}], dry_run=True)
    assert plan["cases_per_model"] == (3 + 1) * 2 and [t["id"] for t in plan["tasks"]] == ["ficha", "url-1"] and plan["browser"]["found"]
    card = svc.webrebuild.start(models=["copista"], task_ids=["ficha"], start_thread=False)
    assert tool(svc, "webrebuild_cancel", run=card["id"])["state"] == "cancelled"
    with pytest.raises(ValueError):
        tool(svc, "webrebuild_delete", run=card["id"])
    assert tool(svc, "webrebuild_delete", run=card["id"], confirm=True) == {"deleted": card["id"]}
    with pytest.raises(GaltonError):
        svc.web_store.run(card["id"])
    cat = tool(svc, "webrebuild_catalog")
    assert len(cat["tasks"]) == 4 and "checkpoint" in cat["step_kinds"] and cat["websitebench"]["imported"] is False
    assert tool(svc, "webrebuild_site", id="tareas")["case_list"][1]["id"] == "tareas-anadir"
    svc.settings.set_many({"webrebuild.browser_python": ""})
    monkeypatch.setattr(shots, "browser_python", lambda configured="", fallback="": "")
    with pytest.raises(GaltonError) as e:
        tool(svc, "webrebuild_start", models=["copista"], sites=["ficha"])
    assert e.value.key == "bk_no_browser"


# ---------------------------------------------------------------- WebsiteBench public files and the container benchmark
INDEX = {"schema": "websitebench.release-manifest.v1", "version": "v1", "task_count": 2, "hf_dataset": "x/y", "hub_dataset": "w/W@v1",
         "tasks": {"airtable": {"file_count": 10}, "medium": {"file_count": 20}}}
TOML = b'[task]\nname = "w/medium"\ndescription = "Reconstruct medium"\nkeywords = ["browser-checks"]\n[metadata]\ndifficulty = "hard"\ntask_type = "fullstack-reconstruction"\n' \
       b'[agent]\ntimeout_sec = 57600.0\n[environment]\ncpus = 8\nmemory_mb = 8192\n'
MANIFEST = {"status": "sealed", "suite_profile": "p", "cases": [{"kind": "http", "tier": "T1"}, {"kind": "browser", "tier": "T2"}, {"kind": "http", "tier": "T2"}]}


def fake_fetch(url: str) -> bytes:
    if url.endswith("websitebench-v1.json"):
        return json.dumps(INDEX).encode()
    if url.endswith("/LICENSE") or url.endswith("/NOTICE"):
        return b"Apache License 2.0"
    if url.endswith("task.toml"):
        return TOML
    if url.endswith("instruction.md"):
        return b"<!-- harbor-canary GUID 0000 -->\n# Reconstruct the offline website\nExplore it."
    if url.endswith("case-manifest.json"):
        return json.dumps(MANIFEST).encode()
    raise OSError("404")


def test_websitebench_public_files_go_to_the_data_folder_with_their_licence(tmp_path):
    assert websitebench.listing(tmp_path)["imported"] is False
    out = websitebench.import_public(tmp_path, tasks=["medium"], fetch=fake_fetch)
    assert out["imported"] == ["medium"] and out["task_count"] == 2 and "Apache" in out["credit"]
    folder = websitebench.folder(tmp_path)
    assert (folder / "LICENSE").read_bytes() == b"Apache License 2.0" and (folder / "NOTICE").is_file() and (folder / "CREDITS.txt").is_file()
    listing = websitebench.listing(tmp_path)
    rows = {r["id"]: r for r in listing["tasks"]}
    assert listing["with_details"] == 1 and rows["airtable"]["details"] is None
    d = rows["medium"]["details"]
    assert d["cases"] == 3 and d["by_kind"] == {"http": 2, "browser": 1} and d["difficulty"] == "hard" and d["cpus"] == 8 and d["agent_timeout_s"] == 57600.0
    text = websitebench.instruction(tmp_path, "medium")
    assert text["instruction"].startswith("# Reconstruct") and "canary" not in text["instruction"]
    with pytest.raises(GaltonError) as e:
        websitebench.import_public(tmp_path, tasks=["ghost"], fetch=fake_fetch)
    assert e.value.key == "wr_wb_no_task"
    with pytest.raises(GaltonError) as e:
        websitebench.instruction(tmp_path, "airtable")
    assert e.value.key == "wr_wb_not_imported"

    def down(url):
        raise OSError("offline")
    with pytest.raises(GaltonError) as e:
        websitebench.import_public(tmp_path, fetch=down)
    assert e.value.key == "wr_wb_fetch"


def test_websitebench_runs_in_containers_through_the_agent_bench(tmp_path):
    b = spec.benchmark("websitebench")
    assert b.dataset == "websitebench/WebsiteBench@v1" and b.kind == "web_rebuild" and b.n > b.dev
    h = spec.parse_harness_line("agent = some-agent")
    job = harbor.job_command(exe="runner", bench=b, harness=h, tasks=["medium"], dataset_dir=tmp_path / "ds", jobs_dir=tmp_path / "jobs", job_name="j",
                             model="m", base_url="http://x/v1", base_env={})
    argv = " ".join(job.argv)
    assert "--allow-environment-host registry.npmjs.org" in argv and "--allow-environment-host snapshot.debian.org" in argv
    plain = harbor.job_command(exe="runner", bench=spec.benchmark("tb2"), harness=h, tasks=["a"], dataset_dir=tmp_path / "ds", jobs_dir=tmp_path / "jobs",
                               job_name="k", model="m", base_url="http://x/v1", base_env={})
    assert "--allow-environment-host" not in plain.argv


# ---------------------------------------------------------------- the real browser, when this computer has one
LIVE = os.environ.get("VIDEOBENCH_TEST_PYTHON", "")
REAL_PY = LIVE or shots.browser_python("")
# the hermetic fixture moves the profile folders, and a Python with Playwright finds its packages and browsers under the real ones
PROFILE = {k: os.environ[k] for k in ("APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOME") if os.environ.get(k)}


@pytest.mark.skipif(not REAL_PY, reason="no Python with Playwright on this computer")
def test_the_real_worker_runs_every_reference_case_and_draws_the_same_picture_twice(tmp_path, monkeypatch):
    for key, value in PROFILE.items():
        monkeypatch.setenv(key, value)
    from galton_hoard.webrebuild.browser import Browser
    from galton_hoard.webrebuild.serve import Site
    task = tasks_lib.by_id()["tareas"]
    with Browser(REAL_PY) as b, Site(SITES / "tareas") as base:
        snap = b.call({"cmd": "explore", "base": base, "width": 1280, "height": 800, "action": {"goto": "/"}})
        assert snap["ok"] and snap["snapshot"]["title"] == "Lista de tareas" and any(i.get("placeholder") for i in snap["snapshot"]["items"])
        after = b.call({"cmd": "explore", "base": base, "width": 1280, "height": 800, "action": {"click": "text=Hechas"}})
        assert after["ok"]
        for case in task["cases"]:
            r = b.call({"cmd": "case", "base": base, "width": 390, "height": 844, "case": case, "out": str(tmp_path / "a")}, timeout_s=120)
            assert not r["error"] and all(c["ok"] for c in r["checks"]), r
        again = b.call({"cmd": "case", "base": base, "width": 390, "height": 844, "case": task["cases"][1], "out": str(tmp_path / "b")}, timeout_s=120)
    first = sorted((tmp_path / "a").glob("tareas-anadir--*.png"))[0]
    second = sorted((tmp_path / "b").glob("tareas-anadir--*.png"))[0]
    assert wf.normalized_ssim(first, second)["score"] == 1.0
    assert re.search(r"--dos--390\.png$", first.name)
