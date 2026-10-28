"""Video bank: the briefs and their provenance, the page contract, the jury, a whole run with a fake model and a fake renderer, and (with a browser) the real renderer."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evaluation_tests.conftest import tool
from evaluation_tests.helpers import add_gguf, build_services
from galton_hoard.errors import GaltonError
from galton_hoard.main import create_app
from galton_hoard.videobench import (DIMENSIONS, PROMPTS, PROMPTS_BY_ID, build_messages, extract_html, jury_messages, parse_jury, public, select, static_check)
from galton_hoard.videobench import render as rendering
from galton_hoard.videobench.jury import SCORED
from galton_hoard.videobench.service import normalise_video_settings

PAGE = ("<!doctype html><html><head><title>t</title></head><body style='margin:0'><canvas id=c width=1280 height=720></canvas><script>"
        "window.DURATION=6;const c=document.getElementById('c'),x=c.getContext('2d');"
        "window.seek=function(t){x.fillStyle='#102030';x.fillRect(0,0,1280,720);x.fillStyle='#ff8800';x.fillRect(100+t*150,200,160,160);};window.seek(0);</script></body></html>")
GOOD_JURY = {"scores": {"style_fidelity": 7, "concept_wow": 6, "motion_craft": 7, "design_typography": 6, "finish_texture": 5, "technical": 8}, "summary": "Solid but plain.",
             "critical": ["generic palette"], "fixes": [{"t": "2s", "problem": "static title", "fix": "ease it in over 12 frames"}]}


# ---------------------------------------------------------------- the briefs
def test_the_briefs_are_complete_prompts_with_their_provenance():
    assert len(PROMPTS) == len(PROMPTS_BY_ID) >= 100
    for p in PROMPTS:
        assert len(p["prompt"]) > 400 and p["chars"] == len(p["prompt"])
        assert p["author"] and p["author_url"].startswith("https://") and p["post_url"].startswith("https://")
        assert p["category"] in ("motion", "3d", "interactive", "explainer")
    assert any(p["standalone"] for p in PROMPTS) and any(not p["standalone"] for p in PROMPTS)
    assert all("attached" not in p["prompt"].lower() for p in PROMPTS if p["standalone"])
    shown = public(PROMPTS[0])
    assert "prompt" not in shown and shown["author"] == PROMPTS[0]["author"] and public(PROMPTS[0], full=True)["prompt"] == PROMPTS[0]["prompt"]


def test_selection_is_deterministic_and_standalone_by_default():
    a, b = select(count=10, seed=7), select(count=10, seed=7)
    assert [p["id"] for p in a] == [p["id"] for p in b] and len(a) == 10 and all(p["standalone"] for p in a)
    assert [p["id"] for p in select(count=10, seed=8)] != [p["id"] for p in a]
    assert {p["category"] for p in select(category="explainer", count=50)} == {"explainer"}
    assert [p["id"] for p in select([PROMPTS[3]["id"], "nope"])] == [PROMPTS[3]["id"]]


# ---------------------------------------------------------------- the contract
def test_the_reply_is_read_as_a_html_document():
    assert extract_html(f"Here you go.\n```html\n{PAGE}\n```\nEnjoy.") == PAGE
    assert extract_html(PAGE) == PAGE
    assert extract_html("no markup at all") == ""
    cut = extract_html("```html\n<!doctype html><html><body><script>window.seek=function(){}")
    assert cut.startswith("<!doctype html>") and "```" not in cut          # a reply cut by the token limit is still a page to try
    assert "seek(t)" in build_messages("a brief")[0]["content"] and build_messages("a brief")[1]["content"].endswith("a brief")


def test_static_check_sees_the_contract_and_the_network():
    ok = static_check(PAGE)
    assert ok["defines_seek"] and ok["defines_duration"] and ok["external_references"] == 0
    bad = static_check("<script src='https://cdn.example/x.js'></script><script>fetch('http://a');window.seek=()=>Date.now()</script>")
    assert bad["external_references"] == 2 and bad["reads_clock"] and bad["defines_seek"] and not bad["defines_duration"]


# ---------------------------------------------------------------- the jury
def test_jury_scores_are_parsed_clamped_and_incomplete_replies_are_refused():
    v = parse_jury("Sure!\n" + json.dumps({**GOOD_JURY, "scores": {**GOOD_JURY["scores"], "concept_wow": 14, "motion_craft": 0}}))
    assert v["scores"]["concept_wow"] == 10.0 and v["scores"]["motion_craft"] == 1.0 and v["dimensions_scored"] == 6
    assert v["mean"] == round(sum(v["scores"].values()) / 6, 2) and v["fixes"][0]["t"] == "2s" and "sound_sync" not in v["scores"]
    assert parse_jury("I loved it") is None and parse_jury('{"scores": {}}') is None and parse_jury('{"scores": {"style_fidelity": "high"}}') is None
    assert set(SCORED) | {"sound_sync"} == set(DIMENSIONS)


def test_a_jury_that_cannot_see_gets_the_facts_and_the_source():
    auto, render = {"blank_frames": 0, "motion_mean": 0.03, "still_steps": 1, "first_last_change": 0.2, "luminance": [0.2, 0.3], "motion": [0.03]}, {"duration": 6, "times": [0, 6], "texts": ["Hi", ""], "has_seek": True, "deterministic": True}
    seeing = jury_messages("brief", auto, render, images=True)
    blind = jury_messages("brief", auto, render, images=False, html="<html>SOURCE</html>")
    assert "contact sheet" in seeing[1]["content"] and "SOURCE" not in seeing[1]["content"]
    assert "cannot see" in blind[1]["content"] and "SOURCE" in blind[1]["content"] and "visible text: Hi" in blind[1]["content"]


def test_settings_are_checked_and_the_run_settings_are_shared():
    rs = normalise_video_settings({"frames": 8, "jury_images": "false", "seed": 3})
    assert rs["frames"] == 8 and rs["jury_images"] is False and rs["seed"] == 3 and rs["max_tokens"] == 16000 and rs["temperature"] == 0.7
    with pytest.raises(GaltonError):
        normalise_video_settings({"frames": 2})
    with pytest.raises(GaltonError):
        normalise_video_settings({"nonsense": 1})


def test_a_perfect_loop_still_counts_as_moving_and_a_frozen_page_does_not(tmp_path):
    from PIL import Image, ImageDraw
    def frames(name, positions):
        folder = tmp_path / name
        folder.mkdir()
        files = []
        for i, x in enumerate(positions):
            im = Image.new("RGB", (1280, 720), (16, 32, 48))
            ImageDraw.Draw(im).rectangle([x, 200, x + 200, 400], fill=(255, 136, 0))
            im.save(folder / f"f{i:02d}.png")
            files.append(f"f{i:02d}.png")
        return folder, files, [float(i) for i in range(len(files))]
    looped = rendering.measure(*frames("loop", [100, 400, 800, 400, 100]))
    frozen = rendering.measure(*frames("still", [100, 100, 100, 100, 100]))
    assert looped["first_last_change"] < 0.001 and looped["max_change_from_first"] > 0.01 and looped["moves"]
    assert not frozen["moves"] and frozen["max_change_from_first"] == 0


def test_the_render_rate_counts_a_rendered_item_whose_jury_failed(tmp_path):
    svc = build_services(tmp_path)
    run = svc.video_store.create_run(model="m", jury="m", prompts=["a", "b"], settings={}, progress={}, total=6, source="test", label="t")
    svc.video_store.upsert_item(run["id"], "a", state="done", outcome="done", auto={"usable": True, "moves": True}, render={"deterministic": True}, mean=5.0, jury={"scores": {"technical": 5}})
    svc.video_store.upsert_item(run["id"], "b", state="done", outcome="none", reason="jury_unreadable", auto={"usable": True, "moves": True}, render={"deterministic": True})
    out = svc.videobench.summary(run["id"])
    assert out["rendered"] == 2 and out["answered"] == 2 and out["render_rate"] == 1.0 and out["usable_rate"] == 1.0 and out["judged"] == 1 and out["none"] == 1
    svc.stop()


# ---------------------------------------------------------------- a whole run with a fake model and a fake renderer
def fake_render(html_path, out_dir, *, python, frames, duration_cap, default_duration, timeout_s=20.0, total_timeout_s=240.0):
    """Stands for the browser: writes ``frames`` PNGs with a rectangle that moves, unless the page is the broken one."""
    from PIL import Image, ImageDraw
    html = Path(html_path).read_text(encoding="utf-8")
    if "BROKEN" in html:
        return {"ok": False, "error": "window.seek is not a function", "has_seek": False}
    out_dir.mkdir(parents=True, exist_ok=True)
    times = [round(6 * i / (frames - 1), 3) for i in range(frames)]
    files = []
    for i, t in enumerate(times):
        im = Image.new("RGB", (1280, 720), (16, 32, 48))
        ImageDraw.Draw(im).rectangle([100 + int(t * 150), 200, 260 + int(t * 150), 360], fill=(255, 136, 0))
        im.save(out_dir / f"f{i:02d}.png")
        files.append(f"f{i:02d}.png")
    return {"ok": True, "has_seek": True, "duration": 6.0, "declared_duration": 6, "times": times, "files": files, "texts": ["" for _ in times], "frame_ms": [5] * frames,
            "deterministic": "NONDET" not in html, "console_errors": [], "page_errors": [], "blocked_requests": [], "render_s": 0.1}


def writer_and_jury(broken_ids=(), nondet_ids=(), bad_jury=False):
    """One model that writes a page when asked for one and scores when asked to judge."""
    seen = {"write": [], "jury": [], "images": []}

    def responder(req):
        system = req.messages[0]["content"]
        user = req.messages[-1]["content"]
        if "jury member" in system:
            seen["jury"].append(user)
            seen["images"].append(len(req.images))
            return "I cannot say." if bad_jury else json.dumps(GOOD_JURY)
        seen["write"].append(user)
        page = PAGE.replace("</body>", "BROKEN</body>") if any(b in user for b in broken_ids) else PAGE
        if any(n in user for n in nondet_ids):
            page = PAGE.replace("</body>", "NONDET</body>")
        return "Here is the page:\n```html\n" + page + "\n```"
    return responder, seen


@pytest.fixture
def svc(tmp_path):
    s = build_services(tmp_path)
    s.settings.set_many({"videobench.browser_python": "fake-python"})
    s.videobench.render_fn = fake_render
    yield s
    s.db.close()


def _wait(svc, run_id):
    entry = svc.videobench.threads.get(run_id)
    if entry:
        entry[0].join(60)


def test_a_run_writes_renders_and_judges_and_every_number_adds_up(svc):
    responder, seen = writer_and_jury()
    add_gguf(svc, "scribe", responder=responder, vision=True)
    ids = [p["id"] for p in select(count=3, seed=1)]
    out = tool(svc, "videobench_start", model="scribe", prompts=ids, wait_s=60)
    run = out["run"]
    assert run["state"] == "done" and run["n"] == 3 and run["progress"]["jury_mode"] == "images" and run["done"] == run["total"] == 9
    res = tool(svc, "videobench_result")
    assert res["summary"]["prompts"] == 3 and res["summary"]["judged"] == 3 and res["summary"]["none"] == 0 and res["summary"]["render_rate"] == 1.0
    assert res["summary"]["mean_score"] == round(sum(GOOD_JURY["scores"].values()) / 6, 2) and res["summary"]["by_dimension"]["technical"] == 8.0
    assert res["summary"]["same_model_judges"] is True and set(res["summary"]["by_category"]) <= {"motion", "3d", "interactive", "explainer"}
    assert all(r["outcome"] == "done" and r["usable"] and r["has_sheet"] and r["moves"] and r["deterministic"] for r in res["items"])
    assert seen["images"] == [3, 3, 3] and len(seen["write"]) == 3 and "| brief |" in res["markdown"]
    # the files of an item and the item detail
    item = tool(svc, "videobench_item", prompt=ids[0], html=True)
    assert item["brief"]["prompt"] and item["brief"]["author"] and item["files"]["sheet.png"] and item["html"].startswith("<!doctype html>")
    assert item["jury"]["fixes"][0]["fix"] and item["auto"]["usable"] and item["render"]["duration"] == 6.0
    sheet = svc.videobench.file_path(run["id"], ids[0], "sheet.png")
    from PIL import Image
    with Image.open(sheet) as im:
        assert im.size == (1920, 810)                                 # 12 frames, 4 columns of 480x270
    for bad in ("../x", "index.htm", "frames/../../x.png", "sheet.PNG"):
        with pytest.raises(GaltonError):
            svc.videobench.file_path(run["id"], ids[0], bad)
    assert tool(svc, "videobench_list")["count"] == 1 and tool(svc, "videobench_status")["run"]["id"] == run["id"]


def test_a_page_that_does_not_render_counts_against_the_model_but_a_dead_server_is_none(svc):
    ids = [p["id"] for p in select(count=4, seed=2)]
    responder, _ = writer_and_jury(broken_ids=[PROMPTS_BY_ID[ids[0]]["prompt"][:80]], nondet_ids=[PROMPTS_BY_ID[ids[1]]["prompt"][:80]])
    add_gguf(svc, "scribe", responder=responder)
    run = tool(svc, "videobench_start", model="scribe", prompts=ids, wait_s=60)["run"]
    res = tool(svc, "videobench_result", run=run["id"])
    broken = next(r for r in res["items"] if r["prompt"] == ids[0])
    assert broken["outcome"] == "unrendered" and broken["reason"] == "render_failed" and broken["mean"] is None
    s = res["summary"]
    assert s["answered"] == 4 and s["rendered"] == 3 and s["render_rate"] == 0.75 and s["judged"] == 3 and s["none"] == 0
    assert s["jury_mode"] == "text"                                    # the fake model has no vision: the jury reads the checks and the source
    clocky = next(r for r in res["items"] if r["prompt"] == ids[1])
    assert clocky["outcome"] == "done" and clocky["deterministic"] is False and clocky["usable"] is False and s["usable"] == 2 and s["deterministic_rate"] == round(2 / 3, 3)


def test_a_jury_without_scores_leaves_none_not_zero(svc):
    ids = [p["id"] for p in select(count=2, seed=3)]
    responder, _ = writer_and_jury(bad_jury=True)
    add_gguf(svc, "scribe", responder=responder)
    run = tool(svc, "videobench_start", model="scribe", prompts=ids, wait_s=60)["run"]
    res = tool(svc, "videobench_result", run=run["id"])
    assert [r["outcome"] for r in res["items"]] == ["none", "none"] and all(r["reason"] == "jury_unreadable" for r in res["items"])
    assert res["summary"]["judged"] == 0 and res["summary"]["mean_score"] is None and res["summary"]["none"] == 2 and res["summary"]["rendered"] == 2


def test_dry_run_and_errors(svc):
    add_gguf(svc, "scribe", responder=writer_and_jury()[0])
    plan = tool(svc, "videobench_start", model="scribe", count=5, seed=4, dry_run=True)
    assert plan["n"] == 5 and plan["browser"]["found"] and plan["calls"] == {"writing": 5, "jury": 5} and plan["same_model_judges"] is True
    assert svc.video_store.runs() == []                              # nothing was started
    with pytest.raises(GaltonError) as e:
        tool(svc, "videobench_start", model="scribe", prompts=["no-such-brief"])
    assert e.value.key == "vb_no_prompt"
    with pytest.raises(GaltonError) as e:
        tool(svc, "videobench_result", run="vb_missing")
    assert e.value.key == "vb_no_run"
    cat = tool(svc, "videobench_catalog", category="explainer", standalone=True, limit=5)
    assert cat["matching"] >= 5 and len(cat["prompts"]) == 5 and all(p["category"] == "explainer" for p in cat["prompts"]) and "license" in cat and cat["scored_dimensions"] == list(SCORED)
    assert tool(svc, "videobench_prompt", prompt=cat["prompts"][0]["id"])["prompt"]


def test_without_a_browser_the_run_is_refused_up_front(tmp_path, monkeypatch):
    s = build_services(tmp_path)
    monkeypatch.setattr(rendering, "browser_python", lambda configured="": "")
    add_gguf(s, "scribe", responder=writer_and_jury()[0])
    with pytest.raises(GaltonError) as e:
        tool(s, "videobench_start", model="scribe", count=1)
    assert e.value.key == "vb_no_browser"
    s.db.close()


def test_cancel_and_delete(svc):
    add_gguf(svc, "scribe", responder=writer_and_jury()[0])
    card = svc.videobench.start(model="scribe", prompts=[p["id"] for p in select(count=2, seed=5)], start_thread=False)
    out = tool(svc, "videobench_cancel", run=card["id"])
    assert out["state"] == "cancelled" and svc.video_store.run(card["id"])["state"] == "cancelled"
    assert svc.videobench.delete(card["id"]) == {"deleted": card["id"]}
    with pytest.raises(GaltonError):
        svc.video_store.run(card["id"])


def test_the_screen_gets_the_frames_and_a_sandboxed_page(tmp_path):
    s = build_services(tmp_path)
    s.settings.set_many({"videobench.browser_python": "fake-python"})
    s.videobench.render_fn = fake_render
    add_gguf(s, "scribe", responder=writer_and_jury()[0])
    pid = select(count=1, seed=6)[0]["id"]
    run = tool(s, "videobench_start", model="scribe", prompts=[pid], wait_s=60)["run"]
    with TestClient(create_app(s.config, services=s), base_url="http://127.0.0.1") as c:
        sheet = c.get(f"/api/videobench/{run['id']}/{pid}/sheet.png")
        assert sheet.status_code == 200 and sheet.headers["content-type"] == "image/png" and sheet.content[:4] == b"\x89PNG"
        play = c.get(f"/api/videobench/{run['id']}/{pid}/play.html")
        assert play.status_code == 200 and "sandbox allow-scripts" in play.headers["content-security-policy"] and "default-src 'none'" in play.headers["content-security-policy"]
        assert "window.seek" in play.text and "videobench:'ready'" in play.text and play.text.index("videobench:'ready'") < play.text.index("window.DURATION=6")
        assert c.get(f"/api/videobench/{run['id']}/{pid}/..%2Fx.png").status_code in (400, 404)
        assert c.get(f"/api/videobench/{run['id']}/{pid}/sheet.png", headers={"sec-fetch-site": "cross-site"}).status_code == 403
        assert c.get(f"/api/videobench/{run['id']}/{pid}/notes.txt").status_code == 404
    s.db.close()


# ---------------------------------------------------------------- the real renderer (only where a browser is available)
LIVE = os.environ.get("VIDEOBENCH_TEST_PYTHON", "")
# the hermetic fixture moves the profile folders, and a Python with Playwright finds its packages and browsers under the real ones: remember them before that happens
PROFILE = {k: os.environ[k] for k in ("APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOME") if os.environ.get(k)}


@pytest.mark.skipif(not (LIVE or rendering.browser_python()), reason="no Python with Playwright")
def test_the_real_renderer_takes_frames_checks_determinism_and_blocks_the_network(tmp_path, monkeypatch):
    for key, value in PROFILE.items():
        monkeypatch.setenv(key, value)
    python = LIVE or rendering.browser_python()
    page = tmp_path / "index.html"
    page.write_text(PAGE.replace("<canvas", "<img src='https://example.invalid/x.png'><canvas"), encoding="utf-8")
    report = rendering.render(page, tmp_path / "frames", python=python, frames=6, duration_cap=30, default_duration=10, total_timeout_s=120)
    assert report["ok"], report
    assert report["has_seek"] and report["duration"] == 6.0 and len(report["files"]) == 6 and report["deterministic"] is True
    assert report["blocked_requests"] == ["https://example.invalid/x.png"] and not report["page_errors"]
    auto = rendering.measure(tmp_path / "frames", report["files"], report["times"])
    assert auto["moves"] and auto["not_blank"] and auto["blank_frames"] == 0
    # a page whose picture depends on the clock is caught, and a page without seek is reported
    clocky = tmp_path / "clock.html"
    clocky.write_text("<!doctype html><body><canvas id=c width=1280 height=720></canvas><script>window.DURATION=4;window.seek=function(t){var x=document.getElementById('c').getContext('2d');x.fillStyle='rgb('+(Date.now()%255)+',0,0)';x.fillRect(0,0,1280,720)}</script>", encoding="utf-8")
    assert rendering.render(clocky, tmp_path / "f2", python=python, frames=4, duration_cap=30, default_duration=10, total_timeout_s=120)["deterministic"] is False
    none = tmp_path / "none.html"
    none.write_text("<!doctype html><body>nothing</body>", encoding="utf-8")
    assert rendering.render(none, tmp_path / "f3", python=python, frames=4, duration_cap=30, default_duration=10, total_timeout_s=120)["ok"] is False
