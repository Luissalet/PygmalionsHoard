"""Brief bench: the same brief to several models with time, tokens and cost, the grid, the blind judge, command-line agents, and (with a browser) the real renderer."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from evaluation_tests.conftest import tool
from evaluation_tests.helpers import add_gguf, build_services
from galton_hoard.briefbench import (BRIEFS, BY_ID, DIMENSIONS, build_messages, custom, extract_answer, inline_libraries, judge_messages, library_file, parse_judge, public,
                                     static_check)
from galton_hoard.briefbench import cli as cli_lib
from galton_hoard.briefbench import shot as shots
from galton_hoard.briefbench.service import cost_of, normalise_brief_settings, parse_prices
from galton_hoard.errors import GaltonError
from galton_hoard.main import create_app

PAGE = ("<!doctype html><html><head><title>t</title></head><body style='margin:0'><canvas id=c width=1280 height=720></canvas><script>"
        "const x=document.getElementById('c').getContext('2d');let t=0;function f(){t++;x.fillStyle='#102030';x.fillRect(0,0,1280,720);x.fillStyle='#ff8800';x.fillRect(100+t*5,200,160,160);requestAnimationFrame(f)}f();"
        "</script></body></html>")
GOOD_JUDGE = {"scores": {"fidelity": 8, "craft": 6, "polish": 5, "technical": 9}, "summary": "Solid but plain.", "problems": ["generic palette"]}
SCENE = "Build a 3D scene with Three.js: a dragon circles above a castle."


# ---------------------------------------------------------------- the briefs and the contract
def test_the_catalogue_holds_page_and_text_briefs_with_unique_ids():
    assert len({b["id"] for b in BRIEFS}) == len(BRIEFS) == len(BY_ID) >= 6
    assert {b["kind"] for b in BRIEFS} == {"page", "text"} and all(len(b["prompt"]) > 80 for b in BRIEFS)
    assert "dragon" in BY_ID["dragon-castle"]["prompt"] and BY_ID["dragon-castle"]["libraries"] == ["three"]
    shown = public(BY_ID["snake-game"])
    assert "prompt" not in shown and public(BY_ID["snake-game"], full=True)["prompt"] == BY_ID["snake-game"]["prompt"]
    own = custom("  a brief of my own  ", index=2, kind="text")
    assert own["id"] == "own-2" and own["prompt"] == "a brief of my own" and own["kind"] == "text"


def test_the_reply_is_read_as_a_page_or_as_a_text():
    assert extract_answer(f"Here you go.\n```html\n{PAGE}\n```\nEnjoy.", "page") == PAGE
    assert extract_answer("no markup at all", "page") == ""
    assert extract_answer("  La recursividad es...  ", "text") == "La recursividad es..." and extract_answer("", "text") == ""
    check = static_check(PAGE + "<script src='https://cdn.example/x.js'></script>")
    assert check["uses_canvas"] and check["external_references"] == 1 and not check["loads_library"]


def test_libraries_are_offered_by_file_and_put_into_the_page_when_it_is_rendered(tmp_path):
    (tmp_path / "three.min.js").write_text("window.THREE={v:1};/* </script> inside */", encoding="utf-8")
    assert library_file(str(tmp_path), "three") == "three.min.js" and library_file(str(tmp_path), "d3") is None and library_file(str(tmp_path), "../x") is None
    with_lib = build_messages(BY_ID["dragon-castle"], {"three": "three.min.js"})
    assert 'lib/three.min.js' in with_lib[0]["content"] and "not available" not in with_lib[0]["content"]
    without = build_messages(BY_ID["dragon-castle"], {})
    assert "three is not available" in without[0]["content"] and "lib/" not in without[0]["content"]
    assert "Respect any length" in build_messages(BY_ID["find-the-bug"], {})[0]["content"] or build_messages(BY_ID["find-the-bug"], {})[0]["role"] == "system"
    page = '<html><body><script src="lib/three.min.js"></script><script src="lib/missing.js"></script><script src="https://x/y.js"></script></body></html>'
    out = inline_libraries(page, str(tmp_path))
    assert "window.THREE={v:1}" in out and "<\\/script> inside" in out and 'src="lib/missing.js"' in out and 'src="https://x/y.js"' in out and 'src="lib/three.min.js"' not in out
    assert static_check(page)["loads_library"] and inline_libraries(page, "") == page


def test_judge_scores_are_parsed_clamped_and_replies_without_scores_are_refused():
    v = parse_judge("Sure!\n" + json.dumps({**GOOD_JUDGE, "scores": {**GOOD_JUDGE["scores"], "craft": 14, "polish": 0}}))
    assert v["scores"]["craft"] == 10.0 and v["scores"]["polish"] == 1.0 and v["dimensions_scored"] == 4 and v["mean"] == round((8 + 10 + 1 + 9) / 4, 2)
    assert parse_judge("I loved it") is None and parse_judge('{"scores": {}}') is None and parse_judge('{"scores": {"fidelity": "high"}}') is None
    assert set(DIMENSIONS) == {"fidelity", "craft", "polish", "technical"}


def test_the_judge_never_gets_a_model_name_and_a_blind_judge_gets_the_source():
    brief = {**custom("a brief about something", kind="page", criteria="must have a night mode")}
    seeing = judge_messages(brief, "SRC", {"blank": False, "moves": True}, {"motion_ms": 1200, "page_errors": []}, {"chars": 3}, images=True)
    blind = judge_messages(brief, "<html>SOURCE</html>", {"blank": False}, {"page_errors": ["boom"]}, {"chars": 3}, images=False)
    assert "pictures follow" in seeing[1]["content"] and "SRC" not in seeing[1]["content"] and "night mode" in seeing[1]["content"]
    assert "cannot see the page" in blind[1]["content"] and "SOURCE" in blind[1]["content"] and "boom" in blind[1]["content"]
    text_brief = custom("a brief about something", kind="text")
    assert "ANSWER:\nthe words" in judge_messages(text_brief, "the words", {}, {}, {}, images=False)[1]["content"]


def test_settings_prices_and_costs():
    assert parse_prices(["gpt = 3, 15", "flat = 2", "# off = 9", "broken", "x = a, b", "too = 1, 2, 3"]) == {"gpt": {"input": 3.0, "output": 15.0}, "flat": {"input": 2.0, "output": 2.0}}
    rs = normalise_brief_settings({"settle_ms": 500, "judge_images": "false", "seed": 3})
    assert rs["settle_ms"] == 500 and rs["judge_images"] is False and rs["seed"] == 3 and rs["max_tokens"] == 16000 and rs["motion_ms"] == 1200
    with pytest.raises(GaltonError):
        normalise_brief_settings({"render_timeout_s": 2})
    with pytest.raises(GaltonError):
        normalise_brief_settings({"nonsense": 1})
    assert cost_of({"input": 3.0, "output": 15.0}, 1000, 2000) == round((3000 + 30000) / 1_000_000, 6)
    assert cost_of(None, 1000, 2000) is None and cost_of({"input": 1, "output": 1}, 10, None) is None and cost_of({"input": 0, "output": 0}, 10, 10) == 0.0


# ---------------------------------------------------------------- a whole run with fake models and a fake browser
def fake_render(html_path, out_dir, *, python, settle_ms, motion_ms, timeout_s=30.0, total_timeout_s=120.0):
    """Stands for the browser: writes two pictures (the second differs unless the page says STILL), unless the page says BROKEN."""
    from PIL import Image, ImageDraw
    html = Path(html_path).read_text(encoding="utf-8")
    if "BROKEN" in html:
        return {"ok": False, "error": "net::ERR_FAILED"}
    out_dir.mkdir(parents=True, exist_ok=True)
    for n, name in enumerate(("shot.png", "shot2.png")):
        im = Image.new("RGB", (1280, 720), (16, 32, 48))
        ImageDraw.Draw(im).rectangle([100 + (0 if "STILL" in html else n * 300), 200, 300 + (0 if "STILL" in html else n * 300), 400], fill=(255, 136, 0))
        im.save(out_dir / name)
    return {"ok": True, "files": ["shot.png", "shot2.png"], "title": "t", "text_chars": 10, "elements": 5, "canvases": 1, "svgs": 0, "page_height": 720, "page_width": 1280,
            "console_errors": [], "page_errors": ["boom"] if "ERRORS" in html else [], "blocked_requests": [], "render_s": 0.1, "LIB": "THREE" in html}


def responder(marker: str = "", broken_for: str = "", empty_for: str = ""):
    seen = {"write": [], "judge": [], "images": []}

    def respond(req):
        system, user = req.messages[0]["content"], req.messages[-1]["content"]
        if "demanding reviewer" in system:
            seen["judge"].append(user)
            seen["images"].append(len(req.images))
            return json.dumps(GOOD_JUDGE)
        seen["write"].append(user)
        if empty_for and empty_for in user:
            return ""
        if "building a web page" in system or "web page from a brief" in system:
            page = PAGE.replace("</body>", f"{marker}</body>")
            if broken_for and broken_for in user:
                page = PAGE.replace("</body>", "BROKEN</body>")
            return "Here is the page:\n```html\n" + page + "\n```"
        return f"The answer {marker}: base case and recursive case."
    return respond, seen


@pytest.fixture
def svc(tmp_path):
    s = build_services(tmp_path)
    s.settings.set_many({"briefbench.browser_python": "fake-python"})
    s.briefbench.render_fn = fake_render
    yield s
    s.db.close()


def test_three_models_one_brief_with_time_tokens_cost_and_a_judge(svc):
    r1, seen = responder("ONE")
    add_gguf(svc, "swift", responder=r1, tps=80.0, vision=True)
    add_gguf(svc, "steady", responder=responder("TWO")[0], tps=20.0)
    add_gguf(svc, "pricey", responder=responder("THREE")[0], tps=40.0)
    prices = {"swift": {"input": 1.0, "output": 2.0}, "steady": {"input": 0.5, "output": 1.0}}
    out = tool(svc, "briefbench_start", models=["swift", "steady", "pricey"], judge="swift", brief_ids=["pricing-page", "explain-recursion"], prices=prices, wait_s=60)
    run = out["run"]
    assert run["state"] == "done" and run["n"] == 6 and run["done"] == run["total"] == 12 and run["judge"]["name"] == "swift" and run["same_model_judges"] is True
    assert [m["slot"] for m in run["models"]] == ["m1", "m2", "m3"] and run["progress"]["judge_mode"] == "images"
    res = tool(svc, "briefbench_result")
    by_brief = {b["id"]: b for b in res["briefs"]}
    page_rows = {r["model"]: r for r in by_brief["pricing-page"]["items"]}
    assert set(page_rows) == {"swift", "steady", "pricey"} and all(r["outcome"] == "done" and r["has_shot"] and r["moves"] and r["usable"] for r in page_rows.values())
    # time and speed come from the server: the faster model took less time per answer
    assert page_rows["swift"]["latency_s"] < page_rows["pricey"]["latency_s"] < page_rows["steady"]["latency_s"] and page_rows["swift"]["tokens_per_s"] > page_rows["steady"]["tokens_per_s"]
    assert page_rows["swift"]["cost"] == cost_of(prices["swift"], page_rows["swift"]["tokens_in"], page_rows["swift"]["tokens_out"]) and page_rows["swift"]["cost"] > 0
    assert page_rows["pricey"]["cost"] is None                      # no price: unknown, not zero
    assert page_rows["swift"]["mean"] == round((8 + 6 + 5 + 9) / 4, 2) and page_rows["swift"]["scores"]["technical"] == 9.0
    text_rows = {r["model"]: r for r in by_brief["explain-recursion"]["items"]}
    assert all(r["has_text"] and not r["has_shot"] and r["outcome"] == "done" for r in text_rows.values())
    s = res["summary"]
    swift = next(m for m in s["models"] if m["name"] == "swift")
    pricey = next(m for m in s["models"] if m["name"] == "pricey")
    assert swift["answered"] == 2 and swift["judged"] == 2 and swift["cost_unknown"] == 0 and swift["tokens_per_s"] and swift["mean_score"] == 7.0
    assert pricey["cost_total"] is None and pricey["cost_unknown"] == 2
    assert s["fastest"] == "swift" and s["cheapest"] == "steady" and s["same_model_judges"] is True and s["currency"] == "USD"
    assert "| model | outcome | time s |" in res["markdown"] and "pricing" in res["markdown"].lower()
    assert seen["images"].count(2) == 3 and seen["images"].count(0) == 3            # the judge saw two pictures of each page, none for a text
    assert all("swift" not in u and "steady" not in u and "pricey" not in u for u in seen["judge"])     # the judge never gets a model name
    # the item and its files
    item = tool(svc, "briefbench_item", brief="pricing-page", model="steady", answer=True)
    assert item["model_name"] == "steady" and item["answer"].startswith("<!doctype html>") and item["files"]["shot.png"] and item["price"] == prices["steady"] and item["tokens_per_s"]
    assert tool(svc, "briefbench_item", brief="explain-recursion", model="m2", answer=True)["answer"].startswith("The answer TWO")
    assert tool(svc, "briefbench_list")["count"] == 1 and tool(svc, "briefbench_status")["run"]["id"] == run["id"]
    # the grid of a brief is one PNG with every model
    grid = svc.briefbench.grid_path(run["id"], "pricing-page")
    from PIL import Image
    with Image.open(grid) as im:
        assert im.width == 3 * (480 + 16) and im.height > 270
    assert svc.briefbench.grid_path(run["id"], "explain-recursion").is_file()


def test_failures_count_against_the_model_but_a_dead_server_is_none_not_zero(svc):
    add_gguf(svc, "wobbly", responder=responder("W", broken_for="pricing", empty_for="Explica")[0])
    add_gguf(svc, "fine", responder=responder("F")[0])
    run = tool(svc, "briefbench_start", models=["wobbly", "fine"], brief_ids=["pricing-page", "explain-recursion"], wait_s=60)["run"]
    res = tool(svc, "briefbench_result", run=run["id"])
    rows = {(b["id"], r["model"]): r for b in res["briefs"] for r in b["items"]}
    assert rows[("pricing-page", "wobbly")]["outcome"] == "unrendered" and rows[("pricing-page", "wobbly")]["reason"] == "render_failed" and rows[("pricing-page", "wobbly")]["mean"] is None
    assert rows[("explain-recursion", "wobbly")]["outcome"] == "none" and rows[("explain-recursion", "wobbly")]["reason"] == "empty"
    assert rows[("pricing-page", "fine")]["outcome"] == "done" and rows[("pricing-page", "fine")]["scores"] == {}          # no judge: no scores
    wobbly = next(m for m in res["summary"]["models"] if m["name"] == "wobbly")
    assert wobbly["usable"] == 0 and wobbly["failed"] == 1 and wobbly["none"] == 1 and wobbly["mean_score"] is None


def test_a_model_that_cannot_be_used_does_not_stop_the_others(svc, monkeypatch):
    add_gguf(svc, "gone", responder=responder()[0])
    add_gguf(svc, "here", responder=responder("H")[0])
    original = svc.runner.session
    from contextlib import contextmanager

    @contextmanager
    def session(contestant, rs, cancel, note=lambda s: None):
        if contestant["name"] == "gone":
            raise GaltonError("unavailable", "load_local_off", name="gone")
        with original(contestant, rs, cancel, note) as s:
            yield s
    monkeypatch.setattr(svc.runner, "session", session)
    run = tool(svc, "briefbench_start", models=["gone", "here"], brief_ids=["explain-recursion"], wait_s=60)["run"]
    assert run["state"] == "done"
    rows = {r["model"]: r for r in tool(svc, "briefbench_result")["briefs"][0]["items"]}
    assert rows["gone"]["outcome"] == "none" and rows["gone"]["reason"] == "session" and rows["here"]["outcome"] == "done"


def test_briefs_written_in_the_call_with_their_own_kind_and_criteria(svc):
    r, seen = responder("X")
    add_gguf(svc, "scribe", responder=r)
    rj, seen_judge = responder()
    add_gguf(svc, "judge", responder=rj)
    run = tool(svc, "briefbench_start", models=["scribe"], judge="judge", brief="A page with a big clock that shows the time in three cities.", title="Clock", criteria="needs three clocks",
               briefs=[{"prompt": "Write two sentences about the sea.", "kind": "text", "title": "Sea"}], wait_s=60)["run"]
    assert run["state"] == "done" and [b["id"] for b in run["briefs"]] == ["own-1", "own-2"] and run["same_model_judges"] is False
    res = tool(svc, "briefbench_result")
    assert [b["kind"] for b in res["briefs"]] == ["page", "text"] and res["briefs"][0]["items"][0]["mean"] == 7.0
    assert any("needs three clocks" in u for u in seen_judge["judge"]) and any("sea" in u for u in seen_judge["judge"])
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_start", models=["scribe"], brief="short")
    assert e.value.key == "bk_brief_text"
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_start", models=["scribe"], brief="A long enough brief text", kind="video")
    assert e.value.key == "bk_bad_kind"


def test_dry_run_errors_and_catalog(svc):
    add_gguf(svc, "scribe", responder=responder()[0])
    plan = tool(svc, "briefbench_start", models=["scribe"], brief_ids=["dragon-castle", "find-the-bug"], prices={"scribe": 3}, dry_run=True)
    assert plan["calls"] == {"answers": 2, "judge": 0} and plan["browser"]["found"] and plan["models"][0]["priced"] is True and plan["judge"] is None
    assert any("three" in str(n) for n in plan["notes"]) and svc.brief_store.runs() == []        # nothing started; the missing library is said
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_start", models=["scribe"], brief_ids=["no-such-brief"])
    assert e.value.key == "bk_no_brief"
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_start", models=["scribe"], brief_ids=[])
    assert e.value.key == "bk_no_briefs"
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_start", models=["scribe"], brief_ids=["snake-game"], prices={"scribe": {"input": "lots"}})
    assert e.value.key == "bk_bad_price"
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_start", models=["scribe"], judge="cli:x", brief_ids=["snake-game"])
    assert e.value.key == "bk_cli_judge"
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_result", run="bk_missing")
    assert e.value.key == "bk_no_run"
    cat = tool(svc, "briefbench_catalog", kind="text")
    assert cat["matching"] == 2 and all(b["kind"] == "text" for b in cat["briefs"]) and cat["browser"]["found"] and cat["libraries"]["three"] is None
    assert tool(svc, "briefbench_brief", id="snake-game")["prompt"].startswith("Write the game Snake")
    assert tool(svc, "briefbench_catalog", query="orrery")["matching"] == 1


def test_prices_come_from_the_setting_and_the_call_overrides_them(svc):
    svc.settings.set_many({"briefbench.prices": ["scribe = 1, 1", "OTHER = 5", "# off = 9", "broken line"], "briefbench.currency": "EUR"})
    add_gguf(svc, "scribe", responder=responder()[0])
    out = tool(svc, "briefbench_start", models=["scribe"], brief_ids=["find-the-bug"], wait_s=60)["run"]
    row = tool(svc, "briefbench_result")["briefs"][0]["items"][0]
    assert row["cost"] == cost_of({"input": 1.0, "output": 1.0}, row["tokens_in"], row["tokens_out"]) and out["currency"] == "EUR"
    again = tool(svc, "briefbench_start", models=["scribe"], brief_ids=["find-the-bug"], prices={"SCRIBE": {"input": 2, "output": 2}}, wait_s=60)["run"]
    row2 = tool(svc, "briefbench_result", run=again["id"])["briefs"][0]["items"][0]
    assert row2["cost"] == cost_of({"input": 2.0, "output": 2.0}, row2["tokens_in"], row2["tokens_out"])


def test_a_command_line_agent_is_a_contestant_with_unknown_tokens(svc, tmp_path):
    script = tmp_path / "agent.py"
    script.write_text("import sys, pathlib\nbrief = pathlib.Path(sys.argv[1]).read_text(encoding='utf-8')\nprint('CLI saw: ' + brief[:20])\n"
                      "pathlib.Path(sys.argv[2], 'index.html').write_text('<!doctype html><html><body>" + "x" * 250 + "</body></html>', encoding='utf-8')\n", encoding="utf-8")
    line = f'robot = "{sys.executable}" "{script}" {{prompt_file}} {{workdir}}'
    svc.settings.set_many({"briefbench.cli": [line, "# off = nothing"]})
    assert cli_lib.parse(svc.settings.get("briefbench.cli")).keys() == {"robot"} and svc.briefbench.cli_agents() == ["robot"]
    add_gguf(svc, "scribe", responder=responder("S")[0])
    run = tool(svc, "briefbench_start", models=["scribe", "cli:robot"], brief_ids=["pricing-page"], wait_s=60)["run"]
    assert run["state"] == "done" and [m["kind"] for m in run["models"]] == ["model", "cli"]
    rows = {r["model"]: r for r in tool(svc, "briefbench_result")["briefs"][0]["items"]}
    assert rows["robot"]["outcome"] == "done" and rows["robot"]["has_shot"] and rows["robot"]["latency_s"] is not None
    assert rows["robot"]["tokens_in"] is None and rows["robot"]["tokens_out"] is None and rows["robot"]["cost"] is None and rows["robot"]["tokens_per_s"] is None
    assert not (svc.briefbench.root / run["id"] / "pricing-page" / "m2" / "work").exists()          # the scratch folder is gone
    with pytest.raises(GaltonError) as e:
        tool(svc, "briefbench_start", models=["cli:ghost"], brief_ids=["pricing-page"])
    assert e.value.key == "bk_no_cli"
    failing = tmp_path / "fail.py"
    failing.write_text("import sys\nsys.stderr.write('no way')\nsys.exit(3)\n", encoding="utf-8")
    svc.settings.set_many({"briefbench.cli": [f'bad = "{sys.executable}" "{failing}"']})
    out = tool(svc, "briefbench_start", models=["cli:bad"], brief_ids=["explain-recursion"], wait_s=60)["run"]
    row = tool(svc, "briefbench_result", run=out["id"])["briefs"][0]["items"][0]
    assert row["outcome"] == "none" and row["reason"] == "cli_error" and "no way" in str(row["detail"])


def test_without_a_browser_a_page_run_is_refused_up_front_but_a_text_run_goes_on(tmp_path, monkeypatch):
    s = build_services(tmp_path)
    monkeypatch.setattr(shots, "browser_python", lambda configured="", fallback="": "")
    add_gguf(s, "scribe", responder=responder()[0])
    with pytest.raises(GaltonError) as e:
        tool(s, "briefbench_start", models=["scribe"], brief_ids=["snake-game"])
    assert e.value.key == "bk_no_browser"
    assert tool(s, "briefbench_start", models=["scribe"], brief_ids=["explain-recursion"], wait_s=60)["run"]["state"] == "done"
    s.db.close()


def test_cancel_and_delete(svc):
    add_gguf(svc, "scribe", responder=responder()[0])
    card = svc.briefbench.start(models=["scribe"], brief_ids=["explain-recursion", "find-the-bug"], start_thread=False)
    out = tool(svc, "briefbench_cancel", run=card["id"])
    assert out["state"] == "cancelled" and svc.brief_store.run(card["id"])["state"] == "cancelled"
    with pytest.raises(ValueError):
        tool(svc, "briefbench_delete", run=card["id"])
    assert tool(svc, "briefbench_delete", run=card["id"], confirm=True) == {"deleted": card["id"]}
    with pytest.raises(GaltonError):
        svc.brief_store.run(card["id"])


def test_the_screen_gets_the_grid_the_pictures_and_a_sandboxed_page_with_its_library(tmp_path):
    s = build_services(tmp_path)
    libs = tmp_path / "libs"
    libs.mkdir()
    (libs / "three.min.js").write_text("window.THREE={v:1};", encoding="utf-8")
    s.settings.set_many({"briefbench.browser_python": "fake-python", "briefbench.libraries": str(libs)})
    s.briefbench.render_fn = fake_render
    page = PAGE.replace("</body>", '<script src="lib/three.min.js"></script></body>')
    prompts = []
    add_gguf(s, "scribe", responder=lambda req: (prompts.append(req.messages[0]["content"]), "```html\n" + page + "\n```")[1])
    add_gguf(s, "other", responder=lambda req: "```html\n" + PAGE + "\n```")
    run = tool(s, "briefbench_start", models=["scribe", "other"], brief_ids=["dragon-castle"], wait_s=60)["run"]
    assert "lib/three.min.js" in prompts[0]          # the model is told which file to load
    with TestClient(create_app(s.config, services=s), base_url="http://127.0.0.1") as c:
        base = f"/api/briefbench/{run['id']}/dragon-castle"
        grid = c.get(f"{base}/grid.png")
        assert grid.status_code == 200 and grid.headers["content-type"] == "image/png" and grid.content[:4] == b"\x89PNG"
        shot = c.get(f"{base}/m1/shot.png")
        assert shot.status_code == 200 and shot.content[:4] == b"\x89PNG"
        play = c.get(f"{base}/m1/play.html")
        assert play.status_code == 200 and "sandbox allow-scripts" in play.headers["content-security-policy"] and "default-src 'none'" in play.headers["content-security-policy"]
        assert "window.THREE={v:1}" in play.text and 'src="lib/three.min.js"' not in play.text            # the library is put in when the page is played
        source = c.get(f"{base}/m1/index.html")
        assert source.status_code == 200 and 'src="lib/three.min.js"' in source.text and source.headers["content-type"].startswith("text/plain")
        assert c.get(f"{base}/m1/..%2Fx.png").status_code in (400, 404)
        assert c.get(f"{base}/m1/notes.txt").status_code == 404 and c.get(f"{base}/m9/shot.png").status_code == 404
        assert c.get(f"{base}/m1/shot.png", headers={"sec-fetch-site": "cross-site"}).status_code == 403
        assert c.get(f"{base}/grid.png", headers={"sec-fetch-site": "cross-site"}).status_code == 403
        listed = c.post("/api/ui/call", json={"name": "briefbench_list", "arguments": {}}).json()
        assert listed["count"] == 1
    s.db.close()


# ---------------------------------------------------------------- the real renderer (only where a browser is available)
LIVE = os.environ.get("VIDEOBENCH_TEST_PYTHON", "")
# the hermetic fixture moves the profile folders, and a Python with Playwright finds its packages and browsers under the real ones: remember them before that happens
PROFILE = {k: os.environ[k] for k in ("APPDATA", "LOCALAPPDATA", "USERPROFILE", "HOME") if os.environ.get(k)}


@pytest.mark.skipif(not (LIVE or shots.browser_python()), reason="no Python with Playwright")
def test_the_real_renderer_takes_pictures_sees_motion_and_blocks_the_network(tmp_path, monkeypatch):
    for key, value in PROFILE.items():
        monkeypatch.setenv(key, value)
    python = LIVE or shots.browser_python()
    moving = tmp_path / "moving"
    page = tmp_path / "moving.html"
    page.write_text(PAGE.replace("<canvas", "<img src='https://example.invalid/x.png'><canvas"), encoding="utf-8")
    report = shots.render(page, moving, python=python, settle_ms=200, motion_ms=600, total_timeout_s=120)
    assert report["ok"], report
    assert report["files"] == ["shot.png", "shot2.png"] and report["canvases"] == 1 and report["blocked_requests"] == ["https://example.invalid/x.png"] and not report["page_errors"]
    auto = shots.measure(moving, report["files"])
    assert auto["moves"] and auto["not_blank"] and auto["motion"] > 0.002
    still = tmp_path / "still"
    quiet = tmp_path / "still.html"
    quiet.write_text("<!doctype html><body style='margin:0;background:#224'><h1 style='color:#fff'>Hello there</h1></body>", encoding="utf-8")
    report2 = shots.render(quiet, still, python=python, settle_ms=100, motion_ms=300, total_timeout_s=120)
    auto2 = shots.measure(still, report2["files"])
    assert report2["ok"] and report2["text_chars"] == len("Hello there") and auto2["moves"] is False and auto2["not_blank"]
    broken = tmp_path / "broken.html"
    broken.write_text("<!doctype html><body><script>throw new Error('kaput')</script></body>", encoding="utf-8")
    report3 = shots.render(broken, tmp_path / "b", python=python, settle_ms=100, motion_ms=0, total_timeout_s=120)
    assert report3["ok"] and any("kaput" in e for e in report3["page_errors"])
    webgl = tmp_path / "webgl.html"
    webgl.write_text("<!doctype html><body><canvas id=c width=200 height=100></canvas><script>window.__gl=!!document.getElementById('c').getContext('webgl');"
                     "document.title=window.__gl?'webgl-ok':'webgl-missing'</script></body>", encoding="utf-8")
    assert shots.render(webgl, tmp_path / "g", python=python, settle_ms=100, motion_ms=0, total_timeout_s=120)["title"] == "webgl-ok"
