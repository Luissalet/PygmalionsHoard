"""Evaluation of suits: executable checks, aggregation across the three arms with the build case left out, and the export to Darwin's Hoard."""
from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath

import pytest

from evaluation_tests.conftest import tool
from galton_hoard.errors import GaltonError
from galton_hoard.suiteval import checks as C, stats as S

SCHEMA = {"type": "object", "required": ["total", "lines"], "properties": {"total": {"type": "number", "minimum": 0},
                                                                       "lines": {"type": "array", "minItems": 1, "items": {"type": "object", "required": ["part"]}}},
          "additionalProperties": False}


def folder(tmp_path: Path, name: str, files: dict[str, str]) -> Path:
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    for rel, text in files.items():
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text(text, encoding="utf-8")
    return d


# ------------------------------------------------------------------ the executable checks
def test_file_json_numeric_regex_and_judge_checks(tmp_path):
    d = folder(tmp_path, "out", {"report.md": "Total: 120.5 EUR\nsource: https://example.test/p", "data.json": json.dumps({"total": 120.5, "lines": [{"part": "a"}]}),
                                 "empty.txt": ""})
    specs = C.normalize([
        {"kind": "file_exists", "path": "report.md"},
        {"id": "nonempty", "kind": "file_exists", "path": "empty.txt"},
        {"id": "shape", "kind": "json_schema", "path": "data.json", "schema": SCHEMA},
        {"id": "sum", "kind": "numeric", "path": "data.json", "pointer": "total", "expected": 120, "tolerance": 1},
        {"id": "src", "kind": "regex", "path": "report.md", "pattern": r"https?://\S+"},
        {"id": "nolorem", "kind": "regex", "path": "report.md", "pattern": "lorem", "forbid": True},
        {"id": "ans", "kind": "regex", "target": "answer", "pattern": "^ok$", "flags": "im"},
        {"id": "num-value", "kind": "numeric", "value": "n", "expected": 10, "tolerance": 0.1, "relative": True},
        {"id": "good", "kind": "judge", "rubric": "reads well"}], "k1")
    ok, res = C.run_checks(specs, root=d, answer="fine\nOK", values={"n": 10.5}, judgements={"good": {"ok": True, "judge": "logit:m"}})
    by = {r["id"]: r for r in res}
    assert by["c1"]["ok"] and not by["nonempty"]["ok"] and by["shape"]["ok"] and by["sum"]["ok"] and by["src"]["ok"] and by["nolorem"]["ok"] and by["ans"]["ok"]
    assert by["num-value"]["ok"] is True and by["good"]["ok"] is True and by["good"]["judge"] == "logit:m"
    assert ok is False                                   # the empty file failed one check
    only = [s for s in specs if s["id"] != "nonempty"]
    assert C.run_checks(only, root=d, answer="OK", values={"n": 10.5}, judgements={"good": {"ok": True}})[0] is True
    pending, res2 = C.run_checks(only, root=d, answer="OK", values={"n": 10.5})
    assert pending is None and [r for r in res2 if r["id"] == "good"][0]["ok"] is None   # no verdict yet: neither passed nor failed
    assert C.run_checks(only, root=d, answer="OK", values={"n": 99})[0] is False


def test_numeric_tolerance_and_schema_failures_say_why(tmp_path):
    d = folder(tmp_path, "o", {"d.json": json.dumps({"total": -1, "lines": [], "extra": 1})})
    bad = C.run_check(C.normalize([{"kind": "json_schema", "path": "d.json", "schema": SCHEMA}], "k")[0], root=d)
    assert bad["ok"] is False and "minimum" in bad["detail"]
    far = C.run_check(C.normalize([{"kind": "numeric", "path": "d.json", "pointer": "total", "expected": 5, "tolerance": 0.5}], "k")[0], root=d)
    assert far["ok"] is False and "found -1" in far["detail"]
    not_json = folder(tmp_path, "p", {"d.json": "{oops"})
    assert C.run_check(C.normalize([{"kind": "json_schema", "path": "d.json", "schema": {}}], "k")[0], root=not_json)["ok"] is False
    missing = C.run_check({"id": "x", "kind": "file_exists", "path": "nope.txt"}, root=d)
    assert missing["ok"] is False and missing["detail"] == "file not found"


@pytest.mark.parametrize("path", ["../secret.txt", "a/../../b", "/etc/passwd", "C:\\Windows\\win.ini", "c:/x", "~/x", "a//b", "", "..\\x"])
def test_paths_cannot_leave_the_deliverable_folder(tmp_path, path):
    with pytest.raises(GaltonError) as e:
        C.safe_path(tmp_path, path)
    assert e.value.key == "se_bad_path"
    res = C.run_check({"id": "x", "kind": "file_exists", "path": path}, root=tmp_path)
    assert res["ok"] is False


def test_a_symlink_out_of_the_folder_is_refused(tmp_path):
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    d = tmp_path / "d"
    d.mkdir()
    try:
        (d / "link.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("no symbolic links here")
    assert C.run_check({"id": "x", "kind": "file_exists", "path": "link.txt"}, root=d)["ok"] is False


def test_windows_separators_inside_the_folder_are_accepted(tmp_path):
    d = folder(tmp_path, "w", {"sub/file.txt": "x"})
    assert C.safe_path(d, "sub\\file.txt") == (d / "sub" / "file.txt").resolve()
    assert PureWindowsPath("sub\\file.txt").parts == ("sub", "file.txt")


def test_bad_check_specs_fail_when_written():
    for checks, key in [([], "se_case_no_checks"), ([{"kind": "magic"}], "se_check_kind"), ([{"kind": "regex"}], "se_check_field"),
                        ([{"kind": "regex", "pattern": "("}], "se_check_regex"), ([{"kind": "numeric", "expected": "x", "path": "a"}], "se_check_field"),
                        ([{"kind": "json_schema", "path": "a", "schema": {"oneOf": []}}], "se_schema_keyword"),
                        ([{"id": "a", "kind": "judge", "rubric": "r"}, {"id": "a", "kind": "judge", "rubric": "r"}], "se_check_dup")]:
        with pytest.raises(GaltonError) as e:
            C.normalize(checks, "k1")
        assert e.value.key == key, (checks, e.value.key)


# ------------------------------------------------------------------ aggregation
def rec(case, arm, ok, *, run=1, minutes=None, time=None, defects=None, tokens=None):
    return {"case_id": case, "arm": arm, "run": run, "success": ok, "total_time_s": time, "correction_minutes": minutes, "defects": defects, "tokens": tokens}


DEF = {"cases": [{"id": "k1", "title": "built on", "used_for_building": True}, {"id": "k2"}, {"id": "k3"}, {"id": "k4"}],
       "candidate": {"description": "x", "built_from_cases": []}}


def test_aggregation_across_arms_excludes_the_build_case_and_shows_spread():
    records = [
        # the build case: the candidate passes it, the others do not. It must not reach the score.
        rec("k1", "generic", False, time=100), rec("k1", "current", False, time=90), rec("k1", "candidate", True, time=10),
        rec("k2", "generic", False, minutes=30, time=100, defects=3, tokens=1000), rec("k2", "current", True, minutes=10, time=60, defects=1, tokens=900),
        rec("k2", "candidate", True, minutes=5, time=50, defects=0, tokens=800),
        rec("k3", "generic", True, minutes=20, time=80), rec("k3", "current", False, minutes=40, time=70), rec("k3", "candidate", True, time=40),
        rec("k4", "generic", False, time=120), rec("k4", "current", True, time=65), rec("k4", "candidate", True, time=45),
    ]
    out = S.aggregate(DEF, records)
    assert out["cases_total"] == 4 and out["cases_scored"] == 3 and out["excluded"] == [{"case": "k1", "reason": "used_for_building"}]
    a = out["aggregate"]
    assert (a["generic"]["cases"], a["current"]["cases"], a["candidate"]["cases"]) == (3, 3, 3)
    assert a["generic"]["success_rate"] == pytest.approx(1 / 3, abs=1e-4) and a["current"]["success_rate"] == pytest.approx(2 / 3, abs=1e-4)
    assert a["candidate"]["success_rate"] == 1.0                      # with the build case it would be the same, but the generic arm would look worse still
    assert a["candidate"]["success_ci95"][0] < 1.0 and a["candidate"]["success_ci95"][1] == 1.0
    assert a["generic"]["time_s"]["n"] == 3 and a["generic"]["time_s"]["mean"] == pytest.approx((100 + 80 + 120) / 3, abs=1e-3)
    assert a["generic"]["time_s"]["sd"] == pytest.approx(20.0, abs=1e-3) and a["generic"]["time_s"]["min"] == 80 and a["generic"]["time_s"]["max"] == 120
    # human correction minutes are optional: only the cases that have them count
    assert a["generic"]["correction_minutes"]["n"] == 2 and a["generic"]["correction_minutes"]["sum"] == 50
    assert a["candidate"]["correction_minutes"]["n"] == 1 and a["candidate"]["tokens"]["n"] == 1
    p = out["paired"]["candidate_vs_current"]
    assert (p["cases"], p["better"], p["worse"], p["same"]) == (3, 1, 0, 2) and p["sign_test_p"] == 1.0
    assert out["paired"]["candidate_vs_generic"]["better"] == 2
    # the excluded case is still listed, with its records, but marked
    k1 = [c for c in out["per_case"] if c["case"] == "k1"][0]
    assert k1["excluded"] == "used_for_building" and k1["arms"]["candidate"]["success"] == 1.0
    keys = [w.key for w in out["warnings"]]
    assert "se_warn_excluded" in keys and "se_warn_tuned" in keys          # the candidate was run on the case it was built from: flagged


def test_the_score_without_the_exclusion_would_differ():
    records = [rec("k1", "candidate", True), rec("k2", "candidate", False), rec("k3", "candidate", False), rec("k4", "candidate", False)]
    flagged = S.aggregate(DEF, records)["aggregate"]["candidate"]
    plain = S.aggregate({**DEF, "cases": [{**c, "used_for_building": False} for c in DEF["cases"]]}, records)["aggregate"]["candidate"]
    assert flagged["success_rate"] == 0.0 and plain["success_rate"] == 0.25


def test_built_from_cases_also_excludes_and_unmarked_candidates_are_flagged():
    d = {"cases": [{"id": "a"}, {"id": "b"}], "candidate": {"built_from_cases": ["a"]}}
    out = S.aggregate(d, [rec("a", "candidate", True), rec("b", "candidate", True)])
    assert out["excluded"] == [{"case": "a", "reason": "used_for_building"}] and out["cases_scored"] == 1
    none = S.aggregate({"cases": [{"id": "a"}], "candidate": {}}, [rec("a", "candidate", True)])
    assert "se_warn_no_build_cases" in [w.key for w in none["warnings"]]


def test_runs_are_averaged_pending_judge_cases_are_not_counted_and_nothing_scored_is_said():
    d = {"cases": [{"id": "a"}, {"id": "b"}], "candidate": {}}
    out = S.aggregate(d, [rec("a", "current", True, run=1), rec("a", "current", False, run=2), rec("b", "current", None)])
    cur = out["aggregate"]["current"]
    assert cur["cases"] == 1 and cur["pending"] == 1 and cur["success_rate"] == 0.5 and cur["runs"] == 3
    empty = S.aggregate({"cases": [{"id": "a", "used_for_building": True}], "candidate": {}}, [rec("a", "candidate", True)])
    assert empty["cases_scored"] == 0 and "se_warn_no_scored" in [w.key for w in empty["warnings"]]


def test_sign_test_and_wilson_are_sound():
    assert S.sign_test(0, 0) is None and S.sign_test(5, 0) == pytest.approx(0.0625) and S.sign_test(3, 3) == 1.0
    lo, hi = S.wilson(8, 10)
    assert 0.49 < lo < 0.50 and 0.94 < hi < 0.95
    s = S.summarize([1, 2, 3, None])
    assert s["n"] == 3 and s["mean"] == 2 and s["sd"] == 1 and s["median"] == 2


# ------------------------------------------------------------------ the service
def define(svc, tmp_path, **over):
    args = dict(suit="part-budget", name="Budget", cases=[
        {"id": "k1", "title": "built on", "used_for_building": True, "checks": [{"kind": "file_exists", "path": "out.md"}]},
        {"id": "k2", "title": "two", "checks": [{"id": "f", "kind": "file_exists", "path": "out.md"}, {"id": "n", "kind": "numeric", "path": "t.json", "pointer": "total", "expected": 10, "tolerance": 0.5}]},
        {"id": "k3", "title": "three", "checks": [{"id": "j", "kind": "judge", "rubric": "clear"}]}],
        candidate={"description": "a better recipe", "built_from_cases": ["k1"]})
    args.update(over)
    return tool(svc, "pygmalion_suit_eval_define", **args)


def test_define_record_score_and_a_human_adds_minutes_later(svc, tmp_path):
    ev = define(svc, tmp_path)["evaluation"]
    good = folder(tmp_path, "good", {"out.md": "x", "t.json": '{"total": 10.2}'})
    bad = folder(tmp_path, "bad", {"t.json": '{"total": 99}'})
    for case, arm, d in [("k1", "candidate", good), ("k2", "generic", bad), ("k2", "current", good), ("k2", "candidate", good)]:
        tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case=case, arm=arm, deliverable=str(d), total_time_s=10.0 if arm != "generic" else 30.0)
    r = tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k3", arm="current", answer="fine")
    assert r["success"] is None and r["checks"][0]["ok"] is None                   # the judge has not spoken
    j = tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k3", arm="current", judgements={"j": {"ok": True, "judge": "panel"}})
    assert j["success"] is True and j["inputs"]["answer"] == "fine"               # earlier inputs are kept
    again = tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k2", arm="current", correction_minutes=12.5)
    assert again["correction_minutes"] == 12.5 and again["success"] is True and again["defects"] == 0   # minutes added later; checks untouched
    manual = tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k2", arm="generic", defects=4)
    assert manual["defects"] == 4 and manual["success"] is False
    sc = tool(svc, "pygmalion_suit_eval_score", id=ev["id"])
    assert sc["cases_scored"] == 2 and sc["excluded"][0]["case"] == "k1"
    assert sc["aggregate"]["current"]["correction_minutes"]["mean"] == 12.5 and sc["aggregate"]["generic"]["success_rate"] == 0.0
    assert sc["aggregate"]["candidate"]["cases"] == 1
    assert [w.key for w in sc["warnings"] if w.key == "se_warn_tuned"]
    full = tool(svc, "pygmalion_suit_eval_get", id=ev["id"])
    assert len(full["records"]) == 5
    listed = tool(svc, "pygmalion_suit_eval_list", suit="part-budget")
    assert listed["count"] == 1 and listed["evaluations"][0]["excluded"] == ["k1"]


def test_definition_rules(svc, tmp_path):
    with pytest.raises(GaltonError) as e:
        define(svc, tmp_path, suit="Bad Slug")
    assert e.value.key == "se_bad_suit"
    with pytest.raises(GaltonError) as e:
        define(svc, tmp_path, candidate={"built_from_cases": ["nope"]})
    assert e.value.key == "se_build_unknown"
    with pytest.raises(GaltonError) as e:
        define(svc, tmp_path, cases=[{"id": "a", "checks": [{"kind": "judge", "rubric": "r"}]}, {"id": "a", "checks": [{"kind": "judge", "rubric": "r"}]}])
    assert e.value.key == "se_case_dup"
    ev = define(svc, tmp_path)["evaluation"]
    tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k3", arm="generic", answer="x")
    with pytest.raises(GaltonError) as e:                # a case with results cannot disappear
        define(svc, tmp_path, id=ev["id"], cases=[{"id": "k1", "checks": [{"kind": "judge", "rubric": "r"}]}])
    assert e.value.key == "se_case_has_records"
    with pytest.raises(GaltonError):
        tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="zz", arm="generic")
    with pytest.raises(GaltonError):
        tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k3", arm="generic", deliverable=str(tmp_path / "no-such-folder"))


def test_changing_a_check_rescores_what_was_recorded(svc, tmp_path):
    ev = define(svc, tmp_path)["evaluation"]
    d = folder(tmp_path, "d", {"out.md": "x", "t.json": '{"total": 10}'})
    first = tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k2", arm="current", deliverable=str(d))
    assert first["success"] is True
    cases = [{"id": "k1", "used_for_building": True, "checks": [{"kind": "file_exists", "path": "out.md"}]},
             {"id": "k2", "checks": [{"id": "n", "kind": "numeric", "path": "t.json", "pointer": "total", "expected": 50, "tolerance": 1}]},
             {"id": "k3", "checks": [{"id": "j", "kind": "judge", "rubric": "clear"}]}]
    tool(svc, "pygmalion_suit_eval_define", id=ev["id"], suit="part-budget", cases=cases, candidate={"built_from_cases": ["k1"]})
    assert tool(svc, "pygmalion_suit_eval_get", id=ev["id"])["records"][0]["success"] is False


def test_try_checks_without_saving(svc, tmp_path):
    d = folder(tmp_path, "t", {"a.txt": "hello 42"})
    out = tool(svc, "pygmalion_suit_eval_check", checks=[{"kind": "regex", "path": "a.txt", "pattern": r"\d+"}], deliverable=str(d))
    assert out["success"] is True and tool(svc, "pygmalion_suit_eval_list")["count"] == 0


# ------------------------------------------------------------------ export
def filled(svc, tmp_path):
    ev = define(svc, tmp_path)["evaluation"]
    good = folder(tmp_path, "g", {"out.md": "x", "t.json": '{"total": 10}'})
    for arm in ("generic", "current", "candidate"):
        tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k2", arm=arm, deliverable=str(good))
    return ev["id"]


def test_export_goes_to_darwin_when_it_answers(svc, tmp_path):
    eid = filled(svc, tmp_path)
    sent = []
    svc.suiteval.post_fn = lambda url, payload: (sent.append((url, payload)) or (200, {"imported": True}))
    out = tool(svc, "pygmalion_suit_eval_export", id=eid)
    assert out["exported"]["delivered"] is True and sent[0][0] == "http://127.0.0.1:5233/api/evaluations"
    p = sent[0][1]
    assert p["format"] == "pygmalion-suit-eval/1" and p["suit"] == "part-budget" and p["cases_scored"] == 1
    assert p["excluded"] == [{"case": "k1", "reason": "used_for_building"}] and set(p["aggregate"]) == {"generic", "current", "candidate"}
    assert all(a["cases"] == 1 for a in p["aggregate"].values())
    assert tool(svc, "pygmalion_suit_eval_list")["evaluations"][0]["exported"]["delivered"] is True


def test_export_falls_back_to_a_file_darwin_can_import(svc, tmp_path):
    eid = filled(svc, tmp_path)

    def down(url, payload):
        raise ConnectionError("refused")
    svc.suiteval.post_fn = down
    out = tool(svc, "pygmalion_suit_eval_export", id=eid)
    info = out["exported"]
    assert info["delivered"] is False and "refused" in info["reason"]
    written = json.loads(Path(info["file"]).read_text(encoding="utf-8"))
    assert written["format"] == "pygmalion-suit-eval/1" and written["evaluation_id"] == eid and Path(info["file"]).name == f"part-budget--{eid}.json"
    with pytest.raises(GaltonError) as e:
        tool(svc, "pygmalion_suit_eval_export", id=eid, mode="api")
    assert e.value.key == "se_export_failed"
    svc.suiteval.post_fn = lambda url, payload: (422, {"error": "no"})
    assert tool(svc, "pygmalion_suit_eval_export", id=eid)["exported"]["delivered"] is False        # refused by Darwin: kept as a file
    only_file = tool(svc, "pygmalion_suit_eval_export", id=eid, mode="file")
    assert only_file["exported"]["file"] and only_file["exported"]["delivered"] is False


def test_nothing_scored_is_not_exported(svc, tmp_path):
    ev = define(svc, tmp_path)["evaluation"]
    good = folder(tmp_path, "g2", {"out.md": "x"})
    tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case="k1", arm="candidate", deliverable=str(good))       # only the build case has a result
    with pytest.raises(GaltonError) as e:
        tool(svc, "pygmalion_suit_eval_export", id=ev["id"])
    assert e.value.key == "se_nothing_scored"


def test_delete_needs_confirmation(svc, tmp_path):
    eid = filled(svc, tmp_path)
    with pytest.raises(GaltonError):
        tool(svc, "pygmalion_suit_eval_delete", id=eid)
    assert tool(svc, "pygmalion_suit_eval_delete", id=eid, confirm=True) == {"deleted": eid}
    assert tool(svc, "pygmalion_suit_eval_list")["count"] == 0


def test_rest_routes(client, tmp_path):
    body = {"suit": "part-budget", "name": "R", "cases": [{"id": "a", "checks": [{"kind": "regex", "pattern": "ok"}]}]}
    made = client.post("/api/suit-evals", json=body).json()
    eid = made["evaluation"]["id"]
    r = client.post(f"/api/suit-evals/{eid}/records", json={"case": "a", "arm": "current", "answer": "ok"}).json()
    assert r["success"] is True
    assert client.get(f"/api/suit-evals/{eid}/score").json()["aggregate"]["current"]["success_rate"] == 1.0
    assert client.get("/api/suit-evals").json()["count"] == 1
    assert client.get(f"/api/suit-evals/{eid}").json()["records"][0]["case_id"] == "a"
    assert client.delete(f"/api/suit-evals/{eid}").status_code == 400
    assert client.delete(f"/api/suit-evals/{eid}?confirm=true").json() == {"deleted": eid}
