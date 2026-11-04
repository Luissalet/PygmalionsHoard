"""The decision of a suit evaluation and the suit library: deliverables first, then human measures; tokens never decide; build cases never count."""
from __future__ import annotations

from evaluation_tests.conftest import tool
from galton_hoard.suiteval import stats as S, verdict as V


def rec(case, arm, ok, *, minutes=None, time=None, defects=None, tokens=None):
    return {"case_id": case, "arm": arm, "run": 1, "success": ok, "total_time_s": time, "correction_minutes": minutes, "defects": defects, "tokens": tokens}


def cases(n, build=()):
    return {"cases": [{"id": f"k{i}", "used_for_building": f"k{i}" in build} for i in range(1, n + 1)], "candidate": {"built_from_cases": []}}


def decide(definition, records, a="candidate", b="current"):
    return V.decide(S.aggregate(definition, records), a, b)


def keys(d):
    return [r.key for r in d["reasons"]]


def test_tokens_alone_never_adopt_a_candidate():
    recs = []
    for i in range(1, 5):
        recs += [rec(f"k{i}", "current", True, minutes=5, time=60, defects=0, tokens=10000),
                 rec(f"k{i}", "candidate", True, minutes=5, time=60, defects=0, tokens=2000)]
    d = decide(cases(4), recs)
    assert d["decision"] == "keep" and "sv_tokens_only" in keys(d)
    assert d["tokens"]["effect"] == "better" and d["tokens"]["decides"] is False


def test_more_accepted_deliverables_adopt_even_with_more_tokens():
    recs = []
    for i in range(1, 5):
        recs += [rec(f"k{i}", "current", i == 1, tokens=1000), rec(f"k{i}", "candidate", True, tokens=5000)]
    d = decide(cases(4), recs)
    assert d["decision"] == "adopt" and keys(d)[0] == "sv_success_better" and "sv_tokens_note" in keys(d)
    assert d["success"] == {"better": 3, "worse": 0, "same": 1, "sign_test_p": 0.25} and "sv_weak" in keys(d)


def test_fewer_accepted_deliverables_keep_the_current_suit_whatever_the_tokens():
    recs = []
    for i in range(1, 4):
        recs += [rec(f"k{i}", "current", True, tokens=9000), rec(f"k{i}", "candidate", i == 1, tokens=100)]
    assert decide(cases(3), recs)["decision"] == "keep"


def test_same_deliverables_then_correction_minutes_defects_and_time_decide():
    better = []
    for i in range(1, 4):
        better += [rec(f"k{i}", "current", True, minutes=20, time=100), rec(f"k{i}", "candidate", True, minutes=5, time=100)]
    d = decide(cases(3), better)
    assert d["decision"] == "adopt" and "sv_better_correction_minutes" in keys(d)
    mixed = []
    for i in range(1, 4):
        mixed += [rec(f"k{i}", "current", True, minutes=20, time=100), rec(f"k{i}", "candidate", True, minutes=5, time=300)]
    t = decide(cases(3), mixed)
    assert t["decision"] == "tradeoff" and {"sv_better_correction_minutes", "sv_worse_time_s", "sv_tradeoff"} <= set(keys(t))
    small = []
    for i in range(1, 4):
        small += [rec(f"k{i}", "current", True, minutes=20, defects=1), rec(f"k{i}", "candidate", True, minutes=19.5, defects=1)]
    n = decide(cases(3), small)
    assert n["decision"] == "keep" and "sv_no_difference" in keys(n)       # half a minute on twenty is not a difference a person notices


def test_too_few_cases_is_insufficient_and_build_cases_do_not_count():
    recs = [rec("k1", "current", False), rec("k1", "candidate", True), rec("k2", "current", False), rec("k2", "candidate", True),
            rec("k3", "current", True), rec("k3", "candidate", True)]
    assert decide(cases(3), recs)["decision"] == "adopt"
    # the same results, but the candidate was built on k1 and k2: only k3 is left, and it shows no difference
    d = decide(cases(3, build=("k1", "k2")), recs)
    assert d["decision"] == "insufficient" and d["cases"] == 1 and keys(d) == ["sv_insufficient"]


def test_the_suit_against_plain_tools_is_also_answered():
    recs = []
    for i in range(1, 4):
        recs += [rec(f"k{i}", "generic", False, minutes=40), rec(f"k{i}", "current", True, minutes=10)]
    out = V.verdicts(S.aggregate(cases(3), recs))
    assert out["current_vs_generic"]["decision"] == "adopt" and out["candidate_vs_current"]["decision"] == "insufficient"


# ------------------------------------------------------------------ service, export, library, REST
def _define_and_fill(svc):
    ev = tool(svc, "pygmalion_suit_eval_define", suit="part-budget", name="Budget",
              cases=[{"id": f"k{i}", "used_for_building": i == 1, "checks": [{"kind": "regex", "target": "answer", "pattern": "^ok$"}]} for i in range(1, 5)],
              candidate={"description": "better recipe", "built_from_cases": ["k1"]})["evaluation"]
    for i in range(1, 5):
        tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case=f"k{i}", arm="current", answer="ok", correction_minutes=15, tokens=900)
        tool(svc, "pygmalion_suit_eval_record", id=ev["id"], case=f"k{i}", arm="candidate", answer="ok", correction_minutes=4, tokens=1500)
    return ev


def test_score_list_and_export_carry_the_decision(svc):
    ev = _define_and_fill(svc)
    sc = tool(svc, "pygmalion_suit_eval_score", id=ev["id"])
    v = sc["verdict"]["candidate_vs_current"]
    assert v["decision"] == "adopt" and v["cases"] == 3 and "sv_better_correction_minutes" in [r.key for r in v["reasons"]]
    assert "sv_tokens_note" in [r.key for r in v["reasons"]]                     # more tokens, reported, not decisive
    listed = tool(svc, "pygmalion_suit_eval_list")
    assert listed["evaluations"][0]["decisions"] == {"candidate_vs_current": "adopt", "current_vs_generic": "insufficient"}
    payload = svc.suiteval.payload(ev["id"])
    assert payload["verdict"]["candidate_vs_current"]["decision"] == "adopt"
    assert all(set(r) == {"key", "text"} for r in payload["verdict"]["candidate_vs_current"]["reasons"])


def test_library_joins_darwin_suits_with_local_evaluations(svc):
    ev = _define_and_fill(svc)
    tool(svc, "pygmalion_suit_eval_define", suit="only-here", cases=[{"id": "a", "checks": [{"kind": "regex", "target": "answer", "pattern": "x"}]}])
    seen = []
    svc.suiteval.get_fn = lambda url: (seen.append(url) or (200, {"suits": [
        {"slug": "part-budget", "name": "Part budget", "objective": "cheapest valid combination", "status": "proposed", "missing": ["optimality"]},
        {"slug": "repository-intake", "name": "Repository intake", "status": "partial"}]}))
    lib = tool(svc, "pygmalion_suit_library")
    assert seen == ["http://127.0.0.1:5233/api/suits"] and lib["darwin"]["reachable"] is True
    by = {s["slug"]: s for s in lib["suits"]}
    assert by["part-budget"]["in_darwin"] and by["part-budget"]["latest"]["id"] == ev["id"]
    assert by["part-budget"]["latest"]["decisions"]["candidate_vs_current"] == "adopt" and by["part-budget"]["missing"] == ["optimality"]
    assert by["repository-intake"]["evaluations"] == [] and by["only-here"]["in_darwin"] is False
    assert lib["count"] == 3 and lib["evaluated_here"] == 2


def test_library_without_darwin_lists_what_is_known_here(svc):
    _define_and_fill(svc)

    def down(url):
        raise ConnectionError("refused")
    svc.suiteval.get_fn = down
    lib = tool(svc, "pygmalion_suit_library")
    assert lib["darwin"]["reachable"] is False and "refused" in lib["darwin"]["reason"]
    assert [s["slug"] for s in lib["suits"]] == ["part-budget"] and lib["suits"][0]["in_darwin"] is False


def test_rest_library_and_score_verdict(client):
    client.svc.suiteval.get_fn = lambda url: (200, {"suits": [{"slug": "part-budget", "name": "Part budget"}]})
    _define_and_fill(client.svc)
    lib = client.get("/api/suit-evals/library").json()
    assert lib["suits"][0]["slug"] == "part-budget" and lib["suits"][0]["latest"]["decisions"]["candidate_vs_current"] == "adopt"
    eid = lib["suits"][0]["latest"]["id"]
    v = client.get(f"/api/suit-evals/{eid}/score").json()["verdict"]["candidate_vs_current"]
    assert v["decision"] == "adopt" and v["reasons"][0]["key"] == "sv_success_same" and v["reasons"][0]["params"]["a"]["key"] == "arm_candidate"
