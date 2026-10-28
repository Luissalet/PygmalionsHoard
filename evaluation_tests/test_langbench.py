"""Language bench: the same tool tasks in Spanish and English, localised by type, with the language metrics and the language axis of the matrix."""

from __future__ import annotations

import json
import re

import pytest

from evaluation_tests.helpers import add_gguf
from galton_hoard.backends import ChatRequest
from galton_hoard.errors import GaltonError
from galton_hoard.fakes import _speak, agent_responder, completion_from, harness_dialect, harness_task
from galton_hoard.harnessbench import BASES, build_matrix, cell_stats, parse_harness_id, run_trial
from galton_hoard.harnessbench.scoring import language_report
from galton_hoard.harnessbench.store import column_id, split_column
from galton_hoard.harnessbench.tasks import new_world
from galton_hoard.langbench import I18N_TASK_IDS, KINDS, LANGS, SPECS_BY_ID, TYPOLOGY, arg_matches_data, consistency, localize, public, select
from galton_hoard.langbench import family, shop
from galton_hoard.langbench.bm25 import Bm25
from galton_hoard.langbench.canon import BRANDS, CANON, PEOPLE, aliases, every_label, fold, label
from galton_hoard.langbench.tasks import mentions, resolve

VARIANTS = [f"{b}{s}" for b in BASES for s in ("", ".long", ".trim", ".long.trim")]


def _chat(responder):
    def chat(messages, tools):
        req = ChatRequest(messages=messages, tools=tools or [])
        return completion_from(responder(req), req, tps=40.0, ttft_ms=100.0)
    return chat


def _rows(responder, harness, lang, ids=None, **kw):
    h = parse_harness_id(harness)
    return [run_trial(localize(i, lang), h, _chat(responder), **kw).as_dict() for i in (ids or I18N_TASK_IDS)]


# ---------------------------------------------------------------- the tasks and the canonical map
def test_there_are_between_twenty_and_thirty_tasks_with_every_kind_in_both_domains():
    assert 20 <= len(I18N_TASK_IDS) <= 30 and len(set(I18N_TASK_IDS)) == len(I18N_TASK_IDS)
    assert {SPECS_BY_ID[i]["kind"] for i in I18N_TASK_IDS} == set(KINDS)
    assert {SPECS_BY_ID[i]["domain"] for i in I18N_TASK_IDS} == {"shop", "family"}
    assert select(kinds=["notool"]) == [i for i in I18N_TASK_IDS if SPECS_BY_ID[i]["kind"] == "notool"]
    assert LANGS == ("en", "es") and set(TYPOLOGY) == {"preserve", "translate_free", "translate_canonical"}


def test_the_canonical_map_has_both_languages_for_every_entity():
    for kind, table in CANON.items():
        for key, entry in table.items():
            for lang in LANGS:
                assert str(entry[lang]).strip(), (kind, key, lang)
            assert label(kind, key, "es") and aliases(kind, key, "en")[0] == label(kind, key, "en")
    assert set(CANON["product"]) == {p["id"] for p in shop.PRODUCTS} and len(shop.PRODUCTS) == 36
    assert {r["concept"] for r in family.EXPENSES} <= set(CANON["concept"]) and {r["category"] for r in family.EXPENSES} <= set(CANON["expense_category"])
    assert {r["title"] for r in family.APPOINTMENTS} <= set(CANON["appointment"]) and {r["item"] for r in family.SHIPMENTS} <= set(CANON["parcel"])
    assert {r["item"] for r in family.PANTRY} <= set(CANON["pantry"]) and {p["category"] for p in shop.PRODUCTS} <= set(CANON["category"]) and {p["color"] for p in shop.PRODUCTS} <= set(CANON["color"])
    assert {p["brand"] for p in shop.PRODUCTS} <= set(BRANDS) and {r["payer"] for r in family.EXPENSES} <= set(PEOPLE)


def _calls_shape(task):
    return [(n, tuple(sorted(a))) for n, a in task["oracle"]]


@pytest.mark.parametrize("task_id", I18N_TASK_IDS)
def test_the_two_versions_of_a_task_preserve_ids_tools_arguments_and_enums(task_id):
    en, es = localize(task_id, "en"), localize(task_id, "es")
    assert en["id"] == es["id"] == task_id and en["kind"] == es["kind"] and en["tools"] == es["tools"]
    assert en["prompt"] != es["prompt"] and "{" not in en["prompt"] + es["prompt"]
    assert _calls_shape(en) == _calls_shape(es)
    for (_, ea), (_, sa) in zip(en["oracle"], es["oracle"]):
        for key in ("product_id", "record_id", "category", "collection", "month", "sort_by", "max_price", "quantity"):
            if key in ea or key in sa:
                assert ea[key] == sa[key], key                                       # Preserve: identical in both versions
    for name in en["tools"]:
        pe, ps = en["specs"][name].parameters, es["specs"][name].parameters
        assert list(pe["properties"]) == list(ps["properties"]) and pe["required"] == ps["required"]
        for key in pe["properties"]:
            assert pe["properties"][key].get("enum") == ps["properties"][key].get("enum") and pe["properties"][key]["type"] == ps["properties"][key]["type"]
        assert en["specs"][name].description != es["specs"][name].description


def test_the_prompts_are_in_their_language_and_name_entities_from_the_canonical_map():
    for i in I18N_TASK_IDS:
        s = SPECS_BY_ID[i]
        for lang in LANGS:
            text = resolve(s["prompt"][lang], lang)
            for kind, key in re.findall(r"\{(\w+):([^}]+)\}", s["prompt"][lang]):
                assert label(kind, key, lang) in text
        es_words = consistency(resolve(s["prompt"]["es"], "es"), "es")
        en_words = consistency(resolve(s["prompt"]["en"], "en"), "en")
        for res in (es_words, en_words):
            assert res["score"] is None or res["score"] >= 0.8, (i, res)


@pytest.mark.parametrize("lang", LANGS)
@pytest.mark.parametrize("harness", VARIANTS)
def test_the_oracle_passes_every_task_in_every_harness_without_extras_or_wrong_language(harness, lang):
    rows = _rows(agent_responder(), harness, lang)
    bad = [(r["task_id"], r["outcome"], r["reason"], r["detail"]) for r in rows if r["outcome"] != "pass"]
    assert not bad, bad
    assert all(r["extra_calls"] == 0 and r["format_failures"] == 0 and r["lang"] == lang for r in rows)
    assert all(r["empty_searches"] == 0 and r["arg_mismatch"] == 0 for r in rows), [r["task_id"] for r in rows if r["empty_searches"] or r["arg_mismatch"]]
    judged = [r["lang_consistency"] for r in rows if r["lang_consistency"] is not None]
    assert len(judged) >= len(rows) - 4 and min(judged) >= 0.85, sorted((r["lang_consistency"], r["task_id"]) for r in rows if r["lang_consistency"] is not None)[:3]


# ---------------------------------------------------------------- the data and the search
def test_bm25_finds_by_stem_and_ignores_function_words():
    idx = Bm25(["Zapatillas de running Nimbus", "Botas de cuero", "Mochila escolar"])
    assert idx.rank("zapatilla") == [0] and idx.rank("la mochila de") == [2] and idx.rank("sandalias") == []


def test_a_search_in_the_other_language_than_the_catalogue_comes_back_empty():
    es, en = localize("shop-price-lookup", "es"), localize("shop-price-lookup", "en")
    w_es, w_en = new_world(es), new_world(en)
    assert w_es.run("search_products", {"query": "zapatillas running"}).empty is False
    r = w_es.run("search_products", {"query": "sneakers boots hiking"})
    assert r.ok and r.empty and "No se encontraron" in r.text
    r = w_en.run("search_products", {"query": "zapatillas botas"})
    assert r.ok and r.empty and "No products found" in r.text
    assert w_en.run("search_products", {"query": "boots"}).text.count("P-") == 1 and not w_en.run("search_products", {"query": ""}).empty


def test_the_shop_tools_filter_sort_cart_and_refuse_what_is_not_possible():
    w = new_world(localize("shop-cheapest-bag", "es"))
    out = w.run("search_products", {"query": "", "category": "bags", "sort_by": "price_asc"}).text.splitlines()
    assert out[1].startswith("P-3005") and "19.90 EUR" in out[1] and len(out) == 7
    assert w.run("search_products", {"query": "", "max_price": 30}).text.count("P-") == 8
    assert not w.run("add_to_cart", {"product_id": "P-2004", "quantity": 1}).ok and "agotado" in w.run("add_to_cart", {"product_id": "P-2004", "quantity": 1}).text
    assert "Error" in w.run("add_to_cart", {"product_id": "P-9999", "quantity": 1}).text
    assert not w.run("add_to_cart", {"product_id": "P-1003", "quantity": 7}).ok
    assert w.run("add_to_cart", {"product_id": "P-1003", "quantity": 2}).text.endswith("258.00 EUR")
    assert "x2 = 258.00 EUR" in w.run("view_cart", {}).text and w.cart == {"P-1003": 2}
    assert new_world(localize("shop-cheapest-bag", "en")).run("get_product", {"product_id": "P-3004"}).text.splitlines()[2] == "Brand: Terrabruno"
    from galton_hoard.harnessbench.world import World
    spec = localize("shop-cheapest-bag", "en")["specs"]["search_products"]
    assert World.check_args(spec, {"query": "", "category": "gadgets"})[0] is None and "one of" in World.check_args(spec, {"query": "", "sort_by": "cheapest"})[1]


def test_the_household_tools_search_total_and_keep_notes():
    w = new_world(localize("fam-food-sept", "es"))
    assert w.run("family_total", {"month": "2026-09", "category": "food"}).text == "Total alimentación 2026-09: 134.30 EUR (2 gasto(s))"
    assert new_world(localize("fam-food-sept", "en")).run("family_total", {"month": "2026-10"}).text.startswith("Total 2026-10: 376.35 EUR")
    assert not w.run("family_total", {"month": "septiembre"}).ok
    assert w.run("family_search", {"collection": "appointments", "query": "dentista"}).text.count("C-") == 1
    r = w.run("family_search", {"collection": "appointments", "query": "dentist"})
    assert r.empty and "Ningún registro" in r.text
    assert w.run("family_search", {"collection": "expenses", "query": "", "month": "2026-09"}).text.startswith("8 registro(s)")
    assert w.run("family_search", {"collection": "pantry", "query": ""}).text.count("D-4") == 6
    assert w.run("family_note", {"record_id": "C-201", "note": "Llevar la cartilla"}).ok and "Nota: Llevar la cartilla" in w.run("family_get", {"record_id": "C-201"}).text
    assert not w.run("family_note", {"record_id": "C-999", "note": "x"}).ok and not w.run("family_note", {"record_id": "C-201", "note": "  "}).ok


# ---------------------------------------------------------------- the checks read the canonical map
def test_mentions_accept_the_localised_name_an_alias_or_the_model_name_in_the_right_language():
    assert mentions("Cheapest: the Vuelo waist bag", "product", "P-3005", "en") and mentions("la riñonera vuelo", "product", "P-3005", "es")
    assert mentions("the Nimbus 3 ones", "product", "P-1001", "en") and not mentions("Nimbus Trail trail running shoes", "product", "P-1001", "en")
    assert mentions("the school backpack", "parcel", "school_backpack", "en") and mentions("las mochilas escolares", "parcel", "school_backpack", "es")
    assert mentions("Yoghurt", "pantry", "yogurt", "en") and not mentions("rice", "pantry", "yogurt", "en")


def test_verifiers_reject_the_wrong_answers_of_each_kind():
    def check(task_id, lang, final, mutate=None):
        t = localize(task_id, lang)
        w = new_world(t)
        if mutate:
            mutate(w)
        return t["verify"](final, w)[0]
    assert check("shop-price-lookup", "es", "Cuestan 89,90 EUR.") and not check("shop-price-lookup", "es", "Cuestan 64.50 EUR.")
    assert check("shop-black-audio-budget", "en", "Eco wireless earbuds.") and not check("shop-black-audio-budget", "en", "Eco wireless earbuds and Estudio noise-cancelling headphones.")
    assert not check("shop-out-of-stock-answer", "es", "Sí, está disponible.") and check("shop-out-of-stock-answer", "es", "No, está agotado.")
    assert not check("shop-cart-boots-slippers", "en", "The total is 282.90 EUR.")                               # nothing was added
    assert check("shop-cart-boots-slippers", "en", "The total is 282.90 EUR.", lambda w: w.cart.update({"P-1003": 2, "P-1006": 1}))
    assert not check("fam-note-dentist", "es", "Hecho.") and check("fam-note-dentist", "es", "Hecho.", lambda w: w.notes.update({"C-201": "Llevar la cartilla de vacunación"}))
    assert check("fam-note-dentist", "en", "Done.", lambda w: w.notes.update({"C-201": "bring the vaccination card"}))
    assert not check("fam-expiring", "en", "yogurt, milk, eggs and rice") and check("fam-expiring", "en", "yogurt, milk and eggs")
    assert not check("fam-explain-budget", "es", "Because it is useful.") and check("fam-explain-budget", "es", "Un presupuesto ayuda a ahorrar.")


# ---------------------------------------------------------------- the language judge
def test_the_judge_looks_only_at_the_prose():
    assert consistency("El producto P-1003 cuesta 129.00 EUR y está disponible en la tienda.", "es")["score"] == 1.0
    assert consistency("The product costs 129.00 EUR and it is available in the shop.", "es")["score"] == 0.0
    mixed = consistency("The Alpina boots are available y cuestan mucho menos que las otras botas de la tienda.", "es")
    assert 0.3 < mixed["score"] < 1.0
    assert consistency("Las Zapatillas de running Nimbus 3 de Solana cuestan 89.90 EUR.", "es")["score"] == 1.0
    assert consistency("The Botas de montaña de cuero Alpina by Terrabruno cost 129.00 EUR, which is the price.", "en")["score"] == 1.0, "names of the other language are not prose"
    assert consistency("P-1003 129.00 EUR", "es")["score"] is None and consistency("", "en")["judged"] is False
    assert "Terrabruno" in every_label() and "Zapatillas de running Nimbus 3" in every_label()


def test_a_tool_argument_is_judged_against_the_language_of_the_data():
    assert arg_matches_data("zapatillas de running", "es") is True and arg_matches_data("running sneakers boots", "es") is False
    assert arg_matches_data("rechargeable flashlight", "en") is True and arg_matches_data("linterna recargable", "en") is False
    assert arg_matches_data("P-1003", "es") is None and arg_matches_data("Nimbus 3", "en") is None and arg_matches_data("", "es") is None


# ---------------------------------------------------------------- a model that translates its searches
def translating(responder_lang="en"):
    """An agent that follows the oracle of the other language: it searches in English a Spanish catalogue."""
    def respond(req):
        task = harness_task(req)
        if task is None:
            return ""
        other = localize(task["id"], "en" if task["lang"] == "es" else "es")
        dialect = harness_dialect(req)
        done = sum(1 for m in req.messages if m.get("role") == "assistant")
        steps = other["oracle"]
        if done < len(steps):
            return _speak(dialect, *steps[done])
        return other["final"] if dialect != "react" else f"Thought: done\nFinal Answer: {other['final']}"
    return respond


def test_a_translated_search_is_counted_as_empty_and_as_an_argument_in_the_wrong_language():
    ids = ["shop-flashlight", "fam-dentist", "fam-guitar", "fam-cinema", "shop-price-lookup"]
    rows = _rows(translating(), "native_tools", "es", ids)
    by = {r["task_id"]: r for r in rows}
    for i in ("shop-flashlight", "fam-dentist", "fam-guitar", "fam-cinema"):
        assert by[i]["empty_searches"] >= 1 and by[i]["arg_mismatch"] >= 1 and by[i]["search_calls"] >= 1, i
        assert by[i]["lang_consistency"] is None or by[i]["lang_consistency"] < 0.3, i
    assert by["shop-flashlight"]["outcome"] == "fail_answer"
    assert sum(1 for r in rows if r["lang_consistency"] is not None) >= 3
    assert by["shop-price-lookup"]["arg_mismatch"] == 0 or by["shop-price-lookup"]["empty_searches"] == 0
    stats = cell_stats(rows, 1)
    assert stats["lang"] == "es" and stats["empty_search_rate"] >= 0.6 and stats["arg_mismatch_rate"] > 0.5 and stats["lang_consistency"] < 0.5 and stats["inconsistent_rate"] > 0.5


# ---------------------------------------------------------------- the language axis of the matrix
def _synthetic(model, harness, lang, passes, calls, cons=1.0, empties=0, searches=2, mism=0, args=2):
    rows = []
    for i, ok in enumerate(passes):
        rows.append({"contestant_id": model, "harness": harness, "lang": lang, "task_id": f"t{i}", "sample": 0, "outcome": "pass" if ok else "fail_answer", "passed": ok, "format_failures": 0,
                     "extra_calls": 0, "calls": calls, "turns": 2, "tokens_in": 10, "tokens_out": 5, "latency_ms": 1.0, "reason": "", "detail": "", "lang_consistency": cons,
                     "search_calls": searches, "empty_searches": empties, "arg_calls": args, "arg_mismatch": mism})
    return rows


def test_the_language_report_compares_each_language_with_english():
    rows = (_synthetic("m", "native_tools", "en", [1, 1, 1, 1], 2) + _synthetic("m", "native_tools", "es", [1, 0, 0, 1], 3, cons=0.5, empties=1, mism=1)
            + _synthetic("m", "json_text", "en", [1, 1, 0, 0], 2) + _synthetic("m", "json_text", "es", [1, 1, 0, 0], 2))
    cols = [column_id(h, lang) for h in ("native_tools", "json_text") for lang in ("en", "es")]
    matrix = build_matrix(rows, ["m"], cols, 1)
    assert matrix["cells"]["m"]["native_tools@es"]["lang"] == "es" and matrix["cells"]["m"]["native_tools@es"]["empty_search_rate"] == 0.5
    rep = language_report(rows, matrix, ["m"], ["native_tools", "json_text"], ["en", "es"], {"m": "M"})
    nt = rep["pairs"]["m"]["native_tools"]["es"]
    assert nt["pass_delta_points"] == -50.0 and nt["calls_vs_base"] == 1.5 and nt["lang_consistency"] == 0.5 and nt["arg_mismatch_rate"] == 0.5
    assert rep["pairs"]["m"]["json_text"]["es"]["pass_delta_points"] == 0.0 and rep["pairs"]["m"]["json_text"]["es"]["calls_vs_base"] == 1.0
    assert rep["base"] == "en" and rep["effects"]["language_points"] == 25.0 and rep["effects"]["verdict"] in ("language_larger", "harness_larger", "similar", "model_larger", "not_comparable")
    assert "1.25 times the tool calls" in rep["summary"]["en"] and "1.25 veces" in rep["summary"]["es"]
    assert split_column("json_text.long@es") == ("json_text.long", "es") and column_id("native_tools") == "native_tools"


# ---------------------------------------------------------------- through the services
def _models(svc, **specs):
    return [add_gguf(svc, name, responder=r) for name, r in specs.items()]


def _start(svc, models, **kw):
    return svc.start_matrix(models=[m["id"] for m in models], **kw)["matrix"]["id"]


def test_a_language_matrix_runs_through_the_scheduler_path_and_reports_the_language_axis(svc):
    good, trans = _models(svc, bueno=agent_responder(), traductor=translating())
    mid = _start(svc, [good, trans], harnesses=["native_tools", "json_text"], langs=["es", "en"], kinds=["lookup"])
    m = svc.matrix_store.matrix(mid)
    n = len(select(kinds=["lookup"]))
    assert m["state"] == "done" and m["langs"] == ["es", "en"] and m["done"] == m["total"] == 2 * 2 * 2 * n
    res = svc.matrices.result(mid)
    assert [c["id"] for c in res["columns"]] == ["native_tools@es", "native_tools@en", "json_text@es", "json_text@en"]
    assert res["cells"][good["id"]]["native_tools@es"]["pass_at_1"] == 1.0 and res["cells"][good["id"]]["native_tools@es"]["empty_search_rate"] == 0.0
    t_es, t_en = res["cells"][trans["id"]]["native_tools@es"], res["cells"][trans["id"]]["native_tools@en"]
    assert t_es["empty_search_rate"] > 0.2 and t_es["arg_mismatch_rate"] > 0.5 and t_es["lang_consistency"] < 0.3
    assert t_en["empty_search_rate"] > 0.2 and t_en["lang_consistency"] < 0.3, "the translating agent speaks the other language in both runs"
    assert res["cells"][good["id"]]["native_tools@es"]["lang_consistency"] >= 0.9
    lang = res["language"]
    assert lang["pairs"][good["id"]]["native_tools"]["es"]["pass_delta_points"] == 0.0 and lang["by_lang"]["es"]["empty_search_rate"] > 0
    assert res["conclusion"]["language"]["language_points"] is not None and "language" in res["conclusion"]["summary"]["en"].lower()
    assert "| bueno |" in res["markdown"] and "native_tools@es" in res["markdown"]
    # the cell and one task of it, by column id
    cell = svc.matrices.cell(mid, "traductor", "native_tools@es")
    assert cell["lang"] == "es" and cell["stats"]["lang"] == "es" and all("empty_searches" in r for r in cell["tasks"])
    one = svc.matrices.cell(mid, "traductor", "native_tools@es", "fam-dentist")
    assert one["trials"][0]["lang"] == "es" and one["trials"][0]["trace"] and one["task"]["id"] == "fam-dentist"
    with pytest.raises(GaltonError) as e:
        svc.matrices.cell(mid, "traductor", "native_tools@fr")
    assert e.value.key == "hb_unknown_lang"
    assert svc.matrix_store.matrix(mid)["summary"]["summary"]


def test_language_requests_are_planned_and_refused_with_a_catalogue_message(svc):
    (a,) = _models(svc, uno=agent_responder())
    plan = svc.matrices.plan(models=[a["id"]], harnesses=["native_tools"], langs=["es", "en"], k=2)
    assert plan["langs"] == ["es", "en"] and plan["tasks"] == len(I18N_TASK_IDS) and plan["trials"] == 2 * 2 * len(I18N_TASK_IDS)
    assert svc.matrix_store.matrices() == []
    for kw, key in (({"langs": ["fr"]}, "hb_unknown_lang"), ({"langs": ["es"], "tasks": ["calc-sum-product"]}, "hb_no_task_lang"), ({"tasks": ["shop-brand"]}, "hb_lang_needed"),
                    ({"langs": ["es"], "kinds": ["calc"]}, "hb_no_tasks")):
        with pytest.raises(GaltonError) as e:
            svc.start_matrix(models=[a["id"]], **kw)
        assert e.value.key == key, kw
    assert svc.matrices.catalog()["languages"]["langs"] == ["en", "es"] and len(svc.matrices.catalog()["languages"]["tasks"]) == len(I18N_TASK_IDS)


def test_the_base_matrix_is_unchanged_by_the_language_axis(svc):
    (a,) = _models(svc, uno=agent_responder())
    mid = _start(svc, [a], harnesses=["native_tools"], kinds=["calc"])
    res = svc.matrices.result(mid)
    assert [c["id"] for c in res["columns"]] == ["native_tools"] and "language" not in res and "lang" not in res["cells"][a["id"]]["native_tools"]
    assert svc.matrix_store.matrix(mid)["langs"] == []


def test_the_migration_adds_the_language_columns(svc):
    cols = {r["name"] for r in svc.db.query("PRAGMA table_info(hb_trials)")}
    assert {"lang", "search_calls", "empty_searches", "arg_calls", "arg_mismatch", "lang_consistency"} <= cols
    assert "langs" in {r["name"] for r in svc.db.query("PRAGMA table_info(hb_matrices)")}


def test_the_mcp_tools_list_the_tasks_and_judge_text(svc):
    from galton_hoard.agent_tools import LangbenchJudgeArgs, LangbenchTaskArgs, run_langbench_judge, run_langbench_task
    listing = run_langbench_task(svc, LangbenchTaskArgs())
    assert len(listing["tasks"]) == len(I18N_TASK_IDS) and "canonical" not in listing
    one = run_langbench_task(svc, LangbenchTaskArgs(task="fam-dentist", canonical=True))
    assert set(one["versions"]) == {"en", "es"} and one["versions"]["es"]["oracle"][0]["args"]["query"] == "dentista" and one["canonical"]["product"]["P-1001"]["es"].startswith("Zapatillas")
    with pytest.raises(GaltonError):
        run_langbench_task(svc, LangbenchTaskArgs(task="nope"))
    assert run_langbench_judge(svc, LangbenchJudgeArgs(text="Cuesta mucho menos que la otra.", lang="es"))["score"] == 1.0
    assert run_langbench_judge(svc, LangbenchJudgeArgs(text="running sneakers", lang="es", argument=True))["in_language_of_data"] is False
