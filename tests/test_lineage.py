"""Artifacts and their family: resolving, walking, the graph, the portable recipe, editing and the safe delete."""

import threading
from pathlib import Path

import pytest

from conftest import call
from models import make_model
from pygmalion_hoard.errors import PygmalionError
from pygmalion_hoard.lineage import is_quantized, portable, verdict_of


@pytest.fixture
def family(ctx, base_model, dataset):
    """base -> adapter -> merged -> gguf -> quantized gguf (with a matrix) -> ollama tag."""
    s = ctx.svc.store
    out = ctx.work / "outputs"
    out.mkdir(exist_ok=True)

    def folder(name):
        p = out / name
        p.mkdir()
        (p / "f.bin").write_bytes(b"x" * 100)
        return p

    adapter = s.create_artifact("adapter", "ad", path=str(folder("ad")), size=100, parents=[base_model["id"]], dataset_version=dataset["id"],
                                recipe={"job_kind": "train", "params": {"rank": 16, "base": str(base_model["path"])}}, metrics={"training": {"final_loss": 1.25}})
    merged = s.create_artifact("merged", "mg", path=str(folder("mg")), size=100, parents=[base_model["id"], adapter["id"]], dataset_version=dataset["id"])
    gguf = s.create_artifact("gguf", "mg-f16", path=str(folder("g")) + "/f.bin", size=100, parents=[merged["id"]])
    matrix = s.create_artifact("imatrix", "mg-imx", path=str(out / "ad" / "f.bin"), size=10, parents=[gguf["id"]])
    quant = s.create_artifact("gguf", "mg-q4", path=str(out / "mg" / "f.bin"), size=30, parents=[gguf["id"], matrix["id"]], metrics={"quant": "Q4_K_M"},
                              recipe={"params": {"qtype": "Q4_K_M"}})
    tag = s.create_artifact("ollama", "pyg-mg:latest", path="pyg-mg:latest", parents=[quant["id"]])
    return dict(base=base_model, adapter=adapter, merged=merged, gguf=gguf, matrix=matrix, quant=quant, tag=tag)


def ids(items):
    return [a["id"] for a in items]


def test_ensure_base_is_idempotent_and_needs_a_model_folder(ctx, tmp_path):
    folder = make_model(tmp_path / "m", seed=1)
    a = ctx.svc.lineage.ensure_base(folder, "acme/m")
    assert ctx.svc.lineage.ensure_base(folder)["id"] == a["id"] and a["kind"] == "base" and a["size"] > 0
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.lineage.ensure_base(tmp_path / "nothing")
    assert exc.value.code == "invalid"


def test_resolve_by_id_name_path_and_kind(ctx, family, tmp_path):
    lin = ctx.svc.lineage
    assert lin.resolve(family["adapter"]["id"])["name"] == "ad" and lin.resolve("ad")["id"] == family["adapter"]["id"]
    assert lin.resolve(family["base"]["path"])["id"] == family["base"]["id"]
    folder = make_model(tmp_path / "loose", seed=5)
    assert lin.resolve(str(folder))["kind"] == "base", "a model folder becomes a base on the fly"
    with pytest.raises(PygmalionError) as exc:
        lin.resolve("ad", ("gguf",))
    assert exc.value.code == "invalid" and "adapter" in exc.value.message
    for bad in ("", "  ", "no-such"):
        with pytest.raises(PygmalionError):
            lin.resolve(bad)


def test_ancestors_are_nearest_first_and_unique(ctx, family):
    names = [a["name"] for a in ctx.svc.lineage.ancestors(family["tag"]["id"])]
    assert names[0] == "mg-q4" and len(names) == len(set(names)) == 6 and names[-1] in ("acme/tiny-base", "ad")


def test_descendants_cover_the_whole_branch(ctx, family):
    desc = ids(ctx.svc.lineage.descendants(family["adapter"]["id"]))
    assert set(desc) == {family[k]["id"] for k in ("merged", "gguf", "matrix", "quant", "tag")}
    assert ctx.svc.lineage.descendants(family["tag"]["id"]) == []


def test_nearest_comparable_is_the_closest_gguf_or_tag(ctx, family):
    assert ctx.svc.lineage.nearest_comparable(family["tag"]["id"])["id"] == family["quant"]["id"]
    assert ctx.svc.lineage.nearest_comparable(family["quant"]["id"])["id"] == family["gguf"]["id"]
    assert ctx.svc.lineage.nearest_comparable(family["gguf"]["id"]) is None


def test_recipe_lists_parents_before_children_with_portable_paths(ctx, family, dataset):
    recipe = ctx.svc.lineage.recipe(family["tag"]["id"])
    assert recipe["format"] == "pygmalion-recipe/1"
    order = [s["artifact"] for s in recipe["steps"]]
    for art in ctx.svc.store.artifacts():
        for parent in art["parents"]:
            assert order.index(parent) < order.index(art["id"]), (parent, art["id"])
    assert recipe["steps"][0]["kind"] == "base" and recipe["steps"][-1]["kind"] == "ollama"
    step = next(s for s in recipe["steps"] if s["kind"] == "adapter")
    assert step["dataset"]["name"] == "Casa de la abuela" and step["dataset"]["sha256"] == dataset["sha256"]
    assert step["recipe"]["params"]["base"] == Path(family["base"]["path"]).name, "absolute paths are reduced to their last component"
    assert step["metrics"] == {"training": {"final_loss": 1.25}}
    assert "/" not in str(recipe["steps"][1]["recipe"].get("params", {}).get("base", ""))


def test_portable_reduces_only_absolute_paths():
    assert portable({"a": "/home/user/models/x", "b": ["C:\\Users\\me\\y", "plain", "a/b"], "c": 3}) == {"a": "x", "b": ["y", "plain", "a/b"], "c": 3}
    assert portable("/") == "/" and portable("/x") == "/x"


def test_quantization_and_verdict_helpers(ctx, family):
    assert is_quantized(family["quant"]) and not is_quantized(family["gguf"]) and not is_quantized(family["merged"])
    assert verdict_of(family["gguf"]) is None
    ctx.svc.lineage.add_metrics(family["gguf"]["id"], "galton", {"verdict": "better"})
    assert verdict_of(ctx.svc.store.artifact(family["gguf"]["id"])) == "better"
    assert ctx.svc.lineage.card(ctx.svc.store.artifact(family["gguf"]["id"]))["verdict"] == "better"


def test_card_reports_losses_ppl_and_missing_files(ctx, family):
    card = ctx.svc.lineage.card(family["adapter"])
    assert card["final_loss"] == 1.25 and card["exists"] is True
    gone = ctx.svc.store.create_artifact("gguf", "gone", path="/nowhere/x.gguf")
    assert ctx.svc.lineage.card(gone)["exists"] is False
    assert ctx.svc.lineage.card(family["tag"])["exists"] is None


def test_detail_has_dataset_family_and_comparable_parent(ctx, family):
    d = ctx.svc.lineage.detail(family["merged"]["id"])
    assert d["dataset"]["dataset"] == "Casa de la abuela" and d["dataset"]["records"] == 60
    assert {a["id"] for a in d["ancestors"]} == {family["base"]["id"], family["adapter"]["id"]}
    assert d["comparable_parent"] is None and len(d["descendants"]) == 4
    assert ctx.svc.lineage.detail(family["quant"]["id"])["comparable_parent"]["id"] == family["gguf"]["id"]


def test_detail_survives_a_deleted_dataset(ctx, family, dataset):
    ctx.svc.datasets.delete(dataset["dataset_id"])
    assert ctx.svc.lineage.detail(family["adapter"]["id"])["dataset"]["missing"] is True


def test_graph_places_nodes_in_columns_and_connects_them(ctx, family):
    g = ctx.svc.lineage.graph()
    col = {n["kind"]: n["col"] for n in g["nodes"]}
    assert col["base"] == 0 and col["adapter"] == 1 and col["merged"] == 2 and col["gguf"] == 3 and col["ollama"] == 4
    assert len(g["edges"]) == 8 and {"from": family["adapter"]["id"], "to": family["merged"]["id"]} in g["edges"]
    assert len({(n["col"], n["row"]) for n in g["nodes"]}) == len(g["nodes"]) and g["width"] > 0 and g["height"] > 0
    assert [c["label"] for c in g["columns"]][0] == "Base"


def test_graph_can_focus_on_one_family_and_on_kinds(ctx, family, tmp_path):
    other = ctx.svc.lineage.ensure_base(make_model(tmp_path / "solo", seed=9), "solo")
    assert other["id"] not in ids(ctx.svc.lineage.graph(family["quant"]["id"])["nodes"])
    assert len(ctx.svc.lineage.graph(family["quant"]["id"])["nodes"]) == 7
    only = ctx.svc.lineage.graph(kinds=["gguf"])
    assert {n["kind"] for n in only["nodes"]} == {"gguf"} and only["edges"] == [{"from": family["gguf"]["id"], "to": family["quant"]["id"]}]


def test_update_renames_notes_and_pins(ctx, family):
    out = ctx.svc.lineage.update(family["merged"]["id"], name="  Mi fusión ", notes="buena", pinned=True)
    assert out["name"] == "Mi fusión" and out["notes"] == "buena" and out["pinned"] == 1
    with pytest.raises(PygmalionError):
        ctx.svc.lineage.update(family["merged"]["id"], name="   ")
    pinned = call(ctx.svc, "artifacts_list", pinned=True)
    assert [a["id"] for a in pinned["artifacts"]] == [family["merged"]["id"]]


def test_published_artifacts_cannot_be_deleted(ctx, family):
    ctx.svc.lineage.mark_published(family["quant"]["id"], {"type": "ollama", "name": "pyg-mg:latest"})
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.lineage.delete(family["quant"]["id"])
    assert exc.value.code == "conflict" and "Unpublish" in exc.value.hint
    ctx.svc.lineage.unmark_published(family["quant"]["id"], "ollama", "pyg-mg:latest")
    assert ctx.svc.lineage.delete(family["quant"]["id"])["children_left_orphaned"] == [family["tag"]["id"]]


def test_mark_published_replaces_the_same_entry(ctx, family):
    for port in (8100, 8101):
        ctx.svc.lineage.mark_published(family["gguf"]["id"], {"type": "llama", "name": "pyg-x", "port": port})
    published = ctx.svc.store.artifact(family["gguf"]["id"])["published"]
    assert len(published) == 1 and published[0]["port"] == 8101


def test_delete_with_files_removes_them_inside_the_work_folder(ctx, family):
    folder = Path(family["adapter"]["path"])
    out = call(ctx.svc, "artifact_delete", artifact=family["adapter"]["id"], delete_files=True, confirm=True)
    assert out["files_removed"] is True and not folder.exists()
    assert family["base"]["id"] in {a["id"] for a in ctx.svc.store.artifacts()}


def test_delete_never_touches_files_outside_the_work_folder(ctx, tmp_path):
    outside = tmp_path / "precious"
    outside.mkdir()
    (outside / "thesis.txt").write_text("keep", encoding="utf-8")
    art = ctx.svc.store.create_artifact("merged", "elsewhere", path=str(outside))
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "artifact_delete", artifact=art["id"], delete_files=True, confirm=True)
    assert exc.value.code == "forbidden" and (outside / "thesis.txt").read_text(encoding="utf-8") == "keep"
    assert ctx.svc.store.artifact(art["id"])


def test_delete_never_removes_a_base_models_files(ctx, base_model):
    out = call(ctx.svc, "artifact_delete", artifact=base_model["id"], delete_files=True, confirm=True)
    assert out["files_removed"] is False and (Path(base_model["path"]) / "config.json").is_file()


def test_delete_needs_confirmation(ctx, family):
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "artifact_delete", artifact=family["adapter"]["id"])
    assert exc.value.code == "confirm_required"


def test_storage_adds_up_by_kind(ctx, family):
    st = ctx.svc.lineage.storage()
    assert st["by_kind"]["gguf"] == 130 and st["by_kind"]["adapter"] == 100 and st["total"] == sum(st["by_kind"].values())


# ------------------------------------------------------------------------------------------------ perplexity side by side
def measure(value, error=0.05, **kw):
    return {"value": value, "error": error, "ctx": 512, "chunks": 3, "text_tag": "bundled", **kw}


@pytest.fixture
def measured(ctx, family):
    """The family with perplexity on the fine-tune's f16 and Q4_K_M, plus the base model's own f16 and Q4_K_M measured the same way."""
    s = ctx.svc.store
    out = ctx.work / "outputs"
    base_f16 = s.create_artifact("gguf", "base-f16", path=str(out / "g" / "f.bin"), size=100, parents=[family["base"]["id"]])
    base_q4 = s.create_artifact("gguf", "base-q4", path=str(out / "g" / "f.bin"), size=30, parents=[base_f16["id"]], metrics={"quant": "Q4_K_M"},
                                recipe={"params": {"qtype": "Q4_K_M"}})
    for art, value in ((base_f16, 8.0), (base_q4, 8.4), (family["gguf"], 7.0), (family["quant"], 7.35)):
        s.merge_metrics(art["id"], "ppl", measure(value))
    return dict(family, base_f16=base_f16, base_q4=base_q4)


def test_the_measured_perplexity_and_its_error_are_on_the_card(ctx, measured):
    card = ctx.svc.lineage.card(ctx.svc.store.artifact(measured["quant"]["id"]))
    assert card["ppl"]["value"] == 7.35 and card["ppl"]["error"] == 0.05


def test_a_quantization_is_compared_with_its_unquantized_parent(ctx, measured):
    cards = ctx.svc.lineage.annotate_ppl([ctx.svc.lineage.card(ctx.svc.store.artifact(measured["quant"]["id"]))])
    parent = cards[0]["ppl_cmp"]["parent"]
    assert parent["id"] == measured["gguf"]["id"] and parent["delta"] == pytest.approx(0.35) and parent["pct"] == pytest.approx(5.0)


def test_a_fine_tune_is_compared_with_the_bases_own_file_of_the_same_quantization(ctx, measured):
    cards = ctx.svc.lineage.annotate_ppl([ctx.svc.lineage.card(ctx.svc.store.artifact(measured["quant"]["id"]))])
    base = cards[0]["ppl_cmp"]["base"]
    assert base["id"] == measured["base_q4"]["id"] and base["delta"] == pytest.approx(7.35 - 8.4), "negative: the fine-tune is better"


def test_the_bases_own_files_are_not_compared_with_themselves(ctx, measured):
    cards = ctx.svc.lineage.annotate_ppl([ctx.svc.lineage.card(ctx.svc.store.artifact(measured["base_q4"]["id"]))])
    assert "base" not in cards[0].get("ppl_cmp", {})


def test_numbers_from_another_text_or_context_are_never_compared(ctx, measured):
    ctx.svc.store.merge_metrics(measured["gguf"]["id"], "ppl", measure(7.0, ctx=2048))
    cards = ctx.svc.lineage.annotate_ppl([ctx.svc.lineage.card(ctx.svc.store.artifact(measured["quant"]["id"]))])
    assert "parent" not in cards[0].get("ppl_cmp", {})


def test_the_family_table_lists_every_file_measured_like_the_one_asked_about(ctx, measured):
    rows = ctx.svc.lineage.ppl_family(ctx.svc.store.artifact(measured["quant"]["id"]))
    assert {r["id"] for r in rows} == {measured[k]["id"] for k in ("base_f16", "base_q4", "gguf", "quant")}
    ctx.svc.store.merge_metrics(measured["base_f16"]["id"], "ppl", measure(8.0, chunks=9))
    rows = ctx.svc.lineage.ppl_family(ctx.svc.store.artifact(measured["quant"]["id"]))
    assert measured["base_f16"]["id"] not in {r["id"] for r in rows}
    assert "ppl_family" in ctx.svc.lineage.detail(measured["quant"]["id"])


def test_two_measurements_added_to_one_artifact_never_overwrite_each_other(ctx, family):
    gguf = family["gguf"]["id"]
    threads = [threading.Thread(target=ctx.svc.store.merge_metrics, args=(gguf, f"k{i}", i)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    metrics = ctx.svc.store.artifact(gguf)["metrics"]
    assert all(metrics.get(f"k{i}") == i for i in range(20))
