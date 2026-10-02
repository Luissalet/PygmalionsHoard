"""Building, versioning, reviewing, exporting and deleting datasets through the service and the tools."""

import json
from pathlib import Path

import pytest

from conftest import call
from helpers import chat_records
from pygmalion_hoard.errors import PygmalionError


def jsonl(records):
    return "\n".join(json.dumps(r, ensure_ascii=False) for r in records)


def build(svc, records=None, **spec):
    spec.setdefault("name", "Prueba")
    spec.setdefault("sources", [{"type": "jsonl", "text": jsonl(records or chat_records(40))}])
    return svc.datasets.build(spec)


def read(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines()]


# ------------------------------------------------------------------------------------------------ build
def test_a_chat_dataset_is_detected_and_split(ctx):
    out = build(ctx.svc, chat_records(100))
    v = out["version"]
    assert v["n"] == 1 and v["kind"] == "chat" and v["records"] == 100 and v["usable"] == 100
    assert v["eval"] == 20 and v["train"] == 80, "the default is 5 % with a minimum of 20 records"
    assert v["stats"]["report"][0]["op"] == "load" and Path(v["path"]).is_file()
    assert out["dataset"]["latest"]["id"] == v["id"] and len(v["sha256"]) == 64


def test_the_split_can_be_chosen(ctx):
    v = build(ctx.svc, chat_records(100), split={"eval_pct": 10, "min_eval": 5, "seed": 7})["version"]
    assert v["eval"] == 10 and v["splits"]["seed"] == 7


def test_the_same_input_gives_the_same_split_and_hash(ctx):
    a = build(ctx.svc, chat_records(50), name="A")["version"]
    b = build(ctx.svc, chat_records(50), name="B")["version"]
    assert a["sha256"] == b["sha256"] and ctx.svc.store.version(a["id"])["splits"] == ctx.svc.store.version(b["id"])["splits"]


def test_instruction_and_chat_mix_becomes_chat(ctx):
    mixed = chat_records(5) + [{"instruction": f"Di {i}", "output": f"Dije {i}"} for i in range(5)]
    assert build(ctx.svc, mixed)["version"]["kind"] == "chat"


def test_text_records_stay_text_and_can_be_forced(ctx):
    texts = [{"text": f"Párrafo {i} con bastante contenido para ser contado como texto corrido."} for i in range(30)]
    assert build(ctx.svc, texts, name="T")["version"]["kind"] == "text"
    assert build(ctx.svc, chat_records(10), name="F", kind="instruction")["version"]["kind"] == "instruction"
    with pytest.raises(PygmalionError):
        build(ctx.svc, chat_records(3), name="K", kind="poem")


def test_a_second_version_is_added_to_the_same_dataset(ctx):
    first = build(ctx.svc, chat_records(30), name="Crece")
    second = ctx.svc.datasets.build({"dataset": "Crece", "sources": [{"type": "jsonl", "text": jsonl(chat_records(60))}]})
    assert second["version"]["n"] == 2 and second["dataset"]["id"] == first["dataset"]["id"] and second["dataset"]["versions"] == 2
    assert read(first["version"]["path"])[0] == read(first["version"]["path"])[0] and first["version"]["records"] == 30
    assert ctx.svc.datasets.get_version("Crece")["n"] == 2 and ctx.svc.datasets.get_version("Crece", 1)["records"] == 30


def test_kinds_cannot_be_mixed_in_one_dataset(ctx):
    build(ctx.svc, chat_records(10), name="Chats")
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.datasets.build({"dataset": "Chats", "sources": [{"type": "jsonl", "text": jsonl([{"text": "x" * 80}] * 3)}]})
    assert "holds chat records" in exc.value.message


@pytest.mark.parametrize("spec,fragment", [
    ({"sources": []}, "at least one source"),
    ({"name": "", "sources": [{"type": "jsonl", "text": '{"text":"hola mundo hola"}'}]}, "needs a `name`"),
    ({"name": "bad/name", "sources": [{"type": "jsonl", "text": '{"text":"hola mundo hola"}'}]}, "needs a `name`"),
    ({"name": "x", "sources": [{"type": "jsonl", "text": "no json at all"}]}, "No usable records"),
    ({"name": "x", "sources": [{"type": "jsonl", "text": jsonl(chat_records(3))}], "operations": [{"op": "length", "min_chars": 100000}]}, "removed every record"),
])
def test_bad_specs_are_explained(ctx, spec, fragment):
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.datasets.build(spec)
    assert fragment in exc.value.message + exc.value.hint


def test_operations_in_the_spec_run_in_order_and_are_reported(ctx):
    records = chat_records(30) + chat_records(5)             # five exact duplicates
    records.append({"messages": [{"role": "user", "content": "Mi correo es ana@example.com"}, {"role": "assistant", "content": "Anotado, gracias por contármelo."}]})
    out = build(ctx.svc, records, operations=[{"op": "dedupe_exact"}, {"op": "pii", "mode": "mask"}])
    report = {r["op"]: r for r in out["version"]["stats"]["report"]}
    assert report["dedupe_exact"]["before"] == 36 and report["dedupe_exact"]["after"] == 31
    assert out["version"]["records"] == 31
    text = Path(out["version"]["path"]).read_text(encoding="utf-8")
    assert "ana@example.com" not in text


def test_the_recipe_never_keeps_inline_text_or_secrets(ctx):
    out = build(ctx.svc, chat_records(10), sources=[{"type": "jsonl", "text": jsonl(chat_records(10))}])
    recipe = json.dumps(ctx.svc.store.version(out["version"]["id"])["recipe"])
    assert "Pregunta número" not in recipe


def test_csv_and_file_sources_work_through_the_service(ctx, tmp_path):
    csv_text = "pregunta,respuesta\n" + "\n".join(f"¿Qué es {i}?,Es el número {i} de la serie." for i in range(25))
    out = ctx.svc.datasets.build({"name": "Tabla", "sources": [{"type": "csv", "text": csv_text, "columns": {"prompt": "pregunta", "response": "respuesta"}}]})
    assert out["version"]["records"] == 25 and out["version"]["kind"] in ("instruction", "chat")
    folder = tmp_path / "textos"
    folder.mkdir()
    (folder / "uno.txt").write_text("Una historia larga. " * 300, encoding="utf-8")
    (folder / ".secreto.txt").write_text("no debe entrar " * 50, encoding="utf-8")
    out = ctx.svc.datasets.build({"name": "Carpeta", "sources": [{"type": "folder", "path": str(folder), "extensions": [".txt"]}]})
    assert out["version"]["kind"] == "text" and out["version"]["records"] >= 3
    assert "no debe entrar" not in Path(out["version"]["path"]).read_text(encoding="utf-8")


def test_a_build_can_be_cancelled(ctx):
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.datasets.build({"name": "C", "sources": [{"type": "jsonl", "text": jsonl(chat_records(5))}]}, cancelled=lambda: True)
    assert exc.value.message == "Cancelled."
    assert ctx.svc.store.datasets() == []


def test_progress_is_reported_per_source(ctx):
    seen = []
    ctx.svc.datasets.build({"name": "P", "sources": [{"type": "jsonl", "text": jsonl(chat_records(5))}, {"type": "jsonl", "text": jsonl(chat_records(6))}]},
                           progress=lambda m, s, t: seen.append((m, s, t)))
    assert [s[1:] for s in seen[:2]] == [(0, 2), (1, 2)] and seen[0][0].startswith("source 1/2")


def test_preview_does_not_create_anything(ctx):
    out = ctx.svc.datasets.preview({"type": "jsonl", "text": jsonl(chat_records(8))}, limit=3)
    assert out["count"] == 8 and len(out["items"]) == 3 and out["kind"] == "chat"
    assert ctx.svc.store.datasets() == []


# ------------------------------------------------------------------------------------------------ versions: operations and review
def test_apply_makes_a_new_version_and_keeps_the_old_one(ctx):
    v1 = build(ctx.svc, chat_records(20) + chat_records(10))["version"]
    out = ctx.svc.datasets.apply_operations("Prueba", [{"op": "dedupe_exact"}])
    v2 = out["version"]
    assert v2["n"] == 2 and v2["parent"] == v1["id"] and v2["records"] == 20 and v1["records"] == 30
    assert len(read(v1["path"])) == 30 and len(read(v2["path"])) == 20 and out["report"][0]["op"] == "dedupe_exact"
    assert ctx.svc.store.version(v2["id"])["recipe"]["from_version"] == v1["id"]


def test_apply_that_removes_everything_is_an_error_and_makes_no_version(ctx):
    build(ctx.svc, chat_records(5))
    with pytest.raises(PygmalionError):
        ctx.svc.datasets.apply_operations("Prueba", [{"op": "length", "min_chars": 99999}])
    assert len(ctx.svc.store.versions(ctx.svc.store.dataset("Prueba")["id"])) == 1


def test_records_can_be_filtered_and_paged(ctx):
    recs = chat_records(30)
    recs[3]["messages"][0]["content"] = "Mi teléfono es 612 345 678 llámame cuando quieras."
    build(ctx.svc, recs)
    d = ctx.svc.datasets
    page = d.records("Prueba", offset=5, limit=10)
    assert page["total"] == 30 and [r["index"] for r in page["records"]] == list(range(5, 15)) and page["records"][0]["kind"] == "chat"
    assert d.records("Prueba", text="número 7 ")["total"] == 1 and d.records("Prueba", text="NÚMERO 7 ")["total"] == 1
    assert d.records("Prueba", has_pii=True)["total"] == 1 and d.records("Prueba", has_pii=True)["records"][0]["index"] == 3
    splits = {s: d.records("Prueba", split=s, limit=200)["total"] for s in ("train", "eval", "none")}
    assert splits == {"train": 15, "eval": 15, "none": 0}, "the evaluation share is never more than half"
    assert d.records("Prueba", lang="es", limit=200)["total"] == 30 and d.records("Prueba", lang="en")["total"] == 0
    assert all("_meta" not in r["record"] for r in d.records("Prueba")["records"])


def test_review_accepts_rejects_and_edits_into_a_new_version(ctx):
    build(ctx.svc, chat_records(30))
    new = {"messages": [{"role": "user", "content": "Pregunta corregida"}, {"role": "assistant", "content": "Respuesta corregida por una persona."}]}
    out = ctx.svc.datasets.review("Prueba", reject=[0, 1], edit={"2": new})
    v = out["version"]
    assert out["changed"] == {"accepted": 0, "rejected": 2, "edited": 1} and v["n"] == 2 and v["usable"] == 28 and v["rejected"] == 2
    page = ctx.svc.datasets.records("Prueba", limit=5)
    assert [r["status"] for r in page["records"][:3]] == ["rejected", "rejected", "ok"]
    assert page["records"][2]["record"]["messages"][0]["content"] == "Pregunta corregida"
    assert ctx.svc.datasets.records("Prueba", n=1, limit=5)["records"][0]["status"] == "ok", "version 1 is untouched"


def test_review_checks_indexes_and_edits(ctx):
    build(ctx.svc, chat_records(10))
    for kwargs in ({"accept": [10]}, {"reject": [-1]}, {"edit": {"99": {"text": "x"}}}):
        with pytest.raises(PygmalionError) as exc:
            ctx.svc.datasets.review("Prueba", **kwargs)
        assert "out of range" in exc.value.message
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.datasets.review("Prueba", edit={"0": {"messages": []}})
    assert "not valid" in exc.value.message


def test_synthetic_records_wait_for_review_and_stay_out_of_training(ctx):
    ctx.svc.datasets.teacher_factory = lambda: (lambda messages, max_tokens: json.dumps(
        [{"prompt": f"Pregunta sintética {i} bastante larga", "response": f"Respuesta sintética {i} bastante larga también"} for i in range(6)]))
    out = ctx.svc.datasets.build({"name": "Sint", "sources": [{"type": "jsonl", "text": jsonl(chat_records(10))},
                                                              {"type": "synthetic", "task": "Preguntas sobre la casa", "from": [{"type": "jsonl", "text": jsonl(chat_records(4))}],
                                                               "max_items": 6}]})
    v = out["version"]
    assert v["pending"] > 0 and v["usable"] == v["records"] - v["pending"]
    export = ctx.svc.datasets.export_for_training("Sint", ctx.tmp / "exp")
    assert export["n_train"] + export["n_eval"] == v["usable"]
    accepted = ctx.svc.datasets.review("Sint", accept_all_pending=True)
    assert accepted["version"]["pending"] == 0 and accepted["version"]["usable"] == v["records"]


# ------------------------------------------------------------------------------------------------ export and delete
def test_export_writes_disjoint_clean_files(ctx):
    build(ctx.svc, chat_records(60))
    ctx.svc.datasets.review("Prueba", reject=[5, 6])
    out = ctx.svc.datasets.export_for_training("Prueba", ctx.tmp / "export")
    train, ev = read(out["train"]), read(out["eval"])
    assert out["n_train"] == len(train) and out["n_eval"] == len(ev) == 20 and len(train) + len(ev) == 58
    texts = [json.dumps(r, sort_keys=True) for r in train + ev]
    assert len(set(texts)) == 58 and all("_meta" not in r for r in train + ev)
    assert not any("número 5 " in json.dumps(r, ensure_ascii=False) or "número 6 " in json.dumps(r, ensure_ascii=False) for r in train + ev)
    again = ctx.svc.datasets.export_for_training("Prueba", ctx.tmp / "export2")
    assert read(again["train"]) == train and read(again["eval"]) == ev, "the export is deterministic"


def test_export_of_an_older_version_by_number(ctx):
    build(ctx.svc, chat_records(40))
    ctx.svc.datasets.review("Prueba", reject=list(range(10)))
    assert ctx.svc.datasets.export_for_training("Prueba", ctx.tmp / "e1", n=1)["n_train"] + ctx.svc.datasets.export_for_training("Prueba", ctx.tmp / "e1", n=1)["n_eval"] == 40
    latest = ctx.svc.datasets.export_for_training("Prueba", ctx.tmp / "e2")
    assert latest["n_train"] + latest["n_eval"] == 30


def test_delete_removes_files_and_reports_artifacts_that_still_cite_it(ctx, base_model, dataset):
    folder = Path(ctx.svc.store.version(dataset["id"])["path"]).parent
    ctx.svc.store.create_artifact("adapter", "ad", dataset_version=dataset["id"], parents=[base_model["id"]])
    out = ctx.svc.datasets.delete("Casa de la abuela")
    assert out["still_referenced_by"] == ["ad"] and not folder.exists()
    assert ctx.svc.store.artifacts(kind="adapter")[0]["name"] == "ad"
    with pytest.raises(PygmalionError):
        ctx.svc.datasets.get("Casa de la abuela")


def test_a_missing_version_file_is_reported_not_crashed(ctx, dataset):
    Path(ctx.svc.store.version(dataset["id"])["path"]).unlink()
    with pytest.raises(PygmalionError) as exc:
        ctx.svc.datasets.records("Casa de la abuela")
    assert exc.value.code == "not_found" and "rebuild" in exc.value.hint


# ------------------------------------------------------------------------------------------------ through the tools
def test_dataset_tools_end_to_end(ctx):
    made = call(ctx.svc, "dataset_create", name="Por tool", sources=[{"type": "jsonl", "text": jsonl(chat_records(30))}], wait_s=30)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(made["job"]["id"])
    assert job["state"] == "done" and job["result"]["version"]["records"] == 30
    listed = call(ctx.svc, "datasets_list")
    assert [d["name"] for d in listed["datasets"]] == ["Por tool"]
    got = call(ctx.svc, "dataset_get", dataset="Por tool")
    assert got["dataset"]["version_list"][0]["tokens"] > 0 and got["version"]["n"] == 1 and got["version"]["stats"]["report"]
    recs = call(ctx.svc, "dataset_records", dataset="Por tool", limit=2, text="número 1 ")
    assert recs["total"] == 1
    reviewed = call(ctx.svc, "dataset_review", dataset="Por tool", reject=[0])
    assert reviewed["changed"]["rejected"] == 1
    applied = call(ctx.svc, "dataset_apply", dataset="Por tool", operations=[{"op": "dedupe_exact"}])
    assert applied["version"]["n"] == 3
    with pytest.raises(PygmalionError) as exc:
        call(ctx.svc, "dataset_delete", dataset="Por tool")
    assert exc.value.code == "confirm_required"
    assert call(ctx.svc, "dataset_delete", dataset="Por tool", confirm=True)["deleted"] == "Por tool"
    assert call(ctx.svc, "datasets_list")["datasets"] == []


def test_dataset_preview_source_tool(ctx):
    out = call(ctx.svc, "dataset_preview_source", source={"type": "jsonl", "text": jsonl(chat_records(8))}, limit=3)
    assert out and ctx.svc.store.datasets() == []


def test_a_failed_build_job_keeps_the_reason(ctx):
    made = call(ctx.svc, "dataset_create", name="Roto", sources=[{"type": "jsonl", "text": "{nope"}], wait_s=0)
    ctx.svc.jobs.run_until_idle()
    job = ctx.svc.store.job(made["job"]["id"])
    assert job["state"] == "failed" and "No usable records" in job["error"] and job["hint"]


def test_a_path_pasted_with_its_quotes_is_read_and_the_folder_that_holds_the_data_folder_is_refused(ctx, tmp_path):
    f = tmp_path / "notas.txt"
    f.write_text("Una historia larga. " * 300, encoding="utf-8")
    out = ctx.svc.datasets.build({"name": "Comillas", "sources": [{"type": "files", "paths": [f'"{f}"']}]})
    assert out["version"]["records"] >= 1
    with pytest.raises(PygmalionError) as info:
        ctx.svc.datasets.build({"name": "Padre", "sources": [{"type": "folder", "path": str(ctx.data.parent)}]})
    assert info.value.code == "forbidden"
