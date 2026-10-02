"""Record formats, personal-data masking, duplicate detection, language, splits and statistics."""

import random

import pytest

from pygmalion_hoard.datasets import formats as F
from pygmalion_hoard.datasets import operations as O
from pygmalion_hoard.datasets import pii
from pygmalion_hoard.datasets import sources as S
from pygmalion_hoard.errors import PygmalionError


# ------------------------------------------------------------------ formats
@pytest.mark.parametrize("rec,kind", [({"messages": [1]}, "chat"), ({"conversations": []}, "chat"), ({"prompt": "a", "response": "b"}, "instruction"),
                                      ({"instruction": "a", "output": "b"}, "instruction"), ({"text": "x"}, "text"), ({"foo": 1}, None), ("str", None)])
def test_detect_kind(rec, kind):
    assert F.detect_kind(rec) == kind


def test_sharegpt_conversations_are_normalised():
    rec = {"conversations": [{"from": "human", "value": " hola "}, {"from": "gpt", "value": "buenas"}]}
    assert F.normalize_record(rec) == {"messages": [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "buenas"}]}


def test_alpaca_input_is_appended_to_the_prompt():
    rec = {"instruction": "Traduce", "input": "hello", "output": "hola"}
    assert F.normalize_record(rec) == {"prompt": "Traduce\n\nhello", "response": "hola"}


@pytest.mark.parametrize("rec", [{"messages": []}, {"messages": [{"role": "user", "content": "q"}]}, {"messages": [{"role": "robot", "content": "x"}]},
                                 {"messages": ["x"]}, {"prompt": "", "response": "x"}, {"prompt": "x", "response": " "}, {"text": "   "},
                                 {"messages": [{"role": "assistant", "content": ""}]}, {"text": "x" * (F.MAX_RECORD_CHARS + 1)}])
def test_unusable_records_are_dropped(rec):
    assert F.normalize_record(rec) is None


def test_meta_survives_normalisation_and_helpers():
    rec = F.normalize_record({"prompt": "a", "response": "b", "_meta": {"status": "pending", "source": "s"}})
    assert F.status_of(rec) == "pending" and F.meta_of(rec)["source"] == "s"
    assert F.status_of({"text": "x"}) == "ok" and F.status_of({"_meta": {"status": "weird"}}) == "ok"
    assert "_meta" not in F.strip_meta(rec) and F.with_meta(rec, status="ok")["_meta"]["source"] == "s"


def test_kind_conversions():
    inst = {"prompt": "p", "response": "r", "system": "s"}
    chat = F.to_kind(inst, "chat")
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]
    assert F.to_kind(chat, "instruction") == inst
    multi = {"messages": [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}, {"role": "user", "content": "c"}, {"role": "assistant", "content": "d"}]}
    assert F.to_kind(multi, "instruction") is None
    assert F.to_kind(inst, "text")["text"] == "p\n\nr" and F.to_kind({"text": "x"}, "chat") is None


def test_record_text_and_content_text():
    chat = {"messages": [{"role": "user", "content": "hola"}, {"role": "assistant", "content": "adiós"}]}
    assert F.record_text(chat) == "user: hola\nassistant: adiós" and F.content_text(chat) == "hola\nadiós"


def test_map_text_touches_every_field():
    assert F.map_text({"prompt": "a", "response": "b", "system": "c"}, str.upper) == {"prompt": "A", "response": "B", "system": "C"}
    assert F.map_text({"text": "a"}, str.upper) == {"text": "A"}


# ------------------------------------------------------------------ personal data
def test_valid_spanish_identifiers_are_found():
    text = "DNI 12345678Z, NIE X1234567L, correo ana.perez@example.com, tel: 612 345 678, IBAN ES9121000418450200051332."
    kinds = sorted(h["kind"] for h in pii.scan(text))
    assert kinds == ["dni", "email", "iban", "nie", "phone"]
    hit = next(h for h in pii.scan(text) if h["kind"] == "iban")
    assert hit["text"] == "ES9121000418450200051332" and text[hit["start"]:hit["end"]] == hit["text"]


def test_numbers_that_only_look_like_identifiers_are_left_alone():
    """Only values whose checksum is right count: the DNI letter, the NIE letter, the CIF control character and the IBAN sum."""
    assert pii.scan("DNI 12345678A, NIE X1234567A, CIF A58818502 sin control") == []
    assert [h["kind"] for h in pii.scan("CIF de la empresa: A58818501")] == ["cif"]    # a CIF is found now (it was not before)
    assert pii.scan("CIF de la empresa: A58818502") == []


def test_a_bare_nine_digit_number_is_not_a_phone_but_a_prefixed_labelled_or_grouped_one_is():
    assert pii.scan("el pedido 612345678 salió") == []
    for text in ("llama al +34 612 345 678", "tel: 612345678", "móvil 612 345 678", "612 34 56 78"):
        assert [h["kind"] for h in pii.scan(text)] == ["phone"], text


def test_card_numbers_need_a_valid_luhn_sum():
    assert [h["kind"] for h in pii.scan("tarjeta 4111 1111 1111 1111")] == ["card"]
    assert pii.scan("pedido 4111 1111 1111 1112") == []


def test_invalid_iban_is_ignored():
    assert pii.scan("ES0000000000000000000000") == []
    from pygmalion_hoard.hoard_link.idcheck import iban_ok
    assert iban_ok("ES91 2100 0418 4502 0005 1332") and not iban_ok("ES91 2100 0418 4502 0005 1333")


def test_mask_is_idempotent_and_keeps_the_text_around():
    text = "Escríbeme a luis@example.org o llama al 612 345 678 por favor."
    once = pii.mask(text)
    assert once == "Escríbeme a <EMAIL> o llama al <PHONE> por favor." and pii.mask(once) == once


def test_plain_numbers_are_not_flagged():
    assert pii.scan("Tenía 123456789 manzanas y el año 2024.") == []
    assert pii.scan("La versión 3.13.1 salió en octubre.") == []


def test_counts():
    assert pii.counts("a@b.es y c@d.es y 12345678Z") == {"email": 2, "dni": 1}


# ------------------------------------------------------------------ duplicates
def rec(text):
    return {"text": text}


def test_exact_dedupe_ignores_case_accents_and_spacing():
    kept, removed = O.dedupe_exact([rec("Árbol  grande"), rec("arbol grande"), rec("otro")])
    assert len(kept) == 2 and removed == 1


def test_exact_dedupe_keeps_the_first():
    kept, _ = O.dedupe_exact([rec("Hola"), rec("hola")])
    assert kept[0]["text"] == "Hola"


BASE = "El zorro marrón salta sobre el perro perezoso mientras la luna sube despacio por encima de los tejados de la ciudad vieja y todos duermen"


def test_near_dedupe_drops_small_edits_and_keeps_different_texts():
    near = BASE.replace("marrón", "pardo")
    other = "Una receta sencilla de lentejas con verduras, pimentón y laurel que mejora de un día para otro cuando se deja reposar en la nevera"
    kept, removed = O.dedupe_near([rec(BASE), rec(near), rec(other)], 0.7)
    assert removed == 1 and [k["text"] for k in kept] == [BASE, other]


def test_near_dedupe_threshold_one_keeps_distinct_texts():
    kept, removed = O.dedupe_near([rec(BASE), rec(BASE.replace("marrón", "pardo"))], 1.0)
    assert removed == 0 and len(kept) == 2


def test_sketch_properties():
    assert O.sketch(BASE) == O.sketch(BASE) and len(O.sketch(BASE)) <= O.SKETCH_K
    assert O.jaccard_estimate(O.sketch(BASE), O.sketch(BASE)) == 1.0
    assert O.jaccard_estimate(O.sketch(BASE), O.sketch("completamente distinto " * 20)) < 0.2


def test_near_dedupe_scales_to_many_records():
    rng = random.Random(0)
    words = [f"palabra{i}" for i in range(400)]
    recs = [rec(" ".join(rng.sample(words, 30))) for _ in range(300)]
    kept, removed = O.dedupe_near(recs + recs[:50], 0.85)
    assert removed >= 50 and len(kept) >= 290


# ------------------------------------------------------------------ language, length, operations
def test_language_detection():
    assert O.detect_language("El perro de mi vecino ladra por la noche y no me deja dormir")[0] == "es"
    assert O.detect_language("The dog of my neighbour barks at night and does not let me sleep")[0] == "en"
    assert O.detect_language("xx yy")[0] == "unknown" and O.detect_language("zzz qqq www vvv uuu")[0] == "unknown"


def test_language_filter_keeps_unknown_text():
    recs = [rec("El perro de mi vecino ladra por la noche y no me deja dormir"), rec("The dog of my neighbour barks at night and does not sleep"), rec("???")]
    kept, removed = O.filter_language(recs, ["es"])
    assert removed == 1 and len(kept) == 2


def test_length_filter():
    kept, removed = O.filter_length([rec("a"), rec("a" * 50), rec("a" * 500)], min_chars=10, max_chars=100)
    assert [len(k["text"]) for k in kept] == [50] and removed == 2


def test_pii_operation_modes():
    recs = [rec("hola"), rec("mi correo es a@b.es")]
    masked, changed = O.pii_apply(recs, "mask")
    assert masked[1]["text"] == "mi correo es <EMAIL>" and changed == 1
    dropped, changed = O.pii_apply(recs, "drop")
    assert len(dropped) == 1 and changed == 1
    assert O.pii_scan(recs) == {"records_with_pii": 1, "by_kind": {"email": 1}}


def test_run_operations_reports_each_step():
    out, report = O.run_operations([rec("a" * 30), rec("a" * 30), rec("b" * 5)], [{"op": "dedupe_exact"}, {"op": "length", "min_chars": 10}])
    assert len(out) == 1 and [r["op"] for r in report] == ["dedupe_exact", "length"] and report[0]["changed"] == 1 and report[1]["after"] == 1


def test_unknown_operation_is_an_error():
    with pytest.raises(ValueError):
        O.run_operations([rec("a")], [{"op": "explode"}])


# ------------------------------------------------------------------ split and statistics
def test_split_is_deterministic_and_disjoint():
    usable = list(range(200))
    a, b = O.make_split(usable, 5, 5, seed=3), O.make_split(usable, 5, 5, seed=3)
    assert a == b and len(a["eval"]) == 10 and O.make_split(usable, 5, 5, seed=4)["eval"] != a["eval"]
    train, held = O.split_indices(usable, a)
    assert not set(train) & set(held) and len(train) + len(held) == 200


def test_split_has_a_minimum_but_never_more_than_half():
    assert len(O.make_split(list(range(100)), 5, 20)["eval"]) == 20
    assert len(O.make_split(list(range(10)), 5, 20)["eval"]) == 5
    assert O.make_split([0], 5, 20)["eval"] == [] and O.make_split([], 5, 20)["eval"] == []


def test_split_only_uses_the_usable_indices():
    usable = [1, 3, 5, 7, 9, 11, 13, 15]
    assert set(O.make_split(usable, 50, 1)["eval"]) <= set(usable)


def test_stats_counts_everything():
    recs = [{"messages": [{"role": "user", "content": "hola amigo"}, {"role": "assistant", "content": "buenas"}], "_meta": {"synthetic": True, "status": "pending"}},
            {"prompt": "a b c", "response": "d e f"}, {"text": "x" * 50, "_meta": {"status": "rejected"}}]
    s = O.stats(recs)
    assert s["records"] == 3 and s["usable"] == 1 and s["pending"] == 1 and s["rejected"] == 1 and s["synthetic"] == 1
    assert s["roles"]["user"] == 2 and sum(b["count"] for b in s["histogram"]) == 3 and s["tokens"] > 0


def test_stats_of_nothing():
    assert O.stats([])["records"] == 0 and O.stats([])["max_chars"] == 0


def test_preview_text_truncates():
    assert O.preview_text({"text": "x" * 500}).endswith("…") and len(O.preview_text({"text": "x" * 500})) == 240


# ------------------------------------------------------------------ source helpers
def test_chunk_text_respects_paragraphs_and_the_limit():
    text = "\n\n".join(f"Párrafo {i}. " + "palabra " * 30 for i in range(10))
    chunks = S.chunk_text(text, 500)
    assert all(len(c) <= 500 for c in chunks) and len(chunks) > 3 and "Párrafo 0" in chunks[0]


def test_chunk_text_splits_a_giant_sentence_hard():
    chunks = S.chunk_text("x" * 1000, 300)
    assert all(len(c) <= 300 for c in chunks) and sum(len(c) for c in chunks) >= 1000


def test_dig_and_family_items():
    data = {"a": {"b": [{"c": 5}, {"c": 6}]}}
    assert S.dig(data, "a.b.1.c") == 6 and S.dig(data, "a.x") is None and S.dig(data, "a.b.9") is None
    assert S.family_items({"cards": [1, 2]}) == [1, 2] and S.family_items(data, "a.b") == [{"c": 5}, {"c": 6}] and S.family_items(5) == []


def test_map_item_requires_every_field():
    assert S.map_item({"front": "q", "back": "a"}, {"prompt": "front", "response": "back"}) == {"prompt": "q", "response": "a"}
    assert S.map_item({"front": "q", "back": ""}, {"prompt": "front", "response": "back"}) == {}


def test_parse_json_items_variants():
    assert S.parse_json_items('[{"prompt": "a", "response": "b"}]') == [{"prompt": "a", "response": "b"}]
    assert S.parse_json_items('Claro:\n```json\n[{"prompt": "a", "response": "b"}]\n```\nespero que sirva') == [{"prompt": "a", "response": "b"}]
    assert S.parse_json_items('{"items": [{"prompt": "a"}]}') == [{"prompt": "a"}]
    assert S.parse_json_items("no json at all") == [] and S.parse_json_items("[1, 2]") == []


def test_unknown_source_type():
    with pytest.raises(PygmalionError) as exc:
        S.load_source({"type": "ftp"}, S.SourceContext())
    assert exc.value.code == "invalid"


def test_csv_source_maps_columns_and_detects_the_delimiter():
    ctx = S.SourceContext()
    rows = list(S.load_source({"type": "csv", "text": "pregunta;respuesta\nhola;buenas\nadiós;hasta luego\n", "columns": {"prompt": "pregunta", "response": "respuesta"}}, ctx))
    assert [r["prompt"] for r in rows] == ["hola", "adiós"] and rows[0]["_meta"]["source"] == "pasted csv"


def test_csv_source_reports_missing_columns():
    with pytest.raises(PygmalionError) as exc:
        list(S.load_source({"type": "csv", "text": "a,b\n1,2\n", "columns": {"prompt": "zzz"}}, S.SourceContext()))
    assert "zzz" in exc.value.message and "a, b" in exc.value.hint
    with pytest.raises(PygmalionError):
        list(S.load_source({"type": "csv", "text": "a\n1\n"}, S.SourceContext()))


def test_jsonl_source_skips_bad_lines_and_accepts_arrays():
    ctx = S.SourceContext()
    rows = list(S.load_source({"type": "jsonl", "text": '{"text": "a"}\nbroken\n[1]\n{"text": "b"}\n'}, ctx))
    assert len(rows) == 2 and "2 lines" in ctx.notes[0]
    rows = list(S.load_source({"type": "jsonl", "text": '[{"text": "a"}, {"text": "b"}]'}, S.SourceContext()))
    assert len(rows) == 2


def test_source_without_text_or_path_is_invalid():
    with pytest.raises(PygmalionError):
        list(S.load_source({"type": "jsonl"}, S.SourceContext()))


def test_files_source_chunks_inline_texts():
    rows = list(S.load_source({"type": "files", "items": [{"name": "a.txt", "text": "uno\n\ndos\n\ntres"}], "chunk_chars": 100}, S.SourceContext()))
    assert rows and rows[0]["_meta"]["source"] == "a.txt"


def test_folder_source_skips_hidden_and_secret_files(tmp_path):
    root = tmp_path / "a" / "b" / "c" / "d" / "e" / "f"        # deep on purpose: the depth of the folder must not matter
    (root / "docs").mkdir(parents=True)
    (root / "docs" / "nota.md").write_text("Contenido útil de la nota.", encoding="utf-8")
    (root / "docs" / "secrets.txt").write_text("clave", encoding="utf-8")
    (root / ".git").mkdir()
    (root / ".git" / "x.txt").write_text("no", encoding="utf-8")
    (root / "src" / "node_modules" / "pkg").mkdir(parents=True)
    (root / "src" / "node_modules" / "pkg" / "readme.md").write_text("no", encoding="utf-8")
    (root / "otro.pdf").write_bytes(b"%PDF")
    rows = list(S.load_source({"type": "folder", "path": str(root)}, S.SourceContext()))
    assert [r["_meta"]["source"] for r in rows] == ["docs/nota.md"], "labels use / on every OS"


@pytest.mark.parametrize("relative,skipped", [
    (r".git\x.txt", True), (r"sub\.git\objects\x.txt", True), (r"a\AppData\Roaming\n.md", True), (r"docs\node_modules\p\r.md", True),
    (r"docs\nota.md", False), (r"nota.md", False), (r"docs\secrets.txt", True), (r"docs\.env", True), (r"docs\.hidden.md", True),
    (r"gitlike\x.txt", False)])
def test_folder_filter_looks_at_path_parts_so_windows_paths_work(relative, skipped):
    """The filter must not depend on "/" in a string: a path as Windows writes it (backslashes) is judged the same way."""
    from pathlib import PureWindowsPath
    assert S.skipped_in_folder(PureWindowsPath(relative)) is skipped
    assert S.skipped_in_folder(PureWindowsPath(relative.upper())) is skipped, "Windows names are case-insensitive"


def test_folder_source_refuses_system_folders():
    with pytest.raises(PygmalionError) as exc:
        list(S.load_source({"type": "folder", "path": "/etc"}, S.SourceContext()))
    assert exc.value.code == "forbidden"


def test_family_source_needs_the_hub():
    with pytest.raises(PygmalionError) as exc:
        list(S.load_source({"type": "family", "app": "a", "tool": "t", "mapping": {"text": "x"}}, S.SourceContext()))
    assert exc.value.code == "offline"


def test_family_source_maps_items_and_caches_the_call():
    calls = []

    def fake(app, tool, args):
        calls.append((app, tool, args))
        return {"ok": True, "result": {"cards": [{"front": "q1", "back": "a1"}, {"front": "q2"}]}}

    ctx = S.SourceContext(family_call=fake)
    spec = {"type": "family", "app": "cards", "tool": "list", "mapping": {"prompt": "front", "response": "back"}}
    rows = list(S.load_source(spec, ctx))
    assert len(rows) == 1 and rows[0]["_meta"]["source"] == "cards.list" and "1 items" in ctx.notes[0]
    S.preview_source(spec, ctx)
    assert len(calls) == 1


def test_family_failure_becomes_a_clear_error():
    ctx = S.SourceContext(family_call=lambda *a: {"ok": False, "error": "app down"})
    with pytest.raises(PygmalionError) as exc:
        list(S.load_source({"type": "family", "app": "x", "tool": "y", "mapping": {"text": "t"}}, ctx))
    assert "app down" in exc.value.message


def test_synthetic_source_uses_the_teacher_and_marks_items_pending():
    answer = '[{"prompt": "¿Qué es X?", "response": "Es Y."}, {"question": "Otra", "answer": "Sí"}, {"prompt": "", "response": "vacío"}]'
    ctx = S.SourceContext(teacher=lambda msgs, n: answer)
    spec = {"type": "synthetic", "task": "preguntas", "per_chunk": 3, "from": [{"type": "files", "items": [{"name": "a", "text": "Texto de origen."}]}]}
    rows = list(S.load_source(spec, ctx))
    assert len(rows) == 2 and all(r["_meta"]["status"] == "pending" and r["_meta"]["synthetic"] for r in rows)


def test_synthetic_source_survives_a_failing_call_and_honours_the_cap():
    state = {"n": 0}

    def teacher(msgs, n):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError("boom")
        return '[{"prompt": "p", "response": "r"}, {"prompt": "p2", "response": "r2"}]'

    ctx = S.SourceContext(teacher=teacher)
    items = [{"name": str(i), "text": f"texto {i}"} for i in range(4)]
    rows = list(S.load_source({"type": "synthetic", "task": "t", "per_chunk": 2, "max_items": 3, "from": [{"type": "files", "items": items, "chunk_chars": 100}]}, ctx))
    assert len(rows) == 3 and "boom" in ctx.notes[0]


def test_synthetic_needs_a_teacher_and_a_task():
    with pytest.raises(PygmalionError) as exc:
        list(S.load_source({"type": "synthetic", "task": "t", "from": []}, S.SourceContext()))
    assert exc.value.code == "offline"
    with pytest.raises(PygmalionError):
        list(S.load_source({"type": "synthetic", "from": []}, S.SourceContext(teacher=lambda m, n: "")))


def test_synthetic_preview_only_estimates_unless_sampled():
    called = []
    ctx = S.SourceContext(teacher=lambda m, n: called.append(1) or "[]")
    spec = {"type": "synthetic", "task": "t", "from": [{"type": "files", "items": [{"name": "a", "text": "x"}]}]}
    out = S.preview_source(spec, ctx)
    assert out["estimate"]["calls"] == 1 and not called


def test_chunks_never_pass_the_limit_and_end_at_the_best_break():
    text = "\n\n".join(f"Párrafo {i}. " + "Una frase corta. " * 20 for i in range(12))
    chunks = S.chunk_text(text, 400)
    assert len(chunks) > 5 and all(len(c) <= 400 for c in chunks)
    assert all(c.rstrip()[-1] in ".!?" for c in chunks)                       # cut at sentence ends, not in the middle of a word
    assert S.chunk_text("", 400) == [] and S.chunk_text("corto", 50) == ["corto"] and S.chunk_text("a\r\n\r\nb", 400) == ["a\n\nb"]
    assert max(len(c) for c in S.chunk_text("palabra " * 500, 10)) <= 100          # the smallest size is 100


def test_text_files_are_read_whatever_their_encoding(tmp_path):
    text = "Canción del niño: ¿qué pasó? Árbol, corazón y camión."
    for name, data in (("utf8.txt", text.encode("utf-8")), ("bom.txt", b"\xef\xbb\xbf" + text.encode("utf-8")), ("ansi.txt", text.encode("cp1252")),
                       ("utf16.txt", text.encode("utf-16"))):
        (tmp_path / name).write_bytes(data)
        assert S._read_text_file(str(tmp_path / name), None) == text, name
    (tmp_path / "carpeta").mkdir()
    (tmp_path / "carpeta" / "ansi.txt").write_bytes(text.encode("cp1252"))
    rows = list(S.load_source({"type": "folder", "path": str(tmp_path / "carpeta")}, S.SourceContext()))
    assert rows and text[:20] in rows[0]["text"]
