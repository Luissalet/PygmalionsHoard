"""Small pure helpers."""

import json

import pytest

from pygmalion_hoard import util
from pygmalion_hoard.errors import PygmalionError


def test_new_id_is_a_lowercase_ulid_that_sorts_by_creation_time():
    ids = [util.new_id("j") for _ in range(200)]
    assert all(i.startswith("j_") and len(i) == 28 and i == i.lower() for i in ids)
    assert ids == sorted(ids) and len(set(ids)) == 200        # strictly increasing inside the process, even inside one millisecond


def test_new_ids_are_unique():
    assert len({util.new_id("x") for _ in range(500)}) == 500


def test_old_ids_still_resolve(ctx):
    """The ids of the earlier format (``a_`` + 12 characters) are plain text keys: every lookup by id keeps working next to the new ones."""
    store = ctx.svc.store
    old = store.create_artifact("adapter", "viejo", artifact_id="a_0abcdefghjkm")
    new = store.create_artifact("adapter", "nuevo")
    assert store.artifact("a_0abcdefghjkm")["name"] == "viejo" and store.artifact(new["id"])["name"] == "nuevo" and old["id"] != new["id"]
    assert len(new["id"]) == 28


def test_fold_drops_accents_and_case():
    assert util.fold("ÁRBOL Ñandú") == "arbol nandu"


@pytest.mark.parametrize("text,expected", [("Casa de la abuela", "casa-de-la-abuela"), ("  Ñu!!  ", "nu"), ("", "model"), ("***", "model")])
def test_slug(text, expected):
    assert util.slug(text) == expected


def test_slug_limit_never_ends_with_hyphen():
    out = util.slug("a" * 10 + " " + "b" * 10, limit=11)
    assert not out.endswith("-") and len(out) <= 11


def test_est_tokens_is_monotonic():
    assert util.est_tokens(0) == 0 and util.est_tokens(3600) == 1000 and util.est_tokens(7200) > util.est_tokens(3600)


@pytest.mark.parametrize("n,expected", [(0, "0 B"), (1023, "1023 B"), (1024, "1.0 KB"), (5 * 1024 ** 3, "5.0 GB")])
def test_human_bytes(n, expected):
    assert util.human_bytes(n) == expected


def test_clamp_text_collapses_whitespace_and_cuts():
    assert util.clamp_text("a   b\n c", 20) == "a b c"
    assert util.clamp_text("x" * 50, 10).endswith("…") and len(util.clamp_text("x" * 50, 10)) == 10


def test_sha256_text_and_file_agree(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hola", encoding="utf-8")
    assert util.sha256_file(p) == util.sha256_text("hola")


def test_loads_dumps_roundtrip_and_default():
    assert util.loads(util.dumps({"a": [1, 2]}), None) == {"a": [1, 2]}
    assert util.loads("not json", {"d": 1}) == {"d": 1}
    assert util.loads(None, 7) == 7


def test_write_json_atomic_leaves_no_temp_file(tmp_path):
    p = tmp_path / "sub" / "x.json"
    util.write_json_atomic(p, {"k": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"k": 1}
    assert [c.name for c in p.parent.iterdir()] == ["x.json"]


def test_read_json_default_on_missing_or_broken(tmp_path):
    assert util.read_json(tmp_path / "none.json", "d") == "d"
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert util.read_json(tmp_path / "bad.json", []) == []


def test_dir_size_counts_nested_files(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x").write_bytes(b"1" * 10)
    (tmp_path / "y").write_bytes(b"1" * 5)
    assert util.dir_size(tmp_path) == 15


def test_tail_lines_returns_the_last_ones(tmp_path):
    p = tmp_path / "log"
    p.write_text("\n".join(f"line {i}" for i in range(100)), encoding="utf-8")
    assert util.tail_lines(p, 3) == ["line 97", "line 98", "line 99"]
    assert util.tail_lines(tmp_path / "missing", 3) == []


def test_repo_dirname_is_filesystem_safe():
    assert util.repo_dirname("acme/tiny-base") == "acme--tiny-base"


def test_error_to_dict_and_status():
    e = PygmalionError("not_found", "No such thing.", "Look again.", extra=1)
    body = e.to_dict()
    assert body["code"] == "not_found" and body["hint"] == "Look again." and body["extra"] == 1 and e.status == 404
    assert PygmalionError("offline", "x").status == 503
