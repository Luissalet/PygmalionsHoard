"""GGUF metadata reading, the KV-cache table and the conversion-script architecture detection."""

import struct

import pytest

from gguf_builder import build_gguf
from models import make_model
from pygmalion_hoard import gguf_meta as G
from pygmalion_hoard import convert_detect as D


def test_read_metadata_and_summary(tmp_path):
    p = build_gguf(tmp_path / "m.gguf", arch="llama", context=8192, blocks=32, heads=32, kv_heads=8, embedding=4096, file_type=15, name="Tiny")
    meta = G.read_metadata(p)
    s = G.summarize(meta)
    assert meta["general.architecture"] == "llama"
    assert s["architecture"] == "llama" and s["context_length"] == 8192 and s["block_count"] == 32 and s["head_count_kv"] == 8 and s["file_type"] == "Q4_K_M"
    assert s["has_chat_template"] is True


def test_summary_without_chat_template(tmp_path):
    s = G.summarize(G.read_metadata(build_gguf(tmp_path / "m.gguf", chat_template=False)))
    assert s["has_chat_template"] is False


def test_long_token_arrays_are_truncated(tmp_path):
    p = build_gguf(tmp_path / "m.gguf", tokens=[f"t{i}" for i in range(1000)])
    tokens = G.read_metadata(p)["tokenizer.ggml.tokens"]
    assert len(tokens) <= G.KEPT_ARRAY_ITEMS + 1 or isinstance(tokens, dict)


def test_not_a_gguf_is_rejected(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"NOPE" + b"\0" * 40)
    with pytest.raises(G.GgufError):
        G.read_metadata(tmp_path / "x.gguf")


def test_truncated_gguf_is_rejected(tmp_path):
    p = build_gguf(tmp_path / "m.gguf")
    p.write_bytes(p.read_bytes()[:60])
    with pytest.raises(G.GgufError):
        G.read_metadata(p)


def test_absurd_counts_are_rejected(tmp_path):
    (tmp_path / "x.gguf").write_bytes(b"GGUF" + struct.pack("<IQQ", 3, 0, 10 ** 12))
    with pytest.raises(G.GgufError):
        G.read_metadata(tmp_path / "x.gguf")


def test_kv_cache_bytes_follow_the_formula(tmp_path):
    s = G.summarize(G.read_metadata(build_gguf(tmp_path / "m.gguf", blocks=4, heads=8, kv_heads=2, embedding=256)))
    head_dim = 256 // 8
    assert G.kv_cache_bytes(s, 1000) == 2 * 4 * 2 * head_dim * 1000 * 2


def test_fit_table_marks_what_fits(tmp_path):
    s = G.summarize(G.read_metadata(build_gguf(tmp_path / "m.gguf", blocks=32, heads=32, kv_heads=8, embedding=4096)))
    rows = G.fit_table(s, [4096, 131072, 1_000_000], free_mb=4000)
    assert len(rows) == 3 and rows[0]["fits"] is True and rows[2]["fits"] is False
    assert rows[0]["kv_mb"] < rows[1]["kv_mb"]


def test_max_trained_context_prefers_rope_original(tmp_path):
    s = G.summarize(G.read_metadata(build_gguf(tmp_path / "m.gguf", context=131072, rope={"scaling.type": "yarn", "scaling.factor": 4.0, "scaling.original_context_length": 32768})))
    assert G.max_trained_context(s) == 32768


SCRIPT = '''
@ModelBase.register("LlamaForCausalLM", "MistralForCausalLM")
class LlamaModel(TextModel):
    pass

@ModelBase.register(
    "Qwen3ForCausalLM",
    'Qwen3MoeForCausalLM')
class Q(TextModel):
    pass

@Model.register("OldForCausalLM")
class O(Model):
    pass
'''


def test_registered_architectures_reads_every_decorator(tmp_path):
    p = tmp_path / "convert_hf_to_gguf.py"
    p.write_text(SCRIPT, encoding="utf-8")
    assert D.registered_architectures(p) == {"LlamaForCausalLM", "MistralForCausalLM", "Qwen3ForCausalLM", "Qwen3MoeForCausalLM", "OldForCausalLM"}


NEW_SCRIPT = """
from conversion import (
    ModelBase, TextModel,
)
from conversion import get_model_class


def main():
    pass
"""

QWEN_MODULE = """
from .base import ModelBase, TextModel


@ModelBase.register("Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM")
class Qwen35Model(TextModel):
    pass


@ModelBase.register(
    "Qwen3_5MoeForConditionalGeneration",   # the ")" in this comment must not end the call
    'Qwen3_5MoeForCausalLM',
)
class Qwen35MoeModel(TextModel):
    pass
"""

LLAMA_MODULE = """
@ModelBase.register("LlamaForCausalLM", "MistralForCausalLM", "VLlama3ForCausalLM")
class LlamaModel(TextModel):
    pass
"""


def new_layout(tmp_path):
    src = tmp_path / "src"
    (src / "conversion" / "vision").mkdir(parents=True)
    (src / "convert_hf_to_gguf.py").write_text(NEW_SCRIPT, encoding="utf-8")
    (src / "conversion" / "__init__.py").write_text("from .base import ModelBase\n", encoding="utf-8")
    (src / "conversion" / "qwen.py").write_text(QWEN_MODULE, encoding="utf-8")
    (src / "conversion" / "vision" / "llama.py").write_text(LLAMA_MODULE, encoding="utf-8")
    return src


def test_registrations_in_the_conversion_package_are_found(tmp_path):
    src = new_layout(tmp_path)
    assert D.registered_architectures(src / "convert_hf_to_gguf.py") == {
        "Qwen3_5ForConditionalGeneration", "Qwen3_5ForCausalLM", "Qwen3_5MoeForConditionalGeneration", "Qwen3_5MoeForCausalLM",
        "LlamaForCausalLM", "MistralForCausalLM", "VLlama3ForCausalLM"}


def test_the_old_single_file_layout_still_works_and_both_layouts_add_up(tmp_path):
    src = new_layout(tmp_path)
    (src / "convert_hf_to_gguf.py").write_text(SCRIPT + NEW_SCRIPT, encoding="utf-8")
    names = D.registered_architectures(src / "convert_hf_to_gguf.py")
    assert {"OldForCausalLM", "Qwen3ForCausalLM", "Qwen3_5ForCausalLM", "LlamaForCausalLM"} <= names


def test_a_script_that_imports_the_package_but_has_none_next_to_it_lists_nothing(tmp_path):
    p = tmp_path / "convert_hf_to_gguf.py"
    p.write_text(NEW_SCRIPT, encoding="utf-8")
    assert D.registered_architectures(p) == frozenset()
    result = D.convertible(make_model(tmp_path / "m", arch="LlamaForCausalLM"), p)
    assert result["known"] is None


def test_convertible_with_the_conversion_package(tmp_path):
    src = new_layout(tmp_path)
    script = src / "convert_hf_to_gguf.py"
    qwen = make_model(tmp_path / "q", arch="Qwen3_5ForConditionalGeneration")
    other = make_model(tmp_path / "o", arch="NeverHeardForCausalLM")
    assert D.convertible(qwen, script)["known"] is True
    assert D.convertible(other, script)["known"] is False and "NeverHeardForCausalLM" in D.convertible(other, script)["note"]


def test_a_change_inside_the_package_is_picked_up(tmp_path):
    src = new_layout(tmp_path)
    script = src / "convert_hf_to_gguf.py"
    assert "NewArchForCausalLM" not in D.registered_architectures(script)
    (src / "conversion" / "new.py").write_text('@ModelBase.register("NewArchForCausalLM")\nclass N: pass\n', encoding="utf-8")
    assert "NewArchForCausalLM" in D.registered_architectures(script)


def test_registered_architectures_of_a_missing_script_is_empty(tmp_path):
    assert D.registered_architectures(tmp_path / "nope.py") == frozenset()


def test_convertible_answers(tmp_path):
    p = tmp_path / "convert_hf_to_gguf.py"
    p.write_text(SCRIPT, encoding="utf-8")
    known = make_model(tmp_path / "a", arch="LlamaForCausalLM")
    unknown = make_model(tmp_path / "b", arch="ZzzForCausalLM")
    assert D.convertible(known, p)["known"] is True
    r = D.convertible(unknown, p)
    assert r["known"] is False and "ZzzForCausalLM" in r["note"]
    assert D.convertible(known, None)["known"] is None
    empty = tmp_path / "empty.py"
    empty.write_text("x = 1", encoding="utf-8")
    assert D.convertible(known, empty)["known"] is None


def test_architecture_from_a_nested_text_config(tmp_path):
    folder = tmp_path / "m"
    folder.mkdir()
    (folder / "config.json").write_text('{"text_config": {"architectures": ["Gemma3ForCausalLM"]}}', encoding="utf-8")
    assert D.model_architectures(folder) == ["Gemma3ForCausalLM"]
