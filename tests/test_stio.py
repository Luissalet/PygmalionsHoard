"""The streaming safetensors reader and writer, checked against the real safetensors package."""

import json
import struct
from pathlib import Path

import numpy as np
import pytest

import models  # noqa: F401  (puts the workers folder on sys.path)
import _stio as S
from models import make_model, read_all, tensor_names


def test_our_writer_is_read_by_the_reference_library(tmp_path):
    from safetensors.numpy import load_file
    arrays = {"a": np.arange(6, dtype="<f4").reshape(2, 3), "b": np.ones((4,), dtype="<f4")}
    S.write_shard(tmp_path / "x.safetensors", [(n, "F32", list(a.shape)) for n, a in arrays.items()], lambda n: S.from_float32(arrays[n], "F32"))
    got = load_file(str(tmp_path / "x.safetensors"))
    assert all(np.array_equal(got[k], arrays[k]) for k in arrays)


def test_the_reference_writer_is_read_by_our_reader(tmp_path):
    from safetensors.numpy import save_file
    arrays = {"w": np.random.default_rng(0).standard_normal((5, 7)).astype("<f4"), "v": np.arange(3, dtype="<f4")}
    save_file(arrays, str(tmp_path / "model.safetensors"))
    tensors = S.model_tensors(tmp_path)
    assert set(tensors) == {"w", "v"}
    assert np.array_equal(S.read_tensor(tensors["w"]), arrays["w"])


@pytest.mark.parametrize("dtype,tol", [("F32", 0), ("F16", 1e-2), ("BF16", 1e-1), ("F64", 0)])
def test_dtype_roundtrip(dtype, tol):
    a = np.random.default_rng(1).standard_normal((3, 5)).astype(np.float32)
    back = S.to_float32(S.from_float32(a, dtype), dtype, [3, 5])
    assert np.allclose(a, back, atol=tol, rtol=tol)


def test_bf16_matches_the_reference_rounding():
    import struct as st
    values = np.array([1.0, 3.14159265, -2.71828, 1e-3, 65504.0], dtype=np.float32)
    ours = np.frombuffer(S.from_float32(values, "BF16"), dtype="<u2")
    for v, bits in zip(values, ours):
        ref = np.uint32(st.unpack("<I", st.pack("<f", float(v)))[0])
        expected = np.uint16(((ref + np.uint32(0x7FFF) + ((ref >> 16) & 1)) >> 16) & 0xFFFF)
        assert bits == expected


def test_bf16_keeps_infinity_and_nan():
    back = S.to_float32(S.from_float32(np.array([np.inf, -np.inf, np.nan], dtype=np.float32), "BF16"), "BF16", [3])
    assert back[0] == np.inf and back[1] == -np.inf and np.isnan(back[2])


def test_unsupported_dtype_raises():
    with pytest.raises(S.StioError):
        S.to_float32(b"\0", "I8", [1])
    with pytest.raises(S.StioError):
        S.from_float32(np.zeros(1), "I8")


def test_header_is_eight_byte_aligned(tmp_path):
    S.write_shard(tmp_path / "x.safetensors", [("t", "F32", [3])], lambda n: b"\0" * 12)
    (size,) = struct.unpack("<Q", (tmp_path / "x.safetensors").read_bytes()[:8])
    assert (8 + size) % 8 == 0


def test_wrong_length_from_produce_is_refused(tmp_path):
    with pytest.raises(S.StioError):
        S.write_shard(tmp_path / "x.safetensors", [("t", "F32", [3])], lambda n: b"\0")
    assert not (tmp_path / "x.safetensors").exists()


def test_truncated_and_garbage_files_are_refused(tmp_path):
    (tmp_path / "a.safetensors").write_bytes(b"123")
    with pytest.raises(S.StioError):
        S.read_header(tmp_path / "a.safetensors")
    (tmp_path / "b.safetensors").write_bytes(struct.pack("<Q", 10 ** 12) + b"{}")
    with pytest.raises(S.StioError):
        S.read_header(tmp_path / "b.safetensors")
    with pytest.raises(S.StioError):
        S.model_tensors(tmp_path / "empty")


def test_truncated_tensor_data_is_detected(tmp_path):
    p = tmp_path / "x.safetensors"
    S.write_shard(p, [("t", "F32", [4])], lambda n: b"\1" * 16)
    p.write_bytes(p.read_bytes()[:-4])
    entry = S.model_tensors(tmp_path)["t"]
    with pytest.raises(S.StioError):
        S.read_tensor(entry)


def test_plan_shards_respects_the_limit_and_oversize_tensors():
    entries = [("a", "F32", [10]), ("b", "F32", [10]), ("c", "F32", [100]), ("d", "F32", [1])]
    shards = S.plan_shards(entries, 100)
    assert [[e[0] for e in s] for s in shards] == [["a", "b"], ["c"], ["d"]]


def test_sharded_model_is_read_in_index_order(tmp_path):
    folder = make_model(tmp_path / "m", layers=3, shard_bytes=600)
    assert (folder / "model.safetensors.index.json").is_file()
    assert len(S.weight_files(folder)) > 1
    assert set(S.model_tensors(folder)) == {n for n, _ in tensor_names(3)}


def test_check_compatible_accepts_equal_architectures(tmp_path):
    a, b = make_model(tmp_path / "a", seed=1), make_model(tmp_path / "b", seed=2)
    report = S.check_compatible([a, b])
    assert report["ok"] and report["problems"] == [] and report["tensors"] == len(tensor_names(2))


def test_check_compatible_reports_missing_tensors_and_shapes(tmp_path):
    a, b = make_model(tmp_path / "a", layers=2), make_model(tmp_path / "b", layers=3)
    report = S.check_compatible([a, b])
    assert not report["ok"] and any("has" in p for p in report["problems"])


def test_check_compatible_reports_dtype_differences_without_failing(tmp_path):
    a, b = make_model(tmp_path / "a"), make_model(tmp_path / "b", dtype="F16")
    report = S.check_compatible([a, b])
    assert report["ok"] and report["dtype_differences"]


def test_copy_non_weights_skips_weights(tmp_path):
    src = make_model(tmp_path / "src")
    (src / "extra.bin").write_bytes(b"x")
    (src / "chat_template.jinja").write_text("t", encoding="utf-8")
    dst = tmp_path / "dst"
    dst.mkdir()
    copied = S.copy_non_weights(src, dst)
    assert "config.json" in copied and "chat_template.jinja" in copied and "extra.bin" not in copied and "model.safetensors" not in copied


def test_read_all_returns_float32(tmp_path):
    arrays = read_all(make_model(tmp_path / "m", dtype="BF16"))
    assert all(a.dtype == np.float32 for a in arrays.values())
