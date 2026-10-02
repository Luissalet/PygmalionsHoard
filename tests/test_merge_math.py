"""Merge arithmetic on small arrays."""

import numpy as np
import pytest

from pygmalion_hoard import merge_math as M

RNG = np.random.default_rng(0)
A, B, C = (RNG.normal(size=(6, 8)).astype(np.float32) for _ in range(3))


def test_normalize_weights_defaults_to_equal():
    assert M.normalize_weights(None, 4) == [0.25] * 4
    assert M.normalize_weights([1, 3], 2) == [0.25, 0.75]


@pytest.mark.parametrize("bad", [[1], [-1, 2], [0, 0]])
def test_normalize_weights_rejects_bad_input(bad):
    with pytest.raises(ValueError):
        M.normalize_weights(bad, 2)


def test_linear_is_a_weighted_mean():
    out = M.linear([A, B], [1, 3])
    assert np.allclose(out, 0.25 * A + 0.75 * B, atol=1e-6) and out.dtype == np.float32


def test_linear_of_one_model_is_that_model():
    assert np.allclose(M.linear([A]), A)


def test_slerp_endpoints():
    assert np.allclose(M.slerp(A, B, 0.0), A, atol=1e-5)
    assert np.allclose(M.slerp(A, B, 1.0), B, atol=1e-5)


def test_slerp_midpoint_keeps_the_norm_for_equal_norm_vectors():
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    mid = M.slerp(a, b, 0.5)
    assert np.isclose(np.linalg.norm(mid), 1.0, atol=1e-6) and np.allclose(mid, [2 ** -0.5, 2 ** -0.5], atol=1e-6)


def test_slerp_falls_back_to_linear_for_parallel_vectors():
    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(M.slerp(a, a * 2, 0.5), 1.5 * a, atol=1e-6)


def test_slerp_with_a_zero_tensor_is_linear():
    z = np.zeros(4, dtype=np.float32)
    b = np.ones(4, dtype=np.float32)
    assert np.allclose(M.slerp(z, b, 0.25), 0.25 * b)


def test_trim_keeps_the_largest_entries():
    d = np.array([0.1, -5.0, 0.2, 4.0, -0.3, 0.05], dtype=np.float32)
    out = M.trim_to_density(d, 0.33)
    assert (out != 0).sum() == 2 and out[1] == -5.0 and out[3] == 4.0


def test_trim_density_one_is_identity_and_zero_empties():
    d = np.arange(5, dtype=np.float32)
    assert np.array_equal(M.trim_to_density(d, 1.0), d)
    assert not M.trim_to_density(d, 0.0).any()


def test_random_drop_rescales_survivors():
    d = np.ones(10_000, dtype=np.float32)
    out = M.random_drop(d, 0.25, np.random.default_rng(1))
    assert set(np.unique(out)) <= {0.0, 4.0} and abs(out.mean() - 1.0) < 0.1


def test_elect_and_merge_disagreeing_signs_follow_the_larger_mass():
    d1 = np.array([2.0, -1.0], dtype=np.float32)
    d2 = np.array([1.0, 3.0], dtype=np.float32)
    out = M.elect_and_merge([d1, d2], [1.0, 1.0])
    assert np.allclose(out, [1.5, 3.0])


def test_ties_with_full_density_and_one_model_returns_that_model():
    out = M.ties(A, [B], density=1.0)
    assert np.allclose(out, B, atol=1e-5)


def test_ties_result_stays_close_to_base_when_density_is_small():
    out = M.ties(A, [B, C], density=0.1)
    assert np.abs(out - A).mean() < np.abs(B - A).mean()


def test_dare_is_deterministic_per_tensor_name_and_seed():
    one = M.dare(A, [B, C], density=0.5, seed=7, name="layers.0.w")
    two = M.dare(A, [B, C], density=0.5, seed=7, name="layers.0.w")
    other = M.dare(A, [B, C], density=0.5, seed=7, name="layers.1.w")
    assert np.array_equal(one, two) and not np.array_equal(one, other)


def test_dare_with_ties_consensus_runs():
    out = M.dare(A, [B, C], density=0.6, seed=1, name="n", consensus="ties")
    assert out.shape == A.shape and np.isfinite(out).all()


def test_tensor_seed_changes_with_name_and_seed():
    assert M.tensor_seed("a", 1) != M.tensor_seed("b", 1) != M.tensor_seed("b", 2)


def test_merge_tensor_dispatches_each_method():
    for method in M.METHODS:
        arrays = [A, B] if method in ("linear", "slerp") else [B, C]
        out = M.merge_tensor(method, arrays, base=A, name="t")
        assert out.shape == A.shape


@pytest.mark.parametrize("method,kwargs", [("slerp", {"arrays": [A, B, C]}), ("ties", {"arrays": [B]}), ("dare", {"arrays": [B]}), ("nope", {"arrays": [B]})])
def test_merge_tensor_errors(method, kwargs):
    with pytest.raises(ValueError):
        M.merge_tensor(method, kwargs["arrays"])
