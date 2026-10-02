"""The arithmetic of model merging on numpy arrays, one tensor at a time. Pure numpy: ``workers/merge_models.py`` loads this file by
path and streams the tensors of the shards through it; the tests run it on small arrays.

Methods (every function returns float32; the caller casts back to the model's dtype):

* ``linear``: weighted average of the models.
* ``slerp``: spherical interpolation between two models at ``t`` (falls back to linear interpolation when the tensors are almost
  parallel or one of them is zero).
* ``ties``: task vectors against a base, trimmed to the largest ``density`` fraction, a sign elected per element, the vectors
  that agree averaged (disjoint mean), scaled by ``lam`` and added to the base.
* ``dare``: task vectors with a random fraction ``1 - density`` of the entries dropped and the rest rescaled by ``1 / density``
  (seeded per tensor so a merge can be repeated), combined with ``linear`` or ``ties`` consensus.
"""

from __future__ import annotations

import zlib
from typing import Optional, Sequence

import numpy as np

METHODS = ("linear", "slerp", "ties", "dare")


def normalize_weights(weights: Optional[Sequence[float]], n: int) -> list[float]:
    """Positive weights that sum to 1 (equal weights when none are given)."""
    if not weights:
        return [1.0 / n] * n
    if len(weights) != n:
        raise ValueError(f"{len(weights)} weights for {n} models")
    values = [float(w) for w in weights]
    if any(w < 0 for w in values) or sum(values) <= 0:
        raise ValueError("weights must be non-negative and not all zero")
    total = sum(values)
    return [w / total for w in values]


def linear(arrays: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None) -> np.ndarray:
    w = normalize_weights(weights, len(arrays))
    out = np.zeros(arrays[0].shape, dtype=np.float32)
    for weight, array in zip(w, arrays):
        out += np.float32(weight) * array.astype(np.float32, copy=False)
    return out


def slerp(a: np.ndarray, b: np.ndarray, t: float, eps: float = 1e-8, parallel: float = 0.9995) -> np.ndarray:
    """Spherical interpolation of two tensors treated as flat vectors: ``t`` = 0 gives ``a``, 1 gives ``b``."""
    a32, b32 = a.astype(np.float32, copy=False), b.astype(np.float32, copy=False)
    fa, fb = a32.reshape(-1).astype(np.float64), b32.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(fa), np.linalg.norm(fb)
    if na < eps or nb < eps:
        return ((1.0 - t) * a32 + t * b32).astype(np.float32)
    cos = float(np.clip(np.dot(fa / na, fb / nb), -1.0, 1.0))
    if abs(cos) > parallel:
        return ((1.0 - t) * a32 + t * b32).astype(np.float32)
    omega = np.arccos(cos)
    so = np.sin(omega)
    out = (np.sin((1.0 - t) * omega) / so) * a32.astype(np.float64) + (np.sin(t * omega) / so) * b32.astype(np.float64)
    return out.astype(np.float32)


def trim_to_density(delta: np.ndarray, density: float) -> np.ndarray:
    """Keep the ``density`` fraction of entries with the largest magnitude, zero the rest."""
    if density >= 1.0:
        return delta
    flat = np.abs(delta).reshape(-1)
    keep = int(np.ceil(density * flat.size))
    if keep <= 0:
        return np.zeros_like(delta)
    if keep >= flat.size:
        return delta
    threshold = np.partition(flat, flat.size - keep)[flat.size - keep]
    return np.where(np.abs(delta) >= threshold, delta, np.float32(0))


def random_drop(delta: np.ndarray, density: float, rng: np.random.Generator, rescale: bool = True) -> np.ndarray:
    """DARE: keep each entry with probability ``density`` and divide the survivors by ``density``."""
    if density >= 1.0:
        return delta
    mask = rng.random(delta.shape) < density
    out = np.where(mask, delta, np.float32(0))
    return (out / np.float32(density)).astype(np.float32) if rescale else out.astype(np.float32)


def elect_and_merge(deltas: Sequence[np.ndarray], weights: Sequence[float], normalize: bool = True) -> np.ndarray:
    """TIES consensus: the elected sign of an element is the sign of the weighted mass; only deltas with that sign contribute,
    averaged over their weights (disjoint mean)."""
    w = [np.float32(x) for x in weights]
    stacked = np.stack([d * wi for d, wi in zip(deltas, w)])
    sign = np.sign(stacked.sum(axis=0))
    agree = np.sign(np.stack(list(deltas))) == sign
    agree &= sign != 0
    contribution = np.where(agree, stacked, np.float32(0)).sum(axis=0)
    if not normalize:
        return contribution.astype(np.float32)
    denom = np.zeros(contribution.shape, dtype=np.float32)
    for wi, ok in zip(w, agree):
        denom += wi * ok
    return np.where(denom > 0, contribution / np.where(denom > 0, denom, np.float32(1)), np.float32(0)).astype(np.float32)


def ties(base: np.ndarray, models: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None, density: float = 0.5,
         lam: float = 1.0, normalize: bool = True) -> np.ndarray:
    b = base.astype(np.float32, copy=False)
    w = [1.0] * len(models) if not weights else [float(x) for x in weights]
    deltas = [trim_to_density(m.astype(np.float32, copy=False) - b, density) for m in models]
    return (b + np.float32(lam) * elect_and_merge(deltas, w, normalize)).astype(np.float32)


def tensor_seed(name: str, seed: int) -> int:
    return (zlib.crc32(name.encode("utf-8")) ^ (int(seed) * 2654435761)) & 0xFFFFFFFF


def dare(base: np.ndarray, models: Sequence[np.ndarray], weights: Optional[Sequence[float]] = None, density: float = 0.5,
         lam: float = 1.0, seed: int = 0, name: str = "", consensus: str = "linear", normalize: bool = True) -> np.ndarray:
    b = base.astype(np.float32, copy=False)
    rng = np.random.default_rng(tensor_seed(name, seed))
    deltas = [random_drop(m.astype(np.float32, copy=False) - b, density, rng) for m in models]
    if consensus == "ties":
        w = [1.0] * len(models) if not weights else [float(x) for x in weights]
        merged = elect_and_merge(deltas, w, normalize)
    else:
        w = normalize_weights(weights, len(models)) if normalize else (list(weights) if weights else [1.0] * len(models))
        merged = np.zeros(b.shape, dtype=np.float32)
        for wi, delta in zip(w, deltas):
            merged += np.float32(wi) * delta
    return (b + np.float32(lam) * merged).astype(np.float32)


def merge_tensor(method: str, arrays: Sequence[np.ndarray], *, base: Optional[np.ndarray] = None, weights: Optional[Sequence[float]] = None,
                 t: float = 0.5, density: float = 0.5, lam: float = 1.0, seed: int = 0, name: str = "",
                 consensus: str = "linear", normalize: bool = True) -> np.ndarray:
    """One entry point for the worker: ``arrays`` are the models' versions of the same tensor, in the order given."""
    if method == "linear":
        return linear(arrays, weights)
    if method == "slerp":
        if len(arrays) != 2:
            raise ValueError("slerp merges exactly two models")
        return slerp(arrays[0], arrays[1], t)
    if method == "ties":
        if base is None:
            raise ValueError("ties needs a base model")
        return ties(base, arrays, weights, density, lam, normalize)
    if method == "dare":
        if base is None:
            raise ValueError("dare needs a base model")
        return dare(base, arrays, weights, density, lam, seed, name, consensus, normalize)
    raise ValueError(f"unknown merge method {method!r}; use one of {', '.join(METHODS)}")
