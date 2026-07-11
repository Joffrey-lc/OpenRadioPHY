"""Bit scrambling and block interleaving."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def _scrambler_mask(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 2, size=length, dtype=np.uint8)


def scramble_bits(bits: np.ndarray, *, enabled: bool, seed: int) -> np.ndarray:
    """Apply the seeded XOR scrambler."""
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if not enabled or bits.size == 0:
        return bits.copy()
    mask = _scrambler_mask(bits.size, seed)
    return np.bitwise_xor(bits, mask)


@lru_cache(maxsize=None)
def _block_interleaver_permutation(length: int, rows: int) -> tuple[np.ndarray, np.ndarray]:
    if length <= 0:
        raise ValueError("length must be positive")
    if rows <= 0:
        raise ValueError("rows must be positive")

    cols = int(math.ceil(length / rows))
    index_grid = np.full((rows, cols), -1, dtype=np.int64)
    index_grid.flat[:length] = np.arange(length, dtype=np.int64)
    perm = index_grid.T.reshape(-1)
    perm = perm[perm >= 0].astype(np.int64, copy=False)

    inv = np.empty_like(perm)
    inv[perm] = np.arange(length, dtype=np.int64)
    return perm, inv


def interleave_bits(bits: np.ndarray, mode: str, *, rows: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if mode == "off":
        return bits.copy()
    if mode != "block":
        raise ValueError(f"unsupported interleaver mode: {mode}")
    perm, _ = _block_interleaver_permutation(bits.size, rows)
    return bits[perm]


def deinterleave_bits(bits: np.ndarray, mode: str, *, rows: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if mode == "off":
        return bits.copy()
    if mode != "block":
        raise ValueError(f"unsupported interleaver mode: {mode}")
    _, inv = _block_interleaver_permutation(bits.size, rows)
    return bits[inv]
