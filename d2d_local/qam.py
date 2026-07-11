"""Gray-coded constellation mapping and hard demapping."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np


def _gray_sequence(bits: int) -> list[tuple[int, ...]]:
    values: list[tuple[int, ...]] = []
    for index in range(1 << bits):
        gray = index ^ (index >> 1)
        label = tuple(int(bit) for bit in f"{gray:0{bits}b}")
        values.append(label)
    return values


@lru_cache(maxsize=None)
def constellation_map(bits_per_symbol: int) -> dict[tuple[int, ...], complex]:
    if bits_per_symbol == 1:
        return {(0,): -1.0 + 0.0j, (1,): 1.0 + 0.0j}

    if bits_per_symbol % 2 != 0:
        raise ValueError("Only BPSK and square QAM are supported in this implementation")

    axis_bits = bits_per_symbol // 2
    side = 1 << axis_bits
    levels = np.arange(-(side - 1), side, 2, dtype=np.float64)
    labels = _gray_sequence(axis_bits)
    level_map = {label: level for label, level in zip(labels, levels)}

    symbols: dict[tuple[int, ...], complex] = {}
    for i_label in labels:
        for q_label in labels:
            bits = tuple(i_label + q_label)
            symbols[bits] = complex(level_map[i_label], level_map[q_label])

    average_power = (2.0 / 3.0) * ((1 << bits_per_symbol) - 1)
    scale = math.sqrt(average_power)
    return {bits: symbol / scale for bits, symbol in symbols.items()}


@lru_cache(maxsize=None)
def ordered_constellation(bits_per_symbol: int) -> tuple[np.ndarray, np.ndarray]:
    mapping = constellation_map(bits_per_symbol)
    labels = sorted(mapping.keys())
    points = np.asarray([mapping[label] for label in labels], dtype=np.complex128)
    bits = np.asarray(labels, dtype=np.uint8)
    return points, bits


def peak_magnitude(bits_per_symbol: int) -> float:
    points, _ = ordered_constellation(bits_per_symbol)
    return float(np.max(np.abs(points)))


def bits_to_symbols(bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % bits_per_symbol != 0:
        raise ValueError("bit count must be divisible by bits_per_symbol")
    mapping = constellation_map(bits_per_symbol)
    groups = bits.reshape(-1, bits_per_symbol)
    return np.asarray([mapping[tuple(group.tolist())] for group in groups], dtype=np.complex128)


def symbols_to_bits_hard(symbols: np.ndarray, bits_per_symbol: int) -> np.ndarray:
    symbols = np.asarray(symbols, dtype=np.complex128).reshape(-1)
    points, labels = ordered_constellation(bits_per_symbol)
    distance = np.abs(symbols[:, None] - points[None, :]) ** 2
    nearest = np.argmin(distance, axis=1)
    return labels[nearest].reshape(-1).astype(np.uint8)
