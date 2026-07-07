"""Preamble generation and detection helpers."""

from __future__ import annotations

import math

import numpy as np


def zadoff_chu_seq(length: int, root: int) -> np.ndarray:
    if length <= 1:
        raise ValueError("length must be greater than 1")
    if math.gcd(length, root) != 1:
        raise ValueError("root must be coprime with length for a Zadoff-Chu sequence")

    n = np.arange(length, dtype=np.float64)
    if length % 2 == 0:
        phase = -np.pi * root * (n**2) / length
    else:
        phase = -np.pi * root * n * (n + 1) / length
    return np.exp(1j * phase).astype(np.complex128)


def detect_preamble_start(samples: np.ndarray, preamble: np.ndarray) -> tuple[int, np.ndarray]:
    samples = np.asarray(samples, dtype=np.complex128).reshape(-1)
    preamble = np.asarray(preamble, dtype=np.complex128).reshape(-1)
    if samples.size < preamble.size:
        raise ValueError("sample buffer is shorter than the preamble")
    matched = np.convolve(samples, np.conj(preamble[::-1]), mode="valid")
    metric = np.abs(matched)
    start = int(np.argmax(metric))
    return start, metric
