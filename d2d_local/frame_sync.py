"""Preamble synchronization: ``capture[S] -> aligned frame payload``."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .interfaces import complex_vector
from .ofdm_conf import PhyProfile
from .preamble import detect_preamble_start
from .viz import save_correlation_plot


@dataclass(frozen=True)
class CoarseSyncResult:
    sync_index: int
    correlation_metric: np.ndarray
    payload_with_margin: np.ndarray


def synchronize_frame(
    received: np.ndarray,
    profile: PhyProfile,
    *,
    search: bool,
) -> CoarseSyncResult:
    samples = complex_vector(received, "received")
    if search:
        sync_index, metric = detect_preamble_start(samples, profile.preamble)
    else:
        sync_index = 0
        metric = np.zeros(1, dtype=np.complex128)
    payload_start = sync_index + profile.preamble.size
    required = profile.payload_samples + profile.config.cp_len
    payload = samples[payload_start : payload_start + required]
    if payload.size < required:
        payload = np.pad(payload, (0, required - payload.size), constant_values=0.0)
    return CoarseSyncResult(sync_index, metric, complex_vector(payload, "payload_with_margin"))


def find_frame_starts(
    samples: np.ndarray,
    preamble: np.ndarray,
    frame_samples: int,
    *,
    threshold_factor: float = 0.5,
    median_factor: float = 4.0,
    min_separation: int | None = None,
    correlation_plot_out: str | Path | None = None,
    correlation_plot_title: str | None = None,
) -> list[int]:
    capture = np.asarray(samples, dtype=np.complex64).reshape(-1)
    reference = np.asarray(preamble, dtype=np.complex64).reshape(-1)
    if capture.size < reference.size:
        return []
    matched = np.convolve(capture, np.conj(reference[::-1]), mode="valid")
    metric = np.abs(matched).astype(np.float32)
    peak = float(metric.max()) if metric.size else 0.0
    if peak <= 0.0:
        return []
    threshold = max(threshold_factor * peak, median_factor * float(np.median(metric)))
    step = min_separation if min_separation is not None else max(1, frame_samples - 100)
    starts: list[int] = []
    index = 0
    while index < metric.size:
        if metric[index] >= threshold:
            end = min(index + reference.size, metric.size)
            local = index + int(np.argmax(metric[index:end]))
            starts.append(local)
            index = local + step
        else:
            index += 1
    if correlation_plot_out is not None:
        save_correlation_plot(
            metric,
            correlation_plot_out,
            threshold=threshold,
            starts=starts,
            title=correlation_plot_title,
        )
    return starts
