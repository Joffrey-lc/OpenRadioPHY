"""NumPy array validation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ofdm_conf import PhyProfile


def bit_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [B]")
    if np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only binary values")
    return array


def complex_vector(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 1:
        raise ValueError(f"{name} must have shape [S]")
    return array


def complex_matrix(
    value: np.ndarray,
    name: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional complex matrix")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} must have {rows} rows, got {array.shape}")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} must have {columns} columns, got {array.shape}")
    return array


@dataclass(frozen=True)
class TxFrameTrace:
    """Transmitter intermediate values."""

    profile: PhyProfile
    protocol_bits: np.ndarray       # [B_protocol], uint8
    coded_bits: np.ndarray          # [B], uint8
    data_symbols: np.ndarray        # [Q], complex128
    resource_grid: np.ndarray       # [N, F], complex128
    time_symbols: np.ndarray        # [N, F], complex128
    symbols_with_cp: np.ndarray     # [N+C, F], complex128
    payload_stream: np.ndarray      # [(N+C)F], complex128
    waveform: np.ndarray            # [ZC+(N+C)F], complex128


@dataclass(frozen=True)
class RxFrameTrace:
    """Receiver intermediate values."""

    profile: PhyProfile
    sync_index: int
    correlation_metric: np.ndarray
    timing_offset: int
    cfo_radians_per_sample: float
    symbols_with_cp: np.ndarray     # [N+C, F], complex128
    time_symbols: np.ndarray        # [N, F], complex128
    frequency_grid: np.ndarray      # [N, F], complex128
    channel_grid: np.ndarray        # [N, F], complex128
    equalized_grid: np.ndarray      # [N, F], complex128
    data_symbols: np.ndarray        # [Q], complex128
    coded_bits: np.ndarray          # [B], uint8
    protocol_bits: np.ndarray       # [B], uint8
