"""Pilot channel estimator: ``frequency_grid[N,F] -> channel_grid[N,F]``."""

import math

import numpy as np

from .interfaces import complex_matrix
from .ofdm_conf import PhyProfile


def _interpolate_with_edge_hold(xp: np.ndarray, fp: np.ndarray, xq: np.ndarray) -> np.ndarray:
    if xp.size == 0:
        return np.zeros(xq.size, dtype=np.complex128)
    if xp.size == 1:
        return np.full(xq.size, fp[0], dtype=np.complex128)
    output = np.interp(xq, xp, fp.real) + 1j * np.interp(xq, xp, fp.imag)
    output[xq < xp[0]] = fp[0]
    output[xq > xp[-1]] = fp[-1]
    return output.astype(np.complex128)


def estimate_channel(
    frequency_grid: np.ndarray,
    profile: PhyProfile,
    *,
    block_len: int | None = None,
) -> np.ndarray:
    grid = complex_matrix(
        frequency_grid,
        "frequency_grid",
        rows=profile.config.fft_len,
        columns=profile.num_ofdm_symbols,
    )
    plan = profile.plan
    effective = plan.effective_bins
    n_effective = effective.size
    channel_grid = np.zeros_like(grid, dtype=np.complex128)
    effective_index = {int(bin_index): index for index, bin_index in enumerate(effective)}
    jumps = np.flatnonzero(np.diff(effective) != 1)
    segment_bounds = np.concatenate(([0], jumps + 1, [n_effective]))

    block_len = block_len or profile.config.equalizer_block_len
    block_len = max(1, min(block_len, profile.num_ofdm_symbols))
    num_blocks = int(math.ceil(profile.num_ofdm_symbols / block_len))
    if num_blocks > 1:
        tail = profile.num_ofdm_symbols - (num_blocks - 1) * block_len
        if tail < max(1, block_len // 2):
            num_blocks -= 1

    for block_index in range(num_blocks):
        start = block_index * block_len
        end = (
            profile.num_ofdm_symbols
            if block_index == num_blocks - 1
            else (block_index + 1) * block_len
        )
        h_sum = np.zeros(n_effective, dtype=np.complex128)
        h_count = np.zeros(n_effective, dtype=np.int32)
        for symbol_index in range(start, end):
            pilot_bins = plan.pilot_bins_by_symbol[symbol_index]
            shifted = np.asarray(
                [effective_index[int(value)] for value in pilot_bins], dtype=np.int32
            )
            h_sum[shifted] += grid[pilot_bins, symbol_index] / profile.pilot_symbol
            h_count[shifted] += 1

        valid = h_count > 0
        if not np.any(valid):
            h_effective = np.ones(n_effective, dtype=np.complex128)
        else:
            h_effective = np.zeros(n_effective, dtype=np.complex128)
            h_effective[valid] = h_sum[valid] / h_count[valid]
            if not np.all(valid):
                known_all = np.flatnonzero(valid)
                for segment_start, segment_end in zip(
                    segment_bounds[:-1], segment_bounds[1:]
                ):
                    segment_start, segment_end = int(segment_start), int(segment_end)
                    segment = slice(segment_start, segment_end)
                    known = np.flatnonzero(valid[segment])
                    if known.size == 0:
                        midpoint = (segment_start + segment_end - 1) / 2.0
                        nearest = known_all[np.argmin(np.abs(known_all - midpoint))]
                        h_effective[segment] = h_effective[nearest]
                    elif known.size < segment_end - segment_start:
                        h_effective[segment] = _interpolate_with_edge_hold(
                            known.astype(np.float64),
                            h_effective[segment][known],
                            np.arange(segment_end - segment_start, dtype=np.float64),
                        )
        for symbol_index in range(start, end):
            channel_grid[effective, symbol_index] = h_effective
    return complex_matrix(channel_grid, "channel_grid", rows=grid.shape[0], columns=grid.shape[1])
