"""MMSE-like equalizer: ``Y[N,F], H[N,F] -> equalized_grid[N,F]``."""

import numpy as np

from .interfaces import complex_matrix
from .ofdm_conf import PhyProfile


def equalize(
    frequency_grid: np.ndarray,
    channel_grid: np.ndarray,
    profile: PhyProfile,
) -> np.ndarray:
    received = complex_matrix(
        frequency_grid,
        "frequency_grid",
        rows=profile.config.fft_len,
        columns=profile.num_ofdm_symbols,
    )
    channel = complex_matrix(
        channel_grid,
        "channel_grid",
        rows=profile.config.fft_len,
        columns=profile.num_ofdm_symbols,
    )
    output = np.zeros_like(received, dtype=np.complex128)
    effective = profile.plan.effective_bins
    effective_index = {int(bin_index): index for index, bin_index in enumerate(effective)}
    for symbol_index in range(profile.num_ofdm_symbols):
        h_effective = channel[effective, symbol_index]
        magnitude_squared = np.abs(h_effective) ** 2
        nonzero = magnitude_squared[magnitude_squared > 0]
        floor = max(1e-3 * float(np.median(nonzero)), 1e-8) if nonzero.size else 1e-8
        equalizer = np.conj(h_effective) / (magnitude_squared + floor)
        equalized = equalizer * received[effective, symbol_index]

        pilot_bins = profile.plan.pilot_bins_by_symbol[symbol_index]
        if pilot_bins.size:
            shifted = np.asarray(
                [effective_index[int(value)] for value in pilot_bins], dtype=np.int32
            )
            equalized_pilots = equalized[shifted]
            weights = np.abs(h_effective[shifted]) ** 2
            if float(np.sum(weights)) <= 1e-12:
                cpe = np.angle(np.sum(equalized_pilots * np.conj(profile.pilot_symbol)))
            else:
                cpe = np.angle(
                    np.sum(weights * equalized_pilots * np.conj(profile.pilot_symbol))
                )
            equalized *= np.exp(-1j * cpe)
        output[effective, symbol_index] = equalized
    return complex_matrix(output, "equalized_grid", rows=received.shape[0], columns=received.shape[1])
