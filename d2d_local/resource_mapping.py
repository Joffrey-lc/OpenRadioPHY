"""Pilot/data mapper: ``data_symbols[Q] -> resource_grid[N,F]``."""

import numpy as np

from .interfaces import complex_matrix, complex_vector
from .ofdm_conf import PhyProfile


def map_to_resource_grid(data_symbols: np.ndarray, profile: PhyProfile) -> np.ndarray:
    symbols = complex_vector(data_symbols, "data_symbols")
    if symbols.size != profile.data_symbols_count:
        raise ValueError(
            f"data_symbols must contain {profile.data_symbols_count} values, got {symbols.size}"
        )
    grid = np.zeros(
        (profile.config.fft_len, profile.num_ofdm_symbols), dtype=np.complex128
    )
    cursor = 0
    for symbol_index in range(profile.num_ofdm_symbols):
        pilot_bins = profile.plan.pilot_bins_by_symbol[symbol_index]
        data_bins = profile.plan.data_bins_by_symbol[symbol_index]
        grid[pilot_bins, symbol_index] = profile.pilot_symbol
        grid[data_bins, symbol_index] = symbols[cursor : cursor + data_bins.size]
        cursor += int(data_bins.size)
    return complex_matrix(
        grid,
        "resource_grid",
        rows=profile.config.fft_len,
        columns=profile.num_ofdm_symbols,
    )
