"""Data-resource extractor: ``equalized_grid[N,F] -> data_symbols[Q]``."""

import numpy as np

from .interfaces import complex_matrix, complex_vector
from .ofdm_conf import PhyProfile


def extract_data_symbols(equalized_grid: np.ndarray, profile: PhyProfile) -> np.ndarray:
    grid = complex_matrix(
        equalized_grid,
        "equalized_grid",
        rows=profile.config.fft_len,
        columns=profile.num_ofdm_symbols,
    )
    symbols = np.concatenate(
        [
            grid[profile.plan.data_bins_by_symbol[index], index]
            for index in range(profile.num_ofdm_symbols)
        ]
    )[: profile.data_symbols_count]
    return complex_vector(symbols, "data_symbols")
