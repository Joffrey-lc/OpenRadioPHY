"""OFDM DFT: ``time_symbols[N,F] -> frequency_grid[N,F]`` along axis N."""

import numpy as np

from .interfaces import complex_matrix


def dft(time_symbols: np.ndarray) -> np.ndarray:
    symbols = complex_matrix(time_symbols, "time_symbols")
    return complex_matrix(np.fft.fft(symbols, axis=0), "frequency_grid")
