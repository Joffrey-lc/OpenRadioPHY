"""OFDM IDFT: ``resource_grid[N,F] -> time_symbols[N,F]`` along axis N."""

import numpy as np

from .interfaces import complex_matrix


def idft(resource_grid: np.ndarray) -> np.ndarray:
    grid = complex_matrix(resource_grid, "resource_grid")
    return complex_matrix(np.fft.ifft(grid, axis=0), "time_symbols")
