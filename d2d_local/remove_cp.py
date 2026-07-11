"""Cyclic-prefix removal: ``symbols_with_cp[N+C,F] -> time_symbols[N,F]``."""

import numpy as np

from .interfaces import complex_matrix


def remove_cp(symbols_with_cp: np.ndarray, cp_len: int) -> np.ndarray:
    matrix = complex_matrix(symbols_with_cp, "symbols_with_cp")
    if cp_len < 0 or cp_len >= matrix.shape[0]:
        raise ValueError("cp_len must be non-negative and smaller than the row count")
    output = matrix if cp_len == 0 else matrix[cp_len:, :]
    return complex_matrix(output, "time_symbols")
