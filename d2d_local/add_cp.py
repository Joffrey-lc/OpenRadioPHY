"""Cyclic-prefix insertion: ``time_symbols[N,F] -> symbols_with_cp[N+C,F]``."""

import numpy as np

from .interfaces import complex_matrix


def add_cp(time_symbols: np.ndarray, cp_len: int) -> np.ndarray:
    symbols = complex_matrix(time_symbols, "time_symbols")
    if cp_len < 0 or cp_len > symbols.shape[0]:
        raise ValueError("cp_len must be between zero and the IDFT size")
    if cp_len == 0:
        return symbols.copy()
    return complex_matrix(
        np.concatenate([symbols[-cp_len:, :], symbols], axis=0),
        "symbols_with_cp",
    )
