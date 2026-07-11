"""OFDM serializer: ``symbols_with_cp[N+C,F] -> payload_stream[(N+C)F]``."""

import numpy as np

from .interfaces import complex_matrix, complex_vector


def serialize_symbols(symbols_with_cp: np.ndarray) -> np.ndarray:
    matrix = complex_matrix(symbols_with_cp, "symbols_with_cp")
    return complex_vector(matrix.reshape(-1, order="F"), "payload_stream")
