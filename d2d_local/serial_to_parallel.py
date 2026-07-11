"""OFDM deserializer: ``payload[(N+C)F] -> symbols_with_cp[N+C,F]``."""

import numpy as np

from .interfaces import complex_matrix, complex_vector


def deserialize_symbols(payload: np.ndarray, rows: int, columns: int) -> np.ndarray:
    vector = complex_vector(payload, "payload")
    if rows <= 0 or columns <= 0 or vector.size != rows * columns:
        raise ValueError(f"payload must contain rows*columns={rows * columns} samples")
    return complex_matrix(
        vector.reshape(rows, columns, order="F"),
        "symbols_with_cp",
        rows=rows,
        columns=columns,
    )
