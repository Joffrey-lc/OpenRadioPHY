"""Optional smooth magnitude limiter: ``samples[S] -> samples[S]``."""

import numpy as np

from .interfaces import complex_vector


KNEE_RATIO = 0.85


def soft_clip(samples: np.ndarray, peak_clip: float | None) -> np.ndarray:
    vector = complex_vector(samples, "samples")
    if peak_clip is None:
        return vector
    if peak_clip <= 0:
        raise ValueError("peak_clip must be positive")
    magnitude = np.abs(vector)
    knee = KNEE_RATIO * peak_clip
    over = magnitude > knee
    if not np.any(over):
        return vector
    headroom = peak_clip - knee
    limited = knee + headroom * np.tanh((magnitude[over] - knee) / headroom)
    scale = np.ones_like(magnitude, dtype=np.float64)
    scale[over] = limited / magnitude[over]
    return vector * scale
