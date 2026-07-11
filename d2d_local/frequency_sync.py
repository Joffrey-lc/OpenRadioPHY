"""CP carrier-offset correction: ``payload[S] -> corrected_payload[S]``."""

import numpy as np

from .interfaces import complex_vector
from .ofdm_conf import PhyProfile


def correct_frequency_offset(
    aligned_payload: np.ndarray,
    profile: PhyProfile,
) -> tuple[np.ndarray, float]:
    payload = complex_vector(aligned_payload, "aligned_payload")
    if payload.size != profile.payload_samples:
        raise ValueError(f"aligned_payload must have {profile.payload_samples} samples")
    cfg = profile.config
    if cfg.cp_len == 0:
        return payload.copy(), 0.0
    matrix = payload.reshape(profile.symbol_span, profile.num_ofdm_symbols, order="F")
    cp_head = matrix[: cfg.cp_len, :]
    matching_tail = matrix[cfg.fft_len : cfg.fft_len + cfg.cp_len, :]
    correlation = np.sum(np.conj(cp_head) * matching_tail)
    radians_per_sample = (
        float(np.angle(correlation)) / cfg.fft_len if abs(correlation) > 0.0 else 0.0
    )
    sample_index = np.arange(payload.size, dtype=np.float64)
    corrected = payload * np.exp(-1j * radians_per_sample * sample_index)
    return complex_vector(corrected, "frequency_corrected_payload"), radians_per_sample
