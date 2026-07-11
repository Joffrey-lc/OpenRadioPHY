"""CP fine timing: ``payload_with_margin -> aligned_payload[(N+C)F]``."""

import numpy as np

from .interfaces import complex_vector
from .ofdm_conf import PhyProfile


def align_payload(
    payload_with_margin: np.ndarray,
    profile: PhyProfile,
    *,
    symbols_to_use: int = 4,
) -> tuple[np.ndarray, int]:
    payload = complex_vector(payload_with_margin, "payload_with_margin")
    cfg = profile.config
    if cfg.cp_len == 0:
        return payload[: profile.payload_samples].copy(), 0
    best_offset, best_metric = 0, -1.0
    use_symbols = min(profile.num_ofdm_symbols, symbols_to_use)
    for offset in range(cfg.cp_len):
        metric_sum, valid = 0.0, 0
        for symbol_index in range(use_symbols):
            start = offset + symbol_index * profile.symbol_span
            end = start + profile.symbol_span
            if end > payload.size:
                break
            symbol = payload[start:end]
            cp = symbol[: cfg.cp_len]
            tail = symbol[cfg.fft_len : cfg.fft_len + cfg.cp_len]
            denominator = float(np.linalg.norm(cp) * np.linalg.norm(tail))
            if denominator <= 1e-12:
                continue
            metric_sum += abs(np.vdot(tail, cp)) / denominator
            valid += 1
        if valid and metric_sum / valid > best_metric:
            best_metric = metric_sum / valid
            best_offset = offset
    aligned = payload[best_offset : best_offset + profile.payload_samples]
    if aligned.size < profile.payload_samples:
        aligned = np.pad(
            aligned, (0, profile.payload_samples - aligned.size), constant_values=0.0
        )
    return complex_vector(aligned, "aligned_payload"), best_offset
