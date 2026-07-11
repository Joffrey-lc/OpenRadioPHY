"""Preamble/frame assembly: ``preamble[Z], payload[S] -> waveform[Z+S]``."""

import numpy as np

from .clipping import soft_clip
from .interfaces import complex_vector


def assemble_frame(
    preamble: np.ndarray,
    payload_stream: np.ndarray,
    *,
    normalize: bool,
    peak_clip: float | None,
) -> np.ndarray:
    preamble_out = complex_vector(preamble, "preamble").copy()
    payload = complex_vector(payload_stream, "payload_stream")
    if normalize:
        payload_peak = float(np.max(np.abs(payload))) if payload.size else 0.0
        preamble_peak = float(np.max(np.abs(preamble_out))) if preamble_out.size else 0.0
        if payload_peak > 0.0 and preamble_peak > 0.0:
            preamble_out *= payload_peak / preamble_peak
    waveform = np.concatenate([preamble_out, payload])
    if normalize and waveform.size:
        peak = float(np.max(np.abs(waveform)))
        if peak > 0.0:
            waveform /= peak
    return complex_vector(soft_clip(waveform, peak_clip), "waveform")
