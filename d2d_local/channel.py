"""Simulation channel: ``tx_waveform[S] -> received_samples[S']``."""

import math
import json
from pathlib import Path

import numpy as np

from .config import ChannelConfig
from .interfaces import complex_vector


def load_channel_config(path: str | Path) -> ChannelConfig:
    """Load and validate a channel JSON file."""
    raw = json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))
    expected = {
        "snr_db",
        "time_offset",
        "frequency_offset_hz",
        "seed",
        "multipath_taps",
    }
    if not isinstance(raw, dict) or set(raw) != expected:
        raise ValueError(f"channel JSON must contain exactly {sorted(expected)}")
    config = ChannelConfig(
        snr_db=float(raw["snr_db"]),
        time_offset=int(raw["time_offset"]),
        frequency_offset_hz=float(raw["frequency_offset_hz"]),
        seed=int(raw["seed"]),
        multipath_taps=tuple(
            (int(delay), float(real), float(imag))
            for delay, real, imag in raw["multipath_taps"]
        ),
    )
    config.validate()
    return config


def apply_channel(
    tx_waveform: np.ndarray,
    config: ChannelConfig,
    *,
    sample_rate: float,
) -> np.ndarray:
    samples = complex_vector(tx_waveform, "tx_waveform")
    config.validate()
    if sample_rate <= 0.0 or not math.isfinite(sample_rate):
        raise ValueError("sample_rate must be a positive finite number")
    max_delay = max(tap[0] for tap in config.multipath_taps)
    impulse = np.zeros(max_delay + 1, dtype=np.complex128)
    for delay, real, imag in config.multipath_taps:
        impulse[delay] = complex(real, imag)
    impulse /= math.sqrt(float(np.sum(np.abs(impulse) ** 2)))
    received = np.convolve(samples, impulse, mode="full")
    if config.frequency_offset_hz:
        index = np.arange(received.size, dtype=np.float64)
        received *= np.exp(1j * 2.0 * np.pi * config.frequency_offset_hz * index / sample_rate)
    signal_power = float(np.mean(np.abs(received) ** 2)) if received.size else 1.0
    noise_power = signal_power / max(10.0 ** (config.snr_db / 10.0), 1e-12)
    sigma = math.sqrt(noise_power / 2.0)
    rng = np.random.default_rng(config.seed)
    noise = sigma * (
        rng.standard_normal(received.size) + 1j * rng.standard_normal(received.size)
    )
    received = np.pad(
        received + noise, (config.time_offset, 0), constant_values=0.0
    )
    return received.astype(np.complex64)
