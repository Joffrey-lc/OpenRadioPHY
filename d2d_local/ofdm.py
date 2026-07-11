"""Backward-compatible OFDM API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .channel import apply_channel
from .interfaces import RxFrameTrace
from .ofdm_conf import (
    OfdmConfig,
    PhyProfile,
    SubcarrierPlan,
    build_phy_profile,
    build_subcarrier_plan,
)
from .phy_chain import OfdmReceiver, OfdmTransmitter


@dataclass(frozen=True)
class TxWaveform:
    waveform: np.ndarray
    preamble: np.ndarray
    pilot_symbol: complex
    plan: SubcarrierPlan
    protocol_bits: np.ndarray
    padded_bits: np.ndarray
    num_ofdm_symbols: int
    data_symbols_count: int
    profile: PhyProfile


@dataclass(frozen=True)
class RxRecovery:
    sync_index: int
    recovered_bits: np.ndarray
    equalized_symbols: np.ndarray
    pilot_channel_estimates: np.ndarray
    correlation_metric: np.ndarray
    trace: RxFrameTrace | None = None


_DEFAULT_TRANSMITTER = OfdmTransmitter()
_DEFAULT_RECEIVER = OfdmReceiver()


def encode_frame(protocol_bits: np.ndarray, profile: PhyProfile) -> np.ndarray:
    return _DEFAULT_TRANSMITTER(protocol_bits, profile).waveform.astype(np.complex128)


def decode_frame(
    received: np.ndarray,
    profile: PhyProfile,
    *,
    sync_search: bool = True,
) -> RxRecovery:
    trace = _DEFAULT_RECEIVER(received, profile, sync_search=sync_search)
    return RxRecovery(
        sync_index=trace.sync_index,
        recovered_bits=trace.protocol_bits,
        equalized_symbols=trace.data_symbols,
        pilot_channel_estimates=trace.channel_grid,
        correlation_metric=trace.correlation_metric,
        trace=trace,
    )


def iter_tx_frames(
    protocol_bits: np.ndarray,
    profile: PhyProfile,
    *,
    bits_per_frame: int,
) -> Iterator[np.ndarray]:
    bits = np.asarray(protocol_bits, dtype=np.uint8).reshape(-1)
    if bits_per_frame <= 0 or bits_per_frame > profile.padded_bit_count:
        raise ValueError("invalid bits_per_frame")
    for start in range(0, bits.size, bits_per_frame):
        yield encode_frame(bits[start : start + bits_per_frame], profile)


def build_tx_waveform(protocol_bits: np.ndarray, config: OfdmConfig) -> TxWaveform:
    bits = np.asarray(protocol_bits, dtype=np.uint8).reshape(-1)
    profile = build_phy_profile(config, bits.size)
    trace = _DEFAULT_TRANSMITTER(bits, profile)
    return TxWaveform(
        waveform=trace.waveform,
        preamble=profile.preamble,
        pilot_symbol=profile.pilot_symbol,
        plan=profile.plan,
        protocol_bits=bits,
        padded_bits=np.pad(
            bits, (0, profile.padded_bit_count - bits.size), constant_values=0
        ),
        num_ofdm_symbols=profile.num_ofdm_symbols,
        data_symbols_count=profile.data_symbols_count,
        profile=profile,
    )


def recover_protocol_bits(
    received: np.ndarray,
    profile_or_tx: PhyProfile | TxWaveform,
    config: OfdmConfig | None = None,
) -> RxRecovery:
    del config
    profile = profile_or_tx.profile if isinstance(profile_or_tx, TxWaveform) else profile_or_tx
    return decode_frame(received, profile)


__all__ = [
    "OfdmConfig",
    "PhyProfile",
    "RxRecovery",
    "SubcarrierPlan",
    "TxWaveform",
    "apply_channel",
    "build_phy_profile",
    "build_subcarrier_plan",
    "build_tx_waveform",
    "decode_frame",
    "encode_frame",
    "iter_tx_frames",
    "recover_protocol_bits",
]
