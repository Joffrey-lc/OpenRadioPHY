"""OFDM configuration and frame layout."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .preamble import zadoff_chu_seq
from .qam import peak_magnitude


@dataclass(frozen=True)
class OfdmConfig:
    fft_len: int = 256
    cp_len: int = 64
    guard_carriers: int = 16
    dc_carriers: int = 4
    pilot_symbols: int = 16
    pilot_hop: int = 3
    bits_per_symbol: int = 2
    zc_length: int = 1023
    zc_root: int = 1
    sample_rate: float = 5e6
    normalize_waveform: bool = True
    payload_peak_clip: float | None = None
    interleaver: str = "block"
    interleaver_rows: int = 16
    scrambler: bool = True
    scrambler_seed: int = 0x7F
    equalizer_block_len: int = 250
    threshold_factor: float = 0.5

    def validate(self) -> None:
        if self.fft_len <= 0 or self.cp_len < 0:
            raise ValueError("fft_len must be positive and cp_len must be non-negative")
        if self.guard_carriers < 0:
            raise ValueError("guard_carriers must be non-negative")
        if self.dc_carriers not in (2, 4):
            raise ValueError("dc_carriers must be 2 or 4")
        if self.pilot_symbols <= 0 or self.pilot_hop < 0:
            raise ValueError("pilot_symbols must be positive and pilot_hop non-negative")
        if self.bits_per_symbol not in (1, 2, 4, 6):
            raise ValueError("bits_per_symbol must be one of 1, 2, 4, 6")
        if self.zc_length <= 1 or self.sample_rate <= 0:
            raise ValueError("zc_length must exceed one and sample_rate must be positive")
        if self.payload_peak_clip is not None and self.payload_peak_clip <= 0:
            raise ValueError("payload_peak_clip must be positive when enabled")
        if self.interleaver not in ("off", "block") or self.interleaver_rows <= 0:
            raise ValueError("invalid interleaver configuration")
        if self.scrambler_seed < 0 or self.equalizer_block_len <= 0:
            raise ValueError("invalid scrambler/equalizer configuration")
        if self.threshold_factor < 0:
            raise ValueError("threshold_factor must be non-negative")


@dataclass(frozen=True)
class SubcarrierPlan:
    fft_len: int
    guard_bins: np.ndarray
    dc_bins: np.ndarray
    effective_bins: np.ndarray
    pilot_locs: np.ndarray
    pilot_bins_by_symbol: tuple[np.ndarray, ...]
    data_bins_by_symbol: tuple[np.ndarray, ...]
    data_carriers_per_symbol: int


@dataclass(frozen=True)
class PhyProfile:
    config: OfdmConfig
    plan: SubcarrierPlan
    preamble: np.ndarray
    pilot_symbol: complex
    num_ofdm_symbols: int
    data_symbols_count: int
    padded_bit_count: int

    @property
    def symbol_span(self) -> int:
        return self.config.fft_len + self.config.cp_len

    @property
    def payload_samples(self) -> int:
        return self.num_ofdm_symbols * self.symbol_span

    @property
    def frame_samples(self) -> int:
        return int(self.preamble.size) + self.payload_samples


def build_subcarrier_plan(config: OfdmConfig, num_ofdm_symbols: int) -> SubcarrierPlan:
    config.validate()
    if num_ofdm_symbols <= 0:
        raise ValueError("num_ofdm_symbols must be positive")
    n_fft = config.fft_len
    guard_bins = np.arange(
        n_fft // 2 - config.guard_carriers,
        n_fft // 2 + config.guard_carriers,
        dtype=np.int32,
    )
    dc_bins = (
        np.asarray([0, n_fft - 1], dtype=np.int32)
        if config.dc_carriers == 2
        else np.asarray([0, 1, n_fft - 2, n_fft - 1], dtype=np.int32)
    )
    excluded = set(guard_bins.tolist()) | set(dc_bins.tolist())
    effective_bins = np.asarray(
        [index for index in range(n_fft) if index not in excluded], dtype=np.int32
    )
    pilot_stride = int(math.ceil(effective_bins.size / config.pilot_symbols))
    pilot_locs = np.arange(0, effective_bins.size, pilot_stride, dtype=np.int32)

    pilot_bins_by_symbol: list[np.ndarray] = []
    data_bins_by_symbol: list[np.ndarray] = []
    for symbol_index in range(num_ofdm_symbols):
        shifted = (pilot_locs + symbol_index * config.pilot_hop) % effective_bins.size
        pilot_bins = np.sort(effective_bins[shifted])
        pilot_bins_by_symbol.append(pilot_bins)
        pilot_set = set(int(value) for value in pilot_bins.tolist())
        data_bins_by_symbol.append(
            np.asarray(
                [index for index in effective_bins.tolist() if index not in pilot_set],
                dtype=np.int32,
            )
        )
    if not data_bins_by_symbol or data_bins_by_symbol[0].size == 0:
        raise ValueError("no data carriers are available with the current OFDM profile")
    return SubcarrierPlan(
        fft_len=n_fft,
        guard_bins=guard_bins,
        dc_bins=dc_bins,
        effective_bins=effective_bins,
        pilot_locs=pilot_locs,
        pilot_bins_by_symbol=tuple(pilot_bins_by_symbol),
        data_bins_by_symbol=tuple(data_bins_by_symbol),
        data_carriers_per_symbol=int(data_bins_by_symbol[0].size),
    )


def build_phy_profile(config: OfdmConfig, num_protocol_bits: int) -> PhyProfile:
    config.validate()
    if num_protocol_bits < 0:
        raise ValueError("num_protocol_bits must be non-negative")
    trial_plan = build_subcarrier_plan(config, 1)
    bits_per_symbol = trial_plan.data_carriers_per_symbol * config.bits_per_symbol
    num_symbols = max(1, int(math.ceil(max(num_protocol_bits, 1) / bits_per_symbol)))
    plan = build_subcarrier_plan(config, num_symbols)
    padded_bits = plan.data_carriers_per_symbol * num_symbols * config.bits_per_symbol
    return PhyProfile(
        config=config,
        plan=plan,
        preamble=zadoff_chu_seq(config.zc_length, config.zc_root),
        pilot_symbol=peak_magnitude(config.bits_per_symbol) * np.exp(1j * np.pi / 4.0),
        num_ofdm_symbols=num_symbols,
        data_symbols_count=padded_bits // config.bits_per_symbol,
        padded_bit_count=padded_bits,
    )
