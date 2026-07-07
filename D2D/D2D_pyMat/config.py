"""Shared configuration objects for D2D_pyMat."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ProtocolConfig:
    payload_size: int = 1024
    metadata_size: int = 128

    def validate(self) -> None:
        if self.payload_size <= 0:
            raise ValueError("payload_size must be positive")
        if self.metadata_size <= 0:
            raise ValueError("metadata_size must be positive")


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
    tx_warmup_ms: float = 500.0
    rx_settle_ms: float = 500.0
    # Relative to the strongest correlation peak in the capture. Values much
    # above 0.5 drop valid frames whenever the received amplitude varies
    # across the capture; the median-CFAR branch in find_frame_starts keeps
    # the false-alarm rate in check at 0.5.
    threshold_factor: float = 0.5

    def validate(self) -> None:
        if self.fft_len <= 0 or self.cp_len < 0:
            raise ValueError("fft_len must be positive and cp_len must be non-negative")
        if self.guard_carriers < 0:
            raise ValueError("guard_carriers must be non-negative")
        if self.dc_carriers not in (2, 4):
            raise ValueError("dc_carriers must be 2 or 4 for this implementation")
        if self.pilot_symbols <= 0:
            raise ValueError("pilot_symbols must be positive")
        if self.pilot_hop < 0:
            raise ValueError("pilot_hop must be non-negative")
        if self.bits_per_symbol not in (1, 2, 4, 6):
            raise ValueError("bits_per_symbol must be one of 1, 2, 4, 6")
        if self.zc_length <= 1:
            raise ValueError("zc_length must be greater than 1")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.payload_peak_clip is not None and self.payload_peak_clip <= 0:
            raise ValueError("payload_peak_clip must be positive when enabled")
        if self.interleaver not in ("off", "block"):
            raise ValueError("interleaver must be 'off' or 'block'")
        if self.interleaver_rows <= 0:
            raise ValueError("interleaver_rows must be positive")
        if self.scrambler_seed < 0:
            raise ValueError("scrambler_seed must be non-negative")
        if self.equalizer_block_len <= 0:
            raise ValueError("equalizer_block_len must be positive")
        if self.tx_warmup_ms < 0:
            raise ValueError("tx_warmup_ms must be non-negative")
        if self.rx_settle_ms < 0:
            raise ValueError("rx_settle_ms must be non-negative")
        if self.threshold_factor < 0:
            raise ValueError("threshold_factor must be non-negative")


@dataclass(frozen=True)
class ChannelConfig:
    snr_db: float = 35.0
    time_offset: int = 32
    frequency_offset_hz: float = 0.0

    def validate(self) -> None:
        if self.time_offset < 0:
            raise ValueError("time_offset must be non-negative")


@dataclass(frozen=True)
class SimulationConfig:
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    ofdm: OfdmConfig = field(default_factory=OfdmConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)

    def validate(self) -> None:
        self.protocol.validate()
        self.ofdm.validate()
        self.channel.validate()
