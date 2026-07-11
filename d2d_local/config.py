"""PHY configuration dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
import math

from .ofdm_conf import OfdmConfig


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
class ChannelConfig:
    snr_db: float = 35.0
    time_offset: int = 32
    frequency_offset_hz: float = 0.0
    seed: int = 20260710
    multipath_taps: tuple[tuple[int, float, float], ...] = ((0, 1.0, 0.0),)

    def validate(self) -> None:
        if not math.isfinite(self.snr_db):
            raise ValueError("snr_db must be finite")
        if self.time_offset < 0:
            raise ValueError("time_offset must be non-negative")
        if not math.isfinite(self.frequency_offset_hz):
            raise ValueError("frequency_offset_hz must be finite")
        if self.seed < 0 or not self.multipath_taps:
            raise ValueError("seed must be non-negative and multipath_taps non-empty")
        delays: set[int] = set()
        energy = 0.0
        for delay, real, imag in self.multipath_taps:
            if not isinstance(delay, int) or delay < 0 or delay in delays:
                raise ValueError("multipath tap delays must be unique non-negative integers")
            if not math.isfinite(real) or not math.isfinite(imag):
                raise ValueError("multipath tap gains must be finite")
            delays.add(delay)
            energy += real * real + imag * imag
        if energy <= 0.0:
            raise ValueError("multipath_taps must contain non-zero energy")


@dataclass(frozen=True)
class SimulationConfig:
    protocol: ProtocolConfig = field(default_factory=ProtocolConfig)
    ofdm: OfdmConfig = field(default_factory=OfdmConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    payload_mode: str = "file"

    def validate(self) -> None:
        self.protocol.validate()
        self.ofdm.validate()
        self.channel.validate()
        if self.payload_mode not in {"file", "raw_rgb"}:
            raise ValueError("payload_mode must be 'file' or 'raw_rgb'")


__all__ = ["ChannelConfig", "OfdmConfig", "ProtocolConfig", "SimulationConfig"]
