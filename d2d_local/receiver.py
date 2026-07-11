"""Frame-set construction and IQ-stream decoding for OpenRadioPHY."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import OfdmConfig
from .frame_sync import find_frame_starts
from .ofdm import (
    PhyProfile,
    RxRecovery,
    build_phy_profile,
    decode_frame,
    encode_frame,
)
from .phy_chain import OfdmReceiver, OfdmTransmitter
from .protocol import bytes_to_bits


@dataclass(frozen=True)
class PhyFrameSet:
    """Encoded frames for one transfer."""

    profile: PhyProfile
    frame_bits: int  # bits in one protocol frame (all frames equal-size by construction)
    frame_byte_len: int  # protocol frame size in bytes (= frame_bits // 8)
    iq: np.ndarray  # concatenated IQ for all (optionally repeated) frames, complex64
    frames: int
    repeats: int


@dataclass(frozen=True)
class DecodedPhyFrame:
    """Decoded frame and diagnostics."""

    frame_bytes: bytes
    sync_index: int
    recovery: RxRecovery


def format_rx_level_report(samples: np.ndarray, *, clip_threshold: float = 0.95) -> str:
    """Summarize normalized IQ levels."""
    samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
    if samples.size == 0:
        return "rx_level: empty capture"
    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    peak = float(np.max(np.abs(samples)))
    clipped_pct = float(np.mean(np.abs(samples) > clip_threshold)) * 100.0
    fs_usage_db = 20.0 * math.log10(peak) if peak > 0 else float("-inf")
    if peak < 0.05:
        hint = " -- signal level is very weak"
    elif peak < 0.2:
        hint = " -- signal level is low"
    elif clipped_pct > 0.1:
        hint = " -- substantial clipping detected"
    else:
        hint = ""
    return (
        f"rx_level: rms={rms:.4f} peak={peak:.4f} "
        f"fs_usage={fs_usage_db:.1f}dBFS clipped={clipped_pct:.3f}%{hint}"
    )


def write_tx_iq_file(
    frame_bytes: Iterable[bytes],
    profile: PhyProfile,
    path: str | Path,
) -> tuple[Path, int]:
    """Write encoded frames as complex64 IQ."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    total_samples = 0
    with path.open("wb") as handle:
        for chunk in frame_bytes:
            iq = encode_frame(bytes_to_bits(bytes(chunk)), profile).astype(np.complex64)
            iq.tofile(handle)
            total_samples += int(iq.size)
    return path, total_samples


def required_tx_cycles_for_duration(
    frame_count: int,
    profile: PhyProfile,
    duration: float,
    *,
    extra_cycles: int = 1,
) -> int:
    """Return the frame cycles needed for a duration."""
    if frame_count <= 0:
        raise ValueError("frame_count must be positive")
    if duration <= 0.0:
        raise ValueError("duration must be positive")
    frames_needed = int(math.ceil(duration * profile.config.sample_rate / profile.frame_samples))
    return max(1, int(math.ceil(frames_needed / frame_count)) + extra_cycles)


def build_tx_frame_set(
    frame_bytes: Iterable[bytes],
    ofdm_config: OfdmConfig,
    *,
    repeats: int = 1,
    transmitter: OfdmTransmitter | None = None,
) -> PhyFrameSet:
    """Encode and concatenate equal-sized protocol frames."""
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    frame_list = [bytes(chunk) for chunk in frame_bytes]
    if not frame_list:
        raise ValueError("frame_bytes is empty")

    sizes = {len(chunk) for chunk in frame_list}
    if len(sizes) != 1:
        raise ValueError(f"protocol frames must be equal-sized, got sizes={sorted(sizes)}")
    frame_byte_len = sizes.pop()
    frame_bits = frame_byte_len * 8

    profile = build_phy_profile(ofdm_config, frame_bits)

    tx = transmitter or OfdmTransmitter()
    per_frame_iq = [tx(bytes_to_bits(chunk), profile).waveform for chunk in frame_list]
    one_cycle = np.concatenate(per_frame_iq)
    repeated = np.tile(one_cycle, repeats) if repeats > 1 else one_cycle
    return PhyFrameSet(
        profile=profile,
        frame_bits=frame_bits,
        frame_byte_len=frame_byte_len,
        iq=repeated.astype(np.complex64),
        frames=len(frame_list),
        repeats=repeats,
    )


def decode_iq_stream(
    samples: np.ndarray,
    profile: PhyProfile,
    *,
    frame_byte_len: int,
    threshold_factor: float = 0.5,
    correlation_plot_out: str | Path | None = None,
    correlation_plot_title: str | None = None,
) -> tuple[list[bytes], list[int]]:
    """Decode detected frames and trim them to ``frame_byte_len``."""
    samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
    if frame_byte_len <= 0:
        raise ValueError("frame_byte_len must be positive")
    if frame_byte_len * 8 > profile.padded_bit_count:
        raise ValueError(
            f"frame_byte_len={frame_byte_len} exceeds PHY capacity "
            f"({profile.padded_bit_count // 8} bytes)"
        )
    decoded = decode_iq_stream_diagnostics(
        samples,
        profile,
        frame_byte_len=frame_byte_len,
        threshold_factor=threshold_factor,
        correlation_plot_out=correlation_plot_out,
        correlation_plot_title=correlation_plot_title,
    )
    frame_bytes_out = [item.frame_bytes for item in decoded]
    sync_indices = [item.sync_index for item in decoded]
    return frame_bytes_out, sync_indices


def decode_iq_stream_diagnostics(
    samples: np.ndarray,
    profile: PhyProfile,
    *,
    frame_byte_len: int,
    threshold_factor: float = 0.5,
    max_frames: int | None = None,
    correlation_plot_out: str | Path | None = None,
    correlation_plot_title: str | None = None,
    receiver: OfdmReceiver | None = None,
) -> list[DecodedPhyFrame]:
    """Decode PHY frames and keep per-frame recovery diagnostics."""
    samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
    if frame_byte_len <= 0:
        raise ValueError("frame_byte_len must be positive")
    if frame_byte_len * 8 > profile.padded_bit_count:
        raise ValueError(
            f"frame_byte_len={frame_byte_len} exceeds PHY capacity "
            f"({profile.padded_bit_count // 8} bytes)"
        )
    frame_samples = profile.frame_samples
    starts = find_frame_starts(
        samples,
        profile.preamble,
        frame_samples,
        threshold_factor=threshold_factor,
        correlation_plot_out=correlation_plot_out,
        correlation_plot_title=correlation_plot_title,
    )

    out: list[DecodedPhyFrame] = []
    for start in starts:
        if max_frames is not None and len(out) >= max_frames:
            break
        end = start + frame_samples
        if end > samples.size:
            break
        # Keep a CP-length margin for fine timing.
        window = samples[start : min(end + profile.config.cp_len, samples.size)]
        # Frame starts are already synchronized.
        if receiver is None:
            recovery = decode_frame(window, profile, sync_search=False)
        else:
            trace = receiver(window, profile, sync_search=False)
            recovery = RxRecovery(
                sync_index=trace.sync_index,
                recovered_bits=trace.protocol_bits,
                equalized_symbols=trace.data_symbols,
                pilot_channel_estimates=trace.channel_grid,
                correlation_metric=trace.correlation_metric,
                trace=trace,
            )
        bits = recovery.recovered_bits[: frame_byte_len * 8]
        packed = np.packbits(bits).tobytes()[:frame_byte_len]
        out.append(
            DecodedPhyFrame(
                frame_bytes=packed,
                sync_index=start,
                recovery=recovery,
            )
        )
    return out
