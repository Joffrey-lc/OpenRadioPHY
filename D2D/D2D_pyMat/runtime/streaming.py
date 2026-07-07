"""Streaming-oriented TX/RX helpers built on the explicit PHY.

This module has no GNU Radio dependency.

- ``build_tx_frame_set`` is kept as a GR-free helper for offline loopback
  tests and benchmarks where a concatenated IQ vector is useful.
- ``decode_iq_stream`` is the frame-oriented RX decoder used both by the
  offline streaming loopback and by the real-time USRP capture path.

The real-time TX path itself streams frame IQ on demand and does not need a full-file IQ
vector; it uses ``runtime.gr_blocks.PyMatFrameSource`` to encode one frame at
a time.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from ..config import OfdmConfig
from ..ofdm import (
    PhyProfile,
    RxRecovery,
    build_phy_profile,
    decode_frame,
    encode_frame,
)
from ..protocol import bytes_to_bits
from ..viz import save_correlation_plot


@dataclass(frozen=True)
class PhyFrameSet:
    """Precomputed TX material for a full transfer.

    Primarily useful for offline loopback tests and benchmarks. The real-time
    TX path uses ``runtime.gr_blocks.PyMatFrameSource`` which encodes frames
    lazily and does not materialize the whole transfer.
    """

    profile: PhyProfile
    frame_bits: int  # bits in one protocol frame (all frames equal-size by construction)
    frame_byte_len: int  # protocol frame size in bytes (= frame_bits // 8)
    iq: np.ndarray  # concatenated IQ for all (optionally repeated) frames, complex64
    frames: int
    repeats: int


@dataclass(frozen=True)
class DecodedPhyFrame:
    """Recovered PHY frame plus diagnostics useful for plotting/debug."""

    frame_bytes: bytes
    sync_index: int
    recovery: RxRecovery


def format_rx_level_report(samples: np.ndarray, *, clip_threshold: float = 0.95) -> str:
    """Return a one-line health report on RX ADC utilization.

    USRP fc32 samples are in [-1, 1]. A capture whose RMS is far below 1.0
    wastes ADC bits and makes EVM extremely sensitive to small path-loss
    changes, which is the most common cause of run-to-run variance.
    """
    samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
    if samples.size == 0:
        return "rx_level: empty capture"
    rms = float(np.sqrt(np.mean(np.abs(samples) ** 2)))
    peak = float(np.max(np.abs(samples)))
    clipped_pct = float(np.mean(np.abs(samples) > clip_threshold)) * 100.0
    fs_usage_db = 20.0 * math.log10(peak) if peak > 0 else float("-inf")
    if peak < 0.05:
        hint = " -- signal very weak, raise --rx-gain"
    elif peak < 0.2:
        hint = " -- low level, consider raising --rx-gain"
    elif clipped_pct > 0.1:
        hint = " -- clipping, lower --rx-gain or --amplitude"
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
    """Encode protocol frames and persist concatenated complex64 IQ to disk."""
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
    """Return how many full protocol-frame cycles to stage for ``duration``.

    This is used by the real-time TX/TRX apps to avoid exercising the custom
    GNU Radio source block's runtime wrap-around path during fixed-duration
    tests. The byte sequence for ``N`` staged cycles is identical to a shorter
    list that loops, but staging enough cycles up front avoids mid-run list
    wrap and makes the TX behavior deterministic for a bounded capture window.
    """
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
) -> PhyFrameSet:
    """Encode each protocol frame as one PHY frame and concatenate the IQ.

    All protocol frames are expected to share the same byte length (the
    protocol layer already pads payloads), so a single ``PhyProfile`` covers
    every frame.
    """
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

    per_frame_iq = [encode_frame(bytes_to_bits(chunk), profile) for chunk in frame_list]
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


def find_frame_starts(
    samples: np.ndarray,
    preamble: np.ndarray,
    frame_samples: int,
    *,
    threshold_factor: float = 0.5,
    median_factor: float = 4.0,
    min_separation: int | None = None,
    correlation_plot_out: str | Path | None = None,
    correlation_plot_title: str | None = None,
) -> list[int]:
    """Find indices in ``samples`` where a preamble peak occurs.

    Uses a matched filter; after each peak, skip forward ~one frame to avoid
    re-detecting the same preamble. The detection threshold is a CFAR-style
    ``max(threshold_factor * global_peak, median_factor * median(metric))`` so a
    single strong interferer cannot raise the bar above weaker but valid
    preamble peaks.
    """
    samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
    preamble = np.asarray(preamble, dtype=np.complex64).reshape(-1)
    if samples.size < preamble.size:
        return []
    matched = np.convolve(samples, np.conj(preamble[::-1]), mode="valid")
    metric = np.abs(matched).astype(np.float32)
    peak = float(metric.max()) if metric.size else 0.0
    if peak <= 0.0:
        return []
    median = float(np.median(metric))
    threshold = max(threshold_factor * peak, median_factor * median)
    step_back = min_separation if min_separation is not None else max(1, frame_samples - 100)

    starts: list[int] = []
    i = 0
    limit = metric.size
    while i < limit:
        if metric[i] >= threshold:
            window_end = min(i + preamble.size, limit)
            local = i + int(np.argmax(metric[i:window_end]))
            starts.append(local)
            i = local + step_back
        else:
            i += 1

    if correlation_plot_out is not None:
        save_correlation_plot(
            metric,
            correlation_plot_out,
            threshold=threshold,
            starts=starts,
            title=correlation_plot_title,
        )
    return starts


def decode_iq_stream(
    samples: np.ndarray,
    profile: PhyProfile,
    *,
    frame_byte_len: int,
    threshold_factor: float = 0.5,
    correlation_plot_out: str | Path | None = None,
    correlation_plot_title: str | None = None,
) -> tuple[list[bytes], list[int]]:
    """Decode every PHY frame found in the captured IQ buffer.

    ``frame_byte_len`` is the exact protocol frame size (in bytes) used by
    the TX; each output block is trimmed to that length. The remaining bits
    in the PHY capacity beyond ``frame_byte_len`` are zero-padding produced
    by the TX and are dropped here so the byte stream is a clean
    concatenation of protocol frames for ``parse_frames``.
    """
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
        # The CP timing search may read up to cp_len samples past the nominal
        # frame end; include those samples when they are present in the capture.
        window = samples[start : min(end + profile.config.cp_len, samples.size)]
        # We already aligned the frame via ``find_frame_starts``; tell
        # ``decode_frame`` to trust that and skip its own internal argmax,
        # which at low SNR / long payload can jump to a spurious peak.
        recovery = decode_frame(window, profile, sync_search=False)
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
