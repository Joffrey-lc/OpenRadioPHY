"""OFDM TX/RX helpers for the D2D_pyMat link.

The module exposes deterministic PHY profiles, per-frame encoder/decoder
primitives, and whole-waveform helpers for simulation. The receiver uses
CP-based fine timing, scattered-pilot channel estimation, and per-symbol common
phase correction for USRP captures.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterator

import numpy as np

from .config import ChannelConfig, OfdmConfig
from .interleaver import deinterleave_bits, interleave_bits, scramble_bits
from .preamble import detect_preamble_start, zadoff_chu_seq
from .qam import bits_to_symbols, peak_magnitude, symbols_to_bits_hard


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
    """Deterministic PHY descriptor for frames of a given bit capacity.

    Both TX and RX can build this from ``OfdmConfig`` alone (plus the number of
    protocol bits per frame). The RX no longer depends on a TX-side object.
    """

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


def build_subcarrier_plan(config: OfdmConfig, num_ofdm_symbols: int) -> SubcarrierPlan:
    config.validate()
    n_fft = config.fft_len
    guard_bins = np.arange(
        n_fft // 2 - config.guard_carriers,
        n_fft // 2 + config.guard_carriers,
        dtype=np.int32,
    )
    if config.dc_carriers == 2:
        dc_bins = np.asarray([0, n_fft - 1], dtype=np.int32)
    else:
        dc_bins = np.asarray([0, 1, n_fft - 2, n_fft - 1], dtype=np.int32)

    excluded = set(guard_bins.tolist()) | set(dc_bins.tolist())
    effective_bins = np.asarray(
        [index for index in range(n_fft) if index not in excluded],
        dtype=np.int32,
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
        data_bins = np.asarray(
            [index for index in effective_bins.tolist() if index not in pilot_set],
            dtype=np.int32,
        )
        data_bins_by_symbol.append(data_bins)

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
    """Build a PHY profile sized to fit ``num_protocol_bits`` per frame."""
    config.validate()
    if num_protocol_bits < 0:
        raise ValueError("num_protocol_bits must be non-negative")
    trial_plan = build_subcarrier_plan(config, 1)
    bits_per_ofdm_symbol = trial_plan.data_carriers_per_symbol * config.bits_per_symbol
    num_ofdm_symbols = max(1, int(math.ceil(max(num_protocol_bits, 1) / bits_per_ofdm_symbol)))
    plan = build_subcarrier_plan(config, num_ofdm_symbols)
    padded_bit_count = plan.data_carriers_per_symbol * num_ofdm_symbols * config.bits_per_symbol
    preamble = zadoff_chu_seq(config.zc_length, config.zc_root)
    pilot_symbol = peak_magnitude(config.bits_per_symbol) * np.exp(1j * np.pi / 4.0)
    data_symbols_count = padded_bit_count // config.bits_per_symbol
    return PhyProfile(
        config=config,
        plan=plan,
        preamble=preamble,
        pilot_symbol=pilot_symbol,
        num_ofdm_symbols=num_ofdm_symbols,
        data_symbols_count=data_symbols_count,
        padded_bit_count=padded_bit_count,
    )


def _complex_interp_hold_edges(
    xp: np.ndarray,
    fp: np.ndarray,
    xq: np.ndarray,
) -> np.ndarray:
    xp = np.asarray(xp, dtype=np.float64).reshape(-1)
    fp = np.asarray(fp, dtype=np.complex128).reshape(-1)
    xq = np.asarray(xq, dtype=np.float64).reshape(-1)
    if xp.size == 0:
        return np.zeros(xq.size, dtype=np.complex128)
    if xp.size == 1:
        return np.full(xq.size, fp[0], dtype=np.complex128)

    real = np.interp(xq, xp, fp.real)
    imag = np.interp(xq, xp, fp.imag)
    out = real + 1j * imag
    out[xq < xp[0]] = fp[0]
    out[xq > xp[-1]] = fp[-1]
    return out.astype(np.complex128)


_SOFT_LIMIT_KNEE_RATIO = 0.85


def _soft_limit_magnitude(samples: np.ndarray, peak_clip: float) -> np.ndarray:
    """Apply a smooth magnitude limiter without changing sample phase.

    Samples at or below the knee (85% of ``peak_clip``) pass through
    untouched; only the excess above the knee is tanh-compressed into the
    remaining headroom, saturating at ``peak_clip``. A plain
    ``peak_clip * tanh(magnitude / peak_clip)`` limiter is nonlinear over the
    whole amplitude range and can raise EVM even without channel noise.
    """
    samples = np.asarray(samples, dtype=np.complex128)
    magnitude = np.abs(samples)
    knee = _SOFT_LIMIT_KNEE_RATIO * peak_clip
    over = magnitude > knee
    if not np.any(over):
        return samples

    headroom = peak_clip - knee
    limited = knee + headroom * np.tanh((magnitude[over] - knee) / headroom)
    scale = np.ones_like(magnitude, dtype=np.float64)
    scale[over] = limited / magnitude[over]
    return samples * scale


def _estimate_cp_timing_offset(
    payload: np.ndarray,
    profile: PhyProfile,
    *,
    symbols_to_use: int = 4,
) -> int:
    cfg = profile.config
    cp_len = cfg.cp_len
    if cp_len <= 0:
        return 0

    symbol_span = profile.symbol_span
    use_symbols = min(profile.num_ofdm_symbols, symbols_to_use)
    max_offset = cp_len - 1
    best_offset = 0
    best_metric = -1.0

    for offset in range(max_offset + 1):
        metric_sum = 0.0
        valid = 0
        for si in range(use_symbols):
            start = offset + si * symbol_span
            end = start + symbol_span
            if end > payload.size:
                break
            symbol = payload[start:end]
            cp = symbol[:cp_len]
            tail = symbol[cfg.fft_len : cfg.fft_len + cp_len]
            denom = float(np.linalg.norm(cp) * np.linalg.norm(tail))
            if denom <= 1e-12:
                continue
            metric_sum += abs(np.vdot(tail, cp)) / denom
            valid += 1
        if valid == 0:
            continue
        metric = metric_sum / valid
        if metric > best_metric:
            best_metric = metric
            best_offset = offset
    return best_offset


def _blockwise_equalize(
    freq_grid: np.ndarray,
    profile: PhyProfile,
    *,
    block_len: int,
) -> tuple[np.ndarray, np.ndarray]:
    plan = profile.plan
    effective_bins = plan.effective_bins
    num_symbols = profile.num_ofdm_symbols
    n_eff = effective_bins.size

    equalized_grid = np.zeros_like(freq_grid, dtype=np.complex128)
    channel_grid = np.zeros_like(freq_grid, dtype=np.complex128)
    effective_index = {int(bin_idx): idx for idx, bin_idx in enumerate(effective_bins.tolist())}

    # Interpolate each contiguous subcarrier segment independently. Guard-band
    # and DC gaps separate distant frequencies, so complex interpolation must
    # not bridge across those gaps.
    segment_jumps = np.flatnonzero(np.diff(effective_bins) != 1)
    segment_bounds = np.concatenate(([0], segment_jumps + 1, [n_eff]))

    block_len = max(1, min(block_len, num_symbols))
    num_blocks = int(math.ceil(num_symbols / block_len))
    # Avoid a short tail block with too few pilot observations; merge it into
    # the preceding block so the channel estimate remains stable.
    if num_blocks > 1:
        tail_size = num_symbols - (num_blocks - 1) * block_len
        if tail_size < max(1, block_len // 2):
            num_blocks -= 1

    for block_index in range(num_blocks):
        start_idx = block_index * block_len
        end_idx = num_symbols if block_index == num_blocks - 1 else (block_index + 1) * block_len
        block_cols = range(start_idx, end_idx)

        h_sum = np.zeros(n_eff, dtype=np.complex128)
        h_count = np.zeros(n_eff, dtype=np.int32)

        for si in block_cols:
            pb = plan.pilot_bins_by_symbol[si]
            if pb.size == 0:
                continue
            shifted = np.asarray([effective_index[int(value)] for value in pb], dtype=np.int32)
            h_pilot = freq_grid[pb, si] / profile.pilot_symbol
            h_sum[shifted] += h_pilot
            h_count[shifted] += 1

        valid = h_count > 0
        if not np.any(valid):
            h_block_eff = np.ones(n_eff, dtype=np.complex128)
        else:
            h_block_eff = np.zeros(n_eff, dtype=np.complex128)
            h_block_eff[valid] = h_sum[valid] / h_count[valid]
            if not np.all(valid):
                known_all = np.flatnonzero(valid)
                for seg_start, seg_end in zip(segment_bounds[:-1], segment_bounds[1:]):
                    seg_start = int(seg_start)
                    seg_end = int(seg_end)
                    seg = slice(seg_start, seg_end)
                    seg_known = np.flatnonzero(valid[seg])
                    if seg_known.size == 0:
                        # No pilot landed in this segment for this block:
                        # borrow the nearest estimate instead of leaving zeros.
                        seg_mid = (seg_start + seg_end - 1) / 2.0
                        nearest = known_all[np.argmin(np.abs(known_all - seg_mid))]
                        h_block_eff[seg] = h_block_eff[nearest]
                    elif seg_known.size < seg_end - seg_start:
                        h_block_eff[seg] = _complex_interp_hold_edges(
                            seg_known.astype(np.float64),
                            h_block_eff[seg][seg_known],
                            np.arange(seg_end - seg_start, dtype=np.float64),
                        )

        mag2 = np.abs(h_block_eff) ** 2
        nz = mag2[mag2 > 0]
        floor_val = max(1e-3 * float(np.median(nz)), 1e-8) if nz.size else 1e-8
        equalizer = np.conj(h_block_eff) / (mag2 + floor_val)

        for si in block_cols:
            x_eq = equalizer * freq_grid[effective_bins, si]
            pb = plan.pilot_bins_by_symbol[si]
            if pb.size:
                shifted = np.asarray([effective_index[int(value)] for value in pb], dtype=np.int32)
                eq_pilots = x_eq[shifted]
                weights = np.abs(h_block_eff[shifted]) ** 2
                if float(np.sum(weights)) <= 1e-12:
                    cpe = np.angle(np.sum(eq_pilots * np.conj(profile.pilot_symbol)))
                else:
                    cpe = np.angle(
                        np.sum(weights * eq_pilots * np.conj(profile.pilot_symbol))
                    )
                x_eq *= np.exp(-1j * cpe)

            equalized_grid[effective_bins, si] = x_eq
            channel_grid[effective_bins, si] = h_block_eff

    return equalized_grid, channel_grid


def encode_frame(protocol_bits: np.ndarray, profile: PhyProfile) -> np.ndarray:
    """Encode one frame's bits into complex IQ samples (preamble + CP-OFDM)."""
    bits = np.asarray(protocol_bits, dtype=np.uint8).reshape(-1)
    cfg = profile.config
    if bits.size > profile.padded_bit_count:
        raise ValueError(
            f"frame bits ({bits.size}) exceed profile capacity ({profile.padded_bit_count})"
        )
    padded = np.pad(bits, (0, profile.padded_bit_count - bits.size), constant_values=0)
    # Scramble BEFORE interleave so structured inputs (e.g. image zero-runs)
    # are whitened in the constellation domain. This drops OFDM PAPR from
    # ~14-17 dB down to the Gaussian ~10 dB bound for structured data.
    padded = scramble_bits(padded, enabled=cfg.scrambler, seed=cfg.scrambler_seed)
    padded = interleave_bits(
        padded,
        cfg.interleaver,
        rows=cfg.interleaver_rows,
    )
    mapped_symbols = bits_to_symbols(padded, profile.config.bits_per_symbol)
    grid = np.zeros((cfg.fft_len, profile.num_ofdm_symbols), dtype=np.complex128)
    cursor = 0
    for si in range(profile.num_ofdm_symbols):
        pb = profile.plan.pilot_bins_by_symbol[si]
        db = profile.plan.data_bins_by_symbol[si]
        grid[pb, si] = profile.pilot_symbol
        grid[db, si] = mapped_symbols[cursor : cursor + db.size]
        cursor += int(db.size)

    time_domain = np.fft.ifft(grid, axis=0)
    with_cp = np.concatenate([time_domain[-cfg.cp_len :, :], time_domain], axis=0)
    payload = with_cp.reshape(-1, order="F")

    preamble = profile.preamble
    if cfg.normalize_waveform:
        # Scale the constant-envelope ZC preamble so its magnitude equals the
        # OFDM payload's peak, then peak-normalize the combined frame. This
        # aligns the preamble peak and the payload peak at the DAC input so a
        # single --amplitude sets the PA backoff for both. Without this step
        # the ZC (|zc|=1) dominates the frame peak while the payload sits
        # ~10 dB below, which forces a trade-off between ZC entering PA
        # compression vs. payload being buried in quantization noise.
        payload_peak = float(np.max(np.abs(payload)))
        preamble_peak = float(np.max(np.abs(preamble)))
        if payload_peak > 0 and preamble_peak > 0:
            preamble = preamble * (payload_peak / preamble_peak)
    waveform = np.concatenate([preamble, payload])
    if cfg.normalize_waveform:
        peak = float(np.max(np.abs(waveform)))
        if peak > 0:
            waveform = waveform / peak
    if cfg.payload_peak_clip is not None:
        waveform = _soft_limit_magnitude(waveform, cfg.payload_peak_clip)
    return waveform.astype(np.complex128)


def iter_tx_frames(
    protocol_bits: np.ndarray,
    profile: PhyProfile,
    *,
    bits_per_frame: int,
) -> Iterator[np.ndarray]:
    """Yield per-frame IQ for streaming transmission."""
    bits = np.asarray(protocol_bits, dtype=np.uint8).reshape(-1)
    if bits_per_frame <= 0:
        raise ValueError("bits_per_frame must be positive")
    if bits_per_frame > profile.padded_bit_count:
        raise ValueError(
            f"bits_per_frame ({bits_per_frame}) exceeds profile capacity "
            f"({profile.padded_bit_count})"
        )
    for start in range(0, bits.size, bits_per_frame):
        yield encode_frame(bits[start : start + bits_per_frame], profile)


def decode_frame(
    received: np.ndarray,
    profile: PhyProfile,
    *,
    sync_search: bool = True,
) -> RxRecovery:
    """Decode one PHY frame from a received IQ buffer.

    ``sync_search`` controls the internal matched-filter preamble search:

    - ``True``:
      run a full matched-filter over ``received`` and pick the global
      argmax. Appropriate when the caller does not know the preamble
      position.
    - ``False``: trust that ``received[0]`` is already the first sample
      of the preamble. The streaming path (``decode_iq_stream_diagnostics``)
      has already localised each frame via ``find_frame_starts``; running
      a second full-window argmax there can relocate sync onto a
      payload-induced spurious peak at marginal SNR, silently corrupting
      the FFT window. The caller guarantees alignment, so we skip it.
    """
    samples = np.asarray(received, dtype=np.complex128).reshape(-1)
    if sync_search:
        sync_index, correlation_metric = detect_preamble_start(samples, profile.preamble)
    else:
        sync_index = 0
        correlation_metric = np.zeros(1, dtype=np.complex128)
    payload_start = sync_index + profile.preamble.size

    cfg = profile.config
    symbol_span = profile.symbol_span
    payload_len = profile.num_ofdm_symbols * symbol_span
    payload = samples[payload_start : payload_start + payload_len + cfg.cp_len]
    if payload.size < payload_len + cfg.cp_len:
        payload = np.pad(
            payload,
            (0, payload_len + cfg.cp_len - payload.size),
            constant_values=0.0,
        )

    timing_offset = _estimate_cp_timing_offset(payload, profile)
    payload = payload[timing_offset : timing_offset + payload_len]
    if payload.size < payload_len:
        payload = np.pad(payload, (0, payload_len - payload.size), constant_values=0.0)
    payload_matrix = payload.reshape(symbol_span, profile.num_ofdm_symbols, order="F")
    payload_no_cp = payload_matrix[cfg.cp_len :, :]
    freq_grid = np.fft.fft(payload_no_cp, axis=0)
    equalized_grid, channel_estimates = _blockwise_equalize(
        freq_grid, profile, block_len=cfg.equalizer_block_len,
    )

    recovered_symbols: list[np.ndarray] = []
    for si in range(profile.num_ofdm_symbols):
        db = profile.plan.data_bins_by_symbol[si]
        recovered_symbols.append(equalized_grid[db, si])
    equalized_symbols = np.concatenate(recovered_symbols)[: profile.data_symbols_count]
    recovered_bits = symbols_to_bits_hard(equalized_symbols, cfg.bits_per_symbol)
    recovered_bits = recovered_bits[: profile.padded_bit_count]
    recovered_bits = deinterleave_bits(
        recovered_bits,
        cfg.interleaver,
        rows=cfg.interleaver_rows,
    )
    recovered_bits = scramble_bits(
        recovered_bits, enabled=cfg.scrambler, seed=cfg.scrambler_seed
    )

    return RxRecovery(
        sync_index=sync_index,
        recovered_bits=recovered_bits,
        equalized_symbols=equalized_symbols,
        pilot_channel_estimates=channel_estimates,
        correlation_metric=correlation_metric,
    )


def build_tx_waveform(protocol_bits: np.ndarray, config: OfdmConfig) -> TxWaveform:
    """Build a complete TX waveform for simulation."""
    bits = np.asarray(protocol_bits, dtype=np.uint8).reshape(-1)
    profile = build_phy_profile(config, bits.size)
    waveform = encode_frame(bits, profile)
    padded_bits = np.pad(bits, (0, profile.padded_bit_count - bits.size), constant_values=0)
    return TxWaveform(
        waveform=waveform,
        preamble=profile.preamble,
        pilot_symbol=profile.pilot_symbol,
        plan=profile.plan,
        protocol_bits=bits,
        padded_bits=padded_bits,
        num_ofdm_symbols=profile.num_ofdm_symbols,
        data_symbols_count=profile.data_symbols_count,
        profile=profile,
    )


def apply_channel(
    tx_waveform: np.ndarray,
    config: ChannelConfig,
    *,
    sample_rate: float,
) -> np.ndarray:
    samples = np.asarray(tx_waveform, dtype=np.complex128).reshape(-1)
    config.validate()

    if config.frequency_offset_hz:
        n = np.arange(samples.size, dtype=np.float64)
        phase = 2.0 * np.pi * config.frequency_offset_hz * n / sample_rate
        samples = samples * np.exp(1j * phase)

    signal_power = float(np.mean(np.abs(samples) ** 2)) if samples.size else 1.0
    snr_linear = 10.0 ** (config.snr_db / 10.0)
    noise_power = signal_power / max(snr_linear, 1e-12)
    noise_sigma = math.sqrt(noise_power / 2.0)
    noise = noise_sigma * (
        np.random.standard_normal(samples.size) + 1j * np.random.standard_normal(samples.size)
    )

    padded = np.pad(samples + noise, (config.time_offset, 0), constant_values=0.0)
    return padded.astype(np.complex128)


def recover_protocol_bits(
    received: np.ndarray,
    profile_or_tx: PhyProfile | TxWaveform,
    config: OfdmConfig | None = None,  # kept for backward-compat; unused
) -> RxRecovery:
    """Recover bits from a received IQ buffer.

    Accepts either a ``PhyProfile`` (new, preferred 闁?RX no longer needs a
    TX object) or a complete ``TxWaveform`` from simulation.
    """
    del config  # profile carries its own OfdmConfig now
    profile = profile_or_tx.profile if isinstance(profile_or_tx, TxWaveform) else profile_or_tx
    return decode_frame(received, profile)
