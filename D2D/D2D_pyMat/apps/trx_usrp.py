"""Single-USRP image link over the D2D_pyMat OFDM PHY.

The app transmits an image payload through one USRP, captures the looped-back
IQ stream, decodes the received frames, and writes link diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from .. import __version__
from ..config import OfdmConfig, ProtocolConfig
from ..ofdm import build_phy_profile, encode_frame
from ..protocol import (
    build_protocol_frames,
    build_protocol_frames_from_bytes,
    bytes_to_bits,
    frame_size as protocol_frame_size,
    parse_frames,
)
from ..runtime.streaming import decode_iq_stream_diagnostics, format_rx_level_report, required_tx_cycles_for_duration
from ..viz import save_constellation_plot, save_papr_ccdf_plot

MODULATION_LABELS = {1: "BPSK", 2: "QPSK", 4: "16QAM", 6: "64QAM"}
DEFAULT_OUTPUT_DIR = Path("trx_outputs")


@dataclass(frozen=True)
class RawImagePayload:
    width: int
    height: int
    raw_bytes: bytes
    png_name: str
    raw_name: str


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a non-negative finite number")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a positive finite number")
    return parsed


def path_arg(value: str) -> Path:
    return Path(value).expanduser()


def _modulation_slug(bits_per_symbol: int) -> str:
    return MODULATION_LABELS.get(bits_per_symbol, str(bits_per_symbol)).lower()


def _file_sha256(path: Path, *, chunk_size: int = 1 << 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_raw_image(input_path: Path, output_name: str | None) -> RawImagePayload:
    with Image.open(input_path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        raw_bytes = rgb.tobytes()

    stem = Path(output_name).stem if output_name else f"{input_path.stem}_rx"
    if not stem:
        stem = input_path.stem or "rx_image"
    return RawImagePayload(
        width=int(width),
        height=int(height),
        raw_bytes=raw_bytes,
        png_name=f"{stem}.png",
        raw_name=f"{stem}.raw.rgb",
    )


def _recover_debug_pixels(frames, *, total_size: int, payload_size: int) -> tuple[bytes, int]:
    expected_chunks = max(1, math.ceil(total_size / payload_size))
    chunks: dict[int, bytes] = {}
    for frame in frames:
        if frame.chunk_count != expected_chunks:
            continue
        if 0 <= frame.chunk_index < expected_chunks and frame.chunk_index not in chunks:
            chunks[frame.chunk_index] = frame.payload

    recovered = bytearray(total_size)
    for chunk_index, payload in chunks.items():
        start = chunk_index * payload_size
        end = min(total_size, start + payload_size)
        recovered[start:end] = payload[: end - start]
    return bytes(recovered), len(chunks)


def _write_raw_rgb_outputs(payload: RawImagePayload, recovered_bytes: bytes, output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_len = payload.width * payload.height * 3
    if len(recovered_bytes) < expected_len:
        pixel_bytes = recovered_bytes + b"\x00" * (expected_len - len(recovered_bytes))
    else:
        pixel_bytes = recovered_bytes[:expected_len]

    raw_path = output_dir / payload.raw_name
    png_path = output_dir / payload.png_name
    raw_path.write_bytes(pixel_bytes)
    Image.frombytes("RGB", (payload.width, payload.height), pixel_bytes).save(png_path, format="PNG")
    return raw_path, png_path


def _file_output_name(input_path: Path, output_name: str | None) -> str:
    if output_name:
        name = Path(output_name).name
        if name:
            return name
    suffix = input_path.suffix or ".bin"
    return f"{input_path.stem}_rx{suffix}"


def _write_file_output(recovered_bytes: bytes, output_dir: Path, output_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / Path(output_name).name
    output_path.write_bytes(recovered_bytes)
    return output_path


def _compute_tx_papr_report(
    frame_bytes: tuple[bytes, ...],
    profile,
    *,
    max_plot_samples: int = 200_000,
) -> tuple[dict[str, object], np.ndarray]:
    total_expected = max(1, len(frame_bytes) * profile.frame_samples)
    plot_stride = max(1, int(math.ceil(total_expected / max_plot_samples)))
    plot_chunks: list[np.ndarray] = []
    sample_offset = 0
    sample_count = 0
    sum_power = 0.0
    max_power = 0.0

    for chunk in frame_bytes:
        waveform = encode_frame(bytes_to_bits(chunk), profile)
        power = np.abs(waveform) ** 2
        if power.size == 0:
            continue
        sample_count += int(power.size)
        sum_power += float(np.sum(power, dtype=np.float64))
        max_power = max(max_power, float(np.max(power)))

        first = (-sample_offset) % plot_stride
        if first < power.size:
            plot_chunks.append(power[first::plot_stride].astype(np.float64, copy=False))
        sample_offset += int(power.size)

    if sample_count == 0 or sum_power <= 0.0:
        raise ValueError("cannot compute PAPR for an empty or zero-power waveform")

    avg_power = sum_power / float(sample_count)
    rms = math.sqrt(avg_power)
    peak = math.sqrt(max_power)
    papr_db = 10.0 * math.log10(max_power / avg_power) if max_power > 0.0 else float("-inf")

    plot_power = np.concatenate(plot_chunks) if plot_chunks else np.asarray([avg_power], dtype=np.float64)
    power_ratio_db = 10.0 * np.log10(np.maximum(plot_power / avg_power, np.finfo(np.float64).tiny))
    percentiles = {
        "p50_db": float(np.percentile(power_ratio_db, 50.0)),
        "p90_db": float(np.percentile(power_ratio_db, 90.0)),
        "p99_db": float(np.percentile(power_ratio_db, 99.0)),
        "p999_db": float(np.percentile(power_ratio_db, 99.9)),
    }
    report: dict[str, object] = {
        "frames": int(len(frame_bytes)),
        "samples": int(sample_count),
        "plot_samples": int(power_ratio_db.size),
        "plot_stride": int(plot_stride),
        "avg_power": float(avg_power),
        "rms": float(rms),
        "peak": float(peak),
        "papr_db": float(papr_db),
        **percentiles,
    }
    return report, power_ratio_db


def build_parser() -> argparse.ArgumentParser:
    defaults = OfdmConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", action="version", version=f"D2D_pyMat {__version__}")
    parser.add_argument("input", type=path_arg, help="image file to transmit")
    parser.add_argument("--addr", default="addr=192.168.10.2")
    parser.add_argument("--freq", type=float, required=True)
    parser.add_argument("--sample-rate", type=float, default=2e6)
    parser.add_argument("--tx-gain", type=float, default=20.0)
    parser.add_argument("--rx-gain", type=float, default=20.0)
    parser.add_argument("--amplitude", type=float, default=0.2)
    parser.add_argument("--papr-clip", type=positive_float, default=None,
                        help="enable TX soft clipping at this normalized magnitude, e.g. 0.8; omit to disable")
    parser.add_argument("--papr-report", action="store_true",
                        help="compute TX PAPR and write a CCDF plot before starting the USRP flowgraph")
    parser.add_argument("--clock-source", choices=("internal", "external", "gpsdo"), default="internal")
    parser.add_argument("--time-source", choices=("internal", "external", "gpsdo"), default="internal")
    parser.add_argument("--tx-antenna", default="TX/RX")
    parser.add_argument("--rx-antenna", default="RX2")
    parser.add_argument("--tx-channel", type=non_negative_int, default=0)
    parser.add_argument("--rx-channel", type=non_negative_int, default=0)
    parser.add_argument("--payload-size", type=positive_int, default=1024)
    parser.add_argument("--metadata-size", type=positive_int, default=64)
    parser.add_argument("--bits-per-symbol", type=int, choices=(1, 2, 4, 6), default=4)
    parser.add_argument("--repeats", type=positive_int, default=1)
    parser.add_argument("--continuous", action="store_true", help="stage enough frame cycles for --duration")
    parser.add_argument("--duration", type=float, default=6.0,
                        help="usable receive window in seconds after the TX warmup interval")
    parser.add_argument("--tx-warmup-ms", type=non_negative_float, default=defaults.tx_warmup_ms,
                        help="TX warmup interval aired before the first data frame")
    parser.add_argument("--rx-settle-ms", type=non_negative_float, default=defaults.rx_settle_ms,
                        help="initial RX samples discarded before frame detection")
    parser.add_argument("--threshold-factor", type=non_negative_float, default=defaults.threshold_factor,
                        help="relative preamble detection threshold; median-CFAR is also applied")
    parser.add_argument("--tx-source-mode", choices=("streaming",), default="streaming")
    parser.add_argument("--crc-mode", choices=("debug", "strict"), default="debug",
                        help="debug writes a best-effort raw-RGB PNG even with CRC errors; "
                             "strict only writes output when frame CRCs and file CRC pass")
    parser.add_argument("--payload-mode", choices=("auto", "file", "raw-rgb"), default="auto",
                        help="auto uses raw RGB in debug mode and original file bytes in strict mode")
    parser.add_argument("--constellation-frames", type=positive_int, default=8,
                        help="number of detected PHY frames used for the constellation plot")
    parser.add_argument("--output-name", default=None, help="output name; raw-rgb mode writes PNG using the stem, file mode preserves the extension")
    parser.add_argument("--output-dir", type=path_arg, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--raw-out", type=path_arg, default=None, help="raw IQ capture path")
    parser.add_argument("--summary-json", type=path_arg, default=None, help="run summary JSON path")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.duration <= 0:
        raise SystemExit("--duration must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    crc_debug = args.crc_mode == "debug"
    payload_mode = args.payload_mode
    if payload_mode == "auto":
        payload_mode = "raw-rgb" if crc_debug else "file"

    modulation = _modulation_slug(args.bits_per_symbol)
    raw_out = args.raw_out or (args.output_dir / f"trx_pymat_{modulation}.fc32")
    summary_json = args.summary_json or (args.output_dir / f"trx_summary_{modulation}.json")
    diagnostics_dir = args.output_dir / "diagnostics"
    correlation_out = diagnostics_dir / f"trx_correlation_{modulation}.png"
    constellation_out = diagnostics_dir / f"trx_constellation_{modulation}.png"
    papr_ccdf_out = diagnostics_dir / f"trx_papr_ccdf_{modulation}.png"
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    image_payload = _load_raw_image(args.input, args.output_name) if payload_mode == "raw-rgb" else None
    protocol_cfg = ProtocolConfig(payload_size=args.payload_size, metadata_size=args.metadata_size)
    ofdm_cfg = OfdmConfig(
        bits_per_symbol=args.bits_per_symbol,
        sample_rate=args.sample_rate,
        payload_peak_clip=args.papr_clip,
        tx_warmup_ms=args.tx_warmup_ms,
        rx_settle_ms=args.rx_settle_ms,
        threshold_factor=args.threshold_factor,
    )

    if payload_mode == "raw-rgb":
        assert image_payload is not None
        payload_bytes = image_payload.raw_bytes
        build = build_protocol_frames_from_bytes(
            payload_bytes,
            "raw.rgb",
            "image/rgb",
            protocol_cfg,
            input_path=args.input,
        )
        payload_desc = f"raw-rgb size={image_payload.width}x{image_payload.height}"
    else:
        build = build_protocol_frames(args.input, protocol_cfg)
        payload_bytes = build.source_bytes
        payload_desc = f"file media={build.media_type}"

    frame_byte_len = protocol_frame_size(protocol_cfg.metadata_size, protocol_cfg.payload_size)
    profile = build_phy_profile(ofdm_cfg, frame_byte_len * 8)
    one_cycle_seconds = len(build.frame_bytes) * profile.frame_samples / args.sample_rate

    tx_cycle = build.frame_bytes * args.repeats
    tx_warmup_samples = max(0, int(round(args.sample_rate * ofdm_cfg.tx_warmup_ms / 1000.0)))
    tx_warmup_frames = int(math.ceil(tx_warmup_samples / profile.frame_samples)) if tx_warmup_samples else 0
    warmup_prefix = (bytes(frame_byte_len),) * tx_warmup_frames
    warmup_seconds = tx_warmup_frames * profile.frame_samples / args.sample_rate
    staged_cycles = 1
    if args.continuous:
        staged_cycles = required_tx_cycles_for_duration(
            len(tx_cycle),
            profile,
            args.duration + warmup_seconds,
        )
    frame_bytes = warmup_prefix + (tx_cycle * staged_cycles)
    # TX airs warmup frames before the first data frame, so the capture window
    # covers both the warmup interval and the requested data window.
    capture_seconds = args.duration + warmup_seconds

    print(
        f"trx input={args.input} payload={payload_mode} {payload_desc} "
        f"payload_bytes={len(payload_bytes)} frames={len(build.frame_bytes)} "
        f"one_cycle_seconds={one_cycle_seconds:.3f} staged_cycles={staged_cycles} "
        f"warmup_seconds={warmup_seconds:.3f} capture_seconds={capture_seconds:.3f} "
        f"modulation={MODULATION_LABELS[args.bits_per_symbol]} papr_clip={args.papr_clip}"
    )

    papr_summary = None
    if args.papr_report:
        papr_summary, power_ratio_db = _compute_tx_papr_report(build.frame_bytes, profile)
        save_papr_ccdf_plot(
            power_ratio_db,
            papr_ccdf_out,
            title=f"TX PAPR CCDF ({MODULATION_LABELS[args.bits_per_symbol]})",
        )
        papr_summary["ccdf_out"] = str(papr_ccdf_out)
        papr_summary["clip"] = args.papr_clip
        print(
            f"papr_report frames={papr_summary['frames']} samples={papr_summary['samples']} "
            f"papr_db={papr_summary['papr_db']:.2f} peak={papr_summary['peak']:.4f} "
            f"rms={papr_summary['rms']:.4f} clip={args.papr_clip} ccdf_out={papr_ccdf_out}"
        )

    from gnuradio import gr  # type: ignore
    from ..runtime.gr_flowgraphs import PyMatUsrpTrx

    setup_start = time.perf_counter()
    tb = PyMatUsrpTrx(
        frame_bytes,
        profile,
        str(raw_out),
        device_addr=args.addr,
        sample_rate=args.sample_rate,
        center_freq=args.freq,
        tx_gain=args.tx_gain,
        rx_gain=args.rx_gain,
        amplitude=args.amplitude,
        clock_source=args.clock_source,
        time_source=args.time_source,
        tx_antenna=args.tx_antenna,
        rx_antenna=args.rx_antenna,
        tx_channel=args.tx_channel,
        rx_channel=args.rx_channel,
    )
    if gr.enable_realtime_scheduling() != gr.RT_OK:
        print("warning: realtime scheduling was not enabled")
    setup_elapsed = time.perf_counter() - setup_start

    run_start = time.perf_counter()
    tb.run_for(capture_seconds)
    run_elapsed = time.perf_counter() - run_start

    iq = np.fromfile(raw_out, dtype=np.complex64)
    settle_samples = max(0, int(round(args.sample_rate * ofdm_cfg.rx_settle_ms / 1000.0)))
    if settle_samples > 0 and iq.size > settle_samples + profile.frame_samples:
        iq = iq[settle_samples:]
    else:
        if settle_samples > 0:
            print(
                f"warning: capture ({iq.size} samples) is too short to drop "
                f"rx_settle_samples={settle_samples}; keeping the whole capture"
            )
        settle_samples = 0
    print(f"raw_iq={raw_out} captured_iq={iq.size} rx_settle_samples={settle_samples}")
    print(format_rx_level_report(iq))

    decoded_frames = decode_iq_stream_diagnostics(
        iq,
        profile,
        frame_byte_len=frame_byte_len,
        threshold_factor=ofdm_cfg.threshold_factor,
        correlation_plot_out=correlation_out,
        correlation_plot_title="TRX Preamble Matched Filter",
    )
    frame_bytes_list = [item.frame_bytes for item in decoded_frames]
    sync_indices = [item.sync_index for item in decoded_frames]
    print(f"correlation_out={correlation_out}")

    constellation_path = None
    if decoded_frames:
        plot_frames = decoded_frames[: args.constellation_frames]
        plot_symbols = np.concatenate([item.recovery.equalized_symbols for item in plot_frames])
        if plot_symbols.size:
            save_constellation_plot(
                plot_symbols,
                constellation_out,
                bits_per_symbol=args.bits_per_symbol,
                title=f"TRX RX Constellation ({MODULATION_LABELS[args.bits_per_symbol]})",
            )
            constellation_path = constellation_out
            constellation_ofdm_symbols = len(plot_frames) * profile.num_ofdm_symbols
            print(
                f"constellation_out={constellation_path} frames={len(plot_frames)} "
                f"ofdm_symbols={constellation_ofdm_symbols} qam_symbols={plot_symbols.size}"
            )
    else:
        print("constellation_out skipped: no detected PHY frames available")
    frames = parse_frames(b"".join(frame_bytes_list), verify_frame_crc=not crc_debug)
    frame_crc_ok = sum(1 for frame in frames if frame.frame_crc_ok)
    recovered_bytes, received_chunks = _recover_debug_pixels(
        frames,
        total_size=len(payload_bytes),
        payload_size=args.payload_size,
    )
    complete = received_chunks == len(build.frame_bytes)
    file_crc_ok = (zlib.crc32(recovered_bytes) & 0xFFFFFFFF) == (zlib.crc32(payload_bytes) & 0xFFFFFFFF)
    raw_rgb_path = None
    png_path = None
    file_output_path = None
    if payload_mode == "raw-rgb" and image_payload is not None and (crc_debug or (complete and file_crc_ok)):
        raw_rgb_path, png_path = _write_raw_rgb_outputs(image_payload, recovered_bytes, args.output_dir)
    elif payload_mode == "file" and complete and (file_crc_ok or crc_debug):
        file_output_path = _write_file_output(
            recovered_bytes,
            args.output_dir,
            _file_output_name(args.input, args.output_name),
        )

    print(
        f"detected_frames={len(frame_bytes_list)} parsed_frames={len(frames)} "
        f"frame_crc_ok={frame_crc_ok} chunks={received_chunks}/{len(build.frame_bytes)} "
        f"complete={complete} file_crc_ok={file_crc_ok} crc_mode={args.crc_mode}"
    )
    if raw_rgb_path is not None and png_path is not None:
        print(f"raw_payload_output={raw_rgb_path}")
        print(f"output={png_path}")
    elif file_output_path is not None:
        print(f"output={file_output_path}")
    elif args.crc_mode == "strict":
        print("strict_output=skipped because CRC validation did not pass")

    summary = {
        "version": __version__,
        "input": str(args.input),
        "input_sha256": _file_sha256(args.input),
        "input_size": int(args.input.stat().st_size),
        "payload_mode": payload_mode,
        "image_width": image_payload.width if image_payload is not None else None,
        "image_height": image_payload.height if image_payload is not None else None,
        "payload_bytes": len(payload_bytes),
        "payload_size": args.payload_size,
        "metadata_size": args.metadata_size,
        "repeats": args.repeats,
        "continuous": args.continuous,
        "staged_cycles": staged_cycles,
        "one_cycle_seconds": one_cycle_seconds,
        "warmup_seconds": warmup_seconds,
        "capture_seconds": capture_seconds,
        "tx_warmup_ms": args.tx_warmup_ms,
        "rx_settle_ms": args.rx_settle_ms,
        "threshold_factor": ofdm_cfg.threshold_factor,
        "bits_per_symbol": args.bits_per_symbol,
        "sample_rate": args.sample_rate,
        "freq": args.freq,
        "tx_gain": args.tx_gain,
        "rx_gain": args.rx_gain,
        "amplitude": args.amplitude,
        "papr_clip": args.papr_clip,
        "papr_report": papr_summary,
        "duration": args.duration,
        "crc_mode": args.crc_mode,
        "raw_out": str(raw_out),
        "correlation_out": str(correlation_out),
        "constellation_out": str(constellation_path) if constellation_path is not None else None,
        "constellation_frames": len(plot_frames) if decoded_frames else 0,
        "constellation_ofdm_symbols": (
            len(plot_frames) * profile.num_ofdm_symbols if decoded_frames else 0
        ),
        "constellation_qam_symbols": int(plot_symbols.size) if decoded_frames else 0,
        "raw_payload_output": str(raw_rgb_path) if raw_rgb_path is not None else None,
        "output": str(png_path or file_output_path) if (png_path is not None or file_output_path is not None) else None,
        "detected_frames": len(frame_bytes_list),
        "sync_indices": sync_indices[:20],
        "parsed_frames": len(frames),
        "frame_crc_ok": frame_crc_ok,
        "received_chunks": received_chunks,
        "chunk_count": len(build.frame_bytes),
        "complete": complete,
        "file_crc_ok": file_crc_ok,
        "setup_elapsed": setup_elapsed,
        "run_elapsed": run_elapsed,
    }
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={summary_json}")

    if not frames:
        raise SystemExit("no protocol frames decoded")
    if received_chunks == 0:
        raise SystemExit(2)
    if args.crc_mode == "strict" and (not complete or not file_crc_ok):
        raise SystemExit(2)


if __name__ == "__main__":
    main()