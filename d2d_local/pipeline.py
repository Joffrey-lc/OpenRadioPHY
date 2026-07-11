"""Common receiver pipeline for simulated and point-to-point OTA IQ."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping
import zlib

import numpy as np

from .capture import CaptureBundle, sha256_bytes, sha256_file
from .image_payload import save_raw_rgb
from .ofdm import build_phy_profile
from .protocol import (
    frame_size as protocol_frame_size,
    parse_frames,
    reassemble_transfer,
    recover_payload_by_position,
)
from .receiver import decode_iq_stream_diagnostics, format_rx_level_report
from .phy_chain import OfdmReceiver
from .viz import save_constellation_plot, save_papr_ccdf_plot


@dataclass(frozen=True)
class DecodeOptions:
    strict: bool = False
    constellation_frames: int = 8
    output_name: str | None = None

    def validate(self) -> None:
        if self.constellation_frames <= 0:
            raise ValueError("constellation_frames must be positive")
        if self.output_name is not None and (
            not self.output_name or Path(self.output_name).name != self.output_name
        ):
            raise ValueError("output_name must be a non-empty plain filename")


@dataclass(frozen=True)
class DecodeResult:
    success: bool
    output_path: Path | None
    summary_path: Path
    correlation_path: Path | None
    constellation_path: Path | None
    papr_path: Path | None
    detected_frames: int
    parsed_frames: int
    crc_ok: bool
    expected_sha256_ok: bool | None


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _papr_report(samples: np.ndarray, path: Path) -> dict[str, Any] | None:
    samples = np.asarray(samples, dtype=np.complex64).reshape(-1)
    if samples.size == 0:
        return None
    power = np.abs(samples).astype(np.float64) ** 2
    average = float(np.mean(power))
    peak = float(np.max(power))
    if average <= 0.0:
        return None
    ratio_db = 10.0 * np.log10(np.maximum(power / average, np.finfo(np.float64).tiny))
    max_points = 200_000
    if ratio_db.size > max_points:
        ratio_db = ratio_db[:: int(math.ceil(ratio_db.size / max_points))]
    save_papr_ccdf_plot(ratio_db, path, title="Simulation TX PAPR CCDF")
    return {
        "samples": int(samples.size),
        "average_power": average,
        "peak_power": peak,
        "papr_db": float(10.0 * math.log10(peak / average)),
        "plot": str(path),
    }


def decode_capture(
    bundle: CaptureBundle,
    output_dir: str | Path,
    options: DecodeOptions | None = None,
    *,
    receiver: OfdmReceiver | None = None,
) -> DecodeResult:
    """Run synchronization through payload reconstruction on a capture."""
    bundle.validate()
    options = options or DecodeOptions()
    options.validate()
    output_dir = Path(output_dir).expanduser().resolve()
    diagnostics_dir = output_dir / "diagnostics"
    recovered_dir = output_dir / "recovered"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    recovered_dir.mkdir(parents=True, exist_ok=True)

    frame_byte_len = protocol_frame_size(
        bundle.protocol_config.metadata_size,
        bundle.protocol_config.payload_size,
    )
    profile = build_phy_profile(bundle.ofdm_config, frame_byte_len * 8)
    correlation_path = diagnostics_dir / "correlation.png"
    constellation_path = diagnostics_dir / "constellation.png"
    papr_path = diagnostics_dir / "papr_ccdf.png"
    decoded = decode_iq_stream_diagnostics(
        bundle.samples,
        profile,
        frame_byte_len=frame_byte_len,
        threshold_factor=bundle.ofdm_config.threshold_factor,
        correlation_plot_out=correlation_path,
        correlation_plot_title="Preamble Matched Filter",
        receiver=receiver,
    )
    if not correlation_path.exists():
        correlation_output: Path | None = None
    else:
        correlation_output = correlation_path

    constellation_output: Path | None = None
    constellation_symbols = 0
    if decoded:
        selected = decoded[: options.constellation_frames]
        symbols = np.concatenate([item.recovery.equalized_symbols for item in selected])
        if symbols.size:
            save_constellation_plot(
                symbols,
                constellation_path,
                bits_per_symbol=bundle.ofdm_config.bits_per_symbol,
                title=f"RX Constellation ({bundle.ofdm_config.bits_per_symbol} bits/symbol)",
            )
            constellation_output = constellation_path
            constellation_symbols = int(symbols.size)

    decoded_frame_bytes = [item.frame_bytes for item in decoded]
    protocol_frames = parse_frames(b"".join(decoded_frame_bytes), verify_frame_crc=False)
    output_name = options.output_name
    expected = bundle.expected_payload
    if output_name is None and expected is not None:
        output_name = expected.filename
    if output_name is None and protocol_frames:
        output_name = Path(protocol_frames[0].filename).name

    positional = None
    if expected is not None:
        positional = recover_payload_by_position(
            decoded_frame_bytes,
            bundle.protocol_config,
            expected.size,
        )

    reassembly = None
    reassembly_summary: dict[str, Any] | None = None
    output_path: Path | None = None
    recovered_payload: bytes | None = None
    expected_sha256_ok: bool | None = None
    payload_sha256: str | None = None
    output_sha256: str | None = None
    frame_crc_ok = 0
    file_crc_ok = False
    crc_ok = False
    received_payload_frames = 0

    if expected is not None and expected.encoding == "raw_rgb":
        assert positional is not None
        assert expected.width is not None and expected.height is not None
        recovered_payload = positional.payload
        payload_sha256 = sha256_bytes(recovered_payload)
        expected_sha256_ok = payload_sha256 == expected.sha256
        raw_output_name = Path(output_name or expected.filename)
        if raw_output_name.suffix.lower() != ".png":
            raw_output_name = raw_output_name.with_suffix(".png")
        output_path = save_raw_rgb(
            recovered_payload,
            expected.width,
            expected.height,
            recovered_dir / raw_output_name.name,
        )
        frame_crc_ok = positional.frame_crc_ok
        received_payload_frames = positional.received_frames
        if positional.file_crc32 is not None:
            file_crc_ok = (
                zlib.crc32(recovered_payload) & 0xFFFFFFFF
            ) == positional.file_crc32
        crc_ok = bool(
            positional.complete
            and positional.frame_crc_ok == positional.expected_frames
            and file_crc_ok
        )
        reassembly_summary = {
            "transfer_id": positional.transfer_id,
            "filename": output_path.name,
            "chunk_count": positional.expected_frames,
            "received_chunks": positional.received_frames,
            "complete": positional.complete,
            "crc_ok": crc_ok,
            "method": "raw_rgb_positional",
        }
    else:
        reassembly = reassemble_transfer(
            protocol_frames,
            recovered_dir,
            output_name=output_name,
            require_file_crc=False,
        )
        if reassembly is not None:
            recovered_payload = reassembly.recovered_bytes
            output_path = reassembly.output_path
            reassembly_summary = {
                "transfer_id": reassembly.transfer_id,
                "filename": reassembly.filename,
                "chunk_count": reassembly.chunk_count,
                "received_chunks": reassembly.received_chunks,
                "complete": reassembly.complete,
                "crc_ok": reassembly.crc_ok,
                "method": "protocol",
            }
            received_payload_frames = reassembly.received_chunks

        if output_path is None and positional is not None and output_name is not None:
            recovered_payload = positional.payload
            output_path = recovered_dir / Path(output_name).name
            output_path.write_bytes(recovered_payload)
            reassembly_summary = {
                "transfer_id": None,
                "filename": output_path.name,
                "chunk_count": positional.expected_frames,
                "received_chunks": positional.received_frames,
                "complete": positional.complete,
                "crc_ok": False,
                "method": "fixed_position",
            }
            received_payload_frames = positional.received_frames

        frame_crc_ok = sum(1 for frame in protocol_frames if frame.frame_crc_ok)
        if positional is not None:
            frame_crc_ok = max(frame_crc_ok, positional.frame_crc_ok)
        if recovered_payload is not None:
            payload_sha256 = sha256_bytes(recovered_payload)
            if expected is not None:
                expected_sha256_ok = payload_sha256 == expected.sha256
        crc_ok = bool(
            reassembly is not None
            and reassembly.complete
            and reassembly.crc_ok
            and frame_crc_ok >= reassembly.chunk_count
        )
        file_crc_ok = bool(reassembly is not None and reassembly.complete and reassembly.crc_ok)

    if output_path is not None:
        output_sha256 = sha256_file(output_path)

    papr_report = None
    papr_output: Path | None = None
    if bundle.tx_samples is not None:
        papr_report = _papr_report(bundle.tx_samples, papr_path)
        if papr_report is not None:
            papr_output = papr_path

    integrity_ok = crc_ok and expected_sha256_ok is not False
    success = bool(
        output_path is not None
        and received_payload_frames > 0
        and (integrity_ok or not options.strict)
    )

    summary_path = output_dir / "summary.json"
    summary = {
        "version": "0.1.0",
        "success": success,
        "output_generated": output_path is not None,
        "crc_ok": crc_ok,
        "crc_status": "passed" if crc_ok else "failed",
        "file_crc_ok": file_crc_ok,
        "integrity_ok": integrity_ok,
        "strict_status": options.strict,
        "provenance": _jsonable(bundle.provenance),
        "sample_count": int(np.asarray(bundle.samples).size),
        "iq_level": format_rx_level_report(bundle.samples),
        "ofdm": asdict(bundle.ofdm_config),
        "protocol": asdict(bundle.protocol_config),
        "detected_frames": len(decoded),
        "sync_indices": [item.sync_index for item in decoded[:20]],
        "parsed_frames": len(protocol_frames),
        "frame_crc_ok": frame_crc_ok,
        "reassembly": reassembly_summary,
        "expected_payload": None
        if bundle.expected_payload is None
        else asdict(bundle.expected_payload),
        "expected_sha256_ok": expected_sha256_ok,
        "payload_sha256": payload_sha256,
        "output_sha256": output_sha256,
        "output": str(output_path) if output_path is not None else None,
        "correlation": str(correlation_output) if correlation_output is not None else None,
        "constellation": str(constellation_output) if constellation_output is not None else None,
        "constellation_symbols": constellation_symbols,
        "papr": papr_report,
    }
    summary_path.write_text(
        json.dumps(_jsonable(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return DecodeResult(
        success=success,
        output_path=output_path,
        summary_path=summary_path,
        correlation_path=correlation_output,
        constellation_path=constellation_output,
        papr_path=papr_output,
        detected_frames=len(decoded),
        parsed_frames=len(protocol_frames),
        crc_ok=crc_ok,
        expected_sha256_ok=expected_sha256_ok,
    )
