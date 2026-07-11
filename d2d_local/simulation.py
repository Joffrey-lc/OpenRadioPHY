"""End-to-end image simulation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from .capture import CaptureBundle, PayloadExpectation, sha256_bytes
from .config import SimulationConfig
from .image_payload import load_raw_rgb
from .ofdm import apply_channel
from .protocol import build_protocol_frames, build_protocol_frames_from_bytes
from .receiver import build_tx_frame_set
from .phy_chain import OfdmTransmitter


def simulate_capture(
    input_path: str | Path,
    simulation_config: SimulationConfig | None = None,
    *,
    transmitter: OfdmTransmitter | None = None,
) -> CaptureBundle:
    """Encode a file and pass it through the configured channel."""
    input_path = Path(input_path).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"simulation input not found: {input_path}")
    config = simulation_config or SimulationConfig()
    config.validate()

    if config.payload_mode == "raw_rgb":
        source_bytes, width, height = load_raw_rgb(input_path)
        output_name = f"{input_path.stem}.png"
        media_type = "image/rgb"
    else:
        source_bytes = input_path.read_bytes()
        width = None
        height = None
        output_name = input_path.name
        media_type = None

    source_sha256 = sha256_bytes(source_bytes)
    # Stable ID keeps simulations reproducible.
    if config.payload_mode == "raw_rgb":
        build = build_protocol_frames_from_bytes(
            source_bytes,
            output_name,
            media_type,
            config.protocol,
            input_path=input_path,
            transfer_id=source_sha256[:16],
        )
    else:
        build = build_protocol_frames(input_path, config.protocol, transfer_id=source_sha256[:16])
    frame_set = build_tx_frame_set(
        build.frame_bytes, config.ofdm, repeats=1, transmitter=transmitter
    )
    received = apply_channel(
        frame_set.iq,
        config.channel,
        sample_rate=config.ofdm.sample_rate,
    )
    bundle = CaptureBundle(
        samples=received,
        ofdm_config=config.ofdm,
        protocol_config=config.protocol,
        provenance={
            "source_type": "simulation",
            "input": str(input_path),
            "transfer_id": build.transfer_id,
            "payload_mode": config.payload_mode,
            "channel": asdict(config.channel),
        },
        expected_payload=PayloadExpectation(
            filename=output_name,
            media_type=build.media_type,
            size=len(source_bytes),
            sha256=source_sha256,
            encoding=config.payload_mode,
            width=width,
            height=height,
        ),
        tx_samples=frame_set.iq,
    )
    bundle.validate()
    return bundle
