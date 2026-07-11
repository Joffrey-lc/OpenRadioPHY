"""OpenRadioPHY command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from d2d_local.add_cp import add_cp
from d2d_local.capture import load_ota_capture
from d2d_local.channel import load_channel_config
from d2d_local.channel_estimation import estimate_channel
from d2d_local.coding import encode_bits
from d2d_local.config import ChannelConfig, ProtocolConfig, SimulationConfig
from d2d_local.decoding import decode_bits
from d2d_local.demodulation import demodulate_symbols
from d2d_local.dft import dft
from d2d_local.equalization import equalize
from d2d_local.frame_assembly import assemble_frame
from d2d_local.frame_sync import synchronize_frame
from d2d_local.frequency_sync import correct_frequency_offset
from d2d_local.idft import idft
from d2d_local.modulation import modulate_bits
from d2d_local.ofdm_conf import OfdmConfig
from d2d_local.parallel_to_serial import serialize_symbols
from d2d_local.phy_chain import OfdmReceiver, OfdmTransmitter, RxModules, TxModules
from d2d_local.pipeline import DecodeOptions, DecodeResult, decode_capture
from d2d_local.remove_cp import remove_cp
from d2d_local.resource_demapping import extract_data_symbols
from d2d_local.resource_mapping import map_to_resource_grid
from d2d_local.result_viewer import show_result_plots
from d2d_local.serial_to_parallel import deserialize_symbols
from d2d_local.simulation import simulate_capture
from d2d_local.timing_sync import align_payload


ROOT = Path(__file__).resolve().parent

TX_MODULES = TxModules(
    coding=encode_bits,
    modulation=modulate_bits,
    resource_mapping=map_to_resource_grid,
    idft=idft,
    add_cp=add_cp,
    parallel_to_serial=serialize_symbols,
    frame_assembly=assemble_frame,
)

RX_MODULES = RxModules(
    frame_sync=synchronize_frame,
    timing_sync=align_payload,
    frequency_sync=correct_frequency_offset,
    serial_to_parallel=deserialize_symbols,
    remove_cp=remove_cp,
    dft=dft,
    channel_estimation=estimate_channel,
    equalization=equalize,
    resource_demapping=extract_data_symbols,
    demodulation=demodulate_symbols,
    decoding=decode_bits,
)

TRANSMITTER = OfdmTransmitter(TX_MODULES)
RECEIVER = OfdmReceiver(RX_MODULES)
TX_CHAIN = "coding -> modulation -> mapping -> IDFT -> CP -> frame"
RX_CHAIN = "sync -> CP removal -> DFT -> estimation -> equalization -> decoding"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode")

    simulate = subparsers.add_parser("simulate", help="run image-to-image simulation")
    simulate.add_argument("input", nargs="?", type=Path, default=ROOT / "samples" / "NJU.jpg")
    simulate.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "main_simulate")
    simulate.add_argument("--bits-per-symbol", type=int, choices=(1, 2, 4, 6), default=2)
    simulate.add_argument("--payload-mode", choices=("raw-rgb", "file"), default="raw-rgb")
    simulate.add_argument("--channel-config", type=Path, default=None)
    simulate.add_argument("--constellation-frames", type=int, default=8)
    simulate.add_argument(
        "--no-show",
        dest="show_results",
        action="store_false",
        help="do not open the result window",
    )
    simulate.set_defaults(show_results=True)

    replay = subparsers.add_parser("replay", help="decode point-to-point measured IQ")
    replay.add_argument(
        "--capture",
        choices=("bpsk", "qpsk", "16qam", "64qam"),
        default="qpsk",
    )
    replay.add_argument("--manifest", type=Path, default=None)
    replay.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "main_replay")
    replay.add_argument("--constellation-frames", type=int, default=8)
    replay.add_argument(
        "--no-show",
        dest="show_results",
        action="store_false",
        help="do not open the result window",
    )
    replay.set_defaults(show_results=True)
    return parser


def show_results(result: DecodeResult, *, enabled: bool, title: str) -> None:
    if not enabled:
        return
    try:
        show_result_plots(result, title=title)
    except Exception as exc:
        print(f"result window unavailable: {exc}", file=sys.stderr)


def run_simulation(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"input image not found: {input_path}")
    channel = (
        load_channel_config(args.channel_config)
        if args.channel_config is not None
        else ChannelConfig(snr_db=25.0, time_offset=32, seed=20260710)
    )
    config = SimulationConfig(
        protocol=ProtocolConfig(payload_size=1024, metadata_size=128),
        ofdm=OfdmConfig(bits_per_symbol=args.bits_per_symbol, sample_rate=2e6),
        channel=channel,
        payload_mode=args.payload_mode.replace("-", "_"),
    )
    bundle = simulate_capture(input_path, config, transmitter=TRANSMITTER)
    result = decode_capture(
        bundle,
        args.output_dir,
        DecodeOptions(constellation_frames=args.constellation_frames),
        receiver=RECEIVER,
    )
    print(f"TX: {TX_CHAIN}")
    print(f"RX: {RX_CHAIN}")
    print(f"success={result.success} crc_ok={result.crc_ok} summary={result.summary_path}")
    if result.output_path:
        print(f"output={result.output_path}")
    show_results(
        result,
        enabled=args.show_results,
        title=f"OpenRadioPHY Simulation ({args.bits_per_symbol} bits/symbol)",
    )
    return 0 if result.success else 2


def run_replay(args: argparse.Namespace) -> int:
    manifest = args.manifest or ROOT / "samples" / "ota" / args.capture / "manifest.json"
    bundle = load_ota_capture(manifest)
    capture_slug = {1: "bpsk", 2: "qpsk", 4: "16qam", 6: "64qam"}[
        bundle.ofdm_config.bits_per_symbol
    ]
    result = decode_capture(
        bundle,
        args.output_dir / capture_slug,
        DecodeOptions(constellation_frames=args.constellation_frames),
        receiver=RECEIVER,
    )
    print(f"RX: {RX_CHAIN}")
    print(f"success={result.success} crc_ok={result.crc_ok} summary={result.summary_path}")
    if result.output_path:
        print(f"output={result.output_path}")
    show_results(
        result,
        enabled=args.show_results,
        title=f"OpenRadioPHY OTA Decode ({capture_slug})",
    )
    return 0 if result.success else 2


def main(argv: list[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    if not values:
        values = ["simulate"]
    args = build_parser().parse_args(values)
    return run_replay(args) if args.mode == "replay" else run_simulation(args)


if __name__ == "__main__":
    raise SystemExit(main())
