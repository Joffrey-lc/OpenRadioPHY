"""Composable OFDM transmitter and receiver."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .add_cp import add_cp
from .channel_estimation import estimate_channel
from .coding import encode_bits
from .decoding import decode_bits
from .demodulation import demodulate_symbols
from .dft import dft
from .equalization import equalize
from .frame_assembly import assemble_frame
from .frame_sync import synchronize_frame
from .frequency_sync import correct_frequency_offset
from .idft import idft
from .interfaces import RxFrameTrace, TxFrameTrace
from .modulation import modulate_bits
from .ofdm_conf import PhyProfile
from .parallel_to_serial import serialize_symbols
from .remove_cp import remove_cp
from .resource_demapping import extract_data_symbols
from .resource_mapping import map_to_resource_grid
from .serial_to_parallel import deserialize_symbols
from .timing_sync import align_payload


@dataclass(frozen=True)
class TxModules:
    coding: Callable = encode_bits
    modulation: Callable = modulate_bits
    resource_mapping: Callable = map_to_resource_grid
    idft: Callable = idft
    add_cp: Callable = add_cp
    parallel_to_serial: Callable = serialize_symbols
    frame_assembly: Callable = assemble_frame


@dataclass(frozen=True)
class RxModules:
    frame_sync: Callable = synchronize_frame
    timing_sync: Callable = align_payload
    frequency_sync: Callable = correct_frequency_offset
    serial_to_parallel: Callable = deserialize_symbols
    remove_cp: Callable = remove_cp
    dft: Callable = dft
    channel_estimation: Callable = estimate_channel
    equalization: Callable = equalize
    resource_demapping: Callable = extract_data_symbols
    demodulation: Callable = demodulate_symbols
    decoding: Callable = decode_bits


class OfdmTransmitter:
    def __init__(self, modules: TxModules | None = None) -> None:
        self.modules = modules or TxModules()

    def forward(self, protocol_bits: np.ndarray, profile: PhyProfile) -> TxFrameTrace:
        coded_bits = self.modules.coding(protocol_bits, profile)
        data_symbols = self.modules.modulation(coded_bits, profile.config)
        resource_grid = self.modules.resource_mapping(data_symbols, profile)
        time_symbols = self.modules.idft(resource_grid)
        symbols_with_cp = self.modules.add_cp(time_symbols, profile.config.cp_len)
        payload_stream = self.modules.parallel_to_serial(symbols_with_cp)
        waveform = self.modules.frame_assembly(
            profile.preamble,
            payload_stream,
            normalize=profile.config.normalize_waveform,
            peak_clip=profile.config.payload_peak_clip,
        )
        return TxFrameTrace(
            profile=profile,
            protocol_bits=np.asarray(protocol_bits, dtype=np.uint8).reshape(-1),
            coded_bits=coded_bits,
            data_symbols=data_symbols,
            resource_grid=resource_grid,
            time_symbols=time_symbols,
            symbols_with_cp=symbols_with_cp,
            payload_stream=payload_stream,
            waveform=waveform,
        )

    __call__ = forward


class OfdmReceiver:
    def __init__(self, modules: RxModules | None = None) -> None:
        self.modules = modules or RxModules()

    def forward(
        self,
        received_frame: np.ndarray,
        profile: PhyProfile,
        *,
        sync_search: bool = True,
    ) -> RxFrameTrace:
        coarse = self.modules.frame_sync(received_frame, profile, search=sync_search)
        aligned, timing_offset = self.modules.timing_sync(
            coarse.payload_with_margin, profile
        )
        corrected, cfo = self.modules.frequency_sync(aligned, profile)
        symbols_with_cp = self.modules.serial_to_parallel(
            corrected, profile.symbol_span, profile.num_ofdm_symbols
        )
        time_symbols = self.modules.remove_cp(symbols_with_cp, profile.config.cp_len)
        frequency_grid = self.modules.dft(time_symbols)
        channel_grid = self.modules.channel_estimation(frequency_grid, profile)
        equalized_grid = self.modules.equalization(
            frequency_grid, channel_grid, profile
        )
        data_symbols = self.modules.resource_demapping(equalized_grid, profile)
        coded_bits = self.modules.demodulation(data_symbols, profile)
        protocol_bits = self.modules.decoding(coded_bits, profile)
        return RxFrameTrace(
            profile=profile,
            sync_index=coarse.sync_index,
            correlation_metric=coarse.correlation_metric,
            timing_offset=timing_offset,
            cfo_radians_per_sample=cfo,
            symbols_with_cp=symbols_with_cp,
            time_symbols=time_symbols,
            frequency_grid=frequency_grid,
            channel_grid=channel_grid,
            equalized_grid=equalized_grid,
            data_symbols=data_symbols,
            coded_bits=coded_bits,
            protocol_bits=protocol_bits,
        )

    __call__ = forward


def build_default_transmitter() -> OfdmTransmitter:
    return OfdmTransmitter(TxModules())


def build_default_receiver() -> OfdmReceiver:
    return OfdmReceiver(RxModules())
