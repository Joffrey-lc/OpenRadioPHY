"""GNU Radio flowgraph for the single-USRP D2D_pyMat TRX demo."""

from __future__ import annotations

import time
from typing import Callable

_LOCAL_UHD_CHANNEL = 0


def _require_gr():
    from gnuradio import blocks, gr, uhd  # type: ignore

    return blocks, gr, uhd


def _create_uhd_block(
    factory: Callable[[], object],
    *,
    block_name: str,
    device_addr: str,
    channel: int,
    retries: int = 3,
    retry_delay: float = 0.6,
) -> object:
    last_error: RuntimeError | None = None
    for attempt in range(1, retries + 1):
        try:
            return factory()
        except RuntimeError as err:
            last_error = err
            if attempt == retries:
                break
            print(
                f"warning: {block_name} init failed for {device_addr} channel={channel} "
                f"(attempt {attempt}/{retries}): {err}"
            )
            time.sleep(retry_delay * attempt)
    raise RuntimeError(
        f"failed to initialize {block_name} for {device_addr} channel={channel} "
        f"after {retries} attempts"
    ) from last_error


def _configure_reference(usrp, *, clock_source: str, time_source: str) -> None:
    usrp.set_clock_source(clock_source, 0)
    usrp.set_time_source(time_source, 0)


def _close_file_sink(file_sink) -> None:
    try:
        file_sink.close()
    except Exception as err:
        print(f"warning: failed to close file sink cleanly: {err}")


class PyMatUsrpTrx:
    """Stream D2D_pyMat frames and capture received IQ with one USRP."""

    def __init__(
        self,
        frame_bytes_list,
        profile,
        raw_out_path: str,
        *,
        device_addr: str,
        sample_rate: float,
        center_freq: float,
        tx_gain: float,
        rx_gain: float,
        amplitude: float = 0.2,
        clock_source: str = "internal",
        time_source: str = "internal",
        tx_antenna: str | None = "TX/RX",
        rx_antenna: str | None = "RX2",
        tx_channel: int = 0,
        rx_channel: int = 0,
        usrp_init_retries: int = 3,
    ) -> None:

        blocks, gr, uhd = _require_gr()
        from .gr_blocks import PyMatFrameSource

        self.tb = gr.top_block("D2D_pyMat_TRX")
        self.source = PyMatFrameSource(frame_bytes_list, profile, loop=False)
        self.amp = blocks.multiply_const_cc(float(amplitude))

        self.usrp_sink = _create_uhd_block(
            lambda: uhd.usrp_sink(
                device_addr,
                uhd.stream_args(cpu_format="fc32", args="", channels=[tx_channel]),
            ),
            block_name="usrp_sink",
            device_addr=device_addr,
            channel=tx_channel,
            retries=usrp_init_retries,
        )
        _configure_reference(self.usrp_sink, clock_source=clock_source, time_source=time_source)
        self.usrp_sink.set_samp_rate(sample_rate)
        if not self.usrp_sink.set_center_freq(uhd.tune_request(center_freq), _LOCAL_UHD_CHANNEL):
            raise RuntimeError(f"failed to set TX center_freq={center_freq}")
        self.usrp_sink.set_gain(tx_gain, _LOCAL_UHD_CHANNEL)
        if tx_antenna:
            self.usrp_sink.set_antenna(tx_antenna, _LOCAL_UHD_CHANNEL)

        self.usrp_source = _create_uhd_block(
            lambda: uhd.usrp_source(
                device_addr,
                uhd.stream_args(cpu_format="fc32", args="", channels=[rx_channel]),
            ),
            block_name="usrp_source",
            device_addr=device_addr,
            channel=rx_channel,
            retries=usrp_init_retries,
        )
        _configure_reference(self.usrp_source, clock_source=clock_source, time_source=time_source)
        self.usrp_source.set_samp_rate(sample_rate)
        if not self.usrp_source.set_center_freq(uhd.tune_request(center_freq), _LOCAL_UHD_CHANNEL):
            raise RuntimeError(f"failed to set RX center_freq={center_freq}")
        self.usrp_source.set_gain(rx_gain, _LOCAL_UHD_CHANNEL)
        if rx_antenna:
            self.usrp_source.set_antenna(rx_antenna, _LOCAL_UHD_CHANNEL)

        self.file_sink = blocks.file_sink(gr.sizeof_gr_complex, raw_out_path, False)
        self.file_sink.set_unbuffered(False)

        self.tb.connect(self.source, self.amp, self.usrp_sink)
        self.tb.connect(self.usrp_source, self.file_sink)

    def run_for(self, seconds: float) -> None:
        self.tb.start()
        try:
            time.sleep(seconds)
        finally:
            self.tb.stop()
            self.tb.wait()
            _close_file_sink(self.file_sink)