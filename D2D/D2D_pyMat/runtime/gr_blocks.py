"""Custom GNU Radio blocks wrapping the D2D_pyMat explicit PHY.

Imported lazily by ``gr_flowgraphs``; importing this module requires
``gnuradio``. Keeping the block definition isolated makes it easy to swap
the per-frame encoder (e.g. add FEC) without touching the flowgraph.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from gnuradio import gr  # type: ignore

from ..ofdm import PhyProfile, encode_frame
from ..protocol import bytes_to_bits


class PyMatFrameSource(gr.sync_block):
    """Streaming IQ source: encodes one PHY frame at a time, on demand.

    Memory footprint is O(one frame's IQ + raw protocol-frame bytes), not
    O(whole file). Frames are encoded lazily instead of materializing the full
    transfer as a Python list.
    """

    def __init__(
        self,
        frame_bytes_list: Iterable[bytes],
        profile: PhyProfile,
        *,
        loop: bool = True,
    ) -> None:
        gr.sync_block.__init__(
            self,
            name="PyMatFrameSource",
            in_sig=None,
            out_sig=[np.complex64],
        )
        frames = [bytes(b) for b in frame_bytes_list]
        if not frames:
            raise ValueError("frame_bytes_list is empty")
        self._frames = frames
        self._profile = profile
        self._loop = bool(loop)
        self._idx = 0
        self._pos = 0
        self._cached_iq: np.ndarray | None = None
        self._done = False

    def _current_iq(self) -> np.ndarray:
        if self._cached_iq is None:
            bits = bytes_to_bits(self._frames[self._idx])
            self._cached_iq = encode_frame(bits, self._profile).astype(np.complex64)
        return self._cached_iq

    def work(self, input_items, output_items):  # noqa: D401 - GR API
        out = output_items[0]
        n = len(out)
        produced = 0
        while produced < n:
            if self._done:
                return -1 if produced == 0 else produced
            frame = self._current_iq()
            remain = frame.size - self._pos
            take = min(remain, n - produced)
            out[produced : produced + take] = frame[self._pos : self._pos + take]
            produced += take
            self._pos += take
            if self._pos >= frame.size:
                self._pos = 0
                self._cached_iq = None
                self._idx += 1
                if self._idx >= len(self._frames):
                    if self._loop:
                        self._idx = 0
                    else:
                        self._done = True
        return produced
