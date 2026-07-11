"""TX bit processing: ``protocol_bits[Bp] -> coded_bits[B]``."""

import numpy as np

from .interfaces import bit_vector
from .interleaver import interleave_bits, scramble_bits
from .ofdm_conf import PhyProfile


def encode_bits(protocol_bits: np.ndarray, profile: PhyProfile) -> np.ndarray:
    bits = bit_vector(protocol_bits, "protocol_bits")
    if bits.size > profile.padded_bit_count:
        raise ValueError("protocol_bits exceed the configured PHY-frame capacity")
    coded = np.pad(bits, (0, profile.padded_bit_count - bits.size), constant_values=0)
    cfg = profile.config
    coded = scramble_bits(coded, enabled=cfg.scrambler, seed=cfg.scrambler_seed)
    coded = interleave_bits(coded, cfg.interleaver, rows=cfg.interleaver_rows)
    return bit_vector(coded, "coded_bits")
