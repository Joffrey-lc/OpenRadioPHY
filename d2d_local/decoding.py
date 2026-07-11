"""RX bit processing: ``coded_bits[B] -> protocol_bits[B]``."""

import numpy as np

from .interfaces import bit_vector
from .interleaver import deinterleave_bits, scramble_bits
from .ofdm_conf import PhyProfile


def decode_bits(coded_bits: np.ndarray, profile: PhyProfile) -> np.ndarray:
    bits = bit_vector(coded_bits, "coded_bits")
    if bits.size != profile.padded_bit_count:
        raise ValueError(f"coded_bits must contain {profile.padded_bit_count} values")
    cfg = profile.config
    decoded = deinterleave_bits(bits, cfg.interleaver, rows=cfg.interleaver_rows)
    decoded = scramble_bits(decoded, enabled=cfg.scrambler, seed=cfg.scrambler_seed)
    return bit_vector(decoded, "protocol_bits")
