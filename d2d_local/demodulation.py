"""QAM hard detector: ``data_symbols[Q] -> coded_bits[B]``."""

import numpy as np

from .interfaces import bit_vector, complex_vector
from .ofdm_conf import PhyProfile
from .qam import symbols_to_bits_hard


def demodulate_symbols(data_symbols: np.ndarray, profile: PhyProfile) -> np.ndarray:
    symbols = complex_vector(data_symbols, "data_symbols")
    bits = symbols_to_bits_hard(symbols, profile.config.bits_per_symbol)
    return bit_vector(bits[: profile.padded_bit_count], "coded_bits")
