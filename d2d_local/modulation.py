"""QAM mapping: ``coded_bits[B] -> data_symbols[Q]``."""

import numpy as np

from .interfaces import bit_vector, complex_vector
from .ofdm_conf import OfdmConfig
from .qam import bits_to_symbols


def modulate_bits(coded_bits: np.ndarray, config: OfdmConfig) -> np.ndarray:
    bits = bit_vector(coded_bits, "coded_bits")
    symbols = bits_to_symbols(bits, config.bits_per_symbol)
    return complex_vector(symbols, "data_symbols")
