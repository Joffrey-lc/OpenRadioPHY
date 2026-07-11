"""OpenRadioPHY public Python physical-layer package."""

from .capture import CaptureBundle, CaptureManifest, load_ota_capture
from .config import ChannelConfig, OfdmConfig, ProtocolConfig, SimulationConfig
from .pipeline import DecodeOptions, DecodeResult, decode_capture
from .phy_chain import OfdmReceiver, OfdmTransmitter, RxModules, TxModules
from .interfaces import RxFrameTrace, TxFrameTrace
from .simulation import simulate_capture

__version__ = "0.1.0"

__all__ = [
    "CaptureBundle",
    "CaptureManifest",
    "ChannelConfig",
    "DecodeOptions",
    "DecodeResult",
    "OfdmConfig",
    "OfdmReceiver",
    "OfdmTransmitter",
    "ProtocolConfig",
    "SimulationConfig",
    "RxFrameTrace",
    "RxModules",
    "TxFrameTrace",
    "TxModules",
    "decode_capture",
    "load_ota_capture",
    "simulate_capture",
    "__version__",
]
