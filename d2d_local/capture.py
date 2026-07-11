"""IQ capture containers and manifest handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .config import OfdmConfig, ProtocolConfig


SCHEMA_VERSION = 1
SOURCE_TYPE_OTA = "point_to_point_ota"
MODULATION_LABELS = {1: "BPSK", 2: "QPSK", 4: "16QAM", 6: "64QAM"}


def sha256_file(path: Path, *, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _strict_keys(
    value: Mapping[str, Any],
    required: set[str],
    name: str,
    *,
    optional: set[str] | None = None,
) -> None:
    allowed = required | (optional or set())
    unknown = set(value) - allowed
    missing = required - set(value)
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {sorted(unknown)}")
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}")


@dataclass(frozen=True)
class PayloadExpectation:
    filename: str
    media_type: str
    size: int
    sha256: str
    encoding: str = "file"
    width: int | None = None
    height: int | None = None

    def validate(self) -> None:
        if Path(self.filename).name != self.filename or not self.filename:
            raise ValueError("payload filename must be a plain filename")
        if not self.media_type:
            raise ValueError("payload media_type must not be empty")
        if self.size < 0:
            raise ValueError("payload size must be non-negative")
        if len(self.sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.sha256):
            raise ValueError("payload sha256 must be 64 lowercase hexadecimal characters")
        if self.encoding not in {"file", "raw_rgb"}:
            raise ValueError("payload encoding must be 'file' or 'raw_rgb'")
        if self.encoding == "raw_rgb":
            if self.width is None or self.height is None or self.width <= 0 or self.height <= 0:
                raise ValueError("raw_rgb payload requires positive width and height")
            if self.size != self.width * self.height * 3:
                raise ValueError("raw_rgb payload size must equal width * height * 3")
            if self.media_type != "image/rgb" or Path(self.filename).suffix.lower() != ".png":
                raise ValueError("raw_rgb payload requires image/rgb and a .png output filename")


@dataclass(frozen=True)
class CaptureBundle:
    samples: np.ndarray
    ofdm_config: OfdmConfig
    protocol_config: ProtocolConfig
    provenance: Mapping[str, Any]
    expected_payload: PayloadExpectation | None = None
    tx_samples: np.ndarray | None = None

    def validate(self) -> None:
        samples = np.asarray(self.samples)
        if samples.ndim != 1 or not np.iscomplexobj(samples):
            raise ValueError("capture samples must be a one-dimensional complex array")
        if samples.size == 0:
            raise ValueError("capture samples must not be empty")
        self.ofdm_config.validate()
        self.protocol_config.validate()
        if self.expected_payload is not None:
            self.expected_payload.validate()


@dataclass(frozen=True)
class CaptureManifest:
    capture_id: str
    source_type: str
    modulation: str
    iq_file: str
    iq_sha256: str
    dtype: str
    byte_order: str
    sample_count: int
    sample_rate_hz: float
    ofdm_config: OfdmConfig
    protocol_config: ProtocolConfig
    payload: PayloadExpectation

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CaptureManifest":
        raw = _mapping(raw, "manifest")
        _strict_keys(
            raw,
            {
                "schema_version",
                "capture_id",
                "source_type",
                "modulation",
                "iq",
                "sample_rate_hz",
                "ofdm",
                "protocol",
                "payload",
            },
            "manifest",
        )
        if raw["schema_version"] != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported manifest schema_version={raw['schema_version']!r}; "
                f"expected {SCHEMA_VERSION}"
            )

        iq = _mapping(raw["iq"], "manifest.iq")
        _strict_keys(iq, {"file", "sha256", "dtype", "byte_order", "sample_count"}, "manifest.iq")
        ofdm_raw = _mapping(raw["ofdm"], "manifest.ofdm")
        ofdm_allowed = {field.name for field in fields(OfdmConfig)}
        _strict_keys(ofdm_raw, ofdm_allowed, "manifest.ofdm")
        protocol_raw = _mapping(raw["protocol"], "manifest.protocol")
        protocol_allowed = {field.name for field in fields(ProtocolConfig)}
        _strict_keys(protocol_raw, protocol_allowed, "manifest.protocol")
        payload_raw = _mapping(raw["payload"], "manifest.payload")
        _strict_keys(
            payload_raw,
            {"filename", "media_type", "size", "sha256"},
            "manifest.payload",
            optional={"encoding", "width", "height"},
        )

        manifest = cls(
            capture_id=str(raw["capture_id"]),
            source_type=str(raw["source_type"]),
            modulation=str(raw["modulation"]),
            iq_file=str(iq["file"]),
            iq_sha256=str(iq["sha256"]),
            dtype=str(iq["dtype"]),
            byte_order=str(iq["byte_order"]),
            sample_count=int(iq["sample_count"]),
            sample_rate_hz=float(raw["sample_rate_hz"]),
            ofdm_config=OfdmConfig(**dict(ofdm_raw)),
            protocol_config=ProtocolConfig(**dict(protocol_raw)),
            payload=PayloadExpectation(**dict(payload_raw)),
        )
        manifest.validate()
        return manifest

    @classmethod
    def read(cls, path: str | Path) -> "CaptureManifest":
        path = Path(path).expanduser().resolve()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as error:
            raise FileNotFoundError(f"OTA manifest not found: {path}") from error
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid OTA manifest JSON at {path}: {error}") from error
        return cls.from_dict(raw)

    def validate(self) -> None:
        if not self.capture_id or any(ch not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for ch in self.capture_id):
            raise ValueError("capture_id must use lowercase letters, numbers, '-' or '_'")
        if self.source_type != SOURCE_TYPE_OTA:
            raise ValueError(f"source_type must be {SOURCE_TYPE_OTA!r}")
        if Path(self.iq_file).name != self.iq_file or not self.iq_file:
            raise ValueError("iq.file must be a plain filename next to the manifest")
        if self.dtype != "complex64":
            raise ValueError("iq.dtype must be 'complex64'")
        if self.byte_order != "little":
            raise ValueError("iq.byte_order must be 'little'")
        if self.sample_count <= 0:
            raise ValueError("iq.sample_count must be positive")
        if len(self.iq_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in self.iq_sha256):
            raise ValueError("iq.sha256 must be 64 lowercase hexadecimal characters")
        if self.sample_rate_hz <= 0.0:
            raise ValueError("sample_rate_hz must be positive")
        if abs(self.ofdm_config.sample_rate - self.sample_rate_hz) > 1e-6:
            raise ValueError("sample_rate_hz must match ofdm.sample_rate")
        expected_modulation = MODULATION_LABELS.get(self.ofdm_config.bits_per_symbol)
        if self.modulation != expected_modulation:
            raise ValueError(
                f"modulation={self.modulation!r} does not match "
                f"bits_per_symbol={self.ofdm_config.bits_per_symbol}"
            )
        self.ofdm_config.validate()
        self.protocol_config.validate()
        self.payload.validate()

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "filename": self.payload.filename,
            "media_type": self.payload.media_type,
            "size": self.payload.size,
            "sha256": self.payload.sha256,
        }
        if self.payload.encoding == "raw_rgb":
            payload.update(
                {
                    "encoding": self.payload.encoding,
                    "width": self.payload.width,
                    "height": self.payload.height,
                }
            )
        return {
            "schema_version": SCHEMA_VERSION,
            "capture_id": self.capture_id,
            "source_type": self.source_type,
            "modulation": self.modulation,
            "iq": {
                "file": self.iq_file,
                "sha256": self.iq_sha256,
                "dtype": self.dtype,
                "byte_order": self.byte_order,
                "sample_count": self.sample_count,
            },
            "sample_rate_hz": self.sample_rate_hz,
            "ofdm": asdict(self.ofdm_config),
            "protocol": asdict(self.protocol_config),
            "payload": payload,
        }


def load_ota_capture(manifest_path: str | Path) -> CaptureBundle:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = CaptureManifest.read(manifest_path)
    iq_path = manifest_path.parent / manifest.iq_file
    if not iq_path.is_file():
        raise FileNotFoundError(f"OTA IQ file not found: {iq_path}")
    expected_bytes = manifest.sample_count * np.dtype("<c8").itemsize
    actual_bytes = iq_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"OTA IQ size mismatch for {iq_path}: expected {expected_bytes} bytes, "
            f"found {actual_bytes}"
        )
    actual_sha256 = sha256_file(iq_path)
    if actual_sha256 != manifest.iq_sha256:
        raise ValueError(
            f"OTA IQ SHA-256 mismatch for {iq_path}: expected {manifest.iq_sha256}, "
            f"found {actual_sha256}"
        )
    samples = np.memmap(iq_path, dtype="<c8", mode="r", shape=(manifest.sample_count,))
    bundle = CaptureBundle(
        samples=samples,
        ofdm_config=manifest.ofdm_config,
        protocol_config=manifest.protocol_config,
        provenance={
            "source_type": manifest.source_type,
            "capture_id": manifest.capture_id,
            "modulation": manifest.modulation,
            "manifest": str(manifest_path),
            "iq_file": str(iq_path),
        },
        expected_payload=manifest.payload,
    )
    bundle.validate()
    return bundle
