"""Protocol framing, parsing, CRC, and file reassembly."""

from __future__ import annotations

from dataclasses import dataclass
import mimetypes
from pathlib import Path
import secrets
import struct
from typing import Iterable
import zlib

import numpy as np

from .config import ProtocolConfig

MAGIC = b"PMAT"
VERSION = 1
PREFIX = struct.Struct("!4sBHH")
HEADER_FIXED = struct.Struct("!8sIIQIHBB")
CRC = struct.Struct("!I")


@dataclass(frozen=True)
class ProtocolFrame:
    transfer_id: str
    chunk_index: int
    chunk_count: int
    total_size: int
    file_crc32: int
    valid_len: int
    filename: str
    media_type: str
    payload: bytes
    frame_crc_ok: bool


@dataclass(frozen=True)
class ProtocolBuildResult:
    transfer_id: str
    input_path: Path
    source_bytes: bytes
    frame_bytes: tuple[bytes, ...]
    metadata_size: int
    payload_size: int
    media_type: str


@dataclass(frozen=True)
class ReassemblyResult:
    transfer_id: str
    filename: str
    media_type: str
    chunk_count: int
    received_chunks: int
    total_size: int
    complete: bool
    crc_ok: bool
    output_path: Path | None
    recovered_bytes: bytes | None


@dataclass(frozen=True)
class PositionalPayloadRecovery:
    payload: bytes
    expected_frames: int
    received_frames: int
    frame_crc_ok: int
    complete: bool
    transfer_id: str | None
    file_crc32: int | None


def detect_media_type(path: Path) -> str:
    guessed, _ = mimetypes.guess_type(path.name)
    if guessed:
        return guessed
    if path.suffix.lower() in {".txt", ".md", ".csv", ".json"}:
        return "text/plain"
    return "application/octet-stream"


def bytes_to_bits(data: bytes) -> np.ndarray:
    if not data:
        return np.empty(0, dtype=np.uint8)
    return np.unpackbits(np.frombuffer(data, dtype=np.uint8))


def bits_to_bytes(bits: np.ndarray) -> bytes:
    if bits.size == 0:
        return b""
    trimmed = np.asarray(bits, dtype=np.uint8).reshape(-1)
    pad = (-trimmed.size) % 8
    if pad:
        trimmed = np.pad(trimmed, (0, pad), constant_values=0)
    return np.packbits(trimmed).tobytes()


def frame_size(metadata_size: int, payload_size: int) -> int:
    return PREFIX.size + metadata_size + payload_size + CRC.size


def required_metadata_size(input_path: Path) -> int:
    """Minimum metadata_size needed to fit the header + filename + media_type."""
    filename_blob = input_path.name.encode("utf-8")
    media_type_blob = detect_media_type(input_path).encode("ascii")
    return HEADER_FIXED.size + len(filename_blob) + len(media_type_blob)


def build_protocol_frames(
    input_path: Path,
    config: ProtocolConfig,
    *,
    transfer_id: str | bytes | None = None,
) -> ProtocolBuildResult:
    config.validate()
    source_bytes = input_path.read_bytes()
    media_type = detect_media_type(input_path)
    if transfer_id is None:
        transfer_id_bytes = secrets.token_bytes(8)
    elif isinstance(transfer_id, str):
        try:
            transfer_id_bytes = bytes.fromhex(transfer_id)
        except ValueError as err:
            raise ValueError("transfer_id must be a 16-character hex string") from err
    else:
        transfer_id_bytes = bytes(transfer_id)
    if len(transfer_id_bytes) != 8:
        raise ValueError("transfer_id must be exactly 8 bytes")
    transfer_id = transfer_id_bytes.hex()
    chunk_count = max(1, (len(source_bytes) + config.payload_size - 1) // config.payload_size)
    file_crc32 = zlib.crc32(source_bytes) & 0xFFFFFFFF

    filename_blob = input_path.name.encode("utf-8")
    media_type_blob = media_type.encode("ascii")
    required_metadata = HEADER_FIXED.size + len(filename_blob) + len(media_type_blob)
    if config.metadata_size < required_metadata:
        raise ValueError(
            f"metadata_size={config.metadata_size} is too small for this file; "
            f"need at least {required_metadata} bytes (header={HEADER_FIXED.size} + "
            f"filename={len(filename_blob)} + media_type={len(media_type_blob)}). "
            "Increase --metadata-size on BOTH TX and RX to the same value."
        )
    metadata_size = config.metadata_size

    frame_bytes: list[bytes] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * config.payload_size
        payload = source_bytes[start : start + config.payload_size]
        valid_len = len(payload)
        payload_padded = payload.ljust(config.payload_size, b"\x00")

        header_fixed = HEADER_FIXED.pack(
            transfer_id_bytes,
            chunk_index,
            chunk_count,
            len(source_bytes),
            file_crc32,
            valid_len,
            len(filename_blob),
            len(media_type_blob),
        )
        metadata = (header_fixed + filename_blob + media_type_blob).ljust(
            metadata_size, b"\x00"
        )
        prefix = PREFIX.pack(MAGIC, VERSION, metadata_size, config.payload_size)
        frame_crc = zlib.crc32(metadata)
        frame_crc = zlib.crc32(payload_padded, frame_crc) & 0xFFFFFFFF
        frame_bytes.append(prefix + metadata + payload_padded + CRC.pack(frame_crc))

    return ProtocolBuildResult(
        transfer_id=transfer_id,
        input_path=input_path,
        source_bytes=source_bytes,
        frame_bytes=tuple(frame_bytes),
        metadata_size=metadata_size,
        payload_size=config.payload_size,
        media_type=media_type,
    )

def build_protocol_frames_from_bytes(
    source_bytes: bytes,
    source_name: str,
    media_type: str,
    config: ProtocolConfig,
    *,
    input_path: Path | None = None,
    transfer_id: str | bytes | None = None,
) -> ProtocolBuildResult:
    """Build protocol frames from bytes."""
    config.validate()
    source_bytes = bytes(source_bytes)
    filename = Path(source_name).name
    if not filename:
        raise ValueError("source_name must include a filename")
    try:
        media_type_blob = media_type.encode("ascii")
    except UnicodeEncodeError as err:
        raise ValueError("media_type must be ASCII") from err

    if transfer_id is None:
        transfer_id_bytes = secrets.token_bytes(8)
    elif isinstance(transfer_id, str):
        try:
            transfer_id_bytes = bytes.fromhex(transfer_id)
        except ValueError as err:
            raise ValueError("transfer_id must be a 16-character hex string") from err
    else:
        transfer_id_bytes = bytes(transfer_id)
    if len(transfer_id_bytes) != 8:
        raise ValueError("transfer_id must be exactly 8 bytes")
    transfer_id_hex = transfer_id_bytes.hex()

    chunk_count = max(1, (len(source_bytes) + config.payload_size - 1) // config.payload_size)
    file_crc32 = zlib.crc32(source_bytes) & 0xFFFFFFFF
    filename_blob = filename.encode("utf-8")
    required_metadata = HEADER_FIXED.size + len(filename_blob) + len(media_type_blob)
    if config.metadata_size < required_metadata:
        raise ValueError(
            f"metadata_size={config.metadata_size} is too small for this payload; "
            f"need at least {required_metadata} bytes (header={HEADER_FIXED.size} + "
            f"filename={len(filename_blob)} + media_type={len(media_type_blob)}). "
            "Increase --metadata-size on BOTH TX and RX to the same value."
        )
    metadata_size = config.metadata_size

    frame_bytes: list[bytes] = []
    for chunk_index in range(chunk_count):
        start = chunk_index * config.payload_size
        payload = source_bytes[start : start + config.payload_size]
        valid_len = len(payload)
        payload_padded = payload.ljust(config.payload_size, b"\x00")

        header_fixed = HEADER_FIXED.pack(
            transfer_id_bytes,
            chunk_index,
            chunk_count,
            len(source_bytes),
            file_crc32,
            valid_len,
            len(filename_blob),
            len(media_type_blob),
        )
        metadata = (header_fixed + filename_blob + media_type_blob).ljust(
            metadata_size, b"\x00"
        )
        prefix = PREFIX.pack(MAGIC, VERSION, metadata_size, config.payload_size)
        frame_crc = zlib.crc32(metadata)
        frame_crc = zlib.crc32(payload_padded, frame_crc) & 0xFFFFFFFF
        frame_bytes.append(prefix + metadata + payload_padded + CRC.pack(frame_crc))

    return ProtocolBuildResult(
        transfer_id=transfer_id_hex,
        input_path=input_path or Path(filename),
        source_bytes=source_bytes,
        frame_bytes=tuple(frame_bytes),
        metadata_size=metadata_size,
        payload_size=config.payload_size,
        media_type=media_type,
    )

def serialize_frames(frame_bytes: Iterable[bytes]) -> bytes:
    return b"".join(frame_bytes)


def parse_frames(raw: bytes, *, verify_frame_crc: bool = True) -> list[ProtocolFrame]:
    frames: list[ProtocolFrame] = []
    i = 0
    while True:
        pos = raw.find(MAGIC, i)
        if pos < 0 or pos + PREFIX.size > len(raw):
            break
        magic, version, metadata_size, payload_size = PREFIX.unpack_from(raw, pos)
        if magic != MAGIC or version != VERSION:
            i = pos + 1
            continue
        end = pos + frame_size(metadata_size, payload_size)
        if end > len(raw):
            break

        metadata_start = pos + PREFIX.size
        payload_start = metadata_start + metadata_size
        crc_start = payload_start + payload_size

        metadata_padded = raw[metadata_start:payload_start]
        payload_padded = raw[payload_start:crc_start]
        expected_crc = CRC.unpack_from(raw, crc_start)[0]
        actual_crc = zlib.crc32(metadata_padded)
        actual_crc = zlib.crc32(payload_padded, actual_crc) & 0xFFFFFFFF
        frame_crc_ok = actual_crc == expected_crc
        if verify_frame_crc and not frame_crc_ok:
            i = pos + 1
            continue

        try:
            (
                transfer_id,
                chunk_index,
                chunk_count,
                total_size,
                file_crc32,
                valid_len,
                filename_len,
                media_type_len,
            ) = HEADER_FIXED.unpack_from(metadata_padded, 0)
            filename_start = HEADER_FIXED.size
            media_type_start = filename_start + filename_len
            filename_end = media_type_start
            media_type_end = media_type_start + media_type_len
            if media_type_end > metadata_size:
                i = pos + 1
                continue
            filename = metadata_padded[filename_start:filename_end].decode("utf-8")
            media_type = metadata_padded[media_type_start:media_type_end].decode("ascii")
        except (UnicodeDecodeError, struct.error):
            i = pos + 1
            continue

        if valid_len < 0 or valid_len > payload_size:
            i = pos + 1
            continue

        frames.append(
            ProtocolFrame(
                transfer_id=transfer_id.hex(),
                chunk_index=chunk_index,
                chunk_count=chunk_count,
                total_size=total_size,
                file_crc32=file_crc32,
                valid_len=valid_len,
                filename=filename,
                media_type=media_type,
                payload=payload_padded[:valid_len],
                frame_crc_ok=frame_crc_ok,
            )
        )
        i = end
    return frames


def recover_payload_by_position(
    frames: Iterable[bytes],
    config: ProtocolConfig,
    total_size: int,
) -> PositionalPayloadRecovery:
    config.validate()
    if total_size < 0:
        raise ValueError("total_size must be non-negative")

    expected_frames = max(1, (total_size + config.payload_size - 1) // config.payload_size)
    expected_frame_size = frame_size(config.metadata_size, config.payload_size)
    payload_start = PREFIX.size + config.metadata_size
    payload_end = payload_start + config.payload_size
    candidates = [bytes(frame) for frame in frames if len(frame) >= expected_frame_size]

    records: list[tuple[bytes, bool, ProtocolFrame | None]] = []
    for frame in candidates:
        expected_crc = CRC.unpack_from(frame, payload_end)[0]
        actual_crc = zlib.crc32(frame[PREFIX.size:payload_start])
        actual_crc = zlib.crc32(frame[payload_start:payload_end], actual_crc) & 0xFFFFFFFF
        crc_ok = actual_crc == expected_crc
        parsed = parse_frames(frame, verify_frame_crc=False)
        header = parsed[0] if len(parsed) == 1 else None
        if header is not None and (
            header.chunk_count != expected_frames
            or header.total_size != total_size
            or not 0 <= header.chunk_index < expected_frames
        ):
            header = None
        records.append((frame, crc_ok, header))

    chunks: dict[int, tuple[bytes, bool]] = {}
    transfer_id: str | None = None
    file_crc32: int | None = None
    header_groups: dict[str, list[tuple[bytes, bool, ProtocolFrame]]] = {}
    for frame, crc_ok, header in records:
        if header is not None:
            header_groups.setdefault(header.transfer_id, []).append((frame, crc_ok, header))

    if header_groups:
        def group_score(group: list[tuple[bytes, bool, ProtocolFrame]]) -> tuple[int, int, int]:
            good_chunks = {header.chunk_index for _, crc_ok, header in group if crc_ok}
            all_chunks = {header.chunk_index for _, _, header in group}
            return len(good_chunks), len(all_chunks), sum(crc_ok for _, crc_ok, _ in group)

        selected_group = max(header_groups.values(), key=group_score)
        representative = next(
            (record for record in selected_group if record[1]),
            selected_group[0],
        )
        transfer_id = representative[2].transfer_id
        file_crc32 = representative[2].file_crc32
        for frame, crc_ok, header in selected_group:
            previous = chunks.get(header.chunk_index)
            if previous is None or (crc_ok and not previous[1]):
                chunks[header.chunk_index] = (frame, crc_ok)
    else:
        if len(records) > expected_frames:
            scores = [
                sum(crc_ok for _, crc_ok, _ in records[start : start + expected_frames])
                for start in range(len(records) - expected_frames + 1)
            ]
            best_start = int(np.argmax(scores))
            selected = records[best_start : best_start + expected_frames]
        else:
            selected = records
        for index, (frame, crc_ok, _) in enumerate(selected):
            chunks[index] = (frame, crc_ok)

    recovered = bytearray(total_size)
    for index, (frame, _) in chunks.items():
        start = index * config.payload_size
        end = min(total_size, start + config.payload_size)
        recovered[start:end] = frame[payload_start : payload_start + end - start]

    received_frames = len(chunks)
    return PositionalPayloadRecovery(
        payload=bytes(recovered),
        expected_frames=expected_frames,
        received_frames=received_frames,
        frame_crc_ok=sum(crc_ok for _, crc_ok in chunks.values()),
        complete=received_frames == expected_frames,
        transfer_id=transfer_id,
        file_crc32=file_crc32,
    )


def reassemble_transfer(
    frames: list[ProtocolFrame],
    output_dir: Path,
    *,
    output_name: str | None = None,
    require_file_crc: bool = True,
) -> ReassemblyResult | None:
    if not frames:
        return None

    grouped: dict[str, list[ProtocolFrame]] = {}
    for frame in frames:
        grouped.setdefault(frame.transfer_id, []).append(frame)

    transfer_id, selected = max(
        grouped.items(),
        key=lambda item: len({frame.chunk_index for frame in item[1]}),
    )
    first = next((frame for frame in selected if frame.frame_crc_ok), selected[0])
    chunks: dict[int, ProtocolFrame] = {}
    for frame in selected:
        if frame.chunk_count != first.chunk_count:
            continue
        if not 0 <= frame.chunk_index < frame.chunk_count:
            continue
        previous = chunks.get(frame.chunk_index)
        if previous is None or (frame.frame_crc_ok and not previous.frame_crc_ok):
            chunks[frame.chunk_index] = frame

    complete = len(chunks) == first.chunk_count
    recovered_bytes: bytes | None = None
    crc_ok = False
    output_path: Path | None = None
    requested_name = output_name or first.filename
    filename = Path(requested_name).name
    if not filename or filename != requested_name:
        raise ValueError("output name must be a non-empty plain filename")

    if complete:
        recovered_bytes = b"".join(chunks[index].payload for index in range(first.chunk_count))[
            : first.total_size
        ]
        crc_ok = (zlib.crc32(recovered_bytes) & 0xFFFFFFFF) == first.file_crc32
        # Write corrupt output only when explicitly requested.
        if crc_ok or not require_file_crc:
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / filename
            output_path.write_bytes(recovered_bytes)

    return ReassemblyResult(
        transfer_id=transfer_id,
        filename=filename,
        media_type=first.media_type,
        chunk_count=first.chunk_count,
        received_chunks=len(chunks),
        total_size=first.total_size,
        complete=complete,
        crc_ok=crc_ok,
        output_path=output_path,
        recovered_bytes=recovered_bytes,
    )
