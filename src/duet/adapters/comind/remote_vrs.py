"""Bounded CoMind VRS2 range access, preserving RECORD_TIME and exact DEVICE_TIME.

Parsers are shared with the original offline probe. The supported format is the
observed classic v2 Zstd index, v2 descriptions, and uncompressed RGB JPEG data
records. Unsupported variants fail explicitly. No raw VRS ranges are persisted.

Format references: facebookresearch/vrs vrs/FileFormat.h, DescriptionRecord.cpp,
IndexRecord.cpp, DataLayout.cpp and Record.h (also cited by the original probe).
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import os
import struct
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Self
from urllib.parse import urlsplit

import numpy as np
import pyarrow as pa
from PIL import Image

from duet.adapters.comind.frame_map import image_fingerprint
from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.vrs import native_calibration_summary
from duet.schemas.common import Provenance

DISK_INFO = np.dtype(
    [
        ("timestamp", "<f8"),
        ("size", "<u4"),
        ("type", "u1"),
        ("stream_type", "<i4"),
        ("instance", "<u2"),
    ]
)
RECORD_HEADER = struct.Struct("<IIiIdHBBI")
SOURCE_BASE = "https://github.com/facebookresearch/vrs/blob/main/vrs/"


class Cursor:
    """Bounded little-endian description reader with no implicit padding."""

    def __init__(self, data: bytes):
        self.data, self.position = data, 0

    def take(self, length: int) -> bytes:
        if length < 0 or self.position + length > len(self.data):
            raise ValueError("truncated description")
        result = self.data[self.position : self.position + length]
        self.position += length
        return result

    def integer(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def string(self) -> str:
        return self.take(self.integer()).decode("utf-8")

    def mapping(self) -> dict[str, str]:
        count = self.integer()
        result = {}
        for _ in range(count):
            name, value = self.string(), self.string()
            if name in result:
                raise ValueError("duplicate description map key")
            result[name] = value
        return result


def description(data: bytes) -> dict:
    reader = Cursor(data)
    streams = {}
    for _ in range(reader.integer()):
        kind, instance = struct.unpack("<iH", reader.take(6))
        name = f"{kind}-{instance}"
        if name in streams:
            raise ValueError("duplicate stream description")
        streams[name] = {"user": reader.mapping(), "vrs": reader.mapping()}
    tags = reader.mapping()
    if reader.position != len(data):
        raise ValueError("unconsumed description bytes")
    return {"streams": streams, "file_tags": tags}


def index(payload: bytes, *, uncompressed_size: int, first_user_offset: int, file_size: int):
    """Decode bounded classic-v2 index; its preallocated tail may contain padding."""
    reader = Cursor(payload)
    stream_count = reader.integer()
    stream_ids = [struct.unpack("<iH", reader.take(6)) for _ in range(stream_count)]
    if len(stream_ids) != len(set(stream_ids)):
        raise ValueError("duplicate index stream identifier")
    record_count = reader.integer()
    byte_count = record_count * DISK_INFO.itemsize
    if not 0 < byte_count <= 128_000_000 or byte_count + reader.position != uncompressed_size:
        raise ValueError("inconsistent or excessive decompressed index size")
    # A streaming decoder stops after the known record array, before reserved padding.
    with pa.CompressedInputStream(pa.BufferReader(payload[reader.position :]), "zstd") as stream:
        decoded = stream.read(byte_count)
    if len(decoded) != byte_count:
        raise ValueError("truncated compressed index")
    records = np.frombuffer(decoded, dtype=DISK_INFO)
    if (
        not np.isfinite(records["timestamp"]).all()
        or np.any(records["size"] < RECORD_HEADER.size)
        or not np.isin(records["type"], [1, 2, 3, 4]).all()
    ):
        raise ValueError("invalid index record")
    actual_ids = set(zip(records["stream_type"].tolist(), records["instance"].tolist()))
    if actual_ids != set(stream_ids):
        raise ValueError("index records and stream inventory disagree")
    offsets = np.r_[
        np.uint64(first_user_offset),
        first_user_offset + np.cumsum(records["size"], dtype=np.uint64),
    ]
    if int(offsets[-1]) != file_size:
        raise ValueError("index record sizes do not end at the verified remote file size")
    return records, offsets


def processed_directory(path: Path) -> Path:
    """Require derived caches beneath a real data/processed directory, not raw."""
    resolved = path.resolve()
    parts = resolved.parts
    if not any(parts[i : i + 2] == ("data", "processed") for i in range(len(parts) - 1)):
        raise ValueError("remote VRS caches must be under data/processed/")
    if any(parts[i : i + 2] == ("data", "raw") for i in range(len(parts) - 1)):
        raise ValueError("remote VRS caches cannot be under data/raw/")
    return resolved


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class RangeReader(Protocol):
    size: int
    identity: str

    def read(self, offset: int, length: int) -> bytes: ...


class HttpRangeReader:
    """Exact HTTPS ranges with a persistent conservative budget and immutable ETag.

    No redirects, retries, HEAD, full-body fallback, or compressed transfer is
    permitted. Every attempted range is budgeted before opening the connection.
    The ledger contains offsets, checksums and byte counts only, never payloads.
    Use this as a context manager; one process exclusively owns a ledger.
    """

    transport_allowance = 65_536

    def __init__(
        self,
        url: str,
        *,
        size: int,
        budget_bytes: int,
        ledger: Path,
        timeout: float = 120.0,
        max_request_bytes: int = 32_000_000,
    ) -> None:
        import fcntl

        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.fragment:
            raise ValueError("an uncredentialed HTTPS VRS URL is required")
        if not parsed.path.endswith(".vrs"):
            raise ValueError("range source must be a VRS URL")
        if any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in (size, budget_bytes, max_request_bytes)
        ):
            raise ValueError("positive integer size and budgets required")
        self.size, self.identity, self.budget_bytes = size, url, budget_bytes
        self.ledger = processed_directory(ledger)
        self.ledger.parent.mkdir(parents=True, exist_ok=True)
        self._lock = self.ledger.with_suffix(".lock").open("a")
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.report = (
                json.loads(self.ledger.read_text())
                if self.ledger.exists()
                else {
                    "url": url,
                    "file_size_bytes": size,
                    "etag": None,
                    "reserved_bytes": 0,
                    "body_bytes_read": 0,
                    "requests": [],
                }
            )
            if self.report.get("url") != url or self.report.get("file_size_bytes") != size:
                raise ValueError("range ledger belongs to a different remote source")
            entries = self.report["requests"]
            if any(
                not isinstance(row.get("reserved_bytes"), int)
                or isinstance(row["reserved_bytes"], bool)
                or row["reserved_bytes"] < self.transport_allowance
                for row in entries
            ):
                raise ValueError("invalid persistent range budget")
            if self.report["reserved_bytes"] != sum(row["reserved_bytes"] for row in entries):
                raise ValueError("persistent range budget sum mismatch")
            self.etag = self.report["etag"]
            self._connection = None
            self._host, self._port = parsed.hostname, parsed.port
            self._path = parsed.path + ("?" + parsed.query if parsed.query else "")
            self.timeout, self.max_request_bytes = timeout, max_request_bytes
        except BaseException:
            self._lock.close()
            raise

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._lock.close()

    def read(self, offset: int, length: int) -> bytes:
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (offset, length)):
            raise TypeError("integer range coordinates required")
        if (
            offset < 0
            or length <= 0
            or offset + length > self.size
            or length >= self.size
            or length > self.max_request_bytes
        ):
            raise ValueError("invalid, full-file, or excessive VRS byte range")
        reserve = length + self.transport_allowance
        if self.report["reserved_bytes"] + reserve > self.budget_bytes:
            raise ValueError("cumulative VRS range budget exceeded before request")
        entry = {
            "offset": offset,
            "length": length,
            "reserved_bytes": reserve,
            "body_bytes_read": 0,
            "status": "PENDING",
        }
        self.report["requests"].append(entry)
        self.report["reserved_bytes"] += reserve
        atomic_json(self.ledger, self.report)
        headers = {
            "Range": f"bytes={offset}-{offset + length - 1}",
            "Accept-Encoding": "identity",
            "User-Agent": "Duet-minimal-VRS/1",
        }
        if self.etag is not None:
            headers["If-Match"] = self.etag
        started = time.monotonic()
        try:
            if self._connection is None:
                self._connection = http.client.HTTPSConnection(
                    self._host, port=self._port, timeout=self.timeout
                )
            self._connection.request("GET", self._path, headers=headers)
            response = self._connection.getresponse()
            entry["http_status"] = response.status
            etag = response.getheader("ETag")
            if (
                response.status != 206
                or response.getheader("Content-Range")
                != f"bytes {offset}-{offset + length - 1}/{self.size}"
                or response.getheader("Content-Length") != str(length)
                or response.getheader("Content-Encoding", "identity") != "identity"
                or response.getheader("Transfer-Encoding") is not None
                or not etag
                or etag.startswith("W/")
                or (self.etag is not None and etag != self.etag)
            ):
                raise ValueError("rejected HTTP range headers/ETag without reading body")
            self.etag = etag
            self.report["etag"] = etag
            chunks = []
            while entry["body_bytes_read"] < length:
                if time.monotonic() - started > self.timeout:
                    raise TimeoutError("VRS range deadline exceeded")
                block = response.read(min(65_536, length - entry["body_bytes_read"]))
                if not block:
                    raise ValueError("truncated VRS range response")
                entry["body_bytes_read"] += len(block)
                self.report["body_bytes_read"] += len(block)
                chunks.append(block)
            payload = b"".join(chunks)
            entry.update(status="COMPLETE", sha256=hashlib.sha256(payload).hexdigest())
            return payload
        except BaseException as error:
            entry.update(status="FAILED", error=f"{type(error).__name__}: {error}")
            if self._connection is not None:
                self._connection.close()
                self._connection = None
            raise
        finally:
            entry["elapsed_seconds"] = time.monotonic() - started
            atomic_json(self.ledger, self.report)


@dataclass
class RemoteVrs:
    """An RGB-only derived view of a VRS index; source records stay in memory."""

    reader: RangeReader
    stream_id: str
    description: dict
    record_time_seconds: np.ndarray
    record_offsets: np.ndarray
    record_sizes: np.ndarray
    device_timestamps_ns: np.ndarray
    source_frame_numbers: np.ndarray
    metadata_known: np.ndarray
    fingerprints: dict[int, np.ndarray]
    range_index_bytes: int

    @property
    def rgb_count(self) -> int:
        return len(self.record_sizes)

    @property
    def image_size(self) -> tuple[int, int]:
        config = self.description["rgb_configuration"]
        return (config["image_width"], config["image_height"])

    @classmethod
    def open(cls, reader: RangeReader) -> RemoteVrs:
        header = reader.read(0, 80)
        fields = struct.unpack("<IIQIIqqqQQQII", header)
        if header[:8] != b"VisionRe" or header[72:] != b"cordVRS2" or fields[3:5] != (80, 32):
            raise ValueError("unsupported VRS file header")
        index_offset, description_offset, first_user = fields[5:8]
        if not 80 <= description_offset < index_offset < first_user < reader.size:
            raise ValueError("unsupported VRS description/index placement")
        if index_offset - description_offset > 1_000_000:
            raise ValueError("excessive VRS description")
        desc_bytes = reader.read(description_offset, index_offset - description_offset)
        desc_header = RECORD_HEADER.unpack_from(desc_bytes)
        if desc_header[0] != len(desc_bytes) or desc_header[2:4] != (2, 2) or desc_header[7] != 0:
            raise ValueError("unsupported VRS description record")
        parsed = description(desc_bytes[RECORD_HEADER.size :])
        index_header = RECORD_HEADER.unpack(reader.read(index_offset, RECORD_HEADER.size))
        if (
            index_header[2:4] != (1, 2)
            or index_header[7] != 2
            or index_header[0] + index_offset != first_user
            or not 32 < index_header[0] <= 32_000_000
        ):
            raise ValueError("unsupported or excessive VRS index record")
        payload = reader.read(index_offset + 32, index_header[0] - 32)
        records, offsets = index(
            payload,
            uncompressed_size=index_header[8],
            first_user_offset=first_user,
            file_size=reader.size,
        )
        actual_ids = {
            f"{kind}-{instance}"
            for kind, instance in zip(records["stream_type"].tolist(), records["instance"].tolist())
        }
        if actual_ids != set(parsed["streams"]):
            raise ValueError("VRS description and index stream inventories disagree")
        rgb_ids = [
            key
            for key, value in parsed["streams"].items()
            if value["vrs"].get("VRS_Original_Recordable_Name") == "RGB Camera Class"
        ]
        if len(rgb_ids) != 1:
            raise ValueError("expected one explicitly named RGB stream")
        stream_id = rgb_ids[0]
        kind, instance = map(int, stream_id.split("-"))
        rows = np.flatnonzero(
            (records["stream_type"] == kind)
            & (records["instance"] == instance)
            & (records["type"] == 3)
        )
        times = records["timestamp"][rows].copy()
        if len(rows) == 0 or np.any(np.diff(times) < 0):
            raise ValueError("empty or nonmonotonic RGB RECORD_TIME")
        configuration_rows = np.flatnonzero(
            (records["stream_type"] == kind)
            & (records["instance"] == instance)
            & (records["type"] == 2)
        )
        if len(configuration_rows) != 1:
            raise ValueError("expected one stable RGB configuration record")
        configuration_row = int(configuration_rows[0])
        tags = parsed["streams"][stream_id]["vrs"]
        config_layout = json.loads(tags["DL:Configuration:2:0"])["data_layout"]
        config_fields = {field["name"]: field for field in config_layout}
        config_types = {
            "image_width": ("uint32_t", "<I", 4),
            "image_height": ("uint32_t", "<I", 4),
            "nominal_rate": ("double", "<d", 8),
        }
        for name, (dtype, _, _) in config_types.items():
            field = config_fields[name]
            if (
                field["type"] != f"DataPieceValue<{dtype}>"
                or isinstance(field["offset"], bool)
                or not isinstance(field["offset"], int)
                or not 0 <= field["offset"] <= 4096
            ):
                raise ValueError("unsupported RGB configuration metadata layout")
        config_size = int(records["size"][configuration_row])
        if not 32 < config_size <= 1_000_000:
            raise ValueError("RGB configuration record exceeds bound")
        config_bytes = reader.read(int(offsets[configuration_row]), config_size)
        config_header = RECORD_HEADER.unpack_from(config_bytes)
        if (
            config_header[0] != int(records["size"][configuration_row])
            or config_header[2:4] != (kind, 2)
            or config_header[5:7] != (instance, 2)
            or config_header[7] not in (0, 2)
        ):
            raise ValueError("unsupported RGB configuration record")
        config_payload = config_bytes[32:]
        if config_header[7] == 2:
            if not 0 < config_header[8] <= 1_000_000:
                raise ValueError("excessive decompressed RGB configuration")
            config_payload = pa.decompress(
                config_payload, decompressed_size=config_header[8], codec="zstd"
            ).to_pybytes()
        required_config_bytes = max(
            config_fields[name]["offset"] + width for name, (_, _, width) in config_types.items()
        )
        if len(config_payload) < required_config_bytes:
            raise ValueError("truncated RGB configuration metadata")
        config = {
            name: struct.unpack_from(fmt, config_payload, config_fields[name]["offset"])[0]
            for name, (_, fmt, _) in config_types.items()
        }
        if (
            not 0 < config["image_width"] <= 4096
            or not 0 < config["image_height"] <= 4096
            or not np.isfinite(config["nominal_rate"])
            or config["nominal_rate"] <= 0
        ):
            raise ValueError("invalid RGB image dimensions or nominal rate")
        # Keep only relevant native RGB layout/calibration metadata, not other stream payloads.
        derived_description = {
            "rgb_configuration": config,
            "streams": {stream_id: parsed["streams"][stream_id]},
            "file_tags": {
                key: parsed["file_tags"][key]
                for key in ("calib_json", "metadata")
                if key in parsed["file_tags"]
            },
        }
        return cls(
            reader,
            stream_id,
            derived_description,
            times,
            offsets[rows].copy(),
            records["size"][rows].copy(),
            np.full(len(rows), -1, dtype=np.int64),
            np.full(len(rows), -1, dtype=np.int64),
            np.zeros(len(rows), dtype=bool),
            {},
            80 + len(desc_bytes) + index_header[0] + config_size,
        )

    def _fields(self) -> dict:
        tags = self.description["streams"][self.stream_id]["vrs"]
        if tags.get("RF:Data:2") != "data_layout+image/jpg":
            raise ValueError("unsupported RGB record format; JPEG required")
        fields = json.loads(tags["DL:Data:2:0"])["data_layout"]
        result = {value["name"]: value for value in fields}
        if len(result) != len(fields):
            raise ValueError("duplicate VRS RGB layout fields")
        for name, dtype in (("capture_timestamp_ns", "int64_t"), ("frame_number", "uint64_t")):
            field = result[name]
            if (
                field["type"] != f"DataPieceValue<{dtype}>"
                or isinstance(field["offset"], bool)
                or not isinstance(field["offset"], int)
                or not 0 <= field["offset"] <= 4096
            ):
                raise ValueError("unsupported RGB timestamp metadata layout")
        return result

    def _check_index(self, rgb_index: int) -> int:
        if isinstance(rgb_index, bool) or not isinstance(rgb_index, (int, np.integer)):
            raise TypeError("RGB index must be integer")
        value = int(rgb_index)
        if not 0 <= value < self.rgb_count:
            raise IndexError("VRS RGB index out of range")
        return value

    def _parse_metadata(self, rgb_index: int, payload: bytes) -> None:
        fields = self._fields()
        header = RECORD_HEADER.unpack_from(payload)
        kind, instance = map(int, self.stream_id.split("-"))
        if (
            header[0] != int(self.record_sizes[rgb_index])
            or header[2:4] != (kind, 2)
            or header[5:8] != (instance, 3, 0)
            or header[4] != float(self.record_time_seconds[rgb_index])
        ):
            raise ValueError("RGB data record and validated index disagree")
        capture = struct.unpack_from("<q", payload, 32 + fields["capture_timestamp_ns"]["offset"])[
            0
        ]
        frame = struct.unpack_from("<Q", payload, 32 + fields["frame_number"]["offset"])[0]
        if capture < 0 or frame > np.iinfo(np.int64).max:
            raise ValueError("invalid RGB DEVICE_TIME/frame number")
        if self.metadata_known[rgb_index] and (
            self.device_timestamps_ns[rgb_index] != capture
            or self.source_frame_numbers[rgb_index] != frame
        ):
            raise ValueError("cached exact RGB metadata changed")
        self.device_timestamps_ns[rgb_index] = capture
        self.source_frame_numbers[rgb_index] = frame
        self.metadata_known[rgb_index] = True

    def capture_metadata(self, indices: Iterable[int]) -> np.ndarray:
        """Read exact DEVICE_TIME int64 prefixes, never round index doubles to ns."""
        requested = [self._check_index(value) for value in indices]
        fields = self._fields()
        length = (
            32
            + max(fields[name]["offset"] for name in ("capture_timestamp_ns", "frame_number"))
            + 8
        )
        for rgb_index in requested:
            if not self.metadata_known[rgb_index]:
                if length > self.record_sizes[rgb_index]:
                    raise ValueError("RGB metadata exceeds record extent")
                payload = self.reader.read(int(self.record_offsets[rgb_index]), length)
                self._parse_metadata(rgb_index, payload)
        known_times = self.device_timestamps_ns[self.metadata_known]
        if np.any(np.diff(known_times) < 0):
            raise ValueError("nonmonotonic exact RGB DEVICE_TIME")
        return self.device_timestamps_ns[requested].copy()

    def rgb_image(self, rgb_index: int) -> np.ndarray:
        """Decode one explicitly selected JPEG record in memory; never persist pixels."""
        rgb_index = self._check_index(rgb_index)
        length = int(self.record_sizes[rgb_index])
        if not 32 < length <= 8_000_000:
            raise ValueError("excessive RGB image record")
        fields = self._fields()
        sizes = {"uint32_t": 4, "uint64_t": 8, "int64_t": 8, "double": 8}
        fixed_end = 0
        occupied = []
        vectors = []
        for field in fields.values():
            value_type = field["type"]
            if value_type.startswith("DataPieceValue<"):
                dtype = value_type.removeprefix("DataPieceValue<").removesuffix(">")
                if dtype not in sizes:
                    raise ValueError("unsupported fixed RGB layout field")
                offset = field["offset"]
                if (
                    isinstance(offset, bool)
                    or not isinstance(offset, int)
                    or not 0 <= offset <= 4096
                ):
                    raise ValueError("invalid fixed RGB layout field offset")
                occupied.extend(range(offset, offset + sizes[dtype]))
                fixed_end = max(fixed_end, offset + sizes[dtype])
            elif value_type == "DataPieceVector<uint8_t>" and field.get("index") == 0:
                vectors.append(field)
            else:
                raise ValueError("unsupported variable RGB metadata layout")
        if len(occupied) != len(set(occupied)) or set(occupied) != set(range(fixed_end)):
            raise ValueError("RGB fixed metadata must be contiguous and nonoverlapping")
        if len(vectors) != 1:
            raise ValueError("expected one RGB image_metadata vector")
        payload = self.reader.read(int(self.record_offsets[rgb_index]), length)
        self._parse_metadata(rgb_index, payload)
        # CoMind's supported RGB layout has an empty metadata vector. Fail on
        # nonempty variants instead of guessing a JPEG boundary from magic bytes.
        vector_offset, vector_size = struct.unpack_from("<II", payload, 32 + fixed_end)
        if (vector_offset, vector_size) != (0, 0):
            raise ValueError("nonempty RGB metadata vector is not yet supported")
        jpeg = payload[32 + fixed_end + 8 :]
        if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
            raise ValueError("declared RGB JPEG boundaries are invalid")
        # Pillow/libjpeg matches Project Aria's RGB decode exactly on the local
        # first/middle/last fixtures; FFmpeg chroma upsampling differs visibly.
        with Image.open(io.BytesIO(jpeg)) as decoded:
            if decoded.format != "JPEG" or decoded.size != self.image_size:
                raise ValueError("JPEG dimensions and RGB configuration disagree")
            image = np.asarray(decoded.convert("RGB"))
        self.fingerprints[rgb_index] = image_fingerprint(image)
        return image

    def fingerprint(self, rgb_index: int) -> np.ndarray:
        rgb_index = self._check_index(rgb_index)
        if rgb_index not in self.fingerprints:
            self.rgb_image(rgb_index)
        return self.fingerprints[rgb_index]

    def save_metadata(self, directory: Path, *, role: str, recording_id: str) -> None:
        """Write derived inventory, exact observed times, sparse fingerprints/calibration.

        Unknown DEVICE_TIME entries are -1 and metadata_known is authoritative;
        callers MUST NOT treat RECORD_TIME-derived rounding as exact capture time.
        """
        directory = processed_directory(directory)
        if role not in ("helper", "leader"):
            raise ValueError("unsupported CoMind participant")
        known_times = self.device_timestamps_ns[self.metadata_known]
        if np.any(known_times < 0) or np.any(np.diff(known_times) < 0):
            raise ValueError("cannot cache invalid or nonmonotonic exact RGB DEVICE_TIME")
        directory.mkdir(parents=True, exist_ok=True)
        sparse = np.array(sorted(self.fingerprints), dtype=np.int64)
        atomic_npz(
            directory / f"{role}_rgb_metadata.npz",
            record_time_seconds=self.record_time_seconds,
            record_file_offsets=self.record_offsets,
            record_sizes=self.record_sizes,
            device_timestamps_ns=self.device_timestamps_ns,
            source_frame_numbers=self.source_frame_numbers,
            metadata_known=self.metadata_known,
            vrs_rgb_frame_index=np.arange(self.rgb_count, dtype=np.int64),
            clock_domain=np.array(device_clock(recording_id, role).name),
            stream_id=np.array(self.stream_id),
            fingerprint_indices=sparse,
            fingerprints=np.array(
                [self.fingerprints[int(value)] for value in sparse], dtype=np.uint8
            ).reshape((-1, 32, 32)),
        )
        path = directory / f"{role}_rgb_metadata.npz"
        atomic_json(
            directory / f"{role}_remote_vrs.json",
            {
                "schema_version": 1,
                "recording_id": recording_id,
                "participant": role,
                "url": self.reader.identity,
                "file_size_bytes": self.reader.size,
                "etag": getattr(self.reader, "etag", None),
                "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "rgb_count": self.rgb_count,
                "range_index_bytes": self.range_index_bytes,
                "exact_device_times_known": int(self.metadata_known.sum()),
                "description": self.description,
                "record_time_semantics": "Original IEEE754 RECORD_TIME seconds from VRS index",
                "device_time_semantics": "Exact capture_timestamp_ns; -1 unless metadata_known",
            },
        )
        calibration_json = self.description["file_tags"].get("calib_json")
        if calibration_json:
            from projectaria_tools.core import calibration

            device = calibration.device_calibration_from_json_string(calibration_json)
            camera = None if device is None else device.get_camera_calib("camera-rgb")
            if camera is None:
                raise ValueError("VRS description lacks native RGB calibration")
            camera = calibration.rescale_camera_calibration(
                camera, self.image_size, device.get_device_version()
            )
            result = native_calibration_summary(
                camera,
                participant=role,
                provenance=Provenance(
                    self.reader.identity, "VRS description calib_json via ranges"
                ),
            )
            atomic_json(directory / f"{role}_native_rgb_calibration.json", result)

    @classmethod
    def from_cache(cls, reader: RangeReader, directory: Path, *, role: str) -> RemoteVrs:
        """Restore a compact cache, binding future ranges to the same strong ETag."""
        directory = processed_directory(directory)
        metadata = json.loads((directory / f"{role}_remote_vrs.json").read_text())
        path = directory / f"{role}_rgb_metadata.npz"
        if (
            metadata["url"] != reader.identity
            or metadata["file_size_bytes"] != reader.size
            or metadata["npz_sha256"] != hashlib.sha256(path.read_bytes()).hexdigest()
        ):
            raise ValueError("remote VRS cache source or checksum changed")
        if hasattr(reader, "etag") and (not metadata["etag"] or metadata["etag"] != reader.etag):
            raise ValueError("range ledger and metadata cache ETag disagree")
        with np.load(path, allow_pickle=False) as arrays:
            count = metadata["rgb_count"]
            names = (
                "record_time_seconds",
                "record_file_offsets",
                "record_sizes",
                "device_timestamps_ns",
                "source_frame_numbers",
                "metadata_known",
            )
            if any(arrays[name].shape != (count,) for name in names):
                raise ValueError("remote VRS cache dimensions changed")
            if not np.array_equal(arrays["vrs_rgb_frame_index"], np.arange(count)):
                raise ValueError("remote VRS cache RGB indices changed")
            fingerprints = {
                int(i): value.copy()
                for i, value in zip(
                    arrays["fingerprint_indices"], arrays["fingerprints"], strict=True
                )
            }
            return cls(
                reader,
                str(arrays["stream_id"]),
                metadata["description"],
                *(arrays[name].copy() for name in names),
                fingerprints,
                metadata["range_index_bytes"],
            )


def estimate_sparse_transfer(
    *,
    target_size_bytes: int,
    reference_size_bytes: int,
    reference_rgb_index: Path,
    reference_index_bytes: int,
    handover_bounds: list[tuple[int, int]],
    context_frames: int = 120,
) -> dict[str, object]:
    """Estimate future range traffic from local evidence only; never open a URL.

    Target index/frame count is intentionally unknown until the user runs the
    transfer. Duration is approximated by VRS byte-size scaling, JPEG bytes by the
    prior recording's size distribution. This is an estimate, not a quote or an
    upper bound. The actual HTTP budget is enforced separately before each range.
    """
    if min(target_size_bytes, reference_size_bytes, reference_index_bytes) <= 0:
        raise ValueError("positive VRS source sizes required")
    if context_frames < 0 or not handover_bounds:
        raise ValueError("nonnegative context and at least one handover required")
    if any(start < 0 or end < start for start, end in handover_bounds):
        raise ValueError("invalid handover bounds")
    with np.load(reference_rgb_index, allow_pickle=False) as archive:
        sizes = archive["record_sizes"].copy()
    if sizes.ndim != 1 or not len(sizes) or np.any(sizes < 32):
        raise ValueError("invalid local reference RGB index")
    count = max(
        int(np.ceil(len(sizes) * target_size_bytes / reference_size_bytes)),
        max(end for _, end in handover_bounds) + 1,
    )
    requested = np.zeros(count, dtype=bool)
    boundaries = [frame for bounds in handover_bounds for frame in bounds]
    anchors = {
        0,
        (count - 1) // 2,
        count - 1,
        *range(900, count, 900),
        *(
            i
            for boundary in boundaries
            for i in (boundary - 120, boundary, boundary + 120)
            if 0 <= i < count
        ),
    }
    for frame in anchors:
        requested[max(0, frame - 16) : min(count, frame + 17)] = True
    for start, end in handover_bounds:
        requested[max(0, start - context_frames - 8) : min(count, end + context_frames + 9)] = True
    middle = (count - 1) // 2
    requested[max(0, middle - 8) : min(count, middle + 9)] = True
    requested[: min(100, count)] = True
    unique_records = int(requested.sum())
    # Audits intentionally decode three full resolution pairs again rather than
    # retaining images in the derived fingerprint cache.
    rgb_requests = unique_records + 3
    index_bytes = int(np.ceil(reference_index_bytes * target_size_bytes / reference_size_bytes))
    typical_image_bytes = float(np.mean(sizes))
    p95_image_bytes = float(np.percentile(sizes, 95))
    expected = index_bytes + int(np.ceil(rgb_requests * typical_image_bytes))
    p95_scenario = index_bytes + int(np.ceil(rgb_requests * p95_image_bytes))
    request_count = 5 + rgb_requests
    return {
        "method": "Offline prior-recording index/JPEG scaling; no target HTTP requests",
        "reference_index": str(reference_rgb_index),
        "reference_vrs_size_bytes": reference_size_bytes,
        "target_vrs_size_bytes": target_size_bytes,
        "estimated_rgb_frame_count": count,
        "handover_count": len(handover_bounds),
        "context_frames_each_side": context_frames,
        "estimated_anchor_count": len(anchors),
        "estimated_unique_rgb_records": unique_records,
        "estimated_index_description_config_bytes": index_bytes,
        "reference_rgb_record_mean_bytes": typical_image_bytes,
        "reference_rgb_record_p95_bytes": p95_image_bytes,
        "estimated_requests": request_count,
        "estimated_body_bytes": expected,
        "p95_record_size_scenario_body_bytes": p95_scenario,
        "estimated_conservative_reserved_bytes": expected + request_count * 65_536,
        "recommended_budget_bytes": int(np.ceil((p95_scenario + request_count * 65_536) * 1.25)),
        "caveat": "Target duration, JPEG size, offsets and visual ambiguity are unobserved. "
        "Estimate assumes candidate windows find direct matches; failed matches "
        "remain unresolved. Transfer requires explicit user invocation and "
        "stops at its separately configured cumulative byte budget.",
    }
