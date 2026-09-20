#!/usr/bin/env python3
"""Inspect previously bounded VRS probe artifacts only; this script has no network IO.

Supports the observed VRS2 file, v2 description, and classic v2 Zstd index.
Parsing follows facebookresearch/vrs FileFormat.h, DescriptionRecord.cpp,
IndexRecord.h, and IndexRecord.cpp. It never binds MP4 frames to VRS records.
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

from duet.adapters.comind.validation import ensure_output_outside_raw

ROOT = Path("outputs/comind_sync_forensics/remote_vrs")
REPORT = ROOT.parent / "43276420-701f-4731-b9ab-bebc7fd14994_remote_vrs.json"
from duet.adapters.comind.remote_vrs import (
    DISK_INFO,  # noqa: F401 -- compatibility export for existing parser tests
    RECORD_HEADER,
    SOURCE_BASE,
    description,
    index,
)


def artifact(pattern: str) -> bytes:
    """Read exactly one cached artifact, rejecting ambiguous or raw-tree paths."""
    matches = list(ROOT.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected exactly one cached artifact for {pattern}")
    ensure_output_outside_raw(matches[0], Path("data/raw/comind"))
    return matches[0].read_bytes()


def inspect(role: str, file_size: int, mp4_scalar_ns: int) -> dict:
    header = artifact(f"{role}_header_0_4096.bin")
    extra = artifact(f"{role}_description_index_header_*.bin")
    prefix = header + extra
    fields = struct.unpack_from("<IIQIIqqqQQQII", prefix)
    if prefix[:8] != b"VisionRe" or prefix[72:80] != b"cordVRS2" or fields[3:5] != (80, 32):
        raise ValueError("unsupported VRS header")
    index_offset, description_offset, first_user = fields[5:8]
    description_header = RECORD_HEADER.unpack_from(prefix, description_offset)
    index_header = RECORD_HEADER.unpack_from(prefix, index_offset)
    if description_header[2:4] != (2, 2) or description_header[7] != 0:
        raise ValueError("unsupported description record")
    if index_header[2:4] != (1, 2) or index_header[7] != 2:
        raise ValueError("unsupported index record")
    if description_offset + description_header[0] != index_offset:
        raise ValueError("unexpected description boundary")
    parsed = description(prefix[description_offset + 32 : index_offset])
    payload = artifact(f"{role}_index_payload_*.bin")
    if len(payload) != index_header[0] - 32 or index_offset + index_header[0] != first_user:
        raise ValueError("unexpected index payload size or boundary")
    records, offsets = index(
        payload,
        uncompressed_size=index_header[8],
        first_user_offset=first_user,
        file_size=file_size,
    )
    index_streams = {
        f"{kind}-{instance}"
        for kind, instance in zip(records["stream_type"].tolist(), records["instance"].tolist())
    }
    if index_streams != set(parsed["streams"]):
        raise ValueError("description and index stream inventories disagree")
    inventory = []
    for stream_id, tags in parsed["streams"].items():
        kind, instance = map(int, stream_id.split("-"))
        matching = (records["stream_type"] == kind) & (records["instance"] == instance)
        times = records["timestamp"][matching & (records["type"] == 3)]
        inventory.append(
            {
                "stream_id": stream_id,
                "name": tags["vrs"].get("VRS_Original_Recordable_Name"),
                "data_record_count": len(times),
                "configuration_record_count": int(
                    np.count_nonzero(matching & (records["type"] == 2))
                ),
                "first_record_time_seconds": float(times[0]) if len(times) else None,
                "last_record_time_seconds": float(times[-1]) if len(times) else None,
            }
        )
    rgb_ids = [item["stream_id"] for item in inventory if item["name"] == "RGB Camera Class"]
    if len(rgb_ids) != 1:
        raise ValueError("expected one explicitly named RGB stream")
    rgb_kind, rgb_instance = map(int, rgb_ids[0].split("-"))
    rgb_rows = np.flatnonzero(
        (records["stream_type"] == rgb_kind)
        & (records["instance"] == rgb_instance)
        & (records["type"] == 3)
    )
    rgb = records[rgb_rows]
    endpoints = []
    layout = json.loads(parsed["streams"][rgb_ids[0]]["vrs"]["DL:Data:2:0"])["data_layout"]
    layout_fields = {field["name"]: field for field in layout}
    for label, row in zip(("first", "last"), rgb_rows[[0, -1]]):
        offset = int(offsets[row])
        data = artifact(f"{role}_{label}_rgb_metadata_{offset}_160.bin")
        record = RECORD_HEADER.unpack_from(data)
        if record[2:4] != (rgb_kind, 2) or record[5:8] != (rgb_instance, 3, 0):
            raise ValueError("unsupported RGB metadata record")
        if record[0] != int(records["size"][row]) or record[4] != float(records["timestamp"][row]):
            raise ValueError("RGB endpoint and index disagree")
        capture_field, frame_field = (
            layout_fields["capture_timestamp_ns"],
            layout_fields["frame_number"],
        )
        if (
            capture_field["type"] != "DataPieceValue<int64_t>"
            or frame_field["type"] != "DataPieceValue<uint64_t>"
        ):
            raise ValueError("unexpected endpoint metadata types")
        capture_ns = struct.unpack_from("<q", data, 32 + capture_field["offset"])[0]
        frame_number = struct.unpack_from("<Q", data, 32 + frame_field["offset"])[0]
        endpoints.append(
            {
                "endpoint": label,
                "record_offset": offset,
                "capture_timestamp_ns": capture_ns,
                "source_frame_number": frame_number,
                "record_time_seconds": record[4],
                "capture_equals_record_time_at_double_precision": capture_ns / 1e9 == record[4],
            }
        )
    candidate = int(np.argmin(np.abs(rgb["timestamp"] - mp4_scalar_ns / 1e9)))
    intervals = np.diff(rgb["timestamp"]) * 1000
    metadata = json.loads(parsed["file_tags"]["metadata"])
    calibration = json.loads(parsed["file_tags"]["calib_json"])
    (ROOT / f"{role}_description.json").write_text(json.dumps(parsed, indent=2) + "\n")
    np.savez_compressed(
        ROOT / f"{role}_rgb_index.npz",
        record_time_seconds=rgb["timestamp"],
        record_file_offsets=offsets[rgb_rows],
        record_sizes=rgb["size"],
    )
    return {
        "file_size_bytes": file_size,
        "file_format": "VRS2",
        "description_version": 2,
        "index_version": 2,
        "index_record_count": len(records),
        "index_record_sizes_end_at_file_size": True,
        "stream_inventory": inventory,
        "file_tag_keys": sorted(parsed["file_tags"]),
        "calibration_json_present": "calib_json" in parsed["file_tags"],
        "native_calibration_json": {
            "parsed": True,
            "top_level_keys": sorted(calibration),
            "camera_labels": [
                camera.get("Label") for camera in calibration.get("CameraCalibrations", [])
            ],
            "binding_to_exported_mp4_verified": False,
        },
        "source_synchronization_metadata": {
            key: metadata.get(key)
            for key in ("ticsync_enabled", "ticsync_mode", "shared_session_id", "timecode_enabled")
        },
        "rgb": {
            "stream_id": rgb_ids[0],
            "data_record_count": len(rgb),
            "trimmed_mp4_frame_count": 21109,
            "count_difference_vrs_minus_mp4": len(rgb) - 21109,
            "complete_stream_index_bijection_rejected": len(rgb) != 21109,
            "unique_vrs_record_for_every_mp4_frame_possible_by_count": len(rgb) >= 21109,
            "prefix_subset_or_repeated_frame_mapping_verified": False,
            "record_time_strictly_increasing": bool(np.all(intervals > 0)),
            "duplicate_record_time_count": int(np.count_nonzero(intervals == 0)),
            "interval_ms_median_p95_max": np.percentile(intervals, [50, 95, 100]).tolist(),
            "endpoints": endpoints,
            "mp4_retained_scalar_ns": mp4_scalar_ns,
            "scalar_nearest_rgb_record_index": candidate,
            "scalar_minus_index_time_ns_double": float(
                (mp4_scalar_ns / 1e9 - rgb["timestamp"][candidate]) * 1e9
            ),
            "scalar_correspondence_is_image_mapping": False,
            "complete_capture_timestamp_ns_array_read": False,
            "index_time_precision": "Original IEEE754 double RECORD_TIME seconds retained; not promoted to exact DEVICE_TIME ns",
        },
    }


def main() -> None:
    for path in (ROOT, REPORT):
        ensure_output_outside_raw(path, Path("data/raw/comind"))
    report = json.loads(REPORT.read_text())
    for request in report["requests"]:
        if request.get("status") == "EXACT_RANGE_SAVED":
            path = Path(request["artifact"])
            ensure_output_outside_raw(path, Path("data/raw/comind"))
            payload = path.read_bytes()
            if (
                len(payload) != request["body_bytes_read"]
                or hashlib.sha256(payload).hexdigest() != request["sha256"]
            ):
                raise ValueError("cached probe artifact no longer matches the HTTP read log")
    for role in ("helper", "leader"):
        etags = {
            request["response_headers"].get("ETag")
            for request in report["requests"]
            if request["role"] == role
        }
        if len(etags) != 1 or None in etags:
            raise ValueError("remote ETag changed or was missing during partial reads")
    report["inspection"] = {
        "helper": inspect("helper", 12_572_405_231, 919623571487),
        "leader": inspect("leader", 11_901_082_187, 825528737150),
    }
    report["official_format_sources"] = [
        SOURCE_BASE + name
        for name in (
            "FileFormat.h",
            "DescriptionRecord.cpp",
            "IndexRecord.h",
            "IndexRecord.cpp",
            "Record.h",
        )
    ]
    report["conclusion"] = (
        "Remote partial inspection succeeded under 20 MB: complete stream inventories and RGB "
        "counts, index record-time arrays, file calibration tags, and exact first/last capture "
        "timestamps are available. Both RGB counts differ from the trimmed MP4 count. No "
        "MP4/VRS per-frame mapping, complete exact capture array, or synchronized viewer follows."
    )
    REPORT.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps({role: value["rgb"] for role, value in report["inspection"].items()}, indent=2)
    )


if __name__ == "__main__":
    main()
