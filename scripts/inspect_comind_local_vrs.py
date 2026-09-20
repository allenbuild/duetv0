#!/usr/bin/env python3
"""Extract compact exact VRS RGB timestamps/calibration without decoding RGB images."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from uuid import UUID

import numpy as np

from duet.adapters.comind.validation import ensure_output_outside_raw
from duet.adapters.comind.vrs import (
    extract_rgb_metadata,
    native_calibration_summary,
    read_source_frame_numbers,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--recording-id", default="43276420-701f-4731-b9ab-bebc7fd14994")
    parser.add_argument(
        "--probe-root", type=Path, default=Path("outputs/comind_sync_forensics/remote_vrs")
    )
    args = parser.parse_args()
    recording_id = str(UUID(args.recording_id))
    output_root = Path(f"data/processed/comind/{recording_id}/vrs")
    report_path = Path(f"outputs/comind_vrs/{recording_id}.json")
    for output in (output_root, report_path):
        ensure_output_outside_raw(output, args.root)
    output_root.mkdir(parents=True, exist_ok=True)
    report = {
        "recording_id": recording_id,
        "projectaria_tools_version": version("projectaria-tools"),
        "participants": {},
    }
    for role in ("helper", "leader"):
        path = args.root / "recordings" / recording_id / "trimmed_vrs" / f"{role}_trimmed.vrs"
        before = path.stat()
        print(f"Reading exact {role} RGB DEVICE_TIME metadata; RGB pixels disabled", flush=True)
        metadata, camera = extract_rgb_metadata(path, recording_id=recording_id, participant=role)
        index_path = args.probe_root / f"{role}_rgb_index.npz"
        description_path = args.probe_root / f"{role}_description.json"
        if index_path.exists() and description_path.exists():
            description = json.loads(description_path.read_text())
            layout = json.loads(description["streams"][metadata.stream_id]["vrs"]["DL:Data:2:0"])
            with np.load(index_path, allow_pickle=False) as index:
                numbers = read_source_frame_numbers(
                    path,
                    record_offsets=index["record_file_offsets"],
                    metadata=metadata,
                    declared_data_layout=layout,
                )
            metadata = replace(metadata, source_frame_numbers=numbers)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("raw VRS changed while metadata was read")
        cache_path = output_root / f"{role}_rgb_metadata.npz"
        arrays = {
            "device_timestamps_ns": metadata.device_timestamps_ns,
            "vrs_rgb_frame_index": np.arange(len(metadata.device_timestamps_ns), dtype=np.int64),
            "clock_domain": np.array(metadata.clock_domain.name),
            "stream_id": np.array(metadata.stream_id),
        }
        if metadata.source_frame_numbers is not None:
            arrays["source_frame_numbers"] = metadata.source_frame_numbers
        np.savez_compressed(cache_path, **arrays)
        calibration = native_calibration_summary(
            camera, participant=role, provenance=metadata.provenance
        )
        calibration_path = output_root / f"{role}_native_rgb_calibration.json"
        calibration_path.write_text(json.dumps(calibration, indent=2, allow_nan=False) + "\n")
        report["participants"][role] = metadata.summary() | {
            "cache_path": str(cache_path.resolve()),
            "calibration_path": str(calibration_path.resolve()),
            "native_calibration": calibration,
            "raw_size_mtime_unchanged": True,
            "source_frame_numbers_verified_against_all_sdk_capture_timestamps": metadata.source_frame_numbers
            is not None,
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps(metadata.summary(), indent=2), flush=True)


if __name__ == "__main__":
    main()
