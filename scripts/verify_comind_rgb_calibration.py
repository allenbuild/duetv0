#!/usr/bin/env python3
"""Bind cached native RGB calibration using cached visual anchors; no raw/image reads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from duet.adapters.comind.frame_map import load_frame_map
from duet.adapters.comind.validation import ensure_output_outside_raw
from duet.adapters.comind.vrs import (
    ImageOrientation,
    camera_from_summary,
    exported_calibration_evidence,
    verify_cached_orientation_audit,
    verify_clockwise_export_anchors,
    verify_export_pixel_geometry,
)
from duet.schemas.common import Provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", default="43276420-701f-4731-b9ab-bebc7fd14994")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument(
        "--orientation-audit",
        type=Path,
        help="Additive cached-image audit from audit_comind_cached_orientation.py; original coarse anchors and temporal maps remain unchanged",
    )
    args = parser.parse_args()
    recording_id = str(UUID(args.recording_id))
    root = args.processed_root / recording_id
    ensure_output_outside_raw(root, Path("data/raw/comind"))
    orientation_audit = None
    if args.orientation_audit is not None:
        ensure_output_outside_raw(args.orientation_audit, Path("data/raw/comind"))
        orientation_audit = json.loads(args.orientation_audit.read_text())
        if orientation_audit["recording_id"] != recording_id:
            raise ValueError("orientation audit belongs to another recording")
    pending = []
    for role in ("helper", "leader"):
        anchor_path = root / "frame_maps" / f"{role}_anchors.json"
        camera_path = root / "vrs" / f"{role}_native_rgb_calibration.json"
        video_path = root / "frame_maps" / "fingerprints" / f"{role}_mp4" / "video.json"
        video = json.loads(video_path.read_text())
        camera = camera_from_summary(json.loads(camera_path.read_text()))
        width, height = (int(value) for value in camera.get_image_size())
        if [video["width"], video["height"]] != [height, width]:
            raise ValueError("MP4 resolution does not match a pure CW90 native-image rotation")
        pixel_path = root / "frame_maps" / f"{role}_pixel_geometry.json"
        pixel_audit = json.loads(pixel_path.read_text())
        frame_map_path = root / "frame_maps" / f"{role}_frame_map.npz"
        with frame_map_path.open("rb") as stream:
            frame_map_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        if orientation_audit is None:
            evidence = verify_clockwise_export_anchors(
                json.loads(anchor_path.read_text()), frame_count=video["frame_count"]
            )
        else:
            frame_map = load_frame_map(frame_map_path)
            if (frame_map.recording_id, frame_map.participant) != (
                recording_id,
                role,
            ) or frame_map.frame_count != video["frame_count"]:
                raise ValueError("frame-map recording, participant, or video count mismatch")
            with frame_map_path.with_suffix(".json").open("rb") as stream:
                metadata_hash = hashlib.file_digest(stream, "sha256").hexdigest()
            evidence = verify_cached_orientation_audit(
                orientation_audit["participants"][role],
                frame_map,
                frame_map_sha256=frame_map_hash,
                frame_map_metadata_sha256=metadata_hash,
            )
            anchor_path = args.orientation_audit
        pixel_evidence = verify_export_pixel_geometry(
            pixel_audit,
            recording_id=recording_id,
            participant=role,
            frame_count=video["frame_count"],
            image_size=(height, width),
            frame_map_sha256=frame_map_hash,
        )
        with np.load(frame_map_path, allow_pickle=False) as frame_map:
            for sample in pixel_audit["samples"]:
                index = sample["mp4_frame_index"]
                if (
                    frame_map["status"][index] != "VERIFIED"
                    or frame_map["vrs_rgb_frame_index"][index] != sample["vrs_rgb_frame_index"]
                    or frame_map["vrs_device_timestamp_ns"][index] != sample["device_timestamp_ns"]
                ):
                    raise ValueError("pixel audit source frame does not match verified frame map")
        result = exported_calibration_evidence(
            camera,
            participant=role,
            orientation=ImageOrientation.CLOCKWISE_90,
            verification=evidence["verification"],
            provenance=Provenance(
                str(anchor_path),
                "Visual rotation evidence plus SDK native Fisheye624 model",
                (Provenance(str(camera_path)),),
            ),
        )
        result["orientation_evidence"] = evidence
        result["orientation_evidence_source"] = str(anchor_path.resolve())
        result["pixel_geometry_evidence"] = pixel_evidence
        result["pixel_geometry_source"] = str(pixel_path.resolve())
        result["recording_id"] = recording_id
        result["native_calibration_source"] = str(camera_path.resolve())
        output = root / "vrs" / f"{role}_export_rgb_calibration.json"
        ensure_output_outside_raw(output, Path("data/raw/comind"))
        pending.append((role, output, result))
    # Both roles must pass every existing orientation/pixel gate before writing outputs.
    for role, output, result in pending:
        output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "participant": role,
                    "output": str(output),
                    "orientation": result["orientation_evidence"],
                    "projection_roundtrip_max_error_pixels": result[
                        "projection_roundtrip_max_error_pixels"
                    ],
                    "official_rotation_max_error_pixels": result[
                        "official_rotation_max_error_pixels"
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
