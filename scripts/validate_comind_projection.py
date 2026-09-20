#!/usr/bin/env python3
"""Project cached aligned wrists into verified ego cameras; no raw or image reads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from duet.adapters.comind.validation import ensure_output_outside_raw
from duet.adapters.comind.vrs import (
    ImageOrientation,
    bind_exported_calibration,
    camera_from_summary,
)
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations
from duet.geometry.transforms import RigidTransform
from duet.qc.projection import check_point_projection_coverage
from duet.schemas.common import DistanceUnit, FrameId, Provenance

ROLES = ("helper", "leader")
LABELS = tuple(f"{role}/{side}/wrist" for role in ROLES for side in ("left", "right"))


def outside_raw(path: Path) -> Path:
    ensure_output_outside_raw(path, Path("data/raw/comind"))
    return path


def load_json(path: Path) -> dict:
    return json.loads(outside_raw(path).read_text())


def digest(path: Path) -> str:
    with outside_raw(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", default="43276420-701f-4731-b9ab-bebc7fd14994")
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument("--validation-report", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--minimum-coverage-fraction", type=float, default=0.0)
    args = parser.parse_args()
    recording_id = str(UUID(args.recording_id))
    directory = outside_raw(args.processed_root / recording_id)
    input_report_path = outside_raw(
        args.validation_report or Path(f"outputs/comind_frame_validation/{recording_id}.json")
    )
    report = load_json(input_report_path)
    if report["recording_id"] != recording_id or report["timeline"] != "comind_sync_frame_index":
        raise ValueError("aligned report recording or timeline mismatch")
    if report["shared_world"]["status"] != "VERIFIED":
        raise ValueError("projection requires a verified shared world")
    world = FrameId(report["shared_world"]["frame"])
    arrays, cameras, sources = {}, {}, {}
    for role in ROLES:
        cache_path = outside_raw(directory / "frame_validation" / f"{role}_aligned.npz")
        cache_hash = digest(cache_path)
        if report["aligned_cache_identities"][role]["sha256"] != cache_hash:
            raise ValueError("aligned cache no longer matches the validation report")
        with np.load(cache_path, allow_pickle=False) as source:
            arrays[role] = {name: source[name] for name in source.files}
        if str(arrays[role]["generation_id"].item()) != report["generation_id"]:
            raise ValueError("aligned cache generation mismatch")
        calibration_path = directory / "vrs" / f"{role}_export_rgb_calibration.json"
        evidence = load_json(calibration_path)
        if (
            evidence["status"] != "VERIFIED"
            or evidence["recording_id"] != recording_id
            or evidence["participant"] != role
            or evidence["pixel_rotation"] != "clockwise_90"
            or evidence["camera_frame"] != f"{role}/camera-rgb"
            or evidence["device_frame"] != f"{role}/device"
        ):
            raise ValueError("verified calibration evidence belongs to a different camera")
        native_path = directory / "vrs" / f"{role}_native_rgb_calibration.json"
        cameras[role] = bind_exported_calibration(
            camera_from_summary(load_json(native_path)),
            participant=role,
            orientation=ImageOrientation.CLOCKWISE_90,
            image_size=tuple(evidence["resolution"]),
            verification=evidence["verification"],
            provenance=Provenance(str(calibration_path)),
        )
        if not np.array_equal(
            cameras[role].transform_device_camera.matrix, evidence["T_device_camera"]
        ):
            raise ValueError("native and export camera extrinsics disagree")
        sources[role] = {
            "aligned_cache": str(cache_path),
            "aligned_cache_sha256": cache_hash,
            "export_calibration": str(calibration_path),
            "export_calibration_sha256": digest(calibration_path),
            "native_calibration": str(native_path),
            "native_calibration_sha256": digest(native_path),
        }
    if not np.array_equal(arrays["helper"]["mp4_frame_index"], arrays["leader"]["mp4_frame_index"]):
        raise ValueError("participants do not have corresponding synchronization frame indices")
    rows = []
    for handover in report["handover_ranking"]["handovers"]:
        start, end = handover["start_frame"], handover["end_frame"]
        if not 0 <= start <= end < len(arrays["helper"]["mp4_frame_index"]):
            raise ValueError("handover frame range outside aligned caches")
        indices = slice(start, end + 1)
        points = np.concatenate(
            [arrays[role]["hand_points_shared_m"][indices, :, 5, :] for role in ROLES], axis=1
        )
        present = np.concatenate([arrays[role]["hand_present"][indices] for role in ROLES], axis=1)
        camera_results = {}
        for role in ROLES:
            data, camera = arrays[role], cameras[role]
            accepted = data["pose_accepted"][indices] & data["mapping_verified"][indices]
            transforms: list[RigidTransform | None] = [None] * (end - start + 1)
            rotations = quaternion_xyzw_batch_to_rotations(
                data["pose_quaternion_xyzw"][indices][accepted]
            )
            translations = data["pose_translation_m"][indices][accepted]
            for local_index, rotation, translation in zip(
                np.flatnonzero(accepted), rotations, translations, strict=True
            ):
                matrix = np.eye(4)
                matrix[:3, :3], matrix[:3, 3] = rotation, translation
                device = RigidTransform(
                    matrix,
                    FrameId(f"{role}/device"),
                    world,
                    DistanceUnit.METERS,
                    Provenance(sources[role]["aligned_cache"]),
                    tolerance=1e-7,
                )
                transforms[local_index] = device.compose(camera.transform_device_camera)
            result = check_point_projection_coverage(
                points,
                point_present=present,
                point_labels=LABELS,
                world_frame=world,
                camera_frame=camera.transform_device_camera.source,
                transforms_world_camera=transforms,
                project_camera_point=camera.project,
                image_size=camera.image_size,
                minimum_coverage_fraction=args.minimum_coverage_fraction,
                provenance=(Provenance(str(input_report_path)),),
            )
            camera_results[role] = {
                "status": result.status.value,
                "metrics": dict(result.metrics),
                "thresholds": dict(result.thresholds),
            }
            midpoint = (start + end) // 2
            local_midpoint = midpoint - start
            mid_transform = transforms[local_midpoint]
            projected = {}
            for column, label in enumerate(LABELS):
                pixel = None
                reason = "missing tracked wrist"
                if present[local_midpoint, column] and mid_transform is None:
                    reason = "missing accepted camera pose"
                elif present[local_midpoint, column]:
                    camera_point = mid_transform.inverse().apply(
                        points[local_midpoint, column], unit=DistanceUnit.METERS
                    )
                    pixel = camera.project(camera_point)
                    reason = "outside calibrated projection domain" if pixel is None else None
                projected[label] = {
                    "pixel_xy": None if pixel is None else pixel.tolist(),
                    "in_image": bool(
                        pixel is not None
                        and 0 <= pixel[0] <= camera.image_size[0] - 1
                        and 0 <= pixel[1] <= camera.image_size[1] - 1
                    ),
                    "missing_reason": reason,
                }
            camera_results[role]["midpoint_projection"] = {
                "comind_sync_frame_index": midpoint,
                "mp4_frame_index": int(data["mp4_frame_index"][midpoint]),
                "device_timestamp_ns": int(data["device_timestamp_ns"][midpoint]),
                "mapping_status": str(data["mapping_status"][midpoint]),
                "points": projected,
            }
        rows.append(
            {
                "annotation_id": handover["annotation_id"],
                "start_frame": start,
                "end_frame": end,
                "cameras": camera_results,
            }
        )
    result = {
        "recording_id": recording_id,
        "generation_id": report["generation_id"],
        "timeline": report["timeline"],
        "shared_world": report["shared_world"],
        "selected_annotation_id": report["handover_ranking"]["selected_annotation_id"],
        "boundary_evaluation": "Both annotated boundary frames included for QC; source endpoint semantics unchanged",
        "landmark": "Aria wrist index 5 for helper/leader left/right hands",
        "interpretation": "In-image calibrated projection coverage; no occlusion or actual visibility was evaluated. Missing points/camera poses excluded from evaluable denominator, retained in all-frame coverage. Zero-confidence present hands remain included.",
        "inputs": sources,
        "validation_report": str(input_report_path),
        "validation_report_sha256": digest(input_report_path),
        "handovers": rows,
        "raw_reads": False,
        "image_decodes": 0,
    }
    output = outside_raw(
        args.output or Path(f"outputs/comind_v0/{recording_id}_projection_qc.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"report": str(output), "handover_count": len(rows)}))


if __name__ == "__main__":
    main()
