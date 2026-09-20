#!/usr/bin/env python3
"""Audit existing VERIFIED image correspondences using local compact caches only.

This command does not open videos/VRS files, import a remote reader, contact a
server, run temporal matching, or modify frame maps and fingerprint caches.
Native fingerprints retain their original orientation. All four rotations are
compared at the same already-VERIFIED native source identity for each MP4 frame.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID

import numpy as np

from duet.adapters.comind.frame_map import load_frame_map
from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.vrs import verify_clockwise_export_anchors, verify_export_pixel_geometry

ROOT = Path(__file__).resolve().parents[1]
THRESHOLDS = {
    "maximum_rmse": 1.5,
    "minimum_rotation_margin": 5.0,
    "minimum_anchor_count": 20,
    "maximum_anchor_gap_frames": 900,
}
ROTATION_LABELS = ("identity", "clockwise_90", "rotation_180", "counterclockwise_90")


def _local(path: Path) -> Path:
    path = path.resolve()
    if path.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("orientation audit may access derived caches only, never data/raw")
    return path


def _digest(path: Path) -> str:
    with _local(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json(path: Path):
    return json.loads(_local(path).read_text())


def _array(path: Path, dtype, shape: tuple[int, ...]) -> np.ndarray:
    array = np.load(_local(path), mmap_mode="r", allow_pickle=False)
    if array.dtype != dtype or array.shape != shape:
        raise ValueError(f"incompatible cached array dimensions or dtype: {path.name}")
    return array


def _summary(values: np.ndarray) -> dict[str, float]:
    return dict(
        zip(
            ("minimum", "median", "p95", "maximum"),
            map(float, np.quantile(values, [0, 0.5, 0.95, 1])),
            strict=True,
        )
    )


def _cache(directory: Path, *, count: int, allowed_kinds: tuple[str, ...]) -> dict:
    metadata_path = directory / "source.json"
    metadata = _json(metadata_path)
    if (
        metadata.get("kind") not in allowed_kinds
        or metadata.get("frame_count") != count
        or metadata.get("fingerprint_size") != 32
        or metadata.get("method") != "FFmpeg gray 32x32, native orientation for VRS"
    ):
        raise ValueError("fingerprint cache method, kind, or dimensions are incompatible")
    return {
        "fingerprints": _array(directory / "fingerprints.npy", np.dtype("uint8"), (count, 32, 32)),
        "valid": _array(directory / "valid.npy", np.dtype("bool"), (count,)),
        "done": _array(directory / "done.npy", np.dtype("bool"), (count,)),
        "pts": _array(directory / "pts.npy", np.dtype("int64"), (count,)),
        "identity": {
            "directory": str(directory.resolve()),
            "source_metadata_sha256": _digest(metadata_path),
            "method": metadata["method"],
            "kind": metadata["kind"],
            # Source URLs are deliberately not copied into the derived report.
        },
    }


def orientation_only_samples(
    mp4: dict, native: dict, mapping, frames: list[int], *, candidate_radius: int = 128
) -> list[dict]:
    """Audit explicitly requested cached images for rotation, without assigning time.

    A bounded search uses only existing completed fingerprints. The winning
    native image must have both adjacent cached images unless at a true source
    endpoint, and must pass the original visual residual/uniqueness thresholds.
    This does not create a VERIFIED temporal correspondence.
    """
    rows = []
    native_count = len(native["fingerprints"])
    for frame in frames:
        if (
            type(frame) is not int
            or not 0 <= frame < mapping.frame_count
            or mapping.status[frame] != "UNRESOLVED"
            or not (mp4["valid"][frame] and mp4["done"][frame])
            or mp4["pts"][frame] != mapping.mp4_pts[frame]
        ):
            raise ValueError("orientation-only requests require valid cached unresolved MP4 frames")
        lo, hi = max(0, frame - candidate_radius), min(native_count, frame + candidate_radius + 1)
        candidates = np.flatnonzero(native["valid"][lo:hi] & native["done"][lo:hi]) + lo
        if len(candidates) < 2:
            raise ValueError("orientation-only comparison needs at least two cached native images")
        target = mp4["fingerprints"][frame].astype(np.float32)
        source = native["fingerprints"][candidates]
        errors = np.stack(
            [
                np.sqrt(
                    np.mean(
                        (target - np.rot90(source, -rotation, axes=(1, 2)).astype(np.float32)) ** 2,
                        axis=(1, 2),
                    )
                )
                for rotation in range(4)
            ],
            axis=1,
        )
        ranked = np.argsort(errors[:, 1], kind="stable")
        source_position = int(ranked[0])
        source_index = int(candidates[source_position])
        same_source_scores = errors[source_position]
        best, second = map(float, errors[ranked[:2], 1])
        margin = float(np.min(same_source_scores[[0, 2, 3]]) - same_source_scores[1])
        neighbors = [i for i in (source_index - 1, source_index + 1) if 0 <= i < native_count]
        if (
            best > THRESHOLDS["maximum_rmse"]
            or margin < THRESHOLDS["minimum_rotation_margin"]
            or second == best
            or second - best < 0.08
            or second < best * 1.15
            or not all(native["valid"][index] and native["done"][index] for index in neighbors)
            or source_index in (lo, hi - 1)
            and source_index not in (0, native_count - 1)
        ):
            raise ValueError(
                "cached orientation-only candidate fails residual, separation, or boundary gates"
            )
        rows.append(
            {
                "mp4_frame_index": frame,
                "orientation_candidate_vrs_rgb_frame_index": source_index,
                "temporal_assignment": False,
                "original_map_status": str(mapping.status[frame]),
                "clockwise_quarter_turns": int(np.argmin(same_source_scores)),
                "best_rmse": best,
                "second_rmse": float(np.sort(same_source_scores)[1]),
                "cw90_other_rotation_margin": margin,
                "all_rotations": [
                    {
                        "clockwise_quarter_turns": rotation,
                        "best_rmse": float(same_source_scores[rotation]),
                    }
                    for rotation in range(4)
                ],
                "source_image_competition": {
                    "cached_candidate_indices": candidates.tolist(),
                    "candidate_window_inclusive": [lo, hi - 1],
                    "candidate_radius": candidate_radius,
                    "second_native_image_rmse": second,
                    "native_image_margin": second - best,
                    "native_image_ratio": None if best == 0 else second / best,
                    "required_margin": 0.08,
                    "required_ratio": 1.15,
                    "cached_adjacent_native_indices": neighbors,
                    "scope": "Uniqueness only within the explicitly listed cached candidates; no capture time assigned",
                },
                "mp4_fingerprint_sha256": hashlib.sha256(
                    mp4["fingerprints"][frame].tobytes()
                ).hexdigest(),
                "native_fingerprint_sha256": hashlib.sha256(
                    native["fingerprints"][source_index].tobytes()
                ).hexdigest(),
            }
        )
    if len({row["mp4_frame_index"] for row in rows}) != len(rows):
        raise ValueError("orientation-only anchor frame indices must be unique")
    return rows


def audit_role(
    processed_root: Path,
    recording_id: str,
    role: str,
    *,
    orientation_frames: list[int] | None = None,
) -> dict:
    """Measure four rotations at every existing VERIFIED correspondence, without searches."""
    root = _local(processed_root / recording_id)
    directory = root / "frame_maps"
    map_path = directory / f"{role}_frame_map.npz"
    metadata = _json(map_path.with_suffix(".json"))
    mapping = load_frame_map(_local(map_path))
    if mapping.recording_id != recording_id or mapping.participant != role:
        raise ValueError("frame-map recording or participant differs from requested audit")
    frame_map_hash = _digest(map_path)
    verified = np.flatnonzero(mapping.status == "VERIFIED")
    if not len(verified):
        raise ValueError("orientation audit requires existing VERIFIED correspondences")
    native_indices = mapping.vrs_rgb_frame_index[verified]
    native_count = metadata["vrs_rgb_frame_count"]
    if np.any(native_indices >= native_count):
        raise ValueError("verified native indices exceed cached native inventory")
    native_directories = [
        directory / "fingerprints" / f"{role}_{kind}"
        for kind in ("vrs", "remote_vrs")
        if (directory / "fingerprints" / f"{role}_{kind}").is_dir()
    ]
    if len(native_directories) != 1:
        raise ValueError("exactly one native-orientation fingerprint cache is required")
    mp4 = _cache(
        directory / "fingerprints" / f"{role}_mp4",
        count=mapping.frame_count,
        allowed_kinds=("mp4",),
    )
    native = _cache(native_directories[0], count=native_count, allowed_kinds=("vrs", "remote_vrs"))
    if not (
        mp4["valid"][verified].all()
        and mp4["done"][verified].all()
        and native["valid"][native_indices].all()
        and native["done"][native_indices].all()
    ):
        raise ValueError("a VERIFIED correspondence lacks completed valid cached fingerprints")
    if not np.array_equal(mp4["pts"][verified], mapping.mp4_pts[verified]):
        raise ValueError("cached MP4 presentation timestamps differ from verified map")
    timestamp_path = root / "vrs" / f"{role}_rgb_metadata.npz"
    with np.load(_local(timestamp_path), allow_pickle=False) as timestamps:
        if (
            str(timestamps["clock_domain"]) != device_clock(recording_id, role).name
            or str(timestamps["stream_id"]) != "214-1"
            or not np.array_equal(timestamps["vrs_rgb_frame_index"], np.arange(native_count))
            or not np.array_equal(
                timestamps["device_timestamps_ns"][native_indices],
                mapping.vrs_device_timestamp_ns[verified],
            )
            or (
                "metadata_known" in timestamps
                and not timestamps["metadata_known"][native_indices].all()
            )
        ):
            raise ValueError("native timestamps or participant-local stream identity disagree")
    errors = np.empty((len(verified), 4), dtype=np.float64)
    mp4_digest, native_digest = hashlib.sha256(), hashlib.sha256()
    for start in range(0, len(verified), 256):
        target = mp4["fingerprints"][verified[start : start + 256]]
        source = native["fingerprints"][native_indices[start : start + 256]]
        mp4_digest.update(target.tobytes())
        native_digest.update(source.tobytes())
        for rotation in range(4):
            delta = target.astype(np.float32) - np.rot90(source, -rotation, axes=(1, 2)).astype(
                np.float32
            )
            errors[start : start + len(target), rotation] = np.sqrt(
                np.mean(delta * delta, axis=(1, 2))
            )
    winners = np.argmin(errors, axis=1)
    cw_margin = np.min(errors[:, [0, 2, 3]], axis=1) - errors[:, 1]
    pass_pairs = (
        (winners == 1)
        & (errors[:, 1] <= THRESHOLDS["maximum_rmse"])
        & (cw_margin >= THRESHOLDS["minimum_rotation_margin"])
    )
    samples = []
    for position, frame in enumerate(verified):
        ranked = np.sort(errors[position])
        samples.append(
            {
                "mp4_frame_index": int(frame),
                "vrs_rgb_frame_index": int(native_indices[position]),
                "device_timestamp_ns": int(mapping.vrs_device_timestamp_ns[frame]),
                "mapping_status": "VERIFIED",
                "clockwise_quarter_turns": int(winners[position]),
                "best_rmse": float(ranked[0]),
                "second_rmse": float(ranked[1]),
                "cw90_other_rotation_margin": float(cw_margin[position]),
                "all_rotations": [
                    {
                        "clockwise_quarter_turns": rotation,
                        "best_rmse": float(errors[position, rotation]),
                    }
                    for rotation in range(4)
                ],
            }
        )
    gaps = np.diff(verified)
    excessive_gaps = [
        [int(verified[index]), int(verified[index + 1])]
        for index in np.flatnonzero(gaps > THRESHOLDS["maximum_anchor_gap_frames"])
    ]
    coverage_passes = (
        verified[0] == 0
        and verified[-1] == mapping.frame_count - 1
        and len(verified) >= THRESHOLDS["minimum_anchor_count"]
        and not excessive_gaps
    )
    sample_by_frame = {row["mp4_frame_index"]: row for row in samples}
    anchor_path = directory / f"{role}_anchors.json"
    anchors = _json(anchor_path)
    initial_offset = anchors[0]["vrs_index"] - anchors[0]["mp4_frame_index"]
    coarse_rows = []
    for anchor in anchors:
        frame = anchor["mp4_frame_index"]
        actual = sample_by_frame.get(frame)
        center = min(native_count - 1, frame + initial_offset)
        window = [max(0, center - 16), min(native_count - 1, center + 16)]
        coarse_rows.append(
            {
                "mp4_frame_index": frame,
                "original_vrs_index": anchor["vrs_index"],
                "original_best_rmse": anchor["best_rmse"],
                "original_clockwise_quarter_turns": anchor["clockwise_quarter_turns"],
                "original_candidate_window_inclusive": window,
                "final_map_status": str(mapping.status[frame]),
                "final_verified_pair": actual,
                "original_source_differs_from_verified_map": None
                if actual is None
                else anchor["vrs_index"] != actual["vrs_rgb_frame_index"],
                "verified_source_outside_original_window": None
                if actual is None
                else not window[0] <= actual["vrs_rgb_frame_index"] <= window[1],
            }
        )
    pixel_path = directory / f"{role}_pixel_geometry.json"
    pixel = _json(pixel_path)
    for sample in pixel["samples"]:
        actual = sample_by_frame.get(sample["mp4_frame_index"])
        if actual is None or any(
            sample[name] != actual[name] for name in ("vrs_rgb_frame_index", "device_timestamp_ns")
        ):
            raise ValueError("cached pixel geometry is not bound to the exact VERIFIED map pair")
    pixel_verification = verify_export_pixel_geometry(
        pixel,
        recording_id=recording_id,
        participant=role,
        frame_count=mapping.frame_count,
        image_size=tuple(metadata["mp4_dimensions"]),
        frame_map_sha256=frame_map_hash,
    )
    orientation_rows = orientation_only_samples(mp4, native, mapping, orientation_frames or [])
    combined = sorted([*samples, *orientation_rows], key=lambda row: row["mp4_frame_index"])
    try:
        combined_verification = verify_clockwise_export_anchors(
            combined, frame_count=mapping.frame_count, **THRESHOLDS
        )
    except ValueError as error:
        combined_verification = {
            "status": "FAIL",
            "reason": str(error),
            "thresholds": THRESHOLDS.copy(),
        }
    return {
        "recording_id": recording_id,
        "participant": role,
        "frame_map_npz_sha256": frame_map_hash,
        "frame_map_metadata_sha256": _digest(map_path.with_suffix(".json")),
        "native_timestamp_cache_sha256": _digest(timestamp_path),
        "original_coarse_anchors_sha256": _digest(anchor_path),
        "mp4_cache": {
            **mp4["identity"],
            "verified_pair_fingerprints_sha256": mp4_digest.hexdigest(),
        },
        "native_cache": {
            **native["identity"],
            "verified_pair_fingerprints_sha256": native_digest.hexdigest(),
        },
        "mp4_frame_count": mapping.frame_count,
        "verified_pair_count": len(verified),
        "thresholds": THRESHOLDS.copy(),
        "orientation_at_verified_pairs": "VERIFIED" if pass_pairs.all() else "FAIL",
        "pairs_passing_cw90_thresholds": int(pass_pairs.sum()),
        "rotation_summary": [
            {
                "clockwise_quarter_turns": rotation,
                "label": ROTATION_LABELS[rotation],
                "rmse": _summary(errors[:, rotation]),
                "winning_pair_count": int(np.sum(winners == rotation)),
            }
            for rotation in range(4)
        ],
        "cw90_other_rotation_margin": _summary(cw_margin),
        "coverage": {
            "first_last_verified_mp4_frame": [int(verified[0]), int(verified[-1])],
            "maximum_verified_frame_gap": int(gaps.max()) if len(gaps) else 0,
            "gaps_exceeding_threshold": excessive_gaps,
            "strict_full_span_anchor_coverage": "PASS" if coverage_passes else "INSUFFICIENT_DATA",
            "scope": "Evidence covers existing VERIFIED pairs only; unresolved rows are never promoted",
        },
        "strict_full_span_orientation_gate": "PASS"
        if coverage_passes and pass_pairs.all()
        else "FAIL",
        "samples": samples,
        "orientation_only_samples": orientation_rows,
        "combined_orientation_verification": combined_verification,
        "coarse_anchor_audit": coarse_rows,
        "coarse_anchor_summary": {
            "count": len(coarse_rows),
            "verified_source_mismatch_count": sum(
                row["original_source_differs_from_verified_map"] is True for row in coarse_rows
            ),
            "verified_source_outside_original_window_count": sum(
                row["verified_source_outside_original_window"] is True for row in coarse_rows
            ),
            "unresolved_anchor_indices": [
                row["mp4_frame_index"] for row in coarse_rows if row["final_verified_pair"] is None
            ],
            "search_provenance": "build_comind_frame_map.sparse_anchors: initial first-frame offset, fixed +/-16 candidate window; diagnostic reconstruction only, never a correspondence",
        },
        "cached_full_resolution_pixel_geometry": {
            "path": str(pixel_path.resolve()),
            "sha256": _digest(pixel_path),
            "raw_all_hypotheses_identity_winner_flag": pixel[
                "zero_shift_unit_scale_wins_all_samples"
            ],
            "strict_existing_verifier": pixel_verification,
        },
        "mirror_hypotheses_tested": False,
        "mirror_scope": "No mirror was inferred or applied; four-rotation evidence is reported without claiming mirrors tested",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording-id", required=True)
    parser.add_argument("--processed-root", type=Path, default=ROOT / "data/processed/comind")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--orientation-only-anchor",
        action="append",
        default=[],
        metavar="ROLE:FRAME",
        help="Explicit bounded cached-image orientation comparison; never updates temporal mapping",
    )
    args = parser.parse_args(argv)
    recording_id = str(UUID(args.recording_id))
    output = _local(
        args.output or ROOT / "outputs/comind_v0" / f"{recording_id}_orientation_audit.json"
    )
    processed = _local(args.processed_root)
    if output.is_relative_to(processed):
        raise ValueError("audit output must be separate from all processed source caches")
    orientation_frames = {role: [] for role in ("helper", "leader")}
    for value in args.orientation_only_anchor:
        try:
            role, frame = value.split(":")
            if role not in orientation_frames or not frame.isdecimal():
                raise ValueError
            orientation_frames[role].append(int(frame))
        except ValueError as error:
            raise ValueError(
                "orientation-only anchors require helper:INDEX or leader:INDEX"
            ) from error
    report = {
        "schema_version": 1,
        "recording_id": recording_id,
        "method": "Four rotations at existing VERIFIED MP4/native fingerprint pairs; no temporal matching",
        "network_requests": 0,
        "network_bytes": 0,
        "video_or_vrs_decodes": 0,
        "maps_anchors_fingerprints_modified": False,
        "participants": {
            role: audit_role(
                processed, recording_id, role, orientation_frames=orientation_frames[role]
            )
            for role in ("helper", "leader")
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, separators=(",", ":"), allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "network_bytes": 0,
                "participants": {
                    role: {
                        "verified_pair_count": result["verified_pair_count"],
                        "orientation_at_verified_pairs": result["orientation_at_verified_pairs"],
                        "coverage": result["coverage"],
                    }
                    for role, result in report["participants"].items()
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
