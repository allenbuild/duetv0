#!/usr/bin/env python3
"""Audit reported clocks/spatial plausibility once and cache bounded derived arrays."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import sys
import zlib
from pathlib import Path
from uuid import UUID

import numpy as np

from duet.adapters.comind.layout import ParticipantLayout, discover_recording
from duet.adapters.comind.mps import HAND_LANDMARK_NAMES, MPS_TRANSFORM_TOLERANCE, device_clock
from duet.adapters.comind.timing import UTC_EVIDENCE, clock_pair_statistics, percentiles
from duet.adapters.comind.validation import (
    _integer_column,
    _strict_csv_chunks,
    ensure_output_outside_raw,
)
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations
from duet.qc.comind_spatial import check_device_motion, check_hand_plausibility
from duet.qc.result import QCResult
from duet.schemas.common import Provenance

DEFAULT_RECORDING = "43276420-701f-4731-b9ab-bebc7fd14994"


def semantic_trajectory_source(
    paths: ParticipantLayout, *, recording_id: str, processed_root: Path
) -> tuple[Path, str | None]:
    """Choose an individual official trajectory or the previously verified recovery.

    Published extracted trajectories need no archive or standard-MPS trajectory.
    The old helper recovery is the sole fallback with independently established
    CRC provenance; a similarly named derived file is never trusted implicitly.
    """
    if paths.multislam_dir is not None:
        path = paths.multislam_dir / "closed_loop_trajectory.csv"
        if path.is_file():
            return path, None
    if recording_id == DEFAULT_RECORDING and paths.role == "helper":
        recovered = (
            processed_root
            / recording_id
            / paths.multislam_index
            / "slam"
            / "closed_loop_trajectory.csv"
        )
        if recovered.is_file():
            return recovered, "529bb29f"
    raise FileNotFoundError(
        f"{paths.role}: semantic cache requires the individual Multi-SLAM "
        "closed_loop_trajectory.csv; no unverified archive recovery is used"
    )


class ChecksumReader(io.RawIOBase):
    """Compute CRC during the one existing sequential read, without rereading bytes."""

    def __init__(self, path: Path):
        super().__init__()
        self.source = path.open("rb")
        self.crc32 = 0
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer: bytearray) -> int:
        count = self.source.readinto(buffer)
        if count:
            self.crc32 = zlib.crc32(memoryview(buffer)[:count], self.crc32)
            self.bytes_read += count
        return count

    def close(self) -> None:
        self.source.close()
        super().close()


def read_trajectory_once(
    path: Path, *, expected_crc: str | None
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    """Read compact numeric columns once; validate every source transform in chunks."""
    initial = path.stat()
    raw = ChecksumReader(path)
    arrays: dict[str, list[np.ndarray]] = {
        name: [] for name in ("device_us", "utc_ns", "translation", "quaternion", "quality")
    }
    graphs: set[str] = set()
    count = 0
    max_orthonormality = max_determinant = 0.0
    with io.TextIOWrapper(io.BufferedReader(raw), encoding="utf-8", newline="") as text:
        reader = csv.reader(text)
        header = next(reader)
        required = {"tracking_timestamp_us", "utc_timestamp_ns", "graph_uid", "quality_score"}
        required.update(f"t{axis}_world_device" for axis in "xyz")
        required.update(f"q{axis}_world_device" for axis in "xyzw")
        if len(set(header)) != len(header) or not required <= set(header):
            raise ValueError(f"invalid trajectory header: {path}")
        for chunk in _strict_csv_chunks(reader, header=header, chunk_size=50_000, source=str(path)):
            device = _integer_column(chunk["tracking_timestamp_us"], "tracking_timestamp_us")
            utc = _integer_column(chunk["utc_timestamp_ns"].replace("", "-1"), "utc_timestamp_ns")
            translation = chunk[[f"t{axis}_world_device" for axis in "xyz"]].to_numpy(dtype=float)
            quaternion = chunk[[f"q{axis}_world_device" for axis in "xyzw"]].to_numpy(dtype=float)
            quality = chunk["quality_score"].to_numpy(dtype=float)
            if not np.isfinite(translation).all() or not np.isfinite(quality).all():
                raise ValueError(f"nonfinite trajectory values in {path}")
            if np.any((quality < 0) | (quality > 1)):
                raise ValueError("trajectory quality_score is outside the documented [0, 1] range")
            rotation = quaternion_xyzw_batch_to_rotations(
                quaternion, norm_tolerance=MPS_TRANSFORM_TOLERANCE
            )
            ortho = np.max(np.abs(np.swapaxes(rotation, 1, 2) @ rotation - np.eye(3)))
            determinant = np.max(np.abs(np.linalg.det(rotation) - 1))
            max_orthonormality = max(max_orthonormality, float(ortho))
            max_determinant = max(max_determinant, float(determinant))
            if ortho > MPS_TRANSFORM_TOLERANCE or determinant > MPS_TRANSFORM_TOLERANCE:
                raise ValueError(f"invalid rigid rotations in {path}")
            if not chunk["graph_uid"].str.strip().ne("").all():
                raise ValueError("empty graph UID")
            graphs.update(chunk["graph_uid"].unique())
            for key, value in (
                ("device_us", device),
                ("utc_ns", utc),
                ("translation", translation),
                ("quaternion", quaternion),
                ("quality", quality),
            ):
                arrays[key].append(value)
            count += len(chunk)
    final = path.stat()
    crc = f"{raw.crc32:08x}"
    if expected_crc is not None and crc != expected_crc.lower():
        raise ValueError(
            f"recovered trajectory CRC mismatch: expected {expected_crc}, observed {crc}"
        )
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError("trajectory changed during its read")
    if not count or len(graphs) != 1:
        raise ValueError("trajectory must contain exactly one nonempty world graph")
    result = {key: np.concatenate(parts) for key, parts in arrays.items()}
    if np.any(np.diff(result["device_us"]) <= 0):
        raise ValueError("real-data cache requires strictly increasing device timestamps")
    return result, {
        "path": str(path.resolve()),
        "bytes": raw.bytes_read,
        "crc32": crc,
        "expected_crc32": expected_crc,
        "crc_verified_against_recovered_member": expected_crc is not None,
        "mtime_ns": final.st_mtime_ns,
        "unchanged_during_read": True,
        "row_count": count,
        "graph_uid": next(iter(graphs)),
        "max_rotation_orthonormality_error": max_orthonormality,
        "max_rotation_determinant_error": max_determinant,
        "transform_invalid_count": 0,
    }


def read_hands_once(path: Path) -> dict[str, np.ndarray]:
    """Retain compact 21-landmark arrays; -1 placeholders are never interpreted.

    As in the canonical parser, any blank coordinate makes that entire hand
    explicitly missing, while nonfinite tokens in a present hand are errors.
    A blank confidence is unknown and does not discard finite geometry.
    """
    times, points, confidences = [], [], []
    with path.open(encoding="utf-8", newline="") as text:
        reader = csv.reader(text)
        header = next(reader)
        keys = [
            f"t{axis}_{side}_landmark_{index}_device"
            for side in ("left", "right")
            for index in range(21)
            for axis in "xyz"
        ]
        required = {
            "tracking_timestamp_us",
            "left_tracking_confidence",
            "right_tracking_confidence",
            *keys,
        }
        if len(set(header)) != len(header) or not required <= set(header):
            raise ValueError("invalid hand CSV header")
        for chunk in _strict_csv_chunks(reader, header=header, chunk_size=10_000, source=str(path)):
            timestamp = _integer_column(chunk["tracking_timestamp_us"], "tracking_timestamp_us")
            raw_confidence = chunk[["left_tracking_confidence", "right_tracking_confidence"]]
            confidence = raw_confidence.replace("", "nan").to_numpy(dtype=float)
            if np.any((raw_confidence.to_numpy() != "") & ~np.isfinite(confidence)) or np.any(
                np.isfinite(confidence) & (confidence != -1) & ((confidence < 0) | (confidence > 1))
            ):
                raise ValueError("invalid hand confidence")
            raw_points = chunk[keys].to_numpy(dtype=object).reshape(-1, 2, 21, 3)
            missing = (confidence == -1) | (raw_points == "").any(axis=(2, 3))
            raw_points[missing] = "nan"
            values = raw_points.astype(float)
            if not np.isfinite(values[~missing]).all():
                raise ValueError(
                    "nonfinite coordinates for a present hand; only blank coordinates denote missing geometry"
                )
            values[missing] = np.nan
            times.append(timestamp)
            points.append(values)
            confidences.append(confidence)
    if not times:
        raise ValueError("empty hand stream")
    return {
        "device_us": np.concatenate(times),
        "points": np.concatenate(points),
        "confidence": np.concatenate(confidences),
    }


def nearest_indices(source_us: np.ndarray, queries_us: np.ndarray) -> np.ndarray:
    """Exact integer nearest selection; ties prefer earlier source time."""
    right = np.searchsorted(source_us, queries_us)
    right = np.minimum(right, len(source_us) - 1)
    left = np.maximum(right - 1, 0)
    choose_left = np.abs(source_us[left] - queries_us) <= np.abs(source_us[right] - queries_us)
    return np.where(choose_left, left, right)


def qc_dict(result: QCResult) -> dict[str, object]:
    return {
        "status": result.status.value,
        "metrics": dict(result.metrics),
        "thresholds": dict(result.thresholds),
        "message": result.message,
    }


def verify_cache_report(processed_root: Path, recording_id: str) -> dict[str, object]:
    """Validate derived cache structure/content without rereading source trajectories.

    Original CRC/geometry statistics remain generation-time evidence, explicitly
    not a fresh integrity claim about raw/recovered source files.
    """
    cache_root = processed_root / recording_id / "semantics"
    report = json.loads((cache_root / "manifest.json").read_text())
    if report.get("recording_id") != recording_id or report["cross_device_time"]["verified"]:
        raise ValueError("cache manifest cannot silently establish cross-device synchronization")
    verified = {}
    for role in ("helper", "leader"):
        participant = report["participants"][role]
        with np.load(cache_root / f"{role}.npz", allow_pickle=False) as cache:
            times = cache["trajectory_device_us"]
            if (
                times.dtype.kind != "i"
                or times.ndim != 1
                or not len(times)
                or np.any(np.diff(times) <= 0)
            ):
                raise ValueError(
                    "cache trajectory device timestamps must be strictly increasing integers"
                )
            n = len(times)
            if (
                cache["trajectory_translation_m"].shape != (n, 3)
                or not np.isfinite(cache["trajectory_translation_m"]).all()
            ):
                raise ValueError("invalid cached translations")
            quaternion_xyzw_batch_to_rotations(cache["trajectory_quaternion_xyzw"])
            quality = cache["trajectory_quality_score"]
            if (
                quality.shape != (n,)
                or not np.isfinite(quality).all()
                or np.any((quality < 0) | (quality > 1))
            ):
                raise ValueError("invalid cached trajectory quality scores")
            utc = cache["trajectory_reported_utc_ns"]
            if utc.shape != (n,) or utc.dtype.kind != "i":
                raise ValueError("cache UTC values must remain integer source pairs")
            hand_times = cache["hand_device_us"]
            if hand_times.ndim != 1 or hand_times.dtype.kind != "i":
                raise ValueError("cache hand timestamps must remain integers")
            h = len(hand_times)
            points = cache["hand_points_device_m"]
            shared = cache["hand_points_shared_m"]
            confidence = cache["hand_confidence"]
            if (
                points.shape != (h, 2, 21, 3)
                or shared.shape != points.shape
                or confidence.shape != (h, 2)
            ):
                raise ValueError("invalid cache hand dimensions")
            for side in range(2):
                check_hand_plausibility(
                    hand_times,
                    points[:, side],
                    confidence[:, side],
                    max_device_distance_m=2,
                    max_wrist_speed_m_s=5,
                    max_gap_seconds=0.1,
                )
            if (
                not np.isnan(points[confidence == -1]).all()
                or not np.isnan(shared[confidence == -1]).all()
            ):
                raise ValueError("missing hand placeholders cannot become valid cached points")
            residual = cache["hand_pose_device_us"] - hand_times
            if not np.array_equal(residual, cache["hand_pose_residual_us"]):
                raise ValueError("cache hand-pose residuals disagree with original device times")
            accepted = cache["hand_pose_accepted"]
            maximum = participant["same_device_hand_pose_matching"]["max_gap_seconds"]
            if not np.array_equal(accepted, np.abs(residual) <= maximum * 1e6):
                raise ValueError("cache acceptance disagrees with explicit same-device gap")
            if not np.isnan(shared[~accepted]).all():
                raise ValueError("gap-rejected poses must not produce shared hand coordinates")
            graph = str(cache["graph_uid"].item())
            if (
                graph != participant["source"]["graph_uid"]
                or graph != report["shared_world"]["graph_uid"]
            ):
                raise ValueError("cache world graph does not match the complete source audit")
            verified[role] = {
                "trajectory_samples": n,
                "hand_rows": h,
                "graph_uid": graph,
                "cache_bytes": (cache_root / f"{role}.npz").stat().st_size,
            }
    return {
        "cache_validation": "pass",
        "participants": verified,
        "source_files_reread": False,
        "source_integrity_basis": "Generation-time complete reads and recorded CRC values; source CRCs are not refreshed in cache verification",
        "cross_device_time_verified": False,
        "report": str((cache_root / "manifest.json").resolve()),
    }


def analyze_role(
    trajectory: dict[str, np.ndarray],
    hands: dict[str, np.ndarray],
    *,
    role: str,
    source: dict[str, object],
    cache_path: Path,
    cache_max_hz: float,
    max_hand_gap_seconds: float,
) -> dict[str, object]:
    """Compute all dynamics in device time and write only a derived visualization cache."""
    d = trajectory["device_us"]
    provenance = Provenance(
        str(source["path"]), "Full source traversal; meter shared-world positions"
    )
    clock_stats = clock_pair_statistics(
        d, trajectory["utc_ns"], offset_jump_threshold_seconds=0.001
    )
    device_qc = check_device_motion(
        d,
        trajectory["translation"],
        max_speed_m_s=2.0,
        max_step_m=0.05,
        max_gap_seconds=0.1,
        provenance=(provenance,),
    )
    hand_qc = {
        side: qc_dict(
            check_hand_plausibility(
                hands["device_us"],
                hands["points"][:, index],
                hands["confidence"][:, index],
                max_device_distance_m=2.0,
                max_wrist_speed_m_s=5.0,
                max_gap_seconds=0.1,
                provenance=(provenance,),
            )
        )
        for index, side in enumerate(("left", "right"))
    }
    matched = nearest_indices(d, hands["device_us"])
    residual = d[matched] - hands["device_us"]
    accepted = np.abs(residual) <= max_hand_gap_seconds * 1e6
    rotations = quaternion_xyzw_batch_to_rotations(trajectory["quaternion"][matched])
    shared_points = (
        np.einsum("nij,nhkj->nhki", rotations, hands["points"])
        + trajectory["translation"][matched, None, None, :]
    )
    shared_points[~accepted] = np.nan
    interval_us = math.ceil(1_000_000 / cache_max_hz)
    bins = (d - d[0]) // interval_us
    keep = np.flatnonzero(np.r_[True, bins[1:] != bins[:-1]])
    if keep[-1] != len(d) - 1:
        keep = np.r_[keep, len(d) - 1]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        trajectory_device_us=d[keep],
        trajectory_reported_utc_ns=trajectory["utc_ns"][keep],
        trajectory_translation_m=trajectory["translation"][keep],
        trajectory_quaternion_xyzw=trajectory["quaternion"][keep],
        trajectory_quality_score=trajectory["quality"][keep],
        hand_device_us=hands["device_us"],
        hand_points_device_m=hands["points"],
        hand_confidence=hands["confidence"],
        hand_points_shared_m=shared_points,
        hand_pose_device_us=d[matched],
        hand_pose_reported_utc_ns=trajectory["utc_ns"][matched],
        hand_pose_residual_us=residual,
        hand_pose_accepted=accepted,
        graph_uid=np.asarray(source["graph_uid"]),
        hand_landmark_names=np.asarray(HAND_LANDMARK_NAMES),
    )
    return {
        "source": source,
        "reported_clock": clock_stats,
        "device_motion": qc_dict(device_qc),
        "hands": hand_qc,
        "same_device_hand_pose_matching": {
            "query_count": len(matched),
            "accepted_count": int(accepted.sum()),
            "rejected_count": int((~accepted).sum()),
            "max_gap_seconds": max_hand_gap_seconds,
            "signed_residual_us": percentiles(residual),
            "absolute_residual_us": percentiles(np.abs(residual)),
        },
        "cache": {
            "path": str(cache_path.resolve()),
            "bytes": cache_path.stat().st_size,
            "trajectory_sample_count": len(keep),
            "sampling": f"First source sample in each {interval_us}us own-device bin, plus endpoints",
            "hand_row_count": len(hands["device_us"]),
            "missing_points_encoding": "NaN; confidence -1 raw missing sentinel retained",
            "missing_data_policy": "Confidence -1 or a blank source coordinate makes the whole hand missing; nonfinite present coordinates fail; blank confidence remains unknown with finite geometry retained",
            "participant": role,
            "cross_device_time_binding": None,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--recording-id", default=DEFAULT_RECORDING)
    parser.add_argument("--processed-root", type=Path, default=Path("data/processed/comind"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/comind_semantics"))
    parser.add_argument("--cache-max-hz", type=float, default=30)
    parser.add_argument("--max-hand-gap-seconds", type=float, default=0.01)
    parser.add_argument(
        "--verify-cache",
        action="store_true",
        help="Check derived arrays/report without rereading source files",
    )
    args = parser.parse_args()
    if args.verify_cache:
        print(json.dumps(verify_cache_report(args.processed_root, args.recording_id), indent=2))
        return
    if str(UUID(args.recording_id)) != args.recording_id:
        raise ValueError("recording ID must be a canonical UUID")
    if not math.isfinite(args.cache_max_hz) or args.cache_max_hz <= 0 or args.cache_max_hz > 1000:
        raise ValueError("cache-max-hz must be finite and in (0,1000]")
    if not math.isfinite(args.max_hand_gap_seconds) or args.max_hand_gap_seconds < 0:
        raise ValueError("max-hand-gap-seconds must be finite and nonnegative")
    layout = discover_recording(args.dataset_root, args.recording_id, minimal=True)
    ensure_output_outside_raw(args.processed_root, args.dataset_root)
    ensure_output_outside_raw(args.output_root, args.dataset_root)
    cache_root = args.processed_root / args.recording_id / "semantics"
    report: dict[str, object] = {
        "recording_id": args.recording_id,
        "cross_device_time": {
            "status": "UNRESOLVED",
            "verified": False,
            "accuracy_bound_seconds": None,
            "reason": "Recorded RTC pairs and small fit residuals do not verify common physical capture time; external TimeCode/TICSync or documented CoMind sync mapping is missing",
            "evidence": [UTC_EVIDENCE],
        },
        "participants": {},
    }
    for role in ("helper", "leader"):
        paths = layout.participants[role]
        path, expected_crc = semantic_trajectory_source(
            paths, recording_id=args.recording_id, processed_root=args.processed_root
        )
        print(
            f"{role}: one-pass full paired-clock/geometry read, then compact hand read", flush=True
        )
        trajectory, source = read_trajectory_once(path, expected_crc=expected_crc)
        hands = read_hands_once(paths.hands_path)
        participant_report = analyze_role(
            trajectory,
            hands,
            role=role,
            source=source,
            cache_path=cache_root / f"{role}.npz",
            cache_max_hz=args.cache_max_hz,
            max_hand_gap_seconds=args.max_hand_gap_seconds,
        )
        participant_report["device_clock"] = device_clock(args.recording_id, role).name
        participant_report["reported_utc_clock"] = (
            f"comind/{args.recording_id}/{role}/utc_unverified"
        )
        report["participants"][role] = participant_report
        del trajectory, hands
    roles = report["participants"]
    graphs = {roles[role]["source"]["graph_uid"] for role in roles}
    if len(graphs) != 1:
        raise ValueError("Multi-SLAM graph membership differs across participants")
    graph = next(iter(graphs))
    report["shared_world"] = {
        "status": "VERIFIED",
        "graph_uid": graph,
        "frame": f"comind/{args.recording_id}/multislam/{graph}",
    }
    ranges = [roles[role]["reported_clock"]["reported_utc_range_ns"] for role in roles]
    start, end = max(value[0] for value in ranges), min(value[1] for value in ranges)
    report["reported_utc_overlap"] = {
        "classification": "INFERRED coarse reported-wall-clock extent only",
        "start_ns": start,
        "end_ns": end,
        "duration_seconds": max(0, end - start) / 1e9,
        "simultaneous_capture_verified": False,
    }
    report["cross_person_distance"] = {
        "status": "UNAVAILABLE",
        "reason": "No verified common physical timeline; no time-index or UTC proximity inference used",
    }
    report["handover_distance_qc"] = {
        "status": "UNAVAILABLE",
        "reason": "Annotation-to-device and cross-device mappings remain unresolved",
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    output = args.output_root / f"{args.recording_id}.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (cache_root / "manifest.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "report": str(output.resolve()),
                "cache_manifest": str((cache_root / "manifest.json").resolve()),
                "reported_utc_overlap": report["reported_utc_overlap"],
                "cross_device_time": report["cross_device_time"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as error:
        print(f"Semantic validation failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
