#!/usr/bin/env python3
"""Study one standard MPS world versus cached Multi-SLAM; never enable a scene.

Reads the standard trajectory once through the existing strict streaming audit.
All paired observations and indices are cached outside raw for repeatable study.
No Multi-SLAM CSV, scan point file, VRS, or video is read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from uuid import UUID

import numpy as np

from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.validation import audit_trajectory, ensure_output_outside_raw
from duet.geometry.alignment import (
    PosePairs,
    fit_pose_alignment,
    pose_alignment_residuals,
    rotation_angles,
)
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import DistanceUnit, FrameId, Provenance

DEFAULT_RECORDING = "43276420-701f-4731-b9ab-bebc7fd14994"
EVIDENCE = (
    "https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/slam/mps_trajectory"
)


def statistics(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("cannot report nonfinite statistics")
    return {
        "count": len(values),
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.percentile(values, 95)) if len(values) else None,
        "max": float(np.max(values)) if len(values) else None,
    }


def summarize_residuals(pairs: PosePairs, transform: RigidTransform) -> dict:
    translation, rotation = pose_alignment_residuals(pairs, transform)
    return {
        "translation_m": statistics(translation),
        "rotation_degrees": statistics(np.rad2deg(rotation)),
    }


def relative_consistency(pairs: PosePairs, timestamps: np.ndarray, stride: int) -> dict:
    """Compare device-i to device-j transforms, independent of any fitted world transform."""
    if len(timestamps) <= stride:
        return {"pair_count": 0}
    source_first = pairs.source_rotations[:-stride].transpose(0, 2, 1)
    destination_first = pairs.destination_rotations[:-stride].transpose(0, 2, 1)
    source_rotation = source_first @ pairs.source_rotations[stride:]
    destination_rotation = destination_first @ pairs.destination_rotations[stride:]
    source_translation = np.einsum(
        "nij,nj->ni",
        source_first,
        pairs.source_positions[stride:] - pairs.source_positions[:-stride],
    )
    destination_translation = np.einsum(
        "nij,nj->ni",
        destination_first,
        pairs.destination_positions[stride:] - pairs.destination_positions[:-stride],
    )
    return {
        "pair_count": len(source_rotation),
        "stride_in_accepted_samples": stride,
        "elapsed_seconds": statistics((timestamps[stride:] - timestamps[:-stride]) / 1e6),
        "translation_m": statistics(
            np.linalg.norm(source_translation - destination_translation, axis=1)
        ),
        "rotation_degrees": statistics(
            np.rad2deg(rotation_angles(source_rotation.transpose(0, 2, 1) @ destination_rotation))
        ),
    }


def scale_diagnostic(pairs: PosePairs, transform: RigidTransform) -> dict:
    """Least-squares position scale with the fitted rotation held fixed; diagnostic only."""
    rotated = pairs.source_positions @ transform.matrix[:3, :3].T
    source = rotated - rotated.mean(axis=0)
    destination = pairs.destination_positions - pairs.destination_positions.mean(axis=0)
    denominator = float(np.sum(source**2))
    if not np.isfinite(denominator):
        raise ValueError("scale diagnostic overflowed")
    if denominator <= np.finfo(float).eps:
        return {"identifiable": False, "reason": "insufficient positional motion"}
    scale = float(np.sum(source * destination) / denominator)
    return {
        "identifiable": True,
        "scale_with_fixed_joint_pose_rotation": scale,
        "difference_from_one": scale - 1,
        "position_residual_after_diagnostic_scale_m": statistics(
            np.linalg.norm(scale * source - destination, axis=1)
        ),
        "applied_to_output_transform": False,
    }


def study(pairs: PosePairs, timestamps: np.ndarray, *, length: float, huber: float) -> dict:
    if len(timestamps) < 20:
        raise ValueError("alignment study requires at least 20 accepted matched poses")
    holdout = np.arange(len(timestamps)) % 5 == 0
    training = pairs.subset(~holdout)
    result = fit_pose_alignment(training, orientation_length_scale_m=length, huber_delta_m=huber)
    heldout_stats = summarize_residuals(pairs.subset(holdout), result.transform)
    translation_limit_m = 0.01
    rotation_limit_degrees = 0.5
    maximum_translation_limit_m = 0.03
    maximum_rotation_limit_degrees = 2.0
    passes = (
        result.converged
        and heldout_stats["translation_m"]["p95"] <= translation_limit_m
        and heldout_stats["translation_m"]["max"] <= maximum_translation_limit_m
        and heldout_stats["rotation_degrees"]["p95"] <= rotation_limit_degrees
        and heldout_stats["rotation_degrees"]["max"] <= maximum_rotation_limit_degrees
    )
    bins = []
    span = int(timestamps[-1]) - int(timestamps[0]) + 1
    time_bins = np.array([(int(t) - int(timestamps[0])) * 10 // span for t in timestamps])
    for index in range(10):
        mask = time_bins == index
        test_mask = mask & holdout
        if not test_mask.any():
            continue
        if np.count_nonzero(mask & ~holdout) < 3:
            bins.append(
                {
                    "bin": index,
                    "status": "INSUFFICIENT_LOCAL_TRAINING_DATA",
                    "heldout_residuals_from_global_fit": summarize_residuals(
                        pairs.subset(test_mask), result.transform
                    ),
                }
            )
            continue
        local = fit_pose_alignment(
            pairs.subset(mask & ~holdout),
            orientation_length_scale_m=length,
            huber_delta_m=huber,
        )
        local_delta = local.transform.matrix @ result.transform.inverse().matrix
        bins.append(
            {
                "bin": index,
                "device_timestamp_us_range": [
                    int(timestamps[mask][0]),
                    int(timestamps[mask][-1]),
                ],
                "heldout_residuals_from_global_fit": summarize_residuals(
                    pairs.subset(test_mask), result.transform
                ),
                "local_training_fit_converged": local.converged,
                "local_heldout_residuals": summarize_residuals(
                    pairs.subset(test_mask), local.transform
                ),
                "local_vs_global_translation_m": float(np.linalg.norm(local_delta[:3, 3])),
                "local_vs_global_rotation_degrees": float(
                    np.rad2deg(rotation_angles(local_delta[:3, :3]))
                ),
            }
        )
    return {
        "classification": "NUMERIC_GATE_PASSED" if passes else "CONSTANT_SE3_NOT_VALIDATED",
        "semantic_registration_verified": False,
        "scene_enabled": False,
        "reason": (
            "Fit is an estimate. Whole-PLY source frame and any BLK export transform contract "
            "remain unresolved; no frame-graph edge or scene registration is created."
        ),
        "method": {
            "objective": "Huber robust joint translation and chordal orientation SE3; scale fixed 1",
            "orientation_length_scale_m": length,
            "huber_delta_m": huber,
            "training_count": int((~holdout).sum()),
            "heldout_count": int(holdout.sum()),
            "holdout_policy": "Every fifth accepted timestamp in source order, chosen before fit",
            "converged": result.converged,
            "iterations": result.iterations,
            "downweighted_training_pairs": result.downweighted_pairs,
        },
        "numeric_gate": {
            "heldout_translation_p95_max_m": translation_limit_m,
            "heldout_translation_max_m": maximum_translation_limit_m,
            "heldout_rotation_p95_max_degrees": rotation_limit_degrees,
            "heldout_rotation_max_degrees": maximum_rotation_limit_degrees,
            "passed": passes,
            "status": "Explicit exploratory QC thresholds, not a dataset accuracy guarantee",
        },
        "T_destination_source_estimate": result.transform.matrix.tolist(),
        "source_frame": pairs.source_world.name,
        "destination_frame": pairs.destination_world.name,
        "device_frame": pairs.device_frame.name,
        "translation_unit": "m",
        "training_residuals": summarize_residuals(training, result.transform),
        "heldout_residuals": heldout_stats,
        "all_matched_residuals": summarize_residuals(pairs, result.transform),
        "scale_diagnostic_heldout": scale_diagnostic(pairs.subset(holdout), result.transform),
        "time_bins": bins,
        "relative_motion_consistency": [
            relative_consistency(pairs, timestamps, stride) for stride in (1, 30, 300)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/comind"))
    parser.add_argument("--recording-id", default=DEFAULT_RECORDING)
    parser.add_argument("--max-gap-us", type=int, default=1000)
    parser.add_argument("--orientation-length-scale-m", type=float, default=1.0)
    parser.add_argument("--huber-delta-m", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.max_gap_us < 0:
        parser.error("max-gap-us must be nonnegative")
    # UUID shape prevents paths from escaping through a recording identifier.
    recording_id = str(UUID(args.recording_id))
    standard_path = (
        args.root
        / "recordings"
        / recording_id
        / "mps_leader_trimmed_vrs/slam/closed_loop_trajectory.csv"
    ).resolve()
    cache_path = Path(f"data/processed/comind/{recording_id}/semantics/leader.npz").resolve()
    paired_path = Path(f"data/processed/comind/{recording_id}/alignment/leader_pose_pairs.npz")
    output_path = args.output or Path(f"outputs/comind_world_alignment/{recording_id}.json")
    for path in (paired_path, output_path):
        ensure_output_outside_raw(path, dataset_root=args.root)
        if path.resolve() in (standard_path, cache_path):
            raise ValueError("output must not overwrite an input")
    initial = standard_path.stat()
    with np.load(cache_path, allow_pickle=False) as cached:
        times = cached["trajectory_device_us"]
        destination_positions = cached["trajectory_translation_m"]
        destination_quaternions = cached["trajectory_quaternion_xyzw"]
        destination_graph = str(cached["graph_uid"].item())
    if times.dtype != np.int64 or times.ndim != 1 or np.any(times[1:] <= times[:-1]):
        raise ValueError("cached participant-local timestamps must be strictly increasing int64")
    if destination_positions.shape != (len(times), 3) or destination_quaternions.shape != (
        len(times),
        4,
    ):
        raise ValueError("cached trajectory dimensions disagree")
    print(
        "Reading standard leader trajectory once; reusing cached Multi-SLAM poses", file=sys.stderr
    )
    with standard_path.open(encoding="utf-8", newline="") as stream:
        audit = audit_trajectory(
            stream,
            provenance=Provenance(str(standard_path)),
            query_timestamps_us=times.tolist(),
        )
    final = standard_path.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError("source changed during read")
    if len(audit.graph_uids) != 1 or audit.stats["transform_invalid_count"]:
        raise ValueError("standard trajectory must contain one graph and valid poses")
    if any(row is None for row in audit.selected_rows):
        raise ValueError("no standard trajectory sample exists for a query")
    residuals = np.array(
        [int(row.timestamp_us) - int(query) for row, query in zip(audit.selected_rows, times)],
        dtype=object,
    )
    accepted = np.abs(residuals) <= args.max_gap_us
    accepted = np.asarray(accepted, dtype=bool)
    selected = [row for row, keep in zip(audit.selected_rows, accepted) if keep]
    if len(selected) < 20:
        raise ValueError("fewer than 20 timestamp matches pass the explicit maximum gap")
    source_positions = np.array(
        [[float(row.values[f"t{axis}_world_device"]) for axis in "xyz"] for row in selected]
    )
    source_quaternions = np.array(
        [[float(row.values[f"q{axis}_world_device"]) for axis in "xyzw"] for row in selected]
    )
    source_graph = next(iter(audit.graph_uids))
    pairs = PosePairs(
        source_positions,
        destination_positions[accepted],
        quaternion_xyzw_batch_to_rotations(source_quaternions),
        quaternion_xyzw_batch_to_rotations(destination_quaternions[accepted]),
        FrameId(f"leader/mps_world/{source_graph}"),
        FrameId(f"comind/{recording_id}/multislam/{destination_graph}"),
        FrameId("leader/device"),
        DistanceUnit.METERS,
        Provenance(
            "comind same-leader DEVICE_TIME nearest matching",
            f"maximum absolute gap {args.max_gap_us}us; earlier timestamp then first duplicate ties",
            parents=(Provenance(str(standard_path)), Provenance(str(cache_path))),
        ),
    )
    print(f"Fitting and holding out {len(selected)} accepted paired poses", file=sys.stderr)
    report = study(
        pairs,
        times[accepted],
        length=args.orientation_length_scale_m,
        huber=args.huber_delta_m,
    )
    report.update(
        {
            "recording_id": recording_id,
            "source_standard_trajectory": str(standard_path),
            "source_multislam_cache": str(cache_path),
            "derived_paired_cache": str(paired_path.resolve()),
            "read_scope": "One standard leader CSV pass; existing leader NPZ; no scan/VRS/MultiSLAM CSV",
            "raw_size_mtime_unchanged": True,
            "standard_trajectory_audit": audit.stats,
            "official_pose_format_evidence": EVIDENCE,
            "matching": {
                "clock": device_clock(recording_id, "leader").name,
                "query_count": len(times),
                "accepted_count": int(accepted.sum()),
                "rejected_count": int((~accepted).sum()),
                "maximum_gap_us": args.max_gap_us,
                "accepted_absolute_residual_us": statistics(np.abs(residuals[accepted])),
                "all_absolute_residual_us": statistics(np.abs(residuals.astype(object))),
                "exact_matches": int(np.count_nonzero(residuals == 0)),
                "accepted_multislam_timestamp_us_range": [
                    int(times[accepted][0]),
                    int(times[accepted][-1]),
                ],
                "accepted_standard_timestamp_us_range": [
                    selected[0].timestamp_us,
                    selected[-1].timestamp_us,
                ],
            },
        }
    )
    paired_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        paired_path,
        source_positions_m=pairs.source_positions,
        destination_positions_m=pairs.destination_positions,
        source_rotations=pairs.source_rotations,
        destination_rotations=pairs.destination_rotations,
        source_device_us=np.array([row.timestamp_us for row in selected], dtype=np.int64),
        destination_device_us=times[accepted],
        source_indices=np.array([row.source_index for row in selected], dtype=np.int64),
        destination_cache_indices=np.flatnonzero(accepted),
        source_graph_uid=source_graph,
        destination_graph_uid=destination_graph,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
