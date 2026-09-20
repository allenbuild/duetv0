"""Bounded-memory real-recording audit and verified shared-world smoke validation."""

from __future__ import annotations

import csv
import io
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from functools import partial
from itertools import islice
from pathlib import Path
from typing import TextIO

import numpy as np
import pandas as pd

from duet.adapters.comind.archive import recover_complete_zip_member
from duet.adapters.comind.layout import ROLES, ParticipantLayout, discover_recording
from duet.adapters.comind.mps import (
    MPS_TRANSFORM_TOLERANCE,
    CameraCalibrationSample,
    MpsHandFrame,
    MpsTrajectorySample,
    iter_trajectory,
    load_calibrations,
    load_hands,
)
from duet.adapters.comind.multislam import (
    TRAJECTORY_MEMBER,
    SlamSource,
    verify_shared_graph_uids,
)
from duet.adapters.comind.scan import inspect_scan
from duet.adapters.comind.shared_world import camera_to_shared_world, hand_to_shared_world
from duet.adapters.comind.video import EgoVideo
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations
from duet.schemas.common import FrameId, Provenance
from duet.synchronization.matching import nonnegative_seconds

_TRANSLATION_FIELDS = [f"t{axis}_world_device" for axis in "xyz"]
_QUATERNION_FIELDS = [f"q{axis}_world_device" for axis in "xyzw"]
_REQUIRED_FIELDS = {
    "graph_uid",
    "tracking_timestamp_us",
    "quality_score",
    *_TRANSLATION_FIELDS,
    *_QUATERNION_FIELDS,
}


@dataclass(frozen=True)
class SelectedTrajectoryRow:
    """One untouched source row and its original zero-based data-row index."""

    source_index: int
    timestamp_us: int
    values: dict[str, str]


@dataclass(frozen=True)
class TrajectoryAudit:
    """Full-stream statistics plus nearest selected rows in original query order."""

    stats: dict[str, object]
    graph_uids: frozenset[str]
    selected_rows: tuple[SelectedTrajectoryRow | None, ...]


def _integer_column(column: pd.Series, name: str) -> np.ndarray:
    """Parse signed 64-bit timestamp integers directly, with no float intermediary."""
    if not column.str.fullmatch(r"[+-]?[0-9]+").all():
        raise ValueError(f"{name} requires nonempty integer tokens")
    try:
        return column.to_numpy().astype(np.int64)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{name} exceeds signed 64-bit integer representation") from exc


def _update_range(
    stats: dict[str, object], name: str, values: np.ndarray, *, missing_sentinel: int | None = None
) -> None:
    present = values if missing_sentinel is None else values[values != missing_sentinel]
    stats[f"{name}_missing_count"] += len(values) - len(present)
    if present.size:
        existing = stats[f"{name}_range"]
        low, high = int(present.min()), int(present.max())
        stats[f"{name}_range"] = (
            [low, high] if existing is None else [min(existing[0], low), max(existing[1], high)]
        )


def _selected_row(
    position: int,
    *,
    chunk: pd.DataFrame,
    times: np.ndarray,
    row_offset: int,
    cache: dict[int, SelectedTrajectoryRow],
) -> SelectedTrajectoryRow:
    if position not in cache:
        cache[position] = SelectedTrajectoryRow(
            row_offset + position, int(times[position]), chunk.iloc[position].to_dict()
        )
    return cache[position]


def _strict_csv_chunks(
    reader: Iterator[list[str]], *, header: list[str], chunk_size: int, source: str
) -> Iterator[pd.DataFrame]:
    """Validate row widths before DataFrame construction can infer or pad fields."""
    offset = 0
    while rows := list(islice(reader, chunk_size)):
        for index, row in enumerate(rows):
            if len(row) != len(header):
                raise ValueError(
                    f"{source}: trajectory data row {offset + index} has {len(row)} fields; "
                    f"expected {len(header)}"
                )
        yield pd.DataFrame(rows, columns=header, dtype=str)
        offset += len(rows)


def audit_trajectory(
    stream: TextIO,
    *,
    provenance: Provenance,
    query_timestamps_us: Sequence[int] = (),
    chunk_size: int = 50_000,
) -> TrajectoryAudit:
    """Audit all trajectory rows and select nearest samples in one chunked pass.

    Query values are participant-device microseconds; the caller must never mix
    participant clocks. Source device times must be nondecreasing; only UTC uses
    the verified -1 missing sentinel. Ties prefer the earlier time, then first original duplicate, even
    across chunk boundaries. No gap acceptance or frame binding occurs here.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in query_timestamps_us):
        raise TypeError("queries must contain original integer microseconds")
    try:
        queries = np.asarray(query_timestamps_us, dtype=np.int64)
    except OverflowError as exc:
        raise ValueError("query timestamp exceeds signed 64-bit representation") from exc
    order = np.argsort(queries, kind="stable")
    ordered_queries = queries[order]
    selected: list[SelectedTrajectoryRow | None] = [None] * len(queries)
    reader = csv.reader(stream, strict=True)
    header = next(reader, [])
    if not header or len(header) != len(set(header)) or not _REQUIRED_FIELDS <= set(header):
        raise ValueError(f"{provenance.source}: missing/duplicate trajectory CSV fields")
    stats: dict[str, object] = {
        "source": provenance.source,
        "row_count": 0,
        "tracking_timestamp_us_range": None,
        "tracking_timestamp_us_missing_count": 0,
        "utc_timestamp_ns_range": None,
        "utc_timestamp_ns_missing_count": 0,
        "utc_field_present": "utc_timestamp_ns" in header,
        "duplicate_timestamp_count": 0,
        "transform_valid_count": 0,
        "transform_invalid_count": 0,
        "invalid_source_indices_first_10": [],
        "max_quaternion_norm_error": 0.0,
        "max_rotation_orthonormality_error": 0.0,
        "max_rotation_determinant_error": 0.0,
        "quality_score_range": None,
        "first_device_position_m": None,
        "last_device_position_m": None,
        "transform_absolute_tolerance": MPS_TRANSFORM_TOLERANCE,
    }
    graphs: set[str] = set()
    previous: SelectedTrajectoryRow | None = None
    cursor = 0
    row_offset = 0
    for chunk in _strict_csv_chunks(
        reader, header=header, chunk_size=chunk_size, source=provenance.source
    ):
        if chunk.empty:
            continue
        try:
            times = _integer_column(chunk["tracking_timestamp_us"], "tracking_timestamp_us")
            _update_range(stats, "tracking_timestamp_us", times)
            if "utc_timestamp_ns" in header:
                _update_range(
                    stats,
                    "utc_timestamp_ns",
                    _integer_column(
                        chunk["utc_timestamp_ns"].replace("", "-1"), "utc_timestamp_ns"
                    ),
                    missing_sentinel=-1,
                )
            identifiers = chunk["graph_uid"]
            if not identifiers.str.strip().ne("").all():
                raise ValueError("graph_uid cannot be empty")
            graphs.update(identifiers.unique())
            translation = chunk[_TRANSLATION_FIELDS].to_numpy(dtype=np.float64)
            quaternions = chunk[_QUATERNION_FIELDS].to_numpy(dtype=np.float64)
            quality = chunk["quality_score"].to_numpy(dtype=np.float64)
            if not np.isfinite(quality).all():
                raise ValueError("quality_score must be finite")
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(
                f"{provenance.source}: chunk starting data row {row_offset}: {exc}"
            ) from exc
        quality_range = [float(quality.min()), float(quality.max())]
        if stats["quality_score_range"] is not None:
            quality_range = [
                min(stats["quality_score_range"][0], quality_range[0]),
                max(stats["quality_score_range"][1], quality_range[1]),
            ]
        stats["quality_score_range"] = quality_range
        with np.errstate(over="ignore", invalid="ignore"):
            norms = np.linalg.norm(quaternions, axis=1)
            norm_error = np.abs(norms - 1)
        finite_norm_error = norm_error[np.isfinite(norm_error)]
        if finite_norm_error.size:
            stats["max_quaternion_norm_error"] = max(
                stats["max_quaternion_norm_error"], float(finite_norm_error.max())
            )
        quaternion_valid = np.isfinite(quaternions).all(axis=1) & (
            norm_error <= MPS_TRANSFORM_TOLERANCE
        )
        valid = np.isfinite(translation).all(axis=1) & quaternion_valid
        indices = np.flatnonzero(quaternion_valid)
        if indices.size:
            rotations = quaternion_xyzw_batch_to_rotations(
                quaternions[indices], norm_tolerance=MPS_TRANSFORM_TOLERANCE
            )
            errors = np.max(
                np.abs(np.swapaxes(rotations, 1, 2) @ rotations - np.eye(3)), axis=(1, 2)
            )
            determinant_errors = np.abs(np.linalg.det(rotations) - 1)
            valid[indices] &= (errors <= MPS_TRANSFORM_TOLERANCE) & (
                determinant_errors <= MPS_TRANSFORM_TOLERANCE
            )
            stats["max_rotation_orthonormality_error"] = max(
                stats["max_rotation_orthonormality_error"], float(errors.max())
            )
            stats["max_rotation_determinant_error"] = max(
                stats["max_rotation_determinant_error"], float(determinant_errors.max())
            )
        stats["transform_valid_count"] += int(valid.sum())
        stats["transform_invalid_count"] += int((~valid).sum())
        invalid_indices = stats["invalid_source_indices_first_10"]
        invalid_indices.extend(
            (np.flatnonzero(~valid)[: 10 - len(invalid_indices)] + row_offset).tolist()
        )
        if row_offset == 0 and np.isfinite(translation[0]).all():
            stats["first_device_position_m"] = translation[0].tolist()
        stats["last_device_position_m"] = (
            translation[-1].tolist() if np.isfinite(translation[-1]).all() else None
        )
        present_positions = np.arange(len(times))
        present_times = times
        source_row = partial(
            _selected_row, chunk=chunk, times=times, row_offset=row_offset, cache={}
        )

        if present_times.size:
            if np.any(present_times[1:] < present_times[:-1]) or (
                previous is not None and int(present_times[0]) < previous.timestamp_us
            ):
                raise ValueError(f"{provenance.source}: trajectory timestamps decrease")
            stats["duplicate_timestamp_count"] += int(
                np.count_nonzero(present_times[1:] == present_times[:-1])
            ) + int(previous is not None and int(present_times[0]) == previous.timestamp_us)
            stop = int(np.searchsorted(ordered_queries, present_times[-1], side="right"))
            while cursor < stop:
                query = int(ordered_queries[cursor])
                right = int(np.searchsorted(present_times, query, side="left"))
                candidates: list[SelectedTrajectoryRow] = []
                if right < len(present_times):
                    candidates.append(source_row(int(present_positions[right])))
                if right > 0:
                    left = int(
                        np.searchsorted(present_times, present_times[right - 1], side="left")
                    )
                    candidates.append(source_row(int(present_positions[left])))
                if previous is not None:
                    candidates.append(previous)
                selected[int(order[cursor])] = min(
                    candidates,
                    key=lambda row: (
                        abs(row.timestamp_us - query),
                        row.timestamp_us,
                        row.source_index,
                    ),
                )
                cursor += 1
            last_time = int(present_times[-1])
            if previous is None or previous.timestamp_us != last_time:
                first_last = int(np.searchsorted(present_times, last_time, side="left"))
                previous = source_row(int(present_positions[first_last]))
        row_offset += len(chunk)
    while cursor < len(order):
        selected[int(order[cursor])] = previous
        cursor += 1
    stats["row_count"] = row_offset
    stats["graph_uids"] = sorted(graphs)
    stats["selected_unique_row_count"] = len(
        {row.source_index for row in selected if row is not None}
    )
    return TrajectoryAudit(stats, frozenset(graphs), tuple(selected))


def parse_selected_poses(
    audit: TrajectoryAudit,
    *,
    participant: str,
    recording_id: str,
    world_frame: FrameId,
    provenance: Provenance,
) -> tuple[MpsTrajectorySample | None, ...]:
    """Parse unique selected raw rows after full-stream shared-graph verification.

    Original row indices are carried in selection provenance. The tiny one-row
    CSV is an explicit parsing view, not an alternative original file location.
    """
    parsed: dict[int, MpsTrajectorySample] = {}
    result: list[MpsTrajectorySample | None] = []
    for row in audit.selected_rows:
        if row is None:
            result.append(None)
            continue
        if row.source_index not in parsed:
            buffer = io.StringIO()
            writer = csv.DictWriter(buffer, fieldnames=list(row.values))
            writer.writeheader()
            writer.writerow(row.values)
            buffer.seek(0)
            selection = Provenance(
                f"selection:{provenance.source}#csv-row={row.source_index + 2}",
                f"Untouched original data row {row.source_index}; one-row parsing view",
                (provenance,),
            )
            parsed[row.source_index] = next(
                iter_trajectory(
                    buffer,
                    participant=participant,
                    recording_id=recording_id,
                    world_frame=world_frame,
                    provenance=selection,
                )
            )
        result.append(parsed[row.source_index])
    return tuple(result)


def ensure_output_outside_raw(path: Path, dataset_root: Path) -> None:
    """Reject outputs under either the supplied raw dataset or this repository's raw tree."""
    destination = path.resolve()
    repository_raw = Path(__file__).resolve().parents[4] / "data" / "raw"
    roots = [dataset_root.resolve(), repository_raw.resolve()]
    if dataset_root.parent.name == "raw":
        roots.append(dataset_root.parent.resolve())
    if any(destination.is_relative_to(root) for root in roots):
        raise ValueError("validation outputs must stay outside immutable raw data")


def raw_inventory(root: Path) -> dict[str, tuple[int, int]]:
    """Record sizes and modification times only; do not read or hash large payloads."""
    return {
        str(path.relative_to(root)): (stat.st_size, stat.st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
        for stat in [path.stat()]
    }


def _slam_source(
    participant: ParticipantLayout,
    *,
    dataset_root: Path,
    recording_id: str,
    processed_root: Path,
    recover_incomplete_zip: bool,
) -> tuple[SlamSource, dict[str, object]]:
    source = SlamSource(participant.multislam_dir, participant.multislam_zip)
    if participant.multislam_dir is not None:
        return source, {"storage": "directory", "source": source.description}
    if participant.multislam_zip is None:
        raise FileNotFoundError("Multi-SLAM source is missing")
    try:
        with zipfile.ZipFile(participant.multislam_zip) as archive:
            archive.getinfo(TRAJECTORY_MEMBER)
        return source, {"storage": "zip", "source": source.description}
    except zipfile.BadZipFile as exc:
        if not recover_incomplete_zip:
            raise ValueError(
                f"{participant.multislam_zip} is not a complete ZIP; explicitly enable "
                "--recover-incomplete-zip to recover only a complete CRC-verified trajectory member"
            ) from exc
    output_directory = processed_root / recording_id / participant.multislam_index / "slam"
    ensure_output_outside_raw(output_directory, dataset_root)
    declared_raw_root = dataset_root.parent if dataset_root.parent.name == "raw" else dataset_root
    recovered = recover_complete_zip_member(
        participant.multislam_zip,
        TRAJECTORY_MEMBER,
        output_directory=output_directory,
        raw_root=declared_raw_root,
    )
    source = SlamSource(directory=output_directory, recovery_provenance=recovered.provenance)
    return source, {
        "storage": "explicit_recovered_member",
        "archive_status": "incomplete; only the trajectory member is recovered and verified",
        "source": source.description,
        "original_archive": str(recovered.archive),
        "compressed_bytes": recovered.compressed_bytes,
        "uncompressed_bytes": recovered.uncompressed_bytes,
        "crc32": f"{recovered.crc32:08x}",
    }


def _hand_summary(frames: list[MpsHandFrame]) -> dict[str, object]:
    times = [frame.timestamp.raw_value for frame in frames]
    result: dict[str, object] = {
        "row_count": len(frames),
        "tracking_timestamp_us_range": [min(times), max(times)] if times else None,
    }
    for side in ("left", "right"):
        hands = [getattr(frame, side) for frame in frames]
        present = sum(hand.points is not None for hand in hands)
        result[side] = {
            "present_count": present,
            "missing_count": len(hands) - present,
            "coverage_fraction": present / len(hands) if hands else None,
            "zero_confidence_count": sum(
                getattr(frame, f"{side}_confidence_raw") == 0 for frame in frames
            ),
            "missing_sentinel_count": sum(
                getattr(frame, f"{side}_confidence_raw") == -1 for frame in frames
            ),
        }
    return result


def _calibration_summary(
    path: Path,
    *,
    participant: str,
    recording_id: str,
) -> tuple[dict[str, object], tuple[CameraCalibrationSample, ...]]:
    labels: set[str] = set()
    first = last = None
    count = transform_count = missing_utc = 0
    time_low = time_high = None
    for frame in load_calibrations(path, participant=participant, recording_id=recording_id):
        labels.update(camera.label for camera in frame.cameras)
        count += 1
        transform_count += len(frame.cameras)
        missing_utc += int(frame.utc_timestamp_ns_raw == -1)
        time = frame.timestamp.raw_value
        time_low = time if time_low is None else min(time_low, time)
        time_high = time if time_high is None else max(time_high, time)
        if first is None:
            first = frame
        last = frame
    samples = () if first is None else (first,) if first is last else (first, last)
    return {
        "row_count": count,
        "labels": sorted(labels),
        "tracking_timestamp_us_range": None if time_low is None else [time_low, time_high],
        "utc_missing_sentinel_count": missing_utc,
        "transform_valid_count": transform_count,
        "transform_invalid_count": 0,
    }, samples


def _residual_summary(residuals: list[float], *, matched: int, rejected: int) -> dict[str, object]:
    absolute = np.abs(residuals)
    return {
        "candidate_count": len(residuals),
        "accepted_count": matched,
        "gap_rejected_count": rejected,
        "signed_min_seconds": min(residuals, default=None),
        "signed_max_seconds": max(residuals, default=None),
        "median_absolute_seconds": float(np.median(absolute)) if absolute.size else None,
        "p95_absolute_seconds": float(np.quantile(absolute, 0.95)) if absolute.size else None,
        "max_absolute_seconds": float(absolute.max()) if absolute.size else None,
    }


def run_validation(
    dataset_root: str | Path,
    recording_id: str,
    *,
    max_hand_gap_seconds: float,
    recover_incomplete_zip: bool = False,
    processed_root: Path = Path("data/processed/comind"),
    progress: Callable[[str], None] | None = None,
) -> dict[str, object]:
    """Audit all streams once and exercise same-device hands/camera transform chains.

    Only explicitly requested member recovery writes files, exclusively outside
    raw. This function does not claim video/device, cross-device or scan alignment.
    """
    maximum_gap = nonnegative_seconds(max_hand_gap_seconds, "max_hand_gap_seconds")
    emit = progress or (lambda _message: None)
    layout = discover_recording(dataset_root, recording_id)
    ensure_output_outside_raw(processed_root, layout.dataset_root)
    report: dict[str, object] = {
        "recording_id": recording_id,
        "dataset_root": str(layout.dataset_root),
        "maximum_hand_pose_gap_seconds": float(maximum_gap),
        "participants": {},
    }
    all_hands: dict[str, list[MpsHandFrame]] = {}
    calibrations: dict[str, tuple[CameraCalibrationSample, ...]] = {}
    sources: dict[str, SlamSource] = {}
    audits: dict[str, TrajectoryAudit] = {}
    for role in ROLES:
        paths = layout.participants[role]
        emit(f"{role}: reading ego-video headers, hand rows and calibration rows")
        video = EgoVideo(
            paths.video_path, participant_id=paths.participant_id, recording_uuid=recording_id
        ).metadata()
        all_hands[role] = list(
            load_hands(paths.hands_path, participant=role, recording_id=recording_id)
        )
        calibration_stats, calibrations[role] = _calibration_summary(
            paths.calibration_path,
            participant=role,
            recording_id=recording_id,
        )
        participant_report = {
            "video": {
                "path": str(video.path),
                "codec": video.codec,
                "width": video.width,
                "height": video.height,
                "average_rate": str(video.average_rate),
                "time_base": str(video.time_base),
                "start_pts": video.start_pts,
                "duration_pts": video.duration_pts,
                "duration_seconds": None
                if video.duration_seconds is None
                else float(video.duration_seconds),
                "duration_seconds_exact": None
                if video.duration_seconds is None
                else str(video.duration_seconds),
                "frame_count": video.frame_count,
                "container_duration_us": video.container_duration_us,
                "clock_domain": video.clock_domain.name,
            },
            "hands": _hand_summary(all_hands[role]),
            "calibration": calibration_stats,
        }
        report["participants"][role] = participant_report
        emit(f"{role}: auditing every standard MPS trajectory row")
        with paths.trajectory_path.open(encoding="utf-8", newline="") as stream:
            participant_report["standard_mps"] = audit_trajectory(
                stream, provenance=Provenance(str(paths.trajectory_path))
            ).stats
        sources[role], participant_report["multislam_storage"] = _slam_source(
            paths,
            dataset_root=layout.dataset_root,
            recording_id=recording_id,
            processed_root=processed_root,
            recover_incomplete_zip=recover_incomplete_zip,
        )
        query_times = [frame.timestamp.raw_value for frame in all_hands[role]]
        query_times.extend(frame.timestamp.raw_value for frame in calibrations[role])
        emit(f"{role}: auditing every Multi-SLAM row and selecting nearest same-device poses")
        with sources[role].open_trajectory() as stream:
            audits[role] = audit_trajectory(
                stream,
                provenance=sources[role].provenance,
                query_timestamps_us=query_times,
            )
        participant_report["multislam"] = audits[role].stats
    verification = verify_shared_graph_uids(
        {role: audits[role].graph_uids for role in ROLES},
        sources=sources,
        recording_id=recording_id,
    )
    report["shared_world"] = {
        "verified": True,
        "graph_uid": verification.graph_uid,
        "frame": verification.world_frame.name,
        "time_synchronization_verified": False,
    }
    for role in ROLES:
        emit(f"{role}: validating selected shared-world hand and RGB camera transform chains")
        poses = parse_selected_poses(
            audits[role],
            participant=role,
            recording_id=recording_id,
            world_frame=verification.world_frame,
            provenance=sources[role].provenance,
        )
        residuals: list[float] = []
        matched = rejected = transformed_count = 0
        for index, frame in enumerate(all_hands[role]):
            for side in ("left", "right"):
                transformed, match = hand_to_shared_world(
                    getattr(frame, side),
                    poses[index],
                    verification=verification,
                    max_gap_seconds=float(maximum_gap),
                )
                transformed_count += int(transformed.points is not None)
                if side == "left":
                    matched += int(match.accepted)
                    rejected += int(not match.accepted)
                    if match.signed_residual_seconds is not None:
                        residuals.append(float(match.signed_residual_seconds))
        report["participants"][role]["shared_hands"] = {
            "transformed_present_hand_count": transformed_count,
            "residuals_per_hand_timestamp": _residual_summary(
                residuals, matched=matched, rejected=rejected
            ),
        }
        labels = report["participants"][role]["calibration"]["labels"]
        if "camera-rgb" not in labels:
            raise ValueError(
                f"{role}: inspected calibration does not contain verified RGB label camera-rgb"
            )
        camera_report = []
        for index, calibration in enumerate(calibrations[role]):
            camera, match = camera_to_shared_world(
                calibration,
                poses[len(all_hands[role]) + index],
                camera_label="camera-rgb",
                participant=role,
                verification=verification,
                max_gap_seconds=float(maximum_gap),
            )
            camera_report.append(
                {
                    "tracking_timestamp_us": calibration.timestamp.raw_value,
                    "state": camera.metadata.state.value,
                    "same_device_residual_seconds": None
                    if match.signed_residual_seconds is None
                    else float(match.signed_residual_seconds),
                    "position_shared_world_m": None
                    if camera.transform is None
                    else camera.transform.matrix[:3, 3].tolist(),
                }
            )
        report["participants"][role]["rgb_camera_first_last"] = camera_report
    emit("Reading scan matrix and PLY headers without binding scan frames")
    scan = inspect_scan(layout.scan_transform_path, layout.scan_ply_paths)
    report["scan"] = {
        "transform_validity": scan.registration.validation.status.value,
        "transform_metrics": dict(scan.registration.validation.metrics),
        "registration_semantics_verified": False,
        "distance_unit": None,
        "clouds": [
            {
                "path": str(cloud.path),
                "encoding": cloud.header.encoding,
                "vertex_count": cloud.header.vertex_count,
                "comments": list(cloud.header.comments),
            }
            for cloud in scan.clouds
        ],
    }
    report["unresolved"] = [
        "MP4 PTS to participant device tracking time mapping",
        "Helper-to-leader cross-device clock synchronization",
        "Scan registration direction, units and exact Multi-SLAM graph identity",
        "MP4 display orientation relative to physical RGB camera projection",
        "Handover annotation clock and left/right physical participant mapping",
    ]
    return report
