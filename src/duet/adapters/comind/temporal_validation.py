"""Same-participant frame/pose/hand validation and paired-frame handover ranking.

Only VERIFIED visual correspondences supply queries. Exact integer nanoseconds
are retained; no UTC comparison, FPS-derived device time, interpolation, or
cross-participant device-time subtraction is performed here.
"""

from __future__ import annotations

import csv
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from duet.adapters.comind.annotation_timing import audit_annotation_frame_times
from duet.adapters.comind.mps import HAND_LANDMARK_NAMES, MPS_TRANSFORM_TOLERANCE, device_clock
from duet.adapters.comind.semantics import MappingEvidence
from duet.adapters.comind.validation import _integer_column, _strict_csv_chunks
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations
from duet.schemas.episode import HandoverAnnotation

ROLES = ("helper", "leader")
NO_RESIDUAL = np.iinfo(np.int64).min
WRIST_INDEX = HAND_LANDMARK_NAMES.index("wrist_joint")


def _integers(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1 or array.dtype.kind != "i" or array.dtype.itemsize > 8:
        raise ValueError(f"{name} must be a one-dimensional signed-integer array")
    return array.astype(np.int64, copy=False)


def _times(value: np.ndarray, name: str, *, microseconds: bool = False) -> np.ndarray:
    result = _integers(value, name)
    if np.any(result < 0) or np.any(result[1:] < result[:-1]):
        raise ValueError(f"{name} must be nonnegative and nondecreasing")
    if microseconds and np.any(result > np.iinfo(np.int64).max // 1000):
        raise ValueError(f"{name} would overflow exact nanoseconds")
    return result * 1000 if microseconds else result


def statistics(values: np.ndarray) -> dict[str, float | int | None]:
    """Finite-value distribution; an empty distribution has explicit null values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        **{
            key: float(np.percentile(values, percentile)) if len(values) else None
            for key, percentile in (("min", 0), ("median", 50), ("p95", 95), ("max", 100))
        },
    }


@dataclass(frozen=True)
class TemporalThresholds:
    pose_max_gap_ns: int = 2_000_000
    hand_max_gap_ns: int = 20_000_000
    hand_pose_max_gap_ns: int = 10_000_000
    minimum_hand_confidence: float = 0.5
    maximum_wrist_speed_m_s: float = 5.0
    continuity_max_gap_ns: int = 100_000_000

    def __post_init__(self) -> None:
        for name in (
            "pose_max_gap_ns",
            "hand_max_gap_ns",
            "hand_pose_max_gap_ns",
            "continuity_max_gap_ns",
        ):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be an explicit nonnegative integer")
        if not np.isfinite(self.minimum_hand_confidence) or not (
            0 <= self.minimum_hand_confidence <= 1
        ):
            raise ValueError("minimum_hand_confidence must be in [0,1]")
        if not np.isfinite(self.maximum_wrist_speed_m_s) or self.maximum_wrist_speed_m_s <= 0:
            raise ValueError("maximum_wrist_speed_m_s must be finite and positive")


DEFAULT_THRESHOLDS = TemporalThresholds()


@dataclass(frozen=True)
class ParticipantStreams:
    """Compact full trajectory and cached hands in one verified shared meter frame.

    Hand world coordinates are evaluated at each hand's own source timestamp,
    using the previously checked full-rate pose chain. ``hand_pose_residual_us``
    retains that secondary alignment separately from video-to-hand residuals.
    """

    recording_id: str
    participant: str
    graph_uid: str
    device_clock_name: str
    pose_device_us: np.ndarray
    pose_translation_m: np.ndarray
    pose_quaternion_xyzw: np.ndarray
    hand_device_us: np.ndarray
    hand_points_shared_m: np.ndarray
    hand_confidence: np.ndarray
    hand_pose_residual_us: np.ndarray
    hand_pose_accepted: np.ndarray
    hand_points_device_m: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.participant not in ROLES or not self.graph_uid:
            raise ValueError("streams require an identified participant and shared graph")
        if self.device_clock_name != device_clock(self.recording_id, self.participant).name:
            raise ValueError("streams must retain that participant's own device clock")
        pose = _times(self.pose_device_us, "pose_device_us", microseconds=True)
        hand = _times(self.hand_device_us, "hand_device_us", microseconds=True)
        for name, shape in (
            ("pose_translation_m", (len(pose), 3)),
            ("pose_quaternion_xyzw", (len(pose), 4)),
            ("hand_points_shared_m", (len(hand), 2, 21, 3)),
            ("hand_confidence", (len(hand), 2)),
            ("hand_pose_residual_us", (len(hand),)),
            ("hand_pose_accepted", (len(hand),)),
        ):
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not np.isfinite(self.pose_translation_m).all():
            raise ValueError("pose translations must be finite meters")
        if len(pose):
            quaternion_xyzw_batch_to_rotations(
                self.pose_quaternion_xyzw, norm_tolerance=MPS_TRANSFORM_TOLERANCE
            )
        if np.asarray(self.hand_pose_accepted).dtype.kind != "b":
            raise ValueError("hand_pose_accepted must be Boolean")
        residual = _integers(self.hand_pose_residual_us, "hand_pose_residual_us")
        if np.any(np.abs(residual.astype(object)) > np.iinfo(np.int64).max // 1000):
            raise ValueError("hand pose residual would overflow nanoseconds")
        confidence = np.asarray(self.hand_confidence)
        if np.isinf(confidence).any() or np.any(
            np.isfinite(confidence) & (confidence != -1) & ((confidence < 0) | (confidence > 1))
        ):
            raise ValueError("invalid hand confidence")
        points = np.asarray(self.hand_points_shared_m)
        present = np.isfinite(points).all(axis=(2, 3))
        absent = np.isnan(points).all(axis=(2, 3))
        if not (present | absent).all() or np.any(present & (confidence == -1)):
            raise ValueError("hands must have complete finite geometry or be entirely missing")
        if self.hand_points_device_m is not None:
            points_device = np.asarray(self.hand_points_device_m)
            if points_device.shape != points.shape:
                raise ValueError("device hand points must match shared hand point dimensions")
            device_present = np.isfinite(points_device).all(axis=(2, 3))
            device_absent = np.isnan(points_device).all(axis=(2, 3))
            if not (device_present | device_absent).all() or np.any(
                device_present & (confidence == -1)
            ):
                raise ValueError(
                    "source hands must be complete finite geometry or entirely missing"
                )

    @property
    def world_frame(self) -> str:
        return f"comind/{self.recording_id}/multislam/{self.graph_uid}"


def read_full_trajectory_once(path: Path, *, expected_graph_uid: str) -> dict[str, np.ndarray]:
    """One bounded CSV pass, retaining only exact device time and valid rigid poses.

    Source bytes remain untouched. This does not use the downsampled display
    trajectory for nearest-sample residuals. Callers may cache the returned
    numeric arrays outside raw and reuse them across mappings.
    """
    before = path.stat()
    columns = [f"t{axis}_world_device" for axis in "xyz"]
    quaternions = [f"q{axis}_world_device" for axis in "xyzw"]
    required = {"tracking_timestamp_us", "graph_uid", *columns, *quaternions}
    output: dict[str, list[np.ndarray]] = {
        "device_us": [],
        "translation_m": [],
        "quaternion_xyzw": [],
    }
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(set(header)) != len(header) or not required <= set(header):
            raise ValueError("invalid full trajectory header")
        for chunk in _strict_csv_chunks(reader, header=header, chunk_size=50_000, source=str(path)):
            if not (chunk["graph_uid"] == expected_graph_uid).all():
                raise ValueError("trajectory is not the verified shared world graph")
            times = _integer_column(chunk["tracking_timestamp_us"], "tracking_timestamp_us")
            translation = chunk[columns].to_numpy(dtype=float)
            quaternion = chunk[quaternions].to_numpy(dtype=float)
            if not np.isfinite(translation).all():
                raise ValueError("nonfinite trajectory translation")
            quaternion_xyzw_batch_to_rotations(quaternion, norm_tolerance=MPS_TRANSFORM_TOLERANCE)
            output["device_us"].append(times)
            output["translation_m"].append(translation)
            output["quaternion_xyzw"].append(quaternion)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("trajectory source changed during read")
    if not output["device_us"]:
        raise ValueError("full trajectory is empty")
    result = {name: np.concatenate(parts) for name, parts in output.items()}
    _times(result["device_us"], "pose_device_us", microseconds=True)
    return result


def nearest_local_ns(
    source_ns: np.ndarray, queries_ns: np.ndarray, query_valid: np.ndarray, *, maximum_gap_ns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact nearest timestamps, earlier time then first source duplicate on ties.

    Returns candidate source indices, signed source-minus-query ns residuals,
    and accepted masks. Rejected gaps retain the candidate residual. Missing
    queries/empty streams have index -1 and residual ``NO_RESIDUAL``.
    """
    source = _times(source_ns, "source_ns")
    queries = _integers(queries_ns, "queries_ns")
    valid = np.asarray(query_valid)
    if valid.dtype.kind != "b" or valid.shape != queries.shape:
        raise ValueError("query_valid must be a matching Boolean mask")
    if type(maximum_gap_ns) is not int or maximum_gap_ns < 0 or np.any(queries[valid] < 0):
        raise ValueError("query times and maximum gap must be nonnegative")
    indices = np.full(len(queries), -1, dtype=np.int64)
    residuals = np.full(len(queries), NO_RESIDUAL, dtype=np.int64)
    accepted = np.zeros(len(queries), dtype=bool)
    if not len(source) or not valid.any():
        return indices, residuals, accepted
    query = queries[valid]
    right = np.minimum(np.searchsorted(source, query, side="left"), len(source) - 1)
    left = np.maximum(right - 1, 0)
    selected = np.where(np.abs(source[left] - query) <= np.abs(source[right] - query), left, right)
    selected = np.searchsorted(source, source[selected], side="left")
    residual = source[selected] - query
    indices[valid], residuals[valid] = selected, residual
    accepted[valid] = np.abs(residual) <= maximum_gap_ns
    return indices, residuals, accepted


@dataclass(frozen=True)
class FrameValidation:
    recording_id: str
    participant: str
    world_frame: str
    arrays: Mapping[str, np.ndarray]
    summary: Mapping[str, object]


def _match_summary(indices: np.ndarray, residual: np.ndarray, accepted: np.ndarray) -> dict:
    candidates = indices >= 0
    return {
        "frame_count": len(indices),
        "accepted_count": int(accepted.sum()),
        "coverage": float(accepted.mean()) if len(indices) else 0.0,
        "missing_query_or_empty_stream_count": int((~candidates).sum()),
        "gap_rejected_count": int((candidates & ~accepted).sum()),
        "all_candidates_absolute_residual_ms": statistics(np.abs(residual[candidates]) / 1e6),
        "accepted_absolute_residual_ms": statistics(np.abs(residual[accepted]) / 1e6),
        "residual_sign": "source timestamp minus mapped video DEVICE_TIME",
    }


def validate_participant_frames(
    frame_map: Mapping[str, np.ndarray],
    streams: ParticipantStreams,
    *,
    recording_id: str,
    participant: str,
    thresholds: TemporalThresholds = DEFAULT_THRESHOLDS,
) -> FrameValidation:
    """Assemble one video using only its own device clock and VERIFIED map rows.

    The supplied map must come from the validated VideoVrsFrameMap loader; this
    function also checks row shapes, classification, monotonicity and gates.
    No value is upgraded from INFERRED, even when its numeric time looks valid.
    """
    if (recording_id, participant) != (streams.recording_id, streams.participant):
        raise ValueError("frame map and streams must identify the same participant and recording")
    frame = _integers(frame_map["mp4_frame_index"], "mp4_frame_index")
    count = len(frame)
    if not np.array_equal(frame, np.arange(count)):
        raise ValueError("MP4 frame rows must be complete and ordered from zero")
    timestamps = _integers(frame_map["vrs_device_timestamp_ns"], "vrs_device_timestamp_ns")
    vrs_index = _integers(frame_map["vrs_rgb_frame_index"], "vrs_rgb_frame_index")
    status, confidence = np.asarray(frame_map["status"]), np.asarray(frame_map["match_confidence"])
    if any(value.shape != (count,) for value in (timestamps, vrs_index, status, confidence)):
        raise ValueError("frame map columns must have equal lengths")
    if not np.isin(status, ("VERIFIED", "INFERRED", "UNRESOLVED")).all():
        raise ValueError("unknown mapping classification")
    if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
        raise ValueError("mapping confidence must be finite and in [0,1]")
    assigned = status != "UNRESOLVED"
    if np.any(timestamps[assigned] < 0) or np.any(vrs_index[assigned] < 0):
        raise ValueError("assigned mapping must retain exact timestamp and VRS source index")
    if np.any(timestamps[~assigned] != -1) or np.any(vrs_index[~assigned] != -1):
        raise ValueError("unresolved mappings must not carry a usable timestamp or index")
    _times(timestamps[assigned], "assigned device timestamps")
    _times(vrs_index[assigned], "assigned VRS indices")
    verified = status == "VERIFIED"
    pose_i, pose_r, pose_ok = nearest_local_ns(
        _times(streams.pose_device_us, "pose_device_us", microseconds=True),
        timestamps,
        verified,
        maximum_gap_ns=thresholds.pose_max_gap_ns,
    )
    hand_i, hand_r, hand_ok = nearest_local_ns(
        _times(streams.hand_device_us, "hand_device_us", microseconds=True),
        timestamps,
        verified,
        maximum_gap_ns=thresholds.hand_max_gap_ns,
    )
    translations, quaternions = np.full((count, 3), np.nan), np.full((count, 4), np.nan)
    translations[pose_ok] = streams.pose_translation_m[pose_i[pose_ok]]
    quaternions[pose_ok] = streams.pose_quaternion_xyzw[pose_i[pose_ok]]
    points = np.full((count, 2, 21, 3), np.nan)
    hand_confidence = np.full((count, 2), np.nan)
    hand_pose_r = np.full(count, NO_RESIDUAL, dtype=np.int64)
    hand_pose_ok = np.zeros(count, dtype=bool)
    hand_confidence[hand_ok] = streams.hand_confidence[hand_i[hand_ok]]
    hand_pose_r[hand_ok] = streams.hand_pose_residual_us[hand_i[hand_ok]] * 1000
    hand_pose_ok[hand_ok] = streams.hand_pose_accepted[hand_i[hand_ok]] & (
        np.abs(hand_pose_r[hand_ok]) <= thresholds.hand_pose_max_gap_ns
    )
    points[hand_pose_ok] = streams.hand_points_shared_m[hand_i[hand_pose_ok]]
    present = np.isfinite(points).all(axis=(2, 3))
    good = present & (hand_confidence >= thresholds.minimum_hand_confidence)
    arrays = {
        "mp4_frame_index": frame,
        "device_timestamp_ns": timestamps,
        "mapping_status": status,
        "mapping_confidence": confidence,
        "mapping_verified": verified,
        "pose_source_index": pose_i,
        "pose_residual_ns": pose_r,
        "pose_accepted": pose_ok,
        "pose_translation_m": translations,
        "pose_quaternion_xyzw": quaternions,
        "hand_source_index": hand_i,
        "hand_residual_ns": hand_r,
        "hand_accepted": hand_ok,
        "hand_points_shared_m": points,
        "hand_confidence": hand_confidence,
        "hand_present": present,
        "hand_high_confidence": good,
        "hand_pose_residual_ns": hand_pose_r,
        "hand_pose_accepted": hand_pose_ok,
        "pose_timestamp_ns": np.where(pose_ok, timestamps + np.where(pose_ok, pose_r, 0), -1),
        "hand_timestamp_ns": np.where(hand_ok, timestamps + np.where(hand_ok, hand_r, 0), -1),
    }
    summary = {
        "recording_id": recording_id,
        "participant": participant,
        "device_clock": streams.device_clock_name,
        "world_frame": streams.world_frame,
        "mapping_counts": {
            name: int((status == name).sum()) for name in ("VERIFIED", "INFERRED", "UNRESOLVED")
        },
        "mapping_confidence_is_probability": False,
        "source_hand_row_count": len(streams.hand_device_us),
        "pose": _match_summary(pose_i, pose_r, pose_ok),
        "hands": _match_summary(hand_i, hand_r, hand_ok),
        "hand_geometry": {
            side: {
                "present_count": int(present[:, index].sum()),
                "present_coverage": float(present[:, index].mean()) if count else 0.0,
                "high_confidence_count": int(good[:, index].sum()),
                "high_confidence_coverage": float(good[:, index].mean()) if count else 0.0,
                "missing_source_geometry_count_among_accepted_rows": int(
                    (
                        hand_ok
                        & ~np.isfinite(
                            streams.hand_points_device_m[np.maximum(hand_i, 0), index]
                        ).all(axis=(1, 2))
                    ).sum()
                )
                if len(streams.hand_device_us) and streams.hand_points_device_m is not None
                else (0 if not len(streams.hand_device_us) else None),
                "source_missing_geometry_count": int(
                    (~np.isfinite(streams.hand_points_device_m[:, index]).all(axis=(1, 2))).sum()
                )
                if streams.hand_points_device_m is not None
                else None,
                "source_missing_confidence_sentinel_count": int(
                    (streams.hand_confidence[:, index] == -1).sum()
                ),
                "confidence": statistics(hand_confidence[present[:, index], index]),
            }
            for index, side in enumerate(("left", "right"))
        },
        "hand_pose_secondary_gap_or_source_rejected_count": int((hand_ok & ~hand_pose_ok).sum()),
    }
    return FrameValidation(recording_id, participant, streams.world_frame, arrays, summary)


@dataclass(frozen=True)
class BoundHandover:
    annotation: HandoverAnnotation
    start_frame: int
    end_frame: int
    binding_evidence: MappingEvidence


def bind_annotations(
    annotations: Sequence[HandoverAnnotation],
    *,
    recording_id: str,
    frame_count: int,
    binding_evidence: MappingEvidence,
) -> tuple[BoundHandover, ...]:
    """Bind supplied source bounds to the explicitly authorized sync-frame contract.

    Arithmetic is an additional check, never the source of binding authority.
    QC evaluates both boundary frames inclusively; source endpoint-inclusion
    semantics remain uninterpreted. Initiator/delivery source labels are intact.
    """
    if binding_evidence.name != "annotation_to_comind_sync_frame_index":
        raise ValueError("explicit annotation_to_comind_sync_frame_index evidence is required")
    binding_evidence.require_verified()
    if any(annotation.recording_id != recording_id for annotation in annotations):
        raise ValueError("annotations belong to a different recording")
    audit = audit_annotation_frame_times(
        annotations, frame_rate=30, tolerance_seconds=1e-9, video_frame_count=frame_count
    )
    if annotations and not audit.hypotheses[0].consistent:
        raise ValueError("annotation source bounds fail the zero-based 30fps arithmetic audit")
    result = []
    for annotation in annotations:
        assert annotation.start_frame is not None and annotation.end_frame is not None
        result.append(
            BoundHandover(
                annotation, annotation.start_frame, annotation.end_frame, binding_evidence
            )
        )
    return tuple(result)


def rank_handovers(
    helper: FrameValidation,
    leader: FrameValidation,
    annotations: Sequence[BoundHandover],
    *,
    alignment_evidence: MappingEvidence,
    context_frames: int = 120,
    thresholds: TemporalThresholds = DEFAULT_THRESHOLDS,
) -> dict[str, object]:
    """Rank observed quality on common video indices, never cross-clock differences.

    Selection requires VERIFIED frame mappings throughout the proposed clip.
    Visibility remains unavailable without a verified image projection/occlusion
    test. Hand tracking presence is not claimed to be visibility.
    """
    if alignment_evidence.name != "paired_video_frame_alignment":
        raise ValueError("paired_video_frame_alignment evidence is required")
    alignment_evidence.require_verified()
    if (helper.participant, leader.participant) != ROLES:
        raise ValueError("ranking requires helper then leader")
    if helper.recording_id != leader.recording_id or helper.world_frame != leader.world_frame:
        raise ValueError("ranking requires the same recording and verified shared world")
    if type(context_frames) is not int or context_frames < 0:
        raise ValueError("context_frames must be a nonnegative integer")
    count = len(helper.arrays["mp4_frame_index"])
    if count != len(leader.arrays["mp4_frame_index"]):
        raise ValueError("paired video frame counts differ")
    rows = []
    for index, binding in enumerate(annotations):
        annotation = binding.annotation
        if annotation.recording_id != helper.recording_id:
            raise ValueError("annotation belongs to another recording")
        start, end = binding.start_frame, binding.end_frame
        if not 0 <= start <= end < count:
            raise ValueError("annotation bound outside paired videos")
        window = slice(start, end + 1)
        clip_start, clip_end = max(0, start - context_frames), min(count - 1, end + context_frames)
        context = slice(clip_start, clip_end + 1)
        participants, valid_hands, wrist_positions = {}, [], []
        all_context_verified = True
        quality_values, pose_rates, residuals, motion_rates, map_confidences = [], [], [], [], []
        for validation in (helper, leader):
            a = validation.arrays
            verified = a["mapping_verified"][window]
            all_context_verified &= bool(a["mapping_verified"][context].all())
            high = a["hand_high_confidence"][window]
            present = a["hand_present"][window]
            positions = a["hand_points_shared_m"][window, :, WRIST_INDEX, :]
            valid_hands.append(high)
            wrist_positions.append(positions)
            dt = np.diff(a["hand_timestamp_ns"][window]).astype(float) / 1e9
            continuity = (
                high[1:]
                & high[:-1]
                & (dt[:, None] > 0)
                & (dt[:, None] <= thresholds.continuity_max_gap_ns / 1e9)
            )
            speeds = np.full(continuity.shape, np.nan)
            displacement = np.linalg.norm(np.diff(positions, axis=0), axis=2)
            np.divide(displacement, dt[:, None], out=speeds, where=continuity)
            outliers = speeds > thresholds.maximum_wrist_speed_m_s
            motion_rates.append(float(outliers.sum() / max(1, continuity.sum())))
            confidence = a["hand_confidence"][window][present]
            quality_values.extend(confidence[np.isfinite(confidence)].tolist())
            map_confidence = a["mapping_confidence"][window][verified]
            map_confidences.append(float(np.median(map_confidence)) if len(map_confidence) else 0.0)
            pose_rates.append(float(a["pose_accepted"][window].mean()))
            local_residuals = []
            for prefix in ("pose", "hand"):
                residual = a[f"{prefix}_residual_ns"][window]
                local_residuals.extend((np.abs(residual[residual != NO_RESIDUAL]) / 1e6).tolist())
            residuals.extend(local_residuals)
            participants[validation.participant] = {
                "mapping_verified_coverage": float(verified.mean()),
                "mapping_confidence": statistics(a["mapping_confidence"][window][verified]),
                "context_mapping_verified_coverage": float(a["mapping_verified"][context].mean()),
                "pose_coverage": pose_rates[-1],
                "hand_row_coverage": float(a["hand_accepted"][window].mean()),
                "hand_present_coverage": present.mean(axis=0).tolist(),
                "hand_high_confidence_coverage": high.mean(axis=0).tolist(),
                "hand_confidence": statistics(confidence),
                "pose_absolute_residual_ms": statistics(
                    np.abs(
                        a["pose_residual_ns"][window][a["pose_residual_ns"][window] != NO_RESIDUAL]
                    )
                    / 1e6
                ),
                "hand_absolute_residual_ms": statistics(
                    np.abs(
                        a["hand_residual_ns"][window][a["hand_residual_ns"][window] != NO_RESIDUAL]
                    )
                    / 1e6
                ),
                "world_wrist_speed_m_s": statistics(speeds),
                "wrist_speed_outlier_count": int(outliers.sum()),
                "continuity_evaluated_interval_count": int(continuity.sum()),
                "nonincreasing_selected_hand_time_interval_count": int((dt <= 0).sum()),
                "continuity_time_basis": "Selected hand source DEVICE_TIME, same participant only",
            }
        pair_valid = valid_hands[0][:, :, None] & valid_hands[1][:, None, :]
        distance = np.linalg.norm(
            wrist_positions[0][:, :, None, :] - wrist_positions[1][:, None, :, :], axis=3
        )
        distance[~pair_valid] = np.nan
        nearest = np.min(np.where(pair_valid, distance, np.inf), axis=(1, 2))
        nearest[~np.isfinite(nearest)] = np.nan
        both_any = valid_hands[0].any(axis=1) & valid_hands[1].any(axis=1)
        all_four = valid_hands[0].all(axis=1) & valid_hands[1].all(axis=1)
        score = [
            float(both_any.mean()),
            float(all_four.mean()),
            min(pose_rates),
            min(map_confidences),
            float(np.median(quality_values)) if quality_values else -1.0,
            -float(np.percentile(residuals, 95)) if residuals else -1e30,
            -float(np.median(nearest[np.isfinite(nearest)]))
            if np.isfinite(nearest).any()
            else -1e30,
            -max(motion_rates),
        ]
        rows.append(
            {
                "annotation_index": index,
                "annotation_id": annotation.annotation_id,
                "start_frame": start,
                "end_frame": end,
                "context_start_frame": clip_start,
                "context_end_frame": clip_end,
                "eligible_verified_context": all_context_verified,
                "lexicographic_quality_score": score,
                "both_participants_any_high_confidence_hand_coverage": float(both_any.mean()),
                "all_four_high_confidence_hands_coverage": float(all_four.mean()),
                "participants": participants,
                "nearest_cross_participant_wrist_distance_m": statistics(nearest),
                "cross_person_distance_frame_coverage": float(np.isfinite(nearest).mean()),
                "visibility": {
                    "status": "UNAVAILABLE",
                    "reason": "No per-frame calibrated projection and occlusion test supplied; tracking is not visibility",
                },
                "initiator_source_label": annotation.initiator,
                "delivering_flow_source_label": annotation.delivering_flow,
                "object_category_level_1": annotation.object_category_level_1,
                "object_category_level_2": annotation.object_category_level_2,
                "object_category_level_3": annotation.object_category_level_3,
                "initiation_type": annotation.initiation_type,
            }
        )
    ranked = sorted(
        rows,
        key=lambda row: (
            row["eligible_verified_context"],
            row["lexicographic_quality_score"],
            -row["annotation_index"],
        ),
        reverse=True,
    )
    selected = next((row for row in ranked if row["eligible_verified_context"]), None)
    return {
        "common_timeline": "comind_sync_frame_index",
        "paired_video_alignment": alignment_evidence.to_dict(),
        "cross_participant_device_times_compared": False,
        "boundary_evaluation": "Both supplied boundary frames included for QC; source endpoint semantics preserved",
        "ranking_order": [
            "both participants have any confident hand",
            "all four confident hands",
            "minimum participant pose coverage",
            "minimum participant median mapping separation confidence",
            "median hand confidence",
            "lower p95 residual",
            "lower median nearest cross-participant wrist distance",
            "lower wrist speed outlier fraction",
            "source annotation order",
        ],
        "selected_annotation_index": None if selected is None else selected["annotation_index"],
        "selected_annotation_id": None if selected is None else selected["annotation_id"],
        "selection_status": "VERIFIED_CONTEXT_AVAILABLE" if selected else "NO_VERIFIED_CONTEXT",
        "handovers": ranked,
    }
