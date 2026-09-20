"""Flat, scoped demonstration quality metrics from already validated frame caches.

Tracking, calibrated projection and actual visibility are different claims.
These functions perform no image/CSV/VRS reads and never compare participants'
device clocks. Distances compare accepted geometry at the same declared paired
video index. Publication recommendations concern tracking quality only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from decimal import Decimal
from numbers import Real
from typing import TypeAlias

import numpy as np

from duet.adapters.comind.playback import PlaybackBundle
from duet.adapters.comind.temporal_validation import NO_RESIDUAL, WRIST_INDEX
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations

Scalar: TypeAlias = str | int | float | bool | None
ROLES = ("helper", "leader")
SIDES = ("left", "right")
OBJECTIVE_RANKING_RULE = (
    "Lexicographic: verified context mapping; minimum event/context pose coverage; "
    "minimum event/context all-four hand coverage (worst shown scope); event all-four hand "
    "coverage; minimum camera event wrist projection fraction; median accepted context "
    "hand confidence; lower maximum accepted context residual; source segment key tie-break"
)


def _scalar(value: object) -> Scalar:
    """Preserve scalar labels and encode source list/object labels as JSON text."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, Decimal)):
        if not np.isfinite(float(value)):
            raise ValueError("source labels cannot contain nonfinite numbers")
        return str(value) if isinstance(value, Decimal) else value
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)


def _maximum(values: list[np.ndarray]) -> float | None:
    nonempty = [np.asarray(value) for value in values if len(value)]
    return float(max(np.max(value) for value in nonempty)) if nonempty else None


def _cross_person_distances(
    helper_points: np.ndarray,
    leader_points: np.ndarray,
    helper_valid: np.ndarray,
    leader_valid: np.ndarray,
) -> tuple[float | None, float | None, int]:
    """Minimum over 21×21 landmarks and separately wrist index 5, in meters."""
    landmark_minimum, wrist_minimum = None, None
    frame_has_pair = helper_valid.any(axis=1) & leader_valid.any(axis=1)
    for helper_side in range(2):
        for leader_side in range(2):
            valid = helper_valid[:, helper_side] & leader_valid[:, leader_side]
            if not valid.any():
                continue
            helper = helper_points[valid, helper_side]
            leader = leader_points[valid, leader_side]
            wrist = np.linalg.norm(helper[:, WRIST_INDEX] - leader[:, WRIST_INDEX], axis=1)
            value = float(np.min(wrist))
            wrist_minimum = value if wrist_minimum is None else min(value, wrist_minimum)
            # Bound temporary memory even if a future annotation spans a long recording.
            for start in range(0, len(helper), 256):
                delta = (
                    helper[start : start + 256, :, None, :]
                    - leader[start : start + 256, None, :, :]
                )
                value = float(np.sqrt(np.min(np.sum(delta * delta, axis=3))))
                landmark_minimum = (
                    value if landmark_minimum is None else min(value, landmark_minimum)
                )
    return landmark_minimum, wrist_minimum, int(frame_has_pair.sum())


def _scope_metrics(bundle: PlaybackBundle, start: int, end: int, scope: str) -> dict[str, Scalar]:
    window = slice(start, end + 1)
    count = end - start + 1
    result: dict[str, Scalar] = {f"{scope}_frame_count": count}
    hand_masks, point_arrays, accepted_confidences = [], [], []
    residual_groups: dict[str, list[np.ndarray]] = {
        "pose": [],
        "hand": [],
        "secondary_hand_pose": [],
    }
    role_pose_coverage, all_mapping, any_qc = [], True, False
    aggregate_rejections = {"pose": 0, "hand": 0, "secondary_hand_pose": 0}
    total_motion_outliers = 0
    for role in ROLES:
        arrays = bundle.aligned_arrays[role]
        prefix = f"{scope}_{role}"
        verified = np.asarray(arrays["mapping_verified"][window], dtype=bool)
        all_mapping &= bool(verified.all())
        result[f"{prefix}_mapping_verified_coverage"] = float(verified.mean())
        pose = arrays["pose_accepted"][window]
        hand_row = arrays["hand_accepted"][window]
        secondary = arrays["hand_pose_accepted"][window]
        present = arrays["hand_present"][window]
        accepted = arrays["hand_high_confidence"][window]
        points = arrays["hand_points_shared_m"][window]
        confidence = arrays["hand_confidence"][window]
        if np.any(accepted & (~present | ~hand_row[:, None] | ~secondary[:, None])):
            raise ValueError("accepted hand geometry violates source/pose availability")
        if np.any(accepted & ~np.isfinite(points).all(axis=(2, 3))):
            raise ValueError("accepted hand geometry must be finite")
        hand_masks.append(accepted)
        point_arrays.append(points)
        accepted_confidences.append(confidence[accepted])
        available_confidence = confidence[accepted]
        result[f"{prefix}_any_hand_coverage"] = float(accepted.any(axis=1).mean())
        result[f"{prefix}_median_accepted_hand_confidence"] = (
            float(np.median(available_confidence)) if len(available_confidence) else None
        )
        role_pose_coverage.append(float(pose.mean()))
        result[f"{prefix}_pose_coverage"] = role_pose_coverage[-1]
        result[f"{prefix}_hand_sample_coverage"] = float(hand_row.mean())
        for side_index, side in enumerate(SIDES):
            result[f"{prefix}_{side}_hand_coverage"] = float(accepted[:, side_index].mean())
            result[f"{prefix}_{side}_missing_geometry_count"] = int((~present[:, side_index]).sum())
            result[f"{prefix}_{side}_below_or_unknown_confidence_count"] = int(
                (present[:, side_index] & ~accepted[:, side_index]).sum()
            )
        for stream, mask in (
            ("pose", pose),
            ("hand", hand_row),
            ("secondary_hand_pose", secondary),
        ):
            source_key = (
                "hand_pose_residual_ns"
                if stream == "secondary_hand_pose"
                else f"{stream}_residual_ns"
            )
            residuals = arrays[source_key][window]
            if np.any(residuals[mask] == NO_RESIDUAL):
                raise ValueError("accepted sample is missing its timestamp residual")
            values = np.abs(residuals[mask]).astype(float) / 1e6
            residual_groups[stream].append(values)
            result[f"{prefix}_{stream}_max_accepted_residual_ms"] = _maximum([values])
            rejected = (
                int((hand_row & ~secondary).sum())
                if stream == "secondary_hand_pose"
                else int((~mask).sum())
            )
            aggregate_rejections[stream] += rejected
            result[f"{prefix}_{stream}_rejected_count"] = rejected
            if stream != "secondary_hand_pose":
                has_candidate = arrays[f"{stream}_source_index"][window] >= 0
                result[f"{prefix}_{stream}_gap_rejected_count"] = int((has_candidate & ~mask).sum())
                result[f"{prefix}_{stream}_missing_sample_count"] = int((~has_candidate).sum())
        hand_times = arrays.get("hand_timestamp_ns")
        if hand_times is None:
            device = arrays["device_timestamp_ns"][window]
            residual = arrays["hand_residual_ns"][window]
            hand_times = np.where(hand_row, device + np.where(hand_row, residual, 0), -1)
        else:
            hand_times = hand_times[window]
        dt = np.diff(hand_times).astype(float) / 1e9
        valid_step = accepted[1:] & accepted[:-1] & (dt[:, None] > 0) & (dt[:, None] <= 0.1)
        step = np.linalg.norm(np.diff(points[:, :, WRIST_INDEX], axis=0), axis=2)
        speed = np.zeros(step.shape)
        np.divide(step, dt[:, None], out=speed, where=valid_step)
        motion_outliers = int((valid_step & (speed > 5.0)).sum())
        total_motion_outliers += motion_outliers
        result[f"{prefix}_wrist_speed_outlier_count"] = motion_outliers
        any_qc |= bool((~pose).any() or (~accepted).any() or motion_outliers)
    all_four = hand_masks[0].all(axis=1) & hand_masks[1].all(axis=1)
    result[f"{scope}_mapping_verified"] = all_mapping
    result[f"{scope}_all_four_hands_coverage"] = float(all_four.mean())
    result[f"{scope}_minimum_participant_pose_coverage"] = min(role_pose_coverage)
    pooled = np.concatenate(accepted_confidences)
    if len(pooled) and (not np.isfinite(pooled).all() or np.any((pooled < 0) | (pooled > 1))):
        raise ValueError("accepted hand confidence must be finite and in [0,1]")
    result[f"{scope}_median_accepted_hand_confidence"] = (
        float(np.median(pooled)) if len(pooled) else None
    )
    for stream, values in residual_groups.items():
        result[f"{scope}_{stream}_max_accepted_residual_ms"] = _maximum(values)
        result[f"{scope}_{stream}_rejected_count"] = aggregate_rejections[stream]
    result[f"{scope}_max_accepted_residual_ms"] = _maximum(
        [value for values in residual_groups.values() for value in values]
    )
    landmark, wrist, pair_count = _cross_person_distances(
        point_arrays[0], point_arrays[1], hand_masks[0], hand_masks[1]
    )
    result[f"{scope}_min_cross_person_landmark_distance_m"] = landmark
    result[f"{scope}_min_cross_person_wrist_distance_m"] = wrist
    result[f"{scope}_cross_person_distance_frame_coverage"] = pair_count / count
    result[f"{scope}_wrist_speed_outlier_count"] = total_motion_outliers
    result[f"{scope}_any_missing_or_tracking_qc"] = any_qc or not all_mapping
    return result


def _projection_metrics(
    bundle: PlaybackBundle, report: Mapping, selection: Mapping | None = None
) -> dict[str, Scalar]:
    selection = bundle.selection if selection is None else selection
    result: dict[str, Scalar] = {
        "event_projection_available": False,
        "event_min_camera_wrist_projection_fraction": None,
        "context_projection_available": False,
        "context_min_camera_wrist_projection_fraction": None,
        "projection_scope": "Event-only calibrated image/domain coverage; no occlusion or actual visibility",
    }
    for role in ROLES:
        result[f"event_{role}_camera_wrist_projection_fraction"] = None
        result[f"context_{role}_camera_wrist_projection_fraction"] = None
    if not report:
        return result
    generations = {str(bundle.aligned_arrays[role]["generation_id"].item()) for role in ROLES}
    if report.get("recording_id") != bundle.bindings.recording_id or generations != {
        report.get("generation_id")
    }:
        raise ValueError("projection report belongs to another recording or cache generation")
    if (
        report.get("timeline") != "comind_sync_frame_index"
        or report.get("shared_world", {}).get("status") != "VERIFIED"
        or report.get("shared_world", {}).get("frame") != bundle.bindings.world_frame.name
    ):
        raise ValueError("projection report has a different timeline or world frame")
    candidates = [
        item
        for item in report.get("handovers", ())
        if item.get("annotation_id") == selection["annotation_id"]
    ]
    if not candidates:
        return result
    if len(candidates) != 1:
        raise ValueError("projection annotation ID must be unique")
    selected = candidates[0]
    start, end = selection["start_frame"], selection["end_frame"]
    if (selected["start_frame"], selected["end_frame"]) != (start, end):
        raise ValueError("projection annotation bounds differ from the selected source interval")
    count = end - start + 1
    required = {f"{role}/{side}/wrist" for role in ROLES for side in SIDES}
    fractions = []
    for role in ROLES:
        camera = selected.get("cameras", {}).get(role)
        if camera is None:
            continue
        metrics = camera["metrics"]
        if (
            metrics.get("camera_frame") != f"{role}/camera-rgb"
            or metrics.get("world_frame") != bundle.bindings.world_frame.name
        ):
            raise ValueError("projection camera or coordinate frame identity differs")
        if (
            type(metrics["snapshot_count"]) is not int
            or metrics["snapshot_count"] != count
            or set(metrics["points"]) != required
        ):
            raise ValueError("projection report must contain all four wrists for the event")
        counts = [item["in_image_count"] for item in metrics["points"].values()]
        if any(type(value) is not int or not 0 <= value <= count for value in counts):
            raise ValueError("invalid projection coverage count")
        fraction = sum(counts) / (4 * count)
        result[f"event_{role}_camera_wrist_projection_fraction"] = fraction
        fractions.append(fraction)
    if len(fractions) == 2:
        result["event_projection_available"] = True
        result["event_min_camera_wrist_projection_fraction"] = min(fractions)
    return result


def public_recommendation(row: Mapping[str, Scalar]) -> tuple[str, str]:
    """A disclosed tracking-quality rule, not a general publication/safety claim."""
    if not row["context_mapping_verified"]:
        return "exclude_from_public_demo", "At least one context mapping is not VERIFIED"
    if row["event_minimum_participant_pose_coverage"] < 1:
        return "exclude_from_public_demo", "Event device-pose coverage is incomplete"
    if row["event_all_four_hands_coverage"] < 0.90:
        return "exclude_from_public_demo", "All-four event hand coverage is below 90%"
    if row["event_all_four_hands_coverage"] < 1:
        return "tracking_caution", "All-four event hand coverage is at least 90% but below 100%"
    if (
        row["context_all_four_hands_coverage"] < 0.75
        or row["context_minimum_participant_pose_coverage"] < 1
    ):
        return (
            "tracking_caution",
            "Event tracking is complete but context tracking has substantial gaps",
        )
    if row["event_wrist_speed_outlier_count"] or row["context_wrist_speed_outlier_count"]:
        return "tracking_caution", "Accepted wrist motion exceeds the disclosed 5 m/s QC threshold"
    return (
        "acceptable",
        "All-four event tracking and event poses are complete; context passes disclosed tracking thresholds",
    )


def _annotation_metrics(
    bundle: PlaybackBundle,
    projection_report: Mapping,
    *,
    selection: Mapping,
    context_start: int,
    context_end: int,
) -> dict[str, Scalar]:
    """Return JSON/CSV-safe scalar metrics with explicit event/context denominators.

    ``numeric_id`` is the distinct original source field when already joined to
    selection; otherwise it is None. It is never fabricated from annotation_id.
    Source list/dictionary labels are represented as JSON text for CSV safety.
    """
    annotation_id = selection["annotation_id"]
    start, end = selection["start_frame"], selection["end_frame"]
    if not isinstance(annotation_id, str) or not 0 <= context_start <= start <= end <= context_end:
        raise ValueError(
            "metrics require one identified event contained within its selected context"
        )
    row: dict[str, Scalar] = {
        "recording_id": bundle.bindings.recording_id,
        "annotation_id": annotation_id,
        "numeric_id": _scalar(selection.get("numeric_id")),
        "start_frame": start,
        "end_frame": end,
        "clip_start_frame": context_start,
        "clip_end_frame": context_end,
        "event_start_frame": start,
        "event_end_frame": end,
        "context_start_frame": context_start,
        "context_end_frame": context_end,
        "duration_seconds": (context_end - context_start + 1) / 30,
        "event_duration_seconds": (end - start + 1) / 30,
        "duration_semantics": "Inclusive frame count / 30 presentation fps; not DEVICE_TIME",
        "object_category": _scalar(selection.get("object_category_level_1")),
        "object_category_level_1": _scalar(selection.get("object_category_level_1")),
        "object_category_level_2": _scalar(selection.get("object_category_level_2")),
        "object_category_level_3": _scalar(selection.get("object_category_level_3")),
        "initiation_type": _scalar(selection.get("initiation_type")),
        "initiator_source_label": _scalar(selection.get("initiator_source_label")),
        "delivering_flow_source_label": _scalar(selection.get("delivering_flow_source_label")),
        "hand_confidence_threshold": bundle.bindings.minimum_hand_confidence,
        "wrist_speed_threshold_m_s": 5.0,
        "continuity_max_gap_seconds": 0.1,
        "minimum_public_event_four_hand_coverage": 0.9,
        "minimum_context_four_hand_coverage_without_caution": 0.75,
        "missing_points_and_rejected_samples_are_excluded_from_distances": True,
        "annotation_boundary_policy": "Both source boundary frames included for display/QC",
    }
    row.update(_scope_metrics(bundle, start, end, "event"))
    row.update(_scope_metrics(bundle, context_start, context_end, "context"))
    row.update(_projection_metrics(bundle, projection_report, selection))
    recommendation, reason = public_recommendation(row)
    row.update(public_recommendation=recommendation, public_recommendation_reason=reason)
    row["recommendation_scope"] = (
        "Tracking quality only; no judgment of other publication requirements"
    )
    return row


def annotation_metrics(bundle: PlaybackBundle, projection_report: Mapping) -> dict[str, Scalar]:
    """Return scoped metrics and the unchanged strict public recommendation."""
    return _annotation_metrics(
        bundle,
        projection_report,
        selection=bundle.selection,
        context_start=bundle.start_index,
        context_end=bundle.end_index,
    )


def demo_qc_assessment(row: Mapping[str, Scalar]) -> dict[str, Scalar]:
    """Apply the separate DEMO_QC_V1 rule to scoped canonical metrics.

    ANY hand means the per-frame left/right union for each participant, never
    the maximum individual-hand coverage. Confidence pools available tracked
    hand observations at the canonical >=0.5 gate. This is an availability
    proxy; it does not identify the contacting hand or establish visibility.
    """
    if row["hand_confidence_threshold"] != 0.5:
        raise ValueError("DEMO_QC_V1 requires the canonical 0.5 tracked-hand confidence gate")
    if row["wrist_speed_threshold_m_s"] != 5.0 or row["continuity_max_gap_seconds"] != 0.1:
        raise ValueError("DEMO_QC_V1 requires 5 m/s wrist QC over valid steps at most 0.1 seconds")
    if row["public_recommendation"] not in (
        "acceptable",
        "tracking_caution",
        "exclude_from_public_demo",
    ):
        raise ValueError("unrecognized strict recommendation")
    coverage_fields = [
        f"{scope}_{role}_pose_coverage" for scope in ("event", "context") for role in ROLES
    ] + [f"event_{role}_any_hand_coverage" for role in ROLES]
    for name in coverage_fields:
        value = row[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise ValueError("demo coverage must be a finite fraction in [0,1]")
    for scope in ("event", "context"):
        if type(row[f"{scope}_mapping_verified"]) is not bool:
            raise ValueError("demo mapping verification must be an explicit boolean")
        count = row[f"{scope}_wrist_speed_outlier_count"]
        if type(count) is not int or count < 0:
            raise ValueError("demo wrist outlier counts must be nonnegative integers")
    confidence = row["event_median_accepted_hand_confidence"]
    if confidence is not None and (
        isinstance(confidence, bool)
        or not isinstance(confidence, Real)
        or not np.isfinite(confidence)
        or not 0 <= confidence <= 1
    ):
        raise ValueError("available-hand confidence must be missing or finite in [0,1]")
    failures = []
    for scope in ("event", "context"):
        if not row[f"{scope}_mapping_verified"]:
            failures.append(f"{scope} mappings are not all VERIFIED")
        for role in ROLES:
            if row[f"{scope}_{role}_pose_coverage"] < 0.95:
                failures.append(f"{scope} {role} pose coverage is below 95%")
        if row[f"{scope}_wrist_speed_outlier_count"]:
            failures.append(f"{scope} accepted wrist speed exceeds 5 m/s")
    for role in ROLES:
        if row[f"event_{role}_any_hand_coverage"] < 0.80:
            failures.append(f"event {role} ANY tracked-hand coverage is below 80%")
    if confidence is None or not np.isfinite(confidence) or confidence < 0.90:
        failures.append("pooled event available-hand median confidence is below 0.90 or missing")
    return {
        "demo_qc_profile": "DEMO_QC_V1",
        "demo_qc_pass": not failures,
        "demo_qc_recommendation": "exclude_from_demo" if failures else "acceptable",
        "demo_qc_reason": "; ".join(failures)
        if failures
        else "Verified event/context maps, sufficient poses and per-participant ANY-hand coverage, high pooled available-hand confidence, and no severe accepted wrist-speed outlier",
        "demo_qc_minimum_pose_coverage": 0.95,
        "demo_qc_minimum_participant_any_hand_coverage": 0.80,
        "demo_qc_minimum_pooled_hand_confidence": 0.90,
        "demo_qc_tracked_hand_confidence_gate": 0.5,
        "demo_qc_maximum_wrist_speed_m_s": 5.0,
        "demo_qc_continuity_max_gap_seconds": 0.1,
        "demo_qc_hand_proxy": "Per-frame left/right union of available tracked observations per participant; pooled event confidence across those observations. No interacting-hand identity or actual visibility inferred; per-role confidence medians are diagnostic only.",
        "strict_qc_pass": row["public_recommendation"] in ("acceptable", "tracking_caution"),
        "strict_qc_recommendation": row["public_recommendation"],
        "strict_qc_reason": row["public_recommendation_reason"],
    }


def assess_demo_qc(
    base: PlaybackBundle,
    annotation_id: str,
    projection_report: Mapping,
    *,
    context_frames: int = 120,
) -> dict[str, Scalar]:
    """Assess any source annotation from full cached arrays without making it playable.

    Unresolved mappings produce a rejected diagnostic row. This function never
    constructs/reselects a PlaybackBundle or calls its frame iterator. Invalid
    accepted geometry fails closed with ValueError; absent or rejected samples
    retain ordinary missing-coverage counts. Rendering still requires separate
    canonical playback validation after this assessment passes.
    """
    if type(context_frames) is not int or context_frames < 0:
        raise ValueError("context_frames must be a nonnegative integer")
    matches = [row for row in base.annotations if row["annotation_id"] == annotation_id]
    if len(matches) != 1:
        raise ValueError("annotation_id must identify exactly one source annotation")
    selection = matches[0]
    counts = {base.frame_maps[role].frame_count for role in ROLES}
    if len(counts) != 1:
        raise ValueError("paired video frame counts differ")
    count = counts.pop()
    start, end = selection["start_frame"], selection["end_frame"]
    if type(start) is not int or type(end) is not int or not 0 <= start <= end < count:
        raise ValueError("annotation bounds must be integer frame indices within the videos")
    context_start, context_end = (
        max(0, start - context_frames),
        min(count - 1, end + context_frames),
    )
    window = slice(context_start, context_end + 1)
    for role in ROLES:
        arrays = base.aligned_arrays[role]
        actual_verified = base.frame_maps[role].status[window] == "VERIFIED"
        if not np.array_equal(arrays["mapping_verified"][window], actual_verified):
            raise ValueError("cached mapping mask disagrees with original frame-map status")
        accepted = arrays["pose_accepted"][window]
        if not np.isfinite(arrays["pose_translation_m"][window][accepted]).all():
            raise ValueError("accepted pose geometry must be finite")
        if "pose_quaternion_xyzw" in arrays:
            quaternion_xyzw_batch_to_rotations(arrays["pose_quaternion_xyzw"][window][accepted])
        tracked = arrays["hand_high_confidence"][window]
        confidence = arrays["hand_confidence"][window][tracked]
        if not np.isfinite(confidence).all() or np.any((confidence < 0.5) | (confidence > 1)):
            raise ValueError("tracked-hand mask violates the canonical >=0.5 confidence gate")
    row = _annotation_metrics(
        base,
        projection_report,
        selection=selection,
        context_start=context_start,
        context_end=context_end,
    )
    row.update(demo_qc_assessment(row))
    row["demo_qc_geometry_validation"] = (
        "Accepted source geometry must be finite and canonical pose rotations valid; invalid accepted geometry raises an error. Missing geometry is not promoted. Any accepted >5m/s wrist interval in event or context rejects this profile."
    )
    row["demo_qc_assessment_grants_playback"] = False
    return row


def rank_examples(rows: Sequence[Mapping[str, Scalar]]) -> list[dict[str, Scalar]]:
    """Deterministic objective ranking; unavailable metrics sort behind observed ones."""
    identifiers = [row["annotation_id"] for row in rows]
    if any(not isinstance(value, str) for value in identifiers) or len(set(identifiers)) != len(
        identifiers
    ):
        raise ValueError("ranking requires distinct source annotation ID strings")

    def quality(row: Mapping[str, Scalar]) -> tuple:
        projection = row["event_min_camera_wrist_projection_fraction"]
        confidence = row["context_median_accepted_hand_confidence"]
        residual = row["context_max_accepted_residual_ms"]
        return (
            -int(bool(row["context_mapping_verified"])),
            -min(
                float(row["event_minimum_participant_pose_coverage"]),
                float(row["context_minimum_participant_pose_coverage"]),
            ),
            -min(
                float(row["event_all_four_hands_coverage"]),
                float(row["context_all_four_hands_coverage"]),
            ),
            -float(row["event_all_four_hands_coverage"]),
            -float(projection) if projection is not None else 1.0,
            -float(confidence) if confidence is not None else 1.0,
            float(residual) if residual is not None else float("inf"),
            row["annotation_id"],
        )

    result = []
    for index, source in enumerate(sorted(rows, key=quality), start=1):
        row = dict(source)
        row["objective_rank"] = index
        row["objective_ranking_rule"] = OBJECTIVE_RANKING_RULE
        result.append(row)
    return result
