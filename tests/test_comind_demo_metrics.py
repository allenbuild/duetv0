"""Small derived-cache metrics; no images, files, SDK or raw dataset access."""

import json
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from duet.adapters.comind.demo_metrics import (
    annotation_metrics,
    assess_demo_qc,
    demo_qc_assessment,
    rank_examples,
)
from duet.adapters.comind.frame_map import VideoVrsFrameMap
from duet.adapters.comind.playback import PlaybackBundle
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.schemas.common import FrameId, Provenance
from duet.visualization.comind_handover import FrameIndexBindings

RECORDING = "43276420-701f-4731-b9ab-bebc7fd14994"


@pytest.fixture
def metrics_bundle():
    count = 10
    frame = FrameId(f"comind/{RECORDING}/multislam/world")
    evidence = MappingEvidence(
        "paired_video_frame_alignment", EvidenceStatus.VERIFIED, "Synthetic contract", ("fixture",)
    )
    maps, aligned = {}, {}
    for role, origin in (("helper", 1_000_000_000), ("leader", 70_000_000_000)):
        times = origin + np.arange(count, dtype=np.int64) * 33_333_000
        points = np.full(
            (count, 2, 21, 3), [0, 0, 0] if role == "helper" else [1, 0, 0], dtype=float
        )
        points[:, :, 0] = [10 if role == "helper" else 10.001, 0, 0]
        maps[role] = VideoVrsFrameMap(
            RECORDING,
            role,
            np.arange(count),
            np.arange(count),
            times,
            np.full(count, "VERIFIED"),
            np.full(count, 0.9),
            np.arange(count) * 512,
            Fraction(1, 15360),
            Provenance("fixture"),
        )
        aligned[role] = {
            "generation_id": np.asarray("fixture-generation"),
            "mapping_verified": np.ones(count, dtype=bool),
            "mapping_status": np.full(count, "VERIFIED"),
            "device_timestamp_ns": times,
            "pose_accepted": np.ones(count, dtype=bool),
            "pose_source_index": np.arange(count),
            "pose_residual_ns": np.full(count, 100, dtype=np.int64),
            "pose_translation_m": np.zeros((count, 3)),
            "hand_accepted": np.ones(count, dtype=bool),
            "hand_source_index": np.arange(count),
            "hand_residual_ns": np.full(count, 200, dtype=np.int64),
            "hand_timestamp_ns": times + 200,
            "hand_pose_accepted": np.ones(count, dtype=bool),
            "hand_pose_residual_ns": np.full(count, 300, dtype=np.int64),
            "hand_present": np.ones((count, 2), dtype=bool),
            "hand_high_confidence": np.ones((count, 2), dtype=bool),
            "hand_confidence": np.full((count, 2), 0.8),
            "hand_points_shared_m": points,
        }
    selection = {
        "annotation_id": "000002",
        "numeric_id": "19",
        "start_frame": 2,
        "end_frame": 6,
        "object_category_level_1": "bowl",
        "object_category_level_2": "bowl",
        "object_category_level_3": "serveware",
        "initiation_type": ["verbal"],
        "initiator_source_label": "left",
        "delivering_flow_source_label": "rtl",
    }
    return PlaybackBundle(
        FrameIndexBindings(RECORDING, frame, evidence),
        maps,
        0,
        9,
        {},
        selection,
        aligned,
        (selection,),
        Provenance("fixture"),
    )


def projection(bundle):
    cameras = {}
    for role in ("helper", "leader"):
        cameras[role] = {
            "metrics": {
                "snapshot_count": 5,
                "camera_frame": f"{role}/camera-rgb",
                "world_frame": bundle.bindings.world_frame.name,
                "points": {
                    f"{person}/{side}/wrist": {"in_image_count": 5}
                    for person in ("helper", "leader")
                    for side in ("left", "right")
                },
            }
        }
    return {
        "recording_id": RECORDING,
        "generation_id": "fixture-generation",
        "timeline": "comind_sync_frame_index",
        "shared_world": {"status": "VERIFIED", "frame": bundle.bindings.world_frame.name},
        "handovers": [
            {"annotation_id": "000002", "start_frame": 2, "end_frame": 6, "cameras": cameras}
        ],
    }


def test_scopes_and_distinct_source_identifiers_are_preserved(metrics_bundle):
    row = annotation_metrics(metrics_bundle, projection(metrics_bundle))
    assert row["annotation_id"] == "000002" and row["numeric_id"] == "19"
    assert row["event_frame_count"] == 5 and row["context_frame_count"] == 10
    assert row["duration_seconds"] == 10 / 30
    assert row["event_all_four_hands_coverage"] == 1
    assert row["context_all_four_hands_coverage"] == 1
    assert row["event_max_accepted_residual_ms"] == 0.0003
    assert row["event_median_accepted_hand_confidence"] == 0.8
    assert row["public_recommendation"] == "acceptable"
    assert row["initiation_type"] == '["verbal"]'
    assert row["initiator_source_label"] == "left"
    assert all(
        value is None or isinstance(value, (str, int, float, bool)) for value in row.values()
    )
    json.dumps(row, allow_nan=False)


def test_missing_numeric_id_is_not_fabricated_from_segment_key(metrics_bundle):
    selection = dict(metrics_bundle.selection)
    del selection["numeric_id"]
    row = annotation_metrics(replace(metrics_bundle, selection=selection), {})
    assert row["numeric_id"] is None and row["annotation_id"] == "000002"


def test_landmark_minimum_and_wrist_index_five_are_distinct(metrics_bundle):
    row = annotation_metrics(metrics_bundle, {})
    assert row["event_min_cross_person_landmark_distance_m"] == pytest.approx(0.001)
    assert row["event_min_cross_person_wrist_distance_m"] == 1.0
    assert row["event_cross_person_distance_frame_coverage"] == 1


def test_weak_or_missing_hands_do_not_contribute_distances_or_confidence(metrics_bundle):
    arrays = metrics_bundle.aligned_arrays["helper"]
    arrays["hand_high_confidence"][:] = False
    arrays["hand_confidence"][:] = 0
    row = annotation_metrics(metrics_bundle, {})
    assert row["event_min_cross_person_landmark_distance_m"] is None
    assert row["event_min_cross_person_wrist_distance_m"] is None
    assert row["event_cross_person_distance_frame_coverage"] == 0
    assert row["event_median_accepted_hand_confidence"] == 0.8  # leader only
    assert row["event_helper_left_below_or_unknown_confidence_count"] == 5
    assert row["event_any_missing_or_tracking_qc"]
    assert row["public_recommendation"] == "exclude_from_public_demo"


def test_rejected_samples_do_not_inflate_accepted_residual_maximum(metrics_bundle):
    arrays = metrics_bundle.aligned_arrays["helper"]
    arrays["pose_accepted"][3] = False
    arrays["pose_residual_ns"][3] = 99_000_000
    row = annotation_metrics(metrics_bundle, {})
    assert row["event_helper_pose_gap_rejected_count"] == 1
    assert row["event_pose_rejected_count"] == 1
    assert row["event_helper_pose_coverage"] == 0.8
    assert row["event_max_accepted_residual_ms"] == 0.0003
    assert row["public_recommendation"] == "exclude_from_public_demo"


def test_context_tracking_gaps_are_explicit_even_when_event_is_complete(metrics_bundle):
    arrays = metrics_bundle.aligned_arrays["leader"]
    arrays["hand_high_confidence"][[0, 1, 7, 8, 9], 1] = False
    arrays["hand_confidence"][[0, 1, 7, 8, 9], 1] = 0
    row = annotation_metrics(metrics_bundle, {})
    assert row["event_all_four_hands_coverage"] == 1
    assert row["context_all_four_hands_coverage"] == 0.5
    assert not row["event_any_missing_or_tracking_qc"]
    assert row["context_any_missing_or_tracking_qc"]
    assert row["public_recommendation"] == "tracking_caution"


def test_projection_uses_all_event_frames_and_does_not_claim_visibility(metrics_bundle):
    report = projection(metrics_bundle)
    report["handovers"][0]["cameras"]["helper"]["metrics"]["points"]["helper/left/wrist"][
        "in_image_count"
    ] = 0
    row = annotation_metrics(metrics_bundle, report)
    assert row["event_helper_camera_wrist_projection_fraction"] == 0.75
    assert row["event_leader_camera_wrist_projection_fraction"] == 1
    assert row["event_min_camera_wrist_projection_fraction"] == 0.75
    assert row["context_min_camera_wrist_projection_fraction"] is None
    assert not row["context_projection_available"]
    assert "no occlusion" in row["projection_scope"]


@pytest.mark.parametrize("mutation", ["generation", "recording", "bounds", "camera"])
def test_projection_identity_and_boundaries_must_match(metrics_bundle, mutation):
    report = projection(metrics_bundle)
    if mutation == "generation":
        report["generation_id"] = "stale"
    elif mutation == "recording":
        report["recording_id"] = "other"
    elif mutation == "bounds":
        report["handovers"][0]["start_frame"] = 1
    else:
        report["handovers"][0]["cameras"]["helper"]["metrics"]["camera_frame"] = "leader/camera-rgb"
    with pytest.raises(ValueError):
        annotation_metrics(metrics_bundle, report)


def test_motion_outliers_are_reported_in_local_hand_time(metrics_bundle):
    arrays = metrics_bundle.aligned_arrays["helper"]
    arrays["hand_points_shared_m"][4, 0, 5, 0] = 2
    row = annotation_metrics(metrics_bundle, {})
    assert row["event_helper_wrist_speed_outlier_count"] == 2
    assert row["public_recommendation"] == "tracking_caution"


def test_worst_shown_scope_ranking_does_not_hide_poor_context(metrics_bundle):
    base = annotation_metrics(metrics_bundle, projection(metrics_bundle))
    complete_event_poor_context = dict(
        base,
        annotation_id="000003",
        event_all_four_hands_coverage=1.0,
        context_all_four_hands_coverage=0.59,
    )
    better_context = dict(
        base,
        annotation_id="000004",
        event_all_four_hands_coverage=0.96,
        context_all_four_hands_coverage=0.89,
    )
    original = deepcopy([complete_event_poor_context, better_context])
    ranked = rank_examples([complete_event_poor_context, better_context])
    assert [row["annotation_id"] for row in ranked] == ["000004", "000003"]
    assert [row["objective_rank"] for row in ranked] == [1, 2]
    assert [complete_event_poor_context, better_context] == original
    assert "worst shown scope" in ranked[0]["objective_ranking_rule"]


def test_ranking_ties_use_exact_source_key_and_reject_duplicate_ids(metrics_bundle):
    base = annotation_metrics(metrics_bundle, {})
    ranked = rank_examples([dict(base, annotation_id="000010"), dict(base, annotation_id="000002")])
    assert [row["annotation_id"] for row in ranked] == ["000002", "000010"]
    with pytest.raises(ValueError, match="distinct"):
        rank_examples([base, base])


def demo_bundle(bundle):
    for arrays in bundle.aligned_arrays.values():
        arrays["hand_confidence"][:] = 0.95
    return bundle


def test_demo_any_hand_coverage_is_union_not_maximum_individual_coverage(metrics_bundle):
    bundle = demo_bundle(metrics_bundle)
    for arrays in bundle.aligned_arrays.values():
        even = np.arange(10) % 2 == 0
        arrays["hand_high_confidence"][:, 0] = even
        arrays["hand_high_confidence"][:, 1] = ~even
        arrays["hand_confidence"][~arrays["hand_high_confidence"]] = 0.1
    row = assess_demo_qc(bundle, "000002", {}, context_frames=120)
    assert row["event_helper_left_hand_coverage"] == 0.6
    assert row["event_helper_right_hand_coverage"] == 0.4
    assert row["event_helper_any_hand_coverage"] == 1
    assert row["event_leader_any_hand_coverage"] == 1
    assert row["event_all_four_hands_coverage"] == 0
    assert row["demo_qc_pass"]
    assert not row["strict_qc_pass"]
    assert row["public_recommendation"] == "exclude_from_public_demo"
    assert "No interacting-hand identity" in row["demo_qc_hand_proxy"]


def test_demo_pooled_confidence_does_not_add_a_per_role_threshold(metrics_bundle):
    bundle = demo_bundle(metrics_bundle)
    bundle.aligned_arrays["helper"]["hand_confidence"][:] = 0.85
    bundle.aligned_arrays["leader"]["hand_confidence"][:] = 0.99
    row = assess_demo_qc(bundle, "000002", {})
    assert row["event_helper_median_accepted_hand_confidence"] == 0.85
    assert row["event_median_accepted_hand_confidence"] == pytest.approx(0.92)
    assert row["demo_qc_pass"]


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("event_helper_pose_coverage", 0.95, True),
        ("event_helper_pose_coverage", 0.949, False),
        ("context_leader_pose_coverage", 0.95, True),
        ("context_leader_pose_coverage", 0.949, False),
        ("event_helper_any_hand_coverage", 0.80, True),
        ("event_leader_any_hand_coverage", 0.799, False),
        ("event_median_accepted_hand_confidence", 0.90, True),
        ("event_median_accepted_hand_confidence", 0.899, False),
        ("event_median_accepted_hand_confidence", None, False),
        ("event_mapping_verified", False, False),
        ("context_mapping_verified", False, False),
        ("event_wrist_speed_outlier_count", 1, False),
        ("context_wrist_speed_outlier_count", 1, False),
    ],
)
def test_demo_profile_thresholds_are_explicit_and_inclusive(metrics_bundle, field, value, expected):
    row = annotation_metrics(demo_bundle(metrics_bundle), {})
    row[field] = value
    result = demo_qc_assessment(row)
    assert result["demo_qc_pass"] is expected
    assert result["demo_qc_maximum_wrist_speed_m_s"] == 5
    assert result["demo_qc_continuity_max_gap_seconds"] == 0.1


def test_demo_rejects_unresolved_context_without_creating_playback(metrics_bundle, monkeypatch):
    bundle = demo_bundle(metrics_bundle)
    original = bundle.frame_maps["helper"]
    status = original.status.copy()
    indices = original.vrs_rgb_frame_index.copy()
    timestamps = original.vrs_device_timestamp_ns.copy()
    status[0], indices[0], timestamps[0] = "UNRESOLVED", -1, -1
    bundle.frame_maps["helper"] = replace(
        original, status=status, vrs_rgb_frame_index=indices, vrs_device_timestamp_ns=timestamps
    )
    arrays = bundle.aligned_arrays["helper"]
    arrays["mapping_verified"][0] = False
    for key in (
        "pose_accepted",
        "hand_accepted",
        "hand_pose_accepted",
        "hand_present",
        "hand_high_confidence",
    ):
        arrays[key][0] = False
    arrays["hand_points_shared_m"][0] = np.nan
    arrays["hand_confidence"][0] = -1

    def forbidden(*args, **kwargs):
        raise AssertionError("diagnostic metrics must not make a playable selection")

    monkeypatch.setattr(PlaybackBundle, "with_annotation", forbidden)
    monkeypatch.setattr(PlaybackBundle, "iter_frames", forbidden)
    row = assess_demo_qc(bundle, "000002", {})
    assert row["event_mapping_verified"]
    assert not row["context_mapping_verified"]
    assert not row["demo_qc_pass"]
    assert "context mappings" in row["demo_qc_reason"]
    assert not row["demo_qc_assessment_grants_playback"]
    assert bundle.start_index == 0 and bundle.end_index == 9


def test_demo_severe_wrist_speed_rejects_while_strict_caution_still_passes(metrics_bundle):
    bundle = demo_bundle(metrics_bundle)
    bundle.aligned_arrays["helper"]["hand_points_shared_m"][4, 0, 5, 0] = 2
    row = assess_demo_qc(bundle, "000002", {})
    assert row["event_wrist_speed_outlier_count"] == 2
    assert not row["demo_qc_pass"]
    assert row["strict_qc_pass"]
    assert row["public_recommendation"] == "tracking_caution"
    assert row["strict_qc_recommendation"] == "tracking_caution"


@pytest.mark.parametrize(
    "invalid", ["hand_geometry", "pose_geometry", "rotation", "tracked_confidence"]
)
def test_demo_invalid_accepted_geometry_and_tracking_masks_fail_closed(metrics_bundle, invalid):
    bundle = demo_bundle(metrics_bundle)
    arrays = bundle.aligned_arrays["helper"]
    if invalid == "hand_geometry":
        arrays["hand_points_shared_m"][3, 0, 5, 0] = np.nan
    elif invalid == "pose_geometry":
        arrays["pose_translation_m"][3, 0] = np.inf
    elif invalid == "rotation":
        arrays["pose_quaternion_xyzw"] = np.tile([0, 0, 0, 2.0], (10, 1))
    else:
        arrays["hand_confidence"][3, 0] = 0.49
    with pytest.raises(ValueError):
        assess_demo_qc(bundle, "000002", {})


def test_demo_assessment_does_not_change_strict_recommendation_or_ranking(metrics_bundle):
    bundle = demo_bundle(metrics_bundle)
    before = annotation_metrics(bundle, {})
    assessed = assess_demo_qc(bundle, "000002", {})
    after = annotation_metrics(bundle, {})
    assert before == after
    assert assessed["public_recommendation"] == before["public_recommendation"]
    assert assessed["public_recommendation_reason"] == before["public_recommendation_reason"]
    assert (
        rank_examples([before])[0]["objective_rank"]
        == rank_examples([assessed])[0]["objective_rank"]
    )
    assert all(
        value is None or isinstance(value, (str, int, float, bool)) for value in assessed.values()
    )
    json.dumps(assessed, allow_nan=False)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("event_helper_pose_coverage", np.nan),
        ("event_leader_any_hand_coverage", np.inf),
        ("context_helper_pose_coverage", True),
        ("event_helper_any_hand_coverage", 1.01),
        ("event_mapping_verified", "VERIFIED"),
        ("context_mapping_verified", 1),
        ("context_wrist_speed_outlier_count", -1),
        ("event_wrist_speed_outlier_count", 0.0),
        ("event_median_accepted_hand_confidence", np.nan),
        ("public_recommendation", "unknown"),
        ("wrist_speed_threshold_m_s", 100.0),
        ("hand_confidence_threshold", 0.49),
    ],
)
def test_demo_profile_rejects_malformed_scalar_metrics(metrics_bundle, field, value):
    row = annotation_metrics(demo_bundle(metrics_bundle), {})
    row[field] = value
    with pytest.raises(ValueError):
        demo_qc_assessment(row)
