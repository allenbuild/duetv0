"""Small synthetic frame maps; no dataset or image dependencies."""

from dataclasses import replace

import numpy as np
import pytest

from duet.adapters.comind.annotations import parse_handover_segments
from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.adapters.comind.temporal_validation import (
    NO_RESIDUAL,
    WRIST_INDEX,
    ParticipantStreams,
    TemporalThresholds,
    bind_annotations,
    nearest_local_ns,
    rank_handovers,
    read_full_trajectory_once,
    validate_participant_frames,
)
from duet.schemas.common import Provenance

RECORDING = "43276420-701f-4731-b9ab-bebc7fd14994"
GRAPH = "synthetic-shared-world"


def evidence(name: str, status: EvidenceStatus = EvidenceStatus.VERIFIED) -> MappingEvidence:
    return MappingEvidence(name, status, "Explicit synthetic contract", ("synthetic fixture",))


def streams(role="helper", count=6, *, origin=1_000_000):
    times = origin + np.arange(count, dtype=np.int64) * 33_333
    points = np.zeros((count, 2, 21, 3))
    points[..., 0] = 1 if role == "leader" else 0
    return ParticipantStreams(
        RECORDING,
        role,
        GRAPH,
        device_clock(RECORDING, role).name,
        times,
        np.zeros((count, 3)),
        np.tile([0, 0, 0, 1.0], (count, 1)),
        times,
        points,
        np.ones((count, 2)),
        np.zeros(count, dtype=np.int64),
        np.ones(count, dtype=bool),
    )


def frame_map(source, *, delta=0):
    count = len(source.pose_device_us)
    return {
        "mp4_frame_index": np.arange(count, dtype=np.int64),
        "vrs_rgb_frame_index": np.arange(count, dtype=np.int64),
        "vrs_device_timestamp_ns": source.pose_device_us * 1000 + delta,
        "status": np.full(count, "VERIFIED", dtype="U10"),
        "match_confidence": np.full(count, 0.8),
    }


def validate(source, mapping=None, **kwargs):
    return validate_participant_frames(
        frame_map(source) if mapping is None else mapping,
        source,
        recording_id=RECORDING,
        participant=source.participant,
        **kwargs,
    )


def bound(start=1, end=3, *, count=6):
    annotations = parse_handover_segments(
        [
            {
                "start_frame": start,
                "end_frame": end,
                "start_time": start / 30,
                "end_time": end / 30,
                "skip": False,
                "initiator": "left",
                "delivering_flow": "rtl",
                "initiation_type": ["verbal"],
                "object_category_level_1": "bowl",
            }
        ],
        recording_uuid=RECORDING,
        provenance=Provenance("synthetic"),
    )
    return bind_annotations(
        annotations,
        recording_id=RECORDING,
        frame_count=count,
        binding_evidence=evidence("annotation_to_comind_sync_frame_index"),
    )


def rank(helper, leader, annotations, **kwargs):
    return rank_handovers(
        helper,
        leader,
        annotations,
        alignment_evidence=evidence("paired_video_frame_alignment"),
        **kwargs,
    )


def test_exact_integer_nearest_earlier_tie_and_first_duplicate():
    base = 2**60
    indices, residuals, accepted = nearest_local_ns(
        np.array([base, base + 10, base + 10, base + 20]),
        np.array([base + 5, base + 10, base + 15]),
        np.ones(3, dtype=bool),
        maximum_gap_ns=5,
    )
    assert indices.tolist() == [0, 1, 1]
    assert residuals.tolist() == [-5, 0, -5]
    assert accepted.all()


def test_nearest_empty_missing_and_gap_rejections_retain_residual():
    query = np.array([1, 50, -1])
    valid = np.array([True, True, False])
    indices, residuals, accepted = nearest_local_ns(np.array([1]), query, valid, maximum_gap_ns=5)
    assert indices.tolist() == [0, 0, -1]
    assert residuals.tolist() == [0, -49, NO_RESIDUAL]
    assert accepted.tolist() == [True, False, False]
    indices, residuals, accepted = nearest_local_ns(
        np.array([], dtype=np.int64), query, valid, maximum_gap_ns=5
    )
    assert (indices == -1).all() and (residuals == NO_RESIDUAL).all() and not accepted.any()


def test_every_frame_has_separate_pose_hand_residual_and_coverage():
    source = streams()
    result = validate(source, frame_map(source, delta=1_500_000))
    assert result.arrays["pose_residual_ns"].tolist() == [-1_500_000] * 6
    assert result.summary["pose"]["coverage"] == 1
    assert result.summary["hands"]["accepted_absolute_residual_ms"]["median"] == 1.5
    rejected = validate(source, frame_map(source, delta=3_000_000))
    assert not rejected.arrays["pose_accepted"].any()
    assert rejected.arrays["hand_accepted"].all()
    assert rejected.summary["pose"]["gap_rejected_count"] == 6
    assert np.isnan(rejected.arrays["pose_translation_m"]).all()


def test_inferred_and_missing_never_supply_pose_or_hands():
    source = streams()
    mapping = frame_map(source)
    mapping["status"][1] = "INFERRED"
    mapping["status"][2] = "UNRESOLVED"
    mapping["vrs_rgb_frame_index"][2] = -1
    mapping["vrs_device_timestamp_ns"][2] = -1
    result = validate(source, mapping)
    assert result.summary["mapping_counts"] == {"VERIFIED": 4, "INFERRED": 1, "UNRESOLVED": 1}
    assert result.summary["pose"]["coverage"] == 4 / 6
    assert np.isnan(result.arrays["hand_points_shared_m"][1:3]).all()
    assert (result.arrays["hand_residual_ns"][1:3] == NO_RESIDUAL).all()


def test_zero_confidence_is_present_but_not_high_confidence_and_missing_stays_missing():
    source = streams()
    confidence = source.hand_confidence.copy()
    confidence[0, 0] = 0
    confidence[1, 1] = -1
    points = source.hand_points_shared_m.copy()
    points[1, 1] = np.nan
    result = validate(replace(source, hand_confidence=confidence, hand_points_shared_m=points))
    assert result.arrays["hand_present"][0, 0]
    assert not result.arrays["hand_high_confidence"][0, 0]
    assert result.arrays["hand_confidence"][0, 0] == 0
    assert not result.arrays["hand_present"][1, 1]
    assert np.isnan(result.arrays["hand_points_shared_m"][1, 1]).all()


def test_secondary_hand_pose_gap_rejects_geometry_without_hiding_video_hand_residual():
    source = streams()
    residual = source.hand_pose_residual_us.copy()
    residual[2] = 10_001
    result = validate(replace(source, hand_pose_residual_us=residual))
    assert result.arrays["hand_accepted"][2]
    assert result.arrays["hand_residual_ns"][2] == 0
    assert not result.arrays["hand_pose_accepted"][2]
    assert not result.arrays["hand_present"][2].any()
    assert result.summary["hand_pose_secondary_gap_or_source_rejected_count"] == 1


def test_empty_streams_return_missing_output():
    source = streams(count=0)
    mapping = frame_map(streams(count=3))
    result = validate(source, mapping)
    assert result.summary["pose"]["coverage"] == 0
    assert result.summary["hands"]["missing_query_or_empty_stream_count"] == 3
    assert np.isnan(result.arrays["hand_points_shared_m"]).all()


def test_wrong_participant_clock_recording_and_graph_are_rejected():
    source = streams()
    with pytest.raises(ValueError, match="own device clock"):
        replace(source, device_clock_name=device_clock(RECORDING, "leader").name)
    with pytest.raises(ValueError, match="same participant"):
        validate_participant_frames(
            frame_map(source), source, recording_id=RECORDING, participant="leader"
        )
    with pytest.raises(ValueError, match="shared world"):
        rank(validate(source), validate(replace(streams("leader"), graph_uid="other")), bound())


@pytest.mark.parametrize(
    "mutation", ["bad_status", "unresolved_timestamp", "decreasing", "bad_confidence"]
)
def test_invalid_map_rows_rejected(mutation):
    source = streams()
    mapping = frame_map(source)
    if mutation == "bad_status":
        mapping["status"][0] = "LIKELY"
    elif mutation == "unresolved_timestamp":
        mapping["status"][0] = "UNRESOLVED"
    elif mutation == "decreasing":
        mapping["vrs_device_timestamp_ns"][1] = 0
    else:
        mapping["match_confidence"][0] = np.nan
    with pytest.raises(ValueError):
        validate(source, mapping)


def test_annotation_direct_binding_requires_evidence_preserves_labels_and_bounds():
    bindings = bound()
    assert (bindings[0].start_frame, bindings[0].end_frame) == (1, 3)
    assert bindings[0].annotation.initiator == "left"
    assert bindings[0].annotation.delivering_flow == "rtl"
    with pytest.raises(ValueError, match="inferred"):
        bind_annotations(
            [bindings[0].annotation],
            recording_id=RECORDING,
            frame_count=6,
            binding_evidence=evidence(
                "annotation_to_comind_sync_frame_index", EvidenceStatus.INFERRED
            ),
        )
    with pytest.raises(ValueError, match="arithmetic"):
        bind_annotations(
            [bindings[0].annotation],
            recording_id=RECORDING,
            frame_count=2,
            binding_evidence=evidence("annotation_to_comind_sync_frame_index"),
        )


def test_two_independent_device_clocks_assemble_on_sync_frame_and_use_wrist_index_five():
    helper = streams()
    leader = streams("leader", origin=70_000_000)
    leader_points = leader.hand_points_shared_m.copy()
    leader_points[:, :, 0] = [100, 0, 0]  # thumb tip must not define wrist distance
    leader_points[:, :, WRIST_INDEX] = [0.25, 0, 0]
    leader = replace(leader, hand_points_shared_m=leader_points)
    report = rank(validate(helper), validate(leader), bound(), context_frames=1)
    assert report["selected_annotation_index"] == 0
    row = report["handovers"][0]
    assert row["context_start_frame"] == 0 and row["context_end_frame"] == 4
    assert row["nearest_cross_participant_wrist_distance_m"]["median"] == 0.25
    assert row["visibility"]["status"] == "UNAVAILABLE"
    assert not report["cross_participant_device_times_compared"]
    assert "source endpoint semantics" in report["boundary_evaluation"]


def test_ranking_requires_verified_context_and_prefers_both_participants_hands():
    helper = streams()
    leader = streams("leader")
    points = leader.hand_points_shared_m.copy()
    points[:3] = np.nan
    leader = replace(leader, hand_points_shared_m=points)
    annotations = (*bound(0, 1), *bound(4, 5))
    report = rank(validate(helper), validate(leader), annotations, context_frames=0)
    assert report["selected_annotation_index"] == 1
    mapping = frame_map(helper)
    mapping["status"][3] = "INFERRED"
    report = rank(validate(helper, mapping), validate(leader), bound(4, 5), context_frames=1)
    assert report["selected_annotation_index"] is None
    assert report["selection_status"] == "NO_VERIFIED_CONTEXT"


def test_pair_alignment_evidence_scope_is_required():
    with pytest.raises(ValueError, match="paired_video_frame_alignment"):
        rank_handovers(
            validate(streams()),
            validate(streams("leader")),
            bound(),
            alignment_evidence=evidence("some_other_fact"),
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pose_max_gap_ns": -1},
        {"hand_max_gap_ns": True},
        {"minimum_hand_confidence": 2},
        {"maximum_wrist_speed_m_s": float("nan")},
    ],
)
def test_explicit_thresholds_validate(kwargs):
    with pytest.raises(ValueError):
        TemporalThresholds(**kwargs)


def test_full_trajectory_extraction_retains_full_rate_and_rejects_wrong_graph(tmp_path):
    path = tmp_path / "trajectory.csv"
    text = (
        "tracking_timestamp_us,graph_uid,tx_world_device,ty_world_device,tz_world_device,"
        "qx_world_device,qy_world_device,qz_world_device,qw_world_device\n"
        "1000000,world,0,0,0,0,0,0,1\n1000001,world,1,2,3,0,0,0,1\n"
    )
    path.write_text(text)
    before = path.stat()
    result = read_full_trajectory_once(path, expected_graph_uid="world")
    assert result["device_us"].tolist() == [1000000, 1000001]
    assert result["translation_m"].tolist() == [[0, 0, 0], [1, 2, 3]]
    assert path.read_text() == text and path.stat().st_mtime_ns == before.st_mtime_ns
    with pytest.raises(ValueError, match="shared world graph"):
        read_full_trajectory_once(path, expected_graph_uid="another")
