"""Synthetic shared-space chains keep participant clocks and source frames explicit."""

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from duet.adapters.comind.mps import (
    CameraCalibration,
    CameraCalibrationSample,
    MpsTrajectorySample,
    device_clock,
)
from duet.adapters.comind.multislam import SlamSource, verify_shared_world
from duet.adapters.comind.shared_world import camera_to_shared_world, hand_to_shared_world
from duet.geometry.transforms import RigidTransform
from duet.qc.result import QCStatus
from duet.schemas.common import (
    Confidence,
    DistanceUnit,
    FrameId,
    ParticipantId,
    Provenance,
    SampleMetadata,
    SampleState,
)
from duet.schemas.episode import Episode, HandSample
from duet.schemas.time import Timestamp, TimeUnit
from duet.synchronization.matching import MatchStatus

RECORDING_ID = "00000000-0000-0000-0000-000000000001"
PROVENANCE = Provenance("synthetic shared-world fixture")


@pytest.fixture
def verification(tmp_path):
    sources = {}
    for role in ("helper", "leader"):
        directory = tmp_path / role
        directory.mkdir()
        (directory / "closed_loop_trajectory.csv").write_text(
            "graph_uid\nsynthetic-shared\n", encoding="utf-8"
        )
        sources[role] = SlamSource(directory=directory)
    return verify_shared_world(sources, recording_id=RECORDING_ID)


def _timestamp(role="helper", value=1_005_000):
    return Timestamp(value, TimeUnit.MICROSECONDS, device_clock(RECORDING_ID, role), PROVENANCE)


def _transform(matrix, source, destination):
    return RigidTransform(matrix, FrameId(source), destination, DistanceUnit.METERS, PROVENANCE)


def _pose(verification, role="helper", value=1_000_000):
    matrix = np.array([[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]], dtype=float)
    if role == "leader":
        matrix[:3, 3] = [-10, -20, -30]
    return MpsTrajectorySample(
        timestamp=_timestamp(role, value),
        utc_timestamp=None,
        utc_timestamp_ns_raw=None,
        graph_uid=verification.graph_uid,
        quality_score=0.8,
        transform=_transform(matrix, f"{role}/device", verification.world_frame),
        participant_id=ParticipantId(f"participant/{role}"),
        provenance=PROVENANCE,
    )


def _hand(role="helper", *, missing=False, confidence=0.7):
    return HandSample(
        participant_id=ParticipantId(f"participant/{role}"),
        hand_id="left",
        timestamp=_timestamp(role),
        frame=FrameId(f"{role}/device"),
        points=None if missing else np.tile([1.0, 2.0, 3.0], (21, 1)),
        unit=DistanceUnit.METERS,
        provenance=PROVENANCE,
        metadata=(
            SampleMetadata(SampleState.MISSING, reason="source confidence -1")
            if missing
            else SampleMetadata(confidence=Confidence(confidence, PROVENANCE))
        ),
    )


def _calibration(role="helper", *, calibrated=True):
    # Rotate 90 degrees about device X and translate; this must be composed on
    # the right of the world-device pose, not inverted or commuted.
    matrix = [[1, 0, 0, 1], [0, 0, -1, 2], [0, 1, 0, 3], [0, 0, 0, 1]]
    camera = CameraCalibration(
        label="camera-rgb",
        transform=_transform(matrix, f"{role}/camera-rgb", FrameId(f"{role}/device")),
        projection={"Name": "FisheyeRadTanThinPrism", "Params": tuple(range(15))},
        image_size=(1408, 1408),
        calibrated=calibrated,
    )
    return CameraCalibrationSample(_timestamp(role), None, None, (camera,), PROVENANCE)


def test_two_participant_hands_share_world_without_merging_device_clocks(verification):
    transformed = []
    for role, expected in (("helper", [8, 21, 33]), ("leader", [-12, -19, -27])):
        source = _hand(role)
        hand, match = hand_to_shared_world(
            source, _pose(verification, role), verification=verification, max_gap_seconds=0.01
        )
        assert hand.frame == verification.world_frame
        np.testing.assert_array_equal(hand.points, np.tile(expected, (21, 1)))
        assert hand.timestamp == source.timestamp
        assert match.signed_residual_seconds == Fraction(-5, 1000)
        assert hand.metadata.confidence == source.metadata.confidence
        assert hand.qc[-1].status is QCStatus.PASS
        transformed.append(hand)
    assert transformed[0].timestamp.clock_domain != transformed[1].timestamp.clock_domain
    episode = Episode(
        "synthetic",
        tuple(hand.participant_id for hand in transformed),
        verification.world_frame,
        PROVENANCE,
        hand_samples=tuple(transformed),
    )
    assert len(episode.hand_samples) == 2


@pytest.mark.parametrize("change", ["standard_world", "different_graph", "wrong_device"])
def test_hand_chain_rejects_unverified_pose_space(verification, change):
    pose = _pose(verification)
    if change == "different_graph":
        pose = replace(pose, graph_uid="not-the-shared-graph")
    else:
        transform = pose.transform
        pose = replace(
            pose,
            transform=_transform(
                transform.matrix,
                "leader/device" if change == "wrong_device" else "helper/device",
                FrameId("helper/mps_world/standard")
                if change == "standard_world"
                else verification.world_frame,
            ),
        )
    with pytest.raises(ValueError, match="world graph|T_shared_world_device"):
        hand_to_shared_world(_hand(), pose, verification=verification, max_gap_seconds=0.01)


def test_hand_chain_rejects_wrong_pose_participant(verification):
    with pytest.raises(ValueError, match="different participant"):
        hand_to_shared_world(
            _hand(),
            _pose(verification, "leader"),
            verification=verification,
            max_gap_seconds=0.01,
        )


@pytest.mark.parametrize("wrong_clock", ["hand", "pose"])
def test_hand_chain_rejects_cross_device_time_even_when_values_equal(verification, wrong_clock):
    hand, pose = _hand(), _pose(verification)
    if wrong_clock == "hand":
        hand = replace(hand, timestamp=_timestamp("leader"))
    else:
        pose = replace(pose, timestamp=_timestamp("leader", 1_000_000))
    with pytest.raises(ValueError, match="device clock|clock domains"):
        hand_to_shared_world(hand, pose, verification=verification, max_gap_seconds=0.01)


@pytest.mark.parametrize("frame", ["leader/device", "helper/camera-rgb"])
def test_hand_chain_requires_participant_device_frame(verification, frame):
    with pytest.raises(ValueError, match="device frame"):
        hand_to_shared_world(
            replace(_hand(), frame=FrameId(frame)),
            _pose(verification),
            verification=verification,
            max_gap_seconds=0.01,
        )


def test_hand_pose_gap_yields_absent_world_points_and_retained_residual(verification):
    result, match = hand_to_shared_world(
        _hand(), _pose(verification), verification=verification, max_gap_seconds=0.001
    )
    assert result.points is None
    assert result.metadata.state is SampleState.UNKNOWN
    assert "gap_exceeded" in result.metadata.reason
    assert match.status is MatchStatus.GAP_EXCEEDED
    assert match.signed_residual_seconds == Fraction(-5, 1000)
    assert result.qc[-1].status is QCStatus.FAIL


def test_hand_without_pose_is_explicitly_unknown(verification):
    result, match = hand_to_shared_world(
        _hand(), None, verification=verification, max_gap_seconds=0.01
    )
    assert result.points is None
    assert result.metadata.state is SampleState.UNKNOWN
    assert match.status is MatchStatus.NO_SAMPLES


def test_missing_hand_stays_missing_and_has_no_zero_placeholder(verification):
    source = _hand(missing=True)
    result, match = hand_to_shared_world(
        source, _pose(verification), verification=verification, max_gap_seconds=0.01
    )
    assert match.accepted
    assert result.points is None
    assert result.metadata == source.metadata
    assert result.frame == verification.world_frame


def test_zero_confidence_hand_is_transformed_and_preserved(verification):
    result, match = hand_to_shared_world(
        _hand(confidence=0), _pose(verification), verification=verification, max_gap_seconds=0.01
    )
    assert match.accepted
    assert result.points is not None
    assert result.metadata.state is SampleState.PRESENT
    assert result.metadata.confidence.value == 0


def test_camera_chain_composes_nontrivial_rotation_translation_in_correct_order(verification):
    calibration = _calibration()
    result, match = camera_to_shared_world(
        calibration,
        _pose(verification),
        camera_label="camera-rgb",
        participant="helper",
        verification=verification,
        max_gap_seconds=0.01,
    )
    assert match.accepted
    assert result.camera_frame == FrameId("helper/camera-rgb")
    assert result.transform.destination == verification.world_frame
    assert result.timestamp == calibration.timestamp
    np.testing.assert_array_equal(
        result.transform.matrix,
        [[0, 0, 1, 8], [1, 0, 0, 21], [0, 1, 0, 33], [0, 0, 0, 1]],
    )


@pytest.mark.parametrize("reason", ["uncalibrated", "missing_pose", "gap"])
def test_unavailable_camera_pose_is_explicitly_unknown(verification, reason):
    calibration = _calibration(calibrated=reason != "uncalibrated")
    result, match = camera_to_shared_world(
        calibration,
        None if reason == "missing_pose" else _pose(verification),
        camera_label="camera-rgb",
        participant="helper",
        verification=verification,
        max_gap_seconds=0.001 if reason == "gap" else 0.01,
    )
    assert result.transform is None
    assert result.metadata.state is SampleState.UNKNOWN
    assert result.metadata.reason
    if reason == "gap":
        assert match.signed_residual_seconds == Fraction(-5, 1000)


def test_camera_label_must_be_found_in_inspected_calibration(verification):
    with pytest.raises(ValueError, match="exactly one calibration"):
        camera_to_shared_world(
            _calibration(),
            _pose(verification),
            camera_label="guessed-label",
            participant="helper",
            verification=verification,
            max_gap_seconds=0.01,
        )


@pytest.mark.parametrize("source", ["leader/camera-rgb", "helper/camera-slam-left"])
def test_camera_source_frame_must_agree_with_label_and_participant(verification, source):
    calibration = _calibration()
    camera = calibration.cameras[0]
    wrong = replace(
        camera, transform=_transform(camera.transform.matrix, source, FrameId("helper/device"))
    )
    with pytest.raises(ValueError, match="source frame"):
        camera_to_shared_world(
            replace(calibration, cameras=(wrong,)),
            _pose(verification),
            camera_label="camera-rgb",
            participant="helper",
            verification=verification,
            max_gap_seconds=0.01,
        )


def test_camera_destination_frame_must_agree_with_participant(verification):
    calibration = _calibration()
    camera = calibration.cameras[0]
    wrong = replace(
        camera,
        transform=_transform(
            camera.transform.matrix, "helper/camera-rgb", FrameId("leader/device")
        ),
    )
    with pytest.raises(ValueError, match="same participant"):
        camera_to_shared_world(
            replace(calibration, cameras=(wrong,)),
            _pose(verification),
            camera_label="camera-rgb",
            participant="helper",
            verification=verification,
            max_gap_seconds=0.01,
        )
