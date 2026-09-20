"""Schema invariants independent of any source dataset."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import numpy as np
import pytest

from duet.adapters.comind.annotations import parse_handover_segments
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import (
    Confidence,
    DistanceUnit,
    FrameId,
    ParticipantId,
    Provenance,
    SampleMetadata,
    SampleState,
)
from duet.schemas.episode import CameraSample, Episode, FrameIdentifier, HandSample
from duet.schemas.time import (
    ClockDomain,
    Timestamp,
    TimeUnit,
    VerifiedClockMapping,
    comparable_seconds,
)

SOURCE = Provenance("synthetic:test_schemas")
CLOCK = ClockDomain("synthetic:clock")
WORLD = FrameId("synthetic:world")
CAMERA = FrameId("synthetic:person_a:camera")
PERSON = ParticipantId("person_a")
TIME = Timestamp(0, TimeUnit.SECONDS, CLOCK, SOURCE)


def test_preserves_large_raw_nanoseconds_and_decimal_precision() -> None:
    first = Timestamp(1_900_000_000_000_000_001, TimeUnit.NANOSECONDS, CLOCK, SOURCE)
    second = Timestamp(1_900_000_000_000_000_002, TimeUnit.NANOSECONDS, CLOCK, SOURCE)
    assert second.seconds - first.seconds == Fraction(1, 1_000_000_000)
    raw = Decimal("0.123456789012345678901234567890")
    timestamp = Timestamp(raw, TimeUnit.SECONDS, CLOCK, SOURCE)
    assert timestamp.raw_value is raw
    assert timestamp.seconds == Fraction(raw)


@pytest.mark.parametrize("raw", [float("nan"), float("inf"), Decimal("NaN"), True, "1"])
def test_rejects_invalid_timestamp_values(raw: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Timestamp(raw, TimeUnit.SECONDS, CLOCK, SOURCE)


def test_missing_timestamp_keeps_clock_and_requires_mapping() -> None:
    missing = Timestamp(None, TimeUnit.MILLISECONDS, CLOCK, SOURCE)
    assert missing.seconds is None
    with pytest.raises(ValueError, match="verified mapping"):
        comparable_seconds(missing, ClockDomain("other"))
    with pytest.raises(TypeError):
        _ = TIME < Timestamp(1, TimeUnit.SECONDS, CLOCK, SOURCE)


def test_verified_affine_mapping_preserves_original_and_does_not_infer_inverse() -> None:
    target = ClockDomain("synthetic:reference")
    mapping = VerifiedClockMapping(CLOCK, target, Fraction(1001, 1000), 10, SOURCE, "fixture")
    timestamp = Timestamp(2000, TimeUnit.MILLISECONDS, CLOCK, SOURCE)
    assert comparable_seconds(timestamp, target, [mapping]) == Fraction(6001, 500)
    assert timestamp.raw_value == 2000
    assert timestamp.clock_domain == CLOCK
    with pytest.raises(ValueError, match="verified mapping"):
        comparable_seconds(Timestamp(12, TimeUnit.SECONDS, target, SOURCE), CLOCK, [mapping])
    with pytest.raises(ValueError, match="exactly one"):
        comparable_seconds(timestamp, target, [mapping, mapping])


@pytest.mark.parametrize(
    "scale, offset, evidence",
    [(0, 0, "test"), (-1, 0, "test"), (1, float("inf"), "test"), (1, 0, "")],
)
def test_mapping_requires_valid_parameters_and_evidence(scale, offset, evidence) -> None:
    with pytest.raises(ValueError):
        VerifiedClockMapping(CLOCK, ClockDomain("other"), scale, offset, SOURCE, evidence)


def test_missing_payloads_require_explicit_state_and_reason() -> None:
    with pytest.raises(ValueError, match="state"):
        CameraSample(PERSON, CAMERA, TIME, None, SOURCE)
    with pytest.raises(ValueError, match="reason"):
        SampleMetadata(SampleState.MISSING)
    missing = CameraSample(
        PERSON,
        CAMERA,
        TIME,
        None,
        SOURCE,
        metadata=SampleMetadata(SampleState.MISSING, reason="source pose unavailable"),
    )
    assert missing.metadata.confidence is None
    confidence = Confidence(0, SOURCE)
    assert confidence.value == 0  # Zero confidence and absent confidence differ.
    with pytest.raises(ValueError):
        Confidence(1.1, SOURCE)


def test_hand_shape_units_and_immutable_coordinates() -> None:
    points = np.array([[1, 2, 3]], dtype=float)
    hand = HandSample(
        PERSON,
        "left",
        TIME,
        WORLD,
        points,
        DistanceUnit.METERS,
        SOURCE,
        landmark_names=("synthetic_tip",),
    )
    points[0, 0] = 99
    assert hand.points[0, 0] == 1
    with pytest.raises(ValueError):
        hand.points.setflags(write=True)
    for invalid in (
        np.zeros((3,)),
        np.zeros((0, 3)),
        [[1, 2, float("nan")]],
        [[1 + 2j, 2, 3]],
    ):
        with pytest.raises(ValueError):
            HandSample(PERSON, "left", TIME, WORLD, invalid, DistanceUnit.METERS, SOURCE)
    with pytest.raises(ValueError, match="METERS"):
        HandSample(PERSON, "left", TIME, WORLD, [[1, 2, 3]], DistanceUnit.MILLIMETERS, SOURCE)


def test_episode_rejects_undeclared_participant_and_nonshared_geometry() -> None:
    transform = RigidTransform(np.eye(4), CAMERA, WORLD, DistanceUnit.METERS, SOURCE)
    camera = CameraSample(PERSON, CAMERA, TIME, transform, SOURCE)
    with pytest.raises(ValueError, match="declared"):
        Episode("test", (ParticipantId("other"),), WORLD, SOURCE, camera_samples=(camera,))
    with pytest.raises(ValueError, match="shared frame"):
        Episode("test", (PERSON,), FrameId("other_world"), SOURCE, camera_samples=(camera,))
    with pytest.raises(ValueError, match="source"):
        CameraSample(PERSON, FrameId("different_camera"), TIME, transform, SOURCE)


def test_episode_does_not_claim_clocks_are_synchronized() -> None:
    transform = RigidTransform(np.eye(4), CAMERA, WORLD, DistanceUnit.METERS, SOURCE)
    camera = CameraSample(PERSON, CAMERA, TIME, transform, SOURCE)
    other = CameraSample(
        PERSON,
        CAMERA,
        Timestamp(100, TimeUnit.SECONDS, ClockDomain("other"), SOURCE),
        transform,
        SOURCE,
    )
    episode = Episode("test", (PERSON,), WORLD, SOURCE, camera_samples=(camera, other))
    assert len({sample.timestamp.clock_domain for sample in episode.camera_samples}) == 2


def test_frame_identifier_is_distinct_from_time_and_coordinate_frame() -> None:
    frame = FrameIdentifier("synthetic:video", index=123, source_id="original-frame-name")
    assert frame.index == 123
    with pytest.raises(ValueError):
        FrameIdentifier("synthetic:video")
    with pytest.raises(ValueError):
        FrameIdentifier("synthetic:video", index=True)


def test_annotation_json_is_detached_from_source_buffers() -> None:
    segment = {"skip": False, "initiation_type": ["gestural"], "bbox": [1, 2, 3, 4]}
    (annotation,) = parse_handover_segments(
        [segment],
        recording_uuid="00000000-0000-0000-0000-000000000001",
        provenance=SOURCE,
    )
    segment["initiation_type"].append("changed")
    segment["bbox"][0] = 99
    assert annotation.initiation_type == ["gestural"]
    assert annotation.source_fields["bbox"] == [1, 2, 3, 4]
    assert annotation.start_time.raw_value is None
    with pytest.raises(ValueError, match="record is present"):
        replace(annotation, metadata=SampleMetadata(SampleState.MISSING, reason="absent record"))
