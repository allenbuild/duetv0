"""Small fixtures verify paired-index rendering without physical clock merging."""

from dataclasses import replace

import numpy as np
import pytest

from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import (
    Confidence,
    DistanceUnit,
    FrameId,
    ParticipantId,
    Provenance,
    SampleMetadata,
)
from duet.schemas.episode import HandSample
from duet.schemas.time import Timestamp, TimeUnit
from duet.visualization.comind_handover import (
    FrameIndexBindings,
    MappedParticipantFrame,
    MappedRenderFrame,
    assemble_index_frame,
    save_index_episode,
)

PROVENANCE = Provenance("synthetic verified capture correspondence")
WORLD = FrameId("comind/synthetic/multislam/shared")


def bindings():
    return FrameIndexBindings(
        "synthetic",
        WORLD,
        MappingEvidence(
            "paired_video_frame_alignment",
            EvidenceStatus.VERIFIED,
            "Fixture exports represent paired observations at each index",
            ("fixture",),
        ),
    )


def timestamp(role, ns):
    return Timestamp(ns, TimeUnit.NANOSECONDS, device_clock("synthetic", role), PROVENANCE)


def participant(role):
    ns = 1_000_000 if role == "helper" else 901_000_000
    stamp = timestamp(role, ns)
    pose = RigidTransform(
        np.eye(4), FrameId(f"{role}/device"), WORLD, DistanceUnit.METERS, PROVENANCE
    )
    hand = HandSample(
        ParticipantId(f"participant/{role}"),
        "left",
        stamp,
        WORLD,
        np.array([[0, 0, 1], [0, 1, 1.0]]),
        DistanceUnit.METERS,
        PROVENANCE,
        metadata=SampleMetadata(confidence=Confidence(0.9, PROVENANCE)),
    )
    return MappedParticipantFrame(
        role,
        stamp,
        7 if role == "helper" else 0,
        EvidenceStatus.VERIFIED,
        0.95,
        pose,
        stamp,
        (hand,),
    )


def frame(index=100):
    return MappedRenderFrame(index, tuple(participant(role) for role in ("helper", "leader")))


def test_index_assembly_keeps_distinct_device_clocks_and_clears_missing_hands():
    updates, status = assemble_index_frame(frame(), bindings())
    assert status["comind_sync_frame_index"] == 100
    assert status["participants"]["helper"]["device_timestamp_ns"] == 1_000_000
    assert status["participants"]["leader"]["device_timestamp_ns"] == 901_000_000
    assert {u.path for u in updates if u.kind == "clear"} >= {
        "world/helper/hands/right",
        "world/leader/hands/right",
        "world/helper/camera_rgb/frustum",
        "world/leader/camera_rgb/frustum",
    }


@pytest.mark.parametrize("state", [EvidenceStatus.INFERRED, EvidenceStatus.UNRESOLVED])
def test_inferred_mapping_cannot_enable_frame(state):
    helper = replace(participant("helper"), mapping_status=state)
    with pytest.raises(ValueError, match="VERIFIED"):
        assemble_index_frame(MappedRenderFrame(0, (helper, participant("leader"))), bindings())


def test_cross_participant_pose_clock_rejected():
    helper = replace(participant("helper"), pose_timestamp=timestamp("leader", 1_000_000))
    with pytest.raises(ValueError, match="own device clock"):
        assemble_index_frame(MappedRenderFrame(0, (helper, participant("leader"))), bindings())


def test_large_pose_gap_clears_pose():
    helper = replace(participant("helper"), pose_timestamp=timestamp("helper", 4_000_001))
    updates, status = assemble_index_frame(
        MappedRenderFrame(0, (helper, participant("leader"))), bindings()
    )
    assert next(u for u in updates if u.path == "world/helper/device").kind == "clear"
    assert status["participants"]["helper"]["pose_residual_ns"] == 3_000_001


def test_low_confidence_hand_is_reported_but_not_rendered():
    helper = participant("helper")
    low = replace(helper.hands[0], metadata=SampleMetadata(confidence=Confidence(0.1, PROVENANCE)))
    helper = replace(helper, hands=(low,))
    updates, status = assemble_index_frame(
        MappedRenderFrame(0, (helper, participant("leader"))), bindings()
    )
    assert next(u for u in updates if u.path == "world/helper/hands/left").kind == "clear"
    assert status["participants"]["helper"]["hands"]["left"]["confidence"] == 0.1


def test_duplicate_capture_can_remain_two_sync_frames():
    first = assemble_index_frame(frame(0), bindings())[1]
    second = assemble_index_frame(frame(1), bindings())[1]
    assert first["participants"] == second["participants"]
    assert first["comind_sync_frame_index"] != second["comind_sync_frame_index"]


def test_tiny_real_rrd_generation(tmp_path):
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    destination = tmp_path / "handover.rrd"
    saved = save_index_episode(
        destination,
        [frame(0), frame(1)],
        bindings=bindings(),
        video_frames={role: iter([(0, pixels), (1, pixels)]) for role in ("helper", "leader")},
    )
    assert saved == destination
    assert destination.stat().st_size > 0


def test_misindexed_video_is_rejected(tmp_path):
    pixels = np.zeros((8, 8, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="decoded MP4 index"):
        save_index_episode(
            tmp_path / "bad.rrd",
            [frame(0)],
            bindings=bindings(),
            video_frames={role: iter([(1, pixels)]) for role in ("helper", "leader")},
        )
