"""Synthetic paired-video indexing without comparing participants' device clocks."""

from fractions import Fraction

import pytest

from duet.adapters.comind.frame_timeline import PairedFrameTimeline
from duet.adapters.comind.mp4_timestamps import Mp4DeviceTimestamps
from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.schemas.common import Provenance
from duet.schemas.time import Timestamp, TimeUnit
from duet.synchronization.timeline import order_timestamps

PROVENANCE = Provenance("synthetic exact exported-frame timestamps")


def stream(role, values, recording="synthetic"):
    return Mp4DeviceTimestamps(recording, role, values, len(values), PROVENANCE)


def evidence(status=EvidenceStatus.VERIFIED):
    return MappingEvidence(
        "paired_video_frame_alignment",
        status,
        "Same indices represent synchronized moments",
        ("synthetic fixture paired-capture contract",),
    )


def paired():
    return PairedFrameTimeline(
        stream("helper", [1_000_000, 2_000_000, 3_000_000]),
        stream("leader", [81_000_000, 82_000_000, 83_000_000]),
        evidence(),
    )


def local(role, raw_ns):
    clock = device_clock("synthetic", role)
    return order_timestamps(
        [Timestamp(value, TimeUnit.NANOSECONDS, clock, PROVENANCE) for value in raw_ns],
        clock_domain=clock,
    )


def test_common_index_preserves_two_distinct_local_device_times():
    frame = paired().frame(1)
    assert frame.timeline_name == "comind/synthetic/comind_sync_frame_index"
    assert frame.frame_index == 1
    assert frame.device_timestamps["helper"].raw_value == 2_000_000
    assert frame.device_timestamps["leader"].raw_value == 82_000_000
    assert (
        frame.device_timestamps["helper"].clock_domain
        != frame.device_timestamps["leader"].clock_domain
    )
    with pytest.raises(TypeError):
        frame.device_timestamps["helper"] = frame.device_timestamps["leader"]


def test_local_lookup_uses_capture_time_not_video_index_and_reports_residual():
    result = paired().match_local(
        1, "leader", local("leader", [82_000_123]), max_gap_seconds=Fraction(123, 10**9)
    )
    assert result.accepted
    assert result.matched_index == 0
    assert result.signed_residual_seconds == Fraction(123, 10**9)
    rejected = paired().match_local(1, "leader", local("leader", [82_000_123]), max_gap_seconds=0)
    assert not rejected.accepted
    assert rejected.matched_index is None


def test_same_numeric_values_cannot_match_the_wrong_participant_clock():
    with pytest.raises(ValueError, match="same participant"):
        paired().match_local(1, "helper", local("leader", [2_000_000]), max_gap_seconds=1)


def test_two_participant_frame_assembly_keeps_independent_missing_streams():
    timeline = paired()
    matches = {
        "helper": timeline.match_local(
            2, "helper", local("helper", [3_000_000]), max_gap_seconds=0
        ),
        "leader": timeline.match_local(2, "leader", local("leader", []), max_gap_seconds=0),
    }
    assert matches["helper"].accepted
    assert not matches["leader"].accepted
    assert timeline.frame_count == 3


def test_duplicate_capture_timestamps_remain_separate_video_frames():
    timeline = PairedFrameTimeline(
        stream("helper", [10, 10, 20]), stream("leader", [30, 40, 40]), evidence()
    )
    assert timeline.frame(0).frame_index != timeline.frame(1).frame_index
    assert (
        timeline.frame(0).device_timestamps["helper"].raw_value
        == timeline.frame(1).device_timestamps["helper"].raw_value
    )


@pytest.mark.parametrize("status", [EvidenceStatus.INFERRED, EvidenceStatus.UNRESOLVED])
def test_equal_video_counts_cannot_replace_alignment_evidence(status):
    with pytest.raises(ValueError):
        PairedFrameTimeline(stream("helper", [1]), stream("leader", [2]), evidence(status))


def test_verified_evidence_for_an_unrelated_claim_does_not_enable_pairing():
    unrelated = MappingEvidence(
        "native_rgb_extrinsics",
        EvidenceStatus.VERIFIED,
        "Native camera pose is known",
        ("synthetic calibration source",),
    )
    with pytest.raises(ValueError, match="specifically establish"):
        PairedFrameTimeline(stream("helper", [1]), stream("leader", [2]), unrelated)


@pytest.mark.parametrize(
    "helper,leader",
    [
        (stream("helper", [1, 2]), stream("leader", [3])),
        (stream("leader", [1]), stream("helper", [2])),
        (stream("helper", [1]), stream("leader", [2], "other")),
    ],
)
def test_bad_pairing_rejected(helper, leader):
    with pytest.raises(ValueError):
        PairedFrameTimeline(helper, leader, evidence())


@pytest.mark.parametrize("index", [-1, 3, True, 1.5])
def test_frame_index_must_be_an_in_range_integer(index):
    with pytest.raises((ValueError, TypeError, IndexError)):
        paired().frame(index)
