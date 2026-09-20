"""Two fake participants share geometry through timestamp matches, never index alignment."""

from fractions import Fraction

import numpy as np

from duet.geometry.frame_graph import FrameGraph
from duet.geometry.transforms import RigidTransform
from duet.qc.geometry import TrajectorySample, check_trajectory_continuity, check_transform
from duet.qc.result import QCStatus
from duet.schemas.common import DistanceUnit, FrameId, ParticipantId, Provenance
from duet.schemas.episode import CameraSample, Episode, FrameIdentifier, HandSample
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit
from duet.synchronization.matching import nearest_timestamp
from duet.synchronization.timeline import timestamp_overlap


def test_two_participants_in_synthetic_shared_frame() -> None:
    provenance = Provenance("synthetic:two_participant_fixture", "All clocks and axes defined here")
    clock = ClockDomain("synthetic:shared_clock")
    world = FrameId("synthetic:world")
    participants = (ParticipantId("alice"), ParticipantId("bob"))
    cameras = tuple(FrameId(f"synthetic:{person.name}:camera") for person in participants)

    def time(value: int) -> Timestamp:
        return Timestamp(value, TimeUnit.MILLISECONDS, clock, provenance)

    # Different order, cadence, counts, and indices ensure index matching fails.
    streams = ([time(0), time(1000), time(2000)], [time(1010), time(2010), time(10), time(3010)])
    query = time(1005)
    matches = tuple(nearest_timestamp(query, stream, max_gap_seconds=0.01) for stream in streams)
    assert all(match.accepted for match in matches)
    assert tuple(match.matched_index for match in matches) == (1, 0)
    assert tuple(match.signed_residual_seconds for match in matches) == (
        Fraction(-5, 1000),
        Fraction(5, 1000),
    )
    overlap = timestamp_overlap(streams, clock_domain=clock)
    assert overlap.start_seconds == Fraction(1, 100)
    assert overlap.end_seconds == 2

    # Bob faces Alice: a proper 180-degree rotation about z, with a 2 m offset.
    matrices = (np.eye(4), np.diag([-1.0, -1.0, 1.0, 1.0]))
    matrices[1][0, 3] = 2.0
    transforms = tuple(
        RigidTransform(matrix, camera, world, DistanceUnit.METERS, provenance)
        for matrix, camera in zip(matrices, cameras)
    )
    graph = FrameGraph(transforms)
    camera_samples = []
    hand_samples = []
    for person, camera, stream, match in zip(participants, cameras, streams, matches):
        assert match.matched_index is not None
        transform = graph.get_transform(camera, world)
        timestamp = stream[match.matched_index]
        camera_samples.append(
            CameraSample(
                person,
                camera,
                timestamp,
                transform,
                provenance,
                frame_identifier=FrameIdentifier(
                    f"synthetic:{person.name}:video", match.matched_index
                ),
            )
        )
        for hand_id, local_point in (("left", [1, 0.1, 0]), ("right", [1, -0.1, 0])):
            hand_samples.append(
                HandSample(
                    person,
                    hand_id,
                    timestamp,
                    world,
                    transform.apply([local_point], unit=DistanceUnit.METERS),
                    DistanceUnit.METERS,
                    Provenance(
                        "synthetic:hand_to_world", parents=(provenance, transform.provenance)
                    ),
                    landmark_names=("synthetic_tip",),
                )
            )
    geometry_qc = tuple(check_transform(item.matrix) for item in transforms)
    continuity = check_trajectory_continuity(
        [TrajectorySample(time(0), transforms[0]), TrajectorySample(time(1000), transforms[0])],
        max_translation_speed_m_s=0.1,
        max_angular_speed_rad_s=0.1,
    )
    episode = Episode(
        "synthetic:paired_episode",
        participants,
        world,
        provenance,
        camera_samples=tuple(camera_samples),
        hand_samples=tuple(hand_samples),
        qc=(*geometry_qc, continuity),
    )
    assert all(result.status == QCStatus.PASS for result in episode.qc)
    assert len(episode.camera_samples) == 2
    assert len(episode.hand_samples) == 4
    assert {sample.frame for sample in episode.hand_samples} == {world}
    # Alice's left and Bob's right points coincide in world coordinates.
    np.testing.assert_allclose(episode.hand_samples[0].points, episode.hand_samples[3].points)
    np.testing.assert_allclose(
        graph.get_transform(cameras[1], cameras[0]).apply([[1, 0, 0]], unit=DistanceUnit.METERS),
        [[1, 0, 0]],
    )
