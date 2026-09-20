"""Verified device-hand and RGB camera transform chains, without cross-device time matching."""

from dataclasses import replace

from duet.adapters.comind.mps import CameraCalibrationSample, MpsTrajectorySample, device_clock
from duet.adapters.comind.multislam import ROLES, SharedWorldVerification
from duet.qc.synchronization import check_nearest_residuals
from duet.schemas.common import (
    FrameId,
    ParticipantId,
    Provenance,
    SampleMetadata,
    SampleState,
)
from duet.schemas.episode import CameraSample, HandSample
from duet.schemas.time import Timestamp
from duet.synchronization.matching import NearestTimestampMatch, nearest_timestamp


def _match_pose(
    query: Timestamp,
    pose: MpsTrajectorySample | None,
    *,
    participant: str,
    verification: SharedWorldVerification,
    max_gap_seconds: float,
) -> NearestTimestampMatch:
    """Validate the selected pose's role, graph, frames and same-device clock."""
    if participant not in ROLES:
        raise ValueError("participant must be helper or leader")
    clock = device_clock(verification.recording_id, participant)
    if query.clock_domain != clock:
        raise ValueError("query must use that participant's own device clock")
    if pose is not None:
        if pose.participant_id != ParticipantId(f"participant/{participant}"):
            raise ValueError("pose belongs to a different participant")
        if pose.graph_uid != verification.graph_uid:
            raise ValueError("pose does not belong to the verified Multi-SLAM world graph")
        if (
            pose.transform.destination != verification.world_frame
            or pose.transform.source != FrameId(f"{participant}/device")
        ):
            raise ValueError("pose must be T_shared_world_device, not a standard MPS solution")
    return nearest_timestamp(
        query,
        [] if pose is None else [pose.timestamp],
        max_gap_seconds=max_gap_seconds,
        clock_domain=clock,
    )


def hand_to_shared_world(
    hand: HandSample,
    pose: MpsTrajectorySample | None,
    *,
    verification: SharedWorldVerification,
    max_gap_seconds: float,
) -> tuple[HandSample, NearestTimestampMatch]:
    """Apply a selected nearest T_shared_world_device to 21 device landmarks.

    The caller selects the nearest candidate from the participant trajectory;
    this function independently enforces its residual limit and clock/frame
    identity. No cross-device timestamps are ever compared. Missing data and gap
    rejections yield an explicitly absent shared-world hand, never a zero pose.
    """
    roles = {f"participant/{role}": role for role in ROLES}
    if hand.participant_id.name not in roles:
        raise ValueError("hand must belong to the verified helper or leader participant")
    participant = roles[hand.participant_id.name]
    if hand.frame != FrameId(f"{participant}/device"):
        raise ValueError("source hand landmarks must be in this participant's device frame")
    match = _match_pose(
        hand.timestamp,
        pose,
        participant=participant,
        verification=verification,
        max_gap_seconds=max_gap_seconds,
    )
    metadata = hand.metadata
    points = None
    if hand.points is not None:
        if match.accepted:
            assert pose is not None
            points = pose.transform.apply(hand.points, unit=hand.unit)
        else:
            metadata = SampleMetadata(
                SampleState.UNKNOWN,
                confidence=hand.metadata.confidence,
                reason=f"Shared-world pose unavailable: {match.status.value}",
            )
    parents = (hand.provenance, verification.provenance)
    if pose is not None:
        parents += (pose.provenance,)
    transformed = replace(
        hand,
        frame=verification.world_frame,
        points=points,
        metadata=metadata,
        provenance=Provenance(
            "duet.comind.hand_to_shared_world",
            "T_shared_world_device applied to device-meter landmarks; same-device nearest time",
            parents,
        ),
        qc=hand.qc + (check_nearest_residuals([match], max_residual_seconds=max_gap_seconds),),
    )
    return transformed, match


def camera_to_shared_world(
    calibration_frame: CameraCalibrationSample,
    pose: MpsTrajectorySample | None,
    *,
    camera_label: str,
    participant: str,
    verification: SharedWorldVerification,
    max_gap_seconds: float,
) -> tuple[CameraSample, NearestTimestampMatch]:
    """Compose T_shared_world_device @ T_device_camera at calibration device time.

    The caller supplies the camera label after inspecting calibration and camera
    identity evidence. This returns the physical camera frame pose; it does not
    infer MP4 display orientation, apply rolling shutter, or map video PTS.
    """
    cameras = [camera for camera in calibration_frame.cameras if camera.label == camera_label]
    if len(cameras) != 1:
        raise ValueError("requested camera label must identify exactly one calibration")
    camera = cameras[0]
    if camera.transform.destination != FrameId(f"{participant}/device"):
        raise ValueError("calibration must target the same participant's device frame")
    if camera.transform.source != FrameId(f"{participant}/{camera_label}"):
        raise ValueError("calibration source frame must match the participant and camera label")
    match = _match_pose(
        calibration_frame.timestamp,
        pose,
        participant=participant,
        verification=verification,
        max_gap_seconds=max_gap_seconds,
    )
    transform = None
    metadata = SampleMetadata()
    if not camera.calibrated:
        metadata = SampleMetadata(SampleState.UNKNOWN, reason="Camera is not calibrated")
    elif not match.accepted:
        metadata = SampleMetadata(
            SampleState.UNKNOWN, reason=f"Shared-world pose unavailable: {match.status.value}"
        )
    else:
        assert pose is not None
        transform = pose.transform.compose(camera.transform)
    parents = (calibration_frame.provenance, verification.provenance)
    if pose is not None:
        parents += (pose.provenance,)
    result = CameraSample(
        participant_id=ParticipantId(f"participant/{participant}"),
        camera_frame=camera.transform.source,
        timestamp=calibration_frame.timestamp,
        transform=transform,
        provenance=Provenance("duet.comind.camera_to_shared_world", parents=parents),
        metadata=metadata,
        qc=(check_nearest_residuals([match], max_residual_seconds=max_gap_seconds),),
    )
    return result, match
