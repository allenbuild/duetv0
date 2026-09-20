"""Render verified local capture mappings on CoMind's paired frame-index timeline.

The common sequence index is deliberately not a timestamp or a clock transform.
Each participant's pose and hands are checked against their own DEVICE_TIME.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import FrameId, ParticipantId
from duet.schemas.episode import HandSample
from duet.schemas.time import Timestamp
from duet.synchronization.matching import nonnegative_seconds
from duet.visualization.rerun_episode import (
    COLORS,
    ROLES,
    EntityUpdate,
    VerifiedImageGeometry,
    _log_trajectories,
    _output_path,
    _recording,
)


@dataclass(frozen=True)
class MappedParticipantFrame:
    """Source observations for one verified MP4-to-VRS image correspondence."""

    role: str
    device_timestamp: Timestamp
    vrs_rgb_index: int
    mapping_status: EvidenceStatus
    mapping_confidence: float
    device_pose: RigidTransform | None
    pose_timestamp: Timestamp | None
    hands: tuple[HandSample, ...] = ()
    qc_warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class MappedRenderFrame:
    frame_index: int
    participants: tuple[MappedParticipantFrame, ...]
    annotations: tuple[Mapping[str, object], ...] = ()


@dataclass(frozen=True)
class FrameIndexBindings:
    """Paired-video contract plus independent spatial and image evidence."""

    recording_id: str
    world_frame: FrameId
    alignment: MappingEvidence
    pose_max_gap_seconds: float = 0.002
    hand_max_gap_seconds: float = 0.020
    minimum_hand_confidence: float = 0.5
    camera_extrinsics: Mapping[str, RigidTransform] = field(default_factory=dict)
    image_geometry: Mapping[str, VerifiedImageGeometry] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.alignment.name != "paired_video_frame_alignment":
            raise ValueError("paired_video_frame_alignment evidence is required")
        self.alignment.require_verified()
        for value in (self.pose_max_gap_seconds, self.hand_max_gap_seconds):
            nonnegative_seconds(value, "maximum gap")
        if not 0 <= self.minimum_hand_confidence <= 1:
            raise ValueError("hand confidence threshold must be in [0, 1]")
        if set(self.camera_extrinsics) - set(ROLES) or set(self.image_geometry) - set(ROLES):
            raise ValueError("unknown participant in camera bindings")
        for role, transform in self.camera_extrinsics.items():
            if transform.source != FrameId(
                f"{role}/camera-rgb"
            ) or transform.destination != FrameId(f"{role}/device"):
                raise ValueError("camera extrinsics must be T_device_camera for this participant")
        for role, geometry in self.image_geometry.items():
            if role not in self.camera_extrinsics or geometry.camera_frame != FrameId(
                f"{role}/camera-rgb"
            ):
                raise ValueError("verified image rays require matching camera extrinsics")


def _local_residual(sample: Timestamp | None, query: Timestamp) -> Fraction | None:
    if sample is None:
        return None
    if sample.clock_domain != query.clock_domain:
        raise ValueError("pose/hand lookup must stay in the participant's own device clock")
    if sample.seconds is None:
        return None
    return sample.seconds - query.seconds


def _exact_ns(timestamp: Timestamp) -> int:
    if timestamp.seconds is None:
        raise ValueError("capture timestamp must be present")
    value = timestamp.seconds * 1_000_000_000
    if value.denominator != 1 or not 0 <= value < 2**63:
        raise ValueError("capture timestamp must be exactly representable as int64 nanoseconds")
    return int(value)


def assemble_index_frame(
    frame: MappedRenderFrame, bindings: FrameIndexBindings
) -> tuple[tuple[EntityUpdate, ...], dict[str, object]]:
    """Clear missing/rejected geometry; never promote an inferred frame mapping."""
    if type(frame.frame_index) is not int or frame.frame_index < 0:
        raise ValueError("sync frame index must be a nonnegative integer")
    if len(frame.participants) != 2 or {p.role for p in frame.participants} != set(ROLES):
        raise ValueError("one sample from each participant is required")
    updates = []
    status = {
        "comind_sync_frame_index": frame.frame_index,
        "timeline_authority": bindings.alignment.statement,
        "participants": {},
        "active_handovers": [dict(annotation) for annotation in frame.annotations],
        "annotation_endpoint_policy": "display includes both supplied boundary indices",
        "scene": "disabled: scan-to-shared-world registration remains unresolved",
    }
    for participant in frame.participants:
        role, query = participant.role, participant.device_timestamp
        if participant.mapping_status is not EvidenceStatus.VERIFIED:
            raise ValueError("every displayed MP4 frame requires a VERIFIED VRS mapping")
        if query.clock_domain != device_clock(bindings.recording_id, role) or query.seconds is None:
            raise ValueError("mapping must supply this participant's exact DEVICE_TIME")
        if type(participant.vrs_rgb_index) is not int or participant.vrs_rgb_index < 0:
            raise ValueError("verified mapping requires an original VRS RGB index")
        if not 0 <= participant.mapping_confidence <= 1:
            raise ValueError("mapping confidence must be finite and in [0, 1]")
        residual = _local_residual(participant.pose_timestamp, query)
        pose = participant.device_pose
        if pose is not None and (
            pose.source != FrameId(f"{role}/device") or pose.destination != bindings.world_frame
        ):
            raise ValueError("device pose must target the verified shared world")
        pose_ok = (
            pose is not None
            and residual is not None
            and abs(residual) <= nonnegative_seconds(bindings.pose_max_gap_seconds, "pose gap")
        )
        updates.append(EntityUpdate(f"world/{role}/device", "pose" if pose_ok else "clear", pose))
        item = {
            "device_timestamp_ns": _exact_ns(query),
            "device_clock": query.clock_domain.name,
            "vrs_rgb_index": participant.vrs_rgb_index,
            "mapping_status": participant.mapping_status.name,
            "mapping_confidence": participant.mapping_confidence,
            "pose_residual_ns": None if residual is None else int(residual * 1_000_000_000),
            "pose_present": pose_ok,
            "hands": {},
            "qc_warnings": list(participant.qc_warnings),
        }
        hands = {hand.hand_id: hand for hand in participant.hands}
        if len(hands) != len(participant.hands) or set(hands) - {"left", "right"}:
            raise ValueError("duplicate or unsupported hand identity")
        for side in ("left", "right"):
            hand = hands.get(side)
            hand_residual, confidence, present = None, None, False
            if hand is not None:
                if hand.participant_id != ParticipantId(f"participant/{role}") or (
                    hand.frame != bindings.world_frame
                ):
                    raise ValueError("hand participant/shared-world identity is incompatible")
                hand_residual = _local_residual(hand.timestamp, query)
                confidence = (
                    None if hand.metadata.confidence is None else hand.metadata.confidence.value
                )
                present = (
                    hand.points is not None
                    and hand_residual is not None
                    and abs(hand_residual)
                    <= nonnegative_seconds(bindings.hand_max_gap_seconds, "hand gap")
                    and confidence is not None
                    and confidence >= bindings.minimum_hand_confidence
                )
            updates.append(
                EntityUpdate(
                    f"world/{role}/hands/{side}",
                    "points" if present else "clear",
                    None if hand is None else hand.points,
                )
            )
            item["hands"][side] = {
                "present": present,
                "confidence": confidence,
                "residual_ns": None
                if hand_residual is None
                else int(hand_residual * 1_000_000_000),
                "source_state": None if hand is None else hand.metadata.state.value,
            }
        extrinsic = bindings.camera_extrinsics.get(role)
        camera = pose.compose(extrinsic) if pose_ok and extrinsic is not None else None
        updates.append(
            EntityUpdate(f"world/{role}/camera_rgb", "pose" if camera else "clear", camera)
        )
        geometry = bindings.image_geometry.get(role)
        updates.append(
            EntityUpdate(
                f"world/{role}/camera_rgb/frustum",
                "frustum" if camera and geometry else "clear",
                geometry,
            )
        )
        item["camera_present"] = camera is not None
        item["frustum_verified"] = geometry is not None
        status["participants"][role] = item
    return tuple(updates), status


def save_index_episode(
    output: str | Path,
    frames: Iterable[MappedRenderFrame],
    *,
    bindings: FrameIndexBindings,
    video_frames: Mapping[str, Iterator[tuple[int, np.ndarray]]],
    trajectories: Mapping[str, np.ndarray] | None = None,
) -> Path:
    """Stream a validated contiguous handover clip to RRD with bounded image memory.

    The caller supplies exact-index image iterators backed by original MP4 PTS.
    No video-time arithmetic or cross-device clock conversion occurs here.
    """
    import rerun as rr

    if set(video_frames) != set(ROLES):
        raise ValueError("both paired ego-video iterators are required")
    output = _output_path(output)
    recording = _recording(output, "duet_comind_synchronized_handover")
    try:
        _log_trajectories(recording, trajectories or {})
        previous_index = None
        count = 0
        for frame in frames:
            if previous_index is not None and frame.frame_index != previous_index + 1:
                raise ValueError("handover output requires consecutive sync indices")
            updates, status = assemble_index_frame(frame, bindings)
            previous_index = frame.frame_index
            recording.set_time("comind_sync_frame_index", sequence=frame.frame_index)
            for update in updates:
                if update.kind == "clear":
                    recording.log(update.path, rr.Clear(recursive=True), strict=True)
                elif update.kind == "pose":
                    pose = update.value
                    recording.log(
                        update.path,
                        rr.Transform3D(translation=pose.matrix[:3, 3], mat3x3=pose.matrix[:3, :3]),
                        rr.TransformAxes3D(0.08),
                        strict=True,
                    )
                elif update.kind == "points":
                    role = update.path.split("/")[1]
                    recording.log(
                        update.path,
                        rr.Points3D(update.value, radii=0.008, colors=COLORS[role]),
                        strict=True,
                    )
                elif update.kind == "frustum":
                    rays = update.value.boundary_rays_camera * 0.20
                    strips = [np.array([[0, 0, 0], ray]) for ray in rays]
                    strips.append(np.vstack((rays, rays[:1])))
                    recording.log(update.path, rr.LineStrips3D(strips), strict=True)
            for role in ROLES:
                try:
                    index, pixels = next(video_frames[role])
                except StopIteration as error:
                    raise ValueError("video ended before the requested handover window") from error
                if index != frame.frame_index:
                    raise ValueError("decoded MP4 index differs from synchronized frame index")
                geometry = bindings.image_geometry.get(role)
                if (
                    geometry is not None
                    and (pixels.shape[1], pixels.shape[0]) != geometry.resolution
                ):
                    raise ValueError("video image dimensions disagree with verified calibration")
                recording.log(
                    f"video/{role}", rr.Image(pixels).compress(jpeg_quality=85), strict=True
                )
            recording.log(
                "status",
                rr.TextDocument(
                    "```json\n" + json.dumps(status, indent=2) + "\n```", media_type="text/markdown"
                ),
                strict=True,
            )
            count += 1
        if not count:
            raise ValueError("handover output must contain at least one verified frame")
        recording.flush()
    finally:
        recording.disconnect()
    return output
