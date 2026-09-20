"""Rerun assembly with explicit common-clock and image-geometry evidence.

The renderer never fits clocks or approximates a fisheye camera as a pinhole.
Frame assembly is independent of the Rerun GUI. Real-data diagnostics can show
verified static trajectories without claiming simultaneous participant poses.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from duet.adapters.comind.video import EgoVideo, VideoFrameMatch
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import DistanceUnit, FrameId, ParticipantId, Provenance, require_name
from duet.schemas.episode import CameraSample, HandoverAnnotation, HandSample
from duet.schemas.time import ClockDomain, Timestamp, VerifiedClockMapping, comparable_seconds
from duet.synchronization.matching import nonnegative_seconds

ROLES = ("helper", "leader")
COLORS = {"helper": [60, 180, 255], "leader": [255, 160, 70]}


@dataclass(frozen=True, eq=False)
class VerifiedImageGeometry:
    """Actual model-unprojected perimeter rays in the exported camera frame.

    Ray order follows the image perimeter. This supports nonlinear fisheye
    frustum outlines without claiming a pinhole projection or 2D reprojection.
    The evidence must establish exported-pixel orientation and the camera model.
    """

    camera_frame: FrameId
    resolution: tuple[int, int]
    boundary_rays_camera: ArrayLike
    projection_model: str
    verification: str
    provenance: Provenance

    def __post_init__(self) -> None:
        require_name(self.verification, "image geometry verification")
        require_name(self.projection_model, "projection model")
        if not isinstance(self.camera_frame, FrameId) or not isinstance(
            self.provenance, Provenance
        ):
            raise TypeError("image geometry requires an explicit frame and provenance")
        if len(self.resolution) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.resolution
        ):
            raise ValueError("image resolution must contain two positive integer dimensions")
        if np.iscomplexobj(self.boundary_rays_camera):
            raise ValueError("camera rays must be real")
        rays = np.asarray(self.boundary_rays_camera, dtype=np.float64)
        if rays.ndim != 2 or rays.shape[1] != 3 or len(rays) < 4 or not np.isfinite(rays).all():
            raise ValueError("camera boundary requires at least four finite 3D rays")
        if not np.allclose(np.linalg.norm(rays, axis=1), 1, atol=1e-8, rtol=0):
            raise ValueError("camera boundary rays must already be unit length; no repair is done")
        frozen = np.frombuffer(rays.tobytes(), dtype=np.float64).reshape(rays.shape)
        object.__setattr__(self, "boundary_rays_camera", frozen)


@dataclass(frozen=True)
class ViewerBindings:
    """Only explicitly verified mappings may drive the common timeline.

    A reported UTC pair/fitted offset is intentionally not accepted in place of
    VerifiedClockMapping. Image, video, and annotation layers are independently
    optional; absence disables the layer. No CLI option upgrades inference.
    """

    world_frame: FrameId
    common_clock: ClockDomain
    device_time_mappings: Mapping[str, VerifiedClockMapping]
    max_gap_seconds: Fraction | float
    image_geometry: Mapping[str, VerifiedImageGeometry] = field(default_factory=dict)
    video_time_mappings: Mapping[str, VerifiedClockMapping] = field(default_factory=dict)
    annotation_mapping: VerifiedClockMapping | None = None
    annotation_end_inclusive: bool | None = None
    annotation_interval_verification: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.world_frame, FrameId) or not isinstance(
            self.common_clock, ClockDomain
        ):
            raise TypeError("viewer requires explicit world and common clock identities")
        if set(self.device_time_mappings) != set(ROLES):
            raise ValueError("both participant device clocks require verified mappings")
        for field_name in ("device_time_mappings", "video_time_mappings"):
            mappings = dict(getattr(self, field_name))
            if set(mappings) - set(ROLES):
                raise ValueError("unknown participant in viewer clock mappings")
            for mapping in mappings.values():
                if not isinstance(mapping, VerifiedClockMapping):
                    raise TypeError("viewer clock mappings must be explicitly verified")
                if mapping.destination != self.common_clock:
                    raise ValueError("all viewer mappings must target the common clock")
            object.__setattr__(self, field_name, MappingProxyType(mappings))
        if set(self.image_geometry) - set(ROLES):
            raise ValueError("unknown participant in image geometry")
        if not all(
            isinstance(item, VerifiedImageGeometry) for item in self.image_geometry.values()
        ):
            raise TypeError("image geometry must be explicitly verified")
        object.__setattr__(self, "image_geometry", MappingProxyType(dict(self.image_geometry)))
        object.__setattr__(
            self, "max_gap_seconds", nonnegative_seconds(self.max_gap_seconds, "gap")
        )
        if self.annotation_mapping is not None:
            if not isinstance(self.annotation_mapping, VerifiedClockMapping):
                raise TypeError("annotation mapping must be explicitly verified")
            if self.annotation_mapping.destination != self.common_clock:
                raise ValueError("annotation mapping must target the common clock")
            if not isinstance(self.annotation_end_inclusive, bool):
                raise ValueError("annotation interval inclusion must be explicitly specified")
            require_name(self.annotation_interval_verification, "annotation interval verification")


@dataclass(frozen=True)
class ParticipantRenderFrame:
    role: str
    device_timestamp: Timestamp
    device_pose: RigidTransform | None
    camera: CameraSample | None = None
    hands: tuple[HandSample, ...] = ()


@dataclass(frozen=True)
class EpisodeRenderFrame:
    """One explicitly synchronized request; source observations retain their timestamps."""

    common_timestamp: Timestamp
    participants: tuple[ParticipantRenderFrame, ...]


@dataclass(frozen=True)
class EntityUpdate:
    path: str
    kind: str
    value: object


@dataclass(frozen=True)
class FramePlan:
    common_nanoseconds: int
    updates: tuple[EntityUpdate, ...]
    status: Mapping[str, object]


def _residual(
    sample: Timestamp, query: Timestamp, mapping: VerifiedClockMapping
) -> Fraction | None:
    if sample.clock_domain != mapping.source:
        raise ValueError("sample belongs to the wrong participant or source clock")
    seconds = comparable_seconds(sample, query.clock_domain, [mapping])
    return None if seconds is None else seconds - query.seconds


def _pose_check(pose: RigidTransform, source: FrameId, destination: FrameId) -> None:
    if pose.source != source or pose.destination != destination:
        raise ValueError("rendered transform has incompatible source/destination frames")


def assemble_frame(frame: EpisodeRenderFrame, bindings: ViewerBindings) -> FramePlan:
    """Validate mappings and produce GUI-free log instructions, clearing stale data."""
    query = frame.common_timestamp
    if query.clock_domain != bindings.common_clock or query.seconds is None:
        raise ValueError("render request requires a present timestamp in the verified common clock")
    nanoseconds = query.seconds * 1_000_000_000
    if nanoseconds.denominator != 1 or not -(2**63) <= nanoseconds < 2**63:
        raise ValueError("Rerun timeline requires exactly representable signed 64-bit nanoseconds")
    if len(frame.participants) != 2 or {sample.role for sample in frame.participants} != set(ROLES):
        raise ValueError("render frame must contain exactly one helper and one leader")
    updates: list[EntityUpdate] = []
    status: dict[str, object] = {
        "common_clock": bindings.common_clock.name,
        "common_timestamp_ns": int(nanoseconds),
        "participants": {},
    }
    for sample in frame.participants:
        role = sample.role
        identity = ParticipantId(f"participant/{role}")
        mapping = bindings.device_time_mappings[role]
        residual = _residual(sample.device_timestamp, query, mapping)
        accepted = residual is not None and abs(residual) <= bindings.max_gap_seconds
        device_path = f"world/{role}/device"
        if sample.device_pose is not None:
            _pose_check(sample.device_pose, FrameId(f"{role}/device"), bindings.world_frame)
        updates.append(
            EntityUpdate(
                device_path,
                "pose" if accepted and sample.device_pose is not None else "clear",
                sample.device_pose if accepted else None,
            )
        )
        participant_status: dict[str, object] = {
            "source_timestamp": str(sample.device_timestamp.raw_value),
            "source_unit": sample.device_timestamp.unit.value,
            "source_clock": sample.device_timestamp.clock_domain.name,
            "device_residual_seconds": None if residual is None else float(residual),
            "device_present": accepted and sample.device_pose is not None,
            "hands": {},
        }
        hands = {hand.hand_id: hand for hand in sample.hands}
        if len(hands) != len(sample.hands) or set(hands) - {"left", "right"}:
            raise ValueError("duplicate or unsupported hand identity")
        for side in ("left", "right"):
            hand = hands.get(side)
            hand_residual = None
            present = False
            if hand is not None:
                if hand.participant_id != identity or hand.frame != bindings.world_frame:
                    raise ValueError("hand participant or shared frame is incompatible")
                hand_residual = _residual(hand.timestamp, query, mapping)
                present = (
                    hand.points is not None
                    and hand_residual is not None
                    and abs(hand_residual) <= bindings.max_gap_seconds
                )
            updates.append(
                EntityUpdate(
                    f"world/{role}/hands/{side}",
                    "points" if present else "clear",
                    hand.points if present else None,
                )
            )
            participant_status["hands"][side] = {
                "present": present,
                "source_state": None if hand is None else hand.metadata.state.value,
                "confidence": None
                if hand is None or hand.metadata.confidence is None
                else hand.metadata.confidence.value,
                "residual_seconds": None if hand_residual is None else float(hand_residual),
                "source_timestamp": None if hand is None else str(hand.timestamp.raw_value),
                "qc": []
                if hand is None
                else [
                    {"name": result.check, "status": result.status.value, "message": result.message}
                    for result in hand.qc
                ],
            }
        camera = sample.camera
        camera_path = f"world/{role}/camera_rgb"
        camera_present = False
        camera_residual = None
        if camera is not None:
            if camera.participant_id != identity or camera.camera_frame != FrameId(
                f"{role}/camera-rgb"
            ):
                raise ValueError("camera participant or frame is incompatible")
            camera_residual = _residual(camera.timestamp, query, mapping)
            camera_present = (
                camera.transform is not None
                and camera_residual is not None
                and abs(camera_residual) <= bindings.max_gap_seconds
            )
            if camera.transform is not None:
                _pose_check(camera.transform, camera.camera_frame, bindings.world_frame)
        updates.append(
            EntityUpdate(
                camera_path,
                "pose" if camera_present else "clear",
                camera.transform if camera_present else None,
            )
        )
        geometry = bindings.image_geometry.get(role)
        if geometry is not None and geometry.camera_frame != FrameId(f"{role}/camera-rgb"):
            raise ValueError("image rays refer to the wrong camera frame")
        updates.append(
            EntityUpdate(
                f"{camera_path}/frustum",
                "frustum" if camera_present and geometry else "clear",
                geometry if camera_present else None,
            )
        )
        participant_status["camera_present"] = camera_present
        participant_status["camera_source_timestamp"] = (
            None if camera is None else str(camera.timestamp.raw_value)
        )
        participant_status["camera_residual_seconds"] = (
            None if camera_residual is None else float(camera_residual)
        )
        participant_status["camera_state"] = None if camera is None else camera.metadata.state.value
        participant_status["camera_qc"] = (
            []
            if camera is None
            else [
                {"name": result.check, "status": result.status.value, "message": result.message}
                for result in camera.qc
            ]
        )
        participant_status["frustum"] = (
            "verified" if geometry else "disabled: image geometry unresolved"
        )
        status["participants"][role] = participant_status
    return FramePlan(int(nanoseconds), tuple(updates), status)


def lookup_video_on_common(
    video: EgoVideo,
    query: Timestamp,
    mapping: VerifiedClockMapping,
    *,
    max_gap_seconds: float | Fraction,
) -> tuple[VideoFrameMatch, Fraction | None]:
    """Invert an explicit verified affine map and decode only a nearby video frame."""
    if not isinstance(mapping, VerifiedClockMapping):
        raise TypeError("video lookup requires a verified mapping")
    if mapping.source != video.clock_domain or mapping.destination != query.clock_domain:
        raise ValueError("video mapping endpoints do not match the requested clocks")
    if query.seconds is None:
        raise ValueError("common video query cannot be missing")
    maximum = nonnegative_seconds(max_gap_seconds, "video common gap")
    video_seconds = (query.seconds - mapping.offset_seconds) / mapping.scale
    result = video.nearest_frame(video_seconds, max_gap_seconds=maximum / mapping.scale)
    candidate = result.candidate
    if candidate is not None and candidate.clock_domain != mapping.source:
        raise ValueError("decoded video frame has a different clock domain")
    residual = (
        None
        if candidate is None
        else (mapping.scale * candidate.seconds + mapping.offset_seconds - query.seconds)
    )
    return result, residual


def active_annotations(
    annotations: Iterable[HandoverAnnotation], query: Timestamp, bindings: ViewerBindings
) -> tuple[HandoverAnnotation, ...]:
    """Select only with a verified annotation map and explicit endpoint convention."""
    mapping = bindings.annotation_mapping
    if mapping is None:
        return ()
    if query.clock_domain != bindings.common_clock or query.seconds is None:
        raise ValueError("annotation query must use the common clock")
    active = []
    for annotation in annotations:
        if annotation.start_time.clock_domain != mapping.source:
            raise ValueError("annotation timestamp does not belong to the mapped source clock")
        start = comparable_seconds(annotation.start_time, query.clock_domain, [mapping])
        end = comparable_seconds(annotation.end_time, query.clock_domain, [mapping])
        if (
            start is not None
            and end is not None
            and start <= query.seconds
            and (
                query.seconds < end or (bindings.annotation_end_inclusive and query.seconds == end)
            )
        ):
            active.append(annotation)
    return tuple(active)


def _output_path(path: str | Path) -> Path:
    output = Path(path).resolve()
    raw = Path(__file__).resolve().parents[3] / "data" / "raw"
    if output.is_relative_to(raw):
        raise ValueError("Rerun output must stay outside immutable raw data")
    if output.suffix != ".rrd":
        raise ValueError("Rerun recording output must end in .rrd")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _recording(path: Path, application_id: str):
    import rerun as rr
    import rerun.blueprint as rrb

    recording = rr.RecordingStream(application_id)
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world", name="Verified shared-space geometry"),
            rrb.Vertical(
                rrb.TextDocumentView(origin="status", name="Evidence, timing and QC"),
                rrb.Spatial2DView(origin="video/helper", name="Helper ego (mapped only)"),
                rrb.Spatial2DView(origin="video/leader", name="Leader ego (mapped only)"),
            ),
        ),
        auto_views=False,
    )
    recording.save(path, default_blueprint=blueprint)
    return recording


def _log_trajectories(recording, trajectories: Mapping[str, ArrayLike]) -> None:
    import rerun as rr

    for role, points in trajectories.items():
        if role not in ROLES:
            raise ValueError("unknown trajectory participant")
        if np.iscomplexobj(points):
            raise ValueError("trajectory points must be real")
        array = np.asarray(points, dtype=float)
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError("trajectory sample positions must be finite (N, 3) meters")
        # Points, rather than connected strips, do not invent motion through gaps.
        recording.log(
            f"world/{role}/trajectory",
            rr.Points3D(
                array,
                colors=COLORS[role],
                radii=0.006,
            ),
            static=True,
            strict=True,
        )


def save_spatial_diagnostic(
    output: str | Path,
    *,
    world_frame: FrameId,
    trajectories: Mapping[str, ArrayLike],
    status: Mapping[str, object],
    provenance: Provenance,
) -> Path:
    """Save only static spatial evidence; no participant clocks are synchronized."""
    import rerun as rr

    output = _output_path(output)
    if not isinstance(world_frame, FrameId) or not isinstance(provenance, Provenance):
        raise TypeError("diagnostic geometry requires frame and provenance")
    recording = _recording(output, "duet_comind_spatial_diagnostic")
    try:
        _log_trajectories(recording, trajectories)
        body = {
            "mode": "SPATIAL DIAGNOSTIC ONLY — NO VERIFIED COMMON TIMELINE",
            "world_frame": world_frame.name,
            "source": provenance.source,
            "disabled_layers": [
                "current participant poses",
                "hands at a common instant",
                "synchronized ego video",
                "camera frustums",
                "active handover",
                "scan registration",
            ],
            "evidence_and_qc": dict(status),
        }
        recording.log(
            "status",
            rr.TextDocument(
                "# Spatial diagnostic — not synchronized V0\n\n```json\n"
                + json.dumps(body, indent=2)
                + "\n```",
                media_type="text/markdown",
            ),
            static=True,
            strict=True,
        )
        recording.flush()
    finally:
        recording.disconnect()
    return output


def save_episode(
    output: str | Path,
    frames: Iterable[EpisodeRenderFrame],
    *,
    bindings: ViewerBindings,
    trajectories: Mapping[str, ArrayLike] | None = None,
    videos: Mapping[str, EgoVideo] | None = None,
    annotations: tuple[HandoverAnnotation, ...] = (),
    scene_points_world: ArrayLike | None = None,
    scene_unit: DistanceUnit | None = None,
    scene_frame: FrameId | None = None,
    scene_registration_verification: str | None = None,
) -> Path:
    """Stream synchronized frames to RRD; optional layers require independent evidence.

    This API is ready for verified bindings. The current CoMind files do not
    establish those bindings. Fisheye boundary rays stay a nonlinear frustum
    outline; they are never logged as an invented pinhole camera.
    """
    import rerun as rr

    if not isinstance(bindings, ViewerBindings):
        raise TypeError("synchronized rendering requires verified ViewerBindings")
    if scene_points_world is not None:
        require_name(scene_registration_verification, "scan registration verification")
        if scene_unit is not DistanceUnit.METERS or scene_frame != bindings.world_frame:
            raise ValueError("scene must already be verified in the shared frame, in meters")
        if np.iscomplexobj(scene_points_world):
            raise ValueError("scene points must be real")
        scene = np.asarray(scene_points_world, dtype=float)
        if scene.ndim != 2 or scene.shape[1] != 3 or not np.isfinite(scene).all():
            raise ValueError("scene points must have finite shape (N, 3)")
    output = _output_path(output)
    recording = _recording(output, "duet_verified_episode")
    try:
        _log_trajectories(recording, trajectories or {})
        if scene_points_world is not None:
            recording.log("world/scene", rr.Points3D(scene, radii=0.004), static=True, strict=True)
        previous_time = None
        for frame in frames:
            plan = assemble_frame(frame, bindings)
            if previous_time is not None and plan.common_nanoseconds <= previous_time:
                raise ValueError("render frames must have strictly increasing common timestamps")
            previous_time = plan.common_nanoseconds
            # Integer sequence nanoseconds preserve exact common values without
            # falsely asserting that an arbitrary verified clock is Unix UTC.
            recording.set_time("common_time_ns", sequence=plan.common_nanoseconds)
            status = dict(plan.status)
            status["videos"] = {}
            for update in plan.updates:
                if update.kind == "clear":
                    recording.log(update.path, rr.Clear(recursive=True), strict=True)
                elif update.kind == "pose":
                    pose = update.value
                    recording.log(
                        update.path,
                        rr.Transform3D(
                            translation=pose.matrix[:3, 3],
                            mat3x3=pose.matrix[:3, :3],
                        ),
                        rr.TransformAxes3D(0.08),
                        strict=True,
                    )
                elif update.kind == "points":
                    recording.log(update.path, rr.Points3D(update.value, radii=0.008), strict=True)
                elif update.kind == "frustum":
                    rays = update.value.boundary_rays_camera * 0.2
                    strips = [np.array([[0, 0, 0], ray]) for ray in rays]
                    strips.append(np.vstack((rays, rays[:1])))
                    recording.log(update.path, rr.LineStrips3D(strips), strict=True)
            for role in ROLES:
                video, mapping = (videos or {}).get(role), bindings.video_time_mappings.get(role)
                if video is None or mapping is None:
                    recording.log(f"video/{role}", rr.Clear(recursive=True), strict=True)
                    status["videos"][role] = "disabled: verified video mapping/source unavailable"
                    continue
                match, residual = lookup_video_on_common(
                    video,
                    frame.common_timestamp,
                    mapping,
                    max_gap_seconds=bindings.max_gap_seconds,
                )
                if match.accepted and match.candidate is not None:
                    pixels = match.candidate.frame.to_ndarray(format="rgb24")
                    geometry = bindings.image_geometry.get(role)
                    if (
                        geometry is not None
                        and (pixels.shape[1], pixels.shape[0]) != geometry.resolution
                    ):
                        raise ValueError("exported image size disagrees with verified geometry")
                    recording.log(
                        f"video/{role}", rr.Image(pixels).compress(jpeg_quality=85), strict=True
                    )
                else:
                    recording.log(f"video/{role}", rr.Clear(recursive=True), strict=True)
                status["videos"][role] = {
                    "state": match.status.value,
                    "original_pts": None if match.candidate is None else match.candidate.pts,
                    "time_base": None
                    if match.candidate is None
                    else str(match.candidate.time_base),
                    "common_residual_seconds": None if residual is None else float(residual),
                }
            active = active_annotations(annotations, frame.common_timestamp, bindings)
            status["annotations"] = (
                "disabled: mapping/inclusion unresolved"
                if (bindings.annotation_mapping is None)
                else [
                    {
                        "object_category": item.object_category_level_1,
                        "initiator_source_label": item.initiator,
                        "delivering_flow_source_label": item.delivering_flow,
                    }
                    for item in active
                ]
            )
            status["scene"] = (
                "verified"
                if scene_points_world is not None
                else "disabled: registration unresolved"
            )
            recording.log(
                "status",
                rr.TextDocument(
                    "```json\n" + json.dumps(status, indent=2, default=str) + "\n```",
                    media_type="text/markdown",
                ),
                strict=True,
            )
        recording.flush()
    finally:
        recording.disconnect()
    return output
