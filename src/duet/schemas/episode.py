"""Canonical episodes whose available spatial samples share one explicit frame.

Samples retain independent source timestamps. Membership in one episode never
implies that their clocks are synchronized or that array positions correspond.
"""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal
from types import MappingProxyType
from typing import TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from duet.geometry.transforms import RigidTransform
from duet.qc.result import QCResult
from duet.schemas.common import (
    DistanceUnit,
    FrameId,
    ParticipantId,
    Provenance,
    SampleMetadata,
    SampleState,
    require_name,
)
from duet.schemas.time import Timestamp

JSONValue: TypeAlias = (
    str | int | float | Decimal | bool | None | list["JSONValue"] | dict[str, "JSONValue"]
)


def _validate_sample(
    participant_id: ParticipantId,
    timestamp: Timestamp,
    provenance: Provenance,
    metadata: SampleMetadata,
    payload_present: bool,
) -> None:
    if not isinstance(participant_id, ParticipantId):
        raise TypeError("sample requires a ParticipantId")
    if not isinstance(timestamp, Timestamp) or not isinstance(provenance, Provenance):
        raise TypeError("sample requires a Timestamp and Provenance")
    if not isinstance(metadata, SampleMetadata):
        raise TypeError("sample metadata must be SampleMetadata")
    if payload_present != (metadata.state == SampleState.PRESENT):
        raise ValueError("payload availability must agree with the explicit sample state")


def _frame_index(value: int | None, name: str) -> None:
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class FrameIdentifier:
    """Source image/frame identifier, distinct from a spatial FrameId.

    Indices retain the source numbering and never serve as a clock mapping.
    source_id can preserve an opaque identifier when the source has no index.
    """

    stream_id: str
    index: int | None = None
    source_id: str | None = None

    def __post_init__(self) -> None:
        require_name(self.stream_id, "frame stream ID")
        _frame_index(self.index, "frame index")
        if self.source_id is not None:
            require_name(self.source_id, "source frame ID")
        if self.index is None and self.source_id is None:
            raise ValueError("frame identifier requires an index or source ID")


@dataclass(frozen=True)
class CameraSample:
    """A timestamped T_destination_camera in meters, or an explicit missing pose."""

    participant_id: ParticipantId
    camera_frame: FrameId
    timestamp: Timestamp
    transform: RigidTransform | None
    provenance: Provenance
    frame_identifier: FrameIdentifier | None = None
    metadata: SampleMetadata = field(default_factory=SampleMetadata)
    qc: tuple[QCResult, ...] = ()

    def __post_init__(self) -> None:
        _validate_sample(
            self.participant_id,
            self.timestamp,
            self.provenance,
            self.metadata,
            self.transform is not None,
        )
        if not isinstance(self.camera_frame, FrameId):
            raise TypeError("camera frame must be a FrameId")
        if self.transform is not None:
            if not isinstance(self.transform, RigidTransform):
                raise TypeError("camera transform must be a RigidTransform")
            if self.transform.source != self.camera_frame:
                raise ValueError("camera transform source must match camera_frame")
        if self.frame_identifier is not None and not isinstance(
            self.frame_identifier, FrameIdentifier
        ):
            raise TypeError("frame_identifier must be a FrameIdentifier or None")
        object.__setattr__(self, "qc", _qc_tuple(self.qc))


@dataclass(frozen=True, eq=False)
class HandSample:
    """Named hand landmarks in an explicit coordinate frame, in meters.

    hand_id and landmark_names are caller-supplied, verified labels; no skeleton,
    left/right association, or landmark ordering is assumed. Missing hands use
    points=None with a missing/unknown state, rather than zero-filled coordinates.
    """

    participant_id: ParticipantId
    hand_id: str
    timestamp: Timestamp
    frame: FrameId
    points: ArrayLike | None
    unit: DistanceUnit
    provenance: Provenance
    landmark_names: tuple[str, ...] = ()
    metadata: SampleMetadata = field(default_factory=SampleMetadata)
    qc: tuple[QCResult, ...] = ()

    def __post_init__(self) -> None:
        _validate_sample(
            self.participant_id,
            self.timestamp,
            self.provenance,
            self.metadata,
            self.points is not None,
        )
        require_name(self.hand_id, "hand ID")
        if not isinstance(self.frame, FrameId):
            raise TypeError("hand frame must be a FrameId")
        if not isinstance(self.unit, DistanceUnit) or self.unit != DistanceUnit.METERS:
            raise ValueError("canonical hand points must explicitly use DistanceUnit.METERS")
        names = tuple(self.landmark_names)
        for name in names:
            require_name(name, "landmark name")
        if len(set(names)) != len(names):
            raise ValueError("landmark names must be unique")
        object.__setattr__(self, "landmark_names", names)
        if self.points is not None:
            if np.iscomplexobj(self.points):
                raise ValueError("hand points must be real numbers")
            points = np.asarray(self.points, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
                raise ValueError("hand points must have shape (N, 3) with N > 0")
            if not np.isfinite(points).all():
                raise ValueError("hand points must be finite; mark absent samples explicitly")
            if names and len(names) != len(points):
                raise ValueError("landmark names must match the number of points")
            # Immutable bytes backing prevents mutation through both input aliases
            # and setflags(write=True), preserving validated dimensions and values.
            immutable: NDArray[np.float64] = np.frombuffer(points.tobytes(), dtype=np.float64)
            object.__setattr__(self, "points", immutable.reshape(points.shape))
        object.__setattr__(self, "qc", _qc_tuple(self.qc))


@dataclass(frozen=True)
class HandoverAnnotation:
    """Original source interval and labels, without participant identity inference.

    Frame-boundary inclusion and cross-modality timestamp relationships are not
    inferred. Initiation types and categories retain the original scalar/list
    representation until source semantics are established.
    """

    recording_id: str
    annotation_id: str | None
    start_frame: int | None
    end_frame: int | None
    start_time: Timestamp
    end_time: Timestamp
    initiator: str | None
    delivering_flow: str | None
    initiation_type: JSONValue
    object_category_level_1: JSONValue
    object_category_level_2: JSONValue
    object_category_level_3: JSONValue
    provenance: Provenance
    metadata: SampleMetadata = field(default_factory=SampleMetadata)
    source_fields: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_name(self.recording_id, "annotation recording ID")
        if self.annotation_id is not None:
            require_name(self.annotation_id, "annotation ID")
        _frame_index(self.start_frame, "start frame")
        _frame_index(self.end_frame, "end frame")
        if (
            self.start_frame is not None
            and self.end_frame is not None
            and self.start_frame > self.end_frame
        ):
            raise ValueError("handover start frame must not exceed end frame")
        if not isinstance(self.start_time, Timestamp) or not isinstance(self.end_time, Timestamp):
            raise TypeError("handover boundaries must be Timestamp objects")
        if self.start_time.clock_domain != self.end_time.clock_domain:
            raise ValueError("handover boundaries must share one clock domain")
        start, end = self.start_time.seconds, self.end_time.seconds
        if start is not None and end is not None and start > end:
            raise ValueError("handover start time must not exceed end time")
        for label in (self.initiator, self.delivering_flow):
            if label is not None and not isinstance(label, str):
                raise TypeError("source labels must be strings or None")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("handover annotation requires provenance")
        if not isinstance(self.metadata, SampleMetadata):
            raise TypeError("annotation metadata must be SampleMetadata")
        if self.metadata.state != SampleState.PRESENT:
            raise ValueError(
                "an annotation record is present; absent boundaries use raw_value=None"
            )
        # Keep raw JSON list/scalar distinctions while separating canonical data
        # from the caller's mutable parsing buffers.
        for name in (
            "initiation_type",
            "object_category_level_1",
            "object_category_level_2",
            "object_category_level_3",
        ):
            object.__setattr__(self, name, deepcopy(getattr(self, name)))
        object.__setattr__(
            self, "source_fields", MappingProxyType(deepcopy(dict(self.source_fields)))
        )


def _qc_tuple(results: tuple[QCResult, ...]) -> tuple[QCResult, ...]:
    results = tuple(results)
    if not all(isinstance(result, QCResult) for result in results):
        raise TypeError("QC metadata must contain QCResult objects")
    return results


@dataclass(frozen=True)
class Episode:
    """Available camera/hand geometry in one shared frame, with independent clocks.

    This is a canonical spatial representation, not a synchronization claim.
    Build timestamp matches explicitly before sampling multimodal data together.
    """

    episode_id: str
    participants: tuple[ParticipantId, ...]
    shared_frame: FrameId
    provenance: Provenance
    camera_samples: tuple[CameraSample, ...] = ()
    hand_samples: tuple[HandSample, ...] = ()
    handovers: tuple[HandoverAnnotation, ...] = ()
    qc: tuple[QCResult, ...] = ()

    def __post_init__(self) -> None:
        require_name(self.episode_id, "episode ID")
        if not isinstance(self.shared_frame, FrameId) or not isinstance(
            self.provenance, Provenance
        ):
            raise TypeError("episode requires an explicit shared frame and provenance")
        participants = tuple(self.participants)
        if not participants or not all(isinstance(item, ParticipantId) for item in participants):
            raise ValueError("episode must declare at least one ParticipantId")
        if len(set(participants)) != len(participants):
            raise ValueError("episode participant IDs must be unique")
        object.__setattr__(self, "participants", participants)
        cameras, hands, handovers = (
            tuple(self.camera_samples),
            tuple(self.hand_samples),
            tuple(self.handovers),
        )
        if not all(isinstance(sample, CameraSample) for sample in cameras):
            raise TypeError("camera_samples must contain CameraSample objects")
        if not all(isinstance(sample, HandSample) for sample in hands):
            raise TypeError("hand_samples must contain HandSample objects")
        if not all(isinstance(item, HandoverAnnotation) for item in handovers):
            raise TypeError("handovers must contain HandoverAnnotation objects")
        for sample in (*cameras, *hands):
            if sample.participant_id not in participants:
                raise ValueError("sample participant must be declared in the episode")
        for camera in cameras:
            if camera.transform is not None and camera.transform.destination != self.shared_frame:
                raise ValueError("episode camera transforms must target the shared frame")
        for hand in hands:
            if hand.frame != self.shared_frame:
                raise ValueError("episode hand samples must use the shared frame")
        object.__setattr__(self, "camera_samples", cameras)
        object.__setattr__(self, "hand_samples", hands)
        object.__setattr__(self, "handovers", handovers)
        object.__setattr__(self, "qc", _qc_tuple(self.qc))
