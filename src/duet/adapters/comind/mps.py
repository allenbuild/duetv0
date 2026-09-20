"""Streaming readers for documented Project Aria MPS fields used by CoMind.

Trajectory and hand coordinates use meters. Every participant has its own device
clock. UTC fields are retained separately in an unverified participant-scoped
clock; their presence never establishes a cross-device synchronization mapping.
The documented hand confidence -1 means missing, while 0 is a present hand with
zero confidence. No wrist/normal placeholders from missing hands are consumed.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TextIO

import numpy as np

from duet.geometry.rotations import quaternion_xyzw_to_rotation
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import (
    Confidence,
    DistanceUnit,
    FrameId,
    ParticipantId,
    Provenance,
    SampleMetadata,
    SampleState,
    require_name,
)
from duet.schemas.episode import HandSample
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit

HAND_FORMAT_EVIDENCE = (
    "https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/hand_tracking"
)
MPS_TRANSFORM_TOLERANCE = 1e-7

# Project Aria hand_tracking_results.csv landmark IDs, not a MediaPipe ordering.
HAND_LANDMARK_NAMES = (
    "thumb_fingertip",
    "index_finger_fingertip",
    "middle_finger_fingertip",
    "ring_finger_fingertip",
    "pinky_finger_fingertip",
    "wrist_joint",
    "thumb_intermediate",
    "thumb_distal",
    "index_finger_proximal",
    "index_finger_intermediate",
    "index_finger_distal",
    "middle_finger_proximal",
    "middle_finger_intermediate",
    "middle_finger_distal",
    "ring_finger_proximal",
    "ring_finger_intermediate",
    "ring_finger_distal",
    "pinky_finger_proximal",
    "pinky_finger_intermediate",
    "pinky_finger_distal",
    "palm_center",
)


@dataclass(frozen=True)
class MpsTrajectorySample:
    """Timestamped T_world_device; quality_score remains an uninterpreted raw score."""

    timestamp: Timestamp
    utc_timestamp: Timestamp | None
    utc_timestamp_ns_raw: int | None
    graph_uid: str
    quality_score: float
    transform: RigidTransform
    participant_id: ParticipantId
    provenance: Provenance


@dataclass(frozen=True)
class MpsHandFrame:
    """One device-time row with both hands and original confidence sentinel values."""

    timestamp: Timestamp
    left: HandSample
    right: HandSample
    left_confidence_raw: float | None
    right_confidence_raw: float | None
    provenance: Provenance


@dataclass(frozen=True)
class CameraCalibration:
    """T_device_camera and model-native projection parameters, without reinterpretation."""

    label: str
    transform: RigidTransform
    projection: Mapping[str, object]
    image_size: tuple[int, int] | None
    calibrated: bool
    time_offset_seconds: float | None = None
    readout_seconds: float | None = None

    def __post_init__(self) -> None:
        # The parser freezes the numeric Params list; preserve additional verified
        # projection metadata without exposing a mutable top-level dictionary.
        object.__setattr__(self, "projection", MappingProxyType(dict(self.projection)))


@dataclass(frozen=True)
class CameraCalibrationSample:
    """Online calibration at a participant's tracking time, retaining UTC sentinel."""

    timestamp: Timestamp
    utc_timestamp: Timestamp | None
    utc_timestamp_ns_raw: int | None
    cameras: tuple[CameraCalibration, ...]
    provenance: Provenance


def _participant(participant: str) -> ParticipantId:
    if participant not in ("helper", "leader"):
        raise ValueError("CoMind participant must be the verified helper or leader role")
    return ParticipantId(f"participant/{participant}")


def device_clock(recording_id: str, participant: str) -> ClockDomain:
    """Device timestamps are comparable only within this recording and participant."""
    require_name(recording_id, "recording ID")
    _participant(participant)
    return ClockDomain(f"comind/{recording_id}/{participant}/device_time")


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise TypeError(f"{field} must be an integer")
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise TypeError(f"{field} must be a finite number")
    try:
        result = float(value)
    except (ValueError, OverflowError) as exc:
        raise ValueError(f"{field} must be a finite number") from exc
    if not isfinite(result):
        raise ValueError(f"{field} must be a finite number")
    return result


def _utc(
    row: Mapping[str, object], *, recording_id: str, participant: str, provenance: Provenance
) -> tuple[Timestamp | None, int | None]:
    if "utc_timestamp_ns" not in row:
        return None, None
    value = row["utc_timestamp_ns"]
    raw = None if value in (None, "") else _integer(value, "utc_timestamp_ns")
    # -1 is emitted by online calibration; retain it separately rather than make
    # it a usable timestamp. No agreement among participant UTC estimates is assumed.
    timestamp = Timestamp(
        None if raw == -1 else raw,
        TimeUnit.NANOSECONDS,
        ClockDomain(f"comind/{recording_id}/{participant}/utc_unverified"),
        provenance,
    )
    return timestamp, raw


def _rows(stream: TextIO, required: set[str]) -> Iterator[tuple[int, dict[str, str]]]:
    reader = csv.DictReader(stream)
    header = reader.fieldnames
    if not header or len(header) != len(set(header)):
        raise ValueError("CSV requires a nonempty header with unique column names")
    missing = required - set(header)
    if missing:
        raise ValueError(f"CSV missing required columns: {', '.join(sorted(missing))}")
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError(f"CSV row {reader.line_num} does not match the header width")
        yield reader.line_num, row


def _transform(
    translation: list[float],
    quaternion_xyzw: list[float],
    *,
    source: FrameId,
    destination: FrameId,
    provenance: Provenance,
) -> RigidTransform:
    matrix = np.eye(4)
    matrix[:3, :3] = quaternion_xyzw_to_rotation(
        quaternion_xyzw, norm_tolerance=MPS_TRANSFORM_TOLERANCE
    )
    matrix[:3, 3] = translation
    return RigidTransform(
        matrix,
        source,
        destination,
        DistanceUnit.METERS,
        provenance,
        tolerance=MPS_TRANSFORM_TOLERANCE,
    )


def iter_trajectory(
    stream: TextIO,
    *,
    participant: str,
    recording_id: str,
    provenance: Provenance,
    world_frame: FrameId | None = None,
) -> Iterator[MpsTrajectorySample]:
    """Parse trajectory CSV rows lazily as T_world_device in meters.

    The default world ID includes graph_uid, isolating disconnected graph islands.
    A caller-supplied world frame requires a constant graph_uid in this stream;
    cross-recording graph compatibility must be verified by the Multi-SLAM adapter.
    """
    participant_id = _participant(participant)
    clock = device_clock(recording_id, participant)
    source = FrameId(f"{participant}/device")
    if world_frame is not None and not isinstance(world_frame, FrameId):
        raise TypeError("world_frame must be an explicit FrameId")
    required = {"graph_uid", "tracking_timestamp_us", "quality_score"}
    required.update(f"t{axis}_world_device" for axis in "xyz")
    required.update(f"q{axis}_world_device" for axis in "xyzw")
    first_graph: str | None = None
    frames: dict[str, FrameId] = {}
    for line, row in _rows(stream, required):
        row_provenance = Provenance(provenance.source, f"CSV row {line}", (provenance,))
        try:
            graph_uid = row["graph_uid"]
            require_name(graph_uid, "graph_uid")
            if first_graph is None:
                first_graph = graph_uid
            if world_frame is not None and graph_uid != first_graph:
                raise ValueError(
                    "one explicit world_frame cannot contain different graph_uid values"
                )
            destination = world_frame
            if destination is None:
                destination = frames.get(graph_uid)
                if destination is None:
                    destination = FrameId(f"{participant}/mps_world/{graph_uid}")
                    frames[graph_uid] = destination
            timestamp = Timestamp(
                _integer(row["tracking_timestamp_us"], "tracking_timestamp_us"),
                TimeUnit.MICROSECONDS,
                clock,
                row_provenance,
            )
            utc, utc_raw = _utc(
                row, recording_id=recording_id, participant=participant, provenance=row_provenance
            )
            transform = _transform(
                [_number(row[f"t{axis}_world_device"], f"t{axis}_world_device") for axis in "xyz"],
                [_number(row[f"q{axis}_world_device"], f"q{axis}_world_device") for axis in "xyzw"],
                source=source,
                destination=destination,
                provenance=row_provenance,
            )
            yield MpsTrajectorySample(
                timestamp,
                utc,
                utc_raw,
                graph_uid,
                _number(row["quality_score"], "quality_score"),
                transform,
                participant_id,
                row_provenance,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{provenance.source}: trajectory row {line}: {exc}") from exc


def _hand(
    row: Mapping[str, str],
    side: str,
    *,
    participant_id: ParticipantId,
    timestamp: Timestamp,
    frame: FrameId,
    provenance: Provenance,
) -> tuple[HandSample, float | None]:
    raw_value = row[f"{side}_tracking_confidence"]
    raw_confidence = None if raw_value == "" else _number(raw_value, f"{side}_tracking_confidence")
    if raw_confidence is not None and raw_confidence != -1 and not 0 <= raw_confidence <= 1:
        raise ValueError(
            f"{side} confidence must be -1 (missing) or in the documented [0, 1] range"
        )
    points = None
    if raw_confidence == -1:
        metadata = SampleMetadata(
            state=SampleState.MISSING,
            reason="Project Aria tracking_confidence=-1; landmark/wrist/normal placeholders ignored",
        )
    else:
        confidence = (
            None
            if raw_confidence is None
            else Confidence(
                raw_confidence,
                Provenance(
                    HAND_FORMAT_EVIDENCE, "source confidence already in [0, 1]", (provenance,)
                ),
            )
        )
        values = [
            row[f"t{axis}_{side}_landmark_{index}_device"] for index in range(21) for axis in "xyz"
        ]
        if any(value == "" for value in values):
            metadata = SampleMetadata(
                state=SampleState.MISSING,
                confidence=confidence,
                reason="Incomplete 21-landmark hand: at least one coordinate is absent",
            )
        else:
            points = np.asarray([_number(value, f"{side} landmark coordinate") for value in values])
            points = points.reshape(21, 3)
            metadata = SampleMetadata(confidence=confidence)
    return (
        HandSample(
            participant_id,
            side,
            timestamp,
            frame,
            points,
            DistanceUnit.METERS,
            provenance,
            landmark_names=HAND_LANDMARK_NAMES,
            metadata=metadata,
        ),
        raw_confidence,
    )


def iter_hands(
    stream: TextIO,
    *,
    participant: str,
    recording_id: str,
    provenance: Provenance,
) -> Iterator[MpsHandFrame]:
    """Read both hands lazily, retaining all rows and explicit confidence/missing state.

    Missing confidence alone does not erase finite landmarks. Blank coordinates
    make the entire canonical hand missing; nonfinite/malformed values fail parsing.
    Wrist transforms and normals are outside the V0 landmark representation.
    """
    participant_id = _participant(participant)
    clock = device_clock(recording_id, participant)
    frame = FrameId(f"{participant}/device")
    required = {"tracking_timestamp_us", "left_tracking_confidence", "right_tracking_confidence"}
    required.update(
        f"t{axis}_{side}_landmark_{index}_device"
        for side in ("left", "right")
        for index in range(21)
        for axis in "xyz"
    )
    for line, row in _rows(stream, required):
        row_provenance = Provenance(provenance.source, f"CSV row {line}", (provenance,))
        try:
            timestamp = Timestamp(
                _integer(row["tracking_timestamp_us"], "tracking_timestamp_us"),
                TimeUnit.MICROSECONDS,
                clock,
                row_provenance,
            )
            left, left_confidence = _hand(
                row,
                "left",
                participant_id=participant_id,
                timestamp=timestamp,
                frame=frame,
                provenance=row_provenance,
            )
            right, right_confidence = _hand(
                row,
                "right",
                participant_id=participant_id,
                timestamp=timestamp,
                frame=frame,
                provenance=row_provenance,
            )
            yield MpsHandFrame(
                timestamp, left, right, left_confidence, right_confidence, row_provenance
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{provenance.source}: hand row {line}: {exc}") from exc


def _camera_calibration(
    camera: object,
    *,
    participant: str,
    image_size: object,
    readout_seconds: float | None,
    provenance: Provenance,
) -> CameraCalibration:
    if not isinstance(camera, dict):
        raise TypeError("camera calibration must be a JSON object")
    label = camera.get("Label")
    require_name(label, "camera Label")
    calibrated = camera.get("Calibrated")
    if not isinstance(calibrated, bool):
        raise TypeError("camera Calibrated must be a boolean")
    projection = camera.get("Projection")
    if not isinstance(projection, dict):
        raise TypeError("camera Projection must be an object")
    require_name(projection.get("Name"), "Projection.Name")
    params = projection.get("Params")
    if not isinstance(params, list) or not params:
        raise ValueError("Projection.Params must be a nonempty numeric array")
    projection = dict(projection)
    projection["Params"] = tuple(_number(value, "Projection.Params") for value in params)
    transform_json = camera.get("T_Device_Camera")
    if not isinstance(transform_json, dict):
        raise TypeError("T_Device_Camera must be a JSON object")
    translation = transform_json.get("Translation")
    quaternion = transform_json.get("UnitQuaternion")
    if not isinstance(translation, list) or len(translation) != 3:
        raise ValueError("T_Device_Camera.Translation must contain three coordinates")
    if (
        not isinstance(quaternion, list)
        or len(quaternion) != 2
        or not isinstance(quaternion[1], list)
        or len(quaternion[1]) != 3
    ):
        raise ValueError("T_Device_Camera.UnitQuaternion must be [w, [x, y, z]]")
    transform = _transform(
        [_number(value, "Translation") for value in translation],
        [_number(value, "UnitQuaternion") for value in [*quaternion[1], quaternion[0]]],
        source=FrameId(f"{participant}/{label}"),
        destination=FrameId(f"{participant}/device"),
        provenance=provenance,
    )
    size = None
    if image_size is not None:
        if not isinstance(image_size, list) or len(image_size) != 2:
            raise ValueError("camera ImageSizes entry must be [width, height]")
        size = tuple(_integer(value, "ImageSizes") for value in image_size)
        if any(value <= 0 for value in size):
            raise ValueError("camera ImageSizes must be positive")
    offset = camera.get("TimeOffsetSec_Device_Camera")
    offset_seconds = None if offset is None else _number(offset, "TimeOffsetSec_Device_Camera")
    return CameraCalibration(
        label, transform, projection, size, calibrated, offset_seconds, readout_seconds
    )


def iter_calibrations(
    stream: TextIO,
    *,
    participant: str,
    recording_id: str,
    provenance: Provenance,
) -> Iterator[CameraCalibrationSample]:
    """Read online calibration JSONL, preserving camera labels and native lens models.

    ImageSizes entries correspond by camera-list index, as in the official reader.
    ReadoutTimesSec index-duration pairs and optional camera time offsets are
    preserved without applying a time correction. No RGB label is selected
    implicitly and no MP4 image orientation is assumed.
    """
    clock = device_clock(recording_id, participant)
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        row_provenance = Provenance(provenance.source, f"JSONL line {line_number}", (provenance,))
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError("online calibration must be a JSON object")
            timestamp = Timestamp(
                _integer(row.get("tracking_timestamp_us"), "tracking_timestamp_us"),
                TimeUnit.MICROSECONDS,
                clock,
                row_provenance,
            )
            utc, utc_raw = _utc(
                row, recording_id=recording_id, participant=participant, provenance=row_provenance
            )
            camera_rows = row.get("CameraCalibrations")
            if not isinstance(camera_rows, list) or not camera_rows:
                raise ValueError("CameraCalibrations must be a nonempty array")
            image_sizes = row.get("ImageSizes")
            if image_sizes is not None and (
                not isinstance(image_sizes, list) or len(image_sizes) != len(camera_rows)
            ):
                raise ValueError("ImageSizes must contain one entry per CameraCalibrations entry")
            readouts: dict[int, float] = {}
            raw_readouts = row.get("ReadoutTimesSec", [])
            if not isinstance(raw_readouts, list):
                raise TypeError("ReadoutTimesSec must be an array of camera-index/duration pairs")
            for entry in raw_readouts:
                if not isinstance(entry, list) or len(entry) != 2:
                    raise ValueError("ReadoutTimesSec entries must be [camera_index, seconds]")
                index = _integer(entry[0], "ReadoutTimesSec camera index")
                readout = _number(entry[1], "ReadoutTimesSec duration")
                if index < 0 or index >= len(camera_rows) or index in readouts or readout < 0:
                    raise ValueError(
                        "ReadoutTimesSec requires unique valid indices and nonnegative times"
                    )
                readouts[index] = readout
            cameras = tuple(
                _camera_calibration(
                    camera,
                    participant=participant,
                    image_size=None if image_sizes is None else image_sizes[index],
                    readout_seconds=readouts.get(index),
                    provenance=row_provenance,
                )
                for index, camera in enumerate(camera_rows)
            )
            if len({camera.label for camera in cameras}) != len(cameras):
                raise ValueError("camera labels must be unique within a calibration sample")
            yield CameraCalibrationSample(timestamp, utc, utc_raw, cameras, row_provenance)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{provenance.source}: calibration line {line_number}: {exc}") from exc


def load_trajectory(
    path: str | Path,
    *,
    participant: str,
    recording_id: str,
    world_frame: FrameId | None = None,
) -> Iterator[MpsTrajectorySample]:
    """Stream a trajectory path; files are opened read-only and closed on exhaustion/close."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        yield from iter_trajectory(
            stream,
            participant=participant,
            recording_id=recording_id,
            world_frame=world_frame,
            provenance=Provenance(str(path)),
        )


def load_hands(path: str | Path, *, participant: str, recording_id: str) -> Iterator[MpsHandFrame]:
    """Stream a hand-tracking path without requiring a dataset-wide layout."""
    with Path(path).open(encoding="utf-8", newline="") as stream:
        yield from iter_hands(
            stream,
            participant=participant,
            recording_id=recording_id,
            provenance=Provenance(str(path)),
        )


def load_calibrations(
    path: str | Path, *, participant: str, recording_id: str
) -> Iterator[CameraCalibrationSample]:
    """Stream an online-calibration path without buffering the full recording."""
    with Path(path).open(encoding="utf-8") as stream:
        yield from iter_calibrations(
            stream,
            participant=participant,
            recording_id=recording_id,
            provenance=Provenance(str(path)),
        )
