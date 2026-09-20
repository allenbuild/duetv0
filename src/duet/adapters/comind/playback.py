"""Verified derived-cache bridge to the paired-frame renderer; no raw data reads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from duet.adapters.comind.frame_map import VideoVrsFrameMap, load_frame_map
from duet.adapters.comind.mps import HAND_LANDMARK_NAMES, MPS_TRANSFORM_TOLERANCE, device_clock
from duet.adapters.comind.semantics import EvidenceStatus, MappingEvidence
from duet.adapters.comind.temporal_validation import NO_RESIDUAL, TemporalThresholds
from duet.geometry.rotations import quaternion_xyzw_batch_to_rotations
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
from duet.schemas.episode import HandSample
from duet.schemas.time import Timestamp, TimeUnit
from duet.visualization.comind_handover import (
    FrameIndexBindings,
    MappedParticipantFrame,
    MappedRenderFrame,
)
from duet.visualization.rerun_episode import VerifiedImageGeometry

ROLES = ("helper", "leader")


def _derived(path: Path) -> Path:
    path = path.resolve()
    if any(a == "data" and b == "raw" for a, b in zip(path.parts, path.parts[1:])):
        raise ValueError("playback metadata/cache paths must be outside data/raw")
    return path


def _json(path: Path) -> dict:
    value = json.loads(_derived(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected derived JSON object")
    return value


def _digest(path: Path) -> str:
    with _derived(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _check_identity(expected_path: Path, recorded_path: str, expected_digest: str) -> None:
    if _derived(Path(recorded_path)) != _derived(expected_path):
        raise ValueError("derived cache identity points outside the expected recording location")
    if _digest(expected_path) != expected_digest:
        raise ValueError("derived cache hash differs from the temporal-validation generation")


def _evidence(value: Mapping) -> MappingEvidence:
    return MappingEvidence(
        value["name"],
        EvidenceStatus(value["status"]),
        value["statement"],
        tuple(value.get("evidence", ())),
        tuple(value.get("missing_assets", ())),
    )


def _timestamp(value: int | None, recording: str, role: str, provenance: Provenance) -> Timestamp:
    return Timestamp(value, TimeUnit.NANOSECONDS, device_clock(recording, role), provenance)


def _validate_aligned(
    arrays: Mapping[str, np.ndarray],
    mapping: VideoVrsFrameMap,
    generation: str,
    thresholds: TemporalThresholds,
) -> None:
    count = mapping.frame_count
    if str(arrays["generation_id"].item()) != generation:
        raise ValueError("aligned cache belongs to a different validation generation")
    for name, shape in (
        ("mp4_frame_index", (count,)),
        ("device_timestamp_ns", (count,)),
        ("mapping_status", (count,)),
        ("mapping_confidence", (count,)),
        ("mapping_verified", (count,)),
        ("pose_source_index", (count,)),
        ("pose_residual_ns", (count,)),
        ("pose_accepted", (count,)),
        ("pose_translation_m", (count, 3)),
        ("pose_quaternion_xyzw", (count, 4)),
        ("hand_source_index", (count,)),
        ("hand_residual_ns", (count,)),
        ("hand_accepted", (count,)),
        ("hand_points_shared_m", (count, 2, 21, 3)),
        ("hand_confidence", (count, 2)),
        ("hand_present", (count, 2)),
        ("hand_high_confidence", (count, 2)),
        ("hand_pose_residual_ns", (count,)),
        ("hand_pose_accepted", (count,)),
    ):
        if arrays[name].shape != shape:
            raise ValueError(f"aligned {name} has incorrect dimensions")
    for name, expected in (
        ("mp4_frame_index", mapping.mp4_frame_index),
        ("device_timestamp_ns", mapping.vrs_device_timestamp_ns),
        ("mapping_status", mapping.status),
        ("mapping_confidence", mapping.match_confidence),
        ("mapping_verified", mapping.status == "VERIFIED"),
    ):
        if not np.array_equal(arrays[name], expected):
            raise ValueError(f"aligned {name} differs from current validated frame map")
    for name in (
        "pose_source_index",
        "hand_source_index",
        "pose_residual_ns",
        "hand_residual_ns",
        "hand_pose_residual_ns",
        "device_timestamp_ns",
    ):
        if arrays[name].dtype.kind != "i" or arrays[name].dtype.itemsize != 8:
            raise ValueError(f"{name} must retain signed int64 values")
    for name in (
        "mapping_verified",
        "pose_accepted",
        "hand_accepted",
        "hand_pose_accepted",
        "hand_present",
        "hand_high_confidence",
    ):
        if arrays[name].dtype.kind != "b":
            raise ValueError(f"{name} must be Boolean")
    for prefix, gap in (("pose", thresholds.pose_max_gap_ns), ("hand", thresholds.hand_max_gap_ns)):
        source = arrays[f"{prefix}_source_index"]
        residual = arrays[f"{prefix}_residual_ns"]
        candidate = source >= 0
        if np.any((residual != NO_RESIDUAL) != candidate):
            raise ValueError("candidate availability disagrees with residual availability")
        expected = candidate & arrays["mapping_verified"]
        expected[candidate] &= np.abs(residual[candidate]) <= gap
        if not np.array_equal(arrays[f"{prefix}_accepted"], expected):
            raise ValueError(f"{prefix} accepted mask violates local residual gate")
    pose_ok = arrays["pose_accepted"]
    if not np.isfinite(arrays["pose_translation_m"][pose_ok]).all():
        raise ValueError("accepted pose has nonfinite translation")
    if pose_ok.any():
        quaternion_xyzw_batch_to_rotations(
            arrays["pose_quaternion_xyzw"][pose_ok], norm_tolerance=MPS_TRANSFORM_TOLERANCE
        )
    if (
        not np.isnan(arrays["pose_translation_m"][~pose_ok]).all()
        or not np.isnan(arrays["pose_quaternion_xyzw"][~pose_ok]).all()
    ):
        raise ValueError("rejected pose must not retain a displayable payload")
    points = arrays["hand_points_shared_m"]
    present = np.isfinite(points).all(axis=(2, 3))
    absent = np.isnan(points).all(axis=(2, 3))
    if not (present | absent).all() or not np.array_equal(arrays["hand_present"], present):
        raise ValueError("hand payload availability disagrees with source state")
    if np.any(present & ~arrays["hand_pose_accepted"][:, None]):
        raise ValueError("present hand lacks an accepted secondary hand-to-pose chain")
    secondary = arrays["hand_pose_accepted"]
    if np.any(secondary & ~arrays["hand_accepted"]):
        raise ValueError("secondary hand pose cannot be accepted without its source hand")
    if np.any(np.abs(arrays["hand_pose_residual_ns"][secondary]) > thresholds.hand_pose_max_gap_ns):
        raise ValueError("secondary hand-to-pose residual exceeds its gate")
    confidence = arrays["hand_confidence"]
    if np.isinf(confidence).any() or np.any(
        np.isfinite(confidence) & (confidence != -1) & ((confidence < 0) | (confidence > 1))
    ):
        raise ValueError("invalid source hand confidence")
    if not np.array_equal(
        arrays["hand_high_confidence"], present & (confidence >= thresholds.minimum_hand_confidence)
    ):
        raise ValueError("high-confidence hand mask violates the configured threshold")


def _cameras(directory: Path, recording_id: str) -> tuple[dict, dict]:
    extrinsics, geometry = {}, {}
    for role in ROLES:
        path = directory / "vrs" / f"{role}_export_rgb_calibration.json"
        if not path.exists():
            continue
        document = _json(path)
        if document.get("status") != "VERIFIED":
            continue
        if document.get("recording_id") != recording_id or document.get("participant") != role:
            raise ValueError("export calibration belongs to another participant/recording")
        if (
            document["camera_frame"] != f"{role}/camera-rgb"
            or document["device_frame"] != f"{role}/device"
        ):
            raise ValueError("export camera frame identity is incompatible")
        provenance = Provenance(document["provenance_source"], document["provenance_detail"])
        extrinsics[role] = RigidTransform(
            np.asarray(document["T_device_camera"]),
            FrameId(document["camera_frame"]),
            FrameId(document["device_frame"]),
            DistanceUnit.METERS,
            provenance,
            tolerance=MPS_TRANSFORM_TOLERANCE,
        )
        geometry[role] = VerifiedImageGeometry(
            FrameId(document["camera_frame"]),
            tuple(document["resolution"]),
            np.asarray(document["boundary_rays_camera"]),
            document["projection_model"],
            document["verification"],
            provenance,
        )
    return extrinsics, geometry


@dataclass(frozen=True)
class PlaybackBundle:
    bindings: FrameIndexBindings
    frame_maps: Mapping[str, VideoVrsFrameMap]
    start_index: int
    end_index: int
    trajectories: Mapping[str, np.ndarray]
    selection: Mapping[str, object]
    aligned_arrays: Mapping[str, Mapping[str, np.ndarray]]
    annotations: tuple[Mapping[str, object], ...]
    provenance: Provenance

    def with_annotation(self, annotation_id: str, context_frames: int = 120) -> PlaybackBundle:
        """Select another bound source interval using the same checked in-memory caches.

        Source segment keys remain strings, including their leading zeros. The
        requested context is clipped only at the paired videos' actual bounds;
        both boundary frames are included. Every frame in both selected maps
        must remain VERIFIED. No source files, reports or images are reopened.
        """
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("annotation_id must be an exact nonempty source segment key")
        if type(context_frames) is not int or context_frames < 0:
            raise ValueError("context_frames must be a nonnegative integer")
        matches = [item for item in self.annotations if item.get("annotation_id") == annotation_id]
        if len(matches) != 1:
            raise ValueError("annotation ID must identify exactly one bound source annotation")
        annotation = matches[0]
        start, end = annotation["start_frame"], annotation["end_frame"]
        count = self.frame_maps["helper"].frame_count
        if type(start) is not int or type(end) is not int or not 0 <= start <= end < count:
            raise ValueError("annotation boundaries are outside the paired video frame range")
        clip_start, clip_end = max(0, start - context_frames), min(count - 1, end + context_frames)
        window = slice(clip_start, clip_end + 1)
        trajectories = {}
        for role in ROLES:
            mapping, arrays = self.frame_maps[role], self.aligned_arrays[role]
            if mapping.frame_count != count or (mapping.recording_id, mapping.participant) != (
                self.bindings.recording_id,
                role,
            ):
                raise ValueError("paired map identities or frame counts differ")
            if (
                not (mapping.status[window] == "VERIFIED").all()
                or not arrays["mapping_verified"][window].all()
            ):
                raise ValueError("every selected context frame requires a VERIFIED mapping")
            if not np.array_equal(
                arrays["device_timestamp_ns"][window], mapping.vrs_device_timestamp_ns[window]
            ) or not np.array_equal(arrays["mapping_status"][window], mapping.status[window]):
                raise ValueError("selected aligned cache differs from the validated frame map")
            points = arrays["pose_translation_m"][window]
            trajectories[role] = points[arrays["pose_accepted"][window]]
        selection = dict(annotation)
        selection.update(
            context_start_frame=clip_start,
            context_end_frame=clip_end,
            eligible_verified_context=True,
        )
        return replace(
            self,
            start_index=clip_start,
            end_index=clip_end,
            trajectories=trajectories,
            selection=selection,
        )

    def iter_frames(self) -> Iterator[MappedRenderFrame]:
        """Yield canonical payloads only within the fully verified selected clip."""
        for index in range(self.start_index, self.end_index + 1):
            participants = []
            for role in ROLES:
                mapping, arrays = self.frame_maps[role], self.aligned_arrays[role]
                query = mapping.timestamp_at(index)
                if query is None:
                    raise ValueError("selected clip unexpectedly lost its VERIFIED frame mapping")
                query_ns = int(mapping.vrs_device_timestamp_ns[index])
                pose, pose_timestamp = None, None
                if arrays["pose_source_index"][index] >= 0:
                    pose_timestamp = _timestamp(
                        query_ns + int(arrays["pose_residual_ns"][index]),
                        self.bindings.recording_id,
                        role,
                        self.provenance,
                    )
                if arrays["pose_accepted"][index]:
                    matrix = np.eye(4)
                    matrix[:3, :3] = quaternion_xyzw_batch_to_rotations(
                        arrays["pose_quaternion_xyzw"][index : index + 1],
                        norm_tolerance=MPS_TRANSFORM_TOLERANCE,
                    )[0]
                    matrix[:3, 3] = arrays["pose_translation_m"][index]
                    pose = RigidTransform(
                        matrix,
                        FrameId(f"{role}/device"),
                        self.bindings.world_frame,
                        DistanceUnit.METERS,
                        self.provenance,
                        tolerance=MPS_TRANSFORM_TOLERANCE,
                    )
                hands, warnings = [], []
                if pose is None:
                    warnings.append("pose missing or rejected by local timestamp gap")
                hand_time = (
                    query_ns + int(arrays["hand_residual_ns"][index])
                    if arrays["hand_source_index"][index] >= 0
                    else None
                )
                for side_index, side in enumerate(("left", "right")):
                    score = arrays["hand_confidence"][index, side_index]
                    confidence = (
                        Confidence(float(score), self.provenance) if 0 <= score <= 1 else None
                    )
                    present = bool(arrays["hand_high_confidence"][index, side_index])
                    reason = (
                        None
                        if present
                        else "Missing, gap-rejected, or below-confidence-threshold hand"
                    )
                    if not present:
                        warnings.append(f"{side} hand: {reason}")
                    hands.append(
                        HandSample(
                            participant_id=ParticipantId(f"participant/{role}"),
                            hand_id=side,
                            timestamp=_timestamp(
                                hand_time, self.bindings.recording_id, role, self.provenance
                            ),
                            frame=self.bindings.world_frame,
                            points=arrays["hand_points_shared_m"][index, side_index]
                            if present
                            else None,
                            unit=DistanceUnit.METERS,
                            provenance=self.provenance,
                            landmark_names=HAND_LANDMARK_NAMES,
                            metadata=SampleMetadata(
                                SampleState.PRESENT if present else SampleState.UNKNOWN,
                                confidence=confidence,
                                reason=reason,
                            ),
                        )
                    )
                participants.append(
                    MappedParticipantFrame(
                        role,
                        query,
                        int(mapping.vrs_rgb_frame_index[index]),
                        EvidenceStatus.VERIFIED,
                        float(mapping.match_confidence[index]),
                        pose,
                        pose_timestamp,
                        tuple(hands),
                        tuple(warnings),
                    )
                )
            active = tuple(
                {
                    key: item[key]
                    for key in (
                        "annotation_index",
                        "annotation_id",
                        "start_frame",
                        "end_frame",
                        "initiator_source_label",
                        "delivering_flow_source_label",
                        "object_category_level_1",
                        "object_category_level_2",
                        "object_category_level_3",
                        "initiation_type",
                    )
                }
                for item in self.annotations
                if item["start_frame"] <= index <= item["end_frame"]
            )
            yield MappedRenderFrame(index, tuple(participants), active)


def load_playback(
    processed_root: Path,
    report_path: Path,
    recording_id: str,
    *,
    annotation_id: str | None = None,
    context_frames: int = 120,
) -> PlaybackBundle:
    """Reject stale/wrong-role caches and create a bounded canonical clip iterator.

    Every aligned payload is tied by hash and generation to the validation
    report. Every map is revalidated and checked against exact native timestamp
    arrays. These checks read derived caches only; videos, VRS and CSVs remain
    unopened. Inferred or ambiguous rows cannot enter the selected context.
    An explicit annotation selects another bound interval, including when the
    report's preferred context is unavailable. Its entire requested context
    must pass the same map, native timestamp, cache and evidence checks.
    """
    directory = _derived(processed_root / recording_id)
    report = _json(report_path)
    if (
        report.get("recording_id") != recording_id
        or report.get("timeline") != "comind_sync_frame_index"
    ):
        raise ValueError("temporal report must identify this recording's paired-frame timeline")
    generation = report["generation_id"]
    if not isinstance(generation, str) or not generation:
        raise ValueError("temporal report requires a generation identity")
    thresholds = TemporalThresholds(**report["thresholds"])
    shared = report["shared_world"]
    if shared["status"] != "VERIFIED" or shared["frame"] != (
        f"comind/{recording_id}/multislam/{shared['graph_uid']}"
    ):
        raise ValueError("temporal report does not establish the expected shared world")
    ranking = report["handover_ranking"]
    if annotation_id is None:
        selected_index = ranking["selected_annotation_index"]
        if selected_index is None or ranking["selection_status"] != "VERIFIED_CONTEXT_AVAILABLE":
            raise ValueError("no handover has a fully VERIFIED playback context")
        candidates = [
            item for item in ranking["handovers"] if item["annotation_index"] == selected_index
        ]
        if len(candidates) != 1 or not candidates[0]["eligible_verified_context"]:
            raise ValueError("selected handover is missing, duplicated or not eligible")
        selection = dict(candidates[0])
        start, end = selection["context_start_frame"], selection["context_end_frame"]
    else:
        if not isinstance(annotation_id, str) or not annotation_id:
            raise ValueError("annotation_id must be an exact nonempty source segment key")
        if type(context_frames) is not int or context_frames < 0:
            raise ValueError("context_frames must be a nonnegative integer")
        candidates = [
            item for item in ranking["handovers"] if item.get("annotation_id") == annotation_id
        ]
        if len(candidates) != 1:
            raise ValueError("annotation ID must identify exactly one bound source annotation")
        selection = dict(candidates[0])
        event_start, event_end = selection["start_frame"], selection["end_frame"]
        if (
            type(event_start) is not int
            or type(event_end) is not int
            or not 0 <= event_start <= event_end
        ):
            raise ValueError("annotation boundaries must be ordered integer frames")
        start, end = max(0, event_start - context_frames), event_end + context_frames
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError("selected context must have ordered integer frame bounds")
    maps, aligned, trajectories = {}, {}, {}
    for role in ROLES:
        map_path = directory / "frame_maps" / f"{role}_frame_map.npz"
        native_path = directory / "vrs" / f"{role}_rgb_metadata.npz"
        cache_path = directory / "frame_validation" / f"{role}_aligned.npz"
        inputs = report["frame_map_inputs"][role]
        _check_identity(map_path, inputs["path"], inputs["sha256"])
        _check_identity(
            map_path.with_suffix(".json"), inputs["metadata_path"], inputs["metadata_sha256"]
        )
        _check_identity(
            native_path, inputs["native_metadata_path"], inputs["native_metadata_sha256"]
        )
        identity = report["aligned_cache_identities"][role]
        _check_identity(cache_path, identity["path"], identity["sha256"])
        mapping = load_frame_map(map_path)
        if (mapping.recording_id, mapping.participant) != (recording_id, role):
            raise ValueError("frame map role/recording identity differs")
        if annotation_id is not None:
            if selection["end_frame"] >= mapping.frame_count:
                raise ValueError("annotation boundaries are outside the paired video frame range")
            end = min(end, mapping.frame_count - 1)
        with np.load(native_path, allow_pickle=False) as source:
            native = source["device_timestamps_ns"]
        assigned = mapping.status != "UNRESOLVED"
        source_indices = mapping.vrs_rgb_frame_index[assigned]
        if (
            np.any(source_indices >= len(native))
            or np.any(source_indices < 0)
            or not np.array_equal(native[source_indices], mapping.vrs_device_timestamp_ns[assigned])
        ):
            raise ValueError("mapped timestamps differ from exact native VRS capture metadata")
        if end >= mapping.frame_count or not (mapping.status[start : end + 1] == "VERIFIED").all():
            raise ValueError("every selected clip frame requires a VERIFIED VRS correspondence")
        with np.load(cache_path, allow_pickle=False) as source:
            arrays = {name: source[name] for name in source.files}
        _validate_aligned(arrays, mapping, generation, thresholds)
        if report["participants"][role]["world_frame"] != shared["frame"]:
            raise ValueError("participant aligned poses do not share the verified world")
        if report["participants"][role]["device_clock"] != device_clock(recording_id, role).name:
            raise ValueError("participant cache uses the wrong local device clock")
        maps[role], aligned[role] = mapping, arrays
        points = arrays["pose_translation_m"]
        trajectories[role] = points[np.isfinite(points).all(axis=1)]
    if maps["helper"].frame_count != maps["leader"].frame_count:
        raise ValueError("paired video frame counts differ")
    if annotation_id is not None:
        selection.update(
            context_start_frame=start, context_end_frame=end, eligible_verified_context=True
        )
    alignment = _evidence(ranking["paired_video_alignment"])
    annotation_evidence = _evidence(report["annotation_binding"])
    if annotation_evidence.name != "annotation_to_comind_sync_frame_index":
        raise ValueError("annotation binding evidence has the wrong scope")
    annotation_evidence.require_verified()
    extrinsics, geometry = _cameras(directory, recording_id)
    bindings = FrameIndexBindings(
        recording_id,
        FrameId(shared["frame"]),
        alignment,
        thresholds.pose_max_gap_ns / 1e9,
        thresholds.hand_max_gap_ns / 1e9,
        thresholds.minimum_hand_confidence,
        extrinsics,
        geometry,
    )
    return PlaybackBundle(
        bindings,
        maps,
        start,
        end,
        trajectories,
        selection,
        aligned,
        tuple(ranking["handovers"]),
        Provenance(str(_derived(report_path)), "Hash/generation-checked derived temporal assembly"),
    )
