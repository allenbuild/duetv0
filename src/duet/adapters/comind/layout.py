"""Discover the verified CoMind recording layout without filename heuristics."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from uuid import UUID

from duet.schemas.common import ParticipantId

ROLES = ("helper", "leader")


@dataclass(frozen=True)
class ParticipantLayout:
    """Known participant assets; MPS and Multi-SLAM remain distinct solutions."""

    role: str
    participant_id: ParticipantId
    video_path: Path
    trajectory_path: Path | None
    calibration_path: Path | None
    hands_path: Path
    multislam_index: str
    multislam_dir: Path | None
    multislam_zip: Path | None


@dataclass(frozen=True)
class RecordingLayout:
    """Resolved recording paths, with participant mapping read from the source JSON."""

    dataset_root: Path
    recording_uuid: str
    recording_root: Path
    participants: Mapping[str, ParticipantLayout]
    mapping_path: Path
    scan_transform_path: Path | None
    scan_ply_paths: tuple[Path, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "participants", MappingProxyType(dict(self.participants)))


def _require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"required CoMind file is missing: {path}")
    return path


def discover_recording(
    dataset_root: str | Path, recording_uuid: str, *, minimal: bool = False
) -> RecordingLayout:
    """Validate explicitly documented paths; never search for lookalike assets.

    This performs read-only path/mapping inspection. Archive existence does not
    prove integrity: Multi-SLAM readers separately validate ZIP contents. No
    archive is extracted and no directory is created by discovery.

    ``minimal=True`` requires only synchronized videos, hands, the explicit VRS
    role mapping, and shared-world trajectories. Independent MPS trajectories,
    online calibration, and scans are not consumed by the synchronized demo;
    native RGB calibration is separately recovered from bounded VRS metadata.
    """
    from duet.adapters.comind.multislam import parse_vrs_mapping

    if not isinstance(recording_uuid, str) or str(UUID(recording_uuid)) != recording_uuid:
        raise ValueError("recording_uuid must be a canonical UUID string")
    dataset_root = Path(dataset_root).resolve()
    recording_root = dataset_root / "recordings" / recording_uuid
    if not recording_root.is_dir():
        raise FileNotFoundError(f"CoMind recording directory is missing: {recording_root}")
    mapping_path = _require_file(recording_root / "multislam_output" / "vrs_to_multi_slam.json")
    mapping = parse_vrs_mapping(mapping_path, recording_id=recording_uuid)
    participants: dict[str, ParticipantLayout] = {}
    for role in ROLES:
        mps_root = recording_root / f"mps_{role}_trimmed_vrs"
        multislam_root = recording_root / "multislam_output" / mapping[role]
        directory = multislam_root / "slam"
        archive = multislam_root / "slam.zip"
        has_directory = (directory / "closed_loop_trajectory.csv").is_file()
        if not has_directory and not archive.is_file():
            raise FileNotFoundError(
                f"missing Multi-SLAM trajectory directory or ZIP: {multislam_root}"
            )
        participants[role] = ParticipantLayout(
            role=role,
            participant_id=ParticipantId(f"participant/{role}"),
            video_path=_require_file(recording_root / "mp4s" / f"{role}_trimmed_sync.mp4"),
            trajectory_path=(
                None if minimal else _require_file(mps_root / "slam" / "closed_loop_trajectory.csv")
            ),
            calibration_path=(
                None if minimal else _require_file(mps_root / "slam" / "online_calibration.jsonl")
            ),
            hands_path=_require_file(mps_root / "hand_tracking" / "hand_tracking_results.csv"),
            multislam_index=mapping[role],
            multislam_dir=directory if has_directory else None,
            multislam_zip=archive if archive.is_file() else None,
        )
    return RecordingLayout(
        dataset_root=dataset_root,
        recording_uuid=recording_uuid,
        recording_root=recording_root,
        participants=participants,
        mapping_path=mapping_path,
        scan_transform_path=(
            None
            if minimal
            else _require_file(recording_root / "scan" / "T_ariaWorld_from_blkWorld.txt")
        ),
        scan_ply_paths=()
        if minimal
        else tuple(
            _require_file(recording_root / "scan" / name)
            for name in ("aria_semidense_points.ply", "blk_scan_aria_aligned.ply")
        ),
    )
