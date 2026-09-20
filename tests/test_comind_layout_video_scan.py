"""Tiny synthetic fixtures for path discovery, lazy video access and scan headers."""

import json
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import pytest

from duet.adapters.comind.layout import discover_recording
from duet.adapters.comind.scan import inspect_scan, load_scan_transform, read_ply_header
from duet.adapters.comind.video import EgoVideo
from duet.qc.result import QCStatus
from duet.schemas.common import ParticipantId
from duet.synchronization.matching import MatchStatus

RECORDING_ID = "12345678-1234-4234-9234-123456789abc"


def synthetic_layout(root: Path) -> Path:
    recording = root / "recordings" / RECORDING_ID
    mapping = {"helper": "7", "leader": "4"}
    files = [
        "multislam_output/vrs_to_multi_slam.json",
        "scan/T_ariaWorld_from_blkWorld.txt",
        "scan/aria_semidense_points.ply",
        "scan/blk_scan_aria_aligned.ply",
    ]
    for role, index in mapping.items():
        files.extend(
            [
                f"mp4s/{role}_trimmed_sync.mp4",
                f"mps_{role}_trimmed_vrs/slam/closed_loop_trajectory.csv",
                f"mps_{role}_trimmed_vrs/slam/online_calibration.jsonl",
                f"mps_{role}_trimmed_vrs/hand_tracking/hand_tracking_results.csv",
                f"multislam_output/{index}/slam.zip",
            ]
        )
    files.append("multislam_output/4/slam/closed_loop_trajectory.csv")
    for relative in files:
        path = recording / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"")
    (recording / "multislam_output/vrs_to_multi_slam.json").write_text(
        json.dumps(
            {
                f"{RECORDING_ID}/trimmed_vrs/{role}_trimmed.vrs": index
                for role, index in mapping.items()
            }
        )
    )
    return recording


def test_layout_uses_exact_ego_names_and_actual_mapping(tmp_path):
    recording = synthetic_layout(tmp_path)
    layout = discover_recording(tmp_path, RECORDING_ID)
    assert layout.recording_root == recording
    helper = layout.participants["helper"]
    leader = layout.participants["leader"]
    assert helper.participant_id == ParticipantId("participant/helper")
    assert helper.video_path.name == "helper_trimmed_sync.mp4"
    assert leader.video_path.name == "leader_trimmed_sync.mp4"
    assert helper.multislam_index == "7"
    assert helper.multislam_dir is None
    assert helper.multislam_zip == recording / "multislam_output/7/slam.zip"
    assert leader.multislam_dir == recording / "multislam_output/4/slam"
    assert leader.multislam_zip is not None
    with pytest.raises(TypeError):
        layout.participants["gopro"] = helper


def test_layout_rejects_missing_ego_instead_of_substituting_gopro(tmp_path):
    recording = synthetic_layout(tmp_path)
    (recording / "mp4s/helper_trimmed_sync.mp4").unlink()
    (recording / "mp4s/gopro_front_sync.mp4").write_bytes(b"")
    with pytest.raises(FileNotFoundError, match="helper_trimmed_sync"):
        discover_recording(tmp_path, RECORDING_ID)


def test_layout_requires_a_multislam_source(tmp_path):
    recording = synthetic_layout(tmp_path)
    (recording / "multislam_output/7/slam.zip").unlink()
    with pytest.raises(FileNotFoundError, match="Multi-SLAM"):
        discover_recording(tmp_path, RECORDING_ID)


def test_minimal_layout_has_no_unused_mps_or_scan_dependencies(tmp_path):
    recording = synthetic_layout(tmp_path)
    for path in (recording / "scan").iterdir():
        path.unlink()
    for role in ("helper", "leader"):
        for path in (recording / f"mps_{role}_trimmed_vrs" / "slam").iterdir():
            path.unlink()
    layout = discover_recording(tmp_path, RECORDING_ID, minimal=True)
    assert layout.scan_transform_path is None
    assert layout.scan_ply_paths == ()
    assert all(p.trajectory_path is None for p in layout.participants.values())
    assert all(p.calibration_path is None for p in layout.participants.values())
    assert layout.participants["helper"].multislam_index == "7"
    with pytest.raises(FileNotFoundError):
        discover_recording(tmp_path, RECORDING_ID)


def test_layout_rejects_path_traversal_recording_id(tmp_path):
    with pytest.raises(ValueError):
        discover_recording(tmp_path, "../other")


@pytest.fixture
def tiny_video(tmp_path):
    path = tmp_path / "helper_trimmed_sync.mp4"
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        stream.codec_context.time_base = Fraction(1, 10)
        for index in range(5):
            frame = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), index * 40, dtype=np.uint8), format="rgb24"
            )
            frame.pts = index
            frame.time_base = Fraction(1, 10)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return EgoVideo(
        path, participant_id=ParticipantId("participant/helper"), recording_uuid=RECORDING_ID
    )


def test_video_constructor_is_lazy(tmp_path):
    video = EgoVideo(
        tmp_path / "not_downloaded.mp4",
        participant_id=ParticipantId("participant/helper"),
        recording_uuid=RECORDING_ID,
    )
    assert "video_pts" in video.clock_domain.name
    with pytest.raises(FileNotFoundError):
        video.metadata()


def test_video_metadata_retains_exact_timebase(tiny_video):
    metadata = tiny_video.metadata()
    assert (metadata.width, metadata.height, metadata.frame_count) == (16, 16, 5)
    assert isinstance(metadata.time_base, Fraction)
    assert metadata.duration_seconds == Fraction(1, 2)
    assert metadata.average_rate == 10
    assert metadata.start_pts == 0


def test_video_lookup_preserves_pts_and_deterministically_selects_earlier_tie(tiny_video):
    match = tiny_video.nearest_frame(Fraction(3, 20), max_gap_seconds=Fraction(1, 20))
    assert match.accepted
    assert match.candidate.seconds == Fraction(1, 10)
    assert match.candidate.seconds == match.candidate.pts * match.candidate.time_base
    assert match.residual_seconds == Fraction(-1, 20)
    assert match.candidate.frame.width == 16


def test_video_lookup_enforces_maximum_gap_and_preserves_rejected_candidate(tiny_video):
    match = tiny_video.nearest_frame(Fraction(3, 20), max_gap_seconds=Fraction(1, 100))
    assert not match.accepted
    assert match.status is MatchStatus.GAP_EXCEEDED
    assert match.candidate.seconds == Fraction(1, 10)
    assert match.residual_seconds == Fraction(-1, 20)


def test_video_lookup_can_find_first_and_last_frames_without_index_alignment(tiny_video):
    first = tiny_video.nearest_frame(Fraction(-1, 10), max_gap_seconds=1)
    last = tiny_video.nearest_frame(100, max_gap_seconds=100)
    assert first.candidate.seconds == 0
    assert last.candidate.seconds == Fraction(2, 5)


def test_video_lookup_rejects_invalid_queries_and_gaps(tiny_video):
    with pytest.raises(TypeError):
        tiny_video.nearest_frame(True, max_gap_seconds=1)
    with pytest.raises(ValueError):
        tiny_video.nearest_frame(0, max_gap_seconds=-1)


def binary_ply(path: Path) -> bytes:
    header = (
        b"ply\nformat binary_little_endian 1.0\ncomment synthetic fixture\n"
        b"element vertex 2\nproperty double x\nproperty double y\nproperty double z\n"
        b"property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    path.write_bytes(header + b"\xff\xff\xff")
    return header


def test_ply_reads_only_header_and_preserves_encoding_properties(tmp_path):
    path = tmp_path / "synthetic.ply"
    source_header = binary_ply(path)
    header = read_ply_header(path)
    assert header.encoding == "binary_little_endian"
    assert header.version == "1.0"
    assert header.vertex_count == 2
    assert header.header_bytes == len(source_header)
    assert header.comments == ("synthetic fixture",)
    assert [(prop.name, prop.data_type) for prop in header.elements[0].properties] == [
        ("x", "double"),
        ("y", "double"),
        ("z", "double"),
        ("red", "uchar"),
        ("green", "uchar"),
        ("blue", "uchar"),
    ]


def test_ply_header_limits_and_missing_terminator_are_explicit(tmp_path):
    path = tmp_path / "bad.ply"
    binary_ply(path)
    with pytest.raises(ValueError, match="byte limits"):
        read_ply_header(path, max_header_bytes=20)
    path.write_bytes(b"ply\nformat ascii 1.0\nelement vertex 0\n")
    with pytest.raises(ValueError, match="end_header"):
        read_ply_header(path)


def test_scan_validates_matrix_without_binding_units_frames_or_graph(tmp_path):
    matrix_path = tmp_path / "T_ariaWorld_from_blkWorld.txt"
    matrix = np.eye(4)
    matrix[:3, 3] = [1, 2, 3]
    np.savetxt(matrix_path, matrix)
    ply_path = tmp_path / "blk_scan_aria_aligned.ply"
    binary_ply(ply_path)
    metadata = inspect_scan(matrix_path, (ply_path,))
    registration = metadata.registration
    assert registration.validation.status is QCStatus.PASS
    assert not registration.semantics_verified
    assert registration.source_frame is None
    assert registration.destination_frame is None
    assert registration.unit is None
    assert registration.filename_claim == "T_ariaWorld_from_blkWorld"
    assert not metadata.clouds[0].frame_semantics_verified
    assert metadata.clouds[0].unit is None
    np.testing.assert_array_equal(registration.matrix, matrix)
    with pytest.raises(ValueError):
        registration.matrix.setflags(write=True)


@pytest.mark.parametrize("matrix", [np.diag([-1, 1, 1, 1]), np.eye(3), np.eye(4) * 2])
def test_invalid_scan_transform_is_rejected(tmp_path, matrix):
    path = tmp_path / "transform.txt"
    np.savetxt(path, matrix)
    with pytest.raises(ValueError, match="invalid scan transform"):
        load_scan_transform(path)
