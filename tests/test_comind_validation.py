"""End-to-end validation against a tiny synthetic recording; no real raw assets."""

import csv
import json
from fractions import Fraction

import numpy as np
import pytest

from duet.adapters.comind.validation import (
    ensure_output_outside_raw,
    parse_selected_poses,
    raw_inventory,
    run_validation,
)
from duet.adapters.comind.video import EgoVideo, VideoMetadata

RECORDING = "00000000-0000-0000-0000-000000000001"


def _csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trajectory(time, x, graph):
    return {
        "graph_uid": graph,
        "tracking_timestamp_us": time,
        "utc_timestamp_ns": 1_700_000_000_000_000_001 + time,
        "tx_world_device": x,
        "ty_world_device": 0,
        "tz_world_device": 0,
        "qx_world_device": 0,
        "qy_world_device": 0,
        "qz_world_device": 0,
        "qw_world_device": 1,
        "quality_score": 1,
    }


def _hand(time, left_confidence, right_confidence):
    return {
        "tracking_timestamp_us": time,
        "left_tracking_confidence": left_confidence,
        "right_tracking_confidence": right_confidence,
        **{
            f"t{axis}_{side}_landmark_{index}_device": 1
            for side in ("left", "right")
            for index in range(21)
            for axis in "xyz"
        },
    }


def _calibration(time):
    return {
        "tracking_timestamp_us": time,
        "utc_timestamp_ns": -1,
        "ImageSizes": [[1408, 1408]],
        "CameraCalibrations": [
            {
                "Label": "camera-rgb",
                "Calibrated": True,
                "Projection": {"Name": "FisheyeRadTanThinPrism", "Params": [700, 704, 704]},
                "T_Device_Camera": {"Translation": [0.1, 0, 0], "UnitQuaternion": [1, [0, 0, 0]]},
            }
        ],
    }


@pytest.fixture
def synthetic_recording(tmp_path, monkeypatch):
    root = tmp_path / "raw" / "comind"
    recording = root / "recordings" / RECORDING
    for index, role in enumerate(("helper", "leader")):
        offset = index * 1_000_000
        mps = recording / f"mps_{role}_trimmed_vrs"
        rows = [
            _trajectory(offset + time, index * 10 + x, "shared-graph")
            for x, time in enumerate((1000, 2000, 3000))
        ]
        _csv(
            recording / "multislam_output" / str(index) / "slam" / "closed_loop_trajectory.csv",
            rows,
        )
        _csv(
            mps / "slam" / "closed_loop_trajectory.csv",
            [{**row, "graph_uid": f"{role}-standard", "tx_world_device": 999} for row in rows],
        )
        _csv(
            mps / "hand_tracking" / "hand_tracking_results.csv",
            [_hand(offset + 1500, 1, -1), _hand(offset + 2500, 0, 1)],
        )
        (mps / "slam" / "online_calibration.jsonl").write_text(
            "\n".join(json.dumps(_calibration(offset + time)) for time in (1000, 3000)) + "\n"
        )
        video_path = recording / "mp4s" / f"{role}_trimmed_sync.mp4"
        video_path.parent.mkdir(exist_ok=True)
        video_path.write_bytes(b"metadata mocked for this integration test")
    (recording / "multislam_output" / "vrs_to_multi_slam.json").write_text(
        json.dumps(
            {
                f"{RECORDING}/trimmed_vrs/{role}_trimmed.vrs": str(index)
                for index, role in enumerate(("helper", "leader"))
            }
        )
    )
    scan = recording / "scan"
    scan.mkdir()
    np.savetxt(scan / "T_ariaWorld_from_blkWorld.txt", np.eye(4))
    for name in ("aria_semidense_points.ply", "blk_scan_aria_aligned.ply"):
        (scan / name).write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 0\n"
            b"property double x\nproperty double y\nproperty double z\nend_header\n"
        )

    def metadata(self):
        return VideoMetadata(
            self.path,
            self.participant_id,
            self.clock_domain,
            "synthetic",
            1408,
            1408,
            0,
            Fraction(1, 15360),
            0,
            1024,
            2,
            Fraction(30),
            66_667,
            1,
            0,
            self.provenance,
        )

    monkeypatch.setattr(EgoVideo, "metadata", metadata)
    return root


def test_real_validator_pipeline_uses_multi_world_with_separate_participant_clocks(
    synthetic_recording, tmp_path
):
    before = raw_inventory(synthetic_recording)
    progress = []
    report = run_validation(
        synthetic_recording,
        RECORDING,
        max_hand_gap_seconds=0.001,
        processed_root=tmp_path / "processed",
        progress=progress.append,
    )
    assert raw_inventory(synthetic_recording) == before
    assert report["shared_world"]["verified"]
    assert report["shared_world"]["graph_uid"] == "shared-graph"
    assert not report["shared_world"]["time_synchronization_verified"]
    for role, offset in (("helper", 0), ("leader", 10)):
        participant = report["participants"][role]
        assert participant["standard_mps"]["first_device_position_m"] == [999, 0, 0]
        assert participant["multislam"]["first_device_position_m"] == [offset, 0, 0]
        assert participant["multislam"]["last_device_position_m"] == [offset + 2, 0, 0]
        assert participant["multislam"]["row_count"] == 3
        assert participant["multislam"]["transform_valid_count"] == 3
        assert participant["calibration"]["labels"] == ["camera-rgb"]
        assert participant["calibration"]["utc_missing_sentinel_count"] == 2
        assert participant["hands"]["left"]["zero_confidence_count"] == 1
        assert participant["hands"]["right"]["missing_sentinel_count"] == 1
        assert participant["shared_hands"]["transformed_present_hand_count"] == 3
        residuals = participant["shared_hands"]["residuals_per_hand_timestamp"]
        assert residuals["accepted_count"] == 2
        assert residuals["signed_min_seconds"] == -0.0005
        assert participant["rgb_camera_first_last"][0]["position_shared_world_m"] == [
            offset + 0.1,
            0,
            0,
        ]
    assert report["scan"]["transform_validity"] == "pass"
    assert report["scan"]["distance_unit"] is None
    assert not report["scan"]["registration_semantics_verified"]
    assert len(progress) >= 8
    json.dumps(report, allow_nan=False)


def test_validation_gap_rejection_keeps_hands_absent(synthetic_recording, tmp_path):
    report = run_validation(
        synthetic_recording,
        RECORDING,
        max_hand_gap_seconds=0,
        processed_root=tmp_path / "processed",
    )
    for role in ("helper", "leader"):
        hands = report["participants"][role]["shared_hands"]
        assert hands["transformed_present_hand_count"] == 0
        assert hands["residuals_per_hand_timestamp"]["gap_rejected_count"] == 2


def test_output_guards_reject_dataset_and_sibling_raw_paths(tmp_path):
    root = tmp_path / "raw" / "comind"
    for path in (root / "report.json", root.parent / "another_dataset" / "report.json"):
        with pytest.raises(ValueError, match="immutable raw"):
            ensure_output_outside_raw(path, root)
    ensure_output_outside_raw(tmp_path / "processed" / "report.json", root)


def test_output_guard_resolves_symlinks(tmp_path):
    root = tmp_path / "raw" / "comind"
    root.mkdir(parents=True)
    alias = tmp_path / "outputs"
    alias.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError, match="immutable raw"):
        ensure_output_outside_raw(alias / "report.json", root)


def test_selected_pose_parser_retains_original_raw_row_evidence_and_reuses_duplicates():
    from duet.adapters.comind.validation import SelectedTrajectoryRow, TrajectoryAudit
    from duet.schemas.common import FrameId, Provenance

    raw = {key: str(value) for key, value in _trajectory(123, 4, "shared").items()}
    row = SelectedTrajectoryRow(27, 123, raw)
    audit = TrajectoryAudit({}, frozenset({"shared"}), (row, None, row))
    poses = parse_selected_poses(
        audit,
        participant="helper",
        recording_id=RECORDING,
        world_frame=FrameId("shared"),
        provenance=Provenance("source.csv"),
    )
    assert poses[0] is poses[2]
    assert poses[1] is None
    assert poses[0].timestamp.raw_value == 123
    assert "csv-row=29" in poses[0].provenance.source
    assert poses[0].transform.destination == FrameId("shared")
