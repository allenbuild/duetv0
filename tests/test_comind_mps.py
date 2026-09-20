"""Small synthetic MPS streams; tests never need downloaded dataset files."""

import csv
import io
import json
from math import sqrt
from pathlib import Path

import numpy as np
import pytest

from duet.adapters.comind.mps import (
    HAND_LANDMARK_NAMES,
    device_clock,
    iter_calibrations,
    iter_hands,
    iter_trajectory,
    load_calibrations,
    load_hands,
    load_trajectory,
)
from duet.schemas.common import DistanceUnit, FrameId, ParticipantId, Provenance, SampleState
from duet.schemas.time import TimeUnit
from duet.synchronization.matching import nearest_timestamp

RECORDING = "synthetic-recording"
PROVENANCE = Provenance("synthetic MPS fixture")


def csv_stream(rows: list[dict[str, object]]) -> io.StringIO:
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return stream


def trajectory_row(**overrides: object) -> dict[str, object]:
    return {
        "graph_uid": "graph-a",
        "tracking_timestamp_us": 1_234_567,
        "utc_timestamp_ns": 1_774_714_644_554_113_156,
        "tx_world_device": 1,
        "ty_world_device": 2,
        "tz_world_device": 3,
        "qx_world_device": 0,
        "qy_world_device": 0,
        "qz_world_device": sqrt(0.5),
        "qw_world_device": sqrt(0.5),
        "quality_score": 3.5,
        **overrides,
    }


def hand_row(**overrides: object) -> dict[str, object]:
    row = {
        "tracking_timestamp_us": 1_234_567,
        "left_tracking_confidence": 0.75,
        "right_tracking_confidence": 0,
    }
    row.update(
        {
            f"t{axis}_{side}_landmark_{index}_device": index + coordinate / 10
            for side in ("left", "right")
            for index in range(21)
            for coordinate, axis in enumerate("xyz")
        }
    )
    return {**row, **overrides}


def calibration_row() -> dict[str, object]:
    return {
        "tracking_timestamp_us": 1_234_567,
        "utc_timestamp_ns": -1,
        "ImageSizes": [[640, 480], [1408, 1408]],
        "CameraCalibrations": [
            {
                "Label": label,
                "Calibrated": True,
                "Projection": {"Name": "FisheyeRadTanThinPrism", "Params": [600, 704, 704]},
                "T_Device_Camera": {
                    "Translation": [1, 0, 0],
                    "UnitQuaternion": [sqrt(0.5), [0, 0, sqrt(0.5)]],
                },
            }
            for label in ("camera-slam-left", "camera-rgb")
        ],
    }


def trajectories(rows: list[dict[str, object]], **kwargs: object):
    return iter_trajectory(
        csv_stream(rows),
        participant="helper",
        recording_id=RECORDING,
        provenance=PROVENANCE,
        **kwargs,
    )


def hands(rows: list[dict[str, object]]):
    return iter_hands(
        csv_stream(rows), participant="helper", recording_id=RECORDING, provenance=PROVENANCE
    )


def calibrations(rows: list[dict[str, object]]):
    return iter_calibrations(
        io.StringIO("\n".join(json.dumps(row) for row in rows)),
        participant="helper",
        recording_id=RECORDING,
        provenance=PROVENANCE,
    )


def test_trajectory_preserves_exact_times_frames_units_and_raw_quality() -> None:
    sample = next(trajectories([trajectory_row()]))
    assert sample.participant_id == ParticipantId("participant/helper")
    assert sample.timestamp.raw_value == 1_234_567
    assert sample.timestamp.unit == TimeUnit.MICROSECONDS
    assert sample.timestamp.clock_domain == device_clock(RECORDING, "helper")
    assert sample.utc_timestamp.raw_value == 1_774_714_644_554_113_156
    assert sample.utc_timestamp_ns_raw == 1_774_714_644_554_113_156
    assert sample.utc_timestamp.unit == TimeUnit.NANOSECONDS
    assert sample.utc_timestamp.clock_domain != sample.timestamp.clock_domain
    assert sample.quality_score == 3.5  # Deliberately not treated as normalized confidence.
    assert sample.graph_uid == "graph-a"
    assert sample.transform.source == FrameId("helper/device")
    assert sample.transform.destination == FrameId("helper/mps_world/graph-a")
    assert sample.transform.unit == DistanceUnit.METERS
    np.testing.assert_allclose(sample.transform.matrix[:3, 3], [1, 2, 3])
    np.testing.assert_allclose(sample.transform.matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    assert "row 2" in sample.provenance.detail


def test_device_clocks_are_participant_and_recording_specific() -> None:
    assert device_clock(RECORDING, "helper") != device_clock(RECORDING, "leader")
    assert device_clock(RECORDING, "helper") != device_clock("other-recording", "helper")
    helper = next(trajectories([trajectory_row()])).timestamp
    leader = next(
        iter_trajectory(
            csv_stream([trajectory_row()]),
            participant="leader",
            recording_id=RECORDING,
            provenance=PROVENANCE,
        )
    ).timestamp
    with pytest.raises(ValueError, match="verified mapping"):
        nearest_timestamp(helper, [leader], max_gap_seconds=0)


def test_graph_islands_use_distinct_default_frames_and_cannot_share_explicit_frame() -> None:
    rows = [trajectory_row(), trajectory_row(graph_uid="graph-b")]
    samples = list(trajectories(rows))
    assert samples[0].transform.destination != samples[1].transform.destination
    with pytest.raises(ValueError, match="different graph_uid"):
        list(trajectories(rows, world_frame=FrameId("verified/shared")))
    assert next(
        trajectories([rows[0]], world_frame=FrameId("verified/shared"))
    ).transform.destination == (FrameId("verified/shared"))


def test_trajectory_parser_is_lazy_and_reports_malformed_row_location() -> None:
    stream = trajectories([trajectory_row(), trajectory_row(qw_world_device=5)])
    assert next(stream).graph_uid == "graph-a"
    with pytest.raises(ValueError, match="trajectory row 3.*unit norm"):
        next(stream)


@pytest.mark.parametrize(
    "overrides",
    [
        {"tracking_timestamp_us": "1.25"},
        {"utc_timestamp_ns": "1.5"},
        {"quality_score": "nan"},
        {"tx_world_device": "inf"},
        {"graph_uid": ""},
    ],
)
def test_trajectory_invalid_fields_fail(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        next(trajectories([trajectory_row(**overrides)]))


def test_utc_sentinel_is_retained_outside_usable_timestamp() -> None:
    sample = next(trajectories([trajectory_row(utc_timestamp_ns=-1)]))
    assert sample.utc_timestamp_ns_raw == -1
    assert sample.utc_timestamp.raw_value is None
    row = trajectory_row()
    del row["utc_timestamp_ns"]
    sample = next(trajectories([row]))
    assert sample.utc_timestamp is None
    assert sample.utc_timestamp_ns_raw is None


def test_21_hand_landmarks_use_documented_names_device_frame_and_order() -> None:
    row = next(hands([hand_row()]))
    assert row.left.points.shape == (21, 3)
    assert row.right.points.shape == (21, 3)
    assert row.left.frame == FrameId("helper/device")
    assert row.left.unit == DistanceUnit.METERS
    assert row.left.landmark_names == HAND_LANDMARK_NAMES
    assert row.left.landmark_names[5] == "wrist_joint"
    assert row.left.landmark_names[20] == "palm_center"
    np.testing.assert_allclose(row.left.points[5], [5, 5.1, 5.2])
    assert row.left.metadata.confidence.value == 0.75
    assert row.left_confidence_raw == 0.75
    assert (
        row.left.timestamp.clock_domain
        == next(trajectories([trajectory_row()])).timestamp.clock_domain
    )


def test_minus_one_means_missing_but_zero_confidence_is_present() -> None:
    row = hand_row(left_tracking_confidence=-1)
    row["tx_left_landmark_0_device"] = "unusable sentinel placeholder"
    parsed = next(hands([row]))
    assert parsed.left_confidence_raw == -1
    assert parsed.left.points is None
    assert parsed.left.metadata.state == SampleState.MISSING
    assert parsed.left.metadata.confidence is None
    assert "-1" in parsed.left.metadata.reason
    assert parsed.right.metadata.state == SampleState.PRESENT
    assert parsed.right.points is not None
    assert parsed.right.metadata.confidence.value == 0


def test_blank_confidence_is_unknown_without_erasing_present_coordinates() -> None:
    row = next(hands([hand_row(left_tracking_confidence="")]))
    assert row.left_confidence_raw is None
    assert row.left.metadata.confidence is None
    assert row.left.metadata.state == SampleState.PRESENT


def test_incomplete_hand_is_explicitly_missing_without_zero_filling() -> None:
    row = next(hands([hand_row(tx_left_landmark_7_device="")]))
    assert row.left.points is None
    assert row.left.metadata.state == SampleState.MISSING
    assert row.left.metadata.confidence.value == 0.75
    assert "Incomplete" in row.left.metadata.reason
    assert row.right.points is not None


@pytest.mark.parametrize("confidence", [-0.5, -2, 1.1, "nan", "inf"])
def test_invalid_hand_confidence_rejected(confidence: object) -> None:
    with pytest.raises(ValueError, match="hand row 2"):
        next(hands([hand_row(left_tracking_confidence=confidence)]))


def test_present_nonfinite_landmarks_rejected() -> None:
    with pytest.raises(ValueError, match="finite"):
        next(hands([hand_row(tx_right_landmark_1_device="nan")]))


def test_hand_schema_requires_all_21_landmark_columns() -> None:
    row = hand_row()
    del row["tz_right_landmark_20_device"]
    with pytest.raises(ValueError, match="missing required columns"):
        next(hands([row]))


def test_calibration_labels_native_projection_and_scalar_vector_quaternion() -> None:
    sample = next(calibrations([calibration_row()]))
    assert sample.timestamp.clock_domain == device_clock(RECORDING, "helper")
    assert sample.utc_timestamp.raw_value is None
    assert sample.utc_timestamp_ns_raw == -1
    assert [camera.label for camera in sample.cameras] == ["camera-slam-left", "camera-rgb"]
    rgb = sample.cameras[1]
    assert rgb.image_size == (1408, 1408)
    assert rgb.calibrated
    assert rgb.projection["Name"] == "FisheyeRadTanThinPrism"
    assert rgb.projection["Params"] == (600, 704, 704)
    assert rgb.transform.source == FrameId("helper/camera-rgb")
    assert rgb.transform.destination == FrameId("helper/device")
    np.testing.assert_allclose(rgb.transform.matrix[:3, :3] @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_calibration_world_camera_composition_uses_verified_direction() -> None:
    pose = next(trajectories([trajectory_row()]))
    camera = next(calibrations([calibration_row()])).cameras[1]
    result = pose.transform.compose(camera.transform)
    assert result.source == FrameId("helper/camera-rgb")
    assert result.destination == pose.transform.destination
    np.testing.assert_allclose(result.matrix[:3, 3], [1, 3, 3], atol=1e-12)
    np.testing.assert_allclose(result.matrix[:3, :3] @ [1, 0, 0], [-1, 0, 0], atol=1e-12)


def test_optional_image_sizes_and_uncalibrated_flag_are_preserved() -> None:
    row = calibration_row()
    del row["ImageSizes"]
    row["CameraCalibrations"][0]["Calibrated"] = False
    sample = next(calibrations([row]))
    assert sample.cameras[0].image_size is None
    assert not sample.cameras[0].calibrated


@pytest.mark.parametrize(
    "field,value",
    [
        ("UnitQuaternion", [0, 0, 0, 1]),
        ("UnitQuaternion", [2, [0, 0, 0]]),
        ("Translation", [1, 2]),
        ("Translation", [1, 2, float("nan")]),
    ],
)
def test_calibration_invalid_transform_fails(field: str, value: object) -> None:
    row = calibration_row()
    row["CameraCalibrations"][0]["T_Device_Camera"][field] = value
    with pytest.raises(ValueError, match="calibration line 1"):
        next(calibrations([row]))


def test_calibration_rejects_misaligned_dimensions_and_duplicate_labels() -> None:
    row = calibration_row()
    row["ImageSizes"] = [[640, 480]]
    with pytest.raises(ValueError, match="one entry per"):
        next(calibrations([row]))
    row = calibration_row()
    row["CameraCalibrations"][1]["Label"] = "camera-slam-left"
    with pytest.raises(ValueError, match="labels must be unique"):
        next(calibrations([row]))


def test_csv_rejects_duplicate_headers_and_extra_cells() -> None:
    with pytest.raises(ValueError, match="unique column"):
        next(
            iter_trajectory(
                io.StringIO("graph_uid,graph_uid\na,b\n"),
                participant="helper",
                recording_id=RECORDING,
                provenance=PROVENANCE,
            )
        )
    content = csv_stream([trajectory_row()]).getvalue().rstrip() + ",extra\n"
    with pytest.raises(ValueError, match="header width"):
        next(
            iter_trajectory(
                io.StringIO(content),
                participant="helper",
                recording_id=RECORDING,
                provenance=PROVENANCE,
            )
        )


def test_read_only_path_wrappers_use_synthetic_files(tmp_path: Path) -> None:
    trajectory = tmp_path / "trajectory.csv"
    trajectory.write_text(csv_stream([trajectory_row()]).getvalue())
    hand = tmp_path / "hands.csv"
    hand.write_text(csv_stream([hand_row()]).getvalue())
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(json.dumps(calibration_row()) + "\n")
    assert len(list(load_trajectory(trajectory, participant="helper", recording_id=RECORDING))) == 1
    assert len(list(load_hands(hand, participant="helper", recording_id=RECORDING))) == 1
    assert (
        len(list(load_calibrations(calibration, participant="helper", recording_id=RECORDING))) == 1
    )


def test_unverified_participant_label_rejected() -> None:
    with pytest.raises(ValueError, match="verified helper or leader"):
        device_clock(RECORDING, "left")


def test_calibration_preserves_optional_time_offset_and_indexed_readout() -> None:
    row = calibration_row()
    row["ReadoutTimesSec"] = [[1, 0.005]]
    row["CameraCalibrations"][1]["TimeOffsetSec_Device_Camera"] = -0.002
    parsed = next(calibrations([row]))
    assert parsed.cameras[0].readout_seconds is None
    assert parsed.cameras[0].time_offset_seconds is None
    assert parsed.cameras[1].readout_seconds == 0.005
    assert parsed.cameras[1].time_offset_seconds == -0.002
    assert parsed.timestamp.raw_value == row["tracking_timestamp_us"]


@pytest.mark.parametrize("readouts", [[[1, -1]], [[2, 0.01]], [[1, 0.01], [1, 0.02]], [[0]]])
def test_invalid_readout_metadata_is_rejected(readouts: list[list[float]]) -> None:
    row = calibration_row()
    row["ReadoutTimesSec"] = readouts
    with pytest.raises(ValueError, match="ReadoutTimesSec"):
        next(calibrations([row]))


def test_calibration_numeric_overflow_reports_source_line() -> None:
    row = calibration_row()
    row["CameraCalibrations"][0]["Projection"]["Params"][0] = 10**400
    with pytest.raises(ValueError, match="calibration line 1.*finite number"):
        next(calibrations([row]))
