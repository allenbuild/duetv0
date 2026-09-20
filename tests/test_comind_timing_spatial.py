"""Small clock-pair and spatial diagnostics; no downloaded files are needed."""

import csv
import importlib.util
import io
import zlib
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.timing import ReportedUtcPairs, clock_pair_statistics
from duet.qc.comind_spatial import check_device_motion, check_hand_plausibility
from duet.qc.result import QCStatus
from duet.schemas.common import Provenance
from duet.schemas.time import Timestamp, TimeUnit, comparable_seconds

PROVENANCE = Provenance("synthetic paired RTC evidence")
EPOCH = 1_800_000_000_000_000_001


def timestamp(value: int, role: str = "helper") -> Timestamp:
    return Timestamp(value, TimeUnit.MICROSECONDS, device_clock("synthetic", role), PROVENANCE)


def test_exact_reported_utc_pairs_preserve_nanoseconds_and_remain_participant_scoped() -> None:
    helper = ReportedUtcPairs("synthetic", "helper", [1, 2], [EPOCH, EPOCH + 1000], PROVENANCE)
    leader = ReportedUtcPairs("synthetic", "leader", [5, 6], [EPOCH, EPOCH + 1000], PROVENANCE)
    result = helper.exact(timestamp(1))
    assert result.raw_value == EPOCH
    assert result.unit == TimeUnit.NANOSECONDS
    assert result.clock_domain != leader.reported_clock
    with pytest.raises(ValueError, match="verified mapping"):
        comparable_seconds(result, leader.reported_clock)
    lookup = helper.lookup(timestamp(1), max_gap_seconds=0)
    assert lookup.accepted
    assert lookup.uncertainty_seconds is None
    assert not lookup.cross_device_synchronization_verified


def test_nearest_pair_preserves_source_value_and_reports_residual_without_interpolation() -> None:
    pairs = ReportedUtcPairs(
        "synthetic", "helper", [10, 10, 14], [EPOCH, EPOCH, EPOCH + 4000], PROVENANCE
    )
    result = pairs.lookup(timestamp(12), max_gap_seconds=Fraction(2, 1_000_000))
    assert result.source_index == 0
    assert result.reported_utc.raw_value == EPOCH
    assert result.residual_seconds == Fraction(-2, 1_000_000)
    assert result.accepted
    assert not pairs.lookup(timestamp(12), max_gap_seconds=0).accepted
    with pytest.raises(ValueError, match="exact"):
        pairs.exact(timestamp(12))


def test_pair_queries_reject_other_participant_clocks() -> None:
    pairs = ReportedUtcPairs("synthetic", "helper", [1], [EPOCH], PROVENANCE)
    with pytest.raises(ValueError, match="same participant"):
        pairs.lookup(timestamp(1, "leader"), max_gap_seconds=1)


def test_missing_utc_and_negative_device_time_are_distinct() -> None:
    pairs = ReportedUtcPairs("synthetic", "helper", [-1, 0], [-1, EPOCH], PROVENANCE)
    result = pairs.lookup(timestamp(-1), max_gap_seconds=0)
    assert result.paired_device_timestamp.raw_value == -1
    assert result.reported_utc is None
    assert not result.accepted
    assert pairs.exact(timestamp(0)).raw_value == EPOCH


def test_pairs_are_immutable_and_reject_conflicting_duplicate_values() -> None:
    values = np.array([1, 2], dtype=np.int64)
    pairs = ReportedUtcPairs("synthetic", "helper", values, [EPOCH, EPOCH + 1000], PROVENANCE)
    values[0] = 999
    assert pairs.device_us[0] == 1
    with pytest.raises(ValueError):
        pairs.device_us.setflags(write=True)
    with pytest.raises(ValueError, match="conflicting"):
        ReportedUtcPairs("synthetic", "helper", [1, 1], [EPOCH, EPOCH + 1], PROVENANCE)
    with pytest.raises(ValueError, match="nondecreasing"):
        ReportedUtcPairs("synthetic", "helper", [2, 1], [EPOCH, EPOCH + 1], PROVENANCE)


def test_diagnostic_affine_fit_detects_drift_without_claiming_external_accuracy() -> None:
    report = clock_pair_statistics(
        [1_000_000, 2_000_000, 3_000_000],
        [EPOCH, EPOCH + 1_000_001_000, EPOCH + 2_000_002_000],
        offset_jump_threshold_seconds=0.001,
    )
    assert report["reported_utc_range_ns"] == [EPOCH, EPOCH + 2_000_002_000]
    assert report["offset_origin_ns"] == EPOCH - 1_000_000_000
    assert report["offset_net_change_ns"] == 2000
    assert report["fit"]["slope_drift_ppm"] == pytest.approx(1)
    assert report["fit"]["rms_residual_ns"] < 1e-6
    assert report["utc_accuracy_bound_seconds"] is None
    assert not report["fit"]["external_accuracy_verified"]


def test_clock_audit_detects_reported_utc_reversal_and_offset_jump() -> None:
    report = clock_pair_statistics(
        [0, 1000, 2000, 3000],
        [EPOCH, EPOCH + 1_000_000, EPOCH - 5_000_000, EPOCH - 4_000_000],
        offset_jump_threshold_seconds=0.001,
    )
    assert report["device_decreasing_count"] == 0
    assert report["reported_utc_decreasing_count"] == 1
    assert report["offset_jump_count"] == 1
    assert report["fit"]["rms_residual_ns"] > 1


def test_reported_utc_staircase_is_not_installed_as_an_affine_clock_mapping() -> None:
    report = clock_pair_statistics(
        [0, 1000, 2000, 3000],
        [EPOCH, EPOCH, EPOCH, EPOCH + 3_000_000],
        offset_jump_threshold_seconds=0.001,
    )
    assert report["reported_utc_duplicate_count"] == 2
    assert report["reported_utc_unique_value_count"] == 2
    assert report["reported_utc_plateaus_observed"]
    assert not report["conversion_model_installed"]


def test_empty_pairs_and_bad_dtypes_are_explicit() -> None:
    pairs = ReportedUtcPairs("synthetic", "helper", [], [], PROVENANCE)
    with pytest.raises(ValueError, match="no source"):
        pairs.exact(timestamp(1))
    with pytest.raises(ValueError, match="signed-integer"):
        ReportedUtcPairs("synthetic", "helper", [1.0], [EPOCH], PROVENANCE)


def test_device_motion_metrics_use_device_duration_and_do_not_claim_floor_height() -> None:
    qc = check_device_motion(
        [0, 1_000_000, 2_000_000],
        [[0, 0, 1], [1, 0, 1], [2, 0, 1]],
        max_speed_m_s=2,
        max_step_m=2,
        max_gap_seconds=1,
    )
    assert qc.status == QCStatus.PASS
    assert qc.metrics["speed_m_s"]["p50"] == 1
    assert qc.metrics["path_length_m"] == 2
    assert qc.metrics["net_displacement_m"] == 2
    assert not qc.metrics["height_above_floor_available"]
    assert qc.metrics["world_z_m"]["p50"] == 1


def test_device_motion_flags_jumps_gaps_and_nonmonotonic_values() -> None:
    qc = check_device_motion(
        [0, 1000, 1000],
        [[0, 0, 0], [1, 0, 0], [1, 0, 0]],
        max_speed_m_s=2,
        max_step_m=0.05,
        max_gap_seconds=0.1,
    )
    assert qc.status == QCStatus.FAIL
    assert qc.metrics["non_increasing_count"] == 1
    assert qc.metrics["excessive_speed_count"] == 1
    gap = check_device_motion(
        [0, 2_000_000], [[0, 0, 0], [1, 0, 0]], max_speed_m_s=2, max_step_m=2, max_gap_seconds=0.1
    )
    assert gap.status == QCStatus.INSUFFICIENT_DATA
    assert gap.metrics["speed_m_s"]["count"] == 0


def test_missing_hands_break_continuity_and_zero_confidence_is_present() -> None:
    points = np.zeros((3, 21, 3))
    points[2, :, 0] = 1
    qc = check_hand_plausibility(
        [0, 100_000, 200_000],
        points,
        [0, -1, 1],
        max_device_distance_m=2,
        max_wrist_speed_m_s=1,
        max_gap_seconds=0.1,
    )
    assert qc.metrics["present_count"] == 2
    assert qc.metrics["missing_count"] == 1
    assert qc.metrics["zero_confidence_count"] == 1
    assert qc.metrics["evaluated_continuity_interval_count"] == 0
    assert qc.metrics["speed_outlier_interval_count"] == 0


def test_hand_distance_speed_and_bad_confidence_are_flagged() -> None:
    points = np.zeros((2, 21, 3))
    points[1, :, 0] = 3
    qc = check_hand_plausibility(
        [0, 100_000],
        points,
        [1, 1],
        max_device_distance_m=2,
        max_wrist_speed_m_s=5,
        max_gap_seconds=0.1,
    )
    assert qc.status == QCStatus.FAIL
    assert qc.metrics["distance_outlier_sample_count"] == 1
    assert qc.metrics["speed_outlier_interval_count"] == 1
    assert qc.metrics["speed_outlier_interval_end_device_us_first_20"] == [100_000]
    for confidence in ([float("inf"), 1], [-2, 1], [1.1, 1]):
        with pytest.raises(ValueError):
            check_hand_plausibility(
                [0, 100_000],
                points,
                confidence,
                max_device_distance_m=2,
                max_wrist_speed_m_s=5,
                max_gap_seconds=0.1,
            )


def _semantics_script():
    path = Path(__file__).parents[1] / "scripts" / "validate_comind_semantics.py"
    spec = importlib.util.spec_from_file_location("semantics_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_single_pass_trajectory_read_checks_crc_and_preserves_exact_pairs(tmp_path: Path) -> None:
    module = _semantics_script()
    buffer = io.StringIO()
    row = {
        "graph_uid": "g",
        "tracking_timestamp_us": 1000,
        "utc_timestamp_ns": EPOCH,
        "tx_world_device": 0,
        "ty_world_device": 0,
        "tz_world_device": 0,
        "qx_world_device": 0,
        "qy_world_device": 0,
        "qz_world_device": 0,
        "qw_world_device": 1,
        "quality_score": 1,
    }
    writer = csv.DictWriter(buffer, fieldnames=list(row))
    writer.writeheader()
    writer.writerow(row)
    writer.writerow({**row, "tracking_timestamp_us": 2000, "utc_timestamp_ns": EPOCH + 1_000_000})
    payload = buffer.getvalue().encode()
    path = tmp_path / "trajectory.csv"
    path.write_bytes(payload)
    expected = f"{zlib.crc32(payload):08x}"
    arrays, source = module.read_trajectory_once(path, expected_crc=expected)
    assert arrays["utc_ns"].tolist() == [EPOCH, EPOCH + 1_000_000]
    assert source["bytes"] == len(payload)
    assert source["crc32"] == expected
    assert source["crc_verified_against_recovered_member"]
    with pytest.raises(ValueError, match="CRC mismatch"):
        module.read_trajectory_once(path, expected_crc="00000000")


def test_semantics_nearest_indices_use_timestamps_not_array_index() -> None:
    module = _semantics_script()
    indices = module.nearest_indices(np.array([10, 20, 30]), np.array([15, 29, 0, 100]))
    assert indices.tolist() == [0, 2, 0, 2]


def test_cache_verification_checks_derived_arrays_without_reading_sources(tmp_path: Path) -> None:
    import json

    module = _semantics_script()
    cache_root = tmp_path / "synthetic" / "semantics"
    trajectory = {
        "device_us": np.array([0, 33334]),
        "utc_ns": np.array([EPOCH, EPOCH + 33334000]),
        "translation": np.zeros((2, 3)),
        "quaternion": np.array([[0.0, 0.0, 0.0, 1.0]] * 2),
        "quality": np.ones(2),
    }
    points = np.zeros((2, 2, 21, 3))
    points[0, 0] = np.nan
    hands = {
        "device_us": np.array([0, 33334]),
        "points": points,
        "confidence": np.array([[-1, 1], [0, 1]]),
    }
    report = {
        "recording_id": "synthetic",
        "cross_device_time": {"verified": False},
        "shared_world": {"graph_uid": "graph-g"},
        "participants": {},
    }
    for role in ("helper", "leader"):
        report["participants"][role] = module.analyze_role(
            trajectory,
            hands,
            role=role,
            source={"path": "source-that-does-not-exist.csv", "graph_uid": "graph-g"},
            cache_path=cache_root / f"{role}.npz",
            cache_max_hz=30,
            max_hand_gap_seconds=0.01,
        )
    (cache_root / "manifest.json").write_text(json.dumps(report))
    result = module.verify_cache_report(tmp_path, "synthetic")
    assert result["cache_validation"] == "pass"
    assert not result["source_files_reread"]
    assert not result["cross_device_time_verified"]


def test_spatial_qc_never_casts_away_imaginary_coordinates_or_confidence() -> None:
    with pytest.raises(ValueError, match="real"):
        check_device_motion([0], [[1j, 0, 0]], max_speed_m_s=2, max_step_m=0.1, max_gap_seconds=0.1)
    for points, confidence in [(np.full((1, 21, 3), 1j), [1]), (np.zeros((1, 21, 3)), [1j])]:
        with pytest.raises(ValueError, match="real"):
            check_hand_plausibility(
                [0],
                points,
                confidence,
                max_device_distance_m=2,
                max_wrist_speed_m_s=5,
                max_gap_seconds=0.1,
            )
    points = np.zeros((1, 21, 3))
    points[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="all-NaN"):
        check_hand_plausibility(
            [0], points, [1], max_device_distance_m=2, max_wrist_speed_m_s=5, max_gap_seconds=0.1
        )


@pytest.mark.parametrize("value", ["nan", "inf"])
def test_semantics_reader_rejects_nonfinite_present_hand_coordinates(
    tmp_path: Path, value: str
) -> None:
    module = _semantics_script()
    row = {
        "tracking_timestamp_us": 0,
        "left_tracking_confidence": 1,
        "right_tracking_confidence": 1,
    }
    row.update(
        {
            f"t{axis}_{side}_landmark_{index}_device": 0
            for side in ("left", "right")
            for index in range(21)
            for axis in "xyz"
        }
    )
    row["tx_left_landmark_0_device"] = value
    path = tmp_path / "hands.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    with pytest.raises(ValueError, match="nonfinite coordinates"):
        module.read_hands_once(path)
    row["left_tracking_confidence"] = -1
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    arrays = module.read_hands_once(path)
    assert np.isnan(arrays["points"][0, 0]).all()


def test_semantics_reader_rejects_invalid_recording_before_source_access(
    monkeypatch,
) -> None:
    import sys

    module = _semantics_script()
    monkeypatch.setattr(
        sys, "argv", ["validate_comind_semantics.py", "--recording-id", "unsupported"]
    )
    with pytest.raises(ValueError, match="UUID"):
        module.main()


def test_semantics_builds_new_recording_without_standard_mps_calibration_or_scans(
    tmp_path: Path, monkeypatch
) -> None:
    """The minimal published CSVs must exercise the existing shared-world cache."""
    import json
    import sys

    module = _semantics_script()
    recording_id = "12345678-1234-4234-9234-123456789abc"
    raw = tmp_path / "raw"
    processed = tmp_path / "processed"
    base = raw / "recordings" / recording_id
    mapping = {"helper": "7", "leader": "4"}
    trajectory_row = {
        "graph_uid": "shared-g",
        "tracking_timestamp_us": 0,
        "utc_timestamp_ns": EPOCH,
        **{f"t{axis}_world_device": 0 for axis in "xyz"},
        **{f"q{axis}_world_device": 0 for axis in "xyz"},
        "qw_world_device": 1,
        "quality_score": 1,
    }
    hand_row = {
        "tracking_timestamp_us": 0,
        "left_tracking_confidence": 1,
        "right_tracking_confidence": 1,
        **{
            f"t{axis}_{side}_landmark_{index}_device": 0
            for side in ("left", "right")
            for index in range(21)
            for axis in "xyz"
        },
    }
    for role, index in mapping.items():
        video = base / "mp4s" / f"{role}_trimmed_sync.mp4"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.touch()
        trajectory = base / "multislam_output" / index / "slam/closed_loop_trajectory.csv"
        hands = base / f"mps_{role}_trimmed_vrs/hand_tracking/hand_tracking_results.csv"
        for path, row in ((trajectory, trajectory_row), (hands, hand_row)):
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
                second = {**row, "tracking_timestamp_us": 33334}
                if "utc_timestamp_ns" in second:
                    second["utc_timestamp_ns"] += 33334000
                writer.writerow(second)
    (base / "multislam_output/vrs_to_multi_slam.json").write_text(
        json.dumps(
            {
                f"{recording_id}/trimmed_vrs/{role}_trimmed.vrs": index
                for role, index in mapping.items()
            }
        )
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_comind_semantics.py",
            "--recording-id",
            recording_id,
            "--dataset-root",
            str(raw),
            "--processed-root",
            str(processed),
            "--output-root",
            str(tmp_path / "outputs"),
        ],
    )
    module.main()
    report = json.loads((processed / recording_id / "semantics/manifest.json").read_text())
    assert report["shared_world"]["graph_uid"] == "shared-g"
    for role, index in mapping.items():
        source = report["participants"][role]["source"]
        assert f"multislam_output/{index}/slam/closed_loop_trajectory.csv" in source["path"]
        assert not source["crc_verified_against_recovered_member"]
    assert module.verify_cache_report(processed, recording_id)["cache_validation"] == "pass"


def test_semantics_does_not_accept_an_unverified_derived_trajectory(tmp_path: Path) -> None:
    from types import SimpleNamespace

    module = _semantics_script()
    recording_id = "12345678-1234-4234-9234-123456789abc"
    derived = tmp_path / recording_id / "0/slam/closed_loop_trajectory.csv"
    derived.parent.mkdir(parents=True)
    derived.touch()
    with pytest.raises(FileNotFoundError, match="no unverified archive recovery"):
        module.semantic_trajectory_source(
            SimpleNamespace(role="helper", multislam_dir=None, multislam_index="0"),
            recording_id=recording_id,
            processed_root=tmp_path,
        )
