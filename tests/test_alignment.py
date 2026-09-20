"""Dataset-independent joint SE(3) alignment and heldout failure cases."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from duet.geometry.alignment import (
    PosePairs,
    fit_pose_alignment,
    pose_alignment_residuals,
    rotation_angles,
)
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import DistanceUnit, FrameId, Provenance


def rotation(angle):
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])


def examples(count=100):
    generator = np.random.default_rng(71)
    positions = generator.normal(size=(count, 3))
    rotations = np.array([rotation(angle) for angle in np.linspace(-2, 2, count)])
    world_rotation = rotation(0.73)
    translation = np.array([2.4, -1.1, 0.8])
    pairs = PosePairs(
        positions,
        positions @ world_rotation.T + translation,
        rotations,
        world_rotation @ rotations,
        FrameId("original/world"),
        FrameId("shared/world"),
        FrameId("device"),
        DistanceUnit.METERS,
        Provenance("synthetic matched same-device poses"),
    )
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = world_rotation, translation
    return pairs, matrix


def fit(pairs, **options):
    return fit_pose_alignment(pairs, orientation_length_scale_m=1, huber_delta_m=0.05, **options)


def test_recovers_asymmetric_world_transform_with_heldout_positions_and_orientations():
    pairs, expected = examples()
    indices = np.arange(len(pairs.source_positions))
    result = fit(pairs.subset(indices % 5 != 0))
    np.testing.assert_allclose(result.transform.matrix, expected, atol=1e-12)
    distances, angles = pose_alignment_residuals(pairs.subset(indices % 5 == 0), result.transform)
    assert distances.max() < 1e-12
    assert angles.max() < 1e-12
    assert result.converged
    assert result.downweighted_pairs == 0
    assert result.transform.source == pairs.source_world
    assert result.transform.destination == pairs.destination_world


def test_orientation_evidence_identifies_rotation_with_stationary_positions():
    pairs, expected = examples()
    pairs = replace(
        pairs,
        source_positions=np.zeros((100, 3)),
        destination_positions=np.broadcast_to(expected[:3, 3], (100, 3)),
    )
    np.testing.assert_allclose(fit(pairs).transform.matrix, expected, atol=1e-12)


def test_robust_fit_downweights_large_position_and_orientation_outliers():
    pairs, expected = examples(120)
    positions = pairs.destination_positions.copy()
    rotations = pairs.destination_rotations.copy()
    positions[-10:] += [50, -20, 10]
    rotations[-10:] = rotation(2.0) @ rotations[-10:]
    result = fit(replace(pairs, destination_positions=positions, destination_rotations=rotations))
    np.testing.assert_allclose(result.transform.matrix, expected, atol=0.01)
    assert result.downweighted_pairs == 10
    assert result.converged


def test_drifting_world_relation_does_not_pass_heldout_pose_residuals():
    pairs, _ = examples()
    drift = np.linspace(0, 0.7, 100)
    pairs = replace(
        pairs,
        destination_positions=pairs.destination_positions + np.c_[drift, drift * 0, drift * 0],
        destination_rotations=np.array([rotation(v) for v in drift]) @ pairs.destination_rotations,
    )
    train = np.arange(100) % 5 != 0
    result = fit(pairs.subset(train))
    distances, angles = pose_alignment_residuals(pairs.subset(~train), result.transform)
    assert np.percentile(distances, 95) > 0.1
    assert np.rad2deg(np.percentile(angles, 95)) > 5


def test_scale_is_not_fitted_or_hidden():
    pairs, _ = examples()
    scaled = replace(pairs, destination_positions=pairs.destination_positions * 1.2)
    result = fit(scaled)
    distances, _ = pose_alignment_residuals(scaled, result.transform)
    assert np.median(distances) > 0.1
    np.testing.assert_allclose(
        result.transform.matrix[:3, :3].T @ result.transform.matrix[:3, :3], np.eye(3), atol=1e-12
    )


def test_input_arrays_immutable_and_frame_mismatches_rejected():
    pairs, _ = examples()
    with pytest.raises(ValueError):
        pairs.source_positions.setflags(write=True)
    result = fit(pairs)
    wrong = RigidTransform(
        result.transform.matrix,
        FrameId("unrelated/world"),
        pairs.destination_world,
        DistanceUnit.METERS,
        pairs.provenance,
    )
    with pytest.raises(ValueError, match="world frames"):
        pose_alignment_residuals(pairs, wrong)


@pytest.mark.parametrize("invalid", [np.diag([1, 1, -1]), np.eye(3) * 1.01])
def test_reflection_and_invalid_rotation_rejected(invalid):
    pairs, _ = examples()
    rotations = pairs.source_rotations.copy()
    rotations[0] = invalid
    with pytest.raises(ValueError, match=r"SO\(3\)"):
        replace(pairs, source_rotations=rotations)


@pytest.mark.parametrize("field", ["source_positions", "source_rotations"])
def test_complex_and_nonfinite_geometry_rejected(field):
    pairs, _ = examples()
    with pytest.raises(ValueError, match="real"):
        replace(pairs, **{field: getattr(pairs, field).astype(complex) + 1j})
    values = getattr(pairs, field).copy()
    values[0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        replace(pairs, **{field: values})


def test_units_and_distinct_frames_are_required():
    pairs, _ = examples()
    with pytest.raises(ValueError, match="meters"):
        replace(pairs, unit=DistanceUnit.MILLIMETERS)
    with pytest.raises(ValueError, match="distinct"):
        replace(pairs, source_world=pairs.destination_world)
    with pytest.raises(ValueError, match="device frame"):
        replace(pairs, device_frame=pairs.source_world)


@pytest.mark.parametrize("value", [True, 0, -1, np.nan, np.inf])
def test_invalid_estimator_thresholds_rejected(value):
    pairs, _ = examples()
    with pytest.raises((ValueError, TypeError)):
        fit_pose_alignment(pairs, orientation_length_scale_m=value, huber_delta_m=0.1)


def test_insufficient_data_and_explicit_nonconvergence():
    pairs, _ = examples()
    with pytest.raises(ValueError, match="three"):
        fit(pairs.subset([0, 1]))
    shifted = pairs.destination_positions.copy()
    shifted[:10] += [1, 2, 3]
    result = fit(replace(pairs, destination_positions=shifted), max_iterations=1)
    assert not result.converged
    assert result.iterations == 1


def test_numeric_overflow_raises_instead_of_reporting_success():
    pairs, _ = examples()
    overflow = replace(pairs, source_positions=pairs.source_positions * 1e300)
    with pytest.raises(ValueError, match="overflow"):
        fit(overflow)


@pytest.mark.parametrize("invalid", [np.zeros((4, 4)), np.eye(3) + 1j, np.full((3, 3), np.inf)])
def test_rotation_residual_shape_complex_and_nonfinite_rejected(invalid):
    with pytest.raises(ValueError):
        rotation_angles(invalid)


def alignment_script():
    path = Path(__file__).resolve().parents[1] / "scripts/validate_comind_world_alignment.py"
    spec = importlib.util.spec_from_file_location("alignment_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_numeric_alignment_report_never_claims_scan_registration():
    pairs, _ = examples(200)
    report = alignment_script().study(
        pairs, np.arange(200, dtype=np.int64) * 33000, length=1, huber=0.05
    )
    assert report["classification"] == "NUMERIC_GATE_PASSED"
    assert report["semantic_registration_verified"] is False
    assert report["scene_enabled"] is False
    assert report["method"]["training_count"] == 160
    assert report["method"]["heldout_count"] == 40
    assert len(report["time_bins"]) == 10
    assert report["scale_diagnostic_heldout"][
        "scale_with_fixed_joint_pose_rotation"
    ] == pytest.approx(1)
    assert report["relative_motion_consistency"][0]["translation_m"]["max"] < 1e-12


def test_numeric_report_rejects_scale_change_and_handles_sparse_time_bins():
    pairs, _ = examples(20)
    pairs = replace(pairs, destination_positions=1.2 * pairs.destination_positions)
    report = alignment_script().study(
        pairs, np.arange(20, dtype=np.int64) * 33000, length=1, huber=0.05
    )
    assert report["classification"] == "CONSTANT_SE3_NOT_VALIDATED"
    assert not report["numeric_gate"]["passed"]
    assert all(bin_["status"] == "INSUFFICIENT_LOCAL_TRAINING_DATA" for bin_ in report["time_bins"])
