"""Synthetic checks of geometry; no dataset files are needed or accessed."""

import numpy as np
import pytest

from duet.geometry.frame_graph import FrameGraph
from duet.geometry.transforms import RigidTransform, rigid_matrix_errors
from duet.qc.geometry import TrajectorySample, check_trajectory_continuity, check_transform
from duet.qc.result import QCStatus
from duet.schemas.common import DistanceUnit, FrameId, Provenance
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit

PROVENANCE = Provenance("synthetic_test", detail="Known right-handed synthetic coordinates")
CAMERA = FrameId("synthetic/participant_a/camera")
HAND = FrameId("synthetic/participant_a/hand")
WORLD = FrameId("synthetic/world")
CLOCK = ClockDomain("synthetic/common_clock")


def make_transform(source=CAMERA, destination=WORLD, translation=(0, 0, 0), rotation=None):
    matrix = np.eye(4)
    matrix[:3, 3] = translation
    if rotation is not None:
        matrix[:3, :3] = rotation
    return RigidTransform(matrix, source, destination, DistanceUnit.METERS, PROVENANCE)


def timestamp(seconds, clock=CLOCK):
    return Timestamp(seconds, TimeUnit.SECONDS, clock, PROVENANCE)


def test_composition_matches_column_vector_convention():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    world_camera = make_transform(translation=(1, 2, 3), rotation=rotation)
    camera_hand = make_transform(HAND, CAMERA, translation=(2, 0, 0))
    world_hand = world_camera.compose(camera_hand)
    assert world_hand.source == HAND
    assert world_hand.destination == WORLD
    np.testing.assert_allclose(world_hand.matrix[:3, 3], [1, 4, 3])
    assert world_hand.provenance.parents == (world_camera.provenance, camera_hand.provenance)


def test_incompatible_frame_composition_rejected():
    with pytest.raises(ValueError, match="incompatible frames"):
        make_transform().compose(make_transform(HAND, WORLD))


def test_inverse_reverses_transform_and_restores_points():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = make_transform(translation=(1, 2, 3), rotation=rotation)
    inverse = transform.inverse()
    assert inverse.source == WORLD
    assert inverse.destination == CAMERA
    np.testing.assert_allclose(inverse.compose(transform).matrix, np.eye(4), atol=1e-12)
    points = np.array([[1, 2, 3], [4, 5, 6]])
    np.testing.assert_allclose(
        inverse.apply(transform.apply(points, unit=DistanceUnit.METERS), unit=DistanceUnit.METERS),
        points,
        atol=1e-12,
    )


def test_points_rotate_then_translate_and_keep_shape():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    transform = make_transform(translation=(1, 2, 3), rotation=rotation)
    np.testing.assert_allclose(transform.apply([2, 0, 0], unit=DistanceUnit.METERS), [1, 4, 3])
    assert transform.apply(np.zeros((2, 4, 3)), unit=DistanceUnit.METERS).shape == (2, 4, 3)
    assert transform.apply(np.empty((0, 3)), unit=DistanceUnit.METERS).shape == (0, 3)


@pytest.mark.parametrize(
    "rotation, message",
    [
        (np.diag([2, 1, 1]), "orthonormal"),
        (np.diag([-1, 1, 1]), "determinant"),
        (np.zeros((3, 3)), "orthonormal"),
        (np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]]), "orthonormal"),
    ],
)
def test_invalid_rotations_and_reflections_are_rejected(rotation, message):
    with pytest.raises(ValueError, match=message):
        make_transform(rotation=rotation)


@pytest.mark.parametrize("matrix", [np.eye(3), np.full((4, 4), np.nan), np.eye(4) * 2])
def test_invalid_matrix_shape_finiteness_or_final_row(matrix):
    with pytest.raises(ValueError):
        RigidTransform(matrix, CAMERA, WORLD, DistanceUnit.METERS, PROVENANCE)
    assert check_transform(matrix).status is QCStatus.FAIL


def test_complex_geometry_rejected_without_discarding_imaginary_values():
    matrix = np.eye(4, dtype=complex)
    matrix[0, 3] = 1j
    assert rigid_matrix_errors(matrix) == ("matrix must contain real numbers",)
    with pytest.raises(ValueError, match="real numbers"):
        make_transform().apply([1j, 0, 0], unit=DistanceUnit.METERS)


def test_transform_and_points_require_explicit_meter_units():
    matrix = np.eye(4)
    matrix[0, 3] = 1000
    with pytest.raises(ValueError, match="meters"):
        RigidTransform(matrix, CAMERA, WORLD, DistanceUnit.MILLIMETERS, PROVENANCE)
    converted = RigidTransform.from_matrix(
        matrix,
        source=CAMERA,
        destination=WORLD,
        unit=DistanceUnit.MILLIMETERS,
        provenance=PROVENANCE,
    )
    assert converted.unit is DistanceUnit.METERS
    assert converted.matrix[0, 3] == 1
    assert "mm to m" in converted.provenance.detail
    assert converted.provenance.parents == (PROVENANCE,)
    assert matrix[0, 3] == 1000
    with pytest.raises(ValueError, match="meters"):
        converted.apply([0, 0, 0], unit=DistanceUnit.CENTIMETERS)


def test_matrix_is_defensively_copied_and_cannot_be_made_writable():
    original = np.eye(4)
    transform = RigidTransform(original, CAMERA, WORLD, DistanceUnit.METERS, PROVENANCE)
    original[0, 3] = 42
    assert transform.matrix[0, 3] == 0
    with pytest.raises(ValueError):
        transform.matrix[0, 3] = 42
    with pytest.raises(ValueError):
        transform.matrix.setflags(write=True)


def test_frame_graph_resolves_unique_path_in_both_directions():
    graph = FrameGraph(
        [make_transform(translation=(1, 0, 0)), make_transform(HAND, CAMERA, (0, 2, 0))]
    )
    np.testing.assert_allclose(graph.get_transform(HAND, WORLD).matrix[:3, 3], [1, 2, 0])
    np.testing.assert_allclose(graph.get_transform(WORLD, HAND).matrix[:3, 3], [-1, -2, 0])
    np.testing.assert_allclose(graph.get_transform(WORLD, WORLD).matrix, np.eye(4))


def test_frame_graph_rejects_redundant_and_contradictory_paths():
    graph = FrameGraph([make_transform(), make_transform(HAND, CAMERA)])
    for transform in [make_transform(), make_transform(HAND, WORLD, (99, 0, 0))]:
        with pytest.raises(ValueError, match="cycle"):
            graph.add_transform(transform)
    assert len(graph.frames) == 3
    np.testing.assert_allclose(graph.get_transform(HAND, WORLD).matrix, np.eye(4))


def test_frame_graph_rejects_unknown_and_disconnected_frames():
    graph = FrameGraph(
        [make_transform(), make_transform(FrameId("unrelated/a"), FrameId("unrelated/b"))]
    )
    with pytest.raises(KeyError, match="unknown"):
        graph.get_transform(CAMERA, FrameId("unknown"))
    with pytest.raises(ValueError, match="disconnected"):
        graph.get_transform(CAMERA, FrameId("unrelated/a"))


def test_transform_qc_and_threshold_validation():
    assert check_transform(np.eye(4)).status is QCStatus.PASS
    with pytest.raises(ValueError, match="tolerance"):
        check_transform(np.eye(4), tolerance=-1)


def test_trajectory_continuity_reports_translation_and_rotation_speed():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    samples = [
        TrajectorySample(timestamp(0), make_transform()),
        TrajectorySample(timestamp(2), make_transform(translation=(2, 0, 0), rotation=rotation)),
    ]
    result = check_trajectory_continuity(
        samples, max_translation_speed_m_s=1, max_angular_speed_rad_s=1
    )
    assert result.status is QCStatus.PASS
    assert result.metrics["translation_speeds_m_s"] == (1.0,)
    assert result.metrics["angular_speeds_rad_s"] == pytest.approx((np.pi / 4,))
    failure = check_trajectory_continuity(
        samples, max_translation_speed_m_s=0.5, max_angular_speed_rad_s=0.5
    )
    assert failure.status is QCStatus.FAIL
    assert failure.metrics["excessive_speed_interval_end_indices"] == (1,)


@pytest.mark.parametrize("second_time", [0, -1])
def test_trajectory_qc_rejects_duplicate_or_decreasing_times(second_time):
    result = check_trajectory_continuity(
        [
            TrajectorySample(timestamp(0), make_transform()),
            TrajectorySample(timestamp(second_time), make_transform()),
        ],
        max_translation_speed_m_s=1,
        max_angular_speed_rad_s=1,
    )
    assert result.status is QCStatus.FAIL
    assert result.metrics["invalid_intervals"][0]["reason"] == "non_increasing_timestamp"


def test_trajectory_qc_rejects_cross_clock_and_frame_changes():
    for second, reason in [
        (
            TrajectorySample(timestamp(1, ClockDomain("different")), make_transform()),
            "incompatible_clock_domains",
        ),
        (TrajectorySample(timestamp(1), make_transform(HAND)), "incompatible_coordinate_frames"),
    ]:
        result = check_trajectory_continuity(
            [TrajectorySample(timestamp(0), make_transform()), second],
            max_translation_speed_m_s=1,
            max_angular_speed_rad_s=1,
        )
        assert result.status is QCStatus.FAIL
        assert result.metrics["checked_interval_count"] == 0
        assert result.metrics["invalid_intervals"][0]["reason"] == reason


def test_trajectory_qc_missing_samples_break_continuity():
    result = check_trajectory_continuity(
        [
            TrajectorySample(timestamp(0), make_transform()),
            TrajectorySample(timestamp(1), None),
            TrajectorySample(timestamp(2), make_transform(translation=(100, 0, 0))),
        ],
        max_translation_speed_m_s=1,
        max_angular_speed_rad_s=1,
    )
    assert result.status is QCStatus.INSUFFICIENT_DATA
    assert result.metrics["checked_interval_count"] == 0
    assert result.metrics["missing_sample_count"] == 1
    assert (
        check_trajectory_continuity(
            [], max_translation_speed_m_s=1, max_angular_speed_rad_s=1
        ).status
        is QCStatus.INSUFFICIENT_DATA
    )


def test_trajectory_qc_preserves_nanosecond_deltas_at_large_epoch():
    epoch_ns = 1_700_000_000_000_000_000
    samples = [
        TrajectorySample(
            Timestamp(epoch_ns, TimeUnit.NANOSECONDS, CLOCK, PROVENANCE), make_transform()
        ),
        TrajectorySample(
            Timestamp(epoch_ns + 1, TimeUnit.NANOSECONDS, CLOCK, PROVENANCE),
            make_transform(translation=(1e-9, 0, 0)),
        ),
    ]
    result = check_trajectory_continuity(
        samples, max_translation_speed_m_s=1, max_angular_speed_rad_s=0
    )
    assert result.status is QCStatus.PASS
    assert result.metrics["translation_speeds_m_s"] == pytest.approx((1,))


def test_trajectory_qc_stationary_nontrivial_rotation_has_zero_speed():
    angle = np.pi / 3
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    transform = make_transform(rotation=rotation)
    result = check_trajectory_continuity(
        [TrajectorySample(timestamp(0), transform), TrajectorySample(timestamp(1), transform)],
        max_translation_speed_m_s=0,
        max_angular_speed_rad_s=0,
    )
    assert result.status is QCStatus.PASS
    assert result.metrics["angular_speeds_rad_s"] == (0,)


@pytest.mark.parametrize("tolerance", [True, np.bool_(False), "1e-8", None, np.array(1e-8), 1j])
def test_tolerance_requires_real_numeric_type(tolerance):
    with pytest.raises(TypeError, match="tolerance"):
        check_transform(np.eye(4), tolerance=tolerance)


@pytest.mark.parametrize("tolerance", [float("inf"), float("nan"), -1, 0, 10**1000])
def test_tolerance_requires_finite_positive_value(tolerance):
    with pytest.raises(ValueError, match="tolerance"):
        check_transform(np.eye(4), tolerance=tolerance)


@pytest.mark.parametrize("threshold", [True, np.bool_(False), "1", None, np.array(1), 1j])
@pytest.mark.parametrize("field", ["max_translation_speed_m_s", "max_angular_speed_rad_s"])
def test_continuity_threshold_requires_real_numeric_type(threshold, field):
    thresholds = {"max_translation_speed_m_s": 1, "max_angular_speed_rad_s": 1}
    thresholds[field] = threshold
    with pytest.raises(TypeError, match=field):
        check_trajectory_continuity([], **thresholds)


@pytest.mark.parametrize("threshold", [float("inf"), float("nan"), -1, 10**1000])
@pytest.mark.parametrize("field", ["max_translation_speed_m_s", "max_angular_speed_rad_s"])
def test_continuity_threshold_requires_finite_nonnegative_value(threshold, field):
    thresholds = {"max_translation_speed_m_s": 1, "max_angular_speed_rad_s": 1}
    thresholds[field] = threshold
    with pytest.raises(ValueError, match=field):
        check_trajectory_continuity([], **thresholds)


def test_same_frame_transform_must_be_identity():
    with pytest.raises(ValueError, match="itself must be identity"):
        make_transform(WORLD, WORLD, translation=(1, 0, 0))


def test_point_transform_rejects_arithmetic_overflow():
    transform = make_transform(translation=(1e308, 0, 0))
    with pytest.raises(ValueError, match="overflowed"):
        transform.apply([1e308, 0, 0], unit=DistanceUnit.METERS)


@pytest.mark.parametrize(
    "before_position, after_position, end_time",
    [(0, 1e308, 1e-300), (-1e308, 1e308, 1)],
)
def test_continuity_qc_rejects_nonfinite_arithmetic(before_position, after_position, end_time):
    result = check_trajectory_continuity(
        [
            TrajectorySample(timestamp(0), make_transform(translation=(before_position, 0, 0))),
            TrajectorySample(
                timestamp(end_time), make_transform(translation=(after_position, 0, 0))
            ),
        ],
        max_translation_speed_m_s=1e308,
        max_angular_speed_rad_s=1,
    )
    assert result.status is QCStatus.FAIL
    assert result.metrics["translation_speeds_m_s"] == ()
    assert result.metrics["max_observed_translation_speed_m_s"] is None
    assert result.metrics["invalid_intervals"][0]["reason"] == "non_finite_motion"


@pytest.mark.parametrize("displacement, end_time", [(1e200, 1e200), (1e-200, 1e-200)])
def test_continuity_qc_uses_stable_translation_norm(displacement, end_time):
    result = check_trajectory_continuity(
        [
            TrajectorySample(timestamp(0), make_transform()),
            TrajectorySample(timestamp(end_time), make_transform(translation=(displacement, 0, 0))),
        ],
        max_translation_speed_m_s=1,
        max_angular_speed_rad_s=0,
    )
    assert result.status is QCStatus.PASS
    assert result.metrics["translation_speeds_m_s"] == pytest.approx((1,))
