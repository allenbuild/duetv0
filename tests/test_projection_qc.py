"""Projection coverage tests use synthetic framed points and a simple projector."""

import numpy as np
import pytest

from duet.geometry.transforms import RigidTransform
from duet.qc.projection import check_point_projection_coverage
from duet.qc.result import QCStatus
from duet.schemas.common import DistanceUnit, FrameId, Provenance

WORLD = FrameId("world")
CAMERA = FrameId("camera")


def transform():
    return RigidTransform(
        [[0, -1, 0, 10], [1, 0, 0, 20], [0, 0, 1, 30], [0, 0, 0, 1]],
        CAMERA,
        WORLD,
        DistanceUnit.METERS,
        Provenance("synthetic"),
    )


def projector(point):
    return None if point[2] <= 0 else point[:2] / point[2] + [5, 5]


def run(points, transforms, present=None, **kwargs):
    return check_point_projection_coverage(
        points,
        point_present=np.ones(np.asarray(points).shape[:2], dtype=bool)
        if present is None
        else present,
        point_labels=("hand",),
        world_frame=WORLD,
        camera_frame=CAMERA,
        transforms_world_camera=transforms,
        project_camera_point=projector,
        image_size=(10, 10),
        minimum_coverage_fraction=0.5,
        **kwargs,
    )


def test_asymmetric_camera_transform_is_inverted_and_points_classified():
    t = transform()
    camera_points = np.array([[1, 2, 1], [10, 1, 1], [0, 0, -1]])
    points = t.apply(camera_points, unit=DistanceUnit.METERS)[:, None, :]
    result = run(points, [t] * 3)
    metric = result.metrics["points"]["hand"]
    assert metric["in_image_count"] == 1
    assert metric["projected_outside_image_count"] == 1
    assert metric["outside_calibrated_projection_domain_count"] == 1
    assert result.status is QCStatus.FAIL
    assert not result.metrics["actual_visibility_evaluated"]


def test_missing_point_or_camera_are_excluded_from_evaluable_denominator():
    t = transform()
    points = np.array([[[10, 20, 31]], [[np.nan, np.nan, np.nan]], [[10, 20, 31]]])
    result = run(points, [t, t, None], [[True], [False], [True]])
    metric = result.metrics["points"]["hand"]
    assert metric["camera_and_point_available_count"] == 1
    assert metric["in_image_fraction_of_evaluable"] == 1
    assert metric["in_image_fraction_of_all_snapshots"] == 1 / 3
    assert result.status is QCStatus.PASS


def test_no_camera_or_points_is_insufficient():
    result = run(np.zeros((1, 1, 3)), [None])
    assert result.status is QCStatus.INSUFFICIENT_DATA
    assert result.metrics["points"]["hand"]["in_image_fraction_of_evaluable"] is None


def test_incompatible_frames_rejected():
    with pytest.raises(ValueError, match="declared camera"):
        run([[[0, 0, 1]]], [transform().inverse()])


def test_nonfinite_present_points_and_nonboolean_presence_rejected():
    with pytest.raises(ValueError, match="finite"):
        run([[[np.nan, 0, 1]]], [transform()])
    with pytest.raises(ValueError, match="boolean"):
        run([[[0, 0, 1]]], [transform()], [[1]])


def test_pixel_center_edges_are_inclusive():
    t = transform()
    points = t.apply([[-5, -5, 1], [4, 4, 1]], unit=DistanceUnit.METERS)[:, None, :]
    assert run(points, [t, t]).metrics["points"]["hand"]["in_image_count"] == 2
