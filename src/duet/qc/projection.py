"""Image-domain coverage of framed 3D points; this does not measure occlusion."""

from collections.abc import Callable, Sequence
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from duet.geometry.transforms import RigidTransform
from duet.qc.result import QCResult, QCStatus
from duet.schemas.common import DistanceUnit, FrameId, Provenance


def check_point_projection_coverage(
    points_world_m: ArrayLike,
    *,
    point_present: ArrayLike,
    point_labels: Sequence[str],
    world_frame: FrameId,
    camera_frame: FrameId,
    transforms_world_camera: Sequence[RigidTransform | None],
    project_camera_point: Callable[[np.ndarray], ArrayLike | None],
    image_size: tuple[int, int],
    minimum_coverage_fraction: float,
    provenance: tuple[Provenance, ...] = (),
) -> QCResult:
    """Check corresponding snapshots with explicit ``T_world_camera`` in meters.

    Input points have shape (snapshots, labels, 3), and the boolean presence mask
    has shape (snapshots, labels). A missing transform skips that camera snapshot.
    The projector must use the named camera frame and verified exported pixels;
    ``None`` means the point is outside its calibrated projection domain. The
    image interval uses pixel centers [0, width-1] x [0, height-1]. Each point's
    coverage denominator includes only present points with an available camera.
    Snapshot correspondence must already be verified by the caller; this function
    performs no temporal matching. It cannot detect occlusion or hand visibility.
    """
    if not isinstance(world_frame, FrameId) or not isinstance(camera_frame, FrameId):
        raise TypeError("world and camera frames must be explicit FrameId values")
    if len(image_size) != 2 or any(
        isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in image_size
    ):
        raise ValueError("image size must contain two positive integer dimensions")
    if isinstance(minimum_coverage_fraction, (bool, np.bool_)) or not isinstance(
        minimum_coverage_fraction, Real
    ):
        raise TypeError("minimum coverage must be a real fraction")
    if not np.isfinite(minimum_coverage_fraction) or not 0 <= minimum_coverage_fraction <= 1:
        raise ValueError("minimum coverage must be finite and in [0, 1]")
    labels = tuple(point_labels)
    if len(set(labels)) != len(labels) or not all(isinstance(v, str) and v for v in labels):
        raise ValueError("point labels must be distinct nonempty strings")
    if np.iscomplexobj(points_world_m):
        raise ValueError("points must use real meter coordinates")
    points = np.asarray(points_world_m, dtype=np.float64)
    present = np.asarray(point_present)
    count = len(transforms_world_camera)
    if points.shape != (count, len(labels), 3):
        raise ValueError("points require snapshot x label x 3 coordinates")
    if present.shape != points.shape[:2] or present.dtype.kind != "b":
        raise ValueError("presence must be a boolean snapshot x label mask")
    if not np.isfinite(points[present]).all():
        raise ValueError("present points must be finite")
    eligible = np.zeros_like(present)
    in_image = np.zeros_like(present)
    outside_model = np.zeros_like(present)
    camera_count = 0
    width, height = image_size
    for row, transform in enumerate(transforms_world_camera):
        if transform is None:
            continue
        if not isinstance(transform, RigidTransform):
            raise TypeError("camera transforms must be validated RigidTransform values")
        if transform.source != camera_frame or transform.destination != world_frame:
            raise ValueError("camera transform does not map the declared camera to world frame")
        camera_count += 1
        indices = np.flatnonzero(present[row])
        eligible[row, indices] = True
        camera_points = transform.inverse().apply(points[row, indices], unit=DistanceUnit.METERS)
        for column, point in zip(indices, camera_points, strict=True):
            pixel = project_camera_point(point)
            if pixel is None:
                outside_model[row, column] = True
                continue
            if np.iscomplexobj(pixel):
                raise ValueError("projector returned non-real pixel coordinates")
            pixel = np.asarray(pixel, dtype=np.float64)
            if pixel.shape != (2,) or not np.isfinite(pixel).all():
                raise ValueError("projector must return two finite pixel coordinates or None")
            in_image[row, column] = 0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1
    by_point = {}
    fractions = []
    for column, label in enumerate(labels):
        evaluated = int(eligible[:, column].sum())
        accepted = int(in_image[:, column].sum())
        fraction = accepted / evaluated if evaluated else None
        if fraction is not None:
            fractions.append(fraction)
        by_point[label] = {
            "tracked_point_count": int(present[:, column].sum()),
            "camera_and_point_available_count": evaluated,
            "in_image_count": accepted,
            "outside_calibrated_projection_domain_count": int(outside_model[:, column].sum()),
            "projected_outside_image_count": int(
                (eligible[:, column] & ~outside_model[:, column] & ~in_image[:, column]).sum()
            ),
            "in_image_fraction_of_evaluable": fraction,
            "in_image_fraction_of_all_snapshots": accepted / count if count else None,
        }
    status = (
        QCStatus.FAIL
        if any(value < minimum_coverage_fraction for value in fractions)
        else QCStatus.INSUFFICIENT_DATA
        if len(fractions) != len(labels) or not fractions
        else QCStatus.PASS
    )
    return QCResult(
        "point_projection_coverage",
        status,
        {
            "snapshot_count": count,
            "camera_available_count": camera_count,
            "camera_frame": camera_frame.name,
            "world_frame": world_frame.name,
            "image_size": image_size,
            "points": by_point,
            "occlusion_evaluated": False,
            "actual_visibility_evaluated": False,
        },
        {"minimum_coverage_fraction": float(minimum_coverage_fraction)},
        "Calibrated in-image projection only; this does not establish actual visibility or absence of occlusion.",
        provenance,
    )
