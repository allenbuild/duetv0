"""Explicit quaternion conversion without normalization or invalid-input repair."""

from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray


def quaternion_xyzw_to_rotation(
    quaternion: ArrayLike, *, norm_tolerance: float = 1e-7
) -> NDArray[np.float64]:
    """Convert a unit quaternion ordered x,y,z,w to a column-vector rotation.

    norm_tolerance is an absolute tolerance for the quaternion norm. Rounding
    within that tolerance is retained; the quaternion is never normalized.
    Consumers constructing rigid transforms still validate the resulting matrix.
    """
    if np.iscomplexobj(quaternion):
        raise ValueError("quaternion must contain real values")
    values = np.asarray(quaternion, dtype=np.float64)
    if values.shape != (4,):
        raise ValueError("quaternion must contain four finite values ordered x,y,z,w")
    return quaternion_xyzw_batch_to_rotations(values.reshape(1, 4), norm_tolerance=norm_tolerance)[
        0
    ]


def quaternion_xyzw_batch_to_rotations(
    quaternions: ArrayLike, *, norm_tolerance: float = 1e-7
) -> NDArray[np.float64]:
    """Convert an Nx4 array of xyzw unit quaternions into Nx3x3 rotations.

    This is the same non-normalizing conversion as quaternion_xyzw_to_rotation,
    vectorized for bounded chunks of trajectories. Empty Nx4 batches are valid.
    Any invalid row rejects the whole batch; no rows are silently dropped.
    """
    if isinstance(norm_tolerance, (bool, np.bool_)) or not isinstance(norm_tolerance, Real):
        raise TypeError("quaternion norm tolerance must be a real number")
    if not np.isfinite(norm_tolerance) or norm_tolerance <= 0:
        raise ValueError("quaternion norm tolerance must be finite and positive")
    if np.iscomplexobj(quaternions):
        raise ValueError("quaternions must contain real values")
    values = np.asarray(quaternions, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 4 or not np.isfinite(values).all():
        raise ValueError("quaternions must be an Nx4 array of finite x,y,z,w values")
    with np.errstate(over="ignore", invalid="ignore"):
        norms = np.linalg.norm(values, axis=1)
    invalid = ~np.isfinite(norms) | (np.abs(norms - 1.0) > norm_tolerance)
    if invalid.any():
        indices = np.flatnonzero(invalid)[:5].tolist()
        raise ValueError(
            f"quaternions must have unit norm at rows {indices}; no implicit normalization"
        )
    x, y, z, w = values.T
    rotations = np.empty((len(values), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - z * w)
    rotations[:, 0, 2] = 2 * (x * z + y * w)
    rotations[:, 1, 0] = 2 * (x * y + z * w)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - x * w)
    rotations[:, 2, 0] = 2 * (x * z - y * w)
    rotations[:, 2, 1] = 2 * (y * z + x * w)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rotations
