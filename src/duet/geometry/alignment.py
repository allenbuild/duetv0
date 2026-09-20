"""Robust fixed-scale world alignment from already matched poses of one device.

This estimates a transform; statistical fit does not establish dataset semantics
or verify that another asset belongs to either input world. Timestamp matching,
clock identity, and correspondence provenance remain the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from duet.geometry.transforms import RigidTransform
from duet.schemas.common import DistanceUnit, FrameId, Provenance


def _positive(value: float, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


def _array(value: ArrayLike, shape: tuple[int, ...], name: str) -> NDArray[np.float64]:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real values")
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(shape)


@dataclass(frozen=True, eq=False)
class PosePairs:
    """Matched T_sourceWorld_device and T_destinationWorld_device observations.

    Positions are explicitly meters; rotations are column-vector device-to-world
    rotations. Both observations must describe the same device at matched times.
    Arrays are immutable. No unit conversion or rotation repair occurs here.
    """

    source_positions: NDArray[np.float64] = field(repr=False)
    destination_positions: NDArray[np.float64] = field(repr=False)
    source_rotations: NDArray[np.float64] = field(repr=False)
    destination_rotations: NDArray[np.float64] = field(repr=False)
    source_world: FrameId
    destination_world: FrameId
    device_frame: FrameId
    unit: DistanceUnit
    provenance: Provenance
    rotation_tolerance: float = 1e-7

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, FrameId)
            for value in (self.source_world, self.destination_world, self.device_frame)
        ):
            raise TypeError("all pose frames must be explicit FrameId values")
        if self.source_world == self.destination_world:
            raise ValueError("alignment requires distinct world frames")
        if self.device_frame in (self.source_world, self.destination_world):
            raise ValueError("device frame must differ from both world frames")
        if self.unit is not DistanceUnit.METERS:
            raise ValueError("pose alignment requires explicit meters")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("pose correspondences require provenance")
        tolerance = _positive(self.rotation_tolerance, "rotation tolerance")
        raw_positions = np.asarray(self.source_positions)
        if raw_positions.ndim != 2 or raw_positions.shape[1:] != (3,):
            raise ValueError("source positions must have shape (N, 3)")
        count = len(raw_positions)
        for name in ("source_positions", "destination_positions"):
            object.__setattr__(self, name, _array(getattr(self, name), (count, 3), name))
        for name in ("source_rotations", "destination_rotations"):
            rotations = _array(getattr(self, name), (count, 3, 3), name)
            with np.errstate(over="ignore", invalid="ignore"):
                errors = np.abs(rotations.transpose(0, 2, 1) @ rotations - np.eye(3))
                determinant = np.linalg.det(rotations)
            if np.any(errors > tolerance) or not np.all(
                np.isfinite(determinant) & (np.abs(determinant - 1) <= tolerance)
            ):
                raise ValueError(f"{name} must contain valid SO(3) rotations; no repairs")
            object.__setattr__(self, name, rotations)

    def subset(self, indices: ArrayLike) -> PosePairs:
        """Select correspondence rows without changing their frame identities."""
        return PosePairs(
            self.source_positions[indices],
            self.destination_positions[indices],
            self.source_rotations[indices],
            self.destination_rotations[indices],
            self.source_world,
            self.destination_world,
            self.device_frame,
            self.unit,
            self.provenance,
            self.rotation_tolerance,
        )


@dataclass(frozen=True)
class PoseAlignment:
    """An estimated T_destinationWorld_sourceWorld, never a semantic verification."""

    transform: RigidTransform
    converged: bool
    iterations: int
    downweighted_pairs: int
    orientation_length_scale_m: float
    huber_delta_m: float


def _proper_rotation(covariance: NDArray[np.float64]) -> NDArray[np.float64]:
    """Solve the estimator's SO(3) optimization, not input-matrix repair."""
    if not np.isfinite(covariance).all():
        raise ValueError("alignment covariance overflowed")
    left, _, right = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def _combined_residual(
    pairs: PosePairs, rotation: NDArray, translation: NDArray, length: float
) -> NDArray:
    with np.errstate(over="ignore", invalid="ignore"):
        position = pairs.source_positions @ rotation.T + translation - pairs.destination_positions
        orientation = rotation @ pairs.source_rotations - pairs.destination_rotations
        residual = np.sqrt(
            np.sum(position**2, axis=1) + length**2 / 2 * np.sum(orientation**2, axis=(1, 2))
        )
    if not np.isfinite(residual).all():
        raise ValueError("alignment residual overflowed")
    return residual


def fit_pose_alignment(
    pairs: PosePairs,
    *,
    orientation_length_scale_m: float,
    huber_delta_m: float,
    max_iterations: int = 100,
    convergence_tolerance: float = 1e-10,
) -> PoseAlignment:
    """Estimate one fixed-scale SE(3) using positions AND pose orientations.

    IRLS minimizes a Huber loss of sqrt(position_error_m**2 +
    orientation_length_scale_m**2 / 2 * ||R R_source - R_destination||_F**2).
    The length scale makes the orientation/translation tradeoff explicit. A
    median relative-rotation matrix and median translation initialize the fit.
    Every iteration solves the weighted joint SO(3) objective by SVD. Scale is
    fixed to one; convergence and downweighting are returned, not hidden.
    """
    if not isinstance(pairs, PosePairs) or len(pairs.source_positions) < 3:
        raise ValueError("at least three explicit pose correspondences are required")
    length = _positive(orientation_length_scale_m, "orientation length scale")
    delta = _positive(huber_delta_m, "Huber delta")
    tolerance = _positive(convergence_tolerance, "convergence tolerance")
    if (
        isinstance(max_iterations, bool)
        or not isinstance(max_iterations, int)
        or max_iterations < 1
    ):
        raise ValueError("max_iterations must be a positive integer")
    relatives = pairs.destination_rotations @ pairs.source_rotations.transpose(0, 2, 1)
    rotation = _proper_rotation(np.median(relatives, axis=0))
    translation = np.median(
        pairs.destination_positions - pairs.source_positions @ rotation.T, axis=0
    )
    converged = False
    for iteration in range(1, max_iterations + 1):
        residual = _combined_residual(pairs, rotation, translation, length)
        weights = np.minimum(1.0, delta / np.maximum(residual, np.finfo(float).tiny))
        weights /= weights.sum()
        source_mean = weights @ pairs.source_positions
        destination_mean = weights @ pairs.destination_positions
        source_centered = pairs.source_positions - source_mean
        destination_centered = pairs.destination_positions - destination_mean
        with np.errstate(over="ignore", invalid="ignore"):
            covariance = (destination_centered * weights[:, None]).T @ source_centered
            covariance += length**2 / 2 * np.einsum("n,nij->ij", weights, relatives)
        next_rotation = _proper_rotation(covariance)
        next_translation = destination_mean - next_rotation @ source_mean
        change = max(
            float(np.linalg.norm(next_translation - translation)),
            float(np.linalg.norm(next_rotation - rotation)),
        )
        rotation, translation = next_rotation, next_translation
        if change <= tolerance:
            converged = True
            break
    residual = _combined_residual(pairs, rotation, translation, length)
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, translation
    transform = RigidTransform(
        matrix,
        pairs.source_world,
        pairs.destination_world,
        DistanceUnit.METERS,
        Provenance(
            "duet.geometry.fit_pose_alignment",
            "Estimated joint pose fit, scale fixed at 1; no asset-frame verification",
            parents=(pairs.provenance,),
        ),
    )
    return PoseAlignment(
        transform, converged, iteration, int(np.count_nonzero(residual > delta)), length, delta
    )


def rotation_angles(rotations: ArrayLike) -> NDArray[np.float64]:
    """Geodesic rotation magnitudes in radians for already validated rotations.

    atan2 of skew magnitude and trace is stable near zero and pi. This helper
    measures residual products; it does not normalize them.
    """
    if np.iscomplexobj(rotations):
        raise ValueError("rotation residuals must contain real values")
    values = np.asarray(rotations, dtype=float)
    if values.ndim < 2 or values.shape[-2:] != (3, 3) or not np.isfinite(values).all():
        raise ValueError("rotation residuals must be finite (..., 3, 3) arrays")
    skew = np.stack(
        (
            values[..., 2, 1] - values[..., 1, 2],
            values[..., 0, 2] - values[..., 2, 0],
            values[..., 1, 0] - values[..., 0, 1],
        ),
        axis=-1,
    )
    sine = np.linalg.norm(skew, axis=-1) / 2
    cosine = (np.trace(values, axis1=-2, axis2=-1) - 1) / 2
    return np.arctan2(sine, cosine)


def pose_alignment_residuals(
    pairs: PosePairs, transform: RigidTransform
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return translation meters and orientation radians for each matched pair."""
    if transform.source != pairs.source_world or transform.destination != pairs.destination_world:
        raise ValueError("alignment transform does not match correspondence world frames")
    predicted_positions = transform.apply(pairs.source_positions, unit=DistanceUnit.METERS)
    with np.errstate(over="ignore", invalid="ignore"):
        distances = np.linalg.norm(predicted_positions - pairs.destination_positions, axis=1)
    errors = (transform.matrix[:3, :3] @ pairs.source_rotations).transpose(
        0, 2, 1
    ) @ pairs.destination_rotations
    angles = rotation_angles(errors)
    if not np.isfinite(distances).all() or not np.isfinite(angles).all():
        raise ValueError("pose residuals overflowed")
    return distances, angles
