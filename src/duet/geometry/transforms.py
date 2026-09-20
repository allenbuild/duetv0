"""Rigid geometry using column-vector ``T_destination_source`` conventions.

Canonical transform translations and transformed points are in meters. Conversion
from another verified distance unit must be explicitly requested through
``RigidTransform.from_matrix``. No validation path repairs an invalid matrix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike, NDArray

from duet.schemas.common import DistanceUnit, FrameId, Provenance


def rigid_matrix_errors(matrix: ArrayLike, *, tolerance: float = 1e-8) -> tuple[str, ...]:
    """Return validation failures without modifying or repairing ``matrix``.

    ``tolerance`` is an absolute tolerance for rotation orthonormality, determinant
    +1, and the homogeneous final row. Translation units cannot be inferred from
    a matrix; callers must provide those separately.
    """
    if isinstance(tolerance, (bool, np.bool_)) or not isinstance(tolerance, Real):
        raise TypeError("tolerance must be a real number, not a boolean")
    try:
        tolerance = float(tolerance)
    except OverflowError as exc:
        raise ValueError("tolerance must be finite and positive") from exc
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise ValueError("tolerance must be finite and positive")
    try:
        source_array = np.asarray(matrix)
        if np.iscomplexobj(source_array):
            return ("matrix must contain real numbers",)
        array = np.asarray(source_array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return ("matrix must contain real numbers",)
    if array.shape != (4, 4):
        return (f"matrix must have shape (4, 4), received {array.shape}",)
    if not np.all(np.isfinite(array)):
        return ("matrix must contain only finite values",)
    errors: list[str] = []
    if not np.allclose(array[3], (0, 0, 0, 1), atol=tolerance, rtol=0):
        errors.append("homogeneous final row must be [0, 0, 0, 1]")
    rotation = array[:3, :3]
    with np.errstate(over="ignore", invalid="ignore"):
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=tolerance, rtol=0):
            errors.append("rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1, atol=tolerance, rtol=0):
            errors.append("rotation determinant must be +1 (reflections are invalid)")
    return tuple(errors)


@dataclass(frozen=True, eq=False)
class RigidTransform:
    """Map source points to destination points with ``T_destination_source``.

    The operation is ``p_destination = R @ p_source + t`` with ``t`` in meters.
    A transform contains no time semantics; attach timestamps to samples, and
    build separate frame graphs for distinct snapshots when transforms vary.
    Input matrices are defensively copied into immutable storage.
    """

    matrix: NDArray[np.float64] = field(repr=False)
    source: FrameId
    destination: FrameId
    unit: DistanceUnit
    provenance: Provenance
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        if not isinstance(self.source, FrameId) or not isinstance(self.destination, FrameId):
            raise TypeError("source and destination must be explicit FrameId values")
        if self.unit is not DistanceUnit.METERS:
            raise ValueError("canonical transforms require meters; use from_matrix to convert")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("transform provenance must be supplied")
        errors = rigid_matrix_errors(self.matrix, tolerance=self.tolerance)
        if errors:
            raise ValueError("; ".join(errors))
        object.__setattr__(self, "tolerance", float(self.tolerance))
        array = np.asarray(self.matrix, dtype=np.float64)
        if self.source == self.destination and not np.allclose(
            array, np.eye(4), atol=self.tolerance, rtol=0
        ):
            raise ValueError("a transform from a frame to itself must be identity")
        # A bytes-backed array cannot be made writable again with setflags.
        immutable = np.frombuffer(array.tobytes(), dtype=np.float64).reshape(4, 4)
        object.__setattr__(self, "matrix", immutable)

    @classmethod
    def from_matrix(
        cls,
        matrix: ArrayLike,
        *,
        source: FrameId,
        destination: FrameId,
        unit: DistanceUnit,
        provenance: Provenance,
        tolerance: float = 1e-8,
    ) -> RigidTransform:
        """Explicitly convert translation from a verified source unit to meters.

        Rotation and the homogeneous row are validated before conversion and
        are never normalized. The original provenance remains in the lineage.
        """
        if not isinstance(unit, DistanceUnit):
            raise TypeError("unit must be a DistanceUnit")
        errors = rigid_matrix_errors(matrix, tolerance=tolerance)
        if errors:
            raise ValueError("; ".join(errors))
        converted = np.array(matrix, dtype=np.float64, copy=True)
        converted[:3, 3] *= unit.factor_to_meters
        if unit is not DistanceUnit.METERS:
            provenance = Provenance(
                source="duet.geometry.unit_conversion",
                detail=f"translation converted from {unit.value} to m",
                parents=(provenance,),
            )
        return cls(converted, source, destination, DistanceUnit.METERS, provenance, tolerance)

    @classmethod
    def identity(cls, frame: FrameId, *, provenance: Provenance) -> RigidTransform:
        """Construct an identity transform within an explicitly named frame."""
        return cls(np.eye(4), frame, frame, DistanceUnit.METERS, provenance)

    def inverse(self) -> RigidTransform:
        """Return ``T_source_destination`` with inherited provenance."""
        inverted = np.eye(4)
        rotation = self.matrix[:3, :3]
        inverted[:3, :3] = rotation.T
        with np.errstate(over="ignore", invalid="ignore"):
            inverted[:3, 3] = -rotation.T @ self.matrix[:3, 3]
        return RigidTransform(
            inverted,
            self.destination,
            self.source,
            self.unit,
            Provenance("duet.geometry.inverse", parents=(self.provenance,)),
            self.tolerance,
        )

    def compose(self, inner: RigidTransform) -> RigidTransform:
        """Return ``self @ inner`` only if ``inner.destination == self.source``.

        For example ``T_world_camera.compose(T_camera_hand)`` yields
        ``T_world_hand``. Numerical validation also applies to the result.
        """
        if not isinstance(inner, RigidTransform):
            raise TypeError("inner must be a RigidTransform")
        if inner.destination != self.source:
            raise ValueError(
                f"incompatible frames: inner destination {inner.destination.name!r} "
                f"does not match outer source {self.source.name!r}"
            )
        if self.unit is not inner.unit:
            raise ValueError("cannot compose transforms with different distance units")
        with np.errstate(over="ignore", invalid="ignore"):
            composed = self.matrix @ inner.matrix
        return RigidTransform(
            composed,
            inner.source,
            self.destination,
            self.unit,
            Provenance("duet.geometry.compose", parents=(self.provenance, inner.provenance)),
            max(self.tolerance, inner.tolerance),
        )

    def apply(self, points: ArrayLike, *, unit: DistanceUnit) -> NDArray[np.float64]:
        """Map source-frame points of shape ``(..., 3)`` into destination meters.

        Points must already be in meters and the caller must explicitly declare
        their unit. Their frame is the transform's named source frame.
        """
        if unit is not DistanceUnit.METERS:
            raise ValueError("points must be explicitly expressed in meters")
        source_array = np.asarray(points)
        if np.iscomplexobj(source_array):
            raise ValueError("points must contain real numbers")
        array = np.asarray(source_array, dtype=np.float64)
        if array.ndim < 1 or array.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        if not np.all(np.isfinite(array)):
            raise ValueError("points must contain only finite values")
        with np.errstate(over="ignore", invalid="ignore"):
            transformed = array @ self.matrix[:3, :3].T + self.matrix[:3, 3]
        if not np.all(np.isfinite(transformed)):
            raise ValueError("transformed points overflowed finite floating-point geometry")
        return transformed
