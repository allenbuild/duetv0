"""Structured QC for rigid matrices and timestamped meter trajectories."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from numbers import Real

import numpy as np
from numpy.typing import ArrayLike

from duet.geometry.transforms import RigidTransform, rigid_matrix_errors
from duet.qc.result import QCResult, QCStatus
from duet.schemas.common import Provenance
from duet.schemas.time import Timestamp


def check_transform(
    matrix: ArrayLike,
    *,
    tolerance: float = 1e-8,
    provenance: tuple[Provenance, ...] = (),
) -> QCResult:
    """Report raw rigid-matrix validity without raising for an invalid matrix.

    This checks matrix geometry only; matrix values cannot establish transform
    direction or distance units. Invalid threshold configuration still raises.
    """
    errors = rigid_matrix_errors(matrix, tolerance=tolerance)
    metrics: dict[str, object] = {"errors": errors}
    if not errors:
        array = np.asarray(matrix, dtype=np.float64)
        metrics["rotation_determinant"] = float(np.linalg.det(array[:3, :3]))
        metrics["rotation_orthonormality_error"] = float(
            np.max(np.abs(array[:3, :3].T @ array[:3, :3] - np.eye(3)))
        )
    return QCResult(
        check="transform_validity",
        status=QCStatus.FAIL if errors else QCStatus.PASS,
        metrics=metrics,
        thresholds={"absolute_tolerance": tolerance},
        message="; ".join(errors),
        provenance=provenance,
    )


@dataclass(frozen=True)
class TrajectorySample:
    """A transform at its preserved timestamp; ``None`` means a missing pose."""

    timestamp: Timestamp
    transform: RigidTransform | None

    def __post_init__(self) -> None:
        if not isinstance(self.timestamp, Timestamp):
            raise TypeError("timestamp must be a Timestamp")
        if self.transform is not None and not isinstance(self.transform, RigidTransform):
            raise TypeError("transform must be a RigidTransform or None")


def check_trajectory_continuity(
    samples: Sequence[TrajectorySample],
    *,
    max_translation_speed_m_s: float,
    max_angular_speed_rad_s: float,
) -> QCResult:
    """Check consecutive observed transforms using explicit speed thresholds.

    Translation speed is displacement of the source origin in destination meters
    per second. Angular speed is the shortest relative rotation angle per second.
    Frames and clocks must agree; cross-clock pairs fail and are never compared.
    Callers may explicitly map timestamps first using a verified clock mapping.
    Input order is retained so duplicate or decreasing times are reported rather
    than hidden by sorting. Missing samples break continuity; no gap is bridged.
    """
    for name, threshold in (
        ("max_translation_speed_m_s", max_translation_speed_m_s),
        ("max_angular_speed_rad_s", max_angular_speed_rad_s),
    ):
        if isinstance(threshold, (bool, np.bool_)) or not isinstance(threshold, Real):
            raise TypeError(f"{name} must be a real number, not a boolean")
        try:
            value = float(threshold)
        except OverflowError as exc:
            raise ValueError(f"{name} must be finite and nonnegative") from exc
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    translation_speeds: list[float] = []
    angular_speeds: list[float] = []
    failed_intervals: list[int] = []
    invalid_intervals: list[dict[str, object]] = []
    missing_count = sum(
        sample.transform is None or sample.timestamp.seconds is None for sample in samples
    )
    for index, (previous, current) in enumerate(pairwise(samples), start=1):
        previous_time = previous.timestamp.seconds
        current_time = current.timestamp.seconds
        before = previous.transform
        after = current.transform
        if previous_time is None or current_time is None or before is None or after is None:
            continue
        reason = ""
        if previous.timestamp.clock_domain != current.timestamp.clock_domain:
            reason = "incompatible_clock_domains"
        elif before.source != after.source or before.destination != after.destination:
            reason = "incompatible_coordinate_frames"
        elif current_time <= previous_time:
            reason = "non_increasing_timestamp"
        if reason:
            invalid_intervals.append({"end_index": index, "reason": reason})
            continue
        try:
            delta_seconds = float(current_time - previous_time)
        except OverflowError:
            delta_seconds = float("inf")
        if not np.isfinite(delta_seconds) or delta_seconds <= 0:
            invalid_intervals.append(
                {"end_index": index, "reason": "timestamp_interval_not_representable_as_float"}
            )
            continue
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            # hypot avoids the squaring overflow/underflow of a naive Euclidean norm.
            translation_speed = float(
                np.hypot.reduce(after.matrix[:3, 3] - before.matrix[:3, 3]) / delta_seconds
            )
            relative_rotation = before.matrix[:3, :3].T @ after.matrix[:3, :3]
            cosine = (np.trace(relative_rotation) - 1) / 2
            sine = (
                np.hypot.reduce(
                    [
                        relative_rotation[2, 1] - relative_rotation[1, 2],
                        relative_rotation[0, 2] - relative_rotation[2, 0],
                        relative_rotation[1, 0] - relative_rotation[0, 1],
                    ]
                )
                / 2
            )
            # atan2 is stable for both zero and pi; no source matrix is changed.
            angular_speed = float(np.arctan2(sine, cosine) / delta_seconds)
        if not np.isfinite(translation_speed) or not np.isfinite(angular_speed):
            invalid_intervals.append({"end_index": index, "reason": "non_finite_motion"})
            continue
        translation_speeds.append(translation_speed)
        angular_speeds.append(angular_speed)
        if translation_speed > max_translation_speed_m_s or angular_speed > max_angular_speed_rad_s:
            failed_intervals.append(index)
    if invalid_intervals or failed_intervals:
        status = QCStatus.FAIL
    elif not translation_speeds or missing_count:
        status = QCStatus.INSUFFICIENT_DATA
    else:
        status = QCStatus.PASS
    return QCResult(
        check="trajectory_continuity",
        status=status,
        metrics={
            "sample_count": len(samples),
            "missing_sample_count": missing_count,
            "checked_interval_count": len(translation_speeds),
            "translation_speeds_m_s": tuple(translation_speeds),
            "angular_speeds_rad_s": tuple(angular_speeds),
            "max_observed_translation_speed_m_s": max(translation_speeds, default=None),
            "max_observed_angular_speed_rad_s": max(angular_speeds, default=None),
            "excessive_speed_interval_end_indices": tuple(failed_intervals),
            "invalid_intervals": tuple(invalid_intervals),
        },
        thresholds={
            "max_translation_speed_m_s": max_translation_speed_m_s,
            "max_angular_speed_rad_s": max_angular_speed_rad_s,
        },
        message=("Checks use consecutive observed pairs only; missing samples are not bridged."),
        provenance=tuple(
            provenance
            for sample in samples
            for provenance in (
                (sample.timestamp.provenance,)
                if sample.transform is None
                else (sample.timestamp.provenance, sample.transform.provenance)
            )
        ),
    )
