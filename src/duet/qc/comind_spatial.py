"""Vectorized spatial diagnostics using each participant's own device clock.

Thresholds are configurable diagnostic flags, not proof of anatomical validity.
World z is relative to the SLAM origin; no floor/height calibration is assumed.
"""

import numpy as np
from numpy.typing import ArrayLike

from duet.adapters.comind.timing import integer_timestamps, percentiles
from duet.qc.result import QCResult, QCStatus
from duet.schemas.common import Provenance
from duet.synchronization.matching import nonnegative_seconds


def check_device_motion(
    device_us: ArrayLike,
    positions_m: ArrayLike,
    *,
    max_speed_m_s: float,
    max_step_m: float,
    max_gap_seconds: float,
    provenance: tuple[Provenance, ...] = (),
) -> QCResult:
    """Check consecutive device-origin positions without bridging large time gaps."""
    maximum_speed = float(nonnegative_seconds(max_speed_m_s, "max_speed_m_s"))
    maximum_step = float(nonnegative_seconds(max_step_m, "max_step_m"))
    maximum_gap = float(nonnegative_seconds(max_gap_seconds, "max_gap_seconds"))
    times = integer_timestamps(device_us, "device_us")
    if np.iscomplexobj(positions_m):
        raise ValueError("device positions must contain real values")
    points = np.asarray(positions_m, dtype=np.float64)
    if points.shape != (len(times), 3) or not np.isfinite(points).all():
        raise ValueError("device positions must be finite Nx3 meter coordinates")
    dt = np.diff(times).astype(np.float64) / 1e6
    steps = np.linalg.norm(np.diff(points, axis=0), axis=1)
    usable = (dt > 0) & (dt <= maximum_gap)
    speeds = steps[usable] / dt[usable]
    excessive_steps = steps > maximum_step
    excessive_speed_count = int((speeds > maximum_speed).sum())
    invalid_times = int((dt <= 0).sum())
    gap_count = int((dt > maximum_gap).sum())
    failed = excessive_speed_count or excessive_steps.any() or invalid_times
    status = (
        QCStatus.FAIL
        if failed
        else QCStatus.INSUFFICIENT_DATA
        if not usable.any() or gap_count
        else QCStatus.PASS
    )
    bounds = (
        None
        if not len(points)
        else {"min": points.min(axis=0).tolist(), "max": points.max(axis=0).tolist()}
    )
    return QCResult(
        "comind_device_motion",
        status,
        {
            "sample_count": len(times),
            "evaluated_interval_count": int(usable.sum()),
            "non_increasing_count": invalid_times,
            "excessive_gap_count": gap_count,
            "speed_m_s": percentiles(speeds),
            "step_m": percentiles(steps),
            "excessive_speed_count": excessive_speed_count,
            "excessive_step_count": int(excessive_steps.sum()),
            "bbox_m": bounds,
            "world_z_m": percentiles(points[:, 2]),
            "height_above_floor_available": False,
            "net_displacement_m": None
            if not len(points)
            else float(np.linalg.norm(points[-1] - points[0])),
            "path_length_m": float(steps[usable].sum()),
        },
        {
            "max_speed_m_s": maximum_speed,
            "max_step_m": maximum_step,
            "max_gap_seconds": maximum_gap,
        },
        "Dynamics use device time; world z is not height above a verified floor.",
        provenance,
    )


def check_hand_plausibility(
    device_us: ArrayLike,
    points_device_m: ArrayLike,
    confidence_raw: ArrayLike,
    *,
    max_device_distance_m: float,
    max_wrist_speed_m_s: float,
    max_gap_seconds: float,
    provenance: tuple[Provenance, ...] = (),
) -> QCResult:
    """Check one side's 21 device-frame landmarks; missing hands break continuity.

    Confidence -1 is absent; zero remains present. Anatomical wrist index 5 is
    the documented Aria ordering. Device-relative speed includes head motion.
    """
    distance_limit = float(nonnegative_seconds(max_device_distance_m, "max_device_distance_m"))
    speed_limit = float(nonnegative_seconds(max_wrist_speed_m_s, "max_wrist_speed_m_s"))
    gap_limit = float(nonnegative_seconds(max_gap_seconds, "max_gap_seconds"))
    times = integer_timestamps(device_us, "device_us")
    if np.iscomplexobj(points_device_m) or np.iscomplexobj(confidence_raw):
        raise ValueError("hand points and confidence must contain real values")
    points = np.asarray(points_device_m, dtype=np.float64)
    confidence = np.asarray(confidence_raw, dtype=np.float64)
    if points.shape != (len(times), 21, 3) or confidence.shape != (len(times),):
        raise ValueError("hand arrays require Nx21x3 points and N confidence values")
    if np.isinf(confidence).any() or np.any(
        np.isfinite(confidence) & (confidence != -1) & ((confidence < 0) | (confidence > 1))
    ):
        raise ValueError("hand confidence must be missing -1, NaN, or in [0, 1]")
    finite_geometry = np.isfinite(points).all(axis=(1, 2))
    explicitly_missing_geometry = np.isnan(points).all(axis=(1, 2))
    if np.any((confidence != -1) & ~(finite_geometry | explicitly_missing_geometry)):
        raise ValueError("hand coordinates must be finite or an explicitly all-NaN missing hand")
    present = (confidence != -1) & finite_geometry
    distances = np.linalg.norm(points[present], axis=2)
    wrist_distances = np.linalg.norm(points[present, 5], axis=1)
    dt = np.diff(times).astype(np.float64) / 1e6
    adjacent = present[1:] & present[:-1] & (dt > 0) & (dt <= gap_limit)
    wrist_steps = np.linalg.norm(np.diff(points[:, 5], axis=0)[adjacent], axis=1)
    speeds = wrist_steps / dt[adjacent]
    distance_outliers = int((distances > distance_limit).any(axis=1).sum())
    speed_outliers = int((speeds > speed_limit).sum())
    speed_outlier_indices = (np.flatnonzero(adjacent) + 1)[speeds > speed_limit]
    non_increasing_count = int((dt <= 0).sum())
    status = (
        QCStatus.FAIL
        if distance_outliers or speed_outliers or non_increasing_count
        else QCStatus.INSUFFICIENT_DATA
        if not present.any()
        else QCStatus.PASS
    )
    return QCResult(
        "comind_hand_plausibility",
        status,
        {
            "sample_count": len(times),
            "present_count": int(present.sum()),
            "missing_count": int((~present).sum()),
            "zero_confidence_count": int((confidence == 0).sum()),
            "unknown_confidence_count": int(np.isnan(confidence).sum()),
            "confidence": percentiles(confidence[present]),
            "landmark_device_distance_m": percentiles(distances),
            "wrist_device_distance_m": percentiles(wrist_distances),
            "device_relative_wrist_speed_m_s": percentiles(speeds),
            "device_relative_wrist_step_m": percentiles(wrist_steps),
            "evaluated_continuity_interval_count": int(adjacent.sum()),
            "distance_outlier_sample_count": distance_outliers,
            "speed_outlier_interval_count": speed_outliers,
            "speed_outlier_interval_end_indices_first_20": speed_outlier_indices[:20].tolist(),
            "speed_outlier_interval_end_device_us_first_20": times[
                speed_outlier_indices[:20]
            ].tolist(),
            "non_increasing_timestamp_count": non_increasing_count,
            "continuity_missing_or_gap_interval_count": int((~adjacent).sum()),
        },
        {
            "max_device_distance_m": distance_limit,
            "max_wrist_speed_m_s": speed_limit,
            "max_gap_seconds": gap_limit,
        },
        "Missing hands and excessive timestamp gaps are never bridged; confidence is preserved.",
        provenance,
    )
