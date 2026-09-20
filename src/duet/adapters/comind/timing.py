"""Paired device/RTC evidence without claiming accurate cross-device synchronization.

Project Aria documents RTC UTC as a rough estimate that can be adjusted by NTP.
An exact source-row pair is verified evidence of the reported wall-clock value;
neither a small affine-fit residual nor nanosecond storage verifies UTC accuracy.
"""

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from numpy.typing import ArrayLike, NDArray

from duet.adapters.comind.mps import device_clock
from duet.schemas.common import Provenance
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit
from duet.synchronization.matching import nonnegative_seconds

UTC_EVIDENCE = (
    "https://facebookresearch.github.io/projectaria_tools/docs/data_formats/mps/mps_summary"
)
TIMECODE_EVIDENCE = "https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/timestamps_in_aria_vrs"


def integer_timestamps(values: ArrayLike, name: str) -> NDArray[np.int64]:
    """Require a one-dimensional signed-integer array, without float coercion."""
    array = np.asarray(values)
    if array.ndim == 1 and not array.size:
        return np.empty(0, dtype=np.int64)
    if array.ndim != 1 or array.dtype.kind != "i":
        raise ValueError(f"{name} must be a one-dimensional signed-integer array")
    if array.size and int(array.max()) - int(array.min()) > np.iinfo(np.int64).max:
        raise ValueError(f"{name} span exceeds safe signed 64-bit differencing")
    return np.asarray(array, dtype=np.int64)


def percentiles(values: ArrayLike) -> dict[str, float | int | None]:
    """Compact finite-value distribution; absent measurements stay None."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = values[np.isfinite(values)]
    result: dict[str, float | int | None] = {"count": len(finite)}
    for key, quantile in (("min", 0), ("p50", 50), ("p95", 95), ("p99", 99), ("max", 100)):
        result[key] = None if not len(finite) else float(np.percentile(finite, quantile))
    return result


@dataclass(frozen=True)
class ReportedUtcLookup:
    """A paired source sample, with sampling residual and unknown UTC accuracy."""

    query: Timestamp
    paired_device_timestamp: Timestamp
    reported_utc: Timestamp | None
    source_index: int
    residual_seconds: Fraction
    accepted: bool
    uncertainty_seconds: None = field(default=None, init=False)
    cross_device_synchronization_verified: bool = field(default=False, init=False)


@dataclass(frozen=True, eq=False)
class ReportedUtcPairs:
    """Exact source-row pairs in participant-scoped clocks; no affine interpolation.

    -1 is the documented missing UTC sentinel. Device -1 remains a real integer.
    Duplicate device values are allowed only if their reported UTC values agree.
    No constructor can promote these pairs to a verified common clock.
    """

    recording_id: str
    participant: str
    device_us: ArrayLike
    reported_utc_ns: ArrayLike
    provenance: Provenance

    def __post_init__(self) -> None:
        device_clock(self.recording_id, self.participant)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("paired timestamp evidence requires provenance")
        device = integer_timestamps(self.device_us, "device_us")
        utc = integer_timestamps(self.reported_utc_ns, "reported_utc_ns")
        if len(device) != len(utc):
            raise ValueError("device and UTC source rows must have equal lengths")
        if np.any(device[1:] < device[:-1]):
            raise ValueError("device timestamps must be nondecreasing")
        if np.any((device[1:] == device[:-1]) & (utc[1:] != utc[:-1])):
            raise ValueError("duplicate device timestamps have conflicting reported UTC values")
        for name, values in (("device_us", device), ("reported_utc_ns", utc)):
            immutable = np.frombuffer(values.tobytes(), dtype=np.int64)
            object.__setattr__(self, name, immutable)

    @property
    def reported_clock(self) -> ClockDomain:
        return ClockDomain(f"comind/{self.recording_id}/{self.participant}/utc_unverified")

    def lookup(self, query: Timestamp, *, max_gap_seconds: float | Fraction) -> ReportedUtcLookup:
        """Return nearest paired UTC without shifting/interpolating its original value.

        Equal distances prefer earlier device time, then its first source index.
        A rejected candidate retains its sampling residual; UTC accuracy is unknown.
        """
        maximum = nonnegative_seconds(max_gap_seconds, "max_gap_seconds")
        if query.clock_domain != device_clock(self.recording_id, self.participant):
            raise ValueError("query must use the same participant's device clock")
        if query.seconds is None:
            raise ValueError("missing query timestamp cannot be converted")
        if not len(self.device_us):
            raise ValueError("no source timestamp pairs are available")
        target_us = query.seconds * 1_000_000
        # Search with exact integer ceil, avoiding epoch-to-float conversion.
        ceiling = -(-target_us.numerator // target_us.denominator)
        right = int(np.searchsorted(self.device_us, ceiling, side="left"))
        candidates = []
        if right < len(self.device_us):
            candidates.append(right)
        if right:
            candidates.append(
                int(np.searchsorted(self.device_us, self.device_us[right - 1], side="left"))
            )
        index = min(
            candidates,
            key=lambda i: (abs(int(self.device_us[i]) - target_us), int(self.device_us[i]), i),
        )
        paired = Timestamp(
            int(self.device_us[index]), TimeUnit.MICROSECONDS, query.clock_domain, self.provenance
        )
        residual = paired.seconds - query.seconds
        raw_utc = int(self.reported_utc_ns[index])
        utc = (
            None
            if raw_utc == -1
            else Timestamp(raw_utc, TimeUnit.NANOSECONDS, self.reported_clock, self.provenance)
        )
        return ReportedUtcLookup(
            query, paired, utc, index, residual, utc is not None and abs(residual) <= maximum
        )

    def exact(self, query: Timestamp) -> Timestamp:
        """Return a verified source-row UTC value only for an exact paired device time."""
        result = self.lookup(query, max_gap_seconds=0)
        if not result.accepted or result.reported_utc is None:
            raise ValueError("query has no exact nonmissing reported UTC source pair")
        return result.reported_utc


def clock_pair_statistics(
    device_us: ArrayLike,
    utc_ns: ArrayLike,
    *,
    offset_jump_threshold_seconds: float,
) -> dict[str, object]:
    """Quantify recorded clock behavior; affine fit is diagnostic, never a mapping.

    Fits use centered integer differences before float conversion. UTC source
    values and offset extrema remain exact integers, even at large epochs.
    """
    threshold = nonnegative_seconds(offset_jump_threshold_seconds, "offset_jump_threshold_seconds")
    device = integer_timestamps(device_us, "device_us")
    utc = integer_timestamps(utc_ns, "utc_ns")
    if len(device) != len(utc):
        raise ValueError("timestamp columns must have equal lengths")
    dt = np.diff(device)
    valid = utc != -1
    result: dict[str, object] = {
        "pair_count": len(device),
        "missing_utc_count": int((~valid).sum()),
        "device_range_us": None if not len(device) else [int(device.min()), int(device.max())],
        "reported_utc_range_ns": None
        if not valid.any()
        else [int(utc[valid].min()), int(utc[valid].max())],
        "device_decreasing_count": int((dt < 0).sum()),
        "device_duplicate_count": int((dt == 0).sum()),
        "device_step_us": percentiles(dt),
        "cross_device_synchronization_verified": False,
        "utc_accuracy_bound_seconds": None,
        "offset_jump_threshold_seconds": float(threshold),
        "evidence": [UTC_EVIDENCE, TIMECODE_EVIDENCE],
        "fit_classification": "INFERRED description of recorded pairs, not a synchronization calibration",
        "conversion_model_installed": False,
    }
    if valid.sum() < 2:
        result["fit"] = None
        return result
    d, u = device[valid], utc[valid]
    device_span = int(d.max()) - int(d.min())
    utc_span = int(u.max()) - int(u.min())
    if device_span * 1000 + utc_span > np.iinfo(np.int64).max:
        raise ValueError("device span too large for exact nanosecond differences")
    device_delta_ns = (d - d[0]) * 1000
    utc_delta_ns = u - u[0]
    offset_delta_ns = utc_delta_ns - device_delta_ns
    offset_origin = int(u[0]) - int(d[0]) * 1000
    adjacent_valid = valid[1:] & valid[:-1]
    utc_steps = np.diff(utc)[adjacent_valid]
    offset_steps = utc_steps - dt[adjacent_valid] * 1000
    x = device_delta_ns.astype(np.float64) / 1e9
    y = offset_delta_ns.astype(np.float64)
    centered = x - x.mean()
    denominator = float(centered @ centered)
    slope = 0.0 if denominator == 0 else float(centered @ (y - y.mean()) / denominator)
    intercept = float(y.mean() - slope * x.mean())
    residual_ns = y - (intercept + slope * x)
    result.update(
        {
            "reported_utc_decreasing_count": int((utc_steps < 0).sum()),
            "reported_utc_duplicate_count": int((utc_steps == 0).sum()),
            "reported_utc_unique_value_count": len(np.unique(u)),
            "reported_utc_plateaus_observed": bool(np.any(utc_steps == 0)),
            "mapping_behavior": "Direct source-row pairs only; repeated UTC values are retained. No interpolation or affine conversion is installed.",
            "reported_utc_step_ns": percentiles(utc_steps),
            "offset_origin_ns": offset_origin,
            "offset_range_ns": [
                offset_origin + int(offset_delta_ns.min()),
                offset_origin + int(offset_delta_ns.max()),
            ],
            "offset_net_change_ns": int(offset_delta_ns[-1]),
            "offset_net_change_interpretation": "Endpoint difference includes staircase phase; it is not a calibrated physical drift estimate.",
            "offset_jump_count": int((np.abs(offset_steps) > float(threshold) * 1e9).sum()),
            "offset_step_ns": percentiles(offset_steps),
            "offset_step_interpretation": "Changes of UTC minus device time between adjacent rows, including staircase quantization; these are not independently verified clock-reset events.",
            "fit": {
                "origin_device_us": int(d[0]),
                "origin_reported_utc_ns": int(u[0]),
                "offset_intercept_correction_ns": intercept,
                "slope_drift_ppm": slope / 1000,
                "absolute_residual_ns": percentiles(np.abs(residual_ns)),
                "rms_residual_ns": float(np.sqrt(np.mean(residual_ns**2))),
                "external_accuracy_verified": False,
            },
        }
    )
    return result
