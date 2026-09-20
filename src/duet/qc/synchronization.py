"""Structured timestamp QC with explicit second-based thresholds.

Clock incompatibility and invalid thresholds raise ValueError (or TypeError for
invalid threshold types), rather than being reported as ordinary data quality
failures. Missing evidence is INSUFFICIENT_DATA unless an observed failure exists.
"""

from collections.abc import Sequence
from decimal import Decimal
from fractions import Fraction

from duet.qc.result import QCResult, QCStatus
from duet.schemas.time import ClockDomain, Timestamp, VerifiedClockMapping
from duet.synchronization.matching import (
    MatchStatus,
    NearestTimestampMatch,
    nonnegative_seconds,
)
from duet.synchronization.timeline import order_timestamps, timestamp_overlap


def check_timestamp_overlap(
    streams: Sequence[Sequence[Timestamp | None]],
    *,
    clock_domain: ClockDomain,
    minimum_overlap_seconds: Fraction | float | Decimal,
    mappings: Sequence[VerifiedClockMapping] = (),
) -> QCResult:
    """Check inclusive stream extents; continuity within the interval is a separate QC."""
    minimum = nonnegative_seconds(minimum_overlap_seconds, "minimum_overlap_seconds")
    overlap = timestamp_overlap(streams, clock_domain=clock_domain, mappings=mappings)
    duration = overlap.duration_seconds
    if len(overlap.ranges) < 2 or any(interval is None for interval in overlap.ranges):
        status = QCStatus.INSUFFICIENT_DATA
        message = "At least two nonempty streams are needed to assess timestamp overlap"
    elif duration is None:
        status = QCStatus.FAIL
        message = "Stream timestamp ranges do not overlap"
    elif duration < minimum:
        status = QCStatus.FAIL
        message = "Timestamp overlap is shorter than the configured minimum"
    else:
        status = QCStatus.PASS
        message = "Stream timestamp extents meet the configured overlap minimum"
    return QCResult(
        check="timestamp_overlap",
        status=status,
        metrics={
            "clock_domain": clock_domain.name,
            "stream_ranges": overlap.ranges,
            "start_seconds": overlap.start_seconds,
            "end_seconds": overlap.end_seconds,
            "duration_seconds": duration,
            "missing_counts": overlap.missing_counts,
            "duplicate_counts": overlap.duplicate_counts,
        },
        thresholds={"minimum_overlap_seconds": minimum},
        message=message,
        provenance=overlap.provenance,
    )


def check_nearest_residuals(
    matches: Sequence[NearestTimestampMatch],
    *,
    max_residual_seconds: Fraction | float | Decimal,
) -> QCResult:
    """Check candidate residuals, retaining gap rejections and missing-match evidence.

    A candidate rejected by its original match gap is always a failure, even if
    this check's residual threshold is looser. Unevaluated matches prevent PASS.
    """
    maximum = nonnegative_seconds(max_residual_seconds, "max_residual_seconds")
    residuals = tuple(
        match.signed_residual_seconds
        for match in matches
        if match.signed_residual_seconds is not None
    )
    excessive_indices = tuple(
        index
        for index, match in enumerate(matches)
        if match.absolute_residual_seconds is not None and match.absolute_residual_seconds > maximum
    )
    rejected_indices = tuple(
        index for index, match in enumerate(matches) if match.status == MatchStatus.GAP_EXCEEDED
    )
    missing_indices = tuple(
        index for index, match in enumerate(matches) if match.signed_residual_seconds is None
    )
    if excessive_indices or rejected_indices:
        status = QCStatus.FAIL
        message = "Nearest candidates exceed a residual threshold or their original match gap"
    elif not residuals or missing_indices:
        status = QCStatus.INSUFFICIENT_DATA
        message = "Residuals are unavailable for one or more requested matches"
    else:
        status = QCStatus.PASS
        message = "Every nearest-sample residual meets the configured limits"
    return QCResult(
        check="nearest_sample_residuals",
        status=status,
        metrics={
            "match_count": len(matches),
            "evaluated_count": len(residuals),
            "signed_residuals_seconds": residuals,
            "max_absolute_residual_seconds": max(map(abs, residuals), default=None),
            "excessive_match_indices": excessive_indices,
            "gap_rejected_match_indices": rejected_indices,
            "missing_match_indices": missing_indices,
            "clock_domains": tuple(match.clock_domain.name for match in matches),
            "missing_sample_counts": tuple(match.missing_sample_count for match in matches),
            "duplicate_sample_counts": tuple(match.duplicate_sample_count for match in matches),
        },
        thresholds={"max_residual_seconds": maximum},
        message=message,
        provenance=tuple(item for match in matches for item in match.provenance),
    )


def check_synchronization_gaps(
    timestamps: Sequence[Timestamp | None],
    *,
    clock_domain: ClockDomain,
    max_gap_seconds: Fraction | float | Decimal,
    mappings: Sequence[VerifiedClockMapping] = (),
) -> QCResult:
    """Flag excessive consecutive gaps after sorting; never interpolate missing data.

    Duplicate timestamps produce zero-duration gaps and are reported explicitly.
    Fewer than two distinct times is insufficient evidence of temporal coverage.
    Missing sample values also prevent PASS unless a known excessive gap fails QC.
    """
    maximum = nonnegative_seconds(max_gap_seconds, "max_gap_seconds")
    timeline = order_timestamps(timestamps, clock_domain=clock_domain, mappings=mappings)
    pairs = tuple(zip(timeline.samples, timeline.samples[1:]))
    gaps = tuple(second.seconds - first.seconds for first, second in pairs)
    excessive_pairs = tuple(
        (first.index, second.index)
        for first, second in pairs
        if second.seconds - first.seconds > maximum
    )
    unique_count = len(timeline.samples) - timeline.duplicate_count
    if excessive_pairs:
        status = QCStatus.FAIL
        message = "Consecutive sample timestamps contain an excessive gap"
    elif unique_count < 2 or timeline.missing_indices:
        status = QCStatus.INSUFFICIENT_DATA
        message = "At least two distinct times and no missing values are needed for complete gap QC"
    else:
        status = QCStatus.PASS
        message = "Every consecutive timestamp gap meets the configured maximum"
    return QCResult(
        check="synchronization_gaps",
        status=status,
        metrics={
            "clock_domain": clock_domain.name,
            "sample_count": len(timeline.samples),
            "missing_indices": timeline.missing_indices,
            "duplicate_count": timeline.duplicate_count,
            "consecutive_gaps_seconds": gaps,
            "max_gap_seconds": max(gaps, default=None),
            "excessive_source_index_pairs": excessive_pairs,
        },
        thresholds={"max_gap_seconds": maximum},
        message=message,
        provenance=timeline.provenance,
    )
