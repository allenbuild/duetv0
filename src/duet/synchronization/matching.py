"""Nearest timestamp matching with explicit clocks, residuals, and gap limits."""

from bisect import bisect_left
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction

from duet.schemas.common import Provenance
from duet.schemas.time import (
    ClockDomain,
    Timestamp,
    VerifiedClockMapping,
    as_fraction,
    comparable_seconds,
)
from duet.synchronization.timeline import OrderedTimeline, order_timestamps


class MatchStatus(str, Enum):
    MATCHED = "matched"
    GAP_EXCEEDED = "gap_exceeded"
    NO_SAMPLES = "no_samples"
    MISSING_QUERY = "missing_query"


@dataclass(frozen=True, slots=True)
class NearestTimestampMatch:
    """Nearest candidate and signed residual = candidate minus query, in seconds.

    A gap-rejected candidate remains present for QC, but is not a usable match.
    Consumers must check accepted or use matched_index before accessing samples.
    """

    status: MatchStatus
    clock_domain: ClockDomain
    query: Timestamp
    query_seconds: Fraction | None
    candidate_index: int | None
    candidate_timestamp: Timestamp | None
    candidate_seconds: Fraction | None
    signed_residual_seconds: Fraction | None
    absolute_residual_seconds: Fraction | None
    max_gap_seconds: Fraction
    missing_sample_count: int
    duplicate_sample_count: int
    provenance: tuple[Provenance, ...]

    @property
    def accepted(self) -> bool:
        return self.status == MatchStatus.MATCHED

    @property
    def matched_index(self) -> int | None:
        return self.candidate_index if self.accepted else None


def nonnegative_seconds(value: Fraction | float | Decimal, name: str) -> Fraction:
    """Validate a finite, nonnegative threshold and retain exact numeric precision."""
    seconds = as_fraction(value, name)
    if seconds < 0:
        raise ValueError(f"{name} must be nonnegative")
    return seconds


def nearest_in_timeline(
    query: Timestamp,
    timeline: OrderedTimeline,
    *,
    max_gap_seconds: Fraction | float | Decimal,
    mappings: Sequence[VerifiedClockMapping] = (),
) -> NearestTimestampMatch:
    """Match in the timeline clock, accepting a residual equal to the gap limit.

    Equal distances prefer the earlier timestamp. Duplicate candidates at that
    timestamp prefer the first original index. Missing queries/samples are never
    matched. Cross-clock queries require a verified direct mapping even if missing.
    """
    max_gap = nonnegative_seconds(max_gap_seconds, "max_gap_seconds")
    query_seconds = comparable_seconds(query, timeline.clock_domain, mappings)
    provenance = (
        (query.provenance,)
        + timeline.provenance
        + tuple(
            mapping.provenance
            for mapping in mappings
            if mapping.source == query.clock_domain
            and mapping.destination == timeline.clock_domain
            and mapping.source != mapping.destination
        )
    )
    common = {
        "clock_domain": timeline.clock_domain,
        "query": query,
        "query_seconds": query_seconds,
        "max_gap_seconds": max_gap,
        "missing_sample_count": len(timeline.missing_indices),
        "duplicate_sample_count": timeline.duplicate_count,
        "provenance": provenance,
    }
    if query_seconds is None or not timeline.samples:
        return NearestTimestampMatch(
            status=MatchStatus.MISSING_QUERY if query_seconds is None else MatchStatus.NO_SAMPLES,
            candidate_index=None,
            candidate_timestamp=None,
            candidate_seconds=None,
            signed_residual_seconds=None,
            absolute_residual_seconds=None,
            **common,
        )
    seconds = timeline.seconds
    right = bisect_left(seconds, query_seconds)
    candidates = []
    if right < len(seconds):
        candidates.append(timeline.samples[right])
    if right > 0:
        # The predecessor can be the last duplicate; choose its first occurrence.
        left = bisect_left(seconds, seconds[right - 1])
        candidates.append(timeline.samples[left])
    candidate = min(
        candidates,
        key=lambda sample: (abs(sample.seconds - query_seconds), sample.seconds, sample.index),
    )
    residual = candidate.seconds - query_seconds
    return NearestTimestampMatch(
        status=MatchStatus.MATCHED if abs(residual) <= max_gap else MatchStatus.GAP_EXCEEDED,
        candidate_index=candidate.index,
        candidate_timestamp=candidate.timestamp,
        candidate_seconds=candidate.seconds,
        signed_residual_seconds=residual,
        absolute_residual_seconds=abs(residual),
        **common,
    )


def nearest_timestamp(
    query: Timestamp,
    timestamps: Sequence[Timestamp | None],
    *,
    max_gap_seconds: Fraction | float | Decimal,
    clock_domain: ClockDomain | None = None,
    mappings: Sequence[VerifiedClockMapping] = (),
) -> NearestTimestampMatch:
    """Order a stream then find its nearest sample; default target clock is query's.

    Use order_timestamps and nearest_in_timeline to reuse one sorted timeline for
    many queries. Sequence indices identify source samples, never temporal alignment.
    """
    target_clock = query.clock_domain if clock_domain is None else clock_domain
    timeline = order_timestamps(timestamps, clock_domain=target_clock, mappings=mappings)
    return nearest_in_timeline(query, timeline, max_gap_seconds=max_gap_seconds, mappings=mappings)
