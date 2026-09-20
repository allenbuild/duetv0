"""Exact timestamp ordering and inclusive overlap within an explicit clock domain."""

from collections.abc import Sequence
from dataclasses import dataclass, field
from fractions import Fraction

from duet.schemas.common import Provenance
from duet.schemas.time import ClockDomain, Timestamp, VerifiedClockMapping, comparable_seconds


@dataclass(frozen=True, slots=True)
class OrderedTimestamp:
    """A timestamp's original sequence index and seconds in its timeline's clock."""

    index: int
    timestamp: Timestamp
    seconds: Fraction


@dataclass(frozen=True, slots=True)
class OrderedTimeline:
    """Verified stable time order with immutable cached lookup keys.

    Direct construction has the same clock-verification requirements as the
    order_timestamps factory. Stored seconds must exactly match source timestamps
    under the supplied mappings. Missing samples retain original sequence indices.
    """

    clock_domain: ClockDomain
    samples: tuple[OrderedTimestamp, ...]
    missing_indices: tuple[int, ...]
    provenance: tuple[Provenance, ...]
    mappings: tuple[VerifiedClockMapping, ...] = ()
    seconds: tuple[Fraction, ...] = field(init=False)
    _duplicate_count: int = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.clock_domain, ClockDomain):
            raise TypeError("Timeline clock must be an explicit ClockDomain")
        for name in ("samples", "missing_indices", "provenance", "mappings"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not all(isinstance(mapping, VerifiedClockMapping) for mapping in self.mappings):
            raise TypeError("Timeline mappings must be verified clock mappings")
        if not all(isinstance(item, Provenance) for item in self.provenance):
            raise TypeError("Timeline provenance must contain Provenance objects")
        for sample in self.samples:
            if not isinstance(sample, OrderedTimestamp) or not isinstance(
                sample.timestamp, Timestamp
            ):
                raise TypeError("Timeline samples must contain OrderedTimestamp objects")
            if not isinstance(sample.seconds, Fraction):
                raise TypeError("Normalized timeline seconds must be exact Fraction values")
            expected = comparable_seconds(sample.timestamp, self.clock_domain, self.mappings)
            if expected is None:
                raise ValueError("Missing-valued timestamps cannot be present timeline samples")
            if sample.seconds != expected:
                raise ValueError("Timeline seconds do not match the verified timestamp conversion")
        keys = [(sample.seconds, sample.index) for sample in self.samples]
        if keys != sorted(keys):
            raise ValueError("Timeline samples must be ordered by seconds, then original index")
        indices = [sample.index for sample in self.samples] + list(self.missing_indices)
        if any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0 for index in indices
        ) or len(set(indices)) != len(indices):
            raise ValueError("Timeline sample and missing indices must be distinct and nonnegative")
        seconds = tuple(sample.seconds for sample in self.samples)
        object.__setattr__(self, "seconds", seconds)
        object.__setattr__(self, "_duplicate_count", len(seconds) - len(set(seconds)))

    @property
    def duplicate_count(self) -> int:
        """Number of present timestamps beyond the first at each exact time."""
        return self._duplicate_count


@dataclass(frozen=True, slots=True)
class TimestampRange:
    """Inclusive seconds interval of one nonempty stream in the overlap's clock."""

    start_seconds: Fraction
    end_seconds: Fraction
    sample_count: int


@dataclass(frozen=True, slots=True)
class TimestampOverlap:
    """Common inclusive interval, or no interval if streams are empty or disjoint."""

    clock_domain: ClockDomain
    ranges: tuple[TimestampRange | None, ...]
    missing_counts: tuple[int, ...]
    duplicate_counts: tuple[int, ...]
    start_seconds: Fraction | None
    end_seconds: Fraction | None
    provenance: tuple[Provenance, ...]

    @property
    def has_overlap(self) -> bool:
        return self.start_seconds is not None and self.end_seconds is not None

    @property
    def duration_seconds(self) -> Fraction | None:
        """Overlap duration; zero means one common instant, None means no interval."""
        if self.start_seconds is None or self.end_seconds is None:
            return None
        return self.end_seconds - self.start_seconds


def order_timestamps(
    timestamps: Sequence[Timestamp | None],
    *,
    clock_domain: ClockDomain,
    mappings: Sequence[VerifiedClockMapping] = (),
) -> OrderedTimeline:
    """Order present timestamps stably without changing their raw values or clocks.

    Cross-clock entries, including missing-valued Timestamp objects, require a
    verified direct mapping. Bare None entries have no clock and are only counted
    as missing. Equal timestamps retain original input order.
    """
    if not isinstance(clock_domain, ClockDomain):
        raise TypeError("Timeline clock must be an explicit ClockDomain")
    samples: list[OrderedTimestamp] = []
    missing_indices: list[int] = []
    provenance: list[Provenance] = []
    used_mappings: set[tuple[ClockDomain, ClockDomain]] = set()
    for index, timestamp in enumerate(timestamps):
        if timestamp is None:
            missing_indices.append(index)
            continue
        seconds = comparable_seconds(timestamp, clock_domain, mappings)
        provenance.append(timestamp.provenance)
        if timestamp.clock_domain != clock_domain:
            used_mappings.add((timestamp.clock_domain, clock_domain))
        if seconds is None:
            missing_indices.append(index)
        else:
            samples.append(OrderedTimestamp(index, timestamp, seconds))
    provenance.extend(
        mapping.provenance
        for mapping in mappings
        if (mapping.source, mapping.destination) in used_mappings
    )
    samples.sort(key=lambda sample: (sample.seconds, sample.index))
    return OrderedTimeline(
        clock_domain, tuple(samples), tuple(missing_indices), tuple(provenance), tuple(mappings)
    )


def timestamp_overlap(
    streams: Sequence[Sequence[Timestamp | None]],
    *,
    clock_domain: ClockDomain,
    mappings: Sequence[VerifiedClockMapping] = (),
) -> TimestampOverlap:
    """Intersect stream extents, without implying continuous coverage between samples.

    Empty/all-missing streams and zero streams have no overlap interval. Disjoint
    nonempty ranges also have no interval; ranges preserve evidence distinguishing
    these cases. Clock incompatibility raises ValueError via comparable_seconds.
    """
    if not isinstance(clock_domain, ClockDomain):
        raise TypeError("Overlap clock must be an explicit ClockDomain")
    timelines = tuple(
        order_timestamps(stream, clock_domain=clock_domain, mappings=mappings) for stream in streams
    )
    ranges = tuple(
        TimestampRange(
            timeline.samples[0].seconds, timeline.samples[-1].seconds, len(timeline.samples)
        )
        if timeline.samples
        else None
        for timeline in timelines
    )
    present_ranges = [interval for interval in ranges if interval is not None]
    start: Fraction | None = None
    end: Fraction | None = None
    if ranges and len(present_ranges) == len(ranges):
        candidate_start = max(interval.start_seconds for interval in present_ranges)
        candidate_end = min(interval.end_seconds for interval in present_ranges)
        if candidate_start <= candidate_end:
            start, end = candidate_start, candidate_end
    return TimestampOverlap(
        clock_domain=clock_domain,
        ranges=ranges,
        missing_counts=tuple(len(timeline.missing_indices) for timeline in timelines),
        duplicate_counts=tuple(timeline.duplicate_count for timeline in timelines),
        start_seconds=start,
        end_seconds=end,
        provenance=tuple(item for timeline in timelines for item in timeline.provenance),
    )
