"""Preserved source timestamps and explicit, verified clock conversion.

Unit conversion does not establish synchronization. Fraction arithmetic preserves
integer nanosecond differences at large epochs without converting them to floats.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from fractions import Fraction

from duet.schemas.common import Provenance, require_name

RawTimestamp = int | float | Decimal


def as_fraction(value: float | Decimal | Fraction, name: str) -> Fraction:
    """Convert a finite numeric quantity without rounding large integers."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal, Fraction)):
        raise TypeError(f"{name} must be a finite number")
    try:
        return Fraction(str(value)) if isinstance(value, (float, Decimal)) else Fraction(value)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError(f"{name} must be finite") from exc


class TimeUnit(str, Enum):
    SECONDS = "s"
    MILLISECONDS = "ms"
    MICROSECONDS = "us"
    NANOSECONDS = "ns"

    @property
    def seconds_per_unit(self) -> Fraction:
        return {
            self.SECONDS: Fraction(1),
            self.MILLISECONDS: Fraction(1, 1_000),
            self.MICROSECONDS: Fraction(1, 1_000_000),
            self.NANOSECONDS: Fraction(1, 1_000_000_000),
        }[self]


@dataclass(frozen=True)
class ClockDomain:
    """A named clock and origin; equal units alone do not make clocks comparable."""

    name: str

    def __post_init__(self) -> None:
        require_name(self.name, "clock domain")


@dataclass(frozen=True)
class Timestamp:
    """Original numeric value and unit in an explicitly named source clock.

    None means missing. NaN and infinity are invalid. No implicit ordering is
    provided: callers must choose a clock and supply verified mappings as needed.
    """

    raw_value: RawTimestamp | None
    unit: TimeUnit
    clock_domain: ClockDomain
    provenance: Provenance

    def __post_init__(self) -> None:
        if not isinstance(self.unit, TimeUnit):
            raise TypeError("timestamp unit must be a TimeUnit")
        if not isinstance(self.clock_domain, ClockDomain):
            raise TypeError("timestamp requires an explicit ClockDomain")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("timestamp requires provenance")
        if self.raw_value is not None:
            if not isinstance(self.raw_value, (int, float, Decimal)):
                raise TypeError("raw timestamp must be int, float, Decimal, or None")
            as_fraction(self.raw_value, "raw timestamp")

    @property
    def seconds(self) -> Fraction | None:
        """Unit-normalized time, still in the original clock domain."""
        if self.raw_value is None:
            return None
        return as_fraction(self.raw_value, "raw timestamp") * self.unit.seconds_per_unit


@dataclass(frozen=True)
class VerifiedClockMapping:
    """Verified affine relation: destination_seconds = scale * source_seconds + offset.

    Construction is the caller's assertion that the supplied evidence verifies
    this relation over the data being processed. No fit, offset, inverse, or
    transitive relation is inferred. Original Timestamp objects are never changed.
    """

    source: ClockDomain
    destination: ClockDomain
    scale: Fraction | int | float
    offset_seconds: Fraction | int | float
    provenance: Provenance
    verification: str

    def __post_init__(self) -> None:
        if not isinstance(self.source, ClockDomain) or not isinstance(
            self.destination, ClockDomain
        ):
            raise TypeError("mapping endpoints must be ClockDomain objects")
        if self.source == self.destination:
            raise ValueError("mapping endpoints must be different clock domains")
        scale = as_fraction(self.scale, "clock scale")
        if scale <= 0:
            raise ValueError("clock scale must be positive")
        object.__setattr__(self, "scale", scale)
        object.__setattr__(self, "offset_seconds", as_fraction(self.offset_seconds, "clock offset"))
        require_name(self.verification, "clock mapping verification")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("clock mapping requires provenance")

    def map_seconds(self, timestamp: Timestamp) -> Fraction | None:
        """Return destination-clock seconds only for the declared source clock."""
        if timestamp.clock_domain != self.source:
            raise ValueError("timestamp clock does not match mapping source")
        seconds = timestamp.seconds
        return None if seconds is None else self.scale * seconds + self.offset_seconds


def comparable_seconds(
    timestamp: Timestamp,
    clock_domain: ClockDomain,
    mappings: Sequence[VerifiedClockMapping] = (),
) -> Fraction | None:
    """Normalize into an explicit clock, requiring one unambiguous direct mapping."""
    if not isinstance(clock_domain, ClockDomain):
        raise TypeError("target clock must be a ClockDomain")
    if timestamp.clock_domain == clock_domain:
        return timestamp.seconds
    candidates = [
        mapping
        for mapping in mappings
        if mapping.source == timestamp.clock_domain and mapping.destination == clock_domain
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"clock domains {timestamp.clock_domain.name!r} and {clock_domain.name!r} "
            "require exactly one explicit verified mapping"
        )
    return candidates[0].map_seconds(timestamp)
