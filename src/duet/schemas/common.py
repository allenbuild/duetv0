"""Explicit identities, units, provenance, and observation quality."""

from dataclasses import dataclass
from enum import Enum
from math import isfinite


def require_name(value: str, field: str) -> None:
    """Reject anonymous identifiers instead of inventing their meaning."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")


@dataclass(frozen=True)
class FrameId:
    """Coordinate-frame identity; the name alone does not establish axes or handedness."""

    name: str

    def __post_init__(self) -> None:
        require_name(self.name, "frame name")


@dataclass(frozen=True)
class ParticipantId:
    """Explicit episode-local participant identity, independent of source labels."""

    name: str

    def __post_init__(self) -> None:
        require_name(self.name, "participant name")


@dataclass(frozen=True)
class Provenance:
    """Source reference and derivation evidence; parents retain the original sources."""

    source: str
    detail: str = ""
    parents: tuple["Provenance", ...] = ()

    def __post_init__(self) -> None:
        require_name(self.source, "provenance source")
        if not isinstance(self.detail, str):
            raise TypeError("provenance detail must be a string")
        object.__setattr__(self, "parents", tuple(self.parents))
        if not all(isinstance(parent, Provenance) for parent in self.parents):
            raise TypeError("provenance parents must be Provenance objects")


class DistanceUnit(str, Enum):
    """Source distance units; canonical spatial samples always use meters."""

    METERS = "m"
    CENTIMETERS = "cm"
    MILLIMETERS = "mm"

    @property
    def factor_to_meters(self) -> float:
        return {self.METERS: 1.0, self.CENTIMETERS: 0.01, self.MILLIMETERS: 0.001}[self]


class SampleState(str, Enum):
    PRESENT = "present"
    MISSING = "missing"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class Confidence:
    """An explicitly normalized score, not an assertion of calibrated probability.

    Source-specific scores must only be normalized after their scale is verified;
    record that conversion in provenance. Omit confidence if its meaning is unknown.
    """

    value: float
    provenance: Provenance

    def __post_init__(self) -> None:
        if isinstance(self.value, bool) or not isfinite(self.value) or not 0 <= self.value <= 1:
            raise ValueError("confidence must be finite and in [0, 1]")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("confidence requires provenance")


@dataclass(frozen=True)
class SampleMetadata:
    """Availability and optional confidence for a sample's payload.

    A missing timestamp is independent of payload availability. Unknown confidence
    is None, never zero. Missing/unknown payloads require an explanatory reason.
    """

    state: SampleState = SampleState.PRESENT
    confidence: Confidence | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.state, SampleState):
            raise TypeError("state must be a SampleState")
        if self.confidence is not None and not isinstance(self.confidence, Confidence):
            raise TypeError("confidence must be a Confidence or None")
        if self.state != SampleState.PRESENT:
            require_name(self.reason, "missing/unknown reason")
        if self.reason is not None:
            require_name(self.reason, "sample reason")
