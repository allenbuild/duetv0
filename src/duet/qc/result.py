"""Shared result vocabulary for geometry and synchronization QC."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from duet.schemas.common import Provenance, require_name


class QCStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass(frozen=True)
class QCResult:
    """Metrics retain units in their names; every applied threshold is reported."""

    check: str
    status: QCStatus
    metrics: Mapping[str, object]
    thresholds: Mapping[str, object]
    message: str = ""
    provenance: tuple[Provenance, ...] = ()

    def __post_init__(self) -> None:
        require_name(self.check, "QC check")
        if not isinstance(self.status, QCStatus):
            raise TypeError("QC status must be a QCStatus")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "thresholds", MappingProxyType(dict(self.thresholds)))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        if not all(isinstance(item, Provenance) for item in self.provenance):
            raise TypeError("QC provenance must contain Provenance objects")
