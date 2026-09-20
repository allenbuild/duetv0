"""Evidence records for dataset mappings; uncertain claims grant no capability.

These records describe reviewable evidence. They are not clock mappings or
geometric transforms and cannot substitute for validated numerical mappings.
"""

from dataclasses import dataclass
from enum import Enum

from duet.schemas.common import require_name


class EvidenceStatus(str, Enum):
    VERIFIED = "verified"
    INFERRED = "inferred"
    UNRESOLVED = "unresolved"


@dataclass(frozen=True)
class MappingEvidence:
    """The scope of a mapping claim and exact sources or missing artifacts.

    ``verified`` refers only to ``statement``. For example, verified native
    camera extrinsics do not establish an exported image's pixel geometry.
    """

    name: str
    status: EvidenceStatus
    statement: str
    evidence: tuple[str, ...] = ()
    missing_assets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        require_name(self.name, "mapping evidence name")
        require_name(self.statement, "mapping evidence statement")
        if not isinstance(self.status, EvidenceStatus):
            raise TypeError("status must be an explicit EvidenceStatus")
        for field in ("evidence", "missing_assets"):
            values = getattr(self, field)
            if not isinstance(values, tuple):
                raise TypeError(f"{field} must be an immutable tuple")
            for value in values:
                require_name(value, field)
        if self.status is EvidenceStatus.VERIFIED:
            if not self.evidence:
                raise ValueError("verified mapping claims require supporting evidence")
            if self.missing_assets:
                raise ValueError("verified mapping claims cannot have missing required assets")

    @property
    def verified(self) -> bool:
        return self.status is EvidenceStatus.VERIFIED

    def require_verified(self) -> None:
        """Reject unresolved and inferred claims before enabling a dependent layer."""
        if not self.verified:
            raise ValueError(f"{self.name}: {self.status.value}: {self.statement}")

    def to_dict(self) -> dict[str, object]:
        """Return ordinary JSON-compatible values without losing evidence status."""
        return {
            "name": self.name,
            "status": self.status.value,
            "statement": self.statement,
            "evidence": list(self.evidence),
            "missing_assets": list(self.missing_assets),
        }
