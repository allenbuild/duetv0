"""Multi-SLAM storage and verified shared-world membership, without clock alignment."""

import csv
import io
import json
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, TextIO
from uuid import UUID

from duet.schemas.common import FrameId, Provenance, require_name

if TYPE_CHECKING:
    from duet.adapters.comind.mps import MpsTrajectorySample

ROLES = ("helper", "leader")
TRAJECTORY_MEMBER = "closed_loop_trajectory.csv"


def _unique_mapping(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate VRS mapping key {key!r}")
        result[key] = value
    return result


def parse_vrs_mapping(path: str | Path, *, recording_id: str) -> dict[str, str]:
    """Read exact verified VRS keys; never derive participant identity from ordering."""
    UUID(recording_id)
    with Path(path).open(encoding="utf-8") as stream:
        values = json.load(stream, object_pairs_hook=_unique_mapping)
    if not isinstance(values, dict):
        raise TypeError("vrs_to_multi_slam.json must contain a mapping")
    mapping = {}
    for role in ROLES:
        key = f"{recording_id}/trimmed_vrs/{role}_trimmed.vrs"
        if key not in values:
            raise ValueError(f"missing verified mapping key {key!r}")
        identifier = values[key]
        if not isinstance(identifier, str) or not identifier.isascii() or not identifier.isdigit():
            raise ValueError("Multi-SLAM output ID must be a numeric directory-name string")
        mapping[role] = identifier
    if len(set(mapping.values())) != len(ROLES):
        raise ValueError("helper and leader must map to distinct Multi-SLAM outputs")
    return mapping


@dataclass(frozen=True)
class SlamSource:
    """One alternative Multi-SLAM solution, stored as a directory or flat ZIP.

    Prefer an extracted directory when both exist. A recovery directory must be
    supplied explicitly; corrupt ZIPs are never silently repaired or skipped.
    """

    directory: Path | None = None
    archive: Path | None = None
    recovery_provenance: Provenance | None = None

    @property
    def description(self) -> str:
        if self.directory is not None and (self.directory / TRAJECTORY_MEMBER).is_file():
            return str(self.directory / TRAJECTORY_MEMBER)
        if self.archive is not None:
            return f"{self.archive}!/{TRAJECTORY_MEMBER}"
        raise FileNotFoundError("no Multi-SLAM trajectory source exists")

    @property
    def provenance(self) -> Provenance:
        return Provenance(
            self.description,
            "Multi-SLAM alternative spatial solution; no cross-device time mapping",
            () if self.recovery_provenance is None else (self.recovery_provenance,),
        )

    @contextmanager
    def open_trajectory(self) -> Iterator[TextIO]:
        """Stream only the requested member; ZIP access writes no extracted files."""
        if self.directory is not None and (self.directory / TRAJECTORY_MEMBER).is_file():
            with (self.directory / TRAJECTORY_MEMBER).open(encoding="utf-8", newline="") as stream:
                yield stream
            return
        if self.archive is None:
            raise FileNotFoundError("Multi-SLAM trajectory directory and ZIP are unavailable")
        with zipfile.ZipFile(self.archive) as archive:
            matches = [entry for entry in archive.infolist() if entry.filename == TRAJECTORY_MEMBER]
            if len(matches) != 1:
                raise ValueError("ZIP must contain exactly one closed_loop_trajectory.csv member")
            with (
                archive.open(matches[0]) as binary,
                io.TextIOWrapper(binary, encoding="utf-8", newline="") as stream,
            ):
                yield stream


def trajectory_graph_uids(source: SlamSource) -> frozenset[str]:
    """Inspect every row, so disconnected graph islands cannot pass by intersection."""
    identifiers = set()
    with source.open_trajectory() as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "graph_uid" not in reader.fieldnames:
            raise ValueError("trajectory lacks graph_uid")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("trajectory CSV header must have unique names")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("trajectory row width does not match CSV header")
            identifier = row["graph_uid"]
            if not isinstance(identifier, str) or not identifier.strip():
                raise ValueError("trajectory contains a missing graph_uid")
            identifiers.add(identifier)
    if not identifiers:
        raise ValueError("cannot verify a shared world from an empty trajectory")
    return frozenset(identifiers)


@dataclass(frozen=True)
class SharedWorldVerification:
    """Evidence that both complete inspected trajectory streams use one world graph.

    This proves common spatial coordinates only. Device clock domains stay separate.
    """

    recording_id: str
    graph_uid: str
    world_frame: FrameId
    sources: Mapping[str, SlamSource]
    provenance: Provenance

    def __post_init__(self) -> None:
        UUID(self.recording_id)
        require_name(self.graph_uid, "shared graph UID")
        if self.world_frame != FrameId(f"comind/{self.recording_id}/multislam/{self.graph_uid}"):
            raise ValueError("shared world frame must identify the verified recording and graph")
        if set(self.sources) != set(ROLES) or not all(
            isinstance(source, SlamSource) for source in self.sources.values()
        ):
            raise ValueError("shared world requires one Multi-SLAM source per participant")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("shared world verification requires provenance")
        object.__setattr__(self, "sources", MappingProxyType(dict(self.sources)))


def verify_shared_world(
    sources: Mapping[str, SlamSource], *, recording_id: str
) -> SharedWorldVerification:
    """Require exactly one identical graph UID in helper and leader trajectories.

    Multiple islands are deliberately rejected until explicit island selection is
    implemented; mere overlap between sets does not prove a shared whole recording.
    """
    UUID(recording_id)
    if set(sources) != set(ROLES):
        raise ValueError("shared-world verification requires helper and leader sources")
    graphs = {role: trajectory_graph_uids(sources[role]) for role in ROLES}
    return verify_shared_graph_uids(graphs, sources=sources, recording_id=recording_id)


def verify_shared_graph_uids(
    graphs: Mapping[str, frozenset[str]],
    *,
    sources: Mapping[str, SlamSource],
    recording_id: str,
) -> SharedWorldVerification:
    """Verify complete graph sets collected by an earlier full trajectory audit.

    Callers must supply IDs from every row, not representative samples. This API
    lets the validation pipeline reuse its full geometry/statistics traversal.
    """
    UUID(recording_id)
    if set(graphs) != set(ROLES) or set(sources) != set(ROLES):
        raise ValueError("shared-world verification requires helper and leader sources")
    if any(len(values) != 1 for values in graphs.values()) or graphs["helper"] != graphs["leader"]:
        raise ValueError(f"Multi-SLAM graph UIDs do not establish one shared world: {graphs}")
    graph_uid = next(iter(graphs["helper"]))
    if not isinstance(graph_uid, str) or not graph_uid.strip():
        raise ValueError("shared graph UID must be a nonempty string")
    return SharedWorldVerification(
        recording_id,
        graph_uid,
        FrameId(f"comind/{recording_id}/multislam/{graph_uid}"),
        sources,
        Provenance(
            "duet.comind.verify_shared_world",
            f"All trajectory rows in both participant outputs use graph_uid={graph_uid}",
            tuple(sources[role].provenance for role in ROLES),
        ),
    )


def iter_shared_trajectory(
    verification: SharedWorldVerification, *, participant: str
) -> Iterator["MpsTrajectorySample"]:
    """Parse only a verified Multi-SLAM source into its explicit shared world."""
    from duet.adapters.comind.mps import iter_trajectory

    if participant not in ROLES:
        raise ValueError("participant must be helper or leader")
    source = verification.sources[participant]
    with source.open_trajectory() as stream:
        for sample in iter_trajectory(
            stream,
            participant=participant,
            recording_id=verification.recording_id,
            world_frame=verification.world_frame,
            provenance=Provenance(
                source.description, "Verified shared-world trajectory", (verification.provenance,)
            ),
        ):
            if sample.graph_uid != verification.graph_uid:
                raise ValueError("trajectory graph changed since shared-world verification")
            yield sample
