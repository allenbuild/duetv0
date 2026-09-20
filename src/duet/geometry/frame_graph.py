"""Unique-path graphs of verified static transforms or one temporal snapshot."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from duet.geometry.transforms import RigidTransform
from duet.schemas.common import FrameId, Provenance


class FrameGraph:
    """A forest of frames with a single unambiguous path between connected frames.

    A graph is valid for static transforms or one caller-selected snapshot only;
    it does not synchronize or interpolate trajectories. All redundant edges and
    cycles are rejected, including consistent cycles, rather than choosing one
    of multiple paths or silently overriding an existing calibration.
    """

    def __init__(self, transforms: Iterable[RigidTransform] = ()) -> None:
        self._edges: dict[FrameId, dict[FrameId, RigidTransform]] = {}
        for transform in transforms:
            self.add_transform(transform)

    @property
    def frames(self) -> tuple[FrameId, ...]:
        """Return known frames in deterministic name order."""
        return tuple(sorted(self._edges, key=lambda frame: frame.name))

    def _path(self, source: FrameId, destination: FrameId) -> list[RigidTransform] | None:
        queue: deque[tuple[FrameId, list[RigidTransform]]] = deque([(source, [])])
        visited = {source}
        while queue:
            current, path = queue.popleft()
            if current == destination:
                return path
            for neighbor, transform in self._edges.get(current, {}).items():
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, [*path, transform]))
        return None

    def add_transform(self, transform: RigidTransform) -> None:
        """Add a verified edge; reject self-edges, duplicate edges and cycles."""
        if not isinstance(transform, RigidTransform):
            raise TypeError("transform must be a RigidTransform")
        if self._path(transform.source, transform.destination) is not None:
            raise ValueError("edge would introduce a cycle or a redundant/ambiguous frame path")
        inverse = transform.inverse()
        self._edges.setdefault(transform.source, {})[transform.destination] = transform
        self._edges.setdefault(transform.destination, {})[transform.source] = inverse

    def get_transform(self, source: FrameId, destination: FrameId) -> RigidTransform:
        """Return ``T_destination_source`` or reject unknown/disconnected frames."""
        if not isinstance(source, FrameId) or not isinstance(destination, FrameId):
            raise TypeError("source and destination must be explicit FrameId values")
        if source not in self._edges or destination not in self._edges:
            raise KeyError("source or destination frame is unknown")
        path = self._path(source, destination)
        if path is None:
            raise ValueError("source and destination frames are disconnected")
        if not path:
            return RigidTransform.identity(
                source,
                provenance=Provenance("duet.geometry.frame_graph", detail="same-frame identity"),
            )
        result = path[0]
        for transform in path[1:]:
            result = transform.compose(result)
        return result
