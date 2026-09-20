"""Read scan registration candidates and bounded PLY metadata without binding frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from duet.qc.geometry import check_transform
from duet.qc.result import QCResult, QCStatus
from duet.schemas.common import DistanceUnit, FrameId, Provenance

_PLY_SCALAR_TYPES = frozenset(
    {
        "char",
        "uchar",
        "short",
        "ushort",
        "int",
        "uint",
        "float",
        "double",
        "int8",
        "uint8",
        "int16",
        "uint16",
        "int32",
        "uint32",
        "float32",
        "float64",
    }
)


@dataclass(frozen=True)
class PlyProperty:
    name: str
    data_type: str
    list_count_type: str | None = None


@dataclass(frozen=True)
class PlyElement:
    name: str
    count: int
    properties: tuple[PlyProperty, ...]


@dataclass(frozen=True)
class PlyHeader:
    """Only source header fields; no coordinate units or frames are inferred."""

    encoding: str
    version: str
    elements: tuple[PlyElement, ...]
    comments: tuple[str, ...]
    object_info: tuple[str, ...]
    header_bytes: int
    provenance: Provenance

    @property
    def vertex_count(self) -> int | None:
        return next((element.count for element in self.elements if element.name == "vertex"), None)


@dataclass(frozen=True, eq=False)
class ScanRegistration:
    """Numerically valid matrix whose direction, units and graph remain unverified.

    This is deliberately not a RigidTransform: a filename claim is insufficient
    evidence for canonical frame endpoints or meter units. No graph edge is made.
    """

    matrix: NDArray[np.float64] = field(repr=False)
    filename_claim: str
    validation: QCResult
    provenance: Provenance
    source_frame: FrameId | None = None
    destination_frame: FrameId | None = None
    unit: DistanceUnit | None = None
    semantics_verified: bool = False


@dataclass(frozen=True)
class ScanCloudMetadata:
    path: Path
    header: PlyHeader
    frame: FrameId
    unit: DistanceUnit | None = None
    frame_semantics_verified: bool = False


@dataclass(frozen=True)
class ScanMetadata:
    registration: ScanRegistration
    clouds: tuple[ScanCloudMetadata, ...]


@dataclass(frozen=True, eq=False)
class PointCloudSample:
    """Bounded original-coordinate sample, deliberately without an assigned world/unit.

    Source vertex indices remain explicit. Sampling does not establish units,
    graph membership, registration direction, or whether a transform was applied.
    """

    path: Path
    points: NDArray[np.float64] = field(repr=False)
    colors: NDArray[np.uint8] | None = field(repr=False)
    source_indices: NDArray[np.int64] = field(repr=False)
    total_vertices: int
    header: PlyHeader
    provenance: Provenance


_PLY_DTYPES = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
    "int8": "i1",
    "uint8": "u1",
    "int16": "i2",
    "uint16": "u2",
    "int32": "i4",
    "uint32": "u4",
    "float32": "f4",
    "float64": "f8",
}


def _immutable_array(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.tobytes(), dtype=values.dtype).reshape(values.shape)


def _sample_vertex_indices(count: int, maximum: int, block_size: int) -> NDArray[np.int64]:
    """Use separated contiguous blocks to avoid paging through an entire large PLY."""
    if count <= maximum:
        return np.arange(count, dtype=np.int64)
    block_size = min(block_size, maximum)
    block_count = (maximum + block_size - 1) // block_size
    starts = np.linspace(0, count - block_size, block_count, dtype=np.int64)
    indices = np.unique(np.concatenate([np.arange(start, start + block_size) for start in starts]))
    return indices[:maximum]


def sample_ply_vertices(
    path: str | Path, *, max_points: int = 16_384, block_size: int = 256
) -> PointCloudSample:
    """Read a bounded deterministic sample from a binary fixed-stride vertex PLY.

    Only vertex-first, scalar-property layouts are supported. Binary little- and
    big-endian formats are explicit; ASCII or list-valued vertex data is rejected
    rather than guessed. No complete point-cloud payload is materialized. The
    file is opened read-only, and copied sampled arrays are immutable.
    """
    for name, value in (("max_points", max_points), ("block_size", block_size)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    path = Path(path)
    header = read_ply_header(path)
    if header.encoding not in ("binary_little_endian", "binary_big_endian"):
        raise ValueError("bounded point sampling currently requires an explicit binary PLY")
    if not header.elements or header.elements[0].name != "vertex":
        raise ValueError("bounded point sampling requires the vertex element first")
    vertex = header.elements[0]
    if any(prop.list_count_type is not None for prop in vertex.properties):
        raise ValueError("list-valued vertex properties have no fixed stride")
    names = {prop.name for prop in vertex.properties}
    if not {"x", "y", "z"} <= names:
        raise ValueError("point-cloud vertices require explicit x, y and z properties")
    endian = "<" if header.encoding == "binary_little_endian" else ">"
    dtype = np.dtype(
        [(prop.name, endian + _PLY_DTYPES[prop.data_type]) for prop in vertex.properties]
    )
    if path.stat().st_size < header.header_bytes + vertex.count * dtype.itemsize:
        raise ValueError("PLY vertex payload is truncated relative to its declared count")
    indices = _sample_vertex_indices(vertex.count, max_points, block_size)
    points = np.empty((len(indices), 3), dtype=np.float64)
    rgb_names = {"red", "green", "blue"}
    has_colors = rgb_names <= names
    if names & rgb_names and not has_colors:
        raise ValueError("partial RGB properties cannot be interpreted as complete colors")
    if has_colors and any(
        dtype.fields[name][0].kind != "u" or dtype.fields[name][0].itemsize != 1
        for name in rgb_names
    ):
        raise ValueError("RGB sampling requires explicit unsigned-byte channels")
    colors = np.empty((len(indices), 3), dtype=np.uint8) if has_colors else None
    if len(indices):
        # Mapping is read-only; only the selected blocks are touched, not every vertex.
        vertices = np.memmap(
            path, mode="r", offset=header.header_bytes, dtype=dtype, shape=(vertex.count,)
        )
        try:
            selected = vertices[indices]
            for axis_index, axis in enumerate("xyz"):
                points[:, axis_index] = selected[axis]
            if colors is not None:
                for index, name in enumerate(("red", "green", "blue")):
                    colors[:, index] = selected[name]
        finally:
            # Close the mapping before returning copied arrays, including on errors.
            vertices._mmap.close()
    if not np.isfinite(points).all():
        raise ValueError("sampled point coordinates must be finite; no points are silently dropped")
    return PointCloudSample(
        path,
        _immutable_array(points),
        None if colors is None else _immutable_array(colors),
        _immutable_array(indices),
        vertex.count,
        header,
        Provenance(
            str(path),
            f"Read-only deterministic block sample of {len(indices)}/{vertex.count} vertices; units and registration unverified",
        ),
    )


def point_cloud_sample_qc(sample: PointCloudSample) -> QCResult:
    """Report numerical scale in source units, never a verified physical/world scale.

    Bounds and percentiles cover the sampled vertices only and are not bounds of
    the full cloud. A PASS means finite sample coordinates, not valid registration.
    """
    points = sample.points
    metrics: dict[str, object] = {
        "sample_count": len(points),
        "total_vertices": sample.total_vertices,
        "sample_fraction": len(points) / sample.total_vertices if sample.total_vertices else None,
        "distance_unit": None,
        "registration_verified": False,
        "bounds_scope": "sampled vertices only",
    }
    if len(points):
        lower, upper = points.min(axis=0), points.max(axis=0)
        with np.errstate(over="ignore", invalid="ignore"):
            extent = upper - lower
            diagonal = float(np.hypot.reduce(extent))
        if not np.isfinite(extent).all() or not np.isfinite(diagonal):
            metrics["numerical_failure"] = "sample coordinate extent exceeds finite representation"
            return QCResult(
                "scan_sample_geometry",
                QCStatus.FAIL,
                metrics,
                {},
                "Finite source coordinates produced nonfinite extents; no scale claim is made.",
                (sample.provenance,),
            )
        metrics.update(
            {
                "sample_min_xyz_source_units": lower.tolist(),
                "sample_max_xyz_source_units": upper.tolist(),
                "sample_extent_xyz_source_units": extent.tolist(),
                "sample_coordinate_percentiles_1_50_99": np.percentile(
                    points, [1, 50, 99], axis=0
                ).tolist(),
                "sample_extent_diagonal_source_units": diagonal,
            }
        )
    return QCResult(
        "scan_sample_geometry",
        QCStatus.PASS if len(points) else QCStatus.INSUFFICIENT_DATA,
        metrics,
        {},
        "Coordinate magnitudes do not establish meters or shared-world membership.",
        (sample.provenance,),
    )


def read_ply_header(
    path: str | Path,
    *,
    max_header_bytes: int = 1_048_576,
    max_line_bytes: int = 4096,
) -> PlyHeader:
    """Read only a bounded ASCII header, including for binary PLY payloads."""
    for name, limit in (("max_header_bytes", max_header_bytes), ("max_line_bytes", max_line_bytes)):
        if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
            raise ValueError(f"{name} must be a positive integer")
    path = Path(path)
    comments: list[str] = []
    object_info: list[str] = []
    elements: list[tuple[str, int, list[PlyProperty]]] = []
    encoding = version = None
    bytes_read = 0
    with path.open("rb") as stream:
        first = True
        while True:
            raw_line = stream.readline(min(max_line_bytes, max_header_bytes - bytes_read) + 1)
            bytes_read += len(raw_line)
            if not raw_line:
                raise ValueError("PLY header ended before end_header")
            if len(raw_line) > max_line_bytes or bytes_read > max_header_bytes:
                raise ValueError("PLY header exceeds configured byte limits")
            line = raw_line.decode("ascii").strip()
            if first:
                first = False
                if line != "ply":
                    raise ValueError("PLY file must begin with its ply signature")
                continue
            fields = line.split()
            if not fields:
                raise ValueError("unexpected blank PLY header line")
            if fields[0] == "end_header":
                if len(fields) != 1 or encoding is None or not elements:
                    raise ValueError("incomplete or malformed PLY header")
                break
            if fields[0] == "comment":
                comments.append(line[len("comment") :].lstrip())
            elif fields[0] == "obj_info":
                object_info.append(line[len("obj_info") :].lstrip())
            elif fields[0] == "format":
                if encoding is not None or len(fields) != 3:
                    raise ValueError("duplicate or malformed PLY format")
                encoding, version = fields[1:]
                if encoding not in {"ascii", "binary_little_endian", "binary_big_endian"}:
                    raise ValueError("unsupported PLY encoding")
                if version != "1.0":
                    raise ValueError("unsupported PLY version")
            elif fields[0] == "element":
                if len(fields) != 3 or encoding is None:
                    raise ValueError("malformed PLY element")
                count = int(fields[2])
                if count < 0 or any(element[0] == fields[1] for element in elements):
                    raise ValueError("negative count or duplicate PLY element")
                elements.append((fields[1], count, []))
            elif fields[0] == "property":
                if not elements:
                    raise ValueError("PLY property must follow its element")
                if len(fields) == 3 and fields[1] in _PLY_SCALAR_TYPES:
                    prop = PlyProperty(name=fields[2], data_type=fields[1])
                elif (
                    len(fields) == 5
                    and fields[1] == "list"
                    and fields[2] in _PLY_SCALAR_TYPES
                    and fields[3] in _PLY_SCALAR_TYPES
                ):
                    prop = PlyProperty(fields[4], fields[3], fields[2])
                else:
                    raise ValueError("malformed or unsupported PLY property")
                if any(existing.name == prop.name for existing in elements[-1][2]):
                    raise ValueError("duplicate PLY property")
                elements[-1][2].append(prop)
            else:
                raise ValueError(f"unrecognized PLY header field: {fields[0]}")
    return PlyHeader(
        encoding=encoding,
        version=version,
        elements=tuple(
            PlyElement(name, count, tuple(properties)) for name, count, properties in elements
        ),
        comments=tuple(comments),
        object_info=tuple(object_info),
        header_bytes=bytes_read,
        provenance=Provenance(str(path), detail="PLY header only; payload was not loaded"),
    )


def load_scan_transform(path: str | Path, *, tolerance: float = 1e-8) -> ScanRegistration:
    """Parse and validate the 4x4 matrix without inventing registration semantics."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        text = stream.read(65_537)
    if len(text) > 65_536:
        raise ValueError("scan transform text exceeds the 64 KiB safety bound")
    matrix = np.loadtxt(StringIO(text), dtype=np.float64)
    provenance = Provenance(
        str(path), detail="Unbound scan registration candidate; units unresolved"
    )
    validation = check_transform(matrix, tolerance=tolerance, provenance=(provenance,))
    if validation.status is not QCStatus.PASS:
        raise ValueError(f"invalid scan transform: {validation.message}")
    immutable_matrix = np.frombuffer(matrix.tobytes(), dtype=np.float64).reshape(4, 4)
    return ScanRegistration(immutable_matrix, path.stem, validation, provenance)


def inspect_scan(
    transform_path: str | Path,
    ply_paths: tuple[str | Path, ...],
    *,
    tolerance: float = 1e-8,
) -> ScanMetadata:
    """Inspect registration and headers; keep every scan frame disconnected."""
    registration = load_scan_transform(transform_path, tolerance=tolerance)
    clouds = tuple(
        ScanCloudMetadata(
            path=Path(path),
            header=read_ply_header(path),
            frame=FrameId(f"scan/{Path(path).stem}"),
        )
        for path in ply_paths
    )
    return ScanMetadata(registration, clouds)
