"""Read-only Aria RGB metadata and evidence-gated exported-image calibration.

DEVICE_TIME is participant-local. RGB stream indices are zero-based VRS data
record indices, distinct from source frame_number and MP4 synchronized indices.
Native fisheye calibration remains nonlinear; pixel rotation never invents a
pinhole model or changes the declared native camera coordinate frame.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from duet.adapters.comind.frame_map import VideoVrsFrameMap
from duet.adapters.comind.mps import device_clock
from duet.geometry.transforms import RigidTransform
from duet.schemas.common import DistanceUnit, FrameId, ParticipantId, Provenance, require_name
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit

TIMESTAMP_EVIDENCE = (
    "https://github.com/facebookresearch/projectaria_tools/blob/main/core/data_provider/"
    "TimestampIndexMapper.cpp"
)
CALIBRATION_EVIDENCE = (
    "https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/"
    "advanced_code_snippets/image_utilities"
)


def _integers(values: ArrayLike, name: str) -> NDArray[np.int64]:
    array = np.asarray(values)
    if array.ndim == 1 and not array.size:
        return np.frombuffer(b"", dtype=np.int64)
    if array.ndim != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{name} must be a one-dimensional integer array; no float conversion")
    if array.size and (int(array.min()) < 0 or int(array.max()) > np.iinfo(np.int64).max):
        raise ValueError(f"{name} must contain nonnegative signed-64-bit integers")
    return np.frombuffer(array.astype(np.int64).tobytes(), dtype=np.int64)


@dataclass(frozen=True, eq=False)
class VrsRgbMetadata:
    """Exact SDK DEVICE_TIME values and optional independently read frame numbers."""

    path: Path
    recording_id: str
    participant: str
    stream_id: str
    device_timestamps_ns: NDArray[np.int64] = field(repr=False)
    image_size: tuple[int, int]
    nominal_rate_hz: float
    provenance: Provenance
    source_frame_numbers: NDArray[np.int64] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        device_clock(self.recording_id, self.participant)
        require_name(self.stream_id, "RGB stream ID")
        times = _integers(self.device_timestamps_ns, "DEVICE_TIME timestamps")
        if times.size and np.any(times[1:] < times[:-1]):
            raise ValueError("VRS DEVICE_TIME is not monotonic in source RGB index order")
        object.__setattr__(self, "device_timestamps_ns", times)
        if len(self.image_size) != 2 or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.image_size
        ):
            raise ValueError("RGB dimensions must be positive integer width and height")
        object.__setattr__(self, "image_size", tuple(self.image_size))
        if (
            isinstance(self.nominal_rate_hz, bool)
            or not np.isfinite(self.nominal_rate_hz)
            or self.nominal_rate_hz <= 0
        ):
            raise ValueError("nominal frame rate must be finite and positive")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("RGB metadata requires provenance")
        if self.source_frame_numbers is not None:
            numbers = _integers(self.source_frame_numbers, "source frame numbers")
            if numbers.shape != times.shape:
                raise ValueError("source frame-number count differs from RGB count")
            object.__setattr__(self, "source_frame_numbers", numbers)

    @property
    def clock_domain(self) -> ClockDomain:
        return device_clock(self.recording_id, self.participant)

    @property
    def participant_id(self) -> ParticipantId:
        return ParticipantId(f"participant/{self.participant}")

    def timestamp(self, rgb_index: int) -> Timestamp:
        if isinstance(rgb_index, bool) or not isinstance(rgb_index, int):
            raise TypeError("RGB index must be an integer")
        if not 0 <= rgb_index < len(self.device_timestamps_ns):
            raise IndexError("RGB index is out of range")
        return Timestamp(
            int(self.device_timestamps_ns[rgb_index]),
            TimeUnit.NANOSECONDS,
            self.clock_domain,
            Provenance(
                str(self.path), f"camera-rgb data record index {rgb_index}", (self.provenance,)
            ),
        )

    def summary(self) -> dict[str, object]:
        times = self.device_timestamps_ns
        intervals = np.diff(times)
        return {
            "path": str(self.path),
            "recording_id": self.recording_id,
            "participant": self.participant,
            "stream_id": self.stream_id,
            "clock_domain": self.clock_domain.name,
            "timestamp_authority": "Project Aria get_timestamps_ns(camera-rgb, DEVICE_TIME), exact integer metadata",
            "frame_count": len(times),
            "first_device_timestamp_ns": int(times[0]) if len(times) else None,
            "last_device_timestamp_ns": int(times[-1]) if len(times) else None,
            "strictly_increasing": bool(np.all(intervals > 0)),
            "duplicate_timestamp_count": int(np.count_nonzero(intervals == 0)),
            "interval_ns_median_p95_max": np.percentile(intervals, [50, 95, 100]).tolist()
            if len(intervals)
            else None,
            "image_size": self.image_size,
            "nominal_rate_hz": self.nominal_rate_hz,
            "index_semantics": "Zero-based camera-rgb data-record index; not MP4 index or source frame_number",
            "source_frame_number_first_last": (
                [int(self.source_frame_numbers[0]), int(self.source_frame_numbers[-1])]
                if self.source_frame_numbers is not None and len(times)
                else None
            ),
            "source_frame_number_nonunit_steps": (
                int(np.count_nonzero(np.diff(self.source_frame_numbers) != 1))
                if self.source_frame_numbers is not None
                else None
            ),
        }


def extract_rgb_metadata(
    path: str | Path, *, recording_id: str, participant: str, provider: Any | None = None
) -> tuple[VrsRgbMetadata, Any]:
    """Use the SDK metadata-only timestamp scan; return metadata and native calibration.

    get_timestamps_ns(DEVICE_TIME) explicitly disables image-content reads in the
    official SDK. No image access method is called here. A provider may be reused
    by the caller; supplying one also allows dataset-free synthetic tests.
    """
    from projectaria_tools.core import data_provider
    from projectaria_tools.core.sensor_data import TimeDomain

    device_clock(recording_id, participant)
    path = Path(path)
    if provider is None:
        provider = data_provider.create_vrs_data_provider(str(path))
    if provider is None:
        raise ValueError(f"Project Aria could not open VRS: {path}")
    stream_id = provider.get_stream_id_from_label("camera-rgb")
    if stream_id is None:
        raise ValueError("VRS has no explicitly labeled camera-rgb stream")
    count = provider.get_num_data(stream_id)
    values = provider.get_timestamps_ns(stream_id, TimeDomain.DEVICE_TIME)
    if len(values) != count:
        raise ValueError("SDK RGB timestamp count differs from RGB data-record count")
    configuration = provider.get_image_configuration(stream_id)
    device_calibration = provider.get_device_calibration()
    camera = (
        None if device_calibration is None else device_calibration.get_camera_calib("camera-rgb")
    )
    if camera is None:
        raise ValueError("native VRS camera-rgb calibration is missing")
    image_size = (int(configuration.image_width), int(configuration.image_height))
    if tuple(int(value) for value in camera.get_image_size()) != image_size:
        raise ValueError("native RGB configuration and SDK calibration image sizes differ")
    return VrsRgbMetadata(
        path,
        recording_id,
        participant,
        str(stream_id),
        values,
        image_size,
        float(configuration.nominal_rate_hz),
        Provenance(str(path), f"SDK exact RGB DEVICE_TIME metadata; {TIMESTAMP_EVIDENCE}"),
    ), camera


def read_source_frame_numbers(
    path: str | Path,
    *,
    record_offsets: ArrayLike,
    metadata: VrsRgbMetadata,
    declared_data_layout: dict[str, Any],
) -> NDArray[np.int64]:
    """Read only declared fixed metadata fields at previously validated RGB offsets.

    Each exact capture timestamp is checked against the independent official SDK
    value. Compressed records and undeclared/unsupported field layouts fail; no
    source frame number is inferred from a contiguous index sequence.
    """
    offsets = _integers(record_offsets, "RGB record offsets")
    if len(offsets) != len(metadata.device_timestamps_ns) or np.any(offsets[1:] <= offsets[:-1]):
        raise ValueError("RGB offset count/order mismatch")
    fields = {value["name"]: value for value in declared_data_layout["data_layout"]}
    capture, number = fields["capture_timestamp_ns"], fields["frame_number"]
    if capture["type"] != "DataPieceValue<int64_t>" or number["type"] != "DataPieceValue<uint64_t>":
        raise ValueError("unsupported declared RGB metadata field types")
    for definition in (capture, number):
        offset = definition["offset"]
        if isinstance(offset, bool) or not isinstance(offset, int) or not 0 <= offset <= 4096:
            raise ValueError("unsupported fixed metadata field offset")
    size = 32 + max(capture["offset"], number["offset"]) + 8
    kind, instance = map(int, metadata.stream_id.split("-"))
    result = []
    with Path(path).open("rb") as stream:
        for index, offset in enumerate(offsets):
            stream.seek(int(offset))
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("truncated RGB metadata record")
            header = struct.unpack_from("<IIiIdHBBI", data)
            if header[0] < size or header[2:4] != (kind, 2) or header[5:8] != (instance, 3, 0):
                raise ValueError("unsupported or mismatched RGB record header")
            timestamp = struct.unpack_from("<q", data, 32 + capture["offset"])[0]
            if timestamp != int(metadata.device_timestamps_ns[index]):
                raise ValueError(f"RGB metadata differs from SDK DEVICE_TIME at index {index}")
            result.append(struct.unpack_from("<Q", data, 32 + number["offset"])[0])
    return _integers(result, "source frame numbers")


class ImageOrientation(str, Enum):
    NATIVE = "native"
    CLOCKWISE_90 = "clockwise_90"


@dataclass(frozen=True)
class ExportedRgbCalibration:
    """Exact nonlinear projection into a verified native or CW90 exported image.

    3D points/rays always use the *native camera frame*. The camera extrinsic is
    T_device_nativeCamera; only the pixel mapping rotates. This avoids using the
    SDK rotation utility changes both camera axes and projection parameters;
    this equivalent wrapper deliberately retains the native camera axes.
    """

    camera: Any = field(repr=False)
    transform_device_camera: RigidTransform
    orientation: ImageOrientation
    verification: str
    provenance: Provenance

    def __post_init__(self) -> None:
        require_name(self.verification, "export orientation verification")
        if not isinstance(self.orientation, ImageOrientation):
            raise TypeError("export orientation must be explicit")
        if not isinstance(self.provenance, Provenance):
            raise TypeError("calibration requires provenance")
        sdk_matrix = np.asarray(self.camera.get_transform_device_camera().to_matrix())
        if not np.allclose(sdk_matrix, self.transform_device_camera.matrix, atol=1e-8, rtol=0):
            raise ValueError("native calibration and canonical camera extrinsic disagree")

    @property
    def image_size(self) -> tuple[int, int]:
        native = tuple(int(value) for value in self.camera.get_image_size())
        return native if self.orientation is ImageOrientation.NATIVE else native[::-1]

    def project(self, point_native_camera: ArrayLike) -> NDArray[np.float64] | None:
        """Project a native-camera 3D point into exported pixel-center coordinates."""
        point = np.asarray(point_native_camera)
        if np.iscomplexobj(point) or point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("camera point must contain three finite real values")
        pixel = self.camera.project(np.asarray(point, dtype=float))
        if pixel is None:
            return None
        pixel = np.asarray(pixel, dtype=float)
        if self.orientation is ImageOrientation.CLOCKWISE_90:
            height = int(self.camera.get_image_size()[1])
            pixel = np.array([height - 1 - pixel[1], pixel[0]])
        return pixel

    def unproject(self, pixel_export: ArrayLike) -> NDArray[np.float64] | None:
        """Unproject an exported pixel to a unit ray in the native camera frame."""
        pixel = np.asarray(pixel_export)
        if np.iscomplexobj(pixel) or pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("pixel must contain two finite real values")
        pixel = np.asarray(pixel, dtype=float)
        width, height = self.image_size
        if not (0 <= pixel[0] <= width - 1 and 0 <= pixel[1] <= height - 1):
            return None
        if self.orientation is ImageOrientation.CLOCKWISE_90:
            native_height = int(self.camera.get_image_size()[1])
            pixel = np.array([pixel[1], native_height - 1 - pixel[0]])
        ray = self.camera.unproject(pixel)
        if ray is None:
            return None
        ray = np.asarray(ray, dtype=float)
        norm = np.linalg.norm(ray)
        if not np.isfinite(ray).all() or norm == 0 or not np.isfinite(norm):
            raise ValueError("calibration produced an invalid camera ray")
        return ray / norm


def bind_exported_calibration(
    camera: Any,
    *,
    participant: str,
    orientation: ImageOrientation,
    image_size: tuple[int, int],
    verification: str,
    provenance: Provenance,
) -> ExportedRgbCalibration:
    """Bind a native SDK camera to an explicitly verified pixel rotation and size."""
    if participant not in ("helper", "leader"):
        raise ValueError("participant must be helper or leader")
    transform = RigidTransform(
        camera.get_transform_device_camera().to_matrix(),
        FrameId(f"{participant}/camera-rgb"),
        FrameId(f"{participant}/device"),
        DistanceUnit.METERS,
        provenance,
    )
    result = ExportedRgbCalibration(camera, transform, orientation, verification, provenance)
    if result.image_size != tuple(image_size):
        raise ValueError(
            "exported resolution does not match the verified rotation; crop/resize unresolved"
        )
    return result


def native_calibration_summary(camera: Any, *, participant: str, provenance: Provenance) -> dict:
    """Serialize native model/extrinsic values without implying an MP4 binding."""
    bound = bind_exported_calibration(
        camera,
        participant=participant,
        orientation=ImageOrientation.NATIVE,
        image_size=tuple(int(value) for value in camera.get_image_size()),
        verification="SDK camera-rgb calibration describes native VRS pixels",
        provenance=provenance,
    )
    return {
        "label": camera.get_label(),
        "model": str(camera.get_model_name()),
        "projection_params": np.asarray(camera.get_projection_params()).tolist(),
        "image_size": bound.image_size,
        "T_device_native_camera": bound.transform_device_camera.matrix.tolist(),
        "source_frame": bound.transform_device_camera.source.name,
        "destination_frame": bound.transform_device_camera.destination.name,
        "distance_unit": "m",
        "exported_mp4_orientation_verified": False,
        "native_camera_projection_unchanged": True,
        "valid_radius": camera.get_valid_radius(),
        "max_solid_angle": camera.get_max_solid_angle(),
        "serial_number": camera.get_serial_number(),
        "time_offset_seconds_device_camera": camera.get_time_offset_sec_device_camera(),
        "readout_time_seconds": camera.get_readout_time_sec(),
        "rotation_utility": "SDK rotate_camera_calib_cw90deg supports installed Fisheye624 despite its stale Linear-only pybind docstring; numerical equivalence checked separately",
        "evidence": CALIBRATION_EVIDENCE,
    }


def camera_from_summary(summary: dict[str, Any]) -> Any:
    """Reconstruct the same SDK native model from a compact authoritative cache."""
    from projectaria_tools.core import calibration, sophus

    transform = RigidTransform(
        summary["T_device_native_camera"],
        FrameId(summary["source_frame"]),
        FrameId(summary["destination_frame"]),
        DistanceUnit.METERS,
        Provenance("cached native SDK camera calibration"),
    )
    model = summary["model"].removeprefix("CameraModelType.")
    if np.iscomplexobj(summary["projection_params"]):
        raise ValueError("cached projection parameters must be real")
    parameters = np.asarray(summary["projection_params"], dtype=float)
    if not np.isfinite(parameters).all():
        raise ValueError("cached projection parameters must be finite")
    width, height = summary["image_size"]
    result = calibration.CameraCalibration(
        summary["label"],
        calibration.CameraModelType.__members__[model],
        parameters,
        sophus.SE3.from_matrix(transform.matrix),
        width,
        height,
        summary["valid_radius"],
        summary["max_solid_angle"],
        summary["serial_number"],
        summary["time_offset_seconds_device_camera"],
        summary["readout_time_seconds"],
    )
    focal_lengths = np.asarray(result.get_focal_lengths())
    if not np.isfinite(focal_lengths).all() or np.any(focal_lengths <= 0):
        raise ValueError("cached camera focal lengths must be finite and positive")
    return result


def exported_calibration_evidence(
    camera: Any,
    *,
    participant: str,
    orientation: ImageOrientation,
    verification: str,
    provenance: Provenance,
    segments: int = 64,
    boundary_fraction: float = 0.999,
) -> dict[str, Any]:
    """Validate nonlinear export projection and return native-frame outline rays.

    The contour is the SDK's valid native projection domain (image rectangle,
    circular validity mask, and angular FOV), inset by boundary_fraction. Every ray
    must pass SDK visibility and projection/unprojection checks. This function
    requires external visual orientation evidence; a numerical check alone does
    not establish how an MP4 was exported.
    """
    from projectaria_tools.core import calibration

    if isinstance(segments, bool) or not isinstance(segments, int) or segments < 4:
        raise ValueError("outline needs at least four segments")
    if (
        isinstance(boundary_fraction, bool)
        or not np.isfinite(boundary_fraction)
        or not 0 < boundary_fraction < 1
    ):
        raise ValueError("boundary_fraction must be finite and strictly between zero and one")
    width, height = (int(value) for value in camera.get_image_size())
    image_size = (width, height) if orientation is ImageOrientation.NATIVE else (height, width)
    bound = bind_exported_calibration(
        camera,
        participant=participant,
        orientation=orientation,
        image_size=image_size,
        verification=verification,
        provenance=provenance,
    )
    angle = np.arange(segments) * (2 * np.pi / segments)
    direction = np.c_[np.cos(angle), np.sin(angle)]
    center = np.asarray(camera.get_principal_point(), dtype=float)

    def to_export(pixel: NDArray) -> NDArray:
        return (
            np.array([height - 1 - pixel[1], pixel[0]])
            if orientation is ImageOrientation.CLOCKWISE_90
            else pixel
        )

    def valid(pixel: NDArray) -> bool:
        ray = bound.unproject(to_export(pixel))
        return ray is not None and bound.project(ray) is not None

    if not valid(center):
        raise ValueError("native principal point is not a valid projection center")
    radius = np.empty(segments)
    for index, vector in enumerate(direction):
        low, high = 0.0, float(np.hypot(width, height))
        for _ in range(40):
            middle = (low + high) / 2
            if valid(center + middle * vector):
                low = middle
            else:
                high = middle
        radius[index] = low
    native_pixels = center + boundary_fraction * radius[:, None] * direction
    export_pixels = native_pixels.copy()
    if orientation is ImageOrientation.CLOCKWISE_90:
        export_pixels = np.c_[height - 1 - native_pixels[:, 1], native_pixels[:, 0]]
    rays = [bound.unproject(pixel) for pixel in export_pixels]
    if any(ray is None for ray in rays):
        raise ValueError("declared calibrated outline contains a nonprojectable pixel")
    rays = np.asarray(rays)
    projected = [bound.project(ray) for ray in rays]
    if any(pixel is None for pixel in projected):
        raise ValueError("outline ray cannot be projected back into native calibration")
    error = np.linalg.norm(np.asarray(projected) - export_pixels, axis=1)
    if not np.isfinite(error).all() or error.max() > 1e-5:
        raise ValueError("nonlinear projection round trip failed")
    utility_error = None
    if orientation is ImageOrientation.CLOCKWISE_90:
        upright = calibration.rotate_camera_calib_cw90deg(camera)
        native_rotation = bound.transform_device_camera.matrix[:3, :3]
        upright_rotation = np.asarray(upright.get_transform_device_camera().to_matrix())[:3, :3]
        upright_rays = rays @ (upright_rotation.T @ native_rotation).T
        utility_pixels = [upright.project(ray) for ray in upright_rays]
        if any(pixel is None for pixel in utility_pixels):
            raise ValueError("official rotated camera rejects an otherwise valid outline ray")
        utility_error = float(
            np.max(np.linalg.norm(np.asarray(utility_pixels) - export_pixels, axis=1))
        )
        if not np.isfinite(utility_error) or utility_error > 1e-5:
            raise ValueError("official camera rotation disagrees with pixel-space rotation")
    return {
        "status": "VERIFIED",
        "participant": participant,
        "camera_frame": bound.transform_device_camera.source.name,
        "device_frame": bound.transform_device_camera.destination.name,
        "T_device_camera": bound.transform_device_camera.matrix.tolist(),
        "resolution": list(bound.image_size),
        "projection_model": str(camera.get_model_name())
        + " with explicit "
        + orientation.value
        + " pixel mapping",
        "pixel_rotation": orientation.value,
        "boundary_rays_camera": rays.tolist(),
        "boundary_pixels_export": export_pixels.tolist(),
        "boundary_definition": "SDK-valid projection domain contour found by 40-step radial bisection from principal point, then explicitly inset; native camera axes retained",
        "boundary_fraction": boundary_fraction,
        "projection_roundtrip_max_error_pixels": float(error.max()),
        "official_rotation_max_error_pixels": utility_error,
        "verification": verification,
        "provenance_source": provenance.source,
        "provenance_detail": provenance.detail,
    }


def verify_clockwise_export_anchors(
    anchors: list[dict[str, Any]],
    *,
    frame_count: int,
    maximum_rmse: float = 1.5,
    minimum_rotation_margin: float = 5.0,
    minimum_anchor_count: int = 20,
    maximum_anchor_gap_frames: int = 900,
) -> dict[str, Any]:
    """Verify consistent CW90 visual orientation across a complete video span.

    This gate concerns the geometric image rotation, independent of whether the
    nearest timestamp/image candidate is unique enough for a temporal frame map.
    Scores must come from comparing all four rotations of the same unresized
    native-frame fingerprints against the exported MP4 image fingerprints.
    """
    for value in (maximum_rmse, minimum_rotation_margin):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
            raise ValueError("orientation score thresholds must be finite and positive")
    for value in (frame_count, minimum_anchor_count, maximum_anchor_gap_frames):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("orientation frame/count thresholds must be positive integers")
    if len(anchors) < minimum_anchor_count:
        raise ValueError("insufficient visual anchors for export orientation")
    indices = [row["mp4_frame_index"] for row in anchors]
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise ValueError("anchor frame indices must be integers")
    if (
        indices[0] != 0
        or indices[-1] != frame_count - 1
        or any(
            not 0 < right - left <= maximum_anchor_gap_frames for left, right in pairwise(indices)
        )
    ):
        raise ValueError("visual anchors do not cover the video with the declared maximum gap")
    errors, margins = [], []
    for row in anchors:
        turns = [entry["clockwise_quarter_turns"] for entry in row["all_rotations"]]
        if any(isinstance(turn, bool) or not isinstance(turn, int) for turn in turns):
            raise ValueError("rotation hypotheses must use integer quarter turns")
        scores = {
            entry["clockwise_quarter_turns"]: entry["best_rmse"] for entry in row["all_rotations"]
        }
        if len(row["all_rotations"]) != 4 or set(scores) != {0, 1, 2, 3}:
            raise ValueError("all four rotation hypotheses must be measured")
        if any(
            isinstance(score, bool) or not np.isfinite(score) or score < 0
            for score in scores.values()
        ):
            raise ValueError("rotation errors must be finite and nonnegative")
        margin = min(scores[index] for index in (0, 2, 3)) - scores[1]
        if (
            isinstance(row["clockwise_quarter_turns"], bool)
            or not isinstance(row["clockwise_quarter_turns"], int)
            or row["clockwise_quarter_turns"] != 1
            or scores[1] > maximum_rmse
            or margin < minimum_rotation_margin
        ):
            raise ValueError("visual evidence does not establish CW90 export orientation")
        errors.append(scores[1])
        margins.append(margin)
    return {
        "status": "VERIFIED",
        "clockwise_quarter_turns": 1,
        "anchor_count": len(anchors),
        "first_last_mp4_frame": [indices[0], indices[-1]],
        "max_cw90_rmse": max(errors),
        "min_other_rotation_rmse_margin": min(margins),
        "thresholds": {
            "maximum_rmse": maximum_rmse,
            "minimum_rotation_margin": minimum_rotation_margin,
            "minimum_anchor_count": minimum_anchor_count,
            "maximum_anchor_gap_frames": maximum_anchor_gap_frames,
        },
        "verification": f"All {len(anchors)} visual anchors across MP4 frames {indices[0]}–{indices[-1]} select native RGB rotated clockwise90; all four rotations tested with explicit residual/margin gates",
    }


def verify_export_pixel_geometry(
    audit: dict[str, Any],
    *,
    recording_id: str,
    participant: str,
    frame_count: int,
    image_size: tuple[int, int],
    frame_map_sha256: str,
) -> dict[str, Any]:
    """Check full-resolution evidence against explicit shift/scale alternatives.

    Raw zero-shift scores must beat all eight integer shifts in [-1,1]^2.
    Symmetrically low-pass filtered images control interpolation's codec-noise
    smoothing effect; in this comparison identity must additionally beat center
    scales 0.999 and 1.001. This bounded test cannot exclude arbitrary tiny warps.
    """
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count < 3:
        raise ValueError("pixel geometry audit requires at least three frames")
    if (
        audit["recording_id"] != recording_id
        or audit["participant"] != participant
        or audit["frame_map_npz_sha256"] != frame_map_sha256
        or audit["pixel_geometry_method_version"] != 2
        or audit["clockwise_quarter_turns"] != 1
    ):
        raise ValueError("pixel geometry audit identity, method, or frame map mismatch")
    samples = audit["samples"]
    frames = [sample["mp4_frame_index"] for sample in samples]
    if frames != [0, (frame_count - 1) // 2, frame_count - 1]:
        raise ValueError("pixel geometry audit must cover first, middle, and final frames")
    shifts = {(1.0, dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    hypotheses = shifts | {(0.999, 0, 0), (1.001, 0, 0)}
    identity = (1.0, 0, 0)
    rows = []
    for sample in samples:
        if any(
            isinstance(sample[name], bool) or not isinstance(sample[name], int) or sample[name] < 0
            for name in ("mp4_frame_index", "vrs_rgb_frame_index", "device_timestamp_ns")
        ):
            raise ValueError("pixel geometry source frame indices/timestamps must be integers")
        if tuple(sample["image_dimensions"]) != image_size:
            raise ValueError("pixel geometry audit resolution mismatch")
        control = sample["symmetric_lowpass_control"]
        if control["symmetric_binomial_prefilter_passes"] != 3:
            raise ValueError("pixel geometry scale comparison requires symmetric low-pass control")
        comparisons = []
        for measurement, required in ((sample, shifts), (control, hypotheses)):
            candidates = measurement["ranked_candidates"]
            if any(
                isinstance(row[name], bool) or not isinstance(row[name], int)
                for row in candidates
                for name in ("dx", "dy")
            ):
                raise ValueError("pixel geometry shifts must be integers")
            scores = {(row["scale"], row["dx"], row["dy"]): row["rmse"] for row in candidates}
            if len(scores) != len(candidates) or set(scores) != hypotheses:
                raise ValueError("pixel geometry hypotheses must be complete and unique")
            if any(isinstance(v, bool) or not np.isfinite(v) or v < 0 for v in scores.values()):
                raise ValueError("pixel geometry residuals must be finite and nonnegative")
            runner_up = min(scores[key] for key in required - {identity})
            if scores[identity] >= runner_up:
                raise ValueError("pixel geometry evidence does not favor zero shift and unit scale")
            comparisons.append({"identity_rmse": scores[identity], "runner_up_rmse": runner_up})
        rows.append(
            {
                "mp4_frame_index": sample["mp4_frame_index"],
                "vrs_rgb_frame_index": sample["vrs_rgb_frame_index"],
                "device_timestamp_ns": sample["device_timestamp_ns"],
                "raw_integer_shift_comparison": comparisons[0],
                "symmetric_lowpass_shift_scale_comparison": comparisons[1],
            }
        )
    return {
        "status": "VERIFIED",
        "samples": rows,
        "frame_map_sha256": frame_map_sha256,
        "scope": "Pure CW90 wins against all integer shifts ±1px and center scales 0.999/1.001 at first/middle/final frames; symmetric low-pass comparison controls interpolation/codec smoothing. Arbitrarily small warps are not ruled out.",
    }


def verify_cached_orientation_audit(
    audit: dict[str, Any],
    frame_map: VideoVrsFrameMap,
    *,
    frame_map_sha256: str,
    frame_map_metadata_sha256: str,
) -> dict[str, Any]:
    """Check a new cached-image orientation audit without promoting temporal rows.

    Every originally VERIFIED map pair must appear with its exact source index
    and integer DEVICE_TIME. Optional orientation-only samples explicitly carry
    no timestamp assignment; they can establish image rotation at otherwise
    unresolved frames. The combined samples must pass the unchanged whole-video
    orientation gate, including its maximum 900-frame coverage gap. This function
    does not create, modify, or return any temporal frame correspondence.
    """
    if not isinstance(frame_map, VideoVrsFrameMap):
        raise TypeError("orientation evidence requires a validated VideoVrsFrameMap")
    if (
        audit["recording_id"] != frame_map.recording_id
        or audit["participant"] != frame_map.participant
        or audit["frame_map_npz_sha256"] != frame_map_sha256
        or audit["frame_map_metadata_sha256"] != frame_map_metadata_sha256
    ):
        raise ValueError("orientation audit recording, participant, or frame-map hash mismatch")
    samples = audit["samples"]
    indices = []
    for row in samples:
        for name in ("mp4_frame_index", "vrs_rgb_frame_index", "device_timestamp_ns"):
            if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
                raise ValueError("orientation source indices and timestamps must be exact integers")
        index = row["mp4_frame_index"]
        if not 0 <= index < frame_map.frame_count:
            raise ValueError("orientation audit frame index is outside the source video")
        if (
            frame_map.status[index] != "VERIFIED"
            or frame_map.vrs_rgb_frame_index[index] != row["vrs_rgb_frame_index"]
            or frame_map.vrs_device_timestamp_ns[index] != row["device_timestamp_ns"]
        ):
            raise ValueError("orientation audit source does not match the exact VERIFIED map pair")
        indices.append(index)
    if not np.array_equal(indices, np.flatnonzero(frame_map.status == "VERIFIED")):
        raise ValueError("orientation audit must cover every VERIFIED map pair in source order")
    additional = audit.get("orientation_only_samples", [])
    for row in additional:
        if (
            any(name in row for name in ("device_timestamp_ns", "vrs_device_timestamp_ns"))
            or row.get("temporal_assignment") is not False
        ):
            raise ValueError("orientation-only evidence must not assign DEVICE_TIME")
        for name in ("mp4_frame_index", "orientation_candidate_vrs_rgb_frame_index"):
            if isinstance(row[name], bool) or not isinstance(row[name], int) or row[name] < 0:
                raise ValueError("orientation-only source indices must be exact integers")
        index = row["mp4_frame_index"]
        if not 0 <= index < frame_map.frame_count or index in indices:
            raise ValueError(
                "orientation-only frame is outside the video or duplicates a verified pair"
            )
    combined = sorted([*samples, *additional], key=lambda row: row["mp4_frame_index"])
    evidence = verify_clockwise_export_anchors(combined, frame_count=frame_map.frame_count)
    return evidence | {
        "evidence_basis": "Original VERIFIED-map image pairs plus explicitly separate orientation-only cached image comparisons; no temporal assignments created",
        "verified_map_pair_count": len(samples),
        "orientation_only_sample_count": len(additional),
        "orientation_only_mp4_frames": [row["mp4_frame_index"] for row in additional],
        "frame_map_sha256": frame_map_sha256,
        "frame_map_metadata_sha256": frame_map_metadata_sha256,
        "temporal_map_modified": False,
    }
