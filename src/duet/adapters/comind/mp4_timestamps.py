"""Validate Project Aria's embedded per-frame DEVICE_TIME metadata.

The official extractor accepts a bare scalar description as a one-element array.
Extraction success alone therefore does not establish a video-to-device mapping:
every video frame must have an integer device timestamp in a complete ordered
array. Repeated timestamps are preserved because the official exporter can repeat
the preceding valid image. Participant clocks always remain separate.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from numbers import Integral
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.timing import integer_timestamps
from duet.schemas.common import Provenance
from duet.schemas.time import Timestamp, TimeUnit

TIMESTAMP_EVIDENCE = (
    "https://facebookresearch.github.io/projectaria_tools/"
    "docs/data_utilities/advanced_code_snippets/vrs_to_mp4"
)
OFFICIAL_MODULES = (
    "projectaria_tools.tools.vrs_to_mp4.vrs_to_mp4_utils",
    "projectaria_tools.utils.vrs_to_mp4_utils",
)


def _positive_frame_count(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError("frame_count must be a positive integer from video metadata")


@dataclass(frozen=True, eq=False)
class Mp4DeviceTimestamps:
    """Complete frame-index to participant DEVICE_TIME nanoseconds, without PTS.

    This validates a local mapping. It does not establish synchronization between
    two videos; that additionally requires explicit dataset frame-alignment evidence.
    The array is copied into immutable storage so later mutation cannot break it.
    """

    recording_id: str
    participant: str
    device_timestamps_ns: ArrayLike
    frame_count: int
    provenance: Provenance

    def __post_init__(self) -> None:
        device_clock(self.recording_id, self.participant)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("MP4 timestamp mapping requires provenance")
        _positive_frame_count(self.frame_count)
        timestamps = integer_timestamps(self.device_timestamps_ns, "device_timestamps_ns")
        if len(timestamps) != self.frame_count:
            raise ValueError(
                f"embedded timestamp count {len(timestamps)} does not match "
                f"video frame count {self.frame_count}"
            )
        if np.any(timestamps[1:] < timestamps[:-1]):
            raise ValueError("per-frame device timestamps must be nondecreasing")
        object.__setattr__(
            self, "device_timestamps_ns", np.frombuffer(timestamps.tobytes(), dtype=np.int64)
        )

    def timestamp_at(self, frame_index: int) -> Timestamp:
        """Return an exact nanosecond Timestamp in this participant's device clock."""
        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
            raise TypeError("frame_index must be an integer")
        if not 0 <= frame_index < self.frame_count:
            raise IndexError("frame_index is outside the video")
        return Timestamp(
            int(self.device_timestamps_ns[frame_index]),
            TimeUnit.NANOSECONDS,
            device_clock(self.recording_id, self.participant),
            self.provenance,
        )


def _load_official_extractor() -> tuple[Callable[[str], object], str, str]:
    """Load the actual optional Project Aria API; never substitute our own parser."""
    for module_name in OFFICIAL_MODULES:
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            # Only try the historical path when the requested module is absent.
            # An installed module's missing dependency must remain visible.
            if error.name and (
                module_name == error.name or module_name.startswith(error.name + ".")
            ):
                continue
            raise
        return (
            module.get_timestamp_from_mp4,
            f"{module_name}.get_timestamp_from_mp4",
            version("projectaria-tools"),
        )
    raise ModuleNotFoundError(
        "Project Aria MP4 timestamp extractor is unavailable; install the optional "
        "projectaria-tools package and its required import dependencies"
    )


def _video_headers(path: Path) -> tuple[int, str | None]:
    """Inspect container headers with PyAV; never decode video frames."""
    import av

    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError("expected exactly one video stream")
        return container.streams.video[0].frames, container.metadata.get("description")


def _source_provenance(path: Path, function_path: str, package_version: str) -> Provenance:
    return Provenance(
        str(path),
        detail=(
            f"Embedded description read by {function_path}, projectaria-tools {package_version}; "
            "DEVICE_TIME nanoseconds; no PTS substitution or timestamp interpolation"
        ),
        parents=(Provenance(TIMESTAMP_EVIDENCE),),
    )


def load_mp4_device_timestamps(
    path: str | Path,
    *,
    recording_id: str,
    participant: str,
    expected_frame_count: int | None = None,
) -> Mp4DeviceTimestamps:
    """Call the official extractor and require one ordered integer value per frame.

    ``projectaria-tools`` and its ``ffprobe`` executable are optional runtime
    requirements for this function. Missing dependencies or malformed metadata
    surface directly. Unit tests can mock the official import without installing it.
    """
    path = Path(path)
    if expected_frame_count is not None:
        _positive_frame_count(expected_frame_count)
    frame_count, _ = _video_headers(path)
    if expected_frame_count is not None and frame_count != expected_frame_count:
        raise ValueError(
            f"video frame count {frame_count} does not match expected {expected_frame_count}"
        )
    extractor, function_path, package_version = _load_official_extractor()
    timestamps = extractor(str(path))
    return Mp4DeviceTimestamps(
        recording_id,
        participant,
        timestamps,
        frame_count,
        _source_provenance(path, function_path, package_version),
    )


def probe_mp4_device_timestamps(
    path: str | Path,
    *,
    recording_id: str,
    participant: str,
    expected_frame_count: int | None = None,
    trajectory_ranges_us: Mapping[str, tuple[int, int]] | None = None,
) -> dict[str, object]:
    """Report exact official outcomes and failed validation without promoting them.

    Interval statistics are undefined for fewer than two timestamps. Coverage
    describes the extracted values, even when incomplete; it is explicitly not
    video coverage. Ranges must come from this participant's existing audit.
    """
    path = Path(path)
    device_clock(recording_id, participant)
    if expected_frame_count is not None:
        _positive_frame_count(expected_frame_count)
    for bounds in (trajectory_ranges_us or {}).values():
        if (
            len(bounds) != 2
            or any(isinstance(value, bool) or not isinstance(value, Integral) for value in bounds)
            or bounds[1] < bounds[0]
        ):
            raise ValueError(
                "trajectory ranges must be inclusive ordered integer microsecond bounds"
            )
    frame_count, description = _video_headers(path)
    result: dict[str, object] = {
        "path": str(path),
        "participant": participant,
        "device_clock": device_clock(recording_id, participant).name,
        "video_frame_count": frame_count,
        "expected_frame_count": expected_frame_count,
        "matches_expected_frame_count": expected_frame_count is None
        or frame_count == expected_frame_count,
        "description_raw": description,
        "description_is_bracketed_array": description is not None
        and description.strip().startswith("[")
        and description.strip().endswith("]"),
        "evidence": TIMESTAMP_EVIDENCE,
        "official_call_succeeded": False,
        "per_frame_device_mapping_verified": False,
    }
    try:
        extractor, function_path, package_version = _load_official_extractor()
        result.update(official_function=function_path, projectaria_tools_version=package_version)
        raw = extractor(str(path))
    except (ImportError, OSError, ValueError, KeyError, AttributeError, TypeError) as error:
        result.update(error_type=type(error).__name__, error=str(error))
        return result
    result.update(
        official_call_succeeded=True,
        returned_type=f"{type(raw).__module__}.{type(raw).__qualname__}",
        returned_shape=list(raw.shape) if isinstance(raw, np.ndarray) else None,
        returned_dtype=str(raw.dtype) if isinstance(raw, np.ndarray) else None,
    )
    try:
        timestamps = integer_timestamps(raw, "official returned timestamps")
    except (TypeError, ValueError) as error:
        result.update(validation_error=str(error))
        return result
    count = len(timestamps)
    intervals = np.diff(timestamps)
    result.update(
        extracted_timestamp_count=count,
        count_equals_video_frame_count=count == frame_count,
        first_timestamp_ns=int(timestamps[0]) if count else None,
        last_timestamp_ns=int(timestamps[-1]) if count else None,
        returned_values_ns=timestamps.tolist() if count <= 16 else None,
        nondecreasing=bool(np.all(intervals >= 0)) if count > 1 else None,
        strictly_increasing=bool(np.all(intervals > 0)) if count > 1 else None,
        duplicate_timestamp_count=count - len(np.unique(timestamps)),
        interval_count=len(intervals),
        timestamp_interval_ns={
            key: float(np.percentile(intervals, percentile)) if len(intervals) else None
            for key, percentile in (("median", 50), ("p95", 95), ("max", 100))
        },
    )
    coverage = {}
    for label, bounds in (trajectory_ranges_us or {}).items():
        start_ns, end_ns = int(bounds[0]) * 1000, int(bounds[1]) * 1000
        inside = (timestamps >= start_ns) & (timestamps <= end_ns)
        coverage[label] = {
            "device_range_us": [int(value) for value in bounds],
            "extracted_values_inside_range": int(inside.sum()),
            "all_extracted_values_inside_range": bool(inside.all()) if count else None,
            "first_minus_range_start_ns": int(timestamps[0]) - start_ns if count else None,
            "not_video_coverage": True,
        }
    result["extracted_value_range_checks"] = coverage
    try:
        if expected_frame_count is not None and frame_count != expected_frame_count:
            raise ValueError("actual video frame count differs from expected frame count")
        Mp4DeviceTimestamps(
            recording_id,
            participant,
            timestamps,
            frame_count,
            _source_provenance(path, function_path, package_version),
        )
    except (TypeError, ValueError) as error:
        result["validation_error"] = str(error)
    else:
        result["per_frame_device_mapping_verified"] = True
    return result
