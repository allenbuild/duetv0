"""Image-supported MP4-to-VRS correspondences in participant-local device clocks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path

import av
import numpy as np
from numpy.typing import ArrayLike

from duet.adapters.comind.mps import device_clock
from duet.adapters.comind.timing import integer_timestamps
from duet.schemas.common import Provenance
from duet.schemas.time import Timestamp, TimeUnit


def image_fingerprint(rgb: np.ndarray, *, size: int = 32) -> np.ndarray:
    """Area-like FFmpeg grayscale reduction, preserving orientation for explicit testing."""
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.dtype != np.uint8:
        raise ValueError("expected native uint8 RGB image")
    return (
        av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        .reformat(width=size, height=size, format="gray")
        .to_ndarray()
    )


def image_errors(target: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """RMSE in grayscale 0..255 units; no pixel equality or learned model required."""
    if target.dtype != np.uint8 or candidates.dtype != np.uint8:
        raise ValueError("fingerprints must remain uint8")
    if candidates.ndim != 3 or target.shape != candidates.shape[1:]:
        raise ValueError("candidate fingerprints have incompatible dimensions")
    delta = candidates.astype(np.float32) - target.astype(np.float32)
    return np.sqrt(np.mean(delta * delta, axis=(1, 2)))


def pixel_geometry_scores(
    target: np.ndarray, source: np.ndarray, *, border: int = 16, prefilter_passes: int = 0
) -> dict[str, object]:
    """Compare full-resolution gray pixels at unit scale/zero shift and nearby alternatives.

    Scale candidates use centered bilinear sampling, without optimizing a warp or
    changing the accepted calibration. Positive dx/dy sample source pixels to the
    right/below target pixels. Photometric error is diagnostic, not a pose fit.
    """
    if (
        target.shape != source.shape
        or target.ndim != 2
        or target.dtype != np.uint8
        or source.dtype != np.uint8
    ):
        raise ValueError("pixel alignment requires equal-sized uint8 grayscale images")
    height, width = target.shape
    if border < 2 or min(height, width) <= 2 * border + 2:
        raise ValueError("image border leaves no valid comparison region")
    target_float = target.astype(np.float32)
    source_float = source.astype(np.float32)
    if (
        isinstance(prefilter_passes, bool)
        or not isinstance(prefilter_passes, Integral)
        or not 0 <= prefilter_passes <= 8
    ):
        raise ValueError("prefilter_passes must be an integer from0 through8")
    for _ in range(prefilter_passes):
        for image in (target_float, source_float):
            padded = np.pad(image, ((0, 0), (1, 1)), mode="reflect")
            image[:] = (padded[:, :-2] + 2 * padded[:, 1:-1] + padded[:, 2:]) / 4
            padded = np.pad(image, ((1, 1), (0, 0)), mode="reflect")
            image[:] = (padded[:-2] + 2 * padded[1:-1] + padded[2:]) / 4
    reference = target_float[border:-border, border:-border]
    candidates = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            image = source_float[
                border + dy : height - border + dy, border + dx : width - border + dx
            ]
            candidates.append(
                {
                    "scale": 1.0,
                    "dx": dx,
                    "dy": dy,
                    "rmse": float(np.sqrt(np.mean((reference - image) ** 2))),
                }
            )
    for scale in (0.999, 1.001):
        x = (np.arange(border, width - border) - (width - 1) / 2) / scale + (width - 1) / 2
        y = (np.arange(border, height - border) - (height - 1) / 2) / scale + (height - 1) / 2
        x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
        if x0.min() < 0 or y0.min() < 0 or x0.max() + 1 >= width or y0.max() + 1 >= height:
            raise ValueError("comparison border is too small for the scale candidates")
        wx, wy = (x - x0).astype(np.float32)[None, :], (y - y0).astype(np.float32)[:, None]
        image = (
            source_float[y0[:, None], x0[None, :]] * (1 - wx) * (1 - wy)
            + source_float[y0[:, None], x0[None, :] + 1] * wx * (1 - wy)
            + source_float[y0[:, None] + 1, x0[None, :]] * (1 - wx) * wy
            + source_float[y0[:, None] + 1, x0[None, :] + 1] * wx * wy
        )
        candidates.append(
            {
                "scale": scale,
                "dx": 0,
                "dy": 0,
                "rmse": float(np.sqrt(np.mean((reference - image) ** 2))),
            }
        )
    ranked = sorted(candidates, key=lambda row: row["rmse"])
    return {
        "comparison_border_pixels": border,
        "symmetric_binomial_prefilter_passes": prefilter_passes,
        "ranked_candidates": ranked,
        "zero_shift_unit_scale_wins": ranked[0]["scale"] == 1
        and ranked[0]["dx"] == 0
        and ranked[0]["dy"] == 0,
        "scope": "Tests integer translations ±1px and center scales0.999/1.001; does not prove absence of arbitrarily small warps",
    }


def match_fingerprints(
    mp4: np.ndarray,
    vrs: np.ndarray,
    *,
    candidate_centers: ArrayLike,
    vrs_valid: ArrayLike,
    radius: int = 8,
    maximum_rmse: float = 1.5,
    minimum_margin: float = 0.08,
    minimum_ratio: float = 1.15,
) -> dict[str, np.ndarray]:
    """Direct local visual matching; ambiguous/nonmonotonic rows remain missing.

    Candidate centers only bound computation. They never assign a source frame.
    Duplicated MP4 frames may reuse a VRS index; skipped VRS indices are retained.
    Confidence is image separation, not a calibrated probability.
    """
    if mp4.ndim != 3 or vrs.ndim != 3 or mp4.shape[1:] != vrs.shape[1:]:
        raise ValueError("MP4/VRS fingerprint dimensions differ")
    centers = integer_timestamps(candidate_centers, "candidate_centers")
    valid = np.asarray(vrs_valid)
    if len(centers) != len(mp4) or valid.shape != (len(vrs),) or valid.dtype != np.bool_:
        raise ValueError("candidate centers/validity do not match frame counts")
    if isinstance(radius, bool) or not isinstance(radius, Integral) or radius < 1:
        raise ValueError("candidate radius must be a positive integer")
    if not np.isfinite([maximum_rmse, minimum_margin, minimum_ratio]).all():
        raise ValueError("image thresholds must be finite")
    if maximum_rmse <= 0 or minimum_margin < 0 or minimum_ratio < 1:
        raise ValueError("image thresholds are invalid")
    count = len(mp4)
    indices = np.full(count, -1, dtype=np.int64)
    candidates_best = indices.copy()
    best = np.full(count, np.nan)
    second = best.copy()
    confidence = np.zeros(count)
    status = np.full(count, "UNRESOLVED", dtype="U10")
    reason = np.full(count, "no_valid_candidate", dtype="U40")
    for index, center in enumerate(centers):
        candidates = np.arange(
            max(0, int(center) - radius), min(len(vrs), int(center) + radius + 1)
        )
        candidates = candidates[valid[candidates]]
        if len(candidates) < 2:
            continue
        errors = image_errors(mp4[index], vrs[candidates])
        ranked = np.argsort(errors, kind="stable")
        first, runner_up = float(errors[ranked[0]]), float(errors[ranked[1]])
        candidate = int(candidates[ranked[0]])
        candidates_best[index] = candidate
        best[index], second[index] = first, runner_up
        confidence[index] = 0 if runner_up == 0 else max(0, 1 - first / runner_up)
        # A best candidate at a truncated search edge may hide a better adjacent
        # source frame. Source-file endpoints are genuine boundaries, however.
        edge = (candidate == candidates[0] and candidate > 0) or (
            candidate == candidates[-1] and candidate < len(vrs) - 1
        )
        if edge:
            reason[index] = "search_boundary"
        elif first > maximum_rmse:
            reason[index] = "high_image_error"
        elif (
            runner_up == first
            or runner_up - first < minimum_margin
            or runner_up < first * minimum_ratio
        ):
            reason[index] = "ambiguous_image_match"
        else:
            indices[index] = candidate
            status[index] = "VERIFIED"
            reason[index] = "direct_unique_visual_match"
    # Refuse to silently force a monotonically constrained but visually inferior
    # match. Reject conflicts, then rescan until remaining verified rows agree.
    while True:
        accepted = np.flatnonzero(indices >= 0)
        conflicts = np.flatnonzero(np.diff(indices[accepted]) < 0)
        if not len(conflicts):
            break
        rejected = np.unique(np.concatenate([accepted[conflicts], accepted[conflicts + 1]]))
        indices[rejected] = -1
        status[rejected] = "UNRESOLVED"
        reason[rejected] = "nonmonotonic_visual_candidates"
    return {
        "vrs_rgb_frame_index": indices,
        "candidate_vrs_rgb_frame_index": candidates_best,
        "status": status,
        "match_confidence": confidence,
        "best_rmse": best,
        "second_rmse": second,
        "mapping_reason": reason,
    }


@dataclass(frozen=True, eq=False)
class VideoVrsFrameMap:
    """Immutable per-frame source identities; unresolved rows contain no device time."""

    recording_id: str
    participant: str
    mp4_frame_index: ArrayLike
    vrs_rgb_frame_index: ArrayLike
    vrs_device_timestamp_ns: ArrayLike
    status: ArrayLike
    match_confidence: ArrayLike
    mp4_pts: ArrayLike
    mp4_time_base: Fraction
    provenance: Provenance

    def __post_init__(self) -> None:
        device_clock(self.recording_id, self.participant)
        if not isinstance(self.provenance, Provenance):
            raise TypeError("frame mapping requires provenance")
        frames = integer_timestamps(self.mp4_frame_index, "mp4_frame_index")
        if not np.array_equal(frames, np.arange(len(frames))):
            raise ValueError("MP4 frame indices must cover every frame in order")
        arrays = {"mp4_frame_index": frames}
        for name in ("vrs_rgb_frame_index", "vrs_device_timestamp_ns", "mp4_pts"):
            arrays[name] = integer_timestamps(getattr(self, name), name)
        status = np.asarray(self.status)
        confidence = np.asarray(self.match_confidence)
        if confidence.dtype.kind not in "fiu" or np.iscomplexobj(confidence):
            raise ValueError("confidence must contain real numbers")
        confidence = confidence.astype(float)
        if (
            any(len(array) != len(frames) for array in arrays.values())
            or status.shape != frames.shape
            or confidence.shape != frames.shape
        ):
            raise ValueError("all mapping columns must have the full MP4 frame count")
        if not np.isin(status, ["VERIFIED", "INFERRED", "UNRESOLVED"]).all():
            raise ValueError("mapping status must be explicit")
        if not np.isfinite(confidence).all() or np.any((confidence < 0) | (confidence > 1)):
            raise ValueError("match confidence must be finite and in [0,1]")
        assigned = status != "UNRESOLVED"
        indices = arrays["vrs_rgb_frame_index"]
        timestamps = arrays["vrs_device_timestamp_ns"]
        if (
            np.any(indices[assigned] < 0)
            or np.any(timestamps[assigned] < 0)
            or np.any(indices[~assigned] != -1)
            or np.any(timestamps[~assigned] != -1)
        ):
            raise ValueError("unresolved mappings must not assign a VRS frame or timestamp")
        if np.any(np.diff(indices[assigned]) < 0) or np.any(np.diff(timestamps[assigned]) < 0):
            raise ValueError("assigned frame mappings must be monotonic")
        selected = np.flatnonzero(assigned)
        if len(selected) > 1:
            repeated = np.diff(indices[selected]) == 0
            if np.any(np.diff(timestamps[selected])[repeated] != 0):
                raise ValueError("the same VRS frame cannot have different device timestamps")
        if not isinstance(self.mp4_time_base, Fraction) or self.mp4_time_base <= 0:
            raise ValueError("MP4 time_base must be an explicit positive Fraction")
        if np.any(np.diff(arrays["mp4_pts"]) <= 0):
            raise ValueError("MP4 presentation timestamps must be strictly increasing")
        arrays.update(status=status.astype("U10"), match_confidence=confidence)
        for name, array in arrays.items():
            object.__setattr__(
                self, name, np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)
            )

    @property
    def frame_count(self) -> int:
        return len(self.mp4_frame_index)

    def timestamp_at(self, frame_index: int, *, allow_inferred: bool = False) -> Timestamp | None:
        if isinstance(frame_index, bool) or not isinstance(frame_index, Integral):
            raise TypeError("frame index must be an integer")
        if not 0 <= frame_index < self.frame_count:
            raise IndexError("frame index is outside the MP4")
        status = self.status[frame_index]
        if status == "UNRESOLVED" or (status == "INFERRED" and not allow_inferred):
            return None
        return Timestamp(
            int(self.vrs_device_timestamp_ns[frame_index]),
            TimeUnit.NANOSECONDS,
            device_clock(self.recording_id, self.participant),
            self.provenance,
        )


def load_frame_map(path: str | Path) -> VideoVrsFrameMap:
    path = Path(path)
    metadata = json.loads(path.with_suffix(".json").read_text())
    if metadata.get("npz_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise ValueError("frame-map payload does not match its generation metadata checksum")
    with np.load(path, allow_pickle=False) as arrays:
        return VideoVrsFrameMap(
            metadata["recording_id"],
            metadata["participant"],
            *(
                arrays[name]
                for name in (
                    "mp4_frame_index",
                    "vrs_rgb_frame_index",
                    "vrs_device_timestamp_ns",
                    "status",
                    "match_confidence",
                    "mp4_pts",
                )
            ),
            Fraction(
                int(arrays["mp4_time_base_numerator"]), int(arrays["mp4_time_base_denominator"])
            ),
            Provenance(str(path), detail=metadata["mapping_evidence"]),
        )
