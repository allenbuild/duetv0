"""Audit annotation frame/time arithmetic without inventing a video binding.

Agreement with ``frame / fps`` identifies a numerical convention in the source
fields. It cannot identify the actual video asset, physical clock, first image,
or whether an interval's final frame is included.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction

from duet.schemas.episode import HandoverAnnotation
from duet.schemas.time import as_fraction


def _quantile(values: Sequence[Fraction], probability: Fraction) -> Fraction | None:
    """Linear interpolation at (N - 1) * p, using exact arithmetic."""
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = position.numerator // position.denominator
    remainder = position - lower
    return (
        ordered[lower]
        if not remainder
        else (ordered[lower] * (1 - remainder) + ordered[lower + 1] * remainder)
    )


def _summary(values: Sequence[Fraction]) -> dict[str, object]:
    absolute = [abs(value) for value in values]
    quantities = {
        "signed_median_seconds": _quantile(values, Fraction(1, 2)),
        "signed_p95_seconds": _quantile(values, Fraction(95, 100)),
        "signed_min_seconds": min(values, default=None),
        "signed_max_seconds": max(values, default=None),
        "absolute_median_seconds": _quantile(absolute, Fraction(1, 2)),
        "absolute_p95_seconds": _quantile(absolute, Fraction(95, 100)),
        "absolute_max_seconds": max(absolute, default=None),
    }
    return {
        "count": len(values),
        **{name: None if value is None else float(value) for name, value in quantities.items()},
        "exact_rationals": {
            name: None if value is None else str(value) for name, value in quantities.items()
        },
    }


@dataclass(frozen=True)
class AnnotationBoundaryResidual:
    annotation_id: str | None
    boundary: str
    source_frame: int
    source_time_seconds: Fraction
    residual_seconds: Fraction
    frame_in_range: bool | None


@dataclass(frozen=True)
class FrameOriginAudit:
    """One explicit arithmetic hypothesis, not a verified annotation mapping."""

    frame_origin: int
    residuals: tuple[AnnotationBoundaryResidual, ...]
    missing_boundary_count: int
    tolerance_seconds: Fraction

    @property
    def consistent(self) -> bool:
        return (
            bool(self.residuals)
            and not self.missing_boundary_count
            and all(
                abs(item.residual_seconds) <= self.tolerance_seconds
                and item.frame_in_range is not False
                for item in self.residuals
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "frame_origin_hypothesis": self.frame_origin,
            "predicted_time_formula": f"(source_frame - {self.frame_origin}) / frame_rate",
            "residual_sign": "predicted_frame_time_minus_source_annotation_time",
            "missing_boundary_count": self.missing_boundary_count,
            "consistent_with_all_supplied_boundaries": self.consistent,
            "tolerance_seconds": float(self.tolerance_seconds),
            "out_of_range_boundary_count": sum(
                item.frame_in_range is False for item in self.residuals
            ),
            "frame_range_checked": any(item.frame_in_range is not None for item in self.residuals),
            "combined": _summary([item.residual_seconds for item in self.residuals]),
            "start": _summary(
                [item.residual_seconds for item in self.residuals if item.boundary == "start"]
            ),
            "end": _summary(
                [item.residual_seconds for item in self.residuals if item.boundary == "end"]
            ),
            "boundaries": [
                {
                    "annotation_id": item.annotation_id,
                    "boundary": item.boundary,
                    "source_frame": item.source_frame,
                    "source_time_seconds_exact": str(item.source_time_seconds),
                    "residual_seconds_exact": str(item.residual_seconds),
                    "residual_seconds": float(item.residual_seconds),
                    "frame_in_range": item.frame_in_range,
                }
                for item in self.residuals
            ],
        }


@dataclass(frozen=True)
class AnnotationTimingAudit:
    annotation_count: int
    frame_rate: Fraction
    hypotheses: tuple[FrameOriginAudit, FrameOriginAudit]

    @property
    def consistent_numerical_origins(self) -> tuple[int, ...]:
        return tuple(item.frame_origin for item in self.hypotheses if item.consistent)

    def to_dict(self) -> dict[str, object]:
        return {
            "annotation_count": self.annotation_count,
            "frame_rate_exact": str(self.frame_rate),
            "consistent_numerical_origins": list(self.consistent_numerical_origins),
            "quantile_method": "linear interpolation at (N-1)*p with exact rational arithmetic",
            "actual_video_frame_origin_verified": False,
            "annotation_to_video_asset_binding_verified": False,
            "scope": "Source-field arithmetic only; no clock or physical-image mapping is created.",
            "hypotheses": {str(item.frame_origin): item.to_dict() for item in self.hypotheses},
        }


def audit_annotation_frame_times(
    annotations: Sequence[HandoverAnnotation],
    *,
    frame_rate: float | Decimal | Fraction,
    tolerance_seconds: float | Decimal | Fraction,
    video_frame_count: int | None = None,
) -> AnnotationTimingAudit:
    """Compare zero- and one-based time formulas for one recording and source clock.

    Only complete frame/time pairs are compared. Missing boundaries are counted
    and prevent full-recording consistency. If supplied, ``video_frame_count``
    tests bounds for each hypothesis; it does not assert the video's identity.
    Pass already-filtered usable annotations, keeping the adapter's skip policy.
    """
    rate = as_fraction(frame_rate, "annotation comparison frame rate")
    tolerance = as_fraction(tolerance_seconds, "annotation comparison tolerance")
    if rate <= 0 or tolerance < 0:
        raise ValueError("frame rate must be positive and tolerance nonnegative")
    if video_frame_count is not None and (
        type(video_frame_count) is not int or video_frame_count <= 0
    ):
        raise ValueError("video_frame_count must be a positive integer or None")
    if not isinstance(annotations, Sequence) or isinstance(annotations, (str, bytes)):
        raise TypeError("annotations must be a sequence")
    if not all(isinstance(item, HandoverAnnotation) for item in annotations):
        raise TypeError("annotations must contain canonical HandoverAnnotation objects")
    recordings = {item.recording_id for item in annotations}
    clocks = {
        timestamp.clock_domain
        for item in annotations
        for timestamp in (item.start_time, item.end_time)
    }
    if len(recordings) > 1 or len(clocks) > 1:
        raise ValueError("one audit requires one recording and one annotation clock domain")
    hypotheses = []
    for origin in (0, 1):
        residuals = []
        missing = 0
        for annotation in annotations:
            for boundary in ("start", "end"):
                frame = getattr(annotation, f"{boundary}_frame")
                seconds = getattr(annotation, f"{boundary}_time").seconds
                if frame is None or seconds is None:
                    missing += 1
                    continue
                residuals.append(
                    AnnotationBoundaryResidual(
                        annotation.annotation_id,
                        boundary,
                        frame,
                        seconds,
                        Fraction(frame - origin) / rate - seconds,
                        None
                        if video_frame_count is None
                        else 0 <= frame - origin < video_frame_count,
                    )
                )
        hypotheses.append(FrameOriginAudit(origin, tuple(residuals), missing, tolerance))
    return AnnotationTimingAudit(len(annotations), rate, (hypotheses[0], hypotheses[1]))
