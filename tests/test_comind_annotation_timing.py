"""Frame/time arithmetic can favor an origin without proving a video mapping."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction

import pytest

from duet.adapters.comind.annotation_timing import audit_annotation_frame_times
from duet.adapters.comind.annotations import parse_handover_segments
from duet.schemas.common import Provenance
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit

RECORDING = "00000000-0000-0000-0000-000000000001"
SOURCE = Provenance("synthetic annotation arithmetic")


def annotation(**updates):
    fields = {"skip": False, "start_frame": 0, "end_frame": 30, "start_time": 0, "end_time": 1}
    fields.update(updates)
    return parse_handover_segments([fields], recording_uuid=RECORDING, provenance=SOURCE)[0]


def audit(items, **updates):
    arguments = {"frame_rate": 30, "tolerance_seconds": Decimal("1e-12")}
    arguments.update(updates)
    return audit_annotation_frame_times(items, **arguments)


def test_zero_based_formula_is_supported_without_claiming_a_video_binding():
    result = audit([annotation()], video_frame_count=31)
    assert result.consistent_numerical_origins == (0,)
    report = result.to_dict()
    assert not report["actual_video_frame_origin_verified"]
    assert not report["annotation_to_video_asset_binding_verified"]
    assert report["hypotheses"]["0"]["combined"]["absolute_max_seconds"] == 0
    assert result.hypotheses[1].residuals[0].residual_seconds == -Fraction(1, 30)


def test_one_based_formula_is_distinguishable_from_zero_based():
    result = audit([annotation(start_frame=1, end_frame=31)], video_frame_count=31)
    assert result.consistent_numerical_origins == (1,)
    assert result.hypotheses[0].residuals[0].residual_seconds == Fraction(1, 30)


def test_decimal_rounding_and_explicit_tolerance():
    item = annotation(
        start_frame=1,
        end_frame=2,
        start_time=Decimal("0.03333333333333333"),
        end_time=Decimal("0.06666666666666667"),
    )
    result = audit([item])
    assert result.consistent_numerical_origins == (0,)
    assert result.hypotheses[0].residuals[0].residual_seconds == Fraction(1, 300000000000000000)
    assert audit([item], tolerance_seconds=0).consistent_numerical_origins == ()


def test_start_end_quantiles_and_sign_use_exact_linear_interpolation():
    # Predicted minus source: [0, 1, 2, 3] seconds; this is deliberately inconsistent.
    items = [
        annotation(start_frame=0, end_frame=90, start_time=0, end_time=2),
        annotation(start_frame=120, end_frame=180, start_time=2, end_time=3),
    ]
    report = audit(items).to_dict()["hypotheses"]["0"]
    assert report["combined"]["absolute_median_seconds"] == 1.5
    assert report["combined"]["absolute_p95_seconds"] == 2.85
    assert report["combined"]["absolute_max_seconds"] == 3
    assert report["combined"]["exact_rationals"]["absolute_p95_seconds"] == "57/20"
    assert report["start"]["signed_median_seconds"] == 1
    assert report["end"]["signed_median_seconds"] == 2


def test_missing_and_empty_boundaries_are_not_silent_success():
    result = audit([annotation(end_time=None)])
    assert result.hypotheses[0].missing_boundary_count == 1
    assert result.consistent_numerical_origins == ()
    empty = audit([])
    assert empty.consistent_numerical_origins == ()
    assert empty.to_dict()["hypotheses"]["0"]["combined"]["absolute_max_seconds"] is None


def test_duplicates_are_preserved_and_wide_tolerance_is_ambiguous():
    result = audit([annotation(), annotation()], tolerance_seconds=Fraction(1, 30))
    assert len(result.hypotheses[0].residuals) == 4
    assert result.consistent_numerical_origins == (0, 1)


def test_correct_arithmetic_outside_video_bounds_is_not_consistent():
    result = audit([annotation()], video_frame_count=30)
    assert result.consistent_numerical_origins == ()
    assert result.hypotheses[0].to_dict()["out_of_range_boundary_count"] == 1


def test_time_unit_normalization_is_explicit_and_does_not_change_source():
    item = annotation()
    ending = Timestamp(1000, TimeUnit.MILLISECONDS, item.end_time.clock_domain, SOURCE)
    item = replace(item, end_time=ending)
    assert audit([item]).consistent_numerical_origins == (0,)
    assert item.end_time.raw_value == 1000
    assert item.end_time.unit is TimeUnit.MILLISECONDS


def test_mixed_recordings_or_clocks_are_rejected():
    item = annotation()
    other_recording = replace(item, recording_id="00000000-0000-0000-0000-000000000002")
    other_clock = ClockDomain("different annotation origin")
    moved = replace(
        item,
        start_time=replace(item.start_time, clock_domain=other_clock),
        end_time=replace(item.end_time, clock_domain=other_clock),
    )
    for other in [other_recording, moved]:
        with pytest.raises(ValueError, match="one recording and one annotation clock"):
            audit([item, other])


@pytest.mark.parametrize(
    "updates",
    [
        {"frame_rate": 0},
        {"frame_rate": float("inf")},
        {"frame_rate": True},
        {"tolerance_seconds": -1},
        {"video_frame_count": True},
        {"video_frame_count": 0},
    ],
)
def test_invalid_comparison_parameters_rejected(updates):
    with pytest.raises((TypeError, ValueError)):
        audit([annotation()], **updates)
