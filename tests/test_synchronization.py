"""Synthetic, exact-clock tests; no dataset files are needed or inspected."""

from decimal import Decimal
from fractions import Fraction

import pytest

from duet.qc.result import QCStatus
from duet.qc.synchronization import (
    check_nearest_residuals,
    check_synchronization_gaps,
    check_timestamp_overlap,
)
from duet.schemas.common import Provenance
from duet.schemas.time import ClockDomain, Timestamp, TimeUnit, VerifiedClockMapping
from duet.synchronization.matching import (
    MatchStatus,
    nearest_in_timeline,
    nearest_timestamp,
)
from duet.synchronization.timeline import (
    OrderedTimeline,
    OrderedTimestamp,
    order_timestamps,
    timestamp_overlap,
)

CLOCK = ClockDomain("synthetic/shared")
OTHER_CLOCK = ClockDomain("synthetic/device")
SOURCE = Provenance("synthetic fixture")


def stamp(
    value: float | Decimal | None,
    clock: ClockDomain = CLOCK,
    unit: TimeUnit = TimeUnit.SECONDS,
) -> Timestamp:
    return Timestamp(value, unit, clock, SOURCE)


def mapping() -> VerifiedClockMapping:
    return VerifiedClockMapping(
        source=OTHER_CLOCK,
        destination=CLOCK,
        scale=2,
        offset_seconds=10,
        provenance=SOURCE,
        verification="Synthetic clock equation: shared_seconds = 2 * device_seconds + 10",
    )


def test_order_preserves_original_indices_missing_entries_and_duplicate_order() -> None:
    stamps = [stamp(2), None, stamp(1), stamp(None), stamp(1)]
    timeline = order_timestamps(stamps, clock_domain=CLOCK)
    assert [item.index for item in timeline.samples] == [2, 4, 0]
    assert [item.seconds for item in timeline.samples] == [1, 1, 2]
    assert timeline.samples[0].timestamp is stamps[2]
    assert timeline.missing_indices == (1, 3)
    assert timeline.duplicate_count == 1


def test_nearest_uses_time_not_sequence_position() -> None:
    match = nearest_timestamp(stamp(6), [stamp(10), stamp(2), stamp(5)], max_gap_seconds=2)
    assert match.accepted
    assert match.matched_index == 2
    assert match.signed_residual_seconds == -1
    assert match.absolute_residual_seconds == 1


@pytest.mark.parametrize("values,expected", [([7, 3], 1), ([3, 7], 0), ([7, 3, 3], 1)])
def test_ties_prefer_earlier_time_then_first_duplicate(values: list[int], expected: int) -> None:
    match = nearest_timestamp(stamp(5), [stamp(value) for value in values], max_gap_seconds=2)
    assert match.matched_index == expected
    assert match.candidate_seconds == 3


@pytest.mark.parametrize("query,expected", [(3, 0), (4, 0), (0, 0), (8, 2)])
def test_duplicate_timestamps_preserve_first_original_index(query: int, expected: int) -> None:
    match = nearest_timestamp(stamp(query), [stamp(3), stamp(3), stamp(7)], max_gap_seconds=10)
    assert match.matched_index == expected
    assert match.duplicate_sample_count == 1


def test_maximum_gap_rejection_retains_candidate_and_residual() -> None:
    match = nearest_timestamp(stamp(10), [stamp(8)], max_gap_seconds=1)
    assert match.status == MatchStatus.GAP_EXCEEDED
    assert not match.accepted
    assert match.matched_index is None
    assert match.candidate_index == 0
    assert match.signed_residual_seconds == -2
    assert match.absolute_residual_seconds == 2


def test_maximum_gap_is_inclusive_and_zero_allows_exact_match() -> None:
    assert nearest_timestamp(stamp(3), [stamp(2)], max_gap_seconds=1).accepted
    assert nearest_timestamp(stamp(2), [stamp(2)], max_gap_seconds=0).accepted
    assert not nearest_timestamp(stamp(3), [stamp(2)], max_gap_seconds=0).accepted


@pytest.mark.parametrize("threshold", [-1, float("nan"), float("inf"), Decimal("NaN")])
def test_invalid_gap_threshold_is_rejected(threshold: float | Decimal) -> None:
    with pytest.raises(ValueError):
        nearest_timestamp(stamp(1), [], max_gap_seconds=threshold)


@pytest.mark.parametrize("threshold", [True, "1"])
def test_nonnumeric_gap_threshold_is_rejected(threshold: object) -> None:
    with pytest.raises(TypeError):
        nearest_timestamp(stamp(1), [], max_gap_seconds=threshold)


@pytest.mark.parametrize("values", [[], [None], [stamp(None), None]])
def test_empty_and_all_missing_streams_have_no_candidate(values: list[Timestamp | None]) -> None:
    match = nearest_timestamp(stamp(1), values, max_gap_seconds=1)
    assert match.status == MatchStatus.NO_SAMPLES
    assert match.candidate_index is None
    assert match.signed_residual_seconds is None
    assert not match.accepted


def test_missing_query_reports_no_residual() -> None:
    match = nearest_timestamp(stamp(None), [stamp(1)], max_gap_seconds=1)
    assert match.status == MatchStatus.MISSING_QUERY
    assert match.signed_residual_seconds is None


def test_overlap_is_inclusive_and_uses_ordered_extents() -> None:
    overlap = timestamp_overlap(
        [[stamp(5), stamp(0), None], [stamp(6), stamp(2), stamp(2)]], clock_domain=CLOCK
    )
    assert overlap.has_overlap
    assert overlap.start_seconds == 2
    assert overlap.end_seconds == 5
    assert overlap.duration_seconds == 3
    assert overlap.missing_counts == (1, 0)
    assert overlap.duplicate_counts == (0, 1)
    touching = timestamp_overlap([[stamp(0), stamp(2)], [stamp(2)]], clock_domain=CLOCK)
    assert touching.has_overlap
    assert touching.duration_seconds == 0


@pytest.mark.parametrize(
    "streams",
    [[], [[]], [[stamp(1)], []], [[None], [stamp(1)]], [[stamp(0)], [stamp(1)]]],
)
def test_no_overlap_cases(streams: list[list[Timestamp | None]]) -> None:
    overlap = timestamp_overlap(streams, clock_domain=CLOCK)
    assert not overlap.has_overlap
    assert overlap.start_seconds is None
    assert overlap.end_seconds is None
    assert overlap.duration_seconds is None


@pytest.mark.parametrize("value", [1, None])
def test_cross_clock_samples_rejected_even_when_missing(value: int | None) -> None:
    with pytest.raises(ValueError):
        nearest_timestamp(stamp(1), [stamp(value, OTHER_CLOCK)], max_gap_seconds=1)
    with pytest.raises(ValueError):
        timestamp_overlap([[stamp(1)], [stamp(value, OTHER_CLOCK)]], clock_domain=CLOCK)


def test_cross_clock_query_requires_mapping_even_for_empty_timeline() -> None:
    timeline = order_timestamps([], clock_domain=CLOCK)
    with pytest.raises(ValueError):
        nearest_in_timeline(stamp(None, OTHER_CLOCK), timeline, max_gap_seconds=1)


def test_verified_mapping_allows_explicit_clock_comparison_without_mutating_source() -> None:
    source = stamp(2, OTHER_CLOCK)
    match = nearest_timestamp(stamp(14), [source], max_gap_seconds=0, mappings=[mapping()])
    assert match.accepted
    assert match.candidate_seconds == 14
    assert match.candidate_timestamp is source
    assert source.raw_value == 2
    assert source.clock_domain == OTHER_CLOCK
    overlap = timestamp_overlap(
        [[stamp(12), stamp(16)], [stamp(1, OTHER_CLOCK), stamp(4, OTHER_CLOCK)]],
        clock_domain=CLOCK,
        mappings=[mapping()],
    )
    assert overlap.duration_seconds == 4


def test_query_mapping_and_reverse_mapping_are_explicit() -> None:
    match = nearest_timestamp(
        stamp(2, OTHER_CLOCK),
        [stamp(14)],
        clock_domain=CLOCK,
        max_gap_seconds=0,
        mappings=[mapping()],
    )
    assert match.accepted
    with pytest.raises(ValueError):
        nearest_timestamp(
            stamp(2, OTHER_CLOCK), [stamp(14)], max_gap_seconds=0, mappings=[mapping()]
        )


def test_ambiguous_duplicate_clock_mappings_rejected() -> None:
    with pytest.raises(ValueError):
        nearest_timestamp(
            stamp(14), [stamp(2, OTHER_CLOCK)], max_gap_seconds=0, mappings=[mapping(), mapping()]
        )


def test_nanosecond_precision_is_retained_at_large_epochs() -> None:
    epoch = 1_800_000_000_000_000_000
    values = [
        stamp(epoch + 3, unit=TimeUnit.NANOSECONDS),
        stamp(epoch + 1, unit=TimeUnit.NANOSECONDS),
    ]
    query = stamp(epoch + 2, unit=TimeUnit.NANOSECONDS)
    match = nearest_timestamp(query, values, max_gap_seconds=Fraction(1, 1_000_000_000))
    assert match.accepted
    assert match.matched_index == 1
    assert match.signed_residual_seconds == Fraction(-1, 1_000_000_000)
    timeline = order_timestamps(values, clock_domain=CLOCK)
    assert timeline.samples[1].seconds - timeline.samples[0].seconds == Fraction(2, 1_000_000_000)


def test_mixed_units_are_comparable_only_within_same_clock() -> None:
    match = nearest_timestamp(
        stamp(Decimal("1.001")), [stamp(1001, unit=TimeUnit.MILLISECONDS)], max_gap_seconds=0
    )
    assert match.accepted
    assert match.absolute_residual_seconds == 0


def test_qc_overlap_pass_fail_insufficient_and_explicit_threshold() -> None:
    assert (
        check_timestamp_overlap(
            [[stamp(0), stamp(4)], [stamp(2), stamp(6)]],
            clock_domain=CLOCK,
            minimum_overlap_seconds=2,
        ).status
        == QCStatus.PASS
    )
    assert (
        check_timestamp_overlap(
            [[stamp(0)], [stamp(1)]], clock_domain=CLOCK, minimum_overlap_seconds=0
        ).status
        == QCStatus.FAIL
    )
    assert (
        check_timestamp_overlap(
            [[stamp(0), stamp(4)], [stamp(2), stamp(6)]],
            clock_domain=CLOCK,
            minimum_overlap_seconds=3,
        ).status
        == QCStatus.FAIL
    )
    assert (
        check_timestamp_overlap(
            [[], [stamp(1)]], clock_domain=CLOCK, minimum_overlap_seconds=0
        ).status
        == QCStatus.INSUFFICIENT_DATA
    )


def test_qc_residuals_preserve_rejected_candidates_and_missing_evidence() -> None:
    accepted = nearest_timestamp(stamp(1), [stamp(2)], max_gap_seconds=1)
    rejected = nearest_timestamp(stamp(1), [stamp(3)], max_gap_seconds=1)
    missing = nearest_timestamp(stamp(1), [], max_gap_seconds=1)
    assert check_nearest_residuals([accepted], max_residual_seconds=1).status == QCStatus.PASS
    assert check_nearest_residuals([accepted], max_residual_seconds=0).status == QCStatus.FAIL
    result = check_nearest_residuals([accepted, rejected, missing], max_residual_seconds=10)
    assert result.status == QCStatus.FAIL
    assert result.metrics["gap_rejected_match_indices"] == (1,)
    assert result.metrics["missing_match_indices"] == (2,)
    assert result.metrics["max_absolute_residual_seconds"] == 2
    assert check_nearest_residuals([accepted, missing], max_residual_seconds=1).status == (
        QCStatus.INSUFFICIENT_DATA
    )
    assert check_nearest_residuals([], max_residual_seconds=1).status == QCStatus.INSUFFICIENT_DATA


def test_qc_gaps_sort_timestamps_and_report_original_indices() -> None:
    result = check_synchronization_gaps(
        [stamp(10), stamp(0), stamp(2), stamp(2)], clock_domain=CLOCK, max_gap_seconds=3
    )
    assert result.status == QCStatus.FAIL
    assert result.metrics["consecutive_gaps_seconds"] == (2, 0, 8)
    assert result.metrics["excessive_source_index_pairs"] == ((3, 0),)
    assert result.metrics["duplicate_count"] == 1
    assert result.thresholds["max_gap_seconds"] == 3
    assert (
        check_synchronization_gaps(
            [stamp(0), stamp(2)], clock_domain=CLOCK, max_gap_seconds=2
        ).status
        == QCStatus.PASS
    )
    assert (
        check_synchronization_gaps(
            [stamp(2), stamp(2)], clock_domain=CLOCK, max_gap_seconds=2
        ).status
        == QCStatus.INSUFFICIENT_DATA
    )
    assert (
        check_synchronization_gaps(
            [stamp(0), None, stamp(2)], clock_domain=CLOCK, max_gap_seconds=2
        ).status
        == QCStatus.INSUFFICIENT_DATA
    )


def test_qc_rejects_cross_clock_without_mapping() -> None:
    with pytest.raises(ValueError):
        check_synchronization_gaps(
            [stamp(0), stamp(1, OTHER_CLOCK)], clock_domain=CLOCK, max_gap_seconds=1
        )


def test_manual_timeline_cannot_bypass_clock_verification() -> None:
    sample = OrderedTimestamp(0, stamp(2, OTHER_CLOCK), Fraction(14))
    with pytest.raises(ValueError, match="verified mapping"):
        OrderedTimeline(CLOCK, (sample,), (), (SOURCE,))
    timeline = OrderedTimeline(CLOCK, (sample,), (), (SOURCE,), (mapping(),))
    assert nearest_in_timeline(stamp(14), timeline, max_gap_seconds=0).accepted


@pytest.mark.parametrize("source_clock,maps", [(CLOCK, ()), (OTHER_CLOCK, (mapping(),))])
def test_manual_timeline_rejects_fabricated_seconds(
    source_clock: ClockDomain, maps: tuple[VerifiedClockMapping, ...]
) -> None:
    sample = OrderedTimestamp(0, stamp(2, source_clock), Fraction(999))
    with pytest.raises(ValueError, match="verified timestamp conversion"):
        OrderedTimeline(CLOCK, (sample,), (), (SOURCE,), maps)


def test_manual_timeline_rejects_missing_present_sample() -> None:
    sample = OrderedTimestamp(0, stamp(None), Fraction(1))
    with pytest.raises(ValueError, match="Missing-valued"):
        OrderedTimeline(CLOCK, (sample,), (), (SOURCE,))


def test_empty_streams_still_require_explicit_clock_domain() -> None:
    with pytest.raises(TypeError, match="ClockDomain"):
        order_timestamps([], clock_domain="synthetic/shared")
    with pytest.raises(TypeError, match="ClockDomain"):
        timestamp_overlap([], clock_domain="synthetic/shared")


def test_manual_timeline_copies_constructor_sequences_and_caches_exact_keys() -> None:
    sample_list = [OrderedTimestamp(0, stamp(2, OTHER_CLOCK), Fraction(14))]
    mapping_list = [mapping()]
    missing_list = [1]
    timeline = OrderedTimeline(CLOCK, sample_list, missing_list, [SOURCE], mapping_list)
    sample_list.clear()
    mapping_list.clear()
    missing_list.clear()
    assert timeline.seconds == (Fraction(14),)
    assert timeline.missing_indices == (1,)
    assert timeline.mappings == (mapping(),)
    assert nearest_in_timeline(stamp(14), timeline, max_gap_seconds=0).matched_index == 0


def test_factory_retains_verified_mapping_for_reusable_timeline() -> None:
    timeline = order_timestamps(
        [stamp(2, OTHER_CLOCK), stamp(2, OTHER_CLOCK)], clock_domain=CLOCK, mappings=[mapping()]
    )
    assert timeline.mappings == (mapping(),)
    assert timeline.seconds == (Fraction(14), Fraction(14))
    assert timeline.duplicate_count == 1
    assert nearest_in_timeline(stamp(14), timeline, max_gap_seconds=0).matched_index == 0
