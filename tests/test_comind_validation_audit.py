"""Independent synthetic checks of chunked trajectory audit against canonical parsing."""

import csv
import io

import numpy as np
import pytest

from duet.adapters.comind.mps import iter_trajectory
from duet.adapters.comind.validation import audit_trajectory, parse_selected_poses
from duet.schemas.common import FrameId, Provenance

PROVENANCE = Provenance("synthetic audit fixture")


def row(time: int, index: int = 0, **changes: object) -> dict[str, object]:
    return {
        "graph_uid": "graph-a",
        "tracking_timestamp_us": time,
        "utc_timestamp_ns": 1_800_000_000_000_000_001 + index,
        "tx_world_device": 0,
        "ty_world_device": 0,
        "tz_world_device": 0,
        "qx_world_device": 0,
        "qy_world_device": 0,
        "qz_world_device": 0,
        "qw_world_device": 1,
        "quality_score": index + 10,
        **changes,
    }


def stream(rows: list[dict[str, object]]) -> io.StringIO:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    output.seek(0)
    return output


@pytest.mark.parametrize("chunk_size", [1, 2, 3, 50])
def test_chunk_boundary_duplicates_and_ties_preserve_first_source_index(chunk_size: int) -> None:
    times = [0, 2, 2, 2, 6, 6, 10]
    queries = [15, -5, 6, 2, 4, 2, 7]
    rows = [row(time, index) for index, time in enumerate(times)]
    audited = audit_trajectory(
        stream(rows), provenance=PROVENANCE, query_timestamps_us=queries, chunk_size=chunk_size
    )
    expected = [
        min(range(len(times)), key=lambda index: (abs(times[index] - query), times[index], index))
        for query in queries
    ]
    assert [sample.source_index for sample in audited.selected_rows] == expected
    assert audited.stats["duplicate_timestamp_count"] == 3
    assert audited.stats["selected_unique_row_count"] == len(set(expected))
    for sample, index in zip(audited.selected_rows, expected, strict=True):
        assert sample.timestamp_us == times[index]
        assert sample.values["quality_score"] == str(index + 10)


def test_tracking_negative_one_is_not_an_invented_missing_sentinel() -> None:
    audited = audit_trajectory(
        stream([row(-1), row(0)]),
        provenance=PROVENANCE,
        query_timestamps_us=[-1],
        chunk_size=1,
    )
    assert audited.selected_rows[0].source_index == 0
    assert audited.stats["tracking_timestamp_us_range"] == [-1, 0]
    assert audited.stats["tracking_timestamp_us_missing_count"] == 0


def test_integer_ranges_and_nearest_residuals_preserve_large_epoch_precision() -> None:
    epoch = 1_800_000_000_000_000_000
    rows = [row(epoch + offset, index) for index, offset in enumerate([0, 2, 4])]
    audited = audit_trajectory(
        stream(rows),
        provenance=PROVENANCE,
        query_timestamps_us=[epoch + 1, epoch + 3],
        chunk_size=1,
    )
    assert [sample.source_index for sample in audited.selected_rows] == [0, 1]
    assert audited.stats["tracking_timestamp_us_range"] == [epoch, epoch + 4]
    assert audited.stats["utc_timestamp_ns_range"] == [epoch + 1, epoch + 3]
    poses = parse_selected_poses(
        audited,
        participant="helper",
        recording_id="synthetic",
        world_frame=FrameId("shared"),
        provenance=PROVENANCE,
    )
    assert [pose.timestamp.raw_value for pose in poses] == [epoch, epoch + 2]
    assert poses[0].utc_timestamp.raw_value == epoch + 1
    assert "data row 0" in poses[0].provenance.parents[0].detail


def test_integer_extremes_do_not_overflow_during_nearest_selection() -> None:
    minimum, maximum = -(2**63), 2**63 - 1
    audited = audit_trajectory(
        stream([row(minimum), row(-1), row(maximum)]),
        provenance=PROVENANCE,
        query_timestamps_us=[minimum + 1, -2, 0, maximum],
        chunk_size=2,
    )
    assert [sample.source_index for sample in audited.selected_rows] == [0, 1, 1, 2]
    assert audited.stats["tracking_timestamp_us_range"] == [minimum, maximum]


def test_utc_only_minus_one_sentinel_and_optional_column() -> None:
    audited = audit_trajectory(
        stream([row(0, utc_timestamp_ns=-1), row(1)]), provenance=PROVENANCE, chunk_size=1
    )
    assert audited.stats["utc_timestamp_ns_missing_count"] == 1
    assert audited.stats["utc_timestamp_ns_range"] == [1_800_000_000_000_000_001] * 2
    without_utc = row(0)
    del without_utc["utc_timestamp_ns"]
    audited = audit_trajectory(stream([without_utc]), provenance=PROVENANCE)
    assert audited.stats["utc_timestamp_ns_range"] is None
    assert not audited.stats["utc_field_present"]


def test_blank_optional_utc_is_missing_consistently_with_canonical_parser() -> None:
    rows = [row(0, utc_timestamp_ns=""), row(1)]
    sample = next(
        iter_trajectory(
            stream(rows), participant="helper", recording_id="synthetic", provenance=PROVENANCE
        )
    )
    assert sample.utc_timestamp.raw_value is None
    assert sample.utc_timestamp_ns_raw is None
    audited = audit_trajectory(stream(rows), provenance=PROVENANCE, chunk_size=1)
    assert audited.stats["utc_timestamp_ns_missing_count"] == 1
    assert audited.stats["utc_timestamp_ns_range"] == [1_800_000_000_000_000_001] * 2


def test_all_rows_contribute_graphs_and_transform_validity_even_when_unselected() -> None:
    rows = [
        row(0),
        row(1, qw_world_device=0),
        row(2, tx_world_device="nan", graph_uid="graph-b"),
        row(3, qw_world_device=2),
        row(4, graph_uid="graph-c"),
    ]
    audited = audit_trajectory(
        stream(rows), provenance=PROVENANCE, query_timestamps_us=[0], chunk_size=2
    )
    assert audited.stats["row_count"] == 5
    assert audited.graph_uids == frozenset({"graph-a", "graph-b", "graph-c"})
    assert audited.stats["transform_valid_count"] == 2
    assert audited.stats["transform_invalid_count"] == 3
    assert audited.stats["invalid_source_indices_first_10"] == [1, 2, 3]
    assert audited.selected_rows[0].source_index == 0
    valid_count = 0
    for value in rows:
        try:
            next(
                iter_trajectory(
                    stream([value]),
                    participant="helper",
                    recording_id="synthetic",
                    provenance=PROVENANCE,
                )
            )
            valid_count += 1
        except ValueError:
            pass
    assert audited.stats["transform_valid_count"] == valid_count


def test_quaternion_boundary_rejection_is_counted_consistently_with_canonical_parser() -> None:
    # np.hypot.reduce and np.linalg.norm disagree by one ulp at this threshold.
    quaternion = [0.24948380439906756, 0.37532383692148724, 0.8218592925331811, -0.348478624965678]
    values = {
        f"q{axis}_world_device": value for axis, value in zip("xyzw", quaternion, strict=True)
    }
    value = row(0, **values)
    audited = audit_trajectory(stream([value]), provenance=PROVENANCE)
    assert audited.stats["transform_invalid_count"] == 1
    with pytest.raises(ValueError, match="unit norm"):
        next(
            iter_trajectory(
                stream([value]),
                participant="helper",
                recording_id="synthetic",
                provenance=PROVENANCE,
            )
        )


@pytest.mark.parametrize("times,chunk_size", [([0, 2, 1], 1), ([0, 2, 1], 3), ([0, -1], 1)])
def test_decreasing_tracking_time_is_rejected(times: list[int], chunk_size: int) -> None:
    with pytest.raises(ValueError, match="timestamps decrease"):
        audit_trajectory(
            stream([row(time) for time in times]), provenance=PROVENANCE, chunk_size=chunk_size
        )


def test_empty_stream_has_no_selected_candidates() -> None:
    header = stream([row(0)]).getvalue().splitlines()[0]
    audited = audit_trajectory(
        io.StringIO(header + "\n"), provenance=PROVENANCE, query_timestamps_us=[0, 1]
    )
    assert audited.selected_rows == (None, None)
    assert audited.stats["row_count"] == 0
    assert audited.graph_uids == frozenset()


@pytest.mark.parametrize(
    "field,value",
    [
        ("tracking_timestamp_us", "1.5"),
        ("utc_timestamp_ns", "1.25"),
        ("tracking_timestamp_us", str(2**63)),
        ("utc_timestamp_ns", str(2**63)),
        ("quality_score", "nan"),
    ],
)
def test_invalid_numeric_tokens_fail_explicitly(field: str, value: str) -> None:
    with pytest.raises(ValueError):
        audit_trajectory(stream([row(0, **{field: value})]), provenance=PROVENANCE)


def test_extra_first_row_cell_cannot_silently_shift_columns_into_an_inferred_index() -> None:
    text = stream([row(0)]).getvalue().rstrip() + ",1\n"
    with pytest.raises(ValueError):
        audit_trajectory(io.StringIO(text), provenance=PROVENANCE)


def test_short_row_is_rejected_even_when_only_optional_tail_field_is_absent() -> None:
    value = row(0)
    value["extra_source_metadata"] = "observed"
    lines = stream([value]).getvalue().splitlines()
    short = lines[1].rsplit(",", 1)[0]
    with pytest.raises(ValueError):
        audit_trajectory(io.StringIO(lines[0] + "\n" + short + "\n"), provenance=PROVENANCE)


def test_audit_retains_invalid_nearest_row_rather_than_silently_skipping_to_farther_pose() -> None:
    audited = audit_trajectory(
        stream([row(0), row(1, qw_world_device=0), row(2)]),
        provenance=PROVENANCE,
        query_timestamps_us=[1],
        chunk_size=1,
    )
    assert audited.selected_rows[0].source_index == 1
    with pytest.raises(ValueError):
        parse_selected_poses(
            audited,
            participant="helper",
            recording_id="synthetic",
            world_frame=FrameId("shared"),
            provenance=PROVENANCE,
        )


def test_query_noninteger_values_are_rejected_before_any_matching() -> None:
    for queries in ([True], [0.5], [np.float64(2)]):
        with pytest.raises(TypeError):
            audit_trajectory(stream([row(0)]), provenance=PROVENANCE, query_timestamps_us=queries)
