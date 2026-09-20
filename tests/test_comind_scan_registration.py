"""Small binary PLY fixtures prove bounded sampling without invented registration."""

import numpy as np
import pytest

from duet.adapters.comind.scan import point_cloud_sample_qc, sample_ply_vertices
from duet.qc.result import QCStatus


def write_ply(path, points, *, endian="<", colors=None, declared_count=None):
    count = len(points) if declared_count is None else declared_count
    encoding = "binary_little_endian" if endian == "<" else "binary_big_endian"
    header = f"ply\nformat {encoding} 1.0\nelement vertex {count}\n"
    header += "property double x\nproperty double y\nproperty double z\n"
    properties = [(axis, endian + "f8") for axis in "xyz"]
    if colors is not None:
        header += "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        properties.extend((name, "u1") for name in ("red", "green", "blue"))
    header += "end_header\n"
    records = np.empty(len(points), dtype=np.dtype(properties))
    for column, name in enumerate("xyz"):
        records[name] = np.asarray(points)[:, column]
    if colors is not None:
        for column, name in enumerate(("red", "green", "blue")):
            records[name] = np.asarray(colors)[:, column]
    path.write_bytes(header.encode("ascii") + records.tobytes())


@pytest.mark.parametrize("endian", ["<", ">"])
def test_point_sample_preserves_values_and_rgb_from_explicit_binary_encoding(tmp_path, endian):
    path = tmp_path / "sample.ply"
    points = np.array([[0.25, -1, 2], [3, 4, 5], [9, 10, 11]])
    colors = np.array([[1, 2, 3], [255, 0, 10], [40, 50, 60]], dtype=np.uint8)
    write_ply(path, points, endian=endian, colors=colors)
    before = (path.stat().st_size, path.stat().st_mtime_ns)
    sample = sample_ply_vertices(path, max_points=10)
    np.testing.assert_array_equal(sample.points, points)
    np.testing.assert_array_equal(sample.colors, colors)
    np.testing.assert_array_equal(sample.source_indices, [0, 1, 2])
    assert (path.stat().st_size, path.stat().st_mtime_ns) == before
    for array in (sample.points, sample.colors, sample.source_indices):
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_large_declared_cloud_is_sampled_in_deterministic_bounded_blocks(tmp_path):
    path = tmp_path / "large.ply"
    points = np.column_stack([np.arange(1000)] * 3)
    write_ply(path, points)
    sample = sample_ply_vertices(path, max_points=12, block_size=4)
    assert len(sample.points) == 12
    assert sample.total_vertices == 1000
    np.testing.assert_array_equal(
        sample.source_indices, [0, 1, 2, 3, 498, 499, 500, 501, 996, 997, 998, 999]
    )
    np.testing.assert_array_equal(sample.points, points[sample.source_indices])
    repeated = sample_ply_vertices(path, max_points=12, block_size=4)
    np.testing.assert_array_equal(sample.source_indices, repeated.source_indices)


def test_sample_numeric_qc_never_promotes_units_or_frame_semantics(tmp_path):
    path = tmp_path / "blk_scan_aria_aligned.ply"
    write_ply(path, np.array([[0, 0, 0], [3, 4, 0]], dtype=float))
    result = point_cloud_sample_qc(sample_ply_vertices(path))
    assert result.status is QCStatus.PASS
    assert result.metrics["sample_extent_diagonal_source_units"] == 5
    assert result.metrics["distance_unit"] is None
    assert not result.metrics["registration_verified"]
    assert result.metrics["bounds_scope"] == "sampled vertices only"
    assert "unverified" in result.provenance[0].detail


def test_empty_point_cloud_is_insufficient_data_without_memmapping_empty_payload(tmp_path):
    path = tmp_path / "empty.ply"
    write_ply(path, np.empty((0, 3)))
    sample = sample_ply_vertices(path)
    assert sample.points.shape == (0, 3)
    assert point_cloud_sample_qc(sample).status is QCStatus.INSUFFICIENT_DATA


def test_payload_truncation_is_rejected_before_sampling(tmp_path):
    path = tmp_path / "truncated.ply"
    write_ply(path, np.array([[1, 2, 3]]), declared_count=100)
    with pytest.raises(ValueError, match="truncated"):
        sample_ply_vertices(path, max_points=1)


def test_nonfinite_sample_is_rejected_instead_of_dropping_points(tmp_path):
    path = tmp_path / "nan.ply"
    write_ply(path, np.array([[1, np.nan, 3]]))
    with pytest.raises(ValueError, match="finite"):
        sample_ply_vertices(path)


def test_extent_overflow_fails_qc_instead_of_passing_infinite_metrics(tmp_path):
    path = tmp_path / "huge.ply"
    write_ply(path, np.array([[-1e308, 0, 0], [1e308, 0, 0]]))
    result = point_cloud_sample_qc(sample_ply_vertices(path))
    assert result.status is QCStatus.FAIL
    assert "numerical_failure" in result.metrics


@pytest.mark.parametrize(
    "header,error",
    [
        (
            "ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nproperty float z\nend_header\n",
            "binary",
        ),
        (
            "ply\nformat binary_little_endian 1.0\nelement face 0\nelement vertex 0\nproperty double x\nproperty double y\nproperty double z\nend_header\n",
            "vertex element first",
        ),
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty list uchar double xyz\nend_header\n",
            "fixed stride",
        ),
        (
            "ply\nformat binary_little_endian 1.0\nelement vertex 0\nproperty double x\nproperty double y\nend_header\n",
            "x, y and z",
        ),
    ],
)
def test_unsupported_vertex_layouts_fail_explicitly(tmp_path, header, error):
    path = tmp_path / "unsupported.ply"
    path.write_bytes(header.encode("ascii"))
    with pytest.raises(ValueError, match=error):
        sample_ply_vertices(path)


@pytest.mark.parametrize("limit", [True, 0, -1, 1.5])
def test_sample_limits_require_positive_integers(tmp_path, limit):
    with pytest.raises(ValueError, match="positive integer"):
        sample_ply_vertices(tmp_path / "absent.ply", max_points=limit)
