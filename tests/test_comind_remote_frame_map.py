"""Sparse remote matching uses exact observed candidates without network access."""

import importlib.util
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from duet.adapters.comind.frame_map import VideoVrsFrameMap
from duet.schemas.common import Provenance


@pytest.fixture
def cli(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    path = scripts / "build_comind_remote_frame_map.py"
    spec = importlib.util.spec_from_file_location("remote_map_cli_tests", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fingerprints(count: int = 30) -> np.ndarray:
    return np.random.default_rng(9).integers(0, 255, (count, 32, 32), dtype=np.uint8)


def test_requested_frames_cover_context_and_exact_pixel_audit_midpoint(cli):
    annotations = [SimpleNamespace(start_frame=200, end_frame=205)]
    selected = cli.requested_frames(1000, annotations, [{"mp4_frame_index": 700}])
    expected = np.array(sorted({0, 499, 700, 999, *range(80, 326)}))
    np.testing.assert_array_equal(selected, expected)


@pytest.mark.parametrize("count,anchors", [(0, []), (-1, []), (10, [{"mp4_frame_index": 10}])])
def test_requested_frames_reject_empty_video_or_out_of_bounds_anchor(cli, count, anchors):
    with pytest.raises(ValueError):
        cli.requested_frames(count, [], anchors)


@pytest.mark.parametrize("start,end", [(None, 5), (10, None), (-1, 5), (20, 10), (0, 1000)])
def test_requested_frames_reject_invalid_annotation_bounds(cli, start, end):
    with pytest.raises(ValueError, match="bounds"):
        cli.requested_frames(1000, [SimpleNamespace(start_frame=start, end_frame=end)], [])


def test_sparse_matches_keep_unrequested_rows_unresolved_and_reuse_timeline(cli):
    images = fingerprints()
    requested = np.array([5, 12, 20], dtype=np.int64)
    result = cli.match_requested(
        images,
        images,
        np.ones(len(images), dtype=bool),
        requested=requested,
        centers=np.arange(len(images)),
        radius=2,
    )
    other = np.ones(len(images), dtype=bool)
    other[requested] = False
    assert (result["status"][requested] == "VERIFIED").all()
    assert (result["status"][other] == "UNRESOLVED").all()
    assert (result["vrs_rgb_frame_index"][other] == -1).all()
    assert (result["match_confidence"][other] == 0).all()
    assert (result["candidate_vrs_rgb_frame_index"][other] == -1).all()
    times = np.full(len(images), -1, dtype=np.int64)
    times[requested] = requested * 33333333
    mapping = VideoVrsFrameMap(
        "00000000-0000-0000-0000-000000000001",
        "helper",
        np.arange(len(images)),
        result["vrs_rgb_frame_index"],
        times,
        result["status"],
        result["match_confidence"],
        np.arange(len(images)),
        Fraction(1, 30),
        Provenance("synthetic exact sparse matches"),
    )
    assert mapping.timestamp_at(0) is None
    assert mapping.timestamp_at(5).raw_value == 5 * 33333333


def test_unknown_candidate_cannot_inflate_unique_match_confidence(cli):
    images = fingerprints()
    valid = np.ones(len(images), dtype=bool)
    valid[4] = False
    result = cli.match_requested(
        images,
        images,
        valid,
        requested=np.array([5, 20]),
        centers=np.arange(len(images)),
        radius=2,
    )
    # Unobserved index 4 could be an equally good match: candidate 5 is unproven.
    assert result["status"][5] == "UNRESOLVED"
    assert result["vrs_rgb_frame_index"][5] == -1
    assert result["status"][20] == "VERIFIED"


def test_remote_fingerprint_cache_rejects_changed_source_identity(tmp_path, cli):
    identity = {
        "source": "https://example.test/helper.vrs",
        "size": 1000,
        "manifest_hash": "abc",
        "etag": '"version1"',
    }
    directory = tmp_path / "fingerprints"
    original = cli.FingerprintCache(
        directory, source=None, count=2, kind="remote_vrs", source_identity=identity
    )
    original.fingerprints[0] = 41
    original.valid[0] = original.done[0] = True
    original.flush()
    for name, value in (("manifest_hash", "changed"), ("etag", '"version2"')):
        with pytest.raises(ValueError, match="source identity changed"):
            cli.FingerprintCache(
                directory,
                source=None,
                count=2,
                kind="remote_vrs",
                source_identity={**identity, name: value},
            )
    reused = cli.FingerprintCache(
        directory, source=None, count=2, kind="remote_vrs", source_identity=identity
    )
    assert reused.done.tolist() == [True, False]
    assert (reused.fingerprints[0] == 41).all()


def test_cli_cannot_open_network_without_explicit_fetch(cli, monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        raise AssertionError("network reader must not be constructed")

    monkeypatch.setattr(cli, "HttpRangeReader", forbidden)
    with pytest.raises(SystemExit, match="2"):
        cli.main(
            [
                "--recording-id",
                "00000000-0000-0000-0000-000000000001",
                "--manifest",
                str(tmp_path / "not-read.json"),
            ]
        )
