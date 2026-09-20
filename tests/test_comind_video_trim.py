"""Image matching and bounded seek checks using tiny synthetic MP4s only."""

import importlib.util
from fractions import Fraction
from itertools import pairwise
from pathlib import Path

import av
import numpy as np
import pytest


@pytest.fixture(scope="module")
def probe():
    path = Path(__file__).resolve().parents[1] / "scripts" / "probe_comind_video_trim.py"
    spec = importlib.util.spec_from_file_location("duet_video_trim_probe_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_consecutive_image_search_uses_content_and_rejects_single_frame_distractor(probe):
    rng = np.random.default_rng(7)
    original = {i: rng.integers(10, 245, (16, 16), dtype=np.uint8) for i in range(30)}
    original[2] = original[11].copy()
    # Mild intensity changes stand in for re-encoding; no timestamps are available.
    targets = {i: original[i + 11] + 1 for i in range(3)}
    assert probe.rank_fingerprints(targets[0], original)[0][1] == 2
    ranked = probe.consecutive_offset_search(targets, original)
    assert ranked[0] == (1.0, 11)
    assert ranked[1][0] > 20


def test_repeated_images_are_ambiguous_and_ties_do_not_establish_capture_identity(probe):
    image = np.arange(64, dtype=np.uint8).reshape(8, 8)
    ranked = probe.rank_fingerprints(image, {12: image, 10: image.copy(), 11: image + 1})
    assert ranked == [(0.0, 10), (0.0, 12), (1.0, 11)]
    assert ranked[1][0] - ranked[0][0] == 0


@pytest.mark.parametrize("second", [np.ones((2, 2), dtype=float), np.ones((3, 3), dtype=np.uint8)])
def test_fingerprint_comparison_rejects_shape_and_dtype_coercion(probe, second):
    with pytest.raises(ValueError, match="equal-sized uint8"):
        probe.fingerprint_error(np.ones((2, 2), dtype=np.uint8), second)


def test_sample_windows_include_endpoints_and_consecutive_neighborhoods(probe):
    windows = probe.merged_windows([0, 20, 50, 70, 99], count=100)
    assert windows[0] == (0, 7)
    assert windows[-1] == (92, 99)
    assert (49, 51) in windows
    assert all(right[0] > left[1] for left, right in pairwise(windows))


def write_video(path: Path, count: int = 24) -> None:
    rng = np.random.default_rng(41)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=30)
        stream.width = 96
        stream.height = 64
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "18", "g": "6"}
        for index in range(count):
            pixels = rng.integers(0, 256, (64, 96, 3), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            frame.time_base = Fraction(1, 30)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def test_reader_prefix_seek_and_eof_agree_with_synthetic_frame_ordinals(tmp_path, probe):
    path = tmp_path / "synthetic.mp4"
    write_video(path)
    reader = probe.FingerprintReader(path, size=16)
    try:
        prefix = reader.read(0, 17)
        assert list(prefix) == list(range(18))
        assert reader.ordinal_prefix_frames_verified >= 18
        seeked = reader.read(13, 17)
        for index in seeked:
            np.testing.assert_array_equal(seeked[index], prefix[index])
        tail = reader.read(20, 23, full_rgb_hashes=True)
        assert list(tail) == [20, 21, 22, 23]
        assert reader.eof_last_index == 23
        assert len(set(reader.rgb_hashes.values())) == 4
        assert reader.headers()["duration_pts"] == 24 * reader.headers()["pts_step_per_frame"]
    finally:
        reader.close()


def test_initial_search_needs_images(probe):
    with pytest.raises(ValueError, match="nonempty"):
        probe.consecutive_offset_search({}, {})


def test_exact_repeated_image_suffix_locates_boundary_without_capture_time_claim(probe):
    result = probe.identical_suffix_run({100: "different", 101: "same", 102: "same", 103: "same"})
    assert result["start_index"] == 101
    assert result["end_index"] == 103
    assert result["length"] == 3
    assert result["preceding_index"] == 100
    assert result["boundary_observed_inside_window"]
    assert not result["capture_timestamp_duplication_verified"]
    truncated = probe.identical_suffix_run({101: "same", 102: "same"})
    assert not truncated["boundary_observed_inside_window"]
    with pytest.raises(ValueError, match="consecutive"):
        probe.identical_suffix_run({101: "same", 103: "same"})


def test_summary_distinguishes_start_match_nonconstant_offsets_and_many_to_one(probe):
    rows = []
    for trimmed, untrimmed in [(0, 11), (1, 11), (20, 30)]:
        rows.append(
            {
                "trimmed_index": trimmed,
                "best_untrimmed_index": untrimmed,
                "best_offset": untrimmed - trimmed,
                "best_rmse": 0.5,
                "second_rmse": 4.0,
                "second_over_best_rmse": 8.0,
                "confident_image_match": True,
            }
        )
    result = {"matches": rows, "trimmed": {"frame_count": 21}, "untrimmed": {"frame_count": 31}}
    probe.summarize_correspondences(result)
    assert result["first_frame_visual_correspondence"]["best_untrimmed_index"] == 11
    assert result["first_frame_offset_implied_end_untrimmed_index"] == 31
    assert result["first_frame_offset_implied_end_is_out_of_range"]
    assert result["constant_offset_rejected_by_confident_visual_correspondences"]
    anomaly = result["consecutive_visual_index_irregularities"][0]
    assert anomaly["trimmed_indices"] == [0, 1]
    assert anomaly["best_untrimmed_indices"] == [11, 11]
    assert "cause is unknown" in anomaly["interpretation"]
