"""Synthetic frame correspondence, duplicate/drop behavior, and local-clock gates."""

import hashlib
import importlib.util
import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest

from duet.adapters.comind.frame_map import (
    VideoVrsFrameMap,
    image_errors,
    image_fingerprint,
    load_frame_map,
    match_fingerprints,
    pixel_geometry_scores,
)
from duet.adapters.comind.mps import device_clock
from duet.schemas.common import Provenance


def make_map(**changes):
    values = {
        "recording_id": "synthetic",
        "participant": "helper",
        "mp4_frame_index": [0, 1, 2],
        "vrs_rgb_frame_index": [0, 0, 2],
        "vrs_device_timestamp_ns": [2**53 + 1, 2**53 + 1, 2**53 + 9],
        "status": ["VERIFIED"] * 3,
        "match_confidence": [0.9] * 3,
        "mp4_pts": [0, 512, 1024],
        "mp4_time_base": Fraction(1, 15360),
        "provenance": Provenance("synthetic visual source matches"),
    }
    values.update(changes)
    return VideoVrsFrameMap(**values)


def test_capture_time_is_exact_and_duplicate_mp4_frames_reuse_source_identity():
    result = make_map()
    assert result.timestamp_at(0).raw_value == 2**53 + 1
    assert result.timestamp_at(1).raw_value == result.timestamp_at(0).raw_value
    assert result.timestamp_at(2).seconds == Fraction(2**53 + 9, 10**9)
    assert result.timestamp_at(0).clock_domain == device_clock("synthetic", "helper")
    assert (
        make_map(participant="leader").timestamp_at(0).clock_domain
        != result.timestamp_at(0).clock_domain
    )
    with pytest.raises(ValueError):
        result.status.setflags(write=True)


def test_missing_and_inferred_status_cannot_silently_supply_verified_device_time():
    result = make_map(
        vrs_rgb_frame_index=[-1, 0, 2],
        vrs_device_timestamp_ns=[-1, 5, 9],
        status=["UNRESOLVED", "INFERRED", "VERIFIED"],
    )
    assert result.timestamp_at(0) is None
    assert result.timestamp_at(1) is None
    assert result.timestamp_at(1, allow_inferred=True).raw_value == 5


@pytest.mark.parametrize(
    "changes",
    [
        {"vrs_device_timestamp_ns": [-1, -1, 9]},
        {"vrs_rgb_frame_index": [0, 2, 1]},
        {"vrs_device_timestamp_ns": [1, 2, 3]},
        {"mp4_frame_index": [1, 2, 3]},
        {"mp4_pts": [0, 0, 512]},
        {"match_confidence": [0.9, np.nan, 0.9]},
        {"match_confidence": [1j, 1j, 1j]},
        {"status": ["UNKNOWN"] * 3},
        {"status": ["UNRESOLVED", "VERIFIED", "VERIFIED"]},
    ],
)
def test_bad_mapping_columns_rejected(changes):
    with pytest.raises(ValueError):
        make_map(**changes)


@pytest.mark.parametrize("index", [True, 1.0, -1, 3])
def test_frame_index_has_no_implicit_coercion_or_negative_wraparound(index):
    with pytest.raises((TypeError, IndexError)):
        make_map().timestamp_at(index)


def synthetic_fingerprints(count=12):
    return np.random.default_rng(5).integers(5, 245, (count, 16, 16), dtype=np.uint8)


def test_direct_visual_matching_handles_duplicates_skips_and_piecewise_offsets():
    source = synthetic_fingerprints()
    expected = np.array([0, 1, 1, 3, 4, 5, 5, 6, 8, 9], dtype=np.int64)
    video = source[expected] + 1
    result = match_fingerprints(
        video,
        source,
        candidate_centers=np.arange(len(video)),
        vrs_valid=np.ones(len(source), dtype=bool),
        radius=4,
    )
    np.testing.assert_array_equal(result["vrs_rgb_frame_index"], expected)
    assert np.all(result["status"] == "VERIFIED")
    assert np.any(np.diff(expected) == 0)
    assert np.any(np.diff(expected) > 1)


def test_identical_source_images_are_ambiguous_even_when_a_monotonic_guess_exists():
    source = synthetic_fingerprints(5)
    source[3] = source[2]
    result = match_fingerprints(
        source[2:3], source, candidate_centers=[2], vrs_valid=np.ones(5, dtype=bool), radius=3
    )
    assert result["vrs_rgb_frame_index"].tolist() == [-1]
    assert result["mapping_reason"].tolist() == ["ambiguous_image_match"]


def test_exact_ties_stay_unresolved_even_when_configured_margin_is_zero():
    source = np.zeros((3, 8, 8), dtype=np.uint8)
    result = match_fingerprints(
        source[:1],
        source,
        candidate_centers=[1],
        vrs_valid=np.ones(3, dtype=bool),
        radius=2,
        minimum_margin=0,
        minimum_ratio=1,
    )
    assert result["status"].tolist() == ["UNRESOLVED"]


def test_invalid_native_images_and_wrong_images_are_not_assigned():
    source = synthetic_fingerprints(5)
    valid = np.ones(5, dtype=bool)
    valid[2] = False
    result = match_fingerprints(
        source[2:3], source, candidate_centers=[2], vrs_valid=valid, radius=3
    )
    assert result["vrs_rgb_frame_index"].tolist() == [-1]
    assert result["status"].tolist() == ["UNRESOLVED"]


def test_nonmonotonic_best_images_are_rejected_not_silently_reordered():
    source = synthetic_fingerprints(5)
    result = match_fingerprints(
        source[[2, 1, 3]],
        source,
        candidate_centers=[2, 2, 3],
        vrs_valid=np.ones(5, dtype=bool),
        radius=4,
    )
    assert result["vrs_rgb_frame_index"].tolist() == [-1, -1, 3]
    assert result["mapping_reason"][0] == "nonmonotonic_visual_candidates"


def test_candidate_window_edge_is_not_treated_as_complete_visual_search():
    source = synthetic_fingerprints(10)
    result = match_fingerprints(
        source[4:5], source, candidate_centers=[2], vrs_valid=np.ones(10, dtype=bool), radius=2
    )
    assert result["mapping_reason"].tolist() == ["search_boundary"]


def test_rotation_and_actual_jpeg_reencoding_preserve_image_correspondence():
    y, x = np.mgrid[:96, :96]
    native = np.stack([(x * 2) % 256, (y * 2) % 256, ((x + y) * 2) % 256], axis=-1).astype(np.uint8)
    native[12:37, 49:81] = [235, 20, 10]
    upright = np.ascontiguousarray(np.rot90(native, -1))
    encoder = av.CodecContext.create("mjpeg", "w")
    encoder.width, encoder.height = 96, 96
    encoder.pix_fmt = "yuvj420p"
    encoder.time_base = Fraction(1, 30)
    packets = encoder.encode(av.VideoFrame.from_ndarray(upright, format="rgb24"))
    decoder = av.CodecContext.create("mjpeg", "r")
    decoded = decoder.decode(packets[0])[0].to_ndarray(format="rgb24")
    target = image_fingerprint(decoded)
    original = image_fingerprint(native)
    candidates = np.stack([np.rot90(original, -k) for k in range(4)])
    errors = image_errors(target, candidates)
    assert int(np.argmin(errors)) == 1
    assert errors[1] < 2
    assert min(errors[[0, 2, 3]]) > 25


def test_loader_checks_payload_checksum_before_assigning_device_time(tmp_path):
    path = tmp_path / "helper_frame_map.npz"
    result = make_map()
    np.savez(
        path,
        **{
            name: getattr(result, name)
            for name in (
                "mp4_frame_index",
                "vrs_rgb_frame_index",
                "vrs_device_timestamp_ns",
                "status",
                "match_confidence",
                "mp4_pts",
            )
        },
        mp4_time_base_numerator=1,
        mp4_time_base_denominator=15360,
    )
    metadata = {
        "recording_id": "synthetic",
        "participant": "helper",
        "mapping_evidence": "synthetic direct matches",
        "npz_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    assert load_frame_map(path).timestamp_at(2).raw_value == 2**53 + 9
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="checksum"):
        load_frame_map(path)


def test_full_resolution_geometry_check_identifies_shift_and_unit_scale():
    source = np.random.default_rng(40).integers(0, 256, (64, 64), dtype=np.uint8)
    direct = pixel_geometry_scores(source, source, border=4, prefilter_passes=3)
    assert direct["zero_shift_unit_scale_wins"]
    assert direct["ranked_candidates"][0]["rmse"] == 0
    shifted = np.roll(source, 1, axis=1)
    diagnostic = pixel_geometry_scores(shifted, source, border=4)
    best = diagnostic["ranked_candidates"][0]
    assert best["dx"] == -1
    assert best["dy"] == 0
    assert best["scale"] == 1
    assert not diagnostic["zero_shift_unit_scale_wins"]


def builder_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "build_comind_frame_map.py"
    spec = importlib.util.spec_from_file_location("duet_frame_map_builder_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fingerprint_cache_guards_source_identity_and_skips_second_decode(tmp_path, monkeypatch):
    builder = builder_module()
    source = tmp_path / "synthetic.mp4"
    source.write_bytes(b"synthetic headers mocked")
    decoder_calls = []

    class Frame:
        def __init__(self, index):
            self.pts = index * 512
            self.value = index

        def reformat(self, **kwargs):
            return self

        def to_ndarray(self):
            return np.full((32, 32), self.value, dtype=np.uint8)

    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(
                video=[
                    SimpleNamespace(
                        frames=4,
                        width=96,
                        height=64,
                        time_base=Fraction(1, 15360),
                        codec_context=SimpleNamespace(thread_count=0),
                    )
                ]
            )

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            decoder_calls.append(True)
            return iter(Frame(index) for index in range(4))

    monkeypatch.setattr(builder.av, "open", lambda *args, **kwargs: Container())
    first, metadata = builder.cache_mp4(source, tmp_path / "cache")
    assert first.done.all()
    assert first.pts.tolist() == [0, 512, 1024, 1536]
    assert metadata["time_base_denominator"] == 15360
    second, _ = builder.cache_mp4(source, tmp_path / "cache")
    assert len(decoder_calls) == 1
    np.testing.assert_array_equal(first.fingerprints, second.fingerprints)
    source.write_bytes(b"changed source bytes")
    with pytest.raises(ValueError, match="source identity"):
        builder.cache_mp4(source, tmp_path / "cache")
