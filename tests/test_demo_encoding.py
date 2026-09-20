"""Small real H.264 fixtures and failure cases; no datasets or external binaries."""

import json
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import av
import numpy as np
import pytest
from PIL import Image

from duet.visualization import demo_encoding as module
from duet.visualization.demo_encoding import encode_video, inspect_video


def frames(count: int, *, size=(64, 48)):
    for index in range(count):
        pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        pixels[:] = [index * 25, 80, 150]
        yield pixels


def encode_small(path: Path, count=6):
    return encode_video(path, frames(count), size=(64, 48), fps=6, expected_frames=count)


def test_h264_roundtrip_metadata_exact_timeline_and_requested_pngs(tmp_path):
    path = tmp_path / "demo.mp4"
    encoded = encode_small(path)
    screenshots = {
        0: tmp_path / "first.png",
        2: tmp_path / "thumbnail.png",
        5: tmp_path / "last.png",
    }
    result = inspect_video(
        path,
        expected_size=(64, 48),
        expected_fps=6,
        expected_frames=6,
        screenshot_paths=screenshots,
    )
    assert encoded["encoded_frame_count"] == result["decoded_frame_count"] == 6
    assert result["codec"] == "h264"
    assert result["pixel_format"] == "yuv420p"
    assert result["duration_seconds"] == 1
    assert result["fps"] == 6
    assert result["audio_stream_count"] == 0
    assert result["file_size_bytes"] == path.stat().st_size > 0
    assert result["first_pts"] == 0
    json.dumps(result, allow_nan=False)
    payload = path.read_bytes()
    assert payload.index(b"moov") < payload.index(b"mdat")
    for index, screenshot in screenshots.items():
        with Image.open(screenshot) as image:
            assert image.format == "PNG"
            assert image.size == (64, 48)
            assert abs(np.asarray(image)[24, 32, 0].astype(int) - index * 25) < 8
    assert not list(tmp_path.glob(".*.partial.*"))


@pytest.mark.parametrize("count", [0, 2, 4])
def test_incorrect_input_count_preserves_existing_output(tmp_path, count):
    path = tmp_path / "demo.mp4"
    path.write_bytes(b"existing valid artifact placeholder")
    with pytest.raises(ValueError, match="frames"):
        encode_video(path, frames(count), size=(64, 48), fps=6, expected_frames=3)
    assert path.read_bytes() == b"existing valid artifact placeholder"
    assert not list(tmp_path.glob(".*.partial.mp4"))


@pytest.mark.parametrize(
    "pixels",
    [
        np.zeros((48, 64, 3), dtype=float),
        np.zeros((48, 64, 4), dtype=np.uint8),
        np.zeros((64, 48, 3), dtype=np.uint8),
        [[0, 0, 0]],
    ],
)
def test_bad_rgb_arrays_fail_without_publishing(tmp_path, pixels):
    path = tmp_path / "bad.mp4"
    with pytest.raises(ValueError, match="frame 0"):
        encode_video(path, [pixels], size=(64, 48), fps=6, expected_frames=1)
    assert not path.exists()
    assert not list(tmp_path.glob(".*.partial.mp4"))


def test_generator_failure_preserves_existing_output(tmp_path):
    path = tmp_path / "demo.mp4"
    path.write_bytes(b"previous")

    def failing():
        yield next(frames(1))
        raise RuntimeError("render failed")

    with pytest.raises(RuntimeError, match="render failed"):
        encode_video(path, failing(), size=(64, 48), fps=6, expected_frames=3)
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*.partial.mp4"))


def test_encoder_close_errors_propagate_and_do_not_replace_existing_output(tmp_path, monkeypatch):
    path = tmp_path / "demo.mp4"
    path.write_bytes(b"previous")

    class Container:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            raise OSError("container close failed")

        def add_stream(self, *args, **kwargs):
            return SimpleNamespace(encode=lambda frame: [])

    monkeypatch.setattr(module.av, "open", lambda *args, **kwargs: Container())
    with pytest.raises(OSError, match="container close failed"):
        encode_video(path, frames(1), size=(64, 48), fps=6, expected_frames=1)
    assert path.read_bytes() == b"previous"
    assert not list(tmp_path.glob(".*.partial.mp4"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": (63, 48)},
        {"size": (64, 0)},
        {"fps": 0},
        {"fps": 6.0},
        {"expected_frames": 0},
        {"expected_frames": True},
    ],
)
def test_invalid_encoding_expectations_do_not_create_files(tmp_path, kwargs):
    values = {"size": (64, 48), "fps": 6, "expected_frames": 1}
    values.update(kwargs)
    with pytest.raises(ValueError):
        encode_video(tmp_path / "demo.mp4", frames(1), **values)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_frames": 5},
        {"expected_fps": 7},
        {"expected_size": (32, 48)},
    ],
)
def test_inspection_rejects_metadata_mismatches(tmp_path, kwargs):
    path = tmp_path / "demo.mp4"
    encode_small(path)
    values = {"expected_size": (64, 48), "expected_fps": 6, "expected_frames": 6}
    values.update(kwargs)
    with pytest.raises(ValueError):
        inspect_video(path, **values)


@pytest.mark.parametrize("index", [-1, 6, True, 1.5])
def test_invalid_screenshot_indices_are_rejected_before_decoding(tmp_path, index):
    with pytest.raises(ValueError, match="screenshot index"):
        inspect_video(
            tmp_path / "absent.mp4",
            expected_size=(64, 48),
            expected_fps=6,
            expected_frames=6,
            screenshot_paths={index: tmp_path / "bad.png"},
        )


def test_duplicate_screenshot_destinations_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        inspect_video(
            tmp_path / "absent.mp4",
            expected_size=(64, 48),
            expected_fps=6,
            expected_frames=6,
            screenshot_paths={0: tmp_path / "same.png", 2: tmp_path / "same.png"},
        )


def test_inspection_checks_each_decoded_pts_and_preserves_screenshots_on_failure(
    tmp_path, monkeypatch
):
    path = tmp_path / "fake.mp4"
    path.write_bytes(b"synthetic headers")
    screenshot = tmp_path / "first.png"
    screenshot.write_bytes(b"previous screenshot")
    context = SimpleNamespace(
        name="h264", format=SimpleNamespace(name="yuv420p"), width=64, height=48
    )
    stream = SimpleNamespace(
        codec_context=context,
        average_rate=Fraction(6),
        frames=3,
        time_base=Fraction(1, 6),
        start_time=0,
        duration=3,
    )

    class Container:
        def __init__(self):
            self.streams = SimpleNamespace(video=[stream], audio=[])
            self.duration = 500_000

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def decode(self, stream):
            for index in (0, 2, 3):
                frame = av.VideoFrame.from_ndarray(
                    np.zeros((48, 64, 3), dtype=np.uint8), format="rgb24"
                )
                frame = frame.reformat(format="yuv420p")
                frame.pts, frame.time_base = index, Fraction(1, 6)
                yield frame

    monkeypatch.setattr(module.av, "open", lambda *args, **kwargs: Container())
    with pytest.raises(ValueError, match="presentation time"):
        inspect_video(
            path,
            expected_size=(64, 48),
            expected_fps=6,
            expected_frames=3,
            screenshot_paths={0: screenshot},
        )
    assert screenshot.read_bytes() == b"previous screenshot"
    assert not list(tmp_path.glob(".*.partial.png"))


def test_raw_guard_rejects_before_any_file_access(tmp_path, monkeypatch):
    immutable = tmp_path / "immutable"
    monkeypatch.setattr(module, "_RAW_ROOT", immutable)
    with pytest.raises(ValueError, match="data/raw"):
        encode_video(immutable / "demo.mp4", frames(1), size=(64, 48), fps=6, expected_frames=1)
    with pytest.raises(ValueError, match="data/raw"):
        inspect_video(immutable / "demo.mp4")
    assert not immutable.exists()
