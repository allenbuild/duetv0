"""Stream RGB demo frames into H.264, then validate/export frames without ffmpeg CLI."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Mapping
from fractions import Fraction
from numbers import Integral
from pathlib import Path

import av
import numpy as np

_RAW_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"


def _positive_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _expectations(size: tuple[int, int], fps: int, count: int) -> tuple[int, int, int, int]:
    if not isinstance(size, tuple) or len(size) != 2:
        raise ValueError("size must be a (width, height) tuple")
    width = _positive_integer(size[0], "width")
    height = _positive_integer(size[1], "height")
    return width, height, _positive_integer(fps, "fps"), _positive_integer(count, "expected_frames")


def _artifact_path(path: Path) -> Path:
    path = Path(path)
    if path.resolve().is_relative_to(_RAW_ROOT.resolve()):
        raise ValueError("demo encoding and inspection must not access immutable data/raw")
    return path


def _temporary_sibling(path: Path, suffix: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.stem}.", suffix=suffix)
    os.close(descriptor)
    return Path(name)


def _headers(
    container: av.container.InputContainer,
    path: Path,
    *,
    width: int,
    height: int,
    fps: int,
    expected_frames: int,
) -> tuple[av.video.stream.VideoStream, dict[str, object]]:
    if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
        raise ValueError("demo must contain exactly one video stream and no audio")
    stream = container.streams.video[0]
    codec = stream.codec_context.name
    pixel_format = stream.codec_context.format.name if stream.codec_context.format else None
    if codec != "h264" or pixel_format != "yuv420p":
        raise ValueError(f"expected H.264 yuv420p, found {codec} {pixel_format}")
    if (stream.codec_context.width, stream.codec_context.height) != (width, height):
        raise ValueError("video dimensions differ from the expected size")
    if stream.average_rate is None or Fraction(stream.average_rate) != fps:
        raise ValueError("video average frame rate differs from the expected fps")
    if stream.frames != expected_frames:
        raise ValueError(
            f"header frame count {stream.frames} differs from expected {expected_frames}"
        )
    if stream.time_base is None or stream.time_base <= 0 or stream.start_time != 0:
        raise ValueError("video requires a positive time base and zero initial presentation time")
    if stream.duration is None:
        raise ValueError("video is missing its stream duration")
    duration = stream.duration * Fraction(stream.time_base)
    expected_duration = Fraction(expected_frames, fps)
    if duration != expected_duration:
        raise ValueError("video duration differs from expected_frames / fps")
    metadata = {
        "path": str(path),
        "codec": codec,
        "pixel_format": pixel_format,
        "width": width,
        "height": height,
        "fps": float(stream.average_rate),
        "average_rate": str(stream.average_rate),
        "header_frame_count": stream.frames,
        "frame_count": expected_frames,
        "duration_seconds": float(duration),
        "duration_seconds_exact": str(duration),
        "container_duration_seconds": None
        if container.duration is None
        else container.duration / av.time_base,
        "time_base": str(stream.time_base),
        "audio_stream_count": 0,
        "file_size_bytes": path.stat().st_size,
    }
    return stream, metadata


def encode_video(
    path: Path,
    frames: Iterable[np.ndarray],
    *,
    size: tuple[int, int] = (1920, 1080),
    fps: int = 30,
    expected_frames: int = 253,
) -> dict[str, object]:
    """Encode one RGB uint8 frame at a time; atomically publish a complete MP4.

    Arrays must have shape ``(height, width, 3)``. No whole-video frame collection
    or intermediate PNGs are created. libx264 uses CRF18/medium and yuv420p;
    faststart places playback metadata before the encoded packets. An existing
    destination survives input, encoder-flush, or container-close failures.
    """
    path = _artifact_path(path)
    width, height, fps, expected_frames = _expectations(size, fps, expected_frames)
    if width % 2 or height % 2:
        raise ValueError("yuv420p encoding requires even width and height")
    temporary = _temporary_sibling(path, ".partial.mp4")
    try:
        with av.open(
            str(temporary), mode="w", format="mp4", options={"movflags": "+faststart"}
        ) as container:
            stream = container.add_stream("libx264", rate=fps)
            stream.width, stream.height = width, height
            stream.pix_fmt = "yuv420p"
            stream.time_base = Fraction(1, fps)
            stream.options = {"crf": "18", "preset": "medium"}
            count = 0
            for index, pixels in enumerate(frames):
                if index >= expected_frames:
                    raise ValueError(f"received more than {expected_frames} frames")
                if not isinstance(pixels, np.ndarray) or pixels.dtype != np.uint8:
                    raise ValueError(f"frame {index} must be a uint8 RGB numpy array")
                if pixels.shape != (height, width, 3):
                    raise ValueError(
                        f"frame {index} has shape {pixels.shape}; expected {(height, width, 3)}"
                    )
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                frame.pts = index
                frame.time_base = Fraction(1, fps)
                for packet in stream.encode(frame):
                    container.mux(packet)
                count += 1
            if count != expected_frames:
                raise ValueError(f"received {count} frames; expected {expected_frames}")
            for packet in stream.encode(None):
                container.mux(packet)
        # Container closure (including faststart relocation) must succeed before
        # header validation or replacement of an existing artifact.
        with av.open(str(temporary), mode="r") as container:
            _, metadata = _headers(
                container,
                temporary,
                width=width,
                height=height,
                fps=fps,
                expected_frames=expected_frames,
            )
        metadata.update(
            path=str(path), encoded_frame_count=count, crf=18, preset="medium", faststart=True
        )
        temporary.replace(path)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)


def inspect_video(
    path: Path,
    *,
    expected_size: tuple[int, int] = (1920, 1080),
    expected_fps: int = 30,
    expected_frames: int = 253,
    screenshot_paths: Mapping[int, Path] | None = None,
) -> dict[str, object]:
    """Validate every decoded presentation timestamp and optionally write selected PNGs.

    A single sequential decode retains only the current image and decoder buffers.
    Requested screenshots are staged individually and published after the entire
    video passes validation; failed inspection preserves existing screenshot files.
    """
    path = _artifact_path(path)
    width, height, fps, count_expected = _expectations(expected_size, expected_fps, expected_frames)
    requested: dict[int, Path] = {}
    resolved_destinations = set()
    for index, destination in (screenshot_paths or {}).items():
        if (
            isinstance(index, bool)
            or not isinstance(index, Integral)
            or not 0 <= index < count_expected
        ):
            raise ValueError("screenshot index must be an integer inside the expected frame range")
        destination = _artifact_path(destination)
        resolved = destination.resolve()
        if destination.suffix.lower() != ".png" or resolved == path.resolve():
            raise ValueError("screenshots require a separate .png output path")
        if resolved in resolved_destinations:
            raise ValueError("each requested screenshot needs a distinct output path")
        resolved_destinations.add(resolved)
        requested[int(index)] = destination
    staged: dict[int, Path] = {}
    try:
        with av.open(str(path), mode="r") as container:
            stream, metadata = _headers(
                container, path, width=width, height=height, fps=fps, expected_frames=count_expected
            )
            decoded_count = 0
            first_pts = last_pts = None
            for index, frame in enumerate(container.decode(stream)):
                if index >= count_expected:
                    raise ValueError("decoded more frames than expected")
                if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
                    raise ValueError(f"decoded frame {index} has missing PTS or invalid time base")
                if frame.pts * Fraction(frame.time_base) != Fraction(index, fps):
                    raise ValueError(
                        f"decoded frame {index} does not have the expected presentation time"
                    )
                if (frame.width, frame.height) != (width, height) or frame.format.name != "yuv420p":
                    raise ValueError("decoded frame dimensions or pixel format changed")
                first_pts = frame.pts if first_pts is None else first_pts
                last_pts = frame.pts
                if index in requested:
                    staged[index] = _temporary_sibling(requested[index], ".partial.png")
                    frame.to_image().save(staged[index], format="PNG")
                decoded_count += 1
            if decoded_count != count_expected:
                raise ValueError(f"decoded {decoded_count} frames; expected {count_expected}")
        metadata.update(
            decoded_frame_count=decoded_count,
            first_pts=first_pts,
            last_pts=last_pts,
            screenshots={str(index): str(destination) for index, destination in requested.items()},
        )
        for index, temporary in staged.items():
            temporary.replace(requested[index])
        return metadata
    finally:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
