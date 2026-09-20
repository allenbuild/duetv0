"""Stream excerpts from rendered demo MP4s without mixing unrelated scene states.

Ordinals and FPS in this module describe presentation frames, never device time.
All labels remain embedded in their original rendered frames.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from fractions import Fraction
from numbers import Integral
from pathlib import Path

import av
import numpy as np

from duet.visualization.demo_encoding import encode_video, inspect_video

_RAW_ROOT = Path(__file__).resolve().parents[3] / "data" / "raw"


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def choose_segment(
    clip_start: int, clip_end: int, event_start: int, event_end: int, frames: int
) -> tuple[int, int]:
    """Return inclusive source indices centered on the event and bounded by the clip.

    A half-frame centering tie chooses the earlier start. An event longer than
    the requested excerpt is allowed; its center still determines the selection.
    """
    clip_start = _integer(clip_start, "clip_start")
    clip_end = _integer(clip_end, "clip_end")
    event_start = _integer(event_start, "event_start")
    event_end = _integer(event_end, "event_end")
    frames = _integer(frames, "frames", minimum=1)
    if not clip_start <= event_start <= event_end <= clip_end:
        raise ValueError("event must be an ordered inclusive interval within the clip")
    if frames > clip_end - clip_start + 1:
        raise ValueError("requested segment is longer than the clip")
    centered_start = (event_start + event_end - frames + 1) // 2
    start = min(max(centered_start, clip_start), clip_end - frames + 1)
    return start, start + frames - 1


def _rendered_path(value: object) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise TypeError("rendered video paths must be filesystem paths")
    path = Path(value)
    if path.resolve().is_relative_to(_RAW_ROOT.resolve()):
        raise ValueError("compilations may access rendered videos only, never data/raw")
    return path


@dataclass(frozen=True)
class _Source:
    path: Path
    size: tuple[int, int]
    frame_count: int
    time_base: Fraction
    signature: tuple[int, int]


@dataclass(frozen=True)
class _Segment:
    source: _Source
    start: int
    end: int
    annotation_id: str
    object_category: str

    @property
    def count(self) -> int:
        return self.end - self.start + 1


def _signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def _probe_source(path: Path, fps: int) -> _Source:
    signature = _signature(path)
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 0:
            raise ValueError("source demo must contain one video stream and no audio")
        stream = container.streams.video[0]
        codec = stream.codec_context
        if codec.name != "h264" or codec.format is None or codec.format.name != "yuv420p":
            raise ValueError("source demo must be H.264 yuv420p")
        size = codec.width, codec.height
        if any(value <= 0 or value % 2 for value in size):
            raise ValueError("source demo dimensions must be positive and even")
        if stream.average_rate is None or Fraction(stream.average_rate) != fps:
            raise ValueError("source demo FPS differs from compilation FPS")
        if stream.frames < 1 or stream.start_time != 0:
            raise ValueError("source demo requires a frame count and zero initial PTS")
        if stream.time_base is None or stream.time_base <= 0 or stream.duration is None:
            raise ValueError("source demo requires a positive time base and duration")
        time_base = Fraction(stream.time_base)
        if stream.duration * time_base != Fraction(stream.frames, fps):
            raise ValueError("source demo duration is inconsistent with its CFR frame count")
        source = _Source(path, size, stream.frames, time_base, signature)
    if _signature(path) != signature:
        raise ValueError("source demo changed during header inspection")
    return source


def _segments(
    specifications: Sequence[Mapping[str, object]], output: Path, fps: int, transition_frames: int
) -> list[_Segment]:
    if not specifications:
        raise ValueError("at least one compilation segment is required")
    sources: dict[Path, _Source] = {}
    segments = []
    for specification in specifications:
        if not isinstance(specification, Mapping):
            raise TypeError("each segment must be a mapping")
        required = {"path", "start_ordinal", "end_ordinal", "annotation_id", "object_category"}
        if not required <= specification.keys():
            raise ValueError(
                f"segment is missing fields: {sorted(required - specification.keys())}"
            )
        path = _rendered_path(specification["path"])
        resolved = path.resolve()
        if resolved == output.resolve():
            raise ValueError("compilation output must differ from every source video")
        start = _integer(specification["start_ordinal"], "start_ordinal")
        end = _integer(specification["end_ordinal"], "end_ordinal")
        labels = specification["annotation_id"], specification["object_category"]
        if any(not isinstance(label, str) or not label.strip() for label in labels):
            raise ValueError("annotation ID and object category must be nonempty source labels")
        if resolved not in sources:
            sources[resolved] = _probe_source(path, fps)
        source = sources[resolved]
        if not start <= end < source.frame_count:
            raise ValueError("segment ordinals must form an inclusive interval inside its source")
        segments.append(_Segment(source, start, end, *labels))
    if len({source.size for source in sources.values()}) != 1:
        raise ValueError("all source demo dimensions must match")
    outgoing, incoming = transition_frames // 2, transition_frames - transition_frames // 2
    for index, segment in enumerate(segments):
        affected = (incoming if index > 0 else 0) + (outgoing if index < len(segments) - 1 else 0)
        if affected > segment.count:
            raise ValueError("transition fades must not overlap within a segment")
    return segments


def _decode_segment(segment: _Segment, fps: int) -> Iterator[np.ndarray]:
    """Seek once, then verify exact decoded CFR ordinals through the selected endpoint."""
    source = segment.source
    if _signature(source.path) != source.signature:
        raise ValueError("source demo changed after header inspection")
    with av.open(str(source.path), mode="r") as container:
        stream = container.streams.video[0]
        target_pts = Fraction(segment.start, fps) / source.time_base
        if target_pts.denominator != 1:
            raise ValueError("source time base cannot represent the requested frame exactly")
        container.seek(int(target_pts), stream=stream, backward=True, any_frame=False)
        previous = None
        selected_count = 0
        for frame in container.decode(stream):
            if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
                raise ValueError("source frame is missing a valid presentation timestamp")
            ordinal_fraction = frame.pts * Fraction(frame.time_base) * fps
            if ordinal_fraction.denominator != 1:
                raise ValueError("source frame PTS is not on its declared CFR grid")
            ordinal = int(ordinal_fraction)
            if not 0 <= ordinal < source.frame_count:
                raise ValueError("source frame PTS is outside its declared frame count")
            if previous is not None and ordinal != previous + 1:
                raise ValueError("source decoded frame ordinals contain a gap or duplicate")
            previous = ordinal
            if (frame.width, frame.height) != source.size or frame.format.name != "yuv420p":
                raise ValueError("source decoded dimensions or pixel format changed")
            if ordinal < segment.start:
                continue
            if ordinal != segment.start + selected_count:
                raise ValueError("seek did not reach the exact selected source frame")
            yield frame.to_ndarray(format="rgb24")
            selected_count += 1
            if ordinal == segment.end:
                break
        if selected_count != segment.count:
            raise ValueError("source video ended before the selected segment was complete")
    if _signature(source.path) != source.signature:
        raise ValueError("source demo changed during segment decoding")


def _frames(segments: Sequence[_Segment], fps: int, transition_frames: int) -> Iterator[np.ndarray]:
    outgoing, incoming = transition_frames // 2, transition_frames - transition_frames // 2
    for index, segment in enumerate(segments):
        with closing(_decode_segment(segment, fps)) as decoded:
            for ordinal, pixels in enumerate(decoded):
                gain = 1.0
                if index > 0 and ordinal < incoming:
                    gain = ordinal / incoming
                if index < len(segments) - 1 and ordinal >= segment.count - outgoing:
                    gain = (segment.count - ordinal - 1) / outgoing
                if gain != 1:
                    pixels = np.rint(pixels.astype(np.float32) * gain).astype(np.uint8)
                yield pixels


def export_compilation(
    output: Path,
    segments: Sequence[Mapping[str, object]],
    *,
    fps: int = 30,
    transition_frames: int = 8,
) -> dict[str, object]:
    """Encode and verify a compilation from inclusive ordinals in rendered MP4s.

    A cut uses ``transition_frames // 2`` outgoing frames and the remaining
    incoming frames to fade through black. Each pixel comes from only one scene
    state. No frames are added, removed, or blended between clips; first/last
    compilation edges are unchanged. Zero disables fades. Source paths and
    output must be outside ``data/raw``. Existing output survives any failure.
    """
    output = _rendered_path(output)
    fps = _integer(fps, "fps", minimum=1)
    transition_frames = _integer(transition_frames, "transition_frames")
    selected = _segments(segments, output, fps, transition_frames)
    size = selected[0].source.size
    frame_count = sum(segment.count for segment in selected)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        dir=output.parent, prefix=f".{output.stem}.", suffix=".compilation.mp4"
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        with closing(_frames(selected, fps, transition_frames)) as frames:
            encoded = encode_video(
                temporary, frames, size=size, fps=fps, expected_frames=frame_count
            )
        inspected = inspect_video(
            temporary, expected_size=size, expected_fps=fps, expected_frames=frame_count
        )
        offset = 0
        segment_metadata = []
        for segment in selected:
            segment_metadata.append(
                {
                    "path": str(segment.source.path),
                    "start_ordinal": segment.start,
                    "end_ordinal": segment.end,
                    "frame_count": segment.count,
                    "annotation_id": segment.annotation_id,
                    "object_category": segment.object_category,
                    "output_start_ordinal": offset,
                    "output_end_ordinal": offset + segment.count - 1,
                }
            )
            offset += segment.count
        metadata = {
            **encoded,
            **inspected,
            "path": str(output),
            "segments": segment_metadata,
            "transition_frames": transition_frames,
            "transition_policy": "fade through black at internal cuts; no cross-clip blending",
        }
        temporary.replace(output)
        return metadata
    finally:
        temporary.unlink(missing_ok=True)
