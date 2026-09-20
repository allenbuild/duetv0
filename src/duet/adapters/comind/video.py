"""Lazy ego-video access preserving exact MP4 presentation timestamps.

The video PTS clock is deliberately separate from device tracking and UTC clocks.
Neither equal durations nor synchronized filenames verify a numerical clock map.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING

from duet.schemas.common import ParticipantId, Provenance, require_name
from duet.schemas.time import ClockDomain, as_fraction
from duet.synchronization.matching import MatchStatus

if TYPE_CHECKING:
    import av


@dataclass(frozen=True)
class VideoMetadata:
    """Container metadata, keeping stream duration distinct from container duration."""

    path: Path
    participant_id: ParticipantId
    clock_domain: ClockDomain
    codec: str
    width: int
    height: int
    stream_index: int
    time_base: Fraction
    start_pts: int | None
    duration_pts: int | None
    frame_count: int
    average_rate: Fraction | None
    container_duration_us: int | None
    video_stream_count: int
    audio_stream_count: int
    provenance: Provenance

    @property
    def duration_seconds(self) -> Fraction | None:
        return None if self.duration_pts is None else self.duration_pts * self.time_base

    @property
    def start_seconds(self) -> Fraction | None:
        return None if self.start_pts is None else self.start_pts * self.time_base


@dataclass(frozen=True)
class VideoFrameSample:
    """One decoded frame with original integer PTS and rational seconds-per-tick."""

    pts: int
    time_base: Fraction
    clock_domain: ClockDomain
    provenance: Provenance
    frame: av.VideoFrame = field(repr=False)

    @property
    def seconds(self) -> Fraction:
        """Exact presentation seconds in this video's own clock domain."""
        return self.pts * self.time_base


@dataclass(frozen=True)
class VideoFrameMatch:
    """Nearest candidate and signed candidate-minus-query residual in seconds.

    Rejected candidates remain available for diagnostics; check ``accepted``
    before using the frame. Ties select the earlier presentation timestamp.
    """

    status: MatchStatus
    candidate: VideoFrameSample | None
    query_seconds: Fraction
    residual_seconds: Fraction | None
    max_gap_seconds: Fraction

    @property
    def accepted(self) -> bool:
        return self.status is MatchStatus.MATCHED


class EgoVideo:
    """Open the selected ego MP4 only when metadata or a frame is requested.

    ``nearest_frame`` seeks to a preceding keyframe, decodes until the requested
    time is bracketed, and retains at most one candidate image. PyAV decoder
    buffers remain bounded; no complete-video image collection is created.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        participant_id: ParticipantId,
        recording_uuid: str,
    ) -> None:
        if not isinstance(participant_id, ParticipantId):
            raise TypeError("participant_id must be explicit")
        require_name(recording_uuid, "recording UUID")
        self.path = Path(path)
        self.participant_id = participant_id
        self.clock_domain = ClockDomain(
            f"comind:{recording_uuid}:{participant_id.name}:video_pts:{self.path.name}"
        )
        self.provenance = Provenance(str(self.path), detail="Original MP4 PTS and time_base")

    @staticmethod
    def _video_stream(container: av.container.InputContainer) -> av.video.stream.VideoStream:
        if len(container.streams.video) != 1:
            raise ValueError("expected exactly one video stream in the selected ego MP4")
        stream = container.streams.video[0]
        if stream.time_base is None or stream.time_base <= 0:
            raise ValueError("video stream requires a positive explicit time_base")
        return stream

    def metadata(self) -> VideoMetadata:
        """Read container headers without decoding the complete video."""
        import av

        with av.open(str(self.path), mode="r") as container:
            stream = self._video_stream(container)
            return VideoMetadata(
                path=self.path,
                participant_id=self.participant_id,
                clock_domain=self.clock_domain,
                codec=stream.codec_context.name,
                width=stream.codec_context.width,
                height=stream.codec_context.height,
                stream_index=stream.index,
                time_base=Fraction(stream.time_base),
                start_pts=stream.start_time,
                duration_pts=stream.duration,
                frame_count=stream.frames,
                average_rate=None if stream.average_rate is None else Fraction(stream.average_rate),
                container_duration_us=container.duration,
                video_stream_count=len(container.streams.video),
                audio_stream_count=len(container.streams.audio),
                provenance=self.provenance,
            )

    def nearest_frame(
        self,
        seconds: float | Decimal | Fraction,
        *,
        max_gap_seconds: float | Decimal | Fraction,
    ) -> VideoFrameMatch:
        """Find a frame near exact PTS seconds; do not convert from another clock.

        The caller supplies time in this video's own clock. Signed residuals are
        candidate minus query. Exactly-at-limit residuals are accepted. Invalid
        or absent frame PTS are rejected instead of inventing times from indices.
        """
        import av

        query = as_fraction(seconds, "video query seconds")
        maximum_gap = as_fraction(max_gap_seconds, "maximum video gap seconds")
        if maximum_gap < 0:
            raise ValueError("maximum video gap seconds must be nonnegative")
        best: VideoFrameSample | None = None
        best_distance: Fraction | None = None
        with av.open(str(self.path), mode="r") as container:
            stream = self._video_stream(container)
            seek_pts = query // Fraction(stream.time_base)
            if stream.start_time is not None:
                seek_pts = max(seek_pts, stream.start_time)
                if stream.duration is not None and stream.duration > 0:
                    seek_pts = min(seek_pts, stream.start_time + stream.duration - 1)
            container.seek(seek_pts, stream=stream, backward=True, any_frame=False)
            previous_seconds: Fraction | None = None
            for frame in container.decode(stream):
                if frame.pts is None or frame.time_base is None or frame.time_base <= 0:
                    raise ValueError("decoded video frame has missing PTS or invalid time_base")
                sample = VideoFrameSample(
                    pts=frame.pts,
                    time_base=Fraction(frame.time_base),
                    clock_domain=self.clock_domain,
                    provenance=self.provenance,
                    frame=frame,
                )
                if previous_seconds is not None and sample.seconds < previous_seconds:
                    raise ValueError("decoded presentation timestamps are not nondecreasing")
                previous_seconds = sample.seconds
                distance = abs(sample.seconds - query)
                if best_distance is None or distance < best_distance:
                    best, best_distance = sample, distance
                if sample.seconds >= query:
                    break
        residual = None if best is None else best.seconds - query
        status = (
            MatchStatus.NO_SAMPLES
            if best is None
            else MatchStatus.MATCHED
            if best_distance <= maximum_gap
            else MatchStatus.GAP_EXCEEDED
        )
        return VideoFrameMatch(status, best, query, residual, maximum_gap)
