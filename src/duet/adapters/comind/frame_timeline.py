"""Evidence-gated paired video frame indices with independent device clocks.

A frame index is a discrete CoMind synchronization key, not seconds and not a
device timestamp. The complete per-video DEVICE_TIME arrays supply the local
queries. No UTC, affine fit, array-index pose matching, or inferred offset is used.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
from types import MappingProxyType

from duet.adapters.comind.mp4_timestamps import Mp4DeviceTimestamps
from duet.adapters.comind.semantics import MappingEvidence
from duet.schemas.time import Timestamp
from duet.synchronization.matching import NearestTimestampMatch, nearest_in_timeline
from duet.synchronization.timeline import OrderedTimeline


@dataclass(frozen=True)
class SynchronizedVideoFrame:
    """One declared paired-video moment and its two original local timestamps."""

    timeline_name: str
    frame_index: int
    device_timestamps: Mapping[str, Timestamp]


@dataclass(frozen=True)
class PairedFrameTimeline:
    """Bind complete MP4/device arrays through verified paired-video alignment.

    Construction requires both arrays to have exactly one timestamp per video
    frame. Equal counts alone never verify alignment: a separate evidence record
    is mandatory. Duplicated device timestamps remain duplicated image captures.
    """

    helper: Mp4DeviceTimestamps
    leader: Mp4DeviceTimestamps
    alignment_evidence: MappingEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.alignment_evidence, MappingEvidence):
            raise TypeError("paired-video frame alignment requires explicit evidence")
        if self.alignment_evidence.name != "paired_video_frame_alignment":
            raise ValueError("evidence must specifically establish paired_video_frame_alignment")
        self.alignment_evidence.require_verified()
        for role in ("helper", "leader"):
            source = getattr(self, role)
            if not isinstance(source, Mp4DeviceTimestamps):
                raise TypeError("each video requires validated complete DEVICE_TIME timestamps")
            if source.participant != role:
                raise ValueError("video timestamp arrays belong to the wrong participants")
        if self.helper.recording_id != self.leader.recording_id:
            raise ValueError("paired videos must belong to the same recording")
        if self.helper.frame_count != self.leader.frame_count:
            raise ValueError("paired synchronized videos must have equal frame counts")

    @property
    def timeline_name(self) -> str:
        return f"comind/{self.helper.recording_id}/comind_sync_frame_index"

    @property
    def frame_count(self) -> int:
        return self.helper.frame_count

    def frame(self, frame_index: int) -> SynchronizedVideoFrame:
        """Select the same verified export index while retaining independent clocks."""
        timestamps = {
            role: getattr(self, role).timestamp_at(frame_index) for role in ("helper", "leader")
        }
        return SynchronizedVideoFrame(self.timeline_name, frame_index, MappingProxyType(timestamps))

    def match_local(
        self,
        frame_index: int,
        participant: str,
        stream: OrderedTimeline,
        *,
        max_gap_seconds: Fraction | float | Decimal,
    ) -> NearestTimestampMatch:
        """Match pose/hand timestamps in this participant's DEVICE_TIME only.

        Residuals refer to local sample time minus this video's original capture
        time. Returned sample indices refer to that source stream; they are never
        assumed to equal the synchronized video frame index.
        """
        if participant not in ("helper", "leader"):
            raise ValueError("participant must be helper or leader")
        query = getattr(self, participant).timestamp_at(frame_index)
        if stream.clock_domain != query.clock_domain:
            raise ValueError("source stream must use the same participant's device clock")
        return nearest_in_timeline(query, stream, max_gap_seconds=max_gap_seconds)
