"""Tiny MP4 clips test exact cached-PTS decoding without dataset assets."""

from fractions import Fraction

import av
import numpy as np
import pytest

from duet.adapters.comind.indexed_video import iter_indexed_rgb


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "tiny.mp4"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width = stream.height = 16
        stream.pix_fmt = "yuv420p"
        for index in range(5):
            image = av.VideoFrame.from_ndarray(
                np.full((16, 16, 3), index * 40, dtype=np.uint8), format="rgb24"
            )
            image.pts = index
            image.time_base = Fraction(1, 10)
            for packet in stream.encode(image):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    with av.open(str(path)) as container:
        time_base = Fraction(container.streams.video[0].time_base)
        pts = np.array([image.pts for image in container.decode(video=0)], dtype=np.int64)
    return path, pts, time_base


def test_one_seek_clip_retains_exact_indices(clip):
    path, pts, time_base = clip
    frames = list(iter_indexed_rgb(path, pts, time_base, start_index=2, end_index=4))
    assert [index for index, _ in frames] == [2, 3, 4]
    assert frames[0][1].shape == (16, 16, 3)
    assert abs(float(frames[0][1].mean()) - 80) < 4


def test_altered_pts_table_rejected(clip):
    path, pts, time_base = clip
    pts[3] += 1
    with pytest.raises(ValueError, match="cached ordinal/PTS"):
        list(iter_indexed_rgb(path, pts, time_base, start_index=2, end_index=4))


def test_timebase_mismatch_rejected(clip):
    path, pts, time_base = clip
    with pytest.raises(ValueError, match="headers disagree"):
        list(iter_indexed_rgb(path, pts, time_base * 2, start_index=0, end_index=1))


@pytest.mark.parametrize("start,end", [(-1, 1), (2, 1), (0, 5), (True, 1)])
def test_invalid_clip_bounds(clip, start, end):
    path, pts, time_base = clip
    with pytest.raises(ValueError, match="indices"):
        list(iter_indexed_rgb(path, pts, time_base, start_index=start, end_index=end))


def test_duplicate_presentation_time_rejected(clip):
    path, pts, time_base = clip
    pts[1] = pts[0]
    with pytest.raises(ValueError, match="strictly increasing"):
        list(iter_indexed_rgb(path, pts, time_base, start_index=0, end_index=1))
