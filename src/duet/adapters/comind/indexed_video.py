"""Bounded MP4 clip decoding against an independently cached ordinal/PTS table."""

from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike


def iter_indexed_rgb(
    path: str | Path,
    pts: ArrayLike,
    time_base: Fraction,
    *,
    start_index: int,
    end_index: int,
) -> Iterator[tuple[int, np.ndarray]]:
    """Seek once, then yield one RGB image at a time with its verified MP4 index.

    ``pts`` comes from sequential source decoding, never index divided by FPS.
    The inclusive end bound is a requested output window, not annotation semantics.
    Duplicate DEVICE_TIME mappings are allowed; duplicate video PTS are not.
    """
    import av

    values = np.asarray(pts)
    if values.ndim != 1 or values.dtype.kind not in "iu" or values.dtype.kind == "b":
        raise ValueError("original MP4 PTS must be a one-dimensional integer array")
    if np.any(values > np.iinfo(np.int64).max):
        raise ValueError("MP4 PTS exceed signed int64")
    values = values.astype(np.int64)
    if np.any(values[1:] <= values[:-1]):
        raise ValueError("source MP4 PTS must be strictly increasing")
    if not isinstance(time_base, Fraction) or time_base <= 0:
        raise ValueError("MP4 time_base must be an exact positive Fraction")
    if (
        type(start_index) is not int
        or type(end_index) is not int
        or not (0 <= start_index <= end_index < len(values))
    ):
        raise ValueError("invalid inclusive video clip indices")
    with av.open(str(path), mode="r") as container:
        if len(container.streams.video) != 1:
            raise ValueError("paired ego MP4 must have exactly one video stream")
        stream = container.streams.video[0]
        if stream.time_base != time_base or stream.frames != len(values):
            raise ValueError("video headers disagree with cached ordinal/PTS evidence")
        container.seek(int(values[start_index]), stream=stream, backward=True, any_frame=False)
        index = start_index
        for image in container.decode(stream):
            if image.pts is None:
                raise ValueError("decoded MP4 image has no original PTS")
            if image.pts < values[start_index]:
                continue
            if image.pts != values[index] or image.time_base != time_base:
                raise ValueError("decoded MP4 PTS disagree with cached ordinal/PTS evidence")
            yield index, image.to_ndarray(format="rgb24")
            if index == end_index:
                return
            index += 1
        raise ValueError("MP4 ended before the requested cached frame index")
