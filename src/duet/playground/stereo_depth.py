"""Stereo depth from a ZED left/right pair without the ZED SDK (OpenCV SGBM), writes zed/<stream>/depth/*.png.

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def stereo_depth(ep: Episode) -> None:
    ep.set_status("stereo_depth", "skipped", "not implemented yet")
