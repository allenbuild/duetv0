"""Gaze proxy from head pose: forward ray, 'looking at partner/object' features.

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def gaze_proxy(ep: Episode) -> None:
    ep.set_status("gaze_proxy", "skipped", "not implemented yet")
