"""Person identity tracking: stable ids across frames (and views when world3d exists).

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def track(ep: Episode) -> None:
    ep.set_status("track", "skipped", "not implemented yet")
