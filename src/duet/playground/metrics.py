"""Operator metrics per session (handover count, latency, hold, idle fraction).

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def metrics(ep: Episode) -> None:
    ep.set_status("metrics", "skipped", "not implemented yet")
