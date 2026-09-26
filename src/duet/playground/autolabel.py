"""Proposed handover / joint-attention events for human review.

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def autolabel(ep: Episode) -> None:
    ep.set_status("autolabel", "skipped", "not implemented yet")
