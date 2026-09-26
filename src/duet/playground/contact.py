"""Hand-object contact: which object is in which hand, per frame.

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def contact(ep: Episode) -> None:
    ep.set_status("contact", "skipped", "not implemented yet")
