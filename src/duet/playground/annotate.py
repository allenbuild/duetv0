"""Dense VLM annotation every 2 s (Claude API / claude CLI / dry backend).

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def annotate(ep: Episode) -> None:
    ep.set_status("annotate", "skipped", "not implemented yet")
