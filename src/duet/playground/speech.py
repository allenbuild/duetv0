"""Speech transcription (faster-whisper) + speaker attribution across ego mics.

Stub: replaced by the real stage. Keeps the runner and UI working meanwhile.
"""
from __future__ import annotations

from .episode import Episode


def speech(ep: Episode) -> None:
    ep.set_status("speech", "skipped", "not implemented yet")
