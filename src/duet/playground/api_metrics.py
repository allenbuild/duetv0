"""GET /api/episode/{name}/metrics: the operator metrics written by the metrics stage (derived/metrics/session.json).

Auto-mounted by server.py (any duet.playground.api_*.router). Must not import server.py.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .episode import Episode

EPISODES = Path(__file__).resolve().parents[3] / "data/playground/episodes"
router = APIRouter()


@router.get("/api/episode/{name}/metrics")
def get_metrics(name: str):
    d = EPISODES / name
    if not (d / "episode.json").exists():
        raise HTTPException(404, f"no episode {name}")
    ep = Episode.load(d); p = ep.derived / "metrics" / "session.json"
    if not p.exists():
        return {"available": False, "episode": name, "status": ep.status.get("metrics")}
    out = {"available": True, "status": ep.status.get("metrics"), **json.load(open(p)), "frames": {}}
    # per-frame boolean strips for the timeline: idle_<person>, hand_visible_<person> (metrics) and speaking_<person> (speech)
    for extra in (ep.derived / "metrics" / "records_extra.parquet", ep.derived / "speech" / "records_extra.parquet"):
        if extra.exists():
            try:
                import pandas as pd
                df = pd.read_parquet(extra)
                for c in df.columns:
                    key = c.replace("speech_", "", 1) if c.startswith("speech_") else c
                    if key.startswith(("idle_", "hand_visible_", "speaking_")):
                        out["frames"][key] = df[c].fillna(False).astype(bool).tolist()
            except Exception as e:  # noqa: BLE001
                out.setdefault("notes", []).append(f"could not read {extra.parent.name}/records_extra.parquet: {e}")
    return out
