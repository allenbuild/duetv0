"""Read-only API for the annotate stage: GET /api/episode/{name}/annotations (auto-mounted by server.py)."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

from .episode import Episode

router = APIRouter()
EPISODES = Path(__file__).resolve().parents[3] / "data/playground/episodes"


def _ep(name: str) -> Episode:
    d = EPISODES / name
    if not (d / "episode.json").exists():
        raise HTTPException(404, f"no episode {name}")
    return Episode.load(d)


def _read_json(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.load(open(p))
    except json.JSONDecodeError:
        return default


@router.get("/api/episode/{name}/annotations")
def annotations(name: str):
    """Merged keyframe annotations with absolute image URLs, plus the run log and stage status."""
    ep = _ep(name)
    d = ep.dir / "derived" / "annotate"
    items = _read_json(d / "annotations.json", None)
    out = {"episode": name, "available": isinstance(items, list), "status": ep.status.get("annotate"), "log": _read_json(d / "log.json", None),
           "common_start_s": ep.common_start_s, "common_end_s": ep.common_end_s, "proc_fps": ep.proc_fps,
           "streams": [{"name": s.name, "role": s.role, "person": s.person} for s in ep.streams], "items": []}
    if out["available"]:
        for it in items:
            it = dict(it)
            it["image_url"] = f"/episodes/{name}/{it.get('image', '')}"
            out["items"].append(it)
    return out
