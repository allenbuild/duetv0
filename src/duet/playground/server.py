"""FastAPI server for the ego/exo playground: episode list, per-stream overlays, stage runner, static UI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .episode import Episode
from .run import ORDER, is_running, run_in_background

ROOT = Path(__file__).resolve().parents[3]
EPISODES = ROOT / "data/playground/episodes"
STATIC = Path(__file__).with_name("static")
app = FastAPI(title="Duet ego/exo playground")


def _ep(name: str) -> Episode:
    d = EPISODES / name
    if not (d / "episode.json").exists():
        raise HTTPException(404, f"no episode {name}")
    return Episode.load(d)


def _clean(x):
    """numpy -> json with NaN -> null."""
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, list):
        return [_clean(v) for v in x]
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    if isinstance(x, (np.floating, np.integer)):
        return _clean(float(x))
    return x


@app.get("/api/episodes")
def episodes():
    out = []
    for d in sorted(EPISODES.glob("*/episode.json")):
        ep = Episode.load(d.parent)
        out.append({"name": ep.name, "streams": [{"name": s.name, "role": s.role} for s in ep.streams],
                    "duration_s": ep.common_end_s - ep.common_start_s, "status": ep.status, "running": is_running(d.parent)})
    return out


@app.get("/api/episode/{name}")
def episode(name: str):
    ep = _ep(name)
    d = json.load(open(ep.dir / "episode.json")); d["running"] = is_running(ep.dir); d["stages"] = ORDER
    qc = ep.derived / "qc" / "report.json"
    d["qc"] = json.load(open(qc)) if qc.exists() else None
    return d


@app.post("/api/episode/{name}/run")
def run(name: str, stages: str = "", force: bool = False):
    ep = _ep(name)
    ok = run_in_background(ep.dir, stages.split(",") if stages else None, force)
    return {"started": ok}


@app.get("/api/episode/{name}/video/{stream}")
def video(name: str, stream: str):
    ep = _ep(name); s = ep.stream(stream)
    return FileResponse(ep.dir / s.path, media_type="video/mp4")


@app.get("/api/episode/{name}/overlay/{stream}")
def overlay(name: str, stream: str):
    """Everything drawn on one view: body2d, hands, objects, frame size, timing."""
    ep = _ep(name); s = ep.stream(stream); out = {"proc_fps": ep.proc_fps, "common_start_s": ep.common_start_s, "offset_s": s.offset_s}
    z = ep.derived / "body2d" / f"{s.name}.npz"
    if z.exists():
        b = np.load(z); out["body2d"] = _clean(np.round(b["kpts"], 1)); out["img_w"] = int(b["img_w"]); out["img_h"] = int(b["img_h"])
    z = ep.derived / "hands" / f"{s.name}.npz"
    if z.exists():
        h = np.load(z); out["hands2d"] = _clean(np.round(h["lm2d"], 1)); out["hands3d"] = _clean(np.round(h["lm3d"], 4))
    z = ep.derived / "objects" / f"{s.name}.npz"
    if z.exists():
        o = np.load(z); out["objects"] = _clean(np.round(o["boxes"], 1)); out["object_names"] = o["names"].tolist()
    z = ep.derived / "qc" / f"{s.name}.npz"
    if z.exists():
        q = np.load(z); out["qc"] = {k: _clean(np.round(q[k], 2)) for k in ("sharp", "bright", "motion")}
    return JSONResponse(out)


@app.get("/api/episode/{name}/body3d")
def body3d(name: str):
    ep = _ep(name); z = ep.derived / "body3d" / "body3d.npz"
    if not z.exists():
        return {"available": False}
    b = np.load(z)
    return JSONResponse({"available": True, "stream": str(b["stream"]), "proc_fps": ep.proc_fps, "world": _clean(np.round(b["world"], 4))})


@app.get("/api/episode/{name}/imu_arm/{stream}")
def imu_arm(name: str, stream: str):
    ep = _ep(name); z = ep.derived / "imu_arm" / f"{stream}.npz"
    if not z.exists():
        return {"available": False}
    a = np.load(z)
    return JSONResponse({"available": True, "t_s": _clean(np.round(a["t_s"], 3)), "points": _clean(np.round(a["points"], 4)), "elbow_flex_deg": _clean(np.round(a["elbow_flex_deg"], 1))})


# videos are served from a static mount because browsers need HTTP Range support to seek (FileResponse lacks it)
EPISODES.mkdir(parents=True, exist_ok=True)
app.mount("/episodes", StaticFiles(directory=str(EPISODES), follow_symlink=True), name="episodes")
app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
