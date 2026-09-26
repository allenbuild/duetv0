"""FastAPI server for the ego/exo playground: episode list, per-stream overlays, stage runner, static UI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import re
import shutil

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .episode import Episode
from .run import ORDER, is_running, run_in_background
from . import zed_bundle

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


@app.post("/api/upload")
async def upload(name: str = Form(...), roles: str = Form(...), persons: str = Form(""), reference: str = Form(""), fps: float = Form(10.0),
                 files: list[UploadFile] = File(...)):
    """Create an episode from uploaded videos and start the pipeline.

    roles/persons are comma-separated, one per file, in file order. Stream names come from the
    filenames (sanitised). The first ego stream is the time reference unless `reference` is given.
    A .zip produced by scripts/zed_kit.py (ZED export: left.mp4 + calibration + tracking + depth) is
    accepted as a file too: it is unpacked to <episode>/zed/<stream>/ and its left.mp4 becomes the stream.
    """
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())[:64] or "episode"
    if (EPISODES / name).exists():
        raise HTTPException(409, f"episode {name} exists")
    role_list = [r.strip() for r in roles.split(",")]; person_list = [p.strip() or None for p in persons.split(",")] if persons else [None] * len(files)
    if len(role_list) != len(files):
        raise HTTPException(400, "one role per file")
    tmp = EPISODES / f"_upload_{name}"; tmp.mkdir(parents=True, exist_ok=True)
    videos = []
    for f, role, person in zip(files, role_list, person_list):
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(f.filename).stem)[:32] or "stream"
        dst = tmp / f"{stem}{Path(f.filename).suffix.lower() or '.mp4'}"
        with open(dst, "wb") as out:
            shutil.copyfileobj(f.file, out, 16 << 20)
        if dst.suffix == ".zip":
            if not zed_bundle.is_bundle(dst):
                shutil.rmtree(tmp, ignore_errors=True); raise HTTPException(400, f"{f.filename} is not a ZED export bundle")
            stem = stem[:-4] if stem.endswith("_zed") else stem
            dst = zed_bundle.ingest(dst, EPISODES / name, stem); role = role if role in ("ego", "exo") else "ego"
        videos.append((stem, role if role in ("ego", "exo") else "exo", dst, person))
    ep = Episode.create(EPISODES, name, videos, None, reference or None, link=False)
    ep.proc_fps = fps; ep.save(); shutil.rmtree(tmp, ignore_errors=True)
    run_in_background(ep.dir, None, False)
    return {"name": ep.name, "streams": [s.name for s in ep.streams]}


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


@app.get("/api/episode/{name}/world3d")
def world3d(name: str):
    ep = _ep(name); z = ep.derived / "world3d" / "world3d.npz"
    if not z.exists():
        return {"available": False}
    w = np.load(z); out = {"available": True, "proc_fps": ep.proc_fps, "bodies": _clean(np.round(w["bodies"], 4))}
    for key in w.files:
        if key.startswith(("head_", "object_", "hands3d_")):
            out[key] = _clean(np.round(w[key], 4))
        if key.startswith("feat_"):
            out[key] = _clean(np.round(w[key], 4))
    c = ep.derived / "calib" / "calib.json"
    if c.exists():
        out["cameras"] = {k: v["T_world_cam"] for k, v in json.load(open(c))["cameras"].items() if v["T_world_cam"] is not None}
    hp = {}
    for s in ep.egos():
        z2 = ep.derived / "headpose" / f"{s.name}.npz"
        if z2.exists():
            hp[s.name] = str(np.load(z2)["backend"])
    out["headpose_backends"] = hp
    return JSONResponse(out)


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
