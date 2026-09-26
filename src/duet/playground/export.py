"""Export an episode to the Duet per-frame record (one parquet row per common-timeline frame).

Columns (all on the common reference timeline at proc_fps):
  t_s
  <ego>_hand_{L,R}_{present,score}, <ego>_hand_{L,R}_lm3d (63 floats, MediaPipe world, m, wrist-relative)
  <ego>_hand_{L,R}_lm2d (42 floats, px in the extracted frame)
  <ego>_objects (json list of [name, conf, x0, y0, x1, y1])
  <stream>_body2d_p{0..3} (51 floats: 17 x (x, y, conf)) for every stream
  body3d_p{0,1} (99 floats: 33 x xyz, m, hip-centred; monocular) + body3d_stream
  <stream>_imu_arm (24 floats: 2 sides x 4 joints x xyz) where IMU exists
  <stream>_qc_{sharp,bright,motion}

This is the same "who is where, what do they hold, what do they look at" table the CoMind
benchmark trains on, minus gaze (no eye tracker on these rigs) and plus body pose.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .episode import Episode


def export(ep: Episode) -> None:
    ep.set_status("export", "running")
    n = min(len(list((ep.derived / "frames" / s.name).glob("*.jpg"))) for s in ep.streams)
    cols = {"t_s": ep.common_start_s + np.arange(n) / ep.proc_fps}
    for s in ep.streams:
        z = ep.derived / "body2d" / f"{s.name}.npz"
        if z.exists():
            k = np.load(z)["kpts"][:n]
            for p in range(k.shape[1]):
                cols[f"{s.name}_body2d_p{p}"] = list(k[:, p].reshape(n, -1))
        z = ep.derived / "hands" / f"{s.name}.npz"
        if z.exists():
            h = np.load(z)
            for i, side in enumerate("LR"):
                cols[f"{s.name}_hand_{side}_present"] = np.isfinite(h["lm2d"][:n, i, 0, 0])
                cols[f"{s.name}_hand_{side}_score"] = h["score"][:n, i]
                cols[f"{s.name}_hand_{side}_lm2d"] = list(h["lm2d"][:n, i].reshape(n, -1))
                cols[f"{s.name}_hand_{side}_lm3d"] = list(h["lm3d"][:n, i].reshape(n, -1))
        z = ep.derived / "objects" / f"{s.name}.npz"
        if z.exists():
            o = np.load(z); b, nm = o["boxes"][:n], o["names"][:n]
            cols[f"{s.name}_objects"] = [json.dumps([[str(nm[k, j]), round(float(b[k, j, 4]), 3), *[round(float(v), 1) for v in b[k, j, :4]]] for j in range(b.shape[1]) if np.isfinite(b[k, j, 4])]) for k in range(n)]
        z = ep.derived / "imu_arm" / f"{s.name}.npz"
        if z.exists():
            a = np.load(z); t = cols["t_s"]; idx = np.clip(np.searchsorted(a["t_s"], t), 0, len(a["t_s"]) - 1)
            ok = np.abs(a["t_s"][idx] - t) < 0.1
            pts = a["points"][idx].reshape(n, -1); pts[~ok] = np.nan
            cols[f"{s.name}_imu_arm"] = list(pts)
        z = ep.derived / "qc" / f"{s.name}.npz"
        if z.exists():
            q = np.load(z)
            for key in ("sharp", "bright", "motion"):
                cols[f"{s.name}_qc_{key}"] = q[key][:n]
    z = ep.derived / "world3d" / "world3d.npz"
    if z.exists():
        w3 = np.load(z)
        for p in range(w3["bodies"].shape[1]):
            cols[f"world_body_p{p}"] = list(w3["bodies"][:n, p].reshape(n, -1))
        for key in w3.files:
            if key.startswith(("hands3d_", "head_", "object_")):
                cols[f"world_{key}"] = list(w3[key][:n].reshape(n, -1))
            if key.startswith("feat_"):
                cols[key[5:]] = w3[key][:n]
    for s in ep.egos():
        z = ep.derived / "headpose" / f"{s.name}.npz"
        if z.exists():
            hp = np.load(z); cols[f"{s.name}_T_world_cam"] = list(hp["T_world_cam"][:n].reshape(n, -1)); cols[f"{s.name}_headpose_backend"] = [str(hp["backend"])] * n
    z = ep.derived / "body3d" / "body3d.npz"
    if z.exists():
        b = np.load(z); w = b["world"][:n]
        for p in range(w.shape[1]):
            cols[f"body3d_p{p}"] = list(w[:, p].reshape(n, -1))
        cols["body3d_stream"] = [str(b["stream"])] * n
    df = pd.DataFrame(cols)
    # any stage may contribute per-frame columns by writing derived/<stage>/records_extra.parquet
    # (one row per common-timeline frame, no t_s needed; extra rows are cut, missing rows padded with NaN)
    for extra in sorted(ep.derived.glob("*/records_extra.parquet")):
        try:
            ex = pd.read_parquet(extra).iloc[:n]
        except Exception as e:  # noqa: BLE001
            ep.notes.append(f"export: could not read {extra.name} from {extra.parent.name}: {e}"); continue
        ex = ex.reindex(range(n)); ex.columns = [c if c.startswith(extra.parent.name) else f"{extra.parent.name}_{c}" for c in ex.columns]
        df = pd.concat([df, ex.reset_index(drop=True)], axis=1)
    out = ep.derived / "records.parquet"; df.to_parquet(out, index=False)
    ep.set_status("export", "done", f"{len(df)} rows x {len(df.columns)} columns -> {out.relative_to(ep.dir)}")
