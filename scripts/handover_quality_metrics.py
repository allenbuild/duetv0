#!/usr/bin/env python3
"""Collaboration-quality metrics and kinematic sub-stages for every annotated handover.

Sebastian asked for a collaboration metric before anything else; Lawrence asked for sub-stage
labels (approach / transfer / retract). Both are derived here from the kinematics, per handover:

  giver_reach_onset_s   first frame after the annotated start where the giver's wrist speed exceeds
                        its pre-onset baseline + 2 sd for >= 5 frames (approach begins)
  receiver_response_s   same for the receiver, measured from the giver's reach onset (reaction latency;
                        negative = receiver moved first, i.e. anticipated)
  transfer_s            frame of minimum giver-receiver wrist distance (shared world frame, where available),
                        else the giver's speed minimum after the reach peak (own frame)
  giver_hold_s          time the giver's wrist is extended and nearly still before transfer (waiting)
  duration_s            annotated end - start
  synchrony             max normalised cross-correlation between the two wrist-speed profiles, lag within ±2 s

Giver/receiver identity comes from `delivering_flow` (ltr = leader gives). Output: one CSV row per handover,
summary by cue type, and the stage boundaries as a per-frame label array (0 none, 1 approach, 2 transfer, 3 retract)
saved to kinematics-adjacent handover_stages_v0.npz for later training.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.adapters.comind.kinematics import HAND_DIM, person_slice
FPS = 30

def wrist_speed(f, role):
    s = person_slice(role); best = np.zeros(len(f)); anyv = np.zeros(len(f), bool)
    for h in range(2):
        b = s.start + h * HAND_DIM; w = f[:, b + 15: b + 18]; v = f[:, b + 25] > 0
        sp = np.linalg.norm(np.diff(w, axis=0, prepend=w[:1]), axis=1) * FPS; sp[~v] = 0
        best = np.maximum(best, sp); anyv |= v
    k = 5; return np.convolve(best, np.ones(k) / k, mode="same"), anyv

def onset_after(speed, start, base_lo, base_hi, min_run=5, max_ahead=4 * FPS):
    base = speed[max(0, base_lo): max(base_lo + 1, base_hi)]
    thr = base.mean() + 2 * base.std() + 0.05
    above = speed[start: start + max_ahead] > thr
    run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_run: return start + i - min_run + 1
    return None

rows = []
for j in sorted((ROOT / "data/processed/comind").glob("*/kinematics_v0.json")):
    m = json.load(open(j)); rid = m["recording_id"]; d = j.parent
    if not m["handovers"]: continue
    f = np.load(d / "kinematics_v0.npz")["features"]; n = len(f)
    world = np.load(d / "world_v0.npz")["world"] if (d / "world_v0.npz").exists() else None
    spd = {r: wrist_speed(f, r) for r in ("leader", "helper")}
    stages = np.zeros(n, np.int8)
    for h in m["handovers"]:
        s0, e0 = h["start_frame"], h["end_frame"]
        if s0 < 6 * FPS or e0 >= n: continue
        giver, recv = ("leader", "helper") if h["flow_leader_to_helper"] else ("helper", "leader")
        gs, rs = spd[giver][0], spd[recv][0]
        g_on = onset_after(gs, s0, s0 - 3 * FPS, s0)
        r_on = onset_after(rs, s0 - FPS, s0 - 4 * FPS, s0 - FPS, max_ahead=5 * FPS)  # may precede the giver's reach (anticipation)
        # transfer: min inter-wrist distance in shared world if available, else giver speed minimum after its peak
        if world is not None and world[s0: e0 + 1, 4].max() > 0:
            dmin = np.where(world[s0: e0 + 1, 4] > 0, world[s0: e0 + 1, 4], np.inf); t_tr = s0 + int(np.argmin(dmin)); d_tr = float(dmin.min())
        else:
            pk = s0 + int(np.argmax(gs[s0: e0 + 1])); t_tr = pk + int(np.argmin(gs[pk: e0 + 1])) if pk < e0 else pk; d_tr = np.nan
        # giver hold: frames between reach peak and transfer where giver speed < 0.1 m/s
        pk = s0 + int(np.argmax(gs[s0: e0 + 1]))
        hold = float((gs[pk: t_tr + 1] < 0.1).sum() / FPS) if t_tr > pk else 0.0
        # synchrony: max normalised cross-correlation of speed profiles within ±2 s
        a = gs[s0 - FPS: e0 + FPS]; b = rs[s0 - FPS: e0 + FPS]
        if a.std() > 1e-6 and b.std() > 1e-6:
            a = (a - a.mean()) / a.std(); b = (b - b.mean()) / b.std()
            lags = range(-2 * FPS, 2 * FPS + 1); cc = [np.mean(a[max(0, l): len(a) + min(0, l)] * b[max(0, -l): len(b) - max(0, l)]) for l in lags]
            sync = float(np.max(cc)); sync_lag = float(list(lags)[int(np.argmax(cc))] / FPS)
        else:
            sync, sync_lag = np.nan, np.nan
        rows.append({"recording": rid[:8], "key": h["key"], "giver": giver, "cue": "+".join(h["initiation_type"]), "object": h["object_l1"],
                     "duration_s": (e0 - s0) / FPS, "giver_reach_onset_s": None if g_on is None else (g_on - s0) / FPS,
                     "receiver_response_s": None if (g_on is None or r_on is None) else (r_on - g_on) / FPS,
                     "transfer_s": (t_tr - s0) / FPS, "min_wrist_dist_m": d_tr, "giver_hold_s": hold, "synchrony": sync, "synchrony_lag_s": sync_lag,
                     "has_world": world is not None})
        stages[s0: t_tr] = 1; stages[max(s0, t_tr - 5): min(e0 + 1, t_tr + 6)] = 2; stages[t_tr + 6: e0 + 1] = 3
    np.savez_compressed(d / "handover_stages_v0.npz", stages=stages)
df = pd.DataFrame(rows); out = ROOT / "outputs/paired_benchmark"; df.to_csv(out / "handover_quality_metrics.csv", index=False)
print(f"{len(df)} handovers, {df.recording.nunique()} recordings\n")
print("Overall (median [IQR]):")
for c in ["duration_s", "giver_reach_onset_s", "receiver_response_s", "transfer_s", "giver_hold_s", "synchrony", "synchrony_lag_s", "min_wrist_dist_m"]:
    v = pd.to_numeric(df[c], errors="coerce").dropna(); print(f"  {c:22s} n={len(v):3d}  {v.median():6.2f}  [{v.quantile(.25):5.2f}, {v.quantile(.75):5.2f}]")
print("\nBy cue type (median):")
g = df.copy(); g["receiver_response_s"] = pd.to_numeric(g["receiver_response_s"], errors="coerce")
print(g.groupby("cue")[["duration_s", "receiver_response_s", "giver_hold_s", "synchrony"]].median().round(2).assign(n=g.groupby("cue").size()).to_string())
print("\nBy giver:")
print(g.groupby("giver")[["duration_s", "receiver_response_s", "giver_hold_s", "synchrony"]].median().round(2).assign(n=g.groupby("giver").size()).to_string())
rr = g.receiver_response_s.dropna(); print(f"\nreceiver moved before the giver's reach (anticipation): {(rr < 0).mean():.0%} of {len(rr)} measurable handovers")
