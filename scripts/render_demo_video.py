#!/usr/bin/env python3
"""Render a demo clip: both ego views side by side with live model meters underneath.

Meters come from HELD-OUT predictions (the model never saw this recording).
Gold bands on the scrolling strip are annotated handovers; the dotted line is the
onset. Output: outputs/paired_benchmark/demo/<rid8>_<start>s.mp4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
FPS = 30


def extract_frames(mp4: Path, start_s: float, dur_s: float, out_fps: int, size: int, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{start_s:.3f}", "-t", f"{dur_s:.3f}", "-i", str(mp4),
                    "-vf", f"fps={out_fps},scale={size}:{size}", str(out_dir / "%05d.png")], check=True)
    return sorted(out_dir.glob("*.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="outputs/paired_benchmark_v3_speech/A")
    ap.add_argument("--view", default="both")
    ap.add_argument("--recording", default="43276420-701f-4731-b9ab-bebc7fd14994")
    ap.add_argument("--start-s", type=float, default=None, help="default: 12 s before the densest handover cluster")
    ap.add_argument("--dur-s", type=float, default=40.0)
    ap.add_argument("--out-fps", type=int, default=10)
    ap.add_argument("--tasks", default="onset_within_5s,handover_active,ja_active")
    ap.add_argument("--normalize", default="rank", choices=["none", "rank"], help="rank: show each score as its percentile within this recording (labelled as such)")
    a = ap.parse_args()
    rid = a.recording; rid8 = rid[:8]
    proc = ROOT / "data/processed/comind" / rid; raw = ROOT / "data/raw/comind/recordings" / rid
    meta = json.load(open(proc / "kinematics_v0.json"))
    info = json.load(open(Path(a.result_dir) / "results.json"))["info"]
    tasks_all = info.get("tasks", ["handover_active", "onset_within_1s", "onset_within_2s", "ja_active"])
    probs = np.load(Path(a.result_dir) / f"preds_{a.view}_{rid8}.npz")["probs"].astype(np.float32)
    z = np.load(proc / "kinematics_v0.npz")
    show = [t for t in a.tasks.split(",") if t in tasks_all]
    if a.start_s is None:
        onsets = np.array([h["start_frame"] for h in meta["handovers"]]) / FPS
        best = max(onsets, key=lambda o: ((onsets >= o) & (onsets < o + a.dur_s)).sum()) if len(onsets) else 30.0
        a.start_s = max(0.0, best - 12.0)
    out_dir = ROOT / "outputs/paired_benchmark/demo"; tmp = out_dir / f"_tmp_{rid8}"
    frames = {}
    for role in ("leader", "helper"):
        frames[role] = extract_frames(raw / "mp4s" / f"{role}_trimmed_sync.mp4", a.start_s, a.dur_s, a.out_fps, 420, tmp / role)
    n = min(len(frames["leader"]), len(frames["helper"]))
    k = 15
    sm = np.stack([np.convolve(probs[:, tasks_all.index(t)], np.ones(k) / k, mode="same") for t in show], axis=1)
    if a.normalize == "rank":  # percentile of the score within this recording; rare-event probabilities are tiny in absolute terms
        sm = np.stack([np.argsort(np.argsort(sm[:, r])) / (len(sm) - 1) for r in range(sm.shape[1])], axis=1)
    unit = "percentile in this recording" if a.normalize == "rank" else "probability"
    labels = {"onset_within_5s": "handover coming (<5 s)", "onset_within_2s": "handover coming (<2 s)", "handover_active": "handover in progress", "ja_active": "joint attention"}
    colors = {"onset_within_5s": "#d95f02", "onset_within_2s": "#d95f02", "handover_active": "#1b9e77", "ja_active": "#7570b3"}
    png_dir = tmp / "composite"; png_dir.mkdir(exist_ok=True)
    t_axis = np.arange(len(probs)) / FPS
    for i in range(n):
        t = a.start_s + i / a.out_fps; fi = int(round(t * FPS))
        fig = plt.figure(figsize=(12.6, 8.2), dpi=80); fig.patch.set_facecolor("#111")
        gs = fig.add_gridspec(3, 2, height_ratios=[4.2, 1.0, 1.6], hspace=0.28, wspace=0.03, left=0.03, right=0.97, top=0.90, bottom=0.08)
        for j, role in enumerate(("leader", "helper")):
            ax = fig.add_subplot(gs[0, j]); ax.imshow(plt.imread(frames[role][i])); ax.set_axis_off()
            ax.set_title(f"{role} (ego view)", color="w", fontsize=13, pad=4)
        ax = fig.add_subplot(gs[1, :]); ax.set_facecolor("#111")
        for r, tname in enumerate(show):
            v = float(np.clip(sm[fi, r], 0, 1))
            ax.barh(r, 0.66, color="#333", height=0.6); ax.barh(r, 0.66 * v, color=colors[tname], height=0.6)
            ax.text(0.675, r, f"{labels[tname]}  {v:.2f}", va="center", color="w", fontsize=11)
        ax.set_xlim(0, 1)
        ax.text(0.0, -0.55, f"meters: {unit}, smoothed 0.5 s", color="#888", fontsize=8); ax.set_ylim(-0.6, len(show) - 0.4); ax.invert_yaxis(); ax.set_axis_off()
        ax = fig.add_subplot(gs[2, :]); ax.set_facecolor("#1a1a1a")
        lo, hi = t - 15, t + 5
        for h in meta["handovers"]:
            s0, s1 = h["start_frame"] / FPS, h["end_frame"] / FPS
            if s1 > lo and s0 < hi:
                ax.axvspan(s0, s1, color="gold", alpha=0.35, lw=0); ax.axvline(s0, color="gold", ls=":", lw=1.2)
        m = (t_axis >= lo) & (t_axis <= t)
        for r, tname in enumerate(show):
            ax.plot(t_axis[m], sm[m, r], color=colors[tname], lw=1.8)
        ax.axvline(t, color="w", lw=1); ax.set_xlim(lo, hi); ax.set_ylim(0, 1)
        ax.tick_params(colors="#bbb", labelsize=9); ax.set_xlabel("time in recording (s)   |   gold = annotated handover, dotted = onset   |   model sees only the past", color="#bbb", fontsize=10)
        for sp in ax.spines.values(): sp.set_color("#444")
        fig.text(0.5, 0.965, f"Duet v0 · CoMind recording {rid8} · held-out (never trained on) · hands + gaze + speech, no pixels", color="#ddd", ha="center", fontsize=12)
        fig.savefig(png_dir / f"{i:05d}.png"); plt.close(fig)
    out = out_dir / f"{rid8}_{int(a.start_s)}s_{a.view}.mp4"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-framerate", str(a.out_fps), "-i", str(png_dir / "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(out)], check=True)
    subprocess.run(["rm", "-rf", str(tmp)])
    print(out, f"{n / a.out_fps:.0f}s")


if __name__ == "__main__":
    main()
