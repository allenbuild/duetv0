#!/usr/bin/env python3
"""Figures for the paired-vs-single kinematic benchmark.

1. results bar chart (AP per task per view, with chance line)
2. held-out timeline for the demo recording: p(handover onset within 2 s) for
   each view, with annotated handovers shaded
3. paired ego frames from both videos at the best-anticipated handover
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
from duet.ml.paired_benchmark import CLS_TASKS, FPS, REG_TASK

VIEW_LABEL = {"leader": "leader only", "helper": "helper only", "both": "both (paired)", "both_shuffled": "both (partner shuffled)"}
TICK_LABEL = {"leader": "leader\nonly", "helper": "helper\nonly", "both": "both\n(paired)", "both_shuffled": "both\n(shuffled)"}
VIEW_COLOR = {"leader": "#8da0cb", "helper": "#fc8d62", "both": "#1b9e77", "both_shuffled": "#999999"}
TASK_LABEL = {"handover_active": "handover in progress", "onset_within_1s": "handover starts in <1 s", "onset_within_2s": "handover starts in <2 s",
              "onset_within_3s": "handover starts in <3 s", "onset_within_5s": "handover starts in <5 s", "ja_active": "joint attention now"}


def load_results(dirs):
    pooled, per_fold = {}, []
    for d in dirs:
        r = json.load(open(Path(d) / "results.json"))
        pooled.update(r["pooled"]); per_fold += r["per_fold"]; info = r["info"]
    return pooled, per_fold, info


def fig_results(pooled, per_fold, info, out: Path):
    """Per-fold mean AUROC (robust to per-fold calibration; chance = 0.5), plus TTH MAE."""
    views = [v for v in ("leader", "helper", "both_shuffled", "both") if v in pooled]
    fig, axes = plt.subplots(1, len(CLS_TASKS) + 1, figsize=(4.2 * (len(CLS_TASKS) + 1), 4.4))
    for ax, t in zip(axes, CLS_TASKS):
        au = [np.array([r["test"][t]["auroc"] for r in per_fold if r["view"] == v]) for v in views]
        ax.bar(range(len(views)), [a.mean() for a in au], yerr=[a.std() for a in au], color=[VIEW_COLOR[v] for v in views], capsize=3)
        ax.axhline(0.5, ls="--", c="k", lw=1, label="chance")
        ax.set_xticks(range(len(views))); ax.set_xticklabels([TICK_LABEL[v] for v in views], fontsize=8)
        ax.set_title(TASK_LABEL[t], fontsize=11); ax.set_ylabel("AUROC (mean ± sd over 5 held-out folds)")
        ax.set_ylim(0.4, max(0.75, max(a.mean() for a in au) + 0.1))
        for i, a in enumerate(au):
            ax.text(i, a.mean() + a.std() + 0.008, f"{a.mean():.3f}", ha="center", fontsize=8)
        ax.legend(fontsize=8, loc="upper left")
    ax = axes[-1]
    mae = [pooled[v][REG_TASK]["mae_s"] for v in views]
    ax.bar(range(len(views)), mae, color=[VIEW_COLOR[v] for v in views])
    ax.set_xticks(range(len(views))); ax.set_xticklabels([TICK_LABEL[v] for v in views], fontsize=8)
    ax.set_title("time-to-handover error (<3 s window)", fontsize=11); ax.set_ylabel("MAE (s), lower is better")
    ax.set_ylim(0, max(mae) * 1.25)
    for i, a in enumerate(mae):
        ax.text(i, a + 0.02 * max(mae), f"{a:.2f}", ha="center", fontsize=8)
    fig.suptitle(f"CoMind kinematics-only benchmark: {info['n_recordings']} recordings, {info['hours']:.1f} h, "
                 f"{info['handovers']} handovers, {info['folds']}-fold recording-level CV, causal TCN on hands+gaze (no pixels)", fontsize=11)
    fig.tight_layout(); fig.savefig(out / "benchmark_results.png", dpi=150); plt.close(fig)


def fig_timeline(pred_dir_map, proc_root: Path, rid: str, out: Path, task="onset_within_2s", window_s=None):
    z = np.load(proc_root / rid / "kinematics_v0.npz")
    meta = json.load(open(proc_root / rid / "kinematics_v0.json"))
    n = len(z["features"]); t = np.arange(n) / FPS
    ti = CLS_TASKS.index(task)
    fig, ax = plt.subplots(figsize=(16, 4.2))
    if task == "ja_active":
        ax.fill_between(t, 0, z["ja_active"], color="gold", alpha=0.35, lw=0, step="mid", label="annotated joint attention")
    else:
        for h in meta["handovers"]:
            ax.axvspan(h["start_frame"] / FPS, h["end_frame"] / FPS, color="gold", alpha=0.45, lw=0)
            ax.axvline(h["start_frame"] / FPS, color="goldenrod", lw=1)
    for v in ("leader", "helper", "both_shuffled", "both"):
        p = pred_dir_map.get(v)
        if p is None or not p.exists():
            continue
        probs = np.load(p)["probs"].astype(np.float32)[:, ti]
        k = 15  # 0.5 s smoothing for display only
        sm = np.convolve(probs, np.ones(k) / k, mode="same")
        ax.plot(t, sm, lw=1.2 if v == "both" else 0.8, color=VIEW_COLOR[v], label=VIEW_LABEL[v], alpha=0.95 if v == "both" else 0.8)
    ax.set_ylim(0, 1); ax.set_xlim(0, t[-1] if window_s is None else window_s[1]);
    if window_s: ax.set_xlim(*window_s)
    ax.set_xlabel("time in recording (s)"); ax.set_ylabel(f"p({TASK_LABEL[task]})")
    ax.set_title(f"Held-out recording {rid[:8]} (never seen in training). Gold = annotated {'joint attention' if task == 'ja_active' else 'handovers'}.")
    ax.legend(loc="upper right", ncol=4, fontsize=9)
    fig.tight_layout(); fig.savefig(out / f"timeline_{rid[:8]}_{task}{'_zoom' if window_s else ''}.png", dpi=140); plt.close(fig)


def fig_frames(raw_root: Path, proc_root: Path, rid: str, pred_both: Path, out: Path, task="onset_within_2s"):
    meta = json.load(open(proc_root / rid / "kinematics_v0.json"))
    probs = np.load(pred_both)["probs"].astype(np.float32)[:, CLS_TASKS.index(task)]
    # pick the handover whose preceding 2 s has the highest mean probability
    best = max(meta["handovers"], key=lambda h: probs[max(0, h["start_frame"] - 60): h["start_frame"]].mean())
    s = best["start_frame"]
    offsets = [-90, -60, -30, 0, 15]
    fig, axes = plt.subplots(2, len(offsets), figsize=(3.2 * len(offsets), 6.8))
    for j, off in enumerate(offsets):
        f = max(0, s + off)
        for i, role in enumerate(("leader", "helper")):
            mp4 = raw_root / "recordings" / rid / "mp4s" / f"{role}_trimmed_sync.mp4"
            png = out / f"_frame_{role}_{f}.png"
            if mp4.exists() and not png.exists():
                subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{f / FPS:.4f}", "-i", str(mp4), "-frames:v", "1", "-vf", "scale=352:352", str(png)])
            ax = axes[i, j]
            if png.exists():
                ax.imshow(plt.imread(png))
            ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title(f"t = {off / FPS:+.1f} s\np(onset<2s) = {probs[f]:.2f}", fontsize=10)
            if j == 0:
                ax.set_ylabel(role, fontsize=12)
    fig.suptitle(f"Handover {best['key']} ({best['object_l1']}, {'leader→helper' if best['flow_leader_to_helper'] else 'helper→leader'}, "
                 f"cue={'+'.join(best['initiation_type'])}); paired-view model, held-out recording", fontsize=11)
    fig.tight_layout(); fig.savefig(out / f"frames_{rid[:8]}_{best['key']}.png", dpi=130); plt.close(fig)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dirs", nargs="+", default=["outputs/paired_benchmark/A", "outputs/paired_benchmark/B"])
    ap.add_argument("--out", default="outputs/paired_benchmark/figures")
    ap.add_argument("--raw-root", default="data/raw/comind")
    ap.add_argument("--proc-root", default="data/processed/comind")
    ap.add_argument("--demo-recording", default="43276420-701f-4731-b9ab-bebc7fd14994")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pooled, per_fold, info = load_results(a.result_dirs)
    global CLS_TASKS
    CLS_TASKS = tuple(info.get("tasks", CLS_TASKS)); pb.set_tasks(CLS_TASKS)
    fig_results(pooled, per_fold, info, out)
    rid = a.demo_recording
    preds = {}
    for d in a.result_dirs:
        for p in Path(d).glob(f"preds_*_{rid[:8]}.npz"):
            preds[p.name.split("preds_")[1].rsplit("_", 1)[0]] = p
    if preds:
        fig_timeline(preds, Path(a.proc_root), rid, out)
        fig_timeline(preds, Path(a.proc_root), rid, out, task="ja_active")
        meta = json.load(open(Path(a.proc_root) / rid / "kinematics_v0.json"))
        if meta["handovers"]:
            mid = np.median([h["start_frame"] for h in meta["handovers"]]) / FPS
            fig_timeline(preds, Path(a.proc_root), rid, out, window_s=(max(0, mid - 90), mid + 90))
        if "both" in preds:
            best = fig_frames(Path(a.raw_root), Path(a.proc_root), rid, preds["both"], out)
            print("frames figure for handover", best["key"])
    print("figures in", out)


if __name__ == "__main__":
    main()
