#!/usr/bin/env python3
"""One figure: best model per view on the three tasks that matter, with the shuffled-partner control."""
from __future__ import annotations
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("outputs/paired_benchmark/figures_final"); OUT.mkdir(parents=True, exist_ok=True)
VIEWS = ["leader", "helper", "both_shuffled", "both"]
TICK = {"leader": "leader\nonly", "helper": "helper\nonly", "both_shuffled": "both\n(shuffled)", "both": "both\n(paired)"}
COL = {"leader": "#8da0cb", "helper": "#fc8d62", "both_shuffled": "#999999", "both": "#1b9e77"}

def tcn(run, task):
    pf = []
    for sub in ("A", "B"):
        p = Path(run) / sub / "results.json"
        if p.exists(): pf += json.load(open(p))["per_fold"]
    return {v: np.array([r["test"][task]["auroc"] for r in pf if r["view"] == v]) for v in VIEWS if any(r["view"] == v for r in pf)}

def gbdt(files, task):
    out = {}
    for f in files:
        r = json.load(open(f))["results"]
        out.update({v: (r[v][task]["auroc_fold_mean"], r[v][task]["auroc_fold_sd"]) for v in r if task in r[v] and np.isfinite(r[v][task]["auroc_fold_mean"])})
    return out

G = ["outputs/paired_benchmark/baseline_gbdt_speech.json", "outputs/paired_benchmark/baseline_gbdt_speech_h12.json", "outputs/paired_benchmark/baseline_gbdt_speech_shuffled.json"]

panels = [
    ("joint attention now", None, gbdt(G, "ja_active")),
    ("handover in progress", None, gbdt(G, "handover_active")),
    ("handover starts in <5 s", None, gbdt(G, "onset_within_5s")),
]
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (title, t, g) in zip(axes, panels):
    vals = {v: (a.mean(), a.std()) for v, a in t.items()} if t else g
    vs = [v for v in VIEWS if v in vals]
    ax.bar(range(len(vs)), [vals[v][0] for v in vs], yerr=[vals[v][1] for v in vs], color=[COL[v] for v in vs], capsize=3)
    ax.axhline(0.5, ls="--", c="k", lw=1, label="chance"); ax.set_xticks(range(len(vs))); ax.set_xticklabels([TICK[v] for v in vs], fontsize=9)
    ax.set_ylim(0.4, 0.8); ax.set_title(title, fontsize=11); ax.set_ylabel("AUROC (mean ± sd, 5 held-out folds)")
    for i, v in enumerate(vs): ax.text(i, vals[v][0] + vals[v][1] + 0.008, f"{vals[v][0]:.3f}", ha="center", fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
fig.suptitle("Best model (gradient-boosted trees on causal window features). CoMind, 44 recordings / 21 h / 248 handovers, recording-level CV.\nNo pixels: Aria hand tracking + gaze + transcript speech features. Shuffled = partner stream time-shifted ≥ 60 s (same inputs, interaction destroyed).", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "best_models.png", dpi=150); print(OUT / "best_models.png")
