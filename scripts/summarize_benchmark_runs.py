#!/usr/bin/env python3
"""Consolidate every paired-benchmark run into one markdown table (per-fold mean±sd AUROC)."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np

RUNS = [
    ("v0 kinematics, GroupNorm, uniform crops (625K params)", "outputs/paired_benchmark"),
    ("v2 kinematics, causal LN, onset crops (165K)", "outputs/paired_benchmark_v2"),
    ("v3 + speech (hashed BoW), AUROC selection", "outputs/paired_benchmark_v3_speech"),
    ("v5 + speech, 5 s horizon", "outputs/paired_benchmark_v5_h5"),
    ("v6 + speech, time-to-completion head", "outputs/paired_benchmark_v6_tte"),
    ("v7 + speech (hashed + MiniLM embeddings)", "outputs/paired_benchmark_v7_emb"),
    ("v4 shared-world block, 11 labelled recs (world)", "outputs/paired_benchmark_v4_world/world"),
    ("v4 same 11 recs, no world block", "outputs/paired_benchmark_v4_world/noworld"),
]
TASKS = ["handover_active", "onset_within_1s", "onset_within_2s", "onset_within_5s", "ja_active", "scoia_active", "scoia_onset_within_2s", "scoia_onset_within_5s"]
VIEWS = ["leader", "helper", "both_shuffled", "both"]

def load(d):
    pf, info = [], None
    for sub in ("A", "B", ""):
        p = Path(d) / sub / "results.json"
        if p.exists():
            r = json.load(open(p)); pf += r["per_fold"]; info = r["info"]
    return pf, info

lines = ["| run | view | " + " | ".join(t.replace("scoia_onset_within_", "helper act<").replace("scoia_active", "helper act now").replace("onset_within_", "onset<").replace("handover_active", "handover now").replace("ja_active", "joint attn") for t in TASKS) + " |",
         "|---|---|" + "---|" * len(TASKS)]
for name, d in RUNS:
    pf, info = load(d)
    if not pf:
        continue
    for v in VIEWS:
        rs = [r for r in pf if r["view"] == v]
        if not rs:
            continue
        cells = []
        for t in TASKS:
            if t in rs[0]["test"]:
                a = np.array([r["test"][t]["auroc"] for r in rs]); cells.append(f"{a.mean():.3f} ± {a.std():.3f}")
            else:
                cells.append("")
        lines.append(f"| {name} ({info['n_recordings']} recs) | {v} | " + " | ".join(cells) + " |")
for f, label in [("outputs/paired_benchmark/baseline_handcrafted.json", "linear (logistic) on wrist-speed features"), ("outputs/paired_benchmark/baseline_gbdt_speech.json", "GBDT on window features + speech"),
                 ("outputs/paired_benchmark/baseline_gbdt_speech_h12.json", "GBDT on window features + speech"), ("outputs/paired_benchmark/baseline_gbdt_speech_shuffled.json", "GBDT on window features + speech"),
                 ("outputs/paired_benchmark/baseline_gbdt_grasp.json", "GBDT + speech + grasp-state proxies"), ("outputs/paired_benchmark/baseline_gbdt_scoia.json", "GBDT + speech (helper-action tasks)")]:
    p = Path(f)
    if not p.exists():
        continue
    r = json.load(open(p)); res = r.get("results", r)
    for v, tv in res.items():
        cells = []
        for t in TASKS:
            if t in tv:
                x = tv[t]; cells.append(f"{x.get('auroc_fold_mean', x.get('auroc', float('nan'))):.3f}" + (f" ± {x['auroc_fold_sd']:.3f}" if "auroc_fold_sd" in x else ""))
            else:
                cells.append("")
        lines.append(f"| {label} | {v} | " + " | ".join(cells) + " |")
out = "\n".join(lines)
Path("outputs/paired_benchmark/all_runs.md").write_text(out + "\n")
print(out)
