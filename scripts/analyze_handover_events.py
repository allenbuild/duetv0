#!/usr/bin/env python3
"""Event-level handover anticipation analysis from pooled held-out predictions.

Frame-level AP is hard to read. This converts p(onset_within_2s) into:
  * event-level detection: for every annotated handover, the peak probability in
    the 2 s before onset, vs. peak probability in random 2 s windows >= 5 s away
    from any handover (event AUROC, recall at fixed false-alarm rates)
  * lead time: at a threshold giving ~2 false alarms per hour of negatives, how
    many seconds before onset the paired model first fires
  * a breakdown by cue type (verbal / gestural / implicit) and flow direction
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

FPS = 30
TASK_IDX = None  # resolved from results.json "tasks" (falls back to the v0 order)


def smooth(p, k=9):
    return np.convolve(p, np.ones(k) / k, mode="same")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dirs", nargs="+", default=["outputs/paired_benchmark/A", "outputs/paired_benchmark/B"])
    ap.add_argument("--proc-root", default="data/processed/comind")
    ap.add_argument("--out", default="outputs/paired_benchmark/event_analysis.json")
    ap.add_argument("--task", default="onset_within_2s")
    a = ap.parse_args()
    info = json.load(open(Path(a.result_dirs[0]) / "results.json"))["info"]
    tasks = info.get("tasks", ["handover_active", "onset_within_1s", "onset_within_2s", "ja_active"])
    ti = tasks.index(a.task)
    preds = defaultdict(dict)  # view -> rid8 -> probs
    for d in a.result_dirs:
        for p in Path(d).glob("preds_*.npz"):
            view, rid8 = p.stem[len("preds_"):].rsplit("_", 1)
            preds[view][rid8] = smooth(np.load(p)["probs"].astype(np.float32)[:, ti])
    metas = {p.parent.name[:8]: json.load(open(p)) for p in Path(a.proc_root).glob("*/kinematics_v0.json")}
    rng = np.random.default_rng(0)
    report = {}
    for view, per_rec in sorted(preds.items()):
        pos, pos_meta, neg, lead_curves = [], [], [], []
        neg_hours = 0.0
        for rid8, p in per_rec.items():
            m = metas[rid8]; n = len(p)
            onsets = [h["start_frame"] for h in m["handovers"]]
            far = np.ones(n, bool)
            for h in m["handovers"]:
                far[max(0, h["start_frame"] - 5 * FPS): min(n, h["end_frame"] + 5 * FPS)] = False
            far[: 4 * FPS] = False
            for h in m["handovers"]:
                s = h["start_frame"]
                if s < 3 * FPS or s >= n:
                    continue
                pos.append(p[s - 2 * FPS: s].max()); pos_meta.append(h)
                lead_curves.append(p[s - 6 * FPS: s] if s >= 6 * FPS else np.pad(p[:s], (6 * FPS - s, 0)))
            # random negative windows, 2 s each, only where the whole window is far from handovers
            cand = np.flatnonzero(far[: n - 2 * FPS])
            cand = cand[far[cand + 2 * FPS - 1]]
            for s in rng.choice(cand, size=min(200, len(cand)), replace=False):
                neg.append(p[s: s + 2 * FPS].max())
            neg_hours += far.sum() / FPS / 3600
        pos, neg = np.array(pos), np.array(neg)
        y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
        auroc = roc_auc_score(y, np.r_[pos, neg])
        # recall at false-alarm rate (fraction of negative 2 s windows that would fire)
        rec_at = {}
        for far_rate in (0.01, 0.05, 0.10):
            thr = np.quantile(neg, 1 - far_rate)
            rec_at[f"recall@FAR{int(far_rate*100)}%"] = float((pos >= thr).mean())
            rec_at[f"thr@FAR{int(far_rate*100)}%"] = float(thr)
        thr = np.quantile(neg, 0.95)
        L = np.stack(lead_curves)  # [E, 180]
        first = np.array([(6 * FPS - np.argmax(c >= thr)) / FPS if (c >= thr).any() else np.nan for c in L])
        # breakdowns
        by = defaultdict(list)
        for h, pk in zip(pos_meta, pos):
            by["flow:" + ("leader→helper" if h["flow_leader_to_helper"] else "helper→leader")].append(pk >= thr)
            by["cue:" + "+".join(h["initiation_type"])].append(pk >= thr)
            by["init:" + ("leader" if h["initiator_is_leader"] else "helper")].append(pk >= thr)
        report[view] = {
            "n_events": len(pos), "n_neg_windows": len(neg),
            "event_auroc": float(auroc), **rec_at,
            "lead_time_s_at_FAR5%": {"median": float(np.nanmedian(first)), "mean": float(np.nanmean(first)),
                                      "detected_frac": float(np.isfinite(first).mean())},
            "mean_prob_curve_before_onset": [float(v) for v in L.mean(0)[::15]],  # every 0.5 s from -6 s to 0
            "recall_breakdown@FAR5%": {k: {"recall": float(np.mean(v)), "n": len(v)} for k, v in sorted(by.items())},
        }
    json.dump(report, open(a.out, "w"), indent=1)
    views = [v for v in ("leader", "helper", "both_shuffled", "both") if v in report]
    print(f"Event-level anticipation of handovers, task={a.task} (n={report[views[0]]['n_events']} held-out handovers)")
    print("| view | event AUROC | recall @1% FAR | recall @5% FAR | recall @10% FAR | median lead (s) @5% FAR |")
    print("|---|---|---|---|---|---|")
    for v in views:
        r = report[v]
        print(f"| {v} | {r['event_auroc']:.3f} | {r['recall@FAR1%']:.2f} | {r['recall@FAR5%']:.2f} | {r['recall@FAR10%']:.2f} | {r['lead_time_s_at_FAR5%']['median']:.2f} |")
    print("\nrecall @5% FAR by cue type / flow (both vs best single):")
    for k in sorted(report[views[-1]]["recall_breakdown@FAR5%"]):
        row = " ".join(f"{v}={report[v]['recall_breakdown@FAR5%'].get(k, {}).get('recall', float('nan')):.2f}" for v in views)
        print(f"  {k:24s} n={report[views[-1]]['recall_breakdown@FAR5%'][k]['n']:3d}  {row}")


if __name__ == "__main__":
    main()
