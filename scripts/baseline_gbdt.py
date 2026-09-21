#!/usr/bin/env python3
"""Gradient-boosted-tree baseline on multi-scale causal window features (same CV folds as the TCN).

Rare-event anticipation is often better served by trees on engineered window statistics
than by a small sequence model trained end to end. Per person: wrist speeds, hand-hand
distance, hand height, gaze point, validity, speech flags; each summarised (last, mean,
max, min) over causal windows of 1 s, 3 s and 6 s. With --world, the cross-person block is
summarised the same way (paired view only; recordings restricted to those that have it).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.adapters.comind.kinematics import HAND_DIM, person_slice  # noqa: E402
from duet.ml.paired_benchmark import FPS, load_recordings  # noqa: E402

WINDOWS = (FPS, 3 * FPS)
STRIDE = 6  # evaluate every 6th frame (5 Hz) to keep the tree fit tractable


def causal_stats(x: np.ndarray, w: int) -> np.ndarray:
    """[N, D] -> [N, 3D]: mean, max, min over the trailing window of w frames (causal)."""
    pad = np.repeat(x[:1], w - 1, axis=0)
    xp = np.concatenate([pad, x], axis=0)
    v = sliding_window_view(xp, w, axis=0)  # [N, D, w]
    return np.concatenate([v.mean(-1), v.max(-1), v.min(-1)], axis=1)


def person_base(f: np.ndarray, role: str, speech: np.ndarray | None) -> np.ndarray:
    s = person_slice(role)
    cols, hands = [], []
    for h in range(2):
        b = s.start + h * HAND_DIM
        w = f[:, b + 15: b + 18]; v = f[:, b + 25] > 0
        sp = np.linalg.norm(np.diff(w, axis=0, prepend=w[:1]), axis=1) * FPS; sp[~v] = 0
        cols += [sp, v.astype(np.float32), w[:, 1] * v, w[:, 2] * v]  # speed, valid, height-ish, depth-ish
        hands.append((w, v))
    d = np.linalg.norm(hands[0][0] - hands[1][0], axis=1) * (hands[0][1] & hands[1][1])
    g = f[:, s.start + 2 * HAND_DIM: s.start + 2 * HAND_DIM + 3]
    cols += [d, g[:, 0], g[:, 1], g[:, 2], f[:, s.start + 2 * HAND_DIM + 6]]
    base = np.stack(cols, axis=1)
    if speech is not None:
        half = speech.shape[1] // 2
        blk = speech[:, :half] if role == "leader" else speech[:, half:]
        base = np.concatenate([base, blk[:, :4]], axis=1)  # speaking, since-last-word, word rate, request kw
    return base.astype(np.float32)


def featurize(base: np.ndarray) -> np.ndarray:
    base = np.nan_to_num(base, nan=0.0, posinf=0.0, neginf=0.0)
    return np.nan_to_num(np.concatenate([base] + [causal_stats(base, w) for w in WINDOWS], axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speech", action="store_true")
    ap.add_argument("--world", action="store_true", help="add cross-person block (paired view) and restrict to recordings that have it")
    ap.add_argument("--world-subset", action="store_true", help="restrict to recordings with world features without using them")
    ap.add_argument("--tasks", default="handover_active,onset_within_2s,onset_within_5s,ja_active")
    ap.add_argument("--views", default="leader,helper,both")
    ap.add_argument("--out", default="outputs/paired_benchmark/baseline_gbdt.json")
    ap.add_argument("--save-preds-dir", default=None, help="also write held-out per-frame probs as preds_<view>_<rid8>.npz (+ results.json info) for the demo renderer")
    a = ap.parse_args()
    tasks = a.tasks.split(",")
    src = (ROOT / "scripts/comind_download.py").read_text()
    train_ids = re.findall(r'"([0-9a-f-]{36})"', re.search(r"TRAIN_IDS\s*=\s*\[(.*?)\]", src, re.S).group(1))
    recs = load_recordings(ROOT / "data/processed/comind", train_ids)
    if a.world or a.world_subset:
        recs = [r for r in recs if r.w is not None]
    rng = np.random.default_rng(0); order = rng.permutation(len(recs)); K = 5
    folds = [sorted(order[i::K].tolist()) for i in range(K)]
    print(f"recordings {len(recs)}, handovers {sum(len(r.handover_onsets) for r in recs)}, speech={a.speech} world={a.world}", flush=True)
    X = {}
    for r in recs:
        sp = r.s if a.speech else None
        X[r.rid] = {"leader": featurize(person_base(r.x, "leader", sp)), "helper": featurize(person_base(r.x, "helper", sp))}
        if a.world and r.w is not None:
            X[r.rid]["world"] = featurize(r.w[:, :31].astype(np.float32))
    res = {}
    for view in a.views.split(","):
        pool = {t: ([], []) for t in tasks}; per_fold = {t: [] for t in tasks}
        for k in range(K):
            te = [recs[i] for i in folds[k]]; tr = [recs[i] for j in range(K) if j != k for i in folds[j]]

            def mk(rs, view=view):
                blocks = ("leader", "helper") if view in ("both", "both_shuffled") else (view,)
                xs = []
                for r in rs:
                    parts = [X[r.rid][b] for b in blocks]
                    if view == "both_shuffled":  # same control as the TCN: partner stream circularly shifted >= 60 s
                        n = len(parts[1]); shift = int(np.random.default_rng(hash(r.rid) % 2**32).integers(60 * FPS, max(60 * FPS + 1, n - 60 * FPS)))
                        parts[1] = np.roll(parts[1], shift, axis=0)
                    if view == "both" and "world" in X[r.rid]:
                        parts.append(X[r.rid]["world"])
                    xs.append(np.concatenate(parts, axis=1)[2 * FPS::STRIDE])
                return np.concatenate(xs), {t: np.concatenate([r.y[t][2 * FPS::STRIDE] for r in rs]) for t in tasks}

            Xtr, Ytr = mk(tr); Xte, Yte = mk(te)
            fold_probs = {}
            for t in tasks:
                pos = Ytr[t].mean()
                clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=200,
                                                     l2_regularization=1.0, class_weight={0: 1.0, 1: float(min(50, (1 - pos) / max(pos, 1e-4)))}, random_state=k)
                clf.fit(Xtr, Ytr[t]); p = np.nan_to_num(clf.predict_proba(Xte)[:, 1], nan=0.5)
                pool[t][0].append(Yte[t]); pool[t][1].append(p)
                fold_probs[t] = clf
                if 0 < Yte[t].sum() < len(Yte[t]):
                    per_fold[t].append(roc_auc_score(Yte[t], p))
            if a.save_preds_dir:
                out_dir = Path(a.save_preds_dir); out_dir.mkdir(parents=True, exist_ok=True)
                for r in te:
                    Xr, _ = mk([r])
                    full = np.zeros((len(r.x), len(tasks)), np.float32)
                    for ti, t in enumerate(tasks):
                        pr = fold_probs[t].predict_proba(Xr)[:, 1]
                        idx = np.arange(2 * FPS, len(r.x), STRIDE)[: len(pr)]
                        full[:, ti] = np.interp(np.arange(len(r.x)), idx, pr)
                    np.savez_compressed(out_dir / f"preds_{view}_{r.rid[:8]}.npz", probs=full.astype(np.float16))
                json.dump({"info": {"tasks": tasks, "n_recordings": len(recs), "model": "HistGradientBoosting on window features"}}, open(Path(a.save_preds_dir) / "results.json", "w"))
            print(f"  {view} fold {k} done", flush=True)
        res[view] = {t: {"ap_pooled": float(average_precision_score(np.concatenate(pool[t][0]), np.concatenate(pool[t][1]))),
                         "auroc_fold_mean": float(np.mean(per_fold[t])), "auroc_fold_sd": float(np.std(per_fold[t]))} for t in tasks}
        print(view, " | ".join(f"{t[:15]} AUROC {res[view][t]['auroc_fold_mean']:.3f}±{res[view][t]['auroc_fold_sd']:.3f} AP {res[view][t]['ap_pooled']:.3f}" for t in tasks), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": vars(a), "n_recordings": len(recs), "results": res}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
