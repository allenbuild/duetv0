#!/usr/bin/env python3
"""Hand-crafted-feature logistic-regression floor for the paired benchmark (same CV folds as the TCN)."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.adapters.comind.kinematics import HAND_DIM, person_slice
from duet.ml.paired_benchmark import CLS_TASKS, FPS, load_recordings


def person_feats(f, role):
    """Per-frame causal summary: wrist speeds (L,R) now, max/mean over last 1 s, hand validity, gaze point z, hand-hand distance."""
    s = person_slice(role); out = []
    hands = []
    for h in range(2):
        b = s.start + h * HAND_DIM
        w = f[:, b + 15: b + 18]; v = f[:, b + 25] > 0
        sp = np.linalg.norm(np.diff(w, axis=0, prepend=w[:1]), axis=1) * FPS; sp[~v] = 0
        cs = np.cumsum(np.r_[0, sp]); mean1 = (cs[FPS:] - cs[:-FPS]) / FPS; mean1 = np.r_[np.zeros(FPS - 1), mean1]
        mx1 = np.maximum.accumulate(sp) * 0  # placeholder replaced below
        from numpy.lib.stride_tricks import sliding_window_view
        mx1 = np.r_[np.zeros(FPS - 1), sliding_window_view(sp, FPS).max(axis=1)]
        out += [sp, mean1, mx1, v.astype(float), w[:, 2] * v]
        hands.append((w, v))
    d = np.linalg.norm(hands[0][0] - hands[1][0], axis=1) * (hands[0][1] & hands[1][1])
    out += [d, f[:, s.start + 52 + 2], f[:, s.start + 52 + 6]]
    return np.stack(out, axis=1)

def main():
    src = (ROOT / "scripts/comind_download.py").read_text()
    train_ids = re.findall(r'"([0-9a-f-]{36})"', re.search(r"TRAIN_IDS\s*=\s*\[(.*?)\]", src, re.DOTALL).group(1))
    recs = load_recordings(ROOT / "data/processed/comind", train_ids)
    rng = np.random.default_rng(0); order = rng.permutation(len(recs)); K = 5
    folds = [sorted(order[i::K].tolist()) for i in range(K)]
    X = {r.rid: {"leader": person_feats(r.x, "leader"), "helper": person_feats(r.x, "helper")} for r in recs}
    res = {}
    for view in ("leader", "helper", "both"):
        pool = {t: ([], []) for t in CLS_TASKS}
        for k in range(K):
            te = [recs[i] for i in folds[k]]; tr = [recs[i] for j in range(K) if j != k for i in folds[j]]
            def mk(rs):
                xs = [np.concatenate([X[r.rid][v] for v in (("leader", "helper") if view == "both" else (view,))], axis=1)[2 * FPS::3] for r in rs]
                return np.concatenate(xs), {t: np.concatenate([r.y[t][2 * FPS::3] for r in rs]) for t in CLS_TASKS}
            Xtr, Ytr = mk(tr); Xte, Yte = mk(te)
            for t in CLS_TASKS:
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight="balanced", C=0.1))
                clf.fit(Xtr, Ytr[t]); p = clf.predict_proba(Xte)[:, 1]
                pool[t][0].append(Yte[t]); pool[t][1].append(p)
        res[view] = {t: {"ap": float(average_precision_score(np.concatenate(pool[t][0]), np.concatenate(pool[t][1]))),
                         "auroc": float(roc_auc_score(np.concatenate(pool[t][0]), np.concatenate(pool[t][1])))} for t in CLS_TASKS}
        print(view, " ".join(f"{t} AP={res[view][t]['ap']:.3f} AUROC={res[view][t]['auroc']:.3f}" for t in CLS_TASKS), flush=True)
    Path("outputs/paired_benchmark").mkdir(parents=True, exist_ok=True)
    json.dump(res, open("outputs/paired_benchmark/baseline_handcrafted.json", "w"), indent=1)

if __name__ == "__main__":
    main()
