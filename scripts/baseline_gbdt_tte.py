#!/usr/bin/env python3
"""Time-to-completion within an ongoing handover: GBDT regressor on window features, frames inside
[start, end] only (same CV folds as the TCN). Reports MAE vs predicting the constant median."""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baseline_gbdt import featurize, person_base  # noqa: E402
from duet.ml.paired_benchmark import load_recordings  # noqa: E402

src = (ROOT / "scripts/comind_download.py").read_text()
train_ids = re.findall(r'"([0-9a-f-]{36})"', re.search(r"TRAIN_IDS\s*=\s*\[(.*?)\]", src, re.S).group(1))
recs = load_recordings(ROOT / "data/processed/comind", train_ids)
rng = np.random.default_rng(0); order = rng.permutation(len(recs)); K = 5
folds = [sorted(order[i::K].tolist()) for i in range(K)]
X = {r.rid: {"leader": featurize(person_base(r.x, "leader", r.s)), "helper": featurize(person_base(r.x, "helper", r.s))} for r in recs}
res = {}
for view in ("leader", "helper", "both"):
    err, base, n = [], [], 0
    for k in range(K):
        te = [recs[i] for i in folds[k]]; tr = [recs[i] for j in range(K) if j != k for i in folds[j]]
        def mk(rs):
            xs, ys = [], []
            for r in rs:
                m = np.isfinite(r.y["tte_s"]) & (r.y["tte_s"] <= 8)
                blocks = ("leader", "helper") if view == "both" else (view,)
                xs.append(np.concatenate([X[r.rid][b] for b in blocks], axis=1)[m]); ys.append(r.y["tte_s"][m])
            return np.concatenate(xs), np.concatenate(ys)
        Xtr, ytr = mk(tr); Xte, yte = mk(te)
        reg = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=50, l2_regularization=1.0, random_state=k).fit(Xtr, ytr)
        p = np.clip(reg.predict(Xte), 0, 8)
        err.append(np.abs(p - yte)); base.append(np.abs(np.median(ytr) - yte)); n += len(yte)
    e, b = np.concatenate(err), np.concatenate(base)
    res[view] = {"mae_s": float(e.mean()), "baseline_mae_s": float(b.mean()), "n_frames": int(n)}
    print(f"{view:8s} time-to-completion MAE {e.mean():.2f} s  (constant-median baseline {b.mean():.2f} s, n={n} frames)", flush=True)
json.dump(res, open("outputs/paired_benchmark/baseline_gbdt_tte.json", "w"), indent=1)
