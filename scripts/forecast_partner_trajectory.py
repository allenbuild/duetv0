#!/usr/bin/env python3
"""Partner-trajectory forecasting: does seeing the partner improve prediction of a person's future hand motion?

Lawrence: "predict my collaborator's future action trajectory". Kaichen: "the most important thing is the
pose of the human beings". This is the closest proxy we can run to a policy-learning signal without a robot.

Target: for the TARGET person, both wrist positions H frames ahead (0.5 s and 1.0 s), expressed as the
displacement from the current wrist position in the target person's device frame (metres). Frames where a
wrist is missing now or at the horizon are excluded.

Inputs (views): target's own kinematics only; target + partner (paired); target + partner time-shuffled.
Baselines: zero-motion (hand stays), constant velocity (extrapolate last 5 frames).

Metric: mean Euclidean displacement error (m) at each horizon, over all valid frames and within ±3 s of an
annotated handover onset (where the partner should matter most). Recording-level 5-fold CV as everywhere.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.adapters.comind.kinematics import HAND_DIM, PERSON_DIM, person_slice  # noqa: E402
from duet.ml.paired_benchmark import CausalTCN, Normalizer, add_deltas, load_recordings, FPS  # noqa: E402

HORIZONS = (15, 30)  # frames: 0.5 s, 1.0 s
DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def wrists(x, role):
    s = person_slice(role)
    w = np.stack([x[:, s.start + h * HAND_DIM + 15: s.start + h * HAND_DIM + 18] for h in range(2)], axis=1)  # [N,2,3]
    v = np.stack([x[:, s.start + h * HAND_DIM + 25] > 0 for h in range(2)], axis=1)
    return w, v


def make_targets(x, role):
    """[N, 2*3*len(H)] displacement targets and [N, 2*len(H)] validity."""
    w, v = wrists(x, role)
    n = len(x)
    T = np.zeros((n, 2, len(HORIZONS), 3), np.float32); M = np.zeros((n, 2, len(HORIZONS)), bool)
    for hi, H in enumerate(HORIZONS):
        T[: n - H, :, hi] = w[H:] - w[: n - H]
        M[: n - H, :, hi] = v[H:] & v[: n - H]
    return T.reshape(n, -1), M.reshape(n, -1)


def const_velocity(x, role, k=5):
    w, v = wrists(x, role); n = len(x)
    vel = np.zeros_like(w); vel[k:] = (w[k:] - w[:-k]) / k
    ok = np.ones((n, 2), bool); ok[k:] = v[k:] & v[:-k]
    P = np.zeros((n, 2, len(HORIZONS), 3), np.float32)
    for hi, H in enumerate(HORIZONS):
        P[:, :, hi] = vel * H
    P[~ok] = 0
    return P.reshape(n, -1)


def inputs(x, target, view, rng, speech=None):
    me = x[:, person_slice(target)]
    other = "helper" if target == "leader" else "leader"
    partner = x[:, person_slice(other)]
    if speech is not None:
        half = speech.shape[1] // 2
        me = np.concatenate([me, speech[:, :half] if target == "leader" else speech[:, half:]], axis=1)
        partner = np.concatenate([partner, speech[:, half:] if target == "leader" else speech[:, :half]], axis=1)
    if view == "self":
        return me
    if view == "paired":
        return np.concatenate([me, partner], axis=1)
    if view == "shuffled":
        n = len(x); shift = int(rng.integers(60 * FPS, max(60 * FPS + 1, n - 60 * FPS)))
        return np.concatenate([me, np.roll(partner, shift, axis=0)], axis=1)
    raise ValueError(view)


class Forecaster(nn.Module):
    def __init__(self, d_in, d_out, c=64):
        super().__init__()
        self.tcn = CausalTCN(d_in, c=c, n_cls=d_out, dropout=0.2)

    def forward(self, x):
        return self.tcn(add_deltas(x))[0]


def near_handover_mask(rec, pad=3 * FPS):
    m = np.zeros(len(rec.x), bool)
    for o in rec.handover_onsets:
        m[max(0, o - pad): min(len(m), o + pad)] = True
    return m


def train_eval(train, test, target, view, seed, steps, speech, log):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    Xtr = {r.rid: inputs(r.x, target, view, rng, r.s if speech else None) for r in train}
    norm = Normalizer(list(Xtr.values()))
    Xtr = {k: norm(v).astype(np.float32) for k, v in Xtr.items()}
    Y = {r.rid: make_targets(r.x, target) for r in train + test}
    d_in = 2 * next(iter(Xtr.values())).shape[1]; d_out = Y[train[0].rid][0].shape[1]
    model = Forecaster(d_in, d_out).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    crop, B = 384, 48
    t0 = time.time()
    for step in range(1, steps + 1):
        model.train()
        xs, ys, ms = [], [], []
        for _ in range(B):
            r = train[rng.integers(len(train))]; n = len(r.x); s = int(rng.integers(0, n - crop))
            xs.append(Xtr[r.rid][s: s + crop]); ys.append(Y[r.rid][0][s: s + crop]); ms.append(Y[r.rid][1][s: s + crop])
        x = torch.from_numpy(np.stack(xs)).to(DEV); y = torch.from_numpy(np.stack(ys)).to(DEV); m = torch.from_numpy(np.stack(ms)).to(DEV)
        m[:, : 2 * FPS] = False
        pred = model(x)
        m3 = m.repeat_interleave(3, dim=-1)
        loss = F.smooth_l1_loss(pred[m3], y[m3], beta=0.02) if m3.any() else pred.sum() * 0
        opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sched.step()
        if step % 500 == 0:
            log(f"  [{target} {view} s{seed}] step {step} loss {loss.item():.4f} ({time.time() - t0:.0f}s)")
    # evaluate
    model.eval(); out = {"all": {}, "near_handover": {}}
    errs = {k: {H: [] for H in HORIZONS} for k in ("model", "zero", "cv")}
    errs_h = {k: {H: [] for H in HORIZONS} for k in ("model", "zero", "cv")}
    with torch.no_grad():
        for r in test:
            xin = norm(inputs(r.x, target, view, np.random.default_rng(321), r.s if speech else None)).astype(np.float32)
            pred = model(torch.from_numpy(xin[None]).to(DEV))[0].cpu().numpy()
            T, M = Y[r.rid]; cv = const_velocity(r.x, target)
            valid = M.copy(); valid[: 2 * FPS] = False
            nh = near_handover_mask(r)
            T = T.reshape(len(r.x), 2, len(HORIZONS), 3); P = pred.reshape(T.shape); C = cv.reshape(T.shape); V = valid.reshape(len(r.x), 2, len(HORIZONS))
            for hi, H in enumerate(HORIZONS):
                v = V[:, :, hi]
                em = np.linalg.norm(P[:, :, hi] - T[:, :, hi], axis=-1); ez = np.linalg.norm(T[:, :, hi], axis=-1); ec = np.linalg.norm(C[:, :, hi] - T[:, :, hi], axis=-1)
                errs["model"][H].append(em[v]); errs["zero"][H].append(ez[v]); errs["cv"][H].append(ec[v])
                vh = v & nh[:, None]
                errs_h["model"][H].append(em[vh]); errs_h["zero"][H].append(ez[vh]); errs_h["cv"][H].append(ec[vh])
    for name, E in (("all", errs), ("near_handover", errs_h)):
        for k in E:
            out[name][k] = {f"{H / FPS:.1f}s": float(np.concatenate(E[k][H]).mean()) for H in HORIZONS}
        out[name]["n_frames"] = int(sum(len(a) for a in E["model"][HORIZONS[0]]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--speech", action="store_true")
    ap.add_argument("--targets", default="helper,leader")
    ap.add_argument("--views", default="self,shuffled,paired")
    ap.add_argument("--out", default="outputs/paired_benchmark/forecast_partner_trajectory.json")
    a = ap.parse_args()
    src = (ROOT / "scripts/comind_download.py").read_text()
    ids = re.findall(r'"([0-9a-f-]{36})"', re.search(r"TRAIN_IDS\s*=\s*\[(.*?)\]", src, re.S).group(1))
    recs = load_recordings(ROOT / "data/processed/comind", ids)
    rng = np.random.default_rng(0); order = rng.permutation(len(recs)); folds = [sorted(order[i::a.folds].tolist()) for i in range(a.folds)]
    log_f = open(Path(a.out).with_suffix(".log"), "a")
    def log(s):
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()
    results = {"config": vars(a), "n_recordings": len(recs), "per_fold": []}
    for target in a.targets.split(","):
        for view in a.views.split(","):
            for k in range(a.folds):
                te = [recs[i] for i in folds[k]]; tr = [recs[i] for j in range(a.folds) if j != k for i in folds[j]]
                res = train_eval(tr, te, target, view, k, a.steps, a.speech, log)
                res.update({"target": target, "view": view, "fold": k}); results["per_fold"].append(res)
                log(f"FOLD target={target} view={view} k={k}: " + " ".join(f"{n}: model {r['model']['1.0s']:.4f} cv {r['cv']['1.0s']:.4f} zero {r['zero']['1.0s']:.4f} (1.0 s, m)" for n, r in res.items() if n != "target" and isinstance(r, dict)))
                json.dump(results, open(a.out, "w"), indent=1)
    # summary
    lines = ["| target | view | 0.5 s all | 1.0 s all | 0.5 s near handover | 1.0 s near handover |", "|---|---|---|---|---|---|"]
    for target in a.targets.split(","):
        for name in ("zero", "cv"):
            rs = [r for r in results["per_fold"] if r["target"] == target and r["view"] == a.views.split(",")[0]]
            lines.append(f"| {target} | baseline: {'hand stays still' if name == 'zero' else 'constant velocity'} | " + " | ".join(
                f"{np.mean([r[part][name][h] for r in rs]) * 100:.1f} cm" for part in ("all", "near_handover") for h in ("0.5s", "1.0s")) + " |")
        for view in a.views.split(","):
            rs = [r for r in results["per_fold"] if r["target"] == target and r["view"] == view]
            lines.append(f"| {target} | {view} | " + " | ".join(f"{np.mean([r[part]['model'][h] for r in rs]) * 100:.1f} ± {np.std([r[part]['model'][h] for r in rs]) * 100:.1f} cm"
                                                          for part in ("all", "near_handover") for h in ("0.5s", "1.0s")) + " |")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n"); log("\n".join(lines))


if __name__ == "__main__":
    main()
