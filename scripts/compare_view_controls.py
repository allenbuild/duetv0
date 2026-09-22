#!/usr/bin/env python3
"""Paired-vs-single with the controls that separate 'more input' from 'interaction'.

The published benchmark compares ``both`` against ``both_shuffled`` (partner circularly
shifted >= 60 s). That control destroys the rolled stream's alignment with the partner AND
with the labels, so the shuffled model's best strategy is to ignore the rolled person; its
expected score is the un-rolled person's single-view score. It therefore rules out "the
gain is just more input columns" but cannot distinguish

  (a) two independently informative streams being combined, from
  (b) the temporal correspondence between them carrying information neither has alone.

Only (b) is an interaction claim. This script adds the two controls that separate them:

  late_fusion_prob : mean of the leader-only and helper-only held-out probabilities
  late_fusion_rank : mean of their within-fold ranks (calibration-free version of the same)

Late fusion has both streams' marginal information and no access to their correspondence.
``both`` > late fusion is evidence for (b); ``both`` ~= late fusion is (a).

  leader_shuffled  : the mirror of both_shuffled (leader rolled, helper aligned), so the
                     control is not always applied to whichever partner happens to matter.

Statistics. Views share folds and recordings, so every comparison is paired:

  per-fold  : difference within each fold, mean +- sd, and folds-won (5 units, matches
              how the design docs report, but underpowered)
  bootstrap : per-recording AUROC, then a paired bootstrap over recordings (up to 44
              units); reports the 95% interval of the mean difference and P(diff > 0)

Model: HistGradientBoosting on causal window features, identical to baseline_gbdt.py
(the best model in the v1/v2 tables). --seeds runs several independent fold partitions.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from baseline_gbdt import STRIDE, featurize, person_base  # noqa: E402
from duet.ml.paired_benchmark import FPS, load_recordings, shuffle_shift  # noqa: E402

SINGLE = ("leader", "helper")
FUSIONS = ("late_fusion_prob", "late_fusion_rank")


def fit_views(recs, tasks, views, seed, folds_k, speech, log):
    """Train every view on shared folds. Returns per-recording held-out probabilities.

    out[view][task][rid] -> probs on the strided evaluation frames
    y[task][rid]         -> labels on the same frames
    fold_of[rid]         -> which fold held this recording out
    """
    t0 = time.time()
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(recs))
    folds = [sorted(order[i::folds_k].tolist()) for i in range(folds_k)]
    fold_of = {recs[i].rid: k for k in range(folds_k) for i in folds[k]}

    # Features are strided to the evaluation rate up front: the design matrices are only
    # ever used at that rate, and holding them at 30 Hz costs 6x the memory for nothing.
    # Labels are strided identically, so rows correspond by construction.
    X = {}
    for r in recs:
        sp = r.s if speech else None
        X[r.rid] = {role: featurize(person_base(r.x, role, sp))[2 * FPS::STRIDE] for role in SINGLE}

    y = {t: {r.rid: r.y[t][2 * FPS::STRIDE] for r in recs} for t in tasks}
    out = {v: {t: {} for t in tasks} for v in views}

    def design(r, view):
        if view in SINGLE:
            return X[r.rid][view]
        L, H = X[r.rid]["leader"], X[r.rid]["helper"]
        if view == "both":
            return np.concatenate([L, H], axis=1)
        # shuffled controls: roll one block by a deterministic per-recording shift.
        # Rows are strided, so the frame-domain shift is converted to strided units.
        shift = shuffle_shift(len(r.x), r.rid, None) // STRIDE
        if view == "both_shuffled":
            return np.concatenate([L, np.roll(H, shift, axis=0)], axis=1)
        if view == "leader_shuffled":
            return np.concatenate([np.roll(L, shift, axis=0), H], axis=1)
        raise ValueError(view)

    # Design matrices are built lazily, one recording at a time. Materialising all 44 at once
    # alongside X costs an extra ~0.5 GB per paired view, which pushed a 16 GB machine into
    # swap and slowed each fold by 5x. Rebuilding per fold is cheap by comparison.
    def stack_design(rs, view):
        """Concatenate designs without holding every recording's block at once.

        np.concatenate([...]) would materialise all N blocks plus the result; filling a
        preallocated buffer keeps peak memory at result + one block.
        """
        first = design(rs[0], view)
        rows = sum(len(X[r.rid]["leader"]) for r in rs)
        buf = np.empty((rows, first.shape[1]), np.float32)
        at = 0
        for r in rs:
            blk = design(r, view)
            buf[at:at + len(blk)] = blk
            at += len(blk)
        assert at == rows, (at, rows)
        return buf

    for view in views:
        for k in range(folds_k):
            te = [recs[i] for i in folds[k]]
            tr = [recs[i] for j in range(folds_k) if j != k for i in folds[j]]
            Xtr = stack_design(tr, view)
            for t in tasks:
                ytr = np.concatenate([y[t][r.rid] for r in tr])
                pos = ytr.mean()
                clf = HistGradientBoostingClassifier(
                    max_iter=100, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=200,
                    l2_regularization=1.0, random_state=k,
                    class_weight={0: 1.0, 1: float(min(50, (1 - pos) / max(pos, 1e-4)))})
                clf.fit(Xtr, ytr)
                for r in te:
                    out[view][t][r.rid] = np.nan_to_num(clf.predict_proba(design(r, view))[:, 1], nan=0.5)
            del Xtr
            log(f"    seed {seed} {view} fold {k} ({time.time() - t0:.0f}s elapsed)")
        log(f"  seed {seed}: {view} done ({time.time() - t0:.0f}s elapsed)")
    return out, y, fold_of


def add_fusions(out, y, fold_of, tasks, folds_k):
    """Combine the two single-view held-out probabilities: mean prob and mean within-fold rank."""
    for t in tasks:
        for rid in y[t]:
            a, b = out["leader"][t][rid], out["helper"][t][rid]
            out["late_fusion_prob"][t][rid] = 0.5 * (a + b)
        for k in range(folds_k):  # ranks are only comparable within one fold's model pair
            rids = [r for r in y[t] if fold_of[r] == k]
            if not rids:
                continue
            lens = [len(y[t][r]) for r in rids]
            cuts = np.cumsum(lens)[:-1]
            ra = rankdata(np.concatenate([out["leader"][t][r] for r in rids]))
            rb = rankdata(np.concatenate([out["helper"][t][r] for r in rids]))
            fused = 0.5 * (ra + rb)
            for r, part in zip(rids, np.split(fused, cuts)):
                out["late_fusion_rank"][t][r] = part
    return out


def per_recording_auroc(out, y, view, task):
    """AUROC per recording, skipping recordings without both classes present."""
    res = {}
    for rid, yy in y[task].items():
        if 0 < yy.sum() < len(yy):
            res[rid] = roc_auc_score(yy, out[view][task][rid])
    return res


def per_fold_auroc(out, y, fold_of, view, task, folds_k):
    """AUROC pooled within each held-out fold (what the design docs' mean +- sd reports)."""
    vals = []
    for k in range(folds_k):
        rids = [r for r in y[task] if fold_of[r] == k]
        yy = np.concatenate([y[task][r] for r in rids])
        pp = np.concatenate([out[view][task][r] for r in rids])
        vals.append(roc_auc_score(yy, pp) if 0 < yy.sum() < len(yy) else np.nan)
    return np.array(vals)


def paired_bootstrap(a: dict, b: dict, n_boot: int, seed: int):
    """Paired bootstrap over recordings of mean(a - b). Returns mean, lo, hi, P(diff>0)."""
    rids = sorted(set(a) & set(b))
    d = np.array([a[r] - b[r] for r in rids])
    if len(d) < 3:
        return {"n": len(d), "mean": float(d.mean()) if len(d) else float("nan")}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(axis=1)
    return {"n": len(d), "mean": float(d.mean()),
            "ci95_lo": float(np.quantile(boots, 0.025)), "ci95_hi": float(np.quantile(boots, 0.975)),
            "p_gt_0": float((boots > 0).mean())}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="handover_active,onset_within_5s,ja_active,scoia_onset_within_2s")
    ap.add_argument("--views", default="leader,helper,both,both_shuffled,leader_shuffled")
    ap.add_argument("--seeds", default="0", help="comma-separated fold-partition seeds")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--speech", action="store_true")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--proc-root", type=Path, default=ROOT / "data/processed/comind")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/paired_benchmark/view_controls")
    a = ap.parse_args()

    tasks = a.tasks.split(",")
    views = a.views.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    a.out.mkdir(parents=True, exist_ok=True)
    log_f = open(a.out / "run.log", "a")

    def log(s):
        print(s, flush=True)
        log_f.write(s + "\n")
        log_f.flush()

    src = (ROOT / "scripts/comind_download.py").read_text()
    train_ids = re.findall(r'"([0-9a-f-]{36})"', re.search(r"TRAIN_IDS\s*=\s*\[(.*?)\]", src, re.S).group(1))
    recs = load_recordings(a.proc_root, train_ids)
    recs = [r for r in recs if all(t in r.y for t in tasks)]
    log(f"{len(recs)} recordings, {sum(len(r.handover_onsets) for r in recs)} handovers, "
        f"{sum(len(r.x) for r in recs) / FPS / 3600:.1f} h, tasks={tasks}, seeds={seeds}")

    all_views = views + list(FUSIONS)
    results = {"config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
               "n_recordings": len(recs), "per_seed": []}

    for seed in seeds:
        out, y, fold_of = fit_views(recs, tasks, views, seed, a.folds, a.speech, log)
        for v in FUSIONS:
            out[v] = {t: {} for t in tasks}
        out = add_fusions(out, y, fold_of, tasks, a.folds)

        entry = {"seed": seed, "per_fold": {}, "per_recording": {}, "comparisons": {}}
        for t in tasks:
            entry["per_fold"][t] = {v: per_fold_auroc(out, y, fold_of, v, t, a.folds).tolist() for v in all_views}
            entry["per_recording"][t] = {v: per_recording_auroc(out, y, v, t) for v in all_views}

            pf = {v: np.array(entry["per_fold"][t][v]) for v in all_views}
            best_single = max(SINGLE, key=lambda v: np.nanmean(pf[v]))
            rec = entry["per_recording"][t]
            cmps = {}
            for name, ref in (("vs_best_single", best_single), ("vs_both_shuffled", "both_shuffled"),
                              ("vs_leader_shuffled", "leader_shuffled"), ("vs_late_fusion_prob", "late_fusion_prob"),
                              ("vs_late_fusion_rank", "late_fusion_rank")):
                if ref not in pf:
                    continue
                d = pf["both"] - pf[ref]
                cmps[name] = {
                    "reference": ref,
                    "per_fold_diff_mean": float(np.nanmean(d)),
                    "per_fold_diff_sd": float(np.nanstd(d)),
                    "folds_won": int(np.nansum(d > 0)),
                    "bootstrap": paired_bootstrap(rec["both"], rec[ref], a.n_boot, seed),
                }
            entry["comparisons"][t] = {"best_single": best_single, **cmps}
        results["per_seed"].append(entry)
        json.dump(results, open(a.out / "results.json", "w"), indent=1)

    # ---- report
    lines = []
    for t in tasks:
        lines.append(f"\n### {t}\n")
        lines.append("| view | " + " | ".join(f"seed {s} AUROC" for s in seeds) + " |")
        lines.append("|---|" + "---|" * len(seeds))
        for v in all_views:
            cells = []
            for e in results["per_seed"]:
                pf = np.array(e["per_fold"][t][v])
                cells.append(f"{np.nanmean(pf):.3f} ± {np.nanstd(pf):.3f}")
            star = "**" if v == "both" else ""
            lines.append(f"| {star}{v}{star} | " + " | ".join(cells) + " |")
        lines.append("")
        lines.append("| comparison (both − ref) | per-fold mean ± sd | folds won | bootstrap mean [95% CI] | P(>0) |")
        lines.append("|---|---|---|---|---|")
        for name in ("vs_best_single", "vs_both_shuffled", "vs_leader_shuffled", "vs_late_fusion_prob", "vs_late_fusion_rank"):
            e = results["per_seed"][0]["comparisons"][t]
            if name not in e:
                continue
            c = e[name]
            b = c["bootstrap"]
            ci = f"[{b['ci95_lo']:+.3f}, {b['ci95_hi']:+.3f}]" if "ci95_lo" in b else "n/a"
            lines.append(f"| {name.replace('vs_', 'vs ')} ({c['reference']}) | {c['per_fold_diff_mean']:+.3f} ± "
                         f"{c['per_fold_diff_sd']:.3f} | {c['folds_won']}/{len(seeds) and results['config']['folds']} | "
                         f"{b.get('mean', float('nan')):+.3f} {ci} | {b.get('p_gt_0', float('nan')):.3f} |")
    report = "\n".join(lines)
    (a.out / "results.md").write_text(report + "\n")
    log(report)
    log(f"\nwrote {a.out / 'results.md'} and results.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
