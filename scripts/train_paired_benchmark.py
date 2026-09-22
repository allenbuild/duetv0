#!/usr/bin/env python3
"""Single-view vs paired-view kinematic benchmark on CoMind (recording-level K-fold CV).

CoMind withholds annotations for its official test split, so all evaluation is
recording-level cross-validation over the annotated train-split recordings.
For each fold: test = fold k, val = fold k+1 (model selection), train = the rest.
Test predictions are pooled across folds so rare-event AP is computed over every
annotated handover in the dataset.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sklearn.metrics import average_precision_score, roc_auc_score

import duet.ml.paired_benchmark as pb
from duet.ml.paired_benchmark import (
    FPS,
    REG_TASK,
    VIEWS,
    TrainConfig,
    load_recordings,
    predict,
    train_view,
    view_input,
)


def official_split() -> tuple[list[str], list[str]]:
    src = (ROOT / "scripts/comind_download.py").read_text()

    def block(name):
        m = re.search(name + r"\s*=\s*\[(.*?)\]", src, re.DOTALL)
        return re.findall(r'"([0-9a-f-]{36})"', m.group(1))

    return block("TRAIN_IDS"), block("TEST_IDS")


def pooled_metrics(pool):
    CLS_TASKS = pb.CLS_TASKS
    REG_TASK = pb.REG_TASK
    out = {}
    for t in CLS_TASKS:
        y = np.concatenate(pool[t]["y"]); p = np.concatenate(pool[t]["p"])
        out[t] = {"ap": float(average_precision_score(y, p)), "auroc": float(roc_auc_score(y, p)), "prevalence": float(y.mean()),
                  "n_pos_frames": int(y.sum())}
    e = np.concatenate(pool[REG_TASK])
    out[REG_TASK] = {"mae_s": float(e.mean()), "n_frames": len(e)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc-root", default="data/processed/comind")
    ap.add_argument("--out", default="outputs/paired_benchmark")
    ap.add_argument("--views", default=",".join(VIEWS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--demo-recording", default="43276420-701f-4731-b9ab-bebc7fd14994")
    ap.add_argument("--onset-crop-frac", type=float, default=0.0, help="fraction of training crops forced to contain a handover onset")
    ap.add_argument("--task-weights", default="", help="comma-separated loss weights, one per task (default all 1)")
    ap.add_argument("--pos-weight-cap", type=float, default=20.0)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--speech", action="store_true", help="append per-person speech features from transcripts")
    ap.add_argument("--world", action="store_true", help="append cross-person shared-world block to the paired view; restricts to recordings that have it")
    ap.add_argument("--world-subset", action="store_true", help="restrict to recordings with world features without using them (control)")
    ap.add_argument("--tasks", default=",".join(pb.CLS_TASKS), help="comma-separated classification tasks (label names in kinematics_v0.npz)")
    ap.add_argument("--speech-source", default="hashed", choices=["hashed", "emb", "both"])
    ap.add_argument("--reg-task", default="tth_s", choices=["tth_s", "tte_s"])
    ap.add_argument("--reg-max-s", type=float, default=8.0)
    ap.add_argument("--select-metric", default="mean_auroc", choices=["mean_auroc", "mean_ap"])
    a = ap.parse_args()
    pb.set_tasks(a.tasks.split(","))
    pb.set_reg_task(a.reg_task, a.reg_max_s)
    CLS_TASKS = pb.CLS_TASKS
    REG_TASK = pb.REG_TASK
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    train_ids, _ = official_split()
    recs = load_recordings(Path(a.proc_root), train_ids, speech_source=a.speech_source)
    if a.world or "--world-subset" in sys.argv:
        recs = [r for r in recs if r.w is not None]
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(recs))
    folds = [sorted(order[i::a.folds].tolist()) for i in range(a.folds)]
    info = {"n_recordings": len(recs), "hours": sum(len(r.x) for r in recs) / FPS / 3600,
            "handovers": sum(len(r.handover_onsets) for r in recs), "folds": a.folds, "steps": a.steps,
            "device": TrainConfig().device, "onset_crop_frac": a.onset_crop_frac, "task_weights": a.task_weights, "pos_weight_cap": a.pos_weight_cap, "dropout": a.dropout, "channels": a.channels, "speech": a.speech, "world": a.world, "select_metric": a.select_metric, "tasks": list(CLS_TASKS), "reg_task": a.reg_task, "speech_source": a.speech_source, "reg_max_s": a.reg_max_s, "recording_ids": [r.rid for r in recs]}
    print(json.dumps({k: v for k, v in info.items() if k != "recording_ids"}, indent=1), flush=True)
    log_f = open(out / "train.log", "a")

    def log(s):
        print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

    all_results = {"info": info, "per_fold": [], "pooled": {}}
    for view in a.views.split(","):
        pool = {t: {"y": [], "p": []} for t in CLS_TASKS}; pool[REG_TASK] = []
        for k in range(a.folds):
            te = [recs[i] for i in folds[k]]
            va = [recs[i] for i in folds[(k + 1) % a.folds]]
            tr = [recs[i] for j in range(a.folds) if j not in (k, (k + 1) % a.folds) for i in folds[j]]
            cfg = TrainConfig(steps=a.steps, eval_every=a.eval_every, channels=a.channels, seed=a.seed * 100 + k,
                              onset_crop_frac=a.onset_crop_frac, task_weights=tuple(float(v) for v in a.task_weights.split(",")) if a.task_weights else (),
                              pos_weight_cap=a.pos_weight_cap, dropout=a.dropout, use_speech=a.speech, use_world=a.world, select_metric=a.select_metric)
            t0 = time.time()
            res, _, model, norm = train_view(tr, va, te, view, cfg, log=log)
            res["fold"] = k
            all_results["per_fold"].append(res)
            log(f"FOLD {view} k={k}: " + " ".join(f"{t} AP={res['test'][t]['ap']:.3f}" for t in CLS_TASKS)
                + f" tthMAE={res['test'][REG_TASK]['mae_s']:.2f}s ({time.time()-t0:.0f}s)")
            # pool held-out predictions
            for r in te:
                p, reg = predict(model, norm(view_input(r.x, view, np.random.default_rng(321), r.s if cfg.use_speech else None, r.w if cfg.use_world else None, rid=r.rid)), cfg)
                valid = np.ones(len(r.x), bool); valid[: 2 * FPS] = False
                for i, t in enumerate(CLS_TASKS):
                    pool[t]["y"].append(r.y[t][valid]); pool[t]["p"].append(p[valid, i])
                m = np.isfinite(r.y[REG_TASK]) & (r.y[REG_TASK] <= pb.REG_MAX_S) & valid
                pool[REG_TASK].append(np.abs(np.clip(reg[m], 0, pb.REG_MAX_S) - r.y[REG_TASK][m]))
                np.savez_compressed(out / f"preds_{view}_{r.rid[:8]}.npz", probs=p.astype(np.float16), tth=reg.astype(np.float16))
            torch.save({"state": model.state_dict(), "norm_mean": norm.mean, "norm_std": norm.std, "cfg": cfg.__dict__, "view": view, "d_in": res["d_in"]},
                       out / f"model_{view}_k{k}.pt")
        all_results["pooled"][view] = pooled_metrics(pool)
        log(f"POOLED {view}: " + " ".join(f"{t} AP={all_results['pooled'][view][t]['ap']:.3f} AUROC={all_results['pooled'][view][t]['auroc']:.3f}" for t in CLS_TASKS)
            + f" tthMAE={all_results['pooled'][view][REG_TASK]['mae_s']:.2f}s")
        json.dump(all_results, open(out / "results.json", "w"), indent=1)

    views = a.views.split(",")
    lines = ["| view | " + " | ".join(f"{t} AP" for t in CLS_TASKS) + " | TTH MAE (s) |", "|---|" + "---|" * (len(CLS_TASKS) + 1)]
    first = all_results["pooled"][views[0]]
    lines.append("| chance (prevalence) | " + " | ".join(f"{first[t]['prevalence']:.3f}" for t in CLS_TASKS) + " | - |")
    for view in views:
        pm = all_results["pooled"][view]
        fr = [r for r in all_results["per_fold"] if r["view"] == view]
        cells = [f"**{pm[t]['ap']:.3f}** (folds {np.mean([r['test'][t]['ap'] for r in fr]):.3f} ± {np.std([r['test'][t]['ap'] for r in fr]):.3f})" for t in CLS_TASKS]
        cells.append(f"{pm[REG_TASK]['mae_s']:.2f}")
        lines.append(f"| {view} | " + " | ".join(cells) + " |")
    (out / "results.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
