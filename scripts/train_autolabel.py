#!/usr/bin/env python3
"""Train the playground autolabel model on CoMind with the REDUCED feature set of duet.playground.autolabel.

Data: the 44 labelled CoMind recordings with a kinematics cache (same recordings, folds and tree settings as
scripts/baseline_gbdt.py), resampled to 10 Hz (the playground's proc_fps). Per person: the Aria wrist (landmark 5)
in the device frame is projected with an equidistant fisheye about the RGB optical axis so its 2D speed is in image
widths per second like the MediaPipe wrist on the playground side; hands present; held flags from objects_v0.npz
(what each hand holds); head-forward angle to the partner's head and inter-person distances from world_v0.npz
(11 recordings; zeros + validity flag elsewhere). Then the same causal window statistics as the stage.

Tasks: joint attention (ja_active) and handover in progress (handover_active), recording-level 5-fold CV, plus a
direction model (A gives to B) trained on handover frames only. The model is trained on both person orderings
(A,B) and (B,A) so the stage does not need to know who is the leader.

Writes data/playground/models/autolabel_gbdt.pkl: {"models": {task: clf}, "feature_names", "thresholds": {task: {hi, lo,
target_precision, ...}}, "meta": {cv results, comparison}}.
Usage: .venv/bin/python scripts/train_autolabel.py [--out cv_results.json] [--target-precision 0.5]
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.adapters.comind.kinematics import HAND_DIM, person_slice  # noqa: E402
from duet.adapters.comind.labels import load_handovers  # noqa: E402
from duet.adapters.comind.shared_world_features import rgb_forward_axis  # noqa: E402
from duet.ml.paired_benchmark import load_recordings  # noqa: E402
from duet.playground import autolabel as AL  # noqa: E402

PROC = ROOT / "data/processed/comind"; RAW = ROOT / "data/raw/comind"
MODEL_OUT = ROOT / "data/playground/models/autolabel_gbdt.pkl"
COMPARISON_JSON = "outputs/paired_benchmark/baseline_gbdt_grasp.json"
SRC_FPS, FPS = 30, 10.0; DS = SRC_FPS // int(FPS)
START_S, STRIDE = 3.0, 2  # skip the window warm-up; evaluate/train at 5 Hz like the benchmark
OBJ = 31; HELD_COLS = (8, 17)
TASKS = ("ja_active", "handover_active")
TOL_S = 1.0


def train_ids():
    src = (ROOT / "scripts/comind_download.py").read_text()
    return re.findall(r'"([0-9a-f-]{36})"', re.search(r"TRAIN_IDS\s*=\s*\[(.*?)\]", src, re.S).group(1))


def comind_person_base(r, role: str, objs) -> np.ndarray:
    s = person_slice(role); n = len(r.x)
    wr = np.stack([r.x[:, s.start + h * HAND_DIM + 15: s.start + h * HAND_DIM + 18] for h in range(2)], axis=1).astype(np.float64)
    valid = np.stack([r.x[:, s.start + h * HAND_DIM + 25] > 0 for h in range(2)], axis=1)
    uv = AL.project_equidistant(wr, rgb_forward_axis(PROC / r.rid, role)); uv[~valid] = np.nan
    held = None
    if objs is not None:
        i = 0 if role == "leader" else 1
        blk = objs[:, i * OBJ: (i + 1) * OBJ]
        if blk[:, OBJ - 1].max() > 0:  # has_video flag: the object pass ran for this role
            held = blk[:, list(HELD_COLS)]
    ang = None
    if r.w is not None:
        col = 29 if role == "leader" else 30; ok = r.w[:, 31] > 0
        ang = np.where(ok, np.degrees(np.arccos(np.clip(r.w[:, col], -1, 1))), np.nan)
    return AL.person_base(uv[::DS], None if held is None else held[::DS], None if ang is None else ang[::DS], FPS)


def comind_inter_base(r, n10: int) -> np.ndarray:
    if r.w is None:
        return AL.inter_base(None, None, n10)
    ok = r.w[:, 31] > 0
    hd = np.where(ok, r.w[:, 6], np.nan)[::DS]; mw = np.where(ok & (r.w[:, 4] > 0), r.w[:, 4], np.nan)[::DS]
    return AL.inter_base(hd, mw, n10)


def direction_labels(r) -> np.ndarray:
    """Per 30 Hz frame: 1 if the active handover flows leader -> helper (A gives), 0 if helper -> leader, NaN outside."""
    out = np.full(len(r.x), np.nan, np.float32)
    for h in load_handovers(RAW / "annotations", r.rid):
        s, e = max(0, h.start_frame), min(len(out) - 1, h.end_frame)
        if e >= s:
            out[s: e + 1] = 1.0 if h.flow_leader_to_helper else 0.0
    return out


def build(recs):
    data = {}
    for r in recs:
        p = PROC / r.rid / "objects_v0.npz"; objs = np.load(p)["objects"] if p.exists() else None
        bL, bH = comind_person_base(r, "leader", objs), comind_person_base(r, "helper", objs)
        n10 = len(bL); inter = comind_inter_base(r, n10)
        X_ab = AL.assemble(bL, bH, inter, FPS); X_ba = AL.assemble(bH, bL, inter, FPS)
        y = {t: r.y[t][::DS][:n10] for t in TASKS}
        d = direction_labels(r)[::DS][:n10]
        hos = [(h.start_frame / SRC_FPS, (h.end_frame + 1) / SRC_FPS) for h in load_handovers(RAW / "annotations", r.rid)]
        data[r.rid] = {"X_ab": X_ab, "X_ba": X_ba, "y": y, "dir": d, "n": n10, "handovers_s": hos,
                       "has_world": r.w is not None, "has_objects": objs is not None}
    return data


def rows(n: int) -> np.ndarray:
    return np.arange(int(round(START_S * FPS)), n, STRIDE)


def make_clf(y, seed):
    pos = float(np.mean(y))
    return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                                          class_weight={0: 1.0, 1: float(min(50, (1 - pos) / max(pos, 1e-4)))}, random_state=seed)


def stack(data, rids, task, both_orders=True):
    xs, ys = [], []
    for rid in rids:
        d = data[rid]; idx = rows(d["n"])
        xs.append(d["X_ab"][idx]); ys.append(d["y"][task][idx])
        if both_orders:
            xs.append(d["X_ba"][idx]); ys.append(d["y"][task][idx])
    return np.concatenate(xs), np.concatenate(ys)


def stack_direction(data, rids, both_orders=True):
    xs, ys = [], []
    for rid in rids:
        d = data[rid]; m = np.isfinite(d["dir"]) & (d["y"]["handover_active"] > 0)
        idx = np.flatnonzero(m); idx = idx[idx >= int(round(START_S * FPS))]
        if len(idx) == 0:
            continue
        xs.append(d["X_ab"][idx]); ys.append(d["dir"][idx])
        if both_orders:
            xs.append(d["X_ba"][idx]); ys.append(1.0 - d["dir"][idx])
    return (np.concatenate(xs), np.concatenate(ys)) if xs else (None, None)


def intervals_to_s(iv, fps=FPS):
    return [(k0 / fps, k1 / fps) for k0, k1 in iv]


def event_prf(props, truth, tol=TOL_S):
    """Event-level precision/recall: a proposal is a hit if it overlaps a truth interval or its onset is within tol of one."""
    def hit(a, b):
        return (a[0] < b[1] and b[0] < a[1]) or abs(a[0] - b[0]) <= tol
    tp_p = sum(any(hit(p, t) for t in truth) for p in props); tp_t = sum(any(hit(p, t) for p in props) for t in truth)
    return {"n_proposals": len(props), "n_truth": len(truth), "precision": tp_p / len(props) if props else None, "recall": tp_t / len(truth) if truth else None}


def label_intervals(y: np.ndarray, fps=FPS):
    y = np.asarray(y) > 0; out = []; k = 0
    while k < len(y):
        if y[k]:
            j = k
            while j < len(y) and y[j]:
                j += 1
            out.append((k / fps, j / fps)); k = j
        else:
            k += 1
    return out


def event_level(data, oof_per_rec, task, hi, lo):
    tot = {"n_proposals": 0, "n_truth": 0, "tp_p": 0, "tp_t": 0}
    for rid, d in data.items():
        props = intervals_to_s(AL.hysteresis_intervals(oof_per_rec[rid][task], FPS, hi, lo)); truth = label_intervals(d["y"][task])
        e = event_prf(props, truth); tot["n_proposals"] += e["n_proposals"]; tot["n_truth"] += e["n_truth"]
        tot["tp_p"] += round((e["precision"] or 0) * e["n_proposals"]); tot["tp_t"] += round((e["recall"] or 0) * e["n_truth"])
    P = tot["tp_p"] / max(1, tot["n_proposals"]); R = tot["tp_t"] / max(1, tot["n_truth"])
    return {"n_proposals": tot["n_proposals"], "n_truth": tot["n_truth"], "precision": P, "recall": R, "f1": 2 * P * R / max(P + R, 1e-9),
            "criterion": f"overlap or onset within {TOL_S} s, hysteresis {AL.MIN_ON_S}/{AL.MIN_OFF_S} s"}


def choose_thresholds(y, p, target_precision, data, oof_per_rec, task):
    """hi: the frame-level threshold reaching `target_precision` on the pooled OOF probabilities with the highest recall; if
    unreachable, the hi (from a quantile grid) that maximises the EVENT-level F1 of the hysteresis on the OOF traces. lo = 0.6 hi."""
    prec, rec, th = precision_recall_curve(y, p)
    prec, rec = prec[:-1], rec[:-1]
    ok = np.flatnonzero(prec >= target_precision)
    grid = None
    if len(ok):
        i = ok[np.argmax(rec[ok])]; hi = float(th[i]); how = f"precision >= {target_precision} (frame level, pooled OOF)"
        frame = {"precision_at_hi": float(prec[i]), "recall_at_hi": float(rec[i])}
    else:
        grid = []
        for q in (0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995):
            h = float(np.quantile(p, q)); e = event_level(data, oof_per_rec, task, h, 0.6 * h); grid.append({"quantile": q, "hi": h, **e})
        best = max(grid, key=lambda g: g["f1"]); hi = best["hi"]; how = f"target precision unreachable; max event-level F1 over OOF quantile grid (q={best['quantile']})"
        j = int(np.searchsorted(th, hi)); j = min(j, len(prec) - 1); frame = {"precision_at_hi": float(prec[j]), "recall_at_hi": float(rec[j])}
    lo = 0.6 * hi
    return {"hi": hi, "lo": lo, "rule": how, **frame, "target_precision": target_precision, "oof_event_level": event_level(data, oof_per_rec, task, hi, lo), "grid": grid}


def validate_clip(episode: str, rid: str, offset_s: float, tol: float = TOL_S) -> dict:
    """Compare derived/autolabel/proposals.json of a playground episode cut from CoMind recording `rid` (episode reference
    time 0 = source leader video at `offset_s`) with the CoMind annotations inside the episode's common window."""
    from duet.adapters.comind.labels import load_joint_attention_intervals
    from duet.playground.episode import Episode
    ep = Episode.load(ROOT / "data/playground/episodes" / episode)
    props = json.load(open(ep.derived / "autolabel" / "proposals.json"))
    lo, hi = ep.common_start_s, ep.common_end_s
    truth = {"handover": [((h.start_frame / SRC_FPS) - offset_s, ((h.end_frame + 1) / SRC_FPS) - offset_s, ("leader" if h.flow_leader_to_helper else "helper"), ("helper" if h.flow_leader_to_helper else "leader"))
                          for h in load_handovers(RAW / "annotations", rid)],
             "joint_attention": [((s / SRC_FPS) - offset_s, ((e + 1) / SRC_FPS) - offset_s, None, None) for s, e, _ in load_joint_attention_intervals(RAW / "annotations", rid)]}
    truth = {k: [t for t in v if t[1] > lo and t[0] < hi] for k, v in truth.items()}
    out = {"episode": episode, "recording": rid, "clip_offset_s": offset_s, "window_s": [lo, hi], "tolerance_s": tol, "events": {}}
    # per-frame AUROC of the probabilities against the annotation masks on the episode timeline
    probs = None
    pq = ep.derived / "autolabel" / "records_extra.parquet"
    if pq.exists():
        import pandas as pd
        probs = pd.read_parquet(pq); t = lo + np.arange(len(probs)) / ep.proc_fps
    for ev, tv in truth.items():
        pv = [p for p in props if p["event"] == ev]
        crit = {"onset_within_tol": lambda p, t_: abs(p["t0"] - t_[0]) <= tol,
                "onset_and_offset_within_tol": lambda p, t_: abs(p["t0"] - t_[0]) <= tol and abs(p["t1"] - t_[1]) <= tol,
                "overlap": lambda p, t_: p["t0"] < t_[1] and t_[0] < p["t1"]}
        r = {"n_proposals": len(pv), "n_truth": len(tv), "truth": [{"t0": round(a, 2), "t1": round(b, 2), "giver": g, "receiver": rc} for a, b, g, rc in tv], "proposals": pv}
        for name, fn in crit.items():
            tp_p = sum(any(fn(p, t_) for t_ in tv) for p in pv); tp_t = sum(any(fn(p, t_) for p in pv) for t_ in tv)
            r[name] = {"precision": tp_p / len(pv) if pv else None, "recall": tp_t / len(tv) if tv else None, "tp_proposals": tp_p, "tp_truth": tp_t}
        if ev == "handover" and pv:
            hits = [(p, t_) for p in pv for t_ in tv if crit["overlap"](p, t_)]
            r["direction_correct"] = {"n": len(hits), "correct": sum(p["giver"] == t_[2] for p, t_ in hits)}
        if probs is not None and f"p_{ev}" in probs:
            y = np.zeros(len(probs), bool)
            for a, b, _, _ in tv:
                y |= (t >= a) & (t < b)
            p = probs[f"p_{ev}"].to_numpy()
            r["frame_level"] = {"prevalence": float(y.mean()), "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None, "ap": float(average_precision_score(y, p)) if y.any() else None}
        out["events"][ev] = r
    # Domain-shift check: the same window featurised from the CoMind side (Aria hands, objects_v0 held flags) through the
    # same model. If the model works here but not on the playground features, the gap is the featurisation, not the model.
    recs = [r for r in load_recordings(PROC, [rid]) if r.rid == rid]
    if recs and probs is not None:
        d = build(recs)[rid]; model = AL.load_model(MODEL_OUT)
        k0, k1 = int(round((offset_s + lo) * FPS)), int(round((offset_s + hi) * FPS))
        Xc = d["X_ab"][k0:k1]; n = min(len(Xc), len(probs)); Xc = Xc[:n]
        base_names = AL.feature_names()[: 2 * len(AL.PERSON_FEATS) + len(AL.INTER_FEATS)]
        Xp, _ = AL.playground_features(ep, len(probs))
        out["comind_side"] = {"n_frames": n, "base_feature_means": {nm: {"playground": float(Xp[:n, i].mean()), "comind": float(Xc[:, i].mean())} for i, nm in enumerate(base_names)}}
        for ev, task in (("joint_attention", "ja_active"), ("handover", "handover_active")):
            yc = d["y"][task][k0:k1][:n] > 0; pc = np.nan_to_num(model["models"][task].predict_proba(Xc)[:, 1], nan=0.0)
            th = model["thresholds"][task]; props_c = [(lo + a / FPS, lo + b / FPS) for a, b in AL.hysteresis_intervals(pc, FPS, th["hi"], th["lo"])]
            tv = truth[ev]; e = event_prf(props_c, [(a, b) for a, b, _, _ in tv], tol)
            out["comind_side"][ev] = {"frame_auroc": float(roc_auc_score(yc, pc)) if 0 < yc.sum() < len(yc) else None, "prevalence": float(yc.mean()), "mean_p": float(pc.mean()),
                                      "proposals": [(round(a, 2), round(b, 2)) for a, b in props_c], "event_level_overlap_or_onset": e}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the CV results JSON here")
    ap.add_argument("--target-precision", type=float, default=0.5)
    ap.add_argument("--model-out", default=str(MODEL_OUT))
    ap.add_argument("--validate-clip", default=None, metavar="EPISODE", help="skip training; compare EPISODE's proposals with the CoMind annotations of --recording")
    ap.add_argument("--recording", default="43276420-701f-4731-b9ab-bebc7fd14994")
    ap.add_argument("--clip-offset-s", type=float, default=590.0, help="episode reference t=0 in the source leader video (s)")
    a = ap.parse_args()
    if a.validate_clip:
        res = validate_clip(a.validate_clip, a.recording, a.clip_offset_s)
        for ev, r in res["events"].items():
            print(f"{ev}: {r['n_proposals']} proposals vs {r['n_truth']} annotated in window {res['window_s']}")
            for name in ("onset_within_tol", "onset_and_offset_within_tol", "overlap"):
                print(f"  {name}: precision {r[name]['precision']} recall {r[name]['recall']}")
            if "frame_level" in r:
                print(f"  frame-level: prevalence {r['frame_level']['prevalence']:.2f} AUROC {r['frame_level']['auroc']} AP {r['frame_level']['ap']}")
            if "direction_correct" in r:
                print(f"  direction correct on overlapping proposals: {r['direction_correct']}")
            for p in r["proposals"]:
                print("   proposal", {k: v for k, v in p.items() if k in ('t0', 't1', 'confidence', 'giver', 'receiver', 'p_a_gives')})
            for t_ in r["truth"]:
                print("   truth   ", t_)
        if "comind_side" in res:
            c = res["comind_side"]; print(f"same window featurised from the CoMind side ({c['n_frames']} frames), same model:")
            for ev in ("joint_attention", "handover"):
                e = c[ev]; print(f"  {ev}: frame AUROC {e['frame_auroc']} (prevalence {e['prevalence']:.2f}, mean p {e['mean_p']:.3f}); proposals {e['proposals']}; event-level {e['event_level_overlap_or_onset']}")
            print("  base feature means (playground vs CoMind side):")
            for nm, v in c["base_feature_means"].items():
                print(f"    {nm:28s} {v['playground']:9.4f} {v['comind']:9.4f}")
        if a.out:
            json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)
        return
    t0 = time.time()
    recs = load_recordings(PROC, train_ids())
    print(f"{len(recs)} recordings; building reduced features at {FPS:g} Hz", flush=True)
    data = build(recs)
    names = AL.feature_names(); assert data[recs[0].rid]["X_ab"].shape[1] == len(names), (data[recs[0].rid]["X_ab"].shape, len(names))
    print(f"  features {len(names)}; rows/recording ~{np.mean([len(rows(d['n'])) for d in data.values()]):.0f}; with world {sum(d['has_world'] for d in data.values())}, with objects {sum(d['has_objects'] for d in data.values())}; built in {time.time() - t0:.0f}s", flush=True)
    K = 5; rng = np.random.default_rng(0); order = rng.permutation(len(recs)); folds = [sorted(order[i::K].tolist()) for i in range(K)]
    res = {"n_recordings": len(recs), "fps": FPS, "n_features": len(names), "tasks": {}, "direction": {}, "comparison_json": COMPARISON_JSON}
    ref = json.load(open(ROOT / COMPARISON_JSON))["results"]
    oof = {t: (np.zeros(0), np.zeros(0)) for t in TASKS}; oof_per_rec = {}
    for t in TASKS:
        per_fold, pool_y, pool_p = [], [], []
        for k in range(K):
            te = [recs[i].rid for i in folds[k]]; tr = [recs[i].rid for j in range(K) if j != k for i in folds[j]]
            Xtr, ytr = stack(data, tr, t); clf = make_clf(ytr, k); clf.fit(Xtr, ytr)
            Xte, yte = stack(data, te, t, both_orders=False); p = np.nan_to_num(clf.predict_proba(Xte)[:, 1], nan=0.5)
            if 0 < yte.sum() < len(yte):
                per_fold.append(float(roc_auc_score(yte, p)))
            pool_y.append(yte); pool_p.append(p)
            for rid in te:  # full-rate OOF probabilities for the event-level check
                oof_per_rec.setdefault(rid, {})[t] = np.nan_to_num(clf.predict_proba(data[rid]["X_ab"])[:, 1], nan=0.5)
            print(f"  {t} fold {k}: AUROC {per_fold[-1]:.3f} (train rows {len(ytr)}, pos {ytr.mean():.3f}) [{time.time() - t0:.0f}s]", flush=True)
        y_all, p_all = np.concatenate(pool_y), np.concatenate(pool_p); oof[t] = (y_all, p_all)
        res["tasks"][t] = {"auroc_fold_mean": float(np.mean(per_fold)), "auroc_fold_sd": float(np.std(per_fold)), "auroc_folds": per_fold,
                           "ap_pooled": float(average_precision_score(y_all, p_all)), "prevalence": float(y_all.mean()),
                           "reference_full_features": {v: ref[v][t] for v in ref if t in ref[v]}}
        print(f"  {t}: reduced AUROC {np.mean(per_fold):.3f} +- {np.std(per_fold):.3f} | full-feature ({COMPARISON_JSON}) both {ref['both'][t]['auroc_fold_mean']:.3f} +- {ref['both'][t]['auroc_fold_sd']:.3f}, helper {ref['helper'][t]['auroc_fold_mean']:.3f} +- {ref['helper'][t]['auroc_fold_sd']:.3f}", flush=True)
    # direction model CV (handover frames only)
    per_fold, ev_acc, ev_n = [], 0, 0
    for k in range(K):
        te = [recs[i].rid for i in folds[k]]; tr = [recs[i].rid for j in range(K) if j != k for i in folds[j]]
        Xtr, ytr = stack_direction(data, tr); Xte, yte = stack_direction(data, te, both_orders=False)
        if Xtr is None or Xte is None:
            continue
        clf = make_clf(ytr, k); clf.fit(Xtr, ytr); p = np.nan_to_num(clf.predict_proba(Xte)[:, 1], nan=0.5)
        if 0 < yte.sum() < len(yte):
            per_fold.append(float(roc_auc_score(yte, p)))
        for rid in te:  # event level: mean prob over each annotated handover
            d = data[rid]; pr = np.nan_to_num(clf.predict_proba(d["X_ab"])[:, 1], nan=0.5)
            for (s, e) in d["handovers_s"]:
                k0, k1 = int(s * FPS), max(int(s * FPS) + 1, int(e * FPS))
                if k1 > d["n"] or not np.isfinite(d["dir"][k0:k1]).any():
                    continue
                truth = np.nanmean(d["dir"][k0:k1]) >= 0.5; ev_acc += int((pr[k0:k1].mean() >= 0.5) == truth); ev_n += 1
    res["direction"] = {"frame_auroc_fold_mean": float(np.mean(per_fold)) if per_fold else None, "frame_auroc_fold_sd": float(np.std(per_fold)) if per_fold else None,
                        "event_accuracy": ev_acc / ev_n if ev_n else None, "n_events": ev_n}
    print(f"  direction (A gives): frame AUROC {res['direction']['frame_auroc_fold_mean']:.3f}, event accuracy {res['direction']['event_accuracy']:.3f} on {ev_n} handovers", flush=True)
    # thresholds from pooled OOF probabilities, and the event-level behaviour of the hysteresis on OOF traces
    thresholds = {t: choose_thresholds(*oof[t], a.target_precision, data, oof_per_rec, t) for t in TASKS}
    for t in TASKS:
        e = thresholds[t]["oof_event_level"]
        print(f"  thresholds {t}: hi {thresholds[t]['hi']:.3f} lo {thresholds[t]['lo']:.3f} ({thresholds[t]['rule']}; frame P {thresholds[t]['precision_at_hi']:.2f} R {thresholds[t]['recall_at_hi']:.2f}); OOF events: {e['n_proposals']} proposals vs {e['n_truth']} truth, P {e['precision']:.2f} R {e['recall']:.2f} F1 {e['f1']:.2f}", flush=True)
    res["thresholds"] = thresholds
    # final fit on all recordings
    models = {}
    for t in TASKS:
        X, y = stack(data, [r.rid for r in recs], t); models[t] = make_clf(y, 0).fit(X, y)
    Xd, yd = stack_direction(data, [r.rid for r in recs]); models["handover_direction"] = make_clf(yd, 0).fit(Xd, yd) if Xd is not None else None
    meta = {"trained": str(date.today()), "n_recordings": len(recs), "fps": FPS, "cv": res, "comparison_json": COMPARISON_JSON, "feature_source": "duet.playground.autolabel (reduced set)"}
    Path(a.model_out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.model_out, "wb") as f:
        pickle.dump({"models": models, "feature_names": names, "thresholds": thresholds, "meta": meta}, f)
    print(f"saved {a.model_out} ({Path(a.model_out).stat().st_size / 1e6:.1f} MB) in {time.time() - t0:.0f}s", flush=True)
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
