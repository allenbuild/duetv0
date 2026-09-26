#!/usr/bin/env python3
"""How good is "head forward = gaze"? Validation of the playground gaze_proxy stage on CoMind.

CoMind has real eye tracking (gaze in CPF) and Multi-SLAM head poses, so we can measure what the playground
loses by using the ego camera's optical axis as the gaze ray.

(a) angle between the real gaze direction and the RGB optical axis (head-forward), per frame, in the wearer's device
    frame: median / mean / p90, overall and inside annotated joint-attention intervals. All 44 labelled recordings with
    a kinematics cache (both roles).
(b) AUROC for the per-frame joint-attention label from -angle(forward ray, partner head) vs -angle(real gaze, partner
    head), per recording (mean +- sd) and pooled. Needs both people's poses in one world: the 11 labelled recordings
    with Multi-SLAM closed-loop trajectories (the same subset as world_v0.npz).
(c) --gbdt: the GBDT benchmark (scripts/baseline_gbdt.py functions, same folds/trees/tasks) with the gaze columns
    zeroed, i.e. what a rig without an eye tracker gets; and on the 11-recording paired subset, the cross-person block
    with its gaze-derived columns (25-28) zeroed, keeping the head-forward cosines (29-30).

Reuses duet.adapters.comind.shared_world_features for gaze-to-device, poses and the RGB axis rather than re-deriving.
Usage: .venv/bin/python scripts/eval_gaze_proxy_comind.py [--gbdt] [--out results.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "scripts"))
from duet.adapters.comind.kinematics import HAND_DIM, ROLES, frame_device_times_ns, parse_mp4_tail, person_slice  # noqa: E402
from duet.adapters.comind.shared_world_features import (  # noqa: E402
    NOMINAL_RGB_AXIS_DEVICE, T_DEVICE_CPF, load_poses, multislam_dir, rgb_forward_axis,
)
from duet.ml.paired_benchmark import load_recordings  # noqa: E402

PROC = ROOT / "data/processed/comind"; RAW = ROOT / "data/raw/comind"
LOOK_DEG = 15.0


def train_ids() -> list[str]:
    src = (ROOT / "scripts/comind_download.py").read_text()
    return re.findall(r'"([0-9a-f-]{36})"', re.search(r"TRAIN_IDS\s*=\s*\[(.*?)\]", src, re.S).group(1))


def rgb_axis_from_online_calib(rec: Path, role: str) -> np.ndarray | None:
    """Optical axis of camera-rgb in the device frame from the first online-calibration record (projectaria_tools)."""
    p = rec / f"mps_{role}_trimmed_vrs/slam/online_calibration.jsonl"
    if not p.exists():
        return None
    try:
        from projectaria_tools.core import mps
        cal = mps.read_online_calibration(str(p))
        cam = [c for c in cal[0].camera_calibs if c.get_label() == "camera-rgb"][0]
        ax = cam.get_transform_device_camera().rotation().to_matrix()[:, 2]
        return ax / np.linalg.norm(ax)
    except Exception:  # noqa: BLE001
        return None


def gaze_dir_device(x: np.ndarray, role: str):
    """Unit gaze direction in the device frame [N,3] and validity [N] from the kinematics gaze block (CPF direction)."""
    s = person_slice(role); g = s.start + 2 * HAND_DIM
    d_cpf = x[:, g + 3: g + 6].astype(np.float64); valid = x[:, g + 6] > 0
    d = d_cpf @ T_DEVICE_CPF[:3, :3].T
    nrm = np.linalg.norm(d, axis=1); valid &= nrm > 0
    return d / np.where(nrm > 0, nrm, 1.0)[:, None], valid


def angle_deg(a, b):
    c = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-12)
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


def stats(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return {"n": int(len(v)), "median": float(np.median(v)), "mean": float(v.mean()), "p90": float(np.percentile(v, 90)), "frac_lt_15deg": float((v < LOOK_DEG).mean())} if len(v) else {"n": 0}


def part_a(recs):
    per_rec, pooled, pooled_ja = [], {r: [] for r in ROLES}, {r: [] for r in ROLES}
    axis_dev = []; dirs = {r: [] for r in ROLES}; dirs_ja = {r: [] for r in ROLES}
    for r in recs:
        rec = RAW / "recordings" / r.rid; row = {"rid": r.rid}
        for role in ROLES:
            ax = rgb_axis_from_online_calib(rec, role); src = "online_calibration"
            if ax is None:
                ax = rgb_forward_axis(PROC / r.rid, role); src = "nominal"
            else:
                axis_dev.append(angle_deg(ax, NOMINAL_RGB_AXIS_DEVICE / np.linalg.norm(NOMINAL_RGB_AXIS_DEVICE)))
            d, v = gaze_dir_device(r.x, role)
            a = np.where(v, angle_deg(d, ax[None]), np.nan)
            ja = r.y["ja_active"] > 0
            row[role] = {"axis_source": src, "all": stats(a), "ja": stats(a[ja]), "not_ja": stats(a[~ja]), "gaze_valid": float(v.mean())}
            pooled[role].append(a[v]); pooled_ja[role].append(a[v & ja]); dirs[role].append(d[v]); dirs_ja[role].append(d[v & ja])
        per_rec.append(row)
    out = {"per_recording": per_rec, "n_recordings": len(recs), "axis_from_online_calib_vs_nominal_deg": stats(axis_dev) if axis_dev else None}
    # is the gaze-vs-forward offset a fixed bias (people look below the camera axis) or scatter? Mean gaze axis per role.
    nom = NOMINAL_RGB_AXIS_DEVICE / np.linalg.norm(NOMINAL_RGB_AXIS_DEVICE); out["mean_gaze_axis"] = {}
    for role in ROLES:
        D = np.concatenate(dirs[role]); m = D.mean(0); m /= np.linalg.norm(m)
        Dj = np.concatenate(dirs_ja[role]); mj = Dj.mean(0); mj /= np.linalg.norm(mj)
        out["mean_gaze_axis"][role] = {"axis_device": m.tolist(), "angle_to_rgb_axis_deg": float(angle_deg(m, nom)), "angle_to_device_z_deg": float(angle_deg(m, np.array([0, 0, 1.0]))),
                                       "residual_vs_mean_gaze_axis": stats(angle_deg(D, m[None])), "residual_vs_mean_gaze_axis_ja": stats(angle_deg(Dj, mj[None])),
                                       "vs_device_z": stats(angle_deg(D, np.array([[0, 0, 1.0]])))}
    for role in ROLES:
        out[f"pooled_{role}"] = {"all": stats(np.concatenate(pooled[role])), "ja": stats(np.concatenate(pooled_ja[role]))}
        out[f"per_recording_median_{role}"] = {"all_mean": float(np.mean([x[role]["all"]["median"] for x in per_rec])), "all_sd": float(np.std([x[role]["all"]["median"] for x in per_rec])),
                                               "ja_mean": float(np.mean([x[role]["ja"]["median"] for x in per_rec if x[role]["ja"]["n"]])), "ja_sd": float(np.std([x[role]["ja"]["median"] for x in per_rec if x[role]["ja"]["n"]]))}
    both = np.concatenate([np.concatenate(pooled[r]) for r in ROLES]); both_ja = np.concatenate([np.concatenate(pooled_ja[r]) for r in ROLES])
    out["pooled_both_roles"] = {"all": stats(both), "ja": stats(both_ja)}
    return out


def _auc(y, s):
    y = np.asarray(y) > 0
    return float(roc_auc_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def part_b(recs):
    per, pooled = [], {}
    for r in recs:
        rec = RAW / "recordings" / r.rid; n = len(r.x)
        csvs = {role: multislam_dir(rec, role) for role in ROLES}
        if any(c is None for c in csvs.values()):
            continue
        t0 = time.time(); poses = {}
        for role in ROLES:
            anchor = parse_mp4_tail((PROC / r.rid / f"mp4_tail_{role}.bin").read_bytes())
            poses[role] = load_poses(csvs[role], frame_device_times_ns(anchor)[:n])
        RL, tL, vL, uL = poses["leader"]; RH, tH, vH, uH = poses["helper"]
        both = vL & vH & (uL == uH)
        fwd = {"leader": RL @ rgb_forward_axis(PROC / r.rid, "leader"), "helper": RH @ rgb_forward_axis(PROC / r.rid, "helper")}
        gz, gv = {}, {}
        for role, R in (("leader", RL), ("helper", RH)):
            d, v = gaze_dir_device(r.x, role); gz[role] = np.einsum("nij,nj->ni", R, d); gv[role] = v
        to = {"leader": tH - tL, "helper": tL - tH}
        ang_f = {role: angle_deg(fwd[role], to[role]) for role in ROLES}
        ang_g = {role: angle_deg(gz[role], to[role]) for role in ROLES}
        ok = both & gv["leader"] & gv["helper"]
        # cross-check against the cached cross-person block (cols 29/30 = cos of the forward angle)
        chk = None
        if r.w is not None:
            wv = r.w[:, 31] > 0
            chk = float(np.nanmax(np.abs(np.degrees(np.arccos(np.clip(r.w[wv & both, 29], -1, 1))) - ang_f["leader"][wv & both]))) if (wv & both).any() else None
        y = r.y["ja_active"][ok]
        scores = {"forward_leader": -ang_f["leader"][ok], "forward_helper": -ang_f["helper"][ok], "forward_min": -np.minimum(ang_f["leader"], ang_f["helper"])[ok],
                  "gaze_leader": -ang_g["leader"][ok], "gaze_helper": -ang_g["helper"][ok], "gaze_min": -np.minimum(ang_g["leader"], ang_g["helper"])[ok]}
        row = {"rid": r.rid, "n_frames_valid": int(ok.sum()), "frac_valid": float(ok.mean()), "ja_prevalence": float(y.mean()) if ok.any() else None,
               "auroc": {k: _auc(y, s) for k, s in scores.items()}, "world_v0_forward_angle_max_abs_diff_deg": chk,
               "median_angle_deg": {k: float(np.median(-s)) for k, s in scores.items()}, "seconds": round(time.time() - t0, 1)}
        per.append(row)
        for k, s in scores.items():
            pooled.setdefault(k, ([], []))[0].append(y); pooled[k][1].append(s)
        print(f"  (b) {r.rid[:8]} valid {ok.mean():.0%} JA {y.mean():.2f} " + " ".join(f"{k}={v:.3f}" for k, v in row["auroc"].items()), flush=True)
    out = {"per_recording": per, "n_recordings": len(per)}
    for k in pooled:
        vals = np.array([p["auroc"][k] for p in per]); vals = vals[np.isfinite(vals)]
        out[k] = {"auroc_rec_mean": float(vals.mean()), "auroc_rec_sd": float(vals.std()), "auroc_pooled": _auc(np.concatenate(pooled[k][0]), np.concatenate(pooled[k][1]))}
    return out


def part_c(recs, tasks=("handover_active", "ja_active"), views=("helper", "both")):
    """Replicates scripts/baseline_gbdt.py main() (folds, trees, stride, speech on) by importing its functions."""
    import baseline_gbdt as B
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import average_precision_score
    FPS = B.FPS; K = 5; results = {}

    def run(recs, mask_person_gaze: bool, world: bool, mask_world_gaze: bool, views):
        rng = np.random.default_rng(0); order = rng.permutation(len(recs)); folds = [sorted(order[i::K].tolist()) for i in range(K)]
        X = {}
        for r in recs:
            blocks = {}
            for role in ("leader", "helper"):
                base = B.person_base(r.x, role, r.s)
                if mask_person_gaze:
                    base[:, 15:19] = 0.0  # gaze point xyz + gaze valid (see person_base column order)
                blocks[role] = B.featurize(base)
            if world and r.w is not None:
                w = r.w[:, :31].astype(np.float32).copy()
                if mask_world_gaze:
                    w[:, 25:29] = 0.0  # gaze-point distances to partner wrists / head
                blocks["world"] = B.featurize(w)
            X[r.rid] = blocks
        res = {}
        for view in views:
            pool = {t: ([], []) for t in tasks}; per_fold = {t: [] for t in tasks}
            for k in range(K):
                te = [recs[i] for i in folds[k]]; tr = [recs[i] for j in range(K) if j != k for i in folds[j]]

                def mk(rs):
                    xs = []
                    for r in rs:
                        parts = [X[r.rid][b] for b in (("leader", "helper") if view == "both" else (view,))]
                        if view == "both" and "world" in X[r.rid]:
                            parts.append(X[r.rid]["world"])
                        xs.append(np.concatenate(parts, axis=1)[2 * FPS::B.STRIDE])
                    return np.concatenate(xs), {t: np.concatenate([r.y[t][2 * FPS::B.STRIDE] for r in rs]) for t in tasks}
                Xtr, Ytr = mk(tr); Xte, Yte = mk(te)
                for t in tasks:
                    pos = Ytr[t].mean()
                    clf = HistGradientBoostingClassifier(max_iter=100, learning_rate=0.15, max_leaf_nodes=31, min_samples_leaf=200, l2_regularization=1.0,
                                                         class_weight={0: 1.0, 1: float(min(50, (1 - pos) / max(pos, 1e-4)))}, random_state=k)
                    clf.fit(Xtr, Ytr[t]); p = np.nan_to_num(clf.predict_proba(Xte)[:, 1], nan=0.5)
                    pool[t][0].append(Yte[t]); pool[t][1].append(p)
                    if 0 < Yte[t].sum() < len(Yte[t]):
                        per_fold[t].append(roc_auc_score(Yte[t], p))
                print(f"    {view} fold {k} done", flush=True)
            res[view] = {t: {"ap_pooled": float(average_precision_score(np.concatenate(pool[t][0]), np.concatenate(pool[t][1]))), "auroc_fold_mean": float(np.mean(per_fold[t])),
                             "auroc_fold_sd": float(np.std(per_fold[t])), "auroc_folds": [float(v) for v in per_fold[t]]} for t in tasks}
        return res

    t0 = time.time()
    print("  (c) 44 recordings, person blocks with gaze (reproduction of baseline_gbdt_grasp.json)", flush=True)
    results["all44_with_gaze"] = run(recs, False, False, False, views)
    print("  (c) 44 recordings, person gaze columns zeroed", flush=True)
    results["all44_gaze_masked"] = run(recs, True, False, False, views)
    wrecs = [r for r in recs if r.w is not None]
    print(f"  (c) {len(wrecs)} recordings with the cross-person block, gaze kept (reproduction of baseline_gbdt_world_cpf.json)", flush=True)
    results["world11_with_gaze"] = run(wrecs, False, True, False, ("both",))
    print("  (c) cross-person block: gaze-derived columns zeroed, person gaze zeroed (head-forward only)", flush=True)
    results["world11_gaze_masked"] = run(wrecs, True, True, True, ("both",))
    results["seconds"] = round(time.time() - t0)
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gbdt", action="store_true", help="also run part (c), the GBDT ablation (~10-20 min)")
    ap.add_argument("--out", default=None, help="write all numbers to this JSON")
    ap.add_argument("--skip-ab", action="store_true")
    a = ap.parse_args()
    recs = load_recordings(PROC, train_ids())
    print(f"{len(recs)} labelled recordings with kinematics; {sum(r.w is not None for r in recs)} with the cross-person block", flush=True)
    out = {"n_recordings": len(recs)}
    if not a.skip_ab:
        t = time.time(); out["a_gaze_vs_forward"] = part_a(recs); print(f"(a) done in {time.time() - t:.0f}s", flush=True)
        A = out["a_gaze_vs_forward"]
        for key in ("pooled_leader", "pooled_helper", "pooled_both_roles"):
            print(f"  {key}: all median {A[key]['all']['median']:.1f} mean {A[key]['all']['mean']:.1f} p90 {A[key]['all']['p90']:.1f} | JA median {A[key]['ja']['median']:.1f} mean {A[key]['ja']['mean']:.1f} p90 {A[key]['ja']['p90']:.1f}")
        for role, m in A["mean_gaze_axis"].items():
            print(f"  mean gaze axis {role}: {m['angle_to_rgb_axis_deg']:.1f} deg off the RGB axis ({m['angle_to_device_z_deg']:.1f} off device +Z); residual median {m['residual_vs_mean_gaze_axis']['median']:.1f} p90 {m['residual_vs_mean_gaze_axis']['p90']:.1f}; vs device +Z median {m['vs_device_z']['median']:.1f}")
        t = time.time(); out["b_auroc_joint_attention"] = part_b(recs); print(f"(b) done in {time.time() - t:.0f}s", flush=True)
        Bres = out["b_auroc_joint_attention"]
        for k in ("forward_leader", "forward_helper", "forward_min", "gaze_leader", "gaze_helper", "gaze_min"):
            print(f"  {k}: per-rec {Bres[k]['auroc_rec_mean']:.3f} +- {Bres[k]['auroc_rec_sd']:.3f}, pooled {Bres[k]['auroc_pooled']:.3f}")
    if a.gbdt:
        out["c_gbdt"] = part_c(recs)
        for cond, res in out["c_gbdt"].items():
            if isinstance(res, dict):
                for view, r in res.items():
                    print(f"  (c) {cond} {view}: " + " | ".join(f"{t} {v['auroc_fold_mean']:.3f}+-{v['auroc_fold_sd']:.3f}" for t, v in r.items()))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)


if __name__ == "__main__":
    main()
