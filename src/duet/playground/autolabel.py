"""Proposed handover / joint-attention events for human review.

A gradient-boosted classifier trained on the 44 labelled CoMind pairs (scripts/train_autolabel.py) is applied to a
REDUCED feature set that both CoMind (Aria hand tracking + Multi-SLAM) and the playground (MediaPipe hands, body2d,
gaze_proxy, world3d, contact) can produce. Everything is computed here so both sides featurise identically.

Per person (11): wrist speed L/R (image widths per second, from the 2D wrist normalised by image width), hand present
L/R, radial offset of each wrist from the image centre (image widths), own L-R wrist distance, held L/R (contact stage;
zeros if absent), angle between the head-forward ray and the partner's head (deg, gaze_proxy) and its validity flag.
Only roll-invariant image quantities are used because the CoMind side projects Aria wrists with an arbitrary in-plane
basis. Inter-person (3): head distance (m), min wrist-wrist distance (m), validity (world3d; zeros if absent).
Each base column is joined by its causal mean / std / max over 1 s and 3 s windows (7 x 25 = 175 features).
Person order is A = first ego, B = second ego; the model is trained on both orderings so it is order-agnostic.

Outputs derived/autolabel/proposals.json ([{event, t0, t1, confidence, giver, receiver, status, source}], reference
seconds) and records_extra.parquet with per-frame probabilities (export prefix "autolabel_").
Intervals come from hysteresis on the probabilities: start when p > hi for 0.5 s, end when p < lo for 1.0 s.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from .episode import Episode

MODEL_PATH = Path(__file__).resolve().parents[3] / "data/playground/models/autolabel_gbdt.pkl"
PERSON_FEATS = ("wrist_speed_L", "wrist_speed_R", "hand_present_L", "hand_present_R", "wrist_r_L", "wrist_r_R",
                "own_hand_dist", "held_L", "held_R", "angle_to_partner_deg", "angle_valid")
INTER_FEATS = ("head_dist_m", "min_wrist_dist_m", "inter_valid")
WINDOWS_S = (1.0, 3.0)
STATS = ("mean", "std", "max")
EVENTS = {"joint_attention": "joint_attention", "handover": "handover"}
MIN_ON_S, MIN_OFF_S = 0.5, 1.0
# Aria Gen1 camera-rgb: 608 px equidistant focal on a 1408 px image. CoMind wrists (device frame, m) are projected with
# this model about the RGB optical axis so their 2D speeds are in the same units as MediaPipe wrists on the same video.
ARIA_FISHEYE_F_NORM = 608.0 / 1408.0


# ----------------------------------------------------------------------------- shared featurisation

def person_base(wrist_uv: np.ndarray, held: np.ndarray | None, angle_deg: np.ndarray | None, fps: float, centre=(0.0, 0.0)) -> np.ndarray:
    """wrist_uv [N,2,2] (L,R; image-width units; NaN when absent), held [N,2] (0/1), angle_deg [N] (NaN invalid) -> [N,11].

    `centre` is the optical axis in the same units (playground: (0.5, 0.5 H/W); CoMind projections are already centred).
    Only roll-invariant quantities are used (speed, radial offset, distances) so an arbitrary in-plane basis is fine."""
    n = wrist_uv.shape[0]
    present = np.isfinite(wrist_uv[:, :, 0]) & np.isfinite(wrist_uv[:, :, 1])
    uv = np.where(present[:, :, None], wrist_uv - np.asarray(centre, np.float32), 0.0)
    d = np.diff(uv, axis=0, prepend=uv[:1])
    speed = np.linalg.norm(d, axis=-1) * fps
    both = present & np.concatenate([present[:1], present[:-1]])
    speed = np.where(both, speed, 0.0)
    radial = np.linalg.norm(uv, axis=-1) * present
    own = np.linalg.norm(uv[:, 0] - uv[:, 1], axis=1) * (present[:, 0] & present[:, 1])
    held = np.zeros((n, 2), np.float32) if held is None else np.nan_to_num(np.asarray(held, np.float32)[:n]) > 0
    if len(held) < n:
        held = np.concatenate([held, np.zeros((n - len(held), 2), bool)])
    ang = np.full(n, np.nan, np.float32) if angle_deg is None else np.asarray(angle_deg, np.float32)[:n]
    if len(ang) < n:
        ang = np.concatenate([ang, np.full(n - len(ang), np.nan, np.float32)])
    av = np.isfinite(ang)
    cols = [speed[:, 0], speed[:, 1], present[:, 0], present[:, 1], radial[:, 0], radial[:, 1], own,
            held[:, 0], held[:, 1], np.where(av, ang, 0.0), av]
    return np.stack([np.asarray(c, np.float32) for c in cols], axis=1)


def inter_base(head_dist_m: np.ndarray | None, min_wrist_dist_m: np.ndarray | None, n: int) -> np.ndarray:
    hd = np.full(n, np.nan, np.float32) if head_dist_m is None else np.asarray(head_dist_m, np.float32)[:n]
    mw = np.full(n, np.nan, np.float32) if min_wrist_dist_m is None else np.asarray(min_wrist_dist_m, np.float32)[:n]
    hd = np.concatenate([hd, np.full(n - len(hd), np.nan, np.float32)]); mw = np.concatenate([mw, np.full(n - len(mw), np.nan, np.float32)])
    valid = np.isfinite(hd) & np.isfinite(mw) & (mw > 0)
    return np.stack([np.where(np.isfinite(hd), hd, 0.0), np.where(np.isfinite(mw), mw, 0.0), valid], axis=1).astype(np.float32)


def window_features(base: np.ndarray, fps: float) -> np.ndarray:
    """[N, D] -> [N, D * (1 + 3 * len(WINDOWS_S))]: base, then causal mean/std/max over each window."""
    base = np.nan_to_num(np.asarray(base, np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    n = len(base); parts = [base]
    for w_s in WINDOWS_S:
        w = max(1, int(round(w_s * fps)))
        pad = np.repeat(base[:1], w - 1, axis=0); xp = np.concatenate([pad, base], axis=0)
        v = np.lib.stride_tricks.sliding_window_view(xp, w, axis=0)  # [N, D, w]
        parts += [v.mean(-1), v.std(-1), v.max(-1)]
    return np.nan_to_num(np.concatenate(parts, axis=1), nan=0.0).astype(np.float32)


def feature_names() -> list[str]:
    base = [f"A_{f}" for f in PERSON_FEATS] + [f"B_{f}" for f in PERSON_FEATS] + list(INTER_FEATS)
    out = list(base)
    for w_s in WINDOWS_S:
        for st in STATS:
            out += [f"{b}_{st}{w_s:g}s" for b in base]
    return out


def assemble(base_a: np.ndarray, base_b: np.ndarray, inter: np.ndarray, fps: float) -> np.ndarray:
    return window_features(np.concatenate([base_a, base_b, inter], axis=1), fps)


def project_equidistant(p: np.ndarray, axis: np.ndarray, f_norm: float = ARIA_FISHEYE_F_NORM) -> np.ndarray:
    """Points [.., 3] in a device frame -> [.., 2] image-width units under an equidistant fisheye about `axis`.

    The in-plane basis is arbitrary (speeds and distances are roll-invariant). Points behind the camera -> NaN."""
    axis = np.asarray(axis, np.float64); axis = axis / np.linalg.norm(axis)
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(axis, ref); e1 /= np.linalg.norm(e1); e2 = np.cross(axis, e1)
    z = p @ axis; x = p @ e1; y = p @ e2
    r = np.hypot(x, y); theta = np.arctan2(r, z)
    with np.errstate(invalid="ignore", divide="ignore"):
        u = f_norm * theta * x / r; v = f_norm * theta * y / r
    u = np.where(r > 0, u, 0.0); v = np.where(r > 0, v, 0.0)
    out = np.stack([u, v], axis=-1)
    out[z <= 0] = np.nan
    return out


# ----------------------------------------------------------------------------- hysteresis

def hysteresis_intervals(p: np.ndarray, fps: float, hi: float, lo: float, min_on_s: float = MIN_ON_S, min_off_s: float = MIN_OFF_S) -> list[tuple[int, int]]:
    """Frame intervals [k0, k1) that start when p > hi for min_on_s and end when p < lo for min_off_s."""
    p = np.asarray(p, float); n = len(p)
    on_f = max(1, int(round(min_on_s * fps))); off_f = max(1, int(round(min_off_s * fps)))
    above = p > hi; below = p < lo
    run_above = np.zeros(n, int); run_below = np.zeros(n, int)
    for k in range(n - 1, -1, -1):  # forward run lengths
        run_above[k] = run_above[k + 1] + 1 if (above[k] and k + 1 < n) else int(above[k])
        run_below[k] = run_below[k + 1] + 1 if (below[k] and k + 1 < n) else int(below[k])
    out = []; k = 0; state = False; k0 = 0
    while k < n:
        if not state:
            if run_above[k] >= on_f:
                state, k0 = True, k; k += on_f; continue
        elif run_below[k] >= off_f:
            out.append((k0, k)); state = False; k += off_f; continue
        k += 1
    if state:
        out.append((k0, n))
    return out


# ----------------------------------------------------------------------------- playground side

def _n_frames(ep: Episode) -> int:
    counts = [len(list((ep.derived / "frames" / s.name).glob("*.jpg"))) for s in ep.streams]
    return min(counts) if counts else 0


def playground_person_base(ep: Episode, s, n: int) -> np.ndarray:
    from .gaze_proxy import _image_size
    w, h = _image_size(ep, s)
    uv = np.full((n, 2, 2), np.nan, np.float32)
    z = ep.derived / "hands" / f"{s.name}.npz"
    if z.exists():
        lm = np.load(z)["lm2d"][:n, :, 0, :]  # MediaPipe landmark 0 = wrist
        uv[: len(lm)] = lm / float(w)
    held = None
    z = ep.derived / "contact" / f"{s.name}.npz"
    if z.exists():
        c = np.load(z)
        if "held" in c.files:  # contact stage: object name per hand ("" = nothing held); older/other layouts may be 0/1
            hd = c["held"][:n]
            held = (hd != "") if hd.dtype.kind in "US" else hd
    ang = None
    z = ep.derived / "gaze_proxy" / f"{s.name}.npz"
    if z.exists():
        ang = np.load(z)["angle_to_partner_deg"][:n]
    return person_base(uv, held, ang, ep.proc_fps, centre=(0.5, 0.5 * h / w))


def playground_inter_base(ep: Episode, n: int) -> np.ndarray:
    z = ep.derived / "world3d" / "world3d.npz"
    if not z.exists():
        return inter_base(None, None, n)
    w = np.load(z)
    return inter_base(w["feat_head_dist_m"] if "feat_head_dist_m" in w.files else None, w["feat_min_wrist_dist_m"] if "feat_min_wrist_dist_m" in w.files else None, n)


def playground_features(ep: Episode, n: int) -> tuple[np.ndarray, list[str]]:
    """X [n, F] for the episode and the person names in A/B order (B is all-zero if there is one ego)."""
    egos = ep.egos()
    bases = [playground_person_base(ep, s, n) for s in egos[:2]]
    names = [(s.person or s.name) for s in egos[:2]]
    while len(bases) < 2:
        bases.append(np.zeros((n, len(PERSON_FEATS)), np.float32)); names.append(None)
    return assemble(bases[0], bases[1], playground_inter_base(ep, n), ep.proc_fps), names


def load_model(path: Path = MODEL_PATH) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def propose(X: np.ndarray, model: dict, fps: float, t0_s: float, names: list) -> tuple[list[dict], dict[str, np.ndarray]]:
    probs = {}; proposals = []
    for event, task in (("joint_attention", "ja_active"), ("handover", "handover_active")):
        clf = model["models"].get(task)
        if clf is None:
            continue
        p = np.nan_to_num(clf.predict_proba(X)[:, 1], nan=0.0); probs[f"p_{event}"] = p.astype(np.float32)
        th = model["thresholds"][task]
        for k0, k1 in hysteresis_intervals(p, fps, th["hi"], th["lo"]):
            item = {"event": event, "t0": round(t0_s + k0 / fps, 3), "t1": round(t0_s + k1 / fps, 3), "confidence": round(float(p[k0:k1].mean()), 4),
                    "giver": None, "receiver": None, "status": "pending", "source": "autolabel_gbdt"}
            if event == "handover" and model["models"].get("handover_direction") is not None and names[1] is not None:
                pd_ = float(np.nan_to_num(model["models"]["handover_direction"].predict_proba(X[k0:k1])[:, 1], nan=0.5).mean())
                item["giver"], item["receiver"] = (names[0], names[1]) if pd_ >= 0.5 else (names[1], names[0])
                item["p_a_gives"] = round(pd_, 4)
            proposals.append(item)
    proposals.sort(key=lambda d: d["t0"])
    return proposals, probs


def autolabel(ep: Episode) -> None:
    ep.set_status("autolabel", "running")
    if not MODEL_PATH.exists():
        ep.set_status("autolabel", "skipped", f"no model at {MODEL_PATH}"); return
    n = _n_frames(ep)
    if n == 0:
        ep.set_status("autolabel", "skipped", "no extracted frames"); return
    model = load_model(MODEL_PATH)
    X, names = playground_features(ep, n)
    if X.shape[1] != len(model["feature_names"]):
        ep.set_status("autolabel", "failed", f"feature mismatch: episode {X.shape[1]} vs model {len(model['feature_names'])}"); return
    proposals, probs = propose(X, model, ep.proc_fps, ep.common_start_s, names)
    out = ep.derived / "autolabel"; out.mkdir(exist_ok=True)
    json.dump(proposals, open(out / "proposals.json", "w"), indent=1)
    pd.DataFrame(probs).to_parquet(out / "records_extra.parquet", index=False)
    counts = {e: sum(p["event"] == e for p in proposals) for e in EVENTS}
    ep.set_status("autolabel", "done", f"{counts['joint_attention']} joint-attention + {counts['handover']} handover proposals (model {model.get('meta', {}).get('trained', '?')})")
