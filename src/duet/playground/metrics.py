"""Operator metrics per session: the numbers an operator sees for a recorded episode.

Writes derived/metrics/session.json and derived/metrics/records_extra.parquet (per-frame hand speed per
person, exported as metrics_hand_speed_<person> etc. by the export stage).

Hand speed per person (the signal everything else is built on)
    world3d hands3d_<stream> wrist (landmark 0) when >= 20% of frames have it: metres/s.
    Otherwise the MediaPipe 2D wrist in that person's ego view, divided by the frame width: frame-widths/s
    ("fw/s"). Ego cameras move with the head, so 2D speed mixes hand and head motion; treat as a proxy.
    Max over both hands, NaN when no hand is visible, 3-frame moving average.

Handover events, in this priority (first source that yields >= 1 event wins)
    1. derived/contact/events.json  {"t","stream","person","hand","object","type": "grasp"|"release"}:
       person A releases object O after holding it >= 0.5 s and person B != A grasps O within 1.5 s and keeps it
       >= 0.5 s (each release used once; overlapping same-object events merged). The hold minimum filters the
       grasp/release chatter that two people working over the same bowl produce.
    2. derived/autolabel/proposals.json {"event","t0","t1","confidence","giver","receiver","status"}:
       events whose name contains "handover" and status != "rejected".
    3. world3d feat_min_wrist_dist_m < 0.25 m for >= 0.3 s.
    4. 2D wrist proximity of the two largest people in the exo view that sees two people most often:
       min inter-person wrist distance < 0.5 torso lengths for >= 0.3 s (windows padded by 1 s each side).
    For 3 and 4 giver/receiver are not observable: the person whose reach onset comes first is called the
    giver (flagged giver_basis = "onset_order"; receiver_response_s is then >= 0 by construction).

Per handover (ported from scripts/handover_quality_metrics.py, symmetric giver/receiver windows)
    giver_reach_onset_s    first run of ONSET_RUN_S where the giver's speed exceeds baseline mean + 2 sd + floor,
                           searched from t0 - SEARCH_LEAD_S for SEARCH_AHEAD_S; baseline = BASELINE_S before that
    receiver_response_s    receiver onset - giver onset, same windows (negative = receiver moved first)
    transfer_s             min inter-wrist distance in [t0, t1] when metric distance exists, else the giver's
                           speed minimum after its peak
    giver_hold_s           seconds between the giver's speed peak and transfer with giver speed below STILL
    synchrony              max normalised cross-correlation of the two speed profiles in [t0 - 1 s, t1 + 1 s], lag within +-2 s

Session
    idle fraction per person: frames in runs >= 2 s with speed below STILL (frames without a visible hand are not idle)
    hands-visible fraction per person; speaking fraction per person from derived/speech/records_extra.parquet
    (speaking_<person>); session synchrony = same cross-correlation on the whole signals; per-minute table.

Coordination score (0-100), the mean of the available components x 100:
    C_sync   = clip(session synchrony, 0, 1)
    C_resp   = 1 - clip(median |receiver_response_s| / 2 s, 0, 1)      (needs >= 1 measurable handover)
    C_hold   = 1 - clip(median giver_hold_s / 3 s, 0, 1)               (needs >= 1 handover)
    C_engage = 1 - mean idle fraction over persons
    Without handovers the score is C_sync and C_engage only and is flagged partial.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .episode import Episode

STILL = {"m/s": 0.10, "fw/s": 0.06}     # "still" / idle threshold per speed unit
ONSET_FLOOR = {"m/s": 0.05, "fw/s": 0.03}
IDLE_MIN_S = 2.0
SEARCH_LEAD_S, BASELINE_S, SEARCH_AHEAD_S, ONSET_RUN_S = 1.5, 3.0, 6.0, 5 / 30
SYNC_LAG_S = 2.0
PROX_M, PROX_MIN_S, PROX_TORSO, PAD_S = 0.25, 0.3, 0.5, 1.0
CONTACT_PAIR_S, CONTACT_MIN_HOLD_S = 1.5, 0.5


# ----------------------------------------------------------------------------- signals

def smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    """NaN-aware moving average."""
    v = np.isfinite(x).astype(float); xf = np.where(np.isfinite(x), x, 0.0); ker = np.ones(k)
    num = np.convolve(xf, ker, "same"); den = np.convolve(v, ker, "same")
    out = np.full_like(x, np.nan, dtype=float); ok = den > 0; out[ok] = num[ok] / den[ok]; return out


def wrist_speed(wrist: np.ndarray, fps: float) -> np.ndarray:
    """wrist [N, 2 hands, D] -> speed [N] (max over hands, NaN where neither hand has two consecutive positions)."""
    d = np.linalg.norm(np.diff(wrist, axis=0), axis=-1) * fps          # [N-1, 2]
    d = np.concatenate([d[:1], d], 0)                                    # frame 0 repeats frame 1
    best = np.nanmax(np.where(np.isfinite(d), d, -np.inf), axis=1); best[~np.isfinite(best)] = np.nan
    return best


def hand_speed_signals(ep: Episode, n: int) -> dict[str, dict]:
    out: dict[str, dict] = {}
    w3 = ep.derived / "world3d" / "world3d.npz"; w3d = np.load(w3) if w3.exists() else None
    for s in ep.egos():
        person = s.person or s.name; z = ep.derived / "hands" / f"{s.name}.npz"
        if not z.exists():
            continue
        h = np.load(z); lm2d = h["lm2d"][:n]; visible = np.isfinite(lm2d[:, :, 0, 0]).any(1)
        key = f"hands3d_{s.name}"
        if w3d is not None and key in w3d.files and np.isfinite(w3d[key][:n, :, 0, 0]).any(1).mean() >= 0.2:
            sp = wrist_speed(w3d[key][:n, :, 0, :], ep.proc_fps); unit, source = "m/s", f"world3d {key} wrist"
        else:
            b = ep.derived / "body2d" / f"{s.name}.npz"; img_w = float(np.load(b)["img_w"]) if b.exists() else float(ep.proc_size)
            sp = wrist_speed(lm2d[:, :, 0, :] / img_w, ep.proc_fps); unit, source = "fw/s", f"hands {s.name} lm2d wrist / frame width ({img_w:g} px)"
        m = min(len(sp), n); full = np.full(n, np.nan); full[:m] = sp[:m]; vis = np.zeros(n, bool); vis[:m] = visible[:m]
        out[person] = {"speed": smooth(full), "visible": vis, "unit": unit, "source": source}
    return out


def onset_after(speed: np.ndarray, start: int, base_lo: int, base_hi: int, min_run: int, max_ahead: int, floor: float) -> int | None:
    """First index >= start where speed stays above baseline mean + 2 sd + floor for min_run frames (NaN counts as 0)."""
    sp = np.where(np.isfinite(speed), speed, 0.0); base = sp[max(0, base_lo): max(base_lo + 1, base_hi)]
    if len(base) < 3:
        base = sp[np.isfinite(speed)] if np.isfinite(speed).any() else np.zeros(1)
    thr = base.mean() + 2 * base.std() + floor
    above = sp[max(0, start): max(0, start) + max_ahead] > thr; run = 0
    for i, a in enumerate(above):
        run = run + 1 if a else 0
        if run >= min_run:
            return max(0, start) + i - min_run + 1
    return None


def xcorr_sync(a: np.ndarray, b: np.ndarray, fps: float, max_lag_s: float = SYNC_LAG_S) -> tuple[float, float]:
    """Max Pearson correlation of the two signals over lags within +-max_lag_s (each lag's overlap is normalised on its own, so
    the value is bounded by 1; scripts/handover_quality_metrics.py normalised once over the whole window and could exceed 1).
    (nan, nan) when either signal is flat. Positive lag = b lags a."""
    a = np.where(np.isfinite(a), a, 0.0).astype(float); b = np.where(np.isfinite(b), b, 0.0).astype(float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan"), float("nan")
    L = int(round(max_lag_s * fps)); best, best_lag = -np.inf, 0
    for lag in range(-L, L + 1):
        if lag >= 0:
            x, y = a[: len(a) - lag] if lag else a, b[lag:]
        else:
            x, y = a[-lag:], b[: len(b) + lag]
        if len(x) < 3 or x.std() < 1e-9 or y.std() < 1e-9:
            continue
        c = float(np.corrcoef(x, y)[0, 1])
        if c > best:
            best, best_lag = c, lag
    return (best, best_lag / fps) if np.isfinite(best) else (float("nan"), float("nan"))


def idle_mask(speed: np.ndarray, thr: float, fps: float, min_s: float = IDLE_MIN_S) -> np.ndarray:
    """Frames belonging to a run of >= min_s where speed < thr (NaN = hand not visible = not idle)."""
    still = np.isfinite(speed) & (speed < thr); out = np.zeros(len(speed), bool); need = max(1, int(round(min_s * fps))); i = 0
    while i < len(still):
        if still[i]:
            j = i
            while j < len(still) and still[j]:
                j += 1
            if j - i >= need:
                out[i:j] = True
            i = j
        else:
            i += 1
    return out


# ----------------------------------------------------------------------------- handover candidates

def _hold_spans(events: list[dict]) -> list[dict]:
    """grasp..release spans per (person, hand, object): {'person','hand','object','t_grasp','t_release' (None if never released)}."""
    open_: dict[tuple, dict] = {}; spans = []
    for e in sorted(events, key=lambda e: (e["t"], e.get("type") == "grasp")):  # releases before grasps at equal t
        key = (e.get("person"), e.get("hand"), e.get("object"))
        if e.get("type") == "grasp":
            if key in open_:
                spans.append(open_.pop(key))
            open_[key] = {"person": key[0], "hand": key[1], "object": key[2], "t_grasp": float(e["t"]), "t_release": None}
        elif e.get("type") == "release" and key in open_:
            s = open_.pop(key); s["t_release"] = float(e["t"]); spans.append(s)
    return spans + list(open_.values())


def handovers_from_contact(events: list[dict], pair_s: float = CONTACT_PAIR_S, min_hold_s: float = CONTACT_MIN_HOLD_S) -> list[dict]:
    """A handover = person A releases object O after holding it >= min_hold_s, and person B != A grasps O within pair_s
    and keeps it >= min_hold_s. Each release is used at most once; overlapping events on the same object are merged."""
    spans = _hold_spans(events)
    gives = [s for s in spans if s["t_release"] is not None and s["t_release"] - s["t_grasp"] >= min_hold_s]
    takes = [s for s in spans if s["t_release"] is None or s["t_release"] - s["t_grasp"] >= min_hold_s]
    used: set[int] = set(); out = []
    for tk in sorted(takes, key=lambda s: s["t_grasp"]):
        cands = [(abs(tk["t_grasp"] - gv["t_release"]), i) for i, gv in enumerate(gives)
                 if i not in used and gv["object"] == tk["object"] and gv["person"] != tk["person"] and abs(tk["t_grasp"] - gv["t_release"]) <= pair_s]
        if not cands:
            continue
        _, i = min(cands); used.add(i); gv = gives[i]
        out.append({"t0": round(min(gv["t_release"], tk["t_grasp"]) - 0.3, 3), "t1": round(max(gv["t_release"], tk["t_grasp"]) + 0.3, 3), "giver": gv["person"], "receiver": tk["person"],
                    "object": tk["object"], "confidence": None, "source": "contact", "giver_basis": "contact_events"})
    merged: list[dict] = []
    for e in sorted(out, key=lambda e: e["t0"]):
        m = merged[-1] if merged else None
        if m and m["object"] == e["object"] and m["giver"] == e["giver"] and m["receiver"] == e["receiver"] and e["t0"] <= m["t1"]:
            m["t1"] = max(m["t1"], e["t1"])
        else:
            merged.append(dict(e))
    return merged


def handovers_from_autolabel(proposals: list[dict]) -> list[dict]:
    return [{"t0": float(p["t0"]), "t1": float(p["t1"]), "giver": p.get("giver"), "receiver": p.get("receiver"), "object": p.get("object"),
             "confidence": p.get("confidence"), "source": "autolabel", "giver_basis": "autolabel"}
            for p in proposals if "handover" in str(p.get("event", "")).lower() and p.get("status") != "rejected" and p.get("t0") is not None]


def _windows(mask: np.ndarray, min_frames: int, merge_frames: int) -> list[tuple[int, int]]:
    runs = []; i = 0
    while i < len(mask):
        if mask[i]:
            j = i
            while j < len(mask) and mask[j]:
                j += 1
            if j - i >= min_frames:
                if runs and i - runs[-1][1] <= merge_frames:
                    runs[-1] = (runs[-1][0], j)
                else:
                    runs.append((i, j))
            i = j
        else:
            i += 1
    return runs


def handovers_from_distance(dist: np.ndarray, fps: float, t0_s: float, thr: float, source: str, pad_s: float = PAD_S) -> list[dict]:
    mask = np.isfinite(dist) & (dist < thr); out = []
    for i, j in _windows(mask, max(1, int(round(PROX_MIN_S * fps))), int(round(1.0 * fps))):
        out.append({"t0": round(t0_s + max(0, i - pad_s * fps) / fps, 3), "t1": round(t0_s + min(len(dist), j + pad_s * fps) / fps, 3), "giver": None, "receiver": None,
                    "object": None, "confidence": None, "source": source, "giver_basis": "onset_order", "min_dist": float(np.nanmin(dist[i:j]))})
    return out


def exo_wrist_proximity(ep: Episode, n: int) -> tuple[np.ndarray, str | None]:
    """Per frame: min inter-person wrist distance / mean torso length, in the exo view that sees two people most often."""
    best, best_cnt, best_k = None, 0, None
    for s in ep.exos():
        z = ep.derived / "body2d" / f"{s.name}.npz"
        if not z.exists():
            continue
        k = np.load(z)["kpts"][:n]; cnt = int((np.isfinite(k[:, :, 0, 0]).sum(1) >= 2).sum())
        if cnt > best_cnt:
            best, best_cnt, best_k = s.name, cnt, k
    d = np.full(n, np.nan)
    if best_k is None:
        return d, None
    for f in range(min(n, len(best_k))):
        kp = best_k[f]; present = np.where(np.isfinite(kp[:, 0, 0]))[0]
        if len(present) < 2:
            continue
        area = [(np.nanmax(kp[p, :, 0]) - np.nanmin(kp[p, :, 0])) * (np.nanmax(kp[p, :, 1]) - np.nanmin(kp[p, :, 1])) for p in present]
        a, b = present[np.argsort(area)[-2:]]
        torso = []
        for p in (a, b):
            if (kp[p, [5, 6, 11, 12], 2] > 0.3).all():
                torso.append(np.linalg.norm(kp[p, [5, 6], :2].mean(0) - kp[p, [11, 12], :2].mean(0)))
        if not torso:
            continue
        wa = [kp[a, i, :2] for i in (9, 10) if kp[a, i, 2] > 0.3]; wb = [kp[b, i, :2] for i in (9, 10) if kp[b, i, 2] > 0.3]
        if wa and wb:
            d[f] = min(np.linalg.norm(x - y) for x in wa for y in wb) / max(1e-6, float(np.mean(torso)))
    return d, best


# ----------------------------------------------------------------------------- per-handover metrics

def handover_metrics(ev: dict, sig: dict[str, dict], dist_m: np.ndarray | None, fps: float, t0_s: float, persons: list[str]) -> dict:
    s0 = int(round((ev["t0"] - t0_s) * fps)); e0 = int(round((ev["t1"] - t0_s) * fps)); n = len(next(iter(sig.values()))["speed"]) if sig else 0
    s0 = int(np.clip(s0, 0, max(0, n - 1))); e0 = int(np.clip(e0, s0, max(0, n - 1)))
    lead, base_n, ahead, run = int(round(SEARCH_LEAD_S * fps)), int(round(BASELINE_S * fps)), int(round(SEARCH_AHEAD_S * fps)), max(2, int(round(ONSET_RUN_S * fps)))
    search_from = s0 - lead; base_lo, base_hi = s0 - lead - base_n, s0 - lead
    onsets = {}
    for p, sg in sig.items():
        onsets[p] = onset_after(sg["speed"], search_from, base_lo, base_hi, run, ahead, ONSET_FLOOR[sg["unit"]])
    giver, recv, basis = ev.get("giver"), ev.get("receiver"), ev.get("giver_basis", "given")
    if (giver not in sig or recv not in sig) and len(persons) >= 2:
        found = [(onsets[p], p) for p in persons if p in sig and onsets[p] is not None]
        if found:
            giver = min(found)[1]; recv = next(p for p in persons if p != giver); basis = "onset_order"
        else:
            giver, recv = persons[0], persons[1]; basis = "unknown"
    out = {**{k: v for k, v in ev.items() if k != "min_dist"}, "giver": giver, "receiver": recv, "giver_basis": basis, "duration_s": round((e0 - s0) / fps, 3),
           "giver_reach_onset_s": None, "receiver_response_s": None, "transfer_s": None, "giver_hold_s": None, "synchrony": None, "synchrony_lag_s": None, "min_wrist_dist_m": ev.get("min_dist") if ev.get("source") == "world3d" else None}
    if giver not in sig or recv not in sig:
        return out
    gs = np.where(np.isfinite(sig[giver]["speed"]), sig[giver]["speed"], 0.0); rs = np.where(np.isfinite(sig[recv]["speed"]), sig[recv]["speed"], 0.0)
    g_on, r_on = onsets.get(giver), onsets.get(recv)
    out["giver_reach_onset_s"] = None if g_on is None else round((g_on - s0) / fps, 3)
    out["receiver_response_s"] = None if (g_on is None or r_on is None) else round((r_on - g_on) / fps, 3)
    if dist_m is not None and np.isfinite(dist_m[s0: e0 + 1]).any():
        dm = np.where(np.isfinite(dist_m[s0: e0 + 1]), dist_m[s0: e0 + 1], np.inf); t_tr = s0 + int(np.argmin(dm)); out["min_wrist_dist_m"] = float(dm.min())
    else:
        pk = s0 + int(np.argmax(gs[s0: e0 + 1])); t_tr = pk + int(np.argmin(gs[pk: e0 + 1])) if pk < e0 else pk
    pk = s0 + int(np.argmax(gs[s0: e0 + 1])); still = STILL[sig[giver]["unit"]]
    out["transfer_s"] = round((t_tr - s0) / fps, 3); out["giver_hold_s"] = round(float((gs[pk: t_tr + 1] < still).sum() / fps), 3) if t_tr > pk else 0.0
    w0, w1 = max(0, s0 - int(fps)), min(n, e0 + int(fps) + 1); sync, lag = xcorr_sync(gs[w0:w1], rs[w0:w1], fps)
    out["synchrony"] = None if not np.isfinite(sync) else round(sync, 3); out["synchrony_lag_s"] = None if not np.isfinite(lag) else round(lag, 3)
    return out


# ----------------------------------------------------------------------------- session

def coordination_score(session_sync: float | None, responses: list[float], holds: list[float], idle: dict[str, float]) -> dict:
    comps = {}
    if session_sync is not None and np.isfinite(session_sync):
        comps["C_sync"] = float(np.clip(session_sync, 0, 1))
    if responses:
        comps["C_resp"] = float(1 - np.clip(np.median(np.abs(responses)) / 2.0, 0, 1))
    if holds:
        comps["C_hold"] = float(1 - np.clip(np.median(holds) / 3.0, 0, 1))
    if idle:
        comps["C_engage"] = float(1 - np.mean(list(idle.values())))
    score = 100 * float(np.mean(list(comps.values()))) if comps else None
    return {"score": None if score is None else round(score, 1), "components": {k: round(v, 3) for k, v in comps.items()}, "partial": not ({"C_resp", "C_hold"} <= set(comps)),
            "formula": "100 * mean(C_sync = clip(session synchrony, 0, 1); C_resp = 1 - clip(median |receiver_response_s| / 2 s, 0, 1); C_hold = 1 - clip(median giver_hold_s / 3 s, 0, 1); C_engage = 1 - mean idle fraction); components without data are left out"}


def _n_frames(ep: Episode) -> int:
    rec = ep.derived / "records.parquet"
    if rec.exists():
        return len(pd.read_parquet(rec, columns=["t_s"]))
    lens = [len(np.load(ep.derived / "hands" / f"{s.name}.npz")["score"]) for s in ep.egos() if (ep.derived / "hands" / f"{s.name}.npz").exists()]
    lens += [len(np.load(ep.derived / "body2d" / f"{s.name}.npz")["kpts"]) for s in ep.streams if (ep.derived / "body2d" / f"{s.name}.npz").exists()]
    return min(lens) if lens else int((ep.common_end_s - ep.common_start_s) * ep.proc_fps)


def compute(ep: Episode) -> tuple[dict, pd.DataFrame]:
    fps = ep.proc_fps; n = _n_frames(ep); t0 = ep.common_start_s; dur = ep.common_end_s - ep.common_start_s; notes = []
    persons = []
    for s in ep.egos():
        p = s.person or s.name
        if p not in persons:
            persons.append(p)
    sig = hand_speed_signals(ep, n)
    # handover candidates
    events, source = [], "none"; dist_m = None
    p = ep.derived / "contact" / "events.json"
    if p.exists():
        try:
            raw = json.load(open(p)); events = handovers_from_contact(raw.get("events", []) if isinstance(raw, dict) else raw); source = "contact" if events else source
        except Exception as e:  # noqa: BLE001
            notes.append(f"contact events unreadable: {e}")
    p = ep.derived / "autolabel" / "proposals.json"
    if not events and p.exists():
        try:
            raw = json.load(open(p)); events = handovers_from_autolabel(raw.get("proposals", raw) if isinstance(raw, dict) else raw); source = "autolabel" if events else source
        except Exception as e:  # noqa: BLE001
            notes.append(f"autolabel proposals unreadable: {e}")
    w3 = ep.derived / "world3d" / "world3d.npz"
    if w3.exists():
        w = np.load(w3)
        if "feat_min_wrist_dist_m" in w.files and np.isfinite(w["feat_min_wrist_dist_m"][:n]).any():
            dist_m = np.full(n, np.nan); d = w["feat_min_wrist_dist_m"][:n]; dist_m[: len(d)] = d
    if not events and dist_m is not None:
        events = [{**e, "source": "world3d"} for e in handovers_from_distance(dist_m, fps, t0, PROX_M, "world3d")]; source = "world3d" if events else source
    if not events:
        prox, view = exo_wrist_proximity(ep, n)
        if view is not None and np.isfinite(prox).any():
            events = handovers_from_distance(prox, fps, t0, PROX_TORSO, f"exo2d:{view}"); source = f"exo2d:{view}" if events else "none"
            notes.append(f"handover candidates from 2D wrist proximity in {view} (< {PROX_TORSO} torso lengths for >= {PROX_MIN_S} s); {np.isfinite(prox).mean():.0%} of frames had both people's wrists")
        else:
            notes.append("no handover source: no contact events, autolabel proposals, world3d distances or exo body2d")
    hand = [handover_metrics(e, sig, dist_m, fps, t0, persons) for e in events]
    for i, h in enumerate(hand):
        h["id"] = i
    # per-person signals
    per_person = {}; idle_masks = {}
    for p_, sg in sig.items():
        im = idle_mask(sg["speed"], STILL[sg["unit"]], fps); idle_masks[p_] = im
        per_person[p_] = {"unit": sg["unit"], "source": sg["source"], "still_threshold": STILL[sg["unit"]], "hands_visible_fraction": round(float(sg["visible"].mean()), 3),
                          "idle_fraction": round(float(im.mean()), 3), "mean_speed": round(float(np.nanmean(sg["speed"])), 4) if np.isfinite(sg["speed"]).any() else None}
    speaking = None; speak_cols = {}
    sp = ep.derived / "speech" / "records_extra.parquet"
    if sp.exists():
        try:
            sdf = pd.read_parquet(sp).reindex(range(n)); speaking = {}
            for c in sdf.columns:
                key = c.replace("speech_", "", 1) if c.startswith("speech_") else c
                if key.startswith("speaking_"):
                    v = sdf[c].fillna(False).astype(bool).to_numpy(); speak_cols[key[9:]] = v; speaking[key[9:]] = round(float(v.mean()), 3)
        except Exception as e:  # noqa: BLE001
            notes.append(f"speech records_extra unreadable: {e}")
    # synchrony on the whole session
    session_sync, session_lag = (float("nan"), float("nan"))
    if len(persons) >= 2 and all(p_ in sig for p_ in persons[:2]):
        session_sync, session_lag = xcorr_sync(sig[persons[0]]["speed"], sig[persons[1]]["speed"], fps)
    # per-minute table
    per_minute = []
    for m in range(int(np.ceil(dur / 60.0))):
        a, b = m * 60.0, min(dur, (m + 1) * 60.0); fa, fb = int(a * fps), max(int(a * fps) + 1, int(b * fps))
        row = {"minute": m, "t0_s": round(t0 + a, 2), "t1_s": round(t0 + b, 2), "handovers": sum(1 for h in hand if a <= h["t0"] - t0 < b),
               "idle_fraction": round(float(np.mean([im[fa:fb].mean() for im in idle_masks.values()])), 3) if idle_masks else None,
               "hands_visible_fraction": round(float(np.mean([sg["visible"][fa:fb].mean() for sg in sig.values()])), 3) if sig else None,
               "speech_fraction": round(float(np.mean([v[fa:fb].mean() for v in speak_cols.values()])), 3) if speak_cols else None}
        per_minute.append(row)
    responses = [h["receiver_response_s"] for h in hand if h["receiver_response_s"] is not None]; holds = [h["giver_hold_s"] for h in hand if h["giver_hold_s"] is not None]
    syncs = [h["synchrony"] for h in hand if h["synchrony"] is not None]
    session = {"episode": ep.name, "computed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "duration_s": round(dur, 2), "fps": fps, "n_frames": n, "persons": persons,
               "hand_speed": per_person, "speaking_fraction": speaking,
               "handovers": {"count": len(hand), "rate_per_min": round(len(hand) / max(dur, 1e-9) * 60, 3), "source": source, "events": hand,
                             "median_receiver_response_s": round(float(np.median(responses)), 3) if responses else None, "median_giver_hold_s": round(float(np.median(holds)), 3) if holds else None,
                             "anticipation_fraction": round(float(np.mean(np.array(responses) < 0)), 3) if responses else None},
               "synchrony": {"session": None if not np.isfinite(session_sync) else round(session_sync, 3), "session_lag_s": None if not np.isfinite(session_lag) else round(session_lag, 3),
                             "median_handover": round(float(np.median(syncs)), 3) if syncs else None},
               "per_minute": per_minute, "coordination_score": coordination_score(session_sync, responses, holds, {p_: v["idle_fraction"] for p_, v in per_person.items()}),
               "definitions": {"still_threshold": STILL, "idle_min_s": IDLE_MIN_S, "search_lead_s": SEARCH_LEAD_S, "baseline_s": BASELINE_S, "search_ahead_s": SEARCH_AHEAD_S, "sync_max_lag_s": SYNC_LAG_S,
                               "world3d_proximity_m": PROX_M, "exo2d_proximity_torso": PROX_TORSO, "proximity_min_s": PROX_MIN_S, "contact_pair_s": CONTACT_PAIR_S, "contact_min_hold_s": CONTACT_MIN_HOLD_S},
               "notes": notes}
    cols = {}
    for p_, sg in sig.items():
        cols[f"hand_speed_{p_}"] = sg["speed"].astype(np.float32); cols[f"idle_{p_}"] = idle_masks[p_]; cols[f"hand_visible_{p_}"] = sg["visible"]
    return session, pd.DataFrame(cols, index=range(n))


def metrics(ep: Episode) -> None:
    ep.set_status("metrics", "running")
    session, extra = compute(ep)
    out = ep.derived / "metrics"; out.mkdir(exist_ok=True)
    json.dump(session, open(out / "session.json", "w"), indent=1); extra.to_parquet(out / "records_extra.parquet", index=False)
    h = session["handovers"]; cs = session["coordination_score"]["score"]
    ep.set_status("metrics", "done", f"{h['count']} handovers ({h['source']}), {h['rate_per_min']:.1f}/min; coordination {cs if cs is None else round(cs)}; idle " +
                  ", ".join(f"{p} {v['idle_fraction']:.0%}" for p, v in session["hand_speed"].items()))
