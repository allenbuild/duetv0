"""autolabel: shared featurisation, hysteresis, and the stage on a synthetic episode with a small stand-in model."""
import json
import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from duet.playground import autolabel as AL
from duet.playground.episode import Episode, Stream


def test_feature_layout_and_units():
    n, fps = 12, 10.0
    uv = np.full((n, 2, 2), np.nan, np.float32)
    uv[:, 0, 0] = np.arange(n) * 0.02; uv[:, 0, 1] = 0.5  # left wrist moves 0.02 widths/frame -> 0.2 widths/s
    uv[:5, 1] = [0.3, 0.5]                                # right wrist present for 5 frames, then lost
    base = AL.person_base(uv, np.ones((n, 2)), np.r_[np.full(6, 10.0), np.full(6, np.nan)], fps)
    assert base.shape == (n, len(AL.PERSON_FEATS))
    F = {k: i for i, k in enumerate(AL.PERSON_FEATS)}
    assert base[1:, F["wrist_speed_L"]] == pytest.approx(0.2, abs=1e-5) and base[0, F["wrist_speed_L"]] == 0.0
    assert base[:, F["hand_present_L"]].all() and base[:, F["hand_present_R"]].tolist() == [1] * 5 + [0] * 7
    assert base[5:, F["wrist_speed_R"]].tolist() == [0.0] * 7 and base[6, F["own_hand_dist"]] == 0.0
    assert base[:, F["angle_valid"]].tolist() == [1] * 6 + [0] * 6 and base[6, F["angle_to_partner_deg"]] == 0.0 and base[0, F["angle_to_partner_deg"]] == 10.0
    inter = AL.inter_base(np.full(n, 1.2), np.r_[np.full(6, 0.4), np.zeros(6)], n)
    assert inter[:, 2].tolist() == [1] * 6 + [0] * 6
    X = AL.assemble(base, base, inter, fps)
    names = AL.feature_names()
    assert X.shape == (n, len(names)) == (n, (2 * len(AL.PERSON_FEATS) + len(AL.INTER_FEATS)) * (1 + 3 * len(AL.WINDOWS_S)))
    j = names.index("A_wrist_speed_L_max1s"); assert X[-1, j] == pytest.approx(0.2, abs=1e-5)
    j = names.index("A_hand_present_R_mean3s"); assert X[-1, j] == pytest.approx(23 / 30)  # 30-frame window: 18 padded copies of row 0 (present) + 5 present + 7 absent
    assert np.isfinite(X).all()


def test_project_equidistant_is_roll_invariant():
    p = np.array([[0.1, -0.2, 0.5], [0.12, -0.2, 0.5], [0.0, 0.0, -1.0]])
    uv = AL.project_equidistant(p, [0.0, 0.0, 1.0])
    assert np.isnan(uv[2]).all() and np.isfinite(uv[:2]).all()
    d = np.linalg.norm(uv[1] - uv[0])
    uv2 = AL.project_equidistant(p, [0.0, 1e-9, 1.0])  # nearly the same axis, different in-plane basis choice would not change distances
    assert np.linalg.norm(uv2[1] - uv2[0]) == pytest.approx(d, rel=1e-4)
    theta = np.arctan2(np.hypot(0.1, -0.2), 0.5)
    assert np.linalg.norm(uv[0]) == pytest.approx(AL.ARIA_FISHEYE_F_NORM * theta, rel=1e-6)


def test_hysteresis_intervals():
    fps = 10.0
    p = np.zeros(100)
    p[10:13] = 0.9          # 0.3 s above hi: too short to start
    p[20:40] = 0.9          # starts at 20
    p[40:45] = 0.5          # between lo and hi: stays on
    p[45:52] = 0.1          # 0.7 s below lo: not enough to end
    p[52:60] = 0.9
    p[60:] = 0.1            # ends at 60 (1 s below lo)
    assert AL.hysteresis_intervals(p, fps, hi=0.7, lo=0.3) == [(20, 60)]
    q = np.zeros(30); q[20:] = 0.95
    assert AL.hysteresis_intervals(q, fps, hi=0.7, lo=0.3) == [(20, 30)]  # still on at the end
    assert AL.hysteresis_intervals(np.zeros(30), fps, 0.7, 0.3) == []


def _episode(tmp_path, n):
    streams = [Stream("A", "ego", "streams/A.mp4", person="alice", width=640, height=640), Stream("B", "ego", "streams/B.mp4", person="bob", width=640, height=640)]
    ep = Episode(name="t", root=str(tmp_path), streams=streams, reference="A", proc_fps=10.0, proc_size=640, common_start_s=2.0)
    for s in streams:
        d = ep.derived / "frames" / s.name; d.mkdir(parents=True)
        for k in range(n):
            (d / f"{k + 1:06d}.jpg").touch()
    ep.save(); return ep


def test_stage_on_synthetic_episode(tmp_path, monkeypatch):
    n = 80; ep = _episode(tmp_path, n)
    rng = np.random.default_rng(0)
    hands = ep.derived / "hands"; hands.mkdir(); b2 = ep.derived / "body2d"; b2.mkdir()
    for s in ep.streams:
        lm = np.full((n, 2, 21, 2), np.nan, np.float32); lm[:, :, 0, :] = 320 + rng.normal(0, 5, (n, 2, 2)); lm[:, 1, 0, 0] += 100
        np.savez(hands / f"{s.name}.npz", lm2d=lm, lm3d=np.zeros((n, 2, 21, 3), np.float32), score=np.ones((n, 2), np.float32))
        np.savez(b2 / f"{s.name}.npz", kpts=np.full((n, 4, 17, 3), np.nan, np.float32), boxes=np.full((n, 4, 5), np.nan, np.float32), img_w=640, img_h=640)
    gp = ep.derived / "gaze_proxy"; gp.mkdir()
    np.savez(gp / "A.npz", angle_to_partner_deg=np.full(n, 40.0, np.float32)); np.savez(gp / "B.npz", angle_to_partner_deg=np.full(n, np.nan, np.float32))
    ct = ep.derived / "contact"; ct.mkdir()  # contact stage layout: object name per hand, "" = nothing held
    held = np.full((n, 2), "", dtype="<U24"); held[10:20, 1] = "bowl"
    np.savez(ct / "A.npz", held=held, score=np.zeros((n, 2), np.float32))
    X, names = AL.playground_features(ep, n)
    assert X.shape == (n, len(AL.feature_names())) and names == ["alice", "bob"]
    F0 = AL.feature_names(); assert X[:, F0.index("A_held_R")].tolist() == [0] * 10 + [1] * 10 + [0] * 60 and X[:, F0.index("A_held_L")].sum() == 0
    assert X[:, F0.index("A_angle_valid")].all() and not X[:, F0.index("B_angle_valid")].any()
    # stand-in model: fires on high left-wrist speed of person A, which we plant in frames 30..60
    F = AL.feature_names(); j = F.index("A_wrist_speed_L")
    Xr = rng.normal(0, 1, (4000, len(F))).astype(np.float32); Xr[:, j] = rng.uniform(0, 5, 4000); yr = (Xr[:, j] > 2.5).astype(int)
    clf = HistGradientBoostingClassifier(max_iter=30).fit(Xr, yr)
    model = {"models": {"ja_active": clf, "handover_active": clf, "handover_direction": HistGradientBoostingClassifier(max_iter=5).fit(Xr, yr)},
             "feature_names": F, "thresholds": {t: {"hi": 0.6, "lo": 0.3} for t in ("ja_active", "handover_active")}, "meta": {"trained": "test"}}
    p = tmp_path / "model.pkl"; pickle.dump(model, open(p, "wb")); monkeypatch.setattr(AL, "MODEL_PATH", p)
    lm = np.load(hands / "A.npz")["lm2d"]; lm[30:60, 0, 0, 0] += np.cumsum(np.full(30, 250.0))  # 0.39 widths per frame -> 3.9 widths/s
    np.savez(hands / "A.npz", lm2d=lm, lm3d=np.zeros((n, 2, 21, 3), np.float32), score=np.ones((n, 2), np.float32))
    AL.autolabel(ep)
    assert ep.status["autolabel"]["state"] == "done", ep.status["autolabel"]
    props = json.load(open(ep.derived / "autolabel" / "proposals.json"))
    assert props and {p["event"] for p in props} == {"joint_attention", "handover"}
    for pr in props:
        assert pr["status"] == "pending" and pr["source"] == "autolabel_gbdt" and 0 <= pr["confidence"] <= 1
        assert pr["t0"] == pytest.approx(2.0 + 30 / 10.0, abs=0.35) and pr["t1"] > pr["t0"]
        if pr["event"] == "handover":
            assert {pr["giver"], pr["receiver"]} == {"alice", "bob"}
        else:
            assert pr["giver"] is None and pr["receiver"] is None
    df = pd.read_parquet(ep.derived / "autolabel" / "records_extra.parquet")
    assert len(df) == n and {"p_joint_attention", "p_handover"} <= set(df.columns)
    assert df["p_handover"][35:55].mean() > 0.8 and df["p_handover"][:25].mean() < 0.2


def test_stage_skips_without_model(tmp_path, monkeypatch):
    ep = _episode(tmp_path, 5); monkeypatch.setattr(AL, "MODEL_PATH", tmp_path / "missing.pkl")
    AL.autolabel(ep); assert ep.status["autolabel"]["state"] == "skipped"
