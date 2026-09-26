"""Metrics stage: synthetic hand-speed signals with known synchrony / latency / idle, contact-event pairing, the
coordination score, and (when the CoMind clip is present) the real episode through the FastAPI TestClient."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.playground import metrics as M
from duet.playground.episode import Episode, Stream

ROOT = Path(__file__).resolve().parents[1]
COMIND = ROOT / "data/playground/episodes/comind_43276420_clip"
FPS = 10.0


def _bump(n, at, width=6, amp=1.0):
    x = np.zeros(n); i = np.arange(n); return x + amp * np.exp(-0.5 * ((i - at) / width) ** 2)


def test_xcorr_recovers_known_lag_and_is_bounded():
    n = 200; a = _bump(n, 100) + 0.01 * np.sin(np.arange(n)); b = np.roll(a, 4)  # b lags a by 0.4 s at 10 fps
    sync, lag = M.xcorr_sync(a, b, FPS)
    assert sync == pytest.approx(1.0, abs=1e-6) and lag == pytest.approx(0.4, abs=1e-9)
    s2, l2 = M.xcorr_sync(a, np.roll(a, -7), FPS)  # b leads a by 0.7 s
    assert s2 == pytest.approx(1.0, abs=1e-6) and l2 == pytest.approx(-0.7, abs=1e-9)
    assert -1 <= M.xcorr_sync(a, np.random.default_rng(0).normal(size=n), FPS)[0] <= 1
    assert np.isnan(M.xcorr_sync(np.zeros(n), a, FPS)[0])


def test_onset_and_receiver_latency_on_synthetic_handover():
    n = 300; rng = np.random.default_rng(1)
    giver = 0.02 + 0.005 * rng.normal(size=n); recv = 0.02 + 0.005 * rng.normal(size=n)
    giver[150:170] = 0.5; recv[156:176] = 0.5  # receiver starts 6 frames = 0.6 s after the giver
    t0 = 1.0; ev = {"t0": t0 + 15.0, "t1": t0 + 18.0, "giver": "a", "receiver": "b", "source": "test", "giver_basis": "given"}
    sig = {"a": {"speed": giver, "visible": np.ones(n, bool), "unit": "m/s"}, "b": {"speed": recv, "visible": np.ones(n, bool), "unit": "m/s"}}
    h = M.handover_metrics(ev, sig, None, FPS, t0, ["a", "b"])
    assert h["giver_reach_onset_s"] == pytest.approx(0.0, abs=0.11)  # onset at frame 150 = t0 + 15 s
    assert h["receiver_response_s"] == pytest.approx(0.6, abs=1e-9)
    assert h["synchrony"] > 0.8 and h["synchrony_lag_s"] == pytest.approx(0.6, abs=1e-9)
    # unknown giver/receiver: the earlier onset is called the giver
    h2 = M.handover_metrics({**ev, "giver": None, "receiver": None, "giver_basis": "onset_order"}, sig, None, FPS, t0, ["b", "a"])
    assert (h2["giver"], h2["receiver"], h2["giver_basis"]) == ("a", "b", "onset_order") and h2["receiver_response_s"] >= 0


def test_giver_hold_counts_still_frames_between_peak_and_transfer():
    n = 200; giver = np.full(n, 0.02); giver[100:105] = 0.6; giver[105:125] = 0.03  # peak then 2 s still, transfer at the giver speed minimum after the peak
    dist = np.full(n, np.nan); dist[100:140] = 1.0; dist[124] = 0.1  # metric transfer at frame 124
    sig = {"a": {"speed": giver, "visible": np.ones(n, bool), "unit": "m/s"}, "b": {"speed": np.full(n, 0.02), "visible": np.ones(n, bool), "unit": "m/s"}}
    h = M.handover_metrics({"t0": 10.0, "t1": 14.0, "giver": "a", "receiver": "b", "source": "world3d", "giver_basis": "given"}, sig, dist, FPS, 0.0, ["a", "b"])
    assert h["transfer_s"] == pytest.approx(2.4) and h["min_wrist_dist_m"] == pytest.approx(0.1)
    assert h["giver_hold_s"] == pytest.approx(2.0, abs=0.11)


def test_idle_mask_requires_two_second_runs_and_ignores_missing_hands():
    sp = np.full(100, 0.5); sp[10:40] = 0.01; sp[50:60] = 0.01; sp[70:95] = np.nan
    m = M.idle_mask(sp, 0.06, FPS)
    assert m[10:40].all() and not m[50:60].any() and not m[70:95].any() and m.mean() == pytest.approx(0.3)


def test_contact_pairing_filters_chatter_and_matches_one_to_one():
    ev = [{"t": 1.0, "person": "a", "hand": "R", "object": "bowl", "type": "grasp"}, {"t": 3.0, "person": "a", "hand": "R", "object": "bowl", "type": "release"},
          {"t": 3.4, "person": "b", "hand": "L", "object": "bowl", "type": "grasp"}, {"t": 6.0, "person": "b", "hand": "L", "object": "bowl", "type": "release"},
          # chatter: 0.2 s touches on the pan by both people should not become handovers
          {"t": 10.0, "person": "a", "hand": "L", "object": "pan", "type": "grasp"}, {"t": 10.2, "person": "a", "hand": "L", "object": "pan", "type": "release"},
          {"t": 10.3, "person": "b", "hand": "L", "object": "pan", "type": "grasp"}, {"t": 10.5, "person": "b", "hand": "L", "object": "pan", "type": "release"},
          # same-person re-grasp is not a handover
          {"t": 20.0, "person": "a", "hand": "R", "object": "knife", "type": "grasp"}, {"t": 21.0, "person": "a", "hand": "R", "object": "knife", "type": "release"},
          {"t": 21.2, "person": "a", "hand": "L", "object": "knife", "type": "grasp"}]
    h = M.handovers_from_contact(ev)
    assert len(h) == 1 and h[0]["giver"] == "a" and h[0]["receiver"] == "b" and h[0]["object"] == "bowl" and h[0]["t0"] == pytest.approx(2.7) and h[0]["t1"] == pytest.approx(3.7)


def test_distance_windows_and_coordination_score():
    d = np.full(120, 1.0); d[30:36] = 0.1; d[37:40] = 0.1; d[80:81] = 0.1  # one 0.9 s window (merged), one too short
    h = M.handovers_from_distance(d, FPS, 5.0, 0.25, "world3d")
    assert len(h) == 1 and h[0]["t0"] == pytest.approx(5.0 + 2.0) and h[0]["t1"] == pytest.approx(5.0 + 5.0)
    cs = M.coordination_score(0.5, [0.4, -0.2, 0.6], [0.3, 0.6], {"a": 0.1, "b": 0.3})
    assert cs["components"] == {"C_sync": 0.5, "C_resp": 0.8, "C_hold": 0.85, "C_engage": 0.8} and cs["score"] == pytest.approx(73.8, abs=0.05) and not cs["partial"]
    assert M.coordination_score(None, [], [], {"a": 0.0})["score"] == 100.0 and M.coordination_score(None, [], [], {})["score"] is None


def test_metrics_stage_on_synthetic_episode(tmp_path):
    n = 100; root = tmp_path / "eps"; d = root / "syn"; (d / "streams").mkdir(parents=True); (d / "derived" / "hands").mkdir(parents=True)
    ep = Episode(name="syn", root=str(d), streams=[Stream("a", "ego", "streams/a.mp4", person="alice"), Stream("b", "ego", "streams/b.mp4", person="bob")], reference="a", common_start_s=2.0, common_end_s=2.0 + n / FPS)
    ep.save(); rng = np.random.default_rng(0)
    for s, shift in (("a", 0), ("b", 5)):
        lm2d = np.full((n, 2, 21, 2), np.nan, np.float32); x = np.cumsum(rng.normal(0, 1.0, n)); x[40 + shift: 60 + shift] += np.linspace(0, 200, 20)
        lm2d[:, 0, :, 0] = x[:, None] + 300; lm2d[:, 0, :, 1] = 300; lm2d[:10] = np.nan  # left hand only, missing for the first second
        np.savez_compressed(d / "derived/hands" / f"{s}.npz", lm2d=lm2d, lm3d=np.full((n, 2, 21, 3), np.nan, np.float32), score=np.ones((n, 2), np.float32))
    M.metrics(ep)
    s = json.load(open(d / "derived/metrics/session.json")); x = pd.read_parquet(d / "derived/metrics/records_extra.parquet")
    assert ep.status["metrics"]["state"] == "done" and s["persons"] == ["alice", "bob"] and s["n_frames"] == n
    assert s["hand_speed"]["alice"]["hands_visible_fraction"] == pytest.approx(0.9) and s["hand_speed"]["alice"]["unit"] == "fw/s"
    assert s["handovers"]["source"] == "none" and s["handovers"]["count"] == 0 and s["coordination_score"]["partial"]
    assert s["synchrony"]["session_lag_s"] == pytest.approx(0.5) and s["synchrony"]["session"] > 0.5
    assert list(x.columns) == ["hand_speed_alice", "idle_alice", "hand_visible_alice", "hand_speed_bob", "idle_bob", "hand_visible_bob"] and len(x) == n
    assert len(s["per_minute"]) == 1 and s["per_minute"][0]["speech_fraction"] is None


@pytest.mark.skipif(not (COMIND / "derived/hands/leader.npz").exists(), reason="CoMind playground episode not present")
def test_metrics_api_on_comind_clip():
    from fastapi.testclient import TestClient
    from duet.playground.server import create_app
    app = create_app(token=None); LOCAL = {"base_url": "http://127.0.0.1:8765", "client": ("127.0.0.1", 50000)}
    ep = Episode.load(COMIND); M.metrics(ep)
    r = TestClient(app, **LOCAL).get("/api/episode/comind_43276420_clip/metrics"); assert r.status_code == 200; s = r.json()
    assert s["available"] and s["persons"] == ["leader", "helper"] and 40 < s["duration_s"] < 50 and s["n_frames"] == 466
    for p in ("leader", "helper"):
        assert 0.5 < s["hand_speed"][p]["hands_visible_fraction"] <= 1.0 and 0 <= s["hand_speed"][p]["idle_fraction"] <= 1
    assert s["handovers"]["count"] == len(s["handovers"]["events"]) and s["handovers"]["rate_per_min"] == pytest.approx(s["handovers"]["count"] / s["duration_s"] * 60, abs=0.01)
    for e in s["handovers"]["events"]:
        assert e["giver"] in s["persons"] and e["receiver"] in s["persons"] and e["giver"] != e["receiver"]
        if e["synchrony"] is not None:
            assert -1 <= e["synchrony"] <= 1
    assert s["coordination_score"]["score"] is None or 0 <= s["coordination_score"]["score"] <= 100
    assert TestClient(app, **LOCAL).get("/api/episode/does_not_exist/metrics").status_code == 404
