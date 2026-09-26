"""Contact stage: per-frame candidate rules, majority vote + hysteresis, ZED depth agreement, and the stage end to end."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.playground import contact as C  # noqa: E402
from duet.playground.episode import Episode, Stream  # noqa: E402

FPS = 10.0


def _hand(cx, cy, spread=20.0):
    """21 landmarks around (cx, cy): wrist below, fingertips above."""
    lm = np.zeros((21, 2), np.float32); lm[:, 0] = cx; lm[:, 1] = cy
    lm[0] = (cx, cy + spread)
    for i, t in enumerate(C.TIPS):
        lm[t] = (cx - spread + i * spread / 2, cy - spread)
    for i, p in enumerate(C.PALM[1:]):
        lm[p] = (cx - spread / 2 + i * spread / 3, cy)
    return lm


def _objects(*items):
    boxes = np.full((12, 5), np.nan, np.float32); names = np.full(12, "", dtype="<U24")
    for j, (name, box, conf) in enumerate(items):
        boxes[j, :4] = box; boxes[j, 4] = conf; names[j] = name
    return boxes, names


def test_frame_contact_rules():
    b, nm = _objects(("bowl", (80, 80, 160, 160), 0.8), ("plate", (0, 0, 300, 300), 0.9))
    name, score, j = C.frame_contact(_hand(120, 120), b, nm)
    assert name == "bowl" and j == 0 and abs(score - 1.0 * 0.8) < 1e-6      # all 5 tips inside, smallest box wins the tie
    name, score, j = C.frame_contact(_hand(400, 400), b, nm)
    assert name == "" and j == -1 and score == 0.0                          # nothing near
    # palm centre inside the (10 % enlarged) box but only 2 fingertips: still a candidate, palm-only score
    lm = _hand(120, 120); lm[[4, 8, 12], 1] = 10.0  # move three tips far above the bowl
    name, score, j = C.frame_contact(lm, b, nm)
    assert name == "bowl" and abs(score - max(2 / 5, C.PALM_ONLY_FRAC) * 0.8) < 1e-6
    # missing hand
    assert C.frame_contact(np.full((21, 2), np.nan, np.float32), b, nm) == ("", 0.0, -1)
    # the enlargement: tips 3 px outside a 100 px box are inside after growing it by 10 %
    b2, nm2 = _objects(("cup", (100, 100, 200, 200), 0.5))
    lm = _hand(150, 150); lm[list(C.TIPS), 0] = 203.0; lm[list(C.TIPS), 1] = 150.0
    assert C.frame_contact(lm, b2, nm2)[0] == "cup"
    lm[list(C.TIPS), 0] = 212.0; lm[list(C.PALM), 0] = 212.0
    assert C.frame_contact(lm, b2, nm2)[0] == ""


def test_majority_vote_and_hysteresis():
    assert C.majority_vote(["a", "a", "b", "a", "a"], 5) == ["a"] * 5
    assert C.majority_vote(["", "", "a", "", ""], 5) == [""] * 5
    # 0.3 s to start (3 frames), 0.5 s to end (5 frames) at 10 fps
    v = [""] * 3 + ["bowl"] * 2 + [""] * 10                      # 2 frames only: never held
    held, ev = C.hysteresis(v, 3, 5); assert not any(held) and ev == []
    v = [""] * 3 + ["bowl"] * 3 + [""] * 3 + ["bowl"] * 2 + [""] * 10   # 3-frame run starts; a 3-frame gap does not end it
    held, ev = C.hysteresis(v, 3, 5)
    assert ev == [(3, "grasp", "bowl"), (11, "release", "bowl")]
    assert held[3:11] == ["bowl"] * 8 and held[:3] == [""] * 3 and held[11:] == [""] * 10
    # switching object: bowl -> cup after 5 frames of cup; release and grasp both at the start of the cup run
    v = ["bowl"] * 6 + ["cup"] * 6
    held, ev = C.hysteresis(v, 3, 5)
    assert ev == [(0, "grasp", "bowl"), (6, "release", "bowl"), (6, "grasp", "cup")] and held[6:] == ["cup"] * 6


def test_contact_stream_synthetic():
    n = 40; lm2d = np.full((n, 2, 21, 2), np.nan, np.float32); boxes = np.full((n, 12, 5), np.nan, np.float32); names = np.full((n, 12), "", dtype="<U24")
    for k in range(n):
        boxes[k], names[k] = _objects(("bowl", (80, 80, 160, 160), 0.8), ("knife", (300, 300, 340, 340), 0.6))
        if 10 <= k < 30:
            lm2d[k, 1] = _hand(120, 120)          # right hand in the bowl for 2 s
        if 20 <= k < 24:
            lm2d[k, 0] = _hand(320, 320)          # left hand on the knife for 0.4 s: too short to be a 0.3 s+0.5 s contact? no: 4 frames >= 3 starts it
        if k in (15, 16):
            lm2d[k, 1] = _hand(500, 500)          # a 2-frame glitch away from the bowl: absorbed by the vote / hysteresis
    r = C.contact_stream(lm2d, boxes, names, FPS)
    assert (r["held"][10:30, 1] == "bowl").all() and (r["held"][:10, 1] == "").all() and (r["held"][30:, 1] == "").all()
    assert r["events"] == [(10, "grasp", "bowl", "R"), (20, "grasp", "knife", "L"), (24, "release", "knife", "L"), (30, "release", "bowl", "R")]
    assert r["score"][12, 1] > 0.5 and r["score"][5, 1] == 0.0


def _episode(tmp_path, n=20, with_depth=False):
    ep = Episode(name="t", root=str(tmp_path), streams=[Stream("ego", "ego", "streams/ego.mp4", person="alice"), Stream("cam", "exo", "streams/cam.mp4")],
                 reference="ego", proc_fps=FPS, common_start_s=1.0, common_end_s=1.0 + n / FPS)
    d = tmp_path / "derived"; (d / "hands").mkdir(parents=True); (d / "objects").mkdir(); fd = d / "frames" / "ego"; fd.mkdir(parents=True)
    for k in range(n):
        cv2.imwrite(str(fd / f"{k+1:06d}.jpg"), np.zeros((360, 640, 3), np.uint8))
    lm2d = np.full((n, 2, 21, 2), np.nan, np.float32); lm3d = np.full((n, 2, 21, 3), np.nan, np.float32); score = np.zeros((n, 2), np.float32)
    boxes = np.full((n, 12, 5), np.nan, np.float32); names = np.full((n, 12), "", dtype="<U24")
    for k in range(n):
        boxes[k], names[k] = _objects(("bowl", (80, 80, 160, 160), 0.8))
        if 5 <= k < 15:
            lm2d[k, 0] = _hand(120, 120); score[k, 0] = 0.9
    np.savez_compressed(d / "hands" / "ego.npz", lm2d=lm2d, lm3d=lm3d, score=score)
    np.savez_compressed(d / "objects" / "ego.npz", boxes=boxes, names=names)
    if with_depth:
        zd = tmp_path / "zed" / "ego"; (zd / "depth").mkdir(parents=True)
        json.dump({"fps": 30}, open(zd / "export.json", "w"))
        for k in range(n):
            kz = int(round((ep.common_start_s + k / ep.proc_fps) * 30))
            depth = np.full((720, 1280), 1000, np.uint16)                  # scene at 1.0 m (mm)
            depth[140:340, 140:340] = 1000 if k < 10 else 1400              # object box region: 1.0 m, then 1.4 m (hand stays at 1.0 m elsewhere)
            depth[220:300, 200:280] = 1000                                  # hand landmarks region (x2 scale of 100-140 px) stays at 1.0 m
            cv2.imwrite(str(zd / "depth" / f"{kz:06d}.png"), depth)
    ep.save(); return ep


def test_stage_end_to_end(tmp_path):
    ep = _episode(tmp_path); C.contact(ep)
    assert ep.status["contact"]["state"] == "done", ep.status["contact"]
    z = np.load(tmp_path / "derived" / "contact" / "ego.npz")
    assert z["held"].shape == (20, 2) and (z["held"][5:15, 0] == "bowl").all() and not bool(z["depth_used"])
    ev = json.load(open(tmp_path / "derived" / "contact" / "events.json"))
    assert [(e["type"], e["object"], e["hand"], e["person"], e["stream"]) for e in ev] == [("grasp", "bowl", "L", "alice", "ego"), ("release", "bowl", "L", "alice", "ego")]
    assert abs(ev[0]["t"] - (1.0 + 5 / FPS)) < 1e-6 and abs(ev[1]["t"] - (1.0 + 15 / FPS)) < 1e-6
    import pandas as pd
    df = pd.read_parquet(tmp_path / "derived" / "contact" / "records_extra.parquet")
    assert list(df.columns) == ["ego_held_L", "ego_contact_score_L", "ego_held_R", "ego_contact_score_R"] and len(df) == 20
    assert df["ego_held_L"].iloc[7] == "bowl" and df["ego_contact_score_L"].iloc[7] > 0 and df["ego_held_R"].iloc[7] == ""


def test_depth_agreement_rejects_far_object(tmp_path):
    ep = _episode(tmp_path, with_depth=True); C.contact(ep)
    z = np.load(tmp_path / "derived" / "contact" / "ego.npz"); rep = json.load(open(tmp_path / "derived" / "contact" / "report.json"))
    assert bool(z["depth_used"]) and rep["streams"]["ego"]["depth_check"].startswith("zed depth used")
    # frames 5..9: hand and bowl both at 1.0 m -> contact; frames 10..14: bowl region at 1.4 m, hand at 1.0 m -> rejected
    assert (z["raw_held"][5:10, 0] == "bowl").all() and (z["raw_held"][10:15, 0] == "").all()


def test_stage_skips_without_inputs(tmp_path):
    ep = Episode(name="t", root=str(tmp_path), streams=[Stream("ego", "ego", "streams/ego.mp4", person="a")], reference="ego"); ep.save(); C.contact(ep)
    assert ep.status["contact"]["state"] == "skipped"
