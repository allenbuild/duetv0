"""gaze_proxy stage on synthetic episodes: the 3D (headpose) path and the 2D (image) fallback."""
import json

import numpy as np
import pandas as pd
import pytest

from duet.playground import gaze_proxy as GP
from duet.playground.episode import Episode, Stream


def _episode(tmp_path, n, streams):
    ep = Episode(name="t", root=str(tmp_path), streams=streams, reference=streams[0].name, proc_fps=10.0, proc_size=640)
    for s in streams:
        d = ep.derived / "frames" / s.name; d.mkdir(parents=True)
        for k in range(n):
            (d / f"{k + 1:06d}.jpg").touch()
    ep.save()
    return ep


def _T(cam_centre, x_cam, y_cam, z_cam):
    T = np.eye(4); T[:3, :3] = np.stack([x_cam, y_cam, z_cam], 1); T[:3, 3] = cam_centre; return T


def test_headpose_path(tmp_path):
    n = 4
    ep = _episode(tmp_path, n, [Stream("A", "ego", "streams/A.mp4", person="alice", width=640, height=640), Stream("B", "ego", "streams/B.mp4", person="bob", width=640, height=640)])
    # A at the origin looking down world +X; B two metres away looking down world +Y (perpendicular)
    TA = _T([0, 0, 0], [0, -1, 0], [0, 0, -1], [1, 0, 0]); TB = _T([2, 0, 0], [1, 0, 0], [0, 0, -1], [0, 1, 0])
    hp = ep.derived / "headpose"; hp.mkdir()
    np.savez(hp / "A.npz", T_world_cam=np.repeat(TA[None], n, 0).astype(np.float32), backend="test")
    TBs = np.repeat(TB[None], n, 0).astype(np.float32); TBs[3] = np.nan  # last frame: B's pose unknown
    np.savez(hp / "B.npz", T_world_cam=TBs, backend="test")
    w3 = ep.derived / "world3d"; w3.mkdir()
    cup = np.full((n, 3), np.nan, np.float32); cup[:2] = [2.0, 0.1, 0.0]  # 2.9 deg off A's axis, straight ahead of B (B looks +Y from (2,0,0))
    np.savez(w3 / "world3d.npz", bodies=np.full((n, 2, 17, 3), np.nan, np.float32), head_A=TA[None, :3, 3].repeat(n, 0), head_B=TB[None, :3, 3].repeat(n, 0), object_cup=cup)
    (ep.dir / "rig.json").write_text(json.dumps({"object_tags": {"cup": 10}}))
    GP.gaze_proxy(ep)
    assert ep.status["gaze_proxy"]["state"] == "done"
    a = np.load(ep.derived / "gaze_proxy" / "A.npz"); b = np.load(ep.derived / "gaze_proxy" / "B.npz")
    assert np.allclose(a["angle_to_partner_deg"], 0.0, atol=1e-4) and a["looking_at_partner"].all()
    assert np.allclose(b["angle_to_partner_deg"][:3], 90.0, atol=1e-4) and not b["looking_at_partner"].any()
    assert (a["method"] == "headpose").all() and (b["method"][:3] == "headpose").all() and b["method"][3] == "none"
    assert not a["mutual_gaze"].any()
    assert a["angle_to_nearest_object_deg"][0] == pytest.approx(np.degrees(np.arctan(0.05)), abs=1e-3)
    assert a["looking_at_object"][0] == "cup" and a["nearest_object"][1] == "cup" and a["looking_at_object"][2] == ""
    assert b["looking_at_object"][0] == "cup" and b["angle_to_nearest_object_deg"][0] == pytest.approx(0.0, abs=1e-4) and b["nearest_object"][3] == ""
    df = pd.read_parquet(ep.derived / "gaze_proxy" / "records_extra.parquet")
    assert len(df) == n and {"A_angle_to_partner_deg", "B_looking_at_partner", "A_looking_at_object", "mutual_gaze"} <= set(df.columns)


def test_image_path_without_headpose(tmp_path):
    n = 3
    ep = _episode(tmp_path, n, [Stream("A", "ego", "streams/A.mp4", person="alice", width=1408, height=1408), Stream("B", "ego", "streams/B.mp4", person="bob", width=1408, height=1408)])
    kp = np.full((n, 4, 17, 3), np.nan, np.float32)
    kp[0, 1, :3] = [[320, 320, 0.9], [318, 316, 0.9], [322, 316, 0.9]]  # partner head at the image centre (slot 1)
    kp[0, 0, 9] = [100, 500, 0.9]                                      # wearer's own wrist in slot 0, no head
    kp[1, 2, :3] = [[320 + 224.066, 320, 0.8], [320 + 224.066, 320, 0.2], [320 + 224.066, 320, 0.1]]  # only the nose counts (conf >= 0.3): 45 deg
    b2 = ep.derived / "body2d"; b2.mkdir()
    np.savez(b2 / "A.npz", kpts=kp, boxes=np.full((n, 4, 5), np.nan, np.float32), img_w=640, img_h=640)
    np.savez(b2 / "B.npz", kpts=np.full((n, 4, 17, 3), np.nan, np.float32), boxes=np.full((n, 4, 5), np.nan, np.float32), img_w=640, img_h=640)
    boxes = np.full((n, 12, 5), np.nan, np.float32); names = np.full((n, 12), "", dtype="<U24")
    boxes[0, 0] = [300, 330, 340, 350, 0.7]; names[0, 0] = "bowl"      # centre (320, 340): 5.1 deg
    boxes[0, 1] = [500, 500, 600, 600, 0.9]; names[0, 1] = "pan"       # farther off-axis
    boxes[2, 0] = [0, 0, 100, 100, 0.5]; names[2, 0] = "knife"        # 55 deg-ish: nearest but not looked at
    ob = ep.derived / "objects"; ob.mkdir(); np.savez(ob / "A.npz", boxes=boxes, names=names)
    GP.gaze_proxy(ep)
    assert ep.status["gaze_proxy"]["state"] == "done"
    a = np.load(ep.derived / "gaze_proxy" / "A.npz")
    assert a["method"].tolist() == ["image", "image", "none"]
    assert a["angle_to_partner_deg"][0] == pytest.approx(0.0, abs=1.0) and a["looking_at_partner"][0]  # nose/eye mean is 2.7 px off centre
    assert a["angle_to_partner_deg"][1] == pytest.approx(45.0, abs=0.1) and not a["looking_at_partner"][1]
    assert np.isnan(a["angle_to_partner_deg"][2])
    assert a["intrinsics_fx_px"] == pytest.approx(224.07, abs=0.1)  # nominal 110 deg pinhole on the 640 px frame
    assert a["looking_at_object"][0] == "bowl" and a["nearest_object"][0] == "bowl"
    assert a["nearest_object"][2] == "knife" and a["looking_at_object"][2] == ""
    assert a["nearest_object"][1] == "" and np.isnan(a["angle_to_nearest_object_deg"][1])
    b = np.load(ep.derived / "gaze_proxy" / "B.npz")
    assert (b["method"] == "none").all() and not b["looking_at_partner"].any()
    assert not a["mutual_gaze"].any()
    assert len(pd.read_parquet(ep.derived / "gaze_proxy" / "records_extra.parquet")) == n


def test_skips_cleanly_without_evidence(tmp_path):
    ep = _episode(tmp_path, 2, [Stream("A", "ego", "streams/A.mp4", width=640, height=640)])
    GP.gaze_proxy(ep)
    assert ep.status["gaze_proxy"]["state"] == "skipped"
    assert len(pd.read_parquet(ep.derived / "gaze_proxy" / "records_extra.parquet")) == 2


def test_helpers():
    intr = GP.G.Intrinsics.nominal(640, 640, 110.0)
    cx, cy = intr.K[0, 2], intr.K[1, 2]
    assert GP.pixel_angle_deg(intr, np.array([[cx, cy], [cx + intr.K[0, 0], cy]])).tolist() == pytest.approx([0.0, 45.0], abs=1e-4)
    assert GP.angle_between_deg(np.array([[1.0, 0, 0]]), np.array([[0, 1.0, 0]]))[0] == pytest.approx(90.0)
    kp = np.full((1, 4, 17, 3), np.nan, np.float32); kp[0, 3, 0] = [10, 20, 0.9]; kp[0, 3, 1] = [12, 18, 0.2]
    assert GP.partner_head_px(kp)[0].tolist() == [10, 20]
