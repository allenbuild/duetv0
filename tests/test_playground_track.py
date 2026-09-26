"""Track stage: synthetic crossing boxes with distinct torso colours must keep their ids; duplicates are suppressed;
the switch metric and the cross-view matchers behave on synthetic data; the stage runs end to end on a tmp episode."""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.playground import track as T  # noqa: E402
from duet.playground.episode import Episode, Stream  # noqa: E402

W, H, FPS = 640, 360, 10.0
RED, BLUE = (0, 0, 220), (220, 60, 0)  # BGR


def _person_box(cx, cy, w=80, h=200):
    return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2, 0.9], np.float32)


def _kpts_for(box):
    """COCO-17 keypoints with shoulders at 25 % and hips at 60 % of the box height, all confident."""
    x0, y0, x1, y1 = box[:4]; kp = np.full((17, 3), np.nan, np.float32)
    kp[:, 0] = (x0 + x1) / 2; kp[:, 1] = (y0 + y1) / 2; kp[:, 2] = 0.9
    for i, fx in ((5, 0.25), (6, 0.75)):
        kp[i, :2] = (x0 + fx * (x1 - x0), y0 + 0.25 * (y1 - y0))
    for i, fx in ((11, 0.3), (12, 0.7)):
        kp[i, :2] = (x0 + fx * (x1 - x0), y0 + 0.6 * (y1 - y0))
    return kp


def _crossing(n=40, swap_slots=True, occlude=()):
    """Two people crossing: A moves left->right, B right->left, drawn with red / blue torsos. Raw slots (like body2d) swap
    at the crossing when swap_slots. Returns boxes [n,4,5], kpts [n,4,17,3], images {k: BGR}, truth [n,4] (0 = A, 1 = B)."""
    boxes = np.full((n, 4, 5), np.nan, np.float32); kpts = np.full((n, 4, 17, 3), np.nan, np.float32); truth = np.full((n, 4), -1, int); imgs = {}
    for k in range(n):
        t = k / (n - 1); ax, bx = 100 + t * 440, 540 - t * 440
        a, b = _person_box(ax, 180), _person_box(bx, 180)
        img = np.full((H, W, 3), 90, np.uint8)
        for box, col in ((a, RED), (b, BLUE)):
            kp = _kpts_for(box); x0, y0, x1, y1 = T.torso_region(box, kp, W, H)
            cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (40, 40, 40), -1)
            cv2.rectangle(img, (x0, y0), (x1, y1), col, -1)
        imgs[k] = img
        sa, sb = (0, 1) if (not swap_slots or t < 0.5) else (1, 0)
        if k not in occlude:
            boxes[k, sa], kpts[k, sa], truth[k, sa] = a, _kpts_for(a), 0
        boxes[k, sb], kpts[k, sb], truth[k, sb] = b, _kpts_for(b), 1
    return boxes, kpts, imgs, truth


def _ids_consistent(ids, truth, boxes=None):
    """Every kept track id maps to exactly one true person over the whole sequence (so the mapping before the crossing
    is the mapping after it). Frames where the two true boxes overlap with IoU > 0.5 are skipped: there one torso is
    fully drawn over the other, so which box is "which person" is undecidable from boxes and colour."""
    m = {}
    for k in range(len(ids)):
        if boxes is not None:
            pres = [s for s in range(boxes.shape[1]) if np.isfinite(boxes[k, s, 4])]
            if len(pres) == 2 and T._iou(boxes[k, pres[0], :4], boxes[k, pres[1], :4]) > 0.5:
                continue
        for s in range(ids.shape[1]):
            if ids[k, s] >= 0 and truth[k, s] >= 0:
                m.setdefault(ids[k, s], set()).add(truth[k, s])
    return all(len(v) == 1 for v in m.values()) and len(m) == 2


def test_crossing_boxes_keep_identity():
    boxes, kpts, imgs, truth = _crossing()
    r = T.link_tracks(boxes, kpts, lambda k: imgs[k], W, H, FPS)
    assert r["n_tracks"] == 2
    assert _ids_consistent(r["ids"], truth, boxes)
    # the raw slots swap at the crossing; the tracker corrects that (slot remap flagged, no residual switch)
    assert r["slot_remap"].any() and not r["switch"].any()
    # the track-ordered boxes move smoothly: one track keeps going right, the other keeps going left (no jump back)
    cx = (r["boxes"][:, :, 0] + r["boxes"][:, :, 2]) / 2; d = np.diff(cx, axis=0); d = d[np.isfinite(d).all(1)]
    assert sorted([bool((d[:, 0] > 0).all()), bool((d[:, 1] > 0).all())]) == [False, True] and (np.abs(d) < 30).all()


def test_crossing_with_occlusion_and_memory():
    occl = set(range(18, 26))  # A hidden for 0.8 s right around the crossing
    boxes, kpts, imgs, truth = _crossing(occlude=occl)
    r = T.link_tracks(boxes, kpts, lambda k: imgs[k], W, H, FPS)
    assert r["n_tracks"] == 2 and _ids_consistent(r["ids"], truth, boxes)
    # after a gap longer than the memory a new id must be created
    long = set(range(10, 35))
    boxes, kpts, imgs, truth = _crossing(occlude=long)
    r = T.link_tracks(boxes, kpts, lambda k: imgs[k], W, H, FPS, memory_s=1.0, min_track_s=0.3)
    assert r["n_raw_tracks"] == 3


def test_iou_only_would_swap_but_appearance_does_not():
    """Same geometry, cost without appearance: at the crossing the two predicted boxes are nearly identical, so the
    assignment is a coin flip; with appearance the colours decide. We only assert the appearance path is consistent
    over several crossing speeds."""
    for n in (20, 30, 60):
        boxes, kpts, imgs, truth = _crossing(n=n)
        r = T.link_tracks(boxes, kpts, lambda k: imgs[k], W, H, FPS)
        assert _ids_consistent(r["ids"], truth, boxes), n


def test_no_images_still_tracks():
    boxes, kpts, _, truth = _crossing(swap_slots=False)
    r = T.link_tracks(boxes, None, None, W, H, FPS, dedup_overlap=1.01)  # no appearance, no keypoints, no duplicate suppression: pure IoU + velocity
    assert r["n_tracks"] == 2 and (r["ids"] >= 0).sum() == (truth >= 0).sum()
    r = T.link_tracks(boxes, kpts, None, W, H, FPS)  # without appearance the two fully overlapping frames are deduplicated (IoU and identical keypoints)
    assert r["n_tracks"] == 2 and r["duplicates"].sum() == 2


def test_dedup_and_short_tracks():
    boxes, kpts, imgs, truth = _crossing(n=30, swap_slots=False)
    # a duplicate (counter-cut) box of person A in slot 2 on every frame, and a 3-frame spurious detection in slot 3
    for k in range(30):
        a = boxes[k, 0].copy(); a[3] -= 40; a[4] = 0.5; boxes[k, 2] = a; kpts[k, 2] = kpts[k, 0]
    for k in range(5, 8):
        boxes[k, 3] = _person_box(320, 60, 30, 60); kpts[k, 3] = _kpts_for(boxes[k, 3])
    r = T.link_tracks(boxes, kpts, lambda k: imgs[k], W, H, FPS)
    assert r["n_tracks"] == 2 and r["duplicates"][:, 2].all() and (r["ids"][:, 3] == -1).all()
    # a small person fully inside a big person's box is NOT a duplicate
    big = np.array([[100, 10, 500, 350, 0.9], [300, 20, 420, 250, 0.8], *([[np.nan] * 5] * 2)], np.float32)
    kept, dropped = T.dedup(big, [0, 1], T.DEDUP_OVERLAP, None)
    assert kept == [0, 1] and dropped == []


def test_switch_metric():
    n = 10; b = np.full((n, 4, 5), np.nan, np.float32)
    for k in range(n):
        b[k, 0] = _person_box(100, 180); b[k, 1] = _person_box(500, 180)
    assert T.count_switches(b, W) == 0
    b[5, 0], b[5, 1] = b[5, 1].copy(), b[5, 0].copy()  # a slot swap at frame 5 and back at 6: two jumps of 400 px = 62 % of W
    assert T.count_switches(b, W) == 2
    b[6, 1] = np.nan  # only one person at frame 6: the second jump no longer counts
    assert T.count_switches(b, W) == 1


def test_match_tracks_to_world_by_reprojection():
    from duet.playground import geometry as G
    intr = G.Intrinsics(np.array([[500.0, 0, 320], [0, 500.0, 180], [0, 0, 1]]), np.zeros(5), W, H)
    T_wc = np.eye(4); n, P = 20, 2
    bodies = np.full((n, 2, 17, 3), np.nan, np.float32)
    for k in range(n):
        for p in range(2):
            bodies[k, p, :, 0] = (-0.6 if p == 0 else 0.6) + np.linspace(-0.1, 0.1, 17); bodies[k, p, :, 1] = np.linspace(-0.5, 0.5, 17); bodies[k, p, :, 2] = 3.0
    kpts = np.full((n, P, 17, 3), np.nan, np.float32)
    for k in range(n):
        for p in range(2):
            px = G.project(intr, G.inv(T_wc), bodies[k, p]); kpts[k, 1 - p, :, :2] = px + np.random.default_rng(k).normal(0, 2, px.shape); kpts[k, 1 - p, :, 2] = 0.9
    gid, dist, used = T.match_tracks_to_world(kpts, bodies, intr, T_wc, W)
    assert gid.tolist() == [1, 0] and used == n and np.all(dist < 0.02)


def test_match_tracks_across_views_fallback():
    n = 30; hA, hB = np.zeros((16, 8), np.float32), np.zeros((16, 8), np.float32); hA[0, 7] = 1; hB[8, 7] = 1
    def view(order):
        b = np.full((n, 2, 5), np.nan, np.float32)
        for k in range(n):
            b[k, order[0]] = _person_box(100, 180); b[k, order[1]] = _person_box(500, 180)
        return b
    # front: A (hA) left, B (hB) right. back: track 0 is on the RIGHT and has A's colours -> the views are mirrored
    views = {"front": {"boxes": view((0, 1)), "hists": [hA, hB], "img_w": W}, "back": {"boxes": view((1, 0)), "hists": [hA, hB], "img_w": W}}
    gids, info = T.match_tracks_across_views(views)
    assert gids["front"].tolist() == [0, 1] and gids["back"].tolist() == [0, 1]
    pair = info["pairs"]["front->back"]
    assert pair["orientation"] == "mirrored" and pair["lr_agreement_mirrored"] == 1.0 and not pair["ambiguous"]
    # same colours on the same side -> same orientation; identical colours -> ambiguous
    views["back"]["boxes"] = view((0, 1)); _, info = T.match_tracks_across_views(views); assert info["pairs"]["front->back"]["orientation"] == "same"
    views["back"]["hists"] = [hA, hA]; views["front"]["hists"] = [hA, hA]; _, info = T.match_tracks_across_views(views); assert info["pairs"]["front->back"]["ambiguous"]


def test_stage_end_to_end(tmp_path):
    n = 30; boxes, kpts, imgs, truth = _crossing(n=n)
    ep = Episode(name="t", root=str(tmp_path), streams=[Stream("cam", "exo", "streams/cam.mp4"), Stream("cam2", "exo", "streams/cam2.mp4"), Stream("ego", "ego", "streams/ego.mp4", person="p")],
                 reference="cam", proc_fps=FPS, common_start_s=0.0, common_end_s=n / FPS)
    (tmp_path / "derived" / "body2d").mkdir(parents=True)
    for name in ("cam", "cam2"):
        fd = tmp_path / "derived" / "frames" / name; fd.mkdir(parents=True)
        for k in range(n):
            cv2.imwrite(str(fd / f"{k+1:06d}.jpg"), imgs[k])
        np.savez_compressed(tmp_path / "derived" / "body2d" / f"{name}.npz", kpts=kpts, boxes=boxes, img_w=W, img_h=H)
    ep.save(); T.track(ep)
    assert ep.status["track"]["state"] == "done", ep.status["track"]
    z = np.load(tmp_path / "derived" / "track" / "cam.npz")
    assert z["ids"].shape == (n, 4) and z["boxes"].shape == (n, 2, 5) and z["kpts"].shape == (n, 2, 17, 3) and int(z["n_tracks"]) == 2
    assert _ids_consistent(z["ids"], truth, boxes)
    rep = json.load(open(tmp_path / "derived" / "track" / "report.json"))
    assert rep["streams"]["cam"]["switches_after"] == 0 and rep["cross_view"]["method"].startswith("cooccurrence")
    assert z["global_ids"].tolist() == [0, 1]  # cam is the reference view
    import pandas as pd
    df = pd.read_parquet(tmp_path / "derived" / "track" / "records_extra.parquet")
    assert len(df) == n and "cam_t0_cx" in df and "cam_switch" in df and "ego" not in "".join(df.columns)


def test_stage_skips_without_body2d(tmp_path):
    ep = Episode(name="t", root=str(tmp_path), streams=[Stream("cam", "exo", "streams/cam.mp4")], reference="cam")
    ep.save(); T.track(ep)
    assert ep.status["track"]["state"] == "skipped"
