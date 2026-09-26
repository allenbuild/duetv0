"""Release packaging on a tiny synthetic episode (no face detector): schema, manifest, consent, licence, README, verify,
plus the pure tracking/blur helpers. The real CoMind records.parquet, when present, must have <= 10% unknown columns."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.playground import release as R
from duet.playground.episode import Episode, Stream

ROOT = Path(__file__).resolve().parents[1]
COMIND = ROOT / "data/playground/episodes/comind_43276420_clip"


def _synthetic_episode(tmp_path: Path, n: int = 5) -> Episode:
    d = tmp_path / "eps" / "syn"; (d / "streams").mkdir(parents=True); (d / "derived" / "speech").mkdir(parents=True); (d / "derived" / "contact").mkdir()
    for name in ("a", "b", "cam"):
        (d / "streams" / f"{name}.mp4").write_bytes(b"\x00" * 64)  # stand-in; --no-blur copies it
    ep = Episode(name="syn", root=str(d), streams=[Stream("a", "ego", "streams/a.mp4", person="alice"), Stream("b", "ego", "streams/b.mp4", person="bob"), Stream("cam", "exo", "streams/cam.mp4")],
                 reference="a", common_start_s=1.0, common_end_s=1.0 + n / 10, status={"probe": {"state": "done"}, "export": {"state": "done", "detail": "x"}})
    ep.save()
    cols = {"t_s": 1.0 + np.arange(n) / 10.0, "a_body2d_p0": list(np.zeros((n, 51), np.float32)), "cam_body2d_p1": list(np.zeros((n, 51), np.float32)),
            "a_hand_L_present": np.ones(n, bool), "a_hand_L_score": np.ones(n, np.float32), "a_hand_L_lm2d": list(np.zeros((n, 42), np.float32)), "a_hand_R_lm3d": list(np.zeros((n, 63), np.float32)),
            "b_objects": ["[]"] * n, "cam_qc_sharp": np.ones(n), "world_body_p0": list(np.zeros((n, 51), np.float32)), "world_hands3d_a": list(np.zeros((n, 126), np.float32)),
            "min_wrist_dist_m": np.ones(n, np.float32), "a_facing_partner_cos": np.ones(n, np.float32), "body3d_p0": list(np.zeros((n, 99), np.float32)), "body3d_stream": ["cam"] * n,
            "track_id_cam_p0": np.zeros(n), "contact_object_alice_R": ["bowl"] * n, "speech_speaking_alice": np.zeros(n, bool), "autolabel_handover": np.zeros(n, bool),
            "metrics_hand_speed_bob": np.zeros(n, np.float32), "gaze_proxy_alice_looking_at": ["bob"] * n, "mystery_column": np.zeros(n)}
    pd.DataFrame(cols).to_parquet(d / "derived" / "records.parquet", index=False)
    json.dump({"segments": []}, open(d / "derived/speech/transcript.json", "w")); json.dump([], open(d / "derived/contact/events.json", "w"))
    return ep


def test_schema_covers_known_families_and_flags_unknown(tmp_path):
    ep = _synthetic_episode(tmp_path); df = pd.read_parquet(ep.derived / "records.parquet")
    sch = R.build_schema(df, [s.name for s in ep.streams], R.persons_in(ep))
    by = {c["name"]: c for c in sch["columns"]}
    assert sch["n_columns"] == len(df.columns) and sch["n_unknown"] == 1 and by["mystery_column"]["meaning"] == "unknown" and sch["unknown_fraction"] <= 0.10
    assert by["a_body2d_p0"]["shape"] == [51] and by["a_body2d_p0"]["dtype"] == "list[float32]" and "stream a" in by["a_body2d_p0"]["meaning"] and by["a_body2d_p0"]["unit"] == "px"
    assert by["a_hand_R_lm3d"]["shape"] == [63] and by["a_hand_R_lm3d"]["unit"] == "m" and "R hand" in by["a_hand_R_lm3d"]["meaning"]
    assert by["world_hands3d_a"]["shape"] == [126] and by["body3d_p0"]["shape"] == [99] and by["b_objects"]["dtype"] == "str" and by["t_s"]["unit"] == "s"
    assert by["speech_speaking_alice"]["unit"] == "bool" and "alice" in by["speech_speaking_alice"]["meaning"]
    assert by["metrics_hand_speed_bob"]["meaning"].startswith("smoothed wrist speed of bob") and by["track_id_cam_p0"]["meaning"].startswith("person identity tracking")
    assert by["contact_object_alice_R"]["unit"] == "label" and by["gaze_proxy_alice_looking_at"]["meaning"].startswith("gaze proxy") and by["autolabel_handover"]["unit"] == "label"


def test_release_without_blur_writes_everything_and_verifies(tmp_path):
    ep = _synthetic_episode(tmp_path); out_root = tmp_path / "rel"
    out = R.release(ep, "v0.0.1", out_root, blur=False, log=lambda *a: None)
    assert out == out_root / "v0.0.1" / "syn"
    for f in ("streams/a.mp4", "streams/b.mp4", "streams/cam.mp4", "derived/records.parquet", "episode.json", "derived/speech/transcript.json", "derived/contact/events.json", "schema.json", "manifest.json", "LICENSE", "consent.json", "README.md"):
        assert (out / f).exists(), f
    assert not (out / "derived/qc/report.json").exists()  # absent in the episode -> not copied, not listed
    man = json.load(open(out / "manifest.json")); listed = {f["path"] for f in man["files"]}
    assert "manifest.json" not in listed and "streams/a.mp4" in listed and "schema.json" in listed and man["release_version"] == "v0.0.1" and man["episode"] == "syn"
    assert man["stage_status"]["export"]["state"] == "done" and man["versions"]["numpy"] and man["created_utc"].endswith("Z") and man["face_blur"]["backend"].startswith("none")
    assert man["copied"] == ["derived/records.parquet", "episode.json", "derived/speech/transcript.json", "derived/contact/events.json"]
    rec = next(f for f in man["files"] if f["path"] == "streams/a.mp4"); assert rec["bytes"] == 64 and rec["sha256"] == R.sha256(out / "streams/a.mp4")
    con = json.load(open(out / "consent.json"))
    assert [p["name"] for p in con["persons"]] == ["alice", "bob"] and all(p["consent_recorded"] is False and p["scope"] == "research dataset" and p["contact"] == "" and p["date"] == "" for p in con["persons"])
    lic = (out / "LICENSE").read_text(); assert "DRAFT, founders to confirm" in lic and "CC BY-NC 4.0" in lic and "creativecommons.org/licenses/by-nc/4.0" in lic
    readme = (out / "README.md").read_text(); assert "NOT anonymised" in readme and "schema.json" in readme and "--verify" in readme
    sch = json.load(open(out / "schema.json")); assert sch["n_columns"] == 22 and sch["unknown_fraction"] <= 0.10
    rep = R.verify_release(out); assert rep["ok"] and rep["checked"] == len(listed) and not rep["untracked"]
    # tamper -> mismatch; remove -> missing; add -> untracked
    (out / "consent.json").write_text("{}"); (out / "LICENSE").unlink(); (out / "extra.txt").write_text("x")
    rep = R.verify_release(out); assert not rep["ok"] and rep["mismatched"] == ["consent.json"] and rep["missing"] == ["LICENSE"] and rep["untracked"] == ["extra.txt"]


def test_consent_falls_back_to_placeholders_without_persons(tmp_path):
    d = tmp_path / "e"; d.mkdir(); ep = Episode(name="e", root=str(d), streams=[Stream("x", "ego", "x.mp4"), Stream("y", "ego", "y.mp4"), Stream("c", "exo", "c.mp4")])
    assert [p["name"] for p in R.consent_template(ep)["persons"]] == ["person_0", "person_1"]


def test_tracker_persists_half_a_second_and_matches_by_iou():
    tr = R.FaceTracker(fps=30.0); assert tr.persist == 15
    assert len(tr.update(0, [(100, 100, 200, 200, 0.9, "t")])) == 1
    assert len(tr.update(5, [(105, 102, 205, 202, 0.9, "t")])) == 1  # same face, moved slightly -> one track
    assert len(tr.update(5 + 15, [])) == 1                            # still blurred 0.5 s after the last detection
    assert len(tr.update(5 + 16, [])) == 0                            # then dropped
    tr.update(40, [(0, 0, 50, 50, 0.9, "t")]); assert len(tr.update(41, [(400, 400, 450, 450, 0.9, "t")])) == 2  # no overlap -> a second track


def test_enlarge_and_blur_helpers():
    assert R.enlarge_box((100, 100, 200, 200), 0.3, 1280, 720) == (85, 85, 215, 215)
    assert R.enlarge_box((0, 0, 100, 100), 0.3, 1280, 720) == (0, 0, 115, 115)  # clipped to the frame
    assert R.blur_kernel((0, 0, 100, 100)) == 61 and R.blur_kernel((0, 0, 10, 10)) == 15 and R.blur_kernel((0, 0, 100, 100)) % 2 == 1
    img = np.zeros((200, 200, 3), np.uint8); yy, xx = np.mgrid[:200, :200]; img[((yy // 2 + xx // 2) % 2) == 1] = 255  # fine checkerboard everywhere
    out = R.blur_boxes(img.copy(), [np.array([80, 80, 120, 120], float)])
    assert out[85:115, 85:115].std() < 0.2 * img[85:115, 85:115].std()  # the face region is smoothed out
    assert np.array_equal(out[:60], img[:60]) and np.array_equal(out[:, 140:], img[:, 140:])  # outside the enlarged box nothing changes
    assert len(R._nms([(0, 0, 10, 10, 0.9, "a"), (1, 1, 11, 11, 0.5, "b"), (50, 50, 60, 60, 0.7, "c")])) == 2


def test_filter_boxes_needs_corroboration_for_tile_detections_and_caps_size():
    W, H = 1280, 720
    pose = (600, 100, 700, 200, 0.9, "pose"); tile_on_face = (610, 110, 690, 190, 0.7, "mp_short_tiles")
    tile_on_bowl = (900, 400, 1100, 600, 0.75, "mp_short_tiles"); tile_confident_alone = (100, 100, 180, 180, 0.9, "mp_short_tiles")
    huge = (0, 0, 400, 400, 0.95, "mp_full")  # 400 px > 35 % of 720
    kept = R.filter_boxes([pose, tile_on_face, tile_on_bowl, tile_confident_alone, huge], W, H)
    assert pose in kept and tile_on_face in kept and tile_confident_alone in kept and tile_on_bowl not in kept and huge not in kept
    assert R.filter_boxes([tile_on_bowl], W, H) == []  # a lone mid-score tile detection never blurs on its own


@pytest.mark.skipif(not (COMIND / "derived/records.parquet").exists(), reason="CoMind playground episode not present")
def test_real_records_schema_has_few_unknown_columns():
    ep = Episode.load(COMIND); df = pd.read_parquet(ep.derived / "records.parquet")
    sch = R.build_schema(df, [s.name for s in ep.streams], R.persons_in(ep))
    unknown = [c["name"] for c in sch["columns"] if c["meaning"] == "unknown"]
    assert sch["unknown_fraction"] <= 0.10, f"{len(unknown)}/{sch['n_columns']} unknown columns: {unknown}"
