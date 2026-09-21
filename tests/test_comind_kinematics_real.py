"""Real-data regression tests for the kinematics adapter (skipped if data is absent)."""
from pathlib import Path

import numpy as np
import pytest

from duet.adapters.comind.kinematics import (
    HAND_DIM,
    NATIVE_FRAME_NS,
    PERSON_DIM,
    build_recording_kinematics,
    parse_mp4_tail,
    person_slice,
)
from duet.adapters.comind.labels import (
    build_frame_labels,
    load_handovers,
    onset_within,
    time_to_next_onset,
)

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/comind"
PROC = ROOT / "data/processed/comind"
RID = "43276420-701f-4731-b9ab-bebc7fd14994"
needs_data = pytest.mark.skipif(not (PROC / RID / "mp4_tail_helper.bin").exists(), reason="CoMind demo recording not downloaded")


@needs_data
def test_mp4_desc_anchor_matches_vrs_image_matching_audit():
    # Values established independently by exhaustive VRS image matching (docs/design/comind_synchronized_v0.md):
    # helper MP4 frame 0 == VRS RGB record 7, leader frame 0 == record 0.
    helper = parse_mp4_tail((PROC / RID / "mp4_tail_helper.bin").read_bytes())
    leader = parse_mp4_tail((PROC / RID / "mp4_tail_leader.bin").read_bytes())
    assert helper.n_frames == leader.n_frames == 21109
    assert leader.frame0_device_time_ns == 825528737150
    assert abs(helper.frame0_device_time_ns - (919390269025 + 7 * NATIVE_FRAME_NS)) < 20_000  # 11 us


@needs_data
def test_feature_coverage_matches_audit():
    out = build_recording_kinematics(RAW, PROC, RID)
    f = out["features"]
    assert f.shape == (21109, 2 * PERSON_DIM)
    assert np.isfinite(f).all()
    s = out["stats"]
    # whole-video high-confidence coverage from the audit: helper L 96.30 / R 97.47, leader L 96.84 / R 93.64 (%)
    assert abs(s["helper"]["left_hand_valid"] - 0.963) < 0.01
    assert abs(s["helper"]["right_hand_valid"] - 0.975) < 0.01
    assert abs(s["leader"]["left_hand_valid"] - 0.968) < 0.01
    assert abs(s["leader"]["right_hand_valid"] - 0.936) < 0.01
    hs = person_slice("helper")
    # audit: helper hands unavailable for the first ~21 frames
    assert f[:20, hs.start + 25].sum() == 0
    # wrist positions are in meters within arm's reach
    wrist = f[f[:, hs.start + 25] > 0, hs.start + 15 : hs.start + 18]
    assert np.all(np.linalg.norm(wrist, axis=1) < 1.2)


@needs_data
def test_labels_for_demo_recording():
    hos = load_handovers(RAW / "annotations", RID)
    assert len(hos) == 6
    lab = build_frame_labels(RAW / "annotations", RID, 21109)["labels"]
    assert lab["handover_active"].sum() > 0
    assert set(np.unique(lab["onset_within_2s"])) <= {0.0, 1.0}
    # each onset contributes exactly 60 positive anticipation frames (unless clipped/overlapping)
    assert lab["onset_within_2s"].sum() <= 60 * len(hos)


def test_onset_within_and_tth_are_strictly_future():
    m = onset_within(100, [50], 10)
    assert m[40:50].all() and m[50] == 0 and m[39] == 0
    t = time_to_next_onset(100, [60], max_s=1.0)
    assert np.isnan(t[60]) and abs(t[59] - 1 / 30) < 1e-6 and np.isnan(t[20])


def test_hand_block_layout():
    assert HAND_DIM == 26 and PERSON_DIM == 59
    assert person_slice("leader") == slice(0, 59) and person_slice("helper") == slice(59, 118)


def test_world_transforms_round_trip():
    from duet.adapters.comind.shared_world_features import quat_xyzw_to_R, to_device, to_world

    rng = np.random.default_rng(0)
    q = rng.normal(size=(5, 4)); q /= np.linalg.norm(q, axis=1, keepdims=True)
    R = quat_xyzw_to_R(q); t = rng.normal(size=(5, 3)); p = rng.normal(size=(5, 3, 3))
    assert np.allclose(np.einsum("nij,nkj->nik", R, R), np.eye(3)[None], atol=1e-9)
    assert np.allclose(to_device(R, t, to_world(R, t, p)), p, atol=1e-9)
    # quarter turn about z maps x -> y (active rotation, same convention as duet.geometry.rotations)
    Rz = quat_xyzw_to_R(np.array([[0, 0, np.sin(np.pi / 4), np.cos(np.pi / 4)]]))
    assert np.allclose(Rz[0] @ np.array([1, 0, 0]), [0, 1, 0], atol=1e-9)
