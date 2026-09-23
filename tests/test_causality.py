"""Every predictor in the benchmark must be strictly causal.

A GroupNorm layer silently normalised over the time axis in v0, leaking future frames
into the present and making a 512-frame training crop statistically different from a
40k-frame evaluation pass; the handover heads sat at chance until it was found. These
tests are the guard against that class of bug returning anywhere in the stack.

The shared method: compute the output, perturb the input strictly after frame ``t``,
recompute, and assert nothing at or before ``t`` moved.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from duet.adapters.comind.labels import onset_within, time_to_next_onset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

torch = pytest.importorskip("torch", reason="ml extra not installed")

T, CUT = 256, 120


def _perturbed(x: np.ndarray, cut: int, rng) -> np.ndarray:
    """Copy of x with everything strictly after `cut` replaced by noise."""
    y = x.copy()
    y[cut + 1 :] = rng.normal(size=y[cut + 1 :].shape).astype(y.dtype) * 10.0
    return y


def test_causal_tcn_output_ignores_the_future():
    from duet.ml.paired_benchmark import CausalTCN, add_deltas

    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    model = CausalTCN(d_in=2 * 8, c=16, n_cls=3).eval()
    x = rng.normal(size=(T, 8)).astype(np.float32)
    xp = _perturbed(x, CUT, rng)

    with torch.no_grad():
        a, ra = model(add_deltas(torch.from_numpy(x[None])))
        b, rb = model(add_deltas(torch.from_numpy(xp[None])))

    assert torch.allclose(a[0, : CUT + 1], b[0, : CUT + 1], atol=1e-6), "classification head sees the future"
    assert torch.allclose(ra[0, : CUT + 1], rb[0, : CUT + 1], atol=1e-6), "regression head sees the future"
    # the perturbation must actually reach the output after the cut, else the test is vacuous
    assert not torch.allclose(a[0, CUT + 1 :], b[0, CUT + 1 :], atol=1e-6)


def test_channel_norm_statistics_do_not_span_time():
    """ChannelNorm must normalise per time step, so one frame cannot alter another."""
    from duet.ml.paired_benchmark import ChannelNorm

    torch.manual_seed(0)
    norm = ChannelNorm(6).eval()
    x = torch.randn(1, 6, T)
    y = x.clone()
    y[:, :, CUT + 1 :] += 50.0
    with torch.no_grad():
        a, b = norm(x), norm(y)
    assert torch.allclose(a[:, :, : CUT + 1], b[:, :, : CUT + 1], atol=1e-6)


def test_channel_norm_is_length_invariant():
    """A training crop and a full sequence must give identical values on shared frames."""
    from duet.ml.paired_benchmark import ChannelNorm

    torch.manual_seed(0)
    norm = ChannelNorm(6).eval()
    x = torch.randn(1, 6, T)
    with torch.no_grad():
        full = norm(x)
        crop = norm(x[:, :, :64])
    assert torch.allclose(full[:, :, :64], crop, atol=1e-6)


def test_causal_window_statistics_ignore_the_future():
    from baseline_gbdt import causal_stats

    rng = np.random.default_rng(1)
    x = rng.normal(size=(T, 4))
    xp = _perturbed(x, CUT, rng)
    a, b = causal_stats(x, 30), causal_stats(xp, 30)
    assert np.allclose(a[: CUT + 1], b[: CUT + 1]), "causal_stats window reaches into the future"
    assert not np.allclose(a[CUT + 1 :], b[CUT + 1 :])


def test_speech_features_ignore_later_words():
    from duet.adapters.comind.speech import speech_features

    early = [(1.0, 1.4, "pass"), (2.0, 2.3, "me")]
    late = early + [(6.0, 6.4, "salt"), (7.0, 7.5, "please")]
    n, cut = 300, int(5.0 * 30)
    a, b = speech_features(early, n), speech_features(late, n)
    assert np.allclose(a[: cut + 1], b[: cut + 1]), "speech features leak later utterances"
    assert not np.allclose(a[cut + 1 :], b[cut + 1 :])


def test_anticipation_labels_never_include_their_own_onset():
    """onset_within is the target of every anticipation claim; it must be strictly ahead."""
    for horizon in (30, 60, 150):
        m = onset_within(500, [300], horizon)
        assert m[300] == 0.0, "the onset frame itself is labelled positive"
        assert m[300 - horizon : 300].all()
        assert m[: 300 - horizon].sum() == 0.0
        assert m[300:].sum() == 0.0
    t = time_to_next_onset(500, [300], max_s=5.0)
    assert np.isnan(t[300]) and t[299] > 0


def test_shuffled_controls_roll_only_their_own_stream():
    """both_shuffled keeps the leader aligned; leader_shuffled keeps the helper aligned."""
    from duet.adapters.comind.kinematics import PERSON_DIM
    from duet.ml.paired_benchmark import view_input

    rng = np.random.default_rng(2)
    x = rng.normal(size=(5000, 2 * PERSON_DIM)).astype(np.float32)
    bs = view_input(x, "both_shuffled", None, rid="rec-a")
    ls = view_input(x, "leader_shuffled", None, rid="rec-a")
    assert np.array_equal(bs[:, :PERSON_DIM], x[:, :PERSON_DIM])
    assert not np.array_equal(bs[:, PERSON_DIM:], x[:, PERSON_DIM:])
    assert np.array_equal(ls[:, PERSON_DIM:], x[:, PERSON_DIM:])
    assert not np.array_equal(ls[:, :PERSON_DIM], x[:, :PERSON_DIM])
    # deterministic per recording: the cache, normaliser and every eval pass must agree
    assert np.array_equal(bs, view_input(x, "both_shuffled", None, rid="rec-a"))
    assert not np.array_equal(bs, view_input(x, "both_shuffled", None, rid="rec-b"))


def test_rgb_forward_axis_is_not_device_z(tmp_path):
    """The wearer's viewing direction is the RGB optical axis, not device +Z.

    Measured on recording 43276420 the two differ by 38.7 degrees, so silently using +Z
    mis-states 'is this person facing their partner' by that much.
    """
    import json

    from duet.adapters.comind.shared_world_features import (
        NOMINAL_RGB_AXIS_DEVICE,
        quat_wxyz_to_R,
        rgb_forward_axis,
    )

    fallback = rgb_forward_axis(tmp_path, "leader")  # no rgb_calib_*.json present
    assert np.allclose(np.linalg.norm(fallback), 1.0)
    angle = np.degrees(np.arccos(np.clip(fallback @ np.array([0.0, 0.0, 1.0]), -1, 1)))
    assert 35.0 < angle < 42.0, f"nominal RGB axis drifted from the measured 38.7 deg: {angle:.1f}"
    assert np.allclose(fallback, NOMINAL_RGB_AXIS_DEVICE / np.linalg.norm(NOMINAL_RGB_AXIS_DEVICE))

    # a per-recording calibration file overrides the nominal axis
    ident = {"T_Device_Camera": {"Translation": [0, 0, 0], "UnitQuaternion": [1.0, [0.0, 0.0, 0.0]]}}
    json.dump(ident, open(tmp_path / "rgb_calib_leader.json", "w"))
    assert np.allclose(rgb_forward_axis(tmp_path, "leader"), [0.0, 0.0, 1.0])
    assert np.allclose(quat_wxyz_to_R(1, 0, 0, 0), np.eye(3))
