"""Audio alignment (duet.playground.align): lag axis and sign convention, short and long clips, partial overlap,
silence / unrelated audio, search-window edge, clock drift, the align(ep) stage logic (fake decoder) and end-to-end
align(ep) on small ffmpeg-generated A/V clips (planted +-2.37 s offsets, 0.5 s audio-start delay, silent and
video-only streams). Convention under test (C1): t_ref = (1 + drift_ppm * 1e-6) * t_stream + offset_s."""
from __future__ import annotations

import io
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np
import pytest

from duet.playground import align as A
from duet.playground.episode import Episode, probe

DT = A.DT
needs_ffmpeg = pytest.mark.skipif(not (shutil.which("ffmpeg") and shutil.which("ffprobe")), reason="ffmpeg/ffprobe not installed")


def _env(n: int, seed: int) -> np.ndarray:
    """Sparse heavy-tailed onset-like envelope (the review's synthetic: max(N(0,1), 0)^3, standardised)."""
    x = np.maximum(np.random.default_rng(seed).standard_normal(n), 0) ** 3
    return (x - x.mean()) / x.std()


def _hops(s: float) -> int:
    return round(s / DT)


# ------------------------------------------------------------------------------------------ correlation core

def test_xcorr_score_is_linear_overlap_normalised_and_clamped():
    rng = np.random.default_rng(0); ref, sig = rng.standard_normal(37), rng.standard_normal(23)
    lags, s = A.xcorr_score(ref, sig, -100, 100, min_overlap=5)
    assert lags[0] == 5 - 23 and lags[-1] == 37 - 5 and len(lags) == len(s) == 37 + 23 - 2 * 5 + 1
    for L, v in zip(lags, s, strict=True):  # brute force: Pearson r over the overlap * sqrt(overlap), no circular wrap
        i = np.arange(max(0, -L), min(23, 37 - L))
        assert v == pytest.approx(np.corrcoef(ref[i + L], sig[i])[0, 1] * np.sqrt(len(i)), abs=1e-9)
    assert len(A.xcorr_score(ref, sig, 40, 100, min_overlap=5)[0]) == 0  # no lag with enough overlap


@pytest.mark.parametrize("shift_s", [3.0, -3.0, 2.37, -2.37])
def test_sign_convention_short_clips(shift_s):
    """Clips < 41 s used to raise IndexError. shift > 0: the stream started later than the reference."""
    base, k = _env(6000, 1), _hops(abs(shift_s))
    ref, sig = (base[:4000], base[k:k + 3700]) if shift_s > 0 else (base[k:k + 4000], base[:3700])
    r = A.estimate_offset(ref, sig)
    assert r.ok and r.offset_s == pytest.approx(shift_s, abs=0.006) and r.drift_ppm == 0.0
    assert r.confidence >= A.RATIO_MIN and r.z >= A.Z_MIN


@pytest.mark.parametrize("dur_s,shift_s", [(70, 45.0), (70, -45.0), (82, 60.0), (44, -30.0)])
def test_large_lags_do_not_alias(dur_s, shift_s):
    """44-82 s clips with |lag| >= 43.8 s used to alias to the wrong sign (+45 s came back as -118.84 s)."""
    base, n, k = _env(30000, 2), _hops(dur_s), _hops(abs(shift_s))
    ref, sig = (base[:n], base[k:k + n]) if shift_s > 0 else (base[k:k + n], base[:n])
    r = A.estimate_offset(ref, sig)
    assert r.ok and r.offset_s == pytest.approx(shift_s, abs=0.006)


def test_partial_overlap_slice_inside_long_reference():
    """A 45 s clip starting 50 s into a 100 s reference (old code: +1.39 s, truncated to the shorter length)."""
    base = _env(10000, 3)
    r = A.estimate_offset(base, base[5000:9500])
    assert r.ok and r.offset_s == pytest.approx(50.0, abs=0.006)
    r = A.estimate_offset(base[5000:9500], base)
    assert r.ok and r.offset_s == pytest.approx(-50.0, abs=0.006)


def test_silence_and_unrelated_audio_are_rejected():
    assert not A.estimate_offset(_env(5000, 4), np.zeros(5000)).ok
    assert not A.estimate_offset(np.zeros(5000), _env(5000, 4)).ok
    old_rule = 0
    for seed in range(30):  # unrelated 30-60 s pairs: never accepted
        n = 3000 + 100 * seed
        ref, sig = _env(n, 100 + seed), _env(n, 200 + seed)
        assert not A.estimate_offset(ref, sig).ok
        lags, s = A.xcorr_score(ref, sig, -12000, 12000, 500); p = A.find_peak(lags, s)
        old_rule += p.ratio >= 1.3  # the old acceptance (ratio only, unclipped): some unrelated pairs pass
    assert old_rule > 0


def test_offset_outside_window_is_not_reported():
    base = _env(40000, 5)
    r = A.estimate_offset(base[:20000], base[15000:35000], max_lag_s=120)  # truth +150 s, beyond +-120 s
    assert not r.ok and "no clear audio match" in r.notes[0]
    r = A.estimate_offset(base[:20000], base[15000:35000], max_lag_s=200)
    assert r.ok and r.offset_s == pytest.approx(150.0, abs=0.006)


def test_peak_at_window_edge_is_flagged():
    base = _env(30000, 6)
    r = A.estimate_offset(base[:20000], base[11980:25000], max_lag_s=120)  # truth +119.8 s: within 0.5 s of the edge
    assert not r.ok and "search edge" in r.notes[0]
    r = A.estimate_offset(base[:20000], base[11980:25000], max_lag_s=130)
    assert r.ok and r.offset_s == pytest.approx(119.8, abs=0.006)


@pytest.mark.parametrize("ppm,b", [(50.0, 3.0), (-20.0, -7.5)])
def test_drift_over_one_hour(ppm, b):
    """1 h with the stream clock off by `ppm`: a constant offset would be 180 ms off at 50 ppm; the windowed fit
    recovers t_ref = (1 + d) t + b."""
    n = 360000; base = _env(n + 4000, 7); t = np.arange(n) * DT
    t_ref = (1 + ppm * 1e-6) * t + b + 20.0  # reference = base shifted by 20 s so negative b stays inside it
    sig = np.interp(t_ref / DT, np.arange(len(base)), base); ref = base[_hops(20.0):]
    r = A.estimate_offset(ref, sig)
    assert r.ok and r.drift_ppm == pytest.approx(ppm, abs=0.5) and r.offset_s == pytest.approx(b, abs=0.005)
    tt = np.array([0.0, 1800.0, 3600.0])
    assert np.abs((1 + r.drift_ppm * 1e-6) * tt + r.offset_s - ((1 + ppm * 1e-6) * tt + b)).max() < 0.005
    assert any("drift" in n and "windows" in n for n in r.notes)


def test_large_drift_smearing_the_global_peak_is_recovered_from_windows():
    """400 ppm over 1 h smears the global peak over 1.4 s (plus independent noise): the global test fails, the
    full-range window search agrees on the lag, and the fit recovers offset and drift."""
    n = 360000; base = _env(n + 40000, 12); t = np.arange(n) * DT
    sig = np.interp(((1 + 400e-6) * t + 30.0) / DT, np.arange(len(base)), base) + 1.5 * _env(n, 13)
    r = A.estimate_offset(base, sig)
    assert r.ok and r.offset_s == pytest.approx(30.0, abs=0.005) and r.drift_ppm == pytest.approx(400.0, abs=1.0)
    assert "global correlation weak" in r.notes[0] and "windows agree" in r.notes[0]


def test_short_recording_reports_zero_drift_with_note():
    base = _env(20000, 8); r = A.estimate_offset(base[:12000], base[500:12000])
    assert r.ok and r.drift_ppm == 0.0 and "drift not estimated" in r.notes[0]


def test_onset_envelope_ignores_padding_boundaries():
    e = np.r_[np.zeros(50), np.full(50, 1e-2), np.zeros(20), np.full(30, 1e-1)]
    on = A.onset_envelope(e)
    assert on[50] == 0 and on[120] == 0 and on.max() == 0  # rises right after digital silence are padding, not onsets
    e2 = np.r_[np.full(50, 1e-4), np.full(50, 1e-2)]
    assert A.onset_envelope(e2)[50] > 1  # a real rise from (quiet) sound is an onset


# ------------------------------------------------------------------------------------------ align(ep) logic

def _episode(tmp_path: Path, spec: dict[str, dict], reference: str, offsets: dict | None = None) -> Episode:
    """Episode over placeholder files; spec[name] = {duration_s, has_audio, video_start_s} (no probe needed)."""
    src = tmp_path / "src"; src.mkdir(parents=True, exist_ok=True)
    for name in spec:
        (src / f"{name}.mp4").write_bytes(b"\0")
    ep = Episode.create(tmp_path / "eps", "ep", [(n, "exo", src / f"{n}.mp4", None) for n in spec], reference=reference, offsets=offsets)
    for s in ep.streams:
        s.duration_s, s.has_audio, s.video_start_s = spec[s.name]["duration_s"], spec[s.name]["has_audio"], spec[s.name].get("video_start_s", 0.0)
    return ep


def _fake_decoder(monkeypatch, table: dict):
    calls = []

    def fake(path, video_start_s=0.0, timeout_s=None):
        calls.append(Path(path).stem); return table[Path(path).stem]
    monkeypatch.setattr(A, "decode_envelope", fake)
    return calls


def test_align_statuses_offsets_and_common_window(tmp_path, monkeypatch):
    base = _env(8000, 9); ok = lambda e: (e, -12.0, "")
    table = {"ref": ok(base[500:5500]), "late": ok(base[737:5000]), "early": ok(base[263:5100]), "silent": (np.zeros(5000), -np.inf, ""),
             "broken": (None, -np.inf, "ffmpeg exit 1: Invalid data")}
    spec = {"ref": {"duration_s": 50.0, "has_audio": True}, "late": {"duration_s": 42.63, "has_audio": True},
            "early": {"duration_s": 48.37, "has_audio": True}, "silent": {"duration_s": 50.0, "has_audio": True},
            "broken": {"duration_s": 50.0, "has_audio": True}, "noaudio": {"duration_s": 50.0, "has_audio": False},
            "manual": {"duration_s": 30.0, "has_audio": False}}
    ep = _episode(tmp_path, spec, "ref", offsets={"manual": 4.0})
    calls = _fake_decoder(monkeypatch, table)
    A.align(ep)
    assert "manual" not in calls and "noaudio" not in calls
    ep = Episode.load(ep.dir); st = {s.name: s for s in ep.streams}
    assert st["ref"].offset_status == "reference" and st["ref"].offset_s == 0.0 and st["ref"].offset_confidence is None
    assert st["late"].offset_status == "audio" and st["late"].offset_s == pytest.approx(2.37, abs=0.006)
    assert st["early"].offset_status == "audio" and st["early"].offset_s == pytest.approx(-2.37, abs=0.006)
    assert st["late"].offset_confidence >= A.RATIO_MIN
    assert st["manual"].offset_status == "manual" and st["manual"].offset_s == 4.0 and st["manual"].offset_confidence is None
    for n in ("silent", "broken", "noaudio"):  # the old code applied -120 s (silent) / 0 (others) and used them
        assert st[n].offset_status == "unaligned" and st[n].offset_s == 0.0 and not ep.usable(st[n])
    # window over usable streams in reference time: ref [0, 50], late [2.37, 45], early [-2.37, 46], manual [4, 34]
    assert ep.common_start_s == pytest.approx(4.0) and ep.common_end_s == pytest.approx(34.0)
    assert ep.status["align"]["state"] == "done" and "3 stream(s) unaligned" in ep.status["align"]["detail"]
    notes = " | ".join(ep.notes)
    assert "silent: silent audio" in notes and "broken: audio decode failed" in notes and "noaudio: no audio track" in notes


def test_align_applies_drift_and_window_uses_ref_time(tmp_path, monkeypatch):
    n = 360000; base = _env(n + 4000, 10); t = np.arange(n) * DT
    sig = np.interp(((1 + 50e-6) * t + 3.0) / DT, np.arange(len(base)), base)
    ep = _episode(tmp_path, {"ref": {"duration_s": 3640.0, "has_audio": True}, "b": {"duration_s": 3600.0, "has_audio": True}}, "ref")
    _fake_decoder(monkeypatch, {"ref": (base, -10.0, ""), "b": (sig, -10.0, "")})
    A.align(ep)
    b = ep.stream("b")
    assert b.offset_status == "audio" and b.drift_ppm == pytest.approx(50.0, abs=0.5) and b.offset_s == pytest.approx(3.0, abs=0.005)
    assert float(ep.ref_time(b, 3600.0)) == pytest.approx((1 + 50e-6) * 3600 + 3.0, abs=0.005)
    # b ends first, at (1 + 50e-6) * 3600 + 3 = 3603.18 s reference time (3603.00 if drift were ignored)
    assert ep.common_start_s == pytest.approx(3.0, abs=0.005) and ep.common_end_s == pytest.approx(3603.18, abs=0.005)
    assert any("drift moves this stream" in x for x in ep.notes)  # 180 ms > half a processed frame at 10 fps


def test_reference_without_usable_audio_fails_unless_all_overridden(tmp_path, monkeypatch):
    spec = {"ref": {"duration_s": 50.0, "has_audio": True}, "b": {"duration_s": 50.0, "has_audio": True}}
    ep = _episode(tmp_path, spec, "ref")
    _fake_decoder(monkeypatch, {"ref": (np.zeros(5000), -np.inf, ""), "b": (_env(5000, 11), -10.0, "")})
    with pytest.raises(RuntimeError, match="reference stream ref has no usable audio"):
        A.align(ep)
    ep = Episode.load(ep.dir)
    assert ep.stream("b").offset_status == "unaligned" and ep.common_end_s == ep.common_start_s == 0.0
    ep2 = _episode(tmp_path / "2", spec, "ref", offsets={"b": 1.25})
    calls = _fake_decoder(monkeypatch, {})
    A.align(ep2)  # every other stream has an override: no decoding at all
    assert calls == [] and ep2.stream("b").offset_status == "manual" and (ep2.common_start_s, ep2.common_end_s) == (1.25, 50.0)


def test_envelope_shorter_than_min_overlap_gives_no_lags(tmp_path, monkeypatch):
    """Either envelope shorter than min_overlap: no lag qualifies (used to trip `assert n.min() >= min_overlap`)."""
    rng = np.random.default_rng(14)
    for nr, ns, m in [(5000, 300, 500), (300, 5000, 500), (3000, 6000, 4000)]:
        lags, s = A.xcorr_score(rng.standard_normal(nr), rng.standard_normal(ns), -12000, 12000, m)
        assert len(lags) == len(s) == 0
    for _ in range(200):  # every returned lag has overlap >= m, and every such lag in [lo, hi] is returned
        nr, ns, m = rng.integers(1, 40, 3); lo, hi = sorted(rng.integers(-50, 50, 2))
        lags, _ = A.xcorr_score(rng.standard_normal(nr), rng.standard_normal(ns), lo, hi, m)
        ok = [L for L in range(lo, hi + 1) if min(ns + min(0, L), nr - max(0, L)) >= max(m, 2)]
        assert list(lags) == ok
    r = A.estimate_offset(_env(5000, 15), _env(300, 16))  # a 3 s stream against the default 5 s minimum overlap
    assert not r.ok and "audio overlap < 5 s at every lag" in r.notes[0]
    ep = _episode(tmp_path, {"ref": {"duration_s": 50.0, "has_audio": True}, "short": {"duration_s": 3.0, "has_audio": True},
                             "b": {"duration_s": 42.63, "has_audio": True}}, "ref")
    base = _env(8000, 17)
    _fake_decoder(monkeypatch, {"ref": (base[500:5500], -10.0, ""), "short": (base[900:1200], -10.0, ""), "b": (base[737:5000], -10.0, "")})
    A.align(ep)
    assert ep.stream("short").offset_status == "unaligned" and ep.stream("b").offset_status == "audio"
    assert any("short: audio 3.0 s < min overlap 5 s" in n for n in ep.notes)


def _no_audio_ref_case(tmp_path, monkeypatch, explicit: bool, offsets: dict | None = None, c_len: int = 5000):
    """Default-first stream a_cam has no audio; b (42.63 s) started 2.37 s after c (c_len hops) in the same room."""
    spec = {"a_cam": {"duration_s": 50.0, "has_audio": False}, "b": {"duration_s": 42.63, "has_audio": True},
            "c": {"duration_s": 50.0, "has_audio": True}, "z": {"duration_s": 50.0, "has_audio": False}}
    ep = _episode(tmp_path, spec, "a_cam", offsets=offsets); ep.reference_explicit = explicit
    base = _env(8000, 18)
    _fake_decoder(monkeypatch, {"b": (base[737:5000], -10.0, ""), "c": (base[500:500 + c_len], -10.0, "")})
    return ep


def test_default_reference_without_audio_switches_to_longest_usable_audio(tmp_path, monkeypatch):
    ep = _no_audio_ref_case(tmp_path, monkeypatch, explicit=False)
    A.align(ep)
    ep = Episode.load(ep.dir); st = {s.name: s for s in ep.streams}
    assert ep.reference == "c" and st["c"].offset_status == "reference" and ep.usable("c")  # 50 s beats b's 42.63 s
    assert st["b"].offset_status == "audio" and st["b"].offset_s == pytest.approx(2.37, abs=0.006)
    assert st["a_cam"].offset_status == "unaligned" and not ep.usable("a_cam") and st["z"].offset_status == "unaligned"
    assert ep.status["align"]["state"] == "done" and ep.status["align"]["detail"].startswith("reference a_cam -> c (auto:")
    assert any("switched to c" in n for n in ep.notes)
    assert (ep.common_start_s, ep.common_end_s) == pytest.approx((2.37, 45.0), abs=0.006)
    ep2 = _no_audio_ref_case(tmp_path / "tie", monkeypatch, explicit=False, c_len=4263)  # equal lengths: ties by name
    A.align(ep2)
    assert ep2.reference == "b" and ep2.stream("c").offset_s == pytest.approx(-2.37, abs=0.006)


def test_explicit_reference_or_manual_offsets_keep_the_failure(tmp_path, monkeypatch):
    ep = _no_audio_ref_case(tmp_path, monkeypatch, explicit=True)
    with pytest.raises(RuntimeError, match="reference stream a_cam has no usable audio"):
        A.align(ep)
    assert Episode.load(ep.dir).reference == "a_cam"
    ep = _no_audio_ref_case(tmp_path / "2", monkeypatch, explicit=False, offsets={"z": 1.0})  # offsets are relative to a_cam
    with pytest.raises(RuntimeError, match=r"not switched automatically: manual offsets \(z\)"):
        A.align(ep)
    assert Episode.load(ep.dir).reference == "a_cam"
    ep = _no_audio_ref_case(tmp_path / "3", monkeypatch, explicit=False)
    _fake_decoder(monkeypatch, {"b": (np.zeros(4263), -np.inf, ""), "c": (np.zeros(5000), -np.inf, "")})  # all silent
    with pytest.raises(RuntimeError, match="no other stream has usable audio either"):
        A.align(ep)


def test_common_window_below_minimum_fails(tmp_path, monkeypatch):
    spec = {"ref": {"duration_s": 10.0, "has_audio": False}, "b": {"duration_s": 10.0, "has_audio": False}}
    ep = _episode(tmp_path, spec, "ref", offsets={"b": 7.0})  # overlap [7, 10] = 3 s < 5 s
    _fake_decoder(monkeypatch, {})
    with pytest.raises(RuntimeError, match="common window of the usable streams is 3.00 s"):
        A.align(ep)
    ep = _episode(tmp_path / "2", spec, "ref", offsets={"b": 7.0})
    A.align(ep, min_overlap_s=2.0)
    assert (ep.common_start_s, ep.common_end_s) == (7.0, 10.0)
    ep = _episode(tmp_path / "3", spec, "ref", offsets={"b": 7.0}); ep.min_overlap_s = 2.0  # the episode setting is honoured
    A.align(ep)
    assert (ep.common_start_s, ep.common_end_s) == (7.0, 10.0)


def test_decode_puts_safe_input_options_before_input(monkeypatch, tmp_path):
    seen = {}

    class FakePopen:
        def __init__(self, cmd, **kw):
            seen["cmd"] = cmd; self.stdout = io.BytesIO()

        def wait(self):
            return 1
    monkeypatch.setattr(A.subprocess, "Popen", FakePopen)
    env, _, err = A.decode_envelope(tmp_path / "x.mp4", 0.25)
    cmd = seen["cmd"]; i = cmd.index("-i")
    assert env is None and err.startswith("ffmpeg exit 1")
    assert cmd[i - 4:i] == ["-protocol_whitelist", "file", "-format_whitelist", A.media_input("x")[3]]
    assert cmd[i + 1].startswith("file:") and "-copyts" in cmd[:i] and "first_pts=2000" in " ".join(cmd)


# ------------------------------------------------------------------------------------------ end to end (ffmpeg)

SRC_SR = 16000


def _source_audio(dur_s: float, seed: int = 0) -> np.ndarray:
    """Irregular decaying sine bursts over a faint noise floor (non-periodic: the correlation peak is unique)."""
    rng = np.random.default_rng(seed); n = int(dur_s * SRC_SR); x = 0.002 * rng.standard_normal(n); t = 0.0
    while (t := t + rng.uniform(0.12, 0.6)) < dur_s - 0.2:
        i = int(t * SRC_SR); tt = np.arange(int(rng.uniform(0.02, 0.12) * SRC_SR)) / SRC_SR
        x[i:i + len(tt)] += rng.uniform(0.1, 0.6) * np.sin(2 * np.pi * rng.uniform(300, 2500) * tt) * np.exp(-tt / 0.03)
    return np.clip(x, -1, 1)


def _wav(path: Path, x: np.ndarray) -> Path:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SRC_SR); w.writeframes((x * 32767).astype("<i2").tobytes())
    return path


def _clip(out: Path, dur_s: float, wav: Path | None = None, audio_delay_s: float = 0.0, silent: bool = False) -> Path:
    """dur_s of testsrc2 video (starts at t=0) + AAC audio from `wav` (optionally starting audio_delay_s late) or a
    digitally silent track, or no audio."""
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=96x64:rate=30:duration={dur_s}"]
    if wav is not None:
        cmd += ["-itsoffset", f"{audio_delay_s}", "-i", str(wav)]
    elif silent:
        cmd += ["-f", "lavfi", "-i", f"anullsrc=r={SRC_SR}:cl=mono"]
    if wav is not None or silent:
        cmd += ["-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "64k"]
    subprocess.run(cmd + ["-c:v", "mpeg4", "-t", f"{dur_s}", str(out)], check=True)
    return out


@needs_ffmpeg
def test_end_to_end_planted_offsets_audio_start_delay_and_unusable_streams(tmp_path):
    src = _source_audio(30.0); m = tmp_path / "media"; m.mkdir()
    cut = lambda name, a, b: _wav(m / f"{name}.wav", src[int(a * SRC_SR):int(b * SRC_SR)])
    vids = [("ref", _clip(m / "ref.mp4", 14, cut("ref", 5.0, 19.0))),
            ("late", _clip(m / "late.mp4", 12, cut("late", 7.37, 19.37))),  # started 2.37 s after ref -> +2.37
            ("early", _clip(m / "early.mkv", 12, cut("early", 2.63, 14.63))),  # started 2.37 s before ref -> -2.37
            # audio track starts 0.5 s after the video but on the same clock (first 0.5 s missing): truth 0.00
            ("astart", _clip(m / "astart.mp4", 14, cut("astart", 5.5, 19.0), audio_delay_s=0.5)),
            ("silent", _clip(m / "silent.mp4", 14, silent=True)), ("noaudio", _clip(m / "noaudio.mp4", 14))]
    ep = Episode.create(tmp_path / "eps", "e2e", [(n, "exo", p, None) for n, p in vids], reference="ref")
    probe(ep); A.align(ep)
    ep = Episode.load(ep.dir); st = {s.name: s for s in ep.streams}
    assert 0.4 < st["astart"].audio_start_s < 0.55  # audio starts late (ffprobe start_time includes AAC priming)
    for name, truth in [("late", 2.37), ("early", -2.37), ("astart", 0.0)]:
        assert st[name].offset_status == "audio" and st[name].offset_s == pytest.approx(truth, abs=0.006), name
    assert st["silent"].offset_status == st["noaudio"].offset_status == "unaligned"
    assert ep.common_start_s == pytest.approx(2.37, abs=0.006) and ep.common_end_s == pytest.approx(-2.37 + 12, abs=0.01)
    assert ep.n_frames() == int(np.floor((ep.common_end_s - ep.common_start_s) * ep.proc_fps + 1e-9))

    ep2 = Episode.create(tmp_path / "eps", "noref", [("ref", "exo", m / "noaudio.mp4", None), ("b", "exo", m / "late.mp4", None)], reference="ref")
    probe(ep2)
    with pytest.raises(RuntimeError, match="no usable audio"):
        A.align(ep2)
