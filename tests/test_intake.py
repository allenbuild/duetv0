"""scripts/intake.py: session splitting and cross-camera matching on synthetic chunk lists, clock-drift estimation +
correction round-trip on synthetic audio with a known stretch, and the SD-dump end-to-end on the CoMind clips
(4 streams cut into 3 chunks each -> one episode, durations within 0.1 s, probe+align offsets within 50 ms)."""
from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import intake as I

from duet.playground.align import _audio_envelope

CLIPS = ROOT / "data/playground/clips"
ORIG = ROOT / "data/playground/episodes/comind_43276420_clip"
T0 = dt.datetime(2026, 9, 20, 10, 0, tzinfo=dt.UTC)


def ch(name: str, start_min: float | None, dur_min: float) -> I.Chunk:
    start = T0 + dt.timedelta(minutes=start_min) if start_min is not None else None
    return I.Chunk(Path(f"/sd/{name}"), dur_min * 60, start, "creation_time" if start else "none")


# ----------------------------------------------------------------------------- sessions
def test_split_sessions_by_gap():
    chunks = [ch("VID_0001.MP4", 0, 10), ch("VID_0002.MP4", 10, 10), ch("VID_0003.MP4", 23, 10),   # 3 min gap: same session
              ch("VID_0004.MP4", 60, 10), ch("VID_0005.MP4", 70, 5)]                                  # 27 min gap: new session
    s = I.split_sessions(chunks, 5 * 60)
    assert [[c.path.name for c in sess] for sess in s] == [["VID_0001.MP4", "VID_0002.MP4", "VID_0003.MP4"], ["VID_0004.MP4", "VID_0005.MP4"]]


def test_split_sessions_without_timestamps_is_one_session():
    chunks = [ch("a", None, 10), ch("b", None, 10), ch("c", None, 10)]
    assert len(I.split_sessions(chunks, 300)) == 1


def test_natural_order_and_parse_time():
    names = [Path("VID_0010.MP4"), Path("VID_0002.MP4"), Path("VID_0001.MP4")]
    assert [p.name for p in sorted(names, key=I.natural_key)] == ["VID_0001.MP4", "VID_0002.MP4", "VID_0010.MP4"]
    assert I.parse_time("2026-09-20T10:15:30.000000Z") == dt.datetime(2026, 9, 20, 10, 15, 30, tzinfo=dt.UTC)
    assert I.parse_time("garbage") is None


def test_parse_map():
    assert I.parse_map("cam_leader=ego:Idhant") == ("cam_leader", ("ego", "Idhant"))
    assert I.parse_map("exoA=exo") == ("exoA", ("exo", None))
    with pytest.raises(SystemExit):
        I.parse_map("exoA=fixed")


def _cams(ref_sessions, other_sessions, other="cam_b"):
    cams = {"cam_a": I.Camera("cam_a", "ego", "A", sessions=ref_sessions), other: I.Camera(other, "ego", "B", sessions=other_sessions)}
    return cams


def test_match_sessions_by_overlap_and_omit_unmatched():
    ref = [[ch("r1", 0, 10), ch("r2", 10, 10)], [ch("r3", 60, 10)]]
    other = [[ch("o1", 62, 5)], [ch("o2", 1, 8)], [ch("o3", 200, 5)]]  # out of order on purpose; o3 overlaps nothing
    plans = I.match_sessions(_cams(ref, other), "cam_a", "ep")
    assert [p.name for p in plans] == ["ep_1", "ep_2"]
    assert [c.path.name for c in plans[0].cams["cam_b"]] == ["o2"] and [c.path.name for c in plans[1].cams["cam_b"]] == ["o1"]
    assert not any("omitted" in w for p in plans for w in p.warnings)


def test_match_sessions_falls_back_to_index_without_timestamps():
    ref = [[ch("r1", 0, 10)], [ch("r2", 60, 10)]]
    other = [[ch("o1", None, 10)], [ch("o2", None, 10)]]
    plans = I.match_sessions(_cams(ref, other), "cam_a", "ep")
    assert plans[0].cams["cam_b"][0].path.name == "o1" and plans[1].cams["cam_b"][0].path.name == "o2"
    assert any("matched by index" in w for w in plans[0].warnings)


def test_match_sessions_omits_camera_with_no_overlap():
    ref = [[ch("r1", 0, 10)], [ch("r2", 60, 10)]]
    other = [[ch("o1", 300, 10)]]
    plans = I.match_sessions(_cams(ref, other), "cam_a", "ep")
    assert "cam_b" not in plans[0].cams and any("omitted" in w for w in plans[0].warnings)


# ----------------------------------------------------------------------------- drift
def _clicky_audio(sr: int, dur_s: float, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed); x = rng.normal(0, 0.005, int(dur_s * sr)).astype(np.float32); t = 0.5
    while t < dur_s - 0.1:
        n0 = int(t * sr); n = int(0.03 * sr)
        x[n0:n0 + n] += (rng.normal(0, 0.5, n) * np.exp(-np.arange(n) / (0.01 * sr))).astype(np.float32)
        t += rng.uniform(0.3, 1.2)
    return x


def _write_wav(path: Path, x: np.ndarray, sr: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _mux(wav: Path, out: Path) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-i", str(wav), "-shortest",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", str(out)], check=True)


def test_drift_estimate_and_correction_round_trip(tmp_path):
    sr, dur, s, lead = 16000, 240.0, 1.0003, 2.0  # sig clock 300 ppm slow (events at t*s, ~54 ms drift between the windows), starts 2 s late
    x = _clicky_audio(sr, dur)
    n_out = int(len(x) * s); y = np.interp(np.arange(n_out) / s, np.arange(len(x)), x).astype(np.float32)
    y = np.concatenate([np.zeros(int(lead * sr), np.float32), y])
    _write_wav(tmp_path / "ref.wav", x, sr); _write_wav(tmp_path / "sig.wav", y, sr)
    _mux(tmp_path / "ref.wav", tmp_path / "ref.mp4"); _mux(tmp_path / "sig.wav", tmp_path / "sig.mp4")
    ref_env, sig_env = _audio_envelope(tmp_path / "ref.mp4"), _audio_envelope(tmp_path / "sig.mp4")
    r = I.estimate_drift(ref_env, sig_env)
    assert r["ok"], r
    assert r["lag_global_s"] == pytest.approx(-lead, abs=0.05)  # sig starts later -> t_ref = t_sig - 2
    assert r["offset_end_s"] - r["offset_start_s"] < -0.02  # sig time runs ahead of ref time: offset shrinks
    assert r["k"] == pytest.approx(1 / s, abs=1.5e-4)
    # too-short overlap is refused
    assert not I.estimate_drift(ref_env[:10000], sig_env[:10000])["ok"]
    # applying k removes the drift
    I.apply_drift(tmp_path / "sig.mp4", tmp_path / "fixed.mp4", r["k"])
    r2 = I.estimate_drift(ref_env, _audio_envelope(tmp_path / "fixed.mp4"))
    assert r2["ok"] and abs(r2["k"] - 1.0) < 2e-4 and abs(r2["offset_end_s"] - r2["offset_start_s"]) <= 0.03  # 10 ms hop quantisation


def test_drift_correct_notes_when_overlap_too_short(tmp_path):
    sr = 16000; x = _clicky_audio(sr, 40.0)
    _write_wav(tmp_path / "a.wav", x, sr); _mux(tmp_path / "a.wav", tmp_path / "a.mp4"); _mux(tmp_path / "a.wav", tmp_path / "b.mp4")
    files, notes = I.drift_correct({"a": tmp_path / "a.mp4", "b": tmp_path / "b.mp4"}, "a", tmp_path, 20.0, 1.2, 60.0, 120.0, log=lambda *_: None)
    assert files["b"] == tmp_path / "b.mp4" and len(notes) == 1 and "not estimated" in notes[0] and "+0.00s" in notes[0]


# ----------------------------------------------------------------------------- end to end on the CoMind clips
@pytest.mark.skipif(not (CLIPS / "comind_leader.mp4").exists() or not (ORIG / "episode.json").exists(), reason="comind clips not present")
def test_sd_dump_intake_reproduces_original_episode(tmp_path, capsys):
    sd = tmp_path / "sd"; cams = {"cam_leader": "leader", "cam_helper": "helper", "exoA": "gopro_front", "exoB": "gopro_back"}
    orig_dur = {}
    for cam, src in cams.items():
        (sd / cam).mkdir(parents=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(CLIPS / f"comind_{src}.mp4"), "-map", "0:v", "-map", "0:a", "-c", "copy",
                        "-f", "segment", "-segment_time", "17", "-reset_timestamps", "1", str(sd / cam / "VID_%04d.MP4")], check=True)
        assert len(list((sd / cam).glob("*.MP4"))) == 3
        orig_dur[cam] = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(CLIPS / f"comind_{src}.mp4")],
                                             capture_output=True, text=True, check=True).stdout)
    args = [str(sd), "--map", "cam_leader=ego:Idhant", "--map", "cam_helper=ego:Allen", "--map", "exoA=exo", "--map", "exoB=exo", "--ref", "cam_leader",
            "--prefix", "tmp_intake", "--episodes", str(tmp_path / "eps"), "--scratch", str(tmp_path / "scratch")]
    I.main(args + ["--dry-run"])
    assert "dry run" in capsys.readouterr().out and not (tmp_path / "eps").exists()
    I.main(args + ["--run", "--stages", "probe,align"])
    eps = sorted((tmp_path / "eps").iterdir())
    assert [e.name for e in eps] == ["tmp_intake_1"]
    ep = json.load(open(eps[0] / "episode.json"))
    assert len(ep["streams"]) == 4 and ep["reference"] == "cam_leader"
    by = {s["name"]: s for s in ep["streams"]}
    assert by["cam_leader"]["role"] == "ego" and by["cam_leader"]["person"] == "Idhant" and by["exoA"]["role"] == "exo"
    for cam in cams:
        assert (eps[0] / by[cam]["path"]).is_file() and not (eps[0] / by[cam]["path"]).is_symlink()
        assert abs(by[cam]["duration_s"] - orig_dur[cam]) < 0.1, (cam, by[cam]["duration_s"], orig_dur[cam])
    orig = {s["name"]: s["offset_s"] for s in json.load(open(ORIG / "episode.json"))["streams"]}
    assert ep["status"]["align"]["state"] == "done"
    for cam, src in cams.items():
        assert abs(by[cam]["offset_s"] - orig[src]) < 0.05, (cam, by[cam]["offset_s"], orig[src])
    assert any("concatenated" in n for n in ep["notes"]) and any("drift not estimated" in n for n in ep["notes"])
