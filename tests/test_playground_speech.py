"""Speech stage: attribution + de-duplication on synthetic energies/segments (no whisper), per-frame columns,
and one slow end-to-end run guarded by DUET_SLOW_TESTS=1 (whisper on ffmpeg-generated speech-free audio, plus
5 s of the CoMind clip when it is present)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from duet.playground import speech as S

ROOT = Path(__file__).resolve().parents[1]
EPISODES = ROOT / "data/playground/episodes"


# ----------------------------------------------------------------------------- attribution
def _mic(sr: int, dur_s: float, floor: float, bursts: list[tuple[float, float, float]], seed: int) -> np.ndarray:
    """White-noise floor at ``floor`` amplitude plus louder bursts (t0, t1, amplitude)."""
    rng = np.random.default_rng(seed); x = rng.normal(0, floor, int(dur_s * sr)).astype(np.float32)
    for t0, t1, amp in bursts:
        x[int(t0 * sr):int(t1 * sr)] += rng.normal(0, amp, int(t1 * sr) - int(t0 * sr)).astype(np.float32)
    return x


def test_pick_speaker_argmax_and_margin():
    spk, margin = S.pick_speaker({"alice": 3.0, "bob": 9.5})
    assert spk == "bob" and margin == pytest.approx(6.5)
    assert S.pick_speaker({}) == ("unknown", 0.0)
    assert S.pick_speaker({"alice": 2.0, "bob": 2.5}, min_margin_db=1.0)[0] == "unknown"


def test_wearer_is_loudest_relative_to_own_median():
    sr = 8000
    # alice speaks 1-2 s, bob speaks 3-4 s. Bob's mic has 10x the gain (louder floor AND louder everything),
    # so raw RMS would attribute both utterances to bob; relative to each mic's own median the wearer wins.
    a = _mic(sr, 6, 0.01, [(1, 2, 0.20), (3, 4, 0.04)], 1)
    b = _mic(sr, 6, 0.10, [(1, 2, 0.30), (3, 4, 2.00)], 2)
    mics = {"alice": (a, sr, S.median_level_db(a, sr)), "bob": (b, sr, S.median_level_db(b, sr))}
    assert S.pick_speaker(S.relative_energies(1.0, 2.0, mics))[0] == "alice"
    assert S.pick_speaker(S.relative_energies(3.0, 4.0, mics))[0] == "bob"
    raw = {p: S.rms_db(x, sr, 1.0, 2.0) for p, (x, sr, _) in mics.items()}
    assert S.pick_speaker(raw)[0] == "bob"  # the failure mode the median normalisation avoids


# ----------------------------------------------------------------------------- de-dup
def seg(start, end, text, speaker, source, prob=0.8):
    return {"start": start, "end": end, "text": text, "speaker": speaker, "source": source, "prob": prob, "words": []}


OWN = {"alice": "cam_a", "bob": "cam_b"}


def test_text_similarity_is_relative_to_the_shorter_text():
    assert S.text_similarity("we can mix it there", "Yeah, we can mix it there.") == pytest.approx(1.0)
    assert S.text_similarity("Is that enough water?", "Okay, but no more carrot?") < 0.6
    assert S.text_similarity("", "anything") == 0.0


def test_identical_utterance_heard_by_three_mics_kept_once_from_own_mic():
    segs = [seg(1.0, 2.5, "Do you want to add it?", "alice", "exo", 0.95),
            seg(1.05, 2.4, "Do you want to add it?", "alice", "cam_b", 0.90),
            seg(1.1, 2.6, "Do you want to add it", "alice", "cam_a", 0.70)]
    kept, dropped = S.dedup_segments(segs, OWN)
    assert len(kept) == 1 and len(dropped) == 2
    assert kept[0]["source"] == "cam_a"  # own mic beats higher probability elsewhere
    assert {d["duplicate_of"] for d in dropped} == {0}


def test_same_speaker_same_time_different_words_is_one_utterance():
    segs = [seg(6.1, 7.2, "Is that enough water?", "bob", "exo", 0.3),
            seg(6.4, 7.3, "Okay, but no more carrot?", "bob", "cam_b", 0.6),
            seg(6.2, 7.3, "You don't want any more hair?", "bob", "cam_a", 0.6)]
    kept, dropped = S.dedup_segments(segs, OWN)
    assert [k["text"] for k in kept] == ["Okay, but no more carrot?"] and len(dropped) == 2


def test_cross_talk_by_different_speakers_is_kept():
    segs = [seg(10.0, 11.0, "We can mix it there.", "alice", "cam_a"), seg(10.1, 11.1, "It might be tricky.", "bob", "cam_b")]
    kept, dropped = S.dedup_segments(segs, OWN)
    assert len(kept) == 2 and not dropped


def test_fragment_inside_same_speaker_utterance_is_dropped():
    segs = [seg(17.6, 20.0, "Wait, I think it's still a bit floury.", "bob", "cam_b", 0.8), seg(19.1, 19.8, "Still with flour.", "bob", "exo", 0.5)]
    kept, _ = S.dedup_segments(segs, OWN)
    assert [k["text"] for k in kept] == ["Wait, I think it's still a bit floury."]


def test_run_on_segment_that_swallowed_several_utterances_is_a_duplicate_by_text():
    segs = [seg(1.0, 15.0, "to add it because its too just there no with the carrot its fine do you want to add it", "bob", "cam_a", 0.2),
            seg(1.0, 4.9, "to add it, because it's too... just... there.", "alice", "cam_b", 0.7),
            seg(8.0, 9.4, "No, with the carrot it's fine.", "alice", "cam_b", 0.8),
            seg(14.3, 15.6, "Do you want to add it?", "alice", "cam_b", 0.9)]
    kept, dropped = S.dedup_segments(segs, OWN)
    assert [k["start"] for k in kept] == [1.0, 8.0, 14.3]
    assert dropped[0]["start"] == 1.0 and dropped[0]["end"] == 15.0


def test_two_long_run_ons_with_different_words_are_not_merged_by_the_time_rule():
    segs = [seg(0.0, 12.0, "alpha beta gamma delta", "bob", "cam_a"), seg(0.5, 12.0, "one two three four five", "bob", "exo")]
    kept, _ = S.dedup_segments(segs, OWN)
    assert len(kept) == 2


def test_dedup_output_sorted_and_duplicate_index_remapped():
    segs = [seg(30.0, 31.0, "later thing", "alice", "cam_a"), seg(5.0, 6.0, "early thing", "bob", "cam_b"), seg(5.1, 6.1, "early thing", "bob", "exo")]
    kept, dropped = S.dedup_segments(segs, OWN)
    assert [k["start"] for k in kept] == [5.0, 30.0]
    assert dropped[0]["duplicate_of"] == 0 and kept[dropped[0]["duplicate_of"]]["text"] == "early thing"


# ----------------------------------------------------------------------------- per-frame columns
def test_per_frame_columns():
    t = np.arange(0, 10, 0.5)  # 20 frames at 2 fps
    segs = [{"start": 1.0, "end": 2.0, "speaker": "alice", "words": [{"end": 1.2}, {"end": 1.9}]},
            {"start": 6.0, "end": 6.4, "speaker": "bob", "words": [{"end": 6.3}]},
            {"start": 8.0, "end": 8.6, "speaker": "unknown", "words": []}]
    df = S.per_frame_columns(segs, ["alice", "bob"], t)
    assert list(df.columns) == ["speaking_alice", "words_last_3s_alice", "speaking_bob", "words_last_3s_bob", "speaking_unknown", "words_last_3s_unknown", "any_speech"]
    assert df.speaking_alice.tolist() == [(1.0 <= x <= 2.0) for x in t]
    assert df.words_last_3s_alice.tolist() == [int(1.2 > x - 3 and 1.2 <= x) + int(1.9 > x - 3 and 1.9 <= x) for x in t]
    assert df.words_last_3s_alice[t == 4.5].item() == 1 and df.words_last_3s_alice[t == 5.0].item() == 0
    assert df.speaking_bob[t == 6.0].item() and not df.speaking_bob[t == 6.5].item()
    assert df.words_last_3s_unknown[t == 8.5].item() == 0 and df.words_last_3s_unknown[t == 9.0].item() == 1  # wordless segment = one word at its end (8.6)
    assert df.any_speech.sum() == df.speaking_alice.sum() + df.speaking_bob.sum() + df.speaking_unknown.sum()
    assert df.speaking_alice.dtype == bool and df.words_last_3s_alice.dtype == np.int32


# ----------------------------------------------------------------------------- slow: real whisper plumbing
slow = pytest.mark.skipif(os.environ.get("DUET_SLOW_TESTS") != "1", reason="set DUET_SLOW_TESTS=1 to run whisper")


def _synthetic_stream(path: Path, seed: int, dur_s: float = 6.0) -> None:
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=160x120:rate=10", "-f", "lavfi",
                    "-i", f"anoisesrc=color=pink:amplitude=0.05:seed={seed}", "-t", f"{dur_s}", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-shortest", str(path)], check=True)


@slow
def test_speech_stage_on_speech_free_synthetic_episode(tmp_path):
    from duet.playground.episode import Episode, probe
    name = f"tmp_speech_synth_{os.getpid()}"
    try:
        _synthetic_stream(tmp_path / "a.mp4", 1); _synthetic_stream(tmp_path / "b.mp4", 2)
        ep = Episode.create(EPISODES, name, [("a", "ego", tmp_path / "a.mp4", "alice"), ("b", "ego", tmp_path / "b.mp4", "bob")], link=False)
        probe(ep); ep.common_start_s, ep.common_end_s = 0.0, 6.0; ep.save()
        S.speech(ep)
        assert ep.status["speech"]["state"] == "done", ep.status["speech"]
        for f in ("a.wav", "b.wav", "a.json", "b.json", "transcript.json", "records_extra.parquet"):
            assert (ep.derived / "speech" / f).exists(), f
        df = pd.read_parquet(ep.derived / "speech" / "records_extra.parquet")
        assert len(df) == 60 and set(df.columns) >= {"speaking_alice", "words_last_3s_alice", "speaking_bob", "words_last_3s_bob", "any_speech"}
        # export merges records_extra.parquet as speech_* columns
        from duet.playground.export import export
        from duet.playground.perception import extract_frames
        extract_frames(ep); export(ep)
        rec = pd.read_parquet(ep.derived / "records.parquet")
        assert {"speech_speaking_alice", "speech_words_last_3s_bob", "speech_any_speech"} <= set(rec.columns) and len(rec) == 60
    finally:
        shutil.rmtree(EPISODES / name, ignore_errors=True)


@slow
@pytest.mark.skipif(not (EPISODES / "comind_43276420_clip/streams/leader.mp4").exists(), reason="comind clip not present")
def test_speech_stage_on_5s_of_comind():
    from duet.playground.episode import Episode, probe
    src = EPISODES / "comind_43276420_clip"; name = f"tmp_speech_comind_{os.getpid()}"
    try:
        ep = Episode.create(EPISODES, name, [("leader", "ego", src / "streams/leader.mp4", "leader"), ("helper", "ego", src / "streams/helper.mp4", "helper")], reference="leader")
        probe(ep); ep.stream("helper").offset_s = -1.73; ep.common_start_s, ep.common_end_s = 1.67, 6.67; ep.save()
        S.speech(ep)
        assert ep.status["speech"]["state"] == "done", ep.status["speech"]
        import json
        tr = json.load(open(ep.derived / "speech" / "transcript.json"))
        assert tr["segments"] and all(1.67 <= g["start"] <= 6.67 for g in tr["segments"])
        assert all(g["speaker"] in ("leader", "helper") for g in tr["segments"])
        assert len(pd.read_parquet(ep.derived / "speech" / "records_extra.parquet")) == 50
    finally:
        shutil.rmtree(EPISODES / name, ignore_errors=True)
