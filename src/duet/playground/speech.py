"""Speech transcription (faster-whisper) + speaker attribution across ego mics.

Every camera hears everyone in the room, so a naive per-stream transcript repeats each utterance
once per microphone. This stage:

1. extracts 16 kHz mono audio for the COMMON window from every stream that has audio
   (reference-time window converted to stream time with ``offset_s``; the wav therefore starts at
   ``common_start_s`` in reference time for every stream);
2. transcribes each wav with faster-whisper (CPU, int8, word timestamps, language auto-detected)
   and writes ``derived/speech/<stream>.json`` with segment and word times in REFERENCE time;
3. attributes every segment to a speaker: the person whose EGO mic is loudest in the segment window
   *relative to that mic's own median level* (a wearer's own voice dominates their own mic; the
   median normalises away per-mic gain). With no ego mics the speaker is ``"unknown"``;
4. de-duplicates across streams (time overlap + text similarity), keeping the copy heard by the
   speaker's own mic (else the highest word-probability copy) -> ``derived/speech/transcript.json``;
5. writes ``derived/speech/records_extra.parquet`` with per-frame columns on the common timeline:
   ``speaking_<person>`` (bool), ``words_last_3s_<person>`` (int) and ``any_speech`` (bool).
   ``export`` merges them as ``speech_*`` columns.

Model files live in ``data/playground/models`` (``DUET_WHISPER_MODEL`` overrides the size, default
``small``). Streams without audio are ignored; the stage is skipped when nothing has audio.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from .episode import Episode, Stream
from .perception import _frame_list

SR = 16000
MODELS = Path(__file__).resolve().parents[3] / "data/playground/models"
MODEL_NAME = os.environ.get("DUET_WHISPER_MODEL", "small")
WORDS_WINDOW_S = 3.0
DEDUP_OVERLAP = 0.5   # intersection / shorter duration
DEDUP_TEXT = 0.6      # matched fraction of the shorter text
DEDUP_SAME_SPK_OVERLAP = 0.8  # same speaker + this much overlap + single-utterance length -> duplicate even if the words differ
MAX_UTTERANCE_S = 8.0
LEVEL_HOP_S = 0.1     # frame length for the per-mic median level
PEAK_NORM = 0.9       # peak level fed to whisper


# ----------------------------------------------------------------------------- audio helpers
def extract_wav(src: Path, dst: Path, start_s: float, dur_s: float) -> bool:
    """16 kHz mono pcm16 wav of ``src`` from stream time ``start_s`` for ``dur_s`` seconds."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", f"{max(0.0, start_s):.3f}", "-t", f"{dur_s:.3f}", "-i", str(src),
                        "-vn", "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", str(dst)], capture_output=True, check=False)
    return r.returncode == 0 and dst.exists() and dst.stat().st_size > 44


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate(); n = w.getnframes(); sw = w.getsampwidth(); ch = w.getnchannels()
        raw = w.readframes(n)
    dt = {1: np.int8, 2: np.int16, 4: np.int32}[sw]
    x = np.frombuffer(raw, dt).astype(np.float32) / float(2 ** (8 * sw - 1))
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x, sr


def rms_db(x: np.ndarray, sr: int, t0: float, t1: float) -> float:
    a, b = max(0, int(t0 * sr)), min(len(x), int(t1 * sr))
    if b - a < 1:
        return -120.0
    return float(10 * np.log10(np.mean(x[a:b] ** 2) + 1e-12))


def median_level_db(x: np.ndarray, sr: int, hop_s: float = LEVEL_HOP_S) -> float:
    hop = max(1, int(hop_s * sr)); n = len(x) // hop
    if n < 1:
        return rms_db(x, sr, 0, len(x) / sr)
    e = (x[: n * hop].reshape(n, hop) ** 2).mean(1)
    return float(np.median(10 * np.log10(e + 1e-12)))


# ----------------------------------------------------------------------------- attribution + de-dup (pure)
def relative_energies(t0: float, t1: float, mics: dict[str, tuple[np.ndarray, int, float]]) -> dict[str, float]:
    """dB above each mic's own median level in [t0, t1]. mics: person -> (samples, sr, median_db); times are wav-relative."""
    return {p: rms_db(x, sr, t0, t1) - med for p, (x, sr, med) in mics.items()}


def pick_speaker(rel_db: dict[str, float], min_margin_db: float = 0.0) -> tuple[str, float]:
    """Speaker = mic with the highest relative energy. Returns (person, margin over the runner-up in dB)."""
    if not rel_db:
        return "unknown", 0.0
    order = sorted(rel_db.items(), key=lambda kv: -kv[1])
    margin = order[0][1] - (order[1][1] if len(order) > 1 else -120.0)
    return (order[0][0] if margin >= min_margin_db else "unknown"), float(margin)


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def text_similarity(a: str, b: str) -> float:
    """Fraction of the SHORTER text (in words) matched by the longer one, so a fragment of a longer segment counts as similar."""
    wa, wb = _norm_words(a), _norm_words(b)
    if not wa or not wb:
        return 0.0
    if len(wa) > len(wb):
        wa, wb = wb, wa
    m = difflib.SequenceMatcher(None, wa, wb, autojunk=False)
    return sum(bl.size for bl in m.get_matching_blocks()) / len(wa)


def time_overlap(a: dict, b: dict) -> float:
    inter = min(a["end"], b["end"]) - max(a["start"], b["start"])
    shorter = max(1e-3, min(a["end"] - a["start"], b["end"] - b["start"]))
    return max(0.0, inter) / shorter


def is_duplicate(a: dict, b: dict, overlap_thr: float = DEDUP_OVERLAP, text_thr: float = DEDUP_TEXT) -> bool:
    """Two rules. (1) Overlap >= 50 % of the shorter AND >= 60 % of the shorter text matched (catches a run-on
    segment that swallowed several utterances). (2) Same attributed speaker, overlap >= 80 % of the shorter and
    both segments single-utterance length (<= 8 s): the same mouth cannot say two different things at the same
    instant, so differing words are two mics' transcriptions (or a fragment) of one utterance. Different speakers
    at the same time are kept (cross-talk); long run-ons only match through rule 1."""
    ov = time_overlap(a, b)
    if ov >= overlap_thr and text_similarity(a["text"], b["text"]) >= text_thr:
        return True
    longest = max(a["end"] - a["start"], b["end"] - b["start"])
    return ov >= DEDUP_SAME_SPK_OVERLAP and longest <= MAX_UTTERANCE_S and a.get("speaker", "unknown") == b.get("speaker", "unknown")


def _preference(seg: dict, own_mic: dict[str, str]) -> tuple:
    """Higher is better: heard by the speaker's own ego mic, then mean word probability, then longer."""
    own = own_mic.get(seg.get("speaker", "unknown")) == seg.get("source")
    return (1 if own else 0, seg.get("prob", 0.0), seg["end"] - seg["start"])


def dedup_segments(segments: list[dict], own_mic: dict[str, str] | None = None) -> tuple[list[dict], list[dict]]:
    """Keep one copy of every utterance. ``own_mic`` maps person -> stream name of their ego mic.

    Segments are visited best-first (see ``_preference``) and kept unless they duplicate a kept one
    (time overlap >= 50 % of the shorter AND >= 60 % of the shorter text matched). Returns (kept sorted by
    start, dropped with a ``duplicate_of`` index into kept)."""
    own_mic = own_mic or {}
    order = sorted(range(len(segments)), key=lambda i: _preference(segments[i], own_mic), reverse=True)
    kept: list[dict] = []; dropped: list[dict] = []
    for i in order:
        s = segments[i]
        dup = next((k for k, t in enumerate(kept) if is_duplicate(s, t)), None)
        if dup is None:
            kept.append(s)
        else:
            dropped.append({**s, "duplicate_of": dup})
    idx = sorted(range(len(kept)), key=lambda k: kept[k]["start"])
    remap = {old: new for new, old in enumerate(idx)}
    for d in dropped:
        d["duplicate_of"] = remap[d["duplicate_of"]]
    return [kept[k] for k in idx], dropped


def per_frame_columns(segments: list[dict], persons: list[str], t: np.ndarray, window_s: float = WORDS_WINDOW_S) -> pd.DataFrame:
    """speaking_<p> (segment covers t), words_last_3s_<p> (words of p ending in (t-3, t]), any_speech (any speaker incl. unknown)."""
    n = len(t); cols: dict[str, np.ndarray] = {}
    labels = list(persons) + [p for p in sorted({s.get("speaker", "unknown") for s in segments}) if p not in persons]
    any_speech = np.zeros(n, bool)
    for p in labels:
        speaking = np.zeros(n, bool); words = np.zeros(n, np.int32)
        for s in segments:
            if s.get("speaker", "unknown") != p:
                continue
            speaking |= (t >= s["start"]) & (t <= s["end"])
            ends = np.array([w["end"] for w in s.get("words", [])] or [s["end"]])
            words += ((ends[None, :] > t[:, None] - window_s) & (ends[None, :] <= t[:, None])).sum(1).astype(np.int32)
        cols[f"speaking_{p}"] = speaking; cols[f"words_last_3s_{p}"] = words; any_speech |= speaking
    cols["any_speech"] = any_speech
    return pd.DataFrame(cols)


# ----------------------------------------------------------------------------- whisper
def load_model(name: str = MODEL_NAME):
    from faster_whisper import WhisperModel
    MODELS.mkdir(parents=True, exist_ok=True)
    return WhisperModel(name, device="cpu", compute_type="int8", download_root=str(MODELS))


def transcribe_wav(model, wav: Path, t_offset: float = 0.0, language: str | None = None) -> tuple[list[dict], dict]:
    """Segments with words; all times shifted by ``t_offset`` (wav time -> reference time).

    The audio is peak-normalised to 0.9 FS before decoding: a quiet ego mic (-42 dB RMS on the Aria
    clip) otherwise decodes into lower-case run-on segments with no punctuation."""
    x, _ = read_wav(wav)
    x = np.clip(x / (np.abs(x).max() + 1e-9) * PEAK_NORM, -1.0, 1.0).astype(np.float32)
    segs, info = model.transcribe(x, language=language, word_timestamps=True, vad_filter=True, beam_size=5)
    out = []
    for s in segs:
        words = [{"word": w.word.strip(), "start": round(w.start + t_offset, 3), "end": round(w.end + t_offset, 3), "prob": round(float(w.probability), 3)}
                 for w in (s.words or [])]
        text = s.text.strip()
        if not text:
            continue
        out.append({"start": round(s.start + t_offset, 3), "end": round(s.end + t_offset, 3), "text": text, "words": words,
                    "prob": round(float(np.mean([w["prob"] for w in words])) if words else float(np.exp(s.avg_logprob)), 3),
                    "no_speech_prob": round(float(s.no_speech_prob), 3)})
    meta = {"language": info.language, "language_probability": round(float(info.language_probability), 3), "duration_s": round(float(info.duration), 3)}
    return out, meta


# ----------------------------------------------------------------------------- stage
def _n_frames(ep: Episode) -> int:
    counts = [len(_frame_list(ep, s)) for s in ep.streams]
    if counts and min(counts) > 0:
        return min(counts)
    return int((ep.common_end_s - ep.common_start_s) * ep.proc_fps)


def speech(ep: Episode) -> None:
    ep.set_status("speech", "running")
    audio_streams = [s for s in ep.streams if s.has_audio]
    if not audio_streams:
        ep.set_status("speech", "skipped", "no stream has audio"); return
    if ep.common_end_s <= ep.common_start_s:
        ep.set_status("speech", "skipped", "empty common window (run probe/align first)"); return
    out = ep.derived / "speech"; out.mkdir(parents=True, exist_ok=True)
    t0, dur = ep.common_start_s, ep.common_end_s - ep.common_start_s

    # 1. audio for the common window from every stream with audio
    wavs: dict[str, Path] = {}
    for s in audio_streams:
        wav = out / f"{s.name}.wav"
        if extract_wav(ep.dir / s.path, wav, t0 - s.offset_s, dur):
            wavs[s.name] = wav
        else:
            ep.notes.append(f"speech: {s.name}: audio extraction failed")
    if not wavs:
        ep.set_status("speech", "failed", "audio extraction failed for every stream"); return

    # 2. ego mic levels for attribution
    egos: list[Stream] = [s for s in ep.egos() if s.name in wavs]
    person_of = {s.name: (s.person or s.name) for s in egos}
    own_mic = {p: n for n, p in person_of.items()}
    mics = {}
    for s in egos:
        x, sr = read_wav(wavs[s.name]); mics[person_of[s.name]] = (x, sr, median_level_db(x, sr))

    # 3. transcribe every stream
    model = load_model()
    all_segments: list[dict] = []; langs = {}
    for name, wav in wavs.items():
        segs, meta = transcribe_wav(model, wav, t_offset=t0)
        langs[name] = meta["language"]
        for g in segs:
            rel = relative_energies(g["start"] - t0, g["end"] - t0, mics)
            spk, margin = pick_speaker(rel)
            g.update(source=name, speaker=spk, speaker_margin_db=round(margin, 2), rel_db={p: round(v, 2) for p, v in rel.items()})
        st = ep.stream(name)
        json.dump({"stream": name, "person": st.person, "role": st.role, "offset_s": st.offset_s, "model": MODEL_NAME, **meta,
                   "time_base": "reference seconds (t_stream = t_ref - offset_s)", "segments": segs}, open(out / f"{name}.json", "w"), indent=1)
        all_segments += segs

    # 4. one copy per utterance
    kept, dropped = dedup_segments(all_segments, own_mic)
    persons = [person_of[s.name] for s in egos]
    json.dump({"episode": ep.name, "model": MODEL_NAME, "languages": langs, "persons": persons, "streams_transcribed": list(wavs),
               "attribution": "speaker = ego mic with highest RMS in the segment window relative to that mic's median level" if egos else "no ego mic: unknown",
               "n_raw": len(all_segments), "n_kept": len(kept), "n_dropped": len(dropped),
               "segments": kept, "dropped": dropped}, open(out / "transcript.json", "w"), indent=1)

    # 5. per-frame columns on the common timeline
    n = _n_frames(ep); t = t0 + np.arange(n) / ep.proc_fps
    df = per_frame_columns(kept, persons, t)
    df.to_parquet(out / "records_extra.parquet", index=False)
    words = sum(len(g["words"]) for g in kept)
    by = {p: sum(1 for g in kept if g["speaker"] == p) for p in persons + ["unknown"]}
    ep.set_status("speech", "done", f"whisper-{MODEL_NAME} on {len(wavs)} streams ({','.join(sorted(set(langs.values())))}): {len(kept)} utterances / {words} words "
                  f"(dropped {len(dropped)} duplicates); by speaker " + ", ".join(f"{p} {c}" for p, c in by.items() if c or p != "unknown"))
