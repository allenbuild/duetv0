# `scripts/intake.py`: SD-card dumps -> playground episodes (2026-09-26)

Body cams and action cams split recordings into chunks every few minutes (VID_0001.MP4, VID_0002.MP4, ...),
one folder per card. `intake.py` turns a root folder with one such subfolder per camera into one playground
episode per recording session, with the streams concatenated, clock drift corrected and the pipeline
optionally started.

```sh
.venv/bin/python scripts/intake.py /Volumes/dump --map cam_leader=ego:Idhant --map cam_helper=ego:Allen \
    --map exoA=exo --map exoB=exo --ref cam_leader [--prefix kitchen_0926] [--gap-min 5] [--dry-run] [--run] \
    [--stages probe,align] [--episodes DIR] [--scratch DIR] [--no-drift] [--drift-ms 20] [--drift-win 60] [--drift-min-span 120]
```

`--map cam=ego:Person` or `cam=exo[:label]`; the camera folder name becomes the stream name. `--ref` is the
time reference. Episodes are named `<prefix>_<session index>` (1-based; prefix defaults to the root folder
name) under `data/playground/episodes`; an existing episode name is skipped.

## Steps

1. **Probe.** ffprobe every video file (`.mp4 .mov .mkv .avi .mts .m4v`): duration and `creation_time`
   (format tag, else the first stream tag). Without a tag the start is `mtime - duration`, flagged `[mtime]`
   in the plan (only as good as the card's clock and the copy). Unreadable files (truncated chunk, no moov
   atom) are skipped with a warning that ends up in the episode notes.
2. **Order.** Chunks are ordered by natural name sort (VID_0002 < VID_0010).
3. **Sessions.** Per camera, a new session starts when `next.start - prev.end > --gap-min` (5 min). The
   reference camera's sessions define the episodes; every other camera's sessions are matched by maximum
   time overlap. If nothing overlaps (no usable timestamps) and the session counts agree they are matched by
   index with a warning; a camera with no matching session is omitted from that episode with a warning.
4. **Concat.** Per session and camera: ffmpeg concat demuxer, `-c copy`, video + audio streams only (GoPro
   `tmcd`/GPMF data streams break the mp4 muxer on copy), into the scratch folder (a temp dir deleted
   afterwards, or `--scratch DIR`).
5. **Clock drift** (`estimate_drift`, reuses `align._audio_envelope` + `align._xcorr_lag`, 10 ms hops).
   Global lag between the stream and the reference from the first `max_lag + 2 windows` seconds
   (a drifting clock smears the peak over a whole hour; the head is enough). Then the local offset on a
   60 s window at the start of the overlap and on one at the end (local search +-2 s around the global lag).
   With `t_ref = t_stream + offset`, the stream clock rate is `k = 1 + (offset_end - offset_start) / span`
   where `span` is the stream time between the two windows. If `|offset_end - offset_start| > --drift-ms`
   (20 ms) and both window confidences are >= 1.2, the stream is re-timed with
   `ffmpeg -filter:v setpts=PTS*k -af atempo=1/k` (libx264 crf 18 veryfast + aac 160k, i.e. re-encoded) before
   the episode is created, and the factor is written to the episode notes. Needs `--drift-min-span` (120 s)
   between the two windows, so nothing shorter than ~4 min is ever corrected; the note then says why. The
   `align` stage afterwards measures the (now constant) offset as usual.
6. **Episode.** `Episode.create(..., link=False)` copies the concatenated (or re-timed) files into
   `streams/`; notes record chunks per camera, warnings and the drift decision per stream. `--run` calls
   `run_stages` (all stages, or `--stages`).

`--dry-run` prints steps 1-3 (chunks with durations, timestamps and their source; sessions; episodes with
per-camera chunk counts and warnings) and writes nothing.

## Verification

`tests/test_intake.py`:

- session splitting (3 min gap joins, 27 min gap splits; no timestamps -> one session), natural ordering,
  time parsing, `--map` parsing, session matching by overlap / by index / omission;
- **drift round-trip on synthetic audio**: 240 s of random clicks, a copy resampled 300 ppm slow (events at
  `t * 1.0003`) with 2 s of leading silence. `estimate_drift` finds the global lag (-2.01 s, true -2.00),
  offsets -2.01 s at the start and -2.06 s at the end 180 s apart, `k = 0.999722` (true 0.999700).
  `apply_drift` with that k, re-estimated: `k = 1.000000`, start and end offsets both -1.98 s (residual 0 ms).
  Overlap shorter than two windows is refused with a note;
- **SD-dump end to end on the CoMind clips**: the 4 streams of `comind_43276420_clip` cut into 3 chunks each
  (`ffmpeg -c copy -f segment -segment_time 17 -reset_timestamps 1`, video+audio only), laid out as
  `cam_leader/ cam_helper/ exoA/ exoB/`, imported with the map above into a temp episodes dir with
  `--run --stages probe,align`. Result: one episode `tmp_intake_1`, 4 copied streams, durations 50.02 s
  (Aria) and 50.07 s (GoPro) vs 50.00 s originals (< 0.1 s; the extra comes from the AAC priming/edit list on
  concat), offsets `cam_helper -1.73 s (x6.1), exoA +0.90 s (x7.4), exoB +1.67 s (x6.9)` = the original
  episode's -1.73 / +0.90 / +1.67 exactly. Drift was not estimated (48 s overlap < 120 s), noted as such.
  The temp episode lives in pytest's tmp dir and is deleted with it.

## Limits

- Drift is modelled as linear (one rate per stream per session). Two windows only; a mid-session
  discontinuity (camera dropped frames) is not detected. Sub-hop precision is 10 ms, so the smallest
  detectable rate is ~20 ms / span.
- Drift correction re-encodes the whole stream (libx264 crf 18): slow for hours of 4K, and it changes the
  video bitstream. Frame duplication/drop at the output frame rate is left to ffmpeg's default vsync.
- Session detection relies on timestamps. Cameras whose chunks all carry the same `creation_time` (the
  recording's start) produce negative gaps and are treated as one session, which is right; cameras with no
  usable clock at all fall back to `mtime`, which is wrong after a plain copy that resets mtimes.
- Not verified on a real multi-hour dump with real drift, only on the synthetic 300 ppm stretch and the
  50 s CoMind clip. A concat across a real chunk boundary with a codec parameter change (resolution switch)
  would fail in ffmpeg; the error propagates.
