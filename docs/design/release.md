# Release packaging: what turns an episode into something that can leave the building

Built 2026-09-26. `src/duet/playground/release.py`, `scripts/release_episode.py`. Tests: `tests/test_playground_release.py`.

```sh
.venv/bin/python scripts/release_episode.py comind_43276420_clip --version v0.1.0            # build (default --out data/playground/releases)
.venv/bin/python scripts/release_episode.py comind_43276420_clip --version v0.1.0 --verify   # re-hash and report
# optional: --example-png PATH --example-frame 300 --example-stream gopro_front   one blurred frame for inspection
#           --no-blur   copies streams unblurred; the manifest and README say NOT ANONYMISED (tests use this)
```

## Layout of `releases/<version>/<episode>/`

| file | what |
|---|---|
| `streams/<stream>.mp4` | every stream re-encoded (libx264 crf 20, yuv420p, faststart) with faces blurred; the original audio track is stream-copied (re-encoded to AAC only if copy fails). Audio is NOT anonymised |
| `derived/records.parquet`, `episode.json` | copied verbatim |
| `derived/speech/transcript.json`, `derived/annotate/annotations.json`, `derived/autolabel/proposals.json`, `derived/contact/events.json`, `derived/metrics/session.json`, `derived/qc/report.json` | copied if present, same relative paths as in the episode |
| `schema.json` | every column of the copied records.parquet: name, dtype, shape, unit, meaning; plus `n_unknown` / `unknown_fraction` |
| `manifest.json` | sha256 + byte size of every file (except itself), `code_git_rev`, stage status snapshot, key package versions + ffmpeg, creation time (UTC), face-blur backend and per-stream statistics, list of copied files |
| `LICENSE` | CC BY-NC 4.0 notice, first line "DRAFT, founders to confirm" |
| `consent.json` | template: one entry per person (`name`, `date`, `consent_recorded: false`, `contact: ""`, `scope: "research dataset"`). Persons = the `person` field of the streams; placeholders `person_<i>` per ego stream when none |
| `README.md` | contents, streams and offsets, anonymisation statistics, what is not anonymised, licence/consent status, verify command |

`--verify` re-hashes every listed file and reports `mismatched`, `missing` and `untracked`; exit code 1 on any mismatch or
missing file. The build ends with a verify pass.

## Face blurring

### Detector choice, measured on this footage

The brief preferred MediaPipe's tasks `FaceDetector` with `blaze_face_short_range.tflite`. The download URL
(`https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite`)
works (229 746 bytes, sha256 `b4578f35...0152b`), and `ensure_face_model()` fetches it once into `data/playground/models/`.
Used alone on full frames it is not fit for this footage: faces in the 1280x720 GoPro views are ~100 px wide, far below the
short-range model's design range, and the boxes it returned with scores 0.6-0.8 were on a shoulder, a bowl and the counter
while both real faces were unboxed (`scratchpad/det_gopro_front_300.jpg` from the exploration). A "frames with a face" count from
that detector is meaningless, so four candidates were compared on 30 sampled frames per stream (overlays saved during the run):

| detector | gopro_front (2 frontal/profile faces) | gopro_back (people mostly turned away) | leader ego | helper ego | verdict |
|---|---|---|---|---|---|
| MediaPipe BlazeFace full-range (bundled, legacy solutions API) | 28/30 frames, boxes on real faces | 2/30 | 5/30 | 0/30 | precise, misses profiles and far/turned faces |
| BlazeFace short-range on 3x2 overlapping tiles, upscaled 2x | 30/30, both faces + a false box on the counter | 6/30 | 24/30 (mostly false: bowl, wall) | 12/30 | good recall on frontal faces, false positives on objects |
| OpenCV Haar frontal | 12/30 | 11/30 | 13/30 (false) | 9/30 | weak |
| YOLOv8n-pose face keypoints -> head box | 30/30 | 29/30, boxes on both heads | 18/30 | 9/30 | the only one that covers turned-away heads |

Decision: the union of full-range + tiled short-range + pose head boxes, NMS at IoU 0.4, then tracking. Haar is used only
if mediapipe cannot be imported. The union is reported in `manifest.json.face_blur.backend` as `mp_full+mp_short_tiles+pose`.

First full build (tile detections kept at score >= 0.6 on their own): the example frame had both faces blurred but also
three large blurred regions with no face in them, the cutting board, the woman's torso, and the bowl with both people's
hands in it. In a handover dataset that destroys exactly what the exo view is for. So `filter_boxes()` now keeps a tile
detection only if a pose or full-range box corroborates it (centre inside, or IoU > 0.1) or its score is >= 0.85, and drops
any box larger than 35 % of the frame's shorter side. On the same 30 sampled frames per stream this left gopro_front with
30/30 frames detected and both faces boxed by `mp_short_tiles 0.93-0.96` inside the pose head boxes, with hands, bowl and
board untouched (`scratchpad/tight_gopro_front_{300,900}.jpg`). "False positives only cost blur" was wrong for this data:
they cost the hands.

### Tracking and blur

`FaceTracker`: detections are associated to tracks by IoU > 0.3 (box smoothed 0.6 new / 0.4 old); a track keeps blurring
for 0.5 s (`round(0.5 * fps)` frames) after its last detection, so a one-frame miss does not unblur a face. Boxes are enlarged
30 % around their centre, clipped to the frame, and Gaussian-blurred with an odd kernel of `max(15, 0.6 * max(w, h))` px.
Detection runs on every video frame at full resolution (a 640/960 px detection size lost recall on gopro_back and helper).

Cost on an M4 Pro: about 60 ms/frame on the 1280x720 GoPros and 150 ms/frame on the 1408x1408 ego streams (decode +
three detectors + encode), i.e. ~10-12 min for this 50 s, 4-stream episode.

### Known weak spots

- Partial faces at the fisheye edge of the ego cameras (e.g. helper frame 900: the leader's face half out of frame) were
  missed by every detector tested.
- Pose head boxes are generous (they include hair and glasses) but come from a person detector, so a person the pose
  model misses (heavy occlusion) gets no box from that source.
- Nothing else is anonymised: voices, names in the audio, body shape, clothing, tattoos, the room, on-screen text.

## Schema dictionary

`build_schema` matches column names against regexes built from the episode's actual stream and person names
(`_families()` in release.py): `t_s`, `<stream>_body2d_p<i>`, `<stream>_hand_<L|R>_{present,score,lm2d,lm3d}`,
`<stream>_objects`, `<stream>_imu_arm`, `<stream>_qc_{sharp,bright,motion}`, `world_body_p<i>`, `world_hands3d_<stream>`,
`world_head_<stream>`, `world_object_<name>`, `head_dist_m`, `min_wrist_dist_m`, `<stream>_facing_partner_cos`,
`<stream>_T_world_cam`, `<stream>_headpose_backend`, `body3d_p<i>`, `body3d_stream`, and the records_extra stage prefixes
`track_`, `contact_`, `gaze_proxy_`, `speech_` (`speech_speaking_<person>` specifically), `autolabel_`, `metrics_`
(`metrics_hand_speed_<person>`, `metrics_idle_<person>`, `metrics_hand_visible_<person>`), `stereo_depth_`, `depth_mono_`,
`annotate_`. Anything unmatched gets unit and meaning `unknown`. dtype and shape come from the data: list columns report
`list[<inner dtype>]` and `[length]` (or `"variable"`). The tests fail when more than 10 % of columns are unknown, on a
synthetic table and on the real CoMind records.parquet (64 columns, 0 unknown at the time of writing; the stage-prefix
families are generic by design because those stages are being built in parallel, so their meanings say which stage a
column comes from rather than what each specific column is).

## Results on `comind_43276420_clip` v0.1.0 (2026-09-26)

Command: `.venv/bin/python scripts/release_episode.py comind_43276420_clip --version v0.1.0 --example-png <scratch>/blurred_gopro_front_f300.png --example-frame 300 --example-stream gopro_front`,
then `... --verify`. Output: `data/playground/releases/v0.1.0/comind_43276420_clip/` (16 files, ~113 MB), code rev
`908b78a1`, backend `mp_full+mp_short_tiles+pose`. Detection statistics on the 1500 video frames (30 fps) per stream:

| stream | frames with >= 1 detection | frames blurred (with 0.5 s persistence) | detections by source |
|---|---|---|---|
| gopro_front (exo) | 1500 / 1500 | 1500 | pose 2014, tiles 2940, full-range 13 |
| gopro_back (exo) | 1495 / 1500 | 1500 | pose 2350, full-range 44, tiles 0 |
| leader (ego) | 883 / 1500 | 1170 | pose 849, tiles 242, full-range 39 |
| helper (ego) | 386 / 1500 | 899 | pose 379, tiles 34, full-range 67 |

Wall time 346 + 175 + 198 + 73 s. Every output has an H.264 video track (1500 frames; leader 1499, one frame short,
not investigated) and the original AAC audio track. `schema.json`: 64 columns, 0 unknown. `--verify`: ok, 16 files checked,
nothing mismatched, missing or untracked. Copied alongside records.parquet and episode.json: speech transcript, annotations,
autolabel proposals, contact events, metrics session, qc report (all existed by the time of the build).

Example frame (gopro_front, frame 300 of the blurred output, honest description): two soft rectangular blurs, one on each
person's head, each roughly 1.5x the head because of the 30 % enlargement on top of the generous pose head box; the blur
fully hides both faces including glasses, and the hair silhouette is still recognisable at the edge. Hands, the bowl the
two people are working in, the cutting board and the room are unblurred. In the first (loose) build the same frame also had
three large blurred rectangles on the cutting board, the woman's torso and the bowl-with-hands, which is why the filter
was added. The exo view with people turned away (gopro_back) is covered almost entirely by pose head boxes (2350 of 2394
detections), i.e. by the person detector rather than by a face detector; the helper ego stream, which rarely sees a face,
was blurred on 899 frames and its known miss (partial face at the fisheye edge) is unchanged.

## Not verified

- The blurred videos were inspected at one frame per stream; nobody has watched all four videos end to end.
- Audio stream-copy was exercised on GoPro/Aria-derived AAC mp4s only; the AAC re-encode fallback path is untested.
- The Haar fallback path was exercised only in the exploration, not through `release()`.
- The download URL was verified once on 2026-09-26.
