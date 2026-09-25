# Duet EgoExo Playground (v0)

A local web app plus processing pipeline that turns a folder of videos (head-mounted and fixed cameras, optionally an IMU harness) into a synchronised, annotated, scrubbable episode and a per-frame record table. Built 2026-09-25. Target: the "EgoExo Playground" flow (align videos -> 2D body pose -> 3D body pose) extended with hands, objects, quality scores and export.

## Run it

```sh
.venv/bin/python scripts/playground.py create my_episode --ego leader=/path/a.mp4:alice --ego helper=/path/b.mp4:bob --exo cam1=/path/c.mp4 [--imu leader=/path/imu.parquet] --ref leader
.venv/bin/python scripts/playground.py run my_episode          # or press "Run pipeline" in the UI
.venv/bin/python scripts/playground.py serve --port 8791       # http://127.0.0.1:8791
```

Episodes live in `data/playground/episodes/<name>/` (`episode.json`, `streams/`, `imu/`, `derived/`).

## Stages (`src/duet/playground/`)

| stage | what | notes |
|---|---|---|
| probe | ffprobe every stream | fps, size, duration, audio |
| align | audio cross-correlation of onset envelopes vs the reference stream | offsets to ~30 ms, confidence = peak ratio; flagged below x1.3. Validated on a CoMind clip with planted offsets (recovered to 0.03 s) and a same-instant frame grid |
| frames | ffmpeg extraction on the common timeline at `proc_fps` (10) | frame k is the same instant in every stream |
| body2d | YOLOv8n-pose, every view, up to 4 IoU-tracked people | 17 COCO keypoints |
| hands | MediaPipe HandLandmarker on ego views | 21 landmarks, image px + metric wrist-relative 3D; MediaPipe 0.10.21 (1.0.x crashes on macOS Metal) |
| objects | YOLO-World, household/kitchen vocabulary, ego views | held object = box containing >= 8 hand landmarks (UI) |
| body3d | MediaPipe PoseLandmarker world landmarks on the exo view with the largest person | monocular 3D, hip-centred metres. NOT triangulated; multi-view triangulation needs calibrated cameras (next) |
| imu_arm | Eidon 7-slot IMU harness -> chest-relative arm chain | ported from Eidon Sim (MIT): quaternionToVectors, chest yaw, humerus 0.30 / radius 0.26 / hand 0.10 m |
| qc | Eidon-style scores per stream: good_frame_percent, hand/person presence, stability, lighting; episode: both people visible, hands in every ego view | thresholds written into the report |
| export | `derived/records.parquet`, one row per common frame | hands 2D/3D, bodies, objects, body3d, IMU arm, QC |

## What we pulled from Eidon

- Dataset: `eidon-ai/tracker-pov` (13,451 ego recordings, 1,274 h, CC-BY-4.0) + `tracker-pov-imu` (7-slot quaternions at 24 Hz). Test episode `eidon_10004` is one recording plus its IMU rows.
- Their QC metric set (metadata.parquet columns) is the model for our `qc` stage.
- `eidon-sim` (MIT) arm kinematics, ported to `imu_arm.py`; their Three.js visualisation informed the 3D panel.
- `eidon-tracker` / `eidon-glove` are open hardware ($80-120 per IMU node, 1 kHz quats over BLE): a cheap body-tracking option for our own capture spec.

## What Munari does that we now also do

Munari (Alan Guo, a16z Speedrun) is a closed product that reconstructs egocentric recordings in 3D and draws hand and camera pathways to score demonstration quality. Nothing to pull; the playground covers the same ground with the QC stage and the overlays, minus SLAM camera trajectories (next step: run a VIO/SLAM on ego streams and add the camera path to the 3D panel).

## Test episodes

- `comind_43276420_clip`: 2 ego (Aria) + 2 exo (GoPro) views, 50 s, deliberately mis-cut so alignment has work to do. All stages pass; 466 frames x 50 columns.
- `eidon_10004`: 1 ego + IMU, 63 s. Alignment skipped (single stream), IMU arm chain rendered in the 3D panel.

## Known gaps

Body3d is monocular; no camera calibration or triangulation yet. No SLAM camera trajectory. Hand tracking on ego views is 2D + wrist-relative 3D, not metric world-frame hands (HaWoR/WiLoR would replace it). Alignment needs audio on every stream (clap at block start).
