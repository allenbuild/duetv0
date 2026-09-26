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

## Geometry stages (added 2026-09-25): calibration, tags, head pose, triangulation

World frame = the ChArUco board at station setup (OpenCV convention: origin top-left corner, X right, Y down the board, Z into the board; cameras above the board have negative Z). Configure a rig with `rig.json` in the episode folder:

```json
{"board": {"squares_x": 7, "squares_y": 5, "square_m": 0.05, "marker_m": 0.037},
 "tag_size_m": 0.08, "head_tags": {"leader": 1, "helper": 2}, "object_tags": {"bowl": 10}, "wall_tags": [20, 21],
 "T_tag_headcam": {"leader": [[1,0,0,0],[0,1,0,0.05],[0,0,1,0],[0,0,0,1]]},
 "intrinsics": {"exoA": {"K": [[...]], "dist": [...], "width": 1280, "height": 720}}, "hfov_deg": {"gopro_front": 120}}
```

| stage | what | backends / notes |
|---|---|---|
| calib | intrinsics per camera and T_world_cam for every camera that sees the board | intrinsics source, in order: ZED factory file (`zed/<stream>/calibration.json`), `rig.json` (calibrated once at install with a moving board), board views if the board moved enough, nominal from FOV. A fixed camera watching a static board cannot self-calibrate; that is why rig intrinsics exist |
| tags | AprilTag 36h11 detection in fixed views, poses in world | pupil-apriltags; tag frame converted to the board convention (flip about X) |
| headpose | ego camera pose per frame | (a) ZED positional tracking from `scripts/zed_export.py` (`zed/<stream>/pose.csv`), registered to the board through the frame where the ZED saw it; (b) head tag seen by a fixed camera times the once-measured tag-to-camera offset |
| world3d | triangulated bodies (2 people, greedy left/right matching), metric hands from ZED depth at MediaPipe landmarks, object positions from tags, head positions, cross-person features (head distance, min wrist-wrist distance, facing) | needs >= 2 cameras with world poses |
| depth_mono | Depth Anything V2 on ego frames without a depth sensor, scaled to metres by tags visible in the ego frame | relative if no anchor; every 0.5 s at 256 px |
| scene_scan | once-per-station COLMAP sparse reconstruction from fixed-camera frames with people masked, aligned to the board frame | skipped unless `colmap` is installed and a fixed camera is registered |

Validated on synthetic scenes (`tests/test_playground_geometry.py`, `tests/test_playground_world_synthetic.py`): intrinsics within 3 %, camera registration within 2-3 cm, tag position within 3 cm, head-from-tag within 4 cm, triangulation within 2 cm. Not yet validated on real board/tag footage: that is what the first ZED + cheap-rig recordings are for.

ZED path: run `scripts/zed_export.py capture.svo2 episodes/<name>/zed/<stream>/` on the NVIDIA machine, register `zed/<stream>/left.mp4` as the ego stream. Cheap path: any MP4 head cam with a tag on the strap, two fixed cameras with `rig.json` intrinsics, board shown at setup.

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
