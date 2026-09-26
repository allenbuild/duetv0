# Duet EgoExo Playground (v0)

This is a local web app plus a processing pipeline. It turns a folder of videos (head-mounted "ego" and fixed "exo" cameras, optionally with an IMU harness) into a synchronised, annotated, scrubbable episode and a per-frame record table. Built 2026-09-25 and reworked 2026-09-26 after a full review and a follow-up code review (findings are summarised in the pull request that introduced the rework, branch `fix/playground-review`). This document describes the code after both rounds. The code is in `src/duet/playground/` (the module docstrings hold the conventions) and the CLI is `scripts/playground.py`. For the static site and deployment, see `docs/design/deploy_playground_plg.md`.

## Install

```sh
pip install -e '.[playground]'            # Python 3.11; add '.[dev]' for the tests
python scripts/playground.py fetch-models
```

- **The `playground` extra.** OpenCV >= 4.8 is pinned on all three wheel flavours, because ultralytics and mediapipe pull different ones; 4.7.0 computes wrong ChArUco poses, and `geometry.py` refuses it. The three flavours install the same `cv2/` files, so never pip-uninstall one of them afterwards: that deletes the shared files and breaks `import cv2`. A headless Linux host needs the system libGL for the GUI flavours (`apt install libgl1`). mediapipe is pinned to 0.10.21, because 1.0.x crashes on macOS Metal. The extra also installs pupil-apriltags, ultralytics, torch, torchvision, transformers, scipy, fastapi >= 0.115.2, starlette >= 0.39, uvicorn and python-multipart. The core dependency floor is av >= 14.1 (PyAV's hwaccel API).
- **System dependencies.** `ffmpeg` and `ffprobe` must be on PATH. COLMAP is needed only for the opt-in `scene_scan` stage (tested with 3.11.1 and 4.2.0). The conda-forge colmap 4.2.0 build for osx-arm64 lacks its openimageio 3.1 dependency, so install that alongside it.
- **`fetch-models`.** It downloads pinned URLs into `$PLAYGROUND_MODELS` (default `data/playground/models`) and checks each file's SHA-256. The `default` group holds MediaPipe `hand_landmarker.task` and `pose_landmarker_lite.task`, ultralytics v8.3.0 `yolov8n-pose.pt` and `yolov8s-worldv2.pt`, and Depth Anything V2 Small at a pinned Hugging Face revision; the `depth-metric-indoor` group adds the metric indoor checkpoint. A file that is present with a different hash is an error unless you pass `--force`.
- **No run-time downloads.** ultralytics runs with `YOLO_AUTOINSTALL=False` and `YOLO_OFFLINE=True`, so nothing is downloaded or pip-installed at run time. If a weight file is missing, its stage fails and the error names the fetch command.
- **YOLO-World vocabulary.** `fetch-models` bakes the objects vocabulary into `yolov8s-worldv2-vocab-<hash>.pt`, so CLIP is only a setup-time dependency, on the machine that runs `fetch-models` (`pip install git+https://github.com/ultralytics/CLIP.git`). Without CLIP, `fetch-models` warns and bakes nothing, and the `objects` stage then fails with instructions. Editing `perception.VOCAB` changes the file name, so re-run `fetch-models` afterwards.
- **depth_mono** needs torch, torchvision and transformers. On an offline host without pre-fetched weights, the stage is recorded as skipped.

## Run it

```sh
PG="python scripts/playground.py"
$PG create my_ep --ego leader=/path/a.mp4 --ego helper=/path/b.mp4 --exo cam1=/path/c.mp4 \
    --person leader=alice --person helper=bob [--imu leader=/path/imu.parquet] [--ref leader]
$PG run my_ep                                  # default stages; cached ones are skipped
$PG run my_ep --stages depth_mono,scene_scan   # opt-in stages
$PG serve                                      # http://127.0.0.1:8765 ("Run pipeline" / "Re-run all" in the UI)
```

| option | command | meaning |
|---|---|---|
| `--person NAME=LABEL` | create | who wears (ego) or is labelled by (exo) stream NAME; the old `NAME=PATH:LABEL` still works when unambiguous |
| `--ref NAME` | create, run | time-reference stream. A reference chosen this way is kept; the default (the first ego) may be replaced by align (see Time model) |
| `--offset NAME=S` | create, run | manual offset `t_ref = t_stream + S` for a stream without usable audio (e.g. ZED `left.mp4`); on `run`, `none` clears it |
| `--imu-offset NAME=S` | create, run | manual IMU clock offset `t_stream = t_imu + S`; negative means the IMU stamps run ahead of the video (eidon_10004 is about -0.41 s); on `run`, `none` restores the automatic estimate |
| `--fps F` | both | processing rate in (0, 60], default 10 |
| `--max-lag S`, `--min-overlap S` | both | align search window +-S (default 120 s); minimum audio overlap and common window (default 5 s) |
| `--device D` | both | `auto` (cuda:0, else mps, else cpu), `cpu`, `mps` or `cuda[:N]`; fp16 on CUDA only |
| `--hwaccel H` | both | ffmpeg decode acceleration: `auto` means videotoolbox on macOS, cuda with an NVIDIA GPU, else none, and a failed hardware attempt is retried in software. Under `auto`, PyAV decoding (the frames stage, and native frames for calib and tags) runs in software, which measured faster than VideoToolbox; an explicit type is used when the PyAV build offers it |
| `--copy`, `--move` | create | the default is relative symlinks. `--move` hard-links (or copies and fsyncs) each file into staging, commits the episode, and only then removes the originals; it takes regular files only and never files under `data/raw` |
| `--overwrite`, `--discard-old-streams`, `--keep-zed` | create | replace an existing episode: derived data is deleted, `rig.json` and `zed/` are kept. Moved or copied-in stream and IMU files (possibly the only copy) are deleted only with `--discard-old-streams`, else the command refuses. `zed/<s>/` is moved to `zed/<s>.old-<time>/` when stream s gets a different source, unless `--keep-zed` |
| `--stages a,b`, `--force` | run | run a subset (always in canonical order); re-run even when cached |
| `--drop-frames` | run | delete the JPEGs after a successful export (see Caching) |
| `--root DIR`, `--host`, `--port` | serve | episodes root (also a global option), bind address, port |

`run` writes its config options (`--ref`, `--offset`, `--fps`, ...) into `episode.json` under the episode's run lock, just before running, so stages whose fingerprints include them re-run. If another pipeline holds the lock, nothing is written and the command exits 2.

Episodes live under `--root`, else `$PLAYGROUND_EPISODES`, else `data/playground/episodes/`; a leading `~` in the environment paths is expanded. An episode holds `<name>/episode.json`, `streams/`, `imu/`, an optional `rig.json` and `zed/<stream>/`, and `derived/` (outputs, with run logs in `derived/logs/`).

Other environment variables:

- `PLAYGROUND_MODELS`.
- For the server: `PLAYGROUND_TOKEN`, `PLAYGROUND_MAX_JOBS`, `PLAYGROUND_KILL_GRACE_S`, `PLAYGROUND_MAX_UPLOAD_MB` and `PLAYGROUND_MIN_FREE_MB`.
- For scene_scan: `PLAYGROUND_CACHE`.
- `PLAYGROUND_QC_WORKERS`.

### ZED flow

```sh
python scripts/zed_export.py take.svo2 <episodes>/my_ep/zed/leader/ --fps <camera fps>    # on the NVIDIA machine
$PG create my_ep --ego leader=<episodes>/my_ep/zed/leader/left.mp4 --exo cam1=/path/c.mp4 --ref cam1 --offset leader=<seconds>
```

- **Creating into the prepared folder.** `create` accepts an existing episode folder that holds only `rig.json` and `zed/`, and it accepts sources inside that `zed/` (symlink or copy mode, not `--move`).
- **Reference and offset.** ZED video has no audio, so the stream needs a manual `--offset`, and the reference must be a stream with audio: align never switches the reference while any stream has a manual offset.
- **Depth.** `--fps` equal to the camera rate exports depth for every grab, which ZED hands need.
- **Re-exporting.** zed_export refuses a non-empty output folder. `--overwrite` deletes an earlier export there, but only the files the script itself writes.
- **Identity check.** `export.json` records the video's frames, fps, duration, size and a head/tail hash. The playground refuses a `zed/<s>/` that does not match stream s ("an export from another take?").

## Stages

The default order is probe, align, frames, calib, body2d, hands, objects, tags, headpose, body3d, world3d, qc, imu_arm, export. qc runs before imu_arm because the IMU clock is estimated from qc's camera-motion signal. depth_mono and scene_scan run only when named in `--stages`. Every stage after align skips streams whose offset is unknown ("unaligned").

| stage | needs | writes (under `derived/`) | what |
|---|---|---|---|
| probe | | `episode.json` | ffprobe: largest non-cover-art video stream, fps, video and audio start, duration, and display orientation from the first frame's matrix (rotation, flip, display size) |
| align | probe | `episode.json` | audio offsets, drift, common window (see Time model) |
| frames | align | `frames/<s>/*.jpg`, `frames/<s>/index.parquet`, `frames/manifest.json` | exactly `n_frames()` JPEGs per usable stream (q85, long side 640), each the source frame nearest a grid time; decoded once with PyAV, streams in parallel, display orientation (rotation, flip) applied |
| calib | frames | `calib/calib.json` | intrinsics and board registration; skipped, with nothing written, when no camera registers |
| body2d | frames | `body2d/<s>.npz` | YOLOv8n-pose on every view; up to 4 people in tracker slots (Hungarian IoU/centre matching, duplicates merged), stable within one stream only |
| hands | frames | `hands/<s>.npz` | MediaPipe HandLandmarker (VIDEO mode, up to 4 hands) on ego views; each hand is attributed jointly to wearer or partner and to left or right. body2d is optional; without it, only geometric cues are used |
| objects | frames | `objects/<s>.npz` | YOLO-World with the baked kitchen/household vocabulary on ego views; class-agnostic NMS; up to 12 boxes per frame |
| tags | frames, calib | `tags/<s>.parquet` | configured AprilTags in registered fixed views, with poses in the world frame; skipped when `rig.json` has no tag ids |
| headpose | frames, calib | `headpose/<ego>.npz` | ego pose per frame: ZED tracking registered to the board, else the head tag composed with `T_tag_headcam` |
| body3d | frames, body2d | `body3d/body3d.npz` | MediaPipe pose world landmarks on the exo view with the largest person; monocular; skipped without an exo view |
| world3d | frames, calib | `world3d/world3d.npz` | bodies associated across views and triangulated, ZED-depth hands, head poses, tag objects, cross-person features |
| qc | frames | `qc/<s>.npz`, `qc/report.json` | Eidon-style quality scores and verdicts |
| imu_arm | align | `imu_arm/<s>.npz` | Eidon 7-slot IMU harness to a torso-relative arm chain on the reference timeline |
| export | frames | `records.parquet` | one row per processed frame |
| depth_mono (opt-in) | frames | `depth_mono/<ego>.npz` | Depth Anything V2 on ego frames; metric where AprilTag anchors fit |
| scene_scan (opt-in) | frames, calib | `scene_scan/` | COLMAP sparse points of the static scene, in board metres |

Some stages also read optional inputs, but only from stages that are done: hands reads body2d; headpose reads tags; world3d reads body2d, hands, tags and headpose; qc reads body2d and hands; imu_arm reads frames and qc; export reads every stage; depth_mono reads calib; and scene_scan reads body2d for people masks. Nothing reads the depth_mono or scene_scan outputs yet.

### Status, caching and invalidation

- **States.** A stage is done, skipped, failed, running, interrupted or stale. A run reports each selected stage as cached, kept, done, skipped, failed or interrupted.
- **Fresh vs done.** Inside a run, stages check `Episode.stage_ok(stage)` (done), because the run marks stale results as it goes. Readers outside a run use `run.fresh_stages` / `stage_fresh` instead: done and all inputs still match, cascaded, without writing (about 0.5 ms). A rig.json edit made outside a run is therefore caught. The server's payloads and QC report, and the static exporter, use only fresh stages; nothing is read just because a file exists.
- **Caching.** A stage is cached only when it is done and its fingerprint matches; otherwise it re-runs. `--force` re-runs the selected stages and bypasses their own caches, such as scene_scan's station cache.
- **Fingerprint inputs.** The episode config groups the stage depends on: stream file size and mtime; the stream list and persons; align parameters and overrides; the timeline (reference, proc_fps and proc_size, window, and each stream's offset, drift, status, video start, rotation, flip, size, fps and duration); sha256 of `rig.json` and `zed/<s>/*.json`, and size and mtime of other zed files (`zed/*.old-*` excluded); IMU file identity and override. The reference is an align input only when chosen explicitly; an automatically picked reference is align output. Also hashed: the state and fingerprint of every upstream stage in its `DEPS` or `USES` (`run.py`), the source of its module and of every `duet.*` module it imports (transitively, found statically), its parameters, and the sha256 of the weight files it loads (including the baked YOLO-World file). device and hwaccel are excluded.
- **Staleness.** The status keeps a hash per group, so a stale stage says what changed. Before and after each stage, and on every exit of `run_stages` (interrupts included), done stages whose fingerprint no longer matches become stale, cascading downstream. Stages recorded by the old code have no fingerprint; they stay done until an upstream stage changes or a required one is not done.
- **Kept.** A stage whose required stages are not done is not run. Without an earlier result it is recorded as `skipped: needs X`. An earlier result is left as it is (files and record) and reported as "kept"; it becomes stale only if its inputs changed.
- **Missing sources.** If a stream file is missing (dangling link, unmounted drive), probe, align and frames are not run: an earlier result is kept, and a stage without one is recorded as skipped. A result whose only changed input is the missing file keeps its status, and so do done stages written by the old code. A weight file that was not fetched on this host likewise keeps the recorded result.
- **Outputs.** A stage's outputs are deleted before it runs, and again if it fails, skips itself or is interrupted.
- **`--drop-frames`.** After a successful export it deletes the JPEGs and marks frames stale (dropped), keeping its fingerprint, so downstream stages stay cached. Re-running frames restores the JPEGs.
- **Writes.** Only probe and align write episode config. `episode.json` is written atomically (temp file, fsync, rename) under an fcntl lock, with a three-way merge per stage and per stream field. The server, a pipeline subprocess and the CLI therefore never tear or clobber it. Old files load unchanged and are upgraded to schema 2 on the first write.
- **Liveness.** Every pipeline holds the fcntl lock `<ep>/run.lock`, which the kernel releases when the process dies. The CLI, the job queue, `create --overwrite` and restart recovery all use it to tell whether a pipeline is running.
- **Logs.** Runs log to `derived/logs/<ts>-<pid>.log` (CLI) or `<ts>-job.log` (server job), with tracebacks and ffmpeg stderr. The newest 50 logs are kept; subfolders such as `derived/logs/scene_scan/` are left alone. Status details are capped at 2,000 characters, with local paths stripped.

## Time model

All times are in seconds. `t_stream` is a stream's own timeline, with 0 at its first video frame (container pts minus `video_start_s`). `t_ref` is the reference stream's timeline. For each stream:

    t_ref = (1 + drift_ppm * 1e-6) * t_stream + offset_s

`Episode.ref_time` and `Episode.stream_time` are the only sanctioned conversions; the stages, the exporter and the UI all use this formula.

- **`offset_status`.** It is one of `reference`; `audio` (estimated by align); `manual` (`--offset`, used as given, with drift 0); or `unaligned` (no usable audio, or no accepted peak). Audio is unusable when it is missing, silent, undecodable, or shorter than `min_overlap_s`. An `unaligned` stream gets offset 0, is left out of the common window and is skipped by every consumer (`Episode.usable`). In legacy files, a stream counts as `audio` when its `offset_confidence` is >= 1.3, else as `unaligned`.
- **Reference choice.** `Episode.reference_explicit` records whether the user chose the reference (`--ref` on create or run, or the upload form). A default reference without usable audio is replaced by the stream with the longest usable audio (ties by name), and the switch is noted. Align never switches while any stream has a manual offset, since offsets are relative to the reference. An explicit reference is always kept. A reference without usable audio then fails align unless every other stream has a manual offset.
- **Align.** Audio is decoded on the file's own timestamps and padded or trimmed so that sample 0 is `t_stream` 0; this honours late audio starts, AAC priming and edit lists. (The old code used container time, so its offsets are off by any difference between the audio start and the video start.) Onset envelopes on 10 ms hops are cross-correlated linearly over their full length. Each lag with abs(lag) <= `max_lag_s` is scored as Pearson r over its overlap, times sqrt(overlap), and lags with less than `min_overlap_s` of overlap are not searched. The peak is refined below one hop by a parabolic fit.
- **Peak acceptance.** A peak is accepted only if its robust z-score is >= 8, it is >= 1.5x the best peak at least 0.5 s away (this ratio is `offset_confidence`), and it lies more than 0.5 s inside +-max_lag. A peak at the edge is reported as "search edge": raise `--max-lag`. A missed alignment shows up as `unaligned` with a note.
- **Drift.** Overlaps of 300 s or more are re-measured in 60 s windows, and a robust line fit gives `offset_s` and `drift_ppm`. Shorter overlaps get drift 0 with a note.
- **Common window.** The common window runs from the latest start to the earliest end of the usable streams, in reference time. Align fails if it is shorter than `min_overlap_s`. Processed frame k is nominally at `t_ref = common_start_s + k / proc_fps`, for k < `n_frames()`.
- **Frames.** Frame k of stream s is the source frame whose pts is nearest to `stream_time(s, t_ref_k)`; ties go to the earlier frame. `frames/<s>/index.parquet` records `k`, `file`, `t_ref_s` (nominal), `t_src_s` (the pts of the frame used, in stream time) and `src_frame`. frames warns when the nearest frame is more than half a processed period away, which happens with gaps or a variable frame rate. The old code kept the last frame in each bucket, so its frames lag by 30-33 ms at 30 fps. Episodes without an index fall back to nominal times.
- **IMU clock.** The model is `t_stream = (1 + imu_drift_ppm * 1e-6) * t_imu + imu_offset_s`, where `t_imu = time_ms / 1000` on the IMU's own clock; then `t_ref = ref_time(s, t_stream)`. imu_arm correlates the chest sensor's angular speed with qc's camera motion over +-2 s in 10 ms steps. It applies the offset only if r >= 0.2, p <= 0.01 against a circular-shift null, the peak ratio is >= 1.3, the peak lies inside the window and at least 100 frames were compared (`imu_arm.SYNC`).
- **IMU drift check.** Recordings of 120 s or more are split into windows: 2 below 3 min, otherwise max(3, duration / 5 min). Each window gets its own gated estimate. If the windows agree within half a processed frame, the constant offset is applied. Otherwise, a well-determined line (>= 3 windows, small residuals and slope uncertainty, abs(drift) <= 1000 ppm) is applied as offset plus drift; if no such line fits, the result is `drift_suspected`. With fewer than 2 significant windows, the constant offset is applied and the check reads "unverified".
- **`imu_sync`.** It is one of `estimated`, `manual` (`--imu-offset`, with drift 0, needing no camera motion; the estimate is still reported), `unsynced` or `drift_suspected`. Only the first two are applied. For the other two, offset and drift are 0, which follows the dataset card's claim that camera and IMU were started back to back, and the viewer flags the clock.

## Coordinate conventions

AGENTS.md forbids silent guesses. These conventions are verified by rendering at known poses (`tests/test_playground_geometry.py`, `tests/test_playground_world*.py`).

Distances are in metres. Image coordinates are pixels, with integer coordinates at pixel centres (the OpenCV convention). `T_a_b` maps b coordinates to a: `p_a = T_a_b @ p_b`. Camera axes follow OpenCV: x right, y down, z forward. Poses are stored as `T_world_cam` (camera to world, so the translation is the camera centre); OpenCV's (rvec, tvec) is `T_cam_world`. `T_tag_headcam` and the `calib.json` poses are validated as rigid before use: finite, orthonormal, determinant +1, last row 0 0 0 1.

- **Display orientation.** probe reads the display matrix from the first frame (container or H.264/HEVC SEI), falling back to the stream's matrix, then to the legacy rotate tag. It stores `Stream.rotation` (0/90/180/270, counter-clockwise as ffprobe reports it), `Stream.flip` (true when the matrix is a reflection) and the raw `display_matrix`. The displayed image is `np.rot90(decoded[::-1] if flip else decoded, rotation // 90)`: flip vertically first, then rotate. A plain horizontal mirror is (180, flip). Width and height swap only for 90 and 270. frames, calib and tags all use this transform, and it matches ffmpeg's autorotate pixel for pixel. `flip` is part of the frames fingerprint.
- **World = board.** The world frame is the ChArUco board at the calibration anchor frame (`calib.json` `anchor_frame`). The origin is the top-left outer corner of the printed board, as `CharucoBoard.generateImage` draws it, seen from the front. X runs along `squares_x` (right), Y along `squares_y` (down the page), and Z = X x Y points into the board. Cameras that see the board are at Z < 0; for a face-up board, +Z points into the table, so "above the table" is Z < 0. The first inner corner is at (square_m, square_m, 0).
- **AprilTag.** tag36h11 tags are detected with pupil-apriltags (AprilTag 3). The tag frame has its origin at the tag centre, X right, Y down and Z into the tag face, for a tag upright as in the official AprilRobotics `apriltag-imgs` bitmaps. That is the board convention: a tag printed upright on the board has `R_world_tag = I`. The old code's 180 deg flip about X was a bug that made head cameras look backwards. `tag_size_m` is the edge of the black border square, which is 8 of the official image's 10 cells (the quiet zone is excluded). It has no default and is required whenever `head_tags` or `object_tags` are set: a wrong default silently scales every tag distance. Print the official PNGs: `cv2.aruco`'s DICT_APRILTAG_36h11 images are rotated 180 deg. Detected corners are shifted by -0.5 px to OpenCV's pixel-centre convention.
- **Head tag.** `T_tag_headcam` is the head camera's pose in the tag frame: it maps head-camera coordinates to tag coordinates, and is measured once per rig. The head camera's world pose is `T_world_headcam = T_world_fixedcam @ T_fixedcam_tag @ T_tag_headcam`.
- **ZED.** `scripts/zed_export.py` (export format 2) writes `pose.csv` as `T_zedworld_camera` in the SDK's IMAGE coordinate system (= OpenCV camera axes), in metres. `export.json` records `coordinate_system`, `coordinate_units` and `video_timeline: capture`, and headpose refuses unknown conventions. Format 1 exports (RIGHT_HANDED_Y_UP, video written by grab count) are converted exactly (`T @ diag(1, -1, -1, 1)`), but frames after the first dropped grab are rejected. Tracking is registered to the board through the ego's board sightings. With 3 or more sightings the registration uses `duet.geometry.alignment.fit_pose_alignment` (IRLS on camera centres and orientations); with 1-2 it uses a robust mean of per-sighting estimates. Unregistered tracking stays NaN and never enters board-frame outputs. Depth is stored as 16-bit PNG, in millimetres along the optical axis. ZED hands use only the depth map of the same grab as the image; frames whose grab has no exported depth stay NaN.
- **Hand `lm3d`.** MediaPipe hand world landmarks in metres, now wrist-relative (landmark 0 subtracted; `lm3d_origin="wrist"`). v1 files (no `schema` key) are hand-centred, with the wrist about 8 cm from the origin. Readers handle both.
- **body3d.** MediaPipe pose world landmarks in metres, hip-centred, with camera-aligned axes: x right, y down, z away from the camera. They are monocular, one exo view at a time. The viewer draws (x, -y + 0.9, -z); the old mapping mirrored the body.
- **IMU arm** (`imu_arm.FRAME`). Metres in Eidon Sim scene axes (+Y up), with the heading removed using the chest sensor and the shoulders at (-0.18, 1.40, 0) and (+0.18, 1.40, 0). This frame is not registered to the board.

Worked example for `T_tag_headcam`. A tag is worn upright on the forehead, facing forward, with the lens 5 cm below its centre. Seen from the front, the tag's X points to the wearer's left, Y points down and Z points into the head. The camera's right, down and forward axes are therefore the tag's -X, +Y and -Z, so R = diag(-1, 1, -1) and t = (0, 0.05, 0):

```json
"T_tag_headcam": {"leader": [[-1, 0, 0, 0], [0, 1, 0, 0.05], [0, 0, -1, 0], [0, 0, 0, 1]]}
```

There is no default, and identity is not neutral: it would mean the camera looks into the tag face. Every head tag needs a rigid `T_tag_headcam` with |t| < 0.5 m, and tag id 0 is valid. A tag seen by one camera can have two planar pose solutions (small, distant or fronto-parallel tags). `tags/<s>.parquet` keeps both and flags `ambiguous`; only temporal continuity resolves which one is right.

- **Intrinsics resolution.** `rig.json` intrinsics may be given at any resolution with the display aspect ratio, for the display orientation (e.g. the native 3840x2160 K). `Intrinsics.scaled_to` rescales them, pixel-centre aware, and an aspect change above 1 % fails calib and names the stream.
- **Intrinsics model.** `model` is `pinhole` (OpenCV k1 k2 p1 p2 [k3]), `rational` (8, 12 or 14 coefficients) or `fisheye` (cv2.fisheye k1..k4). `dist` is required, and `[]` means no distortion. Without `model`, up to 5 coefficients read as pinhole and 8/12/14 as rational; fisheye must be declared. GoPro-class lenses need rational or fisheye: self-calibration rejects a pinhole fit that is more than 10 % worse than a rational one.
- **Intrinsics sources**, in order:
  1. The ZED factory calibration (`zed/<s>/calibration.json`).
  2. `rig.json`.
  3. Board self-calibration. It needs >= 8 views with board normals in >= 3 directions at least 15 deg apart, RMS <= 1 px, a 1-sigma on f and c of <= 1 %, and the principal point in the central 60 %. A fixed camera watching a board slide on one plane cannot self-calibrate.
  4. A nominal FOV from `hfov_deg` (default 100 deg exo, 110 deg ego). This is a guess, so the camera is not registered unless `allow_nominal` is set.
- **Detection.** calib and tags detect the board and tags on native frames, decoded at each index row's `t_src_s` with the long side capped at `detect_max_px`. They fall back to the extracted frames. `calib.json` intrinsics are scaled to the extracted frames, where the body2d, hands and objects pixels live.
- **Registration.** calib samples the first `calib_window_s` of the common window at `calib_hz`, so show the board then. Fixed cameras are registered at the anchor frame where most of them see the board at once, averaged over the samples where the board is still. A camera that saw the board only at other times is chained through a registered camera that saw it at the same frame. Residuals are reported. A camera whose registrations disagree by more than 3 cm / 2 deg, or that moved, is unregistered with a reason. Egos get board poses at their board sightings (`sightings`; `T_world_cam` is null). Without any fixed camera, the board is assumed static, and this is flagged. A camera with nominal-FOV intrinsics is never registered or used as a body view, ego or exo, unless `allow_nominal` is set.

## rig.json

`rig.json` is optional and lives in the episode folder. Missing keys take `world.DEFAULT_RIG`. Every value is validated: sizes must be in metres, and tag ids must be unique non-negative integers.

| key | default | meaning |
|---|---|---|
| `board` | 7 x 5 squares, 0.05 / 0.037 m | `squares_x`, `squares_y`, `square_m`, `marker_m`; optional `legacy: true` for boards from OpenCV < 4.6 or "legacy" generators (their layout differs when `squares_y` is even); optional `dictionary` (a cv2.aruco name, default `DICT_5X5_100`) |
| `tag_size_m` | none | black-border edge in metres; required when `head_tags` or `object_tags` are set (depth_mono anchors need it too) |
| `head_tags` | {} | {person label or stream name: tag id} |
| `object_tags` | {} | {object name: tag id} |
| `T_tag_headcam` | {} | {same key as `head_tags`: 4x4}; required for every head tag |
| `intrinsics` | {} | {stream: {`K`, `dist`, `width`, `height`, `model`}} |
| `hfov_deg`, `lens_model` | {} | {stream: nominal horizontal FOV}; {stream: pinhole / rational / fisheye}, used for self-calibration and nominal FOV |
| `allow_nominal` | false | register cameras whose intrinsics are only a FOV guess |
| `calib_window_s`, `calib_hz` | 20, 2 | which frames calib samples |
| `detect_max_px` | 1920 | long-side cap for native board and tag detection |
| `tags_native` | true | detect tags on native frames; this costs a full decode of each fixed video (about 4.5 instead of 1.3 min per 1080p exo stream-hour) |
| `depth_mono`, `scene_scan` | {} | stage options; the valid keys are each module's `DEFAULTS`, and unknown keys raise |
| `wall_tags` | | not supported (ignored with a note) |

```json
{"board": {"squares_x": 7, "squares_y": 5, "square_m": 0.05, "marker_m": 0.037},
 "tag_size_m": 0.08, "head_tags": {"leader": 1, "helper": 2}, "object_tags": {"bowl": 10},
 "T_tag_headcam": {"leader": [[-1,0,0,0],[0,1,0,0.05],[0,0,-1,0],[0,0,0,1]],
                   "helper": [[-1,0,0,0],[0,1,0,0.05],[0,0,-1,0],[0,0,0,1]]},
 "intrinsics": {"cam1": {"K": [[1400,0,959.5],[0,1400,539.5],[0,0,1]], "dist": [0,0,0,0,0],
                         "width": 1920, "height": 1080, "model": "pinhole"}},
 "lens_model": {"gopro_front": "fisheye"}, "hfov_deg": {"gopro_front": 120}}
```

## Outputs (`derived/`)

- **frames**: `frames/<s>/000001.jpg` onwards (1-based names, 0-based k) and `index.parquet`. `frames/manifest.json` holds each stream's size, its `scale` (extracted width / display width), the source identity, and the offsets, window and proc_fps used.
- **body2d** (`<s>.npz`): `kpts [n,4,17,3]` (x px, y px, conf), `boxes [n,4,5]`, `img_w`, `img_h`, `schema` 2. Empty slots are NaN.
- **hands** (`<s>.npz`, `schema` 2): the wearer's `lm2d [n,2,21,2]` in px, `lm3d [n,2,21,3]` in m (wrist-relative), `present [n,2]` and `score [n,2]`, with slot 0 = left and slot 1 = right. `score` is the MediaPipe handedness probability of the slot's side (NaN when absent), not a detection confidence. The partner's hands use the same keys with a `partner_` prefix. Also stored: `p_wearer`, `partner_p_wearer`, `n_detected [n]`, `n_merged`, `side_midline`, `owner_prior`, `partner_seen_s`, `partner_declared`, `wearer`, `partner` and `lm3d_origin`. The strong wearer prior is used only when the wearer is alone: the episode declares no other person, and body2d shows no face in this view (without body2d the prior stays off). In v1 files there are no partner keys, and `score == 0` means absent.
- **objects** (`<s>.npz`): `boxes [n,12,5]` (x0, y0, x1, y1 and conf, in px, most confident first) and `names [n,12]`.
- **calib** (`calib.json`): top-level `world` "board", `units` "m", `board`, `anchor_frame`, `anchor_segment` and `board_assumed_static`. Per camera: `intrinsics` (at extracted size), `intrinsics_source` (zed_factory / rig_json / board / nominal_fov), `intrinsics_input`, `frame_size`, `detect_size`, `detect_frames` (native / extracted), `board_views`, `registered`, `T_world_cam` (fixed cameras; null otherwise), and `registration` (method, residual and spread, or a `reason`). Egos also get `sightings` [{k, T_world_cam, board: observed / static_inferred / assumed_static, err_px}].
- **tags** (`<s>.parquet`): frame, tag_id, err_px, err_rad, cx/cy (extracted px), `Tc..` (T_cam_tag) and `Tw..` (T_world_tag). The alternate planar solution is in `Ta..`, with `alt_err_px` and `ambiguous`.
- **headpose** (`<ego>.npz`): `T_world_cam [n,4,4]` (board frame, NaN when unknown), `valid [n]` and `backend` (zed_tracking+board / head_tag, or zed_tracking_unregistered with every pose NaN). ZED egos also get `zed_grab` and `zed_t`.
- **body3d** (`body3d.npz`): `world [n,2,33,3]`, `img2d [n,2,33,3]`, `vis [n,2,33]`, `stream`.
- **world3d** (`world3d.npz`, board frame, metres): `bodies [n,2,17,3]`, `body_err [n,2,17]` (px), `body_track [n,2]` and `body_person [2]` (the ego whose wearer fills the slot, or "" if unlinked). Per ego: `hands3d_<ego> [n,2,21,3]` (ZED depth), `head_<ego> [n,3]` and `T_world_cam_<ego> [n,4,4]`. Also `object_<name> [n,3]` and `feat_min_wrist_dist_m`, plus `feat_head_dist_m` and `feat_<ego>_facing_partner_cos` [n] when there are >= 2 egos.
- **imu_arm** (`<s>.npz`, schema 2): the original `t_imu_s` and `imu_time_ms`, plus `t_stream_s` and `t_s` (reference time). `points [T,2,4,3]` holds shoulder, elbow, wrist and fingertip per side (NaN when invalid), with `valid`, `hand_valid` and `slot_valid`. Also stored: the applied `imu_sync`, `imu_offset_s` and `imu_drift_ppm`, and the estimate with its statistics. A slot is valid only between raw samples at most 1.5 median periods apart; 20-40 % of Eidon's 24 Hz snapshots are repeats. A side needs upper arm, forearm and chest, and the fingertip also needs the hand sensor.
- **depth_mono** (`<ego>.npz`, schema 2, one sample every 0.5 s at 256 px): `depth_m` (float16 metres along the optical axis, NaN where the frame has no fit), `pred` (raw model output), `metric`, `fit_source` and anchor tables. `fit_source` is one of `anchors`; `borrowed` (the nearest fit within `borrow_s`, approximate); `model` (the metric checkpoint, uncorrected); or `none`. Metric depth fits `1/Z = a*d + b` to at least 2 AprilTag anchors whose depths differ by >= 1.5x. Anchors need calib done, an explicit `tag_size_m`, and non-nominal intrinsics. ZED egos are skipped.
- **scene_scan** (`points.npz`, schema 2, plus `scan.json`): `xyz_board` (board metres) only when aligned, else `xyz_colmap`, plus `rgb`, `err_px`, `track_len`, `T_world_colmap` and `scale`. The default `triangulate` method keeps the calibrated poses fixed (COLMAP point_triangulator on cross-camera pairs). The opt-in `mapper` method writes board keys only if its alignment passes (rotation <= 1 deg, centre RMS <= 3 cm). The stage needs >= 2 registered static fixed cameras. COLMAP logs go to `derived/logs/scene_scan/`. Results are cached per station in `$PLAYGROUND_CACHE` (default `<episodes root>/_cache`) and reused while the board poses agree within 1 cm / 0.5 deg.

### QC (`qc/report.json`, schema 2)

Per stream, the report gives:

- `good_frame_percent`: sharpness > 40 and grey in (35, 225).
- `hand_presence_ratio` (ego: wearer hands, with `partner_hand_presence_ratio` alongside) or `person_presence_ratio` (exo).
- `lighting_score`: fraction of frames with grey in [35, 225].
- `average_brightness`.
- `stability_score` = clip(1 - median motion / 0.1875, 0, 1). Motion is the median optical flow in frame widths (long side) per second, over actual source-time differences, so it does not depend on proc_fps. Consecutive frames that share one source frame are allowed (when proc_fps >= the source rate, or across VFR gaps: `t_src_s` only has to be non-decreasing); such a pair gives no motion sample.
- `missing`, `flags`, `verdict` and `reasons`.

Per episode, it gives `both_visible_ratio`, `hands_all_egos_ratio`, `min_good_frame_percent` and `min_alignment_confidence`. A metric with missing input is None with a reason, never 0, and aggregates need every stream.

| metric | reject below | flag below |
|---|---|---|
| good_frame_percent | 0.5 | 0.8 |
| hand_presence_ratio | 0.2 | 0.3 |
| average_brightness | 20 | 40 |
| lighting_score | 0.5 | 0.8 |
| person_presence_ratio | | 0.5 |
| stability_score | | 0.25 |
| episode: both_visible_ratio | | 0.5 |
| episode: hands_all_egos_ratio | | 0.3 |

The hand-presence and brightness levels follow the tracker-pov card. The card measures HSV V, while we measure grey, which is <= V. The overall verdict is reject if anything rejects (an unaligned stream rejects); otherwise flag if anything is flagged or missing; otherwise pass. `THR` is written into the report.

### records.parquet (schema v2)

There is one row per processed frame. The column set depends only on the episode configuration: streams, roles, IMU streams, number of egos and `rig.json` object tags. A stage's columns are filled only when it is done, and are null otherwise. Lists are `list<float32>`. null means not available: the stage is not done, the stream is unusable, the entity is absent, or no IMU sample lies within 0.75 periods. NaN inside a list means one missing component.

| columns | content |
|---|---|
| `frame_idx`, `t_ref_s` | k and its nominal reference time |
| `<s>_t_src_s`, `<s>_src_frame`, `<s>_frame_file` | the source frame used |
| `<s>_body2d_p{0..3}` [51], `<s>_body2d_box_p{0..3}` [5] | px |
| `<s>_qc_sharp`, `_qc_bright`, `_qc_motion` | per-frame QC |
| ego: `<s>_hand_{L,R}_*`, `<s>_partner_hand_{L,R}_*` | `present`, `handedness_prob`, `p_wearer`, `lm2d` [42] px, `lm3d` [63] m wrist-relative |
| ego: `<s>_hands_n_detected`, `<s>_objects` (JSON), `<s>_T_world_cam` [16] | row-major 4x4, camera to board |
| IMU: `<s>_imu_arm_{L,R}` [12], `<s>_imu_t_src_s` | nearest raw sample within 0.75 periods, and its original IMU-clock time |
| `world_body_p{0,1}` [51], `world_hands3d_<ego>_{L,R}` [63], `world_head_<ego>` [3], `world_object_<name>` [3], `world_feat_<name>` | board frame, m |
| `body3d_p{0,1}` [99] | monocular, hip-centred |

The parquet key-value metadata holds `duet.schema_version` = "2" and `duet.records`. The latter is JSON with the time model, reference, offsets, drift, frame sizes, stage states and fingerprints, the IMU tolerance, `missing_outputs` (done stages whose file is absent), the coordinate frames, and each column's type, length, unit and frame. Export fails if a per-frame artifact does not have exactly `n_frames()` rows, or if world3d writes a feature that the schema does not declare.

## Server and viewer

`serve` binds 127.0.0.1:8765 by default.

- **Default deny.** Access is decided on the route path, the path the router matches with any root_path stripped. Only the UI shell (`/`, `/index.html`, `/app.js`) and `/api/session` are public; every other path is protected. The UI uses relative URLs, so it also works under a path prefix.
- **Without `PLAYGROUND_TOKEN`**, the server is loopback only. It answers 403 to a non-loopback peer, to a non-localhost Host header (DNS rebinding), and to proxy headers (X-Forwarded-For, Forwarded, X-Real-IP, X-Forwarded-Host). `serve` also refuses a non-loopback bind.
- **With `PLAYGROUND_TOKEN`** (>= 16 characters after stripping surrounding whitespace, which logs a warning), protected paths need `Authorization: Bearer <token>`, `X-Playground-Token: <token>`, or the session cookie from `POST /api/session {"token": ...}`. The cookie is HttpOnly, SameSite=Strict, Secure over https, valid for 12 h, and HMAC-signed with a key derived from the token, so rotating the token revokes every session. The UI shows a sign-in box on 401. Without TLS in front, the token and the video travel in clear text.
- **Rate limiting.** Every failed credential counts against the client IP: a wrong Bearer or X-Playground-Token header, a forged, malformed or non-ASCII cookie, or a failed login. After 10 failures within a minute, further requests from that IP get 429, even with the correct token. Missing or expired credentials get 401 without counting. The failure table is an LRU of up to 4,096 IPs.
- **CSRF.** Every non-GET request needs `X-Playground-Request: 1` (or a valid `X-Playground-Token`). A foreign `Origin` is refused, as is `Sec-Fetch-Site: cross-site`/`same-site` on protected paths and on any state-changing request.
- **Headers.** Responses carry CSP, X-Frame-Options DENY, nosniff, Referrer-Policy no-referrer and CORP same-origin. The UI shell is sent with `no-cache`, and `index.html` loads `app.js?v=<content hash>`, so an old `app.js` never meets a newer payload layout.
- **Videos.** Only the stream videos listed in `episode.json` are served, at `/api/episode/<n>/video/<stream>`, and Range requests get 206. The old `/episodes/<n>/<stream path>` URL serves the same files and nothing else. Videos and Range responses are never gzip-compressed.
- **Derived data** leaves only through the JSON endpoints, and only for fresh stages. `/api/episode/<n>` also returns `stale` ({stage: reason}) and `job`, the last background job. Its status entries show only state, detail, started and finished. The UI shows stale stages as "stale" chips with the reason, and a failed or interrupted last job as a banner.
- **Uploads.** `POST /api/upload` takes multipart name, roles, persons, reference, fps and files (.mp4/.mov/.m4v/.mkv/.webm); the form's reference select makes the chosen reference explicit. The body is streamed once into `<root>/.upload-<random>/`, capped at `PLAYGROUND_MAX_UPLOAD_MB` per request (default 8192), and refused with 507 if less than `PLAYGROUND_MIN_FREE_MB` (default 2048) would remain free. The files are then moved in by `Episode.create(mode="move")` and the pipeline is queued; creation is staged and renamed, so a failed upload leaves nothing behind.
- **Payloads.** `overlay/<s>`, `body3d`, `world3d` and `imu_arm/<s>` take `?start=&end=` in processed frames. They carry an ETag derived from their source files and stage freshness (304 when unchanged), and are gzip-encoded once and cached. Overlays are in extracted-frame pixels. The UI loads episodes of up to 3,000 frames whole, and longer ones in 1,200-frame windows with prefetch.

**Jobs.**

- `POST /api/episode/<n>/run?stages=&force=` queues a run (202). It returns 409 if the episode is already queued or running, including a CLI run that holds `run.lock`. `GET /api/queue` shows the queue.
- One worker thread (`PLAYGROUND_MAX_JOBS`, default 1) runs jobs one at a time, each in its own subprocess (`python -m duet.playground.run <ep_dir>`) and session, so a native crash cannot take the server down.
- `status["_job"]` records every job: running, done, failed (with the log tail) or interrupted. This includes processes that die before any stage starts, such as "killed by SIGKILL" or "exited with code N".
- `POST /api/episode/<n>/cancel` (the UI's Cancel button) drops a queued job. For a running job, it sends SIGTERM to the process group, then SIGKILL after `PLAYGROUND_KILL_GRACE_S` (default 10 s); the stages end interrupted. A job cancelled between leaving the queue and spawning never starts. The route returns 409 when this server has nothing queued or running for the episode, since it does not cancel CLI runs.
- On startup, "running" records of episodes whose run lock is free become interrupted.

**Viewer time.** A `<video>` element's time is `t_stream + video_start_s` for the server's original files, and `t_stream - trim_start_s` for the static site's trimmed clips. IMU arms are drawn only within 0.75 sample periods of the playhead. The 3D panel flags `unsynced` and `drift_suspected` IMU clocks, and `legacy_unverified` ones (v1 imu_arm files from before the sync check).

## Validation

Results are synthetic unless marked real. Run `python -m pytest tests/test_playground_*.py`. The real-COLMAP and real-MediaPipe tests skip when those are not installed.

| area | evidence |
|---|---|
| align | ffmpeg-generated clips with planted +-2.37 s offsets, a 0.5 s audio-start delay, and silent and video-only streams: offsets within 6 ms. Speech-like scenes: 0-0.33 % false accepts on unrelated pairs (old rule: 6-12 %). 100 % of 10 s overlaps aligned at 10 dB SNR, 68 % of 20 s overlaps at 0 dB. Drift over 1 h recovered. A default reference without audio switches to the longest usable audio; an explicit reference or manual offsets keep the failure |
| frames | nearest-frame choice. Display orientation is pixel-identical to ffmpeg autorotate for all 8 rotation/mirror transforms, and for SEI-only rotation and flips, in frames and in calib/tags native decoding. Real CoMind clips: source frames within 3.3 ms of nominal (the committed frames lag 30-33 ms) |
| geometry | board and AprilTag frames checked by rendering official bitmaps at known poses; full 6-DoF tag and head-from-tag poses with a rotated 3-axis lever arm; degenerate calibration view sets rejected; triangulation cheirality, conditioning, outlier view and weighting |
| world stages | 3-camera episode at 1280x720, with the rig K at native resolution and 640 px frames, run through the real probe and frames stages. Camera registration: about 0.2 cm / 0.1 deg with native decode, about 1 cm / 0.5 deg from extracted frames. Head-from-tag: about 0.1 cm / 0.1 deg (the old code was 53-79 cm and about 180 deg off). Also covered: ZED capture timeline with dropped grabs, legacy Y-up conversion, and an export from another take being refused. ZED hands use the depth of their own grab only. ZED registration via fit_pose_alignment lands camera positions within a few mm (the per-sighting mean it replaces: about 9 mm). Association handles mirrored camera order, occlusion, crossing and bystanders, and world3d runtime is linear in length (about 0.7 ms per frame) |
| hands | real: CoMind Aria MPS wearer-hand tracking as ground truth (a one-off measurement, not a repo test). Share of wearer-slot hands that are the partner's: 49.5 to 19.4 % (leader) and 35.4 to 12.2 % (helper). Wearer side correct in 99.2 / 98.4 %; left/right flips <= 0.9 %; every detected hand kept. Wearer recall 40 / 48 %, against a MediaPipe ceiling of 40 / 53 %. Real: on eidon_10004 (single wearer), every hand is attributed to the wearer |
| body2d, objects | real: YOLOv8n-pose on the CoMind GoPros; the phantom third person drops from 47 to 6 frames. Agnostic NMS leaves no duplicate object pairs, and the baked vocabulary gives bit-identical boxes without CLIP |
| IMU clock | planted offsets; 1 h at +100 ppm fitted as +99.0 ppm (3 ms error). Real: eidon_10004 is estimated at -0.41 s, with quaternion and gyro agreeing within 11 ms |
| scene_scan | real COLMAP 3.11.1 and 4.2.0 on a rendered known-pose scene: median distance to the true surfaces about 0.4 mm (triangulate) and 0.6 mm (mapper) |
| server, export | FastAPI TestClient covers: access rules, including route-path decisions under a root_path and rate limiting of every failed credential; traversal; uploads; queue dedupe; cancel before spawn, SIGKILL escalation and job records; only fresh stages served; windowed payloads; and a copy of a committed episode still served. The export schema is stable across stage sets. The static exporter trims MPEG-TS, offset-MKV and offset-MP4 sources to within half a frame |
| core | fingerprint cascade; a final staleness sweep after interrupts; kept results; `run` config waiting for the run lock; `--move` never losing originals; the ZED prepared-folder flow; QC and IMU with repeated source frames |

## Known gaps

- **Real rigs.** Board calibration, tags, head poses and triangulation are validated on synthetic renders only; the first real ZED and cheap-rig recordings are still to come. `scripts/zed_export.py` has run only against a fake `pyzed.sl`, and its `imu.csv` axes are unverified.
- **Sync.** Audio alignment needs audio on every stream (clap at block start), so ZED and other silent streams need `--offset`, plus an explicit `--ref` that has audio. A stream-copied cut (`-c copy -ss`) can shift a file's video against its own audio by about 30 ms, which audio alignment cannot see. Drift is estimated only for overlaps of 5 min or more. The viewer assumes browsers report container time for files whose video starts late; this is unchecked across browsers.
- **IMU.** Drift precision on long real recordings is unmeasured, because there is no long IMU+video episode yet. The arm chain is torso-relative, not in the board frame.
- **Hands.** The owner cues were fitted on Aria glasses and may need refitting for other head cameras. On CoMind, 12-19 % of wearer-slot hands are still the partner's, and 40 of the helper's own hands land in partner slots. MediaPipe caps wearer recall at about 40-50 %. World-frame hands exist only for ZED egos, from depth.
- **Bodies.** Without head poses, the longest tracks fill the two slots. There is no re-identification, so a long-present bystander can take a slot. A two-view wrong-person match cannot be rejected geometrically. In ego-only rigs each person is seen by one camera, so bodies are not triangulated. body3d is monocular.
- **Geometry.** The 3 px board gate cannot catch a wrong focal length when the board is small in the image, which is why only calibrated cameras register. A calibration extrapolates beyond the board's coverage (reported as `coverage` and `max_radius_frac`). The fisheye FOV must be < 180 deg. Single-camera tag orientation ambiguity is resolved only by continuity.
- **No SLAM/VIO.** Head cameras other than ZED need a head tag seen by a registered fixed camera.
- **Opt-in stages.** world3d, export and the viewer do not read the depth_mono or scene_scan outputs yet. depth_mono writes about 180-350 MB and needs about 1 GB RAM per ego stream-hour at defaults. The COLMAP mapper fails its gates on 2-3 camera rigs, so its points stay in COLMAP's frame.
- **Throughput.** It has not been measured on a GPU host.
- **Layout.** The Eidon 7-slot IMU schema is now parsed in `src/duet/adapters/eidon_imu.py`, and `imu_arm` re-exports it. The ZED export format and the CoMind kitchen vocabulary still live in `src/duet/playground/`, not in `src/duet/adapters/` as AGENTS.md asks.

## Test episodes

Both episodes were produced by the old code (v1 layouts, no frames index, no fingerprints), and both are still served; the legacy read paths are tested. Their stream links are absolute paths that resolve only on the machine that created them, and the Eidon source video is not in the repo. Elsewhere, probe, align and frames are therefore not run, and the old results are kept. Both are still tracked in git under `data/playground/`. Review P0 #2 leaves untracking or purging them as a team decision.

- `comind_43276420_clip`: 2 ego (Aria) + 2 exo (GoPro) views, 50 s, deliberately mis-cut. Its `episode.json` lists world3d and depth_mono as done, but their outputs are not in the repo. world3d and export are reported stale: world3d now requires calib, which was skipped because there is no board. Neither the viewer nor the static export serves them.
- `eidon_10004`: 1 ego view + IMU, 62 s; a single stream, so there is no alignment. The IMU arm chain appears in the 3D panel. Its committed imu_arm file predates the sync check, so the viewer labels the clock `legacy_unverified`; the measured IMU clock offset is about -0.41 s.

## Background

- **Eidon.** The datasets are `eidon-ai/tracker-pov` (13,451 ego recordings, 1,274 h, CC-BY-4.0) and `tracker-pov-imu` (7-slot quaternions at 24 Hz; its license is not recorded in the repo). `eidon_10004` is one recording plus its IMU rows. The `qc` stage is modelled on their QC metrics and card thresholds. `imu_arm.py` ports the `eidon-sim` (MIT) arm kinematics. `eidon-tracker` and `eidon-glove` are open hardware (about $80-120 per IMU node, 1 kHz quaternions over BLE) and a cheap body-tracking option for our own capture spec.
- **Munari** (Alan Guo, a16z Speedrun) is a closed product. It reconstructs egocentric recordings in 3D and draws hand and camera paths to score demonstrations. The playground covers similar ground with QC and overlays, minus SLAM camera trajectories.

## Stages added 2026-09-26 (24 stages total)

Each stage has its own design note in `docs/design/stage_<name>.md` with the validation numbers and the exact commands.

| stage | what | validated on |
|---|---|---|
| stereo_depth | depth from a ZED left/right pair without the ZED SDK (SGBM + WLS), PNGs in the SDK layout | synthetic stereo: 0.1–0.5 % median depth error |
| track | stable person ids (Hungarian on IoU + torso colour, 2 s memory), cross-view matching | CoMind exo views: fragmentation removed, 0 jumps; cross-view fallback flagged ambiguous on same-coloured tops |
| contact | which object is in which hand (fingertips in box, hysteresis), grasp/release events | CoMind clip: 2 of 3 annotated handovers recovered within 0.6 s; wearer vs partner hands is the main error |
| gaze_proxy | head-forward gaze ray, angle to partner/objects, mutual gaze (3D via headpose, 2D fallback) | CoMind: head-forward predicts joint attention as well as real gaze (AUROC 0.788 vs 0.776, 11 recs); gaze sits 18° from the RGB axis, mostly pitch |
| speech | faster-whisper transcript per stream, speaker = loudest own mic, de-duplicated across mics | CoMind clip: coherent two-person dialogue; leader mic needed peak normalisation |
| annotate | contact-sheet keyframes every 2 s to a VLM (anthropic SDK / claude CLI / dry), strict JSON | plumbing only: no credentials on this Mac, dry mode |
| autolabel | GBDT trained on 44 CoMind recordings using only features both rigs produce; proposals for review | CV joint attention 0.756±0.034 (full-feature model 0.703), handover 0.604±0.112; playground clip misses handovers from MediaPipe hand-presence shift |
| metrics | operator numbers: handovers/min, receiver latency, hold, synchrony, idle, speaking, coordination score | CoMind clip: 10 contact handovers (upper bound), score 75 |

Also: `scripts/zed_mac_record.py` (ZED as a UVC webcam on macOS, factory calibration by serial, bundle compatible with `zed_kit.py`), `scripts/intake.py` (SD-card dumps → episodes, chunk concat, session split, drift correction), `scripts/release_episode.py` (face blur, schema, manifest, license, consent), review page (`review.html`, human QA with keyboard shortcuts, proposal accept/reject), annotation page, metrics page.
