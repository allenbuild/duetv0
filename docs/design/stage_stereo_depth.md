# `stereo_depth` stage: ZED depth without the ZED SDK

`src/duet/playground/stereo_depth.py`. Runs after `frames` (and `calib`) in the playground pipeline. For every ego
stream that carries a left/right stereo pair it computes metric depth with OpenCV and writes the same depth PNGs the
SDK export (`scripts/zed_export.py`) would have shipped, so nothing downstream has to know which machine recorded.

## Inputs and outputs

```
<episode>/zed/<stream>/left.mp4            left eye (this is also the stream's video)
<episode>/zed/<stream>/right.mp4           right eye, same frame count and fps
<episode>/zed/<stream>/calibration.json    K, dist, width, height, fps, ...  +  "stereo": {right: {K, dist}, R, T (m)}
<episode>/zed/<stream>/export.json         optional; its "fps" is the frame clock (see below)

-> <episode>/zed/<stream>/depth/<kz:06d>.png      uint16 millimetres, 0 = unknown, resolution of left.mp4,
                                                   in the ORIGINAL (unrectified) left pixel grid
-> <episode>/derived/stereo_depth/records_extra.parquet   per processed frame: <stream>_zed_frame, _valid_frac,
                                                   _median_depth_m (export merges these as stereo_depth_<stream>_*)
```

The `stereo` block is written by `scripts/zed_mac_record.py export` from the Stereolabs factory calibration
(`docs/design/zed_mac_record.md`). `T` is in metres in OpenCV's `cv2.stereoRectify` convention (`X_right = R X_left + T`,
right eye at +x of the left eye so `T[0] = -baseline`). Only `|T|` enters the depth formula and rectification is
invariant to the sign, so a `+baseline` file would also work.

## Naming and indexing: what world3d expects

`world.py::world3d` reads depth for processed frame `k` (the common-timeline frame at `proc_fps`) as

```python
kz = int(round((ep.common_start_s + k / ep.proc_fps - s.offset_s) * json.load(open(zd / "export.json"))["fps"]))
dp = zd / "depth" / f"{kz:06d}.png"
depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
```

so a depth PNG is named by the **0-based frame index of `left.mp4` at its native fps** (`zed_export.py` names them by
the SDK grab counter `k`, `f"{k:06d}.png"`, and writes `np.clip(depth_m * 1000, 0, 65535).astype(np.uint16)`).
Hand landmarks are scaled by `depth.shape / intrinsics.(width, height)`, so the PNG may be any resolution; we keep the
native one. The stage mirrors that formula exactly (`zed_frame_indices`, `zed_fps`: `export.json["fps"]`, else the
probed stream fps) and computes depth **only at those `kz`** values, not for every video frame. Example: a 30 fps
recording processed at 10 fps yields `000000.png, 000003.png, 000006.png, ...`.

## Method

1. `cv2.stereoRectify(K_l, d_l, K_r, d_r, (w, h), R, T, CALIB_ZERO_DISPARITY, alpha=-1)` (intrinsics scaled if the
   calibration was written for another width), `initUndistortRectifyMap` for both eyes, plus the inverse map
   original-left-pixel -> rectified-pixel from `cv2.undistortPoints(..., R=R1, P=P1)`.
2. Both eyes remapped, converted to grey, optionally downscaled to `work_width` (default 1280: 720p runs native, 1080p
   and 2K are matched at 1280 wide and the disparity scaled back).
3. `cv2.StereoSGBM_create` (3-way mode, block 5, P1/P2 = 8/32·bs², uniqueness 10, speckle 100/2, LR check 1 px),
   `numDisparities` from `f·B / min_depth_m` (0.25 m -> 192 for a 720p ZED Mini).
4. If `cv2.ximgproc` (opencv-contrib) is importable: right matcher + `createDisparityWLSFilter` (lambda 8000,
   sigma 1.5). Pixels where the raw left disparity was invalid stay invalid (WLS is not allowed to invent depth).
   Without contrib the raw SGBM disparity is used.
5. `Z_rect = f_rect · B / d`; the rectified-frame point `(X, Y, Z_rect)` is rotated by `R1ᵀ` so the value is the
   depth along the **original** left camera's optical axis; then resampled onto the original pixel grid with the
   inverse map (nearest neighbour, so holes and depth edges are not blended). Depth outside
   `[0.5·min_depth_m, max_depth_m]` (default 0.125 to 8 m) is written as 0.

Defaults live in `stereo_depth.DEFAULTS`; `stereo_depth(ep, params={...})` overrides them.

## Skip conditions (status `skipped` with the reason in `detail`)

- the ego stream has no `zed/<stream>/left.mp4` + `right.mp4` + `calibration.json`
- `calibration.json` has no usable `stereo` block
- `zed/<stream>/depth/*.png` already exists (SDK export or an earlier run): never recomputed, never overwritten
- no extracted frames (`frames` has not run)

One episode can mix: a stream with SDK depth is skipped, a Mac-recorded one is computed; the status lists both.

## Validation (synthetic, run 2026-09-26)

`tests/test_playground_stereo_depth.py` renders two pinhole cameras (f = 700 px, 1280x720, baseline 0.063 m) looking at
two textured fronto-parallel planes: a 0.6 x 0.45 m panel at 0.8 m and a background at 1.5 m. Measured median relative
depth error over pixels with a depth value (target < 3 %):

| rig | matcher | near plane 0.8 m | far plane 1.5 m | coverage near / far | time per 720p frame |
|---|---|---|---|---|---|
| identity R, no distortion | SGBM + WLS | 0.11 % (p90 0.11 %) | 0.51 % (p90 0.73 %) | 100 % / 81 % | 99 ms |
| identity R, no distortion | plain SGBM | 0.11 % | 0.51 % | 100 % / 79 % | 52 ms |
| right eye rotated 0.7 deg, both eyes distorted (ZED-like k1..k3, p1, p2) | SGBM + WLS | 0.10 % (p90 0.20 %) | 0.50 % (p90 0.68 %) | 100 % / 76 % | |
| same | plain SGBM | 0.09 % | 0.51 % | 100 % / 74 % | |

The far plane's missing coverage is the left image border that the right eye cannot see (disparity search band) and
the occlusion shadow of the near panel; both are correctly written as 0.

End to end through the CLI (a 3 s, 30 fps static bundle of the same scene, `tmp_zed_synth`, deleted afterwards):

```
.venv/bin/python scripts/playground.py create tmp_zed_synth --zed leader=<bundle>/export:synth --fps 10
.venv/bin/python scripts/playground.py run tmp_zed_synth --stages probe,align,frames,calib,stereo_depth
```

gave `stereo_depth: done leader: 30/30 frames, valid 85%, WLS at fps clock 30`, PNGs `000000.png, 000003.png, ...,
000087.png` (every index world3d's formula produces was present), uint16 720x1280, values 799..1551 mm, per-frame
median error 0.12 % (near) and 0.53 % (far) on all 30 frames. Whole run 5.3 s wall.

Real-camera accuracy will be worse than this (sensor noise, rolling shutter, textureless walls, the factory
calibration's own error); the ZED SDK quotes roughly 1-2 % at 1 m for its own stereo matching, and SGBM on a 63 mm
baseline degrades quadratically with distance (at 3 m one disparity step is ~6 cm). Depth is for "where is the hand
in metres", not surfaces.

## Cost

~0.1 s per 720p frame with WLS on an M-series Mac (0.05 s without), single-threaded OpenCV. A 10-minute session at
`proc_fps` 10 is 6000 frames, about 10 minutes. 1080p/2K sources are matched at 1280 wide, so the cost is the same.

## Known gap: world.py only consumes a `zed/` folder that has `pose.csv`

`world.py::_zed_dir` returns the folder only if `pose.csv` exists. A Mac recording has no SDK tracking and therefore
no `pose.csv`, so today `calib` will not pick up the factory intrinsics and `world3d` will not read these depth PNGs
for such a stream (headpose falls back to the head tag as designed; depth is simply unused). This stage is correct and
complete on its side (the PNGs are exactly what `world3d` reads); the one-line change needed in `world.py` is to gate
`_zed_dir` on `calibration.json` and let `headpose` check for `pose.csv` itself. That file was out of scope for this
change and is left untouched; flagged in the delivery report.
