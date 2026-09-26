# Recording a ZED Mini on a Mac without the ZED SDK

`scripts/zed_mac_record.py`. The ZED SDK needs CUDA, so the NVIDIA-machine path (`scripts/zed_kit.py`,
`docs/design/zed_friend_guide.md`) cannot run on a Mac. The camera itself is a plain UVC device: it streams both eyes
side by side in one video frame, and macOS exposes it through AVFoundation. ffmpeg captures that, we split it, and we
fetch the same factory calibration the SDK downloads. The result is a bundle with the layout `zed_kit.py pack` makes,
so the playground ingests it the same way (`playground.py create --zed`, or the upload page as an `ego` file).

## What you get, and what you do not

| SDK bundle (`zed_kit.py`) | Mac bundle (`zed_mac_record.py`) |
|---|---|
| `left.mp4` (left eye, no audio) | `left.mp4` (left eye, **with** the Mac mic audio, for alignment) |
| | `right.mp4` (right eye) |
| `calibration.json` (left intrinsics) | `calibration.json` (left intrinsics **+ `stereo` block**: right intrinsics, R, T) |
| `pose.csv` (ZED positional tracking) | none: `headpose` falls back to the head tag seen by a fixed camera |
| `imu.csv` | none (the IMU is only reachable through the SDK / HID protocol) |
| `depth/000123.png` (NEURAL depth) | none at record time: the `stereo_depth` stage computes them (`docs/design/stage_stereo_depth.md`) |
| `export.json` `source` absent | `export.json` with `"source": "uvc_mac"`, `fps`, frame count, wall-clock start |

Because `world.py::_zed_dir` currently requires `pose.csv`, a Mac bundle's factory intrinsics and computed depth are
not yet consumed by `calib`/`world3d` (see the gap note in `stage_stereo_depth.md`); the video, audio alignment,
all 2D perception, and head-tag head pose work unchanged.

## UVC modes (both eyes side by side)

| `--res` | frame | fps | per eye | calibration section |
|---|---|---|---|---|
| `720` (default) | 2560x720 | 30 | 1280x720 | `*_HD` |
| `1080` | 3840x1080 | 30 | 1920x1080 | `*_FHD` |
| `2k` | 4416x1242 | 15 | 2208x1242 | `*_2K` |
| `vga` | 1344x376 | 30 | 672x376 | `*_VGA` |

## Commands

```bash
python scripts/zed_mac_record.py list                       # avfoundation devices; the ZED shows as e.g. "[2] ZED-M"
python scripts/zed_mac_record.py record take1 --serial 12345678 [--res 720] [--seconds 600] [--audio "MacBook Pro Microphone"]
python scripts/zed_mac_record.py export take1               # split + calibration; needs internet once per serial
python scripts/zed_mac_record.py pack take1                 # -> zed_captures/take1/take1_zed.zip
python scripts/zed_mac_record.py all take1 --serial 12345678 --seconds 600
```

Files, mirroring `zed_kit.py`:

```
zed_captures/take1/
    sbs.mp4                 raw side-by-side capture (keep it)
    record.json             device, serial, mode, wall-clock start/end, the ffmpeg command, ffprobe of sbs.mp4
    export/left.mp4 right.mp4 calibration.json export.json SN12345678.conf
    take1_zed.zip
zed_captures/calib/SN12345678.conf    cached factory calibration
```

`record`: finds the video device whose name contains "zed" (`--device` to override), the first audio device
(`--audio NAME` or `--audio none`), and runs
`ffmpeg -f avfoundation -framerate 30 -video_size 2560x720 -i <video>:<audio> -c:v h264_videotoolbox -b:v 30M -c:a aac sbs.mp4`
(`--encoder libx264` for a software fallback). Enter, or `--seconds`, stops it (a `q` is written to ffmpeg so the mp4
is finalised). The wall-clock start (`start_utc`, `start_unix`) is the moment ffmpeg was launched; the first frame
arrives a few hundred ms later, so use the audio alignment, not this, for sync. **If no ZED is connected the command
exits with `no ZED found` and the list of devices it did see.** If ffmpeg reports that 2560x720 is not supported, the
device is not a ZED or is on a USB 2 port.

`export`: `ffprobe` on `sbs.mp4` gives the per-eye width, which picks the calibration section
(2208 -> 2K, 1920 -> FHD, 1280 -> HD, 672 -> VGA, `--res-key` to override). One ffmpeg pass writes
`crop=iw/2:ih:0:0` to `left.mp4` (audio copied) and `crop=iw/2:ih:iw/2:0` to `right.mp4` (no audio), libx264 crf 18.
The left eye is the left half (zed-open-capture: `left_raw = frame(cv::Rect(0, 0, cols/2, rows))`).

`pack`: calls `zed_kit.pack` (imported from the sibling script; it imports the SDK lazily so this works on a Mac).

## Factory calibration file

Downloaded from `https://calib.stereolabs.com/?SN=<serial>` (the URL `downloadCalibrationFile` in Stereolabs'
zed-open-capture `examples/include/calibration.hpp` and zed-opencv-native `python/zed_opencv_native.py` use; the SDK
stores the same file as `~/zed/settings/SN<serial>.conf`, which can be copied over and passed with `--calib`). It is
an INI read with a case-insensitive parser (SimpleIni). Format as read by `initCalibration` in that header:

```ini
[LEFT_CAM_HD]            ; also LEFT_CAM_2K, LEFT_CAM_FHD, LEFT_CAM_VGA
fx=... fy=... cx=... cy=...
k1=... k2=... p1=... p2=... k3=...
[RIGHT_CAM_HD]
...
[STEREO]
Baseline=62.99           ; millimetres
TY_HD=... TZ_HD=...      ; millimetres, per resolution
RX_HD=... CV_HD=... RZ_HD=...   ; radians; R = cv::Rodrigues([RX, CV, RZ])
```

`stereo_params` reproduces the C++ exactly: `K = [[fx,0,cx],[0,fy,cy],[0,0,1]]`, `dist = [k1,k2,p1,p2,k3]`,
`R = Rodrigues([RX_<res>, CV_<res>, RZ_<res>])`, `T = [Baseline, TY_<res>, TZ_<res>]` (with `TY`/`TZ` as fallbacks
when the per-resolution key is absent), converted to metres. Sign of `T[0]`: the C++ sample passes `+Baseline` to
`cv::stereoRectify`, the Python sample `-Baseline`; OpenCV's convention (`X_right = R X_left + T`, right eye at +x) is
the negative one, so we store `T[0] = -Baseline` and say so in `calibration.json["stereo"]["T_convention"]`. Rectification
is identical either way and the depth stage uses `|T|`.

`calibration.json` keeps every key `zed_export.py` writes (`K, dist, width, height, fps, serial, model, baseline_m`)
so `zed_bundle` and `world.py::calib` treat both bundles alike, adds `source`, `resolution_key`, and the `stereo`
block `{right: {K, dist}, R, T, rodrigues_rx_cv_rz, units: "m"}`. If the exported eye width differs from the section's
nominal width (e.g. a test video), both K are scaled by the ratio.

## Tested without hardware (`tests/test_zed_mac_record.py`, 4 tests, pass)

- conf parsing against a sample written in the documented format (sections per resolution, `[STEREO]` keys,
  decimal commas), Rodrigues/T assembly, missing-section error, resolution key table
- `calibration.json` shape and K scaling
- `export` on a synthetic side-by-side video (ffmpeg lavfi `testsrc | smptebars` hstack + sine audio): eye sizes,
  audio only on `left.mp4`, halves land in the right files and are not swapped, `export.json` source/fps,
  `zed_bundle.is_bundle` on the folder, rectification maps build from the written calibration; then `pack` and
  `zed_bundle.ingest` of the zip
- `list`, and `record` failing with `no ZED found` when no ZED is attached (macOS only; skipped if a ZED is present)

Not tested (no ZED on this Mac): the actual avfoundation capture of a ZED (device name, that the reported modes are
selectable, pixel format, dropped frames at 30 fps with h264_videotoolbox), the download from calib.stereolabs.com for
a real serial, and the sign/units of a real conf against real depth. First thing to do with a camera:
`python scripts/zed_mac_record.py all check1 --serial <SN> --seconds 20`, then create an episode from the zip and run
`stereo_depth`; a hand held at a measured 0.5 m should read 500 +- 25 mm in `zed/<stream>/depth/*.png`.

`ffmpeg -f avfoundation -list_devices true -i ""` on this Mac (2026-09-26, ffmpeg 9.0.2) listed video
`[0] MacBook Pro Camera`, `[1] MacBook Pro Desk View Camera` and audio `[0] MacBook Pro Microphone`,
`[1] Microsoft Teams Audio`: no ZED, as expected.
