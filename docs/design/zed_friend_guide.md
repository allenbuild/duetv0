# ZED Mini on the NVIDIA computer: what to run before the playground

Yes, there is one script. The ZED Mini does not save video on its own; it streams to the computer, and the ZED SDK
turns that stream into a recording plus depth and head tracking. `scripts/zed_kit.py` wraps all of it in four
commands and produces one zip that the playground understands. Windows or Ubuntu, same commands.

## One-time setup (about 20 minutes)

1. **Install the ZED SDK** from https://www.stereolabs.com/developers/release (pick the installer matching your OS and
   the CUDA version it bundles; the installer offers to install CUDA). Reboot if it asks.
2. **Install Python 3.10–3.12** (Windows: python.org installer, tick "Add to PATH").
3. **Install the ZED Python API**:
   - Windows: `python "C:\Program Files (x86)\ZED SDK\get_python_api.py"`
   - Ubuntu: `python3 /usr/local/zed/get_python_api.py`
4. **Install ffmpeg**: Windows `winget install ffmpeg`; Ubuntu `sudo apt install ffmpeg`.
5. **Get our two scripts**: download `scripts/zed_kit.py` and `scripts/zed_export.py` from the repo into one folder
   (or `git clone https://github.com/allenbuild/duetv0` and work in `duetv0/scripts`).
6. `pip install numpy opencv-python`
7. Plug the ZED Mini into a **USB 3** port (blue, or USB-C). Run:
   ```bash
   python zed_kit.py check
   ```
   It must end with `READY` and list the camera. If it says no camera, try another port; the Mini needs USB 3 bandwidth.

## Every recording session

```bash
python zed_kit.py all dorm_2026-09-27_take1 --seconds 600
```

That records for up to 10 minutes at 720p 30 fps (press Enter to stop earlier), then exports and packs. Clap twice
right after it prints RECORDING, and show the ChArUco board for 10 seconds. When it finishes you have:

```
zed_captures/dorm_2026-09-27_take1/
    capture.svo2                 the raw ZED recording, keep it
    export/left.mp4              left-eye video, the "ego" stream
    export/pose.csv              ZED head tracking, metres
    export/imu.csv               accelerometer + gyro
    export/depth/000000.png ...  depth at 10 fps, millimetres
    export/calibration.json      the camera's factory intrinsics
    dorm_2026-09-27_take1_zed.zip      <-- upload this
```

Export runs after recording, not during, so the GPU is free while people are being recorded. Depth export is the
slow part (roughly real time on an RTX 5060 with `neural_light`); pass `--no-depth` to skip it and get the zip in
a minute, depth can be recomputed later from the SVO.

Steps can also be run one at a time: `record NAME`, then `export NAME`, then `pack NAME`.

## Getting the zip into the playground

- **Upload page**: choose the zip as a file, set its role to `ego`, name the person. Other cameras (body cams,
  fixed cams) go in the same upload as normal MP4s with their roles. The server unpacks the zip to
  `<episode>/zed/<stream>/` and uses `left.mp4` as the video.
- **Command line** (whoever runs the pipeline):
  ```bash
  python scripts/playground.py create dorm_take1 --zed leader=dorm_2026-09-27_take1_zed.zip:Idhant --exo exoA=exoA.mp4 --exo exoB=exoB.mp4
  python scripts/playground.py run dorm_take1
  ```

What the ZED adds over a plain video: `calib` takes the factory intrinsics instead of guessing, `headpose` uses the
ZED's own tracking (registered to the board frame if the board was shown), and `world3d` reads metric depth at the
hand landmarks. No ZED SDK is needed on the machine that runs the pipeline.

## If something fails

| message | fix |
|---|---|
| `pyzed not installed` | step 3 above; the SDK must be installed first |
| `camera open failed: CAMERA NOT DETECTED` | USB 3 port, original cable, unplug/replug |
| `recording failed ... NVENC` | H.264 recording needs an NVIDIA GPU with a hardware encoder; every RTX card has one. Check the SDK is using the NVIDIA GPU, not integrated graphics |
| export errors on an attribute name | `zed_export.py` was written against SDK 4.x/5.x; send us the error line and the SDK version from `check` |
| zip rejected by the upload page | the zip must contain `left.mp4` and `calibration.json` at its top level; use `pack`, do not re-zip the folder by hand |
