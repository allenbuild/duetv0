#!/usr/bin/env python3
"""Export a ZED SVO/SVO2 recording into playground inputs. Run on the NVIDIA machine with the ZED SDK.

    python zed_export.py capture.svo2 out_dir/ [--depth neural] [--fps 30] [--no-depth]

Writes into out_dir/:
  left.mp4              left RGB video (H.264, original fps), the "ego" stream for the playground
  pose.csv              frame_index, timestamp_ns, tx,ty,tz, qx,qy,qz,qw  (T_world_camera from ZED positional tracking, metres)
  imu.csv               timestamp_ns, ax,ay,az, gx,gy,gz (device frame; sensors at native rate)
  depth/000123.png      16-bit depth in millimetres at the left camera, one per exported frame (subsampled to --fps)
  calibration.json      left camera intrinsics (fx, fy, cx, cy, distortion) and resolution, from the ZED factory calibration
  export.json           what was exported, SDK version, settings, tracking state histogram

Playback is offline (no real-time pacing) so this runs as fast as the GPU allows. Tracking is one continuous
pass over the whole file; do not chop the recording into pieces before running this. Frame indices and
ZED timestamps are preserved so the playground can line this up with the other cameras by audio.
Depth is optional (--no-depth) and is the slow part; NEURAL_LIGHT roughly halves its cost.

Not runnable on macOS (the ZED SDK needs CUDA). Written against ZED SDK 4.x/5.x Python API; check
attribute names against your installed version if it errors.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svo"); ap.add_argument("out"); ap.add_argument("--fps", type=float, default=30.0, help="depth export rate (video keeps native fps)")
    ap.add_argument("--depth", default="neural", choices=["neural", "neural_light", "neural_plus", "ultra", "none"]); ap.add_argument("--no-depth", action="store_true")
    ap.add_argument("--gpu", type=int, default=0)
    a = ap.parse_args()
    import pyzed.sl as sl
    out = Path(a.out); (out / "depth").mkdir(parents=True, exist_ok=True)
    init = sl.InitParameters()
    init.set_from_svo_file(a.svo)
    init.svo_real_time_mode = False               # process as fast as possible, never pace to wall clock
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init.sdk_gpu_id = a.gpu
    want_depth = not a.no_depth and a.depth != "none"
    init.depth_mode = {"neural": sl.DEPTH_MODE.NEURAL, "neural_light": sl.DEPTH_MODE.NEURAL_LIGHT, "neural_plus": sl.DEPTH_MODE.NEURAL_PLUS,
                       "ultra": sl.DEPTH_MODE.ULTRA, "none": sl.DEPTH_MODE.NONE}[a.depth if want_depth else "none"]
    cam = sl.Camera()
    err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"open failed: {err}")
    info = cam.get_camera_information()
    calib = info.camera_configuration.calibration_parameters.left_cam
    res = info.camera_configuration.resolution; fps = info.camera_configuration.fps
    json.dump({"K": [[calib.fx, 0, calib.cx], [0, calib.fy, calib.cy], [0, 0, 1]], "dist": list(calib.disto), "width": res.width, "height": res.height,
               "fps": fps, "serial": info.serial_number, "model": str(info.camera_model), "baseline_m": info.camera_configuration.calibration_parameters.get_camera_baseline()},
              open(out / "calibration.json", "w"), indent=1)
    # positional tracking: one continuous pass
    tp = sl.PositionalTrackingParameters(); tp.enable_area_memory = True; tp.set_floor_as_origin = False
    cam.enable_positional_tracking(tp)
    n_total = cam.get_svo_number_of_frames()
    # left video via ffmpeg pipe (raw BGR frames in)
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{res.width}x{res.height}", "-r", str(fps), "-i", "-",
                           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", str(out / "left.mp4")], stdin=subprocess.PIPE)
    img = sl.Mat(); depth = sl.Mat(); pose = sl.Pose(); sensors = sl.SensorsData()
    rt = sl.RuntimeParameters(); rt.enable_depth = want_depth
    pose_rows, imu_rows, states = [], [], {}
    depth_every = max(1, int(round(fps / a.fps))); k = 0
    while True:
        e = cam.grab(rt)
        if e == sl.ERROR_CODE.END_OF_SVO_FILE_REACHED:
            break
        if e != sl.ERROR_CODE.SUCCESS:
            continue
        ts = cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
        cam.retrieve_image(img, sl.VIEW.LEFT); ff.stdin.write(img.get_data().tobytes())
        st = cam.get_position(pose, sl.REFERENCE_FRAME.WORLD); states[str(st)] = states.get(str(st), 0) + 1
        t = pose.get_translation().get(); q = pose.get_orientation().get()
        pose_rows.append((k, ts, *t, *q, str(st)))
        if cam.get_sensors_data(sensors, sl.TIME_REFERENCE.IMAGE) == sl.ERROR_CODE.SUCCESS:
            imu = sensors.get_imu_data(); acc = imu.get_linear_acceleration(); gyr = imu.get_angular_velocity()
            imu_rows.append((sensors.timestamp.get_nanoseconds(), *acc, *gyr))
        if want_depth and k % depth_every == 0:
            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            d = depth.get_data(); d = np.where(np.isfinite(d), d, 0.0); mm = np.clip(d * 1000.0, 0, 65535).astype(np.uint16)
            import cv2; cv2.imwrite(str(out / "depth" / f"{k:06d}.png"), mm)
        k += 1
        if k % 300 == 0:
            print(f"  {k}/{n_total}", flush=True)
    ff.stdin.close(); ff.wait()
    with open(out / "pose.csv", "w") as f:
        f.write("frame_index,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking_state\n")
        for r in pose_rows: f.write(",".join(str(v) for v in r) + "\n")
    with open(out / "imu.csv", "w") as f:
        f.write("timestamp_ns,ax,ay,az,gx,gy,gz\n")
        for r in imu_rows: f.write(",".join(str(v) for v in r) + "\n")
    json.dump({"svo": str(a.svo), "frames": k, "fps": fps, "depth_mode": a.depth if want_depth else "none", "depth_every_n_frames": depth_every,
               "tracking_states": states, "sdk": str(sl.Camera.get_sdk_version()), "coordinate_system": "RIGHT_HANDED_Y_UP, metres, T_world_camera"},
              open(out / "export.json", "w"), indent=1)
    cam.close(); print("done", out)


if __name__ == "__main__":
    main()
