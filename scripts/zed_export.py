#!/usr/bin/env python3
"""Export a ZED SVO/SVO2 recording into playground inputs (export format 2). Run on the NVIDIA machine with the ZED SDK.

    python zed_export.py capture.svo2 out_dir/ [--depth neural] [--fps 30] [--no-depth] [--no-tracking] [--gpu 0] [--overwrite]

out_dir becomes <episode>/zed/<stream>/ (the stream's video is out_dir/left.mp4). Writes:
  left.mp4          left RECTIFIED image, H.264, constant frame rate = the camera fps, on the CAPTURE timeline: the grab
                    captured at t is video frame round((t - t0) * fps) (t0 = first grab). Slots left empty by dropped grabs
                    repeat the previous image; a grab landing in an already written slot (clock jitter) is not written
                    (video_frame -1). Video time therefore stays within half a frame of capture time, and after a drop the
                    video frame index is NOT the grab index: always go through pose.csv's video_frame.
  pose.csv          one row per successful grab: frame_index (grab count, 0-based), timestamp_ns (image capture time, SDK
                    clock), video_frame, tx, ty, tz (metres), qx, qy, qz, qw, tracking_state (SDK enum name; only "OK" is a
                    valid pose). Pose = T_zedworld_camera with COORDINATE_SYSTEM.IMAGE: camera axes x right, y down, z
                    forward (the OpenCV convention the playground uses); the ZED world is the tracking origin.
  imu.csv           frame_index, timestamp_ns (IMU sample time), ax, ay, az (m/s^2), gx, gy, gz (deg/s): ONE sample per grab
                    (the sample the SDK associates with the image, TIME_REFERENCE.IMAGE), not the native IMU rate. Axes as
                    the SDK reports them for the IMAGE coordinate system (not verified here).
  depth/<frame_index:06d>.png  16-bit depth along the optical axis (Z, not range) in millimetres, 0 = invalid, for every
                    depth_every-th grab (--fps sets the depth export rate).
  calibration.json  rectified left camera: K, dist, width, height, fps, serial, model, baseline_m.
  export.json       format 2: coordinate_system "IMAGE", coordinate_units "METER", video_timeline "capture", fps,
                    depth_every_n_frames, depth_units "mm", counts (grabs, video frames, filled slots, grabs not in the
                    video), grab-error and tracking-state histograms, SDK version, complete (false if grabbing kept failing),
                    and video = {frames, fps, duration_s, size, sha256_head_tail} of left.mp4: the playground refuses the
                    export for a stream whose video does not match (e.g. zed/<stream> left over from another take).
out_dir must be empty (or absent); --overwrite first deletes an earlier export there (only the files this script writes;
anything else makes it refuse).
The playground (world.headpose) refuses exports without a known coordinate_system; format 1 exports (this script before
the fix: RIGHT_HANDED_Y_UP, video written by grab count) are still read, converted to OpenCV camera axes, and frames whose
video time drifted from capture time (after dropped grabs) are rejected.

ZED video has no audio, so the playground cannot align it by sound: give the stream a manual offset.
Positional tracking needs depth: --no-depth only skips writing depth PNGs (depth is still computed for tracking);
--depth none requires --no-tracking. Playback is offline (no real-time pacing). Tracking is one continuous pass: do not cut
the SVO before exporting. Not runnable on macOS (the SDK needs CUDA). Written against the ZED SDK 4.x/5.x Python API; the
calls whose names changed between versions are resolved defensively (see _attr) and every SDK return code is checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

EXPORT_FORMAT = 2
EXPORT_FILES = {"left.mp4", "pose.csv", "imu.csv", "calibration.json", "export.json", "depth"}  # what this script writes


def _name(x) -> str:
    """SDK enum -> its name ("OK", "SUCCESS", ...)."""
    return str(getattr(x, "name", x))


def _attr(obj, *names):
    """First attribute of obj that exists among names (SDK spellings differ between versions), else None."""
    return next((getattr(obj, n) for n in names if hasattr(obj, n)), None)


def _vec(x) -> list[float]:
    """sl.Translation / sl.Orientation / numpy -> list of floats."""
    x = x.get() if hasattr(x, "get") else x
    return [float(v) for v in np.asarray(x, float).ravel()]


def _file_identity(p: Path) -> dict:
    """size and sha256 of the first + last MiB (world._file_identity computes the same for the stream file)."""
    st = p.stat(); h = hashlib.sha256(str(st.st_size).encode())
    with open(p, "rb") as f:
        h.update(f.read(1 << 20))
        if st.st_size > 2 << 20:
            f.seek(-(1 << 20), os.SEEK_END); h.update(f.read(1 << 20))
    return {"size": st.st_size, "sha256_head_tail": h.hexdigest()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("svo"); ap.add_argument("out")
    ap.add_argument("--fps", type=float, default=30.0, help="depth export rate in Hz: every round(camera fps / FPS)-th grab (default 30)")
    ap.add_argument("--depth", default="neural", choices=["neural", "neural_light", "neural_plus", "ultra", "quality", "performance", "none"],
                    help="depth mode used for tracking and for the exported depth")
    ap.add_argument("--no-depth", action="store_true", help="do not write depth PNGs (depth is still computed for tracking)")
    ap.add_argument("--no-tracking", action="store_true", help="skip positional tracking (pose.csv rows get tracking_state OFF)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--max-grab-failures", type=int, default=100, help="consecutive failed grabs before giving up")
    ap.add_argument("--overwrite", action="store_true", help="delete out_dir's previous contents (an older export) first")
    a = ap.parse_args(argv)
    if a.depth == "none" and not a.no_tracking:
        ap.error("positional tracking needs depth: choose a --depth mode or pass --no-tracking")
    import cv2
    import pyzed.sl as sl

    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        if not a.overwrite:
            print(f"{out} is not empty (an earlier export?): pass --overwrite to replace it", file=sys.stderr); return 2
        other = sorted(p.name for p in out.iterdir() if p.name not in EXPORT_FILES)
        if other:  # never delete what this script did not write
            print(f"{out} holds files that are not a ZED export ({other[:5]}): refusing to overwrite it", file=sys.stderr); return 2
        for p in out.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    export_depth = not a.no_depth and a.depth != "none"
    if export_depth:
        (out / "depth").mkdir(exist_ok=True)
    ok_code = sl.ERROR_CODE.SUCCESS
    eof_codes = [c for c in (_attr(sl.ERROR_CODE, "END_OF_SVOFILE_REACHED"), _attr(sl.ERROR_CODE, "END_OF_SVO_FILE_REACHED")) if c is not None]
    init = sl.InitParameters()
    init.set_from_svo_file(str(a.svo))
    init.svo_real_time_mode = False  # process as fast as possible, never pace to wall clock
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE  # camera x right, y down, z forward = OpenCV
    init.sdk_gpu_id = a.gpu
    if not hasattr(sl.DEPTH_MODE, a.depth.upper()):
        print(f"this ZED SDK has no DEPTH_MODE.{a.depth.upper()}", file=sys.stderr); return 2
    init.depth_mode = getattr(sl.DEPTH_MODE, a.depth.upper())
    cam = sl.Camera()
    err = cam.open(init)
    if err != ok_code:
        print(f"open failed: {_name(err)}", file=sys.stderr); return 2
    info = cam.get_camera_information(); conf = info.camera_configuration
    left = conf.calibration_parameters.left_cam; res = conf.resolution; fps = float(conf.fps)
    if not fps > 0:
        print(f"camera reports fps {fps}", file=sys.stderr); cam.close(); return 2
    baseline = _attr(conf.calibration_parameters, "get_camera_baseline")
    calib = {"K": [[left.fx, 0, left.cx], [0, left.fy, left.cy], [0, 0, 1]], "dist": _vec(left.disto), "width": int(res.width), "height": int(res.height),
             "fps": fps, "rectified": True, "serial": int(info.serial_number), "model": _name(info.camera_model),
             "baseline_m": float(baseline()) if callable(baseline) else None}
    (out / "calibration.json").write_text(json.dumps(calib, indent=1))
    if not a.no_tracking:
        tp = sl.PositionalTrackingParameters(); tp.enable_area_memory = True; tp.set_floor_as_origin = False
        err = cam.enable_positional_tracking(tp)
        if err != ok_code:
            print(f"enable_positional_tracking failed: {_name(err)}", file=sys.stderr); cam.close(); return 2
    n_total = int(cam.get_svo_number_of_frames())
    ff = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgra", "-s", f"{res.width}x{res.height}",
                           "-framerate", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
                           str(out / "left.mp4")], stdin=subprocess.PIPE)
    img, depth, pose, sensors, rt = sl.Mat(), sl.Mat(), sl.Pose(), sl.SensorsData(), sl.RuntimeParameters()
    rt.enable_depth = a.depth != "none"
    depth_every = max(1, int(round(fps / a.fps))) if a.fps > 0 else 1
    pose_rows, imu_rows = [], []
    states, errors = Counter(), Counter()
    g = 0; fails = 0; t0 = None; last_ts = None; last_slot = -1; prev = None; filled = 0; unwritten = 0; complete = True
    try:
        while True:
            e = cam.grab(rt)
            if e in eof_codes:
                break
            if e != ok_code:
                errors[_name(e)] += 1; fails += 1
                pos = _attr(cam, "get_svo_position")
                if callable(pos) and n_total > 0 and pos() >= n_total - 1:
                    break  # an error on the last frame ends the file
                if fails >= a.max_grab_failures:
                    complete = False; print(f"giving up after {fails} consecutive failed grabs ({_name(e)})", file=sys.stderr); break
                continue
            fails = 0
            ts = int(cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds())
            if last_ts is not None and ts <= last_ts:
                errors["NON_MONOTONIC_TIMESTAMP"] += 1; continue
            if cam.retrieve_image(img, sl.VIEW.LEFT) != ok_code:
                errors["RETRIEVE_IMAGE_FAILED"] += 1; continue
            frame = np.ascontiguousarray(img.get_data()[..., :4]).tobytes()  # BGRA; copy (the Mat is reused)
            t0 = ts if t0 is None else t0; last_ts = ts
            slot = int(round((ts - t0) * fps / 1e9))
            if slot <= last_slot:
                vf = -1; unwritten += 1  # capture-clock jitter: this grab's slot is taken
            else:
                for _ in range(slot - last_slot - 1):  # dropped grabs: hold the previous image
                    ff.stdin.write(prev); filled += 1
                ff.stdin.write(frame); vf = last_slot = slot; prev = frame
            if a.no_tracking:
                st, t, q = "OFF", [np.nan] * 3, [np.nan] * 4
            else:
                st = _name(cam.get_position(pose, sl.REFERENCE_FRAME.WORLD))
                t, q = _vec(pose.get_translation(sl.Translation())), _vec(pose.get_orientation(sl.Orientation()))
            states[st] += 1; pose_rows.append((g, ts, vf, *t, *q, st))
            if cam.get_sensors_data(sensors, sl.TIME_REFERENCE.IMAGE) == ok_code:
                imu = sensors.get_imu_data()
                imu_rows.append((g, int(imu.timestamp.get_nanoseconds()), *_vec(imu.get_linear_acceleration()), *_vec(imu.get_angular_velocity())))
            if export_depth and g % depth_every == 0:
                if cam.retrieve_measure(depth, sl.MEASURE.DEPTH) == ok_code:
                    d = np.asarray(depth.get_data(), np.float32); d = np.where(np.isfinite(d) & (d > 0), d, 0.0)
                    cv2.imwrite(str(out / "depth" / f"{g:06d}.png"), np.clip(np.round(d * 1000.0), 0, 65535).astype(np.uint16))
                else:
                    errors["RETRIEVE_DEPTH_FAILED"] += 1
            g += 1
            if g % 300 == 0:
                print(f"  {g}/{n_total}", flush=True)
    finally:
        ff.stdin.close(); rc = ff.wait(); cam.close()
    if rc != 0:
        print(f"ffmpeg exited with {rc}", file=sys.stderr); return 3
    with open(out / "pose.csv", "w") as f:
        f.write("frame_index,timestamp_ns,video_frame,tx,ty,tz,qx,qy,qz,qw,tracking_state\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in pose_rows)
    with open(out / "imu.csv", "w") as f:
        f.write("frame_index,timestamp_ns,ax,ay,az,gx,gy,gz\n")
        f.writelines(",".join(str(v) for v in r) + "\n" for r in imu_rows)
    video = {"frames": last_slot + 1, "fps": fps, "duration_s": (last_slot + 1) / fps, **_file_identity(out / "left.mp4")}
    (out / "export.json").write_text(json.dumps({
        "format": EXPORT_FORMAT, "svo": Path(a.svo).name, "complete": complete, "fps": fps, "grabs": g, "video_frames": last_slot + 1, "video": video,
        "filled_slots": filled, "grabs_not_in_video": unwritten, "svo_frames": n_total, "coordinate_system": "IMAGE", "coordinate_units": "METER",
        "pose": "T_zedworld_camera; camera axes x right, y down, z forward (OpenCV); ZED world = tracking origin",
        "video_timeline": "capture", "depth_mode": a.depth, "depth_exported": export_depth, "depth_every_n_frames": depth_every, "depth_units": "mm",
        "imu": {"rate": "one sample per grab (TIME_REFERENCE.IMAGE)", "accel_units": "m/s^2", "gyro_units": "deg/s"},
        "tracking": not a.no_tracking, "tracking_states": dict(states), "grab_errors": dict(errors), "sdk": str(sl.Camera.get_sdk_version())}, indent=1))
    print(f"done {out}: {g} grabs, {last_slot + 1} video frames ({filled} filled, {unwritten} grabs not written), tracking {dict(states)}")
    return 0 if complete else 1


if __name__ == "__main__":
    sys.exit(main())
