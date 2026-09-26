#!/usr/bin/env python3
"""ZED kit for the recording machine (the NVIDIA computer the ZED Mini is plugged into).

One tool, four steps, Windows or Ubuntu:

    python zed_kit.py check                       is the ZED SDK installed, is a camera connected, is ffmpeg there
    python zed_kit.py record NAME [--seconds 600]  record an SVO2 (720p30, H.264 on the GPU); Enter stops early
    python zed_kit.py export NAME                  SVO2 -> left.mp4, pose.csv, imu.csv, depth/, calibration.json
    python zed_kit.py pack NAME                    zip the export into NAME_zed.zip, ready for the playground upload
    python zed_kit.py all NAME [--seconds 600]     record + export + pack

Everything lands in ./zed_captures/NAME/. Upload NAME_zed.zip to the playground as an "ego" file
(or hand it to whoever runs the pipeline). The zip carries the factory calibration and the ZED's own
head tracking, so the playground gets metric depth and camera pose without the ZED SDK on its side.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURES = Path.cwd() / "zed_captures"


def _sl():
    try:
        import pyzed.sl as sl
        return sl
    except ImportError:
        sys.exit("pyzed not installed. Install the ZED SDK, then run its get_python_api.py (see docs/design/zed_friend_guide.md).")


def check(_a):
    ok = True
    print("python", sys.version.split()[0])
    print("ffmpeg", "found" if shutil.which("ffmpeg") else "MISSING (needed for export)"); ok &= shutil.which("ffmpeg") is not None
    try:
        import pyzed.sl as sl
        print("ZED SDK", sl.Camera.get_sdk_version())
        devs = sl.Camera.get_device_list()
        if not devs:
            print("camera: NONE connected (check the USB-C cable is in a USB 3 port)"); ok = False
        for d in devs:
            print(f"camera: {d.camera_model} serial {d.serial_number} state {d.camera_state}")
    except ImportError:
        print("ZED SDK python api: MISSING"); ok = False
    try:
        import numpy, cv2  # noqa
        print("numpy/opencv ok")
    except ImportError as e:
        print("python packages missing:", e, "-> pip install numpy opencv-python"); ok = False
    print("READY" if ok else "NOT READY"); return 0 if ok else 1


def record(a):
    sl = _sl()
    out = CAPTURES / a.name; out.mkdir(parents=True, exist_ok=True); svo = out / "capture.svo2"
    if svo.exists() and not a.force:
        sys.exit(f"{svo} exists; use --force to overwrite or pick another NAME")
    init = sl.InitParameters()
    init.camera_resolution = {"720": sl.RESOLUTION.HD720, "1080": sl.RESOLUTION.HD1080}[a.res]
    init.camera_fps = a.fps; init.depth_mode = sl.DEPTH_MODE.NONE; init.coordinate_units = sl.UNIT.METER
    cam = sl.Camera(); err = cam.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        sys.exit(f"camera open failed: {err}")
    rec = sl.RecordingParameters(str(svo), sl.SVO_COMPRESSION_MODE.H264)
    err = cam.enable_recording(rec)
    if err != sl.ERROR_CODE.SUCCESS:
        cam.close(); sys.exit(f"recording failed: {err} (H.264 needs an NVIDIA GPU with NVENC; RTX cards have it)")
    stop = threading.Event()
    def wait_enter():
        try:
            input(); stop.set()
        except EOFError:
            pass
    threading.Thread(target=wait_enter, daemon=True).start()
    print(f"RECORDING {a.res}p{a.fps} -> {svo}\nClap twice now. Press Enter to stop" + (f" (auto-stop after {a.seconds}s)" if a.seconds else ""))
    rt = sl.RuntimeParameters(); t0 = time.time(); n = 0; dropped = 0
    while not stop.is_set():
        if cam.grab(rt) == sl.ERROR_CODE.SUCCESS:
            n += 1
        else:
            dropped += 1
        if n % (a.fps * 10) == 0 and n:
            print(f"  {int(time.time() - t0)} s, {n} frames, {dropped} grab errors", flush=True)
        if a.seconds and time.time() - t0 >= a.seconds:
            break
    cam.disable_recording(); cam.close()
    dur = time.time() - t0
    (out / "record.txt").write_text(f"frames={n}\ndropped={dropped}\nseconds={dur:.1f}\nres={a.res}\nfps={a.fps}\n")
    print(f"saved {svo} ({svo.stat().st_size / 1e6:.0f} MB, {n} frames, {dur:.0f} s)")


def export(a):
    out = CAPTURES / a.name; svo = out / "capture.svo2"
    if not svo.exists():
        sys.exit(f"no {svo}; run record first")
    exp = out / "export"
    cmd = [sys.executable, str(HERE / "zed_export.py"), str(svo), str(exp), "--depth", a.depth, "--fps", str(a.depth_fps)]
    if a.no_depth:
        cmd.append("--no-depth")
    print(" ".join(cmd)); subprocess.run(cmd, check=True)
    print("exported", exp)


def pack(a):
    out = CAPTURES / a.name; exp = out / "export"
    if not (exp / "left.mp4").exists():
        sys.exit(f"no export in {exp}; run export first")
    z = out / f"{a.name}_zed.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zf:  # mp4/png already compressed; store
        for p in sorted(exp.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(exp))
    print(f"packed {z} ({z.stat().st_size / 1e6:.0f} MB). Upload it to the playground as an 'ego' file.")


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check")
    for name in ("record", "all"):
        p = sub.add_parser(name); p.add_argument("name"); p.add_argument("--seconds", type=int, default=0); p.add_argument("--res", default="720", choices=["720", "1080"])
        p.add_argument("--fps", type=int, default=30); p.add_argument("--force", action="store_true")
        p.add_argument("--depth", default="neural_light"); p.add_argument("--depth-fps", type=float, default=10.0); p.add_argument("--no-depth", action="store_true")
    for name in ("export",):
        p = sub.add_parser(name); p.add_argument("name"); p.add_argument("--depth", default="neural_light", choices=["neural", "neural_light", "neural_plus", "ultra", "none"])
        p.add_argument("--depth-fps", type=float, default=10.0); p.add_argument("--no-depth", action="store_true")
    p = sub.add_parser("pack"); p.add_argument("name")
    a = ap.parse_args()
    if a.cmd == "check":
        sys.exit(check(a))
    if a.cmd in ("record", "all"):
        record(a)
    if a.cmd in ("export", "all"):
        export(a)
    if a.cmd in ("pack", "all"):
        pack(a)


if __name__ == "__main__":
    main()
