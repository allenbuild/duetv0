#!/usr/bin/env python3
"""ZED Mini on a Mac WITHOUT the ZED SDK: record it as a plain UVC camera, produce the same bundle zed_kit.py makes.

The ZED is a UVC device. On macOS it shows up as an avfoundation video device (usually named "ZED-M") that streams
both eyes side by side in one frame: 2560x720 @ 30 (720p), 3840x1080 @ 30 (1080p), 4416x1242 @ 15 (2K),
1344x376 (VGA). No SDK, no CUDA, so: no positional tracking (no pose.csv), no IMU, no depth; the playground's
`stereo_depth` stage computes depth from left/right with OpenCV, and `headpose` falls back to the head tag.

    python zed_mac_record.py list                                   avfoundation devices; marks the ZED if present
    python zed_mac_record.py record NAME --serial SN [--res 720] [--seconds N]
                                                                    ffmpeg capture -> zed_captures/NAME/sbs.mp4 (+ mic
                                                                    audio, wall-clock start in record.json); Enter stops
    python zed_mac_record.py export NAME [--serial SN] [--calib SN123.conf]
                                                                    sbs.mp4 -> export/left.mp4 (with audio), right.mp4,
                                                                    calibration.json (factory, from calib.stereolabs.com),
                                                                    export.json
    python zed_mac_record.py pack NAME                              export/ -> NAME_zed.zip (zed_kit.pack)
    python zed_mac_record.py all NAME --serial SN [--seconds N]     record + export + pack

The serial number is printed on the camera (and on its box); the factory calibration is fetched from
https://calib.stereolabs.com/?SN=<serial> (the same file the SDK downloads, an INI with [LEFT_CAM_HD], [RIGHT_CAM_HD],
[STEREO] ... sections; format as read by Stereolabs' zed-open-capture examples/include/calibration.hpp and
zed-opencv-native/python/zed_opencv_native.py) and cached in zed_captures/calib/SN<serial>.conf.

Everything lands in ./zed_captures/NAME/, like zed_kit.py. Upload NAME_zed.zip to the playground as an "ego" file or
`playground.py create EP --zed leader=NAME_zed.zip:Person`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CAPTURES = Path.cwd() / "zed_captures"
CALIB_URL = "https://calib.stereolabs.com/?SN={serial}"
# res flag -> (side-by-side width, height, fps, Stereolabs resolution key)
MODES = {"720": (2560, 720, 30, "HD"), "1080": (3840, 1080, 30, "FHD"), "2k": (4416, 1242, 15, "2K"), "vga": (1344, 376, 30, "VGA")}
RES_KEY_BY_EYE_WIDTH = {2208: "2K", 1920: "FHD", 1280: "HD", 672: "VGA"}   # calibration.hpp: switch(image_size.width)
NOMINAL_EYE_WIDTH = {v: k for k, v in RES_KEY_BY_EYE_WIDTH.items()}


# ----------------------------------------------------------------------------- devices

def list_devices() -> tuple[list[tuple[int, str]], list[tuple[int, str]]]:
    """(video, audio) avfoundation devices as (index, name), parsed from `ffmpeg -f avfoundation -list_devices true -i ""`."""
    if not shutil.which("ffmpeg"):
        sys.exit("ffmpeg not found; brew install ffmpeg")
    r = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""], capture_output=True, text=True)
    video, audio, section = [], [], None
    for line in (r.stderr + r.stdout).splitlines():
        if "AVFoundation video devices" in line:
            section = video; continue
        if "AVFoundation audio devices" in line:
            section = audio; continue
        m = re.search(r"\[(\d+)\]\s+(.+?)\s*$", line)
        if section is not None and m and "AVFoundation" in line:
            section.append((int(m.group(1)), m.group(2)))
    return video, audio


def find_zed(video: list[tuple[int, str]], name: str | None = None) -> list[tuple[int, str]]:
    key = (name or "zed").lower()
    return [(i, n) for i, n in video if key in n.lower()]


def cmd_list(_a) -> int:
    video, audio = list_devices()
    print("video devices:")
    for i, n in video:
        print(f"  [{i}] {n}" + ("   <-- ZED" if find_zed([(i, n)]) else ""))
    print("audio devices:")
    for i, n in audio:
        print(f"  [{i}] {n}")
    if not find_zed(video):
        print("no ZED found (a connected ZED Mini appears as a video device named like 'ZED-M')")
    return 0


# ----------------------------------------------------------------------------- record

def _ffprobe(path: Path) -> dict:
    j = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,avg_frame_rate,nb_frames", "-of", "json", str(path)],
                                  capture_output=True, text=True, check=True).stdout)
    info = {"duration_s": float(j["format"].get("duration", 0) or 0), "has_audio": False, "fps": 0.0, "width": 0, "height": 0, "frames": None}
    for s in j["streams"]:
        if s["codec_type"] == "video" and not info["width"]:
            num, den = s.get("avg_frame_rate", "30/1").split("/"); info["fps"] = float(num) / float(den or 1)
            info["width"], info["height"] = int(s["width"]), int(s["height"])
            if s.get("nb_frames", "").isdigit():
                info["frames"] = int(s["nb_frames"])
        if s["codec_type"] == "audio":
            info["has_audio"] = True
    return info


def cmd_record(a) -> int:
    video, audio = list_devices()
    zeds = find_zed(video, a.device)
    if not zeds:
        names = ", ".join(f"[{i}] {n}" for i, n in video) or "none"
        sys.exit(f"no ZED found. avfoundation video devices: {names}.\nPlug the ZED Mini straight into a USB 3 / USB-C port (not a hub), wait a few seconds, "
                 f"and check `python {Path(__file__).name} list`. If it shows under another name, pass --device NAME.")
    vidx, vname = zeds[0]
    out = CAPTURES / a.name; out.mkdir(parents=True, exist_ok=True); sbs = out / "sbs.mp4"
    if sbs.exists() and not a.force:
        sys.exit(f"{sbs} exists; use --force to overwrite or pick another NAME")
    W, H, fps, key = MODES[a.res]
    aidx, aname = None, None
    if a.audio != "none" and audio:
        match = [(i, n) for i, n in audio if a.audio and a.audio.lower() in n.lower()] if a.audio else []
        aidx, aname = (match or audio)[0]
    inp = f"{vidx}:{aidx}" if aidx is not None else f"{vidx}"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-y", "-f", "avfoundation", "-framerate", str(fps), "-video_size", f"{W}x{H}",
           "-thread_queue_size", "1024", "-i", inp, "-c:v", a.encoder]
    cmd += ["-b:v", a.bitrate, "-pix_fmt", "yuv420p"] if a.encoder == "h264_videotoolbox" else ["-preset", "ultrafast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if aidx is not None:
        cmd += ["-c:a", "aac", "-b:a", "160k"]
    if a.seconds:
        cmd += ["-t", str(a.seconds)]
    cmd += [str(sbs)]
    start = time.time(); start_utc = dt.datetime.now(dt.timezone.utc)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    print(f"RECORDING {vname} {W}x{H}@{fps} (both eyes side by side) + audio {aname or 'none'} -> {sbs}")
    print("Clap twice now. Press Enter to stop" + (f" (auto-stop after {a.seconds}s)" if a.seconds else ""))
    stop = threading.Event()

    def wait_enter():
        try:
            input(); stop.set()
        except EOFError:
            pass
    threading.Thread(target=wait_enter, daemon=True).start()
    while proc.poll() is None:
        if stop.is_set():
            try:
                proc.stdin.write(b"q"); proc.stdin.flush()   # ffmpeg's graceful stop: finalises the mp4
            except (BrokenPipeError, OSError):
                pass
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate(); proc.wait(timeout=10)
            break
        time.sleep(0.2)
    end = time.time()
    if proc.returncode not in (0, None) and not sbs.exists():
        sys.exit(f"ffmpeg failed (exit {proc.returncode}). If it printed 'Selected video size ... is not supported', the device is not a ZED "
                 f"or is on a USB 2 port; if it printed a permission error, allow camera/microphone access for your terminal in System Settings.")
    info = _ffprobe(sbs)
    rec = {"source": "uvc_mac", "device": vname, "device_index": vidx, "audio_device": aname, "serial": int(a.serial), "res": a.res, "resolution_key": key,
           "sbs_size": [W, H], "eye_size": [W // 2, H], "fps_requested": fps, "start_utc": start_utc.isoformat(), "start_unix": start, "end_unix": end,
           "start_note": "start_unix is when ffmpeg was launched; the first frame is ~0.3-1 s later (device warm-up). Align by audio, not by this.",
           "cmd": cmd, "probe": info}
    json.dump(rec, open(out / "record.json", "w"), indent=1)
    print(f"saved {sbs} ({sbs.stat().st_size / 1e6:.0f} MB, {info['duration_s']:.1f} s, {info['width']}x{info['height']} @ {info['fps']:.2f} fps, audio {info['has_audio']})")
    return 0


# ----------------------------------------------------------------------------- calibration file

def parse_conf(text: str) -> dict[str, dict[str, float]]:
    """Stereolabs SN<serial>.conf (INI) -> {section_lower: {key_lower: float}}. Case-insensitive like SimpleIni."""
    out: dict[str, dict[str, float]] = {}; sec = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line[0] in ";#":
            continue
        if line.startswith("[") and line.endswith("]"):
            sec = line[1:-1].strip().lower(); out.setdefault(sec, {}); continue
        if "=" in line and sec is not None:
            k, v = line.split("=", 1)
            try:
                out[sec][k.strip().lower()] = float(v.strip().replace(",", "."))
            except ValueError:
                pass
    return out


def stereo_params(conf: dict[str, dict[str, float]], res_key: str) -> dict:
    """Left/right intrinsics and the (R, T) for cv2.stereoRectify from a parsed conf, for one resolution key
    ("2K", "FHD", "HD", "VGA"). Mirrors zed-open-capture calibration.hpp::initCalibration:
        K = [[fx,0,cx],[0,fy,cy],[0,0,1]], dist = [k1,k2,p1,p2,k3] from [LEFT_CAM_<res>] / [RIGHT_CAM_<res>]
        R = Rodrigues([RX_<res>, CV_<res>, RZ_<res>]), T = [Baseline, TY_<res>, TZ_<res>] from [STEREO]  (millimetres)
    We store T in metres with T[0] = -Baseline (OpenCV's X_right = R X_left + T with the right eye at +x; this is what
    Stereolabs' python sample passes; the C++ sample passes +Baseline. Rectification is identical either way and depth
    uses |T|)."""
    r = res_key.lower(); st = conf.get("stereo")
    if st is None:
        raise ValueError("conf has no [STEREO] section")
    L, Rr = conf.get(f"left_cam_{r}"), conf.get(f"right_cam_{r}")
    if not L or not Rr:
        raise ValueError(f"conf has no [LEFT_CAM_{res_key}]/[RIGHT_CAM_{res_key}] sections (have {sorted(conf)})")

    def K_dist(c):
        K = [[c.get("fx", 0.0), 0.0, c.get("cx", 0.0)], [0.0, c.get("fy", 0.0), c.get("cy", 0.0)], [0.0, 0.0, 1.0]]
        return K, [c.get("k1", 0.0), c.get("k2", 0.0), c.get("p1", 0.0), c.get("p2", 0.0), c.get("k3", 0.0)]
    KL, dL = K_dist(L); KR, dR = K_dist(Rr)
    if KL[0][0] <= 0 or KR[0][0] <= 0:
        raise ValueError("conf has zero focal length; download failed or wrong serial")
    baseline_mm = st.get("baseline", 0.0)
    if not 20.0 <= baseline_mm <= 300.0:
        raise ValueError(f"[STEREO] Baseline={baseline_mm} is not a plausible ZED baseline in mm")
    ty = st.get(f"ty_{r}", st.get("ty", 0.0)); tz = st.get(f"tz_{r}", st.get("tz", 0.0))
    rod = [st.get(f"rx_{r}", 0.0), st.get(f"cv_{r}", 0.0), st.get(f"rz_{r}", 0.0)]
    import cv2, numpy as np  # local: keep `list`/`record` free of OpenCV
    R, _ = cv2.Rodrigues(np.array(rod, float))
    return {"K_left": KL, "dist_left": dL, "K_right": KR, "dist_right": dR, "R": R.tolist(), "T_m": [-baseline_mm / 1000.0, ty / 1000.0, tz / 1000.0],
            "baseline_m": baseline_mm / 1000.0, "rodrigues_rx_cv_rz": rod, "resolution_key": res_key}


def fetch_conf(serial: int, cache_dir: Path) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True); p = cache_dir / f"SN{int(serial)}.conf"
    if p.exists() and p.stat().st_size > 100:
        return p
    url = CALIB_URL.format(serial=int(serial)); print("downloading", url)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read()
    except Exception as e:  # noqa: BLE001
        sys.exit(f"could not download the factory calibration for serial {serial} from {url}: {e}\n"
                 f"On a machine with internet, save that URL as {p} (or copy ~/zed/settings/SN{serial}.conf from a computer with the SDK) and pass --calib.")
    text = data.decode("utf-8", "replace")
    if "[STEREO]".lower() not in text.lower() or "baseline" not in text.lower():
        sys.exit(f"{url} did not return a calibration file (got {len(data)} bytes starting {text[:80]!r}); check the serial number")
    p.write_bytes(data); return p


def calibration_json(sp: dict, eye_w: int, eye_h: int, fps: float, serial: int, model: str) -> dict:
    """calibration.json compatible with zed_export.py (K, dist, width, height, fps, serial, model, baseline_m) plus the
    stereo block. K is scaled from the conf's nominal width to the actual exported width when they differ."""
    import numpy as np
    sc = eye_w / NOMINAL_EYE_WIDTH[sp["resolution_key"]]
    KL = np.array(sp["K_left"], float); KL[:2] *= sc; KR = np.array(sp["K_right"], float); KR[:2] *= sc
    return {"K": KL.tolist(), "dist": list(sp["dist_left"]), "width": int(eye_w), "height": int(eye_h), "fps": float(fps), "serial": int(serial), "model": model,
            "baseline_m": sp["baseline_m"], "source": "stereolabs_factory_conf", "resolution_key": sp["resolution_key"], "intrinsics_scale_from_conf": sc,
            "stereo": {"right": {"K": KR.tolist(), "dist": list(sp["dist_right"])}, "R": sp["R"], "T": sp["T_m"], "rodrigues_rx_cv_rz": sp["rodrigues_rx_cv_rz"],
                       "units": "m", "T_convention": "cv2.stereoRectify: X_right = R @ X_left + T; T[0] = -Baseline (right eye at +x of the left eye)"}}


# ----------------------------------------------------------------------------- export

def split_sbs(sbs: Path, left: Path, right: Path, encoder: str = "libx264", crf: int = 18) -> None:
    """Left half -> left.mp4 (keeps the audio), right half -> right.mp4 (no audio). One ffmpeg pass, two outputs."""
    vid = ["-c:v", encoder] + (["-b:v", "12M"] if encoder == "h264_videotoolbox" else ["-preset", "fast", "-crf", str(crf)]) + ["-pix_fmt", "yuv420p"]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(sbs),
           "-filter_complex", "[0:v]crop=iw/2:ih:0:0[l];[0:v]crop=iw/2:ih:iw/2:0[r]",
           "-map", "[l]", "-map", "0:a?", *vid, "-c:a", "copy", str(left),
           "-map", "[r]", "-an", *vid, str(right)]
    subprocess.run(cmd, check=True)


def cmd_export(a) -> int:
    out = CAPTURES / a.name; sbs = out / "sbs.mp4"
    if not sbs.exists():
        sys.exit(f"no {sbs}; run record first")
    rec = json.load(open(out / "record.json")) if (out / "record.json").exists() else {}
    serial = a.serial or rec.get("serial")
    if not serial and not a.calib:
        sys.exit("need --serial SN (printed on the camera) to fetch the factory calibration, or --calib SN<serial>.conf")
    info = _ffprobe(sbs); eye_w, eye_h = info["width"] // 2, info["height"]
    key = a.res_key or RES_KEY_BY_EYE_WIDTH.get(eye_w) or rec.get("resolution_key")
    if key is None:
        sys.exit(f"cannot map a {eye_w}-px-wide eye to a Stereolabs resolution key (2208->2K, 1920->FHD, 1280->HD, 672->VGA); pass --res-key")
    exp = out / "export"; exp.mkdir(exist_ok=True)
    conf_path = Path(a.calib) if a.calib else fetch_conf(int(serial), CAPTURES / "calib")
    conf = parse_conf(conf_path.read_text(errors="replace")); sp = stereo_params(conf, key)
    shutil.copy2(conf_path, exp / conf_path.name)
    print(f"splitting {sbs} ({info['width']}x{info['height']}, {info['duration_s']:.1f} s) into left/right {eye_w}x{eye_h}")
    split_sbs(sbs, exp / "left.mp4", exp / "right.mp4", a.encoder)
    left = _ffprobe(exp / "left.mp4")
    model = rec.get("device") or a.model
    cal = calibration_json(sp, eye_w, eye_h, left["fps"], int(serial or 0), model)
    json.dump(cal, open(exp / "calibration.json", "w"), indent=1)
    json.dump({"source": "uvc_mac", "sbs": "sbs.mp4", "frames": left["frames"], "fps": left["fps"], "width": eye_w, "height": eye_h, "duration_s": left["duration_s"],
               "depth_mode": "none", "depth_every_n_frames": None, "tracking_states": {}, "sdk": "none (macOS UVC capture via ffmpeg avfoundation)",
               "coordinate_system": "no pose.csv: no ZED positional tracking; headpose falls back to the head tag; depth from the playground stereo_depth stage",
               "calibration_file": conf_path.name, "resolution_key": key, "serial": int(serial or 0), "model": model,
               "record": {k: rec.get(k) for k in ("start_utc", "start_unix", "end_unix", "device", "audio_device", "fps_requested") if k in rec}},
              open(exp / "export.json", "w"), indent=1)
    print(f"exported {exp}: left.mp4 (audio {left['has_audio']}), right.mp4, calibration.json ({key}, baseline {sp['baseline_m'] * 1000:.1f} mm), export.json")
    return 0


def cmd_pack(a) -> int:
    sys.path.insert(0, str(HERE))
    import zed_kit  # same zip layout as the SDK kit; zed_kit imports pyzed lazily so this is safe without the SDK
    zed_kit.CAPTURES = CAPTURES
    zed_kit.pack(a); return 0


# ----------------------------------------------------------------------------- cli

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    for name in ("record", "all"):
        p = sub.add_parser(name); p.add_argument("name"); p.add_argument("--serial", required=True, type=int, help="ZED serial number (on the camera)")
        p.add_argument("--res", default="720", choices=sorted(MODES)); p.add_argument("--seconds", type=int, default=0); p.add_argument("--force", action="store_true")
        p.add_argument("--device", help="video device name substring (default: 'zed')"); p.add_argument("--audio", help="audio device name substring, or 'none' (default: first device)")
        p.add_argument("--encoder", default="h264_videotoolbox", choices=["h264_videotoolbox", "libx264"]); p.add_argument("--bitrate", default="30M")
    for name in ("export",):
        p = sub.add_parser(name); p.add_argument("name"); p.add_argument("--serial", type=int); p.add_argument("--calib", help="local SN<serial>.conf instead of downloading")
        p.add_argument("--res-key", choices=sorted(NOMINAL_EYE_WIDTH), help="override the calibration resolution section"); p.add_argument("--model", default="ZED-M")
        p.add_argument("--encoder", default="libx264", choices=["libx264", "h264_videotoolbox"])
    sub.add_parser("pack").add_argument("name")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        return cmd_list(a)
    if a.cmd == "pack":
        return cmd_pack(a)
    if a.cmd in ("record", "all"):
        cmd_record(a)
    if a.cmd in ("export", "all"):
        if a.cmd == "all":
            a.calib = None; a.res_key = None; a.model = "ZED-M"; a.encoder = "libx264"
        cmd_export(a)
    if a.cmd == "all":
        cmd_pack(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
