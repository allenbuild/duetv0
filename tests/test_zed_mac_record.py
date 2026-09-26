"""scripts/zed_mac_record.py without a ZED: factory-conf parsing (format as in Stereolabs' zed-open-capture
calibration.hpp), the side-by-side split, calibration.json/export.json, pack, and the no-camera failure."""
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.playground import stereo_depth as SD, zed_bundle  # noqa: E402

SCRIPT = ROOT / "scripts" / "zed_mac_record.py"
spec = importlib.util.spec_from_file_location("zed_mac_record", SCRIPT); zmr = importlib.util.module_from_spec(spec); spec.loader.exec_module(zmr)

# Shaped like a real SN<serial>.conf (sections per resolution, [STEREO] with per-resolution extrinsics). Values are
# plausible ZED Mini numbers, not a real camera's.
SAMPLE_CONF = """[LEFT_CAM_2K]
fx=1399.51
fy=1399.51
cx=1118.23
cy=633.71
k1=-0.0419
k2=0.0127
p1=0.0001
p2=-0.0002
k3=-0.003

[LEFT_CAM_HD]
fx=699.755
fy=699.755
cx=640.35
cy=356.42
k1=-0.0419
k2=0.0127
p1=0.0001
p2=-0.0002
k3=-0.003

[RIGHT_CAM_HD]
fx=700.12
fy=700.12
cx=637.90
cy=361.10
k1=-0.0402
k2=0.0110
p1=0.0002
p2=-0.0001
k3=-0.0028

[STEREO]
Baseline=62.9962
TY_HD=0.0197
TZ_HD=-0.0368
CV_HD=0.00389
RX_HD=0.00311
RZ_HD=-0.000474
CV_2K=0.00391
RX_2K=0.00309
RZ_2K=-0.000471

[MISC]
Sensor_ID=0
"""


def test_parse_conf_and_stereo_params():
    conf = zmr.parse_conf(SAMPLE_CONF)
    assert {"left_cam_2k", "left_cam_hd", "right_cam_hd", "stereo", "misc"} <= set(conf)
    assert conf["left_cam_hd"]["fx"] == 699.755 and conf["stereo"]["baseline"] == 62.9962
    sp = zmr.stereo_params(conf, "HD")
    assert sp["K_left"] == [[699.755, 0.0, 640.35], [0.0, 699.755, 356.42], [0.0, 0.0, 1.0]]
    assert sp["dist_left"] == [-0.0419, 0.0127, 0.0001, -0.0002, -0.003]
    assert sp["K_right"][0][0] == 700.12 and sp["dist_right"][0] == -0.0402
    assert abs(sp["baseline_m"] - 0.0629962) < 1e-9
    assert np.allclose(sp["T_m"], [-0.0629962, 0.0197e-3, -0.0368e-3])
    assert sp["rodrigues_rx_cv_rz"] == [0.00311, 0.00389, -0.000474]
    import cv2
    R_expected, _ = cv2.Rodrigues(np.array([0.00311, 0.00389, -0.000474]))
    assert np.allclose(sp["R"], R_expected)
    with pytest.raises(ValueError):  # no RIGHT_CAM_2K section in the sample
        zmr.stereo_params(conf, "2K")
    assert zmr.RES_KEY_BY_EYE_WIDTH == {2208: "2K", 1920: "FHD", 1280: "HD", 672: "VGA"}
    assert zmr.MODES["720"] == (2560, 720, 30, "HD") and zmr.MODES["2k"][:2] == (4416, 1242)
    # decimal-comma files (the C++ reader has a "safety check" for these) still parse
    assert zmr.parse_conf("[STEREO]\nBaseline=62,9962\n")["stereo"]["baseline"] == 62.9962


def test_calibration_json_scaling_and_shape():
    sp = zmr.stereo_params(zmr.parse_conf(SAMPLE_CONF), "HD")
    c = zmr.calibration_json(sp, 320, 180, 30.0, 12345, "ZED-M")
    for key in ("K", "dist", "width", "height", "fps", "serial", "model", "baseline_m"):  # what zed_export.py writes
        assert key in c
    assert c["width"] == 320 and c["height"] == 180 and c["K"][0][0] == pytest.approx(699.755 / 4) and c["K"][0][2] == pytest.approx(640.35 / 4)
    assert c["K"][2] == [0.0, 0.0, 1.0] and len(c["dist"]) == 5
    assert SD.has_stereo_block(c) and c["stereo"]["right"]["K"][0][0] == pytest.approx(700.12 / 4) and c["stereo"]["T"][0] == pytest.approx(-0.0629962)
    # full-size: K unchanged
    assert zmr.calibration_json(sp, 1280, 720, 30.0, 1, "ZED-M")["K"][0][0] == 699.755


def _make_sbs(path: Path, w=640, h=180, fps=30, seconds=1.0):
    """Side-by-side test video: testsrc on the left half, smptebars on the right half, a 440 Hz tone as audio."""
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", f"testsrc=size={w // 2}x{h}:rate={fps}", "-f", "lavfi", "-i", f"smptebars=size={w // 2}x{h}:rate={fps}",
                    "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000", "-filter_complex", "[0:v][1:v]hstack[v]", "-map", "[v]", "-map", "2:a",
                    "-t", str(seconds), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(path)], check=True)


def _first_frame(path: Path):
    import cv2
    cap = cv2.VideoCapture(str(path)); ok, f = cap.read(); cap.release(); assert ok; return f


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg needed")
def test_export_and_pack(tmp_path):
    cap = tmp_path / "zed_captures" / "t1"; cap.mkdir(parents=True); _make_sbs(cap / "sbs.mp4")
    json.dump({"serial": 12345, "device": "ZED-M", "start_utc": "2026-09-26T10:00:00+00:00", "resolution_key": "HD"}, open(cap / "record.json", "w"))
    conf = tmp_path / "SN12345.conf"; conf.write_text(SAMPLE_CONF)
    r = subprocess.run([sys.executable, str(SCRIPT), "export", "t1", "--calib", str(conf), "--res-key", "HD"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    exp = cap / "export"
    for f in ("left.mp4", "right.mp4", "calibration.json", "export.json", "SN12345.conf"):
        assert (exp / f).exists(), f
    li, ri = zmr._ffprobe(exp / "left.mp4"), zmr._ffprobe(exp / "right.mp4")
    assert (li["width"], li["height"]) == (320, 180) and (ri["width"], ri["height"]) == (320, 180)
    assert li["has_audio"] and not ri["has_audio"] and abs(li["fps"] - 30) < 0.01
    # the halves went to the right files
    sbs = _first_frame(cap / "sbs.mp4"); l, rr = _first_frame(exp / "left.mp4"), _first_frame(exp / "right.mp4")
    assert np.abs(sbs[:, :320].astype(float) - l).mean() < 12 and np.abs(sbs[:, 320:].astype(float) - rr).mean() < 12
    assert np.abs(sbs[:, 320:].astype(float) - l).mean() > 30  # and not swapped
    c = json.load(open(exp / "calibration.json"))
    assert c["serial"] == 12345 and c["model"] == "ZED-M" and c["width"] == 320 and c["resolution_key"] == "HD" and SD.has_stereo_block(c)
    assert c["K"][0][0] == pytest.approx(699.755 / 4)
    e = json.load(open(exp / "export.json"))
    assert e["source"] == "uvc_mac" and abs(e["fps"] - 30) < 0.01 and e["depth_mode"] == "none" and e["record"]["start_utc"].startswith("2026-09-26")
    assert zed_bundle.is_bundle(exp)
    # rectification maps build from this calibration at the exported size
    rect = SD.rectification_from_calibration(c, 320, 180); assert rect.map_left[0].shape == (180, 320) and abs(rect.baseline_m - 0.063) < 0.001
    r = subprocess.run([sys.executable, str(SCRIPT), "pack", "t1"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = cap / "t1_zed.zip"; assert z.exists() and zed_bundle.is_bundle(z)
    assert {"left.mp4", "right.mp4", "calibration.json", "export.json"} <= set(zipfile.ZipFile(z).namelist())
    assert "pose.csv" not in zipfile.ZipFile(z).namelist()  # no SDK tracking on the Mac path
    left = zed_bundle.ingest(z, tmp_path / "ep", "leader"); assert left.exists() and (tmp_path / "ep" / "zed" / "leader" / "right.mp4").exists()


@pytest.mark.skipif(platform.system() != "Darwin" or shutil.which("ffmpeg") is None, reason="avfoundation is macOS only")
def test_list_and_record_without_zed(tmp_path):
    video, audio = zmr.list_devices()
    if zmr.find_zed(video):
        pytest.skip("a ZED is connected; the no-camera failure path cannot be tested")
    assert zmr.find_zed([(0, "ZED-M"), (1, "FaceTime HD Camera")]) == [(0, "ZED-M")]
    r = subprocess.run([sys.executable, str(SCRIPT), "list"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0 and "video devices" in r.stdout and "no ZED found" in r.stdout
    r = subprocess.run([sys.executable, str(SCRIPT), "record", "x", "--serial", "1", "--seconds", "1"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode != 0 and "no ZED found" in r.stderr, r.stderr
    assert not (tmp_path / "zed_captures" / "x" / "sbs.mp4").exists()
