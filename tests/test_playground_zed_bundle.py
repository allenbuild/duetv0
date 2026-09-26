"""ZED export bundle: zed_kit pack -> zed_bundle.ingest (folder and zip), and the upload route accepting a zip."""
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.playground import zed_bundle  # noqa: E402


def _fake_export(d: Path):
    d.mkdir(parents=True); (d / "depth").mkdir()
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x180:rate=30", "-t", "1", "-pix_fmt", "yuv420p", str(d / "left.mp4")], check=True)
    json.dump({"K": [[300, 0, 160], [0, 300, 90], [0, 0, 1]], "dist": [0, 0, 0, 0, 0], "width": 320, "height": 180, "fps": 30}, open(d / "calibration.json", "w"))
    (d / "pose.csv").write_text("frame_index,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,tracking_state\n0,0,0,0,0,0,0,0,1,OK\n")
    (d / "depth" / "000000.png").write_bytes(b"")


def test_pack_and_ingest(tmp_path):
    exp = tmp_path / "zed_captures" / "t1" / "export"; _fake_export(exp)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "zed_kit.py"), "pack", "t1"], cwd=tmp_path, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    z = tmp_path / "zed_captures" / "t1" / "t1_zed.zip"; assert z.exists()
    assert {"left.mp4", "calibration.json", "pose.csv"} <= set(zipfile.ZipFile(z).namelist())
    assert zed_bundle.is_bundle(z) and zed_bundle.is_bundle(exp)
    ep = tmp_path / "ep"
    left = zed_bundle.ingest(z, ep, "leader")
    assert left == ep / "zed" / "leader" / "left.mp4" and left.exists() and (ep / "zed" / "leader" / "calibration.json").exists()
    left2 = zed_bundle.ingest(exp, ep, "helper"); assert left2.exists()


def test_ingest_rejects_non_bundle(tmp_path):
    z = tmp_path / "bad.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", "x")
    assert not zed_bundle.is_bundle(z)
    with pytest.raises(ValueError):
        zed_bundle.ingest(z, tmp_path / "ep", "s")
    assert not (tmp_path / "ep" / "zed" / "s").exists()
