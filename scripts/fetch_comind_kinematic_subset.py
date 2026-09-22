#!/usr/bin/env python3
"""Fetch the kinematic subset of CoMind: hands, gaze, transcripts and MP4 tails.

This is the entry point of the paired-benchmark pipeline. ``build_comind_kinematics.py``
reads ``data/processed/comind/<id>/mp4_tail_<role>.bin``; this script is what produces it.

Per recording it transfers roughly 67 MB instead of the full 40 GB:

  mps_<role>_trimmed_vrs/hand_tracking/hand_tracking_results.csv   ~30 MB each
  mps_<role>_trimmed_vrs/eye_gaze/general_eye_gaze.csv             ~2 MB each
  transcripts/<role>_trimmed_sync_transcript.json                  ~0.2 MB each
  last TAIL_BYTES of mp4s/<role>_trimmed_sync.mp4                  2 MB each (HTTP range)

The MP4 tail is fetched rather than the whole 1 GB video because the only thing the
kinematics adapter needs from the video is its ``moov`` atom: the frame-0 DEVICE_TIME in
``desc`` and the frame count in ``stsz``. CoMind writes ``moov`` at the end of the file,
so a suffix range request suffices. ``parse_mp4_tail`` raises if the atom is not complete
in the fetched bytes, so a too-small tail fails loudly rather than silently.

Whole files are verified against the official manifest size and hash. Tails cannot be
(the hash covers the whole file), so they are validated by parsing: a tail that does not
yield an anchor is discarded and reported.

Raw dataset files are written under ``data/raw/comind/recordings/<id>/`` and are never
modified afterwards; tails are derived artefacts and live under ``data/processed/``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.adapters.comind.kinematics import ROLES, parse_mp4_tail  # noqa: E402

BASE_URL = "https://comind.ethz.ch/dataset"
USER_AGENT = "duet-comind-kinematic-subset/1.0 (+https://comind.ethz.ch)"
MANIFEST_NAME = "healthcheck.json"
TAIL_BYTES = 2_500_000
# Escalating suffix sizes; moov grows with frame count, so the longest recordings need more.
TAIL_LADDER = (TAIL_BYTES, 8_000_000, 24_000_000, 64_000_000)
CHUNK = 1 << 20
# One JSONL record of online_calibration.jsonl (~97 MB file); the first record suffices.
CALIB_HEAD_BYTES = 400_000

ANNOTATION_FILES = (
    "dataset_handover_consolidated.json",
    "dataset_joint_attention_consolidated.json",
    "dataset_scoia_consolidated.json",
)


def role_files(role: str) -> tuple[str, ...]:
    return (
        f"mps_{role}_trimmed_vrs/hand_tracking/hand_tracking_results.csv",
        f"mps_{role}_trimmed_vrs/eye_gaze/general_eye_gaze.csv",
        f"transcripts/{role}_trimmed_sync_transcript.json",
    )


def _request(url: str, headers: dict[str, str] | None = None) -> urllib.request.Request:
    h = {"User-Agent": USER_AGENT, "Accept-Encoding": "identity"}
    h.update(headers or {})
    return urllib.request.Request(url, headers=h)


def fetch_manifest(recording_id: str, timeout: float) -> dict:
    url = f"{BASE_URL}/{recording_id}/{MANIFEST_NAME}"
    with urllib.request.urlopen(_request(url), timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
    return {e["path"]: e for e in data["files"]}


def _verified(dest: Path, entry: dict, algo: str = "sha256") -> bool:
    """True if dest already matches the manifest size and hash."""
    if not dest.exists() or dest.stat().st_size != int(entry["size"]):
        return False
    digest = hashlib.new(algo)
    with open(dest, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest() == entry.get("hash")


def download_file(recording_id: str, relpath: str, entry: dict, target: Path, timeout: float) -> str:
    """Stream one whole file, verifying manifest size and hash before publishing."""
    dest = target / relpath
    if _verified(dest, entry):
        return "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    url = f"{BASE_URL}/{recording_id}/{relpath}"
    size = int(entry["size"])
    digest = hashlib.new("sha256")
    got = 0
    with urllib.request.urlopen(_request(url), timeout=timeout) as r, open(partial, "wb") as out:
        if r.status != 200:
            raise ValueError(f"HTTP {r.status} for {relpath}")
        while True:
            block = r.read(CHUNK)
            if not block:
                break
            got += len(block)
            if got > size:
                raise ValueError(f"{relpath}: response exceeded manifest size")
            out.write(block)
            digest.update(block)
    if got != size:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{relpath}: truncated at {got}/{size} bytes")
    if entry.get("hash") and digest.hexdigest() != entry["hash"]:
        partial.unlink(missing_ok=True)
        raise ValueError(f"{relpath}: sha256 mismatch against manifest")
    partial.replace(dest)
    return "downloaded"


def fetch_mp4_tail(recording_id: str, role: str, entry: dict, proc_dir: Path, timeout: float) -> str:
    """Range-fetch the MP4 suffix holding the moov atom; validate by parsing it.

    The moov atom's size scales with the frame count (stsz carries one entry per sample),
    so a fixed suffix is not enough for the longest recordings: 2.5 MB covers most of
    CoMind but not all. Escalate until the atom parses rather than guessing one size.
    """
    dest = proc_dir / f"mp4_tail_{role}.bin"
    if dest.exists():
        try:
            parse_mp4_tail(dest.read_bytes())
            return "cached"
        except ValueError:
            dest.unlink()
    size = int(entry["size"])
    url = f"{BASE_URL}/{recording_id}/mp4s/{role}_trimmed_sync.mp4"
    last_error = None
    for want in TAIL_LADDER:
        start = max(0, size - want)
        req = _request(url, {"Range": f"bytes={start}-{size - 1}"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if r.status != 206:
                raise ValueError(f"expected HTTP 206 for a range request, got {r.status}")
            tail = r.read()
        if len(tail) != size - start:
            raise ValueError(f"range response was {len(tail)} bytes, expected {size - start}")
        try:
            anchor = parse_mp4_tail(tail)
        except ValueError as e:  # moov/desc/stsz not fully inside this suffix
            last_error = e
            if start == 0:  # already fetched the whole file; a larger range cannot help
                raise
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(tail)
        note = "" if want == TAIL_LADDER[0] else f", needed {want // 1_000_000} MB tail"
        return f"downloaded (T0={anchor.frame0_device_time_ns}, {anchor.n_frames} frames{note})"
    raise ValueError(f"moov not found within {TAIL_LADDER[-1] // 1_000_000} MB tail: {last_error}")


def fetch_rgb_calibration(recording_id: str, role: str, proc_dir: Path, timeout: float) -> str:
    """Save the RGB camera's device-frame extrinsic from the head of online_calibration.jsonl.

    ``online_calibration.jsonl`` is ~97 MB but is one self-contained JSON record per line,
    and ``T_Device_Camera`` is a fixed extrinsic, so the first record is enough: a range
    request for the first CALIB_HEAD_BYTES gives it for ~0.4% of the transfer.

    This exists because the device frame is NOT the viewing direction. Measured on
    recording 43276420, camera-rgb's optical axis sits 38.7 degrees off device +Z, so
    treating +Z as "forward" mis-states where a wearer is looking by that much.
    """
    dest = proc_dir / f"rgb_calib_{role}.json"
    if dest.exists():
        return "cached"
    url = f"{BASE_URL}/{recording_id}/mps_{role}_trimmed_vrs/slam/online_calibration.jsonl"
    req = _request(url, {"Range": f"bytes=0-{CALIB_HEAD_BYTES - 1}"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if r.status != 206:
            raise ValueError(f"expected HTTP 206 for a range request, got {r.status}")
        head = r.read().decode("utf-8", "replace")
    line = head.split("\n", 1)[0]
    rec = json.loads(line)  # raises if the first record is not complete in the head
    rgb = [c for c in rec["CameraCalibrations"] if c["Label"] == "camera-rgb"]
    if not rgb:
        raise ValueError("camera-rgb not present in online calibration")
    T = rgb[0]["T_Device_Camera"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"label": "camera-rgb", "T_Device_Camera": T,
               "source": "first record of mps/slam/online_calibration.jsonl",
               "tracking_timestamp_us": rec.get("tracking_timestamp_us")}, open(dest, "w"), indent=1)
    return "downloaded"


def fetch_annotations(target: Path, timeout: float) -> None:
    manifest = fetch_manifest("annotations", timeout)
    out = target / "annotations"
    for name in ANNOTATION_FILES:
        match = [p for p in manifest if p.endswith(name)]
        if not match:
            print(f"  annotations: {name} absent from manifest", flush=True)
            continue
        relpath = match[0]
        dest = out / name
        if _verified(dest, manifest[relpath]):
            print(f"  annotations/{name}: cached", flush=True)
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{BASE_URL}/annotations/{relpath}"
        with urllib.request.urlopen(_request(url), timeout=timeout) as r:
            dest.write_bytes(r.read())
        print(f"  annotations/{name}: {dest.stat().st_size / 1e6:.1f} MB", flush=True)


def official_ids(which: str) -> list[str]:
    src = (ROOT / "scripts/comind_download.py").read_text()
    block = re.search(which + r"\s*=\s*\[(.*?)\]", src, re.DOTALL).group(1)
    return re.findall(r'"([0-9a-f-]{36})"', block)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recordings", default="train", help="'train', 'test', 'all', or a comma-separated list of UUIDs")
    ap.add_argument("--raw-root", type=Path, default=ROOT / "data/raw/comind")
    ap.add_argument("--proc-root", type=Path, default=ROOT / "data/processed/comind")
    ap.add_argument("--roles", default=",".join(ROLES))
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--limit", type=int, default=None, help="stop after N recordings (for a smoke test)")
    ap.add_argument("--skip-annotations", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="report the transfer size and exit")
    a = ap.parse_args()

    if a.recordings == "train":
        ids = official_ids("TRAIN_IDS")
    elif a.recordings == "test":
        ids = official_ids("TEST_IDS")
    elif a.recordings == "all":
        ids = official_ids("TRAIN_IDS") + official_ids("TEST_IDS")
    else:
        ids = [x.strip() for x in a.recordings.split(",") if x.strip()]
    if a.limit:
        ids = ids[: a.limit]
    roles = [r.strip() for r in a.roles.split(",")]

    if a.dry_run:
        m = fetch_manifest(ids[0], a.timeout)
        per = sum(int(m[p]["size"]) for r in roles for p in role_files(r) if p in m)
        per += TAIL_BYTES * len(roles)
        print(f"{len(ids)} recordings x ~{per / 1e6:.0f} MB = ~{len(ids) * per / 1e9:.1f} GB")
        return 0

    if not a.skip_annotations:
        print("annotations:", flush=True)
        fetch_annotations(a.raw_root, a.timeout)

    t0 = time.time()
    ok, failed, bytes_before = [], [], 0
    for i, rid in enumerate(ids, 1):
        target = a.raw_root / "recordings" / rid
        proc = a.proc_root / rid
        try:
            manifest = fetch_manifest(rid, a.timeout)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            failed.append((rid[:8], f"manifest: {str(e)[:60]}"))
            print(f"[{i}/{len(ids)}] {rid[:8]} manifest FAILED: {str(e)[:70]}", flush=True)
            continue
        notes, problem = [], None
        for role in roles:
            for relpath in role_files(role):
                if relpath not in manifest:
                    problem = f"{relpath.split('/')[-1]} absent"
                    break
                try:
                    download_file(rid, relpath, manifest[relpath], target, a.timeout)
                except (urllib.error.URLError, OSError, ValueError) as e:
                    problem = f"{relpath.split('/')[-1]}: {str(e)[:50]}"
                    break
            if problem:
                break
            mp4 = f"mp4s/{role}_trimmed_sync.mp4"
            if mp4 not in manifest:
                problem = f"{mp4} absent"
                break
            try:
                notes.append(f"{role} tail {fetch_mp4_tail(rid, role, manifest[mp4], proc, a.timeout)}")
            except (urllib.error.URLError, OSError, ValueError) as e:
                problem = f"{role} tail: {str(e)[:50]}"
                break
            # RGB extrinsic: needed to know where the wearer is actually looking (device +Z is
            # not the optical axis). Non-fatal: the kinematic features do not depend on it.
            try:
                fetch_rgb_calibration(rid, role, proc, a.timeout)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                notes.append(f"{role} rgb-calib unavailable ({str(e)[:40]})")
        if problem:
            failed.append((rid[:8], problem))
            print(f"[{i}/{len(ids)}] {rid[:8]} FAILED: {problem}", flush=True)
            continue
        ok.append(rid)
        mb = sum(f.stat().st_size for f in target.rglob("*") if f.is_file()) / 1e6
        el = time.time() - t0
        print(f"[{i}/{len(ids)}] {rid[:8]} ok  {mb:6.1f} MB  {'; '.join(notes)}  "
              f"[{el / 60:.1f} min elapsed, ~{el / i * (len(ids) - i) / 60:.0f} min left]", flush=True)

    print(f"\nfetched {len(ok)}/{len(ids)} recordings in {(time.time() - t0) / 60:.1f} min")
    if failed:
        print(f"failed {len(failed)}:")
        for rid8, why in failed:
            print(f"  {rid8}  {why}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
