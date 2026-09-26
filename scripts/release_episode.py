#!/usr/bin/env python3
"""Package an episode for release (face-blurred streams, records, schema, manifest, licence, consent template).

    .venv/bin/python scripts/release_episode.py comind_43276420_clip --version v0.1.0 [--out data/playground/releases]
    .venv/bin/python scripts/release_episode.py comind_43276420_clip --version v0.1.0 --verify

Options:
    --no-blur             copy streams WITHOUT face blurring (marked NOT ANONYMISED in the manifest and README)
    --example-png PATH    also write one blurred frame (of --example-stream, at --example-frame) as a PNG for inspection
    --device mps|cpu      device for the pose-based head detector
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.playground import release as R  # noqa: E402
from duet.playground.episode import Episode  # noqa: E402

EPISODES = ROOT / "data/playground/episodes"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name"); ap.add_argument("--version", required=True); ap.add_argument("--out", default=str(R.RELEASES))
    ap.add_argument("--verify", action="store_true", help="re-hash an existing release and report")
    ap.add_argument("--no-blur", action="store_true"); ap.add_argument("--example-png"); ap.add_argument("--example-frame", type=int, default=0)
    ap.add_argument("--example-stream"); ap.add_argument("--device", default="mps")
    a = ap.parse_args()
    out = Path(a.out) / a.version / a.name
    if a.verify:
        if not (out / "manifest.json").exists():
            print(f"no manifest at {out}"); return 2
        rep = R.verify_release(out); print(json.dumps(rep, indent=1)); return 0 if rep["ok"] else 1
    ep_dir = EPISODES / a.name
    if not (ep_dir / "episode.json").exists():
        print(f"no episode {a.name} in {EPISODES}"); return 2
    ep = Episode.load(ep_dir)
    if a.no_blur:
        print("WARNING: --no-blur: streams are copied without anonymisation")
    det = None if a.no_blur else R.FaceDetectors(device=a.device)
    R.release(ep, a.version, Path(a.out), blur=not a.no_blur, detectors=det, example_png=Path(a.example_png) if a.example_png else None,
              example_frame=a.example_frame, example_stream=a.example_stream)
    rep = R.verify_release(out); print("verify:", "OK" if rep["ok"] else rep)
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
