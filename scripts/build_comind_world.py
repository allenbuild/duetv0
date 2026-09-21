#!/usr/bin/env python3
"""Build cross-person shared-world feature caches (world_v0.npz) where Multi-SLAM trajectories exist."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from duet.adapters.comind.shared_world_features import build_recording_world

raw, proc = Path("data/raw/comind"), Path("data/processed/comind")
ok, skipped = 0, []
for kin in sorted(proc.glob("*/kinematics_v0.npz")):
    rid = kin.parent.name
    out = kin.parent / "world_v0.npz"
    if out.exists() and "--force" not in sys.argv:
        ok += 1; continue
    t0 = time.time()
    try:
        f, st = build_recording_world(raw, proc, rid, np.load(kin)["features"])
    except FileNotFoundError as e:
        skipped.append(rid[:8]); continue
    np.savez_compressed(out, world=f); json.dump(st, open(kin.parent / "world_v0.json", "w"), indent=1); ok += 1
    print(f"{rid[:8]} shared_graph={st['shared_graph']} both_valid={st['both_valid']:.3f} head_dist={st['median_head_dist_m']} min_wrist={st['median_min_wrist_dist_m']} ({time.time()-t0:.0f}s)", flush=True)
print("built", ok, "skipped", len(skipped))
