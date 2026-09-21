#!/usr/bin/env python3
"""Build per-recording speech feature caches (speech_v0.npz) next to kinematics_v0.npz."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from duet.adapters.comind.speech import build_recording_speech

raw, proc = Path("data/raw/comind"), Path("data/processed/comind")
n_ok = 0
for kin in sorted(proc.glob("*/kinematics_v0.npz")):
    rid = kin.parent.name
    n = len(np.load(kin)["features"])
    s = build_recording_speech(raw, rid, n)
    np.savez_compressed(kin.parent / "speech_v0.npz", speech=s)
    n_ok += 1
    print(f"{rid[:8]} N={n} leader speaking {s[:, 0].mean():.3f} helper speaking {s[:, 36].mean():.3f} kw {s[:, 3].mean():.3f}", flush=True)
print("built", n_ok)
