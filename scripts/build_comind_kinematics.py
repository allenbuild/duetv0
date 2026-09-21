#!/usr/bin/env python3
"""Build per-recording kinematic feature/label caches for all CoMind recordings.

Writes data/processed/comind/<id>/kinematics_v0.npz with:
  features [N, 118] float32 (leader block 0:59, helper block 59:118)
  <label arrays> and a JSON sidecar with coverage stats and handover events.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from duet.adapters.comind.kinematics import build_recording_kinematics
from duet.adapters.comind.labels import build_frame_labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root", default="data/raw/comind")
    ap.add_argument("--proc-root", default="data/processed/comind")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    raw, proc = Path(a.raw_root), Path(a.proc_root)
    ids = sorted(p.name for p in (raw / "recordings").iterdir() if p.is_dir())
    ok, skipped = [], []
    for rid in ids:
        out = proc / rid / "kinematics_v0.npz"
        if out.exists() and not a.force:
            ok.append(rid)
            continue
        try:
            k = build_recording_kinematics(raw, proc, rid)
        except (FileNotFoundError, ValueError) as e:
            skipped.append((rid[:8], str(e).split("/")[-1][:60]))
            continue
        n = k["n_frames"]
        lab = build_frame_labels(raw / "annotations", rid, n)
        np.savez_compressed(out, features=k["features"], **lab["labels"])
        meta = {
            "recording_id": rid,
            "n_frames": n,
            "stats": k["stats"],
            "handovers": [dataclasses.asdict(e) for e in lab["handovers"]],
            "n_joint_attention": lab["n_joint_attention"],
            "prevalence": {key: float(v[np.isfinite(v)].mean()) if v.size else 0.0 for key, v in lab["labels"].items() if key not in ("tth_s", "tte_s")},
        }
        json.dump(meta, open(proc / rid / "kinematics_v0.json", "w"), indent=1)
        ok.append(rid)
        print(f"{rid[:8]} N={n:6d} handovers={len(lab['handovers']):3d} JA={lab['n_joint_attention']:4d} "
              f"ho_active={meta['prevalence']['handover_active']:.3f} ja={meta['prevalence']['ja_active']:.3f}", flush=True)
    print(f"built {len(ok)}, skipped {len(skipped)}: {skipped}")


if __name__ == "__main__":
    main()
