"""CoMind annotations -> per-frame labels on the shared sync-frame index.

Label semantics (paper, Sec. 3): in the merged horizontal video the LEFT view is
the leader and the RIGHT view is the helper. Hence ``initiator == "left"`` means
the leader initiated; ``delivering_flow == "ltr"`` means leader -> helper.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FPS = 30.0
INITIATION_TYPES = ("verbal", "gestural", "implicit")


@dataclass
class HandoverEvent:
    recording_id: str
    key: str
    start_frame: int
    end_frame: int
    initiator_is_leader: bool
    flow_leader_to_helper: bool
    initiation_type: tuple[str, ...]
    object_l1: str
    object_l3: str


def load_handovers(annotations_dir: Path, recording_id: str) -> list[HandoverEvent]:
    d = json.load(open(annotations_dir / "dataset_handover_consolidated.json"))["data"]
    out = []
    for key, v in (d.get(recording_id) or {}).items():
        if v is None or v.get("skip"):
            continue
        out.append(
            HandoverEvent(
                recording_id=recording_id,
                key=key,
                start_frame=int(v["start_frame"]),
                end_frame=int(v["end_frame"]),
                initiator_is_leader=(v["initiator"] == "left"),
                flow_leader_to_helper=(v["delivering_flow"] == "ltr"),
                initiation_type=tuple(v.get("initiation_type") or ()),
                object_l1=str(v.get("object_category_level_1")),
                object_l3=str(v.get("object_category_level_3")),
            )
        )
    out.sort(key=lambda e: e.start_frame)
    return out


def load_joint_attention_intervals(annotations_dir: Path, recording_id: str) -> list[tuple[int, int, tuple[str, ...]]]:
    d = json.load(open(annotations_dir / "dataset_joint_attention_consolidated.json"))["data"]
    v = d.get(recording_id) or {}
    items = v.values() if isinstance(v, dict) else v
    out = []
    for x in items:
        if x is None:
            continue
        out.append((int(x["start_frame"]), int(x["end_frame"]), tuple(x.get("cue_types") or ())))
    out.sort()
    return out


def interval_mask(n: int, intervals, inclusive_end: bool = True) -> np.ndarray:
    m = np.zeros(n, dtype=np.float32)
    for s, e in intervals:
        s = max(0, s)
        e = min(n - 1, e if inclusive_end else e - 1)
        if e >= s:
            m[s : e + 1] = 1.0
    return m


def onset_within(n: int, onsets, horizon_frames: int) -> np.ndarray:
    """1 at frame i if some onset o satisfies i < o <= i + horizon (strict future)."""
    m = np.zeros(n, dtype=np.float32)
    for o in onsets:
        lo = max(0, o - horizon_frames)
        hi = min(n, o)  # frames lo..o-1
        if hi > lo:
            m[lo:hi] = 1.0
    return m


def time_to_next_onset(n: int, onsets, max_s: float) -> np.ndarray:
    """Seconds until the next onset (strictly in the future), clipped to max_s; NaN if none within."""
    out = np.full(n, np.nan, dtype=np.float32)
    onsets = sorted(onsets)
    j = 0
    for i in range(n):
        while j < len(onsets) and onsets[j] <= i:
            j += 1
        if j < len(onsets):
            t = (onsets[j] - i) / FPS
            if t <= max_s:
                out[i] = t
    return out


def build_frame_labels(annotations_dir: Path, recording_id: str, n: int, horizons_s=(1.0, 2.0, 3.0, 5.0)) -> dict:
    hos = load_handovers(annotations_dir, recording_id)
    ja = load_joint_attention_intervals(annotations_dir, recording_id)
    onsets = [e.start_frame for e in hos if 0 <= e.start_frame < n]
    labels = {
        "handover_active": interval_mask(n, [(e.start_frame, e.end_frame) for e in hos]),
        "ja_active": interval_mask(n, [(s, e) for s, e, _ in ja]),
        "tth_s": time_to_next_onset(n, onsets, max_s=8.0),
    }
    for h in horizons_s:
        labels[f"onset_within_{h:g}s"] = onset_within(n, onsets, int(round(h * FPS)))
    return {"labels": labels, "handovers": hos, "n_joint_attention": len(ja)}
