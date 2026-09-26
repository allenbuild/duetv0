"""Human QA review API (auto-mounted by server.py).

GET  /api/episode/{name}/review/sample?frac=0.02&seed=0   deterministic per-stream frame sample with that frame's
                                                          body2d / hands / objects, plus autolabel proposals
POST /api/episode/{name}/review                           one verdict -> derived/review/verdicts.jsonl (latest per id+item wins);
                                                          proposal verdicts also update derived/autolabel/proposals.json status
GET  /api/episode/{name}/review/summary                   counts and accuracy per item type and per stream
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .episode import Episode, Stream
from .perception import _frame_list

router = APIRouter()
EPISODES = Path(__file__).resolve().parents[3] / "data/playground/episodes"
ITEM_TYPES = ("body2d", "hands", "objects", "proposal")
VERDICT_STATUS = {"ok": "accepted", "wrong": "rejected", "unsure": "unsure"}


def _ep(name: str) -> Episode:
    d = EPISODES / name
    if not (d / "episode.json").exists():
        raise HTTPException(404, f"no episode {name}")
    return Episode.load(d)


def _clean(x):
    """numpy -> json with NaN/inf -> null."""
    if isinstance(x, np.ndarray):
        return _clean(x.tolist())
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (np.floating, np.integer)):
        x = x.item()
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    return x


def _read_json(p: Path, default):
    if not p.exists():
        return default
    try:
        return json.load(open(p))
    except (json.JSONDecodeError, OSError):
        return default


class _StreamData:
    """Lazily loaded derived arrays for one stream."""

    def __init__(self, ep: Episode, s: Stream):
        self.frames = _frame_list(ep, s)
        self.body = self.hands = self.objects = None
        z = ep.dir / "derived" / "body2d" / f"{s.name}.npz"
        if z.exists():
            b = np.load(z); self.body = b["kpts"]; self.img_w, self.img_h = int(b["img_w"]), int(b["img_h"])
        else:
            self.img_w = self.img_h = 0
        z = ep.dir / "derived" / "hands" / f"{s.name}.npz"
        if z.exists():
            self.hands = np.load(z)["lm2d"]
        z = ep.dir / "derived" / "objects" / f"{s.name}.npz"
        if z.exists():
            o = np.load(z); self.objects = (o["boxes"], o["names"])
        if not self.img_w and self.frames:
            import cv2
            im = cv2.imread(str(self.frames[0]))
            if im is not None:
                self.img_h, self.img_w = im.shape[:2]

    def has(self) -> dict:
        return {"body2d": self.body is not None, "hands": self.hands is not None, "objects": self.objects is not None}

    def item(self, ep_name: str, s: Stream, f: int) -> dict:
        it = {"id": f"{s.name}:{f}", "kind": "frame", "stream": s.name, "role": s.role, "person": s.person, "frame_idx": int(f),
              "image": f"/episodes/{ep_name}/derived/frames/{s.name}/{self.frames[f].name}", "img_w": self.img_w, "img_h": self.img_h,
              "has": self.has(), "body2d": [], "hands": [], "objects": []}
        if self.body is not None and f < len(self.body):
            for p in self.body[f]:
                if np.isfinite(p[:, 0]).any():
                    it["body2d"].append(_clean(np.round(p, 1)))
        if self.hands is not None and f < len(self.hands):
            for side, h in enumerate(self.hands[f]):
                if np.isfinite(h[:, 0]).any():
                    it["hands"].append({"side": "left" if side == 0 else "right", "landmarks": _clean(np.round(h, 1))})
        if self.objects is not None and f < len(self.objects[0]):
            boxes, names = self.objects
            for j in range(boxes.shape[1]):
                b = boxes[f, j]
                if np.isfinite(b[4]):
                    it["objects"].append({"name": str(names[f, j]), "box": _clean(np.round(b[:4], 1)), "conf": round(float(b[4]), 3)})
        return it


def _proposals(ep: Episode) -> list[dict]:
    p = _read_json(ep.dir / "derived" / "autolabel" / "proposals.json", [])
    return [x for x in p if isinstance(x, dict)] if isinstance(p, list) else []


def _proposal_items(ep: Episode) -> list[dict]:
    out = []
    for i, p in enumerate(_proposals(ep)):
        out.append({"id": f"proposal:{i}", "kind": "proposal", "stream": "proposals", "frame_idx": None, "index": i, **p})
    return out


def _latest_verdicts(ep: Episode) -> dict[str, dict[str, dict]]:
    """{id: {item: record}} with the last line for each (id, item) winning."""
    out: dict[str, dict[str, dict]] = {}
    f = ep.dir / "derived" / "review" / "verdicts.jsonl"
    if not f.exists():
        return out
    for line in open(f):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(r, dict) and "id" in r and "item" in r:
            out.setdefault(r["id"], {})[r["item"]] = r
    return out


@router.get("/api/episode/{name}/review/sample")
def sample(name: str, frac: float = 0.02, seed: int = 0):
    ep = _ep(name)
    frac = min(max(frac, 0.0), 1.0)
    items, n_frames = [], {}
    for si, s in enumerate(ep.streams):
        sd = _StreamData(ep, s); n = len(sd.frames); n_frames[s.name] = n
        if not n:
            continue
        k = min(n, max(1, int(round(frac * n))))
        picks = sorted(np.random.default_rng([int(seed), si]).choice(n, size=k, replace=False).tolist())
        for f in picks:
            items.append(sd.item(name, s, f))
    items += _proposal_items(ep)
    return {"episode": name, "frac": frac, "seed": seed, "n_frames": n_frames, "items": items, "verdicts": _latest_verdicts(ep)}


class Verdict(BaseModel):
    id: str = Field(min_length=1, max_length=200)
    stream: str | None = None
    frame_idx: int | None = None
    item: Literal["body2d", "hands", "objects", "proposal"]
    verdict: Literal["ok", "wrong", "unsure"]
    note: str = Field(default="", max_length=2000)


def _update_proposal_status(ep: Episode, v: Verdict) -> bool:
    p = ep.dir / "derived" / "autolabel" / "proposals.json"
    props = _read_json(p, None)
    if not isinstance(props, list):
        return False
    try:
        i = int(v.id.split(":", 1)[1])
    except (IndexError, ValueError):
        return False
    if not (0 <= i < len(props)) or not isinstance(props[i], dict):
        return False
    props[i]["status"] = VERDICT_STATUS[v.verdict]
    if v.note:
        props[i]["review_note"] = v.note
    tmp = p.with_suffix(".json.tmp"); json.dump(props, open(tmp, "w"), indent=1); tmp.replace(p)
    return True


@router.post("/api/episode/{name}/review")
def post_verdict(name: str, v: Verdict):
    ep = _ep(name)
    if v.item == "proposal":
        v.stream = "proposals"
    d = ep.dir / "derived" / "review"; d.mkdir(parents=True, exist_ok=True)
    rec = v.model_dump(); rec["ts"] = time.time()
    with open(d / "verdicts.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    out = {"ok": True, "record": rec}
    if v.item == "proposal":
        out["proposal_updated"] = _update_proposal_status(ep, v)
    return out


def _bucket() -> dict:
    return {"n": 0, "ok": 0, "wrong": 0, "unsure": 0, "accuracy": None}


def _finish(b: dict) -> dict:
    judged = b["ok"] + b["wrong"]
    b["accuracy"] = round(b["ok"] / judged, 3) if judged else None
    return b


@router.get("/api/episode/{name}/review/summary")
def summary(name: str):
    ep = _ep(name)
    latest = _latest_verdicts(ep)
    by_item = {t: _bucket() for t in ITEM_TYPES}; by_stream: dict[str, dict] = {}; by_stream_item: dict[str, dict] = {}
    total = 0
    for _id, per_item in latest.items():
        for item, r in per_item.items():
            vd = r.get("verdict"); st = r.get("stream") or ("proposals" if item == "proposal" else "?")
            if vd not in ("ok", "wrong", "unsure") or item not in ITEM_TYPES:
                continue
            total += 1
            for b in (by_item[item], by_stream.setdefault(st, _bucket()), by_stream_item.setdefault(st, {}).setdefault(item, _bucket())):
                b["n"] += 1; b[vd] += 1
    for b in list(by_item.values()) + list(by_stream.values()):
        _finish(b)
    for d in by_stream_item.values():
        for b in d.values():
            _finish(b)
    props = _proposals(ep)
    pstat = {"total": len(props), "accepted": sum(p.get("status") == "accepted" for p in props), "rejected": sum(p.get("status") == "rejected" for p in props)}
    pstat["pending"] = pstat["total"] - pstat["accepted"] - pstat["rejected"]
    return {"episode": name, "total": total, "by_item": by_item, "by_stream": by_stream, "by_stream_item": by_stream_item, "proposals": pstat}
