#!/usr/bin/env python3
"""Per-frame utterance embeddings (speech_emb_v0.npz): for each person, the sentence
embedding (MiniLM, PCA-reduced to 32 dims over all utterances) of the most recently
completed transcript segment, decayed by recency, plus seconds since that segment ended.
Causal: at frame i only segments that ended at or before i/30 s are visible."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
FPS, K = 30, 32
raw, proc = ROOT / "data/raw/comind", ROOT / "data/processed/comind"
recs = sorted(p.parent.name for p in proc.glob("*/kinematics_v0.npz"))
segs = {}  # (rid, role) -> list of (start, end, text)
texts = []
for rid in recs:
    for role in ("leader", "helper"):
        p = raw / "recordings" / rid / "transcripts" / f"{role}_trimmed_sync_transcript.json"
        out = []
        if p.exists():
            for s in json.load(open(p)).get("segments", []):
                t = (s.get("text") or "").strip()
                if t and "end" in s:
                    out.append((float(s["start"]), float(s["end"]), t)); texts.append(t)
        segs[(rid, role)] = out
print("segments", len(texts), flush=True)
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
E = model.encode(texts, batch_size=256, show_progress_bar=False, normalize_embeddings=True)
pca = PCA(n_components=K, random_state=0).fit(E)
Z = pca.transform(E).astype(np.float32); print("PCA var explained", round(float(pca.explained_variance_ratio_.sum()), 3), flush=True)
ptr = 0
for rid in recs:
    n = len(np.load(proc / rid / "kinematics_v0.npz")["features"])
    blocks = []
    for role in ("leader", "helper"):
        f = np.zeros((n, K + 1), np.float32); f[:, K] = 1.0
        ss = segs[(rid, role)]
        if ss:
            ends = np.array([e for _, e, _ in ss]); z = Z[ptr: ptr + len(ss)]; ptr += len(ss)
            order = np.argsort(ends); ends = ends[order]; z = z[order]
            t = np.arange(n) / FPS
            idx = np.searchsorted(ends, t, side="right") - 1
            ok = idx >= 0
            since = np.where(ok, t - ends[np.clip(idx, 0, None)], 10.0)
            decay = np.exp(-np.clip(since, 0, 10) / 4.0)  # utterance fades over ~4 s
            f[ok, :K] = z[idx[ok]] * decay[ok, None]
            f[:, K] = np.clip(since, 0, 10) / 10.0
        blocks.append(f)
    np.savez_compressed(proc / rid / "speech_emb_v0.npz", speech=np.concatenate(blocks, axis=1))
assert ptr == len(texts)
print("built", len(recs))
