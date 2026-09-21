"""Single-view vs paired-view kinematic benchmark on CoMind.

The question this answers: given only body kinematics (hand landmarks + gaze from
Aria glasses, no pixels), does observing BOTH people improve prediction of
collaborative events over observing ONE person, and is that improvement due to
the *interaction* (temporal correspondence) rather than just extra input?

Views
-----
- ``leader``       : leader's own hands + gaze
- ``helper``       : helper's own hands + gaze
- ``both``         : both streams, time-aligned
- ``both_shuffled``: both streams, but the partner stream is circularly shifted by
                     >= 60 s within the same recording. Marginal statistics are
                     identical to ``both``; only the interaction is destroyed.
                     If ``both`` beats ``both_shuffled`` the gain is interaction.

Tasks (per-frame, causal: the model sees only the past)
------
- handover_active   : a handover is in progress now
- onset_within_1s   : a handover starts in the next 1 s
- onset_within_2s   : a handover starts in the next 2 s
- ja_active         : joint attention is active now
- tth_s             : seconds to next handover onset (regression, <= 5 s)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn

from duet.adapters.comind.kinematics import PERSON_DIM

CLS_TASKS = ("handover_active", "onset_within_2s", "onset_within_5s", "ja_active")
ALL_LABELS = ("handover_active", "onset_within_1s", "onset_within_2s", "onset_within_3s", "onset_within_5s", "ja_active")
REG_MAX_S = 8.0


def set_reg_task(name: str, max_s: float) -> None:
    """Choose the regression target: 'tth_s' (time to next onset) or 'tte_s' (time to completion within a handover)."""
    global REG_TASK, REG_MAX_S
    REG_TASK, REG_MAX_S = name, max_s


def set_tasks(tasks) -> None:
    """Override the classification task set (module-global, read at call time by every function here)."""
    global CLS_TASKS
    CLS_TASKS = tuple(tasks)
REG_TASK = "tth_s"
VIEWS = ("leader", "helper", "both", "both_shuffled")
FPS = 30
SHUFFLE_MIN_SHIFT = 60 * FPS


# --------------------------------------------------------------------------- data

@dataclass
class Recording:
    rid: str
    x: np.ndarray  # [N, 2*PERSON_DIM]
    y: dict[str, np.ndarray]  # per-frame labels
    handover_onsets: list[int] = field(default_factory=list)
    s: np.ndarray | None = None  # [N, 2*SPEECH_DIM] optional speech block (leader, helper)
    w: np.ndarray | None = None  # [N, WORLD_DIM] optional cross-person shared-world block


def load_recordings(proc_root: Path, ids: list[str], speech_source: str = "hashed") -> list[Recording]:
    """speech_source: 'hashed' (speech_v0.npz), 'emb' (speech_emb_v0.npz) or 'both' (concatenated per person)."""
    out = []
    for rid in ids:
        p = proc_root / rid / "kinematics_v0.npz"
        if not p.exists():
            continue
        z = np.load(p)
        meta = json.load(open(proc_root / rid / "kinematics_v0.json"))
        y = {k: z[k] for k in tuple(k for k in ALL_LABELS + ("tth_s", "tte_s") if k in z.files)}
        speech = _load_speech(proc_root / rid, speech_source)
        wp = proc_root / rid / "world_v0.npz"
        world = np.load(wp)["world"] if wp.exists() else None
        out.append(Recording(rid, z["features"], y, [h["start_frame"] for h in meta["handovers"]], speech, world))
    return out


def _load_speech(d: Path, source: str):
    files = {"hashed": ["speech_v0.npz"], "emb": ["speech_emb_v0.npz"], "both": ["speech_v0.npz", "speech_emb_v0.npz"]}[source]
    blocks = [np.load(d / f)["speech"] for f in files if (d / f).exists()]
    if len(blocks) != len(files):
        return None
    if len(blocks) == 1:
        return blocks[0]
    # keep per-person layout: [leader_a, leader_b, helper_a, helper_b]
    halves = [(b[:, : b.shape[1] // 2], b[:, b.shape[1] // 2 :]) for b in blocks]
    return np.concatenate([h[0] for h in halves] + [h[1] for h in halves], axis=1)


def view_input(x: np.ndarray, view: str, rng: np.random.Generator | None, speech: np.ndarray | None = None,
               world: np.ndarray | None = None) -> np.ndarray:
    """Select the input block(s) for a view. Returns [N, D_view].

    If ``speech`` is given, each person's speech block is appended to that person's
    kinematics block (and shuffled together with it for ``both_shuffled``).
    """
    L = x[:, :PERSON_DIM]
    H = x[:, PERSON_DIM:]
    if speech is not None:
        half = speech.shape[1] // 2
        L = np.concatenate([L, speech[:, :half]], axis=1)
        H = np.concatenate([H, speech[:, half:]], axis=1)
    if view == "leader":
        return L
    if view == "helper":
        return H
    if view == "both":
        blocks = [L, H] + ([world] if world is not None else [])
        return np.concatenate(blocks, axis=1)
    if view == "both_shuffled":
        n = len(x)
        rng = rng or np.random.default_rng(0)
        shift = int(rng.integers(SHUFFLE_MIN_SHIFT, max(SHUFFLE_MIN_SHIFT + 1, n - SHUFFLE_MIN_SHIFT)))
        return np.concatenate([L, np.roll(H, shift, axis=0)], axis=1)
    raise ValueError(view)


def add_deltas(x: torch.Tensor) -> torch.Tensor:
    """[B, T, D] -> [B, T, 2D] with causal first differences appended."""
    d = torch.zeros_like(x)
    d[:, 1:] = x[:, 1:] - x[:, :-1]
    return torch.cat([x, d], dim=-1)


class Normalizer:
    def __init__(self, xs: list[np.ndarray]):
        cat = np.concatenate(xs, axis=0)
        self.mean = cat.mean(0, keepdims=True).astype(np.float32)
        self.std = (cat.std(0, keepdims=True) + 1e-4).astype(np.float32)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


# --------------------------------------------------------------------------- model

class ChannelNorm(nn.Module):
    """LayerNorm over channels at each time step: strictly causal and identical at train/eval.

    (GroupNorm/BatchNorm normalize over the time axis, which leaks future frames and gives
    different statistics for a 512-frame training crop vs a 40k-frame evaluation sequence.)
    """

    def __init__(self, c: int):
        super().__init__()
        self.ln = nn.LayerNorm(c)

    def forward(self, x):  # [B, C, T]
        return self.ln(x.transpose(1, 2)).transpose(1, 2)


class CausalBlock(nn.Module):
    def __init__(self, c: int, dilation: int, k: int = 3, p: float = 0.1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv1 = nn.Conv1d(c, c, k, dilation=dilation)
        self.conv2 = nn.Conv1d(c, c, k, dilation=dilation)
        self.norm1 = ChannelNorm(c)
        self.norm2 = ChannelNorm(c)
        self.drop = nn.Dropout(p)

    def forward(self, x):  # [B, C, T]
        h = F.gelu(self.norm1(self.conv1(F.pad(x, (self.pad, 0)))))
        h = self.drop(h)
        h = self.norm2(self.conv2(F.pad(h, (self.pad, 0))))
        return F.gelu(x + h)


class CausalTCN(nn.Module):
    """Dilated causal TCN; receptive field = 1 + sum(2*(k-1)*d) ~ 253 frames (8.4 s)."""

    def __init__(self, d_in: int, c: int = 128, dilations=(1, 2, 4, 8, 16, 32), n_cls: int | None = None, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Conv1d(d_in, c, 1)
        self.in_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([CausalBlock(c, d, p=dropout) for d in dilations])
        self.cls_head = nn.Conv1d(c, n_cls or len(CLS_TASKS), 1)
        self.reg_head = nn.Conv1d(c, 1, 1)

    def forward(self, x):  # x [B, T, D]
        h = self.in_drop(self.inp(x.transpose(1, 2)))
        for b in self.blocks:
            h = b(h)
        return self.cls_head(h).transpose(1, 2), self.reg_head(h).transpose(1, 2)[..., 0]


# --------------------------------------------------------------------------- training

@dataclass
class TrainConfig:
    steps: int = 3000
    batch: int = 48
    crop: int = 512
    lr: float = 1e-3
    eval_every: int = 500
    channels: int = 128
    seed: int = 0
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    # v1 recipe: fraction of training crops forced to contain a handover onset, per-task loss weights,
    # and the cap on the positive-class weight (handover frames are ~0.5% of data).
    onset_crop_frac: float = 0.0
    task_weights: tuple = ()  # empty = all ones
    pos_weight_cap: float = 20.0
    dropout: float = 0.1
    use_speech: bool = False
    use_world: bool = False
    select_metric: str = "mean_auroc"  # or mean_ap


def _pos_weights(recs: list[Recording], cap: float = 20.0) -> torch.Tensor:
    w = []
    for t in CLS_TASKS:
        y = np.concatenate([r.y[t] for r in recs])
        p = y.mean()
        w.append(min(cap, (1 - p) / max(p, 1e-4)))
    return torch.tensor(w, dtype=torch.float32)


def _sample_batch(recs, view, norm, cfg, rng, cache, onset_recs=()):
    xs, ys, ts = [], [], []
    for _ in range(cfg.batch):
        if onset_recs and rng.random() < cfg.onset_crop_frac:
            r = onset_recs[rng.integers(len(onset_recs))]
            n = len(r.x)
            o = r.handover_onsets[rng.integers(len(r.handover_onsets))]
            # place the onset in the causal-valid back 60% of the crop so the model sees >= 3 s of lead-in
            s = int(np.clip(o - rng.integers(int(0.4 * cfg.crop), cfg.crop - 10), 0, max(1, n - cfg.crop)))
        else:
            r = recs[rng.integers(len(recs))]
            n = len(r.x)
            s = int(rng.integers(0, max(1, n - cfg.crop)))
        xv = cache[(r.rid, view)]
        xs.append(xv[s : s + cfg.crop])
        ys.append(np.stack([r.y[t][s : s + cfg.crop] for t in CLS_TASKS], axis=1))
        ts.append(r.y[REG_TASK][s : s + cfg.crop])
    x = torch.from_numpy(np.stack(xs)).float()
    y = torch.from_numpy(np.stack(ys)).float()
    t = torch.from_numpy(np.stack(ts)).float()
    return x, y, t


def _build_cache(recs, view, norm, rng, cfg):
    return {(r.rid, view): norm(view_input(r.x, view, rng, r.s if cfg.use_speech else None, r.w if cfg.use_world else None)).astype(np.float32) for r in recs}


@torch.no_grad()
def predict(model, x_np: np.ndarray, cfg: TrainConfig) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    x = torch.from_numpy(x_np[None]).float().to(cfg.device)
    logits, reg = model(add_deltas(x))
    return torch.sigmoid(logits)[0].cpu().numpy(), reg[0].cpu().numpy()


def _reg_baseline_mae(recs, valid_from: int) -> float:
    """MAE of predicting the constant median target (what a model that ignores its input achieves)."""
    ys = np.concatenate([r.y[REG_TASK][valid_from:] for r in recs])
    ys = ys[np.isfinite(ys) & (ys <= REG_MAX_S)]
    return float(np.abs(ys - np.median(ys)).mean()) if ys.size else float("nan")


def evaluate(model, recs, view, norm, cfg, rng, want_preds=False):
    """Frame-level AP / AUROC per task on full sequences; TTH MAE on frames with tth<=3s."""
    probs = {t: [] for t in CLS_TASKS}
    ys = {t: [] for t in CLS_TASKS}
    tth_err, preds = [], {}
    for r in recs:
        p, reg = predict(model, norm(view_input(r.x, view, rng, r.s if cfg.use_speech else None, r.w if cfg.use_world else None)), cfg)
        # ignore the first 2 s (model warm-up) and frames where both people's hands+gaze are absent
        valid = np.ones(len(r.x), bool)
        valid[: 2 * FPS] = False
        for i, t in enumerate(CLS_TASKS):
            probs[t].append(p[valid, i])
            ys[t].append(r.y[t][valid])
        m = np.isfinite(r.y[REG_TASK]) & (r.y[REG_TASK] <= REG_MAX_S) & valid
        if m.any():
            tth_err.append(np.abs(np.clip(reg[m], 0, REG_MAX_S) - r.y[REG_TASK][m]))
        if want_preds:
            preds[r.rid] = {"probs": p.astype(np.float16), "tth": reg.astype(np.float16)}
    out = {}
    for t in CLS_TASKS:
        y = np.concatenate(ys[t])
        p = np.concatenate(probs[t])
        out[t] = {
            "ap": float(average_precision_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else float("nan"),
            "prevalence": float(y.mean()),
        }
    out[REG_TASK] = {"mae_s": float(np.concatenate(tth_err).mean()) if tth_err else float("nan"),
                     "baseline_mae_s": _reg_baseline_mae(recs, valid_from=2 * FPS)}
    out["_summary_ap"] = float(np.nanmean([out[t]["ap"] for t in CLS_TASKS]))
    out["_summary_auroc"] = float(np.nanmean([out[t]["auroc"] for t in CLS_TASKS]))
    return (out, preds) if want_preds else out


def train_view(train_recs, val_recs, test_recs, view: str, cfg: TrainConfig, log=print, want_preds_for=()):
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    norm = Normalizer([view_input(r.x, view, rng, r.s if cfg.use_speech else None, r.w if cfg.use_world else None) for r in train_recs])
    cache = _build_cache(train_recs, view, norm, rng, cfg)
    d_in = 2 * next(iter(cache.values())).shape[1]
    model = CausalTCN(d_in, c=cfg.channels, dropout=cfg.dropout).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    pos_w = _pos_weights(train_recs, cfg.pos_weight_cap).to(cfg.device)
    task_w = torch.tensor(cfg.task_weights or (1.0,) * len(CLS_TASKS), dtype=torch.float32, device=cfg.device)
    onset_recs = [r for r in train_recs if r.handover_onsets]
    best, best_state, t0 = -1.0, None, time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        x, y, tth = _sample_batch(train_recs, view, norm, cfg, rng, cache, onset_recs)
        x, y, tth = x.to(cfg.device), y.to(cfg.device), tth.to(cfg.device)
        logits, reg = model(add_deltas(x))
        warm = torch.zeros(cfg.crop, device=cfg.device, dtype=torch.bool)
        warm[2 * FPS :] = True  # do not penalize the causal warm-up region
        per_task = F.binary_cross_entropy_with_logits(logits[:, warm], y[:, warm], pos_weight=pos_w, reduction="none").mean(dim=(0, 1))
        cls_loss = (per_task * task_w).sum() / task_w.sum()
        m = torch.isfinite(tth) & warm[None]
        reg_loss = F.smooth_l1_loss(reg[m], tth[m]) if m.any() else logits.sum() * 0
        loss = cls_loss + 0.2 * reg_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            val = evaluate(model, val_recs, view, norm, cfg, np.random.default_rng(123))
            log(f"[{view} s{cfg.seed}] step {step} loss {loss.item():.3f} val meanAUROC {val['_summary_auroc']:.4f} "
                + " ".join(f"{t}={val[t]['ap']:.3f}" for t in CLS_TASKS) + f" tthMAE={val[REG_TASK]['mae_s']:.2f} ({time.time()-t0:.0f}s)")
            score = val["_summary_auroc"] if cfg.select_metric == "mean_auroc" else val["_summary_ap"]
            if best_state is None or score > best:
                best = score if np.isfinite(score) else best
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    test, preds = evaluate(model, test_recs, view, norm, cfg, np.random.default_rng(321), want_preds=True)
    preds = {k: v for k, v in preds.items() if k in set(want_preds_for)}
    return {"view": view, "seed": cfg.seed, "val_best_meanAP": best, "test": test, "d_in": d_in,
            "n_params": sum(p.numel() for p in model.parameters())}, preds, model, norm
