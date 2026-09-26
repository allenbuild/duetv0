"""Dense VLM annotation of an episode: one contact sheet every ``interval_s`` seconds, strict JSON per keyframe.

For every keyframe on the common timeline we tile all streams into ONE JPEG (each tile labelled with the
stream name and the person), save it to ``derived/annotate/keyframes/<idx>.jpg`` and ask a vision-language
model for a fixed JSON schema (persons / coordination / handover / summary). Results are cached per keyframe
as ``<idx>.json`` (skipped when present) and merged into ``derived/annotate/annotations.json``.

Backends (``DUET_ANNOTATE_BACKEND``, ``rig.json`` ``{"annotate": {"backend": ...}}`` or an episode note
``annotate: backend=... model=... interval_s=... max_keyframes=...``):

    anthropic   Anthropic SDK, image as base64. Default model ``claude-haiku-4-5-20251001``. Needs ANTHROPIC_API_KEY.
    claude_cli  local ``claude -p ... --output-format json --allowedTools Read``; the prompt names the absolute
                image path and the CLI reads it with its Read tool. Needs a logged-in CLI.
    dry         no model call: emits the schema with nulls and a note. Auto-selected when neither works.

Auto-selection: anthropic if ANTHROPIC_API_KEY is set, else claude_cli if ``claude -p "reply with exactly: ok"``
succeeds, else dry. A cached keyframe produced by the dry backend is redone when a real backend is available.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

from .episode import Episode
from .perception import _frame_list

DEFAULT_MODEL = {"anthropic": "claude-haiku-4-5-20251001", "claude_cli": None, "codex_cli": None, "dry": None}
CLI_TIMEOUT_S = 240.0
CLI_PROBE_TIMEOUT_S = 90.0
COORDINATION = ("none", "request", "handover", "joint_work", "waiting")

PROMPT = """You are annotating one instant of a two-person collaboration recorded by several cameras at once.
The image is a contact sheet: every tile is a different camera at the SAME moment. Tile labels give the stream
name, its role (ego = head-mounted camera worn by that person, exo = fixed camera) and, for ego tiles, the
person wearing it. In an ego tile you see that person's own hands and what they look at.

Return ONLY a JSON object, no prose, no markdown fences, exactly this schema:
{
  "persons": [
    {"label": "<person name from the tile labels, or 'left person'/'right person' if unnamed>",
     "action": "<verb + object, e.g. 'cutting onion', 'holding bowl', 'reaching for pan'>",
     "holding": ["<object>", ...],
     "attending_to": "<partner | object name | elsewhere>"}
  ],
  "coordination": "none | request | handover | joint_work | waiting",
  "handover": {"occurring": true|false, "giver": "<person or null>", "receiver": "<person or null>", "object": "<object or null>"},
  "summary": "<one sentence describing what the two people are doing together right now>"
}
Rules: one entry per visible person; use the names in the labels; "holding" is a list (empty if hands are free);
"coordination" must be one of the five values; use null for unknown handover fields; keep the summary to one sentence."""

BackendFn = Callable[[Path, str, "str | None"], tuple[str, dict]]  # (image, prompt, model) -> (text, meta)


# ----------------------------------------------------------------------------- schema helpers
def empty_annotation(note: str | None = None) -> dict:
    d = {"persons": [], "coordination": None, "handover": {"occurring": None, "giver": None, "receiver": None, "object": None}, "summary": None}
    if note:
        d["note"] = note
    return d


def normalize(d: dict) -> dict:
    """Coerce a model reply into the fixed schema so the UI/exporter can rely on the keys."""
    out = empty_annotation()
    persons = d.get("persons") if isinstance(d, dict) else None
    for p in persons if isinstance(persons, list) else []:
        if not isinstance(p, dict):
            continue
        holding = p.get("holding")
        if isinstance(holding, str):
            holding = [holding] if holding else []
        out["persons"].append({"label": p.get("label"), "action": p.get("action"), "holding": [str(h) for h in (holding or []) if h is not None],
                               "attending_to": p.get("attending_to")})
    c = d.get("coordination") if isinstance(d, dict) else None
    out["coordination"] = c if c in COORDINATION else (None if c is None else str(c))
    h = d.get("handover") if isinstance(d, dict) else None
    if isinstance(h, dict):
        occ = h.get("occurring")
        out["handover"] = {"occurring": bool(occ) if occ is not None else None, "giver": h.get("giver"), "receiver": h.get("receiver"), "object": h.get("object")}
    s = d.get("summary") if isinstance(d, dict) else None
    out["summary"] = str(s) if s is not None else None
    if isinstance(d, dict) and d.get("note"):
        out["note"] = str(d["note"])
    return out


def extract_json(text: str) -> dict:
    """First JSON object in a model reply: tolerates fences, prose before/after, braces inside strings."""
    if not isinstance(text, str):
        raise ValueError("model reply is not text")
    t = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", t, re.S)
    if m:
        t = m.group(1).strip()
    try:
        d = json.loads(t)
        if isinstance(d, dict):
            return d
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in reply")
    depth, in_str, esc = 0, False, False
    for i in range(start, len(t)):
        ch = t[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                d = json.loads(t[start:i + 1])
                if isinstance(d, dict):
                    return d
                break
    raise ValueError("unbalanced JSON object in reply")


# ----------------------------------------------------------------------------- backends
def call_anthropic(image: Path, prompt: str, model: str | None) -> tuple[str, dict]:
    import anthropic
    client = anthropic.Anthropic()
    data = base64.standard_b64encode(Path(image).read_bytes()).decode("ascii")
    r = client.messages.create(model=model or DEFAULT_MODEL["anthropic"], max_tokens=1024, messages=[{"role": "user", "content": [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}, {"type": "text", "text": prompt}]}])
    text = "".join(b.text for b in r.content if b.type == "text")
    return text, {"input_tokens": r.usage.input_tokens, "output_tokens": r.usage.output_tokens, "stop_reason": r.stop_reason}


def _cli_env() -> dict:
    # a nested CLAUDECODE marker makes the CLI think it runs inside another session; drop it
    return {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}


def call_claude_cli(image: Path, prompt: str, model: str | None) -> tuple[str, dict]:
    full = f"Use the Read tool to read the image file at {Path(image).resolve()} and look at it carefully. Then:\n\n{prompt}"
    cmd = ["claude", "-p", full, "--output-format", "json", "--allowedTools", "Read"]
    if model:
        cmd += ["--model", model]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT_S, env=_cli_env())
    try:
        j = json.loads(r.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude CLI returned non-JSON (rc={r.returncode}): {(r.stdout or r.stderr)[:200]}")
    if j.get("is_error"):
        raise RuntimeError(f"claude CLI error: {str(j.get('result'))[:200]}")
    usage = j.get("usage") or {}
    meta = {"input_tokens": usage.get("input_tokens"), "output_tokens": usage.get("output_tokens"), "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            "cost_usd": j.get("total_cost_usd"), "duration_api_ms": j.get("duration_api_ms"), "num_turns": j.get("num_turns"), "session_id": j.get("session_id")}
    return str(j.get("result", "")), meta


def call_codex_cli(image: Path, prompt: str, model: str | None) -> tuple[str, dict]:
    """OpenAI Codex CLI (``codex exec -i <image> --json``), authenticated by the machine's ChatGPT login (``codex login``).
    The JSONL event stream is parsed for the final agent message; the model defaults to the CLI's own default (-m overrides)."""
    cmd = ["codex", "exec", "--json", "--skip-git-repo-check", "-s", "read-only", "-i", str(Path(image).resolve())]
    if model:
        cmd += ["-m", model]
    cmd.append("Look at the attached image carefully. Then:\n\n" + prompt)
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=CLI_TIMEOUT_S, env=_cli_env(), cwd=str(Path(image).resolve().parent))
    text, meta, errors = "", {"events": 0}, []
    for line in r.stdout.splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        meta["events"] += 1
        item = ev.get("item") or {}
        if ev.get("type") == "item.completed" and item.get("type") == "agent_message":
            text = item.get("text") or text
        elif ev.get("type") in ("turn.failed", "error"):
            errors.append(str(ev.get("error") or ev.get("message") or ev)[:200])
        elif ev.get("type") == "turn.completed":
            u = ev.get("usage") or {}; meta.update({"input_tokens": u.get("input_tokens"), "output_tokens": u.get("output_tokens"), "cached_input_tokens": u.get("cached_input_tokens")})
    if errors or (not text and r.returncode != 0):
        raise RuntimeError(f"codex CLI error (rc={r.returncode}): {'; '.join(errors) or (r.stderr or r.stdout)[:200]}")
    if not text:
        raise RuntimeError(f"codex CLI returned no agent message (rc={r.returncode}): {(r.stderr or r.stdout)[:200]}")
    return text, meta


def call_dry(image: Path, prompt: str, model: str | None) -> tuple[str, dict]:
    return json.dumps(empty_annotation("dry backend: no model was called (set ANTHROPIC_API_KEY, or log in the claude or codex CLI)")), {}


BACKENDS: dict[str, BackendFn] = {"anthropic": call_anthropic, "claude_cli": call_claude_cli, "codex_cli": call_codex_cli, "dry": call_dry}
_codex_probe: tuple[bool, str] | None = None


def codex_available(force: bool = False) -> tuple[bool, str]:
    """Is the OpenAI Codex CLI installed and logged in (``codex login status``)? Cached per process."""
    global _codex_probe
    if _codex_probe is not None and not force:
        return _codex_probe
    try:
        r = subprocess.run(["codex", "login", "status"], capture_output=True, text=True, timeout=CLI_PROBE_TIMEOUT_S, env=_cli_env())
        out = (r.stdout + r.stderr).strip()
        ok = r.returncode == 0 and "logged in" in out.lower() and "not logged in" not in out.lower()
        _codex_probe = (ok, "ok" if ok else f"codex CLI: {out[:120] or 'unknown login state'}")
    except FileNotFoundError:
        _codex_probe = (False, "codex CLI not installed")
    except subprocess.TimeoutExpired:
        _codex_probe = (False, "codex CLI probe timed out")
    except Exception as e:  # noqa: BLE001
        _codex_probe = (False, f"codex CLI probe failed: {type(e).__name__}: {str(e)[:120]}")
    return _codex_probe
_cli_probe: tuple[bool, str] | None = None


def cli_available(force: bool = False) -> tuple[bool, str]:
    """Does ``claude -p`` answer headlessly on this machine? Cached per process."""
    global _cli_probe
    if _cli_probe is not None and not force:
        return _cli_probe
    try:
        r = subprocess.run(["claude", "-p", "reply with exactly: ok", "--output-format", "json"], capture_output=True, text=True, timeout=CLI_PROBE_TIMEOUT_S, env=_cli_env())
        j = json.loads(r.stdout)
        ok = not j.get("is_error") and "ok" in str(j.get("result", "")).lower()
        _cli_probe = (ok, "ok" if ok else f"claude CLI: {str(j.get('result'))[:120]}")
    except FileNotFoundError:
        _cli_probe = (False, "claude CLI not installed")
    except subprocess.TimeoutExpired:
        _cli_probe = (False, "claude CLI probe timed out")
    except Exception as e:  # noqa: BLE001
        _cli_probe = (False, f"claude CLI probe failed: {type(e).__name__}: {str(e)[:120]}")
    return _cli_probe


def select_backend(requested: str | None = None) -> tuple[str, str]:
    """-> (backend, reason)."""
    if requested:
        if requested not in BACKENDS:
            raise ValueError(f"unknown annotate backend {requested!r}; choose from {sorted(BACKENDS)}")
        return requested, "requested"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic", "ANTHROPIC_API_KEY set"
    ok, why = cli_available()
    if ok:
        return "claude_cli", "claude CLI answers headlessly"
    ok2, why2 = codex_available()
    if ok2:
        return "codex_cli", "codex CLI is logged in"
    return "dry", f"no ANTHROPIC_API_KEY; {why}; {why2}"


# ----------------------------------------------------------------------------- options
def _options(ep: Episode) -> dict:
    """Stage options from rig.json -> episode notes -> environment (later wins)."""
    o: dict = {"interval_s": 2.0, "backend": None, "model": None, "max_keyframes": None}
    rig = ep.dir / "rig.json"
    if rig.exists():
        try:
            o.update({k: v for k, v in (json.load(open(rig)).get("annotate") or {}).items() if k in o})
        except (json.JSONDecodeError, AttributeError):
            pass
    for note in ep.notes or []:
        if isinstance(note, str) and note.strip().lower().startswith("annotate:"):
            for tok in note.split(":", 1)[1].split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    if k in o:
                        o[k] = v
    env = {"backend": "DUET_ANNOTATE_BACKEND", "model": "DUET_ANNOTATE_MODEL", "interval_s": "DUET_ANNOTATE_INTERVAL_S", "max_keyframes": "DUET_ANNOTATE_MAX_KEYFRAMES"}
    for k, var in env.items():
        if os.environ.get(var):
            o[k] = os.environ[var]
    o["interval_s"] = float(o["interval_s"])
    o["max_keyframes"] = int(o["max_keyframes"]) if o["max_keyframes"] not in (None, "", "0", 0) else None
    o["backend"] = o["backend"] or None
    o["model"] = o["model"] or None
    return o


# ----------------------------------------------------------------------------- keyframes and contact sheet
def keyframe_indices(n_frames: int, proc_fps: float, interval_s: float, max_keyframes: int | None = None) -> list[int]:
    step = max(1, int(round(interval_s * proc_fps)))
    idx = list(range(0, n_frames, step))
    return idx[:max_keyframes] if max_keyframes else idx


def _label(img: np.ndarray, text: str, y: int = 0) -> None:
    import cv2
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(img, (0, y), (min(img.shape[1], w + 12), y + h + base + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (6, y + h + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def contact_sheet(ep: Episode, frame_idx: int, frames: dict[str, list[Path]], t_s: float | None = None, max_w: int = 1280) -> np.ndarray:
    """All streams at one common-timeline frame tiled into a grid at most ``max_w`` px wide, tiles labelled."""
    import cv2
    tiles = []
    for s in ep.streams:
        fl = frames.get(s.name) or []
        img = cv2.imread(str(fl[frame_idx])) if frame_idx < len(fl) else None
        if img is None:
            img = np.zeros((360, 640, 3), np.uint8)
        tiles.append((s, img))
    n = len(tiles)
    cols = 1 if n == 1 else (2 if n <= 4 else 3)
    tile_w = max_w // cols
    rows = []
    for r0 in range(0, n, cols):
        row = []
        for s, img in tiles[r0:r0 + cols]:
            h, w = img.shape[:2]
            im = cv2.resize(img, (tile_w, max(1, int(round(h * tile_w / w)))), interpolation=cv2.INTER_AREA)
            who = f"  person: {s.person}" if s.person else ""
            _label(im, f"{s.name}  [{s.role}]{who}")
            row.append(im)
        H = max(im.shape[0] for im in row)
        padded = [np.pad(im, ((0, H - im.shape[0]), (0, 0), (0, 0))) for im in row]
        while len(padded) < cols:
            padded.append(np.zeros((H, tile_w, 3), np.uint8))
        rows.append(np.hstack(padded))
    sheet = np.vstack(rows)
    if t_s is not None:
        _label(sheet, f"t = {t_s:.2f} s  (frame {frame_idx})", y=sheet.shape[0] - 26)
    return sheet


# ----------------------------------------------------------------------------- stage
def run_annotate(ep: Episode, *, interval_s: float = 2.0, backend: str | None = None, model: str | None = None, max_keyframes: int | None = None,
                 call: BackendFn | None = None, log: Callable[[str], None] = print) -> dict:
    """Annotate every keyframe; returns the run log. ``call`` overrides the backend function (tests)."""
    ep.set_status("annotate", "running")
    frames = {s.name: _frame_list(ep, s) for s in ep.streams}
    if not frames or not all(frames.values()):
        ep.set_status("annotate", "failed", "no extracted frames; run the frames stage first")
        return {}
    if call is not None:
        backend, reason = backend or "custom", "call() supplied"
    else:
        backend, reason = select_backend(backend)
        call = BACKENDS[backend]
    model = model or DEFAULT_MODEL.get(backend)
    n = min(len(v) for v in frames.values())
    kf = keyframe_indices(n, ep.proc_fps, interval_s, max_keyframes)
    kdir = ep.derived / "annotate" / "keyframes"; kdir.mkdir(parents=True, exist_ok=True)
    import cv2
    items, per_kf, t_start = [], [], time.time()
    cached = errors = in_tok = out_tok = 0
    log(f"  annotate: backend={backend} ({reason}) model={model} keyframes={len(kf)} interval={interval_s}s")
    for idx, k in enumerate(kf):
        t = ep.common_start_s + k / ep.proc_fps
        jpg, js = kdir / f"{idx:04d}.jpg", kdir / f"{idx:04d}.json"
        if not jpg.exists():
            cv2.imwrite(str(jpg), contact_sheet(ep, k, frames, t), [cv2.IMWRITE_JPEG_QUALITY, 85])
        rec = None
        if js.exists():
            try:
                rec = json.load(open(js))
            except json.JSONDecodeError:
                rec = None
            if rec is not None and rec.get("_meta", {}).get("backend") == "dry" and backend != "dry":
                rec = None  # a real backend is available now: redo the placeholder
            if rec is not None and rec.get("error"):
                rec = None  # failed calls are not cached: retry
        if rec is not None:
            cached += 1
        else:
            t0 = time.time()
            try:
                text, meta = call(jpg, PROMPT, model)
                ann = normalize(extract_json(text))
            except Exception as e:  # noqa: BLE001
                ann = empty_annotation(); ann["error"] = f"{type(e).__name__}: {str(e)[:300]}"; meta = {}; errors += 1
                log(f"  annotate: keyframe {idx} failed: {ann['error']}")
            el = time.time() - t0; per_kf.append(round(el, 2))
            in_tok += int(meta.get("input_tokens") or 0); out_tok += int(meta.get("output_tokens") or 0)
            rec = {"idx": idx, "frame": int(k), "t": round(t, 3), "image": f"derived/annotate/keyframes/{idx:04d}.jpg", **ann,
                   "_meta": {"backend": backend, "model": model, "elapsed_s": round(el, 2), **{kk: vv for kk, vv in meta.items() if vv is not None}}}
            json.dump(rec, open(js, "w"), indent=1)
        items.append(rec)
    elapsed = time.time() - t_start
    json.dump(items, open(ep.derived / "annotate" / "annotations.json", "w"), indent=1)
    run_log = {"backend": backend, "backend_reason": reason, "model": model, "interval_s": interval_s, "count": len(items), "new": len(items) - cached, "cached": cached,
               "errors": errors, "elapsed_s": round(elapsed, 2), "per_keyframe_s": per_kf, "input_tokens": in_tok, "output_tokens": out_tok, "finished": time.time()}
    json.dump(run_log, open(ep.derived / "annotate" / "log.json", "w"), indent=1)
    log(f"  annotate: {len(items)} keyframes ({cached} cached, {errors} errors) in {elapsed:.1f}s; tokens in/out {in_tok}/{out_tok}")
    detail = f"{backend}{'/' + model if model else ''}: {len(items)} keyframes every {interval_s:g}s ({cached} cached, {errors} errors), {elapsed:.1f}s"
    ep.set_status("annotate", "done" if errors < max(1, len(items)) else "failed", detail)
    return run_log


def annotate(ep: Episode) -> None:
    o = _options(ep)
    run_annotate(ep, interval_s=o["interval_s"], backend=o["backend"], model=o["model"], max_keyframes=o["max_keyframes"])


def main(argv: list[str] | None = None) -> None:
    import argparse
    ap = argparse.ArgumentParser(description="dense VLM annotation of a playground episode")
    ap.add_argument("episode", help="episode name under data/playground/episodes or a path")
    ap.add_argument("--backend", choices=sorted(BACKENDS)); ap.add_argument("--model"); ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--max-keyframes", type=int, default=None); ap.add_argument("--force", action="store_true", help="discard cached keyframe JSON first")
    a = ap.parse_args(argv)
    p = Path(a.episode)
    if not (p / "episode.json").exists():
        p = Path(__file__).resolve().parents[3] / "data/playground/episodes" / a.episode
    ep = Episode.load(p); o = _options(ep)
    if a.force:
        for f in (ep.derived / "annotate" / "keyframes").glob("*.json"):
            f.unlink()
    run_annotate(ep, interval_s=a.interval or o["interval_s"], backend=a.backend or o["backend"], model=a.model or o["model"], max_keyframes=a.max_keyframes or o["max_keyframes"])


if __name__ == "__main__":
    main(sys.argv[1:])
