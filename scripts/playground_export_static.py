#!/usr/bin/env python3
"""Export ALLOWLISTED processed episodes as a self-contained static site (no server needed).

Only episodes listed in scripts/playground_publish.json are exported; each entry records license, consent_basis,
attribution and notes. An entry is published if its license is known OR it carries an explicit "approved" record
({"by", "date" YYYY-MM-DD, "note"}), e.g. the repo owner's decision to keep an episode live while its terms are not yet
recorded (a warning is printed); an UNKNOWN license without approval is refused unless --allow-unknown-license.
--episodes must be a subset of the allowlist (default: all of it).

Published per episode: every usable stream's video trimmed to the analysed common window (+/- MARGIN_S), downscaled,
re-encoded to H.264 without audio, subtitles, data tracks or metadata; overlay / body3d / world3d / IMU-arm JSON from
the same builders as the server; episode.json with stage STATES only (no details, logs or paths), the per-stream trim
offset (trim_start_s = stream time C1 of the exported clip's t=0) and the attribution, which the page footer shows.

The site is built in a fresh directory next to --out and swapped in only when every requested episode exported (or
with --allow-partial, which drops the failed ones from the site), so removed or renamed episodes never linger;
--out/.vercel/ (the Vercel project link) is carried over. Exit codes: 0 site updated, 3 episodes failed (site
unchanged; each failure is printed with its reason), 1 configuration error, 2 usage. Encoded videos are cached in <out>.cache/, keyed by source identity (resolved path, size, mtime) and the
encode parameters, and every video is verified with ffprobe. index.html / app.js are the server UI with the
SERVER-ONLY blocks removed and the static-mode config injected (checked: no '/api/' may remain).

  scripts/playground_export_static.py --out data/playground/static_export [--episodes a,b]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.playground import runtime, server  # noqa: E402
from duet.playground.episode import NAME_RE, Episode  # noqa: E402

PUBLISH = ROOT / "scripts" / "playground_publish.json"
STATIC = ROOT / "src" / "duet" / "playground" / "static"
REQUIRED = ("license", "consent_basis", "attribution", "notes")
EXPORT_VERSION = 3  # part of the video cache key; bump when the encode recipe changes (3: absolute coarse + exact seek, measured trim)
MARGIN_S = 0.5  # seconds of video kept on each side of the common window (seeking at the edges)
SEEK_MARGIN_S = 5.0  # coarse input seek this far before the trim point; the exact cut is an output-side seek
JS_BLOCK, HTML_BLOCK = ("/* SERVER-ONLY BEGIN */", "/* SERVER-ONLY END */"), ("<!-- SERVER-ONLY BEGIN -->", "<!-- SERVER-ONLY END -->")
STATIC_MARKER = "<!-- STATIC-CONFIG -->"
MARKER = ".duet-playground-export"  # written into every site this exporter builds; a non-empty --out without it is never replaced
API_REF = re.compile(r"\bapi/(?:episodes?|upload|session|queue)\b")  # server endpoints (relative or absolute) must not ship
SCRIPT_TAG = '<script src="app.js"></script>'
OLD_FILES = {"index.html", "app.js", "episodes.json", "vercel.json", "version.json", "robots.txt"}
OLD_EPISODE_FILE = re.compile(r"(episode|body3d|world3d)\.json|(overlay|imu)_[A-Za-z0-9_-]+\.json")
NOTICE = "Static export, processed offline (research preview)."
EXIT_EPISODES_FAILED = 3
SITE_ORIGIN = "https://plg.duetlabs.co"  # canonical origin: the only one granted cross-origin reads (replaces Vercel's default "*")
CSP = ("default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def vercel_config(site_origin: str = SITE_ORIGIN) -> dict:
    """vercel.json: security headers, cache policy, and Access-Control-Allow-Origin pinned to the site's own origin
    (same-origin requests never need it, so no other origin can read the data cross-origin)."""
    return {"$schema": "https://openapi.vercel.sh/vercel.json", "headers": [
        {"source": "/(.*)", "headers": [{"key": "Access-Control-Allow-Origin", "value": site_origin},
                                        {"key": "Content-Security-Policy", "value": CSP}, {"key": "X-Frame-Options", "value": "DENY"},
                                        {"key": "X-Content-Type-Options", "value": "nosniff"}, {"key": "Referrer-Policy", "value": "no-referrer"},
                                        {"key": "X-Robots-Tag", "value": "noindex, nofollow, noarchive"},
                                        {"key": "Permissions-Policy", "value": "camera=(), microphone=(), geolocation=()"}]},
        {"source": "/", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
        {"source": "/(.*)\\.(html|js|json|txt)", "headers": [{"key": "Cache-Control", "value": "no-cache"}]},
        {"source": "/(.*)\\.mp4", "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}]}]}  # names carry a content key
_ABS_PATH = re.compile(r"(?<![\w.~-])(?:~|[A-Za-z]:)?[/\\](?:[^\s'\"<>|/\\]+[/\\])+[^\s'\"<>|]*")


class ExportError(Exception):
    """One episode could not be exported (the others continue)."""


def sanitize(text: object, limit: int = 500) -> str:
    """Public text: absolute paths replaced by <path>, length capped."""
    return _ABS_PATH.sub("<path>", str(text))[:limit]


def _sanitize_tree(x: object) -> object:
    if isinstance(x, dict):
        return {str(k): _sanitize_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_sanitize_tree(v) for v in x]
    return sanitize(x) if isinstance(x, str) else x


def load_allowlist(path: Path) -> dict[str, dict[str, str]]:
    """{episode: {license, consent_basis, attribution, notes, ...}}; SystemExit when missing or incomplete."""
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        raise SystemExit(f"cannot read the publish allowlist {path}: {e}") from None
    eps = d.get("episodes") if isinstance(d, dict) else None
    if not isinstance(eps, dict) or not eps:
        raise SystemExit(f'{path}: needs a non-empty "episodes" object')
    for name, meta in eps.items():
        if not NAME_RE.fullmatch(name):  # keys become paths under the build dir: never "..", "/", ""
            raise SystemExit(f"{path}: {name!r} is not a valid episode name ({NAME_RE.pattern})")
        missing = [k for k in REQUIRED if not (isinstance(meta, dict) and isinstance(meta.get(k), str) and meta[k].strip())]
        if missing:
            raise SystemExit(f"{path}: episode {name!r} lacks {missing}")
        ok = meta.get("approved")
        if ok is not None and not (isinstance(ok, dict) and isinstance(ok.get("by"), str) and ok["by"].strip()
                                   and isinstance(ok.get("date"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", ok["date"])
                                   and isinstance(ok.get("note", ""), str)):
            raise SystemExit(f'{path}: episode {name!r}: "approved" must be {{"by": <who>, "date": "YYYY-MM-DD", "note": <why>}}')
    return eps


def unknown(v: str) -> bool:
    return v.strip().upper().startswith("UNKNOWN")


def refusal(name: str, meta: dict, allow_unknown_license: bool = False) -> str | None:
    """Why an allowlisted entry may not be published (None = publish). Prints the warnings for what is published anyway."""
    for k in REQUIRED:
        if unknown(meta[k]):
            print(f"WARNING {name}: {k} is {meta[k]!r}", file=sys.stderr)
    if not unknown(meta["license"]):
        return None
    ok = meta.get("approved")
    if ok:
        print(f"WARNING {name}: publishing with license UNKNOWN on the explicit approval by {ok['by']} ({ok['date']}): {ok.get('note', '')}", file=sys.stderr)
        return None
    if allow_unknown_license:
        print(f"WARNING {name}: publishing with license UNKNOWN because of --allow-unknown-license", file=sys.stderr)
        return None
    return ('license is UNKNOWN and the entry has no "approved" record: record the terms or an explicit approval in '
            "scripts/playground_publish.json (or pass --allow-unknown-license)")


def _old_export_strays(o: Path) -> list[str]:
    """Entries that an export of the old exporter (no marker) cannot contain; empty = safe to adopt and replace."""
    bad = []
    for p in o.iterdir():
        if p.name == ".vercel" and p.is_dir() and not p.is_symlink():
            continue
        if p.is_file() and not p.is_symlink() and p.name in OLD_FILES:
            continue
        if p.is_dir() and not p.is_symlink() and NAME_RE.fullmatch(p.name) and (p / "episode.json").is_file():
            for q in p.iterdir():
                ok = (q.is_file() and not q.is_symlink() and OLD_EPISODE_FILE.fullmatch(q.name)) or (
                    q.name == "streams" and q.is_dir() and not q.is_symlink()
                    and all(v.is_file() and not v.is_symlink() and v.suffix == ".mp4" for v in q.iterdir()))
                if not ok:
                    bad.append(f"{p.name}/{q.name}")
            continue
        bad.append(p.name)
    return bad


def check_out(out: Path, episodes_root: Path, adopt_old: bool = False) -> None:
    """Refuse output locations the swap could destroy: /, $HOME, the repo or an ancestor of it, source dirs, the
    episodes, and any existing non-empty directory without this exporter's marker file. An export of the old exporter
    (no marker) is replaced only with adopt_old AND when every entry fits the old site layout (.vercel/ is kept)."""
    o, repo, eps = out.resolve(), ROOT.resolve(), episodes_root.resolve()
    if o in (Path("/"), Path.home().resolve()) or repo.is_relative_to(o) or o.is_relative_to(eps) or eps.is_relative_to(o) \
            or any(o.is_relative_to(repo / d) for d in ("src", "scripts", "tests", "docs", ".git", "data/raw")):
        raise SystemExit(f"refusing --out {out}: not a safe place for a generated site")
    if o.exists():
        if not o.is_dir() or out.is_symlink():
            raise SystemExit(f"refusing --out {out}: exists and is not a plain directory")
        if any(o.iterdir()) and not (o / MARKER).is_file():
            if not adopt_old:
                raise SystemExit(f"refusing --out {out}: not empty and not marked as a playground export ({MARKER} missing). Use a new or "
                                 "empty directory, or pass --adopt-old-export if it is the output of the previous exporter.")
            strays = _old_export_strays(o)
            if strays:
                raise SystemExit(f"refusing to adopt {out}: it holds files an old playground export never has: {', '.join(strays[:8])}")


def _rmtree_inside(target: Path, parent: Path) -> None:
    """shutil.rmtree, only for a path strictly inside `parent` (both resolved; a symlink pointing out is refused)."""
    t, p = target.resolve(), parent.resolve()
    if t == p or not t.is_relative_to(p):
        raise RuntimeError(f"refusing to delete {target}: not inside {parent}")
    shutil.rmtree(t, ignore_errors=True)


def stale_stages(ep: Episode) -> list[str]:
    """Stages marked done whose results are no longer current (run.stage_fresh): their outputs are not published."""
    fs = server.fresh_set(ep)
    return [st for st in ("calib", "body2d", "hands", "objects", "headpose", "body3d", "world3d", "imu_arm", "qc") if ep.stage_ok(st) and st not in fs]


def preflight(ep: Episode) -> list[str]:
    """Problems that make an episode unexportable (all of them, so they can be fixed in one go)."""
    problems, name = [], ep.dir.name
    try:
        ep.n_frames()
    except ValueError as e:
        problems.append(str(e))
    for s in ep.streams:
        if not ep.usable(s):
            continue
        p = ep.dir / s.path
        if not p.is_file():
            problems.append(f"stream {s.name}: source video {name}/{s.path} missing" + (f" (dangling symlink -> {os.readlink(p)})" if p.is_symlink() else ""))
        elif not s.duration_s > 0:
            problems.append(f"stream {s.name}: not probed (run the probe stage)")
    return problems


def _verify(path: Path) -> float:
    """ffprobe the exported clip: one H.264 video stream, no audio, duration > 0. Returns the duration (s)."""
    info = runtime.ffprobe_json(path, "-show_entries", "format=duration:stream=codec_type,codec_name,width,height")
    kinds = [s.get("codec_type") for s in info.get("streams", [])]
    duration = float((info.get("format") or {}).get("duration") or 0)
    if kinds != ["video"] or not duration > 0 or not int(info["streams"][0].get("width") or 0) > 0:
        raise ExportError(f"{path.name}: encoded clip failed verification (streams {kinds}, duration {duration})")
    return duration


def encode_stream(ep: Episode, s, dst_dir: Path, cache: Path, crf: int, width: int, used: set[str]) -> dict:
    """Trim stream s to the common window (+/- MARGIN_S), downscale, re-encode (cached by source identity).

    Seeking: a coarse ABSOLUTE input seek (-seek_timestamp 1; by default ffmpeg adds the file's start_time, which put
    MPEG-TS / offset MKV clips 1.5-2 s late) SEEK_MARGIN_S early (demuxer seeks, e.g. MPEG-TS, can land late), then an
    exact output-side -ss (decode and discard). trim_start_s is measured, not assumed: the stream time of the clip's
    first frame (P, from the source) minus that frame's time in the clip (Q)."""
    src = ep.dir / s.path
    t0 = max(0.0, float(ep.stream_time(s, ep.common_start_s)) - MARGIN_S)
    t1 = min(float(s.duration_s), float(ep.stream_time(s, ep.common_end_s)) + MARGIN_S)
    if not t1 - t0 > 0.1:
        raise ExportError(f"stream {s.name}: common window lies outside the video ({t0:.2f}-{t1:.2f} s)")
    vstart = float(s.video_start_s or 0.0)
    ss = t0 + vstart  # absolute container time of stream time t0 (C1: t_stream 0 = first video frame)
    coarse = max(ss - SEEK_MARGIN_S, 0.0)
    vmap = f"0:{s.video_index}" if s.video_index is not None else "0:v:0"
    st = src.stat()
    key = hashlib.sha256(json.dumps({"src": os.path.realpath(src), "size": st.st_size, "mtime_ns": st.st_mtime_ns, "ss": round(ss, 6),
                                     "t": round(t1 - t0, 6), "map": vmap, "width": width, "crf": crf, "v": EXPORT_VERSION},
                                    sort_keys=True).encode()).hexdigest()
    cached, meta = cache / f"{key}.mp4", cache / f"{key}.json"
    try:
        info = json.loads(meta.read_text()) if cached.is_file() else None
        duration = _verify(cached) if info else None
    except (ExportError, RuntimeError, ValueError, KeyError, OSError):
        info = duration = None
    if duration is None:
        tmp = cache / f"{key}.tmp-{os.getpid()}.mp4"
        try:
            runtime.run_ffmpeg(["-seek_timestamp", "1", "-ss", f"{coarse:.6f}", "-i", "file:" + os.path.abspath(src), "-ss", f"{ss - coarse:.6f}",
                                "-t", f"{t1 - t0:.6f}", "-map", vmap, "-vf", f"scale='min({width},iw)':-2", "-c:v", "libx264", "-preset", "veryfast",
                                "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", "-sn", "-dn", "-map_metadata", "-1", "-map_chapters", "-1",
                                "-movflags", "+faststart", str(tmp)])
            duration = _verify(tmp)
            info = {"trim_start_s": _measure_trim(src, s, ss, vstart, tmp)}
            os.replace(tmp, cached)
            meta.write_text(json.dumps(info))
        finally:
            tmp.unlink(missing_ok=True)
    used.update({cached.name, meta.name})
    rel = f"streams/{s.name}-{key[:12]}.mp4"
    try:
        os.link(cached, dst_dir.parent / rel)
    except OSError:
        shutil.copy2(cached, dst_dir.parent / rel)
    return {"path": rel, "trim_start_s": round(float(info["trim_start_s"]), 6), "clip_duration_s": round(duration, 3), "bytes": cached.stat().st_size}


def _first_pts(path: Path, select: str, interval: str, at: float | None = None) -> float | None:
    """Smallest video packet pts (container seconds) in `interval` (ffprobe -read_intervals, absolute times), >= at."""
    info = runtime.ffprobe_json(path, "-select_streams", select, "-read_intervals", interval, "-show_entries", "packet=pts_time")
    ts = [float(p["pts_time"]) for p in info.get("packets", []) if p.get("pts_time") not in (None, "N/A")]
    ts = [t for t in ts if at is None or t >= at - 1e-4]
    return min(ts) if ts else None


def _measure_trim(src: Path, s, ss: float, vstart: float, clip: Path) -> float:
    """Stream time shown at the clip's t = 0: (P - vstart) - Q, P = pts of the first source frame at/after the trim
    point ss (read from the source), Q = pts of the clip's first frame. Q must equal P - ss up to the output frame grid
    (half a frame): a larger Q means the seek landed late and frames are missing -> ExportError."""
    P = _first_pts(src, str(s.video_index) if s.video_index is not None else "v:0", f"{max(ss - 1.0, 0.0):.6f}%+3", at=ss)
    Q = _first_pts(clip, "v:0", "%+#30")
    half = 0.5 / (s.fps if s.fps and s.fps > 0 else 30.0) + 2e-3  # + timebase rounding
    if P is None or Q is None or abs(Q - (P - ss)) > half:
        raise ExportError(f"stream {s.name}: clip starts at {Q} s but the first source frame after the trim point is "
                          f"{None if P is None else round(P - ss, 4)} s after it (tolerance {half:.4f} s): frames are missing")
    return (P - vstart) - Q


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def export_episode(ep: Episode, pub: dict, build: Path, cache: Path, crf: int, width: int, used: set[str]) -> dict:
    """Write <build>/<episode>/... and return its episodes.json entry."""
    name, d = ep.dir.name, build / ep.dir.name
    (d / "streams").mkdir(parents=True)
    n = ep.n_frames()
    streams, skipped = [], [s.name for s in ep.streams if not ep.usable(s)]
    for s in ep.streams:
        if s.name in skipped:
            continue
        v = encode_stream(ep, s, d / "streams", cache, crf, width, used)
        imu = server.imu_arm_payload(ep, s) if s.imu else {"available": False}
        if imu.get("available"):
            _write(d / f"imu_{s.name}.json", server.dumps(imu))
        _write(d / f"overlay_{s.name}.json", server.dumps(server.overlay_payload(ep, s)))
        streams.append({"name": s.name, "role": s.role, "person": s.person, "fps": s.fps, "width": s.width, "height": s.height,
                        "offset_s": s.offset_s, "offset_status": s.offset_status, "drift_ppm": s.drift_ppm, "video_start_s": 0.0,
                        "trim_start_s": v["trim_start_s"], "clip_duration_s": v["clip_duration_s"], "path": v["path"], "usable": True,
                        "imu": bool(imu.get("available"))})
    if not streams:
        raise ExportError("no usable (aligned) stream to publish")
    _write(d / "body3d.json", server.dumps(server.body3d_payload(ep)))
    _write(d / "world3d.json", server.dumps(server.world3d_payload(ep)))
    qc_path = ep.dir / "derived" / "qc" / "report.json"
    try:
        qc = _sanitize_tree(json.loads(qc_path.read_text())) if "qc" in server.fresh_set(ep) and qc_path.is_file() else None
    except (OSError, ValueError):
        qc = None
    stale = stale_stages(ep)
    notes = [sanitize(x) for x in ep.notes] + [f"not published (unaligned): {', '.join(skipped)}"] * bool(skipped) + \
        [f"not published (stale results; re-run the pipeline): {', '.join(stale)}"] * bool(stale)
    meta = {"name": name, "reference": ep.reference, "proc_fps": ep.proc_fps, "common_start_s": ep.common_start_s, "common_end_s": ep.common_end_s,
            "n_frames": n, "streams": streams,
            "status": {k: {"state": "stale" if k in stale else (v or {}).get("state")} for k, v in ep.status.items() if not k.startswith("_")},
            "stages": server.stage_list(ep), "running": False, "qc": qc, "notes": notes,
            "license": pub["license"], "attribution": pub["attribution"]}
    _write(d / "episode.json", server.dumps(meta))
    return {"name": name, "streams": [{"name": s["name"], "role": s["role"]} for s in streams], "duration_s": ep.common_end_s - ep.common_start_s,
            "license": pub["license"], "attribution": pub["attribution"], "video_bytes": sum((d / s["path"]).stat().st_size for s in streams),
            "stale": stale}


def strip_blocks(text: str, begin: str, end: str, what: str) -> str:
    """Remove every begin...end block; the markers must pair up (an unchecked rewrite once shipped a broken site)."""
    if text.count(begin) == 0 or text.count(begin) != text.count(end):
        raise SystemExit(f"{what}: SERVER-ONLY markers missing or unbalanced")
    out = re.sub(re.escape(begin) + r".*?" + re.escape(end), "", text, flags=re.S)
    if begin in out or end in out:
        raise SystemExit(f"{what}: SERVER-ONLY markers nested or out of order")
    return out


def git_version() -> dict:
    """HEAD commit and whether the playground sources differ from it (read-only git)."""
    def git(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=30)
    try:
        head = git("rev-parse", "HEAD")
        dirty = git("status", "--porcelain", "--untracked-files=all", "--", "src/duet/playground", "scripts/playground_export_static.py",
                    "scripts/playground_publish.json")
        return {"git_commit": head.stdout.strip() if head.returncode == 0 else "unknown", "git_dirty": bool(dirty.stdout.strip())}
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_dirty": True}


def write_site_files(build: Path, index: list[dict], version: dict, site_origin: str = SITE_ORIGIN) -> None:
    """index.html + app.js in static mode, episodes.json, version.json, vercel.json, robots.txt."""
    tag = version["git_commit"][:12] + ("-dirty" if version["git_dirty"] else "")
    cfg = {"static": True, "version": tag, "notice": NOTICE,
           "attributions": [{"episode": e["name"], "attribution": e["attribution"], "license": e["license"]} for e in index]}
    html = strip_blocks((STATIC / "index.html").read_text(), *HTML_BLOCK, "index.html")
    js = strip_blocks((STATIC / "app.js").read_text(), *JS_BLOCK, "app.js")
    if html.count(STATIC_MARKER) != 1:
        raise SystemExit(f"index.html: expected exactly one {STATIC_MARKER}")
    blob = json.dumps(cfg, separators=(",", ":")).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    html = html.replace(STATIC_MARKER, f'<script id="playground-static" type="application/json">{blob}</script>')
    if html.count(SCRIPT_TAG) != 1:
        raise SystemExit(f"index.html: expected exactly one {SCRIPT_TAG}")
    html = html.replace(SCRIPT_TAG, f'<script src="app.js?v={hashlib.sha256(js.encode()).hexdigest()[:12]}"></script>')  # cache-busting
    for fname, text in (("index.html", html), ("app.js", js)):
        if API_REF.search(text):
            raise SystemExit(f"{fname}: a server API reference ({API_REF.search(text).group(0)}) is left after removing the SERVER-ONLY blocks")
    if 'id="uploadForm"' in html or 'integrity="sha384-' not in html:
        raise SystemExit("index.html: upload form left in, or the three.js <script> lost its SRI hash")
    _write(build / "index.html", html)
    _write(build / "app.js", js)
    _write(build / "episodes.json", json.dumps([{k: e[k] for k in ("name", "streams", "duration_s", "license", "attribution")} for e in index]))
    _write(build / "version.json", json.dumps({**version, "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                               "episodes": [e["name"] for e in index], "exporter": "scripts/playground_export_static.py"}, indent=1))
    _write(build / "vercel.json", json.dumps(vercel_config(site_origin), indent=1))
    _write(build / "robots.txt", "User-agent: *\nDisallow: /\n")
    _write(build / MARKER, json.dumps({"exporter": "scripts/playground_export_static.py", "format": 1}))


def swap_in(build: Path, out: Path) -> None:
    """Replace `out` by `build` (two renames in one directory), carrying over out/.vercel. check_out has verified that
    an existing `out` is ours (marker) or an adoptable old export."""
    assert (build / MARKER).is_file(), "build dir lacks the export marker"
    if (out / ".vercel").is_dir():
        shutil.copytree(out / ".vercel", build / ".vercel", symlinks=True)
    old = out.with_name(f".{out.name}.old-{os.getpid()}-{secrets.token_hex(4)}")
    if out.exists():
        os.rename(out, old)
    try:
        os.rename(build, out)
    except OSError:
        if old.exists():
            os.rename(old, out)
        raise
    if old.exists():
        _rmtree_inside(old, out.parent)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default="data/playground/static_export", help="site directory (relative paths are relative to the repo)")
    ap.add_argument("--episodes", default="", help="comma-separated subset of the allowlist (default: every allowlisted episode)")
    ap.add_argument("--episodes-root", default=None, help="default: env PLAYGROUND_EPISODES, else data/playground/episodes")
    ap.add_argument("--publish", default=str(PUBLISH), help="allowlist JSON (default scripts/playground_publish.json)")
    ap.add_argument("--crf", type=int, default=30); ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--allow-unknown-license", action="store_true", help="also publish episodes whose license is UNKNOWN")
    ap.add_argument("--allow-partial", action="store_true", help="publish the episodes that exported even if others failed (the failed ones "
                                                                   "disappear from the site)")
    ap.add_argument("--adopt-old-export", action="store_true", help=f"replace an --out written by the previous exporter (no {MARKER}) if it "
                                                                      "holds nothing but an old site")
    ap.add_argument("--site-origin", default=SITE_ORIGIN, help=f"origin allowed in Access-Control-Allow-Origin (default {SITE_ORIGIN})")
    a = ap.parse_args(argv)
    out = Path(a.out) if Path(a.out).is_absolute() else ROOT / a.out
    root = Path(a.episodes_root) if a.episodes_root else runtime.episodes_root()
    publish = Path(a.publish) if Path(a.publish).is_absolute() else ROOT / a.publish
    if not re.fullmatch(r"https?://[A-Za-z0-9.-]+(:\d{1,5})?", a.site_origin):
        raise SystemExit(f"--site-origin must be an origin like https://plg.duetlabs.co, got {a.site_origin!r}")
    allow = load_allowlist(publish)
    names = [n.strip() for n in a.episodes.split(",") if n.strip()] or list(allow)
    outside = [n for n in names if n not in allow]
    if outside:
        raise SystemExit(f"not in the publish allowlist {publish.name}: {outside} (add them with license/consent/attribution first)")
    check_out(out, root, a.adopt_old_export)
    build = out.with_name(f".{out.name}.build-{os.getpid()}-{secrets.token_hex(4)}")
    cache = out.with_name(out.name + ".cache")
    build.mkdir(parents=True); cache.mkdir(parents=True, exist_ok=True)
    index, failed, used = [], {}, set()
    try:
        for n in names:
            pub = allow[n]
            try:
                why = refusal(n, pub, a.allow_unknown_license)
                if why:
                    raise ExportError(why)
                ep = Episode.load(root / n)
                problems = preflight(ep)
                if problems:
                    raise ExportError("; ".join(problems))
                index.append(export_episode(ep, pub, build, cache, a.crf, a.width, used))
            except Exception as e:  # noqa: BLE001 - report every episode, then decide
                failed[n] = f"{type(e).__name__}: {e}" if not isinstance(e, ExportError) else str(e)
                if (build / n).exists():
                    _rmtree_inside(build / n, build)
        print(f"static export -> {out}")
        for e in index:
            print(f"  ok      {e['name']:28} {len(e['streams'])} streams, {e['duration_s']:.1f} s window, {e['video_bytes'] / 1e6:.1f} MB video"
                  + (f"; stale, not published: {', '.join(e['stale'])}" if e["stale"] else ""))
        for n, why in failed.items():
            print(f"  FAILED  {n:28} {why}")
        sys.stdout.flush()  # keep the summary above the stderr verdict when both go to one terminal/log
        if not index or (failed and not a.allow_partial):
            print(f"site NOT updated (unchanged): {len(failed)} of {len(names)} requested episodes failed: {', '.join(failed)} (reasons above).",
                  file=sys.stderr)
            if index:
                print(f"To publish only {', '.join(e['name'] for e in index)}, re-run with --allow-partial; {', '.join(failed)} would then "
                      "DISAPPEAR from the site.", file=sys.stderr)
            return EXIT_EPISODES_FAILED
        write_site_files(build, index, git_version(), a.site_origin)
        swap_in(build, out)
    finally:
        if build.exists():
            _rmtree_inside(build, out.parent)
    for p in [*cache.glob("*.mp4"), *cache.glob("*.json")]:  # keep only what the published site uses
        if p.name not in used:
            p.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
