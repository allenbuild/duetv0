#!/usr/bin/env python3
"""Export processed episodes as a self-contained static site (no server needed).

Writes <out>/index.html, app.js, episodes.json and, per episode, episode.json + overlay/body3d/imu JSON
and small re-encoded videos. Deployable to Vercel/Netlify/S3 as-is; the UI is the same as the live
playground minus the "run pipeline" buttons.
"""
from __future__ import annotations
import argparse, json, re, shutil, subprocess, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.playground.episode import Episode  # noqa: E402
from duet.playground.server import _clean  # noqa: E402

def export_episode(ep: Episode, out: Path, crf: int, width: int) -> dict:
    d = out / ep.name; (d / "streams").mkdir(parents=True, exist_ok=True)
    meta = json.load(open(ep.dir / "episode.json")); meta["stages"] = list(ep.status)
    qc = ep.derived / "qc" / "report.json"; meta["qc"] = json.load(open(qc)) if qc.exists() else None
    for s in ep.streams:
        dst = d / "streams" / f"{s.name}.mp4"
        if not dst.exists():
            subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(ep.dir / s.path), "-vf", f"scale='min({width},iw)':-2", "-c:v", "libx264", "-crf", str(crf),
                            "-preset", "veryfast", "-an", "-movflags", "+faststart", str(dst)], check=True)
        s_meta = next(m for m in meta["streams"] if m["name"] == s.name); s_meta["path"] = f"streams/{s.name}.mp4"
        o = {"proc_fps": ep.proc_fps, "common_start_s": ep.common_start_s, "offset_s": s.offset_s}
        z = ep.derived / "body2d" / f"{s.name}.npz"
        if z.exists(): b = np.load(z); o["body2d"] = _clean(np.round(b["kpts"], 1)); o["img_w"] = int(b["img_w"]); o["img_h"] = int(b["img_h"])
        z = ep.derived / "hands" / f"{s.name}.npz"
        if z.exists(): h = np.load(z); o["hands2d"] = _clean(np.round(h["lm2d"], 1))
        z = ep.derived / "objects" / f"{s.name}.npz"
        if z.exists(): ob = np.load(z); o["objects"] = _clean(np.round(ob["boxes"], 1)); o["object_names"] = ob["names"].tolist()
        json.dump(o, open(d / f"overlay_{s.name}.json", "w"), separators=(",", ":"))
        z = ep.derived / "imu_arm" / f"{s.name}.npz"
        if z.exists(): a = np.load(z); json.dump({"available": True, "t_s": _clean(np.round(a["t_s"], 3)), "points": _clean(np.round(a["points"], 4))}, open(d / f"imu_{s.name}.json", "w"), separators=(",", ":"))
    z = ep.derived / "body3d" / "body3d.npz"
    json.dump({"available": True, "stream": str(np.load(z)["stream"]), "world": _clean(np.round(np.load(z)["world"], 4))} if z.exists() else {"available": False}, open(d / "body3d.json", "w"), separators=(",", ":"))
    json.dump(meta, open(d / "episode.json", "w"))
    return {"name": ep.name, "streams": [{"name": s.name, "role": s.role} for s in ep.streams], "duration_s": ep.common_end_s - ep.common_start_s}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/playground/static_export"); ap.add_argument("--episodes", default="")
    ap.add_argument("--crf", type=int, default=30); ap.add_argument("--width", type=int, default=640); a = ap.parse_args()
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    root = ROOT / "data/playground/episodes"; names = a.episodes.split(",") if a.episodes else [p.parent.name for p in sorted(root.glob("*/episode.json"))]
    index = [export_episode(Episode.load(root / n), out, a.crf, a.width) for n in names]
    json.dump(index, open(out / "episodes.json", "w"))
    static = ROOT / "src/duet/playground/static"
    html = (static / "index.html").read_text()
    import re as _re
    html = _re.sub(r'<nav id="pages".*?</nav>', "", html, flags=_re.S)  # review/metrics/annotate pages need the API server, not in the static export
    html = html.replace('<button id="runAll" class="primary">Run pipeline</button>\n  <button id="rerun">Re-run all</button>', '<span class="mono" style="color:var(--mute)">static export · processed offline</span>')
    (out / "index.html").write_text(html)
    js = (static / "app.js").read_text()
    js = js.replace('await api("/api/episodes")', 'await api("episodes.json")')
    js = js.replace('await api(`/api/episode/${name}`)', 'await api(`${name}/episode.json`)')
    js = js.replace('src="/episodes/${name}/${s.path}"', 'src="${name}/${s.path}"')
    js = js.replace('api(`/api/episode/${name}/overlay/${s.name}`)', 'api(`${name}/overlay_${s.name}.json`)')
    js = js.replace('api(`/api/episode/${name}/body3d`)', 'api(`${name}/body3d.json`)')
    js = js.replace('api(`/api/episode/${name}/imu_arm/${s.name}`)', 'api(`${name}/imu_${s.name}.json`)')
    js = re.sub(r'\$\("#runAll"\)\.addEventListener[^\n]*\n', '', js); js = re.sub(r'\$\("#rerun"\)\.addEventListener[^\n]*\n', '', js)
    js = js.replace('$("#runAll").addEventListener("click", () => runPipeline(false)); $("#rerun").addEventListener("click", () => runPipeline(true));', '')
    (out / "app.js").write_text(js)
    (out / "vercel.json").write_text(json.dumps({"headers": [{"source": "/(.*)", "headers": [{"key": "Cache-Control", "value": "public, max-age=3600"}]}]}, indent=1))
    print("exported", [e["name"] for e in index], "->", out); subprocess.run(["du", "-sh", str(out)])

if __name__ == "__main__":
    main()
