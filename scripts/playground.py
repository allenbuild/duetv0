#!/usr/bin/env python3
"""Ego/exo playground CLI.

  playground.py create NAME --ego name=path[:person] ... --exo name=path ... [--imu stream=path.parquet] [--ref STREAM]
  playground.py run NAME [--stages probe,align,...] [--force]
  playground.py serve [--port 8765]
Episodes live under data/playground/episodes/.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from duet.playground.episode import Episode  # noqa: E402
from duet.playground.run import ORDER, run_stages  # noqa: E402
EPISODES = ROOT / "data/playground/episodes"

def parse_stream(spec, role):
    name, rest = spec.split("=", 1); path, _, person = rest.partition(":"); return (name, role, Path(path), person or None)

def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create"); c.add_argument("name"); c.add_argument("--ego", action="append", default=[]); c.add_argument("--exo", action="append", default=[])
    c.add_argument("--imu", action="append", default=[]); c.add_argument("--ref"); c.add_argument("--copy", action="store_true"); c.add_argument("--fps", type=float, default=10.0)
    r = sub.add_parser("run"); r.add_argument("name"); r.add_argument("--stages", default=",".join(ORDER)); r.add_argument("--force", action="store_true")
    s = sub.add_parser("serve"); s.add_argument("--port", type=int, default=8765); s.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    if a.cmd == "create":
        vids = [parse_stream(x, "ego") for x in a.ego] + [parse_stream(x, "exo") for x in a.exo]
        imus = {k: Path(v) for k, v in (x.split("=", 1) for x in a.imu)}
        ep = Episode.create(EPISODES, a.name, vids, imus, a.ref, link=not a.copy); ep.proc_fps = a.fps; ep.save(); print("created", ep.dir)
    elif a.cmd == "run":
        ep = Episode.load(EPISODES / a.name); run_stages(ep, a.stages.split(","), a.force)
    elif a.cmd == "serve":
        import uvicorn; from duet.playground.server import app
        uvicorn.run(app, host=a.host, port=a.port, log_level="warning")

if __name__ == "__main__":
    main()
