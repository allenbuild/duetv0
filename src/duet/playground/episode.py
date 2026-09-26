"""Episode model for the ego/exo playground.

An episode is a directory:

    <episodes_root>/<name>/
        episode.json          streams, offsets, stage status (written by us)
        streams/<stream>.mp4  the videos (copied or symlinked)
        imu/<stream>.parquet  optional IMU (Eidon 7-slot schema)
        derived/...           everything the pipeline produces

Stream roles: "ego" (head-mounted) or "exo" (fixed). One stream is the time reference;
every other stream carries ``offset_s`` such that  t_ref = t_stream + offset_s.
"""
from __future__ import annotations

import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ("probe", "align", "frames", "calib", "body2d", "hands", "objects", "tags", "headpose", "body3d", "world3d", "imu_arm", "depth_mono", "scene_scan", "qc", "export")


@dataclass
class Stream:
    name: str
    role: str  # ego | exo
    path: str  # relative to episode dir
    person: str | None = None  # who wears it (ego) or label (exo)
    fps: float = 0.0
    width: int = 0
    height: int = 0
    duration_s: float = 0.0
    has_audio: bool = False
    offset_s: float = 0.0  # t_ref = t_stream + offset_s
    offset_confidence: float | None = None
    imu: str | None = None  # relative path to IMU parquet, if any


@dataclass
class Episode:
    name: str
    root: str  # absolute episode directory
    streams: list[Stream] = field(default_factory=list)
    reference: str = ""
    proc_fps: float = 10.0
    proc_size: int = 640
    status: dict = field(default_factory=dict)  # stage -> {state, started, finished, detail}
    common_start_s: float = 0.0  # in reference time
    common_end_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def dir(self) -> Path:
        return Path(self.root)

    @property
    def derived(self) -> Path:
        d = self.dir / "derived"; d.mkdir(exist_ok=True); return d

    def stream(self, name: str) -> Stream:
        return next(s for s in self.streams if s.name == name)

    def egos(self) -> list[Stream]:
        return [s for s in self.streams if s.role == "ego"]

    def exos(self) -> list[Stream]:
        return [s for s in self.streams if s.role == "exo"]

    def set_status(self, stage: str, state: str, detail: str = "") -> None:
        st = self.status.setdefault(stage, {})
        st["state"] = state; st["detail"] = detail
        if state == "running":
            st["started"] = time.time()
        if state in ("done", "failed"):
            st["finished"] = time.time()
        self.save()

    def save(self) -> None:
        d = asdict(self); d.pop("root")
        json.dump(d, open(self.dir / "episode.json", "w"), indent=1)

    @classmethod
    def load(cls, path: Path) -> "Episode":
        d = json.load(open(Path(path) / "episode.json"))
        streams = [Stream(**s) for s in d.pop("streams")]
        return cls(root=str(Path(path).resolve()), streams=streams, **d)

    @classmethod
    def create(cls, root: Path, name: str, videos: list[tuple[str, str, Path, str | None]], imus: dict[str, Path] | None = None,
               reference: str | None = None, link: bool = True) -> "Episode":
        """videos: (stream_name, role, source_path, person). Copies or symlinks into the episode dir."""
        d = Path(root) / name; (d / "streams").mkdir(parents=True, exist_ok=True); (d / "imu").mkdir(exist_ok=True)
        ep = cls(name=name, root=str(d.resolve()))
        for sname, role, src, person in videos:
            dst = d / "streams" / f"{sname}{Path(src).suffix.lower()}"
            if not dst.exists():
                if link:
                    dst.symlink_to(Path(src).resolve())
                else:
                    import shutil; shutil.copy2(src, dst)
            ep.streams.append(Stream(name=sname, role=role, path=str(dst.relative_to(d)), person=person))
        for sname, src in (imus or {}).items():
            dst = d / "imu" / f"{sname}.parquet"
            if not dst.exists():
                import shutil; shutil.copy2(src, dst)
            ep.stream(sname).imu = str(dst.relative_to(d))
        ep.reference = reference or (ep.egos()[0].name if ep.egos() else ep.streams[0].name)
        ep.save()
        return ep


def ffprobe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,width,height,r_frame_rate,avg_frame_rate",
                          "-of", "json", str(path)], capture_output=True, text=True, check=True).stdout
    j = json.loads(out); info = {"duration_s": float(j["format"]["duration"]), "has_audio": False, "fps": 0.0, "width": 0, "height": 0}
    for s in j["streams"]:
        if s["codec_type"] == "video" and not info["width"]:
            num, den = s.get("avg_frame_rate", s.get("r_frame_rate", "30/1")).split("/")
            info["fps"] = float(num) / float(den or 1); info["width"] = int(s["width"]); info["height"] = int(s["height"])
        if s["codec_type"] == "audio":
            info["has_audio"] = True
    return info


def probe(ep: Episode) -> None:
    ep.set_status("probe", "running")
    for s in ep.streams:
        info = ffprobe(ep.dir / s.path)
        s.fps, s.width, s.height, s.duration_s, s.has_audio = info["fps"], info["width"], info["height"], info["duration_s"], info["has_audio"]
    ep.set_status("probe", "done", f"{len(ep.streams)} streams")
