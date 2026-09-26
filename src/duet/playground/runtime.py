"""Runtime helpers shared by the playground stages: compute device, safe ffmpeg/ffprobe calls, atomic
writes, eager npz loading and JSON hygiene.

Nothing here imports cv2/torch at module level, so ``import duet.playground.run`` works in a minimal
environment (numpy/pandas/pyarrow only).

ffmpeg on user media: every ffmpeg/ffprobe call on user-supplied files puts ``FFMPEG_SAFE_INPUT`` before
``-i`` (only the ``file`` protocol; only the mov/mp4, matroska/webm, avi and mpegts demuxers, so HLS/concat
playlists disguised as videos cannot make ffmpeg read other files or URLs). ``run_ffmpeg`` inserts it
automatically when the caller did not; ``media_input(path)`` gives the ``-i file:<abs path>`` pair.
"""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import io
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np

REPO = Path(__file__).resolve().parents[3]
FFMPEG_DEMUXERS = "mov,mp4,m4a,3gp,3g2,mj2,matroska,webm,avi,mpegts"
FFMPEG_SAFE_INPUT: list[str] = ["-protocol_whitelist", "file", "-format_whitelist", FFMPEG_DEMUXERS]
_DEVICE_RE = re.compile(r"^(auto|cpu|mps|cuda(:\d+)?)$")
HWACCELS = ("auto", "none", "videotoolbox", "cuda", "vaapi", "qsv", "d3d11va", "dxva2")


def models_dir() -> Path:
    """Model weights directory: env PLAYGROUND_MODELS ("~" expanded), else <repo>/data/playground/models."""
    return Path(os.environ.get("PLAYGROUND_MODELS") or REPO / "data/playground/models").expanduser()


def episodes_root() -> Path:
    """Episodes directory: env PLAYGROUND_EPISODES ("~" expanded), else <repo>/data/playground/episodes."""
    return Path(os.environ.get("PLAYGROUND_EPISODES") or REPO / "data/playground/episodes").expanduser()


# ----------------------------------------------------------------------------- model weights

# name -> (pinned URL, sha256). MediaPipe hashes = the files committed under data/playground/models (identical to
# these URLs on 2026-09-26); ultralytics hashes from the v8.3.0 assets release.
MODEL_FILES: dict[str, tuple[str, str]] = {
    "hand_landmarker.task": ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
                             "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1"),
    "pose_landmarker_lite.task": ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
                                  "59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a"),
    "yolov8n-pose.pt": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-pose.pt",
                        "c6fa93dd1ee4a2c18c900a45c1d864a1c6f7aba75d84f91648a30b7fb641d212"),
    "yolov8s-worldv2.pt": ("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s-worldv2.pt",
                           "9b2c17ab6124a913e9b3a5c170617920d91b0f01111a8479da69f00e2cf27792"),
}
# Hugging Face repos at a pinned commit: (repo_id, revision) -> {file: sha256}; fetched file by file from
# https://huggingface.co/<repo>/resolve/<revision>/<file> into hf_weights_dir(repo, revision) (depth_mono.MODELS; sha256
# of model.safetensors = the LFS pointer's, the JSON files hashed on 2026-09-26).
HF_SNAPSHOTS: dict[tuple[str, str], dict[str, str]] = {
    ("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99"): {
        "config.json": "c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75",
        "preprocessor_config.json": "d41175c0d889477ca8fc67191e540faef14baf6275157b3fdecf78469e6bbf84",
        "model.safetensors": "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"},
    ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "8078d68a9c75a972131914f6afd0c1723be0da7f"): {
        "config.json": "0b1d9fc591693d16b864249a9cfcf5264c8c00be999c1643fa1c64dc105d55f8",
        "preprocessor_config.json": "533b16a60445d7cab5086d39b45f92be45624b977972c62d5984d93e98366063",
        "model.safetensors": "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3"},
}


def hf_dirname(model_id: str, revision: str | None) -> str:
    return f"{model_id.replace('/', '--')}@{revision}"


def hf_weights_dir(model_id: str, revision: str | None) -> Path:
    """Pre-fetched Hugging Face snapshot: <models_dir>/<model id, "/" -> "--">@<revision>/."""
    return models_dir() / hf_dirname(model_id, revision)


MODEL_FILES.update({f"{hf_dirname(repo, rev)}/{f}": (f"https://huggingface.co/{repo}/resolve/{rev}/{f}", sha)
                    for (repo, rev), files in HF_SNAPSHOTS.items() for f, sha in files.items()})
_DEPTH_RELATIVE = ("depth-anything/Depth-Anything-V2-Small-hf", "5426e4f0f36572d16453bbda7a8389317b1bef99")
_DEPTH_METRIC_INDOOR = ("depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", "8078d68a9c75a972131914f6afd0c1723be0da7f")
# fetch-models groups; "default" is fetched when no names are given (the metric-indoor depth model is optional)
MODEL_GROUPS: dict[str, list[str]] = {
    "default": ["hand_landmarker.task", "pose_landmarker_lite.task", "yolov8n-pose.pt", "yolov8s-worldv2.pt",
                *[f"{hf_dirname(*_DEPTH_RELATIVE)}/{f}" for f in HF_SNAPSHOTS[_DEPTH_RELATIVE]]],
    "depth-metric-indoor": [f"{hf_dirname(*_DEPTH_METRIC_INDOOR)}/{f}" for f in HF_SNAPSHOTS[_DEPTH_METRIC_INDOOR]],
}


def model_path(name: str) -> Path:
    """Path of a weights file in models_dir(); FileNotFoundError with the fix when it is missing."""
    p = models_dir() / name
    if not p.is_file():
        raise FileNotFoundError(f"model file {name} not found in {models_dir()}: run `python scripts/playground.py fetch-models`")
    return p


def file_sha256(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def expand_model_names(names: list[str] | None) -> list[str]:
    """Group names (MODEL_GROUPS) -> file keys; no names = the "default" group."""
    return list(dict.fromkeys(f for n in (names or ["default"]) for f in MODEL_GROUPS.get(n, [n])))


def fetch_models(names: list[str] | None = None, force: bool = False, log: Callable[[str], Any] = print,
                 opener: Callable[..., Any] | None = None, timeout: float = 120) -> dict[str, str]:
    """Download pinned model files (names or MODEL_GROUPS keys; default group "default") into models_dir(), verifying
    SHA-256 (a mismatch discards the download and raises). Present files with the right hash are kept; a present file
    with a different hash is an error unless ``force``. Returns {name: "ok" | "downloaded"}."""
    import urllib.request
    opener = opener or urllib.request.urlopen
    names = expand_model_names(names)
    bad = [n for n in names if n not in MODEL_FILES]
    if bad:
        raise ValueError(f"unknown model(s) {bad}; groups: {sorted(MODEL_GROUPS)}; files: {sorted(MODEL_FILES)}")
    out_dir = models_dir(); out_dir.mkdir(parents=True, exist_ok=True); res: dict[str, str] = {}
    for n in names:
        url, want = MODEL_FILES[n]; dst = out_dir / n
        if dst.is_file():
            have = file_sha256(dst)
            if have == want:
                res[n] = "ok"; log(f"{n}: ok"); continue
            if not force:
                raise RuntimeError(f"{dst} exists with sha256 {have[:12]}..., expected {want[:12]}... (use --force to replace)")
        log(f"{n}: downloading {url}")

        def write(f: io.BufferedWriter, n: str = n, url: str = url, want: str = want) -> None:
            h = hashlib.sha256()
            with opener(url, timeout=timeout) as r:
                while b := r.read(1 << 20):
                    h.update(b); f.write(b)
            if h.hexdigest() != want:
                raise RuntimeError(f"{n}: sha256 mismatch (got {h.hexdigest()}, expected {want}); download discarded")
        atomic_write(dst, write)
        res[n] = "downloaded"; log(f"{n}: downloaded, sha256 verified")
    return res


# ----------------------------------------------------------------------------- device

def valid_device(pref: str) -> bool:
    return bool(_DEVICE_RE.fullmatch(str(pref or "")))


def pick_device(pref: str = "auto") -> str:
    """Torch device string: explicit ``pref`` honoured; "auto" -> "cuda:0" if CUDA, else "mps", else "cpu".

    torch is imported lazily; without torch the answer is "cpu"."""
    pref = (pref or "auto").strip()
    if not _DEVICE_RE.fullmatch(pref):
        raise ValueError(f"device must be auto|cpu|mps|cuda[:N], got {pref!r}")
    if pref != "auto":
        return pref
    try:
        import torch
    except Exception:  # noqa: BLE001 - torch missing or broken: CPU
        return "cpu"
    if torch.cuda.is_available():
        return "cuda:0"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


# ----------------------------------------------------------------------------- display matrix

# sign pattern (a, b, c, d) of the normalised 2x2 part of an FFmpeg display matrix -> (rotation, flip), measured
# against ffmpeg 9 autorotate for all 8 -display_rotation x -display_hflip x -display_vflip combinations and H.264
# display-orientation SEI (tests/test_playground_core.py checks every case pixel by pixel)
_DISPLAY: dict[tuple[int, int, int, int], tuple[int, bool]] = {
    (1, 0, 0, 1): (0, False), (-1, 0, 0, -1): (180, False), (0, -1, 1, 0): (90, False), (0, 1, -1, 0): (270, False),
    (1, 0, 0, -1): (0, True), (-1, 0, 0, 1): (180, True), (0, 1, 1, 0): (270, True), (0, -1, -1, 0): (90, True)}


def display_transform(matrix: Any) -> tuple[int, bool]:
    """(rotation, flip) of an FFmpeg 3x3 display matrix (9 int32, 2x2 part in 16.16 fixed point; None = identity).

    rotation = round(av_display_rotation_get(M)) mod 360 (ffprobe's "rotation", counter-clockwise, 0/90/180/270);
    flip = the matrix is a reflection (det < 0). The displayed image - exactly ffmpeg's autorotate - is
    ``np.rot90(decoded[::-1] if flip else decoded, rotation // 90)`` (vertical flip FIRST, then rotate); a plain
    horizontal mirror is (180, True). Width/height swap iff rotation is 90 or 270. Matrices that are not a multiple of
    90 degrees count as identity (ffmpeg does not autorotate them either)."""
    if matrix is None:
        return 0, False
    m = [float(x) / 65536 for x in list(matrix)[:5]]
    a, b, c, d = m[0], m[1], m[3], m[4]
    s0, s1 = math.hypot(a, c), math.hypot(b, d)
    if not (s0 and s1):
        return 0, False
    v = (a / s0, b / s1, c / s0, d / s1)
    if any(abs(x - round(x)) > 1e-3 for x in v):
        return 0, False
    return _DISPLAY.get(tuple(int(round(x)) for x in v), (0, False))  # type: ignore[arg-type]


def parse_displaymatrix(text: str | None) -> list[int] | None:
    """ffprobe's "displaymatrix" dump ("00000000:  0  65536  0\n00000001: ...") -> 9 ints (None if absent/garbled)."""
    if not text:
        return None
    vals = [int(x) for line in str(text).strip().splitlines() if ":" in line for x in line.split(":", 1)[1].split()]
    return vals if len(vals) == 9 else None


# ----------------------------------------------------------------------------- ffmpeg

def media_input(path: str | os.PathLike) -> list[str]:
    """``[*FFMPEG_SAFE_INPUT, "-i", "file:<absolute path>"]`` for one user-supplied media file."""
    return [*FFMPEG_SAFE_INPUT, "-i", "file:" + os.path.abspath(path)]


def _guard_inputs(args: list[str]) -> list[str]:
    """Insert FFMPEG_SAFE_INPUT before every ``-i`` unless the caller already whitelisted protocols."""
    if "-protocol_whitelist" in args:
        return list(args)
    out: list[str] = []
    for a in args:
        if a == "-i":
            out += FFMPEG_SAFE_INPUT
        out.append(a)
    return out


_HW_CACHE: dict[str, list[str]] = {}


def _ffmpeg_hwaccels() -> set[str]:
    try:
        r = subprocess.run(["ffmpeg", "-hide_banner", "-hwaccels"], capture_output=True, text=True, timeout=20)
        return {ln.strip() for ln in r.stdout.splitlines()[1:] if ln.strip()}
    except Exception:  # noqa: BLE001
        return set()


def hwaccel_args(pref: str = "auto") -> list[str]:
    """Decoder hwaccel input options for this host: videotoolbox on macOS, cuda when an NVIDIA GPU is present
    (nvidia-smi on PATH and ffmpeg built with cuda), none otherwise. "none" disables; any other name is passed on."""
    pref = (pref or "auto").strip()
    if pref not in HWACCELS:
        raise ValueError(f"hwaccel must be one of {HWACCELS}, got {pref!r}")
    if pref == "none":
        return []
    if pref != "auto":
        return ["-hwaccel", pref]
    if "auto" not in _HW_CACHE:
        avail = _ffmpeg_hwaccels(); hw: list[str] = []
        if platform.system() == "Darwin" and "videotoolbox" in avail:
            hw = ["-hwaccel", "videotoolbox"]
        elif shutil.which("nvidia-smi") and "cuda" in avail:
            hw = ["-hwaccel", "cuda"]
        _HW_CACHE["auto"] = hw
    return list(_HW_CACHE["auto"])


def _tail(text: str | bytes | None, n: int = 1500) -> str:
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    text = (text or "").strip()
    return text if len(text) <= n else "..." + text[-n:]


def run_ffmpeg(args: list[str], *, hwaccel: str = "auto", timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run ``ffmpeg <args>`` (args exclude the program name; -y/-nostdin/-loglevel are added).

    The platform hwaccel goes before the first input; if the hardware attempt fails the command is retried in
    software. FFMPEG_SAFE_INPUT is inserted before each ``-i`` when absent. stderr is captured; on failure a
    RuntimeError carries the command and the stderr tail."""
    base = ["ffmpeg", "-hide_banner", "-nostdin", "-y", "-loglevel", "error"]
    args = _guard_inputs([str(a) for a in args]); hw = hwaccel_args(hwaccel) if "-hwaccel" not in args else []
    attempts = [base + hw + args] + ([base + args] if hw else [])
    err = ""
    for i, cmd in enumerate(attempts):
        try:
            r = subprocess.run(cmd, capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            raise RuntimeError(f"ffmpeg timed out after {timeout}s: {' '.join(cmd)}\n{_tail(e.stderr)}") from None
        except FileNotFoundError:
            raise RuntimeError("ffmpeg not found on PATH (system dependency)") from None
        if r.returncode == 0:
            return r
        err = _tail(r.stderr)
        if i + 1 < len(attempts):
            print(f"[runtime] ffmpeg with {' '.join(hw)} failed (exit {r.returncode}); retrying in software. stderr:\n{err}",
                  file=sys.stderr, flush=True)
    raise RuntimeError(f"ffmpeg failed (exit {r.returncode}): {' '.join(attempts[-1])}\n{err}")


def pyav_options() -> dict[str, str]:
    """FFMPEG_SAFE_INPUT (["-opt", "value", ...]) as libavformat options for ``av.open(container_options=...)``."""
    a = FFMPEG_SAFE_INPUT
    assert len(a) % 2 == 0 and all(x.startswith("-") for x in a[::2]), a
    return {k[1:]: str(v) for k, v in zip(a[::2], a[1::2])}


def pyav_hwaccel(pref: str = "auto") -> tuple[Any, str]:
    """(PyAV HWAccel or None, label). "auto"/"none" decode in software (PyAV's frame-threaded software decoder measured
    faster than VideoToolbox: 1408^2 H.264 495 vs 278 fps, 4K HEVC 182 vs 129 fps, and it is identical on every host);
    an explicit device type is used when this PyAV build offers it (PyAV >= 14.1), with software fallback."""
    pref = (pref or "auto").strip()
    if pref not in HWACCELS:
        raise ValueError(f"hwaccel must be one of {HWACCELS}, got {pref!r}")
    if pref in ("auto", "none"):
        return None, "none"
    try:
        from av.codec.hwaccel import HWAccel, hwdevices_available
    except ImportError:
        return None, "none (PyAV without hwaccel support)"
    if pref not in hwdevices_available():
        return None, f"none ({pref} not available in this PyAV build)"
    return HWAccel(device_type=pref, allow_software_fallback=True), pref


@dataclasses.dataclass
class PyAVInput:
    """An opened user media file: ``container`` (av InputContainer), its video ``stream``, the ``hwaccel`` label.
    A context manager: closes the container."""
    container: Any
    stream: Any
    hwaccel: str

    def close(self) -> None:
        self.container.close()

    def __enter__(self) -> PyAVInput:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def pyav_video_stream(container: Any, video_index: int | None = None) -> Any:
    """The video stream at ``video_index`` (Stream.video_index from probe) when valid, else the largest video stream
    that is not an attached picture (cover art) - probe's rule. ValueError when there is none."""
    import av
    streams = list(container.streams)
    if video_index is not None and 0 <= int(video_index) < len(streams) and streams[int(video_index)].type == "video":
        return streams[int(video_index)]
    pic = getattr(getattr(av.stream, "Disposition", None), "attached_pic", None)
    cand = [st for st in streams if st.type == "video" and not (pic is not None and st.disposition & pic)]  # flag test (no int(): PyAV 14.1)
    if not cand:
        raise ValueError("no video stream")
    area = lambda st: int(getattr(st.codec_context, "width", 0) or 0) * int(getattr(st.codec_context, "height", 0) or 0)  # noqa: E731
    return max(cand, key=lambda st: (area(st), -st.index))


def pyav_open(path: str | os.PathLike, *, video_index: int | None = None, hwaccel: str = "auto") -> PyAVInput:
    """Open user media with PyAV like every ffmpeg call on user media: ``file:<abs path>`` + the FFMPEG_SAFE_INPUT
    protocol/demuxer whitelists (``pyav_options``), optional hwaccel (``pyav_hwaccel``), and pick the video stream
    (``pyav_video_stream``). Frames are NOT autorotated: apply Stream.rotation / the display matrix yourself."""
    import av
    hw, label = pyav_hwaccel(hwaccel)
    c = av.open("file:" + os.path.abspath(path), container_options=pyav_options(), **({"hwaccel": hw} if hw else {}))
    try:
        return PyAVInput(c, pyav_video_stream(c, video_index), label)
    except BaseException:
        c.close()
        raise


def ffprobe_json(path: str | os.PathLike, *args: str, timeout: float | None = 120) -> dict:
    """``ffprobe -of json <args>`` on one user media file (with FFMPEG_SAFE_INPUT). RuntimeError with stderr on failure."""
    cmd = ["ffprobe", "-v", "error", *FFMPEG_SAFE_INPUT, "-of", "json", *args, "-i", "file:" + os.path.abspath(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except FileNotFoundError:
        raise RuntimeError("ffprobe not found on PATH (system dependency)") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"ffprobe timed out after {timeout}s on {os.path.basename(path)}") from None
    if r.returncode != 0:
        raise RuntimeError(f"ffprobe failed (exit {r.returncode}) on {os.path.basename(path)}: {_tail(r.stderr, 1000)}")
    return json.loads(r.stdout or "{}")


# ----------------------------------------------------------------------------- JSON / atomic writes

def json_safe(obj: Any) -> Any:
    """Recursively convert to plain JSON types: numpy -> python, NaN/inf -> None, Path -> str, tuples/sets -> lists."""
    if obj is None or isinstance(obj, (bool, str, int)):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return json_safe(obj.item())
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, os.PathLike):
        return os.fspath(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", "replace")
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return json_safe(dataclasses.asdict(obj))
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    raise TypeError(f"not JSON-serialisable: {type(obj).__name__}")


def _fsync_dir(d: Path) -> None:
    with contextlib.suppress(OSError):
        fd = os.open(d, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic_write(path: str | os.PathLike, write: Callable[[io.BufferedWriter], None]) -> None:
    """Write via ``write(fileobj)`` into a temp file in the same directory, fsync, then os.replace onto ``path``:
    readers see the old or the new file, never a partial one; a crash leaves the old file intact."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f:
            write(f); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise
    _fsync_dir(path.parent)


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    atomic_write(path, lambda f: f.write(data))


def atomic_write_json(path: str | os.PathLike, obj: Any, indent: int | None = 1) -> None:
    """json_safe(obj) serialised with allow_nan=False (strict JSON), written atomically."""
    atomic_write_bytes(path, json.dumps(json_safe(obj), indent=indent, allow_nan=False).encode())


def atomic_savez(path: str | os.PathLike, **arrays: Any) -> None:
    """np.savez_compressed written atomically. Object arrays are refused (files must load with allow_pickle=False)."""
    arrs = {k: np.asarray(v) for k, v in arrays.items()}
    bad = [k for k, v in arrs.items() if v.dtype == object]
    if bad:
        raise TypeError(f"object arrays cannot be saved without pickle: {bad}")
    atomic_write(path, lambda f: np.savez_compressed(f, **arrs))


def atomic_write_parquet(df, path: str | os.PathLike, metadata: dict[str, Any] | None = None) -> None:
    """pandas DataFrame (index dropped) or pyarrow Table -> parquet, written atomically; ``metadata`` values (non-str
    are JSON-encoded via json_safe) are added to the parquet key-value metadata next to the existing ones."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = df if isinstance(df, pa.Table) else pa.Table.from_pandas(df, preserve_index=False)
    if metadata:
        md = dict(table.schema.metadata or {})
        for k, v in metadata.items():
            md[str(k).encode()] = (v if isinstance(v, str) else json.dumps(json_safe(v), allow_nan=False)).encode()
        table = table.replace_schema_metadata(md)
    atomic_write(path, lambda f: pq.write_table(table, f))


def load_npz(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """Eagerly load every array of an npz (allow_pickle=False) and close the file."""
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
