"""FastAPI server for the ego/exo playground: episode list, per-stream overlays, stage runner, static UI.

Access model (docs/design/egoexo_playground.md):
- Decisions are made on the ROUTE path (starlette get_route_path: the path minus root_path, exactly what the router
  matches), and everything is protected except PUBLIC_PATHS (the UI shell files and /api/session): default-deny.
- No PLAYGROUND_TOKEN: loopback only. A request from a non-loopback peer, with a non-loopback Host header (DNS
  rebinding) or carrying proxy headers (X-Forwarded-For, Forwarded, ...) gets 403, and `serve()` refuses to bind a
  non-loopback host.
- PLAYGROUND_TOKEN set (>= 16 chars after stripping whitespace): protected paths need `Authorization: Bearer <token>`,
  `X-Playground-Token: <token>`, or the HttpOnly SameSite=Strict session cookie issued by POST /api/session (what the
  UI and its <video> elements use). Failed credential checks of any kind are rate-limited per client IP (429).
- CSRF: every non-GET request needs the `X-Playground-Request` header (or a valid X-Playground-Token), and requests
  whose Sec-Fetch-Site is cross-site / same-site, or whose Origin is foreign, are refused on protected paths.
- Only the stream videos listed in episode.json are served; derived data leaves only through the JSON endpoints.
- Pipeline jobs go through run.submit / run.cancel (one queue, one subprocess per job). Uploads are streamed once
  into a staging dir inside the episodes root (cap PLAYGROUND_MAX_UPLOAD_MB, free-space floor PLAYGROUND_MIN_FREE_MB)
  and moved into place by Episode.create(mode="move").

Payload units/frames: 2D overlays in extracted-frame pixels (img_w x img_h, the frames stage output, not source
pixels); body3d in metres, camera-aligned MediaPipe world axes (x right, y down, z AWAY from the camera, hip-centred);
world3d in metres in the board frame (X right, Y down the board, Z into the board); head/camera poses are T_world_cam
(board <- camera, OpenCV camera axes). Per-frame arrays cover processed frames [start, end) of the common grid (C1).
A stage's outputs are served only while run.fresh_stages says they are current (done AND inputs unchanged).
"""
from __future__ import annotations

import errno
import gzip
import hashlib
import hmac
import ipaddress
import json
import logging
import math
import os
import re
import secrets
import shutil
import pkgutil
import importlib
import zipfile
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from email.utils import formatdate
from pathlib import Path
from typing import Annotated, Any, Callable
from urllib.parse import urlsplit

import anyio
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from python_multipart.exceptions import FormParserError
from python_multipart.multipart import MultipartParser, parse_options_header
from starlette.concurrency import run_in_threadpool
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import ClientDisconnect

from . import run as _run, runtime
from . import zed_bundle
from .episode import NAME_RE, Episode, Stream
from .runtime import json_safe

try:
    from starlette._utils import get_route_path as _route_path  # the path the router matches (root_path stripped)
except ImportError:  # pragma: no cover - same rule as starlette >= 0.33
    def _route_path(scope: dict) -> str:
        path, root = scope["path"], scope.get("root_path", "")
        if not root or not path.startswith(root):
            return path
        return "" if path == root else path[len(root):] if path[len(root)] == "/" else path

log = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")
EPISODE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")  # read side (old CLI names may contain dots)
VIDEO_TYPES = {".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".webm": "video/webm"}
SESSION_COOKIE = "pg_session"
PAYLOAD_VERSION = 3  # part of every payload ETag; bump when a payload layout changes
MAX_FILES, MAX_FIELD_BYTES = 16, 64 << 10
CSP = ("default-src 'self'; script-src 'self' https://cdnjs.cloudflare.com; style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
       "media-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
_SECURITY_HEADERS = [(b"x-content-type-options", b"nosniff"), (b"x-frame-options", b"DENY"), (b"referrer-policy", b"no-referrer"),
                     (b"cross-origin-resource-policy", b"same-origin"), (b"content-security-policy", CSP.encode())]
SHELL_PATHS = frozenset({"", "/", "/index.html", "/app.js"})  # the UI shell holds no data
PUBLIC_PATHS = SHELL_PATHS | {"/api/session"}  # everything else needs a token (when one is configured)
_MEDIA_PATH = re.compile(r"^/(api/episode/[^/]+/video/|episodes/)")
_PROXY_HEADERS = ("x-forwarded-for", "forwarded", "x-real-ip", "x-forwarded-host")
_ENV = object()


# ---------------------------------------------------------------- app / config / serve

class _FailLimiter:
    """Failed credential checks per client IP within `window_s`; an LRU of at most `max_ips` IPs (the least recently
    failing IP is evicted; the table is never cleared wholesale, so flooding it cannot wipe a guesser's count)."""

    def __init__(self, limit: int = 10, window_s: float = 60.0, max_ips: int = 4096) -> None:
        self.limit, self.window_s, self.max_ips = limit, window_s, max_ips
        self.fails: OrderedDict[str, list[float]] = OrderedDict()
        self.lock = threading.Lock()

    def _recent(self, ip: str, now: float) -> list[float]:
        ts = [t for t in self.fails.get(ip, ()) if now - t < self.window_s]
        if ts:
            self.fails[ip] = ts
        else:
            self.fails.pop(ip, None)
        return ts

    def blocked(self, ip: str) -> bool:
        with self.lock:
            return len(self._recent(ip, time.time())) >= self.limit

    def fail(self, ip: str) -> None:
        now = time.time()
        with self.lock:
            self.fails[ip] = (self._recent(ip, now) + [now])[-self.limit:]
            self.fails.move_to_end(ip)
            while len(self.fails) > self.max_ips:
                self.fails.popitem(last=False)


@dataclass
class _Config:
    root: Path
    token: str | None
    max_upload: int  # bytes per upload request
    min_free: int  # bytes that must stay free on the episodes volume
    session_ttl_s: int = 12 * 3600
    fails: _FailLimiter = field(default_factory=_FailLimiter)


def _env_mb(key: str, default_mb: int) -> int:
    v = os.environ.get(key, "").strip()
    mb = float(v) if v else float(default_mb)
    if not (math.isfinite(mb) and mb > 0):
        raise ValueError(f"{key} must be a positive number of MB, got {v!r}")
    return int(mb * (1 << 20))


def _clean_token(tok: str | None) -> str | None:
    """Strip surrounding whitespace (a trailing CR/newline from a .env file would lock the browser out: the UI trims)."""
    if not tok:
        return None
    t = tok.strip()
    if t != tok:
        log.warning("PLAYGROUND_TOKEN has leading/trailing whitespace; using the stripped value")
    return t or None


def create_app(episodes_root: Path | str | None = None, token: str | None | object = _ENV) -> FastAPI:
    """Build the playground app for one episodes root.

    episodes_root: default env PLAYGROUND_EPISODES, else <repo>/data/playground/episodes (runtime.episodes_root).
    token: default env PLAYGROUND_TOKEN; None or "" = loopback-only mode. A token must be >= 16 characters after
    stripping surrounding whitespace.
    """
    root = Path(episodes_root or runtime.episodes_root()).expanduser()
    tok = _clean_token(os.environ.get("PLAYGROUND_TOKEN") if token is _ENV else token)  # type: ignore[arg-type]
    if tok is not None and len(tok) < 16:
        raise ValueError("PLAYGROUND_TOKEN must be at least 16 characters")
    root.mkdir(parents=True, exist_ok=True)
    cfg = _Config(root.resolve(), tok, _env_mb("PLAYGROUND_MAX_UPLOAD_MB", 8192), _env_mb("PLAYGROUND_MIN_FREE_MB", 2048))
    app = FastAPI(title="Duet ego/exo playground", docs_url=None, redoc_url=None, openapi_url=None, lifespan=_lifespan)
    app.state.pg = cfg
    app.include_router(router)
    for _m in pkgutil.iter_modules([str(Path(__file__).parent)]):  # optional API modules (review, metrics, annotate ...)
        if _m.name.startswith("api_"):
            try:
                app.include_router(importlib.import_module(f"duet.playground.{_m.name}").router)
            except Exception as _e:  # noqa: BLE001
                print(f"playground: api module {_m.name} not mounted: {_e}")
    app.add_exception_handler(Exception, _internal_error)
    app.mount("/", StaticFiles(directory=str(STATIC), html=True), name="static")
    app.add_middleware(_GZipExceptMedia)
    app.add_middleware(_Guard, cfg=cfg)  # outermost: nothing runs before the access checks
    return app


def __getattr__(name: str) -> Any:
    """`from duet.playground.server import app` (uvicorn, scripts/playground.py) builds the app from env on first use."""
    if name == "app":
        app = globals()["app"] = create_app()
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def is_loopback_host(host: str) -> bool:
    """True for "localhost" and loopback IPs (127.0.0.0/8, ::1). 0.0.0.0, :: and other hostnames are NOT loopback."""
    h = (host or "").strip().strip("[]").lower()
    if h == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(h.split("%")[0])
    except ValueError:
        return False
    return (getattr(ip, "ipv4_mapped", None) or ip).is_loopback  # ::ffff:127.0.0.1 from a dual-stack socket


def serve(host: str = "127.0.0.1", port: int = 8765, episodes_root: Path | str | None = None, log_level: str = "info") -> None:
    """Run uvicorn. Refuses a non-loopback bind unless PLAYGROUND_TOKEN is set (defence in depth with the CLI check)."""
    token = _clean_token(os.environ.get("PLAYGROUND_TOKEN"))
    if token is None and not is_loopback_host(host):
        raise SystemExit(f"refusing to serve on {host!r} without PLAYGROUND_TOKEN (>= 16 chars): bind 127.0.0.1 or set a token")
    import uvicorn
    uvicorn.run(create_app(episodes_root, token), host=host, port=port, log_level=log_level)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    cfg: _Config = app.state.pg
    try:
        _run.recover_stale(cfg.root)  # stages left "running" by a dead worker -> "interrupted"
    except Exception:  # noqa: BLE001 - never block startup on bookkeeping
        log.exception("recover_stale failed")
    _clean_staging(cfg.root)
    yield


def _clean_staging(root: Path, max_age_s: float = 6 * 3600) -> None:
    """Remove upload staging dirs left by a crashed server (only ours: dot-prefixed, older than max_age_s)."""
    for d in root.glob(".upload-*"):
        try:
            if d.is_dir() and time.time() - d.stat().st_mtime > max_age_s:
                shutil.rmtree(d, ignore_errors=True)
        except OSError:
            pass


async def _internal_error(request: Request, exc: Exception) -> JSONResponse:
    log.error("unhandled error on %s %s", request.method, request.url.path, exc_info=exc)
    return JSONResponse({"detail": "internal server error (see the server log)"}, 500)


class _GZipExceptMedia:
    """GZip for JSON/HTML only. Requests with a Range header and media routes bypass it whatever the starlette version
    (starlette < 1.5 also compressed video/* and 206 responses, which breaks <video> seeking)."""

    def __init__(self, app: Any) -> None:
        self.app, self.gzip = app, GZipMiddleware(app, minimum_size=1024, compresslevel=6)

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] == "http" and not _MEDIA_PATH.match(_route_path(scope)) and not any(k == b"range" for k, _ in scope.get("headers", ())):
            await self.gzip(scope, receive, send)
        else:
            await self.app(scope, receive, send)


# ---------------------------------------------------------------- access control (pure ASGI, runs first)

class _Guard:
    """Loopback policy, authentication (default-deny) and CSRF checks in front of every route; adds security headers."""

    def __init__(self, app: Any, cfg: _Config) -> None:
        self.app, self.cfg = app, cfg

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        path = _route_path(scope)

        async def send_secured(msg: dict) -> None:
            if msg["type"] == "http.response.start":
                hs = list(msg.get("headers", []))
                have = {k.lower() for k, _ in hs}
                hs += [(k, v) for k, v in _SECURITY_HEADERS if k not in have]
                if path in SHELL_PATHS and b"cache-control" not in have:
                    hs.append((b"cache-control", b"no-cache"))  # an old cached app.js must not meet a newer payload layout
                msg["headers"] = hs
            await send(msg)

        denied = _deny_reason(self.cfg, Request(scope), path)
        if denied:
            status, detail = denied
            resp = JSONResponse({"detail": detail}, status, headers={"WWW-Authenticate": "Bearer"} if status == 401 else None)
            await resp(scope, receive, send_secured)
            return
        await self.app(scope, receive, send_secured)


def _client_ip(req: Request) -> str:
    return str(req.client.host) if req.client else "?"


def _deny_reason(cfg: _Config, req: Request, path: str) -> tuple[int, str] | None:
    unsafe, protected = req.method not in ("GET", "HEAD", "OPTIONS"), path not in PUBLIC_PATHS
    if cfg.token is None and not _local_request(req):
        return 403, "no PLAYGROUND_TOKEN is configured, so this server only answers loopback requests (localhost Host, no proxy headers)"
    site = req.headers.get("sec-fetch-site")
    if (protected or unsafe) and site in ("cross-site", "same-site"):
        return 403, "cross-site request refused"
    if cfg.token is not None and protected:
        verdict = _auth_verdict(cfg, req)
        if verdict != "ok":
            ip = _client_ip(req)
            if verdict == "bad":  # wrong token / forged cookie: a guess (an expired session or no credentials is not)
                if cfg.fails.blocked(ip):
                    return 429, "too many failed attempts; wait a minute"
                cfg.fails.fail(ip)
            return 401, "authentication required: Authorization: Bearer <token>, X-Playground-Token, or POST /api/session"
        if cfg.fails.blocked(_client_ip(req)):  # a correct guess inside a lockout does not get through either
            return 429, "too many failed attempts; wait a minute"
    if unsafe:
        if site not in (None, "same-origin"):
            return 403, f"request refused (Sec-Fetch-Site: {site})"
        origin = req.headers.get("origin")
        if origin is not None and not _same_origin(origin, req):
            return 403, "cross-origin request refused (Origin does not match Host)"
        if not req.headers.get("x-playground-request") and not _token_match(cfg, req.headers.get("x-playground-token")):
            return 403, "state-changing requests need the X-Playground-Request header"
    return None


def _host_of(req: Request) -> tuple[str | None, int | None]:
    try:
        u = urlsplit("//" + req.headers.get("host", ""))
        return u.hostname, u.port
    except ValueError:
        return None, None


def _local_request(req: Request) -> bool:
    client = req.scope.get("client")
    if not client or not is_loopback_host(str(client[0])) or any(h in req.headers for h in _PROXY_HEADERS):
        return False
    host, _ = _host_of(req)
    return bool(host) and is_loopback_host(host)


def _same_origin(origin: str, req: Request) -> bool:
    try:
        o = urlsplit(origin)
        if o.scheme not in ("http", "https") or not o.hostname:
            return False
        host, port = _host_of(req)
        dflt = 443 if o.scheme == "https" else 80  # a same-origin browser request uses the Origin's scheme
        return o.hostname == host and (o.port or dflt) == (port or dflt)
    except ValueError:
        return False


def _token_match(cfg: _Config, value: str | None) -> bool:
    return bool(cfg.token and value) and hmac.compare_digest(value.encode("utf-8", "replace"), cfg.token.encode())


def _bearer(req: Request) -> str | None:
    auth = req.headers.get("authorization", "")
    return auth[7:].strip() if auth[:7].lower() == "bearer " else None


def _session_sign(cfg: _Config, msg: str) -> str:
    key = hashlib.sha256(b"duet-playground-session\0" + cfg.token.encode()).digest()
    return hmac.new(key, msg.encode(), hashlib.sha256).hexdigest()


def _new_session(cfg: _Config) -> str:
    msg = f"{int(time.time()) + cfg.session_ttl_s}.{secrets.token_hex(12)}"
    return f"{msg}.{_session_sign(cfg, msg)}"


def _session_state(cfg: _Config, value: str | None) -> str | None:
    """Stateless session "<expiry>.<nonce>.<hmac>" keyed by the token (rotating the token revokes every session).
    None = no cookie, "ok", "expired" (genuine but old), "bad" (forged, malformed, non-ASCII)."""
    if not value:
        return None
    if cfg.token is None or not value.isascii():
        return "bad"
    msg, _, sig = value.rpartition(".")
    try:
        expires = int(msg.split(".", 1)[0])
    except ValueError:
        return "bad"
    if not hmac.compare_digest(sig.encode(), _session_sign(cfg, msg).encode()):
        return "bad"
    return "ok" if expires > time.time() else "expired"


def _auth_verdict(cfg: _Config, req: Request) -> str:
    """"ok", "none" (no credentials), "expired" (old session cookie) or "bad" (wrong token / forged cookie)."""
    bearer, header = _bearer(req), req.headers.get("x-playground-token")
    if _token_match(cfg, bearer) or _token_match(cfg, header):
        return "ok"
    sess = _session_state(cfg, req.cookies.get(SESSION_COOKIE))
    if sess == "ok":
        return "ok"
    if bearer or header or sess == "bad":
        return "bad"
    return sess or "none"


# ---------------------------------------------------------------- helpers

router = APIRouter()


def _cfg(request: Request) -> _Config:
    return request.app.state.pg


def _json(obj: Any, status: int = 200) -> JSONResponse:
    return JSONResponse(json_safe(obj), status)


def _valid_episode_name(name: str) -> bool:
    return bool(EPISODE_NAME.match(name)) and ".." not in name


def _load_episode(cfg: _Config, name: str) -> Episode:
    d = cfg.root / name
    if not _valid_episode_name(name) or not (d / "episode.json").is_file():  # dirs without episode.json = absent
        raise HTTPException(404, "no such episode")
    try:
        return Episode.load(d)
    except Exception as e:  # noqa: BLE001
        log.warning("episode %s unreadable: %r", name, e)
        raise HTTPException(500, f"episode {name}: unreadable episode.json ({type(e).__name__})") from None


def _stream(ep: Episode, name: str) -> Stream:
    for s in ep.streams:
        if s.name == name:
            return s
    raise HTTPException(404, "no such stream")


def _read_json(p: Path) -> Any:
    with open(p) as f:
        return json.load(f)  # NaN tokens parse to float nan; json_safe turns them into null


def fresh_set(ep: Episode) -> set[str]:
    """Stages whose outputs may be served/published: run.fresh_stages (done AND inputs/config/code still match,
    cascaded; legacy records without a fingerprint are trusted until an upstream changes). ~0.5 ms: computed once per
    request, reads are gated on membership."""
    return set(_run.fresh_stages(ep))


def _fresh(ep: Episode, stage: str) -> bool:
    return stage in fresh_set(ep)


def stale_reasons(ep: Episode) -> dict[str, str]:
    """{stage: why its "done" result is out of date} (run.stale_stages; nothing written). Those results are not served."""
    try:
        return {str(k): str(v)[:300] for k, v in _run.stale_stages(ep).items()}
    except Exception:  # noqa: BLE001 - informational only
        log.exception("run.stale_stages failed")
        return {}


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _problem(ep: Episode) -> str | None:
    """What makes an episode unusable in the viewer (it is still listed and served, with this message)."""
    if not (_num(ep.proc_fps) and ep.proc_fps > 0):
        return f"invalid proc_fps {ep.proc_fps!r} in episode.json"
    if not (_num(ep.common_start_s) and _num(ep.common_end_s)):
        return "invalid common window in episode.json (run align)"
    return None


def _busy(ep_dir: Path) -> bool:
    try:
        return bool(_run.is_running(ep_dir))  # queued or running
    except Exception:  # noqa: BLE001
        log.exception("run.is_running failed")
        return False


def stage_list(ep: Episode) -> list[str]:
    """Default stage order, then any other stage present in the status (opt-in stages that ran, legacy names);
    underscore keys (e.g. fx-core's "_job" record) are not stages."""
    return list(_run.DEFAULT_ORDER) + [k for k in (ep.status or {}) if k not in _run.DEFAULT_ORDER and not k.startswith("_")]


def _queue() -> dict | None:
    """run.queue_state() for the browser: episode names, stages and times only (no pids, log paths)."""
    try:
        q = _run.queue_state()
    except Exception:  # noqa: BLE001
        log.exception("run.queue_state failed")
        return None
    job = lambda j: {k: j.get(k) for k in ("episode", "stages", "force", "queued", "started")}  # noqa: E731
    return {"max_jobs": q.get("max_jobs"), "running": [job(j) for j in q.get("running", [])], "queued": [job(j) for j in q.get("queued", [])]}


def _n_frames(ep: Episode) -> int | None:
    try:
        return ep.n_frames()
    except (TypeError, ValueError):  # empty/invalid window (not aligned yet), null/non-numeric proc_fps
        return None


_STATUS_FIELDS = ("state", "detail", "started", "finished")


# ---------------------------------------------------------------- UI shell

_SHELL_CACHE: dict[tuple, str] = {}


def ui_index_html() -> str:
    """index.html with the script URL versioned by app.js's content hash (app.js?v=<sha256[:12]>)."""
    idx, js = STATIC / "index.html", STATIC / "app.js"
    key = (idx.stat().st_mtime_ns, js.stat().st_mtime_ns)
    if key not in _SHELL_CACHE:
        html, tag = idx.read_text(), '<script src="app.js"></script>'
        if html.count(tag) != 1:
            raise RuntimeError("index.html must contain exactly one <script src=\"app.js\"></script>")
        _SHELL_CACHE.clear()
        _SHELL_CACHE[key] = html.replace(tag, f'<script src="app.js?v={hashlib.sha256(js.read_bytes()).hexdigest()[:12]}"></script>')
    return _SHELL_CACHE[key]


@router.get("/", include_in_schema=False)
@router.get("/index.html", include_in_schema=False)
def ui_index():
    return HTMLResponse(ui_index_html(), headers={"Cache-Control": "no-cache"})


# ---------------------------------------------------------------- session / episodes / run

@router.get("/api/session")
def session_state(request: Request):
    cfg = _cfg(request)
    return {"auth_required": cfg.token is not None, "authenticated": cfg.token is None or _auth_verdict(cfg, request) == "ok"}


@router.post("/api/session")
async def session_login(request: Request):
    """Exchange the token (X-Playground-Token, Bearer, or JSON {"token": ...}) for an HttpOnly SameSite=Strict cookie."""
    cfg = _cfg(request)
    if cfg.token is None:
        return {"auth_required": False, "authenticated": True}
    ip = _client_ip(request)
    if cfg.fails.blocked(ip):
        raise HTTPException(429, "too many failed attempts; wait a minute")
    token = request.headers.get("x-playground-token") or _bearer(request) or (await _small_json(request)).get("token")
    if not isinstance(token, str) or not _token_match(cfg, token.strip()):
        cfg.fails.fail(ip)
        raise HTTPException(401, "invalid token")
    resp = JSONResponse({"auth_required": True, "authenticated": True})
    resp.set_cookie(SESSION_COOKIE, _new_session(cfg), max_age=cfg.session_ttl_s, path="/", httponly=True, samesite="strict",
                    secure=request.url.scheme == "https")
    return resp


@router.delete("/api/session")
def session_logout():
    resp = JSONResponse({"authenticated": False})
    resp.delete_cookie(SESSION_COOKIE, path="/", httponly=True, samesite="strict")
    return resp


async def _small_json(request: Request, limit: int = 4096) -> dict:
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > limit:
            raise HTTPException(413, "request body too large")
    if not body.strip():
        return {}
    try:
        d = json.loads(body)
    except ValueError:
        raise HTTPException(422, "expected a JSON object") from None
    return d if isinstance(d, dict) else {}


@router.get("/api/episodes")
def list_episodes(request: Request):
    """All episodes. An unreadable episode.json is listed with "error" (not selectable); an unusable config (e.g. a
    null proc_fps) with "problem"; neither fails the list."""
    cfg, out = _cfg(request), []
    for d in sorted(cfg.root.iterdir()):
        if not (_valid_episode_name(d.name) and (d / "episode.json").is_file()):
            continue
        try:
            ep = Episode.load(d)
            problem = _problem(ep)
            status = ep.status or {}
            out.append({"name": d.name, "streams": [{"name": s.name, "role": s.role} for s in ep.streams],
                        "duration_s": None if problem else ep.common_end_s - ep.common_start_s, "running": _busy(ep.dir),
                        "status": {k: (v or {}).get("state") for k, v in status.items() if not k.startswith("_")},
                        "job": (status.get("_job") or {}).get("state") if isinstance(status.get("_job"), dict) else None,
                        **({"problem": problem} if problem else {})})
        except Exception as e:  # noqa: BLE001 - one corrupt episode must not hide the others
            log.warning("episode %s unreadable: %r", d.name, e)
            out.append({"name": d.name, "error": f"unreadable episode.json ({type(e).__name__})", "streams": [], "duration_s": None, "running": False})
    return _json(out)


@router.get("/api/episode/{name}")
def episode_detail(request: Request, name: str):
    """episode.json (normalised by Episode.load: legacy files get the new fields, unknown keys are kept) plus
    running/queue flags, stage order, frame count, the last job record, a config problem if any, and the QC report."""
    ep = _load_episode(_cfg(request), name)
    d = ep.to_dict()
    status = d.get("status") or {}
    job = status.get("_job")
    d["status"] = {k: {f: v[f] for f in _STATUS_FIELDS if f in v} for k, v in status.items()
                   if isinstance(v, dict) and not k.startswith("_")}  # no pid/host/fingerprint/inputs/log internals
    d.update(name=name, running=_busy(ep.dir), stages=stage_list(ep), n_frames=_n_frames(ep), qc=None, problem=_problem(ep),
             job={f: job[f] for f in _STATUS_FIELDS if f in job} if isinstance(job, dict) else None, stale=stale_reasons(ep))
    by_name = {s.name: s for s in ep.streams}
    for sd in d.get("streams", []):
        s = by_name.get(sd.get("name"))
        sd["usable"] = ep.usable(s) if s is not None else False
    qc = ep.dir / "derived" / "qc" / "report.json"
    if _fresh(ep, "qc") and qc.is_file():
        try:
            d["qc"] = _read_json(qc)
        except (OSError, ValueError):
            d["qc_error"] = "unreadable qc/report.json"
    return _json(d)


@router.post("/api/episode/{name}/run")
def run_pipeline(request: Request, name: str, stages: str = "", force: bool = False):
    """Queue a pipeline run (202). 409 when this episode is already queued or running; 422 for unknown stages."""
    ep = _load_episode(_cfg(request), name)
    wanted = [s.strip() for s in stages.split(",") if s.strip()] or None
    unknown = [s for s in wanted or [] if s not in _run.ALL_STAGES]
    if unknown:
        raise HTTPException(422, f"unknown stages {unknown}; known: {', '.join(_run.ALL_STAGES)}")
    if not _run.submit(ep.dir, wanted, force):
        raise HTTPException(409, f"episode {name} is already queued or running")
    return _json({"queued": True, "queue": _queue()}, 202)


@router.post("/api/episode/{name}/cancel")
def cancel_pipeline(request: Request, name: str):
    """Cancel this episode's job (run.cancel): a queued job is dropped, a running one gets SIGTERM on its process group
    and its running stages end "interrupted". 202 when something was cancelled; 409 when this server has nothing
    queued or running for the episode (a CLI run holding the lock is not ours to cancel)."""
    ep = _load_episode(_cfg(request), name)
    if not _run.cancel(ep.dir):
        raise HTTPException(409, f"nothing to cancel: no job for episode {name} is queued or running in this server")
    return _json({"cancelled": True, "queue": _queue()}, 202)


@router.get("/api/queue")
def queue_state():
    return _json(_queue())


# ---------------------------------------------------------------- upload (streamed once, validated, moved into place)

class _UploadError(Exception):
    def __init__(self, status: int, detail: str) -> None:
        super().__init__(detail)
        self.status, self.detail = status, detail


def _label(filename: str) -> str:
    return re.sub(r"[^\w .-]", "_", filename)[:80]


def _looks_like_video(head: bytes) -> bool:
    """ISO-BMFF (mp4/mov/m4v: a box type at bytes 4..8) or EBML (mkv/webm)."""
    return head[:4] == b"\x1a\x45\xdf\xa3" or head[4:8] in (b"ftyp", b"moov", b"mdat", b"free", b"skip", b"wide", b"pnot", b"uuid")


def _clean_episode_name(raw: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", raw.strip()).strip("_-")[:64]
    if not NAME_RE.match(name):  # C1 stream-name rule, also used for uploaded episode names
        raise _UploadError(422, "episode name must contain letters or digits")
    return name


def _clean_person(raw: str) -> str | None:
    return re.sub(r"[^\w .-]+", "", raw).strip()[:32] or None


def _stream_names(filenames: list[str]) -> list[str]:
    """Sanitised filename stems, de-duplicated case-insensitively (GX010001, GX010001_2, ...)."""
    out, seen = [], set()
    for fn in filenames:
        base = fn.replace("\\", "/").rsplit("/", 1)[-1]
        stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(base).stem).strip("_-")[:32] or "stream"
        name, i = stem, 2
        while name.lower() in seen:
            name, i = f"{stem[:28]}_{i}", i + 1
        seen.add(name.lower())
        out.append(name)
    return out


class _MultipartSink:
    """python-multipart callbacks: text fields kept in memory (bounded), file parts written ONCE into `staging`.

    feed()/finish() do blocking file IO and run in a worker thread, never on the event loop."""

    def __init__(self, boundary: bytes, staging: Path, cfg: _Config) -> None:
        self.staging, self.cfg = staging, cfg
        self.fields: dict[str, str] = {}
        self.files: list[dict[str, Any]] = []
        self.ended = False
        self._headers: dict[bytes, bytes] = {}
        self._hname, self._hval, self._buf = bytearray(), bytearray(), bytearray()
        self._field: str | None = None
        self._fh: Any = None
        self._written, self._next_disk_check = 0, 64 << 20
        self._parser = MultipartParser(boundary, {
            "on_part_begin": self._part_begin, "on_header_field": self._header_field, "on_header_value": self._header_value,
            "on_header_end": self._header_end, "on_headers_finished": self._headers_finished, "on_part_data": self._part_data,
            "on_part_end": self._part_end, "on_end": self._end})

    def feed(self, data: bytes) -> None:
        self._parser.write(data)

    def finish(self) -> None:
        self._parser.finalize()
        if not self.ended:
            raise _UploadError(400, "truncated multipart body")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def _part_begin(self) -> None:
        self._headers, self._field = {}, None
        self._hname.clear(); self._hval.clear()

    def _header_field(self, data: bytes, start: int, end: int) -> None:
        self._hname += data[start:end]
        if len(self._hname) > 256:
            raise _UploadError(400, "multipart header too long")

    def _header_value(self, data: bytes, start: int, end: int) -> None:
        self._hval += data[start:end]
        if len(self._hval) > 4096:
            raise _UploadError(400, "multipart header too long")

    def _header_end(self) -> None:
        self._headers[bytes(self._hname).lower()] = bytes(self._hval)
        self._hname.clear(); self._hval.clear()

    def _headers_finished(self) -> None:
        _, opts = parse_options_header(self._headers.get(b"content-disposition", b""))
        name = opts.get(b"name", b"").decode("utf-8", "replace")
        if b"filename" not in opts:
            if name not in ("name", "roles", "persons", "reference", "fps"):
                raise _UploadError(422, f"unexpected form field {_label(name)!r}")
            if name in self.fields:
                raise _UploadError(422, f"duplicate form field {name!r}")
            self._field = name
            self._buf.clear()
            return
        if name != "files":
            raise _UploadError(422, f"unexpected file field {_label(name)!r}; send the videos as 'files'")
        if len(self.files) >= MAX_FILES:
            raise _UploadError(422, f"at most {MAX_FILES} files per upload")
        filename = opts[b"filename"].decode("utf-8", "replace")
        ext = Path(filename.replace("\\", "/")).suffix.lower()
        if ext not in VIDEO_TYPES and ext != ".zip":
            raise _UploadError(422, f"{_label(filename)}: unsupported file type; allowed: {' '.join(sorted(VIDEO_TYPES))} .zip (ZED bundle from zed_kit.py)")
        if ext == ".zip":  # ZED bundle: the stream is named after the zip minus a trailing _zed
            stem = Path(filename.replace("\\", "/")).stem
            filename = (stem[:-4] if stem.endswith("_zed") else stem) + ".zip"
        path = self.staging / f"{len(self.files):02d}{ext}"
        self._fh = open(path, "xb")
        self.files.append({"filename": filename, "path": path, "size": 0, "head": b""})

    def _part_data(self, data: bytes, start: int, end: int) -> None:
        chunk = data[start:end]
        if self._fh is not None:
            f = self.files[-1]
            if len(f["head"]) < 16:
                f["head"] += chunk[:16 - len(f["head"])]
            self._fh.write(chunk)
            f["size"] += len(chunk)
            self._written += len(chunk)
            if self._written >= self._next_disk_check:
                self._next_disk_check += 64 << 20
                if shutil.disk_usage(self.staging).free < self.cfg.min_free:
                    raise _UploadError(507, "the episodes volume is nearly full; upload aborted")
        elif self._field is not None:
            self._buf += chunk
            if len(self._buf) > MAX_FIELD_BYTES:
                raise _UploadError(413, f"form field {self._field!r} too large")

    def _part_end(self) -> None:
        if self._fh is not None:
            self.close()
            f = self.files[-1]
            if f["size"] == 0:
                raise _UploadError(422, f"{_label(f['filename'])} is empty")
            if f["path"].suffix == ".zip":
                if not f["head"].startswith(b"PK\x03\x04") or not zed_bundle.is_bundle(f["path"]):
                    raise _UploadError(422, f"{_label(f['filename'])} is not a ZED bundle (zip with left.mp4 + calibration.json from zed_kit.py)")
            elif not _looks_like_video(f["head"]):
                raise _UploadError(422, f"{_label(f['filename'])} is not an MP4/MOV/MKV/WebM video")
        elif self._field is not None:
            try:
                self.fields[self._field] = self._buf.decode("utf-8")
            except UnicodeDecodeError:
                raise _UploadError(422, f"form field {self._field!r} is not UTF-8") from None
            if self._field == "name" and (self.cfg.root / _clean_episode_name(self.fields["name"]) / "episode.json").exists():
                raise _UploadError(409, "an episode with this name exists")  # before the (large) files arrive
            self._field = None

    def _end(self) -> None:
        self.ended = True


def _upload_spec(sink: _MultipartSink) -> dict[str, Any]:
    """Validate the parsed form. Raises _UploadError(422) with a message the UI shows as is."""
    f, files = sink.fields, sink.files
    if not files:
        raise _UploadError(422, "no video files (form field 'files')")
    name = _clean_episode_name(f.get("name", ""))
    roles = [r.strip().lower() for r in f.get("roles", "").split(",")]
    if len(roles) != len(files):
        raise _UploadError(422, f"roles: need exactly one role per file ({len(files)} files, {len(roles)} roles)")
    if set(roles) - {"ego", "exo"}:
        raise _UploadError(422, "roles must be 'ego' or 'exo'")
    persons_raw = f.get("persons", "")
    persons = [None] * len(files) if not persons_raw.strip() else [_clean_person(p) for p in persons_raw.split(",")]
    if len(persons) != len(files):
        raise _UploadError(422, f"persons: need one entry per file ({len(files)} files, {len(persons)} entries; leave entries empty for none)")
    try:
        fps = float(f.get("fps", "10") or "10")
    except ValueError:
        raise _UploadError(422, "fps must be a number") from None
    if not (math.isfinite(fps) and 0 < fps <= 60):
        raise _UploadError(422, "fps must be finite and in (0, 60]")
    streams = _stream_names([x["filename"] for x in files])
    ref = f.get("reference", "").strip() or None
    if ref is not None and ref not in streams:
        ref = {x["filename"]: s for x, s in zip(files, streams)}.get(ref)
        if ref is None:
            raise _UploadError(422, f"reference must be one of the streams {streams} (or their original filenames)")
    return {"name": name, "fps": fps, "reference": ref,
            "videos": [(s, role, x["path"], p) for s, role, x, p in zip(streams, roles, files, persons)]}


_CREATE_LOCK = threading.Lock()


def _create_uploaded(cfg: _Config, spec: dict[str, Any]) -> Episode:
    """Episode.create(mode="move") from the staged files. A directory without episode.json (debris of the old,
    non-atomic upload code) counts as absent and is replaced (rig.json / zed/ are kept by create) unless it holds real
    stream/IMU files, which may be the only copy: create refuses that and the 409 says which files."""
    final = cfg.root / spec["name"]
    with _CREATE_LOCK:  # the debris check and the overwrite must not interleave with another upload of this name
        debris = final.is_dir() and not final.is_symlink() and not (final / "episode.json").exists()
        if (final / "episode.json").exists():
            raise _UploadError(409, f"episode {spec['name']} exists")
        videos = []
        for sname, role, path, person in spec["videos"]:
            if Path(path).suffix == ".zip":  # ZED bundle: unpack next to the zip, keep the whole export under zed/<stream>/
                bdir = Path(path).with_suffix("")
                with zipfile.ZipFile(path) as zf:
                    zf.extractall(bdir)
                if not zed_bundle.is_bundle(bdir):
                    raise _UploadError(422, f"{sname}: zip is not a ZED bundle")
                zed_bundle.ingest(bdir, final, sname); path = bdir / "left.mp4"; debris = True
            videos.append((sname, role, path, person))
        spec = {**spec, "videos": videos}
        try:
            return Episode.create(cfg.root, spec["name"], spec["videos"], None, spec["reference"], mode="move",
                                  proc_fps=spec["fps"], overwrite=debris)
        except FileExistsError as e:
            raise _UploadError(409, str(e)[:500] if debris else f"episode {spec['name']} exists") from None
        except RuntimeError:  # overwrite refused while a pipeline runs
            raise _UploadError(409, f"episode {spec['name']} exists") from None
        except ValueError as e:
            raise _UploadError(422, str(e)[:500]) from None


@router.post("/api/upload", status_code=201)
async def upload(request: Request):
    """Create an episode from uploaded videos and queue the pipeline (201).

    multipart/form-data: name; roles (comma list, one per file: ego|exo); persons (optional comma list, one per file);
    reference (optional: a stream name or original filename); fps (processing rate, 0 < fps <= 60, default 10);
    files (videos .mp4 .mov .m4v .mkv .webm). Stream names are the sanitised filename stems, de-duplicated with _2,
    _3, ... The body is streamed once into <root>/.upload-<random>/ (never the system temp dir) and moved into the
    episode by Episode.create(mode="move"); the staging dir is removed whatever happens.
    """
    cfg = _cfg(request)
    ctype, opts = parse_options_header(request.headers.get("content-type", ""))
    if ctype.lower() != b"multipart/form-data" or not opts.get(b"boundary"):
        raise HTTPException(415, "expected multipart/form-data")
    try:
        declared = int(request.headers.get("content-length", "0"))
    except ValueError:
        raise HTTPException(400, "bad Content-Length") from None
    if declared > cfg.max_upload:
        raise HTTPException(413, f"upload exceeds the {cfg.max_upload >> 20} MB limit (PLAYGROUND_MAX_UPLOAD_MB)")
    if shutil.disk_usage(cfg.root).free < declared + cfg.min_free:
        raise HTTPException(507, "not enough free disk space for this upload")
    staging: Path | None = None
    sink: _MultipartSink | None = None
    try:
        staging = cfg.root / f".upload-{secrets.token_hex(8)}"
        staging.mkdir(mode=0o700)
        sink = _MultipartSink(opts[b"boundary"], staging, cfg)  # raises FormParserError for an invalid boundary
        got, buf = 0, bytearray()
        async for chunk in request.stream():
            got += len(chunk)
            if got > cfg.max_upload:
                raise _UploadError(413, f"upload exceeds the {cfg.max_upload >> 20} MB limit (PLAYGROUND_MAX_UPLOAD_MB)")
            buf += chunk
            if len(buf) >= 1 << 20:
                data = bytes(buf); buf.clear()
                await anyio.to_thread.run_sync(sink.feed, data)
        await anyio.to_thread.run_sync(sink.feed, bytes(buf))
        await anyio.to_thread.run_sync(sink.finish)
        spec = _upload_spec(sink)
        ep = await run_in_threadpool(_create_uploaded, cfg, spec)
    except _UploadError as e:
        raise HTTPException(e.status, e.detail) from None
    except FormParserError:
        raise HTTPException(400, "malformed multipart body") from None
    except ClientDisconnect:
        return Response(status_code=400)
    except OSError as e:
        if e.errno == errno.ENOSPC:
            raise HTTPException(507, "the episodes volume is full; upload aborted") from None
        raise
    finally:
        if sink is not None:
            sink.close()
        if staging is not None:
            await anyio.to_thread.run_sync(lambda: shutil.rmtree(staging, ignore_errors=True))
    try:
        queued = _run.submit(ep.dir, None, False)
    except Exception:  # noqa: BLE001 - the episode exists; the user can press Run later
        log.exception("run.submit failed for %s", ep.dir.name)
        queued = False
    return _json({"name": ep.dir.name, "streams": [s.name for s in ep.streams], "queued": bool(queued)}, 201)


# ---------------------------------------------------------------- media

def _video_response(ep: Episode, s: Stream) -> FileResponse:
    rel = Path(s.path)
    if rel.is_absolute() or ".." in rel.parts or rel.parts[:1] != ("streams",):
        raise HTTPException(404, "no such stream file")
    p = ep.dir / rel
    if not p.is_file():  # includes dangling symlinks
        raise HTTPException(404, f"stream file for {s.name} is missing")
    return FileResponse(p, media_type=VIDEO_TYPES.get(p.suffix.lower(), "application/octet-stream"), headers={"Cache-Control": "private, no-cache"})


@router.get("/api/episode/{name}/video/{stream}")
def stream_video(request: Request, name: str, stream: str):
    """The stream's video as stored (HTTP Range -> 206, which <video> seeking needs)."""
    ep = _load_episode(_cfg(request), name)
    return _video_response(ep, _stream(ep, stream))


@router.get("/episodes/{name}/{rest:path}")
def episode_media(request: Request, name: str, rest: str):
    """Old URL layout /episodes/<name>/<stream path>: serves ONLY stream videos listed in episode.json."""
    ep = _load_episode(_cfg(request), name)
    s = next((s for s in ep.streams if s.path == rest), None)
    if s is None:
        raise HTTPException(404, "not found")
    return _video_response(ep, s)


# ---------------------------------------------------------------- JSON payloads (shared with scripts/playground_export_static.py)

def dumps(obj: Any) -> str:
    """Compact JSON. Float ndarrays go through tolist() (C speed, shortest repr of the already ROUNDED float64 values)
    with NaN/inf -> null; everything else through runtime.json_safe. Never emits NaN."""
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind == "f":
            s = json.dumps(obj.astype(np.float64).tolist(), separators=(",", ":"))
            return s.replace("-Infinity", "null").replace("Infinity", "null").replace("NaN", "null")
        return json.dumps(json_safe(obj.tolist()), separators=(",", ":"), allow_nan=False)
    if isinstance(obj, dict):
        return "{" + ",".join(json.dumps(str(k)) + ":" + dumps(v) for k, v in obj.items()) + "}"
    if isinstance(obj, (list, tuple)):
        return "[" + ",".join(dumps(v) for v in obj) + "]"
    return json.dumps(json_safe(obj), separators=(",", ":"), allow_nan=False)


def _r(a: np.ndarray, decimals: int) -> np.ndarray:
    """Round in float64 (a new array): rounding float32 then tolist() yields 37.400001525878906."""
    return np.round(np.asarray(a, dtype=np.float64), decimals)


class _LRU:
    """Thread-safe LRU bounded by total bytes (`weigh(value)`); values larger than a quarter of the budget are not kept."""

    def __init__(self, max_bytes: int, weigh: Callable[[Any], int]) -> None:
        self.max, self.weigh, self.size, self.items, self.lock = max_bytes, weigh, 0, OrderedDict(), threading.Lock()

    def get(self, key: tuple) -> Any:
        with self.lock:
            v = self.items.get(key)
            if v is not None:
                self.items.move_to_end(key)
            return v

    def put(self, key: tuple, value: Any) -> None:
        w = self.weigh(value)
        if w > self.max // 4:
            return
        with self.lock:
            old = self.items.pop(key, None)
            self.size += w - (self.weigh(old) if old is not None else 0)
            self.items[key] = value
            while self.size > self.max:
                self.size -= self.weigh(self.items.popitem(last=False)[1])


_BODIES = _LRU(256 << 20, len)  # encoded payloads, keyed by (ETag, gzip)
_ARRAYS = _LRU(512 << 20, lambda d: sum(a.nbytes for a in d.values() if isinstance(a, np.ndarray)))  # decoded npz members


def _load_arrays(path: Path, members: tuple[str, ...] | None = None, prep: Callable[[dict], dict] | None = None) -> dict[str, np.ndarray]:
    """Decoded npz members (only those asked for: each npz member is decompressed on access), cached by file identity.
    The arrays are read-only and shared between requests; callers slice them, then derive new arrays."""
    st = path.stat()
    key = (str(path), st.st_mtime_ns, st.st_size, members, getattr(prep, "__name__", None))
    d = _ARRAYS.get(key)
    if d is None:
        with np.load(path, allow_pickle=False) as z:  # closed right away (never keep a lazy NpzFile)
            d = {k: z[k] for k in (z.files if members is None else [m for m in members if m in z.files])}
        d = prep(d) if prep else d
        for a in d.values():
            if isinstance(a, np.ndarray):
                a.flags.writeable = False
        _ARRAYS.put(key, d)
    return d


def _npz(ep: Episode, stage: str, filename: str, members: tuple[str, ...] | None = None, prep: Callable[[dict], dict] | None = None,
         fresh: dict[str, bool] | None = None) -> dict[str, np.ndarray] | None:
    """A stage output's arrays, or None unless that stage is current (fresh; C1: stale files of older runs, or of a
    stage skipped for missing deps, are never read)."""
    ok = fresh[stage] if fresh is not None and stage in fresh else _fresh(ep, stage)
    p = ep.dir / "derived" / stage / filename
    if not ok or not p.is_file():
        return None
    return _load_arrays(p, members, prep)


def _span(n: int, start: int | None, end: int | None) -> tuple[int, int]:
    """Processed-frame window [s0, s1) clipped to [0, n)."""
    s0 = 0 if start is None else min(max(start, 0), n)
    return s0, (n if end is None else max(s0, min(end, n)))


def _manifest_size(ep: Episode, s: Stream) -> tuple[int, int] | None:
    """Extracted frame size of one stream from derived/frames/manifest.json (C4), for overlays without body2d/hands v2.
    Read only while frames is done or stale (--drop-frames keeps the manifest of the extraction the overlays came from)."""
    p = ep.dir / "derived" / "frames" / "manifest.json"
    if (ep.status.get("frames") or {}).get("state") not in ("done", "stale"):
        return None
    try:
        m = _read_json(p)
        m = (m.get("streams") or m)[s.name]  # {"streams": {s: {...}}} or {s: {...}}
        w, h = int(m["width"]), int(m["height"])
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None
    return (w, h) if w > 0 and h > 0 else None


def _prep_objects(d: dict) -> dict:
    """<U24 names [n,K] (96 bytes each) -> vocab [V] + int32 class ids [n,K] (-1 = empty slot), once per file."""
    names = d.pop("names").astype(str)
    vocab, idx = np.unique(names, return_inverse=True)
    idx = idx.reshape(names.shape).astype(np.int32)
    idx[names == ""] = -1
    return {**d, "vocab": vocab, "idx": idx}


def _prep_world3d(d: dict) -> dict:
    return {k: v for k, v in d.items() if k not in ("body_err", "body_track")}  # not sent to the browser


_HAND_MEMBERS = ("schema", "lm2d", "present", "score", "partner_lm2d", "partner_present", "img_w", "img_h")
OVERLAY_STAGES, BODY3D_STAGES, WORLD3D_STAGES, IMU_STAGES = ("body2d", "hands", "objects"), ("body3d",), ("world3d", "headpose", "calib"), ("imu_arm",)


def overlay_sources(ep: Episode, s: Stream) -> list[Path]:
    d = ep.dir / "derived"
    return [ep.dir / "episode.json", d / "body2d" / f"{s.name}.npz", d / "hands" / f"{s.name}.npz", d / "objects" / f"{s.name}.npz",
            d / "frames" / "manifest.json"]


def overlay_payload(ep: Episode, s: Stream, start: int | None = None, end: int | None = None, fresh: dict[str, bool] | None = None) -> dict[str, Any]:
    """Everything drawn on one view for processed frames [start, end), in extracted-frame pixels (img_w x img_h).

    body2d [n,P,17,3] (x, y, conf; COCO-17). hands2d [n,2,21,2]: the WEARER's hands, slot 0 left / 1 right (hands v2),
    or MediaPipe handedness slots (v1); partner_hands2d (v2 only): other people's hands seen in this view. objects
    [n,K,5] (x0, y0, x1, y1, conf) with object_idx [n,K] into object_vocab (-1 = empty). Absent = null. Arrays are
    sliced to the window BEFORE any conversion (a 1 h episode is never rounded whole for a 2 min window).
    """
    out: dict[str, Any] = {"stream": s.name, "proc_fps": ep.proc_fps, "common_start_s": ep.common_start_s, "offset_s": s.offset_s}
    b = _npz(ep, "body2d", f"{s.name}.npz", ("kpts", "img_w", "img_h"), fresh=fresh)
    h = _npz(ep, "hands", f"{s.name}.npz", _HAND_MEMBERS, fresh=fresh)
    o = _npz(ep, "objects", f"{s.name}.npz", ("boxes", "names"), _prep_objects, fresh=fresh)
    n = max([len(x) for x in (b and b["kpts"], h and h["lm2d"], o and o["boxes"]) if x is not None], default=0)
    s0, s1 = _span(n, start, end)
    if b is not None:
        kp = b["kpts"][s0:s1]
        assert kp.ndim == 4 and kp.shape[2:] == (17, 3), f"body2d kpts shape {kp.shape}"
        out["body2d"] = np.concatenate([_r(kp[..., :2], 1), _r(kp[..., 2:], 2)], axis=-1)
        out["img_w"], out["img_h"] = int(b["img_w"]), int(b["img_h"])
    if h is not None:
        v2 = "schema" in h and int(h["schema"]) >= 2
        present = h["present"][s0:s1].astype(bool) if v2 else np.nan_to_num(h["score"][s0:s1]) > 0  # v1: score == 0 means absent
        lm = _r(h["lm2d"][s0:s1], 1)
        assert lm.shape[1:] == (2, 21, 2) and present.shape == lm.shape[:2], f"hands lm2d {lm.shape} present {present.shape}"
        lm[~present] = np.nan
        out["hands2d"], out["hands_schema"] = lm, 2 if v2 else 1
        if "img_w" not in out and "img_w" in h:  # v2 hands carry the extracted frame size too
            out["img_w"], out["img_h"] = int(h["img_w"]), int(h["img_h"])
        if v2 and "partner_lm2d" in h:
            pl = _r(h["partner_lm2d"][s0:s1], 1)
            pl[~h["partner_present"][s0:s1].astype(bool)] = np.nan
            out["partner_hands2d"] = pl
    if o is not None:
        bx = o["boxes"][s0:s1]
        out["objects"] = np.concatenate([_r(bx[..., :4], 1), _r(bx[..., 4:], 2)], axis=-1)
        out["object_vocab"], out["object_idx"] = o["vocab"].tolist(), o["idx"][s0:s1]
    if "img_w" not in out and (size := _manifest_size(ep, s)):
        out["img_w"], out["img_h"] = size
    out.update(start=s0, end=s1, n_frames=n)
    return out


def body3d_sources(ep: Episode) -> list[Path]:
    return [ep.dir / "episode.json", ep.dir / "derived" / "body3d" / "body3d.npz"]


def body3d_payload(ep: Episode, start: int | None = None, end: int | None = None, fresh: dict[str, bool] | None = None) -> dict[str, Any]:
    """Monocular MediaPipe world landmarks world [n,2,33,3]: metres, hip-centred, camera-aligned axes x right, y down,
    z AWAY from the camera (a right-handed frame). Absent when the body3d stage is not current (e.g. skipped: no exo)."""
    b = _npz(ep, "body3d", "body3d.npz", ("world", "vis", "stream", "view"), fresh=fresh)
    if b is None:
        return {"available": False}
    n = len(b["world"]); s0, s1 = _span(n, start, end)
    out = {"available": True, "stream": str(b["stream"] if "stream" in b else b.get("view", "")), "proc_fps": ep.proc_fps,
           "axes": "MediaPipe world landmarks: metres, hip-centred; x right, y down, z away from the camera",
           "world": _r(b["world"][s0:s1], 3), "start": s0, "end": s1, "n_frames": n}
    if "vis" in b:
        out["vis"] = _r(b["vis"][s0:s1], 2)
    return out


def world3d_sources(ep: Episode) -> list[Path]:
    d = ep.dir / "derived"
    return [ep.dir / "episode.json", d / "world3d" / "world3d.npz", d / "calib" / "calib.json"] + \
        [d / "headpose" / f"{s.name}.npz" for s in ep.egos()]


def _pose_or_nan(T: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    T = np.asarray(T, dtype=np.float64).copy()
    assert T.ndim == 3 and T.shape[1:] == (4, 4), f"pose array shape {T.shape}"
    if valid is not None:
        T[~np.asarray(valid, bool)] = np.nan
    return _r(T, 4)


def world3d_payload(ep: Episode, start: int | None = None, end: int | None = None, fresh: dict[str, bool] | None = None) -> dict[str, Any]:
    """Board-frame world (metres; X right, Y down the board, Z into the board).

    bodies [n,2,17,3] with body_person [2] (ego linked to each slot, "" if none); heads {ego: [n,3]}; head_poses {ego:
    [n,4,4] T_world_cam, board <- camera, OpenCV camera axes} from world3d.npz T_world_cam_<ego>, else headpose/<ego>.npz;
    objects {name: [n,3]}; hands3d {ego: [n,2,21,3]}; features {name: [n]}; cameras {fixed cam: T_world_cam 4x4}
    (registered cameras only); headpose_backends.
    """
    w = _npz(ep, "world3d", "world3d.npz", None, _prep_world3d, fresh=fresh)
    if w is None:
        return {"available": False}
    n = len(w["bodies"]); s0, s1 = _span(n, start, end)
    out: dict[str, Any] = {"available": True, "proc_fps": ep.proc_fps, "frame": "board (X right, Y down the board, Z into the board), metres",
                           "bodies": _r(w["bodies"][s0:s1], 3), "heads": {}, "head_poses": {}, "objects": {}, "hands3d": {}, "features": {},
                           "start": s0, "end": s1, "n_frames": n}
    for key, a in w.items():
        if key.startswith("T_world_cam_"):
            out["head_poses"][key[12:]] = _pose_or_nan(a[s0:s1])
        elif key.startswith("head_"):
            out["heads"][key[5:]] = _r(a[s0:s1], 3)
        elif key.startswith("object_"):
            out["objects"][key[7:]] = _r(a[s0:s1], 3)
        elif key.startswith("hands3d_"):
            out["hands3d"][key[8:]] = _r(a[s0:s1], 3)
        elif key.startswith("feat_"):
            out["features"][key[5:]] = _r(a[s0:s1], 3)
    backends = {}
    for s in ep.egos():
        hp = _npz(ep, "headpose", f"{s.name}.npz", ("T_world_cam", "valid", "backend"), fresh=fresh)
        if hp is None:
            continue
        backends[s.name] = str(hp["backend"])
        if s.name not in out["head_poses"] and "T_world_cam" in hp and len(hp["T_world_cam"]) == n:
            out["head_poses"][s.name] = _pose_or_nan(hp["T_world_cam"][s0:s1], hp["valid"][s0:s1] if "valid" in hp else None)
    out["headpose_backends"], out["cameras"] = backends, {}
    if "body_person" in w:
        out["body_person"] = [str(x) for x in w["body_person"].tolist()]
    c = ep.dir / "derived" / "calib" / "calib.json"
    if (fresh["calib"] if fresh is not None and "calib" in fresh else _fresh(ep, "calib")) and c.is_file():
        try:
            cams = _read_json(c).get("cameras", {})
        except (OSError, ValueError, AttributeError):
            cams = {}
        for k, v in cams.items():
            T = np.asarray(v.get("T_world_cam") if isinstance(v, dict) and v.get("registered", True) else None, dtype=np.float64)
            if T.shape == (4, 4) and np.isfinite(T).all():
                out["cameras"][k] = _r(T, 4)
    return out


def imu_arm_sources(ep: Episode, s: Stream) -> list[Path]:
    return [ep.dir / "episode.json", ep.dir / "derived" / "imu_arm" / f"{s.name}.npz"]


def imu_arm_payload(ep: Episode, s: Stream, start: int | None = None, end: int | None = None, fresh: dict[str, bool] | None = None) -> dict[str, Any]:
    """IMU arm chain (Eidon 7-slot): t_s in reference-clock seconds (v2: the original, irregular IMU sample times, with
    gaps), points [T,2 sides,4 joints,3] metres in the harness scene frame (y up, chest-relative); samples with
    valid == False are null. period_s = median IMU period (draw a sample only within 0.75 period of the playhead);
    imu_sync = "estimated" | "manual" | "unsynced" | "drift_suspected" | "legacy_unverified" (the last three: the IMU
    clock was not verified against the video; "legacy_unverified" = a v1 file from before the IMU sync check). A frame
    window [start, end) selects the samples covering those processed frames' reference times (plus one each side)."""
    a = _npz(ep, "imu_arm", f"{s.name}.npz", ("t_s", "points", "valid", "elbow_flex_deg", "period_s", "imu_sync", "imu_offset_s", "imu_drift_ppm"),
             fresh=fresh)
    if a is None:
        return {"available": False}
    t = np.asarray(a["t_s"], np.float64)
    lo, hi = 0, len(t)
    fps = ep.proc_fps if _num(ep.proc_fps) and ep.proc_fps > 0 and _num(ep.common_start_s) else None
    if fps and (start is not None or end is not None):
        lo = int(np.searchsorted(t, ep.common_start_s + (start or 0) / fps, "left")) - 1
        hi = len(t) if end is None else int(np.searchsorted(t, ep.common_start_s + end / fps, "right")) + 1
        lo, hi = max(lo, 0), min(max(hi, 0), len(t))
    pts = np.asarray(a["points"][lo:hi], np.float64)  # a copy (float32 source)
    if "valid" in a:
        pts[~a["valid"][lo:hi].astype(bool)] = np.nan
    out = {"available": True, "stream": s.name, "t_s": _r(t[lo:hi], 3), "points": _r(pts, 3), "n_samples": len(t),
           "imu_sync": str(a["imu_sync"]) if "imu_sync" in a else "legacy_unverified"}
    if "elbow_flex_deg" in a:
        out["elbow_flex_deg"] = _r(a["elbow_flex_deg"][lo:hi], 1)
    dt = np.diff(t)
    out["period_s"] = float(a["period_s"]) if "period_s" in a else (float(np.median(dt)) if len(dt) else None)  # v1: 24 Hz grid
    for k in ("imu_offset_s", "imu_drift_ppm"):
        if k in a:
            out[k] = a[k].item() if a[k].ndim == 0 else a[k].tolist()
    return out


def _stat_sig(paths: list[Path]) -> tuple:
    sig = []
    for p in paths:
        try:
            st = p.stat()
            sig.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            sig.append((str(p), None, None))
    return tuple(sig)


def _payload(request: Request, name: str, key: str, stages: tuple[str, ...], sources: Callable[[Episode], list[Path]],
             build: Callable[[Episode, dict[str, bool]], Any]) -> Response:
    """ETag = hash(payload key, source files' (mtime, size) taken BEFORE the episode snapshot the body is built from,
    stage freshness); 304 on If-None-Match; gzip encoded once and cached. A body is cached only if no source changed
    while it was built, so a stale body never sits under a fresh ETag."""
    cfg = _cfg(request)
    srcs = sources(_load_episode(cfg, name))  # 404s; which files matter
    sig = _stat_sig(srcs)
    ep = _load_episode(cfg, name)  # snapshot at least as new as `sig`
    fs = fresh_set(ep)
    fresh = {st: st in fs for st in stages}
    h = hashlib.sha256(repr((PAYLOAD_VERSION, key, sig, sorted(fresh.items()))).encode()).hexdigest()[:32]
    etag, last = f'W/"{h}"', max((m / 1e9 for _, m, _ in sig if m), default=0.0)
    headers = {"ETag": etag, "Cache-Control": "private, no-cache", "Vary": "Accept-Encoding"}
    if last:
        headers["Last-Modified"] = formatdate(last, usegmt=True)
    inm = request.headers.get("if-none-match", "")
    if inm and (inm.strip() == "*" or etag in {t.strip() for t in inm.split(",")}):
        return Response(status_code=304, headers=headers)
    gz = "gzip" in request.headers.get("accept-encoding", "").lower()
    body = _BODIES.get((etag, gz))
    if body is None:
        raw = dumps(build(ep, fresh)).encode()
        body = gzip.compress(raw, 6) if gz else raw
        if _stat_sig(srcs) == sig:
            _BODIES.put((etag, gz), body)
        else:  # a source changed during the build: serve it once, uncached and without a validator
            headers = {"Cache-Control": "no-store", "Vary": "Accept-Encoding"}
    if gz:
        headers["Content-Encoding"] = "gzip"
    return Response(body, media_type="application/json", headers=headers)


Start = Annotated[int | None, Query(ge=0, description="first processed frame (inclusive)")]
End = Annotated[int | None, Query(ge=0, description="last processed frame (exclusive)")]


@router.get("/api/episode/{name}/overlay/{stream}")
def overlay(request: Request, name: str, stream: str, start: Start = None, end: End = None):
    return _payload(request, name, f"overlay|{name}|{stream}|{start}|{end}", OVERLAY_STAGES, lambda ep: overlay_sources(ep, _stream(ep, stream)),
                    lambda ep, fresh: overlay_payload(ep, _stream(ep, stream), start, end, fresh))


@router.get("/api/episode/{name}/body3d")
def body3d(request: Request, name: str, start: Start = None, end: End = None):
    return _payload(request, name, f"body3d|{name}|{start}|{end}", BODY3D_STAGES, body3d_sources,
                    lambda ep, fresh: body3d_payload(ep, start, end, fresh))


@router.get("/api/episode/{name}/world3d")
def world3d(request: Request, name: str, start: Start = None, end: End = None):
    return _payload(request, name, f"world3d|{name}|{start}|{end}", WORLD3D_STAGES, world3d_sources,
                    lambda ep, fresh: world3d_payload(ep, start, end, fresh))


@router.get("/api/episode/{name}/imu_arm/{stream}")
def imu_arm(request: Request, name: str, stream: str, start: Start = None, end: End = None):
    return _payload(request, name, f"imu_arm|{name}|{stream}|{start}|{end}", IMU_STAGES, lambda ep: imu_arm_sources(ep, _stream(ep, stream)),
                    lambda ep, fresh: imu_arm_payload(ep, _stream(ep, stream), start, end, fresh))


@router.api_route("/api/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
@router.api_route("/episodes/{rest:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"], include_in_schema=False)
def not_found(rest: str):
    """Nothing under /api/ or /episodes/ falls through to the static UI mount (e.g. a decoded "../..")."""
    raise HTTPException(404, "not found")
