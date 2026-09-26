"""Playground server (auth, CSRF, uploads, payloads) and static exporter tests.

Every test builds its episodes in tmp_path and points the app there (create_app(root) / --episodes-root); the repo's
data/ is only ever read (one test copies the committed eidon_10004 JSON/npz into tmp_path). Pipeline jobs are stubbed.
"""
from __future__ import annotations

import asyncio
import errno
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("python_multipart")
from fastapi.testclient import TestClient  # noqa: E402

from duet.playground import run as prun  # noqa: E402
from duet.playground import server  # noqa: E402
from duet.playground.episode import Episode  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
TOKEN = "correct-horse-battery-staple-42"
LOCAL = {"base_url": "http://127.0.0.1:8765", "client": ("127.0.0.1", 50000)}
CSRF = {"X-Playground-Request": "1"}
MP4 = b"\x00\x00\x00\x18ftypisom\x00\x00\x02\x00isomiso2" + b"\x00" * 64  # ISO-BMFF header; content is never decoded
N = 30  # processed frames of the synthetic episode (3 s at 10 fps)


# ---------------------------------------------------------------- fixtures

@pytest.fixture
def jobs(monkeypatch):
    """Stub run.submit/is_running/queue_state/recover_stale: nothing is ever executed."""
    st = {"busy": set(), "submitted": [], "recovered": []}

    def submit(ep_dir, stages=None, force=False, **kw):
        k = os.path.realpath(ep_dir)
        if k in st["busy"]:
            return False
        st["busy"].add(k); st["submitted"].append((Path(ep_dir).name, stages, force))
        return True
    monkeypatch.setattr(prun, "submit", submit)
    monkeypatch.setattr(prun, "is_running", lambda d: os.path.realpath(d) in st["busy"])
    monkeypatch.setattr(prun, "queue_state", lambda: {"max_jobs": 1, "queued": [], "running": [
        {"episode": Path(k).name, "stages": None, "force": False, "queued": 1.0, "started": 2.0, "pid": 4242, "log": "/abs/secret.log"} for k in st["busy"]]})
    monkeypatch.setattr(prun, "recover_stale", lambda root: st["recovered"].append(Path(root)) or [])
    return st


@pytest.fixture
def env(monkeypatch):
    for k in ("PLAYGROUND_TOKEN", "PLAYGROUND_MAX_UPLOAD_MB", "PLAYGROUND_MIN_FREE_MB", "PLAYGROUND_EPISODES"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("PLAYGROUND_MIN_FREE_MB", "1")
    return monkeypatch


def client(app, **kw):
    return TestClient(app, raise_server_exceptions=False, **{**LOCAL, **kw})


def f32(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float32)


def make_episode(root: Path, name: str = "synth", *, hands: str = "v2", video: bytes | Path = MP4, window=(0.5, 3.5),
                 status: dict | None = None, extra: dict | None = None, world_poses: bool = False, imu_v2: bool = False,
                 stream_extra: dict | None = None) -> Path:
    """Old-layout episode.json (as the committed episodes) + v1/v2 derived outputs, all synthetic."""
    d = root / name; (d / "streams").mkdir(parents=True)
    for s in ("ego", "cam"):
        if isinstance(video, Path):
            shutil.copy(video, d / "streams" / f"{s}.mp4")
        else:
            (d / "streams" / f"{s}.mp4").write_bytes(video)
    st = {k: {"state": "done", "detail": "ok", "pid": 4242, "host": "gpu-box", "log": "derived/logs/x.log"}  # legacy: no fingerprint
          for k in ("probe", "align", "frames", "calib", "body2d", "hands", "objects", "headpose", "body3d", "world3d", "imu_arm", "qc")}
    st.update(status or {})
    ej = {"name": name, "streams": [
        {"name": "ego", "role": "ego", "path": "streams/ego.mp4", "person": "alice", "fps": 30.0, "width": 320, "height": 240,
         "duration_s": 4.0, "has_audio": True, "offset_s": 0.0, "offset_confidence": None, "imu": "imu/ego.parquet"},
        {"name": "cam", "role": "exo", "path": "streams/cam.mp4", "person": None, "fps": 30.0, "width": 320, "height": 240,
         "duration_s": 4.0, "has_audio": True, "offset_s": 0.5, "offset_confidence": 6.0, "imu": None}],
        "reference": "ego", "proc_fps": 10.0, "proc_size": 640, "status": st, "common_start_s": window[0], "common_end_s": window[1],
        "notes": ["cam: offset from audio"], **(extra or {})}
    for sd in ej["streams"]:
        sd.update(stream_extra or {})
    (d / "episode.json").write_text(json.dumps(ej))
    n = int(np.floor((window[1] - window[0]) * 10 + 1e-9))
    rng = np.random.default_rng(0)
    der = d / "derived"
    for sub in ("body2d", "hands", "objects", "body3d", "world3d", "calib", "headpose", "imu_arm", "qc", "frames"):
        (der / sub).mkdir(parents=True)
    kp = np.full((n, 4, 17, 3), np.nan, np.float32); kp[:, 0, :, :2] = 37.4 + rng.random((n, 17, 2)) * 100; kp[:, 0, :, 2] = 0.9
    np.savez_compressed(der / "body2d" / "cam.npz", kpts=kp, boxes=np.full((n, 4, 5), np.nan, np.float32), img_w=np.int64(640), img_h=np.int64(480))
    lm = f32(10.3 + rng.random((n, 2, 21, 2)) * 200)
    if hands == "v2":
        present = np.ones((n, 2), bool); present[0, 1] = False; lm[0, 1] = np.nan
        plm = f32(5.5 + rng.random((n, 2, 21, 2)) * 100); pp = np.zeros((n, 2), bool); pp[:, 0] = True; plm[~pp] = np.nan
        np.savez_compressed(der / "hands" / "ego.npz", schema=np.int64(2), lm2d=lm, lm3d=f32(rng.random((n, 2, 21, 3)) * 0.1), score=f32(np.full((n, 2), 0.9)),
                            present=present, partner_lm2d=plm, partner_lm3d=f32(np.zeros((n, 2, 21, 3))), partner_score=f32(np.full((n, 2), np.nan)),
                            partner_present=pp, n_detected=np.full(n, 3, np.int64), lm3d_origin=np.str_("wrist"), img_w=np.int64(640), img_h=np.int64(480))
    else:  # v1: no schema key, score == 0 means absent (landmarks NaN there)
        score = np.full((n, 2), 0.9, np.float32); score[0, 1] = 0.0; lm[0, 1] = np.nan
        np.savez_compressed(der / "hands" / "ego.npz", lm2d=lm, lm3d=f32(rng.random((n, 2, 21, 3))), score=score)
    boxes = np.full((n, 12, 5), np.nan, np.float32); boxes[:, 0] = [10.04, 20.0, 110.0, 220.0, 0.8765]
    names = np.full((n, 12), "", "<U24"); names[:, 0] = "bowl"
    np.savez_compressed(der / "objects" / "ego.npz", boxes=boxes, names=names)
    np.savez_compressed(der / "body3d" / "body3d.npz", world=f32(rng.random((n, 2, 33, 3)) - 0.5), img2d=f32(np.zeros((n, 2, 33, 3))),
                        vis=f32(np.ones((n, 2, 33))), stream=np.str_("cam"))
    T = np.tile(np.eye(4, dtype=np.float32), (n, 1, 1)); T[:, :3, 3] = [0.1, -0.2, 1.5]; valid = np.ones(n, bool); valid[1] = False
    np.savez_compressed(der / "headpose" / "ego.npz", T_world_cam=T, valid=valid, backend=np.str_("head_tag"))
    np.savez_compressed(der / "world3d" / "world3d.npz", bodies=f32(rng.random((n, 2, 17, 3))), body_err=f32(np.zeros((n, 2, 17))),
                        head_ego=f32(np.tile([0.1, -0.2, 1.5], (n, 1))), object_bowl=f32(np.full((n, 3), np.nan)),
                        hands3d_ego=f32(rng.random((n, 2, 21, 3))), feat_head_dist_m=f32(np.full(n, 0.8)), body_person=np.array(["ego", ""]),
                        body_track=np.zeros((n, 2), np.int32), **({"T_world_cam_ego": T} if world_poses else {}))
    (der / "calib" / "calib.json").write_text(json.dumps({"cameras": {
        "cam": {"T_world_cam": np.eye(4).tolist(), "registered": True}, "nominal": {"T_world_cam": np.eye(4).tolist(), "registered": False},
        "unseen": {"T_world_cam": None}}}))
    t = np.arange(0, 4, 1 / 24.0)
    if imu_v2:  # original irregular IMU times with a 0.5 s gap, sync state, no 24 Hz grid
        t = np.concatenate([np.arange(0, 1.5, 0.041), np.arange(2.0, 4, 0.041)]) + 0.003
    pts = f32(rng.random((len(t), 2, 4, 3))); v = np.ones((len(t), 2), bool); v[0, 1] = False
    extra_imu = {"schema": np.int64(2), "period_s": np.float64(0.041), "imu_sync": np.str_("unsynced"), "imu_offset_s": np.float64(0.0)} if imu_v2 else \
        {"elbow_flex_deg": f32(np.full((len(t), 2), 45.25))}
    np.savez_compressed(der / "imu_arm" / "ego.npz", t_s=t, points=pts, valid=v, chest_yaw=f32(np.zeros(len(t))), **extra_imu)
    (der / "qc" / "report.json").write_text(json.dumps({"streams": {"ego": {"good_frame_percent": float("nan"), "stability_score": 0.5}},
                                                        "episode": {"both_visible_ratio": float("inf")}}))
    (der / "frames" / "manifest.json").write_text(json.dumps({"streams": {"ego": {"width": 640, "height": 480}, "cam": {"width": 640, "height": 480}}}))
    return d


@pytest.fixture
def root(tmp_path):
    r = tmp_path / "episodes"; r.mkdir()
    make_episode(r)
    (tmp_path / "sibling_canary.txt").write_text("CANARY-SIBLING")
    (tmp_path / "outside").mkdir(); (tmp_path / "outside" / "canary.txt").write_text("CANARY-OUTSIDE")
    return r


def upload(c, name="up1", files=(("GX010001.MP4", MP4),), roles="ego", headers=CSRF, **fields):
    data = {"name": name, "roles": roles, **fields}
    return c.post("/api/upload", data=data, files=[("files", (fn, content, "video/mp4")) for fn, content in files], headers=headers)


def leftovers(root: Path) -> list[str]:
    """Upload staging dirs (ours: .upload-*) and anything Episode.create left in its .staging dir."""
    return sorted([p.name for p in root.glob(".upload-*")] + [p.name for p in root.glob(".staging/*")])


# ---------------------------------------------------------------- access control

def test_no_token_means_loopback_only(root, env, jobs):
    app = server.create_app(root, token=None)
    assert client(app).get("/api/episodes").status_code == 200
    assert client(app, base_url="http://10.1.2.3:8765", client=("10.1.2.3", 5555)).get("/api/episodes").status_code == 403
    assert client(app, client=("192.168.1.20", 5555)).get("/").status_code == 403  # the UI too
    c = client(app)
    assert c.get("/api/episodes", headers={"Host": "attacker.example:8765"}).status_code == 403  # DNS rebinding
    assert c.get("/api/episodes", headers={"X-Forwarded-For": "8.8.8.8"}).status_code == 403  # behind a proxy -> needs a token
    assert c.get("/api/episodes", headers={"Host": "localhost:8765"}).status_code == 200
    assert client(app, client=("::ffff:127.0.0.1", 5555)).get("/api/episodes").status_code == 200  # dual-stack loopback
    assert client(app, client=("::ffff:10.0.0.9", 5555)).get("/api/episodes").status_code == 403


def test_token_auth_bearer_header_and_session_cookie(root, env, jobs):
    app = server.create_app(root, token=TOKEN)
    c = client(app, base_url="http://10.1.2.3:8765", client=("10.1.2.3", 5555))  # remote clients are fine with a token
    assert c.get("/").status_code == 200 and c.get("/app.js").status_code == 200  # the UI shell holds no data
    for url in ("/api/episodes", "/api/episode/synth", "/api/episode/synth/video/ego", "/episodes/synth/streams/ego.mp4", "/api/queue"):
        r = c.get(url)
        assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer", url
    assert c.get("/api/episodes", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    assert c.get("/api/episodes", headers={"X-Playground-Token": TOKEN}).status_code == 200
    assert c.get("/api/episodes", headers={"Authorization": "Bearer wrong-token-wrong-token"}).status_code == 401
    assert c.get("/api/session").json() == {"auth_required": True, "authenticated": False}
    r = c.post("/api/session", json={"token": TOKEN}, headers=CSRF)
    assert r.status_code == 200
    sc = r.headers["set-cookie"]
    assert "HttpOnly" in sc and "SameSite=strict" in sc.replace("Strict", "strict") and "Path=/" in sc and "Secure" not in sc  # http here
    cookie = {"Cookie": sc.split(";")[0]}
    c.cookies.clear()
    r = c.get("/api/episode/synth/video/ego", headers={**cookie, "Range": "bytes=0-9"})
    assert r.status_code == 206 and r.content == MP4[:10]  # <video> media works with the cookie, Range included
    assert c.get("/episodes/synth/streams/ego.mp4", headers=cookie).status_code == 200
    assert c.get("/api/session", headers=cookie).json()["authenticated"] is True
    forged = cookie["Cookie"][:-4] + ("0000" if not cookie["Cookie"].endswith("0000") else "1111")
    assert c.get("/api/episodes", headers={"Cookie": forged}).status_code == 401
    cfg = app.state.pg; cfg.session_ttl_s = -5; expired = server._new_session(cfg); cfg.session_ttl_s = 3600
    assert c.get("/api/episodes", headers={"Cookie": f"{server.SESSION_COOKIE}={expired}"}).status_code == 401
    assert c.delete("/api/session", headers=CSRF).status_code == 200


def test_login_rate_limit(root, env, jobs):
    c = client(server.create_app(root, token=TOKEN))
    codes = [c.post("/api/session", json={"token": "nope-nope-nope-nope"}, headers=CSRF).status_code for _ in range(11)]
    assert codes == [401] * 10 + [429]
    assert c.post("/api/session", json={"token": TOKEN}, headers=CSRF).status_code == 429  # still blocked for this minute


def test_serve_refuses_public_bind_without_token(env, tmp_path, monkeypatch):
    uvicorn = pytest.importorskip("uvicorn")
    calls = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kw: calls.append(kw))
    for host in ("0.0.0.0", "::", "192.168.1.5", "example.org"):
        with pytest.raises(SystemExit):
            server.serve(host=host, episodes_root=tmp_path)
    assert calls == []
    server.serve(host="127.0.0.1", port=9999, episodes_root=tmp_path)
    monkeypatch.setenv("PLAYGROUND_TOKEN", TOKEN)
    server.serve(host="0.0.0.0", port=9999, episodes_root=tmp_path)
    assert [c["host"] for c in calls] == ["127.0.0.1", "0.0.0.0"] and calls[0]["log_level"] == "info"
    with pytest.raises(ValueError):
        server.create_app(tmp_path, token="short")


def test_csrf_rules(root, env, jobs):
    c = client(server.create_app(root, token=None))
    assert c.post("/api/episode/synth/run").status_code == 403  # a cross-site <form> cannot add the header
    assert c.post("/api/episode/synth/run", headers={**CSRF, "Origin": "http://evil.example"}).status_code == 403
    assert c.post("/api/episode/synth/run", headers={**CSRF, "Origin": "http://127.0.0.1:9999"}).status_code == 403
    assert c.post("/api/episode/synth/run", headers={**CSRF, "Origin": "null"}).status_code == 403
    assert c.post("/api/episode/synth/run", headers={**CSRF, "Sec-Fetch-Site": "cross-site"}).status_code == 403
    assert c.post("/api/episode/synth/run", headers={**CSRF, "Sec-Fetch-Site": "same-site"}).status_code == 403
    assert c.get("/api/episode/synth/video/ego", headers={"Sec-Fetch-Site": "cross-site"}).status_code == 403  # no embedding
    assert upload(c, headers={}).status_code == 403
    r = c.post("/api/episode/synth/run", headers={**CSRF, "Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin"})
    assert r.status_code == 202
    ct = client(server.create_app(root, token=TOKEN))
    assert ct.post("/api/episode/synth/run?force=true", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 403  # auth != CSRF proof
    assert ct.post("/api/episode/synth/run?force=true", headers={"X-Playground-Token": TOKEN}).status_code == 409  # already queued above


def test_path_traversal_blocked(root, env, jobs):
    (root / ".upload-deadbeef").mkdir(); (root / ".upload-deadbeef" / "00.mp4").write_bytes(MP4)
    (root / "_upload_old").mkdir(); (root / "_upload_old" / "a.mp4").write_bytes(MP4)
    urls = ["/api/episode/..", "/api/episode/%2e%2e", "/api/episode/..%2f", "/api/episode/%2e%2e%2f%2e%2e", "/api/episode/..%5c..",
            "/api/episode/synth%00", "/api/episode/synth/video/..%2f..%2fepisode.json", "/api/episode/synth/overlay/..%2fego",
            "/episodes/../sibling_canary.txt", "/episodes/%2e%2e/sibling_canary.txt", "/episodes/..%2fsibling_canary.txt",
            "/episodes/synth/../../sibling_canary.txt", "/episodes/synth/%2e%2e/%2e%2e/outside/canary.txt", "/episodes//etc/hosts",
            "/episodes/synth/episode.json", "/episodes/synth/derived/body2d/cam.npz", "/episodes/.upload-deadbeef/00.mp4",
            "/episodes/_upload_old/a.mp4", "/episodes/synth/streams/../episode.json"]
    app = server.create_app(root, token=None)
    for u in urls:
        status, body = raw_get(app, u)  # sent as is: httpx would normalise "../" away before the server sees it
        assert status in (400, 404), (u, status)
        assert b"CANARY" not in body and b'"streams"' not in body and b"<html" not in body, u
    assert raw_get(app, "/episodes/synth/streams/ego.mp4") == (200, MP4)  # the one thing /episodes/ still serves


def raw_get(app, raw: str, root_path: str = "", headers: tuple = (), client: str = "127.0.0.1", method: str = "GET") -> tuple[int, bytes]:
    """A request through the ASGI app exactly as uvicorn delivers it: path = unquote(raw) (dot segments kept) and, with
    --root-path, path = root_path + request path and scope["root_path"] = root_path. `headers` are raw byte pairs."""
    import anyio
    from urllib.parse import unquote
    out = {"status": 0, "body": b""}

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(m):
        if m["type"] == "http.response.start":
            out["status"] = m["status"]
        elif m["type"] == "http.response.body":
            out["body"] += m.get("body", b"")
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http", "path": unquote(raw),
             "raw_path": raw.encode(), "root_path": root_path, "query_string": b"", "headers": [(b"host", b"127.0.0.1:8765"), *headers],
             "client": (client, 50000), "server": ("127.0.0.1", 8765), "state": {}}
    anyio.run(app, scope, receive, send)
    return out["status"], out["body"]


def test_security_headers(root, env, jobs):
    c = client(server.create_app(root, token=None))
    for u in ("/", "/api/episodes", "/api/episode/nope"):
        h = c.get(u).headers
        assert h["x-content-type-options"] == "nosniff" and h["x-frame-options"] == "DENY" and "frame-ancestors 'none'" in h["content-security-policy"]
        assert "googleapis" not in h["content-security-policy"] and "gstatic" not in h["content-security-policy"]
    page = c.get("/").text
    assert "fonts.googleapis.com" not in page and "fonts.gstatic.com" not in page  # no third-party font requests
    assert re.search(r'<button id="cancelRun"[^>]*\shidden[\s>]', page)  # Cancel exists in the server UI, hidden until a job runs


# ---------------------------------------------------------------- uploads

def test_upload_dedupes_names_moves_files_and_sanitises(root, env, jobs):
    c = client(server.create_app(root, token=None))
    r = upload(c, "dorm handovers!", files=(("GX010001.MP4", MP4 + b"FRONT"), ("GX010001.MP4", MP4 + b"BACK"), ("../../evil\\head.mov", MP4)),
               roles="exo,exo,ego", persons=',,<b id=m1>marker</b>', fps="12.5", reference="head")
    assert r.status_code == 201, r.text
    j = r.json()
    assert j["name"] == "dorm_handovers" and j["streams"] == ["GX010001", "GX010001_2", "head"] and j["queued"] is True
    ep = Episode.load(root / "dorm_handovers")
    assert (ep.dir / "streams/GX010001.mp4").read_bytes().endswith(b"FRONT") and (ep.dir / "streams/GX010001_2.mp4").read_bytes().endswith(b"BACK")
    assert ep.stream("head").path == "streams/head.mov" and ep.reference == "head" and ep.proc_fps == 12.5
    person = ep.stream("head").person
    assert person and "<" not in person and ">" not in person and "=" not in person
    assert not any(p.is_symlink() for p in (ep.dir / "streams").iterdir())
    assert leftovers(root) == [] and jobs["submitted"] == [("dorm_handovers", None, False)]
    listed = c.get("/api/episode/dorm_handovers").json()
    assert all(re.fullmatch(r"[A-Za-z0-9_-]+", s["name"]) and re.fullmatch(r"streams/[A-Za-z0-9_-]+\.(mp4|mov)", s["path"]) for s in listed["streams"])


@pytest.mark.parametrize("kw, status", [
    ({"persons": "alice"}, 422),  # 2 files, 1 person: nothing may be dropped silently
    ({"roles": "ego"}, 422),
    ({"roles": "ego,head"}, 422),
    ({"fps": "nan"}, 422), ({"fps": "inf"}, 422), ({"fps": "1e6"}, 422), ({"fps": "0"}, 422), ({"fps": "-3"}, 422), ({"fps": "abc"}, 422),
    ({"reference": "nonexistent"}, 422),
    ({"name": "  !!  "}, 422),
    ({"files": (("cam.x\"><b id=m3>", MP4), ("b.mp4", MP4))}, 422),  # extension allowlist
    ({"files": (("notes.txt", MP4), ("b.mp4", MP4))}, 422),
    ({"files": (("a.mp4", b"AAAA" * 10), ("b.mp4", MP4))}, 422),  # not a video container
    ({"files": (("a.mp4", b""), ("b.mp4", MP4))}, 422),
])
def test_upload_validation(root, env, jobs, kw, status):
    c = client(server.create_app(root, token=None))
    args = {"name": "bad", "files": (("a.mp4", MP4), ("b.mp4", MP4)), "roles": "ego,exo", **kw}
    r = upload(c, args.pop("name"), args.pop("files"), args.pop("roles"), **args)
    assert r.status_code == status, r.text
    assert isinstance(r.json()["detail"], str)
    assert not (root / "bad").exists() and leftovers(root) == [] and jobs["submitted"] == []


def test_upload_wrong_content_type_and_existing_name(root, env, jobs):
    c = client(server.create_app(root, token=None))
    assert c.post("/api/upload", data={"name": "x"}, headers=CSRF).status_code == 415
    bad = c.post("/api/upload", content=b"--b0undary42\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nx\r\n--b0undary42--garbage" * 2,
                 headers={**CSRF, "Content-Type": "multipart/form-data; boundary=b0undary42"})
    assert bad.status_code in (400, 422) and leftovers(root) == []  # malformed body: a client error, never a 500
    assert c.post("/api/upload", content=b"--b0undary42\r\nContent-Disposition: form-data; name=\"name\"\r\n\r\nx",
                  headers={**CSRF, "Content-Type": "multipart/form-data; boundary=b0undary42"}).status_code == 400  # truncated
    assert upload(c, "synth").status_code == 409  # refused before the files are stored
    (root / "debris" / "derived").mkdir(parents=True); (root / "debris" / "derived" / "junk.npz").write_bytes(b"old")
    assert c.get("/api/episode/debris").status_code == 404 and "debris" not in [e["name"] for e in c.get("/api/episodes").json()]
    r = upload(c, "debris")  # a dir without episode.json (old failed upload) counts as absent
    assert r.status_code == 201, r.text
    assert not (root / "debris" / "derived" / "junk.npz").exists()
    assert upload(c, "debris").status_code == 409
    (root / "orphan" / "streams").mkdir(parents=True); (root / "orphan" / "streams" / "only_copy.mp4").write_bytes(b"precious")
    r = upload(c, "orphan")  # ...unless it holds real stream files (maybe the only copy): refused, and the reason names them
    assert r.status_code == 409 and "streams/only_copy.mp4" in r.json()["detail"] and (root / "orphan/streams/only_copy.mp4").read_bytes() == b"precious"
    assert leftovers(root) == []


def test_upload_size_cap(root, env, jobs):
    env.setenv("PLAYGROUND_MAX_UPLOAD_MB", "1")
    c = client(server.create_app(root, token=None))
    big = MP4 + b"\x00" * (1 << 20)
    assert upload(c, "big", files=(("a.mp4", big),)).status_code == 413  # Content-Length check
    body, ctype = _multipart({"name": "big2", "roles": "ego"}, [("a.mp4", big)])
    r = c.post("/api/upload", content=(body[i:i + 65536] for i in range(0, len(body), 65536)), headers={**CSRF, "Content-Type": ctype})
    assert r.status_code == 413  # chunked body without Content-Length: the streaming counter stops it
    assert not (root / "big").exists() and not (root / "big2").exists() and leftovers(root) == []


def test_upload_disk_checks_and_cleanup_on_failure(root, env, jobs, monkeypatch):
    c = client(server.create_app(root, token=None))
    real = shutil.disk_usage
    monkeypatch.setattr(server.shutil, "disk_usage", lambda p: real(p)._replace(free=0))
    assert upload(c, "full").status_code == 507
    monkeypatch.setattr(server.shutil, "disk_usage", real)

    def enospc(*a, **k):
        raise OSError(errno.ENOSPC, "No space left on device")
    monkeypatch.setattr(Episode, "create", classmethod(lambda cls, *a, **k: enospc()))
    r = upload(c, "full2", files=(("a.mp4", MP4), ("b.mp4", MP4)), roles="ego,exo")
    assert r.status_code == 507
    assert not (root / "full2").exists() and leftovers(root) == [] and jobs["submitted"] == []

    def running(*a, **k):
        raise RuntimeError("overwrite refused: a pipeline is running")
    monkeypatch.setattr(Episode, "create", classmethod(lambda cls, *a, **k: running()))
    assert upload(c, "busy").status_code == 409 and leftovers(root) == []


def test_upload_parsing_runs_off_the_event_loop(root, env, jobs, monkeypatch):
    seen = []
    real_feed = server._MultipartSink.feed

    def feed(self, data):
        try:
            asyncio.get_running_loop(); seen.append("event-loop")
        except RuntimeError:
            seen.append("worker")
        return real_feed(self, data)
    monkeypatch.setattr(server._MultipartSink, "feed", feed)
    assert upload(client(server.create_app(root, token=None)), "offloop").status_code == 201
    assert seen and set(seen) == {"worker"}


def _multipart(fields: dict, files: list[tuple[str, bytes]]) -> tuple[bytes, str]:
    b = "b0undary42"
    parts = [f'--{b}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode() for k, v in fields.items()]
    parts += [f'--{b}\r\nContent-Disposition: form-data; name="files"; filename="{fn}"\r\nContent-Type: video/mp4\r\n\r\n'.encode() + data + b"\r\n"
              for fn, data in files]
    return b"".join(parts) + f"--{b}--\r\n".encode(), f"multipart/form-data; boundary={b}"


# ---------------------------------------------------------------- episodes, run queue, robustness

def test_nan_becomes_null_and_unknown_stream_404(tmp_path, env, jobs):
    r = tmp_path / "eps"; r.mkdir()
    make_episode(r, extra={"proc_fps": float("nan")})
    c = client(server.create_app(r, token=None))
    d = c.get("/api/episode/synth")
    assert d.status_code == 200 and "NaN" not in d.text and "Infinity" not in d.text
    j = d.json()
    assert j["proc_fps"] is None and j["n_frames"] is None
    assert j["qc"]["streams"]["ego"]["good_frame_percent"] is None and j["qc"]["episode"]["both_visible_ratio"] is None
    assert c.get("/api/episodes").status_code == 200
    ov = c.get("/api/episode/synth/overlay/ego")
    assert ov.status_code == 200 and "NaN" not in ov.text and ov.json()["proc_fps"] is None
    for u in ("/api/episode/synth/overlay/nope", "/api/episode/synth/video/nope", "/api/episode/synth/imu_arm/nope", "/api/episode/nope",
              "/api/episode/nope/overlay/ego"):
        assert c.get(u).status_code == 404, u


def test_corrupt_episode_json_does_not_hide_the_others(root, env, jobs):
    (root / "broken").mkdir(); (root / "broken" / "episode.json").write_text('{"name": "broken", "streams": [')
    c = client(server.create_app(root, token=None))
    r = c.get("/api/episodes")
    assert r.status_code == 200
    by = {e["name"]: e for e in r.json()}
    assert "error" not in by["synth"] and "unreadable" in by["broken"]["error"]
    r = c.get("/api/episode/broken")
    assert r.status_code == 500 and "unreadable" in r.json()["detail"]


def test_run_queue_dedupe_and_status(root, env, jobs):
    with TestClient(server.create_app(root, token=None), raise_server_exceptions=False, **LOCAL) as c:
        assert jobs["recovered"] == [root.resolve()]  # recover_stale at startup
        r = c.post("/api/episode/synth/run?stages=hands,qc&force=true", headers=CSRF)
        assert r.status_code == 202 and r.json()["queued"] is True
        assert jobs["submitted"] == [("synth", ["hands", "qc"], True)]
        q = r.json()["queue"]
        assert q["running"][0]["episode"] == "synth" and "pid" not in q["running"][0] and "log" not in q["running"][0]
        assert c.post("/api/episode/synth/run", headers=CSRF).status_code == 409
        assert c.post("/api/episode/synth/run?stages=hands,bogus", headers=CSRF).status_code == 422
        d = c.get("/api/episode/synth").json()
        assert d["running"] is True and d["stages"][:3] == ["probe", "align", "frames"] and d["n_frames"] == N
        assert [s["offset_status"] for s in d["streams"]] == ["reference", "audio"]  # legacy episode.json, normalised on load
        assert d["status"]["hands"] == {"state": "done", "detail": "ok"}  # no pid/host/fingerprint/log internals
        assert all(s["usable"] for s in d["streams"])
        assert c.get("/api/queue").json()["running"][0]["episode"] == "synth"


STUB_STAGES = """
import time
from duet.playground.run import StageSpec


def probe(ep):
    ep.set_status("probe", "running"); ep.set_status("probe", "done", "stub")


def slow(ep):  # runs until <episode>/go exists
    ep.set_status("slow", "running")
    for _ in range(1200):
        if (ep.dir / "go").exists():
            break
        time.sleep(0.05)
    ep.set_status("slow", "done", "slow ok")


SLOW = {"probe": StageSpec(probe), "slow": StageSpec(slow, deps=("probe",))}
"""


def _wait(pred, timeout: float = 60.0) -> None:
    t0 = time.time()
    while not pred():
        assert time.time() - t0 < timeout, "timed out"
        time.sleep(0.05)


@pytest.fixture
def real_jobs(tmp_path, monkeypatch):
    """The real run queue (one worker thread, one subprocess per job) with a stub stage registry, through fx-core's
    test hook run.JOB_EXTRA_ARGS = ["--specs", "module:ATTR"]. Every started episode is released at teardown."""
    (tmp_path / "pg_srv_stub.py").write_text(STUB_STAGES)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(tmp_path), str(REPO / "src")]))
    monkeypatch.delenv("PLAYGROUND_MAX_JOBS", raising=False)
    monkeypatch.setattr(prun, "JOB_EXTRA_ARGS", ["--specs", "pg_srv_stub:SLOW"])
    eps: list[Path] = []
    yield eps
    for d in eps:
        (d / "go").write_text("1"); prun.cancel(d)
    _wait(lambda: not any(prun.is_running(d) for d in eps))


def test_cancel_queued_and_running_jobs(root, env, real_jobs):
    make_episode(root, "second")
    real_jobs.extend([root / "synth", root / "second"])
    c = client(server.create_app(root, token=None))
    state = lambda n, st: (c.get(f"/api/episode/{n}").json()["status"].get(st) or {}).get("state")  # noqa: E731
    assert c.post("/api/episode/synth/cancel").status_code == 403  # same CSRF rule as /run
    assert client(server.create_app(root, token=TOKEN)).post("/api/episode/synth/cancel", headers=CSRF).status_code == 401  # and auth
    assert c.post("/api/episode/synth/cancel", headers=CSRF).status_code == 409  # nothing queued or running
    assert c.post("/api/episode/nope/cancel", headers=CSRF).status_code == 404
    assert c.post("/api/episode/synth/run", headers=CSRF).status_code == 202
    _wait(lambda: state("synth", "slow") == "running")  # the stub job's subprocess is inside the "slow" stage
    assert c.post("/api/episode/second/run", headers=CSRF).status_code == 202  # queued behind it (one worker)
    assert [j["episode"] for j in c.get("/api/queue").json()["queued"]] == ["second"]
    r = c.post("/api/episode/second/cancel", headers=CSRF)  # queued job: dropped
    assert r.status_code == 202 and r.json()["queue"]["queued"] == [] and c.get("/api/episode/second").json()["running"] is False
    assert c.post("/api/episode/synth/cancel", headers=CSRF).status_code == 202  # running job: SIGTERM to its process group
    _wait(lambda: not c.get("/api/episode/synth").json()["running"])
    assert state("synth", "slow") == "interrupted" and state("second", "slow") is None
    assert c.post("/api/episode/synth/cancel", headers=CSRF).status_code == 409


# ---------------------------------------------------------------- payloads

def test_overlay_payload_compact_windowed_cached(root, env, jobs):
    c = client(server.create_app(root, token=None))
    r = c.get("/api/episode/synth/overlay/ego", headers={"Accept-Encoding": "gzip"})
    assert r.status_code == 200 and r.headers["content-encoding"] == "gzip" and r.headers["etag"].startswith('W/"')
    assert "private" in r.headers["cache-control"] and r.headers["last-modified"]
    floats = re.findall(r"-?\d+\.(\d+)", r.text)
    assert floats and max(len(f) for f in floats) <= 2  # float32 noise (37.400001525878906) is gone
    o = r.json()
    assert o["img_w"] == 640 and o["img_h"] == 480  # no body2d for ego: frame size from frames/manifest.json
    assert (o["start"], o["end"], o["n_frames"]) == (0, N, N) and len(o["hands2d"]) == N
    assert o["hands_schema"] == 2 and o["hands2d"][0][1][0][0] is None and o["hands2d"][0][0][0][0] is not None
    assert o["partner_hands2d"][0][0][0][0] is not None and o["partner_hands2d"][0][1][0][0] is None
    assert o["object_vocab"][o["object_idx"][0][0]] == "bowl" and o["object_idx"][0][1] == -1 and o["objects"][0][0] == [10.0, 20.0, 110.0, 220.0, 0.88]
    assert c.get("/api/episode/synth/overlay/ego", headers={"If-None-Match": r.headers["etag"]}).status_code == 304
    w = c.get("/api/episode/synth/overlay/ego?start=10&end=15").json()
    assert (w["start"], w["end"], w["n_frames"]) == (10, 15, N) and len(w["hands2d"]) == 5 and len(w["objects"]) == 5
    assert c.get("/api/episode/synth/overlay/ego?start=-1").status_code == 422
    body = c.get("/api/episode/synth/overlay/cam").json()
    assert body["img_w"] == 640 and len(body["body2d"]) == N and body["body2d"][0][0][0][2] == 0.9 and "hands2d" not in body
    z = root / "synth/derived/hands/ego.npz"; os.utime(z, (z.stat().st_atime, z.stat().st_mtime + 5))
    assert c.get("/api/episode/synth/overlay/ego").headers["etag"] != r.headers["etag"]  # source change -> new ETag


def test_hands_v1_fallback(tmp_path, env, jobs):
    r = tmp_path / "eps"; r.mkdir(); make_episode(r, hands="v1")
    o = client(server.create_app(r, token=None)).get("/api/episode/synth/overlay/ego").json()
    assert o["hands_schema"] == 1 and "partner_hands2d" not in o and (o["img_w"], o["img_h"]) == (640, 480)  # v1: size from the manifest
    (r / "synth/derived/frames/manifest.json").unlink()
    assert "img_w" not in client(server.create_app(r, token=None)).get("/api/episode/synth/overlay/ego").json()  # old episode: no size known
    assert o["hands2d"][0][1][0][0] is None and o["hands2d"][0][0][0][0] is not None


def test_stage_gating_body3d_world3d_imu(tmp_path, env, jobs):
    r = tmp_path / "eps"; r.mkdir()
    make_episode(r, "skipped", status={"body3d": {"state": "skipped", "detail": "no exo"}, "world3d": {"state": "failed"}, "hands": {"state": "stale"}})
    make_episode(r, "full")
    c = client(server.create_app(r, token=None))
    assert c.get("/api/episode/skipped/body3d").json() == {"available": False}  # stale file on disk is ignored
    assert c.get("/api/episode/skipped/world3d").json() == {"available": False}
    assert "hands2d" not in c.get("/api/episode/skipped/overlay/ego").json()
    make_episode(r, "noframes", hands="v1", status={"frames": {"state": "skipped", "detail": "needs align"}})  # old manifest still on disk
    assert "img_w" not in c.get("/api/episode/noframes/overlay/ego").json()
    (r / "full/derived/frames/manifest.json").unlink()
    assert c.get("/api/episode/full/overlay/ego").json()["img_w"] == 640  # hands v2 carry img_w/img_h themselves
    b = c.get("/api/episode/full/body3d?start=2&end=4").json()
    assert b["available"] and b["stream"] == "cam" and len(b["world"]) == 2 and "z away from the camera" in b["axes"]
    w = c.get("/api/episode/full/world3d").json()
    assert w["available"] and len(w["bodies"]) == N and list(w["heads"]) == ["ego"] and list(w["objects"]) == ["bowl"]
    assert list(w["cameras"]) == ["cam"]  # unregistered and missing poses are not drawn
    assert w["head_poses"]["ego"][0][0][3] == pytest.approx(0.1) and w["head_poses"]["ego"][1][0][0] is None  # headpose valid == False -> null
    assert w["headpose_backends"] == {"ego": "head_tag"} and w["features"]["head_dist_m"][0] == pytest.approx(0.8)
    assert w["body_person"] == ["ego", ""] and "body_track" not in w and "body_err" not in w
    make_episode(r, "posed", world_poses=True)  # world3d.npz carries T_world_cam_<ego> itself (fx-world layout)
    wp = c.get("/api/episode/posed/world3d?start=1&end=3").json()
    assert len(wp["head_poses"]["ego"]) == 2 and wp["head_poses"]["ego"][0][0][0] == 1.0  # from world3d (no valid mask applied)
    a = c.get("/api/episode/full/imu_arm/ego").json()
    assert a["available"] and a["points"][0][1][0][0] is None and a["elbow_flex_deg"][0][0] in (45.2, 45.3)
    aw = c.get("/api/episode/full/imu_arm/ego?start=10&end=20").json()  # frames 10-20 = ref time 1.5-2.5 s
    assert aw["t_s"][0] <= 1.5 <= aw["t_s"][1] and aw["t_s"][-2] <= 2.5 <= aw["t_s"][-1]
    assert c.get("/api/episode/full/imu_arm/cam").json() == {"available": False}
    assert a["period_s"] == pytest.approx(1 / 24) and a["imu_sync"] == "legacy_unverified"  # v1: 24 Hz grid, clock never verified
    make_episode(r, "imu2", imu_v2=True)
    a2 = c.get("/api/episode/imu2/imu_arm/ego").json()
    assert a2["period_s"] == 0.041 and a2["imu_sync"] == "unsynced" and "elbow_flex_deg" not in a2
    g = c.get("/api/episode/imu2/imu_arm/ego?start=14&end=20").json()["t_s"]  # ref 1.9-2.5 s: straddles the 1.5-2.0 s gap
    assert g[0] < 1.5 and g[1] >= 2.0 and g[-1] >= 2.5  # one sample of margin each side, the gap is not filled


def test_committed_episode_copy_still_served(tmp_path, env, jobs):
    src = REPO / "data/playground/episodes/eidon_10004"
    if not (src / "episode.json").is_file():
        pytest.skip("committed eidon_10004 episode not present")
    dst = tmp_path / "eps" / "eidon_10004"
    shutil.copytree(src, dst, symlinks=True, ignore=shutil.ignore_patterns("*.jpg", "*.parquet"))
    c = client(server.create_app(tmp_path / "eps", token=None))
    assert [e["name"] for e in c.get("/api/episodes").json()] == ["eidon_10004"]
    d = c.get("/api/episode/eidon_10004").json()
    assert d["n_frames"] == 617 and d["streams"][0]["usable"] is True
    o = c.get("/api/episode/eidon_10004/overlay/ego").json()
    assert o["img_w"] == 640 and len(o["body2d"]) == 617 and o["hands_schema"] == 1
    assert c.get("/api/episode/eidon_10004/body3d").json()["available"] is True
    assert c.get("/api/episode/eidon_10004/imu_arm/ego").json()["available"] is True
    src_video = (dst / "streams" / "ego.mp4")
    assert c.get("/api/episode/eidon_10004/video/ego").status_code == (200 if src_video.exists() else 404)  # dangling absolute symlink -> clean 404


# ---------------------------------------------------------------- static exporter

@pytest.fixture(scope="module")
def exporter():
    spec = importlib.util.spec_from_file_location("playground_export_static", REPO / "scripts/playground_export_static.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _ffmpeg_clip(path: Path, seconds: float = 4.0) -> Path:
    """Video whose frame n has luma 2n (to check the trim mapping) plus an audio track (must not be published)."""
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                    f"nullsrc=size=64x48:rate=30:duration={seconds},geq=lum='2*N':cb=128:cr=128", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-c:v", "libx264", "-g", "30", "-pix_fmt", "yuv420p", "-c:a", "aac",
                    "-shortest", str(path)], check=True)
    return path


def _first_frame_luma(path: Path, w: int = 64, h: int = 48) -> float:
    """Mean of the raw Y plane (not "gray", which rescales limited-range luma) of the first decoded frame."""
    raw = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(path), "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-"],
                         capture_output=True, check=True).stdout
    return float(np.frombuffer(raw[: w * h], np.uint8).mean())


@pytest.fixture
def site(tmp_path, env):
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not on PATH")
    root = tmp_path / "eps"; root.mkdir()
    clip = _ffmpeg_clip(tmp_path / "clip.mp4")
    make_episode(root, "synth", video=clip, window=(1.5, 3.0), status={"objects": {"state": "failed", "detail": "Traceback: /Users/someone/secret/x.py"}})
    pub = tmp_path / "publish.json"
    pub.write_text(json.dumps({"episodes": {"synth": {"license": "CC-BY-4.0", "consent_basis": "test", "attribution": "Synthetic test clip, CC-BY-4.0",
                                                      "notes": "internal note"},
                                            "unknown_terms": {"license": "UNKNOWN - see review", "consent_basis": "UNKNOWN", "attribution": "UNKNOWN",
                                                              "notes": "n/a"},
                                            "approved_unknown": {"license": "UNKNOWN - see review", "consent_basis": "UNKNOWN", "attribution": "UNKNOWN",
                                                                 "notes": "n/a", "approved": {"by": "repo owner", "date": "2026-09-26",
                                                                                              "note": "kept live by decision"}}}}))
    out = tmp_path / "site"
    (out / "old_episode").mkdir(parents=True); (out / "old_episode" / "episode.json").write_text("{}")
    (out / "episodes.json").write_text("[]"); (out / ".vercel").mkdir(); (out / ".vercel" / "project.json").write_text('{"projectId": "prj_x"}')
    (out / ".duet-playground-export").write_text("{}")  # written by this exporter on a previous run
    return {"root": root, "pub": pub, "out": out, "clip": clip, "args": ["--out", str(out), "--episodes-root", str(root), "--publish", str(pub)]}


def test_export_end_to_end(exporter, site):
    out = site["out"]
    assert exporter.main(site["args"] + ["--episodes", "synth"]) == 0
    assert not (out / "old_episode").exists()  # stale content removed
    assert json.loads((out / ".vercel/project.json").read_text()) == {"projectId": "prj_x"}  # Vercel link preserved
    html, js = (out / "index.html").read_text(), (out / "app.js").read_text()
    assert "/api/" not in html and "/api/" not in js and "SERVER-ONLY" not in html + js
    assert "uploadForm" not in html and "Files are stored on this server" not in html and 'id="runAll"' not in html and 'id="cancelRun"' not in html
    assert '<script id="playground-static" type="application/json">' in html and 'integrity="sha384-' in html
    assert f'<script src="app.js?v={hashlib.sha256(js.encode()).hexdigest()[:12]}"></script>' in html and "upRef" not in html
    assert not re.search(r"\bapi/(episodes?|upload|session|queue)\b", html + js) and (out / ".duet-playground-export").is_file()
    cfg = json.loads(re.search(r'<script id="playground-static" type="application/json">(.*?)</script>', html).group(1))
    assert cfg["static"] is True and cfg["attributions"] == [{"episode": "synth", "attribution": "Synthetic test clip, CC-BY-4.0", "license": "CC-BY-4.0"}]
    for f in ("episodes.json", "version.json", "vercel.json", "robots.txt", "synth/episode.json", "synth/overlay_ego.json", "synth/overlay_cam.json",
              "synth/body3d.json", "synth/world3d.json", "synth/imu_ego.json"):
        assert (out / f).is_file(), f
    assert json.loads((out / "synth/world3d.json").read_text())["available"] is True
    v = json.loads((out / "version.json").read_text())
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", v["git_commit"]) and v["episodes"] == ["synth"]
    assert "googleapis" not in html and "gstatic" not in html
    vc = json.loads((out / "vercel.json").read_text())
    _check_vercel_schema(vc)
    keys = {h["key"]: h["value"] for rule in vc["headers"] for h in rule["headers"]}
    assert keys["Access-Control-Allow-Origin"] == "https://plg.duetlabs.co"  # replaces Vercel's default "*"
    assert "googleapis" not in keys["Content-Security-Policy"] and "gstatic" not in keys["Content-Security-Policy"]
    assert "frame-ancestors 'none'" in keys["Content-Security-Policy"] and keys["X-Frame-Options"] == "DENY"
    assert keys["X-Content-Type-Options"] == "nosniff" and keys["Referrer-Policy"] == "no-referrer" and "noindex" in keys["X-Robots-Tag"]
    cache = {r["source"]: r["headers"][0]["value"] for r in vc["headers"] if r["headers"][0]["key"] == "Cache-Control"}
    assert cache["/"] == "no-cache" and "immutable" in cache["/(.*)\\.mp4"]
    ep = json.loads((out / "synth/episode.json").read_text())
    assert ep["status"]["objects"] == {"state": "failed"} and ep["license"] == "CC-BY-4.0" and ep["attribution"].startswith("Synthetic")
    everything = "".join(p.read_text() for p in out.rglob("*.json"))
    assert "/Users/someone" not in everything and "internal note" not in everything and "consent_basis" not in everything
    streams = {s["name"]: s for s in ep["streams"]}
    assert streams["ego"]["trim_start_s"] == pytest.approx(1.0) and streams["cam"]["trim_start_s"] == pytest.approx(0.5)  # window 1.5 s - 0.5 s margin - offset
    assert streams["ego"]["imu"] is True and streams["cam"]["imu"] is False
    for s in streams.values():
        mp4 = out / "synth" / s["path"]
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-show_entries", "format=duration:stream=codec_type", str(mp4)],
                                         capture_output=True, text=True, check=True).stdout)
        assert [x["codec_type"] for x in info["streams"]] == ["video"]  # audio dropped
        assert float(info["format"]["duration"]) == pytest.approx(2.5, abs=0.1)  # window 1.5 s + 2 x 0.5 s margins
        n0 = _first_frame_luma(mp4) / 2  # source frame index of the clip's first frame
        assert abs(n0 / 30.0 - s["trim_start_s"]) <= 1.5 / 30, (s["name"], n0)  # exported t=0 == stream time trim_start_s


def test_export_allowlist_and_license_gate(exporter, site, capsys):
    with pytest.raises(SystemExit, match="allowlist"):
        exporter.main(site["args"] + ["--episodes", "synth,not_listed"])
    for n in ("unknown_terms", "approved_unknown"):
        make_episode(site["root"], n, video=site["clip"], window=(1.5, 3.0))
    before = sorted(p.name for p in site["out"].iterdir())
    assert exporter.main(site["args"]) == exporter.EXIT_EPISODES_FAILED  # unapproved UNKNOWN license -> nothing swapped
    cap = capsys.readouterr()
    assert "FAILED  unknown_terms" in cap.out and 'no "approved" record' in cap.out and "approved_unknown" not in cap.out.split("FAILED")[-1]
    assert "--allow-partial" in cap.err and "unknown_terms would then DISAPPEAR" in cap.err
    assert sorted(p.name for p in site["out"].iterdir()) == before
    assert exporter.main(site["args"] + ["--episodes", "synth,approved_unknown"]) == 0  # explicit approval -> published, with a warning
    assert "publishing with license UNKNOWN on the explicit approval by repo owner (2026-09-26)" in capsys.readouterr().err
    assert exporter.main(site["args"] + ["--allow-partial"]) == 0  # publish the rest
    assert sorted(e["name"] for e in json.loads((site["out"] / "episodes.json").read_text())) == ["approved_unknown", "synth"]
    assert exporter.main(site["args"] + ["--allow-unknown-license"]) == 0
    assert (site["out"] / "unknown_terms/episode.json").is_file()
    pub = json.loads(site["pub"].read_text()); pub["episodes"]["approved_unknown"]["approved"] = {"by": "", "date": "yesterday"}
    site["pub"].write_text(json.dumps(pub))
    with pytest.raises(SystemExit, match="approved"):
        exporter.main(site["args"])


def test_repo_allowlist_publishes_the_live_episodes(exporter, capsys):
    allow = exporter.load_allowlist(exporter.PUBLISH)
    assert set(allow) == {"comind_43276420_clip", "eidon_10004"}
    for name, meta in allow.items():
        assert exporter.refusal(name, meta) is None, name  # known license or the owner's explicit approval
        assert meta["approved"]["by"] == "repo owner"
    assert exporter.unknown(allow["comind_43276420_clip"]["license"]) and exporter.unknown(allow["comind_43276420_clip"]["consent_basis"])
    assert "WARNING comind_43276420_clip: publishing with license UNKNOWN on the explicit approval" in capsys.readouterr().err


def test_vercel_config_origin(exporter):
    vc = exporter.vercel_config("https://duet-playground.vercel.app")
    _check_vercel_schema(vc)
    assert vc["headers"][0]["headers"][0] == {"key": "Access-Control-Allow-Origin", "value": "https://duet-playground.vercel.app"}
    with pytest.raises(SystemExit, match="site-origin"):
        exporter.main(["--site-origin", "*", "--out", "/nonexistent-site-dir"])


def _check_vercel_schema(vc: dict) -> None:
    """The parts of https://openapi.vercel.sh/vercel.json that apply (additionalProperties false throughout)."""
    assert set(vc) <= {"$schema", "headers"} and isinstance(vc["headers"], list) and len(vc["headers"]) <= 2048
    for rule in vc["headers"]:
        assert set(rule) <= {"source", "headers", "has", "missing"} and {"source", "headers"} <= set(rule)
        assert isinstance(rule["source"], str) and len(rule["source"]) <= 4096 and len(rule["headers"]) <= 1024
        for h in rule["headers"]:
            assert set(h) == {"key", "value"} and isinstance(h["key"], str) and isinstance(h["value"], str) and len(h["value"]) <= 32768


def test_export_missing_source_fails_early_and_keeps_site(exporter, site, capsys):
    s = site["root"] / "synth" / "streams" / "ego.mp4"; s.unlink(); s.symlink_to("/nonexistent/elsewhere/ego.mp4")
    assert exporter.main(site["args"] + ["--episodes", "synth"]) == exporter.EXIT_EPISODES_FAILED
    cap = capsys.readouterr()
    assert "FAILED  synth" in cap.out and "stream ego: source video synth/streams/ego.mp4 missing (dangling symlink -> /nonexistent/elsewhere/ego.mp4)" in cap.out
    assert "site NOT updated (unchanged): 1 of 1 requested episodes failed: synth" in cap.err
    assert (site["out"] / "old_episode").exists() and not list(site["out"].parent.glob(".site.build-*"))
    make_episode(site["root"], "approved_unknown", video=site["clip"], window=(1.5, 3.0))
    assert exporter.main(site["args"] + ["--episodes", "synth,approved_unknown"]) == exporter.EXIT_EPISODES_FAILED
    assert "To publish only approved_unknown, re-run with --allow-partial; synth would then DISAPPEAR from the site." in capsys.readouterr().err


def test_export_reuses_cache_until_source_changes(exporter, site, monkeypatch):
    assert exporter.main(site["args"] + ["--episodes", "synth"]) == 0
    first = json.loads((site["out"] / "synth/episode.json").read_text())["streams"][0]["path"]
    calls = []
    real = exporter.runtime.run_ffmpeg
    monkeypatch.setattr(exporter.runtime, "run_ffmpeg", lambda *a, **k: calls.append(a) or real(*a, **k))
    assert exporter.main(site["args"] + ["--episodes", "synth"]) == 0 and calls == []  # verified cache hit
    src = site["root"] / "synth/streams/ego.mp4"; st = src.stat(); os.utime(src, (st.st_atime, st.st_mtime + 10))
    assert exporter.main(site["args"] + ["--episodes", "synth"]) == 0 and len(calls) == 1  # only the changed source
    assert json.loads((site["out"] / "synth/episode.json").read_text())["streams"][0]["path"] != first
    assert len(list((site["out"].with_name("site.cache")).glob("*.mp4"))) == 2  # unused cache entries pruned (+ their trim sidecars)
    assert len(list((site["out"].with_name("site.cache")).glob("*.json"))) == 2


def test_export_refuses_dangerous_out_and_unbalanced_markers(exporter, site, tmp_path):
    for bad in (REPO, REPO / "src" / "x", site["root"], Path.home()):
        with pytest.raises(SystemExit):
            exporter.check_out(bad, site["root"])
    stranger = tmp_path / "stranger"; stranger.mkdir(); (stranger / "notes.txt").write_text("mine")
    with pytest.raises(SystemExit, match="not marked as a playground export"):
        exporter.check_out(stranger, site["root"])
    with pytest.raises(SystemExit):
        exporter.strip_blocks("a /* SERVER-ONLY BEGIN */ b", *exporter.JS_BLOCK, "x.js")
    assert exporter.strip_blocks("a /* SERVER-ONLY BEGIN */ /api/ /* SERVER-ONLY END */ c", *exporter.JS_BLOCK, "x.js") == "a  c"


# ---------------------------------------------------------------- code review 2026-09-26: security

@pytest.mark.parametrize("root_path", ["/pg", "/"])
def test_auth_decides_on_the_route_path_under_a_root_path(root, env, jobs, root_path):
    app, pre = server.create_app(root, token=TOKEN), root_path  # uvicorn: scope path = root_path + request path
    for u in ("/api/episodes", "/api/episode/synth", "/api/episode/synth/video/ego", "/episodes/synth/streams/ego.mp4", "/api/queue", "/favicon.ico"):
        assert raw_get(app, pre + u, root_path=root_path)[0] == 401, u  # default-deny whatever the prefix
    assert raw_get(app, pre + "/api/episode/synth/cancel", root_path=root_path, method="POST", headers=((b"x-playground-request", b"1"),))[0] == 401
    ok = ((b"authorization", f"Bearer {TOKEN}".encode()),)
    assert raw_get(app, pre + "/api/episodes", root_path=root_path, headers=ok)[0] == 200  # the route itself still works
    assert raw_get(app, pre + "/", root_path=root_path)[0] == 200 and raw_get(app, pre + "/api/session", root_path=root_path)[0] == 200  # public
    assert raw_get(server.create_app(root, token=None), pre + "/api/episodes", root_path=root_path, client="10.0.0.7")[0] == 403


def test_non_ascii_credentials_are_401_not_500(root, env, jobs):
    app = server.create_app(root, token=TOKEN)
    for h in ((b"cookie", "pg_session=1.\u00e4.\u00e9".encode()), (b"cookie", b"pg_session=\xff\xfe"), (b"x-playground-token", "t\u00f6k\u00e9n".encode())):
        assert raw_get(app, "/api/episodes", headers=(h,))[0] == 401, h
    assert raw_get(app, "/api/session", headers=((b"cookie", "pg_session=\u00e4".encode()),))[0] == 200


def test_every_failed_credential_check_is_rate_limited(root, env, jobs):
    app = server.create_app(root, token=TOKEN)
    c, other = client(app), client(app, client=("127.0.0.2", 5555))
    bad = {"Authorization": "Bearer " + "x" * 24}
    assert [c.get("/api/episodes", headers=bad).status_code for _ in range(11)] == [401] * 10 + [429]
    assert c.get("/api/episodes", headers={"X-Playground-Token": TOKEN}).status_code == 429  # a right guess inside the lockout too
    assert c.post("/api/session", json={"token": TOKEN}, headers=CSRF).status_code == 429  # one budget for every entry point
    assert other.get("/api/episodes", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200  # per client IP
    app2 = server.create_app(root, token=TOKEN); cfg = app2.state.pg
    cfg.session_ttl_s = -5; old = server._new_session(cfg); cfg.session_ttl_s = 3600
    assert {client(app2).get("/api/episodes", headers={"Cookie": f"pg_session={old}"}).status_code for _ in range(15)} == {401}  # expiry is no guess


def test_fail_limiter_is_a_bounded_lru():
    lim = server._FailLimiter(limit=2, window_s=60, max_ips=3)
    for ip in ("a", "a", "b", "c"):
        lim.fail(ip)
    assert lim.blocked("a") and not lim.blocked("b")
    lim.fail("d")  # 4 IPs > 3: only the least recently failing IP is evicted, never the whole table
    assert list(lim.fails) == ["b", "c", "d"] and not lim.blocked("a")
    lim.fail("b")
    assert lim.blocked("b")


def test_token_whitespace_is_stripped(tmp_path, env, jobs):
    app = server.create_app(tmp_path, token=TOKEN + "\r\n")
    assert client(app).get("/api/episodes", headers={"Authorization": f"Bearer {TOKEN}"}).status_code == 200
    with pytest.raises(ValueError):
        server.create_app(tmp_path, token="  short-token-x  ")  # 13 characters once stripped
    env.setenv("PLAYGROUND_TOKEN", "  " + TOKEN + "\n")
    assert client(server.create_app(tmp_path)).post("/api/session", json={"token": TOKEN}, headers=CSRF).status_code == 200


# ---------------------------------------------------------------- code review 2026-09-26: correctness / robustness

def test_payloads_serve_only_fresh_stages(root, env, jobs, monkeypatch):
    monkeypatch.setattr(prun, "fresh_stages", lambda ep, specs=None: {"frames", "objects"})
    monkeypatch.setattr(prun, "stale_stages", lambda ep, specs=None: {"hands": "rig changed since this ran"})
    c = client(server.create_app(root, token=None))
    o = c.get("/api/episode/synth/overlay/ego").json()
    assert "hands2d" not in o and "objects" in o  # hands is "done" on disk but no longer current
    d = c.get("/api/episode/synth").json()
    assert d["qc"] is None and d["stale"] == {"hands": "rig changed since this ran"} and d["status"]["hands"]["state"] == "done"
    assert c.get("/api/episode/synth/body3d").json() == {"available": False}


def test_export_publishes_stale_stages_as_stale_without_their_data(exporter, site, monkeypatch):
    monkeypatch.setattr(prun, "fresh_stages", lambda ep, specs=None: {"probe", "align", "frames", "body2d", "objects", "qc", "imu_arm", "calib"})
    assert exporter.main(site["args"] + ["--episodes", "synth"]) == 0
    ep = json.loads((site["out"] / "synth/episode.json").read_text())
    assert ep["status"]["hands"] == {"state": "stale"} and ep["status"]["world3d"] == {"state": "stale"} and ep["status"]["body2d"] == {"state": "done"}
    assert "hands2d" not in json.loads((site["out"] / "synth/overlay_ego.json").read_text())
    assert json.loads((site["out"] / "synth/world3d.json").read_text()) == {"available": False}
    assert any("stale" in n and "hands" in n for n in ep["notes"])


def test_body_built_while_a_source_changed_is_not_cached(root, env, jobs, monkeypatch):
    c, z, real = client(server.create_app(root, token=None)), root / "synth/derived/hands/ego.npz", server.overlay_payload

    def racing(*a, **k):  # the pipeline rewrites hands while the body is being built
        out = real(*a, **k); st = z.stat(); os.utime(z, ns=(st.st_atime_ns, st.st_mtime_ns + 7_000_000_000))
        return out
    monkeypatch.setattr(server, "overlay_payload", racing)
    r = c.get("/api/episode/synth/overlay/ego")
    assert r.status_code == 200 and "etag" not in r.headers and r.headers["cache-control"] == "no-store"
    monkeypatch.setattr(server, "overlay_payload", real)
    r2 = c.get("/api/episode/synth/overlay/ego")
    assert c.get("/api/episode/synth/overlay/ego", headers={"If-None-Match": r2.headers["etag"]}).status_code == 304


def test_null_proc_fps_is_reported_not_500(tmp_path, env, jobs):
    r = tmp_path / "eps"; r.mkdir(); make_episode(r, extra={"proc_fps": None})
    c = client(server.create_app(r, token=None))
    d = c.get("/api/episode/synth")
    assert d.status_code == 200 and "proc_fps" in d.json()["problem"] and d.json()["n_frames"] is None
    e = c.get("/api/episodes").json()[0]
    assert "proc_fps" in e["problem"] and "error" not in e and e["duration_s"] is None
    for u in ("/api/episode/synth/overlay/ego?start=0&end=5", "/api/episode/synth/imu_arm/ego?start=0&end=5", "/api/episode/synth/world3d"):
        assert c.get(u).status_code == 200, u


def test_upload_with_an_invalid_boundary_is_400_and_leaves_nothing(root, env, jobs):
    c = client(server.create_app(root, token=None))
    r = c.post("/api/upload", content=b"x", headers={**CSRF, "Content-Type": "multipart/form-data; boundary=" + "b" * 200})
    assert r.status_code == 400 and leftovers(root) == []


def test_ui_shell_is_not_cached_and_the_script_is_versioned(root, env, jobs):
    c = client(server.create_app(root, token=TOKEN))  # the shell is public even with a token
    v = hashlib.sha256((REPO / "src/duet/playground/static/app.js").read_bytes()).hexdigest()[:12]
    r = c.get("/")
    assert r.status_code == 200 and r.headers["cache-control"] == "no-cache" and f'<script src="app.js?v={v}"></script>' in r.text
    assert c.get("/index.html").text == r.text and 'id="upRef"' in r.text
    js = c.get(f"/app.js?v={v}")
    assert js.status_code == 200 and js.headers["cache-control"] == "no-cache"


def test_media_is_never_gzipped(root, env, jobs):
    c = client(server.create_app(root, token=None))
    (root / "synth/streams/ego.mp4").write_bytes(MP4 + bytes(8192))  # above the gzip minimum size
    for h in ({"Accept-Encoding": "gzip"}, {"Accept-Encoding": "gzip", "Range": "bytes=0-4095"}):
        r = c.get("/api/episode/synth/video/ego", headers=h)
        assert r.status_code in (200, 206) and "content-encoding" not in r.headers and r.content[:24] == MP4[:24], h


def test_job_failure_record_is_surfaced(tmp_path, env, jobs):
    r = tmp_path / "eps"; r.mkdir()
    make_episode(r, status={"_job": {"state": "failed", "detail": "pipeline process exited with code 2 before any stage", "pid": 99, "host": "h"}})
    c = client(server.create_app(r, token=None))
    d = c.get("/api/episode/synth").json()
    assert d["job"] == {"state": "failed", "detail": "pipeline process exited with code 2 before any stage"}
    assert "_job" not in d["stages"] and "_job" not in d["status"] and c.get("/api/episodes").json()[0]["job"] == "failed"


# ---------------------------------------------------------------- code review 2026-09-26: efficiency

def test_windows_reuse_decoded_arrays_and_read_only_needed_members(root, env, jobs, monkeypatch):
    loads, real = [], np.load
    monkeypatch.setattr(np, "load", lambda *a, **k: loads.append(Path(a[0]).parent.name) or real(*a, **k))
    c = client(server.create_app(root, token=None))
    assert len(c.get("/api/episode/synth/overlay/ego?start=0&end=10").json()["hands2d"]) == 10
    first = sorted(loads)
    for w in ((10, 20), (20, 30), (5, 25)):
        assert c.get(f"/api/episode/synth/overlay/ego?start={w[0]}&end={w[1]}").json()["start"] == w[0]
    assert first == ["hands", "objects"] and len(loads) == 2  # decoded once; every other window is sliced from the cache
    d = server._load_arrays(root / "synth/derived/hands/ego.npz", server._HAND_MEMBERS)
    assert "lm3d" not in d and "partner_lm3d" not in d and not d["lm2d"].flags.writeable
    o = server._load_arrays(root / "synth/derived/objects/ego.npz", ("boxes", "names"), server._prep_objects)
    assert "names" not in o and o["idx"].dtype == np.int32 and o["vocab"].tolist() == ["", "bowl"]


# ---------------------------------------------------------------- code review 2026-09-26: exporter

def test_export_refuses_unmarked_dirs_and_adopts_only_old_sites(exporter, site, tmp_path):
    args = ["--episodes", "synth", "--episodes-root", str(site["root"]), "--publish", str(site["pub"])]
    repo_like = tmp_path / "vercel_repo"  # looks like "a previous export" by the old rule (.vercel, episodes.json)
    for f in (".vercel/project.json", "package.json", "src/app.ts", ".git/HEAD", "episodes.json"):
        (repo_like / f).parent.mkdir(parents=True, exist_ok=True); (repo_like / f).write_text("mine")
    with pytest.raises(SystemExit, match="not marked"):
        exporter.main(args + ["--out", str(repo_like)])
    with pytest.raises(SystemExit, match="never has"):
        exporter.main(args + ["--out", str(repo_like), "--adopt-old-export"])
    assert all((repo_like / f).read_text() == "mine" for f in ("package.json", "src/app.ts", ".git/HEAD"))
    old = tmp_path / "old_site"
    for f in ("index.html", "app.js", "episodes.json", "vercel.json", ".vercel/project.json", "ep1/episode.json", "ep1/overlay_a.json", "ep1/streams/a.mp4"):
        (old / f).parent.mkdir(parents=True, exist_ok=True); (old / f).write_text("x")
    with pytest.raises(SystemExit, match="not marked"):
        exporter.main(args + ["--out", str(old)])
    assert exporter.main(args + ["--out", str(old), "--adopt-old-export"]) == 0
    assert (old / ".duet-playground-export").is_file() and (old / ".vercel/project.json").read_text() == "x" and not (old / "ep1").exists()


def test_allowlist_keys_are_validated_and_deletes_stay_inside(exporter, tmp_path):
    pub = tmp_path / "pub.json"
    for key in ("..", "../x", "a/b", ".hidden", ""):
        pub.write_text(json.dumps({"episodes": {key: {"license": "CC0", "consent_basis": "x", "attribution": "x", "notes": "x"}}}))
        with pytest.raises(SystemExit, match="not a valid episode name"):
            exporter.load_allowlist(pub)
    inside = tmp_path / "b"; inside.mkdir(); (tmp_path / "keep.txt").write_text("keep")
    (inside / "link").symlink_to(tmp_path)
    for target in (inside / "..", tmp_path, inside, inside / "link"):
        with pytest.raises(RuntimeError):
            exporter._rmtree_inside(target, inside)
    assert (tmp_path / "keep.txt").exists()


@pytest.mark.parametrize("kind", ["mpegts", "mkv_offset", "mp4_video_offset"])
def test_export_trims_sources_with_a_start_offset(exporter, site, tmp_path, kind):
    src, base = tmp_path / f"src_{kind}", str(site["clip"])
    cmd = {"mpegts": ["-i", base, "-map", "0:v", "-c", "copy", "-f", "mpegts"],
           "mkv_offset": ["-itsoffset", "1.5", "-i", base, "-map", "0:v", "-c", "copy", "-avoid_negative_ts", "disabled", "-f", "matroska"],
           "mp4_video_offset": ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-itsoffset", "0.8", "-i", base, "-map", "0:a", "-map", "1:v",
                                "-t", "4.8", "-c:v", "copy", "-c:a", "aac", "-f", "mp4"]}[kind]
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *cmd, str(src)], check=True)
    v = json.loads(subprocess.run(["ffprobe", "-v", "error", "-of", "json", "-select_streams", "v:0", "-show_entries", "stream=start_time,index", str(src)],
                                  capture_output=True, text=True, check=True).stdout)["streams"][0]
    assert float(v["start_time"]) > 0.5  # the case the old -ss arithmetic put 1.5-2 s late
    root = tmp_path / "eps2"; root.mkdir()
    make_episode(root, "offset", video=src, window=(1.5, 3.0), stream_extra={"video_start_s": float(v["start_time"]), "video_index": int(v["index"])})
    pub = tmp_path / "pub2.json"
    pub.write_text(json.dumps({"episodes": {"offset": {"license": "CC0", "consent_basis": "test", "attribution": "test", "notes": "test"}}}))
    out = tmp_path / "site2"
    assert exporter.main(["--out", str(out), "--episodes-root", str(root), "--publish", str(pub)]) == 0
    for sm in json.loads((out / "offset/episode.json").read_text())["streams"]:
        n0 = _first_frame_luma(out / "offset" / sm["path"]) / 2  # source frame index shown at exported t=0
        assert abs(n0 / 30.0 - sm["trim_start_s"]) <= 1.5 / 30, (kind, sm["name"], n0, sm["trim_start_s"])


def test_git_version_counts_untracked_playground_files(exporter, monkeypatch):
    calls = []

    def fake_run(cmd, **kw):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, stdout="?? src/duet/playground/new.py\n" if "status" in cmd else "0" * 40 + "\n", stderr="")
    monkeypatch.setattr(exporter.subprocess, "run", fake_run)
    assert exporter.git_version() == {"git_commit": "0" * 40, "git_dirty": True}
    assert any("--untracked-files=all" in c for c in calls)
