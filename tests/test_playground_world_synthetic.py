"""End-to-end check of the world stages on a synthetic 3-camera episode rendered at the NATIVE 1280x720 resolution: two
fixed cameras and one head camera see a ChArUco board, a head tag rigidly mounted on the head camera (official AprilTag
bitmap, rotated mount, 3-axis lever arm) and an object tag. Source videos are written at 1280x720 and the C4 frames
(JPEGs + index.parquet + manifest.json) at 640x360, while rig.json gives the intrinsics at 1280x720 (the design doc's
example), so the stages must rescale K and detect on native frames. calib -> tags -> headpose -> world3d run through the
real stage code, with upstream stage states set as run.py would.

The helpers here are shared with tests/test_playground_world.py."""
import json
import shutil
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

cv2 = pytest.importorskip("cv2")

from duet.playground import geometry as G  # noqa: E402
from duet.playground import world as W  # noqa: E402
from duet.playground.episode import Episode, Stream  # noqa: E402

NW, NH = 1280, 720          # native (source video) size
FW, FH = 640, 360           # extracted frame size (proc_size 640)
K_TRUE = np.array([[600.0, 0, 639.5], [0, 600.0, 359.5], [0, 0, 1]])  # 1280x720, ~94 deg HFOV
SPEC = G.BoardSpec(7, 5, 0.08, 0.06); TAG_M = 0.08
BC = (SPEC.squares_x * SPEC.square_m / 2, SPEC.squares_y * SPEC.square_m / 2)  # board centre (world, Z = 0)


# ----------------------------------------------------------------------------- shared helpers

def rot(axis: str, deg: float) -> np.ndarray:
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return {"x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]), "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
            "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])}[axis]


def T_from(R, t) -> np.ndarray:
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = t; return T


def look_at(cam_pos, target, up=(0, 1, 0)) -> np.ndarray:
    """T_world_cam, OpenCV camera (+Z forward, +Y down) at cam_pos looking at target."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float); z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(up, float)); x /= np.linalg.norm(x); y = np.cross(z, x)
    return T_from(np.stack([x, y, z], 1), cam_pos)


def rot_err_deg(Ra, Rb) -> float:
    """Angle between two rotations, from the chord |Ra - Rb|_F = 2 sqrt(2) sin(angle / 2) (well conditioned near 0)."""
    return float(np.degrees(2 * np.arcsin(min(1.0, np.linalg.norm(np.asarray(Ra, float) - np.asarray(Rb, float)) / np.sqrt(8)))))


def render_plane(img_plane, px_per_m, T_cam_plane, K, size=(NW, NH), plane_origin=(0.0, 0.0)) -> np.ndarray:
    """Warp a planar image into a camera. Plane frame = board convention: origin at the image's top-left corner, X along
    columns, Y along rows, Z into the plane; plane_origin shifts the image corner (metres). uint8, white background."""
    h, w = img_plane.shape[:2]
    corners = np.array([[0, 0, 0], [w, 0, 0], [w, h, 0], [0, h, 0]], float) / px_per_m + np.array([*plane_origin, 0.0])
    Xc = corners @ T_cam_plane[:3, :3].T + T_cam_plane[:3, 3]; dst = ((Xc / Xc[:, 2:]) @ K.T)[:, :2].astype(np.float32)
    proj_w = max(np.linalg.norm(dst[1] - dst[0]), np.linalg.norm(dst[2] - dst[3])); factor = min(1.0, 1.5 * proj_w / w)
    if factor < 1.0:
        img_plane = cv2.resize(img_plane, (max(8, int(w * factor)), max(8, int(h * factor))), interpolation=cv2.INTER_AREA); h, w = img_plane.shape[:2]
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32) - 0.5  # pixel-centre convention in both images: outer corner = -0.5
    return cv2.warpPerspective(img_plane, cv2.getPerspectiveTransform(src, dst), size, flags=cv2.INTER_LINEAR, borderValue=255)


def board_image(spec=SPEC, ppm=2500):
    return spec.board().generateImage((int(spec.squares_x * spec.square_m * ppm), int(spec.squares_y * spec.square_m * ppm))), ppm


def tag_canvas(tag_id: int, tag_m: float = TAG_M, px: int = 400, border: int = 60):
    """OFFICIAL tag36h11 bitmap (cv2.aruco's DICT_APRILTAG_36h11 image rotated 180 deg), black square = tag_m, on a white
    margin. Returns (canvas, px per metre, plane_origin placing the tag centre at the plane-frame origin)."""
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    tag = np.rot90(cv2.aruco.generateImageMarker(d, tag_id, px), 2)
    c = np.full((px + 2 * border, px + 2 * border), 255, np.uint8); c[border:-border, border:-border] = tag
    ppm = px / tag_m; o = -(tag_m / 2 + border / ppm)
    return c, ppm, (o, o)


def write_video(path, frames: list[np.ndarray], fps: float) -> None:
    """Lossless H.264 mp4 (PyAV) of grayscale frames at a constant fps, pts = i / fps from 0."""
    av = pytest.importorskip("av")
    h, w = frames[0].shape; tb = Fraction(1, int(round(fps)))
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=Fraction(fps).limit_denominator(1000)); st.width, st.height, st.pix_fmt = w, h, "yuv420p"
        st.options = {"qp": "0", "preset": "ultrafast"}
        for i, g in enumerate(frames):
            f = av.VideoFrame.from_ndarray(np.ascontiguousarray(np.repeat(g[..., None], 3, -1)), format="rgb24"); f.pts, f.time_base = i, tb
            for p in st.encode(f):
                c.mux(p)
        for p in st.encode():
            c.mux(p)


def new_episode(root, streams: list[Stream], n: int, fps: float = 10.0) -> Episode:
    """Episode with streams[0] as the reference and the others manually aligned at offset 0 (usable), window n/fps."""
    d = root / "ep"; (d / "streams").mkdir(parents=True)
    for i, s in enumerate(streams):
        s.offset_status = "reference" if i == 0 else "manual"
    ep = Episode(name="ep", root=str(d), streams=streams, reference=streams[0].name, proc_fps=fps, common_start_s=0.0, common_end_s=n / fps)
    ep.save()
    return ep


def write_frames(ep: Episode, name: str, native: list[np.ndarray] | None, n: int | None = None, src_fps: float | None = None,
                 size=(FW, FH), src_frames=None, t_src=None) -> None:
    """C4 frames for one stream: derived/frames/<name>/NNNNNN.jpg (native images downscaled to size, or grey when
    native is None), index.parquet (k, file, t_ref_s, t_src_s, src_frame) and manifest.json. Default: source frame k at
    k / src_fps (src_fps = proc_fps)."""
    n = len(native) if native is not None else n; src_fps = src_fps or ep.proc_fps
    d = ep.derived / "frames" / name; d.mkdir(parents=True, exist_ok=True); files = []
    for k in range(n):
        img = cv2.resize(native[k], size, interpolation=cv2.INTER_AREA) if native is not None else np.full(size[::-1], 128, np.uint8)
        f = f"{k + 1:06d}.jpg"; cv2.imwrite(str(d / f), img, [cv2.IMWRITE_JPEG_QUALITY, 95]); files.append(f)
    src = np.arange(n) if src_frames is None else np.asarray(src_frames)
    pd.DataFrame({"k": np.arange(n), "file": files, "t_ref_s": ep.common_start_s + np.arange(n) / ep.proc_fps,
                  "t_src_s": src / src_fps if t_src is None else np.asarray(t_src, float), "src_frame": src}).to_parquet(d / "index.parquet", index=False)
    mp = ep.derived / "frames" / "manifest.json"; m = json.loads(mp.read_text()) if mp.exists() else {"proc_fps": ep.proc_fps, "streams": {}}
    s = ep.stream(name); m["streams"][name] = {"width": size[0], "height": size[1], "scale": size[0] / (s.width or size[0])}; mp.write_text(json.dumps(m))


def set_done(ep: Episode, *stages: str) -> None:
    for st in stages:
        ep.set_status(st, "done", "synthetic")


def rig_json(ep: Episode, **extra) -> None:
    r = {"board": {"squares_x": SPEC.squares_x, "squares_y": SPEC.squares_y, "square_m": SPEC.square_m, "marker_m": SPEC.marker_m}, "tag_size_m": TAG_M}
    r.update(extra); (ep.dir / "rig.json").write_text(json.dumps(r))


# ----------------------------------------------------------------------------- the end-to-end test

# the head tag sits 7 cm up / 2 cm right / 3 cm behind the lens, tilted 25 deg and turned 30 deg: a full 3-axis mount
T_TAG_HEADCAM = T_from(rot("z", 30) @ rot("x", 25), [0.02, 0.07, 0.03])  # head camera pose in the tag frame


@pytest.mark.parametrize("frames", ["helper", "no_video", "extract_frames"])
def test_world_stages_on_synthetic_episode(tmp_path, frames):
    """frames="helper": C4 frames written by write_frames + native source videos; "no_video": the source videos are
    missing (dangling links), so detection falls back to the extracted frames; "extract_frames": the real pipeline path,
    episode.probe + perception.extract_frames on the synthetic videos."""
    pytest.importorskip("pupil_apriltags"); pytest.importorskip("av"); native_video = frames != "no_video"
    n, fps = 12, 10.0
    board, ppm_b = board_image()
    T_wc = {"exoA": look_at([BC[0] - 0.5, BC[1] - 0.3, -1.1], [*BC, 0]), "exoB": look_at([BC[0] + 0.6, BC[1] - 0.25, -1.0], [*BC, 0])}
    head = [look_at([BC[0] + 0.12 * np.cos(2 * np.pi * k / n), BC[1] + 0.08 * np.sin(2 * np.pi * k / n), -0.5], [BC[0], BC[1] + 0.05 * k / n, 0]) for k in range(n)]
    obj_c, obj_ppm, obj_o = tag_canvas(10); head_c, head_ppm, head_o = tag_canvas(1)
    T_world_obj = T_from(np.eye(3), [0.65, 0.15, 0.0])  # object tag printed upright, lying on the table next to the board
    streams = [Stream("head", "ego", "streams/head.mp4", person="leader", fps=fps, width=NW, height=NH),
               Stream("exoA", "exo", "streams/exoA.mp4", fps=fps, width=NW, height=NH), Stream("exoB", "exo", "streams/exoB.mp4", fps=fps, width=NW, height=NH)]
    ep = new_episode(tmp_path, streams, n, fps)
    imgs = {"head": [], "exoA": [], "exoB": []}
    for k in range(n):
        T_world_tag = head[k] @ G.inv(T_TAG_HEADCAM)
        for name in ("exoA", "exoB"):
            T_cw = G.inv(T_wc[name])
            imgs[name].append(np.minimum.reduce([render_plane(board, ppm_b, T_cw, K_TRUE), render_plane(obj_c, obj_ppm, T_cw @ T_world_obj, K_TRUE, plane_origin=obj_o),
                                                 render_plane(head_c, head_ppm, T_cw @ T_world_tag, K_TRUE, plane_origin=head_o)]))
        imgs["head"].append(render_plane(board, ppm_b, G.inv(head[k]), K_TRUE))
    for name, views in imgs.items():
        if native_video:
            write_video(ep.dir / "streams" / f"{name}.mp4", views, fps)
        if frames != "extract_frames":
            write_frames(ep, name, views)
    if frames == "extract_frames":
        if not shutil.which("ffprobe"):
            pytest.skip("ffprobe not installed")
        from duet.playground import perception
        from duet.playground.episode import probe
        probe(ep); perception.extract_frames(ep); assert ep.stage_ok("frames"), ep.status["frames"]
        assert perception.frame_size(ep, "exoA") == (FW, FH) and np.allclose(perception.frame_index(ep, "exoA")["t_src_s"], np.arange(n) / fps)
    native_K = G.Intrinsics(K_TRUE, np.zeros(5), NW, NH).to_json()
    rig_json(ep, head_tags={"leader": 1}, object_tags={"bowl": 10}, T_tag_headcam={"leader": T_TAG_HEADCAM.tolist()},
             intrinsics={name: native_K for name in ("exoA", "exoB", "head")})
    set_done(ep, "probe", "align", "frames")

    W.calib(ep); assert ep.stage_ok("calib"), ep.status["calib"]
    rep = W.calib_report(ep)
    for name in ("exoA", "exoB"):
        cam = rep["cameras"][name]; T = np.array(cam["T_world_cam"])
        assert cam["intrinsics"]["width"] == FW and abs(cam["intrinsics"]["K"][0][0] - 300.0) < 1e-6, cam["intrinsics"]  # rig K rescaled to the frames
        assert cam["detect_frames"] == ("native" if native_video else "extracted") and cam["detect_size"] == ([NW, NH] if native_video else [FW, FH])
        tol = 0.015 if native_video else 0.03
        assert np.linalg.norm(T[:3, 3] - T_wc[name][:3, 3]) < tol and rot_err_deg(T[:3, :3], T_wc[name][:3, :3]) < 1.0, (name, T, T_wc[name])
    assert rep["cameras"]["head"]["moving"] and rep["cameras"]["head"]["T_world_cam"] is None and rep["cameras"]["head"]["sightings"]

    set_done(ep, "body2d", "hands")
    W.tags(ep); assert ep.stage_ok("tags"), ep.status["tags"]
    obj = W.tag_world_poses(ep, 10); assert len(obj) >= n - 2
    assert np.linalg.norm(np.median([p[:3, 3] for p in obj.values()], axis=0) - T_world_obj[:3, 3]) < 0.02
    assert np.median([rot_err_deg(p[:3, :3], np.eye(3)) for p in obj.values()]) < 3.0  # upright tag on the board plane: R_world_tag = I

    W.headpose(ep); assert ep.stage_ok("headpose") and "head_tag" in ep.status["headpose"]["detail"], ep.status["headpose"]
    hp = np.load(ep.derived / "headpose" / "head.npz"); T = hp["T_world_cam"].astype(float)
    got = [k for k in range(n) if hp["valid"][k]]; assert len(got) >= n - 2 and str(hp["backend"]) == "head_tag"
    pos = [np.linalg.norm(T[k][:3, 3] - head[k][:3, 3]) for k in got]; ang = [rot_err_deg(T[k][:3, :3], head[k][:3, :3]) for k in got]
    assert np.median(pos) < (0.02 if native_video else 0.04) and np.median(ang) < (2.0 if native_video else 4.0), (pos, ang)  # full pose, not just position

    W.world3d(ep); assert ep.stage_ok("world3d"), ep.status["world3d"]
    w = np.load(ep.derived / "world3d" / "world3d.npz")
    assert np.isfinite(w["object_bowl"][:, 0]).sum() >= n - 2
    np.testing.assert_allclose(w["T_world_cam_head"], T.astype(np.float32), equal_nan=True)  # per-frame head poses for the viewer
    np.testing.assert_allclose(w["head_head"], T[:, :3, 3].astype(np.float32), equal_nan=True)
    assert "bodies" in ep.status["world3d"]["detail"] and w["bodies"].shape == (n, 2, 17, 3)
    assert "feat_min_wrist_dist_m" in w and not np.isfinite(w["feat_min_wrist_dist_m"]).any()  # always written, NaN without bodies
