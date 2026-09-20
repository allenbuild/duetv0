"""Tiny derived caches verify CLI gating without a GUI or any raw dataset."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

RECORDING = "00000000-0000-0000-0000-000000000001"
GRAPH = "00000000-0000-0000-0000-000000000002"


@pytest.fixture
def cli(monkeypatch, tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts" / "visualize_comind.py"
    spec = importlib.util.spec_from_file_location("visualize_comind_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "REPOSITORY_ROOT", tmp_path)
    return module


@pytest.fixture
def cache(tmp_path):
    processed = tmp_path / "data" / "processed" / "comind"
    directory = processed / RECORDING / "semantics"
    directory.mkdir(parents=True)
    reports = tmp_path / "outputs" / "comind_semantics"
    reports.mkdir(parents=True)
    manifest = {
        "recording_id": RECORDING,
        "shared_world": {
            "status": "VERIFIED",
            "graph_uid": GRAPH,
            "frame": f"comind/{RECORDING}/multislam/{GRAPH}",
        },
        "cross_device_time": {"verified": False, "reason": "TICSync evidence absent"},
        "participants": {},
    }
    for index, role in enumerate(("helper", "leader")):
        path = directory / f"{role}.npz"
        np.savez_compressed(
            path,
            graph_uid=np.array(GRAPH),
            trajectory_translation_m=np.array([[index, 0, 0], [index, 1, 0]], dtype=float),
            # Reading this unused member would fail allow_pickle=False. It must remain unread.
            hand_points_device_m=np.array([{"not": "read"}], dtype=object),
        )
        manifest["participants"][role] = {
            "source": {
                "graph_uid": GRAPH,
                "transform_invalid_count": 0,
                "unchanged_during_read": True,
                "path": str(tmp_path / "data" / "raw" / "must_not_be_read.csv"),
            },
            "cache": {
                "participant": role,
                "bytes": path.stat().st_size,
                "trajectory_sample_count": 2,
                "path": str(path),
            },
        }
    video = {
        "recording_id": RECORDING,
        "mappings": {"video_pts_to_device": {"status": "unresolved"}},
    }

    def write(*, clock=None):
        (directory / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        (reports / f"{RECORDING}.json").write_text(
            json.dumps(manifest if clock is None else clock), encoding="utf-8"
        )
        (reports / f"{RECORDING}_video_annotations.json").write_text(
            json.dumps(video), encoding="utf-8"
        )

    write()
    return processed, reports, directory, manifest, video, write


def arguments(cache):
    processed, reports, *_ = cache
    return [
        "--recording-id",
        RECORDING,
        "--diagnostic",
        "--processed-root",
        str(processed),
        "--reports-root",
        str(reports),
    ]


def test_default_synchronized_mode_fails_before_reading_any_file(cli, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("default synchronized mode must not read files or launch renderer")

    monkeypatch.setattr(cli, "load_diagnostic", forbidden)
    monkeypatch.setattr(cli, "save_spatial_diagnostic", forbidden)
    with pytest.raises(SystemExit) as error:
        cli.main(["--recording-id", RECORDING])
    assert error.value.code == 2
    message = capsys.readouterr().err
    assert "only one DEVICE_TIME value" in message
    assert "common video index" in message
    assert "--diagnostic" in message


def test_force_flag_does_not_exist(cli, capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--recording-id", RECORDING, "--force"])
    assert error.value.code == 2
    assert "unrecognized arguments: --force" in capsys.readouterr().err


def test_synchronized_mode_dispatches_to_verified_loader(cli, monkeypatch):
    calls = []

    def synchronized(args, recording_id):
        calls.append((args.synchronized, recording_id))
        return 0

    monkeypatch.setattr(cli, "_synchronized", synchronized)
    assert cli.main(["--recording-id", RECORDING, "--synchronized"]) == 0
    assert calls == [(True, RECORDING)]


def test_synchronized_and_diagnostic_are_mutually_exclusive(cli):
    with pytest.raises(SystemExit):
        cli.main(["--recording-id", RECORDING, "--synchronized", "--diagnostic"])


def test_diagnostic_renders_static_points_and_full_evidence_only(cli, cache, monkeypatch, tmp_path):
    calls = []

    def render(output, **kwargs):
        calls.append((output, kwargs))
        return output

    monkeypatch.setattr(cli, "save_spatial_diagnostic", render)
    monkeypatch.chdir(tmp_path)
    assert cli.main(arguments(cache)) == 0
    output, request = calls[0]
    assert (
        output == tmp_path / "outputs/comind_visualization" / f"{RECORDING}_spatial_diagnostic.rrd"
    )
    assert set(request["trajectories"]) == {"helper", "leader"}
    assert request["trajectories"]["helper"].shape == (2, 3)
    assert request["world_frame"].name == f"comind/{RECORDING}/multislam/{GRAPH}"
    assert "hands at a common instant" in request["status"]["disabled_layers"]
    assert "active handover" in request["status"]["disabled_layers"]
    assert not request["status"]["clock_and_motion_evidence"]["cross_device_time"]["verified"]
    assert request["provenance"].parents
    assert "raw sources were not reopened" in request["status"]["cache_provenance_scope"]
    assert not (tmp_path / "data/raw/must_not_be_read.csv").exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("frame", "helper/mps_world/standard"),
        ("status", "INFERRED"),
    ],
)
def test_rejects_unverified_or_wrong_shared_frame(cli, cache, field, value):
    processed, reports, _, manifest, _, write = cache
    manifest["shared_world"][field] = value
    write()
    with pytest.raises(ValueError, match="shared-world"):
        cli.load_diagnostic(processed, reports, RECORDING)


@pytest.mark.parametrize(
    "part,key,value",
    [
        ("source", "graph_uid", "another-graph"),
        ("source", "transform_invalid_count", 1),
        ("cache", "participant", "leader"),
        ("cache", "bytes", 0),
        ("cache", "trajectory_sample_count", 3),
    ],
)
def test_rejects_inconsistent_participant_cache_metadata(cli, cache, part, key, value):
    processed, reports, _, manifest, _, write = cache
    manifest["participants"]["helper"][part][key] = value
    write()
    with pytest.raises(ValueError, match="helper"):
        cli.load_diagnostic(processed, reports, RECORDING)


def test_rejects_npz_graph_different_from_manifest(cli, cache):
    processed, reports, directory, manifest, _, write = cache
    path = directory / "helper.npz"
    np.savez_compressed(
        path, graph_uid=np.array("wrong graph"), trajectory_translation_m=np.zeros((2, 3))
    )
    manifest["participants"]["helper"]["cache"]["bytes"] = path.stat().st_size
    write()
    with pytest.raises(ValueError, match="cached graph differs"):
        cli.load_diagnostic(processed, reports, RECORDING)


@pytest.mark.parametrize(
    "points",
    [np.zeros((2, 4)), np.full((2, 3), np.nan), np.ones((2, 3), dtype=complex), np.empty((0, 3))],
)
def test_rejects_invalid_cached_positions(cli, cache, points):
    processed, reports, directory, manifest, _, write = cache
    path = directory / "helper.npz"
    np.savez_compressed(path, graph_uid=np.array(GRAPH), trajectory_translation_m=points)
    manifest["participants"]["helper"]["cache"]["bytes"] = path.stat().st_size
    write()
    with pytest.raises(ValueError, match="helper"):
        cli.load_diagnostic(processed, reports, RECORDING)


def test_rejects_report_identity_and_provenance_mismatch(cli, cache):
    processed, reports, _, manifest, _, write = cache
    clock = deepcopy(manifest)
    clock["participants"]["helper"]["source"]["path"] = "different audit"
    write(clock=clock)
    with pytest.raises(ValueError, match="provenance disagree"):
        cli.load_diagnostic(processed, reports, RECORDING)


def test_raw_output_and_symlink_are_rejected_before_loading(cli, cache, tmp_path, monkeypatch):
    raw = tmp_path / "data/raw"
    raw.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(raw, target_is_directory=True)
    monkeypatch.setattr(cli, "load_diagnostic", lambda *args: pytest.fail("read before raw guard"))
    for parent in (raw, alias):
        with pytest.raises(SystemExit) as error:
            cli.main(arguments(cache) + ["--output", str(parent / "bad.rrd")])
        assert error.value.code == 2
        assert not (parent / "bad.rrd").exists()


def test_raw_input_symlink_rejected_without_read(cli, cache, tmp_path):
    processed, reports, directory, *_ = cache
    raw = tmp_path / "data/raw"
    raw.mkdir()
    (directory / "helper.npz").unlink()
    (directory / "helper.npz").symlink_to(raw / "not_present.npz")
    with pytest.raises(ValueError, match="data/raw"):
        cli.load_diagnostic(processed, reports, RECORDING)


def test_missing_report_is_clear_cli_error_without_rendering(cli, cache, monkeypatch, capsys):
    _, reports, *_ = cache
    (reports / f"{RECORDING}_video_annotations.json").unlink()
    monkeypatch.setattr(cli, "save_spatial_diagnostic", lambda *a, **k: pytest.fail("rendered"))
    with pytest.raises(SystemExit) as error:
        cli.main(arguments(cache))
    assert error.value.code == 2
    assert "video_annotations.json" in capsys.readouterr().err


def test_recording_path_traversal_and_wrong_output_extension_rejected(cli, cache):
    with pytest.raises(SystemExit):
        cli.main(["--recording-id", "../escape", "--diagnostic"])
    with pytest.raises(SystemExit):
        cli.main(arguments(cache) + ["--output", "bad.json"])
