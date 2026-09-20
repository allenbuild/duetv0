"""Offline selection and mock HTTP tests never contact a dataset endpoint."""

import hashlib
import importlib.util
import io
import json
from pathlib import Path

import pytest

from duet.adapters.comind.minimal_download import (
    HANDOVER_ANNOTATION,
    MAPPING_PATH,
    SelectedFile,
    _rename_no_replace,
    download_selected_file,
    parse_manifest,
    safe_destination,
    select_recording_files,
    verify_existing,
)

RECORDING = "2fc0aa53-9070-4c86-81c2-41450253c74d"
PAYLOAD = b"synthetic asset\n"
DIGEST = hashlib.sha256(PAYLOAD).hexdigest()
REQUIRED = [
    "mp4s/helper_trimmed_sync.mp4",
    "mp4s/leader_trimmed_sync.mp4",
    "mps_helper_trimmed_vrs/hand_tracking/hand_tracking_results.csv",
    "mps_leader_trimmed_vrs/hand_tracking/hand_tracking_results.csv",
    "multislam_output/0/slam/closed_loop_trajectory.csv",
    "multislam_output/1/slam/closed_loop_trajectory.csv",
    MAPPING_PATH,
]


def manifest_bytes(paths=REQUIRED, recording_id=RECORDING):
    return json.dumps(
        {
            "recording_id": recording_id,
            "hash_algorithm": "sha256",
            "files": [{"path": path, "size": len(PAYLOAD), "hash": DIGEST} for path in paths],
        }
    ).encode()


def manifest(paths=REQUIRED):
    return parse_manifest(
        manifest_bytes(paths),
        recording_id=RECORDING,
        is_excluded=lambda path: path.endswith(".vrs") and not path.startswith("trimmed_vrs/"),
    )


def selected_entry():
    return SelectedFile(
        RECORDING,
        REQUIRED[0],
        f"recordings/{RECORDING}/{REQUIRED[0]}",
        len(PAYLOAD),
        DIGEST,
        "sha256",
        "synthetic test",
    )


class Response(io.BytesIO):
    def __init__(self, data=PAYLOAD, *, status=200, headers=None):
        super().__init__(data)
        self.status = status
        self.headers = {"Content-Length": str(len(data))} if headers is None else headers


def test_selects_only_seven_exact_inputs_preserving_official_layout():
    extra = [
        "trimmed_vrs/helper_trimmed.vrs",
        "helper.vrs",
        "mp4s/helper_sync.mp4",
        "mp4s/gopro_front_sync.mp4",
        "mps_helper_trimmed_vrs/slam/online_calibration.jsonl",
        "multislam_output/0/slam/summary.json",
        "multislam_output/summary.json.zip",
        "scan/blk_scan_aria_aligned.ply",
    ]
    parsed = manifest(REQUIRED + extra)
    selected = select_recording_files(parsed)
    assert [entry.path for entry in selected] == REQUIRED
    assert all(entry.destination == f"recordings/{RECORDING}/{entry.path}" for entry in selected)
    assert parsed.total_size_bytes == (len(REQUIRED) + len(extra) - 1) * len(PAYLOAD)


def test_extra_trajectory_directories_require_verified_mapping():
    parsed = manifest(REQUIRED + ["multislam_output/2/slam/closed_loop_trajectory.csv"])
    with pytest.raises(ValueError, match="rather than guessing"):
        select_recording_files(parsed)
    selected = select_recording_files(parsed, verified_mapping={"helper": "2", "leader": "0"})
    assert "multislam_output/2/slam/closed_loop_trajectory.csv" in [x.path for x in selected]
    assert "multislam_output/1/slam/closed_loop_trajectory.csv" not in [x.path for x in selected]


def test_missing_required_individual_asset_is_not_replaced_by_part_or_archive():
    parsed = manifest(REQUIRED[1:] + ["mp4s.zip"])
    with pytest.raises(ValueError, match="required individual file"):
        select_recording_files(parsed)


@pytest.mark.parametrize("path", ["../x", "/x", "x//y", "x/./y", "x\\y", "x\n", ""])
def test_manifest_rejects_unsafe_paths(path):
    with pytest.raises(ValueError, match="unsafe manifest path"):
        manifest([path])


def test_manifest_rejects_wrong_recording_duplicate_path_missing_hash_and_bad_size():
    value = json.loads(manifest_bytes())
    value["recording_id"] = "different"
    with pytest.raises(ValueError, match="recording_id"):
        parse_manifest(
            json.dumps(value).encode(), recording_id=RECORDING, is_excluded=lambda p: False
        )
    with pytest.raises(ValueError, match="duplicate"):
        manifest(REQUIRED + [REQUIRED[0]])
    for key, replacement in [("hash", None), ("size", True), ("size", -1), ("size", "15")]:
        value = json.loads(manifest_bytes())
        value["files"][0][key] = replacement
        with pytest.raises(ValueError):
            parse_manifest(
                json.dumps(value).encode(), recording_id=RECORDING, is_excluded=lambda p: False
            )


def test_verified_existing_file_skips_http_and_is_unchanged(tmp_path):
    entry = selected_entry()
    destination = tmp_path / entry.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(PAYLOAD)
    before = destination.stat()

    def forbidden(*args, **kwargs):
        pytest.fail("verified raw data must not trigger HTTP")

    assert (
        download_selected_file(
            entry, target=tmp_path, url="https://invalid.example/asset", timeout=1, opener=forbidden
        )
        == "verified_existing"
    )
    assert destination.stat().st_mtime_ns == before.st_mtime_ns
    assert destination.read_bytes() == PAYLOAD


@pytest.mark.parametrize("existing", [b"wrong", b"x" * len(PAYLOAD)])
def test_existing_unverified_file_is_never_overwritten(tmp_path, existing):
    entry = selected_entry()
    destination = tmp_path / entry.destination
    destination.parent.mkdir(parents=True)
    destination.write_bytes(existing)
    with pytest.raises(ValueError, match="will not overwrite"):
        download_selected_file(
            entry,
            target=tmp_path,
            url="https://invalid.example/asset",
            timeout=1,
            opener=lambda *a, **k: pytest.fail("preflight must fail before HTTP"),
        )
    assert destination.read_bytes() == existing


def test_successful_transfer_has_partial_only_until_hash_and_size_verified(tmp_path):
    entry = selected_entry()
    destination = tmp_path / entry.destination
    partial = destination.with_name(destination.name + ".part")

    def open_mock(request, timeout):
        assert not destination.exists()
        assert partial.is_file()
        assert request.get_header("Accept-encoding") == "identity"
        return Response()

    assert (
        download_selected_file(
            entry, target=tmp_path, url="https://invalid.example/asset", timeout=1, opener=open_mock
        )
        == "downloaded_verified"
    )
    assert verify_existing(destination, entry)
    assert not partial.exists()


@pytest.mark.parametrize(
    "response,match",
    [
        (lambda: Response(status=206), "expected HTTP 200"),
        (lambda: Response(headers={"Content-Encoding": "gzip"}), "encoded"),
        (lambda: Response(headers={"Content-Length": "9999"}), "Content-Length"),
        (lambda: Response(data=b"short", headers={}), "truncated"),
        (lambda: Response(data=PAYLOAD + b"extra", headers={}), "exceeded"),
        (lambda: Response(data=b"x" * len(PAYLOAD)), "hash"),
    ],
)
def test_failed_response_never_publishes_raw_file(tmp_path, response, match):
    entry = selected_entry()
    with pytest.raises(ValueError, match=match):
        download_selected_file(
            entry,
            target=tmp_path,
            url="https://invalid.example/asset",
            timeout=1,
            opener=lambda *args, **kwargs: response(),
        )
    assert not (tmp_path / entry.destination).exists()
    assert not list(tmp_path.rglob("*.part"))


def test_existing_partial_is_not_overwritten_and_http_is_not_started(tmp_path):
    entry = selected_entry()
    partial = tmp_path / (entry.destination + ".part")
    partial.parent.mkdir(parents=True)
    partial.write_bytes(b"prior partial")
    with pytest.raises(FileExistsError):
        download_selected_file(
            entry,
            target=tmp_path,
            url="https://invalid.example/asset",
            timeout=1,
            opener=lambda *args, **kwargs: pytest.fail("must not contact HTTP"),
        )
    assert partial.read_bytes() == b"prior partial"


def test_atomic_rename_will_not_clobber_destination_that_appears_concurrently(tmp_path):
    source = tmp_path / "asset.part"
    destination = tmp_path / "asset"
    source.write_bytes(PAYLOAD)
    destination.write_bytes(b"someone else's verified original")
    with pytest.raises(FileExistsError):
        _rename_no_replace(source, destination)
    assert destination.read_bytes() == b"someone else's verified original"
    assert source.read_bytes() == PAYLOAD


def test_symlinked_root_parent_final_and_partial_are_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        safe_destination(alias, "file")
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        safe_destination(root, "nested/file")
    (root / "file").symlink_to(outside / "absent")
    with pytest.raises(ValueError, match="symlink"):
        safe_destination(root, "file")
    entry = selected_entry()
    partial = root / (entry.destination + ".part")
    partial.parent.mkdir(parents=True)
    partial.symlink_to(outside / "new")
    with pytest.raises(ValueError, match="symlink"):
        download_selected_file(entry, target=root, url="https://invalid.example/asset", timeout=1)
    assert not (outside / "new").exists()


@pytest.fixture
def cli(monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "minimal_cli_test", scripts / "comind_download_minimal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_offline_dry_run_uses_no_network_or_raw_writes(tmp_path, monkeypatch, cli, capsys):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_bytes(manifest_bytes())
    annotation_path = tmp_path / "annotations_manifest.json"
    annotation_path.write_bytes(manifest_bytes([HANDOVER_ANNOTATION], "annotations"))
    target = tmp_path / "raw"
    report = tmp_path / "processed/report.json"

    def forbidden(*args, **kwargs):
        pytest.fail("offline dry-run must make no network or raw download calls")

    monkeypatch.setattr(cli, "fetch_manifest", forbidden)
    monkeypatch.setattr(cli, "download_selected_file", forbidden)
    assert (
        cli.main(
            [
                RECORDING,
                "--dry-run",
                "--manifest",
                str(manifest_path),
                "--annotations-manifest",
                str(annotation_path),
                "--target",
                str(target),
                "--report",
                str(report),
                "--remote-vrs-estimate-bytes",
                "123",
            ]
        )
        == 0
    )
    assert not target.exists()
    saved = json.loads(report.read_text())
    assert saved["expected_total_transfer_bytes"] == 8 * len(PAYLOAD) + 123
    assert len(saved["selected_files"]) == 8
    assert "No dataset asset requests made" in capsys.readouterr().out


def test_default_mode_is_dry_run_and_download_is_mutually_exclusive(cli):
    assert not cli.parse_args([RECORDING]).download
    with pytest.raises(SystemExit):
        cli.parse_args([RECORDING, "--download", "--dry-run"])


def test_cli_rejects_mismatched_mapping_before_other_asset_transfers(tmp_path, monkeypatch, cli):
    parsed = manifest()
    entry = next(entry for entry in select_recording_files(parsed) if entry.path == MAPPING_PATH)
    path = tmp_path / entry.destination
    path.parent.mkdir(parents=True)
    path.write_bytes(b"bad mapping")
    with pytest.raises(ValueError, match="will not overwrite"):
        cli.verified_local_mapping(parsed, tmp_path)
