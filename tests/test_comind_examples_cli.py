"""Batch selection and source-label joins using synthetic reports only."""

import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from duet.visualization.demo_encoding import encode_video

RECORDING = "00000000-0000-0000-0000-000000000001"
OTHER_RECORDING = "00000000-0000-0000-0000-000000000002"


@pytest.fixture
def cli():
    path = Path(__file__).resolve().parents[1] / "scripts" / "export_comind_v0_examples.py"
    spec = importlib.util.spec_from_file_location("comind_examples_cli_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_row(annotation_id="002131", start=2131, end=2366, category="carrot peel"):
    return {
        "recording_id": RECORDING,
        "annotation_id": annotation_id,
        "numeric_id": None,
        "start_frame": start,
        "end_frame": end,
        "clip_start_frame": start - 120,
        "clip_end_frame": end + 120,
        "object_category": category,
        "video_path": f"derived/handover_{annotation_id}.mp4",
    }


def source_report(row, *, numeric_id=72):
    return {
        "recording_id": RECORDING,
        "annotation_observations": {
            "segments": [
                {
                    "source_key": row["annotation_id"],
                    "numeric_id": numeric_id,
                    "skip": False,
                    "start_frame": row["start_frame"],
                    "end_frame": row["end_frame"],
                    "object_category_level_1": row["object_category"],
                }
            ]
        },
    }


@pytest.mark.parametrize("numeric_id", [0, 72, "0009"])
def test_numeric_id_preserves_original_source_value_distinct_from_segment_key(cli, numeric_id):
    row = metric_row()
    source = source_report(row, numeric_id=numeric_id)
    before = deepcopy(source)
    cli.bind_numeric_ids([row], source, RECORDING)
    assert row["numeric_id"] == numeric_id
    assert type(row["numeric_id"]) is type(numeric_id)
    assert row["annotation_id"] == "002131"
    assert row["start_frame"] == 2131
    assert source == before


def test_numeric_id_join_uses_source_key_not_source_order(cli):
    first = metric_row()
    second = metric_row("018240", 18240, 18252, "bowl")
    source = source_report(first, numeric_id=97)
    source["annotation_observations"]["segments"].insert(
        0, source_report(second, numeric_id=4)["annotation_observations"]["segments"][0]
    )
    cli.bind_numeric_ids([first, second], source, RECORDING)
    assert [row["numeric_id"] for row in (first, second)] == [97, 4]


@pytest.mark.parametrize("mode", ["missing", "duplicate", "skipped", "zero_skip", "missing_skip"])
def test_numeric_id_requires_one_literal_not_skipped_source_match(cli, mode):
    row = metric_row()
    source = source_report(row)
    segments = source["annotation_observations"]["segments"]
    if mode == "missing":
        segments[0]["source_key"] = "002132"
    elif mode == "duplicate":
        segments.append(deepcopy(segments[0]))
    elif mode == "skipped":
        segments[0]["skip"] = True
    elif mode == "zero_skip":
        segments[0]["skip"] = 0
    else:
        del segments[0]["skip"]
    with pytest.raises(ValueError, match="exactly one usable"):
        cli.bind_numeric_ids([row], source, RECORDING)
    assert row["numeric_id"] is None


@pytest.mark.parametrize("field", ["start_frame", "end_frame"])
def test_numeric_id_rejects_mismatched_annotation_bounds(cli, field):
    row = metric_row()
    source = source_report(row)
    source["annotation_observations"]["segments"][0][field] += 1
    with pytest.raises(ValueError, match="boundaries"):
        cli.bind_numeric_ids([row], source, RECORDING)


def test_numeric_id_rejects_mismatched_category(cli):
    row = metric_row()
    source = source_report(row)
    source["annotation_observations"]["segments"][0]["object_category_level_1"] = "bowl"
    with pytest.raises(ValueError, match="category"):
        cli.bind_numeric_ids([row], source, RECORDING)


def test_numeric_id_rejects_missing_numeric_source_field(cli):
    row = metric_row()
    source = source_report(row)
    del source["annotation_observations"]["segments"][0]["numeric_id"]
    with pytest.raises((KeyError, ValueError), match="numeric_id"):
        cli.bind_numeric_ids([row], source, RECORDING)
    assert row["numeric_id"] is None


def test_numeric_id_rejects_source_report_from_different_recording(cli):
    row = metric_row()
    source = source_report(row)
    source["recording_id"] = OTHER_RECORDING
    with pytest.raises(ValueError, match="recording"):
        cli.bind_numeric_ids([row], source, RECORDING)


def test_numeric_id_rejects_metric_row_from_different_recording(cli):
    row = metric_row()
    source = source_report(row)
    row["recording_id"] = OTHER_RECORDING
    with pytest.raises(ValueError, match="recording"):
        cli.bind_numeric_ids([row], source, RECORDING)


def test_annotation_source_fallback_reads_only_existing_annotations(tmp_path, cli):
    row = metric_row()
    segment = source_report(row, numeric_id="0072")["annotation_observations"]["segments"][0]
    source = tmp_path / "raw/annotations/dataset_handover_consolidated.json"
    source.parent.mkdir(parents=True)
    source.write_text(
        json.dumps(
            {
                "data": {
                    RECORDING: {
                        row["annotation_id"]: segment,
                        "skipped": {**segment, "skip": True},
                    }
                }
            }
        )
    )
    before = source.read_bytes()
    absent_report = tmp_path / "derived/absent.json"
    report = cli.load_annotation_source_report(
        absent_report, dataset_root=tmp_path / "raw", recording_id=RECORDING
    )
    cli.bind_numeric_ids([row], report, RECORDING)
    assert row["numeric_id"] == "0072"
    assert len(report["annotation_observations"]["segments"]) == 1
    assert "no mapping verification" in report["inspection_scope"]
    assert not absent_report.exists()
    assert source.read_bytes() == before


def test_annotation_source_reuses_richer_report_without_raw_access(tmp_path, cli):
    report = source_report(metric_row())
    report["existing_evidence"] = "preserved"
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    assert (
        cli.load_annotation_source_report(
            path, dataset_root=tmp_path / "does-not-exist", recording_id=RECORDING
        )
        == report
    )


def all_rows():
    return [
        metric_row("002131", 2131, 2366, "carrot peel"),
        metric_row("015185", 15185, 15199, "paper plate"),
        metric_row("015204", 15204, 15212, "carrot"),
        metric_row("018240", 18240, 18252, "bowl"),
        metric_row("018260", 18260, 18286, "peeler"),
        metric_row("018941", 18941, 18994, "knife"),
    ]


@pytest.mark.parametrize("best", [False, True])
def test_compilation_segment_indices_are_local_ordinals_not_absolute_frames(cli, best):
    rows = all_rows()
    before = deepcopy(rows)
    selected = cli._segments(rows, best=best)
    for row, segment in zip(rows, selected, strict=True):
        count = min(240, max(225, row["end_frame"] - row["start_frame"] + 1)) if best else 150
        absolute = cli.choose_segment(
            row["clip_start_frame"],
            row["clip_end_frame"],
            row["start_frame"],
            row["end_frame"],
            count,
        )
        assert (segment["start_ordinal"], segment["end_ordinal"]) == (
            absolute[0] - row["clip_start_frame"],
            absolute[1] - row["clip_start_frame"],
        )
        assert 0 <= segment["start_ordinal"] <= segment["end_ordinal"]
        assert segment["end_ordinal"] <= row["clip_end_frame"] - row["clip_start_frame"]
        assert segment["path"] == row["video_path"]
        assert segment["annotation_id"] == row["annotation_id"]
        assert segment["object_category"] == row["object_category"]
    assert rows == before


def test_all_six_montage_includes_every_example_in_requested_order_and_exactly_900_frames(cli):
    rows = all_rows()
    selected = cli._segments(rows, best=False)
    counts = [segment["end_ordinal"] - segment["start_ordinal"] + 1 for segment in selected]
    assert counts == [150] * 6
    assert sum(counts) == 900
    assert sum(counts) / cli.FPS == 30
    assert [segment["annotation_id"] for segment in selected] == [
        row["annotation_id"] for row in rows
    ]
    assert "018941" in [segment["annotation_id"] for segment in selected]


def test_best_of_keeps_full_236_frame_carrot_event_and_other_segments_at_225(cli):
    rows = [all_rows()[0], all_rows()[3], all_rows()[1]]
    selected = cli._segments(rows, best=True)
    counts = [segment["end_ordinal"] - segment["start_ordinal"] + 1 for segment in selected]
    assert counts == [236, 225, 225]
    assert all(count <= 240 for count in counts)
    assert counts[0] >= rows[0]["end_frame"] - rows[0]["start_frame"] + 1
    assert selected[0]["start_ordinal"] + rows[0]["clip_start_frame"] <= rows[0]["start_frame"]
    assert selected[0]["end_ordinal"] + rows[0]["clip_start_frame"] >= rows[0]["end_frame"]
    assert sum(counts) == 686
    assert sum(counts) / cli.FPS == pytest.approx(22.8666666667)


def test_best_excerpt_is_capped_at_240_for_a_longer_future_event(cli):
    row = metric_row("future", 1000, 1300, "source category")
    segment = cli._segments([row], best=True)[0]
    assert segment["end_ordinal"] - segment["start_ordinal"] + 1 == 240


@pytest.mark.parametrize("best, count", [(False, 900), (True, 686)])
def test_cli_segment_output_integrates_with_streaming_compilation(tmp_path, cli, best, count):
    rows = all_rows() if not best else [all_rows()[0], all_rows()[3], all_rows()[1]]
    source = tmp_path / "synthetic_rendered_clip.mp4"
    pixels = np.full((24, 32, 3), [30, 60, 90], np.uint8)
    encode_video(source, (pixels for _ in range(476)), size=(32, 24), fps=30, expected_frames=476)
    for row in rows:
        row["video_path"] = str(source)
    result = cli.export_compilation(
        tmp_path / "batch.mp4", cli._segments(rows, best=best), fps=cli.FPS
    )
    assert result["decoded_frame_count"] == count
    assert result["duration_seconds"] == count / 30
    assert result["width"] == 32 and result["height"] == 24
    assert [item["annotation_id"] for item in result["segments"]] == [
        row["annotation_id"] for row in rows
    ]


def fake_maps(count=1000):
    from types import SimpleNamespace

    return {
        role: SimpleNamespace(
            recording_id=RECORDING,
            participant=role,
            frame_count=count,
            status=np.full(count, "VERIFIED", dtype="U10"),
        )
        for role in ("helper", "leader")
    }


def planned_source():
    first = metric_row("000200", 200, 210, "bowl")
    second = metric_row("000600", 600, 610, "spoon")
    source = source_report(first, numeric_id="7")
    source["annotation_observations"]["segments"].extend(
        source_report(second, numeric_id="0008")["annotation_observations"]["segments"]
    )
    return source


def test_independent_plans_keep_four_seconds_and_source_ids(cli):
    rows = cli.plan_annotations(planned_source(), fake_maps(), RECORDING)
    assert [row["context_frames_each_side"] for row in rows] == [120, 120]
    assert rows[0]["clip_start_frame"] == 80
    assert rows[0]["clip_end_frame"] == 330
    assert rows[0]["numeric_id"] == "7"
    assert rows[0]["annotation_id"] == "000200"
    assert rows[1]["numeric_id"] == "0008"


def test_only_three_second_fallback_when_outer_context_has_gap(cli):
    maps = fake_maps()
    maps["helper"].status[85] = "UNRESOLVED"
    rows = cli.plan_annotations(planned_source(), maps, RECORDING)
    assert rows[0]["context_frames_each_side"] == 90
    assert (rows[0]["clip_start_frame"], rows[0]["clip_end_frame"]) == (110, 300)
    assert rows[0]["actual_context_before_frames"] == 90
    assert rows[1]["context_frames_each_side"] == 120
    assert rows[0]["mapping_rejections"][0]["context_frames"] == 120


@pytest.mark.parametrize("status", ["UNRESOLVED", "INFERRED"])
def test_event_gap_rejects_only_that_annotation_without_fallback(cli, status):
    maps = fake_maps()
    maps["leader"].status[205] = status
    rows = cli.plan_annotations(planned_source(), maps, RECORDING)
    assert rows[0]["export_status"] == "rejected"
    assert rows[0]["rejection_stage"] == "event_mapping"
    assert rows[0]["context_frames_each_side"] is None
    assert len(rows[0]["mapping_rejections"]) == 1
    assert rows[1]["export_status"] == "pending"


def test_inner_context_gap_records_exact_status_and_reason(cli):
    maps = fake_maps()
    maps["helper"].status[150] = "UNRESOLVED"
    reasons = {"helper": np.full(1000, "", dtype="U32")}
    reasons["helper"][150] = "ambiguous_image_match"
    rows = cli.plan_annotations(planned_source(), maps, RECORDING, mapping_reasons=reasons)
    assert rows[0]["export_status"] == "rejected"
    assert rows[0]["rejection_stage"] == "context_mapping"
    assert [attempt["context_frames"] for attempt in rows[0]["mapping_rejections"]] == [120, 90]
    assert rows[0]["mapping_rejections"][0]["frames"] == [
        {
            "participant": "helper",
            "frame_index": 150,
            "status": "UNRESOLVED",
            "mapping_reason": "ambiguous_image_match",
        }
    ]
    assert rows[1]["export_status"] == "pending"


@pytest.mark.parametrize(
    "field,value",
    [("recording_id", OTHER_RECORDING), ("participant", "helper"), ("frame_count", 999)],
)
def test_planning_rejects_wrong_map_identity(cli, field, value):
    maps = fake_maps()
    setattr(maps["leader"], field, value)
    with pytest.raises(ValueError, match="identities or counts"):
        cli.plan_annotations(planned_source(), maps, RECORDING)


@pytest.fixture
def gated_batch(tmp_path, cli, monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(cli, "ROOT", tmp_path)
    source = planned_source()
    source_path = tmp_path / f"outputs/comind_semantics/{RECORDING}_video_annotations.json"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(json.dumps(source))
    projection_path = tmp_path / f"outputs/comind_v0/{RECORDING}_projection_qc.json"
    projection_path.parent.mkdir(parents=True)
    projection_path.write_text(json.dumps({"validation_report_sha256": "fixture-hash"}))
    maps = fake_maps()
    directory = tmp_path / f"data/processed/comind/{RECORDING}"
    for role in maps:
        path = directory / "frame_maps" / f"{role}_frame_map.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mapping_reason=np.full(1000, "fixture reason"))
        calibration = directory / "vrs" / f"{role}_export_rgb_calibration.json"
        calibration.parent.mkdir(parents=True, exist_ok=True)
        calibration.write_text(
            json.dumps({"recording_id": RECORDING, "participant": role, "status": "VERIFIED"})
        )
    monkeypatch.setattr(cli, "load_frame_map", lambda path: maps[path.name.split("_")[0]])
    monkeypatch.setattr(cli, "_digest", lambda path: "fixture-hash")
    loaded, rendered, compiled = [], [], []
    selections = {
        item["source_key"]: item for item in source["annotation_observations"]["segments"]
    }

    def select(annotation_id, *, context_frames):
        item = selections[annotation_id]
        return SimpleNamespace(
            selection={"annotation_id": annotation_id},
            start_index=item["start_frame"] - context_frames,
            end_index=item["end_frame"] + context_frames,
        )

    base = SimpleNamespace(
        with_annotation=select,
        bindings=SimpleNamespace(
            camera_extrinsics={role: object() for role in maps},
            image_geometry={role: object() for role in maps},
        ),
    )

    def load(*args, **kwargs):
        loaded.append(kwargs)
        return base

    monkeypatch.setattr(cli, "load_playback", load)

    def metrics(bundle, projection):
        item = selections[bundle.selection["annotation_id"]]
        row = metric_row(
            item["source_key"],
            item["start_frame"],
            item["end_frame"],
            item["object_category_level_1"],
        )
        row.update(
            clip_start_frame=bundle.start_index,
            clip_end_frame=bundle.end_index,
            public_recommendation="acceptable",
        )
        return row

    monkeypatch.setattr(cli, "annotation_metrics", metrics)
    monkeypatch.setattr(cli, "rank_examples", lambda rows: list(rows))

    def render(bundle, row, **kwargs):
        rendered.append(row["annotation_id"])
        return {
            "artifacts": {"video": f"{row['annotation_id']}.mp4", "thumbnail": "thumbnail.png"},
            "video": {
                "file_size_bytes": 123,
                "frame_count": bundle.end_index - bundle.start_index + 1,
            },
        }

    monkeypatch.setattr(cli, "_render_example", render)
    monkeypatch.setattr(cli, "export_compilation", lambda *args, **kwargs: compiled.append(args))
    output = tmp_path / "out"
    argv = [
        "--recording-id",
        RECORDING,
        "--output-dir",
        str(output),
        "--qc-passing-only",
        "--skip-compilations",
        "--workers",
        "1",
    ]
    return SimpleNamespace(
        cli=cli,
        maps=maps,
        base=base,
        loaded=loaded,
        rendered=rendered,
        compiled=compiled,
        output=output,
        argv=argv,
        directory=directory,
        metrics=metrics,
        render=render,
    )


def test_gated_export_continues_after_mapping_rejection_and_skips_compilations(gated_batch):
    case = gated_batch
    case.maps["helper"].status[150] = "UNRESOLVED"
    assert case.cli.main(case.argv) == 0
    assert case.rendered == ["000600"]
    assert case.compiled == []
    assert case.loaded == [{"annotation_id": "000600", "context_frames": 120}]
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert len(summary["rows"]) == 1
    assert summary["rows"][0]["numeric_id"] == "0008"
    assert summary["rejected_annotations"][0]["annotation_id"] == "000200"
    assert "fixture reason" in (case.output / "examples_rejected.csv").read_text()


def test_all_mapping_blocked_writes_rejections_without_loading_caches(gated_batch):
    case = gated_batch
    for mapping in case.maps.values():
        mapping.status[:] = "UNRESOLVED"
    assert case.cli.main(case.argv) == 0
    assert case.loaded == case.rendered == case.compiled == []
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert summary["status"] == "NO_ELIGIBLE_EXPORTS"
    assert len(summary["rejected_annotations"]) == 2
    assert (case.output / "examples_manifest.csv").read_text().startswith("recording_id,")


def test_missing_verified_calibration_blocks_all_and_records_reason(gated_batch):
    case = gated_batch
    (case.directory / "vrs/helper_export_rgb_calibration.json").write_text(
        json.dumps({"status": "INFERRED"})
    )
    assert case.cli.main(case.argv) == 0
    assert case.loaded == case.rendered == []
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert all("not VERIFIED" in row["rejection_reason"] for row in summary["rejected_annotations"])


def test_qc_excluded_event_does_not_block_acceptable_event(gated_batch, monkeypatch):
    case = gated_batch

    def metrics(bundle, projection):
        row = case.metrics(bundle, projection)
        if row["annotation_id"] == "000200":
            row["public_recommendation"] = "exclude_from_public_demo"
        return row

    monkeypatch.setattr(case.cli, "annotation_metrics", metrics)
    assert case.cli.main(case.argv) == 0
    assert case.rendered == ["000600"]
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert summary["rejected_annotations"][0]["rejection_stage"] == "tracking_qc"


def test_render_failure_is_recorded_without_blocking_remaining_clip(gated_batch, monkeypatch):
    case = gated_batch

    def render(bundle, row, **kwargs):
        if row["annotation_id"] == "000200":
            raise ValueError("synthetic failed video verification")
        return case.render(bundle, row, **kwargs)

    monkeypatch.setattr(case.cli, "_render_example", render)
    assert case.cli.main(case.argv) == 0
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert summary["rows"][0]["annotation_id"] == "000600"
    assert summary["rejected_annotations"][0]["rejection_stage"] == "render_validation"


def test_fallback_context_is_recorded_and_passed_to_canonical_loader(gated_batch):
    case = gated_batch
    case.maps["helper"].status[85] = "UNRESOLVED"
    assert case.cli.main(case.argv) == 0
    assert case.loaded[0] == {"annotation_id": "000200", "context_frames": 90}
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert summary["rows"][0]["context_frames_each_side"] == 90
    assert summary["rows"][0]["clip_start_frame"] == 110
    assert summary["rows"][1]["context_frames_each_side"] == 120


def test_default_export_path_remains_opt_in(cli, monkeypatch, tmp_path):
    def unexpected(*args, **kwargs):
        raise AssertionError("gated mode must remain opt-in")

    monkeypatch.setattr(cli, "_qc_passing_export", unexpected)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        cli.main(["--recording-id", RECORDING, "--output-dir", str(tmp_path / "out")])


def test_annotation_assembly_failure_does_not_block_another_annotation(gated_batch, monkeypatch):
    case = gated_batch
    select = case.base.with_annotation

    def with_annotation(annotation_id, *, context_frames):
        if annotation_id == "000200":
            raise ValueError("synthetic annotation binding mismatch")
        return select(annotation_id, context_frames=context_frames)

    monkeypatch.setattr(case.base, "with_annotation", with_annotation)
    assert case.cli.main(case.argv) == 0
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert [row["annotation_id"] for row in summary["rows"]] == ["000600"]
    assert summary["rejected_annotations"][0]["rejection_stage"] == "annotation_validation"


def test_declared_calibration_without_canonical_frustum_is_rejected(gated_batch):
    case = gated_batch
    del case.base.bindings.image_geometry["helper"]
    assert case.cli.main(case.argv) == 0
    assert case.rendered == []
    summary = json.loads((case.output / "examples_validation.json").read_text())
    assert all("frustums" in row["rejection_reason"] for row in summary["rejected_annotations"])


@pytest.fixture
def demo_batch(gated_batch, monkeypatch):
    case = gated_batch
    case.assessed = []
    case.argv = [
        value for value in case.argv if value not in ("--qc-passing-only", "--skip-compilations")
    ]
    case.argv.extend(["--qc-profile", "demo", "--workers", "4"])

    def assess(base, annotation_id, projection_report, *, context_frames=120):
        case.assessed.append((annotation_id, context_frames))
        bundle = base.with_annotation(annotation_id, context_frames=context_frames)
        row = case.metrics(bundle, projection_report)
        row.update(
            qc_profile="demo",
            public_recommendation="exclude_from_public_demo",
            public_recommendation_reason="Synthetic strict four-hand failure",
            strict_qc_pass=False,
            demo_qc_pass=True,
            demo_qc_reason="Synthetic demo pass",
            event_helper_any_hand_coverage=0.95,
            event_leader_any_hand_coverage=0.9,
        )
        return row

    monkeypatch.setattr(case.cli.demo_metric_api, "assess_demo_qc", assess, raising=False)

    def compilation(path, segments, **kwargs):
        case.compiled.append((path, segments, kwargs))
        return {
            "path": str(path),
            "frame_count": sum(
                item["end_ordinal"] - item["start_ordinal"] + 1 for item in segments
            ),
        }

    monkeypatch.setattr(case.cli, "export_compilation", compilation)
    case.assess = assess
    return case


def test_demo_profile_preserves_every_existing_strict_artifact(demo_batch):
    case = demo_batch
    case.output.mkdir(parents=True)
    protected = [
        "examples_manifest.csv",
        "examples_validation.json",
        "examples_rejected.csv",
        "examples_eligibility.csv",
        "duet_v0_montage.mp4",
        "duet_v0_best_examples.mp4",
        "examples/handover_strict.mp4",
    ]
    for name in protected:
        path = case.output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"immutable strict artifact")
    assert case.cli.main(case.argv) == 0
    assert all(
        (case.output / name).read_bytes() == b"immutable strict artifact" for name in protected
    )
    assert (case.output / "examples_demo_qc").is_dir()
    assert (case.output / "demo_qc_manifest.csv").exists()
    summary = json.loads((case.output / "demo_qc_validation.json").read_text())
    assert summary["qc_profile"] == "demo"
    assert summary["annotation_count"] == 2
    assert summary["strict_qc_pass_count"] == 0
    assert summary["demo_qc_pass_count"] == summary["exported_count"] == 2
    assert len(case.compiled) == 1
    assert case.compiled[0][0].name == "duet_v0_montage_demo_qc.mp4"
    assert all(
        item["end_ordinal"] - item["start_ordinal"] + 1 == 120 for item in case.compiled[0][1]
    )


def test_demo_profile_routes_renderer_only_to_new_directory_and_labels_profile(
    demo_batch, monkeypatch
):
    case = demo_batch
    calls = []

    def render(bundle, row, **kwargs):
        calls.append((row["qc_profile"], kwargs["output_dir"], kwargs["input_identity"]))
        return case.render(bundle, row, **kwargs)

    monkeypatch.setattr(case.cli, "_render_example", render)
    assert case.cli.main(case.argv) == 0
    assert len(calls) == 2
    for profile, output, identity in calls:
        assert profile == identity["qc_profile"] == "demo"
        assert output == case.output / "examples_demo_qc"
    assert not (case.output / "examples").exists()
    assert not (case.output / "examples_manifest.csv").exists()


def test_demo_manifest_includes_mapping_rejected_annotation_metrics(demo_batch):
    case = demo_batch
    case.maps["helper"].status[150] = "UNRESOLVED"
    assert case.cli.main(case.argv) == 0
    assert case.assessed == [("000200", 120), ("000600", 120)]
    assert case.rendered == ["000600"]
    summary = json.loads((case.output / "demo_qc_validation.json").read_text())
    rejected, accepted = summary["rows"]
    assert rejected["demo_qc_pass"] is False
    assert rejected["export_status"] == "rejected"
    assert rejected["event_helper_any_hand_coverage"] == 0.95
    assert rejected["mapping_rejections"][0]["frames"][0]["frame_index"] == 150
    assert rejected["numeric_id"] == "7"
    assert accepted["numeric_id"] == "0008"
    assert accepted["export_status"] == "exported"
    assert "000200" in (case.output / "demo_qc_manifest.csv").read_text()


def test_demo_tracking_failure_is_reported_without_affecting_other_export(demo_batch, monkeypatch):
    case = demo_batch

    def assess(*args, **kwargs):
        row = case.assess(*args, **kwargs)
        if row["annotation_id"] == "000200":
            row.update(demo_qc_pass=False, demo_qc_reason="Any-hand event coverage below 80%")
        return row

    monkeypatch.setattr(case.cli.demo_metric_api, "assess_demo_qc", assess)
    assert case.cli.main(case.argv) == 0
    assert case.rendered == ["000600"]
    summary = json.loads((case.output / "demo_qc_validation.json").read_text())
    assert summary["rows"][0]["rejection_reason"] == "Any-hand event coverage below 80%"
    assert summary["rows"][0]["rejection_stage"] == "demo_tracking_qc"


def test_demo_default_output_is_recording_scoped(demo_batch, monkeypatch):
    case = demo_batch
    argv = list(case.argv)
    position = argv.index("--output-dir")
    del argv[position : position + 2]
    assert case.cli.main(argv) == 0
    expected = case.cli.ROOT / "outputs/v0_demo" / RECORDING
    assert (expected / "demo_qc_validation.json").is_file()
    assert not (case.cli.ROOT / "outputs/v0_demo/demo_qc_validation.json").exists()


def test_demo_montage_uses_exact_four_seconds_of_each_passing_clip(cli):
    rows = [metric_row("000200", 200, 210, "bowl"), metric_row("000600", 600, 820, "spoon")]
    segments = cli._demo_segments(rows)
    assert len(segments) == 2
    for row, segment in zip(rows, segments, strict=True):
        assert segment["end_ordinal"] - segment["start_ordinal"] + 1 == 120
        assert (
            0
            <= segment["start_ordinal"]
            <= segment["end_ordinal"]
            <= row["clip_end_frame"] - row["clip_start_frame"]
        )


def test_demo_reuse_identity_includes_profile_and_partial_tracking_label(
    cli, tmp_path, monkeypatch
):
    from types import SimpleNamespace

    output = tmp_path / "examples_demo_qc"
    output.mkdir()
    stem = "handover_000200_bowl"
    suffixes = (
        ".mp4",
        "_thumbnail.png",
        "_first.png",
        "_middle.png",
        "_last.png",
        "_validation.json",
    )
    for suffix in suffixes:
        (output / f"{stem}{suffix}").write_text("fixture")
    identity = {
        "fixture": True,
        "annotation_id": "000200",
        "object_category": "bowl",
        "start_frame": 0,
        "end_frame": 2,
        "qc_label": "Partial hand tracking",
        "render_recipe_version": 1,
        "qc_profile": "demo",
    }
    previous = {"render_identity": identity, "video_sha256": "fixture-hash"}
    (output / f"{stem}_validation.json").write_text(json.dumps(previous))
    bundle = SimpleNamespace(
        selection={"annotation_id": "000200", "object_category_level_1": "bowl"},
        start_index=0,
        end_index=2,
        aligned_arrays={
            role: {"hand_high_confidence": np.zeros((3, 2), dtype=bool)}
            for role in ("helper", "leader")
        },
    )
    monkeypatch.setattr(cli, "_digest", lambda path: "fixture-hash")
    monkeypatch.setattr(cli, "inspect_video", lambda *args, **kwargs: {})
    result = cli._render_example(
        bundle,
        {"qc_profile": "demo"},
        dataset_root=tmp_path / "raw",
        output_dir=output,
        input_identity={"fixture": True},
        reuse_clips=True,
    )
    assert result == previous


@pytest.mark.parametrize(
    "recommendation,expected", [("tracking_caution", True), ("unrecognized", False)]
)
def test_demo_manifest_strict_pass_uses_only_existing_accepted_labels(
    demo_batch, monkeypatch, recommendation, expected
):
    case = demo_batch

    def assess(*args, **kwargs):
        row = case.assess(*args, **kwargs)
        row["public_recommendation"] = recommendation
        return row

    monkeypatch.setattr(case.cli.demo_metric_api, "assess_demo_qc", assess)
    assert case.cli.main(case.argv) == 0
    summary = json.loads((case.output / "demo_qc_validation.json").read_text())
    assert all(row["strict_qc_pass"] is expected for row in summary["rows"])
