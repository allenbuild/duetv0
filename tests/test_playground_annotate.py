"""annotate stage: keyframe schedule, contact sheet, JSON extraction, cached runs with a mocked backend, CLI parsing, API."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.playground import annotate as A, api_annotate  # noqa: E402
from duet.playground.episode import Episode, Stream  # noqa: E402

REPLY = {"persons": [{"label": "leader", "action": "stirring bowl", "holding": ["whisk"], "attending_to": "bowl"},
                     {"label": "helper", "action": "holding bowl", "holding": "bowl", "attending_to": "partner"}],
         "coordination": "joint_work", "handover": {"occurring": False, "giver": None, "receiver": None, "object": None}, "summary": "Two people mix batter together."}


@pytest.fixture
def ep(tmp_path):
    import cv2
    d = tmp_path / "tmp_ann_unit"; d.mkdir()
    e = Episode(name="tmp_ann_unit", root=str(d), proc_fps=10.0, common_start_s=1.0, common_end_s=3.5,
                streams=[Stream(name="leader", role="ego", path="streams/leader.mp4", person="leader"), Stream(name="cam", role="exo", path="streams/cam.mp4")])
    for s, (w, h) in (("leader", (64, 64)), ("cam", (96, 54))):
        fd = d / "derived" / "frames" / s; fd.mkdir(parents=True)
        for k in range(25):
            im = np.full((h, w, 3), (k * 10) % 255, np.uint8); cv2.imwrite(str(fd / f"{k + 1:06d}.jpg"), im)
    e.save()
    return e


def test_keyframe_indices():
    assert A.keyframe_indices(25, 10.0, 2.0) == [0, 20]
    assert A.keyframe_indices(466, 10.0, 2.0)[:3] == [0, 20, 40] and len(A.keyframe_indices(466, 10.0, 2.0)) == 24
    assert A.keyframe_indices(466, 10.0, 2.0, max_keyframes=10) == list(range(0, 200, 20))
    assert A.keyframe_indices(5, 10.0, 0.01) == [0, 1, 2, 3, 4]


def test_contact_sheet_layout(ep):
    from duet.playground.perception import _frame_list
    frames = {s.name: _frame_list(ep, s) for s in ep.streams}
    sheet = A.contact_sheet(ep, 3, frames, t_s=1.3, max_w=1280)
    assert sheet.ndim == 3 and sheet.shape[1] == 1280 and sheet.shape[0] > 0
    sheet2 = A.contact_sheet(ep, 3, frames, max_w=400)
    assert sheet2.shape[1] == 400


@pytest.mark.parametrize("text", [
    json.dumps(REPLY),
    "Here you go:\n```json\n" + json.dumps(REPLY) + "\n```\nHope this helps.",
    "Sure. " + json.dumps(REPLY) + " Let me know.",
    json.dumps({**REPLY, "summary": "braces {inside} \"quoted}\" text"}) + " trailing } brace",
])
def test_extract_json(text):
    d = A.extract_json(text)
    assert d["coordination"] == "joint_work" and len(d["persons"]) == 2


def test_extract_json_rejects_prose():
    with pytest.raises(ValueError):
        A.extract_json("I cannot see any image.")


def test_normalize_coerces_schema():
    n = A.normalize(REPLY)
    assert n["persons"][1]["holding"] == ["bowl"] and n["handover"]["occurring"] is False
    n2 = A.normalize({"coordination": "dancing", "persons": "nope"})
    assert n2["persons"] == [] and n2["coordination"] == "dancing" and n2["handover"]["occurring"] is None and n2["summary"] is None


def test_run_with_mock_backend_caches(ep):
    calls = []

    def fake(image, prompt, model):
        calls.append(Path(image)); assert Path(image).exists() and "JSON" in prompt
        return "```json\n" + json.dumps(REPLY) + "\n```", {"input_tokens": 100, "output_tokens": 20}

    log = A.run_annotate(ep, interval_s=2.0, call=fake, backend="mock", model="m", log=lambda s: None)
    assert log["count"] == 2 and log["new"] == 2 and log["input_tokens"] == 200 and log["output_tokens"] == 40 and len(calls) == 2
    items = json.load(open(ep.dir / "derived/annotate/annotations.json"))
    assert isinstance(items, list) and [it["idx"] for it in items] == [0, 1] and [it["frame"] for it in items] == [0, 20]
    assert [it["t"] for it in items] == [1.0, 3.0]
    assert items[0]["image"] == "derived/annotate/keyframes/0000.jpg" and (ep.dir / items[0]["image"]).exists()
    assert items[0]["persons"][0]["action"] == "stirring bowl" and items[0]["coordination"] == "joint_work" and items[0]["summary"]
    assert items[0]["_meta"]["backend"] == "mock" and items[0]["_meta"]["model"] == "m"
    assert (ep.dir / "derived/annotate/keyframes/0001.json").exists()
    assert Episode.load(ep.dir).status["annotate"]["state"] == "done"
    # second run: everything cached, backend not called again
    log2 = A.run_annotate(ep, interval_s=2.0, call=fake, backend="mock", log=lambda s: None)
    assert log2["cached"] == 2 and log2["new"] == 0 and len(calls) == 2
    # max_keyframes limits the schedule
    log3 = A.run_annotate(ep, interval_s=1.0, max_keyframes=1, call=fake, backend="mock", log=lambda s: None)
    assert log3["count"] == 1 and len(json.load(open(ep.dir / "derived/annotate/annotations.json"))) == 1


def test_backend_errors_are_recorded_and_retried(ep):
    n = {"calls": 0}

    def bad(image, prompt, model):
        n["calls"] += 1
        if n["calls"] == 1:
            raise RuntimeError("boom")
        return json.dumps(REPLY), {}

    log = A.run_annotate(ep, interval_s=2.0, call=bad, backend="mock", log=lambda s: None)
    items = json.load(open(ep.dir / "derived/annotate/annotations.json"))
    assert log["errors"] == 1 and items[0]["error"].startswith("RuntimeError") and items[0]["persons"] == [] and items[1]["summary"]
    assert Episode.load(ep.dir).status["annotate"]["state"] == "done"  # partial failure keeps the stage usable
    log2 = A.run_annotate(ep, interval_s=2.0, call=bad, backend="mock", log=lambda s: None)
    assert log2["errors"] == 0 and log2["new"] == 1 and log2["cached"] == 1  # failed keyframe retried, good one cached


def test_dry_backend_and_upgrade(ep):
    log = A.run_annotate(ep, interval_s=2.0, backend="dry", log=lambda s: None)
    items = json.load(open(ep.dir / "derived/annotate/annotations.json"))
    assert log["backend"] == "dry" and items[0]["persons"] == [] and items[0]["summary"] is None and "dry backend" in items[0]["note"]
    # a real backend later redoes dry placeholders
    log2 = A.run_annotate(ep, interval_s=2.0, call=lambda i, p, m: (json.dumps(REPLY), {}), backend="mock", log=lambda s: None)
    assert log2["new"] == 2 and json.load(open(ep.dir / "derived/annotate/annotations.json"))[0]["summary"]


def test_stage_entry_point_reads_options(ep, monkeypatch):
    monkeypatch.setenv("DUET_ANNOTATE_BACKEND", "dry"); monkeypatch.setenv("DUET_ANNOTATE_INTERVAL_S", "1.0"); monkeypatch.setenv("DUET_ANNOTATE_MAX_KEYFRAMES", "2")
    A.annotate(ep)
    items = json.load(open(ep.dir / "derived/annotate/annotations.json"))
    assert len(items) == 2 and [it["frame"] for it in items] == [0, 10] and ep.status["annotate"]["state"] == "done"
    ep.notes = ["annotate: backend=dry max_keyframes=3 interval_s=0.5"]; monkeypatch.delenv("DUET_ANNOTATE_INTERVAL_S"); monkeypatch.delenv("DUET_ANNOTATE_MAX_KEYFRAMES")
    assert A._options(ep) == {"interval_s": 0.5, "backend": "dry", "model": None, "max_keyframes": 3}


def test_select_backend(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(A, "cli_available", lambda force=False: (False, "not logged in"))
    monkeypatch.setattr(A, "codex_available", lambda force=False: (False, "codex CLI: Not logged in"))
    assert A.select_backend() == ("dry", "no ANTHROPIC_API_KEY; not logged in; codex CLI: Not logged in")
    monkeypatch.setattr(A, "cli_available", lambda force=False: (True, "ok"))
    assert A.select_backend()[0] == "claude_cli"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    assert A.select_backend()[0] == "anthropic"
    assert A.select_backend("dry") == ("dry", "requested")
    with pytest.raises(ValueError):
        A.select_backend("gpt")


def test_claude_cli_backend_parses_envelope(monkeypatch, tmp_path):
    img = tmp_path / "kf.jpg"; img.write_bytes(b"\xff\xd8\xff")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd; seen["env"] = kw.get("env")
        env = {"is_error": False, "result": "Here is the JSON:\n" + json.dumps(REPLY), "usage": {"input_tokens": 1500, "output_tokens": 120}, "total_cost_usd": 0.01, "duration_api_ms": 4000, "num_turns": 2}
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(env), stderr="")

    monkeypatch.setattr(A.subprocess, "run", fake_run); monkeypatch.setenv("CLAUDECODE", "1")
    text, meta = A.call_claude_cli(img, A.PROMPT, "haiku")
    assert A.extract_json(text)["coordination"] == "joint_work" and meta["input_tokens"] == 1500 and meta["output_tokens"] == 120 and meta["cost_usd"] == 0.01
    assert seen["cmd"][:2] == ["claude", "-p"] and str(img.resolve()) in seen["cmd"][2] and "--output-format" in seen["cmd"] and "json" in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--allowedTools") + 1] == "Read" and seen["cmd"][-2:] == ["--model", "haiku"] and "CLAUDECODE" not in seen["env"]

    def fake_fail(cmd, **kw):
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"is_error": True, "result": "Not logged in"}), stderr="")

    monkeypatch.setattr(A.subprocess, "run", fake_fail)
    with pytest.raises(RuntimeError, match="Not logged in"):
        A.call_claude_cli(img, A.PROMPT, None)
    assert A.cli_available(force=True) == (False, "claude CLI: Not logged in")
    monkeypatch.setattr(A.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"is_error": False, "result": "ok"}), stderr=""))
    assert A.cli_available(force=True) == (True, "ok")
    A._cli_probe = None  # do not leak the fake probe into other tests


def test_anthropic_backend_request_shape(monkeypatch, tmp_path):
    import types
    img = tmp_path / "kf.jpg"; img.write_bytes(b"\xff\xd8\xff\x00")
    seen = {}

    class FakeMessages:
        def create(self, **kw):
            seen.update(kw)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text=json.dumps(REPLY))], usage=types.SimpleNamespace(input_tokens=900, output_tokens=80), stop_reason="end_turn")

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = FakeMessages()

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", FakeClient)
    text, meta = A.call_anthropic(img, A.PROMPT, None)
    assert meta == {"input_tokens": 900, "output_tokens": 80, "stop_reason": "end_turn"} and A.extract_json(text)["summary"]
    assert seen["model"] == "claude-haiku-4-5-20251001"
    blocks = seen["messages"][0]["content"]
    assert blocks[0]["type"] == "image" and blocks[0]["source"]["type"] == "base64" and blocks[0]["source"]["media_type"] == "image/jpeg" and blocks[1]["text"] == A.PROMPT


def test_api_annotations(ep, monkeypatch):
    monkeypatch.setattr(api_annotate, "EPISODES", ep.dir.parent)
    app = FastAPI(); app.include_router(api_annotate.router); c = TestClient(app)
    r = c.get(f"/api/episode/{ep.name}/annotations"); assert r.status_code == 200 and r.json()["available"] is False and r.json()["items"] == []
    A.run_annotate(ep, interval_s=2.0, call=lambda i, p, m: (json.dumps(REPLY), {"input_tokens": 1, "output_tokens": 1}), backend="mock", log=lambda s: None)
    j = c.get(f"/api/episode/{ep.name}/annotations").json()
    assert j["available"] and len(j["items"]) == 2 and j["items"][0]["image_url"] == f"/episodes/{ep.name}/derived/annotate/keyframes/0000.jpg"
    assert j["log"]["backend"] == "mock" and j["status"]["state"] == "done" and j["items"][1]["t"] == 3.0
    assert c.get("/api/episode/nope/annotations").status_code == 404
