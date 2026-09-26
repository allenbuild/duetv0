"""Review API on the CoMind clip: deterministic sample (read-only on the real episode), verdicts + summary + proposal status
(on a symlinked copy of the episode so test verdicts never land in the real derived/review)."""
import json
import os
import shutil
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from duet.playground import api_review  # noqa: E402

REAL = ROOT / "data/playground/episodes/comind_43276420_clip"
pytestmark = pytest.mark.skipif(not (REAL / "derived/body2d/leader.npz").exists(), reason="comind_43276420_clip with derived stages not present")


def _client():
    app = FastAPI(); app.include_router(api_review.router); return TestClient(app)


def test_sample_is_deterministic_and_json_clean():
    c = _client()
    r = c.get("/api/episode/comind_43276420_clip/review/sample?frac=0.02&seed=0"); assert r.status_code == 200
    j = r.json(); frames = [it for it in j["items"] if it["kind"] == "frame"]
    assert j["n_frames"] == {"leader": 466, "helper": 466, "gopro_front": 466, "gopro_back": 466}
    assert {it["stream"] for it in frames} == {"leader", "helper", "gopro_front", "gopro_back"}
    assert all(sum(it["stream"] == s for it in frames) == 9 for s in j["n_frames"])  # round(0.02 * 466)
    same = c.get("/api/episode/comind_43276420_clip/review/sample?frac=0.02&seed=0").json()
    assert [it["id"] for it in same["items"]] == [it["id"] for it in j["items"]]
    other = c.get("/api/episode/comind_43276420_clip/review/sample?frac=0.02&seed=1").json()
    assert [it["id"] for it in other["items"]] != [it["id"] for it in j["items"]]
    it = next(x for x in frames if x["stream"] == "leader")
    assert it["id"] == f"leader:{it['frame_idx']}" and it["image"].startswith("/episodes/comind_43276420_clip/derived/frames/leader/") and it["img_w"] == 640 and it["img_h"] == 640
    assert it["has"] == {"body2d": True, "hands": True, "objects": True}
    exo = next(x for x in frames if x["stream"] == "gopro_front"); assert exo["has"] == {"body2d": True, "hands": False, "objects": False} and exo["img_w"] == 640
    # shapes and NaN -> null
    text = json.dumps(j); assert "NaN" not in text and "Infinity" not in text
    for f in frames:
        for p in f["body2d"]:
            assert len(p) == 17 and all(len(k) == 3 for k in p)
        for h in f["hands"]:
            assert h["side"] in ("left", "right") and len(h["landmarks"]) == 21
        for o in f["objects"]:
            assert len(o["box"]) == 4 and 0 <= o["conf"] <= 1 and o["name"]
    assert any(f["body2d"] for f in frames if f["stream"].startswith("gopro")) and any(f["hands"] for f in frames if f["stream"] in ("leader", "helper"))
    assert c.get("/api/episode/nope/review/sample").status_code == 404


@pytest.fixture
def copy_ep(tmp_path, monkeypatch):
    """episode.json copied, derived frames/body2d/hands/objects symlinked; review + autolabel dirs private to the test."""
    d = tmp_path / "comind_43276420_clip"; (d / "derived").mkdir(parents=True)
    shutil.copy(REAL / "episode.json", d / "episode.json")
    for sub in ("frames", "body2d", "hands", "objects"):
        os.symlink(REAL / "derived" / sub, d / "derived" / sub)
    (d / "derived/autolabel").mkdir()
    json.dump([{"event": "handover", "t0": 12.0, "t1": 13.5, "confidence": 0.7, "giver": "leader", "receiver": "helper", "status": "proposed"},
               {"event": "joint_attention", "t0": 30.0, "t1": 34.0, "confidence": 0.4, "giver": None, "receiver": None, "status": "proposed"}], open(d / "derived/autolabel/proposals.json", "w"))
    monkeypatch.setattr(api_review, "EPISODES", tmp_path)
    return d


def test_verdicts_summary_and_proposals(copy_ep):
    c = _client(); name = "comind_43276420_clip"
    j = c.get(f"/api/episode/{name}/review/sample?frac=0.01&seed=3").json()
    frames = [it for it in j["items"] if it["kind"] == "frame"]; props = [it for it in j["items"] if it["kind"] == "proposal"]
    assert len(props) == 2 and props[0]["id"] == "proposal:0" and props[0]["event"] == "handover" and j["verdicts"] == {}
    s0 = c.get(f"/api/episode/{name}/review/summary").json()
    assert s0["total"] == 0 and s0["by_item"]["body2d"]["accuracy"] is None and s0["proposals"] == {"total": 2, "accepted": 0, "rejected": 0, "pending": 2}
    a, b = frames[0], next(x for x in frames if x["stream"] != frames[0]["stream"])
    post = lambda **kw: c.post(f"/api/episode/{name}/review", json=kw)  # noqa: E731
    assert post(id=a["id"], stream=a["stream"], frame_idx=a["frame_idx"], item="body2d", verdict="ok", note="").status_code == 200
    assert post(id=a["id"], stream=a["stream"], frame_idx=a["frame_idx"], item="hands", verdict="wrong", note="missed left hand").status_code == 200
    assert post(id=b["id"], stream=b["stream"], frame_idx=b["frame_idx"], item="body2d", verdict="unsure", note="").status_code == 200
    assert post(id=a["id"], stream=a["stream"], frame_idx=a["frame_idx"], item="body2d", verdict="wrong", note="changed my mind").status_code == 200  # latest wins
    r = post(id="proposal:0", stream="proposals", frame_idx=None, item="proposal", verdict="ok", note="")
    assert r.status_code == 200 and r.json()["proposal_updated"] is True
    assert post(id="proposal:1", item="proposal", verdict="wrong", note="no").status_code == 200
    lines = [json.loads(l) for l in open(copy_ep / "derived/review/verdicts.jsonl")]
    assert len(lines) == 6 and lines[0]["item"] == "body2d" and lines[-1]["stream"] == "proposals" and all("ts" in l for l in lines)
    props_file = json.load(open(copy_ep / "derived/autolabel/proposals.json"))
    assert props_file[0]["status"] == "accepted" and props_file[1]["status"] == "rejected" and props_file[1]["review_note"] == "no"
    s = c.get(f"/api/episode/{name}/review/summary").json()
    assert s["total"] == 5  # 4 distinct (id,item) frame verdicts... 3 frame + 2 proposal
    assert s["by_item"]["body2d"] == {"n": 2, "ok": 0, "wrong": 1, "unsure": 1, "accuracy": 0.0}
    assert s["by_item"]["hands"] == {"n": 1, "ok": 0, "wrong": 1, "unsure": 0, "accuracy": 0.0}
    assert s["by_item"]["proposal"] == {"n": 2, "ok": 1, "wrong": 1, "unsure": 0, "accuracy": 0.5}
    assert s["by_stream"][a["stream"]]["n"] == 2 and s["by_stream"]["proposals"]["n"] == 2 and s["by_stream_item"][a["stream"]]["hands"]["wrong"] == 1
    assert s["proposals"] == {"total": 2, "accepted": 1, "rejected": 1, "pending": 0}
    # the sample now carries the latest verdicts so the page can resume
    j2 = c.get(f"/api/episode/{name}/review/sample?frac=0.01&seed=3").json()
    assert j2["verdicts"][a["id"]]["body2d"]["verdict"] == "wrong" and j2["verdicts"][a["id"]]["hands"]["note"] == "missed left hand"
    assert next(p for p in j2["items"] if p["id"] == "proposal:0")["status"] == "accepted"


def test_verdict_validation(copy_ep):
    c = _client(); name = "comind_43276420_clip"
    assert c.post(f"/api/episode/{name}/review", json={"id": "leader:1", "item": "body2d", "verdict": "maybe"}).status_code == 422
    assert c.post(f"/api/episode/{name}/review", json={"id": "leader:1", "item": "face", "verdict": "ok"}).status_code == 422
    assert c.post("/api/episode/nope/review", json={"id": "x", "item": "body2d", "verdict": "ok"}).status_code == 404
    r = c.post(f"/api/episode/{name}/review", json={"id": "proposal:99", "item": "proposal", "verdict": "ok"})
    assert r.status_code == 200 and r.json()["proposal_updated"] is False  # recorded, but no such proposal to update
