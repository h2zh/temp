"""End to end through the HTTP API with real COLMAP: check-ins grow from one facade to the full model.

The prototype spec's acceptance scenario: synthetic photos (640x480, 4 per sector,
all 8 sectors, plus one photo of each bad kind) uploaded to an explicit landmark,
with jobs run inline. Each visitor uploads their own photos in one request.
"""

from __future__ import annotations

import importlib.util
import json
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import uvicorn
from fastapi.testclient import TestClient

from crowd3d.api import create_app
from crowd3d.config import Settings
from crowd3d.geo import angle_diff_deg
from crowd3d.synthetic import generate_dataset

pytestmark = pytest.mark.slow

LANDMARK_ID = "e2e-building"
SIZE = (640, 480)
PER_SECTOR = 4
NUM_SECTORS = 8


@dataclass
class Stage:
    """Upload results of one stage, paired with the manifest entries they belong to."""

    results: list[tuple[dict[str, Any], dict[str, Any]]] = field(default_factory=list)
    triggered_jobs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class Run:
    client: TestClient
    settings: Settings
    manifest: dict[str, Any]
    first_side: Stage
    detail_after_first_side: dict[str, Any]
    rest: Stage
    bad: Stage
    detail: dict[str, Any]
    truth: dict[str, dict[str, Any]]      # photo_id -> manifest entry, accepted photos only
    full_points: np.ndarray               # (n, 3) ENU points of the latest full model


def upload_by_user(client: TestClient, photo_dir: Path, entries: list[dict[str, Any]]) -> Stage:
    """POST /api/photos once per user (in order of first appearance) with that user's photos."""
    by_user: dict[str, list[dict[str, Any]]] = {}
    for e in entries:
        by_user.setdefault(e["user_id"], []).append(e)
    stage = Stage()
    for user, batch in by_user.items():
        files = [("files", (e["file"], (photo_dir / e["file"]).read_bytes(), "image/jpeg")) for e in batch]
        r = client.post("/api/photos", data={"user_id": user, "landmark_id": LANDMARK_ID}, files=files)
        assert r.status_code == 200, r.text
        body = r.json()
        assert [res["filename"] for res in body["results"]] == [e["file"] for e in batch]
        stage.results += zip(batch, body["results"])
        stage.triggered_jobs += body["triggered_jobs"]
    return stage


def read_ply_xyz(text: str) -> np.ndarray:
    lines = text.splitlines()
    body = lines[lines.index("end_header") + 1:]
    return np.array([[float(v) for v in line.split()[:3]] for line in body]).reshape(-1, 3)


def camera_errors(detail: dict[str, Any], truth: dict[str, dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Per photo with an estimate: distance to the true camera centre, heading error vs the true heading."""
    estimated = [p for p in detail["photos"] if p["estimated_enu"] is not None]
    pos = [np.linalg.norm(np.subtract(p["estimated_enu"], truth[p["id"]]["true_enu"])) for p in estimated]
    head = [abs(angle_diff_deg(p["estimated_heading_deg"], truth[p["id"]]["true_heading_deg"])) for p in estimated]
    return np.array(pos), np.array(head)


@pytest.fixture(scope="module")
def run(tmp_path_factory) -> Iterator[Run]:
    root = tmp_path_factory.mktemp("e2e")
    photo_dir = root / "photos"
    manifest = generate_dataset(photo_dir, size=SIZE, photos_per_sector=PER_SECTOR, num_sectors=NUM_SECTORS, seed=0)
    good = [e for e in manifest["photos"] if e["kind"] == "good"]
    bad = [e for e in manifest["photos"] if e["kind"] != "good"]
    lm = manifest["landmark"]
    settings = Settings(data_dir=root / "data", reconstructor="colmap", inline_jobs=True)

    with TestClient(create_app(settings)) as client:
        r = client.post(
            "/api/landmarks",
            json={"id": LANDMARK_ID, "name": lm["name"], "lat": lm["lat"], "lon": lm["lon"], "radius_m": 150},
        )
        assert r.status_code == 201, r.text

        first_side = upload_by_user(client, photo_dir, [e for e in good if e["sector"] == 0])
        detail_after_first_side = client.get(f"/api/landmarks/{LANDMARK_ID}").json()
        rest = upload_by_user(client, photo_dir, [e for e in good if e["sector"] != 0])
        # Bad photos last: the duplicate is only a duplicate once its original is in.
        bad_stage = upload_by_user(client, photo_dir, bad)
        detail = client.get(f"/api/landmarks/{LANDMARK_ID}").json()

        truth = {res["photo_id"]: e for e, res in first_side.results + rest.results if res["accepted"]}
        full = detail["latest_full_model"]
        points = np.zeros((0, 3))
        if full is not None:
            ply = client.get(full["url"])
            assert ply.status_code == 200
            points = read_ply_xyz(ply.text)
        yield Run(client, settings, manifest, first_side, detail_after_first_side, rest, bad_stage, detail,
                  truth, points)


def test_first_side_gives_one_georegistered_facade_model(run):
    assert all(res["accepted"] for _, res in run.first_side.results)
    (job,) = run.first_side.triggered_jobs
    assert (job["kind"], job["sector"], job["status"], job["num_photos"]) == ("face", 0, "done", PER_SECTOR)
    assert job["error"] is None

    d = run.detail_after_first_side
    assert [j["id"] for j in d["jobs"]] == [job["id"]]
    (model,) = d["models"]
    assert (model["kind"], model["sector"], model["num_photos"]) == ("face", 0, PER_SECTOR)
    assert model["georegistered"]
    assert model["num_registered"] >= 3
    assert model["num_points"] >= 1000
    assert d["latest_full_model"] is None and not d["full_ready"]
    assert d["progress"] == pytest.approx(1 / NUM_SECTORS)
    states = [(s["state"], s["has_face_model"]) for s in d["sectors"]]
    assert states == [("face_ready", True)] + [("locked", False)] * (NUM_SECTORS - 1)

    # Estimated poses come from the facade model and match the true cameras.
    pos, head = camera_errors(d, run.truth)
    assert len(pos) == model["num_registered"]
    assert pos.max() < 5.0, pos
    assert head.max() < 10.0, head


def test_full_model_meets_acceptance_thresholds(run):
    full = run.detail["latest_full_model"]
    assert full is not None
    n = NUM_SECTORS * PER_SECTOR
    assert full["num_photos"] == n
    assert full["num_registered"] >= 0.8 * n
    assert full["georegistered"]
    assert full["alignment_rmse_m"] < 5.0
    assert full["heading_mae_deg"] < 10.0
    assert full["num_points"] >= 1000

    # Exported points are in ENU around the landmark centre.
    assert len(run.full_points) == full["num_points"]
    assert np.linalg.norm(np.median(run.full_points[:, :2], axis=0)) < 6.0

    meta = json.loads((run.settings.models_dir / LANDMARK_ID / f"{full['id']}.json").read_text(encoding="utf-8"))
    cameras = meta["cameras"]
    assert len(cameras) == full["num_registered"]
    pos = np.array([np.linalg.norm(np.subtract(c["enu"], run.truth[pid]["true_enu"])) for pid, c in cameras.items()])
    head = np.array([abs(angle_diff_deg(c["heading_deg"], run.truth[pid]["true_heading_deg"]))
                     for pid, c in cameras.items()])
    assert (pos < 5.0).mean() >= 0.8, pos
    assert (head < 10.0).mean() >= 0.8, head

    # The API's per-photo estimates prefer the full model over the facade models.
    for p in run.detail["photos"]:
        if p["id"] in cameras:
            assert p["estimated_enu"] == cameras[p["id"]]["enu"]
            assert p["estimated_heading_deg"] == cameras[p["id"]]["heading_deg"]


def test_bad_photos_are_rejected_with_their_expected_reasons(run):
    assert len(run.bad.results) == 6
    for entry, res in run.bad.results:
        assert not res["accepted"], entry["file"]
        assert res["reasons"] == [entry["expected_reason"]], entry["file"]
        assert res["photo_id"] is not None     # recorded for feedback, bytes not kept
    assert run.bad.triggered_jobs == []
    accepted_ids = {p["id"] for p in run.detail["photos"]}
    assert not accepted_ids & {res["photo_id"] for _, res in run.bad.results}
    stored = list((run.settings.photos_dir / LANDMARK_ID).iterdir())
    assert len(stored) == 2 * NUM_SECTORS * PER_SECTOR    # original + processed copy per accepted photo


def test_landmark_detail_reflects_the_crowd(run):
    d = run.detail
    good = [e for e in run.manifest["photos"] if e["kind"] == "good"]
    assert all(res["accepted"] for _, res in run.first_side.results + run.rest.results)
    assert d["num_photos"] == len(good) == len(d["photos"])
    assert d["progress"] == 1.0 and d["full_ready"]
    assert [(s["count"], s["state"], s["has_face_model"]) for s in d["sectors"]] == (
        [(PER_SECTOR, "face_ready", True)] * NUM_SECTORS
    )

    per_user = Counter(e["user_id"] for e in good)
    assert d["num_contributors"] == len(per_user)
    assert d["contributors"] == [
        {"user_id": u, "photos": c} for u, c in sorted(per_user.items(), key=lambda uc: (-uc[1], uc[0]))
    ]

    # Every job succeeded: facade models for every sector (a sector may be rebuilt once neighbours'
    # photos inside face_margin_deg add up), and a single full model once all sides were in.
    assert {j["status"] for j in d["jobs"]} == {"done"}
    assert {j["sector"] for j in d["jobs"] if j["kind"] == "face"} == set(range(NUM_SECTORS))
    (full_job,) = [j for j in d["jobs"] if j["kind"] == "full"]
    assert run.client.get(f"/api/jobs/{full_job['id']}").json() == full_job
    assert [(m["kind"], m["sector"]) for m in d["models"]] == (
        [("full", None)] + [("face", s) for s in range(NUM_SECTORS)]
    )

    # Uploads report the sector the photographer stood in; the overview lists the full model too.
    for entry, res in run.first_side.results + run.rest.results:
        assert res["sector"] == entry["sector"]
    assert sum(p["estimated_heading_deg"] is not None for p in d["photos"]) >= 0.8 * len(good)
    summaries = {s["id"]: s for s in run.client.get("/api/landmarks").json()["landmarks"]}
    assert summaries[LANDMARK_ID]["latest_full_model"] == d["latest_full_model"]


def test_demo_script_against_a_live_server(tmp_path, capsys):
    """scripts/demo.py over real HTTP; the fake reconstructor keeps it quick."""
    settings = Settings(data_dir=tmp_path / "data", reconstructor="fake")
    app = create_app(settings)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started
    port = server.servers[0].sockets[0].getsockname()[1]
    spec = importlib.util.spec_from_file_location("demo", Path(__file__).resolve().parent.parent / "scripts" / "demo.py")
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    try:
        rc = demo.main(["--server", f"http://127.0.0.1:{port}", "--photos", str(tmp_path / "photos"), "--users", "3"])
        store = app.state.store
        photos = store.list_photos("demo-building", "accepted")
        jobs = store.list_jobs("demo-building")
    finally:
        server.should_exit = True
        thread.join(10)
    out = capsys.readouterr().out
    assert rc == 0, out
    assert len(photos) == NUM_SECTORS * PER_SECTOR
    assert {p.user_id for p in photos} == {"游客1", "游客2", "游客3"}
    assert {j.status for j in jobs} == {"done"}
    (full_job,) = [j for j in jobs if j.kind == "full"]
    assert len(full_job.photo_ids) == NUM_SECTORS * PER_SECTOR    # the last stage completes the ring at once
    assert out.count("✗") == 6 and "照片太模糊" in out
    assert f"http://127.0.0.1:{port}/landmark.html?id=demo-building" in out
