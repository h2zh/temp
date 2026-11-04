import json
import logging
import math
import os
import re
import stat
import time
from contextlib import ExitStack
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from crowd3d import api as api_module
from crowd3d.api import LANDMARK_ID_PATTERN, WEB_DIR, create_app, slugify
from crowd3d import photoposes as pp
from crowd3d.consent import (PUBLIC_ID_SALT_ENV, SALT_CHECK_KEY, SaltMismatch, deployment_salt, manage_token,
                             public_id, salt_check, salt_path)
from crowd3d.consent import main as consent_main
from crowd3d.reconstruct import FakeReconstructor
from crowd3d.records import (
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_STYLIZE,
    LOCATION_CONSENT_CURRENT,
    LOCATION_CONSENT_V1,
    LOCATION_CONSENT_V2,
    PHOTO_ACCEPTED,
    Job,
    ModelRecord,
    PhotoRecord,
    location_shared,
)
from crowd3d.store import Store
from crowd3d.synthetic import CAMERA_RADIUS_M, DEMO_LAT, DEMO_LON, DEMO_NAME

LM = "demo"
SUMMARY_KEYS = {"id", "name", "lat", "lon", "radius_m", "num_photos", "num_contributors", "progress", "full_ready",
                "sectors", "latest_full_model"}
DETAIL_KEYS = SUMMARY_KEYS | {"contributors", "models", "jobs", "thresholds"}   # no per-photo rows (D30)
SECTOR_KEYS = {"index", "label", "count", "state", "bearing_center_deg", "has_face_model"}
MODEL_KEYS = {"id", "kind", "sector", "url", "created_at", "num_photos", "num_registered", "num_points",
              "georegistered", "alignment_rmse_m", "heading_mae_deg", "geo", "stylized"}
GEO_KEYS = {"level", "sigma", "source", "calibration_id", "position_sigma_m", "calibrated"}
JOB_KEYS = {"id", "landmark_id", "kind", "sector", "status", "created_at", "started_at", "finished_at", "error",
            "num_photos", "model_id"}
UPLOAD_KEYS = {"filename", "photo_id", "accepted", "reasons", "warnings", "landmark_id", "sector", "bearing_deg",
               "distance_m", "share_location", "public_id", "manage_token"}
POSES_KEYS = {"format", "version", "landmark", "frame", "as_of", "rounding", "uncertainty", "counts", "photos"}
COUNT_KEYS = {"photos", "shared", "shown", "sfm", "loc", "exif", "not_determined"}
# Lower full-model thresholds so the two-sector dataset can reach them.
TWO_SECTOR_FULL = {"full_min_sectors": 2, "full_min_photos": 8}


class SizeTaggingReconstructor(FakeReconstructor):
    """Fake reconstruction whose estimated headings equal the job's photo count, to tell models apart."""

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir):
        result = super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)
        for cam in result.cameras.values():
            cam["heading_deg"] = float(len(photos))
        return result


@pytest.fixture
def make_client(settings):
    with ExitStack() as stack:

        def make(reconstructor=None, **overrides):
            app = create_app(replace(settings, **overrides), reconstructor)
            return stack.enter_context(TestClient(app))

        yield make


@pytest.fixture
def client(make_client):
    return make_client()


def create_landmark(client, ds=None, **fields):
    payload = {"id": LM, "name": "Demo", "lat": ds.lat if ds else DEMO_LAT, "lon": ds.lon if ds else DEMO_LON}
    r = client.post("/api/landmarks", json={**payload, **fields})
    assert r.status_code == 201, r.text
    return r.json()


def upload(client, ds, entries, *, user_id="alice", landmark_id=LM, share_location=None):
    """share_location: the form field's raw value, sent only when given."""
    files = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in entries]
    data = {"user_id": user_id} | ({"landmark_id": landmark_id} if landmark_id else {})
    data |= {"share_location": share_location} if share_location is not None else {}
    r = client.post("/api/photos", data=data, files=files)
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == {"results", "triggered_jobs"}
    for res in body["results"]:
        assert set(res) == UPLOAD_KEYS
    for job in body["triggered_jobs"]:
        assert set(job) == JOB_KEYS
    return body


def detail(client, landmark_id=LM):
    r = client.get(f"/api/landmarks/{landmark_id}")
    assert r.status_code == 200, r.text
    return r.json()


# Health and static files


def test_health_names_the_reconstructor(client):
    assert client.get("/api/health").json() == {"ok": True, "reconstructor": "fake"}


def test_web_ui_is_served_at_root_without_shadowing_the_api(client):
    r = client.get("/")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert "<html" in r.text.lower()
    assert client.get("/landmark.html").status_code == 200
    assert client.get("/api.js").status_code == 200
    assert client.get("/api/landmarks").json() == {"landmarks": []}
    assert client.get("/api/nothing-here").status_code == 404


# Landmarks


def test_create_landmark_returns_an_empty_summary(client):
    lm = create_landmark(client)
    assert set(lm) == SUMMARY_KEYS
    assert (lm["id"], lm["name"], lm["lat"], lm["lon"]) == (LM, "Demo", DEMO_LAT, DEMO_LON)
    assert lm["radius_m"] == 150.0  # Settings.default_radius_m
    assert (lm["num_photos"], lm["num_contributors"], lm["progress"], lm["full_ready"]) == (0, 0, 0.0, False)
    assert lm["latest_full_model"] is None
    assert [s["label"] for s in lm["sectors"]] == ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    for i, s in enumerate(lm["sectors"]):
        assert set(s) == SECTOR_KEYS
        assert (s["index"], s["count"], s["state"], s["has_face_model"]) == (i, 0, "locked", False)
        assert s["bearing_center_deg"] == pytest.approx(45.0 * i)
    assert client.get("/api/landmarks").json() == {"landmarks": [lm]}


def test_landmark_detail_and_404(client):
    create_landmark(client, radius_m=80)
    d = detail(client)
    assert set(d) == DETAIL_KEYS
    assert d["radius_m"] == 80
    assert (d["contributors"], d["models"], d["jobs"]) == ([], [], [])
    assert d["thresholds"] == {"face_min_photos": 4, "full_min_photos": 24, "full_min_sectors": 6, "num_sectors": 8}
    assert client.get("/api/landmarks/nope").status_code == 404


@pytest.mark.parametrize(
    "name,expected",
    [
        ("Eiffel Tower", "eiffel-tower"),
        ("  Café de Flore! ", "cafe-de-flore"),
        ("Tower 42 -- North", "tower-42-north"),
        ("上海 Tower", "tower"),
    ],
)
def test_landmark_id_is_slugged_from_the_name(client, name, expected):
    r = client.post("/api/landmarks", json={"name": name, "lat": 1.0, "lon": 2.0})
    assert r.status_code == 201
    assert r.json()["id"] == expected and r.json()["name"] == name.strip()


def test_name_without_ascii_falls_back_to_a_random_id(client):
    ids = [client.post("/api/landmarks", json={"name": "外滩钟楼", "lat": 1, "lon": 2}).json()["id"] for _ in range(2)]
    assert all(re.fullmatch(r"lm-[0-9a-f]{8}", i) for i in ids)
    assert ids[0] != ids[1]


def test_slug_is_capped_at_64_chars_and_matches_the_id_pattern():
    assert slugify("a" * 63 + " b") == "a" * 63
    assert slugify("x" * 100) == "x" * 64
    for name in ["a" * 63 + " b", "Ünïcödé — ok", "--x--", "x" * 100]:
        assert re.fullmatch(LANDMARK_ID_PATTERN, slugify(name))


def test_duplicate_landmark_id_is_409(client):
    create_landmark(client)
    assert client.post("/api/landmarks", json={"id": LM, "name": "Other", "lat": 0, "lon": 0}).status_code == 409
    assert client.post("/api/landmarks", json={"name": "Eiffel Tower", "lat": 0, "lon": 0}).status_code == 201
    r = client.post("/api/landmarks", json={"name": "eiffel tower", "lat": 5, "lon": 5})
    assert r.status_code == 409 and "eiffel-tower" in r.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "x", "lat": 90.01, "lon": 0},
        {"name": "x", "lat": -91, "lon": 0},
        {"name": "x", "lat": 0, "lon": 180.5},
        {"name": "x", "lat": 0, "lon": -181},
        {"name": "x", "lat": 0, "lon": 0, "radius_m": 0},
        {"name": "x", "lat": 0, "lon": 0, "radius_m": -5},
        {"name": "x", "lat": 0, "lon": 0, "radius_m": 5000.1},
        {"name": "", "lat": 0, "lon": 0},
        {"name": "   ", "lat": 0, "lon": 0},
        {"name": "n" * 101, "lat": 0, "lon": 0},
        {"name": "x", "lat": 0, "lon": 0, "id": "Upper"},
        {"name": "x", "lat": 0, "lon": 0, "id": "-leading-dash"},
        {"name": "x", "lat": 0, "lon": 0, "id": "under_score"},
        {"name": "x", "lat": 0, "lon": 0, "id": "a" * 65},
        {"name": "x", "lat": 0, "lon": 0, "id": ""},
        {"name": "x", "lon": 0},
        {"lat": 0, "lon": 0},
    ],
)
def test_invalid_landmark_is_422(client, payload):
    r = client.post("/api/landmarks", json=payload)
    assert r.status_code == 422, r.text
    assert isinstance(r.json()["detail"], list)


def test_landmark_limits_are_inclusive(client):
    for i, payload in enumerate([
        {"lat": 90, "lon": 180, "radius_m": 5000, "name": "n" * 100},
        {"lat": -90, "lon": -180, "radius_m": 0.5, "name": "m"},
    ]):
        r = client.post("/api/landmarks", json={"id": f"edge-{i}", **payload})
        assert r.status_code == 201, r.text
    assert client.post("/api/landmarks", json={"id": "a" * 64, "name": "x", "lat": 0, "lon": 0}).status_code == 201


def test_non_finite_numbers_are_422_not_500(client):
    r = client.post("/api/landmarks", content=b'{"name": "x", "lat": NaN, "lon": Infinity}',
                    headers={"content-type": "application/json"})
    assert r.status_code == 422
    assert {e["loc"][-1] for e in r.json()["detail"]} == {"lat", "lon"}


# Uploads


def test_upload_face_then_full_with_every_bad_photo_rejected(make_client, synthetic_dataset):
    ds = synthetic_dataset
    client = make_client(**TWO_SECTOR_FULL)
    create_landmark(client, ds)

    # Sector 0 reaches face_min_photos -> exactly one facade job, run inline.
    sector0 = ds.entries("good", sector=0)
    first = upload(client, ds, sector0)
    assert [r["accepted"] for r in first["results"]] == [True] * 4
    assert [r["filename"] for r in first["results"]] == [e["file"] for e in sector0]
    for res, e in zip(first["results"], sector0):
        # the test landmark has no bound building outline: the subject rule is skipped, with a warning (D31)
        assert (res["landmark_id"], res["sector"], res["reasons"], res["warnings"]) == (LM, 0, [], ["subject_unchecked"])
        assert len(res["photo_id"]) == 32 and res["bearing_deg"] is not None
        assert CAMERA_RADIUS_M[0] - 10 < res["distance_m"] < CAMERA_RADIUS_M[1] + 10
    (job,) = first["triggered_jobs"]
    assert (job["kind"], job["sector"], job["status"], job["num_photos"], job["landmark_id"]) == (
        "face", 0, "done", 4, LM)
    assert client.get(f"/api/jobs/{job['id']}").json() == job

    d = detail(client)
    assert d["sectors"][0] == {
        "index": 0, "label": "N", "count": 4, "state": "face_ready", "bearing_center_deg": 0.0, "has_face_model": True}
    (model,) = d["models"]
    assert set(model) == MODEL_KEYS
    assert (model["kind"], model["sector"], model["num_photos"], model["num_registered"]) == ("face", 0, 4, 4)
    assert model["georegistered"] is True and model["num_points"] > 0
    assert model["url"] == f"/files/models/{LM}/{model['id']}.ply"
    assert set(model["geo"]) == GEO_KEYS and (model["geo"]["level"], model["geo"]["calibrated"]) == ("full", False)
    assert d["latest_full_model"] is None and d["progress"] == pytest.approx(4 / 4 / 8)

    # Sector 1 plus every bad photo (the duplicate copies a sector-1 photo, so it must come after it).
    rest = ds.entries("good", sector=1) + [e for e in ds.manifest["photos"] if e["kind"] != "good"]
    second = upload(client, ds, rest, user_id="bob")
    for res, e in zip(second["results"], rest):
        assert res["filename"] == e["file"]
        assert res["accepted"] is e["expected_valid"], (e["file"], res)
        assert res["reasons"] == ([] if e["expected_valid"] else [e["expected_reason"]])
        assert res["photo_id"] and res["landmark_id"] == LM
    assert sorted((j["kind"], j["sector"] if j["sector"] is not None else -1) for j in second["triggered_jobs"]) == [
        ("face", 1), ("full", -1)]
    assert all(j["status"] == "done" for j in second["triggered_jobs"])

    d = detail(client)
    assert (d["num_photos"], d["num_contributors"], d["full_ready"]) == (8, 2, True)
    assert d["contributors"] == [{"user_id": "alice"}, {"user_id": "bob"}]       # no per-person counts (D30)
    full = d["latest_full_model"]
    assert full["kind"] == "full" and full["sector"] is None and full["num_photos"] == 8
    assert [(m["kind"], m["sector"]) for m in d["models"]] == [("full", None), ("face", 0), ("face", 1)]
    assert [s["has_face_model"] for s in d["sectors"]] == [True, True] + [False] * 6
    assert set(d) == DETAIL_KEYS
    assert len(d["jobs"]) == 3 and d["jobs"][-1]["id"] == job["id"]  # newest first

    # The model file is served at its URL.
    r = client.get(full["url"])
    assert r.status_code == 200 and r.content.startswith(b"ply\n")


def estimates(client):
    """api.camera_estimates of the store: internal (the benchmark reads it), never served."""
    return api_module.camera_estimates(client.app.state.store.published_models(LM))


def accepted_photos(client):
    return {p.id: p for p in client.app.state.store.list_photos(LM, PHOTO_ACCEPTED)}


def test_camera_estimates_come_from_the_newest_model_preferring_full(make_client, synthetic_dataset):
    ds = synthetic_dataset
    client = make_client(SizeTaggingReconstructor(), **TWO_SECTOR_FULL)
    create_landmark(client, ds)
    upload(client, ds, ds.entries("good", sector=0))
    photos = accepted_photos(client)
    est = estimates(client)
    assert set(est) == set(photos)
    for pid, cam in est.items():
        assert cam["heading_deg"] == 4.0  # the 4-photo facade model
        e, n, z = cam["enu"]
        assert math.hypot(e, n) == pytest.approx(photos[pid].distance_m, rel=1e-6) and z == 1.6
        assert photos[pid].heading_deg is not None and photos[pid].heading_deg != cam["heading_deg"]

    upload(client, ds, ds.entries("good", sector=1))
    est = estimates(client)
    assert set(photos) < set(est) == set(accepted_photos(client))
    # Every photo is in the 8-photo full model, which wins over the (newer or older) facade models.
    assert [c["heading_deg"] for c in est.values()] == [8.0] * 8
    assert "photos" not in detail(client)


def test_photo_without_model_has_no_estimate(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    body = upload(client, ds, ds.entries("good", sector=0)[:2])
    assert body["triggered_jobs"] == []
    assert len(accepted_photos(client)) == 2 and estimates(client) == {}


class MissingPhotoFirstReconstructor(FakeReconstructor):
    """Fails its first job the way a vanished photo file does: an OSError naming the photo and its server path."""

    def __init__(self, settings=None):
        super().__init__(settings)
        self.calls = 0

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, **kw):
        self.calls += 1
        if self.calls == 1:
            p = photos[0]
            raise FileNotFoundError(2, f"No such file or directory (photo {p.photo_id})", str(p.image_path))
        return super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir, **kw)


_NUMBER_IN_TEXT = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_HEX_ID = re.compile(r"(?<![0-9a-fA-F])[0-9a-f]{32,}(?![0-9a-fA-F])")


def _numbers_in_text(text: str) -> list[float]:
    """Every number printed in a response body, in order: JSON numbers and numbers inside strings alike (the digits
    of a model or job id are not numbers)."""
    return [float(x) for x in _NUMBER_IN_TEXT.findall(_HEX_ID.sub(" ", text))]


def _near(numbers: list[float], value: float, tol: float) -> bool:
    return any(abs(x - value) <= tol for x in numbers)


def test_no_public_get_serves_per_photo_data(make_client, synthetic_dataset, settings):
    """D30: no unauthenticated GET (the list, the detail, the public poses, each job, the served model files) serves a
    store photo id, a server path, the salt, an uploader beside a photo, a photo's position (GPS, bearing, distance),
    heading or upload time, or a camera's pose, in any field or string (a failed job's error included), except the
    public poses of the photos whose uploader agreed: exactly their rounded public rows. The contributor list
    (nicknames) and the sector counts are aggregates."""
    ds = synthetic_dataset
    client = make_client(MissingPhotoFirstReconstructor(settings), **TWO_SECTOR_FULL)
    create_landmark(client, ds)
    shared = upload(client, ds, ds.entries("good", sector=0), user_id="alice", share_location="true")
    upload(client, ds, ds.entries("good", sector=1), user_id="bob")
    tokens = [res["manage_token"] for res in shared["results"]]
    assert len(tokens) == 4 and all(tokens)
    secret = deployment_salt(settings.data_dir)
    assert client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}).status_code == 202
    photos = accepted_photos(client)
    est = estimates(client)
    assert len(photos) == 8 and set(est) == set(photos)          # every photo has an internal camera estimate
    store = client.app.state.store
    (failed,) = [j for j in store.list_jobs(LM) if j.status == JOB_FAILED]
    assert str(settings.data_dir) in failed.error and any(pid in failed.error for pid in photos)  # the store keeps it
    poses_path = f"/api/landmarks/{LM}/poses"
    docs = {"/api/health": client.get("/api/health"), "/api/landmarks": client.get("/api/landmarks"),
            f"/api/landmarks/{LM}": client.get(f"/api/landmarks/{LM}"), poses_path: client.get(poses_path)}
    for j in store.list_jobs(LM):
        docs[f"/api/jobs/{j.id}"] = client.get(f"/api/jobs/{j.id}")
    assert len(docs) >= 8 and all(r.status_code == 200 for r in docs.values())
    # the public poses: alice's four photos, as their public rows exactly; nothing of bob's
    consented = {res["photo_id"] for res in shared["results"]}
    assert consented < set(photos) and all(photos[pid].user_id == "alice" for pid in consented)
    poses = docs[poses_path].json()
    internal = {p.photo_id: p for p in pp.photo_poses(photos.values(), store.published_models(LM), ds.lat, ds.lon,
                                                      aspects=pp.processed_aspects(photos.values(), settings.data_dir))}
    assert poses["photos"] == sorted((pp.public_row(internal[pid], secret) for pid in consented),
                                     key=lambda row: row["id"])
    assert {row["id"] for row in poses["photos"]} == {res["public_id"] for res in shared["results"]}
    assert poses["counts"]["shared"] == 4 and poses["counts"]["photos"] == 8
    assert (poses["rounding"], poses["uncertainty"]) == (pp.public_rounding(), pp.public_uncertainty())
    assert poses["frame"]["origin"] == {"lat": ds.lat, "lon": ds.lon} and set(poses) == POSES_KEYS
    for row in poses["photos"]:
        assert set(row) == set(pp.PUBLIC_KEYS) and all(isinstance(v, int) for v in row["p"][:2] + [row["yaw"]])
    for pid in set(photos) - consented:                 # no row stands where a non-consented camera stood
        e, n, _ = est[pid]["enu"]
        assert not any(abs(row["p"][0] - e) <= 0.5 and abs(row["p"][1] - n) <= 0.5 for row in poses["photos"]), pid
    assert docs[f"/api/jobs/{failed.id}"].json()["error"] == (
        "FileNotFoundError: [Errno 2] No such file or directory (photo <id>): '<path>'")
    assert failed.id in {j["id"] for j in docs[f"/api/landmarks/{LM}"].json()["jobs"]}
    users = {p.user_id for p in photos.values()}
    assert users == {"alice", "bob"}
    for url, r in docs.items():
        text = r.text
        for pid in photos:
            assert pid not in text, (url, pid)
        assert secret not in text and not any(t in text for t in tokens), url      # D30 C1: salt, manage tokens
        assert salt_check(secret) not in text, url
        assert str(settings.data_dir) not in text and "/data/" not in text, url
        for key in ("estimated_enu", "estimated_heading_deg", "bearing_deg", "distance_m", "uploaded_at", "photo_ids"):
            assert f'"{key}"' not in text, (url, key)
        body = r.json()
        # the public poses' rows are the consented photos' public rows and its rounding, uncertainty and frame are
        # constants (checked above): scan the rest of that document, and every other GET whole, for anything of any
        # photo
        constant = ("photos", "rounding", "uncertainty", "frame")
        scanned = json.dumps({k: v for k, v in body.items() if k not in constant}) if url == poses_path else text
        nums = _numbers_in_text(scanned)
        pairs = list(zip(nums, nums[1:]))
        for pid, cam in est.items():
            e, n, _ = cam["enu"]
            # printed to 0.1 m anywhere, or to whole metres (or finer) side by side, in either order
            assert not (_near(nums, e, 0.05) and _near(nums, n, 0.05)), (url, pid, cam["enu"])
            assert not any(abs(a - e) <= 0.5 and abs(b - n) <= 0.5 or abs(a - n) <= 0.5 and abs(b - e) <= 0.5
                           for a, b in pairs), (url, pid, cam["enu"])
        for p in photos.values():
            assert not (_near(nums, p.lat, 1e-5) and _near(nums, p.lon, 1e-5)), (url, p.id)   # GPS to 5 decimals
            # its upload time, exactly (a job's public created_at follows it within milliseconds)
            assert not _near(nums, p.distance_m, 0.05) and p.uploaded_at not in nums, (url, p.id)
        # headings to 1 degree (the photo's compass and the camera's): one may match a public number by chance
        # (a sector centre, a count), every photo's cannot
        headings = [h for p in photos.values() for h in (p.heading_deg, est[p.id]["heading_deg"]) if h is not None]
        assert len(headings) == 16
        hits = sum(_near(nums, h, 0.5) for h in headings)
        assert hits <= len(headings) // 4, (url, hits)
        # an uploader appears only in the contributor list, never beside a photo, a sector or a job
        rest = json.dumps({k: v for k, v in body.items() if k != "contributors"}, ensure_ascii=False)
        for u in users:
            assert u not in rest, (url, u)
    # the served point clouds carry no photo id either
    for m in docs[f"/api/landmarks/{LM}"].json()["models"]:
        r = client.get(m["url"])
        assert r.status_code == 200 and not any(pid.encode() in r.content for pid in photos), m["url"]
    d = docs[f"/api/landmarks/{LM}"].json()
    # nicknames only, by name: per-person counts would tie public poses to nicknames (D30, owner question 2)
    assert "photos" not in d and d["contributors"] == [{"user_id": "alice"}, {"user_id": "bob"}]
    assert not any("photos" in m for m in d["models"]) and not any("photos" in j for j in d["jobs"])


def test_public_text_drops_server_paths_and_ids():
    pid, model, other = "53a62d31b8bd4b788d3e8d6dd505ce12", "0123456789abcdef0123456789abcdef", "f" * 32
    assert api_module.public_text(
        f"FileNotFoundError: [Errno 2] No such file or directory: '/srv/crowd3d/data/photos/demo/{pid}.proc.jpg'"
    ) == "FileNotFoundError: [Errno 2] No such file or directory: '<path>'"
    assert api_module.public_text(f"ValueError: photo {pid} has no processed copy or GPS position") == (
        "ValueError: photo <id> has no processed copy or GPS position")
    assert api_module.public_text(f"COLMAP failed: cannot read 00001_{pid}.jpg in photos/demo/{pid}.jpg") == (
        "COLMAP failed: cannot read 00001_<id>.jpg in photos/demo/<id>.jpg")
    assert api_module.public_text(f"sha256 {pid}{pid}") == "sha256 <id>"
    # model ids are public; ids the response serves anyway may stay
    msg = f"superseded: model {model} is no longer the one its target shows (model {other} is)"
    assert api_module.public_text(msg) == msg
    assert api_module.public_text(f"job {pid} ok", {pid}) == f"job {pid} ok"
    # no false hits on ordinary text
    for text in ("quality check failed, rejected: 3/5 photos registered", "km/h a / b", "", None):
        assert api_module.public_text(text) == text


class PriorOnlyReconstructor(FakeReconstructor):
    """The fake reconstruction, recorded as not georegistered (a prior-only anchor: compass yaw, a guessed scale)."""

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, **kw):
        result = super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir, **kw)
        result.georegistered = False
        result.anchor = {**result.anchor, "level": "prior_only"}
        return result


def poses(client, landmark_id=LM):
    r = client.get(f"/api/landmarks/{landmark_id}/poses")
    assert r.status_code == 200, r.text
    body = r.json()
    assert set(body) == POSES_KEYS and set(body["counts"]) == COUNT_KEYS
    assert (body["format"], body["version"]) == (pp.POSES_FORMAT, pp.POSES_VERSION) == ("crowd3d.photo-poses", 2)
    for row in body["photos"]:
        assert set(row) == set(pp.PUBLIC_KEYS)
    assert [row["id"] for row in body["photos"]] == sorted(row["id"] for row in body["photos"])   # no upload order
    return body


def internal_rows(client, salt, *, snaps=None):
    """public_row of every consented photo's internal pose (photoposes.photo_poses on the store), by public id."""
    store, settings = client.app.state.store, client.app.state.settings
    lm = store.get_landmark(LM)
    photos = [p for p in store.list_photos(LM, PHOTO_ACCEPTED) if location_shared(p)]
    out = pp.photo_poses(photos, store.published_models(LM), lm.lat, lm.lon, snaps=snaps,
                         aspects=pp.processed_aspects(photos, settings.data_dir))
    return {row["id"]: row for pose in out if (row := pp.public_row(pose, salt)) is not None}


def test_public_poses_show_consented_photos_only_as_their_rounded_public_rows(make_client, synthetic_dataset,
                                                                               settings):
    """D30: GET /api/landmarks/{id}/poses lists the accepted photos whose uploader agreed, as public_row of their
    internal pose (whole metres and degrees, the salted public id the upload returned), sorted by id; a photo without
    consent (NULL, as every photo stored before the consent existed) or with a consent version this server never
    showed is not in it (a V1 consent, given under the earlier hint for the same sharing, still is), and counts say
    how many agreed and how many could be placed."""
    ds = synthetic_dataset
    client = make_client(**TWO_SECTOR_FULL)
    create_landmark(client, ds)
    assert client.get("/api/landmarks/nowhere/poses").status_code == 404
    empty = poses(client)
    assert (empty["as_of"], empty["photos"]) == (None, [])
    assert empty["counts"] == dict.fromkeys(COUNT_KEYS, 0)
    alice = upload(client, ds, ds.entries("good", sector=0), user_id="alice", share_location="true")["results"]
    upload(client, ds, ds.entries("good", sector=1), user_id="bob")
    store, salt = client.app.state.store, deployment_salt(settings.data_dir)
    # an old row with a consent text this server never showed counts as not shared
    (bobs,) = [p for p in store.list_photos(LM, PHOTO_ACCEPTED) if p.user_id == "bob"][:1]
    with store._lock:
        store._conn.execute("UPDATE photos SET location_consent = 'share-location-v0' WHERE id = ?", (bobs.id,))
        store._conn.execute("UPDATE photos SET location_consent = ? WHERE id = ?",
                            (LOCATION_CONSENT_V1, alice[0]["photo_id"]))
    assert store.get_photo(alice[0]["photo_id"]).location_consent == LOCATION_CONSENT_V1
    body = poses(client)
    ids = {res["public_id"] for res in alice}
    assert all(re.fullmatch(r"[0-9a-f]{12}", i) for i in ids) and len(ids) == 4
    assert ids == {public_id(res["photo_id"], salt) for res in alice} == {row["id"] for row in body["photos"]}
    assert {row["id"]: row for row in body["photos"]} == internal_rows(client, salt)
    assert body["counts"] == {"photos": 8, "shared": 4, "shown": 4, "sfm": 4, "loc": 0, "exif": 0,
                              "not_determined": 0}
    for row in body["photos"]:
        assert row["src"] == "sfm" and all(isinstance(v, int) for v in row["p"]) and 0 <= row["yaw"] < 360
    assert body["landmark"] == {"id": LM, "name": "Demo"} and body["frame"]["origin"] == {"lat": ds.lat, "lon": ds.lon}
    finished = [j.finished_at for j in store.list_jobs(LM) if j.kind in (JOB_FACE, "full")]
    assert body["as_of"] == max(finished)
    text = json.dumps(body, ensure_ascii=False)
    for p in store.list_photos(LM):
        assert p.id not in text and p.user_id not in text
    for key in ("user_id", "photo_id", "uploaded_at", "model_id", "sector", "time"):
        assert f'"{key}"' not in text, key


def test_public_poses_are_a_snapshot_of_the_last_finished_build_and_a_withdrawal_counts_at_once(
        client, synthetic_dataset, settings):
    """D30, owner question 2: the public poses cover the photos uploaded by the end of the landmark's newest
    reconstruction job (done or failed), so an upload alone changes nothing a visitor can diff; consent is read when
    served, so a withdrawn photo leaves at once."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    store, salt = client.app.state.store, deployment_salt(settings.data_dir)
    first = upload(client, ds, ds.entries("good", sector=0)[:2], share_location="true")
    assert first["triggered_jobs"] == []
    assert poses(client)["photos"] == [] and poses(client)["counts"]["shared"] == 0       # no build finished yet
    second = upload(client, ds, ds.entries("good", sector=0)[2:], share_location="true")
    (job,) = second["triggered_jobs"]
    body = poses(client)
    assert body["as_of"] == store.get_job(job["id"]).finished_at and body["counts"]["shown"] == 4
    four = {row["id"] for row in body["photos"]}
    # one more consented photo, no build: not shown yet, not even counted
    (late,) = upload(client, ds, ds.entries("good", sector=1)[:1], share_location="true")["results"]
    assert poses(client) == body
    # a withdrawal leaves the public poses at once
    res = first["results"][0]
    assert withdraw(client, res["photo_id"], res["manage_token"]).status_code == 200
    after = poses(client)
    assert {row["id"] for row in after["photos"]} == four - {res["public_id"]}
    assert after["counts"]["shared"] == 3 and after["as_of"] == body["as_of"]
    # the next build (any reconstruction job) brings the late photo in
    assert client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}).status_code == 202
    later = poses(client)
    assert later["as_of"] > body["as_of"] and later["counts"]["photos"] == 5 and later["counts"]["shared"] == 4
    assert late["public_id"] in {r["id"] for r in later["photos"]} and later["counts"]["shown"] == 4
    assert {r["id"]: r for r in later["photos"]} == internal_rows(client, salt)


def test_anyone_can_move_the_poses_snapshot_soon_after_a_new_consented_upload_as_the_consent_hint_says(
        make_client, synthetic_dataset, settings):
    """The remaining linking risk the consent hint states (LOCATION_CONSENT_V2, web/landmark.html share-location-hint,
    pinned in web/tests/pages.test.mjs): the snapshot holds a new consented row back only until the next reconstruction
    job ends, and anyone can make one end soon, by an anonymous rebuild or by a few uploads of their own (enough to
    start a facade job: face_min_photos on a new facade, face_rebuild_growth on a built one). Right after a new
    nickname appears among the contributors, the row that appears next can be tied to it. If either route stops moving
    the snapshot, the hint can change with it (under a new consent version)."""
    ds = synthetic_dataset
    client = make_client(face_min_photos=3, face_rebuild_growth=2, full_min_photos=100)
    create_landmark(client, ds)
    salt = deployment_salt(settings.data_dir)
    good0, good1 = ds.entries("good", sector=0), ds.entries("good", sector=1)
    assert upload(client, ds, good0[:3], user_id="alice", share_location="true")["triggered_jobs"]
    before = poses(client)
    assert before["counts"]["shown"] == 3

    def contributors():
        return {c["user_id"] for c in client.get(f"/api/landmarks/{LM}").json()["contributors"]}

    def new_rows(prev):
        now = poses(client)
        return now, {r["id"] for r in now["photos"]} - {r["id"] for r in prev["photos"]}

    # carol uploads one consented photo: her nickname shows at once, her row does not
    names = contributors()
    uploaded = upload(client, ds, good0[3:], user_id="carol", share_location="true")
    (carol,) = uploaded["results"]
    assert carol["share_location"] and uploaded["triggered_jobs"] == [] and poses(client) == before
    assert contributors() - names == {"carol"}
    # route 1: an anonymous rebuild ends a job, and the one new row is hers
    assert client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}).status_code == 202
    after, added = new_rows(before)
    assert after["as_of"] > before["as_of"] and added == {carol["public_id"]}
    # route 2: dan uploads one consented photo of another facade; someone else's own uploads there, without consent,
    # bring it to face_min_photos and start a job
    uploaded = upload(client, ds, good1[:1], user_id="dan", share_location="true")
    (dan,) = uploaded["results"]
    assert dan["share_location"] and uploaded["triggered_jobs"] == [] and poses(client) == after
    assert upload(client, ds, good1[1:3], user_id="mallory")["triggered_jobs"], "a few uploads start a job"
    final, added = new_rows(after)
    assert final["as_of"] > after["as_of"] and added == {dan["public_id"]}
    assert {r["id"]: r for r in final["photos"]} == internal_rows(client, salt)


GOLDEN_SPARSE = os.path.join(os.path.dirname(__file__), "fixtures", "recon_golden", "sparse")


class SparseWriting(FakeReconstructor):
    """The fake reconstruction that also leaves a COLMAP sparse model (the golden fixture's) in its work dir, so a
    published model gets a saved map, a candidate map of the localize jobs."""

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, **_):
        import shutil

        result = super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)
        shutil.copytree(GOLDEN_SPARSE, work_dir / "sparse" / "0", dirs_exist_ok=True)
        return result


class AcceptingLocalizer:
    """Stands in for localize.ColmapLocalizer: every candidate map loads, each query is localized into the first one,
    2 m east of its GPS fix, facing its compass heading."""

    def __init__(self):
        self.calls = []

    def load_maps(self, root, maps, cameras):
        return list(maps), {}

    def localize(self, queries, maps, photos, lm, work_dir, **_):
        from crowd3d import localize as loc

        self.calls.append([q.id for q in queries])
        run = loc.LocalizeRun()
        for q in queries:
            query = loc.make_query(q, 640, 480, lm.lat, lm.lon)
            e, n = query.gps_en
            stats = loc.PoseStats(maps[0].model_id, maps[0].kind, 200, 640, query.focal_px, focal_px=query.focal_px,
                                  focal_free_px=query.focal_px, inliers=190, reproj_px=0.5, enu=[e + 2.0, n, 1.5],
                                  heading_deg=q.heading_deg, pitch_deg=3.0, roll_deg=-1.0, look_deg=5.0)
            run.outcomes[q.id] = loc.Outcome(q, query, stats, None, [stats], {})
        return run


def test_a_consented_upload_no_model_shows_is_localized_by_an_internal_job_and_shown_as_a_loc_row(
        make_client, synthetic_dataset, settings):
    """D30, the upload path with Settings.localize_photos on: the request only queues a localize job (its check is
    cheap; the job runs after the reconstructions); the job is internal (never in triggered_jobs, the detail or
    /api/jobs/{id}); its localization is the photo's "loc" row in the poses, public_row of the internal pose, inside
    the snapshot like every row; a withdrawal removes the row and the record at once."""
    pytest.importorskip("pycolmap")
    ds = synthetic_dataset
    client = make_client(SparseWriting(), localize_photos=True)
    fake = client.app.state.runner._localizer_arg = AcceptingLocalizer()
    create_landmark(client, ds)
    store, salt = client.app.state.store, deployment_salt(settings.data_dir)
    # carol's consented photo of a side without a model (and outside the next facade job's margin): no map to
    # localize it in yet (recorded as no_maps)
    uploaded = upload(client, ds, ds.entries("good", sector=1)[-1:], user_id="carol", share_location="true")
    (carol,) = uploaded["results"]
    (first,) = [j for j in store.list_jobs(LM) if j.kind == "localize"]
    assert uploaded["triggered_jobs"] == [] and first.status == JOB_DONE and fake.calls == []
    assert store.get_localization(carol["photo_id"]).reason == "no_maps"
    assert client.get(f"/api/jobs/{first.id}").status_code == 404                  # internal
    # alice's photos make a facade model with a saved map: carol's photo is localized into it, after that job
    triggered = upload(client, ds, ds.entries("good", sector=0), user_id="alice", share_location="true")
    (face,) = triggered["triggered_jobs"]
    assert face["kind"] == JOB_FACE
    second = [j for j in store.list_jobs(LM) if j.kind == "localize"][1]
    assert second.status == JOB_DONE and second.started_at >= store.get_job(face["id"]).finished_at
    assert fake.calls == [[carol["photo_id"]]] and client.get(f"/api/jobs/{second.id}").status_code == 404
    assert {j["kind"] for j in detail(client)["jobs"]} == {JOB_FACE}
    rec = store.get_localization(carol["photo_id"])
    (model,) = store.published_models(LM)
    assert (rec.status, rec.model_id) == ("accepted", model.id)
    body = poses(client)
    assert body["counts"] == {"photos": 5, "shared": 5, "shown": 5, "sfm": 4, "loc": 1, "exif": 0,
                              "not_determined": 0}
    photos = [p for p in store.list_photos(LM, PHOTO_ACCEPTED) if location_shared(p)]
    internal = pp.photo_poses(photos, store.published_models(LM), ds.lat, ds.lon,
                              localizations=store.localizations(LM),
                              aspects=pp.processed_aspects(photos, settings.data_dir))
    rows = {row["id"]: row for pose in internal if (row := pp.public_row(pose, salt)) is not None}
    assert {r["id"]: r for r in body["photos"]} == rows and rows[carol["public_id"]]["src"] == "loc"
    assert carol["photo_id"] not in json.dumps(body)
    # withdrawn: the row and the localization go at once
    assert withdraw(client, carol["photo_id"], carol["manage_token"]).status_code == 200
    assert store.get_localization(carol["photo_id"]) is None
    assert carol["public_id"] not in {r["id"] for r in poses(client)["photos"]}


def test_switching_localization_off_withholds_the_stored_loc_rows_at_once(synthetic_dataset, settings):
    """D30: Settings.localize_photos off is how the owner withdraws the loc poses (no pose rather than a crude one):
    a server started with it off on a data dir whose localizations were stored with it on (by this server, or by a
    benchmark run) shows none of them; the photo falls back to the next source or is not determined, as if it had
    never been localized. The records are kept: switched on again, the same rows show, and no localize job is
    queued for them."""
    pytest.importorskip("pycolmap")
    ds, on = synthetic_dataset, replace(settings, localize_photos=True)
    with TestClient(create_app(on, SparseWriting())) as client:
        client.app.state.runner._localizer_arg = AcceptingLocalizer()
        create_landmark(client, ds)
        (carol,) = upload(client, ds, ds.entries("good", sector=1)[-1:], user_id="carol",
                          share_location="true")["results"]
        upload(client, ds, ds.entries("good", sector=0), user_id="alice", share_location="true")
        shown = poses(client)
        assert shown["counts"]["loc"] == 1
        rec = client.app.state.store.get_localization(carol["photo_id"])
        assert rec.status == "accepted"
    with TestClient(create_app(settings, SparseWriting())) as client:           # the flag off: none shown
        store, salt = client.app.state.store, deployment_salt(settings.data_dir)
        off = poses(client)
        assert off["counts"]["loc"] == 0 and "loc" not in {r["src"] for r in off["photos"]}
        assert off["counts"]["shown"] + off["counts"]["not_determined"] == off["counts"]["shared"] == 5
        photos = [p for p in store.list_photos(LM, PHOTO_ACCEPTED) if location_shared(p)]
        unlocalized = pp.photo_poses(photos, store.published_models(LM), ds.lat, ds.lon,
                                     aspects=pp.processed_aspects(photos, settings.data_dir))
        assert {r["id"]: r for r in off["photos"]} == {
            row["id"]: row for pose in unlocalized if (row := pp.public_row(pose, salt)) is not None}
        assert {r["id"]: r for r in off["photos"] if r["id"] != carol["public_id"]} == {
            r["id"]: r for r in shown["photos"] if r["id"] != carol["public_id"]}         # only the loc row changed
        assert store.get_localization(carol["photo_id"]) == rec                          # kept
        jobs = store.list_jobs(LM)
    with TestClient(create_app(on, SparseWriting())) as client:                 # on again: the same rows
        assert poses(client) == shown and client.app.state.store.list_jobs(LM) == jobs


def test_the_poses_snapshot_is_taken_when_a_reconstruction_job_ends_done_or_failed():
    def j(kind, status, finished):
        return Job(id=f"{kind}{status}{finished}", landmark_id=LM, kind=kind, sector=None, status=status, photo_ids=[],
                   created_at=0.0, finished_at=finished)

    assert api_module.poses_snapshot_time([]) is None
    assert api_module.poses_snapshot_time([j(JOB_FACE, "running", None), j(JOB_FACE, "queued", None)]) is None
    jobs = [j(JOB_FACE, JOB_DONE, 10.0), j("full", JOB_FAILED, 20.0), j(JOB_FACE, "running", None)]
    assert api_module.poses_snapshot_time(jobs) == 20.0                    # a failed build ends a batch too
    assert api_module.poses_snapshot_time([*jobs, j(JOB_STYLIZE, JOB_DONE, 30.0)]) == 20.0   # a stylize job does not


SNAP = {"rotation_deg": 90.0, "shift_m": [10.0, -5.0], "scale": 1.0, "accepted": True, "reason": ""}


def test_public_poses_move_cameras_with_the_snap_recorded_for_the_stylized_glb(make_client, synthetic_dataset,
                                                                               settings):
    """The rows are in the stylized GLB's frame: a model's cameras move with the accepted footprint snap its stylize
    job recorded (or, for a record from before the job runner kept it, the GLB's summary beside the PLY); a refused
    snap, or one of a model without a GLB, moves nothing. An accepted snap also fixes a prior-only model's frame,
    whose cameras are otherwise withheld (no crude pose)."""
    ds = synthetic_dataset
    client = make_client(PriorOnlyReconstructor())
    create_landmark(client, ds)
    upload(client, ds, ds.entries("good", sector=0), share_location="true")
    store, salt = client.app.state.store, deployment_salt(settings.data_dir)
    (model,) = store.published_models(LM)
    # prior-only, no snap: the cameras are withheld; the photos fall back to their own GPS and true-north compass
    # (exif: no measured height, pitch or roll)
    body = poses(client)
    assert {r["src"] for r in body["photos"]} == {"exif"} and body["counts"]["exif"] == 4
    assert all(r["p"][2] is None and r["pitch"] is None and r["roll"] is None for r in body["photos"])
    assert {r["id"]: r for r in body["photos"]} == internal_rows(client, salt)
    glb = (settings.data_dir / model.ply_path).with_suffix(".glb")
    glb.write_bytes(b"glTF fake")
    rel = glb.relative_to(settings.data_dir).as_posix()
    base = {"glb_path": rel, "registration": {"accepted": True, "reason": None}}

    store.set_model_stylized(model.id, {**base, "snap": SNAP})
    moved = poses(client)
    assert {r["src"] for r in moved["photos"]} == {"sfm"}
    assert {r["id"]: r for r in moved["photos"]} == internal_rows(client, salt, snaps={model.id: SNAP})
    cams = model.meta["cameras"]
    for res_pid, cam in cams.items():
        p, yaw = pp.apply_snap(SNAP, cam["enu"], cam["heading_deg"])
        (row,) = [r for r in moved["photos"] if r["id"] == public_id(res_pid, salt)]
        assert row["p"] == [math.floor(v + 0.5) for v in p] and row["yaw"] == math.floor(yaw + 0.5) % 360
        assert row["sp"] == math.ceil(math.hypot(2 * pp.SNAPPED_SIGMA_CAP, pp.PUBLIC_SP_ROUNDING_M))
    snapped = lambda: detail(client)["models"][0]["stylized"]["snapped"]   # noqa: E731
    assert detail(client)["models"][0]["stylized"]["poses_url"] == f"/api/landmarks/{LM}/poses"
    assert snapped() is True        # the landmark page then draws no markers on this model's (unmoved) point cloud

    assert api_module.stylized_snaps(settings, store.published_models(LM)) == {model.id: SNAP}
    store.set_model_stylized(model.id, {**base, "snap": {**SNAP, "accepted": False}})     # refused: nothing moves
    assert poses(client)["photos"] == body["photos"]
    assert snapped() is False
    assert api_module.stylized_snaps(settings, store.published_models(LM)) == {}
    store.set_model_stylized(model.id, {**base, "snap": None})
    assert poses(client)["photos"] == body["photos"]
    assert snapped() is False
    # a record from before the job runner kept the snap: the GLB's summary beside the PLY
    summary = glb.with_suffix(".style.json")
    summary.write_text(json.dumps({"snap": SNAP, "registration": {"accepted": True}}))
    store.set_model_stylized(model.id, base)
    assert poses(client)["photos"] == moved["photos"]
    assert snapped() is True
    summary.write_text(json.dumps({"snap": {**SNAP, "accepted": False}}))
    assert poses(client)["photos"] == body["photos"]
    assert snapped() is False
    # no GLB recorded: no snap, whatever lies beside the PLY
    summary.write_text(json.dumps({"snap": SNAP}))
    store.set_model_stylized(model.id, None)
    assert poses(client)["photos"] == body["photos"]


def test_landmark_is_detected_from_gps_when_not_given(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    entries = [ds.entries("good", sector=1)[0], ds.entry("far"), ds.entry("no_gps")]
    results = upload(client, ds, entries, landmark_id=None)["results"]
    assert [(r["accepted"], r["landmark_id"], r["reasons"]) for r in results] == [
        (True, LM, []),
        (False, None, ["no_landmark"]),
        (False, None, ["no_gps"]),
    ]
    assert results[0]["sector"] == 1
    assert upload(client, ds, entries[:1], landmark_id="")["results"][0]["reasons"] == ["duplicate"]


def test_upload_to_unknown_landmark_is_404(client, synthetic_dataset):
    ds = synthetic_dataset
    files = [("files", ("a.jpg", ds.read(ds.entries("good")[0]), "image/jpeg"))]
    r = client.post("/api/photos", data={"user_id": "u", "landmark_id": "nope"}, files=files)
    assert r.status_code == 404


@pytest.mark.parametrize("user_id", ["", "   ", "u" * 65, "bell\x07", "two\nlines", "nul\x00byte"])
def test_bad_user_id_is_422(client, user_id):
    r = client.post("/api/photos", data={"user_id": user_id}, files=[("files", ("a.jpg", b"x", "image/jpeg"))])
    assert r.status_code == 422, r.text
    assert r.json()["detail"][0]["loc"] == ["body", "user_id"]


def test_user_id_is_stripped_and_may_be_any_script(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    upload(client, ds, ds.entries("good", sector=0)[:1], user_id="  小明  ")
    upload(client, ds, ds.entries("good", sector=0)[1:2], user_id="u" * 64)
    assert {c["user_id"] for c in detail(client)["contributors"]} == {"小明", "u" * 64}


def test_file_count_limits(client):
    one = [("files", ("a.jpg", b"x", "image/jpeg"))]
    assert client.post("/api/photos", data={"user_id": "u"}, files=one * 51).status_code == 422
    assert client.post("/api/photos", data={"user_id": "u"}).status_code == 422
    r = client.post("/api/photos", data={"user_id": "u"}, files=one * 50)
    assert r.status_code == 200 and len(r.json()["results"]) == 50


def test_too_large_file_is_a_per_file_result(make_client, synthetic_dataset):
    ds = synthetic_dataset
    good = ds.entries("good", sector=0)
    client = make_client(max_upload_bytes=max(len(ds.read(e)) for e in good[:1]))
    create_landmark(client, ds)
    big = b"\xff" * (client.app.state.settings.max_upload_bytes + 1)
    files = [("files", ("big.jpg", big, "image/jpeg")), ("files", (good[0]["file"], ds.read(good[0]), "image/jpeg"))]
    r = client.post("/api/photos", data={"user_id": "u", "landmark_id": LM}, files=files)
    assert r.status_code == 200
    big_res, ok_res = r.json()["results"]
    assert big_res == {"filename": "big.jpg", "photo_id": None, "accepted": False, "reasons": ["too_large"],
                       "warnings": [], "landmark_id": None, "sector": None, "bearing_deg": None, "distance_m": None,
                       "share_location": False, "public_id": None, "manage_token": None}
    assert ok_res["accepted"]


def test_garbage_file_is_rejected_not_an_error(client):
    r = client.post("/api/photos", data={"user_id": "u"}, files=[("files", ("x.jpg", b"garbage", "image/jpeg"))])
    assert r.status_code == 200
    (res,) = r.json()["results"]
    assert (res["accepted"], res["reasons"], res["landmark_id"]) == (False, ["unreadable_image"], None)


# Location consent (D30)


@pytest.mark.parametrize(("value", "shared"), [(None, False), ("", False), ("false", False), ("0", False),
                                               ("off", False), ("on", True), ("true", True), ("1", True)])
def test_share_location_is_off_unless_the_uploader_ticked_it(client, synthetic_dataset, value, shared):
    """The form field (a browser checkbox sends "on") stores the current consent text (LOCATION_CONSENT_V2, the hint that
    states the rebuild risk) only when set; the result echoes it and hands out the manage token, HMAC(salt, "manage:" +
    photo id), only then."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    (res,) = upload(client, ds, ds.entries("good", sector=0)[:1], share_location=value)["results"]
    assert res["accepted"] and res["share_location"] is shared
    photo = client.app.state.store.get_photo(res["photo_id"])
    assert LOCATION_CONSENT_CURRENT == LOCATION_CONSENT_V2
    assert photo.location_consent == (LOCATION_CONSENT_V2 if shared else None)
    salt = deployment_salt(client.app.state.settings.data_dir)
    assert res["manage_token"] == (manage_token(res["photo_id"], salt) if shared else None)
    assert res["public_id"] == (public_id(res["photo_id"], salt) if shared else None)
    if shared:
        assert re.fullmatch(r"[0-9a-f]{64}", res["manage_token"])


def test_share_location_must_be_a_boolean(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    files = [("files", ("a.jpg", ds.read(ds.entries("good")[0]), "image/jpeg"))]
    r = client.post("/api/photos", data={"user_id": "u", "landmark_id": LM, "share_location": "maybe"}, files=files)
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "share_location"]
    assert client.app.state.store.list_photos(LM) == []


def test_a_rejected_photo_with_consent_shares_nothing(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    results = upload(client, ds, [ds.entry("blurry"), ds.entry("far")], share_location="on")["results"]
    for res in results:
        assert not res["accepted"] and res["photo_id"]
        assert (res["share_location"], res["public_id"], res["manage_token"]) == (False, None, None)
        assert client.app.state.store.get_photo(res["photo_id"]).location_consent is None


def withdraw(client, photo_id, token, share=False):
    return client.post(f"/api/photos/{photo_id}/share_location", json={"share": share, "token": token})


def test_the_uploader_withdraws_consent_with_the_manage_token(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    store = client.app.state.store
    first, second = upload(client, ds, ds.entries("good", sector=0)[:2], share_location="true")["results"]
    pid, token = first["photo_id"], first["manage_token"]
    before = store.get_photo(pid)

    # a wrong token (another photo's) or a malformed one changes nothing
    assert withdraw(client, pid, second["manage_token"]).status_code == 403
    for bad in ("", "x" * 64, token[:-1], token.upper(), token + "0"):
        assert withdraw(client, pid, bad).status_code == 422, bad
    # consent is given at upload only: share true is not accepted
    assert withdraw(client, pid, token, share=True).status_code == 422
    assert client.post(f"/api/photos/{pid}/share_location", json={"token": token}).status_code == 422
    assert store.get_photo(pid) == before

    r = withdraw(client, pid, token)
    assert r.status_code == 200 and r.json() == {"photo_id": pid, "share_location": False}
    assert store.get_photo(pid).location_consent is None
    assert withdraw(client, pid, token).status_code == 200                    # again: still withdrawn
    assert store.get_photo(second["photo_id"]).location_consent == LOCATION_CONSENT_V2   # the other one is untouched

    # a token is only valid for its own photo id; an unknown id answers 403 like a wrong token, unless the token was
    # made with this deployment's salt for exactly that id
    salt = deployment_salt(client.app.state.settings.data_dir)
    assert withdraw(client, "0" * 32, token).status_code == 403
    assert withdraw(client, "0" * 32, manage_token("0" * 32, salt)).status_code == 404


def test_the_public_id_salt_is_private_stable_and_per_deployment(settings, tmp_path, monkeypatch, synthetic_dataset):
    """<data_dir>/public_id.salt: made on first start with mode 0600, kept across restarts (a manage token from before
    a restart still works), different in another data dir, never served (D30)."""
    ds = synthetic_dataset
    monkeypatch.delenv(PUBLIC_ID_SALT_ENV, raising=False)
    path = salt_path(settings.data_dir)
    with TestClient(create_app(settings)) as client:
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        salt = path.read_text(encoding="ascii").strip()
        assert re.fullmatch(r"[0-9a-f]{64}", salt)
        create_landmark(client, ds)
        (res,) = upload(client, ds, ds.entries("good", sector=0)[:1], share_location="true")["results"]
        root = settings.models_dir.parent
        assert path.parent == root and settings.models_dir != root and WEB_DIR not in path.parents
        for url in ("/public_id.salt", "/files/models/public_id.salt", "/files/public_id.salt",
                    "/files/models/%2e%2e/public_id.salt", "/%2e%2e/data/public_id.salt"):
            r = client.get(url)
            assert r.status_code == 404 and salt not in r.text, url
    with TestClient(create_app(settings)) as client:                              # a restart keeps the salt
        assert path.read_text(encoding="ascii").strip() == salt
        assert withdraw(client, res["photo_id"], res["manage_token"]).status_code == 200
    other = replace(settings, data_dir=tmp_path / "other")
    with TestClient(create_app(other)) as client:
        assert salt_path(other.data_dir).read_text(encoding="ascii").strip() != salt
        assert manage_token(res["photo_id"], deployment_salt(other.data_dir)) != res["manage_token"]

    os.chmod(path, 0o644)                                                         # readable by others: narrowed
    assert deployment_salt(settings.data_dir) == salt and stat.S_IMODE(path.stat().st_mode) == 0o600
    monkeypatch.setenv(PUBLIC_ID_SALT_ENV, "s" * 32)                              # the environment's salt wins
    assert deployment_salt(settings.data_dir) == "s" * 32
    monkeypatch.setenv(PUBLIC_ID_SALT_ENV, "short")
    with pytest.raises(ValueError, match="at least 32"):
        deployment_salt(settings.data_dir)
    monkeypatch.delenv(PUBLIC_ID_SALT_ENV)
    path.write_text("")                                                           # an emptied salt is not replaced
    with pytest.raises(RuntimeError, match="no usable salt") as e:
        create_app(settings)
    assert "delete it" not in str(e.value) and "forget-lost-salt" in str(e.value)
    assert path.read_text() == ""


def test_another_salt_refuses_to_start_while_a_photo_is_shared(settings, monkeypatch, synthetic_dataset, caplog):
    """The salt is tied to the store it keys (D30): another one would make every manage token useless while the
    photos stay shared. An env salt set or changed, the salt file deleted (or a data dir restored without it) or
    replaced: the server refuses to start, writing no new salt file. With the old salt back, the old tokens
    withdraw. With no photo shared nothing can be lost, and another salt is adopted with a warning."""
    ds = synthetic_dataset
    monkeypatch.delenv(PUBLIC_ID_SALT_ENV, raising=False)
    path = salt_path(settings.data_dir)
    with TestClient(create_app(settings)) as client:
        create_landmark(client, ds)
        first, second = upload(client, ds, ds.entries("good", sector=0)[:2], share_location="true")["results"]
        salt = path.read_text(encoding="ascii").strip()
        recorded = client.app.state.store.get_meta(SALT_CHECK_KEY)
    assert recorded == salt_check(salt) and re.fullmatch(r"[0-9a-f]{64}", recorded) and recorded != salt

    def refused(match):
        with pytest.raises(SaltMismatch, match=match) as e:
            create_app(settings)
        assert "forget-lost-salt" in str(e.value) and salt not in str(e.value)

    monkeypatch.setenv(PUBLIC_ID_SALT_ENV, "e" * 40)                              # an env salt set
    refused(r"the salt in \$CROWD3D_PUBLIC_ID_SALT is not the salt the location consents of 2 photos")
    monkeypatch.setenv(PUBLIC_ID_SALT_ENV, salt)                                  # the same salt from the env is fine
    with TestClient(create_app(settings)):
        pass
    monkeypatch.delenv(PUBLIC_ID_SALT_ENV)
    path.unlink()                                                                 # deleted: no new one is made
    refused(r"no salt \(the file is missing\) is not the salt the location consents of 2 photos")
    assert not path.exists()
    path.write_text("f" * 64 + "\n")                                             # replaced by another
    refused(r"the salt in .*public_id\.salt is not the salt the location consents of 2 photos")
    path.write_text(salt + "\n")                                                 # restored from a backup
    with TestClient(create_app(settings)) as client:
        store = client.app.state.store
        assert store.get_photo(first["photo_id"]).location_consent == LOCATION_CONSENT_V2      # refusals change nothing
        assert withdraw(client, first["photo_id"], first["manage_token"]).status_code == 200
        monkeypatch.setenv(PUBLIC_ID_SALT_ENV, "e" * 40)
        refused(r"consents of 1 photo in")                                        # one is still shared
        monkeypatch.delenv(PUBLIC_ID_SALT_ENV)
        assert withdraw(client, second["photo_id"], second["manage_token"]).status_code == 200
        assert store.get_meta(SALT_CHECK_KEY) == recorded

    monkeypatch.setenv(PUBLIC_ID_SALT_ENV, "e" * 40)
    with caplog.at_level(logging.WARNING, logger="crowd3d.consent"):
        with TestClient(create_app(settings)) as client:
            (res,) = upload(client, ds, ds.entries("good", sector=1)[:1], share_location="true")["results"]
            assert res["manage_token"] == manage_token(res["photo_id"], "e" * 40)
            assert client.app.state.store.get_meta(SALT_CHECK_KEY) == salt_check("e" * 40)
    assert "no photo's location is shared, so it is adopted" in caplog.text
    monkeypatch.delenv(PUBLIC_ID_SALT_ENV)
    refused(r"the salt in .*public_id\.salt is not the salt the location consents of 1 photo in")  # adopted: the env's


def test_a_lost_salt_is_forgotten_only_on_purpose_and_withdraws_every_consent(settings, monkeypatch,
                                                                              synthetic_dataset, capsys):
    """python -m crowd3d.consent forget-lost-salt (D30): refused with no database, while a server holds the data dir,
    or while the salt still matches; else every consent is withdrawn (nobody could withdraw it any more), an
    unusable salt file is replaced, and the server starts again, handing out tokens that work."""
    ds = synthetic_dataset
    monkeypatch.delenv(PUBLIC_ID_SALT_ENV, raising=False)
    path = salt_path(settings.data_dir)
    argv = ["forget-lost-salt", "--data-dir", str(settings.data_dir)]
    assert consent_main(argv) == 1 and "holds no Crowd3D database" in capsys.readouterr().err
    assert not settings.db_path.exists()
    with TestClient(create_app(settings)) as client:
        create_landmark(client, ds)
        results = upload(client, ds, ds.entries("good", sector=0)[:3], share_location="true")["results"]
        assert consent_main(argv) == 1 and "another Crowd3D server" in capsys.readouterr().err
    assert consent_main(argv) == 1 and "nothing to forget" in capsys.readouterr().err
    path.write_text("")                                                           # the salt is lost
    with pytest.raises(RuntimeError, match="no usable salt"):
        create_app(settings)
    assert consent_main(argv) == 0
    assert "withdrew the location consent of 3 photos" in capsys.readouterr().out
    new = path.read_text(encoding="ascii").strip()
    assert re.fullmatch(r"[0-9a-f]{64}", new) and stat.S_IMODE(path.stat().st_mode) == 0o600
    with TestClient(create_app(settings)) as client:
        store = client.app.state.store
        assert store.get_meta(SALT_CHECK_KEY) == salt_check(new)
        for res in results:
            p = store.get_photo(res["photo_id"])
            assert (p.status, p.location_consent) == (PHOTO_ACCEPTED, None)       # kept, no longer shared
            assert withdraw(client, res["photo_id"], res["manage_token"]).status_code == 403
        (res,) = upload(client, ds, ds.entries("good", sector=1)[:1], share_location="true")["results"]
        assert withdraw(client, res["photo_id"], res["manage_token"]).status_code == 200
    assert consent_main(argv) == 1 and "nothing to forget" in capsys.readouterr().err


def test_a_data_dir_inside_the_served_web_dir_is_refused(settings):
    inside = WEB_DIR / "d30-test-data-dir"
    assert not inside.exists()
    with pytest.raises(ValueError, match="served web dir"):
        create_app(replace(settings, data_dir=inside))
    with pytest.raises(ValueError, match="served web dir"):
        create_app(replace(settings, data_dir=WEB_DIR))
    assert not inside.exists()


# Rebuilds and jobs


def test_rebuild(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    assert client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}).status_code == 409
    upload(client, ds, ds.entries("good", sector=0))

    r = client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0})
    assert r.status_code == 202
    job = r.json()
    assert set(job) == JOB_KEYS
    assert (job["kind"], job["sector"], job["status"], job["num_photos"]) == ("face", 0, "done", 4)
    assert client.get(f"/api/jobs/{job['id']}").json() == job
    assert len([m for m in detail(client)["models"] if m["kind"] == "face"]) == 1  # latest per sector only

    r = client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "full"})
    assert r.status_code == 409 and "full model needs" in r.json()["detail"]
    assert client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 1}).status_code == 409


def test_full_rebuild_ignores_a_sector(make_client, synthetic_dataset):
    ds = synthetic_dataset
    client = make_client(**TWO_SECTOR_FULL)
    create_landmark(client, ds)
    upload(client, ds, ds.entries("good"))
    r = client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "full", "sector": 3})
    assert r.status_code == 202 and r.json()["sector"] is None and r.json()["kind"] == "full"


@pytest.mark.parametrize(
    "body",
    [{"kind": "bogus"}, {}, {"kind": "face"}, {"kind": "face", "sector": -1}, {"kind": "face", "sector": 8},
     {"kind": "face", "sector": "north"}, {"kind": "face", "sector": 1.5}],
)
def test_bad_rebuild_request_is_422(client, body):
    create_landmark(client)
    r = client.post(f"/api/landmarks/{LM}/rebuild", json=body)
    assert r.status_code == 422, r.text


def test_rebuild_unknown_landmark_is_404(client):
    assert client.post("/api/landmarks/nope/rebuild", json={"kind": "full"}).status_code == 404


def test_unknown_job_is_404(client):
    assert client.get("/api/jobs/does-not-exist").status_code == 404


def test_background_jobs_run_after_the_response(make_client, synthetic_dataset):
    ds = synthetic_dataset
    client = make_client(inline_jobs=False)
    create_landmark(client, ds)
    (job,) = upload(client, ds, ds.entries("good", sector=0))["triggered_jobs"]
    assert job["status"] in ("queued", "running")
    assert client.app.state.runner.drain(timeout=30)
    done = client.get(f"/api/jobs/{job['id']}").json()
    assert done["status"] == "done" and done["finished_at"] >= done["started_at"]
    assert detail(client)["sectors"][0]["has_face_model"]


# Detail list caps and robustness against odd reconstructor metadata


def test_detail_caps_jobs_newest_first_and_lists_no_photos(client):
    create_landmark(client)
    store = client.app.state.store
    for i in range(25):
        store.create_job(Job(id=f"job-{i:02d}", landmark_id=LM, kind=JOB_FACE, sector=0, status=JOB_DONE,
                             photo_ids=[], created_at=1000.0 + i))
    for i in range(205):
        store.add_photo(PhotoRecord(id=f"photo-{i:03d}", landmark_id=LM, user_id=f"u{i % 3}", status="accepted",
                                    uploaded_at=1000.0 + i, sector=i % 8, bearing_deg=45.0 * (i % 8),
                                    distance_m=30.0))
    d = detail(client)
    assert [j["id"] for j in d["jobs"]] == [f"job-{i:02d}" for i in range(24, 4, -1)]
    assert "photos" not in d and "photo-" not in client.get(f"/api/landmarks/{LM}").text
    assert d["num_photos"] == 205 and d["num_contributors"] == 3
    assert d["contributors"] == [{"user_id": "u0"}, {"user_id": "u1"}, {"user_id": "u2"}]   # no counts (D30)


def test_detail_jobs_list_only_reconstruction_jobs(client):
    """Stylize jobs never enter the detail's jobs window (they would push a facade's failed build out of it)."""
    create_landmark(client)
    store = client.app.state.store
    for i in range(25):
        kind = JOB_FACE if i % 2 == 0 else JOB_STYLIZE
        store.create_job(Job(id=f"job-{i:02d}", landmark_id=LM, kind=kind, sector=0, status=JOB_DONE,
                             photo_ids=[], created_at=1000.0 + i, model_id=None if kind == JOB_FACE else "m"))
    jobs = detail(client)["jobs"]
    assert [j["id"] for j in jobs] == [f"job-{i:02d}" for i in range(24, -1, -2)]
    assert {j["kind"] for j in jobs} == {JOB_FACE}
    assert client.get("/api/jobs/job-01").json()["kind"] == JOB_STYLIZE      # still readable on its own


def test_non_finite_model_metrics_are_reported_as_null(client):
    create_landmark(client)
    store = client.app.state.store
    store.create_job(Job(id="j", landmark_id=LM, kind="full", sector=None, status=JOB_DONE, photo_ids=["p"],
                         created_at=time.time()))
    meta = {"num_registered": 3, "num_points": 10, "georegistered": False, "alignment_rmse_m": float("nan"),
            "heading_mae_deg": float("inf"), "cameras": {"p": {"enu": [float("nan"), 0, 0], "heading_deg": None}}}
    store.add_model(ModelRecord(id="m", landmark_id=LM, job_id="j", kind="full", sector=None,
                                ply_path=f"models/{LM}/m.ply", num_photos=1, meta=meta, created_at=time.time()))
    full = detail(client)["latest_full_model"]
    assert (full["alignment_rmse_m"], full["heading_mae_deg"], full["georegistered"]) == (None, None, False)
    assert client.get("/api/landmarks").status_code == 200


def test_model_geo_reports_the_anchor_with_the_current_floors(settings):
    """ModelInfo.geo (additive, informational): the GeoAnchor with the current calibration floors, the 1-sigma at the
    points' median, and calibrated only for a calibration listed in Settings.calibrated_anchor_ids (none by default);
    None for a model recorded before anchors and for a malformed record."""
    from crowd3d.anchor import calibration
    from test_anchor import make_anchor

    old = make_anchor()                                # made with other floors ("test")
    meta = {"georegistered": True, "points_median_en": [20.0, 5.0], "anchor": old.to_meta()}
    rec = ModelRecord(id="m", landmark_id=LM, job_id="j", kind="face", sector=0, ply_path=f"models/{LM}/m.ply",
                      num_photos=5, meta=meta, created_at=0.0)
    geo = api_module.model_info(settings, rec)["geo"]
    now = old.recalibrated()
    assert now.calibration == calibration() and now.sigma != old.sigma
    h, v = now.position_sigma_m([20.0, 5.0])
    assert geo == {"level": "full", "sigma": now.sigma, "source": {"yaw": "gps+compass"},
                   "calibration_id": calibration()["id"],
                   "position_sigma_m": {"horizontal": round(h, 2), "vertical": round(v, 2)}, "calibrated": False}
    assert h > now.position_sigma_m(now.pivot_enu)[0]   # the walls, not the cameras' centre
    on = replace(settings, calibrated_anchor_ids=(calibration()["id"],))
    assert api_module.model_info(on, rec)["geo"]["calibrated"] is True
    stale = replace(settings, calibrated_anchor_ids=("test",))   # only the calibration the sigmas are shown with
    assert api_module.model_info(stale, rec)["geo"]["calibrated"] is False
    for legacy in ({"georegistered": True}, {"anchor": {"schema": "crowd3d.anchor/1"}}, {"anchor": {"schema": "x"}}):
        assert api_module.model_info(settings, replace(rec, meta=legacy))["geo"] is None


# CLI


def test_main_seeds_the_demo_landmark_and_runs_uvicorn(tmp_path, monkeypatch):
    calls = []

    def fake_run(app, host, port):
        calls.append((app, host, port))
        app.state.store.close()

    monkeypatch.setattr(api_module.uvicorn, "run", fake_run)
    argv = ["--data-dir", str(tmp_path / "d"), "--reconstructor", "fake", "--seed-demo", "--port", "9123"]
    assert api_module.main(argv) == 0
    assert api_module.main(argv) == 0  # seeding twice is a no-op

    (app, host, port), _ = calls
    assert (host, port) == ("127.0.0.1", 9123)
    assert app.state.settings.data_dir == tmp_path / "d"
    assert app.state.runner.reconstructor.name == "fake"
    store = Store(app.state.settings)
    try:
        (lm,) = store.list_landmarks()
        assert (lm.id, lm.name, lm.lat, lm.lon, lm.radius_m) == ("demo-building", DEMO_NAME, DEMO_LAT, DEMO_LON, 150.0)
    finally:
        store.close()


# Robustness of the server as a whole


def test_one_failing_file_does_not_lose_the_others_or_their_jobs(make_client, synthetic_dataset, monkeypatch):
    # A batch whose 5th file hit ENOSPC returned 500: files 1-4 were stored as accepted, never planned,
    # and every client retry was rejected as a duplicate of itself.
    ds = synthetic_dataset
    client = make_client()
    create_landmark(client, ds)
    real_ingest = api_module.ingest_upload

    def ingest(store, settings, *, filename, **kwargs):
        if filename == "boom.jpg":
            raise OSError(28, "No space left on device")
        return real_ingest(store, settings, filename=filename, **kwargs)

    monkeypatch.setattr(api_module, "ingest_upload", ingest)
    entries = ds.entries("good", sector=0)
    files = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in entries]
    files.insert(2, ("files", ("boom.jpg", ds.read(entries[0]), "image/jpeg")))
    r = client.post("/api/photos", data={"user_id": "u", "landmark_id": LM}, files=files)
    assert r.status_code == 200
    results = r.json()["results"]
    assert [res["reasons"] for res in results] == [[], [], ["server_error"], [], []]
    (job,) = r.json()["triggered_jobs"]
    assert (job["kind"], job["sector"], job["num_photos"], job["status"]) == ("face", 0, 4, "done")


def test_model_sidecars_with_camera_poses_are_not_served(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    upload(client, ds, ds.entries("good", sector=0))
    (model,) = detail(client)["models"]
    assert client.get(model["url"]).status_code == 200
    assert (client.app.state.settings.models_dir / LM / f"{model['id']}.json").is_file()
    assert client.get(model["url"].replace(".ply", ".json")).status_code == 404


def test_model_files_serve_only_point_clouds_and_glbs(client):
    """ModelFiles serves .ply and .glb files under models_dir and nothing else: not the .json / .style.json sidecars
    (per-photo camera poses keyed by store photo id), not any other suffix, not a directory, and no path out of
    models_dir even when it ends in .ply (D30)."""
    settings = client.app.state.settings
    lm_dir = settings.models_dir / LM
    lm_dir.mkdir(parents=True)
    served = {"m.ply": b"ply\n", "m.glb": b"glTF"}
    private = ["m.json", "m.style.json", "m.ply.json", "m.glb.bak", "m.PLY", "m.sqlite3", "m.txt", "m.bin",
               "public_id.salt", "cameras"]
    for name, data in served.items():
        (lm_dir / name).write_bytes(data)
    for name in private:
        (lm_dir / name).write_text('{"cameras": {"p": {"enu": [1, 2, 3]}}}')
    (lm_dir / "dir.ply").mkdir()
    (settings.work_dir / "j").mkdir(parents=True)
    (settings.work_dir / "j" / "x.ply").write_bytes(b"ply\n")
    (settings.maps_dir / LM).mkdir(parents=True)
    (settings.maps_dir / LM / "x.ply").write_bytes(b"ply\n")
    for name, data in served.items():
        r = client.get(f"/files/models/{LM}/{name}")
        assert r.status_code == 200 and r.content == data, name
    for name in private + ["dir.ply", "nope.ply"]:
        assert client.get(f"/files/models/{LM}/{name}").status_code == 404, name
        assert client.head(f"/files/models/{LM}/{name}").status_code == 404, name
    for path in ("/files/models/%2e%2e/work/j/x.ply", f"/files/models/{LM}/%2e%2e/%2e%2e/maps/{LM}/x.ply",
                 f"/files/models/{LM}/..%2f..%2fwork/j/x.ply", "/files/models/%2e%2e/crowd3d.sqlite3",
                 "/files/work/j/x.ply", f"/files/maps/{LM}/x.ply", "/files/crowd3d.sqlite3", "/files/models/"):
        r = client.get(path)
        assert r.status_code == 404 and r.content != b"ply\n", path


def test_stylized_models_are_served_beside_the_point_clouds_and_listed_with_their_label(client, synthetic_dataset):
    """ModelFiles serves .ply and .glb only; model_info.stylized: pending while a stylize job is queued or running,
    then the GLB's URL with the registration and its label (the model is clay either way, D29; the label says whether
    the photos' registration was accepted)."""
    from crowd3d.api import STYLIZED_LABEL_ALIGNED, STYLIZED_LABEL_UNALIGNED
    from crowd3d.records import JOB_QUEUED, JOB_STYLIZE, Job

    ds = synthetic_dataset
    create_landmark(client, ds)
    upload(client, ds, ds.entries("good", sector=0))
    (model,) = detail(client)["models"]
    assert model["stylized"] is None                    # no snapshot for this landmark: none made, none coming
    store, settings = client.app.state.store, client.app.state.settings
    stored = store.get_model(model["id"])
    store.create_job(Job(id="styl", landmark_id=LM, kind=JOB_STYLIZE, sector=0, status=JOB_QUEUED,
                         photo_ids=[], created_at=time.time(), model_id=model["id"]))
    d = detail(client)
    assert d["models"][0]["stylized"] == {"status": "pending", "url": None, "registration": None, "label": None,
                                          "poses_url": None, "snapped": None}
    assert all(j["kind"] != JOB_STYLIZE for j in d["jobs"])      # reconstruction jobs only in the window
    assert client.get("/api/jobs/styl").json()["model_id"] == model["id"]
    glb = (settings.data_dir / stored.ply_path).with_suffix(".glb")
    glb.write_bytes(b"glTF fake")
    (settings.data_dir / stored.ply_path).with_suffix(".style.json").write_text("{}")
    rel = glb.relative_to(settings.data_dir).as_posix()
    store.set_model_stylized(model["id"], {"glb_path": rel, "registration": {"accepted": False, "reason": "ambiguous"}})
    refused = detail(client)["models"][0]["stylized"]
    assert refused == {"status": "ready", "url": model["url"].replace(".ply", ".glb"),
                       "registration": {"accepted": False, "reason": "ambiguous"}, "label": STYLIZED_LABEL_UNALIGNED,
                       "poses_url": f"/api/landmarks/{LM}/poses",    # the public photo poses, in the GLB's frame
                       "snapped": False}                             # no snap recorded: the cameras did not move
    assert STYLIZED_LABEL_UNALIGNED == "素模（照片未对齐）"
    store.set_model_stylized(model["id"], {"glb_path": rel, "registration": {"accepted": True, "reason": None}})
    assert detail(client)["models"][0]["stylized"]["label"] == STYLIZED_LABEL_ALIGNED == "素模（照片已对齐）"
    r = client.get(refused["url"])
    assert r.status_code == 200 and r.content == b"glTF fake"
    for sidecar in (".json", ".style.json"):            # camera poses and the summary stay private
        assert client.get(model["url"].replace(".ply", sidecar)).status_code == 404
    (settings.models_dir / LM / "x.sparse").mkdir()
    assert client.get(f"/files/models/{LM}/x.sparse").status_code == 404


def test_request_bodies_are_capped_before_they_are_parsed(make_client):
    client = make_client(max_upload_bytes=1000)      # cap: 50 files x 1000 bytes + 1 MiB
    big = b"\xff" * (1024 * 1024 + 60_000)
    r = client.post("/api/photos", data={"user_id": "u"}, files=[("files", ("big.jpg", big, "image/jpeg"))])
    assert r.status_code == 413

    def chunks():                                    # no Content-Length: counted as it streams in
        yield b'--b\r\nContent-Disposition: form-data; name="files"; filename="a.jpg"\r\n\r\n'
        for _ in range(30):
            yield b"x" * 65536

    r = client.post("/api/photos", content=chunks(), headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert client.get("/api/health").status_code == 200


def test_a_second_server_on_the_same_data_dir_refuses_to_start(make_client, settings):
    client = make_client()
    store = client.app.state.store
    store.create_landmark(api_module.Landmark(id=LM, name="x", lat=1.0, lon=2.0, radius_m=50.0, created_at=0.0))
    store.create_job(Job(id="busy", landmark_id=LM, kind=JOB_FACE, sector=0, status="running", photo_ids=[],
                         created_at=1.0, started_at=1.0))
    with pytest.raises(api_module.DataDirInUse), TestClient(create_app(settings)):
        pass
    assert store.get_job("busy").status == "running"     # not reclaimed from the first server


def test_thresholds_report_the_sides_the_full_model_needs_here(client):
    # 60 photos on three sides only (a building seen from one square): the full model needs those three.
    create_landmark(client)
    store = client.app.state.store
    for i in range(60):
        sector = 1 + i % 3
        store.add_photo(PhotoRecord(id=f"p{i:02d}", landmark_id=LM, user_id="u", status="accepted",
                                    uploaded_at=1000.0 + i, sector=sector, bearing_deg=45.0 * sector, distance_m=40.0))
    d = detail(client)
    assert d["thresholds"]["full_min_sectors"] == 3 and d["full_ready"]
