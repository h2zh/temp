import json
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
from crowd3d.consent import PUBLIC_ID_SALT_ENV, deployment_salt, manage_token, salt_path
from crowd3d.reconstruct import FakeReconstructor
from crowd3d.records import (
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_STYLIZE,
    LOCATION_CONSENT_V1,
    PHOTO_ACCEPTED,
    Job,
    ModelRecord,
    PhotoRecord,
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
               "distance_m", "share_location", "manage_token"}
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
    """D30, closing the live leak: no unauthenticated GET serves a store photo id, a server path, an uploader beside a
    photo, a photo's position (GPS, bearing, distance), heading or upload time, or a camera's pose, in any field or
    string (a failed job's error included). The contributor list (nickname and count) and the sector counts are
    aggregates."""
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
    docs = {"/api/landmarks": client.get("/api/landmarks"), f"/api/landmarks/{LM}": client.get(f"/api/landmarks/{LM}")}
    for j in store.list_jobs(LM):
        docs[f"/api/jobs/{j.id}"] = client.get(f"/api/jobs/{j.id}")
    assert len(docs) >= 6 and all(r.status_code == 200 for r in docs.values())
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
        assert str(settings.data_dir) not in text and "/data/" not in text, url
        for key in ("estimated_enu", "estimated_heading_deg", "bearing_deg", "distance_m", "uploaded_at", "photo_ids"):
            assert f'"{key}"' not in text, (url, key)
        nums = _numbers_in_text(text)
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
        body = r.json()
        rest = json.dumps({k: v for k, v in body.items() if k != "contributors"}, ensure_ascii=False)
        for u in users:
            assert u not in rest, (url, u)
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
                       "share_location": False, "manage_token": None}
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
    """The form field (a browser checkbox sends "on") stores LOCATION_CONSENT_V1 only when set; the result echoes it
    and hands out the manage token, HMAC(salt, "manage:" + photo id), only then."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    (res,) = upload(client, ds, ds.entries("good", sector=0)[:1], share_location=value)["results"]
    assert res["accepted"] and res["share_location"] is shared
    photo = client.app.state.store.get_photo(res["photo_id"])
    assert photo.location_consent == (LOCATION_CONSENT_V1 if shared else None)
    salt = deployment_salt(client.app.state.settings.data_dir)
    assert res["manage_token"] == (manage_token(res["photo_id"], salt) if shared else None)
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
        assert (res["share_location"], res["manage_token"]) == (False, None)
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
    assert store.get_photo(second["photo_id"]).location_consent == LOCATION_CONSENT_V1   # the other one is untouched

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
    with pytest.raises(RuntimeError, match="no usable salt"):
        create_app(settings)
    assert path.read_text() == ""


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
    assert d["models"][0]["stylized"] == {"status": "pending", "url": None, "registration": None, "label": None}
    assert all(j["kind"] != JOB_STYLIZE for j in d["jobs"])      # reconstruction jobs only in the window
    assert client.get("/api/jobs/styl").json()["model_id"] == model["id"]
    glb = (settings.data_dir / stored.ply_path).with_suffix(".glb")
    glb.write_bytes(b"glTF fake")
    (settings.data_dir / stored.ply_path).with_suffix(".style.json").write_text("{}")
    rel = glb.relative_to(settings.data_dir).as_posix()
    store.set_model_stylized(model["id"], {"glb_path": rel, "registration": {"accepted": False, "reason": "ambiguous"}})
    refused = detail(client)["models"][0]["stylized"]
    assert refused == {"status": "ready", "url": model["url"].replace(".ply", ".glb"),
                    "registration": {"accepted": False, "reason": "ambiguous"}, "label": STYLIZED_LABEL_UNALIGNED}
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
