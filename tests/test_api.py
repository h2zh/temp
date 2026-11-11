import http.client
import io
import json
import logging
import math
import os
import platform
import re
import socket
import sqlite3
import stat
import subprocess
import sys
import threading
import time
import zlib
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

import anyio
import numpy as np
import piexif
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from crowd3d import api as api_module
from crowd3d import contributors, csp
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
    PHOTO_ACCEPTED,
    TERMS_CURRENT,
    Job,
    ModelRecord,
    PhotoRecord,
    location_shared,
)
from crowd3d.store import Store
from crowd3d.synthetic import CAMERA_RADIUS_M, DEMO_LAT, DEMO_LON, DEMO_NAME
from jpeg_meta import assert_metadata_free   # (pytest puts tests/ on the path)

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


def register(client, nickname="alice", terms=TERMS_CURRENT):
    """POST /api/contributors (D33): register, agreeing to the user agreement version terms (sent unless None)."""
    return client.post("/api/contributors", json={"nickname": nickname} | ({} if terms is None else {"terms": terms}))


def contributor_token(client, nickname="alice"):
    """The contributor token of nickname on this client's server, registered at its first use (D33)."""
    tokens = vars(client).setdefault("contributor_tokens", {})
    if nickname not in tokens:
        r = register(client, nickname)
        assert r.status_code == 201, r.text
        tokens[nickname] = r.json()["contributor_token"]
    return tokens[nickname]


def form(client, user_id="u", **data):
    """An upload's form fields (D33): the contributor token of user_id (registered once per client) and data."""
    return {"contributor_token": contributor_token(client, user_id), **data}


def upload(client, ds, entries, *, user_id="alice", landmark_id=LM):
    """user_id: the nickname of the registered contributor uploading (once per client and nickname, D33)."""
    files = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in entries]
    data = form(client, user_id) | ({"landmark_id": landmark_id} if landmark_id else {})
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


def test_an_input_a_422_cannot_render_is_not_echoed_so_no_json_route_answers_500(client, caplog):
    """A 422 echoes each error's input, unless the input cannot be rendered as the response's JSON
    (api._rendered_error): a lone UTF-16 surrogate (JSON's \\ud800 escape) anywhere in it, keys too, does not encode
    to UTF-8, and an input nested 5,000 deep is too deep to encode. Each made rendering the 422 fail, a 500 with an
    ERROR traceback, on every route whose JSON body FastAPI parses (D33 code review). Now it is a 422 with that input
    null, the rest of the error as before, and nothing is logged as an error; an input that renders is still echoed."""
    create_landmark(client)
    as_json = {"content-type": "application/json"}
    deep = "[" * 5000 + '"x"' + "]" * 5000
    rebuild, share = f"/api/landmarks/{LM}/rebuild", "/api/photos/p/share_location"
    caplog.set_level(logging.DEBUG)
    for path, body, error in [
        ("/api/landmarks", '{"name": "a\\ud800", "lat": 1, "lon": 2}', ("string_unicode", "name")),
        ("/api/landmarks", '{"name": "\\ud800", "lat": 1, "lon": 2}', ("string_unicode", "name")),
        ("/api/landmarks", '{"id": "\\ud800", "name": "x", "lat": 1, "lon": 2}', ("string_unicode", "id")),
        ("/api/landmarks", '{"name": "x", "lat": "\\ud800", "lon": 2}', ("string_unicode", "lat")),
        ("/api/landmarks", '{"name": ["\\ud800"], "lat": 1, "lon": 2}', ("string_type", "name")),
        ("/api/landmarks", '{"\\ud800": 1, "lat": 1, "lon": 2}', ("missing", "name")),       # in a key of the input
        ("/api/landmarks", '{"\\ud800": 1}', ("missing", "name")),
        ("/api/landmarks", '"\\ud800"', ("model_attributes_type", None)),
        ("/api/landmarks", f'{{"name": {deep}, "lat": 1, "lon": 2}}', ("string_type", "name")),
        ("/api/landmarks", deep, ("model_attributes_type", None)),
        (rebuild, '{"kind": "\\ud800"}', ("string_unicode", "kind")),
        (rebuild, '{"kind": "face", "sector": "\\ud800"}', ("string_unicode", "sector")),
        (rebuild, f'{{"kind": {deep}}}', ("literal_error", "kind")),
        (share, '{"share": false, "token": "\\ud800"}', ("string_unicode", "token")),
        (share, '{"share": "\\ud800", "token": "x"}', ("literal_error", "share")),
        (share, '{"share": {"\\ud800": false}, "token": "x"}', ("literal_error", "share")),
        (share, f'{{"share": false, "token": {deep}}}', ("string_type", "token")),
    ]:
        r = client.post(path, content=body, headers=as_json)
        assert r.status_code == 422, (path, body[:60], r.text)
        found = [e for e in r.json()["detail"] if (e["type"], e["loc"][1] if len(e["loc"]) > 1 else None) == error]
        assert len(found) == 1 and found[0]["input"] is None, (path, body[:60], r.text)
        assert set(found[0]) <= {"type", "loc", "msg", "input", "ctx"}
    assert not [rec for rec in caplog.records if rec.levelno >= logging.ERROR], caplog.text
    assert [lm["id"] for lm in client.get("/api/landmarks").json()["landmarks"]] == [LM]          # none created
    # an input that renders is echoed as before, also beside one that does not
    r = client.post(rebuild, json={"kind": "bogus"})
    assert r.status_code == 422 and r.json()["detail"][0]["input"] == "bogus"
    r = client.post("/api/landmarks", content='{"name": "\\ud800", "lat": 1, "lon": "east"}', headers=as_json)
    assert {e["loc"][1]: e["input"] for e in r.json()["detail"]} == {"name": None, "lon": "east"}, r.text
    r = client.post("/api/landmarks", json={"name": "\U0001f600" * 101, "lat": 1, "lon": 2})       # a pair is text
    assert r.status_code == 422 and r.json()["detail"][0]["input"] == "\U0001f600" * 101


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


def photo_url(row_id, landmark_id=LM):
    """GET .../poses/{public_id}/photo: the photo of one public poses row (D32)."""
    return f"/api/landmarks/{landmark_id}/poses/{row_id}/photo"


def assert_no_such_photo(r):
    """The one 404 of the photo route, whatever the reason: same body, no ETag, no time."""
    assert r.status_code == 404 and r.json() == {"detail": "no such photo"}, (r.status_code, r.headers)
    assert not {"etag", "last-modified", "cache-control"} & set(r.headers), dict(r.headers)


def assert_public_photo(r, row, secrets=()):
    """The photo of one public poses row as served (D32): a JPEG with the route's headers (no time), no metadata nor
    any of the secrets (jpeg_meta.assert_metadata_free), longer side at most 1600 px, the row's aspect. Its pixels."""
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg", (row["id"], r.status_code)
    assert (r.headers["cache-control"], r.headers["etag"]) == ("no-cache", f'W/"{row["id"]}.1"')
    assert r.headers["x-content-type-options"] == "nosniff"
    assert not {"last-modified", "content-disposition", "set-cookie"} & set(r.headers)
    assert_metadata_free(r.content, secrets)
    with Image.open(io.BytesIO(r.content)) as im:
        assert max(im.size) <= 1600 and abs(im.width / im.height - row["aspect"]) <= 0.01, (im.size, row)
        return np.asarray(im.convert("RGB"), float)


def mean_pixel_difference(path, pixels):
    """Mean absolute difference (0-255) of the image at path, resized to the pixels' size, to the pixels."""
    with Image.open(path) as im:
        return np.abs(np.asarray(im.convert("RGB").resize((pixels.shape[1], pixels.shape[0])), float) - pixels).mean()


def test_no_public_get_serves_per_photo_data(make_client, synthetic_dataset, settings):
    """D30, D32: no unauthenticated GET (the list, the detail, the public poses, each job, the served model files, the
    rows' photos) serves a store photo id, a server path, the salt, an uploader beside a photo, a photo's position
    (GPS, bearing, distance), heading or upload time, or a camera's pose, in any field or string (a failed job's error
    included), except the public poses of the shared photos (uploaded under the user agreement and not withdrawn):
    exactly their rounded public rows, and beside each row its own photo, downscaled and without any metadata. A
    withdrawn photo is never in them, nor is its photo served. The contributor list (nicknames) and the sector counts
    are aggregates. Nothing of a registration (D33: the contributor token, its hash, the contributor id) is in any of
    them, the served point clouds or the photos, and no GET serves contributors. A registration's list of its own
    photos (D34: store ids and manage tokens) is no GET either, and holds that registration's photos only."""
    ds = synthetic_dataset
    client = make_client(MissingPhotoFirstReconstructor(settings), **TWO_SECTOR_FULL)
    create_landmark(client, ds)
    alice = upload(client, ds, ds.entries("good", sector=0), user_id="alice")["results"]
    bob = upload(client, ds, ds.entries("good", sector=1), user_id="bob")["results"]
    tokens = [res["manage_token"] for res in alice + bob]
    assert len(tokens) == 8 and all(tokens)
    ctokens = list(vars(client)["contributor_tokens"].values())
    registered = [contributors.authenticate(client.app.state.store, t) for t in ctokens]
    registration = [*ctokens, *(c.token_hash for c in registered), *(c.id for c in registered)]
    assert len(registration) == 6 and client.get("/api/contributors").status_code != 200
    # D34: each registration's own list (a POST with its token) holds its photos and nothing of the other's
    for token, mine, others in zip(ctokens, (alice, bob), (bob, alice)):
        own = client.post("/api/contributors/photos", json={"contributor_token": token})
        assert [p["photo_id"] for p in own.json()["photos"]] == [res["photo_id"] for res in mine]
        assert not any(res[k] in own.text for res in others for k in ("photo_id", "public_id", "manage_token"))
        assert not any(x in own.text for x in registration)
    for url in ("/api/contributors/photos", f"/api/contributors/photos?contributor_token={ctokens[0]}"):
        for method in ("GET", "HEAD"):
            r = client.request(method, url)
            assert r.status_code in (404, 405) and not any(t in r.text for t in tokens), (method, url)
    # bob withdraws one photo: it takes the role of a photo that is not shared
    withdrawn = bob[0]
    assert withdraw(client, withdrawn["photo_id"], withdrawn["manage_token"]).status_code == 200
    shared = alice + bob[1:]
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
    # the public poses: the seven shared photos, as their public rows exactly; nothing of the withdrawn one
    consented = {res["photo_id"] for res in shared}
    assert consented == set(photos) - {withdrawn["photo_id"]}
    poses = docs[poses_path].json()
    internal = {p.photo_id: p for p in pp.photo_poses(photos.values(), store.published_models(LM), ds.lat, ds.lon,
                                                      aspects=pp.processed_aspects(photos.values(), settings.data_dir))}
    assert poses["photos"] == sorted((pp.public_row(internal[pid], secret) for pid in consented),
                                     key=lambda row: row["id"])
    assert {row["id"] for row in poses["photos"]} == {res["public_id"] for res in shared}
    assert withdrawn["public_id"] not in docs[poses_path].text
    assert poses["counts"]["shared"] == 7 and poses["counts"]["photos"] == 8
    assert (poses["rounding"], poses["uncertainty"]) == (pp.public_rounding(), pp.public_uncertainty())
    assert poses["frame"]["origin"] == {"lat": ds.lat, "lon": ds.lon} and set(poses) == POSES_KEYS
    for row in poses["photos"]:
        assert set(row) == set(pp.PUBLIC_KEYS) and all(isinstance(v, int) for v in row["p"][:2] + [row["yaw"]])
    for pid in set(photos) - consented:                 # no row stands where the withdrawn photo's camera stood
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
        assert not any(x in text for x in registration), url                           # D33: registrations
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
        assert not any(x.encode() in r.content for x in registration), m["url"]
    d = docs[f"/api/landmarks/{LM}"].json()
    # nicknames only, by name: per-person counts would tie public poses to nicknames (D30, owner question 2)
    assert "photos" not in d and d["contributors"] == [{"user_id": "alice"}, {"user_id": "bob"}]
    assert not any("photos" in m for m in d["models"]) and not any("photos" in j for j in d["jobs"])
    # D32: each row's photo, at its public id: no metadata, nothing of anyone (no store id, salt, token, nickname,
    # server path or file name), no time (no Last-Modified); the pixels are its own photo's and no other row's
    secrets = (*(pid.encode() for pid in photos), secret.encode(), *(t.encode() for t in tokens),
               *(x.encode() for x in registration),
               *(u.encode() for u in users), str(settings.data_dir).encode(), b"/data/",
               *(e["file"].encode() for e in ds.entries()))
    served = {row["id"]: assert_public_photo(client.get(photo_url(row["id"])), row, secrets) for row in poses["photos"]}
    assert set(served) == {res["public_id"] for res in shared}
    for rid, pixels in served.items():
        diffs = {res["public_id"]: mean_pixel_difference(settings.data_dir / photos[res["photo_id"]].processed_path,
                                                         pixels) for res in shared}
        assert diffs.pop(rid) < 3.0 and min(diffs.values()) > 3.0, (rid, diffs)
    # the withdrawn photo's, never
    assert_no_such_photo(client.get(photo_url(withdrawn["public_id"])))


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


def test_public_poses_show_shared_photos_only_as_their_rounded_public_rows(make_client, synthetic_dataset,
                                                                            settings):
    """D30, D32: GET /api/landmarks/{id}/poses lists the accepted photos uploaded under the user agreement and not
    withdrawn, as public_row of their internal pose (whole metres and degrees, the salted public id the upload
    returned), sorted by id; a withdrawn photo (NULL) or one whose consent is no agreement version (a D30
    share-location text, which the D32 migration replaces on open) is not in it, and counts say how many are shared
    and how many could be placed."""
    ds = synthetic_dataset
    client = make_client(**TWO_SECTOR_FULL)
    create_landmark(client, ds)
    assert client.get("/api/landmarks/nowhere/poses").status_code == 404
    empty = poses(client)
    assert (empty["as_of"], empty["photos"]) == (None, [])
    assert empty["counts"] == dict.fromkeys(COUNT_KEYS, 0)
    alice = upload(client, ds, ds.entries("good", sector=0), user_id="alice")["results"]
    bob = upload(client, ds, ds.entries("good", sector=1), user_id="bob")["results"]
    store, salt = client.app.state.store, deployment_salt(settings.data_dir)
    assert all(store.get_photo(res["photo_id"]).location_consent == TERMS_CURRENT for res in alice + bob)
    # bob withdraws one; another row holds a D30 consent text, which is no agreement version: neither is shared
    assert withdraw(client, bob[0]["photo_id"], bob[0]["manage_token"]).status_code == 200
    with store._lock:
        store._conn.execute("UPDATE photos SET location_consent = 'share-location-v1' WHERE id = ?",
                            (bob[1]["photo_id"],))
    body = poses(client)
    shown = alice + bob[2:]
    ids = {res["public_id"] for res in shown}
    assert all(re.fullmatch(r"[0-9a-f]{12}", i) for i in ids) and len(ids) == 6
    assert ids == {public_id(res["photo_id"], salt) for res in shown} == {row["id"] for row in body["photos"]}
    assert {row["id"]: row for row in body["photos"]} == internal_rows(client, salt)
    assert body["counts"] == {"photos": 8, "shared": 6, "shown": 6, "sfm": 6, "loc": 0, "exif": 0,
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
    reconstruction job (done or failed), so an upload alone changes nothing a visitor can diff, as the agreement says
    (web/terms.html); consent is read when served, so a withdrawn photo leaves at once."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    store, salt = client.app.state.store, deployment_salt(settings.data_dir)
    first = upload(client, ds, ds.entries("good", sector=0)[:2])
    assert first["triggered_jobs"] == []
    assert poses(client)["photos"] == [] and poses(client)["counts"]["shared"] == 0       # no build finished yet
    second = upload(client, ds, ds.entries("good", sector=0)[2:])
    (job,) = second["triggered_jobs"]
    body = poses(client)
    assert body["as_of"] == store.get_job(job["id"]).finished_at and body["counts"]["shown"] == 4
    four = {row["id"] for row in body["photos"]}
    # one more shared photo, no build: not shown yet, not even counted
    (late,) = upload(client, ds, ds.entries("good", sector=1)[:1])["results"]
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


def test_anyone_can_move_the_poses_snapshot_soon_after_a_new_upload_as_the_agreement_says(
        make_client, synthetic_dataset, settings):
    """The remaining linking risk the user agreement states (TERMS_CURRENT: web/terms.html, section risks, and the
    upload hint web/landmark.html agree-terms-hint, pinned in web/tests/pages.test.mjs): the snapshot holds a new row
    back only until the next reconstruction job ends, and anyone can make one end soon, by an anonymous rebuild or by
    a few uploads of their own (enough to start a facade job: face_min_photos on a new facade, face_rebuild_growth on
    a built one), whose public ids they know from their own upload results. Right after a new nickname appears among
    the contributors, the other row that appears next can be tied to it. If either route stops moving the snapshot,
    the agreement can change with it (under a new version)."""
    ds = synthetic_dataset
    client = make_client(face_min_photos=3, face_rebuild_growth=2, full_min_photos=100)
    create_landmark(client, ds)
    salt = deployment_salt(settings.data_dir)
    good0, good1 = ds.entries("good", sector=0), ds.entries("good", sector=1)
    assert upload(client, ds, good0[:3], user_id="alice")["triggered_jobs"]
    before = poses(client)
    assert before["counts"]["shown"] == 3

    def contributors():
        return {c["user_id"] for c in client.get(f"/api/landmarks/{LM}").json()["contributors"]}

    def new_rows(prev):
        now = poses(client)
        return now, {r["id"] for r in now["photos"]} - {r["id"] for r in prev["photos"]}

    # carol uploads one photo: her nickname shows at once, her row does not
    names = contributors()
    uploaded = upload(client, ds, good0[3:], user_id="carol")
    (carol,) = uploaded["results"]
    assert carol["share_location"] and uploaded["triggered_jobs"] == [] and poses(client) == before
    assert contributors() - names == {"carol"}
    # route 1: an anonymous rebuild ends a job, and the one new row is hers
    assert client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}).status_code == 202
    after, added = new_rows(before)
    assert after["as_of"] > before["as_of"] and added == {carol["public_id"]}
    # route 2: dan uploads one photo of another facade; someone else's own uploads there bring it to face_min_photos
    # and start a job, and they know their own rows' public ids
    uploaded = upload(client, ds, good1[:1], user_id="dan")
    (dan,) = uploaded["results"]
    assert dan["share_location"] and uploaded["triggered_jobs"] == [] and poses(client) == after
    mallory = upload(client, ds, good1[1:3], user_id="mallory")
    assert mallory["triggered_jobs"], "a few uploads start a job"
    final, added = new_rows(after)
    own = {r["public_id"] for r in mallory["results"]}
    assert final["as_of"] > after["as_of"] and added - own == {dan["public_id"]}
    assert {r["id"]: r for r in final["photos"]} == internal_rows(client, salt)


def with_heading_ref(data, ref):
    """The upload's bytes with its compass heading's north (EXIF GPSImgDirectionRef) set to ref."""
    exif = piexif.load(data)
    exif["GPS"][piexif.GPSIFD.GPSImgDirectionRef] = ref
    out = io.BytesIO()
    piexif.insert(piexif.dump(exif), data, out)
    return out.getvalue()


def stored_sideways(data):
    """The upload's bytes as many cameras store a photo: its pixels a quarter turn to the left, tagged EXIF Orientation
    6 (turn right to show it), its other EXIF (GPS, compass, focal length) kept. Shown upright, the same photo."""
    exif = piexif.load(data)
    exif["0th"][piexif.ImageIFD.Orientation] = 6
    with Image.open(io.BytesIO(data)) as im:
        turned = im.transpose(Image.Transpose.ROTATE_90)
    out = io.BytesIO()
    turned.save(out, "JPEG", quality=95, exif=piexif.dump(exif))
    return out.getvalue()


def test_a_photo_is_served_only_as_a_row_of_the_public_poses_and_one_404_answers_everything_else(
        make_client, synthetic_dataset, settings):
    """D32: GET /api/landmarks/{id}/poses/{public_id}/photo serves the photo of a row of that landmark's public poses
    now, whatever placed it (an EXIF row too: the page shows it without a model view, D30), upright (the processed
    copy, not the upload as stored), and nothing else: the identical 404 (same body, no ETag) for a withdrawn, a
    rejected, a later (uploaded after the snapshot) and an unplaced (GPS only) photo's public id, a store id, another
    landmark's row, the id in upper case, 13 hex digits, 12 that are no row's and any other string. A matching
    If-None-Match gets a 304, which needs the same check: a withdrawn photo answers 404 at once, also to the ETag a
    browser kept."""
    ds = synthetic_dataset
    client = make_client()
    create_landmark(client, ds)
    create_landmark(client, ds, id="other", name="Other")
    store, salt = client.app.state.store, deployment_salt(settings.data_dir)
    side = ds.entries("good", sector=1)
    # a photo of a side without a model, its compass magnetic: shared, in the snapshot, never placed (src gps)
    files = [("files", ("m.jpg", with_heading_ref(ds.read(side[-1]), b"M"), "image/jpeg"))]
    r = client.post("/api/photos", data=form(client, "carol", landmark_id=LM), files=files)
    (unplaced,) = r.json()["results"]
    # another, its compass true north (the synthetic default): placed by its EXIF alone (src exif). Stored sideways
    # (EXIF Orientation 6), so the upload's shape on disk is not the photo's
    files = [("files", ("s.jpg", stored_sideways(ds.read(side[-2])), "image/jpeg"))]
    r = client.post("/api/photos", data=form(client, "frank", landmark_id=LM), files=files)
    (sideways,) = r.json()["results"]
    (rejected,) = upload(client, ds, [ds.entry("blurry")], user_id="carol")["results"]
    assert unplaced["accepted"] and unplaced["share_location"] and not rejected["accepted"]
    assert sideways["accepted"] and sideways["share_location"]
    built = upload(client, ds, ds.entries("good", sector=0), user_id="alice")
    assert built["triggered_jobs"]
    (later,) = upload(client, ds, side[:1], user_id="dan")["results"]            # after the snapshot
    assert later["share_location"] and store.get_photo(later["photo_id"]).uploaded_at > poses(client)["as_of"]
    other = upload(client, ds, ds.entries("good", sector=0), user_id="erin", landmark_id="other")["results"]
    body = poses(client)
    rows = {row["id"]: row for row in body["photos"]}
    assert set(rows) == {res["public_id"] for res in built["results"]} | {sideways["public_id"]}
    assert rows[sideways["public_id"]]["src"] == "exif" and body["counts"]["exif"] == 1
    assert body["counts"]["shared"] == 6 and body["counts"]["not_determined"] == 1
    gps = pp.photo_poses([store.get_photo(unplaced["photo_id"])], store.published_models(LM), ds.lat, ds.lon)
    assert gps[0].src == "gps"
    other_rows = {row["id"] for row in poses(client, "other")["photos"]}
    assert other_rows == {res["public_id"] for res in other} and not other_rows & set(rows)

    first, *_ = built["results"]
    row_id = first["public_id"]
    ok = client.get(photo_url(row_id))
    etag = ok.headers["etag"]
    assert ok.status_code == 200 and etag == f'W/"{row_id}.1"' and ok.content.startswith(b"\xff\xd8")
    assert client.get(photo_url(sorted(other_rows)[0], "other")).status_code == 200     # a row there
    # the EXIF row's photo: served as every row's, upright (the processed copy, the row's aspect; the upload is stored
    # turned), its own pixels
    stored = store.get_photo(sideways["photo_id"])
    secrets = (sideways["photo_id"].encode(), sideways["manage_token"].encode(), salt.encode(), b"frank",
               str(settings.data_dir).encode())
    pixels = assert_public_photo(client.get(photo_url(sideways["public_id"])), rows[sideways["public_id"]], secrets)
    with Image.open(settings.data_dir / stored.original_path) as up, \
            Image.open(settings.data_dir / stored.processed_path) as proc:
        assert proc.width != proc.height and up.size == proc.size[::-1] == (pixels.shape[0], pixels.shape[1])
    assert mean_pixel_difference(settings.data_dir / stored.processed_path, pixels) < 3.0
    assert client.get(photo_url(row_id, "nowhere")).status_code == 404                # an unknown landmark
    # the identical 404 for everything that is not a row of this landmark now
    for wrong in (public_id(rejected["photo_id"], salt), later["public_id"], unplaced["public_id"],
                  first["photo_id"], *sorted(other_rows), row_id.upper(), row_id + "0", row_id[:-1],
                  next(f"{i:012x}" for i in range(99) if f"{i:012x}" not in rows), "z" * 12, "photo", "%20" * 12):
        assert_no_such_photo(client.get(photo_url(wrong)))
    # a browser that kept the photo asks again (no-cache): the same ETag (weak or strong, "*" too) gets a 304 without
    # a body; another ETag gets the photo
    for tag in (etag, etag.removeprefix("W/"), f'W/"{"0" * 12}.1", {etag}', "*"):
        r = client.get(photo_url(row_id), headers={"If-None-Match": tag})
        assert r.status_code == 304 and r.content == b"" and r.headers["etag"] == etag, tag
        assert r.headers["cache-control"] == "no-cache" and "last-modified" not in r.headers
    assert client.get(photo_url(row_id), headers={"If-None-Match": f'W/"{row_id}.0"'}).content == ok.content
    # served again: the same bytes
    assert client.get(photo_url(row_id)).content == ok.content
    # withdrawn: 404 at once, also to the kept ETag (never a 304)
    assert withdraw(client, first["photo_id"], first["manage_token"]).status_code == 200
    assert_no_such_photo(client.get(photo_url(row_id)))
    assert_no_such_photo(client.get(photo_url(row_id), headers={"If-None-Match": etag}))
    # the next build brings the later photo in: its photo is served from then on
    assert client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}).status_code == 202
    assert later["public_id"] in {row["id"] for row in poses(client)["photos"]}
    assert client.get(photo_url(later["public_id"])).status_code == 200
    assert_no_such_photo(client.get(photo_url(unplaced["public_id"])))


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
    uploaded = upload(client, ds, ds.entries("good", sector=1)[-1:], user_id="carol")
    (carol,) = uploaded["results"]
    (first,) = [j for j in store.list_jobs(LM) if j.kind == "localize"]
    assert uploaded["triggered_jobs"] == [] and first.status == JOB_DONE and fake.calls == []
    assert store.get_localization(carol["photo_id"]).reason == "no_maps"
    assert client.get(f"/api/jobs/{first.id}").status_code == 404                  # internal
    # alice's photos make a facade model with a saved map: carol's photo is localized into it, after that job
    triggered = upload(client, ds, ds.entries("good", sector=0), user_id="alice")
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
    # its photo is served beside the row (D32), as every row's
    pixels = assert_public_photo(client.get(photo_url(carol["public_id"])), rows[carol["public_id"]])
    assert mean_pixel_difference(settings.data_dir / store.get_photo(carol["photo_id"]).processed_path, pixels) < 3.0
    # withdrawn: the row, its photo and the localization go at once
    assert withdraw(client, carol["photo_id"], carol["manage_token"]).status_code == 200
    assert store.get_localization(carol["photo_id"]) is None
    assert carol["public_id"] not in {r["id"] for r in poses(client)["photos"]}
    assert_no_such_photo(client.get(photo_url(carol["public_id"])))


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
        (carol,) = upload(client, ds, ds.entries("good", sector=1)[-1:], user_id="carol")["results"]
        upload(client, ds, ds.entries("good", sector=0), user_id="alice")
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
    upload(client, ds, ds.entries("good", sector=0))
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
    r = client.post("/api/photos", data=form(client, landmark_id="nope"), files=files)
    assert r.status_code == 404


# Registration (D33): the user agreement is accepted once, at registration; uploads carry the contributor token


def contributor_rows(client):
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM contributors").fetchone()[0]


# Tokens that are not ASCII (D33): 64 characters that look like hex digits to a Unicode-aware check (full-width and
# Arabic-Indic digits), and plain non-ASCII letters; each is register_required, never a 500
NON_ASCII_TOKENS = ["é" * 64, "\uff10" * 64, "".join(chr(0xFF10 + i % 10) for i in range(64)), "\u0660" * 64,
                    "\uff41" * 64]


def nothing_stored(client):
    store, settings = client.app.state.store, client.app.state.settings
    return (store.list_photos(LM) == [] and store.list_jobs(LM) == []
            and not any(p.is_file() for p in settings.photos_dir.rglob("*")))


def test_registering_hands_out_the_token_once_and_keeps_only_its_hash(client):
    """D33: POST /api/contributors {nickname, terms} answers 201 with exactly the nickname, the version agreed to and
    the contributor token (64 hex), Cache-Control no-store; never the contributor's id or the token's hash. Nicknames
    are not unique: another registration is another contributor. No GET serves contributors."""
    r = register(client, "alice")
    assert r.status_code == 201, r.text
    assert r.headers["cache-control"] == "no-store"
    body = r.json()
    assert set(body) == {"nickname", "terms", "contributor_token"}
    assert (body["nickname"], body["terms"]) == ("alice", TERMS_CURRENT)
    assert re.fullmatch("[0-9a-f]{64}", body["contributor_token"])
    c = contributors.authenticate(client.app.state.store, body["contributor_token"])
    assert (c.nickname, c.terms) == ("alice", TERMS_CURRENT) and c.id not in r.text and c.token_hash not in r.text
    again = register(client, "alice")
    assert again.status_code == 201 and again.json()["contributor_token"] != body["contributor_token"]
    assert contributor_rows(client) == 2
    for method in ("GET", "HEAD"):
        assert client.request(method, "/api/contributors").status_code == 404
    assert client.get("/api/contributors/accept").status_code in (404, 405)


@pytest.mark.parametrize("nickname", ["", "   ", "u" * 65, "bell\x07", "two\nlines", "nul\x00byte", None,
                                      "a\ud800", "\udfffb", "\ud800\u0001", "\ud800" * 65])
def test_a_bad_nickname_is_422_and_registers_no_one(client, nickname):
    """D33: a nickname that is text comes back in the 422 (input); a lone surrogate (JSON's \\ud800 escape decodes to
    one) is refused like a control character, valid length or not, never a 500 (the store cannot keep it, a response
    cannot carry it), and not echoed (input null)."""
    body = {"terms": TERMS_CURRENT} | ({} if nickname is None else {"nickname": nickname})
    r = client.post("/api/contributors", content=json.dumps(body), headers={"content-type": "application/json"})
    assert r.status_code == 422, r.text
    (err,) = r.json()["detail"]
    assert (err["type"], err["loc"]) == ("value_error", ["body", "nickname"])
    assert err["msg"] == "must be 1-64 characters without control characters"
    lone_surrogate = nickname is not None and any(0xD800 <= ord(c) <= 0xDFFF for c in nickname)
    assert err["input"] == (None if lone_surrogate else nickname), err
    assert contributor_rows(client) == 0


def test_nicknames_are_stripped_and_may_be_any_script(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    assert register(client, "  小明  ").json()["nickname"] == "小明"
    upload(client, ds, ds.entries("good", sector=0)[:1], user_id="  小明  ")
    upload(client, ds, ds.entries("good", sector=0)[1:2], user_id="u" * 64)
    assert {c["user_id"] for c in detail(client)["contributors"]} == {"小明", "u" * 64}


@pytest.mark.parametrize(("terms", "code"), [
    (None, "terms_required"), ("", "terms_required"), ("  ", "terms_required"),
    ("true", "terms_outdated"), ("on", "terms_outdated"), ("share-location-v2", "terms_outdated"),
    ("terms-v0", "terms_outdated"), (f" {TERMS_CURRENT}", "terms_outdated"),
])
def test_registering_without_the_current_agreement_is_refused_first(client, terms, code):
    """D33: registering needs the current user agreement's version, checked before the nickname (a bad one too):
    missing or blank is terms_required, any other text terms_outdated; nothing is stored. The 422 carries the code, an
    English technical msg and ctx {version, url} for the client to translate."""
    for nickname in ("alice", "bell\x07"):
        r = register(client, nickname, terms=terms)
        assert r.status_code == 422, r.text
        (err,) = r.json()["detail"]
        assert (err["type"], err["loc"]) == (code, ["body", "terms"])
        assert err["ctx"] == {"version": TERMS_CURRENT, "url": "/terms.html"}
        assert err["msg"] == "registering needs agreeing to the user agreement terms-v1 (/terms.html)"
    assert contributor_rows(client) == 0


def test_file_count_limits(client):
    one = [("files", ("a.jpg", b"x", "image/jpeg"))]
    data = form(client)
    r = client.post("/api/photos", data=data, files=one * 51)
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "files"]
    assert client.post("/api/photos", data=data).status_code == 422
    r = client.post("/api/photos", data=data, files=one * 50)
    assert r.status_code == 200 and len(r.json()["results"]) == 50


def test_too_large_file_is_a_per_file_result(make_client, synthetic_dataset):
    ds = synthetic_dataset
    good = ds.entries("good", sector=0)
    client = make_client(max_upload_bytes=max(len(ds.read(e)) for e in good[:1]))
    create_landmark(client, ds)
    big = b"\xff" * (client.app.state.settings.max_upload_bytes + 1)
    files = [("files", ("big.jpg", big, "image/jpeg")), ("files", (good[0]["file"], ds.read(good[0]), "image/jpeg"))]
    r = client.post("/api/photos", data=form(client, landmark_id=LM), files=files)
    assert r.status_code == 200
    big_res, ok_res = r.json()["results"]
    assert big_res == {"filename": "big.jpg", "photo_id": None, "accepted": False, "reasons": ["too_large"],
                       "warnings": [], "landmark_id": None, "sector": None, "bearing_deg": None, "distance_m": None,
                       "share_location": False, "public_id": None, "manage_token": None}
    assert ok_res["accepted"]


def test_garbage_file_is_rejected_not_an_error(client):
    r = client.post("/api/photos", data=form(client), files=[("files", ("x.jpg", b"garbage", "image/jpeg"))])
    assert r.status_code == 200
    (res,) = r.json()["results"]
    assert (res["accepted"], res["reasons"], res["landmark_id"]) == (False, ["unreadable_image"], None)


# The user agreement (D32; D33: accepted at registration) and the withdrawal (D30)


def test_an_upload_without_a_registered_contributor_is_refused_before_anything_is_stored(client, synthetic_dataset):
    """D33: an upload needs the token of a registered contributor (form field contributor_token), checked on the server
    before anything else (the file count, the landmark): a missing, blank, malformed or unknown one is register_required
    with input null, so the token never comes back; nothing is stored and no job planned. A D32 client's user_id and
    terms fields count for nothing."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    token = contributor_token(client, "alice")
    c = contributors.authenticate(client.app.state.store, token)
    files = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in ds.entries("good", sector=0)]
    sent = [None, "", "   ", "not-a-token", "0" * 63, "g" * 64, token.upper(), f" {token}", f"{token}0",
            contributors.new_token(), c.token_hash, c.id, *NON_ASCII_TOKENS, f"{token}\n"]
    for value in sent:
        data = {"landmark_id": "nope", "user_id": "alice", "terms": TERMS_CURRENT}
        data |= {} if value is None else {"contributor_token": value}
        for batch in (files, files * 13):                       # also before the file count and the landmark
            r = client.post("/api/photos", data=data, files=batch)
            assert r.status_code == 422, (value, r.text)
            (err,) = r.json()["detail"]
            assert err == {"type": "register_required", "loc": ["body", "contributor_token"], "input": None,
                           "msg": "uploading needs a registered contributor (POST /api/contributors)"}, value
            assert token not in r.text and not (value and value.strip() and value.strip() in r.text), value
    assert nothing_stored(client)


def test_an_outdated_agreement_is_accepted_once_then_uploads_go_through(client, synthetic_dataset):
    """D33: a contributor whose accepted version is not the current one is refused with terms_outdated (at the token,
    input null, ctx {version, url}, msg naming the accept route) before anything is stored; POST
    /api/contributors/accept {contributor_token, terms} records the current version (register_required for an unknown
    token, terms_required / terms_outdated and no change for anything but the current version), and the next upload
    goes through under the new consent, with no new nickname."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    token = contributor_token(client, "alice")
    store, settings = client.app.state.store, client.app.state.settings
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0', terms_accepted_at = 1.0")
    one = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in ds.entries("good", sector=0)[:1]]
    r = client.post("/api/photos", data={"contributor_token": token, "landmark_id": "nope"}, files=one * 51)
    assert r.status_code == 422, r.text
    (err,) = r.json()["detail"]
    assert err == {"type": "terms_outdated", "loc": ["body", "contributor_token"], "input": None,
                   "msg": "the contributor has not accepted the user agreement terms-v1 (/terms.html); accept it with "
                          "POST /api/contributors/accept",
                   "ctx": {"version": TERMS_CURRENT, "url": "/terms.html"}}
    assert token not in r.text and nothing_stored(client)
    stale = contributors.authenticate(store, token)
    assert (stale.terms, stale.terms_accepted_at) == ("terms-v0", 1.0)
    for body, code, loc in [({"contributor_token": contributors.new_token(), "terms": TERMS_CURRENT},
                             "register_required", "contributor_token"),
                            ({"terms": TERMS_CURRENT}, "register_required", "contributor_token"),
                            ({"contributor_token": "x", "terms": TERMS_CURRENT}, "register_required",
                             "contributor_token"),
                            *(({"contributor_token": t, "terms": TERMS_CURRENT}, "register_required",
                               "contributor_token") for t in [*NON_ASCII_TOKENS, f"{token}\n"]),
                            ({"contributor_token": token}, "terms_required", "terms"),
                            ({"contributor_token": token, "terms": " "}, "terms_required", "terms"),
                            ({"contributor_token": token, "terms": "terms-v0"}, "terms_outdated", "terms")]:
        r = client.post("/api/contributors/accept", json=body)
        assert r.status_code == 422, (body, r.text)
        (err,) = r.json()["detail"]
        assert (err["type"], err["loc"]) == (code, ["body", loc]), body
        if code == "register_required":
            assert err["input"] is None and err["msg"] == ("accepting needs a registered contributor "
                                                           "(POST /api/contributors)")
        else:
            assert err["msg"] == "accepting needs agreeing to the user agreement terms-v1 (/terms.html)"
        assert token not in r.text
    assert contributors.authenticate(store, token) == stale                     # the refusals changed nothing
    r = client.post("/api/contributors/accept", json={"contributor_token": token, "terms": TERMS_CURRENT})
    assert r.status_code == 200 and r.json() == {"nickname": "alice", "terms": TERMS_CURRENT}
    accepted = contributors.authenticate(store, token)
    assert accepted.terms == TERMS_CURRENT and accepted.terms_accepted_at > 1.0
    (res,) = upload(client, ds, ds.entries("good", sector=0)[:1], user_id="alice")["results"]
    p = store.get_photo(res["photo_id"])
    assert res["accepted"] and (p.user_id, p.location_consent, p.contributor_id) == ("alice", TERMS_CURRENT,
                                                                                    accepted.id)


def holds_a_piece_of(text, secret, n=16):
    """Whether text holds secret or any n characters of it in a row."""
    return any(secret[i:i + n] in text for i in range(len(secret) - n + 1))


def token_bodies(token):
    """(what, whether it is a well-formed JSON object, request kwargs) for bodies of every shape that carry the
    contributor token (D33): the token as a list, an object, an object key, nested, a number; JSON of another top level;
    the token in the terms or in extra fields; not JSON, cut short, not UTF-8, nested too deep; JSON under another or
    no content type; a form, a multipart form; bigger than the cap, with and without a Content-Length."""
    as_json = {"content-type": "application/json"}
    good = json.dumps({"contributor_token": token, "terms": TERMS_CURRENT})
    number = str(int(token, 16))
    over = api_module.CONTRIBUTOR_BODY_MAX_BYTES

    def chunks():                                   # no Content-Length: the cap counts what arrives
        yield good[:-1].encode() + b', "pad": "'
        for _ in range(over // 1024 + 1):
            yield b"x" * 1024
        yield b'"}'

    return [
        ("token as a list", True, {"json": {"contributor_token": [token], "terms": TERMS_CURRENT}}),
        ("token as an object", True, {"json": {"contributor_token": {"token": token}, "terms": TERMS_CURRENT}}),
        ("token as a key", True, {"json": {"contributor_token": {token: token}, "terms": TERMS_CURRENT}}),
        ("token nested", True, {"json": {"contributor_token": [[{"t": [token]}]], "terms": TERMS_CURRENT}}),
        ("token as a number", True, {"content": f'{{"contributor_token": {number}, "terms": "{TERMS_CURRENT}"}}',
                                     "headers": as_json}),
        ("token true", True, {"json": {"contributor_token": True, "terms": TERMS_CURRENT}}),
        ("token in the terms", True, {"json": {"contributor_token": token, "terms": token}}),
        ("token in a list of terms", True, {"json": {"contributor_token": token, "terms": [token]}}),
        ("token in extra fields", True, {"json": {"contributor_token": "x", "terms": TERMS_CURRENT, "token": token,
                                                  "user_id": token, token: token}}),
        ("a JSON list", False, {"json": [token, TERMS_CURRENT]}),
        ("a JSON string", False, {"json": token}),
        ("a JSON number", False, {"content": number, "headers": as_json}),
        ("JSON null", False, {"content": "null", "headers": as_json}),
        ("not JSON", False, {"content": f"contributor_token: {token}", "headers": as_json}),
        ("JSON cut short", False, {"content": good[:-1], "headers": as_json}),
        ("not UTF-8", False, {"content": good[:-2].encode() + b'\xff"}', "headers": as_json}),
        ("nested too deep", False, {"content": "[" * 5000 + json.dumps(token) + "]" * 5000, "headers": as_json}),
        ("JSON as text/plain", False, {"content": good, "headers": {"content-type": "text/plain"}}),
        ("JSON as form-encoded", False, {"content": good,
                                         "headers": {"content-type": "application/x-www-form-urlencoded"}}),
        ("JSON without a content type", False, {"content": good}),
        ("a form", False, {"data": {"contributor_token": token, "terms": TERMS_CURRENT}}),
        ("a multipart form", False, {"data": {"terms": TERMS_CURRENT}, "files": {"contributor_token": (None, token)}}),
        ("over the cap", False, {"json": {"contributor_token": token, "terms": TERMS_CURRENT, "pad": "x" * over}}),
        ("over the cap, streamed", False, {"content": chunks(), "headers": as_json}),
    ]


def nickname_bodies(token):
    """(what, request kwargs) for registration bodies whose nickname is not text but carries the contributor token
    (D33): only a nickname that is text comes back in a 422, so no piece of these may (nor, nested deep, crash the
    422's rendering)."""
    as_json = {"content-type": "application/json"}
    deep = "[" * 5000 + json.dumps(token) + "]" * 5000
    return [
        ("nickname as a list", {"json": {"nickname": [token], "terms": TERMS_CURRENT}}),
        ("nickname as an object", {"json": {"nickname": {token: token}, "terms": TERMS_CURRENT}}),
        ("nickname nested", {"json": {"nickname": [[{"t": [token]}]], "terms": TERMS_CURRENT}}),
        ("nickname as a number", {"content": f'{{"nickname": {int(token, 16)}, "terms": "{TERMS_CURRENT}"}}',
                                  "headers": as_json}),
        ("nickname nested deep", {"content": f'{{"nickname": {deep}, "terms": "{TERMS_CURRENT}"}}',
                                  "headers": as_json}),
    ]


def test_no_contributor_route_echoes_any_part_of_a_token_whatever_the_body(client):
    """D33: POST /api/contributors/accept reads its body by hand: whatever its shape, no response holds the token or
    any 16 characters of it (FastAPI's own 422 echoed a token sent as a list or an object, or in a body that is not
    JSON). Anything but a JSON object sent as application/json within the cap is refused like an empty body,
    register_required at the token with input null, and changes nothing; a well-formed body is checked as before
    (token, then terms; the terms are not echoed either). POST /api/contributors does the same: terms_required for a
    body of another shape, nothing echoed, nothing stored; and it echoes a nickname only when it is text (not a token
    sent as the nickname in a list, an object or a number)."""
    token = contributor_token(client, "alice")
    store = client.app.state.store
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0', terms_accepted_at = 1.0")
    stale = contributors.authenticate(store, token)
    secrets = [token, str(int(token, 16))]
    for route, doing in (("/api/contributors/accept", "accepting"), ("/api/contributors", "registering")):
        for what, well_formed, kw in token_bodies(token):           # anew for each route: one body is a generator
            r = client.post(route, **kw)
            assert r.status_code == 422, (route, what, r.text)
            assert not any(holds_a_piece_of(r.text, s) for s in secrets), (route, what, r.text)
            if what.endswith("streamed"):
                assert "content-length" not in r.request.headers, what          # the cap counts what arrives
            (err,) = r.json()["detail"]
            assert err["input"] is None, (route, what, err)
            if route == "/api/contributors/accept":
                in_terms = what.startswith("token in") and "terms" in what
                code = ("terms_outdated", "terms") if in_terms else ("register_required", "contributor_token")
                assert (err["type"], err["loc"][1]) == code, (what, err)
            elif not well_formed:
                assert (err["type"], err["loc"], err["msg"]) == (
                    "terms_required", ["body", "terms"],
                    f"{doing} needs agreeing to the user agreement {TERMS_CURRENT} (/terms.html)"), (what, err)
        assert "content-type" not in client.post(route, content="{}").request.headers
    for what, kw in nickname_bodies(token):
        r = client.post("/api/contributors", **kw)
        assert r.status_code == 422, (what, r.text)
        assert not any(holds_a_piece_of(r.text, s) for s in secrets), (what, r.text)
        (err,) = r.json()["detail"]
        assert (err["type"], err["loc"], err["input"]) == ("value_error", ["body", "nickname"], None), (what, err)
    assert contributors.authenticate(store, token) == stale and contributor_rows(client) == 1      # nothing changed
    # a well-formed body is read with any charset parameter, and in any case of the media type
    for content_type in ("application/json; charset=utf-8", "Application/JSON"):
        r = client.post("/api/contributors/accept", headers={"content-type": content_type},
                        content=json.dumps({"contributor_token": token, "terms": TERMS_CURRENT}))
        assert r.status_code == 200 and r.json() == {"nickname": "alice", "terms": TERMS_CURRENT}, r.text
    r = client.post("/api/contributors", headers={"content-type": "application/json; charset=utf-8"},
                    content=json.dumps({"nickname": "bob", "terms": TERMS_CURRENT, "contributor_token": token}))
    assert r.status_code == 201 and not holds_a_piece_of(r.text, token) and contributor_rows(client) == 2
    # the API docs still describe both bodies
    paths = client.get("/openapi.json").json()["paths"]
    for route, fields in (("/api/contributors", {"nickname", "terms"}),
                          ("/api/contributors/accept", {"contributor_token", "terms"})):
        schema = paths[route]["post"]["requestBody"]["content"]["application/json"]["schema"]
        assert set(schema["properties"]) == set(schema["required"]) == fields


def run_asgi(app, method, path, headers, messages):
    """Call app directly (ASGI) with these receive messages (after them: http.disconnect, as a server sends once the
    client has gone); returns the messages it sent. An exception the app does not handle propagates."""
    queue = list(messages)
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"host", b"testserver"), *headers], "client": ("127.0.0.1", 50000),
             "server": ("testserver", 80)}
    sent = []

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    anyio.run(app, scope, receive, send)
    return sent


def test_a_contributor_body_cut_off_by_its_client_is_a_quiet_400(client, caplog):
    """D33: the contributor routes read their body by hand (contributor_fields). A client that leaves before the end
    of its body (the server's receive gives http.disconnect) gets a 400 BODY_CUT_OFF, as FastAPI's own body parsing
    gave, not an uncaught ClientDisconnect (which a server logs as an ERROR traceback, once per request, for any
    client). With or without a Content-Length; nothing is stored, changed or echoed, nothing is logged as an error."""
    token = contributor_token(client, "alice")
    store = client.app.state.store
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0', terms_accepted_at = 1.0")
    stale = contributors.authenticate(store, token)
    as_json = (b"content-type", b"application/json")
    caplog.set_level(logging.DEBUG)
    for route, part in (("/api/contributors/accept", f'{{"contributor_token": "{token}", "terms": "ter'),
                        ("/api/contributors", f'{{"nickname": "{token}", "terms": "{TERMS_CURRENT}'),
                        ("/api/contributors", "")):
        for length in ([(b"content-length", b"200")], []):
            chunks = [{"type": "http.request", "body": part.encode(), "more_body": True}] if part else []
            sent = run_asgi(client.app, "POST", route, [as_json, *length], chunks)
            start, *body = sent
            assert (start["type"], start["status"]) == ("http.response.start", 400), (route, length, sent)
            text = b"".join(m.get("body", b"") for m in body).decode()
            assert json.loads(text) == {"detail": api_module.BODY_CUT_OFF}, text
            assert not holds_a_piece_of(text, token)
    assert contributors.authenticate(store, token) == stale and contributor_rows(client) == 1      # nothing changed
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR], caplog.text


def upload_forms(token):
    """(what, request kwargs) for upload forms that carry the contributor token in a part of the wrong kind (D33):
    FastAPI's own 422 echoed the part: a file part's file name, part headers and temporary file (string_type), a text
    part sent as the files (value_error)."""
    jpg = ("files", ("a.jpg", b"\xff\xd8\xff\xd9", "image/jpeg"))
    as_file = token.encode()
    return [
        ("token as a file part named by it", {"files": [jpg, ("contributor_token", (token, as_file, "text/plain"))]}),
        ("token as a file part typed by it", {"files": [jpg, ("contributor_token", ("t", as_file, f"x/{token}"))]}),
        ("token as a file part", {"files": [jpg, ("contributor_token", ("t.txt", as_file, "text/plain"))]}),
        ("token as text and as a file part", {"files": [jpg, ("contributor_token", (None, token)),
                                                        ("contributor_token", (token, as_file, "text/plain"))]}),
        ("files as text", {"files": [("files", (None, token)), ("contributor_token", (None, token))]}),
        ("files as text, no token", {"files": [("files", (None, token))]}),
        ("files and token form-encoded", {"data": {"files": token, "contributor_token": token}}),
        ("landmark id as a file part named by the token", {"files": [jpg, ("contributor_token", (None, token)),
                                                                     ("landmark_id", (token, b"x", f"x/{token}"))]}),
        ("JSON", {"json": {"files": [token], "contributor_token": [token]}}),
        ("51 files", {"files": [jpg] * 51, "data": {"contributor_token": token}}),
    ]


def test_no_upload_422_echoes_any_input_whatever_the_form(client):
    """D33: the upload's form holds the contributor token, and no 422 of the upload echoes any input
    (api.validation_error), so no piece of the token comes back however the form is built (a part of the wrong kind,
    form-encoded, JSON); FastAPI's own 422 echoed a file part's file name, part headers and temporary file, and a text
    part sent as the files. Nothing is stored."""
    create_landmark(client)
    token = contributor_token(client, "alice")
    for what, kw in upload_forms(token):
        r = client.post("/api/photos", **kw)
        assert r.status_code == 422, (what, r.text)
        assert not holds_a_piece_of(r.text, token), (what, r.text)
        errors = r.json()["detail"]
        assert errors and all(e["input"] is None for e in errors), (what, errors)
        assert not re.search(r"_TemporaryFileArgs|_max_size|content-disposition", r.text), (what, r.text)
    assert nothing_stored(client)


def test_the_nickname_and_the_consent_come_from_the_registration(client, synthetic_dataset):
    """D33: an upload's nickname, consent and registration are the contributor's record, whatever else the form says
    (a D32 client's user_id and terms are ignored)."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    token = contributor_token(client, "alice")
    files = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in ds.entries("good", sector=0)[:2]]
    r = client.post("/api/photos", files=files, data={"contributor_token": token, "landmark_id": LM,
                                                       "user_id": "mallory", "terms": "terms-v0"})
    assert r.status_code == 200, r.text
    store = client.app.state.store
    c = contributors.authenticate(store, token)
    for res in r.json()["results"]:
        p = store.get_photo(res["photo_id"])
        assert (p.user_id, p.location_consent, p.contributor_id) == ("alice", TERMS_CURRENT, c.id)
    assert detail(client)["contributors"] == [{"user_id": "alice"}]
    assert "mallory" not in client.get(f"/api/landmarks/{LM}").text


def test_no_contributor_token_reaches_a_log(make_client, synthetic_dataset, caplog):
    """D33: registering, uploading, accepting and every refusal log nothing of a token, at any level (the token
    travels only in POST bodies, never in a URL). D34: nor do listing a registration's photos and withdrawing by the
    registration, refused or not; and no manage token is logged either."""
    caplog.set_level(logging.DEBUG)
    ds = synthetic_dataset
    client = make_client()
    create_landmark(client, ds)
    token = register(client, "alice").json()["contributor_token"]
    c = contributors.authenticate(client.app.state.store, token)
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0'")
    one = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in ds.entries("good", sector=0)[:1]]
    unknown = contributors.new_token()
    assert client.post("/api/photos", data={"contributor_token": token}, files=one).status_code == 422
    for t in (unknown, f"{token}x"):
        assert client.post("/api/photos", data={"contributor_token": t}, files=one).status_code == 422
        assert client.post("/api/contributors/accept", json={"contributor_token": t,
                                                             "terms": TERMS_CURRENT}).status_code == 422
    assert client.post("/api/contributors/accept", json={"contributor_token": token, "terms": "x"}).status_code == 422
    assert client.post("/api/contributors/accept", json={"contributor_token": token,
                                                         "terms": TERMS_CURRENT}).status_code == 200
    r = client.post("/api/photos", data={"contributor_token": token, "landmark_id": LM},
                    files=[*one, ("files", ("x.jpg", b"garbage", "image/jpeg"))])
    assert r.status_code == 200
    assert register(client, "bob", terms="terms-v0").status_code == 422
    (stored, _) = r.json()["results"]
    share = f"/api/photos/{stored['photo_id']}/share_location"
    listed = client.post("/api/contributors/photos", json={"contributor_token": token, "landmark_id": LM})
    assert listed.status_code == 200 and [p["manage_token"] for p in listed.json()["photos"]] == [stored["manage_token"]]
    for t in (unknown, f"{token}x"):
        assert client.post("/api/contributors/photos", json={"contributor_token": t}).status_code == 422
        assert client.post(share, json={"share": False, "contributor_token": t}).status_code == 403
    assert client.post(share, json={"contributor_token": token}).status_code == 422
    assert client.post(share, json={"share": False, "token": "0" * 64}).status_code == 403
    assert client.post(share, json={"share": False, "contributor_token": token}).status_code == 200
    assert client.post(share, json={"share": False, "token": stored["manage_token"]}).status_code == 200
    assert caplog.records and "/api/photos" in caplog.text                  # the request lines were captured
    assert "/api/contributors/photos" in caplog.text and "share_location" in caplog.text
    for secret in (token, c.token_hash, unknown, stored["manage_token"]):
        assert secret not in caplog.text


def test_every_accepted_upload_is_shared_with_a_manage_token(client, synthetic_dataset):
    """D32: an upload under the agreement shares every accepted photo (its rounded pose now, its photo too): the
    result says so and hands out the public id and the manage token, HMAC(salt, "manage:" + photo id); a rejected
    photo is never shown, so it gets neither and keeps no consent."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    results = upload(client, ds, [*ds.entries("good", sector=0)[:2], ds.entry("blurry"), ds.entry("far")])["results"]
    salt = deployment_salt(client.app.state.settings.data_dir)
    store = client.app.state.store
    assert [r["accepted"] for r in results] == [True, True, False, False]
    for res in results[:2]:
        assert res["share_location"] is True
        assert res["manage_token"] == manage_token(res["photo_id"], salt)
        assert re.fullmatch(r"[0-9a-f]{64}", res["manage_token"])
        assert res["public_id"] == public_id(res["photo_id"], salt)
        assert store.get_photo(res["photo_id"]).location_consent == TERMS_CURRENT == "terms-v1"
    for res in results[2:]:
        assert res["photo_id"]
        assert (res["share_location"], res["public_id"], res["manage_token"]) == (False, None, None)
        assert store.get_photo(res["photo_id"]).location_consent is None


def test_the_user_agreement_is_served(client):
    r = client.get("/terms.html")
    assert r.status_code == 200 and "text/html" in r.headers["content-type"]
    assert r.text.count('data-terms-version="terms-v1"') == 2 and TERMS_CURRENT == "terms-v1"


def withdraw(client, photo_id, token, share=False):
    return client.post(f"/api/photos/{photo_id}/share_location", json={"share": share, "token": token})


def test_the_uploader_withdraws_consent_with_the_manage_token(client, synthetic_dataset):
    ds = synthetic_dataset
    create_landmark(client, ds)
    store = client.app.state.store
    first, second = upload(client, ds, ds.entries("good", sector=0)[:2])["results"]
    pid, token = first["photo_id"], first["manage_token"]
    assert store.get_photo(pid).location_consent == TERMS_CURRENT
    before = store.get_photo(pid)

    # a wrong token (another photo's) or a malformed one changes nothing
    assert withdraw(client, pid, second["manage_token"]).status_code == 403
    for bad in ("", "x" * 64, token[:-1], token.upper(), token + "0"):
        assert withdraw(client, pid, bad).status_code == 422, bad
    # the agreement is given at upload only: share true is not accepted
    assert withdraw(client, pid, token, share=True).status_code == 422
    assert client.post(f"/api/photos/{pid}/share_location", json={"token": token}).status_code == 422
    assert store.get_photo(pid) == before

    r = withdraw(client, pid, token)
    assert r.status_code == 200 and r.json() == {"photo_id": pid, "share_location": False}
    assert store.get_photo(pid).location_consent is None
    assert withdraw(client, pid, token).status_code == 200                    # again: still withdrawn
    assert store.get_photo(second["photo_id"]).location_consent == TERMS_CURRENT     # the other one is untouched

    # a token is only valid for its own photo id; an unknown id answers 403 like a wrong token, unless the token was
    # made with this deployment's salt for exactly that id
    salt = deployment_salt(client.app.state.settings.data_dir)
    assert withdraw(client, "0" * 32, token).status_code == 403
    assert withdraw(client, "0" * 32, manage_token("0" * 32, salt)).status_code == 404


# A contributor can always withdraw their own photos (D34): the registration lists and withdraws them

OWN_PHOTOS = "/api/contributors/photos"
OWN_PHOTO_KEYS = {"photo_id", "landmark_id", "public_id", "manage_token"}
REFUSED = {"detail": "wrong token for this photo"}


def own_photos(client, token, **fields):
    return client.post(OWN_PHOTOS, json={"contributor_token": token, **fields})


def withdraw_as(client, photo_id, token):
    """Withdraw a photo by the registration it was uploaded under (D34), not by its manage token."""
    return client.post(f"/api/photos/{photo_id}/share_location", json={"share": False, "contributor_token": token})


def photo_rows(client):
    """Every photo row of the store, by id: what a refusal must leave as it was."""
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.row_factory = sqlite3.Row
        return {row["id"]: dict(row) for row in conn.execute("SELECT * FROM photos")}


def test_a_registration_lists_its_own_shared_photos_and_never_another_registrations(client, synthetic_dataset):
    """D34: POST /api/contributors/photos {contributor_token, landmark_id?} gives a registration the photos uploaded
    under it that are shared now: exactly photo_id, landmark_id, public_id and manage_token, as its uploads' results
    gave them, in upload order, Cache-Control no-store. A manage token used to exist only in its upload's response,
    so a lost response left a public photo nobody could withdraw. Never a rejected or withdrawn photo, one whose
    consent is no agreement version, or anything of another registration; an outdated agreement lists all the same
    (finding and withdrawing one's photos must not take accepting a new one). No GET."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    assert client.post("/api/landmarks", json={"id": "other", "name": "Other", "lat": ds.lat, "lon": ds.lon}
                       ).status_code == 201
    store = client.app.state.store
    salt = deployment_salt(client.app.state.settings.data_dir)
    first = upload(client, ds, [*ds.entries("good", sector=0), ds.entry("blurry"), ds.entry("far")],
                   user_id="alice")["results"]
    bob = upload(client, ds, ds.entries("good", sector=1)[:2], user_id="bob")["results"]
    later = upload(client, ds, ds.entries("good", sector=1)[2:], user_id="alice")["results"]
    elsewhere = upload(client, ds, ds.entries("good", sector=0)[:2], user_id="alice", landmark_id="other")["results"]
    assert [r["accepted"] for r in first] == [True] * 4 + [False] * 2 and all(r["accepted"] for r in bob + later)
    assert all(r["accepted"] and r["landmark_id"] == "other" for r in elsewhere)
    alice, robert = contributor_token(client, "alice"), contributor_token(client, "bob")

    def row(res):
        return {k: res[k] for k in ("photo_id", "landmark_id", "public_id", "manage_token")}

    mine = [row(r) for r in [*first[:4], *later, *elsewhere]]             # upload order; bob's lie in between
    r = own_photos(client, alice)
    assert r.status_code == 200 and r.json() == {"photos": mine}, r.text
    assert r.headers["cache-control"] == "no-store"
    assert all(set(p) == OWN_PHOTO_KEYS for p in r.json()["photos"])
    for p in mine:                                                        # recomputed, not stored: the same tokens
        assert (p["public_id"], p["manage_token"]) == (public_id(p["photo_id"], salt),
                                                       manage_token(p["photo_id"], salt))
    assert own_photos(client, alice, landmark_id=LM).json() == {"photos": mine[:6]}
    assert own_photos(client, alice, landmark_id="other").json() == {"photos": mine[6:]}
    assert own_photos(client, alice, landmark_id=None).json() == {"photos": mine}
    for nowhere in ("nope", "", " ", "DEMO", "demo ", "%", "de_o", "\x00"):
        assert own_photos(client, alice, landmark_id=nowhere).json() == {"photos": []}, nowhere
    theirs = [row(r) for r in bob]
    assert own_photos(client, robert).json() == {"photos": theirs}
    # nothing of the other registration, of a rejected photo, or of the registration itself, in either answer
    c_alice, c_bob = contributors.authenticate(store, alice), contributors.authenticate(store, robert)
    for token, other, keep in ((alice, theirs, mine), (robert, mine, theirs)):
        text = own_photos(client, token).text
        for p in other:
            assert not any(p[k] in text for k in ("photo_id", "public_id", "manage_token")), p
        for res in first[4:]:
            assert res["photo_id"] not in text
        for secret in (alice, robert, c_alice.token_hash, c_bob.token_hash, c_alice.id, c_bob.id, salt, "alice", "bob"):
            assert secret not in text
        assert all(p["photo_id"] in text for p in keep)
    assert own_photos(client, register(client, "carol").json()["contributor_token"]).json() == {"photos": []}

    # a token the route gives withdraws its photo, and the photo leaves the list at once; so does one withdrawn by
    # the registration, and one whose consent is no agreement version (a D30 text: not shared)
    assert withdraw(client, mine[1]["photo_id"], mine[1]["manage_token"]).status_code == 200
    assert withdraw_as(client, mine[7]["photo_id"], alice).status_code == 200
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE photos SET location_consent = 'share-location-v2' WHERE id = ?", (mine[2]["photo_id"],))
    left = [p for i, p in enumerate(mine) if i not in (1, 2, 7)]
    assert own_photos(client, alice).json() == {"photos": left}
    assert own_photos(client, alice, landmark_id="other").json() == {"photos": [mine[6]]}
    assert own_photos(client, robert).json() == {"photos": theirs}        # untouched

    # an older agreement version: the list still comes, and its tokens still withdraw (no accepting first)
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0', terms_accepted_at = 1.0")
    assert contributors.authenticate(store, alice).terms == "terms-v0"
    assert own_photos(client, alice).json() == {"photos": left}
    assert withdraw(client, left[0]["photo_id"], left[0]["manage_token"]).status_code == 200
    assert own_photos(client, alice).json() == {"photos": left[1:]}
    assert contributors.authenticate(store, alice).terms == "terms-v0"    # listing accepted nothing

    # not a GET (nor any other method): the token travels in a POST body only
    for method in ("GET", "HEAD", "PUT", "DELETE", "PATCH"):
        r = client.request(method, OWN_PHOTOS)
        assert r.status_code in (404, 405) and not any(p["photo_id"] in r.text for p in mine + theirs), method
    r = client.get(OWN_PHOTOS, params={"contributor_token": alice})
    assert r.status_code in (404, 405) and "photos" not in r.text
    # the API docs describe the body
    schema = client.get("/openapi.json").json()["paths"][OWN_PHOTOS]["post"]["requestBody"]["content"][
        "application/json"]["schema"]
    assert set(schema["properties"]) == {"contributor_token", "landmark_id"}
    assert schema["required"] == ["contributor_token"]


def test_the_list_of_own_photos_refuses_without_a_known_registration_and_echoes_nothing(client, synthetic_dataset):
    """D34: POST /api/contributors/photos reads its body by hand (contributor_fields), like the other contributor
    routes. A missing, blank, malformed or unknown token, and every body that is not a JSON object sent as
    application/json within the cap, is register_required at the token with input null; a landmark_id that is not
    text is refused at landmark_id with input null. No response holds the token, any 16 characters of it, or a photo.
    A cut-off body is a quiet 400."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    results = upload(client, ds, ds.entries("good", sector=0)[:2], user_id="alice")["results"]
    token = contributor_token(client, "alice")
    store = client.app.state.store
    c = contributors.authenticate(store, token)
    required = {"type": "register_required", "loc": ["body", "contributor_token"], "input": None,
                "msg": "listing photos needs a registered contributor (POST /api/contributors)"}
    secrets = [token, str(int(token, 16)), *(res["photo_id"] for res in results),
               *(res["manage_token"] for res in results)]
    sent = [None, "", "   ", "not-a-token", "0" * 63, "g" * 64, token.upper(), f" {token}", f"{token}0",
            contributors.new_token(), c.token_hash, c.id, *NON_ASCII_TOKENS, f"{token}\n", results[0]["manage_token"]]
    for value in sent:
        r = client.post(OWN_PHOTOS, json={"landmark_id": LM} | ({} if value is None else {"contributor_token": value}))
        assert r.status_code == 422 and r.json() == {"detail": [required]}, (value, r.text)
        assert "cache-control" not in r.headers
    for what, well_formed, kw in token_bodies(token):
        r = client.post(OWN_PHOTOS, **kw)
        in_other_fields = what.startswith("token in") and "extra" not in what       # a right token, more fields
        if in_other_fields:
            assert well_formed and r.status_code == 200 and len(r.json()["photos"]) == 2, (what, r.text)
        else:
            assert r.status_code == 422 and r.json() == {"detail": [required]}, (what, r.text)
            assert not any(holds_a_piece_of(r.text, s) for s in secrets), (what, r.text)
        assert not any(holds_a_piece_of(r.text, s) for s in secrets[:2]), (what, r.text)
    # the landmark: text or left out
    as_json = {"content-type": "application/json"}
    for landmark in ([LM], {"id": LM}, 7, True, 1.5, [token]):
        r = own_photos(client, token, landmark_id=landmark)
        assert r.status_code == 422, (landmark, r.text)
        (err,) = r.json()["detail"]
        assert (err["type"], err["loc"], err["input"]) == ("value_error", ["body", "landmark_id"], None), landmark
        assert not any(holds_a_piece_of(r.text, s) for s in secrets), (landmark, r.text)
    r = client.post(OWN_PHOTOS, content=f'{{"contributor_token": "{token}", "landmark_id": "\\ud800"}}', headers=as_json)
    assert r.status_code == 422 and r.json()["detail"][0]["loc"] == ["body", "landmark_id"], r.text    # never a 500
    # a client that leaves before the end of its body
    part = f'{{"contributor_token": "{token}", "landmark_id": "de'
    for length in ([(b"content-length", b"200")], []):
        sent_ = run_asgi(client.app, "POST", OWN_PHOTOS, [(b"content-type", b"application/json"), *length],
                         [{"type": "http.request", "body": part.encode(), "more_body": True}])
        status, _, body = asgi_response(sent_)
        assert status == 400 and json.loads(body) == {"detail": api_module.BODY_CUT_OFF}
    assert all(location_shared(store.get_photo(res["photo_id"])) for res in results)      # nothing was withdrawn


def test_a_registration_withdraws_its_own_photos_and_every_refusal_is_the_same_403(client, synthetic_dataset):
    """D34: POST /api/photos/{id}/share_location also takes the registration: {share: false, contributor_token}
    withdraws the photo when it was uploaded under that registration, whatever agreement version it accepted. Every
    failure gets the one 403 a wrong manage token gets, byte for byte: another registration's photo, a photo stored
    before registrations, a photo that does not exist, an unknown or malformed token. So the answer tells nobody
    whether a photo exists or whose it is, and a refusal changes nothing. Exactly one of the two tokens."""
    ds = synthetic_dataset
    create_landmark(client, ds)
    store = client.app.state.store
    mine = upload(client, ds, [*ds.entries("good", sector=0), ds.entry("blurry")], user_id="alice")["results"]
    theirs = upload(client, ds, ds.entries("good", sector=1), user_id="bob")["results"]
    assert [r["accepted"] for r in mine] == [True] * 4 + [False]
    alice, robert = contributor_token(client, "alice"), contributor_token(client, "bob")
    c = contributors.authenticate(store, alice)
    store.add_photo(PhotoRecord(id="before-d33", landmark_id=LM, user_id="alice", status="accepted", uploaded_at=1.0,
                                location_consent=TERMS_CURRENT))
    own, rejected, other = mine[0]["photo_id"], mine[4]["photo_id"], theirs[0]["photo_id"]
    before = photo_rows(client)
    assert before[own]["contributor_id"] == c.id and before["before-d33"]["contributor_id"] is None

    refusals = [
        ("another registration's photo", other, alice),
        ("my photo, another registration", own, robert),
        ("a photo stored before registrations", "before-d33", alice),
        ("a photo that does not exist", "0" * 32, alice),
        ("a photo id that is no id", "nope", alice),
        ("an unknown token, my photo", own, contributors.new_token()),
        ("an unknown token, no photo", "0" * 32, contributors.new_token()),
        ("the token's hash", own, c.token_hash),
        ("the contributor id", own, c.id),
        ("the photo's manage token as the registration", own, mine[0]["manage_token"]),
        *((f"a malformed token {bad!r}", own, bad)
          for bad in ("", " ", "x", alice.upper(), f"{alice}0", alice[:-1], f" {alice}", f"{alice}\n",
                      *NON_ASCII_TOKENS)),
    ]
    assert alice.upper() != alice
    answers = set()
    for what, photo_id, token in refusals:
        r = withdraw_as(client, photo_id, token)
        assert r.status_code == 403 and r.json() == REFUSED, (what, r.status_code, r.text)
        answers.add((r.status_code, r.content, tuple(sorted((k, v) for k, v in r.headers.items()))))
        assert not holds_a_piece_of(r.text, alice) and photo_id not in r.text, what
    wrong_manage = withdraw(client, own, theirs[0]["manage_token"])
    answers.add((wrong_manage.status_code, wrong_manage.content,
                 tuple(sorted((k, v) for k, v in wrong_manage.headers.items()))))
    assert len(answers) == 1, answers                                   # one answer for all: status, body, headers
    assert photo_rows(client) == before                                 # and nothing changed

    # exactly one of the two tokens; share false; no 422 of this route echoes any input (the body may hold the token)
    route = f"/api/photos/{own}/share_location"
    as_json = {"content-type": "application/json"}
    good = json.dumps({"share": False, "contributor_token": alice})
    for what, kw, error in [
        ("both tokens", {"json": {"share": False, "contributor_token": alice, "token": mine[0]["manage_token"]}},
         ("value_error", ["body", "token"])),
        ("neither token", {"json": {"share": False}}, ("value_error", ["body", "token"])),
        ("both null", {"json": {"share": False, "token": None, "contributor_token": None}},
         ("value_error", ["body", "token"])),
        ("share true", {"json": {"share": True, "contributor_token": alice}}, ("literal_error", ["body", "share"])),
        ("no share", {"json": {"contributor_token": alice}}, ("missing", ["body", "share"])),
        ("token as a list", {"json": {"share": False, "contributor_token": [alice]}},
         ("string_type", ["body", "contributor_token"])),
        ("token as an object", {"json": {"share": False, "contributor_token": {alice: alice}}},
         ("string_type", ["body", "contributor_token"])),
        ("token as a number", {"content": f'{{"share": false, "contributor_token": {int(alice, 16)}}}',
                               "headers": as_json}, ("string_type", ["body", "contributor_token"])),
        ("the registration as the manage token", {"json": {"share": False, "token": [alice]}},
         ("string_type", ["body", "token"])),
        ("a JSON list", {"json": [False, alice]}, ("model_attributes_type", ["body"])),
        ("a JSON string", {"json": alice}, ("model_attributes_type", ["body"])),
        ("JSON as text/plain", {"content": good, "headers": {"content-type": "text/plain"}},
         ("model_attributes_type", ["body"])),
        ("a form", {"data": {"share": "false", "contributor_token": alice}}, ("model_attributes_type", ["body"])),
        ("not JSON", {"content": f"contributor_token={alice}", "headers": as_json}, ("json_invalid", None)),
        ("JSON cut short", {"content": good[:-1], "headers": as_json}, ("json_invalid", None)),
    ]:
        r = client.post(route, **kw)
        assert r.status_code == 422, (what, r.status_code, r.text)
        errors = r.json()["detail"]
        found = [e for e in errors if e["type"] == error[0] and (error[1] is None or e["loc"] == error[1])]
        assert found and all(e["input"] is None for e in errors), (what, errors)
        assert not holds_a_piece_of(r.text, alice) and not holds_a_piece_of(r.text, str(int(alice, 16))), (what, r.text)
    assert photo_rows(client) == before

    # the registration withdraws its own photo: the answer of the manage token's withdrawal, the consent gone
    r = withdraw_as(client, own, alice)
    assert r.status_code == 200 and r.json() == {"photo_id": own, "share_location": False}
    assert store.get_photo(own).location_consent is None and not location_shared(store.get_photo(own))
    assert withdraw_as(client, own, alice).status_code == 200                   # again: still withdrawn
    assert withdraw_as(client, rejected, alice).status_code == 200              # its rejected photo: not shared anyway
    assert photo_rows(client) == {**before, own: {**before[own], "location_consent": None}}
    # under an older agreement version too: withdrawing takes no accepting
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0', terms_accepted_at = 1.0")
    second = mine[1]["photo_id"]
    assert withdraw_as(client, second, alice).status_code == 200 and not location_shared(store.get_photo(second))
    assert contributors.authenticate(store, alice).terms == "terms-v0"
    assert withdraw_as(client, other, alice).status_code == 403 and location_shared(store.get_photo(other))
    # the manage token's own withdrawal is as it was
    assert withdraw(client, other, theirs[0]["manage_token"]).json() == {"photo_id": other, "share_location": False}


def test_a_withdrawal_by_the_registration_hides_the_pose_and_the_photo_at_once(make_client, synthetic_dataset):
    """D34: withdrawn by the registration, a photo leaves the public poses and its photo answers 404 at once, as
    after a withdrawal by its manage token; the photos of the same registration that were not withdrawn stay."""
    ds = synthetic_dataset
    client = make_client(SizeTaggingReconstructor())
    create_landmark(client, ds)
    results = upload(client, ds, ds.entries("good", sector=0), user_id="alice")["results"]
    token = contributor_token(client, "alice")
    rows = {row["id"] for row in poses(client)["photos"]}
    assert rows == {res["public_id"] for res in results}
    gone = results[0]
    assert client.get(photo_url(gone["public_id"])).status_code == 200
    assert withdraw_as(client, gone["photo_id"], token).status_code == 200
    assert {row["id"] for row in poses(client)["photos"]} == rows - {gone["public_id"]}
    assert_no_such_photo(client.get(photo_url(gone["public_id"])))
    assert client.get(photo_url(results[1]["public_id"])).status_code == 200
    assert [p["photo_id"] for p in own_photos(client, token).json()["photos"]] == [r["photo_id"] for r in results[1:]]


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
        (res,) = upload(client, ds, ds.entries("good", sector=0)[:1])["results"]
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
        first, second = upload(client, ds, ds.entries("good", sector=0)[:2])["results"]
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
        assert store.get_photo(first["photo_id"]).location_consent == TERMS_CURRENT            # refusals change nothing
        assert withdraw(client, first["photo_id"], first["manage_token"]).status_code == 200
        monkeypatch.setenv(PUBLIC_ID_SALT_ENV, "e" * 40)
        refused(r"consents of 1 photo in")                                        # one is still shared
        monkeypatch.delenv(PUBLIC_ID_SALT_ENV)
        assert withdraw(client, second["photo_id"], second["manage_token"]).status_code == 200
        assert store.get_meta(SALT_CHECK_KEY) == recorded

    monkeypatch.setenv(PUBLIC_ID_SALT_ENV, "e" * 40)
    with caplog.at_level(logging.WARNING, logger="crowd3d.consent"):
        with TestClient(create_app(settings)) as client:
            (res,) = upload(client, ds, ds.entries("good", sector=1)[:1])["results"]
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
        results = upload(client, ds, ds.entries("good", sector=0)[:3])["results"]
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
        (res,) = upload(client, ds, ds.entries("good", sector=1)[:1])["results"]
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
    r = client.post("/api/photos", data=form(client, landmark_id=LM), files=files)
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


def test_stylized_models_are_served_beside_the_point_clouds_and_listed_with_their_registration(client, synthetic_dataset):
    """ModelFiles serves .ply and .glb only; model_info.stylized: pending while a stylize job is queued or running,
    then the GLB's URL with the registration (the model is clay either way, D29; whether the photos' registration was
    accepted is what the client's label says, D32: the API sends no label text)."""
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
    assert d["models"][0]["stylized"] == {"status": "pending", "url": None, "registration": None, "poses_url": None,
                                          "snapped": None}
    assert all(j["kind"] != JOB_STYLIZE for j in d["jobs"])      # reconstruction jobs only in the window
    assert client.get("/api/jobs/styl").json()["model_id"] == model["id"]
    glb = (settings.data_dir / stored.ply_path).with_suffix(".glb")
    glb.write_bytes(b"glTF fake")
    (settings.data_dir / stored.ply_path).with_suffix(".style.json").write_text("{}")
    rel = glb.relative_to(settings.data_dir).as_posix()
    store.set_model_stylized(model["id"], {"glb_path": rel, "registration": {"accepted": False, "reason": "ambiguous"}})
    refused = detail(client)["models"][0]["stylized"]
    assert refused == {"status": "ready", "url": model["url"].replace(".ply", ".glb"),
                       "registration": {"accepted": False, "reason": "ambiguous"},
                       "poses_url": f"/api/landmarks/{LM}/poses",    # the public photo poses, in the GLB's frame
                       "snapped": False}                             # no snap recorded: the cameras did not move
    store.set_model_stylized(model["id"], {"glb_path": rel, "registration": {"accepted": True, "reason": None}})
    assert detail(client)["models"][0]["stylized"]["registration"] == {"accepted": True, "reason": None}
    assert not hasattr(api_module, "STYLIZED_LABEL_ALIGNED") and not hasattr(api_module, "STYLIZED_LABEL_UNALIGNED")
    r = client.get(refused["url"])
    assert r.status_code == 200 and r.content == b"glTF fake"
    for sidecar in (".json", ".style.json"):            # camera poses and the summary stay private
        assert client.get(model["url"].replace(".ply", sidecar)).status_code == 404
    (settings.models_dir / LM / "x.sparse").mkdir()
    assert client.get(f"/files/models/{LM}/x.sparse").status_code == 404


def test_request_bodies_are_capped_before_they_are_parsed(make_client):
    client = make_client(max_upload_bytes=1000)      # cap: 50 files x 1000 bytes + 1 MiB
    big = b"\xff" * (1024 * 1024 + 60_000)
    r = client.post("/api/photos", data={"landmark_id": LM}, files=[("files", ("big.jpg", big, "image/jpeg"))])
    assert r.status_code == 413

    def chunks():                                    # no Content-Length: counted as it streams in
        yield b'--b\r\nContent-Disposition: form-data; name="files"; filename="a.jpg"\r\n\r\n'
        for _ in range(30):
            yield b"x" * 65536

    r = client.post("/api/photos", content=chunks(), headers={"content-type": "multipart/form-data; boundary=b"})
    assert r.status_code == 413
    assert client.get("/api/health").status_code == 200


# Request limits, input echoes and response headers (D34)

AS_JSON = {"content-type": "application/json"}
SMALL = api_module.SMALL_BODY_MAX_BYTES
PARSED_ROUTES = ("/api/landmarks", f"/api/landmarks/{LM}/rebuild", "/api/photos/p/share_location")   # by FastAPI
CONTRIBUTOR_ROUTES = ("/api/contributors", "/api/contributors/accept", "/api/contributors/photos")    # by hand
BASELINE = {"x-content-type-options": "nosniff", "x-frame-options": "DENY",
            "referrer-policy": "strict-origin-when-cross-origin"}


def padded_json(fields, size):
    """fields as JSON of exactly size bytes: trailing spaces are JSON whitespace."""
    body = json.dumps(fields).encode()
    assert len(body) <= size
    return body + b" " * (size - len(body))


def asgi_response(sent):
    """(status, headers as a dict of lists, body) of the messages an ASGI app sent (run_asgi)."""
    start, *rest = sent
    assert start["type"] == "http.response.start"
    headers = {}
    for name, value in start["headers"]:
        headers.setdefault(name.decode().lower(), []).append(value.decode())
    return start["status"], headers, b"".join(m.get("body", b"") for m in rest)


def test_only_the_upload_takes_a_large_body_every_other_request_stops_at_64_kib(client, synthetic_dataset):
    """The body limit is per route (BodySizeLimit): files x size + slack for POST /api/photos alone, 64 KiB for every
    other request. All routes shared the upload's 1.3 GB, and FastAPI reads a JSON body whole into memory before it
    validates it (the review's high finding). Over the limit is a 413 in the existing shape: from Content-Length
    without reading a byte of the body, and for a body without one as soon as what arrived passes the limit; the body
    is never parsed."""
    create_landmark(client)
    assert SMALL == 64 * 1024
    too_large = {"detail": f"request body over {SMALL} bytes"}
    over = padded_json({"name": "x", "lat": 1, "lon": 2, "kind": "full", "share": False, "token": "0" * 64,
                        "nickname": "n", "terms": TERMS_CURRENT}, SMALL + 1)

    def streamed():                                  # no Content-Length
        for i in range(0, len(over), 16 * 1024):
            yield over[i:i + 16 * 1024]

    for route in PARSED_ROUTES + CONTRIBUTOR_ROUTES:
        r = client.post(route, content=over, headers=AS_JSON)
        assert r.status_code == 413 and r.json() == too_large, (route, r.status_code, r.text[:200])
        assert int(r.request.headers["content-length"]) == SMALL + 1
        # decided from the header: the body is not read at all
        chunks = [{"type": "http.request", "body": over, "more_body": False}]
        unread = list(chunks)
        sent = run_asgi_with(client.app, "POST", route, [(b"content-type", b"application/json"),
                                                         (b"content-length", str(len(over)).encode())], unread)
        status, _, body = asgi_response(sent)
        assert status == 413 and json.loads(body) == too_large and unread == chunks, route
    for route in PARSED_ROUTES:
        r = client.post(route, content=streamed(), headers=AS_JSON)
        assert "content-length" not in r.request.headers
        assert r.status_code == 413 and r.json() == too_large, (route, r.status_code, r.text[:200])
        # counted as it arrives: reading stops at the chunk that passes the limit (the fifth of 16 KiB), unparsed
        chunks = [{"type": "http.request", "body": b"[" * (16 * 1024), "more_body": True} for _ in range(10)]
        unread = list(chunks)
        sent = run_asgi_with(client.app, "POST", route, [(b"content-type", b"application/json")], unread)
        status, _, body = asgi_response(sent)
        assert status == 413 and json.loads(body) == too_large and len(unread) == 5, (route, len(unread))
    for route in CONTRIBUTOR_ROUTES:                 # their own reader stops sooner, past 16 KiB, with its own refusal
        chunks = [{"type": "http.request", "body": b"[" * (16 * 1024), "more_body": True} for _ in range(10)]
        sent = run_asgi_with(client.app, "POST", route, [(b"content-type", b"application/json")], chunks)
        status, _, body = asgi_response(sent)
        assert status == 422 and json.loads(body)["detail"][0]["input"] is None and len(chunks) == 8, (route, body)
    # any method, any path: only POST /api/photos is the upload
    for method, path in (("GET", "/api/health"), ("GET", "/"), ("PUT", "/api/photos"), ("POST", "/api/photos/"),
                         ("POST", "/api/photos/x"), ("POST", "/api/nothing-here"), ("DELETE", f"/api/landmarks/{LM}")):
        r = client.request(method, path, content=over, headers=AS_JSON)
        assert r.status_code == 413 and r.json() == too_large, (method, path, r.status_code)
    # a Content-Length that is no number is not trusted: the body is counted (and one of any length is no error)
    for length in (b"abc", b"-1", b"1e3", b"\xb2"):
        chunks = [{"type": "http.request", "body": over, "more_body": False}]
        sent = run_asgi_with(client.app, "POST", "/api/landmarks", [(b"content-type", b"application/json"),
                                                                    (b"content-length", length)], chunks)
        assert asgi_response(sent)[0] == 413 and chunks == [], length
    for length in (b"9" * 5000, b"0" * 30 + str(SMALL + 1).encode(), b" 65537 "):
        chunks = [{"type": "http.request", "body": b"{}", "more_body": False}]
        sent = run_asgi_with(client.app, "POST", "/api/landmarks", [(b"content-type", b"application/json"),
                                                                    (b"content-length", length)], chunks)
        assert asgi_response(sent)[0] == 413 and len(chunks) == 1, length[:20]
    assert [lm["id"] for lm in client.get("/api/landmarks").json()["landmarks"]] == [LM]     # nothing was created

    # at the limit a body is read and answered as before, with and without a Content-Length
    exact = padded_json({"id": "exact", "name": "x", "lat": 1, "lon": 2}, SMALL)
    assert client.post("/api/landmarks", content=exact, headers=AS_JSON).status_code == 201
    exact = padded_json({"id": "exact-streamed", "name": "x", "lat": 1, "lon": 2}, SMALL)
    r = client.post("/api/landmarks", content=iter([exact[:1000], exact[1000:]]), headers=AS_JSON)
    assert r.status_code == 201 and "content-length" not in r.request.headers
    # and the upload itself still takes its files: far more than 64 KiB in one request
    ds = synthetic_dataset
    entries = ds.entries("good", sector=0)
    assert sum(len(ds.read(e)) for e in entries) > 2 * SMALL
    assert all(res["accepted"] for res in upload(client, ds, entries)["results"])


def run_asgi_with(app, method, path, headers, queue):
    """run_asgi, but receive takes its messages off the caller's queue, so what was left unread can be seen."""
    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1", "method": method, "scheme": "http",
             "path": path, "raw_path": path.encode(), "query_string": b"", "root_path": "",
             "headers": [(b"host", b"testserver"), *headers], "client": ("127.0.0.1", 50000),
             "server": ("testserver", 80)}
    sent = []

    async def receive():
        return queue.pop(0) if queue else {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    anyio.run(app, scope, receive, send)
    return sent


def test_the_upload_keeps_its_limit_under_a_root_path(client):
    """Behind a proxy that mounts the app under a prefix (the ASGI root_path), POST <prefix>/api/photos is still the
    upload, and nothing else is."""
    limit = api_module.BodySizeLimit(None, max_bytes=10, upload_max_bytes=1000).limit
    scope = {"type": "http", "method": "POST", "path": "/api/photos", "root_path": ""}
    assert limit(scope) == 1000
    assert limit({**scope, "path": "/crowd/api/photos", "root_path": "/crowd"}) == 1000
    for other in ({"method": "GET"}, {"method": "PUT"}, {"path": "/api/photos/"}, {"path": "/api/photos/p/share_location"},
                  {"path": "/api/landmarks"}, {"path": "/x/api/photos"}, {"path": "/crowd/api/photos", "root_path": "/cro"},
                  {"path": "/crowdapi/photos", "root_path": "/crowd"}):
        assert limit({**scope, **other}) == 10, other


def run_asgi_late_body(app, path, body):
    """Call app directly (ASGI) the way a server does for a GET whose body arrives late: receive gives nothing until
    the application has sent its response start, then these body messages, then nothing more (the client stays to
    read the answer). Returns the messages the app sent."""
    queue, sent = list(body), []
    scope = {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"}, "http_version": "1.1",
             "method": "GET", "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"",
             "root_path": "", "headers": [(b"host", b"testserver"), (b"transfer-encoding", b"chunked")],
             "client": ("127.0.0.1", 50000), "server": ("testserver", 80)}

    async def main():
        started = anyio.Event()

        async def receive():
            await started.wait()
            if queue:
                return queue.pop(0)
            await anyio.sleep_forever()

        async def send(message):
            sent.append(message)
            if message["type"] == "http.response.start":
                started.set()

        with anyio.fail_after(60):
            await app(scope, receive, send)

    anyio.run(main)
    return sent


def test_a_file_is_sent_whole_when_the_requests_body_passes_the_limit_after_the_answer_began(client):
    """Verification of D34's first part: a body without a Content-Length is counted as it arrives, and it can pass
    the limit after the answer began. A file response waits on receive() for the client leaving while it sends the
    file (Starlette's FileResponse; both file mounts), so a GET for a page that carried a chunked body had its 200
    start out, and then the limit sent its 413 start. A server raises on a second start: on a real server 81 of 120
    such requests got a 200 without a byte of the file, each with an unhandled exception in the server's log. The
    answer that began is now finished, by one start and the whole file."""
    settings = client.app.state.settings
    (settings.models_dir / LM).mkdir(parents=True)
    model = b"glTF" + bytes(range(256)) * 1024                     # 256 KiB: sent in several pieces
    (settings.models_dir / LM / "m.glb").write_bytes(model)
    files = {"/": (WEB_DIR / "index.html").read_bytes(), "/style.css": (WEB_DIR / "style.css").read_bytes(),
             "/vendor/three/build/three.module.js": (WEB_DIR / "vendor/three/build/three.module.js").read_bytes(),
             f"/files/models/{LM}/m.glb": model}
    for path, data in files.items():
        for first in (SMALL + 1, 1024):                             # over the limit at once, or at the second message
            body = [{"type": "http.request", "body": b"a" * first, "more_body": True},
                    {"type": "http.request", "body": b"a" * SMALL, "more_body": True},
                    {"type": "http.request", "body": b"", "more_body": False}]
            sent = run_asgi_late_body(client.app, path, body)
            assert [m["type"] for m in sent[1:]] == ["http.response.body"] * (len(sent) - 1), (path, first)
            status, headers, answer = asgi_response(sent)
            assert status == 200 and answer == data and headers["content-length"] == [str(len(data))], (path, first)
            assert sent[-1].get("more_body", False) is False, (path, first)


def test_the_body_limit_after_the_response_start_drops_what_is_over_it():
    """BodySizeLimit around scripted applications. Before the response start nothing changed: a body over the limit
    is the 413, once, whether the application lets the refusal through or answers something of its own. From the
    response start on (verification of D34's first part) there is no 413 to send: the body that is over the limit is
    read and dropped, the application gets what was within the limit and then the client's leaving, and what it
    sends goes out as it is. The start counts as sent from the moment the application hands it over, also while the
    server makes that send wait."""
    limited = lambda app: api_module.BodySizeLimit(app, max_bytes=10, upload_max_bytes=1000)   # noqa: E731
    chunk = lambda size, more=True: {"type": "http.request", "body": b"a" * size, "more_body": more}   # noqa: E731
    start = {"type": "http.response.start", "status": 200, "headers": [(b"content-length", b"2")]}
    end = {"type": "http.response.body", "body": b"ok"}
    gone = {"type": "http.disconnect"}
    too_large = {"detail": "request body over 10 bytes"}
    heard = []

    async def answers_first(scope, receive, send):
        await send(start)
        heard.append(await receive())
        heard.append(await receive())
        await send(end)

    queue = [chunk(6), chunk(5), chunk(100), chunk(0, more=False)]
    assert run_asgi_with(limited(answers_first), "GET", "/x", [], queue) == [start, end]
    assert heard == [chunk(6), gone] and queue == []          # 6 bytes handed on; 5 + 100 + the end read and dropped
    heard.clear()
    queue = [chunk(4), chunk(6), chunk(0, more=False)]         # at the limit: nothing is dropped
    assert run_asgi_with(limited(answers_first), "GET", "/x", [], queue) == [start, end]
    assert heard == [chunk(4), chunk(6)] and len(queue) == 1

    async def reads_first(scope, receive, send):
        try:
            await receive()
        except Exception:                                      # as FastAPI's body parsing: an answer of its own
            await send({"type": "http.response.start", "status": 400, "headers": []})
            await send({"type": "http.response.body", "body": b"bad"})
            raise

    for app in (reads_first, lambda scope, receive, send: receive()):
        sent = run_asgi_with(limited(app), "GET", "/x", [], [chunk(11)])
        status, _, body = asgi_response(sent)
        assert status == 413 and json.loads(body) == too_large and len(sent) == 2

    # the start is with the server, which makes the application wait in send (its write buffer is full), when the
    # body passes the limit: a 413 now would be a second start
    sent = []

    async def start_in_flight():
        taken, released = anyio.Event(), anyio.Event()
        queue = [chunk(11), chunk(50)]

        async def receive():
            await taken.wait()
            if queue:
                return queue.pop(0)
            released.set()
            return gone

        async def send(message):
            sent.append(message)
            if len(sent) == 1:
                taken.set()
                await released.wait()

        async def app(scope, receive, send):
            async def answer():
                await send(start)
                await send(end)

            async with anyio.create_task_group() as tasks:
                tasks.start_soon(answer)
                heard.append(await receive())

        with anyio.fail_after(60):
            await limited(app)({"type": "http", "method": "GET", "path": "/x", "headers": []}, receive, send)

    heard.clear()
    anyio.run(start_in_flight)
    assert sent == [start, end] and heard == [gone]


def test_a_422_never_carries_back_more_than_a_small_input(client):
    """Whatever the body, each error of a 422 echoes an input of at most MAX_ECHO_BYTES of JSON, else null, so the
    response is a few KiB at most. FastAPI echoes the whole body once per missing field (the review measured a 6.3 MB
    answer to a 2 MB body); the cap is on the bytes the response would carry, not on characters."""
    create_landmark(client)
    cap = api_module.MAX_ECHO_BYTES
    assert cap == 1024
    big = SMALL - 2000                                # near the body limit
    deep = "[" * 5000 + '"x"' + "]" * 5000
    bodies = [
        ("extra field, long value", json.dumps({"pad": "y" * big})),
        ("many extra fields", json.dumps({f"k{i}": i for i in range(4000)})),
        ("long values in every field", json.dumps({k: "v" * (big // 10) for k in (
            "id", "name", "lat", "lon", "radius_m", "kind", "sector", "share", "token")})),
        ("a kilobyte in every field", json.dumps({k: "v" * 1000 for k in (
            "id", "name", "lat", "lon", "radius_m", "kind", "sector", "share", "token")})),
        ("a long list in every field", json.dumps({k: list(range(1000)) for k in (
            "id", "name", "lat", "lon", "radius_m", "kind", "sector", "share", "token")})),
        ("deep nesting", deep),
        ("deep nesting in a field", f'{{"name": {deep}, "kind": {deep}, "token": {deep}}}'),
        ("a long string, not an object", json.dumps("s" * big)),
        ("a long list, not an object", json.dumps(list(range(9000)))),
        ("not JSON", "{" + "x" * big),
        ("long keys", json.dumps({"k" * 20000: 1, "j" * 20000: 2})),
        ("long CJK values", json.dumps({"name": "塔" * 9000, "kind": "塔" * 9000}, ensure_ascii=False)),
    ]
    largest = 0
    for route in PARSED_ROUTES:
        for what, body in bodies:
            assert len(body.encode()) <= SMALL, what
            r = client.post(route, content=body.encode(), headers=AS_JSON)
            assert r.status_code == 422, (route, what, r.status_code, r.text[:200])
            for err in r.json()["detail"]:
                echo = json.dumps(err["input"], ensure_ascii=False).encode()
                assert len(echo) <= cap, (route, what, err["type"], len(echo))
                assert set(err) <= {"type", "loc", "msg", "input", "ctx"}
            largest = max(largest, len(r.content))
    assert 3000 < largest <= 8192, largest            # the bound of any 422 here; reached by a kilobyte per field
    deeper = "[" * 30000 + "]" * 30000                # too deep for the JSON parser: FastAPI's 400, a fixed text
    for route in PARSED_ROUTES:
        r = client.post(route, content=deeper, headers=AS_JSON)
        assert r.status_code == 400 and r.json() == {"detail": "There was an error parsing the body"}, route
    # the contributor routes: a nickname is echoed while it is short
    for nickname, echoed in (("n" * 65, True), ("n" * (cap - 2), True), ("n" * (cap - 1), False),
                             ("n" * 15000, False), ("塔" * 340, True), ("塔" * 341, False)):
        r = client.post("/api/contributors", json={"nickname": nickname, "terms": TERMS_CURRENT})
        (err,) = r.json()["detail"]
        assert r.status_code == 422 and err["input"] == (nickname if echoed else None), (len(nickname), err["input"])
        assert len(r.content) < cap + 300
    # the cap is exact, and a small input is echoed as before
    for name, echoed in (("n" * (cap - 2), True), ("n" * (cap - 1), False)):       # as JSON: the quotes count
        r = client.post("/api/landmarks", json={"name": name, "lat": 1, "lon": 2})
        (err,) = r.json()["detail"]
        assert (err["type"], err["input"]) == ("string_too_long", name if echoed else None), len(name)
    r = client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "bogus", "pad": "y" * big})
    assert r.status_code == 422 and [e["input"] for e in r.json()["detail"]] == ["bogus"]
    r = client.post("/api/landmarks", json={"lat": 91, "lon": "east", "name": "x"})
    assert {e["loc"][1]: e["input"] for e in r.json()["detail"]} == {"lat": 91, "lon": "east"}
    r = client.post("/api/landmarks", json={"lat": 1, "lon": 2})
    assert [e["input"] for e in r.json()["detail"]] == [{"lat": 1, "lon": 2}]      # a missing field: the small body
    assert [lm["id"] for lm in client.get("/api/landmarks").json()["landmarks"]] == [LM]


# A server's memory is bounded by the decode limit, not by its worker threads (D34, review round 1)


def test_the_server_pins_the_malloc_threshold_when_it_starts(settings, monkeypatch):
    """create_app's lifespan pins glibc's mmap threshold at Settings.malloc_mmap_threshold_bytes
    (ingest.pin_malloc_threshold); 0 leaves the allocator alone (the tests' own settings)."""
    from crowd3d import ingest as ingest_module

    calls = []
    monkeypatch.setattr(api_module, "pin_malloc_threshold", lambda threshold: calls.append(threshold) or True)
    assert settings.malloc_mmap_threshold_bytes == 0
    app = create_app(replace(settings, malloc_mmap_threshold_bytes=4 * 1024 * 1024), FakeReconstructor())
    assert calls == []                                          # not when the app is made: when the server starts
    with TestClient(app) as c:
        assert c.get("/api/health").status_code == 200
    assert calls == [4 * 1024 * 1024]
    assert ingest_module._malloc_pinned_at is None              # and no test has pinned this process


def process_mb(pid, key):
    """VmRSS or VmHWM (the peak) of a process, in MB."""
    for line in Path(f"/proc/{pid}/status").read_text().splitlines():
        if line.startswith(key + ":"):
            return int(line.split()[1]) / 1024
    raise KeyError(key)


def server_request(port, method, path, body=None, content_type="application/json"):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=120)
    try:
        connection.request(method, path, body=body, headers={"Content-Type": content_type} if body else {})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


@contextmanager
def real_server(tmp_path):
    """A real server process on a data directory of its own (python -m crowd3d.api, the fake reconstructor), with a
    registered contributor: (port, pid, contributor token). Stopped when the block ends."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    env = {k: v for k, v in os.environ.items() if k != "MALLOC_MMAP_THRESHOLD_"} | {"CROWD3D_STRUCTURE_CATALOG": ""}
    with open(tmp_path / "server.log", "wb") as log:
        server = subprocess.Popen([sys.executable, "-m", "crowd3d.api", "--port", str(port), "--data-dir",
                                   str(tmp_path / "data"), "--reconstructor", "fake"],
                                  cwd=Path(api_module.__file__).resolve().parent.parent, env=env, stdout=log,
                                  stderr=subprocess.STDOUT)
    try:
        deadline = time.monotonic() + 60
        while True:
            try:
                assert server_request(port, "GET", "/api/health")[0] == 200
                break
            except OSError:
                assert server.poll() is None and time.monotonic() < deadline, (tmp_path / "server.log").read_text()
                time.sleep(0.1)
        status, registered = server_request(port, "POST", "/api/contributors",
                                            json.dumps({"nickname": "memory", "terms": TERMS_CURRENT}))
        assert status == 201
        yield port, server.pid, registered["contributor_token"]
    finally:
        server.terminate()
        try:
            server.wait(timeout=20)
        except subprocess.TimeoutExpired:
            server.kill()


def upload_at_once(port, token, files, rounds=1):
    """Each of these files as an upload of its own, all at once, rounds times over: the first result of each
    answer, in the files' order."""
    boundary = "crowd3d-memory-test"
    bodies = [b"".join([f'--{boundary}\r\nContent-Disposition: form-data; name="contributor_token"\r\n\r\n{token}\r\n'
                        f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="f{i}"\r\n'
                        f'Content-Type: application/octet-stream\r\n\r\n'.encode(), data,
                        f"\r\n--{boundary}--\r\n".encode()]) for i, data in enumerate(files)]
    results = []
    for _ in range(rounds):
        answers = [None] * len(bodies)

        def upload(i):
            answers[i] = server_request(port, "POST", "/api/photos", bodies[i],
                                        f"multipart/form-data; boundary={boundary}")

        threads = [threading.Thread(target=upload, args=(i,), daemon=True) for i in range(len(bodies))]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=120)
        assert not any(t.is_alive() for t in threads)
        assert all(answer is not None and answer[0] == 200 for answer in answers), answers[:2]
        results += [answer[1]["results"][0] for answer in answers]
    return results


GLIBC_ON_LINUX = pytest.mark.skipif(sys.platform != "linux" or platform.libc_ver()[0] != "glibc",
                                    reason="about glibc's allocator; reads the server's memory from /proc")


@GLIBC_ON_LINUX
def test_a_real_servers_memory_follows_the_decode_limit_not_the_number_of_uploads(tmp_path):
    """The review's finding, on a real server process (python -m crowd3d.api, uvicorn's worker threads): the limit
    of two decodes at once held, but each thread that had decoded a large PNG kept its memory in a heap of its own,
    so the server's memory followed the number of threads that had ever decoded and stayed when it was idle. The
    count of decodes in flight (tests/test_ingest.py) cannot see that; this reads the server's resident memory.
    Three rounds of 16 uploads at once of a flat 16 MP PNG, from 65 MB at the start: 364 MB at the peak and 82 MB
    idle afterwards with glibc's mmap threshold pinned (Settings.malloc_mmap_threshold_bytes); 1455-1612 MB and
    1262-1366 MB without (three runs each). Since review round 2 (the metadata read in a slot too, freed memory
    handed back after each slot): 239-240 MB and 71 MB (three runs)."""
    buf = io.BytesIO()
    Image.new("RGB", (4800, 3400), (120, 130, 140)).save(buf, "PNG", compress_level=1)
    with real_server(tmp_path) as (port, pid, token):
        start = process_mb(pid, "VmRSS")
        results = upload_at_once(port, token, [buf.getvalue()] * 16, rounds=3)
        assert len(results) == 48 and all(r["reasons"] == ["no_exif", "no_gps", "too_blurry"] for r in results)
        time.sleep(0.5)
        peak, idle = process_mb(pid, "VmHWM"), process_mb(pid, "VmRSS")
        print(f"server memory: {start:.0f} MB at the start, {peak:.0f} MB at the peak, {idle:.0f} MB idle afterwards")
        assert peak - start < 600, (start, peak, idle)       # two decodes at once, not sixteen
        assert idle - start < 150, (start, peak, idle)       # and their memory went back to the operating system


@GLIBC_ON_LINUX
def test_a_real_servers_memory_does_not_follow_what_uploads_put_in_their_headers(tmp_path):
    """Review round 2, on a real server process. What an upload costs before and besides its pixels follows the
    decode limit too: reading its header (Pillow keeps Python objects for every segment and chunk; four uploads at
    once of a 24 MB JPEG of fat frame headers took the server to 5.4 GB), and what the worker threads keep of
    memory that was freed in blocks under the pinned threshold (a 69 KB PNG whose 64 text chunks inflate to 1 MiB
    each left 64 MiB in every thread: 1.1 GB idle after 16 at once, 2.3 GB after 40 more).
    Two rounds of 24 uploads at once of 2 MB header floods (hostile.header_floods: 0.5 million empty segments, fat
    frame headers, megabytes of EXIF numbers in a JPEG, empty PNG and WebP chunks, the 69 KB PNG), then three rounds
    of 16 PNGs whose text is within the limits (16 chunks of 1 MiB). From 65 MB at the start: 101-104 MB at the
    peak and 67-68 MB idle after the floods, 126-128 MB and 72 MB after the text (three runs). With the metadata
    read outside the decode slot the text takes 335-337 MB at the peak; without the hand-back of freed memory
    (ingest._release_freed_memory) 343-349 MB, and 293-311 MB stay when the server is idle (two runs each)."""
    import hostile_images as hostile

    floods = hostile.header_floods(2)
    names = ["jpeg: empty APP segments", "jpeg: fat frame headers", "jpeg: EXIF with megabytes of numbers",
             "png: private chunks", "png: compressed text of 64 MiB", "webp: empty chunks"]
    flood_files = [floods[name]() for name in names] * 4
    within = hostile.png_with_chunks(b"".join(
        hostile.png_chunk(b"zTXt", b"k%d\0\0" % i + zlib.compress(b"A" * 1_048_000, 9)) for i in range(16)))
    with real_server(tmp_path) as (port, pid, token):
        start = process_mb(pid, "VmRSS")
        results = upload_at_once(port, token, flood_files, rounds=2)
        assert len(results) == 48 and all(r["reasons"] == ["unreadable_image"] for r in results), results[:6]
        time.sleep(0.5)
        flood_peak, flood_idle = process_mb(pid, "VmHWM"), process_mb(pid, "VmRSS")
        results = upload_at_once(port, token, [within] * 16, rounds=3)
        assert len(results) == 48 and all(r["reasons"] == ["no_exif", "no_gps"] for r in results), results[:2]
        time.sleep(0.5)
        peak, idle = process_mb(pid, "VmHWM"), process_mb(pid, "VmRSS")
        print(f"server memory: {start:.0f} MB at the start; header floods: {flood_peak:.0f} MB at the peak, "
              f"{flood_idle:.0f} MB idle; text within the limits: {peak:.0f} MB at the peak, {idle:.0f} MB idle")
        assert flood_peak - start < 120 and flood_idle - start < 40, (start, flood_peak, flood_idle)
        assert peak - start < 150, (start, peak, idle)       # two uploads' text at once, not sixteen
        assert idle - start < 40, (start, peak, idle)        # and none of it stays in the worker threads


def get_with_a_chunked_body(port, path, body_chunks, late):
    """GET path on a raw socket with a chunked body of body_chunks pieces of 32 KiB. late: the body is sent once the
    head of the answer is here (the server began its answer without having seen any of the body); else with the
    request's head. Returns (status, body of the answer). A server that closes with body left unread resets the
    connection: what it answered before that is still read."""
    piece = b"a" * (32 * 1024)
    body = (b"%x\r\n" % len(piece) + piece + b"\r\n") * body_chunks + b"0\r\n\r\n"
    head = f"GET {path} HTTP/1.1\r\nHost: test\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n".encode()
    answer = b""
    with socket.create_connection(("127.0.0.1", port), timeout=60) as sock:

        def send(data):
            try:
                sock.sendall(data)
            except OSError:                  # the server has answered and closed
                pass

        send(head if late else head + body)
        while True:
            if late and b"\r\n\r\n" in answer:
                send(body)
                late = False
            try:
                data = sock.recv(1 << 16)
            except ConnectionResetError:
                break
            if not data:
                break
            answer += data
    head, _, content = answer.partition(b"\r\n\r\n")
    return int(head.split()[1]), content


def test_a_real_server_sends_the_file_whole_when_the_body_passes_the_limit_after_the_answer_began(tmp_path):
    """Verification of D34's first part, on a real server process (uvicorn's own checks of what an application
    sends): a GET for a file with a chunked body over the 64 KiB limit. When the body arrives after the answer began
    (late), the server used to log an unhandled exception ("Expected ASGI message 'http.response.body', but got
    'http.response.start'", the limit's 413 after the file's 200) and the client got the 200 without the file: 160
    of 160 such requests to four files. Now the file arrives whole and the log stays free of errors. When the body
    is sent with the request, the answer is the 413 or the whole file, by which came first."""
    model = b"glTF" + bytes(range(256)) * 4096                     # 1 MiB
    script = "/vendor/three/build/three.module.js"
    too_large = {"detail": f"request body over {SMALL} bytes"}
    with real_server(tmp_path) as (port, _, _):
        (tmp_path / "data" / "models" / LM).mkdir(parents=True)
        (tmp_path / "data" / "models" / LM / "m.glb").write_bytes(model)
        files = {script: (WEB_DIR / script[1:]).read_bytes(), f"/files/models/{LM}/m.glb": model,
                 "/": (WEB_DIR / "index.html").read_bytes()}
        for path, data in files.items():
            for _ in range(4):
                status, content = get_with_a_chunked_body(port, path, 3, late=True)
                assert status == 200 and content == data, (path, status, len(content), len(data))
            for _ in range(4):
                status, content = get_with_a_chunked_body(port, path, 3, late=False)
                assert (status == 200 and content == data) or (status == 413 and json.loads(content) == too_large), \
                    (path, status, len(content), len(data))
        assert server_request(port, "GET", "/api/health")[0] == 200
        time.sleep(0.2)
        log = (tmp_path / "server.log").read_text()
    assert log.count(f"GET {script} HTTP/1.1") == 8, log[-2000:]
    assert not any(word in log for word in ("ERROR", "Exception", "Traceback")), log[-4000:]


# The upload's form is read within its own limits (D34, review round 1)

BOUNDARY = b"crowd3d-test-boundary"
MULTIPART = (b"content-type", b"multipart/form-data; boundary=" + BOUNDARY)
FORM_END = b"--" + BOUNDARY + b"--\r\n"
FORM_TOO_BIG = {"detail": "upload form over 8 text fields, or a text field over 4096 bytes"}
JPG = b"\xff\xd8\xff\xd9"


def form_part(name, value, filename=None):
    """One part of a multipart form: a text field, or a file part when it has a file name."""
    head = f'form-data; name="{name}"' + ("" if filename is None else f'; filename="{filename}"')
    lines = [b"--" + BOUNDARY, b"Content-Disposition: " + head.encode()]
    if filename is not None:
        lines.append(b"Content-Type: image/jpeg")
    return b"\r\n".join(lines) + b"\r\n\r\n" + value + b"\r\n"


def post_form(app, chunks, header=MULTIPART):
    """POST /api/photos with a body sent in these chunks and no Content-Length: (status, the JSON answer, how many
    chunks the server had not read when it answered)."""
    unread = [{"type": "http.request", "body": chunk, "more_body": i + 1 < len(chunks)}
              for i, chunk in enumerate(chunks)]
    status, _, body = asgi_response(run_asgi_with(app, "POST", "/api/photos", [header], unread))
    return status, json.loads(body), len(unread)


def test_the_upload_form_takes_eight_text_fields_of_4_kib(client):
    """The review's finding: Starlette keeps a form's text fields in memory, up to 1000 of 1 MiB, and parses the
    form before the route sees a token, so one request without a token held 1 GB (and the process kept it). The
    upload's form now takes UPLOAD_MAX_FIELDS text fields of UPLOAD_FIELD_MAX_BYTES each; over either, 413, whoever
    sends it. At the limits the upload goes its usual way."""
    assert (api_module.UPLOAD_MAX_FIELDS, api_module.UPLOAD_FIELD_MAX_BYTES) == (8, 4096)
    create_landmark(client)
    token = contributor_token(client)
    one = [("files", ("a.jpg", JPG, "image/jpeg"))]
    fields = {"contributor_token": token, "landmark_id": LM}
    extra = {f"extra{i}": "x" * 4096 for i in range(6)}                 # eight fields, six of them at the size limit
    r = client.post("/api/photos", data=fields | extra, files=one)
    assert r.status_code == 200 and r.json()["results"][0]["reasons"] == ["unreadable_image"], r.text
    for data in (fields | extra | {"ninth": ""}, fields | {"big": "x" * 4097}, {"big": "x" * 4097},
                 {f"f{i}": "" for i in range(9)}, fields | {"landmark_id": "x" * 4097},
                 {"contributor_token": "0" * 4097}):
        r = client.post("/api/photos", data=data, files=one)
        assert r.status_code == 413 and r.json() == FORM_TOO_BIG, (sorted(data), r.status_code, r.text[:200])
        assert r.headers["x-content-type-options"] == "nosniff"
    # a form that is no multipart form at all stays what it was (and names no limit): Starlette's 400
    status, answer, _ = post_form(client.app, [b"no boundary here"], (b"content-type", b"multipart/form-data"))
    assert (status, answer) == (400, {"detail": "Missing boundary in multipart."})
    assert len(client.app.state.store.list_photos(LM)) == 1             # the one upload at the limits, rejected


def test_a_gigabyte_of_form_text_is_refused_at_its_first_kilobytes(client):
    """The attack itself, as the server sees it: a body of text fields sent in 64 KiB chunks. The parser stops in
    the chunk that takes a field over 4 KiB, or that brings the ninth field, and the answer goes out with the rest
    of the body unread: nothing of it is held. A form-encoded body holds no file and is not read at all (Starlette
    parses those into memory as well)."""
    chunk = b"x" * (64 * 1024)
    megabyte_fields = [form_part("f0", b"")[:-2]] + [chunk] * 16000          # a 1 GB field, were it all read
    status, answer, unread = post_form(client.app, megabyte_fields)
    assert (status, answer) == (413, FORM_TOO_BIG) and unread == 16000 - 1, (status, answer, unread)
    many = [b"".join(form_part(f"f{i}", b"v") for i in range(1000))] * 100   # 100,000 small fields
    status, answer, unread = post_form(client.app, many)
    assert (status, answer) == (413, FORM_TOO_BIG) and unread == 99, (status, answer, unread)
    form_encoded = (b"content-type", b"application/x-www-form-urlencoded")
    for body in ([b"a=" + chunk] + [chunk] * 16000, [b"&".join(b"f%d=v" % i for i in range(1000))] * 100):
        status, answer, unread = post_form(client.app, body, form_encoded)
        assert status == 422 and unread == len(body), (status, unread)
        assert answer == {"detail": [{"type": "missing", "loc": ["body", "files"], "msg": "Field required",
                                      "input": None}]}
    assert nothing_stored(client)


def test_the_upload_form_stops_at_the_51st_file(client):
    """The parser stops at the file part over MAX_FILES_PER_UPLOAD too (Starlette read 1000, each under 1 MiB kept
    in memory), with the rest unread, and the answer is the one such an upload always got, in the same order, from
    the text fields read by then: register_required or terms_outdated for the token, else the file count. A token
    after the 51st file is therefore never read: clients send it first (this project's page does: web/api.js, pinned
    in web/tests/api.test.mjs)."""
    create_landmark(client)
    token = contributor_token(client)
    files = [b"".join(form_part("files", JPG, f"{i}.jpg") for i in range(n, n + 10)) for n in range(0, 1000, 10)]
    token_part = form_part("contributor_token", token.encode())
    too_many = {"type": "value_error", "loc": ["body", "files"], "msg": "upload 1-50 files at a time", "input": None}
    unregistered = {"type": "register_required", "loc": ["body", "contributor_token"], "input": None,
                    "msg": "uploading needs a registered contributor (POST /api/contributors)"}
    for chunks, error, read in (([token_part] + files, too_many, 7),             # the token, then 51 files
                                (files + [token_part], unregistered, 6),         # the token after them: never read
                                (files, unregistered, 6),
                                ([form_part("contributor_token", b"0" * 64)] + files, unregistered, 7),
                                ([token_part, form_part("contributor_token", b"0" * 64)] + files, unregistered, 8),
                                ([form_part("contributor_token", b"x"), token_part] + files, too_many, 8)):
        status, answer, unread = post_form(client.app, chunks + [FORM_END])
        assert (status, answer) == (422, {"detail": [error]}), (status, answer)
        assert unread == len(chunks) + 1 - read, (unread, read)
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0'")
    status, answer, _ = post_form(client.app, [token_part] + files + [FORM_END])
    assert status == 422 and [e["type"] for e in answer["detail"]] == ["terms_outdated"]
    assert token not in json.dumps(answer) and nothing_stored(client)
    # 50 file parts are read, and the route answers
    with sqlite3.connect(client.app.state.settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = ?", (TERMS_CURRENT,))
    status, answer, unread = post_form(client.app, [token_part] + files[:5] + [FORM_END])
    assert status == 200 and len(answer["results"]) == 50 and unread == 0


def test_a_small_file_part_stays_in_memory_up_to_64_kib(client):
    """A file part is kept in memory up to UPLOAD_SPOOL_BYTES and goes to a temporary file above it (Starlette's own
    limit is 1 MiB: 50 parts just under it were 50 MB of one request in memory)."""
    assert api_module.UPLOAD_SPOOL_BYTES == 64 * 1024

    async def rolled(size):
        body = form_part("files", b"x" * size, "a.jpg") + FORM_END
        queue = [{"type": "http.request", "body": body, "more_body": False}]

        async def receive():
            return queue.pop(0)

        scope = {"type": "http", "method": "POST", "path": "/api/photos", "headers": [MULTIPART], "query_string": b""}
        request = api_module.UploadRequest(scope, receive)
        form = await request.form(max_files=1000, max_fields=1000, max_part_size=10 ** 9)   # the upload's own apply
        assert await request.form() is form
        (upload,) = form.getlist("files")
        try:
            return upload.size, upload.file._rolled
        finally:
            await form.close()

    assert anyio.run(rolled, 64 * 1024) == (64 * 1024, False)
    assert anyio.run(rolled, 64 * 1024 + 1) == (64 * 1024 + 1, True)


def test_a_404_quotes_at_most_80_characters_of_what_was_sent(client, synthetic_dataset):
    """An unknown landmark or job is named in the 404 (its id is short); a longer string is cut, so the answer to a
    long path, or to the longest landmark_id the upload's form takes (4 KiB; a longer one is a 413), is short."""
    ds = synthetic_dataset
    known = "a" * 64                                   # the longest landmark id there is
    r = client.get(f"/api/landmarks/{known}")
    assert r.status_code == 404 and r.json() == {"detail": f"unknown landmark '{known}'"}
    long = "b" * 3000
    cut = "b" * 80 + "..."
    for url, detail in ((f"/api/landmarks/{long}", f"unknown landmark '{cut}'"),
                        (f"/api/landmarks/{long}/poses", f"unknown landmark '{cut}'"),
                        (f"/api/jobs/{long}", f"unknown job '{cut}'")):
        r = client.get(url)
        assert r.status_code == 404 and r.json() == {"detail": detail}, url
    r = client.post(f"/api/landmarks/{long}/rebuild", json={"kind": "full"})
    assert r.status_code == 404 and r.json() == {"detail": f"unknown landmark '{cut}'"}
    entry = ds.entries("good")[0]
    one = [("files", (entry["file"], ds.read(entry), "image/jpeg"))]
    r = client.post("/api/photos", data=form(client, landmark_id="c" * api_module.UPLOAD_FIELD_MAX_BYTES), files=one)
    assert r.status_code == 404 and r.json() == {"detail": "unknown landmark '" + "c" * 80 + "...'"}
    assert len(r.content) < 200
    r = client.post("/api/photos", data=form(client, landmark_id="c" * 500_000), files=one)
    assert r.status_code == 413 and len(r.content) < 200


def test_every_response_carries_the_baseline_headers(settings, synthetic_dataset, monkeypatch):
    """BaselineHeaders: X-Content-Type-Options nosniff, X-Frame-Options DENY, Referrer-Policy
    strict-origin-when-cross-origin and a Content-Security-Policy on every response, once each: JSON and pages,
    errors of every kind (the body limit's 413 and an unhandled error's 500 among them), a 304, a model file and a
    served photo. The policy is the strict one (csp.STRICT_POLICY: nothing may be loaded) on everything but the
    site's pages, which carry their own (tests/test_csp.py), on a 304 too. FastAPI's /docs and /redoc, pages that
    load scripts from a CDN, are not served. The Referrer-Policy sends other sites the origin only; it is not
    no-referrer or same-origin because the home page's map tiles come from tile.openstreetmap.org, whose tile usage
    policy asks pages not to strip the Referer."""
    ds = synthetic_dataset
    with TestClient(create_app(settings), raise_server_exceptions=False) as client:
        create_landmark(client, ds)
        results = upload(client, ds, ds.entries("good", sector=0))["results"]
        model = detail(client)["models"][0]
        row = client.get(f"/api/landmarks/{LM}/poses").json()["photos"][0]
        etag = client.get("/api.js").headers["etag"]
        got = {
            "page": client.get("/"),
            "page 304": client.get("/index.html", headers={"if-none-match": client.get("/index.html").headers["etag"]}),
            "script": client.get("/api.js"),
            "style": client.get("/style.css"),
            "theme": client.get("/themes/default.json"),
            "library": client.get("/vendor/three/build/three.module.js"),
            "library image": client.get("/vendor/leaflet/images/marker-icon.png"),
            "304": client.get("/api.js", headers={"if-none-match": etag}),
            "json": client.get("/api/health"),
            "201": register(client, "bob"),
            "202": client.post(f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}),
            "403": client.post(f"/api/photos/{results[0]['photo_id']}/share_location",
                               json={"share": False, "token": "0" * 64}),
            "403 by registration": client.post(f"/api/photos/{results[0]['photo_id']}/share_location",
                                               json={"share": False, "contributor_token": "0" * 64}),
            "own photos": client.post("/api/contributors/photos",
                                      json={"contributor_token": contributor_token(client, "alice")}),
            "422 own photos": client.post("/api/contributors/photos", json={}),
            "404 api": client.get("/api/landmarks/nope"),
            "404 static": client.get("/nope.js"),
            "404 tests": client.get("/tests/api.test.mjs"),
            "405": client.put("/api/health"),
            "405 static": client.post("/style.css"),
            "409": client.post("/api/landmarks", json={"id": LM, "name": "x", "lat": 0, "lon": 0}),
            "413": client.post("/api/landmarks", content=b" " * (SMALL + 1), headers=AS_JSON),
            "422": client.post("/api/landmarks", json={}),
            "422 upload": client.post("/api/photos", data={"landmark_id": LM}),
            "model": client.get(model["url"]),
            "photo": client.get(photo_url(row["id"])),
            "photo 304": client.get(photo_url(row["id"]), headers={"if-none-match": f'W/"{row["id"]}.1"'}),
            "photo 404": client.get(photo_url("0" * 12)),
            "openapi": client.get("/openapi.json"),
            "docs": client.get("/docs"),
            "redoc": client.get("/redoc"),
            "docs redirect": client.get("/docs/oauth2-redirect"),
        }
        monkeypatch.setattr(client.app.state.store, "list_landmarks", lambda: 1 / 0)
        got["500"] = client.get("/api/landmarks")
    statuses = {what: r.status_code for what, r in got.items()}
    assert statuses == {"page": 200, "page 304": 304, "script": 200, "style": 200, "theme": 200, "library": 200,
                        "library image": 200, "304": 304, "json": 200, "201": 201,
                        "202": 202, "403": 403, "403 by registration": 403, "own photos": 200, "422 own photos": 422,
                        "404 api": 404, "404 static": 404, "404 tests": 404, "405": 405,
                        "405 static": 405, "409": 409, "413": 413, "422": 422, "422 upload": 422,
                        "model": 200, "photo": 200, "photo 304": 304, "photo 404": 404, "openapi": 200,
                        "docs": 404, "redoc": 404, "docs redirect": 404, "500": 500}
    home_policy = csp.page_policy((WEB_DIR / "index.html").read_text(encoding="utf-8"))
    assert home_policy.startswith("default-src 'self'; ") and home_policy != csp.STRICT_POLICY
    for what, r in got.items():
        for name, value in BASELINE.items():
            assert r.headers.get_list(name) == [value], (what, name, r.headers.get_list(name))
        policy = home_policy if what in ("page", "page 304") else csp.STRICT_POLICY
        assert r.headers.get_list("content-security-policy") == [policy], (what, r.headers.get_list(csp.HEADER))
    assert csp.STRICT_POLICY == ("default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; "
                                 "frame-ancestors 'none'")
    assert api_module.BASELINE_HEADERS == (*((k.encode(), v.encode()) for k, v in BASELINE.items()),
                                           (b"content-security-policy", csp.STRICT_POLICY.encode()))
    for path in ("/docs", "/redoc"):       # no route at all: the 404 is the static files' own
        assert got[path.strip("/")].json() == {"detail": "Not Found"}
    # The test client accepts gzip, as a browser does: only the two file mounts compress, and only text and models
    # (D34, tests/test_gzipfiles.py). No route's JSON is compressed (the answers that hold tokens among them), nor
    # an error, nor the served photo, which is a JPEG already; and nothing else varies with Accept-Encoding.
    assert "gzip" in client.headers["accept-encoding"]
    compressed = {what for what, r in got.items() if "content-encoding" in r.headers}
    assert compressed == {"page", "script", "style", "library", "model"}
    assert all(got[what].headers["content-encoding"] == "gzip" for what in compressed)
    assert {what for what, r in got.items() if "vary" in r.headers} == compressed | {"page 304", "304"}
    assert got["photo"].content[:2] == b"\xff\xd8" and got["photo"].headers["content-length"] == str(
        len(got["photo"].content))
    # what a client that decodes ends up with is the file, byte for byte
    assert got["model"].content == (settings.models_dir / LM / f"{model['id']}.ply").read_bytes()
    assert got["library"].content == (WEB_DIR / "vendor" / "three" / "build" / "three.module.js").read_bytes()
    assert int(got["library"].headers["content-length"]) < len(got["library"].content) // 4
    assert BASELINE["referrer-policy"] not in ("no-referrer", "same-origin")       # the map tiles need a Referer
    assert "tile.openstreetmap.org" in (WEB_DIR / "index.js").read_text(encoding="utf-8")
    # nosniff makes the declared types binding: scripts and styles must be served as what they are
    assert got["script"].headers["content-type"].split(";")[0] in ("text/javascript", "application/javascript")
    assert got["style"].headers["content-type"].split(";")[0] == "text/css"
    assert got["page"].headers["content-type"].split(";")[0] == "text/html"


def test_the_web_files_are_revalidated_and_the_tests_folder_is_not_served(client, settings):
    """WebFiles: every file of the site (pages, scripts, styles, JSON, and the libraries in web/vendor with their
    images, licences and checksums) is served with Cache-Control no-cache next to its ETag, on a 200 and on a 304,
    so a browser asks before each use and a returning visitor never runs a new module with an old one (StaticFiles
    alone sent no Cache-Control: browsers kept files for a time of their own guessing). web/tests is not part of
    the site: 404, however the path is spelt. The model files keep their own rule (no Cache-Control), and the photo
    route its own (assert_public_photo)."""
    files = sorted(p for p in WEB_DIR.rglob("*") if p.is_file())
    site = [p for p in files if p.relative_to(WEB_DIR).parts[0] != "tests"]
    tests = [p for p in files if p.relative_to(WEB_DIR).parts[0] == "tests"]
    assert len(site) >= 45 and len(tests) >= 15 and (WEB_DIR / "tests" / "api.test.mjs") in tests
    assert sum(p.relative_to(WEB_DIR).parts[0] == "vendor" for p in site) >= 20
    # every kind of file the site has is served as what it is (nosniff makes the type binding): a new kind needs
    # an entry here. A file without a suffix (a library's LICENSE, the vendor folder's SHA256SUMS) is a download.
    kinds = {".html": ("text/html",), ".js": ("text/javascript", "application/javascript"), ".css": ("text/css",),
             ".json": ("application/json",), ".png": ("image/png",), ".md": ("text/markdown",),
             "": ("application/octet-stream", "text/plain")}
    assert {p.suffix for p in site} <= set(kinds), {p.suffix for p in site}
    for path in site:
        url = "/" + path.relative_to(WEB_DIR).as_posix()
        r = client.get(url)
        assert r.status_code == 200 and r.content == path.read_bytes(), url
        assert r.headers.get_list("cache-control") == ["no-cache"] and r.headers["etag"], (url, dict(r.headers))
        assert r.headers["content-type"].split(";")[0] in kinds[path.suffix], (url, r.headers["content-type"])
        again = client.get(url, headers={"if-none-match": r.headers["etag"]})
        assert again.status_code == 304 and again.content == b"", url
        assert again.headers.get_list("cache-control") == ["no-cache"] and again.headers["etag"] == r.headers["etag"]
        changed = client.get(url, headers={"if-none-match": '"another-version"'})
        assert changed.status_code == 200 and changed.content == path.read_bytes(), url
    r = client.get("/")                                # the directory index is index.html, with the same headers
    assert r.content == (WEB_DIR / "index.html").read_bytes() and r.headers["cache-control"] == "no-cache"
    assert client.get("/", headers={"if-modified-since": r.headers["last-modified"]}).status_code == 304
    assert client.head("/api.js").headers["cache-control"] == "no-cache"

    for path in tests:
        url = "/" + path.relative_to(WEB_DIR).as_posix()
        r = client.get(url)
        assert r.status_code == 404 and r.json() == {"detail": "Not Found"}, url
        assert "cache-control" not in r.headers and "etag" not in r.headers
    for raw in ("/tests", "/tests/", "/tests/api.test.mjs", "//tests/api.test.mjs", "/./tests/api.test.mjs",
                "/x/../tests/api.test.mjs", "/tests//api.test.mjs", "/tests/./api.test.mjs", "/TESTS/api.test.mjs",
                "/Tests/fake-dom.mjs", "/themes/../tests/api.test.mjs", "/../web/tests/api.test.mjs",
                "/a/../../web/tests/api.test.mjs", "/../web/tests", "/../web/tests/"):
        status, headers, body = asgi_response(run_asgi(client.app, "GET", raw, [], []))
        assert status == 404 and json.loads(body) == {"detail": "Not Found"}, raw
        assert headers["x-content-type-options"] == ["nosniff"]
    for raw in ("/themes/../api.js", "/../web/api.js"):          # the same walk reaches the site's own files
        status, headers, _ = asgi_response(run_asgi(client.app, "GET", raw, [], []))
        assert status == 200 and headers["cache-control"] == ["no-cache"], raw
    assert asgi_response(run_asgi(client.app, "GET", "/../README.md", [], []))[0] == 404          # and nothing outside

    # the model files: served as before, no Cache-Control of the site's
    folder = settings.models_dir / LM
    folder.mkdir(parents=True)
    (folder / "m.ply").write_bytes(b"ply\n")
    r = client.get(f"/files/models/{LM}/m.ply")
    assert r.status_code == 200 and r.content == b"ply\n" and "cache-control" not in r.headers and r.headers["etag"]


# Chinese characters and CJK / full-width punctuation (the class of web/tests/i18n.test.mjs; re has no Script=Han)
_CJK = re.compile("[\u3000-\u303f\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff00-\uffef\U00020000-\U0002fa1f]")
# Where the API's JSON may hold Chinese (D32): what people typed, as key paths ("*" a list index; everything below
# one of them). Everything else is a code or technical English, which the client words in the page language.
_USER_TEXT_PATHS = (
    ("name",),                          # a landmark's name: its summary (the create response) and its detail
    ("landmarks", "*", "name"),         # the landmark list
    ("landmark", "name"),               # the public poses
    ("contributors", "*", "user_id"),   # the nicknames
    ("nickname",),                      # a registration's nickname (D33: POST /api/contributors and .../accept)
    ("results", "*", "filename"),       # an upload's file names
    ("detail", "*", "input"),           # a 422 echoes the value that was sent
)


def _cjk_in_json(doc, path=()):
    """[(key path, text)] of every key and string in a JSON document that holds Chinese, split by whether the path is
    one of _USER_TEXT_PATHS: (user text, everything else)."""
    user, other = [], []
    if isinstance(doc, dict):
        for k, v in doc.items():
            if _CJK.search(k):
                other.append((path, k))
            u, o = _cjk_in_json(v, path + (k,))
            user += u
            other += o
    elif isinstance(doc, list):
        for v in doc:
            u, o = _cjk_in_json(v, path + ("*",))
            user += u
            other += o
    elif isinstance(doc, str) and _CJK.search(doc):
        (user if any(path[:len(p)] == p for p in _USER_TEXT_PATHS) else other).append((path, doc))
    return user, other


def test_the_api_sends_no_chinese_only_codes_and_technical_english(make_client, synthetic_dataset, settings,
                                                                    monkeypatch):
    """D32: the client words everything in the page language; the API sends codes and technical English. In one
    session, the JSON of the landmark list and details, the public poses, each job, upload results (accepted and
    rejected, server_error among them), registrations and accepting a new agreement (D33), and the 403, 404, 409, 413
    and 422 errors (terms_required, terms_outdated and register_required too) holds no Chinese outside what people
    typed: landmark names, nicknames, file names and a 422's echo of the value sent. The landmark names, the nicknames
    and two file names here are Chinese, so each exemption is used."""
    ds = synthetic_dataset
    client = make_client(MissingPhotoFirstReconstructor(settings), max_upload_bytes=200_000, **TWO_SECTOR_FULL)
    salt = deployment_salt(client.app.state.settings.data_dir)
    got = []                                            # (what, response)

    def call(what, method, url, **kw):
        r = client.request(method, url, **kw)
        got.append((what, r))
        return r

    tokens = {}

    def token_of(nickname):                             # D33: each uploader registers once
        if nickname not in tokens:
            r = call("register", "POST", "/api/contributors", json={"nickname": nickname, "terms": TERMS_CURRENT})
            assert r.status_code == 201, r.text
            tokens[nickname] = r.json()["contributor_token"]
        return tokens[nickname]

    def post_photos(files, user_id="小明", **data):
        return call("upload", "POST", "/api/photos",
                    data={"contributor_token": token_of(user_id), "landmark_id": LM, **data}, files=files)

    assert call("create", "POST", "/api/landmarks",
                json={"id": LM, "name": "测试塔", "lat": ds.lat, "lon": ds.lon}).status_code == 201
    other = call("create (slug)", "POST", "/api/landmarks", json={"name": "外滩钟楼", "lat": ds.lat, "lon": ds.lon})
    assert other.status_code == 201
    assert call("duplicate id", "POST", "/api/landmarks",
                json={"id": LM, "name": "测试塔", "lat": 0, "lon": 0}).status_code == 409
    for what, body in [("lat", {"name": "塔", "lat": 91, "lon": 0}), ("name", {"name": "塔" * 101, "lat": 0, "lon": 0}),
                       ("missing", {"lat": 0, "lon": 0}), ("radius", {"name": "塔", "lat": 0, "lon": 0, "radius_m": 0})]:
        assert call(f"422 {what}", "POST", "/api/landmarks", json=body).status_code == 422

    # uploads: accepted, rejected (a duplicate and a file too large, both named in Chinese), one the server fails
    real_ingest = api_module.ingest_upload

    def ingest(store, settings, *, filename, **kwargs):
        if filename == "boom.jpg":
            raise OSError(28, "No space left on device")
        return real_ingest(store, settings, filename=filename, **kwargs)

    monkeypatch.setattr(api_module, "ingest_upload", ingest)
    good0 = ds.entries("good", sector=0)
    files = [("files", (e["file"], ds.read(e), "image/jpeg")) for e in [*good0, ds.entry("blurry"), ds.entry("far")]]
    files += [("files", ("塔楼.jpg", ds.read(good0[0]), "image/jpeg")), ("files", ("大图.jpg", b"\xff" * 200_001, "image/jpeg")),
              ("files", ("boom.jpg", ds.read(good0[0]), "image/jpeg"))]
    r = post_photos(files)
    assert r.status_code == 200, r.text
    results = r.json()["results"]
    assert {c for res in results for c in res["reasons"]} >= {"duplicate", "too_large", "server_error", "too_blurry"}
    assert sum(res["accepted"] for res in results) == 4
    r = post_photos([("files", (e["file"], ds.read(e), "image/jpeg")) for e in ds.entries("good", sector=1)],
                    user_id="小红")
    assert r.status_code == 200 and all(res["accepted"] for res in r.json()["results"])
    assert call("rebuild", "POST", f"/api/landmarks/{LM}/rebuild", json={"kind": "face", "sector": 0}).status_code == 202
    # the registration's 422s (D33: the agreement first, then the nickname, whose echo is user text)
    for body, code in [({"nickname": "小明"}, "terms_required"),
                       ({"nickname": "小明", "terms": "terms-v0"}, "terms_outdated"),
                       ({"nickname": "小明\x07", "terms": TERMS_CURRENT}, "value_error")]:
        r = call("422 register", "POST", "/api/contributors", json=body)
        assert r.status_code == 422 and r.json()["detail"][0]["type"] == code
    # uploads without a registration, and by a contributor whose accepted version is outdated, who accepts it once
    one = [("files", (good0[0]["file"], ds.read(good0[0]), "image/jpeg"))]
    for data in ({"landmark_id": LM}, {"landmark_id": LM, "contributor_token": "0" * 64}):
        r = call("upload", "POST", "/api/photos", data=data, files=one)
        assert r.status_code == 422 and r.json()["detail"][0]["type"] == "register_required"
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0' WHERE nickname = '小红'")
    r = post_photos(one, user_id="小红")
    assert r.status_code == 422 and r.json()["detail"][0]["type"] == "terms_outdated"
    for body, code in [({"contributor_token": "0" * 64, "terms": TERMS_CURRENT}, "register_required"),
                       ({"contributor_token": tokens["小红"], "terms": "terms-v0"}, "terms_outdated")]:
        r = call("422 accept", "POST", "/api/contributors/accept", json=body)
        assert r.status_code == 422 and r.json()["detail"][0]["type"] == code
    r = call("accept", "POST", "/api/contributors/accept", json={"contributor_token": tokens["小红"],
                                                                  "terms": TERMS_CURRENT})
    assert r.status_code == 200 and r.json() == {"nickname": "小红", "terms": TERMS_CURRENT}
    # the other upload 422s
    assert post_photos(one * 51).status_code == 422
    assert post_photos(one, landmark_id="nope").status_code == 404

    # a stylized version: ready, its registration refused
    store = client.app.state.store
    model = next(m for m in detail(client)["models"] if m["kind"] == "face")
    glb = (settings.data_dir / store.get_model(model["id"]).ply_path).with_suffix(".glb")
    store.set_model_stylized(model["id"], {"glb_path": glb.relative_to(settings.data_dir).as_posix(),
                                           "registration": {"accepted": False, "reason": "ambiguous"}})

    # withdrawal: a wrong token (403), an unknown photo (404), a withdrawal
    accepted = [res for res in results if res["accepted"]]
    assert call("403", "POST", f"/api/photos/{accepted[0]['photo_id']}/share_location",
                json={"share": False, "token": "0" * 64}).status_code == 403
    assert call("404 photo", "POST", "/api/photos/nope/share_location",
                json={"share": False, "token": manage_token("nope", salt)}).status_code == 404
    assert call("withdraw", "POST", f"/api/photos/{accepted[0]['photo_id']}/share_location",
                json={"share": False, "token": accepted[0]["manage_token"]}).status_code == 200
    assert call("422 withdraw", "POST", f"/api/photos/{accepted[1]['photo_id']}/share_location",
                json={"share": True, "token": "x"}).status_code == 422
    # D34: a registration's own photos, and withdrawing by the registration (refused: another's; then its own)
    own = call("own photos", "POST", "/api/contributors/photos",
               json={"contributor_token": tokens["小明"], "landmark_id": LM})
    assert own.status_code == 200 and len(own.json()["photos"]) == 3
    assert call("422 own photos", "POST", "/api/contributors/photos",
                json={"contributor_token": "0" * 64}).status_code == 422
    assert call("422 own photos landmark", "POST", "/api/contributors/photos",
                json={"contributor_token": tokens["小明"], "landmark_id": ["测试塔"]}).status_code == 422
    assert call("403 by registration", "POST", f"/api/photos/{accepted[2]['photo_id']}/share_location",
                json={"share": False, "contributor_token": tokens["小红"]}).status_code == 403
    assert call("withdraw by registration", "POST", f"/api/photos/{accepted[2]['photo_id']}/share_location",
                json={"share": False, "contributor_token": tokens["小明"]}).status_code == 200
    assert call("422 both tokens", "POST", f"/api/photos/{accepted[3]['photo_id']}/share_location",
                json={"share": False, "contributor_token": tokens["小明"], "token": "0" * 64}).status_code == 422

    # rebuilds the API refuses
    other_id = other.json()["id"]
    assert call("409 rebuild", "POST", f"/api/landmarks/{other_id}/rebuild",
                json={"kind": "face", "sector": 0}).status_code == 409
    assert call("422 rebuild", "POST", f"/api/landmarks/{LM}/rebuild", json={"kind": "face"}).status_code == 422
    assert call("422 rebuild kind", "POST", f"/api/landmarks/{LM}/rebuild", json={"kind": "x"}).status_code == 422
    assert call("404 rebuild", "POST", "/api/landmarks/nope/rebuild", json={"kind": "full"}).status_code == 404

    # every public GET
    for url in ["/api/health", "/api/landmarks", f"/api/landmarks/{LM}", f"/api/landmarks/{other_id}",
                f"/api/landmarks/{LM}/poses", f"/api/landmarks/{other_id}/poses"]:
        assert call("get", "GET", url).status_code == 200, url
    for j in store.list_jobs(LM):
        call("job", "GET", f"/api/jobs/{j.id}")
    for url in ["/api/landmarks/nope", "/api/landmarks/nope/poses", "/api/jobs/nope", photo_url("0" * 12),
                "/api/nothing-here"]:
        assert call("404", "GET", url).status_code == 404, url

    def chunks():                                       # over the body cap: 50 files x 200,000 bytes + 1 MiB
        yield b'--b\r\nContent-Disposition: form-data; name="files"; filename="a.jpg"\r\n\r\n'
        for _ in range(170):
            yield b"x" * 65536

    assert call("413", "POST", "/api/photos", content=chunks(),
                headers={"content-type": "multipart/form-data; boundary=b"}).status_code == 413

    statuses = {r.status_code for _, r in got}
    assert statuses >= {200, 201, 202, 403, 404, 409, 413, 422}, statuses
    d = detail(client)
    assert d["name"] == "测试塔" and {c["user_id"] for c in d["contributors"]} == {"小明", "小红"}
    assert any(j["status"] == "failed" and j["error"] for j in d["jobs"])               # a job error, in English
    assert any((m["stylized"] or {}).get("status") == "ready" for m in d["models"])
    user_text = set()
    for what, r in got:
        assert r.headers["content-type"].startswith("application/json"), (what, r.request.url)
        user, other_text = _cjk_in_json(r.json())
        assert other_text == [], (what, r.request.url, other_text)
        user_text |= {path for path, _ in user}
    # each exemption was needed: the Chinese name, nicknames, file names and a 422's echo really went through them
    assert user_text >= {("name",), ("landmarks", "*", "name"), ("landmark", "name"), ("contributors", "*", "user_id"),
                         ("nickname",), ("results", "*", "filename"), ("detail", "*", "input")}, user_text
    # the scanner itself
    assert _cjk_in_json({"a": ["好"], "name": "塔", "detail": [{"input": {"name": "塔"}}], "键": 1}) == (
        [(("name",), "塔"), (("detail", "*", "input", "name"), "塔")], [(("a", "*"), "好"), ((), "键")])
    assert _cjk_in_json({"msg": "a\u3002", "x": "\uff08", "y": "plain ascii, \u00e9"}) == ([], [(("msg",), "a\u3002"),
                                                                                         (("x",), "\uff08")])


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
