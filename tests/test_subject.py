"""D31 qualification geometry (crowd3d.subject): outline distance, frame share, focal, orientation, binding, cache."""

import hashlib
import json
import logging
import math
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from crowd3d import subject as sj
from crowd3d.config import REPO_ROOT, Settings
from crowd3d.exif import PhotoMeta
from crowd3d.geo import LocalENU
from crowd3d.records import Landmark

LAT, LON = 31.2397, 121.4998
SQUARE = np.array([[-50.0, -50.0], [50.0, -50.0], [50.0, 50.0], [-50.0, 50.0]])
COURT = np.array([[-10.0, -10.0], [-10.0, 10.0], [10.0, 10.0], [10.0, -10.0]])     # a hole: clockwise


def box_points(half: float = 50.0, height: float = 20.0) -> np.ndarray:
    xs = (-half, half)
    return np.array([[x, y, z] for x in xs for y in xs for z in (0.0, height)], np.float32)


def subject(rings=(SQUARE,), points=None) -> sj.Subject:
    return sj.Subject("block", LAT, LON, tuple(rings), box_points() if points is None else points, "0" * 64)


def meta_at(e: float, n: float, *, focal_35mm: float | None = 28.0, size=(4000, 3000), orientation=None) -> PhotoMeta:
    lat, lon, _ = LocalENU(LAT, LON).to_geodetic(e, n)
    return PhotoMeta(width=size[0], height=size[1], lat=lat, lon=lon, focal_35mm=focal_35mm, orientation=orientation,
                     has_exif=True)


def test_default_thresholds():
    s = Settings()
    assert (s.subject_max_distance_m, s.subject_min_share) == (50.0, 0.33)
    assert (s.subject_default_focal_35mm, s.subject_min_focal_35mm, s.subject_max_focal_35mm) == (28.0, 10.0, 1200.0)


def test_outline_distance_is_zero_inside_and_measures_to_the_courtyard_ring():
    sub = subject(rings=(SQUARE, COURT))
    assert sj.outline_distance(sub, 30.0, 0.0) == 0.0                       # inside the building
    assert sj.outline_distance(sub, 80.0, 0.0) == pytest.approx(30.0)
    assert sj.outline_distance(sub, 80.0, 90.0) == pytest.approx(50.0)      # to the corner (50, 50)
    assert sj.outline_distance(sub, 0.0, 2.0) == pytest.approx(8.0)         # in the courtyard: to its ring
    assert sj.assess(sub, meta_at(30.0, 0.0), Settings()).outline_distance_m == 0.0
    assert sj.assess(sub, meta_at(0.0, 2.0), Settings()).outline_distance_m == pytest.approx(8.0, abs=1e-4)


def test_angular_spans_take_the_smallest_arc_and_the_elevation_spread():
    pts = np.array([[-10.0, 100.0, 1.6], [10.0, 100.0, 1.6], [0.0, 100.0, 101.6]])
    az, el = sj.angular_spans(pts, np.array([0.0, 0.0, 1.6]))
    assert az == pytest.approx(2 * math.degrees(math.atan(0.1)))
    assert el == pytest.approx(45.0)
    # across north (azimuths 350 and 10): 20 degrees, not 340
    pts = np.array([[-math.sin(math.radians(10)), math.cos(math.radians(10)), 0.0],
                    [math.sin(math.radians(10)), math.cos(math.radians(10)), 0.0]]) * 100
    assert sj.angular_spans(pts, np.zeros(3))[0] == pytest.approx(20.0)
    assert sj.angular_spans(np.zeros((0, 3)), np.zeros(3)) == (0.0, 0.0)


def test_field_of_view_follows_the_products_focal_convention():
    hfov, vfov = sj.field_of_view(28.0, 4000, 3000)
    f_px = 28.0 / 43.27 * 5000.0
    assert hfov == pytest.approx(2 * math.degrees(math.atan(2000 / f_px)))
    assert vfov == pytest.approx(2 * math.degrees(math.atan(1500 / f_px)))


def test_share_is_the_wider_of_width_and_height_and_the_tallest_part_counts():
    flat = np.array([[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]], np.float32)        # 20 m wide, at the ground
    sub = subject(points=flat)
    m = meta_at(0.0, -200.0)
    hfov, vfov = sj.field_of_view(28.0, 4000, 3000)
    az, el = sj.angular_spans(flat, np.array([0.0, -200.0, 1.6]))
    assert sj.assess(sub, m, Settings()).share == pytest.approx(max(az / hfov, el / vfov))
    tall = np.vstack([flat, [[0.0, 0.0, 120.0]]]).astype(np.float32)          # a 120 m spire on the same footprint
    share = sj.assess(subject(points=tall), m, Settings()).share
    _, el = sj.angular_spans(tall, np.array([0.0, -200.0, 1.6]))
    assert share == pytest.approx(el / vfov) and share > 0.5
    near = sj.assess(subject(points=tall), meta_at(0.0, -1.0), Settings())
    assert near.share == 1.0                                                   # capped at the whole frame


def test_the_distance_edge_is_inclusive():
    sub = subject(points=np.array([[-1.0, 50.0, 0.0], [1.0, 50.0, 0.0]], np.float32))   # a 2 m post: tiny share
    m = meta_at(0.0, -100.0)
    q = sj.assess(sub, m, Settings())
    assert q.outline_distance_m == pytest.approx(50.0, abs=1e-4) and q.share < 0.05
    assert sj.assess(sub, m, Settings(subject_max_distance_m=q.outline_distance_m)).qualifies
    assert not sj.assess(sub, m, Settings(subject_max_distance_m=q.outline_distance_m - 1e-6)).qualifies
    assert sj.assess(sub, meta_at(0.0, -99.9), Settings()).qualifies              # 49.9 m with the defaults
    assert not sj.assess(sub, meta_at(0.0, -100.1), Settings()).qualifies         # 50.1 m


def test_the_share_edge_is_inclusive():
    sub = subject()
    m = meta_at(0.0, -400.0)                                                     # 350 m from the outline
    q = sj.assess(sub, m, Settings())
    assert q.outline_distance_m > 50.0 and 0.0 < q.share < 1.0
    assert sj.assess(sub, m, Settings(subject_min_share=q.share)).qualifies
    assert not sj.assess(sub, m, Settings(subject_min_share=q.share + 1e-9)).qualifies
    # with the defaults: the 100 m wide block (its near corners) fills 0.33 of a 28 mm frame from d metres
    hfov = sj.field_of_view(28.0, 4000, 3000)[0]
    d = 50.0 / math.tan(math.radians(0.33 * hfov / 2))
    assert sj.assess(sub, meta_at(0.0, -(50.0 + d - 3.0)), Settings()).qualifies
    assert not sj.assess(sub, meta_at(0.0, -(50.0 + d + 3.0)), Settings()).qualifies


@pytest.mark.parametrize("f35, used, default", [(None, 28.0, True), (4.0, 28.0, True), (9.9, 28.0, True),
                                                (10.0, 10.0, False), (50.0, 50.0, False), (1200.0, 1200.0, False),
                                                (1201.0, 28.0, True)])
def test_focal_default_and_sanity_range(f35, used, default):
    q = sj.assess(subject(), meta_at(0.0, -300.0, focal_35mm=f35), Settings())
    assert (q.focal_35mm, q.default_focal) == (used, default)
    ref = sj.assess(subject(), meta_at(0.0, -300.0, focal_35mm=used), Settings())
    assert q.share == pytest.approx(ref.share)


def test_the_upright_size_swaps_width_and_height_for_orientations_5_to_8():
    for o in (None, 1, 2, 3, 4):
        assert sj.upright_size(meta_at(0, 0, orientation=o)) == (4000, 3000)
    for o in (5, 6, 7, 8):
        assert sj.upright_size(meta_at(0, 0, orientation=o)) == (3000, 4000)
    wide = np.array([[-40.0, 0.0, 0.0], [40.0, 0.0, 0.0]], np.float32)       # only width counts here
    land = sj.assess(subject(points=wide), meta_at(0.0, -300.0), Settings())
    port = sj.assess(subject(points=wide), meta_at(0.0, -300.0, orientation=6), Settings())
    hl, hp = sj.field_of_view(28.0, 4000, 3000)[0], sj.field_of_view(28.0, 3000, 4000)[0]
    assert port.share == pytest.approx(land.share * hl / hp) and port.share > land.share


def test_the_compass_plays_no_part():
    sub = subject()
    m = meta_at(0.0, -300.0)
    assert sj.assess(sub, replace(m, heading_deg=180.0), Settings()) == sj.assess(sub, m, Settings())


def test_sample_points_cover_the_edges_on_a_half_metre_grid():
    tri = np.array([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 5.0]]])
    p = sj.sample_points(tri)
    assert np.allclose(p / 0.5, np.round(p / 0.5))
    assert len(np.unique(p, axis=0)) == len(p)
    assert {(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (0.0, 0.0, 5.0), (5.0, 0.0, 0.0)} <= {tuple(x) for x in p}


# --- binding, load and cache (the catalogue's India Gate: the smallest snapshot) -------------------------------------


@pytest.fixture
def catalog(tmp_path):
    """A copy of the repo catalogue's india-gate entry and its OSM files."""
    entry = next(e for e in json.loads((REPO_ROOT / "eval" / "landmarks.json").read_text()) if e["slug"] == "india-gate")
    (tmp_path / "eval" / "osm").mkdir(parents=True)
    for f in (entry["osm"]["file"], entry["osm_parts"]["file"]):
        shutil.copy(REPO_ROOT / f, tmp_path / f)
    path = tmp_path / "eval" / "landmarks.json"
    path.write_text(json.dumps([entry]))
    return path


@pytest.fixture(autouse=True)
def fresh_memory():
    sj._memory.clear()
    yield
    sj._memory.clear()


def india_gate(catalog: Path, **changes) -> Landmark:
    entry = json.loads(catalog.read_text())[0]
    return replace(Landmark("india-gate", "India Gate", entry["lat"], entry["lon"], 300.0, 0.0), **changes)


def test_a_bound_landmark_gets_its_building(tmp_path, catalog):
    s = Settings(data_dir=tmp_path / "data", structure_catalog=catalog)
    sub = sj.load_subject(s, india_gate(catalog))
    entry = json.loads(catalog.read_text())[0]
    assert (sub.slug, sub.lat, sub.lon, sub.parts_sha256) == ("india-gate", entry["lat"], entry["lon"],
                                                              entry["osm_parts"]["sha256"])
    assert sub.points.dtype == np.float32 and len(sub.points) > 1000 and len(sub.rings) == 1
    assert sub.points[:, 2].max() > 30.0                       # the arch is over 40 m tall in OSM
    # the same binding rule as the stylized models: an explicit binding, or nothing for a landmark elsewhere
    other = india_gate(catalog, id="gate")
    assert sj.load_subject(s, other) is None
    assert sj.load_subject(replace(s, structure_bindings={"gate": "india-gate"}), other) is not None
    assert sj.load_subject(s, india_gate(catalog, lat=entry["lat"] + 0.01)) is None      # 1.1 km off, radius 300 m
    assert sj.load_subject(replace(s, structure_catalog=None), india_gate(catalog)) is None


def test_the_subject_is_cached_in_memory_and_on_disk_by_the_parts_sha(tmp_path, catalog, monkeypatch):
    s = Settings(data_dir=tmp_path / "data", structure_catalog=catalog)
    lm = india_gate(catalog)
    first = sj.load_subject(s, lm)
    sha = json.loads(catalog.read_text())[0]["osm_parts"]["sha256"]
    (cached,) = (s.maps_dir / lm.id).glob("subject-*.npz")
    assert cached.name == f"subject-{sha[:12]}.npz"
    assert sj.load_subject(s, lm) is first                      # memory

    def no_build(*a, **k):
        raise AssertionError("rebuilt")

    monkeypatch.setattr(sj, "build_subject", no_build)
    sj._memory.clear()
    again = sj.load_subject(s, lm)                              # the file
    assert again is not first and np.array_equal(again.points, first.points)
    assert all(np.array_equal(a, b) for a, b in zip(again.rings, first.rings))
    monkeypatch.undo()

    # a new snapshot (another sha in the catalogue) is built afresh into its own file
    parts = catalog.parent.parent / json.loads(catalog.read_text())[0]["osm_parts"]["file"]
    parts.write_bytes(parts.read_bytes() + b"\n")
    entry = json.loads(catalog.read_text())[0]
    entry["osm_parts"]["sha256"] = new_sha = hashlib.sha256(parts.read_bytes()).hexdigest()
    catalog.write_text(json.dumps([entry]))
    built = []
    real = sj.build_subject
    monkeypatch.setattr(sj, "build_subject", lambda *a, **k: built.append(a) or real(*a, **k))
    third = sj.load_subject(s, lm)
    assert built and third.parts_sha256 == new_sha
    assert {p.name for p in (s.maps_dir / lm.id).glob("subject-*.npz")} == {cached.name, f"subject-{new_sha[:12]}.npz"}


def test_a_cache_file_from_other_inputs_is_not_used(tmp_path, catalog):
    s = Settings(data_dir=tmp_path / "data", structure_catalog=catalog)
    lm = india_gate(catalog)
    first = sj.load_subject(s, lm)
    (cached,) = (s.maps_dir / lm.id).glob("subject-*.npz")
    stamp = {"version": sj.SUBJECT_VERSION, "slug": "india-gate", "parts_sha256": first.parts_sha256,
             "outline_sha256": "other", "lat": first.lat, "lon": first.lon}
    sj._write(cached, replace(first, points=first.points[:5]), stamp)
    sj._memory.clear()
    assert len(sj.load_subject(s, lm).points) == len(first.points)
    cached.write_bytes(b"not a zip")
    sj._memory.clear()
    assert len(sj.load_subject(s, lm).points) == len(first.points)


def test_a_snapshot_that_cannot_be_read_fails_open_and_is_logged(tmp_path, catalog, caplog):
    s = Settings(data_dir=tmp_path / "data", structure_catalog=catalog)
    entry = json.loads(catalog.read_text())[0]
    (catalog.parent.parent / entry["osm_parts"]["file"]).write_text("<osm>changed</osm>")     # sha mismatch
    with caplog.at_level(logging.WARNING, logger="crowd3d.subject"):
        assert sj.load_subject(s, india_gate(catalog)) is None
    assert "not checked" in caplog.text
    lm = india_gate(catalog)
    (catalog.parent.parent / entry["osm"]["file"]).unlink()
    assert sj.load_subject(s, lm) is None
    catalog.write_text("{not json")
    assert sj.load_subject(s, lm) is None


def test_jobs_structure_slug_uses_the_same_binding(tmp_path, catalog):
    from crowd3d.jobs import structure_slug

    s = Settings(data_dir=tmp_path / "data", structure_catalog=catalog)
    assert structure_slug(s, india_gate(catalog)) == "india-gate" == sj.binding(s, india_gate(catalog))[0]
    assert structure_slug(s, india_gate(catalog, id="x")) is None is sj.binding(s, india_gate(catalog, id="x"))


# --- what the server process loads on the upload path (D34 part C) ----------------------------------------------------

# The clay model's family, and OpenCV with it: a server answers uploads without them (the stylize job runs them in a
# child process). STRUCTURE is the one exception, and only once per building: reading its OSM snapshot.
CLAY_FAMILY = ("crowd3d.stylize", "crowd3d.structure_pipeline", "crowd3d.judge", "crowd3d.lines3d", "crowd3d.skyline",
               "crowd3d.relief", "crowd3d.correct", "crowd3d.correct_measure", "crowd3d.roof_fit", "crowd3d.scene_io",
               "crowd3d.structmetrics", "crowd3d.tilt")
STRUCTURE = ("crowd3d.structure", "cv2")

SERVER_CHILD = r"""
import io, json, sys
from pathlib import Path

root, data_dir, stage = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
watched = sys.argv[4].split(",")
sys.path.insert(0, root)
out = {}


def loaded():
    return sorted(m for m in watched if m in sys.modules)


import crowd3d.api as api
out["after importing crowd3d.api"] = loaded()
from fastapi.testclient import TestClient
from PIL import Image
from crowd3d import jobs, subject
from crowd3d.config import Settings
from crowd3d.exif import build_exif_bytes
from crowd3d.geo import LocalENU
from crowd3d.records import TERMS_CURRENT, Landmark

settings = Settings(data_dir=data_dir, reconstructor="fake")
assert settings.structure_catalog == Path(root) / "eval" / "landmarks.json", settings.structure_catalog
entry = next(e for e in json.loads(settings.structure_catalog.read_text()) if e["slug"] == "india-gate")


def photo(north):
    lat, lon, _ = LocalENU(entry["lat"], entry["lon"]).to_geodetic(0.0, north)
    buf = io.BytesIO()
    Image.effect_noise((640, 480), 40.0).convert("RGB").save(
        buf, "JPEG", quality=90, exif=build_exif_bytes(lat=lat, lon=lon, focal_35mm=28.0))
    return buf.getvalue()


def upload(client, token, landmark_id, north):
    r = client.post("/api/photos", data={"contributor_token": token, "landmark_id": landmark_id},
                    files=[("files", ("p.jpg", photo(north), "image/jpeg"))])
    assert r.status_code == 200, r.text
    return r.json()["results"][0]


with TestClient(api.create_app(settings)) as client:
    token = client.post("/api/contributors", json={"nickname": "ann", "terms": TERMS_CURRENT}).json()["contributor_token"]
    place = {"name": "x", "lat": entry["lat"], "lon": entry["lon"], "radius_m": 300.0}
    if stage == "first":
        # the review's call: the binding of a landmark the catalogue does not list
        elsewhere = Landmark("elsewhere", "Elsewhere", entry["lat"], entry["lon"], 300.0, 0.0)
        assert subject.binding(settings, elsewhere) is None and jobs.structure_slug(settings, elsewhere) is None
        out["after binding() of an unlisted landmark"] = loaded()
        assert client.post("/api/landmarks", json={"id": "elsewhere", **place}).status_code == 201
        res = upload(client, token, "elsewhere", -30.0)
        assert "subject_unchecked" in res["warnings"], res
        out["after an upload to an unlisted landmark"] = loaded()
        assert client.post("/api/landmarks", json={"id": "india-gate", **place}).status_code == 201
    india_gate = Landmark("india-gate", "x", entry["lat"], entry["lon"], 300.0, 0.0)
    assert jobs.structure_slug(settings, india_gate) == "india-gate"
    out["after planning for a bound landmark"] = loaded()
    res = upload(client, token, "india-gate", -30.0 if stage == "first" else -32.0)
    assert "subject_unchecked" not in res["warnings"], res           # the building was loaded and the photo checked
    out["after an upload to a bound landmark"] = loaded()
out["cache files"] = sorted(p.name for p in settings.maps_dir.rglob("subject-*.npz"))
print(json.dumps(out))
"""


def server_child(tmp_path: Path, stage: str) -> dict:
    """What a fresh interpreter has loaded of CLAY_FAMILY and STRUCTURE after each step of SERVER_CHILD: the default
    structure catalogue (the repository's), a data directory under tmp_path, uploads through the API's own route."""
    r = subprocess.run([sys.executable, "-I", "-c", SERVER_CHILD, str(REPO_ROOT), str(tmp_path / "data"), stage,
                        ",".join(CLAY_FAMILY + STRUCTURE)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_the_upload_path_does_not_load_the_clay_models_family_into_the_server(tmp_path):
    """Every upload with GPS asks the structure catalogue which building the landmark is (subject.binding), and so
    does the planning of a stylize job (jobs.structure_slug). binding() used to import crowd3d.structure_pipeline for
    the six lines that read the catalogue, and the distance to the outline came from crowd3d.stylize: the first
    upload to any landmark -- bound or not -- loaded OpenCV and the whole clay-model family into the server process
    (61 -> 81 MB), and an import error anywhere in that family would have failed the binding (D34 part C).

    Now the catalogue reader (crowd3d.catalog) and the ring helpers (crowd3d.geom2d) import none of it:
    - a landmark the catalogue does not list: nothing of the family, and no OpenCV, after binding(), after an
      upload through the API, after planning;
    - a bound landmark, the first upload ever: its OSM snapshot is parsed once (crowd3d.structure, which needs
      OpenCV: that much is the function's own need) -- and still nothing of the rest of the family;
    - the same landmark in a new process: the building comes from its cache file, and neither is loaded."""
    first = server_child(tmp_path, "first")
    for step in ("after importing crowd3d.api", "after binding() of an unlisted landmark",
                 "after an upload to an unlisted landmark", "after planning for a bound landmark"):
        assert first[step] == [], (step, first[step])
    assert set(first["after an upload to a bound landmark"]) <= set(STRUCTURE), first
    assert len(first["cache files"]) == 1
    again = server_child(tmp_path, "again")
    assert again["cache files"] == first["cache files"]
    assert again["after an upload to a bound landmark"] == [] == again["after importing crowd3d.api"], again
