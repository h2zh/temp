"""scripts/materialize_testset.py offline: selection rule, freeze from a collector run, materialise + integrity."""

from __future__ import annotations

import importlib.util
import io
import json
import struct
import sys
import tarfile
from pathlib import Path

import pytest
from PIL import Image

import test_fetch_commons as tfc   # fake Commons helpers (tests/ is on sys.path under pytest)
from test_fetch_commons import commons_assets, fc  # noqa: F401  (fixtures the fake Commons is built with)

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "materialize_testset.py"
LAT0, LON0 = tfc.LAT0, tfc.LON0


@pytest.fixture(scope="module")
def mt():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("materialize_testset", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- selection rule


def _row(title, sector, dist, author):
    return {"title": title, "sector": sector, "distance_m": dist, "author_key": author}


def test_diversity_order_round_robins_sectors_then_authors_closest_first(mt):
    rows = [_row("a1", 0, 10, "A"), _row("a2", 0, 5, "A"), _row("a3", 0, 1, "A"), _row("b1", 0, 50, "B"),
            _row("e1", 2, 80, "C"), _row("e2", 2, 90, "C"), _row("w1", 6, 30, None), _row("nogps", None, None, "D")]
    order = mt.diversity_order(rows)
    # round 0: closest of each sector; within sector 0, A's closest (a3) before B's only photo (b1)
    assert order[:3] == ["a3", "e1", "w1"]
    assert order[3:5] == ["b1", "e2"]            # round 1: next author in sector 0, next photo in sector 2
    assert order[5:7] == ["a2", "a1"] and order[-1] == "nogps"
    assert mt.diversity_order(list(reversed(rows))) == order          # deterministic, input order irrelevant


def test_diversity_order_caps_a_dominant_author(mt):
    rows = [_row(f"chabe{i:02d}", 2, 30 + i, "Chabe01") for i in range(20)] + [
        _row(f"other{i}", 2, 100 + i, f"user{i}") for i in range(5)]
    order = mt.diversity_order(rows)
    assert sum(t.startswith("chabe") for t in order[:6]) == 1        # one per author before anyone's second
    assert order[6:] == [f"chabe{i:02d}" for i in range(1, 20)]      # then the rest, closest first


def test_screening_order_samples_every_source(mt):
    fc = mt.fc
    cands = ([fc.Candidate(f"File:T{i}.jpg", sources=["titles:mdx.txt"]) for i in range(3)]
             + [fc.Candidate(f"File:S{i}.jpg", sources=["search:s.html"]) for i in range(2)]
             + [fc.Candidate(f"File:C{i}.jpg", sources=["category:X"]) for i in range(4)]
             + [fc.Candidate("File:Both.jpg", sources=["category:X", "search:s.html"])])
    order = [c.title[5:-4] for c in mt.screening_order(cands)]
    assert order == ["Both", "T0", "S0", "C0", "T1", "S1", "C1", "T2", "C2", "C3"]


@pytest.mark.parametrize("title,cats,off", [
    ("File:Battistero.jpg", ["Exterior of the Baptistry (Florence)"], False),
    ("File:Intérieur Baptistère.jpg", ["Baptistry (Florence)"], True),
    ("File:Dome.jpg", ["Interior of Aachen Cathedral"], True),
    ("File:Choir.jpg", ["Choir of Aachen Cathedral (exterior)", "Stained glass"], False),
    ("File:City.jpg", ["Views from the Galata Tower"], True),
    ("File:Deck.jpg", ["CN Tower observation deck"], True),
])
def test_is_off_target(mt, title, cats, off):
    assert mt.is_off_target(title, cats) is off


def test_views_from_the_landmark_are_off_target(mt):
    vf = mt.views_from_re(["Lincoln Memorial", "Frauenkirche (Dresden)"])
    assert mt.is_off_target("File:Mall.jpg", ["Washington Monument from the Lincoln Memorial"], vf)
    assert mt.is_off_target("File:View_from_Frauenkirche_Dresden.jpg", [], vf)
    assert not mt.is_off_target("File:Dresden.jpg", ["Frauenkirche Dresden from Neumarkt"], vf)
    assert not mt.is_off_target("File:Steps.jpg", ["Lincoln Memorial at night"], vf)
    assert mt.views_from_re([]) is None


def test_client_fails_fast_on_a_host_blocked_longer_than_the_caller_waits(mt, tmp_path):
    """Regression (fetch_commons_landmark.PoliteClient): after upload.wikimedia.org answered 429 with
    Retry-After 600 to a full-original request, the next original Range read (auto mode accepts 60 s) used
    to sleep the full 600 s before being refused again; it now fails at once without a request."""
    wm = tfc.FakeWikimedia(mt.fc)
    a, b = mt.fc.upload_url("File:A.jpg"), mt.fc.upload_url("File:B.jpg")
    wm.scripted[a] = [(429, {"retry-after": "600"})]
    wm.files[b] = b"x" * 10
    client = wm.client(tmp_path / "cache")
    with pytest.raises(mt.fc.RateLimited):
        client.get(a, max_wait_s=60)
    t0, n = wm.now, len(wm.calls)
    with pytest.raises(mt.fc.RateLimited, match="pause"):
        client.get(b, byte_range=(0, 9), max_wait_s=60)
    assert wm.now - t0 < 5 and len(wm.calls) == n
    assert client.get(b, byte_range=(0, 9)).status == 206 and wm.now - t0 >= 600   # willing callers still wait


# ---------------------------------------------------------------- MegaDepth-X sparse helpers


def _images_bin(names: list[str], npts: int = 2) -> bytes:
    out = struct.pack("<Q", len(names))
    for i, name in enumerate(names, 1):
        out += struct.pack("<I", i) + struct.pack("<7d", 1, 0, 0, 0, 0, 0, 0) + struct.pack("<I", 1)
        out += name.encode() + b"\0" + struct.pack("<Q", npts) + struct.pack("<ddq", 1.0, 2.0, -1) * npts
    return out


def _sparse(dirpath: Path, names: list[str]) -> Path:
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "images.bin").write_bytes(_images_bin(names))
    (dirpath / "cameras.bin").write_bytes(b"cams")
    (dirpath / "points3D.bin").write_bytes(b"points" * 10)
    return dirpath


def test_colmap_image_names_skips_2d_points(mt, tmp_path):
    p = tmp_path / "images.bin"
    p.write_bytes(_images_bin(["000002.jpg", "000017.jpg", "abc d.jpg"], npts=5))
    assert mt.colmap_image_names(p) == ["000002.jpg", "000017.jpg", "abc d.jpg"]


def test_verify_sparse_reports_ok_changed_missing(mt, tmp_path):
    d = _sparse(tmp_path / "0" / "sparse", ["a.jpg"])
    expected = mt.sparse_manifest(d)
    assert set(expected) == {"cameras.bin", "images.bin", "points3D.bin"}
    assert mt.verify_sparse(d, expected)["status"] == "ok"
    (d / "cameras.bin").write_bytes(b"CAMS")
    assert mt.verify_sparse(d, expected)["files"]["cameras.bin"] == "changed"
    (d / "points3D.bin").unlink()
    assert mt.verify_sparse(d, expected)["status"] == "missing"


def test_stream_extract_sparse_keeps_only_the_wanted_component(mt, tmp_path):
    src = _sparse(tmp_path / "src" / "Scene_X" / "0" / "sparse", ["a.jpg"])
    _sparse(tmp_path / "src" / "Scene_X" / "1" / "sparse", ["b.jpg"])
    (src.parent / "images").mkdir()
    (src.parent / "images" / "a.jpg").write_bytes(b"pixels")
    archive = tmp_path / "Scene_X.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(tmp_path / "src" / "Scene_X", arcname=".")             # members "./0/sparse/..." as on HF
    dest = tmp_path / "mdx" / "scene-x"
    written = mt.stream_extract_sparse(archive.as_uri(), "Scene_X", ["0"], dest)
    assert sorted(p.name for p in written) == ["cameras.bin", "images.bin", "points3D.bin"]
    assert not (dest / "1").exists() and not list(dest.rglob("a.jpg"))
    assert mt.verify_sparse(dest / "0" / "sparse", mt.sparse_manifest(src))["status"] == "ok"


# ---------------------------------------------------------------- collect -> freeze -> materialise


@pytest.fixture
def wm(mt, commons_assets):
    fake = tfc.FakeWikimedia(mt.fc)
    fake.pages, fake.files = dict(commons_assets[0]), dict(commons_assets[1])
    # Originals refused, as from the cloud IP the v1 set was collected on: page-metadata EXIF.
    for url in list(fake.files):
        if url.startswith(mt.fc.UPLOAD_BASE):
            fake.scripted[url] = [(429, {"retry-after": "600"})]
    return fake


@pytest.fixture
def frozen(mt, wm, tmp_path):
    """A collector run against the fake Commons, frozen into a one-building test set."""
    fetch_root = tmp_path / "ten" / "commons"
    cache = fetch_root / "http-cache"
    argv = ["--name", "Test Landmark", "--lat", str(LAT0), "--lon", str(LON0), "--radius", "150",
            "--category", "Test Landmark", "--exclude-category", "interior", "--out", str(fetch_root / "test-lm"),
            "--cache-dir", str(cache), "--max-photos", "5"]
    assert mt.collect(argv, pool=10, max_screen=50, client=wm.client(cache)) == 0
    stats = json.loads((fetch_root / "test-lm" / "screening_stats.json").read_text())
    assert stats["listing"]["selection"]["page_passed"] == 3 and stats["funnel"]["kept"] == 2
    outline = tmp_path / "outline.geojson"
    outline.write_text(json.dumps({"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                       "geometry": {"type": "Polygon", "coordinates": [[[LON0, LAT0], [LON0 + 1e-4, LAT0],
                                                                        [LON0, LAT0 + 1e-4], [LON0, LAT0]]]}}]}))
    landmarks = tmp_path / "landmarks.json"
    landmarks.write_text(json.dumps([{
        "slug": "test-lm", "name": "Test Landmark", "lat": LAT0, "lon": LON0, "radius_m": 150,
        "centre": {"source": "test"}, "commons_categories": ["Test Landmark"],
        "osm": {"type": "way", "id": 1, "url": "https://www.openstreetmap.org/way/1", "file": str(outline),
                "license": "ODbL-1.0", "attribution": "© OpenStreetMap contributors"},
        "megadepth_x": {"scene": "Test_Scene", "component": "0", "archive_mb": 1}}]))
    # GT: "Good small" is registered in component 0 under a nested name (the "commons/..." naming some
    # MegaDepth-X scenes use) and stored rotated (Orientation 6); "Good thumb" is in component 1 only.
    mdx_root = tmp_path / "mdx"
    nested = "commons/Test_Landmark/0/pictures/Good_small.jpg"
    _sparse(mdx_root / "test-lm" / "0" / "sparse", [nested, "000002.jpg"])
    _sparse(mdx_root / "test-lm" / "1" / "sparse", ["000009.jpg"])
    (mdx_root / "test-lm" / "orientation_index.json").write_text(json.dumps(
        {"components": {"0": {nested: 6, "000002.jpg": 1}, "1": {"000009.jpg": 0}}}))
    import pyarrow as pa
    import pyarrow.parquet as pq
    parquet = tmp_path / "license.parquet"
    pq.write_table(pa.table({
        "name_in_megadepth-x": [f"Test_Scene/0/images/{nested}", "Test_Scene/1/images/000009.jpg",
                                "Other_Scene/0/images/000001.jpg"],
        "cat": ["Test_Scene", "Test_Scene", "Other_Scene"],
        "image_name": ["Good small.jpg", "Good thumb.jpg", "Good thumb.jpg"],
        "license_short_name": ["CC BY 4.0", "CC BY 4.0", "CC BY 4.0"]}), parquet)
    path = mt.freeze(testset_id="test-v1", slugs=["test-lm"], landmarks=landmarks, fetch_root=fetch_root,
                     http_cache=cache, mdx_root=mdx_root, licence_parquet=parquet, description="unit test",
                     extra={"licence": "photos keep their Commons licences"}, out_dir=tmp_path / "testsets" / "test-v1")
    return path, cache, fetch_root


def test_freeze_writes_the_required_fields(mt, frozen):
    path, _, fetch_root = frozen
    ts = json.loads(path.read_text())
    assert ts["schema_version"] == 1 and ts["testset_id"] == "test-v1" and ts["created"] and ts["schema_notes"]
    (b,) = ts["buildings"]
    assert b["slug"] == "test-lm" and b["radius_m"] == 150 and b["centre"]["lat"] == LAT0
    outline = path.parent / b["osm_outline"]["file"]
    assert outline.exists() and mt.sha256_file(outline) == b["osm_outline"]["sha256"]
    assert b["osm_outline"]["licence"] == "ODbL-1.0" and "OpenStreetMap" in b["osm_outline"]["attribution"]
    manifest = {p["title"]: p for p in json.loads((fetch_root / "test-lm" / "manifest.json").read_text())["photos"]}
    assert {p["title"] for p in b["photos"]} == set(manifest)
    for p in b["photos"]:
        assert p["sha256"] == manifest[p["title"]]["pixels"]["sha256"]
        assert p["exif_source"] == "commons_page_metadata" and set(p["exif_page_table"]) >= {
            "exif-gpslatitude", "exif-gpslongitude"}
        assert p["thumbnail_width"] in (1280, 1920) and p["original_url"].startswith("https://upload.wikimedia.org/")
        assert set(p["exif"]) >= {"lat", "lon", "gps_accuracy_m", "heading_deg", "focal_mm", "captured_at", "make",
                                  "model"} and p["author"] and p["licence"]
        assert len(p["exif_sha256"]) == 64 and len(p["pixels_sha256"]) == 64 and len(p["dhash"]) == 16
    gt = b["gt_reference"]
    assert gt["scene"] == "Test_Scene" and gt["component"] == "0" and set(gt["sparse_files"]) >= {"images.bin"}
    assert gt["images"] == {"Good small.jpg": {"name": "commons/Test_Landmark/0/pictures/Good_small.jpg",
                                               "name_in_megadepth_x": "Test_Scene/0/images/commons/Test_Landmark/"
                                                                      "0/pictures/Good_small.jpg",
                                               "orientation": 6}}
    (extra,) = gt["additional_components"]
    assert extra["component"] == "1" and extra["images"]["Good thumb.jpg"]["orientation"] is None   # 0 is invalid
    photos = {p["key"]: p for p in b["photos"]}
    assert photos["Good small.jpg"]["gt_image"] == "commons/Test_Landmark/0/pictures/Good_small.jpg"
    assert photos["Good small.jpg"]["gt_orientation"] == 6
    assert photos["Good thumb.jpg"]["gt_image"] is None and photos["Good thumb.jpg"]["gt_additional"]["name"] == \
        "000009.jpg"
    assert b["stats"]["with_gt_pose"] == 1 and b["stats"]["with_gt_pose_additional_component"] == 1
    assert b["stats"]["kept"] == 2
    with pytest.raises(SystemExit, match="immutable"):
        mt.freeze(testset_id="test-v1", slugs=["test-lm"], landmarks=Path("x"), fetch_root=fetch_root,
                  http_cache=fetch_root, mdx_root=None, licence_parquet=None, description="", extra={},
                  out_dir=path.parent)


def test_materialise_from_the_collector_cache_reproduces_every_file(mt, wm, frozen, tmp_path):
    path, cache, fetch_root = frozen
    n = len(wm.calls)
    rep = mt.materialize(path, tmp_path / "mat", client=wm.client(cache, offline=True))
    assert len(wm.calls) == n and rep["complete"] and rep["totals"] == {"photos": 2, "ok": 2, "changed": 0,
                                                                         "missing": 0}
    for row in rep["buildings"]["test-lm"]["rows"]:
        assert mt.sha256_file(Path(row["file"])) == row["sha256_expected"]
    assert Path(rep["report_path"]) == tmp_path / "mat" / "test-v1" / "integrity_report.json"
    assert json.loads(Path(rep["report_path"]).read_text())["testset_sha256"] == mt.sha256_file(path)


def test_materialise_into_a_fresh_cache_downloads_only_the_listed_files(mt, wm, frozen, tmp_path):
    path, _, _ = frozen
    ts = json.loads(path.read_text())
    n = len(wm.calls)
    rep = mt.materialize(path, tmp_path / "fresh", client=wm.client(tmp_path / "fresh" / "http-cache"))
    assert rep["complete"]
    urls = [u for _, u, _ in wm.calls[n:]]
    assert sorted(urls) == sorted(p["pixels_url"] for p in ts["buildings"][0]["photos"])   # no pages, no originals
    gaps = [b[0] - a[0] for a, b in zip(wm.calls[n:], wm.calls[n + 1:])]
    assert all(g >= 1.0 for g in gaps)                                                      # polite spacing
    # A second run finds the materialised files and makes no request at all.
    m = len(wm.calls)
    rep2 = mt.materialize(path, tmp_path / "fresh", client=wm.client(tmp_path / "fresh" / "http-cache"))
    assert rep2["complete"] and len(wm.calls) == m
    assert {r["from"] for r in rep2["buildings"]["test-lm"]["rows"]} == {"materialised_before"}


def _reencode(data: bytes, quality: int) -> bytes:
    buf = io.BytesIO()
    with Image.open(io.BytesIO(data)) as im:
        im.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def test_integrity_report_flags_changed_and_missing_photos(mt, wm, frozen, tmp_path):
    path, _, _ = frozen
    photos = {p["title"]: p for p in json.loads(path.read_text())["buildings"][0]["photos"]}
    thumb, small = photos["File:Good thumb.jpg"], photos["File:Good small.jpg"]
    wm.files[thumb["pixels_url"]] = _reencode(wm.files[thumb["pixels_url"]], 85)   # Wikimedia re-rendered it
    del wm.files[small["pixels_url"]]                                              # file deleted on Commons
    rep = mt.materialize(path, tmp_path / "fresh", client=wm.client(tmp_path / "fresh" / "http-cache"))
    rows = {r["title"]: r for r in rep["buildings"]["test-lm"]["rows"]}
    t = rows["File:Good thumb.jpg"]
    assert t["status"] == "changed" and t["thumbnail_match"] is False and t["exif_match"] is True
    assert t["pixels_equivalent"] is True and Path(t["file"]).exists()        # written anyway, still usable
    assert rows["File:Good small.jpg"]["status"] == "missing" and "404" in rows["File:Good small.jpg"]["detail"]
    assert not rep["complete"] and rep["totals"]["changed"] == 1 and rep["totals"]["missing"] == 1
    assert mt.usable_photos(rep, "test-lm") == [Path(t["file"])]
    assert mt.usable_photos(rep, "test-lm", include_changed=False) == []


def test_changed_exif_table_is_reported_as_exif_change(mt, wm, frozen, tmp_path):
    path, cache, _ = frozen
    ts = json.loads(path.read_text())
    p = ts["buildings"][0]["photos"][0]
    p["exif_page_table"]["exif-gpsimgdirection"] = "123"                         # someone edited the record
    edited = tmp_path / "edited.json"
    edited.write_text(json.dumps(ts))
    rep = mt.materialize(edited, tmp_path / "mat", client=wm.client(cache, offline=True))
    row = next(r for r in rep["buildings"]["test-lm"]["rows"] if r["title"] == p["title"])
    assert row["status"] == "changed" and row["exif_match"] is False and row["thumbnail_match"] is True


def test_rate_limit_stops_and_marks_the_rest_not_attempted(mt, wm, frozen, tmp_path):
    path, _, _ = frozen
    photos = json.loads(path.read_text())["buildings"][0]["photos"]
    wm.scripted[photos[0]["pixels_url"]] = [(429, {"retry-after": "3600"})]
    n = len(wm.calls)
    rep = mt.materialize(path, tmp_path / "fresh", client=wm.client(tmp_path / "fresh" / "http-cache", max_wait_s=300))
    rows = rep["buildings"]["test-lm"]["rows"]
    assert [r["status"] for r in rows] == ["missing", "missing"]
    assert rows[0]["detail"].startswith("rate_limited") and rows[1]["detail"].startswith("not_attempted")
    assert len(wm.calls) == n + 1 and rep["stopped"]                              # no hammering


def test_gt_is_verified_in_place_and_changes_are_reported(mt, frozen, tmp_path):
    path, cache, _ = frozen
    rep = mt.materialize(path, tmp_path / "mat", offline=True, gt=True, gt_dir=tmp_path / "mdx",
                         client=mt.fc.PoliteClient(cache, offline=True))
    assert rep["buildings"]["test-lm"]["gt"]["status"] == "ok"
    (tmp_path / "mdx" / "test-lm" / "0" / "sparse" / "points3D.bin").write_bytes(b"other")
    rep = mt.materialize(path, tmp_path / "mat", offline=True, gt=True, gt_dir=tmp_path / "mdx",
                         client=mt.fc.PoliteClient(cache, offline=True))
    assert rep["buildings"]["test-lm"]["gt"]["status"] == "changed"


def test_cli_refuses_a_cache_inside_the_repo_and_unknown_buildings(mt, frozen, tmp_path):
    path, _, _ = frozen
    with pytest.raises(SystemExit, match="inside the repository"):
        mt.main([str(path), "--cache-dir", str(ROOT / "scripts" / "cache")])
    with pytest.raises(SystemExit, match="no building"):
        mt.materialize(path, tmp_path / "c", buildings=["nope"], offline=True)


# ---------------------------------------------------------------- the committed test sets


def committed_commons_testsets() -> list[Path]:
    """Test sets this script materialises: every photo has a Commons pixels_url."""
    out = []
    for p in sorted((ROOT / "eval" / "testsets").glob("*/testset.json")):
        photos = [ph for b in json.loads(p.read_text(encoding="utf-8")).get("buildings", []) for ph in b["photos"]]
        if photos and all("pixels_url" in ph for ph in photos):
            out.append(p)
    return out


@pytest.mark.parametrize("path", committed_commons_testsets(), ids=lambda p: p.parent.name)
def test_committed_photo_testsets_are_consistent(mt, path):
    ts, _, _ = mt.load_testset(path)
    assert ts["testset_id"] == path.parent.name
    titles = set()
    for b in ts["buildings"]:
        outline = mt.outline_path(path, b)
        assert outline.exists() and mt.sha256_file(outline) == b["osm_outline"]["sha256"]
        assert b["osm_outline"]["licence"] == "ODbL-1.0"
        files = [p["file"] for p in b["photos"]]
        assert len(files) == len(set(files))
        for p in b["photos"]:
            assert p["title"] not in titles
            titles.add(p["title"])
            assert len(p["sha256"]) == 64 and len(p["exif_sha256"]) == 64 and p["licence"] and p["exif"]["lat"]
            assert p["exif_source"] != "commons_page_metadata" or p["exif_page_table"]
            assert p["landmark_to_camera"]["distance_m"] <= b["radius_m"] + 1e-6
            if b["gt_reference"]:
                assert (p["gt_image"] is not None) == (p["key"] in b["gt_reference"]["images"])
    # no pixels in the repo
    assert not [q for q in path.parent.rglob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png", ".ply")]
