import io
import json
import math
import time

import cv2
import numpy as np
import pytest
from PIL import Image

from crowd3d import synthetic as syn
from crowd3d.config import Settings
from crowd3d.coverage import sector_for_bearing
from crowd3d.exif import read_photo_meta
from crowd3d.geo import angle_diff_deg, heading_from_en, local_bearing_distance, normalize_deg
from crowd3d.records import Landmark

SIZE = (640, 480)
BAD_KINDS = {
    "blurry": "too_blurry",
    "far": "too_far",
    "facing_away": "facing_away",
    "duplicate": "duplicate",
    "no_gps": "no_gps",
    "too_small": "too_small",
}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("synthetic")
    manifest = syn.generate_dataset(out, sectors=[0, 1], photos_per_sector=2, num_users=3, size=SIZE, seed=5)
    return out, manifest


def _gray(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("L"), dtype=np.float64)


def _lap_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F, ksize=1).var())


def _dhash(data: bytes) -> np.ndarray:
    px = np.asarray(Image.open(io.BytesIO(data)).convert("L").resize((9, 8), Image.Resampling.LANCZOS), np.int16)
    return (px[:, 1:] > px[:, :-1]).ravel()


def _project(cam, heading, pitch, size, point):
    fwd, right, up = syn.camera_axes(heading, pitch)
    rel = np.asarray(point, float) - np.asarray(cam, float)
    f = syn.focal_px_for(size)
    return size[0] / 2 + f * (rel @ right) / (rel @ fwd), size[1] / 2 - f * (rel @ up) / (rel @ fwd)


def _patch_std(img: np.ndarray, col: float, row: float, r: int = 4) -> float:
    c, rr = int(col), int(row)
    return float(img[rr - r:rr + r, c - r:c + r].astype(float).std(axis=(0, 1)).mean())


def test_camera_axes_follow_compass_convention():
    for heading in (0.0, 37.0, 90.0, 200.0, 315.0):
        fwd, right, up = syn.camera_axes(heading, 0.0)
        assert heading_from_en(fwd[0], fwd[1]) == pytest.approx(heading, abs=1e-9)
        # Right is 90 degrees clockwise of forward, up is +z.
        assert heading_from_en(right[0], right[1]) == pytest.approx(normalize_deg(heading + 90), abs=1e-9)
        assert up == pytest.approx([0, 0, 1])
    fwd, _, up = syn.camera_axes(0.0, 10.0)
    assert math.degrees(math.asin(fwd[2])) == pytest.approx(10.0)
    assert up[2] > 0 and up[1] < 0


def test_render_geometry_matches_pinhole_projection():
    b = syn.BuildingSpec()
    size = (320, 240)
    # Camera south-east of the building looking due north: the building sits left of centre.
    cam, heading, pitch = (4.0, -55.0, 1.6), 0.0, 8.0
    img = syn.render_view(b, cam, heading, pitch, size, syn.focal_px_for(size))
    assert img.shape == (240, 320, 3) and img.dtype == np.uint8

    left, row = _project(cam, heading, pitch, size, (-b.width_e_m / 2, -b.depth_n_m / 2, 6.0))
    right, _ = _project(cam, heading, pitch, size, (b.width_e_m / 2, -b.depth_n_m / 2, 6.0))
    assert 20 < left < size[0] / 2 < right < size[0] - 20
    assert _patch_std(img, right + 10, row) < 4          # sky east of the building, on the right
    assert _patch_std(img, right - 10, row) > 10         # textured facade
    assert _patch_std(img, left + 10, row) > 10
    assert _patch_std(img, left - 10, row) < 4

    mid, roof_row = _project(cam, heading, pitch, size, (0.0, -b.depth_n_m / 2, b.height_m))
    assert 20 < roof_row < size[1] - 20
    assert _patch_std(img, mid, roof_row - 8) < 4        # sky above the roof line
    assert _patch_std(img, mid, roof_row + 8) > 10
    sky = img[int(roof_row) - 8, int(mid)].astype(int)
    assert sky[2] > sky[0]                               # bluish sky
    assert _patch_std(img, mid, size[1] - 10) > 6        # textured ground


def test_render_is_fast_and_rich_in_sift_features():
    b = syn.BuildingSpec()
    syn.render_view(b, (0.0, 40.0, 1.6), 180.0, 8.0, (64, 48), 76.8)  # build and cache the scene
    sift = cv2.SIFT_create()
    for bearing in (0, 45, 90, 135, 180, 225, 270, 315):
        r = syn.CAMERA_RADIUS_M[1]  # farthest allowed distance: the building is smallest in frame
        cam = (r * math.sin(math.radians(bearing)), r * math.cos(math.radians(bearing)), syn.CAMERA_HEIGHT_M)
        t0 = time.perf_counter()
        img = syn.render_view(b, cam, normalize_deg(bearing + 180), 8.0, SIZE, syn.focal_px_for(SIZE))
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.5, f"render took {elapsed:.2f}s"
        keypoints = sift.detect(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), None)
        assert len(keypoints) >= 1500, f"bearing {bearing}: only {len(keypoints)} SIFT keypoints"


def test_facades_are_distinct():
    b = syn.BuildingSpec()
    walls = syn._scene(b.width_e_m, b.depth_n_m, b.height_m, b.seed).walls
    means = [tex.levels[0].reshape(-1, 3).mean(axis=0) for tex, _, _ in walls]
    for i in range(4):
        for j in range(i + 1, 4):
            assert np.abs(means[i] - means[j]).max() > 8, (i, j)


def test_manifest_matches_files_and_exif(dataset):
    out, manifest = dataset
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["landmark"] == {"lat": syn.DEMO_LAT, "lon": syn.DEMO_LON, "name": syn.DEMO_NAME}
    assert manifest["building"] == {"width_e_m": 20.0, "depth_n_m": 14.0, "height_m": 12.0, "seed": 7}
    photos = manifest["photos"]
    assert [p["kind"] for p in photos] == ["good"] * 4 + list(BAD_KINDS)
    assert [p["user_id"] for p in photos] == [f"user{i % 3 + 1}" for i in range(len(photos))]
    for p in photos:
        meta = read_photo_meta((out / p["file"]).read_bytes())
        expected_size = syn.BAD_SIZE if p["kind"] == "too_small" else SIZE
        assert (meta.width, meta.height) == expected_size
        assert meta.focal_35mm == syn.focal_35mm_for(expected_size) == 28
        assert meta.focal_mm == pytest.approx(28 / syn.PHONE_CROP_FACTOR, abs=0.01)
        assert meta.make and meta.model and meta.captured_at
        if p["exif_lat"] is None:
            assert not meta.has_gps and p["kind"] == "no_gps"
        else:
            assert meta.lat == pytest.approx(p["exif_lat"], abs=1e-6)
            assert meta.lon == pytest.approx(p["exif_lon"], abs=1e-6)
        assert meta.heading_deg == pytest.approx(p["exif_heading_deg"], abs=0.01)
        assert meta.heading_ref == "T"


def test_good_photo_poses(dataset):
    _, manifest = dataset
    lat0, lon0 = manifest["landmark"]["lat"], manifest["landmark"]["lon"]
    good = [p for p in manifest["photos"] if p["expected_valid"]]
    assert [p["sector"] for p in good] == [0, 0, 1, 1]
    for p in good:
        x, y, z = p["true_enu"]
        bearing = heading_from_en(x, y)
        assert z == syn.CAMERA_HEIGHT_M
        assert syn.CAMERA_RADIUS_M[0] <= math.hypot(x, y) <= syn.CAMERA_RADIUS_M[1]
        # Inside its sector, at least 4 degrees from either boundary.
        offset = angle_diff_deg(bearing, p["sector"] * 45.0)
        assert abs(offset) <= 22.5 - syn.SECTOR_EDGE_MARGIN_DEG + 1e-9
        # Looks at the building centre within the jitter, pitched up within range.
        assert abs(angle_diff_deg(p["true_heading_deg"], heading_from_en(-x, -y))) <= syn.HEADING_JITTER_DEG + 1e-6
        assert syn.PITCH_RANGE_DEG[0] <= p["true_pitch_deg"] <= syn.PITCH_RANGE_DEG[1]
        # EXIF is noisy, but it agrees on the sector and still faces the landmark.
        exif_bearing, dist = local_bearing_distance(lat0, lon0, p["exif_lat"], p["exif_lon"])
        assert sector_for_bearing(exif_bearing, 8) == p["sector"]
        assert abs(dist - math.hypot(x, y)) <= 6.0 + 1e-6
        assert abs(angle_diff_deg(p["exif_heading_deg"], normalize_deg(exif_bearing + 180))) < 45


def test_bad_photos_are_bad_in_exactly_one_way(dataset):
    out, manifest = dataset
    lat0, lon0 = manifest["landmark"]["lat"], manifest["landmark"]["lon"]
    by_kind = {p["kind"]: p for p in manifest["photos"]}
    data = {p["file"]: (out / p["file"]).read_bytes() for p in manifest["photos"]}
    for kind, reason in BAD_KINDS.items():
        p = by_kind[kind]
        assert not p["expected_valid"] and p["expected_reason"] == reason
        meta = read_photo_meta(data[p["file"]])
        blur = _lap_var(_gray(data[p["file"]]))
        assert (blur < 10) == (kind == "blurry"), (kind, blur)
        assert (min(meta.width, meta.height) < 480) == (kind == "too_small")
        if meta.has_gps:
            bearing, dist = local_bearing_distance(lat0, lon0, meta.lat, meta.lon)
            assert (dist > 150) == (kind == "far")
            facing_off = abs(angle_diff_deg(meta.heading_deg, normalize_deg(bearing + 180)))
            assert (facing_off > 90) == (kind == "facing_away"), (kind, facing_off)
    for p in manifest["photos"]:
        if p["expected_valid"]:
            assert _lap_var(_gray(data[p["file"]])) > 90

    dup = by_kind["duplicate"]
    original = dup["duplicate_of"]
    assert data[dup["file"]] != data[original]
    assert int((_dhash(data[dup["file"]]) != _dhash(data[original])).sum()) <= 2
    # Nothing else looks like a near-duplicate of anything.
    others = [f for f in data if f != dup["file"]]
    hashes = {f: _dhash(data[f]) for f in others}
    for i, a in enumerate(others):
        for b in others[i + 1:]:
            assert int((hashes[a] != hashes[b]).sum()) >= 8, (a, b)


def test_validation_rejects_each_bad_photo_for_its_reason(dataset):
    from crowd3d import validation

    out, manifest = dataset
    settings = Settings()
    lm = manifest["landmark"]
    landmark = Landmark(id="demo-building", name=lm["name"], lat=lm["lat"], lon=lm["lon"], radius_m=150.0, created_at=0)
    hashes: list[str] = []
    for p in manifest["photos"]:
        data = (out / p["file"]).read_bytes()
        try:
            result = validation.validate_photo(data, read_photo_meta(data), landmark, list(hashes), settings)
        except NotImplementedError:
            pytest.skip("validation not implemented yet")
        assert result.accepted == p["expected_valid"], (p["file"], result.reasons)
        assert result.reasons == ([] if p["expected_valid"] else [p["expected_reason"]]), p["file"]
        if p["expected_valid"]:
            assert result.warnings == []
            assert result.sector == p["sector"]
            hashes.append(result.phash)


def test_generation_is_deterministic(tmp_path):
    kwargs = dict(sectors=[3], photos_per_sector=1, size=(320, 240), include_bad=False)
    a = syn.generate_dataset(tmp_path / "a", seed=11, **kwargs)
    b = syn.generate_dataset(tmp_path / "b", seed=11, **kwargs)
    c = syn.generate_dataset(tmp_path / "c", seed=12, **kwargs)
    assert a == b
    assert (tmp_path / "a" / "s3_00.jpg").read_bytes() == (tmp_path / "b" / "s3_00.jpg").read_bytes()
    assert a["photos"][0]["true_enu"] != c["photos"][0]["true_enu"]


def test_cli_writes_dataset(tmp_path, capsys):
    out = tmp_path / "cli"
    rc = syn.main(["--out", str(out), "--sectors", "2,6", "--per-sector", "1", "--size", "320x240",
                   "--no-bad", "--seed", "3", "--users", "2"])
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert [p["sector"] for p in manifest["photos"]] == [2, 6]
    assert [p["user_id"] for p in manifest["photos"]] == ["user1", "user2"]
    assert all(p["expected_valid"] for p in manifest["photos"])
    assert Image.open(out / "s2_00.jpg").size == (320, 240)
    assert "2 photos" in capsys.readouterr().out


@pytest.mark.slow
def test_two_adjacent_sectors_reconstruct_with_colmap(tmp_path):
    pycolmap = pytest.importorskip("pycolmap")
    images = tmp_path / "images"
    manifest = syn.generate_dataset(images, sectors=[0, 1], photos_per_sector=4, size=SIZE, include_bad=False, seed=1)
    names = [p["file"] for p in manifest["photos"]]
    db = tmp_path / "db.db"
    pycolmap.extract_features(db, images, image_names=names, camera_mode=pycolmap.CameraMode.AUTO,
                              device=pycolmap.Device.cpu)
    pycolmap.match_exhaustive(db, device=pycolmap.Device.cpu)
    (tmp_path / "sparse").mkdir()
    recs = pycolmap.incremental_mapping(db, images, tmp_path / "sparse")
    assert recs
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    assert rec.num_reg_images() >= 6

    # Camera centres match the true poses up to a similarity transform (Umeyama, no reflection).
    truth = {p["file"]: p for p in manifest["photos"]}
    regs = list(rec.images.values())
    est = np.array([im.projection_center() for im in regs])
    true = np.array([truth[im.name]["true_enu"] for im in regs])
    e, t = est - est.mean(0), true - true.mean(0)
    u, s, vt = np.linalg.svd(t.T @ e)
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ d @ vt
    scale = np.trace(np.diag(s) @ d) / (e ** 2).sum()
    fitted = scale * e @ rot.T + true.mean(0)
    rmse = float(np.sqrt(((fitted - true) ** 2).sum(axis=1).mean()))
    assert rmse < 0.5
    for im in regs:
        x, y, _ = rot @ im.viewing_direction()
        assert abs(angle_diff_deg(heading_from_en(x, y), truth[im.name]["true_heading_deg"])) < 2.0
