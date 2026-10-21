"""ColmapReconstructor: cheap failure paths, plus real pycolmap runs on synthetic photos (slow)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from crowd3d import synthetic as syn
from crowd3d.config import Settings
from crowd3d.exif import read_photo_meta
from crowd3d.geo import angle_diff_deg, heading_from_en
from crowd3d.ingest import make_processed_copy
from crowd3d.reconstruct import ColmapReconstructor, ReconPhoto, get_reconstructor

SIZE = (640, 480)


def recon_photos(ds: Path, manifest: dict[str, Any], keep=lambda p: True) -> list[ReconPhoto]:
    return [
        ReconPhoto(Path(p["file"]).stem, ds / p["file"], p["exif_lat"], p["exif_lon"], p["exif_heading_deg"])
        for p in manifest["photos"]
        if p["kind"] == "good" and keep(p)
    ]


def reconstruct(tmp: Path, manifest: dict[str, Any], photos: list[ReconPhoto], reconstructor=None):
    reconstructor = reconstructor or ColmapReconstructor(Settings(data_dir=tmp))
    lm = manifest["landmark"]
    return reconstructor.reconstruct(photos, lm["lat"], lm["lon"], tmp / "model.ply", tmp / "work")


def read_ply(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="ascii").splitlines()
    body = lines[lines.index("end_header") + 1:]
    return np.array([[float(v) for v in line.split()] for line in body]).reshape(-1, 6)


def truth_errors(result, manifest) -> tuple[dict[str, float], dict[str, float]]:
    """Per registered photo: (distance to the true camera centre, heading error vs the true heading)."""
    truth = {Path(p["file"]).stem: p for p in manifest["photos"]}
    pos = {k: float(np.linalg.norm(np.array(c["enu"]) - truth[k]["true_enu"])) for k, c in result.cameras.items()}
    head = {k: abs(angle_diff_deg(c["heading_deg"], truth[k]["true_heading_deg"])) for k, c in result.cameras.items()}
    return pos, head


def report(label: str, result) -> None:
    print(f"\n[{label}] {result.message}; timings {result.timings_s}; "
          f"rmse {result.alignment_rmse_m}, heading mae {result.heading_mae_deg}")


# ---------------------------------------------------------------- fast


def test_get_reconstructor_colmap():
    assert isinstance(get_reconstructor("colmap"), ColmapReconstructor)


def test_too_few_photo_files_fails_without_raising(tmp_path):
    pytest.importorskip("pycolmap")
    photos = [ReconPhoto(f"p{i}", tmp_path / f"missing{i}.jpg", 31.0, 121.0, 0.0) for i in range(5)]
    res = ColmapReconstructor(Settings(data_dir=tmp_path)).reconstruct(
        photos, 31.0, 121.0, tmp_path / "m.ply", tmp_path / "work")
    assert not res.ok
    assert "photo files" in res.message
    assert res.num_input == 5
    assert not (tmp_path / "m.ply").exists()
    json.dumps(res.to_meta())


# ---------------------------------------------------------------- slow: real pycolmap


@pytest.fixture(scope="module")
def facade_data(tmp_path_factory):
    """The 4 photos of sector 2 (east), the minimum for a facade job.

    The generator keeps photos >= 4 deg inside their sector, so widening the
    sector by a 5 deg facade margin would add no neighbours; only sector 2 is rendered.
    """
    pytest.importorskip("pycolmap")
    ds = tmp_path_factory.mktemp("facade")
    manifest = syn.generate_dataset(ds, sectors=[2], photos_per_sector=4, size=SIZE, include_bad=False, seed=3)
    return ds, manifest, recon_photos(ds, manifest)


@pytest.mark.slow
def test_single_facade_model(facade_data, tmp_path):
    _, manifest, photos = facade_data
    res = reconstruct(tmp_path, manifest, photos)
    report("facade", res)
    assert res.ok, res.message
    assert res.num_registered >= 3
    assert res.georegistered
    assert res.alignment_rmse_m < 5.0
    pos, head = truth_errors(res, manifest)
    assert max(pos.values()) < 5.0
    assert max(head.values()) < 15.0
    assert res.heading_mae_deg < 15.0
    assert all(abs(c["enu"][2] - 1.6) < 0.5 for c in res.cameras.values())

    pts = read_ply(res.ply_path)
    assert len(pts) == res.num_points >= 300
    assert ((pts[:, 3:] >= 0) & (pts[:, 3:] <= 255)).all()
    # The model sits on the east side of the landmark, upright, building-sized.
    median = np.median(pts[:, :3], axis=0)
    assert abs(angle_diff_deg(heading_from_en(median[0], median[1]), 90.0)) < 45.0
    building = manifest["building"]
    assert 0.5 * building["height_m"] < np.percentile(pts[:, 2], 98) < 1.5 * building["height_m"]


@pytest.mark.slow
def test_spatial_matching_and_rerun_in_same_work_dir(facade_data, tmp_path):
    _, manifest, photos = facade_data
    reconstructor = ColmapReconstructor(Settings(data_dir=tmp_path))
    reconstructor.exhaustive_max_images = 0     # force GPS-guided spatial matching
    first = reconstruct(tmp_path, manifest, photos, reconstructor)
    second = reconstruct(tmp_path, manifest, photos, reconstructor)
    assert first.ok and second.ok, (first.message, second.message)
    assert first.num_registered == second.num_registered >= 3
    assert abs(first.num_points - second.num_points) <= 0.05 * first.num_points


@pytest.mark.slow
def test_photos_that_do_not_overlap_fail_cleanly(tmp_path):
    pytest.importorskip("pycolmap")
    rng = np.random.default_rng(0)
    photos = []
    for i in range(3):
        path = tmp_path / f"noise{i}.jpg"
        Image.fromarray(rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)).save(path)
        photos.append(ReconPhoto(f"noise{i}", path, 31.0 + i * 1e-4, 121.0, None))
    (tmp_path / "corrupt.jpg").write_bytes(b"\xff\xd8\xff\xe0 not a jpeg")
    photos.append(ReconPhoto("corrupt", tmp_path / "corrupt.jpg", 31.0, 121.0, None))
    res = ColmapReconstructor(Settings(data_dir=tmp_path)).reconstruct(
        photos, 31.0, 121.0, tmp_path / "m.ply", tmp_path / "work")
    assert not res.ok
    assert "overlap" in res.message
    assert res.num_input == 4


@pytest.mark.slow
def test_full_ring_model(tmp_path):
    pytest.importorskip("pycolmap")
    ds = tmp_path / "ring"
    manifest = syn.generate_dataset(ds, photos_per_sector=4, size=SIZE, include_bad=False, seed=0)
    photos = recon_photos(ds, manifest)
    assert len(photos) == 32
    res = reconstruct(tmp_path, manifest, photos)
    report("full ring", res)
    assert res.ok, res.message
    assert res.num_registered >= 0.8 * len(photos)
    assert res.georegistered
    assert res.alignment_rmse_m < 5.0
    assert res.heading_mae_deg < 10.0
    pos, head = truth_errors(res, manifest)
    assert sum(e < 5.0 for e in pos.values()) >= 0.8 * res.num_registered
    assert np.median(list(head.values())) < 3.0
    assert res.num_points >= 1000
    pts = read_ply(res.ply_path)
    median = np.median(pts[:, :3], axis=0)
    assert math.hypot(median[0], median[1]) < 6.0
    # Points hug the 20 x 14 x 12 m box: most lie within a metre of its footprint.
    b = manifest["building"]
    outside = np.maximum(np.abs(pts[:, 0]) - b["width_e_m"] / 2, np.abs(pts[:, 1]) - b["depth_n_m"] / 2)
    assert np.mean(outside < 1.0) > 0.9


@pytest.mark.slow
def test_processed_copies_give_colmap_the_exif_focal_prior(facade_data, tmp_path):
    # COLMAP only trusts FocalLengthIn35mmFilm when FocalLength is present too; without
    # the prior it falls back to 1.2 * max(w, h) and never keeps small sets' focal fixed.
    pycolmap = pytest.importorskip("pycolmap")
    ds, manifest, _ = facade_data
    images = tmp_path / "images"
    images.mkdir()
    data = (ds / manifest["photos"][0]["file"]).read_bytes()
    (images / "a.jpg").write_bytes(make_processed_copy(data, read_photo_meta(data), Settings(data_dir=tmp_path)))
    pycolmap.extract_features(tmp_path / "db.db", images, camera_mode=pycolmap.CameraMode.AUTO,
                              device=pycolmap.Device.cpu)
    assert ColmapReconstructor._all_focal_priors(pycolmap, tmp_path / "db.db")
    db = pycolmap.Database.open(tmp_path / "db.db")
    try:
        (camera,) = db.read_all_cameras()
    finally:
        db.close()
    assert camera.focal_length == pytest.approx(manifest["focal_px"], rel=0.02)
