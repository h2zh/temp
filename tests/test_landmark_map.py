"""crowd3d.landmark_map: the feature and match cache (exact row copies, keys, pairing order) and its use by jobs."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crowd3d import landmark_map as lmap
from crowd3d import params
from crowd3d import synthetic as syn
from crowd3d.config import Settings

pycolmap = pytest.importorskip("pycolmap")


@contextmanager
def _sql(db: Path) -> Iterator[sqlite3.Connection]:
    """A committed and closed connection: an open one would race pycolmap's own SQLite on the same WAL file."""
    conn = sqlite3.connect(db)
    try:
        with conn:
            yield conn
    finally:
        conn.close()


def _rows(db: Path, table: str, key: str) -> dict:
    with _sql(db) as c:
        return {r[0]: r[1:] for r in c.execute(f"SELECT * FROM {table} ORDER BY {key}")}


def _images(root: Path, n: int = 4, size=(64, 48)) -> tuple[Path, list[str]]:
    images = root / "images"
    images.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for k in range(n):
        Image.fromarray((rng.random((size[1], size[0], 3)) * 255).astype(np.uint8)).save(images / f"{k:05d}_p{k}.jpg")
    return images, sorted(p.name for p in images.iterdir())


def _database(path: Path, images: Path, names: list[str]) -> dict[str, int]:
    """A COLMAP database of these images (in this id order) with made-up features, matches and geometries for every
    pair, as COLMAP would have written them; returns name -> image id."""
    path.unlink(missing_ok=True)
    pycolmap.Database.open(path).close()
    pycolmap.import_images(path, images, camera_mode=pycolmap.CameraMode.AUTO, image_names=names)
    db = pycolmap.Database.open(path)
    try:
        ids = {im.name: im.image_id for im in db.read_all_images()}
        for name, i in ids.items():
            rng = np.random.default_rng(int(name[:5]))      # a function of the image, not of its id
            db.write_keypoints(i, rng.random((20, 6)).astype(np.float32))
            db.write_descriptors(i, pycolmap.FeatureDescriptors(
                pycolmap.FeatureExtractorType.SIFT, (rng.random((20, 128)) * 255).astype(np.uint8)))
        for a in names:
            for b in names:
                if ids[a] < ids[b]:
                    rng = np.random.default_rng(1000 * int(min(a, b)[:5]) + int(max(a, b)[:5]))
                    m = rng.integers(0, 20, (8, 2)).astype(np.uint32)
                    db.write_matches(ids[a], ids[b], m)
                    g = pycolmap.TwoViewGeometry()
                    g.config = 3
                    g.F = rng.random((3, 3))
                    g.inlier_matches = m[:5]
                    db.write_two_view_geometry(ids[a], ids[b], g)
        return ids
    finally:
        db.close()


def _ids(db: Path) -> dict[str, int]:
    with _sql(db) as c:
        return dict(c.execute("SELECT name, image_id FROM images"))


def _empty(path: Path, images: Path, names: list[str]) -> dict[str, int]:
    path.unlink(missing_ok=True)
    pycolmap.Database.open(path).close()
    pycolmap.import_images(path, images, camera_mode=pycolmap.CameraMode.AUTO, image_names=names)
    return _ids(path)


def _harvested(tmp_path: Path):
    images, names = _images(tmp_path)
    src = tmp_path / "a.db"
    ids = _database(src, images, names)
    cache = lmap.FeatureCache(tmp_path / "maps" / "lm" / lmap.CACHE_FILE)
    keys = {ids[n]: lmap.file_sha256(images / n) for n in names}
    pairs = lmap.job_pairs(pycolmap, src)
    sigs = lmap.camera_signatures(src)
    assert cache.harvest(src, keys, "f1", pairs, sigs, "m1") == (4, 6)
    return images, names, src, cache, keys, pairs, sigs


def test_versions_change_with_the_options_that_change_results_only():
    ex = pycolmap.FeatureExtractionOptions()
    ex.max_image_size = 1600
    v = pycolmap.TwoViewGeometryOptions()
    v.ransac.random_seed = 0
    f0, m0 = lmap.feature_version(pycolmap, ex), lmap.match_version(pycolmap, v)
    assert (f0, m0) == (lmap.feature_version(pycolmap, ex), lmap.match_version(pycolmap, v))
    ex.num_threads = 3                                   # threads do not change SIFT's output
    assert lmap.feature_version(pycolmap, ex) == f0
    ex.max_image_size = 1024
    assert lmap.feature_version(pycolmap, ex) != f0
    v.ransac.random_seed = 1
    assert lmap.match_version(pycolmap, v) != m0
    v.ransac.random_seed = 0
    matching = pycolmap.FeatureMatchingOptions()
    assert lmap.match_version(pycolmap, v, matching) == m0      # COLMAP's defaults are what reconstruct uses
    matching.sift.max_ratio = 0.7
    assert lmap.match_version(pycolmap, v, matching) != m0


def test_harvest_then_seed_gives_the_same_rows_and_harvesting_again_adds_nothing(tmp_path):
    images, names, src, cache, keys, pairs, sigs = _harvested(tmp_path)
    assert cache.harvest(src, keys, "f1", pairs, sigs, "m1") == (0, 0)
    dst = tmp_path / "b.db"
    ids = _empty(dst, images, names)
    keys_b = {ids[n]: lmap.file_sha256(images / n) for n in names}
    assert cache.seed_features(dst, keys_b, "f1") == 4
    assert cache.seed_pairs(dst, lmap.job_pairs(pycolmap, dst), keys_b, lmap.camera_signatures(dst), "m1") == 6
    for table, key in (("keypoints", "image_id"), ("descriptors", "image_id"), ("matches", "pair_id"),
                       ("two_view_geometries", "pair_id")):
        assert _rows(dst, table, key) == _rows(src, table, key), table
    # and pycolmap reads them back as the same features, matches and geometries
    a, b = pycolmap.Database.open(src), pycolmap.Database.open(dst)
    try:
        assert np.array_equal(a.read_descriptors(2).data, b.read_descriptors(2).data)
        assert np.array_equal(a.read_two_view_geometry(1, 3).inlier_matches,
                              b.read_two_view_geometry(1, 3).inlier_matches)
    finally:
        a.close()
        b.close()


def test_a_subset_job_hits_only_its_own_pairs_and_other_versions_miss(tmp_path):
    images, names, src, cache, *_ = _harvested(tmp_path)
    dst = tmp_path / "c.db"
    sub = names[1:]                                      # a later job without the first photo: other image ids
    ids = _empty(dst, images, sub)
    keys_c = {ids[n]: lmap.file_sha256(images / n) for n in sub}
    sigs_c = lmap.camera_signatures(dst)
    assert cache.seed_features(dst, keys_c, "f2") == 0                      # another feature version
    assert cache.seed_features(dst, keys_c, "f1") == 3
    assert cache.seed_pairs(dst, lmap.job_pairs(pycolmap, dst), keys_c, sigs_c, "m2") == 0
    assert cache.seed_pairs(dst, lmap.job_pairs(pycolmap, dst), keys_c, sigs_c, "m1") == 3
    src_ids, name_of = _ids(src), {i: n for n, i in ids.items()}
    got, want = _rows(dst, "matches", "pair_id"), _rows(src, "matches", "pair_id")
    for pid, row in got.items():
        n1, n2 = (name_of[i] for i in pycolmap.pair_id_to_image_pair(pid))
        assert row == want[lmap.pair_id(src_ids[n1], src_ids[n2])]


def test_pairs_miss_when_a_camera_or_the_matching_order_differs(tmp_path):
    images, names, _, cache, *_ = _harvested(tmp_path)
    dst = tmp_path / "d.db"
    ids = _empty(dst, images, names)
    keys_d = {ids[n]: lmap.file_sha256(images / n) for n in names}
    db = pycolmap.Database.open(dst)
    try:                                               # the first photo now has a focal prior (another camera)
        cam = db.read_camera(db.read_image(ids[names[0]]).camera_id)
        cam.has_prior_focal_length = True
        db.update_camera(cam)
    finally:
        db.close()
    assert cache.seed_pairs(dst, lmap.job_pairs(pycolmap, dst), keys_d, lmap.camera_signatures(dst), "m1") == 3
    # the same pairs, matched the other way round (a spatial job can): not the rows COLMAP would compute then
    dst2 = tmp_path / "e.db"
    ids2 = _empty(dst2, images, names)
    keys_e = {ids2[n]: lmap.file_sha256(images / n) for n in names}
    flipped = [(b, a) for a, b in lmap.job_pairs(pycolmap, dst2)]
    assert cache.seed_pairs(dst2, flipped, keys_e, lmap.camera_signatures(dst2), "m1") == 0


def test_job_pairs_follow_the_generators_first_occurrence(tmp_path):
    images, names = _images(tmp_path, n=5)
    db_path = tmp_path / "s.db"
    _empty(db_path, images, names)
    assert lmap.job_pairs(pycolmap, db_path) == [(a, b) for a in range(1, 6) for b in range(a + 1, 6)]
    db = pycolmap.Database.open(db_path)
    try:
        for im, e in zip(db.read_all_images(), (0.0, 50.0, 1.0, 51.0, 2.0)):
            db.write_pose_prior(pycolmap.PosePrior(position=np.array([e, 0.0, 0.0]), corr_data_id=im.data_id,
                                                   coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN))
    finally:
        db.close()
    opts = pycolmap.SpatialPairingOptions()
    opts.max_num_neighbors, opts.max_distance = 2, 10.0
    pairs = lmap.job_pairs(pycolmap, db_path, opts)
    assert len({lmap.pair_id(a, b) for a, b in pairs}) == len(pairs)
    assert {frozenset(p) for p in pairs} == {frozenset(p) for p in [(1, 3), (1, 5), (3, 5), (2, 4)]}


def test_a_geometry_with_cameras_of_its_own_is_not_cached(tmp_path):
    images, names = _images(tmp_path)
    src = tmp_path / "a.db"
    ids = _database(src, images, names)
    with _sql(src) as c:
        c.execute("UPDATE two_view_geometries SET camera1 = x'07000000' WHERE pair_id = ?", (lmap.pair_id(1, 2),))
    cache = lmap.FeatureCache(tmp_path / "cache.sqlite3")
    keys = {ids[n]: lmap.file_sha256(images / n) for n in names}
    assert cache.harvest(src, keys, "f", lmap.job_pairs(pycolmap, src), lmap.camera_signatures(src), "m") == (4, 5)


def test_a_broken_cache_never_fails_the_job(tmp_path, caplog):
    images, names = _images(tmp_path)
    db_path = tmp_path / "a.db"
    _database(db_path, images, names)
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "cache.sqlite3").mkdir()                  # a directory where the cache file should be
    job = lmap.JobCache(lmap.FeatureCache(blocked / "cache.sqlite3"), pycolmap, db_path,
                        {n: images / n for n in names}, pycolmap.FeatureExtractionOptions(),
                        pycolmap.TwoViewGeometryOptions())
    job.seed_features()
    job.seed_pairs()
    job.harvest("x")
    assert not job.ok and "goes on without it" in caplog.text
    assert lmap.FeatureCache(tmp_path / "nothing.sqlite3").uses() == []


def test_the_flag_and_where_the_cache_lives(tmp_path):
    assert params.DEFAULTS.feature_cache is False
    assert params.with_overrides({"feature_cache": True}).feature_cache is True
    with pytest.raises(ValueError, match="true or false"):
        params.with_overrides({"feature_cache": 1})
    s = Settings(data_dir=tmp_path)
    assert s.maps_dir == tmp_path / "maps" and not s.maps_dir.is_relative_to(s.models_dir)


# ---------------------------------------------------------------- slow: real COLMAP on synthetic facade photos


@pytest.fixture(scope="module")
def facade(tmp_path_factory):
    ds = tmp_path_factory.mktemp("facade")
    manifest = syn.generate_dataset(ds, sectors=[2], photos_per_sector=4, size=(640, 480), include_bad=False, seed=3)
    images = ds / "images"
    images.mkdir()
    names = []
    for k, p in enumerate(q for q in manifest["photos"] if q["kind"] == "good"):
        names.append(f"{k:05d}_{Path(p['file']).stem}.jpg")
        (images / names[-1]).symlink_to(ds / p["file"])
    return images, names


def _extract(db: Path, images: Path, names: list[str]) -> None:
    opts = pycolmap.FeatureExtractionOptions()
    opts.max_image_size = 1600
    pycolmap.extract_features(db, images, image_names=names, camera_mode=pycolmap.CameraMode.AUTO,
                              extraction_options=opts, device=pycolmap.Device.cpu)


def _verification():
    v = pycolmap.TwoViewGeometryOptions()
    v.ransac.random_seed = 0
    return v


@pytest.mark.slow
def test_colmap_skips_seeded_features_and_pairs_and_computes_the_same_rest(facade, tmp_path):
    images, names = facade
    full = tmp_path / "full.db"
    _empty(full, images, names)
    _extract(full, images, names)
    again = tmp_path / "again.db"
    _empty(again, images, names)
    _extract(again, images, names)
    assert _rows(again, "keypoints", "image_id") == _rows(full, "keypoints", "image_id")     # CPU SIFT: exact
    assert _rows(again, "descriptors", "image_id") == _rows(full, "descriptors", "image_id")
    pycolmap.match_exhaustive(full, verification_options=_verification(), device=pycolmap.Device.cpu)

    seeded = tmp_path / "seeded.db"
    _empty(seeded, images, names)
    sentinel = np.zeros((3, 6), np.float32)
    with _sql(full) as src, _sql(seeded) as dst:
        kp = src.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = 1").fetchone()
        dst.execute("INSERT INTO keypoints VALUES (1, 3, 6, ?)", (sentinel.tobytes(),))
        dst.execute("INSERT INTO descriptors SELECT * FROM (SELECT 1, 0, 3, 128, ?)", (bytes(3 * 128),))
        assert kp[0] > 3
    _extract(seeded, images, names)
    assert _rows(seeded, "keypoints", "image_id")[1] == (3, 6, sentinel.tobytes())       # left as seeded
    assert {k: v for k, v in _rows(seeded, "keypoints", "image_id").items() if k != 1} == {
        k: v for k, v in _rows(full, "keypoints", "image_id").items() if k != 1}
    with _sql(full) as src, _sql(seeded) as dst:            # the real features, one pair seeded
        dst.execute("DELETE FROM keypoints WHERE image_id = 1")
        dst.execute("DELETE FROM descriptors WHERE image_id = 1")
        for table in ("keypoints", "descriptors"):
            dst.executemany(f"INSERT INTO {table} VALUES ({', '.join('?' * (5 if table == 'descriptors' else 4))})",
                            src.execute(f"SELECT * FROM {table} WHERE image_id = 1").fetchall())
        pid = lmap.pair_id(1, 2)
        dst.execute("INSERT INTO matches SELECT * FROM (SELECT ?, 0, 2, NULL)", (pid,))
        dst.execute("INSERT INTO two_view_geometries (pair_id, rows, cols, data, config) VALUES (?, 0, 2, NULL, 99)",
                    (pid,))
    pycolmap.match_exhaustive(seeded, verification_options=_verification(), device=pycolmap.Device.cpu)
    got, want = _rows(seeded, "two_view_geometries", "pair_id"), _rows(full, "two_view_geometries", "pair_id")
    assert got[pid][3] == 99                                                       # the seeded pair was skipped
    assert {k: v for k, v in got.items() if k != pid} == {k: v for k, v in want.items() if k != pid}


@pytest.mark.slow
def test_a_job_seeded_from_the_cache_holds_the_rows_it_would_compute(facade, tmp_path):
    images, names = facade
    cache = lmap.FeatureCache(tmp_path / lmap.CACHE_FILE)
    ex = pycolmap.FeatureExtractionOptions()
    ex.max_image_size = 1600

    def job(db: Path, subset: list[str]) -> lmap.JobCache:
        _empty(db, images, subset)
        j = lmap.JobCache(cache, pycolmap, db, {n: images / n for n in subset}, ex, _verification())
        j.seed_features()
        _extract(db, images, subset)
        j.seed_pairs()
        pycolmap.match_exhaustive(db, verification_options=_verification(), device=pycolmap.Device.cpu)
        j.harvest(db.stem)
        return j

    first = job(tmp_path / "first.db", names[:3])
    assert first.stats.to_meta() == {"features_hit": 0, "features": 3, "pairs_hit": 0, "pairs": 3} and first.ok
    second = job(tmp_path / "second.db", names)
    assert second.stats.to_meta() == {"features_hit": 3, "features": 4, "pairs_hit": 3, "pairs": 6}
    alone = tmp_path / "alone.db"                                      # the same job without a cache
    _empty(alone, images, names)
    _extract(alone, images, names)
    pycolmap.match_exhaustive(alone, verification_options=_verification(), device=pycolmap.Device.cpu)
    for table, key in (("keypoints", "image_id"), ("descriptors", "image_id"), ("matches", "pair_id"),
                       ("two_view_geometries", "pair_id")):
        assert _rows(tmp_path / "second.db", table, key) == _rows(alone, table, key), table
    uses = [(u["label"], u["features_hit"], u["pairs_hit"]) for u in cache.uses()]
    assert uses == [("first", 0, 0), ("second", 3, 3)]
