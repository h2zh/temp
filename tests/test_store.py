import sqlite3
import threading

import numpy as np
import pytest

from crowd3d.config import Settings
from crowd3d.geo import LocalENU
from crowd3d.records import (
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_FULL,
    JOB_QUEUED,
    JOB_RUNNING,
    PHOTO_ACCEPTED,
    PHOTO_REJECTED,
    Job,
    Landmark,
    ModelRecord,
    PhotoRecord,
)
from crowd3d.store import AlreadyExists, NotFound, Store

LAT, LON = 31.2397, 121.4998


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def store(settings):
    s = Store(settings)
    yield s
    s.close()


def landmark(lid="tower", lat=LAT, lon=LON, radius=150.0, t=1.0):
    return Landmark(id=lid, name=f"名字 {lid}", lat=lat, lon=lon, radius_m=radius, created_at=t)


def photo(pid, landmark_id="tower", user="u1", status=PHOTO_ACCEPTED, t=1.0, phash=None, **kw):
    return PhotoRecord(
        id=pid, landmark_id=landmark_id, user_id=user, status=status, uploaded_at=t, phash=phash, **kw
    )


def job(jid, kind=JOB_FACE, sector=0, status=JOB_QUEUED, t=1.0, landmark_id="tower"):
    return Job(id=jid, landmark_id=landmark_id, kind=kind, sector=sector, status=status,
               photo_ids=["a", "b"], created_at=t)


def model(mid, kind=JOB_FACE, sector=0, t=1.0, job_id="j1"):
    return ModelRecord(id=mid, landmark_id="tower", job_id=job_id, kind=kind, sector=sector,
                       ply_path=f"models/tower/{mid}.ply", num_photos=4, meta={"ok": True}, created_at=t)


def test_creates_data_dirs_and_wal(settings, store):
    assert settings.db_path.is_file()
    for d in (settings.photos_dir, settings.models_dir, settings.work_dir):
        assert d.is_dir()
    assert store._conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert store._conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_landmarks(store):
    store.create_landmark(landmark("b", t=2.0))
    created = store.create_landmark(landmark("a", t=2.0))
    store.create_landmark(landmark("c", t=1.0))
    assert created == landmark("a", t=2.0)
    assert store.get_landmark("a").name == "名字 a"
    assert [lm.id for lm in store.list_landmarks()] == ["c", "a", "b"]
    with pytest.raises(AlreadyExists):
        store.create_landmark(landmark("a"))
    with pytest.raises(NotFound):
        store.get_landmark("nope")


def test_find_landmark_for_location(store):
    enu = LocalENU(LAT, LON)
    store.create_landmark(landmark("big", radius=500.0))
    lat2, lon2, _ = enu.to_geodetic(200.0, 0.0)
    store.create_landmark(landmark("east", lat=lat2, lon=lon2, radius=150.0))
    # 150 m east of "big": inside both radii, nearer to "east".
    assert store.find_landmark_for_location(*enu.to_geodetic(150.0, 0.0)[:2]).id == "east"
    # 100 m west: only "big" covers it.
    assert store.find_landmark_for_location(*enu.to_geodetic(-100.0, 0.0)[:2]).id == "big"
    assert store.find_landmark_for_location(*enu.to_geodetic(-600.0, 0.0)[:2]) is None
    # Radius boundary uses the ENU distance validation uses.
    assert store.find_landmark_for_location(*enu.to_geodetic(0.0, -499.0)[:2]).id == "big"
    assert store.find_landmark_for_location(*enu.to_geodetic(0.0, -501.0)[:2]) is None
    # The antipode has ~zero tangent-plane distance; it must not match.
    assert store.find_landmark_for_location(-LAT, LON - 180.0) is None


def test_photos_roundtrip_and_queries(store):
    store.create_landmark(landmark())
    full = photo(
        "p1", t=3.0, phash="00ff00ff00ff00ff", reasons=[], warnings=["no_heading"], lat=LAT, lon=LON, alt=4.5,
        heading_deg=181.5, bearing_deg=1.25, distance_m=30.0, sector=0,
        width=640, height=480, focal_35mm=30.0,
        original_path="photos/tower/p1.jpg", processed_path="photos/tower/p1.proc.jpg",
        captured_at="2026:10:01 12:00:00",
    )
    assert store.add_photo(full) == full
    store.add_photo(photo("p0", t=3.0, user="u2", phash="1111111111111111"))
    store.add_photo(photo("p2", t=1.0, user="u2"))
    store.add_photo(
        photo("r1", status=PHOTO_REJECTED, reasons=["too_blurry", "duplicate"], phash="2222222222222222")
    )
    orphan = store.add_photo(photo("r2", landmark_id=None, status=PHOTO_REJECTED, reasons=["no_landmark"]))
    assert orphan.landmark_id is None and orphan.reasons == ["no_landmark"]
    assert store.get_photo("p1") == full
    assert store.get_photo("r1").reasons == ["too_blurry", "duplicate"]
    assert [p.id for p in store.list_photos("tower")] == ["p2", "r1", "p0", "p1"]
    assert [p.id for p in store.list_photos("tower", PHOTO_ACCEPTED)] == ["p2", "p0", "p1"]
    assert [p.id for p in store.list_photos("tower", PHOTO_REJECTED)] == ["r1"]
    assert store.list_photos("elsewhere") == []
    assert sorted(store.photo_hashes("tower")) == ["00ff00ff00ff00ff", "1111111111111111"]
    store.add_photo(photo("p3", user="u0"))
    assert store.contributors("tower") == [("u2", 2), ("u0", 1), ("u1", 1)]
    with pytest.raises(AlreadyExists):
        store.add_photo(photo("p1"))
    with pytest.raises(NotFound):
        store.get_photo("missing")


def test_foreign_keys_enforced(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.add_photo(photo("p1", landmark_id="ghost"))
    with pytest.raises(sqlite3.IntegrityError):
        store.create_job(job("j1", landmark_id="ghost"))


def test_jobs(store):
    store.create_landmark(landmark())
    store.create_landmark(landmark("other"))
    created = store.create_job(job("j2", t=2.0))
    assert created == job("j2", t=2.0)
    store.create_job(job("j1", kind=JOB_FULL, sector=None, t=2.0))
    store.create_job(job("j0", status=JOB_RUNNING, t=3.0, landmark_id="other"))
    assert [j.id for j in store.list_jobs()] == ["j1", "j2", "j0"]
    assert [j.id for j in store.list_jobs("tower")] == ["j1", "j2"]
    assert [j.id for j in store.list_jobs(statuses=(JOB_RUNNING,))] == ["j0"]
    assert [j.id for j in store.list_jobs("tower", (JOB_QUEUED, JOB_RUNNING))] == ["j1", "j2"]
    assert store.list_jobs(statuses=()) == []

    updated = store.update_job("j2", status=JOB_FAILED, started_at=5.0, finished_at=6.0, error="boom")
    got = (updated.status, updated.started_at, updated.finished_at, updated.error)
    assert got == (JOB_FAILED, 5.0, 6.0, "boom")
    assert updated.photo_ids == ["a", "b"] and updated.sector == 0
    assert store.update_job("j2", started_at=None).started_at is None
    assert store.get_job("j1").sector is None
    with pytest.raises(ValueError):
        store.update_job("j2", photo_ids=[])
    with pytest.raises(NotFound):
        store.update_job("missing", status=JOB_DONE)
    with pytest.raises(NotFound):
        store.get_job("missing")


def test_models_and_latest(store):
    store.create_landmark(landmark())
    store.create_job(job("j1"))
    assert store.list_models("tower") == []
    store.add_model(model("f0-old", sector=0, t=1.0))
    store.add_model(model("full-old", kind=JOB_FULL, sector=None, t=2.0))
    store.add_model(model("f3", sector=3, t=3.0))
    store.add_model(model("f0-new", sector=0, t=4.0))
    store.add_model(model("full-new", kind=JOB_FULL, sector=None, t=5.0))
    assert [m.id for m in store.list_models("tower")] == ["f0-old", "full-old", "f3", "f0-new", "full-new"]
    assert [m.id for m in store.latest_models("tower")] == ["full-new", "f0-new", "f3"]
    with pytest.raises(AlreadyExists):
        store.add_model(model("f3"))


def test_model_meta_json_roundtrip(store):
    store.create_landmark(landmark())
    store.create_job(job("j1"))
    meta = {
        "num_points": np.int64(1234),
        "alignment_rmse_m": np.float32(0.5),
        "cameras": {"p1": {"enu": np.array([1.0, 2.0, 1.6]), "heading_deg": None}},
        "message": "完成",
    }
    m = model("m1")
    m.meta = meta
    stored = store.add_model(m)
    assert stored.meta == {
        "num_points": 1234,
        "alignment_rmse_m": 0.5,
        "cameras": {"p1": {"enu": [1.0, 2.0, 1.6], "heading_deg": None}},
        "message": "完成",
    }


def test_persists_across_reopen(settings):
    s = Store(settings)
    s.create_landmark(landmark())
    s.add_photo(photo("p1", reasons=["x"]))
    s.close()
    s2 = Store(settings)
    try:
        assert s2.get_landmark("tower").radius_m == 150.0
        assert s2.get_photo("p1").reasons == ["x"]
    finally:
        s2.close()


def test_concurrent_writes(store):
    store.create_landmark(landmark())
    errors = []

    def worker(n):
        try:
            for i in range(25):
                store.add_photo(photo(f"t{n}-{i}", user=f"u{n}", t=float(i)))
                store.list_photos("tower")
        except Exception as e:  # pragma: no cover - surfaced by the assert below
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(store.list_photos("tower")) == 200
    assert store.contributors("tower")[0][1] == 25
