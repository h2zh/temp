import sqlite3
import threading
from dataclasses import replace

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
    LOC_ACCEPTED,
    LOC_REJECTED,
    LOC_SKIPPED,
    LOCATION_CONSENT_V1,
    LOCATION_CONSENT_V2,
    PHOTO_ACCEPTED,
    PHOTO_REJECTED,
    Job,
    Landmark,
    LocRecord,
    ModelRecord,
    PhotoRecord,
    location_shared,
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
        captured_at="2026:10:01 12:00:00", heading_ref="T", location_consent=LOCATION_CONSENT_V1,
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
    # nicknames by name, no per-person counts (D30); rejected photos make no contributor
    store.add_photo(photo("r3", user="a-rejected", status=PHOTO_REJECTED))
    assert store.contributors("tower") == ["u0", "u1", "u2"]
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
    assert store.contributors("tower") == [f"u{n}" for n in range(8)]


def test_a_database_from_before_the_new_columns_is_upgraded_on_open(settings):
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.db_path)
    conn.executescript("""
        CREATE TABLE landmarks (id TEXT PRIMARY KEY, name TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
                                radius_m REAL NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE photos (id TEXT PRIMARY KEY, landmark_id TEXT REFERENCES landmarks(id), user_id TEXT NOT NULL,
            status TEXT NOT NULL, uploaded_at REAL NOT NULL, reasons TEXT NOT NULL, warnings TEXT NOT NULL,
            lat REAL, lon REAL, alt REAL, heading_deg REAL, bearing_deg REAL, distance_m REAL, sector INTEGER,
            width INTEGER, height INTEGER, focal_35mm REAL, phash TEXT, original_path TEXT, processed_path TEXT,
            captured_at TEXT);
        CREATE TABLE jobs (id TEXT PRIMARY KEY, landmark_id TEXT NOT NULL REFERENCES landmarks(id), kind TEXT NOT NULL,
            sector INTEGER, status TEXT NOT NULL, photo_ids TEXT NOT NULL, created_at REAL NOT NULL,
            started_at REAL, finished_at REAL, error TEXT);
        CREATE TABLE models (id TEXT PRIMARY KEY, landmark_id TEXT NOT NULL REFERENCES landmarks(id),
            job_id TEXT NOT NULL REFERENCES jobs(id), kind TEXT NOT NULL, sector INTEGER, ply_path TEXT NOT NULL,
            num_photos INTEGER NOT NULL, meta TEXT NOT NULL, created_at REAL NOT NULL);
        INSERT INTO landmarks VALUES ('tower', 'Tower', 31.0, 121.0, 150.0, 1.0);
        INSERT INTO photos (id, landmark_id, user_id, status, uploaded_at, reasons, warnings)
            VALUES ('p', 'tower', 'u', 'accepted', 1.0, '[]', '[]');
        INSERT INTO jobs (id, landmark_id, kind, sector, status, photo_ids, created_at)
            VALUES ('j', 'tower', 'face', 0, 'running', '["p"]', 1.0);
        INSERT INTO models VALUES ('m', 'tower', 'j', 'face', 0, 'models/tower/m.ply', 1, '{}', 1.0);
    """)
    conn.close()
    s = Store(settings)
    try:
        assert s.get_photo("p").gps_accuracy_m is None
        assert s.get_meta("public_id_salt_check") is None and s.location_consent_count() == 0   # D30 meta table
        # D30: a photo stored before the consent existed is not shared, and has no heading tag
        old = s.get_photo("p")
        assert (old.location_consent, old.heading_ref, location_shared(old)) == (None, None, False)
        s.add_photo(photo("c", heading_ref="M", location_consent=LOCATION_CONSENT_V1))
        assert (s.get_photo("c").heading_ref, location_shared(s.get_photo("c"))) == ("M", True)
        assert s.get_photo("p").outline_distance_m is s.get_photo("p").subject_share is None     # D31 columns
        s.add_photo(photo("r", outline_distance_m=12.5, subject_share=0.4))
        assert (s.get_photo("r").outline_distance_m, s.get_photo("r").subject_share) == (12.5, 0.4)
        assert s.get_job("j").attempts == 0 and s.get_job("j").model_id is None
        assert s.update_job("j", attempts=1).attempts == 1
        s.add_photo(photo("q", gps_accuracy_m=4.6))
        assert s.get_photo("q").gps_accuracy_m == 4.6
        assert s.get_model("m").stylized is None                    # SQL NULL in the added column
        glb = {"glb_path": "models/tower/m.glb"}
        assert s.set_model_stylized("m", glb).stylized == glb and s.list_models("tower")[0].stylized == glb
        s.create_job(Job(id="s", landmark_id="tower", kind="stylize", sector=0, status=JOB_QUEUED, photo_ids=[],
                         created_at=2.0, model_id="m"))
        assert s.get_job("s").model_id == "m"
        # D30: the localizations table is created on open
        assert s.localizations("tower") == {}
        skip = LocRecord("c", "tower", LOC_SKIPPED, 2.0, reason="no_focal", maps_key="k")
        assert s.put_localization(skip) and s.localizations("tower") == {"c": skip}
        assert not s.put_localization(replace(skip, photo_id="p"))      # an old photo: no consent, no localization
    finally:
        s.close()


def test_location_shared_needs_an_accepted_photo_and_a_known_consent():
    """The one rule every reader of consent uses (D30)."""
    assert location_shared(photo("a", location_consent=LOCATION_CONSENT_V1))
    assert not location_shared(photo("a"))                                       # default: not shared
    assert not location_shared(photo("r", status=PHOTO_REJECTED, location_consent=LOCATION_CONSENT_V1))
    for other in ("", "share-location-v0", "true", "yes"):                       # only a consent text we showed
        assert not location_shared(photo("a", location_consent=other))


def test_withdrawing_location_consent_clears_it_and_only_it(store):
    store.create_landmark(landmark())
    store.add_photo(photo("p", location_consent=LOCATION_CONSENT_V1, heading_ref="T"))
    store.add_photo(photo("q", location_consent=LOCATION_CONSENT_V1))
    before = store.get_photo("p")
    after = store.withdraw_location_consent("p")
    assert after.location_consent is None and not location_shared(after)
    assert replace(after, location_consent=before.location_consent) == before
    assert store.withdraw_location_consent("p") == after                         # again: no change
    assert location_shared(store.get_photo("q"))
    with pytest.raises(NotFound):
        store.withdraw_location_consent("missing")


def test_clearing_every_location_consent_counts_them_at_every_landmark(store):
    """For crowd3d.consent.forget_lost_salt (D30): every photo with a consent, at every landmark, and nothing else."""
    store.create_landmark(landmark())
    store.create_landmark(replace(landmark(), id="other"))
    store.add_photo(photo("p", location_consent=LOCATION_CONSENT_V1, heading_ref="T"))
    store.add_photo(photo("q", landmark_id="other", location_consent=LOCATION_CONSENT_V1))
    store.add_photo(photo("n"))
    before = {pid: store.get_photo(pid) for pid in "pqn"}
    assert store.location_consent_count() == 2
    assert store.clear_location_consents() == 2
    assert store.location_consent_count() == 0 and store.clear_location_consents() == 0
    for pid, old in before.items():
        assert store.get_photo(pid) == replace(old, location_consent=None)


def test_meta_values_are_kept_and_set_only_when_asked(settings):
    s = Store(settings)
    try:
        assert s.get_meta("k") is None
        assert s.setdefault_meta("k", "a") == "a"
        assert s.setdefault_meta("k", "b") == "a"                                  # a stored value is kept
        s.set_meta("k", "c")
        assert s.get_meta("k") == "c" and s.get_meta("other") is None
    finally:
        s.close()
    s = Store(settings)                                                             # and kept across a reopen
    try:
        assert s.get_meta("k") == "c"
    finally:
        s.close()


# ---------------------------------------------------------------- visual localizations (D30)


def loc_rec(pid, status=LOC_ACCEPTED, landmark_id="tower", **kw):
    if status == LOC_ACCEPTED:
        kw = {"model_id": "m", "pose": {"enu": [1.25, -30.5, 1.6], "heading_deg": 12.5, "pitch_deg": 3.0,
                                        "roll_deg": -0.5, "hfov_deg": 63.4, "aspect": 1.3333},
              "stats": {"inliers": 120, "gps_conflict": False, "candidates": [{"model_id": "m", "reason": None}]},
              **kw}
    return LocRecord(**{"photo_id": pid, "landmark_id": landmark_id, "status": status, "updated_at": 5.0,
                        "maps_key": "k1", "attempts": 1, **kw})


def test_localizations_round_trip_by_landmark_and_only_for_photos_whose_location_may_be_shown(store):
    store.create_landmark(landmark())
    store.create_landmark(replace(landmark(), id="other"))
    store.add_photo(photo("s", location_consent=LOCATION_CONSENT_V1))
    store.add_photo(photo("v", location_consent=LOCATION_CONSENT_V2))
    store.add_photo(photo("n"))                                                     # no consent
    store.add_photo(photo("r", status=PHOTO_REJECTED, location_consent=LOCATION_CONSENT_V1))
    store.add_photo(photo("u", location_consent="share-location-v0"))              # a consent text we never showed
    store.add_photo(photo("o", landmark_id="other", location_consent=LOCATION_CONSENT_V1))
    rec = loc_rec("s")
    assert store.put_localization(rec)
    assert store.get_localization("s") == rec and store.localizations("tower") == {"s": rec}    # JSON round trip
    # an upsert replaces every field (one row per photo)
    again = loc_rec("s", LOC_REJECTED, reason="too_few_matches", attempts=2, maps_key="k2", updated_at=6.0)
    assert store.put_localization(again) and store.localizations("tower") == {"s": again}
    assert store.get_localization("s").pose is None and store.get_localization("s").stats == {}
    # never for a photo whose location may not be shown, nor under another landmark, nor for an unknown photo
    for refused in (loc_rec("n"), loc_rec("r"), loc_rec("u"), loc_rec("o"), loc_rec("missing")):
        assert not store.put_localization(refused)
        assert store.get_localization(refused.photo_id) is None
    assert store.put_localization(loc_rec("v", LOC_SKIPPED, reason="no_focal"))
    assert store.put_localization(loc_rec("o", landmark_id="other"))
    assert set(store.localizations("tower")) == {"s", "v"} and set(store.localizations("other")) == {"o"}
    with pytest.raises(ValueError):
        store.put_localization(replace(rec, status="maybe"))
    for broken in (replace(rec, pose=None), replace(rec, model_id=None)):
        with pytest.raises(ValueError):                                             # accepted needs model and pose
            store.put_localization(broken)


def test_withdrawing_or_clearing_consents_deletes_the_localizations(store):
    """A localization exists only because of the consent (D30): it goes with it, and cannot come back."""
    store.create_landmark(landmark())
    for pid in "pqr":
        store.add_photo(photo(pid, location_consent=LOCATION_CONSENT_V1))
        assert store.put_localization(loc_rec(pid))
    store.withdraw_location_consent("p")
    assert store.get_localization("p") is None and set(store.localizations("tower")) == {"q", "r"}
    assert not store.put_localization(loc_rec("p"))                                 # a late localizer: refused
    assert store.clear_location_consents() == 2
    assert store.localizations("tower") == {} and not store.put_localization(loc_rec("q"))


def test_a_withdrawal_takes_the_consent_and_the_localization_together_or_neither(store):
    """The two statements of a withdrawal commit together: if deleting the localization fails, the consent stands
    (the uploader can try again) rather than a withdrawn photo keeping its pose on record."""
    store.create_landmark(landmark())
    for pid in "pq":
        store.add_photo(photo(pid, location_consent=LOCATION_CONSENT_V1))
        assert store.put_localization(loc_rec(pid))
    real = store._conn

    class FailingDelete:
        def __getattr__(self, name):
            return getattr(real, name)

        def execute(self, sql, *args):
            if sql.startswith("DELETE FROM localizations"):
                raise sqlite3.OperationalError("disk I/O error")
            return real.execute(sql, *args)

    store._conn = FailingDelete()
    try:
        with pytest.raises(sqlite3.OperationalError):
            store.withdraw_location_consent("p")
        with pytest.raises(sqlite3.OperationalError):
            store.clear_location_consents()
    finally:
        store._conn = real
    assert store.get_photo("p").location_consent == store.get_photo("q").location_consent == LOCATION_CONSENT_V1
    assert set(store.localizations("tower")) == {"p", "q"}
    with pytest.raises(NotFound):
        store.withdraw_location_consent("missing")
    store.withdraw_location_consent("p")                                            # and the store still works
    assert store.get_localization("p") is None and store.get_photo("p").location_consent is None
