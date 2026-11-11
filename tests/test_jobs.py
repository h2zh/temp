import hashlib
import json
import math
import shutil
import sys
import threading
import time
from pathlib import Path

import pytest

from crowd3d.config import Settings
from crowd3d.coverage import sector_center_deg, sector_for_bearing
from crowd3d.geo import LocalENU, normalize_deg
from crowd3d.jobs import JobRunner, superseded_by
from crowd3d.reconstruct import FakeReconstructor, ReconstructionResult
from crowd3d.records import (
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_FULL,
    JOB_QUEUED,
    JOB_LOCALIZE,
    JOB_RUNNING,
    JOB_STYLIZE,
    PHOTO_ACCEPTED,
    TERMS_V1,
    Job,
    Landmark,
    PhotoRecord,
)
from crowd3d.store import Store

LAT, LON = 31.2397, 121.4998
LM = "tower"


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data", reconstructor="fake")


@pytest.fixture
def store(settings):
    s = Store(settings)
    s.create_landmark(Landmark(id=LM, name="Tower", lat=LAT, lon=LON, radius_m=150.0, created_at=0.0))
    yield s
    s.close()


@pytest.fixture
def make_runner(store, settings):
    runners = []

    def make(reconstructor=None, **overrides):
        extra = {"localizer": overrides.pop("localizer")} if "localizer" in overrides else {}
        s = Settings(**{**settings.__dict__, **overrides})
        r = JobRunner(store, s, reconstructor or FakeReconstructor(s), **extra)
        runners.append(r)
        return r

    yield make
    for r in runners:
        r.stop()


_seq = iter(range(10**6))


def add_photos(store, settings, sector, count, landmark_id=LM, **fields):
    """Accepted photos 30 m out in the middle of a sector, with fake processed copies on disk (fields: more
    PhotoRecord fields)."""
    enu = LocalENU(LAT, LON)
    centre = sector_center_deg(sector, settings.num_sectors)
    out = []
    for i in range(count):
        n = next(_seq)
        bearing = normalize_deg(centre - 8 + 16 * i / max(1, count - 1))
        b = math.radians(bearing)
        lat, lon, _ = enu.to_geodetic(30 * math.sin(b), 30 * math.cos(b))
        rel = f"photos/{landmark_id}/p{n:05d}.proc.jpg"
        (settings.data_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (settings.data_dir / rel).write_bytes(b"not really a jpeg")
        out.append(store.add_photo(PhotoRecord(
            id=f"p{n:05d}", landmark_id=landmark_id, user_id=f"u{n % 3}", status=PHOTO_ACCEPTED,
            uploaded_at=time.time(), lat=lat, lon=lon, heading_deg=normalize_deg(bearing + 180),
            bearing_deg=bearing, distance_m=30.0, sector=sector_for_bearing(bearing, settings.num_sectors),
            processed_path=rel, **fields,
        )))
    return out


class RaisingOnce(FakeReconstructor):
    name = "raising"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir):
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("boom")
        return super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)


class Blocking(FakeReconstructor):
    """Holds every job until released."""

    name = "blocking"

    def __init__(self):
        super().__init__()
        self.started = threading.Event()
        self.release = threading.Event()
        self.inputs: list[list[str]] = []

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir):
        self.inputs.append([p.photo_id for p in photos])
        self.started.set()
        assert self.release.wait(10)
        return super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)


class CacheUser(FakeReconstructor):
    """Declares that it uses the feature cache and records what the runner hands it."""

    name = "cache-user"
    feature_cache = True

    def __init__(self):
        super().__init__()
        self.caches = []

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, *, cache=None):
        self.caches.append(cache)
        return super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)


def test_the_feature_cache_reaches_only_reconstructors_that_use_it_when_on(store, settings, make_runner):
    from crowd3d import params
    from crowd3d.landmark_map import CACHE_FILE

    user = CacheUser()
    runner = make_runner(user, inline_jobs=True)
    add_photos(store, settings, 2, 4)
    with params.override(feature_cache=False):
        runner.schedule(LM)
    assert user.caches == [None]                                   # Params.feature_cache off
    with params.override(feature_cache=True):                      # (the default)
        add_photos(store, settings, 2, 3)
        runner.schedule(LM)
        assert user.caches[-1].path == settings.maps_dir / LM / CACHE_FILE
        fake = make_runner(inline_jobs=True)                       # FakeReconstructor: no cache, no maps dir
        add_photos(store, settings, 6, 4)
        (job,) = fake.schedule(LM)
        assert job.status == JOB_DONE and not settings.maps_dir.exists()


def test_each_photos_compass_reference_reaches_the_reconstructor(store, settings, make_runner):
    """ReconPhoto.heading_ref is the row's EXIF GPSImgDirectionRef (D34): what a reading not tagged true north counts
    for is the georegistration's decision (Params.compass_other), so the tag has to travel with the heading."""
    class Recording(FakeReconstructor):
        seen = None

        def reconstruct(self, photos, *args, **kw):
            Recording.seen = photos
            return super().reconstruct(photos, *args, **kw)

    runner = make_runner(Recording(), inline_jobs=True)
    ids = {ref: [p.id for p in add_photos(store, settings, 2, 2, **({"heading_ref": ref} if ref else {}))]
           for ref in ("T", "M", None)}
    # a photo validation kept without believing its heading (Settings.facing_away_warns_other_refs): no compass
    (warned,) = add_photos(store, settings, 2, 1, heading_ref="M", warnings=["heading_away_unverified"])
    (job,) = runner.schedule(LM)
    assert job.status == JOB_DONE
    got = {p.photo_id: (p.heading_ref, p.heading_deg) for p in Recording.seen}
    assert len(got) == 7 and got.pop(warned.id) == ("M", None)
    assert store.get_photo(warned.id).heading_deg is not None                      # the row keeps the reading
    assert all(heading is not None for _, heading in got.values())
    assert {ref: [got[i][0] for i in pids] for ref, pids in ids.items()} == {"T": ["T", "T"], "M": ["M", "M"],
                                                                             None: [None, None]}


def test_inline_schedule_runs_job_and_writes_model(store, settings, make_runner):
    runner = make_runner(inline_jobs=True)
    photos = add_photos(store, settings, 2, 4)
    (job,) = runner.schedule(LM)
    assert (job.kind, job.sector, job.status, job.error) == (JOB_FACE, 2, JOB_DONE, None)
    assert job.photo_ids == [p.id for p in photos]
    assert job.started_at <= job.finished_at
    (model,) = store.list_models(LM)
    assert (model.kind, model.sector, model.job_id, model.num_photos) == (JOB_FACE, 2, job.id, 4)
    assert model.ply_path == f"models/{LM}/{model.id}.ply"
    assert (settings.data_dir / model.ply_path).is_file()
    meta = json.loads((settings.models_dir / LM / f"{model.id}.json").read_text())
    assert meta["ply_path"] == model.ply_path and meta["ok"] and meta["frame"] == "ENU"
    assert meta == model.meta
    assert set(meta["cameras"]) == {p.id for p in photos}
    # Nothing new to do until the sector grows.
    assert runner.schedule(LM) == []
    add_photos(store, settings, 2, 2)
    assert runner.schedule(LM) == []
    add_photos(store, settings, 2, 1)
    (job2,) = runner.schedule(LM)
    assert len(job2.photo_ids) == 7 and job2.status == JOB_DONE


def test_background_full_flow(store, settings, make_runner):
    runner = make_runner()
    runner.start()
    for s in range(6):
        add_photos(store, settings, s, 4)
    jobs = runner.schedule(LM)
    assert [(j.kind, j.sector) for j in jobs] == [(JOB_FACE, s) for s in range(6)] + [(JOB_FULL, None)]
    assert all(j.status == JOB_QUEUED for j in jobs)
    assert runner.drain(10)
    assert [j.status for j in store.list_jobs(LM)] == [JOB_DONE] * 7
    expected = [(JOB_FULL, None)] + [(JOB_FACE, s) for s in range(6)]
    assert [(m.kind, m.sector) for m in store.latest_models(LM)] == expected
    full = store.latest_models(LM)[0]
    assert full.num_photos == 24 and full.meta["num_registered"] == 24


def test_worker_survives_raising_reconstructor(store, settings, make_runner):
    recon = RaisingOnce()
    runner = make_runner(recon)
    runner.start()
    add_photos(store, settings, 0, 4)
    (job,) = runner.schedule(LM)
    assert runner.drain(10)
    failed = store.get_job(job.id)
    assert failed.status == JOB_FAILED and failed.error == "RuntimeError: boom" and failed.finished_at
    assert store.list_models(LM) == []
    # Failed attempts count: no automatic retry without growth, but a manual rebuild works.
    assert runner.schedule(LM) == []
    retry = runner.submit_manual(LM, JOB_FACE, 0)
    assert runner.drain(10)
    assert store.get_job(retry.id).status == JOB_DONE
    assert runner._thread.is_alive()


def test_result_not_ok_marks_failed(store, settings, make_runner):
    runner = make_runner(inline_jobs=True, face_min_photos=1)
    add_photos(store, settings, 4, 1)
    (job,) = runner.schedule(LM)
    assert job.status == JOB_FAILED and job.error == "need at least 2 photos"
    assert store.list_models(LM) == []


def test_missing_processed_copy_marks_failed(store, settings, make_runner):
    runner = make_runner(inline_jobs=True)
    photos = add_photos(store, settings, 1, 4)
    store._conn.execute("UPDATE photos SET processed_path = NULL WHERE id = ?", (photos[0].id,))
    (job,) = runner.schedule(LM)
    assert job.status == JOB_FAILED and "no processed copy" in job.error


def test_concurrent_schedule_creates_no_duplicates(store, settings, make_runner, monkeypatch):
    runner = make_runner()   # not started: jobs stay queued
    for s in range(6):
        add_photos(store, settings, s, 4)
    list_jobs = store.list_jobs

    def slow_list_jobs(*args, **kwargs):
        # Widen the read-plan-create window so a missing lock reliably shows up as duplicates.
        jobs = list_jobs(*args, **kwargs)
        time.sleep(0.005)
        return jobs

    monkeypatch.setattr(store, "list_jobs", slow_list_jobs)
    barrier = threading.Barrier(8)
    results = []

    def call():
        barrier.wait()
        results.append(runner.schedule(LM))

    threads = [threading.Thread(target=call) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    jobs = store.list_jobs(LM)
    expected = [(JOB_FACE, s) for s in range(6)] + [(JOB_FULL, None)]
    assert sorted((j.kind, j.sector or 0) for j in jobs) == sorted((k, s or 0) for k, s in expected)
    assert sum(len(r) for r in results) == 7


def test_photos_arriving_during_a_job_trigger_a_follow_up(store, settings, make_runner):
    recon = Blocking()
    runner = make_runner(recon)
    runner.start()
    first = add_photos(store, settings, 3, 4)
    (job,) = runner.schedule(LM)
    assert recon.started.wait(5)
    assert store.get_job(job.id).status == JOB_RUNNING
    later = add_photos(store, settings, 3, 3)
    assert runner.schedule(LM) == []          # blocked by the running job
    assert runner.drain(0.2) is False
    recon.release.set()
    assert runner.drain(10)
    assert recon.inputs == [[p.id for p in first], [p.id for p in first + later]]
    assert [j.status for j in store.list_jobs(LM)] == [JOB_DONE, JOB_DONE]


def test_submit_manual(store, settings, make_runner):
    runner = make_runner()
    add_photos(store, settings, 5, 3)
    with pytest.raises(ValueError, match="sector 5 has 3"):
        runner.submit_manual(LM, JOB_FACE, 5)
    with pytest.raises(ValueError, match="full model needs"):
        runner.submit_manual(LM, JOB_FULL)
    with pytest.raises(ValueError):
        runner.submit_manual(LM, JOB_FACE)
    with pytest.raises(ValueError):
        runner.submit_manual(LM, JOB_FACE, 8)
    with pytest.raises(ValueError):
        runner.submit_manual(LM, "roof")
    add_photos(store, settings, 5, 1)
    job = runner.submit_manual(LM, JOB_FACE, 5)
    assert (job.kind, job.sector, job.status, len(job.photo_ids)) == (JOB_FACE, 5, JOB_QUEUED, 4)
    # Same inputs while still queued: no duplicate.
    assert runner.submit_manual(LM, JOB_FACE, 5).id == job.id
    runner.start()
    assert runner.drain(10)
    # Regardless of growth thresholds once the previous one finished.
    again = runner.submit_manual(LM, JOB_FACE, 5)
    assert again.id != job.id
    assert runner.drain(10)
    assert [j.status for j in store.list_jobs(LM)] == [JOB_DONE, JOB_DONE]


def test_submit_manual_full_inline(store, settings, make_runner):
    runner = make_runner(inline_jobs=True)
    for s in range(6):
        add_photos(store, settings, s, 4)
    job = runner.submit_manual(LM, JOB_FULL, sector=3)   # sector is ignored for full jobs
    assert (job.kind, job.sector, job.status, len(job.photo_ids)) == (JOB_FULL, None, JOB_DONE, 24)


def _leftover_jobs(store, settings):
    photos = add_photos(store, settings, 0, 4) + add_photos(store, settings, 4, 4)
    queued = store.create_job(Job(id="queued", landmark_id=LM, kind=JOB_FACE, sector=0, status=JOB_QUEUED,
                                  photo_ids=[p.id for p in photos[:4]], created_at=1.0))
    running = store.create_job(Job(id="running", landmark_id=LM, kind=JOB_FACE, sector=4, status=JOB_RUNNING,
                                   photo_ids=[p.id for p in photos[4:]], created_at=2.0, started_at=2.5))
    return queued, running


def test_restart_requeues_leftover_jobs(store, settings, make_runner):
    _leftover_jobs(store, settings)
    runner = make_runner()
    runner.start()
    runner.start()   # idempotent
    assert runner.drain(10)
    jobs = {j.id: j for j in store.list_jobs(LM)}
    assert set(jobs) == {"queued", "running"}
    assert all(j.status == JOB_DONE for j in jobs.values())
    assert jobs["running"].started_at > 2.5
    assert len(store.list_models(LM)) == 2


def test_restart_inline_runs_leftovers(store, settings, make_runner):
    _leftover_jobs(store, settings)
    runner = make_runner(inline_jobs=True)
    runner.start()
    assert [j.status for j in store.list_jobs(LM)] == [JOB_DONE, JOB_DONE]
    assert runner.drain(0)


def test_stop_leaves_queued_jobs_for_next_start(store, settings, make_runner):
    recon = Blocking()
    runner = make_runner(recon)
    runner.start()
    add_photos(store, settings, 0, 4)
    add_photos(store, settings, 4, 4)
    first, second = runner.schedule(LM)
    assert recon.started.wait(5)
    stopper = threading.Thread(target=runner.stop)
    stopper.start()
    deadline = time.monotonic() + 5
    while not runner._stopping.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)
    recon.release.set()   # only now, so the worker cannot pick up the second job
    stopper.join()
    assert store.get_job(first.id).status == JOB_DONE
    assert store.get_job(second.id).status == JOB_QUEUED
    runner2 = make_runner()
    runner2.start()
    assert runner2.drain(10)
    assert store.get_job(second.id).status == JOB_DONE


class TwinFacadeOnRebuild(FakeReconstructor):
    """First model fine; later ones put every other camera in front of a look-alike facade."""

    name = "twins"

    def __init__(self):
        super().__init__()
        self.calls = 0

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir):
        self.calls += 1
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "database.db").write_bytes(b"x" * 1000)
        result = super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)
        if self.calls > 1:
            for k, cam in enumerate(result.cameras.values()):
                if k % 2:
                    cam.update(enu=[cam["enu"][1], -cam["enu"][0], 1.6], compass_offset_deg=-90.0,
                               flags=["gps_and_compass"])
        return result


def test_a_rebuild_that_fails_the_quality_gate_keeps_the_previous_model(store, settings, make_runner):
    # Baptistry: the E facade rebuild registered 6 of 18 photos on the wrong octagon face, was published
    # anyway and replaced a good model; the landmark page then showed those photos up to 74 m off.
    from crowd3d.api import camera_estimates

    runner = make_runner(TwinFacadeOnRebuild(), inline_jobs=True)
    first = add_photos(store, settings, 2, 4)
    (good,) = runner.schedule(LM)
    assert good.status == JOB_DONE
    later = add_photos(store, settings, 2, 3)
    (bad,) = runner.schedule(LM)
    assert bad.status == JOB_FAILED and bad.error.startswith("quality check failed")
    assert "the previous model stays published" in bad.error
    assert "look-alike facades" in bad.error
    models = store.list_models(LM)
    assert [m.meta["published"] for m in models] == [True, False]
    (latest,) = store.latest_models(LM)
    assert latest.job_id == good.id
    estimates = camera_estimates(store.published_models(LM))
    assert set(estimates) == {p.id for p in first}           # nothing from the rejected model
    assert not {p.id for p in later} & set(estimates)
    assert not (settings.work_dir / good.id).exists() and not (settings.work_dir / bad.id).exists()


def test_a_first_model_that_fails_the_quality_gate_does_not_claim_a_previous_one(store, settings, make_runner):
    # Aachen (real photos): every facade job failed QA and said "the previous model stays published",
    # although nothing had ever been published for those sectors.
    class AlwaysTwin(TwinFacadeOnRebuild):
        def __init__(self):
            super().__init__()
            self.calls = 1   # flag cameras from the first call on

    runner = make_runner(AlwaysTwin(), inline_jobs=True)
    add_photos(store, settings, 2, 4)
    (job,) = runner.schedule(LM)
    assert job.status == JOB_FAILED and job.error.startswith("quality check failed")
    assert "nothing is published for it yet" in job.error
    assert "previous model" not in job.error
    assert store.latest_models(LM) == []


def test_flagged_cameras_are_not_shown_and_an_older_estimate_stays(store, settings, make_runner):
    from crowd3d.api import camera_estimates

    class OneFlagged(FakeReconstructor):
        calls = 0

        def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir):
            OneFlagged.calls += 1
            result = super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)
            if OneFlagged.calls > 1:
                cam = result.cameras[photos[0].photo_id]
                cam.update(enu=[400.0, 0.0, 1.6], flags=["far_from_gps"])
            return result

    runner = make_runner(OneFlagged(), inline_jobs=True)
    photos = add_photos(store, settings, 2, 4)
    runner.schedule(LM)
    before = camera_estimates(store.published_models(LM))[photos[0].id]
    add_photos(store, settings, 2, 3)
    (job,) = runner.schedule(LM)
    assert job.status == JOB_DONE                              # 1 of 7 flagged: published
    newest = store.latest_models(LM)[0].meta["cameras"][photos[0].id]
    assert set(newest["flags"]) == {"far_from_gps", "beyond_radius"}
    assert camera_estimates(store.published_models(LM))[photos[0].id] == before


def test_keep_work_dirs(store, settings, make_runner):
    runner = make_runner(TwinFacadeOnRebuild(), inline_jobs=True, keep_work_dirs=True)
    add_photos(store, settings, 2, 4)
    (job,) = runner.schedule(LM)
    assert (settings.work_dir / job.id / "database.db").is_file()


def test_a_job_the_process_keeps_dying_in_is_failed_not_rerun(store, settings, make_runner):
    # Exceptions are caught; an OOM kill or a native crash leaves the job "running". Re-running it on
    # every boot would crash-loop a supervised server.
    photos = add_photos(store, settings, 0, 4) + add_photos(store, settings, 4, 4)
    store.create_job(Job(id="once", landmark_id=LM, kind=JOB_FACE, sector=0, status=JOB_RUNNING,
                         photo_ids=[p.id for p in photos[:4]], created_at=1.0, started_at=1.5, attempts=1))
    store.create_job(Job(id="twice", landmark_id=LM, kind=JOB_FACE, sector=4, status=JOB_RUNNING,
                         photo_ids=[p.id for p in photos[4:]], created_at=2.0, started_at=2.5, attempts=2))
    runner = make_runner(inline_jobs=True)
    runner.start()
    once, twice = store.get_job("once"), store.get_job("twice")
    assert (once.status, once.attempts) == (JOB_DONE, 2)
    assert twice.status == JOB_FAILED and "interrupted 2 times" in twice.error
    assert [j.id for j in store.list_jobs(LM)] == ["once", "twice"]    # no automatic retry without growth


def test_start_plans_photos_whose_request_died_before_planning(store, settings, make_runner):
    # Photos were committed but the request died (ENOSPC, crash) before runner.schedule(); a client
    # retry is rejected as a duplicate, so only a restart can still plan them.
    add_photos(store, settings, 6, 4)
    assert store.list_jobs(LM) == []
    runner = make_runner(inline_jobs=True)
    runner.start()
    (job,) = store.list_jobs(LM)
    assert (job.kind, job.sector, job.status) == (JOB_FACE, 6, JOB_DONE)


# ---------------------------------------------------------------- versioned inputs, the supersede rule, saved maps


GOLDEN_SPARSE = Path(__file__).parent / "fixtures" / "recon_golden" / "sparse"


class SparseWriter(TwinFacadeOnRebuild):
    """A fake that also leaves a real COLMAP sparse model (the golden fixture's) in its work dir, as COLMAP jobs do;
    twins=True makes its later models fail the quality gate."""

    name = "sparse-writer"

    def __init__(self, twins: bool = False):
        super().__init__()
        self.twins = twins

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, **_):
        result = (super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir) if self.twins
                  else FakeReconstructor.reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir))
        shutil.copytree(GOLDEN_SPARSE, work_dir / "sparse" / "0", dirs_exist_ok=True)     # (a rerun: the same dir)
        return result


def _job(store, photos, kind=JOB_FACE, sector=2):
    return store.create_job(Job(id=f"j{next(_seq)}", landmark_id=LM, kind=kind, sector=sector, status=JOB_QUEUED,
                                photo_ids=[p.id for p in photos], created_at=time.time()))


def test_models_record_their_inputs(store, settings, make_runner):
    runner = make_runner(inline_jobs=True)
    photos = add_photos(store, settings, 2, 4)
    runner.schedule(LM)
    (model,) = store.list_models(LM)
    ids = sorted(p.id for p in photos)
    assert model.meta["inputs"] == {"count": 4, "max_uploaded_at": max(p.uploaded_at for p in photos),
                                    "photo_set_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def test_an_older_job_finishing_late_never_replaces_a_newer_model(store, settings, make_runner):
    from crowd3d.api import camera_estimates

    runner = make_runner(inline_jobs=True)
    first = add_photos(store, settings, 2, 4)
    older = _job(store, first)
    later = add_photos(store, settings, 2, 3)
    newer = _job(store, first + later)
    assert runner.run_job(newer).status == JOB_DONE              # the newer job finishes first
    shown = camera_estimates(store.published_models(LM))
    late = runner.run_job(older)
    assert late.status == JOB_DONE and late.error.startswith("superseded: model ")
    models = {m.job_id: m for m in store.list_models(LM)}
    assert models[older.id].meta["published"] is False and models[older.id].meta["qa"]["passed"]
    assert models[older.id].meta["superseded_by"] == models[newer.id].id
    assert [m.job_id for m in store.latest_models(LM)] == [newer.id]
    assert camera_estimates(store.published_models(LM)) == shown
    # the rule itself: newer means photos the other lacks, uploaded after all of its own
    assert superseded_by(store, LM, JOB_FACE, 2, [p.id for p in first]).job_id == newer.id
    assert superseded_by(store, LM, JOB_FACE, 2, [p.id for p in first + later]) is None    # the same inputs
    assert superseded_by(store, LM, JOB_FACE, 3, [p.id for p in first]) is None            # another target
    more = add_photos(store, settings, 2, 1)
    assert superseded_by(store, LM, JOB_FACE, 2, [p.id for p in first + more]) is None     # newer than it


def test_published_colmap_models_are_saved_to_the_map_newest_two_per_target(store, settings, make_runner):
    pytest.importorskip("pycolmap")
    from crowd3d import landmark_map as lmap

    runner = make_runner(SparseWriter(), inline_jobs=True)
    root = settings.maps_dir / LM / lmap.MODELS_DIR
    jobs = []
    for count in (4, 3, 3):
        add_photos(store, settings, 2, count)
        (job,) = runner.schedule(LM)
        assert job.status == JOB_DONE
        jobs.append(job)
    models = {m.job_id: m for m in store.list_models(LM)}
    saved = lmap.saved_maps(root)
    assert [m["job_id"] for m in saved] == [j.id for j in jobs[1:]]           # the oldest one was pruned
    doc = saved[-1]
    model = models[jobs[-1].id]
    assert sorted(p.name for p in (root / model.id).iterdir()) == ["map.json", "sparse"]
    assert (doc["schema"], doc["model_id"], doc["kind"], doc["sector"]) == (lmap.MAP_SCHEMA, model.id, JOB_FACE, 2)
    assert doc["anchor"] == model.meta["anchor"] and doc["inputs"] == model.meta["inputs"]
    assert doc["feature_version"] == lmap.job_feature_version(settings.processing_max_side_px)
    assert (doc["mode"], doc["parent_model_id"], doc["continuations"], doc["n_scratch"]) == ("scratch", None, 0, 10)
    assert len(doc["images"]) == 4 and all(name.endswith(f"_{pid}.jpg") for name, pid in doc["images"].items())
    assert not settings.work_dir.exists() or not any(settings.work_dir.iterdir())     # work dirs still go


def test_no_map_for_unpublished_models_and_none_half_written(store, settings, make_runner, monkeypatch):
    pytest.importorskip("pycolmap")
    from crowd3d import landmark_map as lmap

    runner = make_runner(SparseWriter(twins=True), inline_jobs=True)
    root = settings.maps_dir / LM / lmap.MODELS_DIR
    add_photos(store, settings, 2, 4)
    (good,) = runner.schedule(LM)
    add_photos(store, settings, 2, 3)
    (bad,) = runner.schedule(LM)                                 # fails the quality gate: no map
    assert (good.status, bad.status) == (JOB_DONE, JOB_FAILED)
    assert [m["job_id"] for m in lmap.saved_maps(root)] == [good.id]
    (root / ".tmp-dead-1234").mkdir()                            # a process that died while saving
    monkeypatch.setattr(lmap.os, "replace", lambda *a: (_ for _ in ()).throw(OSError("killed")))
    runner2 = make_runner(SparseWriter(), inline_jobs=True)
    add_photos(store, settings, 2, 3)
    (third,) = runner2.schedule(LM)
    assert third.status == JOB_DONE                              # the model is published even so
    assert [m["job_id"] for m in lmap.saved_maps(root)] == [good.id]
    assert [p.name for p in root.iterdir()] == [lmap.saved_maps(root)[0]["model_id"]]   # no temp dirs left


# ---------------------------------------------------------------- continuing the target's published model (stage 1b)


class Continuer(SparseWriter):
    """A fake that continues: it records the base each call is handed and says it continued it (accept=False: it
    rejected the base and built from scratch); flag_continued makes every continued model fail the quality gate."""

    name = "continuer"
    map_continue = True

    def __init__(self, accept: bool = True, flag_continued: bool = False):
        super().__init__()
        self.accept, self.flag_continued = accept, flag_continued
        self.bases: list = []

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, *, base=None, **_):
        self.bases.append(base)
        result = super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)
        if base is not None:
            result.continued = {"base_model_id": base.model_id, "accepted": self.accept,
                                "reason": None if self.accept else "it kept 1 of the base's 4 registered photos",
                                "base_registered": 4, "base_kept": 4 if self.accept else 1, "registered": len(photos)}
            if self.accept and self.flag_continued:
                for k, cam in enumerate(result.cameras.values()):
                    if k % 2:
                        cam.update(compass_offset_deg=-90.0, flags=["gps_and_compass"])
        return result


def _models(store):
    return {m.job_id: m for m in store.list_models(LM)}


def test_continuity_of_a_version_against_its_parent():
    from crowd3d.jobs import continuity

    parent = {"a": {"enu": [0, 0, 1.6], "heading_deg": 350.0, "flags": []},
              "b": {"enu": [10, 0, 1.6], "heading_deg": 90.0, "flags": []},
              "c": {"enu": [5, 5, 1.6], "heading_deg": 0.0, "flags": ["far_from_gps"]}}
    assert continuity(parent, parent) == {"cameras": 2, "shift_median_m": 0.0, "heading_change_median_deg": 0.0}
    moved = {"a": {"enu": [0, 3, 1.6], "heading_deg": 2.0, "flags": []},
             "b": {"enu": [10, 1, 1.6], "heading_deg": 94.0, "flags": []},
             "c": {"enu": [50, 5, 1.6], "heading_deg": 0.0, "flags": []}, "d": {"enu": [1, 1, 1], "flags": []}}
    assert continuity(parent, moved) == {"cameras": 2, "shift_median_m": 2.0, "heading_change_median_deg": 8.0}
    assert continuity({}, moved)["cameras"] == 0


def test_without_map_continue_no_base_is_handed_and_no_model_records_a_map(store, settings, make_runner):
    pytest.importorskip("pycolmap")
    from crowd3d import params

    user = Continuer()
    runner = make_runner(user, inline_jobs=True)
    first = add_photos(store, settings, 2, 4)
    assert runner.run_job(_job(store, first)).status == JOB_DONE
    second = add_photos(store, settings, 2, 1)
    assert params.current().map_continue is False                       # the default
    assert runner.run_job(_job(store, first + second)).status == JOB_DONE
    assert user.bases == [None, None] and not any("map" in m.meta for m in store.list_models(LM))
    with params.override(map_continue=True):                            # a reconstructor that cannot continue
        fake = make_runner(SparseWriter(), inline_jobs=True)
        assert fake.run_job(_job(store, first + second)).status == JOB_DONE
    assert not any("map" in m.meta for m in store.list_models(LM))


def test_a_job_continues_the_model_its_target_shows(store, settings, make_runner):
    pytest.importorskip("pycolmap")
    from crowd3d import landmark_map as lmap
    from crowd3d import params

    user = Continuer()
    runner = make_runner(user, inline_jobs=True)
    root = settings.maps_dir / LM / lmap.MODELS_DIR
    with params.override(map_continue=True):
        first = add_photos(store, settings, 2, 4)
        j1 = runner.run_job(_job(store, first))
        more = add_photos(store, settings, 2, 1)
        j2 = runner.run_job(_job(store, first + more))
    assert j1.status == j2.status == JOB_DONE
    m1, m2 = _models(store)[j1.id], _models(store)[j2.id]
    assert user.bases[0] is None and user.bases[1].model_id == m1.id
    assert user.bases[1].sparse_dir == root / m1.id / "sparse"
    assert (user.bases[1].n_scratch, user.bases[1].continuations) == (4, 0)
    assert m1.meta["map"] == {"mode": "scratch", "reason": "nothing is published for this target yet",
                              "base_model_id": None, "parent_model_id": None, "n_scratch": 4, "continuations": 0}
    plan = m2.meta["map"]
    assert (plan["mode"], plan["parent_model_id"], plan["n_scratch"], plan["continuations"]) == ("continued", m1.id,
                                                                                               4, 1)
    assert plan["continuity"] == {"cameras": 4, "shift_median_m": 0.0, "heading_change_median_deg": 0.0}
    assert m2.meta["continued"]["accepted"] and m2.meta["published"]
    doc = lmap.load_map(root, m2.id)
    assert (doc["mode"], doc["parent_model_id"], doc["n_scratch"], doc["continuations"]) == ("continued", m1.id, 4, 1)
    assert lmap.load_map(root, "no-such-model") is None


def test_scratch_rebuilds_at_the_growth_ratio_after_the_continuation_limit_and_without_a_base(store, settings,
                                                                                             make_runner, monkeypatch):
    pytest.importorskip("pycolmap")
    from crowd3d import jobs as jobs_module
    from crowd3d import params

    user = Continuer()
    runner = make_runner(user, inline_jobs=True)
    photos = add_photos(store, settings, 2, 4)
    modes = []

    def run(new: int, **overrides):
        photos.extend(add_photos(store, settings, 2, new) if new else [])
        with params.override(map_continue=True, **overrides):
            job = runner.run_job(_job(store, photos))
        plan = _models(store)[job.id].meta["map"]
        modes.append((plan["mode"], plan["continuations"], plan["n_scratch"]))
        return plan

    run(0)
    assert run(1)["mode"] == "continued"                                 # 5 < 1.5 x 4
    assert "at least 1.5 x the 4 of the last build" in run(1)["reason"]  # 6 photos: from scratch
    for _ in range(4):
        run(1, map_rebuild_ratio=100.0)
    assert "continuation 4 in a row" in run(1, map_rebuild_ratio=100.0)["reason"]
    assert modes == [("scratch", 0, 4), ("continued", 1, 4), ("scratch", 0, 6), ("continued", 1, 6),
                     ("continued", 2, 6), ("continued", 3, 6), ("continued", 4, 6), ("scratch", 0, 11)]
    # a job without all of the published model's photos, a saved map of another feature version, no saved map
    assert run_lacking(store, runner, photos)["reason"] == "the job lacks photos of the published model"
    monkeypatch.setattr(jobs_module, "job_feature_version", lambda px: "another")
    assert "features of another version" in run(1)["reason"]
    monkeypatch.undo()
    shutil.rmtree(settings.maps_dir)
    assert run(1)["reason"] == "the published model has no saved map"
    assert [b is None for b in user.bases] == [True, False, True, False, False, False, False, True, True, True, True]


def run_lacking(store, runner, photos):
    from crowd3d import params

    with params.override(map_continue=True):
        job = runner.run_job(_job(store, photos[1:]))
    return _models(store)[job.id].meta["map"]


def test_a_continued_model_failing_the_quality_gate_is_built_once_more_from_scratch(store, settings, make_runner):
    pytest.importorskip("pycolmap")
    from crowd3d import params

    user = Continuer(flag_continued=True)
    runner = make_runner(user, inline_jobs=True)
    first = add_photos(store, settings, 2, 4)
    more = add_photos(store, settings, 2, 1)
    with params.override(map_continue=True):
        j1 = runner.run_job(_job(store, first))
        j2 = runner.run_job(_job(store, first + more))
    assert j2.status == JOB_DONE and [b is None for b in user.bases] == [True, False, True]
    assert len(store.list_models(LM)) == 2                              # the failed continued model is not stored
    m2 = _models(store)[j2.id]
    plan = m2.meta["map"]
    assert (plan["mode"], plan["parent_model_id"], plan["base_model_id"]) == ("continued->scratch", None,
                                                                            _models(store)[j1.id].id)
    assert plan["reason"].startswith("the continued model failed the quality check: ")
    assert "look-alike" in plan["reason"]
    assert (plan["n_scratch"], plan["continuations"]) == (5, 0) and "continuity" not in plan
    assert m2.meta["published"] and "continued" not in m2.meta and not m2.meta["qa"]["reasons"]


def test_a_rejected_continuation_is_recorded_as_built_from_scratch(store, settings, make_runner):
    pytest.importorskip("pycolmap")
    from crowd3d import params

    user = Continuer(accept=False)
    runner = make_runner(user, inline_jobs=True)
    first = add_photos(store, settings, 2, 4)
    more = add_photos(store, settings, 2, 1)
    with params.override(map_continue=True):
        runner.run_job(_job(store, first))
        j2 = runner.run_job(_job(store, first + more))
    plan = _models(store)[j2.id].meta["map"]
    assert len(user.bases) == 2                                         # the reconstructor itself fell back
    assert (plan["mode"], plan["n_scratch"], plan["continuations"]) == ("continued->scratch", 5, 0)
    assert plan["reason"] == "not continued: it kept 1 of the base's 4 registered photos"


# ---------------------------------------------------------------- structure in the product: stylize jobs (Phase F)

# What the stylize child does, without its seconds of work: it reads the spec, records it beside itself, and writes
# what crowd3d.structure_pipeline.run_model writes (<model id>.glb, <model id>.style.json, result.json) into the spec's
# out dir. FAKE_CHILD_MODE: "ok", "refused" (registration refused, all clay), "fail" (exit 1 with an error), "sleep",
# "qa" (a summary and a result that says which gate failed on which part, exit 1), "crash" (a traceback on stderr and
# no result at all, exit 1). Every run says a line on stdout first (the job runner's child.log).
FAKE_CHILD = r'''
import json, os, sys, time
spec_path = sys.argv[sys.argv.index("--spec") + 1]
spec = json.load(open(spec_path))
mode = os.environ.get("FAKE_CHILD_MODE", "ok")
log = os.environ["FAKE_CHILD_LOG"]
with open(log, "a") as f:
    f.write(json.dumps({"spec": spec, "at": time.time(), "cwd": os.getcwd(), "threads": os.environ.get("OMP_NUM_THREADS")}) + "\n")
out, mid = spec["out"], spec["model"]["id"]
print("fake child: stylizing", mid, flush=True)
if mode == "sleep":
    time.sleep(60)
if mode == "fail":
    json.dump({"ok": False, "error": "ValueError: no outline"}, open(os.path.join(out, "result.json"), "w"))
    sys.exit(1)
if mode == "qa":
    json.dump({"qa": {"passed": False, "structure": {"gates": {"G13_not_built": [{"id": "w2", "why": "TypeError: boom"}]}}}},
              open(os.path.join(out, mid + ".style.json"), "w"))
    json.dump({"ok": False, "error": "the stylized model failed its QA: structure", "qa_failed": ["structure"],
               "qa_detail": "structure: G13_not_built [w2: TypeError: boom]"}, open(os.path.join(out, "result.json"), "w"))
    sys.exit(1)
if mode == "crash":
    print("x" * 20000, file=sys.stderr)
    print("Traceback (most recent call last):\n  File \"roof.py\", line 9, in hipped\nIndexError: index 8 is out of bounds",
          file=sys.stderr, flush=True)
    os._exit(1)
accepted = mode == "ok"
snap = {"rotation_deg": 1.5, "shift_m": [0.5, -0.25], "scale": 1.0, "accepted": accepted, "reason": ""}
open(os.path.join(out, mid + ".glb"), "wb").write(b"glTF fake")
json.dump({"registration": {"accepted": accepted}, "snap": snap}, open(os.path.join(out, mid + ".style.json"), "w"))
json.dump({"ok": True, "error": None, "registration": {"accepted": accepted, "reason": None if accepted else "ambiguous"},
           "snap": snap,
           "massing_source": "osm-parts", "parts_built": 3, "judge": {"status": "skipped", "reason": "x"},
           "adoptions": 0, "observed_walls": 0, "num_walls": 4, "glb_bytes": 9, "seconds": 0.1, "cpu_s": 0.1,
           "peak_rss_mb": 50.0, "within_limits": True, "source": {"slug": spec["slug"], "osm_parts_sha256": "s"}},
          open(os.path.join(out, "result.json"), "w"))
'''


@pytest.fixture
def catalog(tmp_path):
    """A catalogue that lists LM (and "far-away", 2 km off) with an OSM parts snapshot and an outline."""
    path = tmp_path / "eval" / "landmarks.json"
    path.parent.mkdir(parents=True)
    far = LocalENU(LAT, LON).to_geodetic(2000.0, 0.0)
    entry = {"osm": {"file": "eval/osm/x.geojson"}, "osm_parts": {"file": "eval/osm/x.parts.osm", "sha256": "0"}}
    path.write_text(json.dumps([{"slug": LM, "lat": LAT, "lon": LON, **entry},
                                {"slug": "far-away", "lat": far[0], "lon": far[1], **entry},
                                {"slug": "no-parts", "lat": LAT, "lon": LON, "osm": entry["osm"]}]))
    return path


@pytest.fixture
def fake_child(tmp_path, monkeypatch):
    """The stylize child replaced by FAKE_CHILD; returns a reader of what each run saw (spec, time, cwd, threads)."""
    script, log = tmp_path / "fake_child.py", tmp_path / "fake_child.log"
    script.write_text(FAKE_CHILD)
    monkeypatch.setenv("FAKE_CHILD_LOG", str(log))
    monkeypatch.setattr(JobRunner, "_stylize_command",
                        lambda self, spec: [sys.executable, str(script), "run", "--spec", str(spec)])
    return lambda: [json.loads(line) for line in log.read_text().splitlines()] if log.is_file() else []


def _stylize_jobs(store):
    return [j for j in store.list_jobs(LM) if j.kind == JOB_STYLIZE]


def test_a_published_model_of_a_landmark_with_a_snapshot_is_stylized_in_a_child(store, settings, make_runner,
                                                                               catalog, fake_child, tmp_path,
                                                                               monkeypatch):
    pytest.importorskip("pycolmap")
    from crowd3d import landmark_map as lmap

    monkeypatch.chdir(tmp_path)          # a server started with relative paths: the child (in the repo) gets absolute
    runner = make_runner(SparseWriter(), inline_jobs=True, data_dir=Path("data"),
                         structure_catalog=catalog.relative_to(tmp_path))
    photos = add_photos(store, settings, 2, 4)
    (job,) = runner.schedule(LM)
    (model,) = store.list_models(LM)
    (styl,) = _stylize_jobs(store)
    assert job.status == JOB_DONE and model.meta["published"]
    assert (styl.status, styl.error, styl.model_id, styl.sector) == (JOB_DONE, None, model.id, 2)
    assert styl.photo_ids == job.photo_ids and styl.started_at >= store.get_job(job.id).finished_at
    (seen,) = fake_child()
    spec = seen["spec"]
    root = settings.maps_dir / LM / lmap.MODELS_DIR
    assert (spec["slug"], spec["catalog"], spec["landmark"]["id"]) == (LM, str(catalog), LM)
    assert spec["model"]["sparse"] == str(root / model.id / "sparse")
    assert spec["model"]["meta"] == str((settings.data_dir / model.ply_path).with_suffix(".json"))
    by_id = {p.id: str(settings.data_dir / p.processed_path) for p in photos}
    images = lmap.load_map(root, model.id)["images"]
    assert spec["model"]["images"] == {n: by_id[pid] for n, pid in images.items() if pid in by_id}
    assert spec["max_seconds"] == settings.stylize_timeout_s and spec["cache"].startswith(str(settings.maps_dir))
    assert seen["threads"] == "2" and Path(seen["cwd"]) == Path(__file__).resolve().parent.parent
    # the GLB and the summary beside the PLY; the record in the store; the work dir gone
    ply = settings.data_dir / model.ply_path
    assert ply.with_suffix(".glb").read_bytes() == b"glTF fake" and ply.with_suffix(".style.json").is_file()
    rec = store.get_model(model.id).stylized
    assert rec["glb_path"] == ply.with_suffix(".glb").relative_to(settings.data_dir).as_posix()
    assert rec["registration"] == {"accepted": True, "reason": None} and rec["job_id"] == styl.id
    # the footprint snap the GLB was built on: the public photo poses move the model's cameras with it (D30)
    assert rec["snap"] == {"rotation_deg": 1.5, "shift_m": [0.5, -0.25], "scale": 1.0, "accepted": True, "reason": ""}
    assert (rec["massing_source"], rec["parts_built"], rec["source"]["slug"]) == ("osm-parts", 3, LM)
    assert not any(settings.work_dir.iterdir())


def test_no_stylize_job_without_the_flag_a_snapshot_a_near_binding_or_a_saved_map(store, settings, make_runner,
                                                                                catalog, fake_child):
    pytest.importorskip("pycolmap")
    from crowd3d.jobs import structure_slug

    sectors = iter([0, 2, 4, 6, 0, 2])                  # a new facade (or a rebuild of one) for every case

    def after_a_publish(**overrides) -> tuple[int, int]:
        """(models published, stylize jobs created) by one planned job."""
        runner = make_runner(overrides.pop("reconstructor", SparseWriter()), inline_jobs=True, **overrides)
        add_photos(store, settings, next(sectors), 4)
        models, jobs = len(store.published_models(LM)), len(_stylize_jobs(store))
        runner.schedule(LM)
        return len(store.published_models(LM)) - models, len(_stylize_jobs(store)) - jobs

    lm = store.get_landmark(LM)
    assert after_a_publish(structure_catalog=catalog, stylize_published=False) == (1, 0)
    assert after_a_publish(structure_catalog=None) == (1, 0)
    assert after_a_publish(structure_catalog=catalog, reconstructor=FakeReconstructor()) == (1, 0)   # no saved map
    assert after_a_publish(structure_catalog=catalog, structure_bindings={LM: "no-parts"}) == (1, 0)
    assert after_a_publish(structure_catalog=catalog, structure_bindings={LM: "far-away"}) == (1, 0)
    assert after_a_publish(structure_catalog=catalog) == (1, 1)
    s = Settings(**{**settings.__dict__, "structure_catalog": catalog})
    other = Landmark(id="my-tower", name="T", lat=LAT, lon=LON, radius_m=150.0, created_at=0.0)
    assert structure_slug(s, lm) == LM and structure_slug(s, other) is None            # bound by id ...
    assert structure_slug(Settings(**{**s.__dict__, "structure_bindings": {"my-tower": LM}}), other) == LM   # ... or so
    assert structure_slug(Settings(**{**s.__dict__, "structure_catalog": catalog.parent / "missing.json"}), lm) is None


def test_reconstruction_jobs_run_before_stylize_jobs(store, settings, make_runner, catalog, fake_child):
    pytest.importorskip("pycolmap")
    runner = make_runner(SparseWriter(), structure_catalog=catalog)
    runner.start()
    add_photos(store, settings, 2, 4)
    add_photos(store, settings, 6, 4)
    assert len(runner.schedule(LM)) == 2
    assert runner.drain(30)
    recon = [j for j in store.list_jobs(LM) if j.kind != JOB_STYLIZE]
    styl = _stylize_jobs(store)
    assert len(recon) == 2 and len(styl) == 2 and {j.status for j in recon + styl} == {JOB_DONE}
    assert max(j.finished_at for j in recon) <= min(j.started_at for j in styl)


def test_a_stylize_job_waits_while_a_reconstruction_job_is_queued(store, settings, make_runner, catalog, fake_child):
    pytest.importorskip("pycolmap")
    first = make_runner(SparseWriter(), inline_jobs=True, structure_catalog=catalog, stylize_published=False)
    first.schedule(LM)                                    # nothing yet: no photos
    add_photos(store, settings, 2, 4)
    (published,) = first.schedule(LM)
    model = store.list_models(LM)[0]
    styl = store.create_job(Job(id="styl", landmark_id=LM, kind=JOB_STYLIZE, sector=2, status=JOB_QUEUED,
                                photo_ids=published.photo_ids, created_at=time.time(), model_id=model.id))
    recon = _job(store, add_photos(store, settings, 6, 4), sector=6)     # queued in the store, not handed over yet
    runner = make_runner(SparseWriter(), structure_catalog=catalog, stylize_published=False)
    runner._stopping = threading.Event()                 # the worker alone (start() would hand both jobs over)
    runner._thread = threading.Thread(target=runner._work, args=(runner._stopping,), daemon=True)
    runner._thread.start()
    runner._enqueue(styl)
    time.sleep(0.6)
    assert store.get_job("styl").status == JOB_QUEUED and fake_child() == []      # held back
    runner._enqueue(recon)
    assert runner.drain(30)
    done = {j.id: j for j in store.list_jobs(LM)}
    assert done[recon.id].status == done["styl"].status == JOB_DONE
    assert done[recon.id].finished_at <= done["styl"].started_at


def test_a_model_its_target_no_longer_shows_is_not_stylized(store, settings, make_runner, catalog, fake_child):
    pytest.importorskip("pycolmap")
    runner = make_runner(SparseWriter(), inline_jobs=True, structure_catalog=catalog, stylize_published=False)
    add_photos(store, settings, 2, 4)
    runner.schedule(LM)
    (old,) = store.list_models(LM)
    add_photos(store, settings, 2, 3)
    runner.schedule(LM)                                   # a newer model of the same target is published
    job = store.create_job(Job(id="late", landmark_id=LM, kind=JOB_STYLIZE, sector=2, status=JOB_QUEUED,
                               photo_ids=[], created_at=time.time(), model_id=old.id))
    done = runner.run_job(job)
    new = next(m for m in store.latest_models(LM))
    assert done.status == JOB_DONE and done.error == (f"superseded: model {old.id} is no longer the one its target"
                                                      f" shows (model {new.id} is); it gets no stylized GLB")
    assert fake_child() == [] and store.get_model(old.id).stylized is None


@pytest.mark.parametrize("mode", ["fail", "sleep"])
def test_a_failing_or_overrunning_child_fails_the_job_and_writes_no_glb(store, settings, make_runner, catalog,
                                                                       fake_child, monkeypatch, mode):
    pytest.importorskip("pycolmap")
    monkeypatch.setenv("FAKE_CHILD_MODE", mode)
    runner = make_runner(SparseWriter(), inline_jobs=True, structure_catalog=catalog, stylize_timeout_s=1.5)
    add_photos(store, settings, 2, 4)
    t0 = time.monotonic()
    runner.schedule(LM)
    (model,) = store.list_models(LM)
    (styl,) = _stylize_jobs(store)
    why = "ValueError: no outline" if mode == "fail" else "timed out after 1.5 s"
    kept = f"models/{LM}/{model.id}.stylize-failed.json"
    assert styl.status == JOB_FAILED
    assert styl.error == f"stylize failed, model {model.id} gets no stylized GLB: {why}; diagnostics kept in {kept}"
    assert time.monotonic() - t0 < 20                     # the overrunning child was killed at the time limit
    assert store.get_model(model.id).stylized is None
    assert sorted(p.name for p in (settings.models_dir / LM).iterdir()) == \
        [f"{model.id}.json", f"{model.id}.ply", f"{model.id}.stylize-failed.json"]
    assert not any(settings.work_dir.iterdir())
    # what the failure leaves next to the model when its work dir goes (D34): the error, the child's result and
    # the end of its output
    rec = json.loads((settings.data_dir / kept).read_text())
    assert (rec["job_id"], rec["attempt"], rec["model_id"], rec["error"]) == (styl.id, 1, model.id, why)
    assert rec["exit_status"] == (1 if mode == "fail" else None) and rec["style_summary"] is None
    assert rec["result"] == ({"ok": False, "error": "ValueError: no outline"} if mode == "fail" else None)
    assert rec["child_log_tail"] == f"fake child: stylizing {model.id}\n" and rec["child_log_bytes"] == len(rec["child_log_tail"])


def test_a_stylize_job_that_fails_its_qa_or_crashes_keeps_the_reason(store, settings, make_runner, catalog, fake_child,
                                                                    monkeypatch):
    """D34. A QA failure deleted the summary that said which gate failed on which part, with the work dir, and the
    job kept "failed its QA: structure"; a crash left "exited with status 1" and its traceback went the same way.
    Now the job's error names the part, and next to the model stay the summary (the whole QA) and a record with
    the child's result and the end of its output. A later stylization that succeeds removes both."""
    pytest.importorskip("pycolmap")
    from crowd3d import jobs as J
    from crowd3d.api import ModelFiles, job_info

    monkeypatch.setenv("FAKE_CHILD_MODE", "qa")
    runner = make_runner(SparseWriter(), inline_jobs=True, structure_catalog=catalog)
    add_photos(store, settings, 2, 4)
    runner.schedule(LM)
    (model,) = store.list_models(LM)
    (styl,) = _stylize_jobs(store)
    folder = settings.models_dir / LM
    record, summary = folder / f"{model.id}.stylize-failed.json", folder / f"{model.id}.stylize-failed.style.json"
    assert styl.status == JOB_FAILED and styl.error == (
        f"stylize failed, model {model.id} gets no stylized GLB: the stylized model failed its QA: structure "
        f"(structure: G13_not_built [w2: TypeError: boom]); diagnostics kept in models/{LM}/{record.name}")
    assert job_info(styl)["error"] == styl.error                     # nothing in it the public job list must hide
    rec = json.loads(record.read_text())
    assert rec["style_summary"] == summary.name and rec["result"]["qa_failed"] == ["structure"] and rec["exit_status"] == 1
    assert json.loads(summary.read_text())["qa"]["structure"]["gates"]["G13_not_built"][0]["id"] == "w2"
    assert not any(settings.work_dir.iterdir()) and not (folder / f"{model.id}.style.json").exists()
    assert not record.name.endswith(ModelFiles.SUFFIXES) and not summary.name.endswith(ModelFiles.SUFFIXES)   # not served
    # a crash: no result at all; the end of the child's output (its traceback) is what explains it
    monkeypatch.setenv("FAKE_CHILD_MODE", "crash")
    again = store.create_job(Job(id="again", landmark_id=LM, kind=JOB_STYLIZE, sector=2, status=JOB_QUEUED,
                                 photo_ids=styl.photo_ids, created_at=time.time(), model_id=model.id))
    done = runner.run_job(again)
    assert done.status == JOB_FAILED and done.error == (
        f"stylize failed, model {model.id} gets no stylized GLB: the stylize process exited with status 1; "
        f"diagnostics kept in models/{LM}/{record.name}")
    rec = json.loads(record.read_text())
    assert rec["job_id"] == "again" and rec["result"] is None and rec["style_summary"] is None and not summary.exists()
    assert rec["child_log_tail"].endswith("IndexError: index 8 is out of bounds\n")
    assert len(rec["child_log_tail"]) == J.CHILD_LOG_TAIL_BYTES and rec["child_log_bytes"] > 20000   # the end only
    # success: the GLB and its summary, and the failure's files are gone
    monkeypatch.setenv("FAKE_CHILD_MODE", "ok")
    third = store.create_job(Job(id="third", landmark_id=LM, kind=JOB_STYLIZE, sector=2, status=JOB_QUEUED,
                                 photo_ids=styl.photo_ids, created_at=time.time(), model_id=model.id))
    assert runner.run_job(third).status == JOB_DONE
    assert sorted(p.name for p in folder.iterdir()) == [f"{model.id}.glb", f"{model.id}.json", f"{model.id}.ply",
                                                        f"{model.id}.style.json"]


def test_work_dirs_are_kept_by_an_environment_variable(store, settings, make_runner, catalog, fake_child, monkeypatch):
    """D34: keep_work_dirs had no switch outside code. $CROWD3D_KEEP_WORK_DIRS is its default; with it a failed
    stylize job's work dir (spec, child.log, out) stays, and the diagnostics next to the model are written all the
    same."""
    pytest.importorskip("pycolmap")
    assert Settings(data_dir=settings.data_dir).keep_work_dirs is False
    for value, on in (("1", True), ("true", True), ("YES", True), ("on", True), ("0", False), ("", False), ("no", False)):
        monkeypatch.setenv("CROWD3D_KEEP_WORK_DIRS", value)
        assert Settings(data_dir=settings.data_dir).keep_work_dirs is on, value
    assert Settings(data_dir=settings.data_dir, keep_work_dirs=True).keep_work_dirs is True
    monkeypatch.setenv("CROWD3D_KEEP_WORK_DIRS", "1")
    monkeypatch.setenv("FAKE_CHILD_MODE", "fail")
    kept = Settings(data_dir=settings.data_dir, reconstructor="fake")
    runner = make_runner(SparseWriter(), inline_jobs=True, structure_catalog=catalog, keep_work_dirs=kept.keep_work_dirs)
    add_photos(store, settings, 2, 4)
    runner.schedule(LM)
    (model,) = store.list_models(LM)
    (styl,) = _stylize_jobs(store)
    work = settings.work_dir / f"{styl.id}.1"
    assert styl.status == JOB_FAILED and sorted(p.name for p in work.iterdir()) == ["child.log", "out", "spec.json"]
    assert (work / "child.log").read_text() == f"fake child: stylizing {model.id}\n"
    assert (settings.models_dir / LM / f"{model.id}.stylize-failed.json").is_file()


def test_a_restart_runs_leftover_stylize_jobs_after_the_reconstructions(store, settings, make_runner, catalog,
                                                                         fake_child):
    pytest.importorskip("pycolmap")
    runner = make_runner(SparseWriter(), inline_jobs=True, structure_catalog=catalog, stylize_published=False)
    add_photos(store, settings, 2, 4)
    (job,) = runner.schedule(LM)
    (model,) = store.list_models(LM)
    store.create_job(Job(id="styl", landmark_id=LM, kind=JOB_STYLIZE, sector=2, status=JOB_RUNNING,
                         photo_ids=job.photo_ids, created_at=1.0, started_at=1.5, attempts=1, model_id=model.id))
    recon = _job(store, add_photos(store, settings, 6, 4), sector=6)       # created after the stylize job
    make_runner(SparseWriter(), inline_jobs=True, structure_catalog=catalog, stylize_published=False).start()
    done = {j.id: j for j in store.list_jobs(LM)}
    assert done["styl"].status == done[recon.id].status == JOB_DONE and done["styl"].attempts == 2
    assert done[recon.id].finished_at <= done["styl"].started_at
    assert store.get_model(model.id).stylized["job_id"] == "styl"


def test_a_localize_job_is_background_work_and_does_nothing_while_localization_is_off(store, settings, make_runner):
    """JOB_LOCALIZE (D30) waits for the reconstructions like a stylize job and never reconstructs; with
    Settings.localize_photos off (the default) a leftover one ends at once, localizes nothing and plans nothing."""
    from crowd3d import jobs as jobs_module
    from crowd3d.records import BACKGROUND_KINDS, RECONSTRUCTION_KINDS

    assert BACKGROUND_KINDS == (JOB_STYLIZE, JOB_LOCALIZE) and RECONSTRUCTION_KINDS == (JOB_FACE, JOB_FULL)
    assert jobs_module._PRIORITY == {JOB_STYLIZE: 1, JOB_LOCALIZE: 1}
    assert Settings().localize_photos is False and Settings().localize_batch == 50
    photos = add_photos(store, settings, 2, 4, focal_35mm=26.0, location_consent=TERMS_V1)
    loc = store.create_job(Job(id="loc", landmark_id=LM, kind=JOB_LOCALIZE, sector=None, status=JOB_QUEUED,
                               photo_ids=[photos[0].id], created_at=1.0))
    fake = FakeLocalizer()
    runner = make_runner(RaisingOnce(), inline_jobs=True, localizer=fake)
    runner.reconstructor.calls = 1                       # no failure: count the calls only
    assert not runner._reconstruction_queued()           # a queued localize job is no reconstruction
    recon = _job(store, photos, sector=2)                # created after the localize job
    assert runner._reconstruction_queued()
    runner.start()
    done = {j.id: j for j in store.list_jobs(LM)}
    assert done[recon.id].status == JOB_DONE and runner.reconstructor.calls == 2      # one reconstruction only
    assert (done["loc"].status, done["loc"].error) == (JOB_DONE, "visual localization is off: nothing localized")
    assert done["loc"].started_at >= done[recon.id].finished_at
    assert len(store.list_models(LM)) == 1 and set(done) == {"loc", recon.id}         # and it planned nothing
    assert fake.calls == [] and store.localizations(LM) == {}


# ---------------------------------------------------------------- visual localization jobs (D30)


class FakeLocalizer:
    """Stands in for localize.ColmapLocalizer: every candidate map loads; each query is localized into the first map
    (1 m east of its GPS fix, its compass heading) unless its id is in reject (too_few_matches). before(query) runs
    before each query's stop check; raise_error makes localize raise. Records what each call got."""

    def __init__(self, reject=(), before=None, raise_error=False, drop=()):
        self.reject, self.before, self.raise_error, self.drop = set(reject), before, raise_error, set(drop)
        self.calls: list[dict] = []

    def load_maps(self, root, maps, cameras):
        return list(maps), {}

    def localize(self, queries, maps, photos, lm, work_dir, *, cache=None, stop=None, deadline=None):
        from crowd3d import localize as loc

        self.calls.append({"queries": [q.id for q in queries], "maps": [m.model_id for m in maps], "cache": cache,
                           "photos": len(photos), "stop": stop})
        if self.raise_error:
            raise RuntimeError("boom")
        work_dir.mkdir(parents=True, exist_ok=True)
        (work_dir / "database.db").write_bytes(b"x")
        run = loc.LocalizeRun()
        for k, q in enumerate(queries):
            if self.before is not None:
                self.before(q)
            if stop is not None and stop():
                run.unfinished = [x.id for x in queries[k:]]
                break
            if q.id in self.drop:                          # a broken localizer: says nothing about it
                continue
            query = loc.make_query(q, 640, 480, lm.lat, lm.lon)
            m = maps[0]
            if q.id in self.reject:
                stats = loc.PoseStats(m.model_id, m.kind, 3, 640, query.focal_px, reason="too_few_matches")
                run.outcomes[q.id] = loc.Outcome(q, query, None, "too_few_matches", [stats], {"pairs": 4})
                continue
            e, n = query.gps_en
            stats = loc.PoseStats(m.model_id, m.kind, 200, 640, query.focal_px, focal_px=query.focal_px,
                                  focal_free_px=query.focal_px, inliers=190, reproj_px=0.5, enu=[e + 1.0, n, 1.5],
                                  heading_deg=q.heading_deg, pitch_deg=2.0, roll_deg=0.5, look_deg=3.0)
            run.outcomes[q.id] = loc.Outcome(q, query, stats, None, [stats], {"pairs": 4})
        return run


SHARED = {"focal_35mm": 26.0, "location_consent": TERMS_V1}


def _localize_jobs(store, landmark_id=LM):
    return [j for j in store.list_jobs(landmark_id) if j.kind == JOB_LOCALIZE]


def _published_ids(store):
    return sorted(m.id for m in store.published_models(LM))


def test_consented_photos_no_model_shows_are_localized_by_a_job_after_the_reconstructions(
        store, settings, make_runner, caplog):
    """D30, the job path: a localize job after the reconstruction, for the consented photos that no published model
    shows and that have an EXIF focal length (one with none is recorded as skipped, a photo without consent and the
    photos the model shows get no record), every record keyed by the candidate maps; the work dir removed; counts
    logged, never a photo id. Planning again changes nothing; a newly published map tries them again."""
    pytest.importorskip("pycolmap")
    from crowd3d import localize as loc
    from crowd3d.landmark_map import FeatureCache
    from crowd3d.records import LOC_ACCEPTED, LOC_REJECTED, LOC_SKIPPED

    caplog.set_level("INFO")
    fake = FakeLocalizer()
    runner = make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=fake)
    shown = add_photos(store, settings, 2, 4, **SHARED)                 # in the facade job: shown as SfM cameras
    q1, q2 = add_photos(store, settings, 6, 2, **SHARED)                # too few for a job of their own
    (nofocal,) = add_photos(store, settings, 4, 1, location_consent=TERMS_V1)
    (private,) = add_photos(store, settings, 4, 1, focal_35mm=26.0)     # no consent
    fake.reject = {q2.id}
    (face,) = runner.schedule(LM)                                       # the localize job is never returned
    (job,) = _localize_jobs(store)
    face = store.get_job(face.id)
    assert (face.kind, face.status, job.status, job.error, job.sector) == (JOB_FACE, JOB_DONE, JOB_DONE, None, None)
    assert job.photo_ids == [q1.id, q2.id] and job.started_at >= face.finished_at
    (model,) = store.published_models(LM)
    (call,) = fake.calls
    assert (call["queries"], call["maps"]) == ([q1.id, q2.id], [model.id])
    assert isinstance(call["cache"], FeatureCache) and call["photos"] == 8 and call["stop"]() is False
    key = loc.maps_key([model.id])
    recs = store.localizations(LM)
    assert set(recs) == {q1.id, q2.id, nofocal.id}
    assert {pid: (r.status, r.reason, r.maps_key, r.attempts) for pid, r in recs.items()} == {
        q1.id: (LOC_ACCEPTED, None, key, 1), q2.id: (LOC_REJECTED, "too_few_matches", key, 1),
        nofocal.id: (LOC_SKIPPED, "no_focal", key, 0)}
    assert recs[q1.id].model_id == model.id and recs[q1.id].pose["heading_deg"] == round(q1.heading_deg, 2)
    assert not any(settings.work_dir.iterdir())
    text = "\n".join(r.getMessage() for r in caplog.records)
    assert f"job {job.id} (localize, landmark {LM}): 2 photos to try, 2 tried (1 localized, 1 rejected)" in text
    for p in [*shown, q1, q2, nofocal, private]:
        assert p.id not in text
    # nothing new to try: no new job, planning again or at a restart
    runner.schedule(LM)
    make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=fake).start()
    assert len(_localize_jobs(store)) == 1 and len(fake.calls) == 1
    # another published model changes the candidate maps: each consented photo without an SfM camera is tried again
    add_photos(store, settings, 0, 4, **SHARED)
    runner.schedule(LM)
    assert len(_localize_jobs(store)) == 2 and fake.calls[1]["queries"] == [q1.id, q2.id]
    assert sorted(fake.calls[1]["maps"]) == _published_ids(store)
    key2 = loc.maps_key(_published_ids(store))
    recs = store.localizations(LM)
    assert {pid: (r.maps_key, r.attempts) for pid, r in recs.items()} == {
        q1.id: (key2, 2), q2.id: (key2, 2), nofocal.id: (key2, 0)}


def test_localization_is_off_by_default_and_without_a_localizer(store, settings, make_runner):
    fake = FakeLocalizer()
    add_photos(store, settings, 2, 4, **SHARED)
    add_photos(store, settings, 6, 2, **SHARED)
    pytest.importorskip("pycolmap")
    make_runner(SparseWriter(), inline_jobs=True, localizer=fake).schedule(LM)              # the default: off
    assert _localize_jobs(store) == [] and fake.calls == []
    add_photos(store, settings, 0, 4, **SHARED)
    runner = make_runner(FakeReconstructor(), inline_jobs=True, localize_photos=True)        # the fake one: no maps
    assert runner._localizer() is None
    runner.schedule(LM)
    assert _localize_jobs(store) == [] and store.localizations(LM) == {}


def test_a_localize_job_stops_for_a_queued_reconstruction_and_a_follow_up_finishes_after_it(
        store, settings, make_runner):
    """Order item 11: no time budget; the job stops between two photos as soon as a reconstruction job is queued,
    stores nothing for the photos it left and queues a follow-up job behind the reconstruction."""
    pytest.importorskip("pycolmap")
    queued = []

    def before(q):
        if q.id == q2.id and not queued:                 # a reconstruction job arrives while q1 was localized
            # (of one photo: it fails, so the maps stay the same)
            queued.append(_job(store, add_photos(store, settings, 4, 1), sector=4))

    fake = FakeLocalizer(before=before)
    runner = make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=fake)
    add_photos(store, settings, 2, 4, **SHARED)
    q1, q2 = add_photos(store, settings, 6, 2, **SHARED)
    runner.schedule(LM)
    first, follow = _localize_jobs(store)
    assert (first.status, first.error) == (JOB_DONE, ("1 photos left for the next localize job (a reconstruction"
                                                      " job goes first)"))
    assert set(store.localizations(LM)) == {q1.id} and follow.status == JOB_QUEUED     # held behind it
    (recon,) = queued
    assert runner.run_job(recon).status == JOB_FAILED
    runner._run_deferred()
    follow = store.get_job(follow.id)
    assert follow.status == JOB_DONE and follow.started_at >= store.get_job(recon.id).finished_at
    assert fake.calls[1]["queries"] == [q2.id] and set(store.localizations(LM)) == {q1.id, q2.id}


def test_in_the_background_a_localize_job_yields_to_a_reconstruction_job_queued_meanwhile(
        store, settings, make_runner):
    pytest.importorskip("pycolmap")
    add_photos(store, settings, 2, 4, **SHARED)
    make_runner(SparseWriter(), inline_jobs=True).schedule(LM)        # a published facade (localization off)
    q1, q2 = add_photos(store, settings, 6, 2, **SHARED)
    holder = {}

    def before(q):
        if q.id == q2.id and "recon" not in holder:      # an upload plans a facade job while the worker localizes
            add_photos(store, settings, 4, 4, **SHARED)
            (holder["recon"],) = holder["runner"].schedule(LM)

    runner = holder["runner"] = make_runner(SparseWriter(), localize_photos=True, localizer=FakeLocalizer(before=before))
    runner.start()                                       # the start plans every landmark: a localize job
    assert runner.drain(30)
    first, follow = _localize_jobs(store)
    recon = store.get_job(holder["recon"].id)
    assert first.status == follow.status == recon.status == JOB_DONE
    assert first.finished_at <= recon.started_at and recon.finished_at <= follow.started_at
    assert first.photo_ids == [q1.id, q2.id] and set(store.localizations(LM)) == {q1.id, q2.id}


def test_a_map_the_localizer_refuses_never_makes_a_second_localize_job(store, settings, make_runner):
    """The records stay keyed by the candidate maps whatever load_sparse_maps refused (the C7 review): the saved map
    here (the golden fixture's images) does not give back this model's cameras, so it is refused, the query is
    recorded as no_maps under the candidates' key, and planning again or a restart queues no other job."""
    pytest.importorskip("pycolmap")
    from crowd3d import localize as loc

    s = Settings(**{**settings.__dict__, "localize_photos": True})
    runner = make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=loc.ColmapLocalizer(s))
    add_photos(store, settings, 2, 4, **SHARED)
    (q1,) = add_photos(store, settings, 6, 1, **SHARED)
    runner.schedule(LM)
    (job,) = _localize_jobs(store)
    (model,) = store.published_models(LM)
    rec = store.get_localization(q1.id)
    assert job.status == JOB_DONE and (rec.status, rec.reason) == ("skipped", "no_maps")
    assert rec.maps_key == loc.maps_key([model.id]) != loc.maps_key([])
    runner.schedule(LM)
    make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=loc.ColmapLocalizer(s)).start()
    assert len(_localize_jobs(store)) == 1


def test_a_failed_localize_job_is_not_retried_until_a_reconstruction_or_stylize_job_ends(store, settings, make_runner):
    from crowd3d.jobs import localize_held

    pytest.importorskip("pycolmap")
    fake = FakeLocalizer(raise_error=True)
    runner = make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=fake)
    add_photos(store, settings, 2, 4, **SHARED)
    add_photos(store, settings, 6, 2, **SHARED)
    runner.schedule(LM)
    (failed,) = _localize_jobs(store)
    assert (failed.status, failed.error) == (JOB_FAILED, "RuntimeError: boom") and store.localizations(LM) == {}
    assert not any(settings.work_dir.iterdir())
    runner.schedule(LM)
    make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=fake).start()
    assert len(_localize_jobs(store)) == 1 and len(fake.calls) == 1
    fake.raise_error = False
    add_photos(store, settings, 0, 4, **SHARED)
    runner.schedule(LM)                                  # a reconstruction job ends: tried again
    assert [j.status for j in _localize_jobs(store)] == [JOB_FAILED, JOB_DONE] and len(store.localizations(LM)) == 2

    # a localizer that says nothing about a photo fails the job (each follow-up would try it again at once)
    fake.drop = {store.list_photos(LM)[4].id}
    add_photos(store, settings, 4, 4, **SHARED)
    runner.schedule(LM)
    last = _localize_jobs(store)[-1]
    assert (last.status, last.error) == (JOB_FAILED, "localize failed: the localizer said nothing about 1 photos")
    assert len(_localize_jobs(store)) == 3

    def job(kind, status, finished):
        return Job(id=f"{kind}{finished}", landmark_id=LM, kind=kind, sector=None, status=status, photo_ids=[],
                   created_at=0.0, finished_at=finished)

    assert not localize_held([])
    assert localize_held([job(JOB_LOCALIZE, JOB_FAILED, 2.0), job(JOB_FACE, JOB_DONE, 1.0)])
    assert not localize_held([job(JOB_LOCALIZE, JOB_FAILED, 2.0), job(JOB_STYLIZE, JOB_FAILED, 3.0)])
    assert not localize_held([job(JOB_LOCALIZE, JOB_FAILED, 2.0), job(JOB_LOCALIZE, JOB_DONE, 3.0)])
    assert localize_held([job(JOB_LOCALIZE, JOB_DONE, 2.0), job(JOB_LOCALIZE, JOB_FAILED, 3.0)])


def test_a_localize_job_tries_a_batch_and_leaves_the_rest_to_the_next_one(store, settings, make_runner):
    pytest.importorskip("pycolmap")
    fake = FakeLocalizer()
    runner = make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localize_batch=1, localizer=fake)
    add_photos(store, settings, 2, 4, **SHARED)
    q1, q2 = add_photos(store, settings, 6, 2, **SHARED)
    runner.schedule(LM)
    first, second = _localize_jobs(store)
    assert (first.photo_ids, first.error) == ([q1.id], "1 photos left for the next localize job")
    assert (second.photo_ids, second.status, second.error) == ([q2.id], JOB_DONE, None)
    assert [c["queries"] for c in fake.calls] == [[q1.id], [q2.id]] and len(store.localizations(LM)) == 2


def test_a_consent_withdrawn_during_a_localize_job_leaves_no_record(store, settings, make_runner):
    pytest.importorskip("pycolmap")
    fake = FakeLocalizer(before=lambda q: store.withdraw_location_consent(q.id) if q.id == q2.id else None)
    runner = make_runner(SparseWriter(), inline_jobs=True, localize_photos=True, localizer=fake)
    add_photos(store, settings, 2, 4, **SHARED)
    q1, q2 = add_photos(store, settings, 6, 2, **SHARED)
    runner.schedule(LM)
    (job,) = _localize_jobs(store)
    assert job.status == JOB_DONE and set(store.localizations(LM)) == {q1.id}
    runner.schedule(LM)                                  # q2 is no longer shared: nothing to try
    assert len(_localize_jobs(store)) == 1


class PriorOnlyMap(SparseWriter):
    """A saved map whose frame is not fixed: its anchor level is prior_only (yaw from the landmark direction, scale
    guessed), so it is a candidate map only once a footprint snap of its stylized GLB was accepted."""

    name = "prior-only"

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, **_):
        result = super().reconstruct(photos, landmark_lat, landmark_lon, out_ply, work_dir)
        result.anchor = {**result.anchor, "level": "prior_only"}
        result.georegistered = False
        return result


def test_a_prior_only_map_is_localized_into_only_once_its_stylized_glbs_snap_was_accepted(
        store, settings, make_runner):
    """The localize jobs take the snaps the public poses use (jobs.stylized_snaps: the stylize record's accepted
    snap): a refused one fixes no frame, so nothing is tried in that map; an accepted one makes it a candidate, and
    the photos recorded as no_maps are tried in it."""
    pytest.importorskip("pycolmap")
    from crowd3d import localize as loc

    fake = FakeLocalizer()
    runner = make_runner(PriorOnlyMap(), inline_jobs=True, localize_photos=True, localizer=fake)
    add_photos(store, settings, 2, 4, **SHARED)
    (q1,) = add_photos(store, settings, 6, 1, **SHARED)
    runner.schedule(LM)
    (model,) = store.published_models(LM)
    assert store.get_localization(q1.id).reason == "no_maps" and fake.calls == []
    snap = {"rotation_deg": 1.0, "shift_m": [0.5, 0.0], "scale": 1.0, "reason": ""}
    store.set_model_stylized(model.id, {"glb_path": "x.glb", "snap": {**snap, "accepted": False}})
    runner.schedule(LM)
    assert len(_localize_jobs(store)) == 1 and fake.calls == []
    store.set_model_stylized(model.id, {"glb_path": "x.glb", "snap": {**snap, "accepted": True}})
    runner.schedule(LM)
    assert len(_localize_jobs(store)) == 2 and fake.calls[0]["maps"] == [model.id]
    rec = store.get_localization(q1.id)
    assert (rec.status, rec.model_id, rec.maps_key) == ("accepted", model.id, loc.maps_key([model.id]))
