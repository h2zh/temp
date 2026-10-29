import json
import math
import threading
import time

import pytest

from crowd3d.config import Settings
from crowd3d.coverage import sector_center_deg, sector_for_bearing
from crowd3d.geo import LocalENU, normalize_deg
from crowd3d.jobs import JobRunner
from crowd3d.reconstruct import FakeReconstructor, ReconstructionResult
from crowd3d.records import (
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_FULL,
    JOB_QUEUED,
    JOB_RUNNING,
    PHOTO_ACCEPTED,
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
        s = Settings(**{**settings.__dict__, **overrides})
        r = JobRunner(store, s, reconstructor or FakeReconstructor(s))
        runners.append(r)
        return r

    yield make
    for r in runners:
        r.stop()


_seq = iter(range(10**6))


def add_photos(store, settings, sector, count, landmark_id=LM):
    """Accepted photos 30 m out in the middle of a sector, with fake processed copies on disk."""
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
            processed_path=rel,
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
    runner.schedule(LM)
    assert user.caches == [None]                                   # Params.feature_cache off
    with params.override(feature_cache=True):
        add_photos(store, settings, 2, 3)
        runner.schedule(LM)
        assert user.caches[-1].path == settings.maps_dir / LM / CACHE_FILE
        fake = make_runner(inline_jobs=True)                       # FakeReconstructor: no cache, no maps dir
        add_photos(store, settings, 6, 4)
        (job,) = fake.schedule(LM)
        assert job.status == JOB_DONE and not settings.maps_dir.exists()


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
