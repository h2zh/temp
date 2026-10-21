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
