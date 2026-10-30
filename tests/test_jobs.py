import hashlib
import json
import math
import shutil
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
