"""Turn coverage changes into reconstruction jobs and run them in the background."""

from __future__ import annotations

import json
import logging
import queue
import shutil
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

from . import params
from .config import Settings
from .coverage import JobRequest, compute_coverage, plan_jobs
from .landmark_map import CACHE_FILE, FeatureCache
from .qa import assess
from .reconstruct import ReconPhoto, Reconstructor
from .records import (
    ACTIVE_JOB_STATUSES,
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_FULL,
    JOB_QUEUED,
    JOB_RUNNING,
    PHOTO_ACCEPTED,
    Job,
    ModelRecord,
    PhotoRecord,
)
from .store import Store, to_json

log = logging.getLogger(__name__)


class JobRunner:
    """Single background worker thread (SfM is CPU-heavy; one at a time).

    With settings.inline_jobs the job runs synchronously inside submit_manual()/schedule().

    After each background job the worker plans the landmark again, so photos that
    arrived while a job for the same target was queued/running (which blocks a new
    one) still trigger a rebuild once they reach the growth threshold.

    On start(), jobs a previous process left queued or running are re-queued
    (running ones go back to queued and start over, unless the process already
    died settings.max_job_attempts times during them: those fail), and every
    landmark is planned again, so photos accepted by a request that died before
    planning still get their jobs.

    A model is published only if it passes qa.assess; otherwise it is stored
    with meta["published"] = False, the job fails with the reasons, and the
    previous model of that target stays the one shown.
    """

    def __init__(self, store: Store, settings: Settings, reconstructor: Reconstructor) -> None:
        self.store = store
        self.settings = settings
        self.reconstructor = reconstructor
        self._queue: queue.Queue[str | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stopping = threading.Event()   # replaced on every start()
        # Job ids handed to the worker and not finished yet; guarded by _cv.
        self._pending: set[str] = set()
        self._cv = threading.Condition()
        self._landmark_locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    def start(self) -> None:
        """Start the worker thread and re-queue jobs left queued/running by a previous process."""
        if self._thread is not None and self._thread.is_alive():
            return
        for job in self.store.list_jobs(statuses=(JOB_RUNNING,)):
            if job.attempts >= self.settings.max_job_attempts:
                # Exceptions are caught in run_job; only a dead process (OOM kill, native crash) leaves
                # a job running, so running it again would most likely take the server down again.
                self._finish(job, JOB_FAILED, f"interrupted {job.attempts} times: the process died while"
                                              " reconstructing these photos")
            else:
                self.store.update_job(job.id, status=JOB_QUEUED, started_at=None)
        leftover = self.store.list_jobs(statuses=(JOB_QUEUED,))
        if self.settings.inline_jobs:
            for job in leftover:
                self.run_job(job)
        else:
            for job in leftover:
                self._enqueue(job.id)
            # A fresh event, so a worker from an earlier stop() that timed out still exits.
            self._stopping = threading.Event()
            self._thread = threading.Thread(
                target=self._work, args=(self._stopping,), name="crowd3d-jobs", daemon=True
            )
            self._thread.start()
        for lm in self.store.list_landmarks():
            try:
                self.schedule(lm.id)
            except Exception:
                log.exception("planning landmark %s on start failed", lm.id)

    def stop(self, timeout: float = 10.0) -> None:
        """Stop after the current job; queued jobs stay queued in the store for the next start()."""
        if self._thread is None:
            return
        self._stopping.set()
        self._queue.put(None)
        self._thread.join(timeout)
        self._thread = None

    def schedule(self, landmark_id: str) -> list[Job]:
        """compute plan_jobs for the landmark, persist the new jobs, submit them; return them."""
        with self._landmark_lock(landmark_id):
            photos = self.store.list_photos(landmark_id, PHOTO_ACCEPTED)
            requests = plan_jobs(photos, self.store.list_jobs(landmark_id), self.settings)
            jobs = [self._create(landmark_id, r) for r in requests]
        return [self._submit(j) for j in jobs]

    def submit_manual(self, landmark_id: str, kind: str, sector: int | None = None) -> Job:
        """Force a job regardless of growth thresholds (still needs the minimum photo counts).

        Raises ValueError when the minimum is not met. An active job with exactly the
        same inputs is returned instead of creating a duplicate.
        """
        if kind == JOB_FULL:
            sector = None
        elif kind != JOB_FACE:
            raise ValueError(f"unknown job kind {kind!r}")
        elif sector is None or not 0 <= sector < self.settings.num_sectors:
            raise ValueError(f"face jobs need a sector in 0..{self.settings.num_sectors - 1}")
        self.store.get_landmark(landmark_id)
        with self._landmark_lock(landmark_id):
            photos = self.store.list_photos(landmark_id, PHOTO_ACCEPTED)
            # With no previous jobs, plan_jobs proposes every target that meets its minimum.
            candidates = plan_jobs(photos, [], self.settings)
            request = next((r for r in candidates if (r.kind, r.sector) == (kind, sector)), None)
            if request is None:
                raise ValueError(self._minimum_message(photos, kind, sector))
            for job in self.store.list_jobs(landmark_id, ACTIVE_JOB_STATUSES):
                if (job.kind, job.sector, job.photo_ids) == (kind, sector, request.photo_ids):
                    return job
            job = self._create(landmark_id, request)
        return self._submit(job)

    def run_job(self, job: Job) -> Job:
        """Execute one job: mark running, reconstruct into
        <models_dir>/<landmark_id>/<model_id>.ply, add a ModelRecord, mark done.
        Any exception or result.ok == False marks the job failed with an error message.
        """
        job = self.store.update_job(
            job.id, status=JOB_RUNNING, started_at=time.time(), finished_at=None, error=None,
            attempts=job.attempts + 1,
        )
        try:
            return self._reconstruct(job)
        except Exception as e:
            log.exception("job %s failed", job.id)
            return self._finish(job, JOB_FAILED, f"{type(e).__name__}: {e}")

    def drain(self, timeout: float = 600.0) -> bool:
        """Block until the queue is empty and the worker is idle. True if drained."""
        deadline = time.monotonic() + timeout
        with self._cv:
            while self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._cv.wait(remaining)
        return True

    def _reconstruct(self, job: Job) -> Job:
        s = self.settings
        lm = self.store.get_landmark(job.landmark_id)
        records, photos = [], []
        for pid in job.photo_ids:
            p = self.store.get_photo(pid)
            if p.processed_path is None or p.lat is None or p.lon is None:
                raise ValueError(f"photo {pid} has no processed copy or GPS position")
            records.append(p)
            photos.append(ReconPhoto(p.id, s.data_dir / p.processed_path, p.lat, p.lon, p.heading_deg,
                                     gps_accuracy_m=p.gps_accuracy_m, focal_35mm=p.focal_35mm))
        model_id = uuid.uuid4().hex
        out_ply = s.models_dir / lm.id / f"{model_id}.ply"
        out_ply.parent.mkdir(parents=True, exist_ok=True)
        work_dir = s.work_dir / job.id
        cache = self._feature_cache(lm.id)
        try:
            result = self.reconstructor.reconstruct(photos, lm.lat, lm.lon, out_ply=out_ply, work_dir=work_dir,
                                                    **({"cache": cache} if cache is not None else {}))
        finally:
            if not s.keep_work_dirs:
                # The job's COLMAP database takes ~1.5 MB per photo; with Params.feature_cache the landmark's
                # FeatureCache keeps its features and matches for the next job.
                shutil.rmtree(work_dir, ignore_errors=True)
        if not result.ok or not out_ply.is_file():
            error = result.message if not result.ok else "reconstructor reported success but wrote no PLY"
            log.warning("job %s (%s, sector %s) failed: %s", job.id, job.kind, job.sector, error)
            return self._finish(job, JOB_FAILED, error or "reconstruction failed")
        report = assess(result, job.kind, records, lm, s)
        ply_rel = out_ply.relative_to(s.data_dir).as_posix()
        meta_text = to_json({**result.to_meta(), "ply_path": ply_rel, "published": report.passed,
                             "qa": report.to_meta()}, indent=2)
        out_ply.with_suffix(".json").write_text(meta_text, encoding="utf-8")
        self.store.add_model(
            ModelRecord(
                id=model_id,
                landmark_id=lm.id,
                job_id=job.id,
                kind=job.kind,
                sector=job.sector,
                ply_path=ply_rel,
                num_photos=len(job.photo_ids),
                meta=json.loads(meta_text),
                created_at=time.time(),
            )
        )
        if not report.passed:
            kept = any((m.kind, m.sector) == (job.kind, job.sector) for m in self.store.published_models(lm.id))
            outcome = "the previous model stays published" if kept else "nothing is published for it yet"
            error = f"quality check failed, {outcome}: " + "; ".join(report.reasons)
            log.warning("job %s (%s, sector %s): %s", job.id, job.kind, job.sector, error)
            return self._finish(job, JOB_FAILED, error)
        return self._finish(job, JOB_DONE, None)

    def _feature_cache(self, landmark_id: str) -> FeatureCache | None:
        """The landmark's feature and match cache, when Params.feature_cache is on and the reconstructor uses one."""
        if params.current().feature_cache and getattr(self.reconstructor, "feature_cache", False):
            return FeatureCache(self.settings.maps_dir / landmark_id / CACHE_FILE)
        return None

    def _finish(self, job: Job, status: str, error: str | None) -> Job:
        return self.store.update_job(job.id, status=status, finished_at=time.time(), error=error)

    def _create(self, landmark_id: str, request: JobRequest) -> Job:
        return self.store.create_job(
            Job(
                id=uuid.uuid4().hex,
                landmark_id=landmark_id,
                kind=request.kind,
                sector=request.sector,
                status=JOB_QUEUED,
                photo_ids=request.photo_ids,
                created_at=time.time(),
            )
        )

    def _submit(self, job: Job) -> Job:
        if self.settings.inline_jobs:
            return self.run_job(job)
        self._enqueue(job.id)
        return job

    def _enqueue(self, job_id: str) -> None:
        with self._cv:
            if job_id in self._pending:
                return
            self._pending.add(job_id)
        self._queue.put(job_id)

    def _work(self, stopping: threading.Event) -> None:
        while not stopping.is_set():
            job_id = self._queue.get()
            if job_id is None:
                continue
            try:
                job = self.store.get_job(job_id)
                # An id left in the queue across stop()/start() may have run meanwhile.
                if job.status == JOB_QUEUED:
                    self.run_job(job)
                    self.schedule(job.landmark_id)
            except Exception:
                log.exception("job worker error on %s", job_id)
            finally:
                with self._cv:
                    self._pending.discard(job_id)
                    self._cv.notify_all()

    @contextmanager
    def _landmark_lock(self, landmark_id: str) -> Iterator[None]:
        with self._locks_guard:
            lock = self._landmark_locks.setdefault(landmark_id, threading.Lock())
        with lock:
            yield

    def _minimum_message(self, photos: Sequence[PhotoRecord], kind: str, sector: int | None) -> str:
        cov = compute_coverage(photos, self.settings)
        if kind == JOB_FACE:
            return (
                f"sector {sector} has {cov.sectors[sector].count} accepted photos, "
                f"needs {self.settings.face_min_photos}"
            )
        return (
            f"full model needs {self.settings.full_min_photos} accepted photos and "
            f"{cov.full_min_sectors} face-ready sectors; have {cov.total} and {len(cov.face_ready)}"
        )
