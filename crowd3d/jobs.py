"""Turn coverage changes into reconstruction jobs and run them in the background."""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
import math
import os
import queue
import shutil
import statistics
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import params
from .config import Settings
from .coverage import JobRequest, compute_coverage, plan_jobs
from .geo import angle_diff_deg, haversine_m
from .landmark_map import (
    CACHE_FILE,
    MODELS_DIR,
    FeatureCache,
    MapBase,
    job_feature_version,
    load_map,
    prune_maps,
    save_map,
)
from .qa import QAReport, assess
from .reconstruct import ReconPhoto, ReconstructionResult, Reconstructor
from .records import (
    ACTIVE_JOB_STATUSES,
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_FULL,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_STYLIZE,
    PHOTO_ACCEPTED,
    Job,
    Landmark,
    ModelRecord,
    PhotoRecord,
)
from .store import Store, to_json

log = logging.getLogger(__name__)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent   # the stylize child runs `python -m crowd3d...` from here
STRUCTURE_CACHE = "structure-cache"     # Settings.maps_dir / <landmark id> / STRUCTURE_CACHE: sky masks, line segments
STYLIZE_THREADS = 2
_PRIORITY = {JOB_STYLIZE: 1}            # the worker's order: reconstruction jobs (0) before stylize jobs


def structure_slug(settings: Settings, lm: Landmark) -> str | None:
    """The catalogue slug whose OSM parts snapshot the landmark's stylized models are built on: the landmark's
    Settings.structure_bindings entry, else its id, when Settings.structure_catalog lists that slug with a snapshot
    and an outline and the slug's centre lies within the landmark's radius (a slug naming a building elsewhere binds
    nothing). None: the landmark gets no stylized models."""
    if not settings.structure_catalog:
        return None
    from .structure_pipeline import catalog_entry

    slug = settings.structure_bindings.get(lm.id, lm.id)
    try:
        entry = catalog_entry(Path(settings.structure_catalog), slug)
    except (OSError, ValueError, KeyError) as e:
        log.warning("structure catalogue %s is unreadable: %s", settings.structure_catalog, e)
        return None
    if not entry or not entry.get("osm_parts") or not (entry.get("osm") or {}).get("file"):
        return None
    try:
        far = haversine_m(lm.lat, lm.lon, float(entry["lat"]), float(entry["lon"])) > lm.radius_m
    except (KeyError, TypeError, ValueError):
        return None
    return None if far else slug


def model_inputs(photos: Sequence[PhotoRecord]) -> dict[str, Any]:
    """A model's versioned inputs (meta["inputs"]): how many photos, the newest upload among them, and the sha256 of
    their sorted ids."""
    ids = sorted(p.id for p in photos)
    return {"count": len(ids), "max_uploaded_at": max((p.uploaded_at for p in photos), default=None),
            "photo_set_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def superseded_by(store: Store, landmark_id: str, kind: str, sector: int | None,
                  photo_ids: Sequence[str]) -> ModelRecord | None:
    """The supersede rule, the one place it is decided: the published model the landmark page shows for the target
    (kind, sector), when it is newer than a model of these photos would be, i.e. it has photos they lack, uploaded
    after every one of theirs (the benchmark's stale_overwrites counts the replacements this rule refuses). Whatever
    holds these older inputs (a job that finishes late, a later step on an older model) must not replace it."""
    current = {(m.kind, m.sector): m for m in store.published_models(landmark_id)}.get((kind, sector))
    if current is None:
        return None
    theirs, mine = set(store.get_job(current.job_id).photo_ids), set(photo_ids)
    if not theirs - mine:
        return None
    uploaded = {p.id: p.uploaded_at for p in store.list_photos(landmark_id)}

    def newest(ids: set[str]) -> float:
        return max((uploaded.get(i, 0.0) for i in ids), default=0.0)

    return current if newest(theirs) > newest(mine) else None


def continuity(parent: dict[str, dict[str, Any]], cameras: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """How far a new version of a target moved against the version it continued (meta["map"]["continuity"],
    recorded only; ARCHITECTURE 3.6 reviews a version that moves more than 2 m or 3 deg): over the photos both show
    (unflagged in both, as the benchmark's version drift), the median ENU distance of their camera centres and the
    median change of their headings."""
    common = [p for p in sorted(parent.keys() & cameras.keys())
              if not parent[p].get("flags") and not cameras[p].get("flags")
              and parent[p].get("enu") is not None and cameras[p].get("enu") is not None]
    if not common:
        return {"cameras": 0, "shift_median_m": None, "heading_change_median_deg": None}
    shift = [math.dist(parent[p]["enu"], cameras[p]["enu"]) for p in common]
    turn = [abs(angle_diff_deg(cameras[p]["heading_deg"], parent[p]["heading_deg"])) for p in common
            if parent[p].get("heading_deg") is not None and cameras[p].get("heading_deg") is not None]
    return {"cameras": len(common), "shift_median_m": round(statistics.median(shift), 3),
            "heading_change_median_deg": round(statistics.median(turn), 3) if turn else None}


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
    previous model of that target stays the one shown. Nor does a model replace
    a newer one of its target (superseded_by, decided under the landmark lock):
    it is stored unpublished with meta["superseded_by"], and the job ends done
    with a note in its error field. Every model records its inputs
    (meta["inputs"], model_inputs); a published COLMAP model is also saved to
    the landmark's map (landmark_map.save_map).

    With Params.map_continue (and a reconstructor whose class sets map_continue)
    a job continues the saved map of the model its target shows (_map_plan):
    its new photos are registered into that model. It builds from scratch when
    there is none to continue, when its inputs reach Params.map_rebuild_ratio
    times those of the last build from scratch, after Params.map_max_continuations
    continuations in a row, and once more when a continued model fails the
    quality gate. Every model then records meta["map"]: its mode (scratch,
    continued, continued->scratch), why, the model it continued and how far it
    moved against it (continuity).

    Structure in the product (Settings.stylize_published): every model published
    for a landmark with an OSM parts snapshot (structure_slug) gets a stylize job
    (JOB_STYLIZE, Job.model_id). The worker takes reconstruction jobs first and
    starts a stylize job only while no reconstruction job is queued; it runs
    crowd3d.structure_pipeline in a child process (Settings.stylize_timeout_s,
    STYLIZE_THREADS threads) on the model's saved map. The child writes only its
    work dir and the landmark's photo cache, never the database: this process
    moves the GLB next to the PLY and records it (ModelRecord.stylized). A model
    that is no longer the one its target shows is not stylized (the job ends done
    with a note); a child that fails or times out fails the job and writes no GLB.
    """

    def __init__(self, store: Store, settings: Settings, reconstructor: Reconstructor) -> None:
        self.store = store
        self.settings = settings
        self.reconstructor = reconstructor
        # (priority, sequence, job id or None to wake the worker): reconstruction jobs first, then by arrival
        self._queue: queue.PriorityQueue[tuple[int, int, str | None]] = queue.PriorityQueue()
        self._seq = itertools.count()
        self._deferred: list[str] = []      # inline_jobs: stylize jobs run once the call's reconstructions are done
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
                self._submit(job)
            self._run_deferred()
        else:
            for job in leftover:
                self._enqueue(job)
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
        self._queue.put((-1, next(self._seq), None))
        self._thread.join(timeout)
        self._thread = None

    def schedule(self, landmark_id: str) -> list[Job]:
        """compute plan_jobs for the landmark, persist the new jobs, submit them; return them."""
        with self._landmark_lock(landmark_id):
            photos = self.store.list_photos(landmark_id, PHOTO_ACCEPTED)
            requests = plan_jobs(photos, self.store.list_jobs(landmark_id), self.settings)
            jobs = [self._create(landmark_id, r) for r in requests]
        submitted = [self._submit(j) for j in jobs]
        self._run_deferred()
        return submitted

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
        job = self._submit(job)
        self._run_deferred()
        return job

    def run_job(self, job: Job) -> Job:
        """Execute one job: mark running, reconstruct into
        <models_dir>/<landmark_id>/<model_id>.ply, add a ModelRecord, mark done
        (a stylize job: _stylize). Any exception or result.ok == False marks the
        job failed with an error message.
        """
        job = self.store.update_job(
            job.id, status=JOB_RUNNING, started_at=time.time(), finished_at=None, error=None,
            attempts=job.attempts + 1,
        )
        try:
            return self._stylize(job) if job.kind == JOB_STYLIZE else self._reconstruct(job)
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
        base, plan = self._map_plan(job)

        def run(base_: MapBase | None) -> ReconstructionResult:
            extra = {**({"cache": cache} if cache is not None else {}),
                     **({"base": base_} if base_ is not None else {})}
            return self.reconstructor.reconstruct(photos, lm.lat, lm.lon, out_ply=out_ply, work_dir=work_dir, **extra)

        try:
            result, report = run(base), None
            if base is not None and not (result.continued or {}).get("accepted"):
                plan = {**plan, "mode": "continued->scratch", "parent_model_id": None, "n_scratch": len(job.photo_ids),
                        "continuations": 0,
                        "reason": f"not continued: {(result.continued or {}).get('reason') or result.message}"}
            elif base is not None and result.ok:
                report = assess(result, job.kind, records, lm, s)
                if not report.passed:      # one rerun from scratch (the feature cache makes it cost mapping only)
                    log.info("job %s (%s, sector %s): the continued model failed the quality check (%s); building it"
                             " from scratch", job.id, job.kind, job.sector, "; ".join(report.reasons))
                    plan = {**plan, "mode": "continued->scratch", "parent_model_id": None,
                            "n_scratch": len(job.photo_ids), "continuations": 0,
                            "reason": "the continued model failed the quality check: " + "; ".join(report.reasons)}
                    result, report = run(None), None
            return self._store_result(job, lm, records, result, model_id, out_ply, work_dir, plan, report)
        finally:
            if not s.keep_work_dirs:
                # The job's COLMAP database takes ~1.5 MB per photo; with Params.feature_cache the landmark's
                # FeatureCache keeps its features and matches for the next job, and a published model's sparse
                # model is saved to the landmark's map first.
                shutil.rmtree(work_dir, ignore_errors=True)

    def _store_result(self, job: Job, lm: Landmark, records: list[PhotoRecord], result: ReconstructionResult,
                      model_id: str, out_ply: Path, work_dir: Path, plan: dict[str, Any] | None = None,
                      report: QAReport | None = None) -> Job:
        s = self.settings
        if not result.ok or not out_ply.is_file():
            error = result.message if not result.ok else "reconstructor reported success but wrote no PLY"
            log.warning("job %s (%s, sector %s) failed: %s", job.id, job.kind, job.sector, error)
            return self._finish(job, JOB_FAILED, error or "reconstruction failed")
        report = report or assess(result, job.kind, records, lm, s)
        if plan is not None and plan.get("parent_model_id"):
            parent = next((m for m in self.store.list_models(job.landmark_id) if m.id == plan["parent_model_id"]), None)
            plan = {**plan, "continuity": continuity((parent.meta if parent else {}).get("cameras") or {},
                                                     result.cameras)}
        ply_rel = out_ply.relative_to(s.data_dir).as_posix()
        with self._landmark_lock(job.landmark_id):
            newer = (superseded_by(self.store, job.landmark_id, job.kind, job.sector, job.photo_ids)
                     if report.passed else None)
            meta_text = to_json({**result.to_meta(), "ply_path": ply_rel, "published": report.passed and newer is None,
                                 "qa": report.to_meta(), "inputs": model_inputs(records),
                                 **({"map": plan} if plan is not None else {}),
                                 **({"superseded_by": newer.id} if newer is not None else {})}, indent=2)
            out_ply.with_suffix(".json").write_text(meta_text, encoding="utf-8")
            model = self.store.add_model(
                ModelRecord(
                    id=model_id,
                    landmark_id=job.landmark_id,
                    job_id=job.id,
                    kind=job.kind,
                    sector=job.sector,
                    ply_path=ply_rel,
                    num_photos=len(job.photo_ids),
                    meta=json.loads(meta_text),
                    created_at=time.time(),
                )
            )
        if newer is not None:
            note = (f"superseded: model {newer.id} of newer photos is published for this target; this model is"
                    " stored unpublished")
            log.info("job %s (%s, sector %s): %s", job.id, job.kind, job.sector, note)
            return self._finish(job, JOB_DONE, note)
        if not report.passed:
            kept = any((m.kind, m.sector) == (job.kind, job.sector)
                       for m in self.store.published_models(job.landmark_id))
            outcome = "the previous model stays published" if kept else "nothing is published for it yet"
            error = f"quality check failed, {outcome}: " + "; ".join(report.reasons)
            log.warning("job %s (%s, sector %s): %s", job.id, job.kind, job.sector, error)
            return self._finish(job, JOB_FAILED, error)
        self._save_map(model, work_dir / "sparse" / "0")
        done = self._finish(job, JOB_DONE, None)
        try:
            self._queue_stylize(model, job.photo_ids)
        except Exception:          # the stylized GLB is extra: never the published model's failure
            log.exception("job %s: queueing the stylize job of model %s failed", job.id, model.id)
        return done

    def _queue_stylize(self, model: ModelRecord, photo_ids: list[str]) -> Job | None:
        """A stylize job for a model just published: when the product makes stylized models
        (Settings.stylize_published), its landmark has an OSM parts snapshot (structure_slug) and the model a saved
        map (a COLMAP model)."""
        s = self.settings
        if not s.stylize_published:
            return None
        if structure_slug(s, self.store.get_landmark(model.landmark_id)) is None:
            return None
        if load_map(s.maps_dir / model.landmark_id / MODELS_DIR, model.id) is None:
            return None
        job = self.store.create_job(Job(id=uuid.uuid4().hex, landmark_id=model.landmark_id, kind=JOB_STYLIZE,
                                        sector=model.sector, status=JOB_QUEUED, photo_ids=list(photo_ids),
                                        created_at=time.time(), model_id=model.id))
        return self._submit(job)

    def _stylize(self, job: Job) -> Job:
        """A stylize job (the class docstring): the model's stylized GLB, made by a child process."""
        s = self.settings
        model = self.store.get_model(job.model_id)
        lm = self.store.get_landmark(job.landmark_id)
        with self._landmark_lock(lm.id):
            shown = {(m.kind, m.sector): m for m in self.store.published_models(lm.id)}.get((model.kind, model.sector))
        if shown is None or shown.id != model.id:
            return self._finish(job, JOB_DONE, f"superseded: model {model.id} is no longer the one its target shows"
                                               f"{f' (model {shown.id} is)' if shown else ''}; it gets no stylized GLB")
        slug = structure_slug(s, lm)
        if slug is None:
            return self._finish(job, JOB_FAILED, "stylize failed: the landmark has no OSM parts snapshot")
        root = s.maps_dir / lm.id / MODELS_DIR
        doc = load_map(root, model.id)
        if doc is None:
            return self._finish(job, JOB_FAILED, f"stylize failed: model {model.id} has no saved map")
        processed = {p.id: p.processed_path for p in self.store.list_photos(lm.id)}
        data = s.data_dir.resolve()                           # the child runs in PACKAGE_ROOT: absolute paths only
        ply = data / model.ply_path
        work = data / s.work_dir.relative_to(s.data_dir) / f"{job.id}.{job.attempts}"     # a fresh dir per attempt
        shutil.rmtree(work, ignore_errors=True)
        (work / "out").mkdir(parents=True)
        spec = {"catalog": str(Path(s.structure_catalog).resolve()), "slug": slug,
                "landmark": {"id": lm.id, "name": lm.name, "lat": lm.lat, "lon": lm.lon},
                "model": {"id": model.id, "meta": str(ply.with_suffix(".json")),
                          "sparse": str(root.resolve() / model.id / "sparse"),
                          "images": {name: str(data / processed[pid])
                                     for name, pid in (doc.get("images") or {}).items() if processed.get(pid)}},
                "out": str(work / "out"), "cache": str((s.maps_dir / lm.id / STRUCTURE_CACHE).resolve()),
                "max_seconds": s.stylize_timeout_s, "kill_after_s": s.stylize_timeout_s + 60,
                "threads": STYLIZE_THREADS}
        (work / "spec.json").write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
        try:
            status, why = self._run_child(work)
            try:
                result = json.loads((work / "out" / "result.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                result = {}
            glb = work / "out" / f"{model.id}.glb"
            if status != 0 or not result.get("ok") or not glb.is_file():
                error = why or result.get("error") or f"the stylize process exited with status {status}"
                log.warning("job %s: stylizing model %s failed: %s", job.id, model.id, error)
                return self._finish(job, JOB_FAILED, f"stylize failed, model {model.id} gets no stylized GLB: {error}")
            os.replace(glb, ply.with_suffix(".glb"))
            summary = work / "out" / f"{model.id}.style.json"
            if summary.is_file():
                os.replace(summary, ply.with_suffix(".style.json"))
            keep = ("registration", "massing_source", "parts_built", "judge", "adoptions", "observed_walls",
                    "num_walls", "glb_bytes", "seconds", "cpu_s", "peak_rss_mb", "within_limits", "source")
            self.store.set_model_stylized(model.id, {
                "glb_path": ply.with_suffix(".glb").relative_to(data).as_posix(), "job_id": job.id,
                "created_at": time.time(), **{k: result.get(k) for k in keep}})
            return self._finish(job, JOB_DONE, None)
        finally:
            if not s.keep_work_dirs:
                shutil.rmtree(work, ignore_errors=True)

    def _run_child(self, work: Path) -> tuple[int | None, str | None]:
        """Run the stylize child on work/spec.json (output to work/child.log): (exit status, None), or (None, why)
        when it was killed at Settings.stylize_timeout_s."""
        timeout = self.settings.stylize_timeout_s
        env = {**os.environ, **{k: str(STYLIZE_THREADS)
                                for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}}
        with open(work / "child.log", "wb") as out:
            try:
                done = subprocess.run(self._stylize_command(work / "spec.json"), cwd=PACKAGE_ROOT, env=env, stdout=out,
                                      stderr=subprocess.STDOUT, timeout=timeout, check=False)
            except subprocess.TimeoutExpired:
                return None, f"timed out after {timeout:g} s"
        return done.returncode, None

    def _stylize_command(self, spec: Path) -> list[str]:
        return [sys.executable, "-m", "crowd3d.structure_pipeline", "run", "--spec", str(spec)]

    def _save_map(self, model: ModelRecord, sparse: Path) -> None:
        """Save a published model's sparse model to the landmark's map (when its reconstructor wrote one), keeping the
        newest landmark_map.MAPS_KEPT of its target. A failure costs the map, never the published model. (A stylize
        job reads the map of its target's latest published model only, always the newest one kept.)"""
        if not (sparse / "images.bin").is_file():
            return
        meta = model.meta
        plan = meta.get("map") or {}
        record = {"landmark_id": model.landmark_id, "kind": model.kind, "sector": model.sector, "job_id": model.job_id,
                  "created_at": model.created_at,
                  "feature_version": job_feature_version(self.settings.processing_max_side_px),
                  "anchor": meta.get("anchor"), "georegistered": meta.get("georegistered"),
                  "yaw_sigma_deg": meta.get("yaw_sigma_deg"), "scale_sigma": meta.get("scale_sigma"),
                  "inputs": meta.get("inputs"), "mode": plan.get("mode", "scratch"),
                  "parent_model_id": plan.get("parent_model_id"),
                  "n_scratch": plan.get("n_scratch", (meta.get("inputs") or {}).get("count")),
                  "continuations": plan.get("continuations", 0)}
        root = self.settings.maps_dir / model.landmark_id / MODELS_DIR
        try:
            save_map(root, model.id, sparse, record)
            prune_maps(root, model.kind, model.sector)
        except (OSError, RuntimeError, ValueError) as e:
            log.warning("model %s: saving its map under %s failed: %s", model.id, root, e)

    def _map_plan(self, job: Job) -> tuple[MapBase | None, dict[str, Any] | None]:
        """With Params.map_continue and a reconstructor that continues: (the saved map the job continues, or None
        when it builds from scratch; its meta["map"] so far). Without: (None, None), and models record no "map".
        The base is the published model its target shows now, when it has a saved map of the current feature version
        and the job has all of its photos, it is no more than Params.map_max_continuations - 1 continuations after a
        build from scratch and the job has fewer than Params.map_rebuild_ratio times that build's photos."""
        p = params.current()
        if not (p.map_continue and getattr(self.reconstructor, "map_continue", False)):
            return None, None
        n = len(job.photo_ids)
        plan: dict[str, Any] = {"mode": "scratch", "reason": None, "base_model_id": None, "parent_model_id": None,
                                "n_scratch": n, "continuations": 0}
        with self._landmark_lock(job.landmark_id):
            current = {(m.kind, m.sector): m for m in self.store.published_models(job.landmark_id)}.get(
                (job.kind, job.sector))
        if current is None:
            return None, {**plan, "reason": "nothing is published for this target yet"}
        plan["base_model_id"] = current.id
        root = self.settings.maps_dir / job.landmark_id / MODELS_DIR
        doc = load_map(root, current.id)
        theirs = set(self.store.get_job(current.job_id).photo_ids)
        n_scratch = int((doc or {}).get("n_scratch") or len(theirs))
        continuations = int((doc or {}).get("continuations") or 0)
        if doc is None:
            reason = "the published model has no saved map"
        elif doc.get("feature_version") != job_feature_version(self.settings.processing_max_side_px):
            reason = "the published model's saved map has features of another version"
        elif not theirs <= set(job.photo_ids):
            reason = "the job lacks photos of the published model"
        elif continuations >= p.map_max_continuations:
            reason = f"the published model is continuation {continuations} in a row"
        elif n >= p.map_rebuild_ratio * n_scratch:
            reason = f"{n} photos, at least {p.map_rebuild_ratio:g} x the {n_scratch} of the last build from scratch"
        else:
            base = MapBase(current.id, root / current.id / "sparse", dict(doc.get("images") or {}), n_scratch,
                           continuations)
            return base, {**plan, "mode": "continued", "parent_model_id": current.id, "n_scratch": n_scratch,
                          "continuations": continuations + 1}
        return None, {**plan, "reason": reason}

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
        if not self.settings.inline_jobs:
            self._enqueue(job)
            return job
        if job.kind == JOB_STYLIZE:
            with self._cv:
                self._deferred.append(job.id)
            return job
        return self.run_job(job)

    def _run_deferred(self) -> None:
        """inline_jobs: run the stylize jobs submitted so far, while no reconstruction job is queued."""
        while self.settings.inline_jobs and not self._reconstruction_queued():
            with self._cv:
                if not self._deferred:
                    return
                job_id = self._deferred.pop(0)
            job = self.store.get_job(job_id)
            if job.status == JOB_QUEUED:
                self.run_job(job)

    def _reconstruction_queued(self) -> bool:
        return any(j.kind != JOB_STYLIZE for j in self.store.list_jobs(statuses=(JOB_QUEUED,)))

    def _enqueue(self, job: Job) -> None:
        with self._cv:
            if job.id in self._pending:
                return
            self._pending.add(job.id)
        self._queue.put((_PRIORITY.get(job.kind, 0), next(self._seq), job.id))

    def _work(self, stopping: threading.Event) -> None:
        while not stopping.is_set():
            item = self._queue.get()
            job_id = item[2]
            if job_id is None:
                continue
            held = False
            try:
                job = self.store.get_job(job_id)
                # An id left in the queue across stop()/start() may have run meanwhile.
                if job.status != JOB_QUEUED:
                    pass
                elif job.kind == JOB_STYLIZE and self._reconstruction_queued():
                    # a reconstruction job (about to be handed over) goes first: this one waits its turn
                    held = True
                    self._queue.put(item)
                    stopping.wait(0.2)
                else:
                    self.run_job(job)
                    if job.kind != JOB_STYLIZE:
                        self.schedule(job.landmark_id)
            except Exception:
                log.exception("job worker error on %s", job_id)
            finally:
                if not held:
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
