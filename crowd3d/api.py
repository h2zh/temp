"""HTTP API and static web UI.

The JSON contract lives in the prototype spec ("HTTP API") and is mirrored by web/*.js.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import re
import time
import unicodedata
import uuid
from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, StringConstraints
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from .config import Settings
from .coverage import compute_coverage
from .ingest import UploadOutcome, ingest_upload
from .jobs import JobRunner
from .reconstruct import Reconstructor, get_reconstructor
from .records import JOB_FACE, JOB_FULL, PHOTO_ACCEPTED, Job, Landmark, ModelRecord, PhotoRecord
from .store import AlreadyExists, NotFound, Store

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
LANDMARK_ID_PATTERN = r"^[a-z0-9][a-z0-9-]{0,63}$"
MAX_FILES_PER_UPLOAD = 50
MAX_USER_ID_CHARS = 64
DETAIL_MAX_JOBS = 20
DETAIL_MAX_PHOTOS = 200
DEMO_LANDMARK_ID = "demo-building"
DEMO_RADIUS_M = 150.0
R_SERVER_ERROR = "server_error"       # upload result: storing this file failed on the server (disk full, ...)
BODY_SLACK_BYTES = 1024 * 1024        # multipart headers and form fields on top of the files

log = logging.getLogger(__name__)


class LandmarkIn(BaseModel):
    id: Annotated[str, Field(pattern=LANDMARK_ID_PATTERN)] | None = None
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=100)]
    lat: Annotated[float, Field(ge=-90, le=90, allow_inf_nan=False)]
    lon: Annotated[float, Field(ge=-180, le=180, allow_inf_nan=False)]
    radius_m: Annotated[float, Field(gt=0, le=5000, allow_inf_nan=False)] | None = None


class RebuildIn(BaseModel):
    kind: Literal["full", "face"]
    sector: int | None = None


def slugify(name: str) -> str:
    """ASCII landmark id from a name; "lm-" + 8 hex chars when nothing ASCII is left (e.g. Chinese names)."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_name.lower()).strip("-")[:64].rstrip("-")
    return slug or f"lm-{uuid.uuid4().hex[:8]}"


class DataDirInUse(RuntimeError):
    pass


@contextmanager
def data_dir_lock(settings: Settings) -> Iterator[None]:
    """Exclusive lock on the data dir for the life of the server.

    The job runner re-queues every job left "running" when it starts, so a
    second server on the same data dir (an accidental second launch, a rolling
    deploy) would run the first one's jobs again in the same work dirs. It must
    fail before touching anything; uvicorn runs the lifespan before binding.
    """
    try:
        import fcntl
    except ImportError:      # not on POSIX: no protection
        yield
        return
    fd = os.open(settings.data_dir / "crowd3d.lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise DataDirInUse(f"another Crowd3D server is using the data dir {settings.data_dir}") from None
        yield
    finally:
        os.close(fd)


class BodySizeLimit:
    """413 for request bodies above max_bytes, before multipart parsing spools them to temp files."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        length = dict(scope.get("headers") or []).get(b"content-length")
        if length is not None and length.isdigit() and int(length) > self.max_bytes:
            await self._reject(send)
            return
        received, exceeded, rejected = 0, False, False

        async def counted_receive() -> Message:
            nonlocal received, exceeded
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    exceeded = True
                    raise _BodyTooLarge
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal rejected
            if not exceeded:
                await send(message)
            elif not rejected:
                # FastAPI turns the aborted body parsing into its own 400; answer 413 instead.
                rejected = True
                await self._reject(send)

        try:
            await self.app(scope, counted_receive, guarded_send)
        except _BodyTooLarge:
            if not rejected:
                rejected = True
                await self._reject(send)

    async def _reject(self, send: Send) -> None:
        response = JSONResponse({"detail": f"request body over {self.max_bytes} bytes"}, status_code=413)
        await send({"type": "http.response.start", "status": 413, "headers": response.raw_headers})
        await send({"type": "http.response.body", "body": response.body})


class _BodyTooLarge(Exception):
    pass


class PlyFiles(StaticFiles):
    """Serves only the model point clouds; the .json sidecars next to them hold per-photo camera poses."""

    async def get_response(self, path: str, scope: Scope):
        if not path.endswith(".ply"):
            raise HTTPException(404)
        return await super().get_response(path, scope)


def create_app(settings: Settings | None = None, reconstructor: Reconstructor | None = None) -> FastAPI:
    settings = settings or Settings()
    settings.ensure_dirs()
    store = Store(settings)
    runner = JobRunner(store, settings, reconstructor or get_reconstructor(settings.reconstructor, settings))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            with data_dir_lock(settings):
                runner.start()
                try:
                    yield
                finally:
                    runner.stop()
        finally:
            store.close()

    app = FastAPI(title="Crowd3D", lifespan=lifespan)
    app.state.settings = settings
    app.state.store = store
    app.state.runner = runner

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes the input back, and a NaN/inf input
        # (Python's json accepts the NaN literal) would crash the JSON rendering.
        detail = jsonable_encoder(exc.errors(), custom_encoder={float: _json_float})
        return JSONResponse(status_code=422, content={"detail": detail})

    def get_landmark(landmark_id: str) -> Landmark:
        try:
            return store.get_landmark(landmark_id)
        except NotFound:
            raise HTTPException(404, f"unknown landmark {landmark_id!r}") from None

    @app.get("/api/health")
    def health():
        return {"ok": True, "reconstructor": runner.reconstructor.name}

    @app.get("/api/landmarks")
    def list_landmarks():
        return {"landmarks": [landmark_summary(store, settings, lm) for lm in store.list_landmarks()]}

    @app.post("/api/landmarks", status_code=201)
    def create_landmark(body: LandmarkIn):
        lm = Landmark(
            id=body.id or slugify(body.name),
            name=body.name,
            lat=body.lat,
            lon=body.lon,
            radius_m=settings.default_radius_m if body.radius_m is None else body.radius_m,
            created_at=time.time(),
        )
        try:
            lm = store.create_landmark(lm)
        except AlreadyExists:
            raise HTTPException(409, f"landmark {lm.id!r} already exists") from None
        return landmark_summary(store, settings, lm)

    @app.get("/api/landmarks/{landmark_id}")
    def landmark_detail_route(landmark_id: str):
        return landmark_detail(store, settings, get_landmark(landmark_id))

    @app.post("/api/photos")
    def upload_photos(
        files: Annotated[list[UploadFile], File()],
        user_id: Annotated[str, Form()],
        landmark_id: Annotated[str | None, Form()] = None,
    ):
        user_id = user_id.strip()
        if not 1 <= len(user_id) <= MAX_USER_ID_CHARS or any(unicodedata.category(c) == "Cc" for c in user_id):
            raise _invalid("user_id", f"must be 1-{MAX_USER_ID_CHARS} characters without control characters", user_id)
        if not 1 <= len(files) <= MAX_FILES_PER_UPLOAD:
            raise _invalid("files", f"upload 1-{MAX_FILES_PER_UPLOAD} files at a time", len(files))
        landmark = get_landmark(landmark_id.strip()) if landmark_id and landmark_id.strip() else None
        outcomes = []
        for f in files:
            try:
                f.file.seek(0)
                # One byte past the limit is enough for ingest to reject the file as too large.
                data = f.file.read(settings.max_upload_bytes + 1)
                outcomes.append(ingest_upload(store, settings, filename=f.filename or "", data=data,
                                              user_id=user_id, landmark=landmark))
            except Exception:
                # One file must not lose the others: earlier ones are stored and still need planning.
                log.exception("storing upload %r failed", f.filename)
                outcomes.append(UploadOutcome(f.filename or "", None, False, [R_SERVER_ERROR], []))
        # Plan once per request, so a batch of photos makes one job rather than one per threshold step.
        touched = dict.fromkeys(o.photo.landmark_id for o in outcomes if o.accepted)
        triggered = [job for lm_id in touched for job in runner.schedule(lm_id)]
        return {"results": [upload_result(o) for o in outcomes], "triggered_jobs": [job_info(j) for j in triggered]}

    @app.post("/api/landmarks/{landmark_id}/rebuild", status_code=202)
    def rebuild(landmark_id: str, body: RebuildIn):
        lm = get_landmark(landmark_id)
        sector = None
        if body.kind == JOB_FACE:
            sector = body.sector
            if sector is None or not 0 <= sector < settings.num_sectors:
                raise _invalid("sector", f"face rebuilds need a sector in 0..{settings.num_sectors - 1}", sector)
        try:
            job = runner.submit_manual(lm.id, body.kind, sector)
        except ValueError as e:
            raise HTTPException(409, str(e)) from None
        return job_info(job)

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str):
        try:
            return job_info(store.get_job(job_id))
        except NotFound:
            raise HTTPException(404, f"unknown job {job_id!r}") from None

    app.mount("/files/models", PlyFiles(directory=settings.models_dir), name="models")
    # Mounted last so every /api route above takes precedence.
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    app.add_middleware(BodySizeLimit, max_bytes=MAX_FILES_PER_UPLOAD * settings.max_upload_bytes + BODY_SLACK_BYTES)
    return app


def _invalid(field: str, msg: str, value: Any) -> RequestValidationError:
    """A 422 shaped like FastAPI's own validation errors, so clients parse one format."""
    return RequestValidationError([{"type": "value_error", "loc": ("body", field), "msg": msg, "input": value}])


def _json_float(value: float) -> float | str:
    return value if math.isfinite(value) else str(value)


def _finite(value: Any) -> float | None:
    # Starlette's JSONResponse refuses NaN/inf; reconstructor metadata may contain them.
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None


def model_url(settings: Settings, model: ModelRecord) -> str:
    rel = (settings.data_dir / model.ply_path).relative_to(settings.models_dir).as_posix()
    return f"/files/models/{rel}"


def model_info(settings: Settings, model: ModelRecord) -> dict[str, Any]:
    meta = model.meta
    return {
        "id": model.id,
        "kind": model.kind,
        "sector": model.sector,
        "url": model_url(settings, model),
        "created_at": model.created_at,
        "num_photos": model.num_photos,
        "num_registered": int(meta.get("num_registered") or 0),
        "num_points": int(meta.get("num_points") or 0),
        "georegistered": bool(meta.get("georegistered")),
        "alignment_rmse_m": _finite(meta.get("alignment_rmse_m")),
        "heading_mae_deg": _finite(meta.get("heading_mae_deg")),
    }


def job_info(job: Job) -> dict[str, Any]:
    return {
        "id": job.id,
        "landmark_id": job.landmark_id,
        "kind": job.kind,
        "sector": job.sector,
        "status": job.status,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
        "error": job.error,
        "num_photos": len(job.photo_ids),
    }


def camera_estimates(models: list[ModelRecord]) -> dict[str, dict[str, Any]]:
    """photo_id -> camera entry of the newest model containing it; full models win over face models.

    Only published models count, and a camera with flags (its pose disagrees with the photo's own
    GPS/compass, lies outside the radius, ...) is not an estimate: an older model's entry stays.
    """
    estimates: dict[str, dict[str, Any]] = {}
    published = [m for m in models if m.meta.get("published", True)]
    for m in sorted(published, key=lambda m: (m.kind == JOB_FULL, m.created_at, m.id)):
        for pid, cam in (m.meta.get("cameras") or {}).items():
            if isinstance(cam, dict) and not cam.get("flags"):
                estimates[pid] = cam
    return estimates


def photo_info(photo: PhotoRecord, camera: dict[str, Any] | None) -> dict[str, Any]:
    enu = None
    heading = None
    if camera:
        heading = _finite(camera.get("heading_deg"))
        raw = camera.get("enu")
        if isinstance(raw, list) and len(raw) == 3 and all(_finite(c) is not None for c in raw):
            enu = [float(c) for c in raw]
    return {
        "id": photo.id,
        "user_id": photo.user_id,
        "sector": photo.sector,
        "bearing_deg": photo.bearing_deg,
        "distance_m": photo.distance_m,
        "heading_deg": photo.heading_deg,
        "estimated_heading_deg": heading,
        "estimated_enu": enu,
        "uploaded_at": photo.uploaded_at,
    }


def upload_result(outcome: UploadOutcome) -> dict[str, Any]:
    p = outcome.photo
    return {
        "filename": outcome.filename,
        "photo_id": p.id if p else None,
        "accepted": outcome.accepted,
        "reasons": outcome.reasons,
        "warnings": outcome.warnings,
        "landmark_id": p.landmark_id if p else None,
        "sector": p.sector if p else None,
        "bearing_deg": p.bearing_deg if p else None,
        "distance_m": p.distance_m if p else None,
    }


def landmark_summary(
    store: Store,
    settings: Settings,
    lm: Landmark,
    photos: list[PhotoRecord] | None = None,
    latest: list[ModelRecord] | None = None,
) -> dict[str, Any]:
    photos = store.list_photos(lm.id, PHOTO_ACCEPTED) if photos is None else photos
    latest = store.latest_models(lm.id) if latest is None else latest
    cov = compute_coverage(photos, settings)
    face_sectors = {m.sector for m in latest if m.kind == JOB_FACE}
    full = next((m for m in latest if m.kind == JOB_FULL), None)
    return {
        "id": lm.id,
        "name": lm.name,
        "lat": lm.lat,
        "lon": lm.lon,
        "radius_m": lm.radius_m,
        "num_photos": len(photos),
        "num_contributors": len({p.user_id for p in photos}),
        "progress": cov.progress,
        "full_ready": cov.full_ready,
        "sectors": [
            {
                "index": s.index,
                "label": s.label,
                "count": s.count,
                "state": s.state,
                "bearing_center_deg": s.bearing_center_deg,
                "has_face_model": s.index in face_sectors,
            }
            for s in cov.sectors
        ],
        "latest_full_model": model_info(settings, full) if full else None,
    }


def landmark_detail(store: Store, settings: Settings, lm: Landmark) -> dict[str, Any]:
    photos = store.list_photos(lm.id, PHOTO_ACCEPTED)
    latest = store.latest_models(lm.id)
    cameras = camera_estimates(store.published_models(lm.id))
    newest_photos = photos[::-1][:DETAIL_MAX_PHOTOS]
    return {
        **landmark_summary(store, settings, lm, photos, latest),
        "contributors": [{"user_id": u, "photos": n} for u, n in store.contributors(lm.id)],
        "models": [model_info(settings, m) for m in latest],
        "jobs": [job_info(j) for j in store.list_jobs(lm.id)[::-1][:DETAIL_MAX_JOBS]],
        "photos": [photo_info(p, cameras.get(p.id)) for p in newest_photos],
        "thresholds": {
            "face_min_photos": settings.face_min_photos,
            "full_min_photos": settings.full_min_photos,
            "full_min_sectors": compute_coverage(photos, settings).full_min_sectors,
            "num_sectors": settings.num_sectors,
        },
    }


def ensure_demo_landmark(store: Store) -> Landmark:
    """The landmark the synthetic demo photos are taken around; created if missing."""
    from . import synthetic  # pulls in OpenCV, so only when asked for

    try:
        return store.get_landmark(DEMO_LANDMARK_ID)
    except NotFound:
        pass
    lm = Landmark(
        id=DEMO_LANDMARK_ID,
        name=synthetic.DEMO_NAME,
        lat=synthetic.DEMO_LAT,
        lon=synthetic.DEMO_LON,
        radius_m=DEMO_RADIUS_M,
        created_at=time.time(),
    )
    try:
        return store.create_landmark(lm)
    except AlreadyExists:
        return store.get_landmark(DEMO_LANDMARK_ID)


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m crowd3d.api [--host 127.0.0.1] [--port 8000] [--data-dir data] [--reconstructor colmap] [--seed-demo]"""
    ap = argparse.ArgumentParser(prog="python -m crowd3d.api", description="Crowd3D HTTP API and web UI")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", type=Path, default=None, help="default: $CROWD3D_DATA_DIR or ./data")
    ap.add_argument(
        "--reconstructor", choices=("colmap", "fake"), default=None, help="default: $CROWD3D_RECONSTRUCTOR or colmap"
    )
    ap.add_argument("--seed-demo", action="store_true", help=f"create the {DEMO_LANDMARK_ID!r} landmark if missing")
    args = ap.parse_args(argv)

    settings = Settings()
    if args.data_dir is not None:
        settings = replace(settings, data_dir=args.data_dir)
    if args.reconstructor is not None:
        settings = replace(settings, reconstructor=args.reconstructor)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app(settings)
    if args.seed_demo:
        lm = ensure_demo_landmark(app.state.store)
        logging.getLogger(__name__).info("demo landmark %s (%s) at %.4f, %.4f", lm.id, lm.name, lm.lat, lm.lon)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
