"""One uploaded file -> validated PhotoRecord, plus the stored image files when accepted."""

from __future__ import annotations

import io
import math
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from PIL import Image, ImageOps

from .config import Settings
from .exif import PhotoMeta, UnreadableImage, build_exif_bytes, read_photo_meta
from .records import PHOTO_ACCEPTED, PHOTO_REJECTED, Landmark, PhotoRecord
from .store import Store
from .validation import validate_photo

R_TOO_LARGE = "too_large"   # over Settings.max_upload_bytes; rejected before any record is made

_landmark_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


@dataclass
class UploadOutcome:
    filename: str                 # client file name, echoed for display only
    photo: PhotoRecord | None     # None only if rejected before a record could be made (too_large)
    accepted: bool
    reasons: list[str]
    warnings: list[str]


def make_processed_copy(image_bytes: bytes, meta: PhotoMeta, settings: Settings) -> bytes:
    """Upright RGB JPEG (q=92), longer side at most processing_max_side_px.

    Only focal length and camera make/model survive in its EXIF. COLMAP derives the
    focal prior from FocalLengthIn35mmFilm and the image diagonal (f35 / 43.27 * diag),
    which a uniform downscale keeps valid, but only when FocalLength is present too,
    so both are kept. GPS and compass come from the PhotoRecord instead.
    """
    limit = settings.processing_max_side_px
    with Image.open(io.BytesIO(image_bytes)) as img:
        longer = max(img.size)
        if longer > limit:
            # JPEG can decode straight at 1/2, 1/4 or 1/8 scale; far cheaper than a full 12 MP decode.
            s = limit / longer
            img.draft("RGB", (math.ceil(img.width * s), math.ceil(img.height * s)))
        out = ImageOps.exif_transpose(img).convert("RGB")
    if max(out.size) > limit:
        out.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    exif = build_exif_bytes(focal_mm=meta.focal_mm, focal_35mm=meta.focal_35mm, make=meta.make, model=meta.model)
    out.save(buf, "JPEG", quality=92, exif=exif)
    return buf.getvalue()


def ingest_upload(
    store: Store,
    settings: Settings,
    *,
    filename: str,
    data: bytes,
    user_id: str,
    landmark: Landmark | None,
) -> UploadOutcome:
    """Validate one upload and record it, accepted or not.

    landmark=None means "the landmark whose radius contains the photo's GPS fix".
    Accepted photos get photos/<landmark_id>/<photo_id>.jpg (the original bytes)
    and <photo_id>.proc.jpg (make_processed_copy); rejected ones keep no bytes.
    The client file name is never used in a path.
    """
    if len(data) > settings.max_upload_bytes:
        return UploadOutcome(filename, None, False, [R_TOO_LARGE], [])
    try:
        meta = read_photo_meta(data)
    except UnreadableImage:
        meta = None
    if landmark is None and meta is not None and meta.has_gps:
        landmark = store.find_landmark_for_location(meta.lat, meta.lon)

    with _landmark_lock(landmark.id if landmark else None):
        # Held from the duplicate check to the insert, so two concurrent uploads of
        # the same photo cannot both be accepted.
        hashes = store.photo_hashes(landmark.id) if landmark else []
        result = validate_photo(data, meta, landmark, hashes, settings)
        photo = PhotoRecord(
            id=uuid.uuid4().hex,
            landmark_id=landmark.id if landmark else None,
            user_id=user_id,
            status=PHOTO_ACCEPTED if result.accepted else PHOTO_REJECTED,
            uploaded_at=time.time(),
            reasons=list(result.reasons),
            warnings=list(result.warnings),
            bearing_deg=result.bearing_deg,
            distance_m=result.distance_m,
            sector=result.sector,
            phash=result.phash,
        )
        if meta is not None:
            photo.lat, photo.lon, photo.alt = meta.lat, meta.lon, meta.alt
            photo.heading_deg = meta.heading_deg
            photo.width, photo.height = meta.width, meta.height
            photo.focal_35mm = meta.focal_35mm
            photo.captured_at = meta.captured_at
        if result.accepted:
            folder = settings.photos_dir / landmark.id
            folder.mkdir(parents=True, exist_ok=True)
            original = folder / f"{photo.id}.jpg"
            processed = folder / f"{photo.id}.proc.jpg"
            original.write_bytes(data)
            processed.write_bytes(make_processed_copy(data, meta, settings))
            photo.original_path = original.relative_to(settings.data_dir).as_posix()
            photo.processed_path = processed.relative_to(settings.data_dir).as_posix()
        photo = store.add_photo(photo)
    return UploadOutcome(filename, photo, result.accepted, photo.reasons, photo.warnings)


@contextmanager
def _landmark_lock(landmark_id: str | None) -> Iterator[None]:
    if landmark_id is None:
        yield
        return
    with _locks_guard:
        lock = _landmark_locks.setdefault(landmark_id, threading.Lock())
    with lock:
        yield
