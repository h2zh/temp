"""One uploaded file -> validated PhotoRecord, plus the stored image files when accepted."""

from __future__ import annotations

import ctypes
import io
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from PIL import Image

from .config import Settings
from .exif import (
    ALLOWED_FORMATS,
    MAX_TEXT_CHARS,
    PhotoMeta,
    UnreadableImage,
    build_exif_bytes,
    capture_time,
    read_photo_meta,
    upright,
)
from .records import PHOTO_ACCEPTED, PHOTO_REJECTED, TERMS_CURRENT, Contributor, Landmark, PhotoRecord
from .store import Store
from .subject import load_subject
from .validation import R_UNREADABLE, validate_photo

log = logging.getLogger(__name__)

# Over Settings.max_upload_bytes, or over Settings.max_image_pixels unless it is a JPEG that decodes reduced
# (PhotoMeta.decodes_reduced): rejected before any pixel is decoded and before any record is made.
R_TOO_LARGE = "too_large"
MAX_EXIF_TEXT_CHARS = MAX_TEXT_CHARS    # make/model copied into the processed copy (exif._text cuts them there)

_landmark_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class _DecodeSlots:
    """The process's image decodes in flight (D34): at most Settings.max_concurrent_decodes uploads have Pillow at
    their bytes at once, whatever request, landmark or thread pool they come from; the others wait their turn. That
    is every step of an upload that opens the image: reading its header and EXIF (read_photo_meta), and decoding its
    pixels. One decode can take hundreds of MB (a 50 MP PNG or progressive JPEG), a server's thread pool runs 40
    requests at once, and a photo that matches no landmark takes no landmark lock. The limit bounds the decodes in
    flight; that their memory goes back to the operating system afterwards, in every worker thread, is
    pin_malloc_threshold's part. An upload waiting for a slot holds its file's bytes, nothing more."""

    def __init__(self) -> None:
        self._changed = threading.Condition()
        self._active = 0

    @contextmanager
    def slot(self, limit: int) -> Iterator[None]:
        limit = max(1, limit)
        with self._changed:
            while self._active >= limit:
                self._changed.wait()
            self._active += 1
        try:
            yield
        finally:
            _release_freed_memory()             # still in the slot: the next upload starts from what was given back
            with self._changed:
                self._active -= 1
                self._changed.notify_all()      # limits may differ between callers: each re-checks its own


_decode_slots = _DecodeSlots()

_M_MMAP_THRESHOLD = -3                  # mallopt's parameter number, <malloc.h>
MALLOC_THRESHOLD_ENV = "MALLOC_MMAP_THRESHOLD_"     # glibc's own switch for the same setting, read at process start
_malloc_pinned_at: int | None = None    # the threshold this process's allocator was pinned at, once
_trim_libc = None                       # glibc, once pin_malloc_threshold was asked to keep the allocator in hand


def pin_malloc_threshold(threshold_bytes: int) -> bool:
    """Keep this process's allocator in hand, which is what makes Settings.max_concurrent_decodes bound a server's
    memory (D34; Settings.malloc_mmap_threshold_bytes says why). Two things, on glibc only and for a threshold over
    0: glibc's mmap threshold is pinned at threshold_bytes (mallopt), once, and from then on every decode slot hands
    the memory its upload freed back to the operating system when it ends (_release_freed_memory).
    True when the threshold is pinned by this call or an earlier one (the first call's value stays). False for a
    threshold of 0 or less and when the C library is not glibc (nothing is touched), and when MALLOC_THRESHOLD_ENV
    is set: the operator's own threshold is in force already and stays, and only the handing back is switched on."""
    global _malloc_pinned_at, _trim_libc
    if _malloc_pinned_at is not None:
        return True
    if threshold_bytes <= 0:
        return False
    try:
        libc = ctypes.CDLL(None)            # the C library this process runs on
        libc.gnu_get_libc_version           # glibc only (AttributeError elsewhere)
        libc.malloc_trim
    except (OSError, AttributeError):
        return False
    _trim_libc = libc
    if os.environ.get(MALLOC_THRESHOLD_ENV):
        return False
    try:
        pinned = libc.mallopt(_M_MMAP_THRESHOLD, int(threshold_bytes)) == 1
    except (TypeError, ValueError, ctypes.ArgumentError):       # (a value no C int holds)
        return False
    if pinned:
        _malloc_pinned_at = int(threshold_bytes)
        log.info("glibc mmap threshold pinned at %d bytes", _malloc_pinned_at)
    return pinned


def _release_freed_memory() -> None:
    """Hand the memory this process has freed back to the operating system (glibc's malloc_trim), at the end of
    every decode slot once pin_malloc_threshold has switched it on (D34, review round 2). The pinned threshold gives
    back blocks of its size or more by itself; smaller ones stay in the heap of the worker thread that freed them,
    and an upload can make many: a 69 KB PNG whose text chunks inflate to 64 MiB left that much behind in every
    thread it passed through, 2.3 GB in an idle server after 56 such uploads. Measured: half a millisecond after a
    12 MP photo, microseconds when there is nothing to give back."""
    libc = _trim_libc
    if libc is not None:
        libc.malloc_trim(0)


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
    with Image.open(io.BytesIO(image_bytes), formats=ALLOWED_FORMATS) as img:
        longer = max(img.size)
        if longer > limit:
            # A sequential JPEG decodes straight at 1/2, 1/4 or 1/8 scale; far cheaper than a full 12 MP decode
            # (any other image decodes in full: ingest_upload caps those at Settings.max_image_pixels).
            s = limit / longer
            img.draft("RGB", (math.ceil(img.width * s), math.ceil(img.height * s)))
        out = upright(img).convert("RGB")
    if max(out.size) > limit:
        out.thumbnail((limit, limit), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    exif = build_exif_bytes(focal_mm=meta.focal_mm, focal_35mm=meta.focal_35mm,
                            make=(meta.make or "")[:MAX_EXIF_TEXT_CHARS] or None,
                            model=(meta.model or "")[:MAX_EXIF_TEXT_CHARS] or None)
    out.save(buf, "JPEG", quality=92, exif=exif)
    return buf.getvalue()


def ingest_upload(
    store: Store,
    settings: Settings,
    *,
    filename: str,
    data: bytes,
    contributor: Contributor,
    landmark: Landmark | None,
) -> UploadOutcome:
    """Validate one upload and record it, accepted or not.

    landmark=None means "the landmark whose radius contains the photo's GPS fix".
    contributor: the registered contributor uploading it (D33, crowd3d.contributors), who must have accepted the
    current user agreement (records.TERMS_CURRENT), else ValueError before anything is read or stored. The record
    takes its user_id (the nickname) and contributor_id from it, and an accepted photo's location_consent is the
    version the contributor accepted (a rejected one gets None: it is never shown).
    Accepted photos get photos/<landmark_id>/<photo_id>.jpg (the original bytes)
    and <photo_id>.proc.jpg (make_processed_copy); rejected ones keep no bytes.
    A photo whose processed copy cannot be made is rejected as unreadable; if
    storing fails, nothing of it is left on disk and the error propagates.
    The client file name is never used in a path.
    """
    if contributor.terms != TERMS_CURRENT:
        raise ValueError("an upload needs a contributor who accepted the current user agreement")
    if len(data) > settings.max_upload_bytes:
        return UploadOutcome(filename, None, False, [R_TOO_LARGE], [])
    # Headers and EXIF only, no pixel is decoded; in a slot of the decode limit all the same (D34, review round 2):
    # Pillow reads a header into Python objects, and what that costs must follow the limit too, not the number of
    # uploads in flight. The slot is given back before the landmark's lock is taken. (The errors are caught inside
    # it: an exception on its way out would keep the reader's objects alive past the slot's end.)
    with _decode_slots.slot(settings.max_concurrent_decodes):
        try:
            meta = read_photo_meta(data)
        except UnreadableImage:
            meta = None
        except Exception as exc:    # whatever else a parser raises on hostile input: unreadable, not a server error
            log.warning("cannot read the metadata of upload %r: %r", filename, exc)
            meta = None
    # The pixel cap, for everything that costs memory by its pixel count: PNG, WebP, and a JPEG that is progressive,
    # multi-scan or lossless (D34). Only a JPEG that decodes reduced is exempt: what phones write, some at 108 MP.
    if meta is not None and not meta.decodes_reduced and meta.width * meta.height > settings.max_image_pixels:
        return UploadOutcome(filename, None, False, [R_TOO_LARGE], [])
    if landmark is None and meta is not None and meta.has_gps:
        landmark = store.find_landmark_for_location(meta.lat, meta.lon)
    # The landmark's building for the qualification (D31), before the lock: a first load parses the OSM snapshot.
    subject = load_subject(settings, landmark) if landmark is not None and meta is not None and meta.has_gps else None

    with _landmark_lock(landmark.id if landmark else None):
        # Held from the duplicate check to the insert, so two concurrent uploads of
        # the same photo cannot both be accepted.
        hashes = store.photo_hashes(landmark.id) if landmark else []
        processed = None
        # Both decodes of this upload (the analysis copy inside validate_photo, the processed copy) in one slot of
        # the process-wide limit, taken after the landmark's lock: a decode never waits for a lock.
        with _decode_slots.slot(settings.max_concurrent_decodes):
            result = validate_photo(data, meta, landmark, hashes, settings, subject=subject)
            if result.accepted:
                try:
                    processed = make_processed_copy(data, meta, settings)
                except Exception:   # a decoder or the EXIF writer choking on hostile input
                    log.warning("cannot make a processed copy of upload %r", filename, exc_info=True)
                    result.accepted = False
                    result.reasons.append(R_UNREADABLE)
        photo = PhotoRecord(
            id=uuid.uuid4().hex,
            landmark_id=landmark.id if landmark else None,
            user_id=contributor.nickname,
            status=PHOTO_ACCEPTED if result.accepted else PHOTO_REJECTED,
            uploaded_at=time.time(),
            reasons=list(result.reasons),
            warnings=list(result.warnings),
            bearing_deg=result.bearing_deg,
            distance_m=result.distance_m,
            sector=result.sector,
            phash=result.phash,
            outline_distance_m=result.outline_distance_m,
            subject_share=result.subject_share,
            location_consent=contributor.terms if result.accepted else None,
            contributor_id=contributor.id,
        )
        if meta is not None:
            photo.lat, photo.lon, photo.alt = meta.lat, meta.lon, meta.alt
            photo.gps_accuracy_m = meta.gps_accuracy_m
            photo.heading_deg = meta.heading_deg
            # true or magnetic north, for the heading's honest uncertainty (D30); any other tag value is no tag
            ref = meta.heading_ref if meta.heading_deg is not None else None
            photo.heading_ref = ref if ref in ("T", "M") else None
            photo.width, photo.height = meta.width, meta.height
            photo.focal_35mm = meta.focal_35mm
            # only EXIF's own 19-character form is kept (D34): the tag is client text and nothing reads the column
            photo.captured_at = capture_time(meta.captured_at)
        written: list = []
        try:
            if result.accepted:
                folder = settings.photos_dir / landmark.id
                folder.mkdir(parents=True, exist_ok=True)
                for path, content in ((folder / f"{photo.id}.jpg", data), (folder / f"{photo.id}.proc.jpg", processed)):
                    written.append(path)
                    path.write_bytes(content)
                photo.original_path = written[0].relative_to(settings.data_dir).as_posix()
                photo.processed_path = written[1].relative_to(settings.data_dir).as_posix()
            photo = store.add_photo(photo)
        except BaseException:
            for path in written:     # no orphan file (with its GPS EXIF) without a record
                path.unlink(missing_ok=True)
            raise
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
