"""Is the landmark's building the photo's subject? The qualification rule of validation.validate_photo (D31).

A photo with GPS qualifies when the camera stands within Settings.subject_max_distance_m of the building's OSM main
outline, or when the building is estimated to fill at least Settings.subject_min_share of the frame. The estimate
uses the GPS position, the OSM structure and the field of view; it never looks at the pixels and never uses the
compass (the camera is assumed to aim at the building: today's facing_away check is the only compass test).

- Outline distance: horizontal, from the camera's EXIF GPS to the main outline (the catalogue's osm.file), in the
  catalogue's ENU frame (geo.LocalENU at the entry's lat/lon). A camera inside the building measures 0; a camera in a
  courtyard (a hole of the outline) measures to the nearest ring, the courtyard's.
- Share: max(min(1, az_span / HFOV), min(1, el_span / VFOV)), from the GPS position at Settings.camera_height_m to
  points sampled on every built OSM part (structure.load_osm_structure: triangle edges every SAMPLE_STEP_M, thinned to
  a THIN_GRID_M grid). az_span is the smallest arc that holds every point's azimuth, el_span the spread of their
  elevations, so the tallest part counts. HFOV and VFOV follow the product's focal convention (f_px = focal_35mm /
  43.27 x image diagonal) on the upright image (width and height swapped for EXIF orientation 5-8). A focal_35mm
  that is missing or outside [Settings.subject_min_focal_35mm, Settings.subject_max_focal_35mm] is replaced by
  Settings.subject_default_focal_35mm.

The building comes from the structure catalogue binding (binding(): Settings.structure_bindings or the landmark id
as the slug, a catalogue entry with an OSM parts snapshot and an outline, its centre within the landmark's radius).
A landmark without one is not checked (validation.W_SUBJECT_UNCHECKED); a snapshot that cannot be read fails open the
same way, logged. The geometry is cached in memory and in Settings.maps_dir/<landmark id>/subject-<parts sha12>.npz.

This module runs in the server on every upload, so it imports nothing of the clay model's family (D34 part C): the
catalogue comes from crowd3d.catalog, the ring helpers from crowd3d.geom2d. Only build_subject loads
crowd3d.structure (and OpenCV with it), once per building and process, when there is no cache file yet.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import catalog_entry, find_source
from .config import Settings
from .exif import PhotoMeta
from .geo import LocalENU, haversine_m
from .geom2d import distance_to_rings, in_rings
from .reconstruct import FULL_FRAME_DIAGONAL_MM
from .records import Landmark

log = logging.getLogger(__name__)

SUBJECT_VERSION = 1          # how the points are sampled; a cached file of another version is rebuilt
SAMPLE_STEP_M = 1.0          # points along every triangle edge of the built parts, about this far apart ...
THIN_GRID_M = 0.5            # ... thinned to one point per cell of this grid
UPRIGHT_SWAPPED = (5, 6, 7, 8)   # EXIF orientations whose upright image is the stored one turned by 90 degrees

_memory: dict[tuple[Any, ...], Subject] = {}
_key_locks: dict[tuple[Any, ...], threading.Lock] = {}
_guard = threading.Lock()


@dataclass(frozen=True)
class Subject:
    """A landmark's building in the catalogue's ENU frame (lat, lon): the main outline and the sampled parts."""
    slug: str
    lat: float
    lon: float
    rings: tuple[np.ndarray, ...]    # the main outline: the outer ring (CCW), then its holes
    points: np.ndarray               # (n, 3) float32: points on every built OSM part
    parts_sha256: str


@dataclass(frozen=True)
class Qualification:
    outline_distance_m: float        # 0 inside the building
    share: float                     # estimated share of the frame the building fills, 0..1
    focal_35mm: float                # the focal the share was measured with
    default_focal: bool              # focal_35mm is Settings.subject_default_focal_35mm (EXIF missing or implausible)
    qualifies: bool


def binding(settings: Settings, lm: Landmark) -> tuple[str, dict[str, Any]] | None:
    """The catalogue slug and entry a landmark is bound to: its Settings.structure_bindings entry, else its id, when
    Settings.structure_catalog lists that slug with an OSM parts snapshot and an outline and the slug's centre lies
    within the landmark's radius (a slug naming a building elsewhere binds nothing). None: no binding. The one rule
    for both the stylized models (jobs.structure_slug) and the qualification."""
    if not settings.structure_catalog:
        return None
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
    return None if far else (slug, entry)


def load_subject(settings: Settings, lm: Landmark) -> Subject | None:
    """The bound building of the landmark (binding()), or None: no binding, or a snapshot or outline that cannot be
    read (fail open, logged: the photo is then not checked). The first load parses the OSM snapshot (0.1-10 s on the
    catalogue's landmarks); later ones come from memory or from the landmark's cache file."""
    bound = binding(settings, lm)
    if bound is None:
        return None
    slug, entry = bound
    try:
        return _cached(settings, lm, slug, entry)
    except Exception:   # whatever a broken snapshot or outline raises, uploads go on unchecked
        log.warning("landmark %s: cannot load the building of %s; its photos are not checked for the subject rule",
                    lm.id, slug, exc_info=True)
        return None


def upright_size(meta: PhotoMeta) -> tuple[int, int]:
    """(width, height) of the photo as it is shown: stored size, turned for EXIF orientations 5-8."""
    return (meta.height, meta.width) if meta.orientation in UPRIGHT_SWAPPED else (meta.width, meta.height)


def field_of_view(focal_35mm: float, width: int, height: int) -> tuple[float, float]:
    """(horizontal, vertical) field of view in degrees: f_px = focal_35mm / 43.27 x image diagonal."""
    diag = math.hypot(width, height)
    f_px = focal_35mm / FULL_FRAME_DIAGONAL_MM * diag
    return 2.0 * math.degrees(math.atan(width / (2.0 * f_px))), 2.0 * math.degrees(math.atan(height / (2.0 * f_px)))


def angular_spans(points: np.ndarray, camera: np.ndarray) -> tuple[float, float]:
    """(az_span, el_span) in degrees of the points seen from camera (E, N, U): the smallest arc that holds every
    azimuth, and the spread of the elevations."""
    v = np.asarray(points, float) - np.asarray(camera, float)[None]
    if not len(v):
        return 0.0, 0.0
    r = np.hypot(v[:, 0], v[:, 1])
    az = np.sort(np.mod(np.degrees(np.arctan2(v[:, 0], v[:, 1])), 360.0))
    el = np.degrees(np.arctan2(v[:, 2], np.maximum(r, 1e-6)))
    gaps = np.diff(np.concatenate([az, [az[0] + 360.0]]))
    return float(360.0 - gaps.max()), float(el.max() - el.min())


def outline_distance(subject: Subject, e: float, n: float) -> float:
    """Horizontal distance (m) from (e, n) to the main outline; 0 inside it (a courtyard is outside)."""
    xy = np.array([[e, n]], float)
    if in_rings(xy, list(subject.rings))[0]:
        return 0.0
    return float(distance_to_rings(xy, list(subject.rings))[0][0])


def assess(subject: Subject, meta: PhotoMeta, settings: Settings) -> Qualification:
    """The qualification of a photo with GPS (the module docstring)."""
    e, n, _ = LocalENU(subject.lat, subject.lon).to_enu(meta.lat, meta.lon)
    distance = outline_distance(subject, e, n)
    f35 = meta.focal_35mm
    default = f35 is None or not settings.subject_min_focal_35mm <= f35 <= settings.subject_max_focal_35mm
    focal = settings.subject_default_focal_35mm if default else float(f35)
    hfov, vfov = field_of_view(focal, *upright_size(meta))
    az_span, el_span = angular_spans(subject.points, np.array([e, n, settings.camera_height_m]))
    share = max(min(1.0, az_span / hfov), min(1.0, el_span / vfov))
    qualifies = distance <= settings.subject_max_distance_m or share >= settings.subject_min_share
    return Qualification(distance, share, focal, default, qualifies)


def sample_points(triangles: np.ndarray, step: float = SAMPLE_STEP_M, grid: float = THIN_GRID_M) -> np.ndarray:
    """Points along every edge of (k, 3, 3) triangles, about `step` apart (vertices included), one per `grid` cell."""
    tris = np.asarray(triangles, float).reshape(-1, 3, 3)
    pts = [tris.reshape(-1, 3)]
    for i, j in ((0, 1), (1, 2), (2, 0)):
        a, b = tris[:, i], tris[:, j]
        count = np.clip(np.ceil(np.linalg.norm(b - a, axis=1) / step).astype(int), 1, 400)
        for k in np.unique(count):
            if k <= 1:
                continue
            sel = count == k
            t = (np.arange(1, k) / k)[None, :, None]
            pts.append((a[sel][:, None] + t * (b[sel] - a[sel])[:, None]).reshape(-1, 3))
    p = np.concatenate(pts)
    if not len(p):
        return np.zeros((0, 3))
    return np.unique(np.round(p / grid).astype(np.int64), axis=0) * grid


def build_subject(slug: str, osm_parts: Path, parts_sha256: str, outline: Path, lat: float, lon: float) -> Subject:
    """The subject geometry from the OSM parts snapshot and the main outline (structure.load_osm_structure)."""
    from . import structure as S        # here, not at the top: the parser and OpenCV load only when a building is parsed

    model = S.load_osm_structure(osm_parts, outline, lat, lon, slug)
    tris = [t for t in (b.triangles() for b in model.built) if len(t)]
    points = sample_points(np.concatenate(tris)) if tris else np.zeros((0, 3))
    rings = tuple(np.asarray(r, float) for r in model.main_rings)
    return Subject(slug, float(lat), float(lon), rings, points.astype(np.float32), parts_sha256)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cached(settings: Settings, lm: Landmark, slug: str, entry: dict[str, Any]) -> Subject:
    root = Path(settings.structure_catalog).resolve().parent.parent   # paths in the catalogue are repo-relative
    sha = str(entry["osm_parts"]["sha256"])
    lat, lon = float(entry["lat"]), float(entry["lon"])
    outline = root / entry["osm"]["file"]
    stamp = {"version": SUBJECT_VERSION, "slug": slug, "parts_sha256": sha, "outline_sha256": _sha256(outline),
             "lat": lat, "lon": lon}
    key = tuple(stamp.values())
    with _guard:
        lock = _key_locks.setdefault(key, threading.Lock())
    with lock:                          # one load per building, however many uploads arrive at once
        hit = _memory.get(key)
        if hit is not None:
            return hit
        path = settings.maps_dir / lm.id / f"subject-{sha[:12]}.npz"
        subject = _read(path, stamp)
        if subject is None:
            source = find_source(slug, lat, lon, Path(settings.structure_catalog))   # checks the parts sha256
            if source is None:
                raise ValueError(f"{slug}: the catalogue lists no snapshot or outline")
            subject = build_subject(slug, source.osm_parts, source.osm_parts_sha256, source.outline, lat, lon)
            _write(path, subject, stamp)
        _memory[key] = subject
        return subject


def _read(path: Path, stamp: dict[str, Any]) -> Subject | None:
    """The cached subject, when the file is there and was made from the same inputs (stamp)."""
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if json.loads(str(z["stamp"])) != stamp:
                return None
            rings = tuple(np.asarray(z[f"ring{i}"], float) for i in range(int(z["rings"])))
            points = np.asarray(z["points"], np.float32)
    except Exception:   # a damaged file (truncated zip, wrong arrays) is rebuilt, never a failed upload
        log.warning("subject cache %s is unreadable; rebuilding it", path, exc_info=True)
        return None
    return Subject(stamp["slug"], stamp["lat"], stamp["lon"], rings, points, stamp["parts_sha256"])


def _write(path: Path, subject: Subject, stamp: dict[str, Any]) -> None:
    """Write the cache file (atomically); a failure only costs the next process a rebuild."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as f:
            np.savez_compressed(f, stamp=np.array(json.dumps(stamp, sort_keys=True)), points=subject.points,
                                rings=np.array(len(subject.rings)),
                                **{f"ring{i}": r for i, r in enumerate(subject.rings)})
        os.replace(tmp, path)
    except OSError:
        log.warning("cannot write the subject cache %s", path, exc_info=True)
        tmp.unlink(missing_ok=True)
