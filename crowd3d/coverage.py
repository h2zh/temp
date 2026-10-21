"""Which sides of a landmark are covered, and which reconstruction jobs to run next.

Sector convention: the ring around the landmark centre is split into
Settings.num_sectors equal sectors. Sector 0 is centred on north (bearing 0)
and indices increase clockwise. A photo belongs to the sector containing the
bearing from the landmark centre to the camera, i.e. the side the
photographer stands on, which is the facade they see.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from .config import Settings
from .geo import angle_diff_deg, normalize_deg
from .records import ACTIVE_JOB_STATUSES, JOB_FACE, JOB_FULL, PHOTO_ACCEPTED, Job, PhotoRecord

SECTOR_LOCKED = "locked"          # no photos yet
SECTOR_PARTIAL = "partial"        # some photos, fewer than face_min_photos
SECTOR_FACE_READY = "face_ready"  # enough photos for a facade model

_LABELS = {
    8: ("N", "NE", "E", "SE", "S", "SW", "W", "NW"),
    4: ("N", "E", "S", "W"),
}


def sector_for_bearing(bearing_deg: float, num_sectors: int) -> int:
    width = 360.0 / num_sectors
    return int(normalize_deg(bearing_deg + width / 2) // width) % num_sectors


def sector_center_deg(index: int, num_sectors: int) -> float:
    return normalize_deg(index * 360.0 / num_sectors)


def sector_label(index: int, num_sectors: int) -> str:
    """8 sectors -> N, NE, E, SE, S, SW, W, NW; 4 -> N, E, S, W; otherwise S0, S1, ..."""
    labels = _LABELS.get(num_sectors)
    return labels[index] if labels else f"S{index}"


@dataclass
class SectorState:
    index: int
    label: str
    count: int
    state: str
    bearing_center_deg: float


@dataclass
class Coverage:
    sectors: list[SectorState]
    total: int                 # accepted photos with a sector
    face_ready: list[int]      # sector indices in SECTOR_FACE_READY
    full_ready: bool           # len(face_ready) >= full_min_sectors and total >= full_min_photos
    progress: float            # mean over sectors of min(1, count / face_min_photos), in [0, 1]
    full_min_sectors: int = 0  # face-ready sectors the full model needs here (full_sectors_required)


@dataclass
class JobRequest:
    kind: str                  # records.JOB_FACE | records.JOB_FULL
    sector: int | None
    photo_ids: list[str]


def full_sectors_required(counts: Sequence[int], settings: Settings) -> int:
    """Face-ready sectors needed for the full model.

    settings.full_min_sectors (K for a landmark that can be photographed from
    every side). Once a landmark has full_reachability_min_photos photos, the
    sides that still have none are taken as unreachable (enclosed, across a
    river) or simply not where crowds stand, and with A sectors that have
    photos K = max(ceil(0.75 A), A - 2), at least 2 (ARCHITECTURE 3.1).
    """
    required = settings.full_min_sectors
    if sum(counts) >= settings.full_reachability_min_photos:
        reachable = sum(c > 0 for c in counts)
        required = min(required, max(math.ceil(0.75 * reachable), reachable - 2, 2))
    return required


def photo_sector(photo: PhotoRecord, num_sectors: int) -> int | None:
    """The photo's sector under the current num_sectors: recomputed from its bearing, which the stored
    index (computed with the num_sectors in force at upload) may not match. None: no sector."""
    if photo.sector is None:
        return None
    if photo.bearing_deg is not None:
        return sector_for_bearing(photo.bearing_deg, num_sectors)
    return photo.sector if 0 <= photo.sector < num_sectors else None


def compute_coverage(photos: Sequence[PhotoRecord], settings: Settings) -> Coverage:
    """Only accepted photos with a sector count."""
    n = settings.num_sectors
    counts = [0] * n
    for p in _usable(photos, n):
        counts[photo_sector(p, n)] += 1
    sectors = []
    for i, c in enumerate(counts):
        if c == 0:
            state = SECTOR_LOCKED
        elif c < settings.face_min_photos:
            state = SECTOR_PARTIAL
        else:
            state = SECTOR_FACE_READY
        sectors.append(SectorState(i, sector_label(i, n), c, state, sector_center_deg(i, n)))
    face_ready = [s.index for s in sectors if s.state == SECTOR_FACE_READY]
    total = sum(counts)
    required = full_sectors_required(counts, settings)
    return Coverage(
        sectors=sectors,
        total=total,
        face_ready=face_ready,
        full_ready=len(face_ready) >= required and total >= settings.full_min_photos,
        progress=sum(min(1.0, c / settings.face_min_photos) for c in counts) / n,
        full_min_sectors=required,
    )


def select_face_photos(photos: Sequence[PhotoRecord], sector: int, settings: Settings) -> list[PhotoRecord]:
    """Accepted photos whose bearing lies within the sector widened by face_margin_deg on each side.

    Sorted by uploaded_at then id so job inputs are deterministic.
    """
    centre = sector_center_deg(sector, settings.num_sectors)
    reach = 180.0 / settings.num_sectors + settings.face_margin_deg

    def wanted(p: PhotoRecord) -> bool:
        # The sector test keeps margin 0 exactly equal to the half-open sector itself.
        if photo_sector(p, settings.num_sectors) == sector:
            return True
        return p.bearing_deg is not None and abs(angle_diff_deg(p.bearing_deg, centre)) < reach

    return _sorted(p for p in _usable(photos, settings.num_sectors) if wanted(p))


def plan_jobs(photos: Sequence[PhotoRecord], jobs: Sequence[Job], settings: Settings) -> list[JobRequest]:
    """Pure decision function, called after every accepted upload.

    Face job for sector s when: s is face-ready, there is no queued/running face
    job for s, and either no face job for s was ever attempted or
    len(select_face_photos(s)) - (largest photo count among previous face jobs
    for s, any status) >= face_rebuild_growth.

    Full job when: coverage.full_ready (full_sectors_required face-ready
    sectors and full_min_photos photos), no queued/running full job, and either
    none was attempted or total - (largest photo count among previous full
    jobs) >= full_rebuild_growth. Full jobs use all accepted photos with a sector.

    Failed attempts count as attempts, so a failing input is only retried after
    the growth threshold is reached again.
    """
    coverage = compute_coverage(photos, settings)
    requests = []
    for s in coverage.face_ready:
        selected = select_face_photos(photos, s, settings)
        if _due(jobs, JOB_FACE, s, len(selected), settings.face_rebuild_growth):
            requests.append(JobRequest(JOB_FACE, s, [p.id for p in selected]))
    if coverage.full_ready:
        selected = _sorted(_usable(photos, settings.num_sectors))
        if _due(jobs, JOB_FULL, None, len(selected), settings.full_rebuild_growth):
            requests.append(JobRequest(JOB_FULL, None, [p.id for p in selected]))
    return requests


def _usable(photos: Iterable[PhotoRecord], num_sectors: int) -> list[PhotoRecord]:
    return [p for p in photos if p.status == PHOTO_ACCEPTED and photo_sector(p, num_sectors) is not None]


def _sorted(photos: Iterable[PhotoRecord]) -> list[PhotoRecord]:
    return sorted(photos, key=lambda p: (p.uploaded_at, p.id))


def _due(jobs: Sequence[Job], kind: str, sector: int | None, count: int, growth: int) -> bool:
    previous = [j for j in jobs if j.kind == kind and j.sector == sector]
    if any(j.status in ACTIVE_JOB_STATUSES for j in previous):
        return False
    return not previous or count - max(len(j.photo_ids) for j in previous) >= growth
