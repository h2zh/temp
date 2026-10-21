import random

import pytest

from crowd3d.config import Settings
from crowd3d.coverage import (
    SECTOR_FACE_READY,
    SECTOR_LOCKED,
    SECTOR_PARTIAL,
    compute_coverage,
    plan_jobs,
    sector_center_deg,
    sector_for_bearing,
    sector_label,
    select_face_photos,
)
from crowd3d.records import (
    JOB_DONE,
    JOB_FACE,
    JOB_FAILED,
    JOB_FULL,
    JOB_QUEUED,
    JOB_RUNNING,
    PHOTO_ACCEPTED,
    PHOTO_REJECTED,
    Job,
    PhotoRecord,
)

S = Settings(data_dir="unused")
_seq = iter(range(10**6))


def photo(bearing, *, status=PHOTO_ACCEPTED, t=None, pid=None, sector="auto"):
    n = next(_seq)
    return PhotoRecord(
        id=pid or f"p{n:05d}",
        landmark_id="lm",
        user_id="u",
        status=status,
        uploaded_at=float(n) if t is None else t,
        bearing_deg=bearing,
        sector=sector_for_bearing(bearing, S.num_sectors) if sector == "auto" else sector,
    )


def in_sector(sector, count, spread=20.0):
    """count photos over the middle `spread` degrees of a sector, outside neighbouring face margins."""
    centre = sector_center_deg(sector, S.num_sectors)
    return [photo(centre - spread / 2 + spread * i / max(1, count - 1)) for i in range(count)]


def job(kind, sector, photos_or_n, status=JOB_DONE):
    if isinstance(photos_or_n, int):
        ids = [f"x{i}" for i in range(photos_or_n)]
    else:
        ids = [p.id for p in photos_or_n]
    return Job(id=f"j{next(_seq)}", landmark_id="lm", kind=kind, sector=sector, status=status,
               photo_ids=ids, created_at=0.0)


def test_sector_helpers():
    assert sector_for_bearing(0, 8) == 0
    assert sector_for_bearing(22.4, 8) == 0
    assert sector_for_bearing(22.5, 8) == 1
    assert sector_for_bearing(337.5, 8) == 0
    assert sector_for_bearing(337.4, 8) == 7
    assert sector_for_bearing(-10, 8) == 0
    assert sector_for_bearing(180, 8) == 4
    assert [sector_label(i, 8) for i in (0, 2, 7)] == ["N", "E", "NW"]
    assert sector_label(3, 4) == "W"
    assert sector_label(5, 6) == "S5"
    assert sector_center_deg(6, 8) == 270


def test_compute_coverage_states_and_progress():
    photos = in_sector(0, 4) + in_sector(2, 2) + in_sector(5, 6)
    photos += [photo(90, status=PHOTO_REJECTED), photo(90, sector=None), photo(91, sector=None)]
    cov = compute_coverage(photos, S)
    states = {s.index: (s.count, s.state) for s in cov.sectors}
    assert states[0] == (4, SECTOR_FACE_READY)
    assert states[2] == (2, SECTOR_PARTIAL)
    assert states[5] == (6, SECTOR_FACE_READY)
    assert states[1] == (0, SECTOR_LOCKED)
    assert cov.total == 12
    assert cov.face_ready == [0, 5]
    assert not cov.full_ready
    assert cov.progress == pytest.approx((1 + 0.5 + 1) / 8)
    assert [s.label for s in cov.sectors] == ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    assert cov.sectors[2].bearing_center_deg == 90


def test_compute_coverage_empty():
    cov = compute_coverage([], S)
    assert cov.total == 0 and cov.progress == 0 and cov.face_ready == [] and not cov.full_ready
    assert all(s.state == SECTOR_LOCKED for s in cov.sectors)


def test_full_ready_needs_sectors_and_photos():
    six_sectors = [p for s in range(6) for p in in_sector(s, 4)]
    assert compute_coverage(six_sectors, S).full_ready
    assert compute_coverage(six_sectors, Settings(data_dir="unused", full_min_photos=25)).full_ready is False
    five_sectors = [p for s in range(5) for p in in_sector(s, 6)]  # 30 photos but only 5 face-ready sectors
    assert not compute_coverage(five_sectors, S).full_ready
    assert compute_coverage(six_sectors + in_sector(7, 3), S).progress == pytest.approx((6 + 0.75) / 8)


def test_select_face_photos_margins():
    inside = photo(10)
    near_neighbour = photo(22.5 + 5)        # 5 deg into sector 1
    far_neighbour = photo(22.5 + 15)        # 15 deg into sector 1
    wrap_near = photo(337.5 - 5)            # 5 deg into sector 7, across north
    wrap_far = photo(337.5 - 15)
    rejected = photo(5, status=PHOTO_REJECTED)
    photos = [inside, near_neighbour, far_neighbour, wrap_near, wrap_far, rejected]
    got = {p.id for p in select_face_photos(photos, 0, S)}
    assert got == {inside.id, near_neighbour.id, wrap_near.id}
    # With no margin the selection is exactly the half-open sector.
    edge = photo(22.5)                       # belongs to sector 1
    start = photo(337.5)                     # belongs to sector 0
    no_margin = Settings(data_dir="unused", face_margin_deg=0)
    got = {p.id for p in select_face_photos(photos + [edge, start], 0, no_margin)}
    assert got == {inside.id, start.id}
    assert edge.id in {p.id for p in select_face_photos([edge], 1, no_margin)}


def test_select_face_photos_is_deterministic():
    photos = [photo(0, t=5.0, pid="b"), photo(3, t=5.0, pid="a"), photo(357, t=1.0, pid="z"),
              photo(8, t=9.0, pid="c")]
    for _ in range(5):
        random.shuffle(photos)
        assert [p.id for p in select_face_photos(photos, 0, S)] == ["z", "a", "b", "c"]


def test_plan_needs_face_min_photos():
    assert plan_jobs(in_sector(2, 3), [], S) == []
    photos = in_sector(2, 4)
    (req,) = plan_jobs(photos, [], S)
    assert (req.kind, req.sector) == (JOB_FACE, 2)
    assert req.photo_ids == [p.id for p in photos]


def test_plan_active_job_blocks_rebuild():
    photos = in_sector(0, 10)
    for status in (JOB_QUEUED, JOB_RUNNING):
        assert plan_jobs(photos, [job(JOB_FACE, 0, 4, status)], S) == []
    # A job for another sector or kind does not block this sector.
    (req,) = plan_jobs(photos, [job(JOB_FACE, 3, 4, JOB_RUNNING), job(JOB_FULL, None, 30, JOB_QUEUED)], S)
    assert req.sector == 0


@pytest.mark.parametrize("previous_status", [JOB_DONE, JOB_FAILED])
def test_plan_face_growth_threshold(previous_status):
    photos = in_sector(1, 4)
    jobs = [job(JOB_FACE, 1, photos, previous_status)]
    assert plan_jobs(photos, jobs, S) == []
    photos += in_sector(1, 2)
    assert plan_jobs(photos, jobs, S) == []          # 6 - 4 < face_rebuild_growth
    photos += in_sector(1, 1)
    (req,) = plan_jobs(photos, jobs, S)               # 7 - 4 == 3
    assert req.sector == 1 and len(req.photo_ids) == 7


def test_plan_growth_counts_from_largest_previous_job():
    photos = in_sector(4, 10)
    jobs = [job(JOB_FACE, 4, 4), job(JOB_FACE, 4, 8, JOB_FAILED), job(JOB_FACE, 4, 5)]
    assert plan_jobs(photos, jobs, S) == []          # 10 - 8 < 3
    (req,) = plan_jobs(photos + in_sector(4, 1), jobs, S)
    assert len(req.photo_ids) == 11


def test_plan_face_growth_includes_margin_photos():
    photos = in_sector(0, 4)
    jobs = [job(JOB_FACE, 0, photos)]
    # Three photos just across the boundary in sector 1: sector 1 is not ready yet,
    # but they widen sector 0's facade input enough to rebuild it.
    photos += [photo(22.5 + 2), photo(22.5 + 4), photo(22.5 + 6)]
    (req,) = plan_jobs(photos, jobs, S)
    assert (req.kind, req.sector, len(req.photo_ids)) == (JOB_FACE, 0, 7)


def test_plan_full_job():
    photos = [p for s in range(6) for p in in_sector(s, 4)]
    reqs = plan_jobs(photos, [], S)
    assert [(r.kind, r.sector) for r in reqs] == [(JOB_FACE, s) for s in range(6)] + [(JOB_FULL, None)]
    full = reqs[-1]
    assert full.photo_ids == [p.id for p in sorted(photos, key=lambda p: (p.uploaded_at, p.id))]
    # Rejected photos and photos without a sector are not part of any job.
    extra = [photo(0, status=PHOTO_REJECTED), photo(0, sector=None)]
    assert plan_jobs(photos + extra, [], S)[-1].photo_ids == full.photo_ids


def test_plan_full_growth_and_blocking():
    photos = [p for s in range(6) for p in in_sector(s, 4)]
    face_jobs = [job(JOB_FACE, s, 4) for s in range(6)]
    full_done = job(JOB_FULL, None, photos)
    assert plan_jobs(photos, face_jobs + [full_done], S) == []
    more = in_sector(6, 3) + in_sector(7, 3) + in_sector(0, 1)   # 7 new: not enough
    assert [r.kind for r in plan_jobs(photos + more, face_jobs + [full_done], S)] == []
    more += in_sector(7, 1)                                       # 8 new; sector 7 now face-ready too
    reqs = plan_jobs(photos + more, face_jobs + [full_done], S)
    assert [(r.kind, r.sector) for r in reqs] == [(JOB_FACE, 7), (JOB_FULL, None)]
    assert len(reqs[-1].photo_ids) == 32
    running_full = job(JOB_FULL, None, 10, JOB_RUNNING)
    assert [r.kind for r in plan_jobs(photos + more, face_jobs + [full_done, running_full], S)] == [JOB_FACE]
