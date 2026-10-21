"""Quality gate a reconstruction must pass before it replaces the published model (ARCHITECTURE 3.8).

Everything here is ground-truth free: the model is compared only with what the
product knows anyway (the photos' GPS and compass, the landmark centre and
radius). Thresholds were set on real-photo runs (Florence Baptistry, Aachen
Cathedral), where every grossly wrong model failed at least one check.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import Settings
from .coverage import compute_coverage, photo_sector
from .geo import angle_diff_deg, heading_from_en
from .reconstruct import MIN_REGISTERED, ReconstructionResult
from .records import JOB_FULL, Landmark, PhotoRecord

FLAG_BEYOND_RADIUS = "beyond_radius"      # estimated outside the landmark's check-in radius
FLAG_FACING_AWAY = "facing_away"          # no compass, and the estimate looks away from the landmark
MAX_FACING_OFFSET_DEG = 90.0
FULL_SECTOR_MIN_REGISTERED = 2            # registered photos that make a face-ready sector "covered"


@dataclass
class QAReport:
    passed: bool
    reasons: list[str] = field(default_factory=list)     # failed checks, human readable
    checks: dict[str, Any] = field(default_factory=dict)  # measured values

    def to_meta(self) -> dict[str, Any]:
        return {"passed": self.passed, "reasons": list(self.reasons), "checks": dict(self.checks)}


def assess(
    result: ReconstructionResult,
    kind: str,
    photos: Sequence[PhotoRecord],
    landmark: Landmark,
    settings: Settings,
) -> QAReport:
    """Check an ok result of a job over `photos`; adds QA flags to result.cameras in place.

    A camera with any flag (the reconstructor's own camera_flags, FLAG_BEYOND_RADIUS,
    FLAG_FACING_AWAY) is not shown as an estimate. The model fails when:
    - fewer than MIN_REGISTERED or a smaller fraction than qa_min_registered_fraction
      (qa_full_min_registered_fraction for the full model) of the photos registered;
    - it has fewer than qa_min_points points, or a mean reprojection error above
      qa_max_reprojection_error_px;
    - more than qa_max_twin_fraction of its cameras disagree with both their GPS
      and their compass (twin facades of a symmetric building), or more than
      qa_max_flagged_fraction carry any flag;
    - the median disagreement with the compass exceeds qa_max_compass_error_deg;
    - its points' horizontal median lies more than half the radius from the centre;
    - (full model) its cameras cover fewer face-ready sectors than the full gate requires.
    """
    by_id = {p.id: p for p in photos}
    cams = result.cameras
    for pid, cam in cams.items():
        flags = cam.setdefault("flags", [])
        e, n = float(cam["enu"][0]), float(cam["enu"][1])
        if math.hypot(e, n) > landmark.radius_m and FLAG_BEYOND_RADIUS not in flags:
            flags.append(FLAG_BEYOND_RADIUS)
        photo = by_id.get(pid)
        heading = cam.get("heading_deg")
        if (photo is not None and photo.heading_deg is None and heading is not None and math.hypot(e, n) > 1.0
                and abs(angle_diff_deg(heading, heading_from_en(-e, -n))) > MAX_FACING_OFFSET_DEG
                and FLAG_FACING_AWAY not in flags):
            flags.append(FLAG_FACING_AWAY)

    reasons: list[str] = []
    n_input = max(result.num_input, 1)
    n_reg = result.num_registered
    min_fraction = (settings.qa_full_min_registered_fraction if kind == JOB_FULL
                    else settings.qa_min_registered_fraction)
    if n_reg < MIN_REGISTERED or n_reg / n_input < min_fraction:
        reasons.append(f"only {n_reg} of {result.num_input} photos registered (need {min_fraction:.0%})")
    if result.num_points < settings.qa_min_points:
        reasons.append(f"only {result.num_points} points (need {settings.qa_min_points})")
    reproj = result.mean_reprojection_error_px
    if reproj is not None and math.isfinite(reproj) and reproj > settings.qa_max_reprojection_error_px:
        reasons.append(f"mean reprojection error {reproj:.2f} px")

    n_cams = max(len(cams), 1)
    twins = sum("gps_and_compass" in c["flags"] for c in cams.values())
    flagged = sum(bool(c["flags"]) for c in cams.values())
    if twins > settings.qa_max_twin_fraction * n_cams:
        reasons.append(f"{twins} of {len(cams)} cameras disagree with both their GPS and compass"
                       " (likely look-alike facades)")
    if flagged > settings.qa_max_flagged_fraction * n_cams:
        reasons.append(f"{flagged} of {len(cams)} cameras are implausible (far from their GPS, outside the"
                       " radius, facing away or degenerate)")
    offsets = sorted(abs(c["compass_offset_deg"]) for c in cams.values() if c.get("compass_offset_deg") is not None)
    compass_error = offsets[len(offsets) // 2] if offsets else None
    if compass_error is not None and compass_error > settings.qa_max_compass_error_deg:
        reasons.append(f"headings are {compass_error:.0f} deg off the compass (median)")
    centre_offset = None
    if result.points_median_en is not None:
        centre_offset = math.hypot(*result.points_median_en[:2])
        if centre_offset > landmark.radius_m / 2:
            reasons.append(f"the point cloud sits {centre_offset:.0f} m from the landmark centre")

    covered = required = None
    if kind == JOB_FULL:
        coverage = compute_coverage(photos, settings)
        registered = [by_id[pid] for pid, c in cams.items() if pid in by_id and not c["flags"]]
        per_sector = [0] * settings.num_sectors
        for p in registered:
            sector = photo_sector(p, settings.num_sectors)
            if sector is not None:
                per_sector[sector] += 1
        covered = sum(per_sector[s] >= FULL_SECTOR_MIN_REGISTERED for s in coverage.face_ready)
        required = coverage.full_min_sectors
        if covered < required:
            reasons.append(f"the model covers {covered} face-ready sides, the full model needs {required}")

    checks = {
        "registered_fraction": round(n_reg / n_input, 3),
        "num_points": result.num_points,
        "twin_cameras": twins,
        "flagged_cameras": flagged,
        "compass_error_median_deg": None if compass_error is None else round(compass_error, 2),
        "points_centre_offset_m": None if centre_offset is None else round(centre_offset, 1),
        "covered_sectors": covered,
        "required_sectors": required,
    }
    return QAReport(passed=not reasons, reasons=reasons, checks=checks)
