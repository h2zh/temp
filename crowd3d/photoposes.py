"""Where each photo was taken and which way it looked: the pose the benchmark's photo_pose_* metrics score.

Per photo of a landmark, the best available source wins (docs/BENCHMARK.md 7.7):

  sfm   registered by SfM in a published model, camera not flagged: exactly the pose api.camera_estimates
        shows (the newest published model containing the photo, a full model beats a facade model), in the
        landmark ENU frame; moved with that model's footprint snap when crowd3d.stylize accepted one.
  exif  the photo's own EXIF GPS position and EXIF compass heading (GPSImgDirection, as tagged).
  gps   EXIF GPS without a compass heading: the position is known, the direction is not. NOT determined:
        the camera is never assumed to face the building.
A rejected upload, or an accepted one without any position, gets no source (src None) and a reason. (The product
has no visual localizer, so there is no "loc" source here; the photo_pose_loc metric counts 0.)

Uncertainty, sp (metres) and sy (degrees): 2 sigma of the rules below, calibrated on the commons-ten-v1 and mdx-two-v1
runs (in-sample) so that each of the two covers about 85-90 % of the measured errors of its source. Both errors inside at
once is rarer, and the errors have heavy tails (wrong geotags, compasses ~90 or ~180 deg off) no circle of a useful size
covers.
  sfm   sigma_pos = max(15 m, alignment RMSE, d_centre * sin|GPS yaw - compass yaw|) / 2 (the display circle
        of VISUAL_LOCALIZATION 3.11), sigma_yaw = max(sqrt(yaw_sigma^2 + 1), |GPS yaw - compass yaw| / 2,
        10 deg). The georegistration's own yaw_sigma (3-5 deg) is NOT enough on its own: real models were
        15-21 deg off. A model whose footprint snap was accepted: sigma <= 3 m / 3 deg.
        A model the georegistration could NOT fit to GPS (meta georegistered False: yaw from the compass or the
        landmark direction alone, scale from _fallback_scale, which runs 1.5-2.4x large on real buildings,
        position from the median GPS offset of a handful of fixes) is as rough as a phone's own GPS: the display
        circle is at least hypot(30 m, 0.6 x the camera's distance from the model's median camera) and the pose is
        marked approx. mdx-two-v1 Aachen's only published model is one: its 4 cameras sat 23.6-24.8 m from the
        reference inside a 15 m circle before this rule. Raw EXIF GPS on commons-ten-v1 lies within 30 m of the
        reference for 72 % of the 527 reference photos; the median of several fixes does better, the scale term
        covers the guessed scale. Ten such cameras have a reference (in-sample, small).
  exif  sigma_pos = max(20 m for a true-north ('T') heading tag, 30 m otherwise, the phone's GPSHPositioningError);
        sigma_yaw = 15 deg for a 'T' tag, 30 deg otherwise (magnetic 'M' tags were often ~90 deg off: 47 % within 15 deg
        against 85 %). The position: raw EXIF GPS of the 180 heading-tagged reference photos lies within 23 m ('T', p90)
        and 60 m ('M', p90) of the reference, but the photos drawn from EXIF are the ones SfM missed, which are worse
        ('T' p85 38-40 m): 2 x 20 m / 2 x 30 m covers 84-86 % of their positions (the old 2 x 12 m covered 73 %).
"""

from __future__ import annotations

import math
import re
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from .geo import LocalENU, normalize_deg
from .records import JOB_FULL, PHOTO_ACCEPTED, ModelRecord, PhotoRecord

SRC_SFM, SRC_EXIF, SRC_GPS = "sfm", "exif", "gps"
DETERMINED_SOURCES = (SRC_SFM, SRC_EXIF)   # position AND viewing direction known

# why a photo's pose (or its direction) is unknown
R_REJECTED = "rejected"                     # the upload failed validation: never placed
R_NO_GPS = "no_gps"                         # accepted without a position (only a future GPS-less path)
R_GPS_WITHOUT_HEADING = "gps_without_heading"  # src "gps": position only

# uncertainty rules (1 sigma; a pose carries COVERAGE_SIGMAS x sigma)
COVERAGE_SIGMAS = 2.0
SFM_POS_FLOOR_M = 15.0          # floor of the display circle R (sigma = R / 2)
SFM_HEADING_FLOOR_DEG = 10.0
SFM_RELATIVE_HEADING_DEG = 1.0  # SfM's own relative orientation error (< 1 deg against MegaDepth-X)
DEFAULT_RMSE_M = 6.0            # a model without an alignment RMSE (Params.gps_sigma_m)
DEFAULT_YAW_SIGMA_DEG = 15.0    # a model without a yaw sigma (Params.compass_sigma_deg)
SNAPPED_SIGMA_CAP = 3.0         # m and deg, a model whose footprint snap was accepted
PRIOR_ONLY_POS_FLOOR_M = 30.0   # display circle R of a model not georegistered from GPS (prior yaw, guessed scale) ...
PRIOR_ONLY_SCALE_ERROR = 0.6    # ... combined with this fraction of the camera's distance from the model's median
                                # camera (_fallback_scale runs 1.5-2.4x large: 1 - 1/2.4 = 0.58 of a drawn offset)
EXIF_GPS_SIGMA_TRUE_M = 20.0    # an EXIF position whose heading tag is true north ('T': phones, mostly iPhones) ...
EXIF_GPS_SIGMA_OTHER_M = 30.0   # ... magnetic ('M': cameras with a GPS unit) or untagged; also a position without a heading
EXIF_HEADING_SIGMA_TRUE_DEG = 15.0
EXIF_HEADING_SIGMA_OTHER_DEG = 30.0
DEFAULT_CAMERA_HEIGHT_M = 1.6

# the georegistration's message names the yaw conflict it fused (reconstruct.georegister)
_YAW_CONFLICT_RE = (re.compile(r"GPS yaw - compass yaw = (-?\d+(?:\.\d+)?) deg"),
                    re.compile(r"GPS yaw is (-?\d+(?:\.\d+)?) deg from the compass yaw"))


@dataclass
class PhotoPose:
    """One photo's pose as shown. src None: nothing known (rejected, or no position); src "gps": position only."""

    photo_id: str
    src: str | None
    reason: str | None = None                # R_* when the pose or its direction is unknown
    p: list[float] | None = None             # camera centre, landmark ENU metres
    yaw: float | None = None                 # compass bearing of the optical axis, degrees
    sp: float | None = None                  # position uncertainty, metres (2 sigma)
    sy: float | None = None                  # heading uncertainty, degrees (2 sigma)
    model_id: str | None = None              # sfm: the model the pose comes from
    snapped: bool = False                    # moved with an accepted footprint snap
    approx: bool = False                     # sfm from a model not georegistered from GPS and not snapped:
                                             # position as rough as phone GPS
    reasons: list[str] = field(default_factory=list)   # a rejected upload's validation codes

    @property
    def determined(self) -> bool:
        return self.src in DETERMINED_SOURCES and self.p is not None and self.yaw is not None


# ---------------------------------------------------------------- per-source helpers


def published_cameras(models: Iterable[ModelRecord]) -> dict[str, tuple[ModelRecord, dict[str, Any]]]:
    """photo_id -> (model, camera entry) as api.camera_estimates picks it: published models only, the newest one
    containing the photo, full models over facade models, flagged cameras never."""
    out: dict[str, tuple[ModelRecord, dict[str, Any]]] = {}
    published = [m for m in models if m.meta.get("published", True)]
    for m in sorted(published, key=lambda m: (m.kind == JOB_FULL, m.created_at, m.id)):
        for pid, cam in (m.meta.get("cameras") or {}).items():
            if isinstance(cam, dict) and not cam.get("flags") and _xyz(cam.get("enu")) is not None \
                    and _finite(cam.get("heading_deg")) is not None:
                out[pid] = (m, cam)
    return out


def yaw_conflict_deg(meta: Mapping[str, Any]) -> float | None:
    """GPS yaw - compass yaw the georegistration fused (its message), or None when it fused no compass."""
    msg = str(meta.get("message") or "")
    for rx in _YAW_CONFLICT_RE:
        m = rx.search(msg)
        if m:
            return float(m.group(1))
    return None


def prior_only(meta: Mapping[str, Any]) -> bool:
    """The georegistration could not fit this model to GPS: yaw from a prior, scale guessed, position from the median
    GPS offset (reconstruct.georegister's fallback)."""
    return meta.get("georegistered") is False


def median_camera_en(meta: Mapping[str, Any]) -> tuple[float, float] | None:
    """Median east / north of every camera of a model (flagged ones too: the fallback shift is their median)."""
    xy = [v for c in (meta.get("cameras") or {}).values() if isinstance(c, dict) and (v := _xyz(c.get("enu")))]
    if not xy:
        return None
    return statistics.median(v[0] for v in xy), statistics.median(v[1] for v in xy)


def sfm_position_rule(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The model-level inputs of the position uncertainty of any camera of a model with this meta, so it can be
    evaluated at any point (position_sigma_at): the circle floor, the sine of the fused yaw conflict and, for a
    prior-only model, its median camera."""
    conflict = abs(yaw_conflict_deg(meta) or 0.0)
    rmse = _finite(meta.get("alignment_rmse_m")) or DEFAULT_RMSE_M
    mc = median_camera_en(meta) if prior_only(meta) else None
    return {"floor_m": max(SFM_POS_FLOOR_M, rmse), "sin_conflict": math.sin(math.radians(min(conflict, 90.0))),
            "prior_only": prior_only(meta), "median_en": list(mc) if mc else None}


def position_sigma_at(rule: Mapping[str, Any], e: float, n: float) -> float:
    """1-sigma position (m) of a camera at (e, n) under an sfm_position_rule: half the display circle max(floor,
    distance from the centre x sin(yaw conflict)), for a prior-only model at least hypot(PRIOR_ONLY_POS_FLOOR_M,
    PRIOR_ONLY_SCALE_ERROR x the distance from its median camera)."""
    circle = max(float(rule["floor_m"]), math.hypot(e, n) * float(rule["sin_conflict"]))
    if rule.get("prior_only"):
        mc = rule.get("median_en")
        d_med = math.hypot(e - mc[0], n - mc[1]) if mc else 0.0
        circle = max(circle, math.hypot(PRIOR_ONLY_POS_FLOOR_M, PRIOR_ONLY_SCALE_ERROR * d_med))
    return circle / 2.0


def sfm_sigma(meta: Mapping[str, Any], enu: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """1-sigma (position m, heading deg) of a camera at enu in a model with this meta (calibrated rule; a prior-only
    model's circle is at least hypot(PRIOR_ONLY_POS_FLOOR_M, PRIOR_ONLY_SCALE_ERROR x the distance from its median
    camera))."""
    yaw = _finite(meta.get("yaw_sigma_deg"))
    yaw = DEFAULT_YAW_SIGMA_DEG if yaw is None else yaw
    conflict = abs(yaw_conflict_deg(meta) or 0.0)
    head = max(math.hypot(yaw, SFM_RELATIVE_HEADING_DEG), conflict / 2.0, SFM_HEADING_FLOOR_DEG)
    return position_sigma_at(sfm_position_rule(meta), float(enu[0]), float(enu[1])), head


def uncertainty_rule() -> dict[str, float]:
    """The constants of the uncertainty rules, recorded with every result that has photo_pose_* metrics."""
    return {"coverage_sigmas": COVERAGE_SIGMAS, "sfm_pos_floor_m": SFM_POS_FLOOR_M,
            "sfm_heading_floor_deg": SFM_HEADING_FLOOR_DEG, "sfm_relative_heading_deg": SFM_RELATIVE_HEADING_DEG,
            "default_rmse_m": DEFAULT_RMSE_M, "default_yaw_sigma_deg": DEFAULT_YAW_SIGMA_DEG,
            "snapped_sigma_cap": SNAPPED_SIGMA_CAP, "prior_only_pos_floor_m": PRIOR_ONLY_POS_FLOOR_M,
            "prior_only_scale_error": PRIOR_ONLY_SCALE_ERROR,
            "exif_gps_sigma_true_m": EXIF_GPS_SIGMA_TRUE_M, "exif_gps_sigma_other_m": EXIF_GPS_SIGMA_OTHER_M,
            "exif_heading_sigma_true_deg": EXIF_HEADING_SIGMA_TRUE_DEG,
            "exif_heading_sigma_other_deg": EXIF_HEADING_SIGMA_OTHER_DEG}


def exif_sigma(gps_accuracy_m: float | None, heading_ref: str | None) -> tuple[float, float]:
    """1-sigma (position m, heading deg) of an EXIF pose: by the heading tag ('T' true north, else magnetic or none),
    the position at least the phone's own GPSHPositioningError."""
    true_north = (heading_ref or "").strip().upper() == "T"
    base = EXIF_GPS_SIGMA_TRUE_M if true_north else EXIF_GPS_SIGMA_OTHER_M
    acc = _finite(gps_accuracy_m)
    pos = max(base, acc) if acc else base
    head = EXIF_HEADING_SIGMA_TRUE_DEG if true_north else EXIF_HEADING_SIGMA_OTHER_DEG
    return pos, head


def apply_snap(snap: Mapping[str, Any] | None, enu: list[float], heading_deg: float) -> tuple[list[float], float]:
    """A camera moved with a footprint snap (stylize.apply_snap_points: p' = s R p + t about the ENU origin, z
    scaled); its heading turns with the snap's counter-clockwise rotation (a compass bearing is clockwise)."""
    if not snap:
        return [float(v) for v in enu], float(heading_deg)
    a = math.radians(float(snap["rotation_deg"]))
    s = float(snap.get("scale", 1.0) or 1.0)
    t = snap.get("shift_m") or snap.get("shift") or (0.0, 0.0)
    e, n, u = (float(v) for v in enu)
    out = [s * (math.cos(a) * e - math.sin(a) * n) + float(t[0]), s * (math.sin(a) * e + math.cos(a) * n) + float(t[1]),
           s * u]
    return out, normalize_deg(float(heading_deg) - float(snap["rotation_deg"]))


def _sfm_pose(photo: PhotoRecord, m: ModelRecord, cam: dict[str, Any], snap: Mapping[str, Any] | None) -> PhotoPose:
    enu, heading = apply_snap(snap, cam["enu"], float(cam["heading_deg"]))
    sp, sy = sfm_sigma(m.meta, cam["enu"])
    if snap:
        sp, sy = min(sp, SNAPPED_SIGMA_CAP), min(sy, SNAPPED_SIGMA_CAP)
    return PhotoPose(photo.id, SRC_SFM, p=enu, yaw=heading, sp=COVERAGE_SIGMAS * sp, sy=COVERAGE_SIGMAS * sy,
                     model_id=m.id, snapped=bool(snap), approx=prior_only(m.meta) and not snap)


def photo_poses(photos: Iterable[PhotoRecord], models: Iterable[ModelRecord], lat0: float, lon0: float, *,
                snaps: Mapping[str, Any] | None = None, heading_refs: Mapping[str, str | None] | None = None,
                camera_height_m: float = DEFAULT_CAMERA_HEIGHT_M) -> list[PhotoPose]:
    """The pose of every photo (accepted or not) of one landmark, in the order given.

    snaps: model_id -> accepted footprint snap meta (FootprintSnap.to_meta; refused snaps must not be passed);
    heading_refs: photo_id -> GPSImgDirectionRef ("T" true north, "M" magnetic; the store does not keep it).
    """
    models = list(models)
    snaps = {k: v for k, v in (snaps or {}).items() if v and v.get("accepted", True)}
    pub = published_cameras(models)
    enu = LocalENU(lat0, lon0)
    out = []
    for ph in photos:
        if ph.status != PHOTO_ACCEPTED:
            out.append(PhotoPose(ph.id, None, R_REJECTED, reasons=list(ph.reasons)))
            continue
        if ph.id in pub:
            m, cam = pub[ph.id]
            out.append(_sfm_pose(ph, m, cam, snaps.get(m.id)))
            continue
        if ph.lat is None or ph.lon is None:
            out.append(PhotoPose(ph.id, None, R_NO_GPS))
            continue
        e, n, _ = enu.to_enu(ph.lat, ph.lon)
        p = [e, n, camera_height_m]
        if _finite(ph.heading_deg) is None:
            sp, _ = exif_sigma(ph.gps_accuracy_m, None)
            out.append(PhotoPose(ph.id, SRC_GPS, R_GPS_WITHOUT_HEADING, p=p, sp=COVERAGE_SIGMAS * sp))
            continue
        sp, sy = exif_sigma(ph.gps_accuracy_m, (heading_refs or {}).get(ph.id))
        out.append(PhotoPose(ph.id, SRC_EXIF, p=p, yaw=normalize_deg(float(ph.heading_deg)), sp=COVERAGE_SIGMAS * sp,
                             sy=COVERAGE_SIGMAS * sy))
    return out


# ---------------------------------------------------------------- small helpers


def _finite(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _xyz(v: Any) -> list[float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(_finite(x) is not None for x in v):
        return [float(x) for x in v]
    return None
