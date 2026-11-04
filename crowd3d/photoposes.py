"""Where each photo was taken and which way it looked: the poses other people may see (D30) and the benchmark scores.

Per photo of a landmark, the best available source wins (docs/BENCHMARK.md 7.7). Every source is a measurement; a pose
the rules below cannot vouch for is withheld (kept on PhotoPose.withheld for the benchmark's withheld_right /
withheld_wrong), never shown as a rough guess:

  sfm   registered by SfM in a published model whose frame is fixed (frame_ok: its GeoAnchor level is "full", i.e. the
        georegistration fitted it to GPS, or crowd3d.stylize accepted a footprint snap that measured it against the OSM
        walls), camera not flagged: the newest such model containing the photo, a full model before a facade model, in
        the landmark ENU frame, moved with the model's accepted snap. Withheld: the camera of a model whose frame is not
        fixed (prior-only or not georegistered, no accepted snap: yaw from a compass or the landmark direction alone,
        scale from _fallback_scale, which runs 1.5-2.4x large on real buildings; mdx-two-v1 Aachen's 4 such cameras
        sat 23.6-24.8 m off), and a flagged camera (far from its own GPS, against its compass, odd focal length: of 34
        with a reference in F, 17 were right and 17 wrong, and nothing independent tells them apart). An older model's
        camera may be shown instead; the withheld one stays on record when it came from the preferred model.
        Pitch and roll come from the stored local pose (cameras[pid].local) through the model's GeoAnchor, the
        horizontal field of view from the camera's focal ratio and the processed copy's aspect.
  loc   visual localization into a published model (crowd3d.localize, a hook until the localizer lands): an accepted
        localization record {status, model_id, pose {enu, heading_deg, pitch_deg, roll_deg, hfov_deg, aspect}} into a
        published model whose frame is fixed, moved with that model's snap. Withheld when the model's frame is not
        fixed or the photo's own camera is flagged in a published model (localizing it again only repeats that pose).
  exif  the photo's own EXIF GPS position and EXIF compass heading, only when the heading is tagged true north
        (GPSImgDirectionRef "T"): "手机 GPS + 罗盘（未经模型核对）". The height is not measured (p[2] None); the
        field of view comes from the 35 mm focal length, else None. A magnetic or untagged heading is withheld
        (magnetic tags were often ~90 deg off: 47 % within 15 deg against 85 % for true north).
  gps   EXIF GPS without a usable heading (none, or not true north): the position is known, the direction is not.
        NOT determined and never shown: the camera is never assumed to face the building.
A rejected upload, or an accepted one without any position, gets no source (src None) and a reason.

Uncertainty, sp (metres) and sy (degrees): 2 sigma of the rules below, calibrated on the commons-ten-v1 and mdx-two-v1
runs (in-sample) so that each of the two covers about 85-90 % of the measured errors of its source. Both errors inside at
once is rarer, and the errors have heavy tails (wrong geotags, compasses ~90 or ~180 deg off) no circle of a useful size
covers.
  sfm   sigma_pos = max(15 m, alignment RMSE, d_centre * sin|GPS yaw - compass yaw|) / 2 (the display circle
        of VISUAL_LOCALIZATION 3.11), sigma_yaw = max(sqrt(yaw_sigma^2 + 1), |GPS yaw - compass yaw| / 2,
        10 deg). The georegistration's own yaw_sigma (3-5 deg) is NOT enough on its own: real models were
        15-21 deg off. A model whose footprint snap was accepted: sigma <= 3 m / 3 deg.
  loc   the sfm sigmas of its model at the localized position with the localizer's own relative error in quadrature:
        hypot(sfm, 1.5 m), hypot(sfm, 0.7 deg) (Baptistery relative error: median 1.56 m / 0.51 deg), so a localized
        pose never claims more than the model's own cameras.
  exif  sigma_pos = max(20 m, the phone's GPSHPositioningError) for a true-north tag (30 m for any other position);
        sigma_yaw = 15 deg. The position: raw EXIF GPS of the 180 heading-tagged reference photos lies within 23 m ('T',
        p90) and 60 m ('M', p90) of the reference, but the photos drawn from EXIF are the ones SfM missed, which are
        worse ('T' p85 38-40 m): 2 x 20 m covers 84-86 % of their positions.

What other people see (public_row, public_poses): only photos whose uploader agreed (records.location_shared) with a
determined pose, as rows of the whitelisted keys PUBLIC_KEYS: id = HMAC(salt, photo_id)[:12] (crowd3d.consent.public_id),
position rounded to 1 m, yaw / pitch / roll to 1 deg, hfov to 5 deg, aspect to 2 decimals, and sp / sy widened by the
rounding and rounded up, so rounding never narrows a range. No uploader, time, model, sector or store id. The benchmark
scores exactly these rows (scripts/eval_real_landmark.py photo_pose_records).
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .anchor import LEVELS, read_anchor
from .consent import public_id
from .geo import LocalENU, angle_diff_deg, normalize_deg
from .records import JOB_FULL, PHOTO_ACCEPTED, ModelRecord, PhotoRecord, location_shared

SRC_SFM, SRC_LOC, SRC_EXIF, SRC_GPS = "sfm", "loc", "exif", "gps"
SOURCES = (SRC_SFM, SRC_LOC, SRC_EXIF, SRC_GPS)
DETERMINED_SOURCES = (SRC_SFM, SRC_LOC, SRC_EXIF)   # position AND viewing direction known

# why a photo's pose (or its direction) is unknown
R_REJECTED = "rejected"                     # the upload failed validation: never placed
R_NO_GPS = "no_gps"                         # accepted without a position (only a future GPS-less path)
R_GPS_WITHOUT_HEADING = "gps_without_heading"  # src "gps": position only, no compass
R_HEADING_NOT_TRUE_NORTH = "heading_not_true_north"   # src "gps": a compass heading not tagged true north

# why a pose is withheld (PhotoPose.withheld[].reason)
W_FRAME_NOT_FIXED = "frame_not_fixed"       # its model's frame is not fixed (frame_ok False)
W_CAMERA_FLAGGED = "camera_flagged"         # the photo's SfM camera is flagged
W_HEADING_NOT_TRUE_NORTH = R_HEADING_NOT_TRUE_NORTH

LOC_ACCEPTED = "accepted"                   # the localization record status that gives a pose

# uncertainty rules (1 sigma; a pose carries COVERAGE_SIGMAS x sigma)
COVERAGE_SIGMAS = 2.0
SFM_POS_FLOOR_M = 15.0          # floor of the display circle R (sigma = R / 2)
SFM_HEADING_FLOOR_DEG = 10.0
SFM_RELATIVE_HEADING_DEG = 1.0  # SfM's own relative orientation error (< 1 deg against MegaDepth-X)
DEFAULT_RMSE_M = 6.0            # a model without an alignment RMSE (Params.gps_sigma_m)
DEFAULT_YAW_SIGMA_DEG = 15.0    # a model without a yaw sigma (Params.compass_sigma_deg)
SNAPPED_SIGMA_CAP = 3.0         # m and deg, a model whose footprint snap was accepted
LOC_REL_SIGMA_M = 1.5           # the localizer's own relative error, in quadrature with its model's sfm sigma
LOC_REL_SIGMA_DEG = 0.7
EXIF_GPS_SIGMA_TRUE_M = 20.0    # an EXIF position whose heading tag is true north ('T': phones, mostly iPhones) ...
EXIF_GPS_SIGMA_OTHER_M = 30.0   # ... any other position (magnetic 'M', untagged, no heading): src gps, never shown
EXIF_HEADING_SIGMA_TRUE_DEG = 15.0
EXIF_HEADING_SIGMA_OTHER_DEG = 30.0
FULL_FRAME_DIAGONAL_MM = 43.27  # the 35 mm equivalent focal length is defined on this diagonal
# a stored camera's local pose must give back its stored ENU pose through the model's anchor this closely, else its
# pitch and roll are not derived (an anchor that is not the one the camera was written with)
TILT_CHECK_M = 0.05
TILT_CHECK_DEG = 0.5

# what other people see (public_row)
PUBLIC_KEYS = ("id", "src", "p", "yaw", "pitch", "roll", "hfov", "aspect", "sp", "sy")
PUBLIC_POSITION_STEP_M = 1      # p rounded half up to whole metres
PUBLIC_ANGLE_STEP_DEG = 1       # yaw (mod 360), pitch and roll to whole degrees
PUBLIC_FOV_STEP_DEG = 5         # hfov to 5 deg steps (lens and zoom say little about the place; coarser fingerprint)
PUBLIC_ASPECT_DECIMALS = 2
PUBLIC_SP_ROUNDING_M = 0.71     # the largest horizontal shift of 1 m rounding, sqrt(0.5^2 + 0.5^2), in quadrature
PUBLIC_SY_ROUNDING_DEG = 0.5    # the largest shift of 1 deg rounding, added

# the georegistration's message names the yaw conflict it fused (reconstruct.georegister)
_YAW_CONFLICT_RE = (re.compile(r"GPS yaw - compass yaw = (-?\d+(?:\.\d+)?) deg"),
                    re.compile(r"GPS yaw is (-?\d+(?:\.\d+)?) deg from the compass yaw"))


@dataclass
class PhotoPose:
    """One photo's pose (internal: exact, keyed by store photo id; other people see public_row of it). src None:
    nothing known (rejected, or no position); src "gps": position only, never shown."""

    photo_id: str
    src: str | None
    reason: str | None = None                # R_* when the pose or its direction is unknown
    p: list[float | None] | None = None      # camera centre, landmark ENU metres; height None when not measured
    yaw: float | None = None                 # compass bearing of the optical axis, degrees
    pitch: float | None = None               # degrees, + up; None when not measured (EXIF)
    roll: float | None = None                # degrees, + when the image's right side is raised
    hfov: float | None = None                # horizontal field of view, degrees; None when not known
    aspect: float | None = None              # width / height of the processed copy (upright)
    sp: float | None = None                  # position uncertainty, metres (2 sigma)
    sy: float | None = None                  # heading uncertainty, degrees (2 sigma)
    model_id: str | None = None              # sfm / loc: the model the pose comes from
    snapped: bool = False                    # moved with an accepted footprint snap
    reasons: list[str] = field(default_factory=list)   # a rejected upload's validation codes
    # poses the rules did not show, best source first: {src, reason, model_id, p, yaw} (internal only)
    withheld: list[dict[str, Any]] = field(default_factory=list)

    @property
    def determined(self) -> bool:
        return self.src in DETERMINED_SOURCES and self.p is not None and self.yaw is not None


# ---------------------------------------------------------------- which model frames count


def anchor_level(meta: Mapping[str, Any]) -> str | None:
    """The georegistration level of a model (anchor.LEVELS: "full", "prior_only", "none"): its GeoAnchor's, else the
    legacy georegistered flag (True = "full"); None when neither says."""
    a = meta.get("anchor")
    if isinstance(a, Mapping) and a.get("level") in LEVELS:
        return str(a["level"])
    g = meta.get("georegistered")
    return "full" if g is True else "prior_only" if g is False else None


def snap_accepted(snap: Any) -> bool:
    """Whether a footprint snap meta (FootprintSnap.to_meta) was accepted: a refused snap (accepted False) moves and
    fixes nothing."""
    return bool(snap) and (not isinstance(snap, Mapping) or bool(snap.get("accepted", True)))


def accepted_snaps(snaps: Mapping[str, Any] | None) -> dict[str, Any]:
    """model_id -> snap meta, the accepted snaps only (snap_accepted)."""
    return {k: v for k, v in (snaps or {}).items() if snap_accepted(v)}


def frame_ok(model: ModelRecord, snaps: Mapping[str, Any]) -> bool:
    """Whether a model's frame is fixed well enough to show its cameras (and to localize into it): its anchor level is
    "full", or a footprint snap was accepted for it (snaps: model_id -> snap meta; a refused one does not count)."""
    return anchor_level(model.meta) == "full" or snap_accepted(snaps.get(model.id))


def _camera_ok(cam: Any) -> bool:
    return isinstance(cam, dict) and _xyz(cam.get("enu")) is not None and _finite(cam.get("heading_deg")) is not None


def _by_preference(models: Iterable[ModelRecord]) -> list[ModelRecord]:
    """Published models, least preferred first: facade before full, older before newer (the last one wins)."""
    return sorted((m for m in models if m.meta.get("published", True)), key=lambda m: (m.kind == JOB_FULL,
                                                                                       m.created_at, m.id))


def _sfm_candidates(models: Iterable[ModelRecord], snaps: Mapping[str, Any]
                    ) -> tuple[dict[str, tuple[int, ModelRecord, dict[str, Any]]],
                               dict[str, tuple[int, ModelRecord, dict[str, Any], str]]]:
    """(shown, withheld): photo_id -> (preference rank, model, camera entry[, reason]) of the preferred published SfM
    camera the rules show (frame fixed, not flagged), and of the preferred one they withhold (W_FRAME_NOT_FIXED or
    W_CAMERA_FLAGGED). A higher rank is preferred (_by_preference)."""
    shown: dict[str, tuple[int, ModelRecord, dict[str, Any]]] = {}
    held: dict[str, tuple[int, ModelRecord, dict[str, Any], str]] = {}
    for rank, m in enumerate(_by_preference(models)):
        ok = frame_ok(m, snaps)
        for pid, cam in (m.meta.get("cameras") or {}).items():
            if not _camera_ok(cam):
                continue
            if not ok:
                held[pid] = (rank, m, cam, W_FRAME_NOT_FIXED)
            elif cam.get("flags"):
                held[pid] = (rank, m, cam, W_CAMERA_FLAGGED)
            else:
                shown[pid] = (rank, m, cam)
    return shown, held


def published_cameras(models: Iterable[ModelRecord],
                      snaps: Mapping[str, Any] | None = None) -> dict[str, tuple[ModelRecord, dict[str, Any]]]:
    """photo_id -> (model, camera entry) of the SfM camera shown for it: published models whose frame is fixed
    (frame_ok), the newest one containing the photo, full models over facade models, flagged cameras never. Where
    every model's frame is fixed this is exactly api.camera_estimates (internal)."""
    return {pid: (m, cam) for pid, (_, m, cam) in _sfm_candidates(models, snaps or {})[0].items()}


def withheld_cameras(models: Iterable[ModelRecord], snaps: Mapping[str, Any] | None = None
                     ) -> dict[str, tuple[ModelRecord, dict[str, Any], str]]:
    """photo_id -> (model, camera entry, reason) of the preferred published SfM camera the rules do not show: flagged
    (W_CAMERA_FLAGGED) or in a model whose frame is not fixed (W_FRAME_NOT_FIXED). photo_poses records it when no
    camera is shown for the photo, or when it comes from a model preferred over the shown camera's (what the photo
    would have shown without the rule)."""
    return {pid: (m, cam, why) for pid, (_, m, cam, why) in _sfm_candidates(models, snaps or {})[1].items()}


def flagged_photos(models: Iterable[ModelRecord], snaps: Mapping[str, Any] | None = None) -> set[str]:
    """Photos with a flagged camera in a published model whose frame is fixed: never localized (sfm_flagged)."""
    snaps = snaps or {}
    return {pid for m in _by_preference(models) if frame_ok(m, snaps)
            for pid, cam in (m.meta.get("cameras") or {}).items() if isinstance(cam, dict) and cam.get("flags")}


# ---------------------------------------------------------------- uncertainty


def yaw_conflict_deg(meta: Mapping[str, Any]) -> float | None:
    """GPS yaw - compass yaw the georegistration fused (its message), or None when it fused no compass."""
    msg = str(meta.get("message") or "")
    for rx in _YAW_CONFLICT_RE:
        m = rx.search(msg)
        if m:
            return float(m.group(1))
    return None


def sfm_position_rule(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The model-level inputs of the position uncertainty of any camera of a model with this meta, so it can be
    evaluated at any point (position_sigma_at): the circle floor and the sine of the fused yaw conflict."""
    conflict = abs(yaw_conflict_deg(meta) or 0.0)
    rmse = _finite(meta.get("alignment_rmse_m")) or DEFAULT_RMSE_M
    return {"floor_m": max(SFM_POS_FLOOR_M, rmse), "sin_conflict": math.sin(math.radians(min(conflict, 90.0)))}


def position_sigma_at(rule: Mapping[str, Any], e: float, n: float) -> float:
    """1-sigma position (m) of a camera at (e, n) under an sfm_position_rule: half the display circle max(floor,
    distance from the centre x sin(yaw conflict))."""
    return max(float(rule["floor_m"]), math.hypot(e, n) * float(rule["sin_conflict"])) / 2.0


def sfm_sigma(meta: Mapping[str, Any], enu: list[float] | tuple[float, ...]) -> tuple[float, float]:
    """1-sigma (position m, heading deg) of a camera at enu in a model with this meta (calibrated rule)."""
    yaw = _finite(meta.get("yaw_sigma_deg"))
    yaw = DEFAULT_YAW_SIGMA_DEG if yaw is None else yaw
    conflict = abs(yaw_conflict_deg(meta) or 0.0)
    head = max(math.hypot(yaw, SFM_RELATIVE_HEADING_DEG), conflict / 2.0, SFM_HEADING_FLOOR_DEG)
    return position_sigma_at(sfm_position_rule(meta), float(enu[0]), float(enu[1])), head


def uncertainty_rule() -> dict[str, float]:
    """The constants of the uncertainty and public rounding rules, recorded with every result that has photo_pose_*
    metrics."""
    return {"coverage_sigmas": COVERAGE_SIGMAS, "sfm_pos_floor_m": SFM_POS_FLOOR_M,
            "sfm_heading_floor_deg": SFM_HEADING_FLOOR_DEG, "sfm_relative_heading_deg": SFM_RELATIVE_HEADING_DEG,
            "default_rmse_m": DEFAULT_RMSE_M, "default_yaw_sigma_deg": DEFAULT_YAW_SIGMA_DEG,
            "snapped_sigma_cap": SNAPPED_SIGMA_CAP, "loc_rel_sigma_m": LOC_REL_SIGMA_M,
            "loc_rel_sigma_deg": LOC_REL_SIGMA_DEG,
            "exif_gps_sigma_true_m": EXIF_GPS_SIGMA_TRUE_M, "exif_gps_sigma_other_m": EXIF_GPS_SIGMA_OTHER_M,
            "exif_heading_sigma_true_deg": EXIF_HEADING_SIGMA_TRUE_DEG,
            "public_position_step_m": PUBLIC_POSITION_STEP_M, "public_angle_step_deg": PUBLIC_ANGLE_STEP_DEG,
            "public_fov_step_deg": PUBLIC_FOV_STEP_DEG, "public_sp_rounding_m": PUBLIC_SP_ROUNDING_M,
            "public_sy_rounding_deg": PUBLIC_SY_ROUNDING_DEG}


# How often the public ranges held where the rules were not tuned: the commons-ten-v1 held-out pair (test split:
# Sagrada Familia, Hungarian Parliament, Lincoln Memorial, India Gate; Sydney Opera House has no verified reference
# pose), the D31 fresh runs (eval/results/commons-ten-v1/d31-test.json: two runs of the same uploads) rescored with
# these rules at d5e51fa (baseline A, metric schema 8). within / n: photos with a reference pose whose public row
# (public_row) had its position AND heading inside its own sp / sy, both runs counted (sfm 67/81 and 79/84, exif 12/15
# and 11/13). What the landmark page and the stylized viewer quote beside every row of that source, not the pooled
# in-sample figure (dev: sfm 362/421, exif 18/21). loc: not measured (no localizer yet). Re-pinned from the D30
# verification run.
MEASURED_COVERAGE: dict[str, dict[str, int] | None] = {
    SRC_SFM: {"within": 146, "n": 165, "runs": 2},
    SRC_LOC: None,
    SRC_EXIF: {"within": 23, "n": 28, "runs": 2},
}
COVERAGE_BASIS = ("commons-ten-v1 held-out pair (test split), D31 fresh runs rescored with the D30 rules (metric schema"
                  " 8), 2 runs of the same uploads")

# the public poses document (crowd3d.api GET /api/landmarks/{id}/poses)
POSES_FORMAT = "crowd3d.photo-poses"
POSES_VERSION = 2


def public_rounding() -> dict[str, Any]:
    """How public_row rounds: whole metres, whole degrees, 5-degree fields of view, 2-decimal aspects; sp and sy
    widened by the rounding and rounded up."""
    return {"position_m": PUBLIC_POSITION_STEP_M, "angle_deg": PUBLIC_ANGLE_STEP_DEG, "fov_deg": PUBLIC_FOV_STEP_DEG,
            "aspect_decimals": PUBLIC_ASPECT_DECIMALS, "sp_rounding_m": PUBLIC_SP_ROUNDING_M,
            "sy_rounding_deg": PUBLIC_SY_ROUNDING_DEG}


def public_uncertainty() -> dict[str, Any]:
    """What a row's sp / sy mean: COVERAGE_SIGMAS sigma, and per source the measured share of poses inside them
    (MEASURED_COVERAGE; None where not measured) with its basis."""
    measured = {src: None if c is None else {**c, "coverage": round(c["within"] / c["n"], 3)}
                for src, c in MEASURED_COVERAGE.items()}
    return {"sigmas": COVERAGE_SIGMAS, "measured_coverage": measured, "basis": COVERAGE_BASIS}


def true_north(heading_ref: str | None) -> bool:
    return (heading_ref or "").strip().upper() == "T"


def exif_sigma(gps_accuracy_m: float | None, heading_ref: str | None) -> tuple[float, float]:
    """1-sigma (position m, heading deg) of an EXIF position and heading: by the heading tag ('T' true north, else
    magnetic or none), the position at least the phone's own GPSHPositioningError. Only a true-north pose is shown."""
    tn = true_north(heading_ref)
    base = EXIF_GPS_SIGMA_TRUE_M if tn else EXIF_GPS_SIGMA_OTHER_M
    acc = _finite(gps_accuracy_m)
    pos = max(base, acc) if acc else base
    head = EXIF_HEADING_SIGMA_TRUE_DEG if tn else EXIF_HEADING_SIGMA_OTHER_DEG
    return pos, head


# ---------------------------------------------------------------- geometry


def apply_snap(snap: Mapping[str, Any] | None, enu: list[float], heading_deg: float) -> tuple[list[float], float]:
    """A camera moved with a footprint snap (stylize.apply_snap_points: p' = s R p + t about the ENU origin, z
    scaled); its heading turns with the snap's counter-clockwise rotation (a compass bearing is clockwise). Pitch and
    roll do not change (a turn about the vertical and a uniform scale)."""
    if not snap:
        return [float(v) for v in enu], float(heading_deg)
    a = math.radians(float(snap["rotation_deg"]))
    s = float(snap.get("scale", 1.0) or 1.0)
    t = snap.get("shift_m") or snap.get("shift") or (0.0, 0.0)
    e, n, u = (float(v) for v in enu)
    out = [s * (math.cos(a) * e - math.sin(a) * n) + float(t[0]), s * (math.sin(a) * e + math.cos(a) * n) + float(t[1]),
           s * u]
    return out, normalize_deg(float(heading_deg) - float(snap["rotation_deg"]))


def pitch_roll_deg(right: np.ndarray, forward: np.ndarray) -> tuple[float, float]:
    """Pitch of the optical axis and roll of the image (+ when its right side is raised), in degrees, from the
    camera's right and forward axes in ENU. Roll is 0 for a camera looking straight up or down."""
    f = np.asarray(forward, float) / np.linalg.norm(forward)
    r = np.asarray(right, float) / np.linalg.norm(right)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, float(f[2])))))
    level_right = np.cross(f, [0.0, 0.0, 1.0])
    if np.linalg.norm(level_right) < 1e-6:
        return pitch, 0.0
    level_right /= np.linalg.norm(level_right)
    level_up = np.cross(level_right, f)
    return pitch, math.degrees(math.atan2(float(r @ level_up), float(r @ level_right)))


def camera_pitch_roll(meta: Mapping[str, Any], cam: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Pitch and roll of a stored SfM camera from its local pose (cameras[pid].local {q, c}) through the model's
    GeoAnchor (the frame its enu and heading_deg were written in); (None, None) without them, or when the anchor does
    not give back the stored centre and heading (TILT_CHECK_M / TILT_CHECK_DEG)."""
    local = cam.get("local")
    if not isinstance(local, Mapping):
        return None, None
    q, c = local.get("q"), local.get("c")
    if not (isinstance(q, (list, tuple)) and len(q) == 4 and all(_finite(x) is not None for x in q)) \
            or _xyz(c) is None:
        return None, None
    try:
        anchor = read_anchor(dict(meta))
    except (KeyError, TypeError, ValueError):
        return None, None
    enu, heading = _xyz(cam.get("enu")), _finite(cam.get("heading_deg"))
    if anchor is None or enu is None or heading is None or not any(q):
        return None, None
    centre, cam_from_enu, h = anchor.camera_enu(list(q), list(c))
    if float(np.linalg.norm(np.asarray(centre) - np.asarray(enu))) > TILT_CHECK_M \
            or abs(angle_diff_deg(h, heading)) > TILT_CHECK_DEG:
        return None, None
    return pitch_roll_deg(cam_from_enu[0], cam_from_enu[2])


def hfov_from_focal_ratio(focal_ratio: float | None, aspect: float | None) -> float | None:
    """Horizontal FOV (deg) from SfM's focal length / longer image side and the image aspect (width / height)."""
    fr, a = _finite(focal_ratio), _finite(aspect)
    if not fr or fr <= 0 or not a or a <= 0:
        return None
    half_width = 0.5 if a >= 1.0 else 0.5 * a      # the horizontal side over the longer side, halved
    return math.degrees(2.0 * math.atan(half_width / fr))


def hfov_from_35mm(focal_35mm: float | None, aspect: float | None) -> float | None:
    """Horizontal FOV (deg) from the 35 mm equivalent focal length (diagonal equivalence, 43.27 mm) and the aspect."""
    f, a = _finite(focal_35mm), _finite(aspect)
    if not f or f <= 0 or not a or a <= 0:
        return None
    half_diag = math.atan(FULL_FRAME_DIAGONAL_MM / (2.0 * f))
    return math.degrees(2.0 * math.atan(math.tan(half_diag) * a / math.hypot(a, 1.0)))


def processed_aspects(photos: Iterable[PhotoRecord], data_dir: Path | str) -> dict[str, float]:
    """photo id -> width / height of its processed copy (upright: the orientation applied; what SfM saw). Reads only
    the JPEG header; a photo without a readable processed copy is left out."""
    from PIL import Image, UnidentifiedImageError

    out: dict[str, float] = {}
    for ph in photos:
        if not ph.processed_path:
            continue
        try:
            with Image.open(Path(data_dir) / ph.processed_path) as im:
                w, h = im.size
        except (OSError, UnidentifiedImageError, ValueError):
            continue
        if w > 0 and h > 0:
            out[ph.id] = w / h
    return out


# ---------------------------------------------------------------- per-source poses


def _sfm_pose(photo: PhotoRecord, m: ModelRecord, cam: dict[str, Any], snap: Mapping[str, Any] | None,
              aspect: float | None) -> PhotoPose:
    enu, heading = apply_snap(snap, cam["enu"], float(cam["heading_deg"]))
    sp, sy = sfm_sigma(m.meta, cam["enu"])
    if snap:
        sp, sy = min(sp, SNAPPED_SIGMA_CAP), min(sy, SNAPPED_SIGMA_CAP)
    pitch, roll = camera_pitch_roll(m.meta, cam)
    return PhotoPose(photo.id, SRC_SFM, p=enu, yaw=heading, pitch=pitch, roll=roll,
                     hfov=hfov_from_focal_ratio(cam.get("focal_ratio"), aspect), aspect=aspect,
                     sp=COVERAGE_SIGMAS * sp, sy=COVERAGE_SIGMAS * sy, model_id=m.id, snapped=bool(snap))


def _rec(rec: Any, key: str) -> Any:
    """A field of a localization record: a mapping (JSON) or an object with attributes (records.LocRecord)."""
    return rec.get(key) if isinstance(rec, Mapping) else getattr(rec, key, None)


def _withheld(src: str, reason: str, model_id: str | None, p: list[float | None], yaw: float) -> dict[str, Any]:
    return {"src": src, "reason": reason, "model_id": model_id, "p": p, "yaw": yaw}


def _loc_pose(photo: PhotoRecord, rec: Any, models: Mapping[str, ModelRecord], snaps: Mapping[str, Any],
              flagged: set[str], aspect: float | None) -> tuple[PhotoPose | None, dict[str, Any] | None]:
    """(the loc pose, or None; a withheld entry, or None) of an accepted localization record."""
    if _rec(rec, "status") != LOC_ACCEPTED:
        return None, None
    pose = _rec(rec, "pose")
    pose = pose if isinstance(pose, Mapping) else {}
    m = models.get(str(_rec(rec, "model_id")))
    enu, heading = _xyz(pose.get("enu")), _finite(pose.get("heading_deg"))
    if m is None or not m.meta.get("published", True) or enu is None or heading is None:
        return None, None
    if not frame_ok(m, snaps):
        return None, _withheld(SRC_LOC, W_FRAME_NOT_FIXED, m.id, enu, normalize_deg(heading))
    snap = snaps.get(m.id)
    p, yaw = apply_snap(snap, enu, heading)
    if photo.id in flagged:
        return None, _withheld(SRC_LOC, W_CAMERA_FLAGGED, m.id, p, yaw)
    sp, sy = sfm_sigma(m.meta, enu)
    if snap:
        sp, sy = min(sp, SNAPPED_SIGMA_CAP), min(sy, SNAPPED_SIGMA_CAP)
    aspect = _finite(pose.get("aspect")) or aspect
    hfov = _finite(pose.get("hfov_deg")) or hfov_from_35mm(photo.focal_35mm, aspect)
    return PhotoPose(photo.id, SRC_LOC, p=p, yaw=yaw, pitch=_finite(pose.get("pitch_deg")),
                     roll=_finite(pose.get("roll_deg")), hfov=hfov, aspect=aspect,
                     sp=COVERAGE_SIGMAS * math.hypot(sp, LOC_REL_SIGMA_M),
                     sy=COVERAGE_SIGMAS * math.hypot(sy, LOC_REL_SIGMA_DEG), model_id=m.id, snapped=bool(snap)), None


def photo_poses(photos: Iterable[PhotoRecord], models: Iterable[ModelRecord], lat0: float, lon0: float, *,
                snaps: Mapping[str, Any] | None = None, localizations: Mapping[str, Any] | None = None,
                heading_refs: Mapping[str, str | None] | None = None,
                aspects: Mapping[str, float] | None = None) -> list[PhotoPose]:
    """The internal pose of every photo (accepted or not, shared or not) of one landmark, in the order given.

    snaps: model_id -> accepted footprint snap meta (FootprintSnap.to_meta; refused snaps move and fix nothing);
    localizations: photo_id -> a localization record (records.LocRecord or its JSON) {status, model_id, pose {enu,
    heading_deg, pitch_deg, roll_deg, hfov_deg, aspect}}; heading_refs: photo_id -> GPSImgDirectionRef for photos
    stored before the store kept it (the record's own heading_ref wins); aspects: photo_id -> the processed copy's
    width / height (processed_aspects).
    """
    models = list(models)
    by_id = {m.id: m for m in models}
    snaps = accepted_snaps(snaps)
    shown, held = _sfm_candidates(models, snaps)
    flagged = flagged_photos(models, snaps)
    aspects = aspects or {}
    enu = LocalENU(lat0, lon0)
    out = []
    for ph in photos:
        if ph.status != PHOTO_ACCEPTED:
            out.append(PhotoPose(ph.id, None, R_REJECTED, reasons=list(ph.reasons)))
            continue
        aspect = _finite(aspects.get(ph.id))
        withheld: list[dict[str, Any]] = []
        if ph.id in held and (ph.id not in shown or held[ph.id][0] > shown[ph.id][0]):
            _, m, cam, why = held[ph.id]
            p, yaw = apply_snap(snaps.get(m.id), cam["enu"], float(cam["heading_deg"]))
            withheld.append(_withheld(SRC_SFM, why, m.id, p, yaw))
        if ph.id in shown:
            _, m, cam = shown[ph.id]
            pose = _sfm_pose(ph, m, cam, snaps.get(m.id), aspect)
            pose.withheld = withheld          # a preferred camera the rule did not show (an older one is shown)
            out.append(pose)
            continue
        rec = (localizations or {}).get(ph.id)
        if rec is not None:
            pose, w = _loc_pose(ph, rec, by_id, snaps, flagged, aspect)
            if w is not None:
                withheld.append(w)
            if pose is not None:
                pose.withheld = withheld
                out.append(pose)
                continue
        if ph.lat is None or ph.lon is None:
            out.append(PhotoPose(ph.id, None, R_NO_GPS, withheld=withheld))
            continue
        e, n, _ = enu.to_enu(ph.lat, ph.lon)
        p = [e, n, None]                     # the camera height is not measured
        heading = _finite(ph.heading_deg)
        ref = ph.heading_ref if ph.heading_ref is not None else (heading_refs or {}).get(ph.id)
        if heading is not None and true_north(ref):
            sp, sy = exif_sigma(ph.gps_accuracy_m, ref)
            out.append(PhotoPose(ph.id, SRC_EXIF, p=p, yaw=normalize_deg(heading),
                                 hfov=hfov_from_35mm(ph.focal_35mm, aspect), aspect=aspect, sp=COVERAGE_SIGMAS * sp,
                                 sy=COVERAGE_SIGMAS * sy, withheld=withheld))
            continue
        if heading is not None:
            withheld.append(_withheld(SRC_EXIF, W_HEADING_NOT_TRUE_NORTH, None, p, normalize_deg(heading)))
        sp, _ = exif_sigma(ph.gps_accuracy_m, None)
        out.append(PhotoPose(ph.id, SRC_GPS, R_GPS_WITHOUT_HEADING if heading is None else R_HEADING_NOT_TRUE_NORTH,
                             p=p, sp=COVERAGE_SIGMAS * sp, withheld=withheld))
    return out


# ---------------------------------------------------------------- what other people see


def _half_up(x: float, step: float = 1.0) -> int:
    """x rounded half up to a multiple of step (12.5 -> 13, -12.5 -> -12): the same on every platform."""
    return int(math.floor(float(x) / step + 0.5) * step)


def public_row(pose: PhotoPose, salt: str) -> dict[str, Any] | None:
    """The row other people see for a determined pose (None for any other): PUBLIC_KEYS only, rounded (module
    docstring). The caller decides who may be seen (records.location_shared); the salt is the deployment's secret."""
    if not pose.determined:
        return None
    e, n, u = pose.p
    return {
        "id": public_id(pose.photo_id, salt),
        "src": pose.src,
        "p": [_half_up(e), _half_up(n), None if u is None else _half_up(u)],
        "yaw": _half_up(pose.yaw) % 360,
        "pitch": None if pose.pitch is None else _half_up(pose.pitch),
        "roll": None if pose.roll is None else (_half_up(pose.roll) + 179) % 360 - 179,     # -179 .. 180
        "hfov": None if pose.hfov is None else _half_up(pose.hfov, PUBLIC_FOV_STEP_DEG),
        "aspect": None if pose.aspect is None else round(float(pose.aspect), PUBLIC_ASPECT_DECIMALS),
        "sp": None if pose.sp is None else int(math.ceil(math.hypot(pose.sp, PUBLIC_SP_ROUNDING_M))),
        "sy": None if pose.sy is None else int(math.ceil(pose.sy + PUBLIC_SY_ROUNDING_DEG)),
    }


def public_poses(photos: Iterable[PhotoRecord], models: Iterable[ModelRecord], lat0: float, lon0: float, *,
                 salt: str, snaps: Mapping[str, Any] | None = None, localizations: Mapping[str, Any] | None = None,
                 aspects: Mapping[str, float] | None = None) -> dict[str, Any]:
    """What other people may see of one landmark's photos: {"counts", "photos"}. photos: public_row of every photo
    whose uploader agreed (records.location_shared) and whose pose is determined, sorted by public id (no upload
    order); counts: accepted photos, shared, shown, per source, and not_determined = shared - shown."""
    photos = list(photos)
    shared = [p for p in photos if location_shared(p)]
    poses = photo_poses(shared, models, lat0, lon0, snaps=snaps, localizations=localizations, aspects=aspects)
    rows = sorted((r for p in poses if (r := public_row(p, salt)) is not None), key=lambda r: r["id"])
    counts = {"photos": sum(p.status == PHOTO_ACCEPTED for p in photos), "shared": len(shared), "shown": len(rows),
              **{s: sum(r["src"] == s for r in rows) for s in DETERMINED_SOURCES},
              "not_determined": len(shared) - len(rows)}
    return {"counts": counts, "photos": rows}


# ---------------------------------------------------------------- small helpers


def _finite(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _xyz(v: Any) -> list[float] | None:
    if isinstance(v, (list, tuple)) and len(v) == 3 and all(_finite(x) is not None for x in v):
        return [float(x) for x in v]
    return None
