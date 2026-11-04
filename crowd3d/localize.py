"""Visual localization of consented photos no published model shows (D30): the pure part.

A photo whose uploader agreed to share its location (records.location_shared) and that no published model shows as an
SfM camera can still be placed precisely: by 2-D/3-D PnP of its processed copy against a saved map of the landmark
(crowd3d.landmark_map: a published model's sparse reconstruction, its features in the landmark's feature cache). Display
only: the outcome is one records.LocRecord per photo, which photoposes ranks between sfm and exif (source "loc"); it
never reaches plan_jobs or a ReconPhoto, so no reconstruction changes. The rules (docs/VISUAL_LOCALIZATION.md section
3, narrowed by the D30 design) are the pure functions below; the COLMAP part (features, matching, the PnP solves) hands
them Solve objects and the correspondence counts.

Which photos (eligible): consented and accepted, with the EXIF 35 mm focal length (FocalLengthIn35mmFilm) and a
processed copy, no SfM camera shown for it (photoposes.published_cameras), its camera not flagged in a published model
whose frame is fixed (photoposes.flagged_photos: localizing it again would only repeat that pose), not registered in a
candidate map. A photo without an EXIF focal length is never tried: PnP with a free focal length was right 7 times and
wrong 9 times against the reference (up to 88 m / 19 deg off), a crude pose D30 rules out. A photo is tried again only
when the landmark's candidate maps change (maps_key); the new outcome replaces the old, so a shown pose is always in the
frame of a map the landmark shows now.

Which maps (candidate_maps): the saved map of the newest published model of each target (kind, sector) whose frame is
fixed (photoposes.frame_ok: GeoAnchor level "full", or an accepted footprint snap; a prior-only anchor takes its yaw
from a compass or the landmark direction and guesses its scale), with features of the current version (otherwise its
keypoint indices do not line up with the cache's) and a GeoAnchor to put the pose on the map.

Pairs (select_pairs): the query against every distinct photo of the candidate maps (one pair per map photo, whichever
maps hold it), at most Params.loc_max_pairs, the ones nearest to the query's GPS first when capped (on the probe, the
10 GPS-nearest images kept only 5-60 % of the correspondences, so the cap is generous).

A pose (pose_stats, check_pose; per map): the camera solved with its focal length FIXED at the EXIF prior
f = f35 / 43.27 x hypot(w, h) of the processed copy (reconstruct's focal prior). A second solve with a free focal length
only checks that prior; it is never the pose. Every gate must pass (VISUAL_LOCALIZATION 3.4-3.8):
  - at least Params.loc_min_inliers (30) distinct correspondences and as many PnP inliers;
  - the free-focal check solved, its focal / longer side within [loc_min_focal_ratio, loc_max_focal_ratio] (0.3-8, as
    the prior's), and within Params.loc_focal_mismatch (20 %) of the prior; else focal_mismatch (a cropped photo whose
    EXIF focal no longer fits: 4 of the 5 Frauenkirche mismatches checked against the reference were wrong);
  - inlier ratio >= loc_min_inlier_ratio (0.75); mean inlier reprojection error <= loc_max_reproj_frac x longer side;
  - the symmetric-alias (wrong-face) check (3.8): a second solve on the correspondences the best one left out; when it
    lands more than loc_second_m away (local distance x the anchor's scale) or loc_second_deg rotated, its inliers must
    stay within loc_second_ratio (0.15) of the best's. On the octagonal Baptistery every wrong-face pose had a ratio of
    0.18 or more and every right one 0.12 or less; with the inlier ratio (wrong faces at most 0.67-0.68, right ones 0.72
    and up) each check alone stopped every wrong face, one building only, so both stay on;
  - plausible: within the landmark radius, height within [loc_min_height_m, loc_max_height_m], optical axis within
    loc_max_look_deg of the model's median point.
Across maps (choose, 3.9): the accepting map with the most inliers (a full model on a tie); another accepting map with
at least loc_across_half of its inliers and a heading more than loc_across_deg away makes it ambiguous_across_models.

Frame (to_enu): the solved pose goes through the model's GeoAnchor (meta["anchor"], the frame the model's SfM cameras
were written in), before any footprint snap; photoposes moves it with the model's accepted snap and gives it the
model's sfm sigma at that point with the localizer's own relative error in quadrature (hypot(sfm, 1.5 m),
hypot(sfm, 0.7 deg)), so a localized pose never claims more than the model's own cameras.

GPS (gps_check): recorded, not a gate while Params.loc_withhold_gps_conflict is off. The record keeps the distance to
the photo's own GPS fix, the offset from its compass and reconstruct.camera_flags as for an SfM camera; gps_conflict
when those flag it (far_from_gps, gps_and_compass), which the benchmark reports as loc_with_gps_conflict. On the probe,
trusted-focal poses up to 89 m from their own fix were right (the Commons geotags were 24-84 m off), so the conflict
alone is no evidence against a pose; if the benchmark finds a wrong one among them, the flag withholds them as it
withholds flagged SfM cameras.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import params
from .anchor import GeoAnchor, quat_xyzw, read_anchor
from .geo import LocalENU, angle_diff_deg
from .photoposes import (
    FULL_FRAME_DIAGONAL_MM,
    TILT_CHECK_DEG,
    TILT_CHECK_M,
    accepted_snaps,
    flagged_photos,
    frame_ok,
    pitch_roll_deg,
    published_cameras,
)
from .reconstruct import camera_flags, gps_inlier_threshold_m
from .records import (
    JOB_FULL,
    LOC_ACCEPTED,
    LOC_REJECTED,
    LOC_SKIPPED,
    LocRecord,
    ModelRecord,
    PhotoRecord,
    location_shared,
)

# The rules of this module; part of maps_key, so a change of them tries every photo again.
LOC_VERSION = 1

# why a consented photo without an SfM pose is not tried (LocRecord status skipped)
S_SFM_FLAGGED = "sfm_flagged"          # its SfM camera is flagged in a published model whose frame is fixed
S_IN_MAP = "registered_in_map"         # registered in a candidate map, its camera there not shown
S_NO_FOCAL = "no_focal"                # no EXIF 35 mm focal length: a free-focal pose is never shown
S_NO_IMAGE = "no_image"                # no processed copy
S_NO_MAPS = "no_maps"                  # the landmark has no candidate map
# why a tried photo got no pose (status rejected); per map: PoseStats.reason
R_TOO_FEW_MATCHES = "too_few_matches"
R_PNP_FAILED = "pnp_failed"
R_FREE_FOCAL = "free_focal"            # the main solve did not keep the EXIF focal length: never a free-focal pose
R_FOCAL_UNCHECKED = "focal_unchecked"  # the free-focal check did not solve: the EXIF focal length cannot be checked
R_FOCAL_MISMATCH = "focal_mismatch"
R_IMPLAUSIBLE_FOCAL = "implausible_focal"
R_INLIERS = "inliers"
R_INLIER_RATIO = "inlier_ratio"
R_REPROJECTION = "reprojection"
R_AMBIGUOUS_IN_MODEL = "ambiguous_in_model"   # the symmetric-alias (wrong-face) check
R_IMPLAUSIBLE_POSITION = "implausible_position"
R_IMPLAUSIBLE_HEIGHT = "implausible_height"
R_LOOKS_AWAY = "looks_away"
R_GPS_CONFLICT = "gps_conflict"        # only with Params.loc_withhold_gps_conflict
R_AMBIGUOUS_ACROSS = "ambiguous_across_models"
# why a published model's map is not a candidate (candidate_maps)
M_NO_MAP = "no_saved_map"
M_FRAME_NOT_FIXED = "frame_not_fixed"
M_NO_ANCHOR = "no_anchor"
M_FEATURE_VERSION = "feature_version"
M_NO_IMAGES = "no_images"

GPS_CONFLICT_FLAGS = frozenset({"far_from_gps", "gps_and_compass"})   # reconstruct.camera_flags about the GPS fix
FOCAL_FIXED_TOL = 1e-6      # relative: the main solve's focal length must be the EXIF prior


# ---------------------------------------------------------------- inputs


@dataclass(frozen=True)
class MapRef:
    """A candidate map: the saved map of a published model (candidate_maps). The COLMAP part loads its sparse model."""

    model_id: str
    kind: str
    sector: int | None
    created_at: float
    anchor: GeoAnchor                       # the model's: local frame -> landmark ENU, before any footprint snap
    images: dict[str, str]                  # image name in the saved map -> photo id (map.json)
    camera_en: dict[str, tuple[float, float]] = field(default_factory=dict)   # photo id -> its SfM camera east/north
    snapped: bool = False                   # its frame is fixed by an accepted footprint snap, not the georegistration

    @property
    def photo_ids(self) -> set[str]:
        return set(self.images.values())


@dataclass(frozen=True)
class Query:
    """A photo to localize: its processed copy's size (upright, what SfM reads), EXIF 35 mm focal length, and its own
    GPS fix and compass for the record's GPS check."""

    photo_id: str
    width: int
    height: int
    focal_35mm: float
    gps_en: tuple[float, float] | None = None     # its EXIF GPS in the landmark ENU frame
    gps_accuracy_m: float | None = None
    heading_deg: float | None = None              # its EXIF compass (any tag), as reconstruct's camera check reads it

    @property
    def long_side(self) -> int:
        return max(self.width, self.height)

    @property
    def aspect(self) -> float:
        return self.width / self.height

    @property
    def focal_px(self) -> float:
        """The focal length the pose is solved with."""
        return prior_focal_px(self.focal_35mm, self.width, self.height)

    @property
    def hfov_deg(self) -> float:
        """Horizontal field of view at that focal length (= photoposes.hfov_from_35mm)."""
        return math.degrees(2.0 * math.atan(self.width / (2.0 * self.focal_px)))


@dataclass(frozen=True, eq=False)
class Solve:
    """One absolute-pose solution in a map's local frame (the COLMAP part's output)."""

    rotation: np.ndarray        # cam_from_local, 3x3
    centre: np.ndarray          # camera centre, local frame
    inliers: int
    reproj_px: float            # mean reprojection error of the inliers, pixels
    focal_px: float             # the solve's focal length (the fixed prior, or the free estimate)


@dataclass
class PoseStats:
    """A query's pose in one candidate map, with what the gates read (pose_stats) and their verdict (reason)."""

    model_id: str
    kind: str
    num_corr: int                         # distinct 2-D/3-D correspondences (query keypoint, map 3-D point)
    long_side_px: int
    focal_prior_px: float                 # the EXIF prior
    focal_px: float | None = None         # the main solve's: must be the prior (None: it did not solve)
    focal_free_px: float | None = None    # the free-focal check's (None: it did not solve)
    inliers: int = 0
    reproj_px: float | None = None
    second_inliers: int = 0               # the second-best solve's, when it lies far from the best (else 0)
    enu: list[float] | None = None        # the camera centre in the model's anchored ENU frame (no snap)
    heading_deg: float | None = None
    pitch_deg: float | None = None
    roll_deg: float | None = None
    look_deg: float | None = None         # optical axis to the direction of the model's median point
    gps_residual_m: float | None = None   # horizontal distance to the photo's own GPS fix
    compass_offset_deg: float | None = None
    flags: list[str] = field(default_factory=list)     # reconstruct.camera_flags (GPS and compass)
    reason: str | None = None             # check_pose: None when every gate passed

    @property
    def inlier_ratio(self) -> float:
        return self.inliers / self.num_corr if self.num_corr else 0.0

    @property
    def second_ratio(self) -> float:
        return self.second_inliers / self.inliers if self.inliers else 0.0

    @property
    def gps_conflict(self) -> bool:
        return bool(GPS_CONFLICT_FLAGS & set(self.flags))

    def summary(self) -> dict[str, Any]:
        """What a record keeps of a map it was tried against (no pose)."""
        return {"model_id": self.model_id, "reason": self.reason, "num_corr": self.num_corr, "inliers": self.inliers,
                "inlier_ratio": round(self.inlier_ratio, 3), "second_ratio": round(self.second_ratio, 3),
                "reproj_px": _round(self.reproj_px, 2),
                "focal_check": None if self.focal_free_px is None else round(self.focal_free_px / self.focal_prior_px,
                                                                             3)}

    def to_meta(self) -> dict[str, Any]:
        """The chosen pose's diagnostics (internal: exact, never served)."""
        return {**self.summary(), "kind": self.kind, "second_inliers": self.second_inliers,
                "focal_prior_px": round(self.focal_prior_px, 2), "focal_free_px": _round(self.focal_free_px, 2),
                "long_side_px": self.long_side_px, "look_deg": _round(self.look_deg, 1),
                "gps_residual_m": _round(self.gps_residual_m, 2),
                "compass_offset_deg": _round(self.compass_offset_deg, 2), "flags": list(self.flags),
                "gps_conflict": self.gps_conflict}


def prior_focal_px(focal_35mm: float, width: int, height: int) -> float:
    """The EXIF focal prior in pixels of a width x height image (reconstruct's: f35 / 43.27 x the diagonal)."""
    return float(focal_35mm) / FULL_FRAME_DIAGONAL_MM * math.hypot(width, height)


def make_query(photo: PhotoRecord, width: int, height: int, lat0: float, lon0: float) -> Query:
    """The Query of a photo whose processed copy is width x height. Raises ValueError without a positive size or an
    EXIF focal length."""
    f35 = _finite(photo.focal_35mm)
    if not f35 or f35 <= 0 or int(width) <= 0 or int(height) <= 0:
        raise ValueError(f"photo {photo.id}: no EXIF focal length or no image size")
    gps = None
    if photo.lat is not None and photo.lon is not None:
        e, n, _ = LocalENU(lat0, lon0).to_enu(photo.lat, photo.lon)
        gps = (float(e), float(n))
    return Query(photo.id, int(width), int(height), f35, gps, _finite(photo.gps_accuracy_m),
                 _finite(photo.heading_deg))


# ---------------------------------------------------------------- maps and photos


def latest_models(models: Iterable[ModelRecord]) -> list[ModelRecord]:
    """The newest published model of each target (kind, sector), as Store.latest_models: full first, then by sector."""
    latest: dict[tuple[str, int | None], ModelRecord] = {}
    for m in sorted(models, key=lambda m: (m.created_at, m.id)):
        if m.meta.get("published", True):
            latest[(m.kind, m.sector)] = m
    return sorted(latest.values(), key=lambda m: (m.sector is not None, m.sector or 0))


def candidate_maps(models: Iterable[ModelRecord], docs: Mapping[str, Mapping[str, Any] | None],
                   snaps: Mapping[str, Any] | None, feature_version: str
                   ) -> tuple[list[MapRef], dict[str, str]]:
    """(the candidate maps, model_id -> why not) of the landmark's latest published models. docs: model_id -> its
    saved map's map.json (landmark_map.load_map; None when there is none); snaps: model_id -> footprint snap meta (a
    refused one fixes nothing); feature_version: the version the localizer's features have
    (landmark_map.job_feature_version)."""
    snaps = accepted_snaps(snaps)
    maps: list[MapRef] = []
    skipped: dict[str, str] = {}
    for m in latest_models(models):
        doc = docs.get(m.id)
        if not isinstance(doc, Mapping):
            skipped[m.id] = M_NO_MAP
            continue
        if not frame_ok(m, snaps):
            skipped[m.id] = M_FRAME_NOT_FIXED
            continue
        try:
            anchor = read_anchor(dict(m.meta))
        except (KeyError, TypeError, ValueError):
            anchor = None
        if anchor is None:
            skipped[m.id] = M_NO_ANCHOR
            continue
        if doc.get("feature_version") != feature_version:
            skipped[m.id] = M_FEATURE_VERSION
            continue
        images = {str(k): str(v) for k, v in (doc.get("images") or {}).items()}
        if not images:
            skipped[m.id] = M_NO_IMAGES
            continue
        camera_en = {}
        for pid, cam in (m.meta.get("cameras") or {}).items():
            enu = cam.get("enu") if isinstance(cam, Mapping) else None
            if isinstance(enu, (list, tuple)) and len(enu) == 3 and all(_finite(v) is not None for v in enu):
                camera_en[str(pid)] = (float(enu[0]), float(enu[1]))
        maps.append(MapRef(m.id, m.kind, m.sector, float(m.created_at), anchor, images, camera_en,
                           snapped=anchor.level != "full"))
    return maps, skipped


def maps_key(maps: Iterable[MapRef | str]) -> str:
    """The key of a set of candidate maps (or their model ids), whatever their order: a photo is tried again only when
    its record's key differs. LOC_VERSION is part of it."""
    ids = sorted({m if isinstance(m, str) else m.model_id for m in maps})
    doc = json.dumps({"loc_version": LOC_VERSION, "maps": ids}, sort_keys=True)
    return hashlib.sha256(doc.encode()).hexdigest()[:32]


def eligible(photos: Iterable[PhotoRecord], models: Iterable[ModelRecord], maps: Sequence[MapRef],
             records: Mapping[str, Any] | None = None, snaps: Mapping[str, Any] | None = None
             ) -> tuple[list[str], dict[str, str]]:
    """(photo ids to try, photo id -> why skipped) among the photos whose location may be shown (location_shared) and
    that no published model shows as an SfM camera; any other photo is in neither (never localized, no record).
    records: photo_id -> its stored localization (records.LocRecord or its JSON): a photo whose record was made
    against these maps (maps_key) is in neither too. snaps: model_id -> footprint snap meta."""
    models = list(models)
    snaps = accepted_snaps(snaps)
    shown = published_cameras(models, snaps)
    flagged = flagged_photos(models, snaps)
    in_maps = {pid for m in maps for pid in m.images.values()}
    key = maps_key(maps)
    records = records or {}
    to_try: list[str] = []
    skipped: dict[str, str] = {}
    for ph in photos:
        if not location_shared(ph) or ph.id in shown:
            continue
        rec = records.get(ph.id)
        if rec is not None and _field(rec, "maps_key") == key:
            continue
        f35 = _finite(ph.focal_35mm)
        if ph.id in flagged:
            skipped[ph.id] = S_SFM_FLAGGED
        elif ph.id in in_maps:
            skipped[ph.id] = S_IN_MAP
        elif not f35 or f35 <= 0:
            skipped[ph.id] = S_NO_FOCAL
        elif not ph.processed_path:
            skipped[ph.id] = S_NO_IMAGE
        elif not maps:
            skipped[ph.id] = S_NO_MAPS
        else:
            to_try.append(ph.id)
    return to_try, skipped


def select_pairs(query: Query, maps: Sequence[MapRef], max_pairs: int | None = None) -> tuple[list[str], bool]:
    """(the map photos to match the query against, sorted; whether the cap cut any): every distinct photo of the
    candidate maps but the query itself, at most max_pairs (default Params.loc_max_pairs), the ones whose SfM camera
    lies nearest to the query's GPS fix first when capped (without a fix or a camera: last, by photo id)."""
    cap = params.current().loc_max_pairs if max_pairs is None else int(max_pairs)
    ids = sorted({pid for m in maps for pid in m.images.values()} - {query.photo_id})
    if len(ids) <= cap:
        return ids, False

    def distance(pid: str) -> float:
        if query.gps_en is None:
            return math.inf
        gx, gy = query.gps_en
        ds = [math.hypot(en[0] - gx, en[1] - gy) for m in maps if (en := m.camera_en.get(pid)) is not None]
        return min(ds) if ds else math.inf

    return sorted(sorted(ids, key=lambda pid: (distance(pid), pid))[:max(cap, 0)]), True


def map_frame_ok(anchor: GeoAnchor, map_poses: Mapping[str, tuple[Sequence[float], Sequence[float]]],
                 cameras: Mapping[str, Any]) -> bool:
    """Whether a saved map is in the frame of the model's anchor: every map image whose photo has an SfM camera in the
    model's meta (cameras[pid] {enu, heading_deg}) gives back that camera through the anchor, within
    photoposes.TILT_CHECK_M / TILT_CHECK_DEG; at least one such photo. map_poses: photo id -> the map image's
    (cam_from_local quaternion xyzw, centre)."""
    checked = 0
    for pid, (q, c) in map_poses.items():
        cam = cameras.get(pid)
        if not isinstance(cam, Mapping):
            continue
        enu, heading = cam.get("enu"), _finite(cam.get("heading_deg"))
        if not (isinstance(enu, (list, tuple)) and len(enu) == 3) or heading is None:
            continue
        centre, _, h = anchor.camera_enu(list(q), list(c))
        if float(np.linalg.norm(np.asarray(centre) - np.asarray(enu, float))) > TILT_CHECK_M \
                or abs(angle_diff_deg(h, heading)) > TILT_CHECK_DEG:
            return False
        checked += 1
    return checked > 0


# ---------------------------------------------------------------- one pose


def to_enu(anchor: GeoAnchor, rotation: np.ndarray, centre: np.ndarray
           ) -> tuple[list[float], float, float, float, np.ndarray]:
    """A local pose (cam_from_local rotation, centre) through the anchor: (centre ENU, heading, pitch, roll,
    cam_from_enu), as photoposes derives an SfM camera's (anchor.camera_enu, photoposes.pitch_roll_deg)."""
    c_enu, cam_from_enu, heading = anchor.camera_enu(quat_xyzw(rotation), list(np.asarray(centre, float)))
    pitch, roll = pitch_roll_deg(cam_from_enu[0], cam_from_enu[2])
    return [float(v) for v in c_enu], float(heading), float(pitch), float(roll), cam_from_enu


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """The angle of the rotation between two rotation matrices."""
    c = (float(np.trace(np.asarray(a, float) @ np.asarray(b, float).T)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def second_is_far(best: Solve, second: Solve, scale: float) -> bool:
    """Whether the second-best solve is another pose (3.8): more than Params.loc_second_m away (local distance x the
    anchor's scale) or loc_second_deg rotated."""
    p = params.current()
    d = float(scale) * float(np.linalg.norm(np.asarray(second.centre, float) - np.asarray(best.centre, float)))
    return d > p.loc_second_m or rotation_angle_deg(second.rotation, best.rotation) > p.loc_second_deg


def look_angle_deg(forward: np.ndarray, centre: Sequence[float], target: Sequence[float]) -> float:
    """Angle between the optical axis and the direction from the camera to a point (0 when they coincide)."""
    d = np.asarray(target, float) - np.asarray(centre, float)
    f = np.asarray(forward, float)
    nd, nf = float(np.linalg.norm(d)), float(np.linalg.norm(f))
    if nd < 1e-9 or nf < 1e-9:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(d @ f) / (nd * nf)))))


def gps_check(query: Query, enu: Sequence[float], heading_deg: float) -> tuple[float | None, float | None, list[str]]:
    """(distance to the photo's own GPS fix, offset from its compass, reconstruct.camera_flags) of a pose, as for an
    SfM camera; (None, offset, []) without a fix."""
    offset = None if query.heading_deg is None else angle_diff_deg(heading_deg, query.heading_deg)
    if query.gps_en is None:
        return None, offset, []
    resid = math.hypot(float(enu[0]) - query.gps_en[0], float(enu[1]) - query.gps_en[1])
    return resid, offset, camera_flags(resid, gps_inlier_threshold_m(query.gps_accuracy_m), offset, None)


def pose_stats(m: MapRef, q: Query, num_corr: int, fixed: Solve | None, free: Solve | None, second: Solve | None,
               target_enu: Sequence[float], radius_m: float) -> PoseStats:
    """The query's pose in map m and the gates' verdict (reason). fixed: the main solve, focal length fixed at the
    EXIF prior (the only one that can give the pose); free: the free-focal check; second: the second-best solve on the
    correspondences fixed left out; target_enu: the model's median 3-D point in its anchored ENU frame."""
    s = PoseStats(m.model_id, m.kind, int(num_corr), q.long_side, q.focal_px,
                  focal_free_px=None if free is None else float(free.focal_px))
    if fixed is not None:
        s.focal_px, s.inliers, s.reproj_px = float(fixed.focal_px), int(fixed.inliers), float(fixed.reproj_px)
        if second is not None and second_is_far(fixed, second, m.anchor.scale):
            s.second_inliers = int(second.inliers)
        s.enu, s.heading_deg, s.pitch_deg, s.roll_deg, cam_from_enu = to_enu(m.anchor, fixed.rotation, fixed.centre)
        s.look_deg = look_angle_deg(cam_from_enu[2], s.enu, target_enu)
        s.gps_residual_m, s.compass_offset_deg, s.flags = gps_check(q, s.enu, s.heading_deg)
    s.reason = check_pose(s, radius_m)
    return s


def pose_failures(s: PoseStats, radius_m: float) -> list[str]:
    """Every gate a pose fails (module docstring), in the order of the pipeline; empty when it passes them all."""
    p = params.current()
    if s.num_corr < p.loc_min_inliers:
        return [R_TOO_FEW_MATCHES]
    if s.focal_px is None or s.enu is None:
        return [R_PNP_FAILED]
    out = []

    def ratio_ok(f: float) -> bool:
        return p.loc_min_focal_ratio <= f / s.long_side_px <= p.loc_max_focal_ratio

    if abs(s.focal_px - s.focal_prior_px) > FOCAL_FIXED_TOL * s.focal_prior_px:
        out.append(R_FREE_FOCAL)
    if not ratio_ok(s.focal_prior_px):
        out.append(R_IMPLAUSIBLE_FOCAL)
    if s.focal_free_px is None:
        out.append(R_FOCAL_UNCHECKED)
    else:
        if not ratio_ok(s.focal_free_px):
            out.append(R_IMPLAUSIBLE_FOCAL)
        if abs(s.focal_free_px - s.focal_prior_px) > p.loc_focal_mismatch * s.focal_prior_px:
            out.append(R_FOCAL_MISMATCH)
    if s.inliers < p.loc_min_inliers:
        out.append(R_INLIERS)
    if s.inlier_ratio < p.loc_min_inlier_ratio:
        out.append(R_INLIER_RATIO)
    if s.reproj_px is None or s.reproj_px > p.loc_max_reproj_frac * s.long_side_px:
        out.append(R_REPROJECTION)
    if s.second_ratio > p.loc_second_ratio:
        out.append(R_AMBIGUOUS_IN_MODEL)
    if math.hypot(s.enu[0], s.enu[1]) > radius_m:
        out.append(R_IMPLAUSIBLE_POSITION)
    if not p.loc_min_height_m <= s.enu[2] <= p.loc_max_height_m:
        out.append(R_IMPLAUSIBLE_HEIGHT)
    if s.look_deg is None or s.look_deg > p.loc_max_look_deg:
        out.append(R_LOOKS_AWAY)
    if p.loc_withhold_gps_conflict and s.gps_conflict:
        out.append(R_GPS_CONFLICT)
    return list(dict.fromkeys(out))


def check_pose(s: PoseStats, radius_m: float) -> str | None:
    """The first gate a pose fails (pose_failures), None when it passes them all."""
    failures = pose_failures(s, radius_m)
    return failures[0] if failures else None


def choose(cands: Sequence[PoseStats]) -> tuple[PoseStats | None, str | None]:
    """(the pose, None) across the maps a query was tried against (3.9), or (None, why): the accepting map with the
    most inliers, a full model on a tie; ambiguous_across_models when another accepting map has at least
    Params.loc_across_half of its inliers and a heading more than loc_across_deg away. Without an accepting map: the
    reason of the map that came closest (most inliers, then correspondences); too_few_matches without any map."""
    p = params.current()

    def order(c: PoseStats) -> tuple:
        return -c.inliers, -c.num_corr, c.kind != JOB_FULL, c.model_id

    ok = sorted((c for c in cands if c.reason is None), key=order)
    if not ok:
        return None, (sorted(cands, key=order)[0].reason if cands else R_TOO_FEW_MATCHES)
    best = ok[0]
    for c in ok[1:]:
        if c.inliers >= p.loc_across_half * best.inliers \
                and abs(angle_diff_deg(c.heading_deg, best.heading_deg)) > p.loc_across_deg:
            return None, R_AMBIGUOUS_ACROSS
    return best, None


# ---------------------------------------------------------------- records


def tried_record(photo: PhotoRecord, query: Query, chosen: PoseStats | None, reason: str | None,
                 cands: Sequence[PoseStats], key: str, previous: Any, now: float,
                 extra: Mapping[str, Any] | None = None) -> LocRecord:
    """The record of a photo the localizer tried against the maps of key: accepted with the chosen pose (choose), else
    rejected with the reason. previous: its stored record (attempts carry on). extra: more diagnostics (pairs, ...).
    Raises ValueError for a chosen pose that failed a gate."""
    attempts = int(_field(previous, "attempts") or 0) + 1 if previous is not None else 1
    stats: dict[str, Any] = {"loc_version": LOC_VERSION, **(extra or {}), "candidates": [c.summary() for c in cands]}
    if chosen is None:
        return LocRecord(photo.id, str(photo.landmark_id), LOC_REJECTED, now, reason=reason or R_TOO_FEW_MATCHES,
                         maps_key=key, attempts=attempts, stats=stats)
    if chosen.reason is not None or chosen.enu is None:
        raise ValueError(f"photo {photo.id}: a pose that failed a gate ({chosen.reason}) is not a localization")
    pose = {"enu": [round(v, 3) for v in chosen.enu], "heading_deg": round(chosen.heading_deg, 2),
            "pitch_deg": round(chosen.pitch_deg, 2), "roll_deg": round(chosen.roll_deg, 2),
            "hfov_deg": round(query.hfov_deg, 2), "aspect": round(query.aspect, 4)}
    stats.update(gps_conflict=chosen.gps_conflict, chosen=chosen.to_meta())
    return LocRecord(photo.id, str(photo.landmark_id), LOC_ACCEPTED, now, maps_key=key, attempts=attempts,
                     model_id=chosen.model_id, pose=pose, stats=stats)


def skipped_record(photo: PhotoRecord, reason: str, key: str, previous: Any, now: float) -> LocRecord:
    """The record of a photo eligible skipped (reason), against the maps of key; attempts carry on."""
    attempts = int(_field(previous, "attempts") or 0) if previous is not None else 0
    return LocRecord(photo.id, str(photo.landmark_id), LOC_SKIPPED, now, reason=reason, maps_key=key,
                     attempts=attempts, stats={"loc_version": LOC_VERSION})


# ---------------------------------------------------------------- small helpers


def _field(rec: Any, key: str) -> Any:
    return rec.get(key) if isinstance(rec, Mapping) else getattr(rec, key, None)


def _finite(v: Any) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) else None


def _round(v: float | None, nd: int) -> float | None:
    return None if v is None else round(float(v), nd)
