"""Visual localization of consented photos no published model shows (D30).

A photo whose uploader agreed to share its location (records.location_shared) and that no published model shows as an
SfM camera can still be placed precisely: by 2-D/3-D PnP of its processed copy against a saved map of the landmark
(crowd3d.landmark_map: a published model's sparse reconstruction, its features in the landmark's feature cache). Display
only: the outcome is one records.LocRecord per photo, which photoposes ranks between sfm and exif (source "loc"); it
never reaches plan_jobs or a ReconPhoto, so no reconstruction changes. The rules (docs/VISUAL_LOCALIZATION.md section
3, narrowed by the D30 design) are the pure functions of the first part; the COLMAP part (ColmapLocalizer: features,
matching, the PnP solves; the last part of this module) hands them Solve objects and the correspondence counts.

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

The COLMAP part (ColmapLocalizer.localize; pycolmap 4.2.1, CPU), per job:
  1. load_sparse_maps (the caller holds the landmark lock): each candidate map's sparse model in memory, kept only when
     it is in its model's frame (map_frame_ok), with the median of its 3-D points (the look-away check's target). The
     job's records are keyed by the candidate maps all the same (maps_key), so a map refused here is not retried
     until the candidates change.
  2. A fresh database in the job's work dir with the queries and the distinct photos of the maps, named in upload order
     (reconstruct.image_name, as a reconstruction job names them), one COLMAP camera per image, the EXIF focal prior
     written as reconstruct writes it. Features: the landmark's FeatureCache first (keyed by the processed copy's
     sha256 and the job feature version), the rest extracted with exactly the reconstruction jobs' options
     (landmark_map.extraction_options), so a map image's keypoint indices are those of its points2D. A map image
     whose keypoint count differs from its points2D count is refused (never matched against), as is an image COLMAP
     cannot read. A query is skipped, never rejected, for such a fault of its inputs: its copy unreadable, no feature
     in it, or no map photo left to match it with (S_NO_IMAGE, S_NO_FEATURES, S_NO_MAP_IMAGES).
  3. Per query (in order; between two queries the job stops when stop() says so or its deadline passed, leaving the
     rest for a later job): its pairs (select_pairs), the cached ones seeded from the FeatureCache in either order
     (a reconstruction job that held the query matched (map photo, query) or (query, map photo)), the rest matched
     in image-id order with match_image_pairs (verification RANSAC seed 0, COLMAP's default matching, as the
     reconstruction jobs do, so the rows are the cache's and theirs); per map the distinct 2-D/3-D correspondences of
     the verified inlier matches
     (correspondences), then pose_in_map: the fixed-focal solve, the free-focal check and the second-best solve
     (estimate_and_refine_absolute_pose, SIMPLE_RADIAL, principal point at the centre, RANSAC threshold
     Params.loc_ransac_frac x the longer side, seed 0), then pose_stats and choose.
  4. The cache gets the job's new features and pairs (each pair under the order its rows were computed in), whatever
     happened. Nothing is logged per photo.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from . import params
from .anchor import GeoAnchor, quat_xyzw, read_anchor
from .config import Settings
from .geo import LocalENU, angle_diff_deg
from .landmark_map import (
    CacheStats,
    FeatureCache,
    camera_signatures,
    extraction_options,
    feature_version,
    file_sha256,
    match_version,
    pair_id,
)
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
from .reconstruct import ColmapReconstructor, camera_flags, gps_inlier_threshold_m, image_name
from .records import (
    JOB_FULL,
    LOC_ACCEPTED,
    LOC_REJECTED,
    LOC_SKIPPED,
    Landmark,
    LocRecord,
    ModelRecord,
    PhotoRecord,
    location_shared,
)

log = logging.getLogger(__name__)

# The rules of this module; part of maps_key, so a change of them tries every photo again.
LOC_VERSION = 1

# why a consented photo without an SfM pose is not tried (LocRecord status skipped). The last three (and no_maps when
# load_sparse_maps refused every candidate) are faults of the inputs ColmapLocalizer.localize meets; the photo was not
# tried, so no verdict on it is recorded (never too_few_matches). Like every record they hold until the candidate maps
# change: none of these faults goes away by itself, so trying again sooner would only meet it again.
S_SFM_FLAGGED = "sfm_flagged"          # its SfM camera is flagged in a published model whose frame is fixed
S_IN_MAP = "registered_in_map"         # registered in a candidate map, its camera there not shown
S_NO_FOCAL = "no_focal"                # no EXIF 35 mm focal length: a free-focal pose is never shown
S_NO_MAPS = "no_maps"                  # the landmark has no candidate map (or none of them could be loaded)
S_NO_IMAGE = "no_image"                # no processed copy, or COLMAP cannot read it
S_NO_FEATURES = "no_features"          # COLMAP found no feature in its processed copy
S_NO_MAP_IMAGES = "no_map_images"      # no map photo to match it with: every one's copy is missing or unreadable, or
#                                        its keypoints are not its points2D (refused)
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
M_NOT_IN_FRAME = "map_not_in_model_frame"   # load_sparse_maps: its images do not give back the model's cameras
M_UNREADABLE = "unreadable_map"             # load_sparse_maps: no sparse model, or pycolmap cannot read it

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
    """The key of a set of candidate maps (candidate_maps, or their model ids), whatever their order: a photo is tried
    again only when its record's key differs. LOC_VERSION is part of it. Every record of a localize job is keyed by
    the candidate maps eligible() was given, whatever load_sparse_maps then refused: what refuses a map there does not
    go away by itself, so a key of the maps loaded would never be the one eligible() compares and every job would try
    the same photos again. The loaded maps (SparseMap) are therefore refused here (TypeError)."""
    ids: set[str] = set()
    for m in maps:
        if isinstance(m, str):
            ids.add(m)
        elif isinstance(m, MapRef):
            ids.add(m.model_id)
        else:
            raise TypeError(f"maps_key takes the candidate maps (MapRef) or their model ids, not {type(m).__name__}")
    doc = json.dumps({"loc_version": LOC_VERSION, "maps": sorted(ids)}, sort_keys=True)
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


# ---------------------------------------------------------------- the COLMAP part (module docstring)


LOCALIZE_THREADS = 2       # a localize job's extraction and matching threads (it shares the machine with the API)
SECOND_MIN_CORR = 6        # the second-best solve needs this many correspondences left over (VISUAL_LOCALIZATION 3.8
#                            measured the check so)
FREE_FOCAL_START = 1.2     # the free-focal check starts from COLMAP's own guess (1.2 x the longer side), not from the
#                            prior it checks


@dataclass(eq=False)
class SparseMap:
    """A candidate map with its sparse model in memory (load_sparse_maps); the localizer only reads it."""

    ref: MapRef
    rec: Any                          # pycolmap.Reconstruction: the published model's job frame (local)
    images: dict[str, int]            # photo id -> its image id in rec
    target_enu: list[float]           # the median of its 3-D points, anchored ENU (the look-away check)
    _point3d_ids: dict[str, np.ndarray] = field(default_factory=dict, repr=False)

    @property
    def model_id(self) -> str:
        return self.ref.model_id

    def num_points2d(self, photo_id: str) -> int:
        return int(self.rec.images[self.images[photo_id]].num_points2D())

    def point3d_ids(self, photo_id: str) -> np.ndarray:
        """Per keypoint of the photo's image in this map, the id of its 3-D point (-1: none)."""
        got = self._point3d_ids.get(photo_id)
        if got is None:
            im = self.rec.images[self.images[photo_id]]
            got = np.array([p.point3D_id if p.has_point3D() else -1 for p in im.points2D], dtype=np.int64)
            self._point3d_ids[photo_id] = got
        return got

    def xyz(self, point3d_ids: np.ndarray) -> np.ndarray:
        """The local positions of these 3-D points."""
        pts = self.rec.points3D
        return np.array([pts[int(i)].xyz for i in point3d_ids], dtype=float).reshape(-1, 3)


def load_sparse_maps(root: Path, maps: Sequence[MapRef], cameras: Mapping[str, Mapping[str, Any]]
                     ) -> tuple[list[SparseMap], dict[str, str]]:
    """(the maps loaded, model_id -> why not) of these candidate maps (candidate_maps), saved under root
    (Settings.maps_dir / <landmark id> / landmark_map.MODELS_DIR). The caller holds the landmark lock: each map is read
    whole into memory before landmark_map.prune_maps can delete it. cameras: model id -> the model's meta["cameras"];
    a map is kept only when its images give back those cameras through the model's anchor (map_frame_ok) and it has
    3-D points. The records stay keyed by maps_key of the candidate maps, not of the maps loaded (maps_key): with
    every map refused, the queries are S_NO_MAPS under that key."""
    import pycolmap

    out: list[SparseMap] = []
    skipped: dict[str, str] = {}
    for m in maps:
        sparse = Path(root) / m.model_id / "sparse"
        rec = None
        if (sparse / "images.bin").is_file():
            try:
                rec = pycolmap.Reconstruction(str(sparse))
            except (RuntimeError, ValueError, OSError) as e:
                log.warning("saved map %s is unreadable: %s", sparse, e)
        if rec is None:
            skipped[m.model_id] = M_UNREADABLE
            continue
        images: dict[str, int] = {}
        poses: dict[str, tuple[list[float], list[float]]] = {}
        for im in rec.images.values():
            pid = m.images.get(im.name)
            if pid is None or not im.has_pose:
                continue
            images[pid] = int(im.image_id)
            poses[pid] = (list(im.cam_from_world().rotation.quat), list(im.projection_center()))
        xyz = np.array([pt.xyz for pt in rec.points3D.values()], dtype=float).reshape(-1, 3)
        if not images or not len(xyz):
            skipped[m.model_id] = M_NO_IMAGES
            continue
        if not map_frame_ok(m.anchor, poses, cameras.get(m.model_id) or {}):
            skipped[m.model_id] = M_NOT_IN_FRAME
            continue
        out.append(SparseMap(m, rec, images, [float(v) for v in np.median(m.anchor.to_enu(xyz), axis=0)]))
    return out, skipped


def correspondences(pairs: Iterable[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """The distinct 2-D/3-D correspondences (query keypoint index, map 3-D point id) of a query in one map, sorted.
    pairs: per map image paired with the query, (its verified inlier matches, k x 2: map keypoint, query keypoint;
    per keypoint of that map image the id of its 3-D point, -1 for none). A match of a map keypoint without a 3-D point
    (or past its keypoints) gives none; one query keypoint matched to one 3-D point through several map images counts
    once."""
    rows = []
    for matches, p3d in pairs:
        m = np.asarray(matches, dtype=np.int64).reshape(-1, 2)
        p3d = np.asarray(p3d, dtype=np.int64)
        m = m[(m[:, 0] >= 0) & (m[:, 0] < len(p3d)) & (m[:, 1] >= 0)]
        ids = p3d[m[:, 0]]
        rows.append(np.stack([m[ids >= 0, 1], ids[ids >= 0]], axis=1))
    if not rows:
        return np.zeros((0, 2), dtype=np.int64)
    return np.unique(np.concatenate(rows).reshape(-1, 2), axis=0)


def solve_pose(p2: np.ndarray, p3: np.ndarray, width: int, height: int, focal_px: float | None,
               max_error_px: float) -> tuple[Solve, np.ndarray] | None:
    """One absolute pose of 2-D points (pixels, COLMAP's keypoint convention) against 3-D points (a map's local frame):
    estimate_and_refine_absolute_pose with a SIMPLE_RADIAL camera of the image's size, principal point at its centre,
    no distortion and none refined, LO-RANSAC with threshold max_error_px and seed 0. focal_px: the focal length,
    fixed (neither estimated nor refined); None: estimated (P4Pf) and refined, from COLMAP's own guess. Returns (the
    Solve: its focal length read back from the camera after the solve, the mean reprojection error of its inliers;
    the inlier mask), None when it does not solve."""
    import pycolmap

    free = focal_px is None
    f0 = FREE_FOCAL_START * max(width, height) if free else float(focal_px)
    cam = pycolmap.Camera(model="SIMPLE_RADIAL", width=int(width), height=int(height),
                          params=[f0, width / 2.0, height / 2.0, 0.0])
    est = pycolmap.AbsolutePoseEstimationOptions()
    est.estimate_focal_length = free
    est.ransac.max_error = float(max_error_px)
    est.ransac.random_seed = 0
    ref = pycolmap.AbsolutePoseRefinementOptions()
    ref.refine_focal_length = free
    ref.refine_extra_params = False
    p2 = np.ascontiguousarray(p2, dtype=np.float64).reshape(-1, 2)
    p3 = np.ascontiguousarray(p3, dtype=np.float64).reshape(-1, 3)
    if len(p2) < 4:
        return None
    res = pycolmap.estimate_and_refine_absolute_pose(p2, p3, cam, est, ref)    # refines cam in place
    if res is None:
        return None
    mask = np.asarray(res["inlier_mask"], dtype=bool).reshape(-1)
    if not mask.any():
        return None
    cfw = res["cam_from_world"]
    rot = np.asarray(cfw.rotation.matrix(), dtype=float)
    t = np.asarray(cfw.translation, dtype=float)
    pc = p3[mask] @ rot.T + t
    if (pc[:, 2] <= 0).any():          # an inlier behind the camera: no honest reprojection error
        reproj = math.inf
    else:
        reproj = float(np.linalg.norm(np.asarray(cam.img_from_cam(pc)) - p2[mask], axis=1).mean())
    return Solve(rot, -rot.T @ t, int(mask.sum()), reproj, float(cam.mean_focal_length())), mask


def pose_in_map(m: MapRef, q: Query, p2: np.ndarray, p3: np.ndarray, target_enu: Sequence[float],
                radius_m: float) -> PoseStats:
    """The query's pose in map m from its distinct 2-D/3-D correspondences (p2: the query keypoints, pixels; p3: the
    map's 3-D points, local frame): the solve with the focal length fixed at the EXIF prior, the second-best solve on
    the correspondences it left out (at least SECOND_MIN_CORR of them) and the free-focal check, each with RANSAC
    threshold Params.loc_ransac_frac x the longer side; then pose_stats. Fewer correspondences than
    Params.loc_min_inliers: no solve (too_few_matches)."""
    p = params.current()
    p2 = np.asarray(p2, dtype=float).reshape(-1, 2)
    p3 = np.asarray(p3, dtype=float).reshape(-1, 3)
    fixed = free = second = None
    if len(p2) >= p.loc_min_inliers:
        max_err = p.loc_ransac_frac * q.long_side
        got = solve_pose(p2, p3, q.width, q.height, q.focal_px, max_err)
        if got is not None:
            fixed, inliers = got
            rest = ~inliers
            if int(rest.sum()) >= SECOND_MIN_CORR:
                sec = solve_pose(p2[rest], p3[rest], q.width, q.height, q.focal_px, max_err)
                second = None if sec is None else sec[0]
            chk = solve_pose(p2, p3, q.width, q.height, None, max_err)
            free = None if chk is None else chk[0]
    return pose_stats(m, q, len(p2), fixed, free, second, target_enu, radius_m)


def _pair_ids(db_path: Path) -> set[int]:
    with closing(sqlite3.connect(db_path, timeout=60.0)) as db:
        return {r[0] for r in db.execute("SELECT pair_id FROM two_view_geometries")}


class LocCache:
    """A localize job's use of the landmark's FeatureCache (landmark_map), as a reconstruction job's JobCache: the
    features seeded before extraction, the pairs before matching, the job's new rows harvested at the end; a failing
    cache never fails the job (it logs once and gets out of the way).

    A pair is looked up in both orders. COLMAP's matching is not symmetric, so the cache keys a pair's rows by the order
    its job matched it in, and a reconstruction job that held the query may have matched (map photo, query) as well as
    (query, map photo). Each order is looked up with this database's own image-id order (FeatureCache.seed_pairs), so
    whatever is seeded is laid out as this database expects. The harvest stores every pair under the order its rows
    came from: a pair seeded in the other order stays as cached, one this job matched is stored in the order it was
    matched (order)."""

    def __init__(self, cache: FeatureCache, pycolmap, db_path: Path, image_paths: Mapping[str, Path], extraction,
                 verification) -> None:
        self.cache, self.db_path = cache, Path(db_path)
        self.image_paths = dict(image_paths)
        self.feature_version = feature_version(pycolmap, extraction)
        self.match_version = match_version(pycolmap, verification)
        self.stats = CacheStats()
        self.seconds = 0.0
        self.keys: dict[int, str] = {}
        self.sigs: dict[int, str] = {}
        self.order: dict[int, tuple[int, int]] = {}     # pair id -> the order its rows are (or will be) computed in
        self.seeded: set[int] = set()                   # pair ids seeded from the cache
        self.ok = True

    def _run(self, what: str, step: Callable[[], Any]) -> None:
        if not self.ok:
            return
        t = time.monotonic()
        try:
            step()
        except (sqlite3.Error, OSError, RuntimeError) as e:
            self.ok = False
            log.warning("feature cache %s: %s failed (%s); this localize job goes on without it", self.cache.path,
                        what, e)
        finally:
            self.seconds += time.monotonic() - t

    def seed_features(self) -> None:
        def step() -> None:
            with closing(sqlite3.connect(self.db_path, timeout=60.0)) as db:
                ids = dict(db.execute("SELECT name, image_id FROM images").fetchall())
            self.keys = {ids[name]: file_sha256(path) for name, path in self.image_paths.items() if name in ids}
            self.stats.features = len(self.keys)
            self.stats.features_hit = self.cache.seed_features(self.db_path, self.keys, self.feature_version)
        self._run("seeding features", step)

    def seed_pairs(self, order: Mapping[int, tuple[int, int]]) -> None:
        """Seed these pairs (pair id -> the order this job would match it in) after the focal priors are written (the
        camera signatures read them); self.order becomes the order each pair's rows are computed in."""
        self.order = dict(order)
        self.stats.pairs = len(self.order)

        def step() -> None:
            self.sigs = camera_signatures(self.db_path)
            wanted = list(self.order.values())
            before = _pair_ids(self.db_path)
            self.cache.seed_pairs(self.db_path, wanted, self.keys, self.sigs, self.match_version)
            first = (_pair_ids(self.db_path) - before) & self.order.keys()
            self.cache.seed_pairs(self.db_path, [(b, a) for a, b in wanted], self.keys, self.sigs, self.match_version)
            second = (_pair_ids(self.db_path) - before - first) & self.order.keys()
            for pid in second:
                a, b = self.order[pid]
                self.order[pid] = (b, a)
            self.seeded = first | second
            self.stats.pairs_hit = len(self.seeded)
        self._run("seeding pairs", step)

    def harvest(self, label: str | None = None) -> None:
        """Store the job's new rows (the pairs it matched, in their order) and record its use."""
        self._run("harvest", lambda: self.cache.harvest(self.db_path, self.keys, self.feature_version,
                                                        list(self.order.values()), self.sigs, self.match_version))
        self._run("recording its use", lambda: self.cache.record_use(self.stats, self.seconds, label))


@dataclass
class Outcome:
    """A query the localizer tried: the chosen pose (choose; None: reason), every map's PoseStats, counts (pairs,
    pairs_capped, pairs_seeded, maps, seconds, cpu_s; never a photo id)."""

    photo: PhotoRecord
    query: Query
    chosen: PoseStats | None
    reason: str | None
    cands: list[PoseStats]
    extra: dict[str, Any] = field(default_factory=dict)

    def record(self, key: str, previous: Any, now: float) -> LocRecord:
        """Its LocRecord against the maps of key (tried_record)."""
        return tried_record(self.photo, self.query, self.chosen, self.reason, self.cands, key, previous, now,
                            self.extra)


@dataclass
class LocalizeRun:
    """What one ColmapLocalizer.localize call did."""

    outcomes: dict[str, Outcome] = field(default_factory=dict)    # photo id -> its outcome, in the order tried
    skipped: dict[str, str] = field(default_factory=dict)         # photo id -> why not tried (S_NO_IMAGE, ...)
    #                                                               (records under the candidate maps' maps_key)
    unfinished: list[str] = field(default_factory=list)           # left for a later job (stop() or the deadline)
    stats: dict[str, Any] = field(default_factory=dict)           # counts and seconds of the job (no photo id)


class ColmapLocalizer:
    """PnP of consented photos into a landmark's saved maps with pycolmap (the module docstring, the COLMAP part)."""

    name = "colmap"

    def __init__(self, settings: Settings | None = None, threads: int = LOCALIZE_THREADS) -> None:
        self.settings = settings or Settings()
        self.threads = int(threads)

    def load_maps(self, root: Path, maps: Sequence[MapRef], cameras: Mapping[str, Mapping[str, Any]]
                  ) -> tuple[list[SparseMap], dict[str, str]]:
        """The candidate maps' sparse models for localize (load_sparse_maps; the caller holds the landmark lock)."""
        return load_sparse_maps(root, maps, cameras)

    def localize(self, queries: Sequence[PhotoRecord], maps: Sequence[SparseMap], photos: Iterable[PhotoRecord],
                 lm: Landmark, work_dir: Path, *, cache: FeatureCache | None = None,
                 stop: Callable[[], bool] | None = None, deadline: float | None = None) -> LocalizeRun:
        """Localize the queries (photos eligible() chose) in these maps (load_sparse_maps). photos: the landmark's
        photos (the maps' photos among them; processed copies under Settings.data_dir); work_dir: the job's own, left
        for the caller to remove; cache: the landmark's FeatureCache (Params.feature_cache), None to extract and match
        everything; stop / deadline (time.monotonic()): checked before each query, the rest is left unfinished.
        Programming errors raise. A fault of the inputs never becomes a verdict on a query: a query is skipped when its
        processed copy is gone or COLMAP cannot read it (S_NO_IMAGE), COLMAP finds no feature in it (S_NO_FEATURES),
        or no map photo is left to match it with (S_NO_MAP_IMAGES: every one's copy is gone or unreadable, or its
        keypoints are not its points2D); the caller records run.skipped as eligible()'s skipped photos, under the
        maps_key of the candidate maps."""
        import pycolmap

        s = self.settings
        t0, c0 = time.monotonic(), time.process_time()
        work_dir = Path(work_dir)
        known = {ph.id: ph for ph in photos}
        known.update({q.id: q for q in queries})
        run = LocalizeRun()

        def path_of(pid: str) -> Path | None:
            ph = known.get(pid)
            path = s.data_dir / ph.processed_path if ph is not None and ph.processed_path else None
            return path if path is not None and path.is_file() else None

        def halted() -> bool:
            return (stop is not None and bool(stop())) or (deadline is not None and time.monotonic() >= deadline)

        map_photos = {pid for m in maps for pid in m.images}
        present = {pid for pid in map_photos if path_of(pid) is not None}
        todo: list[PhotoRecord] = []
        for q in dict((q.id, q) for q in queries).values():
            if path_of(q.id) is None:
                run.skipped[q.id] = S_NO_IMAGE
            elif not maps:
                run.skipped[q.id] = S_NO_MAPS
            elif not present - {q.id}:
                run.skipped[q.id] = S_NO_MAP_IMAGES
            else:
                todo.append(q)
        run.stats = {"queries": len(todo), "maps": len(maps), "map_photos": len(map_photos),
                     "map_photos_missing": len(map_photos - present), "map_photos_unreadable": 0,
                     "map_images_refused": 0, "images": 0, "pairs": 0, "pairs_seeded": 0, "features": 0,
                     "features_seeded": 0}

        def finish() -> LocalizeRun:
            run.stats.update(tried=len(run.outcomes),
                             localized=sum(o.chosen is not None for o in run.outcomes.values()),
                             skipped=len(run.skipped), unfinished=len(run.unfinished),
                             seconds=round(time.monotonic() - t0, 2), cpu_s=round(time.process_time() - c0, 2))
            return run

        if not todo:
            return finish()
        if halted():
            run.unfinished = [q.id for q in todo]
            return finish()

        ids = sorted({q.id for q in todo} | present, key=lambda pid: (known[pid].uploaded_at, pid))
        names = {pid: image_name(k, pid) for k, pid in enumerate(ids)}
        images_dir, db_path, list_path = work_dir / "images", work_dir / "database.db", work_dir / "pairs.txt"
        shutil.rmtree(images_dir, ignore_errors=True)
        images_dir.mkdir(parents=True)
        db_path.unlink(missing_ok=True)
        for pid, name in names.items():
            src = path_of(pid)
            try:
                os.symlink(src.resolve(), images_dir / name)
            except OSError:
                shutil.copyfile(src, images_dir / name)
        run.stats["images"] = len(ids)

        pycolmap.logging.minloglevel = pycolmap.logging.Level.ERROR.value
        pycolmap.set_random_seed(0)
        extraction = extraction_options(pycolmap, s.processing_max_side_px)
        extraction.num_threads = self.threads          # not part of the feature version (landmark_map._options)
        verification = pycolmap.TwoViewGeometryOptions()
        verification.ransac.random_seed = 0
        matching = pycolmap.FeatureMatchingOptions()
        matching.num_threads = self.threads
        all_names = sorted(names.values())
        pycolmap.Database.open(db_path).close()
        pycolmap.import_images(db_path, images_dir, camera_mode=pycolmap.CameraMode.PER_IMAGE, image_names=all_names)
        lc = (LocCache(cache, pycolmap, db_path, {names[pid]: path_of(pid) for pid in ids}, extraction, verification)
              if cache is not None else None)
        try:
            if lc is not None:
                lc.seed_features()
            pycolmap.extract_features(db_path, images_dir, image_names=all_names,
                                      camera_mode=pycolmap.CameraMode.PER_IMAGE, extraction_options=extraction,
                                      device=pycolmap.Device.cpu)
            ColmapReconstructor._write_focal_priors(pycolmap, db_path, {names[pid]: known[pid] for pid in ids})
            image_id, size, num_kp = self._read_images(pycolmap, db_path)
            # an image COLMAP cannot read is not imported at all: such a query is skipped, such a map photo is not used
            unreadable = {pid for pid, name in names.items() if name not in image_id}
            present -= unreadable
            run.stats["map_photos_unreadable"] = len(unreadable & map_photos)
            pid_of_image = {image_id[name]: pid for pid, name in names.items() if pid not in unreadable}

            usable: dict[str, set[str]] = {}       # model id -> its photos whose keypoints are its points2D
            for m in maps:
                mine = {pid for pid in m.images if pid in present}
                usable[m.model_id] = {pid for pid in mine if num_kp.get(image_id[names[pid]]) == m.num_points2d(pid)}
                run.stats["map_images_refused"] += len(mine) - len(usable[m.model_id])
            refs = [replace(m.ref, images={n: pid for n, pid in m.ref.images.items() if pid in usable[m.model_id]})
                    for m in maps]

            plan = []
            for q in todo:
                if q.id in unreadable:
                    run.skipped[q.id] = S_NO_IMAGE
                    continue
                try:
                    query = make_query(q, *size[names[q.id]], lm.lat, lm.lon)
                except ValueError:
                    run.skipped[q.id] = S_NO_FOCAL
                    continue
                qid = image_id[names[q.id]]
                if not num_kp.get(qid):
                    run.skipped[q.id] = S_NO_FEATURES
                    continue
                partners, capped = select_pairs(query, refs)
                if not partners:
                    run.skipped[q.id] = S_NO_MAP_IMAGES
                    continue
                plan.append((q, query, qid, partners, capped, [(qid, image_id[names[pid]]) for pid in partners]))
            # matched in image-id order, as an exhaustive reconstruction job matches a pair (the ids follow the upload
            # order in both), so a later job of the landmark finds what this one harvests
            order = {pair_id(a, b): (min(a, b), max(a, b)) for *_, pairs in plan for a, b in pairs}
            seeded: set[int] = set()
            if lc is not None:
                lc.seed_pairs(order)
                if lc.ok:
                    order, seeded = lc.order, lc.seeded
            run.stats["pairs"] = len(order)

            for k, (q, query, qid, partners, capped, pairs) in enumerate(plan):
                if halted():
                    run.unfinished = [x[0].id for x in plan[k:]]
                    break
                tq, cq = time.monotonic(), time.process_time()
                todo_pairs = [order[pair_id(a, b)] for a, b in pairs if pair_id(a, b) not in seeded]
                if todo_pairs:
                    list_path.write_text("".join(f"{names[pid_of_image[a]]} {names[pid_of_image[b]]}\n"
                                                 for a, b in todo_pairs), encoding="utf-8")
                    pairing = pycolmap.ImportedPairingOptions()
                    pairing.match_list_path = str(list_path)
                    pycolmap.match_image_pairs(db_path, matching_options=matching, pairing_options=pairing,
                                               verification_options=verification, device=pycolmap.Device.cpu)
                cands = self._poses(pycolmap, db_path, maps, usable, query, qid,
                                    {pid: image_id[names[pid]] for pid in partners}, lm.radius_m)
                chosen, reason = choose(cands)
                run.outcomes[q.id] = Outcome(q, query, chosen, reason, cands, {
                    "pairs": len(pairs), "pairs_capped": capped,
                    "pairs_seeded": sum(pair_id(a, b) in seeded for a, b in pairs), "maps": len(maps),
                    "seconds": round(time.monotonic() - tq, 2), "cpu_s": round(time.process_time() - cq, 2)})
        finally:
            if lc is not None:
                lc.harvest(f"localize {work_dir.name}")
                run.stats.update(features=lc.stats.features, features_seeded=lc.stats.features_hit,
                                 pairs_seeded=lc.stats.pairs_hit)
        return finish()

    @staticmethod
    def _read_images(pycolmap, db_path: Path) -> tuple[dict[str, int], dict[str, tuple[int, int]], dict[int, int]]:
        """(image name -> image id, image name -> (width, height) of its camera, image id -> keypoint count)."""
        db = pycolmap.Database.open(db_path)
        try:
            image_id, size, num_kp = {}, {}, {}
            for im in db.read_all_images():
                cam = db.read_camera(im.camera_id)
                image_id[im.name] = int(im.image_id)
                size[im.name] = (int(cam.width), int(cam.height))
                num_kp[int(im.image_id)] = (int(db.num_keypoints_for_image(im.image_id))
                                            if db.exists_keypoints(im.image_id) else 0)
            return image_id, size, num_kp
        finally:
            db.close()

    @staticmethod
    def _poses(pycolmap, db_path: Path, maps: Sequence[SparseMap], usable: Mapping[str, set[str]], query: Query,
               qid: int, partners: Mapping[str, int], radius_m: float) -> list[PoseStats]:
        """The query's PoseStats in every map, from the verified inlier matches with its partners (photo id -> image
        id; read_two_view_geometry(map image, query) gives them as (map keypoint, query keypoint))."""
        db = pycolmap.Database.open(db_path)
        try:
            kp = np.asarray(db.read_keypoints(qid), dtype=float) if db.exists_keypoints(qid) else np.zeros((0, 2))
            kp = kp.reshape(len(kp), -1) if kp.size else np.zeros((0, 2))
            geometry = {}
            for pid, mid in partners.items():
                if db.exists_two_view_geometry(mid, qid):
                    geometry[pid] = np.asarray(db.read_two_view_geometry(mid, qid).inlier_matches,
                                               dtype=np.int64).reshape(-1, 2)
        finally:
            db.close()
        cands = []
        for m in maps:
            corr = correspondences((geometry[pid], m.point3d_ids(pid))      # partners never hold the query itself
                                   for pid in sorted(usable[m.model_id] & geometry.keys()))
            corr = corr[corr[:, 0] < len(kp)]
            cands.append(pose_in_map(m.ref, query, kp[corr[:, 0], :2], m.xyz(corr[:, 1]), m.target_enu, radius_m))
        return cands


# reconstructor name (reconstruct.get_reconstructor) -> the localizer for the maps its models save
LOCALIZERS: dict[str, Callable[[Settings | None], ColmapLocalizer]] = {ColmapReconstructor.name: ColmapLocalizer}


def get_localizer(name: str, settings: Settings | None = None) -> ColmapLocalizer | None:
    """The localizer for the saved maps of the reconstructor of this name; None for one whose models save no map (the
    fake reconstructor) or without pycolmap."""
    factory = LOCALIZERS.get(name)
    if factory is None:
        return None
    try:
        import pycolmap  # noqa: F401
    except ImportError:
        return None
    return factory(settings)
