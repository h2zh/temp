"""Evaluate the Crowd3D pipeline on real photos of a famous building, as a crowd would upload them.

    # Commons test-set dir (scripts/fetch_commons_landmark.py output) + a catalog entry:
    python -I scripts/eval_real_landmark.py --commons-dir <dir> --catalog eval/landmarks.json \
        --slug florence-baptistery --out <run dir> [--gt-sparse <MegaDepth-X sparse> --gt-map <json>]
    # One building of a frozen test set, resolved by scripts/benchmark.py:
    python -I scripts/eval_real_landmark.py --building-spec <spec.json> --config eval/configs/default.json --out <dir>
    # Legacy: a MegaDepth-X scene with its GT-ENU analysis (augmented = simulated tags, off by default):
    python -I scripts/eval_real_landmark.py --scene-dir <scene>/<component> --lat ... --lon ... --mode real

Crowd protocol (the default; the product rule: uploads are camera originals whose own EXIF carries GPS):
1. Only photos whose file carries EXIF GPS are uploaded, through the product code (ingest_upload,
   JobRunner with inline jobs, the configured reconstructor, qa), with product defaults unless a
   pipeline config (--config: reconstructor, Settings overrides, params.Params overrides) says otherwise.
2. Arrival mimics the crowd: one upload request per author and capture day (DateTimeOriginal), in the
   order those sessions ended, at most MAX_FILES_PER_UPLOAD photos per request; jobs are planned after
   every request, as the HTTP API does. Authors are the pseudo users.
3. Ground-truth-free metrics (METRICS, the main ones): validation funnel, sectors, jobs, models
   published / rejected by the quality gate and why, registration rate, an independent
   georegistration check (the published models' wall points fitted to the OSM outline: the rotation
   and shift they need measure the absolute orientation/position error without GPS circularity),
   camera estimates against their own EXIF GPS and compass, runtime.
4. With reference poses (MegaDepth-X COLMAP model, or the synthetic generator's truth) for some photos:
   relative pose accuracy (Sim3 of the reference camera centres onto the model) and gross errors
   (twin facades).
5. Writes summary.json, summary.md and topdown.png into the run dir; the Crowd3D data dir under it can
   be browsed: python -m crowd3d.api --data-dir <run dir>/data

Data: Wikimedia Commons photos keep their own licences; MegaDepth-X (CC BY 4.0 SfM, per-photo licences
in its license.parquet). Never put photos or anything derived from them (data dirs, point clouds,
screenshots) into this repository: --out and --data-dir must lie outside it or under its gitignored
data/ folder.

Legacy MegaDepth-X protocol (--scene-dir/--scene), per mode:
1. Loads the GT model and georeferences it into the landmark's ENU frame using only the photos with
   real EXIF GPS (+ real compass): GT-ENU. This frame is fitted to the same GPS the pipeline uses,
   so errors against it are partly circular; --footprint (e.g. an OSM outline) gives an independent
   check.
2. Builds the uploads. "real" (default): only photos that really carry EXIF GPS, untouched.
   "augmented" (legacy, opt-in): all photos; missing GPS/heading tags are simulated from the GT-ENU
   pose plus seeded Gaussian noise and written into the EXIF of a copy (pixels untouched).
3. Runs the product code in two stages: first the most popular sector only (expect a facade model),
   then everything else, spread over --users pseudo users.
4. Writes summary.json and summary.md into the run dir.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import re
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import piexif

if __package__ in (None, ""):
    # `python scripts/eval_real_landmark.py` from a checkout: make the crowd3d package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd3d import params  # noqa: E402
from crowd3d.config import Settings  # noqa: E402
from crowd3d.coverage import sector_for_bearing, sector_label  # noqa: E402
from crowd3d.exif import PhotoMeta, UnreadableImage, build_exif_bytes, read_photo_meta  # noqa: E402
from crowd3d.geo import LocalENU, angle_diff_deg, heading_from_en, local_bearing_distance  # noqa: E402
from crowd3d.reconstruct import (  # noqa: E402
    GPS_INLIER_M,
    apply_similarity,
    compass_yaw,
    estimate_up,
    photo_id_from_image_name,
    robust_similarity_2d,
    rotation_to_z,
    rotation_z,
    similarity_matrix,
)

log = logging.getLogger("eval_real_landmark")

REPO_ROOT = Path(__file__).resolve().parent.parent
MEGADEPTH_X_URL = "https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X/resolve/main/release/{name}.tar.gz"
DEFAULT_CACHE_DIR = Path(os.environ.get(
    "CROWD3D_REALDATA_DIR",
    "/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad/realdata",
))
MODES = ("real", "augmented")

GT_GPS_SIGMA_M = 5.0                 # real phone GPS in a piazza (ARCHITECTURE 3.8 uses 5-6 m), 1 sigma per axis
GT_GPS_INLIER_M = 3.0 * GT_GPS_SIGMA_M
TOP_STRUCTURE_FRACTION = 0.001       # highest 0.1 % of GT points: lantern / spire / roof ridge
FOOTPRINT_WALL_Z_M = (2.0, 20.0)     # GT points in this height band near the footprint are walls
GROSS_HEADING_DEG = 45.0             # 3 compass sigmas: beyond this a camera is wrong, not noisy
GROSS_POSITION_M = 20.0
SECTOR_NAMES = {i: sector_label(i, 8) for i in range(8)}

ARRIVALS = ("session", "photo", "single")   # one request per author-day / per photo / everything at once
MAX_FILES_PER_UPLOAD = 50            # crowd3d.api accepts at most this many files per request
MAX_USER_ID_CHARS = 64               # crowd3d.api's user_id limit
WALL_RING_MIN_M = 10.0               # model points within the outline's radius plus max(this, ...
WALL_RING_FRACTION = 0.5             # ... this x that radius) of its centroid may be walls
MIN_WALL_POINTS = 50
OUTLINE_TIE_FACTOR = 2.0             # outline fits within this factor (or OUTLINE_TIE_M) of the best wall
OUTLINE_TIE_M = 0.15                 # distance fit equally well: the smallest rotation among them is reported
OUTLINE_AMBIGUOUS_DEG = 2.0          # tied fits further apart than this are flagged ambiguous
OUTLINE_AMBIGUOUS_M = 2.0
MAX_WALL_POINTS = 20000              # the outline fit subsamples to this many
STYLE_MIN_REF_CAMERAS = 3            # a passing snap is checked against the outline-corrected reference with ...
STYLE_RIGHT_ROT_DEG = 5.0            # ... at least this many cameras; it is right when the correction still needed
STYLE_RIGHT_SHIFT_M = 1.5            # ... turns it by at most this, moves its wall points by at most this (90th
STYLE_RIGHT_SCALE = 0.10             # percentile; = snap_max_residual_m, plane_tol_m) and rescales it by at most this
                                     # (a 30 m facade stretched 10 % puts its ends' photos 1.5 m off)
HARNESS_VERSION = 4                  # 3: stylized models (--stylize); 4: the snap gets the model's SnapPrior

# ---------------------------------------------------------------- metrics (the one definition; benchmark.py uses it)

METRIC_SCHEMA_VERSION = 4       # 2: the style_* group (stylized models, docs/STYLIZED_MODEL.md §3.2, §7)
                                # 3: + style_registration_right, style_rejected_placed_right, style_snap_refused_gps,
                                #    style_snap_refused_ambiguous (style_rejected_placed_right was added to 3 in
                                #    place; no result file without it remains)
                                # 4: "right" = within 5 deg, 1.5 m over the wall points (90th percentile) and 10 %
                                #    scale (3: 3 m at the walls' median point and 15 %); + style_gate_passed_not_eligible,
                                #    style_ref_camera_error_worst_m


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    direction: str             # "higher" / "lower" is better; "none": context only
    aggregate: str             # across buildings: "sum", "mean" or "median" of the non-missing values, or
    #                            "pooled": sum(pooled[0]) / sum(pooled[1]) over the buildings that have both
    group: str                 # input, outcome, outline, sensors, gt, runtime, style
    description: str
    min_delta: float = 0.0     # a change up to this is never significant ...
    rel_floor: float = 0.0     # ... nor one up to this fraction of the earlier value
    pooled: tuple[str, str] | None = None   # (numerator, denominator) metric names for aggregate "pooled"


METRICS: tuple[Metric, ...] = (
    Metric("photos_listed", "photos", "none", "sum", "input",
           "Photos of the test set present (and verified) for this building."),
    Metric("photos_uploaded", "photos", "none", "sum", "input",
           "Photos uploaded: those whose own EXIF carries GPS (all of them where the set says so)."),
    Metric("photos_accepted", "photos", "none", "sum", "input", "Photos the product's validation accepted."),
    Metric("acceptance_rate", "fraction", "none", "mean", "input", "photos_accepted / photos_uploaded.", 0.005),
    Metric("sectors_with_photos", "sectors", "none", "sum", "input", "Sectors with at least one accepted photo."),
    Metric("sectors_face_ready", "sectors", "none", "sum", "input",
           "Sectors with enough accepted photos for a facade model."),
    Metric("jobs_run", "jobs", "none", "sum", "outcome", "Reconstruction jobs the planner started."),
    Metric("jobs_failed_sfm", "jobs", "lower", "sum", "outcome",
           "Jobs whose reconstruction failed outright (no model at all)."),
    Metric("models_published", "models", "higher", "sum", "outcome",
           "Models that passed the quality gate, superseded ones included."),
    Metric("models_qa_rejected", "models", "lower", "sum", "outcome", "Models the quality gate kept unpublished."),
    Metric("facades_published", "sectors", "higher", "sum", "outcome",
           "Sectors with a published facade model at the end."),
    Metric("full_model_published", "0/1", "higher", "sum", "outcome", "1 if a full model is published at the end."),
    Metric("registration_rate", "fraction", "higher", "mean", "outcome",
           "Registered / input photos over the latest attempt per target (each facade, the full model);"
           " a failed reconstruction counts as 0 registered.", 0.01),
    Metric("estimate_coverage", "fraction", "higher", "mean", "outcome",
           "Accepted photos for which the landmark page shows an estimated pose.", 0.01),
    Metric("flagged_camera_fraction", "fraction", "lower", "mean", "outcome",
           "Cameras of the latest published models withheld (disagree with their own sensors, outside the"
           " radius, degenerate focal length).", 0.01),
    Metric("outline_rotation_abs_deg", "deg", "lower", "median", "outline",
           "|Rotation| the wall points of the latest published model per target need to fit the OSM outline"
           " (median over those models): a GPS-free check of the absolute orientation.", 0.2),
    Metric("outline_shift_m", "m", "lower", "median", "outline",
           "Shift those wall points need: a GPS-free check of the absolute position (a single straight wall"
           " only constrains the part normal to it).", 0.2),
    Metric("outline_residual_m", "m", "lower", "median", "outline",
           "Median wall-point distance to the outline after that fit: how well the walls match the shape.", 0.05),
    Metric("outline_models", "models", "none", "sum", "outline",
           "Latest published models with enough wall points for the outline fit."),
    Metric("gps_residual_median_m", "m", "lower", "median", "sensors",
           "Median distance of the shown camera estimates from the photos' own EXIF GPS (partly circular:"
           " GPS also drives the georegistration).", 0.1),
    Metric("compass_error_median_deg", "deg", "lower", "median", "sensors",
           "Median |shown estimated heading - EXIF compass| over the photos with a compass reading.", 0.1),
    Metric("compass_photos", "photos", "none", "sum", "sensors", "Shown estimates with an EXIF compass reading."),
    Metric("gt_cameras", "cameras", "none", "sum", "gt",
           "Registered cameras of the latest published models that have a reference pose."),
    Metric("gt_rotation_error_median_deg", "deg", "lower", "median", "gt",
           "Median camera rotation error against the reference after the best global rotation (per model,"
           " pooled over the latest published models).", 0.05),
    Metric("gt_position_error_median_m", "m", "lower", "median", "gt",
           "Median camera position error after a Sim3 of the reference centres onto the model (metres of the"
           " model's frame).", 0.05),
    Metric("gt_gross_cameras", "cameras", "lower", "sum", "gt",
           f"Reference-posed cameras more than {GROSS_HEADING_DEG:.0f} deg or {GROSS_POSITION_M:.0f} m off"
           " (twin facades and the like)."),
    Metric("gt_gross_shown", "cameras", "lower", "sum", "gt", "Of those, the ones without a flag: shown to users."),
    Metric("wall_time_s", "s", "lower", "sum", "runtime", "Wall time of the building run.", 5.0, 0.25),
    Metric("jobs_time_s", "s", "lower", "sum", "runtime", "Sum of the reconstruction jobs' run times.", 5.0, 0.25),
    # Stylized models (benchmark run --stylize, the default): crowd3d.stylize on every published model, as
    # scripts/stylize_model.py --job-dir does; "eligible" = published and georegistered (georeg_level = full).
    Metric("style_models_stylized", "models", "none", "sum", "style",
           "Published models (superseded ones included) the stylization ran on."),
    Metric("style_models_eligible", "models", "none", "sum", "style",
           "Of those, georegistered ones (georeg_level = full): the models the registration gate is for."),
    Metric("style_registration_passed", "models", "higher", "sum", "style",
           "Eligible models passing the registration gate (STYLIZED_MODEL 3.2: snap accepted with median wall"
           " distance <= snap_max_residual_m, and the edge-by-edge verification)."),
    Metric("style_registration_pass_rate", "fraction", "higher", "pooled", "style",
           "style_registration_passed / style_models_eligible (across buildings: pooled over all models).", 0.02,
           pooled=("style_registration_passed", "style_models_eligible")),
    Metric("style_registration_checked", "models", "none", "sum", "style",
           f"Passing models with at least {STYLE_MIN_REF_CAMERAS} cameras that have an outline-corrected reference"
           " pose."),
    Metric("style_registration_wrong", "models", "lower", "sum", "style",
           "Of those, passes that are wrong: the similarity the snapped model's cameras still need to match the"
           f" outline-corrected reference exceeds {STYLE_RIGHT_ROT_DEG:g} deg, moves its wall points by more than"
           f" {STYLE_RIGHT_SHIFT_M:g} m (90th percentile) or rescales it by more than {STYLE_RIGHT_SCALE:.0%}"
           " (photos would colour the wrong wall, or the wrong part of it)."),
    Metric("style_registration_right", "models", "higher", "sum", "style",
           "Of the checked passes, the right ones (the rest of style_registration_checked is wrong)."),
    Metric("style_rejected_placed_right", "models", "lower", "sum", "style",
           f"Eligible models the gate refused although the georegistration alone already placed them right"
           f" (at least {STYLE_MIN_REF_CAMERAS} reference cameras; the correction they needed was within"
           f" {STYLE_RIGHT_ROT_DEG:g} deg, {STYLE_RIGHT_SHIFT_M:g} m over the walls and {STYLE_RIGHT_SCALE:.0%}):"
           " clay where colour would have been right, the price of refusing wrong placements."),
    Metric("style_gate_passed_not_eligible", "models", "lower", "sum", "style",
           "Published models that are not georegistered (prior only) yet pass the registration gate: they would"
           " be coloured from a compass yaw and a guessed scale (STYLIZED_MODEL 3.2 item 4; must be 0)."),
    Metric("style_snap_refused_gps", "models", "none", "sum", "style",
           "Eligible models whose snap was refused because, snapped, their cameras would stand nearer to or"
           " farther from the walls than their own EXIF GPS says (points and cameras disagree)."),
    Metric("style_snap_refused_ambiguous", "models", "none", "sum", "style",
           "Eligible models whose snap was refused because another placement (a parallel face, a bay away)"
           " fits about as well."),
    Metric("style_snap_residual_before_m", "m", "lower", "median", "style",
           "Median wall-point distance to the footprint before snapping (the georegistration alone), median over"
           " the eligible models.", 0.1),
    Metric("style_snap_residual_m", "m", "lower", "median", "style",
           "The same after an accepted snap (gate condition: <= 1.5 m), median over those models.", 0.05),
    Metric("style_ref_camera_error_m", "m", "lower", "median", "style",
           "Median horizontal camera-centre error of the checked passing models after their snap, against the"
           " outline-corrected reference (median over those models).", 0.2),
    Metric("style_ref_camera_error_worst_m", "m", "lower", "max", "style",
           "The largest of those per-model median camera-centre errors. Cameras stand 40-250 m from the walls, so"
           " a 1 deg turn about the walls moves them 0.7-4 m while the photos move far less on the walls: reported,"
           " not part of the right / wrong verdict.", 0.2),
    Metric("style_walls_total", "walls", "none", "sum", "style",
           "Massing walls of the building's best stylized model (most coloured walls, then observed area)."),
    Metric("style_walls_coloured", "walls", "higher", "sum", "style",
           "Walls with photo colour in that model (unlocked + partly unlocked); the rest are clay (待解锁)."),
    Metric("style_walls_unlocked", "walls", "higher", "sum", "style",
           "Of those, unlocked (STYLIZED_MODEL 3.3: mask >= 60 % and >= 3 cameras each seeing >= 15 % of it)."),
    Metric("style_observed_area", "fraction", "higher", "mean", "style",
           "Observed wall area / all wall area of that model, observed-fill excluded.", 0.02),
    Metric("style_roof_observed_area", "fraction", "none", "mean", "style",
           "Observed roof area / all roof area of that model.", 0.02),
    Metric("style_fill_m2", "m2", "none", "sum", "style",
           "Small holes filled with the wall's own base colour (D12, observed-fill) in that model.", 0.1),
    Metric("style_qa_failed", "models", "lower", "sum", "style",
           "Stylized models failing the honesty QA, audit_model or audit_glb (must be 0)."),
    Metric("style_provenance_violations", "count", "lower", "sum", "style",
           "Honesty QA violations summed over all stylized models (non-clay texels outside mask and legal fill,"
           " textured or photo-claiming locked faces, faces without provenance, off-palette colours, illegal"
           " fill regions; must be 0)."),
    Metric("style_glb_kb_gz", "KB", "lower", "max", "style", "gzip size of the best stylized model's GLB.", 2.0),
    Metric("style_time_s", "s", "lower", "sum", "style",
           "Stylization wall time over all stylized models of the building.", 5.0, 0.25),
)
METRIC_BY_NAME = {m.name: m for m in METRICS}
STYLE_METRICS = tuple(m.name for m in METRICS if m.group == "style")


# ---------------------------------------------------------------- small pure helpers


def summarize(values) -> dict[str, Any]:
    """n, median, 90th percentile, mean and max of the finite values (None when there are none)."""
    v = np.asarray([x for x in values if x is not None and math.isfinite(x)], dtype=float)
    if len(v) == 0:
        return {"n": 0, "median": None, "p90": None, "mean": None, "max": None}
    return {
        "n": int(len(v)),
        "median": round(float(np.median(v)), 3),
        "p90": round(float(np.percentile(v, 90)), 3),
        "mean": round(float(v.mean()), 3),
        "max": round(float(v.max()), 3),
    }


def exif_orientation_rotation(orientation: int | None) -> np.ndarray:
    """3x3 matrix taking a raw-pixel camera's cam_from_world rotation to the EXIF-upright image's.

    The pipeline transposes photos upright (ImageOps.exif_transpose) before SfM, the GT model was
    built on the raw pixels. Rows are the upright camera's x (right) and y (down) axes written in
    the raw camera's axes; mirrored orientations (2, 4, 5, 7) are not handled and give identity.
    """
    if orientation == 3:      # rotate 180
        return np.diag([-1.0, -1.0, 1.0])
    if orientation == 6:      # rotate 90 clockwise to display: right = raw up, down = raw right
        return np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    if orientation == 8:      # rotate 90 counter-clockwise: right = raw down, down = raw left
        return np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    return np.eye(3)


def rotation_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Angle of the rotation a @ b.T, in degrees."""
    c = (np.trace(np.asarray(a) @ np.asarray(b).T) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True) -> tuple[float, np.ndarray, np.ndarray]:
    """Least-squares similarity dst ~ s * R @ src + t (any dimension), as (s, R, t)."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    mu_s, mu_d = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_s, dst - mu_d
    u, sv, vt = np.linalg.svd(xd.T @ xs / len(src))
    d = np.eye(src.shape[1])
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1, -1] = -1.0
    r = u @ d @ vt
    var = (xs**2).sum() / len(src)
    s = float(np.trace(np.diag(sv) @ d) / var) if with_scale and var > 0 else 1.0
    return s, r, mu_d - s * r @ mu_s


def mean_rotation(rotations: np.ndarray) -> np.ndarray:
    """Chordal L2 mean of rotation matrices (projection of their sum onto SO(3))."""
    u, _, vt = np.linalg.svd(np.asarray(rotations, dtype=float).sum(axis=0))
    d = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    return u @ d @ vt


def sim3_position_errors(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray]:
    """(scale, per-point error in dst units) of the Sim3 (Umeyama) from src onto dst centres, refit once
    without the points whose error exceeds max(3 x median, 1)."""
    src, dst = np.asarray(src, dtype=float), np.asarray(dst, dtype=float)
    s, r, t = umeyama(src, dst)
    err = np.linalg.norm(s * src @ r.T + t - dst, axis=1)
    keep = err <= max(3.0 * float(np.median(err)), 1.0)
    if 3 <= keep.sum() < len(src):
        s, r, t = umeyama(src[keep], dst[keep])
        err = np.linalg.norm(s * src @ r.T + t - dst, axis=1)
    return s, err


def relative_pose_errors(
    est_rots: np.ndarray, est_centers: np.ndarray, gt_rots: np.ndarray, gt_centers: np.ndarray
) -> dict[str, Any]:
    """GPS-independent accuracy of one SfM model against GT (both cam_from_world rotations and centres).

    Positions: Sim3 (Umeyama) from estimated to GT centres, refit once without cameras whose error
    exceeds max(3 x median, 1 m), then errors of all cameras in GT units (metres for GT-ENU).
    Rotations: per camera after the best global rotation (chordal mean of R_gt^T R_est), plus
    pairwise relative rotation errors, which need no alignment at all.
    """
    est_rots, gt_rots = np.asarray(est_rots, dtype=float), np.asarray(gt_rots, dtype=float)
    est_centers, gt_centers = np.asarray(est_centers, dtype=float), np.asarray(gt_centers, dtype=float)
    n = len(est_rots)
    if n < 3:
        return {"n": n, "message": "fewer than 3 cameras"}
    s, err = sim3_position_errors(est_centers, gt_centers)
    # est cam_from_world = gt cam_from_world @ align  =>  align ~ mean(gt^T @ est)
    align = mean_rotation(np.einsum("nji,njk->nik", gt_rots, est_rots))
    rot_err = [rotation_angle_deg(e, g @ align) for e, g in zip(est_rots, gt_rots)]
    i, j = np.triu_indices(n, k=1)
    pair_err = [rotation_angle_deg(est_rots[a] @ est_rots[b].T, gt_rots[a] @ gt_rots[b].T) for a, b in zip(i, j)]
    return {
        "n": n,
        "sim3_scale": round(s, 6),
        "position_error_m": summarize(err),
        "rotation_error_deg": summarize(rot_err),
        "pairwise_rotation_error_deg": summarize(pair_err),
        "fraction_within_5deg_5m": round(float(np.mean((np.array(rot_err) < 5) & (err < 5))), 3),
        "per_camera": {"position_error_m": err.tolist(), "rotation_error_deg": rot_err},
    }


def assign_users(n: int, num_users: int, rng: np.random.Generator) -> list[int]:
    """Balanced random user index in [0, num_users) for each of n photos."""
    order = rng.permutation(n)
    users = [0] * n
    for k, i in enumerate(order):
        users[int(i)] = k % num_users
    return users


def simulate_tags(
    east: float,
    north: float,
    heading_deg: float,
    enu: LocalENU,
    rng: np.random.Generator,
    gps_noise_m: float,
    heading_noise_deg: float,
) -> tuple[float, float, float]:
    """(lat, lon, compass heading) of a phone at (east, north) looking along heading_deg, with noise."""
    e = east + rng.normal(0.0, gps_noise_m)
    n = north + rng.normal(0.0, gps_noise_m)
    lat, lon, _ = enu.to_geodetic(e, n, 0.0)
    return lat, lon, (heading_deg + rng.normal(0.0, heading_noise_deg)) % 360.0


def tag_jpeg(data: bytes, *, lat: float | None = None, lon: float | None = None,
             heading_deg: float | None = None) -> bytes:
    """Copy of JPEG bytes with the given GPS position and/or compass heading written into its EXIF.

    Tags left as None keep whatever the photo had. Pixels and every other EXIF field (orientation
    above all) are untouched: only the APP1 segment is replaced. The tags themselves come from
    crowd3d.exif.build_exif_bytes, i.e. exactly what the synthetic data and tests write.
    """
    built = piexif.load(build_exif_bytes(lat=lat, lon=lon, heading_deg=heading_deg, make=None, model=None))
    try:
        ex = piexif.load(data)
    except Exception:  # no or broken EXIF: start empty
        ex = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    gps = dict(ex.get("GPS") or {})
    if lat is not None and lon is not None:
        for tag in (piexif.GPSIFD.GPSLatitude, piexif.GPSIFD.GPSLatitudeRef,
                    piexif.GPSIFD.GPSLongitude, piexif.GPSIFD.GPSLongitudeRef):
            gps.pop(tag, None)
    gps.update(built["GPS"])
    ex["GPS"] = gps
    try:
        blob = piexif.dump(ex)
    except Exception:  # odd vendor tags piexif cannot re-serialise: keep orientation + GPS only
        zeroth = {k: v for k, v in (ex.get("0th") or {}).items() if k == piexif.ImageIFD.Orientation}
        blob = piexif.dump({"0th": zeroth, "Exif": {}, "GPS": gps, "1st": {}, "thumbnail": None})
    out = io.BytesIO()
    piexif.insert(blob, data, out)
    return out.getvalue()


def sector_of_en(east: float, north: float, num_sectors: int = 8) -> int:
    """Coverage sector of a camera at (east, north) metres from the landmark centre."""
    return sector_for_bearing(heading_from_en(east, north), num_sectors)


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    """Area centroid of a simple polygon (k, 2); closing vertex optional."""
    p = np.asarray(poly, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    return ((p + q) * cross[:, None]).sum(axis=0) / (6.0 * area)


def distance_to_outline(points: np.ndarray, poly: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For 2-D points: distance to the closed polygon outline and the closest outline point."""
    pts = np.asarray(points, dtype=float)
    a = np.asarray(poly, dtype=float)
    b = np.roll(a, -1, axis=0)
    ab = b - a
    t = ((pts[:, None, :] - a[None]) * ab[None]).sum(axis=2) / np.maximum((ab**2).sum(axis=1), 1e-12)[None]
    closest = a[None] + np.clip(t, 0.0, 1.0)[..., None] * ab[None]
    d = np.linalg.norm(pts[:, None, :] - closest, axis=2)
    k = d.argmin(axis=1)
    rows = np.arange(len(pts))
    return d[rows, k], closest[rows, k]


def fit_outline(points: np.ndarray, poly: np.ndarray, init_shift: np.ndarray, iters: int = 60,
                keep: float = 0.8) -> dict[str, Any]:
    """Trimmed 2-D rigid ICP of wall points onto a footprint outline, starting from a translation.

    Returns the rotation (deg, counter-clockwise) and translation that move the points onto the
    outline, and the median point-to-outline distance before and after.
    """
    pts = np.asarray(points, dtype=float)
    before = distance_to_outline(pts, poly)[0]
    r, t = np.eye(2), np.asarray(init_shift, dtype=float)
    for _ in range(iters):
        moved = pts @ r.T + t
        d, target = distance_to_outline(moved, poly)
        sel = d <= np.quantile(d, keep)
        _, dr, dt = umeyama(moved[sel], target[sel], with_scale=False)
        r, t = dr @ r, dr @ t + dt
        if np.linalg.norm(dt) < 1e-4 and abs(math.atan2(dr[1, 0], dr[0, 0])) < 1e-6:
            break
    after = distance_to_outline(pts @ r.T + t, poly)[0]
    return {
        "rotation_deg": round(math.degrees(math.atan2(r[1, 0], r[0, 0])), 2),
        "translation_m": [round(float(x), 2) for x in t],
        "shift_m": round(float(np.linalg.norm(t)), 2),
        "median_distance_before_m": round(float(np.median(before)), 2),
        "median_distance_after_m": round(float(np.median(after)), 2),
        "num_points": int(len(pts)),
    }


def best_outline_fit(points: np.ndarray, poly: np.ndarray, top_shift: np.ndarray) -> dict[str, Any]:
    """fit_outline from two starts, keeping the closer fit: no shift (GT-ENU is already within GPS
    error) and the shift putting the top structure on the footprint centroid (a central lantern or
    dome). The latter alone lands in a wrong local minimum when the tallest part is at one end of an
    elongated building (Aachen's west tower)."""
    fits = [fit_outline(points, poly, np.zeros(2)), fit_outline(points, poly, top_shift)]
    best = min(fits, key=lambda f: f["median_distance_after_m"])
    best["start"] = "zero" if best is fits[0] else "top_structure"
    best["other_start"] = {k: fits[1 if best is fits[0] else 0][k]
                           for k in ("rotation_deg", "shift_m", "median_distance_after_m")}
    return best


# ---------------------------------------------------------------- ground truth


@dataclass
class GtImage:
    name: str
    rot: np.ndarray        # cam_from_world of the EXIF-upright image, GT SfM frame
    center: np.ndarray     # camera centre, GT SfM frame
    meta: PhotoMeta        # EXIF exactly as the pipeline reads it
    focal_ratio: float     # focal length / longer image side (unchanged by rotation and uniform scaling)


@dataclass
class GroundTruth:
    images: list[GtImage]
    points: np.ndarray     # (n, 3) GT SfM frame
    summary: dict[str, Any]


@dataclass
class GtFrame:
    """GT georeferenced into the landmark ENU frame."""

    transform: np.ndarray                         # 4x4 similarity, GT SfM -> ENU
    report: dict[str, Any]
    poses: dict[str, dict[str, Any]] = field(default_factory=dict)   # name -> enu, heading_deg, rot
    # (rotation deg counter-clockwise, shift) moving GT-ENU walls onto a footprint outline, if one was given
    footprint_fix: tuple[float, np.ndarray] | None = None

    @property
    def rotation(self) -> np.ndarray:
        m = self.transform[:3, :3]
        return m / np.cbrt(np.linalg.det(m))

    def footprint_pose(self, name: str) -> tuple[np.ndarray, float] | None:
        """Horizontal position and heading of a GT camera in the footprint-corrected frame."""
        if self.footprint_fix is None:
            return None
        theta, shift = self.footprint_fix
        c, s = math.cos(math.radians(theta)), math.sin(math.radians(theta))
        xy = np.array([[c, -s], [s, c]]) @ self.poses[name]["enu"][:2] + shift
        return xy, (self.poses[name]["heading_deg"] - theta) % 360.0


def load_ground_truth(scene_dir: Path) -> GroundTruth:
    import pycolmap

    rec = pycolmap.Reconstruction(str(scene_dir / "sparse"))
    images = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        if not im.has_pose:
            continue
        path = scene_dir / "images" / im.name
        if not path.is_file():
            continue
        meta = read_photo_meta(path.read_bytes())
        rot = exif_orientation_rotation(meta.orientation) @ im.cam_from_world().rotation.matrix()
        cam = rec.cameras[im.camera_id]
        images.append(GtImage(im.name, rot, np.asarray(im.projection_center(), dtype=float), meta,
                              float(cam.params[0]) / max(cam.width, cam.height)))
    points = np.array([p.xyz for p in rec.points3D.values()], dtype=float).reshape(-1, 3)
    models = Counter(c.model.name for c in rec.cameras.values())
    summary = {
        "num_images": len(images),
        "num_points": int(len(points)),
        "num_cameras": len(rec.cameras),
        "camera_models": dict(models),
        "mean_reprojection_error_px": round(float(rec.compute_mean_reprojection_error()), 3),
        "focal_over_max_side": summarize([im.focal_ratio for im in images]),
        "exif_orientation": dict(Counter(str(g.meta.orientation) for g in images)),
        "real_gps": sum(g.meta.has_gps for g in images),
        "real_heading": sum(g.meta.heading_deg is not None for g in images),
        "magnetic_heading": sum(g.meta.heading_ref == "M" for g in images),
    }
    return GroundTruth(images, points, summary)


def fit_gt_frame(
    rots: np.ndarray,
    centers: np.ndarray,
    gps_en: np.ndarray,
    headings_deg: np.ndarray,
    *,
    inlier_m: float = GT_GPS_INLIER_M,
    gps_sigma_m: float = GT_GPS_SIGMA_M,
    camera_height_m: float = 1.6,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Similarity from a GT SfM frame to ENU, from the cameras' real GPS (NaN rows: no GPS) and compass.

    Same recipe as reconstruct.georegister, adapted to a large GT model:
    gravity from the right vectors of all cameras (phones are held without roll; no camera-centre
    term, since crowd photos include elevated viewpoints), a robust 2-D similarity on the GPS-tagged
    cameras only, its yaw fused with the compass by inverse variance, and z so that the median camera
    sits at camera_height_m.
    """
    rots, centers = np.asarray(rots, dtype=float), np.asarray(centers, dtype=float)
    gps_en, headings_deg = np.asarray(gps_en, dtype=float), np.asarray(headings_deg, dtype=float)
    ups, rights, views = -rots[:, 1, :], rots[:, 0, :], rots[:, 2, :]
    up = estimate_up(ups, rights)
    mean_up = ups.mean(axis=0) / np.linalg.norm(ups.mean(axis=0))
    up_with_centres = estimate_up(ups, rights, centers)
    rot_g = rotation_to_z(up)
    c, d = centers @ rot_g.T, views @ rot_g.T

    has_gps = np.isfinite(gps_en).all(axis=1)
    fit = robust_similarity_2d(c[has_gps, :2], gps_en[has_gps], inlier_m, min_pair_distance=gps_sigma_m)
    if fit is None:
        raise ValueError("the GPS-tagged GT cameras do not determine a 2-D similarity")
    scale_gps, yaw_gps, _, inliers = fit
    g = gps_en[has_gps][inliers]
    sigma_gps = gps_sigma_m / max(math.sqrt(float(((g - g.mean(axis=0)) ** 2).sum())), 1e-9)
    has_h = np.isfinite(headings_deg)
    compass = compass_yaw(d[has_h], headings_deg[has_h].tolist()) if has_h.any() else None
    yaw, weight = yaw_gps, 0.0
    if compass:
        sigma_compass = math.radians(params.DEFAULTS.compass_sigma_deg) / math.sqrt(compass[1])
        weight = sigma_gps**2 / (sigma_gps**2 + sigma_compass**2)
        yaw += weight * math.radians(angle_diff_deg(math.degrees(compass[0]), math.degrees(yaw_gps)))
    src = c[has_gps][inliers, :2]
    u = np.exp(1j * yaw) * (src[:, 0] + 1j * src[:, 1])
    w = g[:, 0] + 1j * g[:, 1]
    uc, wc = u - u.mean(), w - w.mean()
    scale = float((np.conj(uc) @ wc).real / (np.abs(uc) ** 2).sum())
    shift = w.mean() - scale * u.mean()
    z_shift = camera_height_m - scale * float(np.median(c[:, 2]))
    transform = similarity_matrix(scale, rotation_z(yaw) @ rot_g, np.array([shift.real, shift.imag, z_shift]))

    resid = np.linalg.norm(apply_similarity(transform, centers[has_gps])[:, :2] - gps_en[has_gps], axis=1)
    rolls = np.degrees(np.arcsin(np.clip(rights @ up, -1.0, 1.0)))
    report = {
        "num_cameras": int(len(rots)),
        "num_gps": int(has_gps.sum()),
        "num_gps_inliers": int(inliers.sum()),
        "num_compass": int(has_h.sum()),
        "num_compass_used": compass[1] if compass else 0,
        "inlier_threshold_m": inlier_m,
        "gps_sigma_m": gps_sigma_m,
        "scale_m_per_unit": round(scale, 5),
        "scale_gps_only_m_per_unit": round(scale_gps, 5),
        "scale_and_yaw_sigma_from_gps": round(sigma_gps, 4),     # relative scale = yaw in rad
        "yaw_gps_deg": round(math.degrees(yaw_gps), 2),
        "yaw_compass_deg": round(math.degrees(compass[0]), 2) if compass else None,
        "yaw_compass_minus_gps_deg": (round(angle_diff_deg(math.degrees(compass[0]), math.degrees(yaw_gps)), 2)
                                      if compass else None),
        "yaw_fused_deg": round(math.degrees(yaw), 2),
        "compass_weight": round(weight, 3),
        "gps_residual_m": summarize(resid),
        "gps_residual_inliers_rmse_m": round(float(np.sqrt((resid[inliers] ** 2).mean())), 3),
        "up_rights_vs_rights_and_centres_deg": round(math.degrees(math.acos(min(1.0, float(up @ up_with_centres)))), 2),
        "up_rights_vs_mean_up_deg": round(math.degrees(math.acos(min(1.0, float(up @ mean_up)))), 2),
        "camera_roll_abs_deg": summarize(np.abs(rolls)),
    }
    return transform, report


def georeference_gt(gt: GroundTruth, lat0: float, lon0: float, camera_height_m: float) -> GtFrame:
    enu = LocalENU(lat0, lon0)
    gps = np.full((len(gt.images), 2), np.nan)
    headings = np.full(len(gt.images), np.nan)
    for k, im in enumerate(gt.images):
        if im.meta.has_gps:
            gps[k] = enu.to_enu(im.meta.lat, im.meta.lon)[:2]
        if im.meta.heading_deg is not None:
            headings[k] = im.meta.heading_deg
    rots = np.array([im.rot for im in gt.images])
    centers = np.array([im.center for im in gt.images])
    transform, report = fit_gt_frame(rots, centers, gps, headings, camera_height_m=camera_height_m)
    frame = GtFrame(transform, report)
    pos = apply_similarity(transform, centers)
    r_enu = frame.rotation
    resid = {}
    for k, im in enumerate(gt.images):
        view = r_enu @ im.rot[2]
        frame.poses[im.name] = {
            "enu": pos[k],
            "heading_deg": heading_from_en(view[0], view[1]),
            "rot": im.rot @ r_enu.T,          # cam_from_ENU
            "focal_ratio": im.focal_ratio,
        }
        if np.isfinite(gps[k]).all():
            resid[im.name] = {
                "gps_residual_m": round(float(np.linalg.norm(pos[k, :2] - gps[k])), 2),
                "compass_minus_gt_heading_deg": (round(angle_diff_deg(im.meta.heading_deg,
                                                                      frame.poses[im.name]["heading_deg"]), 2)
                                                 if im.meta.heading_deg is not None else None),
            }
    report["per_photo"] = resid
    report["compass_minus_gt_heading_deg"] = summarize(
        [abs(v["compass_minus_gt_heading_deg"]) for v in resid.values() if v["compass_minus_gt_heading_deg"] is not None])
    return frame


def centre_check(gt: GroundTruth, frame: GtFrame, footprint_en: np.ndarray | None) -> dict[str, Any]:
    """Where the GT model puts the building relative to --lat/--lon (all in ENU metres)."""
    pts = apply_similarity(frame.transform, gt.points)
    median_xy = np.median(pts[:, :2], axis=0)
    top = pts[pts[:, 2] >= np.quantile(pts[:, 2], 1.0 - TOP_STRUCTURE_FRACTION)]
    top_xy = np.median(top[:, :2], axis=0)
    cams = np.array([p["enu"] for p in frame.poses.values()])
    dist = np.linalg.norm(cams[:, :2], axis=1)
    out: dict[str, Any] = {
        "points_horizontal_median_en": [round(float(x), 2) for x in median_xy],
        "points_horizontal_median_distance_m": round(float(np.linalg.norm(median_xy)), 2),
        "top_structure_en": [round(float(x), 2) for x in top_xy],
        "top_structure_distance_m": round(float(np.linalg.norm(top_xy)), 2),
        "top_structure_height_m": round(float(np.median(top[:, 2])), 2),
        "gt_camera_distance_m": summarize(dist),
        "gt_camera_height_m": summarize(cams[:, 2]),
        "gt_camera_sectors": {SECTOR_NAMES[s]: n for s, n in sorted(Counter(
            sector_of_en(x, y) for x, y in cams[:, :2]).items())},
    }
    if footprint_en is not None:
        centroid = polygon_centroid(footprint_en)
        radius = float(np.max(np.linalg.norm(footprint_en - centroid, axis=1)))
        near = np.linalg.norm(pts[:, :2] - top_xy, axis=1) <= radius + 5.0
        band = (pts[:, 2] >= FOOTPRINT_WALL_Z_M[0]) & (pts[:, 2] <= FOOTPRINT_WALL_Z_M[1])
        walls = pts[near & band, :2]
        if len(walls) > 20000:
            walls = walls[np.random.default_rng(0).choice(len(walls), 20000, replace=False)]
        out["footprint"] = {
            "centroid_en": [round(float(x), 2) for x in centroid],
            "given_centre_to_footprint_centroid_m": round(float(np.linalg.norm(centroid)), 2),
            "top_structure_to_footprint_centroid_m": round(float(np.linalg.norm(top_xy - centroid)), 2),
            "footprint_radius_m": round(radius, 2),
            "wall_fit": best_outline_fit(walls, footprint_en, centroid - top_xy) if len(walls) >= 50 else None,
        }
    return out


def sensor_check(gt: GroundTruth, frame: GtFrame, enu: LocalENU) -> dict[str, Any]:
    """Real EXIF GPS and compass against the footprint-corrected GT: the least circular sensor accuracy."""
    gps_err, compass_err = [], []
    for im in gt.images:
        fp = frame.footprint_pose(im.name)
        if fp is None:
            continue
        if im.meta.has_gps:
            gps_err.append(float(np.linalg.norm(np.array(enu.to_enu(im.meta.lat, im.meta.lon)[:2]) - fp[0])))
        if im.meta.heading_deg is not None:
            compass_err.append(angle_diff_deg(im.meta.heading_deg, fp[1]))
    return {
        "real_gps_error_m": summarize(gps_err),
        "real_compass_abs_error_deg": summarize(np.abs(compass_err)),
        "real_compass_signed_median_deg": round(float(np.median(compass_err)), 2) if compass_err else None,
    }


def load_footprint(path: Path, enu: LocalENU) -> np.ndarray:
    """First polygon ring (lon, lat) found in a GeoJSON / Nominatim JSON file, in ENU metres."""

    def find(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "Polygon":
                return obj["coordinates"][0]
            for v in obj.values():
                ring = find(v)
                if ring:
                    return ring
        elif isinstance(obj, list):
            for v in obj:
                ring = find(v)
                if ring:
                    return ring
        return None

    ring = find(json.loads(path.read_text(encoding="utf-8")))
    if not ring:
        raise ValueError(f"no Polygon in {path}")
    en = np.array([enu.to_enu(lat, lon)[:2] for lon, lat in ring])
    return en[:-1] if np.allclose(en[0], en[-1]) else en


# ---------------------------------------------------------------- uploads


@dataclass
class Upload:
    name: str
    data: bytes
    user: str
    lat: float | None
    lon: float | None
    heading_deg: float | None
    gps_source: str            # "real" | "simulated" | "none"
    heading_source: str
    exif_sector: int | None = None
    gt_sector: int | None = None
    stage: int = 0


def build_uploads(
    gt: GroundTruth,
    frame: GtFrame,
    mode: str,
    lat0: float,
    lon0: float,
    *,
    num_users: int,
    gps_noise_m: float,
    heading_noise_deg: float,
    max_images: int | None,
    seed: int,
    scene_dir: Path,
) -> list[Upload]:
    """Photos to upload in this mode, with the tags they will carry and their pseudo users."""
    rng = np.random.default_rng(seed)
    enu = LocalENU(lat0, lon0)
    images = [im for im in gt.images if mode == "augmented" or im.meta.has_gps]
    rank = {int(i): k for k, i in enumerate(rng.permutation(len(images)))}
    # --max-images keeps the photos with real GPS first, then a seeded random choice.
    keep = sorted(range(len(images)), key=lambda i: (not images[i].meta.has_gps, rank[i]))
    images = [images[i] for i in keep[:max_images or None]]
    images.sort(key=lambda im: im.name)
    users = assign_users(len(images), num_users, rng)
    tag_rng = np.random.default_rng(seed + 1)
    uploads = []
    for im, u in zip(images, users):
        data = (scene_dir / "images" / im.name).read_bytes()
        pose = frame.poses[im.name]
        lat, lon, heading = im.meta.lat, im.meta.lon, im.meta.heading_deg
        gps_src = "real" if im.meta.has_gps else "none"
        head_src = "real" if heading is not None else "none"
        # Draw the noise for every photo so a photo's simulated tags do not depend on the others.
        s_lat, s_lon, s_head = simulate_tags(pose["enu"][0], pose["enu"][1], pose["heading_deg"], enu, tag_rng,
                                             gps_noise_m, heading_noise_deg)
        if mode == "augmented" and (gps_src == "none" or head_src == "none"):
            new_gps = gps_src == "none"
            new_head = head_src == "none"
            data = tag_jpeg(data, lat=s_lat if new_gps else None, lon=s_lon if new_gps else None,
                            heading_deg=s_head if new_head else None)
            if new_gps:
                lat, lon, gps_src = s_lat, s_lon, "simulated"
            if new_head:
                heading, head_src = s_head, "simulated"
        up = Upload(im.name, data, f"user-{u + 1}", lat, lon, heading, gps_src, head_src)
        if lat is not None:
            up.exif_sector = sector_for_bearing(local_bearing_distance(lat0, lon0, lat, lon)[0], 8)
        up.gt_sector = sector_of_en(pose["enu"][0], pose["enu"][1])
        uploads.append(up)
    return uploads


def split_stages(uploads: list[Upload]) -> int | None:
    """Stage 1 = the sector with most GPS-tagged uploads (lowest index on ties); sets Upload.stage."""
    counts = Counter(u.exif_sector for u in uploads if u.exif_sector is not None)
    first = min(counts, key=lambda s: (-counts[s], s)) if counts else None
    for u in uploads:
        u.stage = 1 if u.exif_sector == first else 2
    return first


# ---------------------------------------------------------------- pipeline run


def load_job_model(work_dir: Path):
    """The model a ColmapReconstructor job kept (most registered images among work_dir/sparse/*)."""
    import pycolmap

    best = None
    for sub in sorted((work_dir / "sparse").glob("*")):
        if (sub / "images.bin").is_file() or (sub / "images.txt").is_file():
            rec = pycolmap.Reconstruction(str(sub))
            if best is None or rec.num_reg_images() > best.num_reg_images():
                best = rec
    return best


def model_accuracy(model, settings: Settings, uploads_by_id: dict[str, Upload], frame: GtFrame,
                   enu: LocalENU) -> dict[str, Any]:
    """Errors of one pipeline model against GT-ENU: absolute (via its GPS georegistration) and relative.

    Also counts gross errors (heading > GROSS_HEADING_DEG or position > GROSS_POSITION_M off GT-ENU) and
    how a GT-free check would catch them: a camera that disagrees with its own GPS tag by more than the
    pipeline's GPS inlier threshold AND with its compass by more than GROSS_HEADING_DEG.
    """
    cams = model.meta.get("cameras") or {}
    horiz, head_gt, head_compass, head_sim, est_sector_wrong, horiz_fp, head_fp = [], [], [], [], [], [], []
    gross, flagged = set(), set()
    for pid, cam in cams.items():
        up = uploads_by_id.get(pid)
        if up is None:
            continue
        gt = frame.poses[up.name]
        e = np.asarray(cam["enu"], dtype=float)
        horiz.append(float(np.linalg.norm(e[:2] - gt["enu"][:2])))
        head_gt.append(abs(angle_diff_deg(cam["heading_deg"], gt["heading_deg"])))
        if head_gt[-1] > GROSS_HEADING_DEG or horiz[-1] > GROSS_POSITION_M:
            gross.add(pid)
        if up.heading_deg is not None:
            off = abs(angle_diff_deg(cam["heading_deg"], up.heading_deg))
            (head_compass if up.heading_source == "real" else head_sim).append(off)
            gps_off = np.linalg.norm(e[:2] - np.array(enu.to_enu(up.lat, up.lon)[:2]))
            if off > GROSS_HEADING_DEG and gps_off > GPS_INLIER_M:
                flagged.add(pid)
        est_sector_wrong.append(sector_of_en(e[0], e[1]) != up.gt_sector)
        fp = frame.footprint_pose(up.name)
        if fp is not None:
            horiz_fp.append(float(np.linalg.norm(e[:2] - fp[0])))
            head_fp.append(abs(angle_diff_deg(cam["heading_deg"], fp[1])))
    out: dict[str, Any] = {
        "horizontal_error_vs_gt_enu_m": summarize(horiz),
        "heading_error_vs_gt_enu_deg": summarize(head_gt),
        "heading_error_vs_real_compass_deg": summarize(head_compass),
        "heading_error_vs_simulated_compass_deg": summarize(head_sim),
        "horizontal_error_vs_footprint_gt_m": summarize(horiz_fp),
        "heading_error_vs_footprint_gt_deg": summarize(head_fp),
        "estimated_sector_wrong_rate": round(float(np.mean(est_sector_wrong)), 3) if est_sector_wrong else None,
        "gross_errors": {
            "cameras": len(gross),
            "flagged_by_gps_and_compass_check": len(flagged),
            "flagged_and_gross": len(gross & flagged),
            "names": sorted(uploads_by_id[p].name for p in gross),
        },
    }
    rec = load_job_model(settings.work_dir / model.job_id)
    if rec is None:
        out["relative"] = {"message": "job work dir has no sparse model"}
        return out
    est_r, est_c, gt_r, gt_c, focal_err = [], [], [], [], []
    for im in rec.images.values():
        up = uploads_by_id.get(photo_id_from_image_name(im.name))
        if not im.has_pose or up is None:
            continue
        est_r.append(im.cam_from_world().rotation.matrix())
        est_c.append(im.projection_center())
        gt_r.append(frame.poses[up.name]["rot"])
        gt_c.append(frame.poses[up.name]["enu"])
        cam = rec.cameras[im.camera_id]
        focal_err.append(abs(cam.params[0] / max(cam.width, cam.height) / frame.poses[up.name]["focal_ratio"] - 1.0))
    rel = relative_pose_errors(np.array(est_r), np.array(est_c), np.array(gt_r), np.array(gt_c))
    rel.pop("per_camera", None)
    out["relative"] = rel
    out["colmap_cameras"] = {
        "num_cameras": len(rec.cameras),
        "num_images": rec.num_reg_images(),
        "focal_relative_error_vs_gt": summarize(focal_err),
    }
    return out


def run_mode(
    mode: str,
    gt: GroundTruth,
    frame: GtFrame,
    args: argparse.Namespace,
    data_dir: Path,
    upload_dir: Path,
) -> dict[str, Any]:
    from fastapi.testclient import TestClient

    from crowd3d.api import create_app
    from crowd3d.coverage import compute_coverage
    from crowd3d.ingest import ingest_upload
    from crowd3d.records import JOB_FULL, PHOTO_ACCEPTED

    t_mode = time.monotonic()
    uploads = build_uploads(gt, frame, mode, args.lat, args.lon, num_users=args.users,
                            gps_noise_m=args.gps_noise_m, heading_noise_deg=args.heading_noise_deg,
                            max_images=args.max_images, seed=args.seed, scene_dir=args.scene_dir)
    first_sector = split_stages(uploads)
    upload_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for u in uploads:
        if u.gps_source == "simulated" or u.heading_source == "simulated":
            (upload_dir / u.name).write_bytes(u.data)
        manifest.append({k: getattr(u, k) for k in ("name", "user", "lat", "lon", "heading_deg", "gps_source",
                                                     "heading_source", "exif_sector", "gt_sector", "stage")})
    (upload_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    # Work dirs are kept: relative errors are measured on each job's sparse model.
    settings = Settings(data_dir=data_dir, reconstructor="colmap", inline_jobs=True, keep_work_dirs=True)
    overrides = {k: v for k, v in (("processing_max_side_px", args.max_side),
                                   ("full_min_sectors", args.full_min_sectors)) if v is not None}
    overrides = {**getattr(args, "config_settings", {}), **overrides}
    settings = params.settings_with(settings, overrides)
    landmark_id = args.landmark_id
    uploads_by_id: dict[str, Upload] = {}
    rows: dict[str, dict[str, Any]] = {}
    stages = []
    stage_of_job: dict[str, int] = {}
    with TestClient(create_app(settings)) as client:
        r = client.post("/api/landmarks", json={"id": landmark_id, "name": args.name, "lat": args.lat,
                                                "lon": args.lon, "radius_m": args.radius})
        if r.status_code != 201:
            raise RuntimeError(f"creating landmark failed: HTTP {r.status_code} {r.text[:300]}")
        app = client.app
        store, runner = app.state.store, app.state.runner
        landmark = store.get_landmark(landmark_id)
        for stage in (1, 2):
            batch = [u for u in uploads if u.stage == stage]
            t = time.monotonic()
            outcomes = []
            for u in batch:
                o = ingest_upload(store, settings, filename=u.name, data=u.data, user_id=u.user, landmark=landmark)
                outcomes.append(o)
                if o.photo is not None:
                    uploads_by_id[o.photo.id] = u
                rows[u.name] = {
                    "name": u.name, "user": u.user, "stage": stage, "gps": u.gps_source, "heading": u.heading_source,
                    "tag_lat": u.lat, "tag_lon": u.lon, "tag_heading_deg": u.heading_deg,
                    "accepted": o.accepted, "reasons": o.reasons, "warnings": o.warnings,
                    "photo_id": o.photo.id if o.photo else None,
                    "distance_m": round(o.photo.distance_m, 1) if o.photo and o.photo.distance_m is not None else None,
                    "exif_sector": u.exif_sector, "gt_sector": u.gt_sector,
                }
            t_ingest = time.monotonic() - t
            log.info("[%s] stage %d: %d photos ingested (%d accepted) in %.1fs; planning jobs", mode, stage,
                     len(batch), sum(o.accepted for o in outcomes), t_ingest)
            jobs = runner.schedule(landmark_id) if any(o.accepted for o in outcomes) else []
            stage_of_job.update((j.id, stage) for j in jobs)
            detail = client.get(f"/api/landmarks/{landmark_id}").json()
            stages.append({
                "stage": stage,
                "description": (f"sector {SECTOR_NAMES.get(first_sector)} only" if stage == 1
                                else "all remaining photos"),
                "uploaded": len(batch),
                "accepted": sum(o.accepted for o in outcomes),
                "rejected_by_reason": dict(Counter(code for o in outcomes for code in o.reasons)),
                "ingest_s": round(t_ingest, 1),
                "jobs_s": round(time.monotonic() - t - t_ingest, 1),
                "jobs": [{"kind": j.kind, "sector": SECTOR_NAMES.get(j.sector), "status": j.status,
                          "num_photos": len(j.photo_ids), "error": j.error,
                          "seconds": round(j.finished_at - j.started_at, 1) if j.finished_at and j.started_at else None}
                         for j in jobs],
                "sectors_after": {s["label"]: {"count": s["count"], "state": s["state"],
                                               "has_face_model": s["has_face_model"]} for s in detail["sectors"]},
                "full_ready": detail["full_ready"],
                # What the gate says with product defaults, whatever --full-min-sectors was.
                "full_ready_with_product_defaults": compute_coverage(
                    store.list_photos(landmark_id, PHOTO_ACCEPTED), Settings(data_dir=data_dir)).full_ready,
                "has_full_model": detail["latest_full_model"] is not None,
            })
            for j in jobs:
                log.info("[%s]   job %s sector=%s photos=%d -> %s %s", mode, j.kind, SECTOR_NAMES.get(j.sector),
                         len(j.photo_ids), j.status, j.error or "")
        detail = client.get(f"/api/landmarks/{landmark_id}").json()
        if detail["latest_full_model"] is not None:
            r = client.get(detail["latest_full_model"]["url"])
            if r.status_code != 200:
                raise RuntimeError(f"full model PLY not served: HTTP {r.status_code}")
        models = []
        for m in store.list_models(landmark_id):
            meta = m.meta
            job = store.get_job(m.job_id)
            entry = {
                "id": m.id, "kind": m.kind, "sector": SECTOR_NAMES.get(m.sector), "stage": stage_of_job.get(m.job_id),
                "num_photos": m.num_photos, "num_registered": meta.get("num_registered"),
                "num_points": meta.get("num_points"),
                "mean_reprojection_error_px": _round(meta.get("mean_reprojection_error_px"), 3),
                "georegistered": meta.get("georegistered"),
                "alignment_rmse_m": _round(meta.get("alignment_rmse_m"), 2),
                "heading_mae_deg": _round(meta.get("heading_mae_deg"), 2),
                "timings_s": meta.get("timings_s"),
                "job_seconds": round(job.finished_at - job.started_at, 1),
                "message": meta.get("message"),
                "published": meta.get("published", True),
                "qa_reasons": (meta.get("qa") or {}).get("reasons"),
                "flagged_cameras": sum(bool(c.get("flags")) for c in (meta.get("cameras") or {}).values()),
            }
            entry.update(model_accuracy(m, settings, uploads_by_id, frame, LocalENU(args.lat, args.lon)))
            models.append(entry)
        failed = [{"kind": j.kind, "sector": SECTOR_NAMES.get(j.sector), "num_photos": len(j.photo_ids),
                   "error": j.error} for j in store.list_jobs(landmark_id) if j.status == "failed"]
        has_full = any(m.kind == JOB_FULL for m in store.list_models(landmark_id))

    # Per photo, the estimate the UI shows (newest model containing it, full models first).
    shown = {p["id"]: p for p in detail["photos"]}
    horiz, head_gt, head_compass, horiz_fp = [], [], [], []
    for row in rows.values():
        gt_pose = frame.poses[row["name"]]
        row["gt_enu"] = [round(float(x), 2) for x in gt_pose["enu"]]
        row["gt_heading_deg"] = round(gt_pose["heading_deg"], 1)
        p = shown.get(row["photo_id"])
        if p is None or p["estimated_enu"] is None:
            continue
        row["est_enu"] = [round(x, 2) for x in p["estimated_enu"]]
        row["est_heading_deg"] = p["estimated_heading_deg"]
        horiz.append(float(np.linalg.norm(np.subtract(p["estimated_enu"][:2], gt_pose["enu"][:2]))))
        head_gt.append(abs(angle_diff_deg(p["estimated_heading_deg"], gt_pose["heading_deg"])))
        if row["heading"] == "real":
            head_compass.append(abs(angle_diff_deg(p["estimated_heading_deg"], p["heading_deg"])))
        fp = frame.footprint_pose(row["name"])
        if fp is not None:
            horiz_fp.append(float(np.linalg.norm(np.subtract(p["estimated_enu"][:2], fp[0]))))
    accepted = [r for r in rows.values() if r["accepted"]]
    tagged = [u for u in uploads if u.exif_sector is not None]

    def wrong_rate(items: list[Upload]) -> dict[str, Any]:
        wrong = [u.exif_sector != u.gt_sector for u in items]
        return {"n": len(wrong), "wrong": int(sum(wrong)), "rate": round(float(np.mean(wrong)), 3) if wrong else None}

    footprint = load_footprint(args.footprint, LocalENU(args.lat, args.lon)) if args.footprint else None
    topdown = upload_dir.parent / "topdown.png"
    draw_topdown(topdown, gt, frame, list(rows.values()), args.lat, args.lon, args.radius, footprint)
    exif_err = []
    enu = LocalENU(args.lat, args.lon)
    for u in tagged:
        tag = np.array(enu.to_enu(u.lat, u.lon)[:2])
        exif_err.append((u.gps_source, float(np.linalg.norm(tag - frame.poses[u.name]["enu"][:2]))))
        fp = frame.footprint_pose(u.name)
        if fp is not None:
            exif_err.append((f"{u.gps_source}_vs_footprint_gt", float(np.linalg.norm(tag - fp[0]))))
    return {
        "mode": mode,
        "settings": {"processing_max_side_px": settings.processing_max_side_px,
                     "full_min_sectors": settings.full_min_sectors, "full_min_photos": settings.full_min_photos,
                     "face_min_photos": settings.face_min_photos, "radius_m": args.radius,
                     "overridden": overrides},
        "inputs": {
            "photos": len(uploads),
            "gps": dict(Counter(u.gps_source for u in uploads)),
            "heading": dict(Counter(u.heading_source for u in uploads)),
            "users": dict(sorted(Counter(u.user for u in uploads).items())),
            "stage1_sector": SECTOR_NAMES.get(first_sector),
            "exif_sectors": {SECTOR_NAMES[s]: n for s, n in sorted(Counter(u.exif_sector for u in tagged).items())},
            "upload_copies_dir": str(upload_dir),
        },
        "uploads": {
            "accepted": len(accepted),
            "rejected": len(rows) - len(accepted),
            "rejected_by_reason": dict(Counter(c for r in rows.values() for c in r["reasons"])),
            "warnings": dict(Counter(c for r in rows.values() for c in r["warnings"])),
            "distance_m": summarize([r["distance_m"] for r in rows.values()]),
        },
        "exif_gps_error_vs_gt_enu_m": {src: summarize([e for s, e in exif_err if s == src])
                                       for src in sorted({s for s, _ in exif_err})},
        "sector_misassignment": {
            "all": wrong_rate(tagged),
            "real_gps": wrong_rate([u for u in tagged if u.gps_source == "real"]),
            "simulated_gps": wrong_rate([u for u in tagged if u.gps_source == "simulated"]),
        },
        "stages": stages,
        "photo_estimates_as_shown": {
            "with_estimate": len(horiz),
            "accepted": len(accepted),
            "horizontal_error_vs_gt_enu_m": summarize(horiz),
            "heading_error_vs_gt_enu_deg": summarize(head_gt),
            "heading_error_vs_real_compass_deg": summarize(head_compass),
            "horizontal_error_vs_footprint_gt_m": summarize(horiz_fp),
        },
        "final_sectors": {s["label"]: {"count": s["count"], "state": s["state"], "has_face_model": s["has_face_model"]}
                          for s in detail["sectors"]},
        "has_full_model": has_full,
        "failed_jobs": failed,
        "models": models,
        "photos": list(rows.values()),
        "data_dir": str(data_dir),
        "topdown_png": str(topdown),
        "landmark_id": landmark_id,
        "elapsed_s": round(time.monotonic() - t_mode, 1),
    }


def _round(v: Any, nd: int) -> Any:
    return round(float(v), nd) if isinstance(v, (int, float)) and math.isfinite(v) else None


def draw_topdown(path: Path, gt: GroundTruth, frame: GtFrame, rows: list[dict[str, Any]], lat0: float, lon0: float,
                 radius_m: float, footprint_en: np.ndarray | None, size: int = 900) -> None:
    """North-up plan view in ENU: GT points (grey, darker = higher), sector rays, footprint (black),
    GT cameras (blue), uploaded GPS tags (green = real, orange = simulated) and estimates (red, joined
    to their GT camera)."""
    from PIL import Image, ImageDraw

    enu = LocalENU(lat0, lon0)
    cams = np.array([p["enu"] for p in frame.poses.values()])
    pts = apply_similarity(frame.transform, gt.points)
    extent = [np.abs(cams[:, :2]).max(), 30.0]
    half = float(np.max(extent)) * 1.15
    sc = size / (2 * half)

    def px(e: float, n: float) -> tuple[float, float]:
        return size / 2 + e * sc, size / 2 - n * sc

    img = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        dr.line([px(0, 0), px(half * 2 * math.sin(a), half * 2 * math.cos(a))], fill=(225, 225, 235))
    dr.ellipse([*px(-radius_m, radius_m), *px(radius_m, -radius_m)], outline=(200, 200, 220))
    sel = pts if len(pts) <= 80000 else pts[np.random.default_rng(0).choice(len(pts), 80000, replace=False)]
    for x, y, z in sel:
        X, Y = px(x, y)
        if 0 <= X < size and 0 <= Y < size:
            g = int(np.clip(210 - 4 * z, 60, 210))
            img.putpixel((int(X), int(Y)), (g, g, g))
    if footprint_en is not None:
        dr.polygon([px(*p) for p in footprint_en], outline=(0, 0, 0))
    by_name = {r["name"]: r for r in rows}
    for name, pose in frame.poses.items():
        X, Y = px(*pose["enu"][:2])
        h = math.radians(pose["heading_deg"])
        dr.line([(X, Y), (X + 12 * math.sin(h), Y - 12 * math.cos(h))], fill=(40, 90, 220))
        dr.ellipse([X - 3, Y - 3, X + 3, Y + 3], fill=(40, 90, 220))
        r = by_name.get(name)
        if r is None:
            continue
        if r.get("est_enu"):
            E, N = px(*r["est_enu"][:2])
            dr.line([(X, Y), (E, N)], fill=(230, 60, 60))
            dr.ellipse([E - 2, N - 2, E + 2, N + 2], fill=(230, 60, 60))
    for r in rows:
        if r.get("tag_lat") is None:
            continue
        G = px(*enu.to_enu(r["tag_lat"], r["tag_lon"])[:2])
        color = (0, 150, 0) if r["gps"] == "real" else (240, 150, 0)
        dr.rectangle([G[0] - 3, G[1] - 3, G[0] + 3, G[1] + 3], outline=color, width=2)
    c = px(0, 0)
    dr.line([c[0] - 8, c[1], c[0] + 8, c[1]], fill=(0, 0, 0), width=2)
    dr.line([c[0], c[1] - 8, c[0], c[1] + 8], fill=(0, 0, 0), width=2)
    dr.text((8, 8), f"N up, {2 * half:.0f} m across; + = given centre; circle = {radius_m:.0f} m radius", fill=(0, 0, 0))
    img.save(path)


# ---------------------------------------------------------------- crowd protocol: inputs


@dataclass
class CrowdPhoto:
    key: str                         # stable id in the test set: Commons file name, MegaDepth-X image, synthetic file
    path: Path
    title: str | None = None
    author: str | None = None
    licence: str | None = None
    meta: PhotoMeta | None = None    # the file's own EXIF, read by run_crowd


@dataclass
class CrowdBuilding:
    slug: str
    name: str
    lat: float
    lon: float
    radius_m: float
    photos: list[CrowdPhoto]
    outline_path: Path | None = None
    # {"kind": "colmap", "sparse_dir": str, "images": {key: {"name": GT image name, "orientation": int | None}}}
    # or {"kind": "poses", "poses": {key: {"rot": 3x3 cam_from_world, "center": [x, y, z]}}}
    gt: dict[str, Any] | None = None
    upload_all: bool = False         # also upload photos without EXIF GPS (synthetic validation cases)


@dataclass
class UploadRequest:
    """Photos one upload request carries; jobs are planned after each request."""

    items: list[tuple[CrowdPhoto, str]]      # (photo, pseudo user)
    label: str


def pseudo_user(author: str | None, captured_at: str | None) -> str:
    """The author as user id (whitespace collapsed, API length limit), else one unknown user per capture day."""
    if author and author.strip():
        return " ".join(author.split())[:MAX_USER_ID_CHARS]
    day = captured_at[:10].replace(":", "-") if captured_at else "undated"
    return f"unknown-{day}"


def _capture_key(p: CrowdPhoto) -> tuple:
    captured = p.meta.captured_at if p.meta else None
    return (captured is None, captured or "", p.key)


def plan_arrival(photos: list[CrowdPhoto], arrival: str = "session",
                 max_per_request: int = MAX_FILES_PER_UPLOAD) -> list[UploadRequest]:
    """Upload requests in crowd arrival order (EXIF DateTimeOriginal; undated photos last, by key).

    session: one request per (author, capture day), in the order those sessions ended (people upload
    after their visit), split into requests of at most max_per_request photos. photo: one request per
    photo. single: everything at once (one planning round; the cheapest protocol).
    """
    if arrival not in ARRIVALS:
        raise ValueError(f"unknown arrival {arrival!r}; expected one of {ARRIVALS}")
    ordered = sorted(photos, key=_capture_key)

    def user(p: CrowdPhoto) -> str:
        return pseudo_user(p.author, p.meta.captured_at if p.meta else None)

    if arrival == "photo":
        return [UploadRequest([(p, user(p))], p.key) for p in ordered]
    if arrival == "single":
        return [UploadRequest([(p, user(p)) for p in ordered[k:k + max_per_request]], "all")
                for k in range(0, len(ordered), max_per_request)]
    sessions: dict[tuple[str, str], list[CrowdPhoto]] = {}
    for p in ordered:
        captured = p.meta.captured_at if p.meta else None
        sessions.setdefault((user(p), captured[:10] if captured else "undated"), []).append(p)
    keyed = sorted(sessions.items(), key=lambda kv: (_capture_key(kv[1][-1])[:2], kv[0]))
    requests = []
    for (u, day), items in keyed:
        for k in range(0, len(items), max_per_request):
            requests.append(UploadRequest([(p, u) for p in items[k:k + max_per_request]], f"{u} {day}"))
    return requests


def building_from_commons_dir(folder: Path, *, slug: str, name: str, lat: float, lon: float, radius_m: float,
                              outline_path: Path | None = None, gt: dict[str, Any] | None = None) -> CrowdBuilding:
    """A scripts/fetch_commons_landmark.py output folder (manifest.json + photos) as a CrowdBuilding."""
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    photos = [CrowdPhoto(key=e["file_name"], path=folder / e["pixels"]["file"], title=e.get("title"),
                         author=e.get("author"), licence=e.get("licence")) for e in manifest["photos"]]
    return CrowdBuilding(slug, name, lat, lon, radius_m, photos, outline_path, gt)


def building_from_spec(spec: dict[str, Any]) -> CrowdBuilding:
    """The resolved building spec scripts/benchmark.py writes (absolute paths)."""
    photos = [CrowdPhoto(key=p["key"], path=Path(p["path"]), title=p.get("title"), author=p.get("author"),
                         licence=p.get("licence")) for p in spec["photos"]]
    return CrowdBuilding(spec["slug"], spec.get("name") or spec["slug"], float(spec["lat"]), float(spec["lon"]),
                         float(spec["radius_m"]), photos,
                         Path(spec["outline_path"]) if spec.get("outline_path") else None,
                         spec.get("gt"), bool(spec.get("upload_all", False)))


def catalog_entry(catalog: Path, slug: str) -> dict[str, Any]:
    """One entry of eval/landmarks.json by slug."""
    data = json.loads(catalog.read_text(encoding="utf-8"))
    entries = data["landmarks"] if isinstance(data, dict) else data
    for entry in entries:
        if entry.get("slug") == slug:
            return entry
    raise SystemExit(f"no landmark {slug!r} in {catalog}")


def synthetic_gt_poses(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Reference poses of a crowd3d.synthetic dataset: file -> cam_from_world rows (right, down, forward) and centre."""
    from crowd3d.synthetic import camera_axes

    poses = {}
    for p in manifest["photos"]:
        fwd, right, up = camera_axes(p["true_heading_deg"], p["true_pitch_deg"])
        poses[p["file"]] = {"rot": np.array([right, -up, fwd]).tolist(), "center": list(p["true_enu"])}
    return poses


def load_gt_poses(gt: dict[str, Any]) -> dict[str, tuple[np.ndarray, np.ndarray, str]]:
    """key -> (cam_from_world of the EXIF-upright image, camera centre, reference frame id).

    A COLMAP reference may have several components (separate models, e.g. the two faces of a gate);
    each is its own frame, so cameras are only ever compared within one. Image names are used exactly as
    given (MegaDepth-X names can contain sub-folders).
    """
    if gt["kind"] == "poses":
        return {k: (np.asarray(v["rot"], dtype=float), np.asarray(v["center"], dtype=float), "truth")
                for k, v in gt["poses"].items()}
    if gt["kind"] != "colmap":
        raise ValueError(f"unknown reference kind {gt['kind']!r}")
    import pycolmap

    components = gt.get("components") or [{"component": "0", "sparse_dir": gt["sparse_dir"], "images": gt["images"]}]
    poses: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}
    for comp in components:
        rec = pycolmap.Reconstruction(str(comp["sparse_dir"]))
        by_name = {im.name: im for im in rec.images.values() if im.has_pose}
        for key, entry in comp["images"].items():
            im = by_name.get(entry["name"])
            if im is None or key in poses:
                continue
            rot = exif_orientation_rotation(entry.get("orientation")) @ im.cam_from_world().rotation.matrix()
            poses[key] = (rot, np.asarray(im.projection_center(), dtype=float), str(comp.get("component", "0")))
    return poses


# ---------------------------------------------------------------- crowd protocol: model checks


def read_ply_points(path: Path) -> np.ndarray:
    """(n, 3) vertex positions of an ASCII PLY as reconstruct.write_ply writes it."""
    with open(path, encoding="ascii") as f:
        count = 0
        for line in f:
            if line.startswith("element vertex"):
                count = int(line.split()[2])
            if line.strip() == "end_header":
                break
        if count == 0:
            return np.zeros((0, 3))
        return np.loadtxt(f, usecols=(0, 1, 2), max_rows=count, ndmin=2)


def wall_points(points: np.ndarray, outline: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal positions of model points that may be the building's walls, plus the top-structure shift.

    Walls: points in the FOOTPRINT_WALL_Z_M height band (the pipeline puts the ground at z = 0) within the
    outline's radius plus max(WALL_RING_MIN_M, WALL_RING_FRACTION x radius) of its centroid, a ring wide
    enough for a model that is off by tens of metres. The shift moves the median of the highest
    TOP_STRUCTURE_FRACTION of the points in that ring onto the centroid (best_outline_fit's second start).
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    centroid = polygon_centroid(outline)
    radius = float(np.max(np.linalg.norm(outline - centroid, axis=1)))
    near = np.linalg.norm(pts[:, :2] - centroid, axis=1) <= radius + max(WALL_RING_MIN_M, WALL_RING_FRACTION * radius)
    band = (pts[:, 2] >= FOOTPRINT_WALL_Z_M[0]) & (pts[:, 2] <= FOOTPRINT_WALL_Z_M[1])
    shift = np.zeros(2)
    if near.any():
        ring = pts[near]
        top = ring[ring[:, 2] >= np.quantile(ring[:, 2], 1.0 - TOP_STRUCTURE_FRACTION)]
        shift = centroid - np.median(top[:, :2], axis=0)
    return pts[near & band, :2], shift


def pick_outline_fit(fits: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The fit_outline result that measures the correction a model needs, from fits by start name.

    A facade model of a symmetric building fits several faces almost equally well (an octagon every
    45 degrees) and can slide along itself. Fits whose median wall distance is within
    max(OUTLINE_TIE_FACTOR x, OUTLINE_TIE_M +) the best one's count as equally good; of those the one with
    the smallest rotation (within OUTLINE_AMBIGUOUS_DEG; then the smallest shift) is reported, the most
    plausible one and a lower bound of the error.
    "ambiguous" says the equally good fits disagree by more than OUTLINE_AMBIGUOUS_DEG or OUTLINE_AMBIGUOUS_M.
    """
    best = min(f["median_distance_after_m"] for f in fits.values())
    limit = max(OUTLINE_TIE_FACTOR * best, best + OUTLINE_TIE_M)
    tied = {k: f for k, f in fits.items() if f["median_distance_after_m"] <= limit}
    least_turn = min(abs(f["rotation_deg"]) for f in tied.values())
    name = min((k for k, f in tied.items() if abs(f["rotation_deg"]) <= least_turn + OUTLINE_AMBIGUOUS_DEG),
               key=lambda k: (tied[k]["shift_m"], abs(tied[k]["rotation_deg"])))
    chosen = dict(tied[name])
    rots = [f["rotation_deg"] for f in tied.values()]
    shifts = [np.asarray(f["translation_m"]) for f in tied.values()]
    chosen["start"] = name
    chosen["ambiguous"] = bool(max(abs(angle_diff_deg(a, b)) for a in rots for b in rots) > OUTLINE_AMBIGUOUS_DEG
                               or max(float(np.linalg.norm(a - b)) for a in shifts for b in shifts)
                               > OUTLINE_AMBIGUOUS_M)
    chosen["alternatives"] = {k: {x: f[x] for x in ("rotation_deg", "shift_m", "median_distance_after_m")}
                              for k, f in fits.items() if k != name}
    return chosen


def outline_check(points: np.ndarray, outline: np.ndarray, seed: int = 0) -> dict[str, Any]:
    """Fit a model's wall points to the outline (from no shift and from the top-structure start):
    the rotation and shift its georegistration needs (pick_outline_fit)."""
    walls, top_shift = wall_points(points, outline)
    if len(walls) < MIN_WALL_POINTS:
        return {"num_points": int(len(walls)), "fit": None}
    if len(walls) > MAX_WALL_POINTS:
        walls = walls[np.random.default_rng(seed).choice(len(walls), MAX_WALL_POINTS, replace=False)]
    fits = {"zero": fit_outline(walls, outline, np.zeros(2)), "top_structure": fit_outline(walls, outline, top_shift)}
    return {"num_points": int(len(walls)), "fit": pick_outline_fit(fits)}


QA_REASON_CODES = (   # (code, substring of qa.assess's reason text)
    ("registered_fraction", "photos registered"),
    ("points", " points (need"),
    ("reprojection", "reprojection error"),
    ("twins", "disagree with both their GPS and compass"),
    ("flagged", "cameras are implausible"),
    ("compass", "off the compass"),
    ("centre_offset", "the point cloud sits"),
    ("coverage", "face-ready sides"),
)


def qa_reason_code(reason: str) -> str:
    """Short code of one qa.assess reason ("other" when it matches none)."""
    return next((code for code, text in QA_REASON_CODES if text in reason), "other")


def gt_model_check(cameras: dict[str, dict[str, Any]], est_rots: dict[str, np.ndarray],
                   gt_poses: dict[str, tuple], photo_key: dict[str, str], published: bool) -> dict[str, Any]:
    """Relative accuracy of one model's cameras against the reference poses, in metres of the model's frame.

    cameras: the model's ENU camera entries (photo id -> enu, flags); est_rots: photo id -> cam_from_world in
    the model's SfM frame (empty when the reconstructor keeps no COLMAP model: positions only). Per
    reference frame with at least 3 of the model's cameras, the reference is mapped onto the model
    (relative_pose_errors), so position errors are in the model's (metric ENU) units. A camera off by more
    than GROSS_HEADING_DEG or GROSS_POSITION_M is gross; shown = published and without a flag.
    """
    rows = [(pid, photo_key[pid]) for pid in cameras if photo_key.get(pid) in gt_poses]
    frames: dict[str, list[tuple[str, str]]] = {}
    for pid, key in rows:
        frames.setdefault(gt_poses[key][2] if len(gt_poses[key]) > 2 else "0", []).append((pid, key))
    out: dict[str, Any] = {"n_with_reference": len(rows), "frames": {}}
    per_camera: dict[str, dict[str, Any]] = {}
    for frame_id, members in sorted(frames.items()):
        if len(members) < 3:
            out["frames"][frame_id] = {"n": len(members), "message": "fewer than 3 cameras: not measured"}
            continue
        gt_r = np.array([gt_poses[k][0] for _, k in members])
        gt_c = np.array([gt_poses[k][1] for _, k in members])
        est_c = np.array([cameras[pid]["enu"] for pid, _ in members], dtype=float)
        if all(pid in est_rots for pid, _ in members):
            rel = relative_pose_errors(gt_r, gt_c, np.array([est_rots[pid] for pid, _ in members]), est_c)
            per = rel.pop("per_camera")
            rot_err, pos_err = per["rotation_error_deg"], list(per["position_error_m"])
            out["frames"][frame_id] = {k: rel[k] for k in ("n", "sim3_scale", "pairwise_rotation_error_deg")}
        else:
            scale, err = sim3_position_errors(gt_c, est_c)
            rot_err, pos_err = [None] * len(members), err.tolist()
            out["frames"][frame_id] = {"n": len(members), "sim3_scale": round(scale, 6),
                                       "message": "no rotations: positions only"}
        for (pid, key), r, e in zip(members, rot_err, pos_err):
            flagged = bool(cameras[pid].get("flags"))
            gross = (r is not None and r > GROSS_HEADING_DEG) or e > GROSS_POSITION_M
            per_camera[key] = {"rotation_error_deg": None if r is None else round(float(r), 3),
                               "position_error_m": round(float(e), 3), "flagged": flagged, "gross": gross,
                               "frame": frame_id}
    gross_keys = sorted(k for k, c in per_camera.items() if c["gross"])
    out.update({
        "n": len(per_camera),
        "rotation_error_deg": summarize([c["rotation_error_deg"] for c in per_camera.values()]),
        "position_error_m": summarize([c["position_error_m"] for c in per_camera.values()]),
        "gross": len(gross_keys),
        "gross_flagged": sum(per_camera[k]["flagged"] for k in gross_keys),
        "gross_shown": sum(published and not per_camera[k]["flagged"] for k in gross_keys),
        "gross_keys": gross_keys,
        "per_camera": per_camera,
    })
    return out


def prune_work_dir(job_dir: Path, keep_images: bool = False) -> None:
    """Drop a finished job's database, other mapping attempts and (unless the run stylizes, which reads the
    photos through them) its image links; keep sparse/ (the model)."""
    for name in ("database.db", "database.db-shm", "database.db-wal"):
        (job_dir / name).unlink(missing_ok=True)
    for name in ("attempts",) if keep_images else ("attempts", "images"):
        shutil.rmtree(job_dir / name, ignore_errors=True)


def latest_published(models: list) -> dict[tuple[str, int | None], Any]:
    """(kind, sector) -> the newest published ModelRecord: what the landmark page shows."""
    latest: dict[tuple[str, int | None], Any] = {}
    for m in sorted(models, key=lambda m: (m.created_at, m.id)):
        if m.meta.get("published", True):
            latest[(m.kind, m.sector)] = m
    return latest


def _median(values) -> float | None:
    v = [float(x) for x in values if x is not None and math.isfinite(x)]
    return round(float(np.median(v)), 3) if v else None


def crowd_metrics(funnel: dict[str, Any], coverage, jobs: list, models: list[dict[str, Any]],
                  latest_ids: set[str], shown: dict[str, Any], wall_time_s: float, has_gt: bool) -> dict[str, Any]:
    """The METRICS values of one building run (None where a metric cannot be measured)."""
    by_job = {m["job_id"]: m for m in models}
    latest = [m for m in models if m["id"] in latest_ids]
    # Latest attempt per target: a failed reconstruction registered nothing.
    last_job: dict[tuple[str, Any], Any] = {}
    for j in sorted(jobs, key=lambda j: (j.created_at, j.id)):
        last_job[(j.kind, j.sector)] = j
    reg = inp = 0
    for j in last_job.values():
        inp += len(j.photo_ids)
        m = by_job.get(j.id)
        reg += (m["num_registered"] or 0) if m else 0
    cams = sum(m["num_cameras"] for m in latest)
    fits = [m["outline"]["fit"] for m in latest if (m.get("outline") or {}).get("fit")]
    gt_cams = [c for m in latest for c in ((m.get("gt") or {}).get("per_camera") or {}).values()]
    metrics = {
        "photos_listed": funnel["listed"],
        "photos_uploaded": funnel["uploaded"],
        "photos_accepted": funnel["accepted"],
        "acceptance_rate": round(funnel["accepted"] / funnel["uploaded"], 4) if funnel["uploaded"] else None,
        "sectors_with_photos": sum(s.count > 0 for s in coverage.sectors),
        "sectors_face_ready": len(coverage.face_ready),
        "jobs_run": len(jobs),
        "jobs_failed_sfm": sum(j.status == "failed" and j.id not in by_job for j in jobs),
        "models_published": sum(bool(m["published"]) for m in models),
        "models_qa_rejected": sum(not m["published"] for m in models),
        "facades_published": sum(m["kind"] == "face" for m in latest),
        "full_model_published": int(any(m["kind"] == "full" for m in latest)),
        "registration_rate": round(reg / inp, 4) if inp else None,
        "estimate_coverage": round(shown["with_estimate"] / shown["accepted"], 4) if shown["accepted"] else None,
        "flagged_camera_fraction": round(sum(m["flagged_cameras"] for m in latest) / cams, 4) if cams else None,
        "outline_rotation_abs_deg": _median(abs(f["rotation_deg"]) for f in fits),
        "outline_shift_m": _median(f["shift_m"] for f in fits),
        "outline_residual_m": _median(f["median_distance_after_m"] for f in fits),
        "outline_models": len(fits),
        "gps_residual_median_m": shown["gps_residual_m"]["median"],
        "compass_error_median_deg": shown["compass_error_deg"]["median"],
        "compass_photos": shown["compass_error_deg"]["n"],
        "gt_cameras": len(gt_cams) if has_gt else None,
        "gt_rotation_error_median_deg": _median(c["rotation_error_deg"] for c in gt_cams) if has_gt else None,
        "gt_position_error_median_m": _median(c["position_error_m"] for c in gt_cams) if has_gt else None,
        "gt_gross_cameras": sum(c["gross"] for c in gt_cams) if has_gt else None,
        "gt_gross_shown": sum(c["gross"] and not c["flagged"] for c in gt_cams) if has_gt else None,
        "wall_time_s": round(wall_time_s, 1),
        "jobs_time_s": round(sum((j.finished_at or 0) - (j.started_at or 0) for j in jobs if j.started_at), 1),
        **{name: None for name in STYLE_METRICS},       # style_metrics() fills them when the run stylizes
    }
    assert set(metrics) == set(METRIC_BY_NAME), set(metrics) ^ set(METRIC_BY_NAME)
    return metrics


# ---------------------------------------------------------------- crowd protocol: stylized models


def _stylize_modules():
    """(scripts/stylize_model.py, crowd3d.stylize): the stylization runs exactly as the CLI's --job-dir mode."""
    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import stylize_model

    from crowd3d import stylize

    return stylize_model, stylize


def _rot2(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def snap_candidate(points: np.ndarray, footprint: np.ndarray, holes: list[np.ndarray] | None = None) -> dict | None:
    """Diagnostic: the rigid fit crowd3d.stylize.snap_to_footprint converges to before its limits are applied
    (same wall-band selection, subsample, starts, trimming and choice among the starts; no scale step). A
    rejected snap is reset to the identity, so this is what tells a symmetric alias (a far rotation that fits)
    from a model that is simply off by more than the limits."""
    _, st = _stylize_modules()
    p = st.DEFAULT_PARAMS
    pts = np.asarray(points, float)
    fp = st.ensure_ccw(footprint)
    rings = [fp] + list(holes or [])
    centroid = st.polygon_centroid(fp)
    radius = float(np.max(np.linalg.norm(fp - centroid, axis=1)))
    band = (pts[:, 2] >= 2.0) & (pts[:, 2] <= 20.0)
    near = np.linalg.norm(pts[:, :2] - centroid, axis=1) <= radius + p.snap_max_shift_m
    xy = pts[band & near, :2]
    if len(xy) < 50:
        return None
    if len(xy) > 6000:
        xy = xy[np.random.default_rng(0).choice(len(xy), 6000, replace=False)]
    lookup = st._OutlineLookup(fp, radius + 2 * p.snap_max_shift_m, holes=holes)
    best = None
    for shift0 in (np.zeros(2), centroid - np.median(xy, axis=0)):
        for rot0 in (-10.0, 0.0, 10.0):
            r = _rot2(rot0)
            t = shift0 + centroid - r @ centroid
            for _ in range(100):
                moved = xy @ r.T + t
                target = lookup.closest(moved)
                d = np.linalg.norm(moved - target, axis=1)
                keep = d <= np.quantile(d, 0.7)
                dr, dt = st._rigid_2d(moved[keep], target[keep])
                r, t = dr @ r, dr @ t + dt
                if np.linalg.norm(dt) < 1e-4 and abs(math.atan2(dr[1, 0], dr[0, 0])) < 1e-6:
                    break
            after = float(np.median(np.linalg.norm(xy @ r.T + t - lookup.closest(xy @ r.T + t), axis=1)))
            if best is None or after < best[0] - 1e-6:
                best = (after, r, t)
    _, r, t = best
    after = float(np.median(st.distance_to_rings(xy @ r.T + t, rings)[0]))
    return {"rotation_deg": round(math.degrees(math.atan2(r[1, 0], r[0, 0])), 2),
            "shift_m": round(float(np.linalg.norm(t)), 2), "median_after_m": round(after, 3)}


def camera_correction(src_xy: np.ndarray, dst_xy: np.ndarray, anchor: np.ndarray,
                      walls_xy: np.ndarray | None = None) -> dict[str, Any]:
    """The 2-D similarity moving src cameras onto dst (about the ENU origin, as a footprint snap), refit once
    without cameras off by more than max(3 x median, 5 m): its rotation, scale and shift, how far it moves the
    anchor (where the model's walls are) and, given the model's wall points, how far it moves them (median and
    90th percentile: where the photos land on the walls; a turn or a rescale moves the ends of a facade more
    than its middle). Judged at the walls, not at the cameras: they stand 40-250 m away, so a 1 deg turn about
    the walls moves them 0.7-4 m while the photos move far less on the walls."""
    src, dst = np.asarray(src_xy, float), np.asarray(dst_xy, float)

    def near_median(xy: np.ndarray) -> np.ndarray:     # a camera placed kilometres away would decide the fit
        d = np.linalg.norm(xy - np.median(xy, axis=0), axis=1)
        return d <= max(5.0 * float(np.median(d)), 50.0)

    keep = near_median(src) & near_median(dst)
    if keep.sum() < 3:
        keep[:] = True
    s, r, t = umeyama(src[keep], dst[keep], with_scale=True)
    err = np.linalg.norm(s * src @ r.T + t - dst, axis=1)
    keep = err <= max(3.0 * float(np.median(err[keep])), 5.0)
    if keep.sum() >= 3:
        s, r, t = umeyama(src[keep], dst[keep], with_scale=True)
        err = np.linalg.norm(s * src @ r.T + t - dst, axis=1)
    else:
        keep[:] = True
    a = np.asarray(anchor, float)
    out = {"rotation_deg": round(math.degrees(math.atan2(r[1, 0], r[0, 0])), 2), "scale": round(s, 4),
           "shift_m": round(float(np.linalg.norm(t)), 2),
           "at_walls_m": round(float(np.linalg.norm(s * r @ a + t - a)), 2),
           "cameras_used": int(keep.sum()), "residual_median_m": round(float(np.median(err[keep])), 2)}
    if walls_xy is not None and len(walls_xy):
        w = np.asarray(walls_xy, float).reshape(-1, 2)
        moved = np.linalg.norm(s * w @ r.T + t - w, axis=1)
        out.update(walls_median_m=round(float(np.median(moved)), 2), walls_p90_m=round(float(np.percentile(moved, 90)), 2))
    return out


def outline_corrected_reference(gt: dict[str, Any], metas: dict[str, PhotoMeta], outline: np.ndarray,
                                holes: list[np.ndarray], enu: LocalENU, max_points: int = 400_000) -> dict[str, Any]:
    """Reference camera centres (horizontal ENU) per photo key, moved onto the OSM outline: the independent
    answer to "is this snap right" (REAL_WORLD_EVAL 2.4, outline-corrected reference).

    Per reference component: GT-ENU from the mapped test photos' own EXIF GPS and compass (fit_gt_frame,
    product default parameters), then the reference's wall points fitted to the outline (best_outline_fit,
    rigid), then the bounded scale step of crowd3d.stylize.snap_to_footprint (its similarity ICP about the
    outline centroid, kept when it cuts the median wall distance by 10 %), then the edge-by-edge
    registration_check of the corrected reference itself ("verified"). Synthetic truth poses are ENU already.
    """
    out: dict[str, Any] = {"cameras": {}, "components": {}}
    if gt.get("kind") == "poses":
        for key, v in gt["poses"].items():
            out["cameras"][key] = {"xy": np.asarray(v["center"], float)[:2], "component": "truth", "verified": True}
        out["components"]["truth"] = {"verified": True, "note": "generator truth, ENU already"}
        return out
    import pycolmap

    _, st = _stylize_modules()
    components = gt.get("components") or [{"component": "0", "sparse_dir": gt["sparse_dir"], "images": gt["images"]}]
    rings = [outline] + list(holes)
    for comp in components:
        cid = str(comp.get("component", "0"))
        rec = pycolmap.Reconstruction(str(comp["sparse_dir"]))
        by_name = {im.name: im for im in rec.images.values() if im.has_pose}
        keys, rots, centres, gps, heads = [], [], [], [], []
        for key, entry in comp["images"].items():
            im, meta = by_name.get(entry["name"]), metas.get(key)
            if im is None or meta is None:
                continue
            keys.append(key)
            rots.append(exif_orientation_rotation(entry.get("orientation")) @ im.cam_from_world().rotation.matrix())
            centres.append(np.asarray(im.projection_center(), float))
            gps.append(enu.to_enu(meta.lat, meta.lon)[:2] if meta.has_gps else (math.nan, math.nan))
            heads.append(meta.heading_deg if meta.heading_deg is not None else math.nan)
        report: dict[str, Any] = {"mapped_cameras": len(keys), "verified": False}
        out["components"][cid] = report
        if sum(np.isfinite(g).all() for g in np.asarray(gps, float).reshape(-1, 2)) < 3:
            report["message"] = "fewer than 3 mapped cameras with GPS"
            continue
        try:
            with params.using(params.DEFAULTS):
                transform, frame = fit_gt_frame(np.array(rots), np.array(centres), np.asarray(gps, float),
                                                np.asarray(heads, float), camera_height_m=Settings().camera_height_m)
        except ValueError as e:
            report["message"] = str(e)
            continue
        xyz = np.array([p.xyz for p in rec.points3D.values()], dtype=float).reshape(-1, 3)
        del rec, by_name
        if len(xyz) > max_points:
            xyz = xyz[np.random.default_rng(0).choice(len(xyz), max_points, replace=False)]
        pts = apply_similarity(transform, xyz)
        walls, top_shift = wall_points(pts, outline)
        report["gps_fit"] = {k: frame[k] for k in ("num_gps", "num_gps_inliers", "num_compass_used", "scale_m_per_unit",
                                                   "scale_and_yaw_sigma_from_gps", "yaw_fused_deg")}
        if len(walls) < MIN_WALL_POINTS:
            report["message"] = f"only {len(walls)} reference wall points near the outline"
            continue
        if len(walls) > MAX_WALL_POINTS:
            walls = walls[np.random.default_rng(0).choice(len(walls), MAX_WALL_POINTS, replace=False)]
        fit = best_outline_fit(walls, outline, top_shift)
        r1, t1 = _rot2(fit["rotation_deg"]), np.asarray(fit["translation_m"], float)
        w = walls @ r1.T + t1
        resid = float(np.median(st.distance_to_rings(w, rings)[0]))
        # the bounded scale step: p' = s R (p - c) + c + t about the outline centroid c (GPS scale is the weakest part)
        fp = st.ensure_ccw(outline)
        c = st.polygon_centroid(fp)
        margin = float(np.max(np.linalg.norm(fp - c, axis=1))) + 2 * st.DEFAULT_PARAMS.snap_max_shift_m
        lookup = st._OutlineLookup(fp, margin, holes=holes)
        # (the reference keeps the plain 70 % trim it was built with, so the truth stays the same across versions)
        s2, r2, t2 = st._similarity_icp(w, lookup, c, st.DEFAULT_PARAMS.snap_max_scale, inlier_m=math.inf)
        scaled = float(np.median(st.distance_to_rings(s2 * (w - c) @ r2.T + c + t2, rings)[0]))
        # kept only when it helps, is not stuck at the clamp (neighbouring buildings pulled onto the outline:
        # Brandenburg Gate) and agrees with the GPS scale within 3 sigma
        clamped = abs(math.log(s2)) >= math.log(st.DEFAULT_PARAMS.snap_max_scale) - 1e-3
        sigma = float(frame["scale_and_yaw_sigma_from_gps"])
        use_scale = (scaled < 0.9 * resid and abs(s2 - 1.0) > 1e-3 and not clamped
                     and abs(s2 - 1.0) <= 3.0 * sigma)
        report["scale_step_found"] = {"scale": round(s2, 4), "clamped": clamped, "gps_sigma": round(sigma, 4)}
        if not use_scale:
            s2, r2, t2 = 1.0, np.eye(2), np.zeros(2)
        else:
            resid = scaled

        moved = pts.copy()
        moved[:, :2] = s2 * (pts[:, :2] @ r1.T + t1 - c) @ r2.T + c + t2
        reg = st.registration_check(moved, rings)
        cams = apply_similarity(transform, np.array(centres))
        cams[:, :2] = s2 * (cams[:, :2] @ r1.T + t1 - c) @ r2.T + c + t2
        verified = bool(reg["accepted"]) and resid <= st.DEFAULT_PARAMS.snap_max_residual_m
        report.update(verified=verified, outline_fit={k: fit[k] for k in ("rotation_deg", "shift_m",
                                                                          "median_distance_after_m", "start")},
                      scale_step={"scale": round(s2, 4), "used": bool(use_scale),
                                  "rotation_deg": round(math.degrees(math.atan2(r2[1, 0], r2[0, 0])), 2)},
                      median_wall_distance_m=round(resid, 3),
                      registration={k: reg.get(k) for k in ("accepted", "reason", "voting_edges", "verified_edges",
                                                            "verified_share")})
        for key, xy in zip(keys, cams[:, :2], strict=True):
            out["cameras"].setdefault(key, {"xy": xy, "component": cid, "verified": verified})
    return out


def reference_check(cameras: dict[str, dict[str, Any]], photo_key: dict[str, str], reference: dict[str, Any],
                    snap_meta: dict[str, Any] | None, anchor: np.ndarray,
                    walls_xy: np.ndarray | None = None) -> dict[str, Any]:
    """A model's cameras against the outline-corrected reference, before and after its footprint snap.

    Only unflagged cameras count (flagged ones are withheld from users and may sit kilometres away).
    correction_after = the similarity the snapped model still needs (camera_correction): a passing snap is
    "right" when it turns the model by at most STYLE_RIGHT_ROT_DEG, moves its wall points (walls_xy, snapped;
    else the anchor) by at most STYLE_RIGHT_SHIFT_M (90th percentile) and rescales it by at most
    STYLE_RIGHT_SCALE. correction_before = what the georegistered model needed (what the snap should find).
    error_after_m = the snapped cameras' own distance to their reference positions (reported per model)."""
    _, st = _stylize_modules()
    pairs = [(pid, photo_key[pid]) for pid in cameras
             if photo_key.get(pid) in reference["cameras"] and reference["cameras"][photo_key[pid]]["verified"]
             and not cameras[pid].get("flags")]
    out: dict[str, Any] = {"n": len(pairs)}
    if not pairs:
        out["verdict"] = "no reference"
        return out
    model = np.array([cameras[pid]["enu"] for pid, _ in pairs], dtype=float)
    ref = np.array([reference["cameras"][key]["xy"] for _, key in pairs], dtype=float)
    snap = st.FootprintSnap.from_meta(snap_meta) if snap_meta and snap_meta.get("accepted") else None
    snapped = st.apply_snap_points(snap, model) if snap is not None else model
    before = np.linalg.norm(model[:, :2] - ref, axis=1)
    after = np.linalg.norm(snapped[:, :2] - ref, axis=1)
    out["error_before_m"] = {"median": round(float(np.median(before)), 2), "max": round(float(before.max()), 2)}
    out["error_after_m"] = {"median": round(float(np.median(after)), 2), "max": round(float(after.max()), 2)}
    out["camera_distance_m"] = round(float(np.median(np.linalg.norm(ref - anchor, axis=1))), 1)
    if len(pairs) < STYLE_MIN_REF_CAMERAS:
        out["verdict"] = f"unknown ({len(pairs)} reference camera(s))"
        return out
    unsnapped_anchor, unsnapped_walls = anchor, walls_xy
    if snap is not None:      # where the walls were before the snap moved them onto the anchor
        r = snap.rotation()[:2, :2]
        unsnapped_anchor = np.linalg.solve(snap.scale * r, anchor - snap.shift)
        if walls_xy is not None:
            unsnapped_walls = (np.asarray(walls_xy, float) - snap.shift) @ np.linalg.inv(snap.scale * r).T
    out["correction_before"] = camera_correction(model[:, :2], ref, unsnapped_anchor, unsnapped_walls)
    out["correction_after"] = camera_correction(snapped[:, :2], ref, anchor, walls_xy)
    out["placed_right_before"] = correction_is_right(out["correction_before"])
    out["verdict"] = "right" if correction_is_right(out["correction_after"]) else "wrong"
    return out


def correction_is_right(c: dict[str, Any]) -> bool:
    """A placement whose remaining correction to the reference is within the STYLE_RIGHT_* limits (over the
    wall points when they were given, else at the anchor)."""
    walls = c.get("walls_p90_m", c["at_walls_m"])
    return bool(abs(c["rotation_deg"]) <= STYLE_RIGHT_ROT_DEG and walls <= STYLE_RIGHT_SHIFT_M
                and abs(c["scale"] - 1.0) <= STYLE_RIGHT_SCALE)


def diagnose_gate(rec: dict[str, Any], outline_fit: dict[str, Any] | None) -> dict[str, Any]:
    """Why an eligible model failed the registration gate: too_few_points, single_wall (fewer than 2 outline
    edges carry points, or the verified ones are all parallel: a slide along the wall is unobservable),
    ambiguous (two placements fit about equally well), gps_disagrees (snapped, the cameras would stand nearer
    to or farther from the walls than their own GPS says: points and cameras disagree), shape (the outline
    asks for a rescale beyond the bound), scale_uncertain (a one-sided model whose GPS scale is too uncertain
    for the outline to fix), yaw_unverifiable (a near-circle: translation only, and the registered yaw is not known well
    enough to colour without a check), loose_fit (the snapped walls follow the outline only loosely on
    average: the fit cannot fix the yaw), osm_outline (the reference itself does not sit on the outline),
    georegistration (the correct correction is outside the snap's search: beyond its yaw window or shift
    bound, or the best fit lies at the window edge), symmetric_alias (an unconstrained ICP would converge
    > 10 deg from the needed rotation; the window is what keeps the snap out of it) or residual (right basin,
    but the walls do not verify edge by edge). Evidence: the reference when the model has >=
    STYLE_MIN_REF_CAMERAS reference cameras, else the harness's own outline fit (a lower bound on symmetric
    buildings: "ambiguous")."""
    _, st = _stylize_modules()
    p = st.DEFAULT_PARAMS
    reason = rec["gate"]["reason"] or ""
    snap, cand = rec.get("snap") or {}, rec.get("snap_candidate")
    search = (snap.get("meta") or {}).get("search") or {}
    window = float(search.get("window_deg") or 0.0)
    max_shift = float(search.get("max_shift_m") or p.snap_max_shift_m)
    ref = rec.get("reference") or {}
    truth = ref.get("correction_before")
    source = "reference" if truth else "outline_check" if outline_fit else None
    if truth is None and outline_fit:
        truth = {"rotation_deg": outline_fit.get("rotation_deg"), "shift_m": outline_fit.get("shift_m"),
                 "ambiguous": outline_fit.get("ambiguous")}
    out: dict[str, Any] = {"evidence": source, "needed": truth, "candidate": cand, "window_deg": window,
                           "mode": search.get("mode")}
    if any(x in reason for x in ("too few wall points", "no SfM points", "no wall points within")) or cand is None:
        out["cause"] = "too_few_points"
    elif "all parallel" in reason or "fewer than 2 outline edges" in reason:
        out["cause"] = "single_wall"
    elif "ambiguous" in reason:
        out["cause"] = "ambiguous"
    elif "own GPS" in reason:
        out["cause"] = "gps_disagrees"
    elif "rescale beyond" in reason:
        out["cause"] = "shape"
    elif "scale uncertain" in reason:
        out["cause"] = "scale_uncertain"
    elif "yaw unverifiable" in reason:
        out["cause"] = "yaw_unverifiable"
    elif "too loose to fix the yaw" in reason:
        out["cause"] = "loose_fit"
    elif rec.get("reference_component_verified") is False:
        out["cause"] = "osm_outline"
    elif "edge of the yaw window" in reason or "shift bound" in reason:
        out["cause"] = "georegistration"
    elif truth is None:
        out["cause"] = "unknown"
    elif abs(truth["rotation_deg"]) > window or truth["shift_m"] > max_shift:
        out["cause"] = "georegistration"
    elif abs(angle_diff_deg(cand["rotation_deg"], truth["rotation_deg"])) > 10.0:
        out["cause"] = "symmetric_alias"
    else:
        out["cause"] = "residual"
    out["snap_reason"] = snap.get("reason")
    return out


def _wall_states(model, params_) -> dict[str, Any]:
    """STYLIZED_MODEL 3.3 states: unlocked (mask >= 60 % and >= 3 views), partly unlocked (other coloured walls),
    locked (clay); a view = a camera seeing >= min_wall_visible_fraction of the wall."""
    walls = model.summary["walls"]
    states, lock_reasons = [], Counter()
    for w in walls:
        wi = w["index"]
        views = sum(1 for c in model.observations[wi].cameras if c[1] >= params_.min_wall_visible_fraction)
        if wi in model.facades:
            state = "unlocked" if w["mask_fraction"] >= 0.6 and views >= 3 else "partial"
        else:
            state = "locked"
            lock_reasons[re.sub(r"\d+(\.\d+)?", "N", (w.get("reason") or "?").split(":")[0])] += 1
        states.append({"wall": wi, "state": state, "views": views, "mask": w["mask_fraction"]})
    count = Counter(s["state"] for s in states)
    planes = [w["plane"] for w in walls if w["index"] in model.facades and w.get("plane")]
    return {"total": len(walls), "coloured": count["unlocked"] + count["partial"], "unlocked": count["unlocked"],
            "partial": count["partial"], "locked": count["locked"], "lock_reasons": dict(lock_reasons),
            "plane_offset_max_m": max((max(abs(q["offset_a_m"]), abs(q["offset_b_m"])) for q in planes), default=None),
            "plane_yaw_max_deg": max((abs(q["yaw_deg"]) for q in planes), default=None),
            "coloured_walls": [s for s in states if s["state"] != "locked"]}


def stylize_model_record(m: dict[str, Any], *, data_dir: Path, work_dir: Path, scene_base: dict[str, Any],
                         out_dir: Path, reference: dict[str, Any] | None, photo_key: dict[str, str]) -> dict[str, Any]:
    """Stylize one published model as scripts/stylize_model.py --job-dir does (kept work dir + model JSON),
    write <slug>.glb / <slug>.json under out_dir, and record the gate, walls, QA and reference check."""
    import gzip

    sm, st = _stylize_modules()
    rec: dict[str, Any] = {"model_id": m["id"], "kind": m["kind"], "sector": m["sector"], "request": m["request"],
                           "latest_published": m["latest_published"], "eligible": bool(m["georegistered"]),
                           "yaw_source": m["yaw_source"], "registered": m["num_registered"], "points": m["num_points"]}
    job_dir = work_dir / m["job_id"]
    meta_path = Path(m["ply_path"]).with_suffix(".json")
    if not (job_dir / "sparse" / "0").is_dir() or not meta_path.is_file():
        rec["status"] = "skipped (no kept work dir)"
        return rec
    t0 = time.monotonic()
    try:
        points, colors, cams, source, quality = sm.load_job(job_dir, meta_path)
        scene = st.SceneInput(scene_base["name"], scene_base["footprint"], points, colors, cams, scene_base["facts"],
                              scene_base["attribution"], snap=True, holes=tuple(scene_base["holes"]),
                              point_quality=quality, snap_prior=sm.snap_prior(meta_path))
        model = st.stylize_landmark(scene)
    except (SystemExit, st.StylizeError, ValueError) as e:
        rec.update(status=f"failed: {e}", time_s=round(time.monotonic() - t0, 1))
        return rec
    elapsed = time.monotonic() - t0
    s = model.summary
    out_dir.mkdir(parents=True, exist_ok=True)
    glb_path, json_path = out_dir / f"{scene_base['slug']}.glb", out_dir / f"{scene_base['slug']}.json"
    glb_path.write_bytes(model.glb)
    full = dict(s, source=source, total_s=round(elapsed, 2))
    json_path.write_text(json.dumps(full, indent=1, ensure_ascii=False, default=sm._json_default), encoding="utf-8")
    reg = s["registration"]
    snap = s.get("snap") or {}
    qa = s["qa"]
    honesty_keys = ("detail_pixels_outside_mask", "textured_locked_faces", "locked_faces_claiming_photos",
                    "faces_without_provenance", "off_palette_colors", "fill_regions_not_enclosed",
                    "fill_regions_over_cap", "fill_regions_could_hide", "fill_texels_overlapping_observed",
                    "fill_texels_not_base")
    violations = int(sum(int(qa.get(k) or 0) for k in honesty_keys))
    rec.update(
        status="ok",
        snap={"accepted": snap.get("accepted"), "reason": snap.get("reason"), "rotation_deg": snap.get("rotation_deg"),
              "shift_m": round(float(np.linalg.norm(snap.get("shift_m") or [0.0, 0.0])), 2), "scale": snap.get("scale"),
              "median_wall_distance_before_m": snap.get("median_wall_distance_before_m"),
              "median_wall_distance_after_m": snap.get("median_wall_distance_after_m"),
              "num_points": snap.get("num_points"), "mode": (snap.get("search") or {}).get("mode"),
              "window_deg": (snap.get("search") or {}).get("window_deg"),
              "gps_ratio": (snap.get("gps") or {}).get("ratio"), "meta": snap or None},
        registration={k: reg.get(k) for k in ("accepted", "reason", "voting_edges", "verified_edges", "verified_share",
                                               "verified_direction_spread_deg")},
        gate={"passed": bool(reg.get("accepted")), "reason": reg.get("reason") or ""},
        walls=_wall_states(model, st.DEFAULT_PARAMS),
        observed_wall_area_fraction=s["observed_wall_area_fraction"],
        observed_pixel_area_fraction=s["observed_pixel_area_fraction"],
        roof_observed_area_fraction=s["roof_observed_area_fraction"],
        fill={k: s["observed_fill"][k] for k in ("holes", "area_m2", "holes_over_cap", "holes_could_hide")},
        qa={"honesty_passed": violations == 0, "violations": violations,
            "audit_model_passed": bool((qa.get("audit") or {}).get("passed")),
            "audit_glb_passed": bool((qa.get("glb") or {}).get("passed")), "passed": bool(qa["passed"])},
        glb_bytes=len(model.glb), glb_gz_bytes=len(gzip.compress(model.glb, 9)), triangles=s["triangles"],
        time_s=round(elapsed, 1), timings_s=s["timings_s"], glb=str(glb_path), summary_json=str(json_path))
    if rec["eligible"] and not rec["gate"]["passed"]:
        rec["snap_candidate"] = snap_candidate(points, scene_base["footprint"], scene_base["holes"])
    if reference is not None:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        cameras = meta.get("cameras") or {}
        moved = st.apply_snap_points(st.FootprintSnap.from_meta(snap), points) if snap.get("accepted") else points
        rec["reference"] = reference_check(cameras, photo_key, reference, snap if snap.get("accepted") else None,
                                           *wall_anchor(moved, scene_base["footprint"], scene_base["holes"]))
        comps = {reference["cameras"][photo_key[pid]]["component"] for pid in cameras
                 if photo_key.get(pid) in reference["cameras"]}
        if comps:
            rec["reference_component_verified"] = all(reference["components"].get(c, {}).get("verified") for c in comps)
    return rec


def wall_anchor(points: np.ndarray, footprint: np.ndarray, holes: list[np.ndarray],
                max_points: int = 5000) -> tuple[np.ndarray, np.ndarray | None]:
    """Where a (snapped) model's walls are: wall-height points within reg_edge_limit_m of the outline (within
    10 m when fewer than 20 are), subsampled, and their median (the outline's centroid without any)."""
    _, st = _stylize_modules()
    band = (points[:, 2] >= 2.0) & (points[:, 2] <= 20.0)
    d = st.distance_to_rings(points[:, :2], [footprint] + list(holes))[0]
    near = band & (d <= st.DEFAULT_PARAMS.reg_edge_limit_m)
    if near.sum() < 20:
        near = band & (d <= 10.0)
    if near.sum() < 20:
        return st.polygon_centroid(footprint), None
    walls = points[near, :2]
    if len(walls) > max_points:
        walls = walls[np.random.default_rng(0).choice(len(walls), max_points, replace=False)]
    return np.median(points[near, :2], axis=0), walls


def best_style_record(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The stylized model users would see at best: among the latest published ones (else any), most coloured
    walls, then most observed area, then a full model, then most registered photos."""
    ok = [r for r in records if r.get("status") == "ok"]
    shown = [r for r in ok if r["latest_published"]] or ok
    if not shown:
        return None
    return max(shown, key=lambda r: (r["walls"]["coloured"], r["observed_pixel_area_fraction"], r["kind"] == "full",
                                     r["registered"] or 0))


def style_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """The style_* METRICS of one building from its stylized model records."""
    ok = [r for r in records if r.get("status") == "ok"]
    tried = [r for r in records if not str(r.get("status", "")).startswith("skipped")]
    eligible = [r for r in tried if r["eligible"]]
    passed = [r for r in eligible if r.get("status") == "ok" and r["gate"]["passed"]]
    checked = [r for r in passed if (r.get("reference") or {}).get("verdict") in ("right", "wrong")]
    best = best_style_record(records)
    accepted_snaps = [r for r in eligible if r.get("status") == "ok" and r["snap"]["accepted"]]
    return {
        "style_models_stylized": len(ok),
        "style_models_eligible": len(eligible),
        "style_registration_passed": len(passed),
        "style_registration_pass_rate": round(len(passed) / len(eligible), 4) if eligible else None,
        "style_registration_checked": len(checked),
        "style_registration_wrong": sum(r["reference"]["verdict"] == "wrong" for r in checked),
        "style_registration_right": sum(r["reference"]["verdict"] == "right" for r in checked),
        "style_rejected_placed_right": sum(
            r.get("status") == "ok" and not r["gate"]["passed"] and (r.get("reference") or {}).get("correction_before")
            is not None and correction_is_right(r["reference"]["correction_before"]) for r in eligible),
        "style_gate_passed_not_eligible": sum(r.get("status") == "ok" and r["gate"]["passed"]
                                              for r in tried if not r["eligible"]),
        "style_snap_refused_gps": sum("own GPS" in (r["snap"].get("reason") or "") for r in eligible
                                      if r.get("status") == "ok"),
        "style_snap_refused_ambiguous": sum("ambiguous" in (r["snap"].get("reason") or "") for r in eligible
                                            if r.get("status") == "ok"),
        "style_snap_residual_before_m": _median(r["snap"]["median_wall_distance_before_m"] for r in eligible
                                                if r.get("status") == "ok"),
        "style_snap_residual_m": _median(r["snap"]["median_wall_distance_after_m"] for r in accepted_snaps),
        "style_ref_camera_error_m": _median(r["reference"]["error_after_m"]["median"] for r in checked),
        "style_ref_camera_error_worst_m": max((r["reference"]["error_after_m"]["median"] for r in checked),
                                              default=None),
        "style_walls_total": best["walls"]["total"] if best else None,
        "style_walls_coloured": best["walls"]["coloured"] if best else None,
        "style_walls_unlocked": best["walls"]["unlocked"] if best else None,
        "style_observed_area": best["observed_pixel_area_fraction"] if best else None,
        "style_roof_observed_area": best["roof_observed_area_fraction"] if best else None,
        "style_fill_m2": best["fill"]["area_m2"] if best else None,
        "style_qa_failed": sum(not r["qa"]["passed"] for r in ok),
        "style_provenance_violations": sum(r["qa"]["violations"] for r in ok),
        "style_glb_kb_gz": round(best["glb_gz_bytes"] / 1024, 1) if best else None,
        "style_time_s": round(sum(r.get("time_s") or 0.0 for r in records), 1) if records else None,
    }


def stylize_published(models: list[dict[str, Any]], b: CrowdBuilding, *, data_dir: Path, work_dir: Path,
                      out_dir: Path, photo_key: dict[str, str], metas: dict[str, PhotoMeta], enu: LocalENU,
                      deadline: float | None = None) -> dict[str, Any]:
    """Stylize every published model (latest ones first), check passing snaps against the outline-corrected
    reference where there is one, diagnose failing eligible models; returns records, best model and metrics."""
    sm, st = _stylize_modules()
    t0 = time.monotonic()
    published = sorted((m for m in models if m["published"]), key=lambda m: (not m["latest_published"], m["request"]
                                                                              if m["request"] is not None else 1e9))
    if b.outline_path is None or not published:
        return {"records": [], "best": None, "reference": None, "metrics": style_metrics([]),
                "note": "no OSM outline" if published else "no published model"}
    footprint, holes = sm.load_footprint_rings(b.outline_path, enu)
    scene_base = {"slug": b.slug, "name": b.name, "footprint": footprint, "holes": holes,
                  "facts": sm.facts_from_entry({}, b.outline_path),
                  "attribution": ["© OpenStreetMap contributors (ODbL)",
                                  "Facade colours derived from Wikimedia Commons photos (per-photo licences; see the "
                                  "scene's licence table)"]}
    reference, ref_report = None, None
    if b.gt is not None:
        t = time.monotonic()
        try:
            reference = outline_corrected_reference(b.gt, metas, footprint, holes, enu)
            ref_report = {"components": reference["components"], "cameras": len(reference["cameras"]),
                          "seconds": round(time.monotonic() - t, 1)}
        except Exception as e:   # never lose the run over the reference check
            log.exception("[%s] outline-corrected reference failed", b.slug)
            ref_report = {"error": f"{type(e).__name__}: {e}"}
    fits = {m["id"]: (m.get("outline") or {}).get("fit") for m in models}
    records = []
    for k, m in enumerate(published):
        if deadline is not None and time.monotonic() > deadline:
            records.append({"model_id": m["id"], "kind": m["kind"], "sector": m["sector"], "request": m["request"],
                            "latest_published": m["latest_published"], "eligible": bool(m["georegistered"]),
                            "status": "skipped (time cap)"})
            continue
        try:
            rec = stylize_model_record(m, data_dir=data_dir, work_dir=work_dir, scene_base=scene_base,
                                       out_dir=out_dir / f"{k:02d}-{m['kind']}-{m['sector'] or 'all'}",
                                       reference=reference, photo_key=photo_key)
        except Exception as e:    # a stylization bug must not lose the building's pipeline results
            log.exception("[%s] stylizing model %s failed", b.slug, m["id"])
            rec = {"model_id": m["id"], "kind": m["kind"], "sector": m["sector"], "request": m["request"],
                   "latest_published": m["latest_published"], "eligible": bool(m["georegistered"]),
                   "status": f"error: {type(e).__name__}: {e}"}
        if rec.get("status") == "ok" and rec["eligible"] and not rec["gate"]["passed"]:
            rec["diagnosis"] = diagnose_gate(rec, fits.get(m["id"]))
        log.info("[%s] stylized %s %s: %s, gate %s%s, %s/%s walls coloured, %.1f s", b.slug, m["kind"],
                 m["sector"] or "", rec.get("status"), (rec.get("gate") or {}).get("passed"),
                 f" ({rec['gate']['reason']})" if rec.get("gate") and not rec["gate"]["passed"] else "",
                 (rec.get("walls") or {}).get("coloured"), (rec.get("walls") or {}).get("total"),
                 rec.get("time_s") or 0)
        records.append(rec)
    best = best_style_record(records)
    return {"records": records, "best": best["model_id"] if best else None, "reference": ref_report,
            "metrics": style_metrics(records), "seconds": round(time.monotonic() - t0, 1)}


# ---------------------------------------------------------------- crowd protocol: run


def run_crowd(
    b: CrowdBuilding,
    out_dir: Path,
    *,
    reconstructor: str = "colmap",
    settings_overrides: dict[str, Any] | None = None,
    arrival: str = "session",
    seed: int = 0,
    soft_time_cap_s: float | None = None,
    stylize: bool = False,
) -> dict[str, Any]:
    """Upload the building's photos as a crowd would and measure the result (see the module docstring).

    Uses the process-wide params.current(). With a soft time cap, no further upload request starts once
    it is exceeded (the summary is marked truncated); a running job is never interrupted. stylize: keep
    every job's work dir (Settings.keep_work_dirs) and stylize every published model afterwards
    (stylize_published, the style_* metrics); it starts no model after 1.15 x the soft cap.
    """
    from fastapi.testclient import TestClient

    from crowd3d.api import camera_estimates, create_app, photo_info
    from crowd3d.coverage import compute_coverage
    from crowd3d.ingest import ingest_upload
    from crowd3d.records import PHOTO_ACCEPTED

    t0 = time.monotonic()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    settings = params.settings_with(
        Settings(data_dir=data_dir, reconstructor=reconstructor, inline_jobs=True,
                 keep_work_dirs=b.gt is not None or stylize),
        settings_overrides or {})
    enu = LocalENU(b.lat, b.lon)
    present, missing = [], []
    for p in b.photos:
        if not p.path.is_file():
            missing.append(p.key)
            continue
        try:
            p.meta = read_photo_meta(p.path.read_bytes())
        except UnreadableImage:
            p.meta = None
        present.append(p)
    with_gps = [p for p in present if p.meta is not None and p.meta.has_gps]
    uploads = present if b.upload_all else with_gps
    requests = plan_arrival(uploads, arrival)
    gt_poses = load_gt_poses(b.gt) if b.gt else None
    outline = load_footprint(b.outline_path, enu) if b.outline_path else None
    landmark_id = "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in b.slug.lower()).strip("-")[:64]
    landmark_id = landmark_id or "landmark"

    rows: dict[str, dict[str, Any]] = {}
    photo_key: dict[str, str] = {}
    request_log: list[dict[str, Any]] = []
    job_request: dict[str, int] = {}
    truncated = False
    with TestClient(create_app(settings)) as client:
        r = client.post("/api/landmarks", json={"id": landmark_id, "name": b.name, "lat": b.lat, "lon": b.lon,
                                                "radius_m": b.radius_m})
        if r.status_code != 201:
            raise RuntimeError(f"creating landmark failed: HTTP {r.status_code} {r.text[:300]}")
        store, runner = client.app.state.store, client.app.state.runner
        landmark = store.get_landmark(landmark_id)
        for k, req in enumerate(requests):
            t = time.monotonic()
            accepted = 0
            for photo, user in req.items:
                o = ingest_upload(store, settings, filename=photo.key, data=photo.path.read_bytes(), user_id=user,
                                  landmark=landmark)
                accepted += o.accepted
                meta, rec = photo.meta, o.photo
                if rec is not None:
                    photo_key[rec.id] = photo.key
                rows[photo.key] = {
                    "key": photo.key, "title": photo.title, "author": photo.author, "user": user, "request": k,
                    "captured_at": meta.captured_at if meta else None,
                    "lat": meta.lat if meta else None, "lon": meta.lon if meta else None,
                    "heading_deg": meta.heading_deg if meta else None,
                    "heading_ref": meta.heading_ref if meta else None,
                    "accepted": o.accepted, "reasons": o.reasons, "warnings": o.warnings,
                    "photo_id": rec.id if rec else None,
                    "distance_m": round(rec.distance_m, 1) if rec and rec.distance_m is not None else None,
                    "sector": SECTOR_NAMES.get(rec.sector) if rec and rec.sector is not None else None,
                }
            t_ingest = time.monotonic() - t
            jobs = runner.schedule(landmark_id) if accepted else []
            job_request.update((j.id, k) for j in jobs)
            request_log.append({
                "request": k, "label": req.label, "photos": len(req.items), "accepted": accepted,
                "ingest_s": round(t_ingest, 1), "jobs_s": round(time.monotonic() - t - t_ingest, 1),
                "jobs": [{"kind": j.kind, "sector": SECTOR_NAMES.get(j.sector), "photos": len(j.photo_ids),
                          "status": j.status, "error": j.error} for j in jobs],
            })
            for j in jobs:
                prune_work_dir(settings.work_dir / j.id, keep_images=stylize)
                log.info("[%s] request %d/%d (%s): %s %s, %d photos -> %s %s", b.slug, k + 1, len(requests), req.label,
                         j.kind, SECTOR_NAMES.get(j.sector, ""), len(j.photo_ids), j.status, j.error or "")
            (out_dir / "progress.json").write_text(json.dumps(
                {"requests_done": k + 1, "requests": len(requests), "elapsed_s": round(time.monotonic() - t0, 1),
                 "log": request_log}, indent=1), encoding="utf-8")
            if soft_time_cap_s is not None and time.monotonic() - t0 > soft_time_cap_s and k + 1 < len(requests):
                truncated = True
                log.warning("[%s] soft time cap %.0f s reached after %d of %d requests", b.slug, soft_time_cap_s,
                            k + 1, len(requests))
                break
        r = client.get(f"/api/landmarks/{landmark_id}")       # the page's own endpoint must still serve it
        if r.status_code != 200:
            raise RuntimeError(f"landmark detail failed: HTTP {r.status_code} {r.text[:300]}")
        model_records = store.list_models(landmark_id)
        jobs_all = store.list_jobs(landmark_id)
        accepted_photos = store.list_photos(landmark_id, PHOTO_ACCEPTED)
        coverage = compute_coverage(accepted_photos, settings)
        # What the landmark page shows per photo (the detail endpoint lists only the newest 200 photos).
        shown_cams = camera_estimates(store.published_models(landmark_id))
        shown_by_id = {p.id: photo_info(p, shown_cams.get(p.id)) for p in accepted_photos}

    jobs_by_id = {j.id: j for j in jobs_all}
    latest = latest_published(model_records)
    latest_ids = {m.id for m in latest.values()}
    models = []
    for m in sorted(model_records, key=lambda m: (m.created_at, m.id)):
        meta = m.meta
        cams = meta.get("cameras") or {}
        job = jobs_by_id.get(m.job_id)
        qa_reasons = (meta.get("qa") or {}).get("reasons") or []
        entry: dict[str, Any] = {
            "id": m.id, "job_id": m.job_id, "kind": m.kind, "sector": SECTOR_NAMES.get(m.sector),
            "request": job_request.get(m.job_id), "latest_published": m.id in latest_ids,
            "num_photos": m.num_photos, "num_registered": meta.get("num_registered"),
            "num_points": meta.get("num_points"), "num_cameras": len(cams),
            "published": bool(meta.get("published", True)),
            "qa_reasons": qa_reasons, "qa_reason_codes": sorted({qa_reason_code(x) for x in qa_reasons}),
            "qa_checks": (meta.get("qa") or {}).get("checks"),
            "georegistered": meta.get("georegistered"), "yaw_source": meta.get("yaw_source"),
            "alignment_rmse_m": _round(meta.get("alignment_rmse_m"), 2),
            "heading_mae_deg": _round(meta.get("heading_mae_deg"), 2),
            "mean_reprojection_error_px": _round(meta.get("mean_reprojection_error_px"), 3),
            "mapping_attempts": meta.get("mapping_attempts"),
            "flagged_cameras": sum(bool(c.get("flags")) for c in cams.values()),
            "camera_flags": dict(Counter(f for c in cams.values() for f in c.get("flags") or [])),
            "camera_gps_residual_m": summarize([c.get("gps_residual_m") for c in cams.values()]),
            "camera_compass_offset_deg": summarize([abs(c["compass_offset_deg"]) for c in cams.values()
                                                    if c.get("compass_offset_deg") is not None]),
            "job_seconds": round(job.finished_at - job.started_at, 1) if job and job.finished_at and job.started_at
            else None,
            "timings_s": meta.get("timings_s"), "message": meta.get("message"),
            "ply_path": str(data_dir / m.ply_path),
        }
        if outline is not None:
            ply = data_dir / m.ply_path
            entry["outline"] = outline_check(read_ply_points(ply), outline, seed) if ply.is_file() else None
        if gt_poses is not None:
            rec = load_job_model(settings.work_dir / m.job_id) if (settings.work_dir / m.job_id).is_dir() else None
            est_rots = {photo_id_from_image_name(im.name): np.asarray(im.cam_from_world().rotation.matrix())
                        for im in rec.images.values() if im.has_pose} if rec is not None else {}
            entry["gt"] = gt_model_check(cams, est_rots, gt_poses, photo_key, entry["published"])
        models.append(entry)

    gps_res, compass_err = [], []
    for row in rows.values():
        p = shown_by_id.get(row["photo_id"])
        if p is None or p.get("estimated_enu") is None or row["lat"] is None:
            continue
        est = p["estimated_enu"]
        row["est_enu"] = [round(float(x), 2) for x in est]
        row["est_heading_deg"] = p.get("estimated_heading_deg")
        tag = enu.to_enu(row["lat"], row["lon"])[:2]
        row["est_gps_residual_m"] = round(float(np.linalg.norm(np.subtract(est[:2], tag))), 2)
        gps_res.append(row["est_gps_residual_m"])
        if row["heading_deg"] is not None and row["est_heading_deg"] is not None:
            row["est_compass_error_deg"] = round(abs(angle_diff_deg(row["est_heading_deg"], row["heading_deg"])), 2)
            compass_err.append(row["est_compass_error_deg"])
    accepted_rows = [r for r in rows.values() if r["accepted"]]
    shown = {"with_estimate": len(gps_res), "accepted": len(accepted_rows),
             "gps_residual_m": summarize(gps_res), "compass_error_deg": summarize(compass_err)}
    funnel = {
        "listed": len(present), "missing_files": missing, "no_gps_in_file": len(present) - len(with_gps),
        "uploaded": len(rows), "accepted": len(accepted_rows), "rejected": len(rows) - len(accepted_rows),
        "rejected_by_reason": dict(Counter(c for r in rows.values() for c in r["reasons"])),
        "warnings": dict(Counter(c for r in rows.values() for c in r["warnings"])),
        "with_heading": sum(r["heading_deg"] is not None for r in rows.values()),
        "magnetic_heading": sum(r["heading_ref"] == "M" for r in rows.values()),
        "distance_m": summarize([r["distance_m"] for r in rows.values()]),
    }
    wall = time.monotonic() - t0
    metrics = crowd_metrics(funnel, coverage, jobs_all, models, latest_ids, shown, wall, gt_poses is not None)
    style = None
    if stylize:
        style = stylize_published(
            models, b, data_dir=data_dir, work_dir=settings.work_dir, out_dir=out_dir / "style", photo_key=photo_key,
            metas={p.key: p.meta for p in present if p.meta is not None}, enu=enu,
            deadline=t0 + 1.15 * soft_time_cap_s if soft_time_cap_s is not None else None)
        metrics.update(style["metrics"])
    summary = {
        "protocol": "crowd",
        "harness_version": HARNESS_VERSION,
        "metric_schema_version": METRIC_SCHEMA_VERSION,
        "building": {"slug": b.slug, "name": b.name, "lat": b.lat, "lon": b.lon, "radius_m": b.radius_m,
                     "outline": str(b.outline_path) if b.outline_path else None, "photos": len(b.photos),
                     "reference_poses": len(gt_poses) if gt_poses is not None else None},
        "arrival": arrival,
        "seed": seed,
        "effective_params": params.effective(settings),
        "keep_work_dirs": settings.keep_work_dirs,
        "stylize": stylize,
        "truncated": truncated,
        "metrics": metrics,
        "funnel": funnel,
        "sectors": {s.label: {"count": s.count, "state": s.state} for s in coverage.sectors},
        "full_ready": coverage.full_ready,
        "requests": request_log,
        "jobs": [{"id": j.id, "kind": j.kind, "sector": SECTOR_NAMES.get(j.sector), "request": job_request.get(j.id),
                  "photos": len(j.photo_ids), "status": j.status, "error": j.error,
                  "seconds": round(j.finished_at - j.started_at, 1) if j.finished_at and j.started_at else None}
                 for j in sorted(jobs_all, key=lambda j: (j.created_at, j.id))],
        "models": models,
        "estimates": shown,
        "style": style,
        "photos": list(rows.values()),
        "data_dir": str(data_dir),
        "topdown_png": str(out_dir / "topdown.png"),
        "elapsed_s": round(wall, 1),
    }
    draw_crowd_topdown(out_dir / "topdown.png", summary, outline, enu, b.radius_m)
    write_crowd_markdown(summary, out_dir / "summary.md")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default), encoding="utf-8")
    return summary


def draw_crowd_topdown(path: Path, summary: dict[str, Any], outline: np.ndarray | None, enu: LocalENU,
                       radius_m: float, size: int = 900) -> None:
    """North-up plan view in ENU: published model points (grey, darker = higher), outline (black), sector
    rays, accepted GPS tags (green; rejected grey), shown estimates (red, joined to their GPS tag, with a
    heading tick)."""
    from PIL import Image, ImageDraw

    tags = [(r, np.array(enu.to_enu(r["lat"], r["lon"])[:2])) for r in summary["photos"] if r["lat"] is not None]
    extent = [30.0] + [float(np.abs(xy).max()) for _, xy in tags]
    if outline is not None:
        extent.append(float(np.abs(outline).max()))
    half = min(max(extent), 1.5 * radius_m) * 1.15
    sc = size / (2 * half)

    def px(e: float, n: float) -> tuple[float, float]:
        return size / 2 + e * sc, size / 2 - n * sc

    img = Image.new("RGB", (size, size), (255, 255, 255))
    dr = ImageDraw.Draw(img)
    for k in range(8):
        a = math.radians(22.5 + 45 * k)
        dr.line([px(0, 0), px(half * 2 * math.sin(a), half * 2 * math.cos(a))], fill=(225, 225, 235))
    dr.ellipse([*px(-radius_m, radius_m), *px(radius_m, -radius_m)], outline=(200, 200, 220))
    clouds = [read_ply_points(Path(m["ply_path"])) for m in summary["models"]
              if m["latest_published"] and Path(m["ply_path"]).is_file()]
    pts = np.vstack(clouds) if clouds else np.zeros((0, 3))
    if len(pts) > 80000:
        pts = pts[np.random.default_rng(0).choice(len(pts), 80000, replace=False)]
    for x, y, z in pts:
        X, Y = px(x, y)
        if 0 <= X < size and 0 <= Y < size:
            g = int(np.clip(210 - 4 * z, 60, 210))
            img.putpixel((int(X), int(Y)), (g, g, g))
    if outline is not None:
        dr.polygon([px(*q) for q in outline], outline=(0, 0, 0))
    for r, xy in tags:
        G = px(*xy)
        if r.get("est_enu"):
            E = px(*r["est_enu"][:2])
            dr.line([G, E], fill=(230, 60, 60))
            h = math.radians(r["est_heading_deg"] or 0.0)
            dr.line([E, (E[0] + 10 * math.sin(h), E[1] - 10 * math.cos(h))], fill=(230, 60, 60))
            dr.ellipse([E[0] - 2, E[1] - 2, E[0] + 2, E[1] + 2], fill=(230, 60, 60))
        color = (0, 150, 0) if r["accepted"] else (150, 150, 150)
        dr.rectangle([G[0] - 3, G[1] - 3, G[0] + 3, G[1] + 3], outline=color, width=2)
    c = px(0, 0)
    dr.line([c[0] - 8, c[1], c[0] + 8, c[1]], fill=(0, 0, 0), width=2)
    dr.line([c[0], c[1] - 8, c[0], c[1] + 8], fill=(0, 0, 0), width=2)
    dr.text((8, 8), f"N up, {2 * half:.0f} m across; + = landmark centre; circle = {radius_m:.0f} m radius; "
                    "green = GPS tag, red = shown estimate", fill=(0, 0, 0))
    img.save(path)


def write_crowd_markdown(summary: dict[str, Any], path: Path) -> None:
    b, m, f = summary["building"], summary["metrics"], summary["funnel"]
    lines = [
        f"# Crowd evaluation: {b['name']}",
        "",
        f"Centre {b['lat']:.6f}, {b['lon']:.6f}, radius {b['radius_m']} m; {f['listed']} photos listed, "
        f"{f['uploaded']} uploaded ({f['no_gps_in_file']} without EXIF GPS not uploaded), {f['accepted']} accepted; "
        f"rejected {f['rejected_by_reason'] or 'none'}; {f['with_heading']} with a compass heading "
        f"({f['magnetic_heading']} magnetic). Arrival `{summary['arrival']}`, {len(summary['requests'])} upload "
        f"requests; reconstructor `{summary['effective_params']['reconstructor']}`; {summary['elapsed_s']} s"
        f"{' (TRUNCATED by the time cap)' if summary['truncated'] else ''}.",
        "",
        "| metric | value | unit |",
        "|---|---|---|",
    ]
    lines += [f"| {d.name} | {m.get(d.name)} | {d.unit} |" for d in METRICS]
    lines += ["", "Sectors (accepted photos): " + ", ".join(
        f"{k} {v['count']}" for k, v in summary["sectors"].items() if v["count"]), "",
        "| model | request | photos | registered | points | published | QA reasons | outline rot deg / shift m / "
        "resid m | GT rot / pos err (med) | gross (shown) | time s |",
        "|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in summary["models"]:
        fit = (e.get("outline") or {}).get("fit")
        fit_text = f"{fit['rotation_deg']} / {fit['shift_m']} / {fit['median_distance_after_m']}" if fit else "-"
        gt = e.get("gt") or {}
        gt_text = (f"{(gt.get('rotation_error_deg') or {}).get('median')} / "
                   f"{(gt.get('position_error_m') or {}).get('median')}")
        lines.append(
            f"| {e['kind']} {e['sector'] or ''}{' (shown)' if e['latest_published'] else ''} | {e['request']} | "
            f"{e['num_photos']} | {e['num_registered']} | {e['num_points']} | {e['published']} | "
            f"{'; '.join(e['qa_reasons']) or '-'} | {fit_text} | {gt_text} | "
            f"{gt.get('gross')} ({gt.get('gross_shown')}) | {e['job_seconds']} |")
    style = summary.get("style")
    if style and style["records"]:
        lines += ["", f"Stylized models (best: {style['best']}); gate = STYLIZED_MODEL 3.2 registration gate:", "",
                  "| model | eligible | snap rot deg / shift m / scale | wall dist before -> after m | edges verified/voting"
                  " | gate | walls coloured (unlocked) / total | reference: correction after (rot / at walls) | verdict |"
                  " diagnosis | QA | s |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in style["records"]:
            if r.get("status") != "ok":
                lines.append(f"| {r['kind']} {r['sector'] or ''} | {r['eligible']} | {r.get('status')} |"
                             " | | | | | | | | |")
                continue
            sn, rg, w, ref = r["snap"], r["registration"], r["walls"], r.get("reference") or {}
            corr = ref.get("correction_after") or {}
            lines.append(
                f"| {r['kind']} {r['sector'] or ''}{' (shown)' if r['latest_published'] else ''} | {r['eligible']} | "
                f"{sn['rotation_deg']} / {sn['shift_m']} / {sn['scale']} | {sn['median_wall_distance_before_m']} -> "
                f"{sn['median_wall_distance_after_m']} | {rg.get('verified_edges')}/{rg.get('voting_edges')} | "
                f"{'pass' if r['gate']['passed'] else 'fail: ' + r['gate']['reason']} | {w['coloured']} ({w['unlocked']})"
                f" / {w['total']} | {corr.get('rotation_deg', '-')} / {corr.get('at_walls_m', '-')} | "
                f"{ref.get('verdict', '-')} | {(r.get('diagnosis') or {}).get('cause', '-')} | "
                f"{'ok' if r['qa']['passed'] else 'FAILED'} | {r['time_s']} |")
    failed = [j for j in summary["jobs"] if j["status"] == "failed"]
    if failed:
        lines += ["", "Failed jobs:"] + [f"- {j['kind']} {j['sector'] or ''} ({j['photos']} photos): {j['error']}"
                                         for j in failed]
    lines += ["", f"Plan view: `{summary['topdown_png']}`. Data dir: `{summary['data_dir']}`."]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_pipeline_config(path: Path) -> dict[str, Any]:
    """An eval/configs/<id>.json: reconstructor name, Settings overrides, params.Params overrides, harness options.

    Every name is checked against the whitelists (params.SETTINGS_NAMES, params.PARAM_NAMES); unknown keys raise.
    """
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = {"config_id", "description", "reconstructor", "settings", "params", "harness"}
    unknown = sorted(set(cfg) - allowed)
    if unknown:
        raise ValueError(f"{path}: unknown config key(s) {unknown}")
    cfg.setdefault("config_id", Path(path).stem)
    cfg.setdefault("reconstructor", "colmap")
    cfg.setdefault("settings", {})
    cfg.setdefault("params", {})
    cfg.setdefault("harness", {})
    params.settings_with(Settings(), cfg["settings"])     # validates names and types
    params.with_overrides(cfg["params"])
    bad = sorted(set(cfg["harness"]) - {"arrival", "seed"})
    if bad:
        raise ValueError(f"{path}: unknown harness option(s) {bad}")
    if cfg["harness"].get("arrival", "session") not in ARRIVALS:
        raise ValueError(f"{path}: arrival must be one of {ARRIVALS}")
    return cfg


def config_effective(cfg: dict[str, Any]) -> dict[str, Any]:
    """The full effective parameter set a config produces (what a run records)."""
    settings = params.settings_with(Settings(data_dir=Path("data"), reconstructor=cfg["reconstructor"]),
                                    cfg["settings"])
    eff = params.effective(settings, params.with_overrides(cfg["params"]))
    eff["harness"] = {"arrival": cfg["harness"].get("arrival", "session"), "seed": int(cfg["harness"].get("seed", 0))}
    return eff


# ---------------------------------------------------------------- download


def download_scene(name: str, cache_dir: Path) -> Path:
    """Download and extract a MegaDepth-X archive once; returns the extracted folder."""
    slug = "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()
    target = cache_dir / slug
    if target.is_dir() and any(target.iterdir()):
        return target
    cache_dir.mkdir(parents=True, exist_ok=True)
    archive = cache_dir / f"{slug}.tar.gz"
    if not archive.is_file():
        url = MEGADEPTH_X_URL.format(name=urllib.parse.quote(name))
        log.info("downloading %s", url)
        partial = archive.with_suffix(".part")
        with urllib.request.urlopen(url, timeout=120) as r, open(partial, "wb") as f:
            shutil.copyfileobj(r, f, length=1 << 20)
        partial.rename(archive)
    tmp = cache_dir / f"{slug}.extracting"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    with tarfile.open(archive) as tar:
        tar.extractall(tmp, filter="data")   # the archive is untrusted: no links out, no special files
    tmp.rename(target)
    return target


def pick_component(folder: Path) -> Path:
    """The <component> folder (with images/ and sparse/) holding the most images."""
    if (folder / "sparse").is_dir():
        return folder
    found = [p.parent for p in folder.rglob("sparse") if (p.parent / "images").is_dir()]
    if not found:
        raise FileNotFoundError(f"no <component>/{{images,sparse}} under {folder}")
    return max(found, key=lambda p: sum(1 for _ in (p / "images").iterdir()))


# ---------------------------------------------------------------- report


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lm, gtg, cc = summary["landmark"], summary["gt_georef"], summary["centre_check"]
    lines = [
        f"# Real-landmark evaluation: {lm['name']}",
        "",
        f"Scene `{summary['scene']['scene_dir']}`: {summary['scene']['gt']['num_images']} GT images, "
        f"{summary['scene']['gt']['real_gps']} with real GPS, {summary['scene']['gt']['real_heading']} with real compass. "
        f"Landmark centre {lm['lat']:.6f}, {lm['lon']:.6f}, radius {lm['radius_m']} m. Seed {summary['seed']}.",
        "",
        "## Ground truth georeference (from real GPS + compass only)",
        "",
        f"- GPS fixes: {gtg['num_gps']}, inliers {gtg['num_gps_inliers']} (threshold {gtg['inlier_threshold_m']} m); "
        f"residuals median {gtg['gps_residual_m']['median']} m, p90 {gtg['gps_residual_m']['p90']} m, "
        f"max {gtg['gps_residual_m']['max']} m; inlier RMSE {gtg['gps_residual_inliers_rmse_m']} m; "
        f"|real compass - GT heading| median {gtg['compass_minus_gt_heading_deg']['median']} deg.",
        f"- Yaw: GPS {gtg['yaw_gps_deg']} deg, compass {gtg['yaw_compass_deg']} deg "
        f"(difference {gtg['yaw_compass_minus_gps_deg']} deg), fused {gtg['yaw_fused_deg']} deg; "
        f"scale/yaw sigma from GPS {gtg['scale_and_yaw_sigma_from_gps']}.",
        f"- Gravity (right vectors of all cameras): {gtg['up_rights_vs_rights_and_centres_deg']} deg from the "
        f"variant with camera centres, {gtg['up_rights_vs_mean_up_deg']} deg from the mean up vector; "
        f"camera roll median {gtg['camera_roll_abs_deg']['median']} deg.",
        "- Caveat: GT-ENU is fitted to the same real GPS the pipeline sees, so absolute errors of real-GPS photos "
        "against it are partly circular. Relative (Sim3) errors do not depend on GPS except for the metric scale.",
        "",
        "## Landmark centre check",
        "",
        f"- GT point cloud horizontal median: {cc['points_horizontal_median_distance_m']} m from the given centre "
        f"(at {cc['points_horizontal_median_en']} E/N).",
        f"- Highest 0.1 % of GT points (lantern/spire): {cc['top_structure_distance_m']} m from the given centre "
        f"(at {cc['top_structure_en']}, {cc['top_structure_height_m']} m up).",
        f"- GT cameras: distance median {cc['gt_camera_distance_m']['median']} m (max {cc['gt_camera_distance_m']['max']}); "
        f"sectors {cc['gt_camera_sectors']}.",
    ]
    fp = cc.get("footprint")
    if fp:
        lines.append(f"- Footprint centroid {fp['given_centre_to_footprint_centroid_m']} m from the given centre; "
                     f"GT top structure {fp['top_structure_to_footprint_centroid_m']} m from it.")
        if fp.get("wall_fit"):
            wf = fp["wall_fit"]
            lines.append(f"- GT wall points vs footprint outline: median {wf['median_distance_before_m']} m; after a rigid "
                         f"fit ({wf['shift_m']} m shift, {wf['rotation_deg']} deg) {wf['median_distance_after_m']} m. "
                         "That shift/rotation estimates the GT-ENU error independently of GPS.")
        if fp.get("sensors"):
            se = fp["sensors"]
            lines.append(f"- Real sensors vs footprint-corrected GT: GPS error median {se['real_gps_error_m']['median']} m "
                         f"(p90 {se['real_gps_error_m']['p90']}); compass |error| median "
                         f"{se['real_compass_abs_error_deg']['median']} deg (p90 {se['real_compass_abs_error_deg']['p90']}, "
                         f"signed median {se['real_compass_signed_median_deg']}).")
    for mode, res in summary["modes"].items():
        up = res["uploads"]
        lines += [
            "",
            f"## Mode `{mode}` ({res['elapsed_s']} s)",
            "",
            f"- Uploads: {res['inputs']['photos']} photos by {len(res['inputs']['users'])} users; GPS {res['inputs']['gps']}, "
            f"heading {res['inputs']['heading']}; accepted {up['accepted']}, rejected {up['rejected']} "
            f"{up['rejected_by_reason']}; warnings {up['warnings']}.",
            f"- Settings overrides: {res['settings']['overridden'] or 'none'}.",
            f"- Sector misassignment (EXIF GPS sector vs GT-ENU sector): {res['sector_misassignment']}.",
            f"- EXIF GPS error vs GT-ENU (m): {res['exif_gps_error_vs_gt_enu_m']}.",
            "",
            "| stage | uploaded | accepted | jobs | full ready (product defaults) | full model |",
            "|---|---|---|---|---|---|",
        ]
        for st in res["stages"]:
            jobs = ", ".join(" ".join(str(x) for x in (j["kind"], j["sector"], f"{j['num_photos']} photos", j["status"],
                                                        f"{j['seconds']} s") if x) for j in st["jobs"]) or "none"
            lines.append(f"| {st['stage']}: {st['description']} | {st['uploaded']} | {st['accepted']} | {jobs} | "
                         f"{st['full_ready']} ({st['full_ready_with_product_defaults']}) | {st['has_full_model']} |")
        lines += ["", "Final sectors: " + ", ".join(
            f"{k} {v['count']}{' (face model)' if v['has_face_model'] else ''}" for k, v in res["final_sectors"].items()
            if v["count"])]
        if res["failed_jobs"]:
            lines.append(f"Failed jobs: {res['failed_jobs']}")
        shown = res["photo_estimates_as_shown"]
        lines += ["", f"Per-photo estimates as the landmark page shows them: {shown['with_estimate']} of "
                      f"{shown['accepted']} accepted photos; horizontal error vs GT-ENU "
                      f"{shown['horizontal_error_vs_gt_enu_m']}; heading error vs GT-ENU "
                      f"{shown['heading_error_vs_gt_enu_deg']}; vs real compass {shown['heading_error_vs_real_compass_deg']}; "
                      f"horizontal error vs footprint-GT {shown['horizontal_error_vs_footprint_gt_m']}. "
                      f"Plan view: `{res['topdown_png']}`."]
        lines += [
            "",
            "| model (stage) | photos | registered | points | georeg | align RMSE m | heading MAE deg "
            "| horiz err vs GT-ENU m (med/p90) | vs footprint-GT m (med/p90) | heading err vs GT-ENU deg (med/p90) "
            "| vs real compass deg (med) | rel. pos err m (med/p90) | rel. rot err deg (med/p90) "
            "| gross-error cams (flagged) | focal rel. err (med) | time s |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for m in res["models"]:
            h, hf = m["horizontal_error_vs_gt_enu_m"], m["horizontal_error_vs_footprint_gt_m"]
            hg, hc = m["heading_error_vs_gt_enu_deg"], m["heading_error_vs_real_compass_deg"]
            rel = m.get("relative") or {}
            rp, rr = rel.get("position_error_m") or {}, rel.get("rotation_error_deg") or {}
            ge = m.get("gross_errors") or {}
            focal = ((m.get("colmap_cameras") or {}).get("focal_relative_error_vs_gt") or {}).get("median")
            lines.append(
                f"| {m['kind']} {m['sector'] or ''} ({m['stage']}){'' if m.get('published', True) else ' NOT PUBLISHED'} | "
                f"{m['num_photos']} | {m['num_registered']} | "
                f"{m['num_points']} | {m['georegistered']} | {m['alignment_rmse_m']} | {m['heading_mae_deg']} | "
                f"{h['median']} / {h['p90']} | {hf['median']} / {hf['p90']} | {hg['median']} / {hg['p90']} | "
                f"{hc['median']} | {rp.get('median')} / {rp.get('p90')} | {rr.get('median')} / {rr.get('p90')} | "
                f"{ge.get('cameras')} ({ge.get('flagged_and_gross')} of {ge.get('flagged_by_gps_and_compass_check')}) | "
                f"{focal} | {m['job_seconds']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- CLI


def _inside(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def check_outside_repo(path: Path, what: str) -> None:
    if _inside(path, REPO_ROOT) and not _inside(path, REPO_ROOT / "data"):
        raise SystemExit(f"{what} {path} is inside the repository: real photos and anything derived from them "
                         f"must stay outside it (or under its gitignored data/ folder)")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python scripts/eval_real_landmark.py", description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--building-spec", type=Path, help="resolved building spec JSON (scripts/benchmark.py writes it)")
    src.add_argument("--commons-dir", type=Path,
                     help="scripts/fetch_commons_landmark.py output folder (manifest.json + photos)")
    src.add_argument("--scene-dir", type=Path,
                     help="legacy: extracted MegaDepth-X <scene>/<component> (images/, sparse/)")
    src.add_argument("--scene", help="legacy: MegaDepth-X scene name, e.g. 'Baptistry_(Florence)': download + extract")
    ap.add_argument("--catalog", type=Path, default=REPO_ROOT / "eval" / "landmarks.json",
                    help="landmark catalog for --slug (default eval/landmarks.json)")
    ap.add_argument("--slug", help="--commons-dir: catalog entry giving centre, radius and OSM outline")
    ap.add_argument("--config", type=Path, default=None,
                    help="pipeline config (eval/configs/<id>.json); default: product defaults, COLMAP")
    ap.add_argument("--arrival", choices=ARRIVALS, default=None,
                    help="crowd protocol upload order (default: the config's, else session)")
    ap.add_argument("--time-cap", type=float, default=None,
                    help="crowd protocol: start no new upload request after this many seconds")
    ap.add_argument("--stylize", action="store_true",
                    help="crowd protocol: keep work dirs and stylize every published model (style_* metrics)")
    ap.add_argument("--gt-sparse", type=Path, default=None,
                    help="--commons-dir: reference COLMAP model (a MegaDepth-X <component>/sparse folder)")
    ap.add_argument("--gt-map", type=Path, default=None,
                    help="--commons-dir: JSON {Commons file name: reference image name}, or MegaDepth-X's"
                         " license.parquet together with --gt-prefix")
    ap.add_argument("--gt-prefix", default=None, help="with a parquet --gt-map: e.g. 'Baptistry_(Florence)/1/images/'")
    ap.add_argument("--gt-images", type=Path, default=None,
                    help="folder of the reference images, to read their EXIF orientation")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where --scene archives go")
    ap.add_argument("--lat", type=float, default=None, help="landmark centre latitude (overrides the catalog)")
    ap.add_argument("--lon", type=float, default=None, help="landmark centre longitude (overrides the catalog)")
    ap.add_argument("--radius", type=float, default=None, help="check-in radius in metres (default: catalog, else 150)")
    ap.add_argument("--name", default=None, help="landmark name (default: catalog or folder name)")
    ap.add_argument("--landmark-id", default="real-landmark", help="legacy protocol: landmark id")
    ap.add_argument("--mode", choices=(*MODES, "both"), default="real",
                    help="legacy protocol: real (default) or augmented (simulated tags for untagged photos)")
    ap.add_argument("--out", type=Path, default=None, help="run dir (legacy default <cache-dir>/eval-runs/<scene>)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="legacy: base for the Crowd3D data dirs, one sub-folder per mode (default <out>/<mode>/data)")
    ap.add_argument("--overwrite", action="store_true", help="delete existing data dirs first")
    ap.add_argument("--seed", type=int, default=None, help="default: the config's, else 0")
    ap.add_argument("--gps-noise-m", type=float, default=5.0, help="legacy augmented: simulated GPS sigma per axis")
    ap.add_argument("--heading-noise-deg", type=float, default=15.0, help="legacy augmented: simulated compass sigma")
    ap.add_argument("--max-images", type=int, default=None, help="legacy: cap on uploads (real-GPS photos first)")
    ap.add_argument("--users", type=int, default=5, help="legacy: pseudo users the uploads are spread over")
    ap.add_argument("--max-side", type=int, default=None, help="legacy: Settings.processing_max_side_px override")
    ap.add_argument("--full-min-sectors", type=int, default=None,
                    help="legacy: Settings.full_min_sectors override (product default 6); reported as an override")
    ap.add_argument("--footprint", type=Path, default=None,
                    help="GeoJSON or Nominatim JSON with the building outline (overrides the catalog's)")
    return ap.parse_args(argv)


def gt_from_args(args: argparse.Namespace) -> dict[str, Any] | None:
    """The reference spec for --commons-dir from --gt-sparse / --gt-map / --gt-prefix / --gt-images."""
    if args.gt_sparse is None:
        return None
    if args.gt_map is None:
        raise SystemExit("--gt-sparse needs --gt-map")
    if args.gt_map.suffix == ".parquet":
        import pyarrow.parquet as pq

        if not args.gt_prefix:
            raise SystemExit("a parquet --gt-map needs --gt-prefix")
        rows = pq.read_table(args.gt_map, columns=["name_in_megadepth-x", "image_name"]).to_pylist()
        mapping = {r["image_name"]: r["name_in_megadepth-x"][len(args.gt_prefix):] for r in rows
                   if r["name_in_megadepth-x"].startswith(args.gt_prefix)}
    else:
        mapping = json.loads(args.gt_map.read_text(encoding="utf-8"))
    images = {}
    for key, value in mapping.items():
        entry = dict(value) if isinstance(value, dict) else {"name": value, "orientation": None}
        image = args.gt_images / entry["name"] if args.gt_images is not None else None
        if image is not None and entry.get("orientation") is None and image.is_file():
            entry["orientation"] = read_photo_meta(image.read_bytes()).orientation
        images[key] = entry
    return {"kind": "colmap", "sparse_dir": str(args.gt_sparse), "images": images}


def main_crowd(args: argparse.Namespace, cfg: dict[str, Any] | None) -> int:
    if args.building_spec:
        b = building_from_spec(json.loads(args.building_spec.read_text(encoding="utf-8")))
    else:
        entry = catalog_entry(args.catalog, args.slug) if args.slug else {}
        lat = args.lat if args.lat is not None else entry.get("lat")
        lon = args.lon if args.lon is not None else entry.get("lon")
        if lat is None or lon is None:
            raise SystemExit("--commons-dir needs --slug (catalog entry) or --lat and --lon")
        outline = args.footprint or (REPO_ROOT / entry["osm"]["file"] if (entry.get("osm") or {}).get("file") else None)
        b = building_from_commons_dir(
            args.commons_dir, slug=args.slug or args.commons_dir.name, name=args.name or entry.get("name")
            or args.commons_dir.name, lat=lat, lon=lon, radius_m=args.radius or entry.get("radius_m") or 150.0,
            outline_path=outline, gt=gt_from_args(args))
    if args.out is None:
        raise SystemExit("--out is required for the crowd protocol")
    check_outside_repo(args.out, "--out")
    data = args.out / "data"
    if data.exists() and any(data.iterdir()):
        if not args.overwrite:
            raise SystemExit(f"data dir {data} is not empty; pass --overwrite to start over")
        shutil.rmtree(data)
    harness = (cfg or {}).get("harness", {})
    summary = run_crowd(
        b, args.out, reconstructor=(cfg or {}).get("reconstructor", "colmap"),
        settings_overrides=(cfg or {}).get("settings", {}), arrival=args.arrival or harness.get("arrival", "session"),
        seed=args.seed if args.seed is not None else int(harness.get("seed", 0)), soft_time_cap_s=args.time_cap,
        stylize=args.stylize)
    m = summary["metrics"]
    print(f"wrote {args.out / 'summary.json'}, summary.md and topdown.png: {m['photos_accepted']}/"
          f"{m['photos_uploaded']} accepted, {m['models_published']} models published, "
          f"{m['models_qa_rejected']} rejected by QA, {summary['elapsed_s']} s")
    if args.stylize:
        print(f"stylized {m['style_models_stylized']} published models: {m['style_registration_passed']}/"
              f"{m['style_models_eligible']} eligible pass the registration gate, best model "
              f"{m['style_walls_coloured']}/{m['style_walls_total']} walls coloured, {m['style_time_s']} s")
    print(f"browse: python -m crowd3d.api --data-dir {summary['data_dir']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    cfg = load_pipeline_config(args.config) if args.config else None
    if cfg is not None:
        params.activate(params.with_overrides(cfg["params"]))
    if args.building_spec or args.commons_dir:
        return main_crowd(args, cfg)
    if args.lat is None or args.lon is None:
        raise SystemExit("the legacy protocol needs --lat and --lon")
    args.radius = args.radius or 150.0
    args.seed = 0 if args.seed is None else args.seed
    args.config_settings = (cfg or {}).get("settings", {})
    if args.scene:
        args.scene_dir = pick_component(download_scene(args.scene, args.cache_dir))
    args.scene_dir = pick_component(args.scene_dir)
    scene_name = args.scene or args.scene_dir.parent.name
    args.name = args.name or scene_name.replace("_", " ")
    out = args.out or args.cache_dir / "eval-runs" / scene_name
    check_outside_repo(out, "--out")
    modes = MODES if args.mode == "both" else (args.mode,)
    data_dirs = {m: (args.data_dir / m if args.data_dir else out / m / "data") for m in modes}
    for d in data_dirs.values():
        check_outside_repo(d, "data dir")
        if d.exists() and any(d.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"data dir {d} is not empty; pass --overwrite to start over")
            shutil.rmtree(d)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    gt = load_ground_truth(args.scene_dir)
    with params.using(params.DEFAULTS):    # the reference frame must not move with the evaluated config
        frame = georeference_gt(gt, args.lat, args.lon, Settings().camera_height_m)
    footprint = load_footprint(args.footprint, LocalENU(args.lat, args.lon)) if args.footprint else None
    summary: dict[str, Any] = {
        "scene": {"name": scene_name, "scene_dir": str(args.scene_dir), "gt": gt.summary},
        "landmark": {"id": args.landmark_id, "name": args.name, "lat": args.lat, "lon": args.lon,
                     "radius_m": args.radius},
        "seed": args.seed,
        "simulated_noise": {"gps_m": args.gps_noise_m, "heading_deg": args.heading_noise_deg},
        "gt_georef": frame.report,
        "centre_check": centre_check(gt, frame, footprint),
        "modes": {},
    }
    log.info("GT georeference: %s", {k: v for k, v in frame.report.items() if k != "per_photo"})
    log.info("centre check: %s", summary["centre_check"])
    wall_fit = (summary["centre_check"].get("footprint") or {}).get("wall_fit")
    if wall_fit:
        frame.footprint_fix = (wall_fit["rotation_deg"], np.array(wall_fit["translation_m"]))
        summary["centre_check"]["footprint"]["sensors"] = sensor_check(gt, frame, LocalENU(args.lat, args.lon))
    for m in modes:
        summary["modes"][m] = run_mode(m, gt, frame, args, data_dirs[m], out / m / "uploads")
        (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default), encoding="utf-8")
    summary["elapsed_s"] = round(time.monotonic() - t0, 1)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default), encoding="utf-8")
    write_markdown(summary, out / "summary.md")
    print(f"wrote {out / 'summary.json'} and {out / 'summary.md'}")
    for m in modes:
        print(f"browse: python -m crowd3d.api --data-dir {data_dirs[m]}  ->  "
              f"http://127.0.0.1:8000/landmark.html?id={args.landmark_id}")
    return 0


def _json_default(o: Any) -> Any:
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
