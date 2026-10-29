"""Evaluate the Crowd3D pipeline on real photos of a famous building, as a crowd would upload them.

    # Commons test-set dir (scripts/fetch_commons_landmark.py output) + a catalog entry:
    python -I scripts/eval_real_landmark.py --commons-dir <dir> --catalog eval/landmarks.json \
        --slug florence-baptistery --out <run dir> [--gt-sparse <MegaDepth-X sparse> --gt-map <json>]
    # One building of a frozen test set, resolved by scripts/benchmark.py:
    python -I scripts/eval_real_landmark.py --building-spec <spec.json> --config eval/configs/default.json --out <dir>

Crowd protocol (the product rule: uploads are camera originals whose own EXIF carries GPS):
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
screenshots) into this repository: --out must lie outside it or under its gitignored data/ folder.

load_ground_truth / georeference_gt put a MegaDepth-X reference model into the landmark's ENU frame from
its photos' real EXIF GPS and compass (scripts/stylize_model.py --scene uses them).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import resource
import shutil
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

if __package__ in (None, ""):
    # `python scripts/eval_real_landmark.py` from a checkout: make the crowd3d package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd3d import params  # noqa: E402
from crowd3d.anchor import GeoAnchor  # noqa: E402
from crowd3d.config import Settings  # noqa: E402
from crowd3d.coverage import sector_label  # noqa: E402
from crowd3d.exif import PhotoMeta, UnreadableImage, read_photo_meta  # noqa: E402
from crowd3d.geo import LocalENU, angle_diff_deg, heading_from_en  # noqa: E402
from crowd3d.reconstruct import (  # noqa: E402
    ReconPhoto,
    apply_similarity,
    compass_yaw,
    estimate_up,
    georegister_inputs,
    georegister_model,
    photo_id_from_image_name,
    robust_similarity_2d,
    rotation_to_z,
    rotation_z,
    similarity_matrix,
)

log = logging.getLogger("eval_real_landmark")

REPO_ROOT = Path(__file__).resolve().parent.parent
MEGADEPTH_X_URL = "https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X/resolve/main/release/{name}.tar.gz"

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
                                     # (a 30 m facade stretched 10 % puts its ends' photos 1.5 m off) ...
STYLE_RIGHT_CAMERA_M = 1.5           # ... and its cameras' horizontal centres sit within this of the reference (median)
STYLE_TIME_CAP_S = 1800.0            # the stylization's own budget per building (20 models took 40 s + the reference)
HARNESS_VERSION = 7                  # 3: stylized models (--stylize); 4: the snap gets the model's SnapPrior;
                                     # 5: the stylization has its own time budget; flagged cameras colour nothing
                                     # 6: photo poses (crowd3d.photoposes) per uploaded photo against the
                                     #    outline-corrected reference, which is now built once per building
                                     # 7: per upload request the jobs' CPU time and peak memory, and the poses the
                                     #    page shows after it (time to pose); the schema-7 scores after the run
REFERENCE_VERSION = 1                # the outline-corrected reference and its cache file (building_reference); bump
                                     # when either can change. The cache key holds this, not HARNESS_VERSION, so a
                                     # metric change keeps the caches (the reference models may be deleted once cached)
REFERENCE_POINTS_MAX = 100_000       # outline-corrected reference points kept per building (verified components)
REFERENCE_SIGMA = {"yaw_deg": 1.0, "scale": 0.02}   # the outline-corrected reference's own 1-sigma error
                                     # (REAL_WORLD_EVAL 1.4: its yaw is off by 1-2 deg, its scale by a few per cent)
REFERENCE_SIGMA_XY_MIN_M = 0.5       # calib_xy: the reference's own horizontal error per axis, its verified
                                     # components' median wall distance (outline residual) but at least this
REFERENCE_SIGMA_TILT_DEG = 0.5       # calib_tilt: the reference's own tilt per horizontal axis
REGEO_MAX_DIFF_M = 0.01              # rescore --regeo: a replayed georegistration must put every camera this close
                                     # to the model JSON's, else the model is left out of the calibration
POSE_RIGHT_M = 10.0                  # a determined photo pose is right ("大致方位") within this horizontal distance ...
POSE_RIGHT_DEG = 15.0                # ... and this heading error of the reference (the product's compass sigma; buildings
                                     # are 30-150 m across, a phone's field of view ~60 deg)
POSE_STRICT_M = 5.0                  # the stricter limits VISUAL_LOCALIZATION calls a right pose
POSE_STRICT_DEG = 10.0
WRONG_FACE_HEADING_DEG = 30.0        # a shown camera looking this far off its reference heading, or standing this
WRONG_FACE_BEARING_DEG = 45.0        # far round the building from it (bearing from the landmark centre), shows the
                                     # wrong face (VISUAL_LOCALIZATION loc_wrong_face_accepts); a model is placed facing
                                     # the wrong way when its yaw is WRONG_FACE_HEADING_DEG off
COV_BIN_M = 1.0                      # supported coverage: the OSM outer ring in bins of about this arc length; a
COV_NEAR_M = 5.0                     # wall point (FOOTPRINT_WALL_Z_M) lands on the bin of its nearest outline point
COV_MIN_POINTS = 10                  # within this, and a model covers a bin when this many of its points land on it
COV_TOLERANCES_M = (1.5, 5.0)        # supported at these (1.5 m = STYLE_RIGHT_SHIFT_M; local at COV_NEAR_M = covered)
COV_MAX_POINTS = 20_000              # wall points per model (subsampled)
COV_MAJORITY = 0.5                   # cov_local_majority_m_1p5: this share of a bin's landed points within 1.5 m
CALIB_COMPONENTS = ("yaw", "scale", "xy", "tilt")   # xy (at the anchor's pivot) and tilt: the GeoAnchor's sigmas
CALIB_2D = ("xy", "tilt")            # two horizontal axes with one sigma: z from the 2-D radius (calib_z)

# ---------------------------------------------------------------- metrics (the one definition; benchmark.py uses it)

METRIC_SCHEMA_VERSION = 7       # 2: the style_* group (stylized models, docs/STYLIZED_MODEL.md §3.2, §7)
                                # 3: + style_registration_right, style_rejected_placed_right, style_snap_refused_gps,
                                #    style_snap_refused_ambiguous (style_rejected_placed_right was added to 3 in
                                #    place; no result file without it remains)
                                # 4: "right" = within 5 deg, 1.5 m over the wall points (90th percentile) and 10 %
                                #    scale (3: 3 m at the walls' median point and 15 %); + style_gate_passed_not_eligible,
                                #    style_ref_camera_error_worst_m
                                # 5: "right" also needs the cameras' horizontal centres within 1.5 m of the reference
                                #    (median) and the wall points moved <= 1.5 m in 3-D (rotation from the cameras'
                                #    orientations: a tilt counts); style_ref_camera_error_worst_m (the largest per-model
                                #    median) renamed style_ref_camera_median_worst_m; + style_ref_camera_error_max_m
                                #    (worst single camera), style_ref_walls_p90_worst_m, style_incomplete
                                # 6: + the photo_pose_* group (where each photo was taken and which way it looked,
                                #    docs/BENCHMARK.md "照片位姿"); the earlier metrics are unchanged. The *_testset
                                #    metrics (photos the time cap never uploaded count as failures) and the bound
                                #    pair were added to 6 in place before any merge; every v1-poses file was
                                #    re-recorded with them
                                # 7: + geo_* and wrong_face_* (the georegistration against the reference, no snap),
                                #    cov_* (supported outline metres, building-local and geographic), calib_* (z of
                                #    the yaw and scale errors; xy and tilt None until their sigmas exist), the runtime
                                #    cost (CPU time, peak memory, request latency, time to pose, disk),
                                #    stale_overwrites and version_drift_* (from the store) and struct_* (None until
                                #    the structure pipeline fills them: compare reads None -> value as "appeared");
                                #    the earlier metrics are unchanged. style_coloured_walls_wrong was added to 7 in
                                #    place before any schema-7 result was committed


@dataclass(frozen=True)
class Metric:
    name: str
    unit: str
    direction: str             # "higher" / "lower" is better; "none": context only
    aggregate: str             # across buildings: "sum", "mean" or "median" of the non-missing values, or
    #                            "pooled": sum(pooled[0]) / sum(pooled[1]) over the buildings that have both
    group: str                 # input, outcome, outline, sensors, gt, runtime, style, poses, geo, coverage, calib,
    #                            versions, struct
    description: str
    min_delta: float = 0.0     # a change up to this is never significant ...
    rel_floor: float = 0.0     # ... nor one up to this fraction of the earlier value
    pooled: tuple[str, str] | None = None   # (numerator, denominator) metric names for aggregate "pooled"


_CALIB_ERRORS = {   # component: (its error, the model's sigma, the reference's sigma)
    "yaw": ("|yaw| (geo_yaw_error)", "the anchor's yaw sigma (a model JSON without an anchor: yaw_sigma_deg)", "1 deg"),
    "scale": ("|ln scale| (geo_scale_error)", "the anchor's log-scale sigma (without an anchor: scale_sigma)", "2 %"),
    "xy": ("the horizontal error at the anchor's pivot (east, north: the 2-D radius, calib_z)",
           "the anchor's east/north sigma (none without an anchor)",
           f"the reference's median wall distance, at least {REFERENCE_SIGMA_XY_MIN_M:g} m per axis"),
    "tilt": ("the tilt about east and north (the 2-D radius, calib_z)",
             "the anchor's tilt sigma per axis (none without an anchor)", f"{REFERENCE_SIGMA_TILT_DEG:g} deg per axis"),
}


def _calib_metrics() -> tuple[Metric, ...]:
    """calib_<component>_n / _within2 / _cover2 / _beyond3: z = error / sqrt(sigma_model^2 + sigma_ref^2) per checked
    model (the georegistration's own uncertainty against its error; sigma_ref 0 for generator truth). sigma_model is
    the model's GeoAnchor's with the current floors (GeoAnchor.recalibrated)."""
    out = []
    for c in CALIB_COMPONENTS:
        err, sigma, ref = _CALIB_ERRORS[c]
        out += [
            Metric(f"calib_{c}_n", "models", "none", "sum", "calib",
                   f"Checked models (geo_models_checked, with the 3-D similarity) with a {c} sigma: z = {err} / sqrt"
                   f"(sigma_model^2 + sigma_ref^2), sigma_model = {sigma}, sigma_ref = {ref} (0 for generator truth)."
                   + (" None when no checked model has an anchor (rescore --regeo replays one)." if c in CALIB_2D
                      else "")),
            Metric(f"calib_{c}_within2", "models", "none", "sum", "calib", "Of those, |z| <= 2."),
            Metric(f"calib_{c}_cover2", "fraction", "none", "pooled", "calib",
                   f"calib_{c}_within2 / calib_{c}_n: the share inside 2 sigma (0.954 when the sigma is right; fewer"
                   " than 20 models say little).", 0.02, pooled=(f"calib_{c}_within2", f"calib_{c}_n")),
            Metric(f"calib_{c}_beyond3", "models", "lower", "sum", "calib",
                   "Of those, |z| > 3: the georegistration claimed far more certainty than it had."),
        ]
    return tuple(out)


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
    # the cost of a run in CPU time and memory: runs may share the machine's cores, CPU time stays comparable
    Metric("cpu_time_s", "s", "lower", "sum", "runtime",
           "Process CPU time (user + system, all threads) of the building run over the span of wall_time_s: the"
           " primary cost metric (wall time grows when other runs share the cores).", 5.0, 0.25),
    Metric("jobs_cpu_s", "s", "lower", "sum", "runtime", "CPU time (user + system) of the reconstruction jobs.",
           5.0, 0.25),
    Metric("peak_rss_jobs_mb", "MB", "lower", "max", "runtime",
           "Highest resident memory of the process during any upload request's jobs (VmHWM, reset before the jobs;"
           " the process peak ru_maxrss where Linux does not allow the reset).", 50.0, 0.25),
    Metric("baseline_rss_mb", "MB", "none", "max", "runtime",
           "Resident memory just before the jobs of that request: the part of peak_rss_jobs_mb the jobs did not add.",
           50.0, 0.25),
    Metric("request_latency_p50_s", "s", "lower", "median", "runtime",
           "Median time from the start of an upload request to the end of its jobs, over the requests that ran jobs"
           " (how long a contributor waits for the models to change).", 1.0, 0.25),
    Metric("request_latency_p90_s", "s", "lower", "median", "runtime", "90th percentile of the same.", 1.0, 0.25),
    Metric("photo_time_to_pose_p50_s", "s", "lower", "median", "runtime",
           "Median time from a photo's upload request to the first time the landmark page shows a pose for it, over"
           " the photos that got one (estimate_coverage says how many).", 5.0, 0.25),
    Metric("photo_time_to_pose_p90_s", "s", "lower", "median", "runtime", "90th percentile of the same.", 5.0, 0.25),
    Metric("disk_run_mb", "MB", "lower", "sum", "runtime",
           "Size of the run's data dir after the uploads (photos, models, database; the job work dirs only the harness"
           " keeps are left out).", 1.0, 0.1),
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
           "Of those, passes that are wrong: the snapped model's cameras are more than"
           f" {STYLE_RIGHT_CAMERA_M:g} m from the outline-corrected reference (median, horizontal), or the"
           f" similarity they still need exceeds {STYLE_RIGHT_ROT_DEG:g} deg, moves the wall points by more than"
           f" {STYLE_RIGHT_SHIFT_M:g} m (90th percentile; in 3-D, with the rotation from the cameras' orientations,"
           f" so a tilt counts) or rescales the model by more than {STYLE_RIGHT_SCALE:.0%} (photos would colour the"
           " wrong wall, or the wrong part of it)."),
    Metric("style_registration_right", "models", "higher", "sum", "style",
           "Of the checked passes, the right ones (the rest of style_registration_checked is wrong)."),
    Metric("style_coloured_walls_wrong", "walls", "lower", "sum", "style",
           "Walls with photo colour (unlocked + partly unlocked) summed over the passes style_registration_wrong"
           " counts: colour shown on a model placed wrong, taken from photos of another wall or another part of it"
           " (must be 0)."),
    Metric("style_rejected_placed_right", "models", "lower", "sum", "style",
           f"Eligible models the gate refused although the georegistration alone already placed them right"
           f" (at least {STYLE_MIN_REF_CAMERAS} reference cameras; the same limits as style_registration_wrong):"
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
    Metric("style_ref_camera_median_worst_m", "m", "lower", "max", "style",
           f"The largest of those per-model median camera-centre errors (a pass above {STYLE_RIGHT_CAMERA_M:g} m"
           " is wrong).", 0.2),
    Metric("style_ref_camera_error_max_m", "m", "lower", "max", "style",
           "The worst single camera of the checked passing models (horizontal distance to its reference position;"
           " one camera with a bad pose of its own can be far off while the model is right).", 0.5),
    Metric("style_ref_walls_p90_worst_m", "m", "lower", "max", "style",
           "The largest 3-D wall-point displacement (90th percentile) the checked passing models still need to"
           " match the reference (2-D when the orientations are unknown).", 0.2),
    Metric("style_incomplete", "models", "lower", "sum", "style",
           "Published models the stylization did not reach (its own time cap): the building's style_* metrics"
           " then cover only the rest, and compare treats them as incomplete rather than as a change."),
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
    # Photo poses (crowd3d.photoposes): per uploaded photo where it was taken and which way it looked.
    # "Determined" = both known (published SfM camera, visual localization, or EXIF GPS + compass); a rejected upload
    # or GPS without a compass is NOT determined. "Correct" needs a verified reference pose (synthetic truth or the
    # outline-corrected MegaDepth-X reference) within POSE_RIGHT_M / POSE_RIGHT_DEG.
    Metric("photo_pose_determined", "photos", "higher", "sum", "poses",
           "Uploaded photos whose position and viewing direction are both determined (sfm + loc + exif)."),
    Metric("photo_pose_determined_rate", "fraction", "higher", "pooled", "poses",
           "photo_pose_determined / photos_uploaded: every rejected upload and every photo with GPS but no compass"
           " that SfM did not place counts as a failure.", 0.01,
           pooled=("photo_pose_determined", "photos_uploaded")),
    Metric("photo_pose_sfm", "photos", "none", "sum", "poses",
           "Of those, posed by a camera of a published SfM model (what the landmark page shows; moved with the"
           " model's footprint snap where one was accepted)."),
    Metric("photo_pose_loc", "photos", "none", "sum", "poses",
           "Posed by visual localization against a published model (a hook: the product has no localizer yet)."),
    Metric("photo_pose_exif", "photos", "none", "sum", "poses",
           "Posed by the photo's own EXIF GPS and compass (approximate)."),
    Metric("photo_pose_gps_only", "photos", "lower", "sum", "poses",
           "Accepted photos with a GPS position but no direction (no compass, not placed by SfM): failures."),
    Metric("photo_pose_gt", "photos", "none", "sum", "poses",
           "Uploaded photos with a verified reference pose (rejected ones included)."),
    Metric("photo_pose_gt_determined", "photos", "higher", "sum", "poses", "Of those, determined."),
    Metric("photo_pose_correct", "photos", "higher", "sum", "poses",
           f"Of those, within {POSE_RIGHT_M:g} m and {POSE_RIGHT_DEG:g} deg of the reference."),
    Metric("photo_pose_correct_rate", "fraction", "higher", "pooled", "poses",
           "photo_pose_correct / photo_pose_gt: the pass rate on the photos that can be checked (undetermined and"
           " wrong both fail).", 0.01, pooled=("photo_pose_correct", "photo_pose_gt")),
    Metric("photo_pose_wrong", "photos", "lower", "sum", "poses",
           f"Determined photos with a reference pose more than {POSE_RIGHT_M:g} m or {POSE_RIGHT_DEG:g} deg off."),
    Metric("photo_pose_correct_strict", "photos", "higher", "sum", "poses",
           f"Reference photos within the stricter {POSE_STRICT_M:g} m and {POSE_STRICT_DEG:g} deg."),
    Metric("photo_pose_correct_strict_rate", "fraction", "higher", "pooled", "poses",
           "photo_pose_correct_strict / photo_pose_gt.", 0.01, pooled=("photo_pose_correct_strict", "photo_pose_gt")),
    Metric("photo_pose_pass", "photos", "higher", "sum", "poses",
           "Uploaded photos that pass: determined and not shown wrong (a determined photo without a reference"
           " pose cannot be checked and counts as a pass)."),
    Metric("photo_pose_pass_rate", "fraction", "higher", "pooled", "poses",
           "photo_pose_pass / photos_uploaded (optimistic where references are few: quote photo_pose_correct_rate).",
           0.01, pooled=("photo_pose_pass", "photos_uploaded")),
    # per test set: photos the protocol would upload (listed, present, with EXIF GPS unless the set uploads every
    # photo) but the harness time cap never uploaded count as failures
    Metric("photo_pose_testset_photos", "photos", "none", "sum", "poses",
           "Test-set photos the protocol uploads (listed, file present, with EXIF GPS unless the set uploads every"
           " photo); more than photos_uploaded only when the harness time cap stopped the uploads."),
    Metric("photo_pose_not_uploaded", "photos", "lower", "sum", "poses",
           "photo_pose_testset_photos - photos_uploaded: photos the time cap never uploaded (failures in the"
           " *_testset rates)."),
    Metric("photo_pose_determined_rate_testset", "fraction", "higher", "pooled", "poses",
           "photo_pose_determined / photo_pose_testset_photos: never-uploaded photos fail too.", 0.01,
           pooled=("photo_pose_determined", "photo_pose_testset_photos")),
    Metric("photo_pose_gt_testset", "photos", "none", "sum", "poses",
           "Test-set photos with a verified reference pose, uploaded or not."),
    Metric("photo_pose_correct_rate_testset", "fraction", "higher", "pooled", "poses",
           "photo_pose_correct / photo_pose_gt_testset: the reference-subset pass rate with never-uploaded photos as"
           " failures.", 0.01, pooled=("photo_pose_correct", "photo_pose_gt_testset")),
    Metric("photo_pose_pass_rate_low_testset", "fraction", "higher", "pooled", "poses",
           "photo_pose_correct / photo_pose_testset_photos: lower bound of the pass rate over every test-set photo (a"
           " photo without a reference pose counts as a failure). The reference subset over-represents photos SfM"
           " placed: quote this bound pair next to photo_pose_correct_rate.", 0.01,
           pooled=("photo_pose_correct", "photo_pose_testset_photos")),
    Metric("photo_pose_pass_rate_high_testset", "fraction", "higher", "pooled", "poses",
           "photo_pose_pass / photo_pose_testset_photos: upper bound (a determined photo without a reference pose"
           " counts as a pass, an undetermined or never-uploaded one as a failure).", 0.01,
           pooled=("photo_pose_pass", "photo_pose_testset_photos")),
    Metric("photo_pose_sfm_gt", "photos", "none", "sum", "poses", "SfM-posed photos with a reference pose."),
    Metric("photo_pose_sfm_correct", "photos", "higher", "sum", "poses", "Of those, correct."),
    Metric("photo_pose_sfm_correct_rate", "fraction", "higher", "pooled", "poses",
           "photo_pose_sfm_correct / photo_pose_sfm_gt: how far the SfM poses can be trusted.", 0.01,
           pooled=("photo_pose_sfm_correct", "photo_pose_sfm_gt")),
    Metric("photo_pose_exif_gt", "photos", "none", "sum", "poses", "EXIF-posed photos with a reference pose."),
    Metric("photo_pose_exif_correct", "photos", "higher", "sum", "poses", "Of those, correct."),
    Metric("photo_pose_exif_correct_rate", "fraction", "higher", "pooled", "poses",
           "photo_pose_exif_correct / photo_pose_exif_gt.", 0.01,
           pooled=("photo_pose_exif_correct", "photo_pose_exif_gt")),
    Metric("photo_pose_position_error_median_m", "m", "lower", "median", "poses",
           "Median horizontal position error of the determined reference photos.", 0.2),
    Metric("photo_pose_heading_error_median_deg", "deg", "lower", "median", "poses",
           "Median heading error of the determined reference photos.", 0.2),
    Metric("photo_pose_within_uncertainty", "photos", "none", "sum", "poses",
           "Determined reference photos whose errors lie inside the drawn uncertainty (sp metres and sy degrees)."),
    Metric("photo_pose_uncertainty_coverage", "fraction", "none", "pooled", "poses",
           "photo_pose_within_uncertainty / photo_pose_gt_determined: position AND heading inside (each range is"
           " calibrated to about 85-90 % of its own errors, so both at once is less).", 0.02,
           pooled=("photo_pose_within_uncertainty", "photo_pose_gt_determined")),
    # per source
    Metric("photo_pose_sfm_within_uncertainty", "photos", "none", "sum", "poses",
           "SfM-posed reference photos whose position and heading errors both lie inside the drawn sp / sy."),
    Metric("photo_pose_sfm_uncertainty_coverage", "fraction", "none", "pooled", "poses",
           "photo_pose_sfm_within_uncertainty / photo_pose_sfm_gt.", 0.02,
           pooled=("photo_pose_sfm_within_uncertainty", "photo_pose_sfm_gt")),
    Metric("photo_pose_exif_within_uncertainty", "photos", "none", "sum", "poses",
           "EXIF-posed reference photos whose position and heading errors both lie inside the drawn sp / sy."),
    Metric("photo_pose_exif_uncertainty_coverage", "fraction", "none", "pooled", "poses",
           "photo_pose_exif_within_uncertainty / photo_pose_exif_gt.", 0.02,
           pooled=("photo_pose_exif_within_uncertainty", "photo_pose_exif_gt")),
    # The georegistration against the outline-corrected reference (geo_check: no footprint snap): geographic error,
    # kept apart from the relative (Sim3-aligned) gt_* errors. A model is checked when it is published, georegistered
    # and has >= STYLE_MIN_REF_CAMERAS unflagged cameras with a verified reference pose; the anchor's yaw, scale,
    # tilt and wall error come from the 3-D similarity moving it onto the reference (correction_3d: rotation from
    # the cameras' orientations, so models whose reconstructor keeps no orientations have the camera error only).
    Metric("geo_models_checked", "models", "none", "sum", "geo",
           f"Published georegistered models with at least {STYLE_MIN_REF_CAMERAS} unflagged cameras that have a verified"
           " reference pose (superseded ones included)."),
    Metric("geo_camera_error_median_m", "m", "lower", "median", "geo",
           "Horizontal distance of a checked model's cameras from their reference positions (median per model, then"
           " over the models): the geographic error of the poses the page shows.", 0.2),
    Metric("geo_yaw_error_median_deg", "deg", "lower", "median", "geo",
           "|Yaw| of the 3-D similarity moving a checked model onto the reference, median over the models: the"
           " anchor's heading error.", 0.2),
    Metric("geo_scale_error_median_pct", "%", "lower", "median", "geo",
           "|scale - 1| of that similarity, in per cent: the anchor's scale error.", 0.5),
    Metric("geo_tilt_error_median_deg", "deg", "lower", "median", "geo",
           "Tilt of that similarity: how far the model's vertical is off.", 0.1),
    Metric("geo_wall_error_p90_median_m", "m", "lower", "median", "geo",
           "90th percentile of how far that similarity moves the model's wall points (2-20 m up, in 3-D), median over"
           " the models: the geographic error at the walls.", 0.2),
    Metric("wrong_face_models_published", "models", "lower", "sum", "geo",
           f"Checked models whose yaw is {WRONG_FACE_HEADING_DEG:g} deg or more off: placed facing the wrong way (a"
           " symmetric alias, a failed yaw)."),
    Metric("wrong_face_cameras_shown", "cameras", "lower", "sum", "geo",
           "Poses the landmark page shows at the end (one per photo, crowd3d.api.camera_estimates) for photos with a"
           f" verified reference pose that look {WRONG_FACE_HEADING_DEG:g} deg or more off its heading or stand"
           f" {WRONG_FACE_BEARING_DEG:g} deg or more round the building from it (bearing from the landmark centre)."),
    # Supported coverage (coverage_bins): the OSM outer ring in ~COV_BIN_M bins; independent of sectors and of
    # facade/full models. Latest published models only (what the page shows). A wall point (2-20 m up) lands on the
    # bin of its nearest outline point when it is within COV_NEAR_M of it; a model covers a bin when COV_MIN_POINTS
    # of its points land on it. Building-local frame: the points moved by the model's similarity to the reference (the
    # georegistration error removed), error = their distance to the outline. Geographic frame: error = how far the
    # published point is from that outline position (at the moved point's height). A bin is supported at a tolerance
    # when one model has COV_MIN_POINTS points there within it.
    Metric("cov_claimed_m", "m", "none", "sum", "coverage",
           f"Outline metres on which the latest published models show walls: at least {COV_MIN_POINTS} published wall"
           f" points land on the bin (within {COV_NEAR_M:g} m). Needs only the outline.", 1.0),
    Metric("cov_unverified_m", "m", "none", "sum", "coverage",
           "Of those, the metres only models without a similarity to the reference claim (fewer than"
           f" {STYLE_MIN_REF_CAMERAS} oriented reference cameras): not checked.", 1.0),
    Metric("cov_local_supported_m_1p5", "m", "higher", "sum", "coverage",
           f"Outline metres where a model has at least {COV_MIN_POINTS} wall points that, moved by its similarity to the"
           " reference, lie within 1.5 m of the outline: supported in the building-local frame. The primary coverage"
           " metric.", 1.0),
    Metric("cov_local_supported_m_5", "m", "higher", "sum", "coverage",
           f"The same within 5 m: every covered metre (points farther than {COV_NEAR_M:g} m are not on the outline), the"
           " perimeter the checked models cover.", 1.0),
    Metric("cov_geo_supported_m_1p5", "m", "higher", "sum", "coverage",
           f"Outline metres where a model has at least {COV_MIN_POINTS} published wall points within 1.5 m (3-D) of where"
           " they are on the outline: supported geographically (the georegistration error included; expected near 0"
           " today).", 1.0),
    Metric("cov_geo_supported_m_5", "m", "higher", "sum", "coverage", "The same within 5 m.", 1.0),
    Metric("cov_local_wrong_m_1p5", "m", "lower", "sum", "coverage",
           "Covered metres no model supports within 1.5 m in the building-local frame (cov_local_supported_m_5 -"
           " cov_local_supported_m_1p5): the walls' shape is off there.", 1.0),
    Metric("cov_geo_wrong_m_5", "m", "lower", "sum", "coverage",
           "Covered metres no model supports geographically within 5 m (cov_local_supported_m_5 -"
           " cov_geo_supported_m_5).", 1.0),
    Metric("cov_local_majority_m_1p5", "m", "higher", "sum", "coverage",
           f"Outline metres where a model's points within 1.5 m in the building-local frame are at least {COV_MIN_POINTS}"
           f" and at least {COV_MAJORITY:.0%} of its moved points landing on the bin: the share-based companion of"
           " cov_local_supported_m_1p5, which counts points and so rises with point density (10 of 100 points on the"
           " wall support a bin); this one does not rise with density alone, and drops bins seen mostly behind a"
           " colonnade.", 1.0),
    Metric("cov_local_wrong_sum_m_1p5", "m", "lower", "sum", "coverage",
           "Summed over the shown models: the metres each covers but does not support within 1.5 m in the"
           " building-local frame. cov_local_wrong_m_1p5 counts a bin once, and not at all when another shown model"
           " supports it; this counts every shown model's wrong walls.", 1.0),
    Metric("cov_geo_wrong_sum_m_5", "m", "lower", "sum", "coverage",
           "The same geographically within 5 m: summed over the shown models, the metres each covers but does not"
           " support (next to cov_geo_wrong_m_5).", 1.0),
    *_calib_metrics(),
    Metric("calib_sigma_missing", "models", "lower", "sum", "calib",
           "Checked models (with the 3-D similarity) without an anchor whose georegistration left no yaw or no scale"
           " sigma (the snap then has no window to search; an anchor always has every sigma)."),
    # versions of a target, from the store (models, their jobs' photos and upload times)
    Metric("stale_overwrites", "models", "lower", "sum", "versions",
           "Published models that replaced their target's previous published model although it had photos they lack,"
           " uploaded later than any of theirs (an older input set shown over a newer one; must be 0)."),
    Metric("version_drift_median_m", "m", "lower", "median", "versions",
           "Per pair of consecutive published versions of a target, the median 3-D move of their common unflagged"
           " cameras; median over the pairs.", 0.1),
    Metric("version_drift_p90_m", "m", "lower", "median", "versions", "90th percentile of the same over the pairs.",
           0.2),
    # the structure pipeline (OSM parts, D15 / D25) in the benchmark: None until it runs there
    Metric("struct_models", "models", "none", "sum", "struct", "Stylized models built from an OSM parts snapshot."),
    Metric("struct_d15_adopted", "count", "none", "sum", "struct", "Active D15 adoptions (photo-measured heights)."),
    Metric("struct_d25_records", "count", "none", "sum", "struct", "D25 records (photo-measured frame corrections)."),
    Metric("struct_photo_share", "fraction", "none", "mean", "struct",
           "Share of the structure's numbers measured from photos (crowd3d.structmetrics).", 0.02),
    Metric("struct_checked", "count", "none", "sum", "struct",
           "Adopted heights, lower bounds and tilts checked against the reference points."),
    Metric("struct_incorrect", "count", "lower", "sum", "struct",
           "Of those, outside the D15 agreement tolerance of the reference (must be 0)."),
    Metric("struct_unbacked", "count", "lower", "sum", "struct",
           "Structure values without the photo evidence their record names (must be 0)."),
    Metric("struct_adopted_unregistered", "count", "lower", "sum", "struct",
           "Active D15 adoptions in models the registration gate refused (must be 0)."),
    Metric("struct_qa_failed", "models", "lower", "sum", "struct", "Models failing the structure's QA (must be 0)."),
    Metric("struct_time_s", "s", "lower", "sum", "struct", "Time of the structure step over the building's models.",
           5.0, 0.25),
    Metric("struct_peak_rss_mb", "MB", "lower", "max", "struct", "Peak resident memory of the structure step.",
           50.0, 0.25),
)
METRIC_BY_NAME = {m.name: m for m in METRICS}
STYLE_METRICS = tuple(m.name for m in METRICS if m.group == "style")
POSE_METRICS = tuple(m.name for m in METRICS if m.group == "poses")
GEO_METRICS = tuple(m.name for m in METRICS if m.group in ("geo", "coverage", "calib"))    # geo_scores
VERSION_METRICS = tuple(m.name for m in METRICS if m.group == "versions")                  # version_metrics
REQUEST_METRICS = ("jobs_cpu_s", "peak_rss_jobs_mb", "baseline_rss_mb", "request_latency_p50_s",
                   "request_latency_p90_s", "photo_time_to_pose_p50_s", "photo_time_to_pose_p90_s")   # request_metrics


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
    """fit_outline from two starts, keeping the closer fit: no shift (a GPS-georegistered frame is already within
    GPS error) and the shift putting the top structure on the footprint centroid (a central lantern or
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

    @property
    def rotation(self) -> np.ndarray:
        m = self.transform[:3, :3]
        return m / np.cbrt(np.linalg.det(m))


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


# ---------------------------------------------------------------- job models


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


def _round(v: Any, nd: int) -> Any:
    return round(float(v), nd) if isinstance(v, (int, float)) and math.isfinite(v) else None


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
    # {"path": cache file, "key": its key, "about": fields to record in it}: building_reference reads the reference
    # from there when the key matches, else builds it and writes it there (scripts/benchmark.py sets it)
    reference_cache: dict[str, Any] | None = None


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
                         spec.get("gt"), bool(spec.get("upload_all", False)), spec.get("reference_cache"))


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
    metrics = {m.name: None for m in METRICS}     # the groups measured later (style, poses, geo, ...) stay None till then
    metrics.update({
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
    })
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
    """Reference camera poses per photo key, moved onto the OSM outline: the independent answer to "is this
    snap right" (REAL_WORLD_EVAL 2.4, outline-corrected reference). Per camera: "xy" (horizontal ENU centre),
    "xyz" (the centre in 3-D) and "rot" (cam_from_enu of the EXIF-upright image), so that a model's tilt shows.

    Per reference component: GT-ENU from the mapped test photos' own EXIF GPS and compass (fit_gt_frame,
    product default parameters), then the reference's wall points fitted to the outline (best_outline_fit,
    rigid), then the bounded scale step of crowd3d.stylize.snap_to_footprint (its similarity ICP about the
    outline centroid, kept when it cuts the median wall distance by 10 %), then the edge-by-edge
    registration_check of the corrected reference itself ("verified"). Synthetic truth poses are ENU already.
    The reference has errors of its own (Frauenkirche: a 0.77 m rigid outline residual, a yaw held mainly by
    GPS + compass to 1-2 deg, its scale step refused by the GPS 3 sigma rule): verdicts at the 1-2 m level
    carry them in both directions (docs/STYLIZED_MODEL.md 5.2); "sigma_ref" is that error as 1 sigma
    (REFERENCE_SIGMA; 0 for generator truth). "points": per verified component, at most REFERENCE_POINTS_MAX of
    its points in all, moved like its cameras (ENU, cm; none for generator truth).
    """
    out: dict[str, Any] = {"cameras": {}, "components": {}, "points": {}, "sigma_ref": dict(REFERENCE_SIGMA)}
    if gt.get("kind") == "poses":
        for key, v in gt["poses"].items():
            c = np.asarray(v["center"], float)
            out["cameras"][key] = {"xy": c[:2], "xyz": c, "rot": np.asarray(v["rot"], float) if "rot" in v else None,
                                   "component": "truth", "verified": True}
        out["components"]["truth"] = {"verified": True, "note": "generator truth, ENU already"}
        out["sigma_ref"] = {"yaw_deg": 0.0, "scale": 0.0}
        return out
    import pycolmap

    _, st = _stylize_modules()
    components = gt.get("components") or [{"component": "0", "sparse_dir": gt["sparse_dir"], "images": gt["images"]}]
    rings = [outline] + list(holes)
    clouds: dict[str, np.ndarray] = {}
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
        cams[:, 2] *= s2
        # cam_from_enu: world -> GT-ENU is (scale x rs), then the outline correction turns about z by r2 r1
        rs = transform[:3, :3] / np.cbrt(np.linalg.det(transform[:3, :3]))
        rz = np.eye(3)
        rz[:2, :2] = r2 @ r1
        cam_rots = [r @ rs.T @ rz.T for r in rots]
        verified = bool(reg["accepted"]) and resid <= st.DEFAULT_PARAMS.snap_max_residual_m
        if verified:
            clouds[cid] = moved * np.array([1.0, 1.0, s2])     # heights scaled like the cameras'
        report.update(verified=verified, outline_fit={k: fit[k] for k in ("rotation_deg", "shift_m",
                                                                          "median_distance_after_m", "start")},
                      scale_step={"scale": round(s2, 4), "used": bool(use_scale),
                                  "rotation_deg": round(math.degrees(math.atan2(r2[1, 0], r2[0, 0])), 2)},
                      median_wall_distance_m=round(resid, 3),
                      registration={k: reg.get(k) for k in ("accepted", "reason", "voting_edges", "verified_edges",
                                                            "verified_share")})
        for key, xyz, rot in zip(keys, cams, cam_rots, strict=True):
            out["cameras"].setdefault(key, {"xy": xyz[:2], "xyz": xyz, "rot": rot, "component": cid, "verified": verified})
    total = sum(len(c) for c in clouds.values())
    rng = np.random.default_rng(0)
    for cid, c in clouds.items():       # each component keeps its share of REFERENCE_POINTS_MAX
        n = len(c) if total <= REFERENCE_POINTS_MAX else REFERENCE_POINTS_MAX * len(c) // total
        out["points"][cid] = np.round(c[np.sort(rng.choice(len(c), n, replace=False))] if n < len(c) else c, 2)
    return out


def correction_3d(model_rots: np.ndarray, model_xyz: np.ndarray, ref_rots: np.ndarray, ref_xyz: np.ndarray,
                  walls: np.ndarray | None) -> dict[str, Any]:
    """The 3-D similarity moving the model onto the reference (p_ref = s Q p + t), with Q from the cameras'
    orientations (median-robust chordal mean of ref_i^T model_i; cameras more than max(3 x median, 2 deg) off
    it are left out) and s, t from their centres (refit once without cameras off by more than
    max(3 x median, 5 m)), and how far it moves the wall points in 3-D (median, 90th percentile). Camera
    centres alone cannot see a tilt (they lie near one horizontal plane); their orientations can, and a tilt
    about the cameras lifts or sinks the walls 40-250 m away (Frauenkirche SW#14: 2.9 deg, 3-4 m). The
    similarity itself is "scale", "rotation" (Q, 3 x 3) and "translation_m" (t)."""
    mr, mx = np.asarray(model_rots, float), np.asarray(model_xyz, float)
    rr, rx = np.asarray(ref_rots, float), np.asarray(ref_xyz, float)
    qs = np.einsum("nji,njk->nik", rr, mr)            # ref_i^T model_i: model ENU -> reference ENU
    q = mean_rotation(qs)
    ang = np.array([rotation_angle_deg(qi, q) for qi in qs])
    keep_r = ang <= max(3.0 * float(np.median(ang)), 2.0)
    if keep_r.sum() >= 2:
        q = mean_rotation(qs[keep_r])
    a = mx @ q.T

    def fit(sel: np.ndarray) -> tuple[float, np.ndarray]:
        am, bm = a[sel].mean(axis=0), rx[sel].mean(axis=0)
        den = float(((a[sel] - am) ** 2).sum())
        s_ = float(((a[sel] - am) * (rx[sel] - bm)).sum() / den) if den > 1e-9 else 1.0
        return s_, bm - s_ * am

    keep = np.ones(len(a), bool)
    d = np.linalg.norm(rx - np.median(rx, axis=0), axis=1)
    keep &= d <= max(5.0 * float(np.median(d)), 50.0)
    if keep.sum() < 3:
        keep[:] = True
    s, t = fit(keep)
    err = np.linalg.norm(s * a + t - rx, axis=1)
    keep = err <= max(3.0 * float(np.median(err[keep])), 5.0)
    if keep.sum() >= 3:
        s, t = fit(keep)
        err = np.linalg.norm(s * a + t - rx, axis=1)
    else:
        keep[:] = True
    up = q[:, 2]
    out = {"rotation_deg": round(math.degrees(math.acos(min(1.0, max(-1.0, (np.trace(q) - 1.0) / 2.0)))), 2),
           "yaw_deg": round(math.degrees(math.atan2(q[1, 0] - q[0, 1], q[0, 0] + q[1, 1])), 2),
           "tilt_deg": round(math.degrees(math.acos(min(1.0, max(-1.0, float(up[2]))))), 2),
           "scale": round(s, 4), "cameras_used": int(keep.sum()), "orientations_used": int(keep_r.sum()),
           "residual_median_m": round(float(np.median(err[keep])), 2),
           "rotation": np.round(q, 6).tolist(), "translation_m": np.round(t, 3).tolist()}
    if walls is not None and len(walls) and np.asarray(walls).shape[1] == 3:
        w = np.asarray(walls, float)
        moved = s * w @ q.T + t - w
        dist = np.linalg.norm(moved, axis=1)
        out.update(walls_median_m=round(float(np.median(dist)), 2), walls_p90_m=round(float(np.percentile(dist, 90)), 2),
                   walls_vertical_median_m=round(float(np.median(moved[:, 2])), 2))
    return out


def reference_check(cameras: dict[str, dict[str, Any]], photo_key: dict[str, str], reference: dict[str, Any],
                    snap_meta: dict[str, Any] | None, anchor: np.ndarray,
                    walls: np.ndarray | None = None,
                    rotations: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    """A model's cameras against the outline-corrected reference, before and after its footprint snap.

    Only unflagged cameras count (flagged ones are withheld from users and may sit kilometres away).
    correction_after = the 2-D similarity the snapped model's camera centres still need (camera_correction);
    correction_3d_after = the 3-D one with the rotation from the cameras' orientations (correction_3d; needs
    `rotations`, the model's cam_from_enu per photo id before the snap, and reference orientations); walls =
    the model's (snapped) wall points, (n, 2) or (n, 3). A passing snap is "right" (correction_is_right) when
    it turns the model by at most STYLE_RIGHT_ROT_DEG, moves its wall points by at most STYLE_RIGHT_SHIFT_M
    (90th percentile; in 3-D when the orientations are known), rescales it by at most STYLE_RIGHT_SCALE and its
    cameras' horizontal centres are within STYLE_RIGHT_CAMERA_M of the reference (median). correction_before /
    placed_right_before = the same for the georegistered model before the snap (what the snap should find).
    error_after_m = the snapped cameras' horizontal distance to their reference positions."""
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
    w3 = None if walls is None else np.asarray(walls, float).reshape(len(walls), -1)
    walls_xy = None if w3 is None else w3[:, :2]
    unsnapped_anchor, unsnapped_walls = anchor, w3
    if snap is not None:      # where the walls were before the snap moved them onto the anchor
        r = snap.rotation()[:2, :2]
        unsnapped_anchor = np.linalg.solve(snap.scale * r, anchor - snap.shift)
        if w3 is not None:
            unsnapped_walls = w3.copy()
            unsnapped_walls[:, :2] = (w3[:, :2] - snap.shift) @ np.linalg.inv(snap.scale * r).T
            if w3.shape[1] == 3:
                unsnapped_walls[:, 2] /= snap.scale
    out["correction_before"] = camera_correction(model[:, :2], ref, unsnapped_anchor,
                                                 None if unsnapped_walls is None else unsnapped_walls[:, :2])
    out["correction_after"] = camera_correction(snapped[:, :2], ref, anchor, walls_xy)
    with_rot = [k for k, (pid, key) in enumerate(pairs) if rotations is not None and pid in rotations
                and reference["cameras"][key].get("rot") is not None and reference["cameras"][key].get("xyz") is not None]
    if len(with_rot) >= STYLE_MIN_REF_CAMERAS and w3 is not None and w3.shape[1] == 3:
        mr = np.array([rotations[pairs[k][0]] for k in with_rot], dtype=float)
        rr = np.array([reference["cameras"][pairs[k][1]]["rot"] for k in with_rot], dtype=float)
        rx = np.array([reference["cameras"][pairs[k][1]]["xyz"] for k in with_rot], dtype=float)
        out["correction_3d_before"] = correction_3d(mr, model[with_rot], rr, rx, unsnapped_walls)
        sr = snap.rotation() if snap is not None else np.eye(3)
        out["correction_3d_after"] = correction_3d(mr @ sr.T, snapped[with_rot], rr, rx, w3)
    out["placed_right_before"] = correction_is_right(out["correction_before"], out.get("correction_3d_before"),
                                                     out["error_before_m"]["median"])
    right = correction_is_right(out["correction_after"], out.get("correction_3d_after"), out["error_after_m"]["median"])
    out["verdict"] = "right" if right else "wrong"
    if not right:
        out["wrong_because"] = wrong_because(out["correction_after"], out.get("correction_3d_after"),
                                             out["error_after_m"]["median"])
    return out


def _walls_moved(c: dict[str, Any], c3: dict[str, Any] | None) -> float:
    walls = c.get("walls_p90_m", c["at_walls_m"])
    if c3 is not None and c3.get("walls_p90_m") is not None:
        walls = max(walls, c3["walls_p90_m"])
    return walls


def correction_is_right(c: dict[str, Any], c3: dict[str, Any] | None = None, camera_median_m: float | None = None) -> bool:
    """A placement whose remaining correction to the reference is within the STYLE_RIGHT_* limits: turn, scale,
    wall points moved (90th percentile over them when they were given, else at the anchor; the larger of the
    2-D and the 3-D correction, c3, when the orientations were known) and, when given, the median horizontal
    camera-centre error (the user's definition of a right registration, STYLIZED_MODEL 3.2 / 7)."""
    return bool(abs(c["rotation_deg"]) <= STYLE_RIGHT_ROT_DEG and _walls_moved(c, c3) <= STYLE_RIGHT_SHIFT_M
                and abs(c["scale"] - 1.0) <= STYLE_RIGHT_SCALE
                and (camera_median_m is None or camera_median_m <= STYLE_RIGHT_CAMERA_M))


def wrong_because(c: dict[str, Any], c3: dict[str, Any] | None, camera_median_m: float | None) -> list[str]:
    """Which STYLE_RIGHT_* limits a placement breaks (for the report)."""
    out = []
    if abs(c["rotation_deg"]) > STYLE_RIGHT_ROT_DEG:
        out.append("turn")
    if _walls_moved(c, c3) > STYLE_RIGHT_SHIFT_M:
        out.append("walls_3d" if c3 is not None and c3.get("walls_p90_m", 0.0) > STYLE_RIGHT_SHIFT_M
                   and c.get("walls_p90_m", c["at_walls_m"]) <= STYLE_RIGHT_SHIFT_M else "walls")
    if abs(c["scale"] - 1.0) > STYLE_RIGHT_SCALE:
        out.append("scale")
    if camera_median_m is not None and camera_median_m > STYLE_RIGHT_CAMERA_M:
        out.append("cameras")
    return out


def diagnose_gate(rec: dict[str, Any], outline_fit: dict[str, Any] | None) -> dict[str, Any]:
    """Why an eligible model failed the registration gate: too_few_points, single_wall (fewer than 2 outline
    edges carry points, or the verified ones are all parallel: a slide along the wall is unobservable),
    ambiguous (two placements fit about equally well), gps_disagrees (snapped, the cameras would stand nearer
    to or farther from the walls than their own GPS says: points and cameras disagree), shape (the outline
    asks for a rescale beyond the bound), scale_uncertain (a one-sided model whose GPS scale is too uncertain
    for the outline to fix), yaw_unverifiable (a near-circle: translation only, and the registered yaw is not known well
    enough to colour without a check), loose_fit (the snapped walls follow the outline only loosely on
    average: the fit cannot fix the yaw), tilt (the cameras' heights say the model is tilted, or are too few to
    tell), unseen_face (snapped, a face carries wall points no camera can see past the building), no_cameras
    (a georegistered model without camera positions), osm_outline (the reference itself does not sit on the outline),
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
    elif "is tilted" in reason or "the model's tilt" in reason:
        out["cause"] = "tilt"
    elif "no camera can see" in reason:
        out["cause"] = "unseen_face"
    elif "no camera positions" in reason:
        out["cause"] = "no_cameras"
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


def stylize_model_record(m: dict[str, Any], *, work_dir: Path, scene_base: dict[str, Any],
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
    json_path.write_text(json.dumps(full, indent=1, ensure_ascii=False, default=_json_default), encoding="utf-8")
    st.write_osm_fix_report(s, out_dir, scene_base["slug"])   # D25's suggested OSM fixes beside the summary
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
                                           *wall_anchor(moved, scene_base["footprint"], scene_base["holes"]),
                                           rotations={c.photo_id: c.rot for c in cams})
        comps = {reference["cameras"][photo_key[pid]]["component"] for pid in cameras
                 if photo_key.get(pid) in reference["cameras"]}
        if comps:
            rec["reference_component_verified"] = all(reference["components"].get(c, {}).get("verified") for c in comps)
    return rec


def wall_anchor(points: np.ndarray, footprint: np.ndarray, holes: list[np.ndarray],
                max_points: int = 5000) -> tuple[np.ndarray, np.ndarray | None]:
    """Where a (snapped) model's walls are: wall-height points within reg_edge_limit_m of the outline (within
    10 m when fewer than 20 are), subsampled ((n, 3) ENU), and their horizontal median (the outline's centroid
    without any)."""
    _, st = _stylize_modules()
    band = (points[:, 2] >= 2.0) & (points[:, 2] <= 20.0)
    d = st.distance_to_rings(points[:, :2], [footprint] + list(holes))[0]
    near = band & (d <= st.DEFAULT_PARAMS.reg_edge_limit_m)
    if near.sum() < 20:
        near = band & (d <= 10.0)
    if near.sum() < 20:
        return st.polygon_centroid(footprint), None
    walls = points[near, :3]
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
        "style_coloured_walls_wrong": sum(r["walls"]["coloured"] for r in checked
                                          if r["reference"]["verdict"] == "wrong"),
        "style_rejected_placed_right": sum(
            r.get("status") == "ok" and not r["gate"]["passed"] and bool((r.get("reference") or {}).get("placed_right_before"))
            for r in eligible),
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
        "style_ref_camera_median_worst_m": max((r["reference"]["error_after_m"]["median"] for r in checked),
                                               default=None),
        "style_ref_camera_error_max_m": max((r["reference"]["error_after_m"].get("max") for r in checked
                                             if r["reference"]["error_after_m"].get("max") is not None), default=None),
        "style_ref_walls_p90_worst_m": max((_walls_moved(r["reference"]["correction_after"],
                                                         r["reference"].get("correction_3d_after"))
                                            for r in checked if r["reference"].get("correction_after")), default=None),
        "style_incomplete": sum(str(r.get("status", "")).startswith("skipped (time cap)") for r in records),
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


def stylize_published(models: list[dict[str, Any]], b: CrowdBuilding, *, work_dir: Path,
                      out_dir: Path, photo_key: dict[str, str], metas: dict[str, PhotoMeta], enu: LocalENU,
                      deadline: float | None = None,
                      reference_result: tuple[dict | None, dict | None] | None = None) -> dict[str, Any]:
    """Stylize every published model (latest ones first), check passing snaps against the outline-corrected
    reference where there is one, diagnose failing eligible models; returns records, best model and metrics.
    reference_result: (reference, report) when the caller built it already (building_reference)."""
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
    reference, ref_report = reference_result if reference_result is not None else building_reference(b, metas, enu)[:2]
    fits = {m["id"]: (m.get("outline") or {}).get("fit") for m in models}
    records = []
    for k, m in enumerate(published):
        if deadline is not None and time.monotonic() > deadline:
            records.append({"model_id": m["id"], "kind": m["kind"], "sector": m["sector"], "request": m["request"],
                            "latest_published": m["latest_published"], "eligible": bool(m["georegistered"]),
                            "status": "skipped (time cap)"})
            continue
        try:
            rec = stylize_model_record(m, work_dir=work_dir, scene_base=scene_base,
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


# ---------------------------------------------------------------- crowd protocol: photo poses


def building_reference(b: CrowdBuilding, metas: dict[str, PhotoMeta],
                       enu: LocalENU) -> tuple[dict | None, dict | None, dict | None]:
    """(outline-corrected reference, its report, reference poses for gt_model_check) of a building, built once per
    run for the GT model checks, the style check and the photo poses: outline_corrected_reference and load_gt_poses.
    With b.reference_cache, read from that file when its key matches (the reference models are then never loaded
    and may be deleted), else built and written there. (None, report, poses) when the outline-corrected reference
    cannot be built; (None, None, None) without any reference."""
    cache = b.reference_cache
    if cache and Path(cache["path"]).is_file():
        doc = json.loads(Path(cache["path"]).read_text(encoding="utf-8"))
        if doc.get("key") == cache["key"]:
            reference, gt_poses = reference_from_doc(doc)
            return reference, {**doc["report"], "cached": cache["path"]}, gt_poses
    if b.gt is None:
        return None, None, None
    gt_poses = load_gt_poses(b.gt)
    if b.gt.get("kind") != "poses" and b.outline_path is None:
        return None, {"message": "no OSM outline: the reference cannot be corrected onto it"}, gt_poses
    t = time.monotonic()
    try:
        if b.outline_path is not None:
            sm, _ = _stylize_modules()
            footprint, holes = sm.load_footprint_rings(b.outline_path, enu)
        else:
            footprint, holes = np.zeros((0, 2)), []
        reference = outline_corrected_reference(b.gt, metas, footprint, holes, enu)
    except Exception as e:   # never lose the run over the reference check
        log.exception("[%s] outline-corrected reference failed", b.slug)
        return None, {"error": f"{type(e).__name__}: {e}"}, gt_poses
    report = {"components": reference["components"], "cameras": len(reference["cameras"]),
              "seconds": round(time.monotonic() - t, 1)}
    if cache:
        doc = {**reference_cache_doc(reference, report, gt_poses), **(cache.get("about") or {}), "key": cache["key"],
               "reference_version": REFERENCE_VERSION,
               "built": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        path = Path(cache["path"])
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".tmp").write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        path.with_suffix(".tmp").replace(path)
    return reference, report, gt_poses


def reference_cache_doc(reference: dict[str, Any], report: dict[str, Any] | None,
                        gt_poses: dict[str, tuple] | None) -> dict[str, Any]:
    """The JSON of a reference for the cache, at full float precision so that reference_from_doc gives back an
    equal one: its cameras, points and sigma, its report, and the reference poses (load_gt_poses)."""
    def arr(v: Any) -> Any:
        return None if v is None else np.asarray(v, float).tolist()

    return {"report": json.loads(json.dumps(report or {}, default=_json_default)),
            "cameras": {k: {"xy": arr(c["xy"]), "xyz": arr(c.get("xyz")), "rot": arr(c.get("rot")),
                            "component": str(c["component"]), "verified": bool(c["verified"])}
                        for k, c in reference["cameras"].items()},
            "points": {cid: arr(p) for cid, p in reference["points"].items()},
            "sigma_ref": reference["sigma_ref"],
            "gt_poses": {k: {"rot": arr(r), "center": arr(c), "component": comp}
                         for k, (r, c, comp) in (gt_poses or {}).items()}}


def reference_from_doc(doc: dict[str, Any]) -> tuple[dict[str, Any], dict[str, tuple]]:
    """(reference, reference poses) back from a reference_cache_doc."""
    def arr(v: Any) -> np.ndarray | None:
        return None if v is None else np.asarray(v, float)

    cameras = {k: {"xy": arr(c["xy"]), "xyz": arr(c["xyz"]), "rot": arr(c["rot"]), "component": c["component"],
                   "verified": c["verified"]} for k, c in doc["cameras"].items()}
    reference = {"cameras": cameras, "components": doc["report"].get("components") or {},
                 "points": {cid: np.asarray(p, float).reshape(-1, 3) for cid, p in doc["points"].items()},
                 "sigma_ref": doc["sigma_ref"]}
    return reference, {k: (arr(v["rot"]), arr(v["center"]), v["component"]) for k, v in doc["gt_poses"].items()}


def reference_poses(reference: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """key -> {"xy", "heading_deg", "component"} of the verified cameras of an outline-corrected reference."""
    out: dict[str, dict[str, Any]] = {}
    for key, c in ((reference or {}).get("cameras") or {}).items():
        if not c.get("verified") or c.get("rot") is None:
            continue
        f = np.asarray(c["rot"], float)[2]
        out[key] = {"xy": [float(c["xy"][0]), float(c["xy"][1])],
                    "heading_deg": heading_from_en(float(f[0]), float(f[1])), "component": str(c.get("component"))}
    return out


def accepted_snaps(style: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    """model_id -> the footprint snap the stylization accepted for it (refused snaps move nothing)."""
    out = {}
    for r in ((style or {}).get("records") or []):
        snap = r.get("snap") or {}
        if r.get("status") == "ok" and snap.get("accepted") and snap.get("meta"):
            out[r["model_id"]] = snap["meta"]
    return out


R_NOT_UPLOADED = "not_uploaded"     # a test-set photo the harness time cap never uploaded


def photo_pose_records(rows: list[dict[str, Any]], poses: dict[str, Any],
                       reference: dict[str, dict[str, Any]] | None,
                       planned_keys: list[str] | None = None) -> list[dict[str, Any]]:
    """One record per uploaded photo (the harness's photo rows): the pose crowd3d.photoposes gives it and, where the
    photo has a verified reference pose, its errors and verdicts. planned_keys: every test-set photo the protocol
    would upload; the ones the time cap never uploaded get a record too (uploaded False, reason not_uploaded)."""
    from crowd3d.photoposes import R_REJECTED

    recs = []
    for row in rows:
        pose = poses.get(row.get("photo_id")) if row.get("photo_id") else None
        ref = (reference or {}).get(row["key"])
        rec: dict[str, Any] = {
            "key": row["key"], "photo_id": row.get("photo_id"), "accepted": bool(row.get("accepted")),
            "src": pose.src if pose else None, "determined": bool(pose is not None and pose.determined),
            "reason": (pose.reason if pose is not None else R_REJECTED if not row.get("accepted") else "not_stored"),
            "gt": ref is not None, "uploaded": True,
        }
        if pose is not None and pose.p is not None:
            rec.update(p=[round(float(v), 2) for v in pose.p], yaw=None if pose.yaw is None else round(pose.yaw, 2),
                       sp=None if pose.sp is None else round(pose.sp, 2),
                       sy=None if pose.sy is None else round(pose.sy, 2), model=pose.model_id, snapped=pose.snapped,
                       approx=pose.approx)
        if ref is not None and pose is not None and pose.p is not None:
            pos_err = float(math.hypot(pose.p[0] - ref["xy"][0], pose.p[1] - ref["xy"][1]))
            rec["pos_err_m"] = round(pos_err, 2)
            if pose.determined:
                head_err = abs(angle_diff_deg(pose.yaw, ref["heading_deg"]))
                rec.update(head_err_deg=round(head_err, 2),
                           correct=pos_err <= POSE_RIGHT_M and head_err <= POSE_RIGHT_DEG,
                           strict=pos_err <= POSE_STRICT_M and head_err <= POSE_STRICT_DEG,
                           within_uncertainty=bool(pose.sp is not None and pose.sy is not None
                                                   and pos_err <= pose.sp and head_err <= pose.sy))
        recs.append(rec)
    uploaded = {r["key"] for r in recs}
    for key in planned_keys or ():
        if key not in uploaded:
            recs.append({"key": key, "photo_id": None, "accepted": False, "src": None, "determined": False,
                         "reason": R_NOT_UPLOADED, "gt": key in (reference or {}), "uploaded": False})
            uploaded.add(key)
    return recs


def pose_metrics(records: list[dict[str, Any]], has_reference: bool) -> dict[str, Any]:
    """The photo_pose_* METRICS of one building from its photo_pose_records (GT metrics None without a verified
    reference pose for any photo of the building). Every metric is over the uploaded photos except the *_testset
    ones, where test-set photos the time cap never uploaded count as failures."""
    planned = records
    records = [r for r in records if r.get("uploaded", True)]
    det = [r for r in records if r["determined"]]
    gt = [r for r in records if r["gt"]]
    gt_det = [r for r in gt if r["determined"]]
    n = len(records)

    def rate(a: int, b: int) -> float | None:
        return round(a / b, 4) if b else None

    m: dict[str, Any] = {
        "photo_pose_determined": len(det),
        "photo_pose_determined_rate": rate(len(det), n),
        "photo_pose_sfm": sum(r["src"] == "sfm" for r in records),
        "photo_pose_loc": sum(r["src"] == "loc" for r in records),
        "photo_pose_exif": sum(r["src"] == "exif" for r in records),
        "photo_pose_gps_only": sum(r["src"] == "gps" for r in records),
        "photo_pose_pass": sum(r["determined"] and r.get("correct") is not False for r in records),
    }
    m["photo_pose_pass_rate"] = rate(m["photo_pose_pass"], n)
    m.update({"photo_pose_testset_photos": len(planned), "photo_pose_not_uploaded": len(planned) - n,
              "photo_pose_determined_rate_testset": rate(len(det), len(planned)),
              "photo_pose_pass_rate_high_testset": rate(m["photo_pose_pass"], len(planned))})
    if not has_reference:
        m.update({k: None for k in POSE_METRICS if k not in m})
        return m
    correct = sum(bool(r.get("correct")) for r in gt)
    strict = sum(bool(r.get("strict")) for r in gt)
    within = sum(bool(r.get("within_uncertainty")) for r in gt_det)
    m.update({
        "photo_pose_gt": len(gt), "photo_pose_gt_determined": len(gt_det), "photo_pose_correct": correct,
        "photo_pose_correct_rate": rate(correct, len(gt)), "photo_pose_wrong": len(gt_det) - correct,
        "photo_pose_correct_strict": strict, "photo_pose_correct_strict_rate": rate(strict, len(gt)),
        "photo_pose_position_error_median_m": _median(r["pos_err_m"] for r in gt_det),
        "photo_pose_heading_error_median_deg": _median(r["head_err_deg"] for r in gt_det),
        "photo_pose_within_uncertainty": within,
        "photo_pose_uncertainty_coverage": rate(within, len(gt_det)),
        "photo_pose_gt_testset": sum(r["gt"] for r in planned),
        "photo_pose_correct_rate_testset": rate(correct, sum(r["gt"] for r in planned)),
        "photo_pose_pass_rate_low_testset": rate(correct, len(planned)),
    })
    for src in ("sfm", "exif"):
        g = [r for r in gt_det if r["src"] == src]
        ok = sum(bool(r.get("correct")) for r in g)
        inside = sum(bool(r.get("within_uncertainty")) for r in g)
        m.update({f"photo_pose_{src}_gt": len(g), f"photo_pose_{src}_correct": ok,
                  f"photo_pose_{src}_correct_rate": rate(ok, len(g)),
                  f"photo_pose_{src}_within_uncertainty": inside,
                  f"photo_pose_{src}_uncertainty_coverage": rate(inside, len(g))})
    assert set(m) == set(POSE_METRICS), set(m) ^ set(POSE_METRICS)
    return m


def pose_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts behind the photo_pose_* metrics: by source, why photos stayed undetermined, the wrong ones by source
    (uploaded photos; never-uploaded test-set photos only in not_uploaded / not_uploaded_gt)."""
    missing = [r for r in records if not r.get("uploaded", True)]
    records = [r for r in records if r.get("uploaded", True)]
    return {
        "by_source": dict(Counter(r["src"] or "none" for r in records)),
        "undetermined": dict(Counter(r["reason"] or "?" for r in records if not r["determined"])),
        "wrong_by_source": dict(Counter(r["src"] for r in records if r.get("correct") is False)),
        "gt": sum(r["gt"] for r in records),
        "gt_undetermined": dict(Counter(r["reason"] or "?" for r in records if r["gt"] and not r["determined"])),
        "approx": dict(Counter(r["src"] for r in records if r.get("approx"))),
        "not_uploaded": len(missing), "not_uploaded_gt": sum(r["gt"] for r in missing),
    }


def run_photo_poses(rows: list[dict[str, Any]], photos: list, models: list, lat: float, lon: float, *,
                    style: dict[str, Any] | None, reference: dict[str, Any] | None, camera_height_m: float,
                    heading_refs: dict[str, str | None] | None = None,
                    planned_keys: list[str] | None = None) -> dict[str, Any]:
    """Photo poses of a finished building run: metrics, per-photo records and their summary. planned_keys: the
    test-set photos the protocol would upload (photo_pose_records)."""
    from crowd3d.photoposes import photo_poses

    poses = photo_poses(photos, models, lat, lon, snaps=accepted_snaps(style), heading_refs=heading_refs,
                        camera_height_m=camera_height_m)
    refs = reference_poses(reference)
    records = photo_pose_records(rows, {p.photo_id: p for p in poses}, refs, planned_keys)
    return {"metrics": pose_metrics(records, bool(refs)), "records": records, "summary": pose_summary(records),
            "thresholds": {"right_m": POSE_RIGHT_M, "right_deg": POSE_RIGHT_DEG, "strict_m": POSE_STRICT_M,
                           "strict_deg": POSE_STRICT_DEG}}


# ---------------------------------------------------------------- crowd protocol: cost, versions, geographic error


def cpu_seconds() -> float:
    """User + system CPU time of this process (every thread) and of its finished child processes."""
    own, kids = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
    return own.ru_utime + own.ru_stime + kids.ru_utime + kids.ru_stime


def _status_mb(field: str) -> float | None:
    """A memory field of /proc/self/status (VmRSS, VmHWM) in MB; None where there is none."""
    try:
        with open("/proc/self/status", encoding="ascii") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def reset_peak_rss() -> bool:
    """Set this process's peak resident memory (VmHWM) back to its current size (Linux: "5" into clear_refs); False
    where that is not allowed (peak_rss_mb is then the process peak so far)."""
    try:
        with open("/proc/self/clear_refs", "w", encoding="ascii") as f:
            f.write("5")
        return True
    except OSError:
        return False


def peak_rss_mb() -> float:
    """Peak resident memory since the last reset_peak_rss (VmHWM), else of the whole process (ru_maxrss, KB on
    Linux)."""
    hwm = _status_mb("VmHWM")
    return hwm if hwm is not None else resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def dir_size_mb(path: Path, skip: tuple[str, ...] = ()) -> float:
    """Bytes of the files under path in MB (symlinks not followed; the top-level dirs named in skip left out)."""
    total = 0
    for root, dirs, files in os.walk(path):
        if root == str(path):
            dirs[:] = [d for d in dirs if d not in skip]
        total += sum(os.lstat(os.path.join(root, f)).st_size for f in files)
    return round(total / 2**20, 1)


def _percentile(values, q: float) -> float | None:
    v = [float(x) for x in values if x is not None]
    return round(float(np.percentile(v, q)), 1) if v else None


def request_metrics(log: list[dict[str, Any]], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """The REQUEST_METRICS of a run from its request log and photo rows (summary.json "requests" and "photos"); what an
    older log does not record stays None. Latency is over the requests that ran jobs; the peak memory is that of the
    request whose jobs went highest, with the memory just before them."""
    ran = [r for r in log if r.get("jobs")]
    peaks = [r for r in ran if r.get("peak_rss_mb") is not None]
    top = max(peaks, key=lambda r: r["peak_rss_mb"]) if peaks else None
    cpu = [r["cpu_s"] for r in log if r.get("cpu_s") is not None]
    latency = [r["ingest_s"] + r["jobs_s"] for r in ran]
    to_pose = [r.get("time_to_pose_s") for r in rows]
    return {"jobs_cpu_s": round(sum(cpu), 1) if cpu else None,
            "peak_rss_jobs_mb": top["peak_rss_mb"] if top else None,
            "baseline_rss_mb": top.get("rss_before_mb") if top else None,
            "request_latency_p50_s": _percentile(latency, 50), "request_latency_p90_s": _percentile(latency, 90),
            "photo_time_to_pose_p50_s": _percentile(to_pose, 50), "photo_time_to_pose_p90_s": _percentile(to_pose, 90)}


def version_metrics(models: list, jobs: list, photos: list) -> dict[str, Any]:
    """stale_overwrites and version_drift_* from a run's store (ModelRecords, Jobs, PhotoRecords): per target (kind,
    sector) its published models in publication order, each replacing the one before on the landmark page."""
    uploaded = {p.id: p.uploaded_at for p in photos}
    inputs = {j.id: set(j.photo_ids) for j in jobs}

    def newest(ids: set[str]) -> float:
        return max((uploaded.get(p, 0.0) for p in ids), default=0.0)

    targets: dict[tuple, list] = {}
    for m in sorted(models, key=lambda m: (m.created_at, m.id)):
        if m.meta.get("published", True):
            targets.setdefault((m.kind, m.sector), []).append(m)
    stale, drift = 0, []
    for seq in targets.values():
        for old, new in zip(seq, seq[1:]):
            a, b = inputs.get(old.job_id, set()), inputs.get(new.job_id, set())
            stale += bool(a - b) and newest(a) > newest(b)
            ca, cb = old.meta.get("cameras") or {}, new.meta.get("cameras") or {}
            moved = [float(np.linalg.norm(np.subtract(cb[p]["enu"], ca[p]["enu"]))) for p in sorted(ca.keys() & cb.keys())
                     if not ca[p].get("flags") and not cb[p].get("flags")
                     and ca[p].get("enu") is not None and cb[p].get("enu") is not None]
            if moved:
                drift.append(float(np.median(moved)))
    return {"stale_overwrites": int(stale), "version_drift_median_m": _median(drift),
            "version_drift_p90_m": round(float(np.percentile(drift, 90)), 3) if drift else None}


def model_enu_rotations(job_dir: Path, cameras: dict[str, dict[str, Any]]) -> dict[str, np.ndarray]:
    """photo id -> cam_from_enu of the posed images of a job's kept COLMAP model (job_dir/sparse/0): its rotations
    turned into ENU by the similarity taking its camera centres onto the model JSON's camera positions, as
    scripts/stylize_model.load_job does. {} when the job kept no model (another reconstructor, a pruned work dir)."""
    sparse = job_dir / "sparse" / "0"
    if not any((sparse / f).is_file() for f in ("images.bin", "images.txt")):
        return {}
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse))
    ims = {photo_id_from_image_name(im.name): im for im in rec.images.values() if im.has_pose}
    both = [pid for pid in ims if (cameras.get(pid) or {}).get("enu") is not None]
    if len(both) < 3:
        return {}
    _, rot, _ = umeyama(np.array([np.asarray(ims[p].projection_center(), float) for p in both]),
                        np.array([cameras[p]["enu"] for p in both], float))
    return {pid: np.asarray(im.cam_from_world().rotation.matrix()) @ rot.T for pid, im in ims.items()}


def geo_check(cameras: dict[str, dict[str, Any]], rotations: dict[str, np.ndarray], photo_key: dict[str, str],
              reference: dict[str, Any], walls: np.ndarray | None) -> dict[str, Any]:
    """A published model's georegistration (no footprint snap) against the outline-corrected reference: the horizontal
    distance of its unflagged cameras from their verified reference positions and, with >= STYLE_MIN_REF_CAMERAS of
    them oriented (rotations: cam_from_enu per photo id), the 3-D similarity moving it onto the reference
    (correction_3d: its yaw, tilt and scale, how far it moves the wall points, and the similarity as "sim3")."""
    pairs = [(pid, photo_key[pid]) for pid in sorted(cameras)
             if photo_key.get(pid) in reference["cameras"] and reference["cameras"][photo_key[pid]]["verified"]
             and not cameras[pid].get("flags") and cameras[pid].get("enu") is not None]
    out: dict[str, Any] = {"n": len(pairs)}
    if len(pairs) < STYLE_MIN_REF_CAMERAS:
        return out
    model = np.array([cameras[pid]["enu"] for pid, _ in pairs], float)
    err = np.linalg.norm(model[:, :2] - np.array([reference["cameras"][k]["xy"] for _, k in pairs], float), axis=1)
    out.update(camera_median_m=round(float(np.median(err)), 2), camera_max_m=round(float(err.max()), 2))
    oriented = [k for k, (pid, key) in enumerate(pairs) if pid in rotations
                and reference["cameras"][key].get("rot") is not None and reference["cameras"][key].get("xyz") is not None]
    if len(oriented) >= STYLE_MIN_REF_CAMERAS:
        refs = [reference["cameras"][pairs[k][1]] for k in oriented]
        c3 = correction_3d(np.array([rotations[pairs[k][0]] for k in oriented]), model[oriented],
                           np.array([r["rot"] for r in refs], float), np.array([r["xyz"] for r in refs], float), walls)
        out.update({k: c3.get(k) for k in ("yaw_deg", "tilt_deg", "scale", "walls_p90_m", "residual_median_m",
                                           "orientations_used")},
                   sim3={k: c3[k] for k in ("scale", "rotation", "translation_m")})
    return out


def outline_arc(points_xy: np.ndarray, ring: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For 2-D points: distance to the closed ring, the closest ring point and its arc length along the ring from
    vertex 0 (distance_to_outline, in chunks)."""
    a = np.asarray(ring, float)
    ab = np.roll(a, -1, axis=0) - a
    length = np.sqrt((ab**2).sum(axis=1))
    start = np.concatenate([[0.0], np.cumsum(length)[:-1]])
    pts = np.asarray(points_xy, float).reshape(-1, 2)
    d, c, s = np.zeros(len(pts)), np.zeros((len(pts), 2)), np.zeros(len(pts))
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(axis=2) / np.maximum(length**2, 1e-12)[None], 0.0, 1.0)
        closest = a[None] + t[..., None] * ab[None]
        dist = np.linalg.norm(p[:, None, :] - closest, axis=2)
        k = dist.argmin(axis=1)
        rows = np.arange(len(p))
        d[i:i + chunk], c[i:i + chunk], s[i:i + chunk] = dist[rows, k], closest[rows, k], start[k] + t[rows, k] * length[k]
    return d, c, s


def outline_bins(ring: np.ndarray) -> tuple[int, float]:
    """(number, length) of the coverage bins of an outline ring: its perimeter in bins of about COV_BIN_M."""
    perimeter = float(np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1).sum())
    n = max(1, int(round(perimeter / COV_BIN_M)))
    return n, perimeter / n


def wall_band_points(points: np.ndarray, outline: np.ndarray) -> np.ndarray:
    """(n, 3) model points that may be the building's walls: the FOOTPRINT_WALL_Z_M band within wall_points' ring about
    the outline, at most COV_MAX_POINTS of them (a fixed subsample)."""
    pts = np.asarray(points, float).reshape(-1, 3)
    centroid = polygon_centroid(outline)
    radius = float(np.max(np.linalg.norm(outline - centroid, axis=1)))
    near = np.linalg.norm(pts[:, :2] - centroid, axis=1) <= radius + max(WALL_RING_MIN_M, WALL_RING_FRACTION * radius)
    walls = pts[near & (pts[:, 2] >= FOOTPRINT_WALL_Z_M[0]) & (pts[:, 2] <= FOOTPRINT_WALL_Z_M[1])]
    if len(walls) > COV_MAX_POINTS:
        walls = walls[np.sort(np.random.default_rng(0).choice(len(walls), COV_MAX_POINTS, replace=False))]
    return walls


def coverage_bins(walls: np.ndarray, outline: np.ndarray, sim3: dict[str, Any] | None) -> dict[str, Any]:
    """One model's coverage of the outline bins (outline_bins): "claimed", the bins its published wall points land on
    (COV_NEAR_M, COV_MIN_POINTS); with its similarity to the reference (p_ref = scale R p + t), "covered": bin -> ((local),
    (geographic)) counts of the moved points landing there whose error is within each of COV_TOLERANCES_M, and
    "landed": bin -> how many moved points land there. Local error: a moved point's distance to the outline (its shape
    once the georegistration error is removed); geographic: the published point's distance to that outline point, at
    the moved point's height. Counts, not a median: points seen behind a colonnade or in a recess do not outvote the
    ones on the wall (so a denser cloud supports more bins; majority_bins is the share-based companion)."""
    n_bins, bin_m = outline_bins(outline)
    out: dict[str, Any] = {"claimed": [], "covered": {}, "landed": {}, "checked": sim3 is not None}

    def land(xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        d, closest, s = outline_arc(xy, outline)
        return d, closest, np.minimum((s / bin_m).astype(int), n_bins - 1)

    if not len(walls):
        return out
    d, _, b = land(walls[:, :2])
    ids, counts = np.unique(b[d <= COV_NEAR_M], return_counts=True)
    out["claimed"] = [int(k) for k in ids[counts >= COV_MIN_POINTS]]
    if sim3 is None:
        return out
    q = float(sim3["scale"]) * walls @ np.asarray(sim3["rotation"], float).T + np.asarray(sim3["translation_m"], float)
    dq, cq, bq = land(q[:, :2])
    geo = np.sqrt(((walls[:, :2] - cq) ** 2).sum(axis=1) + (walls[:, 2] - q[:, 2]) ** 2)
    on = dq <= COV_NEAR_M
    for k in np.unique(bq[on]):
        sel = on & (bq == k)
        if sel.sum() >= COV_MIN_POINTS:
            out["covered"][int(k)] = tuple(tuple(int((err[sel] <= t).sum()) for t in COV_TOLERANCES_M) for err in (dq, geo))
            out["landed"][int(k)] = int(sel.sum())
    return out


def majority_bins(cov: dict[str, Any]) -> set[int]:
    """The covered bins of one model (coverage_bins) where its points within 1.5 m in the building-local frame are at
    least COV_MIN_POINTS and at least COV_MAJORITY of the moved points landing there."""
    return {k for k, e in cov["covered"].items() if e[0][0] >= max(COV_MIN_POINTS, COV_MAJORITY * cov["landed"][k])}


def unsupported_bins(cov: dict[str, Any], frame: int, tol: int) -> set[int]:
    """The covered bins of one model (coverage_bins) it does not support in frame (0 local, 1 geographic) at
    COV_TOLERANCES_M[tol]."""
    return {k for k, e in cov["covered"].items() if e[frame][tol] < COV_MIN_POINTS}


def wrong_face_cameras(shown: dict[str, dict[str, Any]], photo_key: dict[str, str], reference: dict[str, Any]) -> int:
    """Shown poses (photo id -> camera entry, crowd3d.api.camera_estimates) of photos with a verified reference pose that
    look WRONG_FACE_HEADING_DEG or more off it, or stand WRONG_FACE_BEARING_DEG or more round the building from it."""
    refs = reference_poses(reference)
    wrong = 0
    for pid, cam in shown.items():
        ref = refs.get(photo_key.get(pid))
        if ref is None or cam.get("enu") is None or cam.get("heading_deg") is None:
            continue
        head = abs(angle_diff_deg(cam["heading_deg"], ref["heading_deg"]))
        bearing = abs(angle_diff_deg(heading_from_en(cam["enu"][0], cam["enu"][1]), heading_from_en(*ref["xy"])))
        wrong += head >= WRONG_FACE_HEADING_DEG or bearing >= WRONG_FACE_BEARING_DEG
    return int(wrong)


def calib_z(component: str, error: float | list[float] | None, sigma_model: float | None,
            sigma_ref: float) -> float | None:
    """|error| / sqrt(sigma_model^2 + sigma_ref^2); None without the model's sigma (or a zero one). For the two-axis
    components (CALIB_2D: error = (east, north), or the tilts about them, one sigma per axis) the normal deviate with
    the same tail probability as that 2-D radius (chi-square, 2 degrees of freedom), so that 1, 2 and 3 cover 0.683,
    0.954 and 0.997 for every component."""
    if error is None or sigma_model is None:
        return None
    sd = math.hypot(float(sigma_model), float(sigma_ref))
    if sd <= 0:
        return None
    if component not in CALIB_2D:
        return round(abs(float(error)) / sd, 3)
    d = math.hypot(*(float(e) for e in error)) / sd
    tail = math.exp(-0.5 * d * d)
    return round(abs(NormalDist().inv_cdf(tail / 2.0)), 3) if tail > 0 else round(d, 3)


def model_calibration(geo: dict[str, Any], sim3: dict[str, Any], anchor: dict[str, Any] | None,
                      meta: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    """One checked model's calibration (geo_check with its similarity to the reference, p_ref = s Q p + t): per
    component the signed error (model minus reference: yaw and log-scale from the similarity, the tilt about east
    and north from Q's third column, east/north at the anchor's pivot), the model's sigma ("core": the anchor's own,
    before the floors), the reference's and |z| (calib_z). The sigma is the GeoAnchor's under the current floors
    (GeoAnchor.recalibrated); a model JSON from before anchors gives its yaw_sigma_deg and scale_sigma and no xy or
    tilt."""
    q = np.asarray(sim3["rotation"], float)
    sigma_ref = reference.get("sigma_ref") or REFERENCE_SIGMA
    truth = "truth" in (reference.get("components") or {})
    walls = [c.get("median_wall_distance_m") or 0.0 for c in (reference.get("components") or {}).values()
             if c.get("verified")]
    refs = {"yaw": sigma_ref["yaw_deg"], "scale": sigma_ref["scale"],
            "xy": 0.0 if truth else max([REFERENCE_SIGMA_XY_MIN_M, *walls]),
            "tilt": 0.0 if truth else REFERENCE_SIGMA_TILT_DEG}
    errors: dict[str, Any] = {"yaw": -float(geo["yaw_deg"]), "scale": -math.log(float(geo["scale"])),
                              "tilt": [math.degrees(math.atan2(q[0, 2], q[2, 2])),
                                       math.degrees(math.atan2(q[1, 2], q[2, 2]))]}
    if anchor is None:
        sigmas, cores, out = {"yaw": meta.get("yaw_sigma_deg"), "scale": meta.get("scale_sigma")}, {}, {
            "sigma_from": "legacy", "calibration_id": None}
    else:
        a = GeoAnchor.from_meta(anchor).recalibrated()
        sg, core = a.sigma, a.raw.get("sigma_core") or {}
        sigmas = {"yaw": sg["yaw_deg"], "scale": sg["log_scale"], "xy": sg["east_m"], "tilt": sg["tilt_deg"]}
        cores = {"yaw": core.get("yaw_deg"), "scale": core.get("log_scale"), "xy": core.get("xy_m"),
                 "tilt": core.get("tilt_deg")}
        pivot = np.asarray(a.pivot_enu, float)
        errors["xy"] = (pivot - (float(sim3["scale"]) * q @ pivot + np.asarray(sim3["translation_m"], float)))[:2]
        out = {"sigma_from": "anchor", "calibration_id": a.calibration.get("id"), "level": a.level}
    calib = {}
    for c in CALIB_COMPONENTS:
        z = calib_z(c, errors.get(c), sigmas.get(c), refs[c])
        if z is not None:          # (no sigma, or none on either side: the fake reconstructor's 0 against truth)
            e = errors[c]
            calib[c] = {"e": [round(float(v), 4) for v in e] if c in CALIB_2D else round(e, 5), "core": cores.get(c),
                        "sigma": sigmas[c], "ref": refs[c], "z": z}
    return {**out, "calib": calib}


def model_key(cameras: dict[str, dict[str, Any]], photo_key: dict[str, str]) -> str:
    """A model's identity across repeats of a run (photo ids differ, test-set keys do not): every camera's place."""
    import hashlib

    cams = sorted((photo_key.get(pid, pid), c.get("enu")) for pid, c in cameras.items())
    return hashlib.sha256(json.dumps(cams).encode("utf-8")).hexdigest()[:16]


def geo_scores(models: list, data_dir: Path, outline: np.ndarray | None, reference: dict[str, Any] | None,
               photo_key: dict[str, str], anchors: dict[str, dict[str, Any] | None] | None = None) -> dict[str, Any]:
    """The GEO_METRICS of a finished building run from its store's ModelRecords (PLYs and kept work dirs under data_dir)
    and the per-model records behind them: geo_check of every published model that is georegistered or shown at the
    end (the metrics take the georegistered ones), coverage_bins of the shown ones, the calibration (model_calibration)
    of the checked ones, also as "calibration" rows (benchmark.py calibration). Without a reference only cov_claimed_m
    and cov_unverified_m are measured; without an outline no cov_*. anchors: model id -> a replayed GeoAnchor
    (regeo_models) instead of the model JSON's; a model without one is left out of the calibration."""
    from crowd3d.api import camera_estimates

    latest_ids = {m.id for m in latest_published(models).values()}
    n_bins, bin_m = outline_bins(outline) if outline is not None else (0, 0.0)
    records, cover = [], []
    for m in sorted(models, key=lambda m: (m.created_at, m.id)):
        meta = m.meta
        latest, georeg = m.id in latest_ids, bool(meta.get("georegistered"))
        if not meta.get("published", True) or not (latest or georeg):
            continue
        rec: dict[str, Any] = {"model_id": m.id, "kind": m.kind, "sector": SECTOR_NAMES.get(m.sector),
                               "latest_published": latest, "georegistered": georeg}
        ply = data_dir / m.ply_path
        walls = wall_band_points(read_ply_points(ply), outline) if outline is not None and ply.is_file() else np.zeros((0, 3))
        sim3 = None
        if reference is not None:
            cams = meta.get("cameras") or {}
            geo = geo_check(cams, model_enu_rotations(data_dir / "work" / m.job_id, cams), photo_key, reference,
                            walls if len(walls) else None)
            sim3 = geo.pop("sim3", None)
            if geo.get("yaw_deg") is not None:
                geo.update(yaw_sigma_deg=meta.get("yaw_sigma_deg"), scale_sigma=meta.get("scale_sigma"))
                if anchors is not None and anchors.get(m.id) is None:
                    geo.update(sigma_from="excluded (no faithful replay)", calib={})
                else:
                    geo.update(model_calibration(geo, sim3, anchors[m.id] if anchors is not None
                                                 else meta.get("anchor"), meta, reference))
                geo.update({f"z_{c}": (geo["calib"].get(c) or {}).get("z") for c in CALIB_COMPONENTS})
                rec["key"] = model_key(cams, photo_key)
            rec["geo"] = geo
        if outline is not None and latest:
            cov = coverage_bins(walls, outline, sim3)
            cover.append(cov)
            rec["coverage"] = {"claimed_m": round(len(cov["claimed"]) * bin_m, 1),
                               "covered_m": round(len(cov["covered"]) * bin_m, 1),
                               **{f"{f}_{t:g}_m": round(sum(e[i][j] >= COV_MIN_POINTS for e in cov["covered"].values())
                                                        * bin_m, 1)
                                  for i, f in enumerate(("local", "geo")) for j, t in enumerate(COV_TOLERANCES_M)},
                               "local_majority_1.5_m": round(len(majority_bins(cov)) * bin_m, 1),
                               "local_wrong_1.5_m": round(len(unsupported_bins(cov, 0, 0)) * bin_m, 1),
                               "geo_wrong_5_m": round(len(unsupported_bins(cov, 1, 1)) * bin_m, 1)}
        records.append(rec)
    out: dict[str, Any] = {name: None for name in GEO_METRICS}
    if reference is not None:
        checked = [r for r in records if r["georegistered"] and "camera_median_m" in r["geo"]]
        oriented = [r["geo"] for r in checked if r["geo"].get("yaw_deg") is not None]
        out.update(geo_models_checked=len(checked),
                 geo_camera_error_median_m=_median(r["geo"]["camera_median_m"] for r in checked),
                 geo_yaw_error_median_deg=_median(abs(g["yaw_deg"]) for g in oriented),
                 geo_scale_error_median_pct=_median(100.0 * abs(g["scale"] - 1.0) for g in oriented),
                 geo_tilt_error_median_deg=_median(g["tilt_deg"] for g in oriented),
                 geo_wall_error_p90_median_m=_median(g.get("walls_p90_m") for g in oriented),
                 wrong_face_models_published=sum(abs(g["yaw_deg"]) >= WRONG_FACE_HEADING_DEG for g in oriented),
                 wrong_face_cameras_shown=wrong_face_cameras(camera_estimates(models), photo_key, reference),
                 calib_sigma_missing=sum(g["sigma_from"] == "legacy" and None in (g["yaw_sigma_deg"], g["scale_sigma"])
                                         for g in oriented))
        # xy and tilt need an anchor: None (not 0) for a run whose models all predate anchors
        anchored = anchors is not None or any(g["sigma_from"] == "anchor" for g in oriented)
        for c in CALIB_COMPONENTS if anchored else ("yaw", "scale"):
            zs = [g[f"z_{c}"] for g in oriented if g[f"z_{c}"] is not None]
            within = sum(z <= 2.0 for z in zs)
            out.update({f"calib_{c}_n": len(zs), f"calib_{c}_within2": within,
                      f"calib_{c}_cover2": round(within / len(zs), 4) if zs else None,
                      f"calib_{c}_beyond3": sum(z > 3.0 for z in zs)})
    coverage = None
    if outline is not None:
        def metres(bins: set) -> float:
            return round(len(bins) * bin_m, 1)

        claimed = {k for c in cover for k in c["claimed"]}
        out.update(cov_claimed_m=metres(claimed),
                 cov_unverified_m=metres(claimed - {k for c in cover if c["checked"] for k in c["claimed"]}))
        if reference is not None:
            def supported(frame: int, tol: int) -> set:   # some model has COV_MIN_POINTS points there within the tolerance
                return {k for c in cover for k, e in c["covered"].items() if e[frame][tol] >= COV_MIN_POINTS}

            covered = {k for c in cover for k in c["covered"]}
            local15, local5, geo15, geo5 = supported(0, 0), supported(0, 1), supported(1, 0), supported(1, 1)
            out.update(cov_local_supported_m_1p5=metres(local15), cov_local_supported_m_5=metres(local5),
                     cov_geo_supported_m_1p5=metres(geo15), cov_geo_supported_m_5=metres(geo5),
                     cov_local_wrong_m_1p5=metres(covered - local15), cov_geo_wrong_m_5=metres(covered - geo5),
                     cov_local_majority_m_1p5=metres({k for c in cover for k in majority_bins(c)}),
                     # every shown model's wrong walls, also where another shown model supports the bin
                     cov_local_wrong_sum_m_1p5=round(sum(len(unsupported_bins(c, 0, 0)) for c in cover) * bin_m, 1),
                     cov_geo_wrong_sum_m_5=round(sum(len(unsupported_bins(c, 1, 1)) for c in cover) * bin_m, 1))
        coverage = {"bins": n_bins, "bin_m": round(bin_m, 4), "near_m": COV_NEAR_M, "min_points": COV_MIN_POINTS,
                    "majority": COV_MAJORITY, "tolerances_m": list(COV_TOLERANCES_M)}
    calibration = [{"model_id": r["model_id"], "kind": r["kind"], "sector": r["sector"], "key": r["key"],
                    **{k: r["geo"].get(k) for k in ("sigma_from", "calibration_id")}, "comps": r["geo"]["calib"]}
                   for r in records if r["georegistered"] and (r.get("geo") or {}).get("calib")]
    return {"metrics": out, "models": records, "coverage": coverage, "calibration": calibration}


def load_run_store(data_dir: Path) -> tuple[Any, list, list, list]:
    """(landmark, every photo, every model, every job) of a finished run's data dir, read from a temporary copy of its
    SQLite files: the run dir is never written to (Store migrates a database when it opens it)."""
    import tempfile

    from crowd3d.store import Store

    with tempfile.TemporaryDirectory(prefix="crowd3d-rescore-") as tmp:
        for suffix in ("", "-wal", "-shm"):
            f = data_dir / f"crowd3d.sqlite3{suffix}"
            if f.is_file():
                shutil.copyfile(f, Path(tmp) / f.name)
        if not (Path(tmp) / "crowd3d.sqlite3").is_file():
            raise FileNotFoundError(f"no crowd3d.sqlite3 in {data_dir}")
        store = Store(Settings(data_dir=Path(tmp)))
        try:
            landmarks = store.list_landmarks()
            if len(landmarks) != 1:
                raise ValueError(f"{data_dir}: expected one landmark, found {len(landmarks)}")
            lm = landmarks[0]
            return lm, store.list_photos(lm.id), store.list_models(lm.id), store.list_jobs(lm.id)
        finally:
            store.close()


def planned_upload_keys(b: CrowdBuilding) -> list[str]:
    """Keys of the photos run_crowd uploads, in test-set order: present, and with EXIF GPS unless b.upload_all."""
    keys = []
    for p in b.photos:
        if not p.path.is_file():
            continue
        if b.upload_all:
            keys.append(p.key)
            continue
        try:
            meta = read_photo_meta(p.path.read_bytes())
        except UnreadableImage:
            continue
        if meta.has_gps:
            keys.append(p.key)
    return keys


def run_data_dir(run_dir: Path, summary: dict[str, Any]) -> Path:
    """A recorded building run's data dir: <run_dir>/data (the run may have moved), else where the run wrote it."""
    return run_dir / "data" if (run_dir / "data").is_dir() else Path(summary["data_dir"])


def rescore_poses(run_dir: Path, reference: dict[str, Any] | None,
                  planned_keys: list[str] | None = None, style: dict[str, Any] | None = None) -> dict[str, Any]:
    """The photo_pose_* metrics of a recorded building run (summary.json + data/), without running anything again:
    the published models, accepted snaps and photo rows the run left, against the given reference. planned_keys: the
    test-set photos the protocol would upload (planned_upload_keys); the ones a time cap never uploaded fail in
    the *_testset metrics. style: a new stylization of the run (restylize_run) instead of the recorded one."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    lm, photos, models, _ = load_run_store(run_data_dir(run_dir, summary))
    rows = summary["photos"]
    height = (((summary.get("effective_params") or {}).get("settings") or {}).get("camera_height_m")
              or Settings().camera_height_m)
    out = run_photo_poses(rows, photos, models, lm.lat, lm.lon, style=style or summary.get("style"),
                          reference=reference, camera_height_m=float(height),
                          heading_refs={r["photo_id"]: r.get("heading_ref") for r in rows if r.get("photo_id")},
                          planned_keys=planned_keys)
    out["truncated"] = bool(summary.get("truncated"))
    out["photos_uploaded"] = summary["funnel"]["uploaded"]
    out["funnel"] = {k: summary["funnel"].get(k) for k in ("listed", "no_gps_in_file", "uploaded")}
    return out


def regeo_models(models: list, photos: list, lm: Any, data_dir: Path,
                 summary: dict[str, Any]) -> tuple[dict[str, dict[str, Any] | None], dict[str, Any]]:
    """Every published model's georegistration run again on its job's kept SfM model (work/<job>/sparse/0) with the
    product's code (reconstruct.georegister_inputs, georegister_model), the run's parameters (summary.json) and the
    photos' GPS and compass from its store, but the current GeoAnchor calibration (Params anchor_*). (model id ->
    the replayed anchor, or None when it cannot be replayed or does not put every camera within REGEO_MAX_DIFF_M of
    the model JSON's ENU position; the report.)"""
    import pycolmap

    eff = summary.get("effective_params") or {}
    run_params = params.with_overrides({k: v for k, v in (eff.get("params") or {}).items()
                                        if k in params.PARAM_NAMES and not k.startswith("anchor_")},
                                       base=params.current())
    height = float((eff.get("settings") or {}).get("camera_height_m") or Settings().camera_height_m)
    enu = LocalENU(lm.lat, lm.lon)
    by_id = {p.id: p for p in photos}
    anchors: dict[str, dict[str, Any] | None] = {}
    report: dict[str, Any] = {"published": 0, "replayed": 0, "faithful": 0, "max_diff_m": None, "left_out": []}
    for m in sorted(models, key=lambda m: (m.created_at, m.id)):
        if not m.meta.get("published", True):
            continue
        report["published"] += 1
        anchors[m.id] = None
        sparse = data_dir / "work" / m.job_id / "sparse" / "0"
        if not any((sparse / f).is_file() for f in ("images.bin", "images.txt")):
            report["left_out"].append({"model_id": m.id, "reason": "no kept sparse/0"})
            continue
        rec = pycolmap.Reconstruction(str(sparse))
        by_name = {}
        for im in rec.images.values():
            p = by_id.get(photo_id_from_image_name(im.name))
            if p is not None and p.lat is not None and p.lon is not None:
                by_name[im.name] = ReconPhoto(p.id, Path(), p.lat, p.lon, p.heading_deg,
                                              gps_accuracy_m=p.gps_accuracy_m, focal_35mm=p.focal_35mm)
        gps = {name: np.array(enu.to_enu(p.lat, p.lon)[:2]) for name, p in by_name.items()}
        with params.using(run_params):
            inputs = georegister_inputs(rec, by_name, gps)
            replay = georegister_model(inputs, height) if inputs is not None else None
        if replay is None:
            report["left_out"].append({"model_id": m.id, "reason": "fewer than 3 posed photos of the job"})
            continue
        geo, cams = replay
        stored = m.meta.get("cameras") or {}
        report["replayed"] += 1
        diff = max((float(np.abs(np.subtract(cams[k]["enu"], stored[k]["enu"])).max()) for k in cams if k in stored),
                   default=math.inf)
        report["max_diff_m"] = round(max(report["max_diff_m"] or 0.0, diff), 4)
        if set(cams) != set(stored) or diff > REGEO_MAX_DIFF_M:
            report["left_out"].append({"model_id": m.id, "reason": f"replayed cameras {len(cams)} / {len(stored)},"
                                       f" largest ENU difference {diff:.3f} m"})
            continue
        report["faithful"] += 1
        anchors[m.id] = geo.anchor.to_meta()
    return anchors, report


def rescore_geo(run_dir: Path, b: CrowdBuilding, reference: dict[str, Any] | None,
                regeo: bool = False) -> dict[str, Any]:
    """The GEO_METRICS, VERSION_METRICS and REQUEST_METRICS of a recorded building run (summary.json, its store, PLYs
    and kept work dirs), without running anything again: what run_crowd measures after its uploads, against the given
    reference and b's outline. The request-log metrics an older run did not record stay None. regeo: the calibration
    takes the georegistrations replayed on the kept SfM models (regeo_models), not the model JSONs' anchors."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    data_dir = run_data_dir(run_dir, summary)
    lm, photos, models, jobs = load_run_store(data_dir)
    outline = load_footprint(b.outline_path, LocalENU(lm.lat, lm.lon)) if b.outline_path else None
    anchors, replay = regeo_models(models, photos, lm, data_dir, summary) if regeo else (None, None)
    geo = geo_scores(models, data_dir, outline, reference,
                     {r["photo_id"]: r["key"] for r in summary["photos"] if r.get("photo_id")}, anchors=anchors)
    metrics = {**geo["metrics"], **version_metrics(models, jobs, photos),
               **request_metrics(summary["requests"], summary["photos"])}
    return {"metrics": metrics, "geo": {**{k: geo[k] for k in ("models", "coverage", "calibration")},
                                        **({"regeo": replay} if regeo else {})}}


def restylize_run(run_dir: Path, b: CrowdBuilding, reference_result: tuple[dict | None, dict | None],
                  out_dir: Path) -> dict[str, Any]:
    """stylize_published again, with the current code, on a recorded building run's kept work dirs (the run dir is
    not written to; the stylized models go to out_dir): the style record run_crowd would make now, given the
    building's (reference, report)."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    data_dir, recorded = run_data_dir(run_dir, summary), Path(summary["data_dir"])
    models = [dict(m, ply_path=str(data_dir / Path(m["ply_path"]).relative_to(recorded))) for m in summary["models"]]
    photo_key = {r["photo_id"]: r["key"] for r in summary["photos"] if r.get("photo_id")}
    return stylize_published(models, b, work_dir=data_dir / "work", out_dir=out_dir, photo_key=photo_key, metas={},
                             enu=LocalENU(b.lat, b.lon), deadline=time.monotonic() + STYLE_TIME_CAP_S,
                             reference_result=reference_result)


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
    (stylize_published, the style_* metrics) with a budget of its own (STYLE_TIME_CAP_S from its start; the
    upload cap must not eat into it: a loaded machine once lost a building's whole style record to it); models
    it does not reach count in style_incomplete.
    """
    from fastapi.testclient import TestClient

    from crowd3d.api import camera_estimates, create_app, photo_info
    from crowd3d.coverage import compute_coverage
    from crowd3d.ingest import ingest_upload
    from crowd3d.records import PHOTO_ACCEPTED

    t0, cpu0 = time.monotonic(), cpu_seconds()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = out_dir / "data"
    cached_reference = bool(b.reference_cache) and Path(b.reference_cache["path"]).is_file()
    settings = params.settings_with(
        Settings(data_dir=data_dir, reconstructor=reconstructor, inline_jobs=True,
                 keep_work_dirs=b.gt is not None or cached_reference or stylize),
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
    outline = load_footprint(b.outline_path, enu) if b.outline_path else None
    landmark_id = "".join(ch if ch.isascii() and ch.isalnum() else "-" for ch in b.slug.lower()).strip("-")[:64]
    landmark_id = landmark_id or "landmark"

    rows: dict[str, dict[str, Any]] = {}
    photo_key: dict[str, str] = {}
    request_log: list[dict[str, Any]] = []
    job_request: dict[str, int] = {}
    request_start: list[float] = []
    posed_at: dict[str, float] = {}      # photo id -> when the landmark page first showed a pose for it
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
            request_start.append(t)
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
            jobs, cost = [], {}
            if accepted:      # the jobs' CPU time and peak memory (VmHWM reset first), and the memory before them
                rss0, reset, cpu = _status_mb("VmRSS"), reset_peak_rss(), cpu_seconds()
                jobs = runner.schedule(landmark_id)
                cost = {"cpu_s": round(cpu_seconds() - cpu, 1), "rss_before_mb": _round(rss0, 1),
                        "peak_rss_mb": round(peak_rss_mb(), 1), "peak_reset": reset}
            t_done = time.monotonic()
            job_request.update((j.id, k) for j in jobs)
            request_log.append({
                "request": k, "label": req.label, "photos": len(req.items), "accepted": accepted,
                "ingest_s": round(t_ingest, 1), "jobs_s": round(t_done - t - t_ingest, 1), **cost,
                "jobs": [{"kind": j.kind, "sector": SECTOR_NAMES.get(j.sector), "photos": len(j.photo_ids),
                          "status": j.status, "error": j.error} for j in jobs],
            })
            if jobs:          # the poses the landmark page shows now (each photo's time to its first one)
                for pid in camera_estimates(store.published_models(landmark_id)):
                    posed_at.setdefault(pid, t_done)
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
        all_photos = store.list_photos(landmark_id)
        coverage = compute_coverage(accepted_photos, settings)
        # What the landmark page shows per photo (the detail endpoint lists only the newest 200 photos).
        shown_cams = camera_estimates(store.published_models(landmark_id))
        shown_by_id = {p.id: photo_info(p, shown_cams.get(p.id)) for p in accepted_photos}

    for row in rows.values():
        if row["photo_id"] in posed_at:
            row["time_to_pose_s"] = round(posed_at[row["photo_id"]] - request_start[row["request"]], 1)
    # the reference (an evaluation input, cached across runs: its time is left out of the run's wall and CPU time)
    t_ref, cpu_ref = time.monotonic(), cpu_seconds()
    metas = {p.key: p.meta for p in present if p.meta is not None}
    reference, ref_report, gt_poses = building_reference(b, metas, enu)
    ref_s, ref_cpu = time.monotonic() - t_ref, cpu_seconds() - cpu_ref
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
    wall = time.monotonic() - t0 - ref_s
    metrics = crowd_metrics(funnel, coverage, jobs_all, models, latest_ids, shown, wall, gt_poses is not None)
    metrics.update(cpu_time_s=round(cpu_seconds() - cpu0 - ref_cpu, 1), disk_run_mb=dir_size_mb(data_dir, ("work",)),
                   **request_metrics(request_log, list(rows.values())))
    # the scores of what the run left (an evaluation, as the reference: outside the wall and CPU time)
    geo = geo_scores(model_records, data_dir, outline, reference, photo_key)
    metrics.update(geo["metrics"])
    metrics.update(version_metrics(model_records, jobs_all, all_photos))
    style = None
    if stylize:
        style = stylize_published(
            models, b, work_dir=settings.work_dir, out_dir=out_dir / "style", photo_key=photo_key,
            metas=metas, enu=enu, deadline=time.monotonic() + STYLE_TIME_CAP_S,
            reference_result=(reference, ref_report))
        metrics.update(style["metrics"])
    poses = run_photo_poses(list(rows.values()), all_photos, model_records, b.lat, b.lon, style=style,
                            reference=reference, camera_height_m=settings.camera_height_m,
                            heading_refs={r["photo_id"]: r["heading_ref"] for r in rows.values() if r["photo_id"]},
                            planned_keys=[p.key for p in uploads])
    poses["reference"] = ref_report
    metrics.update(poses["metrics"])
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
        "geo": {k: geo[k] for k in ("models", "coverage", "calibration")},
        "style": style,
        "poses": {k: poses[k] for k in ("summary", "thresholds", "reference", "records")},
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
    ap.add_argument("--lat", type=float, default=None, help="landmark centre latitude (overrides the catalog)")
    ap.add_argument("--lon", type=float, default=None, help="landmark centre longitude (overrides the catalog)")
    ap.add_argument("--radius", type=float, default=None, help="check-in radius in metres (default: catalog, else 150)")
    ap.add_argument("--name", default=None, help="landmark name (default: catalog or folder name)")
    ap.add_argument("--out", type=Path, default=None, help="run dir (required)")
    ap.add_argument("--overwrite", action="store_true", help="delete existing data dirs first")
    ap.add_argument("--seed", type=int, default=None, help="default: the config's, else 0")
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
    return main_crowd(args, cfg)


def _json_default(o: Any) -> Any:
    """json.dumps default for the summaries (also scripts/stylize_model.py's): numpy values, paths."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


if __name__ == "__main__":
    raise SystemExit(main())
