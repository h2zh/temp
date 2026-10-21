"""Evaluate the Crowd3D pipeline on real photos of a famous building against a ground-truth SfM model.

    python scripts/eval_real_landmark.py --scene-dir <scene>/<component> \\
        --lat 43.773056 --lon 11.254722 --radius 150 --mode both --out <run dir>
    python scripts/eval_real_landmark.py --scene "Baptistry_(Florence)" --lat ... --lon ...   # download first

Data: MegaDepth-X (huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X, CVPR 2026) ships, per landmark,
real Wikimedia Commons photos plus a COLMAP model of all of them, used here as ground truth (GT).
SfM and depth data are CC BY 4.0; every photo has its own licence (license/license.parquet in that
repo). Never put the photos, the archives or anything derived from them (data dirs, point clouds,
screenshots) into this repository: --out and --data-dir must lie outside it or under its
gitignored data/ folder.

What a run does, per mode:
1. Loads the GT model and georeferences it into the landmark's ENU frame using only the photos with
   real EXIF GPS (+ real compass): GT-ENU. This frame is fitted to the same GPS the pipeline uses,
   so errors against it are partly circular; --footprint (e.g. an OSM outline) gives an independent
   check.
2. Builds the uploads. "real": only photos that really carry EXIF GPS, untouched. "augmented": all
   photos; missing GPS/heading tags are simulated from the GT-ENU pose plus seeded Gaussian noise and
   written into the EXIF of a copy (pixels untouched).
3. Runs the product code (ingest_upload, JobRunner with inline jobs, ColmapReconstructor, the HTTP
   API for the landmark and its detail) in two stages: first the most popular sector only (expect a
   facade model), then everything else. Photos are spread over --users pseudo users. Jobs are
   planned once per stage, as if each stage were one upload request: planning after every user's
   request would rebuild the same facade after each one (linear growth threshold).
4. Writes summary.json and summary.md into the run dir. The Crowd3D data dir of each mode can be
   browsed afterwards: python -m crowd3d.api --data-dir <run dir>/<mode>/data
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
import os
import shutil
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import piexif

if __package__ in (None, ""):
    # `python scripts/eval_real_landmark.py` from a checkout: make the crowd3d package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd3d.config import Settings  # noqa: E402
from crowd3d.coverage import sector_for_bearing, sector_label  # noqa: E402
from crowd3d.exif import PhotoMeta, build_exif_bytes, read_photo_meta  # noqa: E402
from crowd3d.geo import LocalENU, angle_diff_deg, heading_from_en, local_bearing_distance  # noqa: E402
from crowd3d.reconstruct import (  # noqa: E402
    COMPASS_SIGMA_DEG,
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
    s, r, t = umeyama(est_centers, gt_centers)
    err = np.linalg.norm(s * est_centers @ r.T + t - gt_centers, axis=1)
    keep = err <= max(3.0 * float(np.median(err)), 1.0)
    if 3 <= keep.sum() < n:
        s, r, t = umeyama(est_centers[keep], gt_centers[keep])
        err = np.linalg.norm(s * est_centers @ r.T + t - gt_centers, axis=1)
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
        sigma_compass = math.radians(COMPASS_SIGMA_DEG) / math.sqrt(compass[1])
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
            "wall_fit": fit_outline(walls, footprint_en, centroid - top_xy) if len(walls) >= 50 else None,
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
    settings = replace(settings, **overrides)
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
        dr.rectangle([G[0] - 2, G[1] - 2, G[0] + 2, G[1] + 2], outline=color)
    c = px(0, 0)
    dr.line([c[0] - 8, c[1], c[0] + 8, c[1]], fill=(0, 0, 0), width=2)
    dr.line([c[0], c[1] - 8, c[0], c[1] + 8], fill=(0, 0, 0), width=2)
    dr.text((8, 8), f"N up, {2 * half:.0f} m across; + = given centre; circle = {radius_m:.0f} m radius", fill=(0, 0, 0))
    img.save(path)


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
    src.add_argument("--scene-dir", type=Path, help="extracted MegaDepth-X <scene>/<component> (images/, sparse/)")
    src.add_argument("--scene", help="MegaDepth-X scene name, e.g. 'Baptistry_(Florence)': download + extract")
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="where --scene archives go")
    ap.add_argument("--lat", type=float, required=True, help="landmark centre latitude")
    ap.add_argument("--lon", type=float, required=True, help="landmark centre longitude")
    ap.add_argument("--radius", type=float, default=150.0, help="check-in radius in metres (default 150)")
    ap.add_argument("--name", default=None, help="landmark name (default: from the scene folder)")
    ap.add_argument("--landmark-id", default="real-landmark")
    ap.add_argument("--mode", choices=(*MODES, "both"), default="both")
    ap.add_argument("--out", type=Path, default=None, help="run dir for reports (default <cache-dir>/eval-runs/<scene>)")
    ap.add_argument("--data-dir", type=Path, default=None,
                    help="base for the Crowd3D data dirs, one sub-folder per mode (default <out>/<mode>/data)")
    ap.add_argument("--overwrite", action="store_true", help="delete existing data dirs of the selected modes first")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gps-noise-m", type=float, default=5.0, help="sigma per axis of simulated GPS (default 5)")
    ap.add_argument("--heading-noise-deg", type=float, default=15.0, help="sigma of simulated compass (default 15)")
    ap.add_argument("--max-images", type=int, default=None, help="cap on uploads (real-GPS photos are kept first)")
    ap.add_argument("--users", type=int, default=5, help="pseudo users the uploads are spread over")
    ap.add_argument("--max-side", type=int, default=None, help="Settings.processing_max_side_px override")
    ap.add_argument("--full-min-sectors", type=int, default=None,
                    help="Settings.full_min_sectors override (product default 6); reported as an override")
    ap.add_argument("--footprint", type=Path, default=None,
                    help="GeoJSON or Nominatim JSON with the building outline, for an independent GT check")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.scene:
        args.scene_dir = pick_component(download_scene(args.scene, args.cache_dir))
    args.scene_dir = pick_component(args.scene_dir)
    scene_name = args.scene or args.scene_dir.parent.name
    args.name = args.name or scene_name.replace("_", " ")
    out = args.out or args.cache_dir / "eval-runs" / scene_name
    check_outside_repo(out, "--out")
    modes = MODES if args.mode == "both" else (args.mode,)
    data_dirs = {m: (args.data_dir / m if args.data_dir else out / m / "data") for m in modes}
    for m, d in data_dirs.items():
        check_outside_repo(d, "data dir")
        if d.exists() and any(d.iterdir()):
            if not args.overwrite:
                raise SystemExit(f"data dir {d} is not empty; pass --overwrite to start over")
            shutil.rmtree(d)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    gt = load_ground_truth(args.scene_dir)
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
