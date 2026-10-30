"""Photos -> geo-registered 3D point cloud.

ColmapReconstructor runs incremental SfM with pycolmap, then geo-registers the
result with the standalone numpy helpers below. FakeReconstructor is meant for
fast tests of the job runner and the API.

Output frame for every reconstructor: ENU metres anchored at the landmark
centre (x = east, y = north, z = up), so models are north-aligned and metric.
"""

from __future__ import annotations

import logging
import math
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from . import params
from .anchor import GeoAnchor, calibration, floored_sigma, quat_xyzw
from .config import Settings
from .geo import LocalENU, angle_diff_deg, heading_from_en
from .landmark_map import FeatureCache, JobCache, MapBase, extraction_options
from .params import DEFAULTS as DEFAULT_PARAMS

log = logging.getLogger(__name__)

# What a reconstruction job leaves for the same uploads (its sparse model, PLY, model JSON and QA verdict) is
# version RECON_OUTPUT_VERSION. Bump it with any change that can alter that output: SfM options, model choice,
# georegistration, camera checks, export, the QA gate, job scheduling, ingest or a Params / Settings default.
# A change that only adds keys to the model JSON (every earlier key keeps its value) keeps the version; so does a
# change to the GeoAnchor's values alone, with a new anchor.ANCHOR_METHOD (or its floors, with a new
# Params.anchor_calibration_id), and so did turning on Params.feature_cache (the jobs' COLMAP databases are byte for
# byte the ones computed without it, docs/BENCHMARK.md 6.7). Params.map_continue changes nothing while it is off;
# turning it on by default bumps the version (a continued model is another model of the same uploads).
# tests/test_recon_golden.py (uploads through ingest and the job runner) fails until it is bumped; scripts/benchmark.py
# rescore refuses to rescore artefacts of another version. 1: the output of db06c3c.
RECON_OUTPUT_VERSION = 1
MIN_REGISTERED = 3
FULL_FRAME_DIAGONAL_MM = 43.27        # COLMAP's focal prior: f_px = FocalLengthIn35mmFilm / 43.27 * image diagonal
# Every algorithm tunable (GPS and compass sigmas, inlier thresholds, mapping retries, per-camera checks)
# is a params.Params field, read at call time through params.current(); the defaults are documented there.
GPS_INLIER_M = DEFAULT_PARAMS.gps_inlier_threshold_m(None)   # default threshold for a photo without accuracy


@dataclass
class ReconPhoto:
    photo_id: str
    image_path: Path             # processed copy (orientation applied, downscaled, EXIF focal kept)
    lat: float
    lon: float
    heading_deg: float | None
    gps_accuracy_m: float | None = None   # reported horizontal GPS error, used as sigma per axis
    focal_35mm: float | None = None       # EXIF FocalLengthIn35mmFilm of the original


@dataclass
class ReconstructionResult:
    ok: bool
    message: str
    ply_path: Path | None = None
    num_input: int = 0
    num_registered: int = 0
    num_points: int = 0
    mean_reprojection_error_px: float | None = None
    georegistered: bool = False
    alignment_rmse_m: float | None = None    # camera centres vs GPS after alignment
    heading_mae_deg: float | None = None     # estimated headings vs EXIF compass headings
    # photo_id -> {"enu": [x, y, z], "heading_deg": float, "flags": [...], diagnostics}; registered photos
    # only. A camera with flags disagrees with its own GPS/compass or is degenerate: not to be shown.
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)   # stage -> seconds, for diagnostics
    points_median_en: list[float] | None = None   # horizontal median of the exported points (ENU)
    yaw_source: str | None = None                 # what fixed north: Georegistration.yaw_source
    mapping_attempts: int = 0
    yaw_sigma_deg: float | None = None            # Georegistration.yaw_sigma_deg: the snap's yaw window (stylize)
    scale_sigma: float | None = None              # Georegistration.scale_sigma
    anchor: dict[str, Any] | None = None          # GeoAnchor.to_meta(): ENU from the local frame, per-component sigmas
    cache_stats: dict[str, int] | None = None     # landmark_map.CacheStats of a job that used the feature cache
    # a job handed a base model to continue (stage 1b): {"base_model_id", "accepted", "reason" (why not),
    # "base_registered", "base_kept" (its registered photos the continued model keeps), "registered"}; a rejected
    # continuation falls back to the mapping attempts from scratch
    continued: dict[str, Any] | None = None

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d["ply_path"] = str(self.ply_path) if self.ply_path else None
        d["frame"] = "ENU"
        for key in ("cache_stats", "continued"):  # only a job that used the cache (a base) records it
            if d[key] is None:
                del d[key]
        return d


class Reconstructor(Protocol):
    """name; reconstruct(). A reconstructor whose class sets feature_cache = True is handed the landmark's FeatureCache
    as reconstruct(..., cache=...) when Params.feature_cache is on (jobs.JobRunner); others are never passed one.
    Likewise one whose class sets map_continue = True is handed the saved map of the model its target shows as
    reconstruct(..., base=...) when Params.map_continue is on and the job may continue it."""

    name: str

    def reconstruct(
        self,
        photos: list[ReconPhoto],
        landmark_lat: float,
        landmark_lon: float,
        out_ply: Path,
        work_dir: Path,
        *,
        cache: FeatureCache | None = None,
        base: MapBase | None = None,
    ) -> ReconstructionResult: ...


def write_ply(path: Path, points: list[tuple[float, float, float]], colors: list[tuple[int, int, int]]) -> None:
    """ASCII PLY with per-vertex colour."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for (x, y, z), (r, g, b) in zip(points, colors):
            f.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


FAKE_RING_POINTS = 360


class FakeReconstructor:
    """Instant stand-in: trusts GPS and compass, emits a ring of points around the centre."""

    name = "fake"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, *, cache=None,
                    base=None) -> ReconstructionResult:
        if len(photos) < 2:
            return ReconstructionResult(ok=False, message="need at least 2 photos", num_input=len(photos))
        enu = LocalENU(landmark_lat, landmark_lon)
        cams: dict[str, dict[str, Any]] = {}
        points, colors = [], []
        for p in photos:
            e, n, _ = enu.to_enu(p.lat, p.lon)
            cams[p.photo_id] = {"enu": [e, n, self.settings.camera_height_m], "heading_deg": p.heading_deg,
                                "gps_residual_m": 0.0, "flags": []}
        for i in range(FAKE_RING_POINTS):
            a = 2 * math.pi * i / FAKE_RING_POINTS
            points.append((5 * math.sin(a), 5 * math.cos(a), 0.0))
            colors.append((200, 200, 200))
        write_ply(out_ply, points, colors)
        # no local frame of its own: the identity, with the sigmas it claims (none)
        anchor = GeoAnchor("fake", "full", 1.0, [0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0,
                           [0.0, 0.0, 0.0], dict.fromkeys(("yaw_deg", "log_scale", "east_m", "north_m", "up_m",
                                                            "tilt_deg"), 0.0),
                           dict.fromkeys(("yaw", "scale", "xy", "up", "gravity"), "fake"), {}, calibration())
        return ReconstructionResult(
            ok=True,
            message="fake reconstruction",
            ply_path=out_ply,
            num_input=len(photos),
            num_registered=len(photos),
            num_points=len(points),
            mean_reprojection_error_px=0.0,
            georegistered=True,
            alignment_rmse_m=0.0,
            heading_mae_deg=0.0,
            cameras=cams,
            points_median_en=[0.0, 0.0],
            yaw_source="gps",
            mapping_attempts=1,
            anchor=anchor.to_meta(),
        )


# ---------------------------------------------------------------- geo-registration (pure numpy)


def estimate_up(ups: np.ndarray, rights: np.ndarray, centers: np.ndarray | None = None) -> np.ndarray:
    """World "up" in the SfM frame (unit vector) from camera orientations and centres.

    The mean camera up vector leans back by the typical upward pitch (people
    tilt phones up at buildings), about 10 degrees when all photos show one
    facade and 30 or more right below a tall one. So take the direction most
    perpendicular to the camera right vectors (phones are held without roll),
    then also to the spread of the centres of the cameras near the common
    height (photographers stand at about the same height; a photo from a
    tower or terrace says nothing about the ground plane), with a light pull
    toward the mean up that fixes the sign. Only when that direction is
    unobservable (two smallest eigenvalues within Params.up_min_eigen_gap, e.g. a
    straight line of cameras all facing the same way) is the mean up returned.
    """
    p = params.current()
    ups = np.asarray(ups, dtype=float)
    rights = np.asarray(rights, dtype=float)
    mean_up = ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    m_rights = rights.T @ rights / len(rights)

    def solve(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, v = np.linalg.eigh(m - p.up_prior_weight * np.outer(mean_up, mean_up))
        return (v[:, 0] if v[:, 0] @ mean_up >= 0 else -v[:, 0]), w

    up, w = solve(m_rights)
    if centers is not None and len(centers) >= 2:
        c = np.asarray(centers, dtype=float)
        h = c @ up
        flat = c - np.outer(h, up)
        d = math.sqrt(float(((flat - flat.mean(axis=0)) ** 2).sum(axis=1).mean()))
        level = np.abs(h - np.median(h)) <= p.up_common_height_fraction * d
        if level.sum() >= 2:
            ck = c[level] - c[level].mean(axis=0)
            spread = ck.T @ ck
            if np.trace(spread) > 0:
                up, w = solve(m_rights + spread / np.trace(spread))
    if w[1] - w[0] < p.up_min_eigen_gap:
        return mean_up
    return up


GRAVITY_BOOT_N = 100     # the GeoAnchor's tilt sigma: estimate_up on this many resamples of the cameras (seed 0)


def gravity_bootstrap(ups: np.ndarray, rights: np.ndarray, centers: np.ndarray, n: int = GRAVITY_BOOT_N,
                      seed: int = 0) -> tuple[float, np.ndarray]:
    """Bootstrap spread of estimate_up: (the larger standard deviation, over the resampled camera sets, of the up's
    tangent angles about the frame's x and y axes, in degrees; the whole set's up as those two angles). In a frame
    whose z is the cameras' up this is the tilt uncertainty per horizontal axis. D25 calls it through
    correct_measure.estimate_up_boot with its own n = 200."""
    ups, rights, centers = (np.asarray(a, dtype=float) for a in (ups, rights, centers))
    rng = np.random.default_rng(seed)
    tang = []
    for _ in range(n):
        i = rng.integers(0, len(ups), len(ups))
        u = estimate_up(ups[i], rights[i], centers[i])
        tang.append(np.degrees(np.arctan2(u[:2], u[2])))
    u0 = estimate_up(ups, rights, centers)
    return float(np.array(tang).std(0).max()), np.degrees(np.arctan2(u0[:2], u0[2]))


def rotation_to_z(v: np.ndarray) -> np.ndarray:
    """Proper 3x3 rotation taking direction v onto +z by the shortest arc."""
    v = np.asarray(v, dtype=float) / np.linalg.norm(v)
    axis = np.cross(v, [0.0, 0.0, 1.0])
    s, c = np.linalg.norm(axis), v[2]
    if s < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = axis / s
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + s * kx + (1 - c) * kx @ kx


def rotation_z(yaw_rad: float) -> np.ndarray:
    """Counter-clockwise rotation about +z (east towards north)."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _complex(xy: np.ndarray) -> np.ndarray:
    xy = np.asarray(xy, dtype=float)
    return xy[:, 0] + 1j * xy[:, 1]


def fit_similarity_2d(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None) -> tuple[float, float, np.ndarray]:
    """Least-squares (Umeyama) similarity dst ~ scale * R(yaw) @ src + t without reflection.

    Returns (scale, yaw_rad, t) with yaw counter-clockwise. In complex form the
    model is w = a z + b, whose (optionally weighted) least-squares a is closed-form.
    """
    z, w = _complex(src), _complex(dst)
    om = np.ones(len(z)) if weights is None else np.asarray(weights, dtype=float)
    zm, wm = complex(om @ z) / om.sum(), complex(om @ w) / om.sum()
    zc, wc = z - zm, w - wm
    denom = float(om @ (np.abs(zc) ** 2))
    a = complex((om * np.conj(zc)) @ wc) / denom if denom > 0 else 1.0 + 0j
    b = wm - a * zm
    return abs(a), math.atan2(a.imag, a.real), np.array([b.real, b.imag])


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


def robust_similarity_2d(
    src: np.ndarray,
    dst: np.ndarray,
    inlier_threshold: float | np.ndarray,
    min_pair_distance: float = 0.0,
    max_pairs: int = 2000,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """RANSAC over point pairs, then least squares on the inliers.

    Each pair whose dst points are at least min_pair_distance apart fixes a
    similarity; the one with most inliers (residual < inlier_threshold, a scalar
    or one value per point; ties broken by truncated squared error) wins and is
    refined twice by fit_similarity_2d (with the given weights) on its inliers.
    Pairs are enumerated exhaustively up to max_pairs, else sampled with the
    given seed. Returns (scale, yaw_rad, t, inlier mask), or None when no usable
    pair exists.
    """
    z, w = _complex(src), _complex(dst)
    thr = np.broadcast_to(np.asarray(inlier_threshold, dtype=float), (len(z),))
    weights = None if weights is None else np.asarray(weights, dtype=float)
    i, j = np.triu_indices(len(z), k=1)
    keep = (np.abs(w[i] - w[j]) >= max(min_pair_distance, 1e-9)) & (np.abs(z[i] - z[j]) > 1e-12)
    i, j = i[keep], j[keep]
    if len(i) == 0:
        return None
    if len(i) > max_pairs:
        pick = np.random.default_rng(seed).choice(len(i), max_pairs, replace=False)
        i, j = i[pick], j[pick]
    a = (w[j] - w[i]) / (z[j] - z[i])
    b = w[i] - a * z[i]
    sq = np.abs(a[:, None] * z[None, :] + b[:, None] - w[None, :]) ** 2
    thr2 = thr**2
    count = (sq < thr2).sum(axis=1)
    cost = np.minimum(sq, thr2).sum(axis=1)
    best = int(np.lexsort((cost, -count))[0])
    inliers = sq[best] < thr2

    def fit(mask: np.ndarray) -> tuple[float, float, np.ndarray]:
        return fit_similarity_2d(np.asarray(src)[mask], np.asarray(dst)[mask],
                                 None if weights is None else weights[mask])

    for _ in range(2):
        if inliers.sum() < 2:
            break
        scale, yaw, t = fit(inliers)
        resid = np.abs(scale * np.exp(1j * yaw) * z + complex(*t) - w)
        inliers = resid < thr
    if inliers.sum() < 2:
        return None
    scale, yaw, t = fit(inliers)
    return scale, yaw, t, inliers


def compass_yaw(
    view_dirs: np.ndarray, headings_deg: list[float | None], sigma_deg: float | None = None
) -> tuple[float, int] | None:
    """Yaw (rad, counter-clockwise) turning gravity-aligned viewing directions onto compass headings.

    Robust circular mean: headings more than Params.compass_outlier_sigmas x sigma_deg
    (default Params.compass_sigma_deg) off the plain mean are dropped. Returns
    (yaw_rad, number of headings used) or None.
    """
    p = params.current()
    sigma_deg = p.compass_sigma_deg if sigma_deg is None else sigma_deg
    # Headings turn clockwise and yaw counter-clockwise, so yaw = SfM heading - compass heading.
    diffs = np.radians([
        heading_from_en(d[0], d[1]) - h for d, h in zip(np.asarray(view_dirs, dtype=float), headings_deg)
        if h is not None
    ])
    if len(diffs) == 0:
        return None
    mean = math.atan2(np.sin(diffs).sum(), np.cos(diffs).sum())
    used = np.abs(np.angle(np.exp(1j * (diffs - mean)))) <= math.radians(p.compass_outlier_sigmas * sigma_deg)
    if used.any():
        mean = math.atan2(np.sin(diffs[used]).sum(), np.cos(diffs[used]).sum())
    else:
        used[:] = True
    return mean, int(used.sum())


def yaw_prior(
    view_dirs: np.ndarray, headings_deg: list[float | None], gps_en: np.ndarray
) -> tuple[float, float, str] | None:
    """(yaw_rad, sigma_rad, source) from the compass readings, else from a pseudo-compass.

    The pseudo-compass assumes every photo looks from its GPS fix at the
    landmark centre (the ENU origin). It is used only when no photo has a
    compass reading: on real facades it is 10-20 degrees off, but it never
    turns a model around the way a short or biased GPS baseline can.
    """
    p = params.current()
    compass = compass_yaw(view_dirs, headings_deg)
    if compass:
        sigma = max(math.radians(p.compass_sigma_deg) / math.sqrt(compass[1]),
                    math.radians(p.compass_yaw_min_sigma_deg))
        return compass[0], sigma, "compass"
    towards = [heading_from_en(-e, -n) if math.hypot(e, n) > 1.0 else None for e, n in np.asarray(gps_en, float)]
    pseudo = compass_yaw(view_dirs, towards, sigma_deg=p.pseudo_compass_sigma_deg)
    if pseudo:
        sigma = max(math.radians(p.pseudo_compass_sigma_deg) / math.sqrt(pseudo[1]),
                    math.radians(p.pseudo_compass_min_sigma_deg))
        return pseudo[0], sigma, "landmark-direction"
    return None


def similarity_matrix(scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    """4x4 homogeneous matrix of p -> scale * rotation @ p + translation."""
    t = np.eye(4)
    t[:3, :3] = scale * np.asarray(rotation, dtype=float)
    t[:3, 3] = translation
    return t


def apply_similarity(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply a 4x4 similarity to (n, 3) points."""
    return np.asarray(points, dtype=float) @ transform[:3, :3].T + transform[:3, 3]


@dataclass
class Georegistration:
    transform: np.ndarray        # 4x4 similarity: SfM world -> ENU (always proper: positive scale)
    georegistered: bool          # scale and position come from GPS (else a yaw prior + a scale guess)
    inliers: np.ndarray          # per camera: GPS fix consistent with the fit
    rmse_m: float | None         # horizontal camera-centre vs GPS residual over the inliers
    message: str
    yaw_source: str = "gps"      # gps | gps+compass | gps+landmark-direction | compass | landmark-direction
    gps_yaw_deg: float | None = None
    prior_yaw_deg: float | None = None
    yaw_sigma_deg: float | None = None   # 1-sigma yaw uncertainty, Params.yaw_systematic_sigma_deg included
    scale_sigma: float | None = None     # relative 1-sigma scale uncertainty from the GPS fit (None: not from GPS)
    anchor: GeoAnchor | None = None      # the same transform as a GeoAnchor, with every component's sigma


def _fallback_scale(c: np.ndarray, gps_en: np.ndarray, scene_points: np.ndarray | None,
                    rot_g: np.ndarray) -> float | None:
    """Scale without a usable GPS similarity: camera-to-scene vs GPS camera-to-landmark distances.

    Runs large by the ratio of the distance to the landmark centre over the
    distance to the near facade (about 1.3x on the synthetic box, 1.5-2.4x on
    real buildings); GPS distances between camera pairs were tried and were
    worse whenever the photos span less than a few GPS sigmas.
    """
    if scene_points is None or not len(scene_points):
        return None
    target = np.median(np.asarray(scene_points, dtype=float) @ rot_g.T, axis=0)
    sfm_dist = float(np.median(np.linalg.norm(c[:, :2] - target[:2], axis=1)))
    gps_dist = float(np.median(np.linalg.norm(gps_en, axis=1)))
    return gps_dist / sfm_dist if sfm_dist > 0 and gps_dist > 0 else None


def georegister(
    centers: np.ndarray,
    ups: np.ndarray,
    rights: np.ndarray,
    view_dirs: np.ndarray,
    gps_en: np.ndarray,
    headings_deg: list[float | None],
    camera_height_m: float,
    scene_points: np.ndarray | None = None,
    gps_sigma_m: np.ndarray | float | None = None,
) -> Georegistration:
    """Similarity from the SfM frame to ENU (landmark centre at the origin).

    1. Gravity: rotate estimate_up onto +z. A full 3-D Sim3 against GPS would
       have to recover tilt from the camera centres alone, which is
       ill-conditioned when they lie on a short arc at equal height.
    2. Robust weighted 2-D similarity of camera centres' x/y onto GPS
       east/north. gps_sigma_m is the per-photo GPS sigma (reported accuracy;
       Params.gps_sigma_m when None or NaN); a fix is an inlier within
       Params.gps_inlier_threshold_m.
    3. Yaw: the GPS yaw is fused with yaw_prior (compass, else the
       pseudo-compass) by inverse variance, which matters for a single facade,
       where GPS noise over a 20 m arc can turn the model by tens of degrees.
       When the two disagree by more than Params.max_yaw_conflict_deg the GPS fit is
       not trusted at all: fusing across more than 90 degrees would make the
       least-squares scale negative, i.e. mirror the model.
    4. z offset so cameras sit at camera_height_m.
    Without a usable, consistent GPS fit (too few inliers, RMS spread below
    the GPS sigma, a predicted scale error above Params.max_gps_scale_error, or the
    yaw conflict) the yaw comes from the prior alone, the scale from
    _fallback_scale (the weak GPS fit if that has nothing), the position from
    the median GPS offset, and georegistered is False. The scale is always
    positive, so the transform never mirrors.
    """
    centers = np.asarray(centers, dtype=float)
    gps_en = np.asarray(gps_en, dtype=float)
    n = len(centers)
    p = params.current()
    sigma = np.broadcast_to(np.asarray(p.gps_sigma_m if gps_sigma_m is None else gps_sigma_m, dtype=float), (n,))
    sigma = np.maximum(np.where(np.isfinite(sigma), sigma, p.gps_sigma_m), p.gps_sigma_min_m)
    weights = 1.0 / sigma**2
    thresholds = np.maximum(p.gps_inlier_sigmas * sigma, p.gps_inlier_min_m)
    up_local = estimate_up(ups, rights, centers)
    rot_g = rotation_to_z(up_local)
    c = centers @ rot_g.T
    d = np.asarray(view_dirs, dtype=float) @ rot_g.T
    fit = robust_similarity_2d(c[:, :2], gps_en, thresholds, min_pair_distance=float(np.median(sigma)),
                               weights=weights)
    prior = yaw_prior(d, headings_deg, gps_en)
    inliers = fit[3] if fit else np.zeros(n, dtype=bool)
    spread, sigma_gps = 0.0, math.inf
    if inliers.sum() >= 2:
        g, wt = gps_en[inliers], weights[inliers]
        spread = math.sqrt(float(((g - g.mean(axis=0)) ** 2).sum(axis=1).mean()))
        # Weighted least squares w = a z + b: std(a) / |a| = 1 / sqrt(sum weight |w - mean w|^2), for scale and yaw.
        gm = (wt @ g) / wt.sum()
        sigma_gps = 1.0 / max(math.sqrt(float(wt @ ((g - gm) ** 2).sum(axis=1))), 1e-9)
    usable = (inliers.sum() >= max(MIN_REGISTERED, math.ceil(n / 2))
              and spread >= float(np.median(sigma[inliers])) and sigma_gps <= p.max_gps_scale_error)
    gps_yaw = fit[1] if fit else None
    prior_note = ""
    if prior and gps_yaw is not None:
        prior_note = (f"; GPS yaw - {prior[2]} yaw ="
                      f" {angle_diff_deg(math.degrees(gps_yaw), math.degrees(prior[0])):.0f} deg")

    georegistered, conflict, fallback = False, False, None
    yaw_sigma: float | None = None
    if usable:
        yaw, yaw_sigma, source = gps_yaw, sigma_gps, "gps"
        if prior:
            delta = math.radians(angle_diff_deg(math.degrees(prior[0]), math.degrees(gps_yaw)))
            conflict = abs(delta) > math.radians(p.max_yaw_conflict_deg)
            weight = sigma_gps**2 / (sigma_gps**2 + prior[1] ** 2)
            yaw += weight * delta
            yaw_sigma = 1.0 / math.sqrt(1.0 / sigma_gps**2 + 1.0 / prior[1] ** 2)
            source = f"gps+{prior[2]}"
        if not conflict:
            # Weighted least-squares scale and translation for the fused yaw.
            u = np.exp(1j * yaw) * _complex(c[inliers, :2])
            w, wt = _complex(gps_en[inliers]), weights[inliers]
            um, wm = complex(wt @ u) / wt.sum(), complex(wt @ w) / wt.sum()
            scale = complex((wt * np.conj(u - um)) @ (w - wm)).real / float(wt @ (np.abs(u - um) ** 2))
            shift = wm - scale * um
            georegistered = scale > 0
            message = (f"georegistered from {int(inliers.sum())}/{n} GPS fixes (spread {spread:.1f} m,"
                       f" scale +-{100 * sigma_gps:.0f}%, yaw +-{math.degrees(yaw_sigma):.0f} deg from {source}"
                       f"{prior_note})")
    if not georegistered:
        if prior:
            yaw, source, yaw_sigma = prior[0], prior[2], prior[1]
        else:
            yaw, source = (gps_yaw, "gps") if gps_yaw is not None else (0.0, "none")
            yaw_sigma = sigma_gps if gps_yaw is not None and math.isfinite(sigma_gps) else None
        fallback = _fallback_scale(c, gps_en, scene_points, rot_g)
        scale = fallback or (fit[0] if fit else 1.0)
        u = np.exp(1j * yaw) * _complex(c[:, :2])
        offsets = _complex(gps_en) - scale * u
        shift = complex(float(np.median(offsets.real)), float(np.median(offsets.imag)))
        if conflict:
            reason = (f"GPS yaw is {abs(angle_diff_deg(math.degrees(gps_yaw), math.degrees(prior[0]))):.0f} deg"
                      f" from the {prior[2]} yaw")
        else:
            reason = (f"{int(inliers.sum())}/{n} consistent GPS fixes spread over {spread:.1f} m"
                      " cannot fix the scale")
        message = f"not georegistered: {reason}; {source} yaw, approximate scale"
        inliers = np.abs(offsets - shift) < thresholds

    rotation = rotation_z(yaw) @ rot_g
    z_shift = camera_height_m - scale * float(np.median(c[:, 2]))
    # The GPS and compass sigmas assume independent errors; fixes taken at one site share multipath and
    # hand-placed geotags (real facade jobs were 6-9 deg off while reporting +-2-3 deg): a floor in quadrature.
    yaw_sigma_deg = (None if yaw_sigma is None or not math.isfinite(yaw_sigma)
                     else math.hypot(math.degrees(yaw_sigma), p.yaw_systematic_sigma_deg))
    transform = similarity_matrix(scale, rotation, np.array([shift.real, shift.imag, z_shift]))
    resid = apply_similarity(transform, centers)[:, :2] - gps_en
    rmse = float(np.sqrt((resid[inliers] ** 2).sum(axis=1).mean())) if inliers.any() else None
    anchor = _anchor(centers, ups, rights, up_local, transform, scale, rotation, yaw, bool(georegistered), inliers,
                     weights, sigma, gps_en, source,
                     None if georegistered else "camera-to-scene" if fallback else "gps-weak" if fit else "none",
                     gps_yaw, prior, sigma_gps, spread)
    return Georegistration(
        transform, bool(georegistered), inliers, rmse, message, yaw_source=source,
        gps_yaw_deg=None if gps_yaw is None else math.degrees(gps_yaw),
        prior_yaw_deg=None if prior is None else math.degrees(prior[0]),
        yaw_sigma_deg=None if yaw_sigma_deg is None else round(yaw_sigma_deg, 2),
        scale_sigma=round(sigma_gps, 4) if georegistered and math.isfinite(sigma_gps) else None,
        anchor=anchor,
    )


def _r6(v: float | None) -> float | None:
    return None if v is None or not math.isfinite(v) else round(float(v), 6)


def _anchor(centers: np.ndarray, ups: np.ndarray, rights: np.ndarray, up_local: np.ndarray, transform: np.ndarray,
            scale: float, rotation: np.ndarray, yaw: float, georegistered: bool, inliers: np.ndarray,
            weights: np.ndarray, sigma: np.ndarray, gps_en: np.ndarray, yaw_source: str,
            fallback_scale_source: str | None, gps_yaw: float | None, prior: tuple[float, float, str] | None,
            sigma_gps: float, spread: float) -> GeoAnchor:
    """georegister's result as a GeoAnchor (record only: nothing in georegister reads it). Core sigmas (before the
    Params floors, crowd3d.anchor):
    - georegistered: yaw = the fused least-squares sigma (the GPS one alone without a prior) times the Birge ratio
      max(1, |GPS yaw - prior yaw| / sqrt(sigma_gps^2 + sigma_prior^2)); log-scale = sigma_gps times the variance
      factor vf = max(1, sqrt(sum(w r^2) / (2N - 4))) of the N inliers' weighted residuals; east/north at the pivot
      (the inliers' weighted centre) = vf / sqrt(sum w);
    - otherwise: the prior's yaw sigma (the weak GPS fit's without a prior, 180 deg without either),
      Params.anchor_fallback_log_scale_sigma, and 1.25 x median GPS sigma / sqrt(n) for the median offset at the
      cameras' centre;
    - tilt: gravity_bootstrap of the cameras in ENU (the tilt about east and north), in the order of their ENU
      centres, on a 2^-30 grid (about 1e-9 m, and 1e-9 for the unit vectors): a model read back from disk lists its
      images in another order and another SfM run of the same photos has another gauge (frame), so neither may steer
      the resamples; on the grid the rounding noise they leave is gone, and a resample's copies of one camera average
      to exactly that camera (estimate_up's spread of them is 0, not noise normalised into any direction), which
      makes the sigma a continuous function of the cameras (anchor.ANCHOR_METHOD 2); up: Params.anchor_z_sigma_m."""
    p = params.current()

    def grid(v: np.ndarray) -> np.ndarray:
        return np.round(np.asarray(v, float) * 2.0**30) / 2.0**30

    enu = grid(apply_similarity(transform, np.asarray(centers, float)))
    order = np.lexsort(enu.T[::-1])
    tilt, _ = gravity_bootstrap(grid(np.asarray(ups, float)[order] @ rotation.T),
                                grid(np.asarray(rights, float)[order] @ rotation.T), enu[order])
    conflict = (None if prior is None or gps_yaw is None
                else angle_diff_deg(math.degrees(prior[0]), math.degrees(gps_yaw)))
    birge, vf = 1.0, 1.0
    if georegistered:
        wt = weights[inliers]
        r2 = ((apply_similarity(transform, centers[inliers])[:, :2] - gps_en[inliers]) ** 2).sum(axis=1)
        dof = 2 * int(inliers.sum()) - 4
        vf = max(1.0, math.sqrt(float(wt @ r2) / dof)) if dof > 0 else 1.0
        yaw_core = sigma_gps
        if prior is not None:
            yaw_core = 1.0 / math.sqrt(1.0 / sigma_gps**2 + 1.0 / prior[1] ** 2)
            birge = max(1.0, abs(math.radians(conflict)) / math.hypot(sigma_gps, prior[1]))
            yaw_core *= birge
        xy_ls = 1.0 / math.sqrt(float(wt.sum()))
        core = {"yaw_deg": math.degrees(yaw_core), "log_scale": sigma_gps * vf, "xy_m": vf * xy_ls}
        pivot_local = (wt @ centers[inliers]) / wt.sum()
    else:
        yaw_core = (prior[1] if prior is not None else
                    sigma_gps if gps_yaw is not None and math.isfinite(sigma_gps) else math.pi)
        xy_ls = 1.25 * float(np.median(sigma)) / math.sqrt(len(centers))
        core = {"yaw_deg": min(math.degrees(yaw_core), 180.0), "log_scale": p.anchor_fallback_log_scale_sigma,
                "xy_m": xy_ls}
        pivot_local = centers.mean(axis=0)
    core = {**{k: round(v, 6) for k, v in core.items()}, "tilt_deg": round(tilt, 6)}
    cal = calibration(p)
    pivot = apply_similarity(transform, pivot_local[None])[0]
    return GeoAnchor(
        frame="job", level="full" if georegistered else "none" if yaw_source == "none" else "prior_only",
        scale=float(scale), rotation_xyzw=quat_xyzw(rotation), translation=[float(v) for v in transform[:3, 3]],
        up_local=[float(v) for v in up_local], yaw_deg=math.degrees(yaw), pivot_enu=[round(float(v), 4) for v in pivot],
        sigma=floored_sigma(core, p.anchor_z_sigma_m, cal["floors"]),
        source={"yaw": yaw_source, "scale": "gps" if georegistered else fallback_scale_source,
                "xy": "gps" if georegistered else "gps-median", "up": "camera-height", "gravity": "camera-rights"},
        raw={"yaw_gps_deg": _r6(None if gps_yaw is None else math.degrees(gps_yaw)),
             "yaw_prior_deg": _r6(None if prior is None else math.degrees(prior[0])),
             "yaw_prior_source": None if prior is None else prior[2], "yaw_conflict_deg": _r6(conflict),
             "sigma_gps": _r6(sigma_gps), "sigma_prior_deg": _r6(None if prior is None else math.degrees(prior[1])),
             "birge": _r6(birge), "variance_factor": _r6(vf), "xy_ls_m": _r6(xy_ls), "tilt_boot_deg": _r6(tilt),
             "gps_inliers": int(inliers.sum()), "gps_spread_m": _r6(spread), "sigma_core": core},
        calibration=cal)


def gps_inlier_threshold_m(gps_accuracy_m: float | None) -> float:
    """georegister's inlier threshold for one photo (Params.gps_inlier_threshold_m of the current set)."""
    return params.current().gps_inlier_threshold_m(gps_accuracy_m)


def camera_flags(
    gps_residual_m: float,
    inlier_threshold_m: float,
    compass_offset_deg: float | None,
    focal_ratio: float | None,
) -> list[str]:
    """Why one registered camera's pose cannot be right (empty: no evidence against it).

    - "gps_and_compass": beyond the GPS inlier threshold from its own fix AND
      more than Params.gross_compass_deg off its own compass. Typical of a symmetric
      building, where SfM puts the photo in front of a twin facade.
    - "far_from_gps": more than Params.far_from_gps_factor thresholds from its own fix
      (wrong depth for a telephoto view, a camera off in a fragment).
    - "odd_focal": focal length outside Params.focal_ratio_range x the longer side.
    """
    p = params.current()
    flags = []
    if (gps_residual_m > inlier_threshold_m and compass_offset_deg is not None
            and abs(compass_offset_deg) > p.gross_compass_deg):
        flags.append("gps_and_compass")
    if gps_residual_m > p.far_from_gps_factor * inlier_threshold_m:
        flags.append("far_from_gps")
    if focal_ratio is not None and not p.focal_ratio_min <= focal_ratio <= p.focal_ratio_max:
        flags.append("odd_focal")
    return flags


def check_cameras(
    transform: np.ndarray,
    rots: np.ndarray,
    centers: np.ndarray,
    gps_en: np.ndarray,
    headings_deg: list[float | None],
    gps_accuracy_m: list[float | None],
    focal_ratios: np.ndarray | list[float | None],
    photo_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """ReconstructionResult.cameras entries: ENU pose of each registered camera plus its camera_flags.

    rots are cam_from_world rotations and centers camera centres in the SfM
    frame; transform is the SfM -> ENU similarity.
    """
    cam_enu = apply_similarity(transform, centers)
    view_enu = np.asarray(rots, dtype=float)[:, 2, :] @ transform[:3, :3].T
    cameras: dict[str, dict[str, Any]] = {}
    for pid, pos, v, g, h, acc, f, rot, c in zip(photo_ids, cam_enu, view_enu, np.asarray(gps_en, float), headings_deg,
                                                 gps_accuracy_m, focal_ratios, np.asarray(rots, float),
                                                 np.asarray(centers, float)):
        heading = heading_from_en(v[0], v[1])
        offset = None if h is None else angle_diff_deg(heading, h)
        resid = float(np.linalg.norm(pos[:2] - g))
        cameras[pid] = {
            "enu": [round(float(x), 3) for x in pos],
            "heading_deg": round(heading, 2),
            "gps_residual_m": round(resid, 2),
            "compass_offset_deg": None if offset is None else round(offset, 2),
            "focal_ratio": None if f is None else round(float(f), 4),
            "gps_en": [round(float(g[0]), 2), round(float(g[1]), 2)],   # the photo's own fix (stylize's GPS check)
            "flags": camera_flags(resid, gps_inlier_threshold_m(acc), offset, None if f is None else float(f)),
            # the pose in the reconstruction's own frame (cam_from_local, centre): enu and heading_deg are this pose
            # through the GeoAnchor, written at job time
            "local": {"q": [round(x, 9) for x in quat_xyzw(rot)], "c": [round(float(x), 6) for x in c]},
        }
    return cameras


def image_name(index: int, photo_id: str) -> str:
    """SfM image name: COLMAP numbers images in name order, so the upload order fixes its choices."""
    return f"{index:05d}_{photo_id}.jpg"


def photo_id_from_image_name(name: str) -> str:
    return Path(name).stem.split("_", 1)[-1]


@dataclass
class GeoregisterInputs:
    """What georegister and check_cameras read of one mapped model (georegister_inputs)."""

    images: list[Any]              # its posed images of the job's photos (pycolmap.Image)
    photos: list[ReconPhoto]
    rots: np.ndarray               # cam_from_world rotations
    centers: np.ndarray            # camera centres (SfM frame)
    focal: np.ndarray              # focal length / longer image side
    gps_en: np.ndarray             # each photo's GPS fix, ENU east/north
    accuracy: np.ndarray           # reported GPS accuracy (NaN: none)
    use: np.ndarray                # cameras that steer the alignment (focal ratio in range)
    xyz: np.ndarray                # the model's points


def georegister_inputs(rec, by_name: dict[str, ReconPhoto], gps: dict[str, np.ndarray]) -> GeoregisterInputs | None:
    """The arrays georegister_model needs from a pycolmap.Reconstruction (SfM image name -> photo, and -> its GPS
    east/north); None with fewer than MIN_REGISTERED posed images of those photos. Pure: an offline replay of a kept
    model (scripts/benchmark.py rescore --regeo) runs exactly what the job ran."""
    images = [im for im in rec.images.values() if im.has_pose and im.name in by_name]
    if len(images) < MIN_REGISTERED:
        return None
    photos = [by_name[im.name] for im in images]
    focal = np.array([rec.cameras[im.camera_id].mean_focal_length()
                      / max(rec.cameras[im.camera_id].width, rec.cameras[im.camera_id].height) for im in images])
    # Degenerate cameras (absurd focal length) must not steer the alignment of the others.
    tun = params.current()
    sane = (focal >= tun.focal_ratio_min) & (focal <= tun.focal_ratio_max)
    return GeoregisterInputs(
        images=images, photos=photos,
        rots=np.array([im.cam_from_world().rotation.matrix() for im in images]),
        centers=np.array([im.projection_center() for im in images]),
        focal=focal, gps_en=np.array([gps[im.name] for im in images]),
        accuracy=np.array([np.nan if p.gps_accuracy_m is None else p.gps_accuracy_m for p in photos]),
        use=sane if sane.sum() >= MIN_REGISTERED else np.ones(len(images), dtype=bool),
        xyz=np.array([pt.xyz for pt in rec.points3D.values()]).reshape(-1, 3))


def georegister_model(inputs: GeoregisterInputs, camera_height_m: float) -> tuple[Georegistration, dict[str, Any]]:
    """georegister on the cameras inputs.use selects, then check_cameras on all of them."""
    use, rots, photos = inputs.use, inputs.rots, inputs.photos
    geo = georegister(
        inputs.centers[use],
        ups=-rots[use, 1, :],          # image -y axis in world
        rights=rots[use, 0, :],
        view_dirs=rots[use, 2, :],
        gps_en=inputs.gps_en[use],
        headings_deg=[p.heading_deg for p, u in zip(photos, use) if u],
        camera_height_m=camera_height_m,
        scene_points=inputs.xyz,
        gps_sigma_m=inputs.accuracy[use],
    )
    cameras = check_cameras(geo.transform, rots, inputs.centers, inputs.gps_en, [p.heading_deg for p in photos],
                            [p.gps_accuracy_m for p in photos], inputs.focal, [p.photo_id for p in photos])
    return geo, cameras


# ---------------------------------------------------------------- COLMAP


@dataclass
class _Candidate:
    """One mapped model, geo-registered and checked camera by camera."""

    rec: Any                                  # pycolmap.Reconstruction
    images: list[Any]
    geo: Georegistration
    cameras: dict[str, dict[str, Any]]
    heading_errors: list[float]               # |estimated - compass| per camera with a compass reading
    attempt: str
    num_models: int = 1                       # models that mapping attempt produced

    @property
    def num_flagged(self) -> int:
        return sum(bool(c["flags"]) for c in self.cameras.values())

    @property
    def compass_error_deg(self) -> float | None:
        """Mean compass disagreement, each capped at 90 deg so one wild reading cannot dominate."""
        return float(np.mean(np.minimum(self.heading_errors, 90.0))) if self.heading_errors else None

    def key(self) -> tuple:
        """Larger is better: not degenerate, cameras without evidence against them, then compass agreement."""
        return (self.rec.num_points3D() >= params.current().min_candidate_points, len(self.images) - self.num_flagged,
                -(self.compass_error_deg or 0.0), len(self.images))

    def suspicious(self) -> bool:
        p = params.current()
        return (self.num_flagged > p.retry_flagged_fraction * len(self.images)
                or (self.compass_error_deg or 0.0) > p.retry_compass_error_deg)


class ColmapReconstructor:
    """Incremental SfM with pycolmap (CPU), then gravity + GPS geo-registration.

    1. Symlink (or copy) processed images into work_dir/images, named by
       upload order (image_name). Import them first, single-threaded, so
       COLMAP's image ids do not depend on thread timing; then extract
       features (CPU; max image size from settings). Photos with a 35 mm
       focal length get their focal prior written directly (COLMAP only reads
       it when FocalLength is present too); photos without any focal tag get
       their own camera (COLMAP's AUTO mode).
    2. Exhaustive matching (GPS-guided spatial matching above
       Params.exhaustive_max_images).
       With a FeatureCache (cache=, landmark_map) the database is seeded with the cached features before extraction
       and the cached pairs before matching, so COLMAP computes only what is new; the job's rows are harvested into
       the cache before mapping (the 'cache' timing, result.cache_stats). The database ends the same either way.
    3. Incremental mapping (EXIF focal length kept fixed for small sets). Every
       model is geo-registered (georegister, without degenerate cameras) and
       each camera is checked against its own GPS and compass (camera_flags).
       Small sets are mapped with Params.mapping_attempts seeds, larger ones again
       when the result is suspicious; when several cameras disagree with both
       GPS and compass (a symmetric building's twin facades), the verified
       pairs linking them to the rest are cut and the set is mapped again.
       The model with most unflagged cameras wins and is written to
       work_dir/sparse/0.
       With a base (base=, a landmark_map.MapBase: the saved model the job's target shows) the job's photos are
       first registered into it (_continue, over the same database); an accepted continuation is the result
       (result.continued), a rejected one falls back to the attempts above.
    4. Per registered camera: ENU position, heading from its viewing direction
       and the check results; heading_mae_deg vs EXIF headings when present.
    5. Colours from the images, PLY (in ENU) to out_ply.
    Data problems give ok=False with a message; only programming errors raise.
    """

    name = "colmap"
    feature_cache = True      # reads and fills a landmark_map.FeatureCache when the job runner passes one
    map_continue = True       # continues a landmark_map.MapBase when the job runner passes one (_continue)

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.exhaustive_max_images: int | None = None   # None: Params.exhaustive_max_images

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir, *,
                    cache: FeatureCache | None = None, base: MapBase | None = None) -> ReconstructionResult:
        try:
            import pycolmap
        except ImportError:
            return ReconstructionResult(ok=False, message="pycolmap is not installed", num_input=len(photos))
        out_ply, work_dir = Path(out_ply), Path(work_dir)
        timings: dict[str, float] = {}
        tun = params.current()   # one consistent set for the whole job
        exhaustive_max = tun.exhaustive_max_images if self.exhaustive_max_images is None else self.exhaustive_max_images

        def fail(message: str, **extra: Any) -> ReconstructionResult:
            return ReconstructionResult(ok=False, message=message, num_input=len(photos), timings_s=timings, **extra)

        def lap(stage: str, start: float) -> None:
            timings[stage] = round(timings.get(stage, 0.0) + time.monotonic() - start, 2)

        present = [p for p in photos if Path(p.image_path).is_file()]
        if len(present) < MIN_REGISTERED:
            return fail(f"need at least {MIN_REGISTERED} photo files, found {len(present)} of {len(photos)}")
        by_name = {image_name(k, p.photo_id): p for k, p in enumerate(present)}
        enu = LocalENU(landmark_lat, landmark_lon)
        gps = {name: np.array(enu.to_enu(p.lat, p.lon)[:2]) for name, p in by_name.items()}

        images_dir, sparse_dir, db_path = work_dir / "images", work_dir / "sparse", work_dir / "database.db"
        attempts_dir = work_dir / "attempts"
        for d in (images_dir, sparse_dir, attempts_dir):
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
        db_path.unlink(missing_ok=True)
        for name, p in by_name.items():
            try:
                os.symlink(Path(p.image_path).resolve(), images_dir / name)
            except OSError:
                shutil.copyfile(p.image_path, images_dir / name)

        pycolmap.logging.minloglevel = pycolmap.logging.Level.ERROR.value
        pycolmap.set_random_seed(0)
        names = sorted(by_name)
        job_cache: JobCache | None = None
        try:
            t = time.monotonic()
            pycolmap.Database.open(db_path).close()
            pycolmap.import_images(db_path, images_dir, camera_mode=pycolmap.CameraMode.AUTO, image_names=names)
            extraction = extraction_options(pycolmap, self.settings.processing_max_side_px)
            verification = pycolmap.TwoViewGeometryOptions()
            verification.ransac.random_seed = 0
            lap("extract", t)
            if cache is not None:
                t = time.monotonic()
                job_cache = JobCache(cache, pycolmap, db_path, {n: Path(p.image_path) for n, p in by_name.items()},
                                     extraction, verification)
                job_cache.seed_features()
                lap("cache", t)
            t = time.monotonic()
            pycolmap.extract_features(db_path, images_dir, image_names=names,
                                      camera_mode=pycolmap.CameraMode.AUTO,
                                      extraction_options=extraction, device=pycolmap.Device.cpu)
            self._write_focal_priors(pycolmap, db_path, by_name)
            lap("extract", t)

            t = time.monotonic()
            pairing = self._write_gps_priors(pycolmap, db_path, gps) if len(by_name) > exhaustive_max else None
            lap("match", t)
            if job_cache is not None:
                t = time.monotonic()
                job_cache.seed_pairs(pairing)
                lap("cache", t)
            t = time.monotonic()
            if pairing is not None:
                pycolmap.match_spatial(db_path, pairing_options=pairing, verification_options=verification,
                                       device=pycolmap.Device.cpu)
            else:
                pycolmap.match_exhaustive(db_path, verification_options=verification, device=pycolmap.Device.cpu)
            lap("match", t)
            if job_cache is not None:
                t = time.monotonic()
                job_cache.harvest(work_dir.name)
                lap("cache", t)
            fix_focal = len(by_name) < tun.focal_refine_min_images and self._all_focal_priors(pycolmap, db_path)

            candidates: list[_Candidate] = []

            def mapping_options(seed: int):
                pycolmap.set_random_seed(seed)
                options = pycolmap.IncrementalPipelineOptions()
                options.random_seed = seed
                options.min_model_size = MIN_REGISTERED
                options.extract_colors = True
                if fix_focal:
                    # A few views of one near-planar facade cannot pin down focal length:
                    # refining it shrinks or stretches the model, so trust EXIF instead.
                    options.ba_refine_focal_length = False
                    options.mapper.abs_pose_refine_focal_length = False
                return options

            def attempt(label: str, seed: int, database: Path) -> _Candidate | None:
                t = time.monotonic()
                options = mapping_options(seed)
                out = attempts_dir / label
                out.mkdir(parents=True, exist_ok=True)
                models = pycolmap.incremental_mapping(database, images_dir, out, options=options)
                lap("mapping", t)
                found = [c for c in (self._evaluate(rec, by_name, gps, label) for rec in models.values()) if c]
                if not found:
                    return None
                best = max(found, key=_Candidate.key)
                best.num_models = len(models)
                candidates.append(best)
                log.info("mapping attempt %s: %d/%d registered, %d flagged, compass error %s",
                         label, len(best.images), len(by_name), best.num_flagged, best.compass_error_deg)
                return best

            continued = None
            if base is not None:
                t = time.monotonic()
                kept, continued = self._continue(pycolmap, base, db_path, images_dir, attempts_dir, by_name, gps,
                                                 mapping_options(0))
                lap("mapping", t)
                if kept is not None:
                    candidates.append(kept)
            # Incremental mapping depends on the seed and on which pair it starts from; on real photos one try
            # in three can come out broken (symmetric facades, weak views), so suspicious results are mapped again.
            # An accepted continuation is the result; a rejected one leaves the attempts from scratch to it.
            small = len(by_name) <= tun.small_set_images
            max_attempts = 0 if candidates else tun.mapping_attempts if small else tun.mapping_attempts_large
            current = attempt("seed0", 0, db_path) if max_attempts else None
            tried, seed, cut_done = 1, 1, False
            while tried < max_attempts and (current is None or current.suspicious() or small):
                twins = {photo_id_from_image_name(im.name) for im in current.images
                         if "gps_and_compass" in current.cameras[photo_id_from_image_name(im.name)]["flags"]
                         } if current else set()
                if len(twins) >= 2 and not cut_done:
                    cut_done = True
                    cut_db = attempts_dir / "cut.db"
                    shutil.copyfile(db_path, cut_db)
                    self._cut_pairs(pycolmap, cut_db, {n for n in by_name if photo_id_from_image_name(n) in twins})
                    current = attempt("cut", 0, cut_db)
                else:
                    current = attempt(f"seed{seed}", seed, db_path)
                    seed += 1
                tried += 1
        except RuntimeError as e:   # COLMAP reports failed checks on bad input as RuntimeError
            return fail(f"COLMAP failed: {e}")

        if not candidates:
            return fail("structure-from-motion found no initial image pair; photos may not overlap enough")
        best = max(candidates, key=_Candidate.key)
        rec, images, geo = best.rec, best.images, best.geo
        if len(images) < MIN_REGISTERED:
            return fail(f"only {len(images)} of {len(by_name)} photos could be registered; "
                        f"need at least {MIN_REGISTERED} overlapping views", num_registered=len(images))
        if np.linalg.det(geo.transform[:3, :3]) <= 0:
            return fail("geo-registration produced a mirrored model", num_registered=len(images))

        t = time.monotonic()
        final = sparse_dir / "0"
        final.mkdir(parents=True, exist_ok=True)
        rec.write(final)
        points3d = list(rec.points3D.values())
        xyz = np.array([pt.xyz for pt in points3d]).reshape(-1, 3)
        rgb = np.array([pt.color for pt in points3d], dtype=np.uint8).reshape(-1, 3)
        points = apply_similarity(geo.transform, xyz)
        write_ply(out_ply, points.tolist(), rgb.tolist())
        lap("georegister_export", t)

        flagged = best.num_flagged
        message = (f"registered {len(images)}/{len(photos)} photos, {len(points)} points; {geo.message}"
                   + (f"; {flagged} camera(s) disagree with their own GPS/compass" if flagged else "")
                   + (f"; {best.num_models} separate models, kept the best" if best.num_models > 1 else "")
                   + (f"; best of {len(candidates)} mapping attempts ({best.attempt})" if len(candidates) > 1 else "")
                   + ("" if continued is None else
                      f"; continued model {base.model_id} (kept {continued['base_kept']}/{continued['base_registered']}"
                      " of its registered photos)" if continued["accepted"] else
                      f"; built from scratch, model {base.model_id} not continued: {continued['reason']}"))
        log.info("%s (timings %s)", message, timings)
        return ReconstructionResult(
            ok=True,
            message=message,
            ply_path=out_ply,
            num_input=len(photos),
            num_registered=len(images),
            num_points=len(points),
            mean_reprojection_error_px=float(rec.compute_mean_reprojection_error()),
            georegistered=geo.georegistered,
            alignment_rmse_m=geo.rmse_m,
            heading_mae_deg=float(np.mean(best.heading_errors)) if best.heading_errors else None,
            cameras=best.cameras,
            timings_s=timings,
            points_median_en=[round(float(v), 3) for v in np.median(points[:, :2], axis=0)] if len(points) else None,
            yaw_source=geo.yaw_source,
            mapping_attempts=len(candidates),
            yaw_sigma_deg=geo.yaw_sigma_deg,
            scale_sigma=geo.scale_sigma,
            anchor=geo.anchor.to_meta(),
            cache_stats=job_cache.stats.to_meta() if job_cache is not None else None,
            continued=continued,
        )

    def _continue(self, pycolmap, base: MapBase, db_path: Path, images_dir: Path, attempts_dir: Path,
                  by_name: dict[str, ReconPhoto], gps: dict[str, np.ndarray],
                  options) -> tuple[_Candidate | None, dict[str, Any]]:
        """Register this job's photos into the base model (stage 1b): incremental mapping from it (input_path) over
        the job's own database, its cameras refined too unless Params.map_fix_existing (COLMAP's periodic and final
        global bundle adjustment still run), then the unchanged geo-registration and camera checks (_evaluate).
        Accepted when the model keeps Params.map_continue_min_kept of the base's registered photos, is not degenerate
        nor suspicious and puts fewer than 2 cameras in front of a look-alike facade; returns (the candidate or None,
        ReconstructionResult.continued)."""
        tun = params.current()
        info: dict[str, Any] = {"base_model_id": base.model_id, "accepted": False, "reason": None,
                                "base_registered": None, "base_kept": None, "registered": None}
        try:
            start, base_photos = self._base_start(pycolmap, base, db_path, by_name)
            info["base_registered"] = len(base_photos)
            start_dir, out = attempts_dir / "base", attempts_dir / "continue"
            for d in (start_dir, out):
                d.mkdir(parents=True, exist_ok=True)
            start.write(start_dir)
            options.fix_existing_frames = tun.map_fix_existing
            models = pycolmap.incremental_mapping(db_path, images_dir, out, options=options, input_path=start_dir)
        except (ValueError, RuntimeError) as e:
            info["reason"] = f"cannot continue it: {e}"
            return None, info

        def kept(c: _Candidate) -> int:
            return sum(photo_id_from_image_name(im.name) in base_photos for im in c.images)

        found = [c for c in (self._evaluate(rec, by_name, gps, "continue") for rec in models.values()) if c]
        if not found:
            info["reason"] = f"fewer than {MIN_REGISTERED} photos registered"
            return None, info
        best = max(found, key=lambda c: (kept(c), c.key()))
        best.num_models = len(models)
        twins = sum("gps_and_compass" in c["flags"] for c in best.cameras.values())
        info.update(base_kept=kept(best), registered=len(best.images))
        if kept(best) < tun.map_continue_min_kept * len(base_photos):
            info["reason"] = f"it kept {kept(best)} of the base's {len(base_photos)} registered photos"
        elif best.rec.num_points3D() < tun.min_candidate_points:
            info["reason"] = f"only {best.rec.num_points3D()} points"
        elif best.suspicious():
            info["reason"] = (f"suspicious: {best.num_flagged} of {len(best.images)} cameras disagree with their"
                              f" sensors, compass error {best.compass_error_deg}")
        elif twins >= 2:
            info["reason"] = f"{twins} cameras sit in front of a look-alike facade"
        else:
            info["accepted"] = True
            return best, info
        return None, info

    @staticmethod
    def _base_start(pycolmap, base: MapBase, db_path: Path, by_name: dict[str, ReconPhoto]):
        """The base model in this job's database ids, the start of a continuation: its registered photos under the
        job's image names, image, camera, rig and frame ids (trivial rigs and frames, as import_images writes them),
        its refined cameras, poses and points. Raises ValueError when the job cannot take it over as it is: a photo
        missing, other features (a feature version change), photos grouped into other cameras. Returns (the model,
        the photo ids of its registered images)."""
        rec = pycolmap.Reconstruction(str(base.sparse_dir))
        name_of = {p.photo_id: name for name, p in by_name.items()}
        db = pycolmap.Database.open(db_path)
        try:
            images = {im.name: im for im in db.read_all_images()}
            cameras = {c.camera_id: c for c in db.read_all_cameras()}
            frames = {f.frame_id: f for f in db.read_all_frames()}
            posed, camera_of = [], {}
            for im in sorted(rec.images.values(), key=lambda im: im.name):
                if not im.has_pose:
                    continue
                pid = photo_id_from_image_name(im.name)
                mine = images.get(name_of.get(pid, ""))
                if mine is None:
                    raise ValueError(f"its photo {pid} is not one of this job's")
                frame = frames.get(mine.frame_id)
                if mine.frame_id != mine.image_id or frame is None or frame.rig_id != mine.camera_id:
                    raise ValueError("the job's database has rigs or frames of its own")
                if camera_of.setdefault(mine.camera_id, im.camera_id) != im.camera_id:
                    raise ValueError("this job groups its photos into other cameras")
                keypoints = np.asarray(db.read_keypoints(mine.image_id), dtype=float)[:, :2]
                xy = np.array([p.xy for p in im.points2D], dtype=float).reshape(-1, 2)
                if keypoints.shape != xy.shape or (len(xy) and float(np.abs(keypoints - xy).max()) > 1e-3):
                    raise ValueError(f"its photo {pid} has other features in this job")
                posed.append((im, mine, keypoints))
        finally:
            db.close()
        start = pycolmap.Reconstruction()
        for camera_id, base_camera_id in camera_of.items():
            camera, theirs = pycolmap.Camera(rec.cameras[base_camera_id].todict()), cameras[camera_id]
            if (camera.model, camera.width, camera.height) != (theirs.model, theirs.width, theirs.height):
                raise ValueError(f"camera {camera_id} has another model or size in this job")
            camera.camera_id = camera_id
            start.add_camera_with_trivial_rig(camera)
        ids = {}
        for im, mine, keypoints in posed:
            start.add_image_with_trivial_frame(pycolmap.Image(name=mine.name, keypoints=keypoints,
                                                              camera_id=mine.camera_id, image_id=mine.image_id),
                                               im.cam_from_world())
            ids[im.image_id] = mine.image_id
        for point in rec.points3D.values():
            track = [pycolmap.TrackElement(ids[e.image_id], e.point2D_idx) for e in point.track.elements
                     if e.image_id in ids]
            if len(track) >= 2:
                start.points3D[start.add_point3D(point.xyz, pycolmap.Track(track), point.color)].error = point.error
        return start, {photo_id_from_image_name(im.name) for im, _, _ in posed}

    def _evaluate(self, rec, by_name: dict[str, ReconPhoto], gps: dict[str, np.ndarray],
                  label: str) -> _Candidate | None:
        """Geo-register one mapped model and check every camera against its own sensors."""
        inputs = georegister_inputs(rec, by_name, gps)
        if inputs is None:
            return None
        geo, cameras = georegister_model(inputs, self.settings.camera_height_m)
        errors = [abs(c["compass_offset_deg"]) for c in cameras.values() if c["compass_offset_deg"] is not None]
        return _Candidate(rec, inputs.images, geo, cameras, errors, label)

    @staticmethod
    def _cut_pairs(pycolmap, db_path: Path, group: set[str]) -> int:
        """Delete the verified pairs linking images in group to images outside it; returns how many."""
        db = pycolmap.Database.open(db_path)
        try:
            names = {im.image_id: im.name for im in db.read_all_images()}
            pair_ids, _ = db.read_two_view_geometries()
            cut = 0
            for pair_id in pair_ids:
                i1, i2 = pycolmap.pair_id_to_image_pair(pair_id)
                if (names[i1] in group) != (names[i2] in group):
                    db.delete_two_view_geometry(i1, i2)
                    cut += 1
            return cut
        finally:
            db.close()

    @staticmethod
    def _write_focal_priors(pycolmap, db_path: Path, by_name: dict[str, ReconPhoto]) -> None:
        """Focal prior from the 35 mm equivalent focal length, per COLMAP camera (ARCHITECTURE 3.5.1).

        COLMAP ignores FocalLengthIn35mmFilm unless FocalLength is present too.
        Cameras whose images do not all share one known 35 mm value are left
        to self-calibration. Width and height come from the database: the
        processed copy COLMAP read, not the original upload.
        """
        db = pycolmap.Database.open(db_path)
        try:
            groups: dict[int, list[str]] = {}
            for image in db.read_all_images():
                groups.setdefault(image.camera_id, []).append(image.name)
            for camera_id, names in groups.items():
                values = {by_name[n].focal_35mm if n in by_name else None for n in names}
                if len(values) != 1 or None in values:
                    continue
                camera = db.read_camera(camera_id)
                params = list(camera.params)
                params[camera.focal_length_idxs()[0]] = values.pop() / FULL_FRAME_DIAGONAL_MM * math.hypot(
                    camera.width, camera.height)
                camera.params = params
                camera.has_prior_focal_length = True
                db.update_camera(camera)
        finally:
            db.close()

    @staticmethod
    def _all_focal_priors(pycolmap, db_path: Path) -> bool:
        db = pycolmap.Database.open(db_path)
        try:
            return all(c.has_prior_focal_length for c in db.read_all_cameras())
        finally:
            db.close()

    @staticmethod
    def _write_gps_priors(pycolmap, db_path: Path, gps: dict[str, np.ndarray]):
        """Store GPS east/north as pose priors so spatial matching pairs nearby photos only."""
        db = pycolmap.Database.open(db_path)
        try:
            db.clear_pose_priors()
            for image in db.read_all_images():
                e, n = gps[image.name]
                db.write_pose_prior(pycolmap.PosePrior(
                    position=np.array([e, n, 0.0]),
                    coordinate_system=pycolmap.PosePriorCoordinateSystem.CARTESIAN,
                    corr_data_id=image.data_id,
                ))
        finally:
            db.close()
        pairing = pycolmap.SpatialPairingOptions()
        tun = params.current()
        pairing.max_num_neighbors = tun.spatial_max_neighbors
        pairing.max_distance = tun.spatial_max_distance_m   # metres: photos of the same side of a landmark
        return pairing


# name -> factory(settings). A new reconstructor (e.g. a feed-forward model) registers itself here and is
# then selectable through Settings.reconstructor / CROWD3D_RECONSTRUCTOR and evaluation configs.
RECONSTRUCTORS: dict[str, Callable[[Settings | None], Reconstructor]] = {
    FakeReconstructor.name: FakeReconstructor,
    ColmapReconstructor.name: ColmapReconstructor,
}


def register_reconstructor(name: str, factory: Callable[[Settings | None], Reconstructor]) -> None:
    """Make factory(settings) available as get_reconstructor(name); an existing name is not replaced."""
    if name in RECONSTRUCTORS and RECONSTRUCTORS[name] is not factory:
        raise ValueError(f"reconstructor {name!r} is already registered")
    RECONSTRUCTORS[name] = factory


def get_reconstructor(name: str, settings: Settings | None = None) -> Reconstructor:
    try:
        factory = RECONSTRUCTORS[name]
    except KeyError:
        raise ValueError(f"unknown reconstructor {name!r}; registered: {', '.join(sorted(RECONSTRUCTORS))}") from None
    return factory(settings)
