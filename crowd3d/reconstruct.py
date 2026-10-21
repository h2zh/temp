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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .config import Settings
from .geo import LocalENU, angle_diff_deg, heading_from_en

log = logging.getLogger(__name__)

MIN_REGISTERED = 3
# GPS: real phone fixes near buildings are ~10 m off (median, Florence Baptistry eval); 3 m was synthetic.
GPS_SIGMA_M = 6.0                     # 1 sigma per axis for a photo that reports no accuracy (ARCHITECTURE 3.5.1)
GPS_SIGMA_MIN_M = 2.0                 # floor for a reported GPSHPositioningError
GPS_INLIER_MIN_M = 10.0               # RANSAC inlier threshold is max(3 sigma, this) per camera
GPS_INLIER_M = 3.0 * GPS_SIGMA_M      # that threshold for a photo without reported accuracy
MAX_GPS_SCALE_ERROR = 0.35            # predicted relative scale error (= yaw error in rad) GPS may leave
COMPASS_SIGMA_DEG = 15.0              # phone compasses are poor next to steel and concrete
COMPASS_YAW_MIN_SIGMA_DEG = 3.0       # readings at one site share magnetic disturbances: averaging stops helping
# Without any compass, "every photo looks at the landmark centre" stands in for one (pseudo-compass).
PSEUDO_COMPASS_SIGMA_DEG = 25.0
PSEUDO_COMPASS_MIN_SIGMA_DEG = 10.0   # facade photos aim at parts of the building, not at its centre
MAX_YAW_CONFLICT_DEG = 40.0           # GPS yaw this far from the compass (or pseudo-compass) yaw: GPS is not trusted
UP_PRIOR_WEIGHT = 0.01                # pull of the (pitch-biased) mean camera up in estimate_up
UP_MIN_EIGEN_GAP = 0.02               # estimate_up falls back to the mean up below this (unobservable tilt)
UP_COMMON_HEIGHT_FRACTION = 0.25      # cameras this far (x horizontal spread) off the median height are elevated
EXHAUSTIVE_MAX_IMAGES = 60            # larger sets use GPS-guided spatial matching
FOCAL_REFINE_MIN_IMAGES = 12          # smaller sets keep the EXIF focal length fixed
FULL_FRAME_DIAGONAL_MM = 43.27        # COLMAP's focal prior: f_px = FocalLengthIn35mmFilm / 43.27 * image diagonal
# Per-camera sanity checks against the photo's own sensors (GT-free).
GROSS_COMPASS_DEG = 3 * COMPASS_SIGMA_DEG   # a camera this far off its compass is wrong, not noisy
FAR_FROM_GPS_FACTOR = 3.0             # this many GPS inlier thresholds from its own fix: wrong depth or facade
FOCAL_RATIO_RANGE = (0.25, 12.0)      # focal / longer image side outside this: a degenerate camera
# Incremental mapping depends on the seed and on which pair it starts from; on real photos one try in
# three can come out broken (symmetric facades, weak views), so suspicious results are mapped again.
MAPPING_ATTEMPTS = 3                  # sets of at most SMALL_SET_IMAGES always get all of them (mapping is cheap)
MAPPING_ATTEMPTS_LARGE = 2            # larger sets only retry a suspicious result
SMALL_SET_IMAGES = 30
MIN_CANDIDATE_POINTS = 100            # a mapped model with fewer points is degenerate (Aachen: 4 cameras, 1 point)
RETRY_FLAGGED_FRACTION = 0.05         # retry when more cameras than this disagree with their own sensors ...
RETRY_COMPASS_ERROR_DEG = 20.0        # ... or the mean compass disagreement (each capped at 90 deg) exceeds this


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

    def to_meta(self) -> dict[str, Any]:
        d = asdict(self)
        d["ply_path"] = str(self.ply_path) if self.ply_path else None
        d["frame"] = "ENU"
        return d


class Reconstructor(Protocol):
    name: str

    def reconstruct(
        self,
        photos: list[ReconPhoto],
        landmark_lat: float,
        landmark_lon: float,
        out_ply: Path,
        work_dir: Path,
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

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir) -> ReconstructionResult:
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
    unobservable (two smallest eigenvalues within UP_MIN_EIGEN_GAP, e.g. a
    straight line of cameras all facing the same way) is the mean up returned.
    """
    ups = np.asarray(ups, dtype=float)
    rights = np.asarray(rights, dtype=float)
    mean_up = ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    m_rights = rights.T @ rights / len(rights)

    def solve(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        w, v = np.linalg.eigh(m - UP_PRIOR_WEIGHT * np.outer(mean_up, mean_up))
        return (v[:, 0] if v[:, 0] @ mean_up >= 0 else -v[:, 0]), w

    up, w = solve(m_rights)
    if centers is not None and len(centers) >= 2:
        c = np.asarray(centers, dtype=float)
        h = c @ up
        flat = c - np.outer(h, up)
        d = math.sqrt(float(((flat - flat.mean(axis=0)) ** 2).sum(axis=1).mean()))
        level = np.abs(h - np.median(h)) <= UP_COMMON_HEIGHT_FRACTION * d
        if level.sum() >= 2:
            ck = c[level] - c[level].mean(axis=0)
            spread = ck.T @ ck
            if np.trace(spread) > 0:
                up, w = solve(m_rights + spread / np.trace(spread))
    if w[1] - w[0] < UP_MIN_EIGEN_GAP:
        return mean_up
    return up


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
    view_dirs: np.ndarray, headings_deg: list[float | None], sigma_deg: float = COMPASS_SIGMA_DEG
) -> tuple[float, int] | None:
    """Yaw (rad, counter-clockwise) turning gravity-aligned viewing directions onto compass headings.

    Robust circular mean: headings more than 3 * sigma_deg off the plain mean
    are dropped. Returns (yaw_rad, number of headings used) or None.
    """
    # Headings turn clockwise and yaw counter-clockwise, so yaw = SfM heading - compass heading.
    diffs = np.radians([
        heading_from_en(d[0], d[1]) - h for d, h in zip(np.asarray(view_dirs, dtype=float), headings_deg)
        if h is not None
    ])
    if len(diffs) == 0:
        return None
    mean = math.atan2(np.sin(diffs).sum(), np.cos(diffs).sum())
    used = np.abs(np.angle(np.exp(1j * (diffs - mean)))) <= math.radians(3 * sigma_deg)
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
    compass = compass_yaw(view_dirs, headings_deg)
    if compass:
        sigma = max(math.radians(COMPASS_SIGMA_DEG) / math.sqrt(compass[1]), math.radians(COMPASS_YAW_MIN_SIGMA_DEG))
        return compass[0], sigma, "compass"
    towards = [heading_from_en(-e, -n) if math.hypot(e, n) > 1.0 else None for e, n in np.asarray(gps_en, float)]
    pseudo = compass_yaw(view_dirs, towards, sigma_deg=PSEUDO_COMPASS_SIGMA_DEG)
    if pseudo:
        sigma = max(math.radians(PSEUDO_COMPASS_SIGMA_DEG) / math.sqrt(pseudo[1]),
                    math.radians(PSEUDO_COMPASS_MIN_SIGMA_DEG))
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
       GPS_SIGMA_M when None or NaN); a fix is an inlier within
       max(3 sigma, GPS_INLIER_MIN_M).
    3. Yaw: the GPS yaw is fused with yaw_prior (compass, else the
       pseudo-compass) by inverse variance, which matters for a single facade,
       where GPS noise over a 20 m arc can turn the model by tens of degrees.
       When the two disagree by more than MAX_YAW_CONFLICT_DEG the GPS fit is
       not trusted at all: fusing across more than 90 degrees would make the
       least-squares scale negative, i.e. mirror the model.
    4. z offset so cameras sit at camera_height_m.
    Without a usable, consistent GPS fit (too few inliers, RMS spread below
    the GPS sigma, a predicted scale error above MAX_GPS_SCALE_ERROR, or the
    yaw conflict) the yaw comes from the prior alone, the scale from
    _fallback_scale (the weak GPS fit if that has nothing), the position from
    the median GPS offset, and georegistered is False. The scale is always
    positive, so the transform never mirrors.
    """
    centers = np.asarray(centers, dtype=float)
    gps_en = np.asarray(gps_en, dtype=float)
    n = len(centers)
    sigma = np.broadcast_to(np.asarray(GPS_SIGMA_M if gps_sigma_m is None else gps_sigma_m, dtype=float), (n,))
    sigma = np.maximum(np.where(np.isfinite(sigma), sigma, GPS_SIGMA_M), GPS_SIGMA_MIN_M)
    weights = 1.0 / sigma**2
    thresholds = np.maximum(3.0 * sigma, GPS_INLIER_MIN_M)
    rot_g = rotation_to_z(estimate_up(ups, rights, centers))
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
              and spread >= float(np.median(sigma[inliers])) and sigma_gps <= MAX_GPS_SCALE_ERROR)
    gps_yaw = fit[1] if fit else None
    prior_note = ""
    if prior and gps_yaw is not None:
        prior_note = (f"; GPS yaw - {prior[2]} yaw ="
                      f" {angle_diff_deg(math.degrees(gps_yaw), math.degrees(prior[0])):.0f} deg")

    georegistered, conflict = False, False
    if usable:
        yaw, yaw_sigma, source = gps_yaw, sigma_gps, "gps"
        if prior:
            delta = math.radians(angle_diff_deg(math.degrees(prior[0]), math.degrees(gps_yaw)))
            conflict = abs(delta) > math.radians(MAX_YAW_CONFLICT_DEG)
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
            yaw, source = prior[0], prior[2]
        else:
            yaw, source = (gps_yaw, "gps") if gps_yaw is not None else (0.0, "none")
        scale = _fallback_scale(c, gps_en, scene_points, rot_g) or (fit[0] if fit else 1.0)
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
    transform = similarity_matrix(scale, rotation, np.array([shift.real, shift.imag, z_shift]))
    resid = apply_similarity(transform, centers)[:, :2] - gps_en
    rmse = float(np.sqrt((resid[inliers] ** 2).sum(axis=1).mean())) if inliers.any() else None
    return Georegistration(
        transform, bool(georegistered), inliers, rmse, message, yaw_source=source,
        gps_yaw_deg=None if gps_yaw is None else math.degrees(gps_yaw),
        prior_yaw_deg=None if prior is None else math.degrees(prior[0]),
    )


def gps_inlier_threshold_m(gps_accuracy_m: float | None) -> float:
    """georegister's inlier threshold for one photo: max(3 sigma, GPS_INLIER_MIN_M)."""
    sigma = GPS_SIGMA_M if gps_accuracy_m is None or not math.isfinite(gps_accuracy_m) else gps_accuracy_m
    return max(3.0 * max(sigma, GPS_SIGMA_MIN_M), GPS_INLIER_MIN_M)


def camera_flags(
    gps_residual_m: float,
    inlier_threshold_m: float,
    compass_offset_deg: float | None,
    focal_ratio: float | None,
) -> list[str]:
    """Why one registered camera's pose cannot be right (empty: no evidence against it).

    - "gps_and_compass": beyond the GPS inlier threshold from its own fix AND
      more than GROSS_COMPASS_DEG off its own compass. Typical of a symmetric
      building, where SfM puts the photo in front of a twin facade.
    - "far_from_gps": more than FAR_FROM_GPS_FACTOR thresholds from its own fix
      (wrong depth for a telephoto view, a camera off in a fragment).
    - "odd_focal": focal length outside FOCAL_RATIO_RANGE x the longer side.
    """
    flags = []
    if (gps_residual_m > inlier_threshold_m and compass_offset_deg is not None
            and abs(compass_offset_deg) > GROSS_COMPASS_DEG):
        flags.append("gps_and_compass")
    if gps_residual_m > FAR_FROM_GPS_FACTOR * inlier_threshold_m:
        flags.append("far_from_gps")
    if focal_ratio is not None and not FOCAL_RATIO_RANGE[0] <= focal_ratio <= FOCAL_RATIO_RANGE[1]:
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
    for pid, pos, v, g, h, acc, f in zip(photo_ids, cam_enu, view_enu, np.asarray(gps_en, float), headings_deg,
                                         gps_accuracy_m, focal_ratios):
        heading = heading_from_en(v[0], v[1])
        offset = None if h is None else angle_diff_deg(heading, h)
        resid = float(np.linalg.norm(pos[:2] - g))
        cameras[pid] = {
            "enu": [round(float(x), 3) for x in pos],
            "heading_deg": round(heading, 2),
            "gps_residual_m": round(resid, 2),
            "compass_offset_deg": None if offset is None else round(offset, 2),
            "focal_ratio": None if f is None else round(float(f), 4),
            "flags": camera_flags(resid, gps_inlier_threshold_m(acc), offset, None if f is None else float(f)),
        }
    return cameras


def image_name(index: int, photo_id: str) -> str:
    """SfM image name: COLMAP numbers images in name order, so the upload order fixes its choices."""
    return f"{index:05d}_{photo_id}.jpg"


def photo_id_from_image_name(name: str) -> str:
    return Path(name).stem.split("_", 1)[-1]


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
        return (self.rec.num_points3D() >= MIN_CANDIDATE_POINTS, len(self.images) - self.num_flagged,
                -(self.compass_error_deg or 0.0), len(self.images))

    def suspicious(self) -> bool:
        return (self.num_flagged > RETRY_FLAGGED_FRACTION * len(self.images)
                or (self.compass_error_deg or 0.0) > RETRY_COMPASS_ERROR_DEG)


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
       EXHAUSTIVE_MAX_IMAGES).
    3. Incremental mapping (EXIF focal length kept fixed for small sets). Every
       model is geo-registered (georegister, without degenerate cameras) and
       each camera is checked against its own GPS and compass (camera_flags).
       Small sets are mapped with MAPPING_ATTEMPTS seeds, larger ones again
       when the result is suspicious; when several cameras disagree with both
       GPS and compass (a symmetric building's twin facades), the verified
       pairs linking them to the rest are cut and the set is mapped again.
       The model with most unflagged cameras wins and is written to
       work_dir/sparse/0.
    4. Per registered camera: ENU position, heading from its viewing direction
       and the check results; heading_mae_deg vs EXIF headings when present.
    5. Colours from the images, PLY (in ENU) to out_ply.
    Data problems give ok=False with a message; only programming errors raise.
    """

    name = "colmap"

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self.exhaustive_max_images = EXHAUSTIVE_MAX_IMAGES

    def reconstruct(self, photos, landmark_lat, landmark_lon, out_ply, work_dir) -> ReconstructionResult:
        try:
            import pycolmap
        except ImportError:
            return ReconstructionResult(ok=False, message="pycolmap is not installed", num_input=len(photos))
        out_ply, work_dir = Path(out_ply), Path(work_dir)
        timings: dict[str, float] = {}

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
        try:
            t = time.monotonic()
            pycolmap.Database.open(db_path).close()
            pycolmap.import_images(db_path, images_dir, camera_mode=pycolmap.CameraMode.AUTO, image_names=names)
            extraction = pycolmap.FeatureExtractionOptions()
            extraction.max_image_size = self.settings.processing_max_side_px
            pycolmap.extract_features(db_path, images_dir, image_names=names,
                                      camera_mode=pycolmap.CameraMode.AUTO,
                                      extraction_options=extraction, device=pycolmap.Device.cpu)
            self._write_focal_priors(pycolmap, db_path, by_name)
            lap("extract", t)

            t = time.monotonic()
            verification = pycolmap.TwoViewGeometryOptions()
            verification.ransac.random_seed = 0
            if len(by_name) > self.exhaustive_max_images:
                pairing = self._write_gps_priors(pycolmap, db_path, gps)
                pycolmap.match_spatial(db_path, pairing_options=pairing, verification_options=verification,
                                       device=pycolmap.Device.cpu)
            else:
                pycolmap.match_exhaustive(db_path, verification_options=verification, device=pycolmap.Device.cpu)
            lap("match", t)
            fix_focal = len(by_name) < FOCAL_REFINE_MIN_IMAGES and self._all_focal_priors(pycolmap, db_path)

            candidates: list[_Candidate] = []

            def attempt(label: str, seed: int, database: Path) -> _Candidate | None:
                t = time.monotonic()
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

            max_attempts = MAPPING_ATTEMPTS if len(by_name) <= SMALL_SET_IMAGES else MAPPING_ATTEMPTS_LARGE
            current = attempt("seed0", 0, db_path)
            tried, seed, cut_done = 1, 1, False
            while tried < max_attempts and (current is None or current.suspicious() or len(by_name) <= SMALL_SET_IMAGES):
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
                   + (f"; best of {len(candidates)} mapping attempts ({best.attempt})" if len(candidates) > 1 else ""))
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
        )

    def _evaluate(self, rec, by_name: dict[str, ReconPhoto], gps: dict[str, np.ndarray],
                  label: str) -> _Candidate | None:
        """Geo-register one mapped model and check every camera against its own sensors."""
        images = [im for im in rec.images.values() if im.has_pose and im.name in by_name]
        if len(images) < MIN_REGISTERED:
            return None
        photos = [by_name[im.name] for im in images]
        rots = np.array([im.cam_from_world().rotation.matrix() for im in images])
        centers = np.array([im.projection_center() for im in images])
        focal = np.array([rec.cameras[im.camera_id].mean_focal_length()
                          / max(rec.cameras[im.camera_id].width, rec.cameras[im.camera_id].height) for im in images])
        gps_en = np.array([gps[im.name] for im in images])
        accuracy = np.array([np.nan if p.gps_accuracy_m is None else p.gps_accuracy_m for p in photos])
        # Degenerate cameras (absurd focal length) must not steer the alignment of the others.
        sane = (focal >= FOCAL_RATIO_RANGE[0]) & (focal <= FOCAL_RATIO_RANGE[1])
        use = sane if sane.sum() >= MIN_REGISTERED else np.ones(len(images), dtype=bool)
        xyz = np.array([pt.xyz for pt in rec.points3D.values()]).reshape(-1, 3)
        geo = georegister(
            centers[use],
            ups=-rots[use, 1, :],          # image -y axis in world
            rights=rots[use, 0, :],
            view_dirs=rots[use, 2, :],
            gps_en=gps_en[use],
            headings_deg=[p.heading_deg for p, u in zip(photos, use) if u],
            camera_height_m=self.settings.camera_height_m,
            scene_points=xyz,
            gps_sigma_m=accuracy[use],
        )
        cameras = check_cameras(geo.transform, rots, centers, gps_en, [p.heading_deg for p in photos],
                                [p.gps_accuracy_m for p in photos], focal, [p.photo_id for p in photos])
        errors = [abs(c["compass_offset_deg"]) for c in cameras.values() if c["compass_offset_deg"] is not None]
        return _Candidate(rec, images, geo, cameras, errors, label)

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
        pairing.max_num_neighbors = 30
        pairing.max_distance = 80.0   # metres: photos of the same side of a landmark
        return pairing


def get_reconstructor(name: str, settings: Settings | None = None) -> Reconstructor:
    if name == "fake":
        return FakeReconstructor(settings)
    if name == "colmap":
        return ColmapReconstructor(settings)
    raise ValueError(f"unknown reconstructor {name!r}")
