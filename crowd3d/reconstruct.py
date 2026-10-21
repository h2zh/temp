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
GPS_SIGMA_M = 3.0                     # phone GPS horizontal error, 1 sigma per axis
GPS_INLIER_M = 3.0 * GPS_SIGMA_M      # RANSAC inlier threshold for camera-centre vs GPS residuals
MAX_GPS_SCALE_ERROR = 0.35            # predicted relative scale error (= yaw error in rad) GPS may leave
COMPASS_SIGMA_DEG = 15.0              # phone compasses are poor next to steel and concrete
UP_PRIOR_WEIGHT = 0.01                # pull of the (pitch-biased) mean camera up in estimate_up
MAX_UP_TILT_DEG = 30.0                # estimate_up falls back to the mean up beyond this
EXHAUSTIVE_MAX_IMAGES = 60            # larger sets use GPS-guided spatial matching
FOCAL_REFINE_MIN_IMAGES = 12          # smaller sets keep the EXIF focal length fixed


@dataclass
class ReconPhoto:
    photo_id: str
    image_path: Path             # processed copy (orientation applied, downscaled, EXIF focal kept)
    lat: float
    lon: float
    heading_deg: float | None


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
    # photo_id -> {"enu": [x, y, z], "heading_deg": float}; registered photos only
    cameras: dict[str, dict[str, Any]] = field(default_factory=dict)
    timings_s: dict[str, float] = field(default_factory=dict)   # stage -> seconds, for diagnostics

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


class FakeReconstructor:
    """Instant stand-in: trusts GPS and compass, emits a ring of points plus the cameras."""

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
            cams[p.photo_id] = {"enu": [e, n, self.settings.camera_height_m], "heading_deg": p.heading_deg}
            points.append((e, n, self.settings.camera_height_m))
            colors.append((255, 64, 64))
        for i in range(72):
            a = 2 * math.pi * i / 72
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
        )


# ---------------------------------------------------------------- geo-registration (pure numpy)


def estimate_up(ups: np.ndarray, rights: np.ndarray, centers: np.ndarray | None = None) -> np.ndarray:
    """World "up" in the SfM frame (unit vector) from camera orientations and centres.

    The mean camera up vector leans back by the typical upward pitch (people
    tilt phones up at buildings), about 10 degrees when all photos show one
    facade. So take the direction most perpendicular to the camera right
    vectors (phones are held without roll) and to the spread of the camera
    centres (photographers stand at about the same height), with a light pull
    toward the mean up that also fixes the sign and settles degenerate layouts
    such as a straight line of cameras all facing the same way.
    """
    ups = np.asarray(ups, dtype=float)
    rights = np.asarray(rights, dtype=float)
    mean_up = ups.mean(axis=0)
    mean_up /= np.linalg.norm(mean_up)
    m = rights.T @ rights / len(rights)
    if centers is not None and len(centers) >= 2:
        c = np.asarray(centers, dtype=float)
        c = c - c.mean(axis=0)
        spread = c.T @ c
        if np.trace(spread) > 0:
            m = m + spread / np.trace(spread)
    m = m - UP_PRIOR_WEIGHT * np.outer(mean_up, mean_up)
    up = np.linalg.eigh(m)[1][:, 0]
    up = up if up @ mean_up >= 0 else -up
    if math.degrees(math.acos(min(1.0, float(up @ mean_up)))) > MAX_UP_TILT_DEG:
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


def fit_similarity_2d(src: np.ndarray, dst: np.ndarray) -> tuple[float, float, np.ndarray]:
    """Least-squares (Umeyama) similarity dst ~ scale * R(yaw) @ src + t without reflection.

    Returns (scale, yaw_rad, t) with yaw counter-clockwise. In complex form the
    model is w = a z + b, whose least-squares a is closed-form.
    """
    z, w = _complex(src), _complex(dst)
    zc, wc = z - z.mean(), w - w.mean()
    denom = float((np.abs(zc) ** 2).sum())
    a = complex(np.conj(zc) @ wc) / denom if denom > 0 else 1.0 + 0j
    b = w.mean() - a * z.mean()
    return abs(a), math.atan2(a.imag, a.real), np.array([b.real, b.imag])


def robust_similarity_2d(
    src: np.ndarray,
    dst: np.ndarray,
    inlier_threshold: float,
    min_pair_distance: float = 0.0,
    max_pairs: int = 2000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray, np.ndarray] | None:
    """RANSAC over point pairs, then least squares on the inliers.

    Each pair whose dst points are at least min_pair_distance apart fixes a
    similarity; the one with most inliers (residual < inlier_threshold, ties
    broken by truncated squared error) wins and is refined twice by
    fit_similarity_2d on its inliers. Pairs are enumerated exhaustively up to
    max_pairs, else sampled with the given seed. Returns (scale, yaw_rad, t,
    inlier mask), or None when no usable pair exists.
    """
    z, w = _complex(src), _complex(dst)
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
    thr2 = inlier_threshold**2
    count = (sq < thr2).sum(axis=1)
    cost = np.minimum(sq, thr2).sum(axis=1)
    best = int(np.lexsort((cost, -count))[0])
    inliers = sq[best] < thr2
    for _ in range(2):
        if inliers.sum() < 2:
            break
        scale, yaw, t = fit_similarity_2d(np.asarray(src)[inliers], np.asarray(dst)[inliers])
        resid = np.abs(scale * np.exp(1j * yaw) * z + complex(*t) - w)
        inliers = resid < inlier_threshold
    if inliers.sum() < 2:
        return None
    scale, yaw, t = fit_similarity_2d(np.asarray(src)[inliers], np.asarray(dst)[inliers])
    return scale, yaw, t, inliers


def compass_yaw(view_dirs: np.ndarray, headings_deg: list[float | None]) -> tuple[float, int] | None:
    """Yaw (rad, counter-clockwise) turning gravity-aligned viewing directions onto compass headings.

    Robust circular mean: headings more than 3 * COMPASS_SIGMA_DEG off the
    plain mean are dropped. Returns (yaw_rad, number of headings used) or None.
    """
    # Headings turn clockwise and yaw counter-clockwise, so yaw = SfM heading - compass heading.
    diffs = np.radians([
        heading_from_en(d[0], d[1]) - h for d, h in zip(np.asarray(view_dirs, dtype=float), headings_deg)
        if h is not None
    ])
    if len(diffs) == 0:
        return None
    mean = math.atan2(np.sin(diffs).sum(), np.cos(diffs).sum())
    used = np.abs(np.angle(np.exp(1j * (diffs - mean)))) <= math.radians(3 * COMPASS_SIGMA_DEG)
    if used.any():
        mean = math.atan2(np.sin(diffs[used]).sum(), np.cos(diffs[used]).sum())
    else:
        used[:] = True
    return mean, int(used.sum())


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
    transform: np.ndarray        # 4x4 similarity: SfM world -> ENU
    georegistered: bool          # scale and position come from GPS (else compass yaw + a scale guess)
    inliers: np.ndarray          # per camera: GPS fix consistent with the fit
    rmse_m: float | None         # horizontal camera-centre vs GPS residual over the inliers
    message: str


def georegister(
    centers: np.ndarray,
    ups: np.ndarray,
    rights: np.ndarray,
    view_dirs: np.ndarray,
    gps_en: np.ndarray,
    headings_deg: list[float | None],
    camera_height_m: float,
    scene_points: np.ndarray | None = None,
) -> Georegistration:
    """Similarity from the SfM frame to ENU (landmark centre at the origin).

    1. Gravity: rotate estimate_up onto +z. A full 3-D Sim3 against GPS would
       have to recover tilt from the camera centres alone, which is
       ill-conditioned when they lie on a short arc at equal height.
    2. Robust 2-D similarity of camera centres' x/y onto GPS east/north. Its
       yaw is fused with the compass (inverse-variance weights), which matters
       for a single facade, where a few metres of GPS noise over a 20 m arc
       can turn the model by 10 degrees or more.
    3. z offset so cameras sit at camera_height_m.
    Without a usable GPS baseline (too few inliers, RMS spread below
    GPS_SIGMA_M, or a predicted scale error above MAX_GPS_SCALE_ERROR) the
    yaw comes from the compass alone, the scale from comparing camera-to-scene
    distances with GPS camera-to-landmark distances (the weak GPS fit if no
    scene points are given), and georegistered is False.
    """
    centers = np.asarray(centers, dtype=float)
    gps_en = np.asarray(gps_en, dtype=float)
    n = len(centers)
    rot_g = rotation_to_z(estimate_up(ups, rights, centers))
    c = centers @ rot_g.T
    d = np.asarray(view_dirs, dtype=float) @ rot_g.T
    fit = robust_similarity_2d(c[:, :2], gps_en, GPS_INLIER_M, min_pair_distance=GPS_SIGMA_M)
    compass = compass_yaw(d, headings_deg)
    inliers = fit[3] if fit else np.zeros(n, dtype=bool)
    spread, sigma_gps = 0.0, math.inf
    if inliers.sum() >= 2:
        g = gps_en[inliers]
        sq = float(((g - g.mean(axis=0)) ** 2).sum())
        spread = math.sqrt(sq / len(g))
        # Least squares w = a z + b: std(a) / |a| = sigma / sqrt(sum |w - mean w|^2), for scale and yaw alike.
        sigma_gps = GPS_SIGMA_M / max(math.sqrt(sq), 1e-9)
    usable = (inliers.sum() >= max(MIN_REGISTERED, math.ceil(n / 2))
              and spread >= GPS_SIGMA_M and sigma_gps <= MAX_GPS_SCALE_ERROR)

    if usable:
        scale, yaw, _, _ = fit
        if compass:
            sigma_compass = math.radians(COMPASS_SIGMA_DEG) / math.sqrt(compass[1])
            weight = sigma_gps**2 / (sigma_gps**2 + sigma_compass**2)
            yaw += weight * math.radians(angle_diff_deg(math.degrees(compass[0]), math.degrees(yaw)))
        # Least-squares scale and translation for the fused yaw.
        u = np.exp(1j * yaw) * _complex(c[inliers, :2])
        w = _complex(gps_en[inliers])
        uc, wc = u - u.mean(), w - w.mean()
        scale = float((np.conj(uc) @ wc).real / (np.abs(uc) ** 2).sum())
        shift = w.mean() - scale * u.mean()
        message = (f"georegistered from {int(inliers.sum())}/{n} GPS fixes (spread {spread:.1f} m,"
                   f" scale +-{100 * sigma_gps:.0f}%, yaw +-{math.degrees(sigma_gps):.0f} deg from GPS)")
    else:
        yaw = compass[0] if compass else (fit[1] if fit else 0.0)
        scale = fit[0] if fit else 1.0
        if scene_points is not None and len(scene_points):
            target = np.median(np.asarray(scene_points, dtype=float) @ rot_g.T, axis=0)
            sfm_dist = float(np.median(np.linalg.norm(c[:, :2] - target[:2], axis=1)))
            gps_dist = float(np.median(np.linalg.norm(gps_en, axis=1)))
            if sfm_dist > 0 and gps_dist > 0:
                scale = gps_dist / sfm_dist
        u = np.exp(1j * yaw) * _complex(c[:, :2])
        shift = _complex(gps_en).mean() - scale * u.mean()
        message = (f"not georegistered: {int(inliers.sum())}/{n} consistent GPS fixes spread over {spread:.1f} m"
                   f" cannot fix the scale; {'compass' if compass else 'GPS'} yaw, approximate scale")
        inliers = np.ones(n, dtype=bool)

    rotation = rotation_z(yaw) @ rot_g
    z_shift = camera_height_m - scale * float(np.median(c[:, 2]))
    transform = similarity_matrix(scale, rotation, np.array([shift.real, shift.imag, z_shift]))
    resid = apply_similarity(transform, centers)[:, :2] - gps_en
    rmse = float(np.sqrt((resid[inliers] ** 2).sum(axis=1).mean())) if inliers.any() else None
    return Georegistration(transform, bool(usable), inliers, rmse, message)


# ---------------------------------------------------------------- COLMAP


class ColmapReconstructor:
    """Incremental SfM with pycolmap (CPU), then gravity + GPS geo-registration.

    1. Symlink (or copy) processed images into work_dir/images as <photo_id>.jpg.
    2. pycolmap.extract_features (CPU; max image size from settings), exhaustive
       matching (GPS-guided spatial matching above EXHAUSTIVE_MAX_IMAGES),
       incremental_mapping (EXIF focal length kept fixed for small sets);
       keep the reconstruction with most registered images.
    3. georegister() into ENU at the landmark centre.
    4. Per registered camera: ENU position and heading from its viewing
       direction; heading_mae_deg vs EXIF headings when present.
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
            timings[stage] = round(time.monotonic() - start, 2)

        present = [p for p in photos if Path(p.image_path).is_file()]
        if len(present) < MIN_REGISTERED:
            return fail(f"need at least {MIN_REGISTERED} photo files, found {len(present)} of {len(photos)}")
        by_name = {f"{p.photo_id}.jpg": p for p in present}
        enu = LocalENU(landmark_lat, landmark_lon)
        gps = {name: np.array(enu.to_enu(p.lat, p.lon)[:2]) for name, p in by_name.items()}

        images_dir, sparse_dir, db_path = work_dir / "images", work_dir / "sparse", work_dir / "database.db"
        for d in (images_dir, sparse_dir):
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
        try:
            t = time.monotonic()
            extraction = pycolmap.FeatureExtractionOptions()
            extraction.max_image_size = self.settings.processing_max_side_px
            pycolmap.extract_features(db_path, images_dir, image_names=sorted(by_name),
                                      camera_mode=pycolmap.CameraMode.AUTO,
                                      extraction_options=extraction, device=pycolmap.Device.cpu)
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

            t = time.monotonic()
            options = pycolmap.IncrementalPipelineOptions()
            options.random_seed = 0
            options.min_model_size = MIN_REGISTERED
            options.extract_colors = True
            if len(by_name) < FOCAL_REFINE_MIN_IMAGES and self._all_focal_priors(pycolmap, db_path):
                # A few views of one near-planar facade cannot pin down focal length:
                # refining it shrinks or stretches the model, so trust EXIF instead.
                options.ba_refine_focal_length = False
                options.mapper.abs_pose_refine_focal_length = False
            models = pycolmap.incremental_mapping(db_path, images_dir, sparse_dir, options=options)
            lap("mapping", t)
        except RuntimeError as e:   # COLMAP reports failed checks on bad input as RuntimeError
            return fail(f"COLMAP failed: {e}")

        if not models:
            return fail("structure-from-motion found no initial image pair; photos may not overlap enough")
        rec = max(models.values(), key=lambda r: r.num_reg_images())
        images = [im for im in rec.images.values() if im.has_pose and im.name in by_name]
        if len(images) < MIN_REGISTERED:
            return fail(f"only {len(images)} of {len(by_name)} photos could be registered; "
                        f"need at least {MIN_REGISTERED} overlapping views", num_registered=len(images))

        t = time.monotonic()
        rots = np.array([im.cam_from_world().rotation.matrix() for im in images])
        centers = np.array([im.projection_center() for im in images])
        recon_photos = [by_name[im.name] for im in images]
        headings = [p.heading_deg for p in recon_photos]
        points3d = list(rec.points3D.values())
        xyz = np.array([pt.xyz for pt in points3d]).reshape(-1, 3)
        rgb = np.array([pt.color for pt in points3d], dtype=np.uint8).reshape(-1, 3)
        geo = georegister(
            centers,
            ups=-rots[:, 1, :],          # image -y axis in world
            rights=rots[:, 0, :],
            view_dirs=rots[:, 2, :],
            gps_en=np.array([gps[im.name] for im in images]),
            headings_deg=headings,
            camera_height_m=self.settings.camera_height_m,
            scene_points=xyz,
        )
        cam_enu = apply_similarity(geo.transform, centers)
        view_enu = rots[:, 2, :] @ geo.transform[:3, :3].T
        points = apply_similarity(geo.transform, xyz)

        cameras: dict[str, dict[str, Any]] = {}
        errors = []
        for p, pos, v in zip(recon_photos, cam_enu, view_enu):
            heading = heading_from_en(v[0], v[1])
            cameras[p.photo_id] = {"enu": [round(float(x), 3) for x in pos], "heading_deg": round(heading, 2)}
            if p.heading_deg is not None:
                errors.append(abs(angle_diff_deg(heading, p.heading_deg)))
        write_ply(out_ply, points.tolist(), rgb.tolist())
        lap("georegister_export", t)

        message = (f"registered {len(images)}/{len(photos)} photos, {len(points)} points; {geo.message}"
                   + (f"; {len(models)} separate models, kept the largest" if len(models) > 1 else ""))
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
            heading_mae_deg=float(np.mean(errors)) if errors else None,
            cameras=cameras,
            timings_s=timings,
        )

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
