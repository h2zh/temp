"""Geo-registration helpers on synthetic camera sets (pure numpy, no COLMAP)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from crowd3d import params
from crowd3d.geo import angle_diff_deg, heading_from_en
from crowd3d.reconstruct import (
    GPS_INLIER_M,
    apply_similarity,
    camera_flags,
    check_cameras,
    compass_yaw,
    estimate_up,
    fit_similarity_2d,
    georegister,
    robust_similarity_2d,
    rotation_to_z,
    rotation_z,
    similarity_matrix,
)

CAMERA_HEIGHT = 1.6


def angle_between_deg(a: np.ndarray, b: np.ndarray) -> float:
    c = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def cam_from_enu(heading_deg: float, pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """COLMAP convention rows: image x (right), image y (down), viewing direction."""
    h, p, r = (math.radians(a) for a in (heading_deg, pitch_deg, roll_deg))
    fwd = np.array([math.sin(h) * math.cos(p), math.cos(h) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(h), -math.sin(h), 0.0])
    up = np.cross(right, fwd)
    right, up = math.cos(r) * right + math.sin(r) * up, math.cos(r) * up - math.sin(r) * right
    return np.array([right, -up, fwd])


class Scene:
    """True ENU cameras plus their image in an arbitrary SfM frame (random similarity)."""

    def __init__(self, centers, headings, pitches, rolls, rng, gps_noise=2.0, compass_noise=8.0):
        self.centers = np.asarray(centers, float)
        self.headings = np.asarray(headings, float)
        rots = np.array([cam_from_enu(h, p, r) for h, p, r in zip(headings, pitches, rolls)])
        self.r0 = random_rotation(rng)
        self.s0 = float(rng.uniform(0.05, 20.0))
        self.t0 = rng.normal(0, 50, 3)
        self.sfm_centers = self.to_sfm(self.centers)
        self.sfm_rots = rots @ self.r0.T
        self.gps = self.centers[:, :2] + rng.normal(0, gps_noise, (len(self.centers), 2)) if gps_noise else \
            self.centers[:, :2].copy()
        self.gps_sigma = gps_noise      # the accuracy each photo reports (EXIF GPSHPositioningError)
        self.compass = [float(h + rng.normal(0, compass_noise)) for h in headings]

    def to_sfm(self, points: np.ndarray) -> np.ndarray:
        return self.s0 * np.asarray(points, float) @ self.r0.T + self.t0

    def georegister(self, scene_points=None, compass=None, gps_sigma=None):
        r = self.sfm_rots
        return georegister(self.sfm_centers, -r[:, 1, :], r[:, 0, :], r[:, 2, :], self.gps,
                           self.compass if compass is None else compass, CAMERA_HEIGHT,
                           scene_points=None if scene_points is None else self.to_sfm(scene_points),
                           gps_sigma_m=self.gps_sigma if gps_sigma is None else gps_sigma)

    def estimated_headings(self, transform: np.ndarray) -> np.ndarray:
        views = self.sfm_rots[:, 2, :] @ transform[:3, :3].T
        return np.array([heading_from_en(v[0], v[1]) for v in views])

    def heading_errors(self, transform: np.ndarray) -> np.ndarray:
        return np.array([abs(angle_diff_deg(e, t)) for e, t in zip(self.estimated_headings(transform), self.headings)])


def ring_layout(bearings_deg, rng, radius=(28.0, 40.0), height_sd=0.0):
    b = np.radians(np.asarray(bearings_deg, float))
    r = rng.uniform(*radius, len(b))
    centers = np.c_[r * np.sin(b), r * np.cos(b), CAMERA_HEIGHT + rng.normal(0, height_sd, len(b)) if height_sd
                    else np.full(len(b), CAMERA_HEIGHT)]
    headings = [heading_from_en(-x, -y) + rng.uniform(-5, 5) for x, y, _ in centers]
    return centers, headings


def sector_scene(rng, n=4, half_width_deg=18.5, pitch=(5.0, 12.0), roll_sd=0.0, height_sd=0.0, **kw) -> Scene:
    """n cameras spread over one sector like crowd photos: one per equal slot, random within it."""
    slots = (np.arange(n) + rng.uniform(0.2, 0.8, n)) / n
    centers, headings = ring_layout(-half_width_deg + 2 * half_width_deg * slots, rng, height_sd=height_sd)
    return Scene(centers, headings, rng.uniform(*pitch, n), rng.normal(0, roll_sd, n) if roll_sd else np.zeros(n),
                 rng, **kw)


# ---------------------------------------------------------------- building blocks


@pytest.mark.parametrize("v", [[0, 0, 1], [0, 0, -1], [1e-9, 0, -1], [1, 2, 3], [0.3, -0.2, -0.9], [1, 0, 0]])
def test_rotation_to_z(v):
    r = rotation_to_z(np.array(v, float))
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-9)
    assert np.linalg.det(r) == pytest.approx(1.0)
    assert np.allclose(r @ (np.array(v, float) / np.linalg.norm(v)), [0, 0, 1], atol=1e-9)


def test_similarity_matrix_roundtrip():
    rng = np.random.default_rng(0)
    rot, pts = random_rotation(rng), rng.normal(size=(10, 3))
    t = similarity_matrix(2.5, rot, np.array([1.0, -2.0, 3.0]))
    assert np.allclose(apply_similarity(t, pts), 2.5 * pts @ rot.T + [1.0, -2.0, 3.0])
    assert np.allclose(apply_similarity(np.linalg.inv(t), apply_similarity(t, pts)), pts)


def test_rotation_z_turns_east_to_north():
    assert np.allclose(rotation_z(math.pi / 2) @ [1, 0, 0], [0, 1, 0])


def test_fit_similarity_2d_recovers_exact_transform():
    rng = np.random.default_rng(1)
    src = rng.normal(0, 5, (8, 2))
    scale, yaw, t = 3.2, math.radians(-130), np.array([12.0, -4.0])
    dst = scale * src @ rotation_z(yaw)[:2, :2].T + t
    s, y, tt = fit_similarity_2d(src, dst)
    assert s == pytest.approx(scale)
    assert angle_diff_deg(math.degrees(y), math.degrees(yaw)) == pytest.approx(0.0, abs=1e-9)
    assert np.allclose(tt, t)


def test_robust_similarity_2d_rejects_outliers():
    rng = np.random.default_rng(2)
    src = rng.uniform(-1, 1, (20, 2))
    scale, yaw, t = 30.0, math.radians(75), np.array([-3.0, 8.0])
    dst = scale * src @ rotation_z(yaw)[:2, :2].T + t + rng.normal(0, 2.0, (20, 2))
    outliers = np.zeros(20, bool)
    outliers[[1, 5, 9, 13, 17]] = True
    dst[outliers] += rng.uniform(30, 100, (5, 2)) * rng.choice([-1, 1], (5, 2))
    s, y, tt, inliers = robust_similarity_2d(src, dst, inlier_threshold=9.0, min_pair_distance=3.0)
    assert np.array_equal(inliers, ~outliers)
    assert s == pytest.approx(scale, rel=0.1)
    assert abs(angle_diff_deg(math.degrees(y), 75.0)) < 5.0


def test_robust_similarity_2d_needs_a_baseline():
    src = np.random.default_rng(3).normal(size=(5, 2))
    assert robust_similarity_2d(src, np.zeros((5, 2)), inlier_threshold=9.0, min_pair_distance=3.0) is None


def test_compass_yaw_ignores_a_wild_heading():
    # Viewing directions as seen in a frame whose axes are turned 40 degrees counter-clockwise from ENU.
    true_headings = [10.0, 80.0, 200.0, 300.0, 150.0]
    views = [rotation_z(math.radians(-40)) @ cam_from_enu(h, 5.0)[2] for h in true_headings]
    readings = [h + d for h, d in zip(true_headings, [3.0, -4.0, 2.0, 0.0, 170.0])]
    yaw, used = compass_yaw(np.array(views), readings + [None])
    assert used == 4
    assert abs(angle_diff_deg(math.degrees(yaw), 40.0)) < 2.0
    assert compass_yaw(np.array(views[:1]), [None]) is None


# ---------------------------------------------------------------- gravity


def test_estimate_up_single_facade_is_not_biased_by_pitch():
    rng = np.random.default_rng(4)
    s = sector_scene(rng, n=4, pitch=(10.0, 12.0))
    r = s.sfm_rots
    true_up = s.r0 @ [0.0, 0.0, 1.0]
    up = estimate_up(-r[:, 1, :], r[:, 0, :], s.sfm_centers)
    assert angle_between_deg(up, true_up) < 3.0     # only the light pull toward the mean up is left
    # The naive mean camera up leans back by about the pitch.
    mean_up = (-r[:, 1, :]).mean(axis=0)
    assert angle_between_deg(mean_up, true_up) > 8.0


def test_estimate_up_tolerates_roll_and_height_noise():
    rng = np.random.default_rng(5)
    errors = []
    for _ in range(50):
        s = sector_scene(rng, n=6, pitch=(5.0, 25.0), roll_sd=2.0, height_sd=0.3)
        r = s.sfm_rots
        errors.append(angle_between_deg(estimate_up(-r[:, 1, :], r[:, 0, :], s.sfm_centers), s.r0 @ [0, 0, 1.0]))
    assert np.median(errors) < 3.0
    assert np.percentile(errors, 90) < 6.0


def test_estimate_up_degenerate_street_line_stays_near_mean_up():
    # Walking along a straight street, all facing the same facade: tilt about the street axis is
    # unobservable from right vectors and centres, so the estimate must not run off.
    rng = np.random.default_rng(6)
    centers = np.c_[rng.uniform(-15, 15, 6), np.full(6, -30.0), np.full(6, CAMERA_HEIGHT)]
    s = Scene(centers, np.zeros(6), np.full(6, 10.0), np.zeros(6), rng)
    r = s.sfm_rots
    up = estimate_up(-r[:, 1, :], r[:, 0, :], s.sfm_centers)
    assert angle_between_deg(up, s.r0 @ [0, 0, 1.0]) < 12.0


# ---------------------------------------------------------------- full alignment


def test_georegister_full_ring():
    rng = np.random.default_rng(7)
    bearings = [k * 45 + rng.uniform(-18, 18) for k in range(8) for _ in range(4)]
    centers, headings = ring_layout(bearings, rng)
    s = Scene(centers, headings, rng.uniform(5, 12, 32), np.zeros(32), rng)
    g = s.georegister()
    assert g.georegistered and g.inliers.all()
    enu = apply_similarity(g.transform, s.sfm_centers)
    assert np.abs(enu - s.centers).max() < 2.0
    assert np.allclose(enu[:, 2], CAMERA_HEIGHT, atol=0.05)
    assert s.heading_errors(g.transform).max() < 2.0
    assert 1.0 < g.rmse_m < 4.0     # GPS noise is 2 m per axis


def test_georegister_short_arc_equal_height():
    """One facade: cameras on a ~37 degree arc at equal height, where a 3-D Sim3 to GPS is ill-conditioned."""
    rng = np.random.default_rng(8)
    facade = np.array([[x, 7.0, z] for x in (-10.0, 0.0, 10.0) for z in (0.0, 12.0)])   # north wall
    registered = 0
    for _ in range(20):
        s = sector_scene(rng, n=4, pitch=(5.0, 12.0))
        g = s.georegister(scene_points=facade)
        enu = apply_similarity(g.transform, s.sfm_centers)
        if g.georegistered:
            registered += 1
            assert np.linalg.norm(enu - s.centers, axis=1).max() < 8.0
        assert np.allclose(enu[:, 2], CAMERA_HEIGHT, atol=0.7)    # a 3 degree tilt over a 20 m arc
        assert s.heading_errors(g.transform).max() < 15.0
        wall = apply_similarity(g.transform, s.to_sfm(facade))
        vertical = wall[1::2] - wall[0::2]
        assert max(angle_between_deg(v, np.array([0, 0, 1.0])) for v in vertical) < 3.5
    # Now and then GPS noise squeezes the ~8 m spread enough that the scale is rightly not trusted.
    assert registered >= 17


def test_georegister_drops_a_gps_outlier():
    rng = np.random.default_rng(9)
    s = sector_scene(rng, n=8, half_width_deg=40.0)
    s.gps[3] += [35.0, -20.0]
    g = s.georegister()
    assert g.georegistered
    assert not g.inliers[3] and g.inliers.sum() == 7
    assert g.rmse_m < 4.0
    enu = apply_similarity(g.transform, s.sfm_centers)
    assert np.linalg.norm(enu - s.centers, axis=1).max() < 5.0


def test_georegister_small_baseline_falls_back_to_compass():
    rng = np.random.default_rng(10)
    centers, headings = ring_layout(rng.uniform(-1, 1, 5), rng, radius=(29.5, 30.5))
    s = Scene(centers, headings, rng.uniform(5, 12, 5), np.zeros(5), rng)
    facade = np.array([[x, 7.0, z] for x in np.linspace(-10, 10, 9) for z in np.linspace(0, 12, 5)])
    g = s.georegister(scene_points=facade)
    assert not g.georegistered
    assert "not georegistered" in g.message
    assert s.heading_errors(g.transform).max() < 12.0
    enu = apply_similarity(g.transform, s.sfm_centers)
    assert np.allclose(enu[:, 2], CAMERA_HEIGHT, atol=0.05)
    scale = np.cbrt(np.linalg.det(g.transform[:3, :3])) * s.s0
    assert 0.7 < scale < 1.6


def test_georegister_without_compass_or_baseline_still_returns_a_transform():
    rng = np.random.default_rng(11)
    centers, headings = ring_layout(np.zeros(3), rng, radius=(30.0, 30.2))
    s = Scene(centers, headings, np.full(3, 8.0), np.zeros(3), rng, gps_noise=0.0)
    g = s.georegister(compass=[None, None, None])
    assert not g.georegistered
    assert np.isfinite(g.transform).all()


# ---------------------------------------------------------------- real-photo failure modes


def test_estimate_up_steep_pitch_is_not_replaced_by_the_mean_up():
    # Photos right below a tall building look up 30-45 degrees; the mean up leans back by as much,
    # so falling back to it whenever the estimate is far from it was exactly wrong.
    rng = np.random.default_rng(12)
    errors = []
    for _ in range(50):
        s = sector_scene(rng, n=6, pitch=(30.0, 45.0), roll_sd=2.0, height_sd=0.3)
        r = s.sfm_rots
        errors.append(angle_between_deg(estimate_up(-r[:, 1, :], r[:, 0, :], s.sfm_centers), s.r0 @ [0, 0, 1.0]))
    assert np.median(errors) < 4.0
    assert np.percentile(errors, 90) < 8.0


def test_estimate_up_ignores_an_elevated_camera():
    # One photo from a tower terrace 25 m up must not tilt the "photographers stand at one height" term.
    rng = np.random.default_rng(13)
    errors = []
    for _ in range(50):
        s = sector_scene(rng, n=7, pitch=(5.0, 25.0), roll_sd=2.0, height_sd=0.3)
        s.centers[0, 2] += 25.0
        s.sfm_centers = s.to_sfm(s.centers)
        r = s.sfm_rots
        errors.append(angle_between_deg(estimate_up(-r[:, 1, :], r[:, 0, :], s.sfm_centers), s.r0 @ [0, 0, 1.0]))
    assert np.median(errors) < 4.0
    assert np.percentile(errors, 90) < 8.0


def test_georegister_never_mirrors_when_gps_and_compass_disagree():
    # Aachen east facade: five photos a few metres apart whose GPS row runs the wrong way, compass correct.
    # Fusing a 180 degree disagreement made the least-squares scale negative: a mirrored, upside-down model.
    x = np.arange(5.0)
    centers = np.c_[x, np.zeros(5), np.zeros(5)]
    views, ups, rights = np.tile([0.0, 1.0, 0.0], (5, 1)), np.tile([0.0, 0.0, 1.0], (5, 1)), np.tile([1.0, 0, 0], (5, 1))
    gps = np.c_[-3.0 * x, np.zeros(5)]
    for sigma in (None, 2.0):    # default sigma: baseline too short; 2 m: a "usable" fit that conflicts
        g = georegister(centers, ups, rights, views, gps, [0.0] * 5, CAMERA_HEIGHT, gps_sigma_m=sigma)
        assert np.linalg.det(g.transform[:3, :3]) > 0
        assert not g.georegistered and "not georegistered" in g.message
        heads = [heading_from_en(*(g.transform[:3, :3] @ v)[:2]) for v in views]
        assert max(abs(angle_diff_deg(h, 0.0)) for h in heads) < 1.0
        above = apply_similarity(g.transform, np.array([[2.0, 0.0, 10.0], [2.0, 0.0, 0.0]]))
        assert above[0, 2] > above[1, 2]
    assert "from the compass yaw" in g.message


def test_georegister_without_compass_uses_the_landmark_direction_over_a_wrong_gps_yaw():
    # Real Aachen photos: no compass, and GPS fixes from one phone session that turn a facade by 40-170 deg.
    rng = np.random.default_rng(14)
    for turn in (60.0, 120.0, 170.0):
        s = sector_scene(rng, n=6, gps_noise=1.0)
        mean = s.gps.mean(axis=0)
        s.gps = (s.gps - mean) @ rotation_z(math.radians(turn))[:2, :2].T + mean
        g = s.georegister(compass=[None] * 6)
        assert not g.georegistered and g.yaw_source == "landmark-direction"
        assert np.median(s.heading_errors(g.transform)) < 10.0
        assert np.linalg.det(g.transform[:3, :3]) > 0
    # A short baseline that cannot fix the scale: the yaw also comes from the landmark direction, not from GPS.
    centers, headings = ring_layout(rng.uniform(-1, 1, 5), rng, radius=(29.5, 30.5))
    s = Scene(centers, headings, rng.uniform(5, 12, 5), np.zeros(5), rng, gps_noise=6.0)
    g = s.georegister(compass=[None] * 5)
    assert not g.georegistered and g.yaw_source == "landmark-direction"
    assert s.heading_errors(g.transform).max() < 12.0


def test_georegister_with_real_phone_gps_follows_a_good_compass():
    # Real phones: GPS ~6 m per axis with no accuracy tag, compass within a few degrees. With the old 3 m
    # assumption the GPS yaw of a facade outweighed the compass and headings came out 8-21 deg off.
    rng = np.random.default_rng(15)
    errors = []
    for _ in range(60):
        s = sector_scene(rng, n=8, half_width_deg=22.0, gps_noise=6.0, compass_noise=3.0)
        s.gps_sigma = None
        g = s.georegister()
        errors.append(np.median(s.heading_errors(g.transform)))
    assert np.median(errors) < 3.0
    assert np.percentile(errors, 90) < 6.0


def test_check_cameras_flags_photos_registered_on_a_twin_facade():
    # Florence Baptistry: SfM verified matches between two identical octagon faces and put 6 of 18 photos
    # in front of the face 90 deg away. Their GPS and their compass both disagree with that pose.
    rng = np.random.default_rng(16)
    centers, headings = ring_layout(rng.uniform(70, 110, 18), rng, radius=(55.0, 80.0))
    twins = np.zeros(18, bool)
    twins[[1, 4, 7, 10, 13, 16]] = True
    turn = rotation_z(math.radians(90.0))
    sfm_centers, sfm_headings = centers.copy(), np.array(headings, float)
    sfm_centers[twins] = centers[twins] @ turn.T
    sfm_headings[twins] -= 90.0                      # turned counter-clockwise: heading decreases
    s = Scene(sfm_centers, sfm_headings, rng.uniform(5, 15, 18), np.zeros(18), rng, gps_noise=5.0, compass_noise=4.0)
    s.gps = centers[:, :2] + rng.normal(0, 5.0, (18, 2))        # the sensors saw the true poses
    s.compass = [float(h + rng.normal(0, 4.0)) for h in headings]
    s.gps_sigma = None
    g = s.georegister()
    cams = check_cameras(g.transform, s.sfm_rots, s.sfm_centers, s.gps, s.compass, [None] * 18, [1.0] * 18,
                         [f"p{i}" for i in range(18)])
    flagged = np.array([bool(cams[f"p{i}"]["flags"]) for i in range(18)])
    assert np.array_equal(flagged, twins), [cams[f"p{i}"] for i in range(18)]
    assert all("gps_and_compass" in cams[f"p{i}"]["flags"] for i in np.flatnonzero(twins))


def test_camera_flags():
    thr = GPS_INLIER_M
    assert camera_flags(thr / 2, thr, 80.0, 1.0) == []               # compass alone is not enough
    assert camera_flags(thr * 1.5, thr, 30.0, 1.0) == []             # GPS alone, within 3 thresholds
    assert camera_flags(thr * 1.5, thr, -60.0, 1.0) == ["gps_and_compass"]
    assert camera_flags(thr * 3.5, thr, None, 1.0) == ["far_from_gps"]   # e.g. a telephoto shot placed 88 m off
    assert camera_flags(1.0, thr, 2.0, 2438.6) == ["odd_focal"]
    assert camera_flags(1.0, thr, None, None) == []


# ---------------------------------------------------------------- what the snap needs (STYLIZED_MODEL 3.2)


def test_georegister_reports_its_yaw_sigma_with_the_systematic_floor_and_its_scale_sigma():
    """The snap's yaw window is +-3 sigma of what georegister reports. GPS + compass sigmas assume independent
    errors; real facade jobs were 6-9 deg off while reporting +-2-3 deg, so a floor is added in quadrature."""
    rng = np.random.default_rng(21)
    s = sector_scene(rng, n=12, half_width_deg=40.0)
    g = s.georegister()
    assert g.georegistered and g.yaw_sigma_deg is not None and g.scale_sigma is not None
    printed = float(g.message.split("yaw +-")[1].split(" deg")[0])     # the independent-error sigma, rounded
    assert g.yaw_sigma_deg == pytest.approx(math.hypot(printed, 3.0), abs=0.6)
    assert g.yaw_sigma_deg >= params.DEFAULTS.yaw_systematic_sigma_deg
    assert f"scale +-{100 * g.scale_sigma:.0f}%" in g.message
    with params.override(yaw_systematic_sigma_deg=0.0):
        bare = s.georegister()
    assert bare.yaw_sigma_deg == pytest.approx(printed, abs=0.6) and bare.yaw_sigma_deg < g.yaw_sigma_deg
    # not georegistered: the yaw came from the compass alone, no GPS scale
    centers, headings = ring_layout(rng.uniform(-1, 1, 5), rng, radius=(29.5, 30.5))
    weak = Scene(centers, headings, rng.uniform(5, 12, 5), np.zeros(5), rng).georegister()
    assert not weak.georegistered and weak.scale_sigma is None and weak.yaw_sigma_deg > 3.0


def test_check_cameras_keeps_each_photos_own_gps_fix_for_the_snaps_gps_check():
    rng = np.random.default_rng(22)
    s = sector_scene(rng, n=6, half_width_deg=30.0)
    g = s.georegister()
    cams = check_cameras(g.transform, s.sfm_rots, s.sfm_centers, s.gps, s.compass, [None] * 6, [1.0] * 6,
                         [f"p{i}" for i in range(6)])
    assert all(np.allclose(cams[f"p{i}"]["gps_en"], s.gps[i], atol=0.01) for i in range(6))


def test_check_cameras_keeps_each_cameras_local_pose():
    """cameras[pid]["local"] is the pose in the SfM frame; through the GeoAnchor it gives the recorded ENU pose."""
    rng = np.random.default_rng(23)
    s = sector_scene(rng, n=6, half_width_deg=30.0)
    g = s.georegister()
    cams = check_cameras(g.transform, s.sfm_rots, s.sfm_centers, s.gps, s.compass, [None] * 6, [1.0] * 6,
                         [f"p{i}" for i in range(6)])
    for i in range(6):
        loc = cams[f"p{i}"]["local"]
        assert np.allclose(loc["c"], s.sfm_centers[i], atol=1e-6)
        enu, _, heading = g.anchor.camera_enu(loc["q"], loc["c"])
        assert np.abs(enu - cams[f"p{i}"]["enu"]).max() < 0.002
        assert abs(angle_diff_deg(heading, cams[f"p{i}"]["heading_deg"])) < 0.01


# ---------------------------------------------------------------- the GeoAnchor's sigmas (crowd3d.anchor)


def anchor_errors(s: Scene, g) -> dict[str, float]:
    """The anchor's signed errors against the scene's true similarity: yaw (deg), log-scale, east/north at the pivot."""
    a = g.anchor
    d = a.rotation() @ s.r0           # anchor local -> ENU after true ENU -> local
    piv_local = a.to_local(np.array([a.pivot_enu]))[0]
    true = s.r0.T @ (piv_local - s.t0) / s.s0
    return {"yaw_deg": math.degrees(math.atan2(d[1, 0] - d[0, 1], d[0, 0] + d[1, 1])),
            "log_scale": math.log(a.scale * s.s0), "east_m": a.pivot_enu[0] - true[0],
            "north_m": a.pivot_enu[1] - true[1]}


def test_anchor_sigmas_are_calibrated_on_simulated_facades():
    """300 one-facade scenes (15 cameras over 120 deg, GPS sigma 3 m as reported, compass 15 deg), floors 0: the
    share of yaw, log-scale and east/north (at the pivot) errors within 2 sigma is near 0.954. East/north at the
    pivot being calibrated on its own is the decorrelation the anchor's diagonal sigmas rely on."""
    rng = np.random.default_rng(31)
    z: dict[str, list[float]] = {"yaw_deg": [], "log_scale": [], "east_m": [], "north_m": []}
    with params.override(anchor_yaw_floor_deg=0.0, anchor_scale_floor=0.0, anchor_xy_floor_m=0.0,
                         anchor_tilt_floor_deg=0.0):
        for _ in range(300):
            centers, headings = ring_layout(rng.uniform(-60, 60, 15), rng)
            s = Scene(centers, headings, rng.uniform(5, 15, 15), np.zeros(15), rng, gps_noise=3.0, compass_noise=15.0)
            g = s.georegister()
            assert g.georegistered
            for k, e in anchor_errors(s, g).items():
                z[k].append(e / g.anchor.sigma[k])
    for k, v in z.items():
        assert 0.92 <= np.mean(np.abs(v) <= 2.0) <= 0.98, (k, np.mean(np.abs(v) <= 2.0))


def test_a_gps_compass_conflict_inflates_the_anchor_yaw_sigma_only():
    rng = np.random.default_rng(32)
    s = sector_scene(rng, n=8, half_width_deg=25.0, gps_noise=2.0, compass_noise=2.0)
    calm = s.georegister()
    s.compass = [h + 30.0 for h in s.compass]             # a 30 deg magnetic disturbance (below the 40 deg conflict)
    biased = s.georegister()
    assert calm.georegistered and biased.georegistered
    assert calm.anchor.raw["birge"] == 1.0 and biased.anchor.raw["birge"] > 2.0
    assert abs(biased.anchor.raw["yaw_conflict_deg"]) > 20.0
    assert biased.anchor.sigma["yaw_deg"] > 2.0 * biased.yaw_sigma_deg       # the legacy sigma ignores the conflict
    core = biased.anchor.raw["sigma_core"]["yaw_deg"]
    fused = 1.0 / math.sqrt(1.0 / biased.anchor.raw["sigma_gps"] ** 2
                            + 1.0 / math.radians(biased.anchor.raw["sigma_prior_deg"]) ** 2)
    assert core == pytest.approx(math.degrees(fused) * biased.anchor.raw["birge"], rel=1e-4)


def test_anchor_is_deterministic():
    rng = np.random.default_rng(33)
    s = sector_scene(rng, n=7, gps_noise=4.0)
    a, b = s.georegister().anchor.to_meta(), s.georegister().anchor.to_meta()
    assert a == b


def test_anchor_does_not_depend_on_the_camera_order():
    """A kept model read back from disk lists its images in another order: the replayed anchor must be the same."""
    rng = np.random.default_rng(34)
    s = sector_scene(rng, n=9, gps_noise=3.0)
    g = s.georegister()
    perm = rng.permutation(9)
    r = s.sfm_rots[perm]
    h = georegister(s.sfm_centers[perm], -r[:, 1, :], r[:, 0, :], r[:, 2, :], s.gps[perm],
                    [s.compass[i] for i in perm], CAMERA_HEIGHT, gps_sigma_m=s.gps_sigma)
    a, b = g.anchor.to_meta(), h.anchor.to_meta()
    assert a["sigma"] == b["sigma"] and a["raw"]["sigma_core"] == b["raw"]["sigma_core"]
    assert np.allclose(g.anchor.matrix(), h.anchor.matrix(), atol=1e-9)
