"""Geo-registration helpers on synthetic camera sets (pure numpy, no COLMAP)."""

from __future__ import annotations

import math
from dataclasses import replace as dataclasses_replace

import numpy as np
import pytest

from crowd3d import params
from crowd3d.anchor import ANCHOR_METHOD
from crowd3d.geo import angle_diff_deg, heading_from_en
from crowd3d.reconstruct import (
    GPS_INLIER_M,
    apply_similarity,
    camera_flags,
    check_cameras,
    compass_consensus,
    compass_errors,
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


def test_anchor_does_not_depend_on_the_sfm_frame():
    """Another SfM run of the same photos has another gauge (verify1: Frauenkirche r1 / r2 got tilt sigmas 2.349 vs
    2.418 deg for the same reconstruction): the same cameras in another local frame get the same anchor sigmas, cores
    and ENU placement (ANCHOR_METHOD 2: the tilt bootstrap about east and north, cameras in ENU-centre order, on a
    2^-30 grid; method 1 gave tilt sigmas up to 32 deg apart on 3-6 cameras, where resamples of copies of one camera
    turned rounding noise into any tilt)."""
    rng = np.random.default_rng(35)
    for _ in range(24):
        s = sector_scene(rng, n=int(rng.integers(3, 13)), gps_noise=3.0)
        g = s.georegister()
        q, k, t = random_rotation(rng), float(rng.uniform(0.1, 10.0)), rng.normal(0, 50, 3)
        r = s.sfm_rots @ q.T
        centers = k * s.sfm_centers @ q.T + t                                   # the same model, another frame
        h = georegister(centers, -r[:, 1, :], r[:, 0, :], r[:, 2, :], s.gps, s.compass, CAMERA_HEIGHT,
                        gps_sigma_m=s.gps_sigma)
        a, b = g.anchor, h.anchor
        assert a.method == b.method == ANCHOR_METHOD
        assert a.sigma == pytest.approx(b.sigma, abs=2e-6)
        assert a.raw["sigma_core"] == pytest.approx(b.raw["sigma_core"], abs=2e-6)
        assert np.allclose(a.to_enu(s.sfm_centers), b.to_enu(centers), atol=1e-6)
        assert a.pivot_enu == pytest.approx(b.pivot_enu, abs=2e-4)
        # another SfM run of the same photos also moves the cameras by about 1e-7 of the model's size (verify1's
        # Frauenkirche pairs: 5e-8 and 3e-7): its tilt sigma moves as little (method 1: up to 23 deg)
        size = float(np.linalg.norm(s.sfm_centers - s.sfm_centers.mean(axis=0), axis=1).mean())
        near = k * (s.sfm_centers + rng.normal(0, 1e-7 * size, s.sfm_centers.shape)) @ q.T + t
        n = georegister(near, -r[:, 1, :], r[:, 0, :], r[:, 2, :], s.gps, s.compass, CAMERA_HEIGHT,
                        gps_sigma_m=s.gps_sigma)
        assert abs(n.anchor.sigma["tilt_deg"] - a.sigma["tilt_deg"]) < 1e-4


# ---------------------------------------------------------------- how the evidence is weighed (D34)

FACADE_POINTS = np.array([[x, 7.0, z] for x in np.linspace(-10, 10, 9) for z in np.linspace(0, 12, 5)])


def same_spot_scene(seed: int, n: int, half_width_deg: float = 1.0, gps_noise: float = 2.0) -> Scene:
    """n cameras within half_width_deg of one bearing, 30 m from the wall: about 1 m apart at +-1 deg."""
    rng = np.random.default_rng(1000 + seed)
    centers, headings = ring_layout(rng.uniform(-half_width_deg, half_width_deg, n), rng, radius=(29.5, 30.5))
    return Scene(centers, headings, rng.uniform(5, 12, n), np.zeros(n), rng, gps_noise=gps_noise)


def scale_ratio(s: Scene, g) -> float:
    return float(np.cbrt(np.linalg.det(g.transform[:3, :3])) * s.s0)


@pytest.mark.parametrize("n", [5, 8, 12])
def test_cameras_at_one_spot_are_not_georegistered_when_the_span_is_measured_on_the_fitted_layout(n):
    """The review's finding: the span gate measured the GPS fixes, and fixes scattered round one spot have a spread
    of their own (RMS sigma x sqrt(2 (N - 1) / N)), so same-spot sets passed it: 14, 40 and 46 of these 100 seeds
    at n = 5, 8 and 12 (the verifier's 300 seeds: 15 %, 34 %, 44 %; the one earlier test of this layout,
    test_georegister_small_baseline_falls_back_to_compass, has a single seed), their scale off by a median factor of
    3.8, 2.7 and 2.1. Measured on the fitted camera layout (Params.gps_span_model_side) 2, 0 and 1 of 100 pass:
    the ones whose fit blew the layout up 5 to 10 times, so that even the fitted layout looks wide enough."""
    seeds = 100
    old = [(s, s.georegister(scene_points=FACADE_POINTS)) for s in (same_spot_scene(k, n) for k in range(seeds))]
    passed = [(s, g) for s, g in old if g.georegistered]
    assert len(passed) > 0.10 * seeds                                    # the old gate: no rare accident
    assert np.median([abs(math.log(scale_ratio(s, g))) for s, g in passed]) > math.log(1.9)
    with params.override(gps_span_model_side=True):
        new = [same_spot_scene(k, n).georegister(scene_points=FACADE_POINTS) for k in range(seeds)]
    assert sum(g.georegistered for g in new) <= 0.02 * seeds
    for (s, _), g in zip(old, new):
        if not g.georegistered:                 # what the product then says: compass yaw, the scale a guess
            assert "not georegistered" in g.message and g.yaw_source == "compass" and g.scale_sigma is None
            assert g.anchor.level == "prior_only" and g.anchor.raw["gps_span"] == "model"
            assert s.heading_errors(g.transform).max() < 15.0


def test_the_span_measured_on_the_fitted_layout_keeps_a_real_baseline_and_reports_a_larger_sigma():
    """Cameras spread over a facade (80 deg of arc at 30-40 m, GPS sigma 2 m): georegistered either way, to the same
    place; the fitted layout's spread is the fixes' without their own scatter, so the reported scale sigma grows."""
    registered = 0
    for seed in range(40):
        s = sector_scene(np.random.default_rng(2000 + seed), n=8, half_width_deg=40.0)
        old = s.georegister()
        with params.override(gps_span_model_side=True):
            new = s.georegister()
        assert old.georegistered and old.anchor.raw["gps_span"] == "fixes"
        registered += new.georegistered
        if new.georegistered:
            assert 0.8 < scale_ratio(s, new) < 1.25
            assert new.scale_sigma >= old.scale_sigma - 1e-4
            assert new.anchor.raw["gps_spread_m"] <= old.anchor.raw["gps_spread_m"] + 1e-6
            assert np.abs(apply_similarity(new.transform, s.sfm_centers)
                          - apply_similarity(old.transform, s.sfm_centers)).max() < 1.0
    assert registered >= 39


def facade_scene(seed: int, n: int = 20, length_m: float = 80.0, gps_noise: float = 6.0) -> Scene:
    """n cameras along a facade length_m long, 25-35 m in front of it, all looking at it (north)."""
    rng = np.random.default_rng(3000 + seed)
    centers = np.c_[rng.uniform(-length_m / 2, length_m / 2, n), rng.uniform(-35.0, -25.0, n),
                    np.full(n, CAMERA_HEIGHT)]
    headings = rng.uniform(-10.0, 10.0, n)
    s = Scene(centers, headings, rng.uniform(5, 12, n), np.zeros(n), rng, gps_noise=gps_noise)
    s.gps_sigma = None                                    # phones that report no accuracy: Params.gps_sigma_m
    return s


def compass_readings(s: Scene, offsets: dict[int, float]) -> list[float | None]:
    """A compass reading for the listed cameras only: the true heading plus the given error."""
    return [float(s.headings[i] + offsets[i]) if i in offsets else None for i in range(len(s.headings))]


def test_one_bad_compass_reading_does_not_overrule_a_strong_gps_fit():
    """The review's finding: 20 cameras spread over 80 m (GPS sigma 6 m) and a single compass reading 90 deg off.
    The yaw conflict rule gave the model to the compass: turned 89 deg, scale a guess, not georegistered, none of
    its cameras shown. With Params.yaw_arbitration the GPS fit stands and the reading is left out."""
    errors: dict[str, list[float]] = {"none": [], "veto": [], "arbitrated": [], "two": [], "three good": []}
    overruled_two = 0
    for seed in range(25):
        s = facade_scene(seed)
        free = s.georegister(compass=[None] * 20)
        assert free.georegistered
        errors["none"].append(float(np.median(s.heading_errors(free.transform))))
        bad = compass_readings(s, {3: 90.0})
        veto = s.georegister(compass=bad)
        assert not veto.georegistered and "from the compass yaw" in veto.message and veto.yaw_source == "compass"
        errors["veto"].append(float(np.median(s.heading_errors(veto.transform))))
        with params.override(yaw_arbitration=True):
            g = s.georegister(compass=bad)
            two = s.georegister(compass=compass_readings(s, {3: 90.0, 11: 0.0}))       # one bad, one good
            three = s.georegister(compass=compass_readings(s, {3: 90.0, 5: 2.0, 11: -3.0, 17: 1.0}))
        assert g.georegistered and g.prior_overruled and g.yaw_source == "gps" and g.anchor.level == "full"
        assert "compass yaw of 1 of 1 reading(s) overruled" in g.message and "GPS yaw - compass" not in g.message
        assert g.anchor.raw["yaw_prior_overruled"] is True and g.anchor.raw["birge"] == 1.0
        assert abs(g.anchor.raw["yaw_conflict_deg"]) > 60.0                        # kept on record
        assert abs(angle_diff_deg(g.anchor.yaw_deg, g.gps_yaw_deg)) < 1e-9         # the GPS fit, nothing fused in
        assert g.yaw_sigma_deg == pytest.approx(math.hypot(math.degrees(g.anchor.raw["sigma_gps"]), 3.0), abs=0.01)
        assert 0.8 < scale_ratio(s, g) < 1.25
        errors["arbitrated"].append(float(np.median(s.heading_errors(g.transform))))
        # two readings 90 deg apart cannot outvote each other (no outlier is dropped among two; their mean is
        # 45 deg off): beyond the conflict limit the fit stands, under it they are fused with their 10.6 deg sigma
        assert two.georegistered and (two.prior_overruled or abs(two.anchor.raw["yaw_conflict_deg"]) <= 40.0)
        overruled_two += two.prior_overruled
        errors["two"].append(float(np.median(s.heading_errors(two.transform))))
        # three good readings drop the bad one themselves (compass_yaw): no conflict, fused as ever
        assert three.georegistered and not three.prior_overruled and three.yaw_source == "gps+compass"
        errors["three good"].append(float(np.median(s.heading_errors(three.transform))))
    assert np.median(errors["veto"]) > 80.0 and overruled_two >= 15
    for case in ("none", "arbitrated", "two", "three good"):
        assert np.median(errors[case]) < 4.0 and max(errors[case]) < 12.0, case


def test_a_compass_consensus_or_a_weak_gps_fit_still_gives_a_yaw_conflict_to_the_compass():
    """Params.yaw_arbitration changes nothing unless the GPS fit is strong and the compass prior weak."""
    with params.override(yaw_arbitration=True):
        s = facade_scene(0)
        turned = (s.gps - s.gps.mean(axis=0)) @ rotation_z(math.radians(90.0))[:2, :2].T + s.gps.mean(axis=0)
        # three readings that agree are a consensus: a GPS layout turned 90 deg against them loses, however many
        # fixes it has (hand-placed geotags along the wrong street)
        s.gps = turned
        g = s.georegister(compass=compass_readings(s, {2: 3.0, 9: -4.0, 15: 1.0}))
        assert not g.georegistered and not g.prior_overruled and g.yaw_source == "compass"
        assert np.median(s.heading_errors(g.transform)) < 10.0
        assert g.anchor.raw["yaw_prior_overruled"] is False
        # a pseudo-compass (no reading at all, every camera assumed to face the landmark) is never overruled
        g = s.georegister(compass=[None] * 20)
        assert not g.georegistered and g.yaw_source == "landmark-direction" and not g.prior_overruled
        # five fixes are no strong fit, whatever their sigma: one reading against them wins as before
        few = facade_scene(1, n=5, length_m=60.0, gps_noise=1.0)
        few.gps = (few.gps - few.gps.mean(axis=0)) @ rotation_z(math.radians(120.0))[:2, :2].T + few.gps.mean(axis=0)
        g = few.georegister(compass=compass_readings(few, {0: 2.0}), gps_sigma=1.0)
        assert not g.georegistered and "from the compass yaw" in g.message and not g.prior_overruled
        # a conflict of under yaw_arb_gps_sigmas GPS yaw sigmas is not beyond GPS noise: the compass wins
        with params.override(yaw_arb_gps_sigmas=1e6):
            g = facade_scene(2).georegister(compass=compass_readings(facade_scene(2), {3: 90.0}))
        assert not g.georegistered and not g.prior_overruled
        # the Aachen east facade (five fixes a few metres apart in the wrong order, the compass right): unchanged
        test_georegister_never_mirrors_when_gps_and_compass_disagree()
        test_georegister_without_compass_uses_the_landmark_direction_over_a_wrong_gps_yaw()


def test_compass_consensus_weighs_each_reading_by_its_own_sigma():
    # viewing directions in a frame turned 40 deg counter-clockwise from ENU, as in the compass_yaw test above
    true_headings = [10.0, 80.0, 200.0, 300.0, 150.0, 30.0]
    views = np.array([rotation_z(math.radians(-40)) @ cam_from_enu(h, 5.0)[2] for h in true_headings])
    readings = [h + d for h, d in zip(true_headings, [2.0, -2.0, 90.0, 90.0, 90.0, 90.0])]   # 2 good, 4 bad by 90 deg
    plain = compass_consensus(views, readings)
    assert (plain.used, plain.readings, plain.weight) == (4, 6, 4.0)       # the four bad ones outvote the two good
    assert abs(angle_diff_deg(math.degrees(plain.yaw), 40.0 - 90.0)) < 1.0
    assert compass_yaw(views, readings) == (plain.yaw, plain.used)
    # the four bad ones with a 45 deg sigma weigh 1 / 9 of a reading each: the mean leans atan(4 / 9 / 2) = 12.5 deg
    # their way, and nobody is dropped (they are 77.5 deg off it, within 3 of their own sigmas)
    weak = compass_consensus(views, readings, sigmas_deg=[15.0, 15.0, 45.0, 45.0, 45.0, 45.0])
    assert (weak.used, weak.readings) == (6, 6) and weak.weight == pytest.approx(2.0 + 4.0 / 9.0)
    assert abs(angle_diff_deg(math.degrees(weak.yaw), 40.0)) == pytest.approx(12.5, abs=1.0)
    assert weak.scatter > 1.0
    # left out: exactly the two good readings
    only = compass_consensus(views, readings, sigmas_deg=[15.0, 15.0, None, None, None, None])
    two = compass_consensus(views[:2], readings[:2])
    assert only == two and only.readings == 2 and abs(angle_diff_deg(math.degrees(only.yaw), 40.0)) < 2.5
    assert only.scatter == pytest.approx(2.0 / 15.0, abs=0.01)             # each 2 deg from their mean
    assert compass_consensus(views, readings, sigmas_deg=[None] * 6) is None
    # one sigma for all is the plain circular mean, bit for bit (what a replay of a recorded run relies on)
    assert compass_consensus(views, readings, sigmas_deg=[15.0] * 6) == plain


def magnetic_scene(seed: int) -> tuple[Scene, list[str | None]]:
    """A short facade arc (GPS cannot fix the yaw well) whose two true-north readings are right and whose three
    magnetic ones and one untagged one are 90 deg off."""
    s = sector_scene(np.random.default_rng(4000 + seed), n=6, half_width_deg=12.0, gps_noise=6.0, compass_noise=2.0)
    refs: list[str | None] = ["T", "M", "M", "T", None, "M"]
    s.compass = [c if r == "T" else c + 90.0 for c, r in zip(s.compass, refs)]
    s.gps_sigma = None
    return s, refs


def test_headings_not_tagged_true_north_count_for_what_compass_other_says():
    """ReconPhoto.heading_ref reaches the yaw prior: "trusted" (the default: every reading alike), "weak" (a 45 deg
    sigma) or "ignored" (no reading)."""
    errors: dict[str, list[float]] = {"trusted": [], "weak": [], "ignored": []}
    for seed in range(20):
        s, refs = magnetic_scene(seed)
        r = s.sfm_rots
        for mode in errors:
            with params.override(compass_other=mode):
                g = georegister(s.sfm_centers, -r[:, 1, :], r[:, 0, :], r[:, 2, :], s.gps, s.compass, CAMERA_HEIGHT,
                                heading_refs=refs)
            errors[mode].append(float(np.median(s.heading_errors(g.transform))))
            assert g.anchor.raw["yaw_prior_source"] == "compass"
        legacy = georegister(s.sfm_centers, -r[:, 1, :], r[:, 0, :], r[:, 2, :], s.gps, s.compass, CAMERA_HEIGHT)
        tagged = georegister(s.sfm_centers, -r[:, 1, :], r[:, 0, :], r[:, 2, :], s.gps, s.compass, CAMERA_HEIGHT,
                             heading_refs=refs)
        assert np.array_equal(legacy.transform, tagged.transform)       # the default reads no tag
        assert legacy.anchor.to_meta() == tagged.anchor.to_meta()
    assert np.median(errors["trusted"]) > 45.0          # four wrong readings against two
    assert np.median(errors["weak"]) < 35.0
    assert np.median(errors["ignored"]) < 8.0
    # no true-north reading at all: "ignored" leaves the pseudo-compass
    s, _ = magnetic_scene(0)
    r = s.sfm_rots
    with params.override(compass_other="ignored"):
        g = georegister(s.sfm_centers, -r[:, 1, :], r[:, 0, :], r[:, 2, :], s.gps, s.compass, CAMERA_HEIGHT,
                        heading_refs=["M", None, "M", "M", None, "M"])
    assert "landmark-direction" in g.yaw_source and g.anchor.raw["yaw_prior_source"] == "landmark-direction"


def test_params_say_what_a_compass_reading_is_worth():
    p = params.DEFAULTS
    assert [p.compass_reading_sigma_deg(r) for r in ("T", "t", " T ", "M", None, "")] == [15.0] * 6
    weak = params.with_overrides({"compass_other": "weak"})
    assert [weak.compass_reading_sigma_deg(r) for r in ("T", "M", None)] == [15.0, 45.0, 45.0]
    ignored = params.with_overrides({"compass_other": "ignored", "compass_sigma_deg": 12.0})
    assert [ignored.compass_reading_sigma_deg(r) for r in ("T", "M", None)] == [12.0, None, None]
    with pytest.raises(ValueError, match="compass_other"):
        params.with_overrides({"compass_other": "sometimes"})
    with pytest.raises(ValueError, match="compass_other_sigma_deg"):
        params.with_overrides({"compass_other_sigma_deg": 0.0})


def test_check_cameras_records_what_each_reading_counts_for_and_flags_by_it():
    """cameras[pid]["compass_sigma_deg"]: the reading's own sigma (None: no reading, or one that is left out). The
    gps_and_compass flag needs the offset beyond Params.gross_compass_sigmas of that sigma; compass_errors scales
    each reading's offset to degrees of a true-north one for the candidate ranking and the QA gate."""
    n = 5
    centers = np.c_[np.arange(n) * 10.0, np.zeros(n), np.full(n, CAMERA_HEIGHT)]
    rots = np.array([cam_from_enu(0.0, 0.0)] * n)                # all looking north, the frame is ENU already
    gps = centers[:, :2].copy()
    gps[1] += [0.0, 40.0]                                         # cameras 1 and 2: far from their own fix
    gps[2] += [0.0, 40.0]
    headings = [3.0, 60.0, 60.0, None, 60.0]                     # 60 deg off the estimate: gross for a 15 deg sigma
    refs = ["T", "T", "M", None, "M"]
    args = (np.eye(4), rots, centers, gps, headings, [None] * n, [1.0] * n, [f"p{i}" for i in range(n)])

    def sigmas_and_flags(cams):
        return [(cams[f"p{i}"]["compass_sigma_deg"], cams[f"p{i}"]["flags"]) for i in range(n)]

    legacy = check_cameras(*args)
    assert sigmas_and_flags(legacy) == [(15.0, []), (15.0, ["gps_and_compass"]), (15.0, ["gps_and_compass"]),
                                        (None, []), (15.0, [])]
    assert check_cameras(*args, heading_refs=refs) == legacy                      # "trusted": the tag is not read
    assert [c["compass_offset_deg"] for c in legacy.values()] == [-3.0, -60.0, -60.0, None, -60.0]
    assert compass_errors(legacy) == [3.0, 60.0, 60.0, 60.0]
    with params.override(compass_other="weak"):
        weak = check_cameras(*args, heading_refs=refs)
        assert sigmas_and_flags(weak) == [(15.0, []), (15.0, ["gps_and_compass"]), (45.0, []), (None, []), (45.0, [])]
        assert compass_errors(weak) == pytest.approx([3.0, 60.0, 20.0, 20.0])
        assert camera_flags(40.0, GPS_INLIER_M, 140.0, 1.0, 45.0) == ["gps_and_compass"]     # beyond 3 x 45 deg
    with params.override(compass_other="ignored"):
        ignored = check_cameras(*args, heading_refs=refs)
        assert sigmas_and_flags(ignored) == [(15.0, []), (15.0, ["gps_and_compass"]), (None, []), (None, []),
                                             (None, [])]
        assert compass_errors(ignored) == [3.0, 60.0]
        assert [c["compass_offset_deg"] for c in ignored.values()] == [-3.0, -60.0, -60.0, None, -60.0]   # on record
    # a camera entry written before readings had a sigma of their own counts in full
    assert compass_errors({"a": {"compass_offset_deg": -30.0}, "b": {"compass_offset_deg": None}, "c": {}}) == [30.0]


def test_readings_the_yaw_arbitration_overruled_are_left_out_of_the_camera_checks():
    """georegister_model on the review's scene (20 cameras over 80 m, one compass reading 90 deg off, that camera's
    fix 60 m away). The old rule turns the model onto the reading: its offset reads 0 and nothing is wrong with a
    model 90 deg off. With Params.yaw_arbitration the GPS fit stands, and the reading it overruled is left out
    (compass_sigma_deg None): it would otherwise flag its camera (gps_and_compass) and fail the model at the QA
    gate's compass check (compass_errors: a median of one reading 90 deg off) for the very reading the
    georegistration refused to follow. A reading the fit agrees with stays a reading."""
    from pathlib import Path

    from crowd3d.reconstruct import GeoregisterInputs, ReconPhoto, georegister_model

    def model(readings: list[float | None]):
        s = facade_scene(4)
        n = len(s.headings)
        s.gps[3] += [0.0, 60.0]
        photos = [ReconPhoto(f"p{i}", Path(), 0.0, 0.0, readings(s)[i], heading_ref="T") for i in range(n)]
        inputs = GeoregisterInputs(images=list(range(n)), photos=photos, rots=s.sfm_rots, centers=s.sfm_centers,
                                   focal=np.ones(n), gps_en=s.gps, accuracy=np.full(n, np.nan),
                                   use=np.ones(n, dtype=bool), xyz=np.zeros((0, 3)))
        return s, *georegister_model(inputs, CAMERA_HEIGHT)

    def one_bad(s):
        return compass_readings(s, {3: 90.0})

    s, veto, cams = model(one_bad)
    assert not veto.georegistered and not veto.prior_overruled
    assert np.median(s.heading_errors(veto.transform)) > 80.0
    assert cams["p3"]["compass_sigma_deg"] == 15.0 and abs(cams["p3"]["compass_offset_deg"]) < 1.0
    assert compass_errors(cams) == [abs(cams["p3"]["compass_offset_deg"])]
    with params.override(yaw_arbitration=True):
        s, g, cams = model(one_bad)
        assert g.georegistered and g.prior_overruled and np.median(s.heading_errors(g.transform)) < 5.0
        assert abs(cams["p3"]["compass_offset_deg"]) > 75.0 and cams["p3"]["gps_residual_m"] > 40.0   # on record
        assert cams["p3"]["compass_sigma_deg"] is None and "gps_and_compass" not in cams["p3"]["flags"]
        assert compass_errors(cams) == []
        # the same camera's checks had the reading counted: flagged, and the gate's median 90 deg off
        counted = check_cameras(g.transform, s.sfm_rots, s.sfm_centers, s.gps, one_bad(s), [None] * 20, [1.0] * 20,
                                [f"p{i}" for i in range(20)], heading_refs=["T"] * 20)
        assert "gps_and_compass" in counted["p3"]["flags"] and compass_errors(counted)[0] > 75.0
        # three good readings and the bad one: no conflict, nothing overruled, every reading counts
        s, g, cams = model(lambda s: compass_readings(s, {3: 90.0, 5: 2.0, 11: -3.0, 17: 1.0}))
        assert g.georegistered and not g.prior_overruled
        assert [cams[f"p{i}"]["compass_sigma_deg"] for i in (3, 5, 11, 17)] == [15.0] * 4
        assert "gps_and_compass" in cams["p3"]["flags"] and len(compass_errors(cams)) == 4


def test_a_recorded_runs_parameters_do_not_follow_later_defaults():
    """params.recorded_run (what a replay of a recorded run uses): a run recorded before the D34 switches existed is
    replayed as it ran, whatever those switches' defaults have become since; one that recorded them keeps what it
    recorded; names this code no longer has are dropped; the GeoAnchor floors are the current ones."""
    recorded = {"gps_sigma_m": 5.0, "max_yaw_conflict_deg": 35.0, "a_parameter_long_gone": 1,
                "anchor_yaw_floor_deg": 99.0}
    later = params.with_overrides({"gps_span_model_side": True, "compass_other": "ignored", "yaw_arbitration": True,
                                   "compass_sigma_deg": 12.0, "anchor_yaw_floor_deg": 5.0})
    run = params.recorded_run(recorded, base=later)
    assert (run.gps_span_model_side, run.compass_other, run.yaw_arbitration) == (False, "trusted", False)
    assert (run.gps_sigma_m, run.max_yaw_conflict_deg) == (5.0, 35.0)
    assert run.compass_sigma_deg == 12.0 and run.anchor_yaw_floor_deg == 5.0     # not recorded, no entry: base's
    run = params.recorded_run({**recorded, "gps_span_model_side": True, "compass_other": "weak"}, base=later)
    assert (run.gps_span_model_side, run.compass_other, run.yaw_arbitration) == (True, "weak", False)
    with params.using(later):                                                   # the default base: the current set
        assert params.recorded_run({}) == dataclasses_replace(later, gps_span_model_side=False,
                                                              compass_other="trusted", yaw_arbitration=False)
    assert params.recorded_run({}) == params.DEFAULTS       # today the defaults are the old behaviour
    assert set(params.AS_BEFORE) == {"gps_span_model_side", "compass_other", "yaw_arbitration"}
