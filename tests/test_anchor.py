"""crowd3d.anchor.GeoAnchor: the record georegister makes (pure numpy, no COLMAP)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from crowd3d import params
from crowd3d.anchor import (ANCHOR_METHOD, ANCHOR_SCHEMA, GeoAnchor, calibration, floored_sigma, quat_matrix,
                            quat_xyzw, read_anchor)
from crowd3d.reconstruct import rotation_to_z, rotation_z
from test_georegister import random_rotation, sector_scene


def make_anchor(**kw) -> GeoAnchor:
    up = np.array([0.1, -0.2, 0.97])
    r = rotation_z(math.radians(33.0)) @ rotation_to_z(up)
    core = {"yaw_deg": 2.0, "log_scale": 0.05, "xy_m": 1.5, "tilt_deg": 0.4}
    floors = {"yaw_deg": 3.0, "log_scale": 0.0, "xy_m": 2.0, "tilt_deg": 0.0}
    d = dict(frame="job", level="full", scale=12.5, rotation_xyzw=quat_xyzw(r), translation=[4.0, -7.0, 1.2],
             up_local=up.tolist(), yaw_deg=33.0, pivot_enu=[10.0, 0.0, 1.6],
             sigma=floored_sigma(core, 1.5, floors), source={"yaw": "gps+compass"}, raw={"sigma_core": core},
             calibration={"id": "test", "floors": floors})
    d.update(kw)
    return GeoAnchor(**d)


def test_quaternion_round_trip():
    rng = np.random.default_rng(0)
    for r in [random_rotation(rng) for _ in range(50)] + [np.eye(3), np.diag([1.0, -1, -1]), np.diag([-1.0, 1, -1])]:
        q = quat_xyzw(r)
        assert q[3] >= 0 and abs(np.linalg.norm(q) - 1) < 1e-12
        assert np.allclose(quat_matrix(q), r, atol=1e-12)


def test_matrix_is_the_decomposition_and_to_local_inverts_to_enu():
    a = make_anchor()
    r = rotation_z(math.radians(a.yaw_deg)) @ rotation_to_z(np.array(a.up_local))
    m = a.matrix()
    assert np.allclose(m[:3, :3], a.scale * r, atol=1e-12) and np.allclose(m[:3, 3], a.translation)
    p = np.random.default_rng(1).normal(0, 30, (20, 3))
    assert np.allclose(a.to_enu(p), p @ m[:3, :3].T + m[:3, 3], atol=1e-9)
    assert np.allclose(a.to_local(a.to_enu(p)), p, atol=1e-9)


def test_meta_round_trip_is_json():
    a = make_anchor()
    meta = json.loads(json.dumps(a.to_meta()))
    assert meta["schema"] == ANCHOR_SCHEMA and meta["method"] == ANCHOR_METHOD == a.method
    assert GeoAnchor.from_meta(meta) == a
    assert GeoAnchor.from_meta({k: v for k, v in meta.items() if k != "method"}).method == 1   # recorded before it
    assert read_anchor({"anchor": meta}) == a
    assert read_anchor({"georegistered": True, "yaw_sigma_deg": 4.0}) is None      # a model JSON before anchors
    with pytest.raises(ValueError):
        GeoAnchor.from_meta({**meta, "schema": "crowd3d.anchor/0"})


def test_position_sigma_grows_with_distance_from_the_pivot():
    a = make_anchor()
    s = a.sigma
    assert s["east_m"] == s["north_m"] == pytest.approx(2.5)        # hypot(1.5, 2.0)
    assert s["yaw_deg"] == pytest.approx(math.hypot(2.0, 3.0), abs=1e-6)
    h, v = a.position_sigma_m(a.pivot_enu)
    assert (h, v) == (pytest.approx(2.5), pytest.approx(1.5))
    d = 40.0
    h, v = a.position_sigma_m([a.pivot_enu[0], a.pivot_enu[1] + d, 30.0])
    assert h == pytest.approx(math.sqrt(2.5**2 + d**2 * (math.radians(s["yaw_deg"]) ** 2 + s["log_scale"] ** 2)))
    assert v == pytest.approx(math.sqrt(1.5**2 + d**2 * math.radians(s["tilt_deg"]) ** 2))


def test_camera_enu_turns_a_local_pose_into_enu():
    a = make_anchor()
    rng = np.random.default_rng(2)
    cam_from_local, c = random_rotation(rng), rng.normal(0, 3, 3)
    enu, cam_from_enu, heading = a.camera_enu(quat_xyzw(cam_from_local), c.tolist())
    assert np.allclose(enu, a.to_enu(c[None])[0])
    # a world direction seen by the camera is the same whichever frame it is written in
    d_local = rng.normal(size=3)
    assert np.allclose(cam_from_local @ d_local, cam_from_enu @ (a.rotation() @ d_local), atol=1e-12)
    view = cam_from_enu[2]
    assert heading == pytest.approx(math.degrees(math.atan2(view[0], view[1])) % 360)


def test_recalibrated_applies_the_current_floors_to_the_core():
    a = make_anchor()
    core = a.raw["sigma_core"]
    with params.override(anchor_yaw_floor_deg=5.0, anchor_scale_floor=0.08, anchor_xy_floor_m=4.0,
                         anchor_tilt_floor_deg=1.5, anchor_calibration_id="refit"):
        b = a.recalibrated()
        assert b.recalibrated() is b
    assert b.calibration["id"] == "refit"
    assert b.sigma["yaw_deg"] == pytest.approx(math.hypot(core["yaw_deg"], 5.0), abs=1e-6)
    assert b.sigma["log_scale"] == pytest.approx(math.hypot(core["log_scale"], 0.08), abs=1e-6)
    assert b.sigma["east_m"] == b.sigma["north_m"] == pytest.approx(math.hypot(core["xy_m"], 4.0), abs=1e-6)
    assert b.sigma["tilt_deg"] == pytest.approx(math.hypot(core["tilt_deg"], 1.5), abs=1e-6)
    assert b.sigma["up_m"] == a.sigma["up_m"]
    assert (b.scale, b.rotation_xyzw, b.translation, b.pivot_enu) == (a.scale, a.rotation_xyzw, a.translation,
                                                                       a.pivot_enu)
    same = make_anchor(calibration=calibration())
    assert same.recalibrated() is same


def test_georegister_anchor_is_the_transform_with_every_sigma():
    rng = np.random.default_rng(3)
    s = sector_scene(rng, n=8, gps_noise=2.0)
    g = s.georegister()
    a = g.anchor
    assert g.georegistered and a.level == "full" and a.frame == "job"
    assert np.allclose(a.matrix(), g.transform, atol=1e-9)
    assert set(a.sigma) == {"yaw_deg", "log_scale", "east_m", "north_m", "up_m", "tilt_deg"}
    assert all(v is not None and v > 0 for v in a.sigma.values())
    assert a.sigma["up_m"] == params.current().anchor_z_sigma_m
    assert a.calibration["id"] == params.current().anchor_calibration_id
    assert a.sigma == floored_sigma(a.raw["sigma_core"], a.sigma["up_m"], a.calibration["floors"])
    # the pivot is the GPS-inlier cameras' weighted centre: its ENU is their weighted GPS mean
    w = 1.0 / np.maximum(s.gps_sigma, params.current().gps_sigma_min_m) ** 2 * np.ones(len(s.gps))
    assert np.allclose(a.pivot_enu[:2], (w[g.inliers] @ s.gps[g.inliers]) / w[g.inliers].sum(), atol=1e-3)
    assert a.source == {"yaw": g.yaw_source, "scale": "gps", "xy": "gps", "up": "camera-height",
                        "gravity": "camera-rights"}
    # the yaw core without its Birge inflation, with the legacy 3 deg floor, is the legacy sigma
    core_yaw = a.raw["sigma_core"]["yaw_deg"] / a.raw["birge"]
    assert math.hypot(core_yaw, 3.0) == pytest.approx(g.yaw_sigma_deg, abs=0.006)


def test_georegister_without_gps_fit_records_a_prior_only_anchor():
    rng = np.random.default_rng(4)
    s = sector_scene(rng, n=4, half_width_deg=3.0, gps_noise=12.0)
    g = s.georegister(scene_points=np.c_[rng.uniform(-10, 10, 200), rng.uniform(-2, 2, 200), rng.uniform(0, 15, 200)])
    assert not g.georegistered
    a = g.anchor
    assert a.level == "prior_only" and a.source["xy"] == "gps-median" and a.source["scale"] == "camera-to-scene"
    assert a.raw["sigma_core"]["log_scale"] == params.current().anchor_fallback_log_scale_sigma
    assert np.allclose(a.matrix(), g.transform, atol=1e-9)
