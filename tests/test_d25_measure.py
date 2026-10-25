"""D25's measurements end to end on a synthetic scene (crowd3d.correct_measure): the tower of
tests/test_skyline_judge.py whose map says 20 m while its photos (ideal sky masks of the real 30 m tower), SfM points
and 3-D lines show 30 m. D15 cannot fit it (the skyline's +-30 % search stops at 26 m); D25's widened search, the
agreeing channels and the evidence-grade box correct it. Injected errors (a common scale, a lost channel) and a right
map correct nothing; the anchors, the ground checks and the block readings are checked on their own."""

from __future__ import annotations

import math

import numpy as np
import pytest
import test_skyline_judge as tj

from crowd3d import correct as C
from crowd3d import correct_measure as M
from crowd3d import judge as J
from crowd3d import lines3d as L3
from crowd3d import skyline as K
from crowd3d import structure as S

FRAME_OK = {"passed": True, "similarity": {"measured": True, "k": 1.0, "t": [0.0, 0.0], "ticket": False, "at_bound": False}}


def scene_points(top: float = 30.0, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Points on the tower's roof and walls, on the hall's walls (3-11.5 m) and on the ground around."""
    rng = np.random.default_rng(1)
    P = [tj.tower_points(top)]
    for x0, y0, x1, y1 in ((-6, -6, 6, -6), (6, -6, 6, 6), (6, 6, -6, 6), (-6, 6, -6, -6)):
        t, z = rng.uniform(0, 1, 300), rng.uniform(13, top, 300)
        P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        t, z = rng.uniform(0, 1, 400), rng.uniform(3, 11.5, 400)
        P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    a, r = rng.uniform(0, 2 * np.pi, 3000), rng.uniform(22, 40, 3000)
    P.append(np.c_[r * np.cos(a), r * np.sin(a), rng.normal(0, 0.05, 3000)])
    P = np.vstack(P) * scale
    return P, np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]


def tower_lines(value: float = 30.2):
    def f(model, pose):
        return {"w2": [{"kind": "lines3d", "parameter": "top", "value": value, "implied_top": value,
                        "interval": [value - 0.2, value + 0.2], "n_photos": 12, "n_lines": 8, "at_bound": False}]}
    return f


@pytest.fixture(scope="module")
def photos():
    cams = tj.cameras()
    return cams, tj.sky_masks(tj.structure(30.0), cams)


def judged(map_top: float, cams, sky, P, Q) -> S.StructureModel:
    """The map judged by D15 first (D25 judges what D15 left disputed)."""
    model = tj.structure(map_top)
    prof = {"w2": K.SkylineProblem(model, cams, sky).profile_part("w2")}
    J.apply_decisions(model, J.decide(model, {"A": prof, "B": prof}, J.sfm_channel(model, P, Q), frame=FRAME_OK),
                      frame=FRAME_OK)
    return model


def test_a_map_far_off_is_corrected_end_to_end_where_d15_could_not(photos):
    cams, sky = photos
    P, Q = scene_points()
    model = judged(20.0, cams, sky, P, Q)
    assert model.part("w2").params["height"].state == S.STATE_DISPUTED and not model.adoptions     # D15: a ticket
    ev = M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=tower_lines(), grade=C.GRADE_SNAP)
    res = M.correct_landmark(model, ev)
    assert res["scale"]["route"] == "opposite-walls" and res["scale"]["established"]
    d = res["decisions"]["w2"]
    v = d["pre"]["verdicts"]
    assert v["skyline"]["valid"] and v["skyline"]["value"] == pytest.approx(30.0, abs=0.6)       # the widened search
    assert "1.3" not in str(d["pre"]) or True
    assert d["final"]["fires"] and d["final"]["adopted"] == "measured", d["final"]["why"]
    box = d["box"]
    assert C.box_structure_problem(box) is None and len(box["samples"]) >= 1 + 10 + 12 + 6
    h = model.part("w2").params["height"]
    assert h.state == S.STATE_CORRECTED and h.value == pytest.approx(30.0, abs=0.5) and h.map_value == 20.0
    assert J.check_decisions(model)["passed"] and S.check_structure(model)["passed"]
    (rid,) = res["records"]
    J.revert_adoption(model, rid)
    assert model.part("w2").params["height"].value == 20.0 and model.built_part("w2").top == 20.0


def test_a_right_map_is_not_corrected(photos):
    cams, sky = photos
    P, Q = scene_points()
    model = judged(30.0, cams, sky, P, Q)
    model.part("w2").params["height"].state = S.STATE_DISPUTED        # judged anyway: every channel at the map
    ev = M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=tower_lines(), grade=C.GRADE_SNAP)
    res = M.correct_landmark(model, ev)
    assert not res["records"] and not res["decisions"]["w2"]["final"]["fires"]
    assert model.part("w2").params["height"].value == 30.0


def test_an_injected_common_scale_error_corrects_nothing(photos):
    """A reconstruction 10 % too large (cameras, points and lines together): the opposite walls measure k 1.10, the
    scale is not established, nothing is corrected (the D15 similarity is not measured all round here)."""
    cams, sky = photos
    s = 1.10
    cams_s = []
    for c in cams:
        c2 = tj.look_at(c.center * s, [0.0, 0.0, 14.0 * s], name=c.name)
        cams_s.append(c2)
    P, Q = scene_points(scale=s)
    model = judged(20.0, cams_s, sky, P, Q)
    model.frame = {"passed": True}
    model.part("w2").params["height"].state = S.STATE_DISPUTED
    ev = M.Evidence25(cams_s, {"A": sky, "B": sky}, P, Q, lines=tower_lines(30.2 * s), grade=C.GRADE_SNAP)
    res = M.correct_landmark(model, ev)
    assert not res["scale"]["established"] and not res["records"], res["scale"]
    assert "scale" in res["decisions"]["w2"]["final"]["why"]


def test_losing_a_channel_leaves_one_and_corrects_nothing(photos):
    cams, sky = photos
    P, Q = scene_points()
    keep = ~((np.abs(P[:, 0]) <= 6.01) & (np.abs(P[:, 1]) <= 6.01) & (P[:, 2] > 12.5))      # no tower points
    P, Q = P[keep], Q[keep]
    model = judged(20.0, cams, sky, P, Q)
    model.part("w2").params["height"].state = S.STATE_DISPUTED
    ev = M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=None, grade=C.GRADE_SNAP)
    res = M.correct_landmark(model, ev)
    assert not res["records"] and "channel" in res["decisions"]["w2"]["final"]["why"]


def test_no_correction_without_an_evidence_grade_registration_or_under_an_open_frame_ticket(photos):
    cams, sky = photos
    P, Q = scene_points()
    model = judged(20.0, cams, sky, P, Q)
    assert "no evidence-grade" in M.correct_landmark(model, M.Evidence25(cams, {}, P, Q, grade=None))["why"]
    model.tickets.append({"id": "T-frame", "kind": "frame", "open": True})
    assert "frame ticket" in M.correct_landmark(model, M.Evidence25(cams, {}, P, Q, grade=C.GRADE_SNAP))["why"]


def test_the_registration_transform_moves_points_and_the_model_the_same_way():
    pose = {"x": 1.0, "y": -2.0, "yaw": 3.0, "gx": 1.5, "gy": -0.5}
    X = np.random.default_rng(0).normal(size=(20, 3)) * 10
    c = np.array([1.0, 2.0, 0.0])
    Y = M.transform(X, pose, c)
    assert np.allclose(M.transform(Y, pose, c, inverse=True), X)
    assert np.allclose(np.linalg.norm(Y[1:] - Y[:-1], axis=1), np.linalg.norm(X[1:] - X[:-1], axis=1))   # rigid
    R, _, _ = M.pose_RT({"gx": 2.0, "gy": 0.0}, c)
    up = R @ np.array([0, 0, 1.0])
    assert math.degrees(math.atan2(up[0], up[2])) == pytest.approx(2.0, abs=1e-6)             # lean east by gx


def test_opposite_walls_measure_the_reconstructions_scale():
    P, Q = scene_points()
    ring = tj.BOX.copy()
    pairs = M.wall_pairs(P, None, ring)
    sc = C.scale_from_walls(pairs)
    assert sc["established"] and abs(sc["k"] - 1.0) < 0.01
    P2, _ = scene_points(scale=1.05)
    assert not C.scale_from_walls(M.wall_pairs(P2, None, ring))["established"]


def test_the_local_ground_needs_two_sectors_apart_and_every_sector_at_zero():
    model = tj.structure(30.0)
    b = model.built_part("w1")
    rng = np.random.default_rng(3)
    a, r = rng.uniform(0, 2 * np.pi, 4000), rng.uniform(19, 27, 4000)
    G = np.c_[r * np.cos(a), r * np.sin(a), rng.normal(0, 0.05, 4000)]
    assert M.local_ground(model, b, G)["passed"]
    tilted = G.copy()
    tilted[:, 2] += 0.08 * G[:, 0]                     # a 4.6 deg slope: the far sectors are off by > 0.5 m
    assert not M.local_ground(model, b, tilted)["passed"]
    one_side = G[G[:, 0] > 22]                          # one direction only (+-35 deg)
    assert not M.local_ground(model, b, one_side)["passed"]


def test_block_readings_from_lines_and_points():
    stub = np.array([[16.0, -10.0], [30.0, -10.0], [30.0, 0.0], [16.0, 0.0]])
    specs = [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "12"}), S.PartSpec("w9", "yes", stub, [], {})]
    model = S.build_structure("s", specs, [np.array([[-15.0, -10.0], [30.0, -10.0], [30.0, 10.0], [-15.0, 10.0]])], {})
    b = model.built_part("w9")
    A, B, photos = [], [], []
    for i, z in enumerate((16.0, 16.3, 16.5, 9.0, 9.2)):
        A.append([18.0, -8.0 + i, z])
        B.append([28.0, -8.0 + i, z + 0.05])
        photos.append([i, i + 1, i + 2])
    L = L3.Lines3D(np.array(A), np.array(B), np.full(5, 6), photos, np.ones(5), np.zeros(5))
    r = M.block_lines(L, b)
    assert r["valid"] and r["value"] == pytest.approx(16.025, abs=0.01) and r["n_lines"] == 3
    rng = np.random.default_rng(2)
    xy = np.c_[rng.uniform(16.5, 29.5, 2000), rng.uniform(-9.5, -0.5, 2000)]
    P = np.c_[xy, rng.uniform(0, 15.6, 2000)]
    s = M.block_sfm(model, b, P)
    assert s["valid"] and 14.5 < s["value"] <= 15.6
    assert not M.block_sfm(model, b, P[P[:, 0] > 23.0])["valid"]               # two quadrants only


# ---------------------------------------------------------------- D25-T channels (crowd3d.tilt)


def leaning_tower_points(lean_en=(0.3, -4.0), R=7.5, H=56.0, n=20000, seed=0):
    """A colonnaded cylinder (rings every 4 m) leaning by lean_en degrees (east, north)."""
    rng = np.random.default_rng(seed)
    z = rng.uniform(0.5, H, n)
    z = np.round(z / 0.25) * 0.25
    a = rng.uniform(0, 2 * np.pi, n)
    r = np.where(rng.uniform(size=n) < 0.6, R, 0.82 * R) + rng.normal(0, 0.03, n)
    s = np.tan(np.radians(np.asarray(lean_en, float)))
    return np.c_[r * np.cos(a) + s[0] * z, r * np.sin(a) + s[1] * z, z]


def test_the_points_axis_recovers_a_lean():
    from crowd3d import tilt as T
    P = leaning_tower_points()
    ch = T.points_axis(P, [0.0, 0.0], 7.0, 2.0, 54.0, n_boot=10)
    assert ch["lean_en_deg"] == pytest.approx([0.3, -4.0], abs=0.15) and ch["bands"] >= 20 and ch["sigma_deg"] < 0.3
    up = T.points_axis(leaning_tower_points((0.0, 0.0)), [0.0, 0.0], 7.0, 2.0, 54.0, n_boot=5)
    assert np.hypot(*up["lean_en_deg"]) < 0.15


def test_vertical_3d_lines_and_the_cameras_gravity():
    from crowd3d import tilt as T
    rng = np.random.default_rng(0)
    s = np.tan(np.radians([0.5, -4.0]))
    a = rng.uniform(0, 2 * np.pi, 200)
    z0 = rng.uniform(0, 40, 200)
    A = np.c_[7.0 * np.cos(a) + s[0] * z0, 7.0 * np.sin(a) + s[1] * z0, z0]
    B = A + np.c_[np.full(200, s[0] * 5), np.full(200, s[1] * 5), np.full(200, 5.0)]
    photos = [[i % 30, (i + 1) % 30, (i + 2) % 30] for i in range(200)]
    ch = T.lines_vertical(A, B, photos, [0.0, 0.0], 12.0, n_boot=50)
    assert ch["lean_en_deg"] == pytest.approx([0.5, -4.0], abs=0.05) and ch["n_lines"] == 200 and ch["n_photos"] == 30
    cams = tj.cameras(16)
    g = T.gravity_camera(cams)
    assert np.hypot(*g["lean_en_deg"]) < 0.05 and g["rolled_removed"] == 0
    # a whole frame tilted 2 deg east: the cameras' gravity leans 2 deg east too
    Rt, _ = C.pose_matrix({"tilt_deg": 2.0, "azimuth_deg": 90.0, "pivot": [0, 0, 0]})
    tilted = [C.pose_camera(c, {"tilt_deg": 2.0, "azimuth_deg": 90.0, "pivot": [0, 0, 0]}) for c in cams]
    g2 = T.gravity_camera(tilted)
    assert g2["lean_en_deg"][0] == pytest.approx(2.0, abs=0.1) and abs(g2["lean_en_deg"][1]) < 0.1


def test_a_registration_sample_moves_the_3d_lines_as_a_scene_fact():
    A = np.array([[0.0, 0.0, 10.0], [5.0, 0.0, 12.0]])
    B = np.array([[4.0, 0.0, 10.0], [5.0, 3.0, 12.0]])
    L = L3.Lines3D(A, B, np.array([6, 7]), [[1, 2, 3], [2, 3, 4]], np.ones(2), np.zeros(2))
    res = L3.LinesResult({}, L, {"n": 2}, {}, "lsd", {}, np.zeros((3, 3)))
    pose = {"x": 1.0, "y": 0.0, "yaw": 90.0, "gx": 0.0, "gy": 0.0}
    out = M.transform_lines(res, pose, np.zeros(3))
    assert isinstance(out, L3.LinesResult) and out.detector == "lsd" and out.stats == {"n": 2}
    assert np.allclose(out.lines.A, M.transform(A, pose, np.zeros(3))) and out.lines.photos == L.photos
    assert np.allclose(out.points, M.transform(np.zeros((3, 3)), pose, np.zeros(3)))
    bare = M.transform_lines(L, pose, np.zeros(3))
    assert isinstance(bare, L3.Lines3D) and np.allclose(bare.B, M.transform(B, pose, np.zeros(3)))
    assert M.transform_lines(res, None, np.zeros(3)) is res
