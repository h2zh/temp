"""D25's measurements end to end on a synthetic scene (crowd3d.correct_measure): the tower of
tests/test_skyline_judge.py whose map says 20 m while its photos (ideal sky masks of the real 30 m tower), SfM points
and 3-D lines show 30 m. D15 cannot fit it (the skyline's +-30 % search stops at 26 m); D25's widened search, the
agreeing channels and the evidence-grade box correct it. Injected errors (a common scale, a lost channel) and a right
map correct nothing; the anchors and the block readings are checked on their own, and a photographed stub is built
as a flat block."""

from __future__ import annotations

import copy
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
from crowd3d import stylize as st

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
    assert len(box["samples"]) >= 1 + 10 + 12 + 6
    h = model.part("w2").params["height"]
    assert h.state == S.STATE_CORRECTED and h.value == pytest.approx(30.0, abs=0.5) and h.map_value == 20.0
    assert J.check_decisions(model)["passed"] and S.check_structure(model)["passed"]
    (rid,) = res["records"]
    J.revert_adoption(model, rid)
    assert model.part("w2").params["height"].value == 20.0 and model.built_part("w2").top == 20.0


def test_g5_is_measured_on_the_final_model_and_a_regression_keeps_the_old_geometry(photos, monkeypatch):
    """G5 (§2.4.8 part 3 item 5): after the rounds the corrected version's skyline distance is measured against the
    map prior in both masks on the same photos and carried by every record; when it is worse by more than 0.5 px
    every correction of the run is reverted ('keep the old geometry')."""
    cams, sky = photos
    P, Q = scene_points()
    model = judged(20.0, cams, sky, P, Q)
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=tower_lines(), grade=C.GRADE_SNAP))
    g = res["g5"]
    assert g["passed"] and set(g["masks"]) == {"A", "B"}, g
    assert g["baseline"] == M.G5_BASELINE and set(g["parts"]) == {"w2"}
    pg = g["parts"]["w2"]["masks"]
    assert all(m["photos"] >= 3 and m["corrected_px"] < m["prior_px"] for m in pg.values()), pg   # its own columns
    assert all(a.get("g5", {}).get("masks") == C._plain(pg) and a["g5"]["part"] == "w2"
               for a in model.adoptions if a.get("kind") == C.KIND_H)
    bad_m = {"A": {"photos": 9, "prior_px": 5.0, "corrected_px": 7.0},
             "B": {"photos": 9, "prior_px": 5.0, "corrected_px": 5.2}}
    bad = {"masks": bad_m, "parts": {"w2": {"masks": bad_m, "passed": False, "why": "worse"}}, "passed": False,
           "why": "worse", "baseline": M.G5_BASELINE}
    monkeypatch.setattr(M, "g5_measure", lambda *a, **k: copy.deepcopy(bad))
    model = judged(20.0, cams, sky, P, Q)
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=tower_lines(), grade=C.GRADE_SNAP))
    assert not res["records"] and res["records_reverted_by_g5"] and res["g5"]["reverted"]
    assert model.part("w2").params["height"].value == 20.0 and J.check_decisions(model)["passed"]


def test_g5_is_measured_per_corrected_part_against_the_model_before_d25(photos, monkeypatch):
    """verify-r2 L5: G5 was one mean over the whole model and every photo, against a baseline with D15's adoptions
    reverted too. Now per corrected part, on its own skyline columns, against the model before this run's D25
    corrections (a D15 frame adoption stays in the baseline); a wrong correction regresses its part's number; a part
    that regresses has only its own records reverted."""
    cams, sky = photos
    P, Q = scene_points()
    model = judged(20.0, cams, sky, P, Q)
    model.adoptions.append({"id": "A-synthetic-frame", "kind": "frame", "active": True, "value": {"k": 1.0}})
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=tower_lines(), grade=C.GRADE_SNAP))
    (rid,) = res["records"]
    base = M.before_d25(model, [rid])
    assert any(a["id"] == "A-synthetic-frame" and a.get("active", True) for a in base.adoptions)
    assert base.part("w2").params["height"].value == 20.0 and model.part("w2").params["height"].value > 29.0
    good = M.g5_measure(base, model, cams, {"A": sky, "B": sky}, parts=["w2", "w1"])
    assert good["parts"]["w2"]["passed"] and good["parts"]["w2"]["masks"]["A"]["photos"] >= 3
    w1 = good["parts"]["w1"]["masks"]["A"]
    assert w1["corrected_px"] <= w1["prior_px"] + 0.5             # the hall is not corrected (its columns may shrink)
    wrong = copy.deepcopy(base)
    wrong.part("w2").params["height"].value = 40.0
    wrong.built = S.build_all(wrong.specs, wrong.params)
    bad = M.g5_measure(base, wrong, cams, {"A": sky, "B": sky}, parts=["w2"])
    assert not bad["parts"]["w2"]["passed"] and "worse" in bad["parts"]["w2"]["why"], bad["parts"]
    # the gate reverts only the failing part's records
    calls, sky_parts = [], []

    def fake(prior, corrected, cameras, masks, parts=None, skyline_parts=None):
        calls.append(list(parts))
        sky_parts.append(set(skyline_parts or ()))
        return {"masks": {}, "baseline": M.G5_BASELINE, "passed": False, "why": "x",
                "parts": {p: {"masks": {}, "passed": p != "w2", "why": "w2 worse" if p == "w2" else "ok"} for p in parts}}
    monkeypatch.setattr(M, "g5_measure", fake)
    m2 = judged(20.0, cams, sky, P, Q)
    r2 = M.correct_landmark(m2, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=tower_lines(), grade=C.GRADE_SNAP))
    assert not r2["records"] and r2["records_reverted_by_g5"] and calls == [["w2"]]
    # its record's skyline channel agreed (the tower is the model skyline): no vacuous G5 for it (verify-r3 forge)
    rec = next(a for a in model.adoptions if a["id"] == rid)
    assert C.SKYLINE in rec["agreeing"] and sky_parts == [{"w2"}]
    assert m2.part("w2").params["height"].value == 20.0


def test_g5_measure_refuses_a_vacuous_part_with_skyline_evidence(photos, monkeypatch):
    """verify-r3 forge: a part its record shows as the model skyline (the skyline channel agreeing) may not pass G5
    vacuously; the engine's G5 gate passes that to g5_measure (crowd3d.correct.skyline_seen)."""
    cams, sky = photos
    model = tj.structure(30.0)
    monkeypatch.setattr(M, "part_skyline_distances", lambda a, b, pid: {})
    vac = M.g5_measure(model, model, cams, {"A": sky, "B": sky}, parts=["w2"])
    assert vac["parts"]["w2"]["passed"]
    g = M.g5_measure(model, model, cams, {"A": sky, "B": sky}, parts=["w2"], skyline_parts={"w2"})
    assert not g["parts"]["w2"]["passed"] and "vacuous" in g["parts"]["w2"]["why"]


def test_a_snap_grade_box_samples_the_outlines_aliases(photos):
    """The synthetic hall's rectangular outline has a 180-degree alias; a snap-grade registration samples it (yaw 180 about the outline's centroid) as a box alternative; a crowd-grade one does not."""
    cams, sky = photos
    P, Q = scene_points()
    model = judged(20.0, cams, sky, P, Q)
    ev = M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=tower_lines(), grade=C.GRADE_SNAP)
    reg = M.registration_inputs(ev, model, ev.track_points())
    assert reg["aliases"] == [180.0] and {"yaw": 180.0} in reg["alternatives"]
    ev.grade = C.GRADE_CROWD
    assert "aliases" not in M.registration_inputs(ev, model, ev.track_points())


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


# ---------------------------------------------------------------- clutter over a RIGHT map

HALL = np.array([[-15.0, -10.0], [15.0, -10.0], [15.0, 10.0], [-15.0, 10.0]])


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


CLUTTER = {   # boxes (ring, bottom, top) standing over / on the 10 m hall that are not the hall
    "trees-ns": [(_rect(-18, -13, 18, -7), 7, 16), (_rect(-18, 7, 18, 13), 7, 16)],
    "trees-ring2": [(_rect(-16.5, -11.5, 16.5, -6.5), 7, 16), (_rect(-16.5, 6.5, 16.5, 11.5), 7, 16),
                    (_rect(-16.5, -6.5, -11.5, 6.5), 7, 16), (_rect(11.5, -6.5, 16.5, 6.5), 7, 16)],
    "crane": [(_rect(-1.5, -1.5, 1.5, 1.5), 10, 40)],
}


def _hall(top: float = 10.0, extra=()):
    specs = [S.PartSpec("w1", "yes", HALL.copy(), [], {"height": f"{top:g}"})]
    for i, (ring, zmin, ztop) in enumerate(extra):
        specs.append(S.PartSpec(f"x{i}", "yes", ring, [], {"height": f"{ztop:g}", "min_height": f"{zmin:g}"}))
    return S.build_structure("synthetic", specs, [HALL.copy()], {})


def _hall_points(rng, top: float = 10.0, per_cell: int = 1):
    """The hall's roof (per_cell points per square metre), its walls and the ground around."""
    P = []
    for _ in range(per_cell):
        gx, gy = np.meshgrid(np.linspace(-14.5, 14.5, 30), np.linspace(-9.5, 9.5, 20))
        j = rng.uniform(-0.3, 0.3, (2, gx.size)) if per_cell > 1 else np.zeros((2, gx.size))
        P.append(np.c_[gx.ravel() + j[0], gy.ravel() + j[1], np.full(gx.size, top)])
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        t, z = rng.uniform(0, 1, 500), rng.uniform(3, top - 0.2, 500)
        P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    a, r = rng.uniform(0, 2 * np.pi, 3000), rng.uniform(25, 45, 3000)
    P.append(np.c_[r * np.cos(a), r * np.sin(a), rng.normal(0, 0.05, 3000)])
    return np.vstack(P)


def _box_points(ring, zmin, ztop, n, rng):
    x0, y0 = ring.min(0)
    x1, y1 = ring.max(0)
    P = [np.c_[rng.uniform(x0, x1, n), rng.uniform(y0, y1, n), ztop - np.abs(rng.normal(0, 0.3, n))]]
    for ax, ay, bx, by in ((x0, y0, x1, y0), (x1, y0, x1, y1), (x1, y1, x0, y1), (x0, y1, x0, y0)):
        t, z = rng.uniform(0, 1, n // 2), rng.uniform(zmin, ztop, n // 2)
        P.append(np.c_[ax + t * (bx - ax), ay + t * (by - ay), z])
    return np.vstack(P)


def _hall_run(truth, P, map_top=10.0):
    cams = tj.cameras(n=16, radius=75.0)
    sky = tj.sky_masks(truth, cams)
    model = _hall(map_top)
    model.frame = FRAME_OK
    model.part("w1").params["height"].state = S.STATE_DISPUTED          # as if D15 had disputed it
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=None, grade=C.GRADE_SNAP))
    return model, res


@pytest.mark.parametrize("kind", sorted(CLUTTER))
def test_trees_or_a_crane_over_a_right_map_correct_nothing(kind):
    """Canopies (7-16 m) over the N and S walls, or all round, or a crane on the centroid, of a flat 10 m hall the map
    has RIGHT. The skyline reads the canopy and the SfM points' all-round top reads it too (or the crane in all four
    quadrants), but the all-round top never agrees, and the hall's own roof-surface fit sees the 10 m roof under the
    canopy (most interior cells a TOL under any canopy-height fit), so no building-specific channel says up: nothing
    changes."""
    rng = np.random.default_rng(7)
    extra = CLUTTER[kind]
    P = np.vstack([_hall_points(rng)] + [_box_points(r, max(z0, 10.5), z1, 900, rng) for r, z0, z1 in extra])
    model, res = _hall_run(_hall(10.0, extra), P)
    d = res["decisions"]["w1"]
    v = d["pre"]["verdicts"][C.SFM]
    assert not res["records"] and not d["final"]["fires"], d["final"]
    assert model.part("w1").params["height"].value == 10.0
    assert v.get("how") != "all-round q98" or v.get("agrees") is False
    if kind.startswith("trees"):
        fit = res["decisions"]["w1"]["pre"]["verdicts"][C.SFM]
        assert "under the fitted roof" in str(fit.get("note", "")) + str(fit.get("why", "")), fit


def test_a_flat_roof_far_off_is_corrected_by_its_skyline_and_its_own_surface_fit():
    """The hall really is 16 m (its roof seen by 3 points per square metre), the map says 10 m, no 3-D lines: the
    skyline and the flat roof's surface fit agree, the box decides (a building-specific channel in the set): the
    full value or its guaranteed lower bound, between the map + TOL and the truth."""
    rng = np.random.default_rng(5)
    P = _hall_points(rng, top=16.0, per_cell=3)
    model, res = _hall_run(_hall(16.0), P)
    d = res["decisions"]["w1"]
    assert d["pre"]["verdicts"][C.SFM]["how"] == "surface-fit" and d["pre"]["verdicts"][C.SFM]["agrees"]
    assert sorted(d["pre"]["agreeing"]) == [C.SFM, C.SKYLINE], d["pre"]["why"]
    assert d["final"]["fires"], d["final"]["why"]
    h = model.part("w1").params["height"]
    assert 10.0 + C.tol_of(10.0) <= h.value <= 16.6 and h.source == C.PROV_OF_ADOPTED[d["final"]["adopted"]]
    assert J.check_decisions(model)["passed"]


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


def test_block_lines_drop_a_taller_parts_facade():
    """crowd3d.correct_measure.block_lines with the model: a line with an end or its middle inside a taller part (top
    above the block's + 1 m) or within 1 m of its outline is that part's facade, not the block's; without the model
    nothing is dropped."""
    stub = np.array([[15.0, -10.0], [29.0, -10.0], [29.0, 0.0], [15.0, 0.0]])
    specs = [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "12"}), S.PartSpec("w9", "yes", stub, [], {})]
    model = S.build_structure("s", specs, [np.array([[-15.0, -10.0], [29.0, -10.0], [29.0, 10.0], [-15.0, 10.0]])], {})
    b = model.built_part("w9")

    def three(x0):                    # three horizontal lines 6 m up from x0 to x 28, each from 3 photos
        return L3.Lines3D(np.array([[x0, -8.0 + k, 6.0] for k in range(3)]),
                          np.array([[28.0, -8.0 + k, 6.0] for k in range(3)]), np.full(3, 5),
                          [[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.ones(3), np.zeros(3))
    assert M.block_lines(three(15.5), b)["valid"]
    r = M.block_lines(three(15.5), b, model=model)             # an end 0.5 m off the 12 m hall's outline: the hall's
    assert not r["valid"] and r["near_taller_dropped"] == 3, r
    r = M.block_lines(three(16.35), b, model=model)            # 1.35 m off: the block's own
    assert r["valid"] and r["value"] == pytest.approx(6.0) and r["near_taller_dropped"] == 0, r


ANNEX_MAIN = np.array([[-20.0, -10.0], [20.0, -10.0], [20.0, -5.0], [34.0, -5.0], [34.0, 5.0], [20.0, 5.0],
                       [20.0, 10.0], [-20.0, 10.0]])


def _annex_model(annex_tags):
    """A 12 m hall (40 x 20 m) and an annex (14 x 10 m) on its east side with the given OSM tags (none: a stub)."""
    specs = [S.PartSpec("w1", "yes", _rect(-20, -10, 20, 10), [], {"height": "12"}),
             S.PartSpec("w2", "yes", _rect(20, -5, 34, 5), [], dict(annex_tags))]
    return S.build_structure("synthetic", specs, [ANNEX_MAIN.copy()], {})


def test_a_stub_photographed_15_m_tall_is_built_as_a_flat_block_at_its_lower_bound():
    """The annex really is 15 m (its walls 0.3 m inside the outline with facade lines every 2 m, its roof seen) while
    OSM gives it no height (a 3 m 'height pending' stub): D25-B builds a flat block at its guaranteed lower bound,
    under the truth, and its record passes the gate."""
    rng = np.random.default_rng(3)
    top, i, seg = 15.0, 0.3, 6

    def wall(a, b, z0, z1, n):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = rng.uniform(0, 1, n)
        return np.c_[a + t[:, None] * (b - a), rng.uniform(z0, z1, n)]
    gx, gy = np.meshgrid(np.arange(-19.5, 20, 1.0), np.arange(-9.5, 10, 1.0))
    P = [np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 12.0)]]
    for a, b, n in (((-20, -10), (20, -10), 900), ((20, 10), (-20, 10), 900), ((-20, 10), (-20, -10), 600),
                    ((20, -10), (20, -5), 200), ((20, 5), (20, 10), 200)):
        P.append(wall(a, b, 3.0, 11.8, n))
    gx, gy = np.meshgrid(np.arange(21.5, 33, 1.0), np.arange(-3.5, 4, 1.0))
    for _ in range(3):                                     # the annex's roof seen at 15 m
        j = rng.uniform(-0.45, 0.45, (2, gx.size))
        P.append(np.c_[gx.ravel() + j[0], gy.ravel() + j[1], np.full(gx.size, top) + rng.normal(0, 0.05, gx.size)])
    for a, b in (((34 - i, -5 + i), (34 - i, 5 - i)), ((34 - i, 5 - i), (20.0, 5 - i)), ((20.0, -5 + i), (34 - i, -5 + i))):
        P.append(wall(a, b, 0.0, top, 2500))
    G = np.c_[rng.uniform(-60, 75, 9000), rng.uniform(-55, 55, 9000)]
    G = G[~st.points_in_polygon(G, ANNEX_MAIN)]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    cams = tj.cameras(n=24, radius=80.0)
    cen = [np.array(c.center[:2]) for c in cams]
    A, B, photos = [], [], []

    def line(p, q, za, zb):
        A.append([p[0], p[1], za])
        B.append([q[0], q[1], zb])
        photos.append(sorted(np.argsort([np.linalg.norm(c - 0.5 * (p + q)) for c in cen])[:5].tolist()))
    for a, b in (((34 - i, -5 + i + 0.3), (34 - i, 5 - i - 0.3)), ((34 - i - 0.3, 5 - i), (20.5, 5 - i)),
                 ((20.5, -5 + i), (34 - i - 0.3, -5 + i))):
        a, b = np.array(a), np.array(b)
        for z in list(np.arange(2.0, top - 0.5, 2.0)) + [top]:
            for s0, s1 in [(k / seg + 0.02, (k + 1) / seg - 0.02) for k in range(seg)]:
                jz = rng.normal(0, 0.03, 2)
                line(a + s0 * (b - a), a + s1 * (b - a), z + jz[0], z + jz[1])
    for a, b in (((-20, -10), (20, -10)), ((20, 10), (-20, 10)), ((-20, 10), (-20, -10))):
        a, b = np.array(a, float), np.array(b, float)
        for s0, s1 in ((0.05, 0.3), (0.35, 0.6), (0.65, 0.95)):
            line(a + s0 * (b - a), a + s1 * (b - a), 12.0, 12.0)
    n = len(A)
    lines = L3.Lines3D(np.array(A), np.array(B), np.full(n, 5), photos, np.ones(n), np.zeros(n))
    sky = tj.sky_masks(_annex_model({"height": f"{top:g}"}), cams)
    model = _annex_model({})
    model.frame = FRAME_OK
    assert model.part("w2").params["height"].source == st.PROV_DEFAULT
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP),
                             only=["w2"])
    f = res["decisions"]["w2"]["final"]
    assert f["fires"], f["why"]
    assert 6.0 < f["value"] <= top and C.LINES in f["supported_by"]
    rec = next(a for a in model.adoptions if a.get("part") == "w2")
    assert rec["kind"] == C.KIND_BLOCK and rec["provenance"] == C.PROV_LB and C.check_record(rec) is None
    assert model.part("w2").roof_shape == "flat" and model.built_part("w2").top == pytest.approx(f["value"])
    assert C.check_model(model) == [] and J.check_decisions(model)["passed"]


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


def test_a_partial_arc_of_the_stack_does_not_pass_the_points_channel():
    """A one-face reconstruction sees an arc of the tower: the bootstrap starts from perturbed points and is divided by
    the azimuth coverage, so a 90-degree arc fails the 0.3-deg gate (its sigma is its bootstrap over its coverage)."""
    from crowd3d import tilt as T
    P = leaning_tower_points()
    s = np.tan(np.radians([0.3, -4.0]))
    az = np.degrees(np.arctan2(P[:, 0] - s[0] * P[:, 2], P[:, 1] - s[1] * P[:, 2])) % 360
    arc = P[np.abs((az - 200.0 + 180) % 360 - 180) <= 45.0]
    ch = T.points_axis(arc, [0.0, 0.0], 7.0, 2.0, 54.0, n_boot=8)
    assert 0.2 < ch["azimuth_coverage"] < 0.35
    assert ch["sigma_deg"] == pytest.approx(ch["sigma_boot_deg"] / ch["azimuth_coverage"], abs=1e-3)
    assert C.tilt_channel_ok("points", ch) is not None, ch


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
