"""D25-H upward changes need the part's OWN geometry at the new height (docs/STRUCTURE_FIRST.md §2.4.8 part 4, fixer
round 3 after verify-r2): things standing at or over a part whose map is RIGHT made two channels agree on their own
top, and D25 published it.

- a scaffold NET (opaque, printed) 0.4-1.3 m outside the free walls of a 10 m annex, rising to 15 m; the walls hidden,
  the flat roof unseen: the 3-D lines read the net's top as the eave (it is the densest surface, so the lines' own
  wall check took it for the wall) and the skyline reads it too. U1: an upward change needs a reading of the part's
  roof (its roof-surface fit, or the 3-D lines' apex / ridge) in the agreeing set. The lines also refuse a surface
  more than 1 m outside the edge (crowd3d.lines3d), which alone stops the 1-1.3 m net, not a closer one.
- a tree CANOPY over the whole flat roof of a 10 m hall (overhang 0-2 m, canopy points rough or smooth), walls seen
  3-7 m below it: the roof-surface fit and the skyline read the canopy. U2: the part's own walls must be seen rising
  to the eave the correction builds (crowd3d.correct_measure.wall_reading).

The controls: the annex really 15 m tall (its roof and walls seen) is corrected; the net with the annex's 10 m roof
seen is contradicted. A full ENCLOSURE (the net plus a temporary roof at its top) is the residual geometry cannot
tell from a building (doc §2.4.8 part 15/16: a scaffold mask)."""

from __future__ import annotations

import numpy as np
import pytest
import test_skyline_judge as tj

from crowd3d import correct as C
from crowd3d import correct_measure as M
from crowd3d import judge as J
from crowd3d import lines3d as L3
from crowd3d import structure as S
from crowd3d import stylize as st

FRAME_OK = {"passed": True, "similarity": {"measured": True, "k": 1.0, "t": [0.0, 0.0], "ticket": False, "at_bound": False}}

# ---------------------------------------------------------------- the scaffold net (verify-r2 attack_net.py)

HALL = np.array([[-20.0, -10.0], [20.0, -10.0], [20.0, 10.0], [-20.0, 10.0]])
ANNEX = np.array([[20.0, -5.0], [34.0, -5.0], [34.0, 5.0], [20.0, 5.0]])
MAIN = np.array([[-20.0, -10.0], [20.0, -10.0], [20.0, -5.0], [34.0, -5.0], [34.0, 5.0], [20.0, 5.0], [20.0, 10.0],
                 [-20.0, 10.0]])
TH = 0.3


def _rect(x0, y0, x1, y1):
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _nets(off):
    return [_rect(34 + off, -5 - off - TH, 34 + off + TH, 5 + off + TH),
            _rect(20.0, 5 + off, 34 + off + TH, 5 + off + TH),
            _rect(20.0, -5 - off - TH, 34 + off + TH, -5 - off)]


def _annex_model(annex_h, extra=()):
    specs = [S.PartSpec("w1", "yes", HALL.copy(), [], {"height": "12"}),
             S.PartSpec("w2", "yes", ANNEX.copy(), [], {"height": f"{annex_h:g}"})]
    for i, (ring, z0, z1) in enumerate(extra):
        specs.append(S.PartSpec(f"x{i}", "yes", ring, [], {"height": f"{z1:g}", "min_height": f"{z0:g}"}))
    return S.build_structure("synthetic", specs, [MAIN.copy()], {})


def _wall(a, b, z0, z1, n, rng):
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = rng.uniform(0, 1, n)
    return np.c_[a + t[:, None] * (b - a), rng.uniform(z0, z1, n)]


def _net_scene(kind, top=15.0, off=1.0, seed=3):
    """Points (the hall's roof and walls, the net's outer face or the annex's own walls, the ground) and real 3-D
    lines (ledgers every 2 m and the top edge on each free side, the hall's eaves), as verify-r2 attack_net.py."""
    rng = np.random.default_rng(seed)
    P = []
    gx, gy = np.meshgrid(np.arange(-19.5, 20, 1.0), np.arange(-9.5, 10, 1.0))
    P.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 12.0)])
    for a, b, n in (((-20, -10), (20, -10), 900), ((20, 10), (-20, 10), 900), ((-20, 10), (-20, -10), 600),
                    ((20, -10), (20, -5), 200), ((20, 5), (20, 10), 200)):
        P.append(_wall(a, b, 3.0, 11.8, n, rng))
    roof = {"roofseen": 10.0, "true": top, "enclosure": top}.get(kind)
    if roof is not None:                          # a roof seen: about 4 points per square metre
        gx, gy = np.meshgrid(np.arange(20.5, 34, 1.0), np.arange(-4.5, 5, 1.0))
        for _ in range(4):
            j = rng.uniform(-0.45, 0.45, (2, gx.size))
            P.append(np.c_[gx.ravel() + j[0], gy.ravel() + j[1], np.full(gx.size, roof) + rng.normal(0, 0.05, gx.size)])
    o = 0.0 if kind == "true" else off + TH
    for a, b in (((34 + o, -5 - o), (34 + o, 5 + o)), ((34 + o, 5 + o), (20.0, 5 + o)), ((20.0, -5 - o), (34 + o, -5 - o))):
        P.append(_wall(a, b, 0.0, top, 2500, rng))
    G = np.c_[rng.uniform(-60, 75, 9000), rng.uniform(-55, 55, 9000)]
    inside = st.points_in_polygon(G, MAIN)
    if kind != "true":
        for r in _nets(off - 0.2):
            inside |= st.points_in_polygon(G, np.array([[r[0, 0] - 0.2, r[0, 1] - 0.2], [r[1, 0] + 0.2, r[1, 1] - 0.2],
                                                        [r[2, 0] + 0.2, r[2, 1] + 0.2], [r[3, 0] - 0.2, r[3, 1] + 0.2]]))
    G = G[~inside]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    cams = tj.cameras(n=24, radius=80.0)
    cen = [np.array(c.center[:2]) for c in cams]
    A, B, photos = [], [], []
    lo = 0.0 if kind == "true" else off + 0.15
    sides = [((34 + lo, -5 - lo), (34 + lo, 5 + lo)), ((34 + lo, 5 + lo), (20.5, 5 + lo)), ((20.5, -5 - lo), (34 + lo, -5 - lo))]
    zs = [top] if kind == "noledger" else list(np.arange(2.0, top - 0.5, 2.0)) + [top]
    for a, b in sides:
        a, b = np.array(a), np.array(b)
        for z in zs:
            for s0, s1 in ((0.02, 0.32), (0.35, 0.65), (0.68, 0.98)):
                p, q = a + s0 * (b - a), a + s1 * (b - a)
                jz = rng.normal(0, 0.03, 2)
                A.append([p[0], p[1], z + jz[0]])
                B.append([q[0], q[1], z + jz[1]])
                photos.append(sorted(np.argsort([np.linalg.norm(c - 0.5 * (p + q)) for c in cen])[:5].tolist()))
    for a, b in (((-20, -10), (20, -10)), ((20, 10), (-20, 10)), ((-20, 10), (-20, -10))):
        a, b = np.array(a, float), np.array(b, float)
        for s0, s1 in ((0.05, 0.3), (0.35, 0.6), (0.65, 0.95)):
            p, q = a + s0 * (b - a), a + s1 * (b - a)
            A.append([p[0], p[1], 12.0])
            B.append([q[0], q[1], 12.0])
            photos.append(sorted(np.argsort([np.linalg.norm(c - 0.5 * (p + q)) for c in cen])[:5].tolist()))
    n = len(A)
    lines = L3.Lines3D(np.array(A), np.array(B), np.full(n, 5), photos, np.ones(n), np.zeros(n))
    if kind == "true":
        truth = _annex_model(top)
    else:
        extra = [(r, 0.0, top) for r in _nets(off)]
        if kind == "enclosure":
            extra.append((_rect(20.0, -5 - off - TH, 34 + off + TH, 5 + off + TH), top - 0.3, top))
        truth = _annex_model(10.0, extra)
    return P, lines, cams, tj.sky_masks(truth, cams)


def _run_annex(kind, top=15.0, off=1.0):
    P, lines, cams, sky = _net_scene(kind, top, off)
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    model = _annex_model(10.0)
    model.frame = FRAME_OK
    model.part("w2").params["height"].state = S.STATE_DISPUTED          # D15 disputed it (verify-r2: up? + lines-up)
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP),
                             only=["w2"])
    return model, res, res["decisions"]["w2"], L3.remeasure(L3.LinesResult({}, lines, {}, {}, "", {}, P), model)


def test_a_scaffold_net_over_a_right_annex_is_never_its_height():
    """verify-r2 HIGH: the net 1-1.3 m out, top 15 m, the 10 m annex's walls hidden and its roof unseen. Before: the
    lines' eave 15.01 and the skyline 15.45 agreed, the box adopted the lower bound 12.27 m and every gate passed.
    Now the 3-D lines refuse the net (its surface lies > 1 m outside the edge) and, whatever the lines say, the
    agreeing set reads no part of the annex's roof: nothing changes."""
    model, res, d, lr = _run_annex("net", 15.0, 1.0)
    assert not res["records"] and not d["final"]["fires"]
    assert model.part("w2").params["height"].value == 10.0
    eave = lr.parts["w2"].measurements.get("eave")
    assert eave is None and "outside the outline edge" in lr.parts["w2"].refused["eave"], lr.parts["w2"].refused
    assert J.check_decisions(model)["passed"]


def test_a_net_closer_than_a_metre_fools_the_lines_but_not_the_roof_rule():
    """The same net 0.4 m out: the lines cannot tell it from the wall (their eave reads its top, 15.0 m) and the
    skyline agrees; U1 refuses -- the eave and the skyline are no reading of the annex's roof."""
    model, res, d, lr = _run_annex("net", 15.0, 0.4)
    eave = lr.parts["w2"].measurements.get("eave")
    assert eave is not None and eave["value"] == pytest.approx(15.0, abs=0.2)
    pre = d["pre"]
    assert pre["verdicts"][C.LINES]["valid"] and pre["verdicts"][C.LINES]["how"] == "eave"
    assert not pre["candidate"] and "reads the part's own roof" in pre["why"], pre["why"]
    assert not res["records"] and model.part("w2").params["height"].value == 10.0


@pytest.mark.slow
@pytest.mark.parametrize("kind,off", [("noledger", 0.4), ("net", 0.7)])
def test_net_variants_change_nothing(kind, off):
    model, res, d, _ = _run_annex(kind, 15.0, off)
    assert not res["records"] and model.part("w2").params["height"].value == 10.0, d["final"]


def test_the_annex_really_taller_is_still_corrected_and_its_roof_seen_at_the_map_contradicts_the_net():
    """Controls: the annex really is 15 m (its roof seen at 15 m, its walls rising to it, its eave lines on its own
    walls): the roof-surface fit and the lines' eave agree, the walls back the eave, it is corrected. The net (0.4 m
    out: the lines read it) with the annex's own 10 m roof seen: the points within TOL of the map contradict."""
    model, res, d, lr = _run_annex("true", 15.0)
    assert d["final"]["fires"], (d["pre"]["why"], d["final"].get("why"))
    assert C.SFM in d["pre"]["agreeing"] and d["final"]["walls"]["W_base"] >= 14.0
    assert model.part("w2").params["height"].value == pytest.approx(15.0, abs=1.0) or \
        model.part("w2").params["height"].source == C.PROV_LB
    assert J.check_decisions(model)["passed"]
    model, res, d, _ = _run_annex("roofseen", 15.0, 0.4)           # the net close enough to fool the lines
    assert not res["records"] and "contradicted" in d["pre"]["why"], d["pre"]["why"]


def test_a_full_enclosure_is_the_residual_geometry_cannot_tell_apart():
    """A net with a temporary roof at its top (an enclosure): the roof-surface fit reads the temporary roof, the
    net's top edge is the eave, the net is the wall surface rising to it -- every geometric check passes on a
    structure that IS there. Recorded as what a scaffold mask (doc §4) has to stop; this test pins the behaviour so
    a change to it is seen."""
    model, res, d, _ = _run_annex("enclosure", 15.0, 0.4)
    assert d["pre"]["candidate"] and C.SFM in d["pre"]["agreeing"], d["pre"]["why"]


# ---------------------------------------------------------------- the full canopy (verify-r2 attack_canopy.py)

CHALL = np.array([[-15.0, -10.0], [15.0, -10.0], [15.0, 10.0], [-15.0, 10.0]])


def _chall(extra=()):
    specs = [S.PartSpec("w1", "yes", CHALL.copy(), [], {"height": "10"})]
    for i, (ring, z0, z1) in enumerate(extra):
        specs.append(S.PartSpec(f"x{i}", "yes", ring, [], {"height": f"{z1:g}", "min_height": f"{z0:g}"}))
    return S.build_structure("synthetic", specs, [CHALL.copy()], {})


def _run_canopy(kind, ov, ct=16.0, lo=7.0, seed=9):
    """A flat 10 m hall the map has RIGHT, entirely under a canopy (from lo to ct m in the sky masks, overhanging
    by ov m); canopy points over the whole footprint: smooth (ct - |N(0, 0.3)|) or rough (ct - 1 + 1.5 sin sin +
    N(0, 0.5)); the walls seen from 3 m to lo; no 3-D lines (the canopy hides the eave)."""
    rng = np.random.default_rng(seed)
    can = np.array([[-15 - ov, -10 - ov], [15 + ov, -10 - ov], [15 + ov, 10 + ov], [-15 - ov, 10 + ov]])
    truth = _chall([(can, lo, ct)])
    P = []
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        t, z = rng.uniform(0, 1, 500), rng.uniform(3, lo, 500)
        P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    n = 3000
    xy = np.c_[rng.uniform(-15 - ov, 15 + ov, n), rng.uniform(-10 - ov, 10 + ov, n)]
    if kind == "smooth":
        z = ct - np.abs(rng.normal(0, 0.3, n))
    else:
        z = ct - 1 + 1.5 * np.sin(2 * np.pi * xy[:, 0] / 10) * np.sin(2 * np.pi * xy[:, 1] / 10) + rng.normal(0, 0.5, n)
    P.append(np.c_[xy, z])
    a, r = rng.uniform(0, 2 * np.pi, 3000), rng.uniform(25, 45, 3000)
    P.append(np.c_[r * np.cos(a), r * np.sin(a), rng.normal(0, 0.05, 3000)])
    P = np.vstack(P)
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    cams = tj.cameras(n=16, radius=75.0)
    sky = tj.sky_masks(truth, cams)
    model = _chall()
    model.frame = FRAME_OK
    model.part("w1").params["height"].state = S.STATE_DISPUTED          # D15 disputed it (skyline-up?, sfm-up)
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=None, grade=C.GRADE_SNAP),
                             only=["w1"])
    return model, res, res["decisions"]["w1"]


def test_a_full_canopy_over_a_right_flat_roof_publishes_nothing():
    """verify-r2 MEDIUM: rough canopy points, 1 m overhang. Before: the skyline 16.46 and the surface fit 15.6
    agreed, the box [13.30, 18.85] cleared the midpoint and the FULL value 16.03 m was published. The hall's walls
    are seen only to 7 m, under the canopy: U2 refuses V and the bound alike."""
    model, res, d = _run_canopy("rough", 1.0)
    pre = d["pre"]
    assert pre["candidate"] and sorted(pre["agreeing"]) == [C.SFM, C.SKYLINE], pre["why"]
    assert not d["final"]["fires"] and "walls" in d["final"]["why"], d["final"]
    w = d["final"]["walls"]
    assert w["W_base"] < 7.5 and w["W"] < 6.5
    assert not res["records"] and model.part("w1").params["height"].value == 10.0


def test_a_smooth_canopy_that_agrees_with_the_skyline_is_refused_by_the_walls():
    """The fixer2 smooth canopy was refused only by 0.06 m of AGREE: flush with the walls (no overhang) the fit and
    the skyline agree (15.75 m); the walls stop at 7 m."""
    model, res, d = _run_canopy("smooth", 0.0)
    assert d["pre"]["candidate"], d["pre"]["why"]
    assert not d["final"]["fires"] and "walls" in d["final"]["why"], d["final"]
    assert not res["records"]


@pytest.mark.slow
@pytest.mark.parametrize("kind,ov,ct", [("rough", 0.0, 16.0), ("rough", 2.0, 16.0), ("smooth", 1.0, 16.5),
                                        ("rough", 1.0, 16.5), ("rough", 0.0, 16.5)])
def test_canopy_variants_change_nothing(kind, ov, ct):
    model, res, d = _run_canopy(kind, ov, ct)
    assert not res["records"] and model.part("w1").params["height"].value == 10.0, d["final"]


# ---------------------------------------------------------------- the wall reading itself


def test_the_wall_reading_follows_the_wall_up_to_a_gap_and_a_narrow_thing_does_not_bridge_it():
    """wall_reading: per eave edge the densest surface near it and how high its points rise contiguously; a 2.5 m
    gap stops the run; a narrow tree touching the wall (1.5 m of a 30 m edge) never covers 20 % of the edge's
    cells, so it bridges nothing; a surface found at the base pose keeps its points under a registration sample."""
    rng = np.random.default_rng(4)
    model = _chall()
    b = model.built_part("w1")
    walls = []
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        t, z = rng.uniform(0, 1, 1500), rng.uniform(2, 9.8, 1500)
        walls.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    P = np.vstack(walls)
    r, sel = M.wall_reading(model, b, P)
    assert r["value"] == pytest.approx(9.8, abs=0.1) and len(r["edges"]) == 4
    assert all(abs(e["offset_m"]) < 0.15 for e in r["edges"])
    low = P[P[:, 2] < 6.0]
    high = np.c_[rng.uniform(-15, 15, 600), np.full(600, -10.0), rng.uniform(8.5, 14.0, 600)]    # above a 2.5 m gap
    r2, _ = M.wall_reading(model, b, np.vstack([low, high]))
    assert r2["value"] == pytest.approx(6.0, abs=0.1), r2
    tree = np.c_[rng.uniform(0, 1.5, 400), np.full(400, -10.05), rng.uniform(5.5, 15.0, 400)]   # a trunk / crown
    r3, _ = M.wall_reading(model, b, np.vstack([low, tree]))
    assert r3["value"] == pytest.approx(6.0, abs=0.1), r3
    # under a registration sample the base pose's surface keeps its points (a 3 m shift would re-find another one)
    shifted = M.transform(P, {"x": 3.0, "y": 3.0, "yaw": 0.0}, np.zeros(3))
    r4, _ = M.wall_reading(model, b, shifted, sel=sel)
    assert r4["value"] == pytest.approx(r["value"], abs=1e-6)
    r5, _ = M.wall_reading(model, b, M.transform(P, {"gx": 2.0, "gy": 0.0}, np.zeros(3)), sel=sel)
    assert abs(r5["value"] - r["value"]) > 0.05                     # a tilt moves the heights
    none, _ = M.wall_reading(model, b, P[P[:, 2] > 50])
    assert none["value"] is None and none["why"]
    # a run that starts above the map's eave is not the part's wall (the wall band below the eave holds only a
    # narrow strip; a dense surface stands above it): no reading from that edge
    strip = np.c_[rng.uniform(0, 0.9, 80), np.full(80, -10.0), rng.uniform(1.5, 9.5, 80)]
    above = np.c_[rng.uniform(-15, 15, 900), np.full(900, -10.0), rng.uniform(12.0, 16.0, 900)]
    r6, _ = M.wall_reading(model, b, np.vstack([strip, above]))
    south = next(e for e in r6["edges"] if e.get("offset_m") is not None)
    assert r6["value"] is None and "starts above the map's eave" in south["why"], r6
