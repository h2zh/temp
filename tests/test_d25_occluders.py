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

- (fixer round 4, verify-r3) tree crowns over a right flat roof overhanging its walls by ~1 m, or a canopy flush
  with them: the crowns' points inside the +-0.5 m wall slab continued U2's wall run above the eave. Above the map's
  eave a height bin now counts as the wall only while it stays on it (crowd3d.correct_measure.wall_planar: thin,
  stepping no more than 0.2 m from the bin below, covering the edge like the wall below); the cells of an edge that
  a taller part abuts are not its wall.
- (fixer round 4, verify-r3) the same net just INSIDE the outline of a 'height pending' stub (OSM gives no height,
  the annex is 6 m): the 3-D lines' highest horizontal cluster (ledgers, top edge, wholly inside the outline) and the
  skyline built a D25-B block >= 10.5 m. U1 for blocks: the block's own roof read over its interior (cells >= 2 m
  inside its outline, crowd3d.correct_measure.block_roof) must agree with H.

The controls: the annex really 15 m tall (its roof and walls seen) is corrected; the net with the annex's 10 m roof
seen is contradicted; a stub really 15 m tall (its roof seen) is still built. A full ENCLOSURE (the net plus a temporary roof at its top) is the residual geometry cannot
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


# ---------------------------------------------------------------- the scaffold net around a stub (verify-r3 attack_block_net.py)


def _block_model(annex_tags, extra=(), gap=False):
    specs = [S.PartSpec("w1", "yes", HALL.copy(), [], {"height": "12"})]
    if not (gap and not annex_tags):
        specs.append(S.PartSpec("w2", "yes", ANNEX.copy(), [], dict(annex_tags)))
    for i, (ring, z0, z1) in enumerate(extra):
        specs.append(S.PartSpec(f"x{i}", "yes", ring, [], {"height": f"{z1:g}", "min_height": f"{z0:g}"}))
    return S.build_structure("synthetic", specs, [MAIN.copy()], {})


def _block_scene(kind, inset=0.2, top=15.0, true_h=6.0, seg=6, seed=3):
    """verify-r3 attack_block_net.py: 'net' -- a 0.3 m net `inset` m inside the stub annex's three free edges, 0 to
    `top` m, its outer face densely textured, ledgers every 2 m and the top edge as real 3-D lines in `seg` pieces
    per side; the annex (truly true_h m, no OSM height) unseen behind it. 'roofseen' -- the annex's true roof seen at
    true_h inside the net. 'true' -- no net: the annex is `top` m, its walls `inset` m inside the outline carrying the
    same facade lines, its roof seen."""
    rng = np.random.default_rng(seed)
    i = inset
    P = []
    gx, gy = np.meshgrid(np.arange(-19.5, 20, 1.0), np.arange(-9.5, 10, 1.0))
    P.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 12.0)])
    for a, b, n in (((-20, -10), (20, -10), 900), ((20, 10), (-20, 10), 900), ((-20, 10), (-20, -10), 600),
                    ((20, -10), (20, -5), 200), ((20, 5), (20, 10), 200)):
        P.append(_wall(a, b, 3.0, 11.8, n, rng))
    roof = {"roofseen": true_h, "true": top}.get(kind)
    if roof is not None:
        gx, gy = np.meshgrid(np.arange(21.5, 33, 1.0), np.arange(-3.5, 4, 1.0))
        for _ in range(3):
            j = rng.uniform(-0.45, 0.45, (2, gx.size))
            P.append(np.c_[gx.ravel() + j[0], gy.ravel() + j[1], np.full(gx.size, roof) + rng.normal(0, 0.05, gx.size)])
    for a, b in (((34 - i, -5 + i), (34 - i, 5 - i)), ((34 - i, 5 - i), (20.0, 5 - i)), ((20.0, -5 + i), (34 - i, -5 + i))):
        P.append(_wall(a, b, 0.0, top, 2500, rng))
    G = np.c_[rng.uniform(-60, 75, 9000), rng.uniform(-55, 55, 9000)]
    G = G[~st.points_in_polygon(G, MAIN)]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    cams = tj.cameras(n=24, radius=80.0)
    cen = [np.array(c.center[:2]) for c in cams]
    A, B, photos = [], [], []
    sides = [((34 - i, -5 + i + 0.3), (34 - i, 5 - i - 0.3)), ((34 - i - 0.3, 5 - i), (20.5, 5 - i)),
             ((20.5, -5 + i), (34 - i - 0.3, -5 + i))]
    for a, b in sides:
        a, b = np.array(a), np.array(b)
        for z in list(np.arange(2.0, top - 0.5, 2.0)) + [top]:
            for s0, s1 in [(k / seg + 0.02, (k + 1) / seg - 0.02) for k in range(seg)]:
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
        truth = _block_model({"height": f"{top:g}"})
    else:
        nets = [_rect(34 - i - TH, -5 + i, 34 - i, 5 - i), _rect(20.0, 5 - i - TH, 34 - i, 5 - i),
                _rect(20.0, -5 + i, 34 - i, -5 + i + TH)]
        truth = _block_model({"height": f"{true_h:g}"}, [(r, 0.0, top) for r in nets])
    return P, lines, cams, tj.sky_masks(truth, cams)


def _run_block(kind, gap=False, **kw):
    P, lines, cams, sky = _block_scene(kind, **kw)
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    model = _block_model({}, gap=gap)
    model.frame = FRAME_OK
    pid = next(p.id for p in model.specs if p.kind == "remainder") if gap else "w2"
    assert model.part(pid).params["height"].source == st.PROV_DEFAULT
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP),
                             only=[pid])
    return model, res, res["decisions"][pid], pid


def test_a_scaffold_net_inside_a_stubs_outline_builds_no_block():
    """verify-r3 HIGH: net 0.2 m inside, top 15 m, six line pieces per side. Before: lines3d 14.96 + skyline 15.0
    (box minima 10.54 / 10.84) built a block >= 10.54 m 'photo-corrected (auto, lower bound)' on a 6 m annex (the
    3 m stub was nearer the truth). Nothing is seen inside the net: no interior roof cell, no block."""
    model, res, d, pid = _run_block("net")
    f = d["final"]
    assert f["H"] > 6.0 + 1.0, f                                  # what the old rule would have built (wrong)
    assert not f["fires"] and "own roof" in f["why"] and f["roof_check"]["roof_cells"] == 0, f["why"]
    assert not res["records"] and model.part(pid).params["height"].value == 3.0


def test_the_annexs_true_roof_seen_inside_the_net_contradicts_the_block():
    """verify-r3 'roofseen': the annex's true 6 m roof seen inside the net (cells >= 2 m inside read 6 m) -- before,
    the block was built >= 10.50 m anyway."""
    model, res, d, pid = _run_block("roofseen")
    f = d["final"]
    assert not f["fires"] and "under H" in f["why"] and f["roof_check"]["below_share"] == 1.0, f["why"]
    assert not res["records"]


def test_a_stub_really_taller_with_its_roof_seen_is_still_built():
    """The control: no net, the annex truly 15 m (walls 0.3 m inside the outline with the same facade lines, roof
    seen): the block is built at its guaranteed lower bound, under the truth, and its record passes the gate."""
    model, res, d, pid = _run_block("true", inset=0.3)
    f = d["final"]
    assert f["fires"], f["why"]
    assert 6.0 < f["value"] <= 15.0 and f["roof_check"]["median"] == pytest.approx(15.0, abs=0.2)
    rec = next(a for a in model.adoptions if a.get("part") == pid)
    assert rec["kind"] == C.KIND_BLOCK and rec["roof"]["roof_tops"] and C.check_record(rec) is None
    assert C.check_model(model) == [] and J.check_decisions(model)["passed"]


@pytest.mark.slow
@pytest.mark.parametrize("kind,kw", [("net", {"seg": 4}), ("net", {"seg": 5}), ("net", {"inset": 0.35}),
                                     ("net", {"top": 12.0, "seg": 5}), ("net", {"gap": True}),
                                     ("net", {"inset": 2.5}), ("true", {"inset": 0.3, "gap": True})])
def test_block_net_variants(kind, kw):
    """verify-r3 variants (before: 10.61 / 10.51 / 10.52 / 7.88 / the gap 10.54, all on a 6 m annex) and a net 2.5 m
    inside the outline (its own cells interior, but vertical): no block; the true 15 m block over a gap is built."""
    kw = dict(kw)
    model, res, d, pid = _run_block(kind, gap=kw.pop("gap", False), **kw)
    if kind == "net":
        assert not res["records"] and model.part(pid).params["height"].value == 3.0, d["final"]
    else:
        assert d["final"]["fires"] and 6.0 < d["final"]["value"] <= 15.0, d["final"]


# ---------------------------------------------------------------- tree crowns over the eave (verify-r3 attack_crowns.py)


def _crowns_points(kind, ov=0.3, lo=9.8, ct=16.0, R=6.0, ZC=12.0, seed=11):
    """verify-r3 attack_crowns.py: the flat 10 m hall (30 x 20 m) the map has RIGHT, its walls seen 1-10 m (exactly
    on the outline), its roof unseen; 'over': six spherical crowns (R 6 m, centres 12 m) over the roof overhanging
    the walls by ~1 m, points on their upper hemisphere and sides (noise 0.3 m); 'flush': a box canopy from lo to ct
    hanging ov m out, rough top points and side points (ov + N(0, 0.3) m out). Returns points and the canopies."""
    rng = np.random.default_rng(seed)
    P, extra = [], []
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        t, z = rng.uniform(0, 1, 1500), rng.uniform(1, 10, 1500)
        P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    if kind == "over":
        for cx in (-10.0, 0.0, 10.0):
            for cy in (-5.0, 5.0):
                v = rng.normal(size=(1500, 3))
                v /= np.linalg.norm(v, axis=1)[:, None]
                v = v[v[:, 2] > -0.3]
                P.append(np.c_[cx + R * v[:, 0], cy + R * v[:, 1], ZC + 0.8 * R * v[:, 2]] + rng.normal(0, 0.3, (len(v), 3)))
                a = np.radians(np.arange(0, 360, 45) + 22.5)
                extra.append((np.c_[cx + R * np.cos(a), cy + R * np.sin(a)], ZC - 0.8 * R * 0.3, ZC + 0.8 * R))
    else:
        can = np.array([[-15 - ov, -10 - ov], [15 + ov, -10 - ov], [15 + ov, 10 + ov], [-15 - ov, 10 + ov]])
        extra.append((can, lo, ct))
        xy = np.c_[rng.uniform(-15 - ov, 15 + ov, 3000), rng.uniform(-10 - ov, 10 + ov, 3000)]
        z = ct - 1 + 1.5 * np.sin(2 * np.pi * xy[:, 0] / 10) * np.sin(2 * np.pi * xy[:, 1] / 10) + rng.normal(0, 0.5, 3000)
        P.append(np.c_[xy, z])
        for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
            a, b = np.array([x0, y0], float), np.array([x1, y1], float)
            u = (b - a) / np.linalg.norm(b - a)
            nrm = np.array([u[1], -u[0]])
            t = rng.uniform(0, 1, 800)
            xy = a + t[:, None] * (b - a) + (ov + rng.normal(0, 0.3, 800))[:, None] * nrm
            P.append(np.c_[xy, rng.uniform(lo, ct - 1.0, 800)])
    a, r = rng.uniform(0, 2 * np.pi, 3000), rng.uniform(25, 45, 3000)
    P.append(np.c_[r * np.cos(a), r * np.sin(a), rng.normal(0, 0.05, 3000)])
    return np.vstack(P), extra


def _eave_lines(cams, z=10.0):
    cen = [np.array(c.center[:2]) for c in cams]
    A, B, ph = [], [], []
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        a, b = np.array([x0, y0], float), np.array([x1, y1], float)
        for s0, s1 in ((0.05, 0.3), (0.35, 0.6), (0.65, 0.95)):
            p, q = a + s0 * (b - a), a + s1 * (b - a)
            A.append([p[0], p[1], z])
            B.append([q[0], q[1], z])
            ph.append(sorted(np.argsort([np.linalg.norm(c - 0.5 * (p + q)) for c in cen])[:5].tolist()))
    return L3.Lines3D(np.array(A), np.array(B), np.full(len(A), 5), ph, np.ones(len(A)), np.zeros(len(A)))


def _run_crowns(kind, eavelines=False, **kw):
    P, extra = _crowns_points(kind, **kw)
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    cams = tj.cameras(n=16, radius=75.0)
    sky = tj.sky_masks(_chall(extra), cams)
    model = _chall()
    model.frame = FRAME_OK
    model.part("w1").params["height"].state = S.STATE_DISPUTED          # D15 disputes it (verify-r3: ticket only)
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q,
                                                 lines=_eave_lines(cams) if eavelines else None, grade=C.GRADE_SNAP),
                             only=["w1"])
    return model, res, res["decisions"]["w1"]


def test_crowns_overhanging_the_eave_do_not_extend_the_wall():
    """verify-r3 MEDIUM: before, the surface fit 16.0 and the skyline 16.91 agreed (U1 satisfied) and the crowns'
    points inside the wall slab carried U2's run to W_base 15.36 / W 13.86: a lower bound 13.965 m was published on
    a hall the map has right at 10 m. Above the eave the crown bins are no thin surface on the wall (MAD 0.2-0.47 m):
    the walls read 10 m, and nothing is published."""
    model, res, d = _run_crowns("over")
    f = d["final"]
    assert d["pre"]["candidate"] and C.SFM in d["pre"]["agreeing"], d["pre"]["why"]
    assert not f["fires"] and "walls" in f["why"], f
    assert f["walls"]["W_base"] <= 10.5 and not res["records"] and model.part("w1").params["height"].value == 10.0


def test_a_rough_canopy_hanging_out_beside_the_walls_does_not_extend_them():
    """verify-r3: a box canopy from 9.8 m hanging 0.3 m out (its side points 0.3 + N(0, 0.3) m out) gave the FULL
    value 16.05 m before; its bins above the eave are no thin surface on the wall."""
    model, res, d = _run_crowns("flush", ov=0.3, lo=9.8)
    assert not d["final"]["fires"] and d["final"]["walls"]["W_base"] < 13.0, d["final"]
    assert not res["records"]


@pytest.mark.slow
@pytest.mark.parametrize("kind,kw", [("over", {"eavelines": True}), ("flush", {"ov": 0.0, "lo": 10.0}),
                                     ("over", {"R": 7.0, "ZC": 13.0})])
def test_crown_variants_change_nothing(kind, kw):
    """'over' with the hall's true 10 m eave lines (the 3-D lines' own-wall check refuses them: nothing contradicts),
    a canopy flush with the walls (before: the full value 15.85 m), larger crowns."""
    model, res, d = _run_crowns(kind, **kw)
    assert not res["records"] and model.part("w1").params["height"].value == 10.0, d["final"]


def _hall_walls(rng, z1=10.0, lean=None):
    P = []
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        a, b = np.array([x0, y0], float), np.array([x1, y1], float)
        u = (b - a) / np.linalg.norm(b - a)
        nrm = np.array([u[1], -u[0]])
        t, z = rng.uniform(0, 1, 2500), rng.uniform(1, z1, 2500)
        off = rng.normal(0, 0.05, 2500) + (0.0 if lean is None else lean * np.clip(z - 10.0, 0, None))
        P.append(np.c_[a + t[:, None] * (b - a) + off[:, None] * nrm, z])
    return np.vstack(P)


def test_above_the_eave_only_bins_on_the_wall_extend_it():
    """wall_planar's three conditions each on its own (crowd3d.correct WALL_ABOVE_*): a wall really rising to 16 m
    above the map's 10 m eave -- straight, or leaning 0.07 m per metre (0.4 m over the top 6 m, as the Chorhalle's
    north wall above its map eave) -- reads 16 m; a thin surface stepping 0.35 m out at the eave (a hoarding, a
    canopy's edge), a 6 m wide strip rising flush from a 30 m wall (half the wall's cover), points filling the slab
    (crowns) stop the run at the eave."""
    rng = np.random.default_rng(8)
    model = _chall()
    b = model.built_part("w1")
    for lean in (None, 0.07):
        r, _ = M.wall_reading(model, b, _hall_walls(rng, 16.0, lean))
        assert r["value"] == pytest.approx(16.0, abs=0.15), (lean, r)
    low = _hall_walls(rng, 10.0)
    south = lambda r: next(e for e in r["edges"] if e["edge"] == 0)
    step = np.c_[rng.uniform(-15, 15, 3000), np.full(3000, -10.35) + rng.normal(0, 0.03, 3000), rng.uniform(10, 16, 3000)]
    r, _ = M.wall_reading(model, b, np.vstack([low, step]))
    assert south(r)["top_m"] <= 10.0 and r["value"] <= 10.0, r
    strip = np.c_[rng.uniform(-3, 3, 1500), np.full(1500, -10.0) + rng.normal(0, 0.05, 1500), rng.uniform(10, 16, 1500)]
    r, _ = M.wall_reading(model, b, np.vstack([low, strip]))
    assert south(r)["top_m"] <= 10.0, r
    fill = np.c_[rng.uniform(-15, 15, 3000), np.full(3000, -10.0) + rng.uniform(-0.5, 0.5, 3000), rng.uniform(10, 16, 3000)]
    r, _ = M.wall_reading(model, b, np.vstack([low, fill]))
    assert south(r)["top_m"] <= 10.0 and south(r)["off_plane_bins"], r
    # on a rough wall (offsets N(0, 0.35) m below the eave) the spread allowed is capped at 0.25 m: points filling
    # +-0.4 m above the eave (MAD about 0.3 m) are still no wall
    sw = np.c_[rng.uniform(-15, 15, 4000), -10.0 + rng.normal(0, 0.35, 4000), rng.uniform(1, 10, 4000)]
    fill4 = np.c_[rng.uniform(-15, 15, 3000), np.full(3000, -10.0) + rng.uniform(-0.4, 0.4, 3000), rng.uniform(10, 16, 3000)]
    r, _ = M.wall_reading(model, b, np.vstack([sw, fill4]))
    assert south(r)["top_m"] <= 10.0, r
    # a flat planar enclosure flush with the wall above the eave is the residual: a thin surface on the wall plane
    flush = np.c_[rng.uniform(-15, 15, 3000), np.full(3000, -10.0) + rng.normal(0, 0.05, 3000), rng.uniform(10, 16, 3000)]
    r, _ = M.wall_reading(model, b, np.vstack([low, flush]))
    assert south(r)["top_m"] == pytest.approx(16.0, abs=0.15)


# ---------------------------------------------------------------- a taller part beside an edge (verify-r3 probe_partial_abut.py)


def _abut_model(tower):
    """The 10 m annex (the main outline) and a 30 m tower beside it."""
    A = np.array([[0.0, 0.0], [14.0, 0.0], [14.0, 10.0], [0.0, 10.0]])
    specs = [S.PartSpec("a", "yes", A, [], {"height": "10"}), S.PartSpec("t", "yes", tower, [], {"height": "30"})]
    return S.build_structure("abut", specs, [A.copy()], {})


def _w(rng, a, b, z0, z1, n):
    a, b = np.array(a, float), np.array(b, float)
    t = rng.uniform(0, 1, n)
    return np.c_[a + t[:, None] * (b - a), rng.uniform(z0, z1, n)]


def test_the_cells_a_taller_part_abuts_are_not_the_parts_wall():
    """verify-r3 LOW: a 10 m annex whose north edge a 30 m tower abuts over 6 of its 14 m (the edge's midpoint free,
    so the old 'hidden' test kept the edge): wall_reading returned 29.99 m, climbing the tower's south face. The
    tower's cells of the edge are dropped: 10 m. A tower across a 2 m alley the same; an edge hidden over its whole
    length is not read at all."""
    rng = np.random.default_rng(1)
    annex = [_w(rng, (0, 0), (14, 0), 1, 10, 1500), _w(rng, (14, 0), (14, 10), 1, 10, 1000),
             _w(rng, (14, 10), (6, 10), 1, 10, 900), _w(rng, (0, 10), (0, 0), 1, 10, 1000)]
    T = np.array([[0.0, 10.0], [6.0, 10.0], [6.0, 16.0], [0.0, 16.0]])
    tower = [_w(rng, (6, 10), (0, 10), 10, 30, 1500), _w(rng, (6, 10), (6, 16), 0, 30, 1500),
             _w(rng, (6, 16), (0, 16), 0, 30, 1500), _w(rng, (0, 16), (0, 10), 0, 30, 1500)]
    model = _abut_model(T)
    r, _ = M.wall_reading(model, model.built_part("a"), np.vstack(annex + tower))
    north = next(e for e in r["edges"] if e.get("cells_covered"))
    assert north["cells_covered"] == 6 and north["top_m"] <= 10.0 and r["value"] == pytest.approx(10.0, abs=0.1), r
    # across a 2 m alley (the tower's face 2 m out of the north edge, over 6 m of it): its cells are not the wall
    T2 = np.array([[0.0, 12.0], [6.0, 12.0], [6.0, 18.0], [0.0, 18.0]])
    tower2 = [_w(rng, (6, 12), (0, 12), 0, 30, 4000)]
    model2 = _abut_model(T2)
    r2, _ = M.wall_reading(model2, model2.built_part("a"), np.vstack(annex[:2] + [_w(rng, (14, 10), (6, 10), 1, 10, 300)]
                                                                  + annex[3:] + tower2))
    assert r2["value"] == pytest.approx(10.0, abs=0.1), r2
    # every eave edge behind taller parts: nothing in view
    ring = np.array([[0.0, 0.0], [14.0, 0.0], [14.0, 10.0], [0.0, 10.0]])
    around = [np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 0.0], [-1.0, 0.0]]),
              np.array([[14.0, -1.0], [15.0, -1.0], [15.0, 11.0], [14.0, 11.0]]),
              np.array([[-1.0, 10.0], [15.0, 10.0], [15.0, 11.0], [-1.0, 11.0]]),
              np.array([[-1.0, -1.0], [0.0, -1.0], [0.0, 11.0], [-1.0, 11.0]])]
    specs = [S.PartSpec("a", "yes", ring, [], {"height": "10"})] + \
        [S.PartSpec(f"t{k}", "yes", q, [], {"height": "30"}) for k, q in enumerate(around)]
    m3 = S.build_structure("ring", specs, [np.array([[-1.0, -1.0], [15.0, -1.0], [15.0, 11.0], [-1.0, 11.0]])], {})
    walls = np.vstack([_w(rng, (0, 0), (14, 0), 0, 30, 1500), _w(rng, (14, 0), (14, 10), 0, 30, 1500),
                       _w(rng, (14, 10), (0, 10), 0, 30, 1500), _w(rng, (0, 10), (0, 0), 0, 30, 1500)])
    r3, _ = M.wall_reading(m3, m3.built_part("a"), walls)
    assert r3["value"] is None and r3["edges"] == [] and "in view" in r3["why"], r3
