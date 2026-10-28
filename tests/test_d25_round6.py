"""D25 fixer round 6 (docs/STRUCTURE_FIRST.md §2.4.8 part 19, after verify-r5): the ways a correction was still
published on a part the map has RIGHT, each now refused, and the guards that refuse them.

- A RAISED SECTION and a downward change (verify-r5 attack_raised.py): a 50 x 50 m hall the map has right at 29.5 m
  (its top is an 8 x 8 m penthouse; parapet 22 m, main roof 21 m, seen from four elevated photos). The roof-surface
  fit read the main roof with only 4 % of the interior cells above it, the all-round top (3rd-highest quadrant) read
  the roof too, and the full value 21.79 m was published. Going down, the highest level a contiguous patch of >= 4
  cells over the footprint reaches (crowd3d.correct_measure.raised_patch) must not lie above V + AGREE.
- A ring of VERTICAL cells as the fit's interior support (verify-r5 attack_nc5.py two nets): a scaffold net on the
  facades plus a rooftop net ring 2.5 m inside a map-right 10 m annex; the fit stood on the ring's own cells (22 of 22
  interior cells) and the full value 14.90 m was published. The fit's interior inliers must hold an AREA (a cell whose
  8 neighbours are all seen), and at most 25 % of the interior cells may hold points more than TOL under the fit
  ('volume share': a crown's flanks or a net's lower part seen inside the volume a roof at the fit would close; the
  plane-tree case, LB 12.457 m, falls to it).
- A block on a net whose base is hidden above H - TOL (verify-r5 attack_block_net2.py lo8 / lo9 / lo10): the net's
  22 cells, a ring, were the block's 'roof'. A block needs a roof cell whose 8 neighbours all hold points; crowns from
  trees outside over a strip of the interior (cr5-out15) fall to it too.
- Forgeries the gate passed (verify-r5 forge_r5*.py): the fit's interior counts deleted, a withheld id with no outline,
  a block's base moved consistently.
- Guard tests for the seven mutants that survived verify-r5 (R2, R4, R5, R7, R8, R10, R15)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import test_d25 as td
import test_d25_occluders as TO
import test_d25_round5 as R5
import test_skyline_judge as tj

from crowd3d import correct as C
from crowd3d import correct_measure as M
from crowd3d import judge as J
from crowd3d import lines3d as L3
from crowd3d import structure as S
from crowd3d import stylize as st

FRAME_OK = TO.FRAME_OK
SK, SF, LN = C.SKYLINE, C.SFM, C.LINES

# ---------------------------------------------------------------- a raised section and a downward change


def _raised_scene(top=29.5, eave=22.0, roof=21.0, sx=8.0, sy=8.0, px=5.0, py=5.0, hx=25.0, hy=25.0, dist=60.0,
                  ncam=20, nel=4, elz=45.0, f=200.0, seed=3):
    """verify-r5 attack_raised.py: a 2hx x 2hy hall, walls and parapet to `eave`, a flat main roof at `roof` (seen from
    `nel` elevated photos), a raised section sx x sy at (px, py) rising to `top` (its top and walls seen from them);
    the map: height=top (the true top), no roof:shape. sx = sy = 0.2: the control, really a flat roof at the parapet
    (the map is wrong). Cornice lines at 8, 15 m and the parapet on every facade."""
    rng = np.random.default_rng(seed)
    hall = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    sec = np.array([[px - sx / 2, py - sy / 2], [px + sx / 2, py - sy / 2], [px + sx / 2, py + sy / 2],
                    [px - sx / 2, py + sy / 2]])

    def model(specs):
        return S.build_structure("synthetic", specs, [hall.copy()], {})
    truth = model([S.PartSpec("w1", "yes", hall.copy(), [], {"height": f"{eave:g}"}),
                   S.PartSpec("w2", "yes", sec.copy(), [], {"height": f"{top:g}", "min_height": f"{eave:g}"})])
    P = []
    for (x0, y0), (x1, y1) in zip(hall, np.roll(hall, -1, 0)):
        n = int(60 * math.hypot(x1 - x0, y1 - y0))
        t, z = rng.uniform(0, 1, n), rng.uniform(1, eave, n)
        P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    if nel:
        n = int(2 * 4 * hx * hy)
        xy = np.c_[rng.uniform(-hx + 0.3, hx - 0.3, n), rng.uniform(-hy + 0.3, hy - 0.3, n)]
        xy = xy[~st.points_in_polygon(xy, sec)]
        P.append(np.c_[xy, roof + rng.normal(0, 0.05, len(xy))])
        for (x0, y0), (x1, y1) in zip(hall, np.roll(hall, -1, 0)):
            n = int(10 * math.hypot(x1 - x0, y1 - y0))
            t = rng.uniform(0, 1, n)
            P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), rng.uniform(roof, eave, n)])
        n = int(3 * sx * sy)
        P.append(np.c_[rng.uniform(sec[0, 0], sec[1, 0], n), rng.uniform(sec[0, 1], sec[2, 1], n),
                       top + rng.normal(0, 0.05, n)])
        for (x0, y0), (x1, y1) in zip(sec, np.roll(sec, -1, 0)):
            n = int(40 * math.hypot(x1 - x0, y1 - y0))
            t = rng.uniform(0, 1, n)
            P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), rng.uniform(roof, top, n)])
    G = np.c_[rng.uniform(-90, 90, 9000), rng.uniform(-90, 90, 9000)]
    G = G[~st.points_in_polygon(G, hall)]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    angs = np.linspace(0, 2 * math.pi, ncam, endpoint=False)
    cams = [tj.look_at([dist * math.cos(a), dist * math.sin(a), 1.6],
                       [0.4 * dist * math.cos(a), 0.4 * dist * math.sin(a), eave * 0.85], f=f, name=f"c{i}")
            for i, a in enumerate(angs)]
    for i in range(nel):
        a = math.radians(30 + 10 * i)
        cams.append(tj.look_at([100 * math.cos(a), 100 * math.sin(a), elz], [0, 0, roof], name=f"el{i}"))
    sky = tj.sky_masks(truth, cams)
    cen = [np.array(c.center[:2]) for c in cams[:ncam]]
    A, B, photos = [], [], []
    for (x0, y0), (x1, y1) in zip(hall, np.roll(hall, -1, 0)):
        a, b = np.array([x0, y0], float), np.array([x1, y1], float)
        for z in (8.0, 15.0, eave):
            for s0, s1 in ((0.04, 0.3), (0.36, 0.63), (0.69, 0.96)):
                p, q = a + s0 * (b - a), a + s1 * (b - a)
                jz = rng.normal(0, 0.03, 2)
                A.append([p[0], p[1], z + jz[0]])
                B.append([q[0], q[1], z + jz[1]])
                photos.append(sorted(np.argsort([np.linalg.norm(c - 0.5 * (p + q)) for c in cen])[:5].tolist()))
    n = len(A)
    lines = L3.Lines3D(np.array(A), np.array(B), np.full(n, 5), photos, np.ones(n), np.zeros(n))
    m = model([S.PartSpec("w1", "yes", hall.copy(), [], {"height": f"{top:g}"})])
    m.frame = FRAME_OK
    m.part("w1").params["height"].state = S.STATE_DISPUTED       # forced disputed (D15 adopts the parapet: its rule)
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    return m, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP), P


def _run_raised(**kw):
    model, ev, P = _raised_scene(**kw)
    res = M.correct_landmark(model, ev, only=["w1"])
    return model, res, res["decisions"]["w1"]


def test_raised_patch_reads_the_highest_level_a_contiguous_patch_reaches():
    """crowd3d.correct_measure.raised_patch: the highest level a contiguous (8-neighbour) patch of >= 4 1 m cells
    reaches; a single high point or a one-cell chimney is not a patch, a 2 x 2 m one is."""
    rng = np.random.default_rng(0)
    gx, gy = np.meshgrid(np.arange(0.5, 20, 1.0), np.arange(0.5, 20, 1.0))
    roof = np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 21.0)]
    lo = (0.0, 0.0)
    assert M.raised_patch(roof, lo)["raised_top"] == pytest.approx(21.0)
    spike = np.vstack([roof, [[10.5, 10.5, 29.5]]])                                       # one high point
    assert M.raised_patch(spike, lo)["raised_top"] == pytest.approx(21.0)
    three = np.vstack([roof, [[10.5, 10.5, 29.5], [11.5, 10.5, 29.4], [11.5, 11.5, 29.3]]])  # three cells: no patch
    assert M.raised_patch(three, lo)["raised_top"] == pytest.approx(21.0)
    four = np.vstack([three, [[10.5, 11.5, 29.2]]])                                        # 2 x 2 m: a patch
    assert M.raised_patch(four, lo)["raised_top"] == pytest.approx(29.2)
    diag = np.vstack([roof, [[2.5 + k, 2.5 + k, 29.5] for k in range(4)]])                 # 8-neighbours join
    assert M.raised_patch(diag, lo)["raised_top"] == pytest.approx(29.5)
    apart = np.vstack([roof, [[2.5, 2.5, 29.5], [4.5, 2.5, 29.5], [6.5, 2.5, 29.5], [8.5, 2.5, 29.5]]])   # not joined
    assert M.raised_patch(apart, lo)["raised_top"] == pytest.approx(21.0)
    assert M.raised_patch(roof[:3], lo)["raised_top"] is None                              # too little seen
    r = M.raised_patch(four, lo, k=5)
    assert r["raised_top"] == pytest.approx(21.0) and r["raised_k"] == 5
    # on the grid shifted by half a cell too, the lower: a thin mast on a cell corner splits into four cells of one
    # grid only (verify-r5's 0.2 m 'section' control)
    mast = np.vstack([roof, np.c_[rng.uniform(9.9, 10.1, 40), rng.uniform(9.9, 10.1, 40), rng.uniform(21, 29.5, 40)]])
    r = M.raised_patch(mast, lo)
    assert r["raised_grids"][0] > 26.0 and r["raised_top"] == pytest.approx(21.0), r
    # the all-round measurement carries it: a 2 x 2 m block on a roof seen at 3 points per m^2
    hall = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    model = S.build_structure("synthetic", [S.PartSpec("w1", "yes", hall, [], {"height": "29.5"})], [hall], {})
    pts = np.vstack([np.c_[rng.uniform(0.2, 19.8, (1200, 2)), np.full(1200, 21.0)],
                     np.c_[rng.uniform(10.0, 12.0, (48, 2)), np.full(48, 29.2)]])
    q = M.sfm_quad(model, model.built_part("w1"), pts)
    assert q["raised_k"] == C.RAISED_CELLS and q["raised_top"] == pytest.approx(29.2), q


def test_a_downward_change_is_contradicted_by_a_raised_section_over_the_footprint():
    """The rule on numbers (the parapet fixtures of test_d25, the roof read at 22 m over the interior): a contiguous
    patch reaching above V + AGREE contradicts; no patch measured (or with another patch size) refuses; a patch at the
    roof level (a parapet, the roof itself) does not contradict."""
    fit = dict(td.DOWN_FIT)

    def ch(raised, k=C.RAISED_CELLS):
        q = td.quad((22.3, 22.2, 22.1, 21.5))
        if raised is not None:
            q.update(raised_top=raised, raised_k=k, raised_cells_seen=400)
        return td.parapet(**{SF: {"quad": q, "fit": fit}})
    pre = C.rule_h_pre(td.DOWN_PART, ch(22.3))
    assert pre["candidate"] and pre["direction"] == "down", pre["why"]
    V = pre["V"]
    edge = max(V + C.AGREE_M, V / (1.0 - C.AGREE_REL))                  # rt - V = agree_tol(rt, V)
    assert C.rule_h_pre(td.DOWN_PART, ch(edge - 0.01))["candidate"]
    for raised in (edge + 0.02, 29.5):
        pre = C.rule_h_pre(td.DOWN_PART, ch(raised))
        assert not pre["candidate"] and "contiguous patch" in pre["why"] and "contradicted" in pre["why"], pre["why"]
    for bad in (ch(None), ch(22.3, k=9)):
        pre = C.rule_h_pre(td.DOWN_PART, bad)
        assert not pre["candidate"] and "cannot be ruled out" in pre["why"], pre["why"]
    # going up the patch is not asked for (U1 / U2 hold there)
    assert C.rule_h_pre(td.PART, td.choir())["candidate"]


def test_a_penthouse_on_a_map_right_hall_is_never_lowered_to_the_main_roof():
    """verify-r5 HIGH: before, the fit 21.24 (4 % of the interior cells above it), the cornice lines 22.01 and the
    skyline 22.11 agreed and the full value 21.787 m 'photo-corrected (auto)' was published on a hall the map has
    right at 29.5 m (its 8 x 8 m penthouse). The penthouse is a contiguous patch reaching 29.5 m."""
    model, res, d = _run_raised()
    pre = d["pre"]
    assert not pre["candidate"] and "contiguous patch" in pre["why"], pre["why"]
    assert pre["verdicts"][SF]["agrees"] and pre["verdicts"][SF]["value"] < 22.0      # the fit itself still reads the roof
    assert not res["records"] and model.part("w1").params["height"].value == 29.5


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"sx": 6.0, "sy": 6.0, "px": 19.0, "py": 19.0},       # corner tower: before UB 26.273
                                {"sx": 30.0, "sy": 4.0, "px": 0.0, "py": 8.0},        # clerestory strip: before 21.783
                                {"sx": 6.0, "sy": 6.0, "px": 8.0, "py": 8.0},
                                {"nel": 2}, {"seed": 5}])
def test_raised_section_variants_change_nothing(kw):
    model, res, d = _run_raised(**kw)
    assert not res["records"] and model.part("w1").params["height"].value == 29.5, d["pre"]["why"]


def test_a_really_flat_hall_seen_from_above_is_still_lowered():
    """The control (verify-r5's 0.2 m 'section'): the building really is a flat roof behind a 22 m parapet and the map
    is wrong at 29.5 m; the highest patch is the parapet, the part is lowered to about 21.8 m."""
    model, res, d = _run_raised(sx=0.2, sy=0.2)
    pre, f = d["pre"], d["final"]
    assert pre["candidate"] and pre["direction"] == "down", pre["why"]
    assert f["fires"] and f["value"] <= 22.5, f
    rec = next(a for a in model.adoptions if a.get("part") == "w1")
    assert C.check_record(rec, {rec["id"]: rec}) is None and C.check_model(model) == []


# ---------------------------------------------------------------- the fit's interior support: an area, nothing inside


def _nc5_scene(off=0.4, nt=15.0, crowns=(), sides="ens", net2off=None, net2lo=10.0, ownwalls=False, seed=3):
    """verify-r5 attack_nc5.py: the map-right 10 m annex (test_d25_occluders' scene), a scaffold net at offset `off`
    from its free edges (negative: inside the outline) up to `nt` on `sides`, its roof unseen, tree crowns
    (cx, cy, R, zc, vertical scale) and optionally a second, rooftop net ring at `net2off` from `net2lo` to `nt`."""
    rng = np.random.default_rng(seed)
    th = TO.TH
    P = []
    gx, gy = np.meshgrid(np.arange(-19.5, 20, 1.0), np.arange(-9.5, 10, 1.0))
    P.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 12.0)])
    for a, b, n in (((-20, -10), (20, -10), 900), ((20, 10), (-20, 10), 900), ((-20, 10), (-20, -10), 600),
                    ((20, -10), (20, -5), 200), ((20, 5), (20, 10), 200)):
        P.append(TO._wall(a, b, 3.0, 11.8, n, rng))
    for cx, cy, R, zc, vs in crowns:
        v = rng.normal(size=(int(60 * R * R), 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        v = v[v[:, 2] > -0.3]
        P.append(np.c_[cx + R * v[:, 0], cy + R * v[:, 1], zc + vs * R * v[:, 2]] + rng.normal(0, 0.3, (len(v), 3)))

    def faces(o):
        return {"e": ((34 + o, -5 - o), (34 + o, 5 + o)), "n": ((34 + o, 5 + o), (20.0, 5 + o)),
                "s": ((20.0, -5 - o), (34 + o, -5 - o))}
    if net2off is not None:
        for k in sides:
            P.append(TO._wall(*faces(net2off + th)[k], net2lo, nt, 1500, rng))
    for k in sides:
        P.append(TO._wall(*faces(off + th)[k], 0.0, nt, 2500, rng))
    for k in "ens":
        if k not in sides or ownwalls:
            P.append(TO._wall(*faces(0.0)[k], 0.0, 10.0, 2500, rng))
    G = np.c_[rng.uniform(-60, 75, 9000), rng.uniform(-55, 55, 9000)]
    G = G[~st.points_in_polygon(G, TO.MAIN) & ~np.any([st.points_in_polygon(G, r) for r in TO._nets(off - 0.2)], 0)]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    cams = tj.cameras(n=24, radius=80.0)
    cen = [np.array(c.center[:2]) for c in cams]
    A, B, photos = [], [], []
    o = off + 0.15
    sd = {"e": ((34 + o, -5 - o), (34 + o, 5 + o)), "n": ((34 + o, 5 + o), (20.5, 5 + o)),
          "s": ((20.5, -5 - o), (34 + o, -5 - o))}
    for k in sides:
        a, b = map(np.array, sd[k])
        for z in list(np.arange(2.0, nt - 0.5, 2.0)) + [nt]:
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
    a8 = np.radians(np.arange(0, 360, 45) + 22.5)
    extra = [(r, 0.0, nt) for r in TO._nets(off)] + \
        [(np.c_[cx + R * np.cos(a8), cy + R * np.sin(a8)], zc - 0.3 * vs * R, zc + vs * R) for cx, cy, R, zc, vs in crowns]
    sky = tj.sky_masks(TO._annex_model(10.0, extra), cams)
    model = TO._annex_model(10.0)
    model.frame = FRAME_OK
    model.part("w2").params["height"].state = S.STATE_DISPUTED
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    return model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP), P


def _run_nc5(**kw):
    model, ev, P = _nc5_scene(**kw)
    res = M.correct_landmark(model, ev, only=["w2"])
    return model, res, res["decisions"]["w2"], P


def test_a_rooftop_net_ring_is_no_roof_area():
    """verify-r5 attack_nc5.py two nets: a facade net 0.4 m out plus a rooftop net ring 2.5 m inside (10-15 m); the
    fit 14.4 stood on the ring's 22 cells (22 of 22 interior cells) and the full value 14.897 m was published. The
    ring holds no interior inlier whose 8 neighbours are seen, and every one of its cells holds points far under the
    fit."""
    model, res, d, P = _run_nc5(net2off=-2.8)
    pre = d["pre"]
    assert not pre["candidate"] and not pre["verdicts"][SF].get("agrees"), pre
    fit = M.sfm_fit(TO._annex_model(10.0), TO._annex_model(10.0).built_part("w2"), P)
    assert fit["interior_inliers"] == fit["interior_cells"] and fit["interior_core"] == 0, fit
    assert fit["volume_share"] > 0.9, fit
    assert not res["records"] and model.part("w2").params["height"].value == 10.0


def test_a_plane_tree_over_the_roof_behind_a_net_is_seen_inside_the_volume():
    """verify-r5 MEDIUM (D15 disputes, D25 published): a flat plane-tree crown (R 9, vertical scale 0.45) overhanging
    the roof from 1.5 m outside the south wall, a net 0.4 m outside the east and north facades to 15 m: the fit 15.6
    on the crown (26 of 52 interior cells), the walls the net's, a lower bound 12.457 m. The crown's flanks hang
    inside the volume a roof at 15.6 m would close (37 % of the interior cells)."""
    model, res, d, P = _run_nc5(sides="en", crowns=[(27.0, -6.5, 9.0, 12.0, 0.45)])
    pre = d["pre"]
    assert not pre["candidate"] and "inside the volume" in str(pre["verdicts"][SF]), pre
    assert not res["records"] and model.part("w2").params["height"].value == 10.0


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"net2off": -2.8, "net2lo": 13.0},          # the ring's base hidden near its top
                                {"off": -2.8, "ownwalls": True, "net2off": None, "sides": "ens"},
                                {"sides": "en", "crowns": [(27.0, -6.5, 9.0, 11.0, 0.45)]},
                                {"crowns": [(23.0, 0.0, 3.0, 13.0, 0.7), (27.0, 0.0, 3.0, 13.0, 0.7),
                                            (31.0, 0.0, 3.0, 13.0, 0.7)]}])
def test_fit_support_variants_change_nothing(kw):
    model, res, d, P = _run_nc5(**kw)
    assert not res["records"] and model.part("w2").params["height"].value == 10.0, d["pre"]["why"]


def test_the_fit_needs_an_area_and_nothing_inside_the_volume():
    """The guards on numbers (test_d25's FIT, the choir's roof fit): the volume share over 25 %, missing, or under the
    below share (every cell under the fit by its highest point is under it); no interior core cell; the interior counts
    missing or not nesting (verify-r5 forge_r5b.py: interior_cells deleted, None or 0 made the 40 % share vacuous)."""
    fit = dict(td.FIT)
    assert C.sfm_verdict({"fit": fit}, td.MAP, True)["agrees"]
    assert C.sfm_verdict({"fit": dict(fit, volume_share=0.25, below_share=0.03)}, td.MAP, True)["agrees"]
    bads = [({"volume_share": 0.26}, "inside the volume"), ({"volume_share": None}, "volume"),
            ({"volume_share": 0.02, "below_share": 0.03}, "below share"), ({"interior_core": 0}, "area of roof"),
            ({"interior_cells": None}, "not all there"), ({"interior_cells": 0}, "nest"),
            ({"interior_footprint": None}, "not all there"), ({"interior_core": None}, "not all there"),
            ({"interior_inliers": 41}, "nest"), ({"interior_footprint": 39}, "nest"), ({"interior_core": 31}, "nest")]
    for bad, why in bads:
        got = C._fit_usable(dict(fit, **bad), td.MAP)
        assert got is not None and why in got, (bad, got)
        assert not C.sfm_verdict({"fit": dict(fit, **bad)}, td.MAP, True).get("agrees"), bad
    deleted = {k: v for k, v in fit.items() if k != "interior_cells"}
    assert "not all there" in C._fit_usable(deleted, td.MAP)
    # the measurement: a flat roof seen over the interior has a core and nothing under it; a ring of vertical cells
    # (a rooftop net 2.5 m inside) has neither
    model = TO._annex_model(10.0)
    b = model.built_part("w2")
    rng = np.random.default_rng(2)
    gx, gy = np.meshgrid(np.arange(22.5, 32, 1.0), np.arange(-2.5, 3, 1.0))
    xy = np.c_[gx.ravel(), gy.ravel()]
    roof = np.vstack([np.c_[xy + rng.uniform(-0.4, 0.4, xy.shape), np.full(len(xy), 15.0) + rng.normal(0, 0.05, len(xy))]
                      for _ in range(4)])
    f1 = M.sfm_fit(model, b, roof)
    assert f1["interior_core"] >= 10 and f1["volume_share"] == 0.0 and C._fit_usable(f1, 10.0) is None, f1
    ring = np.vstack([TO._wall((31.4, -2.4), (31.4, 2.4), 10.0, 15.0, 1500, rng),
                      TO._wall((31.4, 2.4), (22.5, 2.4), 10.0, 15.0, 1500, rng),
                      TO._wall((22.5, -2.4), (31.4, -2.4), 10.0, 15.0, 1500, rng)])
    f2 = M.sfm_fit(model, b, ring)
    assert f2["interior_core"] == 0 and f2["volume_share"] > 0.9, f2


def test_the_gate_checks_the_fits_footprint_against_the_outline():
    """check_model and audit_glb bound a D25-H record's fit footprint by its part's outline (block_footprint_bounds)."""
    model = TO._annex_model(10.0)
    b = model.built_part("w2")
    others = [(o.outer, o.holes) for o in model.built if o.id != "w2"]
    lo, hi = C.block_footprint_bounds(b.outer, b.holes, others)
    fit = dict(td.FIT, interior_footprint=hi, interior_cells=40, interior_inliers=30)
    rec = {"channels": {SF: {"fit": fit}}}
    assert C.fit_footprint_problem(rec, b.outer, b.holes, others) is None
    for nf in (hi + 1, 35):
        rec = {"channels": {SF: {"fit": dict(fit, interior_footprint=nf)}}}
        assert C.fit_footprint_problem(rec, b.outer, b.holes, others) is not None, nf
    assert C.fit_footprint_problem({"channels": {SF: {"fit": None}}}, b.outer, b.holes, others) is None


# ---------------------------------------------------------------- D25-B: the roof an area


def test_a_net_with_its_base_hidden_above_h_minus_tol_builds_no_block():
    """verify-r5 MEDIUM attack_block_net2.py lo8: a net 2.5 m inside a 6 m stub, seen only from 8 m up: its 22 cells
    (a ring; their 2nd-lowest points above H - TOL 7.225 m) were the block's roof and >= 9.225 m was built. A ring has
    no roof cell whose 8 neighbours all hold points."""
    model, res, d = R5._run_block2(inset=2.5, netlo=8.0)
    f = d["final"]
    assert not f["fires"] and "as an area" in f["why"] and f["roof_check"]["core_cells"] == 0, f["why"]
    assert not res["records"] and model.part("w2").params["height"].value == 3.0


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"inset": 2.5, "netlo": 9.0}, {"inset": 2.5, "netlo": 10.0},
                                {"inset": 2.5, "netlo": 12.0},
                                {"crown_r": 5.0, "crown_z": 12.0, "crowns": [(24.0, -6.5), (30.0, -6.5)]}])
def test_block_area_variants_build_no_block(kw):
    model, res, d = R5._run_block2(**kw)
    assert not res["records"] and model.part("w2").params["height"].value == 3.0, d["final"]


def test_block_roof_problem_needs_an_area_of_roof():
    """block_roof_problem on numbers: the same 24 roof cells as a 6 x 4 patch pass; laid out as a ring (one cell wide)
    or a two-cell band they fail; one core cell is enough; positions must be distinct grid cells."""
    H = 10.0
    tops = [15.0 + 0.01 * k for k in range(24)]

    def roof(pos):
        cells = [[t, round(t - 0.3, 3), round(t - 0.25, 3), x, y] for t, (x, y) in zip(tops, pos)]
        return {"interior_m": 2.0, "cells_footprint": 40, "z0": 0.0, "tops": list(tops), "roof_tops": list(tops),
                "cells": cells}
    patch = [(k % 6, k // 6) for k in range(24)]
    why, out = C.block_roof_problem(roof(patch), H)
    assert why is None and out["core_cells"] == 8, (why, out)
    ring = [(x, 0) for x in range(8)] + [(x, 5) for x in range(8)] + [(0, y) for y in range(1, 5)] + \
        [(7, y) for y in range(1, 5)]
    why, out = C.block_roof_problem(roof(ring), H)
    assert why is not None and "as an area" in why and out["core_cells"] == 0, (why, out)
    band = [(x, y) for x in range(12) for y in (0, 1)]
    assert "as an area" in (C.block_roof_problem(roof(band), H)[0] or "")
    one = [(x, y) for x in range(3) for y in range(3)] + [(10 + k, 0) for k in range(15)]
    why, out = C.block_roof_problem(roof(one), H)
    assert why is None and out["core_cells"] == 1, (why, out)
    dup = list(patch)
    dup[5] = dup[4]
    assert "distinct" in (C.block_roof_problem(roof(dup), H)[0] or "")
    half = [(x + 0.5, y) for x, y in patch]
    assert "distinct" in (C.block_roof_problem(roof(half), H)[0] or "")
    # the core counts only the gate's own roof cells: the patch's middle made vertical leaves the ring
    r = roof(patch)
    for c in r["cells"]:
        if 1 <= c[3] <= 4 and 1 <= c[4] <= 2:
            c[1] = 0.5                                                   # its points reach the ground: vertical
    why, out = C.block_roof_problem(r, H)
    assert why is not None and out["core_cells"] == 0, (why, out)


def test_the_measurement_records_the_cells_grid_positions():
    """crowd3d.correct_measure.block_roof: each interior cell's [highest, lowest, 2nd-lowest, column, row] and the grid
    origin; the gate's block_cells_problem finds them inside the block's interior."""
    model = TO._block_model({})
    b = model.built_part("w2")
    rng = np.random.default_rng(4)
    gx, gy = np.meshgrid(np.arange(24.5, 32, 1.0), np.arange(-2.5, 3, 1.0))
    xy = np.c_[gx.ravel(), gy.ravel()]
    P = np.vstack([np.c_[xy + rng.uniform(-0.4, 0.4, xy.shape), np.full(len(xy), 15.0)] for _ in range(3)])
    rr = M.block_roof(model, b, P)
    assert rr["grid_lo"] == [round(float(v), 3) for v in b.outer.min(0) - 1.0]
    assert all(len(c) == 5 for c in rr["cells"]) and len(rr["cells"]) == len(xy)
    rec = {"roof": rr}
    assert C.block_cells_problem(rec, b.outer, b.holes) is None
    moved = copy.deepcopy(rec)
    moved["roof"]["cells"][0][3] = 0                                     # a cell moved to the outline's corner
    assert "outside the block's interior" in (C.block_cells_problem(moved, b.outer, b.holes) or "")
    shifted = copy.deepcopy(rec)
    shifted["roof"]["grid_lo"] = [0.0, 0.0]
    assert "grid origin" in (C.block_cells_problem(shifted, b.outer, b.holes) or "")


# ---------------------------------------------------------------- forgeries (verify-r5 forge_r5.py, forge_r5b.py, forge_r5c.py)


def test_a_forged_block_base_is_caught_by_check_model_and_the_glb_audit():
    """verify-r5 forge_r5c.py: the block's roof.z0 and geometry.z0 moved together to -10 or 5 (built z0 0) passed every
    check; the base is the input of the gate's own vertical-cell re-derivation."""
    model, rec = R5._block_record()
    assert C.check_model(model) == []
    sx = model.compact_extras()
    assert not st._audit_glb_d25({}, sx, {"snap": {"accepted": True}})
    for z0 in (-10.0, 5.0):
        m2 = copy.deepcopy(model)
        a = next(x for x in m2.adoptions if x["id"] == rec["id"])
        a["geometry"]["z0"] = a["roof"]["z0"] = z0
        a["roof_check"] = C.block_roof_problem(a["roof"], a["H"], z0)[1]
        assert any("base z0" in v["why"] for v in C.check_model(m2)), C.check_model(m2)
        s2 = copy.deepcopy(sx)
        a2 = next(x for x in s2["adoptions"] if x["id"] == rec["id"])
        a2["geometry"]["z0"] = a2["roof"]["z0"] = z0
        a2["roof_check"] = C.block_roof_problem(a2["roof"], a2["H"], z0)[1]
        assert any("base z0" in v["why"] for v in st._audit_glb_d25({}, s2, {"snap": {"accepted": True}}))


def test_a_withheld_id_without_an_outline_does_not_switch_the_footprint_bound_off():
    """verify-r5 forge_r5.py: 'withheld': ['zz-not-a-part'] in the GLB's structure extras switched the footprint's
    lower bound off, and the consistent B5d forgery (footprint 12, roof check recomputed) passed audit_glb. The lower
    bound now subtracts only the outlines published: a withheld part's (compact_extras 'withheld_rings') or none."""
    model, rec = R5._block_record()
    sx = model.compact_extras()
    assert sx["withheld"] == [] and sx["withheld_rings"] == {}
    a = next(x for x in sx["adoptions"] if x["id"] == rec["id"])
    a["roof"]["cells_footprint"] = 30
    a["roof_check"] = C.block_roof_problem(a["roof"], a["H"], 0.0)[1]
    assert C.check_record(a, {a["id"]: a}) is None
    sx["withheld"] = ["zz-not-a-part"]
    v = st._audit_glb_d25({}, sx, {"snap": {"accepted": True}})
    assert any("outside what the block's outline" in x["why"] for x in v), v
    # a real withheld part travels with its outline (structure.compact_extras)
    india = S.build_structure("synthetic", [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "10"}),
                                            S.PartSpec("w2", "yes", tj.BOX.copy() * 0.3, [],
                                                       {"height": "20", "min_height": "15"})], [tj.BOX.copy()], {})
    ex = india.compact_extras()
    assert ex["withheld"] == ["w2"] and len(ex["withheld_rings"]["w2"][0]) == 4, ex["withheld"]


def test_the_fit_records_interior_counts_the_gate_can_check():
    """verify-r5 forge_r5b.py on a D25-H record: interior_cells deleted / None / 0 made the 40 % share vacuous (15
    inliers of 60 passed); the record's rule re-run (check_record) refuses now, and check_model bounds the fit's
    footprint by the part's outline. A record of the synthetic tower (map 20 m) whose agreeing set holds its fit."""
    model = tj.structure(20.0)
    part = {"id": "w2", "map_top": 20.0, "z0": 0.0, "roof_h": 0.0, "pitched": False,
            "min_width": model.part("w2").min_width()}
    fit = dict(td.FIT, top=30.0, top_p05=29.8, top_p95=30.3, interior_footprint=60, interior_cells=40,
               interior_inliers=30, interior_core=8)
    ch = {SK: td.sky(top=30.2, p05=29.6, p95=30.8), SF: {"quad": td.quad((29.8, 29.9, 30.1, 29.5)), "fit": fit},
          LN: td.lines(30.1, (29.9, 30.3), parameter="eave")}
    pre = C.rule_h_pre(part, ch)
    assert pre["candidate"] and SF in pre["agreeing"], pre["why"]
    box = td.make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.1, walls=40.0)
    fin = C.rule_h_box(pre, box, td.SCALE, part)
    assert fin["fires"], fin["why"]
    (rec,) = td.g5(model, C.apply_height(model, "w2", pre, fin, ch, box, td.SCALE))
    assert C.check_record(rec, {rec["id"]: rec}) is None and C.check_model(model) == [], C.check_model(model)
    for bad in ({"interior_cells": None}, {"interior_cells": 0}, "delete", {"interior_inliers": 15, "interior_cells": 60}):
        f = copy.deepcopy(rec)
        fr = f["channels"][SF]["fit"]
        if bad == "delete":
            del fr["interior_cells"]
        else:
            fr.update(bad)
        why = C.check_record(f, {f["id"]: f})
        assert why is not None and "triggers fail" in why, (bad, why)
    m2 = copy.deepcopy(model)
    a = next(x for x in m2.adoptions if x["id"] == rec["id"])
    a["channels"][SF]["fit"]["interior_footprint"] = 200                      # over the outline's 64
    assert C.check_record(a, {a["id"]: a}) is None
    assert any("fit's interior footprint" in v["why"] for v in C.check_model(m2)), C.check_model(m2)
    m3 = copy.deepcopy(model)
    a = next(x for x in m3.adoptions if x["id"] == rec["id"])
    a["geometry"]["z0"] = 3.0                                                  # another base
    assert any("base z0" in v["why"] for v in C.check_model(m3)), C.check_model(m3)


# ---------------------------------------------------------------- guards for the mutants that survived verify-r5


def test_R2_the_above_share_is_asked_of_every_downward_fit():
    """Mutant R2 (the above share checked only for drops > 2 TOL): a fit just over TOL under the map with 11 % of its
    interior cells above it may not say 'lower'."""
    mt = 31.0
    tol = C.tol_of(mt)
    top = mt - tol - 0.5                                                  # between TOL and 2 TOL under the map
    assert mt - 2 * tol < top < mt - tol
    f = dict(td.DOWN_FIT, top=top, top_p05=top - 0.2, top_p95=top + 0.2, above_share=0.11)
    why = C._fit_usable(f, mt)
    assert why is not None and "reaches higher" in why, why
    assert C._fit_usable(dict(f, above_share=0.10), mt) is None


def test_R4_R15_interior_inliers_and_the_above_share_use_their_own_tolerances():
    """Mutants R4 (interior inliers counted within 3 x FIT_TAU) and R15 (the above share with 3 x TOL): interior cells
    1.5 m off the fitted roof are not on it; cells between TOL and 3 TOL above it count as above."""
    hall = np.array([[0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0]])
    model = S.build_structure("synthetic", [S.PartSpec("w1", "yes", hall, [], {"height": "25"})], [hall], {})
    b = model.built_part("w1")
    rng = np.random.default_rng(6)
    gx, gy = np.meshgrid(np.arange(0.5, 20, 1.0), np.arange(0.5, 20, 1.0))
    xy = np.c_[gx.ravel(), gy.ravel()]
    z = np.full(len(xy), 20.0)
    off = (xy[:, 0] > 2.5) & (xy[:, 0] < 6.5) & (xy[:, 1] > 2.5) & (xy[:, 1] < 17.5)      # 60 interior cells 1.5 m up
    z[off] = 21.5
    bump = (xy[:, 0] > 12.5) & (xy[:, 0] < 16.5) & (xy[:, 1] > 6.5) & (xy[:, 1] < 13.5)   # 24 cells 2 TOL above
    tol = C.tol_of(25.0)
    z[bump] = 20.0 + 2.0 * tol
    P = np.vstack([np.c_[xy + rng.uniform(-0.4, 0.4, xy.shape), z + rng.normal(0, 0.03, len(z))] for _ in range(4)])
    f = M.sfm_fit(model, b, P)
    assert f["top"] == pytest.approx(20.0, abs=0.7), f
    inner = st.points_in_polygon(xy, hall) & (st.distance_to_rings(xy, [hall])[0] >= M.INTERIOR_M)
    n_flat = int(np.sum(inner & ~off & ~bump))
    assert f["interior_inliers"] == n_flat, (f["interior_inliers"], n_flat)        # R4: the 1.5 m cells are off it
    assert f["above_share"] == pytest.approx(np.sum(inner & bump) / np.sum(inner), abs=0.01), f   # R15


def test_R5_ten_interior_inliers_are_needed_even_when_the_share_is_met():
    """Mutant R5 (SFM_FIT_INTERIOR_MIN 10 -> 5): 9 inliers of 20 interior cells meet the 40 % share but not the 10."""
    f = dict(td.FIT, interior_inliers=9, interior_cells=20, interior_core=2)
    why = C._fit_usable(f, td.MAP)
    assert why is not None and "max(10" in why, why
    assert C._fit_usable(dict(f, interior_inliers=10), td.MAP) is None


def test_R7_R8_block_lines_drop_a_line_whose_middle_lies_in_a_taller_part_or_near_a_slightly_taller_one():
    """Mutants R7 (block lines checked at the endpoints only) and R8 (dropped only near parts more than 6 m taller):
    a line across a narrow taller part with both ends > 1 m from it; lines along a wing 5 m taller than the stub."""
    gap = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 20.0], [0.0, 20.0]])
    pier = np.array([[14.0, 8.0], [16.0, 8.0], [16.0, 12.0], [14.0, 12.0]])
    wing = np.array([[0.0, 20.0], [30.0, 20.0], [30.0, 26.0], [0.0, 26.0]])
    main = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 26.0], [0.0, 26.0]])
    model = S.build_structure("synthetic", [S.PartSpec("p", "yes", pier, [], {"height": "20"}),
                                            S.PartSpec("wN", "yes", wing, [], {"height": "8"})], [main], {})
    b = model.built_part(next(p.id for p in model.specs if p.kind == "remainder"))
    assert b.top <= 3.5
    # three lines across the pier: ends 4 m either side of it, middle inside it
    A = np.array([[10.0, 10.0 + 0.2 * k, 6.0] for k in range(3)])
    Bp = np.array([[20.0, 10.0 + 0.2 * k, 6.0] for k in range(3)])
    across = L3.Lines3D(A, Bp, np.full(3, 5), [[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.ones(3), np.zeros(3))
    r = M.block_lines(across, b, model=model)
    assert not r["valid"] and r["near_taller_dropped"] == 3, r
    # three lines 0.5 m inside the gap along the 8 m wing (5 m taller than the 3 m stub)
    A = np.array([[4.0 + 6 * k, 19.5, 6.0] for k in range(3)])
    Bw = np.array([[8.0 + 6 * k, 19.5, 6.0] for k in range(3)])
    along = L3.Lines3D(A, Bw, np.full(3, 5), [[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.ones(3), np.zeros(3))
    r = M.block_lines(along, b, model=model)
    assert not r["valid"] and r["near_taller_dropped"] == 3, r
    # the same lines without the model are the block's own
    assert M.block_lines(along, b)["valid"] and M.block_lines(across, b)["valid"]


def test_R10_the_footprint_upper_bound_counts_only_cells_two_metres_inside():
    """Mutant R10 (the upper bound counts the edge cells too): a footprint forged between the interior count and the
    count with edge cells fails."""
    outer = np.array([[0.0, 0.0], [14.0, 0.0], [14.0, 10.0], [0.0, 10.0]])
    lo, hi = C.block_footprint_bounds(outer, [], [])
    assert lo == hi == 60, (lo, hi)                                       # 10 x 6 cells >= 2 m inside
    rec = {"roof": {"cells_footprint": 61}}
    assert "footprint" in (C.block_footprint_problem(rec, outer, [], []) or "")
    assert C.block_footprint_problem({"roof": {"cells_footprint": 60}}, outer, [], []) is None
