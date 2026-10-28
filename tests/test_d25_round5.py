"""D25 fixer round 5 (docs/STRUCTURE_FIRST.md §2.4.8 part 18, after verify-r4): four ways a correction was published
on a part the map has RIGHT (or on no building at all), each now refused, and the guards that refuse them.

- D1, a DOWNWARD change needs the part's own top over its interior (verify-r4 attack_parapet.py): a hall the map has
  right at 31 m, a hipped roof hidden behind a 22 m parapet in every street photo, OSM without roof:shape: the cornice
  lines and the skyline agreed on the parapet and D25 published 21.99 m as the full value. The same photos of a really
  flat 22 m box give the same numbers, so the eave and the skyline alone are a ticket; the roof-surface fit (with
  hardly any interior cell above it) or the 3-D lines' apex / ridge may lower a part.
- D25-B on an empty cloister garth (verify-r4 attack_garth.py): the wings' inner facade lines, tucked under the gap's
  outline, were the block's '3-D lines'; lines near a taller part are that part's (crowd3d.correct_measure.block_lines).
- D25-B with something seen INSIDE the block's volume (verify-r4 attack_block_net2.py): the true roof under tree crowns,
  a net whose base is hidden -- a cell holding points more than TOL under H counts as 'below' and is no roof cell.
- U2 on a rough wall with tangent crowns, and a non-model neighbour's firewall over the roof (verify-r4
  attack_crowns2.py, attack_crowns_nb.py): the above-eave spread limit is capped at 0.18 m, a bin seen across the edge
  but off the wall ends the run, and above the eave only the cells where the wall is seen below it count.

Plus the gate re-deriving a block's interior footprint count and its roof-check summary (verify-r4 forge_block.py B5,
B6), and guard-level tests for the mutants that survived verify-r4 (V5b, N16, N23, N25)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import test_d25 as td
import test_d25_occluders as TO
import test_skyline_judge as tj

from crowd3d import correct as C
from crowd3d import correct_measure as M
from crowd3d import judge as J
from crowd3d import lines3d as L3
from crowd3d import structure as S
from crowd3d import stylize as st

FRAME_OK = TO.FRAME_OK

# ---------------------------------------------------------------- D1: the hidden roof behind a parapet


def _parapet_scene(map_top=31.0, eave=22.0, roof_top=None, hx=25.0, hy=25.0, dist=60.0, f=200.0, ncam=20,
                   sector=360.0, truth="hipped", roofpts=False, seed=3):
    """verify-r4 attack_parapet.py: a hall (2hx x 2hy) with walls and a parapet to `eave`; truth 'hipped' -- behind
    the parapet a hipped roof rising to roof_top (default map_top), hidden in every street photo `dist` m from the
    centre (no SfM point on it); truth 'flat' -- the building really is a flat `eave` m box (the map is wrong). The
    map: height=map_top, no roof:shape (a flat box). roofpts: the true roof seen from afar (3 points per 1 m cell).
    Cornice lines at 8, 15 and `eave` m on every facade."""
    roof_top = map_top if roof_top is None else roof_top
    rng = np.random.default_rng(seed)
    hall = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])

    def model(tags):
        return S.build_structure("synthetic", [S.PartSpec("w1", "yes", hall.copy(), [], tags)], [hall.copy()], {})
    if truth == "flat":
        true_model = model({"height": f"{eave:g}"})
    else:
        true_model = model({"height": f"{roof_top:g}", "roof:shape": "hipped", "roof:height": f"{roof_top - eave:g}"})
    P = []
    for (x0, y0), (x1, y1) in zip(hall, np.roll(hall, -1, 0)):
        n = int(60 * math.hypot(x1 - x0, y1 - y0))
        t, z = rng.uniform(0, 1, n), rng.uniform(1, eave, n)
        P.append(np.c_[x0 + t * (x1 - x0), y0 + t * (y1 - y0), z])
    if roofpts:
        gx, gy = np.meshgrid(np.arange(-hx + 0.5, hx, 1.0), np.arange(-hy + 0.5, hy, 1.0))
        xy = np.c_[gx.ravel(), gy.ravel()]
        if truth == "flat":
            z = np.full(len(xy), eave)
        else:
            d = np.minimum(hx - np.abs(xy[:, 0]), hy - np.abs(xy[:, 1]))
            z = eave + (roof_top - eave) * np.clip(d / min(hx, hy), 0, 1)
        for _ in range(3):
            j = rng.uniform(-0.45, 0.45, (len(xy), 2))
            P.append(np.c_[xy + j, z + rng.normal(0, 0.05, len(z))])
    G = np.c_[rng.uniform(-80, 80, 9000), rng.uniform(-80, 80, 9000)]
    G = G[~st.points_in_polygon(G, hall)]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    angs = (np.linspace(0, 2 * math.pi, ncam, endpoint=False) if sector >= 360 else
            np.radians(np.linspace(-sector / 2, sector / 2, ncam)))
    cams = [tj.look_at([dist * math.cos(a), dist * math.sin(a), 1.6],
                       [0.4 * dist * math.cos(a), 0.4 * dist * math.sin(a), eave * 0.85], f=f, name=f"c{i}")
            for i, a in enumerate(angs)]
    sky = tj.sky_masks(true_model, cams)
    cen = [np.array(c.center[:2]) for c in cams]
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
    m = model({"height": f"{map_top:g}"})
    m.frame = FRAME_OK
    m.part("w1").params["height"].state = S.STATE_DISPUTED          # D15: disputed (ticket only), as in verify-r4
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    return m, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP)


def _run_parapet(**kw):
    model, ev = _parapet_scene(**kw)
    res = M.correct_landmark(model, ev, only=["w1"])
    return model, res, res["decisions"]["w1"]


def test_a_roof_hidden_behind_a_parapet_is_never_lowered_to_the_parapet():
    """verify-r4 HIGH: map-right 31 m (50 x 50 m hall), the hipped roof hidden behind the 22 m parapet in all 20
    street photos. Before: lines3d 21.99 + skyline 21.99 agreed, box [17.98, 25.31], the full value 21.99 m published
    'photo-corrected (auto)' (9 m under the truth). Now the pair reads nothing over the interior: a ticket."""
    model, res, d = _run_parapet()
    pre = d["pre"]
    assert pre.get("direction") is None and not pre["candidate"], pre
    assert "own top over its interior" in pre["why"], pre["why"]
    for c in (C.SKYLINE, C.LINES):                                   # the pair the old rule took (wrong)
        assert pre["verdicts"][c]["valid"] and pre["verdicts"][c]["value"] == pytest.approx(22.0, abs=0.3)
    assert not res["records"] and model.part("w1").params["height"].value == 31.0


def test_a_really_flat_box_with_its_roof_seen_is_still_lowered_and_without_it_is_a_ticket():
    """The control and the cost (doc part 18): the building really is a flat 22 m box, the map says 30 m. With its
    roof seen (photos from afar) the roof-surface fit joins the lines and the skyline and the part is lowered, under
    the map and at the truth; photographed only from the street (no roof point) the same numbers as the parapet's:
    a ticket now (a flat roof seen only from close range is no longer lowered)."""
    model, res, d = _run_parapet(map_top=30.0, truth="flat", roofpts=True)
    pre, f = d["pre"], d["final"]
    assert pre["candidate"] and pre["direction"] == "down" and C.SFM in pre["agreeing"], pre["why"]
    assert pre["verdicts"][C.SFM]["how"] == "surface-fit" and pre["verdicts"][C.SFM]["agrees"]
    assert f["fires"] and f["value"] <= 22.0 + 1.5, f
    rec = next(a for a in model.adoptions if a.get("part") == "w1")
    assert rec["direction"] == "down" and C.check_record(rec, {rec["id"]: rec}) is None
    assert C.check_model(model) == [] and J.check_decisions(model)["passed"]
    model2, res2, d2 = _run_parapet(map_top=30.0, truth="flat", roofpts=False)
    assert not d2["pre"]["candidate"] and "own top over its interior" in d2["pre"]["why"]
    assert not res2["records"]


def test_a_hidden_roof_seen_from_afar_all_round_is_not_lowered():
    """The hipped roof seen from afar (its points on all four facets): the flat fit under the map has cells far above
    it, the all-round top reads the ridge -- refused (before and after)."""
    model, res, d = _run_parapet(roofpts=True)
    assert not res["records"] and model.part("w1").params["height"].value == 31.0, d["pre"]["why"]


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"sector": 90.0, "ncam": 12},                      # before: UB 26.771 published
                                {"map_top": 36.0, "hx": 15.0, "hy": 15.0, "dist": 30.0, "f": 170.0},   # UB 31.141
                                {"map_top": 40.0, "eave": 24.0, "hx": 15.0, "hy": 15.0, "dist": 30.0, "f": 170.0},
                                {"map_top": 36.0, "dist": 50.0, "f": 170.0}])     # 21.965 (full value)
def test_parapet_variants_are_tickets(kw):
    model, res, d = _run_parapet(**kw)
    assert not res["records"], d["final"]


def test_the_roof_surface_fit_must_hold_over_the_interior():
    """(fixer round 5, found while testing D1) the roof-surface fit is the part's own roof only where it holds over
    the interior: with the roof unseen it lands on whatever stands in the edge cells. verify-r2's scaffold net moved
    just INSIDE the map-right 10 m annex's outline (OSM outlines lie 0.07-0.75 m outside real walls): the fit read the
    net's 14.4 m from the edge cells, U1 (the fit) and U2 (the net is the wall) passed and a lower bound 12.17 m was
    published. Now the fit needs >= 10 interior cells (>= 2 m inside) on the fitted roof and >= 40 % of those seen."""
    model, res, d, _ = TO._run_annex("net", top=15.0, off=-0.6)
    v = d["pre"]["verdicts"][C.SFM]
    assert not v.get("agrees") and "interior" in (v.get("note") or v.get("why") or ""), v
    assert not res["records"] and model.part("w2").params["height"].value == 10.0, d["final"]
    fit = dict(td.FIT)
    assert C.sfm_verdict({"fit": fit}, td.MAP, True)["agrees"]
    for bad in ({"interior_inliers": 9}, {"interior_inliers": 15, "interior_cells": 40},
                {k: v for k, v in fit.items() if k != "interior_inliers"}):
        f2 = bad if "top" in bad else {**fit, **bad}
        assert not C.sfm_verdict({"fit": f2}, td.MAP, True).get("agrees"), bad
    assert C.sfm_verdict({"fit": {**fit, "interior_inliers": 16, "interior_cells": 40}}, td.MAP, True)["agrees"]


def test_the_eave_rule_needs_a_pitched_roof_and_a_usable_fit():
    """(fixer round 5) with the tower's fit unusable (no interior support), the eave rule took the skyline's eave and
    the unusable fit's eave on a FLAT tower and wrote a 0.52 m roof-height record the gate refuses ('a roof height on
    a flat roof'), failing the whole run's check (test_d25_measure's end-to-end tower). The eave rule now refuses a
    roof that is not pitched, and takes the SfM eave only from a fit usable as a roof reading."""
    part = dict(td.PART, id="w505513557")
    ch = {td.SK: td.sky(eave=37.7), td.SF: {"fit": dict(td.FIT, eave=37.44, eave_p05=37.24, eave_p95=37.64)},
          td.LN: {"records": []}}
    pre = C.rule_h_pre(part, ch)
    box = td.make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, eave={td.SK: 37.7, td.SF: 37.44})
    assert set(C.eave_readings(ch, 26.0, 32.0)) == {td.SK, td.SF}
    flat = C.rule_eave(ch, dict(part, pitched=False), pre["direction"], pre["V"], box, td.SCALE)
    assert not flat["adopt"] and "not a pitched roof" in flat["why"], flat
    weak = dict(ch, **{td.SF: {"fit": dict(td.FIT, eave=37.44, eave_p05=37.24, eave_p95=37.64, interior_inliers=5)}})
    assert set(C.eave_readings(weak, 26.0, 32.0)) == {td.SK}
    assert set(C.eave_readings(weak, 26.0)) == {td.SK, td.SF}                  # without the map top: as before


@pytest.mark.slow
@pytest.mark.parametrize("off", [-0.8, -0.45])
def test_net_inside_variants_change_nothing(off):
    model, res, d, _ = TO._run_annex("net", top=15.0, off=off)
    assert not res["records"] and model.part("w2").params["height"].value == 10.0, d["final"]


# ---------------------------------------------------------------- D25-B: the cloister garth


GX, GY = 14.0, 9.0


def _garth_scene(sides=4, r=7.0, zc=11.0, wing=10.0, seg=6, incams=False, seed=5):
    """verify-r4 attack_garth.py: wings (OSM parts, `wing` m, the map right) around an open garth that is a gap of
    the main outline (a 'height pending' stub, 3 m); one tree in it (crown r, centre zc); the wings' inner facade
    lines (4 m, the eave) lie inside the gap outline, which is tucked under the parts. sides 4: an enclosed garth;
    2: an inner corner; 1: a yard along one wing. No building in the garth: any block is wrong."""
    def rect(x0, y0, x1, y1):
        return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    main = rect(-20, -15, 20, 15) if sides == 4 else (rect(-20, -9, 14, 15) if sides == 2 else rect(-14, -9, 14, 15))
    wings = ({"wN": rect(-20, GY, 20, 15), "wS": rect(-20, -15, 20, -GY), "wW": rect(-20, -GY, -GX, GY),
              "wE": rect(GX, -GY, 20, GY)} if sides == 4 else
             ({"wN": rect(-20, GY, 14, 15), "wW": rect(-20, -9, -GX, GY)} if sides == 2 else {"wN": rect(-14, GY, 14, 15)}))
    facades = {"wN": ((-GX, GY), (GX, GY)), "wS": ((GX, -GY), (-GX, -GY)), "wW": ((-GX, -GY), (-GX, GY)),
               "wE": ((GX, GY), (GX, -GY))}
    facades = {k: v for k, v in facades.items() if k in wings}

    def model_of(extra=()):
        specs = [S.PartSpec(k, "yes", v.copy(), [], {"height": f"{wing:g}"}) for k, v in wings.items()]
        for i, (ring, z0, z1) in enumerate(extra):
            specs.append(S.PartSpec(f"x{i}", "yes", ring, [], {"height": f"{z1:g}", "min_height": f"{z0:g}"}))
        return S.build_structure("synthetic", specs, [main.copy()], {})
    rng = np.random.default_rng(seed)
    P = []
    for ring in wings.values():
        lo, hi = ring.min(0), ring.max(0)
        gx, gy = np.meshgrid(np.arange(lo[0] + 0.5, hi[0], 1.0), np.arange(lo[1] + 0.5, hi[1], 1.0))
        P.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, wing)])
    for a, b in zip(main, np.roll(main, -1, 0)):
        P.append(TO._wall(a, b, 0.5, wing - 0.2, int(25 * np.linalg.norm(b - a)), rng))
    for a, b in facades.values():
        P.append(TO._wall(a, b, 0.5, wing - 0.2, int(40 * np.linalg.norm(np.subtract(b, a))), rng))
    v = rng.normal(size=(2500, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    v = v[v[:, 2] > -0.3]
    P.append(np.c_[r * v[:, 0], r * v[:, 1], zc + 0.8 * r * v[:, 2]] + rng.normal(0, 0.3, (len(v), 3)))
    P.append(np.c_[rng.normal(0, 0.15, 150), rng.normal(0, 0.15, 150), rng.uniform(0.3, zc - 0.8 * r * 0.3, 150)])
    a8 = np.radians(np.arange(0, 360, 45) + 22.5)
    extra = [(np.c_[r * np.cos(a8), r * np.sin(a8)], zc - 0.24 * r, zc + 0.8 * r),
             (np.c_[0.4 * np.cos(a8), 0.4 * np.sin(a8)], 0.0, zc - 0.24 * r)]
    G = np.c_[rng.uniform(-70, 70, 9000), rng.uniform(-60, 60, 9000)]
    G = G[~st.points_in_polygon(G, main)]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    cams = tj.cameras(n=24, radius=80.0)
    if incams:
        for j, (x, y, tx, ty) in enumerate(((-12, 7.5, 5, -5), (12, 7.5, -5, -5), (-12, -7.5, 5, 5), (12, -7.5, -5, 5),
                                            (0, 8, 0, -8), (0, -8, 0, 8), (-13, 0, 10, 0), (13, 0, -10, 0))):
            cams.append(tj.look_at([x, y, 1.6], [tx, ty, 9.0], name=f"in{j}"))
    sky = tj.sky_masks(model_of(extra), cams)
    cen = [np.array(c.center[:2]) for c in cams]
    A, B, photos = [], [], []
    for a, b in facades.values():
        a, b = np.array(a, float), np.array(b, float)
        for z in (4.0, wing):
            for s0, s1 in [(k / seg + 0.02, (k + 1) / seg - 0.02) for k in range(seg)]:
                p, q = a + s0 * (b - a), a + s1 * (b - a)
                jz = rng.normal(0, 0.03, 2)
                A.append([p[0], p[1], z + jz[0]])
                B.append([q[0], q[1], z + jz[1]])
                photos.append(sorted(np.argsort([np.linalg.norm(c - 0.5 * (p + q)) for c in cen])[:5].tolist()))
    for a, b in zip(main, np.roll(main, -1, 0)):
        a, b = np.array(a, float), np.array(b, float)
        for s0, s1 in ((0.05, 0.3), (0.35, 0.6), (0.65, 0.95)):
            p, q = a + s0 * (b - a), a + s1 * (b - a)
            A.append([p[0], p[1], wing])
            B.append([q[0], q[1], wing])
            photos.append(sorted(np.argsort([np.linalg.norm(c - 0.5 * (p + q)) for c in cen])[:5].tolist()))
    n = len(A)
    lines = L3.Lines3D(np.array(A), np.array(B), np.full(n, 5), photos, np.ones(n), np.zeros(n))
    model = model_of()
    model.frame = FRAME_OK
    pid = max((p for p in model.specs if p.kind == "remainder"), key=lambda p: p.area).id
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    return model, pid, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP)


def _run_garth(**kw):
    model, pid, ev = _garth_scene(**kw)
    assert model.part(pid).params["height"].source == st.PROV_DEFAULT
    res = M.correct_landmark(model, ev, only=[pid])
    return model, res, res["decisions"][pid], pid


def test_an_empty_garth_with_a_tree_builds_no_block():
    """verify-r4 HIGH: before, lines3d 9.97 (the wings' inner facades, tucked under the gap's outline) + the crown's
    q98 16.47 built a block >= 7.166 m 'photo-corrected (auto, lower bound)' on an empty cloister garth (truth: no
    building; the 3 m stub is nearer). The wings' facade lines are theirs: no 3-D lines reading, no block."""
    model, res, d, pid = _run_garth()
    rd = d["readings"][C.LINES]
    assert not rd["valid"] and rd["near_taller_dropped"] >= 12, rd
    assert not d["final"]["fires"] and "no 3-D lines" in d["final"]["why"], d["final"]
    assert not res["records"] and model.part(pid).params["height"].value == 3.0


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"incams": True}, {"sides": 2}, {"sides": 1}, {"sides": 1, "r": 6.0, "zc": 10.0}])
def test_garth_variants_build_no_block(kw):
    """before: cameras in the cloister 7.166; an inner corner 6.983; a yard along one wing 7.14 (r 7 and r 6)."""
    model, res, d, pid = _run_garth(**kw)
    assert not res["records"] and model.part(pid).params["height"].value == 3.0, d["final"]


def test_block_lines_drop_a_taller_parts_facade():
    """crowd3d.correct_measure.block_lines: a line with an end or its middle inside a taller part (top above the
    block's + 1 m) or within 1 m of its outline is that part's; without the model nothing is dropped (as before)."""
    model, pid, ev = _garth_scene()
    b = model.built_part(pid)
    with_model = M.block_lines(ev.lines, b, model=model)
    without = M.block_lines(ev.lines, b)
    assert without["valid"] and without["value"] == pytest.approx(10.0, abs=0.1)        # the wings' eave lines
    assert not with_model["valid"] and with_model["near_taller_dropped"] >= 12, with_model
    # a line 0.5 m off a taller wing's outline is that wing's (the 1 m margin; mutant B2: 0.1 m kept it)
    near = L3.Lines3D(np.array([[-8.0, -GY + 0.5, 6.0]] * 3), np.array([[8.0, -GY + 0.5, 6.0]] * 3), np.full(3, 5),
                      [[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.ones(3), np.zeros(3))
    r = M.block_lines(near, b, model=model)
    assert not r["valid"] and r["near_taller_dropped"] == 3, r
    # a line of the block's own 1.35 m inside the gap outline stays (just beyond the 1 m)
    own = L3.Lines3D(np.array([[-8.0, -GY + 1.35, 6.0]] * 3), np.array([[8.0, -GY + 1.35, 6.0]] * 3), np.full(3, 5),
                     [[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.ones(3), np.zeros(3))
    r = M.block_lines(own, b, model=model)
    assert r["valid"] and r["value"] == pytest.approx(6.0) and r["near_taller_dropped"] == 0, r


# ---------------------------------------------------------------- D25-B: something seen inside the block's volume


def _block_scene2(kind="net", inset=0.2, top=15.0, true_h=6.0, seg=6, netlo=0.0, crown_r=0.0, crown_z=12.0, seed=3):
    """verify-r4 attack_block_net2.py: test_d25_occluders' block-net scene (a 0.3 m net `inset` m inside the stub
    annex's free edges up to `top`, ledgers and the top edge as 3-D lines) with the net's points only from `netlo` m
    up (its base hidden) and tree crowns (r crown_r, centre crown_z) over the annex at x 24 and 30 m; 'roofseen': the
    annex's true roof seen at true_h inside the net."""
    rng = np.random.default_rng(seed)
    i = inset
    P = []
    gx, gy = np.meshgrid(np.arange(-19.5, 20, 1.0), np.arange(-9.5, 10, 1.0))
    P.append(np.c_[gx.ravel(), gy.ravel(), np.full(gx.size, 12.0)])
    for a, b, n in (((-20, -10), (20, -10), 900), ((20, 10), (-20, 10), 900), ((-20, 10), (-20, -10), 600),
                    ((20, -10), (20, -5), 200), ((20, 5), (20, 10), 200)):
        P.append(TO._wall(a, b, 3.0, 11.8, n, rng))
    if kind == "roofseen":
        gx, gy = np.meshgrid(np.arange(21.5, 33, 1.0), np.arange(-3.5, 4, 1.0))
        for _ in range(3):
            j = rng.uniform(-0.45, 0.45, (2, gx.size))
            P.append(np.c_[gx.ravel() + j[0], gy.ravel() + j[1], np.full(gx.size, true_h) + rng.normal(0, 0.05, gx.size)])
    for a, b in (((34 - i, -5 + i), (34 - i, 5 - i)), ((34 - i, 5 - i), (20.0, 5 - i)), ((20.0, -5 + i), (34 - i, -5 + i))):
        P.append(TO._wall(a, b, netlo, top, 2500, rng))
    crowns = [(24.0, 0.0), (30.0, 0.0)] if crown_r else []
    for cx, cy in crowns:
        v = rng.normal(size=(1500, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        v = v[v[:, 2] > -0.3]
        P.append(np.c_[cx + crown_r * v[:, 0], cy + crown_r * v[:, 1], crown_z + 0.8 * crown_r * v[:, 2]]
                 + rng.normal(0, 0.3, (len(v), 3)))
    G = np.c_[rng.uniform(-60, 75, 9000), rng.uniform(-55, 55, 9000)]
    G = G[~st.points_in_polygon(G, TO.MAIN)]
    P.append(np.c_[G, rng.normal(0, 0.05, len(G))])
    P = np.vstack(P)
    cams = tj.cameras(n=24, radius=80.0)
    cen = [np.array(c.center[:2]) for c in cams]
    A, B, photos = [], [], []
    for a, b in [((34 - i, -5 + i + 0.3), (34 - i, 5 - i - 0.3)), ((34 - i - 0.3, 5 - i), (20.5, 5 - i)),
                 ((20.5, -5 + i), (34 - i - 0.3, -5 + i))]:
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
    a8 = np.radians(np.arange(0, 360, 45) + 22.5)
    nets = [TO._rect(34 - i - TO.TH, -5 + i, 34 - i, 5 - i), TO._rect(20.0, 5 - i - TO.TH, 34 - i, 5 - i),
            TO._rect(20.0, -5 + i, 34 - i, -5 + i + TO.TH)]
    truth = TO._block_model({"height": f"{true_h:g}"}, [(r, 0.0, top) for r in nets] +
                            [(np.c_[cx + crown_r * np.cos(a8), cy + crown_r * np.sin(a8)], crown_z - 0.24 * crown_r,
                              crown_z + 0.8 * crown_r) for cx, cy in crowns])
    model = TO._block_model({})
    model.frame = FRAME_OK
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    sky = tj.sky_masks(truth, cams)
    return model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=lines, grade=C.GRADE_SNAP)


def _run_block2(**kw):
    model, ev = _block_scene2(**kw)
    res = M.correct_landmark(model, ev, only=["w2"])
    return model, res, res["decisions"]["w2"]


def test_the_true_roof_seen_under_crowns_inside_a_net_builds_no_block():
    """verify-r4 attack_block_net2.py roofseen + crowns (r 4.5 m): before, the crown cells' highest points made 54
    'roof cells' over the annex's true 6 m roof (seen under them) and a block >= 10.49 m was published. Every cell
    holds points at 6 m, under H - TOL: something lower is seen inside the block's volume."""
    model, res, d = _run_block2(kind="roofseen", crown_r=4.5)
    f = d["final"]
    assert not f["fires"] and "under H" in f["why"] and f["roof_check"]["inside_cells"] >= 40, f["why"]
    assert not res["records"] and model.part("w2").params["height"].value == 3.0


def test_a_net_standing_inside_with_its_base_hidden_builds_no_block():
    """verify-r4 netlo 4 (kills mutant N23's escape): a net 2.5 m inside the outline seen only from 4 m up (its base
    hidden behind vehicles) put 22 'roof cells' on its own cells (lowest point above the lowest quarter) and a block
    >= 9.225 m was published on a 6 m annex. Its cells hold points down to 4 m, under H - TOL."""
    model, res, d = _run_block2(inset=2.5, netlo=4.0)
    f = d["final"]
    assert not f["fires"] and "under H" in f["why"], f["why"]
    assert not res["records"] and model.part("w2").params["height"].value == 3.0


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"inset": 2.5, "netlo": 3.0}, {"inset": 2.2, "top": 12.0, "netlo": 3.5},
                                {"kind": "roofseen", "crown_r": 4.0, "crown_z": 11.0}])
def test_block_inside_variants_build_no_block(kw):
    model, res, d = _run_block2(**kw)
    assert not res["records"], d["final"]


def _cells(tops, low=None, z0=0.0):
    """A roof reading's per-cell [highest, lowest, 2nd-lowest] (a roof: its points just under its top)."""
    low = low or {}
    return [[t, *low.get(k, (round(t - 0.3, 3), round(t - 0.25, 3)))] for k, t in enumerate(tops)]


def test_block_roof_problem_rederives_the_roof_cells_from_each_cells_points():
    """crowd3d.correct.block_roof_problem, guard by guard: (inside) a cell whose 2nd-lowest point lies more than TOL
    under H counts 'below' and is no roof cell; (vertical, mutant N23) the gate itself re-derives which cells reach
    the lowest quarter of their height -- a cell the reading calls 'roof' is not counted when its points reach there;
    (N25) the below share is taken over every interior cell, never only the roof cells; the per-cell numbers must be
    there and be the cells'; the base must be the block's."""
    H, z0 = 10.0, 0.0
    tops = [15.0 + 0.01 * k for k in range(30)]
    ok = {"interior_m": 2.0, "cells_footprint": 40, "z0": z0, "tops": tops, "roof_tops": list(tops), "cells": _cells(tops)}
    assert C.block_roof_problem(ok, H, z0)[0] is None
    # inside: 9 of 30 cells hold points under H - TOL (30 %): refused as 'below'; 7 (23 %) pass the share but are
    # not counted -- with need 10 still met
    for n_in, refused in ((9, True), (7, False)):
        r = dict(ok, cells=_cells(tops, {k: (5.0, 5.1) for k in range(n_in)}))
        why, out = C.block_roof_problem(r, H, z0)
        assert (why is not None and "under H" in why) == refused, why
        if not refused:
            assert out["roof_cells"] == 30 - n_in and out["inside_cells"] == n_in
    # vertical, re-derived (N23): lowest point at 10 % of the height while the 2nd-lowest stays above H - TOL; the
    # reading calls every cell a roof; 25 cells vertical leave 5 < 10
    vert = dict(ok, cells=_cells(tops, {k: (1.5, 9.0) for k in range(25)}))
    why, out = C.block_roof_problem(vert, H, z0)
    assert why is not None and "own roof" in why and out["roof_cells"] == 5, (why, out)
    # the measurement's own split agrees (crowd3d.correct_measure.block_roof: N23 on the measurement side)
    model = TO._block_model({})
    b = model.built_part("w2")
    rng = np.random.default_rng(4)
    gx, gy = np.meshgrid(np.arange(24.5, 32, 1.0), np.arange(-2.5, 3, 1.0))      # interior, clear of the hall's margin
    xy = np.c_[gx.ravel(), gy.ravel()]
    pts = [np.c_[xy, np.full(len(xy), 15.0)], np.c_[xy, np.full(len(xy), 14.9)],
           np.c_[xy[:20], np.full(20, 1.5)], np.c_[xy[:20], np.full(20, 9.0)]]          # 20 cells reach 10 %
    rr = M.block_roof(model, b, np.vstack(pts) + np.c_[rng.uniform(-0.3, 0.3, (sum(len(p) for p in pts), 2)),
                                                         np.zeros(sum(len(p) for p in pts))])
    assert len(rr["tops"]) == len(xy) and len(rr["roof_tops"]) == len(xy) - 20, rr
    assert sorted(c[0] for c in rr["cells"]) == rr["tops"] and rr["z0"] == 0.0
    # N25: 10 vertical cells low (a 3 m thing standing in the interior), 20 roof cells high: 33 % below over every
    # interior cell (refused), 0 % over the roof cells alone
    t2 = [3.0] * 10 + [15.0] * 20
    low = dict(ok, tops=sorted(t2), roof_tops=[15.0] * 20, cells=_cells(sorted(t2), {k: (0.0, 0.1) for k in range(10)}))
    why, out = C.block_roof_problem(low, H, z0)
    assert why is not None and "under H" in why and out["below_share"] == pytest.approx(10 / 30, abs=1e-3), (why, out)
    # the per-cell numbers missing, not the cells', out of order; another base
    for bad in (dict(ok, cells=None), dict(ok, cells=_cells(tops)[:-1]), dict(ok, cells=_cells([t + 1 for t in tops])),
                dict(ok, cells=[[t, t - 0.2, t - 0.3] for t in tops])):
        assert "lowest points per cell" in (C.block_roof_problem(bad, H, z0)[0] or ""), bad.get("cells")
    assert "base" in (C.block_roof_problem(dict(ok, z0=1.0), H, z0)[0] or "")


def _block_record():
    part = {"id": "w9", "min_width": 10.0, "z0": 0.0, "map_top": 3.0}
    res = C.rule_b(part, td.READ, 22.0, td.block_box(), td.SCALE, td.ROOF)
    model = td.block_model()
    rec = C.apply_block(model, "w9", res, td.READ, td.block_box(), td.SCALE)
    td.g5(model, rec)
    return model, rec


def test_the_gate_rederives_a_blocks_footprint_count_and_roof_summary():
    """verify-r4 forge_block.py B5 (cells_footprint 299 -> 12 let 12 roof cells pass the 25 % rule) and B6 (the
    roof_check summary edited -- median 99, roof cells 999 -- passed and was shown in the evidence summary)."""
    model, rec = _block_record()
    b = model.built_part("w9")
    lo, hi = C.block_footprint_bounds(b.outer, b.holes, [(o.outer, o.holes) for o in model.built if o.id != "w9"])
    assert lo <= hi
    # the fixture's reading (40 cells) sits inside the bounds of the fixture's block
    rec2 = copy.deepcopy(rec)
    rec2["roof"]["cells_footprint"] = int(np.clip(rec2["roof"]["cells_footprint"], lo, hi))
    assert C.block_footprint_problem(rec2, b.outer, b.holes, [(o.outer, o.holes) for o in model.built if o.id != "w9"]) is None
    for foot in (lo - 1, hi + 1):
        f = copy.deepcopy(rec2)
        f["roof"]["cells_footprint"] = foot
        assert "footprint" in (C.block_footprint_problem(f, b.outer, b.holes,
                                                          [(o.outer, o.holes) for o in model.built if o.id != "w9"]) or "")
    # check_model carries it: the record in the model, its footprint forged low and its summary recomputed to match
    # (verify-r4 B5 done consistently: only the bounds can tell; mutant B9 left it to the summary check)
    a = next(x for x in model.adoptions if x["id"] == rec["id"])
    a["roof"]["cells_footprint"] = max(0, lo - 5)
    a["roof_check"] = C.block_roof_problem(a["roof"], a["H"], 0.0)[1]
    assert C.check_record(a, {a["id"]: a}) is None
    assert any("outside what the block's outline" in v["why"] for v in C.check_model(model)), C.check_model(model)
    # B6: the summary must be its numbers' and the evidence summary recomputes it
    model, rec = _block_record()
    f = copy.deepcopy(rec)
    f["roof_check"] = dict(f["roof_check"], median=99.0, roof_cells=999)
    assert "roof check summary" in (C.check_record(f) or ""), C.check_record(f)
    ev = C._evidence_summary(f)
    assert ev["interior_roof"]["median"] == rec["roof_check"]["median"] and ev["interior_roof"]["roof_cells"] != 999


def test_the_glb_audit_rederives_a_blocks_footprint_count():
    """audit_glb's D25 part re-derives the interior footprint count from the published rings (verify-r4 B5 in the
    GLB). The structure extras of a fixture model with one block, its count forged."""
    model, rec = _block_record()
    sx = model.compact_extras()
    assert not [v for v in st._audit_glb_d25({}, sx, {"snap": {"accepted": True}}) if "footprint" in v["why"]]
    a = next(x for x in sx["adoptions"] if x["id"] == rec["id"])
    a["roof"]["cells_footprint"] = 30                       # under the bounds (40-60); as many as its 30 seen cells
    a["roof_check"] = C.block_roof_problem(a["roof"], a["H"], 0.0)[1]
    assert C.check_record(a, {a["id"]: a}) is None
    v = st._audit_glb_d25({}, sx, {"snap": {"accepted": True}})
    assert any("outside what the block's outline" in x["why"] for x in v), v
    # with parts withheld from the GLB only the upper bound is checked: the low count passes, a high one does not
    sx["withheld"] = ["x"]
    assert not any("outside what the block's outline" in x["why"] for x in st._audit_glb_d25({}, sx, {"snap": {"accepted": True}}))
    a["roof"]["cells_footprint"] = 1000
    a["roof_check"] = C.block_roof_problem(a["roof"], a["H"], 0.0)[1]
    assert any("outside what the block's outline" in x["why"] for x in st._audit_glb_d25({}, sx, {"snap": {"accepted": True}}))


# ---------------------------------------------------------------- U2: a rough wall with tangent crowns; a firewall


def _crowns2_scene(r=4.0, zc=12.5, vs=1.0, nx=5, rows=4, sig=0.12, ov=0.0, wsig=0.2, seed=11):
    """verify-r4 attack_crowns2.py 'sphere': the map-right 10 m flat hall (30 x 20 m), walls seen 1-10 m with relief
    (offsets N(0, wsig)), roof unseen; spherical crowns (r, centre zc, vertical scale vs) along the long walls (and
    the short ones with rows 4), their outer edge `ov` m beyond the wall plane, points with noise `sig`."""
    rng = np.random.default_rng(seed)
    P = []
    for x0, y0, x1, y1 in ((-15, -10, 15, -10), (15, -10, 15, 10), (15, 10, -15, 10), (-15, 10, -15, -10)):
        t, z = rng.uniform(0, 1, 1500), rng.uniform(1, 10, 1500)
        u = np.array([x1 - x0, y1 - y0], float)
        u /= np.linalg.norm(u)
        nrm = np.array([u[1], -u[0]])
        off = rng.normal(0, wsig, 1500) if wsig > 0 else np.zeros(1500)
        P.append(np.c_[x0 + t * (x1 - x0) + off * nrm[0], y0 + t * (y1 - y0) + off * nrm[1], z])
    centres = [(cx, sy * (10 + ov - r)) for sy in (-1, 1) for cx in np.linspace(-15 + 30 / (2 * nx), 15 - 30 / (2 * nx), nx)]
    if rows == 4:
        ys = np.linspace(-10 + 20 / (2 * max(1, nx - 1)), 10 - 20 / (2 * max(1, nx - 1)), max(1, nx - 1))
        centres += [(sx * (15 + ov - r), cy) for sx in (-1, 1) for cy in ys]
    a8 = np.radians(np.arange(0, 360, 45) + 22.5)
    extra = []
    for cx, cy in centres:
        v = rng.normal(size=(1500, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        v = v[v[:, 2] > -0.3]
        P.append(np.c_[cx + r * v[:, 0], cy + r * v[:, 1], zc + vs * r * v[:, 2]] + rng.normal(0, sig, (len(v), 3)))
        extra.append((np.c_[cx + r * np.cos(a8), cy + r * np.sin(a8)], zc - vs * r * 0.3, zc + vs * r))
    a, rr = rng.uniform(0, 2 * np.pi, 3000), rng.uniform(25, 45, 3000)
    P.append(np.c_[rr * np.cos(a), rr * np.sin(a), rng.normal(0, 0.05, 3000)])
    return np.vstack(P), extra


def _run_hall(P, extra, nonmodel=()):
    """D25-H on the map-right 10 m hall (D15: disputed) with the points P; the sky masks of the truth: the hall, the
    crowns (extra) and any non-model neighbour."""
    Q = np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]
    cams = tj.cameras(n=16, radius=75.0)
    sky = tj.sky_masks(TO._chall(list(extra) + list(nonmodel)), cams)
    model = TO._chall()
    model.frame = FRAME_OK
    model.part("w1").params["height"].state = S.STATE_DISPUTED
    res = M.correct_landmark(model, M.Evidence25(cams, {"A": sky, "B": sky}, P, Q, lines=None, grade=C.GRADE_SNAP),
                             only=["w1"])
    return model, res, res["decisions"]["w1"]


def test_tangent_crowns_on_a_rough_wall_do_not_extend_it():
    """verify-r4 MEDIUM: the wall's relief (sigma 0.2 m) raised the above-eave spread limit to its 0.25 m cap and the
    tangent crowns (0.16-0.25 m) passed as wall: W_base 14.5, a lower bound 13.821 m published on the map-right 10 m
    hall. The cap is 0.18 m now and an off-wall bin ends the wall."""
    P, extra = _crowns2_scene()
    model, res, d = _run_hall(P, extra)
    f = d["final"]
    assert d["pre"]["candidate"] and C.SFM in d["pre"]["agreeing"], d["pre"]["why"]   # U1 met (the crowns' fit)
    assert not f["fires"] and "walls" in f["why"] and f["walls"]["W_base"] <= 11.0, f
    assert not res["records"] and model.part("w1").params["height"].value == 10.0


@pytest.mark.slow
@pytest.mark.parametrize("kw", [{"seed": 12}, {"seed": 13}, {"sig": 0.15}, {"ov": 0.2},
                                {"r": 3.5, "vs": 1.2, "rows": 2},             # before 13.339 (passed the 0.18 cap alone)
                                {"wsig": 0.0, "sig": 0.05, "r": 3.5, "vs": 1.2}])   # before 13.677 (a smooth wall)
def test_rough_wall_crown_variants_change_nothing(kw):
    P, extra = _crowns2_scene(**kw)
    model, res, d = _run_hall(P, extra)
    assert not res["records"] and model.part("w1").params["height"].value == 10.0, d["final"]


def test_a_non_model_neighbours_firewall_over_the_roof_is_not_the_halls_wall():
    """verify-r4 MEDIUM attack_crowns_nb.py: crowns over the roof (U1 met) and a 15 m neighbour OSM does not give
    (not a part of the model) abutting half the north wall: its firewall above the hall's 10 m eave lies in the wall
    plane, the party wall below shows nothing. Before: W_base 15.0 (the firewall), a lower bound 13.471 m published.
    Above the eave only the cells where the wall is seen below it count."""
    P, extra = TO._crowns_points("over")
    rng = np.random.default_rng(11)
    keep = ~((np.abs(P[:, 1] - 10.0) < 1e-6) & (P[:, 0] < 0.0) & (P[:, 2] <= 10.0))       # the party wall: no points
    P = P[keep]
    nb = np.array([[-15.0, 10.0], [0.0, 10.0], [0.0, 18.0], [-15.0, 18.0]])
    fire = [TO._wall((-15, 10), (0, 10), 10.0, 15.0, 900, rng), TO._wall((0, 10), (0, 18), 0.0, 15.0, 1500, rng),
            TO._wall((-15, 18), (0, 18), 0.0, 15.0, 1500, rng), TO._wall((-15, 10), (-15, 18), 0.0, 15.0, 800, rng)]
    model, res, d = _run_hall(np.vstack([P, *fire]), extra, [(nb, 0.0, 15.0)])
    f = d["final"]
    assert not f["fires"] and "walls" in f["why"] and f["walls"]["W_base"] < 12.0, f          # before: 15.0
    assert not res["records"]
    # the wall reading itself: the north edge's firewall points stand on cells with no wall below the eave
    rr, _ = M.wall_reading(model, model.built_part("w1"), np.vstack([P, *fire]))
    north = next(e for e in rr["edges"] if e["edge"] == 2)
    assert north["top_m"] < 12.0, north                                                    # before: 15.0


def test_wall_planar_continuity_cap_and_end():
    """crowd3d.correct_measure.wall_planar, one guard at a time: (continuity) above the eave a bin's points on cells
    without the wall below are dropped and its numbers taken on the rest; (cap) a rough wall's spread raises the limit
    no higher than 0.18 m; (end) a bin seen across the edge but off the wall ends the run -- the run may not bridge it
    to on-wall bins above; (V5b) the bins below the eave that set the reference need the run's own cover."""
    rng = np.random.default_rng(2)
    ncell, eave = 20, 10.0

    def wall(z0, z1, n, cells=range(20), sd=0.03):
        c = rng.choice(list(cells), n)
        return rng.uniform(z0, z1, n), rng.normal(0, sd, n), c
    def run(*parts):
        z = np.concatenate([p[0] for p in parts])
        d = np.concatenate([p[1] for p in parts])
        c = np.concatenate([p[2] for p in parts])
        keep, pl = M.wall_planar(z, d, c, ncell, 0.0, eave)
        top = M.wall_top_run(z[keep], c[keep], ncell, 0.0)
        return keep, pl, (top or {}).get("top_m")
    # continuity: wall below on cells 0-9 only; above the eave a flush surface on cells 10-19 (a firewall): dropped
    keep, pl, top = run(wall(1, 10, 3000, range(10)), wall(10, 15, 1500, range(10, 20)))
    assert top <= 10.0 and pl["wall_cells"] == 10 and pl["off_cell_points"] == 1500, (top, pl)
    # the same surface over the wall's own cells continues it (a wall really rising above the map's eave)
    keep, pl, top = run(wall(1, 10, 3000, range(10)), wall(10, 15, 1500, range(10)))
    assert top == pytest.approx(15.0, abs=0.1), (top, pl)
    # cap: below the eave sd 0.25 (MAD about 0.25: the uncapped limit would be 0.31, the old cap 0.25); above, sd
    # 0.12 (on the wall) continues, sd 0.21 (MAD about 0.21: under the old cap, over the new 0.18) does not
    keep, pl, top = run(wall(1, 10, 4000, sd=0.25), wall(10, 15, 4000, sd=0.12))
    assert pl["mad_max"] == pytest.approx(C.WALL_ABOVE_MAD_MAX) and C.WALL_ABOVE_MAD_MAX == 0.18 and top > 14.5, (top, pl)
    keep, pl, top = run(wall(1, 10, 4000, sd=0.25), wall(10, 15, 4000, sd=0.21))
    assert top <= 10.6, (top, pl)
    # end: on-wall bins with a 1 m band of crowns (off the plane, seen across the edge) between them: the run stops
    # at the band (it used to bridge the dropped bins under the 2 m gap rule)
    band = (rng.uniform(11.0, 12.0, 800), rng.uniform(-0.5, 0.5, 800), rng.choice(range(20), 800))
    keep, pl, top = run(wall(1, 11, 3000), band, wall(12, 15, 1200))
    assert pl["ended_at"] == pytest.approx(11.0) and top <= 11.0, (top, pl)
    # ... but a gap where the wall is just not seen (too few cells: a dark band, no points) is still bridged
    sparse = (rng.uniform(11.0, 12.0, 30), rng.normal(0, 0.03, 30), rng.choice(range(3), 30))
    keep, pl, top = run(wall(1, 11, 3000), sparse, wall(12, 15, 1200))
    assert pl["ended_at"] is None and top == pytest.approx(15.0, abs=0.1), (top, pl)
    # ... while a band over part of the edge ends it once it holds the run's own cover (6 of 20 cells; the run needs
    # 4, the cells reference for an on-wall bin -- half the median below the eave -- would be 10: mutant W4)
    part = (rng.uniform(11.0, 12.0, 600), rng.uniform(-0.5, 0.5, 600), rng.choice(range(6), 600))
    keep, pl, top = run(wall(1, 11, 3000), part, wall(12, 15, 1200))
    assert pl["ended_at"] == pytest.approx(11.0) and top <= 11.0, (top, pl)
    # V5b: the wall seen in five 0.5 m bands 2 m apart (the run still bridges them) and a drainpipe in one cell all
    # the way up: the pipe's 1-cell bins must not set the reference (with them the cells reference falls to 1 and a
    # 6-cell strip above the eave passes as wall)
    bands = [wall(z, z + 0.5, 400) for z in (1.0, 3.0, 5.0, 7.0, 9.0)]
    pipe = (rng.uniform(1, 10, 900), rng.normal(0, 0.03, 900), np.zeros(900, int))
    keep, pl, top = run(*bands, pipe, wall(10, 15, 900, range(6)))
    assert pl["cells_ref"] >= 15 and top <= 10.0, (top, pl)


def test_a_slightly_taller_abutting_part_covers_the_cells_it_abuts():
    """Mutant N16 (covered cells only for parts taller than the part's top + 1 m): a part whose top lies above this
    part's eave - 0.5 m and whose base is under it covers the cells it abuts -- a 10.8 m neighbour beside the 10 m
    annex's north edge over 6 of its 14 m."""
    rng = np.random.default_rng(1)
    T = np.array([[0.0, 10.0], [6.0, 10.0], [6.0, 16.0], [0.0, 16.0]])
    A = np.array([[0.0, 0.0], [14.0, 0.0], [14.0, 10.0], [0.0, 10.0]])
    model = S.build_structure("abut", [S.PartSpec("a", "yes", A, [], {"height": "10"}),
                                       S.PartSpec("t", "yes", T, [], {"height": "10.8"})], [A.copy()], {})
    pts = [TO._w(rng, (0, 0), (14, 0), 1, 10, 1500), TO._w(rng, (14, 0), (14, 10), 1, 10, 1000),
           TO._w(rng, (14, 10), (0, 10), 1, 10, 1400), TO._w(rng, (0, 10), (0, 0), 1, 10, 1000)]
    r, _ = M.wall_reading(model, model.built_part("a"), np.vstack(pts))
    assert [e.get("cells_covered") for e in r["edges"] if e.get("cells_covered")] == [6], r["edges"]
