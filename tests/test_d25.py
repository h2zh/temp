"""Decision D25 (product owner, 2026-10-25): strong photo evidence corrects the OSM frame (docs/STRUCTURE_FIRST.md
§2.4.8). The rule is a function of numbers (crowd3d.correct): these tests feed it the numbers a measurement would give
(the Aachen choir's shape: map 32 m, skyline 46.7 m and 3-D lines 47.1 m agreeing, SfM points below) and change one
number at a time; the records carry their provenance and the map's value, and reverting gives the fresh OSM model back
exactly."""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest
import test_skyline_judge as tj

from crowd3d import correct as C
from crowd3d import judge as J
from crowd3d import structure as S
from crowd3d import stylize as st

MAP = 32.0
SK, SF, LN = C.SKYLINE, C.SFM, C.LINES


def sky(top=46.7, p05=45.2, p95=48.2, n_vis_at=72, n_sky_at=56, n_vis_map=58, n_sky_map=40, at_bound=False,
        top_b=None, eave=None):
    m = {"top": top, "p05": p05, "p95": p95, "at_bound": at_bound, "n_vis_at": n_vis_at, "n_sky_at": n_sky_at}
    mb = dict(m, top=top if top_b is None else top_b)
    if eave is not None:
        m["eave"] = mb["eave"] = eave
    return {"masks": {"A+R": m, "Dm+R": mb}, "n_vis_map": n_vis_map, "n_sky_map": n_sky_map}


def quad(vals=(40.93, 41.5, 43.0, 38.0), pts=(120, 90, 60, 40)):
    return {"quadrant_q98": list(vals), "quadrant_points": list(pts)}


def lines(value=47.13, iv=(46.91, 47.32), n_photos=12, n_lines=9, parameter="top", **kw):
    return {"records": [{"parameter": parameter, "value": value, "implied_top": value, "interval": list(iv),
                         "n_photos": n_photos, "n_lines": n_lines, "at_bound": False, **kw}]}


def choir(**over):
    ch = {SK: sky(), SF: {"quad": quad(), "fit": None, "ceiling": False}, LN: lines()}
    ch.update(over)
    return ch


PART = {"id": "w505513556", "map_top": MAP, "z0": 0.0, "roof_h": 6.0, "pitched": True, "min_width": 16.0}
WALLS = [{"edges": [67, 199], "normals_deg": [10.0, 190.0], "points": [435, 4721], "map_sep_m": 46.8, "k": 1.0064},
         {"edges": [1, 67], "normals_deg": [100.0, 281.0], "points": [900, 435], "map_sep_m": 38.9, "k": 1.0065}]
SCALE = C.scale_from_walls(WALLS)


def make_box(values: dict[str, float], halves: dict[str, float] | None = None, grade=C.GRADE_SNAP, sens=0.4,
             lose=None, eave: dict[str, float] | None = None, aliases=(180.0,), **sig):
    """A box whose samples move each channel by sens x (x + y) metres (+ 0.2 m per degree of yaw); a snap-grade box
    samples the symmetric aliases of the fixtures' rectangular main outlines (180 deg; the values as at the
    registration: a symmetric building measures the same under its alias)."""
    sigma = C.box_sigma(grade, ground_measured=True, two_gravity_diff_deg=None, estimate_up_boot_deg=0.1,
                        **({"gps_residual_median_m": 8.0, "reported_yaw_sigma_deg": 3.1} if grade == C.GRADE_CROWD else {}))
    sigma.update(sig)
    halves = halves or {c: 0.5 for c in values}
    samples = []
    alts = [{"yaw": float(a)} for a in aliases] if grade == C.GRADE_SNAP else None
    for i, s in enumerate(C.box_poses(sigma, alternatives=alts)):
        p = s["pose"]
        d = 0.0 if s["kind"] == "snap-alternative" else sens * (p["x"] + p["y"]) + 0.2 * p["yaw"]
        vals = {c: {"value": round(v + d, 3), "half": halves[c]} for c, v in values.items()}
        if lose and i in lose:
            for c in lose[i]:
                vals[c] = None
        s = dict(s, values=vals)
        if eave:
            s["eave"] = {c: {"value": round(v + d, 3), "half": 0.4} for c, v in eave.items()}
        samples.append(s)
    return {"grade": grade, "k": C.K_BOX, "sigma": sigma, "samples": samples, "inputs": dict(BOX_INPUTS)}


# what the fixture box's sigma is sized from (recorded with the box as provenance)
BOX_INPUTS = {"ground": {"measured": True, "q01_m": 0.12, "points": 800, "camera_median_z_m": 1.7},
              "estimate_up": {"sigma_deg": 0.1}, "gravity_vp": None}
# G5 'no regression' numbers a correction carries (the engine measures them on the final model; crowd3d.correct)
G5_OK = {"masks": {"A+R": {"photos": 40, "prior_px": 12.0, "corrected_px": 9.5},
                   "Dm+R": {"photos": 38, "prior_px": 11.0, "corrected_px": 9.9}}, "passed": True}


def g5(model, recs):
    """Attach the G5 numbers (each record's own part's) to freshly applied records (a list from apply_height or one
    record from apply_block)."""
    recs = recs if isinstance(recs, list) else [recs]
    C.attach_g5(model, [r["id"] for r in recs], {"baseline": "test", "masks": G5_OK["masks"],
                                                 "parts": {r["part"]: G5_OK for r in recs}})
    return recs if len(recs) != 1 else recs


def decide(ch=None, part=None, box=None, scale=SCALE):
    pre = C.rule_h_pre(part or PART, ch or choir())
    if not pre.get("candidate"):
        return pre, {"fires": False, "why": pre["why"]}
    box = box if box is not None else make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]})
    return pre, C.rule_h_box(pre, box, scale, part or PART)


# ---------------------------------------------------------------- D25-H: the rule


def test_strong_agreeing_channels_adopt_the_photo_value_when_the_box_bound_clears_the_midpoint():
    pre, fin = decide()
    assert pre["candidate"] and sorted(pre["agreeing"]) == [LN, SK]
    assert pre["V"] == pytest.approx((46.7 + 47.13) / 2, abs=0.01)
    assert pre["verdicts"][SF]["valid"] and pre["verdicts"][SF]["value"] < pre["V"]      # SfM below: up-only, no veto
    assert fin["fires"] and fin["adopted"] == "measured" and fin["provenance"] == C.PROV_FULL
    assert fin["value"] == pytest.approx(pre["V"]) and fin["LB"] >= fin["midpoint"]
    assert fin["LB"] - MAP >= C.tol_of(MAP)


def test_a_bound_short_of_the_midpoint_adopts_only_the_bound():
    """A sensitive registration: the box's lower bound clears the map by TOL but not the midpoint: only LB."""
    pre = C.rule_h_pre(PART, choir())
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=1.3)
    fin = C.rule_h_box(pre, box, SCALE, PART)
    assert fin["fires"] and fin["adopted"] == "lower bound" and fin["provenance"] == C.PROV_LB
    assert MAP + C.tol_of(MAP) <= fin["value"] < fin["midpoint"] and fin["value"] == fin["LB"]
    # even more sensitive: the guaranteed change falls below TOL -> refused
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=2.4)
    fin = C.rule_h_box(pre, box, SCALE, PART)
    assert not fin["fires"] and "guaranteed change" in fin["why"]


def test_the_midpoint_rule():
    assert C.adopt_value(40.0, 36.0, 32.0) == ("measured", 40.0)
    assert C.adopt_value(40.0, 35.9, 32.0) == ("lower bound", 35.9)


def test_osm_right_means_no_correction():
    """Every channel within TOL of the map (or two channels off but a third at the map): nothing happens."""
    near = choir(**{SK: sky(top=33.0, p05=32.0, p95=34.0), LN: lines(31.5, (31.3, 31.7))})
    pre, fin = decide(near)
    # the SfM points' all-round top alone says up, and it never agrees: no channel off the map counts
    assert not pre["candidate"] and not fin["fires"] and "0 channel(s) off the map" in pre["why"]
    # the Westwerk: skyline and 3-D lines say up, the SfM points sit at the map's value (within TOL): a contradiction
    ww = choir(**{SF: {"quad": quad((33.0, 32.5, 31.9, 30.0)), "ceiling": False}})
    pre, _ = decide(ww)
    assert not pre["candidate"] and "contradicted" in pre["why"] and SF in pre["why"]


def test_one_channel_is_never_enough():
    pre, fin = decide(choir(**{LN: {"records": []}}))
    assert not fin["fires"] and not pre["candidate"] and "1 channel(s) off the map" in pre["why"]   # SfM: never agrees


def test_skyline_guards_each_refuse_alone():
    base = C.skyline_verdict(sky(), MAP)
    assert base["valid"]
    for kw, why in (({"at_bound": True}, "bound"), ({"top_b": 49.5}, "masks' tops"), ({"n_vis_map": 2}, "map value"),
                    ({"n_vis_at": 9}, "measured value"), ({"n_sky_at": 4}, "sky above"),
                    ({"p05": 40.0, "p95": 52.0}, "PREC")):
        v = C.skyline_verdict(sky(**kw), MAP)
        assert not v["valid"] and why in v["why"], (kw, v)
    # below the map (a reading that may veto an upward change) the sky must be above the part at the map value
    d = C.skyline_verdict(sky(top=25.0, p05=24.0, p95=26.0, n_sky_map=4), MAP)
    assert not d["valid"] and "cannot say lower" in d["why"]
    assert C.skyline_verdict(sky(top=25.0, p05=24.0, p95=26.0, n_sky_map=5), MAP)["valid"]
    one = sky()
    one["masks"].pop("Dm+R")
    assert not C.skyline_verdict(one, MAP)["valid"]


FIT = {"top": 46.44, "top_p05": 46.36, "top_p95": 47.72, "inlier_share": 0.52, "cells": 60,
       "quadrant_cells": [10, 12, 9, 8], "at_bound": False, "eave": 37.44, "eave_p05": 37.0, "eave_p95": 38.0,
       "below_share": 0.03, "below_cells": 60, "below_over": "interior"}


def test_sfm_and_lines_guards():
    fit = dict(FIT)
    assert C.sfm_verdict({"fit": fit}, MAP, True)["how"] == "roof-fit"
    assert C.sfm_verdict({"fit": fit}, MAP, False)["how"] == "surface-fit"
    assert C.sfm_verdict({"fit": fit}, MAP, True)["agrees"]
    for bad in ({"inlier_share": 0.39}, {"cells": 29}, {"quadrant_cells": [10, 12, 2, 2]}, {"at_bound": True},
                {"top_p05": 40.0, "top_p95": 49.0}, {"below_share": 0.26}, {"below_share": None}):
        v = C.sfm_verdict({"fit": {**fit, **bad}}, MAP, True)
        assert not v["valid"] or v["how"] != "roof-fit", bad
    assert C.sfm_verdict({"fit": {**fit, "below_share": 0.25}}, MAP, True)["how"] == "roof-fit"
    assert C.sfm_verdict({"quad": quad(pts=(120, 90, 19, 40))}, MAP, False)["value"] == 38.0
    assert C.sfm_verdict({"quad": quad(pts=(120, 90, 19, 40))}, MAP, False)["agrees"] is False
    assert not C.sfm_verdict({"quad": quad(pts=(120, 19, 19, 40))}, MAP, False)["valid"]
    for kw, why in (({"n_photos": 2}, "photos"), ({"n_lines": 1}, "line"), ({"at_bound": True}, "window"),
                    ({"vote_withheld": "edges disagree"}, "withheld"), ({"iv": (40.0, 47.5)}, "PREC")):
        v = C.lines_verdict(lines(**kw), MAP)
        assert not v["valid"] and why in v["why"], (kw, v)


def test_contradictions_inside_and_outside_the_agreeing_set():
    # a third channel on the same side but outside the set contradicts (SfM ABOVE the value does)
    pre, _ = decide(choir(**{SF: {"quad": quad((52.0, 53.0, 51.5, 50.0)), "ceiling": False}}))
    assert not pre["candidate"] and "contradicts" in pre["why"]
    # channels on both sides of the map
    pre, _ = decide(choir(**{LN: lines(24.0, (23.8, 24.2))}))
    assert not pre["candidate"] and "both sides" in pre["why"]
    # two channels off the map that disagree with each other
    pre, _ = decide(choir(**{LN: lines(52.0, (51.8, 52.2))}))
    assert not pre["candidate"] and "contradicts" in pre["why"] or "no two" in pre["why"]
    # the points stop at the map's top all round: a ceiling vetoes an upward change
    pre, _ = decide(choir(**{SF: {"quad": quad(), "ceiling": True}}))
    assert not pre["candidate"] and "ceiling" in pre["why"]


def test_a_roof_fit_above_the_agreeing_pair_contradicts_it():
    """The SfM roof-surface fit is an agreeing channel: when the skyline and the 3-D lines agree and the fit reads more
    than AGREE above them, the largest agreeing set is the pair and the fit outside it contradicts (an up-only channel
    exempts only values BELOW the set)."""
    fit = dict(FIT, top=50.5, top_p05=50.2, top_p95=50.8)
    pre = C.rule_h_pre(PART, choir(**{SF: {"fit": fit, "quad": quad()}}))
    assert pre["verdicts"][SF]["valid"] and pre["verdicts"][SF]["agrees"]
    assert not pre["candidate"] and "contradicts" in pre["why"] and SF in pre["why"], pre["why"]
    low = dict(FIT, top=44.0, top_p05=43.7, top_p95=44.3)              # below the pair: supports the direction
    pre = C.rule_h_pre(PART, choir(**{SF: {"fit": low, "quad": quad()}}))
    assert pre["candidate"] and sorted(pre["agreeing"]) == [LN, SK], pre["why"]


def test_d25_only_raises():
    """Two channels agreeing below the map correct nothing: D25-H only raises."""
    part = dict(PART, map_top=40.0, roof_h=0.0, pitched=False)
    ch = {SK: sky(top=30.0, p05=29.5, p95=30.5, n_sky_map=12), LN: lines(30.3, (30.1, 30.5)),
          SF: {"quad": quad((28.0, 27.5, 27.0, 20.0))}}
    pre, fin = decide(ch, part)
    assert not pre["candidate"] and not fin["fires"] and "only raises" in pre["why"], pre["why"]
    assert "0 channel(s) off the map" in pre["why"] and pre.get("V") is None


def test_slender_parts_never_move():
    pre, fin = decide(part=dict(PART, min_width=9.0, map_top=32.0))
    assert not fin["fires"] and pre.get("slender") and "slender" in pre["why"]


def test_parents_and_taller_neighbours_first():
    pre, _ = decide(part=dict(PART, parent_ok=False, parent_why="its parent w1 is disputed"))
    assert not pre["candidate"] and pre["deferred"] and "parents first" in pre["why"]
    pre, _ = decide(part=dict(PART, pending=["w505513557"]))
    assert not pre["candidate"] and pre["deferred"] and "taller adjacent" in pre["why"]


def test_the_scale_must_be_established_by_the_buildings_own_walls():
    assert SCALE["established"] and SCALE["k"] == 1.0064 and SCALE["edges"] == [67, 199]
    assert SCALE["k_err"] == pytest.approx(0.0064 + 0.5 / 46.8, abs=1e-5)
    # an injected x1.03 / x1/1.10 scale: the walls measure it and the scale is not established -> nothing fires
    for k in (1.033, 0.917):
        sc = C.scale_from_walls([dict(w, k=k) for w in WALLS])
        assert not sc["established"]
        pre, fin = decide(scale=sc)
        assert not fin["fires"] and "scale is not established" in fin["why"]
    # the pairs disagree (a shifted registration): not established
    assert not C.scale_from_walls([WALLS[0], dict(WALLS[1], k=0.89)])["established"]
    # too close, too few points, not opposite: no qualifying pair
    for bad in ({"map_sep_m": 19.0}, {"points": [199, 900]}, {"normals_deg": [10.0, 140.0]}):
        assert not C.scale_from_walls([dict(WALLS[0], **bad)])["established"], bad
    # the scale-free value V / k must still be off the map
    sc = dict(SCALE, k=1.45)
    pre, fin = decide(scale=sc)
    assert not fin["fires"] and "scale-free value" in fin["why"]
    # route 1 (D15 similarity)
    sim = C.scale_from_similarity(J.scale_status({"similarity": {"measured": True, "k": 1.01, "at_bound": False}}), 40.0)
    assert sim["established"] and sim["k_err"] == pytest.approx(0.01 + 0.5 / 40.0, abs=1e-5)
    assert not C.scale_from_similarity(J.scale_status({}, 1.10), 40.0)["established"]
    best = C.scale_anchor([C.scale_from_walls(WALLS), sim])
    assert best["route"] in ("opposite-walls", "similarity") and best["established"]


def test_the_box_samples_the_base_every_axis_end_corners_and_interior_points():
    sig = C.box_sigma(C.GRADE_SNAP, ground_measured=True, estimate_up_boot_deg=0.1)
    axes = C.box_axes(sig)
    half = {a: C.K_BOX * float(sig[C.AXIS_SIGMA[a]]) for a in axes}
    poses = C.box_poses(sig, alternatives=[{"yaw": 180.0}])
    kinds = [p["kind"] for p in poses]
    assert axes == ["x", "y", "yaw", "gx", "gy"] and kinds.count("base") == 1 and kinds.count("snap-alternative") == 1
    assert kinds.count("axis") == 2 * len(axes) and kinds.count("interior") == C.BOX_INTERIOR
    for a in axes:
        for sg in (-1, 1):
            assert any(p["kind"] == "axis" and p["pose"][a] == pytest.approx(sg * half[a]) for p in poses), (a, sg)
    corners = [p for p in poses if p["kind"] == "corner"]
    assert all(abs(abs(p["pose"][a]) - half[a]) < 1e-9 for p in corners for a in axes)
    assert len({tuple(np.sign(p["pose"][a]) for a in axes) for p in corners}) == C.BOX_CORNERS
    assert all(abs(p["pose"][a]) <= half[a] + 1e-9 for p in poses if p["kind"] == "interior" for a in axes)


def test_a_sample_losing_every_agreeing_channel_refuses_but_losing_one_does_not():
    pre = C.rule_h_pre(PART, choir())
    vals = {c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}
    b = make_box(vals, lose={3: [LN]})
    fin = C.rule_h_box(pre, b, SCALE, PART)
    assert fin["fires"] and fin["box"]["lost"] == [[3, LN]]
    b = make_box(vals, lose={3: [LN, SK]})
    fin = C.rule_h_box(pre, b, SCALE, PART)
    assert not fin["fires"] and "lost its measurement" in fin["why"]


def test_crowd_grade_registrations_do_not_correct_heights():
    pre = C.rule_h_pre(PART, choir())
    b = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, grade=C.GRADE_CROWD, sens=0.05)
    fin = C.rule_h_box(pre, b, SCALE, PART)
    assert not C.CROWD_HB_ENABLED and not fin["fires"] and "crowd-grade" in fin["why"]
    s = C.box_sigma(C.GRADE_CROWD, gps_residual_median_m=12.0, reported_yaw_sigma_deg=3.19)
    assert s["x_m"] == 6.0 and s["yaw_deg"] == pytest.approx(4.785) and s["z_m"] == 1.5
    s = C.box_sigma(C.GRADE_SNAP, snap_yaw_sigma_deg=0.2, ground_measured=False, estimate_up_boot_deg=0.38)
    assert s["x_m"] == 1.0 and s["yaw_deg"] == 0.5 and s["z_m"] == 1.5 and s["gravity_deg"] == pytest.approx(1.52)
    assert C.box_sigma(C.GRADE_SNAP, two_gravity_diff_deg=0.2)["gravity_deg"] == 0.5


def test_the_eave_is_adopted_only_when_two_readings_agree_and_pass_the_box():
    fit = dict(FIT)
    ch = choir(**{SK: sky(eave=37.7), SF: {"fit": fit, "quad": quad()}, LN: {"records": []}})
    part = dict(PART, id="w505513557")
    pre = C.rule_h_pre(part, ch)
    assert pre["candidate"] and sorted(pre["agreeing"]) == [SF, SK]
    vals = {c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}
    box = make_box(vals, eave={SK: 37.7, SF: 37.44})
    fin = C.rule_h_box(pre, box, SCALE, part)
    ev = C.rule_eave(ch, part, pre["V"], box, SCALE)
    assert fin["adopted"] == "measured" and ev["adopt"] and ev["roof_height"] == pytest.approx(pre["V"] - ev["value"], abs=0.01)
    # one eave reading only: the map's roof height is kept
    ch1 = copy.deepcopy(ch)
    ch1[SK]["masks"]["Dm+R"]["eave"] = 40.0
    assert not C.rule_eave(ch1, part, pre["V"], box, SCALE)["adopt"]
    # a sensitive eave: its bound misses the midpoint
    box2 = make_box(vals, eave={SK: 37.7, SF: 37.44})
    for s in box2["samples"]:
        s["eave"] = {c: {"value": r["value"] - 3.0 * abs(s["pose"]["x"]), "half": 0.4} for c, r in s["eave"].items()}
    assert not C.rule_eave(ch, part, pre["V"], box2, SCALE)["adopt"]


def test_the_box_tol_lock_refuses_a_bound_between_the_map_and_map_plus_tol():
    """The central lock: a box lower bound past the midpoint but within TOL of the map is refused."""
    up = choir(**{SK: sky(top=37.6, p05=37.3, p95=37.9), LN: lines(37.6, (37.3, 37.9)),
                  SF: {"quad": quad((37.0, 36.5, 36.0, 35.5)), "fit": None}})
    pre = C.rule_h_pre(PART, up)
    assert pre["candidate"]
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.0, halves={SK: 0.3, LN: 0.3})
    fin = C.rule_h_box(pre, box, SCALE, PART)
    assert 0.0 < fin["LB"] - MAP < C.tol_of(MAP) and fin["LB"] >= fin["midpoint"], fin
    assert fin["LB"] - MAP > C.tol_of(MAP) / 2 and not fin["fires"] and "guaranteed change" in fin["why"]
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.0, halves={SK: 0.9, LN: 0.9})
    fin = C.rule_h_box(pre, box, SCALE, PART)                     # short of the midpoint too: still refused
    assert 0.0 < fin["LB"] - MAP < C.tol_of(MAP) and fin["LB"] < fin["midpoint"]
    assert not fin["fires"] and "guaranteed change" in fin["why"]


def test_each_eave_trigger_refuses_on_its_own():
    """A reading within TOL of the map's eave, an eave whose box bound clears the midpoint but not TOL, an eave whose
    scale-free value E / k lies within TOL: each refuses alone; the same numbers otherwise adopt."""
    part = dict(PART, id="w505513557")                           # map top 32, roof 6: the map's eave 26 m, TOL 2.6
    V = 46.9

    def ch(e_sky, e_sfm):
        return {SK: sky(eave=e_sky), SF: {"fit": dict(FIT, eave=e_sfm, eave_p05=e_sfm - 0.2, eave_p95=e_sfm + 0.2)},
                LN: {"records": []}}

    vals = {SK: 46.7, SF: 46.44}
    ok = C.rule_eave(ch(37.7, 37.44), part, V, make_box(vals, eave={SK: 37.7, SF: 37.44}), SCALE)
    assert ok["adopt"], ok["why"]
    near = C.rule_eave(ch(28.0, 28.4), part, V, make_box(vals, eave={SK: 28.0, SF: 28.4}), SCALE)
    assert not near["adopt"] and "an eave reading lies within TOL" in near["why"], near["why"]
    # E 30.8: every reading 4.8 m above the map's eave (> TOL 2.6); a tight box: LB = 30.8 - 0.3 (yaw) - 1.5 (ground)
    # - k_err E = 28.47, past the midpoint 28.4 but only 2.47 m (< TOL) above the map's eave
    tight = make_box(vals, eave={SK: 30.8, SF: 30.8}, sens=0.0)
    for s_ in tight["samples"]:
        for c in s_["eave"]:
            s_["eave"][c]["half"] = 0.0
    e = C.rule_eave(ch(30.8, 30.8), part, V, tight, SCALE)
    assert not e["adopt"] and "guaranteed change" in e["why"], e
    assert e["box"]["LB"] >= 0.5 * (30.8 + 26.0) and e["box"]["LB"] - 26.0 < C.tol_of(26.0)
    # E 31.5 with k 1.12: E / k = 28.1 m, within TOL of the map's eave although every reading is 5.5 m above it
    e2 = C.rule_eave(ch(31.5, 31.5), part, V, make_box(vals, eave={SK: 31.5, SF: 31.5}), dict(SCALE, k=1.12))
    assert not e2["adopt"] and "scale-free" in e2["why"], e2
    assert "scale-free" not in str(C.rule_eave(ch(31.5, 31.5), part, V, make_box(vals, eave={SK: 31.5, SF: 31.5}),
                                               SCALE).get("why"))


def test_an_upward_change_needs_a_reading_of_the_parts_own_roof():
    """U1: the 3-D lines' EAVE and the skyline read the top of anything standing at the outline (a scaffold net hiding
    the walls, a hoarding); an upward change needs the part's roof in the agreeing set -- the roof-surface fit, or the
    3-D lines' apex / ridge (a 'top' record)."""
    pre, _ = decide(choir(**{LN: lines(parameter="eave")}))
    assert not pre["candidate"] and pre.get("V") is None and "reads the part's own roof" in pre["why"], pre["why"]
    assert C.rule_h_pre(PART, choir())["candidate"]                                # the apex: the roof's own lines
    fit_eave = choir(**{LN: lines(parameter="eave"), SF: {"fit": dict(FIT), "quad": quad()}})
    pre = C.rule_h_pre(PART, fit_eave)
    assert pre["candidate"] and C.SFM in pre["agreeing"] and LN in pre["agreeing"], pre["why"]
    fit_sky = choir(**{LN: {"records": []}, SF: {"fit": dict(FIT), "quad": quad()}})
    assert C.rule_h_pre(PART, fit_sky)["candidate"]
    # the roof reading guard on its own: an SfM value that may not agree, a lines record that is an eave
    ver = {SF: {"agrees": False, "how": "all-round q98"}, LN: {"how": "eave"}}
    assert not C.roof_reading([SF, LN, SK], ver)
    assert C.roof_reading([SF, SK], {SF: {"agrees": True, "how": "surface-fit"}})
    assert C.roof_reading([LN, SK], {LN: {"how": "top"}})
    assert C.roof_reading([SK, LN], {LN: {"how": "top"}, SF: {"agrees": True, "how": "roof-fit"}})
    assert not C.roof_reading([SK], {LN: {"how": "top"}, SF: {"agrees": True, "how": "roof-fit"}})


def test_the_eave_rule_needs_a_pitched_roof_and_a_usable_fit():
    """A flat roof or a dome has no eave of its own to adopt (the gate refuses a roof height on a flat roof); the SfM
    eave comes only from a fit usable as a roof reading."""
    part = dict(PART, id="w505513557")
    ch = {SK: sky(eave=37.7), SF: {"fit": dict(FIT, eave=37.44, eave_p05=37.24, eave_p95=37.64)}, LN: {"records": []}}
    pre = C.rule_h_pre(part, ch)
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, eave={SK: 37.7, SF: 37.44})
    assert set(C.eave_readings(ch, 26.0, 32.0)) == {SK, SF}
    flat = C.rule_eave(ch, dict(part, pitched=False), pre["V"], box, SCALE)
    assert not flat["adopt"] and "not a pitched roof" in flat["why"], flat
    weak = dict(ch, **{SF: {"fit": dict(FIT, eave=37.44, eave_p05=37.24, eave_p95=37.64, inlier_share=0.3)}})
    assert set(C.eave_readings(weak, 26.0, 32.0)) == {SK}


def test_tilts_from_several_reconstructions_disagree_just_outside_the_tolerance():
    a = {"fires": True, "T_en_deg": [0.0, -4.0], "tilt_deg": 4.0}
    tol = max(C.TILT_SPREAD_FLOOR, C.TILT_REL * 4.0)                 # 1.2 deg
    out = {"fires": True, "T_en_deg": [tol + 0.05, -4.0], "tilt_deg": float(np.hypot(tol + 0.05, 4.0))}
    inn = {"fires": True, "T_en_deg": [tol - 0.05, -4.0], "tilt_deg": float(np.hypot(tol - 0.05, 4.0))}
    assert not C.tilt_consensus([a, out])["agree"] and C.tilt_consensus([a, inn])["agree"]
    assert C.tilt_consensus([a, out, {"fires": False}])["results"][2] == {"reconstruction": None, "fires": False}


# ---------------------------------------------------------------- records, the publish gate, revert


def corrected_tower(map_top=20.0):
    """The synthetic tower (map 20 m) with a D25-H correction applied from the choir-like numbers scaled to it."""
    model = tj.structure(map_top)
    part = {"id": "w2", "map_top": map_top, "z0": 0.0, "roof_h": 0.0, "pitched": False,
            "min_width": model.part("w2").min_width()}
    ch = {SK: sky(top=30.0, p05=29.4, p95=30.6), SF: {"quad": quad((29.8, 29.9, 30.1, 29.5))}, LN: lines(30.2, (30.0, 30.4))}
    pre = C.rule_h_pre(part, ch)
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.1)
    fin = C.rule_h_box(pre, box, SCALE, part)
    assert fin["fires"], fin["why"]
    recs = C.apply_height(model, "w2", pre, fin, ch, box, SCALE)
    g5(model, recs)
    return model, recs


def test_a_correction_is_drawn_with_its_provenance_the_map_value_and_every_number_and_passes_the_gate():
    model, (rec,) = corrected_tower()
    h = model.part("w2").params["height"]
    assert h.state == S.STATE_CORRECTED and h.source == C.PROV_FULL and h.map_value == 20.0
    assert model.built_part("w2").top == pytest.approx(rec["value"]) and rec["osm_edited"] is False
    assert rec["kind"] == C.KIND_H and rec["registration_box"]["samples"] and rec["scale"]["route"] == "opposite-walls"
    assert sorted(rec["agreeing"]) == sorted(set(rec["agreeing"])) and len(rec["agreeing"]) >= 2
    assert C.check_record(rec) is None and C.check_model(model) == []
    assert J.check_decisions(model)["passed"] and S.check_structure(model)["passed"]
    json.dumps(model.to_registry())                     # the record is plain JSON (registry and GLB extras)
    assert model.compact_extras()["adoptions"][0]["id"] == rec["id"]


@pytest.mark.parametrize("change,why", [
    (lambda r: r.update(osm_edited=True), "osm_edited"),
    (lambda r: r.update(provenance=C.PROV_LB), "provenance"),
    (lambda r: r.update(before=[]), "before-snapshot"),
    (lambda r: r.update(agreeing=r["agreeing"][:1]), "channel"),
])
def test_an_incomplete_height_record_fails_the_publish_gate(change, why):
    """The gate checks that a record is complete: osm_edited false, the provenance of what it adopted, the snapshot a
    revert restores, >= 2 channels behind it."""
    model, (rec,) = corrected_tower()
    bad = copy.deepcopy(rec)
    change(bad)
    assert why in str(C.check_record(bad)), C.check_record(bad)
    model.adoptions[-1] = bad
    assert not J.check_decisions(model)["passed"]


def test_a_bound_is_written_with_its_own_provenance():
    """A bound short of the midpoint is written as the bound, photo-corrected (auto, lower bound); a record that says
    'lower bound' with the full value's provenance fails."""
    model = tj.structure(20.0)
    part = {"id": "w2", "map_top": 20.0, "z0": 0.0, "roof_h": 0.0, "pitched": False, "min_width": 12.0}
    ch = {SK: sky(top=30.0, p05=29.4, p95=30.6), SF: {"quad": quad((29.8, 29.9, 30.1, 29.5))}, LN: lines(30.2, (30.0, 30.4))}
    pre = C.rule_h_pre(part, ch)
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.7)
    fin = C.rule_h_box(pre, box, SCALE, part)
    assert fin["fires"] and fin["adopted"] == "lower bound"
    (rec,) = C.apply_height(model, "w2", pre, fin, ch, box, SCALE)
    g5(model, [rec])
    assert C.check_record(rec) is None and C.check_model(model) == []
    h = model.part("w2").params["height"]
    assert h.source == C.PROV_LB and h.value == pytest.approx(fin["LB"])
    assert "provenance" in C.check_record(dict(rec, provenance=C.PROV_FULL))


def test_a_corrected_value_that_does_not_match_its_record_or_has_no_record_fails():
    model, (rec,) = corrected_tower()
    model.part("w2").params["height"].value += 1.0
    assert any("does not match" in v["why"] for v in C.check_model(model))
    model, (rec,) = corrected_tower()
    model.adoptions[-1]["active"] = False
    assert any("not active" in v["why"] for v in C.check_model(model))
    m = tj.structure(20.0)
    h = m.part("w2").params["height"]
    h.value, h.source, h.state, h.map_value = 30.0, C.PROV_FULL, S.STATE_CORRECTED, 20.0
    assert any("without a correction" in v["why"] for v in C.check_model(m))
    h.source = st.PROV_S3DB
    assert any("do not match" in v["why"] for v in C.check_model(m))
    model, _ = corrected_tower()
    model.part("w2").params["height"].map_value = 14.0
    assert any("map value" in v["why"] for v in C.check_model(model))


def test_reverting_every_correction_gives_the_fresh_osm_model_back_exactly():
    fresh = tj.structure(20.0)
    model, (rec,) = corrected_tower()
    undone = J.revert_adoption(model, rec["id"])
    assert undone == [rec["id"]] and not model.adoptions[-1]["active"] and model.adoptions[-1]["reverted"]
    for p, q in zip(model.specs, fresh.specs):
        for k in p.params:
            assert (p.params[k].value, p.params[k].source, p.params[k].state) == \
                (q.params[k].value, q.params[k].source, q.params[k].state)
    tri = lambda m: np.concatenate([b.triangles() for b in m.built])
    assert np.array_equal(tri(model), tri(fresh))
    assert J.check_decisions(model)["passed"]


def capped(top=20.0):
    specs = [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "12"}),
             S.PartSpec("w2", "yes", tj.TOWER.copy(), [], {"height": f"{top:g}"}),
             S.PartSpec("w3", "yes", tj.TOWER.copy() * 0.4, [], {"height": f"{top + 4:g}", "min_height": f"{top:g}"})]
    return S.build_structure("synthetic", specs, [tj.BOX.copy()], {})


def test_children_follow_a_corrected_part_and_revert_with_it():
    model = capped()
    part = {"id": "w2", "map_top": 20.0, "z0": 0.0, "roof_h": 0.0, "pitched": False, "min_width": 12.0}
    ch = {SK: sky(top=30.0, p05=29.4, p95=30.6), SF: {"quad": quad((29.8, 29.9, 30.1, 29.5))}, LN: lines(30.2, (30.0, 30.4))}
    pre = C.rule_h_pre(part, ch)
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.1)
    fin = C.rule_h_box(pre, box, SCALE, part)
    (rec,) = C.apply_height(model, "w2", pre, fin, ch, box, SCALE)
    g5(model, [rec])
    w3 = model.part("w3")
    dz = rec["value"] - 20.0
    assert w3.z0 == pytest.approx(20.0 + dz) and w3.params["height"].source == C.PROV_FULL
    assert rec["carried"][0]["part"] == "w3"
    assert C.check_model(model) == [] and J.check_decisions(model)["passed"]
    J.revert_adoption(model, rec["id"])
    assert w3.z0 == 20.0 and w3.params["height"].source == st.PROV_S3DB


def _capped_corrected():
    model = capped()
    part = {"id": "w2", "map_top": 20.0, "z0": 0.0, "roof_h": 0.0, "pitched": False, "min_width": 12.0}
    ch = {SK: sky(top=30.0, p05=29.4, p95=30.6), SF: {"quad": quad((29.8, 29.9, 30.1, 29.5))}, LN: lines(30.2, (30.0, 30.4))}
    pre = C.rule_h_pre(part, ch)
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.1)
    fin = C.rule_h_box(pre, box, SCALE, part)
    (rec,) = C.apply_height(model, "w2", pre, fin, ch, box, SCALE)
    g5(model, [rec])
    assert C.check_model(model) == []
    return model, rec


def test_a_carry_is_listed_by_its_record_and_moves_the_value_by_its_change():
    """A carried parameter: its value is the map's plus the changes its carry entries record, each listed with that
    change by an active D25-H record."""
    model, _ = _capped_corrected()
    v = model.part("w3").params["height"]
    for e in v.evidence:
        if e.get("kind") == "carried":
            e["dz"] += 2.0
    v.value += 2.0
    assert any("lists it" in x["why"] for x in C.check_model(model))
    model, _ = _capped_corrected()
    model.part("w3").params["height"].value += 1.0
    assert any("not the map's plus the changes" in x["why"] for x in C.check_model(model))


def gabled_corrected():
    """A gabled part (map 32 m, roof 6 m) corrected to its full photo value with its eave (d25-height + d25-roof)."""
    ring = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 16.0], [0.0, 16.0]])
    model = S.build_structure("synthetic", [S.PartSpec("w5", "yes", ring, [], {"height": "32", "roof:shape": "gabled",
                                                                                "roof:height": "6"})], [ring.copy()], {})
    b = model.built_part("w5")
    part = {"id": "w5", "map_top": 32.0, "z0": 0.0, "roof_h": float(b.roof.height_m), "pitched": True,
            "min_width": b.spec.min_width()}
    ch = choir(**{SK: sky(eave=37.7), SF: {"fit": dict(FIT), "quad": quad()}, LN: {"records": []}})
    pre = C.rule_h_pre(part, ch)
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, eave={SK: 37.7, SF: 37.44})
    fin = C.rule_h_box(pre, box, SCALE, part)
    ev = C.rule_eave(ch, part, pre["V"], box, SCALE)
    assert fin["adopted"] == "measured" and ev["adopt"], (fin["why"], ev["why"])
    recs = C.apply_height(model, "w5", pre, fin, ch, box, SCALE, eave=ev)
    g5(model, recs[:1])
    assert [r["kind"] for r in recs] == [C.KIND_H, C.KIND_ROOF]
    return model, recs


def test_a_roof_height_record_rests_on_an_active_full_value_height_record():
    """The eave is adopted only beside a full-value top: a roof-height record needs its height record, active, of
    the same part and adopted 'measured'."""
    model, (h, roof) = gabled_corrected()
    act = {a["id"]: a for a in model.adoptions}
    assert C.check_record(roof, act) is None and C.check_model(model) == [] and J.check_decisions(model)["passed"]
    assert "height record" in C.check_record(roof)                                    # never checked alone
    lb = dict(copy.deepcopy(h), adopted="lower bound")
    assert "full-value" in C.check_record(roof, {h["id"]: lb, roof["id"]: roof})
    model.adoptions[-2] = lb
    assert not J.check_decisions(model)["passed"]


def test_g5_no_regression():
    """§2.4.8 part 3 item 5: the corrected version's skyline distance <= the baseline's + 0.5 px in both masks, on the
    same photos; one mask, a mask without photos, or worse by more than 0.5 px fails; a part that is the model
    skyline in no photo in either version cannot regress it. Each record carries its own part's numbers."""
    assert C.g5_problem(G5_OK) is None
    ok = copy.deepcopy(G5_OK)
    ok["masks"]["A+R"]["corrected_px"] = 12.5                         # exactly + 0.5 px: still no regression
    assert C.g5_problem(ok) is None
    for mutate, why in ((lambda g: g["masks"]["A+R"].update(corrected_px=12.51), "worse"),
                        (lambda g: g["masks"].pop("Dm+R"), "1 sky mask"),
                        (lambda g: g["masks"]["Dm+R"].update(photos=0), "no photo"),
                        (lambda g: g["masks"].clear(), "0 sky mask")):
        g = copy.deepcopy(G5_OK)
        mutate(g)
        assert why in C.g5_problem(g), (why, C.g5_problem(g))
    none = {"masks": {m: {"photos": 0, "prior_px": None, "corrected_px": None} for m in ("A+R", "Dm+R")}}
    assert C.g5_problem(none) is None
    assert C.g5_problem({"masks": {"A+R": none["masks"]["A+R"]}}) is not None
    half = copy.deepcopy(none)
    half["masks"]["A+R"] = {"photos": 3, "prior_px": 5.0, "corrected_px": 9.0}
    assert "worse" in C.g5_problem(half) or "no photo" in C.g5_problem(half)
    _, (rec,) = corrected_tower()
    assert rec["g5"]["part"] == rec["part"] and rec["g5"]["masks"] == G5_OK["masks"]


def test_the_main_outlines_symmetric_aliases_are_sampled_by_a_snap_grade_box():
    """The snap gate can accept a symmetric alias (task #73). The main outline's aliases (self-overlap >= 0.85 under a
    turn about its centroid) are registration alternatives of every snap-grade box: a correction must hold under
    them too."""
    rect = np.array([[-15.0, -10.0], [15.0, -10.0], [15.0, 10.0], [-15.0, 10.0]])
    assert C.outline_aliases(rect) == [180.0]
    sq = np.array([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]])
    assert C.outline_aliases(sq) == [90.0, 180.0, 270.0]
    octo = circle(10.0, 8)                       # nearly round: every turn overlaps >= 0.85, the yaw is not fixed
    al = C.outline_aliases(octo)
    assert {45.0 * k for k in range(1, 8)} <= set(al) and max(np.diff([0.0] + al + [360.0])) <= 30.0, al
    ell = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 8.0], [8.0, 8.0], [8.0, 25.0], [0.0, 25.0]])
    assert C.outline_aliases(ell) == []
    # an alias under which the agreeing channels read lower moves the bound (a symmetric building does not)
    pre = C.rule_h_pre(PART, choir())
    vals = {c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}
    box = make_box(vals)
    lb0 = C.rule_h_box(pre, box, SCALE, PART)["LB"]
    for x in box["samples"]:
        if x["kind"] == "snap-alternative":
            x["values"] = {c: {"value": 33.0, "half": 0.5} for c in vals}
    fin = C.rule_h_box(pre, box, SCALE, PART)
    assert fin["LB"] < lb0 and not fin["fires"], fin


def test_withdrawing_keeps_only_corrections_whose_registration_grade_still_holds():
    model, (rec,) = corrected_tower()
    assert C.records_to_keep(model, C.GRADE_SNAP) == {rec["id"]}
    assert C.records_to_keep(model, C.GRADE_CROWD) == set()      # sized for the snap: the snap is gone
    assert C.records_to_keep(model, None) == set()
    J.withdraw_photo_judgements(model, "colour gate refused", keep=C.records_to_keep(model, C.GRADE_SNAP))
    assert model.adoptions[-1]["active"]
    J.withdraw_photo_judgements(model, "snap rejected", keep=C.records_to_keep(model, C.GRADE_CROWD))
    assert not model.adoptions[-1]["active"] and model.part("w2").params["height"].value == 20.0


# ---------------------------------------------------------------- D25-B: blocks


def block_model():
    """A 3 m 'height pending' stub (an OSM part without a height) beside a 12 m hall."""
    specs = [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "12"}),
             S.PartSpec("w9", "yes", np.array([[16.0, -10.0], [30.0, -10.0], [30.0, 0.0], [16.0, 0.0]]), [], {})]
    return S.build_structure("synthetic", specs, [np.array([[-15.0, -10.0], [30.0, -10.0], [30.0, 10.0], [-15.0, 10.0]])], {})


def blk_lines(v, n_lines=16, n_photos=27):
    return {"valid": True, "value": v, "n_lines": n_lines, "n_photos": n_photos, "z_range": [v, round(v + 0.5, 3)]}


def blk_sfm(v, pts=(100, 90, 80, 70)):
    return {"valid": True, "value": v, "quadrant_q98": [v + 2.0, v + 1.0, v, v - 1.0], "quadrant_points": list(pts),
            "points": int(sum(pts))}


def blk_sky(v, n_vis_at=30, n_sky_at=20):
    m = {"top": v + 1.0, "p05": v, "p95": v + 2.0, "n_vis_at": n_vis_at, "n_sky_at": n_sky_at, "at_bound": False}
    return {"valid": True, "value": v, "top": v + 1.0, "masks": {"A+R": dict(m), "Dm+R": dict(m)}}


def blk_read(lines=16.24, sfm=15.58, sky=17.5):
    """D25-B readings with their counts (crowd3d.correct_measure.block_readings)."""
    return {LN: blk_lines(lines), SF: blk_sfm(sfm), SK: blk_sky(sky)}


READ = blk_read()


def block_box(vals=None, sens=0.05):
    return make_box(vals or {c: r["value"] for c, r in READ.items()}, sens=sens)


def test_a_block_takes_the_second_highest_lower_bound_and_stays_flat():
    part = {"id": "w9", "min_width": 10.0, "z0": 0.0, "map_top": 3.0}
    res = C.rule_b(part, READ, 22.0, block_box(), SCALE)
    assert res["fires"] and res["provenance"] == C.PROV_LB
    mins = sorted(res["box_minima"].values(), reverse=True)
    assert res["value"] == pytest.approx(mins[1]) and res["value"] < 16.24
    model = block_model()
    assert model.part("w9").params["height"].source == st.PROV_DEFAULT
    rec = C.apply_block(model, "w9", res, READ, block_box(), SCALE)
    g5(model, rec)
    p = model.part("w9")
    assert p.params["height"].source == C.PROV_LB and p.params["height"].state == S.STATE_CORRECTED
    assert p.roof_shape == "flat" and p.params["roof_height"].value == 0.0
    assert model.built_part("w9").top == pytest.approx(res["value"]) and model.built_part("w9").roof.shape == "flat"
    assert C.check_record(rec) is None and C.check_model(model) == [] and J.check_decisions(model)["passed"]
    assert S.check_structure(model)["passed"]
    # a roof shape on a block is refused by the gate
    p.roof_shape = "gabled"
    assert any("roof shape" in v["why"] for v in C.check_model(model))
    p.roof_shape = "flat"
    J.revert_adoption(model, rec["id"])
    assert p.params["height"].source == st.PROV_DEFAULT and p.params["height"].value == 3.0


def test_block_guards_each_refuse_alone():
    part = {"id": "w9", "min_width": 10.0, "z0": 0.0, "map_top": 3.0}
    assert "carving" in C.rule_b(part, READ, 12.0, block_box(), SCALE)["why"]               # carving below H
    one = {LN: READ[LN], SF: {"valid": False, "why": "fewer than 3 quadrants with >= 20 points"},
           SK: {"valid": False, "why": "too few skyline / sky photos at the value"}}
    assert "reading" in C.rule_b(part, one, 22.0, block_box(), SCALE)["why"]
    low = blk_read(4.0, 4.0, 4.0)
    assert "TOL" in C.rule_b(part, low, 22.0, block_box({c: 4.0 for c in READ}), SCALE)["why"]
    assert "slender" in C.rule_b(dict(part, min_width=3.2), READ, 22.0, block_box(), SCALE)["why"]
    assert "scale" in C.rule_b(part, READ, 22.0, block_box(), {"established": False})["why"]
    assert "crowd" in C.rule_b(part, READ, 22.0, make_box({c: r["value"] for c, r in READ.items()}, grade=C.GRADE_CROWD,
                                                          sens=0.01), SCALE)["why"]
    assert "taller adjacent" in C.rule_b(dict(part, pending=["w1"]), READ, 22.0, block_box(), SCALE)["why"]
    # the readings are judged before the box: without a box the refusal still names the missing lines
    assert "no 3-D lines" in C.rule_b(part, {SF: READ[SF], SK: READ[SK]}, 22.0, None, SCALE)["why"]
    assert C.rule_b(part, READ, 22.0, {"grade": C.GRADE_SNAP}, SCALE)["why"] == "no registration box"


def test_a_block_needs_the_3d_lines_on_every_box_sample():
    """The SfM top and the skyline read trees (or a crane) over a stub as readily as its roof, so the
    building-specific 3-D lines must be one of the readings, measured on every box sample with another reading;
    a reading lost at a registration bounds nothing there (the lines lost, or every other reading lost: refused).
    With the lines lowest, H drops to the lines' minimum."""
    part = {"id": "w9", "min_width": 10.0, "z0": 0.0, "map_top": 3.0}
    no_lines = {SF: READ[SF], SK: READ[SK]}
    r = C.rule_b(part, no_lines, 22.0, block_box({SF: 15.58, SK: 17.5}), SCALE)
    assert not r["fires"] and "no 3-D lines" in r["why"]
    vals = {c: r_["value"] for c, r_ in READ.items()}
    for lose in ({4: [LN]}, {7: [SF, SK]}):
        r = C.rule_b(part, READ, 22.0, make_box(vals, sens=0.05, lose=lose), SCALE)
        assert not r["fires"] and "lost its measurement" in r["why"] and r["samples_lost"] == list(lose), r["why"]
    r = C.rule_b(part, READ, 22.0, make_box(vals, sens=0.05, lose={7: [SF]}), SCALE)      # lines + skyline remain
    assert r["fires"], r["why"]
    lowest = blk_read(lines=12.0, sfm=15.58, sky=17.5)
    r = C.rule_b(part, lowest, 22.0, block_box({c: x["value"] for c, x in lowest.items()}), SCALE)
    assert r["fires"] and r["value"] == pytest.approx(r["box_minima"][LN]) and r["value"] < 12.0
    assert r["supported_by"] == [LN, SF, SK]


# ---------------------------------------------------------------- D25-T: tilt


def circle(r=8.0, n=48):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[r * np.cos(a), r * np.sin(a)]


TILT_CH = {"points": {"lean_en_deg": [0.35, -4.4], "bands": 40, "sigma_deg": 0.05, "sigma_boot_deg": 0.05,
                      "azimuth_coverage": 1.0},
           "silhouette": {"lean_en_deg": [0.7, -3.8], "n_photos": 33, "contrast": 0.4, "inlier_share": 0.9, "sigma_deg": 0.4},
           "lines3d": {"lean_en_deg": [-0.3, -4.9], "n_lines": 641, "n_photos": 57, "sigma_deg": 0.17}}
TILT_G = {"camera": {"lean_en_deg": [0.0, 0.0], "rolled_removed_checked": True, "roll_max_deg": 5.0},
          "scene-verticals": {"lean_en_deg": [0.57, 0.87], "inliers": 900, "photos": 30, "sigma_deg": 0.12}}


def test_the_shape_gate_admits_only_round_tall_stacks():
    assert C.tilt_shape_gate(circle(), 56.0)["passed"]
    assert not C.tilt_shape_gate(circle(), 20.0)["passed"]                    # not tall enough
    octagon = circle(8.0, 8)
    assert C.tilt_shape_gate(octagon, 56.0)["circularity"] < C.TILT_CIRC and not C.tilt_shape_gate(octagon, 56.0)["passed"]
    assert not C.tilt_shape_gate(tj.BOX, 80.0)["passed"]


def test_a_tilt_is_adopted_with_two_channels_two_gravity_references_and_the_yaw_term():
    shape = C.tilt_shape_gate(circle(), 56.0)
    r = C.rule_t(TILT_CH, TILT_G, 1.0, shape)
    assert r["fires"] and 3.5 < r["tilt_deg"] < 5.5 and 160 < r["azimuth_deg"] < 190, r
    assert not C.rule_t(TILT_CH, TILT_G, 8.0, shape)["fires"]                 # the yaw term
    assert "yaw term" in C.rule_t(TILT_CH, TILT_G, 8.0, shape)["why"]
    g1 = {"camera": TILT_G["camera"], "scene-verticals": dict(TILT_G["scene-verticals"], sigma_deg=0.65)}
    assert "gravity" in C.rule_t(TILT_CH, g1, 1.0, shape)["why"]
    one = {"points": TILT_CH["points"], "silhouette": dict(TILT_CH["silhouette"], n_photos=10)}
    assert "channel" in C.rule_t(one, TILT_G, 1.0, shape)["why"]
    noisy = {**TILT_CH, "silhouette": dict(TILT_CH["silhouette"], lean_en_deg=[3.0, -2.0])}
    assert not C.rule_t(noisy, TILT_G, 1.0, shape)["fires"]                   # the spread
    assert "shape gate" in C.rule_t(TILT_CH, TILT_G, 1.0, C.tilt_shape_gate(circle(8, 8), 56.0))["why"]
    rolled = {"camera": dict(TILT_G["camera"], roll_max_deg=40.0), "scene-verticals": TILT_G["scene-verticals"]}
    assert "gravity" in C.rule_t(TILT_CH, rolled, 1.0, shape)["why"]


def pisa_model():
    ring = circle(8.0)
    specs = [S.PartSpec("w1", "yes", ring.copy(), [], {"height": "56"})]
    return S.build_structure("pisa-synthetic", specs, [ring.copy()], {})


TILT_REG = {"grade": C.GRADE_CROWD, "reported_yaw_sigma_deg": 2.0}     # GPS yaw: sigma max(3, 1.5 x 2.0) = 3 deg


def test_the_pose_is_one_rigid_rotation_of_the_stack_and_reverts_exactly():
    model = pisa_model()
    res = C.rule_t(TILT_CH, TILT_G, 3.0, C.tilt_shape_gate(S.ccw(model.main_rings[0]), 56.0))
    assert res["fires"], res["why"]
    rec = C.apply_tilt(model, res, TILT_CH, TILT_G, registration=dict(TILT_REG),
                       consensus=C.tilt_consensus([dict(res, reconstruction="r")]))
    assert rec and model.pose["adoption"] == rec["id"] and C.check_record(rec) is None and C.check_model(model) == []
    # the snap grade's 0.5 deg is no yaw sigma for a round stack (every rotation of it is an alias): no tilt on it
    assert C.apply_tilt(pisa_model(), res, TILT_CH, TILT_G, registration={"grade": C.GRADE_SNAP}) is None
    R, c = C.pose_matrix(model.pose)
    top = R @ np.array([0, 0, 56.0])
    t, az = C.polar(top[:2] / 56.0 * 180 / math.pi)
    assert math.degrees(math.asin(np.hypot(*top[:2]) / 56.0)) == pytest.approx(res["tilt_deg"], abs=1e-3)
    assert az == pytest.approx(res["azimuth_deg"], abs=1e-2)
    X = np.random.default_rng(0).normal(size=(10, 3))
    assert np.allclose(C.apply_pose(C.apply_pose(X, model.pose), model.pose, inverse=True), X)
    assert C.apply_tilt(model, res, TILT_CH, TILT_G) is None                   # one tilt per stack
    tri0 = np.concatenate([b.triangles() for b in pisa_model().built])
    assert np.array_equal(np.concatenate([b.triangles() for b in model.built]), tri0)   # solids stay upright
    J.revert_adoption(model, rec["id"])
    assert model.pose is None and J.check_decisions(model)["passed"]


def test_a_pose_that_is_not_its_tilt_records_fails_the_gate():
    model = pisa_model()
    res = C.rule_t(TILT_CH, TILT_G, 3.0, C.tilt_shape_gate(S.ccw(model.main_rings[0]), 56.0))
    rec = C.apply_tilt(model, res, TILT_CH, TILT_G, registration=dict(TILT_REG),
                       consensus=C.tilt_consensus([dict(res, reconstruction="r")]))
    assert C.check_record(rec) is None and C.check_model(model) == []
    model.pose["tilt_deg"] = 9.0
    assert any("pose" in v["why"] for v in C.check_model(model))


def test_tilts_from_several_reconstructions_must_agree():
    a = {"fires": True, "T_en_deg": [0.35, -4.45], "tilt_deg": 4.46}
    b = {"fires": True, "T_en_deg": [-0.21, -4.1], "tilt_deg": 4.11}
    assert C.tilt_consensus([a, b])["agree"]
    c = {"fires": True, "T_en_deg": [2.0, -2.0], "tilt_deg": 2.83}
    assert not C.tilt_consensus([a, c])["agree"]


# ---------------------------------------------------------------- the report for mappers


def test_osm_fix_suggestions_are_a_report_never_an_edit():
    model, (rec,) = corrected_tower()
    rep = C.osm_fix_suggestions(model)
    assert rep["never_uploaded"] and rep["osm_edited"] is False and "No photos" in rep["note"]
    (row,) = rep["suggestions"]
    assert row["element"] == "main outline: an area without building:part (a new building:part would be needed)" \
        or row["element"].startswith("way/") or row["part"] == "w2"
    assert row["current"] == 20.0 and row["suggested"].startswith("~") and row["correction"] == rec["id"]
    m = tj.structure(20.0)
    m.specs[1].id = "w505513556"
    assert C._osm_element("w505513556") == "way/505513556" and C._osm_element("r12#1") == "relation/12"
    model = pisa_model()
    res = C.rule_t(TILT_CH, TILT_G, 1.0, C.tilt_shape_gate(S.ccw(model.main_rings[0]), 56.0))
    C.apply_tilt(model, res, TILT_CH, TILT_G)
    (row,) = C.osm_fix_suggestions(model)["suggestions"]
    assert row["tag"] is None and "note only" in row["note"]
    # every channel's numbers in the report; a stub or a gap has no height tag (3 m is a placeholder)
    model, (rec,) = corrected_tower()
    (row,) = C.osm_fix_suggestions(model)["suggestions"]
    ev = row["evidence"]
    assert ev["V"] == rec["V"] and ev["box"] == [rec["LB"], rec["UB"]] and ev["scale"]["route"] == "opposite-walls"
    assert ev["channels"][SK]["photos"]["A+R"]["skyline_at_value"] == 72 and ev["channels"][LN]["lines"] == 9
    assert ev["channels"][LN]["photos"] == 12 and ev["registration"]["grade"] == C.GRADE_SNAP
    part = {"id": "w9", "min_width": 10.0, "z0": 0.0, "map_top": 3.0}
    bm = block_model()
    g5(bm, C.apply_block(bm, "w9", C.rule_b(part, READ, 22.0, block_box(), SCALE), READ, block_box(), SCALE))
    (row,) = C.osm_fix_suggestions(bm)["suggestions"]
    assert str(row["current"]).startswith("no height tag") and row["suggested"].startswith(">=")
    assert row["evidence"]["readings"][LN]["lines"] == 16 and row["evidence"]["readings"][SF]["valid"]


# ---------------------------------------------------------------- the box terms, the tilt spread


def test_the_box_bounds_subtract_the_scale_term_and_the_ground_term():
    pre = C.rule_h_pre(PART, choir())
    vals = {c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}
    b = make_box(vals)                                       # ground measured: sigma_z 0.5 m, the term 3 x 0.5 m
    V, k_err = float(pre["V"]), float(SCALE["k_err"])
    bb = C.box_bounds(b, pre["agreeing"], V, k_err)
    lo = min(s["values"][c]["value"] - s["values"][c]["half"] for s in b["samples"] for c in pre["agreeing"])
    hi = max(s["values"][c]["value"] + s["values"][c]["half"] for s in b["samples"] for c in pre["agreeing"])
    assert bb["ground_term_m"] == pytest.approx(1.5) and bb["scale_term_m"] == pytest.approx(k_err * V, abs=1e-3)
    assert bb["LB"] == pytest.approx(lo - k_err * V - 1.5, abs=2e-3)
    assert bb["UB"] == pytest.approx(hi + k_err * V + 1.5, abs=2e-3)
    # an unmeasured ground widens both ends by 3 x 1.5 m; a larger scale error by its k_err x V
    b2 = make_box(vals, z_m=1.5)
    assert C.box_bounds(b2, pre["agreeing"], V, k_err)["LB"] == pytest.approx(bb["LB"] - 3.0, abs=2e-3)
    assert C.box_bounds(b, pre["agreeing"], V, k_err + 0.02)["LB"] == pytest.approx(bb["LB"] - 0.02 * V, abs=2e-3)
    # and the decision uses them: a bound that clears the map's TOL only without the ground term is refused
    near = {c: v for c, v in vals.items()}
    lbs = {}
    for zm in (0.5, 1.5):
        fin = C.rule_h_box(pre, make_box(near, z_m=zm), SCALE, PART)
        lbs[zm] = fin.get("LB", fin.get("box", {}).get("LB"))
    assert lbs[1.5] == pytest.approx(lbs[0.5] - 3.0, abs=2e-3)


def test_a_tilt_whose_spread_is_over_30_percent_is_refused_even_when_the_other_terms_pass():
    shape = C.tilt_shape_gate(circle(), 56.0)
    g = {"camera": dict(TILT_G["camera"], lean_en_deg=[0.0, 0.0]),
         "scene-verticals": dict(TILT_G["scene-verticals"], lean_en_deg=[0.0, 0.0])}
    ch = {"points": dict(TILT_CH["points"], lean_en_deg=[0.0, -10.0]),
          "silhouette": dict(TILT_CH["silhouette"], lean_en_deg=[6.4, -10.0])}
    r = C.rule_t(ch, g, 1.0, shape)
    t, sp = r["tilt_deg"], r["spread_deg"]
    assert t >= max(C.TILT_MIN, C.TILT_K * sp) and t >= 2 * r["U_deg"]       # the other two terms pass
    assert sp > max(C.TILT_SPREAD_FLOOR, C.TILT_REL * t)
    assert not r["fires"] and "spread" in r["why"]
    ok = {"points": ch["points"], "silhouette": dict(ch["silhouette"], lean_en_deg=[5.6, -10.0])}
    assert C.rule_t(ok, g, 1.0, shape)["fires"]


def test_a_block_reading_is_taken_at_its_box_minimum_less_the_ground_and_scale_terms():
    part = {"id": "w9", "min_width": 10.0, "z0": 0.0, "map_top": 3.0}
    box = block_box(sens=0.3)
    res = C.rule_b(part, READ, 22.0, box, SCALE)
    k_err = float(SCALE["k_err"])
    for c in READ:
        m = min(s["values"][c]["value"] for s in box["samples"])
        assert res["box_minima"][c] == pytest.approx(m - 1.5 - k_err * m, abs=2e-3)
    assert res["ground_term_m"] == pytest.approx(1.5)
    res2 = C.rule_b(part, READ, 22.0, make_box({c: r["value"] for c, r in READ.items()}, sens=0.3, z_m=1.5), SCALE)
    assert res2["value"] == pytest.approx(res["value"] - 3.0, abs=2e-3)
