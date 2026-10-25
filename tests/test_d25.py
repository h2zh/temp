"""Decision D25 (product owner, 2026-10-25): strong photo evidence corrects the OSM frame (docs/STRUCTURE_FIRST.md
§2.4.8). The rule is a function of numbers (crowd3d.correct): these tests feed it the numbers a measurement would give
(the Aachen choir's shape: map 32 m, skyline 46.7 m and 3-D lines 47.1 m agreeing, SfM points below) and mutate one
number at a time -- every guard must refuse on its own. The publish gate re-runs the rule on each record's numbers,
so every forgery of a record must fail it; reverting gives the fresh OSM model back exactly."""

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
             lose=None, eave: dict[str, float] | None = None, **sig):
    """A box whose samples move each channel by sens x (x + y) metres (+ 0.2 m per degree of yaw)."""
    sigma = C.box_sigma(grade, ground_measured=True, two_gravity_diff_deg=None, estimate_up_boot_deg=0.1,
                        **({"gps_residual_median_m": 8.0, "reported_yaw_sigma_deg": 3.1} if grade == C.GRADE_CROWD else {}))
    sigma.update(sig)
    halves = halves or {c: 0.5 for c in values}
    samples = []
    for i, s in enumerate(C.box_poses(sigma)):
        p = s["pose"]
        d = sens * (p["x"] + p["y"]) + 0.2 * p["yaw"]
        vals = {c: {"value": round(v + d, 3), "half": halves[c]} for c, v in values.items()}
        if lose and i in lose:
            for c in lose[i]:
                vals[c] = None
        s = dict(s, values=vals)
        if eave:
            s["eave"] = {c: {"value": round(v + d, 3), "half": 0.4} for c, v in eave.items()}
        samples.append(s)
    return {"grade": grade, "k": C.K_BOX, "sigma": sigma, "samples": samples, "local_ground": None}


def decide(ch=None, part=None, box=None, scale=SCALE):
    pre = C.rule_h_pre(part or PART, ch or choir())
    if not pre.get("candidate"):
        return pre, {"fires": False, "why": pre["why"]}
    box = box if box is not None else make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]})
    return pre, C.rule_h_box(pre, box, scale, part or PART)


# ---------------------------------------------------------------- D25-H: the rule


def test_strong_agreeing_channels_adopt_the_photo_value_when_the_box_bound_clears_the_midpoint():
    pre, fin = decide()
    assert pre["candidate"] and pre["direction"] == "up" and sorted(pre["agreeing"]) == [LN, SK]
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


def test_the_midpoint_rule_is_symmetric_going_down():
    assert C.adopt_value("down", 20.0, 18.0, 23.0, 30.0) == ("measured", 20.0)
    assert C.adopt_value("down", 20.0, 18.0, 26.0, 30.0) == ("upper bound", 26.0)
    assert C.adopt_value("up", 40.0, 36.0, 44.0, 32.0) == ("measured", 40.0)
    assert C.adopt_value("up", 40.0, 35.9, 44.0, 32.0) == ("lower bound", 35.9)


def test_osm_right_means_no_correction():
    """Every channel within TOL of the map (or two channels off but a third at the map): nothing happens."""
    near = choir(**{SK: sky(top=33.0, p05=32.0, p95=34.0), LN: lines(31.5, (31.3, 31.7))})
    pre, fin = decide(near)
    assert not pre["candidate"] and not fin["fires"] and "1 channel(s) off the map" in pre["why"]     # SfM alone
    # the Westwerk: skyline and 3-D lines say up, the SfM points sit at the map's value (within TOL): a contradiction
    ww = choir(**{SF: {"quad": quad((33.0, 32.5, 31.9, 30.0)), "ceiling": False}})
    pre, _ = decide(ww)
    assert not pre["candidate"] and "contradicted" in pre["why"] and SF in pre["why"]


def test_one_channel_is_never_enough():
    pre, fin = decide(choir(**{LN: {"records": []}}))
    assert not fin["fires"] and not pre["candidate"] and "no two channels agree" in pre["why"]   # SfM below: no pair


def test_skyline_guards_each_refuse_alone():
    base = C.skyline_verdict(sky(), MAP)
    assert base["valid"]
    for kw, why in (({"at_bound": True}, "bound"), ({"top_b": 49.5}, "masks' tops"), ({"n_vis_map": 2}, "map value"),
                    ({"n_vis_at": 9}, "measured value"), ({"n_sky_at": 4}, "sky above"),
                    ({"p05": 40.0, "p95": 52.0}, "PREC")):
        v = C.skyline_verdict(sky(**kw), MAP)
        assert not v["valid"] and why in v["why"], (kw, v)
    # going down the sky must be above the part at the map value
    d = C.skyline_verdict(sky(top=25.0, p05=24.0, p95=26.0, n_sky_map=4), MAP)
    assert not d["valid"] and "cannot say lower" in d["why"]
    assert C.skyline_verdict(sky(top=25.0, p05=24.0, p95=26.0, n_sky_map=5), MAP)["valid"]
    one = sky()
    one["masks"].pop("Dm+R")
    assert not C.skyline_verdict(one, MAP)["valid"]


def test_sfm_and_lines_guards():
    fit = {"top": 46.44, "top_p05": 46.36, "top_p95": 47.72, "inlier_share": 0.52, "cells": 60,
           "quadrant_cells": [10, 12, 9, 8], "at_bound": False, "eave": 37.44, "eave_p05": 37.0, "eave_p95": 38.0}
    assert C.sfm_verdict({"fit": fit}, MAP, True)["how"] == "roof-fit"
    for bad in ({"inlier_share": 0.39}, {"cells": 29}, {"quadrant_cells": [10, 12, 2, 2]}, {"at_bound": True},
                {"top_p05": 40.0, "top_p95": 49.0}):
        v = C.sfm_verdict({"fit": {**fit, **bad}}, MAP, True)
        assert not v["valid"] or v["how"] != "roof-fit", bad
    assert C.sfm_verdict({"quad": quad(pts=(120, 90, 19, 40))}, MAP, False)["value"] == 38.0
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


def test_going_down_the_points_reaching_above_the_value_veto():
    part = dict(PART, map_top=40.0, roof_h=0.0, pitched=False)
    ch = {SK: sky(top=30.0, p05=29.5, p95=30.5, n_sky_map=12), LN: lines(30.3, (30.1, 30.5)),
          SF: {"quad": quad((28.0, 27.5, 27.0, 20.0))}}
    pre, fin = decide(ch, part)
    assert pre["candidate"] and pre["direction"] == "down" and fin["fires"], (pre["why"], fin["why"])
    ch[SF] = {"quad": quad((34.0, 33.5, 33.0, 20.0))}
    pre, _ = decide(ch, part)
    assert not pre["candidate"] and "all round" in pre["why"]


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
    # route 1 (D15 similarity) and route 2 (eave ratios)
    sim = C.scale_from_similarity(J.scale_status({"similarity": {"measured": True, "k": 1.01, "at_bound": False}}), 40.0)
    assert sim["established"] and sim["k_err"] == pytest.approx(0.01 + 0.5 / 40.0, abs=1e-5)
    assert not C.scale_from_similarity(J.scale_status({}, 1.10), 40.0)["established"]
    assert C.scale_from_eave_ratios({SK: 1.01, LN: 1.02})["established"]
    assert not C.scale_from_eave_ratios({SK: 1.14, LN: 1.14})["established"]
    assert not C.scale_from_eave_ratios({LN: 1.0})["established"]
    best = C.scale_anchor([C.scale_from_walls(WALLS), sim])
    assert best["route"] in ("opposite-walls", "similarity") and best["established"]


def test_the_box_must_be_complete_and_wide_enough():
    pre = C.rule_h_pre(PART, choir())
    vals = {c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}
    good = make_box(vals)
    assert C.box_structure_problem(good) is None
    for mutate, why in ((lambda b: b["samples"].pop(1), "not sampled"),
                        (lambda b: b.update(samples=[s for s in b["samples"] if s["kind"] != "corner"][:]), "corners"),
                        (lambda b: b.update(samples=[s for s in b["samples"] if s["kind"] != "interior"]), "interior"),
                        (lambda b: b["sigma"].update(x_m=0.5), "floor"),
                        (lambda b: b["sigma"].update(yaw_deg=0.2), "floor"),
                        (lambda b: b["sigma"].update(gravity_deg=0.6), "gravity sigma"),
                        (lambda b: b["sigma"].update(z_m=0.2), "ground sigma"),
                        (lambda b: b.update(k=2.0), "box k"),
                        (lambda b: b["samples"][-1]["pose"].update(x=9.0), "outside the box"),
                        (lambda b: b["samples"].pop(0), "no base")):
        b = copy.deepcopy(good)
        mutate(b)
        why_ = C.box_structure_problem(b)
        assert why_ and why in why_, (why, why_)
        fin = C.rule_h_box(pre, b, SCALE, PART)
        assert not fin["fires"] and "registration box" in fin["why"]
    # gravity unsampled only on a local ground that passed its check and its tilt-injection test
    lg = make_box(vals, gravity_deg=None, z_m=0.5)
    lg["samples"] = [dict(s, pose={a: v for a, v in s["pose"].items() if a in ("x", "y", "yaw")}) for s in lg["samples"]]
    assert "local ground" in C.box_structure_problem(lg)
    lg["local_ground"] = {"passed": True, "injection_test_passed": False}
    assert "local ground" in C.box_structure_problem(lg)


def test_a_sample_losing_every_agreeing_channel_refuses_but_losing_one_does_not():
    pre = C.rule_h_pre(PART, choir())
    vals = {c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}
    b = make_box(vals, lose={3: [LN]})
    fin = C.rule_h_box(pre, b, SCALE, PART)
    assert fin["fires"] and fin["box"]["lost"] == [[3, LN]]
    b = make_box(vals, lose={3: [LN, SK]})
    fin = C.rule_h_box(pre, b, SCALE, PART)
    assert not fin["fires"] and "lost its measurement" in fin["why"]


def test_crowd_grade_registrations_do_not_correct_heights_yet():
    pre = C.rule_h_pre(PART, choir())
    b = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, grade=C.GRADE_CROWD, sens=0.05)
    assert C.box_structure_problem(b) is None
    fin = C.rule_h_box(pre, b, SCALE, PART)
    assert not C.CROWD_HB_ENABLED and not fin["fires"] and "crowd-grade" in fin["why"]
    s = C.box_sigma(C.GRADE_CROWD, gps_residual_median_m=12.0, reported_yaw_sigma_deg=3.19)
    assert s["x_m"] == 6.0 and s["yaw_deg"] == pytest.approx(4.785) and s["z_m"] == 1.5
    s = C.box_sigma(C.GRADE_SNAP, snap_yaw_sigma_deg=0.2, ground_measured=False, estimate_up_boot_deg=0.38)
    assert s["x_m"] == 1.0 and s["yaw_deg"] == 0.5 and s["z_m"] == 1.5 and s["gravity_deg"] == pytest.approx(1.52)
    assert C.box_sigma(C.GRADE_SNAP, two_gravity_diff_deg=0.2)["gravity_deg"] == 0.5


def test_the_eave_is_adopted_only_when_two_readings_agree_and_pass_the_box():
    fit = {"top": 46.44, "top_p05": 46.36, "top_p95": 47.72, "inlier_share": 0.52, "cells": 60,
           "quadrant_cells": [10, 12, 9, 8], "at_bound": False, "eave": 37.44, "eave_p05": 37.0, "eave_p95": 38.0}
    ch = choir(**{SK: sky(eave=37.7), SF: {"fit": fit, "quad": quad()}, LN: {"records": []}})
    part = dict(PART, id="w505513557")
    pre = C.rule_h_pre(part, ch)
    assert pre["candidate"] and sorted(pre["agreeing"]) == [SF, SK]
    vals = {c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}
    box = make_box(vals, eave={SK: 37.7, SF: 37.44})
    fin = C.rule_h_box(pre, box, SCALE, part)
    ev = C.rule_eave(ch, part, "up", pre["V"], box, SCALE)
    assert fin["adopted"] == "measured" and ev["adopt"] and ev["roof_height"] == pytest.approx(pre["V"] - ev["value"], abs=0.01)
    # one eave reading only: the map's roof height is kept
    ch1 = copy.deepcopy(ch)
    ch1[SK]["masks"]["Dm+R"]["eave"] = 40.0
    assert not C.rule_eave(ch1, part, "up", pre["V"], box, SCALE)["adopt"]
    # a sensitive eave: its bound misses the midpoint
    box2 = make_box(vals, eave={SK: 37.7, SF: 37.44})
    for s in box2["samples"]:
        s["eave"] = {c: {"value": r["value"] - 3.0 * abs(s["pose"]["x"]), "half": 0.4} for c, r in s["eave"].items()}
    assert not C.rule_eave(ch, part, "up", pre["V"], box2, SCALE)["adopt"]


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


@pytest.mark.parametrize("forge,why", [
    (lambda r: r.update(V=r["V"] + 2.0), "V"),
    (lambda r: r.update(value=r["value"] + 2.0), "adopted value"),
    (lambda r: r.update(LB=r["LB"] + 1.0), "LB / UB"),
    (lambda r: r["registration_box"]["samples"][3]["values"][r["agreeing"][0]].update(value=99.0), "LB / UB"),
    (lambda r: r["channels"][C.SKYLINE]["masks"]["A+R"].update(n_vis_at=4), "agreeing set"),
    (lambda r: r["channels"][C.SKYLINE].update(n_vis_map=1), "agreeing set"),
    (lambda r: r["scale"]["pairs"][1].update(k=0.95), "anchor"),
    (lambda r: r["scale"].update(k_err=0.0001), "k / k_err"),
    (lambda r: r.update(adopted="lower bound"), "midpoint"),
    (lambda r: r.update(provenance=C.PROV_LB), "provenance"),
    (lambda r: r["geometry"].update(min_width=4.0), "slender"),
    (lambda r: r["registration_box"]["sigma"].update(x_m=0.3), "floor"),
    (lambda r: r.update(osm_edited=True), "osm_edited"),
])
def test_every_forgery_of_a_height_record_fails_the_publish_gate(forge, why):
    model, (rec,) = corrected_tower()
    forged = copy.deepcopy(rec)
    forge(forged)
    got = C.check_record(forged)
    assert got is not None, why
    model.adoptions[-1] = forged
    assert not J.check_decisions(model)["passed"]


def test_a_bound_written_as_the_full_value_fails():
    """The midpoint rule recomputed: a record whose LB is below the midpoint but that says 'measured'."""
    model = tj.structure(20.0)
    part = {"id": "w2", "map_top": 20.0, "z0": 0.0, "roof_h": 0.0, "pitched": False, "min_width": 12.0}
    ch = {SK: sky(top=30.0, p05=29.4, p95=30.6), SF: {"quad": quad((29.8, 29.9, 30.1, 29.5))}, LN: lines(30.2, (30.0, 30.4))}
    pre = C.rule_h_pre(part, ch)
    box = make_box({c: pre["verdicts"][c]["value"] for c in pre["agreeing"]}, sens=0.7)
    fin = C.rule_h_box(pre, box, SCALE, part)
    assert fin["fires"] and fin["adopted"] == "lower bound"
    (rec,) = C.apply_height(model, "w2", pre, fin, ch, box, SCALE)
    assert C.check_record(rec) is None
    h = model.part("w2").params["height"]
    assert h.source == C.PROV_LB and h.value == pytest.approx(fin["LB"])
    forged = dict(rec, adopted="measured", value=pre["V"], provenance=C.PROV_FULL)
    assert "midpoint" in C.check_record(forged)


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
    w3 = model.part("w3")
    dz = rec["value"] - 20.0
    assert w3.z0 == pytest.approx(20.0 + dz) and w3.params["height"].source == C.PROV_FULL
    assert rec["carried"][0]["part"] == "w3"
    assert C.check_model(model) == [] and J.check_decisions(model)["passed"]
    J.revert_adoption(model, rec["id"])
    assert w3.z0 == 20.0 and w3.params["height"].source == st.PROV_S3DB


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


READ = {"lines": {"valid": True, "value": 16.24, "n_photos": 27}, "sfm-points": {"valid": True, "value": 15.58, "points": 400},
        "skyline": {"valid": True, "value": 17.5}}


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
    one = {"lines": READ["lines"], "sfm-points": dict(READ["sfm-points"], valid=False), "skyline": dict(READ["skyline"], valid=False)}
    assert "reading" in C.rule_b(part, one, 22.0, block_box(), SCALE)["why"]
    low = {c: dict(r, value=4.0) for c, r in READ.items()}
    assert "TOL" in C.rule_b(part, low, 22.0, block_box({c: 4.0 for c in READ}), SCALE)["why"]
    assert "slender" in C.rule_b(dict(part, min_width=3.2), READ, 22.0, block_box(), SCALE)["why"]
    assert "scale" in C.rule_b(part, READ, 22.0, block_box(), {"established": False})["why"]
    assert "crowd" in C.rule_b(part, READ, 22.0, make_box({c: r["value"] for c, r in READ.items()}, grade=C.GRADE_CROWD,
                                                          sens=0.01), SCALE)["why"]
    assert "taller adjacent" in C.rule_b(dict(part, pending=["w1"]), READ, 22.0, block_box(), SCALE)["why"]


def test_a_forged_block_fails_the_gate():
    part = {"id": "w9", "min_width": 10.0, "z0": 0.0, "map_top": 3.0}
    res = C.rule_b(part, READ, 22.0, block_box(), SCALE)
    model = block_model()
    rec = C.apply_block(model, "w9", res, READ, block_box(), SCALE)
    for forge in (lambda r: r.update(value=r["value"] + 3.0), lambda r: r.update(carve_p25=10.0),
                  lambda r: r.update(provenance=C.PROV_FULL), lambda r: r["geometry"].update(min_width=2.0),
                  lambda r: r["map_value"].update(source=st.PROV_S3DB)):
        f = copy.deepcopy(rec)
        forge(f)
        assert C.check_record(f) is not None


# ---------------------------------------------------------------- D25-T: tilt


def circle(r=8.0, n=48):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.c_[r * np.cos(a), r * np.sin(a)]


TILT_CH = {"points": {"lean_en_deg": [0.35, -4.4], "bands": 40, "sigma_deg": 0.05},
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


def test_the_pose_is_one_rigid_rotation_of_the_stack_and_reverts_exactly():
    model = pisa_model()
    res = C.rule_t(TILT_CH, TILT_G, 1.0, C.tilt_shape_gate(S.ccw(model.main_rings[0]), 56.0))
    rec = C.apply_tilt(model, res, TILT_CH, TILT_G, registration={"grade": C.GRADE_SNAP})
    assert rec and model.pose["adoption"] == rec["id"] and C.check_record(rec) is None and C.check_model(model) == []
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


def test_a_tilt_on_a_building_that_is_not_round_or_with_a_wrong_pose_fails_the_gate():
    model = pisa_model()
    res = C.rule_t(TILT_CH, TILT_G, 1.0, C.tilt_shape_gate(S.ccw(model.main_rings[0]), 56.0))
    rec = C.apply_tilt(model, res, TILT_CH, TILT_G)
    model.pose["tilt_deg"] = 9.0
    assert any("pose" in v["why"] for v in C.check_model(model))
    box = tj.structure(56.0)
    box.pose = dict(model.pose, tilt_deg=rec["value"]["tilt_deg"])
    box.adoptions.append(copy.deepcopy(rec))
    assert any("shape gate" in v["why"] for v in C.check_model(box))
    f = copy.deepcopy(rec)
    f["result"]["sigma_yaw_deg"] = 9.0
    assert "yaw term" in C.check_record(f)
    f = copy.deepcopy(rec)
    f["gravity"]["scene-verticals"]["inliers"] = 10
    assert "gravity" in C.check_record(f)


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
