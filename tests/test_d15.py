"""Decision D15 (product owner, 2026-10-23): a photo-measured value is adopted AUTOMATICALLY when at least two photo
channels agree (docs/STRUCTURE_FIRST.md §6 D15) -- and only then. Synthetic scenes: a known massing's silhouette
rendered into synthetic cameras as the sky masks (tests/test_skyline_judge.py), SfM points on the true roofs, 3-D line
records in crowd3d.lines3d's format (the full line pipeline is tested in tests/test_lines3d.py)."""

from __future__ import annotations

import copy
import math

import numpy as np
import pytest
import test_skyline_judge as tj      # the tower scene (pytest puts tests/ on the path)
import test_structure_stylize as tss
import test_stylize as ts

from crowd3d import judge as J
from crowd3d import lines3d as L3
from crowd3d import skyline as K
from crowd3d import structure as S
from crowd3d import stylize as st

TRUE = tj.TRUE_TOP                    # the real tower is 30 m
AUTO = st.PROV_FITTED_AUTO
# the frame gate passes and the SfM wall planes measure the whole-building similarity all round at k = 1: the
# common scale is established (judge.scale_status), so ④ does not apply unless a test says otherwise
FRAME_OK = {"passed": True, "similarity": {"measured": True, "k": 1.0, "t": [0.0, 0.0], "ticket": False,
                                           "at_bound": False}}
SCALE_OK = J.scale_status(FRAME_OK)


@pytest.fixture(scope="module")
def scene():
    cams = tj.cameras()
    return cams, tj.sky_masks(tj.structure(TRUE), cams)


def quality(P: np.ndarray) -> np.ndarray:
    """Track length 5 (each point seen in 5 photos) and a 10 deg angle per point."""
    return np.c_[np.full(len(P), 5), np.full(len(P), 10.0)]


def sfm_for(model: S.StructureModel, top: float = TRUE) -> dict:
    P = tj.tower_points(top)
    return J.sfm_channel(model, P, quality(P))


def profiles(model: S.StructureModel, cams, sky, ids=("w2",)) -> dict:
    prob = K.SkylineProblem(model, cams, sky)
    p = {pid: prob.profile_part(pid) for pid in ids}
    return {"A": p, "B": p}


def eave_rec(value: float, vote: str | None = None, n_photos: int = 12, n_lines: int = 6, at_bound: bool = False,
             parameter: str = "eave") -> dict:
    return {"kind": "lines3d", "parameter": parameter, "value": value, "interval": [value - 0.2, value + 0.2],
            "n_photos": n_photos, "n_lines": n_lines, "vote": vote, "implied_top": value, "can_vote": ["up", "down"],
            "at_bound": at_bound, "search_m": [0.0, 99.0]}


# ---------------------------------------------------------------- the rule for a part's value


def test_two_agreeing_channels_adopt_the_value_drawn_with_its_provenance_and_the_map_value_archived(scene):
    """Map 26 m, truth 30 m: the skyline's profile (two masks, interior minimum) and the SfM points (each seen in 5
    photos) both measure ~30 m, within max(1 m, 5 %): adopted -- drawn at the new value, provenance photo-fitted (auto),
    the map's 26 m kept beside it and every piece of evidence archived in the reversible list (registry and GLB
    extras); OpenStreetMap is not edited. With the switch off the same decision changes nothing."""
    cams, sky = scene
    model = tj.structure(26.0)
    d = J.decide(model, profiles(model, cams, sky), sfm_for(model), frame=FRAME_OK)["w2"]
    assert d.state == S.STATE_DISPUTED and d.direction == "up"
    assert d.adopt and set(d.adopt["channels"]) == {"skyline", "sfm-points"}, d.adopt_why
    assert d.adopt["value"] == pytest.approx(TRUE, abs=0.3) and "D15: adopt" in d.why
    m0 = tj.structure(26.0)
    J.apply_decisions(m0, {"w2": d}, frame=FRAME_OK, auto_adopt=False)
    assert m0.part("w2").params["height"].value == 26.0 and not m0.adoptions
    J.apply_decisions(model, {"w2": d}, frame=FRAME_OK)                    # D15 is on by default
    h = model.part("w2").params["height"]
    assert J.AUTO_ADOPT and h.source == AUTO and h.state == S.STATE_FITTED and h.map_value == 26.0
    assert h.value == d.adopt["value"] and model.built_part("w2").top == pytest.approx(h.value)
    ev = next(e for e in h.evidence if e["kind"] == "adoption")
    assert ev["map_value"] == 26.0 and ev["map_source"] == st.PROV_S3DB
    assert {e["kind"] for e in h.evidence} >= {"skyline", "sfm-points", "adoption"}
    (a,) = model.adoptions
    assert a["active"] and a["map_value"] == 26.0 and a["provenance"] == AUTO and not a["osm_edited"]
    assert a["evidence"] and a["label"] == "照片量出（自动采用）"
    assert J.check_decisions(model, {"w2": d})["passed"] and S.check_structure(model)["passed"]
    ex = model.compact_extras()
    assert ex["adoptions"][0]["id"] == a["id"] and ex["adoptions"][0]["evidence"]
    assert next(r for r in ex["parts"] if r["id"] == "w2")["params"]["height"][1:] == [S.STATE_FITTED, AUTO]
    assert model.to_registry()["adoptions_active"] == 1


def test_one_channel_opens_a_ticket_and_changes_nothing(scene):
    """Map 20 m, truth 30 m: the skyline's profile reaches only 26 m (its upper bound: no value) and the SfM points
    measure 30 m. Two votes make a dispute, but one channel gives a value: a ticket, nothing adopted."""
    cams, sky = scene
    model = tj.structure(20.0)
    d = J.decide(model, profiles(model, cams, sky), sfm_for(model), frame=FRAME_OK)["w2"]
    assert d.state == S.STATE_DISPUTED and not d.channels["skyline"]["ok"] and d.channels["sfm-points"]["ok"]
    assert d.adopt is None and d.adopt_why.startswith("1 channel(s) give a value"), d.adopt_why
    J.apply_decisions(model, {"w2": d}, frame=FRAME_OK)
    h = model.part("w2").params["height"]
    assert h.value == 20.0 and h.state == S.STATE_DISPUTED and not model.adoptions
    tk = next(t for t in model.tickets if t["kind"] == "part")
    assert tk["open"] and tk["d15"].startswith("1 channel(s)")
    # a 3-D line value from 2 photos, with its vote withheld, or with a 90 % interval over 1.5 m is not a second value
    prof = profiles(tj.structure(20.0), cams, sky)
    for rec, why in ((eave_rec(30.1, "up", n_photos=2), "photos"),
                     (dict(eave_rec(30.1, "up"), vote_withheld="one line only (would be up)"), "withheld"),
                     (dict(eave_rec(30.1, "up"), interval=[29.0, 31.2]), "interval")):
        m = tj.structure(20.0)
        dd = J.decide(m, prof, sfm_for(m), frame=FRAME_OK, lines={"w2": [rec]})["w2"]
        assert dd.adopt is None and why in dd.channels["lines3d"]["why"], (why, dd.channels["lines3d"])


def test_points_below_the_map_never_block_a_lower_value_and_the_skyline_needs_sky_above_to_say_lower(scene):
    """Map 36 m, truth 30 m: the skyline and the 3-D lines both say ~30 m; the SfM points over the roof reach only
    22 m (a roof nobody photographed has no points) -- points cannot say lower, so they give no value and do not
    contradict: adopted. The skyline itself gives a value below the map only with sky above the part in >= 3 photos."""
    cams, sky = scene
    model = tj.structure(36.0)
    prof = profiles(model, cams, sky)
    sfm = {"w2": {"q98": 22.0, "model_q98": 36.0, "points": 300, "min_track": 3}}
    d = J.decide(model, prof, sfm, frame=FRAME_OK, lines={"w2": [eave_rec(30.1, "down")]})["w2"]
    assert d.direction == "down" and not d.channels["sfm-points"]["ok"] and "cannot say lower" in d.channels["sfm-points"]["why"]
    assert d.adopt and set(d.adopt["channels"]) == {"skyline", "lines3d"}, d.adopt_why
    sk = J.channel_tops(model.built_part("w2"), d.params, 2, d.n_vis, 0, None, [])["skyline"]
    assert not sk["ok"] and sk["value"] < 36.0 and "cannot say lower" in sk["why"]
    # and its two masks must agree: the second mask's top 2 m higher (> 1.5 m) gives no value
    pm = copy.deepcopy(d.params)
    pm["eave"]["B"]["top_sub"] = pm["eave"]["B"]["top_sub"] + 2.0
    sk = J.channel_tops(model.built_part("w2"), pm, 2, d.n_vis, d.n_sky, None, [])["skyline"]
    assert not sk["ok"] and "masks' tops" in sk["why"], sk


def test_under_a_snap_rescale_the_channels_must_also_agree_on_a_scale_free_ratio(scene):
    """Snap scale x1.10 (> 3 %): three channels agree on ~30 m, but they share one pose set and scale. Each channel
    that measures the main body's eave (the box, 12 m) gives a scale-free ratio top / eave; the skyline and the 3-D lines
    agree on it, so the value adopted is their mean ratio x the map's 12 m eave (the SfM points measure no eave and
    cannot vouch for the scale). When the lines put the main body's eave at 13.5 m the ratios disagree (> 5 %) while the
    metric values still agree: a ticket only."""
    cams, sky = scene
    model = tj.structure(26.0)
    prof = profiles(model, cams, sky, ("w1", "w2"))
    lines = {"w2": [eave_rec(30.1, vote="up")], "w1": [eave_rec(12.0)]}
    d = J.decide(model, prof, sfm_for(model), frame=FRAME_OK, snap_scale=1.10, lines=lines)
    w2 = d["w2"]
    assert w2.direction == "up" and all(w2.channels[c]["ok"] for c in ("skyline", "sfm-points", "lines3d")), w2.channels
    assert w2.adopt, w2.adopt_why
    sf = w2.adopt["scale_free"]
    assert sf["main_body"] == "w1" and sf["map_eave"] == 12.0 and set(sf["ratios"]) == {"skyline", "lines3d"}
    assert w2.adopt["value"] == pytest.approx(np.mean(list(sf["ratios"].values())) * 12.0, abs=0.01)
    assert J.check_adoption_record(w2.adopt) is None
    lines2 = {"w2": [eave_rec(30.1, vote="up")], "w1": [eave_rec(13.5)]}
    d2 = J.decide(model, prof, sfm_for(model), frame=FRAME_OK, snap_scale=1.10, lines=lines2)["w2"]
    assert set(d2.agreeing) == {"skyline", "sfm-points", "lines3d"}               # metric agreement ...
    assert d2.adopt is None and "scale-free ratios" in d2.adopt_why, d2.adopt_why  # ... but not scale-free
    J.apply_decisions(model, {"w2": d2}, frame=FRAME_OK)
    assert model.part("w2").params["height"].value == 26.0 and not model.adoptions
    # the publish gate rechecks it: a scaled adoption without its scale-free check is rejected
    forged = dict(w2.adopt, scale_free=None)
    assert "scale-free" in J.check_adoption_record(forged)
    # map 20 m: the skyline at its bound gives no value; the points and the lines agree on 30 m, but only the lines
    # measure the main body's eave -- one scale-free ratio is not an agreement: a ticket
    m20 = tj.structure(20.0)
    d3 = J.decide(m20, profiles(m20, cams, sky, ("w1", "w2")), sfm_for(m20), frame=FRAME_OK, snap_scale=1.10,
                  lines=lines)["w2"]
    assert set(d3.agreeing) == {"sfm-points", "lines3d"} and d3.adopt is None, d3.adopt_why
    assert "fewer than two agreeing channels have a scale-free reference" in d3.adopt_why


def test_a_value_at_a_fit_bound_is_never_adopted(scene):
    """(a) map 23 m: the skyline's profile stops at its upper bound 29.9 m, 0.1 m from the points' 30 m -- it gives no
    value, so one channel remains: a ticket. (b) a 3-D line measurement at its search window's edge gives no value
    either; the same measurement off the edge, with the points, is adopted."""
    cams, sky = scene
    model = tj.structure(23.0)
    d = J.decide(model, profiles(model, cams, sky), sfm_for(model), frame=FRAME_OK)["w2"]
    sk = d.channels["skyline"]
    assert not sk["ok"] and "fit bound" in sk["why"] and d.adopt is None and "1 channel(s)" in d.adopt_why, sk
    model = tj.structure(20.0)
    prof = profiles(model, cams, sky)
    d = J.decide(model, prof, sfm_for(model), frame=FRAME_OK, lines={"w2": [eave_rec(30.1, "up", at_bound=True)]})["w2"]
    assert d.adopt is None and "search window" in d.channels["lines3d"]["why"]
    d = J.decide(model, prof, sfm_for(model), frame=FRAME_OK, lines={"w2": [eave_rec(30.1, "up")]})["w2"]
    assert d.adopt and set(d.adopt["channels"]) == {"sfm-points", "lines3d"}, d.adopt_why
    forged = copy.deepcopy(d.adopt)
    forged["channels"]["lines3d"]["at_bound"] = True
    assert "fit bound" in J.check_adoption_record(forged)


def test_contradicting_channels_open_a_ticket(scene):
    """Map 20 m: (a) the points say 30 m and the 3-D lines 34 m -- both up, 4 m apart (> max(1 m, 5 %)); (b) the lines
    measure 19 m, below the map (no vote of their own, but on the other side); (c) a channel voting the other way
    without a value (the skyline at its lower bound). Each: a ticket only."""
    cams, sky = scene
    model = tj.structure(20.0)
    prof = profiles(model, cams, sky)
    d = J.decide(model, prof, sfm_for(model), frame=FRAME_OK, lines={"w2": [eave_rec(34.0, "up")]})["w2"]
    assert d.direction == "up" and d.adopt is None and "contradicting" in d.adopt_why and "apart" in d.adopt_why
    d = J.decide(model, prof, sfm_for(model), frame=FRAME_OK, lines={"w2": [eave_rec(19.0)]})["w2"]
    assert d.direction == "up" and d.adopt is None and "not up of the map" in d.adopt_why, d.adopt_why
    J.apply_decisions(model, {"w2": d}, frame=FRAME_OK)
    assert model.part("w2").params["height"].value == 20.0 and not model.adoptions
    b = model.built_part("w2")
    ch = {"sfm-points": {"ok": True, "value": 30.0, "photos": 3}, "lines3d": {"ok": True, "value": 30.1, "photos": 12},
          "skyline": {"ok": False, "why": "at a fit bound"}}
    dd = J.Decision("w2", S.STATE_DISPUTED, "", votes={"sfm-up": 10.0, "lines-up": 10.1, "skyline-down": "lower bound hit"},
                    direction="up", channels=ch)
    rec, why = J.adoption_rule(model, b, dd, {"w2": dd}, False, 1.0, None, SCALE_OK)
    assert rec is None and "skyline-down" in why
    dd.votes.pop("skyline-down")
    rec, why = J.adoption_rule(model, b, dd, {"w2": dd}, False, 1.0, None, SCALE_OK)
    assert rec is not None and rec["value"] == pytest.approx(30.05), why


def test_an_adoption_that_cannot_be_drawn_or_a_stub_with_a_pending_roof_is_a_ticket():
    """Two channels agreeing on a top that leaves no wall under the roof; a stub whose roof shape is tagged without
    any height (its roof is pending too): a ticket only."""
    model = tj.structure(26.0)
    b = model.built_part("w2")
    low = {"sfm-points": {"ok": True, "value": 0.30, "photos": 3}, "lines3d": {"ok": True, "value": 0.35, "photos": 9}}
    dd = J.Decision("w2", S.STATE_DISPUTED, "", votes={"lines-down": -25.7, "skyline-down": -25.6}, direction="down",
                    channels=low)
    rec, why = J.adoption_rule(model, b, dd, {"w2": dd}, False, 1.0, None, SCALE_OK)
    assert rec is None and "no wall" in why
    specs = [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "12"}),
             S.PartSpec("w5", "yes", tj.TOWER.copy(), [], {"roof:shape": "gabled"})]
    m2 = S.build_structure("synthetic", specs, [tj.BOX.copy()], {})
    st5 = m2.part("w5")
    assert st5.params["height"].source == st.PROV_DEFAULT and st5.params["roof_height"].source == st.PROV_DEFAULT
    up = {"sfm-points": {"ok": True, "value": 17.0, "photos": 3}, "lines3d": {"ok": True, "value": 17.1, "photos": 9}}
    d5 = J.Decision("w5", S.STATE_STUB, "", votes={"sfm-up": 14.0, "lines-up": 14.1}, direction="up", channels=up)
    rec, why = J.adoption_rule(m2, m2.built_part("w5"), d5, {"w5": d5}, False, 1.0, None, SCALE_OK)
    assert rec is None and "roof shape is tagged without a height" in why


def test_slender_parts_and_an_open_frame_ticket_are_never_adopted(scene):
    """The two conditions D15 keeps from the design's table (§2.4.3): never a slender part, never under an open frame
    ticket -- even with two channels agreeing."""
    thin = np.array([[-2.5, -2.5], [2.5, -2.5], [2.5, 2.5], [-2.5, 2.5]])
    cams = tj.cameras()
    sky = tj.sky_masks(tj.structure(TRUE, thin), cams)
    model = tj.structure(26.0, thin)
    P = np.c_[np.random.default_rng(0).uniform(-2.2, 2.2, (80, 2)), np.full(80, TRUE)]
    sfm = J.sfm_channel(model, P, quality(P))
    d = J.decide(model, profiles(model, cams, sky), sfm, frame=FRAME_OK, lines={"w2": [eave_rec(30.1, "up")]})["w2"]
    assert d.slender and d.adopt is None and "slender" in d.adopt_why
    cams, sky = scene
    model = tj.structure(26.0)
    d = J.decide(model, profiles(model, cams, sky), sfm_for(model), frame={"passed": False})["w2"]
    assert d.adopt is None and "frame ticket" in d.adopt_why


# ---------------------------------------------------------------- parents first (⑤)


CAP = np.array([[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]])


def capped(tower_top: float, cap_top: float) -> S.StructureModel:
    specs = [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "12"}),
             S.PartSpec("w2", "yes", tj.TOWER.copy(), [], {"height": f"{tower_top:g}"}),
             S.PartSpec("w3", "yes", CAP.copy(), [], {"height": f"{cap_top:g}", "min_height": f"{tower_top:g}"})]
    return S.build_structure("synthetic", specs, [tj.BOX.copy()], {})


def capped_points() -> np.ndarray:
    """The real tower's roof (30 m) and cap (33 m), and the box's walls all round up to its 12 m eave (so the whole-
    building similarity is measured at k = 1: the common scale is established)."""
    gx, gy = np.meshgrid(np.linspace(-5.5, 5.5, 12), np.linspace(-5.5, 5.5, 12))
    ring = (np.abs(gx) > 2.6) | (np.abs(gy) > 2.6)
    tower = np.c_[gx[ring], gy[ring], np.full(ring.sum(), 30.0)]
    cx, cy = np.meshgrid(np.linspace(-1.8, 1.8, 8), np.linspace(-1.8, 1.8, 8))
    return np.vstack([tower, np.c_[cx.ravel(), cy.ravel(), np.full(cx.size, 33.0)], wall_points(tj.BOX, 11.5, 6)])


def capped_lines(model: S.StructureModel) -> dict:
    """What 3-D lines measure of the real cap (top 33 m), voting against the model's current geometry (as
    crowd3d.lines3d does when it re-assigns its lines to a new version of the structure)."""
    b = model.built_part("w3")
    tol = max(2.0, 0.1 * b.top)
    return {"w3": [dict(eave_rec(33.0), vote=L3._vote(33.0, [32.8, 33.2], b.top, tol))]}


@pytest.fixture(scope="module")
def capped_scene():
    cams = tj.cameras()
    return cams, tj.sky_masks(capped(30.0, 33.0), cams), capped_points()


def test_parents_first_the_part_on_an_adopted_part_follows_it_and_is_rejudged_on_its_new_value(capped_scene):
    """Map: tower 26 m with a 3 m cap on it; truth: tower 30 m, cap 30-33 m. At the map value the cap's own evidence
    (points and lines at 33 m) agrees too, but its parent is being adopted: parents first, it waits. judge_landmark()
    adopts the tower, carries the cap with it (+4 m, its own 3 m kept) and re-judges the cap on the tower's new value,
    where the photos agree with it: nothing more is adopted. Reverting the tower restores both."""
    cams, sky, P = capped_scene
    m0 = capped(26.0, 29.0)
    prob = K.SkylineProblem(m0, cams, sky)
    pr = {pid: prob.profile_part(pid) for pid in ("w2", "w3")}
    d0 = J.decide(m0, {"A": pr, "B": pr}, J.sfm_channel(m0, P, quality(P)), frame=FRAME_OK, lines=capped_lines(m0))
    assert d0["w2"].adopt and d0["w3"].direction == "up", (d0["w2"].adopt_why, d0["w3"].why)
    assert d0["w3"].adopt is None and d0["w3"].adopt_why.startswith("parents first"), d0["w3"].adopt_why
    model = capped(26.0, 29.0)
    ev = J.PhotoEvidence(cams, {"A": sky, "B": sky}, P, quality(P), lines=capped_lines)
    res = J.judge_landmark(model, ev)
    act = [a for a in res["adoptions"] if a["active"]]
    assert [a["part"] for a in act] == ["w2"] and res["rounds"][0]["adopted"] == [act[0]["id"]]
    assert res["rounds"][1]["judged"] == ["w3"] and res["decisions"]["w3"].rejudged == 1
    top2 = model.built_part("w2").top
    assert top2 == pytest.approx(30.0, abs=0.4)
    c = model.built_part("w3")
    assert c.z0 == pytest.approx(top2) and c.top == pytest.approx(top2 + 3.0)        # carried, its own 3 m kept
    assert c.spec.params["height"].source == AUTO and c.spec.params["height"].map_value == 29.0
    assert not res["decisions"]["w3"].direction and res["decisions"]["w3"].adopt is None
    assert not any(t.get("open") and t.get("part") == "w3" for t in model.tickets)
    assert J.check_decisions(model, res["decisions"])["passed"], J.check_decisions(model, res["decisions"])
    assert S.check_structure(model)["passed"]
    J.revert_adoption(model, act[0]["id"])
    assert model.built_part("w2").top == pytest.approx(26.0) and model.built_part("w3").z0 == pytest.approx(26.0)
    assert J.check_decisions(model)["passed"]


def test_a_child_is_never_adopted_in_the_same_pass_as_its_parent(capped_scene):
    """adopt_decisions() itself: given adoption records for both a parent and the part on it, only the parent is
    adopted; the child is carried and returned to be re-judged on the parent's new value (its record was judged at
    the parent's old value)."""
    model = capped(26.0, 29.0)
    rec = tj._adopt_record
    dec = {"w2": J.Decision("w2", S.STATE_DISPUTED, "", direction="up", adopt=rec(26.0, 30.0)),
           "w3": J.Decision("w3", S.STATE_DISPUTED, "", direction="up", adopt=rec(29.0, 33.0))}
    pending = J.adopt_decisions(model, dec)
    assert pending == ["w3"] and [a["part"] for a in model.adoptions] == ["w2"]
    assert model.built_part("w3").top == pytest.approx(33.0)          # carried by +4 m, not adopted on its own
    assert not any(e["kind"] == "adoption" for e in model.part("w3").params["height"].evidence)


# ---------------------------------------------------------------- the frame (⑥)


OCT = np.array([[17.0 * math.cos(math.radians(22.5 + 45 * k)), 17.0 * math.sin(math.radians(22.5 + 45 * k))]
                for k in range(8)])


def oct_model() -> S.StructureModel:
    return S.build_structure("oct", [S.PartSpec("w9", "yes", OCT.copy(), [], {"height": "18"})], [OCT.copy()], {})


def wall_points(real: np.ndarray, z_hi: float = 17.5, n_z: int = 8) -> np.ndarray:
    """SfM points on every wall of the real outline, from 3.5 m up to z_hi (the walls are reconstructed up to their
    top, as the 3-D lines on them are: lines3d keeps only lines SfM points support)."""
    pts = []
    for i in range(len(real)):
        a, b = real[i], real[(i + 1) % len(real)]
        s = np.linspace(0.05, 0.95, 60)
        for z in np.linspace(3.5, z_hi, n_z):
            pts.append(np.c_[a + s[:, None] * (b - a), np.full(len(s), z)])
    return np.vstack(pts)


def wall_lines(real: np.ndarray, heights=(4.0, 8.0, 12.0, 15.0, 18.0)) -> L3.Lines3D:
    """Horizontal 3-D lines on every wall of the real outline (string courses, the eave), each from 8 photos."""
    A, B = [], []
    for i in range(len(real)):
        a, b = real[i], real[(i + 1) % len(real)]
        for z in heights:
            A.append([*(a + 0.2 * (b - a)), z])
            B.append([*(a + 0.8 * (b - a)), z])
    n = len(A)
    return L3.Lines3D(np.array(A), np.array(B), np.full(n, 8), [list(range(8))] * n, np.ones(n), np.zeros(n))


def eave_refs(skyline: float | None = None, lines: float | None = None):
    """An eave_check for decide_frame(): the main body's eave as the skyline and the 3-D lines measure it on the
    trial outline (judge_landmark() measures it from the photos; here the values are given)."""
    def check(trial: S.StructureModel) -> dict:
        mb = S.main_body(trial)
        ch = {name: ({"ok": True, "value": v} if v is not None else {"ok": False, "why": "not measured"})
              for name, v in (("skyline", skyline), ("lines3d", lines))}
        return {"main_body": mb.id, "map_eave": round(float(mb.eave), 2), "channels": ch}
    return check


def test_the_frame_is_adopted_by_the_same_rule_and_reverts():
    """The real octagon is 0.82 x the map's, shifted (3, -1) m (the Baptistery's case). The gate fails; the SfM wall
    planes and the 3-D wall lines each measure the similarity (walls all round) and agree on the width and the shift:
    adopted -- every outline photo-fitted (auto), the frame ticket closed by the adoption, the gate passing on the
    adopted outline. One channel, two channels disagreeing, or a snap scale correction > 3 %: a ticket only."""
    real = OCT * 0.82 + np.array([3.0, -1.0])
    P = wall_points(real)
    q = quality(P)
    model = oct_model()
    gate = J.frame_gate(P, model.main_rings, q)
    ch = J.frame_channels(model, P, q, wall_lines(real))
    assert not gate["passed"] and ch["sfm-points"]["measured"] and ch["lines3d"]["measured"]
    fd = J.decide_frame(model, gate, ch, 1.0, P, q, eave_check=eave_refs(18.03, 18.0))
    assert fd["in_question"] and fd["adopt"], fd["why"]
    assert fd["adopt"]["scale_check"]["passed"] and J.check_frame_record(fd["adopt"]) is None
    assert fd["adopt"]["k"] == pytest.approx(0.82, abs=0.01) and np.allclose(fd["adopt"]["t"], [3.0, -1.0], atol=0.15)
    a = J.adopt_frame(model, fd, gate)
    assert a and all(p.outline_source == AUTO for p in model.specs)
    assert np.allclose(model.main_rings[0], real, atol=0.15) and J.frame_gate(P, model.main_rings, q)["passed"]
    tk = next(t for t in model.tickets if t["kind"] == "frame")
    assert not tk["open"] and tk["resolved_by"] == a["id"]
    assert J.check_decisions(model)["passed"] and model.frame["adopted"]["id"] == a["id"]
    assert J.revert_adoption(model, a["id"]) == [a["id"]]
    assert np.allclose(model.main_rings[0], OCT, atol=1e-6) and all(p.outline_source == st.PROV_S3DB for p in model.specs)
    assert next(t for t in model.tickets if t["kind"] == "frame")["open"]
    # one channel only
    m1 = oct_model()
    fd1 = J.decide_frame(m1, gate, J.frame_channels(m1, P, q, None), 1.0, P, q)
    assert fd1["adopt"] is None and fd1["why"].startswith("1 channel(s)")
    # the lines on an outline 0.90 x the map's: 2.7 m wider than the planes' (> max(1 m, 5 %)): contradicting
    fd2 = J.decide_frame(m1, gate, J.frame_channels(m1, P, q, wall_lines(OCT * 0.90 + np.array([3.0, -1.0]))), 1.0, P, q)
    assert fd2["adopt"] is None and "contradicting" in fd2["why"]
    # the lines shifted the other way: contradicting
    fd3 = J.decide_frame(m1, gate, J.frame_channels(m1, P, q, wall_lines(OCT * 0.82 + np.array([-3.0, 1.0]))), 1.0, P, q)
    assert fd3["adopt"] is None and "contradicting" in fd3["why"]
    # under a snap scale correction > 3 %
    fd4 = J.decide_frame(m1, gate, ch, 1.10, P, q)
    assert fd4["adopt"] is None and "scale-free" in fd4["why"]
    # forged: a frame record from one channel, or from channels 3 m apart, is rejected by adopt_frame and the gate
    forged = copy.deepcopy(fd)
    forged["adopt"]["channels"].pop("lines3d")
    assert J.adopt_frame(oct_model(), forged) is None and "1 channel" in J.check_frame_record(forged["adopt"])
    forged = copy.deepcopy(fd)
    forged["adopt"]["channels"]["lines3d"]["t"] = [6.3, -1.1]
    assert J.adopt_frame(oct_model(), forged) is None and "shift" in J.check_frame_record(forged["adopt"])
    # a channel at its fit bound gives no value
    ch_b = copy.deepcopy(ch)
    ch_b["lines3d"].update(at_bound=True, bound_why="shift 9.10 m > 8.5 m")
    fd5 = J.decide_frame(m1, gate, ch_b, 1.0, P, q)
    assert fd5["adopt"] is None and "fit bound" in fd5["channels"]["lines3d"]["why"]


def test_frame_channels_on_both_sides_of_the_maps_outline_scale_contradict():
    """A 10 m outline: one channel measures it 7 % larger, the other 0.5 % smaller -- 0.75 m apart in width (within
    max(1 m, 5 %)), their mean 3.25 % off the map, but on both sides of the map's scale: a contradiction, a ticket."""
    sq = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    model = S.build_structure("sq", [S.PartSpec("w7", "yes", sq.copy(), [], {"height": "10"})], [sq.copy()], {})

    def ch(k):
        return {"measured": True, "k": k, "t": [0.1, 0.0], "photos_min": 3, "centre": [0.0, 0.0], "edges": [{}] * 4,
                "at_bound": False, "ticket": True}
    fd = J.decide_frame(model, {"passed": False}, {"sfm-points": ch(1.07), "lines3d": ch(0.995)}, 1.0)
    assert fd["adopt"] is None and "both sides" in fd["why"], fd["why"]
    fd = J.decide_frame(model, {"passed": False}, {"sfm-points": ch(1.07), "lines3d": ch(1.04)}, 1.0,
                        eave_check=eave_refs(10.1, 9.9))
    assert fd["adopt"] and fd["adopt"]["k"] == pytest.approx(1.055)


def test_a_frame_whose_residuals_remain_on_the_adopted_outline_is_not_adopted():
    """One wall of the real octagon stands 5 m further in than a similarity allows: both channels' least squares
    still agree on k and t (widths 0.6 m, shifts 0.5 m apart), but on the adopted outline that wall still fails the
    gate -- not a whole-building similarity: a ticket only."""
    real = OCT * 0.82 + np.array([3.0, -1.0])
    bent = real.copy()
    n = np.array([math.cos(math.radians(45.0)), math.sin(math.radians(45.0))])
    bent[0] -= 5.0 * n
    bent[1] -= 5.0 * n
    P = wall_points(bent)
    q = quality(P)
    model = oct_model()
    gate = J.frame_gate(P, model.main_rings, q)
    ch = J.frame_channels(model, P, q, wall_lines(bent))
    fd = J.decide_frame(model, gate, ch, 1.0, P, q)
    assert fd["adopt"] is None and "still fails on the adopted outline" in fd["why"], fd["why"]


def test_reverting_a_frame_adoption_reverts_the_adoptions_made_on_it():
    """Every part adoption made on an adopted frame depends on it: reverting the frame reverts them too (newest
    first), and the part's map value and the map's outline both come back."""
    real = OCT * 0.82 + np.array([3.0, -1.0])
    P = wall_points(real)
    q = quality(P)
    model = oct_model()
    gate = J.frame_gate(P, model.main_rings, q)
    fa = J.adopt_frame(model, J.decide_frame(model, gate, J.frame_channels(model, P, q, wall_lines(real)), 1.0, P, q,
                                             eave_check=eave_refs(18.0, 18.0)), gate)
    d = {"w9": J.Decision("w9", S.STATE_DISPUTED, "", direction="up", adopt=tj._adopt_record(18.0, 21.0))}
    J.apply_decisions(model, d, frame={"passed": True})
    pa = model.adoptions[-1]
    assert pa["kind"] == "part-height" and pa["depends_on"] == [fa["id"]] and model.part("w9").top == 21.0
    assert J.check_decisions(model)["passed"]
    assert J.revert_adoption(model, fa["id"]) == [pa["id"], fa["id"]]
    assert model.part("w9").top == 18.0 and np.allclose(model.main_rings[0], OCT, atol=1e-6)
    assert not any(a["active"] for a in model.adoptions) and J.check_decisions(model)["passed"]


def test_judge_landmark_adopts_the_frame_first_and_judges_the_parts_on_it():
    """The whole judge on the octagon: the frame adopted first (⑥), then the part judged on the adopted outline with
    the skyline and the 3-D lines re-assigned to it (the real eave 18 m: no dispute); the published gate passes."""
    real = OCT * 0.82 + np.array([3.0, -1.0])
    truth = S.build_structure("oct", [S.PartSpec("w9", "yes", real.copy(), [], {"height": "18"})], [real.copy()], {})
    cams = tj.cameras(radius=60.0)
    sky = tj.sky_masks(truth, cams)
    P = wall_points(real)
    model = oct_model()
    res = J.judge_landmark(model, J.PhotoEvidence(cams, {"A": sky, "B": sky}, P, quality(P), lines=wall_lines(real)))
    assert res["frame_adoption"] and res["frame"]["passed"] and res["frame"]["adopted"] == res["frame_adoption"]["id"]
    assert res["frame"]["before_adoption"]["passed"] is False
    d = res["decisions"]["w9"]
    assert not d.direction and d.adopt is None
    assert d.state == S.STATE_VERIFIED, d.why             # skyline 18.0 + the eave lines 18.0 m, on the adopted frame
    assert J.check_decisions(model, res["decisions"])["passed"], J.check_decisions(model, res["decisions"])
    assert not any(t.get("open") and t["kind"] == "frame" for t in model.tickets)
    # reverting the frame withdraws what was judged on it: the frame ticket reopens, the verification goes
    J.revert_adoption(model, res["frame_adoption"]["id"])
    assert model.part("w9").params["height"].state == S.STATE_MAP and any(t.get("open") and t["kind"] == "frame"
                                                                           for t in model.tickets)
    assert J.check_decisions(model)["passed"], J.check_decisions(model)


# ---------------------------------------------------------------- stubs, appearance, honesty


def photographed_stub_scene() -> st.SceneInput:
    """The photographed 10 m box (its south facade and roof seen); OSM maps only its east half, so the west half is
    a 3 m 'height pending' stub (tests/test_structure_stylize.stub_scene without its low points, which fail the frame
    gate there); a photo from a neighbouring tower looks down on it."""
    scene = ts.south_scene()
    east = np.array([[0.0, -6.0], [10.0, -6.0], [10.0, 6.0], [0.0, 6.0]])
    scene.structure = S.build_structure("Test Box", [S.PartSpec("w300", "yes", east, [], {"height": "10"})],
                                        [ts.BOX.copy()], {})
    tower = ts.look_at([-5.0, -24.0, 26.0], [-5.0, -1.0, 3.0], name="tower")
    tower.image = ts.render_south(tower)
    scene.cameras.append(tower)
    return scene


def judged_stub(scene: st.SceneInput, lines: dict | None):
    model = scene.structure
    stub = next(p for p in model.specs if p.kind == "remainder")
    q = quality(scene.points)
    frame = J.frame_gate(scene.points, model.main_rings, q)
    assert frame["passed"]
    # the photos here see the south facade only, so the whole-building similarity is not measured from them; the
    # scale is taken as established (measured all round) -- the tests of ④ above cover the other case
    frame = {**frame, "similarity": FRAME_OK["similarity"]}
    prof = {stub.id: K.SkylineProblem(model, scene.cameras, tss.ideal_sky(scene)).profile_part(stub.id)}
    d = J.decide(model, {"A": prof, "B": prof}, J.sfm_channel(model, scene.points, q), frame=frame, lines=lines)
    return model, stub, frame, d[stub.id]


def test_a_stub_gets_a_photo_measured_height_but_appearance_only_where_observed():
    """A 3 m 'height pending' stub over the west half of the photographed 10 m box: the SfM points (the roof at 10 m,
    each seen in 5 photos) and a 3-D line measurement of its wall top (10 m) agree; the skyline's profile of a 3 m stub
    stops at 3.9 m (its bound: no value). D15 lets a stub get a photo-measured height this way: 10 m, photo-fitted
    (auto), the 3 m default kept as the old value. Its appearance is still only where photos observed it: the
    photographed south face is coloured, the others stay clay; every publish check passes."""
    scene = photographed_stub_scene()
    stub0 = next(p for p in scene.structure.specs if p.kind == "remainder")
    lines = {stub0.id: [dict(eave_rec(10.0, "up", n_photos=6, n_lines=5), interval=[9.85, 10.1])]}
    model, stub, frame, d = judged_stub(scene, lines)
    assert d.state == S.STATE_STUB and d.direction == "up" and not d.channels["skyline"]["ok"]
    assert d.adopt and set(d.adopt["channels"]) == {"sfm-points", "lines3d"}, d.adopt_why
    J.apply_decisions(model, {stub.id: d}, frame=frame)
    h = stub.params["height"]
    assert h.source == AUTO and h.state == S.STATE_FITTED and h.map_value == 3.0
    assert h.value == pytest.approx(10.0, abs=0.2)
    assert next(e for e in h.evidence if e["kind"] == "adoption")["map_source"] == st.PROV_DEFAULT
    assert not st.is_stub_part(stub) and stub.state() == S.STATE_FITTED
    assert J.check_decisions(model)["passed"] and S.check_structure(model)["passed"]
    m = st.stylize_landmark(scene)
    qa = m.summary["qa"]
    assert qa["passed"] and qa["structure"]["passed"], qa
    mine = [p for p in m.parts if p.extras.get("part") == stub.id]
    walls = [p for p in mine if p.extras["kind"] == "wall"]
    for p in mine:
        if p.extras.get("observed"):
            assert p.extras["provenance"]["appearance"] == st.PROV_PHOTO
        else:
            assert p.extras["provenance"]["appearance"] == st.PROV_DEFAULT and p.texture is None
    assert any(p.extras.get("observed") for p in walls), [p.extras.get("wall") for p in walls]
    assert any(not p.extras.get("observed") for p in walls)
    assert all(p.extras["provenance"]["height"] == AUTO and p.extras["adopted"][0]["map"] == 3.0 for p in walls)
    pr = next(p for p in m.massing.prisms if p.part == stub.id)
    assert not pr.stub and pr.z1 == pytest.approx(h.value)
    gx = st.read_glb(m.glb)[0]["scenes"][0]["extras"]["structure"]
    assert gx["adoptions"][0]["part"] == stub.id and gx["adoptions"][0]["map_value"] == 3.0
    assert m.summary["structure"]["adoptions"][0]["map_source"] == st.PROV_DEFAULT


def test_without_two_channels_a_stub_stays_a_stub():
    """The same stub with the SfM points alone (no line measurement): one channel, a ticket; the stub stays a 3 m
    clay 'height pending' stub."""
    model, stub, frame, d = judged_stub(photographed_stub_scene(), None)
    assert d.state == S.STATE_STUB and d.adopt is None and d.adopt_why.startswith("1 channel(s)"), d.adopt_why
    J.apply_decisions(model, {stub.id: d}, frame=frame)
    assert stub.params["height"].source == st.PROV_DEFAULT and stub.params["height"].state == S.STATE_STUB
    assert st.is_stub_part(stub) and J.check_decisions(model)["passed"]


def test_the_publish_gate_rechecks_every_adoption_from_its_records(scene):
    """G4 / G9 never trust the code that wrote an adoption: forged records -- one channel, channels 4 m apart, a value
    that is not their mean, a channel from 2 photos, a channel on the wrong side, a skyline from one mask, an adoption
    no longer active in the reversible list, a parameter that does not match its adoption, a photo-fitted (auto) value
    or outline without any record -- are all caught."""
    cams, sky = scene
    model = tj.structure(26.0)
    d = J.decide(model, profiles(model, cams, sky), sfm_for(model), frame=FRAME_OK)["w2"]
    J.apply_decisions(model, {"w2": d}, frame=FRAME_OK)
    assert J.check_decisions(model)["passed"]
    h = model.part("w2").params["height"]
    e = next(x for x in h.evidence if x["kind"] == "adoption")
    saved = copy.deepcopy(e)

    def whys():
        return " | ".join(v["why"] for v in J.check_decisions(model)["violations"])

    def reset():
        e.clear()
        e.update(copy.deepcopy(saved))

    e["channels"].pop("skyline")
    assert "1 channel(s)" in whys()
    reset()
    e["channels"]["skyline"]["value"] = e["channels"]["sfm-points"]["value"] + 4.0
    assert "disagree" in whys()
    reset()
    e["value"] = round(e["value"] + 2.0, 2)
    assert "not the channels' mean" in whys()
    reset()
    e["channels"]["sfm-points"]["photos"] = 2
    assert "< 3 photos" in whys()
    reset()
    e["channels"]["skyline"]["value"] = 25.0
    assert "not up of" in whys()
    reset()
    e["channels"]["skyline"]["masks"] = 1
    assert "two masks" in whys()
    reset()
    # an 'up' adoption whose own record shows the SfM points ending at the map's top all round (a contradiction)
    e["all_channels"]["sfm-points"]["ceiling"] = True
    assert "ending at the map's top" in whys()
    reset()
    assert J.check_decisions(model)["passed"]
    model.adoptions[0]["active"] = False
    assert "not active" in whys()
    model.adoptions[0]["active"] = True
    h.value = 31.5
    assert "differs from its adoption record" in whys() and "does not match" in whys()
    h.value = saved["value"]
    # apply_decisions itself never applies a record that fails the rule (one channel)
    m1 = tj.structure(26.0)
    bad = J.Decision("w2", S.STATE_DISPUTED, "", direction="up", adopt=dict(saved, channels={"sfm-points": saved["channels"]["sfm-points"]}))
    J.apply_decisions(m1, {"w2": bad}, frame=FRAME_OK)
    assert not m1.adoptions and m1.part("w2").params["height"].value == 26.0
    m2 = tj.structure(26.0)
    m2.part("w2").params["height"].source, m2.part("w2").params["height"].state = AUTO, S.STATE_FITTED
    m2.part("w2").params["height"].map_value = 26.0
    assert any("without an adoption record" in v["why"] for v in J.check_decisions(m2)["violations"])
    m3 = tj.structure(26.0)
    m3.specs[0].outline_source = AUTO
    assert any("outline" in v["why"] for v in J.check_decisions(m3)["violations"])


# ---------------------------------------------------------------- round 2: the frame in question, the common scale


def test_a_frame_ticket_from_the_similarity_blocks_adoption_and_the_published_model_still_passes(scene):
    """The gate passes (walls under 1.5 m off), but the SfM wall planes measure the outline at k 0.90 (a frame
    ticket): two channels agreeing on the part's top are a ticket only -- the frame is open whenever the gate fails
    OR a channel's similarity is in question, in decide() and in apply_decisions() alike -- so the publish gate (G4)
    still passes. A verification is withheld the same way. The 3-D lines' similarity in question counts too."""
    cams, sky = scene
    sim = {"measured": True, "k": 0.90, "t": [0.0, 0.0], "ticket": True, "at_bound": False}
    for frame in ({"passed": True, "similarity": sim}, {"passed": True, "similarity_lines": sim}):
        assert J.frame_in_question(frame) and not J.scale_status(frame)["established"]
        model = tj.structure(26.0)
        d = J.decide(model, profiles(model, cams, sky), sfm_for(model), frame=frame)["w2"]
        assert d.direction == "up" and len(d.agreeing) >= 2 and d.adopt is None and "frame ticket" in d.adopt_why
        J.apply_decisions(model, {"w2": d}, frame=frame)
        assert model.part("w2").params["height"].value == 26.0 and not model.adoptions
        assert any(t["kind"] == "frame" and t["open"] for t in model.tickets)
        assert J.check_decisions(model, {"w2": d})["passed"], J.check_decisions(model, {"w2": d})
    model = tj.structure(TRUE)
    d = J.decide(model, profiles(model, cams, sky), frame={"passed": True, "similarity": sim},
                 lines={"w2": [eave_rec(30.1)]})["w2"]
    assert d.state == S.STATE_MAP and "frame ticket open" in d.why and "photo-verified" in d.why
    J.apply_decisions(model, {"w2": d}, frame={"passed": True, "similarity": sim})
    assert J.check_decisions(model, {"w2": d})["passed"]


def test_scale_status_needs_a_similarity_measured_all_round_within_three_percent():
    """④ applies unless the common scale is established: a snap returning x1.00 says nothing on its own."""
    ok = {"measured": True, "k": 1.02, "t": [0.4, 0.0], "ticket": False, "at_bound": False}
    assert J.scale_status({"passed": True, "similarity": ok})["established"]
    assert J.scale_status({"passed": True, "similarity_lines": ok})["by"]["channel"] == "lines3d"
    assert not J.scale_status({"passed": True, "similarity": ok}, snap_scale=1.10)["established"]
    for bad in ({**ok, "measured": False}, {**ok, "k": 0.95}, {**ok, "at_bound": True}):
        s = J.scale_status({"passed": True, "similarity": bad})
        assert not s["established"] and "not established" in s["why"], s
    assert not J.scale_status({"passed": True})["established"] and not J.scale_status(None)["established"]


def lone_tower(top: float) -> S.StructureModel:
    return S.build_structure("lone", [S.PartSpec("w2", "yes", tj.TOWER.copy(), [], {"height": f"{top:g}"})],
                             [tj.TOWER.copy()], {})


def test_without_an_established_scale_the_main_body_is_not_adopted_and_a_part_needs_the_scale_free_check(scene):
    """Photos from one side: the gate passes but no similarity is measured, so a common scale error (a snap that
    returned x1.00) could move every channel together. (a) A lone 26 m tower that skyline + points + lines all put at
    30 m is the main body itself, the scale-free reference: a ticket, not an adoption. (b) The tower on the 12 m box:
    the skyline and the lines also measure the box's eave, their ratios top / eave agree, and the value adopted is
    the scale-free one; the record carries the scale status and the publish gate rechecks it."""
    cams, sky = scene
    one_sided = {"passed": True, "similarity": {"measured": False, "k": 0.97, "t": [0.0, 0.0], "ticket": False,
                                                "reason": "walls do not face >= 180 deg"}}
    lone = lone_tower(26.0)
    sky_lone = tj.sky_masks(lone_tower(TRUE), cams)
    prob = K.SkylineProblem(lone, cams, sky_lone)
    pr = {"w2": prob.profile_part("w2")}
    d = J.decide(lone, {"A": pr, "B": pr}, sfm_for(lone), frame=one_sided, lines={"w2": [eave_rec(30.1, "up")]})["w2"]
    assert d.direction == "up" and len(d.agreeing) >= 2 and d.adopt is None, d.adopt_why
    assert "not established" in d.adopt_why and "scale-free reference itself" in d.adopt_why, d.adopt_why
    with_refs = J.decide(lone, {"A": pr, "B": pr}, sfm_for(lone), frame=FRAME_OK, lines={"w2": [eave_rec(30.1, "up")]})["w2"]
    assert with_refs.adopt, with_refs.adopt_why                              # the same evidence, the scale established
    model = tj.structure(26.0)
    lines = {"w2": [eave_rec(30.1, vote="up")], "w1": [eave_rec(12.0)]}
    d = J.decide(model, profiles(model, cams, sky, ("w1", "w2")), sfm_for(model), frame=one_sided, lines=lines)["w2"]
    assert d.adopt and d.adopt["scale_free"] and set(d.adopt["scale_free"]["ratios"]) == {"skyline", "lines3d"}, d.adopt_why
    assert not d.adopt["scale"]["established"] and J.check_adoption_record(d.adopt) is None
    forged = copy.deepcopy(d.adopt)
    forged["scale_free"] = None
    assert "not established" in J.check_adoption_record(forged)
    forged["scale"] = {"established": True, "snap_scale": 1.0, "by": {"channel": "sfm-points", "measured": True, "k": 0.9}}
    assert "not established" in J.check_adoption_record(forged)          # a 'by' that does not establish it


def test_a_frame_off_by_more_than_three_percent_needs_the_eave_to_confirm_the_scale():
    """The octagon's walls measured all round at 0.82 x the map's outline (both channels agreeing). That is an OSM
    outline error only if the reconstruction's scale is the map's: the main body's eave measured on the adopted
    outline (skyline + 3-D lines) must agree with the map's 18 m within max(1 m, 5 %), and outline width / eave height
    must still change by > 3 %. A mis-scaled reconstruction shows its heights scaled too (eave 0.82 x 18 m): a
    ticket; so are one channel, two disagreeing, no eave measured, and a change that is not scale-free."""
    real = OCT * 0.82 + np.array([3.0, -1.0])
    P = wall_points(real)
    q = quality(P)
    model = oct_model()
    gate = J.frame_gate(P, model.main_rings, q)
    ch = J.frame_channels(model, P, q, wall_lines(real))
    for check, why in ((None, "not measured"), (eave_refs(14.76, 14.8), "reconstruction's scale may be what is off"),
                       (eave_refs(18.0, None), "1 channel(s)"), (eave_refs(18.0, 19.6), "disagree")):
        fd = J.decide_frame(model, gate, ch, 1.0, P, q, eave_check=check)
        assert fd["adopt"] is None and why in fd["why"], (why, fd["why"])
    fd = J.decide_frame(model, gate, ch, 1.0, P, q, eave_check=eave_refs(18.6, 18.8))     # the Baptistery's numbers
    assert fd["adopt"] and fd["adopt"]["scale_check"]["passed"], fd["why"]
    forged = copy.deepcopy(fd["adopt"])
    forged.pop("scale_check")
    assert "eave" in J.check_frame_record(forged) and J.adopt_frame(oct_model(), {"adopt": forged}) is None
    forged["scale_check"] = {"map_eave": 18.0, "eaves": {"skyline": 14.76, "lines3d": 14.8}}
    assert "does not confirm" in J.check_frame_record(forged)
    # k 0.95 with the eaves 4 % low: outline width / eave height changes by ~1 %: not scale-free
    sq = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    m2 = S.build_structure("sq", [S.PartSpec("w7", "yes", sq.copy(), [], {"height": "18"})], [sq.copy()], {})
    c95 = {"measured": True, "k": 0.95, "t": [0.0, 0.0], "photos_min": 3, "centre": [0.0, 0.0], "edges": [{}] * 4,
           "at_bound": False, "ticket": True}
    fd = J.decide_frame(m2, {"passed": False}, {"sfm-points": c95, "lines3d": dict(c95)}, 1.0,
                        eave_check=eave_refs(17.3, 17.35))
    assert fd["adopt"] is None and "not scale-free" in fd["why"], fd["why"]


def tower_scene(top: float, arc: float | None, scale: float):
    """A correctly mapped 45 m tower (12 x 12 m), rendered (crowd3d.lines3d's synthetic photos), its SfM points, 3-D
    lines and ideal sky masks; the reconstruction (points and camera centres) scaled by `scale`, the snap told
    x1.00 (a common scale error nobody corrected). arc: photos over that many degrees (None: all round)."""
    import test_lines3d as tl
    truth = lone_tower(top)
    bear = np.linspace(0, 360, 16, endpoint=False) if arc is None else np.linspace(0, arc, 12)
    cams0 = tl.ring(bear, radii=(60.0, 90.0), heights=(1.6, 6.0), target=(0, 0, 22))
    ids = tl.render(truth, cams0)
    P0, q = tl.sfm_points(truth, cams0, ids)
    sky = tl.ideal_sky(truth, cams0)
    cams = []
    for c in cams0:
        c2 = copy.copy(c)
        c2.center = c.center * scale
        cams.append(c2)
    model = lone_tower(top)
    return model, J.PhotoEvidence(cams, {"A": sky, "B": sky}, P0 * scale, q,
                                  lines=L3.lines_channel(model, cams, P0 * scale, q), snap_scale=1.0)


@pytest.mark.parametrize("arc,scale", [(None, 0.90), (80.0, 0.88)])
def test_a_common_scale_error_is_not_adopted_end_to_end(arc, scale):
    """The verifier's scale_tower.py: a correctly mapped 45 m tower, the whole reconstruction x0.90 (photos all round)
    or x0.88 (photos from one quadrant), the snap x1.00. The skyline and the 3-D lines agree on ~40 m. All round, the
    SfM wall planes measure k ~0.9: a frame ticket, nothing adopted, the published model passes G4. One quadrant: no
    similarity is measured, the scale is not established, and the lone tower is its own reference: a ticket."""
    model, ev = tower_scene(45.0, arc, scale)
    res = J.judge_landmark(model, ev)
    d = res["decisions"]["w2"]
    assert d.direction == "down" and d.adopt is None, (d.votes, d.adopt_why)
    assert model.part("w2").params["height"].value == 45.0 and not any(a["active"] for a in model.adoptions)
    assert J.check_decisions(model, res["decisions"])["passed"], J.check_decisions(model, res["decisions"])
    if arc is None:
        assert "frame ticket" in d.adopt_why and any(t["kind"] == "frame" and t["open"] for t in model.tickets)
    else:
        assert "not established" in d.adopt_why, d.adopt_why


# ---------------------------------------------------------------- round 2: roof height, stubs carried


def roof_decision(model: S.StructureModel, eaves: dict) -> J.Decision:
    """The verifier's roof-too-low-eave-right-big as judge.decide() saw it: map eave 14 / top 26, truth apex 30; the
    SfM points (28.64 m) and the 3-D lines' apex (30.12 m) agree, the skyline's roof profile is at its bound; `eaves`:
    what the lines (and optionally the skyline's eave profile) measure for the eave."""
    ch = {"skyline": {"ok": False, "why": "roof profile at a fit bound (A)"},
          "sfm-points": {"ok": True, "value": 28.64, "photos": 3, "points": 400},
          "lines3d": {"ok": True, "value": 30.12, "photos": 24, "parameter": "top", "interval": [29.9, 30.3]}}
    ev = [{"kind": "lines3d", "parameter": "top", "value": 30.12, "interval": [29.9, 30.3], "n_photos": 24, "n_lines": 7,
           "vote": "up", "implied_top": 30.12, "snap_scale": 1.0}]
    if "lines3d" in eaves:
        ev.append({"kind": "lines3d", "parameter": "eave", "value": eaves["lines3d"], "interval": [eaves["lines3d"] - 0.15,
                   eaves["lines3d"] + 0.15], "n_photos": 21, "n_lines": 5, "vote": None, "snap_scale": 1.0})
    params = {}
    if "skyline" in eaves:
        prof = {"val_sub": eaves["skyline"], "flat": False, "bound": False, "ci90_m": 0.6, "robust": True}
        params["eave"] = {"A": dict(prof), "B": dict(prof)}
    return J.Decision("w1", S.STATE_DISPUTED, "", votes={"sfm-up": 2.6, "lines-up": 4.1}, params=params, n_vis=24,
                      n_sky=24, evidence=ev, direction="up", channels=ch)


def test_a_channel_measuring_the_eave_at_the_map_value_moves_the_roof_not_the_walls():
    """OSM's roof:height 4 m too small (eave 14 m right, top 26 m, truth 30 m): adopting `height` alone would draw the
    eave at 17.38 m while the lines measured it at 14.06 m. The roof height is adopted instead (12 -> 15.38 m), the
    eave kept; both parameters photo-fitted (auto) with their map values, rechecked by the publish gate, and reverted
    together. If the skyline's eave profile says the eave moved while the lines say it did not: a ticket."""
    import test_lines3d as tl
    model = tl.oct_model(14.0, 26.0)
    b = model.built_part("w1")
    d = roof_decision(model, {"lines3d": 14.06})
    rec, why = J.adoption_rule(model, b, d, {"w1": d}, False, 1.0, None, SCALE_OK)
    assert rec and rec["eave"]["mode"] == "roof", why
    assert rec["value"] == pytest.approx(29.38) and rec["roof_height"]["value"] == pytest.approx(15.38)
    assert J.check_adoption_record(rec) is None
    d.adopt = rec
    J.apply_decisions(model, {"w1": d}, frame=FRAME_OK)
    p, bb = model.part("w1"), model.built_part("w1")
    assert bb.eave == pytest.approx(14.0) and bb.top == pytest.approx(29.38)          # the walls stay, the roof grows
    assert p.params["roof_height"].source == AUTO and p.params["roof_height"].map_value == 12.0
    assert J.check_decisions(model, {"w1": d})["passed"] and S.check_structure(model)["passed"], J.check_decisions(model)
    forged = copy.deepcopy(rec)
    forged.pop("roof_height")
    assert "eave" in J.check_adoption_record(forged)            # the eave at the map's value, but the whole height adopted
    p.params["roof_height"].value = 12.0
    assert any("roof height" in v["why"] or "differs" in v["why"] for v in J.check_decisions(model)["violations"])
    p.params["roof_height"].value = 15.38
    J.revert_adoption(model, model.adoptions[0]["id"])
    assert model.built_part("w1").eave == pytest.approx(14.0) and model.built_part("w1").top == pytest.approx(26.0)
    assert p.params["roof_height"].value == 12.0 and p.params["roof_height"].source != AUTO
    # the channels' eaves disagree: a ticket
    m2 = tl.oct_model(14.0, 26.0)
    d2 = roof_decision(m2, {"lines3d": 14.06, "skyline": 17.4})
    rec2, why2 = J.adoption_rule(m2, m2.built_part("w1"), d2, {"w1": d2}, False, 1.0, None, SCALE_OK)
    assert rec2 is None and "eaves disagree" in why2, why2
    # no eave measured: the height, the roof height kept (as before); all eaves moved with the top: the height too
    for eaves in ({}, {"lines3d": 17.4}):
        m3 = tl.oct_model(14.0, 26.0)
        d3 = roof_decision(m3, eaves)
        rec3, _ = J.adoption_rule(m3, m3.built_part("w1"), d3, {"w1": d3}, False, 1.0, None, SCALE_OK)
        assert rec3 and rec3["eave"]["mode"] == "height" and "roof_height" not in rec3


def test_a_height_pending_stub_resting_on_an_adopted_part_is_carried_but_stays_a_stub(scene):
    """The verifier's stub_carry.py: a stub (min_height 2, no height: the 3 m default) inside the tower that D15
    adopts (26 -> 30 m). It is carried (its base moves with the tower, a carry record), but its placeholder top keeps
    its source default-style and its state stub: still a 'height pending' stub, covered by every stub check; the
    publish gates pass, and the revert restores it."""
    cams, sky = scene
    specs = [S.PartSpec("w1", "yes", tj.BOX.copy(), [], {"height": "12"}),
             S.PartSpec("w2", "yes", tj.TOWER.copy(), [], {"height": "26"}),
             S.PartSpec("s", "yes", CAP.copy(), [], {"min_height": "2"})]
    model = S.build_structure("synthetic", specs, [tj.BOX.copy()], {})
    assert model.part("s").parent == "w2" and st.is_stub_part(model.part("s"))
    d = J.decide(model, profiles(model, cams, sky), sfm_for(model), frame=FRAME_OK)
    assert d["w2"].adopt, d["w2"].adopt_why
    J.apply_decisions(model, d, frame=FRAME_OK)
    s = model.part("s")
    h, mh = s.params["height"], s.params["min_height"]
    dz = model.part("w2").top - 26.0
    assert h.source == st.PROV_DEFAULT and h.state == S.STATE_STUB and h.value == pytest.approx(3.0 + dz)
    assert mh.source == AUTO and mh.value == pytest.approx(2.0 + dz)
    assert st.is_stub_part(s) and "s" in st.stub_part_ids(model) and s.state() == S.STATE_STUB
    assert J.check_decisions(model, d)["passed"] and S.check_structure(model)["passed"], J.check_decisions(model, d)
    h.value += 4.0                                          # a forged placeholder top that does not add up
    assert any(v["id"] == "s" and "carried" in v["why"] for v in J.check_decisions(model)["violations"])
    h.value -= 4.0
    J.revert_adoption(model, model.adoptions[0]["id"])
    assert h.value == 3.0 and mh.value == 2.0 and h.source == st.PROV_DEFAULT and st.is_stub_part(s)


def test_an_adopted_outline_travels_with_every_node_and_its_shape_line():
    """D15 ⑥ in the published model: every part node of a frame-adopted structure carries outline_adopted (the
    adoption's id, scale, shift, the map's outline width), not only the scene extras; the outline provenance is
    photo-fitted (auto)."""
    scene = photographed_stub_scene()
    model = scene.structure
    W = J._outline_width(model)
    ch = {"measured": True, "k": 1.0, "t": [0.3, 0.0], "photos": 3, "at_bound": False}
    rec = {"rule": "D15", "param": "outline", "k": 1.0, "t": [0.3, 0.0], "centre": [5.0, 0.0],
           "channels": {"sfm-points": dict(ch), "lines3d": dict(ch)}, "width_map_m": round(W, 3), "width_m": round(W, 3),
           "snap_scale": 1.0, "provenance": AUTO}
    a = J.adopt_frame(model, {"adopt": rec})
    assert a and J.check_decisions(model)["passed"]
    m = st.stylize_landmark(scene)
    parts = [p for p in m.parts if p.extras.get("part")]
    assert parts and all(p.extras["outline_adopted"]["adoption"] == a["id"] for p in parts), parts[0].extras
    assert all(p.extras["provenance"]["outline"] == AUTO for p in parts)
    oa = parts[0].extras["outline_adopted"]
    assert oa["k"] == 1.0 and oa["t"] == [0.3, 0.0] and oa["width_map_m"] == pytest.approx(W, abs=1e-3)
