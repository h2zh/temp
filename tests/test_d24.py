"""D24 (docs/STRUCTURE_FIRST.md): facade relief built on the photos' points alone where the wall's 3-D lines are at
chance -- the switch, the per-model scale condition, the provenance and evidence each element carries, and the checks
that keep it honest (check_honesty, audit_model, audit_glb, structure_metrics)."""

from __future__ import annotations

import copy

import numpy as np
import pytest
import test_relief as tr      # the synthetic facades (pytest puts tests/ on the path)
import test_stylize as ts
from test_structure_stylize import box_structure

from crowd3d import relief as RL
from crowd3d import stylize as st
from crowd3d import structmetrics as SM
from crowd3d.lines3d import Lines3D

# ------------------------------------------------------------------ scenes: a lined facade, walls seen all round


def grid_lines(base: Lines3D | None, num_cams: int = 14) -> Lines3D:
    """The south wall lined every 0.5 m both ways (tiles, courses, marble stripes) plus `base`: a line lies near any
    edge, so the element's own edge lines are no better than chance (G-R3 fails at chance, edges still confirmed)."""
    xs, zs = np.arange(0.25, 20.0, 0.5) - 10.0, np.arange(0.25, 9.5, 0.5)
    A = [[x, -6.0, 0.5] for x in xs] + [[-9.5, -6.0, z] for z in zs]
    B = [[x, -6.0, 9.0] for x in xs] + [[9.5, -6.0, z] for z in zs]
    if base is not None:
        A += [list(p) for p in base.A]
        B += [list(p) for p in base.B]
    n = len(A)
    return Lines3D(np.array(A, float), np.array(B, float), np.full(n, num_cams), [list(range(num_cams))] * n,
                   np.ones(n), np.zeros(n))


def all_round(scene: st.SceneInput, k: float = 1.0, n: int = 1500, seed: int = 3) -> st.SceneInput:
    """Add points on the box's other three walls (3-9.5 m high, well tracked; scaled by k about its centre), so that
    the whole-building similarity is measured all round (D24's scale condition: established at k = 1)."""
    rng = np.random.default_rng(seed)
    out = []
    for side in (1, 2, 3):
        t, z = rng.uniform(0, 1, n), rng.uniform(3.0, 9.5, n)
        if side == 2:
            x, y = -10 + 20 * t, np.full(n, 6.0)
        else:
            x, y = np.full(n, 10.0 if side == 1 else -10.0), -6 + 12 * t
        out.append(np.stack([x * k, y * k, z], 1))
    P = np.vstack(out)
    scene.points = np.vstack([scene.points, P])
    scene.colors = np.vstack([scene.colors, np.tile([[180, 180, 180]], (len(P), 1))]).astype(np.uint8)
    scene.point_quality = np.vstack([scene.point_quality, np.tile([[5.0, 20.0]], (len(P), 1))])
    return scene


def d24_scene(kind: str = "panel", round_k: float | None = 1.0) -> st.SceneInput:
    """The synthetic south facade (a recessed panel, or two bays and a window) on the OSM structure frame (a flat box
    part), its 3-D lines a grid, and (round_k) points on the other walls."""
    scene = tr.relief_scene() if kind == "panel" else tr.bays_scene()
    scene.structure = box_structure("flat")
    scene.lines3d = grid_lines(scene.lines3d)
    return all_round(scene, round_k) if round_k is not None else scene


@pytest.fixture(scope="module")
def d24_model() -> st.StylizedModel:
    return st.stylize_landmark(d24_scene())


def d24_node(model) -> st.Part:
    return next(p for p in model.parts if p.extras.get("relief"))


def checks(model, parts, scene=None, structure="scene", lines="scene", extras="model"):
    """(check_honesty, audit_model, audit_glb) of (mutated) parts of a D24 model: the audit gets the scene's points,
    3-D lines and structure frame (D24's scale condition re-derived from them), the GLB its scene extras."""
    scene = scene or d24_scene()
    h = st.check_honesty(parts, model.facades)
    a = st.audit_model(model.massing, parts, model.facades, model.orthophotos, model.observations,
                       np.asarray(scene.points, float), [ts.BOX.copy()], scene.cameras,
                       track=np.asarray(scene.point_quality)[:, 0], lines=scene.lines3d if lines == "scene" else lines,
                       structure=scene.structure if structure == "scene" else structure, snap_scale=1.0,
                       quality=np.asarray(scene.point_quality))
    sx = st.read_glb(model.glb)[0]["scenes"][0]["extras"] if extras == "model" else extras
    g = st.audit_glb(st.write_glb(parts, sx))
    return h, a, g


# ------------------------------------------------------------------ the switch and the pipeline


def test_d24_builds_a_recess_on_the_points_alone_where_the_scale_is_established(d24_model):
    """D24 on: the panel passes every point-based guard, its edges are at line clusters but no better than chance on
    the lined facade (G-R3 fails), and the model's common scale is established on its own data (the SfM wall planes
    measured all round, k 1.000): it is built, with provenance 'photo-measured (points, D24)' and its D24 evidence --
    the G-R3 statistics, both photo splits' measurements of it, the ring beside it and the scale condition -- and the
    model passes check_honesty, audit_model (the scale condition re-derived), audit_glb and structure_metrics."""
    s = d24_model.summary
    assert s["qa"]["passed"], s["qa"]
    r = s["relief"]
    assert r["built"] == 1 and r["built_d24"] == 1 and r["by_provenance"] == {RL.PROV_D24: 1}, r
    assert r["d24"]["enabled"] and r["d24"]["allowed"] and r["d24"]["by"] == "scale_status" and r["d24"]["built"] == 1
    assert not r["possible_decisions"] or all("elements" not in d for d in r["possible_decisions"])
    rec = next(x for x in d24_node(d24_model).extras["relief"] if x["built"])
    assert rec["provenance"] == RL.PROV_D24 and rec["type"] == "opening"
    assert abs(rec["u0"] - tr.OPEN[0]) < 0.2 and abs(rec["u1"] - tr.OPEN[1]) < 0.2 and abs(rec["offset_m"] + 0.30) < 0.05
    assert rec["guards"]["G-R3_line_edge"] is False and all(v for k, v in rec["guards"].items() if k != "G-R3_line_edge")
    d = rec["d24"]
    assert d["decision"].startswith("D24") and "2026-10-24" in d["decision"]
    assert d["lines3d"]["edges_confirmed"] >= 1 and d["lines3d"]["p_chance"] > RL.DEFAULT_GUARDS.line_null_p
    assert len(d["lines3d"]["chance"]) == d["lines3d"]["edges"] and min(d["lines3d"]["chance"]) > 0.3
    for split in ("interleaved", "viewpoint"):
        h = d["halves"][split]
        for half in ("a", "b"):
            assert abs(h[half]["u0"] - tr.OPEN[0]) < 0.3 and abs(h[half]["depth"] - 0.30) < 0.05, h
        assert min(h["photos"]) >= RL.DEFAULT_GUARDS.min_photos_per_half
    assert all(x <= -0.15 for x in d["ring"]["ring_m"]) and d["ring"]["interior_ok"] is True
    assert d["scale_condition"] == {"allowed": True, "by": "scale_status", "why": r["d24"]["why"]}
    qa = s["qa"]
    assert qa["relief_unbacked"] == 0 and qa["audit"]["relief"]["d24_records"] == 1
    assert qa["audit"]["relief"]["d24"]["allowed"] is True and qa["glb"]["relief"]["violations"] == []
    ex = st.read_glb(d24_model.glb)[0]["scenes"][0]["extras"]
    assert ex["relief"]["built_d24"] == 1 and ex["relief"]["d24"] == {"enabled": True, "allowed": True, "by": "scale_status"}
    sm = s["structure_metrics"]
    assert sm["metrics"]["struct_facade_elements"] == 1 and sm["metrics"]["struct_facade_elements_d24"] == 1
    assert sm["metrics"]["struct_unbacked"] == 0 and sm["detail"]["photo_elements_d24"] == 1
    assert sm["census"]["facade_elements_built_d24"] == {"opening": 1}
    assert sm["detail"]["photo_dof_measured_d24"] == sm["detail"]["photo_dof_measured"] == 5
    assert checks(d24_model, d24_model.parts)[1]["passed"]


@pytest.mark.parametrize("why, scene_of", [
    ("walls seen on one side only: no similarity measured, no frame adoption", lambda: d24_scene(round_k=None)),
    ("the other walls' points 6 % out: a similarity measured all round contradicts the frame", lambda: d24_scene(round_k=1.06)),
    ("no structure frame (an OSM-footprint massing)", lambda: (lambda s: (setattr(s, "structure", None), s)[1])(d24_scene())),
])
def test_where_the_scale_condition_is_not_met_nothing_is_built_on_the_points_alone(why, scene_of):
    """D24 on, the same panel and lines, but this model's scale condition is not met: nothing is built; the panel is
    listed as a possible decision that says why (the scale condition and its reason), and QA passes."""
    m = st.stylize_landmark(scene_of())
    r = m.summary["relief"]
    assert m.summary["qa"]["passed"], m.summary["qa"]
    assert r["built"] == 0 and r["built_d24"] == 0 and r["d24"]["enabled"] and not r["d24"]["allowed"], (why, r["d24"])
    dec = [d for d in r["possible_decisions"] if "elements" in d]
    assert len(dec) == 1 and dec[0]["elements"][0]["builds_geometry"], dec
    assert "scale condition is not met" in dec[0]["status"] and dec[0]["scale_condition"]["allowed"] is False


def test_with_the_switch_off_nothing_is_built_on_the_points_alone(monkeypatch):
    """D24_ENABLED off (as before the decision): the model whose scale condition is met builds nothing on the points
    alone, lists the panel as switched off, and any D24 record is refused by every check."""
    monkeypatch.setattr(RL, "D24_ENABLED", False)
    m = st.stylize_landmark(d24_scene())
    r = m.summary["relief"]
    assert r["built"] == 0 and r["d24"]["allowed"] and not r["d24"]["enabled"] and m.summary["qa"]["passed"]
    dec = [d for d in r["possible_decisions"] if "elements" in d]
    assert len(dec) == 1 and "switched off" in dec[0]["status"]


def test_a_d24_model_fails_every_check_once_the_switch_is_off(d24_model, monkeypatch):
    assert all(x["passed"] for x in checks(d24_model, d24_model.parts))
    monkeypatch.setattr(RL, "D24_ENABLED", False)
    h, a, g = checks(d24_model, d24_model.parts)
    assert not h["passed"] and not a["passed"] and not g["passed"]
    assert any("switched off" in v["why"] for v in a["relief"]["violations"])


# ------------------------------------------------------------------ the audits on a D24 model (forgeries)


def _forge(model, how):
    parts = [copy.deepcopy(p) if p.extras.get("relief") else p for p in model.parts]
    node = next(p for p in parts if p.extras.get("relief"))
    wall = model.massing.walls[node.extras["wall"]]
    recs = node.extras["relief"]
    rec = next(r for r in recs if r["built"])
    if how == "the D24 evidence block deleted":
        rec.pop("d24")
    elif how == "the record's scale condition set off":
        rec["d24"]["scale_condition"]["allowed"] = False
    elif how == "no edge at a line cluster (lines absent, not at chance)":
        rec["d24"]["lines3d"]["edges_confirmed"] = 0
        rec["channels"]["lines3d"]["edges_confirmed"] = 0
    elif how == "one photo split's measurements deleted":
        rec["d24"]["halves"]["viewpoint"].pop("b")
    elif how == "the halves' measurements of another place (4.5 m away)":
        for h in rec["d24"]["halves"].values():
            for k in ("a", "b"):
                h[k] = dict(h[k], u0=h[k]["u0"] - 4.5, u1=h[k]["u1"] - 4.5)
    elif how == "the ring beside it shows no step":
        rec["d24"]["ring"]["ring_m"] = [0.01, -0.02]
    elif how == "a point-based guard failed":
        rec["guards"]["G-R6_points_recheck"] = False
    elif how == "a normal record carrying a D24 block":
        rec["provenance"] = RL.PROV_MEASURED
    elif how == "a D24 recess deepened, records and geometry consistent":
        rec["offset_m"], rec["depth"] = -0.6, 0.6
        tr._rebuild(node, wall, tr.records_as_elements(recs), recs)
    elif how == "a made-up D24 recess on the flat wall, records and geometry consistent":
        fake = dict(copy.deepcopy(rec), id=f"w{wall.index}-o98", u0=1.5, u1=3.5, z_bottom=2.0, z_top=6.0, offset_m=-0.3)
        for h in fake["d24"]["halves"].values():
            for k in ("a", "b"):
                h[k] = dict(h[k], u0=1.5, u1=3.5)
        recs.append(fake)
        tr._rebuild(node, wall, tr.records_as_elements(recs), recs)
    else:
        raise ValueError(how)
    return parts


@pytest.mark.parametrize("how, caught_by", [
    ("the D24 evidence block deleted", (0, 2)),
    ("the record's scale condition set off", (0, 2)),
    ("no edge at a line cluster (lines absent, not at chance)", (0, 2)),
    ("one photo split's measurements deleted", (0, 2)),
    ("the halves' measurements of another place (4.5 m away)", (0, 2)),
    ("the ring beside it shows no step", (0, 2)),
    ("a point-based guard failed", (0, 2)),
    ("a normal record carrying a D24 block", (0, 2)),
    ("a D24 recess deepened, records and geometry consistent", (1,)),
    ("a made-up D24 recess on the flat wall, records and geometry consistent", (1,)),
])
def test_the_checks_catch_forged_d24_records(d24_model, how, caught_by):
    parts = _forge(d24_model, how)
    got = [x["passed"] for x in checks(d24_model, parts)]
    for i in caught_by:
        assert got[i] is False, (how, got)


def test_audit_model_re_derives_the_scale_condition_and_the_lines_itself(d24_model):
    """The record's own scale condition and line numbers are not the reference: audit_model re-derives D24's scale
    condition from the points, 3-D lines and structure frame it is given -- without the frame (or with the other
    walls' points 6 % out) the D24 record fails, though the record says 'allowed' -- and recomputes the line evidence:
    with no 3-D line at any of its edges (the lines moved 40 m up) the D24 record fails (lines absent, not at chance);
    with clean lines at its edges it passes (lines above chance are no reason to refuse it)."""
    parts = d24_model.parts
    _, a, _ = checks(d24_model, parts, structure=None)
    assert not a["passed"] and any("scale condition" in v["why"] for v in a["relief"]["violations"]), a["relief"]
    _, a, _ = checks(d24_model, parts, scene=d24_scene(round_k=1.06))
    assert not a["passed"] and "in question" in a["relief"]["d24"]["why"], a["relief"].get("d24")
    sc = d24_scene()
    far = Lines3D(sc.lines3d.A + [0.0, 0.0, 40.0], sc.lines3d.B + [0.0, 0.0, 40.0], sc.lines3d.n_images,
                  sc.lines3d.photos, sc.lines3d.support, sc.lines3d.resid)
    _, a, _ = checks(d24_model, parts, lines=far)
    assert not a["passed"] and any("no edge at a line cluster" in v["why"] for v in a["relief"]["violations"])
    a_, b_, c_, d_ = tr.OPEN
    clean = tr.south_lines([(a_, c_, a_, d_), (b_, c_, b_, d_), (a_, c_, b_, c_), (a_, d_, b_, d_)])
    assert checks(d24_model, parts, lines=clean)[1]["passed"]


def test_audit_glb_needs_the_models_d24_state_in_the_scene_extras(d24_model):
    """The GLB's scene extras carry the model's D24 state; a D24 record in a GLB whose extras do not record D24 on and
    allowed, or whose count of built D24 records differs from the nodes', fails audit_glb."""
    sx = st.read_glb(d24_model.glb)[0]["scenes"][0]["extras"]
    assert checks(d24_model, d24_model.parts)[2]["passed"]
    for bad in ({}, {**sx, "relief": {**sx["relief"], "d24": {"enabled": True, "allowed": False, "by": None}}},
                {**sx, "relief": {**sx["relief"], "built_d24": 3}}):
        g = checks(d24_model, d24_model.parts, extras=bad)[2]
        assert not g["passed"] and g["relief"]["violations"], bad.get("relief")


def test_merged_bays_with_d24_records_fail_audit_model():
    """N10 under D24 (the gap the user required closed before D24): on the lined two-bay facade the bays and the
    window are built on the points alone (D24); a forger merges both bays and the pier between them into one D24
    record (records and geometry consistent, the D24 evidence copied, the interior check claimed passed, the lines
    at chance as everywhere on this facade). Every record check passes, but audit_model's interior profile from the
    raw points shows the deleted pier."""
    scene = d24_scene("bays")
    m = st.stylize_landmark(scene)
    s = m.summary
    assert s["qa"]["passed"] and s["relief"]["built_d24"] == 3, (s["qa"], s["relief"])
    parts = [copy.deepcopy(p) if p.extras.get("relief") else p for p in m.parts]
    node = next(p for p in parts if p.extras.get("relief"))
    wall = m.massing.walls[node.extras["wall"]]
    lower = sorted((r for r in node.extras["relief"] if r["built"] and r["z_top"] < 5.5), key=lambda r: r["u0"])
    assert len(lower) == 2 and all(r["provenance"] == RL.PROV_D24 for r in lower)
    keep = [r for r in node.extras["relief"] if r not in lower]
    fake = dict(copy.deepcopy(lower[0]), id=f"w{wall.index}-o97", u0=lower[0]["u0"], u1=lower[1]["u1"])
    fake["recheck"] = dict(fake["recheck"], interior={"v": [], "h": [], "v_m": 0.0, "h_m": 0.0, "step_m": 0.15, "ok": True})
    tr._rebuild(node, wall, tr.records_as_elements(keep + [fake]), keep + [fake])
    sx = st.read_glb(m.glb)[0]["scenes"][0]["extras"]
    h, a, g = checks(m, parts, scene=scene, extras=sx)
    assert not g["passed"] and "count 3 built D24" in g["relief"]["violations"][-1]["why"]    # a careless forger
    sx["relief"]["built_d24"] = 2                                                              # ... and a careful one
    h, a, g = checks(m, parts, scene=scene, extras=sx)
    assert h["passed"] and g["passed"], (h.get("relief_violations"), g["relief"])
    why = [v["why"] for v in a["relief"]["violations"] if v.get("element") == fake["id"]]
    assert not a["passed"] and any("structure at the frame plane" in x for x in why), a["relief"]


# ------------------------------------------------------------------ units: apply_d24, records, metrics


def _guarded_lined(inner=None, lines=True):
    """The synthetic 10 x 12 m wall with a recessed opening (and optionally a deeper one inside it) on a facade lined
    every 0.5 m, guarded with both splits and the raw-point recheck: (wall, elements)."""
    def lined(w):
        return tr.lines_on(w, [(x, 0.2, x, 11.8) for x in np.arange(0.25, 10.0, 0.5)]
                           + [(0.2, z, 9.8, z) for z in np.arange(0.25, 12.0, 0.5)])
    w, P, WR = tr.measured(extra_lines=lined if lines else None, lines=lines, inner=inner)
    U = RL.to_wall(w.a, w.tangent, w.normal, P)
    els = RL.guard_elements(WR, [WR, WR], [10, 10], np.ones((120, 100), bool), w, sides=[WR, WR], per_side=[10, 10],
                            recheck=lambda r: RL.audit_points(r, w, P, None, U=U), clear=lambda r: 1.0)
    return w, els


def _res(els, allowed=True):
    return RL.ReliefResult(walls={0: {"wall": 0}}, elements={0: els},
                           stats={"d24": {"enabled": RL.D24_ENABLED, "allowed": allowed,
                                          "by": "scale_status" if allowed else None, "why": "test"}})


def test_apply_d24_accepts_only_candidates_at_chance_on_an_allowed_model():
    w, els = _guarded_lined()
    e = tr.opening(els)
    assert not e["accepted"] and RL.d24_candidate(e) and e["line_p_chance"] > 0.1 and e["edges_confirmed"] >= 1
    off = copy.deepcopy(els)
    assert RL.apply_d24(_res(off, allowed=False)) == {"applied": 0, "built": 0, "children": 0}
    assert not tr.opening(off)["accepted"]
    res = _res(els)
    assert RL.apply_d24(res)["applied"] >= 1
    e = tr.opening(els)
    assert e["accepted"] and e["builds_geometry"] and e["accepted_by"] == "D24" and e["d24"]["scale_condition"]["allowed"]
    assert res.walls[0]["accepted_d24"] >= 1 and res.stats["d24"]["applied"] >= 1
    rec = RL.element_record(e, True)
    assert rec["provenance"] == RL.PROV_D24 and RL.record_violations(rec) == [], RL.record_violations(rec)
    assert RL.line_chance_decisions(res) == []
    # no line at any edge: no line channel at chance -> not a candidate, never built
    _, bare = _guarded_lined(lines=False)
    e0 = tr.opening(bare)
    assert e0["edges_confirmed"] == 0 and not RL.d24_candidate(e0)
    RL.apply_d24(_res(bare))
    assert not tr.opening(bare)["accepted"]


def test_apply_d24_judges_a_nested_opening_with_its_parent_as_it_now_stands():
    """A door inside the opening, both at chance on the lined facade: before D24 the door fails parent_accepted;
    D24 accepts the parent first, then the door (a candidate itself). A child inside a parent D24 does not accept
    stays out."""
    w, els = _guarded_lined(inner=(3.4, 4.6, 3.0, 5.4, 0.25))
    par, kid = tr.opening(els, 1), tr.opening(els, 2)
    assert not kid["guards"]["parent_accepted"] and RL.d24_candidate(kid)
    blocked = copy.deepcopy(els)
    tr.opening(blocked, 1)["guards"]["G-R2_split_half"] = False         # the parent fails a point-based guard
    RL.apply_d24(_res(blocked))
    assert not tr.opening(blocked, 1)["accepted"] and not tr.opening(blocked, 2)["accepted"]
    RL.apply_d24(_res(els))
    assert par["accepted"] and kid["accepted"] and kid["guards"]["parent_accepted"] and kid["accepted_by"] == "D24"


def test_a_d24_record_needs_its_own_evidence():
    w, els = _guarded_lined()
    RL.apply_d24(_res(els))
    rec = RL.element_record(tr.opening(els), True)
    assert RL.record_violations(rec) == []

    def bad(f):
        r = copy.deepcopy(rec)
        f(r)
        return RL.record_violations(r)
    assert any("without its D24 evidence" in x for x in bad(lambda r: r.pop("d24")))
    assert any("scale condition" in x for x in bad(lambda r: r["d24"]["scale_condition"].update(allowed=False)))
    assert any("at chance" in x for x in bad(lambda r: r["d24"]["lines3d"].update(edges_confirmed=0)))
    assert any("at chance" in x for x in bad(lambda r: r["d24"]["lines3d"].pop("chance")))
    assert any("interleaved photo halves" in x for x in bad(lambda r: r["d24"]["halves"]["interleaved"].pop("a")))
    assert any("do not overlap" in x for x in bad(lambda r: r["d24"]["halves"]["viewpoint"]["b"].update(u0=8.0, u1=9.0)))
    assert any("ring" in x for x in bad(lambda r: r["d24"]["ring"].update(ring_m=[0.02, None])))
    assert any("ring" in x for x in bad(lambda r: r["d24"]["ring"].update(ring_m=[None, None])))
    assert any("point-based guard" in x for x in bad(lambda r: r["guards"].update({"G-R5_observed_texels": False})))
    assert any("does not say D24" in x for x in bad(lambda r: r.update(provenance=RL.PROV_MEASURED)))
    # without D24 provenance the same element is a record whose lines are at chance: unbacked as before
    plain = copy.deepcopy(rec)
    plain["provenance"] = RL.PROV_MEASURED
    plain.pop("d24")
    assert any("at chance" in x for x in RL.record_violations(plain))


def _metrics_element(**kw):
    e = {"type": "opening", "kind": "bay", "u0": 2.0, "u1": 4.0, "z_bottom": 1.0, "z_top": 5.0, "depth": 0.3,
         "provenance": RL.PROV_D24,
         "channels": {"sfm_points": True, "split_half": {"iou_a": 0.9, "iou_b": 0.9, "depth_diff_m": 0.01},
                      "lines3d": {"edges_confirmed": 2, "edges": 4, "photos": 12, "p_chance": 0.6}},
         "photos_per_half": [10, 9], "line_photos": ["a.jpg", "b.jpg", "c.jpg"],
         "guards": {"G-R1": True, "G-R3_line_edge": False}, "built": True,
         "d24": {"scale_condition": {"allowed": True}, "lines3d": {"edges_confirmed": 2, "p_chance": 0.6, "chance": [0.5] * 4},
                 "halves": {s: {"a": {"u0": 2.0}, "b": {"u0": 2.0}} for s in ("interleaved", "viewpoint")}}}
    e.update(kw)
    return e


@pytest.mark.parametrize("change, backed", [
    ({}, True),
    ({"d24": None}, False),
    ({"d24": {"scale_condition": {"allowed": False}, "lines3d": {"edges_confirmed": 2, "p_chance": 0.6, "chance": []},
              "halves": {s: {"a": {}, "b": {}} for s in ("interleaved", "viewpoint")}}}, False),
    ({"guards": {"G-R1": False, "G-R3_line_edge": False}}, False),
    ({"provenance": "photo-measured"}, False),            # lines at chance without the D24 provenance
])
def test_structure_metrics_count_d24_elements_by_their_own_evidence(tmp_path, change, backed):
    from test_structmetrics import model, reg
    e = _metrics_element(**change)
    m = SM.structure_metrics(reg(), model(tmp_path, relief=[e]))["metrics"]
    assert (m["struct_facade_elements"], m["struct_facade_elements_d24"], m["struct_unbacked"]) == \
        ((1, 1, 0) if backed else (0, 0, 1)), (change, m)
