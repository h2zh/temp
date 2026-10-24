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


def checks(model, parts, scene=None, structure="scene", lines="scene", extras="model", tracks="scene"):
    """(check_honesty, audit_model, audit_glb) of (mutated) parts of a D24 model: the audit gets the scene's points,
    3-D lines and structure frame (D24's scale condition re-derived from them), the GLB its scene extras."""
    scene = scene or d24_scene()
    h = st.check_honesty(parts, model.facades)
    a = st.audit_model(model.massing, parts, model.facades, model.orthophotos, model.observations,
                       np.asarray(scene.points, float), [ts.BOX.copy()], scene.cameras,
                       track=np.asarray(scene.point_quality)[:, 0], lines=scene.lines3d if lines == "scene" else lines,
                       structure=scene.structure if structure == "scene" else structure, snap_scale=1.0,
                       quality=np.asarray(scene.point_quality), tracks=scene.point_tracks if tracks == "scene" else tracks)
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


def test_audit_model_re_runs_the_photo_splits_of_every_d24_record(d24_model):
    """verify-r1 M2 (V1 / V1b: elements the detector rejected for G-R2 promoted to D24 records with their halves
    forged as their own box pass every record check): for a D24 element the photo split is its only second channel,
    so audit_model re-runs both splits from the 2-D observations it is given (the wall's points re-triangulated from
    each half's photos and measured again) and needs the record found again, every edge, in both halves of both.
    The control passes; with the observations of one interleaved half deleted inside the element (that half cannot
    triangulate it: what a G-R2-rejected element is) the same, honest-looking record fails; without observations a
    D24 record cannot be re-checked and fails."""
    scene = d24_scene()
    h, a, g = checks(d24_model, d24_model.parts, scene=scene)
    assert h["passed"] and a["passed"] and g["passed"] and a["relief"]["d24_resplit"] >= 1, a["relief"]
    node = d24_node(d24_model)
    wall = d24_model.massing.walls[node.extras["wall"]]
    rec = next(r for r in node.extras["relief"] if r["built"] and r["provenance"] == RL.PROV_D24)
    P = np.asarray(scene.points, float)
    U = RL.to_wall(wall.a, wall.tangent, wall.normal, P)
    inside = (U[:, 0] > rec["u0"] - 0.3) & (U[:, 0] < rec["u1"] + 0.3) & (U[:, 2] > rec["z_bottom"] - 0.3) \
        & (U[:, 2] < rec["z_top"] + 0.3)
    centre = np.mean(np.concatenate([np.stack([w.a, w.b]) for w in d24_model.massing.walls]), axis=0)
    half = RL.split_halves(scene.cameras, centre)
    tk = scene.point_tracks
    keep = ~(inside[tk.point] & (half[tk.camera] == 1))
    blind = RL.PointTracks(tk.point[keep], tk.camera[keep], tk.xy[keep])
    h2, a2, g2 = checks(d24_model, d24_model.parts, scene=scene, tracks=blind)
    why = [v["why"] for v in a2["relief"]["violations"] if v.get("element") == rec["id"]]
    assert h2["passed"] and g2["passed"] and not a2["passed"], a2["relief"]
    assert any("not found again in both halves of both photo splits" in x and "interleaved" in x for x in why), why
    _, a3, _ = checks(d24_model, d24_model.parts, scene=scene, tracks=None)
    assert not a3["passed"] and any("cannot be re-run" in v["why"] for v in a3["relief"]["violations"])


def crowd_scene(people: int = 40, seed: int = 5) -> st.SceneInput:
    """The D24 bays scene with a crowd in front of the wall: people (r 0.25 m, 1.75 m tall) 0.3-2.0 m in front,
    observed by every photo that has them in frame (verify-r1's adversarial crowd)."""
    scene = d24_scene("bays")
    rng = np.random.default_rng(seed)
    P = []
    for _ in range(people):
        x, dist = rng.uniform(-9.5, 9.5), rng.uniform(0.3, 2.0)
        a, z = rng.uniform(0, 2 * np.pi, 300), rng.uniform(0.0, 1.75, 300)
        P.append(np.stack([x + 0.25 * np.cos(a), -6.0 - dist - 0.25 + 0.25 * np.sin(a), z], 1))
    P = np.concatenate(P)
    n0 = len(scene.points)
    scene.points = np.vstack([scene.points, P])
    scene.colors = np.vstack([scene.colors, np.tile([[90, 80, 70]], (len(P), 1))]).astype(np.uint8)
    tk = tr.project_all(P, scene.cameras)
    t0 = scene.point_tracks
    scene.point_tracks = RL.PointTracks(np.r_[t0.point, tk.point + n0], np.r_[t0.camera, tk.camera], np.r_[t0.xy, tk.xy])
    n = np.bincount(tk.point, minlength=len(P))
    scene.point_quality = np.vstack([scene.point_quality, np.stack([n, np.full(len(n), 20.0)], 1)])
    return scene


def test_a_crowd_in_front_of_the_ground_floor_does_not_become_the_sill_of_d24_bays():
    """verify-r1 M3: a crowd standing 0.3-2 m in front of the facade reads as a proud band at the ground (rejected);
    its top became the boundary of the register above, and so the sill of the bays (1.5 m instead of 0.6 m) -- a limit
    edge both photo halves agree on (the people are in every photo), which D24 built. A bottom / top that is its
    register's boundary is no measured edge: D24 builds such an element only where the band making that boundary is
    accepted itself (register_bounds, apply_d24). The window above, with all its edges measured, is still built; the
    model passes every check."""
    m = st.stylize_landmark(crowd_scene())
    s = m.summary
    assert s["qa"]["passed"], s["qa"]
    node = d24_node(m)
    built = [r for r in node.extras["relief"] if r["built"]]
    assert built and all(r["z_bottom"] > 5.0 for r in built), [(r["id"], r["z_bottom"], r["z_top"]) for r in built]
    assert all((r.get("d24") or {}).get("bounded_by") == [] for r in built if r["provenance"] == RL.PROV_D24)


def test_audit_model_needs_the_band_bounding_a_d24_record_to_be_accepted(monkeypatch):
    """The audit's side of the crowd rule: a D24 record whose bottom is its register's boundary needs the band making
    that boundary (re-derived from the points) among the node's accepted records. Built without the rule (what the
    pipeline did before), the crowd scene's bays sit on the crowd's top: their records are self-consistent and both
    photo splits re-run from the observations find them (the people are in every photo), so only this check refuses
    them."""
    scene = crowd_scene()
    with monkeypatch.context() as mp:
        mp.setattr(RL, "register_bounds", lambda *a, **k: {})
        m = st.stylize_landmark(scene)
    node = d24_node(m)
    low = [r for r in node.extras["relief"] if r["built"] and r["z_bottom"] < 2.0]
    assert low and all(r["provenance"] == RL.PROV_D24 for r in low), [(r["id"], r["z_bottom"]) for r in node.extras["relief"]]
    h, a, g = checks(m, m.parts, scene=crowd_scene())
    why = [v["why"] for v in a["relief"]["violations"] if v.get("element") in {r["id"] for r in low}]
    assert h["passed"] and g["passed"] and not a["passed"], a["relief"]["violations"]
    assert why and all("register's boundary" in x for x in why), why


def test_merged_bays_with_d24_records_fail_audit_model():
    """N10 under D24 (the gap the user required closed before D24): on the lined two-bay facade the bays and the
    window are built on the points alone (D24); a forger merges both bays and the pier between them into one D24
    record (records and geometry consistent, the D24 evidence copied, the interior check claimed passed, the lines
    at chance as everywhere on this facade). Every record check passes, but audit_model's interior profile from the
    raw points shows the deleted pier. A forger who keeps one bay's photo-half measurements fails the records' own
    D24 check (every edge found again in both halves of both splits, fixer2); a careful one forges those too."""
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
    assert any("not found again in both photo halves" in v["why"] for v in h["relief_violations"])
    sx["relief"]["built_d24"] = 2                                                              # ... and a careful one
    box = {k: fake[k] for k in ("u0", "u1", "z_bottom", "z_top")}
    for sp in fake["d24"]["halves"].values():
        sp["a"], sp["b"] = dict(sp["a"], **box), dict(sp["b"], **box)
    fake["d24"]["halves_all_edges"] = RL.d24_halves_check(fake, fake["d24"]["halves"])[1]
    tr._rebuild(node, wall, tr.records_as_elements(keep + [fake]), keep + [fake])
    h, a, g = checks(m, parts, scene=scene, extras=sx)
    assert h["passed"] and g["passed"], (h.get("relief_violations"), g["relief"])
    why = [v["why"] for v in a["relief"]["violations"] if v.get("element") == fake["id"]]
    assert not a["passed"] and any("structure at the frame plane" in x for x in why), a["relief"]


@pytest.fixture(scope="module")
def d24_bays_model() -> st.StylizedModel:
    return st.stylize_landmark(d24_scene("bays"))


def _blind_half(model, scene, rec, z_lo, z_hi, half_id=1):
    """The scene's 2-D observations with one interleaved photo half's views of the wall points around `rec` (its
    width + 0.3 m, z_lo..z_hi) deleted: that half re-triangulates nothing there."""
    wall = model.massing.walls[d24_node(model).extras["wall"]]
    U = RL.to_wall(wall.a, wall.tangent, wall.normal, np.asarray(scene.points, float))
    zone = (U[:, 0] > rec["u0"] - 0.3) & (U[:, 0] < rec["u1"] + 0.3) & (U[:, 2] > z_lo) & (U[:, 2] < z_hi)
    centre = np.mean(np.concatenate([np.stack([w.a, w.b]) for w in model.massing.walls]), axis=0)
    half = RL.split_halves(scene.cameras, centre)
    tk = scene.point_tracks
    keep = ~(zone[tk.point] & (half[tk.camera] == half_id))
    return RL.PointTracks(tk.point[keep], tk.camera[keep], tk.xy[keep])


def test_a_d24_record_whose_sill_one_photo_half_measures_elsewhere_fails_audit_model(d24_bays_model):
    """verify-r2 M5 (a surviving mutant: the audit's every-edge check of a re-split D24 record replaced by a pass): the
    photo split is a D24 element's only second channel, so EVERY edge must be found again within d24_edge_m between
    the two halves. One interleaved half that never saw the window's lowest 0.5 m (its observations there deleted)
    still finds the window -- same sides, IoU far above split_iou, the raw points unchanged -- but its sill 0.5 m
    higher: only the every-edge check (d24_halves_check, in audit_model's re-split) refuses the record."""
    m, scene = d24_bays_model, d24_scene("bays")
    win = next(r for r in d24_node(m).extras["relief"] if r["built"] and r["z_bottom"] > 5.0)
    assert win["provenance"] == RL.PROV_D24
    blind = _blind_half(m, scene, win, win["z_bottom"] - 0.3, win["z_bottom"] + 0.5)
    h, a, g = checks(m, m.parts, scene=scene, tracks=blind)
    why = [v["why"] for v in a["relief"]["violations"] if v.get("element") == win["id"]]
    assert h["passed"] and g["passed"] and not a["passed"], a["relief"]
    assert why and all("every edge within" in x and "z_bottom" in x and "IoU" not in x for x in why), why
    _, a0, _ = checks(m, m.parts, scene=scene)                       # the control: every observation kept
    assert a0["passed"], a0["relief"]


def test_audit_model_cuts_the_photo_halves_where_the_points_say_not_where_a_record_says(d24_bays_model):
    """verify-r2: the audit re-ran a D24 record's photo halves cut at the verticals the record named (halves_cut_at);
    a record that emptied the list ([]) made the re-run halves merge bays. The audit now derives the cutters itself
    (halves_cutters on its own first pass, as measure_relief does) and a record naming other cutters fails: an honest
    D24 record with halves_cut_at [] or a made-up cutter fails; a merged D24 record over both bays and the pier, its
    halves and interior claim forged as a careful forger does and halves_cut_at [], fails on the cutters, on the
    re-split (its sides 3.6 m from what each half finds) and on the interior profile."""
    m, scene = d24_bays_model, d24_scene("bays")
    sx = st.read_glb(m.glb)[0]["scenes"][0]["extras"]
    node0 = d24_node(m)
    honest = node0.extras["relief"][0]
    assert honest["provenance"] == RL.PROV_D24 and "halves_cut_at" in honest["d24"]
    for forged in ([], [[7.0, 7.6, 0.5, 5.1]]):
        if forged == honest["d24"]["halves_cut_at"]:
            continue
        parts = [copy.deepcopy(p) if p.extras.get("relief") else p for p in m.parts]
        node = next(p for p in parts if p.extras.get("relief"))
        node.extras["relief"][0]["d24"]["halves_cut_at"] = forged
        h, a, g = checks(m, parts, scene=scene, extras=sx)
        why = [v["why"] for v in a["relief"]["violations"] if v.get("element") == honest["id"]]
        assert h["passed"] and g["passed"] and not a["passed"], (forged, a["relief"])
        assert any("is not the cutters the points confirm" in x for x in why), why
    # a merged D24 record (as test_merged_bays_with_d24_records_fail_audit_model's careful forger) with cut []
    parts = [copy.deepcopy(p) if p.extras.get("relief") else p for p in m.parts]
    node = next(p for p in parts if p.extras.get("relief"))
    wall = m.massing.walls[node.extras["wall"]]
    lower = sorted((r for r in node.extras["relief"] if r["built"] and r["z_top"] < 5.5), key=lambda r: r["u0"])
    keep = [r for r in node.extras["relief"] if r not in lower]
    fake = dict(copy.deepcopy(lower[0]), id=f"w{wall.index}-o97", u0=lower[0]["u0"], u1=lower[1]["u1"])
    fake["recheck"] = dict(fake["recheck"], interior={"v": [], "h": [], "v_m": 0.0, "h_m": 0.0, "step_m": 0.15, "ok": True})
    box = {k: fake[k] for k in ("u0", "u1", "z_bottom", "z_top")}
    for sp in fake["d24"]["halves"].values():
        sp["a"], sp["b"] = dict(sp["a"], **box), dict(sp["b"], **box)
    fake["d24"]["halves_all_edges"] = RL.d24_halves_check(fake, fake["d24"]["halves"])[1]
    fake["d24"]["halves_cut_at"] = []
    tr._rebuild(node, wall, tr.records_as_elements(keep + [fake]), keep + [fake])
    sx2 = copy.deepcopy(sx)
    sx2["relief"]["built_d24"] = 2
    h, a, g = checks(m, parts, scene=scene, extras=sx2)
    why = [v["why"] for v in a["relief"]["violations"] if v.get("element") == fake["id"]]
    assert h["passed"] and g["passed"] and not a["passed"], (h.get("relief_violations"), g["relief"])
    assert any("is not the cutters the points confirm" in x and "sides from the record" in x for x in why), why
    assert any("structure at the frame plane" in x for x in why), why


def heldout_scene(shaded: bool) -> st.SceneInput:
    """The lined two-bay D24 facade whose 3-D lines come from the first 7 of its 14 photos only (the other 7 are held
    out: no line cluster at any element's edge comes from them), its photos rendering a plain cream wall (sky above,
    ground below) with the recesses shaded (darker: their outline is an image edge) or not (no image edge where the
    points put the bays' sides)."""
    scene = d24_scene("bays")
    ln = scene.lines3d
    scene.lines3d = Lines3D(ln.A, ln.B, np.full(len(ln.A), 7), [list(range(7))] * len(ln.A), ln.support, ln.resid)
    for cam in scene.cameras:
        rays = ts._rays(cam)
        img = np.where((rays[..., 2] > 0)[..., None], np.array([150.0, 200.0, 250.0]), np.array([120.0, 120.0, 110.0]))
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (-6.0 - cam.center[1]) / rays[..., 1]
        hit = cam.center + t[..., None] * rays
        u, z = hit[..., 0] + 10.0, hit[..., 2]
        on = (t > 0) & (np.abs(hit[..., 0]) <= 10.0) & (z >= 0) & (z <= ts.WALL_H)
        img[on] = ts.CREAM
        if shaded:
            for a, b, c, d in tr.BAYS_U:
                img[on & (u > a) & (u < b) & (z > c) & (z < d)] *= 0.55
        cam.image = img.astype(np.uint8)
    return scene


def test_d24_builds_only_where_held_out_photos_do_not_contradict_its_sides():
    """verify-r2 (held-out photos): D24's image channel. The photos that gave an element none of the 3-D line
    clusters at its edges score its projected sides (the image gradient across them) against the same sides moved
    0.4 / 0.7 m along the wall. Photographed with the recesses shaded, the sides beat their copies in every held-out
    photo and the three D24 elements are built, each record carrying its held-out scores; photographed as a plain
    facade (no image edge where the points put the bays' sides) the held-out photos contradict every side and D24
    builds nothing -- the elements are listed as refused; both models pass QA."""
    m = st.stylize_landmark(heldout_scene(True))
    s = m.summary
    assert s["qa"]["passed"] and s["relief"]["built_d24"] == 3, (s["qa"], s["relief"].get("d24"))
    for r in d24_node(m).extras["relief"]:
        ho = r["d24"]["heldout"]
        assert ho["sides_ok"] is True and ho["sides"]["photos"] >= RL.HELDOUT_MIN_PHOTOS and "sides" in ho["above_chance"], ho
        assert ho["line_photos_excluded"] == 7
    assert s["relief"]["d24"]["heldout_parts"]["sides"] == [3, 3]
    m2 = st.stylize_landmark(heldout_scene(False))
    s2 = m2.summary
    assert s2["qa"]["passed"] and s2["relief"]["built_d24"] == 0 and s2["relief"]["d24"]["heldout_refused"] == 3, s2["relief"]
    refused = [e for d in s2["relief"]["possible_decisions"] for e in d.get("elements", [])]
    assert len(refused) == 3 and all("held-out photos contradict its sides" in (e["refused"] or "") for e in refused)


def test_audit_model_re_scores_the_held_out_photos_of_every_d24_record():
    """The audit's side of the image channel: it re-selects the held-out photos from the 3-D lines and the cameras
    (nothing from the record's line photos or held-out scores) and re-scores them. The shaded model passes; given
    the plain photos -- what a record whose held-out scores were forged would meet -- its D24 records fail, while
    check_honesty and audit_glb, which see only the records, pass; a record whose own scores say its sides are
    contradicted fails the records' check."""
    scene = heldout_scene(True)
    m = st.stylize_landmark(scene)
    h, a, g = checks(m, m.parts, scene=scene)
    assert h["passed"] and a["passed"] and g["passed"] and a["relief"]["d24_heldout"] == 3, a["relief"]
    plain = heldout_scene(False)
    h2, a2, g2 = checks(m, m.parts, scene=plain)
    why = [v["why"] for v in a2["relief"]["violations"]]
    assert h2["passed"] and g2["passed"] and not a2["passed"], a2["relief"]
    assert sum("held-out photos (re-scored from the photos) contradict its sides" in x for x in why) == 3, why
    parts = [copy.deepcopy(p) if p.extras.get("relief") else p for p in m.parts]
    node = next(p for p in parts if p.extras.get("relief"))
    ho = node.extras["relief"][0]["d24"]["heldout"]
    ho["sides"] = dict(ho["sides"], own=0.1, copies=0.3, beats=0.0)
    assert any("held-out photos contradict its sides" in x for x in RL.record_violations(node.extras["relief"][0]))


def test_heldout_verdict_needs_enough_photos_and_sides_that_beat_their_copies():
    """heldout_verdict: unchecked (None) with fewer than HELDOUT_MIN_PHOTOS photos scoring the sides or for a band
    (no sides); False when the sides' mean does not beat the copies' or beats them in under half the photos."""
    side = lambda n, o, c, b: {"sides": {"photos": n, "own": o, "copies": c, "beats": b}}
    assert RL.heldout_verdict(None)[0] is None and RL.heldout_verdict({"photos": 5, "sill": {"photos": 5}})[0] is None
    assert RL.heldout_verdict(side(RL.HELDOUT_MIN_PHOTOS - 1, 0.1, 0.5, 0.0))[0] is None
    assert RL.heldout_verdict(side(10, 0.4, 0.3, 0.6))[0] is True
    assert RL.heldout_verdict(side(10, 0.3, 0.3, 0.6))[0] is False             # the mean must beat the copies'
    assert RL.heldout_verdict(side(10, 0.5, 0.3, 0.4))[0] is False             # ... and in at least half the photos
    parts = RL.outline_parts({"type": "opening", "u0": 1.0, "u1": 3.0, "z_bottom": 1.0, "z_top": 4.0, "arched": True,
                              "spring_z": 2.5, "nesting": 1})
    assert set(parts) == {"sides", "sill", "head"} and parts["sides"][:, 1].max() <= 2.5 and parts["head"][:, 1].max() <= 4.0
    assert set(RL.outline_parts({"type": "band", "u0": 0.0, "u1": 8.0, "z_bottom": 3.0, "z_top": 3.5})) == {"sill", "head"}


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
    assert RL.apply_d24(_res(off, allowed=False)) == {"applied": 0, "built": 0, "children": 0, "bounded": 0,
                                                     "heldout_refused": 0}
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


def test_d24_needs_every_edge_found_again_in_both_photo_halves():
    """verify-r1 H1 / crowd: D24's only second channel is the photo split, so every edge it builds must be found
    again -- also the limit edges G-R2 does not compare (a bay's sill at the ground, a register boundary): each edge
    within d24_edge_m between the halves and within d24_box_m of the element, in both splits. A half that saw only the
    upper arch of a merged bay (its bottom 4.8 m up) and two halves that agree with each other but not with the
    element both make it no D24 candidate; small differences do not."""
    g = RL.DEFAULT_GUARDS
    w, els = _guarded_lined()
    e = tr.opening(els)
    assert RL.d24_candidate(e)
    ok, info = RL.d24_halves_check(e, {"interleaved": e["split"], "viewpoint": e["split_sides"]})
    assert ok and info["interleaved"]["edge_diff_m"] == 0.0 and info["viewpoint"]["element_diff_m"] == 0.0

    def moved(split, halves, **dz):
        x = copy.deepcopy(e)
        for h in halves:
            x[split][h] = dict(x[split][h], **{k: x[split][h][k] + v for k, v in dz.items()})
        return x
    upper_only = moved("split_sides", "b", z_bottom=4.8)
    assert upper_only["guards"]["G-R2_split_half"] and not RL.d24_candidate(upper_only)
    both_off = moved("split", "ab", z_bottom=g.d24_box_m + 0.1)
    assert RL.d24_halves_check(both_off, {"interleaved": both_off["split"], "viewpoint": both_off["split_sides"]})[1][
        "interleaved"]["edge_diff_m"] == 0.0 and not RL.d24_candidate(both_off)
    assert RL.d24_candidate(moved("split", "b", z_bottom=0.2, u0=-0.2))
    # a band: its bottom and top only (it spans the wall)
    band = {"type": "band", "u0": 0.0, "u1": 10.0, "z_bottom": 9.0, "z_top": 9.6}
    hv = {"a": dict(band, u0=3.0), "b": dict(band, u1=7.0, z_top=9.7)}
    assert RL.d24_halves_check(band, {"interleaved": hv, "viewpoint": hv})[0]
    assert not RL.d24_halves_check(band, {"interleaved": hv, "viewpoint": {"a": hv["a"]}})[0]


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
    # verify-r1 V5c / V5d: one set of line numbers, at chance, following from its own chance rates
    assert any("differ from the record's lines3d channel" in x
               for x in bad(lambda r: r["channels"]["lines3d"].update(p_chance=0.01)))
    def raise_conf(r):
        n = int(r["d24"]["lines3d"]["edges_confirmed"]) + 1
        r["d24"]["lines3d"]["edges_confirmed"] = n
        r["channels"]["lines3d"]["edges_confirmed"] = n
    assert any("does not follow from its own edge chance rates" in x for x in bad(raise_conf))
    def below_null(r):
        r["d24"]["lines3d"]["p_chance"] = 0.05
        r["channels"]["lines3d"]["p_chance"] = 0.05
    assert any("not at chance" in x for x in bad(below_null))
    # fixer2: every edge found again in both halves of both splits (a half that saw only the upper part fails)
    assert any("not found again in both photo halves" in x
               for x in bad(lambda r: r["d24"]["halves"]["interleaved"]["b"].update(z_bottom=r["z_bottom"] + 0.5)))
    # without D24 provenance the same element is a record whose lines are at chance: unbacked as before
    plain = copy.deepcopy(rec)
    plain["provenance"] = RL.PROV_MEASURED
    plain.pop("d24")
    assert any("at chance" in x for x in RL.record_violations(plain))


BOX = {"u0": 2.0, "u1": 4.0, "z_bottom": 1.0, "z_top": 5.0}


def _metrics_element(**kw):
    e = {"type": "opening", "kind": "bay", "u0": 2.0, "u1": 4.0, "z_bottom": 1.0, "z_top": 5.0, "depth": 0.3,
         "provenance": RL.PROV_D24,
         "channels": {"sfm_points": True, "split_half": {"iou_a": 0.9, "iou_b": 0.9, "depth_diff_m": 0.01},
                      "lines3d": {"edges_confirmed": 2, "edges": 4, "photos": 12, "p_chance": 0.6}},
         "photos_per_half": [10, 9], "line_photos": ["a.jpg", "b.jpg", "c.jpg"],
         "guards": {"G-R1": True, "G-R3_line_edge": False}, "built": True,
         "d24": {"scale_condition": {"allowed": True}, "lines3d": {"edges_confirmed": 2, "p_chance": 0.6, "chance": [0.5] * 4},
                 "halves": {s: {"a": dict(BOX), "b": dict(BOX)} for s in ("interleaved", "viewpoint")}}}
    e.update(kw)
    return e


@pytest.mark.parametrize("change, backed", [
    ({}, True),
    ({"d24": None}, False),
    ({"d24": {"scale_condition": {"allowed": False}, "lines3d": {"edges_confirmed": 2, "p_chance": 0.6, "chance": []},
              "halves": {s: {"a": {}, "b": {}} for s in ("interleaved", "viewpoint")}}}, False),
    ({"d24": {"scale_condition": {"allowed": True}, "lines3d": {"edges_confirmed": 2, "p_chance": 0.6, "chance": [0.5] * 4},
              "halves": {"interleaved": {"a": {"u0": 2.0}, "b": {"u0": 2.0}}}}}, False),      # one photo split missing
    ({"d24": {"scale_condition": {"allowed": True}, "lines3d": {"edges_confirmed": 0, "p_chance": 1.0, "chance": [0.5] * 4},
              "halves": {s: {"a": {"u0": 2.0}, "b": {"u0": 2.0}} for s in ("interleaved", "viewpoint")}}}, False),   # no line
    ({"guards": {"G-R1": False, "G-R3_line_edge": False}}, False),
    ({"provenance": "photo-measured"}, False),            # lines at chance without the D24 provenance
    ({"d24": {"scale_condition": {"allowed": True}, "lines3d": {"edges_confirmed": 2, "p_chance": 0.6, "chance": [0.5] * 4},
              "halves": {s: {"a": dict(BOX), "b": dict(BOX, z_bottom=4.6)} for s in ("interleaved", "viewpoint")}}},
     False),                                              # a half that saw only the top of it (fixer2)
    ({"channels": {"sfm_points": True, "split_half": {"iou_a": 0.9, "iou_b": 0.9, "depth_diff_m": 0.01},
                   "lines3d": {"edges_confirmed": 2, "edges": 4, "photos": 12, "p_chance": 0.01}}}, False),  # V5c
])
def test_structure_metrics_count_d24_elements_by_their_own_evidence(tmp_path, change, backed):
    from test_structmetrics import model, reg
    e = _metrics_element(**change)
    m = SM.structure_metrics(reg(), model(tmp_path, relief=[e]))["metrics"]
    assert (m["struct_facade_elements"], m["struct_facade_elements_d24"], m["struct_unbacked"]) == \
        ((1, 1, 0) if backed else (0, 0, 1)), (change, m)
