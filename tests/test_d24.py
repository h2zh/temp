"""D24 (docs/STRUCTURE_FIRST.md): facade relief built on the photos' points alone where the wall's 3-D lines are at
chance -- the per-model scale condition, the held-out photos, the provenance and evidence each element carries, and the
checks that keep it honest (check_honesty, audit_model, audit_glb, structure_metrics)."""

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
    scene.cameras += seeing_cameras()
    return scene


def seeing_cameras() -> list[st.PosedCamera]:
    """One camera (grey photo) in front of each of the east, west and north walls: points on a wall no camera faces
    cannot lie there (the registration gate refuses them, STYLIZED_MODEL 3.2), so the walls whose points measure the
    similarity all round are photographed too. Their bearings sort after the south cameras', so the alternating photo
    split of the south wall is unchanged."""
    north = ts.look_at([0.0, 32.0, 1.6], [0.0, 6.0, 5.0], name="side-north")
    north.image = np.full((north.height, north.width, 3), 128, np.uint8)
    return ts.side_cameras() + [north]


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


# ------------------------------------------------------------------ the pipeline


def test_d24_builds_a_recess_on_the_points_alone_where_the_scale_is_established(d24_model):
    """The panel passes every point-based guard, its edges are at line clusters but no better than chance on the lined
    facade (G-R3 fails), and the model's common scale is established on its own data (the SfM wall planes measured all
    round, k 1.000): it is built, with provenance 'photo-measured (points, D24)' and its D24 evidence -- the G-R3
    statistics, both photo splits' measurements of it, its held-out scores and the scale condition -- and the model
    passes check_honesty, audit_model, audit_glb and structure_metrics."""
    s = d24_model.summary
    assert s["qa"]["passed"], s["qa"]
    r = s["relief"]
    assert r["built"] == 1 and r["built_d24"] == 1 and r["by_provenance"] == {RL.PROV_D24: 1}, r
    assert r["d24"]["allowed"] and r["d24"]["by"] == "scale_status" and r["d24"]["built"] == 1
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
    assert d["heldout"]["sides_ok"] is not False and "why" in d["heldout"]
    assert d["scale_condition"] == {"allowed": True, "by": "scale_status", "why": r["d24"]["why"]}
    # every record carries the model's scale condition (for a line-confirmed element a caveat)
    assert all(x["scale_condition"] == {"met": True, "by": "scale_status"} for x in d24_node(d24_model).extras["relief"])
    qa = s["qa"]
    assert qa["relief_unbacked"] == 0 and qa["audit"]["passed"] and qa["glb"]["relief"]["violations"] == []
    ex = st.read_glb(d24_model.glb)[0]["scenes"][0]["extras"]
    assert ex["relief"]["built_d24"] == 1
    sm = s["structure_metrics"]
    assert sm["metrics"]["struct_facade_elements"] == 1 and sm["metrics"]["struct_facade_elements_d24"] == 1
    assert sm["metrics"]["struct_unbacked"] == 0 and sm["detail"]["photo_elements_d24"] == 1
    assert sm["census"]["facade_elements_built_d24"] == {"opening": 1}
    assert sm["detail"]["photo_dof_measured_d24"] == sm["detail"]["photo_dof_measured"] == 5


@pytest.mark.parametrize("why, scene_of", [
    ("walls seen on one side only: no similarity measured, no frame adoption", lambda: d24_scene(round_k=None)),
    ("the other walls' points 6 % out: a similarity measured all round contradicts the frame", lambda: d24_scene(round_k=1.06)),
    ("no structure frame (an OSM-footprint massing)", lambda: (lambda s: (setattr(s, "structure", None), s)[1])(d24_scene())),
])
def test_where_the_scale_condition_is_not_met_nothing_is_built_on_the_points_alone(why, scene_of):
    """The same panel and lines, but this model's scale condition is not met: nothing is built; the panel is listed as
    a possible decision that says why (the scale condition and its reason), and QA passes."""
    m = st.stylize_landmark(scene_of())
    r = m.summary["relief"]
    assert m.summary["qa"]["passed"], m.summary["qa"]
    assert r["built"] == 0 and r["built_d24"] == 0 and not r["d24"]["allowed"], (why, r["d24"])
    dec = [d for d in r["possible_decisions"] if "elements" in d]
    assert len(dec) == 1 and dec[0]["elements"][0]["builds_geometry"], dec
    assert "scale condition is not met" in dec[0]["status"] and dec[0]["scale_condition"]["allowed"] is False


def test_the_registration_gate_still_decides_on_the_structure_path():
    """Structure-first merge: the OSM-parts frame and D24 sit behind the same registration gate as the default path.
    Without the cameras that face the east, west and north walls, their points could not lie there (no camera sees
    them past the building, STYLIZED_MODEL 3.2): the registration is refused, so no wall shows photos and nothing is
    measured or built, however well the south wall's relief would pass every relief guard."""
    scene = d24_scene()
    scene.cameras = [c for c in scene.cameras if not c.name.startswith("side")]
    m = st.stylize_landmark(scene)
    s = m.summary
    assert s["massing_source"] == "osm-parts" and not s["registration"]["accepted"]
    assert "no camera can see" in s["registration"]["reason"]
    assert s["observed_walls"] == 0 and s["relief"]["built"] == 0 and s["relief"]["walls_measured"] == 0
    assert not any(p.extras.get("relief") for p in m.parts) and s["qa"]["passed"], s["qa"]


# ------------------------------------------------------------------ held-out photos


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
    """D24's image channel. The photos that gave an element none of the 3-D line clusters at its edges score its projected sides (the image gradient across them) against the same sides moved
    0.4 / 0.7 m along the wall. Photographed with the recesses shaded, the sides beat their copies in every held-out
    photo and the three D24 elements are built, each record carrying its held-out scores; photographed as a plain
    facade (no image edge where the points put the bays' sides) the held-out photos contradict every side and D24
    builds nothing -- the elements are listed as refused; both models pass QA."""
    m = st.stylize_landmark(heldout_scene(True))
    s = m.summary
    assert s["qa"]["passed"] and s["relief"]["built_d24"] == 3, (s["qa"], s["relief"].get("d24"))
    for r in d24_node(m).extras["relief"]:
        ho = r["d24"]["heldout"]
        assert ho["sides_ok"] is True and ho["sides"]["photos"] >= RL.HELDOUT_MIN_PHOTOS, ho
        assert ho["line_photos_excluded"] == 7
    m2 = st.stylize_landmark(heldout_scene(False))
    s2 = m2.summary
    assert s2["qa"]["passed"] and s2["relief"]["built_d24"] == 0 and s2["relief"]["d24"]["heldout_refused"] == 3, s2["relief"]
    refused = [e for d in s2["relief"]["possible_decisions"] for e in d.get("elements", [])]
    assert len(refused) == 3 and all("held-out photos contradict its sides" in (e["refused"] or "") for e in refused)


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
    assert set(parts) == {"sides"} and parts["sides"][:, 1].max() <= 2.5                 # an arch: up to its springing line
    assert RL.outline_parts({"type": "band", "u0": 0.0, "u1": 8.0, "z_bottom": 3.0, "z_top": 3.5}) == {}
    # the photos to leave out are camera indices (Lines3D.photos)
    sc = ts.south_scene(6)
    wall = st.Wall(0, 0, 0, np.array([-10.0, -6.0]), np.array([10.0, -6.0]), 0.0, 10.0, st.PROV_OSM, st.PROV_OSM_HEIGHT)
    assert RL.heldout_photos(wall, sc.cameras, list(range(6)), [0, 2, np.int64(4)]) == [1, 3, 5]


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
                           stats={"d24": {"allowed": allowed,
                                          "by": "scale_status" if allowed else None, "why": "test"}})


def test_apply_d24_accepts_only_candidates_at_chance_on_an_allowed_model():
    w, els = _guarded_lined()
    e = tr.opening(els)
    assert not e["accepted"] and RL.d24_candidate(e) and e["line_p_chance"] > 0.1 and e["edges_confirmed"] >= 1
    off = copy.deepcopy(els)
    assert RL.apply_d24(_res(off, allowed=False)) == {"applied": 0, "built": 0, "children": 0, "heldout_refused": 0}
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
    """D24's only second channel is the photo split, so every edge it builds must be found again -- also the limit edges G-R2 does not compare (a bay's sill at the ground, a register boundary): each edge
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
    assert any("at chance" in x for x in bad(lambda r: r["d24"]["lines3d"].update(p_chance=0.05)))     # line-confirmed
    assert any("interleaved photo halves" in x for x in bad(lambda r: r["d24"]["halves"]["interleaved"].pop("a")))
    assert any("held-out photo scores" in x for x in bad(lambda r: r["d24"].pop("heldout")))
    assert any("contradict its sides" in x for x in bad(lambda r: r["d24"]["heldout"].update(sides_ok=False)))
    assert any("guard did not pass" in x for x in bad(lambda r: r["guards"].update({"G-R5_observed_texels": False})))
    assert any("does not say D24" in x for x in bad(lambda r: r.update(provenance=RL.PROV_MEASURED)))
    # without D24 provenance the same element is a record whose G-R3 failed: unbacked
    plain = copy.deepcopy(rec)
    plain["provenance"] = RL.PROV_MEASURED
    plain.pop("d24")
    assert any("guard did not pass" in x for x in RL.record_violations(plain))


BOX = {"u0": 2.0, "u1": 4.0, "z_bottom": 1.0, "z_top": 5.0}


D24 = {"scale_condition": {"allowed": True}, "lines3d": {"edges_confirmed": 2, "p_chance": 0.6, "chance": [0.5] * 4},
       "halves": {s: {"a": dict(BOX), "b": dict(BOX)} for s in ("interleaved", "viewpoint")},
       "heldout": {"photos": 9, "sides_ok": True}}


def _metrics_element(**kw):
    e = {"id": "w0-o1", "type": "opening", "kind": "bay", "u0": 2.0, "u1": 4.0, "z_bottom": 1.0, "z_top": 5.0, "depth": 0.3,
         "provenance": RL.PROV_D24,
         "channels": {"sfm_points": True, "split_half": {"iou_a": 0.9, "iou_b": 0.9, "depth_diff_m": 0.01},
                      "lines3d": {"edges_confirmed": 2, "edges": 4, "photos": 12, "p_chance": 0.6}},
         "photos_per_half": [10, 9], "line_photos": ["a.jpg", "b.jpg", "c.jpg"],
         "guards": {"G-R1": True, "G-R3_line_edge": False}, "built": True, "d24": copy.deepcopy(D24)}
    e.update(kw)
    return e


@pytest.mark.parametrize("change, backed", [
    ({}, True),
    ({"d24": None}, False),
    ({"d24": dict(D24, scale_condition={"allowed": False})}, False),
    ({"d24": dict(D24, halves={"interleaved": D24["halves"]["interleaved"]})}, False),     # one photo split missing
    ({"d24": dict(D24, lines3d={"edges_confirmed": 0, "p_chance": 1.0})}, False),          # no line at any edge
    ({"d24": dict(D24, lines3d={"edges_confirmed": 2, "p_chance": 0.05})}, False),         # lines above chance
    ({"d24": dict(D24, heldout={"photos": 9, "sides_ok": False})}, False),                 # held-out photos contradict
    ({"guards": {"G-R1": False, "G-R3_line_edge": False}}, False),
    ({"provenance": "photo-measured"}, False),            # lines at chance without the D24 provenance
    ({"id": "w3-o1"}, False),                             # a record on another wall than its id names
])
def test_structure_metrics_count_d24_elements_by_their_own_evidence(tmp_path, change, backed):
    from test_structmetrics import model, reg
    e = _metrics_element(**change)
    m = SM.structure_metrics(reg(), model(tmp_path, relief=[e]))["metrics"]
    assert (m["struct_facade_elements"], m["struct_facade_elements_d24"], m["struct_unbacked"]) == \
        ((1, 1, 0) if backed else (0, 0, 1)), (change, m)


