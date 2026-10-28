"""crowd3d.relief: facade relief measured from the photos (detector, split-half re-triangulation, guards G-R1..G-R5,
geometry), and the publication checks that keep it honest (check_honesty, audit_model, audit_glb)."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import numpy as np
import pytest
import test_stylize as ts   # the synthetic south-facade scene (pytest puts tests/ on the path)

from crowd3d import relief as RL
from crowd3d import stylize as st
from crowd3d.lines3d import Lines3D

# ------------------------------------------------------------------ a synthetic wall (detector and guards)


def synthetic_wall(rng, recess=(3.0, 5.0, 2.0, 6.0, 0.30), band=(9.0, 9.6, 0.25), noise=0.01, density=400,
                   inner=None):
    """A 10 m x 12 m wall along +x (outward normal (0, -1)) with one recessed opening (u0, u1, z0, z1, depth), one
    band (z0, z1, projection) and optionally a deeper recess inside the opening (u0, u1, z0, z1, extra depth)."""
    w = st.Wall(0, 0, 0, np.array([0.0, 0.0]), np.array([10.0, 0.0]), 0.0, 12.0, st.PROV_OSM, st.PROV_OSM_HEIGHT)
    n = int(10 * 12 * density)
    u = rng.uniform(0, 10, n)
    z = rng.uniform(0, 12, n)
    d = np.zeros(n)
    u0, u1, z0, z1, depth = recess
    d[(u > u0) & (u < u1) & (z > z0) & (z < z1)] -= depth
    if inner is not None:
        a, b, c, e, extra = inner
        d[(u > a) & (u < b) & (z > c) & (z < e)] -= extra
    b0, b1, p = band
    d[(z > b0) & (z < b1)] += p
    d += rng.normal(0, noise, n)
    P = np.stack([u, -d, z], 1)
    assert np.allclose(w.normal, [0, -1])
    return w, P


def lines_on(wall, segments, photos=range(8)) -> Lines3D:
    """3-D lines on the wall plane: segments (u0, z0, u1, z1), each seen by `photos`."""
    A, B = [], []
    for u0, z0, u1, z1 in segments:
        A.append(np.r_[wall.a + u0 * wall.tangent, z0])
        B.append(np.r_[wall.a + u1 * wall.tangent, z1])
    n = len(A)
    return Lines3D(np.array(A, float).reshape(-1, 3), np.array(B, float).reshape(-1, 3), np.full(n, len(list(photos))),
                   [sorted(photos)] * n, np.ones(n), np.zeros(n))


def concat_lines(*ls: Lines3D) -> Lines3D:
    return Lines3D(np.concatenate([x.A for x in ls]), np.concatenate([x.B for x in ls]),
                   np.concatenate([x.n_images for x in ls]), [p for x in ls for p in x.photos],
                   np.concatenate([x.support for x in ls]), np.concatenate([x.resid for x in ls]))


def box_lines(wall, u0, u1, z0, z1):
    return lines_on(wall, [(u0, z0, u0, z1), (u1, z0, u1, z1), (u0, z0, u1, z0), (u0, z1, u1, z1)])


def test_detects_opening_and_band():
    rng = np.random.default_rng(0)
    w, P = synthetic_wall(rng)
    wr = RL.measure_wall(w, P, np.full(len(P), 5), None)
    assert not wr.refused, wr.refused
    ops = [o for o in wr.openings if o["level"] == 1]
    assert len(ops) == 1
    o = ops[0]
    assert abs(o["u0"] - 3.0) < 0.15 and abs(o["u1"] - 5.0) < 0.15
    assert abs(o["z_bottom"] - 2.0) < 0.15 and abs(o["z_top"] - 6.0) < 0.25
    assert abs(o["depth"] - 0.30) < 0.05
    bands = [b for b in wr.bands if b["sign"] > 0]
    assert len(bands) == 1 and abs(bands[0]["z_bottom"] - 9.0) < 0.15 and abs(bands[0]["z_top"] - 9.6) < 0.15
    assert abs(bands[0]["level"] - 0.25) < 0.05


def test_flat_wall_has_no_elements():
    rng = np.random.default_rng(1)
    w, P = synthetic_wall(rng, recess=(3, 5, 2, 6, 0.0), band=(9, 9.6, 0.0), noise=0.02)
    wr = RL.measure_wall(w, P, np.full(len(P), 5), None)
    assert not wr.refused
    assert not wr.openings and not [b for b in wr.bands if b["kind"] != "plinth"] and not wr.verticals


def test_a_sparse_or_offset_wall_is_refused():
    rng = np.random.default_rng(2)
    w, P = synthetic_wall(rng, density=10)                       # 1200 points: too few for relief
    assert "too few" in RL.measure_wall(w, P, np.full(len(P), 5), None).refused
    w, P = synthetic_wall(rng)
    P[:, 1] -= 1.0                                               # the photographed surface stands 1 m off the frame wall
    assert "off the frame wall" in RL.measure_wall(w, P, np.full(len(P), 5), None).refused


def measured(depth=0.30, lines=True, inner=None, line_box=None, seed=0, recess=(3, 5, 2, 6), extra_lines=None):
    rng = np.random.default_rng(seed)
    w, P = synthetic_wall(rng, recess=(*recess, depth), band=(9, 9.6, 0.0), inner=inner)
    L = box_lines(w, *(line_box or recess)) if lines else None
    if L is not None and extra_lines is not None:
        L = concat_lines(L, extra_lines(w))
    return w, P, RL.measure_wall(w, P, np.full(len(P), 5), L)


def guarded(WR, w, halves=None, per_half=(10, 10), mask=None, sides=None, per_side=None, clear=None):
    return RL.guard_elements(WR, [WR, WR] if halves is None else halves, list(per_half),
                             np.ones((120, 100), bool) if mask is None else mask, w, clear=clear, sides=sides,
                             per_side=None if per_side is None else list(per_side))


def opening(els, level=1):
    return next(e for e in els if e["type"] == "opening" and e["level"] == level)


def test_all_guards_pass_on_a_measured_opening():
    w, P, WR = measured()
    e = opening(guarded(WR, w))
    assert e["accepted"] and e["builds_geometry"] and all(e["guards"].values()), e["guards"]
    assert abs(e["offset_m"] + 0.30) < 0.05 and e["edges_confirmed"] >= 2
    assert e["frame"] is not None and abs(e["frame"]["c0"]) < 0.05


def _flat_half(w):
    return measured(depth=0.0, seed=5)[2]


@pytest.mark.parametrize("mutation, guard", [
    ("few photos in one half", "G-R1_photos_per_half"),
    ("seen in one photo half only", "G-R2_split_half"),
    ("the halves disagree on the depth", "G-R2_split_half"),
    ("the halves disagree on an edge", "G-R2_split_half"),
    ("seen from one viewpoint side only", "G-R2_split_half"),
    ("too few photos on one viewpoint side", "G-R2_split_half"),
    ("no 3-D line at any edge", "G-R3_line_edge"),
    ("lines at its edges no better than chance", "G-R3_line_edge"),
    ("deeper than the limit", "G-R4_depth"),
    ("on unobserved texels", "G-R5_observed_texels"),
    ("hidden by something in front from most photos", "G-R5_observed_texels"),
    ("an edge at a free-standing object", "G-R7_clear_of_objects"),
])
def test_each_guard_refuses_its_failure(mutation, guard):
    """Every guard on its own, with BOTH photo splits (interleaved halves and viewpoint sides) and the clear-view
    share given, as measure_relief gives them: the unmutated element passes all, each mutation fails exactly its
    guard."""
    lined = (lambda w: lines_on(w, [(x, 0.2, x, 11.8) for x in np.arange(0.3, 10.0, 0.5)]
                                + [(0.2, z, 9.8, z) for z in np.arange(0.3, 12.0, 0.5)]))
    w, P, WR = measured(depth=1.8 if mutation == "deeper than the limit" else 0.30,
                        lines=mutation != "no 3-D line at any edge",
                        extra_lines=lined if mutation == "lines at its edges no better than chance" else None)
    kw = {"sides": [WR, WR], "per_side": (10, 10), "clear": lambda r: 1.0}
    if mutation == "few photos in one half":
        kw["per_half"] = (5, 12)
    elif mutation == "seen in one photo half only":
        kw["halves"] = [WR, _flat_half(w)]
    elif mutation == "the halves disagree on the depth":
        kw["halves"] = [WR, measured(depth=0.65, seed=3)[2]]
    elif mutation == "the halves disagree on an edge":
        kw["halves"] = [WR, measured(seed=3, recess=(3.5, 5, 2, 6))[2]]           # IoU 0.75, the left edge 0.5 m off
    elif mutation == "seen from one viewpoint side only":
        kw["sides"] = [WR, _flat_half(w)]
    elif mutation == "too few photos on one viewpoint side":
        kw["per_side"] = (3, 12)
    elif mutation == "on unobserved texels":
        kw["mask"] = np.zeros((120, 100), bool)
    elif mutation == "hidden by something in front from most photos":
        kw["clear"] = lambda r: 0.2
    elif mutation == "an edge at a free-standing object":
        WR.stats["occluders"] = [{"u0": 2.8, "u1": 3.2, "z_bottom": 0.0, "z_top": 9.0}]
    els = guarded(WR, w, **kw)
    e = opening(els)
    assert not e["guards"][guard] and not e["accepted"] and not e["builds_geometry"], (mutation, e["guards"])
    assert all(v for k, v in e["guards"].items() if k != guard), e["guards"]       # only the mutated guard fails
    if mutation == "lines at its edges no better than chance":
        assert e["edges_confirmed"] >= 1 and e["line_p_chance"] > 0.1
    base = opening(guarded(measured()[2], w, sides=[measured()[2]] * 2, per_side=(10, 10), clear=lambda r: 1.0))
    assert base["accepted"] and all(base["guards"].values())


def test_a_nested_opening_needs_an_accepted_parent():
    # a window inside a recessed panel; the 3-D lines confirm the window's edges only, so the panel fails G-R3 and the
    # window, whatever its own evidence, is not accepted without it
    w, P, WR = measured(inner=(3.4, 4.6, 2.4, 5.4, 0.25), line_box=(3.4, 4.6, 2.4, 5.4))
    els = guarded(WR, w)
    parent, child = opening(els, 1), opening(els, 2)
    assert not parent["guards"]["G-R3_line_edge"] and not parent["accepted"]
    assert child["guards"]["G-R3_line_edge"] and not child["guards"]["parent_accepted"] and not child["accepted"]
    # with lines on both, both are accepted and the window lies behind the panel's back
    w, P, WR = measured(inner=(3.4, 4.6, 2.4, 5.4, 0.25))
    WR2 = RL.measure_wall(w, P, np.full(len(P), 5), concat_lines(box_lines(w, 3, 5, 2, 6), box_lines(w, 3.4, 4.6, 2.4, 5.4)))
    els = guarded(WR2, w)
    parent, child = opening(els, 1), opening(els, 2)
    assert parent["accepted"] and child["accepted"] and child["parent_el"] == els.index(parent)
    assert child["offset_m"] < parent["offset_m"] - 0.15


def test_relief_mesh_stays_on_its_wall_and_starts_with_its_frame_corners():
    w, P, WR = measured()
    els = guarded(WR, w)
    pos, nrm, uz, tri, rep = RL.relief_mesh(w, 0.0, w.length, els)
    assert len(rep["recesses"]) == 1 and not rep["boxes"]
    rel = pos[:, :2] - w.a
    uu, dd = rel @ w.tangent, rel @ w.normal
    assert uu.min() >= -1e-6 and uu.max() <= w.length + 1e-6
    assert -RL.RELIEF_MAX_DEPTH <= dd.min() and dd.max() <= RL.RELIEF_MAX_DEPTH
    assert abs(dd.min() + 0.30) < 0.05                                        # the back face at the measured depth
    assert np.allclose(pos[:4], [[0, 0, 0], [10, 0, 0], [10, 0, 12], [0, 0, 12]])   # frame corners, no triangle
    assert tri.min() >= 4
    g = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    assert (np.einsum("ij,ij->i", g, nrm[tri[:, 0]]) >= -1e-9).all()            # every triangle faces its normal
    # nothing outside the piece: a piece that misses the opening builds nothing
    assert RL.relief_mesh(w, 6.0, 10.0, els) is None


# ------------------------------------------------------------------ split-half re-triangulation


def ring_cameras(n=8, radius=30.0):
    cams = []
    for k in range(n):
        a = 2 * np.pi * k / n
        cams.append(ts.look_at([radius * np.cos(a), radius * np.sin(a), 2.0], [0, 0, 5], name=f"r{k}"))
    return cams


def project_all(points, cams, noise=0.0, seed=0):
    rng = np.random.default_rng(seed)
    pt, cam, xy = [], [], []
    for c, C in enumerate(cams):
        uv, z = C.project(points)
        ok = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < C.width) & (uv[:, 1] >= 0) & (uv[:, 1] < C.height)
        idx = np.flatnonzero(ok)
        pt.append(idx)
        cam.append(np.full(len(idx), c))
        xy.append(uv[idx] + rng.normal(0, noise, (len(idx), 2)))
    return RL.PointTracks(np.concatenate(pt), np.concatenate(cam), np.concatenate(xy))


def test_split_halves_alternate_by_bearing():
    cams = ring_cameras(8)
    h = RL.split_halves(cams, np.zeros(2))
    assert sorted(h.tolist()) == [0, 0, 0, 0, 1, 1, 1, 1]
    order = np.argsort([np.arctan2(c.center[1], c.center[0]) for c in cams])
    assert all(h[order[i]] != h[order[i + 1]] for i in range(7))


def test_retriangulation_recovers_points_from_one_half_and_refuses_bad_ones():
    cams = ring_cameras(8)
    rng = np.random.default_rng(0)
    X = rng.uniform([-4, -4, 1], [4, 4, 9], (200, 3))
    tr = project_all(X, cams)
    half = RL.split_halves(cams, np.zeros(2))
    for h in (0, 1):
        Y, n = RL.retriangulate(np.arange(len(X)), tr, cams, half == h)
        ok = np.isfinite(Y[:, 0])
        assert ok.mean() > 0.9 and (n[ok] >= 2).all()
        assert np.abs(Y[ok] - X[ok]).max() < 1e-6
        assert (n[ok] <= (half == h).sum()).all()
    # one camera only: nothing; an observation 20 px off: refused
    Y, _ = RL.retriangulate(np.arange(len(X)), tr, cams, np.arange(8) == 0)
    assert not np.isfinite(Y).any()
    bad = RL.PointTracks(tr.point, tr.camera, tr.xy.copy())
    bad.xy[bad.point == 7] += 20.0
    Y, _ = RL.retriangulate(np.arange(len(X)), bad, cams, np.ones(8, bool))
    assert not np.isfinite(Y[7]).any() and np.isfinite(Y[8]).all()


def test_point_tracks_from_a_colmap_model():
    im = lambda xys: SimpleNamespace(points2D=[SimpleNamespace(xy=np.array(v, float)) for v in xys])
    el = lambda i, j: SimpleNamespace(image_id=i, point2D_idx=j)
    rec = SimpleNamespace(images={10: im([(1, 2), (3, 4)]), 11: im([(5, 6)]), 12: im([(7, 8)])},
                          points3D={100: SimpleNamespace(track=SimpleNamespace(elements=[el(10, 1), el(11, 0)])),
                                    101: SimpleNamespace(track=SimpleNamespace(elements=[el(10, 0), el(12, 0)]))})
    tr = RL.PointTracks.from_colmap(rec, {10: 0, 11: 1})          # image 12 is not a scene camera
    assert tr.point.tolist() == [0, 0, 1] and tr.camera.tolist() == [0, 1, 0]
    assert tr.xy.tolist() == [[3, 4], [5, 6], [1, 2]]


# ------------------------------------------------------------------ end to end: stylize with relief, and the QA

OPEN = (6.0, 10.0, 2.0, 6.0)          # (u0, u1, z0, z1) of a recessed panel on the south wall (u from x = -10)


def relief_scene(depth=0.30, tracks=True, lines=True, num_cams=14) -> st.SceneInput:
    """The synthetic south facade, photographed by num_cams cameras, with dense points (0.1 m) and a recessed
    panel; its points' observations (exact projections) and 3-D lines at the panel's edges."""
    scene = ts.south_scene(num_cams)
    gx, gz = np.meshgrid(np.arange(-9.95, 10, 0.1), np.arange(0.05, ts.WALL_H, 0.1))
    x, z = gx.ravel(), gz.ravel()
    y = np.full(x.size, -6.0)
    u = x + 10.0
    y[(u > OPEN[0]) & (u < OPEN[1]) & (z > OPEN[2]) & (z < OPEN[3])] += depth
    wall = np.stack([x, y, z], 1)
    rx, ry = np.meshgrid(np.arange(-9.5, 10, 1.0), np.arange(-5.5, 6, 1.0))
    roof = np.stack([rx.ravel(), ry.ravel(), np.full(rx.size, ts.WALL_H)], 1)
    scene.points = np.vstack([wall, roof])
    scene.colors = np.vstack([ts.facade_colour(x, z), np.tile([[200, 90, 70]], (len(roof), 1))]).astype(np.uint8)
    if tracks:
        scene.point_tracks = project_all(scene.points, scene.cameras)
        n = np.bincount(scene.point_tracks.point, minlength=len(scene.points))
        scene.point_quality = np.stack([n, np.full(len(n), 20.0)], 1).astype(float)
    if lines:
        a, b, c, d = OPEN
        segs = [(a, c, a, d), (b, c, b, d), (a, c, b, c), (a, d, b, d)]
        A = np.array([[s[0] - 10.0, -6.0, s[1]] for s in segs])
        B = np.array([[s[2] - 10.0, -6.0, s[3]] for s in segs])
        scene.lines3d = Lines3D(A, B, np.full(4, num_cams), [list(range(num_cams))] * 4, np.ones(4), np.zeros(4))
    return scene


@pytest.fixture(scope="module")
def relief_model() -> st.StylizedModel:
    return st.stylize_landmark(relief_scene())


def relief_node(model) -> st.Part:
    return next(p for p in model.parts if p.extras.get("relief"))


def test_a_measured_recess_is_built_on_the_photographed_wall_with_its_record(relief_model):
    s = relief_model.summary
    assert s["qa"]["passed"], s["qa"]
    r = s["relief"]
    assert r["walls_measured"] == 1 and r["accepted"] >= 1 and r["built"] == 1, r
    node = relief_node(relief_model)
    assert node.extras["wall"] == ts.SOUTH and node.extras["observed"] and node.texture is not None
    rec = next(x for x in node.extras["relief"] if x["built"])
    assert rec["provenance"] == "photo-measured" and rec["type"] == "opening"
    assert abs(rec["u0"] - OPEN[0]) < 0.2 and abs(rec["u1"] - OPEN[1]) < 0.2 and abs(rec["offset_m"] + 0.30) < 0.05
    assert min(rec["photos_per_half"]) >= 6 and rec["channels"]["lines3d"]["edges_confirmed"] >= 2
    assert all(rec["guards"].values()) and rec["line_photos"]
    # the geometry: a back face 0.3 m behind the wall face, only on this wall
    wall = relief_model.massing.walls[ts.SOUTH]
    d = (node.positions[:, :2] - wall.a) @ wall.normal
    assert abs(d.min() + 0.30) < 0.05 and d.max() <= 1e-6
    assert not any(p.extras.get("relief") for p in relief_model.parts if p.extras.get("wall") != ts.SOUTH)
    qa = s["qa"]
    assert qa["relief_unbacked"] == 0 and qa["audit"]["relief"]["elements_built"] == 1
    assert qa["glb"]["relief"]["nodes"] == 1 and not qa["glb"]["relief"]["violations"]
    gltf, _ = st.read_glb(relief_model.glb)
    assert gltf["scenes"][0]["extras"]["relief"]["built"] == 1


def test_without_tracks_or_lines_no_relief_is_built():
    m = st.stylize_landmark(relief_scene(tracks=False))
    assert m.summary["relief"]["built"] == 0 and "no point tracks" in m.summary["relief"]["walls"][str(ts.SOUTH)]["refused"]
    assert m.summary["qa"]["passed"]
    m = st.stylize_landmark(relief_scene(lines=False))
    assert m.summary["relief"]["accepted"] == 0 and m.summary["relief"]["built"] == 0 and m.summary["qa"]["passed"]


def _checks(model, parts, orthos=None):
    """(check_honesty passed, audit_model passed, audit_glb passed) for mutated parts of the relief model."""
    scene = relief_scene()
    h = st.check_honesty(parts, model.facades)["passed"]
    a = st.audit_model(model.massing, parts, model.facades, orthos or model.orthophotos, model.observations,
                       np.asarray(scene.points, float), [ts.BOX.copy()], scene.cameras, lines=scene.lines3d)
    g = st.audit_glb(st.write_glb(parts, {}))["passed"]
    return h, a["passed"], g


def _mutate(model, how):
    parts = [copy.deepcopy(p) if p.extras.get("relief") else p for p in model.parts]
    node = next(p for p in parts if p.extras.get("relief"))
    rec = next(r for r in node.extras["relief"] if r["built"])
    wall = model.massing.walls[node.extras["wall"]]
    orthos = None
    if how == "relief on an unphotographed wall":
        k = next(i for i, p in enumerate(parts) if p.extras.get("kind") == "wall" and not p.extras.get("observed"))
        parts[k] = copy.deepcopy(parts[k])
        parts[k].extras["relief"] = copy.deepcopy(node.extras["relief"])
    elif how == "a failed guard":
        rec["guards"]["G-R5_observed_texels"] = False
    elif how == "a provenance other than photo-measured":
        rec["provenance"] = "inferred"
    elif how == "geometry pushed 3 m off the wall":          # the relief, not the frame corners
        node.positions = node.positions.copy()
        node.positions[4:] -= 3.0 * np.r_[wall.normal, 0.0][None]
    elif how == "the whole node moved 3 m off its wall":
        node.positions = node.positions - 3.0 * np.r_[wall.normal, 0.0][None]
    elif how == "an element over unobserved texels":
        orthos = {k: copy.deepcopy(o) for k, o in model.orthophotos.items()}
        orthos[node.extras["wall"]].mask[:] = False
    elif how == "relief geometry without its records":
        node.extras = {k: v for k, v in node.extras.items() if k not in ("relief", "relief_skipped", "piece_u")}
    elif how == "relief geometry copied onto an unphotographed wall, no records":
        k = next(i for i, p in enumerate(parts) if p.extras.get("kind") == "wall" and not p.extras.get("observed")
                 and model.massing.walls[p.extras["wall"]].length >= 11)
        w2 = model.massing.walls[parts[k].extras["wall"]]
        parts[k] = copy.deepcopy(parts[k])
        pos, nrm, uz, tri, _ = RL.relief_mesh(w2, 0.0, w2.length, records_as_elements(node.extras["relief"]))
        parts[k].positions, parts[k].normals, parts[k].indices = pos, nrm, tri
    elif how == "a record id naming another wall":
        node.extras["relief"] = [dict(r, id=r["id"].replace(f"w{wall.index}-", f"w{wall.index + 1}-"))
                                 for r in node.extras["relief"]]
    elif how == "a built record with no geometry behind it":
        node.extras["relief"] = node.extras["relief"] + [dict(rec, id=rec["id"] + "9", u0=0.5, u1=2.5)]
    elif how == "recesses deepened 2.5 times":
        node.positions = node.positions.copy()
        d = (node.positions[:, :2] - wall.a) @ wall.normal
        node.positions[:, :2] += (1.5 * d)[:, None] * wall.normal[None]
    elif how == "relief geometry shifted 1.5 m along the wall":
        node.positions = node.positions.copy()
        d = (node.positions[:, :2] - wall.a) @ wall.normal
        off = np.abs(d) > 1e-6
        node.positions[off, :2] -= 1.5 * wall.tangent[None]
    elif how == "a rejected element built as geometry":
        els = records_as_elements(node.extras["relief"]) + [
            {"type": "opening", "kind": "panel", "level": 1, "u0": 1.0, "u1": 3.0, "z_bottom": 2.0, "z_top": 5.0,
             "depth": 0.3, "offset_m": -0.3, "accepted": False, "builds_geometry": True}]
        pos, nrm, uz, tri, _ = RL.relief_mesh(wall, 0.0, wall.length, els)
        node.positions, node.normals, node.indices = pos, nrm, tri
        node.uvs = np.c_[uz[:, 0] / wall.length, (wall.z1 - uz[:, 1]) / (wall.z1 - wall.z0)]
    return parts, orthos


def records_as_elements(recs):
    """Relief records back as elements relief_mesh can build."""
    out = []
    for r in recs:
        e = dict(r, builds_geometry=bool(r.get("built")), level=r.get("nesting"))
        e.setdefault("depth", r.get("level", r.get("projection")))
        out.append(e)
    return out


@pytest.mark.parametrize("how, caught_by", [
    ("relief on an unphotographed wall", (0, 1, 2)),
    ("a failed guard", (0, 2)),
    ("a provenance other than photo-measured", (0, 2)),
    ("geometry pushed 3 m off the wall", (1, 2)),
    ("the whole node moved 3 m off its wall", (1,)),
    ("an element over unobserved texels", (1,)),
    ("relief geometry without its records", (1, 2)),
    ("relief geometry copied onto an unphotographed wall, no records", (1, 2)),
    ("a record id naming another wall", (0, 1, 2)),
    ("a built record with no geometry behind it", (1, 2)),
    ("recesses deepened 2.5 times", (1, 2)),
    ("relief geometry shifted 1.5 m along the wall", (1, 2)),
    ("a rejected element built as geometry", (1, 2)),
])
def test_the_checks_catch_every_injected_relief_violation(relief_model, how, caught_by):
    assert _checks(relief_model, relief_model.parts) == (True, True, True)       # the untouched model passes
    parts, orthos = _mutate(relief_model, how)
    got = _checks(relief_model, parts, orthos)
    for i in caught_by:
        assert got[i] is False, (how, got)


def test_on_the_osm_frame_the_relief_is_counted_as_photo_measured_structure():
    from test_structure_stylize import box_structure
    scene = relief_scene()
    scene.structure = box_structure("flat")
    m = st.stylize_landmark(scene)
    s = m.summary
    assert s["massing_source"] == "osm-parts" and s["qa"]["passed"], s["qa"]
    assert s["relief"]["built"] == 1
    sm = s["structure_metrics"]["metrics"]
    assert sm["struct_parts"] == 1 and sm["struct_facade_elements"] == 1 and sm["struct_unbacked"] == 0
    assert sm["struct_photo_dof"] >= 3 and sm["struct_share_photo_added_elements"] > 0
    # the frame's numbers are OSM's (outline, height) or the S3DB defaults of the untagged min_height / roof_height
    # (a convention, like a defaulted roof shape): none from the photos
    pr = s["structure_metrics"]["shares"]["params"]
    assert pr["osm"] == 0.5 and pr["convention"] == 0.5 and sum(pr.values()) == pytest.approx(1.0)
    ex = st.read_glb(m.glb)[0]["scenes"][0]["extras"]
    assert ex["structure_metrics"]["metrics"] == sm and ex["structure_metrics"]["unbacked"] == 0


def test_a_measured_rhythm_is_recorded_as_a_possible_decision_and_never_copied():
    w, P, WR = measured()
    wall2 = st.Wall(1, 0, 0, np.array([10.0, 0.0]), np.array([10.0, 10.0]), 0.0, 12.0, st.PROV_OSM, st.PROV_OSM_HEIGHT)
    els = guarded(WR, w)
    e2 = copy.deepcopy(opening(els))
    e2["u0"], e2["u1"] = 6.0, 8.05
    res = RL.ReliefResult(elements={0: els + [e2]})
    dec = RL.rhythm_decisions(res, SimpleNamespace(walls=[w, wall2]))
    assert len(dec) == 1 and dec[0]["measured_on"] == 0 and dec[0]["candidates"] == [1]
    assert dec[0]["status"].startswith("not done")
    assert 1 not in res.elements                           # nothing was put on the other wall


def test_without_given_3d_lines_the_structure_path_computes_them_once_a_wall_can_carry_relief():
    from test_structure_stylize import box_structure
    scene = relief_scene(lines=False)
    scene.structure = box_structure("flat")
    m = st.stylize_landmark(scene)
    st_ = m.summary["relief"]["stats"]
    assert "lines_s" in st_ and st_["lines"] >= 0                 # crowd3d.lines3d ran inside stylize
    assert m.summary["qa"]["passed"], m.summary["qa"]
    for r in (rec for p in m.parts for rec in (p.extras.get("relief") or [])):
        assert all(r["guards"].values()) and r["channels"]["lines3d"]["edges_confirmed"] >= 1


# ------------------------------------------------------------------ geometry against records, independent depth check


def test_relief_plan_builds_only_elements_that_build_geometry():
    w, P, WR = measured()
    els = guarded(WR, w)
    rejected = dict(opening(els), u0=6.5, u1=8.5, accepted=False, builds_geometry=False, offset_m=-0.3)
    rec, boxes, skipped = RL.relief_plan(w, 0.0, w.length, els + [rejected])
    assert [r["el"] for r in rec] == [els.index(opening(els))] and not boxes
    pos, nrm, uz, tri, rep = RL.relief_mesh(w, 0.0, w.length, els + [rejected])
    rel = pos[:, :2] - w.a
    recs = [RL.element_record(e, True) for e in els if e["builds_geometry"]]
    assert RL.geometry_violations(rel @ w.tangent, rel @ w.normal, pos[:, 2], tri, recs) == []
    # the same mesh with the rejected one built: vertices no record explains
    forced = dict(rejected, builds_geometry=True)
    pos, nrm, uz, tri, rep = RL.relief_mesh(w, 0.0, w.length, els + [forced])
    rel = pos[:, :2] - w.a
    v = RL.geometry_violations(rel @ w.tangent, rel @ w.normal, pos[:, 2], tri, recs)
    assert v and "no built record explains" in v[0]


def test_a_shallow_claim_on_a_flat_wall_fails_the_independent_depth_check():
    rng = np.random.default_rng(4)
    w, P = synthetic_wall(rng, recess=(3, 5, 2, 6, 0.30), band=(9, 9.6, 0.0), noise=0.02)
    U = RL.to_wall(w.a, w.tangent, w.normal, P)
    fr = {"c0": 0.0, "c1": 0.0}
    claim = lambda d, u0=6.5, u1=8.5: {"type": "opening", "u0": u0, "u1": u1, "z_bottom": 2.0, "z_top": 8.0,
                                       "depth": d, "offset_m": -d, "frame": fr}
    for d in (0.15, 0.10, 0.27):                                  # on the flat part of the wall
        chk = RL.audit_points(claim(d), w, P, None, U=U)
        assert chk["ok"] is False and abs(chk["measured_m"]) < 0.02, (d, chk)
    real = {"type": "opening", "u0": 3.0, "u1": 5.0, "z_bottom": 2.0, "z_top": 6.0, "depth": 0.3, "offset_m": -0.3, "frame": fr}
    assert RL.audit_points(real, w, P, None, U=U)["ok"] is True
    assert RL.audit_points(dict(real, depth=0.5, offset_m=-0.5), w, P, None, U=U)["ok"] is False     # 0.2 m off
    # a genuinely shallow recess (0.12 m) is confirmed when the points show it
    w, P = synthetic_wall(rng, recess=(3, 5, 2, 6, 0.12), band=(9, 9.6, 0.0), noise=0.02)
    U = RL.to_wall(w.a, w.tangent, w.normal, P)
    assert RL.audit_points(dict(real, depth=0.12, offset_m=-0.12), w, P, None, U=U)["ok"] is True
    # too few points: unverifiable
    assert RL.audit_points(real, w, P[:30], None)["ok"] is None


def test_the_pipeline_refuses_what_the_points_do_not_confirm():
    """G-R6: every element is re-measured from the raw points before it is accepted, so the publication audit (which
    repeats the check on every record) never meets a record it would reject."""
    w, P, WR = measured()
    U = RL.to_wall(w.a, w.tangent, w.normal, P)
    els = RL.guard_elements(WR, [WR, WR], [10, 10], np.ones((120, 100), bool), w,
                            recheck=lambda r: RL.audit_points(r, w, P, None, U=U))
    e = opening(els)
    assert e["guards"]["G-R6_points_recheck"] and e["accepted"] and e["recheck"]["ok"] is True
    # the same detection on points that do not show the recess (the claim is the detector's, the points disagree)
    rng = np.random.default_rng(9)
    w2, P2 = synthetic_wall(rng, recess=(3, 5, 2, 6, 0.0), band=(9, 9.6, 0.0))
    U2 = RL.to_wall(w2.a, w2.tangent, w2.normal, P2)
    els = RL.guard_elements(WR, [WR, WR], [10, 10], np.ones((120, 100), bool), w,
                            recheck=lambda r: RL.audit_points(r, w2, P2, None, U=U2))
    e = opening(els)
    assert not e["guards"]["G-R6_points_recheck"] and not e["accepted"] and not e["builds_geometry"]
    rec = RL.element_record(dict(e, guards={k: True for k in e["guards"]}), True)
    assert rec["recheck"]["measured_m"] is not None


def test_a_record_on_another_wall_than_its_id_names_is_caught():
    recs = [{"id": "w2-o7", "type": "opening", "u0": 1.0, "u1": 3.0, "z_bottom": 0.1, "z_top": 9.0, "depth": 0.27,
             "offset_m": -0.27}]
    assert RL.copied_records([("wall-2-0", 2, recs), ("wall-2-1", 2, recs)]) == []      # two pieces of one wall
    v = RL.copied_records([("wall-1", 1, recs)])
    assert len(v) == 1 and "does not name wall 1" in v[0]["why"]


def test_relief_is_never_built_on_an_unphotographed_piece(relief_model):
    """build_parts draws relief only on the observed strips of a wall; with no observed facade the relief
    elements of that wall build nothing and every wall node stays on its plane."""
    m = relief_model
    node = relief_node(m)
    els = {node.extras["wall"]: records_as_elements(node.extras["relief"])}
    parts = st.build_parts(m.massing, {}, None, st.DEFAULT_PARAMS, m.observations, {}, relief=els)
    assert not any(p.extras.get("relief") for p in parts)
    for p in parts:
        if p.extras.get("kind") == "wall":
            w = m.massing.walls[p.extras["wall"]]
            assert np.abs((p.positions[:, :2] - w.a) @ w.normal).max() < 1e-9


def test_a_relief_node_must_be_observed_even_when_textured():
    """node_violations checks 'observed' on its own (a textured node marked unobserved is refused)."""
    ex = {"kind": "wall", "observed": True, "provenance": {"appearance": "photo-observed"}}
    assert RL.node_violations(ex, True) == []
    assert RL.node_violations(dict(ex, observed=False), True)
    assert RL.node_violations(ex, False)
    assert RL.node_violations(dict(ex, provenance={"appearance": "default-style"}), True)
    assert RL.node_violations(ex, True, {"p1"}) == [] and RL.node_violations(dict(ex, part="p1"), True, {"p1"})


def test_a_wall_node_without_records_must_be_planar(relief_model):
    parts = [copy.deepcopy(p) if p.extras.get("kind") == "wall" else p for p in relief_model.parts]
    k = next(i for i, p in enumerate(parts) if p.extras.get("kind") == "wall" and not p.extras.get("relief"))
    w = relief_model.massing.walls[parts[k].extras["wall"]]
    parts[k].positions = parts[k].positions.copy()
    parts[k].positions[0, :2] += 0.05 * w.normal                     # one corner 5 cm off the wall plane
    assert _checks(relief_model, parts)[1:] == (False, False)


# ------------------------------------------------------------------ objects in front of the facade fail safe


def wall_with_pole(rng, pole=(3.8, 4.2, 1.0, 9.0), recess=(2.0, 6.0, 1.0, 8.0, 0.30), buttress=None, behind=True):
    """A 4 m wide recess and a pole (u0, u1, standoff, height) in front of it: the pole's points (dense, its front
    half) and, when `behind`, the wall still seen behind it from oblique photos. buttress: (u0, u1, projection) of
    an ATTACHED pier: its front face and no wall points behind it."""
    w, P = synthetic_wall(rng, recess=recess, band=(9, 9.6, 0.0))
    u = P[:, 0]
    if not behind:
        P = P[~((u > pole[0]) & (u < pole[1]) & (P[:, 2] < pole[3]))]
    if buttress is not None:
        b0, b1, pb = buttress
        P = P[~((P[:, 0] > b0) & (P[:, 0] < b1))]
        n = int((b1 - b0) * 12 * 400)
        bu, bz = rng.uniform(b0, b1, n), rng.uniform(0, 12, n)
        P = np.vstack([P, np.stack([bu, -(pb + rng.normal(0, 0.01, n)), bz], 1)])
    if pole is not None:
        p0, p1, sd, ph = pole
        n = int((p1 - p0) * ph * 1000)
        pu, pz = rng.uniform(p0, p1, n), rng.uniform(0, ph, n)
        r = 0.5 * (p1 - p0)
        dd = sd + r - np.sqrt(np.maximum(r * r - (pu - 0.5 * (p0 + p1)) ** 2, 0.0))   # its front half
        P = np.vstack([P, np.stack([pu, -dd, pz], 1)])
    return w, P


def test_a_pole_in_front_of_a_recess_is_dropped_and_does_not_split_it():
    rng = np.random.default_rng(11)
    w, P = wall_with_pole(rng)
    wr = RL.measure_wall(w, P, np.full(len(P), 5), None)
    assert not wr.refused, wr.refused
    occ = wr.stats["occluders"]
    assert len(occ) == 1 and occ[0]["u0"] <= 3.85 and occ[0]["u1"] >= 4.15 and occ[0]["front_m"] > 0.9
    assert not [v for v in wr.verticals if v["u1"] > 3.5 and v["u0"] < 4.5]          # no pier where the pole stands
    ops = [o for o in wr.openings if o["level"] == 1]
    assert len(ops) == 1 and abs(ops[0]["u0"] - 2.0) < 0.15 and abs(ops[0]["u1"] - 6.0) < 0.15    # one recess, whole
    assert abs(ops[0]["depth"] - 0.30) < 0.05
    # without the drop, the same points split the recess at the pole
    wr0 = RL._measure_wall_once(w, P, np.full(len(P), 5), None)
    assert len([o for o in wr0.openings if o["level"] == 1]) >= 2 or wr0.verticals


def test_an_attached_buttress_stays_a_buttress():
    rng = np.random.default_rng(12)
    w, P = wall_with_pole(rng, pole=None, buttress=(7.0, 7.6, 0.6))
    wr = RL.measure_wall(w, P, np.full(len(P), 5), None)
    assert wr.stats["occluders"] == []
    v = [v for v in wr.verticals if v["how"] == "proud" and v["u0"] < 7.3 < v["u1"]]
    assert len(v) == 1 and v[0]["kind"] == "buttress" and abs(v[0]["projection"] - 0.6) < 0.1


def test_openings_are_cut_only_at_the_given_verticals():
    """measure_relief re-measures a wall with cut_boxes = its ACCEPTED verticals: a recess is split into bays only
    there, so a rejected strip (an object in front, an unconfirmed strip) cannot make up bays."""
    rng = np.random.default_rng(13)
    w, P = synthetic_wall(rng, recess=(2.0, 6.0, 1.0, 8.0, 0.30), band=(9, 9.6, 0.0))
    one = RL.measure_wall(w, P, np.full(len(P), 5), None, cut_boxes=[])
    ops = [o for o in one.openings if o["level"] == 1]
    assert len(ops) == 1 and abs(ops[0]["u0"] - 2.0) < 0.15 and abs(ops[0]["u1"] - 6.0) < 0.15
    two = RL.measure_wall(w, P, np.full(len(P), 5), None, cut_boxes=[(3.9, 4.3, 0.0, 12.0)])
    ops = sorted((o for o in two.openings if o["level"] == 1), key=lambda o: o["u0"])
    assert len(ops) == 2 and ops[0]["u1"] <= 4.0 and ops[1]["u0"] >= 4.2
    assert two.registers and all(r["cutters"] == [[3.9, 4.3]] for r in two.registers if r["z_top"] - r["z_bottom"] > 6)


def test_g_r7_refuses_an_element_with_an_edge_at_a_free_standing_object():
    occ = [{"u0": 3.8, "u1": 4.2, "z_bottom": 0.0, "z_top": 9.0}]
    e = lambda u0, u1, **k: {"type": "opening", "u0": u0, "u1": u1, "z_bottom": 1.0, "z_top": 8.0,
                             "edges": {"left": {"limit": False}, "right": {"limit": False}}, **k}
    assert RL.clear_of_objects(e(2.0, 6.0), occ)                        # spans it, edges far from it
    assert not RL.clear_of_objects(e(2.0, 3.7), occ)                    # an edge at its silhouette
    assert not RL.clear_of_objects(e(4.3, 6.0), occ)
    assert not RL.clear_of_objects(e(3.85, 4.15), occ)                  # inside it
    assert RL.clear_of_objects(e(2.0, 3.7, z_bottom=9.5, z_top=11.0), occ)     # above it
    assert RL.clear_of_objects({"type": "band", "u0": 0, "u1": 10, "z_bottom": 3, "z_top": 4}, occ)


def test_g_r5_clear_view_counts_things_in_front_of_the_element_at_any_distance():
    """G-R5's second half: the share of the element's surface that the photos mostly see unobstructed. A dense
    screen 1 m in front of a recess (closer than the facade's 1.5 m occluder margin) hides it; a 0.4 m pole hides
    a strip of it; nothing in front hides nothing."""
    w = st.Wall(0, 0, 0, np.array([0.0, 0.0]), np.array([10.0, 0.0]), 0.0, 12.0, st.PROV_OSM, st.PROV_OSM_HEIGHT)
    cams = [ts.look_at([x, -20.0, 2.0], [x, 0.0, 5.0], w=1600, h=1200, f=1400.0, name=f"c{k}")
            for k, x in enumerate(np.linspace(-6, 16, 9))]
    r = {"type": "opening", "u0": 3.0, "u1": 5.0, "z_bottom": 2.0, "z_top": 6.0, "depth": 0.3, "offset_m": -0.3,
         "frame": {"c0": 0.0, "c1": 0.0}}
    assert RL.clear_view_share(r, w, np.zeros((0, 3)), cams) == 1.0
    gx, gz = np.meshgrid(np.arange(2.5, 5.5, 0.05), np.arange(1.5, 6.5, 0.05))
    screen = np.stack([gx.ravel(), np.full(gx.size, -1.0), gz.ravel()], 1)
    assert RL.clear_view_share(r, w, screen, cams) < 0.2
    gx, gz = np.meshgrid(np.arange(3.8, 4.2, 0.03), np.arange(0.0, 9.0, 0.03))
    pole = np.stack([gx.ravel(), np.full(gx.size, -1.0), gz.ravel()], 1)
    assert 0.3 <= RL.clear_view_share(r, w, pole, cams) < 0.9                      # a strip of it


def test_the_photo_halves_are_disjoint_and_an_element_one_half_never_saw_is_refused(relief_model):
    """G-R2 really is two photo sets: the halves split the cameras without overlap, and a recess whose points only
    one half observed is not accepted."""
    scene = relief_scene()
    half = RL.split_halves(scene.cameras, np.mean(np.concatenate([np.stack([w.a, w.b]) for w in relief_model.massing.walls]), 0))
    assert set(np.flatnonzero(half == 0)).isdisjoint(np.flatnonzero(half == 1)) and min(np.bincount(half)) >= 6
    tr = scene.point_tracks
    P = np.asarray(scene.points)
    u, z = P[:, 0] + 10.0, P[:, 2]
    a, b, c, d = OPEN
    inside = (u > a - 0.3) & (u < b + 0.3) & (z > c - 0.3) & (z < d + 0.3) & (P[:, 1] < -5.5)
    keep = ~(inside[tr.point] & (half[tr.camera] == 1))            # half 1 never observed the recess's points
    scene.point_tracks = RL.PointTracks(tr.point[keep], tr.camera[keep], tr.xy[keep])
    m = st.stylize_landmark(scene)
    assert m.summary["relief"]["accepted"] == 0 or all(
        not (e["type"] == "opening") for p in m.parts for e in (p.extras.get("relief") or []))
    assert m.summary["relief"]["built"] == 0 and m.summary["qa"]["passed"]


def test_depth_gap_tells_a_pole_from_a_pier_with_side_faces():
    rng = np.random.default_rng(14)
    pole = np.r_[rng.normal(0.0, 0.02, 500), rng.uniform(0.9, 1.2, 1500)]          # wall behind, pole 0.9 m out
    pier = np.r_[rng.normal(0.0, 0.02, 500), rng.uniform(0.0, 1.2, 1500)]          # side faces from wall to front
    assert RL.depth_gap(pole, 0.1, 1.1) >= 0.7
    assert RL.depth_gap(pier, 0.1, 1.1) < 0.1
    assert RL.depth_gap(pier, 0.5, 0.5) == 0.0


# ------------------------------------------------------------------ G-R1 frame, G-R6 ring, G-R3 chance


def test_the_frame_must_be_the_walls_front_surface():
    """G-R1's one-front-surface test: a sheet (scaffold netting, a neighbour's facade) 0.2 m in front of the middle
    half of a wall would become its frame level -- every depth measured from it -- and is refused; a facade whose frame
    is broken into piers by full-height recessed bays is still one front surface (the frame above the bays joins
    them)."""
    rng = np.random.default_rng(21)
    w, P = synthetic_wall(rng, recess=(3, 5, 2, 6, 0.0), band=(9, 9.6, 0.0))
    u, z = P[:, 0], P[:, 2]
    sheet = (u > 2.5) & (u < 7.5) & (z < 8.5)
    P = P[~sheet]                                              # the sheet hides the wall behind it ...
    n = int(5 * 8.5 * 600)
    su, sz = rng.uniform(2.5, 7.5, n), rng.uniform(0, 8.5, n)
    P = np.vstack([P, np.stack([su, -(0.2 + rng.normal(0, 0.01, n)), sz], 1)])     # ... 0.2 m in front of it
    wf = RL.wall_frame(w, P, np.full(len(P), 5))
    assert "not the wall's front surface" in wf["refused"] and wf["frame"] is None, wf["refused"]
    assert "not the wall's front surface" in RL.measure_wall(w, P, np.full(len(P), 5), None).refused
    assert RL.independent_frame(w, P, np.full(len(P), 5))["refused"]
    # an arcade: three full-height bays 0.3 m deep between piers, the frame above them
    w, P = synthetic_wall(rng, recess=(0.6, 3.0, 0.0, 9.0, 0.30), band=(9, 9.6, 0.0))
    u, z = P[:, 0], P[:, 2]
    for a, b in ((3.6, 6.4), (7.0, 9.4)):
        P[(u > a) & (u < b) & (z < 9.0), 1] += 0.30
    wf = RL.wall_frame(w, P, np.full(len(P), 5))
    assert not wf["refused"] and wf["stats"]["frame_span"] >= 0.9 and abs(wf["frame"]["c0"]) < 0.05, wf["stats"]


def test_the_points_beside_an_element_must_show_its_relief():
    """audit_points (3): a claimed recess on the set-back half of a two-plane wall sits at the claimed offset from the
    frame (the other half) and differs from its base, but has no step to the points right beside it: refused; the
    real recess, with the wall around it at the frame, passes, and its ring is recorded."""
    rng = np.random.default_rng(22)
    w, P = synthetic_wall(rng, recess=(6.5, 8.5, 2.0, 6.0, 0.30), band=(9, 9.6, 0.0))
    P[P[:, 0] < 5.0, 1] += 0.30                                 # u < 5: the whole half 0.3 m behind the frame
    U = RL.to_wall(w.a, w.tangent, w.normal, P)
    fr = {"c0": 0.0, "c1": 0.0}
    fake = {"type": "opening", "u0": 1.0, "u1": 3.0, "z_bottom": 2.0, "z_top": 6.0, "depth": 0.3, "offset_m": -0.3,
            "frame": fr}
    chk = RL.audit_points(fake, w, P, None, U=U)
    assert chk["ok"] is False and abs(chk["measured_m"] + 0.3) < 0.03 and "beside it" in chk["why"], chk
    assert all(abs(x) < 0.03 for x in chk["ring_m"])
    real = dict(fake, u0=6.5, u1=8.5)
    chk = RL.audit_points(real, w, P, None, U=U)
    assert chk["ok"] is True and all(abs(x + 0.3) < 0.03 for x in chk["ring_m"]), chk
    # no point beside it: unverifiable, never passed
    keep = ~(((U[:, 0] > 6.0) & (U[:, 0] < 6.5)) | ((U[:, 0] > 8.5) & (U[:, 0] < 9.0)))
    assert RL.audit_points(real, w, P[keep], None)["ok"] is None


def test_a_clean_narrow_pilaster_with_lines_at_its_edges_passes_g_r3():
    """G-R3's chance rate must not count the element's own edge clusters as chance. A 0.6 m pilaster
    (0.3 m proud) whose only 3-D lines are its two edges: the background rate (within 2 m of it, away from its own
    edges) is 0, so both confirmed edges are far above chance and G-R3 passes; with the same clusters repeated every
    0.6 m along the wall (a lined facade) the rate is high and it fails."""
    rng = np.random.default_rng(23)
    w, P = synthetic_wall(rng, recess=(3, 5, 2, 6, 0.0), band=(9, 9.6, 0.0))
    strip = (P[:, 0] > 4.7) & (P[:, 0] < 5.3)
    P[strip, 1] -= 0.30                                          # proud (outward = -y)
    L = lines_on(w, [(4.7, 0.2, 4.7, 11.8), (5.3, 0.2, 5.3, 11.8)])
    WR = RL.measure_wall(w, P, np.full(len(P), 5), L)
    v = next(v for v in WR.verticals if v["u0"] < 5.0 < v["u1"])
    assert v["how"] == "proud" and abs(v["width"] - 0.6) < 0.15 and v["edges_confirmed"] == 2
    assert v["edge_chance"] == [0.0, 0.0] and v["line_p_chance"] <= 0.1
    e = next(e for e in guarded(WR, w) if e["type"] == "vertical" and e["u0"] < 5.0 < e["u1"])
    assert e["guards"]["G-R3_line_edge"], e
    # a facade lined every 0.6 m: the pilaster's edges are no better than chance
    lined = lines_on(w, [(x, 0.2, x, 11.8) for x in np.arange(0.5, 9.6, 0.6)] + [(4.7, 0.2, 4.7, 11.8), (5.3, 0.2, 5.3, 11.8)])
    WR = RL.measure_wall(w, P, np.full(len(P), 5), lined)
    v = next(v for v in WR.verticals if v["u0"] < 5.0 < v["u1"])
    assert min(v["edge_chance"]) > 0.3 and v["line_p_chance"] > 0.1
    assert not next(e for e in guarded(WR, w) if e["type"] == "vertical" and e["u0"] < 5.0 < e["u1"])["guards"]["G-R3_line_edge"]


def test_g_r3_chance_leaves_out_the_elements_own_edges():
    cl = [{"pos": 4.7}, {"pos": 5.3}]
    assert RL.chance_rate(cl, 2.7, 7.3, 0.2) > 0.1                       # own clusters counted as chance
    assert RL.chance_rate(cl, 2.7, 7.3, 0.2, exclude=(4.7, 5.3)) == 0.0
    assert RL.chance_rate(cl + [{"pos": 3.0}], 2.7, 7.3, 0.2, exclude=(4.7, 5.3)) > 0.05
    assert RL.chance_rate([], 0.0, 0.3, 0.2, exclude=(0.1, 0.2)) == 1.0   # nothing left to sample: no low rate shown


def test_a_bar_standing_in_front_of_the_wall_across_its_width_is_dropped_as_an_object():
    """free_standing_strips catches what standing_objects (narrow vertical objects) cannot: a horizontal bar (a
    cable tray, a banner) 0.6 m in front of the whole width, the wall seen behind it from oblique photos, is not a
    projecting band."""
    rng = np.random.default_rng(32)
    w, P = synthetic_wall(rng, recess=(3, 5, 2, 6, 0.0), band=(9, 9.6, 0.0))
    n = int(10 * 0.6 * 700)                                  # the bar's points outnumber the wall's seen behind it
    bu, bz = rng.uniform(0, 10, n), rng.uniform(5.0, 5.6, n)
    P = np.vstack([P, np.stack([bu, -(0.6 + rng.normal(0, 0.01, n)), bz], 1)])
    wr = RL.measure_wall(w, P, np.full(len(P), 5), None)
    assert not wr.refused, wr.refused
    occ = wr.stats["occluders"]
    assert any(o["type"] == "band" and o["z_bottom"] < 5.3 < o["z_top"] for o in occ), occ
    assert not [b for b in wr.bands if b["sign"] > 0 and b["z_bottom"] < 5.3 < b["z_top"]], wr.bands


def test_the_viewpoint_sides_are_disjoint_and_cut_bays_only_at_confirmed_verticals(relief_model):
    """The recess and a 0.3 m pilaster beside it are observed only by the photos on ONE side of
    the wall's middle (side_halves): the interleaved halves both still find the recess, but the other viewpoint side
    re-triangulates nothing there, so G-R2 refuses both (the sides really are two disjoint photo sets), and the
    pilaster, unconfirmed, is not a cutter: the wall is re-measured with the accepted verticals only."""
    m = relief_model
    scene = relief_scene()
    P = np.asarray(scene.points, float).copy()
    u, z = P[:, 0] + 10.0, P[:, 2]
    wall_pt = P[:, 1] > -6.5
    pil = wall_pt & (u > 2.0) & (u < 2.4) & (z < ts.WALL_H)
    P[pil, 1] -= 0.30                                                       # proud (outward = -y)
    scene.points = P
    tr = project_all(P, scene.cameras)
    wall = m.massing.walls[ts.SOUTH]
    seen = [ci for ci, frac, _ in m.observations[ts.SOUTH].cameras if frac >= RL.DEFAULT_GUARDS.photo_wall_frac]
    side, per_side = RL.side_halves(scene.cameras, wall, seen)
    assert min(per_side) >= RL.DEFAULT_GUARDS.min_photos_per_half
    a, b, c, d = OPEN
    zone = wall_pt & (((u > a - 0.3) & (u < b + 0.3) & (z > c - 0.3) & (z < d + 0.3)) | ((u > 1.7) & (u < 2.7)))
    keep = ~(zone[tr.point] & (side[tr.camera] == 1))                      # side 1 never observed them
    scene.point_tracks = RL.PointTracks(tr.point[keep], tr.camera[keep], tr.xy[keep])
    n = np.bincount(scene.point_tracks.point, minlength=len(P))
    scene.point_quality = np.stack([n, np.full(len(n), 20.0)], 1).astype(float)
    res = {}
    orig = st.measure_facade_relief

    def capture(*a_, **k_):
        res["r"] = orig(*a_, **k_)
        return res["r"]

    st.measure_facade_relief = capture
    try:
        out = st.stylize_landmark(scene)
    finally:
        st.measure_facade_relief = orig
    els = res["r"].elements[ts.SOUTH]
    op = next(e for e in els if e["type"] == "opening" and e["u0"] < 8.0 < e["u1"])
    assert op["split"]["iou_a"] >= 0.3 and op["split"]["iou_b"] >= 0.3                 # both interleaved halves see it
    assert min(op["split_sides"]["iou_a"], op["split_sides"]["iou_b"]) < 0.3             # one viewpoint side does not
    assert not op["guards"]["G-R2_split_half"] and not op["accepted"]
    v = [e for e in els if e["type"] == "vertical" and e["u0"] < 2.2 < e["u1"]]
    assert v and not v[0]["guards"]["G-R2_split_half"], v
    assert res["r"].walls[ts.SOUTH]["recut_at_accepted_verticals"] == 0
    assert out.summary["relief"]["built"] == 0 and out.summary["qa"]["passed"]


# ------------------------------------------------------------------ structure inside a recess (merged bays)


def bay_points(rng, recesses, length=20.0, height=10.0, density=400, noise=0.01):
    """A wall along +x (outward normal (0, -1)), u 0..length, z 0..height, with recessed boxes (u0, u1, z0, z1,
    depth); everything else at the frame plane."""
    w = st.Wall(0, 0, 0, np.array([0.0, 0.0]), np.array([length, 0.0]), 0.0, height, st.PROV_OSM, st.PROV_OSM_HEIGHT)
    n = int(length * height * density)
    u, z = rng.uniform(0, length, n), rng.uniform(0, height, n)
    d = np.zeros(n)
    for u0, u1, z0, z1, depth in recesses:
        d[(u > u0) & (u < u1) & (z > z0) & (z < z1)] = -depth
    d += rng.normal(0, noise, n)
    return w, np.stack([u, -d, z], 1)


BAYS = [(4.0, 7.0, 0.5, 5.0, 0.3), (7.6, 10.6, 0.5, 5.0, 0.3), (4.0, 7.0, 6.0, 9.0, 0.3)]   # two bays + a window above


def claim(u0, u1, z0, z1, offset=-0.3, depth=None, **kw):
    return dict({"type": "opening", "kind": "bay", "u0": u0, "u1": u1, "z_bottom": z0, "z_top": z1,
                 "offset_m": offset, "depth": -offset if depth is None else depth, "frame": {"c0": 0.0, "c1": 0.0}}, **kw)


@pytest.mark.parametrize("name, rec, strip", [
    ("one recess over two bays and the pier between them", claim(4.0, 10.6, 0.5, 5.0), "v"),
    ("the same with an arched top (spring below its top)", claim(4.0, 10.6, 0.5, 5.0, arched=True, spring_z=4.0), "v"),
    ("a bay and the window above it over the frame band between them", claim(4.0, 7.0, 0.5, 9.0), "h"),
])
def test_a_recess_claimed_over_structure_fails_the_re_measurement(name, rec, strip):
    """audit_points (4): inside a claimed recess the raw points show a strip at the frame plane (a pier between two
    bays, the frame band between a bay and the window above) across >= 60 % of it: the claim spans real structure.
    The points' median in the box, the ring beside it (frame on both sides) and the frame all agree with the claim, so
    only the interior profile tells."""
    rng = np.random.default_rng(31)
    w, P = bay_points(rng, BAYS)
    U, _ = RL.wall_points(w, P, np.full(len(P), 5))
    chk = RL.audit_points(rec, w, P, None, U=U)
    assert chk["ok"] is False and chk["interior"]["ok"] is False, (name, chk)
    assert chk["interior"][strip], (name, chk)
    assert "structure at the frame plane" in chk["why"], chk
    if strip == "v" and rec["u1"] > 10:
        a, b = chk["interior"]["v"][0]
        assert 6.9 <= a and b <= 7.75 and b - a >= RL.DEFAULT_GUARDS.interior_w_m, chk["interior"]


def test_the_interior_of_an_arch_is_its_outline_as_built_not_the_box_below_its_springing_line():
    """The re-measurement covers an arch's outline AS BUILT (the rectangle up to the springing line and the half circle
    above it, shrunk by 0.15 m; an arch whose springing line is not 0.1 m above its sill is built, and re-measured, as
    a rectangle): an arch over both bays and the pier, springing 0.3 m above its sill, is refused on the pier."""
    r = claim(4.0, 10.6, 0.5, 5.0, arched=True, spring_z=0.8)
    bu0, bu1, bz0, bz1 = RL.interior_box(r)
    assert (bu0, bu1, bz0) == (4.15, 10.45, 0.65) and abs(bz1 - (0.8 + 3.3 - 0.15)) < 1e-9, RL.interior_box(r)
    inside = RL.in_interior(r, np.array([7.3, 4.3, 10.3, 7.3]), np.array([3.9, 3.0, 0.7, 4.0]))
    assert inside.tolist() == [True, False, True, False], inside            # the pier at the crown in; shoulders out
    flat = claim(4.0, 10.6, 0.5, 5.0, arched=True, spring_z=0.55)           # built as a rectangle (relief_plan)
    assert RL.outline_arch(flat) is None and RL.interior_box(flat)[3] == 4.85
    rng = np.random.default_rng(31)
    w, P = bay_points(rng, BAYS)
    U, _ = RL.wall_points(w, P, np.full(len(P), 5))
    chk = RL.audit_points(r, w, P, None, U=U)
    assert chk["ok"] is False and chk["interior"]["v"], chk["interior"]
    a, b = chk["interior"]["v"][0]
    assert 6.9 <= a and b <= 7.75, chk["interior"]


def test_an_interior_the_points_cannot_measure_is_unverifiable_not_a_pass():
    """A merged claim over both bays whose interior has points on under half of its columns (the points above 1.5 m inside it removed: the pier there unseen)
    is unverifiable (ok None) -- audit_points then cannot pass it -- while the same claim on all the points fails on
    the pier; fewer than 12 points inside is unverifiable too."""
    rng = np.random.default_rng(32)
    w, P = bay_points(rng, BAYS)
    u, z = P[:, 0], P[:, 2]
    P2 = P[~((u > 4.3) & (u < 10.3) & (z > 1.5) & (z < 4.9))]
    rec = claim(4.0, 10.6, 0.5, 5.0)
    U2, _ = RL.wall_points(w, P2, np.full(len(P2), 5))
    chk = RL.audit_points(rec, w, P2, None, U=U2)
    assert chk["ok"] is None and chk["interior"]["ok"] is None and "cannot be measured" in chk["why"], chk
    assert min(chk["interior"]["cover"]) < 0.5, chk["interior"]
    U, _ = RL.wall_points(w, P, np.full(len(P), 5))
    assert RL.audit_points(rec, w, P, None, U=U)["ok"] is False
    few = RL.interior_structure(np.array([5.0, 6.0]), np.array([-0.3, -0.3]), np.array([2.0, 3.0]), rec, -0.3, 0.0)
    assert few["ok"] is None and "points inside it" in few["why"], few


def test_the_real_bays_and_partial_objects_inside_a_recess_pass():
    """The control: each real bay and the window pass audit_points with no strip inside; so does a bay holding things
    of partial extent -- a statue-like block at the frame plane over the lower 40 % of an arch, a 0.1 m impost
    moulding half-way out across it (a member is at least interior_h_m tall)."""
    rng = np.random.default_rng(32)
    w, P = bay_points(rng, BAYS)
    U, _ = RL.wall_points(w, P, np.full(len(P), 5))
    for u0, u1, z0, z1, dep in BAYS:
        chk = RL.audit_points(claim(u0, u1, z0, z1), w, P, None, U=U)
        assert chk["ok"] is True and chk["interior"]["ok"] and not chk["interior"]["v"], chk
    w2, P2 = bay_points(rng, [(4.0, 8.0, 0.5, 9.0, 0.3)])
    u, z = P2[:, 0], P2[:, 2]
    statue = (u > 5.5) & (u < 6.2) & (z > 0.5) & (z < 3.8)                     # 0.7 m wide, 3.3 of 8.5 m tall
    P2[statue, 1] = 0.0
    impost = (u > 4.0) & (u < 8.0) & (z > 5.0) & (z < 5.1)                     # half-way out, 0.1 m tall
    P2[impost, 1] = 0.15
    U2, _ = RL.wall_points(w2, P2, np.full(len(P2), 5))
    chk = RL.audit_points(claim(4.0, 8.0, 0.5, 9.0), w2, P2, None, U=U2)
    assert chk["ok"] is True and chk["interior"]["ok"], chk


def test_interior_structure_is_measured_from_the_points_not_the_record():
    """The interior profile reads only the points: the record's own interior claim, line numbers or frame do not
    enter; a pier over 69 % of the element's height is structure, a flat-frame strip of partial height (42 %) is not."""
    rng = np.random.default_rng(33)
    w, P = bay_points(rng, [(4.0, 10.6, 0.5, 5.0, 0.3)])
    u, z = P[:, 0], P[:, 2]
    pier = (u > 7.0) & (u < 7.3) & (z < 3.4)                                 # 0.3 m wide, 2.9 of 4.2 m (69 %)
    P[pier, 1] = 0.0
    U, _ = RL.wall_points(w, P, np.full(len(P), 5))
    rec = claim(4.0, 10.6, 0.5, 5.0, recheck={"interior": {"ok": True, "v": [], "h": []}})
    chk = RL.audit_points(rec, w, P, None, U=U)
    assert chk["ok"] is False and chk["interior"]["v"], chk["interior"]
    w3, P3 = bay_points(rng, [(4.0, 10.6, 0.5, 5.0, 0.3)])
    P3[(P3[:, 0] > 7.0) & (P3[:, 0] < 7.6) & (P3[:, 2] < 2.4), 1] = 0.0     # 1.75 of 4.2 m: partial, not a pier
    U3, _ = RL.wall_points(w3, P3, np.full(len(P3), 5))
    assert RL.audit_points(claim(4.0, 10.6, 0.5, 5.0), w3, P3, None, U=U3)["interior"]["ok"]
    # a proud element or a frame strip is not a recess: nothing to check inside
    v = {"type": "vertical", "how": "proud", "u0": 7.0, "u1": 7.6, "z_bottom": 0.5, "z_top": 5.0, "projection": 0.3,
         "offset_m": 0.3, "frame": {"c0": 0.0, "c1": 0.0}}
    assert RL.audit_points(v, w, P, None, U=U)["interior"] is None


def test_the_detector_splits_a_recess_at_a_pier_the_strip_detection_missed():
    """Two bays joined by a recessed band above a pier that stops 1 m below their tops: one connected recess, and the
    pier (frame over 76 % of the bays' height) is no full-register strip. The detector finds the pier inside the
    opening (interior_structure), makes it a vertical of its own and cuts there: two bays, each passing the
    re-measurement; with only accepted verticals as cutters (cut_boxes, the pier not among them) the one recess comes
    back and fails G-R6's re-measurement instead -- never one recess over a pier."""
    rng = np.random.default_rng(34)
    w, P = bay_points(rng, [(4.0, 10.6, 0.5, 5.0, 0.3)])
    u, z = P[:, 0], P[:, 2]
    P[(u > 7.0) & (u < 7.6) & (z < 4.0), 1] = 0.0                           # the pier, up to 4.0 m
    trk = np.full(len(P), 5)
    WR = RL.measure_wall(w, P, trk, None)
    ops = sorted((o for o in WR.openings if o["level"] == 1), key=lambda o: o["u0"])
    piers = [v for v in WR.verticals if v.get("found_inside")]
    assert len(piers) == 1 and 6.85 <= piers[0]["u0"] <= 7.15 and 7.45 <= piers[0]["u1"] <= 7.75, WR.verticals
    boxes = [(v["u0"], v["u1"], v["z_bottom"], v["z_top"]) for v in WR.verticals]
    assert len(set(boxes)) == len(boxes) and all(v["u1"] - v["u0"] >= RL.DEFAULT_RELIEF.vert_w[0] for v in piers)
    assert piers[0]["how"] == "frame" and piers[0]["z_bottom"] <= 0.7 and piers[0]["z_top"] >= 4.8
    assert len(ops) == 2 and ops[0]["u1"] <= 7.15 and ops[1]["u0"] >= 7.45, [(o["u0"], o["u1"]) for o in ops]
    U, _ = RL.wall_points(w, P, trk)
    els = guarded(WR, w)
    for e in els:
        if e["type"] == "opening" and e["level"] == 1:
            chk = RL.audit_points(e, w, P, None, U=U)
            assert chk["interior"]["ok"], (e["u0"], e["u1"], chk["interior"])
    # only accepted verticals cut (the pier rejected): the recess stays whole and its re-measurement refuses it
    one = RL.measure_wall(w, P, trk, None, cut_boxes=[])
    merged = [o for o in one.openings if o["level"] == 1 and o["u0"] < 7.0 < 7.6 < o["u1"]]
    assert merged and [v for v in one.verticals if v.get("found_inside")]
    e = next(x for x in guarded(one, w) if x["type"] == "opening" and x["u0"] < 7.0 < 7.6 < x["u1"])
    chk = RL.audit_points(e, w, P, None, U=U)
    assert chk["ok"] is False and chk["interior"]["v"], chk


# two orders (as on the Baptistery): a lower bay and an upper one over the same pilasters, the entablature
# between them standing only part of the way out (here 0.4 of the bays' relief), noisy and sparse as real SfM points
TWO_ORDERS = [(1.0, 3.6, 0.3, 4.5, 0.3), (1.0, 3.6, 5.3, 9.5, 0.3), (1.0, 3.6, 4.5, 5.3, 0.18),
              (4.2, 6.8, 0.3, 4.5, 0.3), (4.2, 6.8, 5.3, 9.5, 0.3), (4.2, 6.8, 4.5, 5.3, 0.18)]


def two_orders(seed, entablature=0.18):
    rng = np.random.default_rng(seed)
    rec = [r if r[2] != 4.5 else r[:4] + (entablature,) for r in TWO_ORDERS]
    w, P = bay_points(rng, rec, length=8.0, height=11.0, density=150, noise=0.1)
    return w, P, RL.wall_points(w, P, np.full(len(P), 5))[0]


@pytest.mark.parametrize("seed", [100, 101, 102])
def test_a_bay_claimed_through_the_entablature_between_two_orders_fails_the_re_measurement(seed):
    """One recess claimed from the ground through the entablature to the upper order. The entablature stands only 0.4
    of the way out of the bays and the points are noisy (sigma 0.1 m) and sparse (150 per m2), so no row of cells
    reaches the frame-plane threshold (half the relief); as a MEMBER across the recess (pooled rows rising >=
    interior_member of the relief over the bays below and above it, across the width) it fails G-R6; the honest lower
    and upper bays pass."""
    w, P, U = two_orders(seed)
    chk = RL.audit_points(claim(1.0, 3.6, 0.3, 9.5), w, P, None, U=U)
    assert chk["ok"] is False and chk["interior"]["h"] and not chk["interior"]["v"], chk
    a, b = chk["interior"]["h"][0]
    assert 4.3 <= a and b <= 5.5 and "horizontal strip" in chk["why"], chk["interior"]
    for z0, z1 in ((0.3, 4.5), (5.3, 9.5)):
        part = RL.audit_points(claim(1.0, 3.6, z0, z1), w, P, None, U=U)
        assert part["ok"] is True and part["interior"]["ok"], (z0, z1, part["interior"])
    # a recess with no member in it stays clean (the noise floor)
    flat_w, flat_P, flat_U = two_orders(seed, entablature=0.3)
    assert RL.audit_points(claim(1.0, 3.6, 0.3, 9.5), flat_w, flat_P, None, U=flat_U)["interior"]["ok"]


def test_a_shallower_strip_at_a_recess_s_own_end_is_no_member():
    """A member stands BETWEEN two parts of a recess (both flanks needed): a recess whose top 0.5 m is shallower (0.4
    of its relief: a moulding under the attic's cornice, the real Baptistery setbacks' tops) holds no member -- its
    top edge may be off by that much, which the edge tolerances bound, but it does not span structure; the same strip
    with recess above it too is one (the entablature)."""
    for seed in (400, 401, 402):
        rng = np.random.default_rng(seed)
        w, P = bay_points(rng, [(1.0, 4.0, 0.3, 9.5, 0.3), (1.0, 4.0, 8.9, 9.5, 0.18)], length=6.0, height=11.0,
                          density=150, noise=0.05)
        U, _ = RL.wall_points(w, P, np.full(len(P), 5))
        chk = RL.audit_points(claim(1.0, 4.0, 0.3, 9.5), w, P, None, U=U)
        assert not chk["interior"]["h"], (seed, chk["interior"])


def test_a_member_seen_only_on_every_third_row_is_still_a_member():
    """Rows of a horizontal profile pool the points within INTERIOR_POOL_M of their centre: SfM points on a moulding
    often come along its courses only (here one 0.1 m row in three over the entablature has points). Pooled, the
    member is seen on every seed; row by row, it is broken into 0.1 m pieces under the strip height."""
    for seed in range(100, 104):
        w, P, _ = two_orders(seed)
        z = P[:, 2]
        P = P[~((z > 4.5) & (z < 5.3) & ((np.floor(z / 0.1).astype(int) % 3) != 0))]
        U, _ = RL.wall_points(w, P, np.full(len(P), 5))
        chk = RL.audit_points(claim(1.0, 3.6, 0.3, 9.5), w, P, None, U=U)
        assert chk["ok"] is False and chk["interior"]["h"], (seed, chk["interior"])


@pytest.mark.parametrize("seed", [300, 301, 302])
def test_a_step_inside_a_recess_is_no_member(seed):
    """A member is a BUMP between two parts of a recess, not a step: a recess whose lower third is shallower (statues
    standing in an arch across its width, a lower part 0.13 m less deep) has, at the top of the shallower part, rows
    above the mean of their flanks -- but not above the flank below: no member there (with the mean alone the top of
    the Baptistery's statues and this step would be cut)."""
    rng = np.random.default_rng(seed)
    w, P = bay_points(rng, [(1.0, 4.0, 0.3, 9.5, 0.3), (1.0, 4.0, 0.3, 3.5, 0.17)], length=6.0, height=11.0,
                      density=150, noise=0.05)
    U, _ = RL.wall_points(w, P, np.full(len(P), 5))
    chk = RL.audit_points(claim(1.0, 4.0, 0.3, 9.5), w, P, None, U=U)
    assert not chk["interior"]["h"], chk["interior"]


def test_the_detector_cuts_a_recess_at_the_entablature_into_two_measured_openings():
    """The detector finds the member inside the tall recess (interior_structure) and cuts it there, as it cuts at a
    pier: the lower bay and the upper one are separate openings, each with its own measured edge at the entablature
    (refined from the raw points) and each passing the re-measurement; the cut is recorded on the
    register (member_cuts). With only accepted verticals as cutters (cut_boxes) the member still cuts."""
    w, P, U = two_orders(100)
    trk = np.full(len(P), 5)
    for cut in (None, []):
        WR = RL.measure_wall(w, P, trk, None, cut_boxes=cut)
        ops = sorted(((o["u0"], o["z_bottom"]), o) for o in WR.openings if o["level"] == 1)
        ops = [o for _, o in ops]
        assert len(ops) == 4, [(o["u0"], o["u1"], o["z_bottom"], o["z_top"]) for o in ops]
        for lo, hi in ((ops[0], ops[1]), (ops[2], ops[3])):
            assert lo["z_top"] <= 4.75 and hi["z_bottom"] >= 5.05 and abs(lo["u0"] - hi["u0"]) < 0.3, (lo, hi)
            for o in (lo, hi):
                chk = RL.audit_points(dict(o, offset_m=-o["depth"], frame={"c0": 0.0, "c1": 0.0}), w, P, None, U=U)
                assert chk["interior"]["ok"], (o["z_bottom"], o["z_top"], chk["interior"])
        assert any(r.get("member_cuts") for r in WR.registers)


def test_a_pier_with_gaps_in_its_points_is_still_structure():
    """A pier between two bays with SfM gaps -- a third of its height without points, in 0.3 m bands, as on the Baptistery's piers. A
    column counts with data on at least half its cells, so the merged claim still fails on the pier."""
    for seed in (200, 201, 202):
        rng = np.random.default_rng(seed)
        w, P = bay_points(rng, [(4.0, 7.0, 0.5, 5.0, 0.3), (7.6, 10.6, 0.5, 5.0, 0.3)], density=150, noise=0.03)
        u, z = P[:, 0], P[:, 2]
        bands = np.arange(0.5, 5.0, 0.3)
        drop = np.zeros(len(P), bool)
        for k in rng.choice(len(bands), size=len(bands) // 3, replace=False):
            drop |= (u > 6.95) & (u < 7.65) & (z >= bands[k]) & (z < bands[k] + 0.3)
        P = P[~drop]
        U, _ = RL.wall_points(w, P, np.full(len(P), 5))
        chk = RL.audit_points(claim(4.0, 10.6, 0.5, 5.0), w, P, None, U=U)
        assert chk["ok"] is False and chk["interior"]["v"], (seed, chk["interior"])
        a, b = chk["interior"]["v"][0]
        assert 6.9 <= a and b <= 7.75, chk["interior"]


# a photographed facade with two bays, a pier between them and a window above the left one (end to end)
BAYS_U = [(4.0, 7.0, 0.6, 5.0), (7.6, 10.6, 0.6, 5.0), (4.0, 7.0, 6.0, 9.0)]


def bays_scene(num_cams=14) -> st.SceneInput:
    scene = ts.south_scene(num_cams)
    gx, gz = np.meshgrid(np.arange(-9.95, 10, 0.1), np.arange(0.05, ts.WALL_H, 0.1))
    x, z = gx.ravel(), gz.ravel()
    y = np.full(x.size, -6.0)
    u = x + 10.0
    for a, b, c, d in BAYS_U:
        y[(u > a) & (u < b) & (z > c) & (z < d)] += 0.30
    wall = np.stack([x, y, z], 1)
    rx, ry = np.meshgrid(np.arange(-9.5, 10, 1.0), np.arange(-5.5, 6, 1.0))
    roof = np.stack([rx.ravel(), ry.ravel(), np.full(rx.size, ts.WALL_H)], 1)
    scene.points = np.vstack([wall, roof])
    scene.colors = np.vstack([ts.facade_colour(x, z), np.tile([[200, 90, 70]], (len(roof), 1))]).astype(np.uint8)
    scene.point_tracks = project_all(scene.points, scene.cameras)
    n = np.bincount(scene.point_tracks.point, minlength=len(scene.points))
    scene.point_quality = np.stack([n, np.full(len(n), 20.0)], 1).astype(float)
    scene.lines3d = south_lines([s for a, b, c, d in BAYS_U for s in ((a, c, a, d), (b, c, b, d), (a, c, b, c), (a, d, b, d))],
                                num_cams)
    return scene


def south_lines(segs, num_cams=14) -> Lines3D:
    """3-D lines on the south wall plane (y = -6): segments (u0, z0, u1, z1), u from x = -10."""
    A = np.array([[s[0] - 10.0, -6.0, s[1]] for s in segs], float).reshape(-1, 3)
    B = np.array([[s[2] - 10.0, -6.0, s[3]] for s in segs], float).reshape(-1, 3)
    return Lines3D(A, B, np.full(len(segs), num_cams), [list(range(num_cams))] * len(segs), np.ones(len(segs)),
                   np.zeros(len(segs)))


@pytest.fixture(scope="module")
def bays_model() -> st.StylizedModel:
    return st.stylize_landmark(bays_scene())


def test_the_clean_bays_are_built_and_pass_every_check(bays_model):
    """Both bays and the window are measured, accepted and built, each record re-measured for structure inside it
    (none), and the model passes check_honesty, audit_model and audit_glb."""
    s = bays_model.summary
    assert s["qa"]["passed"], s["qa"]
    node = relief_node(bays_model)
    built = sorted((r for r in node.extras["relief"] if r["built"] and r["type"] == "opening"), key=lambda r: (r["z_bottom"], r["u0"]))
    assert len(built) == 3, [(r["u0"], r["u1"], r["z_bottom"], r["z_top"]) for r in node.extras["relief"]]
    for r, (a, b, c, d) in zip(built, sorted(BAYS_U, key=lambda t: (t[2], t[0]))):
        assert abs(r["u0"] - a) < 0.2 and abs(r["u1"] - b) < 0.2 and abs(r["z_bottom"] - c) < 0.25
        assert r["recheck"]["interior"]["ok"] is True and not r["recheck"]["interior"]["v"]
    assert s["qa"]["audit"]["passed"] and s["qa"]["glb"]["passed"] and s["qa"]["relief_unbacked"] == 0


def _box_wall_points(rng, k=1.0, sides=(0, 1, 2, 3), n=3000):
    """Points on the walls of ts.BOX (x -10..10, y -6..6), 3-9 m high, the reconstruction scaled by k about the
    box's centre."""
    out = []
    for s in sides:
        t = rng.uniform(0, 1, n)
        z = rng.uniform(3.0, 9.0, n)
        if s in (0, 2):
            x, y = -10 + 20 * t, np.full(n, -6.0 if s == 0 else 6.0)
        else:
            x, y = np.full(n, 10.0 if s == 1 else -10.0), -6 + 12 * t
        out.append(np.stack([x * k, y * k, z * k], 1))
    return np.vstack(out)


def _frame_adoption(passed=True):
    rec = {"rule": "D15", "param": "outline", "k": 0.82, "t": [3.3, -1.2], "centre": [0.0, 0.0], "width_map_m": 20.0,
           "snap_scale": 1.0, "channels": {c: {"measured": True, "photos": 3, "at_bound": False, "k": 0.82, "t": [3.3, -1.2]}
                                            for c in ("sfm-points", "lines3d")},
           "scale_check": {"needed": True, "passed": passed, "map_eave": 10.0,
                           "eaves": {"skyline": 10.2, "lines3d": 10.3} if passed else {"skyline": 12.5}}}
    return {"id": "A-t-1", "kind": "frame", "active": True, "record": rec}


def test_d24_needs_an_established_scale_or_a_frame_adopted_with_the_eave_check():
    """D24's scale condition: established on the data at hand (walls all round, k within 3 % of 1, no snap rescale) ->
    allowed; the same reconstruction scaled x1.06 (or a snap rescale x1.10) -> off, with the reason; walls seen on one
    side only (no similarity measured) -> off, unless the frame was adopted under D15 with a passing eave check -- and
    off when a similarity measured on the data contradicts the scale. The condition alone decides per model
    (d24_allowed)."""
    from types import SimpleNamespace as NS
    rng = np.random.default_rng(51)
    box = NS(main_rings=[ts.BOX.copy()], adoptions=[])
    ok = RL.d24_scale_condition(box, _box_wall_points(rng), None)
    assert ok["allowed"] and ok["by"] == "scale_status" and ok["scale_status"]["established"], ok
    off = RL.d24_scale_condition(box, _box_wall_points(rng, k=1.06), None)
    assert not off["allowed"] and "in question" in off["why"] and abs(off["channels"]["sfm-points"]["k"] - 1.06) < 0.01
    snap = RL.d24_scale_condition(box, _box_wall_points(rng), None, snap_scale=1.10)
    assert not snap["allowed"] and "snap scale" in snap["why"]
    one_side = _box_wall_points(rng, sides=(0,))
    assert not RL.d24_scale_condition(box, one_side, None)["allowed"]
    adopted = NS(main_rings=[ts.BOX.copy()], adoptions=[_frame_adoption(True)])
    a = RL.d24_scale_condition(adopted, one_side, None)
    assert a["allowed"] and a["by"].startswith("D15 frame adoption") and a["frame_adoption"]["id"] == "A-t-1", a
    assert not RL.d24_scale_condition(NS(main_rings=[ts.BOX.copy()], adoptions=[_frame_adoption(False)]), one_side, None)["allowed"]
    assert not RL.d24_scale_condition(adopted, _box_wall_points(rng, k=1.06), None)["allowed"]
    assert not RL.d24_scale_condition(None, _box_wall_points(rng), None)["allowed"]
    assert RL.d24_allowed(RL.ReliefResult(stats={"d24": ok}))
    for cond in (off, snap, a | {"allowed": False}, {}):
        assert not RL.d24_allowed(RL.ReliefResult(stats={"d24": cond}))


def test_the_relief_summary_records_the_d24_scale_condition(relief_model):
    """Every measured model records D24's scale condition: off, with its reason, on the one-facade test box (an OSM
    footprint massing: no structure frame to establish the scale against)."""
    s = relief_model.summary["relief"]
    st_ = s["stats"]
    assert st_["d24"]["allowed"] is False, st_["d24"]
    assert "no structure frame" in st_["d24"]["why"] and st_["d24"]["applied"] == 0
    assert s["d24"]["allowed"] is False and s["d24"]["built"] == 0 and s["built_d24"] == 0


