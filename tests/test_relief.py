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


def measured(depth=0.30, lines=True, inner=None, line_box=None, seed=0):
    rng = np.random.default_rng(seed)
    w, P = synthetic_wall(rng, recess=(3, 5, 2, 6, depth), band=(9, 9.6, 0.0), inner=inner)
    L = box_lines(w, *(line_box or (3, 5, 2, 6))) if lines else None
    return w, P, RL.measure_wall(w, P, np.full(len(P), 5), L)


def guarded(WR, w, halves=None, per_half=(10, 10), mask=None):
    return RL.guard_elements(WR, [WR, WR] if halves is None else halves, list(per_half),
                             np.ones((120, 100), bool) if mask is None else mask, w)


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
    ("no 3-D line at any edge", "G-R3_line_edge"),
    ("deeper than the limit", "G-R4_depth"),
    ("on unobserved texels", "G-R5_observed_texels"),
])
def test_each_guard_refuses_its_failure(mutation, guard):
    w, P, WR = measured(depth=1.8 if mutation == "deeper than the limit" else 0.30,
                        lines=mutation != "no 3-D line at any edge")
    kw = {}
    if mutation == "few photos in one half":
        kw["per_half"] = (5, 12)
    elif mutation == "seen in one photo half only":
        kw["halves"] = [WR, _flat_half(w)]
    elif mutation == "the halves disagree on the depth":
        kw["halves"] = [WR, measured(depth=0.65, seed=3)[2]]
    elif mutation == "on unobserved texels":
        kw["mask"] = np.zeros((120, 100), bool)
    e = opening(guarded(WR, w, **kw))
    assert not e["guards"][guard] and not e["accepted"] and not e["builds_geometry"], (mutation, e["guards"])
    assert all(v for k, v in e["guards"].items() if k != guard), e["guards"]       # only the mutated guard fails


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
    assert qa["relief_unbacked"] == 0 and qa["audit"]["relief"]["points_checked"] >= 1
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
                       np.asarray(scene.points, float), [ts.BOX.copy()], scene.cameras,
                       track=np.asarray(scene.point_quality)[:, 0])
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
    elif how == "an element measured in one photo half only":
        rec["channels"]["split_half"]["iou_b"] = 0.0
    elif how == "an element without a line-confirmed edge":
        rec["channels"]["lines3d"]["edges_confirmed"] = 0
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
    elif how == "a claimed depth the points do not show":
        rec["offset_m"] = -1.2
    elif how == "relief geometry without its records":
        node.extras = {k: v for k, v in node.extras.items() if k not in ("relief", "relief_skipped", "piece_u")}
    elif how == "relief geometry copied onto an unphotographed wall, no records":
        k = next(i for i, p in enumerate(parts) if p.extras.get("kind") == "wall" and not p.extras.get("observed")
                 and model.massing.walls[p.extras["wall"]].length >= 11)
        w2 = model.massing.walls[parts[k].extras["wall"]]
        parts[k] = copy.deepcopy(parts[k])
        pos, nrm, uz, tri, _ = RL.relief_mesh(w2, 0.0, w2.length, records_as_elements(node.extras["relief"]))
        parts[k].positions, parts[k].normals, parts[k].indices = pos, nrm, tri
    elif how == "records and geometry copied onto another wall, ids renamed":
        k = next(i for i, p in enumerate(parts) if p.extras.get("kind") == "wall" and not p.extras.get("observed")
                 and model.massing.walls[p.extras["wall"]].length >= 11)
        w2 = model.massing.walls[parts[k].extras["wall"]]
        twin = copy.deepcopy(node)
        pos, nrm, uz, tri, _ = RL.relief_mesh(w2, 0.0, w2.length, records_as_elements(node.extras["relief"]))
        twin.positions, twin.normals, twin.indices = pos, nrm, tri
        twin.uvs = np.c_[uz[:, 0] / w2.length, (w2.z1 - uz[:, 1]) / (w2.z1 - w2.z0)]
        twin.name = f"wall-{w2.index}"
        twin.extras.update(wall=w2.index, piece_u=[0.0, round(w2.length, 4)])
        twin.extras["relief"] = [dict(r, id=r["id"].replace(f"w{wall.index}-", f"w{w2.index}-"),
                                      line_photos=r["line_photos"][:-1]) for r in twin.extras["relief"]]
        parts[k] = twin
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
    elif how == "a rejected element built as geometry (M9)":
        els = records_as_elements(node.extras["relief"]) + [
            {"type": "opening", "kind": "panel", "level": 1, "u0": 1.0, "u1": 3.0, "z_bottom": 2.0, "z_top": 5.0,
             "depth": 0.3, "offset_m": -0.3, "accepted": False, "builds_geometry": True}]
        pos, nrm, uz, tri, _ = RL.relief_mesh(wall, 0.0, wall.length, els)
        node.positions, node.normals, node.indices = pos, nrm, tri
        node.uvs = np.c_[uz[:, 0] / wall.length, (wall.z1 - uz[:, 1]) / (wall.z1 - wall.z0)]
    return parts, orthos


def records_as_elements(recs):
    """Relief records back as elements relief_mesh can build (a forger's view of the GLB)."""
    out = []
    for r in recs:
        e = dict(r, builds_geometry=bool(r.get("built")), level=r.get("nesting"))
        e.setdefault("depth", r.get("level", r.get("projection")))
        out.append(e)
    return out


@pytest.mark.parametrize("how, caught_by", [
    ("relief on an unphotographed wall", (0, 1, 2)),
    ("an element measured in one photo half only", (0, 2)),
    ("an element without a line-confirmed edge", (0, 2)),
    ("a failed guard", (0, 2)),
    ("a provenance other than photo-measured", (0, 2)),
    ("geometry pushed 3 m off the wall", (1, 2)),
    ("the whole node moved 3 m off its wall", (1,)),
    ("an element over unobserved texels", (1,)),
    ("a claimed depth the points do not show", (1,)),
    ("relief geometry without its records", (1, 2)),
    ("relief geometry copied onto an unphotographed wall, no records", (1, 2)),
    ("records and geometry copied onto another wall, ids renamed", (0, 1, 2)),
    ("a built record with no geometry behind it", (1, 2)),
    ("recesses deepened 2.5 times", (1, 2)),
    ("relief geometry shifted 1.5 m along the wall", (1, 2)),
    ("a rejected element built as geometry (M9)", (1, 2)),
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
    assert s["structure_metrics"]["shares"]["params"]["osm"] == 1.0          # the frame's numbers are all OSM's
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


def test_records_copied_between_walls_are_caught_even_with_new_ids():
    recs = [{"id": "w2-o7", "type": "opening", "u0": 1.0, "u1": 3.0, "z_bottom": 0.1, "z_top": 9.0, "depth": 0.27,
             "offset_m": -0.27}]
    assert RL.copied_records([("wall-2-0", 2, recs), ("wall-2-1", 2, recs)]) == []      # two pieces of one wall
    twin = [dict(recs[0], id="w1-o7")]
    v = RL.copied_records([("wall-2", 2, recs), ("wall-1", 1, twin)])
    assert len(v) == 1 and "a copy, not a measurement" in v[0]["why"]
    v = RL.copied_records([("wall-1", 1, recs)])
    assert len(v) == 1 and "does not name wall 1" in v[0]["why"]
    assert RL.copied_records([("wall-2", 2, recs), ("wall-1", 1, [dict(twin[0], u0=1.013)])]) == []


def test_relief_is_never_built_on_an_unphotographed_piece(relief_model):
    """M15: build_parts draws relief only on the observed strips of a wall; with no observed facade the relief
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
    """M11: node_violations checks 'observed' on its own (a textured node marked unobserved is refused)."""
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
