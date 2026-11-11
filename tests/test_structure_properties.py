"""Property tests of the hand-written geometry, with fixed seeds and no dependency beyond numpy (D34).

The roof primitives and the ear clipping were tested on thirteen hand-picked cases, every one an axis-aligned
rectangle or a regular polygon with all its tags; half a minute of random cases found three defects (a skillion
without roof:direction crashed, a half-hipped roof with its ridge across the long side ended 0.4 m under its
registered top, courtyards and openings in rows folded the triangulation). These tests keep that half minute's
coverage in a few seconds:

- every roof shape on rotated rectangles, L shapes and star outlines with random tag variants: no exception, a
  closed solid, the mesh's top at the registered top, the roof's plan area equal to the outline's;
- polygons with 2-8 courtyards under a random rotation, exact and jittered grids, star-shaped holes, and walls with
  one to three rows of openings: the triangles' absolute areas sum to the polygon's area and none is reversed;
- relief_mesh on walls with rows of openings: the wall face is exact and every triangle faces its normal.
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pytest

from crowd3d import relief as RL
from crowd3d import structure as S
from crowd3d import stylize as st

SEED = 20261111


def rot(p, deg: float) -> np.ndarray:
    a = math.radians(deg)
    return np.asarray(p, float) @ np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]).T


def box(cx, cy, w, h) -> np.ndarray:
    return np.array([[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]])


def star(rng, n: int, r0: float, jitter: float, c=(0.0, 0.0)) -> np.ndarray:
    ang = np.sort(rng.uniform(0, 2 * math.pi, n))
    r = r0 * (1 + jitter * rng.uniform(-1, 1, n))
    return np.c_[c[0] + r * np.cos(ang), c[1] + r * np.sin(ang)]


def simple_star(rng, n: int, r0: float, jitter: float, c=(0.0, 0.0)) -> np.ndarray:
    """A star-shaped ring that is simple (random angles can put all vertices on one side of the centre)."""
    while True:
        p = star(rng, n, r0, jitter, c)
        if st.ring_is_simple(p) and abs(st.signed_area(p)) > 6.0:
            return p


# ---------------------------------------------------------------- roof primitives


def outline(rng, kind: str) -> np.ndarray:
    if kind == "rect":
        w, h = rng.uniform(3, 40), rng.uniform(3, 40)
        p = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    elif kind == "L":
        w, h = rng.uniform(8, 40), rng.uniform(8, 40)
        a, b = rng.uniform(.3, .7) * w, rng.uniform(.3, .7) * h
        p = np.array([[0, 0], [w, 0], [w, b], [a, b], [a, h], [0, h]], float)
    else:
        return simple_star(rng, int(rng.integers(5, 14)), rng.uniform(4, 20), 0.35)
    return rot(p, rng.uniform(0, 360))


def roof_tags(rng, shape: str) -> dict[str, str]:
    tags = {"building:part": "yes", "height": "20", "roof:shape": shape}
    if rng.random() < 0.8:                       # (without it: the shape's conventional height, D17)
        tags["roof:height"] = f"{rng.uniform(1, 8):.1f}"
    if rng.random() < 0.3:
        tags["roof:orientation"] = "across"
    if rng.random() < 0.4:
        tags["roof:direction"] = str(int(rng.integers(0, 360))) if rng.random() < 0.7 else \
            str(rng.choice(["N", "ESE", "SW", "W"]))
    return tags


def roof_checks(bp: S.BuiltPart) -> dict[str, float]:
    """What must hold for any built part: (unmatched + duplicated edges, |mesh top - registered top|, |roof plan
    area - outline area| relative to the outline's, non-finite coordinates)."""
    tri = bp.triangles()
    rt = np.concatenate([f.triangles() for f in bp.roof.faces])
    plan = 0.5 * float(np.cross(rt[:, 1] - rt[:, 0], rt[:, 2] - rt[:, 0])[:, 2].sum())
    area = S.poly_area(bp.outer, bp.holes)
    return {"open_edges": bp.census["unmatched"] + bp.census["duplicated"],
            "top_gap": abs(float(tri[..., 2].max()) - bp.top),
            "plan_gap": abs(plan - area) / max(area, 1.0),
            "non_finite": int((~np.isfinite(tri)).sum())}


@pytest.mark.parametrize("shape", S.SHAPES)
def test_every_roof_shape_builds_a_closed_solid_that_reaches_its_top_on_random_outlines(shape):
    """24 parts per shape (8 rotated rectangles, 8 L shapes, 8 stars; random roof height, orientation, direction):
    build_part raises nothing (a shape that does not fit falls back with its reason, never an exception), the solid
    is closed, its mesh tops out at its registered top and its roof covers exactly its outline. On a rectangle
    every shape is built as tagged."""
    rng = np.random.default_rng([SEED, S.SHAPES.index(shape)])
    built_as_tagged = 0
    for trial in range(24):
        kind = ("rect", "L", "star")[trial % 3]
        ring = outline(rng, kind)
        tags = roof_tags(rng, shape)
        spec = S.PartSpec("w1", "yes", S.ccw(ring), [], tags)
        S.resolve_heights(spec)
        case = (shape, kind, np.round(ring, 2).tolist(), tags)
        bp = S.build_part(spec)                                  # (no exception: a Fallback never leaves build_roof)
        got = roof_checks(bp)
        assert got["open_edges"] == 0 and got["non_finite"] == 0, (case, bp.census)
        assert got["top_gap"] < 1e-6, (case, bp.top, got)
        assert got["plan_gap"] < 1e-3, (case, got)
        assert abs(bp.top - 20.0) < 1e-6 or spec.params["roof_height"].source == st.PROV_ROOF_ANGLE, (case, bp.top)
        if bp.roof.shape != spec.roof_shape:                     # another shape: only by the recorded fallback chain
            assert bp.roof.meta.get("fallback_from") == spec.roof_shape and bp.roof.meta.get("reason"), case
            assert bp.roof.shape in S.FALLBACKS[spec.roof_shape], case
        else:
            built_as_tagged += 1
        if kind == "rect":
            assert bp.roof.shape == shape and not bp.roof.meta.get("fallback_from"), (case, bp.roof.meta)
    assert built_as_tagged >= 8


def test_the_model_gate_agrees_with_the_parts_on_a_street_of_random_roofs():
    """The same through build_structure: 26 parts in a row, two of every shape, all drawn and all built, the
    structure gates pass and nothing is reported as not built."""
    rng = np.random.default_rng([SEED, 99])
    specs = []
    for k, shape in enumerate(S.SHAPES * 2):
        ring = box(30.0 * k, 0.0, rng.uniform(8, 20), rng.uniform(6, 14))
        specs.append(S.PartSpec(f"w{k}", "yes", S.ccw(ring), [], roof_tags(rng, shape)))
    main = S.ccw(box(30.0 * 12.5, 0.0, 30.0 * 27, 40.0))
    m = S.build_structure("street", specs, [main], {"height": "3"})
    chk = S.check_structure(m)
    assert chk["passed"] and chk["G13_not_built"] == [] and chk["G1_not_watertight"] == [], chk
    assert {b.id for b in m.built} >= {f"w{k}" for k in range(26)}
    assert all(p.not_built is None for p in m.specs) and m.report["parts_not_built"] == []


# ---------------------------------------------------------------- ear clipping over holes


def grid(rng, exact: bool, rotate: float | None, rows: int | None = None, cols: int | None = None,
         arched: bool = False, roll: bool = False) -> tuple[np.ndarray, list[np.ndarray]]:
    """A rectangle with a grid of rectangular holes: courtyards of a palace, or the openings of a facade (rows of
    equal windows when exact: equal sills and heads, the case a regular facade is)."""
    W, H = rng.uniform(20, 120), rng.uniform(10, 80)
    nx = cols or int(rng.integers(1, 5))
    ny = rows or int(rng.integers(1, 3))
    cw, ch = W / nx, H / ny
    fw, fh = rng.uniform(.3, .6), rng.uniform(.3, .6)
    holes = []
    for i in range(nx):
        for j in range(ny):
            cx, cy = -W / 2 + (i + .5) * cw, -H / 2 + (j + .5) * ch
            h = box(cx, cy, cw * fw, ch * fh) if exact else \
                box(cx + rng.uniform(-.1, .1) * cw, cy + rng.uniform(-.1, .1) * ch,
                    cw * rng.uniform(.3, .6), ch * rng.uniform(.3, .6))
            if arched and h[2, 1] - h[0, 1] > 0.5 * (h[1, 0] - h[0, 0]) + 0.1:
                h = RL._arch_polygon(h[0, 0], h[1, 0], h[0, 1], h[2, 1] - 0.5 * (h[1, 0] - h[0, 0]), h[2, 1])
            if roll:
                h = np.roll(h, int(rng.integers(0, len(h))), axis=0)
            holes.append(h)
    outer = box(0, 0, W, H)
    if rotate is not None:
        outer, holes = rot(outer, rotate), [rot(h, rotate) for h in holes]
    return outer, holes


def star_holes(rng) -> tuple[np.ndarray, list[np.ndarray]]:
    centres = [(10, 10), (30, 10), (10, 30), (30, 30)][:int(rng.integers(1, 5))]
    return box(20, 20, 40, 40), [simple_star(rng, int(rng.integers(3, 10)), 4, 0.4, c) for c in centres]


FAMILIES = {
    "2-8 courtyards, jittered, random rotation":
        lambda rng: grid(rng, False, rng.uniform(0, 360), rows=int(rng.integers(1, 3)), cols=int(rng.integers(2, 5))),
    "courtyards in an exact grid, random rotation": lambda rng: grid(rng, True, rng.uniform(0, 360)),
    "courtyards in an exact grid, axis-aligned": lambda rng: grid(rng, True, None),
    "courtyards in an exact grid, turned 90 degrees": lambda rng: grid(rng, True, 90.0),
    "a wall with 1-3 exact rows of 2-8 openings":
        lambda rng: grid(rng, True, None, rows=int(rng.integers(1, 4)), cols=int(rng.integers(2, 9))),
    "a wall with 1-3 jittered rows of 2-8 openings":
        lambda rng: grid(rng, False, None, rows=int(rng.integers(1, 4)), cols=int(rng.integers(2, 9))),
    "a wall with rows of arched openings":
        lambda rng: grid(rng, True, None, rows=int(rng.integers(1, 4)), cols=int(rng.integers(2, 6)), arched=True),
    "openings whose rings start at any vertex":
        lambda rng: grid(rng, True, None, rows=int(rng.integers(1, 4)), cols=int(rng.integers(2, 7)), roll=True),
    "1-4 star-shaped courtyards": star_holes,
}


def tiling(outer: np.ndarray, holes: list[np.ndarray]) -> tuple[float, int, st.Triangulation]:
    """(sum of the triangles' absolute areas minus the polygon's area, reversed triangles, the triangulation),
    measured here from the returned vertices and triangles, not read from the triangulation's own account."""
    ring, tris = st.triangulate_with_holes(outer, holes)
    t = np.array(tris, int).reshape(-1, 3)
    a, b, c = ring[t[:, 0]], ring[t[:, 1]], ring[t[:, 2]]
    signed = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    want = abs(st.signed_area(outer)) - sum(abs(st.signed_area(h)) for h in holes)
    return float(np.abs(signed).sum() - want), int((signed < -1e-9).sum()), tris


@pytest.mark.parametrize("family", FAMILIES)
def test_triangles_tile_a_polygon_with_holes_exactly(family):
    """24 polygons per family: the triangles' absolute areas sum to the polygon's area (outline minus holes), no
    triangle is reversed, and the triangulation says so itself (exact). (Before D34, 300 per family: none wrong for
    the exact grid under a random rotation, 4 to 139 for the others, 528 of 2700 in all, each silently.)"""
    rng = np.random.default_rng([SEED, list(FAMILIES).index(family)])
    for _ in range(24):
        outer, holes = FAMILIES[family](rng)
        gap, reversed_, tris = tiling(outer, holes)
        case = (family, np.round(outer, 3).tolist(), [np.round(h, 3).tolist() for h in holes])
        assert abs(gap) < 1e-6 and reversed_ == 0, (gap, reversed_, tris.problem(), case)
        assert tris.exact and tris.problem() is None and not tris.unbridged and not tris.fell_back, case


BAYS = (box(10.0, 4.875, 19.3, 9.75),                       # tests/test_relief.py's wall: two bays and a window
        [box(5.5, 2.75, 3.0, 4.6), box(9.1, 2.75, 3.0, 4.6), box(5.5, 7.55, 3.0, 3.4)])   # over the first
TWO_COURTYARDS = (np.array([[-32.1, 19.2], [-35.8, -10.9], [32.1, -19.2], [35.8, 10.9]]),      # a rotated block with
                  [np.array([[-12.3, 11.4], [-13.0, 6.1], [11.7, 3.0], [12.4, 8.3]]),           # two courtyards
                   np.array([[-12.1, -0.9], [-13.2, -9.1], [22.2, -13.5], [23.2, -5.2]])])


@pytest.mark.parametrize("outer,holes", [BAYS, TWO_COURTYARDS], ids=["bays", "two-courtyards"])
def test_the_inputs_that_folded_are_tiled_exactly_and_the_two_bridge_rules_are_why(outer, holes, monkeypatch):
    """Two inputs the old bridge_holes folded: a wall with two bays and a window above one (an existing test built
    it all along: 307 m2 of triangles on a 150 m2 wall, six reversed and turned round) and a rotated block with only
    two courtyards. Exact now; with the two D34 rules switched off (the corner test, the no-vertex-on-the-bridge
    test) the old result comes back, and the triangulation says it is not exact."""
    gap, reversed_, tris = tiling(outer, holes)
    assert abs(gap) < 1e-9 and reversed_ == 0 and tris.exact
    monkeypatch.setattr(st, "_in_corner", lambda ring, i, m: True)
    monkeypatch.setattr(st, "_passes_through", lambda m, v, pts: False)
    gap, reversed_, tris = tiling(outer, holes)
    assert gap > 1.0 and reversed_ > 0                                             # folded ...
    assert not tris.exact and tris.clockwise == reversed_ and abs(tris.area_error - gap) < 1e-9     # ... and said so
    assert "reversed triangle" in tris.problem()


def test_a_fan_a_reversed_triangle_and_an_uncut_hole_are_reported_never_silent():
    """What Triangulation reports. A ring that crosses itself: ear clipping gets stuck and fans the rest (area that
    is not the ring's). A hole outside its outline gets no bridge: triangulate_with_holes counts it, bridge_holes
    hands its index to the caller that asks and warns the one that does not. A clean polygon: exact."""
    clean = st.triangulate_polygon(box(0, 0, 10, 6))
    assert clean.exact and clean.problem() is None and (clean.stuck, clean.clockwise, clean.unbridged) == (0, 0, 0)
    assert len(clean) == 2 and abs(clean.area - 60.0) < 1e-12 and abs(clean.area_error) < 1e-12
    bow = np.array([[0, 0], [10, 0], [10, 10], [4, 10], [4, -5], [3, -5], [3, 10], [0, 10]], float)   # a spike through
    bad = st.triangulate_polygon(bow)                                                             # its own edge
    assert not bad.exact and (bad.fell_back or bad.clockwise or abs(bad.area_error) > 1e-6), bad.problem()
    assert bad.problem()
    outer, outside = box(0, 0, 40, 30), box(60, 0, 10, 10)
    ring, tris = st.triangulate_with_holes(outer, [box(0, 0, 10, 10), outside])
    assert tris.unbridged == 1 and not tris.exact and "could not be joined" in tris.problem()
    assert len(ring) == 4 + 4 + 2                                                  # the hole inside is cut
    left_out: list[int] = []
    st.bridge_holes(outer, [box(0, 0, 10, 10), outside], left_out)
    assert left_out == [1]
    with pytest.warns(RuntimeWarning, match="hole 1 has no bridge"):
        st.bridge_holes(outer, [box(0, 0, 10, 10), outside])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        st.bridge_holes(outer, [box(0, 0, 10, 10)])                                # nothing to say: no warning


def test_a_bridge_enters_the_corner_it_lands_on_and_passes_through_no_vertex():
    square = [np.array(p, float) for p in [(0, 0), (4, 0), (4, 4), (0, 4)]]         # CCW: the inside is on the left
    assert st._in_corner(square, 0, np.array([1.0, 1.0])) and not st._in_corner(square, 0, np.array([-1.0, -1.0]))
    assert not st._in_corner(square, 0, np.array([2.0, 0.0]))                       # along an edge: outside (strict)
    notch = [np.array(p, float) for p in [(0, 0), (4, 0), (4, 4), (2, 2), (0, 4)]]  # (2, 2): a reflex corner
    assert st._in_corner(notch, 3, np.array([2.0, 0.5])) and st._in_corner(notch, 3, np.array([3.9, 2.0]))
    assert not st._in_corner(notch, 3, np.array([2.0, 3.5]))                        # into the notch: outside
    pts = np.array([[1.0, 1.0], [5.0, 0.0], [2.0, 2.0 + 1e-6]])
    assert st._passes_through(np.zeros(2), np.array([3.0, 3.0]), pts)               # (1, 1) lies on it
    assert not st._passes_through(np.zeros(2), np.array([1.0, 1.0]), pts)           # an end point is not "through"
    assert not st._passes_through(np.zeros(2), np.array([3.0, 3.0]), pts[1:])


# ---------------------------------------------------------------- relief meshes on walls with rows of openings


def opening(u0, u1, z0, z1, depth=0.3, **more) -> dict:
    return {"builds_geometry": True, "type": "opening", "level": 1, "u0": u0, "u1": u1, "z_bottom": z0, "z_top": z1,
            "depth": depth, "offset_m": -depth, **more}


def facade(rng) -> tuple[st.Wall, list[dict], float]:
    """A wall with 1-3 exact rows of 2-6 openings (some arched, some with a window nested in them) and a pilaster;
    returns it with its elements and the area of the level-1 openings."""
    L, H = rng.uniform(12, 50), rng.uniform(8, 24)
    rows, cols = int(rng.integers(1, 4)), int(rng.integers(2, 7))
    wall = st.Wall(0, 0, 0, np.array([3.0, -2.0]), np.array([3.0, -2.0]) + L * np.array([0.8, 0.6]), 0.0, H,
                   st.PROV_OSM, st.PROV_OSM)
    fw, fh = rng.uniform(.35, .6), rng.uniform(.35, .6)
    els, cut = [], 0.0
    for r in range(rows):
        arched = rng.random() < 0.4
        nested = rng.random() < 0.4
        for c in range(cols):
            w, h = L / cols * fw, H / rows * fh
            u0, z0 = L * (c + 0.5) / cols - w / 2, H * (r + 0.5) / rows - h / 2
            e = opening(u0, u0 + w, z0, z0 + h)
            if arched and h > w / 2 + 0.3:
                e.update(arched=True, spring_z=z0 + h - w / 2)
                cut += abs(st.signed_area(RL._arch_polygon(u0, u0 + w, z0, z0 + h - w / 2, z0 + h)))
            else:
                cut += w * h
            els.append(e)
            if nested and not e.get("arched") and w > 1.0 and h > 1.0:
                els.append({**opening(u0 + 0.25 * w, u0 + 0.75 * w, z0 + 0.25 * h, z0 + 0.75 * h, depth=0.2),
                            "level": 2, "parent_el": len(els) - 1, "offset_m": -0.5})
    els.append({"builds_geometry": True, "type": "vertical", "u0": 0.2, "u1": 0.5, "z_bottom": 0.0, "z_top": H,
                "depth": 0.25, "offset_m": 0.25})
    return wall, els, cut


def test_relief_meshes_on_walls_with_rows_of_openings_are_exact_and_face_their_normals():
    """40 facades: relief_mesh builds every opening, the wall face it triangulates covers the wall minus its
    openings exactly (each triangle counter-clockwise on the wall), every triangle faces its own normal, and the
    report counts the quads it wound to their normals (the box's faces: by construction, not a repair)."""
    rng = np.random.default_rng([SEED, 7])
    for _ in range(40):
        wall, els, cut = facade(rng)
        refused: list[dict] = []
        mesh = RL.relief_mesh(wall, 0.0, wall.length, els, refused=refused)
        assert mesh is not None and refused == [], refused[:1]
        pos, nrm, uz, tri, rep = mesh
        assert len(rep["recesses"]) == len(els) - 1 and len(rep["boxes"]) == 1 and not rep["skipped"]
        g = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
        assert (np.einsum("ij,ij->i", g, nrm[tri[:, 0]]) > 0).all()
        n = np.r_[wall.normal, 0.0]
        d = (pos - np.r_[wall.a, 0.0]) @ n
        front = (np.abs(d[tri]).max(axis=1) < 1e-9) & (np.abs(nrm[tri[:, 0]] @ n - 1.0) < 1e-9)   # the wall face itself
        a, b, c = uz[tri[front, 0]], uz[tri[front, 1]], uz[tri[front, 2]]
        signed = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        assert (signed > -1e-9).all()
        assert abs(signed.sum() - (wall.length * (wall.z1 - wall.z0) - cut)) < 1e-6
        assert rep["wound_to_normal"] == 4                       # the box's top and start faces, two triangles each


def test_a_relief_whose_wall_face_cannot_be_tiled_is_refused_not_turned_round(monkeypatch):
    """With the old bridges the bays wall's face folds. relief_mesh used to turn the reversed triangles to face
    the wall and return the mesh; now it builds nothing and says why for every element it would have built."""
    wall = st.Wall(0, 0, 0, np.zeros(2), np.array([19.3, 0.0]), 0.0, 9.75, st.PROV_OSM, st.PROV_OSM)
    els = [opening(3.65, 6.65, 0.45, 5.05), opening(7.25, 10.25, 0.45, 5.05), opening(3.65, 6.65, 5.85, 9.25)]
    assert RL.relief_mesh(wall, 0.0, wall.length, els) is not None
    monkeypatch.setattr(st, "_in_corner", lambda ring, i, m: True)
    monkeypatch.setattr(st, "_passes_through", lambda m, v, pts: False)
    refused: list[dict] = []
    assert RL.relief_mesh(wall, 0.0, wall.length, els, refused=refused) is None
    assert [r["el"] for r in refused] == [0, 1, 2]
    assert all("relief not built on this piece" in r["why"] and "reversed triangle" in r["why"] for r in refused)
    assert RL.relief_mesh(wall, 0.0, wall.length, els) is None                     # (no list given: still nothing)
