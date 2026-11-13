"""Pure 2-D ring and polygon helpers (numpy only): areas and orientation, point-in-polygon, distances to an outline,
Douglas-Peucker, ear clipping with bridged holes and its account of itself (Triangulation).

The lowest layer of the clay model's family (D34 part C): crowd3d.structure, the judges and crowd3d.stylize all
import from here, and this module imports nothing of the project. The definitions were moved out of
crowd3d/stylize.py unchanged (same names, signatures, bodies, docstrings); crowd3d.stylize imports every one of them
back, so stylize.<name> still works. Coordinates are metres in a local frame (the landmark's ENU x, y; or a wall's
u, z).
"""

from __future__ import annotations

import math
import warnings
from typing import Any

import numpy as np


def _cross2(a, b) -> float:
    """z of the cross product of two 2-D vectors (numpy 2 dropped 2-D np.cross)."""
    return float(a[0] * b[1] - a[1] * b[0])


def signed_area(poly: np.ndarray) -> float:
    p = np.asarray(poly, dtype=float)
    q = np.roll(p, -1, axis=0)
    return float((p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]).sum() / 2.0)


def ensure_ccw(poly: np.ndarray) -> np.ndarray:
    """Counter-clockwise ring without a closing vertex (outward normal of edge a->b is (dy, -dx))."""
    p = np.asarray(poly, dtype=float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    return p[::-1].copy() if signed_area(p) < 0 else p.copy()


def polygon_centroid(poly: np.ndarray) -> np.ndarray:
    p = np.asarray(poly, dtype=float)
    q = np.roll(p, -1, axis=0)
    cross = p[:, 0] * q[:, 1] - q[:, 0] * p[:, 1]
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return p.mean(axis=0)
    return ((p + q) * cross[:, None]).sum(axis=0) / (6.0 * a)


def _dp_open(pts: np.ndarray, tol: float) -> list[int]:
    """Douglas-Peucker on an open polyline: indices kept (first and last always)."""
    keep = [0, len(pts) - 1]
    stack = [(0, len(pts) - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        ab = b - a
        n = np.linalg.norm(ab)
        seg = pts[i + 1:j] - a
        d = np.linalg.norm(seg, axis=1) if n < 1e-12 else np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / n
        k = int(d.argmax())
        if d[k] > tol:
            m = i + 1 + k
            keep.append(m)
            stack += [(i, m), (m, j)]
    return sorted(set(keep))


def simplify_polygon(poly: np.ndarray, tol: float) -> np.ndarray:
    """Douglas-Peucker on a closed ring, anchored at the two mutually farthest vertices."""
    p = ensure_ccw(poly)
    if len(p) <= 4 or tol <= 0:
        return p
    j = int(np.linalg.norm(p - p[0], axis=1).argmax())
    i = int(np.linalg.norm(p - p[j], axis=1).argmax())
    i, j = min(i, j), max(i, j)
    first = _dp_open(p[i:j + 1], tol)
    second = _dp_open(np.vstack([p[j:], p[:i + 1]]), tol)
    idx = [i + k for k in first] + [(j + k) % len(p) for k in second[1:-1]]
    out = p[sorted(set(idx), key=lambda v: (v - i) % len(p))]
    return ensure_ccw(out)


def points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd rule, vectorized over points."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(x), dtype=bool)
    p = np.asarray(poly, float)
    q = np.roll(p, -1, axis=0)
    for (x1, y1), (x2, y2) in zip(p, q):
        cond = (y1 > y) != (y2 > y)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (x < xc)
    return inside


def in_rings(xy: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Inside the first ring (the outer) and outside every further one (its holes)."""
    inside = points_in_polygon(xy, rings[0])
    for h in rings[1:]:
        inside &= ~points_in_polygon(xy, h)
    return inside


def distance_to_outline(xy: np.ndarray, poly: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per point: distance to the closed outline, the closest outline point and the edge index."""
    pts = np.asarray(xy, float).reshape(-1, 2)
    a = np.asarray(poly, float)
    ab = np.roll(a, -1, axis=0) - a
    ab2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    dist = np.empty(len(pts))
    closest = np.empty_like(pts)
    edge = np.empty(len(pts), dtype=int)
    for s in range(0, len(pts), chunk):
        p = pts[s:s + chunk]
        t = np.clip(((p[:, None, :] - a[None]) * ab[None]).sum(axis=2) / ab2[None], 0.0, 1.0)
        c = a[None] + t[..., None] * ab[None]
        d = np.linalg.norm(p[:, None, :] - c, axis=2)
        k = d.argmin(axis=1)
        rows = np.arange(len(p))
        dist[s:s + chunk], closest[s:s + chunk], edge[s:s + chunk] = d[rows, k], c[rows, k], k
    return dist, closest, edge


class Triangulation(list):
    """triangulate_polygon's triangles (a list of index triples) with what the ear clipping did, so that no caller
    ships a folded face without knowing (D34). Ear clipping that finds no ear fans the vertices left from the first
    one; on a ring that is not (weakly) simple -- a hole bridged to the wrong copy of a repeated vertex was the case
    found -- that fan overlaps other triangles or turns clockwise.

    stuck: the vertices left when no ear was found (0: the clipping finished); fan_area: the area the fan drew
    (0.0 when the vertices left were collinear: nothing was drawn); clockwise: triangles turning the wrong way;
    area / area_error: the ring's area and the triangles' summed absolute area minus it; unbridged: holes that
    got no bridge and are not cut (triangulate_with_holes)."""

    stuck: int = 0
    fan_area: float = 0.0
    clockwise: int = 0
    area: float = 0.0
    area_error: float = 0.0
    unbridged: int = 0

    @property
    def fell_back(self) -> bool:
        """The fan drew something: the triangles are not the ear clipping of a simple ring."""
        return bool(self.stuck and self.fan_area > TRI_AREA_TOL_M2)

    @property
    def exact(self) -> bool:
        """The triangles tile the polygon: no fan, none clockwise, their areas sum to the polygon's, every hole cut.

        Of the first three, `clockwise` is the one that finds a folded ring. Clipped ears and the fan both telescope:
        their signed areas sum to the ring's, so with none clockwise the absolute areas do too, and every point is
        covered as often as the ring winds round it. The other two are kept on purpose:
        - a fan with none clockwise can be a correct tiling (an L with every vertex written twice: 12 vertices
          fanned, 12 m2, nothing reversed) and is still not exact. That is the rule, not an oversight: ear clipping
          finds an ear on every ring a clean outline gives (0 fans in the 13 catalogue landmarks' 30,402
          triangulations), so a fan means the ring is not what the caller thinks it is;
        - the area test is belt and braces: without a clockwise triangle it only fires if several hundred slivers
          each turn the wrong way by less than TRI_CLOCKWISE_M2.

        What none of them can see is a ring that winds twice round some area (a hole that crosses itself, bridged):
        there the triangles lie in two layers, all turning the right way. bridge_holes does not cut such a hole
        (ring_crosses_itself); the outline is the caller's to clean (clean_ring)."""
        return not (self.fell_back or self.clockwise or self.unbridged
                    or abs(self.area_error) > TRI_AREA_TOL_M2 + TRI_AREA_TOL_REL * self.area)

    def problem(self) -> str | None:
        """What is wrong with it, in words; None when it is exact."""
        if self.exact:
            return None
        why = []
        if self.unbridged:
            why.append(f"{self.unbridged} hole(s) could not be joined to the outline")
        if self.fell_back:
            why.append(f"ear clipping stuck with {self.stuck} vertices left (fanned, {self.fan_area:.3g} m2)")
        if self.clockwise:
            why.append(f"{self.clockwise} reversed triangle(s)")
        if abs(self.area_error) > TRI_AREA_TOL_M2 + TRI_AREA_TOL_REL * self.area:
            why.append(f"triangles cover {self.area_error:+.3g} m2 more than the polygon's {self.area:.3g} m2")
        return "; ".join(why)


TRI_AREA_TOL_M2 = 1e-6            # a triangulation is exact within this area (plus TRI_AREA_TOL_REL x the polygon's:
TRI_AREA_TOL_REL = 1e-9           # float noise on a 10,000 m2 outline; the 13 catalogue landmarks' 30,402 are < 1e-12)
TRI_CLOCKWISE_M2 = 1e-9           # a triangle is clockwise when its signed area is below minus this
EAR_EDGE_TOL_M = 1e-9             # ear clipping: a vertex this close to an ear's edge line counts as on it


def triangulate_polygon(poly: np.ndarray) -> Triangulation:
    """Ear clipping for a simple CCW polygon (a few hundred vertices at most). Triangles are CCW.
    Also takes a weakly simple ring with bridged holes (vertices repeated at the bridges). The result is a list of
    index triples that also says whether it is exact (Triangulation): a caller that publishes the triangles checks
    it, because the fan the clipping ends with when it finds no ear is not a triangulation."""
    p = np.asarray(poly, float)
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if signed_area(p) < 0:
        p = p[::-1].copy()
    idx = list(range(len(p)))
    tris: list[tuple[int, int, int]] = []

    def is_ear(i0, i1, i2) -> bool:
        a, b, c = p[i0], p[i1], p[i2]
        if _cross2(b - a, c - b) <= 1e-12:
            return False
        # a vertex in the triangle or on its boundary blocks the ear -- "on" within EAR_EDGE_TOL_M of each edge's line:
        # a vertex exactly on the ear's diagonal came out at -3e-14 (float noise), the ear was cut through it and the
        # last triangle left was reversed (six courtyards in an exact grid with rounded corners, D34)
        ab, bc, ca = b - a, c - b, a - c
        tab, tbc, tca = (-EAR_EDGE_TOL_M * math.hypot(e[0], e[1]) for e in (ab, bc, ca))
        for j in idx:
            if j in (i0, i1, i2):
                continue
            q = p[j]
            if (np.abs(q - a).max() < 1e-9 or np.abs(q - b).max() < 1e-9 or np.abs(q - c).max() < 1e-9):
                continue                              # a bridge duplicate of a corner: not inside
            if _cross2(ab, q - a) >= tab and _cross2(bc, q - b) >= tbc and _cross2(ca, q - c) >= tca:
                return False
        return True

    while len(idx) > 3:
        for k in range(len(idx)):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % len(idx)]
            if is_ear(i0, i1, i2):
                tris.append((i0, i1, i2))
                del idx[k]
                break
        else:
            break                                     # no ear (collinear rest, or not a simple ring): fan the rest
    out = Triangulation(tris)
    out.stuck = len(idx) if len(idx) > 3 else 0
    n_fan = max(len(idx) - 2, 0)
    for k in range(1, len(idx) - 1):
        out.append((idx[0], idx[k], idx[k + 1]))
    out.area = abs(signed_area(p)) if len(p) >= 3 else 0.0
    if out:
        t = np.array(out, int)
        a, b, c = p[t[:, 0]], p[t[:, 1]], p[t[:, 2]]
        signed = 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        out.clockwise = int((signed < -TRI_CLOCKWISE_M2).sum())
        out.area_error = float(np.abs(signed).sum() - out.area)
        if out.stuck:
            out.fan_area = float(np.abs(signed[len(signed) - n_fan:]).sum())
    return out


def _orient(a, b, c):
    return (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])


def _segments_cross(p1, p2, q1, q2) -> np.ndarray:
    """Proper crossing of segment p1-p2 with each segment q1[i]-q2[i] (touching does not count)."""
    d1, d2 = _orient(q1, q2, p1), _orient(q1, q2, p2)
    d3, d4 = _orient(p1, p2, q1), _orient(p1, p2, q2)
    return (d1 * d2 < -1e-12) & (d3 * d4 < -1e-12)


def ring_is_simple(poly: np.ndarray) -> bool:
    """No two non-adjacent edges of the closed ring cross."""
    p = np.asarray(poly, float)
    n = len(p)
    if n < 3:
        return False
    q = np.roll(p, -1, axis=0)
    for i in range(n - 2):
        j = np.arange(i + 2, n if i > 0 else n - 1)
        if len(j) and _segments_cross(p[i], q[i], p[j], q[j]).any():
            return False
    return True


def clean_ring(ring: Any, min_area_m2: float = 4.0) -> np.ndarray | None:
    """A usable outline ring (CCW, no closing vertex, no repeated vertices) or None when the input is
    missing, too small, degenerate or self-intersecting."""
    if ring is None:
        return None
    try:
        p = np.asarray(ring, float).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) > 1:
        keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > 1e-6]
        p = p[keep]
    if len(p) > 1 and np.allclose(p[0], p[-1]):
        p = p[:-1]
    if len(p) < 3 or abs(signed_area(p)) < min_area_m2 or not ring_is_simple(p):
        return None
    return ensure_ccw(p)


def ring_crosses_itself(ring: np.ndarray) -> bool:
    """Does the ring cross itself? Two ways to tell, either is enough:
    - two of its edges cross (not ring_is_simple);
    - one of its lobes winds against the ring's own direction: its ear clipping has a clockwise triangle.

    The second finds what the first cannot see, a crossing that is not a crossing of two edges: at a point the ring
    passes twice (a way through one node twice), at a vertex that lies on another edge, along a stretch it runs
    twice. Why ear clipping tells: clipped ears and the fan add up, point by point, to how often the ring winds
    round that point (each ear takes its own triangle off the ring), so where the ring winds the other way a
    triangle must turn clockwise. A ring that only touches itself (pinched at a point, both lobes the same way
    round) has none, and neither has a simple ring, on which ear clipping finishes (none of the 5,080 valid
    polygons of the random tests came out differently with this test)."""
    p = np.asarray(ring, float)
    return not ring_is_simple(p) or bool(triangulate_polygon(p).clockwise)


def _in_corner(ring: list, i: int, m: np.ndarray) -> bool:
    """Does the segment from vertex i of a CCW ring to m leave that vertex inside the ring's corner there (left of
    both edges at a convex corner, of at least one at a reflex or straight one)? A bridge repeats the vertex it
    lands on, and the copies stand at the same point with different corners: only the copy whose corner holds
    the next bridge keeps the ring weakly simple. Strict: a segment along one of the corner's edges is outside."""
    v, prev, nxt = ring[i], ring[i - 1], ring[(i + 1) % len(ring)]
    d = m - v
    a, b = _cross2(v - prev, d), _cross2(nxt - v, d)
    if _cross2(v - prev, nxt - v) > 0:
        return a > 0 and b > 0
    return a > 0 or b > 0


def _passes_through(m: np.ndarray, v: np.ndarray, pts: np.ndarray) -> bool:
    """Does the open segment m-v pass through one of the points (within 1e-9 m)?"""
    d = v - m
    ln = float(np.linalg.norm(d))
    if ln < 1e-12:
        return False
    rel = pts - m
    along = rel @ d / ln
    off = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0]) / ln
    return bool(((off < 1e-9) & (along > 1e-9) & (along < ln - 1e-9)).any())


def bridge_holes(outer: np.ndarray, holes: list[np.ndarray], left_out: list[int] | None = None) -> np.ndarray:
    """One weakly simple CCW ring from a CCW outer ring and holes (each cut open along a bridge to a
    visible vertex), for ear clipping. Holes are traversed clockwise. A hole that finds no bridge, or that crosses
    itself (ring_crosses_itself: two of its edges cross, or one of its lobes winds the other way), is not cut: its
    index in `holes` is appended to left_out when the caller gives one (and must then act on it); otherwise a
    RuntimeWarning says so (never silently, D34).

    A bridge is the segment from the hole's rightmost vertex to the nearest ring vertex that (a) crosses no edge
    and has its midpoint in the polygon, (b) leaves both its ends inside the corner there (_in_corner: of the
    copies an earlier bridge made of a ring vertex, only one qualifies; and never along an edge of the hole
    itself) and (c) passes through no other vertex. (b) and (c) are D34: without them openings in exact rows
    and columns -- every regular facade -- gave bridges along the next opening's edge, and the ring folded."""
    ring = [np.asarray(v, float) for v in ensure_ccw(outer)]
    order = sorted(range(len(holes)), key=lambda k: -float(np.asarray(holes[k], float)[:, 0].max()))
    pending = [ensure_ccw(holes[k])[::-1].copy() for k in order]
    while pending:
        h, hole_index = pending.pop(0), order.pop(0)
        m = int(np.argmax(h[:, 0]))
        M = h[m]
        R = np.array(ring)
        edges = [(R, np.roll(R, -1, axis=0))] + [(o, np.roll(o, -1, axis=0)) for o in [h] + pending]
        vertices = np.vstack([R, h, *pending])
        best = None
        # A hole that crosses itself is not cut. Bridged, one of its lobes winds the other way and is covered twice
        # by triangles that all turn the right way: nothing in Triangulation can see that (13 of 20 such stars were
        # called exact, each with 0.04-1.2 m2 under two layers; so were 54 of 100 holes that cross themselves at a
        # node, where no two edges cross). clean_ring, which both product paths run first, drops only a ring whose
        # edges cross: a courtyard way through one of its nodes twice reaches this function
        crosses = ring_crosses_itself(h)
        for idx in () if crosses else np.argsort(np.linalg.norm(R - M, axis=1)):
            V = R[idx]
            blocked = False
            for a, b in edges:
                touch = (np.abs(a - V).max(axis=1) < 1e-9) | (np.abs(b - V).max(axis=1) < 1e-9) \
                    | (np.abs(a - M).max(axis=1) < 1e-9) | (np.abs(b - M).max(axis=1) < 1e-9)
                if (_segments_cross(M, V, a, b) & ~touch).any():
                    blocked = True
                    break
            if blocked:
                continue
            mid = (M + V) / 2
            if not points_in_polygon(mid[None], outer)[0] or any(points_in_polygon(mid[None], o)[0] for o in [h] + pending):
                continue
            if not _in_corner(ring, int(idx), M):
                # a vertex an earlier bridge repeated: this copy's corner does not hold the bridge (taking it twisted
                # the ring -- 37 of 1500 rotated courtyard grids, 85 of 1021 walls with two rows of openings, D34)
                continue
            if not _in_corner(h, m, V) or _passes_through(M, V, vertices):
                # along one of the hole's own edges, or through (or along the edge at) a vertex in between: the
                # crossing test above sees neither (two bays with a window over one, tests/test_relief.py).
                # _passes_through is the test that decides: with it and without the corner test at the hole's own
                # vertex, 3,600 random polygons and every test come out the same (a bridge along the hole's edge
                # passes through that edge's far vertex; one into the hole has its midpoint in it); the other way
                # round, 39 of the 3,580 valid ones are wrong. The corner test at the hole's end is belt and
                # braces, kept because it is rule (b) there and the two tests do not share a tolerance
                continue
            best = int(idx)
            break
        if best is None:                # (a hole on or outside the outline, or crossing itself): it is not cut
            if left_out is not None:
                left_out.append(hole_index)
            else:
                warnings.warn(f"bridge_holes: hole {hole_index} has no bridge to the outline and is not cut"
                              + (" (it crosses itself)" if crosses else ""), RuntimeWarning, stacklevel=2)
            continue
        hole_walk = [h[(m + k) % len(h)] for k in range(len(h))] + [M.copy()]
        ring = ring[:best + 1] + hole_walk + [ring[best].copy()] + ring[best + 1:]
    return np.array(ring)


def triangulate_with_holes(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> tuple[np.ndarray, Triangulation]:
    """(vertices, CCW triangles) of a polygon with holes. The triangles say whether they are exact
    (Triangulation.exact: ear clipping finished without a fan, no clockwise triangle, every hole cut)."""
    left_out: list[int] = []
    ring = bridge_holes(outer, holes, left_out) if holes else ensure_ccw(outer)
    tris = triangulate_polygon(ring)
    tris.unbridged = len(left_out)
    return ring, tris


def distance_to_rings(xy: np.ndarray, rings: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """distance_to_outline over several rings (outer + holes): distance, closest point, ring, edge."""
    best = None
    for r, ring in enumerate(rings):
        d, c, e = distance_to_outline(xy, ring)
        if best is None:
            best = [d, c, np.zeros(len(d), int), e]
            continue
        closer = d < best[0]
        best[0] = np.where(closer, d, best[0])
        best[1] = np.where(closer[:, None], c, best[1])
        best[2] = np.where(closer, r, best[2])
        best[3] = np.where(closer, e, best[3])
    return best[0], best[1], best[2], best[3]
