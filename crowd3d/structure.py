"""Structure-first massing: OpenStreetMap Simple 3D Buildings parts as parametric parts (docs/STRUCTURE_FIRST.md §2.4).

Hover's "semantic wireframe first, photos second", with the wireframe that volunteers already drew for landmarks:
every OSM building:part becomes a parametric solid (a prism from min_height to the eave plus a roof primitive
chosen by roof:shape). Shapes come from facts only (rule 2); where a tag is missing the value used and its
source are recorded per parameter; where a primitive cannot be built the fallback is explicit.

Pipeline (load_osm_structure):
 1. parse the OSM XML snapshot (nodes, ways, multipolygon relations; rings assembled from way segments);
 2. keep building:part pieces whose centroid lies in the main outline buffered 2 m AND that have >= 50 % of their
    area inside that zone (the second test removes ring- or U-shaped neighbours such as a terrace whose centroid
    falls inside the landmark); building:part=no is dropped;
 3. resolve heights with provenance: height / building:levels, min_height / building:min_level, roof:height /
    roof:levels / roof:angle; roof:shape without a roof height takes the shape's conventional proportion
    (s3db-shape-default, D17), never on a 3 m stub; Wikidata's total height is never used for a part;
 4. main-outline area no part covers (S3DB renders only parts): the main outline's own height tag, else the
    min_height of the raised parts resting on it (>= 50 % of their base over the gap; osm-s3db-support), else a 3 m
    'height undetermined' stub (D13); gaps under 20 m2 without any height fact are not drawn;
 5. build every part as a closed solid: walls whose tops follow the roof boundary (gable ends carry a top profile),
    a roof primitive (flat; skillion; gabled / saltbox / gambrel / round as a profile across the ridge; hipped /
    half-hipped / mansard as the lower envelope of planes rising from the support's edges -- the straight
    skeleton of a convex part; pyramidal / cone as a fan to an apex; dome / onion as rings of facets) and a base.
    Each primitive has an explicit fallback chain (the original shape and the reason are recorded);
 6. a parameter registry (massing.json v2: one source and state per parameter, evidence, tickets) and the
    structure QA checks that need no photos (G2 forbidden sources, G3 filter, G6 stubs, G10 support).

Pure numpy + OpenCV (the repo has no shapely): polygon booleans the filter and the gap fill need are done on a
0.25 m raster; roof faces of non-convex parts are the part's ear-clipping triangles clipped by the envelope's
convex regions (exact planes, T-junctions inserted so the solid closes).

Coordinates: the landmark ENU frame (x east, y north, z up, metres), the frame of stylize.SceneInput.
"""
from __future__ import annotations

import json
import math
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import stylize as st
from .geo import LocalENU

# ---------------------------------------------------------------- parameters, sources, states

# geometry states (docs/STRUCTURE_FIRST.md §2.4.6, §2.7)
STATE_MAP = "map"                    # a tagged fact
STATE_CONVENTION = "map-convention"  # derived from facts by a recorded convention (roof pitch, support)
STATE_VERIFIED = "photo-verified"    # photos agree with the map value (value unchanged)
STATE_FITTED = "photo-fitted"        # changed by photo evidence (D15: >= 2 agreeing channels, automatic, reversible)
STATE_CORRECTED = "photo-corrected"  # D25: strong photo evidence corrected the map (crowd3d.correct): the full photo
                                     # value or its guaranteed lower bound, map value + evidence + registration box kept
STATE_DISPUTED = "disputed"          # >= 2 evidence channels disagree in one direction: map value kept, ticket
STATE_CURATOR = "curator"            # a curator's value
STATE_CARVED = "carved"              # height from sky carving + points where no part exists (not built here)
STATE_STUB = "stub"                  # no height fact at all: the 3 m 'height undetermined' stub (D13)
STATES = (STATE_MAP, STATE_CONVENTION, STATE_VERIFIED, STATE_FITTED, STATE_CORRECTED, STATE_DISPUTED, STATE_CURATOR,
          STATE_CARVED, STATE_STUB)
# sources a part parameter may have; anything else (Wikidata's total height, monocular depth as metric, a
# generative model) is forbidden (G2)
SOURCES = (st.PROV_S3DB, st.PROV_LEVELS, st.PROV_ROOF_ANGLE, st.PROV_SHAPE_DEFAULT, st.PROV_SUPPORT, st.PROV_GROUND,
           st.PROV_OSM_HEIGHT, st.PROV_OSM, st.PROV_DEFAULT, st.PROV_CURATOR, st.PROV_FITTED, st.PROV_FITTED_AUTO,
           st.PROV_CORRECTED_AUTO, st.PROV_CORRECTED_LB, st.PROV_CORRECTED_UB)
FORBIDDEN_SOURCES = (st.PROV_WIKIDATA, "monocular-depth", "generative")
STATE_OF_SOURCE = {st.PROV_S3DB: STATE_MAP, st.PROV_LEVELS: STATE_MAP, st.PROV_ROOF_ANGLE: STATE_MAP,
                   st.PROV_OSM_HEIGHT: STATE_MAP, st.PROV_OSM: STATE_MAP, st.PROV_GROUND: STATE_MAP,
                   st.PROV_SHAPE_DEFAULT: STATE_CONVENTION, st.PROV_SUPPORT: STATE_CONVENTION,
                   st.PROV_DEFAULT: STATE_STUB, st.PROV_CURATOR: STATE_CURATOR, st.PROV_FITTED: STATE_FITTED,
                   st.PROV_FITTED_AUTO: STATE_FITTED, st.PROV_CORRECTED_AUTO: STATE_CORRECTED,
                   st.PROV_CORRECTED_LB: STATE_CORRECTED, st.PROV_CORRECTED_UB: STATE_CORRECTED}

SHAPES = ("flat", "skillion", "gabled", "half-hipped", "hipped", "pyramidal", "cone", "dome", "onion", "round",
          "gambrel", "mansard", "saltbox")
ALIASES = {"conical": "cone", "side_hipped": "half-hipped", "half_hipped": "half-hipped", "spherical": "dome",
           "pyramid": "pyramidal", "pitched": "gabled", "shed": "skillion", "lean_to": "skillion", "barrel": "round",
           "domed": "dome", "hip": "hipped"}
# explicit fallback chains (design §2.4.1 item 4): what is tried, in order
FALLBACKS = {
    "gabled": ["gabled", "flat"], "saltbox": ["saltbox", "gabled", "flat"], "gambrel": ["gambrel", "gabled", "flat"],
    "round": ["round", "gabled", "flat"], "hipped": ["hipped", "gabled", "flat"],
    "half-hipped": ["half-hipped", "hipped", "gabled", "flat"], "mansard": ["mansard", "hipped", "gabled", "flat"],
    "pyramidal": ["pyramidal", "flat"], "cone": ["cone", "pyramidal", "flat"], "dome": ["dome", "flat"],
    "onion": ["onion", "dome", "flat"], "skillion": ["skillion", "flat"], "flat": ["flat"],
}
OSM_ATTRIBUTION = "© OpenStreetMap contributors, ODbL 1.0 (building:part, Simple 3D Buildings)"


@dataclass(frozen=True)
class StructureParams:
    level_m: float = 3.0                 # metres per level (building:levels, roof:levels; the S3DB convention)
    keep_buffer_m: float = 2.0           # a part's centroid within the main outline buffered this much ...
    keep_area_share: float = 0.5         # ... and at least this share of its area inside that zone (G3)
    rect_min: float = 0.85               # area / minimum rotated rectangle for ridge (OBB) roofs
    convex_min: float = 0.985            # area / convex hull area for skeleton roofs over the hull
    dome_rings: int = 6                  # facet rings of a dome
    round_facets: int = 6                # facets across a barrel roof
    min_piece_m2: float = 0.05           # smaller pieces are dropped (a 0.3 m column is 0.07 m2)
    small_piece_m2: float = 3.0          # pieces below this are simplified to ...
    small_piece_vertices: int = 8        # ... at most this many vertices (toy scale)
    clean_tol_m: float = 0.05            # vertex clean-up (OSM node noise): near-collinear within this
    raster_m: float = 0.25               # raster for the gap fill (main outline minus parts)
    remainder_min_m2: float = 1.0        # gap pieces from this size become parts
    remainder_stub_min_m2: float = 20.0  # ... but a gap with no height fact (3 m stub) only from this size
    stub_m: float = 3.0                  # 'height undetermined' stub (D13)
    support_overlap: float = 0.5         # a raised part rests on a gap when this share of the 0.5 m band around it is gap
    conventional_roofs: bool = True      # D17: roof:shape without a roof height takes its conventional proportion
    stub_conventional_roofs: bool = False   # D17 recommendation: never on a 3 m stub
    spire_cone_rule: bool = False        # D17 recommendation: off (0.9 x the room on parts <= 12 m across)
    pitch_deg: float = 30.0              # the conventional pitch
    roof_room_share: float = 0.9         # conventions take at most this share of the part's wall room
    inset_step_m: float = 0.01           # lower overlapping parts shrink 1-3 cm (no coplanar z-fighting)
    max_plausible_m: float = 1000.0      # sanity bound: no building part's top is higher (the tallest building is 828 m);
                                         # above it a ticket opens (the value is never changed)
    wikidata_bound: float = 1.1          # a part's top above Wikidata's total height (P2048) x this: a ticket (never a change)
    outlier_ratio: float = 2.5           # a part's top above this x every other part's top: a ticket (13 landmarks: <= 1.51;
                                         # a vandalised height=300 on the Baptistery's lantern: 10.0)
    convention_share: float = 0.7        # a conventional roof (no roof:height) taking more than this share of its part's
                                         # height: a ticket (a wrong roof:shape?), never a change (13 landmarks: only
                                         # Frauenkirche's 4 corner onions, 0.9 of a 4 m drum)
    support_gap_m: float = 0.3           # G10: a base sample is supported when a surface under it is at most this far below
    support_share: float = 0.5           # G10: ... on at least this share of the base


DEFAULT_STRUCTURE = StructureParams()


class Fallback(Exception):
    """A roof primitive cannot be built for this part (reason in the message)."""


# ---------------------------------------------------------------- OSM parsing


def _tags(el) -> dict[str, str]:
    return {t.get("k"): t.get("v") for t in el.findall("tag")}


@dataclass
class OsmData:
    nodes: dict[str, tuple[float, float]]                                  # id -> (lat, lon)
    ways: dict[str, tuple[list[str], dict[str, str]]]                      # id -> (node ids, tags)
    rels: dict[str, tuple[list[tuple[str, str, str]], dict[str, str]]]     # id -> ([(type, ref, role)], tags)
    meta: dict[str, Any]


def parse_osm(src: str | Path | bytes) -> OsmData:
    """An OSM XML snapshot (API 0.6 map / overpass 'out meta'). Entity and DTD declarations are refused (the file
    is data, never markup to expand)."""
    data = src if isinstance(src, bytes) else Path(src).read_bytes()
    head = data[:4096].upper()
    if b"<!DOCTYPE" in head or b"<!ENTITY" in data.upper():
        raise ValueError("OSM snapshot with a DTD or entity declaration refused")
    root = ET.fromstring(data)
    nodes = {n.get("id"): (float(n.get("lat")), float(n.get("lon"))) for n in root.iter("node")
             if n.get("lat") is not None and n.get("lon") is not None}
    ways = {w.get("id"): ([nd.get("ref") for nd in w.findall("nd")], _tags(w)) for w in root.iter("way")}
    rels = {r.get("id"): ([(m.get("type"), m.get("ref"), m.get("role") or "") for m in r.findall("member")], _tags(r))
            for r in root.iter("relation")}
    stamps = [el.get("timestamp") for el in root if el.get("timestamp")]
    meta = {"file": Path(src).name if not isinstance(src, bytes) else None, "generator": root.get("generator"),
            "api_version": root.get("version"), "newest_element": max(stamps) if stamps else None,
            "licence": "ODbL-1.0"}
    return OsmData(nodes, ways, rels, meta)


def assemble_rings(node_lists: list[list[str]]) -> tuple[list[list[str]], int]:
    """Join way segments end to end into closed rings. Returns (rings, number of open fragments)."""
    segs = [list(s) for s in node_lists if len(s) >= 2]
    rings, open_ = [], 0
    while segs:
        cur = segs.pop(0)
        grown = True
        while cur[0] != cur[-1] and grown:
            grown = False
            for i, s in enumerate(segs):
                if s[0] == cur[-1]:
                    cur = cur + s[1:]
                elif s[-1] == cur[-1]:
                    cur = cur + s[::-1][1:]
                elif s[-1] == cur[0]:
                    cur = s + cur[1:]
                elif s[0] == cur[0]:
                    cur = s[::-1] + cur[1:]
                else:
                    continue
                segs.pop(i)
                grown = True
                break
        if cur[0] == cur[-1] and len(cur) >= 4:
            rings.append(cur)
        else:
            open_ += 1
    return rings, open_


NUMERIC_TAGS = ("height", "min_height", "roof:height", "building:levels", "building:min_level", "roof:levels",
                "roof:angle")
LENGTH_TAGS = ("height", "min_height", "roof:height")
_NUM = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*(m|meters?|metres?|ft|feet|'|)\s*$", re.I)
_FTIN = re.compile(r"^\s*(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"\s*$")
_COMPASS = {"N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5, "S": 180,
            "SSW": 202.5, "SW": 225, "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5}


def length_m(v: str | None) -> float | None:
    """OSM length value in metres ('12', '12 m', '40 ft', "40'", "12'6\""); None when absent or unparseable."""
    if v is None:
        return None
    m = _FTIN.match(v)
    if m:
        return float(m.group(1)) * 0.3048 + float(m.group(2)) * 0.0254
    m = _NUM.match(v)
    if not m:
        return None
    x = float(m.group(1).replace(",", "."))
    return x * 0.3048 if m.group(2).lower() in ("ft", "feet", "'") else x


def number(v: str | None) -> float | None:
    try:
        x = float(str(v).replace(",", ".").strip()) if v is not None else None
    except ValueError:
        return None
    return x if x is not None and math.isfinite(x) else None


def direction_deg(v: str | None) -> float | None:
    """roof:direction as degrees clockwise from north ('NE' or '45')."""
    if v is None:
        return None
    if v.strip().upper() in _COMPASS:
        return float(_COMPASS[v.strip().upper()])
    x = number(v)
    return None if x is None else x % 360.0


# ---------------------------------------------------------------- 2-D geometry (numpy, no shapely)


def ring_area(ring: np.ndarray) -> float:
    return st.signed_area(ring)


def poly_area(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> float:
    return abs(ring_area(outer)) - sum(abs(ring_area(h)) for h in holes or [])


def ccw(ring: np.ndarray) -> np.ndarray:
    return st.ensure_ccw(np.asarray(ring, float))


def cw(ring: np.ndarray) -> np.ndarray:
    return ccw(ring)[::-1].copy()


def convex_hull(pts: np.ndarray) -> np.ndarray:
    """Andrew's monotone chain; CCW, float64."""
    P = np.unique(np.asarray(pts, float).reshape(-1, 2), axis=0)
    if len(P) < 3:
        return P

    def half(points):
        h: list[np.ndarray] = []
        for p in points:
            while len(h) >= 2 and st._cross2(h[-1] - h[-2], p - h[-2]) <= 1e-12:
                h.pop()
            h.append(p)
        return h

    lower, upper = half(P), half(P[::-1])
    return np.array(lower[:-1] + upper[:-1])


@dataclass
class Obb:
    """Minimum rotated rectangle: centre, long-axis unit u, short-axis unit v (u x v = +z), half-length L, half-width W."""

    c: np.ndarray
    u: np.ndarray
    v: np.ndarray
    L: float
    W: float

    def corners(self) -> np.ndarray:
        c, u, v, L, W = self.c, self.u, self.v, self.L, self.W
        return np.array([c - L * u - W * v, c + L * u - W * v, c + L * u + W * v, c - L * u + W * v])

    @property
    def area(self) -> float:
        return 4.0 * self.L * self.W


def obb(ring: np.ndarray) -> Obb:
    h = convex_hull(ring)
    best = None
    for i in range(len(h)):
        e = h[(i + 1) % len(h)] - h[i]
        n = float(np.linalg.norm(e))
        if n < 1e-9:
            continue
        u = e / n
        v = np.array([-u[1], u[0]])
        pu, pv = h @ u, h @ v
        area = float((pu.max() - pu.min()) * (pv.max() - pv.min()))
        if best is None or area < best[0] - 1e-12:
            best = (area, u, v, pu.min(), pu.max(), pv.min(), pv.max())
    if best is None:
        c = np.asarray(ring, float).mean(axis=0)
        return Obb(c, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0.0)
    _, u, v, u0, u1, v0, v1 = best
    c = u * (u0 + u1) / 2 + v * (v0 + v1) / 2
    L, W = (u1 - u0) / 2, (v1 - v0) / 2
    if W > L:
        u, v, L, W = v, -u, W, L
    return Obb(c, u, v, float(L), float(W))


def rectangularity(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> float:
    return float(poly_area(outer, holes) / max(obb(outer).area, 1e-9))


def convexity(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> float:
    return float(poly_area(outer, holes) / max(abs(ring_area(convex_hull(outer))), 1e-9))


def clip_halfplane(ring: np.ndarray, a: np.ndarray, c: float) -> np.ndarray:
    """Sutherland-Hodgman: the part of a convex polygon (n, 2) where a . p + c <= 0."""
    out = []
    n = len(ring)
    for i in range(n):
        p, q = ring[i], ring[(i + 1) % n]
        fp, fq = float(a @ p + c), float(a @ q + c)
        if fp <= 1e-12:
            out.append(p)
        if (fp < -1e-12 and fq > 1e-12) or (fp > 1e-12 and fq < -1e-12):
            t = fp / (fp - fq)
            out.append(p + t * (q - p))
    return np.array(out) if len(out) >= 3 else np.zeros((0, 2))


def clip_convex(subject: np.ndarray, clipper: np.ndarray) -> np.ndarray:
    """A convex polygon clipped by a convex CCW polygon (both (n, 2))."""
    out = np.asarray(subject, float)
    m = len(clipper)
    for i in range(m):
        if len(out) < 3:
            return np.zeros((0, 2))
        a, b = clipper[i], clipper[(i + 1) % m]
        d = b - a
        nrm = np.array([d[1], -d[0]])              # keep the left side of a->b: nrm . (p - a) <= 0
        out = clip_halfplane(out, nrm, -float(nrm @ a))
    return out if len(out) >= 3 else np.zeros((0, 2))


def dedupe_ring(ring: np.ndarray, tol: float = 1e-7) -> np.ndarray:
    p = np.asarray(ring, float).reshape(-1, 2)
    if len(p) < 2:
        return p
    keep = np.r_[True, np.linalg.norm(np.diff(p, axis=0), axis=1) > tol]
    p = p[keep]
    if len(p) > 1 and np.linalg.norm(p[0] - p[-1]) <= tol:
        p = p[:-1]
    return p


def drop_collinear(ring: np.ndarray, tol: float) -> np.ndarray:
    """Remove vertices within tol of the line through their neighbours (repeatedly)."""
    p = dedupe_ring(ring)
    changed = True
    while changed and len(p) > 3:
        changed = False
        n = len(p)
        for i in range(n):
            a, b, c = p[i - 1], p[i], p[(i + 1) % n]
            ac = c - a
            L = float(np.linalg.norm(ac))
            d = abs(st._cross2(ac, b - a)) / L if L > 1e-12 else float(np.linalg.norm(b - a))
            if d < tol and (L < 1e-12 or 0.0 <= float((b - a) @ ac) / (L * L) <= 1.0):
                p = np.delete(p, i, axis=0)
                changed = True
                break
    return p


def clean_ring(ring: np.ndarray, params: StructureParams = DEFAULT_STRUCTURE) -> np.ndarray | None:
    """CCW, without duplicate or near-collinear vertices, simple and at least min_piece_m2; small pieces (columns,
    finials) simplified to at most small_piece_vertices vertices. None when unusable."""
    p = drop_collinear(ring, params.clean_tol_m)
    if len(p) < 3 or abs(ring_area(p)) < params.min_piece_m2 or not st.ring_is_simple(p):
        return None
    p = ccw(p)
    area = abs(ring_area(p))
    if area < params.small_piece_m2 and len(p) > params.small_piece_vertices:
        tol = params.clean_tol_m
        while len(p) > params.small_piece_vertices and tol < 1.0:
            tol *= 1.5
            q = st.simplify_polygon(p, tol)
            if len(q) >= 3 and abs(ring_area(q)) > 0.7 * area and st.ring_is_simple(q):
                p = q
            else:
                break
    return ccw(p)


def inset_ring(ring: np.ndarray, d: float) -> np.ndarray | None:
    """Mitre offset of a CCW ring inwards by d (small d only); None when it degenerates."""
    p = ccw(ring)
    n = len(p)
    out = []
    for i in range(n):
        a0, a1, a2 = p[i - 1], p[i], p[(i + 1) % n]
        t1, t2 = a1 - a0, a2 - a1
        t1 /= max(np.linalg.norm(t1), 1e-12)
        t2 /= max(np.linalg.norm(t2), 1e-12)
        n1, n2 = np.array([-t1[1], t1[0]]), np.array([-t2[1], t2[0]])     # inward (left) normals of a CCW ring
        den = st._cross2(t1, t2)
        if abs(den) < 1e-6:
            out.append(a1 + d * n1)
            continue
        q1, q2 = a0 + d * n1, a1 + d * n2
        s = st._cross2(q2 - q1, t2) / den
        v = q1 + s * t1
        if np.linalg.norm(v - a1) > 10 * d + 1e-9:                 # a very sharp spike: do not mitre
            v = a1 + d * (n1 + n2) / max(np.linalg.norm(n1 + n2), 1e-12)
        out.append(v)
    q = np.array(out)
    if abs(ring_area(q)) < 0.5 * abs(ring_area(p)) or not st.ring_is_simple(q) or ring_area(q) <= 0:
        return None
    return q


def point_visible(c: np.ndarray, q: np.ndarray, rings: list[np.ndarray]) -> bool:
    """Does the segment c -> q (shortened a hair at q) stay clear of every ring edge?"""
    end = c + 0.999 * (q - c)
    for r in rings:
        a, b = r, np.roll(r, -1, axis=0)
        if st._segments_cross(c, end, a, b).any():
            return False
    return True


def pole_of_inaccessibility(outer: np.ndarray, holes: list[np.ndarray] | None = None, cells: int = 200) -> np.ndarray:
    """The interior point farthest from the outline (raster distance transform)."""
    lo, hi = outer.min(axis=0), outer.max(axis=0)
    res = max(float((hi - lo).max()) / cells, 1e-3)
    g = _Grid(lo - 2 * res, hi + 2 * res, res)
    m = g.fill(outer, holes)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    iy, ix = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return g.centre(ix, iy)


def star_centre(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> np.ndarray | None:
    """A point from which every outline vertex is visible inside the polygon (centroid first, then the pole of
    inaccessibility); None for polygons with holes or no such point."""
    if holes:
        return None
    for c in (st.polygon_centroid(outer), pole_of_inaccessibility(outer)):
        if not st.points_in_polygon(c[None], outer)[0]:
            continue
        if all(point_visible(c, q, [outer]) for q in outer):
            return np.asarray(c, float)
    return None


class _Grid:
    """A raster over [lo, hi) with square cells of `res` metres; row 0 = south."""

    def __init__(self, lo: np.ndarray, hi: np.ndarray, res: float) -> None:
        self.lo, self.res = np.asarray(lo, float), float(res)
        self.nx = max(1, int(math.ceil((hi[0] - lo[0]) / res)))
        self.ny = max(1, int(math.ceil((hi[1] - lo[1]) / res)))

    def px(self, ring: np.ndarray) -> np.ndarray:
        return np.round((np.asarray(ring, float) - self.lo) / self.res * 8.0 - 0.5 * 8).astype(np.int32)

    def fill(self, outer: np.ndarray, holes: list[np.ndarray] | None = None) -> np.ndarray:
        m = np.zeros((self.ny, self.nx), np.uint8)
        cv2.fillPoly(m, [self.px(outer)], 1, lineType=cv2.LINE_8, shift=3)
        for h in holes or []:
            cv2.fillPoly(m, [self.px(h)], 0, lineType=cv2.LINE_8, shift=3)
        return m

    def centre(self, ix: float, iy: float) -> np.ndarray:
        return self.lo + (np.array([ix, iy], float) + 0.5) * self.res

    def centres(self) -> np.ndarray:
        ys, xs = np.mgrid[0:self.ny, 0:self.nx]
        return self.lo[None, None, :] + (np.stack([xs, ys], -1) + 0.5) * self.res

    def polygons(self, mask: np.ndarray, min_area_m2: float, eps_cells: float = 1.0
                 ) -> list[tuple[np.ndarray, list[np.ndarray]]]:
        """Contours of a mask as [(outer CCW, [holes CW])] in metres (simplified by eps_cells)."""
        contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        if hier is None:
            return []

        def to_xy(c):
            c = cv2.approxPolyDP(c, eps_cells, True)[:, 0, :].astype(float)
            return self.lo + (c + 0.5) * self.res

        out = []
        for i, c in enumerate(contours):
            if hier[0][i][3] != -1:
                continue
            outer = to_xy(c)
            if len(outer) < 3 or abs(ring_area(outer)) < min_area_m2:
                continue
            holes = []
            j = hier[0][i][2]
            while j != -1:
                h = to_xy(contours[j])
                if len(h) >= 3 and abs(ring_area(h)) >= max(min_area_m2, 1.0):
                    holes.append(cw(h))
                j = hier[0][j][0]
            out.append((ccw(outer), holes))
        return out


def share_inside(outer: np.ndarray, holes: list[np.ndarray], zone_rings: list[np.ndarray], buffer_m: float,
                 res: float = 0.25) -> float:
    """Share of a polygon's area inside zone (outer zone_rings[0], holes zone_rings[1:]) buffered by buffer_m."""
    lo, hi = outer.min(axis=0), outer.max(axis=0)
    res = min(res, max(float((hi - lo).max()) / 40.0, 0.02))
    g = _Grid(lo, hi + res, res)
    m = g.fill(outer, holes) > 0
    if not m.any():
        return 0.0
    c = g.centres()[m]
    inside = st.points_in_polygon(c, zone_rings[0])
    for h in zone_rings[1:]:
        inside &= ~st.points_in_polygon(c, h)
    near = st.distance_to_rings(c, zone_rings)[0] <= buffer_m
    return float((inside | near).mean())


def in_zone(xy: np.ndarray, zone_rings: list[np.ndarray], buffer_m: float) -> bool:
    inside = bool(st.points_in_polygon(np.asarray(xy, float)[None], zone_rings[0])[0])
    for h in zone_rings[1:]:
        inside &= not bool(st.points_in_polygon(np.asarray(xy, float)[None], h)[0])
    return inside or float(st.distance_to_rings(np.asarray(xy, float)[None], zone_rings)[0][0]) <= buffer_m


def representative_point(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> np.ndarray:
    """The centroid when it lies in the polygon, else its pole of inaccessibility (a U-shaped part's centroid falls
    in its own notch)."""
    c = poly_centroid(outer, holes)
    inside = bool(st.points_in_polygon(c[None], outer)[0]) and not any(st.points_in_polygon(c[None], h)[0] for h in holes or [])
    return c if inside else pole_of_inaccessibility(outer, holes)


def poly_centroid(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> np.ndarray:
    rings = [ccw(outer)] + [cw(h) for h in holes or []]
    a_tot, c_tot = 0.0, np.zeros(2)
    for r in rings:
        a = ring_area(r)
        if abs(a) < 1e-12:
            continue
        a_tot += a
        c_tot += a * st.polygon_centroid(r)
    return c_tot / a_tot if abs(a_tot) > 1e-12 else np.asarray(outer, float).mean(axis=0)


# ---------------------------------------------------------------- faces


@dataclass
class Face:
    """A planar face: one or more planar polygons (k, 3) that share its plane, CCW seen along the outward normal."""

    polys: list[np.ndarray]
    normal: np.ndarray
    kind: str                            # 'roof' | 'wall' | 'base'
    label: str = ""

    def triangles(self) -> np.ndarray:
        out = []
        for poly in self.polys:
            out.extend(triangulate_planar(poly, self.normal))
        return np.array(out, float).reshape(-1, 3, 3)


def plane_basis(n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(n, float) / max(np.linalg.norm(n), 1e-12)
    t = np.cross([0.0, 0.0, 1.0], n) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t = t - (t @ n) * n
    t /= max(np.linalg.norm(t), 1e-12)
    return t, np.cross(n, t)


def triangulate_planar(poly: np.ndarray, normal: np.ndarray) -> list[np.ndarray]:
    """Triangles (3, 3) of a planar 3-D polygon (simple or weakly simple), wound CCW about `normal`."""
    p = np.asarray(poly, float)
    if len(p) < 3:
        return []
    t, b = plane_basis(normal)
    uv = np.stack([p @ t, p @ b], 1)
    order = np.arange(len(p))
    if ring_area(uv) < 0:
        order = order[::-1]
        uv = uv[::-1]
    tris = st.triangulate_polygon(uv)
    out = []
    for i, j, k in tris:
        tri = p[order[[i, j, k]]]
        if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) < 1e-10:
            continue
        out.append(tri)
    return out


def newell(r: np.ndarray) -> np.ndarray:
    n = np.zeros(3)
    for i in range(len(r)):
        a, b = r[i], r[(i + 1) % len(r)]
        n += [(a[1] - b[1]) * (a[2] + b[2]), (a[2] - b[2]) * (a[0] + b[0]), (a[0] - b[0]) * (a[1] + b[1])]
    ln = np.linalg.norm(n)
    return n / ln if ln > 1e-12 else n


def insert_t_junctions(faces: list[Face], tol: float = 1e-4) -> int:
    """Insert every vertex of a solid that lies inside another face's boundary edge into that edge (the welded
    mesh then has no T-junctions and closes). Returns the number of inserted vertices."""
    allv = np.unique(np.round(np.vstack([q for f in faces for q in f.polys]) / tol).astype(np.int64), axis=0) * tol
    added = 0
    for f in faces:
        for qi, r in enumerate(f.polys):
            out = []
            for i in range(len(r)):
                a, b = r[i], r[(i + 1) % len(r)]
                out.append(a)
                d = b - a
                L2 = float(d @ d)
                if L2 < tol * tol:
                    continue
                s = (allv - a) @ d / L2
                sel = (s > 1e-6) & (s < 1 - 1e-6)
                if not sel.any():
                    continue
                cand, ss = allv[sel], s[sel]
                near = np.linalg.norm(cand - (a + np.outer(ss, d)), axis=1) < tol
                if not near.any():
                    continue
                pts = cand[near][np.argsort(ss[near])]
                pts = [p for p in pts if np.linalg.norm(p - a) > tol and np.linalg.norm(p - b) > tol]
                out.extend(pts)
                added += len(pts)
            f.polys[qi] = np.array(out)
    return added


def watertight(tris: np.ndarray, tol: float = 1e-4) -> dict[str, int]:
    """Welded edge census: a closed 2-manifold with consistent winding has every directed edge once and its reverse."""
    if not len(tris):
        return {"edges": 0, "unmatched": 0, "duplicated": 0}
    pos = tris.reshape(-1, 3)
    key = np.round(pos / tol).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    t = vid.reshape(-1, 3)
    t = t[(t[:, 0] != t[:, 1]) & (t[:, 1] != t[:, 2]) & (t[:, 0] != t[:, 2])]
    e = np.concatenate([t[:, [0, 1]], t[:, [1, 2]], t[:, [2, 0]]])
    directed = Counter(map(tuple, e.tolist()))
    dup = sum(1 for c in directed.values() if c > 1)
    unmatched = sum(1 for (a, b) in directed if (b, a) not in directed)
    return {"edges": len(directed), "unmatched": unmatched, "duplicated": dup}


# ---------------------------------------------------------------- roof primitives


@dataclass
class Roof:
    faces: list[Face]
    shape: str                           # what was built
    height_m: float
    meta: dict[str, Any] = field(default_factory=dict)


def _lift(ring2: np.ndarray, zfun) -> np.ndarray:
    return np.c_[ring2, zfun(ring2)]


def _bridged(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    return st.bridge_holes(outer, holes) if holes else ccw(outer)


def flat_roof(outer: np.ndarray, holes: list[np.ndarray], z: float) -> Roof:
    ring = _bridged(outer, holes)
    return Roof([Face([_lift(ring, lambda r: np.full(len(r), z))], np.array([0.0, 0.0, 1.0]), "roof", "flat")], "flat", 0.0)


def _pieces(outer: np.ndarray, holes: list[np.ndarray]) -> list[np.ndarray]:
    """Convex pieces covering a polygon: itself when convex, else its ear-clipping triangles."""
    if not holes and convexity(outer) > 1.0 - 1e-9:
        return [ccw(outer)]
    verts, tris = st.triangulate_with_holes(ccw(outer), [cw(h) for h in holes])
    return [ccw(verts[[i, j, k]]) for i, j, k in tris if abs(ring_area(verts[[i, j, k]])) > 1e-9]


def envelope_roof(outer: np.ndarray, holes: list[np.ndarray], support: np.ndarray,
                  planes: list[tuple[np.ndarray, float]], z_eave: float, shape: str, labels: list[str] | None = None) -> Roof:
    """Roof surface z = z_eave + min_i (g_i . p + h_i) over a convex support (n, 2), cut to the part. Each plane's
    region of the lower envelope is the support clipped by the half-planes where that plane is the lowest; a face is
    the part's convex pieces clipped by that region (planar by construction)."""
    sup = ccw(support)
    pieces = _pieces(outer, holes)
    faces, zmax = [], 0.0
    for i, (gi, hi) in enumerate(planes):
        reg = sup.copy()
        for j, (gj, hj) in enumerate(planes):
            if i == j or len(reg) == 0:
                continue
            reg = clip_halfplane(reg, gi - gj, hi - hj)     # (g_i - g_j) . p + (h_i - h_j) <= 0
        if len(reg) < 3 or abs(ring_area(reg)) < 1e-6:
            continue
        polys = []
        for pc in pieces:
            q = clip_convex(pc, ccw(reg))
            q = dedupe_ring(q, 1e-9) if len(q) else q
            if len(q) < 3 or abs(ring_area(q)) < 1e-5:
                continue
            q3 = _lift(ccw(q), lambda r, gi=gi, hi=hi: z_eave + r @ gi + hi)
            zmax = max(zmax, float(q3[:, 2].max()) - z_eave)
            polys.append(q3)
        if polys:
            n = np.array([-gi[0], -gi[1], 1.0])
            faces.append(Face(polys, n / np.linalg.norm(n), "roof", (labels or [""] * len(planes))[i]))
    if not faces:
        raise Fallback(f"{shape}: empty envelope")
    return Roof(faces, shape, zmax)


def _edge_planes(support: np.ndarray, slope: float, offsets: list[float] | None = None) -> list[tuple[np.ndarray, float]]:
    """Planes rising inward from each edge of a CCW convex polygon at `slope`: z = slope * d_inward(p) + offset."""
    out = []
    n = len(support)
    for i in range(n):
        a, b = support[i], support[(i + 1) % n]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        nin = np.array([-d[1], d[0]]) / L                  # inward normal of a CCW polygon
        off = 0.0 if offsets is None else offsets[i]
        out.append((slope * nin, -slope * float(nin @ a) + off))
    return out


def _max_inset(support: np.ndarray) -> float:
    """max_p min_i d_i(p) of a convex polygon (Chebyshev radius): the envelope height at unit slope."""
    planes = _edge_planes(support, 1.0)
    best = 0.0
    for i, (gi, hi) in enumerate(planes):
        reg = support.copy()
        for j, (gj, hj) in enumerate(planes):
            if i != j and len(reg):
                reg = clip_halfplane(reg, gi - gj, hi - hj)
        if len(reg):
            best = max(best, float((reg @ gi + hi).max()))
    return best


def _convex_support(outer: np.ndarray, holes: list[np.ndarray], params: StructureParams) -> tuple[np.ndarray, str]:
    """The part's convex hull when it is (nearly) convex without holes, else its minimum rotated rectangle when it
    is rectangular enough."""
    if not holes and convexity(outer) >= params.convex_min:
        return ccw(convex_hull(outer)), "convex"
    if rectangularity(outer, holes) >= params.rect_min:
        return ccw(obb(outer).corners()), "obb"
    raise Fallback(f"not convex (area/hull {convexity(outer, holes):.2f}) and not rectangular "
                   f"(area/OBB {rectangularity(outer, holes):.2f})")


def _axis_frame(outer: np.ndarray, orientation: str | None, direction: float | None):
    """Ridge frame of an OBB roof: centre, ridge unit u, across unit v, half-length along u, half-width across, note."""
    b = obb(outer)
    c, u, v, L, W = b.c, b.u, b.v, b.L, b.W
    note = "ridge along the long side (S3DB default)"
    if direction is not None:            # roof:direction = where a roof face looks -> across the ridge
        dvec = np.array([math.sin(math.radians(direction)), math.cos(math.radians(direction))])
        if abs(dvec @ u) > abs(dvec @ v):
            u, v, L, W = v, u, W, L
        note = f"ridge across roof:direction={direction:g}"
    elif orientation == "across":
        u, v, L, W = v, u, W, L
        note = "ridge across the long side (roof:orientation=across)"
    if st._cross2(u, v) < 0:
        v = -v
    return c, u, v, L, W, note


def profile_roof(outer, holes, z_eave, rh, profile, shape, orientation, direction, params) -> Roof:
    """A roof whose height depends only on the coordinate across the ridge: a concave piecewise-linear profile
    [(w/W in -1..1, z/rh)] (gabled, saltbox, gambrel, round) = the lower envelope of its segments, over the OBB."""
    if rectangularity(outer, holes) < params.rect_min:
        raise Fallback(f"{shape}: area/OBB {rectangularity(outer, holes):.2f} < {params.rect_min}")
    c, u, v, L, W, note = _axis_frame(outer, orientation, direction)
    if W < 0.05:
        raise Fallback(f"{shape}: part too thin")
    corners = np.array([c - L * u - W * v, c + L * u - W * v, c + L * u + W * v, c - L * u + W * v])
    planes, labels = [], []
    for (w0, z0), (w1, z1) in zip(profile[:-1], profile[1:]):
        k = (z1 - z0) * rh / ((w1 - w0) * W)                 # dz / dw (metres per metre across)
        planes.append((k * v, z0 * rh + k * (-(v @ c) - w0 * W)))
        labels.append(f"w{w0:+.2f}..{w1:+.2f}")
    roof = envelope_roof(outer, holes, corners, planes, z_eave, shape, labels)
    roof.meta.update(ridge=note, support="obb", ridge_axis=[round(float(u[0]), 4), round(float(u[1]), 4)])
    roof.height_m = rh
    return roof


def hipped_roof(outer, holes, z_eave, rh, slope, shape, orientation, direction, params) -> Roof:
    """hipped / half-hipped / mansard: the lower envelope of planes rising from every support edge (the straight
    skeleton of a convex part), height rh (or a slope from roof:angle)."""
    sup, kind = _convex_support(outer, holes, params)
    H1 = _max_inset(sup)
    if H1 < 0.05:
        raise Fallback(f"{shape}: support too thin")
    if slope is None:
        slope = rh / H1
    rh = slope * H1
    offsets = None
    labels = [f"edge{i}" for i in range(len(sup))]
    if shape == "half-hipped":
        # the end planes (support edges across the ridge) start half way up: a small gable remains
        c, u, v, L, W, _ = _axis_frame(outer, orientation, direction)
        offsets = []
        for i in range(len(sup)):
            d = sup[(i + 1) % len(sup)] - sup[i]
            d = d / max(np.linalg.norm(d), 1e-12)
            offsets.append(0.5 * rh if abs(d @ v) > 0.7 else 0.0)
        slope = rh / max(W, 1e-6)
    planes = _edge_planes(sup, slope, offsets)
    if shape == "mansard":
        # two pitches on every side: steep to 70 % of the height within the outer 30 % of the inset, then shallow
        db, zb = 0.3 * H1, 0.7 * rh
        s1, s2 = zb / db, (rh - zb) / (H1 - db)
        planes = _edge_planes(sup, s1) + _edge_planes(sup, s2, [zb - s2 * db] * len(sup))
        labels = [f"steep{i}" for i in range(len(sup))] + [f"upper{i}" for i in range(len(sup))]
    roof = envelope_roof(outer, holes, sup, planes, z_eave, shape, labels)
    roof.meta["support"] = kind
    roof.height_m = rh
    return roof


def fan_roof(outer, holes, z_eave, rh, shape) -> Roof:
    c = star_centre(outer, holes)
    if c is None:
        raise Fallback(f"{shape}: no apex point that sees the whole outline (holes or not star-shaped)")
    ring = ccw(outer)
    apex = np.r_[c, z_eave + rh]
    faces = []
    for i in range(len(ring)):
        a, b = np.r_[ring[i], z_eave], np.r_[ring[(i + 1) % len(ring)], z_eave]
        n = np.cross(b - a, apex - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        faces.append(Face([np.array([a, b, apex])], n / np.linalg.norm(n), "roof", f"fan{i}"))
    return Roof(faces, shape, rh, {"apex": [round(float(c[0]), 3), round(float(c[1]), 3)]})


def dome_profile(rings: int) -> list[tuple[float, float]]:
    return [(math.cos(k / rings * math.pi / 2), math.sin(k / rings * math.pi / 2)) for k in range(rings + 1)]


ONION_PROFILE = [(1.0, 0.0), (1.16, 0.12), (1.2, 0.27), (1.06, 0.45), (0.7, 0.62), (0.34, 0.78), (0.12, 0.9), (0.0, 1.0)]


def revolve_roof(outer, holes, z_eave, rh, shape, params) -> Roof:
    """dome / onion: scaled copies of the outline about a centre that sees it all, at rising heights."""
    c = star_centre(outer, holes)
    if c is None:
        raise Fallback(f"{shape}: no centre that sees the whole outline (holes or not star-shaped)")
    ring = ccw(outer)
    prof = dome_profile(params.dome_rings) if shape == "dome" else ONION_PROFILE
    rings = [np.c_[c + s * (ring - c), np.full(len(ring), z_eave + t * rh)] for s, t in prof[:-1]]
    apex = np.r_[c, z_eave + rh]
    faces = []
    n = len(ring)
    for k in range(len(rings)):
        lo = rings[k]
        for i in range(n):
            j = (i + 1) % n
            quad = np.array([lo[i], lo[j], rings[k + 1][j], rings[k + 1][i]]) if k + 1 < len(rings) else \
                np.array([lo[i], lo[j], apex])
            nn = newell(quad)
            if np.linalg.norm(nn) < 0.5:
                continue
            faces.append(Face([quad], nn, "roof", f"ring{k}-{i}"))
    return Roof(faces, shape, rh, {"centre": [round(float(c[0]), 3), round(float(c[1]), 3)], "rings": len(prof) - 1})


def skillion_roof(outer, holes, z_eave, rh, direction) -> Roof:
    note = f"roof:direction={direction:g}"
    if direction is None:
        dvec = obb(outer).v                        # no direction tagged: across the short side (data gap)
        note = "no roof:direction: slope across the short side, low side chosen arbitrarily (data gap)"
    else:
        dvec = np.array([math.sin(math.radians(direction)), math.cos(math.radians(direction))])
    t = -(ccw(outer) @ dvec)
    tmin, tmax = float(t.min()), float(t.max())
    if tmax - tmin < 0.05:
        raise Fallback("skillion: part too thin along the slope")
    k = rh / (tmax - tmin)
    g, h = -k * dvec, -k * tmin
    ring = _bridged(outer, holes)
    nrm = np.array([-g[0], -g[1], 1.0])
    roof = Roof([Face([_lift(ring, lambda r: z_eave + r @ g + h)], nrm / np.linalg.norm(nrm), "roof", "skillion")],
                "skillion", rh, {"direction": note})
    return roof


# ---------------------------------------------------------------- parts


@dataclass
class Param:
    """One parameter of a part with its source, state and the evidence about it (design §2.4.6)."""

    value: float
    source: str
    state: str = ""
    evidence: list = field(default_factory=list)
    ticket: str | None = None
    map_value: float | None = None       # the map's value when a curator / photo fit changed it

    def __post_init__(self) -> None:
        if not self.state:
            self.state = STATE_OF_SOURCE.get(self.source, STATE_MAP)

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"value": round(float(self.value), 3), "state": self.state, "source": self.source}
        if self.evidence:
            out["evidence"] = self.evidence
        if self.ticket:
            out["ticket"] = self.ticket
        if self.map_value is not None:
            out["map_value"] = round(float(self.map_value), 3)
        return out


@dataclass
class PartSpec:
    """A building:part (or a main-outline gap) with its resolved parameters, before geometry."""

    id: str
    kind: str                            # building:part value, or 'remainder'
    outer: np.ndarray                    # (k, 2) CCW
    holes: list[np.ndarray]              # CW
    tags: dict[str, str]
    name: str | None = None
    roof_tag: str | None = None
    roof_shape: str = "flat"             # requested (normalized) shape
    params: dict[str, Param] = field(default_factory=dict)    # min_height, height, roof_height
    outline_source: str = st.PROV_S3DB
    problems: list[str] = field(default_factory=list)
    parent: str | None = None            # the part this one rests on
    inset_m: float = 0.0
    drawn: bool = True                   # False: kept in the registry but not built (contradiction, tiny stub, G10 withheld)
    issues: list = field(default_factory=list)   # OSM values the build could not use as tagged: each opens a ticket
    support: str | None = None           # G10 action when its base is not supported: 'withheld' | 'rests-on-support'

    @property
    def z0(self) -> float:
        return float(self.params["min_height"].value)

    @property
    def top(self) -> float:
        return float(self.params["height"].value)

    @property
    def roof_h(self) -> float:
        return float(self.params["roof_height"].value)

    @property
    def area(self) -> float:
        return poly_area(self.outer, self.holes)

    def min_width(self) -> float:
        return 2.0 * obb(self.outer).W

    def state(self) -> str:
        """The part's geometry state: the least certain of its height parameters (a value photos changed
        automatically, D15, ranks below a mapped fact: a part with a photo-fitted top reads as photo-fitted)."""
        order = (STATE_STUB, STATE_DISPUTED, STATE_CONVENTION, STATE_CORRECTED, STATE_FITTED, STATE_MAP, STATE_VERIFIED,
                 STATE_CARVED, STATE_CURATOR)
        sts = [self.params[k].state for k in ("height", "roof_height", "min_height") if k in self.params]
        return min(sts, key=lambda s: order.index(s) if s in order else 0) if sts else STATE_MAP


def resolve_heights(p: PartSpec, params: StructureParams = DEFAULT_STRUCTURE) -> None:
    """height / building:levels, min_height / building:min_level, roof:height / roof:levels / roof:angle, each with
    its source; missing roof heights by convention (D17), contradictions recorded."""
    t = p.tags
    for k in NUMERIC_TAGS:
        if k in t and (length_m(t[k]) if k in LENGTH_TAGS else number(t[k])) is None:
            p.problems.append(f"unparseable {k}={t[k]!r} (ignored)")
            p.issues.append({"tag": k, "value": t[k], "why": "unparseable numeric tag (ignored)"})
    h = length_m(t.get("height")) if "height" in t else None
    if h is not None and h <= 0:
        p.problems.append(f"non-positive height={t['height']!r} ignored")
        p.issues.append({"tag": "height", "value": t["height"], "why": "non-positive height (ignored)"})
        h = None
    lv = number(t.get("building:levels"))
    rl = number(t.get("roof:levels"))
    if h is not None:
        top = Param(h, st.PROV_S3DB)
    elif lv is not None and lv > 0:
        top = Param((lv + (rl or 0.0)) * params.level_m, st.PROV_LEVELS)
        p.problems.append(f"no height: building:levels x {params.level_m:g} m")
    else:
        top = Param(params.stub_m, st.PROV_DEFAULT)
        p.problems.append(f"missing height ({params.stub_m:g} m 'height undetermined' stub, D13)")
    mh = length_m(t.get("min_height")) if "min_height" in t else None
    ml = number(t.get("building:min_level"))
    if mh is not None:
        z0 = Param(mh, st.PROV_S3DB)
    elif ml is not None:
        z0 = Param(ml * params.level_m, st.PROV_LEVELS)
    else:
        z0 = Param(0.0, st.PROV_GROUND)
    p.params["height"], p.params["min_height"] = top, z0
    if z0.value >= top.value - 0.05:
        p.problems.append(f"min_height {z0.value:g} >= height {top.value:g} (contradiction): part not drawn")
        p.issues.append({"tag": "min_height", "value": t.get("min_height", t.get("building:min_level")),
                         "why": f"min_height {z0.value:g} >= height {top.value:g} (contradiction): part not drawn"})
        p.drawn = False
    tag = t.get("roof:shape")
    p.roof_tag = tag
    shape = (tag or "flat").strip().lower()
    if tag and shape in ALIASES:
        p.problems.append(f"roof:shape={tag} read as {ALIASES[shape]}")
    shape = ALIASES.get(shape, shape)
    if shape not in SHAPES:
        p.problems.append(f"roof:shape={tag} is not a buildable S3DB value: flat")
        shape = "flat"
    p.roof_shape = shape
    room = max(top.value - z0.value, 0.0)
    rh = length_m(t.get("roof:height")) if "roof:height" in t else None
    ang = number(t.get("roof:angle"))
    if rh is not None and rh < 0:
        p.problems.append(f"negative roof:height={t['roof:height']!r} ignored")
        p.issues.append({"tag": "roof:height", "value": t["roof:height"], "why": "negative roof:height (ignored)"})
        rh = None
    if shape == "flat":
        roof = Param(0.0, st.PROV_S3DB if tag else st.PROV_GROUND)
        if rh is not None and rh > 0.3:
            p.problems.append(f"flat roof with roof:height={rh:g} (ignored)")
    elif rh is not None:
        roof = Param(rh, st.PROV_S3DB)
    elif rl is not None:
        roof = Param(rl * params.level_m, st.PROV_LEVELS)
    elif ang is not None:
        roof = Param(obb(p.outer).W * math.tan(math.radians(min(max(ang, 0.0), 85.0))), st.PROV_ROOF_ANGLE)
    elif top.source == st.PROV_DEFAULT and not params.stub_conventional_roofs:
        roof = Param(0.0, st.PROV_DEFAULT)
        p.problems.append(f"roof:shape={tag} on a {params.stub_m:g} m stub: no conventional roof (D17), flat")
        p.roof_shape = "flat"
    elif params.conventional_roofs:
        roof = Param(roof_default_height(shape, p.outer, room, params), st.PROV_SHAPE_DEFAULT)
        p.problems.append(f"roof:shape={tag} without roof:height: {roof.value:.1f} m by convention (D17)")
    else:
        roof = Param(0.0, st.PROV_SHAPE_DEFAULT)
        p.problems.append(f"roof:shape={tag} without roof:height and conventions off: flat")
        p.roof_shape = "flat"
    if roof.value > room + 1e-6:
        p.problems.append(f"roof:height {roof.value:g} > height - min_height {room:g}: clamped")
        if roof.source not in (st.PROV_SHAPE_DEFAULT,) and p.drawn:     # (a contradiction has its own ticket)
            p.issues.append({"tag": "roof:height", "value": round(float(roof.value), 3),
                             "why": f"roof height {roof.value:g} > height - min_height {room:g}: clamped"})
        roof.value = room
    p.params["roof_height"] = roof
    if p.kind == "roof":
        p.problems.append("building:part=roof (no walls): wall band = height - min_height - roof:height")


MAP_TAGS = NUMERIC_TAGS + ("roof:shape",)       # the tags a part's map heights come from (resolve_heights)


def map_value_of(kind: str, tags: dict[str, str], outer: np.ndarray | None, param: str,
                 main_height: str | None = None, params: StructureParams = DEFAULT_STRUCTURE
                 ) -> tuple[float | None, str | None, str]:
    """The value and source the MAP gives one parameter of a part, re-derived from the OSM facts alone: an OSM part's
    own tags through resolve_heights; a main-outline gap's top from the main outline's height tag (else its supports'
    min_height or the 3 m stub: geometry, not re-derived here), its base and roof 0. What a D25 record's 'map value'
    must be (crowd3d.correct.map_value_problem; verify-r2 forged it consistently in the record, the parameter and the
    before-snapshot). Returns (value, source, note); value None where it depends on the outline (a roof height from
    roof:angle or the shape's convention) or on a gap's supports."""
    if kind == "remainder":
        if param in ("min_height", "roof_height"):
            return 0.0, st.PROV_GROUND, ""
        mh = length_m(main_height) if main_height else None
        if param == "height" and mh is not None and mh > 0:
            return float(mh), st.PROV_OSM_HEIGHT, ""
        return None, None, "a gap's top: its supports' min_height or the stub (geometry)"
    ring = np.asarray(outer, float)[:, :2] if outer is not None and len(outer) >= 3 else \
        np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    q = PartSpec("map", kind or "yes", ring, [], dict(tags or {}))
    resolve_heights(q, params)
    par = q.params.get(param)
    if par is None:
        return None, None, f"no {param}"
    if par.source in (st.PROV_ROOF_ANGLE, st.PROV_SHAPE_DEFAULT):
        return None, par.source, "depends on the outline"
    return float(par.value), par.source, ""


def roof_default_height(shape: str, outer: np.ndarray, room: float, params: StructureParams = DEFAULT_STRUCTURE) -> float:
    """Conventional proportion of a tagged roof shape without roof:height / roof:levels / roof:angle (s3db-shape-default;
    always flagged, D17): dome = the inscribed half-width, onion = 1.5 x that, mansard / gambrel = one level,
    round = the half-width, every other pitched shape (cones and pyramids too) = the pitch on the half-width; at
    most roof_room_share of the part's wall room."""
    b = obb(outer)
    r_in = min(b.L, b.W)
    if shape == "dome":
        h = r_in
    elif shape == "onion":
        h = 1.5 * r_in
    elif shape in ("cone", "pyramidal") and params.spire_cone_rule and 2 * b.W <= 12.0:
        h = room
    elif shape in ("mansard", "gambrel"):
        h = params.level_m
    elif shape == "round":
        h = b.W
    else:
        h = b.W * math.tan(math.radians(params.pitch_deg))
    return float(min(h, params.roof_room_share * room))


# ---------------------------------------------------------------- building one part


@dataclass
class WallFace:
    ring: int                            # 0 outer, k >= 1 hole k
    edge: int
    a: np.ndarray                        # (2,) start (CCW along the outer ring, CW along a hole)
    b: np.ndarray
    z0: float
    profile: np.ndarray                  # (m, 2) (u, z) of the wall's top from u = 0 to the length

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def z1(self) -> float:
        return float(self.profile[:, 1].max())

    @property
    def flat_top(self) -> bool:
        return bool(np.ptp(self.profile[:, 1]) < 1e-3)      # (sub-millimetre ripples of the clipping are not a profile)

    def polygon3d(self) -> np.ndarray:
        d = (self.b - self.a) / max(self.length, 1e-12)
        top = [np.r_[self.a + u * d, z] for u, z in self.profile[::-1]]
        return np.array([np.r_[self.a, self.z0], np.r_[self.b, self.z0]] + top)

    def normal(self) -> np.ndarray:
        d = self.b - self.a
        n = np.array([d[1], -d[0], 0.0])
        return n / max(np.linalg.norm(n), 1e-12)


@dataclass
class BuiltPart:
    spec: PartSpec
    outer: np.ndarray                    # as built (inset / scaled)
    holes: list[np.ndarray]
    z0: float
    eave: float
    top: float
    roof: Roof
    walls: list[WallFace]
    base: Face | None
    census: dict[str, int] = field(default_factory=dict)
    t_junctions: int = 0

    @property
    def id(self) -> str:
        return self.spec.id

    def faces(self, base: bool = True) -> list[Face]:
        out = list(self.roof.faces)
        out += [Face([w.polygon3d()], w.normal(), "wall") for w in self.walls]
        if base and self.base is not None:
            out.append(self.base)
        return out

    def triangles(self, base: bool = True) -> np.ndarray:
        tris = [f.triangles() for f in self.faces(base)]
        tris = [t for t in tris if len(t)]
        return np.concatenate(tris) if tris else np.zeros((0, 3, 3))

    def surface_z(self, xy: np.ndarray) -> np.ndarray:
        """Height of the part's roof surface above each (x, y) (NaN outside the part)."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        out = np.full(len(xy), np.nan)
        for f in self.roof.faces:
            n = f.normal
            if abs(n[2]) < 0.05:
                continue
            for poly in f.polys:
                inside = st.points_in_polygon(xy, poly[:, :2])
                if not inside.any():
                    continue
                p0 = poly[0]
                z = p0[2] - ((xy[inside, 0] - p0[0]) * n[0] + (xy[inside, 1] - p0[1]) * n[1]) / n[2]
                out[inside] = np.fmax(out[inside], z)
        return out


def _wall_faces(outer: np.ndarray, holes: list[np.ndarray], z0: float, z_eave: float, roof: Roof) -> list[WallFace]:
    """Vertical walls on every ring edge from z0 up to the roof boundary along it (gable ends follow the roof)."""
    rv = np.vstack([q for f in roof.faces for q in f.polys])
    walls = []
    rings = [ccw(outer)] + [cw(h) for h in holes]
    for ri, ring in enumerate(rings):
        n = len(ring)
        for i in range(n):
            a, b = ring[i], ring[(i + 1) % n]
            d = b - a
            L = float(np.linalg.norm(d))
            if L < 1e-3:
                continue
            s = (rv[:, :2] - a) @ d / (L * L)
            perp = np.abs((rv[:, 0] - a[0]) * d[1] - (rv[:, 1] - a[1]) * d[0]) / L
            sel = (perp < 1e-5) & (s > -1e-7) & (s < 1 + 1e-7)
            tops = sorted(((float(np.clip(si, 0, 1)) * L, float(z)) for si, z in zip(s[sel], rv[sel, 2])), key=lambda x: x[0])
            prof: list[list[float]] = []
            for u, z in tops:
                if prof and abs(u - prof[-1][0]) < 1e-5:
                    prof[-1][1] = max(prof[-1][1], z)
                else:
                    prof.append([u, z])
            if not prof or prof[0][0] > 1e-5:
                prof.insert(0, [0.0, z_eave])
            if prof[-1][0] < L - 1e-5:
                prof.append([L, z_eave])
            prof[0][0], prof[-1][0] = 0.0, L
            P = np.array(prof)
            # drop interior profile points on a straight run (collinear in (u, z))
            keep = [0]
            for k in range(1, len(P) - 1):
                a2, b2, c2 = P[keep[-1]], P[k], P[k + 1]
                if abs(st._cross2(c2 - a2, b2 - a2)) > 1e-6 * max(1.0, np.linalg.norm(c2 - a2)):
                    keep.append(k)
            keep.append(len(P) - 1)
            P = P[keep]
            if P[:, 1].max() - z0 < 1e-3:
                continue                                          # roof-only part at its eave: no wall band
            P[:, 1] = np.maximum(P[:, 1], z0)
            walls.append(WallFace(ri, i, a.copy(), b.copy(), z0, P))
    return walls


def build_roof(p: PartSpec, outer: np.ndarray, holes: list[np.ndarray], z_eave: float, rh: float,
               params: StructureParams = DEFAULT_STRUCTURE) -> Roof:
    """The roof primitive for the part's roof:shape with the explicit fallback chain."""
    shape = p.roof_shape
    t = p.tags
    orientation = (t.get("roof:orientation") or "").strip().lower() or None
    direction = direction_deg(t.get("roof:direction"))
    ang = number(t.get("roof:angle"))
    slope = math.tan(math.radians(min(max(ang, 0.0), 85.0))) if ang is not None and \
        p.params["roof_height"].source == st.PROV_ROOF_ANGLE else None
    chain = FALLBACKS.get(shape, ["flat"])
    if rh <= 0.05:
        chain = ["flat"]
    tried = []
    for s in chain:
        try:
            if s == "flat":
                roof = flat_roof(outer, holes, z_eave)
            elif s == "gabled":
                roof = profile_roof(outer, holes, z_eave, rh, [(-1, 0), (0, 1), (1, 0)], s, orientation, direction, params)
            elif s == "saltbox":
                roof = profile_roof(outer, holes, z_eave, rh, [(-1, 0), (-0.3, 1), (1, 0)], s, orientation, direction, params)
            elif s == "gambrel":
                roof = profile_roof(outer, holes, z_eave, rh, [(-1, 0), (-0.7, 0.7), (0, 1), (0.7, 0.7), (1, 0)], s,
                                    orientation, direction, params)
            elif s == "round":
                prof = [(-math.cos(k / params.round_facets * math.pi), math.sin(k / params.round_facets * math.pi))
                        for k in range(params.round_facets + 1)]
                roof = profile_roof(outer, holes, z_eave, rh, prof, s, orientation, direction, params)
            elif s in ("hipped", "half-hipped", "mansard"):
                roof = hipped_roof(outer, holes, z_eave, rh, slope, s, orientation, direction, params)
            elif s in ("pyramidal", "cone"):
                roof = fan_roof(outer, holes, z_eave, rh, s)
            elif s in ("dome", "onion"):
                roof = revolve_roof(outer, holes, z_eave, rh, s, params)
            elif s == "skillion":
                roof = skillion_roof(outer, holes, z_eave, rh, direction)
            else:
                raise Fallback(f"{s}: not implemented")
        except Fallback as e:
            tried.append(str(e))
            continue
        if s != chain[0] or tried:
            roof.meta["fallback_from"] = shape
            roof.meta["reason"] = "; ".join(tried)
        if s == "flat" and shape != "flat":
            roof.meta.setdefault("fallback_from", shape)
            roof.meta.setdefault("reason", "no roof height")
        return roof
    raise RuntimeError("flat roof failed")


def build_part(p: PartSpec, params: StructureParams = DEFAULT_STRUCTURE, dz: float = 0.0, eave_mult: float = 1.0,
               roof_mult: float = 1.0, scale: float = 1.0, closed: bool = True) -> BuiltPart:
    """One closed solid. eave_mult / roof_mult / scale: the multipliers a skyline profile tries (wall extent,
    roof height, outline about its centroid); dz: the shift its parent's change gives it."""
    outer, holes = p.outer, list(p.holes)
    if p.inset_m > 0:
        q = inset_ring(outer, p.inset_m)
        if q is not None:
            outer = q
    if abs(scale - 1.0) > 1e-9:
        c = poly_centroid(outer, holes)
        outer = c + (outer - c) * scale
        holes = [c + (h - c) * scale for h in holes]
    z0 = p.z0 + dz
    roof_h = p.roof_h * roof_mult
    eave = z0 + max(p.top - p.roof_h - p.z0, 0.0) * eave_mult
    roof = build_roof(p, outer, holes, eave, roof_h, params)
    if roof.shape == "flat":
        # a flat roof, tagged or a fallback, keeps the part's tagged top (recorded when it is a fallback)
        eave = z0 + max(p.top - p.z0, 0.0) * eave_mult
        if p.roof_shape != "flat":
            roof.meta["flat_at_top"] = True
        roof = Roof(flat_roof(outer, holes, eave).faces, "flat", 0.0, roof.meta)
    walls = _wall_faces(outer, holes, z0, eave, roof)
    ring = _bridged(outer, holes)
    base = Face([_lift(ring[::-1].copy(), lambda r: np.full(len(r), z0))], np.array([0.0, 0.0, -1.0]), "base")
    bp = BuiltPart(p, outer, holes, z0, eave, eave + roof.height_m, roof, walls, base)
    faces = bp.faces()
    wall_faces = faces[len(roof.faces):len(roof.faces) + len(walls)]
    if not closed:                                       # a quick trial solid (skyline profiles): no T-junction pass
        return bp
    bp.t_junctions = insert_t_junctions(faces)
    # a wall polygon may have received vertices on its top: keep the profile in step (max z per position)
    for w, f in zip(walls, wall_faces):
        poly = f.polys[0]
        d = (w.b - w.a) / max(w.length, 1e-12)
        top = poly[poly[:, 2] > w.z0 + 1e-9]
        prof: list[list[float]] = []
        for u, z in sorted(zip(((top[:, :2] - w.a) @ d).tolist(), top[:, 2].tolist())):
            u = min(max(u, 0.0), w.length)
            if prof and abs(u - prof[-1][0]) < 1e-6:
                prof[-1][1] = max(prof[-1][1], z)
            else:
                prof.append([u, z])
        if not prof or prof[0][0] > 1e-6:
            prof.insert(0, [0.0, w.z0])
        if prof[-1][0] < w.length - 1e-6:
            prof.append([w.length, w.z0])
        prof[0][0], prof[-1][0] = 0.0, w.length
        w.profile = np.array(prof)
    bp.census = watertight(bp.triangles())
    return bp


# ---------------------------------------------------------------- landmark


@dataclass
class StructureModel:
    """Parametric parts of one landmark: specs (registry rows, drawn or not) and the built solids."""

    landmark: str
    specs: list[PartSpec]
    built: list[BuiltPart]
    main_rings: list[np.ndarray]
    main_tags: dict[str, str]
    dropped: list[dict[str, Any]]
    osm_meta: dict[str, Any]
    report: dict[str, Any]
    params: StructureParams = DEFAULT_STRUCTURE
    frame: dict[str, Any] = field(default_factory=dict)          # frame check results (judge.frame_gate)
    tickets: list[dict[str, Any]] = field(default_factory=list)
    # D15: values changed automatically by >= 2 agreeing photo channels -- the reversible list (crowd3d.judge): per
    # adoption the old values, the new ones, provenance photo-fitted (auto) and all the evidence; OSM is never edited
    adoptions: list[dict[str, Any]] = field(default_factory=list)
    # D25-T: the measured tilt of the whole stack (None: upright, the S3DB convention). Every solid stays upright in
    # the model's own coordinates (evidence is judged in coordinates rotated back to upright); the pose is one rigid
    # rotation about the main outline's centroid at the base, applied only where the model is exported (GLB) and
    # checked there against its record (crowd3d.correct.pose_matrix)
    pose: dict[str, Any] | None = None

    def part(self, pid: str) -> PartSpec:
        return next(p for p in self.specs if p.id == pid)

    def built_part(self, pid: str) -> BuiltPart | None:
        return next((b for b in self.built if b.id == pid), None)

    def rebuild(self, overrides: dict[str, dict[str, float]] | None = None, ids: list[str] | None = None,
                closed: bool = True) -> list[BuiltPart]:
        """The solids with multipliers per part id ({'eave': m, 'roof': m, 'scale': m}); children follow a changed
        parent (a part standing on the parent's top moves with that top, one sunk into its roof with the surface
        under it)."""
        specs = self.specs if ids is None else [p for p in self.specs if p.id in set(ids)]
        return build_all(specs, self.params, overrides, closed=closed)

    def to_registry(self) -> dict[str, Any]:
        """massing.json v2 (design §2.4.6): one source and state per parameter, evidence, tickets."""
        parts = []
        for p in self.specs:
            b = self.built_part(p.id)
            row: dict[str, Any] = {"id": p.id, "kind": p.kind, "name": p.name, "outline_source": p.outline_source,
                                   "drawn": p.drawn and b is not None, "state": p.state(), "parent": p.parent,
                                   "shape": {"roof": b.roof.shape if b else p.roof_shape, "tag": p.roof_tag,
                                             "source": (st.PROV_S3DB if p.roof_tag else st.PROV_GROUND)},
                                   "params": {k: v.to_json() for k, v in p.params.items()},
                                   "area_m2": round(p.area, 2), "min_width_m": round(p.min_width(), 2)}
            if b is not None:
                if b.roof.meta.get("fallback_from"):
                    row["fallback"] = {"from": b.roof.meta["fallback_from"], "to": b.roof.shape,
                                       "reason": b.roof.meta.get("reason", "")}
                row["watertight"] = b.census.get("unmatched", 1) == 0 and b.census.get("duplicated", 1) == 0
                row["built"] = {"z0": round(b.z0, 3), "eave": round(b.eave, 3), "top": round(b.top, 3)}
            if p.inset_m:
                row["inset_m"] = p.inset_m
            if p.support:
                row["support"] = p.support
            if p.problems:
                row["problems"] = list(p.problems)
            parts.append(row)
        return {"landmark": self.landmark, "version": 2,
                "sources": {"osm": {**{k: v for k, v in self.osm_meta.items() if k != "file"}, "licence": "ODbL-1.0"}},
                "frame": self.frame, "parts": parts, "curator_overrides": [],
                "conflicts_open": sum(1 for t in self.tickets if t.get("open", True)), "tickets": self.tickets,
                "adoptions": self.adoptions,
                "adoptions_active": sum(1 for a in self.adoptions if a.get("active", True)),
                "pose": self.pose}

    def compact_extras(self) -> dict[str, Any]:
        """What the GLB's scene extras carry (the full registry is the sidecar)."""
        reg = self.to_registry()
        return {"source": "osm-s3db-parts", "attribution": OSM_ATTRIBUTION, "snapshot": reg["sources"]["osm"],
                "frame": self.frame, "conflicts_open": reg["conflicts_open"],
                # D15's reversible list with all its evidence (the old map values travel with the model)
                "adoptions": self.adoptions,
                "pose": self.pose,
                "withheld": [p.id for p in self.specs if p.support == "withheld"],
                # (fixer round 6) their outlines (OSM, as tagged): the GLB audit subtracts them too when it bounds a
                # D25 record's interior footprint from below (a withheld id alone no longer switches that bound off)
                "withheld_rings": {p.id: [[[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(r, float)[:, :2]]
                                          for r in [p.outer, *p.holes]]
                                   for p in self.specs if p.support == "withheld"},
                # the main outline (D25-T turns the stack about its centroid at the base: audit_glb checks the pivot)
                "main_ring": [[round(float(x), 3), round(float(y), 3)] for x, y in
                              np.asarray(self.main_rings[0], float)[:, :2]] if self.main_rings else None,
                # the main outline's height tag (a gap's map top: audit_glb re-derives D25 records' map values)
                "main_height": self.main_tags.get("height"),
                "parts": [{"id": r["id"], "state": r["state"], "roof": r["shape"]["roof"], "tag": r["shape"]["tag"],
                           "parent": r["parent"], "kind": r["kind"],
                           # the OSM tags its map heights come from (a D25 record's map value is re-derived from them)
                           "osm": {k: v for k, v in (self.part(r["id"]).tags or {}).items() if k in MAP_TAGS},
                           **({"fallback": r["fallback"]["from"]} if "fallback" in r else {}),
                           "params": {k: [v["value"], v["state"], v["source"]] for k, v in r["params"].items()},
                           # what was built from them, for the GLB audit: base / eave / top heights, the outline's
                           # source and its rings as built (ENU x, y; outer first)
                           "outline_source": r["outline_source"], "built": r.get("built"),
                           "rings": self._built_rings(r["id"])}
                          for r in reg["parts"] if r["drawn"]]}

    def _built_rings(self, pid: str) -> list[list[list[float]]]:
        b = self.built_part(pid)
        if b is None:
            return []
        return [[[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(ring, float)[:, :2]]
                for ring in [b.outer, *b.holes]]


def main_body(model: StructureModel) -> BuiltPart | None:
    """The main body: the drawn ground part (base <= 0.5 m) with the largest footprint that is not a 'height pending'
    stub (D13). Its eave is the reference of D15's scale-free check (a part's top / the main body's eave height)."""
    cands = [b for b in model.built if b.z0 <= 0.5 and b.spec.params["height"].source != st.PROV_DEFAULT]
    return max(cands, key=lambda b: (b.spec.area, b.id)) if cands else None


def descendants(model: StructureModel, pid: str) -> list[str]:
    """Ids of the parts resting on pid, directly or through others (parents first)."""
    out: list[str] = []
    todo = [pid]
    while todo:
        q = todo.pop(0)
        for p in model.specs:
            if p.parent == q and p.id not in out and p.id != pid:
                out.append(p.id)
                todo.append(p.id)
    return out


def _order_parents_first(specs: list[PartSpec]) -> list[int]:
    idx = {p.id: i for i, p in enumerate(specs)}
    order, seen = [], set()

    def visit(i, depth=0):
        if i in seen or depth > 64:
            return
        par = specs[i].parent
        if par is not None and par in idx:
            visit(idx[par], depth + 1)
        seen.add(i)
        order.append(i)

    for i in range(len(specs)):
        visit(i)
    return order


def build_all(specs: list[PartSpec], params: StructureParams = DEFAULT_STRUCTURE,
              overrides: dict[str, dict[str, float]] | None = None, closed: bool = True) -> list[BuiltPart]:
    overrides = overrides or {}
    moved = _moved(overrides, specs)
    by_id = {q.id: q for q in specs}
    built: dict[str, BuiltPart] = {}
    out = []
    for i in _order_parents_first(specs):
        p = specs[i]
        if not p.drawn:
            continue
        dz = 0.0
        if p.parent is not None and p.parent in built and p.parent in moved:
            # the parent changed: a part standing on its top follows that top, one sunk into its roof follows the
            # surface under its centroid (the parent's own shift is already in its built solid)
            par_now, par_spec = built[p.parent], by_id[p.parent]
            par_map = build_part(par_spec, params, closed=False)
            if abs(p.z0 - par_spec.top) < 0.6:
                dz = par_now.top - par_map.top
            else:
                rp = poly_centroid(p.outer, p.holes)[None]
                a, b = par_now.surface_z(rp)[0], par_map.surface_z(rp)[0]
                dz = float(a - b) if np.isfinite(a) and np.isfinite(b) else par_now.top - par_map.top
        o = overrides.get(p.id, {})
        try:
            bp = build_part(p, params, dz=dz, eave_mult=o.get("eave", 1.0), roof_mult=o.get("roof", 1.0), closed=closed,
                            scale=o.get("scale", 1.0))
        except Exception as e:  # noqa: BLE001 -- a part that cannot be built is reported, never guessed
            p.problems.append(f"not built: {type(e).__name__}: {e}")
            continue
        built[p.id] = bp
        out.append(bp)
    return out


def _moved(overrides: dict[str, dict[str, float]], specs: list[PartSpec]) -> set[str]:
    """Ids of parts that move: overridden ones and (transitively) their children."""
    moved = {k for k, v in overrides.items() if any(abs(x - 1.0) > 1e-9 for x in v.values())}
    changed = True
    while changed:
        changed = False
        for p in specs:
            if p.id not in moved and p.parent in moved:
                moved.add(p.id)
                changed = True
    return moved


def assign_parents(specs: list[PartSpec]) -> None:
    """A part above the ground rests on the highest part under its centroid whose base is lower (stacking)."""
    for a in specs:
        a.parent = None
        if not a.drawn or a.z0 <= 0.5:
            continue
        rp = poly_centroid(a.outer, a.holes)
        best = None
        for b in specs:
            if b is a or not b.drawn or b.z0 >= a.z0 - 0.5:
                continue
            if not (st.points_in_polygon(rp[None], b.outer)[0] or st.distance_to_rings(rp[None], [b.outer])[0][0] <= 0.5):
                continue
            if best is None or b.top > best.top:
                best = b
        a.parent = best.id if best is not None else None


def main_outline_from_geojson(path: str | Path, enu: LocalENU) -> tuple[list[np.ndarray], dict[str, Any]]:
    """The main outline (largest polygon: outer CCW + holes CW) of eval/osm/<slug>.geojson in ENU, and its properties."""
    gj = json.loads(Path(path).read_text())
    feat = gj["features"][0]
    geom = feat["geometry"]
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    best = None
    for rings in polys:
        enu_rings = [np.array([enu.to_enu(la, lo)[:2] for lo, la in r]) for r in rings]
        # the main outline keeps every OSM vertex (st.clean_ring, as the registration uses it): merging collinear
        # vertices turns a short edge into a long one and changes which edges the frame gate counts
        outer = st.clean_ring(enu_rings[0])
        if outer is None:
            continue
        holes = [cw(h) for h in (st.clean_ring(r, 1.0) for r in enu_rings[1:]) if h is not None]
        if best is None or poly_area(outer, holes) > poly_area(best[0], best[1:]):
            best = [outer] + holes
    if best is None:
        raise ValueError(f"no usable main outline in {path}")
    return best, dict(feat.get("properties", {}))


def parts_from_osm(data: OsmData, enu: LocalENU, main_rings: list[np.ndarray],
                   params: StructureParams = DEFAULT_STRUCTURE) -> tuple[list[PartSpec], list[dict[str, Any]], dict[str, Any]]:
    """building:part ways and relations in the main outline (centroid within keep_buffer_m AND >= keep_area_share of the
    area within it), as PartSpecs without heights. Returns (parts, dropped, census)."""
    ll = data.nodes
    census: dict[str, Any] = {"raw_parts": 0, "ways": 0, "relations": 0, "incomplete": 0, "open_rings": 0}
    dropped: list[dict[str, Any]] = []
    cands: list[tuple[str, list[np.ndarray], list[np.ndarray], dict[str, str]]] = []

    def to_xy(nds):
        return np.array([enu.to_enu(*ll[n])[:2] for n in nds])

    for wid, (nds, t) in data.ways.items():
        if "building:part" not in t:
            continue
        census["raw_parts"] += 1
        census["ways"] += 1
        if len(nds) < 4 or nds[0] != nds[-1] or any(n not in ll for n in nds):
            dropped.append({"id": f"w{wid}", "why": "open or incomplete way"})
            census["incomplete"] += 1
            continue
        cands.append((f"w{wid}", [to_xy(nds)], [], t))
    for rid, (members, t) in data.rels.items():
        if "building:part" not in t:
            continue
        census["raw_parts"] += 1
        census["relations"] += 1
        outers = [data.ways[ref][0] for typ, ref, role in members if typ == "way" and role in ("outer", "") and ref in data.ways]
        inners = [data.ways[ref][0] for typ, ref, role in members if typ == "way" and role == "inner" and ref in data.ways]
        if any(typ == "way" and ref not in data.ways for typ, ref, role in members):
            census["incomplete"] += 1
        o_rings, o_open = assemble_rings(outers)
        i_rings, i_open = assemble_rings(inners)
        census["open_rings"] += o_open + i_open
        if not o_rings or any(n not in ll for r in o_rings + i_rings for n in r):
            dropped.append({"id": f"r{rid}", "why": "relation rings incomplete in snapshot"})
            continue
        cands.append((f"r{rid}", [to_xy(r) for r in o_rings], [to_xy(r) for r in i_rings], t))

    parts: list[PartSpec] = []
    for pid, outers, inners, t in cands:
        kind = t.get("building:part", "yes")
        if kind == "no":
            dropped.append({"id": pid, "why": "building:part=no"})
            continue
        clean_inners = [h for h in (clean_ring(r, params) for r in inners) if h is not None]
        pieces = []
        for r in outers:
            o = clean_ring(r, params)
            if o is None:
                q = drop_collinear(r, params.clean_tol_m)
                small = len(q) < 3 or abs(ring_area(q)) < params.min_piece_m2
                dropped.append({"id": pid, "why": f"piece under {params.min_piece_m2:g} m2" if small else
                                "self-intersecting ring"})
                continue
            hs = [cw(h) for h in clean_inners if st.points_in_polygon(h[:1], o)[0]]
            pieces.append((o, hs))
        for k, (o, hs) in enumerate(pieces):
            sid = pid if len(pieces) == 1 else f"{pid}#{k}"
            cen = representative_point(o, hs)
            if not in_zone(cen, main_rings, params.keep_buffer_m):
                dropped.append({"id": sid, "why": f"centroid outside main outline + {params.keep_buffer_m:g} m",
                                "dist_m": round(float(st.distance_to_rings(cen[None], main_rings)[0][0]), 1),
                                "roof:shape": t.get("roof:shape"), "height": t.get("height"), "name": t.get("name")})
                continue
            share = share_inside(o, hs, main_rings, params.keep_buffer_m)
            if share < params.keep_area_share:
                dropped.append({"id": sid, "why": f"area share inside main outline + {params.keep_buffer_m:g} m "
                                                  f"{share:.2f} < {params.keep_area_share}",
                                "area_m2": round(poly_area(o, hs), 1), "name": t.get("name"), "secondary_filter": True})
                continue
            parts.append(PartSpec(sid, kind, o, hs, dict(t), name=t.get("name")))
    return parts, dropped, census


def remainder_parts(main_rings: list[np.ndarray], parts: list[PartSpec], main_tags: dict[str, str],
                    params: StructureParams = DEFAULT_STRUCTURE) -> list[PartSpec]:
    """Main-outline area no part covers (S3DB renders only parts), as explicit parts with a flat top: the main
    outline's own height tag (osm-height), else the min_height of the raised parts resting against it (the highest,
    so nothing floats; osm-s3db-support), else the 3 m stub (D13). Gaps under remainder_stub_min_m2 without any
    height fact are not drawn.

    A raised part's own footprint is never a gap: the space under a gate's attic or an arch is empty on purpose
    (filling it closed the Brandenburg Gate's passages). So "resting on it" cannot be an area overlap (design
    §2.4.1 item 5 says >= 50 %); it is contact: >= support_overlap of the 0.5 m band around the raised part lies
    on the gap (mere touching, which gave Dresden 36.65 m in the probe, does not count)."""
    if not parts:
        return []
    outer = main_rings[0]
    lo, hi = outer.min(axis=0) - 1.0, outer.max(axis=0) + 1.0
    g = _Grid(lo, hi, params.raster_m)
    main = g.fill(outer, main_rings[1:]) > 0
    union = np.zeros_like(main)
    masks = {}
    for p in parts:
        if p.drawn:
            masks[p.id] = g.fill(p.outer, p.holes) > 0
            union |= masks[p.id]
    rest = (main & ~union).astype(np.uint8)
    rest = cv2.morphologyEx(rest, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))        # no slivers under 0.5 m
    rest = (cv2.dilate(rest, np.ones((3, 3), np.uint8)) > 0) & main                 # tucked 0.25 m under the parts
    band_k = np.ones((2 * max(1, int(round(0.5 / params.raster_m))) + 1,) * 2, np.uint8)
    out = []
    mh = length_m(main_tags.get("height"))
    for i, (o, hs) in enumerate(g.polygons(rest, params.remainder_min_m2)):
        o2 = clean_ring(o, params)
        if o2 is None:
            continue
        hs2 = [cw(h) for h in (clean_ring(h, params) for h in hs) if h is not None]
        p = PartSpec(f"remainder{i}", "remainder", o2, hs2, {"building:part": "remainder"}, outline_source=st.PROV_OSM)
        gm = g.fill(o2, hs2) > 0
        support = []
        for q in parts:
            if q.z0 <= 0.5 or not q.drawn:
                continue
            qm = masks[q.id]
            band = (cv2.dilate(qm.astype(np.uint8), band_k) > 0) & ~qm
            if band.any() and (band & gm).sum() / band.sum() >= params.support_overlap:
                support.append(q.z0)
        if mh is not None and mh > 0:
            p.params["height"] = Param(mh, st.PROV_OSM_HEIGHT)
            p.problems.append("gap: the main outline's height tag, flat")
        elif support:
            p.params["height"] = Param(max(support), st.PROV_SUPPORT)
            p.problems.append(f"gap: top from the min_height of {len(support)} raised part(s) resting against it")
        else:
            p.params["height"] = Param(params.stub_m, st.PROV_DEFAULT)
            p.problems.append(f"gap: no height anywhere ({params.stub_m:g} m stub, D13)")
            if p.area < params.remainder_stub_min_m2:
                p.problems.append(f"stub under {params.remainder_stub_min_m2:g} m2 without height facts: not drawn")
                p.drawn = False
        p.params["min_height"] = Param(0.0, st.PROV_GROUND)
        p.params["roof_height"] = Param(0.0, st.PROV_GROUND)
        out.append(p)
    return out


def assign_insets(specs: list[PartSpec], params: StructureParams = DEFAULT_STRUCTURE) -> None:
    """Micro-offsets against coplanar z-fighting between overlapping parts: by descending top, every part but each
    fourth shrinks by 1-3 cm (invisible at toy scale, recorded)."""
    live = sorted([p for p in specs if p.drawn], key=lambda p: -p.top)
    for rank, p in enumerate(live):
        k = rank % 4
        p.inset_m = round(params.inset_step_m * k, 4) if k else 0.0


def build_structure(landmark: str, specs: list[PartSpec], main_rings: list[np.ndarray], main_tags: dict[str, str],
                    dropped: list[dict[str, Any]] | None = None, osm_meta: dict[str, Any] | None = None,
                    census: dict[str, Any] | None = None, params: StructureParams = DEFAULT_STRUCTURE,
                    wikidata_height_m: float | None = None) -> StructureModel:
    """Resolve heights (parts, then the gaps), stacking, insets; build every drawn part; G10 (unsupported raised parts:
    ticket, withheld or kept on their support, resolve_support()); tickets for OSM values outside sanity bounds or
    unparseable (open_value_tickets()); report."""
    t0 = time.perf_counter()
    for p in specs:
        if p.kind != "remainder" and "height" not in p.params:
            resolve_heights(p, params)
    osm_parts = [p for p in specs if p.kind != "remainder"]
    specs = osm_parts + remainder_parts(main_rings, osm_parts, main_tags, params)
    assign_parents(specs)
    assign_insets(specs, params)
    built = build_all(specs, params)
    model = StructureModel(landmark, specs, built, main_rings, dict(main_tags), list(dropped or []), dict(osm_meta or {}),
                           {}, params)
    resolve_support(model)
    open_value_tickets(model, wikidata_height_m)
    model.report = structure_report(model, census or {}, time.perf_counter() - t0)
    return model


def load_osm_structure(osm_path: str | Path, outline_geojson: str | Path, lat: float, lon: float, landmark: str = "",
                       params: StructureParams = DEFAULT_STRUCTURE, wikidata_height_m: float | None = None) -> StructureModel:
    """The structure of a landmark from its OSM snapshot and main outline (eval/osm/<slug>.geojson), in the ENU frame
    at (lat, lon) (the catalogue centre, the frame of the stylizer's SceneInput)."""
    enu = LocalENU(lat, lon)
    main_rings, props = main_outline_from_geojson(outline_geojson, enu)
    data = parse_osm(osm_path)
    oid = str(props.get("osm_id", ""))
    src = data.ways.get(oid) if props.get("osm_type") == "way" else data.rels.get(oid)
    main_tags = dict(src[1]) if src else {k: str(v) for k, v in props.items() if k in ("height", "building:levels") and v}
    specs, dropped, census = parts_from_osm(data, enu, main_rings, params)
    census["main_element_in_snapshot"] = bool(src)
    return build_structure(landmark or props.get("slug", ""), specs, main_rings, main_tags, dropped, data.meta, census, params,
                           wikidata_height_m)


# ---------------------------------------------------------------- checks that need no photos


def _base_samples(b: BuiltPart, step: float) -> np.ndarray:
    lo, hi = b.outer.min(axis=0), b.outer.max(axis=0)
    gx, gy = np.meshgrid(np.arange(lo[0] + step / 2, hi[0], step), np.arange(lo[1] + step / 2, hi[1], step))
    xy = np.c_[gx.ravel(), gy.ravel()]
    if not len(xy):
        return poly_centroid(b.outer, b.holes)[None]
    inside = st.points_in_polygon(xy, b.outer)
    for h in b.holes:
        inside &= ~st.points_in_polygon(xy, h)
    return xy[inside] if inside.any() else poly_centroid(b.outer, b.holes)[None]


def _surface_under(model: StructureModel, b: BuiltPart, xy: np.ndarray) -> np.ndarray:
    """Per sample: the highest roof surface of the other parts whose base is not above b's base (-inf where none)."""
    best = np.full(len(xy), -np.inf)
    for o in model.built:
        if o is b or o.z0 > b.z0 + 1e-6:
            continue
        z = o.surface_z(xy)
        best = np.fmax(best, np.where(np.isfinite(z), z, -np.inf))
    return best


def support_check(model: StructureModel, step: float = 0.5, gap_m: float | None = None,
                  share: float | None = None) -> list[dict[str, Any]]:
    """G10: every part with min_height > 0 must have >= `share` of its base on another part (whose surface under it is
    at most gap_m below the base) or on the ground. Returns the failures."""
    gap_m = model.params.support_gap_m if gap_m is None else gap_m
    share = model.params.support_share if share is None else share
    out = []
    for b in model.built:
        if b.z0 <= gap_m:
            continue
        xy = _base_samples(b, step)
        best = _surface_under(model, b, xy)
        ok = best >= b.z0 - gap_m
        if ok.mean() < share:
            out.append({"id": b.id, "z0": round(b.z0, 2), "supported_share": round(float(ok.mean()), 3),
                        "max_gap_m": round(float(np.max(b.z0 - best[np.isfinite(best)])) if np.isfinite(best).any() else b.z0, 2)})
    return out


def support_contact(model: StructureModel, b: BuiltPart, step: float = 0.5, gap_m: float | None = None) -> dict[str, Any]:
    """Does a raised part touch anything that can carry it? Vertical contact: base samples with a surface under them at
    most gap_m below the base (or above it: sunk into a part). Lateral contact: samples 0.5 m outside its outline that lie
    in another part (not one standing on it) whose solid spans the base level there (an arch's span between its piers,
    a bridge between towers)."""
    gap_m = model.params.support_gap_m if gap_m is None else gap_m
    xy = _base_samples(b, step)
    vertical = int((_surface_under(model, b, xy) >= b.z0 - gap_m).sum())
    ring = ccw(b.outer)
    ring_xy = []
    for k in range(len(ring)):
        a, c = ring[k], ring[(k + 1) % len(ring)]
        L = float(np.linalg.norm(c - a))
        if L < 1e-6:
            continue
        u = (c - a) / L
        n = np.array([u[1], -u[0]])                      # outward for a CCW ring
        s = np.arange(step / 2, L, step) if L > step else np.array([L / 2])
        ring_xy.append(a + s[:, None] * u + 0.5 * n)
    lateral = 0
    if ring_xy:
        rxy = np.vstack(ring_xy)
        hit = np.zeros(len(rxy), bool)
        for o in model.built:
            if o is b or o.spec.parent == b.id or o.z0 > b.z0 + gap_m:
                continue
            inside = st.points_in_polygon(rxy, o.outer)
            for h in o.holes:
                inside &= ~st.points_in_polygon(rxy, h)
            if not inside.any():
                continue
            z = o.surface_z(rxy[inside])
            hit[np.nonzero(inside)[0][np.isfinite(z) & (z >= b.z0 - gap_m)]] = True
        lateral = int(hit.sum())
    return {"vertical_samples": vertical, "lateral_samples": lateral, "touches": bool(vertical > 0 or lateral >= 2)}


def resolve_support(model: StructureModel) -> list[dict[str, Any]]:
    """G10 (docs/STRUCTURE_FIRST.md §2.8): a raised part whose base has less than support_share on a support gets a
    ticket, and until a curator decides the published model does one of two things, recorded on the ticket and the part:
    - 'rests-on-support': it touches a support (support_contact(): part of its base lies on a surface, is sunk into a
      part, or abuts a part spanning its base level -- an arch's span, an attic on columns, a gallery around a core,
      a bridge between towers). It stays where the map puts it (OSM's own values; nothing is moved or extended).
    - 'withheld': nothing under or beside its base (it floats: India Gate's top slab 2 m above the part below it). It
      is not drawn (its raised position is not published) and stays in the registry with its map values.
    Withholding a part can leave a part that stood on it unsupported: repeated until nothing changes."""
    # parts withheld earlier stay withheld (a curator decides) and keep their tickets; the others are re-checked
    model.tickets = [t for t in model.tickets if not (t.get("kind") == "support" and t.get("action") != "withheld")]
    for p in model.specs:
        if p.support != "withheld":
            p.support = None
    for _ in range(16):
        fails = {f["id"]: f for f in support_check(model)}
        float_ids = [pid for pid in fails if not support_contact(model, model.built_part(pid))["touches"]]
        if not float_ids:
            break
        for pid in float_ids:
            spec = model.part(pid)
            spec.drawn, spec.support = False, "withheld"
            f = fails[pid]
            spec.problems = [x for x in spec.problems if not x.startswith("G10")] + [
                f"G10: base {f['supported_share']:.0%} supported, nothing under or beside it (gap up to {f['max_gap_m']:g} m): "
                f"withheld (not drawn) until a curator decides"]
            model.tickets.append({"id": f"T-{model.landmark}-support-{pid}", "kind": "support", "part": pid, "open": True,
                                  "action": "withheld", "supported_share": f["supported_share"], "max_gap_m": f["max_gap_m"],
                                  "z0": f["z0"], "strength": "low"})
        model.built = build_all(model.specs, model.params)
    for f in support_check(model):
        spec = model.part(f["id"])
        c = support_contact(model, model.built_part(f["id"]))
        spec.support = "rests-on-support"
        spec.problems = [x for x in spec.problems if not x.startswith("G10")] + [
            f"G10: base {f['supported_share']:.0%} supported (< {model.params.support_share:.0%}) but it touches a support "
            f"({c['vertical_samples']} base / {c['lateral_samples']} side samples): kept on it, ticket"]
        model.tickets.append({"id": f"T-{model.landmark}-support-{f['id']}", "kind": "support", "part": f["id"], "open": True,
                              "action": "rests-on-support", "supported_share": f["supported_share"],
                              "max_gap_m": f["max_gap_m"], "z0": f["z0"], "contact": c, "strength": "low"})
    return [t for t in model.tickets if t.get("kind") == "support"]


def open_value_tickets(model: StructureModel, wikidata_height_m: float | None = None) -> list[dict[str, Any]]:
    """Tickets (never changes) for OSM values the build could not use as tagged -- unparseable numeric tags, negative or
    contradictory values, a roof higher than its part -- and for values outside sanity bounds: a part's top above
    max_plausible_m, above Wikidata's total height (P2048, design §2.4.1: only an upper bound and a check) x
    wikidata_bound, or above outlier_ratio x the top of every other part, and a conventional roof (D17, no
    roof:height) taking more than convention_share of its part's height. Idempotent (replaces the earlier value
    tickets)."""
    model.tickets = [t for t in model.tickets if t.get("kind") != "osm-value"]
    P = model.params
    tops = {p.id: float(p.params["height"].value) for p in model.specs if p.drawn and "height" in p.params}
    n = 0
    for p in model.specs:
        issues = list(p.issues)
        if "height" in p.params and p.drawn:
            top = float(p.params["height"].value)
            bounds = []
            if top > P.max_plausible_m:
                bounds.append(f"top {top:g} m above the plausible maximum {P.max_plausible_m:g} m")
            if wikidata_height_m and wikidata_height_m > 0 and top > P.wikidata_bound * wikidata_height_m:
                bounds.append(f"top {top:g} m above Wikidata's total height {wikidata_height_m:g} m x {P.wikidata_bound:g}")
            others = [v for k, v in tops.items() if k != p.id]
            if others and max(others) > 0 and top > P.outlier_ratio * max(others):
                bounds.append(f"top {top:g} m above {P.outlier_ratio:g} x every other part's top ({max(others):g} m)")
            for why in bounds:
                issues.append({"tag": "height", "value": top, "why": why + " (kept; a curator checks it)"})
                if not any(x.startswith("sanity bound:") and why in x for x in p.problems):
                    p.problems.append(f"sanity bound: {why}")
            rh = p.params.get("roof_height")
            room = top - p.z0
            if rh is not None and rh.source == st.PROV_SHAPE_DEFAULT and room > 0 and rh.value > P.convention_share * room:
                why = (f"conventional {p.roof_shape} roof {rh.value:g} m takes {rh.value / room:.0%} of the part's height "
                       f"{room:g} m (> {P.convention_share:.0%}): a wrong roof:shape?")
                issues.append({"tag": "roof:shape", "value": p.roof_tag, "why": why + " (kept; a curator checks it)"})
                if not any(x.startswith("sanity bound:") and why in x for x in p.problems):
                    p.problems.append(f"sanity bound: {why}")
        for iss in issues:
            n += 1
            model.tickets.append({"id": f"T-{model.landmark}-osm-{n}", "kind": "osm-value", "part": p.id, "open": True,
                                  "strength": "low", **iss})
    return [t for t in model.tickets if t.get("kind") == "osm-value"]


def check_structure(model: StructureModel) -> dict[str, Any]:
    """Photo-free structure gates: G1 watertight, G2 every parameter has an allowed source and state (forbidden
    sources = 0) and a 'height pending' stub source is always in state stub, G3 filter, G6 stubs only where there is
    no height fact and >= 20 m2, none with a conventional roof, G10 support: every unsupported raised part has an open
    support ticket and none that was withheld is drawn (resolve_support())."""
    g1 = [b.id for b in model.built if b.census.get("unmatched", 1) or b.census.get("duplicated", 1)]
    g2 = []
    for p in model.specs:
        for k, v in p.params.items():
            if v.source not in SOURCES or v.source in FORBIDDEN_SOURCES or v.state not in STATES:
                g2.append({"id": p.id, "param": k, "source": v.source, "state": v.state})
            elif v.source == st.PROV_DEFAULT and v.state != STATE_STUB:
                g2.append({"id": p.id, "param": k, "source": v.source, "state": v.state, "why": "a stub source must stay a stub"})
    g3 = []
    for p in model.specs:
        if p.kind == "remainder":
            continue
        cen = representative_point(p.outer, p.holes)
        if not in_zone(cen, model.main_rings, model.params.keep_buffer_m) or \
                share_inside(p.outer, p.holes, model.main_rings, model.params.keep_buffer_m) < model.params.keep_area_share:
            g3.append(p.id)
    g6 = []
    for p in model.specs:
        if not p.drawn:
            continue
        stub = p.params["height"].source == st.PROV_DEFAULT
        if stub and p.kind == "remainder" and p.area < model.params.remainder_stub_min_m2:
            g6.append({"id": p.id, "why": "stub under the minimum area"})
        if stub and p.roof_h > 0:
            g6.append({"id": p.id, "why": "conventional roof on a stub"})
    g10 = support_check(model)
    tickets = {t["part"]: t for t in model.tickets if t.get("kind") == "support" and t.get("open", True)}
    g10_open = [f["id"] for f in g10 if f["id"] not in tickets]
    g10_drawn = [pid for pid, t in tickets.items() if t.get("action") == "withheld" and model.built_part(pid) is not None]
    return {"G1_not_watertight": g1, "G2_bad_sources": g2, "G3_filter": g3, "G6_stubs": g6, "G10_unsupported": g10,
            "G10_without_ticket": g10_open, "G10_withheld_but_drawn": g10_drawn,
            "G10_withheld": sorted(pid for pid, t in tickets.items() if t.get("action") == "withheld"),
            "passed": not (g1 or g2 or g3 or g6 or g10_open or g10_drawn)}


def structure_report(model: StructureModel, census: dict[str, Any], seconds: float) -> dict[str, Any]:
    specs = model.specs
    built = {b.id: b for b in model.built}
    osm = [p for p in specs if p.kind != "remainder"]
    tagc: Counter = Counter()
    for p in osm:
        for k in ("height", "min_height", "building:levels", "building:min_level", "roof:shape", "roof:height",
                  "roof:levels", "roof:angle", "roof:orientation", "roof:direction"):
            if k in p.tags:
                tagc[k] += 1
    fallbacks = [{"id": b.id, "from": b.roof.meta["fallback_from"], "to": b.roof.shape, "why": b.roof.meta.get("reason", "")}
                 for b in model.built if b.roof.meta.get("fallback_from")]
    tri = sum(len(b.triangles()) for b in model.built)
    return {"landmark": model.landmark, "census": census, "osm_parts_kept": len(osm), "parts_built": len(model.built),
            "parts_not_drawn": [{"id": p.id, "problems": p.problems} for p in specs if not p.drawn or p.id not in built],
            "dropped": len(model.dropped), "dropped_reasons": dict(Counter(d["why"].split(" (")[0] for d in model.dropped)),
            "tag_coverage": {k: f"{v}/{len(osm)}" for k, v in tagc.most_common()},
            "roof_built": dict(Counter(b.roof.shape for b in model.built)),
            "roof_height_sources": dict(Counter(p.params["roof_height"].source for p in specs
                                                if p.id in built and built[p.id].roof.shape != "flat")),
            "height_sources": dict(Counter(p.params["height"].source for p in specs if p.id in built)),
            "fallbacks": fallbacks,
            "watertight_parts": sum(1 for b in model.built if not b.census.get("unmatched") and not b.census.get("duplicated")),
            "remainder": {"pieces": sum(1 for p in specs if p.kind == "remainder" and p.id in built),
                          "area_m2": round(sum(p.area for p in specs if p.kind == "remainder" and p.id in built), 1),
                          "height_sources": sorted({p.params["height"].source for p in specs if p.kind == "remainder"})},
            "triangles": int(tri), "build_s": round(seconds, 3)}
