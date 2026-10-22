"""Stylized ("toy" / anime) landmark models from facts plus observed photos, CPU only.

The pivot (small budget, no GPU photoreal rendering): instead of a dense or splatted model, the
published model is a simplified, cheerful building in the spirit of flat-colour illustration.

1. Massing from facts. The OSM footprint (ODbL) is simplified (Douglas-Peucker, near-orthogonal
   edges snapped where the outline is rectilinear, true polygons such as an octagon kept) and
   extruded. The base height comes from the SfM points on the walls (OSM height / levels, Wikidata
   height or a default only when the points say nothing). Roofs are fitted to the points inside the
   footprint: flat, or a pyramid when the inside rises; blocks standing above the fitted roof (a
   lantern, a tower, a choir) become their own prisms, recursively.
2. Observation. A wall is "observed" when posed photos look at it from the front within a sane
   angle, nothing in the massing hides it, and SfM points lie on it. Only observed walls get facade
   detail; the rest stay plain clay and are flagged locked ("待解锁"). Nothing is invented for places
   nobody photographed.
3. Facades. Each observed wall gets a rectified orthophoto: the posed photos are projected onto the
   wall plane (cameras in front, within the angle limit, not hidden by other parts of the massing),
   exposure-matched and blended with a per-pixel weighted median (which also removes passers-by). It
   is then stylized: mean-shift smoothing, k-means to <= 6 colours snapped to a cheerful palette,
   specks removed, regions redrawn as simplified polygons, dark near-rectangular regions regularized
   into window / door shapes aligned in rows. An observation mask is kept; outside it the facade is
   the plain wall colour, and check_honesty() verifies that.
4. Output: a GLB (minimal glTF 2.0 writer, no dependency) where every face group is its own node with
   provenance in its extras, plus a JSON-able summary.

Coordinates are the landmark ENU frame (x = east, y = north, z = up, metres). The GLB uses glTF's
y-up convention: ENU (x, y, z) -> glTF (x, z, -y), the same mapping as web/viewer.js.

No generative model is used. Steps where an allowed one (simplification, segmentation or
stylization of observed content only, never completion) would help are marked "GENERATIVE-OK".
"""

from __future__ import annotations

import io
import json
import math
import struct
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- provenance and palette

PROV_PHOTO = "photo-observed"        # appearance from posed photos of this very face
PROV_OSM = "osm-footprint"           # outline from OpenStreetMap (ODbL)
PROV_SFM = "sfm-points"              # height, outline or roof shape measured from the reconstruction's points
PROV_OSM_HEIGHT = "osm-height"       # OSM height=* or building:levels=* tag
PROV_WIKIDATA = "wikidata-height"    # Wikidata P2048 (height)
PROV_DEFAULT = "default-style"       # neither measured nor tagged: a neutral style default
PROVENANCES = (PROV_PHOTO, PROV_OSM, PROV_SFM, PROV_OSM_HEIGHT, PROV_WIKIDATA, PROV_DEFAULT)
PROVENANCE_KEYS = ("outline", "height", "appearance")   # every face carries all three

# Crowd3D's own cheerful palette (no brand's): material families in 2-4 values each, so that snapping
# keeps both the hue family and the light / dark structure of a facade.
PALETTE: tuple[tuple[str, tuple[int, int, int]], ...] = (
    ("snow", (250, 248, 240)),
    ("cream", (255, 240, 205)),
    ("sand", (240, 214, 160)),
    ("ochre", (226, 176, 92)),
    ("sunflower", (255, 200, 60)),
    ("stone-light", (222, 216, 206)),
    ("stone", (186, 176, 162)),
    ("stone-dark", (135, 126, 116)),
    ("peach", (255, 186, 150)),
    ("terracotta", (232, 120, 82)),
    ("brick", (196, 78, 56)),
    ("rust", (140, 62, 44)),
    ("mint", (170, 228, 200)),
    ("leaf", (96, 196, 110)),
    ("forest", (46, 128, 84)),
    ("pine", (30, 84, 64)),
    ("sky", (150, 210, 250)),
    ("lake", (72, 150, 220)),
    ("navy", (44, 78, 140)),
    ("slate", (96, 108, 128)),
    ("night", (52, 60, 80)),
    ("wood", (176, 118, 66)),
    ("cocoa", (112, 72, 44)),
    ("copper", (110, 196, 176)),
    ("teal", (50, 140, 130)),
)
PALETTE_RGB = np.array([c for _, c in PALETTE], dtype=np.uint8)
PALETTE_NAMES = tuple(n for n, _ in PALETTE)
CLAY_RGB = (233, 226, 214)           # locked (unobserved) walls: plain light clay
CLAY_ROOF_RGB = (218, 208, 194)      # roof faces nobody saw
INK_RGB = (45, 42, 50)               # window frames and outlines
SHADE_FACTOR = 0.62                  # region outlines are the region colour times this
BEVEL_SEGMENTS = 3                   # arc segments per rounded corner


def shade(rgb) -> tuple[int, int, int]:
    return tuple(int(round(v * SHADE_FACTOR)) for v in rgb)


def allowed_texture_colors() -> set[tuple[int, int, int]]:
    """Every colour a stylized facade texture may contain: palette, palette shades, ink."""
    out = {tuple(int(v) for v in c) for c in PALETTE_RGB}
    out |= {shade(c) for c in PALETTE_RGB}
    out.add(INK_RGB)
    return out


# ---------------------------------------------------------------- parameters


@dataclass(frozen=True)
class StyleParams:
    # footprint
    simplify_tol_m: float = 0.8          # Douglas-Peucker tolerance on the OSM outline
    min_edge_m: float = 1.5              # shorter edges are merged into their neighbours
    ortho_tol_deg: float = 12.0          # edges within this of the dominant axes snap to them ...
    ortho_min_fraction: float = 0.6      # ... when at least this share of the outline length is near them
    bevel_m: float = 0.3                 # rounded corners of the toy look (0 disables)
    snap_max_rotation_deg: float = 20.0  # model -> footprint snapping limits (ARCHITECTURE 3.5.7 uses 10 deg / 8 m
    snap_max_shift_m: float = 20.0       # for live models; GT-ENU of the real-photo scenes is 7-15 m off)
    # massing
    cell_m: float = 0.75                 # height-field resolution
    wall_band_m: float = 1.0             # points within this horizontal distance of an edge are on its wall
    min_edge_points: int = 40            # an edge's top height needs this many wall points
    edge_top_quantile: float = 0.97
    base_quantile: float = 0.25          # base height: this length-weighted quantile of the measured edge tops
    default_height_m: float = 15.0       # ARCHITECTURE 3.4.1: no tag, no points -> 15 m
    level_height_m: float = 3.0          # building:levels -> metres
    roof_min_rise_m: float = 1.5         # a pyramid needs at least this rise
    step_m: float = 3.0                  # cells this far above the fitted roof form a block of their own
    min_block_m2: float = 6.0
    max_depth: int = 3
    # observation and orthophotos
    px_per_m: float = 24.0               # facade texture resolution ...
    max_texture_px: int = 640            # ... capped on the longer side
    vis_step_m: float = 0.5              # visibility / occlusion test grid
    max_view_angle_deg: float = 72.0     # camera ray vs wall normal
    min_camera_distance_m: float = 2.0   # cameras closer to the wall plane are skipped
    min_views: int = 2                   # a facade pixel is observed when this many photos see it ...
    support_band_m: float = 0.75         # ... and SfM points lie on the wall plane within this ...
    support_radius_m: float = 1.5        # ... horizontally / vertically this close
    max_views_per_wall: int = 24         # best-scoring cameras blended per wall
    min_wall_cameras: int = 2            # a wall is observed when this many cameras see part of it ...
    min_wall_visible_fraction: float = 0.15   # ... (this share of its visibility grid each)
    min_wall_points: int = 20            # ... this many SfM points lie on it ...
    min_observed_fraction: float = 0.10  # ... and its observation mask covers at least this share
    # stylization
    max_colors: int = 6
    min_region_m2: float = 0.10          # smaller colour specks merge into their surroundings
    region_simplify_m: float = 0.06      # polygon simplification of colour regions
    region_outline: bool = True          # thin darker outline around colour regions
    window_dark_delta: float = 18.0      # openings are at least this much darker (CIE L*) than the wall
    window_min_m: tuple[float, float] = (0.35, 0.5)   # (width, height) range of regularized openings
    window_max_m: tuple[float, float] = (5.0, 9.0)
    window_min_fill: float = 0.5         # region area / bounding box area
    window_align_m: float = 0.35         # openings whose tops / bottoms differ less share them


DEFAULT_PARAMS = StyleParams()


# ---------------------------------------------------------------- polygons


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


def _line_intersection(p1, d1, p2, d2) -> np.ndarray | None:
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    t = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
    return np.asarray(p1, float) + t * np.asarray(d1, float)


def drop_short_edges(poly: np.ndarray, min_edge: float) -> np.ndarray:
    """Remove vertices so that no edge is shorter than min_edge (keeps at least a triangle)."""
    p = [np.asarray(v, float) for v in ensure_ccw(poly)]
    while len(p) > 3:
        lengths = [np.linalg.norm(p[(k + 1) % len(p)] - p[k]) for k in range(len(p))]
        k = int(np.argmin(lengths))
        if lengths[k] >= min_edge:
            break
        # drop whichever of the short edge's two vertices bends the outline least
        a, b = p[k], p[(k + 1) % len(p)]
        prev, nxt = p[k - 1], p[(k + 2) % len(p)]
        da = abs(_cross2(b - prev, a - prev)) / max(np.linalg.norm(b - prev), 1e-9)
        db = abs(_cross2(nxt - a, b - a)) / max(np.linalg.norm(nxt - a), 1e-9)
        del p[k if da <= db else (k + 1) % len(p)]
    return ensure_ccw(np.array(p))


def dominant_axis_deg(poly: np.ndarray) -> float:
    """Edge-length weighted dominant direction modulo 90 deg, in [0, 90)."""
    p = ensure_ccw(poly)
    d = np.roll(p, -1, axis=0) - p
    ang = np.arctan2(d[:, 1], d[:, 0]) * 4.0          # x4 folds the four axis directions together
    w = np.linalg.norm(d, axis=1)
    m = math.atan2(float((w * np.sin(ang)).sum()), float((w * np.cos(ang)).sum())) / 4.0
    return math.degrees(m) % 90.0


def orthogonalize(poly: np.ndarray, tol_deg: float, min_fraction: float) -> tuple[np.ndarray, bool]:
    """Snap near-axis edges to the dominant orthogonal axes when the outline is mostly rectilinear.

    Edges off the axes (an octagon's diagonals, an apse) are kept. Returns the ring and whether
    snapping was applied.
    """
    p = ensure_ccw(poly)
    axis = math.radians(dominant_axis_deg(p))
    d = np.roll(p, -1, axis=0) - p
    lengths = np.linalg.norm(d, axis=1)
    ang = np.arctan2(d[:, 1], d[:, 0])
    off = (ang - axis + math.pi / 4) % (math.pi / 2) - math.pi / 4      # offset to the nearest axis
    near = np.abs(off) <= math.radians(tol_deg)
    if lengths[near].sum() < min_fraction * lengths.sum():
        return p, False
    lines = []
    for k in range(len(p)):
        mid = (p[k] + p[(k + 1) % len(p)]) / 2
        a = ang[k] - off[k] if near[k] else ang[k]
        lines.append((mid, np.array([math.cos(a), math.sin(a)])))
    out = []
    for k in range(len(p)):
        x = _line_intersection(lines[k - 1][0], lines[k - 1][1], lines[k][0], lines[k][1])
        if x is None or np.linalg.norm(x - p[k]) > 3.0 * max(lengths[k - 1], lengths[k], 1.0):
            continue                                   # parallel neighbours: the vertex disappears
        out.append(x)
    q = np.array(out)
    if len(q) < 3 or abs(signed_area(q)) < 0.5 * abs(signed_area(p)):
        return p, False
    return ensure_ccw(q), True


def remove_collinear(poly: np.ndarray, tol_deg: float = 3.0) -> np.ndarray:
    p = ensure_ccw(poly)
    keep = []
    for k in range(len(p)):
        a, b, c = p[k - 1], p[k], p[(k + 1) % len(p)]
        u, v = b - a, c - b
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or nv < 1e-6:
            continue
        if abs(_cross2(u, v)) / (nu * nv) > math.sin(math.radians(tol_deg)) or np.dot(u, v) < 0:
            keep.append(k)
    return p[keep] if len(keep) >= 3 else p


def simplify_footprint(poly: np.ndarray, params: StyleParams = DEFAULT_PARAMS,
                       tol_m: float | None = None) -> tuple[np.ndarray, dict[str, Any]]:
    """OSM outline -> clean toy outline. Returns the CCW ring and a report."""
    p0 = ensure_ccw(poly)
    p = simplify_polygon(p0, params.simplify_tol_m if tol_m is None else tol_m)
    p = drop_short_edges(p, params.min_edge_m)
    p, snapped = orthogonalize(p, params.ortho_tol_deg, params.ortho_min_fraction)
    p = remove_collinear(drop_short_edges(p, params.min_edge_m))
    report = {
        "input_vertices": int(len(p0)),
        "output_vertices": int(len(p)),
        "orthogonal_snap": bool(snapped),
        "dominant_axis_deg": round(dominant_axis_deg(p0), 1),
        "area_input_m2": round(abs(signed_area(p0)), 1),
        "area_output_m2": round(abs(signed_area(p)), 1),
        "hausdorff_m": round(_outline_hausdorff(p0, p), 2),
    }
    return p, report


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


def _outline_hausdorff(p: np.ndarray, q: np.ndarray, step: float = 0.25) -> float:
    def dense(poly):
        out = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            n = max(1, int(np.linalg.norm(b - a) / step))
            out.append(a + np.linspace(0, 1, n, endpoint=False)[:, None] * (b - a))
        return np.vstack(out)
    return float(max(distance_to_outline(dense(p), q)[0].max(), distance_to_outline(dense(q), p)[0].max()))


def triangulate_polygon(poly: np.ndarray) -> list[tuple[int, int, int]]:
    """Ear clipping for a simple CCW polygon (a few hundred vertices at most). Triangles are CCW."""
    p = ensure_ccw(poly)
    idx = list(range(len(p)))
    tris: list[tuple[int, int, int]] = []

    def is_ear(i0, i1, i2) -> bool:
        a, b, c = p[i0], p[i1], p[i2]
        if _cross2(b - a, c - b) <= 1e-12:
            return False
        for j in idx:
            if j in (i0, i1, i2):
                continue
            q = p[j]
            if _cross2(b - a, q - a) >= 0 and _cross2(c - b, q - b) >= 0 and _cross2(a - c, q - c) >= 0:
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
            break                                     # degenerate input: fan the rest
    for k in range(1, len(idx) - 1):
        tris.append((idx[0], idx[k], idx[k + 1]))
    return tris


def bevel_polygon(poly: np.ndarray, radius: float, segments: int = 3) -> tuple[np.ndarray, list[int]]:
    """Round every corner with a short arc (the rounded toy look).

    Returns the new ring and, per new vertex, the original edge it lies on: corner k produces
    segments + 1 vertices; the first lies on edge k-1, the last on edge k, the ones between on no
    edge (-1).
    """
    p = ensure_ccw(poly)
    if radius <= 0:
        return p, list(range(len(p)))
    out, owner = [], []
    n = len(p)
    for k in range(n):
        prev, cur, nxt = p[k - 1], p[k], p[(k + 1) % n]
        u = (prev - cur) / max(np.linalg.norm(prev - cur), 1e-9)
        v = (nxt - cur) / max(np.linalg.norm(nxt - cur), 1e-9)
        r = min(radius, 0.3 * np.linalg.norm(prev - cur), 0.3 * np.linalg.norm(nxt - cur))
        a, b = cur + u * r, cur + v * r
        for s in range(segments + 1):          # quadratic Bezier with the corner as control point
            t = s / segments
            out.append((1 - t) ** 2 * a + 2 * (1 - t) * t * cur + t ** 2 * b)
            owner.append((k - 1) % n if s == 0 else (k if s == segments else -1))
    return np.array(out), owner


# ---------------------------------------------------------------- snapping the model onto the footprint


@dataclass
class FootprintSnap:
    rotation_deg: float                 # counter-clockwise, about the ENU origin
    shift: np.ndarray                   # (2,) metres, applied after the rotation
    median_before_m: float
    median_after_m: float
    num_points: int
    accepted: bool
    reason: str = ""

    def rotation(self) -> np.ndarray:
        c, s = math.cos(math.radians(self.rotation_deg)), math.sin(math.radians(self.rotation_deg))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_meta(self) -> dict[str, Any]:
        return {"rotation_deg": round(self.rotation_deg, 2), "shift_m": [round(float(v), 2) for v in self.shift],
                "median_wall_distance_before_m": round(self.median_before_m, 3),
                "median_wall_distance_after_m": round(self.median_after_m, 3),
                "num_points": self.num_points, "accepted": self.accepted, "reason": self.reason}


def _rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((dst - md).T @ (src - ms))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    return r, md - r @ ms


class _OutlineLookup:
    """Closest outline point by table lookup: the outline rasterized at `res`, nearest outline pixel of
    every grid pixel from cv2.distanceTransformWithLabels. ~res/2 accurate, O(1) per query."""

    def __init__(self, poly: np.ndarray, margin: float, res: float = 0.1) -> None:
        self.lo = poly.min(axis=0) - margin
        hi = poly.max(axis=0) + margin
        self.res = res
        self.w, self.h = int(math.ceil((hi[0] - self.lo[0]) / res)), int(math.ceil((hi[1] - self.lo[1]) / res))
        img = np.full((self.h, self.w), 255, np.uint8)
        px = np.round((np.vstack([poly, poly[:1]]) - self.lo) / res * 16).astype(np.int32)
        cv2.polylines(img, [px], False, 0, 1, cv2.LINE_8, shift=4)
        _, labels = cv2.distanceTransformWithLabels(img, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        ys, xs = np.nonzero(img == 0)
        lut = np.zeros((int(labels.max()) + 1, 2))
        lut[labels[ys, xs]] = np.stack([xs, ys], 1)
        self.nearest = lut[labels]                      # (h, w, 2) pixel coordinates of the nearest outline pixel

    def closest(self, xy: np.ndarray) -> np.ndarray:
        ij = np.floor((xy - self.lo) / self.res).astype(int)
        ij[:, 0] = np.clip(ij[:, 0], 0, self.w - 1)
        ij[:, 1] = np.clip(ij[:, 1], 0, self.h - 1)
        return self.lo + (self.nearest[ij[:, 1], ij[:, 0]] + 0.5) * self.res


def snap_to_footprint(points: np.ndarray, footprint: np.ndarray, params: StyleParams = DEFAULT_PARAMS,
                      z_band: tuple[float, float] = (2.0, 20.0), max_points: int = 6000,
                      seed: int = 0) -> FootprintSnap:
    """ARCHITECTURE 3.5.7: trimmed 2-D rigid ICP of wall-height points onto the footprint outline.

    Starts from no shift and from the shift putting the wall-band points' median on the footprint
    centroid, each with three start rotations; keeps the closest fit. Rejected (identity) beyond
    snap_max_rotation_deg / snap_max_shift_m or without improvement.
    """
    pts = np.asarray(points, float)
    fp = ensure_ccw(footprint)
    centroid = polygon_centroid(fp)
    radius = float(np.max(np.linalg.norm(fp - centroid, axis=1)))
    band = (pts[:, 2] >= z_band[0]) & (pts[:, 2] <= z_band[1])
    near = np.linalg.norm(pts[:, :2] - centroid, axis=1) <= radius + params.snap_max_shift_m
    xy = pts[band & near, :2]
    if len(xy) < 50:
        return FootprintSnap(0.0, np.zeros(2), float("nan"), float("nan"), int(len(xy)), False, "too few wall points")
    if len(xy) > max_points:
        xy = xy[np.random.default_rng(seed).choice(len(xy), max_points, replace=False)]
    lookup = _OutlineLookup(fp, radius + 2 * params.snap_max_shift_m)
    before = float(np.median(distance_to_outline(xy, fp)[0]))
    best = None
    for shift0 in (np.zeros(2), centroid - np.median(xy, axis=0)):
        for rot0 in (-10.0, 0.0, 10.0):
            c, s = math.cos(math.radians(rot0)), math.sin(math.radians(rot0))
            r = np.array([[c, -s], [s, c]])
            t = shift0 + centroid - r @ centroid
            for _ in range(100):
                moved = xy @ r.T + t
                target = lookup.closest(moved)
                d = np.linalg.norm(moved - target, axis=1)
                keep = d <= np.quantile(d, 0.7)
                dr, dt = _rigid_2d(moved[keep], target[keep])
                r, t = dr @ r, dr @ t + dt
                if np.linalg.norm(dt) < 1e-4 and abs(math.atan2(dr[1, 0], dr[0, 0])) < 1e-6:
                    break
            after = float(np.median(np.linalg.norm(xy @ r.T + t - lookup.closest(xy @ r.T + t), axis=1)))
            if best is None or after < best[0] - 1e-6:
                best = (after, r, t)
    _, r, t = best
    after = float(np.median(distance_to_outline(xy @ r.T + t, fp)[0]))
    rot = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    snap = FootprintSnap(rot, t, before, after, int(len(xy)), True)
    if abs(rot) > params.snap_max_rotation_deg or np.linalg.norm(t) > params.snap_max_shift_m:
        snap.accepted, snap.reason = False, "correction beyond the snapping limits"
    elif after >= before:
        snap.accepted, snap.reason = False, "no improvement"
    if not snap.accepted:
        snap.rotation_deg, snap.shift = 0.0, np.zeros(2)
    return snap


def apply_snap_points(snap: FootprintSnap, points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, float).copy()
    out[:, :2] = out[:, :2] @ snap.rotation()[:2, :2].T + snap.shift
    return out


# ---------------------------------------------------------------- cameras


@dataclass
class PosedCamera:
    """A registered photo: pinhole intrinsics (plus optional radial k1, k2) and its pose in ENU."""

    name: str
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    rot: np.ndarray                      # (3, 3) cam_from_ENU, COLMAP convention (x right, y down, z forward)
    center: np.ndarray                   # (3,) ENU
    k1: float = 0.0
    k2: float = 0.0
    image_path: Path | None = None       # pixels exactly as the SfM saw them (no EXIF rotation applied)
    image: np.ndarray | None = None      # or the RGB array itself

    def project(self, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ENU points -> pixel coordinates (n, 2) and depth along the optical axis (n,)."""
        pc = (np.asarray(xyz, float).reshape(-1, 3) - self.center) @ self.rot.T
        z = pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            x, y = pc[:, 0] / z, pc[:, 1] / z
        if self.k1 or self.k2:
            r2 = x * x + y * y
            f = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
            x, y = x * f, y * f
        return np.stack([self.fx * x + self.cx, self.fy * y + self.cy], 1), z

    def load(self) -> np.ndarray:
        if self.image is not None:
            return self.image
        with Image.open(self.image_path) as im:          # no exif_transpose: SfM used the stored pixels
            return np.asarray(im.convert("RGB"))

    def snapped(self, snap: FootprintSnap) -> "PosedCamera":
        """The same camera after the model was moved onto the footprint (points' = R p + t)."""
        r = snap.rotation()
        c = self.center.copy()
        c[:2] = r[:2, :2] @ c[:2] + snap.shift
        out = PosedCamera(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        out.rot, out.center = self.rot @ r.T, c
        return out


# ---------------------------------------------------------------- massing


@dataclass
class HeightFacts:
    """Heights the product knows without photos (all optional)."""

    osm_height_m: float | None = None
    osm_levels: float | None = None
    wikidata_height_m: float | None = None


@dataclass
class Prism:
    """One extruded block: a CCW outline from z0 to z1 plus a cap (flat or pyramid)."""

    index: int
    parent: int | None
    depth: int
    polygon: np.ndarray                 # (k, 2) CCW
    z0: float
    z1: float
    outline_source: str                 # PROV_OSM (base) or PROV_SFM (blocks found in the points)
    height_source: str
    cap: str = "flat"                   # "flat" | "pyramid"
    apex: np.ndarray | None = None      # (3,) for a pyramid
    cap_source: str = PROV_DEFAULT      # roof shape: PROV_SFM when fitted to points
    measured_fraction: float = 0.0      # share of this block's height-field cells that had points

    def to_meta(self) -> dict[str, Any]:
        return {"index": self.index, "parent": self.parent, "depth": self.depth, "vertices": int(len(self.polygon)),
                "area_m2": round(abs(signed_area(self.polygon)), 1), "z0": round(self.z0, 2), "z1": round(self.z1, 2),
                "outline_source": self.outline_source, "height_source": self.height_source, "cap": self.cap,
                "apex_z": None if self.apex is None else round(float(self.apex[2]), 2), "cap_source": self.cap_source,
                "measured_fraction": round(self.measured_fraction, 3)}

    def roof_height(self, xy: np.ndarray) -> np.ndarray:
        """Height of this block's cap above each (x, y) inside its outline."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.cap != "pyramid":
            return np.full(len(xy), self.z1)
        return self.z1 + (self.apex[2] - self.z1) * (1.0 - _radial_fraction(xy, self.polygon, self.apex[:2]))


@dataclass
class Wall:
    index: int
    prism: int
    edge: int                           # edge index in the prism's (un-bevelled) outline
    a: np.ndarray                       # (2,) start, CCW order along the outline
    b: np.ndarray
    z0: float
    z1: float
    outline_source: str
    height_source: str
    top_measured_m: float | None = None  # this edge's own top from its wall points, when measured

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.b - self.a))

    @property
    def tangent(self) -> np.ndarray:
        return (self.b - self.a) / max(self.length, 1e-9)

    @property
    def normal(self) -> np.ndarray:
        t = self.tangent
        return np.array([t[1], -t[0]])  # outward for a CCW outline

    @property
    def area(self) -> float:
        return self.length * (self.z1 - self.z0)

    def texture_size(self, params: StyleParams) -> tuple[int, int, float]:
        """(width px, height px, px per metre) of this wall's facade texture."""
        ppm = params.px_per_m
        longest = max(self.length, self.z1 - self.z0)
        if longest * ppm > params.max_texture_px:
            ppm = params.max_texture_px / longest
        return max(2, int(round(self.length * ppm))), max(2, int(round((self.z1 - self.z0) * ppm))), ppm

    def grid(self, w: int, h: int, offset: float = 0.0) -> np.ndarray:
        """(h, w, 3) ENU sample centres; row 0 is the top of the wall, column 0 its start a.
        `offset` moves the plane along the outward normal."""
        u = (np.arange(w) + 0.5) / w * self.length
        z = self.z1 - (np.arange(h) + 0.5) / h * (self.z1 - self.z0)
        xy = self.a[None, :] + u[:, None] * self.tangent[None, :] + offset * self.normal[None, :]
        out = np.empty((h, w, 3))
        out[..., 0] = xy[None, :, 0]
        out[..., 1] = xy[None, :, 1]
        out[..., 2] = z[:, None]
        return out


@dataclass
class Massing:
    prisms: list[Prism]
    walls: list[Wall]
    report: dict[str, Any]

    def triangles(self, skip_wall: int | None = None) -> np.ndarray:
        """(t, 3, 3) ENU triangles of the whole massing (un-bevelled), for occlusion tests."""
        tris = []
        for w in self.walls:
            if w.index == skip_wall:
                continue
            a0, b0 = np.r_[w.a, w.z0], np.r_[w.b, w.z0]
            a1, b1 = np.r_[w.a, w.z1], np.r_[w.b, w.z1]
            tris += [(a0, b0, b1), (a0, b1, a1)]
        for pr in self.prisms:
            p = pr.polygon
            if pr.cap == "pyramid":
                for k in range(len(p)):
                    tris.append((np.r_[p[k], pr.z1], np.r_[p[(k + 1) % len(p)], pr.z1], pr.apex))
            else:
                for i, j, k in triangulate_polygon(p):
                    tris.append((np.r_[p[i], pr.z1], np.r_[p[j], pr.z1], np.r_[p[k], pr.z1]))
        return np.array(tris, dtype=float).reshape(-1, 3, 3)


def height_field(points: np.ndarray, footprint: np.ndarray, cell: float,
                 min_points: int = 3, q: float = 0.9) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Per grid cell inside the footprint: robust top height of the points standing on it (NaN: no data).

    Returns (heights (ny, nx), counts, inside mask, (x0, y0) of the grid's lower-left corner).
    Row 0 is the southernmost row.
    """
    fp = ensure_ccw(footprint)
    lo = fp.min(axis=0) - cell
    hi = fp.max(axis=0) + cell
    nx, ny = int(math.ceil((hi[0] - lo[0]) / cell)), int(math.ceil((hi[1] - lo[1]) / cell))
    gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cell, lo[1] + (np.arange(ny) + 0.5) * cell)
    inside = points_in_polygon(np.stack([gx.ravel(), gy.ravel()], 1), fp).reshape(ny, nx)
    pts = np.asarray(points, float)
    ix = np.floor((pts[:, 0] - lo[0]) / cell).astype(int)
    iy = np.floor((pts[:, 1] - lo[1]) / cell).astype(int)
    ok = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny) & (pts[:, 2] > 0.5)
    ix, iy, z = ix[ok], iy[ok], pts[ok, 2]
    keep = inside[iy, ix]
    ix, iy, z = ix[keep], iy[keep], z[keep]
    flat = iy * nx + ix
    order = np.lexsort((z, flat))
    flat, z = flat[order], z[order]
    heights = np.full(ny * nx, np.nan)
    counts = np.bincount(flat, minlength=ny * nx)
    starts = np.concatenate([[0], np.cumsum(counts)])
    for k in np.nonzero(counts >= min_points)[0]:
        seg = z[starts[k]:starts[k + 1]]
        heights[k] = seg[min(len(seg) - 1, int(q * (len(seg) - 1) + 0.5))]
    return heights.reshape(ny, nx), counts.reshape(ny, nx), inside, (float(lo[0]), float(lo[1]))


def _nan_median_filter(a: np.ndarray, r: int) -> np.ndarray:
    pad = np.pad(a, r, mode="constant", constant_values=np.nan)
    stack = [pad[dy:dy + a.shape[0], dx:dx + a.shape[1]] for dy in range(2 * r + 1) for dx in range(2 * r + 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = np.nanmedian(np.stack(stack), axis=0)
    out[np.isnan(a)] = np.nan
    return out


def _radial_fraction(xy: np.ndarray, poly: np.ndarray, apex_xy: np.ndarray) -> np.ndarray:
    """For points inside a polygon star-shaped from apex_xy: |x - apex| / |boundary point on that ray - apex|."""
    d = np.asarray(xy, float) - apex_xy
    dist = np.linalg.norm(d, axis=1)
    u = d / np.maximum(dist, 1e-9)[:, None]
    a = np.asarray(poly, float) - apex_xy
    b = np.roll(a, -1, axis=0)
    e = b - a
    # ray s*u hits edge a + t*e: s*u = a + t*e
    den = u[:, None, 0] * e[None, :, 1] - u[:, None, 1] * e[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        s = (a[None, :, 0] * e[None, :, 1] - a[None, :, 1] * e[None, :, 0]) / den
        t = (a[None, :, 0] * u[:, None, 1] - a[None, :, 1] * u[:, None, 0]) / den
    ok = (np.abs(den) > 1e-12) & (t >= -1e-9) & (t <= 1 + 1e-9) & (s > 1e-9)
    s = np.where(ok, s, np.inf).min(axis=1)
    return np.clip(dist / np.maximum(s, 1e-9), 0.0, 1.0)


def _star_shaped_from(poly: np.ndarray, apex_xy: np.ndarray, margin: float = 0.05) -> bool:
    p = np.asarray(poly, float)
    q = np.roll(p, -1, axis=0)
    e = q - p
    rel = apex_xy - p
    cross = e[:, 0] * rel[:, 1] - e[:, 1] * rel[:, 0]
    return bool(np.all(cross > margin * np.linalg.norm(e, axis=1)))


def _cell_centres(shape: tuple[int, int], origin: tuple[float, float], cell: float) -> np.ndarray:
    ny, nx = shape
    gx, gy = np.meshgrid(origin[0] + (np.arange(nx) + 0.5) * cell, origin[1] + (np.arange(ny) + 0.5) * cell)
    return np.stack([gx, gy], axis=-1)


def _contour_polygons(mask: np.ndarray, cell: float, origin: tuple[float, float], min_area_m2: float) -> list[np.ndarray]:
    """External contours of a boolean grid mask (row 0 = south) as ENU polygons (upsampled for smoothness)."""
    up = 4
    m = cv2.resize(mask.astype(np.uint8) * 255, (mask.shape[1] * up, mask.shape[0] * up), interpolation=cv2.INTER_LINEAR)
    m = (cv2.GaussianBlur(m, (0, 0), up * 0.6) > 127).astype(np.uint8)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        c = c[:, 0, :].astype(float)
        xy = np.stack([origin[0] + (c[:, 0] + 0.5) / up * cell, origin[1] + (c[:, 1] + 0.5) / up * cell], 1)
        if len(xy) >= 3 and abs(signed_area(xy)) >= min_area_m2:
            out.append(ensure_ccw(xy))
    return out


def edge_tops(poly: np.ndarray, points: np.ndarray, params: StyleParams) -> list[float | None]:
    """Per outline edge: the top of its wall from the points standing on it (None: too few points)."""
    p = ensure_ccw(poly)
    pts = np.asarray(points, float)
    out: list[float | None] = []
    for k in range(len(p)):
        a, b = p[k], p[(k + 1) % len(p)]
        L = float(np.linalg.norm(b - a))
        t = (b - a) / max(L, 1e-9)
        n = np.array([t[1], -t[0]])
        rel = pts[:, :2] - a
        u, d = rel @ t, rel @ n
        sel = (np.abs(d) <= params.wall_band_m) & (u >= 0.1 * L) & (u <= 0.9 * L) & (pts[:, 2] > 1.0)
        out.append(float(np.quantile(pts[sel, 2], params.edge_top_quantile)) if sel.sum() >= params.min_edge_points
                   else None)
    return out


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    return float(v[min(len(v) - 1, int(np.searchsorted(c, q * c[-1])))])


def _fit_cap(pr: Prism, cells_xy: np.ndarray, cells_h: np.ndarray, params: StyleParams) -> np.ndarray:
    """Fit a flat or pyramid cap to the height-field cells of a block; returns the cells' roof heights."""
    flat = np.full(len(cells_h), pr.z1)
    if len(cells_h) < 12:
        return flat
    pr.cap_source = PROV_SFM
    hull = cv2.convexHull(pr.polygon.astype(np.float32))[:, 0, :]
    convexity = abs(signed_area(pr.polygon)) / max(abs(signed_area(hull)), 1e-9)
    if convexity < 0.8:
        return flat
    # apex over the highest cells (lantern / spire), else the centroid
    top = cells_h >= np.quantile(cells_h, 0.97)
    apex_xy = cells_xy[top].mean(axis=0)
    if not _star_shaped_from(pr.polygon, apex_xy):
        apex_xy = polygon_centroid(pr.polygon)
        if not _star_shaped_from(pr.polygon, apex_xy):
            return flat
    w = 1.0 - _radial_fraction(cells_xy, pr.polygon, apex_xy)
    use = (w >= 0.2) & (cells_h >= pr.z1 - 1.0)
    if use.sum() < 8:
        return flat
    rise = float(np.median((cells_h[use] - pr.z1) / w[use]))
    for _ in range(2):                                  # cells far above the pyramid are blocks, not roof
        resid = cells_h - (pr.z1 + rise * w)
        keep = use & (resid < params.step_m)
        if keep.sum() < 8:
            break
        rise = float(np.median((cells_h[keep] - pr.z1) / w[keep]))
    mad_flat = float(np.median(np.abs(cells_h[use] - pr.z1)))
    mad_pyr = float(np.median(np.abs(cells_h[use] - pr.z1 - rise * w[use])))
    if rise < params.roof_min_rise_m or mad_pyr > 0.8 * mad_flat:
        return flat
    pr.cap, pr.apex = "pyramid", np.array([apex_xy[0], apex_xy[1], pr.z1 + rise])
    return pr.z1 + rise * w


def build_massing(footprint: np.ndarray, points: np.ndarray | None, facts: HeightFacts | None = None,
                  params: StyleParams = DEFAULT_PARAMS) -> Massing:
    """Base block from the simplified footprint, roofs fitted to the points, blocks above roofs found
    recursively ("wedding cake" with pyramids).

    Base height: the base_quantile (length-weighted) of the measured edge tops, so that a lower annex
    sets the base and taller parts become blocks of their own. Without usable points the footprint is
    extruded to the OSM / Wikidata height or the style default, with a flat cap.
    """
    facts = facts or HeightFacts()
    t0 = time.perf_counter()
    simple, fp_report = simplify_footprint(footprint, params)
    report: dict[str, Any] = {"footprint": fp_report}
    fallback_h, fallback_src = params.default_height_m, PROV_DEFAULT
    if facts.wikidata_height_m:
        fallback_h, fallback_src = float(facts.wikidata_height_m), PROV_WIKIDATA
    if facts.osm_height_m:
        fallback_h, fallback_src = float(facts.osm_height_m), PROV_OSM_HEIGHT
    elif facts.osm_levels:
        fallback_h, fallback_src = float(facts.osm_levels) * params.level_height_m, PROV_OSM_HEIGHT

    pts = None if points is None or not len(points) else np.asarray(points, float)
    tops = edge_tops(simple, pts, params) if pts is not None else [None] * len(simple)
    lengths = np.linalg.norm(np.roll(simple, -1, axis=0) - simple, axis=1)
    measured = [k for k, v in enumerate(tops) if v is not None]
    if measured:
        base_h = _weighted_quantile(np.array([tops[k] for k in measured]), lengths[measured], params.base_quantile)
        base_src = PROV_SFM
    else:
        base_h, base_src = fallback_h, fallback_src
    report["edge_tops_m"] = [None if v is None else round(v, 2) for v in tops]
    report["base_height"] = {"m": round(base_h, 2), "source": base_src}
    report["fallback_height"] = {"m": fallback_h, "source": fallback_src}

    prisms = [Prism(0, None, 0, simple, 0.0, base_h, PROV_OSM, base_src)]
    if pts is not None:
        cell = params.cell_m
        heights, counts, inside, origin = height_field(pts, simple, cell)
        heights = _nan_median_filter(heights, 1)
        centres = _cell_centres(heights.shape, origin, cell)
        report["measured_cell_fraction"] = round(float((np.isfinite(heights) & inside).sum() / max(inside.sum(), 1)), 3)

        def grow(pr: Prism, region: np.ndarray) -> None:
            cells = region & np.isfinite(heights)
            pr.measured_fraction = float(cells.sum() / max(region.sum(), 1))
            # the outermost ring of cells holds wall tops, not roof
            interior = cells & (cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
            roof = _fit_cap(pr, centres[interior], heights[interior], params)
            if pr.depth >= params.max_depth:
                return
            above = np.zeros(heights.shape, dtype=bool)
            above[interior] = heights[interior] - roof >= params.step_m
            above = cv2.morphologyEx(above.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
            n, labels = cv2.connectedComponents(above.astype(np.uint8), connectivity=8)
            for lab in range(1, n):
                comp = labels == lab
                if comp.sum() * cell * cell < params.min_block_m2:
                    continue
                for poly in _contour_polygons(comp, cell, origin, params.min_block_m2):
                    poly, _ = simplify_footprint(poly, params, tol_m=max(params.simplify_tol_m, 0.6))
                    if len(poly) < 3 or abs(signed_area(poly)) < params.min_block_m2:
                        continue
                    sub = comp & points_in_polygon(centres.reshape(-1, 2), poly).reshape(comp.shape)
                    hs = heights[sub & np.isfinite(heights)]
                    if len(hs) < 4:
                        continue
                    z1 = float(np.median(hs))
                    z0 = float(pr.roof_height(poly).min()) if pr.cap == "pyramid" else pr.z1
                    if z1 - z0 < params.step_m:
                        continue
                    child = Prism(len(prisms), pr.index, pr.depth + 1, poly, z0, z1, PROV_SFM, PROV_SFM)
                    prisms.append(child)
                    grow(child, sub)

        grow(prisms[0], inside.copy())
    else:
        report["measured_cell_fraction"] = 0.0

    walls: list[Wall] = []
    for pr in prisms:
        poly = pr.polygon
        for k in range(len(poly)):
            w = Wall(len(walls), pr.index, k, poly[k].copy(), poly[(k + 1) % len(poly)].copy(),
                     pr.z0, pr.z1, pr.outline_source, pr.height_source)
            if pr.index == 0:
                w.top_measured_m = tops[k]
            walls.append(w)
    report["prisms"] = [p.to_meta() for p in prisms]
    report["num_walls"] = len(walls)
    report["seconds"] = round(time.perf_counter() - t0, 3)
    return Massing(prisms, walls, report)


# ---------------------------------------------------------------- observation


def _ray_hits(origin: np.ndarray, targets: np.ndarray, tris: np.ndarray, t_max: float = 0.995) -> np.ndarray:
    """For rays origin -> target: does any triangle cut the segment before the target? (Moller-Trumbore)."""
    if len(tris) == 0 or len(targets) == 0:
        return np.zeros(len(targets), dtype=bool)
    d = targets - origin                                         # (m, 3)
    v0, e1, e2 = tris[:, 0], tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    hit = np.zeros(len(targets), dtype=bool)
    for s in range(0, len(targets), 2048):
        dd = d[s:s + 2048]
        p = np.cross(dd[:, None, :], e2[None])                   # (m, t, 3)
        det = (p * e1[None]).sum(-1)
        ok = np.abs(det) > 1e-9
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        tv = origin - v0                                         # (t, 3)
        u = (tv[None] * p).sum(-1) * inv
        q = np.cross(tv, e1)                                     # (t, 3)
        v = (dd[:, None, :] * q[None]).sum(-1) * inv
        t = (q[None] * e2[None]).sum(-1) * inv
        h = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-4) & (t < t_max)
        hit[s:s + 2048] = h.any(axis=1)
    return hit


@dataclass
class WallObservation:
    wall: int
    cameras: list[tuple[int, float, float]]   # (camera index, visible fraction of the wall, score)
    num_points: int
    plane_offset_m: float                      # median signed distance of the wall's points to the outline
    support: np.ndarray                        # (gh, gw) bool on the visibility grid: SfM points nearby
    observed: bool = False
    reason: str = ""

    def to_meta(self) -> dict[str, Any]:
        return {"wall": self.wall, "num_cameras": len(self.cameras), "num_points": self.num_points,
                "plane_offset_m": round(self.plane_offset_m, 2), "observed": self.observed, "reason": self.reason,
                "support_fraction": round(float(self.support.mean()), 3) if self.support.size else 0.0}


def _vis_grid_shape(wall: Wall, step: float) -> tuple[int, int]:
    return max(2, int(round((wall.z1 - wall.z0) / step))), max(2, int(round(wall.length / step)))


def _wall_points(wall: Wall, points: np.ndarray, band: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points on a wall's plane: (u along the wall, z, signed distance) for those within the band."""
    rel = points[:, :2] - wall.a
    u, d = rel @ wall.tangent, rel @ wall.normal
    sel = (np.abs(d) <= band) & (u >= 0) & (u <= wall.length) & (points[:, 2] >= wall.z0) & (points[:, 2] <= wall.z1)
    return u[sel], points[sel, 2], d[sel]


def observe_walls(massing: Massing, cameras: list[PosedCamera], points: np.ndarray | None,
                  params: StyleParams = DEFAULT_PARAMS) -> list[WallObservation]:
    """Which cameras see which walls (front side, angle, inside the image, not hidden by the massing),
    and where SfM points support each wall. Decides observed / locked per wall."""
    pts = None if points is None or not len(points) else np.asarray(points, float)
    cos_max = math.cos(math.radians(params.max_view_angle_deg))
    out = []
    for wall in massing.walls:
        gh, gw = _vis_grid_shape(wall, params.vis_step_m)
        offset, npts = 0.0, 0
        support = np.zeros((gh, gw), dtype=bool)
        if pts is not None:
            u, z, d = _wall_points(wall, pts, params.support_band_m + 1.0)
            if len(d) >= params.min_wall_points:
                offset = float(np.clip(np.median(d), -1.0, 1.0))
            near = np.abs(d - offset) <= params.support_band_m
            u, z = u[near], z[near]
            npts = int(len(u))
            if npts:
                iu = np.clip((u / wall.length * gw).astype(int), 0, gw - 1)
                iz = np.clip(((wall.z1 - z) / (wall.z1 - wall.z0) * gh).astype(int), 0, gh - 1)
                support[iz, iu] = True
                r = max(1, int(round(params.support_radius_m / params.vis_step_m)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                support = cv2.dilate(support, kernel) > 0
        grid = wall.grid(gw, gh, offset).reshape(-1, 3)
        tris = massing.triangles(skip_wall=wall.index)
        cams = []
        for ci, cam in enumerate(cameras):
            rel = cam.center[:2] - wall.a
            if rel @ wall.normal - offset < params.min_camera_distance_m:
                continue
            ray = cam.center[None] - grid
            cosang = (ray[:, :2] @ wall.normal) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9)
            uv, depth = cam.project(grid)
            vis = ((depth > 0.5) & (cosang >= cos_max) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width)
                   & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height))
            if vis.mean() < params.min_wall_visible_fraction * 0.5:
                continue
            idx = np.nonzero(vis)[0]
            vis[idx[_ray_hits(cam.center, grid[idx], tris)]] = False
            frac = float(vis.mean())
            if frac < params.min_wall_visible_fraction * 0.5:
                continue
            centre = wall.grid(1, 1, offset).reshape(3)
            dist = float(np.linalg.norm(cam.center - centre))
            ppm = cam.fx / max(dist, 1e-6)                       # image pixels per metre on the wall
            cos_c = float(((cam.center - centre)[:2] @ wall.normal) / max(dist, 1e-9))
            cams.append((ci, frac, frac * max(cos_c, 0.05) * min(ppm, 60.0)))
        cams.sort(key=lambda c: -c[2])
        ob = WallObservation(wall.index, cams, npts, offset, support)
        good = [c for c in cams if c[1] >= params.min_wall_visible_fraction]
        if len(good) < params.min_wall_cameras:
            ob.reason = f"{len(good)} camera(s) see it"
        elif pts is not None and npts < params.min_wall_points:
            ob.reason = f"only {npts} SfM points on it"
        else:
            ob.observed = True
        out.append(ob)
    return out


# ---------------------------------------------------------------- facade orthophotos


@dataclass
class Orthophoto:
    rgb: np.ndarray                     # (h, w, 3) uint8, rectified; row 0 = top of the wall
    count: np.ndarray                   # (h, w) number of photos that saw each pixel
    mask: np.ndarray                    # (h, w) bool: observed (enough views and SfM support)
    px_per_m: float
    cameras: list[str]

    @property
    def observed_fraction(self) -> float:
        return float(self.mask.mean())


def facade_orthophoto(wall: Wall, obs: WallObservation, massing: Massing, cameras: list[PosedCamera],
                      params: StyleParams = DEFAULT_PARAMS, cache: dict[int, np.ndarray] | None = None) -> Orthophoto:
    """Project the wall's best photos onto its plane and blend them (exposure-matched weighted median)."""
    w, h, ppm = wall.texture_size(params)
    gh, gw = _vis_grid_shape(wall, params.vis_step_m)
    grid = wall.grid(w, h, obs.plane_offset_m)
    vgrid = wall.grid(gw, gh, obs.plane_offset_m).reshape(-1, 3)
    tris = massing.triangles(skip_wall=wall.index)
    cos_max = math.cos(math.radians(params.max_view_angle_deg))
    stack, valid, weights, names = [], [], [], []
    for ci, frac, score in obs.cameras[:params.max_views_per_wall]:
        cam = cameras[ci]
        img = cache.get(ci) if cache is not None else None
        if img is None:
            img = cam.load()
            if cache is not None:
                cache[ci] = img
        uv, depth = cam.project(grid.reshape(-1, 3))
        ray = cam.center[None] - grid.reshape(-1, 3)
        cosang = (ray[:, :2] @ wall.normal) / np.maximum(np.linalg.norm(ray, axis=1), 1e-9)
        ok = ((depth > 0.5) & (cosang >= cos_max) & (uv[:, 0] >= 0.5) & (uv[:, 0] <= cam.width - 1.5)
              & (uv[:, 1] >= 0.5) & (uv[:, 1] <= cam.height - 1.5))
        # occlusion by the massing, on the coarse grid, upsampled
        vuv, vdepth = cam.project(vgrid)
        vok = vdepth > 0.5
        vis = np.zeros(len(vgrid), dtype=bool)
        idx = np.nonzero(vok)[0]
        vis[idx] = ~_ray_hits(cam.center, vgrid[idx], tris)
        vis = cv2.resize(vis.reshape(gh, gw).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        ok &= vis.reshape(-1)
        if ok.mean() < 0.02:
            continue
        mx = uv[:, 0].reshape(h, w).astype(np.float32)
        my = uv[:, 1].reshape(h, w).astype(np.float32)
        sample = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        stack.append(sample.astype(np.float32))
        valid.append(ok.reshape(h, w))
        weights.append(score)
        names.append(cam.name)
    if not stack:
        return Orthophoto(np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.int32), np.zeros((h, w), bool), ppm, [])
    S = np.stack(stack)                                   # (n, h, w, 3)
    V = np.stack(valid)                                   # (n, h, w)
    Wt = np.array(weights, dtype=np.float32)
    count = V.sum(axis=0).astype(np.int32)
    rgb = _weighted_median(S, V, Wt)
    for _ in range(2):                                    # exposure: per-photo gains towards the blend
        for k in range(len(S)):
            m = V[k] & (count >= 2)
            if m.sum() < 50:
                continue
            ratio = np.median(rgb[m] + 1.0, axis=0) / np.maximum(np.median(S[k][m] + 1.0, axis=0), 1.0)
            S[k] *= np.clip(ratio, 0.5, 2.0)[None, None, :]
        np.clip(S, 0, 255, out=S)
        rgb = _weighted_median(S, V, Wt)
    support = cv2.resize(obs.support.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    mask = (count >= params.min_views) & support
    # keep the mask tidy: no pinholes, no slivers (half a metre)
    k = max(3, int(round(0.5 * ppm)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = (cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0) & (count >= params.min_views) & support
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    out[count == 0] = 0
    return Orthophoto(out, count, mask, ppm, names)


def _weighted_median(S: np.ndarray, V: np.ndarray, Wt: np.ndarray) -> np.ndarray:
    """Per pixel: the sample at the weighted median of luminance among valid views (keeps RGB consistent)."""
    lum = S @ np.array([0.299, 0.587, 0.114], dtype=np.float32)       # (n, h, w)
    lum = np.where(V, lum, np.inf)
    order = np.argsort(lum, axis=0)
    w = np.where(V, Wt[:, None, None], 0.0)
    ws = np.take_along_axis(w, order, axis=0)
    c = np.cumsum(ws, axis=0)
    half = c[-1] / 2.0
    pick = np.argmax(c >= half[None] - 1e-6, axis=0)                   # (h, w)
    src = np.take_along_axis(order, pick[None], axis=0)[0]
    hh, ww = np.indices(src.shape)
    return S[src, hh, ww]


# ---------------------------------------------------------------- stylization


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """uint8 / float RGB (..., 3) in 0..255 -> CIE Lab float (L 0..100)."""
    a = np.asarray(rgb, dtype=np.float32).reshape(-1, 1, 3) / 255.0
    return cv2.cvtColor(a, cv2.COLOR_RGB2Lab).reshape(np.asarray(rgb).shape).astype(np.float32)


PALETTE_LAB = rgb_to_lab(PALETTE_RGB.astype(np.float32))


def snap_to_palette(rgb: np.ndarray, cheer: float = 1.35) -> np.ndarray:
    """Palette index of each colour (n, 3): chroma boosted by `cheer`, nearest palette entry in Lab."""
    lab = rgb_to_lab(np.asarray(rgb, np.float32).reshape(-1, 3))
    lab[:, 1:] *= cheer
    d = ((lab[:, None, :] - PALETTE_LAB[None]) ** 2).sum(-1)
    return d.argmin(axis=1)


def kmeans(x: np.ndarray, k: int, seed: int = 0, iters: int = 25, max_samples: int = 20000) -> np.ndarray:
    """Deterministic k-means (k-means++ init) on rows of x; returns the centres (k', d), k' <= k."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, np.float32)
    if len(x) > max_samples:
        x = x[rng.choice(len(x), max_samples, replace=False)]
    k = min(k, len(np.unique(x, axis=0)))
    centres = [x[rng.integers(len(x))]]
    d2 = ((x - centres[0]) ** 2).sum(1)
    for _ in range(1, k):
        if d2.sum() <= 0:
            break
        centres.append(x[rng.choice(len(x), p=d2 / d2.sum())])
        d2 = np.minimum(d2, ((x - centres[-1]) ** 2).sum(1))
    c = np.array(centres)
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(len(c))])
        if np.allclose(new, c, atol=1e-3):
            break
        c = new
    return c


@dataclass
class StylizedFacade:
    texture: np.ndarray                 # (h, w, 3) uint8, allowed colours only
    mask: np.ndarray                    # observation mask (same size)
    base_rgb: tuple[int, int, int]      # plain wall colour (outside the mask, and the wall's flat colour)
    colors: list[str]                   # palette names used
    openings: list[dict[str, Any]]      # regularized windows / doors (metres, from the top-left corner)
    report: dict[str, Any]


def _majority(labels: np.ndarray, n: int, k: int, valid: np.ndarray) -> np.ndarray:
    votes = np.stack([cv2.blur(((labels == j) & valid).astype(np.float32), (k, k)) for j in range(n)])
    return votes.argmax(0)


def stylize_facade(ortho: Orthophoto, params: StyleParams = DEFAULT_PARAMS, ground_row: bool = True,
                   seed: int = 0) -> StylizedFacade:
    """Rectified orthophoto -> flat-colour illustration of the observed part of a facade.

    GENERATIVE-OK: a segmentation model (e.g. SAM 2) could find window / door / trim regions on the
    observed orthophoto more reliably than colour clustering; an image-to-image stylizer could restyle
    the observed pixels. Neither may fill pixels outside the observation mask.
    """
    rgb, mask, ppm = ortho.rgb, ortho.mask, ortho.px_per_m
    h, w = mask.shape
    t0 = time.perf_counter()
    if mask.sum() < 20:
        base = CLAY_RGB
        tex = np.empty((h, w, 3), np.uint8)
        tex[:] = base
        return StylizedFacade(tex, mask, base, [], [], {"empty": True})
    fill = rgb.copy()
    fill[~mask] = np.median(rgb[mask], axis=0).astype(np.uint8)
    # edge-preserving smoothing ("paint"): mean shift in colour + space, in metres
    sp = max(3, int(round(0.25 * ppm)))
    smooth = cv2.pyrMeanShiftFiltering(np.ascontiguousarray(fill[..., ::-1]), sp, 22, maxLevel=1)[..., ::-1]
    lab = rgb_to_lab(smooth)
    centres = kmeans(lab[mask], params.max_colors, seed=seed)
    # cluster -> palette (clusters on the same palette colour merge)
    cent_rgb = cv2.cvtColor(centres.reshape(-1, 1, 3).astype(np.float32), cv2.COLOR_Lab2RGB).reshape(-1, 3) * 255
    pal = snap_to_palette(np.clip(cent_rgb, 0, 255))
    d = ((lab[..., None, :] - centres[None, None]) ** 2).sum(-1)
    cl = d.argmin(-1)
    used = sorted(set(pal.tolist()))
    remap = {p: i for i, p in enumerate(used)}
    labels = np.vectorize(lambda c: remap[pal[c]])(cl).astype(np.int32) if len(used) else cl
    n = len(used)
    # majority filter, then specks -> neighbours
    labels = _majority(labels, n, max(3, int(round(0.2 * ppm)) | 1), mask)
    min_px = max(4, int(params.min_region_m2 * ppm * ppm))
    for _ in range(3):
        small = np.zeros_like(mask)
        for j in range(n):
            cnt, cc, stats, _ = cv2.connectedComponentsWithStats(((labels == j) & mask).astype(np.uint8), connectivity=4)
            tiny = np.nonzero(stats[1:, cv2.CC_STAT_AREA] < min_px)[0] + 1
            if len(tiny):
                small |= np.isin(cc, tiny)
        if not small.any():
            break
        votes = np.stack([cv2.blur(((labels == j) & mask & ~small).astype(np.float32), (7, 7)) for j in range(n)])
        labels = np.where(small, votes.argmax(0), labels)
    counts = np.bincount(labels[mask], minlength=n)
    base_j = int(counts.argmax())
    base = tuple(int(v) for v in PALETTE_RGB[used[base_j]])
    used_lab = PALETTE_LAB[used]

    tex = np.empty((h, w, 3), np.uint8)
    tex[:] = base
    eps = max(1.0, params.region_simplify_m * ppm)
    line = max(1, int(round(0.04 * ppm)))
    # openings: dark, near-rectangular components (the union of all labels darker than the wall)
    dark_labels = [j for j in range(n) if used_lab[j, 0] < used_lab[base_j, 0] - params.window_dark_delta]
    dark = np.isin(labels, dark_labels) & mask
    openings = _find_openings(dark, labels, ppm, params, ground_row)
    taken = np.zeros_like(mask)
    for op in openings:
        taken |= op.pop("_mask")
    # other regions as simplified polygons, biggest first
    regions = []
    for j in range(n):
        if j == base_j:
            continue
        cnt, cc, stats, _ = cv2.connectedComponentsWithStats(((labels == j) & mask & ~taken).astype(np.uint8), connectivity=8)
        for c in range(1, cnt):
            if stats[c, cv2.CC_STAT_AREA] >= min_px:
                regions.append((int(stats[c, cv2.CC_STAT_AREA]), j, cc == c))
    regions.sort(key=lambda r: -r[0])
    for _, j, comp in regions:
        contours, _ = cv2.findContours(comp.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        rings = [cv2.approxPolyDP(c, eps, True) for c in contours if len(c) >= 3]
        rings = [r for r in rings if len(r) >= 3]
        if not rings:
            continue
        colour = tuple(int(v) for v in PALETTE_RGB[used[j]])
        cv2.fillPoly(tex, rings, colour, cv2.LINE_8)
        if params.region_outline:
            cv2.polylines(tex, rings, True, shade(colour), line, cv2.LINE_8)
    for op in openings:
        op["rgb"] = tuple(int(v) for v in PALETTE_RGB[used[op["label"]]])
        op["color"] = PALETTE_NAMES[used[op.pop("label")]]
        _draw_opening(tex, op, ppm)
        del op["rgb"]
    tex[~mask] = base                                     # honesty: nothing outside the observation mask
    names = sorted({PALETTE_NAMES[used[j]] for j in range(n) if counts[j] > 0})
    report = {"colors": names, "num_regions": len(regions), "num_openings": len(openings),
              "base": PALETTE_NAMES[used[base_j]], "seconds": round(time.perf_counter() - t0, 3)}
    return StylizedFacade(tex, mask, base, names, openings, report)


def _find_openings(dark: np.ndarray, labels: np.ndarray, ppm: float, params: StyleParams,
                   ground_row: bool) -> list[dict[str, Any]]:
    h, w = dark.shape
    k = max(3, int(round(0.15 * ppm)) | 1)
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((k, k), np.uint8))
    cnt, cc, stats, _ = cv2.connectedComponentsWithStats(dark, connectivity=8)
    found = []
    for c in range(1, cnt):
        x, y, bw, bh, area = (int(v) for v in stats[c])
        wm, hm = bw / ppm, bh / ppm
        if not (params.window_min_m[0] <= wm <= params.window_max_m[0] and params.window_min_m[1] <= hm <= params.window_max_m[1]):
            continue
        if area / max(bw * bh, 1) < params.window_min_fill:
            continue
        comp = cc == c
        top = comp[y:y + max(1, bh // 4), x:x + bw]
        arched = bool(bh >= 1.2 * bw and top.mean() < 0.72)
        door = bool(ground_row and (h - (y + bh)) / ppm < 0.4 and hm >= 1.8)
        lab = labels[comp]
        found.append({"x": x / ppm, "y": y / ppm, "w": wm, "h": hm, "arched": arched, "door": door,
                      "label": int(np.bincount(lab).argmax()), "_mask": comp})
    # regularize: openings whose tops (bottoms) are close share them; similar widths in a row equalize
    tol = params.window_align_m
    for key in ("y", "bottom"):
        vals = [(o["y"] if key == "y" else o["y"] + o["h"]) for o in found]
        order = np.argsort(vals)
        groups, cur = [], []
        for i in order:
            if cur and vals[i] - vals[cur[-1]] > tol:
                groups.append(cur)
                cur = []
            cur.append(i)
        if cur:
            groups.append(cur)
        for g in groups:
            m = float(np.median([vals[i] for i in g]))
            for i in g:
                o = found[i]
                if key == "y":
                    o["h"] += o["y"] - m
                    o["y"] = m
                else:
                    o["h"] = m - o["y"]
    rows: dict[float, list[dict[str, Any]]] = {}
    for o in found:
        rows.setdefault(round(o["y"], 3), []).append(o)
    for row in rows.values():
        ws = np.array([o["w"] for o in row])
        med = float(np.median(ws))
        for o in row:
            if abs(o["w"] - med) <= 0.25 * med:
                cx = o["x"] + o["w"] / 2
                o["x"], o["w"] = cx - med / 2, med
    return [o for o in found if o["h"] > 0.2]


def _draw_opening(tex: np.ndarray, op: dict[str, Any], ppm: float) -> None:
    """A clean window / door: rounded rectangle (arched top when the region is arched), ink frame."""
    pal_colour = op.get("rgb")
    x0, y0 = int(round(op["x"] * ppm)), int(round(op["y"] * ppm))
    x1, y1 = int(round((op["x"] + op["w"]) * ppm)), int(round((op["y"] + op["h"]) * ppm))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    colour = pal_colour if pal_colour is not None else INK_RGB
    m = np.zeros(tex.shape[:2], np.uint8)
    if op["arched"]:
        r = (x1 - x0) // 2
        cv2.rectangle(m, (x0, y0 + r), (x1, y1), 1, -1)
        cv2.ellipse(m, ((x0 + x1) // 2, y0 + r), (r, r), 0, 180, 360, 1, -1)
    else:
        r = max(1, int(0.18 * min(x1 - x0, y1 - y0)))
        cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), 1, -1)
        cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), 1, -1)
        for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
            cv2.circle(m, (cx, cy), r, 1, -1)
    tex[m > 0] = colour
    frame = max(1, int(round(0.07 * ppm)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tex, contours, -1, INK_RGB, frame, cv2.LINE_8)


# ---------------------------------------------------------------- roof colours


def roof_face_colors(massing: Massing, points: np.ndarray | None, colors: np.ndarray | None,
                     min_points: int = 40) -> dict[tuple[int, int], tuple[int, int, int] | None]:
    """(prism, face) -> palette colour of the points on that roof face, None when nobody saw it.
    Face = edge index for a pyramid, 0 for a flat cap. Only the uppermost block over a spot owns it."""
    out: dict[tuple[int, int], tuple[int, int, int] | None] = {}
    if points is None or colors is None or not len(points):
        for pr in massing.prisms:
            for f in range(len(pr.polygon) if pr.cap == "pyramid" else 1):
                out[(pr.index, f)] = None
        return out
    pts = np.asarray(points, float)
    for pr in massing.prisms:
        inside = points_in_polygon(pts[:, :2], pr.polygon)
        for other in massing.prisms:
            if other.parent == pr.index:
                inside &= ~points_in_polygon(pts[:, :2], other.polygon)
        idx = np.nonzero(inside)[0]
        roof = pr.roof_height(pts[idx, :2])
        on = np.abs(pts[idx, 2] - roof) <= 1.5
        idx = idx[on]
        if pr.cap == "pyramid":
            rel = pts[idx, :2] - pr.apex[:2]
            ang = np.arctan2(rel[:, 1], rel[:, 0])
            p = pr.polygon - pr.apex[:2]
            va = np.arctan2(p[:, 1], p[:, 0])
            face = np.zeros(len(idx), dtype=int) - 1
            for k in range(len(p)):
                a0, a1 = va[k], va[(k + 1) % len(p)]
                span = (a1 - a0) % (2 * math.pi)
                face[((ang - a0) % (2 * math.pi)) <= span] = k
            faces = range(len(p))
        else:
            face = np.zeros(len(idx), dtype=int)
            faces = range(1)
        for f in faces:
            sel = idx[face == f]
            if len(sel) < min_points:
                out[(pr.index, f)] = None
                continue
            rgb = np.median(np.asarray(colors)[sel].astype(float), axis=0)
            out[(pr.index, f)] = tuple(int(v) for v in PALETTE_RGB[snap_to_palette(rgb[None])[0]])
    return out


# ---------------------------------------------------------------- mesh parts and GLB


@dataclass
class Part:
    """One node of the GLB: a face group with one material and one provenance record."""

    name: str
    positions: np.ndarray               # (n, 3) ENU
    normals: np.ndarray                 # (n, 3) ENU
    indices: np.ndarray                 # (m, 3)
    rgb: tuple[int, int, int]
    texture: np.ndarray | None = None   # (h, w, 3) uint8
    uvs: np.ndarray | None = None
    extras: dict[str, Any] = field(default_factory=dict)


def _enu_to_gltf(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def _png_bytes(rgb: np.ndarray) -> bytes:
    """Palette PNG when the image has <= 256 colours (stylized facades have ~10)."""
    im = Image.fromarray(np.ascontiguousarray(rgb), "RGB")
    cols = np.unique(rgb.reshape(-1, 3), axis=0)
    buf = io.BytesIO()
    if len(cols) <= 256:
        im = im.quantize(colors=len(cols), method=Image.Quantize.FASTOCTREE, dither=Image.Dither.NONE)
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def write_glb(parts: list[Part], scene_extras: dict[str, Any] | None = None) -> bytes:
    """Minimal glTF 2.0 binary: one node + mesh + material per part, textures embedded as PNG."""
    bin_chunks: list[bytes] = []
    offset = 0
    gltf: dict[str, Any] = {"asset": {"version": "2.0", "generator": "crowd3d.stylize"},
                            "scene": 0, "scenes": [{"nodes": list(range(len(parts)))}], "nodes": [], "meshes": [],
                            "materials": [], "accessors": [], "bufferViews": []}
    if scene_extras:
        gltf["scenes"][0]["extras"] = scene_extras

    def view(data: bytes, target: int | None = None) -> int:
        nonlocal offset
        pad = (-len(data)) % 4
        bv = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target:
            bv["target"] = target
        bin_chunks.append(data + b"\0" * pad)
        offset += len(data) + pad
        gltf["bufferViews"].append(bv)
        return len(gltf["bufferViews"]) - 1

    def accessor(arr: np.ndarray, kind: str, target: int, minmax: bool = False) -> int:
        comp = 5126 if arr.dtype == np.float32 else (5123 if arr.dtype == np.uint16 else 5125)
        acc = {"bufferView": view(arr.tobytes(), target), "componentType": comp, "count": int(len(arr)), "type": kind}
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        gltf["accessors"].append(acc)
        return len(gltf["accessors"]) - 1

    for part in parts:
        pos = _enu_to_gltf(part.positions)
        nrm = _enu_to_gltf(part.normals)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        idx = np.asarray(part.indices).reshape(-1)
        idx = idx.astype(np.uint16 if len(pos) < 65536 else np.uint32)
        attrs = {"POSITION": accessor(pos, "VEC3", 34962, True), "NORMAL": accessor(nrm.astype(np.float32), "VEC3", 34962)}
        mat: dict[str, Any] = {"name": part.name, "doubleSided": False,
                               "pbrMetallicRoughness": {"metallicFactor": 0.0, "roughnessFactor": 1.0}}
        if part.texture is not None and part.uvs is not None:
            attrs["TEXCOORD_0"] = accessor(np.asarray(part.uvs, np.float32), "VEC2", 34962)
            gltf.setdefault("images", []).append({"bufferView": view(_png_bytes(part.texture)), "mimeType": "image/png"})
            gltf.setdefault("samplers", [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}])
            gltf.setdefault("textures", []).append({"sampler": 0, "source": len(gltf["images"]) - 1})
            mat["pbrMetallicRoughness"]["baseColorTexture"] = {"index": len(gltf["textures"]) - 1}
        else:
            mat["pbrMetallicRoughness"]["baseColorFactor"] = [_srgb_to_linear(c / 255.0) for c in part.rgb] + [1.0]
        gltf["materials"].append(mat)
        prim = {"attributes": attrs, "indices": accessor(idx, "SCALAR", 34963), "material": len(gltf["materials"]) - 1}
        gltf["meshes"].append({"name": part.name, "primitives": [prim]})
        node = {"name": part.name, "mesh": len(gltf["meshes"]) - 1}
        if part.extras:
            node["extras"] = part.extras
        gltf["nodes"].append(node)
    binary = b"".join(bin_chunks)
    gltf["buffers"] = [{"byteLength": len(binary)}]
    js = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Parse a GLB into (glTF JSON, binary chunk). Raises ValueError on a malformed file."""
    if len(data) < 20:
        raise ValueError("too short for a GLB")
    magic, version, total = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2 or total != len(data):
        raise ValueError("not a glTF 2.0 binary")
    jlen, jtype = struct.unpack_from("<II", data, 12)
    if jtype != 0x4E4F534A:
        raise ValueError("first chunk is not JSON")
    gltf = json.loads(data[20:20 + jlen].decode("utf-8"))
    binary = b""
    off = 20 + jlen
    if off < len(data):
        blen, btype = struct.unpack_from("<II", data, off)
        if btype != 0x004E4942:
            raise ValueError("second chunk is not BIN")
        binary = data[off + 8:off + 8 + blen]
    return gltf, binary


def glb_accessor(gltf: dict[str, Any], binary: bytes, index: int) -> np.ndarray:
    acc = gltf["accessors"][index]
    bv = gltf["bufferViews"][acc["bufferView"]]
    dtype = {5126: np.float32, 5123: np.uint16, 5125: np.uint32}[acc["componentType"]]
    width = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[acc["type"]]
    arr = np.frombuffer(binary, dtype=dtype, count=acc["count"] * width, offset=bv["byteOffset"] + acc.get("byteOffset", 0))
    return arr.reshape(acc["count"], width) if width > 1 else arr


def _srgb_to_linear(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


# ---------------------------------------------------------------- whole pipeline


@dataclass
class SceneInput:
    name: str
    footprint: np.ndarray               # (k, 2) ENU, OSM outline
    points: np.ndarray | None           # (n, 3) ENU, SfM points
    colors: np.ndarray | None           # (n, 3) uint8
    cameras: list[PosedCamera]
    facts: HeightFacts = field(default_factory=HeightFacts)
    attribution: list[str] = field(default_factory=list)
    snap: bool = True                   # move points + cameras onto the footprint first (ARCHITECTURE 3.5.7)


@dataclass
class StylizedModel:
    glb: bytes
    summary: dict[str, Any]
    massing: Massing
    observations: list[WallObservation]
    orthophotos: dict[int, Orthophoto]
    facades: dict[int, StylizedFacade]
    parts: list[Part]


def _provenance(outline: str, height: str, appearance: str) -> dict[str, str]:
    return {"outline": outline, "height": height, "appearance": appearance}


def build_parts(massing: Massing, observed: dict[int, StylizedFacade], roof_rgb: dict[tuple[int, int], Any],
                params: StyleParams = DEFAULT_PARAMS) -> list[Part]:
    """Bevelled toy mesh: one part per wall, per corner set, per roof face group."""
    parts: list[Part] = []
    for pr in massing.prisms:
        ring, _ = bevel_polygon(pr.polygon, params.bevel_m, BEVEL_SEGMENTS)
        n = len(ring)
        walls = {w.edge: w for w in massing.walls if w.prism == pr.index}
        # straight part of edge k: from corner k's last vertex to corner k+1's first vertex
        segs = BEVEL_SEGMENTS if params.bevel_m > 0 else 0
        corner_pos, corner_nrm, corner_idx = [], [], []
        for k, wall in walls.items():
            if params.bevel_m > 0:
                i0 = k * (segs + 1) + segs
                i1 = ((k + 1) % len(pr.polygon)) * (segs + 1)
            else:
                i0, i1 = k, (k + 1) % n
            pa, pb = ring[i0], ring[i1]
            pos = np.array([[*pa, pr.z0], [*pb, pr.z0], [*pb, pr.z1], [*pa, pr.z1]])
            nrm = np.repeat(np.r_[wall.normal, 0.0][None], 4, axis=0)
            ua = float((pa - wall.a) @ wall.tangent) / max(wall.length, 1e-9)
            ub = float((pb - wall.a) @ wall.tangent) / max(wall.length, 1e-9)
            uvs = np.array([[ua, 1.0], [ub, 1.0], [ub, 0.0], [ua, 0.0]])
            fac = observed.get(wall.index)
            prov = _provenance(wall.outline_source, wall.height_source, PROV_PHOTO if fac else PROV_DEFAULT)
            extras = {"kind": "wall", "wall": wall.index, "prism": pr.index, "observed": bool(fac),
                      "locked": not fac, "provenance": prov}
            if fac is not None:
                parts.append(Part(f"wall-{wall.index}", pos, nrm, np.array([[0, 1, 2], [0, 2, 3]]), fac.base_rgb,
                                  fac.texture, uvs, extras))
            else:
                parts.append(Part(f"wall-{wall.index}", pos, nrm, np.array([[0, 1, 2], [0, 2, 3]]), CLAY_RGB,
                                  None, None, extras))
        if params.bevel_m > 0:
            for k in range(len(pr.polygon)):
                base_i = k * (segs + 1)
                for s in range(segs):
                    pa, pb = ring[base_i + s], ring[base_i + s + 1]
                    t = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
                    nn = np.r_[t[1], -t[0], 0.0]
                    b = len(corner_pos)
                    corner_pos += [[*pa, pr.z0], [*pb, pr.z0], [*pb, pr.z1], [*pa, pr.z1]]
                    corner_nrm += [nn] * 4
                    corner_idx += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
            if corner_pos:
                left_obs = [walls[k].index in observed and walls[(k - 1) % len(pr.polygon)].index in observed
                            for k in range(len(pr.polygon))]
                parts.append(Part(f"corners-{pr.index}", np.array(corner_pos), np.array(corner_nrm), np.array(corner_idx),
                                  CLAY_RGB, None, None,
                                  {"kind": "corners", "prism": pr.index, "observed": False, "locked": not all(left_obs),
                                   "provenance": _provenance(pr.outline_source, pr.height_source, PROV_DEFAULT)}))
        # caps
        if pr.cap == "pyramid":
            groups: dict[Any, list[int]] = {}
            for k in range(len(pr.polygon)):
                groups.setdefault(roof_rgb.get((pr.index, k)), []).append(k)
            for rgb, faces in groups.items():
                pos, nrm, idx = [], [], []
                for k in faces:
                    # fan over the bevelled ring vertices under this face (edge k plus half of each corner)
                    sub = _face_ring_indices(k, len(pr.polygon), segs, n, params.bevel_m > 0)
                    for a_i, b_i in zip(sub[:-1], sub[1:]):
                        a3, b3 = np.r_[ring[a_i], pr.z1], np.r_[ring[b_i], pr.z1]
                        nn = np.cross(b3 - a3, pr.apex - a3)
                        nn /= max(np.linalg.norm(nn), 1e-9)
                        bse = len(pos)
                        pos += [a3, b3, pr.apex]
                        nrm += [nn] * 3
                        idx.append([bse, bse + 1, bse + 2])
                observed_face = rgb is not None
                parts.append(Part(f"roof-{pr.index}-{'obs' if observed_face else 'clay'}-{faces[0]}", np.array(pos),
                                  np.array(nrm), np.array(idx), rgb or CLAY_ROOF_RGB, None, None,
                                  {"kind": "roof", "prism": pr.index, "faces": faces, "observed": observed_face,
                                   "locked": not observed_face,
                                   "provenance": _provenance(pr.outline_source, pr.cap_source,
                                                             PROV_PHOTO if observed_face else PROV_DEFAULT)}))
        else:
            rgb = roof_rgb.get((pr.index, 0))
            tris = triangulate_polygon(ring)
            pos = np.array([[*v, pr.z1] for v in ring])
            nrm = np.repeat(np.array([[0.0, 0.0, 1.0]]), n, axis=0)
            parts.append(Part(f"roof-{pr.index}", pos, nrm, np.array(tris), rgb or CLAY_ROOF_RGB, None, None,
                              {"kind": "roof", "prism": pr.index, "faces": [0], "observed": rgb is not None,
                               "locked": rgb is None,
                               "provenance": _provenance(pr.outline_source, pr.cap_source if pr.cap_source != PROV_DEFAULT
                                                         else pr.height_source,
                                                         PROV_PHOTO if rgb is not None else PROV_DEFAULT)}))
    return parts


def _face_ring_indices(k: int, nk: int, segs: int, n: int, bevel: bool) -> list[int]:
    """Bevelled-ring vertex indices from corner k's arc midpoint along edge k to corner k+1's arc
    midpoint: the outline under pyramid face k (corner arcs are split between the two faces)."""
    if not bevel:
        return [k, (k + 1) % n]
    per = segs + 1
    half = segs // 2
    start = k * per + half
    end = ((k + 1) % nk) * per + half
    out = [start]
    i = start
    while i != end:
        i = (i + 1) % n
        out.append(i)
    return out


def check_honesty(model_parts: list[Part], facades: dict[int, StylizedFacade]) -> dict[str, Any]:
    """QA: no detail outside a facade's observation mask, none on locked faces, provenance everywhere."""
    outside = 0
    for fac in facades.values():
        off = ~fac.mask
        if off.any():
            outside += int((np.abs(fac.texture[off].astype(int) - np.array(fac.base_rgb)).sum(axis=1) > 0).sum())
    locked_detail = sum(1 for p in model_parts if p.extras.get("locked") and p.texture is not None)
    locked_photo = sum(1 for p in model_parts if p.extras.get("locked")
                       and p.extras.get("provenance", {}).get("appearance") == PROV_PHOTO)
    missing = sum(1 for p in model_parts
                  if not all(p.extras.get("provenance", {}).get(k) in PROVENANCES for k in PROVENANCE_KEYS))
    allowed = allowed_texture_colors()
    off_palette = 0
    for fac in facades.values():
        cols = {tuple(int(v) for v in c) for c in np.unique(fac.texture.reshape(-1, 3), axis=0)}
        off_palette += len(cols - allowed)
    return {"detail_pixels_outside_mask": outside, "textured_locked_faces": locked_detail,
            "locked_faces_claiming_photos": locked_photo, "faces_without_provenance": missing,
            "off_palette_colors": off_palette,
            "passed": outside == 0 and locked_detail == 0 and locked_photo == 0 and missing == 0 and off_palette == 0}


def stylize_landmark(scene: SceneInput, params: StyleParams = DEFAULT_PARAMS) -> StylizedModel:
    """Footprint + posed photos + SfM points -> stylized GLB and summary (see the module docstring)."""
    timings: dict[str, float] = {}
    t = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal t
        now = time.perf_counter()
        timings[stage] = round(now - t, 3)
        t = now

    points, cameras = scene.points, list(scene.cameras)
    snap = None
    if scene.snap and points is not None and len(points):
        snap = snap_to_footprint(points, scene.footprint, params)
        if snap.accepted:
            points = apply_snap_points(snap, points)
            cameras = [c.snapped(snap) for c in cameras]
    lap("snap")
    massing = build_massing(scene.footprint, points, scene.facts, params)
    lap("massing")
    observations = observe_walls(massing, cameras, points, params)
    lap("observe")
    orthos: dict[int, Orthophoto] = {}
    facades: dict[int, StylizedFacade] = {}
    cache: dict[int, np.ndarray] = {}
    for ob in observations:
        if not ob.observed:
            continue
        wall = massing.walls[ob.wall]
        ortho = facade_orthophoto(wall, ob, massing, cameras, params, cache)
        orthos[wall.index] = ortho
        if ortho.observed_fraction < params.min_observed_fraction:
            ob.observed, ob.reason = False, f"observation mask covers {ortho.observed_fraction:.0%}"
    lap("orthophotos")
    for wi, ortho in orthos.items():
        if observations[wi].observed:
            facades[wi] = stylize_facade(ortho, params, ground_row=massing.walls[wi].z0 <= 0.5)
    lap("stylize")
    roof_rgb = roof_face_colors(massing, points, scene.colors)
    parts = build_parts(massing, facades, roof_rgb, params)
    honesty = check_honesty(parts, facades)
    extras = {"generator": "crowd3d.stylize", "frame": "ENU (glTF y-up: x=east, y=up, -z=north)",
              "landmark": scene.name, "attribution": list(scene.attribution),
              "provenance_values": list(PROVENANCES), "locked_label": "待解锁"}
    glb = write_glb(parts, extras)
    lap("mesh_glb")

    total_area = sum(w.area for w in massing.walls)
    obs_area = sum(massing.walls[i].area for i in facades)
    mask_area = sum(massing.walls[i].area * facades[i].mask.mean() for i in facades)
    tri = int(sum(len(p.indices) for p in parts))
    summary = {
        "landmark": scene.name,
        "inputs": {"cameras": len(cameras), "points": 0 if points is None else int(len(points)),
                   "footprint_vertices": int(len(scene.footprint))},
        "snap": snap.to_meta() if snap else None,
        "massing": massing.report,
        "walls": [{"index": w.index, "prism": w.prism, "length_m": round(w.length, 2),
                   "z0": round(w.z0, 2), "z1": round(w.z1, 2), **observations[w.index].to_meta(),
                   "mask_fraction": round(orthos[w.index].observed_fraction, 3) if w.index in orthos else 0.0,
                   "facade": facades[w.index].report if w.index in facades else None}
                  for w in massing.walls],
        "observed_walls": len(facades),
        "num_walls": len(massing.walls),
        "observed_wall_area_fraction": round(obs_area / max(total_area, 1e-9), 3),
        "observed_pixel_area_fraction": round(mask_area / max(total_area, 1e-9), 3),
        "roof_faces_observed": sum(v is not None for v in roof_rgb.values()),
        "roof_faces": len(roof_rgb),
        "triangles": tri,
        "vertices": int(sum(len(p.positions) for p in parts)),
        "nodes": len(parts),
        "textures": len(facades),
        "glb_bytes": len(glb),
        "qa": honesty,
        "timings_s": timings,
        "total_s": round(sum(timings.values()), 3),
    }
    return StylizedModel(glb, summary, massing, observations, orthos, facades, parts)
