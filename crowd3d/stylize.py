"""Stylized ("toy" / anime) landmark models from facts plus observed photos, CPU only.

The pivot (docs: small budget, no GPU photoreal rendering): instead of a dense or splatted model,
the published model is a simplified, cheerful building in the spirit of flat-colour illustration:

1. Massing from facts. The OSM footprint (ODbL) is simplified (Douglas-Peucker, near-orthogonal
   edges snapped where the outline is rectilinear, true polygons such as an octagon kept) and
   extruded. Heights come from the SfM points standing on the footprint; OSM height / levels or a
   default only when the points say nothing. The roof is a flat cap, or a pyramid when points
   inside the footprint show a rise.
2. Observation. A wall is "observed" when posed photos look at it from the front within a sane
   angle and SfM points lie on it. Only observed walls get facade detail; the rest stay plain clay
   and are flagged locked ("待解锁"). Nothing is invented for places nobody photographed.
3. Facades. Each observed wall gets a rectified orthophoto: every visible pixel of every posed
   photo is projected onto the wall plane (only cameras in front, within the angle limit, not
   hidden behind other walls), exposure-matched, and median-blended (which also removes passers-by).
   It is then stylized: edge-preserving smoothing, k-means to <= 6 colours snapped to a cheerful
   palette, specks removed, regions redrawn as simplified polygons, near-rectangular dark regions
   regularized into window / door shapes. An observation mask is kept; outside it the facade is the
   plain wall colour, and check_facade_honesty() verifies that.
4. Output: a GLB (minimal glTF 2.0 writer, no dependency) with per-face provenance in node extras,
   and a JSON-able summary.

Coordinates are the landmark ENU frame (x = east, y = north, z = up, metres). The GLB is written
in glTF's y-up convention: ENU (x, y, z) -> glTF (x, z, -y), the same mapping as web/viewer.js.

Generative models are not used here. The places where an allowed one (simplification, segmentation
or stylization of observed content only) would help are marked "GENERATIVE-OK" below.
"""

from __future__ import annotations

import io
import json
import math
import struct
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# ---------------------------------------------------------------- provenance and palette

PROV_PHOTO = "photo-observed"        # pixels / colours that come from posed photos of this face
PROV_OSM = "osm-footprint"           # outline from OpenStreetMap (ODbL)
PROV_SFM = "sfm-points"              # height or roof shape measured from the reconstruction's points
PROV_OSM_HEIGHT = "osm-height"       # OSM height=* or building:levels=* tag
PROV_WIKIDATA = "wikidata-height"    # Wikidata P2048 (height)
PROV_DEFAULT = "default-style"       # neither measured nor tagged: a neutral style default
PROVENANCES = (PROV_PHOTO, PROV_OSM, PROV_SFM, PROV_OSM_HEIGHT, PROV_WIKIDATA, PROV_DEFAULT)

# Crowd3D's own cheerful palette (not any brand's): material families in 2-4 values each, so that
# snapping keeps both the hue family and the light/dark structure of a facade.
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
CLAY_RGB = (233, 226, 214)           # locked (unobserved) faces: plain light clay
ROOF_DEFAULT_RGB = (205, 196, 184)   # roof nobody saw well enough to colour
INK_RGB = (45, 42, 50)               # outline colour (viewer and window frames)


# ---------------------------------------------------------------- parameters


@dataclass(frozen=True)
class StyleParams:
    simplify_tol_m: float = 0.8          # Douglas-Peucker tolerance on the OSM outline
    min_edge_m: float = 1.5              # shorter edges are merged into their neighbours
    ortho_tol_deg: float = 12.0          # edges within this of the dominant axes snap to them ...
    ortho_min_fraction: float = 0.6      # ... when at least this share of the outline length is near them
    bevel_m: float = 0.35                # corner chamfer of the "toy" look (0 disables)
    wall_band_m: float = 1.5             # points within this horizontal distance of a wall belong to it
    height_quantile: float = 0.95        # robust top of a point set
    min_height_points: int = 30
    default_height_m: float = 15.0       # ARCHITECTURE 3.4.1: no tag, no points -> 15 m
    level_height_m: float = 3.0          # building:levels -> metres
    roof_min_rise_m: float = 1.5         # interior points this far above the walls make a pyramid roof
    px_per_m: float = 20.0               # facade texture resolution ...
    max_texture_px: int = 512            # ... capped on the longer side
    max_view_angle_deg: float = 72.0     # camera ray vs wall normal
    min_camera_distance_m: float = 2.0   # cameras closer to the wall plane are skipped
    min_views: int = 2                   # a facade pixel is observed when this many photos see it
    max_views_per_wall: int = 24         # best-scoring cameras used per wall
    min_wall_cameras: int = 2            # a wall is observed when this many cameras see a good part of it ...
    min_wall_visible_fraction: float = 0.15   # ... (fraction of the wall inside each of those images)
    min_wall_points: int = 20            # ... and this many SfM points lie on it (when points are given)
    min_observed_fraction: float = 0.10  # walls whose observation mask covers less stay locked
    max_colors: int = 6
    min_region_m2: float = 0.12          # smaller colour specks merge into their surroundings
    region_simplify_px: float = 1.5      # polygon simplification of colour regions
    window_min_m: tuple[float, float] = (0.35, 0.5)   # (width, height) range of regularized openings
    window_max_m: tuple[float, float] = (5.0, 9.0)
    window_min_fill: float = 0.55        # region area / bounding box area
    snap_max_rotation_deg: float = 20.0  # footprint snapping limits
    snap_max_shift_m: float = 15.0


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
        if n < 1e-12:
            d = np.linalg.norm(seg, axis=1)
        else:
            d = np.abs(ab[0] * seg[:, 1] - ab[1] * seg[:, 0]) / n
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
    i = 0
    j = int(np.linalg.norm(p - p[i], axis=1).argmax())
    i = int(np.linalg.norm(p - p[j], axis=1).argmax())
    i, j = min(i, j), max(i, j)
    first = _dp_open(p[i:j + 1], tol)
    second_pts = np.vstack([p[j:], p[:i + 1]])
    second = _dp_open(second_pts, tol)
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
    changed = True
    while changed and len(p) > 3:
        changed = False
        lengths = [np.linalg.norm(p[(k + 1) % len(p)] - p[k]) for k in range(len(p))]
        k = int(np.argmin(lengths))
        if lengths[k] < min_edge:
            # merge the short edge's two vertices into the one whose removal bends the outline least
            a, b = p[k], p[(k + 1) % len(p)]
            prev, nxt = p[k - 1], p[(k + 2) % len(p)]
            da = abs(_cross2(b - prev, a - prev)) / max(np.linalg.norm(b - prev), 1e-9)
            db = abs(_cross2(nxt - a, b - a)) / max(np.linalg.norm(nxt - a), 1e-9)
            del p[k if da <= db else (k + 1) % len(p)]
            changed = True
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

    Edges off the axes (an octagon's diagonals, an apse) are kept as they are. Returns the polygon and
    whether snapping was applied.
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
        prev = lines[k - 1]
        cur = lines[k]
        x = _line_intersection(prev[0], prev[1], cur[0], cur[1])
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
        s = abs(_cross2(u, v)) / (nu * nv)
        if s > math.sin(math.radians(tol_deg)) or np.dot(u, v) < 0:
            keep.append(k)
    return p[keep] if len(keep) >= 3 else p


def simplify_footprint(poly: np.ndarray, params: StyleParams = DEFAULT_PARAMS) -> tuple[np.ndarray, dict[str, Any]]:
    """OSM outline -> clean toy outline. Returns the ring (CCW) and a report."""
    p0 = ensure_ccw(poly)
    p = simplify_polygon(p0, params.simplify_tol_m)
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
    x, y = np.asarray(xy, float)[:, 0], np.asarray(xy, float)[:, 1]
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
    """Ear clipping for a simple CCW polygon (n < a few hundred). Triangles are CCW (upward normal)."""
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
            if (_cross2(b - a, q - a) >= 0 and _cross2(c - b, q - b) >= 0 and _cross2(a - c, q - c) >= 0):
                return False
        return True

    guard = 0
    while len(idx) > 3 and guard < 10 * len(p) ** 2:
        guard += 1
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


def bevel_polygon(poly: np.ndarray, radius: float, segments: int = 2) -> tuple[np.ndarray, list[int]]:
    """Round every corner with a short arc (the rounded toy look). Returns the new ring and, for each
    new vertex, the index of the original edge it lies on (-1 for arc vertices between edges)."""
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
        # quadratic Bezier from a to b with control point at the corner: a cheap rounded corner
        for s in range(segments + 1):
            t = s / segments
            pt = (1 - t) ** 2 * a + 2 * (1 - t) * t * cur + t ** 2 * b
            out.append(pt)
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

    def matrix(self) -> np.ndarray:
        c, s = math.cos(math.radians(self.rotation_deg)), math.sin(math.radians(self.rotation_deg))
        m = np.eye(4)
        m[:2, :2] = [[c, -s], [s, c]]
        m[:2, 3] = self.shift
        return m

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


def snap_to_footprint(points: np.ndarray, footprint: np.ndarray, params: StyleParams = DEFAULT_PARAMS,
                      z_band: tuple[float, float] = (2.0, 20.0), max_points: int = 6000,
                      seed: int = 0) -> FootprintSnap:
    """ARCHITECTURE 3.5.7: trimmed 2-D rigid ICP of wall-height points onto the footprint outline.

    Starts from no shift and from the shift that puts the wall-band points' centre on the footprint
    centroid (GPS-registered models can be 10 m off), each with a few start rotations; keeps the
    closest fit. Rejected (identity) when it needs more than snap_max_rotation_deg / snap_max_shift_m.
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
    before = float(np.median(distance_to_outline(xy, fp)[0]))
    starts = [np.zeros(2), centroid - np.median(xy, axis=0)]
    best = None
    for shift0 in starts:
        for rot0 in (-10.0, 0.0, 10.0):
            c, s = math.cos(math.radians(rot0)), math.sin(math.radians(rot0))
            r = np.array([[c, -s], [s, c]])
            t = shift0 + centroid - r @ centroid
            for _ in range(80):
                moved = xy @ r.T + t
                d, target, _ = distance_to_outline(moved, fp)
                keep = d <= np.quantile(d, 0.7)
                dr, dt = _rigid_2d(moved[keep], target[keep])
                r, t = dr @ r, dr @ t + dt
                if np.linalg.norm(dt) < 1e-4 and abs(math.atan2(dr[1, 0], dr[0, 0])) < 1e-6:
                    break
            after = float(np.median(distance_to_outline(xy @ r.T + t, fp)[0]))
            if best is None or after < best[0] - 1e-6:
                best = (after, r, t)
    after, r, t = best
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
    m = snap.matrix()
    out = np.asarray(points, float).copy()
    out[:, :2] = out[:, :2] @ m[:2, :2].T + m[:2, 3]
    return out


# ---------------------------------------------------------------- massing


@dataclass
class Prism:
    """One extruded block: a CCW outline from z0 to z1 plus a cap (flat or pyramid)."""

    level: int
    polygon: np.ndarray                 # (k, 2) CCW
    z0: float
    z1: float
    outline_source: str                 # PROV_OSM (level 0) or PROV_SFM (upper levels from the point heights)
    height_source: str                  # PROV_SFM / PROV_OSM_HEIGHT / PROV_WIKIDATA / PROV_DEFAULT
    cap: str = "flat"                   # "flat" | "pyramid"
    apex: np.ndarray | None = None      # (3,) for a pyramid
    cap_source: str = PROV_DEFAULT
    roof_rgb: tuple[int, int, int] = ROOF_DEFAULT_RGB
    roof_color_source: str = PROV_DEFAULT
    measured_fraction: float = 0.0      # share of this block's cells whose height came from points

    def to_meta(self) -> dict[str, Any]:
        return {"level": self.level, "vertices": int(len(self.polygon)), "z0": round(self.z0, 2),
                "z1": round(self.z1, 2), "outline_source": self.outline_source, "height_source": self.height_source,
                "cap": self.cap, "apex_z": None if self.apex is None else round(float(self.apex[2]), 2),
                "cap_source": self.cap_source, "roof_color_source": self.roof_color_source,
                "measured_fraction": round(self.measured_fraction, 3)}


@dataclass
class Wall:
    index: int
    level: int
    a: np.ndarray                       # (2,) start, CCW order along the block outline
    b: np.ndarray
    z0: float
    z1: float
    outline_source: str
    height_source: str

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

    def grid(self, w: int, h: int) -> np.ndarray:
        """(h, w, 3) ENU sample centres; row 0 is the top of the wall, column 0 its start a."""
        u = (np.arange(w) + 0.5) / w * self.length
        z = self.z1 - (np.arange(h) + 0.5) / h * (self.z1 - self.z0)
        xy = self.a[None, :] + u[:, None] * self.tangent[None, :]
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


@dataclass
class HeightFacts:
    """Heights the product knows without photos (all optional)."""

    osm_height_m: float | None = None
    osm_levels: float | None = None
    wikidata_height_m: float | None = None


def _robust_top(z: np.ndarray, q: float) -> float:
    return float(np.quantile(z, q))


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
    cx = lo[0] + (np.arange(nx) + 0.5) * cell
    cy = lo[1] + (np.arange(ny) + 0.5) * cell
    gx, gy = np.meshgrid(cx, cy)
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
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out = np.nanmedian(np.stack(stack), axis=0)
    out[np.isnan(a)] = np.nan
    return out


def find_levels(heights: np.ndarray, cell_area: float, min_area_m2: float, min_sep_m: float,
                min_height_m: float = 4.0, max_levels: int = 4) -> list[float]:
    """Distinct block heights in a height field: area-weighted 1-D clustering, coarse to fine.

    A level needs min_area_m2 of cells whose height is at least that level (minus half the
    separation) - those cells can carry a block of that height. Levels closer than min_sep_m merge.
    """
    h = heights[np.isfinite(heights)]
    h = h[h >= min_height_m]
    if len(h) == 0:
        return []
    need = max(1, int(math.ceil(min_area_m2 / cell_area)))
    hs = np.sort(h)
    candidates = []
    # robust "plateau" heights: medians of height bands one separation wide
    for top in np.arange(hs[0], hs[-1] + min_sep_m, min_sep_m / 2):
        band = hs[(hs >= top - min_sep_m / 2) & (hs < top + min_sep_m / 2)]
        if len(band) >= need:
            candidates.append((len(band), float(np.median(band))))
    levels: list[float] = []
    for _, v in sorted(candidates, reverse=True):
        if all(abs(v - lv) >= min_sep_m for lv in levels) and (hs >= v - min_sep_m / 2).sum() >= need:
            levels.append(v)
        if len(levels) >= max_levels:
            break
    return sorted(levels)


def _contour_polygons(mask: np.ndarray, cell: float, origin: tuple[float, float], min_area_m2: float) -> list[np.ndarray]:
    """External contours of a boolean grid mask (row 0 = south) as ENU polygons, upsampled for smoothness."""
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


def build_massing(footprint: np.ndarray, points: np.ndarray | None, facts: HeightFacts | None = None,
                  params: StyleParams = DEFAULT_PARAMS, cell: float = 0.75,
                  colors: np.ndarray | None = None) -> Massing:
    """Stacked blocks ("wedding cake") from the footprint and the SfM points' height field.

    Level 0 is the simplified OSM outline at the lowest well-supported height; each higher level is the
    region whose points stand at least that high, extruded from the level below. Blocks whose inside
    rises well above their walls get a pyramid cap (near-convex outlines only). Without usable points
    the footprint is extruded to the OSM / Wikidata height or the style default.
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

    levels: list[float] = []
    heights = counts = inside = None
    origin = (0.0, 0.0)
    if points is not None and len(points):
        heights, counts, inside, origin = height_field(points, simple, cell)
        smooth = _nan_median_filter(heights, 1)
        area = abs(signed_area(simple))
        levels = find_levels(smooth, cell * cell, min_area_m2=max(12.0, 0.02 * area), min_sep_m=4.0)
        heights = smooth
    report["levels_m"] = [round(v, 2) for v in levels]
    measured = np.isfinite(heights) & inside if heights is not None else None
    report["measured_cell_fraction"] = (round(float(measured.sum() / max(inside.sum(), 1)), 3)
                                        if measured is not None else 0.0)

    prisms: list[Prism] = []
    if not levels:
        prisms.append(Prism(0, simple, 0.0, fallback_h, PROV_OSM, fallback_src))
    else:
        # level of each measured cell: the highest level it reaches (with half a separation of slack)
        lvl = np.full(heights.shape, -1, dtype=int)
        for k, v in enumerate(levels):
            lvl[measured & (heights >= v - 2.0)] = k
        base = 0
        prisms.append(Prism(0, simple, 0.0, levels[0], PROV_OSM, PROV_SFM,
                            measured_fraction=float(((lvl >= 0) & inside).sum() / max(inside.sum(), 1))))
        for k in range(base + 1, len(levels)):
            mask = (lvl >= k) & inside
            for poly in _contour_polygons(mask, cell, origin, min_area_m2=max(8.0, 0.01 * abs(signed_area(simple)))):
                poly, _ = simplify_footprint(poly, params)
                if len(poly) < 3 or abs(signed_area(poly)) < 4.0:
                    continue
                prisms.append(Prism(k, poly, levels[k - 1], levels[k], PROV_SFM, PROV_SFM, measured_fraction=1.0))
        # caps: a block whose own cells rise well above its top becomes a pyramid (near-convex outlines)
        for pr in prisms:
            _fit_cap(pr, prisms, heights, inside, origin, cell, params)
    if points is not None and colors is not None and len(points):
        for pr in prisms:
            _roof_colour(pr, points, colors)
    walls: list[Wall] = []
    for pr in prisms:
        poly = pr.polygon
        for k in range(len(poly)):
            walls.append(Wall(len(walls), pr.level, poly[k].copy(), poly[(k + 1) % len(poly)].copy(),
                              pr.z0, pr.z1, pr.outline_source, pr.height_source))
    report["prisms"] = [p.to_meta() for p in prisms]
    report["num_walls"] = len(walls)
    report["fallback_height"] = {"m": fallback_h, "source": fallback_src}
    report["seconds"] = round(time.perf_counter() - t0, 3)
    return Massing(prisms, walls, report)


def _cells_of(poly: np.ndarray, shape: tuple[int, int], origin: tuple[float, float], cell: float) -> np.ndarray:
    ny, nx = shape
    gx, gy = np.meshgrid(origin[0] + (np.arange(nx) + 0.5) * cell, origin[1] + (np.arange(ny) + 0.5) * cell)
    return points_in_polygon(np.stack([gx.ravel(), gy.ravel()], 1), poly).reshape(ny, nx)


def _fit_cap(pr: Prism, prisms: list[Prism], heights: np.ndarray, inside: np.ndarray,
             origin: tuple[float, float], cell: float, params: StyleParams) -> None:
    hull = cv2.convexHull(pr.polygon.astype(np.float32))[:, 0, :]
    convexity = abs(signed_area(pr.polygon)) / max(abs(signed_area(hull)), 1e-9)
    mine = _cells_of(pr.polygon, heights.shape, origin, cell) & inside
    for other in prisms:                       # cells covered by a higher block are not this block's roof
        if other.level > pr.level:
            mine &= ~_cells_of(other.polygon, heights.shape, origin, cell)
    h = heights[mine & np.isfinite(heights)]
    if len(h) < 10:
        return
    rise = float(np.quantile(h, 0.9)) - pr.z1
    if convexity < 0.85 or rise < params.roof_min_rise_m:
        if len(h) >= 10:
            pr.cap_source = PROV_SFM
        return
    # apex over the highest cells (top structure), clamped into the outline's interior
    ys, xs = np.nonzero(mine & np.isfinite(heights) & (heights >= np.quantile(h, 0.9)))
    apex_xy = np.array([origin[0] + (xs.mean() + 0.5) * cell, origin[1] + (ys.mean() + 0.5) * cell])
    centroid = polygon_centroid(pr.polygon)
    if not points_in_polygon(apex_xy[None], pr.polygon)[0]:
        apex_xy = centroid
    apex_xy = 0.5 * apex_xy + 0.5 * centroid   # keep the pyramid tidy (toy look)
    pr.cap, pr.apex, pr.cap_source = "pyramid", np.array([apex_xy[0], apex_xy[1], pr.z1 + rise]), PROV_SFM


def _roof_colour(pr: Prism, points: np.ndarray, colors: np.ndarray) -> None:
    """Median colour of the points on this block's top (above 85 % of its height), snapped to the palette."""
    inside = points_in_polygon(points[:, :2], pr.polygon)
    top = inside & (points[:, 2] >= pr.z0 + 0.85 * (pr.z1 - pr.z0))
    if top.sum() < 30:
        return
    rgb = np.median(colors[top].astype(float), axis=0)
    idx = snap_to_palette(rgb[None].astype(np.uint8))[0]
    pr.roof_rgb = tuple(int(v) for v in PALETTE_RGB[idx])
    pr.roof_color_source = PROV_PHOTO
