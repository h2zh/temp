"""Stylized ("toy" / anime) landmark models from facts plus observed photos, CPU only.

The pivot (small budget, no GPU photoreal rendering): instead of a dense or splatted model, the
published model is a simplified, cheerful building in the spirit of flat-colour illustration.

0. Registration. The model (points + cameras) is snapped onto the OSM footprint (rigid ICP, then a
   bounded similarity: the GPS-derived SfM scale is the weakest link) or reuses a snap stored with an
   earlier model version, and registration_check() verifies it edge by edge (most supported outline
   edges, not all parallel, within reg_edge_limit_m). Unverified: the model is published as clay
   massing only. Without a usable footprint the outline is traced from the points (sfm-points).
1. Massing from facts. The OSM footprint (ODbL), with its courtyards, is simplified (Douglas-Peucker
   with a growing tolerance until it has <= ~32 vertices while staying within 3 % of its diameter of
   the input; near-orthogonal edges snapped where the outline is rectilinear; true polygons such as an
   octagon kept). Edges whose facade stands clearly off it are moved onto the vertical plane fitted to
   the points (inwards, or outwards only within the real outline). The base height comes from the SfM
   points on the walls (OSM levels / height, a tower's Wikidata height or a default only when the
   points say nothing). Roofs are fitted to the points: flat, or a pyramid whose faces without point
   coverage stay a neutral flat cap; blocks standing above the fitted roof (a tower, a lantern, a
   choir) become their own prisms, recursively. Heights carried over to walls nobody measured are
   'inferred'.
2. Observation. A wall is "observed" when posed photos look at it from the front within a sane
   angle, nothing in the massing hides it, SfM points lie on it (>= 1 per square metre) and its fitted
   facade plane is within plane_tol_m / plane_tol_deg of the face (and not outside the outline: that
   is something standing in front of it). The rest stays clay and is flagged locked ("待解锁").
   Nothing is invented for places nobody photographed.
3. Facades. Each observed wall gets a rectified orthophoto sampled on its fitted plane: posed photos
   are projected (cameras in front, within the angle limit, not hidden by the massing nor by SfM
   points standing more than occluder_margin_m in front of the wall), exposure-matched and blended
   with a per-pixel weighted median. A pixel is observed with >= 2 agreeing views (photo consistency)
   and SfM support, minus sky and vegetation. Stylization (on observed pixels only): wall / dark /
   coloured layers, the landmark's few material colours snapped to a cheerful palette with their
   lightness band kept and no white balance on the building itself; only regular structure is drawn
   (windows / doors / arches in rows, bands, pilasters, straight inlay lines, large compact shapes).
   Outside the mask the texture is the clay pattern, and the wall is split into its photographed
   strip and clay strips.
4. Roofs. A cap face is coloured as a whole only when its point coverage is >= roof_full_coverage;
   otherwise its observed patches become decals and the rest stays clay.
5. Output: a GLB (minimal glTF 2.0 writer, no dependency) where every face group is its own node with
   provenance {outline, height, appearance} in its extras (plus coverage, area, contributing photos),
   facade textures packed into shared atlases, the registration and snap in the scene extras, plus a
   JSON-able summary. QA: check_honesty() (what the pipeline enforced) and audit_model() (independent
   recomputation from points and cameras of coverage, planes, heights, occlusion and registration).
   web/stylized.html renders it with toon shading and outlines.

Coordinates are the landmark ENU frame (x = east, y = north, z = up, metres). The GLB uses glTF's
y-up convention: ENU (x, y, z) -> glTF (x, z, -y), the same mapping as web/viewer.js.

No generative model is used. Steps where an allowed one (simplification, segmentation or
stylization of observed content only, never completion) would help are marked "GENERATIVE-OK".
"""

# ruff: noqa: RUF046  (int(round(x)) on numpy scalars is deliberate: OpenCV wants Python ints)
from __future__ import annotations

import io
import itertools
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
PROV_INFERRED = "inferred"           # not measured on this face: carried over from measured parts of its block
PROV_DEFAULT = "default-style"       # neither measured nor tagged: a neutral style default
PROVENANCES = (PROV_PHOTO, PROV_OSM, PROV_SFM, PROV_OSM_HEIGHT, PROV_WIKIDATA, PROV_INFERRED, PROV_DEFAULT)
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
    ("pebble", (160, 156, 150)),
    ("stone-dark", (135, 126, 116)),
    ("graphite", (98, 98, 102)),
    ("charcoal", (60, 60, 66)),
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
    ("slate-light", (150, 162, 184)),
    ("slate", (96, 108, 128)),
    ("night", (52, 60, 80)),
    ("wood", (176, 118, 66)),
    ("cocoa", (112, 72, 44)),
    ("gold", (214, 172, 74)),
    ("copper", (110, 196, 176)),
    ("teal", (50, 140, 130)),
)
PALETTE_RGB = np.array([c for _, c in PALETTE], dtype=np.uint8)
PALETTE_NAMES = tuple(n for n, _ in PALETTE)
# near-grey materials snap along this ramp by lightness only (no blue cast for grey stone)
NEUTRAL_RAMP = ("snow", "stone-light", "stone", "pebble", "stone-dark", "graphite", "charcoal")
CLAY_RGB = (228, 232, 238)           # locked (unobserved) faces: cool, light "paper model" clay (photographed
                                     # parts are warm material colours, so the two never read alike)
CLAY_HATCH_RGB = (208, 215, 226)     # its hatch stripes, baked into the unobserved part of a facade texture
CLAY_ROOF_RGB = (214, 220, 229)      # roof faces nobody saw
INK_RGB = (45, 42, 50)               # window frames and outlines
SHADE_FACTOR = 0.62                  # region outlines are the region colour times this
BEVEL_SEGMENTS = 3                   # arc segments per rounded corner
CLAY_PATTERN = (CLAY_RGB, CLAY_HATCH_RGB)


def shade(rgb) -> tuple[int, int, int]:
    return tuple(int(round(v * SHADE_FACTOR)) for v in rgb)


def allowed_texture_colors() -> set[tuple[int, int, int]]:
    """Every colour a stylized facade texture may contain: palette, palette shades, ink, clay pattern."""
    out = {tuple(int(v) for v in c) for c in PALETTE_RGB}
    out |= {shade(c) for c in PALETTE_RGB}
    out.add(INK_RGB)
    out |= set(CLAY_PATTERN)
    return out


class StylizeError(ValueError):
    """The inputs cannot give an honest model (e.g. no usable footprint and no points)."""


# ---------------------------------------------------------------- parameters


@dataclass(frozen=True)
class StyleParams:
    # footprint
    simplify_tol_m: float = 0.8          # Douglas-Peucker tolerance on the OSM outline (grown up to x4 ...
    max_footprint_vertices: int = 32     # ... until the outline has at most this many vertices ...
    max_hausdorff_frac: float = 0.03     # ... while staying within this share of the diameter of the input)
    max_block_vertices: int = 12         # blocks found in the points are simplified harder
    min_edge_m: float = 1.5              # shorter edges are merged into their neighbours
    ortho_tol_deg: float = 12.0          # edges within this of the dominant axes snap to them ...
    ortho_min_fraction: float = 0.6      # ... when at least this share of the outline length is near them
    bevel_m: float = 0.3                 # rounded corners of the toy look (0 disables): at least this ...
    bevel_frac: float = 0.015            # ... or this share of the block's diameter ...
    bevel_max_m: float = 1.2             # ... at most this
    snap_max_rotation_deg: float = 20.0  # model -> footprint snapping limits (ARCHITECTURE 3.5.7 uses 10 deg / 8 m
    snap_max_shift_m: float = 20.0       # for live models; GT-ENU of the real-photo scenes is 7-15 m off) ...
    snap_max_residual_m: float = 1.5     # ... and the snapped wall points must sit this close (median) to the outline
    snap_max_scale: float = 1.15         # GPS-derived SfM scale may be corrected by up to this factor (1: rigid only)
    # registration gate (no photo content at all unless the points sit on the outline edge by edge)
    reg_edge_limit_m: float = 3.0        # an edge is verified when its points' median distance is under this ...
    reg_support_frac: float = 0.25       # ... among edges carrying >= this share of the median edge's point count ...
    reg_min_share: float = 0.6           # ... and this share of those edges must be verified
    # facade planes (offset + yaw per wall, fitted to the points)
    plane_search_m: float = 4.0          # points this close to a face (horizontally) are candidates
    plane_inlier_m: float = 0.3          # RANSAC inlier distance
    plane_min_inliers: int = 25
    plane_min_inlier_share: float = 0.25 # ... and this share of the points near the face (else no single surface is there)
    plane_max_yaw_deg: float = 15.0      # candidate planes at most this far from the face's direction
    plane_tol_m: float = 1.5             # a facade is textured only when its surface is this close to the face ...
    plane_tol_deg: float = 8.0           # ... and this parallel to it
    plane_move_m: float = 1.0            # faces whose surface lies further inside the outline than this (or ...
    plane_move_deg: float = 5.0          # ... turned more than this) are moved onto the fitted plane ...
    plane_max_move_m: float = 5.0        # ... by at most this much
    # occlusion by things standing in front of a wall (SfM points) and photo-consistency
    occluder_margin_m: float = 1.5       # a point this far in front of the fitted wall plane occludes it ...
    occluder_grid: int = 128             # ... in a per-camera depth buffer with this many cells along the longer image side
    occluder_cell_px: float = 8.0        #     (cells at least this many pixels wide) ...
    occluder_min_count: int = 3          # ... when its cell holds at least this many such points (SfM depth noise is sparse) ...
    occluder_min_track: int = 3          # ... counting only points seen by this many photos ...
    occluder_min_angle_deg: float = 5.0  # ... under this triangulation angle (when the point quality is known)
    audit_occluder_margin_m: float = 1.5 # the audit's own occluder margin (independent of the one that built the masks)
    max_occluders: int = 80000           # occluder candidates per wall (random subsample beyond)
    consistency_rgb: float = 60.0        # a view agrees with the blend when its colour is this close (RGB distance) ...
    consistency_blur_m: float = 0.4      # ... compared on averages over this window
    min_agreement: float = 0.5           # ... and this share of a pixel's views must agree (at least min_views)
    # massing
    cell_m: float = 0.75                 # height-field resolution
    wall_band_m: float = 1.0             # points within this horizontal distance of an edge are on its wall
    min_edge_points: int = 40            # an edge's top height needs this many wall points
    edge_top_quantile: float = 0.97
    base_quantile: float = 0.25          # base height: from this length-weighted quantile of the edge tops (then the tier median)
    default_height_m: float = 15.0       # ARCHITECTURE 3.4.1: no tag, no points -> 15 m
    level_height_m: float = 3.0          # building:levels -> metres
    tower_ratio: float = 2.5             # a total height (Wikidata P2048) extrudes the outline only when it is this many
                                         # times the outline's diameter (a tower); otherwise it is usually a spire
    roof_min_rise_m: float = 1.5         # a pyramid needs at least this rise
    step_m: float = 3.0                  # cells this far above the fitted roof form a block of their own
    min_block_m2: float = 6.0
    max_depth: int = 3
    # roof observation
    roof_cell_m: float = 1.0             # roof coverage grid
    roof_face_min_coverage: float = 0.4  # roof faces with less point support get a neutral (flat) cap and stay clay
    roof_full_coverage: float = 0.8      # a face this covered is coloured as a whole; less: observed patches only
    roof_patch_min_m2: float = 4.0       # observed roof patches smaller than this are dropped
    strip_min_m: float = 1.0             # unobserved ends of a facade narrower than this are not split off
    # observation and orthophotos
    px_per_m: float = 16.0               # facade texture resolution (flat colours need little) ...
    max_texture_px: int = 512            # ... capped on the longer side
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
    min_point_density: float = 1.0       # ... at least this many per square metre of wall ...
    min_observed_fraction: float = 0.10  # ... and its observation mask covers at least this share
    # stylization
    work_px_per_m: float = 8.0           # segmentation resolution (details under ~0.25 m are dropped)
    local_window_m: float = 3.0          # dark accents: darker than the mean of this window ...
    dark_contrast: float = 6.0           # ... by this much (CIE L*)
    accent_min_chroma: float = 14.0      # coloured accents: at least this chroma (and 8 above the wall's)
    min_accent_m2: float = 0.06          # accent specks under this go
    min_own_accent_m2: float = 0.25      # a compact coloured accent of this size keeps its own colour (a gilded door)
    min_free_shape_m2: float = 0.6       # free-form dark shapes (not openings / bands) under this go (x building scale) ...
    min_free_solidity: float = 0.7       # ... and so do ragged ones (area / convex hull area under this)
    roof_lift: float = 0.5               # roofs (often dark slate) are lifted towards light: cheerful, readable
    wall_lift: float = 0.2               # wall colour lifted this share of the way to white before snapping
    min_dark_delta: float = 30.0         # dark accents at least this much darker (L*) than the wall colour
    own_wall_delta_e: float = 18.0       # a facade whose wall colour differs this much from the landmark's keeps its own
    band_min_fill: float = 0.45          # long thin regions become straight bands / pilasters:
    band_min_len_m: float = 2.0          #   at least this long ...
    band_max_thick_m: float = 1.2        #   ... at most this thick ...
    band_gap_m: float = 1.5              #   ... and pieces of one row closer than this join
    line_window_m: float = 0.6           # thin dark lines (marble inlays, frames, string courses): darker than a
    line_contrast: float = 8.0           #   closing over this window by this much (L*) ...
    line_min_len_m: float = 1.5          #   ... straight for at least this long (horizontal or vertical) ...
    line_row_tol_m: float = 0.15         #   ... segments in one row / column this close join ...
    line_gap_m: float = 0.8              #   ... across gaps up to this
    region_outline: bool = True          # darker outline around colour regions ...
    region_outline_m: float = 0.06       # ... this thick
    opening_frame_m: float = 0.10        # ink frame of windows / doors
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


def _simplify_once(p0: np.ndarray, tol: float, min_edge: float, params: StyleParams) -> tuple[np.ndarray, bool]:
    p = simplify_polygon(p0, tol)
    p = drop_short_edges(p, min_edge)
    p, snapped = orthogonalize(p, params.ortho_tol_deg, params.ortho_min_fraction)
    return remove_collinear(drop_short_edges(p, min_edge)), snapped


def polygon_diameter(poly: np.ndarray) -> float:
    p = np.asarray(poly, float)
    return float(np.max(np.linalg.norm(p[:, None] - p[None], axis=2))) if len(p) > 1 else 0.0


def simplify_footprint(poly: np.ndarray, params: StyleParams = DEFAULT_PARAMS, tol_m: float | None = None,
                       max_vertices: int | None = None, hausdorff_frac: float | None = None
                       ) -> tuple[np.ndarray, dict[str, Any]]:
    """OSM outline -> clean toy outline. Returns the CCW ring and a report.

    The Douglas-Peucker tolerance grows (x1 .. x4) until the ring has at most `max_vertices`
    vertices, as long as the outline stays within max(1 m, max_hausdorff_frac x diameter) of the
    input: a toy needs ~30 walls, not the 200 vertices of a detailed OSM outline. Minimum edge
    length scales with the building too (3 % of its diameter, at least min_edge_m).
    """
    p0 = ensure_ccw(poly)
    budget = params.max_footprint_vertices if max_vertices is None else max_vertices
    diam = polygon_diameter(p0)
    max_haus = max(1.0, (params.max_hausdorff_frac if hausdorff_frac is None else hausdorff_frac) * diam)
    min_edge = max(params.min_edge_m, 0.03 * diam)
    tol0 = params.simplify_tol_m if tol_m is None else tol_m
    best = None
    for f in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        p, snapped = _simplify_once(p0, tol0 * f, min_edge, params)
        haus = _outline_hausdorff(p0, p)
        if best is not None and haus > max_haus:
            break
        best = (p, snapped, haus, tol0 * f)
        if len(p) <= budget:
            break
    p, snapped, haus, tol = best
    report = {
        "input_vertices": len(p0),
        "output_vertices": len(p),
        "tolerance_m": round(tol, 2),
        "min_edge_m": round(min_edge, 2),
        "orthogonal_snap": bool(snapped),
        "dominant_axis_deg": round(dominant_axis_deg(p0), 1),
        "area_input_m2": round(abs(signed_area(p0)), 1),
        "area_output_m2": round(abs(signed_area(p)), 1),
        "hausdorff_m": round(haus, 2),
    }
    return p, report


def regularize_block(poly: np.ndarray, params: StyleParams = DEFAULT_PARAMS) -> np.ndarray:
    """A block outline traced from the height field -> a toy shape: its minimum-area rectangle or its
    convex hull when they fill well (towers, choirs, lanterns), else a hard-simplified polygon."""
    p = ensure_ccw(poly)
    area = abs(signed_area(p))
    rect = cv2.boxPoints(cv2.minAreaRect(p.astype(np.float32))).astype(float)
    if area >= 0.85 * abs(signed_area(rect)):
        return ensure_ccw(rect)
    hull = cv2.convexHull(p.astype(np.float32))[:, 0, :].astype(float)
    if area >= 0.85 * abs(signed_area(hull)):
        p = ensure_ccw(hull)
    out, _ = simplify_footprint(p, params, tol_m=max(params.simplify_tol_m, 0.6),
                                max_vertices=params.max_block_vertices, hausdorff_frac=0.08)
    return out


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
    """Ear clipping for a simple CCW polygon (a few hundred vertices at most). Triangles are CCW.
    Also takes a weakly simple ring with bridged holes (vertices repeated at the bridges)."""
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
        for j in idx:
            if j in (i0, i1, i2):
                continue
            q = p[j]
            if (np.abs(q - a).max() < 1e-9 or np.abs(q - b).max() < 1e-9 or np.abs(q - c).max() < 1e-9):
                continue                              # a bridge duplicate of a corner: not inside
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


def bridge_holes(outer: np.ndarray, holes: list[np.ndarray]) -> np.ndarray:
    """One weakly simple CCW ring from a CCW outer ring and holes (each cut open along a bridge to a
    visible vertex), for ear clipping. Holes are traversed clockwise."""
    ring = [np.asarray(v, float) for v in ensure_ccw(outer)]
    pending = [ensure_ccw(h)[::-1].copy() for h in holes]
    pending.sort(key=lambda h: -float(h[:, 0].max()))
    while pending:
        h = pending.pop(0)
        m = int(np.argmax(h[:, 0]))
        M = h[m]
        R = np.array(ring)
        edges = [(R, np.roll(R, -1, axis=0))] + [(o, np.roll(o, -1, axis=0)) for o in [h] + pending]
        best = None
        for idx in np.argsort(np.linalg.norm(R - M, axis=1)):
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
            best = int(idx)
            break
        if best is None:
            continue                                  # no bridge (degenerate hole): left out of the cap
        hole_walk = [h[(m + k) % len(h)] for k in range(len(h))] + [M.copy()]
        ring = ring[:best + 1] + hole_walk + [ring[best].copy()] + ring[best + 1:]
    return np.array(ring)


def triangulate_with_holes(outer: np.ndarray, holes: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    """(vertices, CCW triangles) of a polygon with holes."""
    ring = bridge_holes(outer, holes) if holes else ensure_ccw(outer)
    return ring, triangulate_polygon(ring)


def bevel_polygon(poly: np.ndarray, radius: float, segments: int = 3) -> tuple[np.ndarray, list[int]]:
    """Round every corner with a short arc (the rounded toy look); the ring is made CCW first."""
    return bevel_ring(ensure_ccw(poly), radius, segments)


def bevel_ring(poly: np.ndarray, radius: float, segments: int = 3) -> tuple[np.ndarray, list[int]]:
    """Round every corner of a ring (either orientation, kept) with a short arc.

    Returns the new ring and, per new vertex, the original edge it lies on: corner k produces
    segments + 1 vertices; the first lies on edge k-1, the last on edge k, the ones between on no
    edge (-1).
    """
    p = np.asarray(poly, float)
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


def bevel_radius(poly: np.ndarray, params: StyleParams) -> float:
    if params.bevel_m <= 0:
        return 0.0
    return float(min(params.bevel_max_m, max(params.bevel_m, params.bevel_frac * polygon_diameter(poly))))


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
    scale: float = 1.0                  # applied to x, y and z (about the ENU origin; z = 0 is the ground)

    def rotation(self) -> np.ndarray:
        c, s = math.cos(math.radians(self.rotation_deg)), math.sin(math.radians(self.rotation_deg))
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])

    def to_meta(self) -> dict[str, Any]:
        def r(v, k):
            return round(float(v), k) if math.isfinite(v) else None
        return {"rotation_deg": round(self.rotation_deg, 6), "shift_m": [round(float(v), 4) for v in self.shift],
                "scale": round(float(self.scale), 6),
                "median_wall_distance_before_m": r(self.median_before_m, 3),
                "median_wall_distance_after_m": r(self.median_after_m, 3),
                "num_points": self.num_points, "accepted": self.accepted, "reason": self.reason}

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> FootprintSnap:
        """A snap stored with an earlier model version (summary["snap"] / the GLB's scene extras), reused
        so that a republished model does not shift."""
        nan = float("nan")
        return cls(float(meta["rotation_deg"]), np.asarray(meta["shift_m"], float),
                   float(meta.get("median_wall_distance_before_m") or nan), float(meta.get("median_wall_distance_after_m") or nan),
                   int(meta.get("num_points", 0)), bool(meta.get("accepted", True)), "reused from an earlier model version",
                   float(meta.get("scale", 1.0)))


def _rigid_2d(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ms, md = src.mean(axis=0), dst.mean(axis=0)
    u, _, vt = np.linalg.svd((dst - md).T @ (src - ms))
    d = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ d @ vt
    return r, md - r @ ms


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


class _OutlineLookup:
    """Closest outline point by table lookup: the outline rasterized at `res`, nearest outline pixel of
    every grid pixel from cv2.distanceTransformWithLabels. ~res/2 accurate, O(1) per query."""

    def __init__(self, poly: np.ndarray, margin: float, res: float = 0.1, holes: list[np.ndarray] | None = None) -> None:
        self.lo = poly.min(axis=0) - margin
        hi = poly.max(axis=0) + margin
        self.res = res
        self.w, self.h = int(math.ceil((hi[0] - self.lo[0]) / res)), int(math.ceil((hi[1] - self.lo[1]) / res))
        img = np.full((self.h, self.w), 255, np.uint8)
        for ring in [poly] + list(holes or []):
            px = np.round((np.vstack([ring, ring[:1]]) - self.lo) / res * 16).astype(np.int32)
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
                      seed: int = 0, holes: list[np.ndarray] | None = None) -> FootprintSnap:
    """ARCHITECTURE 3.5.7: trimmed 2-D rigid ICP of wall-height points onto the footprint outline.

    Starts from no shift and from the shift putting the wall-band points' median on the footprint
    centroid, each with three start rotations; keeps the closest fit. Rejected (identity) beyond
    snap_max_rotation_deg / snap_max_shift_m or without improvement. Acceptance here is only the
    global fit; registration_check() then verifies it edge by edge.
    """
    pts = np.asarray(points, float)
    fp = ensure_ccw(footprint)
    rings = [fp] + list(holes or [])
    centroid = polygon_centroid(fp)
    radius = float(np.max(np.linalg.norm(fp - centroid, axis=1)))
    band = (pts[:, 2] >= z_band[0]) & (pts[:, 2] <= z_band[1])
    near = np.linalg.norm(pts[:, :2] - centroid, axis=1) <= radius + params.snap_max_shift_m
    xy = pts[band & near, :2]
    if len(xy) < 50:
        return FootprintSnap(0.0, np.zeros(2), float("nan"), float("nan"), len(xy), False, "too few wall points")
    if len(xy) > max_points:
        xy = xy[np.random.default_rng(seed).choice(len(xy), max_points, replace=False)]
    lookup = _OutlineLookup(fp, radius + 2 * params.snap_max_shift_m, holes=holes)
    before = float(np.median(distance_to_rings(xy, rings)[0]))
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
    after = float(np.median(distance_to_rings(xy @ r.T + t, rings)[0]))
    # bounded similarity refinement: the SfM scale from GPS is the weakest part of the georegistration
    # (the Aachen reference model is ~10 % too small for its OSM outline); scale about the outline's centroid
    scale = 1.0
    if params.snap_max_scale > 1.0:
        s2, r2, t2 = _similarity_icp(xy @ r.T + t, lookup, centroid, params.snap_max_scale)
        moved = s2 * (xy @ r.T + t - centroid) @ r2.T + centroid + t2
        after_s = float(np.median(distance_to_rings(moved, rings)[0]))
        if after_s < 0.9 * after and abs(s2 - 1.0) > 1e-3:
            scale = s2
            t = s2 * r2 @ t - s2 * r2 @ centroid + centroid + t2
            r = r2 @ r
            after = after_s
    rot = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    snap = FootprintSnap(rot, t, before, after, len(xy), True, scale=scale)
    if abs(rot) > params.snap_max_rotation_deg or np.linalg.norm(t) > params.snap_max_shift_m:
        snap.accepted, snap.reason = False, "correction beyond the snapping limits"
    elif after >= before:
        snap.accepted, snap.reason = False, "no improvement"
    elif after > params.snap_max_residual_m:
        snap.accepted, snap.reason = False, f"walls still {after:.1f} m off the footprint after snapping"
    if not snap.accepted:
        snap.rotation_deg, snap.shift, snap.scale = 0.0, np.zeros(2), 1.0
    return snap


def _similarity_icp(xy: np.ndarray, lookup: _OutlineLookup, centre: np.ndarray, max_scale: float,
                    iters: int = 60) -> tuple[float, np.ndarray, np.ndarray]:
    """Trimmed 2-D similarity ICP about `centre` with the scale clamped to [1/max_scale, max_scale]:
    p' = s R (p - centre) + centre + t. Returns (s, R, t)."""
    s, r, t = 1.0, np.eye(2), np.zeros(2)
    for _ in range(iters):
        moved = s * (xy - centre) @ r.T + centre + t
        target = lookup.closest(moved)
        d = np.linalg.norm(moved - target, axis=1)
        keep = d <= np.quantile(d, 0.7)
        a, b = moved[keep], target[keep]
        ma, mb = a.mean(axis=0), b.mean(axis=0)
        u, sv, vt = np.linalg.svd((b - mb).T @ (a - ma))
        dd = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
        dr = u @ dd @ vt
        ds = float((sv * np.diag(dd)).sum() / max(((a - ma) ** 2).sum(), 1e-12))
        s_new = float(np.clip(s * ds, 1.0 / max_scale, max_scale))
        ds = s_new / s
        dt = mb - ds * dr @ ma
        # moved' = ds dr moved + dt, with moved = s r (p - c) + c + t
        t = ds * dr @ (centre + t) + dt - centre
        r, s = dr @ r, s_new
        if abs(ds - 1.0) < 1e-6 and np.linalg.norm(dt) < 1e-5 and abs(dr[1, 0]) < 1e-7:
            break
    return s, r, t


def apply_snap_points(snap: FootprintSnap, points: np.ndarray) -> np.ndarray:
    out = np.asarray(points, float).copy()
    s = float(getattr(snap, "scale", 1.0))
    out[:, :2] = s * out[:, :2] @ snap.rotation()[:2, :2].T + snap.shift
    out[:, 2] *= s
    return out


def registration_check(points: np.ndarray | None, rings: list[np.ndarray], params: StyleParams = DEFAULT_PARAMS,
                       z_band: tuple[float, float] = (2.0, 20.0), max_dist_m: float = 10.0) -> dict[str, Any]:
    """Do the (snapped) points sit on the footprint edge by edge? Photo content is only published when
    they do: a single global median is dominated by the most photographed facade, so a shifted, rotated
    or too large outline can pass it (a 1.25x outline did, at 0.21 m).

    Wall-height points outside the outline or at most 3 m inside it (not roofs) are assigned to their
    nearest outline edge (within max_dist_m). Edges carrying at
    least reg_support_frac of the median supported edge's point count vote, one vote each (point-count
    weighting would again let the best facade decide); an edge is verified when its points' median
    distance is under reg_edge_limit_m. Accepted when reg_min_share of the voting edges are verified and
    the verified edges are not all parallel (>= 30 degrees apart).
    """
    out: dict[str, Any] = {"accepted": False, "reason": "", "edges": [], "verified_share": 0.0}
    if points is None or not len(points):
        out["reason"] = "no SfM points"
        return out
    pts = np.asarray(points, float)
    band = (pts[:, 2] >= z_band[0]) & (pts[:, 2] <= z_band[1])
    xy = pts[band, :2]
    if len(xy) > 60000:
        xy = xy[np.random.default_rng(0).choice(len(xy), 60000, replace=False)]
    if len(xy) < 50:
        out["reason"] = "too few wall points"
        return out
    d, _, ring_of, edge_of = distance_to_rings(xy, rings)
    inside = points_in_polygon(xy, rings[0])
    for h in rings[1:]:
        inside &= ~points_in_polygon(xy, h)
    near = (d <= max_dist_m) & (~inside | (d <= 3.0))
    edges = []
    for r, ring in enumerate(rings):
        for k in range(len(ring)):
            sel = near & (ring_of == r) & (edge_of == k)
            if sel.sum():
                edges.append({"ring": r, "edge": k, "points": int(sel.sum()), "median_m": round(float(np.median(d[sel])), 2)})
    if not edges:
        out["reason"] = f"no wall points within {max_dist_m:.0f} m of the outline"
        return out
    ref = float(np.median([e["points"] for e in edges]))
    voting = [e for e in edges if e["points"] >= max(20, params.reg_support_frac * ref)]
    for e in edges:
        e["votes"] = e in voting
        e["verified"] = bool(e["votes"] and e["median_m"] <= params.reg_edge_limit_m)
    share = sum(e["verified"] for e in voting) / max(len(voting), 1)
    out.update(edges=edges, verified_share=round(share, 3), voting_edges=len(voting),
               verified_edges=sum(e["verified"] for e in voting))
    # the verified edges must not all be parallel: a shift along them would go unnoticed
    ang = []
    for e in voting:
        if e["verified"]:
            ring = rings[e["ring"]]
            dv = ring[(e["edge"] + 1) % len(ring)] - ring[e["edge"]]
            ang.append(math.atan2(dv[1], dv[0]) % math.pi)
    spread = max((abs(math.degrees((x - y + math.pi / 2) % math.pi - math.pi / 2)) for x in ang for y in ang), default=0.0)
    out["verified_direction_spread_deg"] = round(spread, 1)
    if len(voting) < 2:
        out["reason"] = "fewer than 2 outline edges carry wall points"
    elif spread < 30.0:
        out["reason"] = "the verified outline edges are all parallel: a shift along them cannot be ruled out"
    elif share < params.reg_min_share:
        out["reason"] = (f"only {out['verified_edges']}/{len(voting)} supported outline edges have their points within "
                         f"{params.reg_edge_limit_m:.1f} m (median)")
    else:
        out["accepted"] = True
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
    photo_id: str | None = None          # Crowd3D photo id (attribution, deletion), when known
    license: str | None = None           # the photo's licence id, when known

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

    def snapped(self, snap: FootprintSnap) -> PosedCamera:
        """The same camera after the model was moved onto the footprint (points' = s R p + t)."""
        r = snap.rotation()
        s = float(getattr(snap, "scale", 1.0))
        c = self.center.copy()
        c[:2] = s * r[:2, :2] @ c[:2] + snap.shift
        c[2] *= s
        out = PosedCamera(**{k: getattr(self, k) for k in self.__dataclass_fields__})
        out.rot, out.center = self.rot @ r.T, c
        return out


# ---------------------------------------------------------------- massing


@dataclass
class HeightFacts:
    """Heights the product knows without photos (all optional).

    osm_height_m: OSM height=* (top of the roof, OSM wiki); osm_roof_height_m: roof:height=*, so that
    eave = height - roof:height; osm_levels: building:levels (eave = levels x level height);
    wikidata_height_m: Wikidata P2048, the total height of the structure (usually a spire or dome tip,
    not the eaves), so it only extrudes a tower-like outline.
    """

    osm_height_m: float | None = None
    osm_levels: float | None = None
    wikidata_height_m: float | None = None
    osm_roof_height_m: float | None = None


@dataclass
class Prism:
    """One extruded block: a CCW outline (plus clockwise courtyard holes) from z0 to z1 and a cap
    (flat, or a pyramid whose unobserved faces are a neutral flat cap)."""

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
    holes: list = field(default_factory=list)          # (k, 2) CW rings: courtyards (base block only)
    face_coverage: list = field(default_factory=list)  # per cap face: share of its area (cells) with roof points
    face_sloped: list = field(default_factory=list)    # per pyramid face: keeps the fitted slope (else neutral flat)
    moved_edges: set = field(default_factory=set)      # (ring, edge) moved onto the facade plane fitted to the points

    @property
    def rings(self) -> list[np.ndarray]:
        return [self.polygon] + list(self.holes)

    def contains(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        inside = points_in_polygon(xy, self.polygon)
        for h in self.holes:
            inside &= ~points_in_polygon(xy, h)
        return inside

    def num_faces(self) -> int:
        return len(self.polygon) if self.cap == "pyramid" else 1

    def face_of(self, xy: np.ndarray) -> np.ndarray:
        """Cap face index of each (x, y): the pyramid sector (edge k -> apex) or 0 for a flat cap."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.cap != "pyramid":
            return np.zeros(len(xy), int)
        rel = xy - self.apex[:2]
        ang = np.arctan2(rel[:, 1], rel[:, 0])
        p = self.polygon - self.apex[:2]
        va = np.arctan2(p[:, 1], p[:, 0])
        face = np.zeros(len(xy), dtype=int)
        for k in range(len(p)):
            span = (va[(k + 1) % len(p)] - va[k]) % (2 * math.pi)
            face[((ang - va[k]) % (2 * math.pi)) <= span] = k
        return face

    def to_meta(self) -> dict[str, Any]:
        return {"index": self.index, "parent": self.parent, "depth": self.depth, "vertices": len(self.polygon),
                "holes": len(self.holes), "area_m2": round(abs(signed_area(self.polygon)) - sum(abs(signed_area(h)) for h in self.holes), 1),
                "z0": round(self.z0, 2), "z1": round(self.z1, 2),
                "outline_source": self.outline_source, "height_source": self.height_source, "cap": self.cap,
                "apex_z": None if self.apex is None else round(float(self.apex[2]), 2), "cap_source": self.cap_source,
                "measured_fraction": round(self.measured_fraction, 3),
                "face_coverage": [round(float(c), 3) for c in self.face_coverage],
                "face_sloped": [bool(s) for s in self.face_sloped],
                "moved_edges": sorted([list(e) for e in self.moved_edges])}

    def roof_height(self, xy: np.ndarray, fitted: bool = False) -> np.ndarray:
        """Height of this block's cap above each (x, y) inside its outline. fitted=True: the fitted pyramid
        everywhere, also where a face was left as a neutral flat cap."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.cap != "pyramid":
            return np.full(len(xy), self.z1)
        h = self.z1 + (self.apex[2] - self.z1) * (1.0 - _radial_fraction(xy, self.polygon, self.apex[:2]))
        if fitted or not self.face_sloped or all(self.face_sloped):
            return h
        sloped = np.asarray(self.face_sloped, bool)[self.face_of(xy)]
        return np.where(sloped, h, self.z1)


@dataclass
class Wall:
    index: int
    prism: int
    edge: int                           # edge index in its ring of the prism's (un-bevelled) outline
    a: np.ndarray                       # (2,) start, CCW order along the outline (clockwise along a hole)
    b: np.ndarray
    z0: float
    z1: float
    outline_source: str
    height_source: str
    top_measured_m: float | None = None  # this edge's own top from its wall points, when measured
    ring: int = 0                        # 0: outer ring, k >= 1: courtyard hole k
    block_measured: bool = False         # its block was traced in the points (height measured on the block itself)

    height_tol_m: float = 3.0            # a measured top within this of z1 confirms the wall's height (= step_m)

    @property
    def height_provenance(self) -> str:
        """sfm-points only when this very wall's top was measured at its height (or its block was traced
        in the points); a block height carried over to a wall without points, or to one whose own top is
        elsewhere, is 'inferred'."""
        if self.height_source == PROV_SFM and not self.block_measured and (
                self.top_measured_m is None or abs(self.top_measured_m - self.z1) > self.height_tol_m):
            return PROV_INFERRED
        return self.height_source

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

    def grid(self, w: int, h: int, offset: float = 0.0, offset_b: float | None = None) -> np.ndarray:
        """(h, w, 3) ENU sample centres; row 0 is the top of the wall, column 0 its start a.
        The sampled plane is moved along the outward normal by `offset` at a and `offset_b` at b
        (default: the same), i.e. a facade plane fitted with an offset and a yaw."""
        u = (np.arange(w) + 0.5) / w * self.length
        z = self.z1 - (np.arange(h) + 0.5) / h * (self.z1 - self.z0)
        ob = offset if offset_b is None else offset_b
        off = offset + (ob - offset) * u / max(self.length, 1e-9)
        xy = self.a[None, :] + u[:, None] * self.tangent[None, :] + off[:, None] * self.normal[None, :]
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
    roof_grids: dict = field(default_factory=dict)     # prism index -> RoofGrid

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
                flat_apex = np.r_[pr.apex[:2], pr.z1]
                for k in range(len(p)):
                    top = pr.apex if (not pr.face_sloped or pr.face_sloped[k]) else flat_apex
                    tris.append((np.r_[p[k], pr.z1], np.r_[p[(k + 1) % len(p)], pr.z1], top))
            else:
                ring, tri = triangulate_with_holes(p, pr.holes)
                for i, j, k in tri:
                    tris.append((np.r_[ring[i], pr.z1], np.r_[ring[j], pr.z1], np.r_[ring[k], pr.z1]))
        return np.array(tris, dtype=float).reshape(-1, 3, 3)


def height_field(points: np.ndarray, footprint: np.ndarray, cell: float,
                 min_points: int = 3, q: float = 0.9, holes: list[np.ndarray] | None = None
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Per grid cell inside the footprint: robust top height of the points standing on it (NaN: no data).

    Returns (heights (ny, nx), counts, inside mask, (x0, y0) of the grid's lower-left corner).
    Row 0 is the southernmost row.
    """
    fp = ensure_ccw(footprint)
    lo = fp.min(axis=0) - cell
    hi = fp.max(axis=0) + cell
    nx, ny = int(math.ceil((hi[0] - lo[0]) / cell)), int(math.ceil((hi[1] - lo[1]) / cell))
    gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * cell, lo[1] + (np.arange(ny) + 0.5) * cell)
    g = np.stack([gx.ravel(), gy.ravel()], 1)
    inside = points_in_polygon(g, fp)
    for h in holes or []:
        inside &= ~points_in_polygon(g, h)
    inside = inside.reshape(ny, nx)
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
    """Per outline edge: the top of its wall from the points standing on it (None: too few points).
    The ring is used as given (a hole keeps its clockwise order)."""
    p = np.asarray(poly, float)
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


@dataclass
class PlaneFit:
    """A vertical facade plane fitted to the points near a wall: its signed distance (outward +) from the
    wall's face at the wall's start a and end b, i.e. an offset and a yaw."""

    offset_a: float
    offset_b: float
    length: float
    inliers: int
    candidates: int
    rms_m: float

    @property
    def yaw_deg(self) -> float:
        return math.degrees(math.atan2(self.offset_b - self.offset_a, max(self.length, 1e-9)))

    @property
    def max_offset(self) -> float:
        return max(abs(self.offset_a), abs(self.offset_b))

    def offset_at(self, u: np.ndarray) -> np.ndarray:
        return self.offset_a + (self.offset_b - self.offset_a) * np.asarray(u, float) / max(self.length, 1e-9)

    def within(self, tol_m: float, tol_deg: float) -> bool:
        return self.max_offset <= tol_m and abs(self.yaw_deg) <= tol_deg

    def to_meta(self) -> dict[str, Any]:
        return {"offset_a_m": round(self.offset_a, 2), "offset_b_m": round(self.offset_b, 2),
                "yaw_deg": round(self.yaw_deg, 1), "inliers": self.inliers, "candidates": self.candidates,
                "rms_m": round(self.rms_m, 3)}


def fit_wall_plane(a: np.ndarray, b: np.ndarray, points: np.ndarray | None, z_lo: float, z_hi: float,
                   params: StyleParams = DEFAULT_PARAMS, seed: int = 0) -> PlaneFit | None:
    """RANSAC line in (along-wall u, signed distance d) over the points within plane_search_m of the face
    (u inside the middle 90 %, z in [z_lo, z_hi]), refined by least squares on its inliers. The densest
    vertical surface wins (a recessed wall between buttresses, not the buttress tips, when it carries more
    points). None when fewer than plane_min_inliers points support any plane. Yaw is fitted only when the
    inliers span 40 % of the wall; otherwise the plane is parallel to the face."""
    if points is None or not len(points):
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return None
    t = (b - a) / L
    n = np.array([t[1], -t[0]])
    pts = np.asarray(points, float)
    rel = pts[:, :2] - a
    u, d = rel @ t, rel @ n
    sel = ((np.abs(d) <= params.plane_search_m) & (u >= 0.05 * L) & (u <= 0.95 * L)
           & (pts[:, 2] >= z_lo) & (pts[:, 2] <= z_hi))
    uu, dd = u[sel], d[sel]
    m = len(uu)
    if m < params.plane_min_inliers:
        return None
    rng = np.random.default_rng(seed)
    if m > 3000:
        pick = rng.choice(m, 3000, replace=False)
        uu, dd = uu[pick], dd[pick]
    tol = params.plane_inlier_m
    max_slope = math.tan(math.radians(params.plane_max_yaw_deg))
    i = rng.integers(len(uu), size=300)
    j = rng.integers(len(uu), size=300)
    du = uu[j] - uu[i]
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(np.abs(du) >= 0.2 * L, (dd[j] - dd[i]) / du, 0.0)
    slope = np.where(np.abs(slope) <= max_slope, slope, 0.0)
    slope = np.r_[slope, np.zeros(100)]                       # plus planes parallel to the face
    icpt = np.r_[dd[i] - slope[:300] * uu[i], dd[rng.integers(len(uu), size=100)]]
    counts = (np.abs(dd[None, :] - (slope[:, None] * uu[None, :] + icpt[:, None])) <= tol).sum(axis=1)
    best = counts.max()
    # among the (near-)best candidates prefer the one closest to the face (ties between parallel surfaces)
    good = np.nonzero(counts >= 0.95 * best)[0]
    k = good[np.argmin(np.abs(icpt[good] + slope[good] * L / 2))]
    s, c = float(slope[k]), float(icpt[k])
    inl = np.abs(dd - (s * uu + c)) <= tol
    for _ in range(3):
        if inl.sum() < params.plane_min_inliers:
            break
        span = np.ptp(uu[inl])
        if span >= 0.4 * L:
            s, c = (float(v) for v in np.polyfit(uu[inl], dd[inl], 1))
            if abs(s) > max_slope:
                s, c = 0.0, float(np.median(dd[inl]))
        else:
            s, c = 0.0, float(np.median(dd[inl]))
        inl = np.abs(dd - (s * uu + c)) <= tol
    if inl.sum() < params.plane_min_inliers:
        return None
    rms = float(np.sqrt(np.mean((dd[inl] - (s * uu[inl] + c)) ** 2)))
    scale = m / len(uu)
    return PlaneFit(c, s * L + c, L, int(round(inl.sum() * scale)), m, rms)


def _line_vertex(p_prev: np.ndarray, d_prev: np.ndarray, p_next: np.ndarray, d_next: np.ndarray,
                 fallback: np.ndarray) -> np.ndarray:
    """Corner of two outline lines; near-parallel lines (< 15 deg) meet at the mean of the fallback
    vertex's projections onto both."""
    s = abs(_cross2(d_prev, d_next)) / max(np.linalg.norm(d_prev) * np.linalg.norm(d_next), 1e-12)
    x = _line_intersection(p_prev, d_prev, p_next, d_next) if s > math.sin(math.radians(15)) else None
    if x is not None:
        return x

    def proj(p, d):
        dn = d / max(np.linalg.norm(d), 1e-12)
        return p + ((fallback - p) @ dn) * dn
    return (proj(p_prev, d_prev) + proj(p_next, d_next)) / 2


def refine_ring(ring: np.ndarray, points: np.ndarray | None, z_lo: float, z_hi: float,
                params: StyleParams = DEFAULT_PARAMS, allowed: Any = None) -> tuple[np.ndarray, list[PlaneFit | None], set[int]]:
    """Move outline edges onto the facade planes fitted to the points where the facade is clearly off the
    simplified outline (OSM outlines include plinths, steps or buttress tips; simplification cuts across
    detail) or turned against it.

    A fitted plane may pull an edge inwards by up to plane_max_move_m; it may push it outwards only while
    the moved face stays within the real (unsimplified) footprint, `allowed(xy)` (recovering detail the
    simplification lost). A surface outside the footprint is something standing in front of the building
    (a neighbouring house, a chapel) and never pulls the outline out; such walls stay untextured
    (observe_walls). The ring keeps its orientation; neighbours are re-intersected. Moves are added
    greedily, largest first, each kept only while the ring stays simple, keeps at least half its area and
    no edge longer than 0.5 m turns by more than 30 degrees.
    Returns (ring, per-edge fits, indices of moved edges)."""
    p = np.asarray(ring, float)
    n = len(p)
    fits = [fit_wall_plane(p[k], p[(k + 1) % n], points, z_lo, z_hi, params) for k in range(n)]
    want: set[int] = set()
    for k, f in enumerate(fits):
        if f is None or min(f.offset_a, f.offset_b) < -params.plane_max_move_m:
            continue
        if not (f.max_offset >= params.plane_move_m or abs(f.yaw_deg) >= params.plane_move_deg):
            continue
        if max(f.offset_a, f.offset_b) > params.plane_tol_m:
            if allowed is None or max(f.offset_a, f.offset_b) > params.plane_max_move_m:
                continue
            a, b = p[k], p[(k + 1) % n]
            t = (b - a) / max(np.linalg.norm(b - a), 1e-12)
            nn = np.array([t[1], -t[0]])
            s = np.linspace(0.05, 0.95, 7)
            line = (a + s[:, None] * (b - a)) + (f.offset_a + s * (f.offset_b - f.offset_a))[:, None] * nn
            if not np.all(allowed(line)):
                continue
        want.add(k)
    area0 = signed_area(p)

    def moved_ring(edges: set[int]) -> np.ndarray | None:
        lines = []
        for k in range(n):
            a, b = p[k], p[(k + 1) % n]
            t = (b - a) / max(np.linalg.norm(b - a), 1e-12)
            nn = np.array([t[1], -t[0]])
            if k in edges:
                pa, pb = a + fits[k].offset_a * nn, b + fits[k].offset_b * nn
                lines.append((pa, pb - pa))
            else:
                lines.append((a, b - a))
        q = np.array([_line_vertex(lines[k - 1][0], lines[k - 1][1], lines[k][0], lines[k][1], p[k]) for k in range(n)])
        if not (ring_is_simple(q) and np.sign(signed_area(q)) == np.sign(area0) and abs(signed_area(q)) >= 0.5 * abs(area0)):
            return None
        if float(np.linalg.norm(q - p, axis=1).max()) > params.plane_max_move_m + 1.0:
            return None
        dq, dp = np.roll(q, -1, axis=0) - q, np.roll(p, -1, axis=0) - p
        lq, lp = np.linalg.norm(dq, axis=1), np.linalg.norm(dp, axis=1)
        cos = (dq * dp).sum(axis=1) / np.maximum(lq * lp, 1e-12)
        if not np.all((cos > math.cos(math.radians(30))) | (lq < 0.5)):
            return None                                   # an edge turned or flipped (short jogs may shrink away)
        return q

    # greedy: the largest corrections first, each kept when the ring stays valid with it
    done: set[int] = set()
    best = p
    for k in sorted(want, key=lambda k: -(fits[k].max_offset + 0.1 * abs(fits[k].yaw_deg))):
        q = moved_ring(done | {k})
        if q is not None:
            done.add(k)
            best = q
    return best, fits, done


def _wall_top_near(wall: Wall, points: np.ndarray, params: StyleParams) -> float | None:
    """This wall's own top from its points (band wall_band_m, middle 80 %, up to step_m above its block's
    height), or None with fewer than min_edge_points: what 'height measured here' means."""
    rel = points[:, :2] - wall.a
    u, d = rel @ wall.tangent, rel @ wall.normal
    sel = ((np.abs(d) <= params.wall_band_m) & (u >= 0.1 * wall.length) & (u <= 0.9 * wall.length)
           & (points[:, 2] > wall.z0 + 1.0) & (points[:, 2] <= wall.z1 + params.step_m))
    if sel.sum() < params.min_edge_points:
        return None
    return float(np.quantile(points[sel, 2], params.edge_top_quantile))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v, w = np.asarray(values, float)[order], np.asarray(weights, float)[order]
    c = np.cumsum(w)
    return float(v[min(len(v) - 1, int(np.searchsorted(c, q * c[-1])))])


def _fit_cap(pr: Prism, cells_xy: np.ndarray, cells_h: np.ndarray, params: StyleParams) -> np.ndarray:
    """Fit a flat or pyramid cap to the height-field cells of a block; returns the cells' roof heights.

    GENERATIVE-OK: a roof-type / shape-simplification model (gable, hip, dome, cone) fitted to the
    observed height field would go here; it may only shape roof parts that have point support."""
    flat = np.full(len(cells_h), pr.z1)
    if len(cells_h) < 12:
        return flat
    pr.cap_source = PROV_SFM
    if pr.holes:
        return flat                                      # courtyards: flat roofs only
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


@dataclass
class RoofGrid:
    """A block's cap on a roof_cell_m grid: which cells belong to it (inside its outline, not under a
    block standing on it), which cap face each cell is on, and which cells have roof points."""

    prism: int
    origin: np.ndarray                  # (2,) lower-left corner
    cell: float
    region: np.ndarray                  # (ny, nx) bool
    face: np.ndarray                    # (ny, nx) int
    covered: np.ndarray                 # (ny, nx) bool: at least one roof point

    def centres(self) -> np.ndarray:
        return _cell_centres(self.region.shape, tuple(self.origin), self.cell)

    def cell_of(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ix = np.floor((xy[:, 0] - self.origin[0]) / self.cell).astype(int)
        iy = np.floor((xy[:, 1] - self.origin[1]) / self.cell).astype(int)
        ok = (ix >= 0) & (ix < self.region.shape[1]) & (iy >= 0) & (iy < self.region.shape[0])
        return ix, iy, ok

    def coverage(self, face: int | None = None) -> float:
        sel = self.region if face is None else self.region & (self.face == face)
        return float(self.covered[sel].mean()) if sel.any() else 0.0

    def area_m2(self, face: int | None = None) -> float:
        sel = self.region if face is None else self.region & (self.face == face)
        return float(sel.sum() * self.cell * self.cell)


def roof_point_mask(pr: Prism, points: np.ndarray, params: StyleParams = DEFAULT_PARAMS) -> np.ndarray:
    """Points on a block's (fitted) cap: from 1.5 m below it to step_m above it."""
    roof = pr.roof_height(points[:, :2], fitted=True)
    return (points[:, 2] >= roof - 1.5) & (points[:, 2] <= roof + params.step_m)


def roof_grid(pr: Prism, children: list[Prism], points: np.ndarray | None, params: StyleParams = DEFAULT_PARAMS) -> RoofGrid:
    cell = params.roof_cell_m
    lo = pr.polygon.min(axis=0) - cell
    hi = pr.polygon.max(axis=0) + cell
    nx, ny = int(math.ceil((hi[0] - lo[0]) / cell)), int(math.ceil((hi[1] - lo[1]) / cell))
    c = _cell_centres((ny, nx), (float(lo[0]), float(lo[1])), cell).reshape(-1, 2)
    region = pr.contains(c)
    for ch in children:
        region &= ~points_in_polygon(c, ch.polygon)
    grid = RoofGrid(pr.index, lo, cell, region.reshape(ny, nx), pr.face_of(c).reshape(ny, nx), np.zeros((ny, nx), bool))
    if points is not None and len(points):
        pts = np.asarray(points, float)
        box = (pts[:, 0] >= lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] >= lo[1]) & (pts[:, 1] < hi[1])
        p = pts[box]
        ix, iy, ok = grid.cell_of(p[:, :2])
        ok &= roof_point_mask(pr, p, params)
        grid.covered[iy[ok], ix[ok]] = True
        grid.covered &= grid.region
    return grid


def fallback_height(facts: HeightFacts, outline: np.ndarray, params: StyleParams = DEFAULT_PARAMS) -> tuple[float, str, str]:
    """(eave height, provenance, note) when the points say nothing about a block's height."""
    if facts.osm_height_m and facts.osm_roof_height_m and facts.osm_height_m > facts.osm_roof_height_m:
        return float(facts.osm_height_m - facts.osm_roof_height_m), PROV_OSM_HEIGHT, "OSM height minus roof:height"
    if facts.osm_levels:
        return float(facts.osm_levels) * params.level_height_m, PROV_OSM_HEIGHT, \
            f"OSM building:levels x {params.level_height_m:g} m"
    if facts.osm_height_m:
        return float(facts.osm_height_m), PROV_OSM_HEIGHT, "OSM height (top of the roof) used for a flat roof: roof shape unknown"
    if facts.wikidata_height_m:
        h = float(facts.wikidata_height_m)
        diam = polygon_diameter(outline)
        if h >= params.tower_ratio * diam:
            return h, PROV_WIKIDATA, f"Wikidata P2048 {h:g} m on a tower-like outline ({diam:.0f} m across)"
        return min(params.default_height_m, h), PROV_DEFAULT, \
            (f"Wikidata P2048 {h:g} m is the total height (spire / dome tip), not the eaves; "
             f"the outline ({diam:.0f} m across) is not a tower, so it is not extruded to it")
    return params.default_height_m, PROV_DEFAULT, "no height facts: style default"


def build_massing(footprint: np.ndarray, points: np.ndarray | None, facts: HeightFacts | None = None,
                  params: StyleParams = DEFAULT_PARAMS, holes: list[np.ndarray] | None = None,
                  outline_source: str = PROV_OSM) -> Massing:
    """Base block from the simplified footprint (with its courtyards), roofs fitted to the points,
    blocks above roofs found recursively ("wedding cake" with pyramids).

    Outline edges whose facade stands clearly inside the outline are moved onto the plane fitted to the
    points (refine_ring). Base height: the base_quantile (length-weighted) of the measured edge tops, so
    that a lower annex sets the base and taller parts become blocks of their own. Without usable points
    the footprint is extruded to the height facts (fallback_height), with a flat cap. Roof faces get
    their point coverage; pyramid faces without enough of it are a neutral flat cap (the fitted slope is
    not extended over places nobody saw).
    """
    facts = facts or HeightFacts()
    t0 = time.perf_counter()
    simple, fp_report = simplify_footprint(footprint, params)
    report: dict[str, Any] = {"footprint": fp_report}
    hole_rings: list[np.ndarray] = []
    for h in holes or []:
        hc = clean_ring(h, 10.0)
        if hc is None:
            continue
        hs, _ = simplify_footprint(hc, params, max_vertices=12)
        if len(hs) >= 3 and points_in_polygon(hs, simple).all() and abs(signed_area(hs)) >= 10.0:
            hole_rings.append(ensure_ccw(hs)[::-1].copy())       # clockwise
    report["footprint"]["holes"] = len(hole_rings)
    fb_h, fb_src, fb_note = fallback_height(facts, simple, params)

    pts = None if points is None or not len(points) else np.asarray(points, float)
    real = [ensure_ccw(footprint)] + [ensure_ccw(h) for h in holes or []]

    def allowed(xy: np.ndarray) -> np.ndarray:
        """Inside the real footprint (not in a courtyard), or within 0.75 m of its outline."""
        inside = points_in_polygon(xy, real[0])
        for h in real[1:]:
            inside &= ~points_in_polygon(xy, h)
        return inside | (distance_to_rings(xy, real)[0] <= 0.75)

    def base_height(rings: list[np.ndarray]) -> tuple[list[list[float | None]], float | None]:
        tops_by_ring = [edge_tops(ring, pts, params) if pts is not None else [None] * len(ring) for ring in rings]
        tops = [v for t in tops_by_ring for v in t]
        lengths = np.concatenate([np.linalg.norm(np.roll(ring, -1, axis=0) - ring, axis=1) for ring in rings])
        measured = [k for k, v in enumerate(tops) if v is not None]
        if not measured:
            return tops_by_ring, None
        # a low quantile so that a lower annex sets the base (taller parts become blocks), then the
        # median of the tops within 1.5 steps above it (one tier: sparse tops underestimate)
        mt, ml = np.array([tops[k] for k in measured]), lengths[measured]
        q = _weighted_quantile(mt, ml, params.base_quantile)
        tier = (mt >= q) & (mt <= q + 1.5 * params.step_m)
        return tops_by_ring, _weighted_quantile(mt[tier], ml[tier], 0.5)

    moved: set[tuple[int, int]] = set()
    if pts is not None:
        # two passes: the ground storey first (z 1-10 m: what an outline describes), then the whole base
        # block height on the refined outline
        fits_report = []
        rings = [simple] + hole_rings
        for z_hi in (10.0, None):
            if z_hi is None:
                h = base_height(rings)[1]
                z_hi = h - 0.5 if h is not None else 10.0      # wall points, not the roof edge
            refined = []
            fits_report = []
            for r, ring in enumerate(rings):
                q, fits, mv = refine_ring(ring, pts, 1.0, z_hi, params, allowed)
                refined.append(q)
                moved |= {(r, k) for k in mv}
                fits_report += [{"ring": r, "edge": k, **(f.to_meta() if f else {}), "moved": k in mv}
                                for k, f in enumerate(fits)]
            rings = refined
        simple, hole_rings = rings[0], rings[1:]
        report["outline_refinement"] = {"moved_edges": len(moved), "fits": fits_report}
    rings = [simple] + hole_rings
    tops_by_ring, base_h = base_height(rings)
    tops = [v for t in tops_by_ring for v in t]
    if base_h is not None:
        base_src = PROV_SFM
    else:
        base_h, base_src = fb_h, fb_src
    report["edge_tops_m"] = [None if v is None else round(v, 2) for v in tops]
    report["base_height"] = {"m": round(base_h, 2), "source": base_src}
    report["fallback_height"] = {"m": fb_h, "source": fb_src, "note": fb_note}
    report["height_facts"] = {k: getattr(facts, k) for k in ("osm_height_m", "osm_roof_height_m", "osm_levels", "wikidata_height_m")}

    prisms = [Prism(0, None, 0, simple, 0.0, base_h, outline_source, base_src, holes=hole_rings, moved_edges=moved)]
    if pts is not None:
        cell = params.cell_m
        heights, _counts, inside, origin = height_field(pts, simple, cell, holes=hole_rings)
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
                    poly = regularize_block(poly, params)
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
                    poly, _, mv = refine_ring(poly, pts, z0 + 0.5, z1 - 0.5, params, allowed)
                    poly, _, mv2 = refine_ring(poly, pts, z0 + 0.5, z1 - 0.5, params, allowed)
                    child = Prism(len(prisms), pr.index, pr.depth + 1, poly, z0, z1, PROV_SFM, PROV_SFM,
                                  moved_edges={(0, k) for k in mv | mv2})
                    prisms.append(child)
                    grow(child, sub)

        grow(prisms[0], inside.copy())
    else:
        report["measured_cell_fraction"] = 0.0

    # roof coverage per face; pyramid faces nobody saw become a neutral flat cap
    grids: dict[int, RoofGrid] = {}
    for pr in prisms:
        children = [c for c in prisms if c.parent == pr.index]
        g = roof_grid(pr, children, pts, params)
        if pr.cap == "pyramid":
            cov = [g.coverage(k) for k in range(len(pr.polygon))]
            pr.face_sloped = [c >= params.roof_face_min_coverage for c in cov]
            if not any(pr.face_sloped):
                pr.cap, pr.apex, pr.face_sloped = "flat", None, []
                g = roof_grid(pr, children, pts, params)
        pr.face_coverage = [g.coverage(k) for k in range(pr.num_faces())]
        if pr.cap == "flat":
            pr.cap_source = PROV_SFM if pr.face_coverage[0] >= params.roof_face_min_coverage else PROV_DEFAULT
        grids[pr.index] = g

    walls: list[Wall] = []
    for pr in prisms:
        for r, ring in enumerate(pr.rings):
            for k in range(len(ring)):
                w = Wall(len(walls), pr.index, k, ring[k].copy(), ring[(k + 1) % len(ring)].copy(),
                         pr.z0, pr.z1, PROV_SFM if (r, k) in pr.moved_edges else pr.outline_source, pr.height_source,
                         ring=r, block_measured=pr.index != 0)
                if pr.index == 0 and pts is not None:
                    w.top_measured_m = _wall_top_near(w, pts, params)
                walls.append(w)
    report["prisms"] = [p.to_meta() for p in prisms]
    report["num_walls"] = len(walls)
    report["seconds"] = round(time.perf_counter() - t0, 3)
    return Massing(prisms, walls, report, grids)


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
    offset_a: float                            # sampled facade plane: signed distance from the face at a ...
    offset_b: float                            # ... and at b (outward +), from the fitted plane (0 without one)
    support: np.ndarray                        # (gh, gw) bool on the visibility grid: SfM points near the plane
    plane: PlaneFit | None = None
    observed: bool = False
    reason: str = ""

    @property
    def plane_offset_m(self) -> float:
        return (self.offset_a + self.offset_b) / 2

    def to_meta(self) -> dict[str, Any]:
        return {"wall": self.wall, "num_cameras": len(self.cameras), "num_points": self.num_points,
                "plane": self.plane.to_meta() if self.plane else None, "observed": self.observed, "reason": self.reason,
                "support_fraction": round(float(self.support.mean()), 3) if self.support.size else 0.0}


def _vis_grid_shape(wall: Wall, step: float) -> tuple[int, int]:
    return max(2, int(round((wall.z1 - wall.z0) / step))), max(2, int(round(wall.length / step)))


def _wall_points(wall: Wall, points: np.ndarray, band: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points on a wall's plane: (u along the wall, z, signed distance) for those within the band."""
    rel = points[:, :2] - wall.a
    u, d = rel @ wall.tangent, rel @ wall.normal
    sel = (np.abs(d) <= band) & (u >= 0) & (u <= wall.length) & (points[:, 2] >= wall.z0) & (points[:, 2] <= wall.z1)
    return u[sel], points[sel, 2], d[sel]


def plane_verdict(fit: PlaneFit | None, params: StyleParams = DEFAULT_PARAMS) -> str:
    """'' when a facade may be textured from this plane fit, else the reason it may not."""
    if fit is None:
        return "no facade plane in the points"
    if fit.inliers < params.plane_min_inlier_share * fit.candidates:
        return (f"no clear facade plane (only {fit.inliers / max(fit.candidates, 1):.0%} of the points near this face "
                f"lie on one surface)")
    if max(fit.offset_a, fit.offset_b) > params.plane_tol_m:
        return (f"the surface the photos show stands {max(fit.offset_a, fit.offset_b):.1f} m outside the outline "
                f"(something in front of this wall)")
    if fit.max_offset > params.plane_tol_m or abs(fit.yaw_deg) > params.plane_tol_deg:
        return f"facade plane {fit.max_offset:.1f} m / {abs(fit.yaw_deg):.0f} deg off this face"
    return ""


def observe_walls(massing: Massing, cameras: list[PosedCamera], points: np.ndarray | None,
                  params: StyleParams = DEFAULT_PARAMS) -> list[WallObservation]:
    """Which cameras see which walls (front side, angle, inside the image, not hidden by the massing),
    where the facade plane is (fitted to the points: offset and yaw, no clipping) and where SfM points
    support it. Decides observed / locked per wall: a wall whose fitted plane is outside the outline or
    more than plane_tol_m / plane_tol_deg off its face is not textured (its photos would land on the
    wrong surface)."""
    pts = None if points is None or not len(points) else np.asarray(points, float)
    cos_max = math.cos(math.radians(params.max_view_angle_deg))
    out = []
    for wall in massing.walls:
        gh, gw = _vis_grid_shape(wall, params.vis_step_m)
        fit = None
        oa = ob_ = 0.0
        npts = 0
        support = np.zeros((gh, gw), dtype=bool)
        if pts is not None:
            fit = fit_wall_plane(wall.a, wall.b, pts, wall.z0 + 0.5, wall.z1 - 0.5, params)
            if fit is not None and fit.within(params.plane_search_m, params.plane_max_yaw_deg):
                oa, ob_ = fit.offset_a, fit.offset_b
            u, z, d = _wall_points(wall, pts, params.plane_search_m + params.support_band_m)
            near = np.abs(d - (oa + (ob_ - oa) * u / max(wall.length, 1e-9))) <= params.support_band_m
            u, z = u[near], z[near]
            npts = len(u)
            if npts:
                iu = np.clip((u / wall.length * gw).astype(int), 0, gw - 1)
                iz = np.clip(((wall.z1 - z) / (wall.z1 - wall.z0) * gh).astype(int), 0, gh - 1)
                support[iz, iu] = True
                r = max(1, int(round(params.support_radius_m / params.vis_step_m)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
                support = cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
                support = cv2.dilate(support, kernel) > 0
        grid = wall.grid(gw, gh, oa, ob_).reshape(-1, 3)
        tris = massing.triangles(skip_wall=wall.index)
        centre = wall.grid(1, 1, (oa + ob_) / 2).reshape(3)
        cams = []
        for ci, cam in enumerate(cameras):
            rel = cam.center[:2] - wall.a
            if rel @ wall.normal - max(oa, ob_) < params.min_camera_distance_m:
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
            dist = float(np.linalg.norm(cam.center - centre))
            ppm = cam.fx / max(dist, 1e-6)                       # image pixels per metre on the wall
            cos_c = float(((cam.center - centre)[:2] @ wall.normal) / max(dist, 1e-9))
            cams.append((ci, frac, frac * max(cos_c, 0.05) * min(ppm, 60.0)))
        cams.sort(key=lambda c: -c[2])
        ob = WallObservation(wall.index, cams, npts, oa, ob_, support, fit)
        good = [c for c in cams if c[1] >= params.min_wall_visible_fraction]
        verdict = plane_verdict(fit, params) if pts is not None else ""
        if len(good) < params.min_wall_cameras:
            ob.reason = f"{len(good)} camera(s) see it"
        elif pts is not None and npts < params.min_wall_points:
            ob.reason = f"only {npts} SfM points on it"
        elif pts is not None and npts < params.min_point_density * wall.area:
            ob.reason = f"only {npts / max(wall.area, 1e-9):.2f} SfM points per m2 on it"
        elif verdict:
            ob.reason = verdict
        else:
            ob.observed = True
        out.append(ob)
    return out


# ---------------------------------------------------------------- facade orthophotos


@dataclass
class Orthophoto:
    rgb: np.ndarray                     # (h, w, 3) uint8, rectified; row 0 = top of the wall
    count: np.ndarray                   # (h, w) number of photos that saw each pixel (unoccluded)
    mask: np.ndarray                    # (h, w) bool: observed (enough agreeing views and SfM support)
    px_per_m: float
    cameras: list[str]                  # photos blended (names)
    contributions: list = field(default_factory=list)   # (camera index, share of the mask it agrees on)
    samples: list = field(default_factory=list)         # audit: (row, col, [camera indices agreeing there])
    stats: dict = field(default_factory=dict)

    @property
    def observed_fraction(self) -> float:
        return float(self.mask.mean())


def outside_footprint(points: np.ndarray, rings: list[np.ndarray]) -> np.ndarray:
    """Points standing outside the building's outline (or in one of its courtyards)."""
    xy = np.asarray(points, float)[:, :2]
    inside = points_in_polygon(xy, rings[0])
    for h in rings[1:]:
        inside &= ~points_in_polygon(xy, h)
    return ~inside


def occluder_eligible(points: np.ndarray, quality: np.ndarray | None, params: StyleParams = DEFAULT_PARAMS) -> np.ndarray:
    """Points trusted as occluders: with a known quality (track length, triangulation angle in degrees),
    those seen by >= occluder_min_track photos under >= occluder_min_angle_deg. Badly triangulated points
    smear along the viewing rays in front of a much-photographed facade (on the Baptistery's east wall
    they formed a 1-6 m deep fan) and would otherwise hide the very wall they belong to."""
    n = len(points)
    if quality is None or len(quality) != n:
        return np.ones(n, dtype=bool)
    q = np.asarray(quality, float)
    return (q[:, 0] >= params.occluder_min_track) & (q[:, 1] >= params.occluder_min_angle_deg)


def wall_occluders(wall: Wall, obs: WallObservation, points: np.ndarray | None, eligible: np.ndarray | None,
                   params: StyleParams = DEFAULT_PARAMS, seed: int = 0) -> np.ndarray:
    """SfM points that can hide this wall from a camera: more than occluder_margin_m in front of its
    fitted plane (a neighbouring house, a tree, flower boxes, a shed, a porch the massing does not
    model), among the eligible (well triangulated) points."""
    if points is None or not len(points):
        return np.zeros((0, 3))
    pts = np.asarray(points, float)
    rel = pts[:, :2] - wall.a
    u, d = rel @ wall.tangent, rel @ wall.normal
    plane = obs.offset_a + (obs.offset_b - obs.offset_a) * u / max(wall.length, 1e-9)
    dp = d - plane
    sel = (dp > params.occluder_margin_m) & (dp < 150.0) & (u > -60.0) & (u < wall.length + 60.0)
    if eligible is not None:
        sel &= eligible
    cand = pts[sel]
    if len(cand) > params.max_occluders:
        cand = cand[np.random.default_rng(seed).choice(len(cand), params.max_occluders, replace=False)]
    return cand


def occluded_by_points(cam: PosedCamera, occluders: np.ndarray, uv: np.ndarray, depth: np.ndarray,
                       params: StyleParams = DEFAULT_PARAMS) -> np.ndarray:
    """Per sample (image position uv, depth along the optical axis): do occluder points project into its
    cell of a coarse per-camera depth buffer, nearer than the sample? A cell counts when it holds at least
    occluder_min_count of them; the buffer (their minimum depth) is widened by one cell, so sparse points
    on a textureless shed still form a surface."""
    out = np.zeros(len(uv), dtype=bool)
    if not len(occluders):
        return out
    puv, pz = cam.project(occluders)
    ok = (pz > 0.3) & (puv[:, 0] >= 0) & (puv[:, 0] < cam.width) & (puv[:, 1] >= 0) & (puv[:, 1] < cam.height)
    if not ok.any():
        return out
    cell = max(params.occluder_cell_px, max(cam.width, cam.height) / params.occluder_grid)
    gw, gh = int(math.ceil(cam.width / cell)), int(math.ceil(cam.height / cell))
    gx = np.clip((puv[ok, 0] / cell).astype(int), 0, gw - 1)
    gy = np.clip((puv[ok, 1] / cell).astype(int), 0, gh - 1)
    flat = gy * gw + gx
    zb = np.full(gh * gw, np.inf, np.float32)
    np.minimum.at(zb, flat, pz[ok].astype(np.float32))
    zb[np.bincount(flat, minlength=gh * gw) < params.occluder_min_count] = np.inf
    zb = cv2.erode(zb.reshape(gh, gw), np.ones((3, 3), np.uint8))
    sx = np.clip((uv[:, 0] / cell).astype(int), 0, gw - 1)
    sy = np.clip((uv[:, 1] / cell).astype(int), 0, gh - 1)
    return zb[sy, sx] < depth - 0.05


def facade_orthophoto(wall: Wall, obs: WallObservation, massing: Massing, cameras: list[PosedCamera],
                      params: StyleParams = DEFAULT_PARAMS, cache: dict[int, np.ndarray] | None = None,
                      points: np.ndarray | None = None, eligible: np.ndarray | None = None) -> Orthophoto:
    """Project the wall's best photos onto its fitted facade plane and blend them (exposure-matched
    weighted median).

    A camera's sample is dropped where the massing hides the wall from it, and where SfM points standing
    in front of the wall (> occluder_margin_m off the plane, well triangulated) are nearer along its ray:
    an occluder that every camera sees would otherwise survive the median and be painted onto the
    landmark. A pixel is observed only where at least min_views samples agree with the blend (photo
    consistency: parallax ghosts, passers-by and occluders the points missed disagree) and SfM points
    support the plane."""
    w, h, ppm = wall.texture_size(params)
    gh, gw = _vis_grid_shape(wall, params.vis_step_m)
    grid = wall.grid(w, h, obs.offset_a, obs.offset_b)
    vgrid = wall.grid(gw, gh, obs.offset_a, obs.offset_b).reshape(-1, 3)
    tris = massing.triangles(skip_wall=wall.index)
    occluders = wall_occluders(wall, obs, points, eligible, params, seed=wall.index)
    cos_max = math.cos(math.radians(params.max_view_angle_deg))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(0.5 * ppm)) | 1),) * 2)
    k_grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(0.4 * ppm)) | 1),) * 2)
    stack, valid, weights, names, index = [], [], [], [], []
    seen_px = occluded_px = 0
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
        _, vdepth = cam.project(vgrid)
        vok = vdepth > 0.5
        vis = np.zeros(len(vgrid), dtype=bool)
        idx = np.nonzero(vok)[0]
        vis[idx] = ~_ray_hits(cam.center, vgrid[idx], tris)
        vis = cv2.resize(vis.reshape(gh, gw).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
        ok &= vis.reshape(-1)
        # occlusion by things in front of the wall (SfM points)
        occ = occluded_by_points(cam, occluders, uv, depth, params).reshape(h, w) & ok.reshape(h, w)
        if occ.any():
            occ = cv2.morphologyEx(occ.astype(np.uint8), cv2.MORPH_CLOSE, k_close)
            occ = cv2.dilate(occ, k_grow) > 0
        seen_px += int(ok.sum())
        occluded_px += int((occ.reshape(-1) & ok).sum())
        ok &= ~occ.reshape(-1)
        if ok.mean() < 0.02:
            continue
        mx = uv[:, 0].reshape(h, w).astype(np.float32)
        my = uv[:, 1].reshape(h, w).astype(np.float32)
        sample = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        stack.append(sample.astype(np.float32))
        valid.append(ok.reshape(h, w))
        weights.append(score)
        names.append(cam.name)
        index.append(ci)
    stats = {"occluded_sample_fraction": round(occluded_px / max(seen_px, 1), 3), "occluder_points": len(occluders)}
    if not stack:
        return Orthophoto(np.zeros((h, w, 3), np.uint8), np.zeros((h, w), np.int32), np.zeros((h, w), bool), ppm, [],
                          stats=stats)
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
    # photo consistency (on small averages: sub-pixel misalignment at edges is not disagreement)
    kb = max(1, int(round(params.consistency_blur_m * ppm)) | 1)
    blur = np.stack([cv2.blur(s, (kb, kb)) for s in S])
    agree = V & (np.linalg.norm(blur - cv2.blur(rgb.astype(np.float32), (kb, kb))[None], axis=-1) <= params.consistency_rgb)
    n_agree = agree.sum(axis=0)
    consistent = (n_agree >= params.min_views) & (n_agree >= params.min_agreement * count)
    support = cv2.resize(obs.support.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
    base = (count >= params.min_views) & support
    mask = base & consistent
    # keep the mask tidy: no pinholes, no slivers (half a metre)
    k = max(3, int(round(0.5 * ppm)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    mask = (cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) > 0) & base
    stats["inconsistent_fraction"] = round(float((base & ~consistent).sum() / max(base.sum(), 1)), 3)
    out = np.clip(rgb, 0, 255).astype(np.uint8)
    out[count == 0] = 0
    msum = max(int(mask.sum()), 1)
    contrib = [(index[k], round(float((agree[k] & mask).sum() / msum), 3)) for k in range(len(index))]
    contrib = sorted([c for c in contrib if c[1] > 0], key=lambda c: -c[1])
    samples = []
    ys, xs = np.nonzero(mask)
    if len(ys):
        pick = np.random.default_rng(wall.index).choice(len(ys), min(300, len(ys)), replace=False)
        samples = [(int(ys[i]), int(xs[i]), [index[k] for k in range(len(index)) if agree[k, ys[i], xs[i]]]) for i in pick]
    return Orthophoto(out, count, mask, ppm, names, contrib, samples, stats)


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
    return cv2.cvtColor(np.clip(a, 0, 1), cv2.COLOR_RGB2Lab).reshape(np.asarray(rgb).shape).astype(np.float32)


PALETTE_LAB = rgb_to_lab(PALETTE_RGB.astype(np.float32))
_CHROMA = np.hypot(PALETTE_LAB[:, 1], PALETTE_LAB[:, 2])
NEUTRALS = np.array([PALETTE_NAMES.index(n) for n in NEUTRAL_RAMP])   # the light / dark ramp for grey stone
CHROMATIC = np.nonzero(_CHROMA >= 15)[0]


@dataclass
class ColorNorm:
    """Scene-wide colour handling of the orthophotos. The camera's white balance is trusted (gains 1):
    a grey-world balance over the building's own pixels shifts the hue of a single-material building or
    of a scene with a coloured occluder (a red shed turned a cream wall mint). Lightness is not
    stretched either: each material keeps its lightness band. l_lo / l_hi are only reported."""

    gains: np.ndarray                   # (3,) RGB multipliers (kept at 1)
    l_lo: float = 0.0
    l_hi: float = 100.0

    def lab(self, rgb: np.ndarray) -> np.ndarray:
        return rgb_to_lab(np.clip(np.asarray(rgb, np.float32) * self.gains, 0, 255))

    def to_meta(self) -> dict[str, Any]:
        return {"white_balance": "camera", "gains": [round(float(g), 3) for g in self.gains],
                "l_p3": round(self.l_lo, 1), "l_p97": round(self.l_hi, 1)}


IDENTITY_NORM = ColorNorm(np.ones(3, np.float32))


def scene_color_norm(orthos: list[Orthophoto]) -> ColorNorm:
    pix = [o.rgb[o.mask] for o in orthos if o.mask.sum() >= 50]
    if not pix:
        return IDENTITY_NORM
    x = np.vstack(pix).astype(np.float32)
    if len(x) > 200000:
        x = x[np.random.default_rng(0).choice(len(x), 200000, replace=False)]
    lum = rgb_to_lab(x)[:, 0]
    return ColorNorm(np.ones(3, np.float32), float(np.percentile(lum, 3)), float(np.percentile(lum, 97)))


def snap_lab_to_palette(lab: np.ndarray, cheer: float = 1.5, lift: float = 0.12, chromatic_only: bool = False) -> np.ndarray:
    """Palette index for Lab colours (n, 3): near-greys go to the neutral ramp (warm greys, no blue) by
    lightness, coloured ones to the nearest palette colour after a chroma boost; everything is lifted a
    little towards light (cheerful, illustration-like) but keeps its lightness band."""
    lab = np.asarray(lab, np.float32).reshape(-1, 3).copy()
    lab[:, 0] = lab[:, 0] + lift * (100.0 - lab[:, 0])
    chroma = np.hypot(lab[:, 1], lab[:, 2])
    boosted = lab.copy()
    boosted[:, 1:] *= cheer
    cand = CHROMATIC if chromatic_only else np.arange(len(PALETTE))
    d = (0.6 * (boosted[:, None, 0] - PALETTE_LAB[None, cand, 0]) ** 2
         + ((boosted[:, None, 1:] - PALETTE_LAB[None, cand, 1:]) ** 2).sum(-1))
    out = cand[d.argmin(axis=1)]
    grey = (chroma < 6.0) & (not chromatic_only)
    if np.any(grey):
        dn = np.abs(lab[grey, None, 0] - PALETTE_LAB[None, NEUTRALS, 0])
        out[grey] = NEUTRALS[dn.argmin(axis=1)]
    return out


def snap_to_palette(rgb: np.ndarray, cheer: float = 1.5) -> np.ndarray:
    """Palette index of each RGB colour (n, 3) (see snap_lab_to_palette)."""
    return snap_lab_to_palette(rgb_to_lab(np.asarray(rgb, np.float32).reshape(-1, 3)), cheer)


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


def clutter_mask(lab: np.ndarray, mask: np.ndarray, ppm: float) -> np.ndarray:
    """Sky and vegetation inside a facade's observation mask: not wall, so not drawn (they become clay
    like any unobserved pixel). Sky: bright and blue, touching the top half of the observed area.
    Vegetation: green-yellow foliage, or foliage in shadow (dark, strongly textured, still olive: dark
    stone and window recesses are neutral), in blobs of at least half a square metre (one square metre in shadow).

    GENERATIVE-OK: a semantic segmentation model (sky / vegetation / people) on the observed orthophoto
    would replace these colour and texture rules; it only removes pixels from the mask, never adds."""
    raw_L = lab[..., 0]
    sm = cv2.blur(lab.astype(np.float32), (max(3, int(0.25 * ppm) | 1),) * 2)   # decide on 25 cm averages, not on noise
    L, a, b = sm[..., 0], sm[..., 1], sm[..., 2]
    h = mask.shape[0]
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(0.3 * ppm) | 1),) * 2)
    # sky: clearly blue and bright, or pale blue and smooth (no masonry texture); in both cases a blob
    # reaching the upper edge of the observed area (sky is above the building, never inside a wall)
    win = max(3, int(0.5 * ppm) | 1)
    mean = cv2.blur(raw_L.astype(np.float32), (win, win))
    std = np.sqrt(np.maximum(cv2.blur((raw_L * raw_L).astype(np.float32), (win, win)) - mean * mean, 0))
    sky = (((b < -15) & (L > 50)) | ((b < -7) & (L > 60) & (std < 4.0))) & mask
    sky = cv2.morphologyEx(sky.astype(np.uint8), cv2.MORPH_CLOSE, k)
    top = np.argmax(mask, axis=0)                        # first observed row per column
    has = mask.any(axis=0)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(sky, connectivity=8)
    keep = []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < 0.25 * ppm * ppm or stats[c, cv2.CC_STAT_TOP] >= 0.6 * h:
            continue
        ys, xs = np.nonzero(cc == c)
        first = np.full(mask.shape[1], h)
        np.minimum.at(first, xs, ys)
        cols = np.nonzero(has & (first < h))[0]
        if len(cols) and np.mean(first[cols] - top[cols] <= max(2, 0.75 * ppm)) >= 0.3:
            keep.append(c)
    sky = np.isin(cc, keep)
    veg = ((a < -7) & (b > 6) & (b > -a * 0.4)) & mask
    veg = cv2.morphologyEx(veg.astype(np.uint8), cv2.MORPH_CLOSE, k)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(veg, connectivity=8)
    keep = [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= 0.5 * ppm * ppm]
    veg = np.isin(cc, keep)
    # foliage in shadow: dark and busy (leaves), olive or neutral (not the blue-grey of glass); its
    # texture is measured on a 1 m window so that a single dark window (smooth inside) does not qualify
    win2 = max(3, int(1.0 * ppm) | 1)
    mean2 = cv2.blur(raw_L.astype(np.float32), (win2, win2))
    std2 = np.sqrt(np.maximum(cv2.blur((raw_L * raw_L).astype(np.float32), (win2, win2)) - mean2 * mean2, 0))
    shade_veg = (mean2 < 38) & (std2 > 7.0) & (a < -2.0) & (b > 3.0) & mask
    shade_veg = cv2.morphologyEx(shade_veg.astype(np.uint8), cv2.MORPH_OPEN, k)
    shade_veg = cv2.morphologyEx(shade_veg, cv2.MORPH_CLOSE, k)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(shade_veg, connectivity=8)
    keep = [c for c in range(1, n) if stats[c, cv2.CC_STAT_AREA] >= 1.0 * ppm * ppm]
    out = (sky | veg | np.isin(cc, keep)) & mask
    return cv2.dilate(out.astype(np.uint8), k) > 0


@dataclass
class StylizedFacade:
    texture: np.ndarray                 # (h, w, 3) uint8, allowed colours only
    mask: np.ndarray                    # final observation mask (geometric mask minus sky / vegetation)
    base_rgb: tuple[int, int, int]      # the facade's wall colour (inside the mask)
    colors: list[str]                   # palette names used
    openings: list[dict[str, Any]]      # regularized windows / doors (metres, from the top-left corner)
    report: dict[str, Any]
    shapes: list[dict[str, Any]] = field(default_factory=list)   # vector regions (metres)


def tidy_mask(mask: np.ndarray, ppm: float, min_island_m2: float = 1.0, max_hole_m2: float = 0.5) -> np.ndarray:
    """A cleaner observation mask for drawing: observed islands under min_island_m2 are dropped, holes
    under max_hole_m2 (below the model's level of detail: a lamp post, a gap in the consistency test) are
    closed, and the edge is softened by a blur that may only remove pixels."""
    m = mask.astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.nonzero(stats[1:, cv2.CC_STAT_AREA] < min_island_m2 * ppm * ppm)[0] + 1
    m[np.isin(cc, small)] = 0
    inv = (1 - m).astype(np.uint8)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
    h, w = m.shape
    for c in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[c])
        touches = x == 0 or y == 0 or x + bw >= w or y + bh >= h
        if not touches and area < max_hole_m2 * ppm * ppm:
            m[cc == c] = 1
    soft = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(0.8, 0.15 * ppm)) > 0.5
    return (m > 0) & soft


def clay_pattern(h: int, w: int, ppm: float, period_m: float = 1.8) -> np.ndarray:
    """(h, w, 3) the locked-face look baked into a texture: light clay with diagonal hatch stripes
    (the viewer draws the same pattern on untextured locked faces)."""
    yy, xx = np.mgrid[0:h, 0:w]
    stripe = np.mod((xx + yy) / max(ppm, 1e-6) / period_m, 1.0) >= 0.72
    out = np.empty((h, w, 3), np.uint8)
    out[:] = CLAY_RGB
    out[stripe] = CLAY_HATCH_RGB
    return out


def is_clay_pattern(rgb: np.ndarray) -> np.ndarray:
    """Per pixel (..., 3): one of the clay pattern's two colours."""
    rgb = np.asarray(rgb)
    return np.any([np.all(rgb == np.array(c, np.uint8), axis=-1) for c in CLAY_PATTERN], axis=0)


def _rounded_rect(m: np.ndarray, x0: int, y0: int, x1: int, y1: int, r: int, value: int = 1) -> None:
    r = max(0, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    if r == 0:
        cv2.rectangle(m, (x0, y0), (x1, y1), value, -1)
        return
    cv2.rectangle(m, (x0 + r, y0), (x1 - r, y1), value, -1)
    cv2.rectangle(m, (x0, y0 + r), (x1, y1 - r), value, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(m, (cx, cy), r, value, -1)


@dataclass
class FacadeLayers:
    """A facade orthophoto decomposed at the working resolution: what is wall, what is a dark accent
    (locally darker than its surroundings: openings, string courses, stripes, frames, arches) and
    what is a coloured accent (clearly more saturated than the wall: bronze doors, brick bands)."""

    mask: np.ndarray                    # (h, w) final observation mask at texture resolution
    clutter: np.ndarray                 # (h, w) sky / vegetation removed from the geometric mask
    mask_s: np.ndarray                  # (sh, sw) the mask at the working resolution
    lab_s: np.ndarray                   # (sh, sw, 3) normalized, smoothed Lab
    dark_s: np.ndarray                  # (sh, sw) bool
    accent_s: np.ndarray                # (sh, sw) bool
    wppm: float                         # working pixels per metre


def facade_layers(ortho: Orthophoto, norm: ColorNorm, params: StyleParams = DEFAULT_PARAMS) -> FacadeLayers:
    """Split an observed facade into wall / dark / coloured layers at the working resolution.

    GENERATIVE-OK: a material / facade segmentation model on the observed orthophoto could give these
    layers (and window / door / cornice classes) instead of local contrast and chroma thresholds."""
    rgb, ppm = ortho.rgb, ortho.px_per_m
    h, w = ortho.mask.shape
    lab_full = norm.lab(rgb)
    clutter = clutter_mask(lab_full, ortho.mask, ppm)
    mask = ortho.mask & ~clutter
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (max(3, int(round(0.5 * ppm)) | 1),) * 2)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, k_open) > 0   # opening: a subset
    f = min(1.0, params.work_px_per_m / ppm)
    sw, sh = max(4, int(round(w * f))), max(4, int(round(h * f)))
    wppm = ppm * sw / w
    lab_s = cv2.resize(lab_full, (sw, sh), interpolation=cv2.INTER_AREA)
    mask_s = cv2.resize(mask.astype(np.float32), (sw, sh), interpolation=cv2.INTER_AREA) > 0.5
    if mask_s.sum() < 8:
        return FacadeLayers(mask, clutter, mask_s, lab_s, np.zeros_like(mask_s), np.zeros_like(mask_s), wppm)
    lab_s[~mask_s] = np.median(lab_s[mask_s], axis=0)
    lab_s = cv2.bilateralFilter(lab_s.astype(np.float32), 5, 12.0, 2.0)
    L = lab_s[..., 0]
    wgt = mask_s.astype(np.float32)
    k = int(params.local_window_m * wppm) | 1
    local = cv2.blur(np.where(mask_s, L, 0).astype(np.float32), (k, k)) / np.maximum(cv2.blur(wgt, (k, k)), 1e-3)
    dark = (L < local - params.dark_contrast) & mask_s
    dark = cv2.morphologyEx(dark.astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8)) > 0
    chroma = np.hypot(lab_s[..., 1], lab_s[..., 2])
    wall_chroma = float(np.median(chroma[mask_s & ~dark])) if (mask_s & ~dark).any() else 0.0
    accent = (chroma > max(params.accent_min_chroma, wall_chroma + 8.0)) & mask_s
    accent = cv2.morphologyEx(accent.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    # light-blue "accents" on a wall that is not blue are sky the clutter mask missed (seen past a
    # corner or through an arcade): not wall, so out of the mask like the rest of the sky
    wall_b = float(np.median(lab_s[..., 2][mask_s & ~dark])) if (mask_s & ~dark).any() else 0.0
    if wall_b > -6:
        skyish = accent & (lab_s[..., 2] < -10) & (L > 55)
        if skyish.any():
            accent &= ~skyish
            mask_s &= ~skyish
            up = cv2.resize(skyish.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            mask = mask & ~up
            clutter = clutter | (up & ortho.mask)
    dark &= ~accent & mask_s
    for layer in (dark, accent):                         # specks go
        _, cc, stats, _ = cv2.connectedComponentsWithStats(layer.astype(np.uint8), connectivity=8)
        tiny = np.nonzero(stats[1:, cv2.CC_STAT_AREA] < params.min_accent_m2 * wppm * wppm)[0] + 1
        layer[np.isin(cc, tiny)] = False
    return FacadeLayers(mask, clutter, mask_s, lab_s, dark, accent, wppm)


def _material_lab(lab: np.ndarray) -> np.ndarray:
    """A surface's material colour from its observed pixels: the 85th percentile of lightness (shading
    only ever darkens; photos of a bright facade against the sky are underexposed) with the median hue."""
    lab = np.asarray(lab, np.float32).reshape(-1, 3)
    return np.array([np.percentile(lab[:, 0], 85), np.median(lab[:, 1]), np.median(lab[:, 2])], np.float32)


@dataclass
class ColorPlan:
    """The landmark's few material colours, shared by every facade so that walls agree: the wall
    colour, the dark-accent colour and up to two coloured accents (palette indices), each the
    palette snap of the median observed colour of that layer pooled over all facades."""

    wall: int
    dark: int
    accents: list[int]
    wall_lab: np.ndarray

    def to_meta(self) -> dict[str, Any]:
        return {"wall": PALETTE_NAMES[self.wall], "dark": PALETTE_NAMES[self.dark],
                "accents": [PALETTE_NAMES[a] for a in self.accents],
                "wall_lab": [round(float(v), 1) for v in self.wall_lab]}


def plan_colors(layers: list[FacadeLayers], params: StyleParams = DEFAULT_PARAMS) -> ColorPlan:
    walls, darks, accents = [], [], []
    for ly in layers:
        if ly.mask_s.sum() < 8:
            continue
        walls.append(ly.lab_s[ly.mask_s & ~ly.dark_s & ~ly.accent_s])
        darks.append(ly.lab_s[ly.dark_s])
        accents.append(ly.lab_s[ly.accent_s])
    wall_lab = _material_lab(np.vstack(walls)) if walls else np.array([80.0, 0.0, 5.0], np.float32)
    dark_pix = np.vstack(darks) if darks else np.zeros((0, 3), np.float32)
    dark_lab = np.median(dark_pix, axis=0) if len(dark_pix) else wall_lab - np.array([35.0, 0, 0])
    wall = int(snap_lab_to_palette(wall_lab[None], lift=params.wall_lift)[0])
    dark = int(snap_lab_to_palette(dark_lab[None], lift=0.15)[0])
    if PALETTE_LAB[wall, 0] - PALETTE_LAB[dark, 0] < params.min_dark_delta:   # keep the accents readable
        cand = [i for i in range(len(PALETTE)) if PALETTE_LAB[wall, 0] - PALETTE_LAB[i, 0] >= params.min_dark_delta]
        if cand:
            target = dark_lab.copy()
            dark = int(min(cand, key=lambda i: (PALETTE_LAB[i, 1] - target[1] * 1.5) ** 2
                           + (PALETTE_LAB[i, 2] - target[2] * 1.5) ** 2
                           + 0.3 * (PALETTE_LAB[i, 0] - min(target[0], PALETTE_LAB[wall, 0] - params.min_dark_delta)) ** 2))
    acc: list[int] = []
    pix = np.vstack(accents) if accents else np.zeros((0, 3), np.float32)
    if len(pix) >= 50:
        cents = kmeans(pix, 2)
        share = np.bincount(((pix[:, None] - cents[None]) ** 2).sum(-1).argmin(1), minlength=len(cents)) / len(pix)
        for c, s in sorted(zip(cents, share), key=lambda t: -t[1]):
            if s >= 0.2:
                p = int(snap_lab_to_palette(c[None])[0])
                if p not in acc and p not in (wall, dark):
                    acc.append(p)
    return ColorPlan(wall, dark, acc, wall_lab.astype(np.float32))


def _smooth_upsample(mask_s: np.ndarray, size: tuple[int, int], sigma_px: float = 0.7) -> np.ndarray:
    """A low-resolution binary mask -> smooth-edged mask at (w, h): blur, bilinear up, threshold."""
    m = cv2.GaussianBlur(mask_s.astype(np.float32), (0, 0), sigma_px) if sigma_px > 0 else mask_s.astype(np.float32)
    return cv2.resize(m, size, interpolation=cv2.INTER_LINEAR) > 0.5


def stylize_facade(ortho: Orthophoto, params: StyleParams = DEFAULT_PARAMS, ground_row: bool = True,
                   seed: int = 0, norm: ColorNorm | None = None, plan: ColorPlan | None = None,
                   layers: FacadeLayers | None = None, size_scale: float = 1.0) -> StylizedFacade:
    """Rectified orthophoto -> flat-colour illustration of the observed part of a facade.

    The facade is posterized into three layers at a coarse working resolution (facade_layers):
    wall, dark accents (locally darker: openings, string courses, stripes, frames) and coloured
    accents. The wall and accent colours come from the landmark's colour plan (palette snaps of the
    pooled observed colours); a compact coloured accent that the plan has no colour for (a gilded door)
    keeps its own palette snap. Only regular structure is drawn: dark compact components of window size
    become clean rounded rectangles / arches aligned in rows (with an ink frame), long thin ones straight
    bands or pilasters, and the remaining dark shapes only when they are large (min_free_shape_m2 x
    size_scale) and compact (solidity >= min_free_solidity), as simplified polygons; crumbs and ragged
    stains are dropped. Outside the observation mask the texture is the clay pattern of locked faces,
    so no colour, not even the wall colour, bleeds onto what nobody photographed.

    GENERATIVE-OK: a facade-parsing / segmentation model (e.g. SAM 2) could find window / door / trim
    regions on the observed orthophoto more reliably than local contrast, and an image-to-image
    stylizer could restyle the observed pixels. Neither may fill pixels outside the observation mask.
    """
    t0 = time.perf_counter()
    norm = norm or IDENTITY_NORM
    ppm = ortho.px_per_m
    h, w = ortho.mask.shape
    ly = layers or facade_layers(ortho, norm, params)
    mask = ly.mask
    if mask.any():
        mask = tidy_mask(mask, ppm)
    clay = clay_pattern(h, w, ppm)
    if mask.sum() < max(20, 0.5 * ppm * ppm) or ly.mask_s.sum() < 8:
        return StylizedFacade(clay, np.zeros_like(mask), CLAY_RGB, [], [],
                              {"empty": True, "clutter_fraction": round(float(ly.clutter.mean()), 3)})
    plan = plan or plan_colors([ly], params)
    wppm = ly.wppm
    # this facade's own wall colour when it clearly differs from the landmark's (a brick annex)
    own = ly.lab_s[ly.mask_s & ~ly.dark_s & ~ly.accent_s]
    wall_idx = plan.wall
    if len(own) >= 20:
        own_lab = _material_lab(own)
        bluish = own_lab[2] < -6 and own_lab[0] > 55             # probably sky the clutter mask missed
        if float(np.linalg.norm(own_lab - plan.wall_lab)) > params.own_wall_delta_e and not bluish \
                and ly.mask_s.mean() >= 0.3:
            wall_idx = int(snap_lab_to_palette(own_lab[None], lift=params.wall_lift)[0])
    base = tuple(int(v) for v in PALETTE_RGB[wall_idx])
    dark_rgb = tuple(int(v) for v in PALETTE_RGB[plan.dark])

    # ---- regularized structure from the dark layer
    labels = np.zeros(ly.mask_s.shape, np.int32)          # openings only need a label for their colour
    openings = _find_openings(ly.dark_s, labels, wppm, params, ground_row)
    dark = ly.dark_s.copy()
    for op in openings:
        dark &= ~op.pop("_mask")
        op["label"] = 0
    shapes: list[dict[str, Any]] = []
    n, cc, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    for c in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[c])
        wm, hm, fill = bw / wppm, bh / wppm, area / max(bw * bh, 1)
        if fill >= params.band_min_fill and wm >= params.band_min_len_m and hm <= params.band_max_thick_m:
            rows = (cc[y:y + bh, x:x + bw] == c).mean(axis=1) >= 0.4
            ys = np.nonzero(rows)[0]
            y0, y1 = (y + ys[0], y + ys[-1] + 1) if len(ys) else (y, y + bh)
            shapes.append({"kind": "hband", "label": 0, "x": x / wppm, "y": y0 / wppm, "w": wm,
                           "h": max(1, y1 - y0) / wppm, "area_m2": round(area / wppm ** 2, 2)})
            dark[cc == c] = False
        elif fill >= params.band_min_fill and hm >= params.band_min_len_m and wm <= params.band_max_thick_m:
            cols = (cc[y:y + bh, x:x + bw] == c).mean(axis=0) >= 0.4
            xs = np.nonzero(cols)[0]
            x0, x1 = (x + xs[0], x + xs[-1] + 1) if len(xs) else (x, x + bw)
            shapes.append({"kind": "vband", "label": 0, "x": x0 / wppm, "y": y / wppm, "w": max(1, x1 - x0) / wppm,
                           "h": hm, "area_m2": round(area / wppm ** 2, 2)})
            dark[cc == c] = False
    shapes = _merge_bands(shapes, params)

    # ---- draw at texture resolution: wall, accents, compact dark shapes, bands, openings
    tex = np.empty((h, w, 3), np.uint8)
    tex[:] = base
    used = {PALETTE_NAMES[wall_idx]}
    acc_drawn = 0
    if ly.accent_s.any():
        n, cc, stats, _ = cv2.connectedComponentsWithStats(ly.accent_s.astype(np.uint8), connectivity=8)
        plan_lab = PALETTE_LAB[plan.accents] if plan.accents else np.zeros((0, 3), np.float32)
        for c in range(1, n):
            comp = cc == c
            area = stats[c, cv2.CC_STAT_AREA] / wppm ** 2
            lab_c = np.median(ly.lab_s[comp], axis=0)
            colour_idx = None
            if len(plan_lab):
                dist = np.linalg.norm(plan_lab - lab_c[None] * np.array([1.0, 1.5, 1.5], np.float32), axis=1)
                if dist.min() <= 30.0:
                    colour_idx = plan.accents[int(dist.argmin())]
            if colour_idx is None and area >= params.min_own_accent_m2:
                colour_idx = int(snap_lab_to_palette(lab_c[None], lift=params.wall_lift, chromatic_only=True)[0])
            if colour_idx is None or colour_idx == wall_idx or area < params.min_own_accent_m2:
                continue
            colour = tuple(int(v) for v in PALETTE_RGB[colour_idx])
            m = _smooth_upsample(comp, (w, h)) & mask
            if not m.any():
                continue
            tex[m] = colour
            _outline(tex, m, shade(colour), params, ppm)
            used.add(PALETTE_NAMES[colour_idx])
            acc_drawn += 1
    # remaining dark shapes: only large, compact ones, drawn as simplified polygons
    min_free = params.min_free_shape_m2 * max(1.0, size_scale)
    n, cc, stats, _ = cv2.connectedComponentsWithStats(dark.astype(np.uint8), connectivity=8)
    free_count = dropped = 0
    free = np.zeros((h, w), np.uint8)
    for c in range(1, n):
        area_px = stats[c, cv2.CC_STAT_AREA]
        if area_px < min_free * wppm * wppm:
            dropped += 1
            continue
        comp = _smooth_upsample(cc == c, (w, h)).astype(np.uint8)
        contours, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        hull_area = cv2.contourArea(cv2.convexHull(cnt))
        if cv2.contourArea(cnt) < params.min_free_solidity * max(hull_area, 1e-9):
            dropped += 1
            continue
        poly = cv2.approxPolyDP(cnt, max(1.0, 0.12 * ppm), True)
        cv2.fillPoly(free, [poly], 1)
        free_count += 1
    free = free > 0
    tex[free] = dark_rgb
    lines = facade_lines(norm.lab(ortho.rgb)[..., 0], mask, ppm, params)
    for ln in lines:
        x0, y0 = int(round(ln["x"] * ppm)), int(round(ln["y"] * ppm))
        x1, y1 = int(round((ln["x"] + ln["w"]) * ppm)) - 1, int(round((ln["y"] + ln["h"]) * ppm)) - 1
        cv2.rectangle(tex, (x0, y0), (max(x0, x1), max(y0, y1)), dark_rgb, -1)
        ln["color"] = PALETTE_NAMES[plan.dark]
        for key in ("x", "y", "w", "h"):
            ln[key] = round(float(ln[key]), 2)
    shapes += lines
    for shp in shapes:
        if shp["kind"] in ("hline", "vline"):
            continue
        m = np.zeros((h, w), np.uint8)
        x0, y0 = int(round(shp["x"] * ppm)), int(round(shp["y"] * ppm))
        x1, y1 = int(round((shp["x"] + shp["w"]) * ppm)) - 1, int(round((shp["y"] + shp["h"]) * ppm)) - 1
        _rounded_rect(m, x0, y0, x1, y1, int(0.5 * min(x1 - x0, y1 - y0)))
        tex[m > 0] = dark_rgb
        shp.pop("label")
        shp["color"] = PALETTE_NAMES[plan.dark]
        for key in ("x", "y", "w", "h"):
            shp[key] = round(float(shp[key]), 2)
    for op in openings:
        op.pop("label")
        op["rgb"] = dark_rgb
        op["color"] = PALETTE_NAMES[plan.dark]
        _draw_opening(tex, op, ppm, params)
        del op["rgb"]
        for key in ("x", "y", "w", "h"):
            op[key] = round(float(op[key]), 2)
    tex[~mask] = clay[~mask]                              # honesty: nothing outside the observation mask
    if free.any() or shapes or openings:
        used.add(PALETTE_NAMES[plan.dark])
    kinds = [s["kind"] for s in shapes]
    report = {"colors": sorted(used), "base": PALETTE_NAMES[wall_idx], "num_openings": len(openings),
              "num_shapes": len(shapes), "shape_kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
              "free_shapes": free_count, "free_shapes_dropped": dropped, "accent_regions": acc_drawn,
              "dark_fraction": round(float(ly.dark_s[ly.mask_s].mean()), 3),
              "accent_fraction": round(float(ly.accent_s[ly.mask_s].mean()), 3),
              "mask_fraction": round(float(mask.mean()), 3),
              "closed_holes_m2": round(float((mask & ~ly.mask).sum()) / ppm ** 2, 2),
              "clutter_fraction": round(float(ly.clutter.mean()), 3), "seconds": round(time.perf_counter() - t0, 3)}
    return StylizedFacade(tex, mask, base, sorted(used), openings, report, shapes)


def facade_lines(L: np.ndarray, mask: np.ndarray, ppm: float, params: StyleParams = DEFAULT_PARAMS
                  ) -> list[dict[str, Any]]:
    """Thin straight dark lines of an observed facade (inlay frames, string courses, pilaster edges,
    stripes): a morphological black-hat (darker than a closing over line_window_m), opened with long
    horizontal / vertical kernels, grouped into rows / columns and regularized into straight segments of
    one thickness each (metres from the top-left corner). Only pixels inside the mask take part.

    GENERATIVE-OK: a facade-parsing model could return these structure lines directly (observed pixels only)."""
    if mask.sum() < 50:
        return []
    Lf = L.astype(np.float32).copy()
    Lf[~mask] = float(np.median(L[mask]))
    k = max(3, int(round(params.line_window_m * ppm)) | 1)
    bh = cv2.morphologyEx(Lf, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    thin = ((bh > params.line_contrast) & mask).astype(np.uint8)
    n_len = max(3, int(round(params.line_min_len_m * ppm)))
    out: list[dict[str, Any]] = []
    for kind, kernel in (("hline", (n_len, 1)), ("vline", (1, n_len))):
        m = cv2.morphologyEx(thin, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, kernel))
        n, _cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        segs = []
        for c in range(1, n):
            x, y, bw, bh_, area = (int(v) for v in stats[c])
            along, across = (bw, bh_) if kind == "hline" else (bh_, bw)
            if along < n_len or across > 0.5 * ppm:
                continue
            thick = area / max(along, 1)
            centre = (y + bh_ / 2) if kind == "hline" else (x + bw / 2)
            start = x if kind == "hline" else y
            segs.append([centre / ppm, start / ppm, (start + along) / ppm, thick / ppm])
        segs.sort(key=lambda s: (s[0], s[1]))
        merged: list[list[float]] = []
        for s in segs:
            for o in merged:
                if abs(o[0] - s[0]) <= params.line_row_tol_m and s[1] - o[2] <= params.line_gap_m and o[1] - s[2] <= params.line_gap_m:
                    wsum = (o[2] - o[1]) + (s[2] - s[1])
                    o[0] = (o[0] * (o[2] - o[1]) + s[0] * (s[2] - s[1])) / wsum
                    o[3] = (o[3] * (o[2] - o[1]) + s[3] * (s[2] - s[1])) / wsum
                    o[1], o[2] = min(o[1], s[1]), max(o[2], s[2])
                    break
            else:
                merged.append(list(s))
        for c_, a0, a1, t in merged:
            t = float(np.clip(t, 0.1, 0.3))
            if kind == "hline":
                out.append({"kind": kind, "x": a0, "y": c_ - t / 2, "w": a1 - a0, "h": t})
            else:
                out.append({"kind": kind, "x": c_ - t / 2, "y": a0, "w": t, "h": a1 - a0})
    return out


def _outline(tex: np.ndarray, m: np.ndarray, colour, params: StyleParams, ppm: float) -> None:
    if not params.region_outline or not m.any():
        return
    contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(tex, contours, -1, colour, max(1, int(round(params.region_outline_m * ppm))), cv2.LINE_8)


def _merge_bands(shapes: list[dict[str, Any]], params: StyleParams) -> list[dict[str, Any]]:
    """Horizontal bands in the same row (vertical overlap >= 50 %) with gaps under band_gap_m join
    into one straight band (a cornice broken by a lamp post is one cornice)."""
    bands = sorted((s for s in shapes if s["kind"] == "hband"), key=lambda s: (s["y"], s["x"]))
    rest = [s for s in shapes if s["kind"] != "hband"]
    out: list[dict[str, Any]] = []
    for s in bands:
        for o in out:
            overlap = min(o["y"] + o["h"], s["y"] + s["h"]) - max(o["y"], s["y"])
            gap = max(o["x"], s["x"]) - min(o["x"] + o["w"], s["x"] + s["w"])
            if overlap >= 0.5 * min(o["h"], s["h"]) and gap <= params.band_gap_m:
                x0, x1 = min(o["x"], s["x"]), max(o["x"] + o["w"], s["x"] + s["w"])
                ya = (o["y"] * o["w"] + s["y"] * s["w"]) / (o["w"] + s["w"])
                ha = (o["h"] * o["w"] + s["h"] * s["w"]) / (o["w"] + s["w"])
                o.update(x=x0, w=x1 - x0, y=ya, h=ha, area_m2=round(o["area_m2"] + s["area_m2"], 2))
                break
        else:
            out.append(dict(s))
    return rest + out


def _find_openings(dark: np.ndarray, labels: np.ndarray, ppm: float, params: StyleParams,
                   ground_row: bool) -> list[dict[str, Any]]:
    h = dark.shape[0]
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


def _draw_opening(tex: np.ndarray, op: dict[str, Any], ppm: float, params: StyleParams = DEFAULT_PARAMS) -> None:
    """A clean window / door: rounded rectangle (arched top when the region is arched), ink frame."""
    pal_colour = op.get("rgb")
    x0, y0 = int(round(op["x"] * ppm)), int(round(op["y"] * ppm))
    x1, y1 = int(round((op["x"] + op["w"]) * ppm)) - 1, int(round((op["y"] + op["h"]) * ppm)) - 1
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    colour = pal_colour if pal_colour is not None else INK_RGB
    m = np.zeros(tex.shape[:2], np.uint8)
    if op["arched"]:
        r = (x1 - x0) // 2
        cv2.rectangle(m, (x0, y0 + r), (x1, y1), 1, -1)
        cv2.ellipse(m, ((x0 + x1) // 2, y0 + r), (r, r), 0, 180, 360, 1, -1)
    else:
        _rounded_rect(m, x0, y0, x1, y1, max(1, int(0.18 * min(x1 - x0, y1 - y0))))
    tex[m > 0] = colour
    frame = max(1, int(round(params.opening_frame_m * ppm)))
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(tex, contours, -1, INK_RGB, frame, cv2.LINE_8)


# ---------------------------------------------------------------- roof colours


@dataclass
class RoofPatch:
    """An observed patch of a roof face (or, rgb None, an unobserved hole inside one), drawn as a decal."""

    prism: int
    face: int
    polygon: np.ndarray                 # (k, 2) CCW, ENU x / y
    rgb: tuple[int, int, int] | None
    coverage: float                     # share of its cells with roof points
    area_m2: float


@dataclass
class RoofLook:
    coverage: dict = field(default_factory=dict)      # (prism, face) -> share of the face's cells with roof points
    area_m2: dict = field(default_factory=dict)       # (prism, face) -> area of the face (grid cells)
    face_rgb: dict = field(default_factory=dict)      # (prism, face) -> colour when observed as a whole, else None
    patches: list = field(default_factory=list)       # RoofPatch


def _mask_polygons(mask: np.ndarray, clip: np.ndarray, cell: float, origin: np.ndarray,
                   min_area_m2: float) -> list[tuple[np.ndarray, list[np.ndarray]]]:
    """Smooth contours of a grid mask (row 0 = south), kept inside `clip`: [(outer CCW, [holes CCW])]."""
    up = 4
    size = (mask.shape[1] * up, mask.shape[0] * up)
    m = cv2.resize(mask.astype(np.uint8) * 255, size, interpolation=cv2.INTER_LINEAR)
    m = (cv2.GaussianBlur(m, (0, 0), up * 0.6) > 127) & (cv2.resize(clip.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0)
    contours, hier = cv2.findContours(m.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hier is None:
        return []

    def to_xy(c):
        c = cv2.approxPolyDP(c, 0.3 * up, True)[:, 0, :].astype(float)
        return np.stack([origin[0] + (c[:, 0] + 0.5) / up * cell, origin[1] + (c[:, 1] + 0.5) / up * cell], 1)

    out = []
    for i, c in enumerate(contours):
        if hier[0][i][3] != -1:
            continue                                     # a hole: collected with its parent
        outer = to_xy(c)
        if len(outer) < 3 or abs(signed_area(outer)) < min_area_m2 or not ring_is_simple(outer):
            continue
        holes = []
        j = hier[0][i][2]
        while j != -1:
            hxy = to_xy(contours[j])
            if len(hxy) >= 3 and abs(signed_area(hxy)) >= min_area_m2 and ring_is_simple(hxy):
                holes.append(ensure_ccw(hxy))
            j = hier[0][j][0]
        out.append((ensure_ccw(outer), holes))
    return out


def roof_appearance(massing: Massing, points: np.ndarray | None, colors: np.ndarray | None,
                    norm: ColorNorm | None = None, params: StyleParams = DEFAULT_PARAMS) -> RoofLook:
    """Roof colours from the points, honest about coverage. A cap face (a flat cap, or one pyramid face)
    whose cells carry roof points on at least roof_full_coverage of its area is coloured as a whole;
    otherwise only its observed patches are (smooth outlines of the covered cells, each patch with
    >= roof_face_min_coverage point support and >= roof_patch_min_m2), and the rest stays clay. The
    points' colours are snapped like a wall material, lifted towards light."""
    norm = norm or IDENTITY_NORM
    look = RoofLook()
    pts = None if points is None or colors is None or not len(points) else np.asarray(points, float)
    cols = None if pts is None else np.asarray(colors)
    for pr in massing.prisms:
        g = massing.roof_grids.get(pr.index)
        if g is None:
            g = roof_grid(pr, [c for c in massing.prisms if c.parent == pr.index], pts, params)
        idx = np.zeros(0, int)
        cell_x = cell_y = np.zeros(0, int)
        if pts is not None:
            ix, iy, ok = g.cell_of(pts[:, :2])
            ok &= np.where(ok, g.region[np.clip(iy, 0, g.region.shape[0] - 1), np.clip(ix, 0, g.region.shape[1] - 1)], False)
            idx = np.nonzero(ok)[0]
            idx = idx[roof_point_mask(pr, pts[idx], params)]
            cell_x, cell_y = ix[idx], iy[idx]

        def colour_of(sel: np.ndarray) -> tuple[int, int, int] | None:
            if len(sel) < 20:
                return None
            lab = _material_lab(norm.lab(cols[sel].astype(np.float32)))
            return tuple(int(v) for v in PALETTE_RGB[snap_lab_to_palette(lab[None], lift=params.roof_lift)[0]])

        for f in range(pr.num_faces()):
            key = (pr.index, f)
            cov = g.coverage(f)
            look.coverage[key] = cov
            look.area_m2[key] = g.area_m2(f)
            look.face_rgb[key] = None
            if pts is None or cov <= 0:
                continue
            if pr.cap == "pyramid" and pr.face_sloped and not pr.face_sloped[f]:
                continue                                  # a neutral flat face is not where the roof is: no patches on it
            on_face = g.face[cell_y, cell_x] == f
            if cov >= params.roof_full_coverage:
                look.face_rgb[key] = colour_of(idx[on_face])
                if look.face_rgb[key] is not None:
                    continue
            clip = g.region & (g.face == f)
            closed = cv2.morphologyEx((g.covered & clip).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) > 0
            for outer, holes in _mask_polygons(closed & clip, clip, g.cell, g.origin, params.roof_patch_min_m2):
                c = g.centres().reshape(-1, 2)
                inside = points_in_polygon(c, outer)
                for hh in holes:
                    inside &= ~points_in_polygon(c, hh)
                inside = inside.reshape(g.region.shape) & clip
                if not inside.any():
                    continue
                pcov = float(g.covered[inside].mean())
                if pcov < params.roof_face_min_coverage:
                    continue
                sel = idx[on_face & inside[cell_y, cell_x]]
                rgb = colour_of(sel)
                if rgb is None:
                    continue
                look.patches.append(RoofPatch(pr.index, f, outer, rgb, pcov, abs(signed_area(outer))))
                for hh in holes:
                    look.patches.append(RoofPatch(pr.index, f, hh, None, 0.0, abs(signed_area(hh))))
    return look


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


def pack_atlases(textures: list[np.ndarray], max_size: int = 2048, gutter: int = 4
                 ) -> tuple[list[np.ndarray], list[tuple[int, float, float, float, float]]]:
    """Shelf-pack textures into as few max_size x max_size atlases as needed (edge pixels replicated
    into a gutter so that filtering does not bleed). Returns the atlases and, per texture,
    (atlas, u0, v0, du, dv) to map its UVs: u' = u0 + u * du, v' = v0 + v * dv."""
    order = sorted(range(len(textures)), key=lambda i: -textures[i].shape[0])
    shelves: list[list[Any]] = []            # per atlas: list of [y, height, x_used]
    place: dict[int, tuple[int, int, int]] = {}
    heights: list[int] = []
    for i in order:
        h, w = textures[i].shape[:2]
        H, W = h + 2 * gutter, w + 2 * gutter
        if H > max_size or W > max_size:
            raise ValueError("texture larger than the atlas")
        done = False
        for a, sh in enumerate(shelves):
            for shelf in sh:
                if shelf[1] >= H and shelf[2] + W <= max_size:
                    place[i] = (a, shelf[2], shelf[0])
                    shelf[2] += W
                    done = True
                    break
            if not done and heights[a] + H <= max_size:
                sh.append([heights[a], H, W])
                place[i] = (a, 0, heights[a])
                heights[a] += H
                done = True
            if done:
                break
        if not done:
            shelves.append([[0, H, W]])
            heights.append(H)
            place[i] = (len(shelves) - 1, 0, 0)
    atlases = []
    for a in range(len(shelves)):
        used_w = max(sh[2] for sh in shelves[a])
        atlases.append(np.zeros((heights[a], used_w, 3), np.uint8))
    out = []
    for i, tex in enumerate(textures):
        a, x, y = place[i]
        h, w = tex.shape[:2]
        atlases[a][y:y + h + 2 * gutter, x:x + w + 2 * gutter] = cv2.copyMakeBorder(
            np.ascontiguousarray(tex), gutter, gutter, gutter, gutter, cv2.BORDER_REPLICATE)
        H, W = atlases[a].shape[:2]
        out.append((a, (x + gutter) / W, (y + gutter) / H, w / W, h / H))
    return atlases, out


def write_glb(parts: list[Part], scene_extras: dict[str, Any] | None = None, atlas_size: int = 2048) -> bytes:
    """Minimal glTF 2.0 binary: one node + mesh per part (so every face group keeps its own
    provenance in extras); facade textures packed into shared atlases (PNG), one material per atlas
    and per flat colour."""
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
        acc = {"bufferView": view(arr.tobytes(), target), "componentType": comp, "count": len(arr), "type": kind}
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        gltf["accessors"].append(acc)
        return len(gltf["accessors"]) - 1

    textured = [i for i, p in enumerate(parts) if p.texture is not None and p.uvs is not None]
    atlases, slots = pack_atlases([parts[i].texture for i in textured], atlas_size) if textured else ([], [])
    slot_of = dict(zip(textured, slots))
    atlas_material: dict[int, int] = {}
    colour_material: dict[tuple[int, int, int], int] = {}
    if atlases:
        gltf["samplers"] = [{"magFilter": 9729, "minFilter": 9729, "wrapS": 33071, "wrapT": 33071}]   # LINEAR, no mipmaps: no atlas bleeding
        gltf["images"], gltf["textures"] = [], []
        for a, img in enumerate(atlases):
            gltf["images"].append({"bufferView": view(_png_bytes(img)), "mimeType": "image/png"})
            gltf["textures"].append({"sampler": 0, "source": a})
            gltf["materials"].append({"name": f"facades-{a}", "doubleSided": False, "pbrMetallicRoughness": {
                "baseColorTexture": {"index": a}, "metallicFactor": 0.0, "roughnessFactor": 1.0}})
            atlas_material[a] = len(gltf["materials"]) - 1

    for pi, part in enumerate(parts):
        pos = _enu_to_gltf(part.positions)
        nrm = _enu_to_gltf(part.normals)
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        idx = np.asarray(part.indices).reshape(-1)
        idx = idx.astype(np.uint16 if len(pos) < 65536 else np.uint32)
        attrs = {"POSITION": accessor(pos, "VEC3", 34962, True), "NORMAL": accessor(nrm.astype(np.float32), "VEC3", 34962)}
        if pi in slot_of:
            a, u0, v0, du, dv = slot_of[pi]
            uv = np.asarray(part.uvs, np.float64)
            uv = np.stack([u0 + uv[:, 0] * du, v0 + uv[:, 1] * dv], 1).astype(np.float32)
            attrs["TEXCOORD_0"] = accessor(uv, "VEC2", 34962)
            material = atlas_material[a]
        else:
            rgb = tuple(int(v) for v in part.rgb)
            if rgb not in colour_material:
                gltf["materials"].append({"name": "flat-{:02x}{:02x}{:02x}".format(*rgb), "doubleSided": False,
                                          "pbrMetallicRoughness": {
                                              "baseColorFactor": [_srgb_to_linear(c / 255.0) for c in rgb] + [1.0],
                                              "metallicFactor": 0.0, "roughnessFactor": 1.0}})
                colour_material[rgb] = len(gltf["materials"]) - 1
            material = colour_material[rgb]
        prim = {"attributes": attrs, "indices": accessor(idx, "SCALAR", 34963), "material": material}
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
    footprint: np.ndarray | None        # (k, 2) ENU, OSM outer ring; None / unusable: outline traced from the points
    points: np.ndarray | None           # (n, 3) ENU, SfM points
    colors: np.ndarray | None           # (n, 3) uint8
    cameras: list[PosedCamera]
    facts: HeightFacts = field(default_factory=HeightFacts)
    attribution: list[str] = field(default_factory=list)
    snap: bool = True                   # move points + cameras onto the footprint first (ARCHITECTURE 3.5.7)
    holes: tuple = ()                   # (k, 2) ENU courtyard rings of the OSM outline (inner rings)
    snap_override: Any = None           # FootprintSnap stored with an earlier model version: reused, not re-estimated
    point_quality: Any = None           # (n, 2) per point: track length, triangulation angle (deg); optional


@dataclass
class StylizedModel:
    glb: bytes
    summary: dict[str, Any]
    massing: Massing
    observations: list[WallObservation]
    orthophotos: dict[int, Orthophoto]
    facades: dict[int, StylizedFacade]
    parts: list[Part]
    roof: RoofLook | None = None


def _provenance(outline: str, height: str, appearance: str) -> dict[str, str]:
    return {"outline": outline, "height": height, "appearance": appearance}


def _quad(pa: np.ndarray, pb: np.ndarray, z0: float, z1: float) -> np.ndarray:
    return np.array([[*pa, z0], [*pb, z0], [*pb, z1], [*pa, z1]])


def _tri_area(pos: np.ndarray, idx: np.ndarray) -> float:
    p = np.asarray(pos, float)[np.asarray(idx).reshape(-1, 3)]
    return float(0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1).sum())


def wall_pieces(wall: Wall, ua: float, ub: float, fac: StylizedFacade | None,
                params: StyleParams = DEFAULT_PARAMS) -> list[tuple[float, float, bool]]:
    """Split a wall (u from ua to ub along it) into vertical strips: the strip spanning the observed
    columns of its facade texture, and clay strips beyond it (each at least strip_min_m wide)."""
    if fac is None or not fac.mask.any():
        return [(ua, ub, False)]
    w = fac.mask.shape[1]
    cols = np.nonzero(fac.mask.any(axis=0))[0]
    o0 = max(ua, cols[0] / w * wall.length)
    o1 = min(ub, (cols[-1] + 1) / w * wall.length)
    if o1 - o0 < 0.25:
        return [(ua, ub, False)]
    if o0 - ua < params.strip_min_m:
        o0 = ua
    if ub - o1 < params.strip_min_m:
        o1 = ub
    out = []
    if o0 > ua + 1e-6:
        out.append((ua, o0, False))
    out.append((o0, o1, True))
    if ub > o1 + 1e-6:
        out.append((o1, ub, False))
    return out


def build_parts(massing: Massing, observed: dict[int, StylizedFacade], roof: RoofLook | None = None,
                params: StyleParams = DEFAULT_PARAMS, observations: list[WallObservation] | None = None,
                photo_refs: dict[int, list[dict[str, Any]]] | None = None) -> list[Part]:
    """Bevelled toy mesh. Parts (one GLB node each, each with its own provenance):
    - wall strips: the strip over a facade's observed columns (textured: stylized inside the observation
      mask, the clay pattern outside it) and clay strips beyond it;
    - bevelled corners (wall colour between two observed strips, else clay);
    - cap faces: coloured as a whole only when observed as a whole, else clay; observed roof patches
      (decals a few cm above the face) and clay holes inside them; vertical clay cuts where a pyramid
      face that keeps its fitted slope meets a neutral flat one.
    """
    roof = roof or RoofLook()
    photo_refs = photo_refs or {}
    obs_by_wall = {o.wall: o for o in observations or []}
    parts: list[Part] = []
    segs = BEVEL_SEGMENTS if params.bevel_m > 0 else 0
    for pr in massing.prisms:
        radius = bevel_radius(pr.polygon, params)
        rings_b = [bevel_ring(r, radius, BEVEL_SEGMENTS)[0] for r in pr.rings]
        walls = {(w.ring, w.edge): w for w in massing.walls if w.prism == pr.index}
        for r, ring in enumerate(pr.rings):
            rb = rings_b[r]
            nk = len(ring)
            ends: dict[int, tuple[bool, bool, Any]] = {}       # edge -> (observed at start, observed at end, colour)
            for k in range(nk):
                wall = walls[(r, k)]
                if segs:
                    i0, i1 = k * (segs + 1) + segs, ((k + 1) % nk) * (segs + 1)
                else:
                    i0, i1 = k, (k + 1) % nk
                pa, pb = rb[i0], rb[i1]
                ua = float((pa - wall.a) @ wall.tangent)
                ub = float((pb - wall.a) @ wall.tangent)
                fac = observed.get(wall.index)
                pieces = wall_pieces(wall, ua, ub, fac, params)
                ends[k] = (pieces[0][2], pieces[-1][2], fac.base_rgb if fac else None)
                nrm = np.repeat(np.r_[wall.normal, 0.0][None], 4, axis=0)
                ob = obs_by_wall.get(wall.index)
                for pi, (u0, u1, is_obs) in enumerate(pieces):
                    q0, q1 = wall.a + u0 * wall.tangent, wall.a + u1 * wall.tangent
                    pos = _quad(q0, q1, wall.z0, wall.z1)
                    hp = wall.height_provenance
                    extras = {"kind": "wall", "wall": wall.index, "prism": pr.index, "ring": r, "piece": pi,
                              "observed": bool(is_obs), "locked": not is_obs,
                              "provenance": _provenance(wall.outline_source, hp, PROV_PHOTO if is_obs else PROV_DEFAULT),
                              "height_measured_here": hp == PROV_SFM,
                              "area_m2": round((u1 - u0) * (wall.z1 - wall.z0), 2)}
                    if ob is not None and ob.plane is not None:
                        extras["plane"] = ob.plane.to_meta()
                    name = f"wall-{wall.index}" if len(pieces) == 1 else f"wall-{wall.index}-{pi}"
                    if is_obs:
                        tw = fac.texture.shape[1]
                        c0 = int(np.clip(math.floor(u0 / wall.length * tw), 0, tw - 1))
                        c1 = int(np.clip(math.ceil(u1 / wall.length * tw), c0 + 1, tw))
                        tex = np.ascontiguousarray(fac.texture[:, c0:c1])
                        uu = [(u / wall.length * tw - c0) / (c1 - c0) for u in (u0, u1)]
                        uvs = np.array([[uu[0], 1.0], [uu[1], 1.0], [uu[1], 0.0], [uu[0], 0.0]])
                        extras["observed_fraction"] = round(float(fac.mask[:, c0:c1].mean()), 3)
                        extras["photos"] = photo_refs.get(wall.index, [])
                        parts.append(Part(name, pos, nrm, np.array([[0, 1, 2], [0, 2, 3]]), fac.base_rgb, tex, uvs, extras))
                    else:
                        if ob is not None and ob.reason:
                            extras["reason"] = ob.reason
                        elif fac is not None:
                            extras["reason"] = "not in any photo's observed part of this facade"
                        parts.append(Part(name, pos, nrm, np.array([[0, 1, 2], [0, 2, 3]]), CLAY_RGB, None, None, extras))
            if segs:
                corner_pos, corner_nrm, corner_idx = [], [], []
                for k in range(nk):
                    base_i = k * (segs + 1)
                    for s in range(segs):
                        pa, pb = rb[base_i + s], rb[base_i + s + 1]
                        t = (pb - pa) / max(np.linalg.norm(pb - pa), 1e-9)
                        nn = np.r_[t[1], -t[0], 0.0]
                        b = len(corner_pos)
                        corner_pos += [[*pa, pr.z0], [*pb, pr.z0], [*pb, pr.z1], [*pa, pr.z1]]
                        corner_nrm += [nn] * 4
                        corner_idx += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
                # a corner between two photographed strips takes the wall colour; others stay clay
                groups: dict[Any, list[int]] = {}
                for k in range(nk):
                    a_end = ends[(k - 1) % nk][1]
                    b_start, b_rgb = ends[k][0], ends[k][2]
                    groups.setdefault(b_rgb if (a_end and b_start and b_rgb) else None, []).append(k)
                cp, cn, ci = np.array(corner_pos), np.array(corner_nrm), np.array(corner_idx)
                for rgb, ks in groups.items():
                    quads = [k * segs + s_ for k in ks for s_ in range(segs)]
                    vsel = np.array([4 * q + i for q in quads for i in range(4)])
                    tsel = np.array([2 * q + i for q in quads for i in range(2)])
                    remap = {int(v): i for i, v in enumerate(vsel)}
                    idx = np.vectorize(remap.get)(ci[tsel])
                    obs_ = rgb is not None
                    hs = [walls[(r, k)].height_provenance for k in ks] + [walls[(r, (k - 1) % nk)].height_provenance for k in ks]
                    hp = PROV_INFERRED if PROV_INFERRED in hs else pr.height_source
                    parts.append(Part(f"corners-{pr.index}-{r}-{'obs' if obs_ else 'clay'}", cp[vsel], cn[vsel], idx,
                                      rgb or CLAY_RGB, None, None,
                                      {"kind": "corners", "prism": pr.index, "ring": r, "corners": ks, "observed": obs_,
                                       "locked": not obs_,
                                       "provenance": _provenance(pr.outline_source, hp, PROV_PHOTO if obs_ else PROV_DEFAULT)}))
        parts += _cap_parts(pr, rings_b, roof, segs, params)
    return parts


def _face_plane_z(pr: Prism, face: int) -> tuple[Any, np.ndarray]:
    """(z(xy) function, unit normal) of a cap face: flat at z1, or the plane of a sloped pyramid face."""
    if pr.cap == "pyramid" and pr.face_sloped and pr.face_sloped[face]:
        a = np.r_[pr.polygon[face], pr.z1]
        b = np.r_[pr.polygon[(face + 1) % len(pr.polygon)], pr.z1]
        n = np.cross(b - a, pr.apex - a)
        n = n / max(np.linalg.norm(n), 1e-12)
        if n[2] < 0:
            n = -n
        return (lambda xy: a[2] - ((xy[:, 0] - a[0]) * n[0] + (xy[:, 1] - a[1]) * n[1]) / max(n[2], 1e-6)), n
    return (lambda xy: np.full(len(xy), pr.z1)), np.array([0.0, 0.0, 1.0])


def _cap_parts(pr: Prism, rings_b: list[np.ndarray], roof: RoofLook, segs: int, params: StyleParams) -> list[Part]:
    parts: list[Part] = []
    ring = rings_b[0]
    n = len(ring)
    nk = len(pr.polygon)

    def roof_extras(faces, observed, height_src, area, **more):
        cov = [round(float(roof.coverage.get((pr.index, f), 0.0)), 3) for f in faces]
        return {"kind": "roof", "prism": pr.index, "faces": list(faces), "observed": observed, "locked": not observed,
                "coverage": cov, "area_m2": round(area, 2),
                "provenance": _provenance(pr.outline_source, height_src, PROV_PHOTO if observed else PROV_DEFAULT), **more}

    if pr.cap == "pyramid":
        sloped = pr.face_sloped or [True] * nk
        flat_apex = np.r_[pr.apex[:2], pr.z1]
        groups: dict[Any, list[int]] = {}
        for k in range(nk):
            groups.setdefault((bool(sloped[k]), roof.face_rgb.get((pr.index, k))), []).append(k)
        for (is_sloped, rgb), faces in groups.items():
            pos, nrm, idx = [], [], []
            top = pr.apex if is_sloped else flat_apex
            for k in faces:
                sub = _face_ring_indices(k, nk, segs, n, segs > 0)
                for a_i, b_i in itertools.pairwise(sub):
                    a3, b3 = np.r_[ring[a_i], pr.z1], np.r_[ring[b_i], pr.z1]
                    nn = np.cross(b3 - a3, top - a3)
                    nn /= max(np.linalg.norm(nn), 1e-9)
                    bse = len(pos)
                    pos += [a3, b3, top]
                    nrm += [nn] * 3
                    idx.append([bse, bse + 1, bse + 2])
            pos, nrm, idx = np.array(pos), np.array(nrm), np.array(idx)
            obs = rgb is not None
            src = PROV_SFM if is_sloped else PROV_DEFAULT
            parts.append(Part(f"roof-{pr.index}-{'slope' if is_sloped else 'flat'}-{'obs' if obs else 'clay'}-{faces[0]}",
                              pos, nrm, idx, rgb or CLAY_ROOF_RGB, None, None,
                              roof_extras(faces, obs, src, _tri_area(pos, idx), cap="pyramid" if is_sloped else "neutral-flat")))
        # vertical cuts where a sloped face meets a neutral flat one (the slope is not extended)
        per, half = segs + 1, segs // 2
        for k in range(nk):
            if bool(sloped[k - 1]) == bool(sloped[k]):
                continue
            corner = ring[(k * per + half) % n] if segs else ring[k]
            a3, b3, c3 = np.r_[corner, pr.z1], np.r_[pr.apex[:2], pr.z1], pr.apex.copy()
            flat_face = k if not sloped[k] else (k - 1) % nk
            fc = polygon_centroid(np.array([pr.polygon[flat_face], pr.polygon[(flat_face + 1) % nk], pr.apex[:2]]))
            nn = np.cross(b3 - a3, c3 - a3)
            nn /= max(np.linalg.norm(nn), 1e-9)
            tri = [a3, b3, c3]
            if (np.r_[fc, pr.z1] - a3) @ nn < 0:
                tri, nn = [a3, c3, b3], -nn
            pos = np.array(tri)
            parts.append(Part(f"roof-{pr.index}-cut-{k}", pos, np.repeat(nn[None], 3, axis=0), np.array([[0, 1, 2]]),
                              CLAY_ROOF_RGB, None, None,
                              {"kind": "roof-cut", "prism": pr.index, "corner": k, "observed": False, "locked": True,
                               "area_m2": round(_tri_area(pos, np.array([[0, 1, 2]])), 2),
                               "provenance": _provenance(pr.outline_source, PROV_DEFAULT, PROV_DEFAULT)}))
    else:
        rgb = roof.face_rgb.get((pr.index, 0))
        verts, tris = triangulate_with_holes(ring, rings_b[1:])
        pos = np.array([[*v, pr.z1] for v in verts])
        nrm = np.repeat(np.array([[0.0, 0.0, 1.0]]), len(pos), axis=0)
        idx = np.array(tris)
        src = pr.cap_source if pr.cap_source != PROV_DEFAULT or pr.height_source == PROV_DEFAULT else PROV_DEFAULT
        parts.append(Part(f"roof-{pr.index}", pos, nrm, idx, rgb or CLAY_ROOF_RGB, None, None,
                          roof_extras([0], rgb is not None, src, _tri_area(pos, idx), cap="flat")))
    # observed patches (decals) and clay holes inside them
    for i, patch in enumerate(p for p in roof.patches if p.prism == pr.index):
        zf, nn = _face_plane_z(pr, patch.face)
        poly = ensure_ccw(patch.polygon)
        tris = triangulate_polygon(poly)
        lift = 0.06 if patch.rgb is not None else 0.11
        pos = np.c_[poly, zf(poly)] + lift * nn[None]
        idx = np.array(tris)
        obs = patch.rgb is not None
        parts.append(Part(f"roof-{pr.index}-patch-{i}", pos, np.repeat(nn[None], len(pos), axis=0), idx,
                          patch.rgb or CLAY_ROOF_RGB, None, None,
                          {"kind": "roof-patch", "prism": pr.index, "faces": [patch.face], "observed": obs, "locked": not obs,
                           "coverage": [round(patch.coverage, 3)], "area_m2": round(_tri_area(pos, idx), 2),
                           "provenance": _provenance(pr.outline_source,
                                                     PROV_SFM if (pr.cap == "pyramid" and pr.face_sloped[patch.face]) else pr.cap_source,
                                                     PROV_PHOTO if obs else PROV_DEFAULT)}))
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
    """QA on what the pipeline itself enforced: outside a facade's observation mask only the clay
    pattern, no texture on locked faces, provenance on every face, palette colours only. The
    independent checks are in audit_model()."""
    outside = 0
    for fac in facades.values():
        off = ~fac.mask
        if off.any():
            outside += int((~is_clay_pattern(fac.texture[off])).sum())
    locked_detail = sum(1 for p in model_parts if p.extras.get("locked") and p.texture is not None)
    locked_photo = sum(1 for p in model_parts if p.extras.get("locked")
                       and p.extras.get("provenance", {}).get("appearance") == PROV_PHOTO)
    missing = sum(1 for p in model_parts
                  if not all(p.extras.get("provenance", {}).get(k) in PROVENANCES for k in PROVENANCE_KEYS))
    allowed = allowed_texture_colors()
    off_palette = 0
    for p in model_parts:
        if p.texture is None:
            continue
        cols = {tuple(int(v) for v in c) for c in np.unique(p.texture.reshape(-1, 3), axis=0)}
        off_palette += len(cols - allowed)
    return {"detail_pixels_outside_mask": outside, "textured_locked_faces": locked_detail,
            "locked_faces_claiming_photos": locked_photo, "faces_without_provenance": missing,
            "off_palette_colors": off_palette,
            "passed": outside == 0 and locked_detail == 0 and locked_photo == 0 and missing == 0 and off_palette == 0}


class _ColumnIndex:
    """SfM points bucketed in 1 m x 1 m columns (for the audit; independent of the roof grids)."""

    def __init__(self, points: np.ndarray, cell: float = 1.0) -> None:
        self.cell = cell
        p = np.asarray(points, float)
        keys = self._key(p[:, :2])
        order = np.argsort(keys, kind="stable")
        self.keys, self.z = keys[order], p[order, 2]

    def _key(self, xy: np.ndarray) -> np.ndarray:
        ix = np.floor(xy[:, 0] / self.cell).astype(np.int64) + 1_000_000
        iy = np.floor(xy[:, 1] / self.cell).astype(np.int64) + 1_000_000
        return ix * 2_000_003 + iy

    def any_between(self, xy: np.ndarray, z_lo: np.ndarray, z_hi: np.ndarray) -> np.ndarray:
        k = self._key(xy)
        lo, hi = np.searchsorted(self.keys, k, "left"), np.searchsorted(self.keys, k, "right")
        out = np.zeros(len(xy), dtype=bool)
        for i in np.nonzero(hi > lo)[0]:
            z = self.z[lo[i]:hi[i]]
            out[i] = bool(np.any((z >= z_lo[i]) & (z <= z_hi[i])))
        return out


def _sample_triangles(pos: np.ndarray, idx: np.ndarray, step: float = 1.0, max_samples: int = 4000) -> np.ndarray:
    """Uniform random points on a triangle mesh, about one per step x step of area (deterministic)."""
    p = np.asarray(pos, float)
    t = p[np.asarray(idx).reshape(-1, 3)]
    area = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1)
    n = int(np.clip(round(area.sum() / step ** 2), 1, max_samples))
    rng = np.random.default_rng(0)
    tri = rng.choice(len(t), n, p=area / max(area.sum(), 1e-12))
    r1, r2 = rng.random(n), rng.random(n)
    flip = r1 + r2 > 1
    r1[flip], r2[flip] = 1 - r1[flip], 1 - r2[flip]
    a = t[tri, 0]
    return a + r1[:, None] * (t[tri, 1] - a) + r2[:, None] * (t[tri, 2] - a)


def audit_model(massing: Massing, parts: list[Part], facades: dict[int, StylizedFacade],
                orthos: dict[int, Orthophoto], observations: list[WallObservation], points: np.ndarray | None,
                rings: list[np.ndarray], cameras: list[PosedCamera], params: StyleParams = DEFAULT_PARAMS,
                eligible: np.ndarray | None = None, registration_mode: str = "osm-footprint") -> dict[str, Any]:
    """Independent publication audit: recomputes from the inputs (points, cameras) what each face claims
    and fails on a mismatch, instead of re-checking what the pipeline just enforced.

    - roof faces / patches claiming photos: point support re-measured by sampling the face itself on a
      1 m lattice and looking for roof points in its 1 m column (not the roof grid used to decide);
    - textured walls: the points' distance to the face over the textured strip (0.25 m histogram) must
      peak within plane_tol_m + 0.25 m of the face, at least half as high as anywhere in +-plane_search_m;
    - walls whose height claims sfm-points must have their own top (from their points) within step_m of it;
    - occlusion: for sampled observed facade pixels, occluder_min_count or more (well triangulated) SfM
      points more than occluder_margin_m in front of the wall must not lie within 0.3 m of the ray from
      a contributing camera (a geometric segment test, not the per-camera depth buffer that built the mask);
    - registration re-checked edge by edge (outline from OSM only).
    """
    out: dict[str, Any] = {"roof_mismatches": [], "wall_plane_mismatches": [], "height_mismatches": [],
                           "occlusion": {}, "registration": None}
    if points is None or not len(points):
        out["passed"] = not any(p.extras.get("provenance", {}).get("appearance") == PROV_PHOTO for p in parts)
        out["note"] = "no points: nothing may claim photos"
        return out
    pts = np.asarray(points, float)
    cols = _ColumnIndex(pts)
    roof_checked = 0
    for p in parts:
        ex = p.extras
        if ex.get("kind") not in ("roof", "roof-patch") or ex.get("provenance", {}).get("appearance") != PROV_PHOTO:
            continue
        s = _sample_triangles(p.positions, p.indices)
        has = cols.any_between(s[:, :2], s[:, 2] - 1.5, s[:, 2] + params.step_m)
        share = float(has.mean())
        need = params.roof_face_min_coverage if ex["kind"] == "roof-patch" else params.roof_full_coverage
        roof_checked += 1
        if share < need - (0.1 if ex["kind"] == "roof-patch" else 0.15):
            out["roof_mismatches"].append({"part": p.name, "claimed": need, "measured": round(share, 3)})
    out["roof_parts_checked"] = roof_checked
    walls_checked = 0
    for p in parts:
        ex = p.extras
        if ex.get("kind") != "wall":
            continue
        wall = massing.walls[ex["wall"]]
        if ex.get("provenance", {}).get("height") == PROV_SFM and not wall.block_measured:
            top = _wall_top_near(wall, pts, params)
            if top is None or abs(top - wall.z1) > params.step_m:
                out["height_mismatches"].append({"part": p.name, "top_m": None if top is None else round(top, 2),
                                                 "z1": round(wall.z1, 2)})
        if not ex.get("observed"):
            continue
        walls_checked += 1
        rel = pts[:, :2] - wall.a
        u, d = rel @ wall.tangent, rel @ wall.normal
        q = p.positions[:2, :2]
        u0, u1 = sorted(float((v - wall.a) @ wall.tangent) for v in q)
        sel = (np.abs(d) <= params.plane_search_m) & (u >= u0) & (u <= u1) & (pts[:, 2] >= wall.z0 + 0.5) & (pts[:, 2] <= wall.z1)
        if sel.sum() < 20:
            out["wall_plane_mismatches"].append({"part": p.name, "points": int(sel.sum())})
            continue
        hist, edges = np.histogram(d[sel], bins=np.arange(-params.plane_search_m, params.plane_search_m + 0.25, 0.25))
        centres = (edges[:-1] + edges[1:]) / 2
        near = np.abs(centres) <= params.plane_tol_m + 0.25
        if hist[near].max() < 0.5 * hist.max():
            mode = float(centres[hist.argmax()])
            out["wall_plane_mismatches"].append({"part": p.name, "surface_offset_m": round(mode, 2),
                                                 "near_peak_share": round(float(hist[near].max() / max(hist.max(), 1)), 2)})
    out["walls_checked"] = walls_checked
    # occlusion: segment test on sampled observed pixels
    elig = np.ones(len(pts), bool) if eligible is None else eligible
    obs_by = {o.wall: o for o in observations}
    tot = blocked_tot = 0
    worst = []
    for wi, fac in facades.items():
        o = orthos.get(wi)
        if o is None or not o.samples:
            continue
        wall, ob = massing.walls[wi], obs_by[wi]
        rel = pts[:, :2] - wall.a
        u, d = rel @ wall.tangent, rel @ wall.normal
        dp = d - (ob.offset_a + (ob.offset_b - ob.offset_a) * u / max(wall.length, 1e-9))
        cand = pts[elig & (dp > params.audit_occluder_margin_m) & (dp < 150.0)]
        if len(cand) > 10000:
            cand = cand[np.random.default_rng(wi).choice(len(cand), 10000, replace=False)]
        h, w = o.mask.shape
        blocked = 0
        samples = o.samples[:100]
        for row, col, cams in samples:
            if not fac.mask[row, col]:
                continue
            uu = (col + 0.5) / w * wall.length
            zz = wall.z1 - (row + 0.5) / h * (wall.z1 - wall.z0)
            off = ob.offset_a + (ob.offset_b - ob.offset_a) * uu / max(wall.length, 1e-9)
            P = np.r_[wall.a + uu * wall.tangent + off * wall.normal, zz]
            hits = 0
            for ci in cams:
                C = cameras[ci].center
                seg = P - C
                L2 = float(seg @ seg)
                if not len(cand) or L2 < 1e-9:
                    continue
                t = np.clip(((cand - C) @ seg) / L2, 0.0, 1.0)
                dist = np.linalg.norm(cand - (C + t[:, None] * seg[None]), axis=1)
                hits += int(np.count_nonzero((dist < 0.3) & (t > 0.02) & (t < 0.98)) >= params.occluder_min_count)
            tot += 1
            if cams and hits >= 0.5 * len(cams):
                blocked += 1
        blocked_tot += blocked
        if samples:
            worst.append((round(blocked / len(samples), 3), wi))
    worst.sort(reverse=True)
    share = blocked_tot / max(tot, 1)
    out["occlusion"] = {"samples": tot, "occluded_share": round(share, 4), "worst_walls": worst[:5]}
    if registration_mode == "osm-footprint":
        reg = registration_check(pts, rings, params)
        out["registration"] = {"accepted": reg["accepted"], "verified_share": reg["verified_share"]}
        reg_ok = reg["accepted"]
    else:
        out["registration"] = {"accepted": True, "note": registration_mode}
        reg_ok = True
    out["passed"] = (not out["roof_mismatches"] and not out["wall_plane_mismatches"] and not out["height_mismatches"]
                     and share <= 0.05 and all(s <= 0.2 for s, _ in worst) and reg_ok)
    return out


def footprint_from_points(points: np.ndarray, params: StyleParams = DEFAULT_PARAMS, radius_m: float = 80.0) -> np.ndarray | None:
    """An outline traced from the points when OSM has none: 1 m cells holding >= 3 points above 2.5 m,
    closed and opened, the component under (or nearest to, within 30 m) the landmark centre (the ENU
    origin), simplified like an OSM outline. Provenance sfm-points; sides nobody photographed have no
    points and are therefore not part of it."""
    p = np.asarray(points, float)
    p = p[(p[:, 2] >= 2.5) & (np.linalg.norm(p[:, :2], axis=1) <= radius_m)]
    if len(p) < 200:
        return None
    cell = 1.0
    lo = p[:, :2].min(axis=0) - 2
    ix = ((p[:, 0] - lo[0]) / cell).astype(int)
    iy = ((p[:, 1] - lo[1]) / cell).astype(int)
    occ = np.zeros((iy.max() + 3, ix.max() + 3), np.int32)
    np.add.at(occ, (iy, ix), 1)
    m = (occ >= 3).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, cc, _stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n < 2:
        return None
    ox, oy = int(-lo[0] / cell), int(-lo[1] / cell)
    lab = cc[oy, ox] if 0 <= oy < cc.shape[0] and 0 <= ox < cc.shape[1] else 0
    if lab == 0:
        d = np.linalg.norm((cent[1:] - [ox, oy]) * cell, axis=1)
        if d.min() > 30:
            return None
        lab = int(d.argmin()) + 1
    polys = _contour_polygons(cc == lab, cell, (float(lo[0]), float(lo[1])), 10.0)
    if not polys:
        return None
    poly, _ = simplify_footprint(max(polys, key=lambda q: abs(signed_area(q))), params)
    return clean_ring(poly)


def stylize_landmark(scene: SceneInput, params: StyleParams = DEFAULT_PARAMS) -> StylizedModel:
    """Footprint + posed photos + SfM points -> stylized GLB and summary (see the module docstring).

    Raises StylizeError when there is neither a usable footprint nor points to trace one from."""
    timings: dict[str, float] = {}
    t = time.perf_counter()

    def lap(stage: str) -> None:
        nonlocal t
        now = time.perf_counter()
        timings[stage] = round(now - t, 3)
        t = now

    points = None if scene.points is None or not len(scene.points) else np.asarray(scene.points, float)
    colors = None if points is None or scene.colors is None else np.asarray(scene.colors)
    cameras = list(scene.cameras)
    outer = clean_ring(scene.footprint)
    holes = [h for h in (clean_ring(h, 10.0) for h in (scene.holes or ())) if h is not None]
    outline_source = PROV_OSM
    snap = None
    if outer is None:
        if points is None:
            raise StylizeError("no usable footprint (missing, empty, under 4 m2 or self-intersecting) and no SfM "
                               "points to trace an outline from")
        outer = footprint_from_points(points, params)
        if outer is None:
            raise StylizeError("no usable footprint, and no building outline found in the points near the landmark centre")
        holes, outline_source = [], PROV_SFM
        registration = {"accepted": True, "reason": "", "mode": "outline traced from the points themselves",
                        "verified_share": 1.0}
    elif points is None:
        registration = {"accepted": False, "reason": "no SfM points"}
    else:
        if scene.snap_override is not None:
            snap = scene.snap_override
        elif scene.snap:
            snap = snap_to_footprint(points, outer, params, holes=holes)
        if snap is not None and snap.accepted:
            points = apply_snap_points(snap, points)
            cameras = [c.snapped(snap) for c in cameras]
        registration = registration_check(points, [outer] + holes, params)
        registration["mode"] = "osm-footprint"
        if snap is not None and not snap.accepted and snap.reason != "no improvement":
            registration["accepted"] = False
            registration["reason"] = f"snap onto the footprint rejected: {snap.reason}"
    photo_ok = bool(registration["accepted"]) and points is not None
    rings = [outer] + holes
    lap("snap")
    massing = build_massing(outer, points if photo_ok else None, scene.facts, params, holes=holes,
                            outline_source=outline_source)
    lap("massing")
    if photo_ok:
        observations = observe_walls(massing, cameras, points, params)
    else:
        why = f"registration not verified: {registration.get('reason') or 'unknown'}"
        observations = [WallObservation(w.index, [], 0, 0.0, 0.0, np.zeros((2, 2), bool), None, False, why)
                        for w in massing.walls]
    lap("observe")
    orthos: dict[int, Orthophoto] = {}
    facades: dict[int, StylizedFacade] = {}
    cache: dict[int, np.ndarray] = {}
    eligible = occluder_eligible(points, scene.point_quality, params) if photo_ok else None
    for ob in observations:
        if not ob.observed:
            continue
        wall = massing.walls[ob.wall]
        ortho = facade_orthophoto(wall, ob, massing, cameras, params, cache, points, eligible)
        orthos[wall.index] = ortho
        if ortho.observed_fraction < params.min_observed_fraction:
            ob.observed, ob.reason = False, f"observation mask covers {ortho.observed_fraction:.0%}"
    cache.clear()
    lap("orthophotos")
    live = {wi: o for wi, o in orthos.items() if observations[wi].observed}
    norm = scene_color_norm(list(live.values()))
    layers = {wi: facade_layers(o, norm, params) for wi, o in live.items()}
    plan = plan_colors(list(layers.values()), params)
    size_scale = max(1.0, massing.prisms[0].z1 / params.default_height_m)
    for wi, ortho in live.items():
        fac = stylize_facade(ortho, params, ground_row=massing.walls[wi].z0 <= 0.5, norm=norm, plan=plan,
                             layers=layers[wi], size_scale=size_scale)
        frac = float(fac.mask.mean())
        if fac.report.get("empty") or frac < params.min_observed_fraction:
            observations[wi].observed = False
            observations[wi].reason = f"observation mask covers {frac:.0%} without sky / vegetation"
            continue
        facades[wi] = fac
    lap("stylize")
    roof = roof_appearance(massing, points if photo_ok else None, colors, norm=norm, params=params)
    photo_refs = {}
    for wi in facades:
        refs = []
        for ci, share in orthos[wi].contributions:
            if share < 0.01:
                continue
            cam = cameras[ci]
            refs.append({"photo": cam.photo_id or cam.name, "license": cam.license, "share": share})
        photo_refs[wi] = refs
    parts = build_parts(massing, facades, roof, params, observations, photo_refs)
    honesty = check_honesty(parts, facades)
    audit = audit_model(massing, parts, facades, orthos, observations, points if photo_ok else None, rings, cameras,
                        params, eligible=eligible, registration_mode=registration.get("mode", ""))
    qa = {**honesty, "audit": audit, "passed": bool(honesty["passed"] and audit["passed"])}
    reg_meta = {k: v for k, v in registration.items() if k != "edges"}
    extras = {"generator": "crowd3d.stylize", "frame": "ENU (glTF y-up: x=east, y=up, -z=north)",
              "landmark": scene.name, "attribution": list(scene.attribution),
              "provenance_values": list(PROVENANCES), "locked_label": "待解锁",
              "registration": reg_meta, "snap": snap.to_meta() if snap else None}
    glb = write_glb(parts, extras)
    lap("mesh_glb")

    def area(kind: str, appearance: str | None = None) -> float:
        return sum(p.extras.get("area_m2", 0.0) for p in parts if p.extras.get("kind") == kind
                   and (appearance is None or p.extras["provenance"]["appearance"] == appearance))

    wall_area = area("wall")
    obs_strip = area("wall", PROV_PHOTO)
    obs_px = sum(p.extras["area_m2"] * p.extras.get("observed_fraction", 0.0) for p in parts
                 if p.extras.get("kind") == "wall" and p.extras.get("observed"))
    roof_area = area("roof")
    roof_obs = area("roof", PROV_PHOTO) + area("roof-patch", PROV_PHOTO)
    tri = int(sum(len(p.indices) for p in parts))
    height_prov = {}
    for w in massing.walls:
        a = w.area
        height_prov[w.height_provenance] = round(height_prov.get(w.height_provenance, 0.0) + a, 1)
    summary = {
        "landmark": scene.name,
        "inputs": {"cameras": len(cameras), "points": 0 if points is None else len(points),
                   "footprint_vertices": 0 if scene.footprint is None else int(np.size(scene.footprint) // 2),
                   "footprint_holes": len(holes), "outline_source": outline_source},
        "snap": snap.to_meta() if snap else None,
        "registration": registration,
        "massing": massing.report,
        "color_norm": norm.to_meta(),
        "color_plan": plan.to_meta(),
        "walls": [{"index": w.index, "prism": w.prism, "ring": w.ring, "length_m": round(w.length, 2),
                   "z0": round(w.z0, 2), "z1": round(w.z1, 2), "outline": w.outline_source, "height": w.height_provenance,
                   **observations[w.index].to_meta(),
                   "geometric_mask_fraction": round(orthos[w.index].observed_fraction, 3) if w.index in orthos else 0.0,
                   "ortho": orthos[w.index].stats if w.index in orthos else None,
                   "mask_fraction": round(float(facades[w.index].mask.mean()), 3) if w.index in facades else 0.0,
                   "facade": facades[w.index].report if w.index in facades else None}
                  for w in massing.walls],
        "observed_walls": len(facades),
        "num_walls": len(massing.walls),
        "observed_wall_area_fraction": round(obs_strip / max(wall_area, 1e-9), 3),
        "observed_pixel_area_fraction": round(obs_px / max(wall_area, 1e-9), 3),
        "wall_height_provenance_m2": height_prov,
        "roof_area_m2": round(roof_area, 1),
        "roof_observed_area_m2": round(roof_obs, 1),
        "roof_observed_area_fraction": round(roof_obs / max(roof_area, 1e-9), 3),
        "roof_faces": len(roof.coverage),
        "roof_faces_observed_whole": sum(v is not None for v in roof.face_rgb.values()),
        "roof_patches": sum(p.rgb is not None for p in roof.patches),
        "appearance_provenance_triangles": {k: int(sum(len(p.indices) for p in parts
                                                     if p.extras.get("provenance", {}).get("appearance") == k))
                                            for k in (PROV_PHOTO, PROV_DEFAULT)},
        "triangles": tri,
        "vertices": int(sum(len(p.positions) for p in parts)),
        "nodes": len(parts),
        "textures": sum(1 for p in parts if p.texture is not None),
        "atlases": len(read_glb(glb)[0].get("images", [])),
        "glb_bytes": len(glb),
        "qa": qa,
        "timings_s": timings,
        "total_s": round(sum(timings.values()), 3),
    }
    return StylizedModel(glb, summary, massing, observations, orthos, facades, parts, roof)
