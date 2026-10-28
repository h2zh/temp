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
   strip and clay strips. One exception (D12, the product owner's decision): a hole of the mask that
   observed texels of the same wall enclose on all sides, that is at most fill_max_hole_m2 (a tree,
   a passer-by) and that cannot hold a whole drawable element (fill_shape_kernels()) takes the wall's
   measured base colour, nothing else; those texels are observed-fill, recorded per texel and never
   counted as observed.
4. Roofs. A cap face is coloured as a whole only when its point coverage is >= roof_full_coverage;
   otherwise its observed patches become decals and the rest stays clay. A face or patch counts as seen only
   when a camera faces it (in front of its plane, its centre in frame); recoloured point colours drop samples
   on the photo's sky mask and need two agreeing views. A structure part that is a 'height pending' stub (D13)
   is clay on every face.
5. Output: a GLB (minimal glTF 2.0 writer, no dependency) where every face group is its own node with
   provenance {outline, height, appearance} in its extras (plus coverage, area, contributing photos),
   facade textures packed into shared lossless atlases, the registration and snap in the scene extras,
   plus a JSON-able summary. What is measured is kept apart from how it looks: the scene extras carry
   the colour table (palette names, families, shades, ink, clay), flat materials name their colour and
   greyscale atlases hold each facade texel's provenance (clay / observed / observed-fill), so a viewer
   theme (web/themes) can restyle the model without re-running this. QA: check_honesty() (what the
   pipeline enforced), audit_model() (independent recomputation from points and cameras of coverage,
   planes, heights, occlusion, registration and the legal fill) and audit_glb() (the published bytes); with a
   structure massing also the structure gates (structure_gates() / finish_structure_gates(): G1-G3, G6, G10, G11
   and the parameters' states, docs/STRUCTURE_FIRST.md §2.8), all ANDed into qa.passed. When the registration is
   refused, everything the judge concluded about the structure from the same photos (D15 adoptions, photo
   verifications) is withdrawn first (crowd3d.judge.withdraw_photo_judgements, on a copy of scene.structure).
   web/stylized.html renders it with toon shading and outlines.

Coordinates are the landmark ENU frame (x = east, y = north, z = up, metres). The GLB uses glTF's
y-up convention: ENU (x, y, z) -> glTF (x, z, -y), the same mapping as web/viewer.js.

No generative model is used. Steps where an allowed one (simplification, segmentation or
stylization of observed content only, never completion) would help are marked "GENERATIVE-OK".
"""

# ruff: noqa: RUF046  (int(round(x)) on numpy scalars is deliberate: OpenCV wants Python ints)
from __future__ import annotations

import copy
import functools
import io
import itertools
import json
import math
import struct
import time
import warnings
from collections import Counter
from dataclasses import dataclass, field, replace
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
PROV_FILL = "observed-fill"          # texel level only (D12): a small hole enclosed by observed texels of the same
                                     # wall, painted that wall's measured base colour; never counted as observed
# structure-first massing (crowd3d.structure, docs/STRUCTURE_FIRST.md §2.4.1): facts and fact-derived conventions
PROV_S3DB = "osm-s3db"               # outline / height / roof shape tagged on an OSM building:part (Simple 3D Buildings)
PROV_LEVELS = "osm-levels"           # building:levels / building:min_level / roof:levels x 3 m
PROV_ROOF_ANGLE = "osm-roof-angle"   # roof height from roof:angle and the part's half-width
PROV_SHAPE_DEFAULT = "s3db-shape-default"   # roof:shape tagged, roof height not: the shape's conventional proportion (D17)
PROV_SUPPORT = "osm-s3db-support"    # a main-outline gap's top: the min_height of the raised parts resting on it
PROV_GROUND = "osm-default-ground"   # no min_height tagged: the S3DB default 0 (on the ground)
PROV_CURATOR = "curator"             # entered or approved by a curator (same parameter format)
PROV_FITTED = "photo-fitted"         # changed by photo evidence (a curator-confirmed photo fit)
PROV_FITTED_AUTO = "photo-fitted (auto)"   # changed automatically: >= 2 photo channels agree (D15, on by default;
                                           # the map value and all evidence archived, reversible; crowd3d.judge)
PROV_CORRECTED_AUTO = "photo-corrected (auto)"   # D25 (strong photo evidence corrects OSM): the full photo value V
                                                 # (or a measured tilt); map value, evidence and the registration box
                                                 # archived, reversible (crowd3d.correct)
PROV_CORRECTED_LB = "photo-corrected (auto, lower bound)"   # D25: only the guaranteed lower bound (a height whose box
                                                            # lower bound sits below the midpoint, every D25-B block)
PROV_CORRECTED_UB = "photo-corrected (auto, upper bound)"   # D25 going down: only the guaranteed upper bound
PROVENANCES = (PROV_PHOTO, PROV_OSM, PROV_SFM, PROV_OSM_HEIGHT, PROV_WIKIDATA, PROV_INFERRED, PROV_DEFAULT,
               PROV_S3DB, PROV_LEVELS, PROV_ROOF_ANGLE, PROV_SHAPE_DEFAULT, PROV_SUPPORT, PROV_GROUND, PROV_CURATOR,
               PROV_FITTED, PROV_FITTED_AUTO, PROV_CORRECTED_AUTO, PROV_CORRECTED_LB, PROV_CORRECTED_UB)
PROVENANCE_KEYS = ("outline", "height", "appearance")   # every face carries all three
TEXEL_PROVENANCES = (PROV_DEFAULT, PROV_PHOTO, PROV_FILL)   # per facade texel; the index is the code in the GLB's
                                                            # texel-provenance atlases (0 clay, 1 observed, 2 fill)

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
    # near-neutral greys (docs/STRUCTURE_FIRST.md §2.5 item 3): the stone ramp is warm (stone b* 8.4), so white
    # marble snapped by lightness read as beige. Light greys keep b* ~ +3..+4: the theme validator needs every
    # observed colour >= 7.5 dE00 from the cool clay, which a truly neutral light grey fails (D21)
    ("ash", (178, 175, 169)),        # L 71.5, b* 3.5
    ("smoke", (140, 141, 143)),      # L 58.6
    ("iron", (116, 117, 119)),       # L 49.3
)
PALETTE_RGB = np.array([c for _, c in PALETTE], dtype=np.uint8)
PALETTE_NAMES = tuple(n for n, _ in PALETTE)
# material families (what a theme may restyle as a group, web/themes); every palette name has one
PALETTE_FAMILIES: dict[str, tuple[str, ...]] = {
    "neutral": ("snow", "stone-light", "stone", "pebble", "stone-dark", "graphite", "charcoal", "ash", "smoke", "iron"),
    "sand": ("cream", "sand", "ochre"),
    "gold": ("sunflower", "gold"),
    "brick": ("peach", "terracotta", "brick", "rust"),
    "green": ("mint", "leaf", "forest", "pine"),
    "blue": ("sky", "lake", "navy"),
    "slate": ("slate-light", "slate", "night"),
    "wood": ("wood", "cocoa"),
    "copper": ("copper", "teal"),
}
FAMILY_OF = {n: fam for fam, names in PALETTE_FAMILIES.items() for n in names}
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


STYLE_VERSION = 1


def color_table() -> list[dict[str, Any]]:
    """What every colour of a published model means, so that a viewer can restyle it (web/themes) without
    re-running this module: the palette (role palette, with its family), its shades (region outlines,
    role shade, of = the palette name), ink and the clay colours of locked faces. Every rgb is distinct
    and atlases are lossless, so a texel's colour identifies its entry exactly. web/themes/palette.json
    is a copy (a test keeps the two equal)."""
    rows: list[dict[str, Any]] = [{"name": n, "rgb": list(c), "role": "palette", "family": FAMILY_OF[n]} for n, c in PALETTE]
    rows += [{"name": f"{n}.shade", "rgb": list(shade(c)), "role": "shade", "of": n, "family": FAMILY_OF[n]} for n, c in PALETTE]
    rows += [{"name": "ink", "rgb": list(INK_RGB), "role": "ink"},
             {"name": "clay", "rgb": list(CLAY_RGB), "role": "clay"},
             {"name": "clay-hatch", "rgb": list(CLAY_HATCH_RGB), "role": "clay-hatch"},
             {"name": "clay-roof", "rgb": list(CLAY_ROOF_RGB), "role": "clay-roof"}]
    return rows


def style_meta() -> dict[str, Any]:
    """The GLB's scene extras "style": colour table plus how shades are derived."""
    return {"version": STYLE_VERSION, "shade_factor": SHADE_FACTOR, "colors": color_table()}


COLOR_NAME_OF_RGB = {tuple(r["rgb"]): r["name"] for r in color_table()}


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
    # model -> footprint snap (STYLIZED_MODEL 3.2): yaw only within the registration's own yaw window ...
    snap_yaw_sigmas: float = 3.0         # ... +-this many of its reported yaw sigmas ...
    snap_default_yaw_sigma_deg: float = 6.7   # ... (a model that reports none: +-20 deg, the former fixed cap) ...
    snap_min_yaw_sigma_deg: float = 3.0  # ... (a reported sigma is floored at this: the pipeline's
                                         # Params.yaw_systematic_sigma_deg, so an understated sigma cannot open the
                                         # translation-only mode or narrow the window below what a site's shared GPS
                                         # and compass errors allow) ...
    snap_symmetry_margin_deg: float = 5.0     # ... and never beyond P/2 - this (P: the outline's symmetry period) ...
    snap_min_window_deg: float = 2.0     # ... narrower than this (a near-circle): translation only, yaw as registered,
    snap_fixed_yaw_max_deg: float = 5.0  # ... and trusted only when snap_yaw_sigmas x its reported sigma is within
                                         # this and moves the outline's boundary by at most snap_max_residual_m
    snap_window_edge_deg: float = 1.0    # an optimum this close to the window edge is not trusted (flagged, clay)
    snap_grid_step_deg: float = 2.0      # coarse search: every rotation this far apart across the window ...
    snap_grid_step_m: float = 1.5        # ... x every shift this far apart within snap_max_shift_m ...
    snap_grid_points: int = 300          # ... scored on this many wall points; its best local minima ...
    snap_grid_starts: int = 8            # ... (at most this many) start the ICP
    snap_icp_inlier_m: float = 1.0       # ICP pairs: the closest 70 % of the points, and only those within
                                         # max(this, 3 x their median distance) (a neighbour, a tree, a dome next to
                                         # the outline turned the fit 3-7 deg on synthetic walls)
    snap_max_shift_m: float = 25.0       # the walls move at most this far (real single-photo GPS p90: 20-55 m)
    snap_max_residual_m: float = 1.5     # the snapped wall points must sit this close (median) to the outline
    snap_max_fit_m: float = 0.45         # ... and the ones near it this close on average (capped at reg_edge_limit_m):
                                         # walls that follow the outline more loosely cannot fix the yaw within 5 deg
                                         # (Frauenkirche SW#9/#20/#24/#25 at 0.50-0.55 m were 3-5 deg off the
                                         # reference; the right SW#11/#14/#15 fitted 0.30-0.34 m; calibrated on one
                                         # building)
    snap_max_scale: float = 1.15         # the outline may correct the SfM scale by up to this factor (1: rigid only)
    snap_ambiguity_m: float = 1.5        # a second verified placement that would put the photos this far from the
    snap_ambiguity_deg: float = 2.0      # best one's (90th percentile over the walls) or turns them this much ...
    snap_ambiguity_margin: float = 0.25  # ... fitting within this share ...
    snap_ambiguity_slack_m: float = 0.1  # ... plus this of the best one's mean distance: ambiguous, clay
    snap_ambiguity_verified_only: bool = True   # ... (only when that placement also verifies edge by edge)
    snap_ambiguity_turn_deg: float = 5.0 # the walls turned 2 / 3.5 / this many deg about themselves (shift re-fitted)
                                         # still fit within that margin and move the photos >= snap_ambiguity_m, or
                                         # turned this much at all (a right placement is within 5 deg): ambiguous
    snap_ambiguity_cameras: bool = True  # ... and how far a rival moves the cameras counts too (median >=
                                         # snap_ambiguity_m; the profile also tries the turn that moves them that far):
                                         # a right registration has its cameras within 1.5 m (STYLIZED_MODEL 7), and
                                         # cameras 80 m away move 1.5 m for a 1 deg turn the walls hardly notice
    snap_scale_steps: tuple[float, ...] = (0.03, 0.06, 0.10)   # the best placement rescaled by +-these about the
                                         # walls (shift re-fitted) ...
    snap_scale_gps_sigmas: float = 2.0   # ... and within this many of the GPS fit's scale sigmas (any, when unknown)
                                         # still fits within the margin and moves the photos or the cameras >=
                                         # snap_ambiguity_m: the scale is open (an L of two faces fits its outline at
                                         # 7 % off): ambiguous
    snap_max_tilt_shift_m: float = 1.5   # tilt check (camera_tilt_check): the unflagged cameras' heights must not rise
                                         # across the model by more than lifts or sinks the walls this much ...
    snap_tilt_sigmas: float = 1.0        # ... (plus this many standard errors of that estimate) ...
    snap_tilt_min_cameras: int = 5       # ... measured on at least this many cameras (fewer: unverifiable, clay)
    snap_symmetry_tol_m: float = 1.0     # outline symmetry: a rotation maps a boundary sample onto the outline ...
    snap_symmetry_min_share: float = 0.3 # ... and is an alias when this share of the samples land (a local maximum ...
    snap_symmetry_prominence: float = 0.1     # ... this far above the dip before it); every rotation keeping ...
    snap_symmetry_continuous: float = 0.9     # ... this share: rotationally symmetric (P = 0)
    snap_max_scale_sigma: float = 0.15   # a GPS scale less certain than this (and not fixed by the outline): refused
    snap_gps_min_cameras: int = 4        # GPS check: unflagged cameras with their own EXIF GPS needed ...
    snap_gps_scale_tol: float = 0.05     # ... cameras -> walls distance vs GPS -> walls: agree within this ...
    snap_gps_sigmas: float = 1.5         # ... or this many standard errors of that ratio, else the snap is refused
    # registration gate (no photo content at all unless the points sit on the outline edge by edge)
    reg_edge_limit_m: float = 3.0        # an edge is verified when its points' median distance is under this ...
    reg_support_frac: float = 0.25       # ... among edges carrying >= this share of the median edge's point count ...
    reg_min_share: float = 0.6           # ... and this share of those edges must be verified
    reg_camera_front_m: float = 1.0      # a voting edge needs a camera at least this far on its outer side (points
                                         # cannot lie on a face that every camera sees from behind: a slide onto the
                                         # far side of the massing, a neighbour's facade moved onto this one)
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
    occluder_tube_m: float = 0.0         #     (> 0: each point hides a disc of this radius in metres instead of its cell;
                                         #      the structure path uses 0.3, audit_model's tube)
    occluder_min_track: int = 3          # ... counting only points seen by this many photos ...
    occluder_min_angle_deg: float = 5.0  # ... under this triangulation angle (when the point quality is known)
    audit_occluder_margin_m: float = 1.5 # the audit's own occluder margin (independent of the one that built the masks)
    max_occluders: int = 80000           # occluder candidates per wall (random subsample beyond)
    consistency_rgb: float = 60.0        # a view agrees with the blend when its colour is this close (RGB distance) ...
    consistency_blur_m: float = 0.4      # ... compared on averages over this window
    min_agreement: float = 0.5           # ... and this share of a pixel's views must agree (at least min_views)
    # massing
    massing_source: str = "osm-parts"    # "osm-parts": a SceneInput.structure (crowd3d.structure, OSM building:part
                                         # solids) is the massing when given (docs/STRUCTURE_FIRST.md); "height-field":
                                         # the point height field below (also the fallback without parts)
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
    fill_max_hole_m2: float = 0.5        # D12: an unobserved hole enclosed by observed texels of the same wall, at most
                                         # this large and too small to hold a whole drawable element (fill_shape_kernels())
                                         # takes the wall's measured base colour (observed-fill, not observed); other
                                         # holes stay clay (0 disables filling)
    # stylization
    work_px_per_m: float = 8.0           # segmentation resolution (details under ~0.25 m are dropped)
    local_window_m: float = 3.0          # dark accents: darker than the mean of this window ...
    dark_contrast: float = 6.0           # ... by this much (CIE L*)
    accent_min_chroma: float = 14.0      # coloured accents: at least this chroma (and 8 above the wall's)
    min_accent_m2: float = 0.06          # accent specks under this go
    min_own_accent_m2: float = 0.25      # a compact coloured accent of this size keeps its own colour (a gilded door)
    min_free_shape_m2: float = 0.6       # free-form dark shapes (not openings / bands) under this go (x building scale) ...
    min_free_solidity: float = 0.7       # ... and so do ragged ones (area / convex hull area under this)
    roof_lift: float = 0.2               # roofs (often dark slate) are lifted towards light: cheerful, readable (was 0.5:
                                         # it turned black slate into stone, docs/STRUCTURE_FIRST.md §2.5 item 5)
    roof_quantile: float = 60.0          # a roof's lightness: this percentile of its points' L* (sky-coloured points dropped)
    wall_lift: float = 0.2               # wall colour lifted this share of the way to white before snapping
    dark_lift: float = 0.0               # dark material (courses, frames) lift before snapping
    # colour chain fixes (docs/STRUCTURE_FIRST.md §2.5, colour probe variant M)
    wb_reanchor: bool = True             # per wall, one per-channel gain back to the per-texel median of the raw views
                                         # (the exposure-matched blend drifts towards its warm, sunlit views)
    neutral_anchor: bool = True          # white balance neutral when the lightest material is neutral in good light
    anchor_max_chroma: float = 4.0       # ... "neutral": C* at most this in the best-exposed quartile of views
    exposure_quantile: float = 0.75      # the best-exposed views: this quantile of brightness on the light material
    exposure_max: float = 1.0            # exposure lift towards those views (1: off, the conservative default; D21)
    stripes: bool = True                 # horizontal courses of light / dark stone (zebra marble) as regular bands
    stripe_contrast: float = 14.0        # a dark course is at least this much darker (L*) than its light neighbours ...
    stripe_min_runs: int = 4             # ... in a run of at least this many regular courses ...
    stripe_thickness_m: tuple[float, float] = (0.12, 0.9)   # ... each this thick ...
    stripe_thickness_cv: float = 0.35    # ... with thicknesses within this coefficient of variation ...
    stripe_period_cv: float = 0.3        # ... and spacing within this one; a column group needs >= stripe_min_runs courses
    recolor_black_share: float = 0.5     # point colours: more than this share black / uncoloured -> recoloured from photos
    recolor_sky_mask: bool = True        # ... never from a pixel the photo's sky mask (crowd3d.skyline A+R, licence-safe)
                                         # calls sky (dilated by recolor_sky_dilate_px): a slender part in front of white
                                         # sky took the sky's colour through the 4 px depth buffer
    recolor_sky_dilate_px: int = 2
    recolor_min_views: int = 2           # ... and a recoloured point counts only when this many photos agree on it ...
    recolor_max_spread: float = 12.0     # ... (median absolute deviation of their L* at most this; else uncoloured)
    roof_min_cameras: int = 1            # a roof face / patch is observed only when this many cameras face it (in front of
                                         # its plane, its centre in frame), not by point presence alone
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
    # photo-measured structure (crowd3d.relief): facade relief on the photographed walls of the frame, built only
    # where every guard passes (two photo halves, a 3-D line, observed texels ...) -- or, D24, every point-based guard
    # with the wall's 3-D lines at chance, on a model whose scale condition is met; needs SceneInput.point_tracks
    relief_detail: bool = True


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
class SnapPrior:
    """What the georegistration says about its own error (STYLIZED_MODEL 3.2): the snap's yaw window comes
    from its reported yaw sigma, the GPS check from the model's cameras and their own EXIF GPS."""

    yaw_sigma_deg: float | None = None     # 1-sigma yaw uncertainty the georegistration reported (None: unknown)
    camera_xy: np.ndarray | None = None    # (n, 2) ENU centres of the model's unflagged cameras that have GPS ...
    gps_xy: np.ndarray | None = None       # ... and those photos' own EXIF GPS positions (ENU), same order
    scale_sigma: float | None = None       # relative 1-sigma scale uncertainty of the GPS fit (None: unknown)
    georegistered: bool | None = None      # False: prior only (compass yaw, approximate scale, no GPS fit): the
                                           # model colours nothing (3.2 item 4); None: not a pipeline model
    camera_xyz: np.ndarray | None = None   # (n, 3) ENU centres of all the model's unflagged cameras (tilt check,
                                           # cameras' displacement in the ambiguity checks, edge visibility)

    def to_meta(self) -> dict[str, Any]:
        return {"yaw_sigma_deg": None if self.yaw_sigma_deg is None else round(float(self.yaw_sigma_deg), 2),
                "scale_sigma": None if self.scale_sigma is None else round(float(self.scale_sigma), 4),
                "gps_cameras": 0 if self.camera_xy is None else int(len(self.camera_xy)),
                "cameras": 0 if self.camera_xyz is None else int(len(self.camera_xyz)),
                "georegistered": self.georegistered}


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
    search: dict[str, Any] = field(default_factory=dict)   # yaw window, symmetry period, mode, starts, alternatives
    gps: dict[str, Any] | None = None   # the cameras' distance to the walls against their own GPS (None: not checked)

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
                "num_points": self.num_points, "accepted": self.accepted, "reason": self.reason,
                "search": self.search, "gps": self.gps}

    @classmethod
    def from_meta(cls, meta: dict[str, Any]) -> FootprintSnap:
        """A snap stored with an earlier model version (summary["snap"] / the GLB's scene extras), reused
        so that a republished model does not shift."""
        nan = float("nan")
        return cls(float(meta["rotation_deg"]), np.asarray(meta["shift_m"], float),
                   float(meta.get("median_wall_distance_before_m") or nan), float(meta.get("median_wall_distance_after_m") or nan),
                   int(meta.get("num_points", 0)), bool(meta.get("accepted", True)), "reused from an earlier model version",
                   float(meta.get("scale", 1.0)), dict(meta.get("search") or {}), meta.get("gps"))


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


def _rot2(deg: float) -> np.ndarray:
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s], [s, c]])


def _boundary_samples(rings: list[np.ndarray], max_samples: int = 1500, step: float = 0.25) -> np.ndarray:
    per = sum(float(np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1).sum()) for r in rings)
    step = max(step, per / max_samples)
    out = []
    for ring in rings:
        for a, b in zip(ring, np.roll(ring, -1, axis=0)):
            n = max(1, int(np.linalg.norm(b - a) / step))
            out.append(a + np.linspace(0.0, 1.0, n, endpoint=False)[:, None] * (b - a))
    return np.vstack(out)


def outline_symmetry_period(footprint: np.ndarray, holes: list[np.ndarray] | None = None,
                            params: StyleParams = DEFAULT_PARAMS) -> float:
    """The outline's rotational symmetry period P in degrees (STYLIZED_MODEL 3.2): the smallest rotation about
    its centroid that lands a large share of its boundary back on it (a regular octagon 45, a rectangle 180),
    where a facade can alias; 0 when every rotation does (a circle, a 16-gon: yaw is unobservable from the
    outline), 360 when none does. A boundary sample has landed within snap_symmetry_tol_m; an alias is a local
    maximum of the landed share >= snap_symmetry_min_share and snap_symmetry_prominence above the lowest share
    at smaller rotations (so the plateau around 0 is not one)."""
    fp = ensure_ccw(np.asarray(footprint, float))
    rings = [fp] + [np.asarray(h, float) for h in (holes or [])]
    centre = polygon_centroid(fp)
    pts = _boundary_samples(rings)
    radius = float(np.max(np.linalg.norm(pts - centre, axis=1)))
    span = float(np.max(fp.max(axis=0) - fp.min(axis=0))) + 2.0 * radius
    lookup = _OutlineLookup(fp, radius + 1.0, res=max(0.05, span / 2000), holes=holes)
    share = np.ones(181)
    for deg in range(1, 181):
        moved = (pts - centre) @ _rot2(deg).T + centre
        share[deg] = float(np.mean(np.linalg.norm(moved - lookup.closest(moved), axis=1) <= params.snap_symmetry_tol_m))
    if share[1:].min() >= params.snap_symmetry_continuous:
        return 0.0
    for k in range(2, 181):
        # a local maximum over +-5 deg (a plateau on the way up to the alias is not one yet)
        if (share[k] >= params.snap_symmetry_min_share and share[k] >= share[max(1, k - 5):k + 6].max()
                and share[k] - share[1:k].min() >= params.snap_symmetry_prominence):
            return float(k)
    return 360.0


def yaw_window(period_deg: float, yaw_sigma_deg: float | None, params: StyleParams = DEFAULT_PARAMS
               ) -> tuple[float, str, float]:
    """(half-width in degrees, mode, sigma used): +-min(snap_yaw_sigmas x sigma, P/2 - snap_symmetry_margin_deg);
    narrower than snap_min_window_deg: "translation-only" (the registered yaw is kept), else "window". A reported
    sigma is floored at snap_min_yaw_sigma_deg."""
    sigma = (params.snap_default_yaw_sigma_deg if yaw_sigma_deg is None or not math.isfinite(yaw_sigma_deg)
             else max(float(yaw_sigma_deg), params.snap_min_yaw_sigma_deg))
    window = params.snap_yaw_sigmas * sigma
    if period_deg < 360.0:
        window = min(window, period_deg / 2.0 - params.snap_symmetry_margin_deg)
    if window < params.snap_min_window_deg:
        return 0.0, "translation-only", sigma
    return window, "window", sigma


def _icp_inliers(d: np.ndarray, inlier_m: float) -> np.ndarray:
    """ICP pairs kept: the closest 70 %, and of those only the ones within max(inlier_m, 3 x median)."""
    return d <= min(float(np.quantile(d, 0.7)), max(inlier_m, 3.0 * float(np.median(d))))


def _snap_icp(xy: np.ndarray, lookup: _OutlineLookup, rot0_deg: float, t0: np.ndarray, window_deg: float,
              scale: float = 1.0, iters: int = 100, inlier_m: float = DEFAULT_PARAMS.snap_icp_inlier_m
              ) -> tuple[float, np.ndarray, float]:
    """Trimmed 2-D ICP of p' = scale R p + t onto the outline, R held within +-window_deg of no rotation (0:
    translation only). Returns (rotation deg, t, median distance). The shift is not bounded here: clamping it
    inside the iterations stopped paths that cross the bound on their way to an optimum inside it (Frauenkirche
    SW#20, the Baptistery reference); snap_to_footprint() refuses a final placement beyond the bound."""
    p = scale * np.asarray(xy, float)
    r, t = _rot2(rot0_deg), np.asarray(t0, float).copy()
    for _ in range(iters):
        moved = p @ r.T + t
        target = lookup.closest(moved)
        d = np.linalg.norm(moved - target, axis=1)
        keep = _icp_inliers(d, inlier_m)
        if window_deg > 0:
            dr, _ = _rigid_2d(moved[keep], target[keep])
            r_new = dr @ r
            ang = math.degrees(math.atan2(r_new[1, 0], r_new[0, 0]))
            if abs(ang) > window_deg:
                r_new = _rot2(math.copysign(window_deg, ang))
        else:
            r_new = r
        t_new = (target[keep] - p[keep] @ r_new.T).mean(axis=0)
        done = np.linalg.norm(t_new - t) < 1e-4 and abs(r_new[1, 0] - r[1, 0]) < 1e-7
        r, t = r_new, t_new
        if done:
            break
    moved = p @ r.T + t
    return math.degrees(math.atan2(r[1, 0], r[0, 0])), t, float(np.median(np.linalg.norm(moved - lookup.closest(moved), axis=1)))


def _placement_grid(xy: np.ndarray, lookup: _OutlineLookup, about: np.ndarray, window_deg: float,
                    params: StyleParams = DEFAULT_PARAMS, seed: int = 0) -> list[tuple[float, np.ndarray, float]]:
    """The coarse search of STYLIZED_MODEL 3.2: the mean wall-point distance to the outline (capped at
    reg_edge_limit_m) on a fixed grid of rotations about `about` (every ~snap_grid_step_deg across
    +-window_deg; none in translation-only mode) x shifts (every snap_grid_step_m within +-snap_max_shift_m).
    Returns its local minima (3 x 3 x 3 neighbourhood) as ICP starts (rotation deg, t of p' = R p + t, cost),
    best first, at most snap_grid_starts. A fixed grid rather than starts from the registered position only:
    the same basins are found whatever small changes a re-run of the SfM makes, so the ambiguity check sees
    every placement that fits about as well (a re-run of Baptistery E#14 found a -5.5 deg basin and missed
    a better one at -1 deg)."""
    p = np.asarray(xy, float)
    if len(p) > params.snap_grid_points:
        p = p[np.random.default_rng(seed).choice(len(p), params.snap_grid_points, replace=False)]
    rots = (np.linspace(-window_deg, window_deg, 2 * math.ceil(window_deg / params.snap_grid_step_deg) + 1)
            if window_deg > 0 else np.zeros(1))
    k = math.floor(params.snap_max_shift_m / params.snap_grid_step_m)
    offs = params.snap_grid_step_m * np.arange(-k, k + 1)
    shifts = np.stack(np.meshgrid(offs, offs, indexing="ij"), -1).reshape(-1, 2)
    cost = np.empty((len(rots), len(offs), len(offs)))
    for i, rot in enumerate(rots):
        moved = ((p - about) @ _rot2(rot).T + about)[None, :, :] + shifts[:, None, :]
        flat = moved.reshape(-1, 2)
        d = np.minimum(np.linalg.norm(flat - lookup.closest(flat), axis=1), params.reg_edge_limit_m)
        cost[i] = d.reshape(len(shifts), -1).mean(axis=1).reshape(len(offs), len(offs))
    padded = np.pad(cost, 1, constant_values=np.inf)
    n0, n1, n2 = cost.shape
    neigh = np.min([padded[1 + a:1 + a + n0, 1 + b:1 + b + n1, 1 + c:1 + c + n2]
                    for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1) if a or b or c], axis=0)
    # nowhere near the outline (every point capped) is no basin
    idx = np.argwhere((cost <= neigh) & (cost < 0.98 * params.reg_edge_limit_m))
    idx = idx[np.argsort(cost[tuple(idx.T)], kind="stable")][:params.snap_grid_starts]
    return [(float(rots[i]), about + np.array([offs[j], offs[m]]) - _rot2(rots[i]) @ about, float(cost[i, j, m]))
            for i, j, m in idx]


def _placement_gap(xy: np.ndarray, a: tuple[float, np.ndarray], b: tuple[float, np.ndarray]) -> float:
    """How far apart two placements p' = R p + t put the same (wall) points: the 90th percentile."""
    pa = xy @ _rot2(a[0]).T + a[1]
    pb = xy @ _rot2(b[0]).T + b[1]
    return float(np.percentile(np.linalg.norm(pa - pb, axis=1), 90))


def gps_depth_check(camera_xy: np.ndarray, gps_xy: np.ndarray, anchor: np.ndarray,
                    params: StyleParams = DEFAULT_PARAMS) -> dict[str, Any]:
    """Do the (snapped) cameras stand as far from the walls (anchor) as their own GPS says? ratio = median of
    |GPS - anchor| / |camera - anchor| with its standard error (1.25 x 1.4826 MAD / sqrt n); it agrees within
    max(snap_gps_scale_tol, snap_gps_sigmas standard errors). A one-sided model whose depth is off still fits
    a straight outline edge (the snap only slides it), so this is what catches a facade drawn too shallow."""
    cam = np.asarray(camera_xy, float).reshape(-1, 2)
    gps = np.asarray(gps_xy, float).reshape(-1, 2)
    dc = np.linalg.norm(cam - anchor, axis=1)
    dg = np.linalg.norm(gps - anchor, axis=1)
    ok = (dc > 1.0) & np.isfinite(dg)
    out: dict[str, Any] = {"cameras": int(ok.sum())}
    if ok.sum() < params.snap_gps_min_cameras:
        out.update(agrees=None, note="too few cameras with GPS")
        return out
    ratio = dg[ok] / dc[ok]
    rho = float(np.median(ratio))
    se = 1.2533 * 1.4826 * float(np.median(np.abs(ratio - rho))) / math.sqrt(len(ratio))
    tol = max(params.snap_gps_scale_tol, params.snap_gps_sigmas * se)
    out.update(ratio=round(rho, 4), se=round(se, 4), tol=round(tol, 4), agrees=bool(abs(rho - 1.0) <= tol),
               camera_distance_m=round(float(np.median(dc[ok])), 1), gps_distance_m=round(float(np.median(dg[ok])), 1))
    return out


def camera_tilt_check(camera_xyz: np.ndarray, walls_xy: np.ndarray, params: StyleParams = DEFAULT_PARAMS
                      ) -> dict[str, Any]:
    """Is the model level? The georegistration takes "up" from the cameras' orientations and spread, and the
    snap only turns about the vertical, so a tilt survives it: the photos then land metres too high or too low
    on walls 40-250 m away (Frauenkirche SW#14: 2.9 deg, 3-4 m), which no 2-D check sees. Photographers stand
    at about one height, so a robust plane z = a + g . (xy - c) through the unflagged cameras' centres (refit
    without cameras more than max(3 x 1.4826 MAD, 0.75 m) off it: a terrace, a tower) measures the tilt; the
    vertical shift it implies at the walls is |g . (wall - c)| (90th percentile over the wall points), with its
    standard error (height scatter floored at 0.25 m). ok = shift + snap_tilt_sigmas x se <=
    snap_max_tilt_shift_m on at least snap_tilt_min_cameras cameras. A street that really climbs that much
    reads as a tilt too: clay, the safe side."""
    cam = np.asarray(camera_xyz, float).reshape(-1, 3)
    walls = np.asarray(walls_xy, float).reshape(-1, 2)
    out: dict[str, Any] = {"cameras": int(len(cam))}
    if len(cam) < params.snap_tilt_min_cameras or not len(walls):
        out.update(ok=False, note=f"{len(cam)} unflagged cameras: too few to check the model's tilt")
        return out
    c0 = np.median(cam[:, :2], axis=0)
    x = np.column_stack([np.ones(len(cam)), cam[:, :2] - c0])
    z = cam[:, 2]
    keep = np.ones(len(cam), bool)
    for _ in range(4):
        coef = np.linalg.lstsq(x[keep], z[keep], rcond=None)[0]
        r = z - x @ coef
        mad = 1.4826 * float(np.median(np.abs(r[keep] - np.median(r[keep]))))
        new = np.abs(r) <= max(3.0 * mad, 0.75)
        if new.sum() < params.snap_tilt_min_cameras or np.array_equal(new, keep):
            break
        keep = new
    coef = np.linalg.lstsq(x[keep], z[keep], rcond=None)[0]
    r = z[keep] - x[keep] @ coef
    n = int(keep.sum())
    sigma = max(math.sqrt(float(r @ r) / max(n - 3, 1)), 0.25)
    g = coef[1:]
    d = walls - c0
    shift = float(np.percentile(np.abs(d @ g), 90))
    try:
        cov = sigma ** 2 * np.linalg.inv(x[keep].T @ x[keep])[1:, 1:]
        se = float(np.sqrt(np.percentile(np.einsum("ni,ij,nj->n", d, cov, d), 90)))
    except np.linalg.LinAlgError:       # cameras on one line: the tilt across it is unobservable
        se = math.inf
    out.update(used=n, slope_deg=round(math.degrees(math.atan(float(np.linalg.norm(g)))), 2),
               shift_m=round(shift, 2), se_m=round(se, 2) if math.isfinite(se) else None,
               ok=bool(shift + params.snap_tilt_sigmas * se <= params.snap_max_tilt_shift_m))
    return out


def unseen_points(points_xy: np.ndarray, cameras_xy: np.ndarray, rings: list[np.ndarray], depth_m: float,
                  chunk: int = 256) -> np.ndarray:
    """Per point: True when no camera can see it past the building (in plan): for every camera, the point
    moved depth_m towards it lies inside the building (outer ring minus courtyards), or the sight line from
    there to the camera crosses the outline. depth_m lets a facade sit that far inside an outline drawn
    around eaves or a plinth; a roof seen over the walls also reads as unseen here (callers judge only points
    low on the walls)."""
    pts = np.asarray(points_xy, float).reshape(-1, 2)
    cams = np.asarray(cameras_xy, float).reshape(-1, 2)
    out = np.ones(len(pts), bool)
    edges = [(np.asarray(r, float), np.roll(np.asarray(r, float), -1, axis=0)) for r in rings]
    for i in range(0, len(pts), chunk):
        p = pts[i:i + chunk]
        v = cams[None, :, :] - p[:, None, :]
        dist = np.linalg.norm(v, axis=2, keepdims=True)
        q = p[:, None, :] + v / np.maximum(dist, 1e-9) * np.minimum(depth_m, np.maximum(dist - 0.01, 0.0))
        flat = q.reshape(-1, 2)
        inside = points_in_polygon(flat, rings[0])
        for h in rings[1:]:
            inside &= ~points_in_polygon(flat, h)
        blocked = inside.reshape(len(p), len(cams))
        for a, b in edges:
            blocked |= _segments_cross(q[:, :, None, :], cams[None, :, None, :], a[None, None], b[None, None]).any(axis=2)
        out[i:i + chunk] = blocked.all(axis=1)
    return out


def edge_away_normals(rings: list[np.ndarray]) -> list[np.ndarray]:
    """Per ring, per edge: the unit normal pointing away from the building (out of the outer ring, into a
    courtyard), whatever the rings' orientation."""
    out = []
    for i, ring in enumerate(rings):
        ring = np.asarray(ring, float)
        d = np.roll(ring, -1, axis=0) - ring
        left = np.stack([-d[:, 1], d[:, 0]], 1) / np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-12)
        into = left if signed_area(ring) > 0 else -left          # into the ring's own interior
        out.append(-into if i == 0 else into)
    return out


def _rival(c: dict[str, Any], rot: float, best_fit: float, params: StyleParams = DEFAULT_PARAMS,
           cameras: bool = True) -> bool:
    """Is candidate placement c (rot, gap, cam_gap, fit, verified) a rival of the best one (rotation rot, mean
    distance best_fit)? It is when it puts the photos elsewhere (turned >= snap_ambiguity_deg, moved >=
    snap_ambiguity_m over the walls, or its cameras that far when cameras count), passes registration_check
    (unless snap_ambiguity_verified_only is off) and fits within the margin."""
    far = (abs(c["rot"] - rot) >= params.snap_ambiguity_deg or c["gap"] >= params.snap_ambiguity_m
           or (cameras and c.get("cam_gap", 0.0) >= params.snap_ambiguity_m))
    return bool(far and (c.get("verified") or not params.snap_ambiguity_verified_only)
                and c["fit"] <= best_fit * (1.0 + params.snap_ambiguity_margin) + params.snap_ambiguity_slack_m)


def snap_to_footprint(points: np.ndarray, footprint: np.ndarray, params: StyleParams = DEFAULT_PARAMS,
                      z_band: tuple[float, float] = (2.0, 20.0), max_points: int = 6000,
                      seed: int = 0, holes: list[np.ndarray] | None = None,
                      prior: SnapPrior | None = None) -> FootprintSnap:
    """STYLIZED_MODEL 3.2, the coarse step: wall-height points moved onto the footprint outline.

    1. Yaw only within the registration's own window, +-min(snap_yaw_sigmas x its reported yaw sigma (prior),
       P/2 - snap_symmetry_margin_deg) with P = outline_symmetry_period(), so a symmetric outline cannot
       alias. Narrower than snap_min_window_deg (a near-circle): the registered yaw is kept and only the
       translation is searched, and since the outline cannot check that yaw, the snap is refused unless
       snap_yaw_sigmas x the reported sigma is within snap_fixed_yaw_max_deg and moves the outline's boundary
       by at most snap_max_residual_m. Search: _placement_grid() (rotation x shift, deterministic), trimmed
       ICP from its best local minima and from the registered placement; the walls (the points within
       reg_edge_limit_m of the outline in the best placement) move by at most snap_max_shift_m.
    2. The outline may correct the scale (<= snap_max_scale, about its centroid): every candidate placement
       gets the similarity step and keeps it when that cuts its median distance by 10 % (window mode only: an
       arc of a near-circle cannot fix a scale); the candidates are then ranked by their mean point distance
       capped at reg_edge_limit_m. A best one stuck at the scale bound is refused (the model's shape, not its
       GPS scale, is off).
    3. The best placement is refused when it lies within snap_window_edge_deg of the window edge or beyond
       the shift bound, when its walls fit the outline more loosely than snap_max_fit_m, or when it is
       ambiguous: another candidate that would put the photos >= snap_ambiguity_m from the best one's (90th
       percentile over the walls), its cameras that far (median; snap_ambiguity_cameras) or turn them >=
       snap_ambiguity_deg also passes registration_check() and fits within snap_ambiguity_margin
       (+ snap_ambiguity_slack_m) of it; or the best one turned about the walls by 2 / 3.5 / 5 deg or by the
       turn that moves its cameras snap_ambiguity_m (shift re-fitted) still fits that well and moves the photos
       or the cameras that far (a flat yaw valley); or it rescaled about the walls by snap_scale_steps within
       snap_scale_gps_sigmas of the GPS fit's scale sigma (any, when unknown) does (an open scale: an L of two
       faces fits its outline 7 % off, and cameras 80 m away move 1.5 m for 2 %).
    4. A one-sided model fits the outline at any scale, so when the outline did not fix it and the GPS fit
       left it uncertain by more than snap_max_scale_sigma, the snap is refused.
    5. Tilt (prior with cameras): camera_tilt_check() of the snapped cameras' heights; a model whose tilt would
       lift or sink the photos on the walls by more than snap_max_tilt_shift_m (or cannot be checked) is
       refused. A georegistered prior without any camera is refused (nothing above could be checked).
    6. GPS (prior with cameras): gps_depth_check() of the snapped cameras. When they would stand closer to
       or farther from the walls than their own GPS says, the points and the cameras disagree (a facade
       reconstructed too shallow, a too-flat arc, a translation onto the wrong parallel face): refused.
       Rescaling about the walls instead was tried on the 10-building runs and made the placement worse
       (the distortion is not a uniform scale).
    Acceptance means the placement can be trusted; registration_check() (edge by edge, with the cameras: no
    face may carry wall points that no camera can see) is still the gate. A refused snap is reset to the
    identity (search details, tilt and GPS check stay in `search` / `gps`).
    """
    pts = np.asarray(points, float)
    fp = ensure_ccw(footprint)
    holes = [np.asarray(h, float) for h in (holes or [])]
    rings = [fp] + holes
    centroid = polygon_centroid(fp)
    radius = float(np.max(np.linalg.norm(fp - centroid, axis=1)))
    band = (pts[:, 2] >= z_band[0]) & (pts[:, 2] <= z_band[1])
    near = np.linalg.norm(pts[:, :2] - centroid, axis=1) <= radius + params.snap_max_shift_m
    sel = pts[band & near]
    if len(sel) < 50:
        return FootprintSnap(0.0, np.zeros(2), float("nan"), float("nan"), len(sel), False, "too few wall points")
    if len(sel) > max_points:
        sel = sel[np.random.default_rng(seed).choice(len(sel), max_points, replace=False)]
    xy = sel[:, :2]
    period = outline_symmetry_period(fp, holes, params)
    reported = prior is not None and prior.yaw_sigma_deg is not None and math.isfinite(prior.yaw_sigma_deg)
    window, mode, sigma = yaw_window(period, prior.yaw_sigma_deg if prior else None, params)
    search: dict[str, Any] = {"mode": mode, "window_deg": round(window, 2), "yaw_sigma_deg": round(sigma, 2),
                              "yaw_sigma_reported": reported, "symmetry_period_deg": period,
                              "max_shift_m": params.snap_max_shift_m}
    span = float(np.max(fp.max(axis=0) - fp.min(axis=0))) + 2 * (radius + 2 * params.snap_max_shift_m)
    lookup = _OutlineLookup(fp, radius + 2 * params.snap_max_shift_m, res=max(0.1, span / 3000), holes=holes)
    before = float(np.median(distance_to_rings(xy, rings)[0]))
    anchor = np.median(xy, axis=0)
    starts = _placement_grid(xy, lookup, anchor, window, params, seed) + [(0.0, np.zeros(2), float("nan"))]
    found = [_snap_icp(xy, lookup, r0, t0, window, inlier_m=params.snap_icp_inlier_m) for r0, t0, _ in starts]
    search["starts"] = len(found)

    def disp(rot: float, t: np.ndarray, s: float = 1.0, at: np.ndarray = anchor) -> np.ndarray:
        return s * _rot2(rot) @ at + t - at

    def score(rot: float, t: np.ndarray) -> float:
        mv = xy @ _rot2(rot).T + t
        return float(np.minimum(np.linalg.norm(mv - lookup.closest(mv), axis=1), params.reg_edge_limit_m).mean())

    def cluster(sols: list[tuple[float, np.ndarray, float]]) -> list[dict[str, Any]]:
        # best first by the mean point distance capped at the edge limit (the grid's own score), not by the
        # median: the median is blind to a minority of points, and a short end wall is often what tells one
        # bay of a colonnade from the next
        out: list[dict[str, Any]] = []
        for rot, t, after in sorted(sols, key=lambda f: score(f[0], f[1])):
            if not any(abs(rot - d["rot"]) < 1.0 and np.linalg.norm(disp(rot, t) - disp(d["rot"], d["t"])) < 1.0
                       for d in out):
                out.append({"rot": rot, "t": t, "after": after})
        return out

    distinct = cluster(found)

    def similarity(rot_: float, t_: np.ndarray) -> tuple[float, float, np.ndarray, float]:
        """The bounded similarity step from a rigid placement: (scale, rotation, t of p' = s R p + t, median)."""
        r_ = _rot2(rot_)
        s2, r2, t2 = _similarity_icp(xy @ r_.T + t_, lookup, centroid, params.snap_max_scale,
                                    inlier_m=params.snap_icp_inlier_m)
        moved_ = s2 * (xy @ r_.T + t_ - centroid) @ r2.T + centroid + t2
        return (s2, math.degrees(math.atan2((r2 @ r_)[1, 0], (r2 @ r_)[0, 0])),
                s2 * r2 @ t_ - s2 * r2 @ centroid + centroid + t2, float(np.median(distance_to_rings(moved_, rings)[0])))

    # the outline may also fix the scale (window mode only: an arc of a near-circle cannot): every candidate gets
    # the bounded similarity step about the outline's centroid, kept as a candidate of its own when it cuts the
    # median distance by 10 % (the Aachen reference model is ~10 % too small for its OSM outline). Rescaled and
    # rigid placements all compete: a model at the wrong scale fits no rigid placement well, so its rigid fits
    # differ in yaw without being rivals (a box 12 % too small read as ambiguous before), while a rescale that
    # turns a model that fitted as well rigidly is no better founded than the rigid fit (Frauenkirche SW#9: the
    # rescale turned it 2.7 deg, 5 deg off the reference)
    cands = []
    for d in distinct[:8]:
        versions = [{"s": 1.0, "rot": d["rot"], "t": d["t"], "after": d["after"], "stuck": False}]
        if mode == "window" and params.snap_max_scale > 1.0:
            s2, rot_s, t_s, after_s = similarity(d["rot"], d["t"])
            if after_s < 0.9 * d["after"] and abs(s2 - 1.0) > 1e-3 and abs(rot_s) <= window - params.snap_window_edge_deg:
                versions.append({"s": s2, "rot": rot_s, "t": t_s, "after": after_s,
                                 "stuck": abs(math.log(s2)) >= math.log(params.snap_max_scale) - 1e-3})
        for c in versions:
            mv = c["s"] * xy @ _rot2(c["rot"]).T + c["t"]
            c["score"] = float(np.minimum(np.linalg.norm(mv - lookup.closest(mv), axis=1), params.reg_edge_limit_m).mean())
            cands.append(c)
    merged: list[dict[str, Any]] = []
    for c in sorted(cands, key=lambda c: c["score"]):
        if not any(abs(c["rot"] - m["rot"]) < 1.0 and abs(c["s"] / m["s"] - 1.0) < 0.01
                   and np.linalg.norm(disp(c["rot"], c["t"], c["s"]) - disp(m["rot"], m["t"], m["s"])) < 1.0
                   for m in merged):
            merged.append(c)
    best = merged[0]
    # how well a placement explains the wall points: their mean distance capped at the edge limit, over the
    # points within that limit of the outline in the best placement (the median would let a third of them, a
    # whole wall, sit metres off; points that are no wall, a dome or a tree, would only dilute the comparison)
    first_moved = best["s"] * xy @ _rot2(best["rot"]).T + best["t"]
    walls_sel = np.linalg.norm(first_moved - lookup.closest(first_moved), axis=1) <= params.reg_edge_limit_m
    if walls_sel.sum() < 20:
        walls_sel[:] = True
    # how far a placement moves the walls is measured at them, not at the median of every band point (a
    # neighbour's or a dome's points turn with the model and would inflate the displacement)
    walls_xy = xy[walls_sel]
    walls_at = np.median(walls_xy, axis=0)
    rot, t, after, scale = best["rot"], best["t"], best["after"], best["s"]
    r = _rot2(rot)

    def fit_of(s_: float, rot_: float, t_: np.ndarray) -> float:
        mv = s_ * xy[walls_sel] @ _rot2(rot_).T + t_
        return float(np.minimum(np.linalg.norm(mv - lookup.closest(mv), axis=1), params.reg_edge_limit_m).mean())

    def gap_of(s_: float, rot_: float, t_: np.ndarray) -> float:
        a_ = s_ * walls_xy @ _rot2(rot_).T + t_
        b_ = scale * walls_xy @ r.T + t
        return float(np.percentile(np.linalg.norm(a_ - b_, axis=1), 90))

    # the unflagged cameras, moved with each placement: how far a rival moves them, whether every face with
    # points has one in front of it, and the tilt check
    cam_xyz = None
    if prior is not None and prior.camera_xyz is not None and len(prior.camera_xyz):
        cam_xyz = np.asarray(prior.camera_xyz, float).reshape(-1, 3)
    elif prior is not None and prior.camera_xy is not None and len(prior.camera_xy):
        cam_xyz = np.column_stack([np.asarray(prior.camera_xy, float).reshape(-1, 2), np.zeros(len(prior.camera_xy))])
    use_cams = cam_xyz is not None and params.snap_ambiguity_cameras

    def cams_of(s_: float, rot_: float, t_: np.ndarray) -> np.ndarray | None:
        return None if cam_xyz is None else s_ * cam_xyz[:, :2] @ _rot2(rot_).T + t_

    def cam_gap_of(s_: float, rot_: float, t_: np.ndarray) -> float:
        if not use_cams:
            return 0.0
        return float(np.median(np.linalg.norm(cams_of(s_, rot_, t_) - cams_of(scale, rot, t), axis=1)))

    def moves(gap: float, cam_gap: float) -> bool:
        return gap >= params.snap_ambiguity_m or cam_gap >= params.snap_ambiguity_m

    def within_margin(fit: float) -> bool:
        return fit <= best_fit * (1.0 + params.snap_ambiguity_margin) + params.snap_ambiguity_slack_m

    best_fit = best["fit"] = fit_of(scale, rot, t)
    merged = merged[:10]
    for c in merged:
        c["fit"], c["gap"] = fit_of(c["s"], c["rot"], c["t"]), gap_of(c["s"], c["rot"], c["t"])
        c["cam_gap"] = cam_gap_of(c["s"], c["rot"], c["t"])
        if c["after"] <= params.snap_max_residual_m:
            c["verified"] = bool(registration_check(np.column_stack(
                [c["s"] * xy @ _rot2(c["rot"]).T + c["t"], c["s"] * sel[:, 2]]), rings, params,
                cameras_xy=cams_of(c["s"], c["rot"], c["t"]))["accepted"])
    search["alternatives"] = [{"rotation_deg": round(c["rot"], 2), "scale": round(c["s"], 4),
                               "walls_moved_m": round(float(np.linalg.norm(disp(c["rot"], c["t"], c["s"], at=walls_at))), 2),
                               "gap_m": round(c["gap"], 2), "cameras_moved_m": round(c["cam_gap"], 2),
                               "median_m": round(c["after"], 3), "fit_m": round(c["fit"], 3),
                               "verified": c.get("verified")} for c in merged[:6]]
    reason = ""
    if best["stuck"]:
        # stuck at the bound: the outline asks for a larger rescale than a GPS scale error explains
        reason = f"the outline asks for a rescale beyond {params.snap_max_scale:.2f}x (the model's shape is off)"
    # a placement that fits as well as the best one but would put the photos elsewhere on the walls (another
    # bay, another face, a few degrees turned) or the cameras elsewhere cannot be told apart from it
    ambiguous = ""
    for c in merged[1:]:
        if _rival(c, rot, best_fit, params, use_cams):
            ambiguous = (f"ambiguous: another placement {c['gap']:.1f} m / {c['rot'] - rot:+.1f} deg away (cameras"
                         f" {c['cam_gap']:.1f} m) fits as well (mean distance {c['fit']:.2f} vs {best_fit:.2f} m)")
            break
    # a flat valley rather than a second basin: the best placement turned about the walls (shift re-fitted)
    # fits about as well and puts the photos elsewhere on them. Real facades (Frauenkirche SW, 20-30 photos)
    # sat in such valleys 3-4 deg off the reference, while the right ones rose past the margin within 2.5 deg.
    # Judged by how far the photos (and the cameras) move: an arc of an octagon turned about its centre hardly
    # moves the photos, while cameras 80 m away move 1.4 m per degree; a yaw left open by
    # snap_ambiguity_turn_deg is open whatever the walls' size (tiny walls hardly move at all).
    turn = params.snap_ambiguity_turn_deg
    cam_dist = None
    if use_cams:
        cam_dist = float(np.median(np.linalg.norm(cams_of(scale, rot, t) - (scale * _rot2(rot) @ walls_at + t), axis=1)))
        search["camera_distance_m"] = round(cam_dist, 1)
    if mode == "window" and turn > 0:
        steps = {x for x in (2.0, 3.5, turn) if x <= turn}
        if cam_dist is not None and cam_dist > 1.0:
            # the turn that moves the cameras snap_ambiguity_m about the walls
            steps.add(round(min(2.0, math.degrees(params.snap_ambiguity_m / cam_dist)), 2))
        profile = []
        for dd in sorted({s_ * x for x in steps for s_ in (-1.0, 1.0)}):
            if abs(rot + dd) > window:
                continue
            t0 = t + scale * (_rot2(rot) @ walls_at - _rot2(rot + dd) @ walls_at)
            rot_d, t_d, _ = _snap_icp(xy, lookup, rot + dd, t0, 0.0, scale=scale, inlier_m=params.snap_icp_inlier_m)
            fit_d, gap, cgap = fit_of(scale, rot_d, t_d), gap_of(scale, rot_d, t_d), cam_gap_of(scale, rot_d, t_d)
            profile.append({"turn_deg": dd, "fit_m": round(fit_d, 3), "gap_m": round(gap, 2),
                            "cameras_moved_m": round(cgap, 2)})
            if not ambiguous and (moves(gap, cgap) or abs(dd) >= turn) and within_margin(fit_d):
                ambiguous = (f"ambiguous: the walls turned {dd:+.1f} deg about themselves fit as well (mean distance"
                             f" {fit_d:.2f} vs {best_fit:.2f} m) and move the photos {gap:.1f} m, the cameras"
                             f" {cgap:.1f} m: the outline does not fix the yaw")
        search["turn_profile"] = {"best_fit_m": round(best_fit, 3), "turns": profile}
    # the scale: the best placement rescaled about the walls (shift re-fitted). A rescale the GPS fit makes
    # implausible is no rival; one that fits as well and moves the photos or the cameras leaves the scale open
    # (an L of two faces fits its outline 7 % off: the photos land 2 m off at the far end)
    gps_scale = None if prior is None or prior.scale_sigma is None else float(prior.scale_sigma)
    scale_profile = []
    for f in sorted({sg * x for x in params.snap_scale_steps for sg in (-1.0, 1.0)}):
        if gps_scale is not None and abs(f) > params.snap_scale_gps_sigmas * gps_scale:
            continue
        s_f = scale * (1.0 + f)
        t0 = (1.0 + f) * t - f * (scale * r @ walls_at + t)        # about the snapped walls
        rot_f, t_f, _ = _snap_icp(xy, lookup, rot, t0, 0.0, scale=s_f, inlier_m=params.snap_icp_inlier_m)
        fit_f, gap, cgap = fit_of(s_f, rot_f, t_f), gap_of(s_f, rot_f, t_f), cam_gap_of(s_f, rot_f, t_f)
        scale_profile.append({"rescale": round(f, 3), "fit_m": round(fit_f, 3), "gap_m": round(gap, 2),
                              "cameras_moved_m": round(cgap, 2)})
        if not ambiguous and moves(gap, cgap) and within_margin(fit_f):
            ambiguous = (f"ambiguous: the walls rescaled {f:+.0%} fit as well (mean distance {fit_f:.2f} vs"
                         f" {best_fit:.2f} m) and move the photos {gap:.1f} m, the cameras {cgap:.1f} m: neither the"
                         f" outline nor the GPS fit (scale sigma {'unknown' if gps_scale is None else f'{gps_scale:.0%}'})"
                         " fixes the scale")
    search["scale_profile"] = scale_profile
    if mode == "translation-only":
        bound = params.snap_yaw_sigmas * sigma
        moves_outline = math.radians(bound) * radius
        search["fixed_yaw"] = {"bound_deg": round(bound, 2), "moves_outline_m": round(moves_outline, 2)}
    if not reason and after > params.snap_max_residual_m:
        reason = f"walls still {after:.1f} m off the footprint after snapping"
    elif not reason and mode == "window" and abs(rot) > window - params.snap_window_edge_deg:
        reason = f"the best fit lies at the edge of the yaw window (+-{window:.1f} deg)"
    elif not reason and np.linalg.norm(disp(rot, t, scale, at=walls_at)) > params.snap_max_shift_m:
        reason = f"the best fit moves the walls beyond the shift bound ({params.snap_max_shift_m:.0f} m)"
    elif not reason and best_fit > params.snap_max_fit_m:
        reason = (f"the walls follow the outline only to {best_fit:.2f} m on average (more than"
                  f" {params.snap_max_fit_m:.2f} m): too loose to fix the yaw")
    elif not reason and ambiguous:
        reason = ambiguous
    elif not reason and mode == "translation-only" and (
            not reported or bound > params.snap_fixed_yaw_max_deg or moves_outline > params.snap_max_residual_m):
        # the outline fits itself at any rotation, so nothing checks the registered yaw
        reason = ("yaw unverifiable (near-rotationally symmetric outline): "
                  + (f"the registered yaw is only known to +-{bound:.1f} deg, which moves the photos up to"
                     f" {moves_outline:.1f} m around the outline" if reported else "the registration reports no yaw sigma"))
    if (not reason and scale == 1.0 and prior is not None and prior.scale_sigma is not None
            and prior.scale_sigma > params.snap_max_scale_sigma):
        # a facade fits a straight outline edge at any scale, so the GPS fit's scale is all there is
        reason = (f"the georegistration leaves the scale uncertain by +-{prior.scale_sigma:.0%}"
                  f" (more than {params.snap_max_scale_sigma:.0%}): photos could land on the wrong part of a wall")
    if not reason and prior is not None and prior.georegistered and cam_xyz is None:
        # a pipeline model always carries its cameras (snap_prior); without them neither the cameras' displacement
        # nor the tilt can be checked
        reason = "the model carries no camera positions: how far a rival placement moves its cameras is unknown"
    tilt = None
    if prior is not None and prior.camera_xyz is not None:
        moved_cams = np.column_stack([cams_of(scale, rot, t), scale * cam_xyz[:, 2]])
        tilt = camera_tilt_check(moved_cams, scale * walls_xy @ r.T + t, params)
        search["tilt"] = tilt
        if not reason and not tilt["ok"]:
            reason = (tilt["note"] if "note" in tilt else
                      f"the model is tilted: its cameras' heights rise {tilt['slope_deg']:.1f} deg across it, which"
                      f" lifts or sinks the photos {tilt['shift_m']:.1f} m (+-{tilt['se_m'] if tilt['se_m'] is not None else 'inf'}"
                      f" m) on the walls")
    gps = None
    if prior is not None and prior.camera_xy is not None and prior.gps_xy is not None and len(prior.camera_xy):
        moved = scale * xy @ r.T + t
        on = distance_to_rings(moved, rings)[0] <= params.reg_edge_limit_m
        walls = np.median(moved[on], axis=0) if on.sum() >= 20 else np.median(moved, axis=0)
        gps = gps_depth_check(scale * np.asarray(prior.camera_xy, float) @ r.T + t, prior.gps_xy, walls, params)
        if not reason and gps["agrees"] is False:
            reason = (f"the cameras would stand {abs(gps['ratio'] - 1.0):.0%} "
                      f"{'closer to' if gps['ratio'] > 1 else 'farther from'} the walls than their own GPS says"
                      f" (median over {gps['cameras']} cameras, tolerance {gps['tol']:.0%})")
    snap = FootprintSnap(rot, t, before, after, len(xy), not reason, reason, scale=scale, search=search, gps=gps)
    if not snap.accepted:
        snap.rotation_deg, snap.shift, snap.scale = 0.0, np.zeros(2), 1.0
    return snap


def _similarity_icp(xy: np.ndarray, lookup: _OutlineLookup, centre: np.ndarray, max_scale: float,
                    iters: int = 60, inlier_m: float = DEFAULT_PARAMS.snap_icp_inlier_m
                    ) -> tuple[float, np.ndarray, np.ndarray]:
    """Trimmed 2-D similarity ICP about `centre` with the scale clamped to [1/max_scale, max_scale]:
    p' = s R (p - centre) + centre + t. Returns (s, R, t)."""
    s, r, t = 1.0, np.eye(2), np.zeros(2)
    for _ in range(iters):
        moved = s * (xy - centre) @ r.T + centre + t
        target = lookup.closest(moved)
        d = np.linalg.norm(moved - target, axis=1)
        keep = _icp_inliers(d, inlier_m)
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
                       z_band: tuple[float, float] = (2.0, 20.0), max_dist_m: float = 10.0,
                       cameras_xy: np.ndarray | None = None) -> dict[str, Any]:
    """Do the (snapped) points sit on the footprint edge by edge? Photo content is only published when
    they do: a single global median is dominated by the most photographed facade, so a shifted, rotated
    or too large outline can pass it (a 1.25x outline did, at 0.21 m).

    Wall-height points outside the outline or at most 3 m inside it (not roofs) are assigned to their
    nearest outline edge (within max_dist_m). Edges carrying at
    least reg_support_frac of the median supported edge's point count vote, one vote each (point-count
    weighting would again let the best facade decide); an edge is verified when its points' median
    distance is under reg_edge_limit_m. Accepted when reg_min_share of the voting edges are verified and
    the verified edges are not all parallel (>= 30 degrees apart). Given the (snapped) cameras, a voting edge
    with no camera at least reg_camera_front_m on its outer side is checked for points no camera can see
    (unseen_points, a facade may sit reg_edge_limit_m inside the outline): when >= 20 and >= 30 % of its points
    within reg_edge_limit_m are such points low on the wall (lower half of the band's height range: not a roof
    seen over the walls), every camera would have seen them through the massing (a slide onto the far side, a
    neighbour's facade moved onto this one), and the placement is refused outright.
    """
    out: dict[str, Any] = {"accepted": False, "reason": "", "edges": [], "verified_share": 0.0}
    if points is None or not len(points):
        out["reason"] = "no SfM points"
        return out
    pts = np.asarray(points, float)
    band = (pts[:, 2] >= z_band[0]) & (pts[:, 2] <= z_band[1])
    bp = pts[band]
    if len(bp) > 60000:
        bp = bp[np.random.default_rng(0).choice(len(bp), 60000, replace=False)]
    xy, z = bp[:, :2], bp[:, 2]
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
    cams = None if cameras_xy is None or not len(cameras_xy) else np.asarray(cameras_xy, float).reshape(-1, 2)
    normals = edge_away_normals(rings) if cams is not None else None
    z_mid = z_band[0] + 0.5 * (float(np.percentile(z[near], 97)) - z_band[0]) if near.any() else z_band[1]
    for e in edges:
        e["votes"] = e in voting
        e["verified"] = bool(e["votes"] and e["median_m"] <= params.reg_edge_limit_m)
        if cams is not None:
            a = np.asarray(rings[e["ring"]][e["edge"]], float)
            e["cameras_in_front"] = int(((cams - a) @ normals[e["ring"]][e["edge"]] >= params.reg_camera_front_m).sum())
            if e["votes"] and not e["cameras_in_front"]:
                on = near & (ring_of == e["ring"]) & (edge_of == e["edge"]) & (d <= params.reg_edge_limit_m)
                low = np.flatnonzero(on & (z <= z_mid))
                if len(low) > 400:
                    low = np.random.default_rng(0).choice(low, 400, replace=False)
                unseen = unseen_points(xy[low], cams, rings, params.reg_edge_limit_m) if len(low) else np.zeros(0, bool)
                frac = float(unseen.mean()) if len(low) else 0.0
                n_unseen = frac * int((on & (z <= z_mid)).sum())
                e["unseen_low_points"] = int(round(n_unseen))
                e["behind_every_camera"] = bool(n_unseen >= max(20, 0.3 * on.sum()))
                if e["behind_every_camera"]:
                    e["verified"] = False
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
    behind = [e for e in voting if e.get("behind_every_camera")]
    if behind:
        out["reason"] = (f"{len(behind)} outline edge(s) carry wall points that no camera can see past the building"
                         f" (edge {behind[0]['ring']}:{behind[0]['edge']}): they cannot lie there")
    elif len(voting) < 2:
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
    # structure-first parts (crowd3d.structure): cap "faces" = the part's roof primitive, a list of planar faces
    # (objects with .polys [(k, 3) ENU polygons sharing a plane] and .normal); the shape is a fact, not a fit
    roof_faces: list = field(default_factory=list)
    part: str | None = None                            # the OSM part id
    roof_shape: str | None = None                      # the roof primitive built (gabled, dome, ...)
    structure: dict = field(default_factory=dict)      # per-part geometry record for the GLB extras
    stub: bool = False                                 # a 'height pending' stub (D13, height default-style): every face
                                                       # is clay, whatever points or photos say (it is not the building)

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
        if self.cap == "faces":
            return len(self.roof_faces)
        return len(self.polygon) if self.cap == "pyramid" else 1

    def _faces_z(self, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """For a "faces" cap: (face index, height) of the topmost roof face above each (x, y); -1 / NaN where
        no face is (faces seen edge-on from above, e.g. an onion's bulge, never count)."""
        face = np.full(len(xy), -1, int)
        z = np.full(len(xy), -np.inf)
        for k, f in enumerate(self.roof_faces):
            n = np.asarray(f.normal, float)
            if abs(n[2]) < 0.05:
                continue
            for poly in f.polys:
                lo, hi = poly[:, :2].min(axis=0), poly[:, :2].max(axis=0)
                box = (xy[:, 0] >= lo[0] - 1e-9) & (xy[:, 0] <= hi[0] + 1e-9) & (xy[:, 1] >= lo[1] - 1e-9) & (xy[:, 1] <= hi[1] + 1e-9)
                if not box.any():
                    continue
                idx = np.nonzero(box)[0]
                inside = points_in_polygon(xy[idx], poly[:, :2])
                idx = idx[inside]
                if not len(idx):
                    continue
                p0 = poly[0]
                zz = p0[2] - ((xy[idx, 0] - p0[0]) * n[0] + (xy[idx, 1] - p0[1]) * n[1]) / n[2]
                up = zz > z[idx]
                z[idx[up]], face[idx[up]] = zz[up], k
        return face, np.where(face >= 0, z, np.nan)

    def face_of(self, xy: np.ndarray) -> np.ndarray:
        """Cap face index of each (x, y): the pyramid sector (edge k -> apex), the topmost roof face of a
        "faces" cap (the nearest face centre where none covers it), or 0 for a flat cap."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.cap == "faces":
            face, _ = self._faces_z(xy)
            miss = face < 0
            if miss.any() and self.roof_faces:
                cen = np.array([np.vstack(f.polys)[:, :2].mean(axis=0) for f in self.roof_faces])
                face[miss] = np.linalg.norm(xy[miss, None, :] - cen[None], axis=2).argmin(axis=1)
            return face
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
                "moved_edges": sorted([list(e) for e in self.moved_edges]),
                **({"part": self.part, "roof_shape": self.roof_shape, "stub": self.stub} if self.part is not None else {})}

    def roof_height(self, xy: np.ndarray, fitted: bool = False) -> np.ndarray:
        """Height of this block's cap above each (x, y) inside its outline. fitted=True: the fitted pyramid
        everywhere, also where a face was left as a neutral flat cap."""
        xy = np.asarray(xy, float).reshape(-1, 2)
        if self.cap == "faces":
            _, z = self._faces_z(xy)
            return np.where(np.isfinite(z), z, self.z1)
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
    # structure-first parts: a wall whose top follows the roof (a gable end, a skillion side) carries its top
    # profile (m, 2) of (u along the wall from a, z), u from 0 to its length; z1 is then the profile's maximum and
    # the texture rectangle above the profile is not wall (never observed, never shown)
    profile: Any = None
    part: str | None = None              # the OSM part id

    def top_at(self, u: np.ndarray) -> np.ndarray:
        """The wall's top above positions u (metres from a)."""
        u = np.asarray(u, float)
        if self.profile is None:
            return np.full(u.shape, self.z1)
        return np.interp(u, self.profile[:, 0], self.profile[:, 1])

    def inside_grid(self, w: int, h: int) -> np.ndarray:
        """(h, w) bool: which cells of a w x h grid over the wall's rectangle (row 0 at z1) lie on the wall face."""
        if self.profile is None:
            return np.ones((h, w), bool)
        u = (np.arange(w) + 0.5) / w * self.length
        z = self.z1 - (np.arange(h) + 0.5) / h * (self.z1 - self.z0)
        return z[:, None] <= self.top_at(u)[None, :] + 1e-6

    def uz_polygon(self, u0: float = 0.0, u1: float | None = None) -> np.ndarray:
        """The face between u0 and u1 as a polygon in (u, z): bottom-left, bottom-right, top-right, the profile
        points between (descending u), top-left -- the first four are the viewer's frame corners."""
        u1 = self.length if u1 is None else u1
        top0, top1 = float(self.top_at(u0)), float(self.top_at(u1))
        mid = []
        if self.profile is not None:
            mid = [(float(u), float(z)) for u, z in self.profile[::-1] if u0 + 1e-6 < u < u1 - 1e-6]
        return np.array([(u0, self.z0), (u1, self.z0), (u1, top1)] + mid + [(u0, top0)], float)

    @property
    def face_area(self) -> float:
        """Area of the wall face itself (below its profile)."""
        if self.profile is None:
            return self.area
        return abs(signed_area(self.uz_polygon()))

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
            if w.profile is not None:
                tris += list(_wall_face_triangles(w))
                continue
            a0, b0 = np.r_[w.a, w.z0], np.r_[w.b, w.z0]
            a1, b1 = np.r_[w.a, w.z1], np.r_[w.b, w.z1]
            tris += [(a0, b0, b1), (a0, b1, a1)]
        for pr in self.prisms:
            p = pr.polygon
            if pr.part is not None:
                tris += list(_part_cap_triangles(pr))         # roof faces (+ a raised part's underside)
                continue
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


def _planar_triangles(poly: np.ndarray, normal: np.ndarray) -> list[np.ndarray]:
    """Triangles (3, 3) of a planar 3-D polygon, wound counter-clockwise about `normal`."""
    p = np.asarray(poly, float)
    n = np.asarray(normal, float) / max(np.linalg.norm(normal), 1e-12)
    t = np.cross([0.0, 0.0, 1.0], n) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    t = t - (t @ n) * n
    t /= max(np.linalg.norm(t), 1e-12)
    uv = np.stack([p @ t, p @ np.cross(n, t)], 1)
    order = np.arange(len(p))
    if signed_area(uv) < 0:
        order, uv = order[::-1], uv[::-1]
    out = []
    for i, j, k in triangulate_polygon(uv):
        tri = p[order[[i, j, k]]]
        if np.linalg.norm(np.cross(tri[1] - tri[0], tri[2] - tri[0])) > 1e-10:
            out.append(tri)
    return out


def _faces_triangles(faces: list) -> np.ndarray:
    """(t, 3, 3) triangles of planar faces (objects with .polys and .normal)."""
    tris = [t for f in faces for poly in f.polys for t in _planar_triangles(poly, f.normal)]
    return np.array(tris, float).reshape(-1, 3, 3)


def _part_cap_triangles(pr: Prism) -> np.ndarray:
    """A structure part's roof (its faces, or a flat cap) plus, when raised, its underside; cached on the prism."""
    cached = getattr(pr, "_tri_cache", None)
    if cached is not None:
        return cached
    tris = []
    if pr.cap == "faces":
        tris.append(_faces_triangles(pr.roof_faces))
    else:
        ring, tri = triangulate_with_holes(pr.polygon, pr.holes)
        tris.append(np.array([(np.r_[ring[i], pr.z1], np.r_[ring[j], pr.z1], np.r_[ring[k], pr.z1]) for i, j, k in tri],
                             float).reshape(-1, 3, 3))
    if pr.z0 > 0.05:
        ring, tri = triangulate_with_holes(pr.polygon, pr.holes)
        tris.append(np.array([(np.r_[ring[i], pr.z0], np.r_[ring[k], pr.z0], np.r_[ring[j], pr.z0]) for i, j, k in tri],
                             float).reshape(-1, 3, 3))
    out = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
    pr._tri_cache = out
    return out


def _wall_face_triangles(w: Wall) -> np.ndarray:
    """A profiled wall's face (below its top profile) as ENU triangles facing outwards; cached on the wall."""
    cached = getattr(w, "_tri_cache", None)
    if cached is not None:
        return cached
    uz = w.uz_polygon()
    pts = np.c_[w.a[None, :] + uz[:, :1] * w.tangent[None, :], uz[:, 1]]
    out = np.array([(pts[i], pts[j], pts[k]) for i, j, k in triangulate_polygon(uz)], float).reshape(-1, 3, 3)
    w._tri_cache = out
    return out


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
    """Points on a block's (fitted) cap: from 1.5 m below it to step_m above it. A structure part's roof is a map
    fact, not fitted to the points: only points within 1.5 m of it count (docs/STRUCTURE_FIRST.md §2.5), so a roof
    the map has too low (a disputed part) gets no colour from the points standing above it."""
    roof = pr.roof_height(points[:, :2], fitted=True)
    above = 1.5 if pr.part is not None else params.step_m
    return (points[:, 2] >= roof - 1.5) & (points[:, 2] <= roof + above)


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
    if points is not None and len(points) and not pr.stub:       # a stub's cap is a placeholder: no point "sees" it
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


_FACT_SOURCES = (PROV_S3DB, PROV_LEVELS, PROV_ROOF_ANGLE, PROV_OSM_HEIGHT, PROV_GROUND, PROV_CURATOR, PROV_FITTED,
                 PROV_FITTED_AUTO, PROV_CORRECTED_AUTO, PROV_CORRECTED_LB, PROV_CORRECTED_UB)


def part_record(built: Any, frame_adoption: dict[str, Any] | None = None,
                corrections: dict[tuple[str, str], dict[str, Any]] | None = None,
                pose: dict[str, Any] | None = None) -> dict[str, Any]:
    """The compact geometry record of a structure part for its GLB nodes: part id, geometry state, roof built
    (with its tag, height source and any fallback) and every parameter as [value, state, source]. frame_adoption:
    the active D15 frame adoption (crowd3d.judge.adopt_frame's record) when the outline is photo-fitted (auto): its
    id, scale, shift and the map's outline width travel with every node (the viewer shows the old outline)."""
    spec = built.spec
    roof = {"shape": built.roof.shape, "tag": spec.roof_tag, "height_source": spec.params["roof_height"].source}
    if built.roof.meta.get("fallback_from"):
        roof["fallback_from"] = built.roof.meta["fallback_from"]
        roof["reason"] = built.roof.meta.get("reason", "")
    rec = {"part": spec.id, "kind": spec.kind, "state": spec.state(), "roof": roof,
           "params": {k: [round(float(v.value), 3), v.state, v.source] for k, v in spec.params.items()}}
    # D15: parameters changed automatically by >= 2 agreeing photo channels (or carried with an adopted parent), with
    # the map's value beside them (the viewer says 照片量出（自动采用） and shows the old value)
    adopted = [{"param": k, "map": round(float(v.map_value), 3), "value": round(float(v.value), 3),
                **({"carried": True} if not any(e.get("kind") == "adoption" for e in v.evidence) else {})}
               for k, v in spec.params.items() if v.source == PROV_FITTED_AUTO and v.map_value is not None]
    if adopted:
        rec["adopted"] = adopted
    if spec.outline_source == PROV_FITTED_AUTO:
        fa = frame_adoption or {}
        val = fa.get("value") or {}
        rec["outline_adopted"] = {"adoption": fa.get("id"), "k": val.get("k"), "t": val.get("t"),
                                  "width_map_m": fa.get("width_map_m"), "width_m": fa.get("width_m")}
    # D25: values strong photo evidence corrected (照片纠正地图), with the map's value, what was adopted (the photo value
    # or only its guaranteed bound) and the box's range; a block's 'at least' height; the stack's tilt
    corr = []
    for k, v in spec.params.items():
        if v.source not in (PROV_CORRECTED_AUTO, PROV_CORRECTED_LB, PROV_CORRECTED_UB) or v.map_value is None:
            continue
        r = (corrections or {}).get((spec.id, k)) or {}
        corr.append({"param": k, "map": round(float(v.map_value), 3), "value": round(float(v.value), 3),
                     "adopted": r.get("adopted") or ("carried" if not r else None), "record": r.get("id"),
                     "kind": r.get("kind"), **({"range": [r.get("LB"), r.get("UB")]} if r.get("LB") is not None else {}),
                     **({"photo_value": r.get("V")} if r.get("V") is not None else {}),
                     **({"carried": True} if not r else {})})
    if corr:
        rec["corrected"] = corr
    if pose:
        rec["tilt"] = {"tilt_deg": pose.get("tilt_deg"), "azimuth_deg": pose.get("azimuth_deg"),
                       "record": pose.get("adoption")}
    return rec


def active_frame_adoption(structure: Any) -> dict[str, Any] | None:
    """The active D15 frame adoption of a structure (StructureModel.adoptions), or None."""
    return next((a for a in getattr(structure, "adoptions", None) or []
                 if a.get("kind") == "frame" and a.get("active", True)), None)


def is_stub_part(spec: Any) -> bool:
    """A structure part whose height is the 3 m 'height pending' placeholder (D13): source default-style, state stub."""
    h = spec.params.get("height")
    return h is not None and (h.source == PROV_DEFAULT or h.state == "stub")


def stub_part_ids(structure: Any) -> set[str]:
    return {p.id for p in structure.specs if is_stub_part(p)} if structure is not None else set()


def build_massing_from_structure(structure: Any, points: np.ndarray | None, params: StyleParams = DEFAULT_PARAMS) -> Massing:
    """Massing from crowd3d.structure parts (docs/STRUCTURE_FIRST.md, structure first): every built part is a prism
    whose cap is its roof primitive (cap "faces"; a flat roof uses the flat cap) and every outline edge a wall from
    the part's base up to the roof boundary (a gable end carries its top profile). Shapes and heights are facts
    (or recorded conventions) from OSM, never fitted to the points: the points only say which roof faces were
    seen (coverage decides colour, not shape: a pitched face nobody saw stays pitched, in clay). A wall's height
    source is its part's height source, or the roof height's when the eave depends on a convention. A 'height
    pending' stub (D13) is a placeholder, not the building: none of its faces is ever observed (stub = True)."""
    t0 = time.perf_counter()
    built = list(structure.built)
    idx_of = {b.id: i for i, b in enumerate(built)}
    fa = active_frame_adoption(structure)
    corrections = {(a.get("part"), a.get("param")): a for a in getattr(structure, "adoptions", None) or []
                   if str(a.get("kind", "")).startswith("d25-") and a.get("active", True)}
    pose = getattr(structure, "pose", None)
    prisms: list[Prism] = []
    for i, b in enumerate(built):
        spec = b.spec
        flat = b.roof.shape == "flat"
        hs = spec.params["height"].source
        rs = spec.params["roof_height"].source
        wall_src = hs if (flat or rs in _FACT_SOURCES) else rs
        pr = Prism(i, idx_of.get(spec.parent) if spec.parent else None, 0, ensure_ccw(np.asarray(b.outer, float)),
                   float(b.z0), float(b.eave), spec.outline_source, wall_src, cap="flat" if flat else "faces",
                   cap_source=hs if flat else rs, holes=[np.asarray(h, float) for h in b.holes],
                   roof_faces=[] if flat else list(b.roof.faces), part=b.id, roof_shape=b.roof.shape,
                   structure=part_record(b, fa, corrections, pose), stub=is_stub_part(spec))
        prisms.append(pr)
    for pr in prisms:
        d, p = 0, pr.parent
        while p is not None and d < 32:
            d, p = d + 1, prisms[p].parent
        pr.depth = d
    walls: list[Wall] = []
    for pr, b in zip(prisms, built):
        for wf in b.walls:
            walls.append(Wall(len(walls), pr.index, wf.edge, np.asarray(wf.a, float), np.asarray(wf.b, float),
                              float(wf.z0), float(wf.z1), pr.outline_source, pr.height_source, ring=wf.ring,
                              profile=None if wf.flat_top else np.asarray(wf.profile, float).copy(), part=b.id))
    pts = None if points is None or not len(points) else np.asarray(points, float)
    grids: dict[int, RoofGrid] = {}
    for pr in prisms:
        g = roof_grid(pr, [c for c in prisms if c.parent == pr.index], pts, params)
        pr.face_coverage = [g.coverage(k) for k in range(pr.num_faces())]
        grids[pr.index] = g
    ground = [(pr.z1, abs(signed_area(pr.polygon))) for pr in prisms if pr.z0 <= 0.5]
    base = _weighted_quantile(np.array([g[0] for g in ground]), np.array([g[1] for g in ground]), 0.5) if ground else \
        (max(pr.z1 for pr in prisms) if prisms else params.default_height_m)
    report = {"source": "osm-s3db-parts", "structure": dict(structure.report),
              "base_height": {"m": round(float(base), 2), "source": "area-weighted median eave of the ground parts"},
              "prisms": [p.to_meta() for p in prisms], "num_walls": len(walls),
              "profiled_walls": sum(1 for w in walls if w.profile is not None),
              "seconds": round(time.perf_counter() - t0, 3)}
    return Massing(prisms, walls, report, grids)


# ---------------------------------------------------------------- observation


def _ray_hits(origin: np.ndarray, targets: np.ndarray, tris: np.ndarray, t_max: float = 0.995) -> np.ndarray:
    """For rays origin -> target: does any triangle cut the segment before the target? (Moller-Trumbore)."""
    if len(tris) == 0 or len(targets) == 0:
        return np.zeros(len(targets), dtype=bool)
    # only triangles whose box meets the box of all the segments can cut one (exact, just fewer to test)
    lo = np.minimum(origin, targets.min(axis=0)) - 1e-6
    hi = np.maximum(origin, targets.max(axis=0)) + 1e-6
    near = (tris.max(axis=1) >= lo).all(axis=1) & (tris.min(axis=1) <= hi).all(axis=1)
    tris = tris[near]
    if len(tris) == 0:
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
        on_face = wall.inside_grid(gw, gh).reshape(-1)       # a gable end's rectangle above its profile is not wall
        n_face = max(int(on_face.sum()), 1)
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
                   & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)) & on_face
            if vis.sum() / n_face < params.min_wall_visible_fraction * 0.5:
                continue
            idx = np.nonzero(vis)[0]
            vis[idx[_ray_hits(cam.center, grid[idx], tris)]] = False
            frac = float(vis.sum() / n_face)
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
        if massing.prisms[wall.prism].stub:
            ob.reason = "a 'height pending' stub (D13): a placeholder, always clay"
        elif len(good) < params.min_wall_cameras:
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
    sx = np.clip((uv[:, 0] / cell).astype(int), 0, gw - 1)
    sy = np.clip((uv[:, 1] / cell).astype(int), 0, gh - 1)
    if params.occluder_tube_m > 0:
        # metric footprint: an occluder point hides a disc of occluder_tube_m around it (in cells at its depth,
        # at least one), and a sample is hidden when >= occluder_min_count points' discs cover it -- a zoomed
        # photo sees a sparse porch or railing in front of the wall as audit_model's 0.3 m tube does
        pzo = pz[ok].astype(np.float32)
        rad = np.clip(np.ceil(cam.fx * params.occluder_tube_m / np.maximum(pzo, 0.3) / cell), 1, 8).astype(int)
        total = np.zeros((gh, gw), np.float32)
        zb = np.full((gh, gw), np.inf, np.float32)
        for r in np.unique(rad):
            sel = rad == r
            zr = np.full(gh * gw, np.inf, np.float32)
            np.minimum.at(zr, flat[sel], pzo[sel])
            cnt = np.bincount(flat[sel], minlength=gh * gw).reshape(gh, gw).astype(np.float32)
            k = 2 * int(r) + 1
            total += cv2.boxFilter(cnt, -1, (k, k), normalize=False, borderType=cv2.BORDER_CONSTANT)
            zb = np.minimum(zb, cv2.erode(zr.reshape(gh, gw), np.ones((k, k), np.uint8)))
        zb[total < params.occluder_min_count] = np.inf
        return zb[sy, sx] < depth - 0.05
    zb = np.full(gh * gw, np.inf, np.float32)
    np.minimum.at(zb, flat, pz[ok].astype(np.float32))
    zb[np.bincount(flat, minlength=gh * gw) < params.occluder_min_count] = np.inf
    zb = cv2.erode(zb.reshape(gh, gw), np.ones((3, 3), np.uint8))
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
    on_face = wall.inside_grid(w, h).reshape(-1)             # texels above a gable end's profile are not wall
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
              & (uv[:, 1] >= 0.5) & (uv[:, 1] <= cam.height - 1.5)) & on_face
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
    S0 = S.copy() if (params.wb_reanchor or params.neutral_anchor) else None
    for _ in range(2):                                    # exposure: per-photo gains towards the blend
        for k in range(len(S)):
            m = V[k] & (count >= 2)
            if m.sum() < 50:
                continue
            ratio = np.median(rgb[m] + 1.0, axis=0) / np.maximum(np.median(S[k][m] + 1.0, axis=0), 1.0)
            S[k] *= np.clip(ratio, 0.5, 2.0)[None, None, :]
        np.clip(S, 0, 255, out=S)
        rgb = _weighted_median(S, V, Wt)
    if params.wb_reanchor and S0 is not None:
        # the blend drifts towards the white balance of the views it picks most (sunlit, warm): one per-channel
        # gain for the wall puts it back on the per-texel median of the raw views, exposure kept
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)              # texels no view sees: NaN, unused
            medv = np.nanmedian(np.where(V[..., None], S0, np.nan), axis=0)
        sel = (count >= 2) & np.isfinite(medv).all(-1)
        if sel.sum() >= 50:
            ratio = np.median(medv[sel] + 1.0, axis=0) / np.maximum(np.median(rgb[sel] + 1.0, axis=0), 1.0)
            corr = (ratio / ratio.mean()).astype(np.float32)
            S *= corr[None, None, None, :]
            np.clip(S, 0, 255, out=S)
            rgb = np.clip(rgb * corr[None, None, :], 0, 255)
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
    if params.neutral_anchor and S0 is not None:
        # how each raw view renders the light material (the brightest tenth of the observed near-neutral texels):
        # [luminance ratio raw / blend (linear light), texels, r, g, b ratios] -> scene_color_norm()
        blab = rgb_to_lab(np.clip(rgb, 0, 255))
        okm = mask & (np.hypot(blab[..., 1], blab[..., 2]) < 15)
        rows = []
        if okm.sum() >= 100:
            anc = okm & (blab[..., 0] >= np.percentile(blab[..., 0][okm], 90))
            wy = np.array([0.2126, 0.7152, 0.0722])
            yb = _srgb_lin(np.clip(rgb, 0, 255)) @ wy
            for k in range(len(S0)):
                vv = V[k] & anc
                if vv.sum() < 30:
                    continue
                lk, lb = _srgb_lin(S0[k][vv]), _srgb_lin(np.clip(rgb[vv], 0, 255))
                ratio = np.median(lk, axis=0) / np.maximum(np.median(lb, axis=0), 1e-6)
                rows.append([round(float(np.median(lk @ wy) / max(np.median(yb[vv]), 1e-6)), 4), int(vv.sum())]
                            + [round(float(v), 4) for v in ratio])
        stats["anchor_brightness"] = rows
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
_HUE = np.degrees(np.arctan2(PALETTE_LAB[:, 2], PALETTE_LAB[:, 1]))
NEUTRALS = np.array([PALETTE_NAMES.index(n) for n in NEUTRAL_RAMP])   # the light / dark ramp for grey stone
CHROMATIC = np.nonzero(_CHROMA >= 15)[0]
GREY_FAMILIES = ("neutral", "slate")    # where a near-grey material snaps (by dE00, keeping its measured cast)
GREYS = np.array([i for i, n in enumerate(PALETTE_NAMES) if FAMILY_OF[n] in GREY_FAMILIES])
GREY_CHROMA = 6.0                       # below this chroma (C*) a material is a near-grey
HUE_GUARD_DEG = 30.0                    # a coloured material snaps within this hue angle (never to another family)


def de00(lab1: np.ndarray, lab2: np.ndarray) -> np.ndarray:
    """CIEDE2000 colour difference, broadcasting over the leading axes of two (..., 3) Lab arrays."""
    L1, a1, b1 = (np.asarray(lab1, np.float64)[..., i] for i in range(3))
    L2, a2, b2 = (np.asarray(lab2, np.float64)[..., i] for i in range(3))
    C1, C2 = np.hypot(a1, b1), np.hypot(a2, b2)
    Cm = (C1 + C2) / 2
    G = 0.5 * (1 - np.sqrt(Cm ** 7 / (Cm ** 7 + 25.0 ** 7)))
    a1p, a2p = (1 + G) * a1, (1 + G) * a2
    C1p, C2p = np.hypot(a1p, b1), np.hypot(a2p, b2)
    h1p = np.degrees(np.arctan2(b1, a1p)) % 360
    h2p = np.degrees(np.arctan2(b2, a2p)) % 360
    dLp, dCp = L2 - L1, C2p - C1p
    dhp = h2p - h1p
    dhp = np.where(dhp > 180, dhp - 360, np.where(dhp < -180, dhp + 360, dhp))
    dhp = np.where(C1p * C2p == 0, 0.0, dhp)
    dHp = 2 * np.sqrt(C1p * C2p) * np.sin(np.radians(dhp / 2))
    Lbp, Cbp = (L1 + L2) / 2, (C1p + C2p) / 2
    hs = h1p + h2p
    hbp = np.where(C1p * C2p == 0, hs, np.where(np.abs(h1p - h2p) > 180, np.where(hs < 360, (hs + 360) / 2, (hs - 360) / 2), hs / 2))
    T = (1 - 0.17 * np.cos(np.radians(hbp - 30)) + 0.24 * np.cos(np.radians(2 * hbp))
         + 0.32 * np.cos(np.radians(3 * hbp + 6)) - 0.20 * np.cos(np.radians(4 * hbp - 63)))
    dth = 30 * np.exp(-(((hbp - 275) / 25) ** 2))
    Rc = 2 * np.sqrt(Cbp ** 7 / (Cbp ** 7 + 25.0 ** 7))
    Sl = 1 + 0.015 * (Lbp - 50) ** 2 / np.sqrt(20 + (Lbp - 50) ** 2)
    Sc, Sh = 1 + 0.045 * Cbp, 1 + 0.015 * Cbp * T
    Rt = -np.sin(np.radians(2 * dth)) * Rc
    return np.sqrt((dLp / Sl) ** 2 + (dCp / Sc) ** 2 + (dHp / Sh) ** 2 + Rt * (dCp / Sc) * (dHp / Sh))


def _srgb_lin(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float64) / 255.0
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _lin_srgb(lin: np.ndarray) -> np.ndarray:
    lin = np.clip(np.asarray(lin, np.float64), 0, 1)
    return np.where(lin <= 0.0031308, 12.92 * lin, 1.055 * np.power(lin, 1 / 2.4) - 0.055) * 255.0


@dataclass
class ColorNorm:
    """Scene-wide colour handling of the orthophotos. The camera's white balance is trusted: a grey-world
    balance over the building's own pixels shifts the hue of a single-material building or of a scene with a
    coloured occluder (a red shed turned a cream wall mint). The one exception (docs/STRUCTURE_FIRST.md §2.5
    item 7): when the landmark's lightest material is itself neutral (C* <= anchor_max_chroma) in the
    best-exposed views, the white balance is set so that it stays neutral (white marble must not read beige);
    a warm stone stays warm. Lightness is not stretched: each material keeps its lightness band (the exposure
    lift is an art-direction option, off by default, D21). l_lo / l_hi are only reported."""

    gains: np.ndarray                   # (3,) RGB multipliers (kept at 1)
    l_lo: float = 0.0
    l_hi: float = 100.0
    lin_gains: np.ndarray | None = None  # (3,) linear-light multipliers (white anchor x exposure), or None
    info: dict = field(default_factory=dict)

    def lab(self, rgb: np.ndarray) -> np.ndarray:
        if self.lin_gains is None:
            return rgb_to_lab(np.clip(np.asarray(rgb, np.float32) * self.gains, 0, 255))
        return rgb_to_lab(_lin_srgb(_srgb_lin(rgb) * self.lin_gains).astype(np.float32))

    def to_meta(self) -> dict[str, Any]:
        return {"white_balance": "camera" if self.lin_gains is None else "camera+neutral-anchor",
                "gains": [round(float(g), 3) for g in self.gains],
                "lin_gains": None if self.lin_gains is None else [round(float(g), 3) for g in self.lin_gains],
                "l_p3": round(self.l_lo, 1), "l_p97": round(self.l_hi, 1), **self.info}


IDENTITY_NORM = ColorNorm(np.ones(3, np.float32))


def _anchor_mask(o: Orthophoto) -> np.ndarray:
    """The light-material texels of an orthophoto: the brightest tenth of its observed near-neutral texels."""
    lab = rgb_to_lab(o.rgb)
    ok = o.mask & (np.hypot(lab[..., 1], lab[..., 2]) < 15)
    if ok.sum() < 100:
        return np.zeros_like(ok)
    return ok & (lab[..., 0] >= np.percentile(lab[..., 0][ok], 90))


def _wq(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """Weighted quantile."""
    o = np.argsort(values)
    c = np.cumsum(np.asarray(weights, np.float64)[o])
    if c[-1] <= 0:
        return float(np.median(values))
    return float(np.asarray(values)[o][min(int(np.searchsorted(c, q * c[-1])), len(o) - 1)])


def scene_color_norm(orthos: list[Orthophoto], params: StyleParams = DEFAULT_PARAMS) -> ColorNorm:
    pix = [o.rgb[o.mask] for o in orthos if o.mask.sum() >= 50]
    if not pix:
        return IDENTITY_NORM
    x = np.vstack(pix).astype(np.float32)
    if len(x) > 200000:
        x = x[np.random.default_rng(0).choice(len(x), 200000, replace=False)]
    lum = rgb_to_lab(x)[:, 0]
    norm = ColorNorm(np.ones(3, np.float32), float(np.percentile(lum, 3)), float(np.percentile(lum, 97)))
    if not params.neutral_anchor:
        return norm
    # the views' brightness on the light material (anchor texels; facade_orthophoto's anchor_brightness rows:
    # [luminance ratio raw / blend, texels, r, g, b ratios]); the best-exposed quartile shows its true colour
    ab = [v for o in orthos for v in o.stats.get("anchor_brightness", []) if v[1] > 0 and len(v) == 5]
    anc = [o.rgb[o.mask & _anchor_mask(o)] for o in orthos if o.mask.sum() >= 50]
    anc = [a for a in anc if len(a)]
    if not ab or not anc:
        return norm
    a = np.array(ab, np.float64)
    q = params.exposure_quantile
    e = float(np.clip(_wq(a[:, 0], a[:, 1], q) / max(_wq(a[:, 0], a[:, 1], 0.5), 1e-6), 1.0, params.exposure_max))
    top = a[a[:, 0] >= _wq(a[:, 0], a[:, 1], q)]
    gb = np.array([_wq(top[:, 2 + c], top[:, 1], 0.5) for c in range(3)])
    alin = np.median(_srgb_lin(np.vstack(anc)), axis=0)
    bright_lab = rgb_to_lab(_lin_srgb(np.clip(alin * gb, 0, 1))[None].astype(np.float32))[0]
    bc = float(np.hypot(bright_lab[1], bright_lab[2]))
    lin_g = np.full(3, e)
    info = {"anchor_bright_lab": [round(float(v), 1) for v in bright_lab], "exposure_lin": round(e, 3)}
    if bc <= params.anchor_max_chroma:
        lin_g = lin_g * (alin.mean() / np.maximum(alin, 1e-6))
        info["anchor_neutral"] = True
    else:
        info["anchor_neutral"] = False
    if np.allclose(lin_g, 1.0, atol=1e-4):
        norm.info = info
        return norm
    norm.lin_gains, norm.info = lin_g.astype(np.float32), info
    return norm


def snap_lab_to_palette(lab: np.ndarray, cheer: float = 1.5, lift: float = 0.12, chromatic_only: bool = False) -> np.ndarray:
    """Palette index for Lab colours (n, 3). Coloured ones go to the nearest palette colour after a chroma
    boost, within HUE_GUARD_DEG of their hue (a material keeps its hue family; none there: it is treated as a
    near-grey); near-greys (C* < GREY_CHROMA) go to the nearest grey (neutral and slate families) by CIEDE2000
    with their measured a*/b*, not by lightness alone (white marble is not beige stone). Everything is lifted a
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
    if chromatic_only:
        return out
    hue = np.degrees(np.arctan2(lab[:, 2], lab[:, 1]))
    dh = np.abs((_HUE[None, :] - hue[:, None] + 180.0) % 360.0 - 180.0)          # (n, palette)
    ok = (dh <= HUE_GUARD_DEG) & (_CHROMA[None, :] >= GREY_CHROMA)
    dd = (0.6 * (boosted[:, None, 0] - PALETTE_LAB[None, :, 0]) ** 2
          + ((boosted[:, None, 1:] - PALETTE_LAB[None, :, 1:]) ** 2).sum(-1))
    guarded = np.where(ok, dd, np.inf).argmin(axis=1)
    out = np.where(ok.any(axis=1), guarded, -1)
    grey = (chroma < GREY_CHROMA) | (out < 0)
    if np.any(grey):
        dn = de00(lab[grey, None, :], PALETTE_LAB[None, GREYS, :])
        out[grey] = GREYS[dn.argmin(axis=1)]
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
    mask: np.ndarray                    # final observation mask (geometric mask minus sky / vegetation); fill excluded
    base_rgb: tuple[int, int, int]      # the facade's wall colour (inside the mask)
    colors: list[str]                   # palette names used
    openings: list[dict[str, Any]]      # regularized windows / doors (metres, from the top-left corner)
    report: dict[str, Any]
    shapes: list[dict[str, Any]] = field(default_factory=list)   # vector regions (metres)
    fill: np.ndarray | None = None      # (h, w) bool: observed-fill texels (D12), disjoint from mask, base_rgb only
    px_per_m: float = 0.0
    face_cols: tuple[int, int] | None = None   # texture columns the GLB shows (between the rounded corners)


def tidy_mask(mask: np.ndarray, ppm: float, min_island_m2: float = 1.0, max_hole_m2: float = 0.5) -> np.ndarray:
    """A cleaner observation mask for drawing: observed islands under min_island_m2 are dropped and the
    edge is softened by a blur that may only remove pixels. Holes stay out of the mask (nobody saw them);
    those under max_hole_m2 are bridged for the blur only, so that their rim is not eaten away
    (enclosed_holes() decides what a hole may show)."""
    m = mask.astype(np.uint8)
    _, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    small = np.nonzero(stats[1:, cv2.CC_STAT_AREA] < min_island_m2 * ppm * ppm)[0] + 1
    m[np.isin(cc, small)] = 0
    bridged = m | enclosed_holes(m > 0, ppm, max_hole_m2)[0].astype(np.uint8)
    soft = cv2.GaussianBlur(bridged.astype(np.float32), (0, 0), max(0.8, 0.15 * ppm)) > 0.5
    return (m > 0) & soft


ACCENT_CHROMA_MARGIN = 8.0           # a coloured accent is at least this much more chromatic than the wall (C*)
MAX_SRGB_CHROMA = 134.0              # the most chroma an sRGB colour has in CIELAB (pure blue: 133.8)


def work_grid(w: int, h: int, ppm: float, params: StyleParams = DEFAULT_PARAMS) -> tuple[int, int]:
    """(columns, rows) of the working image facade_layers() segments a (h, w) facade texture on: about
    work_px_per_m, never finer than the texture. cv2.INTER_AREA makes working pixel j the average of
    texels [j * w / columns, (j + 1) * w / columns) (rows alike), so a texel straddles two pixels unless
    the ratio is a whole number."""
    f = min(1.0, params.work_px_per_m / max(ppm, 1e-9))
    return max(4, int(round(w * f))), max(4, int(round(h * f)))


def min_pixel_cover(params: StyleParams = DEFAULT_PARAMS) -> float:
    """The least share of a working pixel an element must cover to put that pixel in its layer: a dark
    layer pixel is dark_contrast (L*) darker than its surroundings and L* spans 100; an accent pixel is
    ACCENT_CHROMA_MARGIN (C*) more chromatic than the wall and no sRGB colour has more than
    MAX_SRGB_CHROMA. About 0.06 by default (a black element on a white wall, the strongest contrast there
    is; every other element needs more of the pixel)."""
    return min(params.dark_contrast / 100.0, ACCENT_CHROMA_MARGIN / MAX_SRGB_CHROMA)


def _span_texels(n: int, size: int, work: int, cover: float) -> int:
    """The fewest texels along one axis (of `size` texels, `work` working pixels) that an element can
    touch and still reach n consecutive working pixels, by at least `cover` of a pixel at each end (the
    pixels between lie wholly inside it): the minimum over every place on the axis. With a whole number s
    of texels per pixel no texel straddles two pixels and this is (n - 2) s + 2 (at s = 2 a partly covered
    end pixel is half covered); otherwise a texel can reach a pixel by a sliver and it is less."""
    s = size / work
    if n <= 1:
        return max(1, math.ceil(cover * s - 1e-9))
    a = np.arange(work - n + 1)
    if not len(a):
        return size + 1                                  # the axis has no room for n pixels: nothing fits
    first = np.floor((a + 1) * s - cover * s + 1e-9)     # last texel start still reaching pixel a
    end = np.ceil((a + n - 1) * s + cover * s - 1e-9)    # first texel end reaching pixel a + n - 1
    return max(1, int((end - first).min()))


def smallest_element_px(wppm: float, params: StyleParams = DEFAULT_PARAMS) -> tuple[int, int]:
    """(columns, rows) of working pixels the smallest whole compact element stylize_facade() draws spans:
    an opening is found only from a dark region whose bounding box reaches window_min_m (0.35 m x 0.5 m:
    3 x 4 pixels at 8 px/m); an own-colour accent holds a 3 x 3 pixel square (facade_layers() opens the
    accent layer with it; the area it also needs, min_own_accent_m2, is left out: the rule errs small). The
    narrower and the lower of the two: 3 x 3 by default.
    Free dark shapes need min_free_shape_m2 (0.6 m2), more than the fill cap."""
    n_w, n_h = (math.ceil(v * wppm - 1e-9) for v in params.window_min_m)
    return min(n_w, 3), min(n_h, 3)


def fill_shape_kernels(ppm: float, params: StyleParams = DEFAULT_PARAMS, shape: tuple[int, int] | None = None,
                       at_least: tuple[int, int] = (2, 2)) -> list[tuple[int, int]]:
    """(rows, columns) of the blocks a hole must not be able to hold to be filled (D12): the smallest
    photographed evidence of a compact element (smallest_element_px() working pixels, with the end pixels
    only partly covered: _span_texels()) and the shortest line, horizontal and vertical (line_min_len_m,
    counted in texels as facade_lines() counts it; bands are longer). A hole that can hold one of them
    could hide every texel that shows a whole element: it is not a small hole, and filling it would claim
    a flat wall where a window may be. `shape`: the facade texture's (rows, columns), which fixes where the
    working pixels fall. Without it, the least block of every texture of at least `at_least` (rows,
    columns) texels the producer can make (_least_block()): never larger than the block of any of them.
    At 16 px/m on a texture that splits evenly into 8 px/m pixels the block is 4 x 4 texels (0.25 m: a
    pixel, and half a pixel at each end); on one that does not, 3 texels across the uneven axis."""
    if shape is not None:
        h, w = shape
        cover = min_pixel_cover(params)
        sw, sh = work_grid(w, h, ppm, params)
        n_w, n_h = smallest_element_px(ppm * sw / w, params)
        kw, kh = _span_texels(n_w, w, sw, cover), _span_texels(n_h, h, sh, cover)
    else:
        kh, kw = _least_block(float(ppm), params, (int(at_least[0]), int(at_least[1])))
    n_len = max(3, int(round(params.line_min_len_m * ppm)))
    return [(kh, kw), (1, n_len), (n_len, 1)]


@functools.lru_cache(maxsize=256)
def _least_block(ppm: float, params: StyleParams, at_least: tuple[int, int]) -> tuple[int, int]:
    """(rows, columns) no larger than the compact block of fill_shape_kernels() on any facade texture of
    (rows, columns) from `at_least` up to max_texture_px a side (texture_size() makes no larger one), where
    the working pixels fall as work_grid() puts them: per axis, the least _span_texels() over every size,
    counting only blocks a hole in it has room for (an enclosed hole is at most size - 2 texels across), with
    the fewest element pixels any width's working grid gives. work_grid() rounds the pixel count, so a
    texture's texels per pixel can be well below 1 / f: at 16 px/m a 5-texel axis has 4 pixels of 1.25
    texels, and 2 texels reach 3 of them."""
    cover = min_pixel_cover(params)
    lo = [max(2, v) for v in at_least]
    sizes = [range(lo[i], max(params.max_texture_px, lo[i]) + 1) for i in (0, 1)]
    work = {n: work_grid(n, n, ppm, params)[0] for n in set(sizes[0]) | set(sizes[1])}
    n_px = {w: smallest_element_px(ppm * work[w] / w, params) for w in sizes[1]}   # (columns, rows) per width
    n_h = min(v[1] for v in n_px.values())

    def least(spans: list[tuple[int, int]]) -> int:
        room = [k for size, k in spans if k <= size - 2]
        return min(room) if room else min(k for _, k in spans)
    kh = least([(h, _span_texels(n_h, h, work[h], cover)) for h in sizes[0]])
    kw = least([(w, _span_texels(n_px[w][0], w, work[w], cover)) for w in sizes[1]])
    return kh, kw


def smallest_element_m(ppm: float, params: StyleParams = DEFAULT_PARAMS,
                       shape: tuple[int, int] | None = None) -> tuple[float, float]:
    """(width, height) in metres, in the facade's own axes, of the compact block of fill_shape_kernels():
    the least photographed evidence of a whole drawable element (0.25 m x 0.25 m at 16 px/m on an evenly
    split texture). A hole that can hold it is not filled."""
    kh, kw = fill_shape_kernels(ppm, params, shape)[0]
    return kw / ppm, kh / ppm


def could_hide_element(region: np.ndarray, kernels: list[tuple[int, int]]) -> bool:
    """Does a bool region (a hole, cropped) hold a whole block of one of `kernels` (fill_shape_kernels())?
    Erosion by each block, with nothing outside the crop."""
    r = np.asarray(region, np.uint8)
    return any(cv2.erode(r, np.ones(k, np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0).any()
               for k in kernels)


def enclosed_holes(observed: np.ndarray, ppm: float, max_m2: float, cols: tuple[int, int] | None = None,
                   kernels: list[tuple[int, int]] | None = None) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Holes of an observation mask: 8-connected unobserved regions that do not reach the texture's edge,
    so every texel around them (all eight neighbours) is an observed texel of this same wall. Unobserved
    regions that reach the edge (the clay strips at the wall's ends, the band at its foot, the sky above)
    are never holes, nor is anything reaching outside `cols` (the columns the GLB shows: the rest lies
    under the corner bevels, where a hole would be cut open). Returns the fill mask (the holes of at most
    max_m2 that, with `kernels` (fill_shape_kernels()), cannot hold a whole element) and every hole: area,
    bounding box in metres from the top-left corner, filled or not, and whether it could hide an element."""
    h, w = observed.shape
    if cols is not None:
        observed = observed.copy()
        observed[:, :cols[0]] = False
        observed[:, cols[1]:] = False
    fill = np.zeros((h, w), dtype=bool)
    holes: list[dict[str, Any]] = []
    n, cc, stats, _ = cv2.connectedComponentsWithStats((~observed).astype(np.uint8), connectivity=8)
    for c in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[c])
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            continue                                     # open to the edge: not enclosed
        m2 = area / ppm ** 2
        region = cc[y:y + bh, x:x + bw] == c
        hides = bool(kernels) and m2 <= max_m2 and could_hide_element(region, kernels)
        ok = m2 <= max_m2 and not hides
        if ok:
            fill[y:y + bh, x:x + bw] |= region
        holes.append({"area_m2": round(m2, 3), "filled": ok, "could_hide": hides,
                      "bbox_m": [round(x / ppm, 2), round(y / ppm, 2), round(bw / ppm, 2), round(bh / ppm, 2)]})
    return fill, holes


def texel_provenance(fac: StylizedFacade) -> np.ndarray:
    """(h, w) uint8 code per texel of a stylized facade, indexing TEXEL_PROVENANCES: 0 clay (default-style),
    1 photo-observed (the mask), 2 observed-fill."""
    out = np.zeros(fac.mask.shape, np.uint8)
    out[fac.mask] = 1
    if fac.fill is not None:
        out[fac.fill & ~fac.mask] = 2
    return out


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
    sw, sh = work_grid(w, h, ppm, params)
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
    accent = (chroma > max(params.accent_min_chroma, wall_chroma + ACCENT_CHROMA_MARGIN)) & mask_s
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
    dark = int(snap_lab_to_palette(dark_lab[None], lift=params.dark_lift)[0])
    if PALETTE_LAB[wall, 0] - PALETTE_LAB[dark, 0] < params.min_dark_delta:   # keep the accents readable
        cand = [i for i in range(len(PALETTE)) if PALETTE_LAB[wall, 0] - PALETTE_LAB[i, 0] >= params.min_dark_delta]
        if float(np.hypot(dark_lab[1], dark_lab[2])) < GREY_CHROMA:
            # a near-grey dark material stays a grey (a dark grey marble is not dark green or brown)
            cand = [i for i in cand if FAMILY_OF[PALETTE_NAMES[i]] in GREY_FAMILIES] or cand
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
                   layers: FacadeLayers | None = None, size_scale: float = 1.0,
                   extent: tuple[float, float] = (0.0, 1.0)) -> StylizedFacade:
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
    so no colour, not even the wall colour, bleeds onto what nobody photographed. The one exception (D12,
    decided by the product owner): a hole of the mask enclosed by observed texels of this wall, at most
    fill_max_hole_m2 (a tree, a passer-by, a gap in the samples) and too small to hold a whole element
    (fill_shape_kernels(): no room for the least evidence of a window or accent, 0.25 m x 0.25 m at 16 px/m,
    nor for a 1.5 m straight run) takes this wall's base
    colour as snapped, and nothing else: no decal, accent or line, no neighbour's colour, no inpainting.
    Those texels are observed-fill, not observed: they stay out of the mask and of every coverage figure.
    `extent`: the share of the wall's length its flat face spans between the rounded corners
    (wall_extents()); a hole reaching beyond it would be cut open in the GLB, so it is not filled.

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
    fill_mask = np.zeros((h, w), dtype=bool)
    holes: list[dict[str, Any]] = []
    face_cols = face_columns(extent, 1.0, w)
    kernels = fill_shape_kernels(ppm, params, (h, w))
    if mask.any():
        mask = tidy_mask(mask, ppm)
        fill_mask, holes = enclosed_holes(mask, ppm, params.fill_max_hole_m2, face_cols, kernels)
    clay = clay_pattern(h, w, ppm)
    if mask.sum() < max(20, 0.5 * ppm * ppm) or ly.mask_s.sum() < 8:
        return StylizedFacade(clay, np.zeros_like(mask), CLAY_RGB, [], [],
                              {"empty": True, "clutter_fraction": round(float(ly.clutter.mean()), 3)},
                              fill=np.zeros_like(mask), px_per_m=ppm, face_cols=face_cols)
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
    stripe_report = None
    if params.stripes:
        # zebra masonry (light / dark courses of stone): measured runs of observed texels, drawn as clean courses
        full_lab = norm.lab(ortho.rgb)
        smask, sgroups = detect_stripes(full_lab[..., 0], mask, ppm, params)
        if smask.any():
            slab = np.median(full_lab[smask], axis=0)
            si = int(snap_lab_to_palette(slab[None], lift=params.dark_lift)[0])
            srgb = tuple(int(v) for v in PALETTE_RGB[si])
            if srgb == base:
                si, srgb = plan.dark, dark_rgb
            tex[smask] = srgb
            used.add(PALETTE_NAMES[si])
            stripe_report = {"texels": int(smask.sum()), "m2": round(float(smask.sum()) / ppm ** 2, 2),
                             "groups": len(sgroups), "courses": int(max((g["count"] for g in sgroups), default=0)),
                             "color": PALETTE_NAMES[si]}
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
    tex[~mask] = clay[~mask]                              # honesty: nothing outside the observation mask ...
    tex[fill_mask] = base                                 # ... but small enclosed holes: the plain wall colour (D12)
    if free.any() or shapes or openings:
        used.add(PALETTE_NAMES[plan.dark])
    kinds = [s["kind"] for s in shapes]
    filled = [hl["area_m2"] for hl in holes if hl["filled"]]
    over = [hl["area_m2"] for hl in holes if not hl["filled"] and not hl["could_hide"]]
    hiding = [hl["area_m2"] for hl in holes if hl["could_hide"]]
    report = {"colors": sorted(used), "base": PALETTE_NAMES[wall_idx], "num_openings": len(openings),
              "num_shapes": len(shapes), "shape_kinds": {k: kinds.count(k) for k in sorted(set(kinds))},
              "free_shapes": free_count, "free_shapes_dropped": dropped, "accent_regions": acc_drawn,
              "dark_fraction": round(float(ly.dark_s[ly.mask_s].mean()), 3),
              "accent_fraction": round(float(ly.accent_s[ly.mask_s].mean()), 3),
              "mask_fraction": round(float(mask.mean()), 3),
              "fill": {"holes": len(filled), "area_m2": round(float(fill_mask.sum()) / ppm ** 2, 3),
                       "largest_m2": round(max(filled, default=0.0), 3),
                       "holes_over_cap": len(over), "over_cap_m2": round(sum(over), 2),
                       "largest_over_cap_m2": round(max(over, default=0.0), 2),
                       "holes_could_hide": len(hiding), "could_hide_m2": round(sum(hiding), 2),
                       "element_min_m": [round(kernels[0][1] / ppm, 3), round(kernels[0][0] / ppm, 3)]},
              "clutter_fraction": round(float(ly.clutter.mean()), 3), "seconds": round(time.perf_counter() - t0, 3)}
    if stripe_report:
        report["stripes"] = stripe_report
    return StylizedFacade(tex, mask, base, sorted(used), openings, report, shapes, fill_mask, ppm, face_cols)


def detect_stripes(L: np.ndarray, mask: np.ndarray, ppm: float, params: StyleParams = DEFAULT_PARAMS
                   ) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Horizontal courses of alternating light / dark stone (Tuscan / Romanesque zebra masonry), at texture
    resolution. In 0.5 m column windows, the row profile of L* (observed texels only) is split against its 1.2 m
    running mean into runs; a sequence of >= stripe_min_runs dark runs, each stripe_thickness_m thick, alternating
    with light runs, each >= stripe_contrast darker than its light neighbours, with regular thickness and spacing
    (stripe_thickness_cv, stripe_period_cv), marks the columns of the window whose own rows show the same
    contrast. Column groups >= 0.4 m wide are then regularized: each dark row run covering half the group becomes
    one clean course across it; a group needs >= stripe_min_runs courses. Returns the dark-course texel mask
    (observed texels only: every course drawn is a measured run of observed texels) and the groups (metres)."""
    h, w = L.shape
    min_runs, min_contrast = params.stripe_min_runs, params.stripe_contrast
    win = max(4, int(round(0.5 * ppm)))
    step = max(2, win // 2)
    tmin = max(2, int(round(params.stripe_thickness_m[0] * ppm)))
    tmax = max(3, int(round(params.stripe_thickness_m[1] * ppm)))
    k = int(1.2 * ppm) | 1
    out = np.zeros((h, w), bool)
    groups: list[dict[str, Any]] = []
    for x0 in range(0, max(1, w - win + 1), step):
        x1 = min(w, x0 + win)
        m = mask[:, x0:x1]
        ok = m.mean(1) >= 0.75
        if ok.sum() < 2 * min_runs * tmin:
            continue
        prof = np.where(ok, np.where(m, L[:, x0:x1], 0).sum(1) / np.maximum(m.sum(1), 1), 0.0).astype(np.float32)
        num = cv2.blur((prof * ok).reshape(-1, 1), (1, k)).ravel()
        den = cv2.blur(ok.astype(np.float32).reshape(-1, 1), (1, k)).ravel()
        ref = num / np.maximum(den, 1e-3)
        lab_ = np.where(ok, np.where(prof < ref, 1, 2), 0)          # 0 unobserved, 1 dark, 2 light
        runs = []
        y = 0
        while y < h:
            y2 = y
            while y2 < h and lab_[y2] == lab_[y]:
                y2 += 1
            runs.append([int(lab_[y]), y, y2])
            y = y2

        def flush(sq: list[list[int]]) -> None:
            if sum(1 for r in sq if r[0] == 1) < min_runs or sum(1 for r in sq if r[0] == 2) < min_runs - 1:
                return
            good = []
            for i, r in enumerate(sq):
                if r[0] != 1:
                    continue
                nb = [sq[j] for j in (i - 1, i + 1) if 0 <= j < len(sq) and sq[j][0] == 2]
                if nb and np.mean([prof[n[1]:n[2]].mean() for n in nb]) - prof[r[1]:r[2]].mean() >= min_contrast:
                    good.append(r)
            if len(good) < min_runs:
                return
            th = np.array([r[2] - r[1] for r in good], float)
            if th.std() > params.stripe_thickness_cv * th.mean():
                return
            cen = np.array([(r[1] + r[2]) / 2 for r in good], float)
            per = np.diff(cen)
            if len(per) >= 2 and per.std() > params.stripe_period_cv * per.mean():
                return
            rows_d, rows_l = np.zeros(h, bool), np.zeros(h, bool)
            for r in good:
                rows_d[r[1]:r[2]] = True
            for r in sq:
                if r[0] == 2:
                    rows_l[r[1]:r[2]] = True
            for x in range(x0, x1):                      # columns of the window that carry the courses themselves
                md, ml = mask[:, x] & rows_d, mask[:, x] & rows_l
                if md.sum() < tmin * min_runs or ml.sum() < tmin:
                    continue
                if L[ml, x].mean() - L[md, x].mean() >= 0.6 * min_contrast:
                    out[:, x] |= md
            groups.append({"x": x0 / ppm, "w": (x1 - x0) / ppm, "y": good[0][1] / ppm,
                           "h": (good[-1][2] - good[0][1]) / ppm, "count": len(good)})

        seq: list[list[int]] = []
        for r in runs:
            v, a, b = r
            t = b - a
            if v == 0 or t < tmin or t > tmax:
                flush(seq)
                seq = [] if (v == 0 or t > tmax) else [r]
                continue
            seq.append(r)
        flush(seq)
    cols = out.any(axis=0)
    n, cc, stt, _ = cv2.connectedComponentsWithStats(cols.astype(np.uint8)[None, :], connectivity=4)
    for c in range(1, n):
        x0, gw_ = int(stt[c, cv2.CC_STAT_LEFT]), int(stt[c, cv2.CC_STAT_WIDTH])
        if gw_ < 0.4 * ppm:
            out[:, cc[0] == c] = False                   # a lone column is noise
            continue
        rows = out[:, x0:x0 + gw_].mean(axis=1) >= 0.5   # one clean course per dark row run across the group
        nr, ccr, str_, _ = cv2.connectedComponentsWithStats(rows.astype(np.uint8)[:, None], connectivity=4)
        for r in range(1, nr):
            if str_[r, cv2.CC_STAT_HEIGHT] < tmin:
                rows[ccr[:, 0] == r] = False
        courses = cv2.connectedComponentsWithStats(rows.astype(np.uint8)[:, None], connectivity=4)[0] - 1
        out[:, x0:x0 + gw_] = rows[:, None] if courses >= min_runs else False
    return out & mask, groups


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


def guard_point_colors(points: np.ndarray, colors: np.ndarray | None, cameras: list[PosedCamera],
                       params: StyleParams = DEFAULT_PARAMS, max_views: int = 5, max_cameras: int = 120,
                       sky_masks: Any = None) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Point colours a roof may be coloured from. When more than recolor_black_share of them are black or missing
    (a reconstruction exported without colours), they are taken again from the posed photos: per camera, a
    4-pixel z-buffer of the points keeps those on the visible surface; a sample whose pixel the photo's sky mask
    calls sky (recolor_sky_mask; sky_masks[camera index] when given, else crowd3d.skyline.sky_mask, the licence-safe
    A+R mask; dilated by recolor_sky_dilate_px) is dropped -- a slender spire's points fall on the sky at its
    silhouette; and a point is coloured only when at least recolor_min_views photos agree on it (median absolute
    deviation of their L* <= recolor_max_spread): its colour is the per-channel median of up to max_views of them.
    Other points stay uncoloured (valid False). Without photos to read, the colours are dropped: never a roof called
    'photo-observed' from null colours. Returns (colours, valid mask or None, report)."""
    n = len(points)
    black = np.ones(n, bool) if colors is None or len(colors) != n else (np.asarray(colors).max(axis=1) <= 3)
    rep: dict[str, Any] = {"black_share": round(float(black.mean()), 3), "recolored": False}
    if black.mean() <= params.recolor_black_share:
        return colors, None, rep
    t0 = time.perf_counter()
    order = list(range(len(cameras)))
    if len(order) > max_cameras:
        order = [order[i] for i in np.linspace(0, len(order) - 1, max_cameras).round().astype(int)]
    samples = np.zeros((n, max_views, 3), np.uint8)
    count = np.zeros(n, np.int32)
    pts = np.asarray(points, float)
    used = sky_dropped = 0
    kernel = np.ones((2 * params.recolor_sky_dilate_px + 1,) * 2, np.uint8) if params.recolor_sky_dilate_px > 0 else None
    for ci in order:
        cam = cameras[ci]
        try:
            img = cam.load()
        except (OSError, ValueError, TypeError, AttributeError):   # no pixels to read
            continue
        used += 1
        uv, z = cam.project(pts)
        h, w = img.shape[:2]
        sky = None
        if params.recolor_sky_mask:
            sky = None if sky_masks is None else (sky_masks.get(ci) if isinstance(sky_masks, dict)
                                                   else (sky_masks[ci] if ci < len(sky_masks) else None))
            if sky is None:
                from . import skyline as _skyline          # lazy: crowd3d.skyline imports this module
                sky = _skyline.sky_mask(img)
            sky = np.asarray(sky, bool)
            if sky.shape != (h, w):
                sky = cv2.resize(sky.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST) > 0
            if kernel is not None and sky.any():
                sky = cv2.dilate(sky.astype(np.uint8), kernel) > 0
        inframe = (z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        idx = np.nonzero(inframe & (count < max_views))[0]
        if not len(idx):
            continue
        cell = 4.0
        gw = int(math.ceil(w / cell))

        def cell_of(sel):
            return (uv[sel, 1] // cell).astype(np.int64) * gw + (uv[sel, 0] // cell).astype(np.int64)

        zb = np.full(int(math.ceil(h / cell)) * gw, np.inf)
        np.minimum.at(zb, cell_of(inframe), z[inframe])      # every point occludes, not only the uncoloured ones
        vis = z[idx] <= zb[cell_of(idx)] * 1.005 + 0.1
        idx = idx[vis]
        px = np.clip(uv[idx].astype(int), 0, [w - 1, h - 1])
        if sky is not None:
            on_sky = sky[px[:, 1], px[:, 0]]
            sky_dropped += int(on_sky.sum())
            idx, px = idx[~on_sky], px[~on_sky]
        samples[idx, count[idx]] = img[px[:, 1], px[:, 0]]
        count[idx] += 1
    if not used or not (count > 0).any():
        rep.update({"recolored": False, "dropped": True, "reason": "no photos to recolour from"})
        return None, None, rep
    out = np.zeros((n, 3), np.uint8)
    valid = np.zeros(n, bool)
    disagree = 0
    for c in range(max(1, params.recolor_min_views), max_views + 1):
        sel = np.nonzero(count == c)[0]
        if not len(sel):
            continue
        sm = samples[sel, :c]
        L = rgb_to_lab(sm.reshape(-1, 3).astype(np.float32))[:, 0].reshape(len(sel), c)
        mad = np.median(np.abs(L - np.median(L, axis=1, keepdims=True)), axis=1)
        ok = mad <= params.recolor_max_spread
        disagree += int((~ok).sum())
        out[sel[ok]] = np.median(sm[ok], axis=1).astype(np.uint8)
        valid[sel[ok]] = True
    rep.update({"recolored": True, "cameras": used, "colored_share": round(float(valid.mean()), 3),
                "seen_share": round(float((count > 0).mean()), 3),
                "too_few_views": int(((count > 0) & (count < params.recolor_min_views)).sum()),
                "views_disagree": disagree, "sky_samples_dropped": sky_dropped,
                "seconds": round(time.perf_counter() - t0, 1)})
    return out, valid, rep


def facing_cameras(cameras: list[PosedCamera], centre: np.ndarray, normal: np.ndarray, min_cos: float = 0.02) -> int:
    """How many cameras face a surface point: in front of its plane (view ray . normal > 0) with the point in frame."""
    c = np.asarray(centre, float).reshape(3)
    n = np.asarray(normal, float).reshape(3)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    k = 0
    for cam in cameras:
        v = cam.center - c
        d = float(np.linalg.norm(v))
        if d < 1e-6 or float(v @ n) / d <= min_cos:
            continue
        uv, z = cam.project(c[None])
        if z[0] > 0.3 and 0 <= uv[0, 0] < cam.width and 0 <= uv[0, 1] < cam.height:
            k += 1
    return k


def roof_appearance(massing: Massing, points: np.ndarray | None, colors: np.ndarray | None,
                    norm: ColorNorm | None = None, params: StyleParams = DEFAULT_PARAMS,
                    color_valid: np.ndarray | None = None, cameras: list[PosedCamera] | None = None) -> RoofLook:
    """Roof colours from the points, honest about coverage. A cap face (a flat cap, or one pyramid face)
    whose cells carry roof points on at least roof_full_coverage of its area is coloured as a whole;
    otherwise only its observed patches are (smooth outlines of the covered cells, each patch with
    >= roof_face_min_coverage point support and >= roof_patch_min_m2), and the rest stays clay. The
    points' colours are snapped like a wall material, lifted towards light. With cameras, a face or patch is
    observed only when roof_min_cameras of them face it (in front of its plane, its centre in frame): points
    alone do not say a photo saw that face (they may come from a parapet, a tree or withheld photos). A 'height
    pending' stub's cap (D13) is never observed."""
    norm = norm or IDENTITY_NORM
    look = RoofLook()
    pts = None if points is None or colors is None or not len(points) else np.asarray(points, float)
    cols = None if pts is None else np.asarray(colors)

    def faced(pr: Prism, face: int, xy: np.ndarray) -> bool:
        if cameras is None:
            return True
        zf, nn = _face_plane_z(pr, face)
        c = np.asarray(xy, float).reshape(-1, 2).mean(axis=0)
        return facing_cameras(cameras, np.r_[c, zf(c[None])[0]], nn) >= params.roof_min_cameras

    for pr in massing.prisms:
        if pr.stub:
            for f in range(pr.num_faces()):
                look.coverage[(pr.index, f)] = 0.0
                look.area_m2[(pr.index, f)] = 0.0
                look.face_rgb[(pr.index, f)] = None
            continue
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
            if color_valid is not None:
                sel = sel[color_valid[sel]]                    # points no photo coloured say nothing
            if len(sel) < 20:
                return None
            plab = norm.lab(cols[sel].astype(np.float32))
            # points on roof edges and spires take the sky's colour: a light bluish one, or a deep blue one at any
            # lightness (b* < -20: no slate or zinc is that blue; Aachen's spire took (17, 78, 183) without this)
            skyish = ((plab[:, 2] < -9) & (plab[:, 0] > 65)) | (plab[:, 2] < -20)
            if (~skyish).sum() >= 20:
                plab = plab[~skyish]
            elif skyish.all():
                return None
            lab = np.array([np.percentile(plab[:, 0], params.roof_quantile), np.median(plab[:, 1]),
                            np.median(plab[:, 2])], np.float32)
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
            if cov >= params.roof_full_coverage and faced(pr, f, g.centres().reshape(-1, 2)[(g.region & (g.face == f)).reshape(-1)]):
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
                if not faced(pr, f, c[inside.reshape(-1)]):
                    continue                              # no camera faces this patch: its points are not a photo of it
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
    texel_provenance: np.ndarray | None = None   # (h, w) uint8 codes of the texture's texels (TEXEL_PROVENANCES)


def _enu_to_gltf(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, np.float32)
    return np.stack([v[:, 0], v[:, 2], -v[:, 1]], 1)


def _png_bytes(img: np.ndarray) -> bytes:
    """Lossless PNG: an indexed PNG with an exact palette when an RGB image has <= 256 colours (stylized
    facades have ~10), else RGB; a 2-D uint8 array (texel-provenance codes) as greyscale. Exact colours
    matter: the viewer restyles texels by their colour (web/themes). PIL's quantize() is not exact (its
    fast octree merged ink (45, 42, 50) with a palette shade), so the palette is built here."""
    img = np.ascontiguousarray(img)
    buf = io.BytesIO()
    if img.ndim == 2:
        Image.fromarray(img.astype(np.uint8), "L").save(buf, "PNG", optimize=True)
        return buf.getvalue()
    cols, inv = np.unique(img.reshape(-1, 3), axis=0, return_inverse=True)
    if len(cols) <= 256:
        im = Image.fromarray(inv.reshape(img.shape[:2]).astype(np.uint8), "P")
        im.putpalette(cols.astype(np.uint8).reshape(-1).tolist())
    else:
        im = Image.fromarray(img, "RGB")
    im.save(buf, "PNG", optimize=True)
    return buf.getvalue()


def pack_atlases(textures: list[np.ndarray], max_size: int = 2048, gutter: int = 4,
                 layers: list[list[np.ndarray]] | None = None, background: Any = 0
                 ) -> tuple[list[np.ndarray], list[tuple[int, float, float, float, float]]]:
    """Shelf-pack textures into as few max_size x max_size atlases as needed (edge pixels replicated
    into a gutter so that filtering does not bleed). Returns the atlases and, per texture,
    (atlas, u0, v0, du, dv) to map its UVs: u' = u0 + u * du, v' = v0 + v * dv. Unused atlas space holds
    `background`. Each list in `layers` (images of the same sizes as the textures, e.g. their
    texel-provenance codes) is packed into its own atlases with the same layout (unused space 0),
    appended to the returned atlases in that order."""
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
    sizes = [(heights[a], max(sh[2] for sh in shelves[a])) for a in range(len(shelves))]

    def paint(images: list[np.ndarray], bg: Any = 0) -> list[np.ndarray]:
        out_ = [np.full(size + images[0].shape[2:], bg, images[0].dtype) for size in sizes]
        for i, img in enumerate(images):
            a, x, y = place[i]
            h, w = img.shape[:2]
            out_[a][y:y + h + 2 * gutter, x:x + w + 2 * gutter] = cv2.copyMakeBorder(
                np.ascontiguousarray(img), gutter, gutter, gutter, gutter, cv2.BORDER_REPLICATE)
        return out_

    atlases = paint(textures, background)
    for layer in layers or []:
        if [im.shape[:2] for im in layer] != [t.shape[:2] for t in textures]:
            raise ValueError("an atlas layer does not match the textures' sizes")
        atlases += paint(layer)
    out = []
    for i, tex in enumerate(textures):
        a, x, y = place[i]
        h, w = tex.shape[:2]
        H, W = sizes[a]
        out.append((a, (x + gutter) / W, (y + gutter) / H, w / W, h / H))
    return atlases, out


def write_glb(parts: list[Part], scene_extras: dict[str, Any] | None = None, atlas_size: int = 2048) -> bytes:
    """Minimal glTF 2.0 binary: one node + mesh per part (so every face group keeps its own
    provenance in extras); facade textures packed into shared atlases (lossless PNG), one material per
    atlas and per flat colour.

    What the colours mean travels with the file, so that a viewer can restyle it (web/themes): the scene
    extras carry the colour table ("style", color_table()), a flat material names its colour in its
    extras, and when every textured part has its texel provenance, those codes are packed into
    greyscale atlases with the same layout (images listed under the scene extras' "texel_provenance",
    not used by any material)."""
    bin_chunks: list[bytes] = []
    offset = 0
    gltf: dict[str, Any] = {"asset": {"version": "2.0", "generator": "crowd3d.stylize"},
                            "scene": 0, "scenes": [{"nodes": list(range(len(parts)))}], "nodes": [], "meshes": [],
                            "materials": [], "accessors": [], "bufferViews": []}
    extras = dict(scene_extras or {})
    extras.setdefault("style", style_meta())
    gltf["scenes"][0]["extras"] = extras
    pose_m = pose_node_matrix((extras.get("structure") or {}).get("pose"))      # D25-T: the stack's tilt

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
    with_codes = bool(textured) and all(parts[i].texel_provenance is not None for i in textured)
    layers = [[parts[i].texel_provenance for i in textured]] if with_codes else None
    packed, slots = (pack_atlases([parts[i].texture for i in textured], atlas_size, layers=layers, background=CLAY_RGB)
                     if textured else ([], []))   # (unused atlas space: clay, so every texel is a colour of the table)
    n_atlas = len(packed) // 2 if with_codes else len(packed)
    atlases, code_atlases = packed[:n_atlas], packed[n_atlas:]
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
                "baseColorTexture": {"index": a}, "metallicFactor": 0.0, "roughnessFactor": 1.0}, "extras": {"atlas": a}})
            atlas_material[a] = len(gltf["materials"]) - 1
        if code_atlases:
            first = len(gltf["images"])
            for img in code_atlases:
                gltf["images"].append({"bufferView": view(_png_bytes(img)), "mimeType": "image/png"})
            extras["texel_provenance"] = {"images": list(range(first, len(gltf["images"]))),
                                          "values": list(TEXEL_PROVENANCES)}

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
                mat = {"name": "flat-{:02x}{:02x}{:02x}".format(*rgb), "doubleSided": False,
                       "pbrMetallicRoughness": {"baseColorFactor": [_srgb_to_linear(c / 255.0) for c in rgb] + [1.0],
                                                "metallicFactor": 0.0, "roughnessFactor": 1.0}}
                if rgb in COLOR_NAME_OF_RGB:
                    mat["extras"] = {"color": COLOR_NAME_OF_RGB[rgb]}
                gltf["materials"].append(mat)
                colour_material[rgb] = len(gltf["materials"]) - 1
            material = colour_material[rgb]
        prim = {"attributes": attrs, "indices": accessor(idx, "SCALAR", 34963), "material": material}
        gltf["meshes"].append({"name": part.name, "primitives": [prim]})
        node = {"name": part.name, "mesh": len(gltf["meshes"]) - 1}
        if pose_m is not None:
            node["matrix"] = pose_m
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


def with_scene_extras(glb: bytes, add: dict[str, Any]) -> bytes:
    """The same GLB with keys added to (or replaced in) its scene extras; the binary chunk is kept as it is."""
    gltf, binary = read_glb(glb)
    gltf["scenes"][0].setdefault("extras", {}).update(add)
    js = json.dumps(gltf, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    js += b" " * ((-len(js)) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


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
    snap_prior: Any = None              # SnapPrior: the georegistration's yaw sigma and the cameras' own GPS
    point_quality: Any = None           # (n, 2) per point: track length, triangulation angle (deg); optional
    structure: Any = None               # crowd3d.structure.StructureModel (OSM building:part solids in this ENU frame,
                                        # judged or not): the massing when StyleParams.massing_source is "osm-parts"
    sky_masks: Any = None               # optional per camera index: a bool sky mask (crowd3d.skyline.sky_mask, cached per
                                        # photo); recolouring computes the missing ones itself
    point_tracks: Any = None            # crowd3d.relief.PointTracks: the points' 2-D observations (camera index, pixel),
                                        # for the facade relief's split-half check; without them no relief is built
    lines3d: Any = None                 # crowd3d.lines3d Lines3D / LinesResult in the final (snapped) frame, as the judge
                                        # computed them; None: computed here when a wall has enough points for relief


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


def wall_extents(massing: Massing, params: StyleParams = DEFAULT_PARAMS) -> dict[int, tuple[float, float]]:
    """Per wall, the stretch of its flat face between the rounded corners (u from its start a, metres):
    what the GLB shows of its facade; the texture beyond lies under the corner bevels."""
    segs = BEVEL_SEGMENTS if params.bevel_m > 0 else 0
    out: dict[int, tuple[float, float]] = {}
    if not segs:
        return {w.index: (0.0, w.length) for w in massing.walls}
    for pr in massing.prisms:
        radius = bevel_radius(pr.polygon, params)
        walls = {(w.ring, w.edge): w for w in massing.walls if w.prism == pr.index}
        for r, ring in enumerate(pr.rings):
            rb = bevel_ring(ring, radius, BEVEL_SEGMENTS)[0]
            nk = len(ring)
            for k in range(nk):
                wall = walls.get((r, k))
                if wall is None:                 # a structure part's edge without a wall band (roof-only part)
                    continue
                i0, i1 = (k * (segs + 1) + segs, ((k + 1) % nk) * (segs + 1)) if segs else (k, (k + 1) % nk)
                out[wall.index] = (float((rb[i0] - wall.a) @ wall.tangent), float((rb[i1] - wall.a) @ wall.tangent))
    return out


def face_columns(extent_m: tuple[float, float], length_m: float, width_px: int) -> tuple[int, int]:
    """Texture columns [c0, c1) wholly inside a wall's shown stretch (wall_extents())."""
    c0 = int(math.ceil(extent_m[0] / max(length_m, 1e-9) * width_px - 1e-6))
    c1 = int(math.floor(extent_m[1] / max(length_m, 1e-9) * width_px + 1e-6))
    return max(0, c0), min(width_px, max(c1, c0))


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
                photo_refs: dict[int, list[dict[str, Any]]] | None = None,
                relief: dict[int, list[dict[str, Any]]] | None = None) -> list[Part]:
    """Bevelled toy mesh. Parts (one GLB node each, each with its own provenance):
    - wall strips: the strip over a facade's observed columns (textured: stylized inside the observation
      mask, the clay pattern outside it) and clay strips beyond it; on an observed strip, the facade relief the
      photos measured there (relief: wall index -> crowd3d.relief elements with their guards; only the accepted
      ones are built, inside the strip, each with its provenance record in the node's extras "relief");
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
    extents = wall_extents(massing, params)
    for pr in massing.prisms:
        radius = bevel_radius(pr.polygon, params)
        rings_b = [bevel_ring(r, radius, BEVEL_SEGMENTS)[0] for r in pr.rings]
        walls = {(w.ring, w.edge): w for w in massing.walls if w.prism == pr.index}
        for r, ring in enumerate(pr.rings):
            rb = rings_b[r]
            nk = len(ring)
            ends: dict[int, tuple[bool, bool, Any]] = {}       # edge -> (observed at start, observed at end, colour)
            for k in range(nk):
                wall = walls.get((r, k))
                if wall is None:                 # a structure part's edge without a wall band (roof-only part)
                    ends[k] = (False, False, None)
                    continue
                ua, ub = extents[wall.index]
                fac = None if pr.stub else observed.get(wall.index)
                pieces = wall_pieces(wall, ua, ub, fac, params)
                ends[k] = (pieces[0][2], pieces[-1][2], fac.base_rgb if fac else None)
                ob = obs_by_wall.get(wall.index)
                for pi, (u0, u1, is_obs) in enumerate(pieces):
                    if wall.profile is None:
                        q0, q1 = wall.a + u0 * wall.tangent, wall.a + u1 * wall.tangent
                        pos = _quad(q0, q1, wall.z0, wall.z1)
                        uz = np.array([[u0, wall.z0], [u1, wall.z0], [u1, wall.z1], [u0, wall.z1]])
                        tri_idx = np.array([[0, 1, 2], [0, 2, 3]])
                    else:                        # a gable end / skillion side: the face below its top profile
                        uz = wall.uz_polygon(u0, u1)
                        pos = np.c_[wall.a[None, :] + uz[:, :1] * wall.tangent[None, :], uz[:, 1]]
                        tri_idx = np.array(triangulate_polygon(uz), int).reshape(-1, 3)
                    nrm = None
                    rel_recs, rel_rep = None, None
                    if is_obs and relief and relief.get(wall.index):
                        from . import relief as RL
                        els = relief[wall.index]
                        mesh = RL.relief_mesh(wall, u0, u1, els)
                        built: set[int] = set()
                        if mesh is not None:
                            pos, nrm, uz, tri_idx, rel_rep = mesh
                            built = {r["el"] for r in rel_rep["recesses"]} | {b["el"] for b in rel_rep["boxes"]}
                        rel_recs = [RL.element_record(e, k in built) for k, e in enumerate(els)
                                    if e.get("accepted") and e["u1"] > u0 and e["u0"] < u1]
                    if nrm is None:
                        nrm = np.repeat(np.r_[wall.normal, 0.0][None], len(pos), axis=0)
                    hp = wall.height_provenance
                    extras = {"kind": "wall", "wall": wall.index, "prism": pr.index, "ring": r, "piece": pi,
                              "observed": bool(is_obs), "locked": not is_obs,
                              "provenance": _provenance(wall.outline_source, hp, PROV_PHOTO if is_obs else PROV_DEFAULT),
                              "height_measured_here": hp == PROV_SFM,
                              "area_m2": round(abs(signed_area(uz)), 2)}
                    if pr.part is not None:
                        extras["part"] = pr.part
                        extras["geometry"] = pr.structure.get("state", "")
                        if pr.structure.get("adopted"):
                            extras["adopted"] = pr.structure["adopted"]
                        if pr.structure.get("outline_adopted"):
                            extras["outline_adopted"] = pr.structure["outline_adopted"]
                        for k in ("corrected", "tilt"):            # D25: 照片纠正地图 and the stack's tilt
                            if pr.structure.get(k):
                                extras[k] = pr.structure[k]
                        if wall.profile is not None:
                            extras["profiled"] = True
                    if ob is not None and ob.plane is not None:
                        extras["plane"] = ob.plane.to_meta()
                    if rel_recs:
                        extras["relief"] = rel_recs
                        extras["piece_u"] = [round(float(u0), 4), round(float(u1), 4)]   # u of the frame corners
                    if rel_rep is not None and rel_rep["skipped"]:
                        extras["relief_skipped"] = rel_rep["skipped"]
                    name = f"wall-{wall.index}" if len(pieces) == 1 else f"wall-{wall.index}-{pi}"
                    if is_obs:
                        tw = fac.texture.shape[1]
                        c0 = int(np.clip(math.floor(u0 / wall.length * tw), 0, tw - 1))
                        c1 = int(np.clip(math.ceil(u1 / wall.length * tw), c0 + 1, tw))
                        tex = np.ascontiguousarray(fac.texture[:, c0:c1])
                        codes = np.ascontiguousarray(texel_provenance(fac)[:, c0:c1])
                        span = max(wall.z1 - wall.z0, 1e-9)
                        uvs = np.stack([(uz[:, 0] / wall.length * tw - c0) / (c1 - c0), (wall.z1 - uz[:, 1]) / span], 1)
                        extras["observed_fraction"] = round(float(fac.mask[:, c0:c1].mean()), 3)   # fill excluded
                        extras["base_color"] = COLOR_NAME_OF_RGB.get(tuple(fac.base_rgb), "")
                        if fac.fill is not None:
                            m2_per_texel = 1.0 / max(fac.px_per_m, 1e-9) ** 2
                            n_holes = cv2.connectedComponents((codes == 2).astype(np.uint8), connectivity=8)[0] - 1
                            extras["fill"] = {"holes": int(n_holes), "area_m2": round(float((codes == 2).sum()) * m2_per_texel, 3),
                                              "px_per_m": round(float(fac.px_per_m), 4),
                                              "texture_px": [int(tw), int(fac.texture.shape[0])]}
                        extras["photos"] = photo_refs.get(wall.index, [])
                        parts.append(Part(name, pos, nrm, tri_idx, fac.base_rgb, tex, uvs, extras,
                                          codes))
                    else:
                        if ob is not None and ob.reason:
                            extras["reason"] = ob.reason
                        elif fac is not None:
                            extras["reason"] = "not in any photo's observed part of this facade"
                        parts.append(Part(name, pos, nrm, tri_idx, CLAY_RGB, None, None, extras))
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
                                       "provenance": _provenance(pr.outline_source, hp, PROV_PHOTO if obs_ else PROV_DEFAULT),
                                       **({"part": pr.part, **_d15_extras(pr)} if pr.part is not None else {})}))
        parts += _cap_parts(pr, rings_b, RoofLook() if pr.stub else roof, segs, params)
    return parts


def _face_plane_z(pr: Prism, face: int) -> tuple[Any, np.ndarray]:
    """(z(xy) function, unit normal) of a cap face: flat at z1, the plane of a sloped pyramid face, or the plane of
    a structure part's roof face."""
    if pr.cap == "faces":
        f = pr.roof_faces[face]
        n = np.asarray(f.normal, float) / max(np.linalg.norm(f.normal), 1e-12)
        a = np.asarray(f.polys[0][0], float)
        return (lambda xy: a[2] - ((xy[:, 0] - a[0]) * n[0] + (xy[:, 1] - a[1]) * n[1]) / max(n[2], 1e-6)), n
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
        if pr.part is not None:
            more = {**more, "part": pr.part, "geometry": pr.structure.get("state", ""), "roof": pr.structure.get("roof", {}),
                    **({"adopted": pr.structure["adopted"]} if pr.structure.get("adopted") else {}),
                    **({"outline_adopted": pr.structure["outline_adopted"]} if pr.structure.get("outline_adopted") else {}),
                    **{k: pr.structure[k] for k in ("corrected", "tilt") if pr.structure.get(k)}}
        return {"kind": "roof", "prism": pr.index, "faces": list(faces), "observed": observed, "locked": not observed,
                "coverage": cov, "area_m2": round(area, 2),
                "provenance": _provenance(pr.outline_source, height_src, PROV_PHOTO if observed else PROV_DEFAULT), **more}

    if pr.cap == "faces":
        # a structure part's roof primitive: faces grouped by colour (observed as a whole) or clay; the shape stays
        # whatever coverage says (coverage decides colour only)
        groups: dict[Any, list[int]] = {}
        for k in range(len(pr.roof_faces)):
            groups.setdefault(roof.face_rgb.get((pr.index, k)), []).append(k)
        for rgb, faces in groups.items():
            pos, nrm, idx = [], [], []
            for k in faces:
                f = pr.roof_faces[k]
                nn = np.asarray(f.normal, float) / max(np.linalg.norm(f.normal), 1e-12)
                for poly in f.polys:
                    for tri in _planar_triangles(poly, nn):
                        b = len(pos)
                        pos += list(tri)
                        nrm += [nn] * 3
                        idx.append([b, b + 1, b + 2])
            if not idx:
                continue
            pos, nrm, idx = np.array(pos), np.array(nrm), np.array(idx)
            obs = rgb is not None
            parts.append(Part(f"roof-{pr.index}-{'obs' if obs else 'clay'}-{faces[0]}", pos, nrm, idx, rgb or CLAY_ROOF_RGB,
                              None, None, roof_extras(faces, obs, pr.cap_source, _tri_area(pos, idx), cap=pr.roof_shape)))
    elif pr.cap == "pyramid":
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
                               "provenance": _provenance(pr.outline_source, PROV_DEFAULT, PROV_DEFAULT),
                               **({"part": pr.part, **_d15_extras(pr)} if pr.part is not None else {})}))
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
                                                     PROV_PHOTO if obs else PROV_DEFAULT),
                           **({"part": pr.part, **_d15_extras(pr)} if pr.part is not None else {})}))
    if pr.part is not None and pr.z0 > 0.05:
        # a raised part's underside (a lantern on a roof, an overhang): clay, it closes the solid
        verts, tris = triangulate_with_holes(rings_b[0], rings_b[1:])
        pos = np.array([[*v, pr.z0] for v in verts])
        idx = np.array([(i, k, j) for i, j, k in tris])
        z0_src = pr.structure.get("params", {}).get("min_height", [0, "", PROV_GROUND])[2]
        parts.append(Part(f"base-{pr.index}", pos, np.repeat(np.array([[0.0, 0.0, -1.0]]), len(pos), axis=0), idx,
                          CLAY_ROOF_RGB, None, None,
                          {"kind": "base", "prism": pr.index, "observed": False, "locked": True, "part": pr.part,
                           "area_m2": round(_tri_area(pos, idx), 2),
                           "provenance": _provenance(pr.outline_source, z0_src, PROV_DEFAULT), **_d15_extras(pr)}))
    return parts


def _d15_extras(pr: Prism) -> dict[str, Any]:
    """D15's records for every node of a structure part (walls and roofs carry them already): the parameters adopted
    or carried, and the adopted outline (the frame), each with the map's old value."""
    st_ = pr.structure or {}
    return {k: st_[k] for k in ("adopted", "outline_adopted", "corrected", "tilt") if st_.get(k)}


STRUCTURE_LINE_TYPES = ("ridge", "hip", "valley", "eave", "rake")
LINE_CONFIRM_M = 0.3           # a roof line is confirmed by 3-D lines within this distance ...
LINE_CONFIRM_DEG = 7.0         # ... and angle, covering >= max(1 m, 25 %) of it, seen in >= 3 photos ...
LINE_CONFIRM_PHOTOS = 3
LINE_NULL_SHIFT_M = 1.5        # ... and better than its displaced copies (1.5 m up / down / out / in, turned +-15 deg,
LINE_NULL_NEAR_M = (0.4, 0.8)  #     and NEAR copies 0.4 / 0.8 m up / down / out / in, where an edge the map misplaces
LINE_NULL_DEG = 15.0           #     shows up): its covered share must beat the best copy's by LINE_NULL_MARGIN (a
LINE_NULL_MARGIN = 0.1         #     per-line null); a near copy beating the line itself reports an offset
LINE_NULL_ALTERNATIVES = 4 + 8 + 2      # far shifts, near shifts, turns: what a verified line's evidence must have tested
LINE_GEOM_TOL_M = 0.05         # audit_glb: every structure line lies on an edge of its part's roof mesh of its kind


def confirm_segment(a: np.ndarray, b: np.ndarray, lines: Any) -> dict[str, Any] | None:
    """3-D lines (crowd3d.lines3d.Lines3D: A, B, photos) that lie along the segment a-b: {"lines", "photos", "cover"}
    when they cover >= max(1 m, 25 %) of it with >= LINE_CONFIRM_PHOTOS photos, else None."""
    if lines is None or not len(lines):
        return None
    a, b = np.asarray(a, float), np.asarray(b, float)
    L = float(np.linalg.norm(b - a))
    if L < 1e-6:
        return None
    t = (b - a) / L
    A, B = np.asarray(lines.A, float), np.asarray(lines.B, float)
    D = B - A
    ln = np.maximum(np.linalg.norm(D, axis=1), 1e-12)
    par = np.abs(D @ t) / ln >= math.cos(math.radians(LINE_CONFIRM_DEG))

    def off(X):
        r = X - a
        return np.linalg.norm(r - (r @ t)[:, None] * t[None], axis=1)

    near = par & (off(A) <= LINE_CONFIRM_M) & (off(B) <= LINE_CONFIRM_M)
    idx = np.flatnonzero(near)
    if not len(idx):
        return None
    s0, s1 = np.clip(np.minimum((A[idx] - a) @ t, (B[idx] - a) @ t), 0, L), np.clip(np.maximum((A[idx] - a) @ t, (B[idx] - a) @ t), 0, L)
    iv = sorted((float(x), float(y)) for x, y in zip(s0, s1) if y > x)
    cover, cur = 0.0, None
    for x, y in iv:
        if cur is None or x > cur[1]:
            if cur is not None:
                cover += cur[1] - cur[0]
            cur = [x, y]
        else:
            cur[1] = max(cur[1], y)
    if cur is not None:
        cover += cur[1] - cur[0]
    photos = sorted({int(p) for k in idx for p in lines.photos[k]})
    if cover < min(L, max(1.0, 0.25 * L)) or len(photos) < LINE_CONFIRM_PHOTOS:
        return None
    return {"lines": int(len(idx)), "photos": len(photos), "cover": round(cover / L, 3)}


def line_alternatives(a: np.ndarray, b: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """The displaced copies a roof line is tested against (its per-line null): LINE_NULL_SHIFT_M up and down, out and
    in (horizontal, across the line), turned +-LINE_NULL_DEG about the vertical through its middle, and the near
    shifts LINE_NULL_NEAR_M up / down / out / in (names 'up0.4' ...: where the photos put an edge the map places
    a little off)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    h = np.array([-d[1], d[0], 0.0])
    h = h / np.linalg.norm(h) if np.linalg.norm(h) > 1e-9 else np.array([1.0, 0.0, 0.0])
    up = np.array([0.0, 0.0, 1.0])
    s = LINE_NULL_SHIFT_M
    out = [("up", a + s * up, b + s * up), ("down", a - s * up, b - s * up),
           ("out", a + s * h, b + s * h), ("in", a - s * h, b - s * h)]
    m = 0.5 * (a + b)
    for sg in (1, -1):
        c, sn = math.cos(math.radians(sg * LINE_NULL_DEG)), math.sin(math.radians(sg * LINE_NULL_DEG))
        R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]])
        out.append((f"turn{'+' if sg > 0 else '-'}{LINE_NULL_DEG:g}", m + R @ (a - m), m + R @ (b - m)))
    for s in LINE_NULL_NEAR_M:
        for name, v in (("up", up), ("down", -up), ("out", h), ("in", -h)):
            out.append((f"{name}{s:g}", a + s * v, b + s * v))
    return out


def verify_segment(a: np.ndarray, b: np.ndarray, lines: Any) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(verification, check) of a roof line against the 3-D lines: check = confirm_segment on the line itself
    plus its per-line null (the displaced copies of line_alternatives: how many of them the same 3-D lines also
    'confirm' -- the chance rate -- their best covered share and which copy that is) and, when a NEAR copy covers
    LINE_NULL_MARGIN more than the line itself, the offset the photos show ({"shift", "cover"}: the line's position
    is off, its value stays the map's); verification = check when the line is confirmed AND its covered share beats
    every copy's by LINE_NULL_MARGIN, else None. check is None without 3-D lines."""
    if lines is None or not len(lines):
        return None, None
    own = confirm_segment(a, b, lines)
    alts = [(name, confirm_segment(a2, b2, lines)) for name, a2, b2 in line_alternatives(a, b)]
    hits = [(name, c) for name, c in alts if c is not None]
    best = max(hits, key=lambda x: x[1]["cover"], default=(None, None))
    null = {"alternatives": len(alts), "confirmed": len(hits), "chance": round(len(hits) / len(alts), 3),
            "best_cover": round(best[1]["cover"] if best[1] else 0.0, 3), "best": best[0]}
    own_cover = own["cover"] if own is not None else 0.0
    near = [(name, c) for name, c in hits if any(name == f"{d}{s_:g}" for s_ in LINE_NULL_NEAR_M
                                                 for d in ("up", "down", "out", "in"))]
    nb = max(near, key=lambda x: x[1]["cover"], default=(None, None))
    offset = ({"shift": nb[0], "cover": nb[1]["cover"]}
              if nb[1] is not None and nb[1]["cover"] >= own_cover + LINE_NULL_MARGIN else None)
    check = {"confirmed": own is not None, **(own or {}), "null": null, "offset": offset}
    ok = own is not None and own["cover"] >= null["best_cover"] + LINE_NULL_MARGIN
    return (check if ok else None), check


def roof_boundary_segments(T: np.ndarray, tol: float = 1e-3) -> list[tuple[np.ndarray, np.ndarray]]:
    """The outline of a roof mesh (edges of exactly one triangle), consecutive collinear pieces joined: the eaves
    (horizontal) and rakes (sloped) where the roof meets its walls."""
    P = T.reshape(-1, 3)
    key = np.round(P / tol).astype(np.int64)
    _, vid = np.unique(key, axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    pos = np.zeros((vid.max() + 1, 3))
    pos[vid.ravel()] = P
    cnt: dict[tuple[int, int], int] = {}
    for t in vid:
        for k in range(3):
            a, b = int(t[k]), int(t[(k + 1) % 3])
            if a != b:
                e = (min(a, b), max(a, b))
                cnt[e] = cnt.get(e, 0) + 1
    edges = [e for e, c in cnt.items() if c == 1]
    adj: dict[int, list[int]] = {}
    for a, b in edges:
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    used: set[tuple[int, int]] = set()
    out = []
    for a, b in edges:
        if (a, b) in used:
            continue
        used.add((a, b))
        chain = [a, b]
        for end in (0, 1):                       # grow both ways while the next edge is collinear
            while True:
                tip, prev = (chain[-1], chain[-2]) if end else (chain[0], chain[1])
                d0 = pos[tip] - pos[prev]
                d0 /= max(np.linalg.norm(d0), 1e-12)
                nxt = None
                for c in adj.get(tip, []):
                    e = (min(tip, c), max(tip, c))
                    if e in used or c == prev:
                        continue
                    d1 = pos[c] - pos[tip]
                    if np.linalg.norm(d1) > 1e-9 and float(d0 @ d1) / np.linalg.norm(d1) > 0.9995:
                        nxt = (c, e)
                        break
                if nxt is None:
                    break
                used.add(nxt[1])
                if end:
                    chain.append(nxt[0])
                else:
                    chain.insert(0, nxt[0])
        out.append((pos[chain[0]].copy(), pos[chain[-1]].copy()))
    return out


def structure_lines(massing: Massing, lines: Any = None) -> list[dict[str, Any]]:
    """The roof's structural lines of every drawn (non-stub) structure part: ridges, hips and valleys, i.e. creases
    >= 20 deg between its roof faces (crowd3d.structmetrics.part_mesh_census; a curved roof has none), and its eaves
    and rakes (the roof's outline, horizontal or sloped), as drawn from the map's values. Each carries its part,
    whether its roof height is an S3DB convention (D17: drawn dashed until photos verify it) and, when the 3-D lines
    lie along it better than along its displaced copies (verify_segment: confirmed within LINE_CONFIRM_M and above
    its per-line chance, near copies included), its photo verification with that chance rate -- the values stay the
    map's (photo-verified, not fitted). A line the 3-D lines confirm no better than its copies carries line_check
    (status 'at chance'), and one a near copy beats carries line_check with status 'offset' (the photos see the edge
    displaced; not verified, nothing adopted)."""
    from . import structmetrics as SM
    out: list[dict[str, Any]] = []
    for pr in massing.prisms:
        if pr.part is None or pr.cap != "faces" or pr.stub or not pr.roof_faces:
            continue
        tris = []
        for f in pr.roof_faces:
            nn = np.asarray(f.normal, float) / max(np.linalg.norm(f.normal), 1e-12)
            for poly in f.polys:
                tris += _planar_triangles(poly, nn)
        if not tris:
            continue
        T = np.array(tris, float).reshape(-1, 3, 3)
        curved = (pr.roof_shape or "") in SM.CURVED_ROOFS
        mc = SM.part_mesh_census(T, ["roof"] * len(T), np.zeros(len(T), bool), curved, segments=True)
        rh = (pr.structure.get("params") or {}).get("roof_height") or [None, "", ""]
        convention = rh[2] == PROV_SHAPE_DEFAULT and rh[1] not in ("photo-verified", "photo-fitted", "curator")
        segs = [(g["type"], a, b) for g in mc["lines"] if g["type"] in ("ridge", "hip", "valley") for a, b in g["segs"]]
        for a, b in roof_boundary_segments(T):
            L = float(np.linalg.norm(b - a))
            if L >= 0.05:
                segs.append(("eave" if abs(b[2] - a[2]) < 0.05 * L + 0.02 else "rake", a, b))
        for typ, a, b in segs:
            rec = {"part": pr.part, "type": typ, "a": [round(float(v), 3) for v in a],
                   "b": [round(float(v), 3) for v in b], "convention": bool(convention),
                   "source": rh[2] or pr.structure.get("roof", {}).get("height_source", "")}
            ver, chk = verify_segment(a, b, lines)
            if ver is not None:
                rec["photo_verified"] = ver
            elif chk is not None and chk["offset"] is not None:
                rec["line_check"] = dict(chk, status="offset")   # the photos put this edge elsewhere: not verified
            elif chk is not None and chk["confirmed"]:
                rec["line_check"] = dict(chk, status="at chance")   # no better than its displaced copies: not counted
            out.append(rec)
    return out


def structure_lines_extras(recs: list[dict[str, Any]]) -> dict[str, Any]:
    """The GLB scene extras' compact form, in glTF axes (x = east, y = up, z = -north): per line
    [ax, ay, az, bx, by, bz, type index, convention 0/1, photo-verified 0/1, part index]."""
    parts = sorted({r["part"] for r in recs})
    pi = {p: i for i, p in enumerate(parts)}
    g = lambda v: [round(v[0], 3), round(v[2], 3), round(-v[1], 3)]
    return {"frame": "glTF (x=east, y=up, z=-north)", "types": list(STRUCTURE_LINE_TYPES), "parts": parts,
            "fields": ["ax", "ay", "az", "bx", "by", "bz", "type", "convention", "photo_verified", "part"],
            "lines": [g(r["a"]) + g(r["b"]) + [STRUCTURE_LINE_TYPES.index(r["type"]), int(r["convention"]),
                                                 int("photo_verified" in r), pi[r["part"]]] for r in recs],
            # the evidence of every photo-verified line (row index -> 3-D lines, photos, covered share, and its per-line
            # null: displaced copies tested, copies the same 3-D lines 'confirm', their best covered share)
            "verified_fields": ["lines", "photos", "cover", "null_alternatives", "null_confirmed", "null_best_cover"],
            "verified": {str(i): [r["photo_verified"]["lines"], r["photo_verified"]["photos"], r["photo_verified"]["cover"],
                                  r["photo_verified"]["null"]["alternatives"], r["photo_verified"]["null"]["confirmed"],
                                  r["photo_verified"]["null"]["best_cover"]]
                         for i, r in enumerate(recs) if "photo_verified" in r}}


def roof_edge_segments(tris: np.ndarray, crease_deg: float = 20.0, convexity: bool = False) -> tuple[np.ndarray, ...]:
    """(creases, boundary) of a roof mesh (n, 3, 3), each (k, 2, 3) segments: the edges between two of its triangles
    whose up-facing normals differ by >= crease_deg (ridges, hips, valleys), and the edges of exactly one triangle
    (eaves, rakes). Vertices are welded to 1 mm and an edge is split at vertices lying on it (separately triangulated
    nodes of one roof meet at T-junctions). convexity: (convex creases, concave creases, boundary) instead -- a crease
    is convex (a ridge or a hip) when the second face's centre lies below the first face's plane, concave (a valley)
    when above it (crowd3d.structmetrics.part_mesh_census's test)."""
    T = np.asarray(tris, float).reshape(-1, 3, 3)
    if not len(T):
        return (np.zeros((0, 2, 3)),) * (3 if convexity else 2)
    va = np.cross(T[:, 1] - T[:, 0], T[:, 2] - T[:, 0])
    ar = np.linalg.norm(va, axis=1)
    keep = ar > 1e-9
    nrm = np.zeros_like(va)
    nrm[keep] = va[keep] / ar[keep, None]
    nrm[nrm[:, 1] < 0] *= -1.0                               # glTF y-up: roofs face up
    _, vid = np.unique(np.round(T.reshape(-1, 3) * 1000).astype(np.int64), axis=0, return_inverse=True)
    vid = vid.reshape(-1, 3)
    vpos = np.zeros((vid.max() + 1, 3))
    vpos[vid.ravel()] = T.reshape(-1, 3)
    edges: dict[tuple[int, int], list[int]] = {}
    for t in np.flatnonzero(keep):
        for k in range(3):
            a, b = int(vid[t, k]), int(vid[t, (k + 1) % 3])
            if a == b:
                continue
            pa, d = vpos[a], vpos[b] - vpos[a]
            u = (vpos - pa) @ d / max(float(d @ d), 1e-12)
            dist = np.linalg.norm(vpos - pa - u[:, None] * d, axis=1)
            mid = np.flatnonzero((u > 1e-6) & (u < 1 - 1e-6) & (dist < 1e-3))
            chain = [a] + [int(m) for m in mid[np.argsort(u[mid])]] + [b]
            for x, y in zip(chain[:-1], chain[1:]):
                edges.setdefault((min(x, y), max(x, y)), []).append(int(t))
    cos_cr = math.cos(math.radians(crease_deg))
    cent = T.mean(axis=1)
    crease, convex, concave, boundary = [], [], [], []
    for (a, b), ts in edges.items():
        seg = (vpos[a], vpos[b])
        if len(ts) == 1:
            boundary.append(seg)
        elif any(float(nrm[ts[i]] @ nrm[ts[j]]) < cos_cr for i in range(len(ts)) for j in range(i + 1, len(ts))):
            crease.append(seg)
            (convex if float((cent[ts[1]] - vpos[a]) @ nrm[ts[0]]) < 0 else concave).append(seg)
    as_arr = lambda x: np.array(x, float).reshape(-1, 2, 3)
    if convexity:
        return as_arr(convex), as_arr(concave), as_arr(boundary)
    return as_arr(crease), as_arr(boundary)


def _dist_to_segments(X: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Distance of each point of X (n, 3) to the nearest of the segments S (k, 2, 3) (inf without segments)."""
    if not len(S):
        return np.full(len(X), np.inf)
    a, d = S[:, 0], S[:, 1] - S[:, 0]
    dd = np.maximum(np.einsum("ij,ij->i", d, d), 1e-12)
    t = np.clip(np.einsum("nkj,kj->nk", X[:, None, :] - a[None], d) / dd[None], 0.0, 1.0)
    return np.linalg.norm(X[:, None, :] - (a[None] + t[..., None] * d[None]), axis=2).min(axis=1)


def audit_structure_lines(sl: Any, stub_ids: set | None = None, roof_tris: dict[str, np.ndarray] | None = None) -> list[str]:
    """What is wrong with a GLB's structure_lines (empty: fine): well-formed rows of known types and parts, no line on
    a 'height pending' stub, every photo-verified line with its evidence (>= LINE_CONFIRM_PHOTOS photos, lines
    covering part of it) and its per-line null (all LINE_NULL_ALTERNATIVES displaced copies, near ones included,
    tested; its covered share LINE_NULL_MARGIN above the best copy's), and -- with roof_tris, its part's roof mesh
    per part id (glTF axes, as the lines) -- every line ON that mesh: points every 0.5 m along it within
    LINE_GEOM_TOL_M of an edge of its kind (ridge / hip: a convex crease of >= 20 deg between two roof faces; valley:
    a concave one; eave / rake: the roof's boundary), and ridges and eaves level, hips and rakes sloped. A line the
    viewer would draw where the roof has no such edge (floating, moved, made up, a hip called a valley) fails."""
    if not sl:
        return []
    v = []
    types, parts, ver = sl.get("types") or [], sl.get("parts") or [], sl.get("verified") or {}
    edge_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i, row in enumerate(sl.get("lines") or []):
        if not (isinstance(row, list) and len(row) >= 10 and all(isinstance(x, (int, float)) for x in row[:10])):
            v.append(f"line {i}: malformed")
            continue
        if not (0 <= int(row[6]) < len(types)) or not (0 <= int(row[9]) < len(parts)):
            v.append(f"line {i}: unknown type or part")
            continue
        part, typ = parts[int(row[9])], types[int(row[6])]
        if stub_ids and part in stub_ids:
            v.append(f"line {i}: on the stub part {part}")
        if row[8]:
            ev = ver.get(str(i))
            if not (isinstance(ev, list) and len(ev) == 6 and all(isinstance(x, (int, float)) for x in ev)
                    and ev[0] >= 1 and ev[1] >= LINE_CONFIRM_PHOTOS and ev[2] > 0):
                v.append(f"line {i}: photo-verified without its evidence")
            elif not (ev[3] >= LINE_NULL_ALTERNATIVES and ev[2] >= ev[5] + LINE_NULL_MARGIN - 1e-9):
                v.append(f"line {i}: photo-verified no better than its displaced copies (chance {ev[4]}/{ev[3]})")
        if roof_tris is None:
            continue
        a, b = np.array(row[0:3], float), np.array(row[3:6], float)
        L = float(np.linalg.norm(b - a))
        level = abs(b[1] - a[1]) < 0.05 * L + 0.02                   # glTF y is up
        if L < 1e-6 or typ not in STRUCTURE_LINE_TYPES or level != (typ in ("ridge", "eave")) and typ != "valley":
            v.append(f"line {i}: a {typ} that is {'level' if level else 'sloped'}")
            continue
        if part not in roof_tris:
            v.append(f"line {i}: its part {part} has no roof mesh")
            continue
        if part not in edge_cache:
            edge_cache[part] = roof_edge_segments(roof_tris[part], convexity=True)
        convex, concave, boundary = edge_cache[part]
        X = a[None] + np.linspace(0.0, 1.0, max(2, int(math.ceil(L / 0.5)) + 1))[:, None] * (b - a)[None]
        want, name = {"ridge": (convex, "convex creases"), "hip": (convex, "convex creases"),
                      "valley": (concave, "concave creases")}.get(typ, (boundary, "boundary"))
        dist = _dist_to_segments(X, want)
        if not (dist <= LINE_GEOM_TOL_M).all():
            v.append(f"line {i}: a {typ} {float(dist.max()):.2f} m off the {name} of its part's roof mesh")
    return v


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


def check_fill(fac: StylizedFacade, max_m2: float, params: StyleParams = DEFAULT_PARAMS) -> dict[str, int]:
    """D12 on one facade: every observed-fill region (8-connected) must be enclosed (it does not reach the
    texture's edge and all eight neighbours of its texels are observed texels of this wall, within the
    columns the GLB shows), be at most max_m2, be unable to hold a whole drawable element
    (fill_shape_kernels()), lie outside the observation mask and hold the facade's base colour only."""
    out = {"regions": 0, "not_enclosed": 0, "over_cap": 0, "could_hide": 0, "overlapping_observed": 0,
           "texels_not_base": 0}
    fill = fac.fill
    if fill is None or not fill.any():
        return out
    h, w = fill.shape
    ppm = fac.px_per_m if fac.px_per_m > 0 else 1.0
    kernels = fill_shape_kernels(ppm, params, (h, w))
    shown = fac.mask.copy()
    if fac.face_cols is not None:
        shown[:, :fac.face_cols[0]] = False
        shown[:, fac.face_cols[1]:] = False
    out["overlapping_observed"] = int((fill & fac.mask).sum())
    out["texels_not_base"] = int(np.any(fac.texture[fill] != np.array(fac.base_rgb, np.uint8), axis=-1).sum())
    n, cc, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), connectivity=8)
    out["regions"] = n - 1
    ring_kernel = np.ones((3, 3), np.uint8)
    for c in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[c])
        if area / ppm ** 2 > max_m2 + 1e-9:
            out["over_cap"] += 1
        if could_hide_element(cc[y:y + bh, x:x + bw] == c, kernels):
            out["could_hide"] += 1
        if x == 0 or y == 0 or x + bw >= w or y + bh >= h:
            out["not_enclosed"] += 1
            continue
        sl = (slice(y - 1, y + bh + 1), slice(x - 1, x + bw + 1))
        comp = (cc[sl] == c).astype(np.uint8)
        ring = (cv2.dilate(comp, ring_kernel) > 0) & (comp == 0)
        if not shown[sl][ring].all():
            out["not_enclosed"] += 1
    return out


def check_honesty(model_parts: list[Part], facades: dict[int, StylizedFacade],
                  params: StyleParams = DEFAULT_PARAMS, stub_parts: set[str] | None = None) -> dict[str, Any]:
    """QA on what the pipeline itself enforced: outside a facade's observation mask and its legal
    observed-fill (D12) only the clay pattern, every fill region enclosed by observed texels, at most
    fill_max_hole_m2, unable to hold a whole drawable element and in the base colour only, no texture on
    locked faces, provenance on every face, palette colours only, and every face of a 'height pending' stub part
    (stub_parts: ids, D13) clay: never observed, textured or claiming photos. The independent checks are in
    audit_model() and audit_glb()."""
    outside = 0
    fill_tot = {"regions": 0, "not_enclosed": 0, "over_cap": 0, "could_hide": 0, "overlapping_observed": 0,
                "texels_not_base": 0}
    filled_m2 = 0.0
    for fac in facades.values():
        legal = fac.mask | (fac.fill if fac.fill is not None else False)
        off = ~legal
        if off.any():
            outside += int((~is_clay_pattern(fac.texture[off])).sum())
        for k, v in check_fill(fac, params.fill_max_hole_m2, params).items():
            fill_tot[k] += v
        if fac.fill is not None and fac.px_per_m > 0:
            filled_m2 += float(fac.fill.sum()) / fac.px_per_m ** 2
    locked_detail = sum(1 for p in model_parts if p.extras.get("locked") and p.texture is not None)
    locked_photo = sum(1 for p in model_parts if p.extras.get("locked")
                       and p.extras.get("provenance", {}).get("appearance") == PROV_PHOTO)
    missing = sum(1 for p in model_parts
                  if not all(p.extras.get("provenance", {}).get(k) in PROVENANCES for k in PROVENANCE_KEYS))
    stubs = stub_parts or set()
    stub_bad = sum(1 for p in model_parts if p.extras.get("part") in stubs
                   and (p.texture is not None or p.extras.get("observed")
                        or p.extras.get("provenance", {}).get("appearance") != PROV_DEFAULT
                        or tuple(int(v) for v in p.rgb) not in (CLAY_RGB, CLAY_ROOF_RGB, CLAY_HATCH_RGB)))
    allowed = allowed_texture_colors()
    off_palette = 0
    for p in model_parts:
        if p.texture is None:
            continue
        cols = {tuple(int(v) for v in c) for c in np.unique(p.texture.reshape(-1, 3), axis=0)}
        off_palette += len(cols - allowed)
    fill_bad = (fill_tot["not_enclosed"] + fill_tot["over_cap"] + fill_tot["could_hide"] + fill_tot["overlapping_observed"]
                + fill_tot["texels_not_base"])
    # photo-measured facade relief (crowd3d.relief): only on photographed walls of non-stub parts, every element with
    # its evidence record (photo-measured, both photo halves, a line-confirmed edge, every guard passed; or, D24, built
    # on the points alone: its lines at chance, its D24 evidence and the model's scale condition in the record)
    from . import relief as RL
    relief_elements, relief_bad = 0, []
    for p in model_parts:
        rel = p.extras.get("relief")
        if not rel:
            continue
        relief_bad += [{"node": p.name, "why": w} for w in RL.node_violations(p.extras, p.texture is not None, stubs)]
        for r in rel:
            relief_elements += 1
            relief_bad += [{"node": p.name, "element": r.get("id"), "why": w} for w in RL.record_violations(r)]
    relief_bad += RL.copied_records([(p.name, p.extras.get("wall"), p.extras.get("relief")) for p in model_parts
                                     if p.extras.get("relief")])
    return {"detail_pixels_outside_mask": outside, "textured_locked_faces": locked_detail,
            "locked_faces_claiming_photos": locked_photo, "faces_without_provenance": missing,
            "stub_faces_not_clay": stub_bad,
            "off_palette_colors": off_palette,
            "fill_regions": fill_tot["regions"], "fill_m2": round(filled_m2, 3),
            "fill_regions_not_enclosed": fill_tot["not_enclosed"], "fill_regions_over_cap": fill_tot["over_cap"],
            "fill_regions_could_hide": fill_tot["could_hide"],
            "fill_texels_overlapping_observed": fill_tot["overlapping_observed"],
            "fill_texels_not_base": fill_tot["texels_not_base"],
            "relief_elements": relief_elements, "relief_unbacked": len(relief_bad), "relief_violations": relief_bad[:20],
            "passed": (outside == 0 and locked_detail == 0 and locked_photo == 0 and missing == 0 and off_palette == 0
                       and fill_bad == 0 and stub_bad == 0 and not relief_bad)}


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
                eligible: np.ndarray | None = None, registration_mode: str = "osm-footprint",
                track: np.ndarray | None = None, lines: Any = None,
                structure_lines: dict[str, Any] | None = None, structure: Any = None, snap_scale: float = 1.0,
                quality: np.ndarray | None = None, tracks: Any = None) -> dict[str, Any]:
    """Independent publication audit: recomputes from the inputs (points, cameras) what each face claims
    and fails on a mismatch, instead of re-checking what the pipeline just enforced.

    - roof faces / patches claiming photos: point support re-measured by sampling the face itself on a
      1 m lattice and looking for roof points in its 1 m column (not the roof grid used to decide), and for
      every plane among its triangles at least roof_min_cameras cameras in front of it with some of it in frame;
    - no face of a 'height pending' stub part (D13, read from the prism's structure record) claims photos;
    - textured walls: the points' distance to the face over the textured strip (0.25 m histogram) must
      peak within plane_tol_m + 0.25 m of the face, at least half as high as anywhere in +-plane_search_m;
    - walls whose height claims sfm-points must have their own top (from their points) within step_m of it;
    - occlusion: for sampled observed facade pixels, occluder_min_count or more (well triangulated) SfM
      points more than occluder_margin_m in front of the wall must not lie within 0.3 m of the ray from
      a contributing camera (a geometric segment test, not the per-camera depth buffer that built the mask);
    - registration re-checked edge by edge (outline from OSM only);
    - facade relief (crowd3d.relief): a node carrying it is a photographed wall of a non-stub part (read from the
      prism's structure record), its geometry stays on its own wall piece within the depth limit, every built element
      lies >= 50 % on the orthophoto's own geometric observation mask; the wall passes G-R1 re-measured from the raw
      points and recounted from the cameras (photos seeing it per photo half), and every record's frame lies within
      FRAME_TOL_M of the wall's frame re-derived from the points (independent_frame); every record's offset,
      re-measured from the raw points in its box against that re-derived frame and against the points right beside
      it (audit_points: no relief map, no levels, nothing the record declares as its reference; over its outline as
      built, an arch's head included), agrees with its record, and a recess holds no structure at the frame plane
      inside it (crowd3d.relief.interior_structure: a pier or the frame band between two bays the record merged into
      one; an interior the points cannot measure is unverifiable, a violation); its line evidence (G-R3), recomputed
      from the 3-D lines `lines` (crowd3d.relief.line_evidence: nothing from the record's own line numbers), passes --
      a relief record without 3-D lines to check it against fails; and its photo-split evidence (G-R2) is RE-RUN here
      for every record from the 2-D observations `tracks` (crowd3d.relief.PointTracks: the wall's points
      re-triangulated from each half's photos and measured again, cut at the verticals the audit's own first pass
      confirms -- crowd3d.relief.halves_cutters, never what a record names): each record found again in both halves of
      both splits as the same element (split_match: IoU, depths, its sides -- a band's bottom and top -- within
      split_box_m of the record) -- without tracks a relief record fails. A D24 record (provenance photo-measured
      (points, D24): built on the points alone) passes the line
      check with its lines AT chance, but needs >= 1 of its edges at a line cluster in that recomputation, and the
      model's D24 scale condition re-derived here (crowd3d.relief.d24_scale_condition on these points and 3-D lines,
      the structure frame `structure`, the snap's scale `snap_scale`, the points' `quality`; nothing from the record)
      must be met, with D24 switched on; and, since the photo split is a D24 element's only second channel, in the
      re-run splits every edge of it within d24_edge_m between the halves and d24_box_m of the record
      (d24_halves_check), the cutters it records (halves_cut_at) the ones the audit derives, and no bottom / top at a
      register boundary made by a band that is not an accepted record;
    - roof structure lines (`structure_lines`, the GLB's scene extras): every line flagged photo-verified is verified
      again from the 3-D lines (verify_segment on its own segment), or fails.
    """
    out: dict[str, Any] = {"roof_mismatches": [], "wall_plane_mismatches": [], "height_mismatches": [],
                           "occlusion": {}, "registration": None, "stub_mismatches": [], "roof_unfaced": [],
                           "relief": {"nodes": 0, "elements_built": 0, "points_checked": 0, "unverifiable": 0,
                                      "violations": []}}
    # a 'height pending' stub (D13) is a placeholder, not the building: none of its faces may claim photos. Read from
    # each prism's structure record (the part's height parameter), not from the flag the pipeline decided with
    for p in parts:
        pi = p.extras.get("prism")
        if pi is None or not (0 <= pi < len(massing.prisms)) or massing.prisms[pi].part is None:
            continue
        hrec = massing.prisms[pi].structure.get("params", {}).get("height", [None, "", ""])
        if (hrec[2] == PROV_DEFAULT or hrec[1] == "stub") and (
                p.extras.get("provenance", {}).get("appearance") != PROV_DEFAULT or p.texture is not None):
            out["stub_mismatches"].append({"part": p.name, "structure_part": massing.prisms[pi].part})
    _audit_wall_planes(out["relief"], massing, parts)
    # D25 (photo-corrected): every correction re-run on its own numbers, backed, matching the model's geometry
    out["d25"] = _audit_d25_model(structure)
    if points is None or not len(points):
        out["structure_lines"] = recheck_structure_lines(structure_lines, lines)
        out["passed"] = not any(p.extras.get("provenance", {}).get("appearance") == PROV_PHOTO for p in parts) \
            and not out["stub_mismatches"] and not any(p.extras.get("relief") for p in parts) \
            and not out["relief"]["violations"] and not out["structure_lines"]["violations"] and not out["d25"]
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
        # ... and a photo must have faced it: for every plane among its triangles (a node groups the faces of one
        # colour: a spire's four sides, a dome's facets), cameras in front of that plane with some of it in frame --
        # recomputed from the node's own triangles, not the cap faces the pipeline decided with
        tri = np.asarray(p.positions, float)[np.asarray(p.indices).reshape(-1, 3)]
        tn = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        ta = np.linalg.norm(tn, axis=1)
        nonzero = ta > 1e-9
        tri, tn = tri[nonzero], tn[nonzero] / ta[nonzero, None]
        tn[tn[:, 2] < 0] *= -1.0
        label = np.full(len(tn), -1)
        planes = []
        for i in range(len(tn)):
            if label[i] >= 0:
                continue
            same = (label < 0) & (tn @ tn[i] >= 0.999)
            label[same] = len(planes)
            planes.append(tn[i])
        fewest = None
        for pk, nrm in enumerate(planes):
            pts_k = tri[label == pk].reshape(-1, 3)
            cen = pts_k.mean(axis=0)
            probe = np.vstack([cen[None], pts_k[:: max(1, len(pts_k) // 48)]])
            facing = 0
            for cam in cameras:
                v = cam.center - cen
                if float(v @ nrm) <= 0.02 * float(np.linalg.norm(v)):
                    continue
                uv, z = cam.project(probe)
                if np.any((z > 0.3) & (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)):
                    facing += 1
                    if facing >= params.roof_min_cameras:
                        break
            fewest = facing if fewest is None else min(fewest, facing)
        if fewest is not None and fewest < params.roof_min_cameras:
            out["roof_unfaced"].append({"part": p.name, "facing_cameras": fewest, "planes": len(planes)})
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
    # D12, recomputed: the drawn mask claims nothing the geometric observation (views, consistency, SfM
    # support) did not (observed-fill never counts as observed), and every fill texel lies in a hole the
    # audit finds itself (flooding the unobserved texels from the texture's edge) of at most fill_max_hole_m2
    # that cannot hold a whole drawable element (its own block search, not the pipeline's erosion)
    fill_audit = {"observed_beyond_geometric_texels": 0, "illegal_fill_texels": 0}
    extents = wall_extents(massing, params)
    for wi, fac in facades.items():
        o = orthos.get(wi)
        if o is not None and o.mask.shape == fac.mask.shape:
            fill_audit["observed_beyond_geometric_texels"] += int((fac.mask & ~o.mask).sum())
        if fac.fill is not None and fac.fill.any():
            cols = face_columns(extents[wi], massing.walls[wi].length, fac.mask.shape[1]) if wi in extents else None
            legal = _legal_fill(fac.mask, fac.px_per_m, params.fill_max_hole_m2, cols,
                                fill_shape_kernels(fac.px_per_m, params, fac.mask.shape))
            fill_audit["illegal_fill_texels"] += int((fac.fill & ~legal).sum())
    out["fill"] = fill_audit
    _audit_relief(out["relief"], massing, parts, orthos, pts, track, lines, observations, cameras,
                  d24_inputs={"structure": structure, "snap_scale": snap_scale, "quality": quality, "tracks": tracks})
    if registration_mode == "osm-footprint":
        reg = registration_check(pts, rings, params)
        out["registration"] = {"accepted": reg["accepted"], "verified_share": reg["verified_share"]}
        reg_ok = reg["accepted"]
    else:
        out["registration"] = {"accepted": True, "note": registration_mode}
        reg_ok = True
    out["structure_lines"] = recheck_structure_lines(structure_lines, lines)
    out["passed"] = (not out["roof_mismatches"] and not out["wall_plane_mismatches"] and not out["height_mismatches"]
                     and share <= 0.05 and all(s <= 0.2 for s, _ in worst) and reg_ok
                     and not any(fill_audit.values()) and not out["stub_mismatches"] and not out["roof_unfaced"]
                     and not out["relief"]["violations"] and not out["structure_lines"]["violations"]
                     and not out["d25"])
    return out


def _audit_d25_model(structure: Any) -> list[dict[str, Any]]:
    """audit_model's D25 part (crowd3d.correct.check_model): none without a structure."""
    if structure is None:
        return []
    from . import correct as C
    return C.check_model(structure)


def recheck_structure_lines(sl: dict[str, Any] | None, lines: Any) -> dict[str, Any]:
    """audit_model's roof-line part: every structure line (GLB scene extras, glTF axes) flagged photo-verified is
    verified again from the 3-D lines (verify_segment on its own segment, near and far copies) -- a flag the lines
    do not back, or no lines to check it against, is a violation. Lines not flagged claim nothing."""
    out: dict[str, Any] = {"verified": 0, "rechecked": 0, "violations": []}
    rows = (sl or {}).get("lines") or []
    for i, row in enumerate(rows):
        if not (isinstance(row, list) and len(row) >= 10 and row[8]):
            continue
        out["verified"] += 1
        a = np.array([row[0], -row[2], row[1]], float)          # glTF (x, y up, z = -north) -> ENU
        b = np.array([row[3], -row[5], row[4]], float)
        ver, _ = verify_segment(a, b, lines)
        out["rechecked"] += 1
        if ver is None:
            out["violations"].append({"line": i, "why": "photo-verified, but the 3-D lines do not verify it" if lines
                                      is not None and len(lines) else "photo-verified, and no 3-D lines to check it"})
    return out


def _audit_wall_planes(out: dict[str, Any], massing: Massing, parts: list[Part]) -> None:
    """Every wall node WITHOUT relief records lies on its wall's plane (crowd3d.relief.PLANE_TOL_M): relief-like
    geometry without its records fails, whichever wall it is on."""
    from . import relief as RL
    for p in parts:
        ex = p.extras
        if ex.get("kind") != "wall" or ex.get("relief"):
            continue
        wi = ex.get("wall")
        if wi is None or not (0 <= wi < len(massing.walls)) or not len(p.positions):
            continue
        w = massing.walls[wi]
        dd = (np.asarray(p.positions, float)[:, :2] - w.a) @ w.normal
        if float(np.abs(dd).max()) > RL.PLANE_TOL_M:
            out["violations"].append({"node": p.name, "why": f"wall geometry {float(np.abs(dd).max()):.3f} m off its "
                                                             "wall plane without relief records"})


def _audit_relief(out: dict[str, Any], massing: Massing, parts: list[Part], orthos: dict[int, Orthophoto],
                  pts: np.ndarray, track: np.ndarray | None, lines: Any = None,
                  observations: list[WallObservation] | None = None, cameras: list[PosedCamera] | None = None,
                  d24_inputs: dict[str, Any] | None = None) -> None:
    """audit_model's facade-relief part (see there); fills `out`."""
    from . import relief as RL
    v = out["violations"]
    d24_cond: dict[str, Any] | None = None

    def d24_condition() -> dict[str, Any]:
        # D24's scale condition re-derived from the inputs (once, and only when a D24 record is there)
        nonlocal d24_cond
        if d24_cond is None:
            di = d24_inputs or {}
            d24_cond = RL.d24_scale_condition(di.get("structure"), pts, di.get("quality"), lines,
                                              float(di.get("snap_scale") or 1.0))
            out["d24"] = {"allowed": bool(d24_cond.get("allowed")), "by": d24_cond.get("by"), "why": d24_cond.get("why")}
        return d24_cond
    split_cache: dict[int, Any] = {}

    def resplit(wi: int, wall: Any) -> tuple[list, list, list[int], Any, Any] | str:
        # both photo splits re-run from the 2-D observations as measure_relief runs them: the wall's points
        # re-triangulated from each half's photos and measured again -- first with every vertical a half finds as a
        # cutter, then (when not every vertical is confirmed) cut at the verticals the audit itself finds confirmed
        # (crowd3d.relief.halves_cutters on its own first pass: both splits find them, the raw points re-measure them,
        # no free-standing object at them) -- never at what a record names (verify-r2: halves_cut_at emptied to []
        # made the halves merge bays). Cached per wall. A string: why it cannot run.
        tr_ = (d24_inputs or {}).get("tracks")
        if tr_ is None or not len(tr_) or not cameras:
            return "no point tracks / cameras: the photo splits cannot be re-run"
        if wi not in split_cache:
            full = RL.measure_wall(wall, pts, trk, None, RL.DEFAULT_RELIEF, others=massing.walls)
            if full.refused or not full.maps:
                split_cache[wi] = f"the wall cannot be measured from the points: {full.refused}"
                return split_cache[wi]
            ids = full.maps["idx"]                   # the wall's own points, as measure_relief re-triangulates
            centre_ = np.mean(np.concatenate([np.stack([w.a, w.b]) for w in massing.walls]), axis=0)
            hv = RL.split_halves(cameras, centre_)
            ob_ = obs_by.get(wi)
            seen_ = [ci for ci, frac, _ in (ob_.cameras if ob_ is not None else [])
                     if frac >= RL.DEFAULT_GUARDS.photo_wall_frac]
            side, per_side = RL.side_halves(cameras, wall, seen_)
            per_half_ = [sum(1 for c in seen_ if c < len(hv) and hv[c] == h) for h in (0, 1)]
            tri = [[RL.retriangulate(ids, tr_, cameras, split == h) for h in (0, 1)] for split in (hv, side)]

            def measured(cut: Any) -> list[list]:
                res_ = []
                for k_, pair in enumerate(tri):      # (as measure_relief: an empty viewpoint half is None)
                    hs = []
                    for X, c in pair:
                        ok_ = np.isfinite(X[:, 0])
                        hs.append(RL.measure_wall(wall, X[ok_], np.maximum(c[ok_], RL.DEFAULT_RELIEF.min_track), None,
                                                  RL.DEFAULT_RELIEF, others=massing.walls, cut_boxes=cut)
                                  if ok_.any() or k_ == 0 else None)
                    res_.append(hs)
                return res_
            first = measured(None)
            U_w = U_of.get(wi)
            els0 = RL.guard_elements(full, first[0], per_half_, None, wall, sides=first[1], per_side=per_side,
                                     recheck=lambda r_: RL.audit_points(r_, wall, pts, trk, others=massing.walls,
                                                                        U=U_w, strict=True)
                                     if r_["type"] == "vertical" else {"ok": True})
            cut = RL.halves_cutters(els0, full)
            halves_, sides_ = first if cut is None else measured(cut)
            split_cache[wi] = (halves_, sides_, [int(x) for x in per_side], full, cut)
        return split_cache[wi]

    def found_again(r: dict, wi: int, wall: Any, recs: list) -> str | None:
        # G-R2 re-run for every record (verify-r2 N10x: the photo-split numbers of a line-confirmed record lived in
        # the record alone): found again in both halves of both splits as the same element (split_match: IoU, depth,
        # its sides / a band's bottom and top within split_box_m); a D24 record (its only second channel) every edge
        # (d24_halves_check), the cutters it records the ones the audit derives, and no edge at a register boundary
        # made by a band that is not accepted
        got = resplit(wi, wall)
        if isinstance(got, str):
            return got
        halves, sides, per_side, full, cut = got
        is_d24 = r.get("provenance") == RL.PROV_D24
        why = []
        if is_d24:
            d = r.get("d24") if isinstance(r.get("d24"), dict) else {}
            rc = d.get("halves_cut_at")
            try:
                same = (rc is None and cut is None) or (rc is not None and cut is not None and len(rc) == len(cut) and all(
                    abs(float(x) - float(y)) <= 2e-3 for a_, b_ in zip(sorted(rc), sorted(cut)) for x, y in zip(a_[:4], b_)))
            except (TypeError, ValueError, IndexError):
                same = False
            if not same:
                why.append(f"its halves_cut_at {RL.json_safe(rc)} is not the cutters the points confirm "
                           f"({RL.json_safe(cut)})")
            # a bottom / top at its register's boundary is no measured edge: the band making that boundary (re-derived
            # from the points) must be an accepted element itself (a record of this node), as apply_d24 requires
            try:
                zc = 0.5 * (float(r["z_bottom"]) + float(r["z_top"]))
                reg = next((g_["z_bottom"], g_["z_top"]) for g_ in full.registers if g_["z_bottom"] <= zc <= g_["z_top"])
            except (StopIteration, KeyError, TypeError, ValueError):
                reg = None
            if reg is not None and r.get("type") in ("opening", "vertical"):
                for side_, bi in RL.register_bounds(full, dict(r, register=reg), wall).items():
                    band = full.bands[bi] if 0 <= bi < len(full.bands) else None
                    ok_b = band is not None and any(
                        isinstance(x, dict) and x.get("type") == "band" and (x.get("sign") or 0) == (band.get("sign") or 0)
                        and RL.box_iou((0.0, 1.0, float(x["z_bottom"]), float(x["z_top"])),
                                       (0.0, 1.0, float(band["z_bottom"]), float(band["z_top"]))) >= 0.5 for x in recs)
                    if not ok_b:
                        why.append(f"its {side_} is its register's boundary (z {reg[0] if side_ == 'bottom' else reg[1]:.2f} "
                                   f"m), made by {'no band' if band is None else 'a band that is not an accepted element'}: "
                                   f"an edge nothing measured")
        if min(per_side) < RL.DEFAULT_GUARDS.min_photos_per_half:
            return "; ".join(why + [f"viewpoints too one-sided for the left / right split ({per_side})"])
        t = r.get("type")
        key_ = RL.DEPTH_KEY.get(t)
        try:
            el = {"type": t, "sign": r.get("sign"), **{k: float(r[k]) for k in ("u0", "u1", "z_bottom", "z_top")},
                  key_: float(r[key_])}
        except (KeyError, TypeError, ValueError):
            return "; ".join(why + ["a record without its box / depth"])
        boxes = {}
        for name, hs in (("interleaved", halves), ("viewpoint", sides)):
            ok_, info = RL.split_match(el, hs)
            if not ok_:
                why.append(f"{name}: IoU {info.get('iou_a')} / {info.get('iou_b')}, depth difference "
                           f"{info.get('depth_diff')}, sides from the record {info.get('box_diff')} m (> "
                           f"{RL.DEFAULT_GUARDS.split_box_m})")
            boxes[name] = {"a": info.get("a"), "b": info.get("b")}
        if is_d24:
            ok_e, info_e = RL.d24_halves_check(r, boxes)
            if not ok_e:
                why.append(f"every edge within {RL.DEFAULT_GUARDS.d24_edge_m} m between the halves and "
                           f"{RL.DEFAULT_GUARDS.d24_box_m} m of the record: {RL.json_safe(info_e)}")
        return "; ".join(why) or None

    # G-R1's photo count, recounted from the cameras: photos seeing the wall in each interleaved photo half
    obs_by = {o.wall: o for o in observations or []}
    half = None
    if cameras:
        centre = np.mean(np.concatenate([np.stack([w.a, w.b]) for w in massing.walls]), axis=0)
        half = RL.split_halves(cameras, centre)
    U_of: dict[int, np.ndarray] = {}
    fr_of: dict[int, dict[str, Any]] = {}
    photo_edges = RL.PhotoEdges(cameras or [])
    out.setdefault("frames", {})
    trk = np.full(len(pts), RL.DEFAULT_RELIEF.min_track) if track is None else track
    nodes = []
    for p in parts:
        rel = p.extras.get("relief")
        if not rel:
            continue
        out["nodes"] += 1
        ex = p.extras
        pi = ex.get("prism")
        stub = False
        if pi is not None and 0 <= pi < len(massing.prisms) and massing.prisms[pi].part is not None:
            hrec = massing.prisms[pi].structure.get("params", {}).get("height", [None, "", ""])
            stub = hrec[2] == PROV_DEFAULT or hrec[1] == "stub"
        why = RL.node_violations(ex, p.texture is not None) + (["relief on a stub part"] if stub else [])
        wi = ex.get("wall")
        if why or wi is None or not (0 <= wi < len(massing.walls)) or not isinstance(rel, list):
            v += [{"node": p.name, "why": w} for w in (why or ["relief on a node without its wall"])]
            continue
        nodes.append((p.name, wi, rel))
        wall = massing.walls[wi]
        rel_xy = p.positions[:, :2] - wall.a
        uu, dd, zz = rel_xy @ wall.tangent, rel_xy @ wall.normal, p.positions[:, 2]
        pu0, pu1 = float(uu[:2].min()), float(uu[:2].max())
        if (uu < pu0 - 1e-3).any() or (uu > pu1 + 1e-3).any() or (zz < wall.z0 - 1e-3).any() or (zz > wall.z1 + 1e-3).any() \
                or (np.abs(dd) > RL.RELIEF_MAX_DEPTH + 1e-3).any():
            v.append({"node": p.name, "why": "relief geometry leaves its wall piece or the depth limit"})
        # the geometry is what the records say, on this wall: no vertex off the plane outside a built record's box
        # at its offset, a face for every built record
        v += [{"node": p.name, "why": w} for w in RL.geometry_violations(uu, dd, zz, p.indices, rel)]
        o = orthos.get(wi)
        if wi not in U_of:
            U_of[wi] = RL.wall_points(wall, pts, trk, RL.DEFAULT_RELIEF, massing.walls)[0]
            # the wall's G-R1 verdict and frame re-derived from the points alone: nothing a record declares (its
            # frame, its base) is the reference of its own re-measurement
            fr_of[wi] = RL.independent_frame(wall, pts, trk, RL.DEFAULT_RELIEF, massing.walls)
            out["frames"][str(wi)] = {"refused": fr_of[wi]["refused"], "frame": fr_of[wi]["frame"]}
        ind = fr_of[wi]
        if ind["refused"]:
            v.append({"node": p.name, "why": f"its wall fails G-R1 re-measured from the points: {ind['refused']}"})
            continue
        ob = obs_by.get(wi)
        seen = [ci for ci, frac, _ in (ob.cameras if ob is not None else []) if frac >= RL.DEFAULT_GUARDS.photo_wall_frac]
        per_half = [sum(1 for c in seen if half is not None and c < len(half) and half[c] == h) for h in (0, 1)]
        if min(per_half) < RL.DEFAULT_GUARDS.min_photos_per_half:
            v.append({"node": p.name, "why": f"its wall fails G-R1 recounted from the cameras: photos per half {per_half} "
                                             f"(< {RL.DEFAULT_GUARDS.min_photos_per_half})"})
            continue
        if lines is None or not len(lines):
            v.append({"node": p.name, "why": "no 3-D lines to re-check its records' line evidence (G-R3) against"})
        for r in rel:
            if not isinstance(r, dict):
                v.append({"node": p.name, "why": "malformed relief record"})
                continue
            if r.get("built"):
                out["elements_built"] += 1
                share = RL.observed_share(r, wall, None if o is None else o.mask)
                if share < RL.DEFAULT_GUARDS.min_observed_share:
                    v.append({"node": p.name, "element": r.get("id"), "why": f"{r.get('kind')} over {share:.0%} observed texels"})
            try:
                fd = RL.frame_offset_diff(r.get("frame"), ind["frame"], float(r["u0"]), float(r["u1"]))
            except (KeyError, TypeError, ValueError):
                fd = float("inf")
            if not fd <= RL.FRAME_TOL_M:
                v.append({"node": p.name, "element": r.get("id"),
                          "why": f"its frame is {fd:.3f} m off the wall's frame re-derived from the points "
                                 f"(> {RL.FRAME_TOL_M} m)"})
            is_d24 = r.get("provenance") == RL.PROV_D24
            if is_d24:
                out["d24_records"] = out.get("d24_records", 0) + 1
                cond = d24_condition()
                if not RL.D24_ENABLED:
                    v.append({"node": p.name, "element": r.get("id"), "why": "a D24 record while D24 is switched off"})
                if not cond.get("allowed"):
                    v.append({"node": p.name, "element": r.get("id"),
                              "why": f"a D24 record on a model whose scale condition, re-derived from its points and "
                                     f"3-D lines, is not met: {cond.get('why')}"})
                out["d24_resplit"] = out.get("d24_resplit", 0) + 1
                # the image channel re-run: held-out photos (no line cluster at its own edges, from the 3-D lines)
                ob_ = obs_by.get(wi)
                seen_ = [ci for ci, frac, _ in (ob_.cameras if ob_ is not None else [])
                         if frac >= RL.DEFAULT_GUARDS.photo_wall_frac]
                try:
                    ho = RL.heldout_check(RL.record_box(r), wall, ind, lines, cameras or [], seen_, photo_edges)
                    ok_h, why_h = RL.heldout_verdict(ho)
                except (KeyError, TypeError, ValueError):
                    ok_h, why_h = False, "a D24 record without its box"
                out["d24_heldout"] = out.get("d24_heldout", 0) + (ok_h is not None)
                if ok_h is False:
                    v.append({"node": p.name, "element": r.get("id"),
                              "why": f"a D24 record whose held-out photos (re-scored from the photos) contradict its "
                                     f"sides: {why_h}"})
            # G-R2 for every record (a D24 record's only second channel): both photo splits re-run from the 2-D
            # observations, cut where the audit's own first pass says
            fa = found_again(r, wi, wall, rel)
            out["resplit"] = out.get("resplit", 0) + 1
            if fa:
                v.append({"node": p.name, "element": r.get("id"),
                          "why": f"{'a D24 record' if is_d24 else 'a record'} not found again in both halves of both "
                                 f"photo splits re-run from the photos' observations: {fa}"})
            if lines is not None and len(lines):
                ev = RL.line_evidence(r, wall, lines, ind["frame"], ref=ind.get("ref"))
                out["lines_checked"] = out.get("lines_checked", 0) + 1
                if is_d24:
                    # D24: lines at chance are enough, but a line cluster must lie at >= 1 of its edges
                    if not ev["confirmed"] >= RL.DEFAULT_GUARDS.min_line_edges:
                        v.append({"node": p.name, "element": r.get("id"),
                                  "why": f"a D24 record with no edge at a line cluster in the 3-D lines (G-R3 "
                                         f"recomputed: {ev['confirmed']} of {ev['edges']}): its lines are not at "
                                         f"chance, they are absent"})
                elif not ev["ok"]:
                    v.append({"node": p.name, "element": r.get("id"),
                              "why": f"G-R3 recomputed from the 3-D lines: {ev['confirmed']} of {ev['edges']} edges at a "
                                     f"line cluster, chance p {ev['p_chance']} (record says "
                                     f"{((r.get('channels') or {}).get('lines3d') or {}).get('p_chance')})"})
            # every record, built or not, is a claim the points must show (G-R6 accepted it on the same points), here
            # against the re-derived frame and the points beside it
            chk = RL.audit_points(r, wall, pts, track, others=massing.walls, U=U_of[wi], frame=ind["frame"])
            if chk["ok"] is None:
                out["unverifiable"] += 1
                v.append({"node": p.name, "element": r.get("id"), "why": f"cannot be re-measured from the points: {chk['why']}"})
                continue
            out["points_checked"] += 1
            if not chk["ok"]:
                v.append({"node": p.name, "element": r.get("id"),
                          "why": f"re-measured from the points: {chk['why'] or 'no frame / offset'}"})
    v += RL.copied_records(nodes)


def _rgb_keys(rgb: np.ndarray) -> np.ndarray:
    """(..., 3) colours -> (...) int keys 0xRRGGBB."""
    c = np.asarray(rgb).astype(np.int64)
    return (c[..., 0] << 16) | (c[..., 1] << 8) | c[..., 2]


def _holds_block(region: np.ndarray, blocks: list[tuple[int, int]]) -> bool:
    """The audits' own shape test (not could_hide_element()): does a bool region hold an all-True block of
    one of `blocks` (rows, columns)? Every window sum of a summed-area table."""
    s = np.pad(np.asarray(region, np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    for kh, kw in blocks:
        if s.shape[0] - 1 < kh or s.shape[1] - 1 < kw:
            continue
        win = s[kh:, kw:] - s[:-kh, kw:] - s[kh:, :-kw] + s[:-kh, :-kw]
        if (win == kh * kw).any():
            return True
    return False


def _legal_fill(observed: np.ndarray, ppm: float, max_m2: float, cols: tuple[int, int] | None = None,
                blocks: list[tuple[int, int]] | None = None) -> np.ndarray:
    """The audit's own hole finder (not enclosed_holes()): flood the unobserved texels from a frame around
    the texture and from the columns outside `cols` (under the corner bevels), 8-connected; unobserved
    texels the flood cannot reach are enclosed by observed ones, and enclosed regions of at most max_m2
    that hold no block of `blocks` (fill_shape_kernels(): room for a whole drawable element) may hold
    observed-fill."""
    h, w = observed.shape
    free = np.pad(~observed, 1, constant_values=True).astype(np.uint8)
    if cols is not None:
        free[:, :cols[0] + 1] = 1
        free[:, cols[1] + 1:] = 1
    cv2.floodFill(free, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2, flags=8)
    enclosed = free[1:-1, 1:-1] == 1
    n, cc, stats, _ = cv2.connectedComponentsWithStats(enclosed.astype(np.uint8), connectivity=8)
    legal = []
    for c in range(1, n):
        x, y, bw, bh, area = (int(v) for v in stats[c])
        if area > max_m2 * max(ppm, 1e-9) ** 2 + 1e-9:
            continue
        if blocks and _holds_block(cc[y:y + bh, x:x + bw] == c, blocks):
            continue
        legal.append(c)
    return np.isin(cc, legal)


def audit_glb(glb: bytes, params: StyleParams = DEFAULT_PARAMS) -> dict[str, Any]:
    """Audit of the published bytes themselves (what a viewer or a third party gets): the atlases decode to
    colours of the colour table only (lossless, so a theme can restyle them exactly), and per textured
    node, with the texel-provenance atlases: clay texels (code 0) show the clay pattern only, observed-fill
    texels (code 2) the node's base colour only, and every fill region is enclosed by observed texels
    (code 1) within the node, at most fill_max_hole_m2 and unable to hold a whole drawable element
    (fill_shape_kernels(); texel size and the facade texture's size, which places the working pixels, from
    the node's fill extras; without a size that fits the node, the least block of any texture at least as
    large as the node shows)."""
    gltf, binary = read_glb(glb)
    extras = gltf["scenes"][0].get("extras", {})
    table = (extras.get("style") or {}).get("colors") or color_table()
    rgb_of = {r["name"]: tuple(r["rgb"]) for r in table}
    known = _rgb_keys(np.array(list(rgb_of.values())))
    clay = _rgb_keys(np.array([rgb_of.get("clay", CLAY_RGB), rgb_of.get("clay-hatch", CLAY_HATCH_RGB)]))
    out: dict[str, Any] = {"atlases": 0, "off_table_texels": 0, "textured_nodes": 0, "clay_texels_not_clay": 0,
                           "fill_texels_not_base": 0, "fill_regions": 0, "fill_regions_not_enclosed": 0,
                           "fill_regions_over_cap": 0, "fill_regions_could_hide": 0, "fill_holes_mismatch": 0,
                           "texel_provenance": False, "stub_nodes_not_clay": [], "stub_params_not_stub": [],
                           "relief": {"nodes": 0, "elements": 0, "violations": []}}
    # 'height pending' stubs (D13) in the structure extras: a default-style parameter is always in state stub, and
    # every node of a stub part is clay (no texture, a clay material, appearance default-style, not observed)
    sparts = (extras.get("structure") or {}).get("parts") or []
    stub_ids = set()
    for r in sparts:
        prm = r.get("params") or {}
        for k, v in prm.items():
            if isinstance(v, list) and len(v) == 3 and v[2] == PROV_DEFAULT and v[1] != "stub":
                out["stub_params_not_stub"].append([r.get("id"), k, v[1]])
        h = prm.get("height")
        if r.get("state") == "stub" or (isinstance(h, list) and len(h) == 3 and (h[2] == PROV_DEFAULT or h[1] == "stub")):
            stub_ids.add(r.get("id"))
    clay_names = {"clay", "clay-hatch", "clay-roof"}
    for node in gltf["nodes"]:
        ex = node.get("extras", {})
        if ex.get("part") not in stub_ids:
            continue
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        mat = gltf["materials"][prim["material"]]
        if ("TEXCOORD_0" in prim["attributes"] or ex.get("observed")
                or ex.get("provenance", {}).get("appearance") != PROV_DEFAULT
                or (mat.get("extras") or {}).get("color") not in clay_names):
            out["stub_nodes_not_clay"].append(node.get("name"))
    _audit_glb_relief(out["relief"], gltf, binary, stub_ids)
    # the roof mesh of every part (its roof nodes together, glTF axes): structure lines must lie on its edges, and its
    # walls' tops on them. Roof PATCHES are colour decals lifted off a roof face: their outlines run mid-slope and are
    # no roof edge (verify-r3 S5), so they are not part of it
    roof_tris: dict[str, list[np.ndarray]] = {}
    for node in gltf["nodes"]:
        ex = node.get("extras", {})
        if "mesh" not in node or ex.get("kind") not in ROOF_MESH_KINDS or ex.get("part") is None:
            continue
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        pos = glb_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(float)
        idx = glb_accessor(gltf, binary, prim["indices"]).reshape(-1, 3) if "indices" in prim else \
            np.arange(len(pos)).reshape(-1, 3)
        roof_tris.setdefault(ex["part"], []).append(pos[idx])
    roof_mesh = {k: np.concatenate(x) for k, x in roof_tris.items()}
    out["frame"] = _audit_glb_frame(gltf, binary, sparts, roof_mesh)
    out["structure_lines"] = audit_structure_lines(extras.get("structure_lines"), stub_ids, roof_mesh)
    out["d25"] = _audit_glb_d25(gltf, extras.get("structure") or {}, extras)

    def image(index: int) -> np.ndarray:
        bv = gltf["bufferViews"][gltf["images"][index]["bufferView"]]
        im = Image.open(io.BytesIO(binary[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]]))
        return np.asarray(im.convert("L") if im.mode == "L" else im.convert("RGB"))

    textures = gltf.get("textures", [])
    atlases = [image(t["source"]) for t in textures]
    out["atlases"] = len(atlases)
    for a in atlases:
        out["off_table_texels"] += int((~np.isin(_rgb_keys(a), known)).sum())
    tp = extras.get("texel_provenance")
    codes = [image(i) for i in tp["images"]] if tp else []
    out["texel_provenance"] = bool(tp) and len(codes) == len(atlases)
    ring_kernel = np.ones((3, 3), np.uint8)
    for node in gltf["nodes"]:
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        if "TEXCOORD_0" not in prim["attributes"]:
            continue
        out["textured_nodes"] += 1
        if not out["texel_provenance"]:
            continue
        mat = gltf["materials"][prim["material"]]
        a = mat["pbrMetallicRoughness"]["baseColorTexture"]["index"]
        uv = glb_accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"])
        H, W = atlases[a].shape[:2]
        x0, y0 = (int(math.floor(v + 1e-3)) for v in uv.min(axis=0) * (W, H))
        x1, y1 = (int(math.ceil(v - 1e-3)) for v in uv.max(axis=0) * (W, H))
        tex, cod = atlases[a][y0:y1, x0:x1], codes[a][y0:y1, x0:x1]
        ex = node.get("extras", {})
        out["clay_texels_not_clay"] += int((~np.isin(_rgb_keys(tex[cod == 0]), clay)).sum())
        fill = cod == 2
        n, cc, stats, _ = cv2.connectedComponentsWithStats(fill.astype(np.uint8), connectivity=8)
        out["fill_regions"] += n - 1
        fx = ex.get("fill") or {}
        if n - 1 != fx.get("holes", 0):
            out["fill_holes_mismatch"] += 1
        if not fill.any():
            continue
        base = rgb_of.get(ex.get("base_color", ""))
        out["fill_texels_not_base"] += int(fill.sum()) if base is None else \
            int(np.any(tex[fill] != np.array(base, np.uint8), axis=-1).sum())
        ppm = float(fx.get("px_per_m", 0.0))
        m2_per_texel = 1.0 / ppm ** 2 if ppm > 0 else float("inf")      # no texel size: no fill can be shown to fit
        size = fx.get("texture_px")                      # the whole facade texture this node shows columns of
        shape = (int(size[1]), int(size[0])) if (isinstance(size, list) and len(size) == 2
                                                  and int(size[1]) == tex.shape[0] and int(size[0]) >= tex.shape[1]) else None
        # without a size that fits, the least block of every texture at least as large as what the node shows
        blocks = fill_shape_kernels(ppm, params, shape, at_least=tex.shape[:2]) if ppm > 0 else []
        cod_p, cc_p = np.pad(cod, 1, constant_values=255), np.pad(cc, 1)
        for c in range(1, n):
            x, y, bw, bh, area = (int(v) for v in stats[c])
            if area * m2_per_texel > params.fill_max_hole_m2 + 1e-9:
                out["fill_regions_over_cap"] += 1
            if _holds_block(cc[y:y + bh, x:x + bw] == c, blocks):
                out["fill_regions_could_hide"] += 1
            sl = (slice(y, y + bh + 2), slice(x, x + bw + 2))             # the region and its ring, in padded coordinates
            comp = (cc_p[sl] == c).astype(np.uint8)
            ring = (cv2.dilate(comp, ring_kernel) > 0) & (comp == 0)
            if not (cod_p[sl][ring] == 1).all():
                out["fill_regions_not_enclosed"] += 1
    out["passed"] = bool(out["off_table_texels"] == 0 and out["clay_texels_not_clay"] == 0
                         and out["fill_texels_not_base"] == 0 and out["fill_regions_not_enclosed"] == 0
                         and out["fill_regions_over_cap"] == 0 and out["fill_regions_could_hide"] == 0
                         and out["fill_holes_mismatch"] == 0
                         and (out["texel_provenance"] or out["textured_nodes"] == 0)
                         and not out["stub_nodes_not_clay"] and not out["stub_params_not_stub"]
                         and not out["relief"]["violations"] and not out["structure_lines"] and not out["frame"]
                         and not out["d25"])
    return out


def _osm_fix_report(structure: Any) -> dict[str, Any] | None:
    """crowd3d.correct.osm_fix_suggestions of a structure with active D25 corrections, else None."""
    if not any(str(a.get("kind", "")).startswith("d25-") and a.get("active", True)
               for a in getattr(structure, "adoptions", None) or []):
        return None
    from . import correct as C
    return C.osm_fix_suggestions(structure)


def write_osm_fix_report(summary: dict[str, Any], out_dir: Any, slug: str) -> Any:
    """Write the summary's osm_fix_suggestions beside it (<slug>-osm_fix_suggestions.json); None when there is none."""
    rep = (summary or {}).get("osm_fix_suggestions")
    if not rep:
        return None
    from pathlib import Path
    path = Path(out_dir) / f"{slug}-osm_fix_suggestions.json"
    path.write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def pose_node_matrix(pose: dict[str, Any] | None) -> list[float] | None:
    """The glTF node matrix (column-major, glTF axes) of a D25-T pose: the rigid rotation of the stack about its base
    centroid, the same for every node; None when upright."""
    if not pose:
        return None
    from . import correct as C
    R, c = C.pose_matrix(pose)
    A = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])            # ENU -> glTF (x, up, -north)
    Rg = A @ R @ A.T
    cg = A @ c
    M = np.eye(4)
    M[:3, :3] = Rg
    M[:3, 3] = cg - Rg @ cg
    return [round(float(v), 9) for v in M.T.reshape(-1)]


def _audit_glb_d25(gltf: dict[str, Any], sx: dict[str, Any], scene_ex: dict[str, Any] | None = None
                   ) -> list[dict[str, Any]]:
    """audit_glb's D25 part: every active D25 record in the structure extras passes its rule again on its own numbers
    (crowd3d.correct.check_record); its registration grade is one the GLB's own registration has (a snap-grade box
    needs the snap accepted and not ambiguous, a crowd-grade one a georegistered model: verify-r2 forged a rejected
    snap under a snap-grade record); its map value is what the part's own OSM tags give (crowd3d.structure
    .map_value_of, the tags travelling in the part's extras) and the box samples every symmetric alias of the main
    outline; every parameter whose source is photo-corrected has an active record of that part and parameter with
    that value (or a carry record's parent correction); the tilt: a pose only with one active tilt record whose pose
    it equals, on a stack that passes the shape gate on the published outlines, and every node carrying exactly that
    pose's matrix (no node transformed otherwise)."""
    from . import correct as C
    v: list[dict[str, Any]] = []
    recs = [a for a in sx.get("adoptions") or [] if str(a.get("kind", "")).startswith("d25-") and a.get("active", True)]
    by_id = {a.get("id"): a for a in recs}
    for a in recs:
        why = C.check_record(a, by_id)
        if why:
            v.append({"record": a.get("id"), "why": why})
    snap = (scene_ex or {}).get("snap") or {}
    snap_ok = bool(snap.get("accepted")) and not (snap.get("search") or {}).get("ambiguous")
    georeg = ((scene_ex or {}).get("prior") or {}).get("georegistered")
    for a in recs:
        g = C.record_grade(a, by_id)
        if g == C.GRADE_SNAP and not snap_ok:
            v.append({"record": a.get("id"), "why": "a snap-grade record in a model whose snap onto the footprint is not "
                                                    "accepted, or ambiguous"})
        elif g == C.GRADE_CROWD and georeg is not True:
            v.append({"record": a.get("id"), "why": "a crowd-grade record in a model not shown to be georegistered"})
        elif g not in (C.GRADE_SNAP, C.GRADE_CROWD):
            v.append({"record": a.get("id"), "why": f"unknown registration grade {g!r}"})
    pmap = {p.get("id"): p for p in sx.get("parts") or []}
    for a in recs:
        if a.get("kind") not in (C.KIND_H, C.KIND_ROOF, C.KIND_BLOCK):
            continue
        why = C.map_value_problem(a, pmap.get(a.get("part")), sx.get("main_height"))
        if why:
            v.append({"record": a.get("id"), "why": why})
        if a.get("kind") in (C.KIND_H, C.KIND_BLOCK) and sx.get("main_ring"):
            why = C.alias_samples_problem(a.get("registration_box") or {}, np.asarray(sx["main_ring"], float))
            if why:
                v.append({"record": a.get("id"), "why": why})
        if a.get("kind") == C.KIND_BLOCK:
            # the interior footprint count re-derived from the published rings (verify-r4 B5)
            me = pmap.get(a.get("part")) or {}
            rings = me.get("rings") or []
            if not rings:
                v.append({"record": a.get("id"), "why": "a block whose part has no published outline"})
            else:
                others = [(q["rings"][0], q["rings"][1:]) for q in sx.get("parts") or []
                          if q.get("id") != a.get("part") and q.get("rings")]
                why = C.block_footprint_problem(a, rings[0], rings[1:], others, lower=not sx.get("withheld"))
                if why:
                    v.append({"record": a.get("id"), "why": why})
    by = {(a.get("part"), a.get("param")): a for a in recs}
    parts = sx.get("parts") or []
    parents = {p.get("id"): p.get("parent") for p in parts}
    for p in parts:
        for k, val in (p.get("params") or {}).items():
            if not (isinstance(val, list) and len(val) == 3 and val[2] in C.PROVS):
                continue
            if val[1] != "photo-corrected":
                v.append({"part": p.get("id"), "param": k, "why": f"source {val[2]!r} in state {val[1]!r}"})
            a = by.get((p.get("id"), k))
            if a is None:
                # carried with a corrected parent: only by an active D25-H record of an ancestor (the published parts'
                # parent chain) that lists this part with its own change, the value its snapshot plus the changes
                # (verify-r1 F1b: any active record used to exempt every photo-corrected parameter)
                why = _audit_glb_d25_carry(recs, parents, p.get("id"), k, float(val[0]), val[2])
                if why:
                    v.append({"part": p.get("id"), "param": k, "why": why})
                continue
            if abs(float(a.get("value", math.nan)) - float(val[0])) > 0.01 or a.get("provenance") != val[2]:
                v.append({"part": p.get("id"), "param": k, "why": f"value / provenance differ from record {a.get('id')}"})
            if a.get("kind") == "d25-block" and p.get("roof") != "flat":
                v.append({"part": p.get("id"), "param": k, "why": "a photo-measured block with a roof shape"})
    tilts = [a for a in recs if a.get("kind") == "d25-tilt"]
    pose = sx.get("pose")
    if pose and (len(tilts) != 1 or tilts[0].get("id") != pose.get("adoption")
                 or any(abs(float(pose.get(k, 0)) - float((tilts[0].get("value") or {}).get(k, 1e9))) > 1e-6
                        for k in ("tilt_deg", "azimuth_deg"))):
        v.append({"pose": pose.get("adoption"), "why": "a pose without its one active tilt record, or not equal to it"})
    if tilts and not pose:
        v.append({"record": tilts[0].get("id"), "why": "an active tilt record but no pose"})
    if pose:
        mr = sx.get("main_ring")
        z0s_all = [float((r.get("built") or {}).get("z0", 0)) for r in parts if r.get("built")]
        if not mr or not z0s_all:
            v.append({"pose": pose.get("adoption"), "why": "a pose without the main outline and parts it turns about"})
        else:
            from . import structure as S_
            piv = C.tilt_pivot(S_.ccw(np.asarray(mr, float)), min(z0s_all))
            got = np.asarray(pose.get("pivot") or [math.nan] * 3, float)
            if not np.all(np.abs(got - piv) <= C.PIVOT_TOL_M):
                v.append({"pose": pose.get("adoption"), "why": f"the pivot {got.round(3).tolist()} is not the main "
                                                               f"outline's centroid at the base {piv.round(3).tolist()}"})
        rings = [r.get("rings")[0] for r in parts if r.get("rings")]
        if rings:
            big = max(rings, key=lambda r: abs(signed_area(np.asarray(r, float))))
            tops = [float((r.get("built") or {}).get("top", 0)) for r in parts]
            z0s = [float((r.get("built") or {}).get("z0", 0)) for r in parts]
            sh = C.tilt_shape_gate(np.asarray(big, float), max(tops) - min(z0s))
            if not sh["passed"]:
                v.append({"pose": pose.get("adoption"), "why": f"a tilt on a stack that fails the shape gate ({sh})"})
    want = pose_node_matrix(pose)
    for node in gltf.get("nodes") or []:
        if any(k in node for k in ("rotation", "translation", "scale")):
            v.append({"node": node.get("name"), "why": "a node transformed outside the recorded pose"})
        m = node.get("matrix")
        if want is None and m is not None:
            v.append({"node": node.get("name"), "why": "a node matrix without a recorded pose"})
        elif want is not None and (m is None or len(m) != 16 or max(abs(a - b) for a, b in zip(m, want)) > 1e-6):
            v.append({"node": node.get("name"), "why": "a node whose matrix is not the recorded pose"})
    return v


def _audit_glb_d25_carry(recs: list[dict[str, Any]], parents: dict[str, Any], pid: str, k: str, value: float,
                         source: str) -> str | None:
    """A photo-corrected parameter without its own record must be a carry: listed in the 'carried' of >= 1 active
    D25-H record of an ancestor part (crowd3d.correct._carry_problem on the GLB's parent chain), with provenance
    photo-corrected (auto), its value the earliest such record's snapshot of it plus every listed change."""
    from . import correct as C
    if k not in ("height", "min_height"):
        return "photo-corrected without its own record (only heights are carried)"
    hits = [(i, a) for i, a in enumerate(recs) if a.get("kind") == C.KIND_H
            and any(c.get("part") == pid for c in a.get("carried") or [])]
    if not hits:
        return "photo-corrected without a matching D25 record or carry"
    if source != C.PROV_FULL:
        return f"a carried parameter with provenance {source!r} (carries are photo-corrected (auto))"
    for _, a in hits:
        for c in a.get("carried") or []:
            if c.get("part") == pid:
                why = C._carry_problem(None, pid, {"parent": a.get("part"), "adoption": a.get("id"), "dz": c.get("dz")},
                                       a, parents)
                if why:
                    return why
    first = hits[0][1]
    snap = next((b for b in first.get("before") or [] if b.get("part") == pid and b.get("param") == k), None)
    if snap is None:
        return f"carried by {first.get('id')}, which kept no snapshot of it"
    want = float(snap["value"]) + sum(float(c.get("dz") or 0.0) for _, a in hits for c in a.get("carried") or []
                                      if c.get("part") == pid)
    if abs(want - value) > 0.01:
        return f"carried to {value:g} m, but its snapshot plus the listed changes is {want:g} m"
    return None


FRAME_GLB_TOL_M = 0.02      # audit_glb: node heights and wall bases against the part's built heights and outline
PATCH_GLB_TOL_M = 0.3       # ... roof patches float a few cm above their face (lifted along its normal)
ROOF_MESH_KINDS = ("roof", "roof-cut")   # a part's roof mesh (its roof faces; patches are decals above them)


def _audit_glb_frame(gltf: dict[str, Any], binary: bytes, sparts: list[dict[str, Any]],
                     roof_mesh: dict[str, np.ndarray] | None = None) -> list[dict[str, Any]]:
    """audit_glb's frame part: every node of a drawn structure part (extras.structure.parts) is what that part's
    record says was built -- wall nodes from the part's base (z0) to at most its top, with their base on the part's
    outline rings (within FRAME_GLB_TOL_M; a relief node's frame corners) and their top ON the part's roof mesh
    (roof_mesh: every wall vertex above its base -- a relief node's top frame corners -- within FRAME_GLB_TOL_M of an
    edge of the part's roof triangles: a wall ends where its roof begins, at the eave, or along a gable's rakes; a
    wall raised above its eave stands off the roof, verify-r3 P4), roof nodes between its eave and top (its
    roof nodes together reaching both), roof patches near them, its underside at its base -- and every node's
    provenance holds known values only, the outline's being the part's outline source and the height's the source
    of the parameter it was drawn from (walls: the part's height, or its roof height's when the eave depends on a
    convention; roofs: the roof height's, a flat roof's the height's; the underside: min_height's). A roof raised in
    the mesh, a wall moved along the map, or a node relabelled 'photo-measured' fails."""
    v: list[dict[str, Any]] = []
    by_id = {r.get("id"): r for r in sparts or []}
    roof_y: dict[str, list[float]] = {}
    for node in gltf["nodes"]:
        ex = node.get("extras", {})
        if "mesh" not in node:
            continue
        name, prov = node.get("name"), ex.get("provenance") or {}
        bad = [k for k in PROVENANCE_KEYS if prov.get(k) not in PROVENANCES] if prov or ex.get("kind") else []
        if bad:
            v.append({"node": name, "why": f"provenance {[prov.get(k) for k in bad]} not a known value"})
        part = by_id.get(ex.get("part"))
        if part is None:
            continue
        b, prm = part.get("built"), part.get("params") or {}
        rings = part.get("rings")
        if not (isinstance(b, dict) and all(isinstance(b.get(k), (int, float)) for k in ("z0", "eave", "top"))
                and isinstance(rings, list) and rings):
            v.append({"node": name, "why": f"part {part.get('id')} without its built heights and outline"})
            continue
        z0, eave, top = float(b["z0"]), float(b["eave"]), float(b["top"])
        src = lambda k: (prm.get(k) or [None, None, None])[2]
        flat = part.get("roof") == "flat"
        hs, rs = src("height"), src("roof_height")
        kind = str(ex.get("kind", ""))
        want = {"wall": hs if (flat or rs in _FACT_SOURCES) else rs, "roof": hs if flat else rs,
                "roof-patch": hs if flat else rs, "base": src("min_height")}.get(kind)
        if prov.get("outline") != part.get("outline_source"):
            v.append({"node": name, "why": f"outline provenance {prov.get('outline')!r}, the part's is "
                                           f"{part.get('outline_source')!r}"})
        if want is not None and prov.get("height") != want and prov.get("height") != PROV_DEFAULT:
            v.append({"node": name, "why": f"height provenance {prov.get('height')!r}, the part's {kind} is drawn "
                                           f"from {want!r}"})
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        pos = glb_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(float)
        if not len(pos):
            continue
        y0, y1 = float(pos[:, 1].min()), float(pos[:, 1].max())
        t = FRAME_GLB_TOL_M
        if kind == "wall":
            if abs(y0 - z0) > t or y1 > top + t:
                v.append({"node": name, "why": f"wall from {y0:.2f} to {y1:.2f} m, its part is built from {z0:.2f} "
                                               f"to at most {top:.2f} m"})
            base = pos[:2] if ex.get("relief") else pos[np.abs(pos[:, 1] - z0) <= 1e-3]
            xy = np.c_[base[:, 0], -base[:, 2]]
            segs = np.array([[r_[k], r_[(k + 1) % len(r_)]] for r_ in rings if len(r_) >= 2
                             for k in range(len(r_))], float).reshape(-1, 2, 2)
            if len(xy) and len(segs):
                a_, d_ = segs[:, 0], segs[:, 1] - segs[:, 0]
                tt = np.clip(np.einsum("nkj,kj->nk", xy[:, None] - a_[None], d_) / np.maximum((d_ * d_).sum(1), 1e-12), 0, 1)
                dist = np.linalg.norm(xy[:, None] - (a_[None] + tt[..., None] * d_[None]), axis=2).min(axis=1)
                if dist.max() > t:
                    v.append({"node": name, "why": f"wall base {dist.max():.2f} m off its part's outline"})
            rm = (roof_mesh or {}).get(part["id"])
            if rm is not None and len(rm):
                cand = pos[2:4] if ex.get("relief") else pos
                tops = cand[cand[:, 1] > z0 + 1e-3]
                if len(tops):
                    E = np.concatenate([rm[:, [0, 1]], rm[:, [1, 2]], rm[:, [2, 0]]])
                    dtop = _dist_to_segments(tops, E)
                    if dtop.max() > t:
                        v.append({"node": name, "why": f"wall top {dtop.max():.2f} m off its part's roof (at "
                                                       f"{tops[int(np.argmax(dtop)), 1]:.2f} m; eave {eave:.2f}, top "
                                                       f"{top:.2f} m): a wall ends where its roof begins"})
        elif kind == "roof":
            roof_y.setdefault(part["id"], []).extend([y0, y1])
            if y0 < eave - t or y1 > top + t:
                v.append({"node": name, "why": f"roof from {y0:.2f} to {y1:.2f} m, its part's roof is built from "
                                               f"{eave:.2f} to {top:.2f} m"})
        elif kind == "roof-patch":
            if y0 < eave - PATCH_GLB_TOL_M or y1 > top + PATCH_GLB_TOL_M:
                v.append({"node": name, "why": f"roof patch from {y0:.2f} to {y1:.2f} m, off its part's roof "
                                               f"({eave:.2f}-{top:.2f} m)"})
        elif kind == "base":
            if abs(y0 - z0) > t or abs(y1 - z0) > t:
                v.append({"node": name, "why": f"underside at {y0:.2f}-{y1:.2f} m, its part's base is {z0:.2f} m"})
    for pid, ys in roof_y.items():                       # (each node within eave..top above; together they reach both)
        b = by_id[pid]["built"]
        if min(ys) > float(b["eave"]) + FRAME_GLB_TOL_M or max(ys) < float(b["top"]) - FRAME_GLB_TOL_M:
            v.append({"node": f"roof of {pid}", "why": f"its roof nodes span {min(ys):.2f}-{max(ys):.2f} m, not reaching "
                                                      f"the built {float(b['eave']):.2f}-{float(b['top']):.2f} m"})
    return v


def _audit_glb_relief(out: dict[str, Any], gltf: dict[str, Any], binary: bytes, stub_ids: set) -> None:
    """audit_glb's facade-relief part: a node carrying relief records is a textured, observed, photo-appearance wall
    node of a non-stub part; every record is backed (crowd3d.relief.record_violations); the node's vertices stay on its
    frame (the first four vertices: bottom-left, bottom-right, top-right, top-left) within the depth limit; its
    geometry is what its records say (crowd3d.relief.geometry_violations, in the frame of the piece_u it declares); no
    record is a copy of another wall's (crowd3d.relief.copied_records); and every wall node WITHOUT records is planar.
    What lies INSIDE an element (a pier a merged recess deleted) is not in the bytes: the GLB holds the built geometry,
    which is the claim itself, and the photo texture, no depth; record_violations only requires each recess record to
    carry its interior re-measurement (G-R6), and audit_model, which re-measures it from the raw points, is the
    independent check. A D24 record (built on the points alone) needs the model's D24 state in the scene extras
    (relief.d24: switched on, scale condition allowed, and the scale numbers it carries -- relief.d24.scale -- ones
    that allow it: crowd3d.relief.d24_scale_inconsistency; re-deriving the scale itself needs the points, which only
    audit_model has) and the scene's count of built D24 records to match the nodes'."""
    from . import relief as RL
    v = out["violations"]
    nodes = []
    sx = ((gltf.get("scenes") or [{}])[0].get("extras") or {}).get("relief") or {}
    d24_scene = sx.get("d24") if isinstance(sx.get("d24"), dict) else {}
    d24_built = 0
    for node in gltf["nodes"]:
        if "mesh" not in node:
            continue
        ex = node.get("extras", {})
        rel = ex.get("relief")
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        if not rel:
            if ex.get("kind") == "wall":
                pos = glb_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(float)
                dev = RL.plane_violation(pos)
                if dev > RL.PLANE_TOL_M:
                    v.append({"node": node.get("name"), "why": f"wall geometry {dev:.3f} m off its plane without relief records"})
            continue
        out["nodes"] += 1
        v += [{"node": node.get("name"), "why": w}
              for w in RL.node_violations(ex, "TEXCOORD_0" in prim["attributes"], stub_ids)]
        for r in rel if isinstance(rel, list) else []:
            out["elements"] += 1
            v += [{"node": node.get("name"), "element": r.get("id") if isinstance(r, dict) else None, "why": w}
                  for w in (RL.record_violations(r) if isinstance(r, dict) else ["malformed record"])]
            if isinstance(r, dict) and r.get("provenance") == RL.PROV_D24:
                d24_built += bool(r.get("built"))
                if not (d24_scene.get("enabled") is True and d24_scene.get("allowed") is True):
                    v.append({"node": node.get("name"), "element": r.get("id"),
                              "why": f"a D24 record in a model whose scene extras do not record D24 on and allowed "
                                     f"({d24_scene or 'no relief.d24'})"})
                elif RL.d24_scale_inconsistency(d24_scene.get("scale")):
                    v.append({"node": node.get("name"), "element": r.get("id"),
                              "why": f"a D24 record in a model whose scene extras say D24 is allowed over scale numbers "
                                     f"that do not allow it: {RL.d24_scale_inconsistency(d24_scene.get('scale'))}"})
        if not isinstance(rel, list):
            v.append({"node": node.get("name"), "why": "malformed relief records"})
            continue
        nodes.append((node.get("name"), ex.get("wall"), rel))
        pos = glb_accessor(gltf, binary, prim["attributes"]["POSITION"]).astype(float)
        if len(pos) < 4:
            v.append({"node": node.get("name"), "why": "relief node without its frame corners"})
            continue
        t = pos[1] - pos[0]
        L = float(np.linalg.norm(t))
        up = pos[3] - pos[0]
        H = float(np.linalg.norm(up))
        if L < 1e-6 or H < 1e-6:
            v.append({"node": node.get("name"), "why": "relief node with a degenerate frame"})
            continue
        t, up = t / L, up / H
        n = np.cross(t, up)
        rel_p = pos - pos[0]
        uu, hh, dd = rel_p @ t, rel_p @ up, rel_p @ n
        if (uu < -1e-3).any() or (uu > L + 1e-3).any() or (hh < -1e-3).any() or (hh > H + 1e-3).any() \
                or (np.abs(dd) > RL.RELIEF_MAX_DEPTH + 1e-3).any():
            v.append({"node": node.get("name"), "why": "relief geometry leaves its wall frame or the depth limit"})
        pu = ex.get("piece_u")
        if not (isinstance(pu, list) and len(pu) == 2 and all(isinstance(x, (int, float)) for x in pu)
                and abs((pu[1] - pu[0]) - L) <= 2e-3):
            v.append({"node": node.get("name"), "why": "relief node without a piece_u that matches its frame"})
            continue
        idx = glb_accessor(gltf, binary, prim["indices"]).reshape(-1, 3) if "indices" in prim else np.zeros((0, 3), int)
        v += [{"node": node.get("name"), "why": w}
              for w in RL.geometry_violations(uu + float(pu[0]), dd, hh + float(pos[0][1]), idx, rel)]
    v += RL.copied_records(nodes)
    if (d24_built or sx.get("built_d24")) and sx.get("built_d24") != d24_built:
        v.append({"node": None, "why": f"the scene extras count {sx.get('built_d24')} built D24 records, the nodes "
                                       f"carry {d24_built}"})


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


def frame_blocked_walls(massing: Massing, ring: np.ndarray, failing: list[dict[str, Any]], tol_m: float = 1.5,
                        max_deg: float = 20.0) -> dict[int, str]:
    """Walls lying on a main-outline edge the frame gate (G11) failed: both ends within tol_m of the edge and the wall
    within max_deg of its direction. These stay clay (docs/STRUCTURE_FIRST.md §2.2: 'the walls over the limit are not
    textured')."""
    out: dict[int, str] = {}
    if not failing:
        return out
    cos_max = math.cos(math.radians(max_deg))
    n = len(ring)
    for w in massing.walls:
        d = w.b - w.a
        L = float(np.linalg.norm(d))
        if L < 1e-6:
            continue
        for f in failing:
            e = int(f["edge"])
            p, q = ring[e], ring[(e + 1) % n]
            t = q - p
            Le = float(np.linalg.norm(t))
            if Le < 1e-6 or abs(float(d @ t)) / (L * Le) < cos_max:
                continue
            ends = np.stack([w.a, w.b])
            s_ = np.clip((ends - p) @ t / (Le * Le), 0.0, 1.0)
            dist = np.linalg.norm(ends - (p + s_[:, None] * t), axis=1)
            if (dist <= tol_m).all():
                out[w.index] = (f"frame gate (G11): on outline edge {e}, median wall-point residual {f['median_m']} m > "
                                f"1.5 m (frame ticket open): not textured")
                break
    return out


def structure_gates(scene: SceneInput, points: np.ndarray | None, massing: Massing) -> dict[str, Any]:
    """The structure gates that need the photos' points, run before observation (docs/STRUCTURE_FIRST.md §2.8): the
    per-edge residual gate G11 (crowd3d.judge.frame_gate) on the registered points -- failing, it opens a frame ticket
    (nothing may be verified or fitted while it is open) and the walls on the failing edges stay clay -- and the
    sanity-bound tickets for the OSM values (Wikidata's total height as an upper bound, also put beside the map's top
    in an open frame ticket's scale hypothesis). Without registered points the gate does not run."""
    from . import judge as J
    from . import structure as S
    model = scene.structure
    wikidata = getattr(scene.facts, "wikidata_height_m", None) if scene.facts else None
    S.open_value_tickets(model, wikidata)
    for t in model.tickets:                    # the curator's scale question: Wikidata's height beside the map's
        if t.get("kind") == "frame" and isinstance(t.get("scale_hypothesis"), dict):
            J.add_wikidata_to_hypothesis(t["scale_hypothesis"], wikidata)
    if points is None or not len(points):
        frame: dict[str, Any] = {"passed": True, "note": "no registered points: the frame gate does not apply "
                                                         "(no wall shows photos)"}
    else:
        frame = {k: v for k, v in J.frame_gate(points, model.main_rings, scene.point_quality).items() if k != "edges"}
    blocked: dict[int, str] = {}
    if not frame.get("passed", True):
        if not any(t.get("kind") == "frame" and t.get("open", True) for t in model.tickets):
            model.tickets.append({"id": f"T-{model.landmark}-frame", "kind": "frame", "open": True,
                                  "failing_edges": frame.get("failing", []), "opened_by": "publish frame gate (G11)"})
        blocked = frame_blocked_walls(massing, S.ccw(model.main_rings[0]), frame.get("failing", []))
    return {"frame": frame, "frame_blocked_walls": blocked}


def finish_structure_gates(model: Any, gates: dict[str, Any]) -> dict[str, Any]:
    """G1-G3, G6, G10 (crowd3d.structure.check_structure), G2 / G4 / G9 on the parameters' states and evidence
    (crowd3d.judge.check_decisions: also a stub always a stub, nothing verified under an open frame ticket, no slender
    part verified) and G11 (frame gate: failing, a frame ticket is open and none of its walls shows photos). All must
    pass for the version to be published (design §2.8)."""
    from . import judge as J
    from . import structure as S
    g = S.check_structure(model)
    dec = J.check_decisions(model)
    fr = gates["frame"]
    frame_ok = bool(fr.get("passed", True) or (
        any(t.get("kind") == "frame" and t.get("open", True) for t in model.tickets) and not fr.get("blocked_walls_observed")))
    return {"gates": g, "decisions": dec, "frame": fr, "frame_ok": frame_ok,
            "frame_blocked_walls": sorted(gates["frame_blocked_walls"]),
            "tickets_open": sum(1 for t in model.tickets if t.get("open", True)),
            "passed": bool(g["passed"] and dec["passed"] and frame_ok)}


def measure_facade_relief(scene: SceneInput, massing: Massing, observations: list[WallObservation],
                          facades: dict[int, StylizedFacade], points: np.ndarray, quality: np.ndarray | None,
                          cameras: list[PosedCamera], snap: Any, use_parts: bool) -> Any:
    """crowd3d.relief on the photographed walls (points and cameras after the snap). The 3-D lines are the scene's
    (the judge's, same frame) or, when a wall has enough points and the massing is the structure's parts, computed
    here (crowd3d.lines3d)."""
    from . import relief as RL
    lines: Any = scene.lines3d
    if lines is None and use_parts:
        def lines() -> Any:
            from . import lines3d as L3
            scale = float(getattr(snap, "scale", 1.0)) if snap is not None else 1.0
            return L3.lines_channel(scene.structure, cameras, points, quality, snap_scale=scale).lines
    res = RL.measure_relief(massing, observations, facades, points, None if quality is None else quality[:, 0],
                            scene.point_tracks, cameras, lines,
                            stub_parts=stub_part_ids(scene.structure) if use_parts else set())
    if res is not None and res.walls:
        # D24's scale condition on THIS model's data (re-measured similarities, the frame adoption's eave check): with
        # D24 on, elements whose line evidence is at chance may be built only where it is met
        scale = float(getattr(snap, "scale", 1.0)) if snap is not None else 1.0
        d24 = RL.d24_scale_condition(scene.structure if use_parts else None, points, quality, res.lines, scale)
        from . import judge as J
        res.stats["d24"] = {"enabled": RL.D24_ENABLED, **J._plain(d24)}
        for els_ in res.elements.values():                 # the scale caveat every record carries (element_record)
            for e_ in els_:
                e_["scale_condition"] = {"met": bool(d24.get("allowed")), "by": d24.get("by")}
        # D24 (decided 2026-10-24): where the switch is on and the scale condition met, the elements whose only
        # failing guard is G-R3 at chance are accepted on the points alone, each with its D24 record -- unless the
        # photos that gave them no 3-D line contradict their sides (RL.heldout_check: the image channel)
        edges = RL.PhotoEdges(cameras)
        inds: dict[int, Any] = {}
        trk = None if quality is None else quality[:, 0]

        def heldout(e: dict, wi: int) -> dict:
            wall = massing.walls[wi]
            if wi not in inds:
                inds[wi] = RL.independent_frame(wall, points, trk, RL.DEFAULT_RELIEF, massing.walls)
            ob = observations[wi] if observations is not None and wi < len(observations) else None
            seen = [ci for ci, frac, _ in (ob.cameras if ob is not None else []) if frac >= RL.DEFAULT_GUARDS.photo_wall_frac]
            return RL.heldout_check(RL.record_box(e), wall, inds[wi], res.lines, cameras, seen, edges)
        RL.apply_d24(res, heldout=heldout)
    return res


def relief_summary(relief: Any, parts: list[Part], massing: Massing | None = None) -> dict[str, Any]:
    """The summary's "relief" section: the measurement's report plus what the GLB carries (records, built), and the
    possible decisions it raises (a measured rhythm and the unmeasured walls of its part: never applied)."""
    from . import relief as RL
    recs = [r for p in parts for r in (p.extras.get("relief") or [])]
    out: dict[str, Any] = {"provenance": "photo-measured", "records": len(recs), "built": sum(bool(r.get("built")) for r in recs),
                           "by_kind": dict(sorted(Counter(r["kind"] for r in recs if r.get("built")).items())),
                           # D24: of them, built on the points alone (lines at chance): provenance PROV_D24
                           "built_d24": sum(bool(r.get("built")) and r.get("provenance") == RL.PROV_D24 for r in recs),
                           "by_provenance": dict(sorted(Counter(str(r.get("provenance")) for r in recs if r.get("built")).items()))}
    if relief is None:
        return {**out, "walls_measured": 0, "detected": 0, "accepted": 0, "note": "not measured"}
    meta = relief.to_meta()
    d24 = (meta["stats"] or {}).get("d24") or {}
    return {**out, "walls_measured": meta["stats"].get("walls_measured", 0), "detected": meta["detected"],
            "accepted": meta["accepted"], "stats": meta["stats"], "walls": meta["walls"],
            "d24": {"enabled": RL.D24_ENABLED, "allowed": bool(d24.get("allowed")), "by": d24.get("by"),
                    "why": d24.get("why") or "not evaluated", "applied": int(d24.get("applied") or 0),
                    "built": int(d24.get("built") or 0), "heldout_refused": int(d24.get("heldout_refused") or 0),
                    # the outline parts of the D24 records that held-out photos put above chance, of those scored
                    # (sides are required where >= 3 photos score them; sills / heads are recorded, not confirmed)
                    "heldout_parts": {part: [sum(part in ((r.get("d24") or {}).get("heldout") or {}).get("above_chance", [])
                                                 for r in recs if r.get("provenance") == RL.PROV_D24),
                                             sum(int(((((r.get("d24") or {}).get("heldout") or {}).get(part) or {})
                                                      .get("photos") or 0) >= RL.HELDOUT_MIN_PHOTOS)
                                                 for r in recs if r.get("provenance") == RL.PROV_D24)]
                                      for part in ("sides", "sill", "head")}},
            "possible_decisions": (RL.rhythm_decisions(relief, massing) if massing is not None else [])
            + RL.line_chance_decisions(relief)}


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
    elif scene.snap_prior is not None and scene.snap_prior.georegistered is False:
        # STYLIZED_MODEL 3.2 item 4: a prior-only model (georeg_level != full) contributes no facade; its yaw
        # comes from compass readings and its scale is a guess, so neither the snap nor the gate is tried
        registration = {"accepted": False, "mode": "osm-footprint",
                        "reason": "the model is not georegistered (prior only: compass yaw and an approximate scale,"
                                  " no GPS fit)"}
    else:
        if scene.snap_override is not None:
            snap = scene.snap_override
        elif scene.snap:
            snap = snap_to_footprint(points, outer, params, holes=holes, prior=scene.snap_prior)
        if snap is not None and snap.accepted:
            points = apply_snap_points(snap, points)
            cameras = [c.snapped(snap) for c in cameras]
        registration = registration_check(points, [outer] + holes, params,
                                          cameras_xy=np.array([c.center[:2] for c in cameras]) if cameras else None)
        registration["mode"] = "osm-footprint"
        if snap is not None and not snap.accepted:
            registration["accepted"] = False
            registration["reason"] = f"snap onto the footprint rejected: {snap.reason}"
    photo_ok = bool(registration["accepted"]) and points is not None
    rings = [outer] + holes
    lap("snap")
    use_parts = scene.structure is not None and params.massing_source == "osm-parts" and bool(scene.structure.built)
    if use_parts and not photo_ok:
        # the judge measured the structure with these same photos (its D15 adoptions and verifications arrive inside
        # scene.structure); unregistered, nothing they said may be published: withdrawn, on a copy (the caller's
        # structure is left as it was)
        from . import correct as C
        from . import judge as J
        scene = replace(scene, structure=copy.deepcopy(scene.structure))
        # D25: a structural correction whose evidence-grade box was sized for the registration this model has
        # survives a refusal of the colour-grade gate (its tolerance scales with the size of the correction)
        grade = C.publish_grade(snap, scene.snap_prior, points)
        J.withdraw_photo_judgements(scene.structure, f"registration not verified: "
                                                     f"{registration.get('reason') or 'unknown'}",
                                    keep=C.records_to_keep(scene.structure, grade))
    if use_parts and photo_ok and getattr(scene.structure, "pose", None):
        # D25-T: the solids stay upright; the photos' evidence is rotated back to upright so that every face is
        # observed and checked against an upright solid (the pose is applied to the GLB's nodes)
        from . import correct as C
        points = C.apply_pose(points, scene.structure.pose, inverse=True)
        cameras = [C.pose_camera(c, scene.structure.pose, inverse=True) for c in cameras]
    if use_parts:
        # structure first: the OSM parts are the massing (no rounded corners: a part's walls meet its roof
        # primitive edge for edge, so the solids stay closed)
        params = replace(params, bevel_m=0.0, occluder_tube_m=0.3)
        massing = build_massing_from_structure(scene.structure, points if photo_ok else None, params)
    else:
        massing = build_massing(outer, points if photo_ok else None, scene.facts, params, holes=holes,
                                outline_source=outline_source)
    lap("massing")
    # G11 runs on registered points only: unregistered, no wall shows photos and the gate says nothing
    gates = structure_gates(scene, points if photo_ok else None, massing) if use_parts else None
    if photo_ok:
        observations = observe_walls(massing, cameras, points, params)
    else:
        why = f"registration not verified: {registration.get('reason') or 'unknown'}"
        observations = [WallObservation(w.index, [], 0, 0.0, 0.0, np.zeros((2, 2), bool), None, False, why)
                        for w in massing.walls]
    if gates:
        for wi, why in gates["frame_blocked_walls"].items():          # G11: walls on an edge over the limit stay clay
            if observations[wi].observed:
                observations[wi].observed, observations[wi].reason = False, why
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
    norm = scene_color_norm(list(live.values()), params)
    layers = {wi: facade_layers(o, norm, params) for wi, o in live.items()}
    plan = plan_colors(list(layers.values()), params)
    base_z = float(massing.report.get("base_height", {}).get("m", massing.prisms[0].z1)) if use_parts else massing.prisms[0].z1
    size_scale = max(1.0, base_z / params.default_height_m)
    extents = wall_extents(massing, params)
    for wi, ortho in live.items():
        length = massing.walls[wi].length
        fac = stylize_facade(ortho, params, ground_row=massing.walls[wi].z0 <= 0.5, norm=norm, plan=plan,
                             layers=layers[wi], size_scale=size_scale,
                             extent=(extents[wi][0] / length, extents[wi][1] / length))
        frac = float(fac.mask.mean())
        if fac.report.get("empty") or frac < params.min_observed_fraction:
            observations[wi].observed = False
            observations[wi].reason = f"observation mask covers {frac:.0%} without sky / vegetation"
            continue
        facades[wi] = fac
    lap("stylize")
    color_valid, color_report = None, {}
    if photo_ok:
        colors, color_valid, color_report = guard_point_colors(points, colors, cameras, params, sky_masks=scene.sky_masks)
    roof = roof_appearance(massing, points if photo_ok else None, colors, norm=norm, params=params,
                           color_valid=color_valid, cameras=cameras)
    photo_refs = {}
    for wi in facades:
        refs = []
        for ci, share in orthos[wi].contributions:
            if share < 0.01:
                continue
            cam = cameras[ci]
            refs.append({"photo": cam.photo_id or cam.name, "license": cam.license, "share": share})
        photo_refs[wi] = refs
    lap("colours_roofs")
    quality = (np.asarray(scene.point_quality, float) if photo_ok and scene.point_quality is not None
               and len(scene.point_quality) == len(points) else None)
    track = None if quality is None else quality[:, 0]
    relief = None
    if params.relief_detail and photo_ok and facades:
        relief = measure_facade_relief(scene, massing, observations, facades, points, quality, cameras, snap, use_parts)
    lap("relief")
    parts = build_parts(massing, facades, roof, params, observations, photo_refs,
                        relief=relief.elements if relief is not None else None)
    honesty = check_honesty(parts, facades, params, stub_parts=stub_part_ids(scene.structure) if use_parts else None)
    # the 3-D lines the relief and the roof lines used: an input of the audit, like the points
    lines_used = relief.lines if relief is not None and relief.lines is not None else scene.lines3d
    if lines_used is not None and hasattr(lines_used, "lines") and not hasattr(lines_used, "A"):
        lines_used = lines_used.lines
    if not photo_ok:
        lines_used = None
    roof_lines: list[dict[str, Any]] = structure_lines(massing, lines_used) if use_parts else []
    sl_extras = structure_lines_extras(roof_lines) if roof_lines else None
    audit = audit_model(massing, parts, facades, orthos, observations, points if photo_ok else None, rings, cameras,
                        params, eligible=eligible, registration_mode=registration.get("mode", ""), track=track,
                        lines=lines_used, structure_lines=sl_extras,
                        structure=scene.structure if use_parts else None,
                        snap_scale=float(getattr(snap, "scale", 1.0)) if snap is not None else 1.0, quality=quality,
                        tracks=scene.point_tracks if photo_ok else None)
    fills = {wi: fac.report.get("fill") or {} for wi, fac in facades.items()}
    observed_fill = {"max_hole_m2": params.fill_max_hole_m2,
                     "walls": sum(1 for f in fills.values() if f.get("holes")),
                     "holes": sum(f.get("holes", 0) for f in fills.values()),
                     "area_m2": round(sum(f.get("area_m2", 0.0) for f in fills.values()), 3),
                     "largest_hole_m2": max((f.get("largest_m2", 0.0) for f in fills.values()), default=0.0),
                     "holes_over_cap": sum(f.get("holes_over_cap", 0) for f in fills.values()),
                     "over_cap_m2": round(sum(f.get("over_cap_m2", 0.0) for f in fills.values()), 2),
                     "largest_over_cap_m2": max((f.get("largest_over_cap_m2", 0.0) for f in fills.values()), default=0.0),
                     "holes_could_hide": sum(f.get("holes_could_hide", 0) for f in fills.values()),
                     "could_hide_m2": round(sum(f.get("could_hide_m2", 0.0) for f in fills.values()), 2),
                     "element_min_m": [max((f["element_min_m"][i] for f in fills.values() if "element_min_m" in f),
                                           default=round(smallest_element_m(params.px_per_m, params)[i], 3))
                                       for i in (0, 1)],
                     "line_min_m": params.line_min_len_m}
    reg_meta = {k: v for k, v in registration.items() if k != "edges"}
    extras = {"generator": "crowd3d.stylize", "frame": "ENU (glTF y-up: x=east, y=up, -z=north)",
              "landmark": scene.name, "attribution": list(scene.attribution),
              "provenance_values": list(PROVENANCES), "locked_label": "待解锁",
              "observed_fill": {k: observed_fill[k] for k in ("max_hole_m2", "element_min_m", "line_min_m", "holes", "area_m2")},
              "registration": reg_meta, "snap": snap.to_meta() if snap else None,
              # what the georegistration says about itself (audit_glb: a crowd-grade D25 record needs it georegistered)
              "prior": scene.snap_prior.to_meta() if scene.snap_prior is not None else None}
    relief_meta = relief_summary(relief, parts, massing)
    if sl_extras:
        extras["structure_lines"] = sl_extras
    if relief_meta.get("walls_measured"):
        extras["relief"] = {k: relief_meta[k] for k in ("walls_measured", "detected", "accepted", "built", "built_d24",
                                                        "provenance")}
        # the model's D24 state (switch, scale condition): a D24 record in the GLB needs it allowed (audit_glb)
        extras["relief"]["d24"] = {k: relief_meta["d24"][k] for k in ("enabled", "allowed", "by")}
        # ... and the numbers it was decided on (audit_glb checks that they allow it; re-deriving them needs the
        # points: audit_model)
        from . import relief as RL
        extras["relief"]["d24"]["scale"] = RL.d24_scale_numbers(((relief.stats or {}).get("d24") if relief is not None
                                                                 else None))
    if use_parts:
        extras["structure"] = scene.structure.compact_extras()
        att = extras["structure"].get("attribution")
        if att and att not in extras["attribution"]:
            extras["attribution"].append(att)
    structure_qa = None
    if gates:
        # G11 done: no wall over the limit may show photos in what is published
        leaked = sorted({p.extras.get("wall") for p in parts if p.extras.get("kind") == "wall" and p.extras.get("observed")
                         and p.extras.get("wall") in gates["frame_blocked_walls"]})
        gates["frame"]["blocked_walls_observed"] = leaked
        structure_qa = finish_structure_gates(scene.structure, gates)
        extras["structure"] = scene.structure.compact_extras()        # with the tickets the gates opened
    glb = write_glb(parts, extras)
    structure_metrics = None
    if use_parts:
        # how much structure the model has and where each number / element comes from (OSM, convention, stub, photos)
        from . import structmetrics as SM
        structure_metrics = SM.structure_metrics(scene.structure.to_registry(), glb)
        glb = with_scene_extras(glb, {"structure_metrics": SM.compact(structure_metrics)})
    glb_audit = audit_glb(glb, params)
    struct_unbacked = None if structure_metrics is None else int(structure_metrics["honesty"]["unbacked"])
    qa = {**honesty, "audit": audit, "glb": glb_audit, "structure_unbacked": struct_unbacked,
          "passed": bool(honesty["passed"] and audit["passed"] and glb_audit["passed"]
                         and (structure_qa is None or structure_qa["passed"]) and not struct_unbacked)}
    if structure_qa is not None:
        qa["structure"] = structure_qa
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
        "massing_source": "osm-parts" if use_parts else "height-field",
        "relief": relief_meta,
        "structure_lines": {"n": len(roof_lines), "by_type": dict(Counter(r["type"] for r in roof_lines)),
                            "convention": sum(r["convention"] for r in roof_lines),
                            "photo_verified": sum("photo_verified" in r for r in roof_lines), "lines": roof_lines},
        "structure_metrics": structure_metrics,
        "point_colors": color_report,
        "structure": scene.structure.to_registry() if use_parts else None,
        # D25: the suggested OSM fixes for human mappers (a report only; written beside the summary by the entry points)
        "osm_fix_suggestions": _osm_fix_report(scene.structure) if use_parts else None,
        "color_norm": norm.to_meta(),
        "color_plan": plan.to_meta(),
        "walls": [{"index": w.index, "prism": w.prism, "ring": w.ring, "length_m": round(w.length, 2),
                   "z0": round(w.z0, 2), "z1": round(w.z1, 2), "outline": w.outline_source, "height": w.height_provenance,
                   **observations[w.index].to_meta(),
                   "geometric_mask_fraction": round(orthos[w.index].observed_fraction, 3) if w.index in orthos else 0.0,
                   "ortho": orthos[w.index].stats if w.index in orthos else None,
                   "mask_fraction": round(float(facades[w.index].mask.mean()), 3) if w.index in facades else 0.0,
                   "fill": fills.get(w.index),
                   "facade": facades[w.index].report if w.index in facades else None}
                  for w in massing.walls],
        "observed_walls": len(facades),
        "num_walls": len(massing.walls),
        "observed_wall_area_fraction": round(obs_strip / max(wall_area, 1e-9), 3),
        "observed_pixel_area_fraction": round(obs_px / max(wall_area, 1e-9), 3),    # observed-fill excluded
        "observed_fill": observed_fill,
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
        "atlases": len(read_glb(glb)[0].get("textures", [])),
        "glb_bytes": len(glb),
        "qa": qa,
        "timings_s": timings,
        "total_s": round(sum(timings.values()), 3),
    }
    return StylizedModel(glb, summary, massing, observations, orthos, facades, parts, roof)
