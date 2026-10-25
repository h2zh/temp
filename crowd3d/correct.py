"""D25: strong photo evidence corrects the OSM frame (docs/STRUCTURE_FIRST.md §2.4.8; product owner, 2026-10-25).

D15 (crowd3d.judge) adopts a value only when every channel agrees inside the skyline's +-30 % search. D25 judges what
D15 left 'disputed' or 'height pending' again, with three relaxations and one new lock:

- D25-H (part heights, roof heights): the skyline search widens in stages (x1.6 / 2.0 / 2.5 up, x0.4 / 0.1 down; a
  pitched roof by an eave x roof search), the largest agreeing set of >= 2 channels decides when no other channel
  contradicts it (SfM points below the value on an upward change are no contradiction: an unphotographed roof has no
  points), a channel's interval may be max(3 m, 25 % of its change), and the scale may be established by opposite
  walls of the outline. The lock: an evidence-grade registration box (every component at +-3 sigma, corners and
  interior points, scale and ground as additive terms) whose LOWER BOUND must still clear the map value by TOL; the
  photo value V is adopted only when that bound lies above the midpoint between map and photo value (then the model
  is never farther from the truth than the map, whenever the truth lies in the box), else only the bound.
- D25-T (tilt of a round tall stack): >= 2 channels against two independent gravity references, the yaw term of the
  registration, one rigid rotation of the whole stack (StructureModel.pose).
- D25-B (photo-measured flat blocks on 'height pending' stubs and gaps): the 2nd-highest of three lower-bound readings,
  each at its minimum over the box, not above the sky-carving upper bound; a flat top, never a roof shape.
- D25-F (the outline) stays D15 ⑥.

This module holds the rule as functions of NUMBERS (rule_h_pre / rule_h_box, rule_b, rule_t): the measurement engine
(crowd3d.correct_measure, crowd3d.tilt) produces the numbers, the same functions decide, and the publish gate
(check_record, check_model) re-runs them on the numbers each record carries -- it never trusts the code that wrote
the record. apply_*() write a correction into the structure with its provenance, the map's value, all evidence and
the registration box; the record joins the reversible list (StructureModel.adoptions, kinds d25-height, d25-roof,
d25-tilt, d25-block) and crowd3d.judge.revert_adoption undoes it (and what depends on it). OpenStreetMap is never
edited: osm_fix_suggestions() only writes a report for human mappers.
"""
from __future__ import annotations

import copy
import itertools
import math
from typing import Any

import numpy as np

from . import stylize as st
from . import structure as S

RULE = "D25 (2026-10-25)"
PROV_FULL = st.PROV_CORRECTED_AUTO           # the full photo value V (or a measured tilt)
PROV_LB = st.PROV_CORRECTED_LB               # only the guaranteed bound; every D25-B block
PROVS = (PROV_FULL, PROV_LB)
KIND_H, KIND_ROOF, KIND_TILT, KIND_BLOCK = "d25-height", "d25-roof", "d25-tilt", "d25-block"
KINDS = (KIND_H, KIND_ROOF, KIND_TILT, KIND_BLOCK)
LABEL = "照片纠正地图"

# ---------------------------------------------------------------- thresholds (§2.4.8; chosen, not tuned on data)
TOL_M, TOL_REL = 2.0, 0.10                   # off the map by more than max(2 m, 10 % of the map value)
AGREE_M, AGREE_REL = 1.0, 0.05               # two channels agree within max(1 m, 5 %) (D15's)
PREC_M, PREC_REL = 3.0, 0.25                 # a channel's 90 % interval <= max(3 m, 25 % of its change)
SKY_MASKS_M, SKY_MASKS_REL = 1.5, 0.05       # the two sky masks' tops within max(1.5 m, 5 %)
SKY_VIS_MAP, SKY_VIS_AT, SKY_ABOVE = 3, 10, 5    # model skyline photos at the map value / at the value; sky above
SFM_FIT_INLIERS, SFM_FIT_CELLS, SFM_FIT_QCELLS = 0.4, 30, 3   # roof-envelope fit: inlier share, cells, cells/quadrant
SFM_QUADS, SFM_QUAD_POINTS = 3, 20           # all-round top: >= 3 quadrants with >= 20 points of track >= 3
LINES_PHOTOS, LINES_LINES = 3, 2
SLENDER_HW, SLENDER_W = 2.0, 10.0            # own height / narrowest width >= 2 and width <= 10 m: never moved
EAVE_AGREE_M, EAVE_AGREE_REL = 1.5, 0.075
ROOF_MIN_M = 0.5                             # a roof (or a wall band) under this is no roof / no wall
SCALE_K_TOL = 0.03
WALL_ANGLE_TOL, WALL_MIN_SEP_M, WALL_MIN_POINTS = 25.0, 20.0, 200
K_BOX = 3.0                                  # the box: every component at +-3 sigma
BOX_CORNERS, BOX_INTERIOR = 12, 6
SIGMA_SNAP_XY, SIGMA_SNAP_YAW = 1.0, 0.5     # snap accepted (unambiguous, G11 passed)
SIGMA_CROWD_XY, SIGMA_CROWD_GPS, SIGMA_CROWD_YAW, SIGMA_CROWD_YAW_F = 3.0, 0.5, 3.0, 1.5
SIGMA_Z_GROUND, SIGMA_Z_NONE = 0.5, 1.5
SIGMA_G_FLOOR, SIGMA_G_BOOT, SIGMA_G_TWO = 1.0, 4.0, 0.5
SCALE_NO_ANCHOR = (0.10, 0.35)               # crowd, no anchor: >= 20 cameras / fewer (nothing passes with these)
LOCAL_GROUND = {"r_in_m": 2.0, "r_out_m": 15.0, "min_points": 30, "min_sep_deg": 90.0, "z_tol_m": 0.5, "sectors": 8}
TILT_CIRC, TILT_HD = 0.97, 1.5
TILT_POINTS_BANDS, TILT_POINTS_SIGMA = 20, 0.3
TILT_SIL_PHOTOS, TILT_SIL_CONTRAST, TILT_SIL_INLIERS, TILT_SIL_SIGMA = 15, 0.2, 0.8, 0.5
TILT_LINES_N, TILT_LINES_PHOTOS, TILT_LINES_SIGMA = 100, 20, 0.3
TILT_VP_INLIERS, TILT_VP_PHOTOS, TILT_VP_SIGMA = 500, 15, 0.3
TILT_ROLL_MAX = 5.0
TILT_MIN, TILT_K, TILT_REL, TILT_SPREAD_FLOOR = 2.0, 3.0, 0.3, 0.5
BLOCK_MIN_READINGS = 2
BLOCK_STUB_M = 3.0                           # the 'height pending' stub (D13)
# D25-H and D25-B on crowd-grade registrations: OFF until the evidence-reg adversarial set passes on the corner box
CROWD_HB_ENABLED = False
GRADE_SNAP, GRADE_CROWD = "snap-accepted", "crowd"
EPS = 0.011                                  # rounding slack when the gate recomputes recorded numbers


def tol_of(v: float) -> float:
    return max(TOL_M, TOL_REL * abs(float(v)))


def agree_tol(a: float, b: float) -> float:
    return max(AGREE_M, AGREE_REL * max(abs(float(a)), abs(float(b))))


def prec_of(delta: float) -> float:
    return max(PREC_M, PREC_REL * abs(float(delta)))


def eave_agree_tol(a: float, b: float) -> float:
    return max(EAVE_AGREE_M, EAVE_AGREE_REL * max(abs(float(a)), abs(float(b))))


def is_slender(height: float, min_width: float) -> bool:
    w = max(float(min_width), 0.1)
    return bool(float(height) / w >= SLENDER_HW and w <= SLENDER_W)


def largest_agreeing(values: dict[str, float], tol=agree_tol) -> tuple[list[str], float | None]:
    """The largest set of >= 2 channels agreeing pairwise (ties: the tightest), and its mean."""
    names = sorted(values)
    best = None
    for r in range(len(names), 1, -1):
        for sub in itertools.combinations(names, r):
            v = [values[c] for c in sub]
            if all(abs(a - b) <= tol(a, b) for a in v for b in v):
                key = -float(np.ptp(v))
                if best is None or key > best[0]:
                    best = (key, list(sub))
        if best is not None:
            break
    if best is None:
        return [], None
    return best[1], float(np.mean([values[c] for c in best[1]]))


def _plain(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _plain(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_plain(v) for v in o]
    if isinstance(o, np.ndarray):
        return [_plain(v) for v in o.tolist()]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    return o


# ---------------------------------------------------------------- D25-H: each channel's verdict from its numbers


def skyline_verdict(sk: dict[str, Any] | None, map_top: float) -> dict[str, Any]:
    """The widened skyline (both masks): valid when neither mask is at its widened bound, the tops agree within
    max(1.5 m, 5 %), the part is the model skyline in >= 3 photos at the map value and >= 10 at the value, it has
    sky 6 px above in >= 5 photos (at the value going up, at the map value going down), and the 90 % interval
    (the union of both masks') is <= PREC."""
    sk = sk or {}
    ms = sk.get("masks") or {}
    out: dict[str, Any] = {"valid": False}
    if len(ms) < 2:
        out["why"] = "one sky mask only" if ms else "no skyline search"
        return out
    if any(m.get("at_bound") or m.get("flat") for m in ms.values()):
        out["why"] = "at a bound of the widened search (or a flat profile)"
        return out
    if any(m.get("top") is None for m in ms.values()):
        out["why"] = "a mask gives no value"
        return out
    tops = [float(m["top"]) for m in ms.values()]
    if max(tops) - min(tops) > max(SKY_MASKS_M, SKY_MASKS_REL * max(tops)):
        out["why"] = f"the masks' tops {min(tops):.2f}-{max(tops):.2f} m apart"
        return out
    v = float(np.mean(tops))
    lo, hi = min(float(m["p05"]) for m in ms.values()), max(float(m["p95"]) for m in ms.values())
    out.update(value=round(v, 2), interval=[round(lo, 2), round(hi, 2)])
    if int(sk.get("n_vis_map") or 0) < SKY_VIS_MAP:
        out["why"] = f"the model skyline in {sk.get('n_vis_map', 0)} (< {SKY_VIS_MAP}) photos at the map value"
        return out
    nv = min(int(m.get("n_vis_at") or 0) for m in ms.values())
    if nv < SKY_VIS_AT:
        out["why"] = f"the model skyline in {nv} (< {SKY_VIS_AT}) photos at the measured value"
        return out
    d = v - float(map_top)
    if d >= 0:
        ns = min(int(m.get("n_sky_at") or 0) for m in ms.values())
        if ns < SKY_ABOVE:
            out["why"] = f"sky above its edge in {ns} (< {SKY_ABOVE}) photos at the measured value"
            return out
    elif int(sk.get("n_sky_map") or 0) < SKY_ABOVE:
        out["why"] = f"sky above it in {sk.get('n_sky_map', 0)} (< {SKY_ABOVE}) photos at the map value: cannot say lower"
        return out
    if hi - lo > prec_of(d) + 1e-9:
        out["why"] = f"90 % interval {hi - lo:.2f} m > PREC {prec_of(d):.2f} m"
        return out
    out["valid"] = True
    return out


def _fit_usable(fit: dict[str, Any], map_top: float) -> str | None:
    qc = fit.get("quadrant_cells") or []
    if fit.get("top") is None:
        return "no roof-envelope fit"
    if float(fit.get("inlier_share") or 0) < SFM_FIT_INLIERS:
        return f"inlier share {fit.get('inlier_share')} < {SFM_FIT_INLIERS}"
    if int(fit.get("cells") or 0) < SFM_FIT_CELLS:
        return f"{fit.get('cells')} cells < {SFM_FIT_CELLS}"
    if sum(1 for n in qc if n >= SFM_FIT_QCELLS) < 3:
        return f"cells in < 3 quadrants with >= {SFM_FIT_QCELLS} each ({qc})"
    if fit.get("at_bound"):
        return "at its grid bound"
    w = float(fit["top_p95"]) - float(fit["top_p05"])
    if w > prec_of(float(fit["top"]) - float(map_top)) + 1e-9:
        return f"interval {w:.2f} m > PREC"
    return None


def quad_top(quad: dict[str, Any] | None) -> float | None:
    """The 3rd-highest quadrant q98 when >= 3 quadrants hold >= 20 points (each of track >= 3)."""
    q = quad or {}
    vals = [v for v, n in zip(q.get("quadrant_q98") or [], q.get("quadrant_points") or [])
            if v is not None and n >= SFM_QUAD_POINTS]
    if len(vals) < SFM_QUADS:
        return None
    return float(sorted(vals, reverse=True)[SFM_QUADS - 1])


def sfm_verdict(sf: dict[str, Any] | None, map_top: float, pitched: bool) -> dict[str, Any]:
    """Pitched roofs: the roof-envelope fit (inlier share >= 0.4, >= 30 cells, >= 3 quadrants with >= 3 cells, off its
    grid bounds, interval <= PREC); otherwise (or when the fit is not usable) the all-round top. Up only."""
    sf = sf or {}
    out: dict[str, Any] = {"valid": False}
    notes = []
    fit = sf.get("fit") or {}
    if pitched and fit:
        why = _fit_usable(fit, map_top)
        if why is None:
            out.update(valid=True, value=round(float(fit["top"]), 2),
                       interval=[round(float(fit["top_p05"]), 2), round(float(fit["top_p95"]), 2)], how="roof-fit")
            return out
        notes.append(f"roof fit: {why}")
    qt = quad_top(sf.get("quad"))
    if qt is not None:
        out.update(valid=True, value=round(qt, 2), interval=[round(qt, 2), round(qt, 2)], how="all-round q98")
        if notes:
            out["note"] = "; ".join(notes)
        return out
    notes.append(f"all-round top: fewer than {SFM_QUADS} quadrants with >= {SFM_QUAD_POINTS} points")
    out["why"] = "; ".join(notes)
    return out


def lines_record(recs: list[dict[str, Any]] | None) -> dict[str, Any] | None:
    by = {r.get("parameter"): r for r in recs or [] if r.get("value") is not None}
    return by.get("top") or by.get("eave")


def lines_verdict(ln: dict[str, Any] | None, map_top: float) -> dict[str, Any]:
    """The 3-D lines' apex / ridge (else the eave with the map's roof on it): >= 3 photos, >= 2 lines, off its window,
    interval <= PREC, its vote not withheld."""
    out: dict[str, Any] = {"valid": False}
    r = lines_record((ln or {}).get("records"))
    if r is None:
        out["why"] = "no 3-D line measurement"
        return out
    v = float(r.get("implied_top") if r.get("implied_top") is not None else r["value"])
    iv = r.get("interval") or [v, v]
    out.update(value=round(v, 2), interval=[round(float(iv[0]), 2), round(float(iv[1]), 2)], how=r.get("parameter"))
    if int(r.get("n_photos") or 0) < LINES_PHOTOS or int(r.get("n_lines") or 0) < LINES_LINES:
        out["why"] = f"{r.get('n_lines')} line(s) from {r.get('n_photos')} photos"
    elif r.get("at_bound"):
        out["why"] = "at its search window's edge"
    elif r.get("vote_withheld"):
        out["why"] = f"its vote withheld: {r['vote_withheld']}"
    elif float(iv[1]) - float(iv[0]) > prec_of(v - float(map_top)) + 1e-9:
        out["why"] = f"interval {float(iv[1]) - float(iv[0]):.2f} m > PREC"
    else:
        out["valid"] = True
    return out


SKYLINE, SFM, LINES = "skyline", "sfm-points", "lines3d"
CHANNELS = (SKYLINE, SFM, LINES)


def channel_verdicts(channels: dict[str, Any], map_top: float, pitched: bool) -> dict[str, dict[str, Any]]:
    return {SKYLINE: skyline_verdict(channels.get(SKYLINE), map_top),
            SFM: sfm_verdict(channels.get(SFM), map_top, pitched),
            LINES: lines_verdict(channels.get(LINES), map_top)}


# ---------------------------------------------------------------- D25-H: the rule before the box


def rule_h_pre(part: dict[str, Any], channels: dict[str, Any]) -> dict[str, Any]:
    """Triggers 1-3 of D25-H on the channels measured at the map value (part: id, map_top = the value judged against,
    z0, roof_h, pitched, min_width, parent_ok, pending (taller adjacent parts corrected first)). Returns the verdicts,
    the direction, the agreeing set and V, 'candidate' (the box may decide) and why not."""
    mt = float(part["map_top"])
    tol = tol_of(mt)
    ver = channel_verdicts(channels, mt, bool(part.get("pitched")))
    out: dict[str, Any] = {"part": part["id"], "map_top": round(mt, 3), "tol": round(tol, 3), "verdicts": ver,
                           "candidate": False, "why": ""}
    vals = {c: float(v["value"]) for c, v in ver.items() if v["valid"]}
    if is_slender(mt - float(part.get("z0", 0.0)), float(part["min_width"])):
        out["slender"] = True
        out["why"] = "slender part (own height / narrowest width >= 2, width <= 10 m): never moved automatically"
        return out
    ups = {c: v for c, v in vals.items() if v - mt > tol}
    downs = {c: v for c, v in vals.items() if mt - v > tol and c != SFM}
    near = {c: v for c, v in vals.items() if abs(v - mt) <= tol}
    if ups and downs:
        out["why"] = f"channels on both sides of the map: up {ups}, down {downs}"
        return out
    side = ups or downs
    if len(side) < 2:
        out["why"] = (f"{len(side)} channel(s) off the map by > TOL {tol:.2f} m ({side}); valid {vals}; "
                      + "; ".join(f"{c}: {v.get('why')}" for c, v in ver.items() if not v["valid"]))
        return out
    direction = "up" if ups else "down"
    contra = {c: v for c, v in near.items() if not (c == SFM and direction == "down")}
    if contra:
        out["why"] = f"contradicted: {contra} within TOL of the map while {sorted(side)} say {direction}"
        return out
    agree, V = largest_agreeing(side)
    if not agree:
        out["why"] = "no two channels agree: " + ", ".join(f"{c} {v:.2f}" for c, v in side.items())
        return out
    for c, v in side.items():
        if c in agree:
            continue
        if c == SFM and direction == "up" and v < V:
            continue                                  # an up-only channel below the value supports the direction
        out["why"] = f"{c} {v:.2f} m contradicts {' + '.join(agree)} on {V:.2f} m (> AGREE)"
        return out
    sf = channels.get(SFM) or {}
    if direction == "down":
        allround = quad_top(sf.get("quad"))
        if allround is not None and allround > V + agree_tol(allround, V):
            out["why"] = f"down contradicted: SfM points reach {allround:.2f} m all round, above {V:.2f} m + AGREE"
            return out
    if direction == "up" and sf.get("ceiling"):
        out["why"] = "up contradicted: the SfM points stop at the map's top all round (a ceiling)"
        return out
    out.update(direction=direction, agreeing=agree, V=round(V, 3))
    if not part.get("parent_ok", True):
        out["why"] = f"parents first: {part.get('parent_why') or 'its parent is not settled'}"
        out["deferred"] = True
        return out
    if part.get("pending"):
        out["why"] = f"taller adjacent disputed parts {list(part['pending'])} are corrected first: re-judged after them"
        out["deferred"] = True
        return out
    out["candidate"] = True
    out["why"] = f"{' + '.join(agree)} agree on {V:.2f} m ({direction} {abs(V - mt):.2f} m): the box decides"
    return out


# ---------------------------------------------------------------- scale-free anchors (§2.4.8 part 6)


def scale_from_walls(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    """Route 3: opposite outline walls (normals 180 +- 25 deg apart, >= 20 m apart, >= 200 wall points each); every
    qualifying pair within 3 % of the others and the widest pair's k within 3 % of 1; k_err = |k - 1| + 0.5 m / sep."""
    out: dict[str, Any] = {"route": "opposite-walls", "established": False, "pairs": _plain(pairs)}
    ok = [p for p in pairs if abs(abs((float(p["normals_deg"][0]) - float(p["normals_deg"][1])) % 360 - 180))
          <= WALL_ANGLE_TOL and float(p["map_sep_m"]) >= WALL_MIN_SEP_M and min(p["points"]) >= WALL_MIN_POINTS]
    out["qualifying"] = len(ok)
    if not ok:
        out["why"] = "no opposite-wall pair >= 20 m apart with >= 200 wall points on each wall"
        return out
    ks = [float(p["k"]) for p in ok]
    if max(ks) - min(ks) > SCALE_K_TOL + 1e-9:
        out["why"] = f"the opposite-wall pairs disagree: k {min(ks):.4f}-{max(ks):.4f} (> 3 %)"
        return out
    best = max(ok, key=lambda p: float(p["map_sep_m"]))
    k = float(best["k"])
    if abs(k - 1.0) > SCALE_K_TOL + 1e-9:
        out["why"] = f"opposite walls measure k {k:.4f}: off 1 by more than 3 %"
        return out
    out.update(established=True, k=round(k, 4), sep_m=float(best["map_sep_m"]),
               k_err=round(abs(k - 1.0) + 0.5 / float(best["map_sep_m"]), 5), edges=best.get("edges"),
               why=f"opposite walls (edges {best.get('edges')}, {float(best['map_sep_m']):.1f} m apart): k {k:.4f}")
    return out


def scale_from_similarity(status: dict[str, Any] | None, outline_width_m: float) -> dict[str, Any]:
    """Route 1: D15 ④'s whole-building similarity (judge.scale_status established: no snap rescale > 3 %, a similarity
    measured all round with k within 3 % of 1); k_err = |k - 1| + 0.5 m / outline width."""
    sc = status or {}
    by = sc.get("by") or {}
    out: dict[str, Any] = {"route": "similarity", "established": False, "status": _plain(sc),
                           "outline_width_m": round(float(outline_width_m), 3)}
    if not (sc.get("established") and abs(float(sc.get("snap_scale", 1.0)) - 1.0) <= SCALE_K_TOL
            and by.get("measured") and not by.get("at_bound") and by.get("k") is not None
            and abs(float(by["k"]) - 1.0) <= SCALE_K_TOL and outline_width_m > 0):
        out["why"] = sc.get("why") or "the D15 whole-building similarity does not establish the scale"
        return out
    k = float(by["k"])
    out.update(established=True, k=round(k, 4), k_err=round(abs(k - 1.0) + 0.5 / float(outline_width_m), 5),
               why=f"D15 similarity ({by.get('channel')}) k {k:.4f}")
    return out


def scale_from_eave_ratios(ratios: dict[str, float]) -> dict[str, Any]:
    """Route 2: >= 2 channels measure the main body's eave; their ratios (measured / map) within 5 % of each other;
    k = their mean, which must lie within 3 % of 1 (as on the other two routes: V itself is adopted, not V / k), and
    k_err = |k - 1| + ratio difference / mean ratio (the |k - 1| term as on the other routes: a conservative
    addition to the doc's 'ratio difference / mean ratio')."""
    out: dict[str, Any] = {"route": "eave-ratio", "established": False, "ratios": _plain(ratios)}
    r = {c: float(v) for c, v in ratios.items() if v is not None}
    if len(r) < 2:
        out["why"] = f"the main body's eave measured by {len(r)} channel(s) (< 2)"
        return out
    vs = list(r.values())
    mean = float(np.mean(vs))
    if (max(vs) - min(vs)) > 0.05 * max(abs(x) for x in vs) + 1e-9:
        out["why"] = f"the eave ratios disagree ({r})"
        return out
    if abs(mean - 1.0) > SCALE_K_TOL + 1e-9:
        out["why"] = f"the eave ratios say k {mean:.4f}: off 1 by more than 3 %"
        return out
    out.update(established=True, k=round(mean, 4), k_err=round(abs(mean - 1.0) + (max(vs) - min(vs)) / mean, 5),
               why=f"main-body eave ratios {r}")
    return out


def scale_anchor(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Any established route establishes the scale; the one with the smallest k_err is used."""
    est = [c for c in candidates if c.get("established")]
    if not est:
        return {"established": False, "route": None, "why": "; ".join(f"{c.get('route')}: {c.get('why')}" for c in candidates)
                or "no scale-free anchor measured", "routes": _plain(candidates)}
    best = min(est, key=lambda c: float(c["k_err"]))
    return {**_plain(best), "routes": _plain(candidates)}


def check_scale(sc: dict[str, Any] | None) -> str | None:
    """The anchor recomputed from its own numbers."""
    sc = sc or {}
    if not sc.get("established"):
        return "the scale is not established"
    route = sc.get("route")
    if route == "opposite-walls":
        again = scale_from_walls(sc.get("pairs") or [])
    elif route == "similarity":
        again = scale_from_similarity(sc.get("status"), float(sc.get("outline_width_m") or 0.0))
    elif route == "eave-ratio":
        again = scale_from_eave_ratios(sc.get("ratios") or {})
    else:
        return f"unknown scale route {route!r}"
    if not again.get("established"):
        return f"the {route} anchor does not establish the scale again: {again.get('why')}"
    if abs(float(again["k"]) - float(sc.get("k", math.nan))) > 1e-3 or \
            abs(float(again["k_err"]) - float(sc.get("k_err", math.nan))) > 1e-4:
        return f"the {route} anchor's k / k_err are not its numbers' ({again['k']}, {again['k_err']})"
    return None


# ---------------------------------------------------------------- the evidence-grade registration box (part 7)


def box_sigma(grade: str, *, snap_yaw_sigma_deg: float | None = None, gps_residual_median_m: float | None = None,
              reported_yaw_sigma_deg: float | None = None, ground_measured: bool = False,
              estimate_up_boot_deg: float | None = None, two_gravity_diff_deg: float | None = None,
              local_ground: bool = False) -> dict[str, Any]:
    """The sigma of each component (the table in §2.4.8 part 7). Gravity is not sampled on the local-ground route."""
    if grade == GRADE_SNAP:
        xy = SIGMA_SNAP_XY
        yaw = max(SIGMA_SNAP_YAW, float(snap_yaw_sigma_deg or 0.0))
        z = SIGMA_Z_GROUND if ground_measured else SIGMA_Z_NONE
    elif grade == GRADE_CROWD:
        xy = max(SIGMA_CROWD_XY, SIGMA_CROWD_GPS * float(gps_residual_median_m or 0.0))
        yaw = max(SIGMA_CROWD_YAW, SIGMA_CROWD_YAW_F * float(reported_yaw_sigma_deg or 0.0))
        z = SIGMA_Z_NONE
    else:
        raise ValueError(f"unknown registration grade {grade!r}")
    if two_gravity_diff_deg is not None:
        g = max(SIGMA_G_TWO, float(two_gravity_diff_deg))
    else:
        g = max(SIGMA_G_FLOOR, SIGMA_G_BOOT * float(estimate_up_boot_deg or 0.0))
    out = {"x_m": xy, "y_m": xy, "yaw_deg": yaw, "z_m": z, "gravity_deg": g, "ground_measured": bool(ground_measured),
           "two_gravity_diff_deg": two_gravity_diff_deg, "estimate_up_boot_deg": estimate_up_boot_deg,
           "snap_yaw_sigma_deg": snap_yaw_sigma_deg, "gps_residual_median_m": gps_residual_median_m,
           "reported_yaw_sigma_deg": reported_yaw_sigma_deg}
    if local_ground:
        out["z_m"] = SIGMA_Z_GROUND
        out["gravity_deg"] = None
    return out


AXIS_SIGMA = {"x": "x_m", "y": "y_m", "yaw": "yaw_deg", "gx": "gravity_deg", "gy": "gravity_deg"}


def box_axes(sigma: dict[str, Any]) -> list[str]:
    return ["x", "y", "yaw"] + (["gx", "gy"] if sigma.get("gravity_deg") else [])


def box_poses(sigma: dict[str, Any], seed: int = 25, alternatives: list[dict[str, float]] | None = None
              ) -> list[dict[str, Any]]:
    """The samples: the base, both ends of every axis, >= 12 random corners, >= 6 interior points (each component at
    +-3 sigma), and every snap alternative within the ambiguity margin."""
    axes = box_axes(sigma)
    half = {a: K_BOX * float(sigma[AXIS_SIGMA[a]]) for a in axes}
    zero = {a: 0.0 for a in axes}
    out = [{"kind": "base", "pose": dict(zero)}]
    for a in axes:
        for sg in (-1, 1):
            out.append({"kind": "axis", "pose": {**zero, a: sg * half[a]}})
    rng = np.random.default_rng(seed)
    corners = set()
    allc = list(itertools.product((-1, 1), repeat=len(axes)))
    order = rng.permutation(len(allc))
    for i in order[:max(BOX_CORNERS, 0)]:
        corners.add(allc[int(i)])
    for c in sorted(corners):
        out.append({"kind": "corner", "pose": {a: s * half[a] for a, s in zip(axes, c)}})
    for _ in range(BOX_INTERIOR):
        out.append({"kind": "interior", "pose": {a: float(rng.uniform(-1, 1)) * half[a] for a in axes}})
    for alt in alternatives or []:
        out.append({"kind": "snap-alternative", "pose": {**zero, **{a: float(alt.get(a, 0.0)) for a in axes}}})
    return _plain(out)


def box_structure_problem(box: dict[str, Any], need_corners: int = BOX_CORNERS, need_interior: int = BOX_INTERIOR
                          ) -> str | None:
    """The box carries what §2.4.8 part 7 asks: sigma at least the table's floors for its grade, the base sample,
    both ends of every axis at exactly +-3 sigma, >= 12 corners, >= 6 interior points, nothing outside the box."""
    sig = box.get("sigma") or {}
    grade = box.get("grade")
    if float(box.get("k", 0)) < K_BOX - 1e-9:
        return f"box k {box.get('k')} < {K_BOX:g}"
    if grade == GRADE_SNAP:
        floors = {"x_m": SIGMA_SNAP_XY, "y_m": SIGMA_SNAP_XY, "yaw_deg": SIGMA_SNAP_YAW}
    elif grade == GRADE_CROWD:
        floors = {"x_m": SIGMA_CROWD_XY, "y_m": SIGMA_CROWD_XY, "yaw_deg": SIGMA_CROWD_YAW}
    else:
        return f"unknown registration grade {grade!r}"
    for k, f in floors.items():
        if float(sig.get(k) or 0.0) < f - 1e-9:
            return f"box sigma {k} {sig.get(k)} below the {grade} floor {f:g}"
    lg = box.get("local_ground") or {}
    if sig.get("gravity_deg") is None:
        if not (lg.get("passed") and lg.get("injection_test_passed")):
            return "gravity not sampled without a local ground that passes its check and its tilt-injection test"
        if float(sig.get("z_m") or 0.0) < SIGMA_Z_GROUND - 1e-9:
            return "ground sigma below 0.5 m"
    else:
        gfloor = SIGMA_G_TWO if sig.get("two_gravity_diff_deg") is not None else SIGMA_G_FLOOR
        if float(sig["gravity_deg"]) < gfloor - 1e-9:
            return f"gravity sigma {sig['gravity_deg']} below {gfloor:g}"
        zfloor = SIGMA_Z_GROUND if (sig.get("ground_measured") and grade == GRADE_SNAP) else SIGMA_Z_NONE
        if float(sig.get("z_m") or 0.0) < zfloor - 1e-9:
            return f"ground sigma {sig.get('z_m')} below {zfloor:g}"
    axes = box_axes(sig)
    half = {a: K_BOX * float(sig[AXIS_SIGMA[a]]) for a in axes}
    samples = box.get("samples") or []
    kinds = [s.get("kind") for s in samples]
    if "base" not in kinds:
        return "no base sample"
    for s in samples:
        pose = s.get("pose") or {}
        if set(pose) != set(axes):
            return f"a sample's pose {sorted(pose)} does not cover the axes {axes}"
        for a in axes:
            if abs(float(pose[a])) > half[a] + 1e-6 and s.get("kind") != "snap-alternative":
                return f"a sample outside the box ({a} {pose[a]} > {half[a]:.3f})"
        if s.get("kind") == "base" and any(abs(float(pose[a])) > 1e-9 for a in axes):
            return "the base sample is not at the registration"
    for a in axes:
        for sg in (-1, 1):
            if not any(s.get("kind") == "axis" and abs(float(s["pose"][a]) - sg * half[a]) < 1e-6
                       and all(abs(float(s["pose"][b])) < 1e-9 for b in axes if b != a) for s in samples):
                return f"axis {a} end {sg * half[a]:+.3f} not sampled"
    corners = [s for s in samples if s.get("kind") == "corner"
               and all(abs(abs(float(s["pose"][a])) - half[a]) < 1e-6 for a in axes)]
    if len({tuple(np.sign(float(s["pose"][a])) for a in axes) for s in corners}) < min(need_corners, 2 ** len(axes)):
        return f"{len(corners)} distinct corners (< {need_corners})"
    if sum(1 for s in samples if s.get("kind") == "interior") < need_interior:
        return f"fewer than {need_interior} interior samples"
    return None


def box_bounds(box: dict[str, Any], agreeing: list[str], V: float, k_err: float, key: str = "values"
               ) -> dict[str, Any]:
    """LB / UB over the samples: per sample and agreeing channel (value - half its 90 % interval), the minimum, minus
    the scale term k_err x V and the ground term 3 sigma_z (UB symmetric). A channel without a measurement on a
    sample does not count there; a sample where every agreeing channel lost its measurement refuses."""
    zt = K_BOX * float((box.get("sigma") or {}).get("z_m") or SIGMA_Z_NONE)
    st_ = float(k_err) * float(V)
    lo, hi, lost, dead = math.inf, -math.inf, [], []
    for i, s in enumerate(box.get("samples") or []):
        vals = s.get(key) or {}
        got = 0
        for c in agreeing:
            r = vals.get(c)
            if not r or r.get("value") is None:
                lost.append([i, c])
                continue
            got += 1
            v, h = float(r["value"]), abs(float(r.get("half") or 0.0))
            lo, hi = min(lo, v - h), max(hi, v + h)
        if got == 0:
            dead.append(i)
    out = {"scale_term_m": round(st_, 3), "ground_term_m": round(zt, 3), "lost": lost, "samples_without_any": dead}
    if dead or not math.isfinite(lo):
        out["why"] = f"every agreeing channel lost its measurement on sample(s) {dead}"
        return out
    out.update(LB=round(lo - st_ - zt, 3), UB=round(hi + st_ + zt, 3))
    return out


def adopt_value(direction: str, V: float, LB: float, UB: float, map_v: float) -> tuple[str, float]:
    """The midpoint rule: up, LB >= (V + map) / 2 adopts V, else LB; down, UB <= (V + map) / 2 adopts V, else UB."""
    mid = 0.5 * (float(V) + float(map_v))
    if direction == "up":
        return ("measured", float(V)) if LB >= mid - 1e-9 else ("lower bound", float(LB))
    return ("measured", float(V)) if UB <= mid + 1e-9 else ("upper bound", float(UB))


def rule_h_box(pre: dict[str, Any], box: dict[str, Any], scale: dict[str, Any], part: dict[str, Any]) -> dict[str, Any]:
    """Triggers 4-5 and the value of D25-H for a candidate of rule_h_pre: the scale established and V / k still off
    by > TOL; the box well formed; LB - map >= TOL (up) or map - UB >= TOL (down); the midpoint rule; a wall left under
    the roof (height mode). Returns {'fires', 'adopted', 'value', 'LB', 'UB', 'why', ...}."""
    mt, tol = float(pre["map_top"]), float(pre["tol"])
    out: dict[str, Any] = {"fires": False}
    if not pre.get("candidate"):
        out["why"] = pre.get("why") or "not a candidate"
        return out
    V, direction, agree = float(pre["V"]), pre["direction"], list(pre["agreeing"])
    if not (scale or {}).get("established"):
        out["why"] = f"the scale is not established: {(scale or {}).get('why')}"
        return out
    k, k_err = float(scale["k"]), float(scale["k_err"])
    if abs(V / k - mt) <= tol:
        out["why"] = f"the scale-free value V / k = {V / k:.2f} m lies within TOL of the map"
        return out
    if box.get("grade") == GRADE_CROWD and not CROWD_HB_ENABLED:
        out["why"] = "crowd-grade registration: D25-H is off until the corner-box adversarial set passes"
        return out
    bad = box_structure_problem(box)
    if bad:
        out["why"] = f"registration box: {bad}"
        return out
    bb = box_bounds(box, agree, V, k_err)
    out["box"] = {k_: v for k_, v in bb.items()}
    if "LB" not in bb:
        out["why"] = f"registration box: {bb['why']}"
        return out
    LB, UB = float(bb["LB"]), float(bb["UB"])
    out.update(LB=round(LB, 3), UB=round(UB, 3), midpoint=round(0.5 * (V + mt), 3))
    if direction == "up" and LB - mt < tol - 1e-9:
        out["why"] = f"guaranteed change LB - map = {LB - mt:.2f} m < TOL {tol:.2f} m"
        return out
    if direction == "down" and mt - UB < tol - 1e-9:
        out["why"] = f"guaranteed change map - UB = {mt - UB:.2f} m < TOL {tol:.2f} m"
        return out
    adopted, value = adopt_value(direction, V, LB, UB, mt)
    if value - float(part.get("roof_h", 0.0)) - float(part.get("z0", 0.0)) < ROOF_MIN_M:
        out["why"] = f"a top of {value:.2f} m leaves no wall under the {part.get('roof_h')} m roof"
        return out
    out.update(fires=True, adopted=adopted, value=round(value, 3),
               provenance=PROV_FULL if adopted == "measured" else PROV_LB,
               why=(f"{' + '.join(agree)} agree on {V:.2f} m; box [{LB:.2f}, {UB:.2f}] m; "
                    + (f"bound beyond the midpoint {0.5 * (V + mt):.2f} m: V adopted" if adopted == "measured" else
                       f"bound short of the midpoint {0.5 * (V + mt):.2f} m: only the bound {value:.2f} m adopted")))
    return out


# ---------------------------------------------------------------- the eave (roof height) beside a full-value top


def eave_readings(channels: dict[str, Any], map_eave: float) -> dict[str, float]:
    """The eave as each channel measures it: the 2-D skyline search's eave (both masks within 1.5 m, off its bounds),
    the SfM roof fit's eave (a usable fit, its eave interval <= PREC), the 3-D lines' eave record (>= 3 photos,
    >= 2 lines, off its window, not withheld)."""
    out: dict[str, float] = {}
    ms = (channels.get(SKYLINE) or {}).get("masks") or {}
    es = [m.get("eave") for m in ms.values()]
    if len(es) == 2 and all(e is not None for e in es) and abs(es[0] - es[1]) <= SKY_MASKS_M and \
            not any(m.get("at_bound") for m in ms.values()):
        out[SKYLINE] = round(float(np.mean(es)), 2)
    fit = (channels.get(SFM) or {}).get("fit") or {}
    if fit.get("eave") is not None and float(fit.get("inlier_share") or 0) >= SFM_FIT_INLIERS and not fit.get("at_bound") \
            and float(fit["eave_p95"]) - float(fit["eave_p05"]) <= prec_of(float(fit["eave"]) - map_eave) + 1e-9:
        out[SFM] = round(float(fit["eave"]), 2)
    for r in (channels.get(LINES) or {}).get("records") or []:
        if r.get("parameter") == "eave" and r.get("value") is not None and int(r.get("n_lines") or 0) >= LINES_LINES \
                and int(r.get("n_photos") or 0) >= LINES_PHOTOS and not r.get("vote_withheld") and not r.get("at_bound"):
            out[LINES] = round(float(r["value"]), 2)
    return out


def rule_eave(channels: dict[str, Any], part: dict[str, Any], direction: str, V: float, box: dict[str, Any],
              scale: dict[str, Any]) -> dict[str, Any]:
    """The eave adopted with a full-value top: >= 2 eave readings agree within max(1.5 m, 7.5 %), the eave passes
    triggers 1-5 itself (off the map's eave by > TOL on the top's side, no reading within TOL of it, the box) and the
    midpoint rule; the roof height becomes V - eave (>= 0.5 m). Otherwise the map's roof height is kept."""
    map_eave = float(part["map_top"]) - float(part.get("roof_h", 0.0))
    out: dict[str, Any] = {"adopt": False, "map_eave": round(map_eave, 3)}
    rd = eave_readings(channels, map_eave)
    out["readings"] = rd
    if len(rd) < 2:
        out["why"] = f"{len(rd)} eave reading(s): the map's roof height is kept"
        return out
    agree, E = largest_agreeing(rd, eave_agree_tol)
    if not agree:
        out["why"] = f"the eave readings disagree ({rd}): the map's roof height is kept"
        return out
    tol = tol_of(map_eave)
    sg = 1 if direction == "up" else -1
    if any(abs(v - map_eave) <= tol or (v - map_eave) * sg < 0 for v in rd.values()):
        out["why"] = f"an eave reading lies within TOL of the map's eave {map_eave:.2f} m or on its other side"
        return out
    if any(c not in agree for c in rd):
        out["why"] = f"eave readings outside the agreeing set {agree}: {rd}"
        return out
    bb = box_bounds(box, agree, E, float(scale["k_err"]), key="eave")
    out.update(agreeing=agree, E=round(E, 3), box=bb)
    if "LB" not in bb:
        out["why"] = f"eave box: {bb['why']}"
        return out
    if (direction == "up" and bb["LB"] - map_eave < tol - 1e-9) or (direction == "down" and map_eave - bb["UB"] < tol - 1e-9):
        out["why"] = f"the eave's guaranteed change is below TOL {tol:.2f} m"
        return out
    adopted, ev = adopt_value(direction, E, bb["LB"], bb["UB"], map_eave)
    if adopted != "measured":
        out["why"] = f"the eave's bound misses the midpoint: the map's roof height is kept"
        return out
    roof = float(V) - E
    if roof < ROOF_MIN_M or E - float(part.get("z0", 0.0)) < ROOF_MIN_M:
        out["why"] = f"eave {E:.2f} m under a top of {V:.2f} m leaves no roof or no wall"
        return out
    out.update(adopt=True, value=round(E, 3), roof_height=round(roof, 3),
               why=f"{' + '.join(agree)} put the eave at {E:.2f} m: roof height {roof:.2f} m")
    return out


# ---------------------------------------------------------------- D25-B: photo-measured blocks


def rule_b(part: dict[str, Any], readings: dict[str, dict[str, Any]], carve_p25: float | None,
           box: dict[str, Any] | None, scale: dict[str, Any] | None) -> dict[str, Any]:
    """Three lower-bound readings (3-D lines' highest horizontal cluster's lowest end, SfM 3rd-highest quadrant q98,
    the skyline's lower interval end), each at its minimum over the box (less the ground term 3 sigma_z and the scale
    term k_err x the reading: the scale must be established, §2.4.8 part 6); H = the 2nd-highest; the carving upper
    bound's 25 % quantile >= H; H - 3 m >= TOL; not slender at H. part: id, min_width, z0 (0), map_top (the stub)."""
    out: dict[str, Any] = {"fires": False, "part": part["id"]}
    base = {c: float(r["value"]) for c, r in readings.items() if r and r.get("valid") and r.get("value") is not None}
    out["readings"] = {c: round(v, 2) for c, v in base.items()}
    if not (scale or {}).get("established"):
        out["why"] = f"the scale is not established: {(scale or {}).get('why')}"
        return out
    k_err = float(scale["k_err"])
    if box is not None:
        if box.get("grade") == GRADE_CROWD and not CROWD_HB_ENABLED:
            out["why"] = "crowd-grade registration: D25-B is off until the corner-box adversarial set passes"
            return out
        bad = box_structure_problem(box)
        if bad:
            out["why"] = f"registration box: {bad}"
            return out
        mins = {}
        for c in base:
            vs = [float(s["values"][c]["value"]) for s in box.get("samples") or []
                  if (s.get("values") or {}).get(c) and s["values"][c].get("value") is not None]
            if vs:
                mins[c] = min(vs)
        zt = K_BOX * float((box.get("sigma") or {}).get("z_m") or SIGMA_Z_NONE)
        mins = {c: v - zt - k_err * v for c, v in mins.items()}
        out["ground_term_m"] = round(zt, 3)
        out["k_err"] = k_err
    else:
        out["why"] = "no registration box"
        return out
    out["box_minima"] = {c: round(v, 3) for c, v in mins.items()}
    if len(mins) < BLOCK_MIN_READINGS:
        out["why"] = f"{len(mins)} lower-bound reading(s) (< {BLOCK_MIN_READINGS})"
        return out
    H = sorted(mins.values(), reverse=True)[1]
    out["H"] = round(H, 3)
    out["supported_by"] = sorted(c for c, v in mins.items() if v >= H - 1e-9)
    if carve_p25 is None:
        out["why"] = "no sky-carving upper bound"
        return out
    out["carve_p25"] = round(float(carve_p25), 2)
    if float(carve_p25) < H - 1e-9:
        out["why"] = f"the carving upper bound's 25 % quantile {carve_p25:.2f} m lies below H {H:.2f} m: contradiction, a ticket"
        return out
    stub = float(part.get("map_top", BLOCK_STUB_M))
    if H - stub < tol_of(stub) - 1e-9:
        out["why"] = f"H {H:.2f} m - the {stub:g} m stub < TOL {tol_of(stub):.2f} m"
        return out
    if is_slender(H - float(part.get("z0", 0.0)), float(part["min_width"])):
        out["why"] = f"slender at {H:.2f} m (h/w {H / max(float(part['min_width']), 0.1):.1f}): a ticket"
        return out
    if part.get("pending"):
        out["why"] = f"taller adjacent parts {list(part['pending'])} are corrected first"
        return out
    out.update(fires=True, value=round(H, 3), provenance=PROV_LB,
               why=f"at least {H:.2f} m by {out['supported_by']}; carving p25 {float(carve_p25):.2f} m not below")
    return out


# ---------------------------------------------------------------- D25-T: tilt


def tilt_shape_gate(ring: np.ndarray, stack_height: float) -> dict[str, Any]:
    """The main outline's circularity 4 pi A / P^2 >= 0.97 and the stack's height >= 1.5 x its diameter."""
    r = np.asarray(ring, float)
    A = abs(S.ring_area(r))
    P = float(np.sum(np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1)))
    circ = 4 * math.pi * A / max(P * P, 1e-9)
    D = 2.0 * math.sqrt(A / math.pi)
    hd = float(stack_height) / max(D, 1e-9)
    return {"circularity": round(circ, 4), "diameter_m": round(D, 3), "height_m": round(float(stack_height), 3),
            "h_over_d": round(hd, 3), "passed": bool(circ >= TILT_CIRC and hd >= TILT_HD)}


def tilt_channel_ok(name: str, c: dict[str, Any]) -> str | None:
    if c.get("lean_en_deg") is None:
        return "no measurement"
    if name == "points":
        if int(c.get("bands") or 0) < TILT_POINTS_BANDS or float(c.get("sigma_deg", 9)) > TILT_POINTS_SIGMA:
            return f"{c.get('bands')} bands, sigma {c.get('sigma_deg')}"
    elif name == "silhouette":
        n = int(c.get("n_photos") or 0)
        if n < TILT_SIL_PHOTOS or float(c.get("contrast", 0)) < TILT_SIL_CONTRAST or \
                float(c.get("inlier_share", 0)) < TILT_SIL_INLIERS or float(c.get("sigma_deg", 9)) > TILT_SIL_SIGMA:
            return f"{n} photos, contrast {c.get('contrast')}, inliers {c.get('inlier_share')}, sigma {c.get('sigma_deg')}"
    elif name == "lines3d":
        if int(c.get("n_lines") or 0) < TILT_LINES_N or int(c.get("n_photos") or 0) < TILT_LINES_PHOTOS or \
                float(c.get("sigma_deg", 9)) > TILT_LINES_SIGMA:
            return f"{c.get('n_lines')} lines, {c.get('n_photos')} photos, sigma {c.get('sigma_deg')}"
    else:
        return f"unknown tilt channel {name!r}"
    return None


def gravity_ok(name: str, g: dict[str, Any]) -> str | None:
    if g.get("lean_en_deg") is None:
        return "no estimate"
    if name == "camera":
        if not g.get("rolled_removed_checked") or float(g.get("roll_max_deg", 99)) > TILT_ROLL_MAX:
            return f"cameras with |roll| > {TILT_ROLL_MAX:g} deg not removed"
    elif name == "scene-verticals":
        if int(g.get("inliers") or 0) < TILT_VP_INLIERS or int(g.get("photos") or 0) < TILT_VP_PHOTOS or \
                float(g.get("sigma_deg", 9)) > TILT_VP_SIGMA:
            return f"{g.get('inliers')} inlier segments, {g.get('photos')} photos, sigma {g.get('sigma_deg')}"
    else:
        return f"unknown gravity reference {name!r}"
    return None


def polar(v) -> tuple[float, float]:
    v = np.asarray(v, float)
    return float(np.hypot(v[0], v[1])), float(math.degrees(math.atan2(v[0], v[1])) % 360.0)


def rule_t(channels: dict[str, dict[str, Any]], gravity: dict[str, dict[str, Any]], sigma_yaw_deg: float,
           shape: dict[str, Any]) -> dict[str, Any]:
    """D25-T on one reconstruction: the shape gate; >= 2 channels pass; both gravity references pass; T = the
    component-wise median of every channel x gravity lean vector (east, north, degrees), t = |T|, spread = the largest
    distance from T; adopt iff t >= max(2 deg, 3 x spread), spread <= max(0.5 deg, 0.3 t) and t >= 2 (spread +
    t sin(3 sigma_yaw))."""
    out: dict[str, Any] = {"fires": False, "shape": _plain(shape), "sigma_yaw_deg": float(sigma_yaw_deg)}
    if not (shape or {}).get("passed"):
        out["why"] = f"shape gate: circularity {shape.get('circularity')} (>= {TILT_CIRC}), h/d {shape.get('h_over_d')} (>= {TILT_HD})"
        return out
    okc = {n: c for n, c in channels.items() if tilt_channel_ok(n, c) is None}
    out["channels_ok"] = sorted(okc)
    out["channels_refused"] = {n: tilt_channel_ok(n, c) for n, c in channels.items() if n not in okc}
    if len(okc) < 2:
        out["why"] = f"{len(okc)} tilt channel(s) pass (< 2): {out['channels_refused']}"
        return out
    gok = {n: g for n, g in gravity.items() if gravity_ok(n, g) is None}
    out["gravity_refused"] = {n: gravity_ok(n, g) for n, g in gravity.items() if n not in gok}
    if not {"camera", "scene-verticals"} <= set(gok):
        out["why"] = f"two gravity references are needed: {out['gravity_refused'] or 'missing'}"
        return out
    combos = {f"{c}|{g}": (np.asarray(okc[c]["lean_en_deg"], float) - np.asarray(gok[g]["lean_en_deg"], float))
              for c in sorted(okc) for g in sorted(gok)}
    Vs = np.array(list(combos.values()))
    T = np.median(Vs, axis=0)
    spread = float(np.max(np.linalg.norm(Vs - T, axis=1)))
    t, az = polar(T)
    U = spread + t * math.sin(math.radians(K_BOX * float(sigma_yaw_deg)))
    out.update(combos={k: [round(float(x), 4) for x in v] for k, v in combos.items()},
               T_en_deg=[round(float(T[0]), 4), round(float(T[1]), 4)], tilt_deg=round(t, 3), azimuth_deg=round(az, 2),
               spread_deg=round(spread, 3), U_deg=round(U, 3),
               gravity_disagreement_deg=round(float(np.linalg.norm(np.asarray(gok["camera"]["lean_en_deg"], float)
                                                                   - np.asarray(gok["scene-verticals"]["lean_en_deg"], float))), 3))
    if t < max(TILT_MIN, TILT_K * spread) - 1e-9:
        out["why"] = f"tilt {t:.2f} deg < max({TILT_MIN:g}, 3 x spread {spread:.2f})"
        return out
    if spread > max(TILT_SPREAD_FLOOR, TILT_REL * t) + 1e-9:
        out["why"] = f"spread {spread:.2f} deg > max(0.5, 30 % x {t:.2f})"
        return out
    if t < 2.0 * U - 1e-9:
        out["why"] = f"yaw term: tilt {t:.2f} deg < 2 x (spread {spread:.2f} + t sin(3 x {sigma_yaw_deg:.2f} deg)) = {2 * U:.2f}"
        return out
    out.update(fires=True, provenance=PROV_FULL, why=f"tilt {t:.2f} deg toward {az:.1f} deg (spread {spread:.2f}, U {U:.2f})")
    return out


def tilt_consensus(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Several reconstructions that pass D25-T must agree within max(0.5 deg, 0.3 t)."""
    ok = [r for r in results if r.get("fires")]
    out: dict[str, Any] = {"passing": len(ok), "agree": True}
    for a, b in itertools.combinations(ok, 2):
        d = float(np.linalg.norm(np.asarray(a["T_en_deg"]) - np.asarray(b["T_en_deg"])))
        tol = max(TILT_SPREAD_FLOOR, TILT_REL * min(a["tilt_deg"], b["tilt_deg"]))
        if d > tol + 1e-9:
            out.update(agree=False, why=f"two passing reconstructions differ by {d:.2f} deg (> {tol:.2f})")
            return out
    return out


def pose_from_tilt(T_en_deg, pivot) -> dict[str, Any]:
    t, az = polar(T_en_deg)
    return {"kind": "tilt", "lean_en_deg": [round(float(T_en_deg[0]), 4), round(float(T_en_deg[1]), 4)],
            "tilt_deg": round(t, 4), "azimuth_deg": round(az, 3),
            "pivot": [round(float(pivot[0]), 4), round(float(pivot[1]), 4), round(float(pivot[2]), 4)], "parts": "all"}


def pose_matrix(pose: dict[str, Any] | None) -> tuple[np.ndarray, np.ndarray]:
    """(R, pivot) of a tilt pose: the vertical turns toward the azimuth by the tilt (Rodrigues about the horizontal
    axis z x d)."""
    if not pose:
        return np.eye(3), np.zeros(3)
    t = math.radians(float(pose["tilt_deg"]))
    az = math.radians(float(pose["azimuth_deg"]))
    d = np.array([math.sin(az), math.cos(az), 0.0])
    a = np.array([-d[1], d[0], 0.0])
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)
    return R, np.asarray(pose["pivot"], float)


def apply_pose(xyz: np.ndarray, pose: dict[str, Any] | None, inverse: bool = False) -> np.ndarray:
    if not pose:
        return np.asarray(xyz, float)
    R, c = pose_matrix(pose)
    X = np.asarray(xyz, float).reshape(-1, 3)
    Y = (X - c) @ (R if inverse else R.T) + c
    return Y.reshape(np.shape(xyz))


# ---------------------------------------------------------------- applying corrections (the reversible list)


def _active(model: S.StructureModel, kinds=KINDS) -> list[dict[str, Any]]:
    return [a for a in model.adoptions if a.get("active", True) and a.get("kind") in kinds]


def _snapshot(pid: str, k: str, v: S.Param) -> dict[str, Any]:
    return {"part": pid, "param": k, "value": float(v.value), "source": v.source, "state": v.state,
            "evidence": copy.deepcopy(v.evidence), "ticket": v.ticket, "map_value": v.map_value}


def _map_source(v: S.Param) -> str:
    for e in v.evidence:
        if e.get("kind") in ("carried", "adoption", "correction") and e.get("map_source"):
            return e["map_source"]
    return v.source


def _depends(model: S.StructureModel, p: S.PartSpec, extra: list[str] | None) -> list[str]:
    """What a correction rests on: the active frame adoption (D15 ⑥: heights are measured on that outline) and the
    adoptions or corrections that carried this part, plus the caller's (the neighbours it was re-measured after)."""
    dep = set(extra or [])
    dep |= {a["id"] for a in model.adoptions if a.get("kind") == "frame" and a.get("active", True)}
    for v in p.params.values():
        dep |= {e["adoption"] for e in v.evidence if e.get("kind") == "carried" and e.get("adoption")}
    return sorted(dep)


def _next_id(model: S.StructureModel) -> str:
    return f"D25-{model.landmark}-{sum(1 for a in model.adoptions if a.get('kind') in KINDS) + 1}"


def _carry(model: S.StructureModel, pid: str, dz: float, rid: str) -> list[dict[str, Any]]:
    """The parts resting on a corrected part follow it (D16 ②): base and top move by its change, provenance
    photo-corrected (auto) with a carry record; a 'height pending' stub's placeholder top moves but stays a stub."""
    out = []
    if abs(dz) < 1e-9:
        return out
    for cid in S.descendants(model, pid):
        c = model.part(cid)
        stub = c.params["height"].source == st.PROV_DEFAULT
        before = [_snapshot(cid, k, c.params[k]) for k in ("min_height", "height")]
        for k in ("min_height", "height"):
            v = c.params[k]
            ms = _map_source(v)
            if v.map_value is None:
                v.map_value = float(v.value)
            v.value = float(v.value) + dz
            if not (stub and k == "height"):
                v.source, v.state = PROV_FULL, S.STATE_CORRECTED
            v.evidence = list(v.evidence) + [{"kind": "carried", "parent": pid, "adoption": rid, "rule": RULE,
                                              "dz": round(dz, 3), "map_source": ms,
                                              **({"stub": True} if stub and k == "height" else {})}]
        c.problems.append(f"D25 {rid}: carried {dz:+.2f} m with {pid}" + ("; still a 'height pending' stub" if stub else ""))
        out.append({"part": cid, "dz": round(dz, 3), "before": before, **({"stub": True} if stub else {})})
    return out


def _evidence_records(channels: dict[str, Any], verdicts: dict[str, Any], snap_scale: float) -> list[dict[str, Any]]:
    """Per-channel evidence entries on the parameter (structmetrics counts the channels that measured it)."""
    ev = []
    sk = channels.get(SKYLINE) or {}
    for m, x in (sk.get("masks") or {}).items():
        ev.append({"kind": "skyline", "mask": m, "skyline_photos": int(x.get("n_vis_at") or 0),
                   "sky_above_photos": int(x.get("n_sky_at") or 0), "top": x.get("top"), "rule": RULE,
                   "valid": bool(verdicts.get(SKYLINE, {}).get("valid")), "snap_scale": round(float(snap_scale), 4)})
    sf = channels.get(SFM) or {}
    if sf:
        ev.append({"kind": "sfm-points", "value": verdicts.get(SFM, {}).get("value"), "how": verdicts.get(SFM, {}).get("how"),
                   "points": int(sum((sf.get("quad") or {}).get("quadrant_points") or [0])), "rule": RULE,
                   "valid": bool(verdicts.get(SFM, {}).get("valid")), "snap_scale": round(float(snap_scale), 4)})
    r = lines_record((channels.get(LINES) or {}).get("records"))
    if r is not None:
        ev.append({"kind": "lines3d", "parameter": r.get("parameter"), "value": r.get("implied_top", r.get("value")),
                   "n_photos": int(r.get("n_photos") or 0), "n_lines": int(r.get("n_lines") or 0), "rule": RULE,
                   "valid": bool(verdicts.get(LINES, {}).get("valid")), "snap_scale": round(float(snap_scale), 4)})
    return ev


def _part_geometry(p: S.PartSpec) -> dict[str, Any]:
    return {"z0": round(float(p.z0), 3), "roof_h": round(float(p.roof_h), 3), "min_width": round(float(p.min_width()), 3),
            "pitched": bool(p.roof_shape not in ("flat", "dome", "onion") and p.roof_h > ROOF_MIN_M),
            "roof_shape": p.roof_shape}


def apply_height(model: S.StructureModel, pid: str, pre: dict[str, Any], fin: dict[str, Any], channels: dict[str, Any],
                 box: dict[str, Any], scale: dict[str, Any], *, eave: dict[str, Any] | None = None, round_: int = 0,
                 snap_scale: float = 1.0, depends_on: list[str] | None = None) -> list[dict[str, Any]]:
    """Write one D25-H correction (and, with a full-value top and an adopted eave, its roof height as a d25-roof record
    depending on it): the part's height becomes V (photo-corrected (auto)) or the bound (photo-corrected (auto, lower
    bound)), the map's value and source kept, every channel's numbers, the box, the scale anchor and the verdicts in
    the record; children carried; the part's ticket closed (a bound leaves it open: 'at least ...')."""
    p = model.part(pid)
    h = p.params["height"]
    rid = _next_id(model)
    old = float(h.value)
    map_value = float(h.map_value) if h.map_value is not None else old
    map_source = _map_source(h)
    new = float(fin["value"])
    before = [_snapshot(pid, "height", h)]
    geo = _part_geometry(p)
    rec: dict[str, Any] = {
        "id": rid, "kind": KIND_H, "part": pid, "name": p.name, "param": "height",
        "map_value": {"value": round(map_value, 3), "source": map_source}, "judged_against": round(old, 3),
        "value": round(new, 3), "V": pre["V"], "LB": fin["LB"], "UB": fin["UB"], "adopted": fin["adopted"],
        "direction": pre["direction"], "tol": pre["tol"], "agreeing": list(pre["agreeing"]),
        "channels": _plain(channels), "verdicts": _plain(pre["verdicts"]), "geometry": geo,
        "registration_box": _plain(box), "scale": _plain(scale), "round": int(round_),
        "rule": RULE, "provenance": fin["provenance"], "label": LABEL, "active": True, "reversible": True,
        "osm_edited": False, "snap_scale": round(float(snap_scale), 4), "depends_on": _depends(model, p, depends_on),
    }
    ev = {"kind": "correction", "id": rid, "rule": RULE, "map_value": round(map_value, 3), "map_source": map_source,
          "value": round(new, 3), "adopted": fin["adopted"], "snap_scale": round(float(snap_scale), 4)}
    h.evidence = _evidence_records(channels, pre["verdicts"], snap_scale) + [ev]
    h.value, h.source, h.state, h.map_value = new, fin["provenance"], S.STATE_CORRECTED, map_value
    carried = _carry(model, pid, new - old, rid)
    for c in carried:
        before.extend(c.pop("before"))
    rec["carried"] = carried
    rec["before"] = before
    model.adoptions.append(_plain(rec))
    out = [model.adoptions[-1]]
    note = (f"D25 {rid}: height {old:g} -> {new:g} m {fin['provenance']} (V {pre['V']:g}, box [{fin['LB']:g}, "
            f"{fin['UB']:g}]); the map's {map_value:g} m kept")
    p.problems.append(note)
    if eave and eave.get("adopt") and fin["adopted"] == "measured":
        rh = p.params["roof_height"]
        rid2 = _next_id(model)
        rh_map = float(rh.map_value) if rh.map_value is not None else float(rh.value)
        rec2 = {"id": rid2, "kind": KIND_ROOF, "part": pid, "name": p.name, "param": "roof_height",
                "map_value": {"value": round(rh_map, 3), "source": _map_source(rh)},
                "judged_against": round(float(rh.value), 3), "value": round(float(eave["roof_height"]), 3),
                "eave": _plain(eave), "height_record": rid, "top_value": round(new, 3), "geometry": geo,
                "rule": RULE, "provenance": PROV_FULL, "label": LABEL, "active": True, "reversible": True,
                "osm_edited": False, "snap_scale": round(float(snap_scale), 4), "depends_on": [rid],
                "before": [_snapshot(pid, "roof_height", rh)]}
        rh.evidence = list(rh.evidence) + [{"kind": "correction", "id": rid2, "rule": RULE, "map_value": round(rh_map, 3),
                                            "map_source": _map_source(rh), "value": rec2["value"],
                                            "snap_scale": round(float(snap_scale), 4)}] + \
            [e for e in h.evidence if e.get("kind") in ("skyline", "sfm-points", "lines3d")]
        rh.value, rh.source, rh.state, rh.map_value = float(eave["roof_height"]), PROV_FULL, S.STATE_CORRECTED, rh_map
        model.adoptions.append(_plain(rec2))
        out.append(model.adoptions[-1])
        p.problems.append(f"D25 {rid2}: roof height {rec2['judged_against']:g} -> {rec2['value']:g} m (eave {eave['value']:g} m)")
    _tickets(model, pid, rec)
    model.built = S.build_all(model.specs, model.params)
    S.resolve_support(model)
    return out


def _tickets(model: S.StructureModel, pid: str, rec: dict[str, Any]) -> None:
    for t in model.tickets:
        if t.get("kind") == "part" and t.get("part") == pid and t.get("open", True):
            if rec.get("adopted") in ("lower bound", "upper bound") or rec.get("kind") == KIND_BLOCK:
                t["d25"] = f"{rec['id']}: at least {rec['value']:g} m (photo value {rec.get('V', rec['value'])} m)" \
                    if rec.get("direction", "up") == "up" else f"{rec['id']}: at most {rec['value']:g} m"
                t["d25_record"] = rec["id"]
            else:
                t.update(open=False, resolved_by=rec["id"], resolution=rec["provenance"], adopted_value=rec["value"])
                rec["ticket"] = t["id"]


def apply_block(model: S.StructureModel, pid: str, res: dict[str, Any], readings: dict[str, Any],
                box: dict[str, Any], scale: dict[str, Any], *, round_: int = 0, snap_scale: float = 1.0,
                depends_on: list[str] | None = None) -> dict[str, Any]:
    """Write one D25-B block: the stub's flat top at H, photo-corrected (auto, lower bound), the OSM footprint, no
    roof shape; the map's 3 m placeholder kept as its map value."""
    p = model.part(pid)
    h = p.params["height"]
    rid = _next_id(model)
    old = float(h.value)
    before = [_snapshot(pid, "height", h)]
    geo = _part_geometry(p)
    rec = {"id": rid, "kind": KIND_BLOCK, "part": pid, "name": p.name, "param": "height",
           "map_value": {"value": round(old, 3), "source": h.source}, "judged_against": round(old, 3),
           "value": round(float(res["value"]), 3), "H": res["H"], "readings": _plain(readings),
           "box_minima": res.get("box_minima"), "carve_p25": res.get("carve_p25"), "supported_by": res.get("supported_by"),
           "geometry": geo, "registration_box": _plain(box), "scale": _plain(scale), "round": int(round_), "rule": RULE,
           "provenance": PROV_LB, "label": LABEL, "adopted": "lower bound", "direction": "up", "active": True,
           "reversible": True, "osm_edited": False, "snap_scale": round(float(snap_scale), 4),
           "before_roof_shape": p.roof_shape, "depends_on": _depends(model, p, depends_on)}
    ev = [{"kind": {"lines": LINES, "sfm": SFM}.get(c, c), "value": r.get("value"), "rule": RULE,
           "n_photos": r.get("n_photos") or r.get("photos") or 0, "points": r.get("points"),
           "snap_scale": round(float(snap_scale), 4)} for c, r in readings.items() if r and r.get("valid")]
    h.evidence = ev + [{"kind": "correction", "id": rid, "rule": RULE, "map_value": round(old, 3), "map_source": h.source,
                        "value": rec["value"], "adopted": "lower bound", "snap_scale": round(float(snap_scale), 4)}]
    h.value, h.source, h.state, h.map_value = float(res["value"]), PROV_LB, S.STATE_CORRECTED, old
    rh = p.params["roof_height"]
    if rh.value != 0.0:
        before.append(_snapshot(pid, "roof_height", rh))
        rh.value = 0.0
    rec["before"] = before
    p.roof_shape = "flat"
    model.adoptions.append(_plain(rec))
    p.problems.append(f"D25 {rid}: a photo-measured block, at least {res['value']:g} m (roof shape unknown, flat)")
    _tickets(model, pid, rec)
    model.built = S.build_all(model.specs, model.params)
    S.resolve_support(model)
    return model.adoptions[-1]


def apply_tilt(model: S.StructureModel, res: dict[str, Any], channels: dict[str, Any], gravity: dict[str, Any],
               *, reconstruction: str = "", consensus: dict[str, Any] | None = None, registration: dict[str, Any] | None = None,
               depends_on: list[str] | None = None) -> dict[str, Any] | None:
    """Write one D25-T tilt: the whole stack rotated rigidly about the main outline's centroid at its base. A stack is
    tilted at most once (a second tilt is refused: None)."""
    if _active(model, (KIND_TILT,)) or model.pose:
        return None
    ring = S.ccw(model.main_rings[0])
    c = st.polygon_centroid(ring)
    z0 = min((b.z0 for b in model.built), default=0.0)
    pose = pose_from_tilt(res["T_en_deg"], (c[0], c[1], z0))
    rid = _next_id(model)
    frames = [a["id"] for a in model.adoptions if a.get("kind") == "frame" and a.get("active", True)]
    rec = {"id": rid, "kind": KIND_TILT, "part": "stack", "param": "tilt", "map_value": {"value": 0.0, "source": "s3db-upright"},
           "value": pose, "result": _plain(res), "channels": _plain(channels), "gravity": _plain(gravity),
           "reconstruction": reconstruction, "consensus": _plain(consensus or {}), "registration": _plain(registration or {}),
           "rule": RULE, "provenance": PROV_FULL, "label": LABEL, "active": True, "reversible": True, "osm_edited": False,
           "before_pose": copy.deepcopy(model.pose), "depends_on": sorted(set((depends_on or []) + frames)),
           "snap_scale": 1.0}
    model.pose = dict(pose, adoption=rid)
    model.adoptions.append(_plain(rec))
    return model.adoptions[-1]


# ---------------------------------------------------------------- the publish gate (G4 / G9 for D25)


def _pre_from_record(rec: dict[str, Any]) -> dict[str, Any]:
    g = rec.get("geometry") or {}
    part = {"id": rec.get("part"), "map_top": float(rec["judged_against"]), "z0": float(g.get("z0", 0.0)),
            "roof_h": float(g.get("roof_h", 0.0)), "pitched": bool(g.get("pitched")), "min_width": float(g.get("min_width", 0.0))}
    return rule_h_pre(part, rec.get("channels") or {}), part


def check_record(rec: dict[str, Any]) -> str | None:
    """A D25 record's rule re-run on its own numbers (never trusting the code that wrote it)."""
    kind = rec.get("kind")
    if kind not in KINDS:
        return f"not a D25 record ({kind!r})"
    if rec.get("rule") != RULE or rec.get("osm_edited") is not False or not rec.get("reversible"):
        return "a D25 record without its rule version, reversible flag or osm_edited: false"
    if kind == KIND_H:
        pre, part = _pre_from_record(rec)
        if not pre.get("candidate"):
            return f"D25-H triggers fail on the record's channels: {pre.get('why')}"
        if sorted(pre["agreeing"]) != sorted(rec.get("agreeing") or []) or abs(float(pre["V"]) - float(rec["V"])) > EPS:
            return f"the agreeing set / V are not the channels' ({pre['agreeing']}, {pre['V']})"
        if pre["direction"] != rec.get("direction"):
            return "the direction is not the channels'"
        why = check_scale(rec.get("scale"))
        if why:
            return why
        fin = rule_h_box(pre, rec.get("registration_box") or {}, rec.get("scale") or {}, part)
        if not fin.get("fires"):
            return f"D25-H box / value: {fin.get('why')}"
        if abs(float(fin["LB"]) - float(rec["LB"])) > EPS or abs(float(fin["UB"]) - float(rec["UB"])) > EPS:
            return f"LB / UB are not the box samples' ({fin['LB']}, {fin['UB']})"
        if fin["adopted"] != rec.get("adopted"):
            return f"the midpoint rule says {fin['adopted']!r}, the record {rec.get('adopted')!r}"
        if abs(float(fin["value"]) - float(rec["value"])) > EPS:
            return f"the adopted value is not {fin['value']}"
        if fin["provenance"] != rec.get("provenance"):
            return f"provenance {rec.get('provenance')!r} does not match what was adopted ({fin['adopted']})"
        return None
    if kind == KIND_ROOF:
        ev = rec.get("eave") or {}
        if not ev.get("adopt") or rec.get("provenance") != PROV_FULL:
            return "a roof-height correction without an adopted eave"
        if abs(float(rec["value"]) - (float(rec["top_value"]) - float(ev["value"]))) > EPS:
            return "the roof height is not the top minus the eave"
        if float(rec["value"]) < ROOF_MIN_M:
            return "a roof under 0.5 m"
        rd = ev.get("readings") or {}
        agree, E = largest_agreeing({k: float(v) for k, v in rd.items()}, eave_agree_tol)
        if len(agree) < 2 or abs(E - float(ev["value"])) > EPS:
            return "the eave is not >= 2 agreeing readings' mean"
        bb = ev.get("box") or {}
        if "LB" not in bb:
            return "the eave has no box bounds"
        if (rec.get("geometry") or {}).get("roof_shape") in ("flat",):
            return "a roof height on a flat roof"
        return None
    if kind == KIND_BLOCK:
        g = rec.get("geometry") or {}
        if rec.get("provenance") != PROV_LB or rec.get("adopted") != "lower bound":
            return "a block that is not a lower bound"
        part = {"id": rec.get("part"), "min_width": float(g.get("min_width", 0.0)), "z0": float(g.get("z0", 0.0)),
                "map_top": float((rec.get("map_value") or {}).get("value", BLOCK_STUB_M))}
        why = check_scale(rec.get("scale"))
        if why:
            return why
        res = rule_b(part, rec.get("readings") or {}, rec.get("carve_p25"), rec.get("registration_box"), rec.get("scale"))
        if not res.get("fires"):
            return f"D25-B: {res.get('why')}"
        if abs(float(res["value"]) - float(rec["value"])) > EPS:
            return f"the block height is not the 2nd-highest box minimum {res['value']}"
        if (rec.get("map_value") or {}).get("source") != st.PROV_DEFAULT:
            return "a block on a part whose height was not a 'height pending' stub"
        return None
    if kind == KIND_TILT:
        res = rec.get("result") or {}
        again = rule_t(rec.get("channels") or {}, rec.get("gravity") or {}, float(res.get("sigma_yaw_deg", 0.0)),
                       res.get("shape") or {})
        if not again.get("fires"):
            return f"D25-T: {again.get('why')}"
        if np.linalg.norm(np.asarray(again["T_en_deg"]) - np.asarray((rec.get("value") or {}).get("lean_en_deg", [0, 0]))) > 1e-3:
            return "the pose is not the channels' median lean"
        if not (rec.get("consensus") or {}).get("agree", True):
            return "passing reconstructions disagree"
        if rec.get("provenance") != PROV_FULL:
            return "a tilt without provenance photo-corrected (auto)"
        return None
    return None


def check_model(model: S.StructureModel) -> list[dict[str, Any]]:
    """Every photo-corrected parameter is backed by an active D25 record that passes check_record and matches it; the
    model's own geometry agrees with the record (not slender, a block flat on its OSM footprint); a tilt only on a
    stack that passes the shape gate on the model's own outline, at most one, the pose equal to its record."""
    bad: list[dict[str, Any]] = []
    act = {a["id"]: a for a in _active(model)}
    for a in act.values():
        why = check_record(a)
        if why:
            bad.append({"id": a["id"], "why": why})
    for p in model.specs:
        for k, v in p.params.items():
            corrected = v.source in PROVS or v.state == S.STATE_CORRECTED
            if not corrected:
                continue
            if v.source not in PROVS or v.state != S.STATE_CORRECTED:
                bad.append({"id": p.id, "param": k, "why": f"source {v.source!r} and state {v.state!r} do not match"})
                continue
            if v.map_value is None:
                bad.append({"id": p.id, "param": k, "why": "photo-corrected without the map value kept"})
            cor = [e for e in v.evidence if e.get("kind") == "correction"]
            car = [e for e in v.evidence if e.get("kind") == "carried" and e.get("rule") == RULE]
            if cor:
                rec = act.get(cor[-1].get("id"))
                if rec is None:
                    bad.append({"id": p.id, "param": k, "why": "photo-corrected by a record that is not active"})
                    continue
                if rec.get("part") != p.id or rec.get("param") != k or abs(float(rec["value"]) - float(v.value)) > 0.01:
                    bad.append({"id": p.id, "param": k, "why": f"the value does not match its record {rec['id']}"})
                if rec.get("provenance") != v.source:
                    bad.append({"id": p.id, "param": k, "why": "provenance and record disagree"})
                if rec["kind"] == KIND_H and is_slender(float(rec["judged_against"]) - p.z0, p.min_width()):
                    bad.append({"id": p.id, "param": k, "why": "a slender part corrected"})
                if rec["kind"] == KIND_BLOCK:
                    if p.roof_shape != "flat" or float(p.params["roof_height"].value) > 0.0:
                        bad.append({"id": p.id, "param": k, "why": "a photo-measured block with a roof shape"})
                    if is_slender(float(v.value) - p.z0, p.min_width()):
                        bad.append({"id": p.id, "param": k, "why": "a slender block"})
                if rec["kind"] in (KIND_H, KIND_BLOCK) and abs(float((rec.get("geometry") or {}).get("min_width", -1))
                                                              - p.min_width()) > 0.05:
                    bad.append({"id": p.id, "param": k, "why": "the record's part geometry is not the model's"})
            elif car:
                if abs(float(v.value) - float(v.map_value) - sum(float(e["dz"]) for e in car
                       if e.get("rule") == RULE) - _other_carry(v)) > 0.01:
                    bad.append({"id": p.id, "param": k, "why": "carried, but its value is not the map's plus the changes"})
                for e in car:
                    a = act.get(e.get("adoption"))
                    if a is None or a.get("part") != e.get("parent"):
                        bad.append({"id": p.id, "param": k, "why": f"carried by {e.get('adoption')}, which is not active"})
            else:
                bad.append({"id": p.id, "param": k, "why": "photo-corrected without a correction or carry record"})
    tilts = _active(model, (KIND_TILT,))
    if len(tilts) > 1:
        bad.append({"id": "stack", "why": "more than one active tilt"})
    if model.pose and not tilts:
        bad.append({"id": "stack", "why": "a pose without an active tilt record"})
    for a in tilts:
        if not model.pose or model.pose.get("adoption") != a["id"] or any(
                abs(float(model.pose.get(k, 0)) - float((a.get("value") or {}).get(k, 1e9))) > 1e-6
                for k in ("tilt_deg", "azimuth_deg")):
            bad.append({"id": a["id"], "why": "the model's pose is not its tilt record's"})
        top = max((b.top for b in model.built), default=0.0) - min((b.z0 for b in model.built), default=0.0)
        sh = tilt_shape_gate(S.ccw(model.main_rings[0]), top)
        if not sh["passed"]:
            bad.append({"id": a["id"], "why": f"a tilt on a stack that fails the shape gate on its own outline ({sh})"})
    for a in _active(model, (KIND_H, KIND_ROOF, KIND_BLOCK)):
        try:
            v = model.part(a["part"]).params[a["param"]]
        except (StopIteration, KeyError):
            bad.append({"id": a["id"], "why": "a correction of a part that does not exist"})
            continue
        if v.state != S.STATE_CORRECTED or v.source != a.get("provenance") or abs(float(v.value) - float(a["value"])) > 0.01:
            bad.append({"id": a["id"], "why": f"record {a['id']} does not match {a['part']}.{a['param']}"})
    return bad


def _other_carry(v: S.Param) -> float:
    """Carries by D15 adoptions on the same parameter (their own records check them)."""
    return sum(float(e["dz"]) for e in v.evidence if e.get("kind") == "carried" and e.get("rule") != RULE)


def records_to_keep(model: S.StructureModel, grade: str | None) -> set[str]:
    """The active D25 corrections whose registration grade is still available when the model is published: a record
    sized for the snap-accepted grade needs the snap accepted now; a crowd-grade one needs the model georegistered
    (grade 'crowd' or better). grade None (prior-only, no points): nothing is kept."""
    if grade not in (GRADE_SNAP, GRADE_CROWD):
        return set()
    ok = {GRADE_SNAP: {GRADE_SNAP, GRADE_CROWD}, GRADE_CROWD: {GRADE_CROWD}}[grade]
    out = set()
    for a in _active(model):
        g = (a.get("registration_box") or a.get("registration") or {}).get("grade")
        if a["kind"] == KIND_ROOF:
            h = next((x for x in model.adoptions if x.get("id") == a.get("height_record")), None)
            g = ((h or {}).get("registration_box") or {}).get("grade")
        if g in ok:
            out.add(a["id"])
    return out


# ---------------------------------------------------------------- the report for human mappers (never uploaded)


def _osm_element(pid: str) -> str:
    base = pid.split("#")[0]
    if base.startswith("w") and base[1:].isdigit():
        return f"way/{base[1:]}"
    if base.startswith("r") and base[1:].isdigit():
        return f"relation/{base[1:]}"
    return "main outline: an area without building:part (a new building:part would be needed)"


def osm_fix_suggestions(model: S.StructureModel) -> dict[str, Any]:
    """Suggested OSM fixes from the active D25 corrections, for human mappers to check: element, tag, current and
    suggested value ('>=' for a bound), the evidence summary and the correction id. Never uploaded; OSM itself is
    never edited. Whether numbers measured from Commons photos may be entered into OSM (ODbL) is the mapper's call."""
    rows = []
    for a in _active(model):
        if a["kind"] == KIND_TILT:
            v = a.get("value") or {}
            rows.append({"element": "the stack (every building:part of the landmark)", "tag": None, "current": "upright",
                         "suggested": None, "note": f"photos measure a tilt of {v.get('tilt_deg')} deg toward "
                                                    f"{v.get('azimuth_deg')} deg; OSM has no standard tilt tag (a note only)",
                         "correction": a["id"]})
            continue
        bound = a.get("adopted") in ("lower bound", "upper bound")
        sign = ">=" if a.get("adopted") == "lower bound" else "<=" if a.get("adopted") == "upper bound" else "~"
        tag = {"height": "height", "roof_height": "roof:height"}.get(a["param"], a["param"])
        ev = {"channels": a.get("agreeing") or a.get("supported_by"), "V": a.get("V"), "box": [a.get("LB"), a.get("UB")],
              "registration": (a.get("registration_box") or {}).get("grade"),
              "scale": (a.get("scale") or {}).get("route")}
        rows.append({"element": _osm_element(a["part"]), "part": a["part"], "name": a.get("name"), "tag": tag,
                     "current": (a.get("map_value") or {}).get("value"), "suggested": f"{sign} {a['value']:g}",
                     "bound": bound, "evidence": _plain(ev), "correction": a["id"]})
    return {"landmark": model.landmark, "rule": RULE, "never_uploaded": True, "osm_edited": False,
            "note": "Suggestions for human OpenStreetMap volunteers to verify; nothing here is uploaded. Whether values "
                    "measured from Wikimedia Commons photos may be entered into OSM (ODbL compatibility) is for the "
                    "mapper to judge [to be verified]. No photos are included.",
            "suggestions": rows}
