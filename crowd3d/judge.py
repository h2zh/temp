"""Judging structure parts against the photos (docs/STRUCTURE_FIRST.md §2.2, §2.4.2-§2.4.3, §2.8, §6 D15).

Photos never shape a part here: they check the map, and -- decision D15 (product owner, 2026-10-23) -- a value
changes AUTOMATICALLY only when at least two photo channels agree on it. Everything is judged AT THE MAP VALUE.

- frame_gate(): the per-edge residual gate (G11). Every outline edge >= 5 m long with >= 200 wall points (2-20 m
  high, nearest edge within 10 m, points far inside counted too) must have a median distance <= 1.5 m. When it
  fails, a frame ticket opens and no parameter is verified or fitted (the evidence is still recorded).
- frame_similarity() / frame_similarity_lines(): one similarity for the whole building (outline scale k about its
  centre, shift t) per channel: the SfM wall planes (least squares on the edges' median wall positions) and the 3-D
  lines on the walls (crowd3d.lines3d: horizontal lines parallel to an edge, their weighted median offset). Single-
  sided data cannot separate k from t, so each needs walls facing >= 180 deg of directions including two opposite ones.
- sfm_channel(): the SfM-point channel: the 0.98 quantile of point heights over the part's exposed area (minus the
  parts standing on it and 1.5 m around taller neighbours) against the model's own surface over the same area.
  Points can only say "the map is too low" (a roof nobody photographed has no points; trees and neighbours sit
  above parts too, so it never decides alone).
- decide(): the rules of §2.4.3 per part, from one or two sky masks' skyline profiles (crowd3d.skyline), the
  SfM channel and the 3-D line channel (crowd3d.lines3d); tolerance TOL = max(2 m, 10 % of the map top); each
  channel votes only for the direction it can observe (skyline: down, and a weak "up?"; SfM points: up; 3-D lines
  measure eave / apex / ridge heights directly: up and down); a skyline alone never raises anything ("up?" counts
  only with another "up"); slender parts (own height / narrowest width >= 2 and width <= 10 m) never move;
  verification needs >= 2 CHANNELS agreeing: the skyline (two masks agreeing, a stable profile
  off its bounds) and a 3-D line measurement of the same quantity (eave <-> 'eave', apex / ridge <-> 'roof'), within
  max(1 m, 5 %) of each other and the lines within TOL of the map; plus the frame gate passed and a snap scale
  correction <= 3 %. "Off the map value" is judged on the profile's sub-grid minimum with half a grid step of
  margin, and nothing is verified while any channel votes against. States: map, photo-verified, disputed (ticket),
  photo-fitted (D15). One strong single "down" (skyline, every mask) or a single 3-D line vote on a part that is not
  slender opens a low-strength ticket. Under a snap scale correction > 3 % a 3-D line vote cannot decide alone.
- D15, "agree" (adoption_rule(); design §6 D15): a disputed part's top is adopted when
  ① >= 2 channels (skyline, SfM points, 3-D lines) each see the part in >= 3 photos and each give a value
     (channel_tops(): every channel's own conditions; SfM points never give a value below the map, the skyline gives
     one below the map only with sky above it in >= 3 photos);
  ② ALL the values agree pairwise within max(1 m, 5 %) and lie on the dispute's side of the map value (a channel on
     the other side, or a vote the other way, is a contradiction: ticket);
  ③ none is at a fit bound (skyline profile bound, a 3-D line window's edge) and the verification bars hold (two
     masks within 1.5 m, 90 % interval <= 1.5 m, both costs within a step);
  ④ unless the common scale is ESTABLISHED (scale_status(): no snap rescale > 3 % AND a channel's whole-building
     similarity measured all round with k within 3 % of 1 on the frame being judged; a snap returning x1.00 on
     one-sided data establishes nothing): >= 2 of the agreeing channels also measure the main body's eave (the
     reference) and their scale-free ratios (top / main-body eave) agree within 5 %; the value adopted is the mean
     ratio x the map's main-body eave, still off the map by more than TOL (SfM points measure no eave: they cannot
     vouch for the scale, so SfM points with one other channel are not enough; the main body itself is the
     reference and is not adopted this way);
  ⑤ parents first: a part whose parent is adopted is carried with it (D16 ②) and re-judged on the parent's new
     value (judge_landmark()); a part whose parent's change is not accepted is not adopted; not slender; no open
     frame ticket (frame_in_question(): the gate failed or a channel's similarity opens one); a stub (D13) may get a
     measured height this way (appearance still only where observed); a stub resting on an adopted part is carried
     but stays a stub;
  ⑥ the frame (outline position and size) by the same rule (decide_frame(): >= 2 channels' similarities agreeing on
     width within max(1 m, 5 %) and on the shift within max(1 m, 5 %), off the map's frame on the same side, not at
     the fit's bounds, and the gate passing on the adopted outline); an outline scale off by > 3 % also needs ④'s
     scale-free check: the main body's eave measured on the trial outline by >= 2 channels (skyline, 3-D lines)
     agrees with the map's eave within max(1 m, 5 %) -- the reconstruction's scale is then the map's -- and outline
     width / eave height still differs from the map's by > 3 % on the same side (frame_scale_check()). The frame's
     two channels are not independent: the 3-D lines are kept only where SfM points support them.
  ⑦ decision D27 (product owner, 2026-10-28): a change of HEIGHT -- the top, the roof height or eave through
     eave_rule(), a stub's measured height; up or down -- also needs a reading of the part's OWN roof among the
     agreeing evidence (own_roof_reading()): the 3-D lines' apex / ridge, or the part's roof-surface fit on the SfM
     points (crowd3d.roof_fit: D25's fit, usable by its fit_usable(), and seen over >= 10 cells 2 m inside the outline)
     within max(1 m, 5 %) of the value adopted. The skyline with the 3-D lines' eave, or with the points' 0.98
     quantile, read the top of whatever stands at the outline: a ticket ("no reading of the part's own roof"). A stub
     has no fit (only the lines' apex / ridge reads it); parts carried with an adopted parent are carried as before;
     the frame (⑥) is not affected.
  What a channel's value may not hide: a dispute "up" is contradicted by SfM points that reach the map's top (within
  TOL) all round the part (the walls are seen ending there); a channel measuring the eave AT the map's eave while
  the top is off means the roof is what changed: roof_height is adopted with the eave kept (eave_rule(); channels'
  eaves disagreeing: a ticket).
  An adoption writes provenance photo-fitted (auto), keeps the old value (map_value) and archives every piece of
  evidence in the reversible list (StructureModel.adoptions: summary + GLB extras; revert_adoption() undoes one and
  what depends on it). One channel or contradicting channels: a ticket only. OpenStreetMap itself is never edited.
- apply_decisions() writes states, evidence and tickets into the structure's parameters (massing.json v2) and,
  with AUTO_ADOPT (on), the adoptions; check_decisions() is G4 / G9: no verified or fitted value without the
  evidence the rules need, rechecked from the records themselves.
"""
from __future__ import annotations

import copy
import math
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import roof_fit as RF
from . import stylize as st
from . import structure as S
from .lines3d import SNAP_SCALE_TOL, _wmedian       # snap scale corrections over 3 % (D15); weighted median

FRAME_EDGE_MIN_M = 5.0
FRAME_EDGE_MIN_POINTS = 200
FRAME_MAX_MEDIAN_M = 1.5
FRAME_K_TOL = 0.03
FRAME_T_TOL_M = 1.5
MIN_PHOTOS = 3
AGREE_M = 1.0                # two channels agree within max(AGREE_M, AGREE_REL x value) (design D15)
AGREE_REL = 0.05
LINES = "lines3d"
SKYLINE = "skyline"
SFM = "sfm-points"
LINES_KEY = {"eave": "eave", "roof": "top"}     # skyline profile parameter -> 3-D line measurement
# D15 (product owner, 2026-10-23): adopt automatically when >= 2 photo channels agree
AUTO_ADOPT = True
ADOPT_MIN_CHANNELS = 2
SCALE_FREE_REL = 0.05        # ④: the channels' scale-free ratios agree within 5 %
CHANNEL_SPREAD_M = 1.5       # ③: a channel's 90 % interval and its two masks' spread (heights), the verification bars
FRAME_K_BOUNDS = (0.7, 1.3)  # ⑥: the similarity fit's range (outside: at a fit bound)
FRAME_T_BOUND_M = 8.5        # ... a shift or an edge offset beyond this is at the 10 m edge-assignment window
PROV_AUTO = st.PROV_FITTED_AUTO
AUTO_LABEL = "照片量出（自动采用）"


def agree_tol(a: float, b: float) -> float:
    return max(AGREE_M, AGREE_REL * max(abs(a), abs(b)))


def agreeing(values: list[tuple[str, float]]) -> tuple[list[str], float | None]:
    """The largest set of >= 2 channels whose values agree pairwise within max(1 m, 5 %) (ties: the tightest), and
    their mean; ([], None) when no two agree."""
    best: tuple[int, float, list[str], float] | None = None
    n = len(values)
    for mask in range(1, 1 << n):
        sub = [values[i] for i in range(n) if mask >> i & 1]
        if len(sub) < 2:
            continue
        v = [x for _, x in sub]
        if all(abs(a - b) <= agree_tol(a, b) for a in v for b in v):
            key = (len(sub), -float(np.ptp(v)))
            if best is None or key > (best[0], -best[1]):
                best = (len(sub), float(np.ptp(v)), [c for c, _ in sub], float(np.mean(v)))
    return ([], None) if best is None else (best[2], round(best[3], 2))


def frame_in_question(frame: dict[str, Any] | None) -> bool:
    """Is the frame in question and unsettled on the outline being judged? The gate (G11) failed, or a channel's
    whole-building similarity opens a frame ticket (k off by > 3 % or a shift > 1.5 m: decide_frame()'s
    in_question). After a frame adoption the frame passed here is the one measured on the adopted outline. While it
    is, nothing is verified or adopted, and apply_decisions() opens the frame ticket (the publish gate rejects a
    verified or fitted value under an open frame ticket)."""
    if not frame:
        return False
    if not frame.get("passed", True):
        return True
    return any(bool((frame.get(k) or {}).get("ticket")) for k in ("similarity", "similarity_lines"))


def scale_status(frame: dict[str, Any] | None, snap_scale: float = 1.0) -> dict[str, Any]:
    """D15 ④: is the reconstruction's common scale established? All channels share one pose set and one scale, so a
    common scale error moves them together. Established when the snap did not rescale by > 3 % AND a channel's
    whole-building similarity is measured (walls facing >= 180 deg with two opposite ones, off its fit bounds) with
    its outline scale k within 3 % of 1 on the frame being judged: the walls then sit on the map's outline all round
    at the map's scale. A snap returning x1.00 says nothing (its refinement cannot separate scale from shift on
    one-sided data). Not established: D15 needs ④'s scale-free check, exactly as under a snap rescale > 3 %."""
    s = float(snap_scale)
    out: dict[str, Any] = {"snap_scale": round(s, 4), "snap_rescaled": bool(abs(s - 1.0) > SNAP_SCALE_TOL),
                           "established": False, "by": None}
    if out["snap_rescaled"]:
        out["why"] = f"snap scale corrected x{s:.3f} (> {SNAP_SCALE_TOL:.0%})"
        return out
    seen = []
    for key, name in (("similarity", SFM), ("similarity_lines", LINES)):
        sm = (frame or {}).get(key) or {}
        if not sm:
            continue
        if sm.get("measured") and not sm.get("at_bound") and sm.get("k") is not None:
            k = float(sm["k"])
            if abs(k - 1.0) <= FRAME_K_TOL:
                out.update(established=True, by={"channel": name, "measured": True, "at_bound": False, "k": round(k, 4)},
                           why=f"{name}'s whole-building similarity (walls all round): k {k:.3f}, within 3 % of 1")
                return out
            seen.append(f"{name}: k {k:.3f}, off by > 3 %")
        else:
            seen.append(f"{name}: {'at a fit bound' if sm.get('at_bound') else 'not measured (walls do not face >= 180 deg)'}")
    out["why"] = ("the common scale is not established: no whole-building similarity measured within 3 % of k = 1 ("
                  + ("; ".join(seen) or "no similarity measured") + ")")
    return out


def _scale_established(sc: dict[str, Any] | None) -> bool:
    """scale_status()'s verdict rechecked from its record (the publish gate)."""
    sc = sc or {}
    by = sc.get("by") or {}
    return bool(sc.get("established") and abs(float(sc.get("snap_scale", 1.0)) - 1.0) <= SNAP_SCALE_TOL
                and by.get("measured") and not by.get("at_bound") and by.get("k") is not None
                and abs(float(by["k"]) - 1.0) <= FRAME_K_TOL)


def _pairwise_disagreement(values: list[tuple[str, float]], tol=agree_tol) -> str | None:
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            (ca, va), (cb, vb) = values[i], values[j]
            if abs(va - vb) > tol(va, vb):
                return f"{ca} {va:.2f} vs {cb} {vb:.2f} (> {tol(va, vb):.2f})"
    return None


def _plain(o: Any) -> Any:
    """JSON-safe copy (numpy scalars and arrays to Python numbers and lists)."""
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


def lines_records(lines: dict[str, list[dict[str, Any]]] | None, pid: str, snap_scale: float = 1.0) -> list[dict[str, Any]]:
    """The part's 3-D line evidence records, stamped with the judge's snap scale (scale_corrected when > 3 %)."""
    out = []
    for r in (lines or {}).get(pid, []):
        if r.get("kind") != LINES:
            continue
        out.append({**r, "snap_scale": round(float(snap_scale), 4),
                    "scale_corrected": bool(abs(snap_scale - 1.0) > SNAP_SCALE_TOL)})
    return out


def lines_height_vote(recs: list[dict[str, Any]]) -> tuple[str | None, float | None, str | None]:
    """The 3-D lines' say about the part's height: the apex / ridge measurement when there is one, else the eave's
    (with the map's roof on it; a flat roof's eave is its top). Returns (vote, implied top, parameter)."""
    by = {r.get("parameter"): r for r in recs if r.get("value") is not None}
    for key in ("top", "eave"):
        r = by.get(key)
        if r is not None:
            return r.get("vote"), r.get("implied_top", r.get("value")), key
    return None, None, None


def _wall_points(points: np.ndarray, quality: np.ndarray | None, z_lo: float = 2.0, z_hi: float = 20.0,
                 min_track: int = 3) -> np.ndarray:
    p = np.asarray(points, float)
    sel = (p[:, 2] > z_lo) & (p[:, 2] < z_hi)
    if quality is not None and len(quality) == len(p):
        sel &= np.asarray(quality)[:, 0] >= min_track
    return p[sel, :2]


def _edge_assignment(xy: np.ndarray, ring: np.ndarray, max_d: float = 10.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Nearest outline edge of each point (within max_d), its distance, and the signed offset (outward +)."""
    n = len(ring)
    d = np.full(len(xy), np.inf)
    sgn = np.zeros(len(xy))
    k = np.zeros(len(xy), int)
    for e in range(n):                                   # running minimum: O(points) memory, first edge wins ties
        a, b = ring[e], ring[(e + 1) % n]
        t = b - a
        L = float(np.linalg.norm(t))
        if L < 1e-9:
            continue
        u = t / L
        s = np.clip((xy - a) @ u, 0, L)
        de = np.linalg.norm(xy - (a + s[:, None] * u), axis=1)
        better = de < d
        d[better], k[better] = de[better], e
        sgn[better] = ((xy[better] - a) @ np.array([u[1], -u[0]]))
    k[d > max_d] = -1
    return k, d, sgn


def frame_gate(points: np.ndarray, rings: list[np.ndarray], quality: np.ndarray | None = None) -> dict[str, Any]:
    """G11, the per-edge residual gate (design §2.2 step 1)."""
    ring = S.ccw(rings[0])
    xy = _wall_points(points, quality)
    k, d, _ = _edge_assignment(xy, ring)
    edges, failing = [], []
    for e in range(len(ring)):
        m = k == e
        L = float(np.linalg.norm(ring[(e + 1) % len(ring)] - ring[e]))
        if m.sum() < FRAME_EDGE_MIN_POINTS:
            continue
        row = {"edge": e, "len_m": round(L, 2), "points": int(m.sum()), "median_m": round(float(np.median(d[m])), 2)}
        edges.append(row)
        if L >= FRAME_EDGE_MIN_M and row["median_m"] > FRAME_MAX_MEDIAN_M:
            failing.append(row)
    return {"passed": not failing, "edges": edges, "failing": failing,
            "rule": f"edges >= {FRAME_EDGE_MIN_M:g} m with >= {FRAME_EDGE_MIN_POINTS} wall points: median <= {FRAME_MAX_MEDIAN_M:g} m"}


def _similarity_fit(rows: list[dict[str, Any]], c: np.ndarray, channel: str) -> dict[str, Any]:
    """Least squares for position_i = k * d_i + n_i . t over the edges' rows (d_i the edge's distance from the
    outline's centre c, n_i its outward normal, position_i the measured wall's distance); 'measured' needs walls
    facing >= 180 deg of directions with two opposite ones. Also D15's bounds: k outside FRAME_K_BOUNDS, a shift or
    an edge offset beyond FRAME_T_BOUND_M (the 10 m edge-assignment window) is at a fit bound."""
    out: dict[str, Any] = {"channel": channel, "edges": [r["row"] for r in rows],
                           "centre": [round(float(v), 3) for v in c], "measured": False, "at_bound": False}
    if len(rows) < 3:
        out["reason"] = f"{len(rows)} edges with wall evidence (< 3)"
        return out
    A = np.array([[r["di"], r["n"][0], r["n"][1]] for r in rows])
    y = np.array([r["pos"] for r in rows])
    w = np.array([r["w"] for r in rows])
    normals = [r["normal_deg"] for r in rows]
    sol, *_ = np.linalg.lstsq(A * w[:, None], y * w, rcond=None)
    kk, tx, ty = (float(v) for v in sol)
    res = y - A @ sol
    ang = np.sort(np.array(normals))
    gaps = np.diff(np.r_[ang, ang[0] + 360])
    opposite = any(abs(((a1 - a2) % 360) - 180) <= 30 for a1 in normals for a2 in normals)
    covered = bool(gaps.max() <= 181.0 and opposite)          # walls facing >= 180 deg (1 deg slack)
    out.update({"k": round(kk, 4), "t": [round(tx, 3), round(ty, 3)], "residual_max_m": round(float(np.abs(res).max()), 3),
                "coverage_ok": covered, "directions_deg": [round(float(v), 1) for v in ang],
                "measured": covered})
    out["ticket"] = bool(covered and (abs(kk - 1) > FRAME_K_TOL or math.hypot(tx, ty) > FRAME_T_TOL_M))
    bound = []
    if not (FRAME_K_BOUNDS[0] < kk < FRAME_K_BOUNDS[1]):
        bound.append(f"k {kk:.3f} outside {FRAME_K_BOUNDS}")
    if math.hypot(tx, ty) > FRAME_T_BOUND_M:
        bound.append(f"shift {math.hypot(tx, ty):.2f} m > {FRAME_T_BOUND_M:g} m")
    far = [r["row"]["edge"] for r in rows if abs(r["row"]["offset_m"]) > FRAME_T_BOUND_M]
    if far:
        bound.append(f"edges {far} offset beyond {FRAME_T_BOUND_M:g} m (the 10 m window)")
    out["at_bound"] = bool(bound)
    if bound:
        out["bound_why"] = "; ".join(bound)
    return out


def frame_similarity(points: np.ndarray, rings: list[np.ndarray], quality: np.ndarray | None = None,
                     min_points: int = FRAME_EDGE_MIN_POINTS) -> dict[str, Any]:
    """Whole-building similarity from the wall planes (design §2.2 step 2, the wall-plane method): per edge with
    enough wall points (3-16 m high), the median wall position along its outward normal n_i; least squares for
    position_i = k * d_i + n_i . t, d_i the edge's distance from the outline's centroid. 'measured' needs walls
    facing >= 180 deg of directions with two opposite ones (else scale and shift cannot be separated). photos_min: 3
    when the points are filtered to track length >= 3 (D15 ①), None when their quality is unknown."""
    ring = S.ccw(rings[0])
    c = st.polygon_centroid(ring)
    xy = _wall_points(points, quality, 3.0, 16.0)
    k, d, sgn = _edge_assignment(xy, ring)
    rows = []
    for e in range(len(ring)):
        a, b = ring[e], ring[(e + 1) % len(ring)]
        L = float(np.linalg.norm(b - a))
        m = k == e
        if L < 3.0 or m.sum() < min_points:
            continue
        u = (b - a) / L
        s = (xy[m] - a) @ u
        mid = (s > 0.1 * L) & (s < 0.9 * L)
        if mid.sum() < min_points // 2:
            continue
        n = np.array([u[1], -u[0]])
        di = float(n @ (a - c))
        pos = di + float(np.median(sgn[m][mid]))
        rows.append({"di": di, "n": n, "pos": pos, "w": math.sqrt(mid.sum()),
                     "normal_deg": math.degrees(math.atan2(n[1], n[0])) % 360,
                     "row": {"edge": e, "points": int(mid.sum()), "offset_m": round(pos - di, 2)}})
    out = _similarity_fit(rows, c, SFM)
    out["photos_min"] = MIN_PHOTOS if quality is not None and len(quality) == len(points) else None
    return out


def frame_similarity_lines(lines: Any, rings: list[np.ndarray], z_band: tuple[float, float] = (3.0, 16.0),
                           max_in_m: float = 10.0, max_out_m: float = 3.0, parallel_deg: float = 10.0,
                           min_lines: int = 3) -> dict[str, Any]:
    """The same whole-building similarity from the 3-D lines (crowd3d.lines3d.Lines3D, each verified in >= 6 photos and
    supported by SfM points): per outline edge, the horizontal lines (slope < 5 deg, >= 1 m) in the wall band z_band
    parallel to it (within parallel_deg), within max_in_m inside and max_out_m outside, overlapping its middle 80 %;
    their overlap-weighted median offset is the wall's position (>= min_lines lines from >= 3 photos per edge). String
    courses, cornices and the eave all lie in the wall plane, so they all count."""
    ring = S.ccw(rings[0])
    c = st.polygon_centroid(ring)
    A, B = np.asarray(lines.A, float), np.asarray(lines.B, float)
    rows: list[dict[str, Any]] = []
    if len(A):
        D = B - A
        Ln = np.linalg.norm(D, axis=1)
        slope = np.degrees(np.arcsin(np.clip(np.abs(D[:, 2]) / np.maximum(Ln, 1e-9), 0, 1)))
        zmid = 0.5 * (A[:, 2] + B[:, 2])
        sel = np.nonzero((slope < 5.0) & (Ln >= 1.0) & (zmid >= z_band[0]) & (zmid <= z_band[1]))[0]
        mid = 0.5 * (A[sel, :2] + B[sel, :2])
        ke, _, _ = _edge_assignment(mid, ring, max_in_m)
        cosp = math.cos(math.radians(parallel_deg))
        for e in range(len(ring)):
            a, b = ring[e], ring[(e + 1) % len(ring)]
            L = float(np.linalg.norm(b - a))
            m = sel[ke == e]
            if L < 3.0 or not len(m):
                continue
            u = (b - a) / L
            n = np.array([u[1], -u[0]])
            par = np.abs(D[m, :2] @ u) / np.maximum(np.linalg.norm(D[m, :2], axis=1), 1e-9) > cosp
            ta, tb = (A[m, :2] - a) @ u, (B[m, :2] - a) @ u
            off = 0.5 * ((A[m, :2] - a) @ n + (B[m, :2] - a) @ n)
            ovl = np.clip(np.minimum(np.maximum(ta, tb), 0.9 * L) - np.maximum(np.minimum(ta, tb), 0.1 * L), 0, None)
            ok = par & (ovl >= 0.5) & (off >= -max_in_m) & (off <= max_out_m)
            if ok.sum() < min_lines:
                continue
            photos = {p for k in m[ok] for p in lines.photos[k]}
            if len(photos) < MIN_PHOTOS:
                continue
            o = _wmedian(off[ok], ovl[ok])
            di = float(n @ (a - c))
            rows.append({"di": di, "n": n, "pos": di + o, "w": math.sqrt(float(ovl[ok].sum())),
                         "normal_deg": math.degrees(math.atan2(n[1], n[0])) % 360,
                         "row": {"edge": e, "lines": int(ok.sum()), "photos": len(photos), "offset_m": round(o, 2)}})
    out = _similarity_fit(rows, c, LINES)
    out["photos_min"] = MIN_PHOTOS          # every edge's lines come from >= 3 photos (each line from >= 6)
    return out


def frame_channels(model: S.StructureModel, points: np.ndarray | None, quality: np.ndarray | None = None,
                   lines: Any = None) -> dict[str, dict[str, Any]]:
    """Each channel's whole-building similarity of the main outline: SfM wall planes and 3-D wall lines (lines: a
    crowd3d.lines3d.LinesResult or Lines3D)."""
    out: dict[str, dict[str, Any]] = {}
    if points is not None and len(points):
        out[SFM] = frame_similarity(points, model.main_rings, quality)
    L = getattr(lines, "lines", lines)
    if L is not None and hasattr(L, "A"):
        out[LINES] = frame_similarity_lines(L, model.main_rings)
    return out


def sfm_channel(model: S.StructureModel, points: np.ndarray, quality: np.ndarray | None = None, cell: float = 0.5,
                min_points: int = 50) -> dict[str, dict[str, Any]]:
    """Per built part: the SfM points' 0.98 height quantile over its exposed area vs the model's own surface there.
    min_track: 3 when the points are filtered to track length >= 3 (each seen in >= 3 photos, D15 ①), else None."""
    P = np.asarray(points, float)
    min_track = None
    if quality is not None and len(quality) == len(P):
        P = P[np.asarray(quality)[:, 0] >= 3]
        min_track = 3
    built = {b.id: b for b in model.built}
    out: dict[str, dict[str, Any]] = {}
    for b in model.built:
        lo, hi = b.outer.min(axis=0), b.outer.max(axis=0)
        g = S._Grid(lo - cell, hi + cell, cell)
        m = g.fill(b.outer, b.holes) > 0
        for o in model.built:
            if o is b:
                continue
            on_me = o.spec.parent == b.id
            taller = o.top > b.top + 1.0
            if not (on_me or taller):
                continue
            om = g.fill(o.outer, o.holes) > 0
            if taller and not on_me:
                r = max(1, int(round(1.5 / cell)))
                om = cv2.dilate(om.astype(np.uint8), np.ones((2 * r + 1, 2 * r + 1), np.uint8)) > 0
            m &= ~om
        row: dict[str, Any] = {"exposed_m2": round(float(m.sum()) * cell * cell, 1), "points": 0, "q98": None,
                               "model_q98": None, "min_track": min_track}
        if m.sum() >= 2:
            sel = (P[:, 0] >= g.lo[0]) & (P[:, 1] >= g.lo[1]) & (P[:, 0] < g.lo[0] + g.nx * cell) & (P[:, 1] < g.lo[1] + g.ny * cell)
            q = P[sel]
            ix = ((q[:, 0] - g.lo[0]) / cell).astype(int)
            iy = ((q[:, 1] - g.lo[1]) / cell).astype(int)
            ok = m[np.clip(iy, 0, g.ny - 1), np.clip(ix, 0, g.nx - 1)] & (q[:, 2] > b.z0)
            row["points"] = int(ok.sum())
            if ok.sum() >= min_points:
                row["q98"] = round(float(np.quantile(q[ok, 2], 0.98)), 2)
            cen = g.centres()[m]
            z = built[b.id].surface_z(cen)
            z = z[np.isfinite(z)]
            if len(z):
                row["model_q98"] = round(float(np.quantile(z, 0.98)), 2)
            if row["model_q98"] is not None and ok.sum():
                # how the points near the model's top spread over the part: the quadrants (about the exposed area's
                # centre) holding >= 5 points within TOL below the model's surface top -- all round, the walls (or
                # the roof) are seen up to there (judge.channel_tops: the points then contradict a higher top)
                tol = max(2.0, 0.1 * float(b.top))
                near = q[ok][q[ok, 2] >= row["model_q98"] - tol]
                c0 = cen.mean(axis=0)
                quad = [int(np.sum((np.sign(near[:, 0] - c0[0]) == sx) & (np.sign(near[:, 1] - c0[1]) == sy)))
                        for sx in (-1, 1) for sy in (-1, 1)]
                row["near_top_points"] = int(len(near))
                row["near_top_quadrants"] = int(sum(1 for n in quad if n >= 5))
        out[b.id] = row
    return out


@dataclass
class Decision:
    id: str
    state: str
    why: str
    votes: dict[str, Any] = field(default_factory=dict)
    verified: dict[str, bool] = field(default_factory=dict)
    params: dict[str, Any] = field(default_factory=dict)
    n_vis: int = 0
    n_sky: int = 0
    slender: bool = False
    would_fit: dict[str, float] | None = None    # {'top': value} when D15 adopts (the adoption record: adopt)
    evidence: list = field(default_factory=list)
    ticket: str | None = None            # 'dispute' (>= 2 channels) or 'low' (one strong single-channel vote)
    skyline_ok: dict[str, bool] = field(default_factory=dict)    # per parameter: the skyline's own conditions
    agreement: dict[str, dict[str, Any]] = field(default_factory=dict)   # per parameter: skyline vs 3-D lines
    measured: dict[str, float] = field(default_factory=dict)     # channel -> the top it measures (tickets)
    agreeing: list = field(default_factory=list)                 # channels agreeing within max(1 m, 5 %)
    agreed: float | None = None                                  # ... and their mean
    direction: str | None = None         # the dispute's direction ('up' / 'down') when >= 2 channels vote one way
    channels: dict[str, dict[str, Any]] = field(default_factory=dict)   # D15: each channel's top and its conditions
    adopt: dict[str, Any] | None = None  # D15: the adoption record when the rule holds (apply_decisions applies it)
    adopt_why: str = ""                  # ... else why not (a ticket only)
    rejudged: int = 0                    # the round of judge_landmark() that judged it on its parent's new value


def _profile_info(b: S.BuiltPart, key: str, p: Any, minw: float) -> dict[str, Any]:
    """One skyline profile parameter of a part: its best value (grid and sub-grid), the map's, the top there,
    bounds, interval, robustness (decide() and D15's channel conditions)."""
    z0 = b.z0
    pp = p.params[key]
    G = pp["grid"]
    k = G.index(pp["fr"]["argmin"])
    k1 = int(np.argmin(np.abs(np.array(G) - 1.0)))
    if key == "scale":
        L = minw / 2
        val, mapv = L * pp["fr"]["argmin"], L
        topv = pp["metres"][k]["top"]
        q = None
    else:
        L = b.roof.height_m if key == "roof" else b.eave - z0
        q = "top" if key == "roof" else "eave"
        val, mapv = pp["metres"][k][q], pp["metres"][k1][q]
        topv = pp["metres"][k]["top"]
    step = G[1] - G[0]
    # the profile's sub-grid minimum (a parabola through the argmin and its neighbours, at most half a step
    # from it): a grid point can sit up to half a step nearer the map than the photos' best value
    curve = np.asarray(pp["fr"].get("curve") or [], float)
    frac = 0.0
    if len(curve) == len(G) and 0 < k < len(G) - 1:
        c0, c1, c2 = curve[k - 1], curve[k], curve[k + 1]
        den = c0 - 2 * c1 + c2
        if den > 1e-9:
            frac = float(np.clip(0.5 * (c0 - c2) / den, -0.5, 0.5))
    if key == "scale":
        step_m = step * L
    else:
        mk = [m[q] for m in pp["metres"]]
        i0, i1 = max(k - 1, 0), min(k + 1, len(mk) - 1)
        step_m = abs(mk[i1] - mk[i0]) / (i1 - i0) if (i1 > i0 and mk[i0] is not None and mk[i1] is not None) \
            else step * L
    val_sub = val + frac * step_m
    top_sub = None if topv is None else (topv if key == "scale" else topv + frac * step_m)
    return {"val": round(float(val), 2), "val_sub": round(float(val_sub), 2), "step_m": round(float(step_m), 3),
            "flat": bool(len(curve) and float(curve.max() - curve.min()) <= 1e-6),
            "map": round(float(mapv), 2), "top": topv, "top_sub": top_sub, "bound": pp["fr"]["at_bound"],
            "lo_bound": pp["fr"]["lo_bound"], "hi_bound": pp["fr"]["hi_bound"],
            "ci90_m": round((pp["fr"]["p95"] - pp["fr"]["p05"]) * L, 2), "argmin": pp["fr"]["argmin"],
            "robust": abs(pp["fr"]["argmin"] - pp["fo"]["argmin"]) <= step + 1e-9, "n_cams": pp["n_cams"]}


def _skyline_profile_problem(pm: dict[str, dict[str, Any]], key: str) -> str | None:
    """D15 ③ and the verification bars for one skyline parameter over its masks: two masks, none at a fit bound,
    each interval <= 1.5 m and both costs within a step."""
    if len(pm) < 2:
        return f"{key}: one sky mask only"
    for m, v in pm.items():
        if v.get("flat"):
            return f"{key} profile flat ({m}): the skyline says nothing about it"
        if v["bound"]:
            return f"{key} profile at a fit bound ({m})"
        if v["ci90_m"] > CHANNEL_SPREAD_M or not v["robust"]:
            return f"{key} profile unstable ({m}: interval {v['ci90_m']} m, costs agree: {v['robust']})"
    return None


def channel_tops(b: S.BuiltPart, params: dict[str, dict[str, dict[str, Any]]], n_masks: int, n_vis: int, n_sky: int,
                 sfm_row: dict[str, Any] | None, lrecs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """What each photo channel measures for the part's top, with its own D15 conditions: {channel: {'value', 'ok',
    'why', 'photos', ...}} (① >= 3 photos and a value; ③ not at a fit bound and the verification bars).
    - skyline: the top at the profiles' sub-grid minimum (the eave and the roof profile, both masks); the part is the
      model's skyline in >= 3 photos; no profile at a bound, intervals <= 1.5 m, both costs within a step, the masks
      within 1.5 m and the two profiles within max(1 m, 5 %) of each other; a value below the map needs sky above the
      part in >= 3 photos (else it cannot say lower).
    - sfm-points: the map top plus how far the points' q98 rises above the model's own surface over the part's exposed
      area (points of >= 3 photos each); never a value at or below the map (a roof nobody photographed has no points).
    - lines3d: the apex / ridge (else the eave with the map's roof on it); >= 3 photos, not at its search window's
      edge, its 90 % interval <= 1.5 m, its own vote not withheld (one line only, edges disagreeing)."""
    top = float(b.top)
    out: dict[str, dict[str, Any]] = {}
    keys = [k for k in ("eave", "roof") if k in params]
    sk: dict[str, Any] = {"ok": False, "photos": int(n_vis), "sky_above_photos": int(n_sky), "masks": int(n_masks)}
    if not keys:
        sk["why"] = "no height profile"
    elif n_masks < 2:
        sk["why"] = "one sky mask only"
    elif n_vis < MIN_PHOTOS:
        sk["why"] = f"the model's skyline in {n_vis} (< {MIN_PHOTOS}) photos"
    else:
        bad, tops_k = None, {}
        for k in keys:
            bad = _skyline_profile_problem(params[k], k)
            if bad is None and any(v["top_sub"] is None for v in params[k].values()):
                bad = f"{k}: no top at the minimum"
            if bad is None:
                ts = [float(v["top_sub"]) for v in params[k].values()]
                if max(ts) - min(ts) > CHANNEL_SPREAD_M:
                    bad = f"{k}: the masks' tops {min(ts):.2f}-{max(ts):.2f} m apart (> {CHANNEL_SPREAD_M:g} m)"
                tops_k[k] = float(np.mean(ts))
            if bad:
                break
        if bad is None and len(tops_k) == 2 and abs(tops_k["eave"] - tops_k["roof"]) > agree_tol(tops_k["eave"], tops_k["roof"]):
            bad = f"the eave and roof profiles disagree on the top ({tops_k['eave']:.2f} vs {tops_k['roof']:.2f} m)"
        if bad:
            sk["why"] = bad
        else:
            val = float(np.mean(list(tops_k.values())))
            sk["value"] = round(val, 2)
            sk["profiles"] = {k: round(v, 2) for k, v in tops_k.items()}
            if val < top and n_sky < MIN_PHOTOS:
                sk["why"] = f"below the map with sky above it in {n_sky} (< {MIN_PHOTOS}) photos: it cannot say lower"
            else:
                sk["ok"] = True
    out[SKYLINE] = sk
    s = sfm_row or {}
    sf: dict[str, Any] = {"ok": False, "points": int(s.get("points") or 0), "min_track": s.get("min_track")}
    if s.get("q98") is None or s.get("model_q98") is None:
        sf["why"] = f"{sf['points']} points over the exposed area (too few)"
    elif (s.get("min_track") or 0) < MIN_PHOTOS:
        sf["why"] = "the points are not known to be seen in >= 3 photos each (no track lengths)"
    else:
        v = top + float(s["q98"]) - float(s["model_q98"])
        sf.update(value=round(v, 2), q98=s["q98"], model_q98=s["model_q98"], photos=MIN_PHOTOS)
        if v <= top:
            sf["why"] = "at or below the map: points cannot say lower (a roof nobody photographed has none)"
            tol = max(2.0, 0.1 * top)
            if v >= top - tol and (s.get("near_top_quadrants") or 0) >= 3 and (s.get("near_top_points") or 0) >= 20:
                # the points (each from >= 3 photos) reach the map's top, all round the part, and no higher: the
                # walls / roof are seen ending at the map's height. No value below the map, but a higher top is
                # contradicted (scaffolding, a tree or a building behind can make the skyline and the lines agree
                # on a top the part does not have)
                sf["ceiling"] = True
                sf["why"] = (f"the points reach {v:.2f} m, the map's top within {tol:.1f} m, all round the part "
                             f"({s.get('near_top_quadrants')} quadrants, {s.get('near_top_points')} points near the top): "
                             f"they cannot say lower, and they contradict a higher top")
        else:
            sf["ok"] = True
    out[SFM] = sf
    by = {r.get("parameter"): r for r in lrecs if r.get("value") is not None and r.get("parameter") in ("top", "eave")}
    lr: dict[str, Any] = {"ok": False}
    rec = by.get("top") or by.get("eave")
    if rec is None:
        lr["why"] = "no 3-D line measurement"
    else:
        v = float(rec.get("implied_top", rec["value"]))
        iv = rec.get("interval") or [v, v]
        lr.update(value=round(v, 2), parameter=rec.get("parameter"), how=rec.get("how", rec.get("parameter")),
                  photos=int(rec.get("n_photos", 0)), lines=int(rec.get("n_lines", 0)), interval=iv,
                  at_bound=bool(rec.get("at_bound")), search_m=rec.get("search_m"))
        if lr["photos"] < MIN_PHOTOS:
            lr["why"] = f"lines from {lr['photos']} (< {MIN_PHOTOS}) photos"
        elif rec.get("at_bound"):
            lr["why"] = f"at the edge of its search window {rec.get('search_m')} (a fit bound)"
        elif float(iv[1]) - float(iv[0]) > CHANNEL_SPREAD_M:
            lr["why"] = f"90 % interval {float(iv[1]) - float(iv[0]):.2f} m (> {CHANNEL_SPREAD_M:g} m)"
        elif rec.get("vote_withheld"):
            lr["why"] = f"its own vote withheld: {rec['vote_withheld']}"
        else:
            lr["ok"] = True
    out[LINES] = lr
    return out


def decide(model: S.StructureModel, profiles: dict[str, dict[str, Any]], sfm: dict[str, dict[str, Any]] | None = None,
           frame: dict[str, Any] | None = None, snap_scale: float = 1.0,
           lines: dict[str, list[dict[str, Any]]] | None = None,
           scale_refs: dict[str, Any] | None = None, points: np.ndarray | None = None) -> dict[str, Decision]:
    """The §2.4.3 rules for every built part with a profile, and D15's adoption rule (adoption_rule(); the record is
    always computed -- apply_decisions() applies it when the switch is on). profiles: {mask name: {part id:
    skyline.PartProfile}} (one or two masks; verification needs two); sfm: judge.sfm_channel(); lines: {part id:
    lines3d evidence records} (crowd3d.lines3d.LinesResult.evidence()). A verification needs two channels agreeing.
    scale_refs: D15 ④'s references (scale_free_refs()) when this call judges only some parts (judge_landmark's later
    rounds). points: the SfM points (track >= 3 when known) for D27's roof-surface fit (own_roof_reading()), fitted
    only for a part the rule would otherwise adopt.
    frame: the frame gate's result with each channel's similarity ('similarity', 'similarity_lines', as
    judge_landmark() builds it): the frame is open when the gate failed OR a similarity opens a frame ticket
    (frame_in_question()), and the common scale is established only by a similarity measured within 3 % of k = 1
    (scale_status(); a frame without one leaves it unestablished: D15 then needs ④)."""
    masks = list(profiles)
    sfm = sfm or {}
    frame_open = frame_in_question(frame)
    scaled = abs(snap_scale - 1.0) > SNAP_SCALE_TOL
    sstat = scale_status(frame, snap_scale)
    out: dict[str, Decision] = {}
    for b in model.built:
        pid = b.id
        profs = [profiles[m][pid] for m in masks if pid in profiles[m]]
        if not profs:
            continue
        top = b.top
        tol = max(2.0, 0.1 * top)
        minw = max(b.spec.min_width(), 0.1)
        hw = (top - b.z0) / minw
        slender = hw >= 2.0 and minw <= 10.0
        n_vis = min(p.n_vis for p in profs)
        n_sky = min(p.n_sky for p in profs)
        fwd0 = [p.fwd_px for p in profs]
        keys = sorted(set.intersection(*[set(p.params) for p in profs])) if profs else []
        params = {key: {m: _profile_info(b, key, profiles[m][pid], minw) for m in masks if pid in profiles[m]} for key in keys}
        lrecs = lines_records(lines, pid, snap_scale)
        lby = {r.get("parameter"): r for r in lrecs if r.get("value") is not None}
        skyline_ok, verified, agreement = {}, {}, {}
        for key, pm in params.items():
            radius = key == "scale"
            lim = 0.75 if radius else 1.5
            vals = [v["val"] for v in pm.values()]
            mapv = next(iter(pm.values()))["map"]
            vt = max(1.0, 0.1 * mapv) if radius else max(2.0, 0.1 * mapv)
            # "off the map value" is judged on the sub-grid minimum with half a grid step of margin: a 5 % grid point
            # can lie a step nearer the map than the truth (a map 13 % too high verified at the tolerance's edge)
            near = all(abs(v["val_sub"] - mapv) <= vt - 0.5 * v["step_m"] for v in pm.values())
            skyline_ok[key] = bool(len(pm) >= 2 and not scaled and n_vis >= MIN_PHOTOS and n_sky >= MIN_PHOTOS
                                   and all(v["ci90_m"] <= lim and v["robust"] and not v["bound"] for v in pm.values())
                                   and max(vals) - min(vals) <= lim and near)
            # a second channel measuring the same quantity must agree (design: >= 2 channels within max(1 m, 5 %))
            lr = lby.get(LINES_KEY.get(key, ""))
            ok2 = False
            if lr is not None:
                sv = float(np.mean([v["val_sub"] for v in pm.values()]))
                lv = float(lr["value"])
                ok2 = bool(skyline_ok[key] and lr.get("n_photos", 0) >= MIN_PHOTOS and not lr.get("scale_corrected")
                           and not lr.get("vote") and abs(sv - lv) <= agree_tol(sv, lv) and abs(lv - mapv) <= vt)
                agreement[key] = {"parameter": LINES_KEY[key], "values": {"skyline": round(sv, 2), LINES: round(lv, 2)},
                                  "map": round(float(mapv), 2), "tolerance_m": round(agree_tol(sv, lv), 2), "agree": ok2}
            verified[key] = ok2
        lines_contrary = [f"{r['parameter']}:{r['vote']}" for r in lrecs if r.get("vote")]
        tops = [k for k in ("eave", "roof") if k in params]
        votes: dict[str, Any] = {}
        if tops:
            # directions on the sub-grid minimum too (a grid point can sit half a step nearer the map)
            dmin = [min(params[k][m]["top_sub"] - top for k in tops) for m in params[tops[0]]]
            dmax = [max(params[k][m]["top_sub"] - top for k in tops) for m in params[tops[0]]]
            lo = [any(params[k][m]["lo_bound"] for k in tops) for m in params[tops[0]]]
            hi = [any(params[k][m]["hi_bound"] for k in tops) for m in params[tops[0]]]
            if n_sky >= MIN_PHOTOS and all(x < -tol or lb for x, lb in zip(dmin, lo)):
                votes["skyline-down"] = round(float(np.mean(dmin)), 1) if all(x < -tol for x in dmin) else "lower bound hit"
            if all(x > tol or hb for x, hb in zip(dmax, hi)):
                votes["skyline-up?"] = round(float(np.mean(dmax)), 1) if all(x > tol for x in dmax) else "upper bound hit"
        if n_vis >= MIN_PHOTOS and n_sky < MIN_PHOTOS and all((f or 0) >= 20 for f in fwd0):
            votes.setdefault("skyline-up?", "no sky above at the map value")
        s = sfm.get(pid, {})
        if s.get("q98") is not None and s.get("model_q98") is not None and s["q98"] - s["model_q98"] > tol:
            votes["sfm-up"] = round(s["q98"] - s["model_q98"], 1)
        lvote, ltop, lkey = lines_height_vote(lrecs)
        if lvote in ("up", "down") and ltop is not None:
            votes[f"lines-{lvote}"] = round(float(ltop) - top, 1)
        up = [v for v in votes if v.endswith("-up")]
        down = [v for v in votes if v.endswith("-down")]
        guard = None
        if scaled:
            # under a snap scale correction > 3 % the 3-D lines cannot decide alone: their vote counts only beside
            # another strong vote in its direction (a weak 'skyline-up?' is not one)
            for lst, name in ((up, "lines-up"), (down, "lines-down")):
                if name in lst and len(lst) < 2:
                    lst.remove(name)
                    guard = (f"{name} under a snap scale correction x{snap_scale:.3f} (> {SNAP_SCALE_TOL:.0%}): "
                             f"cannot decide alone")
        n_up = len(up) + (1 if "skyline-up?" in votes and up else 0)
        n_down = len(down)
        stable = bool(tops) and all(all(v["ci90_m"] <= 1.5 and v["robust"] and not v["bound"] for v in params[k].values())
                                    and (len(params[k]) < 2 or abs(np.ptp([v["top"] for v in params[k].values()])) <= 1.5)
                                    for k in tops)
        channels = channel_tops(b, params, len(params[tops[0]]) if tops else len(masks), n_vis, n_sky, s, lrecs)
        # what each channel measures for the top (tickets carry it): skyline (its profile's top, bounds included),
        # SfM points (the map top plus how far the points rise above the model's own surface: comparable with the
        # top), 3-D lines (apex / ridge, or the eave with the map's roof on it)
        measured: dict[str, float] = {}
        if tops:
            kt = "roof" if "roof" in tops else "eave"           # the top itself (a flat part's eave is its top)
            if all(params[kt][m]["top_sub"] is not None for m in params[kt]):
                measured[SKYLINE] = round(float(np.mean([params[kt][m]["top_sub"] for m in params[kt]])), 2)
        if s.get("q98") is not None and s.get("model_q98") is not None:
            measured[SFM] = round(top + float(s["q98"]) - float(s["model_q98"]), 2)
        if ltop is not None:
            measured[LINES] = round(float(ltop), 2)
        agree_ch, agree_val = [], None
        ticket = None
        dirn = None
        if n_up >= 2 or n_down >= 2:
            dirn = "up" if n_up >= 2 else "down"
        if n_vis < MIN_PHOTOS:
            state, why = S.STATE_MAP, f"model skyline in {n_vis} (< {MIN_PHOTOS}) photos at the map value: photos insufficient"
            dirn = None
        elif dirn is not None:
            sg = 1 if dirn == "up" else -1
            side = [(c, ch["value"]) for c, ch in channels.items() if ch.get("ok") and (ch["value"] - top) * sg > 0]
            agree_ch, agree_val = agreeing(side)
            state, ticket = S.STATE_DISPUTED, "dispute"
            why = f"{dirn}: " + ", ".join(f"{k}={v}" for k, v in votes.items())
            if agree_ch:
                why += f"; {' + '.join(agree_ch)} agree on {agree_val} m"
        elif slender:
            state, why = S.STATE_MAP, f"slender (h/w {hw:.1f}, width {minw:.1f} m): single channels only recorded"
        elif any(verified.values()) and not votes and not lines_contrary:
            state = S.STATE_VERIFIED
            why = "verified: " + ", ".join(f"{k} (skyline {agreement[k]['values']['skyline']} + lines "
                                           f"{agreement[k]['values'][LINES]} m)" if k in agreement else k
                                           for k, v in verified.items() if v)
        else:
            bits = []
            if n_sky < MIN_PHOTOS:
                bits.append(f"sky above in {n_sky} (< {MIN_PHOTOS}) photos")
            if tops and not stable:
                bits.append("skyline profile unstable or at a bound")
            if len(masks) < 2:
                bits.append("one sky mask only: nothing can be verified")
            if scaled:
                bits.append(f"snap scale corrected x{snap_scale:.3f} (> {SNAP_SCALE_TOL:.0%}): not verified")
            if any(verified.values()):
                bits.append("would verify " + ", ".join(k for k, v in verified.items() if v)
                            + " but a channel votes against it: not verified")
            elif any(skyline_ok.values()):
                no2 = [k for k, v in skyline_ok.items() if v]
                bits.append("the skyline alone would verify " + ", ".join(no2) + f" but no second channel agrees ("
                            + "; ".join(f"{k}: lines {agreement[k]['values'][LINES]} vs skyline "
                                        f"{agreement[k]['values']['skyline']} m" if k in agreement else
                                        (f"{k}: no 3-D line measurement" if k in LINES_KEY else
                                         f"{k}: no second channel measures it") for k in no2) + "): not verified")
            if lines_contrary:
                bits.append("3-D lines: " + ", ".join(lines_contrary))
            if guard:
                bits.append(guard)
            if votes:
                bits.append("single channel: " + ", ".join(f"{k}={v}" for k, v in votes.items()))
            state, why = S.STATE_MAP, "; ".join(bits) or "no evidence"
            if "skyline-down" in votes and len(masks) >= 2:
                # one strong skyline-down vote (every mask, >= 3 photos with sky above) opens a low-strength ticket
                # until a second "down" channel agrees; the value and the state stay the map's
                ticket = "low"
                why += "; strong skyline-down alone: low-strength ticket"
            elif any(v in votes for v in ("lines-up", "lines-down")) and not guard:
                # a 3-D line measurement off the map (its interval excluding the map value) alone: a low-strength
                # ticket carrying the measured value; never under a snap scale correction > 3 % (guard above)
                ticket = "low"
                why += "; 3-D lines alone: low-strength ticket"
        if frame_open and (state in (S.STATE_VERIFIED, S.STATE_DISPUTED) or ticket):
            why = f"frame ticket open: nothing verified or fitted; would be {state}: {why}"
            state, ticket = S.STATE_MAP, None
        if b.spec.params["height"].source == st.PROV_DEFAULT:
            # a 'height pending' stub (D13) is never verified or disputed: it stays a stub (evidence and tickets),
            # unless D15 adopts a photo-measured height for it (adoption_rule)
            if state != S.STATE_MAP or ticket:
                why = f"height pending stub: state stays stub (would be {state}): {why}"
            state = S.STATE_STUB
        evidence = []
        for m in params[tops[0]] if tops else []:
            evidence.append({"kind": "skyline", "mask": m, "skyline_photos": n_vis, "sky_above_photos": n_sky,
                             "top_at_min": round(float(min(params[k][m]["top"] for k in tops)), 2),
                             # per measured quantity (eave height, top) the profile's sub-grid minimum: what the
                             # 3-D lines are compared with (G4 rechecks the agreement from these)
                             "values": {LINES_KEY[k]: params[k][m]["val_sub"] for k in tops},
                             "vote": "down" if "skyline-down" in votes else ("up?" if "skyline-up?" in votes else None),
                             "snap_scale": round(snap_scale, 4)})
        if s.get("q98") is not None:
            evidence.append({"kind": "sfm-points", "q98": s["q98"], "model_q98": s.get("model_q98"),
                             "points": s.get("points"), "min_track": s.get("min_track"),
                             "vote": "up" if "sfm-up" in votes else None, "snap_scale": round(snap_scale, 4)})
        evidence.extend(lrecs)
        if state == S.STATE_VERIFIED:
            for k, ok in verified.items():
                if ok and k in agreement:
                    evidence.append({"kind": "agreement", "channels": ["skyline", LINES], **agreement[k],
                                     "vote": None, "snap_scale": round(snap_scale, 4)})
        # the dispute's direction is kept even under an open frame ticket or on a stub: adoption_rule decides
        out[pid] = Decision(pid, state, why, votes, verified, params, n_vis, n_sky, slender,
                            None, _plain(evidence), ticket, skyline_ok, agreement, measured, agree_ch, agree_val,
                            dirn, _plain(channels))
    # D15: the adoption rule, parents first (model.built is in parents-first order), once every decision exists
    refs = scale_refs
    if refs is None and not sstat["established"]:
        refs = scale_free_refs(model, out, lines, snap_scale)
    for b in model.built:
        d = out.get(b.id)
        if d is None or d.direction is None:
            continue
        d.adopt, d.adopt_why = adoption_rule(model, b, d, out, frame_open, snap_scale, refs, sstat, points)
        d.would_fit = {"top": d.adopt["value"]} if d.adopt else None
        d.why += (f"; D15: adopt {d.adopt['value']} m ({' + '.join(d.adopt['channels'])})" if d.adopt
                  else f"; D15: ticket only ({d.adopt_why})")
    return out


def scale_free_refs(model: S.StructureModel, decisions: dict[str, Decision],
                    lines: dict[str, list[dict[str, Any]]] | None, snap_scale: float = 1.0) -> dict[str, Any]:
    """D15 ④'s reference: the main body's eave height (S.main_body) as each channel measures it, and the map's.
    skyline: the main body's eave profile (two masks, >= 3 photos, off its bounds, interval <= 1.5 m, masks within
    1.5 m); lines3d: its eave measurement (>= 3 photos, >= 2 lines, off its window's edge, interval <= 1.5 m);
    sfm-points: none (the points' q98 is a top over the roof area, not an eave)."""
    mb = S.main_body(model)
    out: dict[str, Any] = {"main_body": None if mb is None else mb.id,
                           "map_eave": None if mb is None else round(float(mb.eave), 2), "channels": {}}
    if mb is None:
        return out
    d = decisions.get(mb.id)
    sk: dict[str, Any] = {"ok": False}
    if d is None or "eave" not in d.params:
        sk["why"] = "the main body has no eave profile"
    elif d.n_vis < MIN_PHOTOS:
        sk["why"] = f"the main body is the model's skyline in {d.n_vis} (< {MIN_PHOTOS}) photos"
    else:
        pm = d.params["eave"]
        bad = _skyline_profile_problem(pm, "eave")
        vs = [float(v["val_sub"]) for v in pm.values()]
        if bad is None and max(vs) - min(vs) > CHANNEL_SPREAD_M:
            bad = f"masks {min(vs):.2f}-{max(vs):.2f} m apart"
        if bad:
            sk["why"] = bad
        else:
            sk.update(ok=True, value=round(float(np.mean(vs)), 2))
    out["channels"][SKYLINE] = sk
    lr: dict[str, Any] = {"ok": False}
    er = next((r for r in lines_records(lines, mb.id, snap_scale) if r.get("parameter") == "eave" and r.get("value") is not None), None)
    if er is None:
        lr["why"] = "no 3-D line measurement of the main body's eave"
    else:
        iv = er.get("interval") or [er["value"], er["value"]]
        if er.get("n_photos", 0) < MIN_PHOTOS or er.get("n_lines", 0) < 2:
            lr["why"] = f"the main body's eave from {er.get('n_lines', 0)} line(s) in {er.get('n_photos', 0)} photos"
        elif er.get("at_bound"):
            lr["why"] = "the main body's eave at its window's edge"
        elif float(iv[1]) - float(iv[0]) > CHANNEL_SPREAD_M or er.get("vote_withheld"):
            lr["why"] = "the main body's eave measurement is not precise or its edges disagree"
        else:
            lr.update(ok=True, value=round(float(er["value"]), 2))
    out["channels"][LINES] = lr
    out["channels"][SFM] = {"ok": False, "why": "SfM points measure a top over the roof area, not an eave: no "
                                                "scale-free reference"}
    return out


def _parent_accepted(model: S.StructureModel, pid: str, decisions: dict[str, Decision]) -> tuple[bool, str]:
    spec = model.part(pid)
    if not spec.parent:
        return True, ""
    pd = decisions.get(spec.parent)
    if pd is not None and pd.adopt:
        return False, f"parents first: re-judged on its parent {spec.parent}'s adopted value"
    try:
        ph = model.part(spec.parent).params["height"]
    except StopIteration:
        return True, ""
    pstate = pd.state if pd is not None else ph.state
    if pstate in (S.STATE_DISPUTED, S.STATE_STUB):
        return False, f"its parent {spec.parent} is {pstate}: its change is not accepted"
    return True, ""


def eave_measurements(d: Decision) -> dict[str, float]:
    """The eave as each channel measures it directly (eave_rule()): the 3-D lines' eave record (>= 3 photos, off its
    window's edge, 90 % interval <= 1.5 m, its vote not withheld) and the skyline's eave profile (two masks, its
    verification bars, masks within 1.5 m; it varies the walls with the map's roof on them)."""
    out: dict[str, float] = {}
    lr = next((e for e in d.evidence if e.get("kind") == LINES and e.get("parameter") == "eave"
               and e.get("value") is not None), None)
    if lr is not None:
        iv = lr.get("interval") or [lr["value"], lr["value"]]
        if (lr.get("n_photos", 0) >= MIN_PHOTOS and not lr.get("at_bound") and not lr.get("vote_withheld")
                and float(iv[1]) - float(iv[0]) <= CHANNEL_SPREAD_M):
            out[LINES] = round(float(lr["value"]), 2)
    pm = d.params.get("eave")
    if pm and d.n_vis >= MIN_PHOTOS and _skyline_profile_problem(pm, "eave") is None:
        vs = [float(v["val_sub"]) for v in pm.values()]
        if max(vs) - min(vs) <= CHANNEL_SPREAD_M:
            out[SKYLINE] = round(float(np.mean(vs)), 2)
    return out


def _eave_class(e: float, kept: float, moved: float) -> str:
    near_k = abs(e - kept) <= agree_tol(e, kept)
    near_m = abs(e - moved) <= agree_tol(e, moved)
    return "kept" if near_k and not near_m else "moved" if near_m and not near_k else "unclear"


def eave_rule(b: S.BuiltPart, d: Decision, value: float, scale_free: dict[str, Any] | None = None) -> dict[str, Any]:
    """Which parameter an adopted top changes (a part with a roof): adopting `height` keeps the roof height, so the
    drawn eave moves by the whole change; when the channels that measure the eave put it at the map's eave (within
    max(1 m, 5 %)) while the top is off, the roof is what changed (OSM's roof:height is often a convention): the
    roof height is adopted with the eave kept. Channels' eaves disagreeing (one at the map's eave, one moved), or one
    that cannot tell: a ticket. No eave measured: the height, the roof height kept (as before). Under ④ (scale_free:
    the common scale not established) each channel's eave is taken scale-free (x the map's main-body eave / its own
    main-body eave); a channel without that reference does not count."""
    top = float(b.top)
    kept, moved = float(b.eave), float(b.eave) + (value - top)
    out: dict[str, Any] = {"map_eave": round(kept, 2), "moved_eave": round(moved, 2), "measurements": {}, "classes": {},
                           "factors": {}, "mode": "height"}
    if b.roof.shape == "flat" or b.roof.height_m <= 0.5:
        out["why"] = "flat (or no) roof: the top is the eave"
        return out
    raw = eave_measurements(d)
    fac = {c: 1.0 for c in raw}
    if scale_free is not None:
        refs_c = scale_free.get("references") or {}
        fac = {c: float(scale_free["map_eave"]) / float(refs_c[c]) for c in raw if refs_c.get(c)}
    meas = {c: round(raw[c] * fac[c], 2) for c in fac}
    cls = {c: _eave_class(e, kept, moved) for c, e in meas.items()}
    out.update(measurements={c: raw[c] for c in fac}, factors={c: round(f, 6) for c, f in fac.items()}, classes=cls)
    if not meas:
        out["why"] = "no channel measures the eave: the height is adopted, the roof height kept"
    elif all(c == "kept" for c in cls.values()):
        out["mode"] = "roof"
        out["roof_height"] = round(float(b.spec.roof_h) + (value - top), 3)
        out["judged_roof_height"] = round(float(b.spec.roof_h), 3)
        out["why"] = (f"{' + '.join(meas)} put the eave at the map's {kept:.2f} m while the top moves to {value:.2f} m: "
                      f"the roof height is adopted ({b.spec.roof_h:g} -> {out['roof_height']:g} m), the eave kept")
        if out["roof_height"] < 0.5:
            out["ticket"] = True
            out["why"] = (f"the eave measured at the map's {kept:.2f} m but the top at {value:.2f} m leaves no roof: "
                          f"the channels contradict each other")
    elif all(c == "moved" for c in cls.values()):
        out["why"] = f"{' + '.join(meas)} put the eave at {moved:.2f} m, moved with the top: the height is adopted"
    else:
        out["ticket"] = True
        out["why"] = ("the channels' eaves disagree or cannot tell whether the walls or the roof changed ("
                      + ", ".join(f"{c} {meas[c]:.2f} m: {cls[c]}" for c in meas)
                      + f"; map eave {kept:.2f} m, moved with the top {moved:.2f} m)")
    return out


def adoption_rule(model: S.StructureModel, b: S.BuiltPart, d: Decision, decisions: dict[str, Decision],
                  frame_open: bool, snap_scale: float, refs: dict[str, Any] | None,
                  scale: dict[str, Any] | None = None,
                  points: np.ndarray | None = None) -> tuple[dict[str, Any] | None, str]:
    """D15 for one disputed part (see the module docstring): the adoption record, or None and why not. scale:
    scale_status() of the frame being judged (None: not established, ④ applies); points: the SfM points for the roof-
    surface fit (D27, own_roof_reading())."""
    if d.direction is None:
        return None, ""
    up = d.direction == "up"
    sg = 1 if up else -1
    top = float(b.top)
    tol = max(2.0, 0.1 * top)
    scale = scale or {"snap_scale": round(float(snap_scale), 4), "established": False, "by": None,
                      "why": "no frame similarity given: the common scale is not established"}
    if frame_open:
        return None, "frame ticket open: nothing is adopted"
    if d.slender:
        return None, "slender part: photos never move it"
    opp = "down" if up else "up"
    contrary = [v for v in d.votes if v.endswith("-" + opp) or (opp == "up" and v == "skyline-up?")]
    contrary += [f"lines3d {e.get('parameter')}:{opp}" for e in d.evidence if e.get("kind") == LINES and e.get("vote") == opp]
    if up and (d.channels.get(SFM) or {}).get("ceiling"):
        contrary.append(f"sfm-points: {d.channels[SFM]['why']}")
    if contrary:
        return None, "contradicting channels: " + ", ".join(contrary)
    valued = {c: ch for c, ch in d.channels.items() if ch.get("ok")}
    other = [c for c, ch in valued.items() if (ch["value"] - top) * sg <= 0]
    if other:
        return None, "contradicting channels: " + ", ".join(f"{c} measures {valued[c]['value']} m, not {d.direction} "
                                                            f"of the map's {top:.2f} m" for c in other)
    if len(valued) < ADOPT_MIN_CHANNELS:
        return None, (f"{len(valued)} channel(s) give a value ({', '.join(valued) or 'none'}); "
                      + "; ".join(f"{c}: {ch.get('why')}" for c, ch in d.channels.items() if not ch.get("ok")))
    vs = [(c, float(ch["value"])) for c, ch in valued.items()]
    dis = _pairwise_disagreement(vs)
    if dis:
        return None, f"contradicting channels: {dis} m apart"
    ok, why = _parent_accepted(model, b.id, decisions)
    if not ok:
        return None, why
    spec = b.spec
    if spec.params["height"].source == st.PROV_DEFAULT and spec.params["roof_height"].source == st.PROV_DEFAULT:
        return None, "a stub whose roof shape is tagged without a height: a curator decides the roof"
    value = float(np.mean([v for _, v in vs]))
    sf = None
    if not scale.get("established"):
        sf, why = _scale_free(model, b, valued, refs, decisions, tol, sg, scale.get("why") or "")
        if sf is None:
            return None, why
        value = sf["value"]
    ev = eave_rule(b, d, value, sf)
    if ev.get("ticket"):
        return None, ev["why"]
    if ev["mode"] == "height" and value - spec.roof_h - spec.z0 < 0.5:
        return None, f"a top of {value:.2f} m leaves no wall under the {spec.roof_h:g} m roof"
    roof, why = own_roof_reading(model, b, valued, value, points)
    if roof is None:
        return None, why
    rec = {"rule": "D15", "param": "height", "direction": d.direction, "judged_against": round(top, 3),
           "value": round(value, 2), "tolerance_m": round(tol, 2),
           "agree_tol_m": round(max(agree_tol(a, b_) for _, a in vs for _, b_ in vs), 2),
           "channels": {c: dict(ch) for c, ch in valued.items()}, "votes": dict(d.votes),
           # every channel as judged (the publish gate rechecks the contradictions from them too)
           "all_channels": {c: dict(ch) for c, ch in d.channels.items()},
           "snap_scale": round(float(snap_scale), 4), "scale": dict(scale), "scale_free": sf, "slender": bool(d.slender),
           "eave": ev, "roof_reading": roof, "provenance": PROV_AUTO}
    if ev["mode"] == "roof":
        rec["roof_height"] = {"judged_against": ev["judged_roof_height"], "value": ev["roof_height"]}
    return _plain(rec), ""


def own_roof_reading(model: S.StructureModel, b: S.BuiltPart, valued: dict[str, dict[str, Any]], value: float,
                     points: np.ndarray | None) -> tuple[dict[str, Any] | None, str]:
    """D27 (product owner, 2026-10-28): a height changes only when the agreeing evidence reads the part's OWN roof, as
    D25's U1 (crowd3d.correct.roof_reading): the 3-D lines' apex / ridge (a 'top' record) among the agreeing channels,
    or the part's roof-surface fit on the SfM points (crowd3d.roof_fit: D25's fit, usable by its fit_usable()) within
    max(1 m, 5 %) of the value adopted -- and seen over the part's interior (_interior_problem()). The skyline and the
    3-D lines' eave read the top of whatever stands at the outline (a scaffold net, a parapet with the roof behind it),
    the points' 0.98 quantile the highest thing over the part. A 'height pending' stub has no fit (D25 fits none: its
    3 m placeholder is no roof to fit around). Returns the reading (kept in the adoption record) or None and why not."""
    lr = valued.get(LINES) or {}
    if lr.get("parameter") == "top":
        return {"by": LINES, "how": lr.get("how"), "value": lr["value"]}, ""
    lines_why = "the 3-D lines measure the eave, not an apex / ridge" if lr else "no 3-D line apex / ridge agrees"
    if b.spec.params["height"].source == st.PROV_DEFAULT:
        fit_why = "none on a 'height pending' stub"
    elif points is None or not len(points):
        fit_why = "no SfM points"
    else:
        fit = RF.sfm_fit(model, b, np.asarray(points, float))
        fit_why = fit.get("why") or RF.fit_usable(fit, b.top) or _interior_problem(fit)
        if fit_why is None:
            ft = float(fit["top"])
            if abs(ft - value) <= agree_tol(ft, value):
                return {"by": SFM, "how": "roof-surface fit", "value": round(ft, 2), "fit": _plain(fit)}, ""
            fit_why = f"its top {ft:.2f} m is not within max(1 m, 5 %) of {value:.2f} m"
    return None, f"no reading of the part's own roof ({lines_why}; roof-surface fit: {fit_why})"


def _interior_problem(fit: dict[str, Any]) -> str | None:
    """A roof-surface fit reads the roof only where the roof is seen: >= 10 cells >= 2 m inside the outline hold points
    (the fit's below share is then taken over them). Without them its cells are the walls' tops along the outline
    (a roof hidden behind a parapet, or seen from the ground not at all)."""
    if fit.get("below_over") == "interior":
        return None
    return (f"fewer than {RF.INTERIOR_MIN_CELLS} cells {RF.INTERIOR_M:g} m inside the outline hold points: the roof "
            f"itself is not seen (the fit reads the walls' tops)")


def _scale_free(model: S.StructureModel, b: S.BuiltPart, valued: dict[str, dict[str, Any]], refs: dict[str, Any] | None,
                decisions: dict[str, Decision], tol: float, sg: int, why_scale: str = "") -> tuple[dict[str, Any] | None, str]:
    """D15 ④: for every agreeing channel that measures the main body's eave, its top / that eave; >= 2 such ratios,
    within 5 % of each other; the value is the mean ratio x the map's main-body eave and must still be off the map by
    more than TOL. A channel without the reference (SfM points) is listed, it neither vouches nor blocks. Applies
    whenever the common scale is not established (scale_status(): a snap rescale > 3 %, or no whole-building
    similarity measured within 3 % of k = 1)."""
    pre = why_scale or "the common scale is not established"
    refs = refs or {}
    mb = refs.get("main_body")
    if mb is None or refs.get("map_eave") in (None, 0):
        return None, f"{pre}, and no main body to divide by (④ needs a scale-free reference)"
    if b.id == mb:
        return None, f"{pre}: the main body is the scale-free reference itself (④), so it is not adopted"
    mbd = decisions.get(mb)
    mstate = mbd.state if mbd is not None else model.part(mb).params["height"].state
    if mstate in (S.STATE_DISPUTED, S.STATE_STUB):
        return None, f"{pre}: the main body {mb} (the scale-free reference) is itself {mstate}"
    ratios, rv, none = {}, {}, {}
    for c, ch in valued.items():
        r = refs.get("channels", {}).get(c, {})
        if not r.get("ok"):
            none[c] = r.get("why", "no main-body eave")
            continue
        ratios[c] = float(ch["value"]) / float(r["value"])
        rv[c] = float(r["value"])
    if len(ratios) < ADOPT_MIN_CHANNELS:
        return None, (f"{pre}: fewer than two agreeing channels have a scale-free reference ("
                      + "; ".join(f"{c}: {w}" for c, w in none.items()) + ")")
    dis = _pairwise_disagreement(list(ratios.items()), tol=lambda a, b_: SCALE_FREE_REL * max(abs(a), abs(b_)))
    if dis:
        return None, f"{pre}: the scale-free ratios (top / main-body eave) disagree: {dis}"
    v = float(np.mean(list(ratios.values()))) * float(refs["map_eave"])
    if (v - float(b.top)) * sg <= tol:
        return None, (f"{pre}: without the common scale the photos say {v:.2f} m, within the "
                      f"map's tolerance of {b.top:.2f} m")
    return {"main_body": mb, "map_eave": refs["map_eave"], "references": {c: round(x, 2) for c, x in rv.items()},
            "ratios": {c: round(x, 4) for c, x in ratios.items()}, "tolerance_rel": SCALE_FREE_REL,
            "without_reference": none, "value": round(v, 2)}, ""


def scale_hypothesis(model: S.StructureModel, decisions: dict[str, Decision], similarity: dict[str, Any] | None,
                     wikidata_height_m: float | None = None) -> dict[str, Any]:
    """What a curator needs before approving a whole-building similarity k: the outline scale alone cannot tell an
    OSM outline error from a mis-scaled reconstruction (review finding 9). A reconstruction off by k is off by k in
    height too, so the photos' heights against the map's (SfM q98 over each part's exposed area / the model's own
    surface there) are put beside k, with Wikidata's total height when known. Reported only: nothing is decided."""
    rows = []
    for pid, d in decisions.items():
        e = next((x for x in d.evidence if x.get("kind") == "sfm-points"), None)
        if e and e.get("q98") and e.get("model_q98"):
            rows.append({"part": pid, "sfm_q98": e["q98"], "map_q98": e["model_q98"], "ratio": round(e["q98"] / e["model_q98"], 3)})
    k = (similarity or {}).get("k")
    med = float(np.median([r["ratio"] for r in rows])) if rows else None
    out: dict[str, Any] = {"outline_k": k, "height_ratio_median": None if med is None else round(med, 3), "parts": rows,
                           "map_top_m": round(max((b.top for b in model.built), default=0.0), 2),
                           "wikidata_height_m": wikidata_height_m}
    add_wikidata_to_hypothesis(out, wikidata_height_m)
    if k is not None and med is not None:
        out["reading"] = ("heights agree with k: possibly a mis-scaled reconstruction, not an outline error"
                          if abs(med - k) <= 0.05 else
                          "heights do not follow k: k alone does not show a mis-scaled reconstruction (SfM heights are "
                          "evidence only: trees, neighbours and unphotographed roofs bias them)")
    return out


def add_wikidata_to_hypothesis(hyp: dict[str, Any], wikidata_height_m: float | None) -> dict[str, Any]:
    """Wikidata's total height (P2048) beside the map's top: near 1, the map's heights hold and photos' heights at k
    point at the reconstruction's scale; far from 1, the map's own heights are in question too."""
    if wikidata_height_m and wikidata_height_m > 0 and hyp.get("map_top_m"):
        hyp["wikidata_height_m"] = wikidata_height_m
        hyp["map_top_vs_wikidata"] = round(hyp["map_top_m"] / wikidata_height_m, 3)
    return hyp


# ---------------------------------------------------------------- writing decisions, D15 adoptions, the reversible list


def _active(model: S.StructureModel, kind: str | None = None) -> list[dict[str, Any]]:
    return [a for a in model.adoptions if a.get("active", True) and (kind is None or a.get("kind") == kind)]


def apply_decisions(model: S.StructureModel, decisions: dict[str, Decision], frame: dict[str, Any] | None = None,
                    auto_adopt: bool = AUTO_ADOPT) -> S.StructureModel:
    """Write the decisions into the structure's parameters (states, evidence, tickets) and the frame result into
    its registry. With auto_adopt (D15, on by default) every adoption the rule allowed is applied, parents first
    (adopt_decisions(); a part carried by an adopted parent waits to be re-judged on its new value: judge_landmark()).
    A part adopted earlier keeps its adopted value; its ticket is closed, resolved by the adoption."""
    model.tickets = [t for t in model.tickets if t.get("kind") not in ("part", "frame") or t.get("resolved_by")]
    if frame is not None:
        keep = {k: v for k, v in model.frame.items() if k in ("adopted", "curator_similarity", "before_adoption")}
        model.frame = {**{k: v for k, v in frame.items() if k != "edges"}, **keep}
        # the same rule as decide()'s: the gate failed or any channel's similarity is in question (frame_in_question)
        if frame_in_question(frame) and \
                not any(t.get("kind") == "frame" and t.get("open", True) for t in model.tickets):
            model.tickets.append({"id": f"T-{model.landmark}-frame", "kind": "frame", "open": True,
                                  "failing_edges": frame.get("failing", []), "similarity": frame.get("similarity"),
                                  "similarity_lines": frame.get("similarity_lines"),
                                  "d15": (frame.get("d15") or {}).get("why"),
                                  "scale_hypothesis": scale_hypothesis(model, decisions, frame.get("similarity"))})
    adopted = {a["part"]: a for a in _active(model, "part-height")}
    n = 0
    for p in model.specs:
        d = decisions.get(p.id)
        if d is None:
            continue
        h = p.params["height"]
        stub = h.source == st.PROV_DEFAULT
        mine = adopted.get(p.id)
        if mine is None:
            carried = [e for e in h.evidence if e.get("kind") == "carried"]
            h.evidence = carried + list(d.evidence)
            if carried and h.source == PROV_AUTO:
                h.state = S.STATE_FITTED            # carried by an adopted parent; its own decision may dispute / verify it
        tk = d.ticket or ("dispute" if d.state == S.STATE_DISPUTED else None)
        if tk:
            n += 1
            tid = f"T-{model.landmark}-{n}"
            h.ticket = tid
            if not stub and mine is None and d.state == S.STATE_DISPUTED:
                h.state = S.STATE_DISPUTED
            # the ticket carries what the channels measured, which of them agree within max(1 m, 5 %) and D15's
            # verdict; an adopted part's ticket is closed, resolved by its adoption (reversible)
            t = {"id": tid, "kind": "part", "part": p.id, "param": "height", "open": True,
                 "votes": d.votes, "why": d.why, "map_value": round(float(h.map_value if h.map_value is not None else p.top), 2),
                 "measured": dict(d.measured), "agreeing_channels": list(d.agreeing), "agreed_value": d.agreed,
                 "strength": "low" if tk == "low" else len([v for v in d.votes if not v.endswith("?")]),
                 "d15": "adopted" if mine is not None else ("adoptable (switch off)" if d.adopt else d.adopt_why or None)}
            if mine is not None:
                t.update(open=False, resolved_by=mine["id"], resolution=PROV_AUTO, adopted_value=mine["value"])
                mine["ticket"] = tid
            model.tickets.append(t)
        elif d.state == S.STATE_VERIFIED and not stub and mine is None:
            if d.verified.get("eave") or d.verified.get("roof"):
                h.state = S.STATE_VERIFIED
            if d.verified.get("roof") and p.params["roof_height"].source not in (st.PROV_DEFAULT,):
                # a verified roof height carries the evidence that verified it (G4 checks every verified parameter)
                p.params["roof_height"].state = S.STATE_VERIFIED
                p.params["roof_height"].evidence = list(d.evidence)
        p.problems = [x for x in p.problems if not x.startswith("judge:")] + [f"judge: {d.state}: {d.why}"]
    if auto_adopt:
        adopt_decisions(model, decisions)
    _close_adopted_tickets(model)
    return model


def _close_adopted_tickets(model: S.StructureModel) -> None:
    """An adopted part's ticket is closed, resolved by its adoption (it reopens if the adoption is reverted)."""
    for a in _active(model, "part-height"):
        for t in model.tickets:
            if t.get("kind") == "part" and t.get("part") == a["part"]:
                if t.get("open", True) and not t.get("resolved_by"):
                    t.update(open=False, resolved_by=a["id"], resolution=PROV_AUTO, adopted_value=a["value"], d15="adopted")
                if t.get("resolved_by") == a["id"]:
                    a["ticket"] = t["id"]
                    model.part(a["part"]).params["height"].ticket = t["id"]


def _snapshot(v: S.Param) -> dict[str, Any]:
    return {"value": float(v.value), "source": v.source, "state": v.state, "evidence": copy.deepcopy(v.evidence),
            "ticket": v.ticket, "map_value": v.map_value}


def _restore(v: S.Param, snap: dict[str, Any]) -> None:
    v.value, v.source, v.state = snap["value"], snap["source"], snap["state"]
    v.evidence, v.ticket, v.map_value = copy.deepcopy(snap["evidence"]), snap["ticket"], snap["map_value"]


def _map_source(v: S.Param) -> str:
    """The source the map gave this parameter (through earlier carries or adoptions)."""
    for e in v.evidence:
        if e.get("kind") in ("carried", "adoption") and e.get("map_source"):
            return e["map_source"]
    return v.source


def adopt_decisions(model: S.StructureModel, decisions: dict[str, Decision]) -> list[str]:
    """D15: apply every adoption record the rule produced, parents first, after rechecking it (check_adoption_record).
    A part carried by a parent adopted in this call is not adopted on evidence judged at the parent's old value: it is
    returned to be re-judged on the parent's new value (with every other part the adoption carried). The solids are
    rebuilt and G10 resolved again after any change."""
    pending: list[str] = []
    moved: set[str] = set()
    changed = False
    done = {a["part"] for a in _active(model, "part-height")}
    for b in list(model.built):                          # parents first
        d = decisions.get(b.id)
        if d is None or not d.adopt or b.id in done:
            continue
        if b.id in moved:
            pending.append(b.id)
            continue
        if check_adoption_record(d.adopt):
            continue
        rec = _adopt_part(model, b.spec, d)
        changed = True
        for c in rec["carried"]:
            moved.add(c["part"])
            pending.append(c["part"])
    if changed:
        model.built = S.build_all(model.specs, model.params)
        S.resolve_support(model)
    order = {b.id: i for i, b in enumerate(model.built)}
    return sorted({p for p in pending if p in order}, key=lambda p: order[p])


def _adopt_part(model: S.StructureModel, p: S.PartSpec, d: Decision) -> dict[str, Any]:
    h = p.params["height"]
    aid = f"A-{model.landmark}-{len(model.adoptions) + 1}"
    old, new = float(h.value), float(d.adopt["value"])
    map_value = float(h.map_value) if h.map_value is not None else old
    map_source = _map_source(h)
    before = [{"part": p.id, "param": "height", **_snapshot(h)}]
    roof = d.adopt.get("roof_height")                   # eave kept: the roof height changes with the top (eave_rule)
    if roof:
        before.append({"part": p.id, "param": "roof_height", **_snapshot(p.params["roof_height"])})
    depends = [a["id"] for a in _active(model, "frame")]
    depends += sorted({e["adoption"] for e in h.evidence if e.get("kind") == "carried" and e.get("adoption")})
    depends += sorted({e["adoption"] for e in p.params["min_height"].evidence
                       if e.get("kind") == "carried" and e.get("adoption")} - set(depends))
    ev = {"kind": "adoption", "id": aid, **d.adopt, "map_value": round(map_value, 3), "map_source": map_source}
    h.value, h.source, h.state, h.map_value = new, PROV_AUTO, S.STATE_FITTED, map_value
    h.evidence = list(d.evidence) + [ev]
    roof_rec = None
    if roof:
        rh = p.params["roof_height"]
        rh_map = float(rh.map_value) if rh.map_value is not None else float(rh.value)
        roof_rec = {"map_value": round(rh_map, 3), "map_source": _map_source(rh), "judged_against": round(float(rh.value), 3),
                    "value": round(float(roof["value"]), 3)}
        rh.evidence = list(rh.evidence) + [{**ev, "roof_map_value": roof_rec["map_value"], "roof_map_source": roof_rec["map_source"]}]
        rh.value, rh.source, rh.state, rh.map_value = float(roof["value"]), PROV_AUTO, S.STATE_FITTED, rh_map
    carried = _carry(model, p.id, new - old, aid, float(d.adopt.get("snap_scale", 1.0)))
    for c in carried:
        before.extend(c.pop("before"))
    rec = {"id": aid, "kind": "part-height", "part": p.id, "name": p.name, "param": "height",
           "map_value": round(map_value, 3), "map_source": map_source, "judged_against": round(old, 3),
           "value": round(new, 3), "provenance": PROV_AUTO, "label": AUTO_LABEL, "rule": "D15",
           "direction": d.adopt["direction"], "channels": d.adopt["channels"], "scale_free": d.adopt.get("scale_free"),
           "scale": d.adopt.get("scale"), "eave": d.adopt.get("eave"), "roof_height": roof_rec,
           "roof_reading": d.adopt.get("roof_reading"),
           "snap_scale": d.adopt.get("snap_scale"), "agree_tol_m": d.adopt.get("agree_tol_m"),
           "evidence": list(d.evidence), "carried": carried, "before": before, "depends_on": depends,
           "rejudged_round": d.rejudged, "active": True, "ticket": None, "reversible": True,
           "osm_edited": False}
    model.adoptions.append(_plain(rec))
    p.problems.append(f"D15 {aid}: height {old:g} -> {new:g} m photo-fitted (auto); the map's {map_value:g} m kept"
                      + (f"; roof height {roof_rec['judged_against']:g} -> {roof_rec['value']:g} m (eave kept)" if roof_rec else ""))
    return model.adoptions[-1]


def _carry(model: S.StructureModel, pid: str, dz: float, aid: str, snap_scale: float) -> list[dict[str, Any]]:
    """D16 ② / D15 ⑤: the parts resting on an adopted part follow it (their base and top move by its change; their
    own extent is kept); each carried parameter records the adoption that moved it (photo-fitted (auto), the map's
    value kept). They are re-judged on the new value (judge_landmark). A 'height pending' stub (D13: height
    default-style) is carried but stays a stub: its base is carried like any other part's, its placeholder top moves
    with it (its own extent kept) and keeps its source default-style and its state stub (a carried placeholder is
    still no measured height); only an adoption on its own evidence could give it one."""
    out = []
    if abs(dz) < 1e-9:
        return out
    for cid in S.descendants(model, pid):
        c = model.part(cid)
        stub = c.params["height"].source == st.PROV_DEFAULT
        before = [{"part": cid, "param": k, **_snapshot(c.params[k])} for k in ("min_height", "height")]
        for k in ("min_height", "height"):
            v = c.params[k]
            ms = _map_source(v)
            if v.map_value is None:
                v.map_value = float(v.value)
            v.value = float(v.value) + dz
            if not (stub and k == "height"):
                v.source, v.state = PROV_AUTO, S.STATE_FITTED
            v.evidence = list(v.evidence) + [{"kind": "carried", "parent": pid, "adoption": aid, "dz": round(dz, 3),
                                              "map_source": ms, "snap_scale": round(snap_scale, 4),
                                              **({"stub": True} if stub and k == "height" else {})}]
        c.problems.append(f"D15 {aid}: carried {dz:+.2f} m with {pid} (re-judged on its new value)"
                          + ("; still a 'height pending' stub" if stub else ""))
        out.append({"part": cid, "dz": round(dz, 3), "before": before, **({"stub": True} if stub else {})})
    return out


def revert_adoption(model: S.StructureModel, aid: str) -> list[str]:
    """Undo one adoption and every active adoption that depends on it (the parts it carried that were re-judged and
    adopted, every part adoption made on an adopted frame), newest first: the old values, sources, states, evidence
    and outlines come back, the tickets reopen, the records stay in the list marked reverted; reverting a frame also
    withdraws the verifications made on it (they wait for a new judge run). Returns the ids undone."""
    act = _active(model)
    if not any(a["id"] == aid for a in act):
        return []
    dep = {aid}
    grew = True
    while grew:
        grew = False
        for a in act:
            if a["id"] not in dep and set(a.get("depends_on") or []) & dep:
                dep.add(a["id"])
                grew = True
    undone = []
    for a in reversed(act):
        if a["id"] not in dep:
            continue
        if a["kind"] == "frame":
            k, t, c = a["value"]["k"], np.asarray(a["value"]["t"], float), np.asarray(a["value"]["centre"], float)
            _transform(model, 1.0 / k, (float(-t[0]), float(-t[1])), c + t, None, f"D15 {a['id']} reverted")
            for p in model.specs:
                p.outline_source = a["before_outline_sources"].get(p.id, p.outline_source)
                # every judgement made on the adopted frame is withdrawn with it: a verification goes back to the
                # state of its source (the part waits for a new judge run; the frame ticket is open again)
                for k, v in p.params.items():
                    if v.state == S.STATE_VERIFIED:
                        v.state = S.STATE_OF_SOURCE.get(v.source, S.STATE_MAP)
                        p.problems.append(f"D15 {a['id']} reverted: the {k} verification made on its frame is withdrawn")
            model.frame.pop("adopted", None)
        elif a["kind"] == "d25-tilt":
            # D25-T: the stack stands upright again (its solids never moved: the pose is applied at export)
            model.pose = copy.deepcopy(a.get("before_pose"))
        else:
            for snap in a["before"]:
                _restore(model.part(snap["part"]).params[snap["param"]], snap)
            if a["kind"] == "d25-block" and a.get("before_roof_shape"):
                model.part(a["part"]).roof_shape = a["before_roof_shape"]
        for t in model.tickets:
            if t.get("resolved_by") == a["id"]:
                t["open"] = True
                t["reverted"] = a["id"]
                t.pop("resolved_by", None)
                t.pop("resolution", None)
            if t.get("d25_record") == a["id"]:
                t.pop("d25_record", None)
                t.pop("d25", None)
        a["active"], a["reverted"] = False, True
        undone.append(a["id"])
    model.built = S.build_all(model.specs, model.params)
    S.resolve_support(model)
    return undone


def withdraw_photo_judgements(model: S.StructureModel, why: str, keep: set[str] | None = None) -> dict[str, Any] | None:
    """The photos a judge measured this structure with were not registered onto the map (the publish step refused
    them: crowd3d.stylize's registration gate, a prior-only model, no points): nothing they said about the structure
    may be published. Every active D15 adoption is reverted (revert_adoption, newest first, with every adoption that
    depends on it: the map's values, sources, outlines and tickets come back, the records stay in the reversible list
    marked reverted) and every photo verification goes back to the state of its source (its value is the map's). A
    curator's value stays (a person's decision); a dispute keeps the map value anyway (its ticket stays for the
    curator). Returns what was withdrawn, also kept in model.frame['withdrawn'] (None: there was nothing to withdraw,
    and the model is unchanged). keep: ids of D25 corrections whose own evidence-grade registration still holds
    (crowd3d.correct.records_to_keep: a refusal of the colour-grade registration does not touch a structural
    correction whose box was sized for the registration it had); a kept record is still reverted when something it
    depends on is."""
    verified = [(p.id, k) for p in model.specs for k, v in p.params.items() if v.state == S.STATE_VERIFIED]
    undone: list[str] = []
    keep = set(keep or ())
    while True:
        act = [a for a in _active(model) if a["id"] not in keep]
        got = revert_adoption(model, act[-1]["id"]) if act else []
        if not got:
            break
        undone += got
    for p in model.specs:
        for k, v in p.params.items():
            if v.state == S.STATE_VERIFIED:          # verifications made on the map's own outline
                v.state = S.STATE_OF_SOURCE.get(v.source, S.STATE_MAP)
                p.problems.append(f"photo judgements withdrawn ({why}): the {k} verification is withdrawn")
    if not undone and not verified:
        return None
    rec ={"why": why, "adoptions_reverted": undone, "verifications_withdrawn": [f"{pid}.{k}" for pid, k in verified]}
    model.frame["withdrawn"] = rec
    return rec


# ---------------------------------------------------------------- the frame (D15 ⑥)


def _outline_width(model: S.StructureModel) -> float:
    return 2.0 * S.obb(S.ccw(model.main_rings[0])).W


def frame_scale_check(refs: dict[str, Any] | None, k: float) -> dict[str, Any]:
    """D15 ④ for the frame (outline width / eave height): an outline scale k off by > 3 % cannot by itself tell an
    OSM outline error from a mis-scaled reconstruction (the snap may return x1.00 on one-sided data, and every channel
    shares its scale). refs: scale_free_refs() measured on the TRIAL outline (the similarity applied). Passed when
    >= 2 channels (skyline, 3-D lines) measure the main body's eave there, agree with each other and with the map's
    eave within max(1 m, 5 %) -- the reconstruction's heights are the map's, so its scale is -- and each channel's
    scale-free outline change k x map eave / its eave is still off by > 3 % on k's side. A mis-scaled reconstruction
    (heights scaled with the outline) fails; a common scale error smaller than the agreement tolerance plus the
    map's own eave error cannot be told apart (an honest limit: e.g. x0.95 on an eave the map has 4 % low)."""
    refs = refs or {}
    E = refs.get("map_eave")
    out: dict[str, Any] = {"needed": True, "k": round(float(k), 4), "main_body": refs.get("main_body"),
                           "map_eave": E, "eaves": {}, "k_scale_free": {}, "passed": False}
    if not E:
        out["why"] = "no main body (the eave reference) to check the scale against"
        return out
    eaves = {c: float(r["value"]) for c, r in (refs.get("channels") or {}).items() if r.get("ok")}
    out["eaves"] = {c: round(v, 2) for c, v in eaves.items()}
    out["k_scale_free"] = {c: round(float(k) * float(E) / v, 4) for c, v in eaves.items()}
    out["not_measured"] = {c: r.get("why") for c, r in (refs.get("channels") or {}).items() if not r.get("ok")}
    if len(eaves) < ADOPT_MIN_CHANNELS:
        out["why"] = (f"the main body's eave on the adopted outline is measured by {len(eaves)} channel(s) "
                      f"(< {ADOPT_MIN_CHANNELS}): the reconstruction's scale is not confirmed")
        return out
    dis = _pairwise_disagreement(list(eaves.items()))
    if dis:
        out["why"] = f"the channels' main-body eaves disagree ({dis})"
        return out
    off = {c: v for c, v in eaves.items() if abs(v - float(E)) > agree_tol(v, float(E))}
    if off:
        out["why"] = ("the photos' main-body eave " + ", ".join(f"{c} {v:.2f} m" for c, v in off.items())
                      + f" is off the map's {float(E):.2f} m by more than max(1 m, 5 %): the reconstruction's scale may "
                        f"be what is off (outline k {k:.3f})")
        return out
    side = 1 if k > 1 else -1
    weak = {c: x for c, x in out["k_scale_free"].items() if (x - 1.0) * side <= FRAME_K_TOL}
    if weak:
        out["why"] = (f"outline width / eave height changes by <= 3 % ({weak}): the outline change is not scale-free, "
                      f"it may be the reconstruction's scale")
        return out
    out["passed"] = True
    out["why"] = (f"the main body's eave {', '.join(f'{c} {v:.2f}' for c, v in eaves.items())} m agrees with the map's "
                  f"{float(E):.2f} m: the reconstruction's scale is the map's; outline width / eave changes by "
                  f"{', '.join(f'x{x:.3f}' for x in out['k_scale_free'].values())}")
    return out


def _frame_scale_problem(a: dict[str, Any]) -> str | None:
    """frame_scale_check() rechecked from a frame adoption record."""
    k = float(a.get("k", 1.0))
    if abs(k - 1.0) <= FRAME_K_TOL:
        return None
    sc = a.get("scale_check") or {}
    E = sc.get("map_eave")
    eaves = {c: float(v) for c, v in (sc.get("eaves") or {}).items()}
    if not E or len(eaves) < ADOPT_MIN_CHANNELS:
        return "outline scale off by > 3 % without the main body's eave measured on it by >= 2 channels (④)"
    if _pairwise_disagreement(list(eaves.items()), tol=lambda x, y: agree_tol(x, y) + 0.01):
        return "the main-body eaves of the frame's scale check disagree"
    side = 1 if k > 1 else -1
    for c, v in eaves.items():
        if abs(v - float(E)) > agree_tol(v, float(E)) + 0.01:
            return f"the {c} main-body eave {v} m does not confirm the reconstruction's scale (map {E} m)"
        if (k * float(E) / v - 1.0) * side <= FRAME_K_TOL:
            return f"the outline change is not scale-free for {c} (k x map eave / eave within 3 % of 1)"
    return None


def decide_frame(model: S.StructureModel, gate: dict[str, Any] | None, channels: dict[str, dict[str, Any]],
                 snap_scale: float = 1.0, points: np.ndarray | None = None,
                 quality: np.ndarray | None = None, eave_check=None) -> dict[str, Any]:
    """D15 ⑥: the whole-building similarity is adopted by the same rule as a part's value. In question when the gate
    (G11) fails or a channel's similarity opens a frame ticket (k off by > 3 % or a shift > 1.5 m). Each channel gives
    a value when it is measured (walls facing >= 180 deg with two opposite ones), its evidence comes from >= 3 photos
    and it is not at a fit bound; >= 2 such channels must agree on the outline's width (k x the narrowest width) and
    on the shift within max(1 m, 5 %), lie on the same side of the map's frame for what is off (k above / below 1,
    shifts pointing the same way), and the gate must pass on the adopted outline (else it is not a whole-building
    similarity). Under a snap scale correction > 3 % such a frame is not adopted (a ticket). An outline scale off by
    > 3 % also needs ④'s scale-free check (frame_scale_check()): eave_check(trial model) -> scale_free_refs() of the
    model with the similarity applied (judge_landmark() measures the main body's eave there with the skyline and the
    3-D lines); without it, a ticket. Returns {'in_question', 'channels', 'adopt' (record or None), 'why'}."""
    W = _outline_width(model)
    in_q = bool((gate is not None and not gate.get("passed", True)) or any(ch.get("ticket") for ch in channels.values()))
    rows = {}
    for name, ch in channels.items():
        r: dict[str, Any] = {"measured": bool(ch.get("measured")), "k": ch.get("k"), "t": ch.get("t"),
                             "edges": len(ch.get("edges", [])), "ok": False, "photos": ch.get("photos_min") or 0,
                             "at_bound": bool(ch.get("at_bound")), "residual_max_m": ch.get("residual_max_m")}
        if not ch.get("measured"):
            r["why"] = ch.get("reason") or "walls do not face >= 180 deg with two opposite ones"
        elif r["photos"] < MIN_PHOTOS:
            r["why"] = "its evidence is not known to come from >= 3 photos"
        elif ch.get("at_bound"):
            r["why"] = f"at a fit bound: {ch.get('bound_why')}"
        else:
            r["ok"] = True
            r["width_m"] = round(float(ch["k"]) * W, 3)
        rows[name] = r
    out: dict[str, Any] = {"in_question": in_q, "channels": rows, "width_map_m": round(W, 3), "adopt": None, "why": "",
                           "snap_scale": round(float(snap_scale), 4)}
    if not in_q:
        out["why"] = "the frame gate passes and no channel opens a frame ticket: nothing to adopt"
        return out
    valued = {n: r for n, r in rows.items() if r["ok"]}
    if len(valued) < ADOPT_MIN_CHANNELS:
        out["why"] = (f"{len(valued)} channel(s) measure the frame ({', '.join(valued) or 'none'}); "
                      + "; ".join(f"{n}: {r.get('why')}" for n, r in rows.items() if not r["ok"]))
        return out
    names = list(valued)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = valued[names[i]], valued[names[j]]
            dw = abs(a["width_m"] - b["width_m"])
            tw = max(AGREE_M, AGREE_REL * max(a["width_m"], b["width_m"]))
            ta, tb = np.asarray(a["t"], float), np.asarray(b["t"], float)
            dt = float(np.linalg.norm(ta - tb))
            tt = max(AGREE_M, AGREE_REL * max(np.linalg.norm(ta), np.linalg.norm(tb)))
            if dw > tw or dt > tt:
                out["why"] = (f"contradicting channels: {names[i]} width {a['width_m']:.2f} m, shift {list(ta)} vs "
                              f"{names[j]} width {b['width_m']:.2f} m, shift {list(tb)} (allowed {tw:.2f} m / {tt:.2f} m)")
                return out
    ks = np.array([float(valued[n]["k"]) for n in names])
    ts = np.array([valued[n]["t"] for n in names], float)
    kbar, tbar = float(ks.mean()), ts.mean(axis=0)
    k_off, t_off = abs(kbar - 1.0) > FRAME_K_TOL, float(np.linalg.norm(tbar)) > FRAME_T_TOL_M
    if not (k_off or t_off):
        out["why"] = ("the channels agree with the map's frame (k within 3 %, shift within 1.5 m): the gate's failure is "
                      "not a whole-building similarity")
        return out
    if k_off and not (np.all(ks > 1.0) or np.all(ks < 1.0)):
        out["why"] = f"contradicting channels: outline scales {list(np.round(ks, 4))} on both sides of the map's"
        return out
    if t_off and any(float(ts[i] @ ts[j]) <= 0 for i in range(len(ts)) for j in range(i + 1, len(ts))):
        out["why"] = "contradicting channels: the shifts point different ways"
        return out
    if abs(snap_scale - 1.0) > SNAP_SCALE_TOL:
        out["why"] = (f"snap scale corrected x{snap_scale:.3f} (> 3 %): the frame needs a scale-free check (outline width "
                      f"/ eave height) and the SfM wall planes measure no eave: a ticket")
        return out
    c = np.asarray(channels[names[0]]["centre"], float)
    adopt = {"rule": "D15", "param": "outline", "k": round(kbar, 4), "t": [round(float(v), 3) for v in tbar],
             "centre": [round(float(v), 3) for v in c], "channels": {n: dict(valued[n]) for n in names},
             "width_map_m": round(W, 3), "width_m": round(kbar * W, 3), "snap_scale": round(float(snap_scale), 4),
             "provenance": PROV_AUTO,
             # disclosed with the adoption: the two channels share evidence (design §2.2's skyline (k, t) search, the
             # more independent second method, is not implemented)
             "channels_independent": False,
             "dependence": "the 3-D lines are kept only where SfM points support them (lines3d S+V): their wall "
                           "offsets largely re-measure the SfM wall points"}
    if points is not None and len(points):
        trial = [c + (np.asarray(r, float) - c) * kbar + tbar for r in model.main_rings]
        g2 = frame_gate(points, trial, quality)
        adopt["gate_on_adopted_outline"] = {k: v for k, v in g2.items() if k != "edges"}
        if not g2["passed"]:
            out["why"] = (f"the gate still fails on the adopted outline (edges {[f['edge'] for f in g2['failing']]}): "
                          f"not a whole-building similarity")
            return out
    if k_off:
        # ④: outline width / eave height -- is it the outline, or the reconstruction's scale?
        if eave_check is None:
            out["why"] = ("outline scale off by > 3 %: the main body's eave must be measured on the adopted outline to "
                          "tell an outline error from a mis-scaled reconstruction (④); not measured: a ticket")
            out["scale_check"] = {"needed": True, "passed": False, "why": "no eave measurement given"}
            return out
        tm = copy.deepcopy(model)
        _transform(tm, kbar, (float(tbar[0]), float(tbar[1])), c, None, "D15 frame scale check (trial)")
        sc = frame_scale_check(eave_check(tm), kbar)
        adopt["scale_check"] = sc
        out["scale_check"] = sc
        if not sc["passed"]:
            out["why"] = f"outline scale off by > 3 % and not scale-free (④): {sc['why']}"
            return out
    out["adopt"] = _plain(adopt)
    return out


def _transform(model: S.StructureModel, k: float, t: tuple[float, float], c: np.ndarray, source: str | None,
               note: str) -> None:
    """A whole-building similarity (scale k about c, then shift t) applied rigidly to every part and the main
    outline (never one edge or one part alone); outlines take `source` (None: kept)."""
    c = np.asarray(c, float)
    tt = np.asarray(t, float)

    def tf(r):
        return c + (np.asarray(r, float) - c) * k + tt

    for p in model.specs:
        p.outer = tf(p.outer)
        p.holes = [tf(h) for h in p.holes]
        if source is not None:
            p.outline_source = source
        p.problems.append(f"{note}: whole-building similarity k={k:.4f}, t=({tt[0]:.2f}, {tt[1]:.2f}) m")
    model.main_rings = [tf(r) for r in model.main_rings]
    model.built = S.build_all(model.specs, model.params)


def adopt_frame(model: S.StructureModel, fd: dict[str, Any], gate: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """D15 ⑥: apply an adopted whole-building similarity (decide_frame's record) to every part and the main outline;
    outlines become photo-fitted (auto), the open frame ticket is closed (resolved by the adoption) and the adoption
    joins the reversible list (the inverse similarity restores the map's outlines)."""
    a = fd.get("adopt")
    if not a or check_frame_record(a):
        return None
    aid = f"A-{model.landmark}-{len(model.adoptions) + 1}"
    before_src = {p.id: p.outline_source for p in model.specs}
    _transform(model, float(a["k"]), tuple(a["t"]), np.asarray(a["centre"], float), PROV_AUTO,
               f"D15 {aid} (photo-fitted (auto))")
    S.resolve_support(model)
    if not any(t.get("kind") == "frame" for t in model.tickets):
        model.tickets.append({"id": f"T-{model.landmark}-frame", "kind": "frame", "open": True,
                              "failing_edges": (gate or {}).get("failing", [])})
    for t in model.tickets:
        if t.get("kind") == "frame" and t.get("open", True):
            t.update(open=False, resolved_by=aid, resolution=PROV_AUTO)
    rec = {"id": aid, "kind": "frame", "param": "outline", "map_value": {"k": 1.0, "t": [0.0, 0.0]},
           "value": {"k": a["k"], "t": a["t"], "centre": a["centre"]}, "provenance": PROV_AUTO, "label": AUTO_LABEL,
           "rule": "D15", "channels": a["channels"], "width_map_m": a["width_map_m"], "width_m": a["width_m"],
           "snap_scale": a["snap_scale"], "gate_before": {k: v for k, v in (gate or {}).items() if k != "edges"},
           "gate_on_adopted_outline": a.get("gate_on_adopted_outline"), "before_outline_sources": before_src,
           "depends_on": [], "active": True, "reversible": True, "osm_edited": False, "record": a}
    model.adoptions.append(_plain(rec))
    model.frame["adopted"] = {"id": aid, "k": a["k"], "t": a["t"], "centre": a["centre"]}
    return model.adoptions[-1]


def apply_frame(model: S.StructureModel, k: float, t: tuple[float, float], centre: np.ndarray | None = None,
                note: str = "curator") -> S.StructureModel:
    """A curator-approved whole-building similarity (scale k about the outline's centre, then shift t) applied
    rigidly to every part and the main outline (never one edge or one part alone); outlines become 'curator'."""
    c = np.asarray(centre if centre is not None else st.polygon_centroid(S.ccw(model.main_rings[0])), float)
    tt = np.asarray(t, float)
    _transform(model, k, (float(tt[0]), float(tt[1])), c, st.PROV_CURATOR, note)
    model.frame = {**model.frame, "curator_similarity": {"k": k, "t": [float(tt[0]), float(tt[1])], "centre": c.tolist(),
                                                         "note": note}}
    model.tickets = [dict(tk, open=False, resolved_by=note) if tk.get("kind") == "frame" else tk for tk in model.tickets]
    return model


# ---------------------------------------------------------------- G4 / G9: the publish gate's recheck


def check_adoption_record(e: dict[str, Any]) -> str | None:
    """D15's rule rechecked from an adoption record alone (the publish gate never trusts the code that wrote it):
    >= 2 channels, each from >= 3 photos and off its fit bounds, each on the record's side of the value it was judged
    against, all pairwise within max(1 m, 5 %); a skyline channel from two masks (and, below the map, sky above in
    >= 3 photos); not slender; under a snap scale correction > 3 % a scale-free check whose ratios agree within 5 %;
    the adopted value = the channels' mean (or the scale-free value); a reading of the part's own roof (D27: the 3-D
    lines' apex / ridge among the channels, or a usable roof-surface fit seen over the part's interior, within max(1 m,
    5 %) of the value)."""
    ch = e.get("channels") or {}
    if len(ch) < ADOPT_MIN_CHANNELS:
        return f"adopted from {len(ch)} channel(s) (< {ADOPT_MIN_CHANNELS})"
    if e.get("slender"):
        return "a slender part adopted"
    sg = 1 if e.get("direction") == "up" else -1 if e.get("direction") == "down" else 0
    ref = e.get("judged_against")
    if not sg or ref is None:
        return "adoption without a direction or the value it was judged against"
    vals = []
    for c, x in ch.items():
        v = x.get("value")
        if v is None:
            return f"{c} gives no value"
        if (x.get("photos") or 0) < MIN_PHOTOS:
            return f"{c} from < {MIN_PHOTOS} photos"
        if x.get("at_bound"):
            return f"{c} at a fit bound"
        if (float(v) - float(ref)) * sg <= 0:
            return f"{c} ({v} m) not {e.get('direction')} of {ref} m"
        if c == SKYLINE and ((x.get("masks") or 0) < 2 or (sg < 0 and (x.get("sky_above_photos") or 0) < MIN_PHOTOS)):
            return "a skyline value without two masks (or, below the map, without sky above in >= 3 photos)"
        vals.append((c, float(v)))
    dis = _pairwise_disagreement(vals, tol=lambda a, b: agree_tol(a, b) + 0.01)
    if dis:
        return f"channels disagree: {dis}"
    expected = float(np.mean([v for _, v in vals]))
    if "snap_scale" not in e:
        return "adoption without its snap scale"
    if sg > 0 and any((x or {}).get("ceiling") for x in (e.get("all_channels") or {}).values()):
        return "adopted higher although the SfM points show the part ending at the map's top all round"
    sf = e.get("scale_free")
    established = _scale_established(e.get("scale")) and abs(float(e["snap_scale"]) - 1.0) <= SNAP_SCALE_TOL
    if not established:
        if not sf:
            return "the common scale is not established (snap rescale > 3 % or no similarity measured within 3 %) " \
                   "without a scale-free check"
        ratios = sf.get("ratios") or {}
        if len(ratios) < ADOPT_MIN_CHANNELS or not set(ratios) <= set(ch):
            return "the scale-free check does not cover two of the agreeing channels"
        dis = _pairwise_disagreement(list(ratios.items()), tol=lambda a, b: SCALE_FREE_REL * max(abs(a), abs(b)) + 1e-4)
        if dis:
            return f"scale-free ratios disagree: {dis}"
        expected = float(np.mean(list(ratios.values()))) * float(sf.get("map_eave") or 0.0)
    if abs(float(e.get("value", math.nan)) - expected) > 0.02:
        return f"adopted value {e.get('value')} is not the channels' {'scale-free ' if sf else ''}mean {expected:.2f}"
    rr = e.get("roof_reading") or {}
    if rr.get("by") == LINES:
        if (ch.get(LINES) or {}).get("parameter") != "top":
            return "the roof reading is a 3-D line apex / ridge that the agreeing channels do not hold"
    elif rr.get("by") == SFM:
        fit = rr.get("fit") or {}
        why = RF.fit_usable(fit, float(ref)) or _interior_problem(fit)
        if why or abs(float(fit["top"]) - float(e["value"])) > agree_tol(float(fit["top"]), float(e["value"])) + 0.01:
            return f"the roof-surface fit is no reading of the part's own roof at the value adopted ({why or 'off it'})"
    else:
        return "a height adopted without a reading of the part's own roof (D27)"
    return _eave_record_problem(e, sf if not established else None)


def _eave_record_problem(e: dict[str, Any], sf: dict[str, Any] | None) -> str | None:
    """eave_rule() rechecked from the record: a channel that measured the eave at the map's eave while the top moved
    means the roof height changes (eave kept); all of them moved: the height; mixed or unclear: no adoption."""
    ev = e.get("eave")
    roof = e.get("roof_height")
    if not ev:
        return "an adoption that changes the roof height without its eave record" if roof else None
    meas = ev.get("measurements") or {}
    fac = ev.get("factors") or {}
    if sf is None and any(abs(float(fac.get(c, 1.0)) - 1.0) > 1e-6 for c in meas):
        return "eave measurements rescaled although the common scale is established"
    if sf is not None:
        refs_c = sf.get("references") or {}
        for c in meas:
            if not refs_c.get(c) or abs(float(fac.get(c, 0.0)) - float(sf["map_eave"]) / float(refs_c[c])) > 1e-4:
                return f"the {c} eave is not taken scale-free under ④"
    kept = float(ev.get("map_eave", math.nan))
    moved = kept + float(e["value"]) - float(e["judged_against"])
    cls = {c: _eave_class(float(m) * float(fac.get(c, 1.0)), kept, moved) for c, m in meas.items()}
    if cls and all(c == "kept" for c in cls.values()):
        if not roof:
            return "a channel measured the eave at the map's eave, but the whole height was adopted (the eave moved)"
        if abs(float(roof["value"]) - (float(roof["judged_against"]) + float(e["value"]) - float(e["judged_against"]))) > 0.02:
            return "the adopted roof height is not the map's plus the top's change"
        return None
    if roof:
        return "the roof height adopted without every eave measurement at the map's eave"
    if cls and not all(c == "moved" for c in cls.values()):
        return f"the channels' eaves disagree or cannot tell what changed: {cls}"
    return None


def check_frame_record(a: dict[str, Any]) -> str | None:
    """D15 ⑥ rechecked from a frame adoption record: >= 2 channels, each measured from >= 3 photos off its bounds,
    agreeing on width and shift within max(1 m, 5 %), the adopted k and t their means, no snap scale correction > 3 %,
    and an outline scale off by > 3 % carrying a scale check that passes again (_frame_scale_problem())."""
    ch = a.get("channels") or {}
    if len(ch) < ADOPT_MIN_CHANNELS:
        return f"frame adopted from {len(ch)} channel(s)"
    if abs(float(a.get("snap_scale", 1.0)) - 1.0) > SNAP_SCALE_TOL:
        return "frame adopted under a snap scale correction > 3 %"
    W = float(a.get("width_map_m") or 0.0)
    ks, ts = [], []
    for n, r in ch.items():
        if not r.get("measured") or (r.get("photos") or 0) < MIN_PHOTOS or r.get("at_bound"):
            return f"{n}: not measured from >= 3 photos off its bounds"
        ks.append(float(r["k"]))
        ts.append(np.asarray(r["t"], float))
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            wa, wb = ks[i] * W, ks[j] * W
            if abs(wa - wb) > max(AGREE_M, AGREE_REL * max(wa, wb)) + 0.01:
                return "channels disagree on the outline's width"
            if np.linalg.norm(ts[i] - ts[j]) > max(AGREE_M, AGREE_REL * max(np.linalg.norm(ts[i]), np.linalg.norm(ts[j]))) + 0.01:
                return "channels disagree on the shift"
    if abs(float(a["k"]) - float(np.mean(ks))) > 1e-3 or np.linalg.norm(np.asarray(a["t"], float) - np.mean(ts, axis=0)) > 0.01:
        return "the adopted similarity is not the channels' mean"
    return _frame_scale_problem(a)


def _fitted_problem(model: S.StructureModel, k: str, v: S.Param) -> str | None:
    """A photo-fitted parameter: its adoption record passes the rule again and is active in the reversible list, or
    it was carried by parents' active adoptions whose changes add up to its own change."""
    ads = [e for e in v.evidence if e.get("kind") == "adoption"]
    car = [e for e in v.evidence if e.get("kind") == "carried"]
    active = {a["id"]: a for a in _active(model)}
    if ads:
        e = ads[-1]
        why = check_adoption_record(e)
        if why:
            return f"photo-fitted: {why}"
        want = e["value"] if k == "height" else (e.get("roof_height") or {}).get("value") if k == "roof_height" else None
        if want is None:
            return f"an adoption record on {k}, which the adoption does not change"
        if abs(float(v.value) - float(want)) > 0.01:
            return "photo-fitted value differs from its adoption record"
        if e.get("id") not in active:
            return "photo-fitted by an adoption that is not active in the reversible list"
        return None
    if car:
        if v.map_value is None or abs(float(v.value) - float(v.map_value) - sum(float(e["dz"]) for e in car)) > 0.01:
            return "carried by its parents' adoptions, but its value is not the map's plus their changes"
        for e in car:
            a = active.get(e.get("adoption"))
            if a is None or a.get("part") != e.get("parent"):
                return f"carried by {e.get('parent')}'s adoption {e.get('adoption')}, which is not active"
        return None
    return f"{v.state} ({v.source}) without an adoption record (D15: >= 2 agreeing channels) or a carry record"


def _second_channel_problem(model: S.StructureModel, p: S.PartSpec, v: S.Param) -> str | None:
    """A verified parameter's second channel, rechecked from its own evidence records: an agreement record whose
    3-D line measurement (>= 3 photos, no snap scale correction) is within max(1 m, 5 %) of the skyline's value (the
    mean of its two masks' records) and within TOL of the map's value (the part's top, or its built eave)."""
    sky = [e for e in v.evidence if e.get("kind") == "skyline"]
    lrs = {e.get("parameter"): e for e in v.evidence if e.get("kind") == LINES}
    ags = [e for e in v.evidence if e.get("kind") == "agreement"]
    if not ags:
        return "verified without a second channel's agreement (the skyline alone; >= 2 channels are needed)"
    b = model.built_part(p.id)
    for e in ags:
        par = e.get("parameter")
        lr = lrs.get(par)
        svals = [x.get("values", {}).get(par) for x in sky]
        if lr is None or lr.get("value") is None or len(svals) < 2 or any(sv is None for sv in svals):
            continue
        sv, lv = float(np.mean(svals)), float(lr["value"])
        mapv = float(p.top) if par == "top" else (float(b.eave) if b is not None else float("nan"))
        if (abs(sv - lv) <= agree_tol(sv, lv) + 0.01 and abs(lv - mapv) <= max(2.0, 0.1 * mapv) + 0.01
                and lr.get("n_photos", 0) >= MIN_PHOTOS and not lr.get("scale_corrected") and not lr.get("vote")):
            return None
    return ("verified without a second channel agreeing (rechecked: 3-D lines within max(1 m, 5 %) of the skyline and "
            "within TOL of the map, >= 3 photos, no snap scale correction)")


def check_decisions(model: S.StructureModel, decisions: dict[str, Decision] | None = None) -> dict[str, Any]:
    """G4 / G9: every photo-verified or photo-fitted parameter carries the evidence the rules need, rechecked from the
    records: a verified one two masks, >= 3 photos with sky above, the 3-D line measurement agreeing with the skyline,
    no vote against it; a fitted one (D15) an adoption record that passes the rule again (check_adoption_record) and is
    active in the reversible list, or carry records adding up to its change; none while a frame ticket is open, no
    slender part verified or fitted on its own evidence, no fitted child of a part whose own change was not accepted,
    no parameter from a forbidden source, a 'height pending' stub source (default-style) never in any state but stub,
    a photo-fitted outline only under an active frame adoption that passes D15 ⑥ again, and every active adoption in
    the list matching its parameter. With the decisions, also their own records (sky photos, slender, contrary votes,
    agreement)."""
    bad = []
    frame_open = any(t.get("kind") == "frame" and t.get("open") for t in model.tickets)
    states = {p.id: p.params["height"].state for p in model.specs if "height" in p.params}
    for p in model.specs:
        for k, v in p.params.items():
            if v.source in S.FORBIDDEN_SOURCES:
                bad.append({"id": p.id, "param": k, "why": f"forbidden source {v.source}"})
            if v.source == st.PROV_DEFAULT and v.state != S.STATE_STUB:
                bad.append({"id": p.id, "param": k, "why": f"a 'height pending' stub source in state {v.state} (must stay stub)"})
            if v.source == PROV_AUTO and v.state not in (S.STATE_FITTED, S.STATE_VERIFIED, S.STATE_DISPUTED):
                bad.append({"id": p.id, "param": k, "why": f"a photo-fitted (auto) value in state {v.state}"})
            carried_stub = v.source == st.PROV_DEFAULT and any(e.get("kind") == "carried" for e in v.evidence)
            if v.source == PROV_AUTO or v.state == S.STATE_FITTED or carried_stub:
                # (a carried 'height pending' placeholder must add up to its parents' active adoptions too)
                why = _fitted_problem(model, k, v)
                if why:
                    bad.append({"id": p.id, "param": k, "why": why})
            if v.state not in (S.STATE_VERIFIED, S.STATE_FITTED):
                continue
            carried_only = (v.state == S.STATE_FITTED and not any(e.get("kind") == "adoption" for e in v.evidence)
                            and any(e.get("kind") == "carried" for e in v.evidence))
            minw = p.min_width()
            if (p.top - p.z0) / max(minw, 0.1) >= 2.0 and minw <= 10.0 and not carried_only:
                bad.append({"id": p.id, "param": k, "why": f"a slender part was {'fitted' if v.state == S.STATE_FITTED else 'verified'}"})
            if frame_open:
                bad.append({"id": p.id, "param": k, "why": f"{v.state} while the frame ticket is open"})
            if any("snap_scale" not in e for e in v.evidence):
                bad.append({"id": p.id, "param": k, "why": "evidence without its snap scale"})
            if v.state == S.STATE_VERIFIED:
                sky = [e for e in v.evidence if e.get("kind") == "skyline"]
                if len({e.get("mask") for e in sky}) < 2:
                    bad.append({"id": p.id, "param": k, "why": f"{v.state} without two sky masks' evidence"})
                if any(e.get("sky_above_photos", 0) < MIN_PHOTOS for e in sky):
                    bad.append({"id": p.id, "param": k, "why": f"{v.state} with < {MIN_PHOTOS} photos showing sky above"})
                if any(e.get("vote") for e in v.evidence if e.get("kind") not in ("carried", "adoption")):
                    bad.append({"id": p.id, "param": k, "why": "verified over a contrary vote in its own evidence "
                                f"{sorted({(e.get('kind'), e.get('vote')) for e in v.evidence if e.get('vote')})}"})
                why2 = _second_channel_problem(model, p, v)
                if why2:
                    bad.append({"id": p.id, "param": k, "why": why2})
            if v.state == S.STATE_FITTED:
                if p.parent is not None and states.get(p.parent) in (S.STATE_DISPUTED, S.STATE_STUB):
                    bad.append({"id": p.id, "param": k, "why": "fitted on a parent whose change is not accepted"})
                if v.map_value is None:
                    bad.append({"id": p.id, "param": k, "why": "fitted without the map value kept"})
    # D25 (photo-corrected): every record complete, every corrected parameter backed by one and equal to it
    # (crowd3d.correct.check_model); none while a frame ticket is open
    from . import correct as C
    bad.extend(C.check_model(model))
    if frame_open:
        bad.extend({"id": p.id, "param": k, "why": "photo-corrected while the frame ticket is open"}
                   for p in model.specs for k, v in p.params.items() if v.state == S.STATE_CORRECTED)
        bad.extend({"id": a["id"], "why": "a tilt while the frame ticket is open"} for a in _active(model, "d25-tilt"))
    frames = _active(model, "frame")
    auto_outlines = [p.id for p in model.specs if p.outline_source == PROV_AUTO]
    if auto_outlines and not frames:
        bad.append({"id": auto_outlines[0], "param": "outline",
                    "why": f"photo-fitted (auto) outlines ({len(auto_outlines)}) without an active frame adoption"})
    for a in frames:
        why = check_frame_record(a.get("record") or {})
        if why:
            bad.append({"id": a["id"], "param": "outline", "why": f"frame adoption: {why}"})
    for a in _active(model, "part-height"):
        try:
            h = model.part(a["part"]).params["height"]
        except StopIteration:
            bad.append({"id": a["id"], "why": "an adoption of a part that does not exist"})
            continue
        if h.state != S.STATE_FITTED or h.source != PROV_AUTO or abs(float(h.value) - float(a["value"])) > 0.01:
            bad.append({"id": a["part"], "param": "height", "why": f"adoption {a['id']} does not match the parameter"})
        if a.get("roof_height"):
            rh = model.part(a["part"]).params["roof_height"]
            if rh.state != S.STATE_FITTED or rh.source != PROV_AUTO or abs(float(rh.value) - float(a["roof_height"]["value"])) > 0.01:
                bad.append({"id": a["part"], "param": "roof_height", "why": f"adoption {a['id']} does not match the roof height"})
    if decisions:
        for pid, d in decisions.items():
            if d.state == S.STATE_VERIFIED and d.n_sky < MIN_PHOTOS:
                bad.append({"id": pid, "why": "verified with too few sky photos"})
            if d.state == S.STATE_VERIFIED and d.slender:
                bad.append({"id": pid, "why": "a slender part verified"})
            if d.state == S.STATE_VERIFIED and d.votes:
                bad.append({"id": pid, "why": f"verified over contrary votes {d.votes}"})
            if d.state == S.STATE_VERIFIED and not any(a.get("agree") for a in d.agreement.values()):
                bad.append({"id": pid, "why": "verified without two channels agreeing"})
            if d.adopt and check_adoption_record(d.adopt):
                bad.append({"id": pid, "why": f"an adoption record that fails D15: {check_adoption_record(d.adopt)}"})
    return {"violations": bad, "passed": not bad}


# ---------------------------------------------------------------- the whole judge, parents first (D15 ⑤, ⑥)


@dataclass
class PhotoEvidence:
    """The photo evidence of one landmark for judge_landmark(), in the structure's ENU frame (after the snap): posed
    cameras and sky masks per mask name (skyline), SfM points with their quality (SfM channel, frame gate), the 3-D
    lines measured once (a crowd3d.lines3d.LinesResult: re-assigned to every version of the structure, never
    re-triangulated) and the snap scale. profiles_at_map: optional precomputed profiles of the unchanged map model."""
    cameras: list
    masks: dict[str, dict[int, np.ndarray]]
    points: np.ndarray | None = None
    quality: np.ndarray | None = None
    lines: Any = None
    snap_scale: float = 1.0
    max_cameras_per_part: int = 60
    lines_params: Any = None
    profiles_at_map: dict | None = None

    def profiles(self, model: S.StructureModel, ids: list[str] | None = None, log=None) -> dict[str, dict[str, Any]]:
        from . import skyline as K
        return {name: K.SkylineProblem(model, self.cameras, m, max_cameras_per_part=self.max_cameras_per_part)
                .profile_all(ids, log=log) for name, m in self.masks.items()}

    def sfm(self, model: S.StructureModel) -> dict[str, dict[str, Any]]:
        return {} if self.points is None or not len(self.points) else sfm_channel(model, self.points, self.quality)

    def track_points(self) -> np.ndarray | None:
        """The SfM points seen in >= 3 photos each (all of them without track lengths), as D25 uses them."""
        if self.points is None or not len(self.points):
            return None
        pts = np.asarray(self.points, float)
        if self.quality is not None and len(self.quality) == len(pts):
            pts = pts[np.asarray(self.quality)[:, 0] >= 3]
        return pts

    def lines_evidence(self, model: S.StructureModel) -> dict[str, list[dict[str, Any]]] | None:
        """The 3-D line records for this version of the structure: the lines re-assigned to its geometry (a
        LinesResult or a bare Lines3D), or a callable(model) -> records, or fixed records."""
        if self.lines is None:
            return None
        if isinstance(self.lines, dict):
            return self.lines                          # records given directly (fixed: they do not follow the model)
        if callable(self.lines):
            return self.lines(model)
        from . import lines3d as L3
        res = self.lines
        if isinstance(res, L3.Lines3D):
            res = L3.LinesResult({}, res, {}, {}, "", {}, self.track_points())
        return L3.remeasure(res, model, self.lines_params or L3.DEFAULT_LINES, self.snap_scale).evidence()

    def frame_lines(self) -> Any:
        return None if self.lines is None or isinstance(self.lines, dict) or callable(self.lines) else self.lines


def judge_landmark(model: S.StructureModel, ev: PhotoEvidence, auto_adopt: bool = AUTO_ADOPT, log=None,
                   max_rounds: int = 8) -> dict[str, Any]:
    """The judge of one landmark with D15 (design §2.4.3, §6 D15): 1. the frame gate and each channel's similarity;
    the frame adopted by the same rule (⑥) and the gate rerun on the adopted outline; 2. every part judged at the map
    value (on the adopted frame) and the adoptions the rule allows applied parents first; 3. the parts an adoption
    carried re-judged on their parent's new value (new profiles, SfM channel and line assignment for the new
    geometry), adopted in turn, until nothing changes (⑤). Returns the decisions, the frame, the adoptions (the
    reversible list), the rounds and the timings."""
    T: dict[str, float] = {}
    t0 = time.perf_counter()
    say = log or (lambda *a: None)
    pts = ev.points
    has_pts = pts is not None and len(pts) > 0
    gate = frame_gate(pts, model.main_rings, ev.quality) if has_pts else \
        {"passed": True, "failing": [], "note": "no registered points: the frame gate does not apply"}
    chans = frame_channels(model, pts if has_pts else None, ev.quality, ev.frame_lines())

    def eave_check(trial: S.StructureModel) -> dict[str, Any]:
        """④ for the frame: the main body's eave measured on the trial outline (skyline profiles of the main body
        with every mask, the 3-D lines re-assigned to the trial geometry)."""
        mb = S.main_body(trial)
        if mb is None:
            return {"main_body": None, "map_eave": None, "channels": {}}
        lev_t = ev.lines_evidence(trial)
        dec_t = decide(trial, ev.profiles(trial, [mb.id]), None, None, ev.snap_scale, lines=lev_t)
        return scale_free_refs(trial, dec_t, lev_t, ev.snap_scale)

    fd = decide_frame(model, gate, chans, ev.snap_scale, pts if has_pts else None, ev.quality,
                      eave_check=eave_check if ev.cameras else None)
    frame = {**{k: v for k, v in gate.items() if k != "edges"}, "similarity": chans.get(SFM),
             "similarity_lines": chans.get(LINES), "d15": {k: v for k, v in fd.items() if k != "adopt"}}
    fa = None
    if auto_adopt and fd.get("adopt"):
        fa = adopt_frame(model, fd, gate)
        if fa is not None:
            g2 = frame_gate(pts, model.main_rings, ev.quality)
            c2 = frame_channels(model, pts, ev.quality, ev.frame_lines())
            frame = {**{k: v for k, v in g2.items() if k != "edges"}, "similarity": c2.get(SFM),
                     "similarity_lines": c2.get(LINES), "adopted": fa["id"],
                     "before_adoption": {k: v for k, v in frame.items()}}
            say(f"D15: frame adopted {fa['id']}: k {fa['value']['k']}, t {fa['value']['t']} (gate on it: {g2['passed']})")
    else:
        say(f"D15 frame: {fd.get('why') or 'adoptable (switch off)'}")
    T["frame_s"] = round(time.perf_counter() - t0, 2)
    t = time.perf_counter()
    profiles = ev.profiles_at_map if (ev.profiles_at_map is not None and fa is None) else ev.profiles(model, log=log)
    T["profiles_s"] = round(time.perf_counter() - t, 2)
    sfm = ev.sfm(model)
    lev = ev.lines_evidence(model)
    dec = decide(model, profiles, sfm, frame, ev.snap_scale, lines=lev, points=ev.track_points())
    refs = scale_free_refs(model, dec, lev, ev.snap_scale)
    apply_decisions(model, dec, frame, auto_adopt=False)
    rounds = [{"round": 0, "judged": sorted(dec), "adopted": []}]
    pending = []
    if auto_adopt:
        n0 = len(model.adoptions)
        pending = adopt_decisions(model, dec)
        rounds[0]["adopted"] = [a["id"] for a in model.adoptions[n0:]]
        apply_decisions(model, dec, frame, auto_adopt=False)          # close the adopted parts' tickets
    r = 0
    while pending and r < max_rounds:
        r += 1
        ids = [pid for pid in pending if model.built_part(pid) is not None]
        if not ids:
            break
        t = time.perf_counter()
        prof = ev.profiles(model, ids)
        sfm = ev.sfm(model)
        lev = ev.lines_evidence(model)
        d2 = decide(model, prof, sfm, frame, ev.snap_scale, lines=lev, scale_refs=refs, points=ev.track_points())
        for d in d2.values():
            d.rejudged = r
        dec.update(d2)
        apply_decisions(model, dec, frame, auto_adopt=False)
        n0 = len(model.adoptions)
        pending = adopt_decisions(model, d2)
        apply_decisions(model, dec, frame, auto_adopt=False)
        rounds.append({"round": r, "judged": ids, "adopted": [a["id"] for a in model.adoptions[n0:]],
                       "seconds": round(time.perf_counter() - t, 2)})
        say(f"D15 round {r}: re-judged {ids} on their parents' new values; adopted {rounds[-1]['adopted']}")
    T["total_s"] = round(time.perf_counter() - t0, 2)
    return {"decisions": dec, "frame": frame, "frame_decision": fd, "frame_adoption": fa, "adoptions": model.adoptions,
            "rounds": rounds, "scale_refs": refs, "timings_s": T}
