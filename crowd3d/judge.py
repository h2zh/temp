"""Judging structure parts against the photos (docs/STRUCTURE_FIRST.md §2.2, §2.4.2-§2.4.3, §2.8).

Photos never shape a part here: they check the map. Everything is decided AT THE MAP VALUE.

- frame_gate(): the per-edge residual gate (G11). Every outline edge >= 5 m long with >= 200 wall points (2-20 m
  high, nearest edge within 10 m, points far inside counted too) must have a median distance <= 1.5 m. When it
  fails, a frame ticket opens and no parameter is verified or fitted (the evidence is still recorded).
- frame_similarity(): one similarity for the whole building (outline scale k about its centre, shift t) from the
  wall planes (least squares on the edges' median wall positions). Only reported and put in the ticket: a
  change of the map's outline is a curator's decision (D15 / D16); single-sided data cannot separate k from t,
  so it needs walls facing >= 180 deg of directions including two opposite ones.
- sfm_channel(): the SfM-point channel: the 0.98 quantile of point heights over the part's exposed area (minus the
  parts standing on it and 1.5 m around taller neighbours) against the model's own surface over the same area.
  Points can only say "the map is too low" (a roof nobody photographed has no points; trees and neighbours sit
  above parts too, so it never decides alone).
- decide(): the rules of §2.4.3 per part, from one or two sky masks' skyline profiles (crowd3d.skyline), the
  SfM channel and the 3-D line channel (crowd3d.lines3d); tolerance TOL = max(2 m, 10 % of the map top); each
  channel votes only for the direction it can observe (skyline: down, and a weak "up?"; SfM points: up; 3-D lines
  measure eave / apex / ridge heights directly: up and down); a skyline alone never raises anything ("up?" counts
  only with another "up"); slender parts (own height / narrowest width >= 2 and width <= 10 m) never move;
  verification needs >= 2 CHANNELS agreeing (VERIFY_MIN_CHANNELS): the skyline (two masks agreeing, a stable profile
  off its bounds) and a 3-D line measurement of the same quantity (eave <-> 'eave', apex / ridge <-> 'roof'), within
  max(1 m, 5 %) of each other and the lines within TOL of the map; plus the frame gate passed and a snap scale
  correction <= 3 %. "Off the map value" is judged on the profile's sub-grid minimum with half a grid step of
  margin (a 5 % grid point can sit a step nearer the map than the truth), and nothing is verified while any channel
  (a line measurement of the eave included) votes against. States: map, photo-verified, disputed (ticket),
  photo-fitted (D15 auto-adopt, off by default: reported as 'would fit'). One strong single "down" (skyline, every
  mask) or a single 3-D line vote on a part that is not slender opens a low-strength ticket (state and value stay the
  map's). Under a snap scale correction > 3 % a 3-D line vote cannot decide alone: it counts only beside another
  strong vote in its direction. Tickets carry the channels' measured values and which of them agree within
  max(1 m, 5 %). A 'height pending' stub (D13) always stays a stub: evidence and tickets only.
- apply_decisions() writes states, evidence and tickets into the structure's parameters (massing.json v2);
  check_decisions() is G4 / G9: no verified or fitted value without the evidence the rules need (for a verified one:
  two masks, the line measurement and the agreement record, rechecked).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from . import stylize as st
from . import structure as S

FRAME_EDGE_MIN_M = 5.0
FRAME_EDGE_MIN_POINTS = 200
FRAME_MAX_MEDIAN_M = 1.5
FRAME_K_TOL = 0.03
FRAME_T_TOL_M = 1.5
SNAP_SCALE_TOL = 0.03
MIN_PHOTOS = 3
VERIFY_MIN_CHANNELS = 2      # channels (skyline, lines3d) that must agree before a value is 'photo-verified'
AGREE_M = 1.0                # two channels agree within max(AGREE_M, AGREE_REL x value) (design D15)
AGREE_REL = 0.05
LINES = "lines3d"
LINES_KEY = {"eave": "eave", "roof": "top"}     # skyline profile parameter -> 3-D line measurement


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


def frame_similarity(points: np.ndarray, rings: list[np.ndarray], quality: np.ndarray | None = None,
                     min_points: int = FRAME_EDGE_MIN_POINTS) -> dict[str, Any]:
    """Whole-building similarity from the wall planes (design §2.2 step 2, the wall-plane method): per edge with
    enough wall points (3-16 m high), the median wall position along its outward normal n_i; least squares for
    position_i = k * d_i + n_i . t, d_i the edge's distance from the outline's centroid. 'measured' needs walls facing
    >= 180 deg of directions with two opposite ones (else scale and shift cannot be separated)."""
    ring = S.ccw(rings[0])
    c = st.polygon_centroid(ring)
    xy = _wall_points(points, quality, 3.0, 16.0)
    k, d, sgn = _edge_assignment(xy, ring)
    A, y, w, used, normals = [], [], [], [], []
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
        A.append([di, n[0], n[1]])
        y.append(pos)
        w.append(math.sqrt(mid.sum()))
        normals.append(math.degrees(math.atan2(n[1], n[0])) % 360)
        used.append({"edge": e, "points": int(mid.sum()), "offset_m": round(pos - di, 2)})
    out: dict[str, Any] = {"edges": used, "centre": [round(float(v), 3) for v in c], "measured": False}
    if len(A) < 3:
        out["reason"] = f"{len(A)} edges with wall points (< 3)"
        return out
    A, y, w = np.array(A), np.array(y), np.array(w)
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
    return out


def sfm_channel(model: S.StructureModel, points: np.ndarray, quality: np.ndarray | None = None, cell: float = 0.5,
                min_points: int = 50) -> dict[str, dict[str, Any]]:
    """Per built part: the SfM points' 0.98 height quantile over its exposed area vs the model's own surface there."""
    P = np.asarray(points, float)
    if quality is not None and len(quality) == len(P):
        P = P[np.asarray(quality)[:, 0] >= 3]
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
        row: dict[str, Any] = {"exposed_m2": round(float(m.sum()) * cell * cell, 1), "points": 0, "q98": None, "model_q98": None}
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
    would_fit: dict[str, float] | None = None
    evidence: list = field(default_factory=list)
    ticket: str | None = None            # 'dispute' (>= 2 channels) or 'low' (one strong single-channel vote)
    skyline_ok: dict[str, bool] = field(default_factory=dict)    # per parameter: the skyline's own conditions
    agreement: dict[str, dict[str, Any]] = field(default_factory=dict)   # per parameter: skyline vs 3-D lines
    measured: dict[str, float] = field(default_factory=dict)     # channel -> the top it measures (tickets)
    agreeing: list = field(default_factory=list)                 # channels agreeing within max(1 m, 5 %)
    agreed: float | None = None                                  # ... and their mean


def decide(model: S.StructureModel, profiles: dict[str, dict[str, Any]], sfm: dict[str, dict[str, Any]] | None = None,
           frame: dict[str, Any] | None = None, snap_scale: float = 1.0, auto_adopt: bool = False,
           lines: dict[str, list[dict[str, Any]]] | None = None,
           verify_channels: int = VERIFY_MIN_CHANNELS) -> dict[str, Decision]:
    """The §2.4.3 rules for every built part. profiles: {mask name: {part id: skyline.PartProfile}} (one or two
    masks; verification needs two); sfm: judge.sfm_channel(); lines: {part id: lines3d evidence records}
    (crowd3d.lines3d.LinesResult.evidence()). verify_channels: channels that must agree for a verification (2; 1 is the
    skyline-only rule of the first prototype, kept for comparison only)."""
    masks = list(profiles)
    sfm = sfm or {}
    frame_open = bool(frame and not frame.get("passed", True))
    scaled = abs(snap_scale - 1.0) > SNAP_SCALE_TOL
    out: dict[str, Decision] = {}
    for b in model.built:
        pid = b.id
        profs = [profiles[m][pid] for m in masks if pid in profiles[m]]
        if not profs:
            continue
        top, z0 = b.top, b.z0
        tol = max(2.0, 0.1 * top)
        minw = max(b.spec.min_width(), 0.1)
        hw = (top - z0) / minw
        slender = hw >= 2.0 and minw <= 10.0
        n_vis = min(p.n_vis for p in profs)
        n_sky = min(p.n_sky for p in profs)
        fwd0 = [p.fwd_px for p in profs]

        def pinfo(key: str, p) -> dict[str, Any]:
            pp = p.params[key]
            G = pp["grid"]
            k = G.index(pp["fr"]["argmin"])
            k1 = int(np.argmin(np.abs(np.array(G) - 1.0)))
            if key == "scale":
                L = minw / 2
                val, mapv = L * pp["fr"]["argmin"], L
                topv = pp["metres"][k]["top"]
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
                    "map": round(float(mapv), 2), "top": topv, "top_sub": top_sub, "bound": pp["fr"]["at_bound"],
                    "lo_bound": pp["fr"]["lo_bound"], "hi_bound": pp["fr"]["hi_bound"],
                    "ci90_m": round((pp["fr"]["p95"] - pp["fr"]["p05"]) * L, 2), "argmin": pp["fr"]["argmin"],
                    "robust": abs(pp["fr"]["argmin"] - pp["fo"]["argmin"]) <= step + 1e-9, "n_cams": pp["n_cams"]}

        keys = sorted(set.intersection(*[set(p.params) for p in profs])) if profs else []
        params = {key: {m: pinfo(key, profiles[m][pid]) for m in masks if pid in profiles[m]} for key in keys}
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
            if verify_channels <= 1:
                verified[key] = skyline_ok[key]
                continue
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
        # what each channel measures for the top (tickets carry it): skyline (its profile's top), SfM points (q98),
        # 3-D lines (apex / ridge, or the eave with the map's roof on it)
        measured: dict[str, float] = {}
        if tops:
            kt = "roof" if "roof" in tops else "eave"           # the top itself (a flat part's eave is its top)
            if all(params[kt][m]["top_sub"] is not None for m in params[kt]):
                measured["skyline"] = round(float(np.mean([params[kt][m]["top_sub"] for m in params[kt]])), 2)
        if s.get("q98") is not None:
            measured["sfm-points"] = round(float(s["q98"]), 2)
        if ltop is not None:
            measured[LINES] = round(float(ltop), 2)
        agree_ch, agree_val = [], None
        would = None
        ticket = None
        if n_vis < MIN_PHOTOS:
            state, why = S.STATE_MAP, f"model skyline in {n_vis} (< {MIN_PHOTOS}) photos at the map value: photos insufficient"
        elif n_up >= 2 or n_down >= 2:
            dirn = "up" if n_up >= 2 else "down"
            # the channels that measure a value in the dispute's direction (the weak 'skyline-up?' does not)
            vals: list[tuple[str, float]] = []
            if dirn == "down" and isinstance(votes.get("skyline-down"), float):
                vals.append(("skyline", top + votes["skyline-down"]))
            if dirn == "up" and "sfm-up" in votes and s.get("q98") is not None:
                vals.append(("sfm-points", float(s["q98"])))
            if f"lines-{dirn}" in (up if dirn == "up" else down) and ltop is not None:
                vals.append((LINES, float(ltop)))
            agree_ch, agree_val = agreeing(vals)
            ok = bool(agree_ch) and stable and not slender and not scaled and not frame_open
            would = {"top": agree_val} if ok else None
            state, ticket = S.STATE_DISPUTED, "dispute"
            why = f"{dirn}: " + ", ".join(f"{k}={v}" for k, v in votes.items())
            if agree_ch:
                why += f"; {' + '.join(agree_ch)} agree on {agree_val} m"
            why += "; D15 would adopt" if ok else "; ticket"
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
            elif verify_channels >= 2 and any(skyline_ok.values()):
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
            state, ticket, would = S.STATE_MAP, None, None
        if b.spec.params["height"].source == st.PROV_DEFAULT:
            # a 'height pending' stub (D13) is never verified, disputed or fitted: evidence and tickets only
            if state != S.STATE_MAP or ticket:
                why = f"height pending stub: state stays stub (would be {state}): {why}"
            state, would = S.STATE_STUB, None
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
                             "vote": "up" if "sfm-up" in votes else None, "snap_scale": round(snap_scale, 4)})
        evidence.extend(lrecs)
        if state == S.STATE_VERIFIED:
            for k, ok in verified.items():
                if ok and k in agreement:
                    evidence.append({"kind": "agreement", "channels": ["skyline", LINES], **agreement[k],
                                     "vote": None, "snap_scale": round(snap_scale, 4)})
        out[pid] = Decision(pid, state, why, votes, verified, params, n_vis, n_sky, slender,
                            would, evidence, ticket, skyline_ok, agreement, measured, agree_ch, agree_val)
    return out


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


def apply_decisions(model: S.StructureModel, decisions: dict[str, Decision], frame: dict[str, Any] | None = None,
                    auto_adopt: bool = False) -> S.StructureModel:
    """Write the decisions into the structure's parameters (states, evidence, tickets) and the frame result into
    its registry. With auto_adopt (D15, off by default) a disputed part whose conditions hold takes the photos'
    top as 'photo-fitted' (the map value kept beside it); otherwise values never change here."""
    model.tickets = [t for t in model.tickets if t.get("kind") not in ("part", "frame")]
    if frame is not None:
        model.frame = {k: v for k, v in frame.items() if k != "edges"}
        if not frame.get("passed", True) or (frame.get("similarity") or {}).get("ticket"):
            model.tickets.append({"id": f"T-{model.landmark}-frame", "kind": "frame", "open": True,
                                  "failing_edges": frame.get("failing", []), "similarity": frame.get("similarity"),
                                  "scale_hypothesis": scale_hypothesis(model, decisions, frame.get("similarity"))})
    n = 0
    fitted = False
    for p in model.specs:
        d = decisions.get(p.id)
        if d is None:
            continue
        h = p.params["height"]
        h.evidence = list(d.evidence)
        stub = h.source == st.PROV_DEFAULT                # a 'height pending' stub keeps its state: ticket and evidence only
        tk = d.ticket or ("dispute" if d.state == S.STATE_DISPUTED else None)
        if tk:
            n += 1
            tid = f"T-{model.landmark}-{n}"
            h.ticket = tid
            if not stub and d.state == S.STATE_DISPUTED:
                h.state = S.STATE_DISPUTED
            # the ticket carries what the channels measured (v0 never adopts it: a curator decides) and which of them
            # agree within max(1 m, 5 %)
            model.tickets.append({"id": tid, "kind": "part", "part": p.id, "param": "height", "open": True,
                                  "votes": d.votes, "why": d.why, "map_value": round(float(p.top), 2),
                                  "measured": dict(d.measured), "agreeing_channels": list(d.agreeing),
                                  "agreed_value": d.agreed,
                                  "strength": "low" if tk == "low" else len([v for v in d.votes if not v.endswith("?")])})
            if auto_adopt and d.would_fit and not stub and d.state == S.STATE_DISPUTED:
                h.map_value, h.value, h.source, h.state = h.value, d.would_fit["top"], st.PROV_FITTED, S.STATE_FITTED
                fitted = True
        elif d.state == S.STATE_VERIFIED and not stub:
            if d.verified.get("eave") or d.verified.get("roof"):
                h.state = S.STATE_VERIFIED
            if d.verified.get("roof") and p.params["roof_height"].source not in (st.PROV_DEFAULT,):
                # a verified roof height carries the evidence that verified it (G4 checks every verified parameter)
                p.params["roof_height"].state = S.STATE_VERIFIED
                p.params["roof_height"].evidence = list(d.evidence)
        p.problems = [x for x in p.problems if not x.startswith("judge:")] + [f"judge: {d.state}: {d.why}"]
    if fitted:
        # the solids follow an adopted value (a 'photo-fitted' top drawn at the map value would misreport the shape),
        # and G10 is resolved again for what stands on them
        model.built = S.build_all(model.specs, model.params)
        S.resolve_support(model)
    return model


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


def check_decisions(model: S.StructureModel, decisions: dict[str, Decision] | None = None,
                    verify_channels: int = VERIFY_MIN_CHANNELS) -> dict[str, Any]:
    """G4 / G9: every photo-verified or photo-fitted parameter carries the evidence the rules need (two masks,
    >= 3 photos with sky above, snap scale recorded; a verified one also the 3-D line measurement agreeing with the
    skyline, rechecked from the records), none while a frame ticket is open, no slender part verified or fitted,
    nothing verified over a vote against it in its own evidence, no fitted child of a part whose own change was not
    accepted, no parameter from a forbidden source, and a 'height pending' stub source (default-style) never in any
    state but stub. With the decisions, also their own records (sky photos, slender, contrary votes, agreement)."""
    bad = []
    frame_open = any(t.get("kind") == "frame" and t.get("open") for t in model.tickets)
    states = {p.id: p.params["height"].state for p in model.specs if "height" in p.params}
    for p in model.specs:
        for k, v in p.params.items():
            if v.source in S.FORBIDDEN_SOURCES:
                bad.append({"id": p.id, "param": k, "why": f"forbidden source {v.source}"})
            if v.source == st.PROV_DEFAULT and v.state != S.STATE_STUB:
                bad.append({"id": p.id, "param": k, "why": f"a 'height pending' stub source in state {v.state} (must stay stub)"})
            if v.state not in (S.STATE_VERIFIED, S.STATE_FITTED):
                continue
            minw = p.min_width()
            if (p.top - p.z0) / max(minw, 0.1) >= 2.0 and minw <= 10.0:
                bad.append({"id": p.id, "param": k, "why": f"a slender part was {'fitted' if v.state == S.STATE_FITTED else 'verified'}"})
            if frame_open:
                bad.append({"id": p.id, "param": k, "why": f"{v.state} while the frame ticket is open"})
            sky = [e for e in v.evidence if e.get("kind") == "skyline"]
            if len({e.get("mask") for e in sky}) < 2:
                bad.append({"id": p.id, "param": k, "why": f"{v.state} without two sky masks' evidence"})
            if any(e.get("sky_above_photos", 0) < MIN_PHOTOS for e in sky):
                bad.append({"id": p.id, "param": k, "why": f"{v.state} with < {MIN_PHOTOS} photos showing sky above"})
            if any("snap_scale" not in e for e in v.evidence):
                bad.append({"id": p.id, "param": k, "why": "evidence without its snap scale"})
            if v.state == S.STATE_VERIFIED and any(e.get("vote") for e in v.evidence):
                bad.append({"id": p.id, "param": k, "why": "verified over a contrary vote in its own evidence "
                            f"{sorted({(e.get('kind'), e.get('vote')) for e in v.evidence if e.get('vote')})}"})
            if v.state == S.STATE_VERIFIED and verify_channels >= 2:
                why2 = _second_channel_problem(model, p, v)
                if why2:
                    bad.append({"id": p.id, "param": k, "why": why2})
            if v.state == S.STATE_FITTED:
                if p.parent is not None and states.get(p.parent) in (S.STATE_DISPUTED,):
                    bad.append({"id": p.id, "param": k, "why": "fitted on a parent whose change is not accepted"})
                if v.map_value is None:
                    bad.append({"id": p.id, "param": k, "why": "fitted without the map value kept"})
    if decisions:
        for pid, d in decisions.items():
            if d.state == S.STATE_VERIFIED and d.n_sky < MIN_PHOTOS:
                bad.append({"id": pid, "why": "verified with too few sky photos"})
            if d.state == S.STATE_VERIFIED and d.slender:
                bad.append({"id": pid, "why": "a slender part verified"})
            if d.state == S.STATE_VERIFIED and d.votes:
                bad.append({"id": pid, "why": f"verified over contrary votes {d.votes}"})
            if d.state == S.STATE_VERIFIED and verify_channels >= 2 and not any(a.get("agree") for a in d.agreement.values()):
                bad.append({"id": pid, "why": "verified without two channels agreeing"})
    return {"violations": bad, "passed": not bad}


def apply_frame(model: S.StructureModel, k: float, t: tuple[float, float], centre: np.ndarray | None = None,
                note: str = "curator") -> S.StructureModel:
    """A curator-approved whole-building similarity (scale k about the outline's centre, then shift t) applied
    rigidly to every part and the main outline (never one edge or one part alone); outlines become 'curator'."""
    c = np.asarray(centre if centre is not None else st.polygon_centroid(S.ccw(model.main_rings[0])), float)
    tt = np.asarray(t, float)

    def tf(r):
        return c + (np.asarray(r, float) - c) * k + tt

    for p in model.specs:
        p.outer = tf(p.outer)
        p.holes = [tf(h) for h in p.holes]
        p.outline_source = st.PROV_CURATOR
        p.problems.append(f"{note}: whole-building similarity k={k:.4f}, t=({tt[0]:.2f}, {tt[1]:.2f}) m")
    model.main_rings = [tf(r) for r in model.main_rings]
    model.built = S.build_all(model.specs, model.params)
    model.frame = {**model.frame, "curator_similarity": {"k": k, "t": [float(tt[0]), float(tt[1])], "centre": c.tolist(),
                                                         "note": note}}
    model.tickets = [dict(tk, open=False, resolved_by=note) if tk.get("kind") == "frame" else tk for tk in model.tickets]
    return model


def decision_table(decisions: dict[str, Decision], model: S.StructureModel) -> list[dict[str, Any]]:
    rows = []
    for b in model.built:
        d = decisions.get(b.id)
        if d is None:
            continue
        rows.append({"id": b.id, "name": b.spec.name, "roof": b.roof.shape, "z0": round(b.z0, 2), "eave": round(b.eave, 2),
                     "top": round(b.top, 2), "min_width_m": round(b.spec.min_width(), 2), "slender": d.slender,
                     "n_vis": d.n_vis, "n_sky": d.n_sky,
                     "profile": {k: {m: f"{v['val']}{'*' if v['bound'] else ''}" for m, v in pm.items()} for k, pm in d.params.items()},
                     "votes": d.votes, "state": d.state, "why": d.why, "would_fit": d.would_fit, "ticket": d.ticket})
    return rows
