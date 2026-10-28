"""Structural-detail metrics of a stylized landmark model: how much structure it has, and where each number and
element comes from (OpenStreetMap, conventions, placeholders, or the photos).

Standalone: numpy + the standard library. Inputs are the two artefacts every structure-first run already writes:
  * the structure registry (StructureModel.to_registry(), massing.json v2): parts, params {value, state, source,
    evidence}, outline_source, shape, adoptions, tickets;
  * the stylized GLB: node extras (kind, part, provenance, geometry, observed, relief[] ...) and the meshes.
The registry is optional: without it the GLB scene extras ("structure.parts") are used (no evidence, so photo claims
there count as unbacked).

Main entry: structure_metrics(registry, glb) -> dict (census, shares, detail, honesty, flat benchmark metrics); glb is
the GLB's bytes or a path. crowd3d.stylize puts it in the summary ("structure_metrics") and compact(...) of it in the
GLB's scene extras. METRIC_SPECS lists the flat metrics with name, unit, direction and description, the way
scripts/benchmark.py lists its metrics.

Provenance classes (one per number / element / triangle share):
  osm             value from an OpenStreetMap tag (registry state 'map', also 'disputed': the map value is kept while
                  photos disagree; reported as a sub-share)
  convention      value from a modelling convention: registry state 'map-convention' (e.g. s3db-shape-default roof
                  height), and the S3DB spec's default when the tag is absent (osm-default-ground: no min_height ->
                  on the ground, no roof:shape -> flat with no roof height) -- the same class for a defaulted height
                  as for the defaulted roof shape (flagged 'osm-default')
  stub            placeholder (registry state 'stub', source default-style): no fact behind it
  photo-verified  map value kept, photos agree (>= 2 channels); adds no new number to the model
  photo-fitted    map value replaced by a photo fit (D15 auto-adoption or curator-approved fit)
  photo-measured  element or number that exists only because photos measured it (facade relief, carved heights; a
                  facade element built on the points alone under D24 -- provenance 'photo-measured (points, D24)' --
                  is in this class, counted again apart: facade_elements_built_d24, photo_elements_d24)
  curator         value entered by a curator (none today)
  unbacked        anything else: unknown provenance, or a photo claim without its evidence. Never rewarded; the
                  count is an honesty metric that must stay 0.
"""
from __future__ import annotations

import json
import math
import struct
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

CLASSES = ("osm", "convention", "stub", "photo-verified", "photo-fitted", "photo-corrected", "photo-measured", "curator",
           "unbacked")
PHOTO_CLASSES = ("photo-verified", "photo-fitted", "photo-corrected", "photo-measured")
PHOTO_VALUE_CLASSES = ("photo-fitted", "photo-corrected", "photo-measured")      # the number itself comes from photos
D25_KINDS = ("d25-height", "d25-roof", "d25-tilt", "d25-block")

STATE_CLASS = {
    "map": "osm", "disputed": "osm", "map-convention": "convention", "stub": "stub",
    "photo-verified": "photo-verified", "photo-fitted": "photo-fitted", "photo-fitted (auto)": "photo-fitted",
    "photo-corrected": "photo-corrected", "carved": "photo-measured", "curator": "curator",
}
OUTLINE_CLASS = {
    "osm-s3db": "osm", "osm-footprint": "osm", "osm-outline": "osm", "osm-s3db-support": "osm",
    "photo-fitted": "photo-fitted", "photo-fitted (auto)": "photo-fitted", "curator": "curator",
    "default-style": "stub",
}
SHAPE_CLASS = {"osm-s3db": "osm", "osm-default-ground": "convention", "s3db-shape-default": "convention",
               "osm-roof-angle": "osm", "default-style": "stub", "curator": "curator",
               "photo-fitted": "photo-fitted", "photo-fitted (auto)": "photo-fitted", "photo-measured": "photo-measured"}
OSM_DEFAULT_SOURCES = {"osm-default-ground"}                     # S3DB defaults: tag absent -> ground / flat
PRECEDENCE = ("unbacked", "stub", "photo-measured", "photo-corrected", "photo-fitted", "curator", "photo-verified",
              "convention", "osm")
CURVED_ROOFS = {"dome", "onion", "round", "cone", "spherical"}
# evidence channel kinds that measure geometry (the judge's 'agreement' records only summarise them)
CHANNEL_KINDS = {"skyline", "sfm-points", "lines3d", "planes", "carving"}
# facade elements (relief probe guards G-R1..G-R4): >= 2 channels, >= 6 photos per split half, >= 1 line-confirmed edge
MIN_PHOTOS_PER_HALF = 6
NUM_FIELDS = ("u0", "u1", "z_bottom", "z_top")
DEPTH_FIELDS = ("depth", "projection", "level")
COORD_TOL_M = 0.05          # distinct-coordinate tolerance for the photo-DOF count
CREASE_DEG = 20.0           # dihedral angle at/above which an edge is a structural line (below: smooth surface)
COPLANAR_DEG = 1.0
MIN_EDGE_M = 0.05


def strongest(classes) -> str:
    cs = set(classes)
    for c in PRECEDENCE:
        if c in cs:
            return c
    return "unbacked"


# ---------------------------------------------------------------- GLB


def read_glb(glb: bytes | str | Path) -> tuple[dict[str, Any], bytes]:
    b = bytes(glb) if isinstance(glb, (bytes, bytearray, memoryview)) else Path(glb).read_bytes()
    magic, _ver, length = struct.unpack_from("<4sII", b, 0)
    if magic != b"glTF":
        raise ValueError("not a GLB")
    off, js, binc = 12, None, b""
    while off < length:
        cl, ct = struct.unpack_from("<I4s", b, off)
        off += 8
        if ct == b"JSON":
            js = json.loads(b[off:off + cl])
        elif ct == b"BIN\x00":
            binc = b[off:off + cl]
        off += cl
    return js, binc


_COMP = {5126: np.float32, 5125: np.uint32, 5123: np.uint16, 5121: np.uint8}
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(js, binc, i) -> np.ndarray:
    a = js["accessors"][i]
    bv = js["bufferViews"][a["bufferView"]]
    dt = np.dtype(_COMP[a["componentType"]])
    nc = _NCOMP[a["type"]]
    off = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
    stride = bv.get("byteStride") or dt.itemsize * nc
    if stride == dt.itemsize * nc:
        return np.frombuffer(binc, dt, count=a["count"] * nc, offset=off).reshape(a["count"], nc)
    raw = np.frombuffer(binc, np.uint8, count=stride * (a["count"] - 1) + dt.itemsize * nc, offset=off)
    return np.stack([raw[k * stride:k * stride + dt.itemsize * nc].view(dt) for k in range(a["count"])])


def _quat_mat(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _node_matrix(n) -> np.ndarray:
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    m = np.eye(4)
    s = np.diag(n.get("scale", [1, 1, 1]))
    m[:3, :3] = _quat_mat(n.get("rotation", [0, 0, 0, 1])) @ s
    m[:3, 3] = n.get("translation", [0, 0, 0])
    return m


def glb_nodes(glb: bytes | str | Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Mesh nodes of a GLB (bytes or path) as [{name, extras, tris (n,3,3) in ENU metres, z up}], plus the scene
    extras."""
    js, binc = read_glb(glb)
    out = []

    def visit(i, parent):
        n = js["nodes"][i]
        m = parent @ _node_matrix(n)
        if "mesh" in n:
            tris = []
            for prim in js["meshes"][n["mesh"]]["primitives"]:
                if prim.get("mode", 4) != 4:
                    continue
                pos = _accessor(js, binc, prim["attributes"]["POSITION"]).astype(float)
                idx = (_accessor(js, binc, prim["indices"]).ravel().astype(np.int64) if "indices" in prim
                       else np.arange(len(pos)))
                p = pos @ m[:3, :3].T + m[:3, 3]
                enu = np.stack([p[:, 0], -p[:, 2], p[:, 1]], 1)        # glTF y-up (x=E, y=U, -z=N) -> ENU
                tris.append(enu[idx.reshape(-1, 3)])
            out.append({"name": n.get("name"), "extras": n.get("extras") or {},
                        "tris": np.concatenate(tris) if tris else np.zeros((0, 3, 3))})
        for c in n.get("children", []):
            visit(c, m)

    scene = js["scenes"][js.get("scene", 0)]
    for i in scene["nodes"]:
        visit(i, np.eye(4))
    return out, scene.get("extras") or {}


# ---------------------------------------------------------------- mesh geometry


def tri_vec_area(t: np.ndarray) -> np.ndarray:
    return 0.5 * np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])


class _UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[b] = a


def part_mesh_census(tris: np.ndarray, kinds: list[str], relief: np.ndarray, curved_roof: bool,
                     node_ids: np.ndarray | None = None, segments: bool = False) -> dict[str, Any]:
    """Faces and structural lines of ONE part's mesh.
    tris (n,3,3) ENU; kinds per triangle ('wall' | 'roof' | 'base'); relief (n,) bool: triangle is facade relief.
    Faces = smooth patches: edge-adjacent triangles of the same kind whose normals differ by < CREASE_DEG are merged
    (so a polygonal round tower is ONE wall face and an octagon is 8); every triangle of a curved roof shape (dome,
    onion, cone ...) is one face. Lines = crease edges (>= CREASE_DEG) grouped by the pair of faces they separate, typed
    by those faces: ridge / hip / valley (roof-roof), eave / rake (roof-wall), corner / wall-step / wall-edge
    (wall-wall), relief (a facade-relief face with a wall face), base (with the base)."""
    n = len(tris)
    res = {"faces": Counter(), "lines": [], "n_tris": n}
    if n == 0:
        return res
    va = tri_vec_area(tris)
    area = np.linalg.norm(va, axis=1)
    keep = area > 1e-6
    nrm = np.zeros_like(va)
    nrm[keep] = va[keep] / area[keep, None]
    q = np.round(tris.reshape(-1, 3) * 1000).astype(np.int64)
    _, vid = np.unique(q, axis=0, return_inverse=True)
    vid = vid.reshape(n, 3)
    P = tris.reshape(-1, 3)
    vpos = np.zeros((vid.max() + 1, 3))
    vpos[vid.ravel()] = P
    edges = defaultdict(list)
    for t in range(n):
        if not keep[t]:
            continue
        for k in range(3):
            a, b = vid[t, k], vid[t, (k + 1) % 3]
            if a == b:
                continue
            # split the edge at vertices lying on it (T-junctions between separately triangulated nodes)
            pa, d = vpos[a], vpos[b] - vpos[a]
            dd = float(d @ d)
            u = (vpos - pa) @ d / dd
            dist = np.linalg.norm(vpos - pa - u[:, None] * d, axis=1)
            mid = np.flatnonzero((u > 1e-6) & (u < 1 - 1e-6) & (dist < 1e-3))
            chain = [a] + [int(m) for m in mid[np.argsort(u[mid])]] + [b]
            for x, y in zip(chain[:-1], chain[1:]):
                edges[(min(x, y), max(x, y))].append(t)
    uf = _UF(n)
    cos_cr = math.cos(math.radians(CREASE_DEG))
    roof_ids = [t for t in range(n) if kinds[t] == "roof" and keep[t] and not relief[t]]
    if curved_roof and roof_ids:
        for t in roof_ids[1:]:
            uf.union(roof_ids[0], t)
    for (a, b), ts in edges.items():
        for i in range(len(ts)):
            for j in range(i + 1, len(ts)):
                t1, t2 = ts[i], ts[j]
                if kinds[t1] == kinds[t2] and relief[t1] == relief[t2] and float(nrm[t1] @ nrm[t2]) >= cos_cr:
                    uf.union(t1, t2)
    if node_ids is not None:     # coplanar pieces of one node are one face even across T-junctions
        off = (nrm * tris[:, 0]).sum(1)
        cos_cop = math.cos(math.radians(COPLANAR_DEG))
        for nid in np.unique(node_ids):
            idx = [t for t in np.flatnonzero(node_ids == nid) if keep[t]]
            for i in range(len(idx)):
                for j in range(i + 1, len(idx)):
                    t1, t2 = idx[i], idx[j]
                    if kinds[t1] == kinds[t2] and relief[t1] == relief[t2] and float(nrm[t1] @ nrm[t2]) >= cos_cop \
                            and abs(off[t1] - off[t2]) < 0.02:
                        uf.union(t1, t2)
    fac = np.array([uf.find(t) for t in range(n)])
    fkind = {}
    for t in range(n):
        if keep[t]:
            fkind.setdefault(fac[t], "relief" if relief[t] else kinds[t])
    for f, k in fkind.items():
        res["faces"][k] += 1
    cent = tris.mean(1)
    groups: dict[tuple, dict[str, Any]] = {}
    for (a, b), ts in edges.items():
        if len(ts) < 2:
            continue
        t1, t2 = ts[0], ts[1]
        if fac[t1] == fac[t2] or float(nrm[t1] @ nrm[t2]) > cos_cr:
            continue
        pa, pb = vpos[a], vpos[b]
        L = float(np.linalg.norm(pb - pa))
        k1, k2 = fkind[fac[t1]], fkind[fac[t2]]
        key = (min(fac[t1], fac[t2]), max(fac[t1], fac[t2]))
        g = groups.setdefault(key, {"kinds": (k1, k2), "len": 0.0, "dz": 0.0, "convex": 0.0, "n": 0, "segs": []})
        if segments:
            g["segs"].append((pa.copy(), pb.copy()))
        g["len"] += L
        g["dz"] += abs(pb[2] - pa[2])
        g["convex"] += L * (1.0 if float((cent[t2] - pa) @ nrm[t1]) < 0 else -1.0)
        g["n"] += 1
    for key, g in groups.items():
        L = g["len"]
        if L < MIN_EDGE_M:
            continue
        k1, k2 = g["kinds"]
        ks = {k1, k2}
        horiz = g["dz"] < 0.05 * L + 0.02
        vert = g["dz"] > 0.95 * L
        rel = "relief" in ks
        if "base" in ks:
            typ = "base"
        elif rel and ("wall" in ks or ks == {"relief"}):
            typ = "relief"
        elif ks <= {"roof"}:
            typ = ("ridge" if horiz else "hip") if g["convex"] > 0 else "valley"
        elif "roof" in ks:                      # roof with a wall or with a relief (recessed) wall face
            typ = "eave" if horiz else "rake"
        elif ks == {"wall"}:
            typ = "corner" if vert else ("wall-step" if horiz else "wall-edge")
        else:
            typ = "other"
        res["lines"].append({"type": typ, "len": L, "relief": rel, **({"segs": g["segs"]} if segments else {})})
    return res


def signed_volume(tris: np.ndarray) -> float:
    """Divergence theorem with F = (0, 0, z): exact for a closed mesh, and for a mesh open only at z = 0."""
    if not len(tris):
        return 0.0
    va = tri_vec_area(tris)
    return float((va[:, 2] * tris[:, :, 2].mean(1)).sum())


def relief_mask(tris: np.ndarray, neighbour_pts: np.ndarray | None = None) -> np.ndarray:
    """Triangles of a relief-bearing wall node that are NOT on its frame plane. The frame is the plane that touches the
    rest of the part (corners shared with the neighbouring walls, the eave shared with the roof): the candidate plane
    holding the most vertices of the part's OTHER nodes; without neighbours, the largest-area plane."""
    if not len(tris):
        return np.zeros(0, bool)
    va = tri_vec_area(tris)
    a = np.linalg.norm(va, axis=1)
    nrm = va / np.maximum(a, 1e-12)[:, None]
    off = (nrm * tris[:, 0]).sum(1)
    key = np.round(np.c_[nrm * 50, off * 20]).astype(int)          # ~1 deg, 5 cm bins
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    w = np.bincount(inv, weights=a)
    score = w / max(w.max(), 1e-9)                                  # tie-break by area
    if neighbour_pts is not None and len(neighbour_pts):
        for g in range(len(w)):
            sel = inv == g
            n0, d0 = nrm[sel].mean(0), off[sel].mean()
            n0 /= np.linalg.norm(n0)
            lo, hi = tris[sel].reshape(-1, 3).min(0) - 0.01, tris[sel].reshape(-1, 3).max(0) + 0.01
            near = neighbour_pts[(neighbour_pts >= lo).all(1) & (neighbour_pts <= hi).all(1)]
            score[g] += float((np.abs(near @ n0 - d0) < 0.01).sum())
    frame = int(np.argmax(score))
    sel = inv == frame
    n0, d0 = nrm[sel].mean(0), off[sel].mean()
    n0 /= np.linalg.norm(n0)
    on = (nrm @ n0 > math.cos(math.radians(2))) & (np.abs(off - d0) < 0.03)
    return ~on


# ---------------------------------------------------------------- registry view


def _parts_view(registry: dict | None, scene_extras: dict) -> list[dict[str, Any]]:
    """Uniform part records: id, drawn, outline_source, shape, params {name: {value, state, source, evidence}}."""
    if registry:
        out = []
        for p in registry["parts"]:
            out.append({"id": p["id"], "drawn": bool(p.get("drawn", True)), "outline_source": p.get("outline_source"),
                        "shape": p.get("shape") or {}, "params": p.get("params") or {}, "built": p.get("built"),
                        "state": p.get("state")})
        return out
    st = (scene_extras or {}).get("structure") or {}
    out = []
    for p in st.get("parts") or []:
        params = {k: {"value": v[0], "state": v[1], "source": v[2]} for k, v in (p.get("params") or {}).items()}
        out.append({"id": p["id"], "drawn": True, "outline_source": None, "shape": {"roof": p.get("roof"),
                    "source": None}, "params": params, "built": None, "state": p.get("state")})
    return out


def _evidence_channels(param: dict) -> dict[str, int]:
    """Geometry-measuring channels in a param's evidence -> photos they name (counts)."""
    ch = {}
    for e in param.get("evidence") or []:
        k = e.get("kind")
        if k in CHANNEL_KINDS:
            n = (e.get("n_photos") or e.get("skyline_photos") or e.get("photos") or e.get("points") or 0)
            ch[k] = max(ch.get(k, 0), int(n) if isinstance(n, (int, float)) else len(n))
        if k == "agreement" and e.get("agree"):
            for c in e.get("channels") or []:
                ch.setdefault(c, 1)
    return ch


def _adoption_backed(ad: dict) -> bool:
    if ad.get("kind") in D25_KINDS:
        # D25 (photo-corrected): >= 2 agreeing channels (heights), >= 2 lower-bound readings (blocks), >= 2 eave
        # readings (roof heights), >= 2 tilt channels (the tilt), the count crowd3d.correct.check_record requires
        n = {"d25-height": len(ad.get("agreeing") or []), "d25-block": len(ad.get("supported_by") or []),
             "d25-roof": len(((ad.get("eave") or {}).get("agreeing")) or []),
             "d25-tilt": len(((ad.get("result") or {}).get("channels_ok")) or [])}[ad["kind"]]
        return n >= 2 and bool(ad.get("active", True))
    chans = [c for c, v in (ad.get("channels") or {}).items() if v.get("measured") and v.get("ok", True)]
    return len(chans) >= 2 and bool(ad.get("active", True))


def param_class(param: dict, *, registry_present: bool, adoption_ok: bool = False) -> tuple[str, list[str], str | None]:
    """(class, flags, why-unbacked) of one numeric parameter."""
    state, source = param.get("state"), param.get("source")
    cls = STATE_CLASS.get(state)
    flags = []
    if state == "disputed":
        flags.append("disputed")
    if source in OSM_DEFAULT_SOURCES and cls == "osm":
        flags.append("osm-default")
        cls = "convention"                  # no tag: the spec's default, as a defaulted roof shape (SHAPE_CLASS)
    if cls is None:
        return "unbacked", flags, f"state {state!r} not in the provenance enum"
    if cls in PHOTO_CLASSES:
        if not registry_present:
            return "unbacked", flags, "photo claim without a registry (no evidence)"
        ch = _evidence_channels(param)
        if len(ch) < 2 and not adoption_ok:
            return "unbacked", flags, f"photo claim with {len(ch)} measuring channel(s)"
    return cls, flags, None


def outline_class(src: str | None, adoption_ok: bool) -> tuple[str, str | None]:
    cls = OUTLINE_CLASS.get(src or "osm-s3db")
    if cls is None:
        return "unbacked", f"outline source {src!r} not in the enum"
    if cls == "photo-fitted" and not adoption_ok:
        return "unbacked", "photo-fitted outline without an active >=2-channel adoption"
    return cls, None


# ---------------------------------------------------------------- facade elements (relief etc.)


def _element_lists(extras: dict) -> list[tuple[str, list[dict]]]:
    """Lists of element records in a node's extras: any list of dicts that carry 'type' and 'provenance'."""
    out = []
    for k, v in extras.items():
        if isinstance(v, list) and v and all(isinstance(e, dict) and "type" in e and "provenance" in e for e in v):
            out.append((k, v))
    return out


PROV_D24 = "photo-measured (points, D24)"      # crowd3d.relief.PROV_D24 (this module is standalone)


D24_NULL_P, D24_EDGE_M, D24_BOX_M = 0.1, 0.3, 0.6    # crowd3d.relief.ReliefGuards: line_null_p, d24_edge_m, d24_box_m


def d24_backing(e: dict) -> str | None:
    """None when a D24 element's own evidence is complete (crowd3d.relief.d24_violations, generically): its d24 block
    with the model's scale condition allowed, its line evidence at chance (>= 1 edge at a line cluster, the chance
    statistics kept, the same numbers as its lines3d channel, above the null), both photo splits' measurements of it
    (two halves each, every edge found again: within D24_EDGE_M between the halves and D24_BOX_M of the element),
    and every guard but G-R3 passed."""
    d = e.get("d24")
    if not isinstance(d, dict):
        return "a D24 element without its D24 evidence"
    if (d.get("scale_condition") or {}).get("allowed") is not True:
        return "a D24 element on a model whose scale condition is not met"
    l3 = d.get("lines3d") or {}
    if not isinstance(l3, dict) or (l3.get("edges_confirmed") or 0) < 1 or l3.get("p_chance") is None \
            or not isinstance(l3.get("chance"), list):
        return "a D24 element without its line evidence at chance (>= 1 edge at a line cluster)"
    ch3 = (e.get("channels") or {}).get("lines3d") or {}
    try:
        if int(ch3.get("edges_confirmed", -1)) != int(l3["edges_confirmed"]) or ch3.get("p_chance") is None \
                or abs(float(ch3["p_chance"]) - float(l3["p_chance"])) > 1e-4:
            return "a D24 element whose D24 line numbers differ from its lines3d channel"
        if float(l3["p_chance"]) <= D24_NULL_P:
            return "a D24 element whose line evidence is not at chance"
    except (TypeError, ValueError):
        return "a D24 element without its line evidence at chance (>= 1 edge at a line cluster)"
    halves = d.get("halves") or {}
    keys = ("z_bottom", "z_top") if e.get("type") == "band" else ("u0", "u1", "z_bottom", "z_top")
    for name in ("interleaved", "viewpoint"):
        h = halves.get(name) if isinstance(halves, dict) else None
        if not isinstance(h, dict) or not all(isinstance(h.get(k), dict) for k in ("a", "b")):
            return f"a D24 element without both {name} photo halves' measurements"
        # every edge found again (crowd3d.relief.d24_halves_check, generically)
        try:
            a, b = h["a"], h["b"]
            if any(abs(float(a[k]) - float(b[k])) > D24_EDGE_M + 1e-9
                   or max(abs(float(a[k]) - float(e[k])), abs(float(b[k]) - float(e[k]))) > D24_BOX_M + 1e-9 for k in keys):
                return f"a D24 element with an edge the {name} photo halves do not find again"
        except (KeyError, TypeError, ValueError):
            return f"a D24 element without both {name} photo halves' boxes"
    g = e.get("guards")
    if isinstance(g, dict) and not all(bool(v) for k, v in g.items() if k != "G-R3_line_edge"):
        return "a point-based guard failed"
    return None


def element_backing(e: dict, node: dict, stub_parts: set[str]) -> str | None:
    """None when a photo element is backed; else why not. Mirrors the relief audit (G-R1..G-R4) generically; a D24
    element (provenance PROV_D24: built on the points alone, its 3-D lines at chance) by its own D24 evidence
    (d24_backing) in place of the line evidence above chance and G-R3."""
    ex = node["extras"]
    d24 = e.get("provenance") == PROV_D24
    if e.get("provenance") != "photo-measured" and not d24:
        return f"provenance {e.get('provenance')!r} (facade elements exist only when photo-measured)"
    if ex.get("kind") != "wall" or not ex.get("observed") or (ex.get("provenance") or {}).get("appearance") != "photo-observed":
        return "element on a surface that was not photographed"
    if ex.get("part") in stub_parts:
        return "element on a stub part"
    ch = e.get("channels") or {}
    n_ch = sum(1 for k, v in ch.items() if v not in (None, False, {}, 0))
    if n_ch < 2:
        return f"{n_ch} channel(s)"
    pph = e.get("photos_per_half") or []
    if len(pph) < 2 or min(pph) < MIN_PHOTOS_PER_HALF:
        return f"photos per split half {pph} < {MIN_PHOTOS_PER_HALF}"
    l3 = ch.get("lines3d") or {}
    if isinstance(l3, dict) and l3 and l3.get("edges_confirmed", 0) < 1:
        return "no line-confirmed edge"
    if d24:
        return d24_backing(e)
    if isinstance(l3, dict) and l3.get("p_chance") is not None and float(l3["p_chance"]) > 0.1:
        return "line confirmation at chance"
    g = e.get("guards")
    if isinstance(g, dict) and not all(bool(v) for v in g.values()):
        return "a guard failed"
    return None


def _wall_frame(tris: np.ndarray) -> tuple[float, float, float]:
    """(length, z0, z1) of a wall node from its frame-plane vertices."""
    p = tris.reshape(-1, 3)
    xy = p[:, :2]
    c = xy.mean(0)
    u, s, vt = np.linalg.svd(xy - c, full_matrices=False)
    t = (xy - c) @ vt[0]
    return float(t.max() - t.min()), float(p[:, 2].min()), float(p[:, 2].max())


def element_dof(els: list[dict], wall_len: float, z0: float, z1: float) -> dict[str, int]:
    """Distinct photo-measured coordinates on one wall (u, z, depth), ignoring values that coincide with the wall's own
    bounds (they come from the frame, not from the measurement) and counting repeated values once."""
    us, zs, ds = [], [], []
    for e in els:
        for f, lo, hi, acc in (("u0", 0.0, wall_len, us), ("u1", 0.0, wall_len, us),
                               ("z_bottom", z0, z1, zs), ("z_top", z0, z1, zs)):
            v = e.get(f)
            if v is None:
                continue
            if abs(v - lo) <= COORD_TOL_M or abs(v - hi) <= COORD_TOL_M or (f == "z_bottom" and v <= z0 + COORD_TOL_M) \
                    or (f == "z_top" and v >= z1 - COORD_TOL_M):
                continue
            acc.append(float(v))
        for f in DEPTH_FIELDS:
            if e.get(f) is not None:
                ds.append(round(float(e[f]), 3))
                break

    def distinct(vals):
        vals = sorted(vals)
        n, last = 0, None
        for v in vals:
            if last is None or v - last > COORD_TOL_M:
                n += 1
                last = v
        return n
    return {"u": distinct(us), "z": distinct(zs), "depth": distinct(ds)}


# ---------------------------------------------------------------- main


def _weights(pairs) -> dict[str, float]:
    w = Counter()
    for c, x in pairs:
        w[c] += x
    return dict(w)


def _share(c: dict[str, float]) -> dict[str, float]:
    tot = sum(c.values())
    return {k: (round(c.get(k, 0.0) / tot, 4) if tot else 0.0) for k in CLASSES}


def structure_metrics(registry: dict | None, glb: bytes | str | Path | None, *, built_only: bool = True) -> dict[str, Any]:
    """Census, provenance shares (parameters, elements on two axes, area, volume), detail score and honesty of a
    model: the structure registry (StructureModel.to_registry(); None: the GLB's scene extras, where photo claims
    count as unbacked) and its GLB (bytes or path)."""
    nodes, sx = glb_nodes(glb) if glb is not None and glb != b"" else ([], {})
    parts = _parts_view(registry, sx)
    reg_present = registry is not None
    adoptions = (registry or {}).get("adoptions") or []
    outline_adopt_ok = any(a.get("param") == "outline" and _adoption_backed(a) for a in adoptions)
    backed_ids = {a.get("id") for a in adoptions if _adoption_backed(a)}
    unbacked: list[dict[str, Any]] = []

    # ---- per-part classes and the parameter census
    pinfo: dict[str, dict[str, Any]] = {}
    param_cnt, param_flags = Counter(), Counter()
    shape_cnt = Counter()
    for p in parts:
        if not p["drawn"]:
            continue
        cls_by = {}
        oc, why = outline_class(p["outline_source"], outline_adopt_ok)
        if why:
            unbacked.append({"what": f"{p['id']}.outline", "why": why})
        cls_by["outline"] = oc
        for name, prm in p["params"].items():
            # a value carried with a parent's backed adoption / correction (its own evidence is the carry record)
            carried_ok = any(e.get("kind") == "carried" and e.get("adoption") in backed_ids for e in prm.get("evidence") or [])
            c, flags, why = param_class(prm, registry_present=reg_present, adoption_ok=carried_ok)
            if why:
                unbacked.append({"what": f"{p['id']}.{name}", "why": why})
            cls_by[name] = c
            for f in flags:
                param_flags[f] += 1
        for c in cls_by.values():
            param_cnt[c] += 1
        shp = p["shape"] or {}
        # a roof shape with no roof:shape tag is the S3DB default (flat): a convention, a stub's a placeholder
        shape_cnt["stub" if cls_by.get("height") == "stub" else SHAPE_CLASS.get(shp.get("source"), "unknown")] += 1
        roof_h = float((p["params"].get("roof_height") or {}).get("value") or 0.0)
        g = lambda k: cls_by.get(k, "unbacked")
        top = [g("height")] + ([g("roof_height")] if roof_h > 0 else [])
        # elevation weights: a wall is bounded by its base (min_height) and its eave (height - roof_height); a roof
        # by its eave and its top (height); a volume by all three
        w_wall = _weights([(g("min_height"), 0.5)] + [(c, 0.5 / len(top)) for c in top])
        w_roof = _weights([(c, 1.0 / len(top)) for c in top])
        vol_e = [g("min_height"), g("height")] + ([g("roof_height")] if roof_h > 0 else [])
        w_vol = _weights([(c, 1.0 / len(vol_e)) for c in vol_e])
        is_stub = cls_by.get("height") == "stub"
        # existence: who says the element is there. A fitted outline is still the mapped outline (transformed);
        # a stub part's elements are placeholders.
        ex_part = "stub" if is_stub else {"photo-fitted": "osm"}.get(oc, oc)
        ex_roof = "stub" if is_stub else SHAPE_CLASS.get(shp.get("source"), "osm" if not reg_present else "unbacked")
        # dimension classes PER PARAMETER (not the strongest class): an element's unit is shared among the classes
        # of the numbers that define it -- a face or line half by the outline, half by the heights bounding it; a
        # part equally by all its numbers. A stub part's elements are stubs.
        oc_ = cls_by.get("outline", "unbacked")
        d_wall = _weights([(oc_, 0.5)] + [(c, 0.5 * x) for c, x in w_wall.items()])
        d_roof = _weights([(oc_, 0.5)] + [(c, 0.5 * x) for c, x in w_roof.items()])
        d_part = _weights([(c, 1.0 / max(len(cls_by), 1)) for c in cls_by.values()])
        if cls_by.get("height") == "stub":
            d_wall = d_roof = d_part = {"stub": 1.0}
        pinfo[p["id"]] = {"cls": strongest(cls_by.values()), "by": cls_by, "w_wall": w_wall, "w_roof": w_roof,
                          "d_wall": d_wall, "d_roof": d_roof, "d_part": d_part,
                          "w_vol": w_vol, "roof": (shp.get("roof") or shp.get("tag")), "stub": is_stub,
                          "ex_part": ex_part, "ex_roof": ex_roof}
    stub_parts = {pid for pid, v in pinfo.items() if v["stub"]}
    n_params = sum(param_cnt.values())

    # ---- geometry: per part meshes (walls, roofs, bases; roof patches are texture overlays and are skipped)
    by_part: dict[str, list[dict]] = defaultdict(list)
    for nd in nodes:
        k = nd["extras"].get("kind")
        if k in ("wall", "roof", "base") and nd["extras"].get("part") is not None:
            by_part[nd["extras"]["part"]].append(nd)
    area_cls, area_plan, area_elev, vol_cls = Counter(), Counter(), Counter(), Counter()
    elem_cls: dict[str, Counter] = defaultdict(Counter)          # element type -> dims class -> n
    elem_ex: dict[str, Counter] = defaultdict(Counter)           # element type -> existence class -> n
    line_m_cls: dict[str, Counter] = defaultdict(Counter)
    wall_faces, total_area, photographed_area = 0, 0.0, 0.0
    detailed_area = 0.0
    dof = {"u": 0, "z": 0, "depth": 0}
    elem_types, elem_measured = Counter(), Counter()
    elem_d24, dof_d24 = Counter(), {"u": 0, "z": 0, "depth": 0}      # of them: built on the points alone (D24)
    photo_edge_m, relief_vol = 0.0, 0.0
    relief_geo = {"n": 0, "m": 0.0}
    signatures: dict[tuple, str] = {}
    curved = 0
    for pid, nds in by_part.items():
        info = pinfo.get(pid)
        if info is None:          # GLB part missing from the registry's drawn parts
            unbacked.append({"what": f"part {pid}", "why": "GLB part not in the registry"})
            info = {"cls": "unbacked", "by": {"outline": "unbacked"}, "w_wall": {"unbacked": 1.0},
                    "w_roof": {"unbacked": 1.0}, "w_vol": {"unbacked": 1.0}, "roof": None, "stub": False,
                    "d_wall": {"unbacked": 1.0}, "d_roof": {"unbacked": 1.0}, "d_part": {"unbacked": 1.0},
                    "ex_part": "unbacked", "ex_roof": "unbacked"}
        T, K, R, NI = [], [], [], []
        for ni, nd in enumerate(nds):
            ex = nd["extras"]
            t = nd["tris"]
            if not len(t):
                continue
            rel = np.zeros(len(t), bool)
            lists = _element_lists(ex)
            if ex.get("kind") == "wall":
                wall_faces += 1
                if lists:
                    others = [o["tris"].reshape(-1, 3) for o in nds if o is not nd and len(o["tris"])]
                    rel = relief_mask(t, np.unique(np.concatenate(others), axis=0) if others else None)
            T.append(t)
            NI.append(np.full(len(t), ni))
            K += [ex.get("kind")] * len(t)
            R.append(rel)
            a = np.linalg.norm(tri_vec_area(t), axis=1)
            if ex.get("kind") in ("wall", "roof"):
                total_area += float(a.sum())
                if ex.get("observed"):
                    photographed_area += float(a.sum())
                # plan half -> outline class; elevation half -> the heights that bound the surface (weights)
                welev = info["w_wall"] if ex.get("kind") == "wall" else info["w_roof"]
                a_rel, a_frame = float(a[rel].sum()), float(a[~rel].sum())
                area_cls["photo-measured"] += a_rel
                area_plan["photo-measured"] += a_rel
                area_elev["photo-measured"] += a_rel
                area_cls[info["by"].get("outline", "unbacked")] += 0.5 * a_frame
                area_plan[info["by"].get("outline", "unbacked")] += a_frame
                for c, wgt in welev.items():
                    area_cls[c] += 0.5 * a_frame * wgt
                    area_elev[c] += a_frame * wgt
            for key, els in lists:
                L, z0, z1 = _wall_frame(t)
                ok_els = []
                for e in els:
                    typ = e.get("kind") if e.get("type") == "vertical" else e.get("type")
                    elem_measured[typ] += 1
                    why = element_backing(e, nd, stub_parts)
                    sig = (typ, tuple(round(float(e.get(f) or 0.0), 2) for f in NUM_FIELDS),
                           frozenset(e.get("line_photos") or []))
                    if why is None and sig[2]:
                        first = signatures.setdefault(sig, nd["name"])
                        if first != nd["name"]:
                            why = f"same record (coordinates and photos) as on {first}: a copy, not a measurement"
                    if why:
                        unbacked.append({"what": f"{nd['name']}.{key}.{typ}", "why": why})
                        elem_cls[typ]["unbacked"] += 1
                        elem_ex[typ]["unbacked"] += 1
                        continue
                    if built_only and not e.get("built", True):
                        continue
                    ok_els.append(e)
                    elem_types[typ] += 1
                    elem_d24[typ] += e.get("provenance") == PROV_D24
                    elem_cls[typ]["photo-measured"] += 1
                    elem_ex[typ]["photo-measured"] += 1
                    w = max(0.0, e["u1"] - e["u0"])
                    h = max(0.0, e["z_top"] - max(e["z_bottom"], z0))
                    d = next((abs(float(e[f])) for f in DEPTH_FIELDS if e.get(f) is not None), 0.0)
                    photo_edge_m += 2 * w if e.get("type") == "band" else 2 * (w + h)
                    relief_vol += w * h * d
                    line_m_cls[typ]["photo-measured"] += 2 * w if e.get("type") == "band" else 2 * (w + h)
                if ok_els:
                    dd = element_dof(ok_els, L, z0, z1)
                    for k2 in dof:
                        dof[k2] += dd[k2]
                    d24_els = [e for e in ok_els if e.get("provenance") == PROV_D24]
                    if d24_els:
                        dd = element_dof(d24_els, L, z0, z1)
                        for k2 in dof_d24:
                            dof_d24[k2] += dd[k2]
                    detailed_area += float(np.linalg.norm(tri_vec_area(t), axis=1).sum())
        if not T:
            continue
        T = np.concatenate(T)
        R = np.concatenate(R)
        is_curved = (info["roof"] or "") in CURVED_ROOFS
        curved += int(is_curved and "roof" in K)
        mc = part_mesh_census(T, K, R, is_curved, np.concatenate(NI))
        pc = info["cls"]

        def add(typ, dist, n=1.0):
            for c, x in dist.items():
                elem_cls[typ][c] += n * x
        add("part", info["d_part"])
        elem_ex["part"][info["ex_part"]] += 1
        for k, v in mc["faces"].items():
            if k in ("wall", "roof"):
                add(f"{k}_face", info["d_roof"] if k == "roof" else info["d_wall"], v)
                elem_ex[f"{k}_face"][info["ex_roof"] if k == "roof" else info["ex_part"]] += v
        for ln in mc["lines"]:
            k = ln["type"]
            if k in ("base", "other"):
                continue
            if k == "relief":           # outlines of facade elements: counted with the elements (cross-check only)
                relief_geo["n"] += 1
                relief_geo["m"] += ln["len"]
                continue
            roofish = k in ("ridge", "hip", "valley", "rake", "eave")
            dist = {"photo-measured": 1.0} if ln["relief"] and pc != "stub" else (info["d_roof"] if roofish else info["d_wall"])
            xc = info["ex_roof"] if k in ("ridge", "hip", "valley", "rake") else info["ex_part"]
            add(f"line_{k}", dist)
            elem_ex[f"line_{k}"][xc] += 1
            for c, x in dist.items():
                line_m_cls[f"line_{k}"][c] += ln["len"] * x
        V = abs(signed_volume(T))
        plan_c = info["by"].get("outline", "unbacked")
        vol_cls[plan_c] += 0.5 * V
        for c, wgt in info["w_vol"].items():
            vol_cls[c] += 0.5 * V * wgt
    if relief_vol:
        vol_cls["photo-measured"] += relief_vol
    # parts that the registry draws but the GLB lacks are counted from the registry
    for pid, info in pinfo.items():
        if pid not in by_part and nodes:
            for c, x in info["d_part"].items():
                elem_cls["part_without_mesh"][c] += x
            elem_ex["part_without_mesh"][info["ex_part"]] += 1
    # wall-face count by node (one node per wall piece)
    wall_nodes = Counter()
    for nd in nodes:
        if nd["extras"].get("kind") == "wall":
            wall_nodes[pinfo.get(nd["extras"].get("part"), {"cls": "unbacked"})["cls"]] += 1
    node_census = {"wall_nodes": dict(wall_nodes)}

    # ---- photo DOF: numbers the photos determined (fitted + measured); verified adds none
    dof_fitted, covered = 0, set()
    photo_fit = []
    for a in adoptions:
        if not _adoption_backed(a):
            continue
        v = a.get("value")
        if a.get("kind") == "d25-tilt":
            # the stack's tilt (D25-T): its lean east and north, photo-measured numbers of the stack
            photo_fit.append({"param": "tilt", "part": "stack", "tilt_deg": (v or {}).get("tilt_deg"),
                              "azimuth_deg": (v or {}).get("azimuth_deg"), "numbers": 2, "kind": "d25-tilt"})
            dof_fitted += 2
            continue
        if a.get("kind") in D25_KINDS:
            photo_fit.append({"param": a.get("param"), "part": a.get("part"), "value": v, "map": (a.get("map_value") or {}).get("value"),
                              "adopted": a.get("adopted"), "numbers": 1, "kind": a.get("kind")})
        if isinstance(v, dict) and "k" in v:
            t = v.get("t") or [None, None]
            photo_fit.append({"param": a.get("param"), "part": a.get("part"), "k": v.get("k"),
                              "tx": t[0] if len(t) > 0 else None, "ty": t[1] if len(t) > 1 else None,
                              "numbers": 1 + len(t), "channels": sorted((a.get("channels") or {}).keys())})
        if isinstance(v, dict):
            dof_fitted += sum(1 if isinstance(x, (int, float)) else len(x) for k, x in v.items() if k in ("k", "t"))
        elif isinstance(v, (int, float)):
            dof_fitted += 1
        covered.add((a.get("part"), a.get("param")))
    dof_fitted += sum(1 for pid, p in pinfo.items() for k, c in p["by"].items()
                      if c in ("photo-fitted", "photo-corrected") and k != "outline" and (pid, k) not in covered)
    dof_measured = int(sum(dof.values())) + sum(1 for p in pinfo.values() for c in p["by"].values() if c == "photo-measured")
    photo_dof = dof_fitted + dof_measured

    # element totals (structural elements: parts, faces, lines, facade elements)
    elem_total, ex_total = Counter(), Counter()
    for typ in elem_cls:
        if typ == "part_without_mesh":
            continue
        elem_total.update(elem_cls[typ])
        ex_total.update(elem_ex[typ])
    census = {
        "parts": {"total": len(parts), "drawn": sum(1 for p in parts if p["drawn"])},
        "elements": {typ: {k: round(v, 3) for k, v in c.items()} for typ, c in sorted(elem_cls.items())},
        "elements_n": {typ: int(round(sum(c.values()))) for typ, c in sorted(elem_cls.items())},
        "elements_existence": {typ: dict(c) for typ, c in sorted(elem_ex.items())},
        "relief_lines_geometric": {"n": relief_geo["n"], "m": round(relief_geo["m"], 1)},
        "nodes": node_census,
        "line_m": {typ: {k: round(v, 1) for k, v in c.items()} for typ, c in sorted(line_m_cls.items())},
        "curved_roofs": curved,
        "facade_elements_measured": dict(elem_measured),
        "facade_elements_built": dict(elem_types),
        "facade_elements_built_d24": {k: v for k, v in elem_d24.items() if v},
        "roof_shape_source": dict(shape_cnt),
    }
    # the roof's structural lines the model lists (crowd3d.stylize.structure_lines): conventional, photo-verified
    rl = [r for r in ((sx or {}).get("structure_lines") or {}).get("lines") or [] if isinstance(r, list) and len(r) >= 9]
    census["roof_lines"] = {"n": len(rl), "convention": sum(1 for r in rl if r[7]),
                            "photo_verified": sum(1 for r in rl if r[8])}
    # ---- the raw numbers that define the model (the headline share): every outline coordinate (2 per vertex of a
    # part's rings, as built: from the GLB's structure extras) and every height parameter by its class -- an adopted
    # (photo-fitted) outline keeps the map's coordinates, a photo-verified value is the map's number checked -- and
    # the numbers the photos determined (photo_dof: the adopted similarity's k, tx, ty, fitted parameters, measured
    # facade-element numbers). None when the GLB carries no rings.
    rings_of = {q.get("id"): q.get("rings") for q in ((sx or {}).get("structure") or {}).get("parts") or []}
    numbers, numbers_verified = Counter(), 0
    if pinfo and all(isinstance(rings_of.get(pid), list) and rings_of.get(pid) for pid in pinfo):
        for pid, info in pinfo.items():
            oc = info["by"].get("outline", "unbacked")
            numbers[{"photo-fitted": "osm"}.get(oc, oc)] += 2 * sum(len(r) for r in rings_of[pid])
            for k, c in info["by"].items():
                if k == "outline" or c in PHOTO_VALUE_CLASSES:          # photo values: counted in photo_dof
                    continue
                numbers["osm" if c == "photo-verified" else c] += 1
                numbers_verified += c == "photo-verified"
        numbers["photo"] += photo_dof
    else:
        numbers = None
    shares = {
        "numbers": None if numbers is None else {k: round(v / max(sum(numbers.values()), 1), 4) for k, v in numbers.items()},
        "numbers_n": None if numbers is None else int(sum(numbers.values())),
        "numbers_counts": None if numbers is None else dict(numbers),
        "numbers_osm_photo_verified": numbers_verified,
        "elements": _share(elem_total), "elements_n": int(round(sum(elem_total.values()))),
        "elements_existence": _share(ex_total),
        "params": _share(param_cnt), "params_n": n_params, "params_counts": dict(param_cnt),
        "params_flags": dict(param_flags),
        "area": _share(area_cls), "area_m2": round(sum(area_cls.values()), 1),
        "area_plan": _share(area_plan), "area_elevation": _share(area_elev),
        "volume": _share(vol_cls), "volume_m3": round(sum(vol_cls.values()), 1),
    }
    detail = {
        "photo_dof": photo_dof, "photo_dof_fitted": dof_fitted, "photo_dof_measured": dof_measured,
        "photo_dof_by_axis": dof,
        "photo_elements": int(sum(elem_types.values())),
        "photo_elements_d24": int(sum(elem_d24.values())),
        "photo_dof_measured_d24": int(sum(dof_d24.values())),
        "photo_edge_m": round(photo_edge_m, 1),
        "photo_relief_volume_m3": round(relief_vol, 2),
        "photo_verified_params": int(param_cnt.get("photo-verified", 0)),
        "photo_fit": photo_fit,
        "photographed_area_m2": round(photographed_area, 1),
        "detailed_area_m2": round(detailed_area, 1),
        "detail_coverage": round(detailed_area / photographed_area, 4) if photographed_area else 0.0,
        "photo_dof_per_100m2": round(100 * photo_dof / photographed_area, 3) if photographed_area else 0.0,
    }
    flat = flat_metrics(census, shares, detail, unbacked)
    return {"census": census, "shares": shares, "detail": detail,
            "honesty": {"unbacked": len(unbacked), "items": unbacked[:50]}, "metrics": flat,
            "total_area_m2": round(total_area, 1)}


def compact(m: dict[str, Any]) -> dict[str, Any]:
    """What the GLB's scene extras carry of structure_metrics(): the flat metrics, the shares and the detail score."""
    sh = m["shares"]
    return {"version": 1, "metrics": m["metrics"],
            "shares": {k: sh[k] for k in ("numbers", "params", "elements", "elements_existence", "area", "volume")},
            "counts": {"numbers": sh["numbers_counts"], "params": sh["params_n"], "elements": sh["elements_n"],
                       "area_m2": sh["area_m2"], "volume_m3": sh["volume_m3"]},
            "detail": {k: m["detail"][k] for k in ("photo_dof", "photo_dof_fitted", "photo_dof_measured", "photo_elements",
                                                   "photo_elements_d24", "photo_dof_measured_d24",
                                                   "photo_edge_m", "detail_coverage", "photo_verified_params",
                                                   "photo_fit")},
            "unbacked": m["honesty"]["unbacked"]}


def flat_metrics(census, shares, detail, unbacked) -> dict[str, float]:
    p, a, v, e = shares["params"], shares["area"], shares["volume"], shares["elements"]
    ph = lambda s, cs: round(sum(s.get(c, 0.0) for c in cs), 4)
    return {
        "struct_elements": shares["elements_n"],
        "struct_parts": census["parts"]["drawn"],
        "struct_roof_faces": census["elements_n"].get("roof_face", 0),
        "struct_wall_faces": census["elements_n"].get("wall_face", 0),
        "struct_lines": sum(n for t, n in census["elements_n"].items() if t.startswith("line_")),
        "struct_facade_elements": detail["photo_elements"],
        "struct_facade_elements_d24": detail["photo_elements_d24"],
        "struct_params": shares["params_n"],
        "struct_numbers": shares["numbers_n"] or 0,
        "struct_share_photo_numbers": ph(shares["numbers"] or {}, ("photo",)),
        "struct_share_osm_params": ph(p, ("osm",)),
        "struct_share_convention_params": ph(p, ("convention",)),
        "struct_share_stub_params": ph(p, ("stub",)),
        "struct_share_photo_checked_params": ph(p, ("photo-verified",)),
        "struct_share_photo_value_params": ph(p, PHOTO_VALUE_CLASSES),
        "struct_share_photo_elements": ph(e, PHOTO_VALUE_CLASSES),
        "struct_share_photo_checked_elements": ph(e, ("photo-verified",)),
        "struct_share_photo_added_elements": ph(shares["elements_existence"], ("photo-measured",)),
        "struct_share_photo_area": ph(a, PHOTO_VALUE_CLASSES),
        "struct_share_photo_checked_area": ph(a, ("photo-verified",)),
        "struct_share_photo_volume": ph(v, PHOTO_VALUE_CLASSES),
        "struct_share_photo_checked_volume": ph(v, ("photo-verified",)),
        "struct_share_stub_area": ph(a, ("stub",)),
        "struct_photo_dof": detail["photo_dof"],
        "struct_photo_edge_m": detail["photo_edge_m"],
        "struct_detail_coverage": detail["detail_coverage"],
        "struct_unbacked": len(unbacked),
        "struct_roof_lines": census["roof_lines"]["n"],
        "struct_share_photo_verified_roof_lines": (round(census["roof_lines"]["photo_verified"] / census["roof_lines"]["n"], 4)
                                                   if census["roof_lines"]["n"] else 0.0),
    }


METRIC_SPECS = [
    {"name": "struct_numbers", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Raw numbers that define the model: 2 per outline vertex of every drawn part (as built), its height "
                    "parameters, and the numbers the photos determined (struct_photo_dof)."},
    {"name": "struct_share_photo_numbers", "unit": "share", "direction": "higher", "aggregate": "mean", "group": "structure",
     "description": "THE headline share: of struct_numbers, those the photos determined (an adopted similarity's k, "
                    "tx, ty; fitted parameters; measured facade-element numbers). An adopted outline keeps the map's "
                    "coordinates and a photo-verified value is the map's number checked: both count as OSM numbers."},
    {"name": "struct_elements", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Structural elements in the shipped model: drawn parts + wall faces + roof faces + structural "
                    "lines (ridges, hips, valleys, eaves, rakes, corners, steps) + built backed facade elements. Faces "
                    "and lines are measured on the GLB geometry (welded per part, T-junctions split)."},
    {"name": "struct_parts", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure", "description": "Drawn building parts."},
    {"name": "struct_roof_faces", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Roof faces: smooth patches separated by creases >= 20 deg (each curved roof shape counts once)."},
    {"name": "struct_wall_faces", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Wall faces: smooth patches separated by creases >= 20 deg (a polygonal round tower counts once); "
                    "facade relief does not split a wall face."},
    {"name": "struct_lines", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Structural lines: mesh edges between two facets meeting at > 10 deg (ridge, hip, valley, eave, "
                    "rake, wall corner, wall step), excluding base and facade-relief edges."},
    {"name": "struct_facade_elements", "unit": "count", "direction": "higher", "aggregate": "sum", "group": "structure", "min_delta": 1,
     "description": "Built, backed photo-measured facade elements (bands, pilasters, buttresses, openings ...)."},
    {"name": "struct_facade_elements_d24", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Of struct_facade_elements, those built on the points alone (D24, provenance 'photo-measured "
                    "(points, D24)'): both photo splits and the raw-point re-measurement, the wall's 3-D lines at "
                    "chance, on a model whose scale condition is met."},
    {"name": "struct_params", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Numeric frame parameters: per drawn part its outline + min_height + height + roof_height."},
    {"name": "struct_share_osm_params", "unit": "share", "direction": "none", "aggregate": "mean", "group": "structure",
     "description": "Share of frame parameters whose value is the OSM tag (or S3DB default), incl. disputed ones."},
    {"name": "struct_share_convention_params", "unit": "share", "direction": "lower", "aggregate": "mean", "group": "structure",
     "description": "Share of frame parameters set by a modelling convention."},
    {"name": "struct_share_stub_params", "unit": "share", "direction": "lower", "aggregate": "mean", "group": "structure",
     "description": "Share of frame parameters that are placeholders (stub)."},
    {"name": "struct_share_photo_checked_params", "unit": "share", "direction": "none", "aggregate": "mean", "group": "structure",
     "description": "Share of frame parameters whose OSM value the photos CHECKED and kept (photo-verified, >= 2 "
                    "channels). These are OSM values, not values from photos."},
    {"name": "struct_share_photo_value_params", "unit": "share", "direction": "higher", "aggregate": "mean", "group": "structure",
     "description": "Share of frame parameters whose VALUE comes from photos (fitted or measured). A D15 outline fit "
                    "counts each outline it moves; the fit itself is 3 numbers (k, tx, ty: detail.photo_fit)."},
    {"name": "struct_share_photo_elements", "unit": "share", "direction": "higher", "aggregate": "mean", "group": "structure",
     "description": "Share of structural elements' DIMENSIONS whose value comes from photos (fitted or measured), per "
                    "parameter: each element's unit is shared among the classes of the numbers that define it (a face "
                    "or line half by its outline, half by the heights bounding it; a part equally by all its numbers; "
                    "facade-relief elements fully photo-measured). Photo-verified (OSM) values do not count."},
    {"name": "struct_share_photo_checked_elements", "unit": "share", "direction": "none", "aggregate": "mean", "group": "structure",
     "description": "As struct_share_photo_elements, for the photo-verified (OSM values checked by photos) share."},
    {"name": "struct_share_photo_added_elements", "unit": "share", "direction": "higher", "aggregate": "mean", "group": "structure",
     "description": "Share of structural elements that EXIST only because photos measured them (existence axis: "
                    "osm / convention / stub / photo-measured; a fitted outline is still the mapped outline)."},
    {"name": "struct_share_photo_area", "unit": "share", "direction": "higher", "aggregate": "mean", "group": "structure",
     "description": "Share of visible surface area (walls + roofs) whose numbers come from photos (fitted or "
                    "measured): each frame triangle counts half to its outline's class and half to the classes of the "
                    "heights that bound it; facade-relief triangles count fully photo-measured. Photo-verified (OSM) "
                    "values do not count (struct_share_photo_checked_area)."},
    {"name": "struct_share_photo_checked_area", "unit": "share", "direction": "none", "aggregate": "mean", "group": "structure",
     "description": "As struct_share_photo_area, for the photo-verified (OSM values checked by photos) share."},
    {"name": "struct_share_photo_volume", "unit": "share", "direction": "higher", "aggregate": "mean", "group": "structure",
     "description": "As struct_share_photo_area, for enclosed volume (relief volume = sum w*h*depth)."},
    {"name": "struct_share_photo_checked_volume", "unit": "share", "direction": "none", "aggregate": "mean", "group": "structure",
     "description": "As struct_share_photo_volume, for the photo-verified (OSM values checked by photos) share."},
    {"name": "struct_share_stub_area", "unit": "share", "direction": "lower", "aggregate": "mean", "group": "structure",
     "description": "Surface-area share attributed to placeholders."},
    {"name": "struct_photo_dof", "unit": "count", "direction": "higher", "aggregate": "sum", "group": "structure", "min_delta": 1,
     "description": "DETAIL SCORE. Independent numbers the photos put into the model: fitted values (a D15 outline "
                    "similarity = 3: k, tx, ty; a fitted height = 1) + per photographed wall the distinct measured "
                    "u-positions, z-levels and depth levels of its backed facade elements (values equal to the "
                    "wall's own bounds and repeats within 5 cm count once or not at all). Verified values add 0. "
                    "Unbacked, stub, convention and copied elements add 0."},
    {"name": "struct_photo_edge_m", "unit": "m", "direction": "higher", "aggregate": "sum", "group": "structure",
     "description": "Length of structural lines that exist only because photos measured them (element outlines)."},
    {"name": "struct_detail_coverage", "unit": "share", "direction": "higher", "aggregate": "mean", "group": "structure",
     "description": "Share of photographed wall+roof area on walls that carry >= 1 backed facade element."},
    {"name": "struct_roof_lines", "unit": "count", "direction": "none", "aggregate": "sum", "group": "structure",
     "description": "Roof structural lines the model lists (ridges, hips, valleys of the drawn parts' roof primitives)."},
    {"name": "struct_share_photo_verified_roof_lines", "unit": "share", "direction": "higher", "aggregate": "mean",
     "group": "structure",
     "description": "Share of those roof lines that 3-D lines from the photos lie along (within 0.3 m and 7 deg, "
                    "covering >= max(1 m, 25 %), >= 3 photos) BETTER than along the line's displaced copies (1.5 m "
                    "up/down/out/in, turned 15 deg; covered share >= 0.1 above the best copy's): photo-verified "
                    "edges above their per-line chance; the values stay the map's."},
    {"name": "struct_unbacked", "unit": "count", "direction": "lower", "aggregate": "sum", "group": "structure", "min_delta": 0,
     "description": "MUST BE 0 (a gate, not a trend). Honesty: parameters/elements claiming photo provenance without >= 2 measuring channels, elements "
                    "on unphotographed or stub surfaces, unknown provenance values. Must be 0."},
]
