"""Tiny GLB writer for the structure-metric tests (tests/test_structmetrics.py): nodes = [(name, extras, tris (n,3,3) ENU)]."""
import json, struct
import numpy as np


def write_glb(path, nodes, scene_extras=None):
    bin_ = bytearray()
    js = {"asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": list(range(len(nodes))), "extras": scene_extras or {}}],
          "nodes": [], "meshes": [], "accessors": [], "bufferViews": [], "buffers": []}
    for i, (name, extras, tris) in enumerate(nodes):
        t = np.asarray(tris, float).reshape(-1, 3)
        g = np.stack([t[:, 0], t[:, 2], -t[:, 1]], 1).astype(np.float32)     # ENU -> glTF y-up
        idx = np.arange(len(g), dtype=np.uint32)
        for arr, typ, comp in ((g, "VEC3", 5126), (idx, "SCALAR", 5125)):
            while len(bin_) % 4:
                bin_.append(0)
            off = len(bin_)
            bin_ += arr.tobytes()
            js["bufferViews"].append({"buffer": 0, "byteOffset": off, "byteLength": arr.nbytes})
            acc = {"bufferView": len(js["bufferViews"]) - 1, "componentType": comp, "count": len(arr), "type": typ}
            if typ == "VEC3":
                acc["min"], acc["max"] = g.min(0).tolist(), g.max(0).tolist()
            js["accessors"].append(acc)
        js["meshes"].append({"primitives": [{"attributes": {"POSITION": len(js["accessors"]) - 2}, "indices": len(js["accessors"]) - 1}]})
        js["nodes"].append({"name": name, "mesh": i, "extras": extras})
    while len(bin_) % 4:
        bin_.append(0)
    js["buffers"].append({"byteLength": len(bin_)})
    j = json.dumps(js).encode()
    j += b" " * ((4 - len(j) % 4) % 4)
    out = struct.pack("<4sII", b"glTF", 2, 12 + 8 + len(j) + 8 + len(bin_)) + struct.pack("<I4s", len(j), b"JSON") + j \
        + struct.pack("<I4s", len(bin_), b"BIN\x00") + bytes(bin_)
    open(path, "wb").write(out)


def quad(a, b, c, d):
    return [(a, b, c), (a, c, d)]


def box_part(pid, x0, y0, x1, y1, z0, z1, roof="flat", apex=None, wall_extras=None, roof_extras=None):
    """Nodes of an axis-aligned box part: 4 walls (CCW outward), a flat or pyramidal roof."""
    P = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    nodes = []
    for k in range(4):
        (ax, ay), (bx, by) = P[k], P[(k + 1) % 4]
        tris = quad((ax, ay, z0), (bx, by, z0), (bx, by, z1), (ax, ay, z1))
        ex = {"kind": "wall", "wall": k, "part": pid, "observed": False,
              "provenance": {"outline": "osm-s3db", "height": "osm-s3db", "appearance": "default-style"}}
        ex.update((wall_extras or {}).get(k, {}))
        nodes.append((f"{pid}-wall-{k}", ex, tris))
    rex = {"kind": "roof", "part": pid, "observed": False, "provenance": {"outline": "osm-s3db"}, "roof": {"shape": roof}}
    rex.update(roof_extras or {})
    if roof == "flat":
        nodes.append((f"{pid}-roof", rex, quad((x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1))))
    else:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        tris = [((ax, ay, z1), (bx, by, z1), (cx, cy, apex)) for (ax, ay), (bx, by) in zip(P, P[1:] + P[:1])]
        nodes.append((f"{pid}-roof", rex, tris))
    return nodes


def recess_wall(pid, k, L, H, u0, u1, zb, zt, depth, extras):
    """Wall k of a box (along +x at y=0, outward normal -y) with one rectangular recess (frame + recessed panel + sides)."""
    y = 0.0
    tris = []
    # frame around the hole: 4 rectangles
    for (a0, a1, b0, b1) in ((0, u0, 0, H), (u1, L, 0, H), (u0, u1, 0, zb), (u0, u1, zt, H)):
        if a1 > a0 and b1 > b0:
            tris += quad((a0, y, b0), (a1, y, b0), (a1, y, b1), (a0, y, b1))
    d = depth
    tris += quad((u0, y + d, zb), (u1, y + d, zb), (u1, y + d, zt), (u0, y + d, zt))         # recessed panel
    tris += quad((u0, y, zb), (u0, y + d, zb), (u0, y + d, zt), (u0, y, zt))                  # sides
    tris += quad((u1, y + d, zb), (u1, y, zb), (u1, y, zt), (u1, y + d, zt))
    tris += quad((u0, y + d, zb), (u0, y, zb), (u1, y, zb), (u1, y + d, zb))
    tris += quad((u0, y, zt), (u0, y + d, zt), (u1, y + d, zt), (u1, y, zt))
    ex = {"kind": "wall", "wall": k, "part": pid, "observed": True,
          "provenance": {"outline": "osm-s3db", "height": "osm-s3db", "appearance": "photo-observed"}}
    ex.update(extras)
    return (f"{pid}-wall-{k}", ex, tris)
