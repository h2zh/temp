"""D29 pins: what deleting the colouring code (decision D29) must not change.

The goldens in tests/fixtures/d29_pins.json were recorded at bc49e0d, before any D29 change, on synthetic scenes
(test_stylize's south facade with and without a shed in front, test_structure_stylize's gabled box and raised dome,
test_relief's recessed panel, and corner_scene below: two photographed walls with sky, a tree, per-photo exposures and
a colour cast, so that the scene norm is not the identity):

- the observation stage: which walls are observed (the `facades` keys), each wall's final observation mask (fac.mask:
  photometric consistency, scene norm, sky / vegetation removal, tidy mask) and its geometric mask (Orthophoto.mask)
  with its occluded share, the scene norm, the summary's walls[].observed / mask_fraction / geometric_mask_fraction,
  and the final masks of three orthophotos run alone (sky and a tree; sky through an arcade and foliage in shadow;
  observed on the left half only). D24 relief reads fac.mask (G-R5) and the facades keys;
They must pass before and after every D29 step. Only the accessor helpers below (observation_masks, geometric_masks,
split_wall, ...) may be adapted when a step renames what they read; the goldens never change in a D29 step. They
are exact float32 / mask digests, so a numpy or OpenCV upgrade can change them: then (and only at a commit where the
observation stage is known unchanged) regenerate with `PYTHONPATH=. python tests/test_d29_pins.py --regen`.
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import test_relief as tr
import test_structure_stylize as tss
import test_stylize as ts   # the synthetic south-facade scene (pytest puts tests/ on the path)

from crowd3d import structure as S
from crowd3d import stylize as st

GOLDEN = Path(__file__).resolve().parent / "fixtures" / "d29_pins.json"
PINNED_KINDS = ("wall", "base")           # per node; corners per corner; roofs as a triangle union per part


# ---------------------------------------------------------------- accessors (adapt these, never the goldens)


def observation_masks(model: st.StylizedModel) -> dict[int, np.ndarray]:
    """Per observed wall, its final observation mask (the walls the model counts as photographed)."""
    return {int(wi): fac.mask for wi, fac in model.facades.items()}


def geometric_masks(model: st.StylizedModel) -> dict[int, np.ndarray]:
    """Per wall with an orthophoto, its geometric mask (enough agreeing, unoccluded views with SfM support)."""
    return {int(wi): o.mask for wi, o in model.orthophotos.items()}


def occluded_shares(model: st.StylizedModel) -> dict[int, float]:
    """Per wall with an orthophoto, the share of its samples hidden by SfM points in front of it."""
    return {int(wi): o.stats.get("occluded_sample_fraction") for wi, o in model.orthophotos.items()}


def wall_records(model: st.StylizedModel) -> dict[str, list]:
    """Per wall of the summary: observed (counts as photographed), final and geometric mask fractions."""
    return {str(w["index"]): [bool(w["observed"]), w["mask_fraction"], w["geometric_mask_fraction"]]
            for w in model.summary["walls"]}


def scene_norm(model: st.StylizedModel) -> list:
    """The scene norm the observation masks were decided in: its white balance and linear gains."""
    meta = model.summary["color_norm"]
    return [meta["white_balance"], meta["lin_gains"]]


def ortho_observation_mask(ortho: st.Orthophoto) -> np.ndarray:
    """The final observation mask of one orthophoto run alone (identity scene norm)."""
    return st.stylize_facade(ortho).mask


def split_wall(wall: st.Wall, ua: float, ub: float, mask: np.ndarray | None) -> list[tuple[float, float, bool]]:
    """wall_pieces on an observation mask (None: the wall is not observed)."""
    return st.wall_pieces(wall, ua, ub, None if mask is None else SimpleNamespace(mask=mask), st.DEFAULT_PARAMS)


# ---------------------------------------------------------------- digests


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:32]


def mask_digest(mask: np.ndarray) -> list:
    m = np.ascontiguousarray(mask, bool)
    return [int(m.shape[0]), int(m.shape[1]), int(m.sum()), _sha(np.packbits(m).tobytes())]


def _node_arrays(gltf: dict, binary: bytes, node: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prim = gltf["meshes"][node["mesh"]]["primitives"][0]
    pos = np.ascontiguousarray(st.glb_accessor(gltf, binary, prim["attributes"]["POSITION"]), "<f4")
    nrm = np.ascontiguousarray(st.glb_accessor(gltf, binary, prim["attributes"]["NORMAL"]), "<f4")
    idx = np.ascontiguousarray(st.glb_accessor(gltf, binary, prim["indices"]), "<u4").reshape(-1, 3)
    return pos, nrm, idx


def _mesh_digest(pos: np.ndarray, nrm: np.ndarray, idx: np.ndarray) -> str:
    return f"{len(pos)}/{len(idx)}:" + _sha(pos.tobytes() + b"|" + nrm.tobytes() + b"|" + idx.tobytes())


def _triangle_union(tris: list[np.ndarray]) -> str:
    t = np.ascontiguousarray(np.concatenate(tris).reshape(-1, 9), "<f4")
    rows = np.sort(t.view(np.dtype((np.void, 36))).ravel())
    return f"{len(rows)}:" + _sha(rows.tobytes())


def glb_pins(glb: bytes) -> dict:
    """Per-node mesh digests of the pinned kinds (key: node name and its observed flag; relief walls carry their
    elements' ids and built flags), each corner's mesh, and each part's roof triangles as a set."""
    gltf, binary = st.read_glb(glb)
    nodes, corners, roofs = {}, {}, {}
    for node in gltf["nodes"]:
        ex = node.get("extras") or {}
        kind = ex.get("kind")
        pos, nrm, idx = _node_arrays(gltf, binary, node)
        if kind in PINNED_KINDS:
            rec = {"mesh": _mesh_digest(pos, nrm, idx), "observed": bool(ex.get("observed"))}
            if ex.get("relief"):
                rec["relief"] = [[r["id"], bool(r.get("built"))] for r in ex["relief"]]
            nodes[node["name"]] = rec
        elif kind == "corners":
            ks = ex["corners"]
            nv, nt = len(pos) // len(ks), len(idx) // len(ks)     # every corner: the same number of quads, in ks order
            for j, k in enumerate(ks):
                corners[f"{ex['prism']}-{ex['ring']}-{k}"] = _mesh_digest(pos[j * nv:(j + 1) * nv], nrm[j * nv:(j + 1) * nv],
                                                                          idx[j * nt:(j + 1) * nt] - j * nv)
        elif kind in ("roof", "roof-cut"):
            roofs.setdefault(f"{ex['prism']}-{kind}", []).append(pos[idx])
    return {"nodes": dict(sorted(nodes.items())), "corners": dict(sorted(corners.items())),
            "roofs": {k: _triangle_union(v) for k, v in sorted(roofs.items())}}


def model_pins(model: st.StylizedModel) -> dict:
    obs, geo = observation_masks(model), geometric_masks(model)
    return {"facades": sorted(obs), "observation_masks": {str(k): mask_digest(v) for k, v in sorted(obs.items())},
            "geometric_masks": {str(k): mask_digest(v) for k, v in sorted(geo.items())},
            "occluded_sample_fraction": {str(k): v for k, v in sorted(occluded_shares(model).items())},
            "walls": wall_records(model), "norm": scene_norm(model), **glb_pins(model.glb)}


# ---------------------------------------------------------------- the scenes


def _dome_scene() -> st.SceneInput:
    dome = S.PartSpec("w200", "yes", np.array([[5.0 * math.cos(a), 5.0 * math.sin(a)]
                                                for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]),
                      [], {"height": "13", "roof:shape": "dome", "roof:height": "3", "min_height": "10"})
    scene = ts.south_scene()
    scene.structure = tss.box_structure("flat", [dome])
    return scene


def _gabled_scene() -> st.SceneInput:
    scene = ts.south_scene()
    scene.structure = tss.box_structure()
    return scene


OCHRE, PLASTER, TREE = (222, 178, 110), (246, 246, 243), (60, 130, 45)
SKY, GROUND = np.array([150, 200, 250], np.uint8), np.array([120, 120, 110], np.uint8)


def _corner_facade(x: np.ndarray, z: np.ndarray, base) -> np.ndarray:
    out = ts.facade_colour(x, z)
    out[np.all(out == ts.CREAM, axis=-1)] = base
    return out


def render_corner(cam: st.PosedCamera, sky_from: float = 8.5, tree=(-4.0, -9.0, 3.0, 2.2)) -> np.ndarray:
    """The south (ochre, y = -6) and east (white plaster, x = 10) walls of ts.BOX; above sky_from the photos show sky
    (the modelled wall is taller than the building); a round green tree with no SfM points stands 3 m in front."""
    rays = ts._rays(cam)
    img = np.where((rays[..., 2] > 0)[..., None], SKY, GROUND).astype(np.uint8)
    best = np.full(rays.shape[:2], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (-6.0 - cam.center[1]) / rays[..., 1]
        hit = cam.center + t[..., None] * rays
        on = (t > 0) & (np.abs(hit[..., 0]) <= 10.0) & (hit[..., 2] >= 0) & (hit[..., 2] <= ts.WALL_H)
        img[on] = _corner_facade(hit[on][:, 0], hit[on][:, 2], OCHRE)
        best[on] = t[on]
        t2 = (10.0 - cam.center[0]) / rays[..., 0]
        hit2 = cam.center + t2[..., None] * rays
        on2 = (t2 > 0) & (np.abs(hit2[..., 1]) <= 6.0) & (hit2[..., 2] >= 0) & (hit2[..., 2] <= ts.WALL_H) & (t2 < best)
        img[on2] = _corner_facade(hit2[on2][:, 1] * 1.5, hit2[on2][:, 2], PLASTER)
        best[on2] = t2[on2]
        z = np.where(on, hit[..., 2], np.where(on2, hit2[..., 2], -1.0))
        img[(on | on2) & (z > sky_from)] = SKY
        cx, cy, cz, r = tree
        t3 = (cy - cam.center[1]) / rays[..., 1]
        h3 = cam.center + t3[..., None] * rays
        img[(t3 > 0) & ((h3[..., 0] - cx) ** 2 + (h3[..., 2] - cz) ** 2 <= r * r) & (t3 < best)] = TREE
    return img


def corner_scene(n: int = 8) -> st.SceneInput:
    """Two photographed walls meeting at the box's south-east corner (a photographed corner between them), with
    sky, a tree, per-photo exposures and a cool colour cast: the scene norm is not the identity (its neutral anchor
    is the plaster wall), and sky / vegetation removal and the consistency test shape both masks."""
    cams = []
    exposures = (1.0, 0.8, 1.1, 0.9, 1.0, 0.75, 1.05, 0.95)
    for i, a in enumerate(np.linspace(-80, -10, n)):
        c = np.array([10.0 + 28.0 * np.cos(np.radians(a)), -6.0 + 28.0 * np.sin(np.radians(a)), 1.6 + (i % 3)])
        cam = ts.look_at(c, [10.0 - 9 * (1 - i / (n - 1)), -6.0 + 7 * (i / (n - 1)), 5.0], name=f"k{i}")
        img = render_corner(cam).astype(np.float32) * np.array([0.97, 1.0, 1.0], np.float32) * exposures[i % 8]
        cam.image = np.clip(img, 0, 255).astype(np.uint8)
        cams.append(cam)
    gx, gz = np.meshgrid(np.arange(-9.75, 10, 0.25), np.arange(0.25, ts.WALL_H, 0.25))
    south = np.stack([gx.ravel(), np.full(gx.size, -6.0), gz.ravel()], 1)
    gy, gz2 = np.meshgrid(np.arange(-5.75, 6, 0.25), np.arange(0.25, ts.WALL_H, 0.25))
    east = np.stack([np.full(gy.size, 10.0), gy.ravel(), gz2.ravel()], 1)
    rx, ry = np.meshgrid(np.arange(-9.5, 10, 1.0), np.arange(-5.5, 6, 1.0))
    roof = np.stack([rx.ravel(), ry.ravel(), np.full(rx.size, ts.WALL_H)], 1)
    colors = np.vstack([_corner_facade(south[:, 0], south[:, 2], OCHRE), _corner_facade(east[:, 1] * 1.5, east[:, 2], PLASTER),
                        np.tile([[200, 90, 70]], (len(roof), 1))]).astype(np.uint8)
    return st.SceneInput("Corner Box", ts.BOX.copy(), np.vstack([south, east, roof]), colors, cams, snap=False)


SCENES = {
    "box": ts.south_scene,                                   # footprint path: bevelled corners, one observed wall
    "corner": corner_scene,                                  # two observed walls, sky, a tree, a non-identity norm
    "shed": lambda: ts.south_scene(shed=ts.SHED),            # an occluder in front of the photographed wall
    "gabled": _gabled_scene,                                 # structure path: gable ends, roof faces
    "dome": _dome_scene,                                     # a raised lantern: its base node
    "relief": tr.relief_scene,                               # a measured recess built on the photographed strip
}


def sky_tree_ortho() -> st.Orthophoto:
    ortho = ts.synthetic_ortho()
    ortho.rgb[:40] = (140, 195, 250)                          # sky above a too-tall wall
    ortho.rgb[120:, :60] = (70, 140, 50)                      # a tree
    return ortho


def arcade_ortho() -> st.Orthophoto:
    """Sky seen through two arcade openings low in the wall (pale-blue "accents": not the top-edge sky the clutter
    mask removes) and foliage in shadow (dark, busy, olive)."""
    ortho = ts.synthetic_ortho()
    ortho.rgb[150:190, 100:140] = (140, 195, 250)
    ortho.rgb[150:190, 200:240] = (140, 195, 250)
    leaves = np.array([40, 48, 25]) + np.random.default_rng(1).normal(0, 15, (50, 60, 3))
    ortho.rgb[120:170, 300:360] = np.clip(leaves, 0, 255).astype(np.uint8)
    return ortho


ORTHOS = {"sky_tree": sky_tree_ortho, "arcade": arcade_ortho, "left_only": lambda: ts.synthetic_ortho(mask_left_only=True)}


def split_pins(model: st.StylizedModel) -> dict:
    """wall_pieces of every wall of the box model, and of its photographed wall with the west half, both ends, all
    but a thin middle stripe, or a strip narrower than strip_min_m at its west end of its observation mask cleared."""
    extents = st.wall_extents(model.massing, st.DEFAULT_PARAMS)
    masks = observation_masks(model)
    r9 = lambda pieces: [[round(float(a), 9), round(float(b), 9), bool(o)] for a, b, o in pieces]
    out = {str(w.index): r9(split_wall(w, *extents[w.index], masks.get(w.index))) for w in model.massing.walls}
    wall, m = model.massing.walls[ts.SOUTH], masks[ts.SOUTH]
    w = m.shape[1]
    cuts = {"west_half": (slice(0, w // 2),), "ends": (slice(0, w // 5), slice(w - w // 7, w)),
            "stripe": (slice(0, w // 2 - 2), slice(w // 2 + 2, w)),
            "narrow_west": (slice(0, 18), slice(w - 30, w))}        # 0.78 m of clay at the west end: not split off
    for name, cols in cuts.items():
        mm = m.copy()
        for c in cols:
            mm[:, c] = False
        out[name] = r9(split_wall(wall, *extents[wall.index], mm))
    return out


@functools.lru_cache(maxsize=None)
def _model(name: str) -> st.StylizedModel:
    return st.stylize_landmark(SCENES[name]())


def compute(name: str) -> dict:
    if name in SCENES:
        return model_pins(_model(name))
    if name in ORTHOS:
        return {"observation_mask": mask_digest(ortho_observation_mask(ORTHOS[name]()))}
    if name == "split":
        return split_pins(_model("box"))
    raise KeyError(name)


ALL = [*SCENES, *ORTHOS, "split"]


# ---------------------------------------------------------------- the tests


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(GOLDEN.read_text())


def _diff(got, want, path="") -> list[str]:
    if isinstance(want, dict) and isinstance(got, dict):
        out = [f"{path}/{k}: missing" for k in want if k not in got] + [f"{path}/{k}: extra" for k in got if k not in want]
        return out + [d for k in want if k in got for d in _diff(got[k], want[k], f"{path}/{k}")]
    return [] if got == want else [f"{path}: {got!r} != {want!r}"]


@pytest.mark.parametrize("name", ALL)
def test_d29_pin(name, golden):
    diffs = _diff(compute(name), golden[name])
    assert not diffs, "\n".join(diffs[:40])


def test_the_pins_cover_every_pinned_node_kind(golden):
    """The goldens are not vacuous: they hold observed and clay walls, a split wall, corners (one between two
    photographed walls), a base, a relief wall, roofs, a wall whose occluded share is large, and final masks that the
    sky / vegetation removal and a non-identity scene norm shaped."""
    nodes = {k: v for s in SCENES for k, v in golden[s]["nodes"].items()}
    assert any(v["observed"] for v in nodes.values()) and any(not v["observed"] for v in nodes.values())
    assert any(k.startswith("base-") for s in SCENES for k in golden[s]["nodes"])
    assert any("relief" in v and any(b for _, b in v["relief"]) for v in golden["relief"]["nodes"].values())
    assert golden["box"]["corners"] and golden["gabled"]["roofs"] and golden["dome"]["roofs"]
    assert golden["shed"]["occluded_sample_fraction"][str(ts.SOUTH)] > 0.2
    assert golden["corner"]["facades"] == [0, 1] and golden["corner"]["norm"][1] not in (None, [1.0, 1.0, 1.0])
    assert any(v["observed"] for v in golden["corner"]["nodes"].values()) and len(golden["corner"]["corners"]) == 4
    for wi in ("0", "1"):                             # sky, the tree and inconsistent texels are out of the final mask
        assert golden["corner"]["observation_masks"][wi][2] < golden["corner"]["geometric_masks"][wi][2]
    assert len(golden["split"]["west_half"]) == 2 and len(golden["split"]["ends"]) == 3
    narrow = golden["split"]["narrow_west"]                       # the narrow west strip stays in the observed one
    assert len(narrow) == 2 and narrow[0][2] and narrow[0][0] == golden["split"][str(ts.SOUTH)][0][0]
    assert golden["arcade"]["observation_mask"][2] < golden["left_only"]["observation_mask"][2] * 2
    for s in SCENES:
        assert golden[s]["facades"] and golden[s]["observation_masks"]


if __name__ == "__main__" and "--regen" in sys.argv:
    GOLDEN.write_text(json.dumps({name: compute(name) for name in ALL}, indent=1, sort_keys=True) + "\n")
    print("wrote", GOLDEN)
