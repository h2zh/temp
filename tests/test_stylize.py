"""crowd3d.stylize on synthetic inputs: massing, observation, facade stylization, honesty, GLB."""

from __future__ import annotations

import io
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crowd3d import stylize as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# A 20 m x 12 m box (CCW): edge 0 is the south wall (y = -6), 1 east, 2 north, 3 west.
BOX = np.array([[-10.0, -6.0], [10.0, -6.0], [10.0, 6.0], [-10.0, 6.0]])
WALL_H = 10.0
SOUTH, NORTH = 0, 2
CREAM, INKY, RED = (235, 225, 205), (40, 45, 60), (190, 70, 50)
WINDOW_XS = (-7.5, -4.5, -1.5, 1.5, 4.5, 7.5)
WINDOW_ROWS = ((2.0, 4.0), (6.0, 8.0))           # z ranges
DOOR_X, DOOR_Z = (8.4, 9.6), (0.0, 4.0)           # a door at the EAST end only: the facade is not symmetric


def facade_colour(x: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Procedural south facade: cream wall, two rows of six dark windows (1.2 m x 2 m), a red band and
    one tall door at the east end (x = 8.4 .. 9.6 m)."""
    out = np.empty(x.shape + (3,), np.uint8)
    out[:] = CREAM
    band = (z >= 5.0) & (z <= 5.4)
    out[band] = RED
    for z0, z1 in WINDOW_ROWS:
        for cx in WINDOW_XS:
            out[(np.abs(x - cx) <= 0.6) & (z >= z0) & (z <= z1)] = INKY
    out[(x >= DOOR_X[0]) & (x <= DOOR_X[1]) & (z >= DOOR_Z[0]) & (z <= DOOR_Z[1])] = INKY
    return out


def look_at(center, target, w=320, h=240, f=250.0, name="cam") -> st.PosedCamera:
    c = np.asarray(center, float)
    fwd = np.asarray(target, float) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0.0, 0.0, 1.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    return st.PosedCamera(name, w, h, f, f, w / 2, h / 2, np.stack([right, down, fwd]), c)


def _rays(cam: st.PosedCamera) -> np.ndarray:
    v, u = np.mgrid[0:cam.height, 0:cam.width]
    return np.stack([(u + 0.5 - cam.cx) / cam.fx, (v + 0.5 - cam.cy) / cam.fy, np.ones(u.shape)], -1) @ cam.rot


def render_south(cam: st.PosedCamera, wall_y: float = -6.0, shed: dict | None = None) -> np.ndarray:
    """What a camera sees of the procedural south facade standing at y = wall_y (sky above, ground
    below), optionally with a red shed (a vertical rectangle) standing in front of it."""
    rays = _rays(cam)
    img = np.where((rays[..., 2] > 0)[..., None], np.array([150, 200, 250], np.uint8),
                   np.array([120, 120, 110], np.uint8)).astype(np.uint8)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (wall_y - cam.center[1]) / rays[..., 1]
    hit = cam.center + t[..., None] * rays
    on = (t > 0) & (np.abs(hit[..., 0]) <= 10.0) & (hit[..., 2] >= 0) & (hit[..., 2] <= WALL_H)
    img[on] = facade_colour(hit[on][:, 0], hit[on][:, 2])
    if shed:
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (shed["y"] - cam.center[1]) / rays[..., 1]
        hit = cam.center + t[..., None] * rays
        on = ((t > 0) & (hit[..., 0] >= shed["x"][0]) & (hit[..., 0] <= shed["x"][1]) & (hit[..., 2] >= 0)
              & (hit[..., 2] <= shed["z"]))
        img[on] = shed["rgb"]
    return img


def south_scene(num_cams: int = 6, wall_y: float = -6.0, shed: dict | None = None) -> st.SceneInput:
    """Only the south facade is photographed: cameras ~24 m in front of it, points on it and on the roof
    (and on the shed, when there is one)."""
    cams = []
    for i, x in enumerate(np.linspace(-8, 8, num_cams)):
        cam = look_at([x, -30.0, 1.6 + (i % 3)], [0.0, -6.0, 5.0], name=f"c{i}")
        cam.image = render_south(cam, wall_y, shed)
        cams.append(cam)
    gx, gz = np.meshgrid(np.arange(-9.75, 10, 0.25), np.arange(0.25, WALL_H, 0.25))
    wall = np.stack([gx.ravel(), np.full(gx.size, wall_y), gz.ravel()], 1)
    rx, ry = np.meshgrid(np.arange(-9.5, 10, 1.0), np.arange(-5.5, 6, 1.0))
    roof = np.stack([rx.ravel(), ry.ravel(), np.full(rx.size, WALL_H)], 1)
    points = [wall, roof]
    colors = [facade_colour(wall[:, 0], wall[:, 2]), np.tile([[200, 90, 70]], (len(roof), 1))]
    if shed:
        sx, sz = np.meshgrid(np.arange(shed["x"][0], shed["x"][1], 0.25), np.arange(0.25, shed["z"], 0.25))
        pts = np.stack([sx.ravel(), np.full(sx.size, shed["y"]), sz.ravel()], 1)
        points.append(pts)
        colors.append(np.tile([shed["rgb"]], (len(pts), 1)))
    return st.SceneInput("Test Box", BOX.copy(), np.vstack(points), np.vstack(colors).astype(np.uint8), cams, snap=False)


def is_reddish(rgb: np.ndarray) -> np.ndarray:
    return st.rgb_to_lab(np.asarray(rgb, np.float32).reshape(-1, 3))[:, 1] > 25


@pytest.fixture(scope="module")
def box_model() -> st.StylizedModel:
    return st.stylize_landmark(south_scene())


# ------------------------------------------------------------------ footprint and massing


def test_noisy_rectangle_simplifies_to_an_orthogonal_box():
    rng = np.random.default_rng(1)
    t = np.linspace(0, 1, 30, endpoint=False)
    edges = [np.stack([-10 + 20 * t, np.full_like(t, -6)], 1), np.stack([np.full_like(t, 10), -6 + 12 * t], 1),
             np.stack([10 - 20 * t, np.full_like(t, 6)], 1), np.stack([np.full_like(t, -10), 6 - 12 * t], 1)]
    noisy = np.vstack(edges) + rng.normal(0, 0.15, (120, 2))
    rot = math.radians(7.0)
    noisy = noisy @ np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]]).T
    poly, report = st.simplify_footprint(noisy)
    assert len(poly) == 4
    assert report["orthogonal_snap"]
    d = np.roll(poly, -1, axis=0) - poly
    cosines = [abs(np.dot(d[k], d[(k + 1) % 4])) / (np.linalg.norm(d[k]) * np.linalg.norm(d[(k + 1) % 4])) for k in range(4)]
    assert max(cosines) < 1e-6                                 # right angles
    assert abs(st.signed_area(poly) - 240.0) < 15.0
    assert st.signed_area(poly) > 0                            # CCW


def test_octagon_keeps_its_eight_sides_and_is_not_squared_off():
    ang = np.radians(np.arange(8) * 45.0 + 22.5)
    octagon = 15.0 * np.stack([np.cos(ang), np.sin(ang)], 1)
    poly, report = st.simplify_footprint(octagon)
    assert len(poly) == 8
    assert not report["orthogonal_snap"]
    assert report["hausdorff_m"] < 0.05


def test_detailed_outline_is_simplified_to_the_vertex_budget():
    ang = np.linspace(0, 2 * np.pi, 200, endpoint=False)
    r = 40.0 + 1.5 * np.sin(ang * 23)                         # a 200-vertex wiggly outline
    poly, report = st.simplify_footprint(np.stack([r * np.cos(ang), r * np.sin(ang)], 1))
    assert report["input_vertices"] == 200
    assert len(poly) <= st.DEFAULT_PARAMS.max_footprint_vertices
    assert report["hausdorff_m"] <= max(1.0, st.DEFAULT_PARAMS.max_hausdorff_frac * 83.0)


def test_extrusion_without_points_uses_the_height_facts_in_priority_order():
    m = st.build_massing(BOX, None, st.HeightFacts())
    assert m.prisms[0].z1 == st.DEFAULT_PARAMS.default_height_m
    assert m.prisms[0].height_source == st.PROV_DEFAULT
    # Wikidata P2048 is a total height (spire / dome tip): it does not extrude a 23 m wide outline ...
    m = st.build_massing(BOX, None, st.HeightFacts(wikidata_height_m=31.0))
    assert (m.prisms[0].z1, m.prisms[0].height_source) == (st.DEFAULT_PARAMS.default_height_m, st.PROV_DEFAULT)
    assert "total height" in m.report["fallback_height"]["note"]
    big = np.array([[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]])
    m = st.build_massing(big, None, st.HeightFacts(wikidata_height_m=94.0))
    assert m.prisms[0].z1 == st.DEFAULT_PARAMS.default_height_m           # no 94 m block over the whole outline
    # ... but it does extrude a tower
    tower = np.array([[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]])
    m = st.build_massing(tower, None, st.HeightFacts(wikidata_height_m=56.0))
    assert (m.prisms[0].z1, m.prisms[0].height_source) == (56.0, st.PROV_WIKIDATA)
    # OSM: building:levels (eaves) before height (top of the roof); height - roof:height when both are known
    m = st.build_massing(BOX, None, st.HeightFacts(osm_levels=4, wikidata_height_m=31.0))
    assert (m.prisms[0].z1, m.prisms[0].height_source) == (12.0, st.PROV_OSM_HEIGHT)
    m = st.build_massing(BOX, None, st.HeightFacts(osm_height_m=20.0, osm_roof_height_m=6.0, osm_levels=4))
    assert (m.prisms[0].z1, m.prisms[0].height_source) == (14.0, st.PROV_OSM_HEIGHT)
    m = st.build_massing(BOX, None, st.HeightFacts(osm_height_m=20.0))
    assert (m.prisms[0].z1, m.prisms[0].height_source) == (20.0, st.PROV_OSM_HEIGHT)
    assert len(m.walls) == 4 and all(w.outline_source == st.PROV_OSM for w in m.walls)
    assert m.prisms[0].cap == "flat"


def test_extrusion_height_comes_from_the_points_on_the_walls():
    scene = south_scene(num_cams=2)
    m = st.build_massing(BOX, scene.points, st.HeightFacts(osm_height_m=30.0))
    assert m.prisms[0].height_source == st.PROV_SFM
    assert abs(m.prisms[0].z1 - WALL_H) < 0.6                 # measured, not the (wrong) tag
    assert len(m.prisms) == 1


def test_a_block_standing_on_the_roof_becomes_its_own_prism():
    rng = np.random.default_rng(0)
    walls = []
    for (ax, ay), (bx, by) in zip(BOX, np.roll(BOX, -1, axis=0)):
        t = rng.random(600)
        walls.append(np.stack([ax + t * (bx - ax), ay + t * (by - ay), rng.random(600) * 8.0], 1))
    roof = np.stack([rng.uniform(-9.5, 9.5, 3000), rng.uniform(-5.5, 5.5, 3000), np.full(3000, 8.0)], 1)
    tower = np.stack([rng.uniform(2, 6, 3000), rng.uniform(-2, 2, 3000), np.full(3000, 20.0)], 1)
    roof = roof[~((roof[:, 0] > 1.5) & (roof[:, 0] < 6.5) & (np.abs(roof[:, 1]) < 2.5))]
    m = st.build_massing(BOX, np.vstack(walls + [roof, tower]))
    assert len(m.prisms) == 2
    block = m.prisms[1]
    assert block.parent == 0 and block.outline_source == st.PROV_SFM
    assert abs(block.z1 - 20.0) < 1.0 and abs(block.z0 - m.prisms[0].z1) < 1e-6
    assert len(block.polygon) == 4                             # a near-rectangle is regularized to one
    assert abs(abs(st.signed_area(block.polygon)) - 16.0) < 6.0


def test_bevel_and_triangulation_keep_the_outline():
    ring, owner = st.bevel_polygon(BOX, 0.5, 3)
    assert len(ring) == 4 * 4 and len(owner) == len(ring)
    assert st.points_in_polygon(ring * 0.999, BOX).all()
    assert abs(st.signed_area(ring) - 240.0) < 1.0
    ell = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
    tris = st.triangulate_polygon(ell)
    area = sum(abs(st.signed_area(ell[list(t)])) for t in tris)
    assert len(tris) == 4 and abs(area - st.signed_area(ell)) < 1e-9


# ------------------------------------------------------------------ observation


def test_only_the_photographed_wall_is_observed(box_model):
    obs = box_model.observations
    assert obs[SOUTH].observed and len(obs[SOUTH].cameras) >= 4
    for k in (1, 2, 3):
        assert not obs[k].observed, obs[k].reason
    assert "camera" in obs[NORTH].reason                       # nobody stands north of it
    assert set(box_model.facades) == {SOUTH}


def test_a_wall_seen_by_cameras_but_without_points_stays_locked():
    scene = south_scene(num_cams=4)
    scene.points = scene.points[scene.points[:, 2] >= WALL_H - 0.01]      # roof only, nothing on the wall
    model = st.stylize_landmark(scene)
    assert not model.observations[SOUTH].observed
    assert "SfM points" in model.observations[SOUTH].reason
    assert model.facades == {}


def test_cameras_behind_a_wall_do_not_count():
    m = st.build_massing(BOX, None, st.HeightFacts(osm_height_m=WALL_H))
    inside = [look_at([0.0, 0.0, 2.0], [0.0, -6.0, 5.0], name="inside")]
    north = [look_at([x, 30.0, 2.0], [0.0, -6.0, 5.0], name=f"n{x}") for x in (-5, 5)]   # looking through the building
    obs = st.observe_walls(m, inside + north, None)
    assert not obs[SOUTH].observed and obs[SOUTH].cameras == []


# ------------------------------------------------------------------ facade stylization


def synthetic_ortho(mask_left_only: bool = False, ppm: float = 20.0) -> st.Orthophoto:
    w, h = int(20 * ppm), int(WALL_H * ppm)
    x = -10 + (np.arange(w) + 0.5) / ppm
    z = WALL_H - (np.arange(h) + 0.5) / ppm
    xx, zz = np.meshgrid(x, z)
    rgb = facade_colour(xx, zz).astype(np.int16)
    rgb = np.clip(rgb + np.random.default_rng(0).normal(0, 6, rgb.shape), 0, 255).astype(np.uint8)
    mask = np.ones((h, w), bool)
    if mask_left_only:
        mask[:, w // 2:] = False
    return st.Orthophoto(rgb, np.full((h, w), 3, np.int32), mask, ppm, ["a", "b", "c"])


def test_stylized_facade_uses_palette_colours_only_and_finds_the_windows():
    fac = st.stylize_facade(synthetic_ortho())
    colours = {tuple(int(v) for v in c) for c in np.unique(fac.texture.reshape(-1, 3), axis=0)}
    assert colours <= st.allowed_texture_colors()
    assert len(fac.colors) <= 4
    assert fac.base_rgb in {tuple(int(v) for v in c) for c in st.PALETTE_RGB}
    assert st.rgb_to_lab(np.array([fac.base_rgb], np.float32))[0, 0] > 80     # a light, cheerful wall
    assert 10 <= len(fac.openings) <= 14
    # regularized: the openings of a row share their top and their width
    tops = sorted({o["y"] for o in fac.openings})
    assert len(tops) == 2
    widths = {o["w"] for o in fac.openings if o["y"] == tops[0]}
    assert len(widths) == 1 and abs(widths.pop() - 1.2) < 0.35


def test_nothing_is_drawn_outside_the_observation_mask():
    ortho = synthetic_ortho(mask_left_only=True)
    fac = st.stylize_facade(ortho)
    right = fac.texture[:, fac.texture.shape[1] // 2:]
    assert st.is_clay_pattern(right).all()                   # clay (with its hatch), not even the wall colour
    assert st.is_clay_pattern(fac.texture[~fac.mask]).all()
    assert all(o["x"] + o["w"] <= 10.0 + 0.3 for o in fac.openings)              # left half: x < 10 m
    assert len(fac.openings) >= 4
    part = st.Part("wall-0", np.zeros((4, 3)), np.zeros((4, 3)), np.array([[0, 1, 2]]), fac.base_rgb, fac.texture,
                   np.zeros((4, 2)), {"kind": "wall", "locked": False,
                                      "provenance": {"outline": st.PROV_OSM, "height": st.PROV_SFM,
                                                     "appearance": st.PROV_PHOTO}})
    assert st.check_honesty([part], {0: fac})["passed"]
    fac.texture[5, -5] = fac.base_rgb                                            # even the plain wall colour ...
    report = st.check_honesty([part], {0: fac})
    assert not report["passed"] and report["detail_pixels_outside_mask"] == 1   # ... may not bleed out of the mask


def test_sky_and_vegetation_are_not_wall():
    ortho = synthetic_ortho()
    ortho.rgb[:40] = (140, 195, 250)                          # sky above a too-tall wall
    ortho.rgb[120:, :60] = (70, 140, 50)                      # a tree
    fac = st.stylize_facade(ortho)
    assert not fac.mask[:30].any()
    assert not fac.mask[150:, 10:50].any()
    assert st.is_clay_pattern(fac.texture[:30]).all()


def test_palette_snapping_keeps_greys_grey():
    greys = st.rgb_to_lab(np.array([[g, g, g] for g in (30, 90, 150, 210, 250)], np.float32))
    idx = st.snap_lab_to_palette(greys)
    assert set(idx.tolist()) <= set(st.NEUTRALS.tolist())
    assert list(st.PALETTE_LAB[idx, 0]) == sorted(st.PALETTE_LAB[idx, 0])          # order of lightness kept
    red = st.snap_lab_to_palette(st.rgb_to_lab(np.array([[180, 60, 40]], np.float32)))[0]
    assert st.PALETTE_NAMES[red] in {"brick", "terracotta", "rust"}


# ------------------------------------------------------------------ whole model and GLB


def test_box_model_summary_and_qa(box_model):
    s = box_model.summary
    assert s["qa"]["passed"], s["qa"]
    assert s["observed_walls"] == 1 and s["num_walls"] == 4
    assert abs(s["observed_wall_area_fraction"] - 200.0 / 640.0) < 0.02
    assert 0 < s["observed_pixel_area_fraction"] <= s["observed_wall_area_fraction"]
    assert s["glb_bytes"] == len(box_model.glb) and s["glb_bytes"] < 200_000
    fac = box_model.facades[SOUTH]
    assert len(fac.openings) >= 8                             # the windows survive the photo -> ortho -> style chain
    json.dumps(s)                                             # the summary is JSON-able as is


def test_glb_round_trips_with_provenance_on_every_node(box_model):
    gltf, binary = st.read_glb(box_model.glb)
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["scenes"][0]["extras"]["locked_label"] == "待解锁"
    nodes = gltf["nodes"]
    assert len(nodes) == len(box_model.parts) == box_model.summary["nodes"]
    for node, part in zip(nodes, box_model.parts):
        prov = node["extras"]["provenance"]
        assert all(prov[k] in st.PROVENANCES for k in st.PROVENANCE_KEYS)
        prim = gltf["meshes"][node["mesh"]]["primitives"][0]
        pos = st.glb_accessor(gltf, binary, prim["attributes"]["POSITION"])
        assert np.allclose(pos, st._enu_to_gltf(part.positions), atol=1e-4)
        idx = st.glb_accessor(gltf, binary, prim["indices"])
        assert len(idx) == 3 * len(part.indices) and idx.max() < len(pos)
    walls = [n["extras"] for n in nodes if n["extras"]["kind"] == "wall"]
    assert [w["wall"] for w in walls if not w["locked"]] == [SOUTH]
    assert all(w["provenance"]["appearance"] == st.PROV_DEFAULT for w in walls if w["locked"])
    # one atlas holding the one facade texture (plus its gutter), a decodable palette PNG whose
    # pixels under the wall's UVs are the stylized facade's (the second image: its texel provenance)
    assert len(gltf["textures"]) == 1 and box_model.summary["atlases"] == 1
    assert gltf["scenes"][0]["extras"]["texel_provenance"]["images"] == [1] and len(gltf["images"]) == 2
    png = _atlas_png(gltf, binary)
    node = next(n for n in nodes if n["extras"].get("wall") == SOUTH and n["extras"]["observed"])
    part = next(p for p in box_model.parts if p.name == node["name"])
    tex = part.texture
    assert png.shape[0] >= tex.shape[0] and png.shape[1] >= tex.shape[1]
    prim = gltf["meshes"][node["mesh"]]["primitives"][0]
    uv = st.glb_accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"])
    x0, y0 = np.rint(uv.min(axis=0) * png.shape[1::-1]).astype(int)
    x1, y1 = np.rint(uv.max(axis=0) * png.shape[1::-1]).astype(int)
    tx0, tx1 = np.rint(np.array([part.uvs[:, 0].min(), part.uvs[:, 0].max()]) * tex.shape[1]).astype(int)
    assert np.array_equal(png[y0:y1, x0:x1], tex[:, tx0:tx1])
    assert node["extras"]["photos"] and all("photo" in p and "share" in p for p in node["extras"]["photos"])
    reg = gltf["scenes"][0]["extras"]["registration"]
    assert reg["accepted"] is True


def _atlas_png(gltf: dict, binary: bytes) -> np.ndarray:
    bv = gltf["bufferViews"][gltf["images"][0]["bufferView"]]
    return np.asarray(Image.open(io.BytesIO(binary[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]])).convert("RGB"))


def test_an_asymmetric_facade_keeps_its_door_at_the_east_end_in_texture_and_glb(box_model):
    """A mirrored UV map would pass a symmetric facade; this one has a single door at the east end."""
    fac = box_model.facades[SOUTH]
    doors = [o for o in fac.openings if o["door"]]
    assert len(doors) == 1
    centre = doors[0]["x"] + doors[0]["w"] / 2                 # metres from the wall's start a (west end)
    assert abs(centre - (np.mean(DOOR_X) + 10.0)) < 0.6
    # in the GLB: the vertices at the east end (ENU x = +10) carry the door's side of the atlas
    gltf, binary = st.read_glb(box_model.glb)
    png = _atlas_png(gltf, binary)
    node = next(n for n in gltf["nodes"] if n["extras"].get("wall") == SOUTH and n["extras"]["observed"])
    prim = gltf["meshes"][node["mesh"]]["primitives"][0]
    pos = st.glb_accessor(gltf, binary, prim["attributes"]["POSITION"])     # glTF: x = east
    uv = st.glb_accessor(gltf, binary, prim["attributes"]["TEXCOORD_0"])
    east, west = uv[pos[:, 0].argmax()], uv[pos[:, 0].argmin()]
    assert east[0] > west[0]
    # sample the atlas at the door (u along the strip, v near the bottom) and at the mirrored spot
    def sample(frac_u: float, frac_v: float) -> np.ndarray:
        u = west[0] + frac_u * (east[0] - west[0])
        v = uv[:, 1].min() + frac_v * (uv[:, 1].max() - uv[:, 1].min())
        return png[int(v * png.shape[0]), int(u * png.shape[1])].astype(float)
    door_u = (np.mean(DOOR_X) + 10.0) / 20.0
    dark = st.rgb_to_lab(np.array([sample(door_u, 0.85)], np.float32))[0, 0]
    mirrored = st.rgb_to_lab(np.array([sample(1.0 - door_u, 0.85)], np.float32))[0, 0]
    assert dark < 45 and mirrored > 70, (dark, mirrored)


def test_atlas_packing_places_every_texture_without_overlap():
    rng = np.random.default_rng(3)
    texs = [np.full((int(h), int(w), 3), i, np.uint8) for i, (h, w) in enumerate(rng.integers(20, 300, (40, 2)))]
    atlases, slots = st.pack_atlases(texs, max_size=512, gutter=2)
    assert len(atlases) >= 2 and all(a.shape[0] <= 512 and a.shape[1] <= 512 for a in atlases)
    for i, (a, u0, v0, du, dv) in enumerate(slots):
        img = atlases[a]
        x0, y0 = round(u0 * img.shape[1]), round(v0 * img.shape[0])
        h, w = texs[i].shape[:2]
        assert round(du * img.shape[1]) == w and round(dv * img.shape[0]) == h
        assert (img[y0:y0 + h, x0:x0 + w] == i).all()        # nobody painted over it


def test_read_glb_rejects_garbage():
    with pytest.raises(ValueError):
        st.read_glb(b"not a glb at all, sorry")
    good = st.write_glb([st.Part("p", np.zeros((3, 3)), np.tile([0.0, 0.0, 1.0], (3, 1)), np.array([[0, 1, 2]]),
                                 (200, 100, 50))])
    st.read_glb(good)
    with pytest.raises(ValueError):
        st.read_glb(good[:-4])


def test_snapping_moves_a_shifted_model_onto_the_footprint_and_refuses_a_bad_fit():
    scene = south_scene(num_cams=2)
    rng = np.random.default_rng(5)
    walls = []
    for (ax, ay), (bx, by) in zip(BOX, np.roll(BOX, -1, axis=0)):
        t = rng.random(800)
        walls.append(np.stack([ax + t * (bx - ax), ay + t * (by - ay), 2 + rng.random(800) * 8.0], 1))
    pts = np.vstack(walls)
    rot = math.radians(4.0)
    r = np.array([[math.cos(rot), -math.sin(rot), 0], [math.sin(rot), math.cos(rot), 0], [0, 0, 1]])
    moved = pts @ r.T + np.array([3.0, -2.0, 0.0])
    snap = st.snap_to_footprint(moved, BOX)
    assert snap.accepted and snap.median_after_m < 0.2
    back = st.apply_snap_points(snap, moved)
    assert np.median(np.linalg.norm(back[:, :2] - pts[:, :2], axis=1)) < 0.3
    noise = np.stack([rng.uniform(-30, 30, 3000), rng.uniform(-30, 30, 3000), rng.uniform(2, 10, 3000)], 1)
    bad = st.snap_to_footprint(noise, BOX)
    assert not bad.accepted and bad.rotation_deg == 0.0 and not bad.shift.any()
    assert scene.footprint.shape == (4, 2)


# ------------------------------------------------------------------ honesty regressions (review findings)


SHED = {"y": -12.0, "x": (-10.0, -1.0), "z": 6.0, "rgb": (200, 40, 40)}


def test_an_occluder_in_front_of_the_wall_is_not_painted_onto_it():
    """Reviewer's t_occluder: a red shed 6 m in front of the south wall, seen by every camera and in the
    SfM points. Before the fix 30 % of the facade texture came out red and QA passed."""
    shed = st.stylize_landmark(south_scene(shed=SHED))
    clean = st.stylize_landmark(south_scene())
    red_shed = is_reddish(shed.facades[SOUTH].texture).mean()
    red_clean = is_reddish(clean.facades[SOUTH].texture).mean()            # the facade's own red band
    assert red_shed <= red_clean + 0.01, (red_shed, red_clean)
    o = shed.orthophotos[SOUTH]
    assert o.stats["occluded_sample_fraction"] > 0.2
    # the wall behind the shed (lower west part) is not observed; the east part still is
    h, w = o.mask.shape
    assert o.mask[int(0.75 * h):, : int(0.25 * w)].mean() < 0.2
    assert o.mask[int(0.75 * h):, int(0.75 * w):].mean() > 0.6
    # the colour plan was not taken over by the shed: same wall colour as without it
    assert shed.summary["color_plan"]["wall"] == clean.summary["color_plan"]["wall"]
    assert shed.summary["qa"]["passed"], shed.summary["qa"]


def test_the_audit_catches_an_occluder_the_mask_missed():
    shed = st.stylize_landmark(south_scene(shed=SHED), st.StyleParams(occluder_margin_m=50.0))   # occlusion test off
    assert is_reddish(shed.facades[SOUTH].texture).mean() > 0.1
    assert not shed.summary["qa"]["audit"]["passed"]
    assert shed.summary["qa"]["audit"]["occlusion"]["occluded_share"] > 0.05


def test_a_partly_seen_flat_roof_is_clay_with_an_observed_patch():
    scene = south_scene()
    keep = ~((scene.points[:, 2] >= WALL_H - 0.01) & (scene.points[:, 1] > 0.0))   # roof points on the south half only
    rng = np.random.default_rng(7)
    side = np.vstack([np.stack([np.full(300, x), rng.uniform(-6, 6, 300), rng.uniform(1, 9, 300)], 1) for x in (-10.0, 10.0)])
    scene.points = np.vstack([scene.points[keep], side])          # (east / west walls in the points: registration)
    scene.colors = np.vstack([scene.colors[keep], np.full((len(side), 3), 200, np.uint8)])
    m = st.stylize_landmark(scene)
    roofs = [p for p in m.parts if p.extras["kind"] == "roof"]
    patches = [p for p in m.parts if p.extras["kind"] == "roof-patch" and p.extras["observed"]]
    assert len(roofs) == 1 and not roofs[0].extras["observed"]            # the cap as a whole: clay
    assert roofs[0].extras["provenance"]["appearance"] == st.PROV_DEFAULT
    assert 0.35 < roofs[0].extras["coverage"][0] < 0.65
    assert patches and all(p.extras["provenance"]["appearance"] == st.PROV_PHOTO for p in patches)
    area = sum(p.extras["area_m2"] for p in patches)
    assert 0.3 * 240 < area < 0.6 * 240                                       # about the photographed half
    centroid_y = np.mean(np.concatenate([p.positions[:, 1] for p in patches]))
    assert centroid_y < 0                                                     # on the south half
    assert m.summary["qa"]["passed"], m.summary["qa"]
    # a whole cap claiming photos with half its area unseen fails the independent audit
    roofs[0].extras["observed"], roofs[0].extras["locked"] = True, False
    roofs[0].extras["provenance"]["appearance"] = st.PROV_PHOTO
    audit = st.audit_model(m.massing, m.parts, m.facades, m.orthophotos, m.observations,
                           scene.points, [BOX], scene.cameras)
    assert not audit["passed"] and audit["roof_mismatches"]


def test_a_partly_seen_wall_is_split_and_unseen_pixels_are_clay():
    m = st.stylize_landmark(south_scene())
    fac = m.facades[SOUTH]
    fac.mask[:, : fac.mask.shape[1] // 2] = False          # pretend nobody saw the west half
    parts = st.build_parts(m.massing, m.facades, m.roof, st.DEFAULT_PARAMS, m.observations)
    south = [p for p in parts if p.extras["kind"] == "wall" and p.extras["wall"] == SOUTH]
    assert len(south) == 2
    clay, seen = sorted(south, key=lambda p: p.extras["observed"])
    assert not clay.extras["observed"] and clay.extras["provenance"]["appearance"] == st.PROV_DEFAULT and clay.texture is None
    assert seen.extras["observed"] and abs(clay.extras["area_m2"] - seen.extras["area_m2"]) < 0.3 * seen.extras["area_m2"]
    assert clay.positions[:, 0].max() <= seen.positions[:, 0].min() + 1e-6      # the clay strip is the west one


def test_heights_carried_over_to_unmeasured_walls_are_inferred(box_model):
    walls = {w.index: w for w in box_model.massing.walls}
    assert walls[SOUTH].height_provenance == st.PROV_SFM
    assert all(walls[k].height_provenance == st.PROV_INFERRED for k in (1, 2, 3))
    gltf, _ = st.read_glb(box_model.glb)
    heights = {n["extras"]["wall"]: n["extras"]["provenance"]["height"] for n in gltf["nodes"] if n["extras"]["kind"] == "wall"}
    assert heights[SOUTH] == st.PROV_SFM and heights[NORTH] == st.PROV_INFERRED
    assert st.PROV_INFERRED in gltf["scenes"][0]["extras"]["provenance_values"]


def test_a_rejected_registration_publishes_clay_only():
    scene = south_scene()
    scene.points = scene.points + np.array([25.0, 0.0, 0.0])        # a model 25 m off its footprint
    for c in scene.cameras:
        c.center = c.center + np.array([25.0, 0.0, 0.0])
    scene.snap = True
    m = st.stylize_landmark(scene)
    assert not m.summary["registration"]["accepted"] and m.summary["registration"]["reason"]
    assert m.facades == {}
    assert all(p.extras["provenance"]["appearance"] == st.PROV_DEFAULT for p in m.parts)
    assert all("registration not verified" in o.reason for o in m.observations)
    assert m.summary["qa"]["passed"]


def test_registration_needs_most_supported_edges_not_just_the_best_facade():
    rng = np.random.default_rng(2)
    walls = []
    for (ax, ay), (bx, by) in zip(BOX, np.roll(BOX, -1, axis=0)):
        t = rng.random(800)
        walls.append(np.stack([ax + t * (bx - ax), ay + t * (by - ay), 2 + rng.random(800) * 8.0], 1))
    pts = np.vstack(walls) * np.array([2.0, 2.5, 1.0])               # a 40 m x 30 m building
    outline = BOX * np.array([2.0, 2.5])
    assert st.registration_check(pts, [outline])["accepted"]
    big = outline * 1.25                                              # an outline that includes a plinth
    assert not st.registration_check(pts, [big])["accepted"]
    shifted = outline + np.array([25.0, 0.0])
    assert not st.registration_check(pts, [shifted])["accepted"]


def test_a_facade_plane_off_the_outline_is_fitted_moved_inward_or_refused_outward():
    # 2 m inside the outline: the face is moved onto the fitted plane and textured from it
    m = st.stylize_landmark(south_scene(wall_y=-4.0))
    south = m.massing.walls[SOUTH]
    assert south.outline_source == st.PROV_SFM and abs(south.a[1] + 4.0) < 0.3
    assert SOUTH in m.facades and len(m.facades[SOUTH].openings) >= 8
    # 2.5 m outside it: something in front of the building as far as the footprint knows; never textured
    m = st.stylize_landmark(south_scene(wall_y=-8.5))
    assert SOUTH not in m.facades and "outside" in m.observations[SOUTH].reason
    assert m.massing.walls[SOUTH].outline_source == st.PROV_OSM


def test_fit_wall_plane_recovers_offset_and_yaw_without_clipping():
    rng = np.random.default_rng(4)
    u = rng.uniform(0, 20, 2000)
    off = -3.0 + 0.1 * u                                          # 3 m inside at a, 1 m inside at b: ~5.7 deg
    pts = np.stack([-10 + u, -6.0 + off * -1.0, rng.uniform(1, 9, 2000)], 1)     # outward normal of edge 0 is -y
    fit = st.fit_wall_plane(BOX[0], BOX[1], pts, 0.5, 10.0)
    assert abs(fit.offset_a + 3.0) < 0.15 and abs(fit.offset_b + 1.0) < 0.15
    assert abs(fit.yaw_deg - math.degrees(math.atan2(2.0, 20.0))) < 1.0


def test_similarity_snap_corrects_the_gps_scale():
    rng = np.random.default_rng(6)
    walls = []
    for (ax, ay), (bx, by) in zip(BOX, np.roll(BOX, -1, axis=0)):
        t = rng.random(800)
        walls.append(np.stack([ax + t * (bx - ax), ay + t * (by - ay), 2 + rng.random(800) * 8.0], 1))
    pts = np.vstack(walls)
    small = pts * np.array([0.92, 0.92, 0.92]) + np.array([1.5, -1.0, 0.0])      # 8 % too small, shifted
    snap = st.snap_to_footprint(small, BOX)
    assert snap.accepted and abs(snap.scale - 1 / 0.92) < 0.02
    back = st.apply_snap_points(snap, small)
    assert np.median(np.linalg.norm(back - pts, axis=1)) < 0.25
    again = st.FootprintSnap.from_meta(json.loads(json.dumps(snap.to_meta())))   # stored and reused
    assert np.allclose(st.apply_snap_points(again, small), back, atol=1e-3)


def test_reusing_a_stored_snap_reproduces_the_model():
    scene = south_scene()
    scene.points = scene.points + np.array([2.0, 1.0, 0.0])
    for c in scene.cameras:
        c.center = c.center + np.array([2.0, 1.0, 0.0])
    scene.snap = True
    first = st.stylize_landmark(scene)
    assert first.summary["snap"]["accepted"]
    scene.snap_override = st.FootprintSnap.from_meta(first.summary["snap"])
    second = st.stylize_landmark(scene)
    assert second.summary["snap"]["shift_m"] == first.summary["snap"]["shift_m"]
    assert np.allclose(second.massing.prisms[0].polygon, first.massing.prisms[0].polygon)


def test_missing_or_broken_footprints_fall_back_to_the_points_or_refuse():
    for bad in (None, np.zeros((0, 2)), np.array([[0, 0], [2, 0], [0, 2.0]]) * 0.5,
                np.array([[-15, -15], [15, 15], [15, -15], [-15, 15.0]])):     # missing, empty, tiny, bow-tie
        scene = south_scene()
        scene.footprint = bad
        m = st.stylize_landmark(scene)
        assert m.summary["inputs"]["outline_source"] == st.PROV_SFM
        assert all(w.outline_source == st.PROV_SFM for w in m.massing.walls if w.prism == 0)
    scene = south_scene()
    scene.footprint, scene.points, scene.colors = None, None, None
    with pytest.raises(st.StylizeError):
        st.stylize_landmark(scene)


def test_courtyards_are_not_roofed_over():
    outer = np.array([[-15.0, -15.0], [15.0, -15.0], [15.0, 15.0], [-15.0, 15.0]])
    hole = np.array([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]])
    scene = st.SceneInput("Courtyard", outer, None, None, [], st.HeightFacts(osm_levels=3), holes=(hole,))
    m = st.stylize_landmark(scene)
    pr = m.massing.prisms[0]
    assert len(pr.holes) == 1
    assert not pr.contains(np.array([[0.0, 0.0]]))[0] and pr.contains(np.array([[10.0, 10.0]]))[0]
    cap = next(p for p in m.parts if p.extras["kind"] == "roof")
    assert abs(cap.extras["area_m2"] - (900 - 100)) < 15                      # (bevelled corners)
    inner = [w for w in m.massing.walls if w.ring == 1]
    assert len(inner) == 4
    for w in inner:                                                           # courtyard walls face the courtyard
        mid = (w.a + w.b) / 2
        assert np.linalg.norm(mid + 0.5 * w.normal) < np.linalg.norm(mid)
    assert m.summary["qa"]["passed"]


def test_height_facts_and_footprint_rings_from_the_cli(tmp_path):
    import stylize_model as sm

    from crowd3d.geo import LocalENU
    assert sm._num("12 m") == 12.0 and sm._num("12,5") == 12.5 and sm._num("14;20") == 14.0
    assert sm._num(None) is None and sm._num("tall") is None and sm._num(-3) is None
    lon0, lat0 = 11.0, 43.0
    d = 0.0002
    outer = [[lon0 - d, lat0 - d], [lon0 + d, lat0 - d], [lon0 + d, lat0 + d], [lon0 - d, lat0 + d], [lon0 - d, lat0 - d]]
    inner = [[lon0 - d / 3, lat0 - d / 3], [lon0 + d / 3, lat0 - d / 3], [lon0 + d / 3, lat0 + d / 3],
             [lon0 - d / 3, lat0 + d / 3], [lon0 - d / 3, lat0 - d / 3]]
    tiny = [[lon0 + 0.01, lat0], [lon0 + 0.01001, lat0], [lon0 + 0.01001, lat0 + 0.00001], [lon0 + 0.01, lat0]]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature",
          "properties": {"height": "30 m", "roof:height": "8", "building:levels": "x"},
          "geometry": {"type": "MultiPolygon", "coordinates": [[tiny], [outer, inner]]}}]}
    f = tmp_path / "fp.geojson"
    f.write_text(json.dumps(gj))
    ring, holes = sm.load_footprint_rings(f, LocalENU(lat0, lon0))
    assert len(ring) == 4 and len(holes) == 1                     # the largest member, with its courtyard
    assert 1400 < abs(st.signed_area(ring)) < 1500                 # 32.6 m x 44.4 m at 43 deg N
    facts = sm.facts_from_entry({"wikidata_height_m": "56"}, f)
    assert (facts.osm_height_m, facts.osm_roof_height_m, facts.osm_levels, facts.wikidata_height_m) == (30.0, 8.0, None, 56.0)
    assert sm.facts_from_entry({}, None) == st.HeightFacts()
    summary = tmp_path / "old.json"
    summary.write_text(json.dumps({"snap": {"rotation_deg": 1.5, "shift_m": [2.0, -1.0], "scale": 1.05, "accepted": True}}))
    snap = sm.snap_from(summary)
    assert snap.scale == 1.05 and snap.accepted and np.allclose(snap.shift, [2.0, -1.0])
    with pytest.raises(SystemExit):
        sm.check_outside_repo(Path(sm.__file__).resolve().parent)


# ------------------------------------------------------------------ D12: small enclosed holes take the wall colour


def test_enclosed_holes_are_closed_small_and_never_reach_the_edge():
    ppm = 10.0                                                    # 1 texel = 0.01 m2; the cap 0.5 m2 = 50 texels
    obs = np.ones((100, 100), bool)
    obs[20:23, 20:23] = False                                     # 0.09 m2, enclosed: filled
    obs[50:80, 50:80] = False                                     # 9 m2, enclosed: over the cap, stays clay
    obs[0:3, 40:43] = False                                       # reaches the top edge: not a hole
    obs[0:10, 85:100] = False                                     # an unobserved corner region ...
    obs[10:13, 82:85] = False                                     # ... and a small gap touching it diagonally: open too
    fill, holes = st.enclosed_holes(obs, ppm, 0.5)
    assert fill[20:23, 20:23].all() and fill.sum() == 9
    assert sorted((h["area_m2"], h["filled"]) for h in holes) == [(0.09, True), (9.0, False)]
    assert not (fill & obs).any()
    assert not st.enclosed_holes(obs, ppm, 0.0)[0].any()          # cap 0: nothing is filled
    # only within the columns the GLB shows: past them the corner bevel would cut the hole open
    assert not st.enclosed_holes(obs, ppm, 0.5, cols=(22, 90))[0].any()
    assert st.enclosed_holes(obs, ppm, 0.5, cols=(19, 90))[0].sum() == 9


def holey_ortho(ppm: float = 20.0) -> tuple[st.Orthophoto, dict[str, tuple[slice, slice]]]:
    """The synthetic facade with unobserved patches (texture rows from the top, columns from the west end):
    'small' 0.25 m2 of plain wall, 'window' 0.25 m2 over a window, 'big' 2.1 m2 of plain wall below the
    windows, 'edge' 0.25 m2 against the west edge, 'corner' 0.25 m2 0.6 m from the east edge."""
    ortho = synthetic_ortho(ppm=ppm)

    def box(x0, x1, z0, z1):
        return slice(int((WALL_H - z1) * ppm), int((WALL_H - z0) * ppm)), slice(int((x0 + 10) * ppm), int((x1 + 10) * ppm))
    spots = {"small": box(-6.25, -5.75, 6.75, 7.25), "window": box(-1.75, -1.25, 6.75, 7.25),
             "big": box(-0.75, 0.75, 0.5, 1.9), "edge": box(-10.0, -9.5, 8.0, 8.5), "corner": box(8.9, 9.4, 8.0, 8.5)}
    for sl in spots.values():
        ortho.mask[sl] = False
    return ortho, spots


def test_small_enclosed_holes_take_the_wall_colour_and_stay_out_of_the_mask():
    ortho, spots = holey_ortho()
    fac = st.stylize_facade(ortho)
    base = np.array(fac.base_rgb, np.uint8)
    centre = {k: (slice(sl[0].start + 2, sl[0].stop - 2), slice(sl[1].start + 2, sl[1].stop - 2)) for k, sl in spots.items()}
    for k in ("small", "window"):                                 # filled: the plain wall colour, even over a window
        assert fac.fill[centre[k]].all(), k
        assert (fac.texture[centre[k]] == base).all(), k
    assert (fac.texture[fac.fill] == base).all()                  # no decal, accent, line or frame grows into a hole
    for k in ("big", "edge"):                                     # over the cap / open to the edge: clay
        assert not fac.fill[spots[k]].any(), k
        assert st.is_clay_pattern(fac.texture[centre[k]]).all(), k
    assert not (fac.fill & fac.mask).any()                        # observed-fill is not observed ...
    assert fac.report["mask_fraction"] == round(float(fac.mask.mean()), 3) < 1.0   # ... nor counted as such
    f = fac.report["fill"]
    assert f["holes"] == 3 and 0.6 < f["area_m2"] < 0.95 and f["largest_m2"] <= 0.5
    assert f["holes_over_cap"] == 1 and 1.8 < f["largest_over_cap_m2"] < 2.5
    codes = st.texel_provenance(fac)
    assert (codes[fac.fill] == 2).all() and (codes[fac.mask] == 1).all()
    assert st.TEXEL_PROVENANCES[2] == st.PROV_FILL
    assert st.check_honesty([], {0: fac})["passed"]
    # enclosed in the texture, but reaching into the last 0.8 m that the east corner's bevel hides: not filled
    assert fac.fill[centre["corner"]].all()
    trimmed = st.stylize_facade(holey_ortho()[0], extent=(0.0, 0.96))
    assert trimmed.face_cols == (0, 384) and not trimmed.fill[spots["corner"]].any()
    assert trimmed.fill[centre["small"]].all()
    forged = st.stylize_facade(holey_ortho()[0], extent=(0.0, 0.96))
    forged.fill[spots["corner"]] = True
    forged.fill &= ~forged.mask
    forged.texture[forged.fill] = base
    assert st.check_honesty([], {0: forged})["fill_regions_not_enclosed"] == 1
    # filling off: the same observation mask, the holes are clay
    off = st.stylize_facade(holey_ortho()[0], st.StyleParams(fill_max_hole_m2=0.0))
    assert not off.fill.any() and np.array_equal(off.mask, fac.mask)
    assert st.is_clay_pattern(off.texture[centre["small"]]).all()


def test_the_honesty_qa_refuses_fill_that_is_open_too_big_or_not_the_wall_colour():
    ortho, spots = holey_ortho()
    fac = st.stylize_facade(ortho)
    base = np.array(fac.base_rgb, np.uint8)
    for name, forge, key in (("big", spots["big"], "fill_regions_over_cap"),
                             ("edge", spots["edge"], "fill_regions_not_enclosed")):
        f = st.stylize_facade(holey_ortho()[0])
        f.fill[forge] = True
        f.fill &= ~f.mask
        f.texture[f.fill] = base
        report = st.check_honesty([], {0: f})
        assert not report["passed"] and report[key] >= 1, (name, report)
        assert st._legal_fill(f.mask, f.px_per_m, 0.5)[forge].sum() == 0     # the audit's own hole finder agrees
    f = st.stylize_facade(holey_ortho()[0])
    y, x = np.argwhere(f.fill)[0]
    f.texture[y, x] = st.INK_RGB                                  # a "detail" inside a filled hole
    report = st.check_honesty([], {0: f})
    assert not report["passed"] and report["fill_texels_not_base"] == 1


def _punch_clutter(monkeypatch, blobs: list[tuple[float, float, float]]) -> None:
    """Make clutter_mask() report round blobs of vegetation (x, z, radius in metres on the south wall):
    the observation mask gets holes there, as behind a tree."""
    real = st.clutter_mask

    def fake(lab, mask, ppm):
        out = real(lab, mask, ppm)
        h, w = mask.shape
        rows, cols = np.mgrid[0:h, 0:w]
        for x, z, r in blobs:
            out |= np.hypot((cols + 0.5) / ppm - (x + 10.0), (rows + 0.5) / ppm - (WALL_H - z)) <= r
        return out & mask
    monkeypatch.setattr(st, "clutter_mask", fake)


def test_filled_holes_are_reported_in_the_summary_and_the_glb_and_never_count_as_observed(monkeypatch):
    _punch_clutter(monkeypatch, [(-6.0, 7.0, 0.3), (0.0, 4.0, 0.85)])        # 0.28 m2: filled; 2.3 m2: clay
    m = st.stylize_landmark(south_scene())
    s = m.summary
    assert s["qa"]["passed"], s["qa"]
    f = s["observed_fill"]
    assert f["walls"] == 1 and f["holes"] == 1 and 0.15 < f["area_m2"] <= 0.5 and f["holes_over_cap"] == 1
    wall = s["walls"][SOUTH]
    assert wall["fill"]["holes"] == 1 and wall["fill"] == wall["facade"]["fill"]
    assert s["qa"]["fill_regions"] == 1 and s["qa"]["glb"]["fill_regions"] == 1
    assert s["qa"]["audit"]["fill"] == {"observed_beyond_geometric_texels": 0, "illegal_fill_texels": 0}
    # coverage figures: exactly what they are with filling switched off
    off = st.stylize_landmark(south_scene(), st.StyleParams(fill_max_hole_m2=0.0))
    assert off.summary["observed_fill"]["holes"] == 0
    assert wall["mask_fraction"] == off.summary["walls"][SOUTH]["mask_fraction"]
    assert s["observed_pixel_area_fraction"] == off.summary["observed_pixel_area_fraction"]
    # in the GLB: the node says so, and the texel-provenance atlas holds the filled texels under its UVs
    gltf, binary = st.read_glb(m.glb)
    node = next(n for n in gltf["nodes"] if n["extras"].get("wall") == SOUTH and n["extras"]["observed"])
    assert node["extras"]["fill"]["holes"] == 1 and node["extras"]["fill"]["area_m2"] == pytest.approx(f["area_m2"], abs=0.01)
    assert node["extras"]["base_color"] == s["walls"][SOUTH]["facade"]["base"]
    assert node["extras"]["observed_fraction"] == off.parts[[p.name for p in off.parts].index(node["name"])].extras["observed_fraction"]
    tp = gltf["scenes"][0]["extras"]["texel_provenance"]
    assert tp["values"] == list(st.TEXEL_PROVENANCES)
    codes = _image(gltf, binary, tp["images"][0])
    part = next(p for p in m.parts if p.name == node["name"])
    assert (codes == 2).sum() >= (part.texel_provenance == 2).sum() > 0
    assert gltf["scenes"][0]["extras"]["observed_fill"]["holes"] == 1


def _image(gltf: dict, binary: bytes, index: int) -> np.ndarray:
    bv = gltf["bufferViews"][gltf["images"][index]["bufferView"]]
    return np.asarray(Image.open(io.BytesIO(binary[bv["byteOffset"]:bv["byteOffset"] + bv["byteLength"]])))


def test_the_glb_audit_rejects_forged_texels(box_model):
    import copy
    parts = copy.deepcopy(box_model.parts)
    part = next(p for p in parts if p.texture is not None)
    assert st.audit_glb(st.write_glb(parts))["passed"]
    h, w = part.texture.shape[:2]
    big = (slice(h // 2 - 20, h // 2 + 20), slice(w // 2 - 20, w // 2 + 20))      # 2.5 m x 2.5 m
    part.texture[big] = part.rgb
    part.texel_provenance[big] = 2
    part.extras["fill"] = {"holes": 1, "area_m2": 6.25, "px_per_m": 16.0}
    report = st.audit_glb(st.write_glb(parts))
    assert not report["passed"] and report["fill_regions_over_cap"] == 1 and report["fill_regions_not_enclosed"] == 0
    part.texel_provenance[big] = 0                                # "clay" texels that are not the clay pattern
    part.extras["fill"]["holes"] = 0
    report = st.audit_glb(st.write_glb(parts))
    assert not report["passed"] and report["clay_texels_not_clay"] == 40 * 40


# ------------------------------------------------------------------ D7: what the colours mean travels with the GLB


def test_atlas_pngs_are_lossless_for_every_allowed_colour():
    """PIL's quantize() merged ink (45, 42, 50) with a palette shade in about a quarter of random atlases:
    a viewer theme restyles texels by their exact colour."""
    cols = np.array(sorted(st.allowed_texture_colors()), np.uint8)
    rng = np.random.default_rng(0)
    for k in (2, 9, 40, len(cols)):
        img = cols[rng.choice(len(cols), k, replace=False)][rng.integers(0, k, (37, 53))]
        back = np.asarray(Image.open(io.BytesIO(st._png_bytes(img))).convert("RGB"))
        assert np.array_equal(back, img), k
    codes = rng.integers(0, 3, (20, 30)).astype(np.uint8)
    assert np.array_equal(np.asarray(Image.open(io.BytesIO(st._png_bytes(codes)))), codes)


def test_unused_atlas_space_is_clay_so_every_texel_is_a_table_colour(box_model):
    part = next(p for p in box_model.parts if p.texture is not None)
    small = st.Part("small", part.positions, part.normals, part.indices, part.rgb, part.texture[:40, :50].copy(),
                    part.uvs, dict(part.extras, fill={"holes": 0, "area_m2": 0.0, "px_per_m": 16.0}),
                    part.texel_provenance[:40, :50].copy())
    glb = st.write_glb([part, small])
    gltf, binary = st.read_glb(glb)
    atlas = _atlas_png(gltf, binary)
    assert atlas.shape[0] > 40 and (atlas[-10:, -10:] == st.CLAY_RGB).all()     # beside the small texture
    report = st.audit_glb(glb)
    assert report["passed"] and report["off_table_texels"] == 0 and report["textured_nodes"] == 2


def test_the_glb_carries_its_colour_table_and_names_its_colours(box_model):
    gltf, binary = st.read_glb(box_model.glb)
    style = gltf["scenes"][0]["extras"]["style"]
    assert style["colors"] == st.color_table() and style["shade_factor"] == st.SHADE_FACTOR
    rgbs = [tuple(r["rgb"]) for r in style["colors"]]
    assert len(set(rgbs)) == len(rgbs)                            # a colour identifies its entry
    assert {r["name"] for r in style["colors"] if r["role"] == "palette"} == set(st.PALETTE_NAMES)
    assert all(r["family"] in st.PALETTE_FAMILIES for r in style["colors"] if r["role"] in ("palette", "shade"))
    assert set(st.allowed_texture_colors()) <= set(rgbs)
    atlas = _atlas_png(gltf, binary)
    assert {tuple(int(v) for v in c) for c in np.unique(atlas.reshape(-1, 3), axis=0)} <= set(rgbs)
    flat = [m for m in gltf["materials"] if "baseColorFactor" in m["pbrMetallicRoughness"]]
    assert flat and all(m["extras"]["color"] in {r["name"] for r in style["colors"]} for m in flat)
    assert box_model.summary["qa"]["glb"]["passed"] and box_model.summary["qa"]["glb"]["off_table_texels"] == 0
    # the viewer's copy of the table (web/themes/palette.json) is this one
    copy = json.loads((Path(__file__).resolve().parent.parent / "web" / "themes" / "palette.json").read_text(encoding="utf-8"))
    assert copy["shade_factor"] == st.SHADE_FACTOR and copy["colors"] == st.color_table()
