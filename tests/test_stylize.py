"""crowd3d.stylize on synthetic inputs: massing, observation masks, honesty, clay GLB."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from crowd3d import params as params_mod
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


def side_cameras() -> list[st.PosedCamera]:
    """One camera in front of each of the east and west walls (grey photos): wall points there are seeable."""
    cams = []
    for x in (32.0, -32.0):
        cam = look_at([x, 0.0, 1.6], [np.sign(x) * 10.0, 0.0, 5.0], name=f"side{int(x)}")
        cam.image = np.full((cam.height, cam.width, 3), 128, np.uint8)
        cams.append(cam)
    return cams


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
    ring, owner = st.bevel_ring(st.ensure_ccw(BOX), 0.5, 3)
    assert len(ring) == 4 * 4 and len(owner) == len(ring)
    assert st.points_in_polygon(ring * 0.999, BOX).all()
    assert abs(st.signed_area(ring) - 240.0) < 1.0
    ell = np.array([[0, 0], [10, 0], [10, 4], [4, 4], [4, 10], [0, 10]], float)
    tris = st.triangulate_polygon(ell)
    area = sum(abs(st.signed_area(ell[list(t)])) for t in tris)
    assert len(tris) == 4 and abs(area - st.signed_area(ell)) < 1e-9


# ------------------------------------------------------------------ observation


def test_register_scene_is_the_first_stage_of_stylize(box_model):
    """crowd3d.structure_pipeline judges the photos on register_scene's output: it must be what stylize_landmark used."""
    reg = st.register_scene(south_scene())
    assert reg.photo_ok and reg.registration == box_model.summary["registration"]
    assert (reg.snap.to_meta() if reg.snap else None) == box_model.summary["snap"] and reg.outline_source == st.PROV_OSM
    assert len(reg.cameras) == len(south_scene().cameras) and len(reg.points) == box_model.summary["inputs"]["points"]
    unreg = st.register_scene(dataclasses.replace(south_scene(), snap_prior=st.SnapPrior(georegistered=False)))
    assert not unreg.photo_ok and unreg.snap is None and "not georegistered" in unreg.registration["reason"]


def test_only_the_photographed_wall_is_observed(box_model):
    obs = box_model.observations
    assert obs[SOUTH].observed and len(obs[SOUTH].cameras) >= 4
    for k in (1, 2, 3):
        assert not obs[k].observed, obs[k].reason
    assert "camera" in obs[NORTH].reason                       # nobody stands north of it
    assert set(box_model.facades) == {SOUTH}


def test_a_wall_seen_by_cameras_but_without_points_is_not_photographed():
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


# ------------------------------------------------------------------ observation masks


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


def test_the_observation_mask_never_claims_a_texel_the_geometric_mask_did_not():
    ortho = synthetic_ortho(mask_left_only=True)
    fac = st.observation_mask(ortho)
    assert not fac.report["empty"] and fac.px_per_m == ortho.px_per_m
    assert not (fac.mask & ~ortho.mask).any()                  # only ever removes texels
    assert not fac.mask[:, fac.mask.shape[1] // 2:].any() and fac.mask[:, : fac.mask.shape[1] // 2].mean() > 0.9
    assert fac.report["mask_fraction"] == round(float(fac.mask.mean()), 3)


def test_sky_and_vegetation_are_not_wall():
    ortho = synthetic_ortho()
    ortho.rgb[:40] = (140, 195, 250)                          # sky above a too-tall wall
    ortho.rgb[120:, :60] = (70, 140, 50)                      # a tree
    fac = st.observation_mask(ortho)
    assert not fac.mask[:30].any()
    assert not fac.mask[150:, 10:50].any()
    assert fac.mask[60:110, 100:300].mean() > 0.95 and fac.report["clutter_fraction"] > 0.1


def test_sky_seen_through_an_arcade_is_not_wall_but_a_window_is():
    """Pale-blue "accents" low in a cream wall (sky through an arcade, not the top-edge sky the clutter mask removes)
    leave the mask; the dark windows (locally darker, not blue) stay in it."""
    ortho = synthetic_ortho()
    ortho.rgb[150:190, 100:140] = (140, 195, 250)
    fac = st.observation_mask(ortho)
    assert not fac.mask[155:185, 105:135].any()
    win = (slice(int((WALL_H - 3.5) * 20), int((WALL_H - 2.5) * 20)), slice(int((-7.5 + 10 - 0.4) * 20), int((-7.5 + 10 + 0.4) * 20)))
    assert fac.mask[win].all()                                # a window (x -7.5 m, z 2-4 m) is wall evidence


def test_a_mask_with_too_little_left_is_empty():
    ortho = synthetic_ortho()
    ortho.mask[:] = False
    ortho.mask[:6, :20] = True                                # 0.3 m2: under half a square metre
    fac = st.observation_mask(ortho)
    assert fac.report["empty"] and not fac.mask.any() and "mask_fraction" not in fac.report


def test_tidy_mask_bridges_small_holes_for_its_blur_only():
    """Holes stay out of the mask (nobody saw them); small enclosed ones (<= 0.5 m2) are bridged for the edge blur
    only, so that the thin observed strips between them are not eaten away; islands under 1 m2 go."""
    ppm = 10.0                                                    # 1 texel = 0.01 m2
    lattice = np.ones((60, 60), bool)
    for y in range(10, 50, 4):
        for x in range(10, 50, 4):
            lattice[y:y + 3, x:x + 3] = False                     # 0.09 m2 holes, 1-texel strips between them
    m = st.tidy_mask(lattice, ppm)
    assert np.array_equal(m, lattice)                             # nothing added, every strip kept
    unbridged = st.tidy_mask(lattice, ppm, max_hole_m2=0.0)
    assert not (unbridged & ~lattice).any() and (lattice & ~unbridged).sum() > 500     # without bridging: eaten
    obs = np.ones((100, 100), bool)
    obs[20:23, 20:23] = False                                     # 0.09 m2, enclosed: a small hole
    obs[50:80, 50:80] = False                                     # 9 m2, enclosed: over the cap
    obs[0:3, 40:43] = False                                       # reaches the top edge: not a hole
    assert st._small_holes(obs, ppm, 0.5).sum() == 9 and st._small_holes(obs, ppm, 9.0).sum() == 909
    assert not st._small_holes(obs, ppm, 0.0).any()
    island = np.zeros((100, 100), bool)
    island[40:49, 40:49] = True                                   # 0.81 m2: dropped
    assert not st.tidy_mask(island, ppm).any()


# ------------------------------------------------------------------ whole model and GLB


def test_box_model_summary_and_qa(box_model):
    s = box_model.summary
    assert s["qa"]["passed"], s["qa"]
    assert s["observed_walls"] == 1 and s["num_walls"] == 4
    south = s["walls"][SOUTH]
    assert south["observed"] and 0.8 < south["mask_fraction"] <= south["geometric_mask_fraction"]
    assert south["clutter_fraction"] is not None and s["walls"][NORTH]["clutter_fraction"] is None
    assert s["observation_norm"]["white_balance"] in ("camera", "camera+neutral-anchor")
    assert s["glb_bytes"] == len(box_model.glb) and s["glb_bytes"] < 30_000          # no atlases: about 14 kB
    # D29: nothing about colour from the photos is published
    assert not {"color_plan", "color_norm", "observed_fill", "observed_wall_area_fraction", "observed_pixel_area_fraction",
                "appearance_provenance_triangles", "textures", "atlases"} & set(s)
    assert not {"fill", "facade"} & set(south)
    json.dumps(s)                                             # the summary is JSON-able as is


def test_glb_round_trips_with_provenance_on_every_node(box_model):
    gltf, binary = st.read_glb(box_model.glb)
    assert gltf["asset"]["version"] == "2.0"
    assert not {"locked_label", "observed_fill", "texel_provenance"} & set(gltf["scenes"][0]["extras"])
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
    assert [w["wall"] for w in walls if w["observed"]] == [SOUTH]
    # D29 (provenance option B): every node's appearance is default-style (the clay look); `observed` alone marks the
    # photographed wall, and "photo-observed" is no provenance value any more
    assert all(n["extras"]["provenance"]["appearance"] == st.PROV_DEFAULT for n in nodes)
    assert "photo-observed" not in st.PROVENANCES
    assert "photo-observed" not in gltf["scenes"][0]["extras"]["provenance_values"]
    # D29: a clay model -- no images, textures or texture coordinates, and no colour extras on any node
    assert not {"images", "textures", "samplers"} & set(gltf)
    assert not any("TEXCOORD_0" in m["primitives"][0]["attributes"] for m in gltf["meshes"])
    assert {m["extras"]["color"] for m in gltf["materials"]} == {"clay", "clay-roof"}
    assert not any({"locked", "photos", "base_color", "fill", "observed_fraction"} & set(n["extras"]) for n in nodes)
    reg = gltf["scenes"][0]["extras"]["registration"]
    assert reg["accepted"] is True


def test_the_glb_audit_refuses_any_image_texture_or_texture_coordinate(box_model):
    """D29: nothing from the photos' pixels is published. A GLB carrying an image, a texture or a node with texture
    coordinates fails the published-bytes audit."""
    assert st.audit_glb(box_model.glb)["passed"]
    gltf, binary = st.read_glb(box_model.glb)
    with_image = json.loads(json.dumps(gltf))
    with_image["images"] = [{"uri": "data:image/png;base64,"}]
    with_image["textures"] = [{"source": 0}]
    report = st.audit_glb(st._glb_bytes(with_image, binary))
    assert not report["passed"] and (report["images"], report["textures"]) == (1, 1)
    with_uv = json.loads(json.dumps(gltf))
    with_uv["meshes"][0]["primitives"][0]["attributes"]["TEXCOORD_0"] = 0
    report = st.audit_glb(st._glb_bytes(with_uv, binary))
    assert not report["passed"] and report["textured_nodes"] == 1


def test_a_face_claiming_a_photo_appearance_fails_the_checks(box_model):
    """D29 (provenance option B): no face takes its look from photos. A node claiming the old 'photo-observed'
    appearance fails check_honesty (no such provenance value, appearance not default-style) and the published-bytes
    audit; with no points, a node counting as photographed fails the independent audit."""
    parts = copy.deepcopy(box_model.parts)
    assert st.check_honesty(parts)["passed"]
    south = next(p for p in parts if p.extras["kind"] == "wall" and p.extras["observed"])
    south.extras["provenance"]["appearance"] = "photo-observed"
    h = st.check_honesty(parts)
    assert not h["passed"] and h["appearance_not_default"] == 1 and h["faces_without_provenance"] == 1
    gltf, _ = st.read_glb(box_model.glb)
    g = st.audit_glb(st.write_glb(parts, gltf["scenes"][0]["extras"]))
    assert not g["passed"] and [v["node"] for v in g["frame"]] == [south.name]
    south.extras["provenance"]["appearance"] = st.PROV_OSM                      # a known value, but not the clay look
    h = st.check_honesty(parts)
    assert not h["passed"] and h["appearance_not_default"] == 1 and h["faces_without_provenance"] == 0
    south.extras["provenance"]["appearance"] = st.PROV_DEFAULT
    m = box_model
    nopts = st.audit_model(m.massing, parts, {}, {}, m.observations, None, [])
    assert not nopts["passed"] and "photographed" in nopts["note"]
    south.extras["observed"] = False
    assert st.audit_model(m.massing, parts, {}, {}, m.observations, None, [])["passed"]


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


def test_an_occluder_in_front_of_the_wall_does_not_count_as_the_wall():
    """Reviewer's t_occluder: a red shed 6 m in front of the south wall, seen by every camera and in the SfM points.
    Before the occlusion fix 30 % of the facade came out red (the shed taken for the wall) and QA passed. The wall
    behind the shed must not count as photographed: D24 relief is measured and built only over observed texels."""
    shed = st.stylize_landmark(south_scene(shed=SHED))
    o = shed.orthophotos[SOUTH]
    assert o.stats["occluded_sample_fraction"] > 0.2
    # the wall behind the shed (lower west part) is not observed; the east part still is
    h, w = o.mask.shape
    assert o.mask[int(0.75 * h):, : int(0.25 * w)].mean() < 0.2
    assert o.mask[int(0.75 * h):, int(0.75 * w):].mean() > 0.6
    fac = shed.facades[SOUTH]                                     # ... in the final observation mask too
    assert fac.mask[int(0.75 * h):, : int(0.25 * w)].mean() < 0.1 and not (fac.mask & ~o.mask).any()
    audit = shed.summary["qa"]["audit"]
    assert audit["occlusion"]["samples"] > 0 and audit["occlusion"]["occluded_share"] <= 0.05
    assert shed.summary["qa"]["passed"], shed.summary["qa"]


def test_the_audit_catches_an_occluder_the_mask_missed():
    """With the mask's occlusion test off, the wall behind the shed counts as photographed: the audit's own segment
    test (independent of the per-camera depth buffer that built the mask) refuses the model."""
    shed = st.stylize_landmark(south_scene(shed=SHED), st.StyleParams(occluder_margin_m=50.0))   # occlusion test off
    h, w = shed.facades[SOUTH].mask.shape
    assert shed.facades[SOUTH].mask[int(0.75 * h):, : int(0.25 * w)].mean() > 0.6       # the shed taken for the wall
    assert not shed.summary["qa"]["audit"]["passed"]
    assert shed.summary["qa"]["audit"]["occlusion"]["occluded_share"] > 0.05
    assert not shed.summary["qa"]["passed"]


def test_a_partly_seen_flat_roof_is_one_clay_node_even_when_a_photo_faces_it():
    """D29: no roof counts as photographed. A flat roof with points on its south half only, photographed from a
    neighbouring tower, is one clay node (no patches) that reports its point coverage; one claiming photos fails
    check_honesty."""
    scene = south_scene()
    keep = ~((scene.points[:, 2] >= WALL_H - 0.01) & (scene.points[:, 1] > 0.0))   # roof points on the south half only
    rng = np.random.default_rng(7)
    side = np.vstack([np.stack([np.full(300, x), rng.uniform(-6, 6, 300), rng.uniform(1, 9, 300)], 1) for x in (-10.0, 10.0)])
    scene.points = np.vstack([scene.points[keep], side])          # (east / west walls in the points: registration)
    scene.colors = np.vstack([scene.colors[keep], np.full((len(side), 3), 200, np.uint8)])
    scene.cameras += side_cameras()                                 # ... seen by a camera each (else unseeable)
    tower = look_at([0.0, -24.0, 26.0], [0.0, -1.0, 9.0], name="tower")
    tower.image = render_south(tower)
    scene.cameras.append(tower)
    m = st.stylize_landmark(scene)
    roofs = [p for p in m.parts if p.extras["kind"] == "roof"]
    assert len(roofs) == 1 and not roofs[0].extras["observed"] and roofs[0].rgb == st.CLAY_ROOF_RGB
    assert roofs[0].extras["provenance"]["appearance"] == st.PROV_DEFAULT
    assert 0.35 < roofs[0].extras["coverage"][0] < 0.65                      # what the points cover, reported
    assert not any(p.extras["kind"] == "roof-patch" for p in m.parts)
    assert m.summary["roof_faces"] == 1 and "roof_patches" not in m.summary
    assert m.summary["qa"]["passed"], m.summary["qa"]
    assert m.summary["qa"]["observed_not_wall"] == []
    roofs[0].extras["observed"] = True                                      # a roof claiming to be photographed
    h = st.check_honesty(m.parts)
    assert not h["passed"] and h["observed_not_wall"] == [roofs[0].name]


def test_a_partly_seen_wall_is_split_into_its_photographed_strip_and_the_rest():
    m = st.stylize_landmark(south_scene())
    fac = m.facades[SOUTH]
    fac.mask[:, : fac.mask.shape[1] // 2] = False          # pretend nobody saw the west half
    parts = st.build_parts(m.massing, m.facades, m.roof, st.DEFAULT_PARAMS, m.observations)
    south = [p for p in parts if p.extras["kind"] == "wall" and p.extras["wall"] == SOUTH]
    assert len(south) == 2
    clay, seen = sorted(south, key=lambda p: p.extras["observed"])
    assert not clay.extras["observed"] and clay.extras["provenance"]["appearance"] == st.PROV_DEFAULT
    assert "not in any photo's observed part" in clay.extras["reason"]
    assert seen.extras["observed"] and abs(clay.extras["area_m2"] - seen.extras["area_m2"]) < 0.3 * seen.extras["area_m2"]
    assert clay.positions[:, 0].max() <= seen.positions[:, 0].min() + 1e-6      # the strip without photos is the west one
    assert clay.rgb == seen.rgb == st.CLAY_RGB                                   # both clay


def test_heights_carried_over_to_unmeasured_walls_are_inferred(box_model):
    walls = {w.index: w for w in box_model.massing.walls}
    assert walls[SOUTH].height_provenance == st.PROV_SFM
    assert all(walls[k].height_provenance == st.PROV_INFERRED for k in (1, 2, 3))
    gltf, _ = st.read_glb(box_model.glb)
    heights = {n["extras"]["wall"]: n["extras"]["provenance"]["height"] for n in gltf["nodes"] if n["extras"]["kind"] == "wall"}
    assert heights[SOUTH] == st.PROV_SFM and heights[NORTH] == st.PROV_INFERRED
    assert st.PROV_INFERRED in gltf["scenes"][0]["extras"]["provenance_values"]


def test_a_rejected_registration_publishes_no_photo_evidence():
    scene = south_scene()
    scene.points = scene.points + np.array([30.0, 0.0, 0.0])        # a model 30 m off (beyond the 25 m bound)
    for c in scene.cameras:
        c.center = c.center + np.array([30.0, 0.0, 0.0])
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
    # 2 m inside the outline: the face is moved onto the fitted plane and photographed on it
    m = st.stylize_landmark(south_scene(wall_y=-4.0))
    south = m.massing.walls[SOUTH]
    assert south.outline_source == st.PROV_SFM and abs(south.a[1] + 4.0) < 0.3
    assert SOUTH in m.facades and m.facades[SOUTH].mask.mean() > 0.8
    # 2.5 m outside it: something in front of the building as far as the footprint knows; never photographed
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
    # points on the east wall too: with the south wall alone a slide along it fits as well (ambiguous: refused)
    gy, gz = np.meshgrid(np.arange(-5.75, 6, 0.25), np.arange(0.25, WALL_H, 0.25))
    east = np.stack([np.full(gy.size, 10.0), gy.ravel(), gz.ravel()], 1)
    scene.points = np.vstack([scene.points, east])
    scene.colors = np.vstack([scene.colors, np.tile(np.array([[235, 225, 205]], np.uint8), (len(east), 1))])
    scene.cameras += side_cameras()[:1]                             # the east wall's camera
    scene.points = scene.points + np.array([2.0, 1.0, 0.0])
    for c in scene.cameras:
        c.center = c.center + np.array([2.0, 1.0, 0.0])
    scene.snap = True
    # the GPS fit fixes the scale (two faces alone cannot)
    scene.snap_prior = st.SnapPrior(4.0, scale_sigma=0.01)
    first = st.stylize_landmark(scene)
    assert first.summary["snap"]["accepted"], first.summary["snap"]["reason"]
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
    with pytest.raises(SystemExit, match="inside the repository"):
        sm.main(["--scene", "x", "--landmark", "x", "--out", str(Path(sm.__file__).resolve().parent)])


def test_the_cli_debug_image_is_a_plan_of_the_photographed_walls(box_model, tmp_path):
    """scripts/stylize_model.py save_debug (D29: no orthophotos or facade sheets any more): one top-down sketch in
    which the walls that count as photographed (WallObservation.observed: here the south one) are thick green."""
    import cv2
    import stylize_model as sm

    files = sm.save_debug(box_model, tmp_path, "box", BOX)
    assert files == ["box-topdown.png"] and sorted(p.name for p in tmp_path.iterdir()) == files
    img = cv2.imread(str(tmp_path / "box-topdown.png"))
    rows = np.nonzero(np.all(img == (60, 170, 60), axis=2))[0]
    # 30 px per metre from y = 11 m down: the south wall (y = -6 m) is row 510, the north one (y = 6 m) row 150
    assert len(rows) > 500 and rows.min() >= 505 and rows.max() <= 515


# ------------------------------------------------------------------ D7: what the colours mean travels with the GLB


def test_the_glb_carries_its_clay_colour_table_and_names_its_colours(box_model):
    gltf, _ = st.read_glb(box_model.glb)
    style = gltf["scenes"][0]["extras"]["style"]
    assert style == {"version": st.STYLE_VERSION, "colors": st.color_table()} and st.STYLE_VERSION == 2
    rgbs = [tuple(r["rgb"]) for r in style["colors"]]
    assert len(set(rgbs)) == len(rgbs)                            # a colour identifies its entry
    assert {r["name"]: tuple(r["rgb"]) for r in style["colors"]} == {
        "clay": st.CLAY_RGB, "clay-hatch": st.CLAY_HATCH_RGB, "clay-roof": st.CLAY_ROOF_RGB}   # D29: clay only
    flat = [m for m in gltf["materials"] if "baseColorFactor" in m["pbrMetallicRoughness"]]
    assert flat == gltf["materials"] and {m["extras"]["color"] for m in flat} == {"clay", "clay-roof"}
    assert box_model.summary["qa"]["glb"]["passed"] and box_model.summary["qa"]["glb"]["images"] == 0


# ------------------------------------------------------------------ the windowed snap (STYLIZED_MODEL 3.2)


def regular_polygon(n: int, radius: float, phase_deg: float = 0.0) -> np.ndarray:
    a = np.radians(phase_deg + 360.0 * np.arange(n) / n)
    return np.stack([radius * np.cos(a), radius * np.sin(a)], 1)


def walls_on(outline: np.ndarray, edges, step: float = 0.4, z=(3.0, 6.0, 9.0, 12.0), t_range=(0.0, 1.0)) -> np.ndarray:
    """Points on the chosen outline edges (heights z), as SfM would see those walls."""
    pts = []
    for k in edges:
        a, b = outline[k], outline[(k + 1) % len(outline)]
        n = max(2, int(np.linalg.norm(b - a) / step))
        for t in np.linspace(*t_range, n):
            for h in z:
                pts.append([*(a + t * (b - a)), h])
    return np.array(pts)


def turned(pts: np.ndarray, deg: float, shift=(0.0, 0.0), about=(0.0, 0.0)) -> np.ndarray:
    th = math.radians(deg)
    r = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    out = pts.copy()
    out[:, :2] = (pts[:, :2] - about) @ r.T + np.asarray(about) + np.asarray(shift, float)
    return out


def test_outline_symmetry_period_names_rectangles_octagons_circles_and_none():
    """P is where the outline first lands back on itself (within snap_symmetry_tol_m): a 20 m octagon fits
    itself from 36 deg on (its 45 deg alias basin), so the window stays under 13 deg there."""
    assert 120.0 <= st.outline_symmetry_period(BOX) <= 180.0
    assert 34.0 <= st.outline_symmetry_period(regular_polygon(8, 20.0)) <= 45.0
    assert st.outline_symmetry_period(regular_polygon(24, 8.3)) == 0.0          # a tower: every rotation fits
    assert st.outline_symmetry_period(regular_polygon(16, 20.0)) == 0.0
    ell = np.array([[0.0, 0.0], [30.0, 0.0], [30.0, 10.0], [10.0, 10.0], [10.0, 25.0], [0.0, 25.0]])
    assert st.outline_symmetry_period(ell) == 360.0


def test_the_yaw_window_follows_the_registration_sigma_and_the_outline_symmetry():
    assert st.yaw_window(360.0, 4.0) == (pytest.approx(12.0), "window", 4.0)
    # a reported sigma under the pipeline's systematic floor is floored here too (verifier r2: with sigma / 3
    # reported, translation-only accepted 12-16 deg errors on near-circles)
    assert st.DEFAULT_PARAMS.snap_min_yaw_sigma_deg == params_mod.DEFAULTS.yaw_systematic_sigma_deg == 3.0
    assert st.yaw_window(360.0, 2.0) == (pytest.approx(9.0), "window", 3.0)
    assert st.yaw_window(360.0, 0.67) == (pytest.approx(9.0), "window", 3.0)
    assert st.yaw_window(45.0, 7.0)[:2] == (pytest.approx(17.5), "window")      # P/2 - 5: never the next face
    assert st.yaw_window(180.0, None)[:2] == (pytest.approx(20.1), "window")     # unknown sigma: the old +-20 deg
    assert st.yaw_window(0.0, 1.0)[:2] == (0.0, "translation-only")
    assert st.yaw_window(12.0, 4.0)[:2] == (0.0, "translation-only")             # 12/2 - 5 < snap_min_window_deg


def test_an_octagon_facade_snaps_inside_its_window_and_never_onto_the_next_face():
    """Florence Baptistery: the unconstrained ICP turned 6-8 photo facade models 79-93 deg onto the neighbouring
    face of the octagon. Inside +-min(3 sigma, P/2 - 5 deg) the alias cannot be reached; a model turned further
    than the window is refused rather than snapped onto the next face (and one whose 3 sigma P/2 - 5 deg would clip
    is refused outright: tests/test_snap_gate73.py)."""
    octagon = regular_polygon(8, 20.0, 22.5)
    pts = walls_on(octagon, (5, 6, 7))
    off = turned(pts, 9.0, (3.0, -2.0))
    snap = st.snap_to_footprint(off, octagon, prior=st.SnapPrior(yaw_sigma_deg=4.0))
    period = snap.search["symmetry_period_deg"]
    assert 34.0 <= period <= 45.0 and snap.search["mode"] == "window"
    assert snap.accepted and snap.search["window_deg"] == pytest.approx(12.0) and 12.0 <= period / 2 - 5.0
    # the coarse step works on a 0.1 m raster of the outline; the per-wall plane fit does the fine step
    assert snap.rotation_deg == pytest.approx(-9.0, abs=1.0)
    back = st.apply_snap_points(snap, off)
    assert np.median(np.linalg.norm(back[:, :2] - pts[:, :2], axis=1)) < 0.4
    # 30 deg off: the right correction is outside the window and the alias (+15 deg) beyond its edge
    far = st.snap_to_footprint(turned(pts, 30.0), octagon, prior=st.SnapPrior(yaw_sigma_deg=4.0))
    assert not far.accepted and far.rotation_deg == 0.0 and not far.shift.any()
    assert all(abs(a["rotation_deg"]) <= far.search["window_deg"] + 1e-6 for a in far.search["alternatives"])


def test_a_near_circular_outline_is_snapped_by_translation_only():
    """Leaning Tower of Pisa: a 23-vertex near-circle fits itself at any rotation, so the ICP spun facade models
    12-17 deg; the yaw stays as registered and only the shift is searched."""
    tower = regular_polygon(24, 8.3)
    pts = walls_on(tower, range(8), step=0.3)
    off = turned(pts, 2.0, (4.0, 3.0))
    low_floor = dataclasses.replace(st.DEFAULT_PARAMS, snap_min_yaw_sigma_deg=0.5)
    snap = st.snap_to_footprint(off, tower, low_floor, prior=st.SnapPrior(yaw_sigma_deg=1.0))
    assert snap.search["mode"] == "translation-only" and snap.search["symmetry_period_deg"] == 0.0
    assert snap.accepted and snap.rotation_deg == 0.0
    assert np.allclose(snap.shift, [-4.0, -3.0], atol=0.3)
    moved = st.apply_snap_points(snap, off)
    assert np.median(np.abs(np.linalg.norm(moved[:, :2], axis=1) - 8.3 * math.cos(math.pi / 24))) < 0.3
    # with the default floor (3 deg: +-9 deg) the same reported 1 deg is not trusted
    floored = st.snap_to_footprint(off, tower, prior=st.SnapPrior(yaw_sigma_deg=1.0))
    assert not floored.accepted and "yaw unverifiable" in floored.reason and floored.search["yaw_sigma_deg"] == 3.0


def test_translation_only_trusts_the_registered_yaw_only_when_its_sigma_is_small():
    """Nothing checks the registered yaw of a near-circle (the round-2 adversarial cases: 34 of 47 passes on
    near-circles were wrong, up to 21 deg off, 11 of them with the right sigma reported). The snap is refused
    unless snap_yaw_sigmas x the reported sigma is within snap_fixed_yaw_max_deg and moves the outline's
    boundary by at most snap_max_residual_m; an unreported sigma is never trusted."""
    tower = regular_polygon(24, 8.3)
    off = turned(walls_on(tower, range(8), step=0.3), 2.0, (4.0, 3.0))
    for sigma in (None, 4.0):                              # 4 deg: +-12 deg, 1.7 m around an 8.3 m tower
        snap = st.snap_to_footprint(off, tower, prior=st.SnapPrior(yaw_sigma_deg=sigma))
        assert snap.search["mode"] == "translation-only"
        assert not snap.accepted and snap.reason.startswith("yaw unverifiable"), snap.reason
        assert snap.rotation_deg == 0.0 and not snap.shift.any()
    low_floor = dataclasses.replace(st.DEFAULT_PARAMS, snap_min_yaw_sigma_deg=1.0)   # (the default floor: 3 deg)
    ok = st.snap_to_footprint(off, tower, low_floor, prior=st.SnapPrior(yaw_sigma_deg=1.5))     # +-4.5 deg: 0.65 m
    assert ok.accepted and ok.search["fixed_yaw"] == {"bound_deg": 4.5, "moves_outline_m": 0.65}
    # the same +-4.5 deg on a 25 m rotunda moves the photos 2 m around it
    rotunda = regular_polygon(24, 25.0)
    wide = st.snap_to_footprint(turned(walls_on(rotunda, range(8), step=0.4), 2.0, (4.0, 3.0)), rotunda, low_floor,
                                prior=st.SnapPrior(yaw_sigma_deg=1.5))
    assert wide.search["mode"] == "translation-only" and not wide.accepted and "yaw unverifiable" in wide.reason
    assert wide.search["fixed_yaw"]["moves_outline_m"] == pytest.approx(1.96, abs=0.01)


def two_sided(scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """South and east walls of BOX seen from the south-east, with the cameras that saw them."""
    pts = walls_on(BOX, (0, 1))
    cams = np.array([[x, -30.0] for x in np.linspace(-8, 8, 5)] + [[34.0, y] for y in np.linspace(-6, 4, 4)])
    return pts * scale, cams * scale, cams


def test_the_snap_refuses_when_the_snapped_cameras_disagree_with_their_own_gps():
    """Brandenburg Gate E facade: the points sat 5-10 m in front of the real facade; snapping them onto the
    outline put the cameras 36-40 m from it while their own GPS said 46-48 m. Rescaling about the walls made it
    worse (the distortion is not a uniform scale), so the snap is refused: clay, not a misplaced facade."""
    pts, cams, _ = two_sided()
    off, cam_off = turned(pts, 2.0, (1.5, -2.0)), turned(np.c_[cams, np.zeros(len(cams))], 2.0, (1.5, -2.0))[:, :2]
    rng = np.random.default_rng(3)
    gps = cams + rng.normal(0, 1.5, cams.shape)
    ok = st.snap_to_footprint(off, BOX, prior=st.SnapPrior(4.0, cam_off, gps, 0.01))
    assert ok.accepted and ok.gps["agrees"] and abs(ok.gps["ratio"] - 1.0) < 0.05
    # the same points, but the photos were taken 25 % farther away than the model's cameras stand
    far_gps = (gps - BOX.mean(axis=0)) * 1.25 + BOX.mean(axis=0)
    bad = st.snap_to_footprint(off, BOX, prior=st.SnapPrior(4.0, cam_off, far_gps, 0.01))
    assert not bad.accepted and "own GPS" in bad.reason and bad.gps["ratio"] > 1.15
    assert bad.rotation_deg == 0.0 and not bad.shift.any()


def test_a_scale_the_gps_left_uncertain_is_refused_for_a_one_sided_model():
    """A facade fits a straight outline edge at any scale; with the GPS scale uncertain by 20 % (Baptistery SE,
    7 cameras over 12 m) photos could land on the wrong part of a wall."""
    pts, cams, _ = two_sided()
    off = turned(pts, 1.0, (1.0, 1.0))
    assert st.snap_to_footprint(off, BOX, prior=st.SnapPrior(4.0, scale_sigma=0.01)).accepted
    unsure = st.snap_to_footprint(off, BOX, prior=st.SnapPrior(4.0, scale_sigma=0.2))
    assert not unsure.accepted and "scale" in unsure.reason
    # the rule itself, without the scale profile that now usually refuses first
    no_profile = dataclasses.replace(st.DEFAULT_PARAMS, snap_scale_steps=())
    assert st.snap_to_footprint(off, BOX, no_profile, prior=st.SnapPrior(4.0, scale_sigma=0.05)).accepted
    unsure = st.snap_to_footprint(off, BOX, no_profile, prior=st.SnapPrior(4.0, scale_sigma=0.2))
    assert not unsure.accepted and "scale uncertain" in unsure.reason


def comb_outline(tooth: float = 4.0) -> np.ndarray:
    """A 40 m x 10 m block whose south face is a colonnade: 2 m deep bays every `tooth` metres."""
    south = []
    for x0 in np.arange(-20.0, 20.0, tooth):
        south += [[x0, -5.0], [x0 + tooth / 2, -5.0], [x0 + tooth / 2, -7.0], [x0 + tooth, -7.0]]
    return np.array(south + [[20.0, -5.0], [20.0, 5.0], [-20.0, 5.0]])


def test_a_periodic_colonnade_makes_the_snap_ambiguous():
    """A facade whose bays repeat fits as well one bay along: the edge-by-edge check passes either way, so the
    placement cannot be told and the snap is refused (the shift scan finds the other basin)."""
    comb = comb_outline()
    south_part = [k for k in range(40) if -10.0 <= comb[k][0] <= 10.0 and comb[k][1] < -4.5]
    pts = walls_on(comb, south_part, step=0.25)
    snap = st.snap_to_footprint(turned(pts, 0.0, (0.6, -0.4)), comb, prior=st.SnapPrior(3.0))
    assert not snap.accepted and snap.reason.startswith("ambiguous")
    assert any(a["verified"] and a["walls_moved_m"] >= 3.0 for a in snap.search["alternatives"])
    # the same colonnade with its west end wall seen too: one placement only
    west = walls_on(comb, (len(comb) - 1,), step=0.25, t_range=(0.0, 0.4))
    sure = st.snap_to_footprint(turned(np.vstack([pts, west]), 0.0, (0.6, -0.4)), comb, prior=st.SnapPrior(3.0))
    assert sure.accepted, sure.reason


def test_the_shift_bound_and_the_ambiguity_distance_are_measured_at_the_walls():
    """How far a snap moves the walls is measured at the points that end up on the outline, not at the median of
    every wall-height point: a neighbour's points turn with the model, and at their median a 9 deg correction
    about the walls read as metres of displacement (the Baptistery reference's Duomo points)."""
    octagon = regular_polygon(8, 20.0, 22.5)
    walls = walls_on(octagon, (5, 6, 7))
    rng = np.random.default_rng(5)
    n = int(len(walls) * 0.3 / 0.7)                       # 30 % of the points: a neighbour 40 m to the north-west
    neighbour = np.c_[rng.uniform(-36, -26, n), rng.uniform(20, 30, n), rng.uniform(3, 12, n)]
    at = np.median(walls[:, :2], axis=0)
    off = turned(np.vstack([walls, neighbour]), 9.0, (1.0, -1.0), about=at)
    snap = st.snap_to_footprint(off, octagon, prior=st.SnapPrior(yaw_sigma_deg=4.0))
    assert snap.accepted, snap.reason
    assert snap.rotation_deg == pytest.approx(-9.0, abs=1.0)
    assert snap.search["alternatives"][0]["walls_moved_m"] == pytest.approx(math.sqrt(2.0), abs=0.15)


def test_the_icp_is_unbounded_and_a_placement_beyond_the_shift_bound_is_refused():
    """Clamping the shift inside the ICP iterations left placements stuck at the bound, which then counted as
    alternatives (Frauenkirche SW#20 was refused as ambiguous); the ICP now runs free and the 25 m bound applies
    to the final placement only."""
    pts, _, _ = two_sided()
    lookup = st._OutlineLookup(BOX, 60.0)
    xy = turned(pts, 3.0, (0.0, -30.0))[:, :2]
    rot, t, med = st._snap_icp(xy, lookup, 0.0, np.array([0.0, 28.0]), 10.0)
    assert med < 0.1 and rot == pytest.approx(-3.0, abs=0.3)
    assert np.linalg.norm(t) > 25.0                       # the ICP itself does not stop at the bound
    tight = st.SnapPrior(4.0, scale_sigma=0.01)             # the GPS fit fixes the scale (two faces cannot)
    near = st.snap_to_footprint(turned(pts, 3.0, (0.0, -20.0)), BOX, prior=tight)
    assert near.accepted and near.search["alternatives"][0]["walls_moved_m"] == pytest.approx(20.0, abs=1.0)
    far = st.snap_to_footprint(turned(pts, 3.0, (0.0, -30.0)), BOX, prior=tight)
    assert not far.accepted and "beyond the shift bound" in far.reason
    assert far.rotation_deg == 0.0 and not far.shift.any()
    # right at the bound: 24 m in, 26.5 m out (the decision is the bound itself, not a margin below it)
    at = np.median(pts[:, :2], axis=0)
    inside = st.snap_to_footprint(turned(pts, 3.0, (0.0, -24.0), about=at), BOX, prior=tight)
    assert inside.accepted, inside.reason
    outside = st.snap_to_footprint(turned(pts, 3.0, (0.0, -26.5), about=at), BOX, prior=tight)
    assert not outside.accepted and "beyond the shift bound" in outside.reason
    edge = st.snap_to_footprint(turned(pts, 3.0, (0.0, -24.8), about=at), BOX, prior=tight)
    assert edge.accepted and 24.5 < edge.search["alternatives"][0]["walls_moved_m"] <= 25.0


def walls_with_neighbour(walls: np.ndarray, centre, share: float = 0.3, half: float = 3.0, seed: int = 3) -> np.ndarray:
    """The walls plus a block of points (share of all) around `centre` that moves with the model (a neighbour)."""
    rng = np.random.default_rng(seed)
    n = int(len(walls) * share / (1.0 - share))
    block = np.c_[rng.uniform(centre[0] - half, centre[0] + half, n), rng.uniform(centre[1] - half, centre[1] + half, n),
                  rng.uniform(3.0, 12.0, n)]
    return np.vstack([walls, block])


def test_the_shift_bound_is_decided_at_the_walls_not_at_the_median_of_every_point():
    """A neighbour holding a third of the wall-height points pulls their median 7 m off the walls; turned with the
    model, that median moves more than the bound while the walls move less: accepted (round-2 mutation check:
    the bound decided at that median survived every test)."""
    params = dataclasses.replace(st.DEFAULT_PARAMS, snap_max_shift_m=12.0, snap_max_scale=1.0)
    walls = walls_on(BOX, (0, 1), step=0.25)
    at = np.median(walls[:, :2], axis=0)
    shift = 11.3 * np.array([math.cos(math.radians(255)), math.sin(math.radians(255))])
    off = turned(walls_with_neighbour(walls, (-12.0, 12.0), share=0.35), 10.0, shift, about=at)
    band = off[(off[:, 2] >= 2.0) & (off[:, 2] <= 20.0)]
    median = np.median(band[np.linalg.norm(band[:, :2], axis=1) <= 11.67 + 12.0, :2], axis=0)
    undo = st._rot2(-10.0)
    assert np.linalg.norm(undo @ median + (at - undo @ (at + shift)) - median) > 12.5     # precondition
    snap = st.snap_to_footprint(off, BOX, params, prior=st.SnapPrior(yaw_sigma_deg=4.0, scale_sigma=0.01))
    assert snap.accepted, snap.reason
    assert snap.rotation_deg == pytest.approx(-10.0, abs=0.3)
    assert snap.search["alternatives"][0]["walls_moved_m"] == pytest.approx(11.3, abs=0.3)


def test_a_neighbour_next_to_the_outline_does_not_turn_the_snap():
    """The ICP pairs only points within max(snap_icp_inlier_m, 3 x median) of the outline: with the closest 70 %
    alone, a block of points 6 m off a corner (35 % of all) turned this model's fit 3-7 deg off (photos 0.5-0.9 m
    off at the median wall point)."""
    params = dataclasses.replace(st.DEFAULT_PARAMS, snap_max_shift_m=12.0, snap_max_scale=1.0)
    walls = walls_on(BOX, (0, 1), step=0.25)
    at = np.median(walls[:, :2], axis=0)
    for centre, angle, turn in (((-12.0, 12.0), 255, 10.0), ((-8.0, 14.0), 255, 10.0), ((-16.0, 4.0), 60, -10.0)):
        shift = 11.3 * np.array([math.cos(math.radians(angle)), math.sin(math.radians(angle))])
        model = walls_with_neighbour(walls, centre, share=0.35)
        snap = st.snap_to_footprint(turned(model, turn, shift, about=at), BOX, params,
                                    prior=st.SnapPrior(4.0, scale_sigma=0.01))
        assert snap.accepted and snap.rotation_deg == pytest.approx(-turn, abs=0.3), (centre, snap.rotation_deg)
        back = st.apply_snap_points(snap, turned(model, turn, shift, about=at))[:len(walls)]
        assert np.median(np.linalg.norm(back[:, :2] - walls[:, :2], axis=1)) < 0.2


def test_a_flat_yaw_valley_is_ambiguous_but_a_turn_that_hardly_moves_the_photos_is_not():
    """Frauenkirche SW#24 / SW#25 (20-30 photos): the walls fit within the ambiguity margin turned 3.5-5 deg about
    themselves, the snap sat 3-4 deg off the reference and coloured 12 walls. Such a turn is ambiguous when it
    moves the photos >= snap_ambiguity_m over the walls (or is snap_ambiguity_turn_deg itself); three faces of an
    octagon turned 3.5 deg about its centre move them 1.2 m, and a neighbour 40 m away that turns with them must
    not count (round-2 mutation check: the distance taken over every point)."""
    octagon = regular_polygon(8, 20.0, 22.5)
    walls = walls_on(octagon, (5, 6, 7))
    at = np.median(walls[:, :2], axis=0)
    rng = np.random.default_rng(3)
    walls[:, :2] += rng.normal(0.0, 0.3, (len(walls), 2))         # SfM noise: a shallow valley
    model = turned(walls_with_neighbour(walls, (-30.0, 25.0)), 4.0, (2.0, -3.0), about=at)
    # turns of 2 / 3.5 / 6 deg: at 6 deg the fit is out of the margin, at 3.5 deg it is not but moves the photos 1.2 m
    params = dataclasses.replace(st.DEFAULT_PARAMS, snap_ambiguity_turn_deg=6.0)
    snap = st.snap_to_footprint(model, octagon, params, prior=st.SnapPrior(yaw_sigma_deg=4.0, scale_sigma=0.01))
    assert snap.accepted, snap.reason
    prof = snap.search["turn_profile"]
    margin = prof["best_fit_m"] * (1 + params.snap_ambiguity_margin) + params.snap_ambiguity_slack_m
    turns = {t["turn_deg"]: t for t in prof["turns"]}
    assert turns[3.5]["fit_m"] <= margin and turns[3.5]["gap_m"] < params.snap_ambiguity_m   # as good, 1.2 m apart
    band = model[(model[:, 2] >= 2.0) & (model[:, 2] <= 20.0), :2]
    turned_all = (band - at) @ st._rot2(3.5).T + at
    assert np.percentile(np.linalg.norm(turned_all - band, axis=1), 90) > 2.0             # ... over every point
    assert turns[6.0]["fit_m"] > margin
    # the default turns up to 5 deg: the walls still fit there, and a right placement is within 5 deg: ambiguous
    wide = st.snap_to_footprint(model, octagon, prior=st.SnapPrior(yaw_sigma_deg=4.0, scale_sigma=0.01))
    assert not wide.accepted and wide.reason.startswith("ambiguous: the walls turned")


def test_a_yaw_the_walls_cannot_fix_within_5_deg_is_ambiguous_however_little_the_photos_move():
    """Two walls of a 6 m x 4 m house fit about as well turned 5 deg (the photos move only 0.3 m): a right
    placement is within 5 deg, so a yaw the walls leave open that far is refused whatever the walls' size."""
    small = np.array([[-3.0, -2.0], [3.0, -2.0], [3.0, 2.0], [-3.0, 2.0]])
    walls = walls_on(small, (0, 1), step=0.2)
    walls[:, :2] += np.random.default_rng(1).normal(0.0, 0.15, (len(walls), 2))
    snap = st.snap_to_footprint(turned(walls, 2.0, (1.0, -1.0), about=np.median(walls[:, :2], axis=0)), small,
                                prior=st.SnapPrior(4.0))
    assert not snap.accepted and snap.reason.startswith("ambiguous: the walls turned")
    assert all(t["gap_m"] < 0.5 for t in snap.search["turn_profile"]["turns"])


def test_every_candidate_gets_the_scale_step_before_they_compete():
    """A box 12 % too small fits no rigid placement well: its rigid fits differ by 2.5 deg and 3 m without being
    different placements. Compared before the rescale they read as ambiguous; each rescaled, they are one."""
    rng = np.random.default_rng(6)
    walls = []
    for (ax, ay), (bx, by) in zip(BOX, np.roll(BOX, -1, axis=0)):
        t = rng.random(800)
        walls.append(np.stack([ax + t * (bx - ax), ay + t * (by - ay), 2 + rng.random(800) * 8.0], 1))
    pts = np.vstack(walls)
    small = pts * 0.88 + np.array([1.5, -1.0, 0.0])
    snap = st.snap_to_footprint(small, BOX, prior=st.SnapPrior(4.0))
    assert snap.accepted, snap.reason
    assert snap.scale == pytest.approx(1 / 0.88, abs=0.01) and abs(snap.rotation_deg) < 0.2
    assert np.median(np.linalg.norm(st.apply_snap_points(snap, small)[:, :2] - pts[:, :2], axis=1)) < 0.2


def test_walls_that_follow_the_outline_only_loosely_cannot_fix_the_yaw():
    """Frauenkirche SW#9 (8 photos): rescaled and turned onto the outline its walls still sat 0.50 m from it on
    average, 5 deg off the reference, and nothing else told; the right SW#11/14/15 fitted 0.30-0.34 m. Above
    snap_max_fit_m the snap is refused."""
    walls = walls_on(BOX, (0, 1, 2), step=0.25)
    for noise, ok in ((0.35, True), (0.7, False)):
        noisy = walls.copy()
        noisy[:, :2] += np.random.default_rng(2).normal(0.0, noise, (len(walls), 2))
        snap = st.snap_to_footprint(turned(noisy, 2.0, (1.0, -1.0)), BOX, prior=st.SnapPrior(4.0))
        assert snap.accepted == ok, snap.reason
        assert ok or "too loose to fix the yaw" in snap.reason


def test_the_coarse_grid_finds_every_basin_whatever_the_registered_position():
    """The starts come from a fixed rotation x shift grid, not from the registered position: the colonnade's
    neighbouring bays are all among them, and the same model registered 9 m elsewhere gives the same snap."""
    comb = comb_outline()
    south_part = [k for k in range(40) if -10.0 <= comb[k][0] <= 10.0 and comb[k][1] < -4.5]
    pts = walls_on(comb, south_part, step=0.25)
    lookup = st._OutlineLookup(st.ensure_ccw(comb), 60.0)
    at = np.median(pts[:, :2], axis=0)
    moves = [st._rot2(r) @ at + t - at for r, t, _ in st._placement_grid(pts[:, :2], lookup, at, 9.0)]
    assert np.linalg.norm(moves[0]) < 0.5                # the best start: where the points are
    bays = {round(float(m[0]) / 4.0) for m in moves if abs(m[1]) < 0.5 and abs(m[0] - 4.0 * round(m[0] / 4.0)) <= 1.0}
    assert len(bays - {0}) >= 2                           # and other bays of the colonnade (4 m apart) too
    west = walls_on(comb, (len(comb) - 1,), step=0.25, t_range=(0.0, 0.4))
    model = np.vstack([pts, west])
    snaps = [st.snap_to_footprint(turned(model, 2.0, off), comb, prior=st.SnapPrior(3.0))
             for off in ((0.6, -0.4), (9.0, 3.0), (-8.0, -5.0))]
    for snap, off in zip(snaps, ((0.6, -0.4), (9.0, 3.0), (-8.0, -5.0)), strict=True):
        assert snap.accepted, snap.reason
        back = st.apply_snap_points(snap, turned(model, 2.0, off))
        assert np.median(np.linalg.norm(back[:, :2] - model[:, :2], axis=1)) < 0.2


def test_snap_prior_comes_from_the_model_json(tmp_path):
    import stylize_model as sm

    meta = {"yaw_sigma_deg": 3.6, "scale_sigma": 0.05, "georegistered": True, "cameras": {
        "a": {"enu": [1.0, -30.0, 1.6], "gps_en": [1.5, -31.0], "flags": []},
        "b": {"enu": [5.0, -30.0, 1.6], "gps_en": [5.2, -29.0], "flags": []},
        "c": {"enu": [90.0, 0.0, 1.6], "gps_en": [9.0, -30.0], "flags": ["gps_and_compass"]}}}
    path = tmp_path / "m.json"
    path.write_text(json.dumps(meta))
    prior = sm.snap_prior(path)
    assert prior.yaw_sigma_deg == 3.6 and prior.scale_sigma == 0.05 and prior.georegistered is True
    assert prior.camera_xy.tolist() == [[1.0, -30.0], [5.0, -30.0]] and prior.gps_xy.tolist() == [[1.5, -31.0],
                                                                                                    [5.2, -29.0]]
    path.write_text(json.dumps({"cameras": {"a": {"enu": [1.0, -30.0, 1.6], "flags": []}}}))   # an older model
    old = sm.snap_prior(path)
    assert old.yaw_sigma_deg is None and old.camera_xy is None and old.scale_sigma is None
    assert old.georegistered is False              # a model JSON that does not say it is georegistered: prior only
    meta["georegistered"] = False
    path.write_text(json.dumps(meta))
    assert sm.snap_prior(path).georegistered is False


def test_a_model_that_is_not_georegistered_contributes_no_photo_evidence(box_model):
    """Frauenkirche SW#17 (compass yaw and a guessed scale, no GPS fit) passed the snap and the edge-by-edge
    check and had 4 walls coloured although the reference says it needed a 19 % rescale (13 m camera error).
    A prior-only model (georeg_level != full) contributes no facade (STYLIZED_MODEL 3.2 item 4): no snap, no
    gate, no photographed wall, whatever its points look like."""
    scene = south_scene()
    scene.snap_prior = st.SnapPrior(yaw_sigma_deg=2.0, georegistered=False)
    model = st.stylize_landmark(scene)
    reg = model.summary["registration"]
    assert not reg["accepted"] and "not georegistered" in reg["reason"] and model.summary["snap"] is None
    assert model.summary["observed_walls"] == 0 and model.summary["qa"]["passed"]
    assert all(w["reason"].startswith("registration not verified") for w in model.summary["walls"])
    # the very same scene georegistered (or not from the pipeline at all, the fixture) has its photographed wall
    assert box_model.summary["observed_walls"] > 0
    scene.snap_prior = st.SnapPrior(yaw_sigma_deg=2.0, georegistered=True)
    assert st.stylize_landmark(scene).summary["observed_walls"] == box_model.summary["observed_walls"]


# ------------------------------------------------------------------ round 3: tilt, sight lines, scale, cameras


def test_load_job_leaves_flagged_cameras_out(tmp_path):
    """Frauenkirche SW#14: a flagged camera (154 m from its reference position) fed all three coloured walls.
    Flagged cameras are withheld from users; they are no photo evidence either (load_job leaves them out)."""
    pycolmap = pytest.importorskip("pycolmap")
    import stylize_model as sm

    job = tmp_path / "job"
    (job / "sparse" / "0").mkdir(parents=True)
    (job / "images").mkdir()
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera(model="SIMPLE_PINHOLE", width=64, height=48,
                                                    params=[50.0, 32.0, 24.0], camera_id=1))
    centres, cams = {}, {}
    for i in range(5):
        pid = f"p{i}"
        im = pycolmap.Image(name=f"{i:05d}_{pid}.jpg", camera_id=1, image_id=i + 1)
        rec.add_image_with_trivial_frame(im, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([-4.0 * i, 0, 0])))
        Image.new("RGB", (64, 48), (120, 120, 120)).save(job / "images" / im.name)
        centres[pid] = [4.0 * i, 0.0, 0.0]
        cams[pid] = {"enu": [4.0 * i, 0.0, 0.0], "gps_en": [4.0 * i, 0.5], "flags": ["far_from_gps"] if i == 2 else []}
    rec.add_point3D(np.array([0.0, 0.0, 5.0]), pycolmap.Track(), np.array([10, 20, 30], np.uint8))
    rec.write(str(job / "sparse" / "0"))
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"georegistered": True, "yaw_sigma_deg": 4.0, "scale_sigma": 0.02, "cameras": cams}))
    _, _, posed, source, _ = sm.load_job(job, meta)
    assert sorted(c.photo_id for c in posed) == ["p0", "p1", "p3", "p4"]
    assert source["flagged_cameras_left_out"] == ["p2"]
    prior = sm.snap_prior(meta)
    assert len(prior.camera_xyz) == 4 and [8.0, 0.0, 0.0] not in prior.camera_xyz.tolist()


def test_load_job_tracks_leave_the_flagged_cameras_observations_out(tmp_path):
    """Structure-first merge: the facade relief's point tracks (with_tracks) index the cameras load_job returns, so a
    flagged camera, which colours nothing, measures nothing either -- its observations are left out of the tracks."""
    pycolmap = pytest.importorskip("pycolmap")
    import stylize_model as sm

    job = tmp_path / "job"
    (job / "sparse" / "0").mkdir(parents=True)
    (job / "images").mkdir()
    rec = pycolmap.Reconstruction()
    rec.add_camera_with_trivial_rig(pycolmap.Camera(model="SIMPLE_PINHOLE", width=64, height=48,
                                                    params=[50.0, 32.0, 24.0], camera_id=1))
    cams, track = {}, pycolmap.Track()
    for i in range(5):
        pid = f"p{i}"
        im = pycolmap.Image(name=f"{i:05d}_{pid}.jpg", camera_id=1, image_id=i + 1)
        im.points2D = pycolmap.Point2DList([pycolmap.Point2D(np.array([10.0 + i, 20.0]))])
        rec.add_image_with_trivial_frame(im, pycolmap.Rigid3d(pycolmap.Rotation3d(np.eye(3)), np.array([-4.0 * i, 0, 0])))
        Image.new("RGB", (64, 48), (120, 120, 120)).save(job / "images" / im.name)
        cams[pid] = {"enu": [4.0 * i, 0.0, 0.0], "gps_en": [4.0 * i, 0.5], "flags": ["far_from_gps"] if i == 2 else []}
        track.add_element(i + 1, 0)
    rec.add_point3D(np.array([0.0, 0.0, 5.0]), track, np.array([10, 20, 30], np.uint8))
    rec.write(str(job / "sparse" / "0"))
    meta = tmp_path / "m.json"
    meta.write_text(json.dumps({"georegistered": True, "yaw_sigma_deg": 4.0, "scale_sigma": 0.02, "cameras": cams}))
    assert len(sm.load_job(job, meta)) == 5                                   # the harness's call is unchanged
    _, _, posed, source, _, tracks = sm.load_job(job, meta, with_tracks=True)
    assert source["flagged_cameras_left_out"] == ["p2"] and len(posed) == 4
    assert tracks.point.tolist() == [0, 0, 0, 0]
    seen = sorted(posed[c].photo_id for c in tracks.camera)
    assert seen == ["p0", "p1", "p3", "p4"]                                   # p2's observation is gone
    for c, xy in zip(tracks.camera, tracks.xy):                               # each observation from its own photo
        assert xy[0] == 10.0 + int(posed[c].photo_id[1:])


def arc_cameras(n: int = 10, dist: float = 80.0, rise_deg: float = 0.0, seed: int = 0, noise: float = 0.2) -> np.ndarray:
    """Cameras south of BOX at 0.8-1.2 x dist, heights 1.6 m (+-noise) rising rise_deg with distance."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-25.0, 25.0, n)
    y = -6.0 - dist * rng.uniform(0.8, 1.2, n)
    z = 1.6 + rng.normal(0.0, noise, n) + math.tan(math.radians(rise_deg)) * (-6.0 - y - dist)
    return np.stack([x, y, z], 1)


def test_camera_tilt_check_reads_a_tilt_from_the_cameras_heights():
    """Frauenkirche SW#14: its cameras' heights rose 2.4 deg with distance and the model was 2.9 deg tilted: the
    photos landed 3-4 m too high on walls 80 m away, which no 2-D check sees."""
    walls = np.stack([np.linspace(-10, 10, 50), np.full(50, -6.0)], 1)
    level = st.camera_tilt_check(arc_cameras(), walls)
    assert level["ok"] and level["shift_m"] < 0.6, level
    tilted = st.camera_tilt_check(arc_cameras(rise_deg=2.9), walls)
    assert not tilted["ok"] and 3.0 < tilted["shift_m"] < 5.0 and tilted["slope_deg"] == pytest.approx(2.9, abs=0.5)
    terrace = arc_cameras()
    terrace[0, 2] += 12.0                                    # one photo from a terrace: left out, still level
    assert st.camera_tilt_check(terrace, walls)["ok"]
    few = st.camera_tilt_check(arc_cameras(n=4), walls)
    assert not few["ok"] and "too few" in few["note"]
    line = arc_cameras()
    line[:, 1] = -86.0                                       # one line along the facade: the tilt towards it is unseen
    assert not st.camera_tilt_check(line, walls)["ok"]
    # level on average, but six cameras over 12 m of depth with 1.2 m of height scatter cannot tell a 1.5 m shift
    # at the walls from none: its standard error counts (snap_tilt_sigmas)
    scatter = np.array([[x, y, 1.6 + dz] for x, y, dz in ((-20, -74, 0.6), (0, -74, -1.2), (20, -74, 0.6),
                                                           (-20, -86, -0.6), (0, -86, 1.2), (20, -86, -0.6))])
    unsure = st.camera_tilt_check(scatter, walls)
    assert unsure["shift_m"] < 0.5 and unsure["se_m"] > 1.5 and not unsure["ok"], unsure


def test_the_snap_refuses_a_tilted_model():
    pts, _, _ = two_sided()
    cams = arc_cameras(dist=30.0, rise_deg=4.0)
    prior = st.SnapPrior(4.0, scale_sigma=0.01, camera_xyz=cams)
    tilted = st.snap_to_footprint(turned(pts, 1.0, (1.0, 1.0)), BOX, prior=prior)
    assert not tilted.accepted and "tilted" in tilted.reason and not tilted.search["tilt"]["ok"], tilted.reason
    # level, but with a depth spread of only 12 m the slope's uncertainty counts too: noisier heights refuse
    level = st.snap_to_footprint(turned(pts, 1.0, (1.0, 1.0)), BOX,
                                 prior=st.SnapPrior(4.0, scale_sigma=0.01, camera_xyz=arc_cameras(dist=30.0, noise=0.05)))
    assert level.accepted and level.search["tilt"]["ok"], level.reason
    unsure = st.snap_to_footprint(turned(pts, 1.0, (1.0, 1.0)), BOX,
                                  prior=st.SnapPrior(4.0, scale_sigma=0.01, camera_xyz=arc_cameras(dist=30.0, noise=0.6)))
    assert not unsure.accepted and unsure.search["tilt"]["se_m"] > 0.5, unsure.search["tilt"]
    # a georegistered model must carry its cameras: without them nothing about them can be checked
    bare = st.snap_to_footprint(turned(pts, 1.0, (1.0, 1.0)), BOX,
                                prior=st.SnapPrior(4.0, scale_sigma=0.01, georegistered=True))
    assert not bare.accepted and "no camera positions" in bare.reason


def test_a_face_that_no_camera_can_see_refuses_the_registration():
    """The neighbour alias (verifier r2, rect 34 x 13.1 m): the snap slid a model so that a neighbour's facade
    across the street sat on the front edge and the real facade 1 m inside the back edge; every edge verified.
    Seen from the cameras in front, those back points lie behind the whole building."""
    rect = np.array([[-17.0, -6.55], [17.0, -6.55], [17.0, 6.55], [-17.0, 6.55]])
    back = walls_on(rect, (2,), step=0.3) + np.array([0.0, -1.0, 0.0])      # the real facade, slid onto the back
    front = walls_on(rect, (0,), step=0.6)                                    # the neighbour's facade
    side = walls_on(rect, (1,), step=0.3)
    pts = np.vstack([back, front, side])
    cams = np.stack([np.linspace(-15, 15, 8), np.full(8, -50.0)], 1)
    assert st.registration_check(pts, [rect])["accepted"]                     # without the cameras: verified
    reg = st.registration_check(pts, [rect], cameras_xy=cams)
    assert not reg["accepted"] and "no camera can see" in reg["reason"]
    assert any(e.get("behind_every_camera") for e in reg["edges"])
    # a facade standing 2 m inside an outline drawn around its plinth is seen fine (and so are its ends,
    # which lie nearer the side edges): accepted
    inner = np.vstack([walls_on(rect, (0,), step=0.3) + np.array([0.0, 2.0, 0.0]), side])
    assert st.registration_check(inner, [rect], cameras_xy=np.vstack([cams, [[45.0, 0.0]]]))["accepted"]
    # a facade may sit up to reg_edge_limit_m (3 m) inside the outline, no more: on a building 6 m deep, points on
    # the back face are still 2-3 m inside the front of it when moved 3 m towards the cameras
    thin = np.array([[-17.0, -3.0], [17.0, -3.0], [17.0, 3.0], [-17.0, 3.0]])
    thin_pts = np.vstack([walls_on(thin, (2,), step=0.3) + np.array([0.0, -0.5, 0.0]), walls_on(thin, (0,), step=0.6),
                          walls_on(thin, (1,), step=0.3)])
    thin_reg = st.registration_check(thin_pts, [thin], cameras_xy=cams)
    assert not thin_reg["accepted"] and "no camera can see" in thin_reg["reason"]
    # a roof edge seen over the walls is not a facade: points high up near the back edge do not refuse
    roof = np.c_[np.linspace(-16, 16, 200), np.full(200, 5.8), np.full(200, 18.0)]
    low_front = walls_on(rect, (0,), step=0.3, z=(3.0, 6.0, 9.0, 12.0, 15.0, 18.0))
    assert st.registration_check(np.vstack([low_front, side, roof]), [rect],
                                 cameras_xy=np.vstack([cams, [[45.0, 0.0]]]))["accepted"]


def test_an_l_of_two_faces_leaves_the_scale_open():
    """Verifier r2 street_L: a 34 m x 13 m L (long + short face) 7 % too large fits its outline as well as at
    the right scale (the points only stop short of a corner or overshoot it a little): with the GPS scale
    known to 5 % the photos at the far end could be 2 m off, so the snap is refused; known to 1 % it is not."""
    rect = st.ensure_ccw(np.array([[-17.0, -6.55], [17.0, -6.55], [17.0, 6.55], [-17.0, 6.55]]))
    walls = walls_on(rect, (0, 1), step=0.3)
    walls[:, :2] += np.random.default_rng(4).normal(0.0, 0.15, (len(walls), 2))
    at = np.median(walls[:, :2], axis=0)
    big = walls.copy()
    big[:, :2] = (walls[:, :2] - at) * 1.07 + at
    big[:, 2] *= 1.07
    open_scale = st.snap_to_footprint(turned(big, 2.0, (1.0, -1.0), about=at), rect,
                                      prior=st.SnapPrior(4.0, scale_sigma=0.05))
    assert not open_scale.accepted and "rescaled" in open_scale.reason, open_scale.reason
    assert any(r["gap_m"] >= 1.5 for r in open_scale.search["scale_profile"])
    known = st.snap_to_footprint(turned(walls, 2.0, (1.0, -1.0), about=at), rect,
                                 prior=st.SnapPrior(4.0, scale_sigma=0.01))
    assert known.accepted, known.reason
    assert known.search["scale_profile"] == []               # +-3 % is beyond 2 sigma of a 1 % scale


def test_cameras_far_away_make_a_small_turn_ambiguous():
    """A right registration has its cameras within 1.5 m of the reference (STYLIZED_MODEL 7): a turn the walls
    hardly notice moves cameras 80 m away 1.4 m per degree. The profile tries the turn that moves them 1.5 m;
    when the walls fit about as well there, the yaw is open at the cameras (Frauenkirche SW#11: -1.2 deg)."""
    walls = walls_on(BOX, (0, 1, 2), step=0.25)
    walls[:, :2] += np.random.default_rng(8).normal(0.0, 0.25, (len(walls), 2))
    cams = arc_cameras(dist=80.0)
    model, cam_model = turned(walls, 1.0, (1.0, -1.0)), turned(cams, 1.0, (1.0, -1.0))
    prior = st.SnapPrior(4.0, scale_sigma=0.01, camera_xyz=cam_model)
    snap = st.snap_to_footprint(model, BOX, prior=prior)
    assert not snap.accepted and snap.reason.startswith("ambiguous"), snap.reason
    turns = {t["turn_deg"]: t for t in snap.search["turn_profile"]["turns"]}
    step = min(abs(k) for k in turns)
    assert step < 1.5 and turns[step]["gap_m"] < 1.5 and turns[step]["cameras_moved_m"] >= 1.5
    walls_only = dataclasses.replace(st.DEFAULT_PARAMS, snap_ambiguity_cameras=False)
    assert st.snap_to_footprint(model, BOX, walls_only, prior=prior).accepted


def test_a_rival_counts_when_it_turns_the_photos_moves_them_or_moves_the_cameras():
    """The rival rule on its own (round-2 mutation check: a rival turned >= 2 deg whose photos move less than
    1.5 m survived every test)."""
    base = {"rot": 2.5, "gap": 0.4, "cam_gap": 0.6, "fit": 0.31, "verified": True}
    assert st._rival(base, 0.0, 0.30)                                       # turned 2.5 deg
    assert not st._rival(dict(base, rot=1.5), 0.0, 0.30)                    # 1.5 deg, 0.4 m, cameras 0.6 m
    assert st._rival(dict(base, rot=1.5, gap=1.6), 0.0, 0.30)               # photos 1.6 m away
    assert st._rival(dict(base, rot=1.5, cam_gap=1.6), 0.0, 0.30)           # cameras 1.6 m away ...
    assert not st._rival(dict(base, rot=1.5, cam_gap=1.6), 0.0, 0.30, cameras=False)
    assert not st._rival(dict(base, verified=False), 0.0, 0.30)             # does not verify edge by edge
    assert not st._rival(dict(base, fit=0.30 * 1.25 + 0.1 + 0.01), 0.0, 0.30)   # fits clearly worse
    assert st._rival(dict(base, fit=0.30 * 1.25 + 0.1 - 0.01), 0.0, 0.30)


def test_an_outline_asking_for_a_larger_rescale_than_the_bound_is_refused():
    """A model 25 % too small: the similarity step stops at the 1.15x bound, and such a best placement is refused
    (the model's shape, not its GPS scale, is off; round-2 mutation check: the refusal survived every test)."""
    walls = walls_on(BOX, (0, 1, 2, 3), step=0.25)
    snap = st.snap_to_footprint(walls * np.array([0.75, 0.75, 1.0]), BOX, prior=st.SnapPrior(4.0, scale_sigma=0.01))
    assert not snap.accepted and "rescale beyond" in snap.reason, snap.reason


def test_walls_still_off_the_outline_after_the_snap_are_refused_as_such():
    """The residual refusal comes before the loose-fit one (round-2 mutation check: removing it survived)."""
    walls = walls_on(BOX, (0, 1, 2, 3), step=0.25)
    walls[:, :2] += np.random.default_rng(9).normal(0.0, 2.5, (len(walls), 2))
    snap = st.snap_to_footprint(walls, BOX, prior=st.SnapPrior(4.0, scale_sigma=0.01))
    assert not snap.accepted and snap.reason.startswith("walls still"), snap.reason


def test_registration_needs_the_share_of_verified_edges():
    """Two of four supported edges verified (0.5 < reg_min_share 0.6): refused (round-2 mutation check: 0.3
    survived every test)."""
    walls = walls_on(BOX, (0, 1, 2, 3), step=0.25)
    out = st.ensure_ccw(BOX)
    north_west = (walls[:, 1] > 5.9) | (walls[:, 0] < -9.9)
    pushed = walls.copy()
    pushed[north_west & (walls[:, 1] > 5.9), 1] += 4.0                     # north wall 4 m out
    pushed[north_west & (walls[:, 0] < -9.9), 0] -= 4.0                    # west wall 4 m out
    reg = st.registration_check(pushed, [out])
    assert reg["voting_edges"] == 4 and reg["verified_edges"] == 2
    assert not reg["accepted"] and reg["reason"].startswith("only 2/4")
    assert st.registration_check(walls, [out])["accepted"]


def test_the_final_gate_takes_the_cameras_too():
    """stylize_landmark's own registration check gets the (snapped) cameras: points on a face that no camera can
    see refuse the colour even without a snap (round-3 mutation check)."""
    rect = np.array([[-17.0, -6.55], [17.0, -6.55], [17.0, 6.55], [-17.0, 6.55]])
    pts = np.vstack([walls_on(rect, (2,), step=0.3) + np.array([0.0, -1.0, 0.0]), walls_on(rect, (0,), step=0.6),
                     walls_on(rect, (1,), step=0.3)])
    cams = []
    for i, x in enumerate(np.linspace(-15, 15, 6)):
        cam = look_at([x, -50.0, 1.6], [0.0, 0.0, 6.0], name=f"c{i}")
        cam.image = np.full((cam.height, cam.width, 3), 150, np.uint8)
        cams.append(cam)
    scene = st.SceneInput("Alias", rect, pts, np.full((len(pts), 3), 200, np.uint8), cams, snap=False)
    model = st.stylize_landmark(scene)
    reg = model.summary["registration"]
    assert not reg["accepted"] and "no camera can see" in reg["reason"] and not model.facades
    assert model.summary["qa"]["passed"]
