"""crowd3d.stylize with the structure-first massing (crowd3d.structure parts) and the adopted colour fixes."""

from __future__ import annotations

import math

import numpy as np
import pytest
import test_stylize as ts   # the synthetic south-facade scene (pytest puts tests/ on the path)

from crowd3d import structure as S
from crowd3d import stylize as st


def box_structure(roof: str = "gabled", extra: list[S.PartSpec] | None = None) -> S.StructureModel:
    tags = {"height": "14", "roof:shape": roof, "roof:height": "4"} if roof != "flat" else {"height": "10"}
    specs = [S.PartSpec("w100", "yes", ts.BOX.copy(), [], tags, name="box")] + list(extra or [])
    return S.build_structure("Test Box", specs, [ts.BOX.copy()], {})


@pytest.fixture(scope="module")
def parts_model() -> st.StylizedModel:
    scene = ts.south_scene()
    scene.structure = box_structure()
    return st.stylize_landmark(scene)


def test_the_parts_are_the_massing_and_every_check_passes(parts_model):
    s = parts_model.summary
    assert s["massing_source"] == "osm-parts"
    qa = s["qa"]
    assert qa["passed"] and qa["audit"]["passed"] and qa["glb"]["passed"], qa
    pr = parts_model.massing.prisms
    assert len(pr) == 1 and pr[0].cap == "faces" and pr[0].roof_shape == "gabled" and pr[0].part == "w100"
    walls = parts_model.massing.walls
    assert len(walls) == 4 and sum(w.profile is not None for w in walls) == 2       # two gable ends
    assert parts_model.observations[ts.SOUTH].observed                            # the photographed long wall
    assert s["structure"]["version"] == 2 and s["structure"]["parts"][0]["params"]["height"]["source"] == st.PROV_S3DB


def test_gable_walls_and_roof_faces_in_the_glb(parts_model):
    gltf, binary = st.read_glb(parts_model.glb)
    extras = gltf["scenes"][0]["extras"]
    assert extras["structure"]["parts"][0]["roof"] == "gabled"
    assert any("OpenStreetMap" in a for a in extras["attribution"])
    nodes = {n["name"]: n for n in gltf["nodes"]}
    gables = [p for p in parts_model.parts if p.extras.get("kind") == "wall" and p.extras.get("profiled")]
    assert gables
    for p in gables:
        pos = p.positions
        assert len(pos) == 5 and len(p.indices) == 3                               # pentagon: 3 triangles
        assert pos[0][2] == pytest.approx(0.0) and pos[1][2] == pytest.approx(0.0)   # bottom-left, bottom-right first
        assert pos[2][2] == pytest.approx(10.0) and pos[3][2] == pytest.approx(14.0)  # top-right, the ridge, top-left
        assert p.extras["provenance"]["outline"] == st.PROV_S3DB and p.extras["part"] == "w100"
        assert nodes[p.name]["extras"]["locked"]
    roofs = [p for p in parts_model.parts if p.extras.get("kind") == "roof"]
    assert roofs and all(p.extras["cap"] == "gabled" and p.extras["roof"]["shape"] == "gabled" for p in roofs)
    assert sum(len(p.extras["faces"]) for p in roofs) == 2                          # two slopes
    tri = np.concatenate([p.positions[p.indices.reshape(-1)] for p in roofs])
    assert tri[:, 2].max() == pytest.approx(14.0) and tri[:, 2].min() == pytest.approx(10.0)
    for p in parts_model.parts:
        assert all(p.extras["provenance"][k] in st.PROVENANCES for k in st.PROVENANCE_KEYS)


def test_a_dome_and_a_raised_lantern_close_and_show_their_underside():
    dome = S.PartSpec("w200", "yes", np.array([[5.0 * math.cos(a), 5.0 * math.sin(a)] for a in np.linspace(0, 2 * math.pi, 12, endpoint=False)]),
                      [], {"height": "13", "roof:shape": "dome", "roof:height": "3", "min_height": "10"})
    model = box_structure("flat", [dome])
    assert model.part("w200").parent == "w100"
    scene = ts.south_scene()
    scene.structure = model
    m = st.stylize_landmark(scene)
    assert m.summary["qa"]["passed"], m.summary["qa"]
    pr = next(p for p in m.massing.prisms if p.part == "w200")
    assert pr.cap == "faces" and pr.num_faces() == 12 * 6 and pr.z0 == pytest.approx(10.0)
    assert pr.roof_height(np.array([[0.2, 0.1]]))[0] == pytest.approx(13.0, abs=0.3)
    assert any(p.extras.get("kind") == "base" and p.extras.get("part") == "w200" for p in m.parts)
    face = pr.face_of(np.array([[0.0, 0.0], [4.9, 0.0]]))
    assert face[0] != face[1]


def test_roof_shape_stays_where_nobody_photographed_it(parts_model):
    """Coverage decides colour, never shape: a pitched face without points stays pitched, in clay."""
    roofs = [p for p in parts_model.parts if p.extras.get("kind") == "roof"]
    clay = [p for p in roofs if not p.extras["observed"]]
    assert clay and all(p.rgb == st.CLAY_ROOF_RGB for p in clay)
    assert max(p.positions[:, 2].max() for p in roofs) == pytest.approx(14.0)


def test_black_point_colours_are_recoloured_from_the_photos_or_dropped():
    scene = ts.south_scene()
    scene.colors = np.zeros_like(scene.colors)
    m = st.stylize_landmark(scene)
    pc = m.summary["point_colors"]
    assert pc["black_share"] == 1.0 and pc["recolored"] and pc["colored_share"] > 0.5
    assert m.summary["qa"]["passed"]
    # without any photo pixels the colours are dropped: no roof claims to be photographed
    scene2 = ts.south_scene()
    scene2.colors = np.zeros_like(scene2.colors)
    for c in scene2.cameras:
        c.image = None
    cols, valid, rep = st.guard_point_colors(scene2.points, scene2.colors, scene2.cameras)
    assert cols is None and rep["dropped"]


def test_sky_coloured_points_do_not_colour_a_roof():
    """Points on a spire or a roof edge recoloured from photos pick up the sky behind them: light bluish
    ones (b* < -9, L* > 65) and deep blue ones (b* < -20) are dropped before the roof colour is taken."""
    rng = np.random.default_rng(3)
    n = 4000
    pts = np.stack([rng.uniform(-9.5, 9.5, n), rng.uniform(-5.5, 5.5, n), np.full(n, 10.0)], 1)
    massing = st.build_massing_from_structure(box_structure("flat"), pts, st.DEFAULT_PARAMS)
    deep_sky, light_sky, slate = (17, 78, 183), (150, 200, 245), (70, 76, 86)
    cols = np.array([deep_sky if k % 5 < 2 else light_sky if k % 5 == 2 else slate for k in range(n)], np.uint8)
    look = st.roof_appearance(massing, pts, cols)
    rgb = look.face_rgb[(0, 0)]
    assert rgb is not None and st.FAMILY_OF[st.PALETTE_NAMES[st.PALETTE_RGB.tolist().index(list(rgb))]] != "blue"
    look2 = st.roof_appearance(massing, pts, np.tile(np.array(deep_sky, np.uint8), (n, 1)))
    assert look2.face_rgb[(0, 0)] is None and not look2.patches                 # only sky: the roof stays clay


def test_zebra_courses_are_drawn_as_regular_bands():
    """Light / dark courses of stone (Tuscan marble): >= 4 regular courses become clean bands in the dark
    material's own (snapped) colour; a single dark line does not."""
    h, w, ppm = 160, 160, 16.0
    L = np.full((h, w), 80.0, np.float32)
    for k in range(6):                                          # 6 dark courses, 0.4 m thick, every 1.2 m
        y0 = int((1.0 + 1.2 * k) * ppm)
        L[y0:y0 + int(0.4 * ppm)] = 45.0
    mask = np.ones((h, w), bool)
    out, groups = st.detect_stripes(L, mask, ppm)
    assert groups and max(g["count"] for g in groups) >= 4
    rows = out.any(axis=1)
    assert rows.sum() == pytest.approx(6 * 0.4 * ppm, abs=6)
    L2 = np.full((h, w), 80.0, np.float32)
    L2[60:66] = 45.0                                            # one line
    assert not st.detect_stripes(L2, mask, ppm)[0].any()
    # only observed texels
    m2 = mask.copy()
    m2[:, :80] = False
    assert not st.detect_stripes(L, m2, ppm)[0][:, :80].any()


def test_near_greys_snap_by_de00_and_hues_stay_in_their_family():
    lab = np.array([[72.0, 0.2, 1.0],          # white marble in the shade: neutral, not beige stone
                    [70.0, 2.0, 9.0],          # a warm stone stays warm
                    [40.0, 28.0, 22.0]],       # red brick stays red
                   np.float32)
    idx = st.snap_lab_to_palette(lab, lift=0.0)
    names = [st.PALETTE_NAMES[i] for i in idx]
    assert names[0] in ("ash", "stone-light") and st.FAMILY_OF[names[0]] == "neutral" and names[0] != "stone"
    assert names[1] in ("stone", "stone-light", "sand", "cream")
    assert st.FAMILY_OF[names[2]] == "brick"
    assert st.de00(np.array([50.0, 2.6772, -79.7751]), np.array([50.0, 0.0, -82.7485])) == pytest.approx(2.0425, abs=1e-4)
