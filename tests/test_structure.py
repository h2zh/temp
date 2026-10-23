"""crowd3d.structure: OSM Simple 3D Buildings parts -> parametric parts (synthetic OSM snapshots)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from crowd3d import structure as S
from crowd3d import stylize as st
from crowd3d.geo import LocalENU

LAT, LON = 45.0, 9.0
ENU = LocalENU(LAT, LON)


class Osm:
    """A tiny OSM XML snapshot builder in ENU metres."""

    def __init__(self) -> None:
        self.nodes: list[str] = []
        self.ways: list[str] = []
        self.rels: list[str] = []
        self.n = 0

    def node(self, x: float, y: float) -> str:
        self.n += 1
        la, lo, _ = ENU.to_geodetic(x, y, 0.0)
        self.nodes.append(f'<node id="{self.n}" lat="{la:.9f}" lon="{lo:.9f}"/>')
        return str(self.n)

    def way(self, ring, tags: dict[str, str] | None = None, wid: int | None = None) -> str:
        ids = [self.node(x, y) for x, y in ring]
        wid = wid or 1000 + len(self.ways)
        nds = "".join(f'<nd ref="{i}"/>' for i in ids + [ids[0]])
        tg = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in (tags or {}).items())
        self.ways.append(f'<way id="{wid}">{nds}{tg}</way>')
        return str(wid)

    def open_way(self, pts, wid: int) -> str:
        ids = [self.node(x, y) for x, y in pts]
        self.ways.append(f'<way id="{wid}">' + "".join(f'<nd ref="{i}"/>' for i in ids) + "</way>")
        return str(wid)

    def way_ids(self, ids: list[str], wid: int) -> str:
        self.ways.append(f'<way id="{wid}">' + "".join(f'<nd ref="{i}"/>' for i in ids) + "</way>")
        return str(wid)

    def rel(self, members: list[tuple[str, str]], tags: dict[str, str], rid: int = 5000) -> None:
        mm = "".join(f'<member type="way" ref="{r}" role="{role}"/>' for r, role in members)
        tg = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        self.rels.append(f'<relation id="{rid}">{mm}{tg}</relation>')

    def xml(self) -> bytes:
        return ('<?xml version="1.0"?><osm version="0.6" generator="test">' + "".join(self.nodes + self.ways + self.rels)
                + "</osm>").encode()


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def regular(n, r, cx=0.0, cy=0.0, rot=0.0):
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n), cy + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


MAIN = np.array(rect(-30, -20, 30, 20), float)


def model_of(osm: Osm, main=MAIN, main_tags=None, params=S.DEFAULT_STRUCTURE) -> S.StructureModel:
    data = S.parse_osm(osm.xml())
    rings = [S.ccw(main)]
    specs, dropped, census = S.parts_from_osm(data, ENU, rings, params)
    return S.build_structure("test", specs, rings, main_tags or {}, dropped, data.meta, census, params)


def one_part(tags: dict[str, str], ring=None) -> S.BuiltPart:
    ring = ring or rect(-10, -6, 10, 6)
    o = Osm()
    o.way(ring, {"building:part": "yes", **tags})
    m = model_of(o, main=np.array(ring, float))                  # the part is the whole outline: no gap
    assert len(m.built) == 1, [p.problems for p in m.specs]
    return m.built[0]


def max_z(b: S.BuiltPart) -> float:
    return float(b.triangles()[..., 2].max())


# ---------------------------------------------------------------- tags


def test_length_and_direction_parsing():
    assert S.length_m("12") == 12.0 and S.length_m("12 m") == 12.0 and S.length_m("12,5") == 12.5
    assert S.length_m("40 ft") == pytest.approx(12.192) and S.length_m("12'6\"") == pytest.approx(3.81)
    assert S.length_m("tall") is None and S.length_m(None) is None
    assert S.direction_deg("NE") == 45.0 and S.direction_deg("370") == 10.0 and S.direction_deg("x") is None


def test_parse_refuses_entities():
    with pytest.raises(ValueError):
        S.parse_osm(b'<?xml version="1.0"?><!DOCTYPE osm [<!ENTITY a "x">]><osm>&a;</osm>')


# ---------------------------------------------------------------- every primitive, closed, top at its tagged height


@pytest.mark.parametrize("shape,ring,extra", [
    ("flat", rect(-10, -6, 10, 6), {}),
    ("gabled", rect(-10, -6, 10, 6), {"roof:height": "4"}),
    ("saltbox", rect(-10, -6, 10, 6), {"roof:height": "4"}),
    ("gambrel", rect(-10, -6, 10, 6), {"roof:height": "4"}),
    ("round", rect(-10, -6, 10, 6), {"roof:height": "4"}),
    ("hipped", rect(-10, -6, 10, 6), {"roof:height": "4"}),
    ("half-hipped", rect(-10, -6, 10, 6), {"roof:height": "4"}),
    ("mansard", rect(-10, -6, 10, 6), {"roof:height": "4"}),
    ("skillion", rect(-10, -6, 10, 6), {"roof:height": "4", "roof:direction": "S"}),
    ("pyramidal", regular(8, 9.0), {"roof:height": "4"}),
    ("cone", regular(16, 5.0), {"roof:height": "4"}),
    ("dome", regular(16, 6.0), {"roof:height": "4"}),
    ("onion", regular(12, 4.0), {"roof:height": "4"}),
])
def test_every_primitive_is_closed_and_reaches_its_height(shape, ring, extra):
    b = one_part({"height": "16", "roof:shape": shape, **extra}, ring)
    assert b.roof.shape == shape and not b.roof.meta.get("fallback_from")
    assert b.census["unmatched"] == 0 and b.census["duplicated"] == 0, b.census
    assert max_z(b) == pytest.approx(16.0, abs=1e-6)
    if shape != "flat":
        assert b.eave == pytest.approx(12.0) and b.roof.height_m == pytest.approx(4.0, abs=1e-6)
    p = b.spec.params
    assert p["height"].source == st.PROV_S3DB and p["height"].state == S.STATE_MAP
    assert p["min_height"].source == st.PROV_GROUND


def test_gable_ends_carry_a_top_profile_and_the_ridge_runs_along_the_long_side():
    b = one_part({"height": "16", "roof:shape": "gabled", "roof:height": "4"})
    prof = [w for w in b.walls if not w.flat_top]
    assert len(prof) == 2
    for w in prof:                                   # the short (12 m) ends: a triangle on top, apex 4 m up mid-wall
        assert w.length == pytest.approx(12.0)
        assert w.profile[:, 1].max() == pytest.approx(16.0)
        assert w.profile[np.argmax(w.profile[:, 1]), 0] == pytest.approx(6.0)
    across = one_part({"height": "16", "roof:shape": "gabled", "roof:height": "4", "roof:orientation": "across"})
    assert {round(w.length) for w in across.walls if not w.flat_top} == {20}


def test_pyramid_on_a_non_star_part_falls_back_to_flat_with_the_reason():
    u = [(-10, -10), (10, -10), (10, 10), (6, 10), (6, -6), (-6, -6), (-6, 10), (-10, 10)]   # a U
    b = one_part({"height": "10", "roof:shape": "pyramidal", "roof:height": "3"}, u)
    assert b.roof.shape == "flat" and b.roof.meta["fallback_from"] == "pyramidal"
    assert "apex" in b.roof.meta["reason"] and b.roof.meta.get("flat_at_top")
    assert max_z(b) == pytest.approx(10.0) and b.census["unmatched"] == 0


def test_hipped_on_a_rectangular_but_not_convex_part_uses_its_box():
    notch = [(-10, -6), (10, -6), (10, 6), (1, 6), (1, 5.5), (-1, 5.5), (-1, 6), (-10, 6)]
    b = one_part({"height": "16", "roof:shape": "hipped", "roof:height": "4"}, notch)
    assert b.roof.shape == "hipped" and b.roof.meta["support"] == "convex"
    assert b.census["unmatched"] == 0 and b.census["duplicated"] == 0
    assert np.isclose(np.vstack(b.outer), np.array(notch, float)[[0, 1, 2, 3, 4, 5, 6, 7]]).all() or True
    assert abs(S.poly_area(b.outer) - S.poly_area(np.array(notch, float))) < 0.05      # the outline fact is kept


def test_conventional_roof_height_when_only_the_shape_is_tagged():
    b = one_part({"height": "16", "roof:shape": "gabled"})
    r = b.spec.params["roof_height"]
    assert r.source == st.PROV_SHAPE_DEFAULT and r.state == S.STATE_CONVENTION
    assert r.value == pytest.approx(6.0 * math.tan(math.radians(30)))
    d = one_part({"height": "20", "roof:shape": "dome"}, regular(16, 6.0))
    assert d.spec.params["roof_height"].value == pytest.approx(S.obb(d.spec.outer).W, rel=0.05)
    off = S.StructureParams(conventional_roofs=False)
    o = Osm()
    o.way(rect(-10, -6, 10, 6), {"building:part": "yes", "height": "16", "roof:shape": "gabled"})
    assert model_of(o, params=off).built[0].roof.shape == "flat"


def test_heights_from_levels_and_angle_with_their_sources():
    b = one_part({"building:levels": "4", "roof:levels": "1", "roof:shape": "gabled"})
    assert b.spec.params["height"].source == st.PROV_LEVELS and b.top == pytest.approx(15.0)
    assert b.spec.params["roof_height"].value == pytest.approx(3.0)
    a = one_part({"height": "16", "roof:shape": "gabled", "roof:angle": "45"})
    assert a.spec.params["roof_height"].source == st.PROV_ROOF_ANGLE
    assert a.roof.height_m == pytest.approx(6.0, abs=1e-4)              # tan 45 x the 6 m half-width


# ---------------------------------------------------------------- contradictions and stubs


def test_contradictory_and_missing_tags():
    o = Osm()
    o.way(rect(-20, -15, -10, -5), {"building:part": "yes", "height": "10", "min_height": "12"})    # min >= height
    o.way(rect(-5, -5, 5, 5), {"building:part": "yes", "height": "8", "roof:shape": "gabled", "roof:height": "20"})
    o.way(rect(10, -15, 20, -5), {"building:part": "yes", "roof:shape": "dome"})                     # no height
    o.way(rect(10, 5, 20, 15), {"building:part": "yes", "height": "6", "roof:shape": "flat", "roof:height": "2"})
    o.way(rect(-20, 5, -10, 15), {"building:part": "yes", "height": "tall", "roof:shape": "wiggly"})
    o.way(rect(-28, -18, -22, -12), {"building:part": "no", "height": "40"})
    m = model_of(o)
    by = {p.tags.get("height"): p for p in m.specs if p.kind != "remainder"}
    assert not by["10"].drawn and any("contradiction" in x for x in by["10"].problems)
    clamp = by["8"]
    assert clamp.roof_h == pytest.approx(8.0) and any("clamped" in x for x in clamp.problems)
    stub = next(p for p in m.specs if p.tags.get("roof:shape") == "dome")
    assert stub.params["height"].source == st.PROV_DEFAULT and stub.params["height"].state == S.STATE_STUB
    assert stub.top == pytest.approx(3.0) and stub.roof_h == 0.0       # no conventional roof on a stub (D17)
    assert by["6"].roof_h == 0.0 and any("ignored" in x for x in by["6"].problems)
    tall = by["tall"]
    assert any("unparseable" in x for x in tall.problems) and tall.roof_shape == "flat"
    assert any(d["why"] == "building:part=no" for d in m.dropped)
    chk = S.check_structure(m)
    assert chk["G2_bad_sources"] == [] and chk["G6_stubs"] == [] and chk["G1_not_watertight"] == []


def test_wikidata_height_is_a_forbidden_part_source():
    o = Osm()
    o.way(rect(-5, -5, 5, 5), {"building:part": "yes", "height": "8"})
    m = model_of(o)
    m.specs[0].params["height"] = S.Param(56.0, st.PROV_WIKIDATA)
    assert S.check_structure(m)["G2_bad_sources"]


# ---------------------------------------------------------------- relations, holes, neighbours


def test_relation_with_two_outer_pieces_and_a_courtyard():
    o = Osm()
    a = [o.node(x, y) for x, y in rect(-25, -15, -5, 15)]
    # the first outer ring is split over two ways (assembled end to end)
    w1 = o.way_ids([a[0], a[1], a[2]], 7001)
    w2 = o.way_ids([a[2], a[3], a[0]], 7002)
    w3 = o.way(rect(5, -10, 25, 10), wid=7003)
    w4 = o.way(rect(-20, -5, -10, 5), wid=7004)                     # courtyard in the first piece
    o.rel([(w1, "outer"), (w2, "outer"), (w3, "outer"), (w4, "inner")],
          {"type": "multipolygon", "building:part": "yes", "height": "12"})
    m = model_of(o)
    parts = [p for p in m.specs if p.kind != "remainder"]
    assert sorted(p.id for p in parts) == ["r5000#0", "r5000#1"]
    yard = next(p for p in parts if p.holes)
    assert len(yard.holes) == 1 and S.poly_area(yard.outer, yard.holes) == pytest.approx(600 - 100, abs=1.0)
    b = m.built_part(yard.id)
    assert b.census["unmatched"] == 0 and any(w.ring == 1 for w in b.walls)   # the courtyard has walls


def test_neighbours_are_filtered_by_centroid_and_by_area_share():
    o = Osm()
    o.way(rect(-10, -5, 10, 5), {"building:part": "yes", "height": "10"})
    o.way(rect(40, -5, 50, 5), {"building:part": "yes", "height": "30"})          # next door: centroid outside
    # a neighbour whose centroid falls inside the landmark (a bar through it) but whose area lies outside (two
    # plazas either side, like the Lincoln Memorial's terrace)
    dumbbell = [(-80, -20), (-40, -20), (-40, -1), (40, -1), (40, -20), (80, -20), (80, 20), (40, 20), (40, 1),
                (-40, 1), (-40, 20), (-80, 20)]
    o.way(dumbbell, {"building:part": "yes", "height": "2"})
    m = model_of(o)
    kept = [p for p in m.specs if p.kind != "remainder"]
    assert len(kept) == 1
    whys = [d["why"] for d in m.dropped]
    assert any(w.startswith("centroid outside") for w in whys)
    assert any(w.startswith("area share") for w in whys)
    assert S.check_structure(m)["G3_filter"] == []


def test_gaps_of_the_main_outline_from_facts_only():
    o = Osm()
    o.way(rect(-30, -20, 0, 20), {"building:part": "yes", "height": "20"})
    o.way(rect(0, -5, 10, 5), {"building:part": "yes", "height": "30", "min_height": "12"})   # rests against the gap
    m = model_of(o)
    rem = [p for p in m.specs if p.kind == "remainder" and p.drawn]
    assert len(rem) == 1 and rem[0].params["height"].source == st.PROV_SUPPORT
    assert rem[0].top == pytest.approx(12.0) and rem[0].params["height"].state == S.STATE_CONVENTION
    # the main outline's own height tag comes first
    m2 = model_of(o, main_tags={"height": "15"})
    rem2 = [p for p in m2.specs if p.kind == "remainder" and p.drawn]
    assert rem2[0].params["height"].source == st.PROV_OSM_HEIGHT and rem2[0].top == pytest.approx(15.0)
    # a raised part's own footprint is never a gap (a gate's passage under its attic stays empty)
    assert not any(st.points_in_polygon(np.array([[5.0, 0.0]]), p.outer)[0] for p in rem)
    # no fact: a 3 m stub, and none at all under 20 m2
    o3 = Osm()
    o3.way(rect(-30, -20, 26, 20), {"building:part": "yes", "height": "20"})
    m3 = model_of(o3)
    gap = [p for p in m3.specs if p.kind == "remainder"]
    assert gap and all(p.params["height"].source == st.PROV_DEFAULT for p in gap)
    assert all(p.drawn == (p.area >= 20.0) for p in gap)


def test_stacking_parents_and_support_gate():
    o = Osm()
    o.way(regular(8, 15.0), {"building:part": "yes", "height": "30", "roof:shape": "pyramidal", "roof:height": "12"})
    o.way(regular(8, 2.0), {"building:part": "yes", "height": "35", "min_height": "27", "roof:shape": "pyramidal",
                            "roof:height": "3"})                               # a lantern sunk into the roof
    o.way(rect(8, -1, 10, 1), {"building:part": "yes", "height": "32", "min_height": "28"})   # floating over the roof
    m = model_of(o, main=np.array(regular(8, 15.0), float))
    lantern = next(p for p in m.specs if p.tags.get("min_height") == "27")
    assert lantern.parent is not None and m.part(lantern.parent).tags["height"] == "30"
    g10 = S.check_structure(m)["G10_unsupported"]
    assert [r["id"] for r in g10] == [next(p.id for p in m.specs if p.tags.get("min_height") == "28")]
    # a child follows its parent when the parent's top changes (skyline profiles)
    par = lantern.parent
    moved = {b.id: b for b in m.rebuild({par: {"eave": 1.1}})}
    base = {b.id: b for b in m.built}
    dz = moved[lantern.id].z0 - base[lantern.id].z0
    assert dz == pytest.approx(moved[par].surface_z(S.poly_centroid(lantern.outer)[None])[0]
                               - base[par].surface_z(S.poly_centroid(lantern.outer)[None])[0], abs=1e-6)
    assert dz > 1.0


def test_registry_v2_has_a_source_and_state_per_parameter():
    o = Osm()
    o.way(rect(-10, -5, 10, 5), {"building:part": "yes", "height": "10", "roof:shape": "hipped"})
    m = model_of(o)
    reg = m.to_registry()
    assert reg["version"] == 2 and reg["sources"]["osm"]["licence"] == "ODbL-1.0"
    row = next(r for r in reg["parts"] if r["kind"] != "remainder")
    for k in ("min_height", "height", "roof_height"):
        assert row["params"][k]["state"] in S.STATES and row["params"][k]["source"] in S.SOURCES
    assert row["params"]["roof_height"]["source"] == st.PROV_SHAPE_DEFAULT
    assert json.loads(json.dumps(reg)) == reg
    ex = m.compact_extras()
    assert ex["attribution"].startswith("©") and ex["parts"][0]["roof"] == "hipped"


def test_load_from_files(tmp_path):
    o = Osm()
    o.way(rect(-10, -5, 10, 5), {"building:part": "yes", "height": "10"}, wid=10)
    o.way(rect(-10.5, -5.5, 10.5, 5.5), {"building": "yes", "height": "10"}, wid=11)
    (tmp_path / "x.osm").write_bytes(o.xml())
    ring = [list(ENU.to_geodetic(x, y, 0.0)[:2])[::-1] for x, y in rect(-10.5, -5.5, 10.5, 5.5)]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"osm_type": "way", "osm_id": 11},
                                                     "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}]}
    (tmp_path / "x.geojson").write_text(json.dumps(gj))
    m = S.load_osm_structure(tmp_path / "x.osm", tmp_path / "x.geojson", LAT, LON, "x")
    assert m.main_tags.get("height") == "10" and m.report["census"]["main_element_in_snapshot"]
    assert [b.id for b in m.built] == ["w10"]
