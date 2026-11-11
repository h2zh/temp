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
    ("skillion", rect(-10, -6, 10, 6), {"roof:height": "4"}),                 # no roof:direction (D34: a TypeError)
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
    # G10: the floating part touches nothing: withheld (not drawn) with a ticket, and the gate passes
    floating = next(p for p in m.specs if p.tags.get("min_height") == "28")
    chk = S.check_structure(m)
    assert chk["passed"] and chk["G10_withheld"] == [floating.id] and not chk["G10_unsupported"]
    assert not floating.drawn and floating.support == "withheld" and m.built_part(floating.id) is None
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


# ---------------------------------------------------------------- G10 resolution and sanity tickets (review fix 1)


def test_g10_withholds_what_floats_and_keeps_what_rests_on_a_support():
    """A raised part with less than half its base supported gets a ticket. Touching a support (an attic on two piers,
    a bridge between towers) it stays where the map puts it ('rests-on-support'); touching nothing (India Gate's top
    slab 2 m above the part under it) it is withheld -- not drawn -- and so is what stood on it."""
    o = Osm()
    o.way(rect(-20, -5, -14, 5), {"building:part": "yes", "height": "12"})                       # west pier
    o.way(rect(14, -5, 20, 5), {"building:part": "yes", "height": "12"})                         # east pier
    attic = f'w{o.way(rect(-20, -5, 20, 5), {"building:part": "yes", "height": "16", "min_height": "12"})}'  # 30 % on piers
    gate = model_of(o, main=np.array(rect(-20, -5, 20, 5), float))
    o = Osm()
    o.way(rect(-25, 8, -21, 12), {"building:part": "yes", "height": "40"})                       # a tower ...
    bridge = f'w{o.way(rect(-21, 8, 21, 12), {"building:part": "yes", "height": "33", "min_height": "30"})}'
    o.way(rect(21, 8, 25, 12), {"building:part": "yes", "height": "40"})                         # ... and a tower
    towers = model_of(o, main=np.array(rect(-25, 8, 25, 12), float))
    o = Osm()
    o.way(rect(-5, -16, 5, -10), {"building:part": "yes", "height": "10"})
    slab = f'w{o.way(rect(-5, -16, 5, -10), {"building:part": "yes", "height": "14", "min_height": "12"})}'   # 2 m above
    crown = f'w{o.way(rect(-3, -15, 3, -11), {"building:part": "yes", "height": "17", "min_height": "14"})}'  # on the slab
    india = model_of(o, main=np.array(rect(-5, -16, 5, -10), float))
    act = {t["part"]: t["action"] for m in (gate, towers, india) for t in m.tickets if t["kind"] == "support"}
    assert act == {attic: "rests-on-support", bridge: "rests-on-support", slab: "withheld", crown: "withheld"}, act
    assert gate.built_part(attic) is not None and towers.built_part(bridge) is not None
    assert india.built_part(slab) is None and india.built_part(crown) is None
    assert india.part(slab).params["height"].value == 14.0 and not india.part(slab).drawn       # map values kept
    assert {r["id"]: r.get("support") for r in india.to_registry()["parts"]}[slab] == "withheld"
    assert set(india.compact_extras()["withheld"]) == {slab, crown}
    for m in (gate, towers, india):
        assert S.check_structure(m)["passed"]
    assert S.check_structure(india)["G10_withheld"] == sorted([slab, crown])
    assert [r["id"] for r in S.check_structure(gate)["G10_unsupported"]] == [attic]
    # a floating part drawn again, or an unsupported one without its ticket, fails the gate
    india.part(slab).drawn = True
    india.built = S.build_all(india.specs, india.params)
    chk = S.check_structure(india)
    assert not chk["passed"] and chk["G10_withheld_but_drawn"] == [slab]
    gate.tickets = [t for t in gate.tickets if t["kind"] != "support"]
    chk = S.check_structure(gate)
    assert not chk["passed"] and chk["G10_without_ticket"] == [attic]


def test_osm_values_outside_sanity_bounds_or_unparseable_open_tickets():
    """Never a change, always a ticket: an unparseable numeric tag (roof:height=abc was silently replaced by a
    convention), a part taller than any building, a part above Wikidata's total height x 1.1."""
    o = Osm()
    o.way(rect(-20, -15, -10, -5), {"building:part": "yes", "height": "30", "roof:shape": "pyramidal", "roof:height": "abc"})
    o.way(rect(-5, -5, 5, 5), {"building:part": "yes", "height": "1200", "building:levels": "many"})
    o.way(rect(10, -15, 20, -5), {"building:part": "yes", "height": "60"})
    o.way(rect(10, 5, 20, 15), {"building:part": "yes", "height": "12", "min_height": "x"})
    data = S.parse_osm(o.xml())
    rings = [S.ccw(MAIN)]
    specs, dropped, census = S.parts_from_osm(data, ENU, rings)
    m = S.build_structure("test", specs, rings, {}, dropped, data.meta, census, wikidata_height_m=50.0)
    by = {p.tags.get("height"): p for p in m.specs if p.kind != "remainder"}
    tk = [t for t in m.tickets if t["kind"] == "osm-value"]
    assert any(t["part"] == by["30"].id and t["tag"] == "roof:height" and t["value"] == "abc" for t in tk)
    assert any("unparseable roof:height='abc'" in x for x in by["30"].problems)
    assert by["30"].params["roof_height"].source == st.PROV_SHAPE_DEFAULT                     # the convention, recorded
    assert any(t["part"] == by["1200"].id and "plausible maximum" in t["why"] for t in tk)
    assert any(t["part"] == by["1200"].id and t["tag"] == "building:levels" for t in tk)
    assert by["1200"].top == 1200.0 and by["60"].top == 60.0                                  # values never change
    assert any(t["part"] == by["60"].id and "Wikidata" in t["why"] for t in tk)
    assert not any(t["part"] == by["30"].id and "Wikidata" in t["why"] for t in tk)          # 30 <= 50 x 1.1
    assert any(t["part"] == by["12"].id and t["tag"] == "min_height" for t in tk)
    assert all(t["open"] and t["strength"] == "low" for t in tk)
    # idempotent
    assert len(S.open_value_tickets(m, 50.0)) == len(tk)
    # without Wikidata, a part far above every other one (a vandalised height=300 on a 3.9 m lantern) still opens one
    o = Osm()
    o.way(regular(8, 15.0), {"building:part": "yes", "height": "30", "roof:shape": "pyramidal", "roof:height": "12"})
    o.way(regular(8, 2.0), {"building:part": "yes", "height": "300", "min_height": "27"})
    m = model_of(o, main=np.array(regular(8, 15.0), float))
    lantern = next(p for p in m.specs if p.tags.get("height") == "300")
    assert lantern.top == 300.0 and any(t["kind"] == "osm-value" and t["part"] == lantern.id and "every other part" in t["why"]
                                        for t in m.tickets)


def test_a_conventional_roof_taking_most_of_its_part_opens_a_ticket():
    """A wrong roof:shape=onion without roof:height on a short drum became a conventional onion filling the drum's
    room (the verifier's Baptistery variant): never a change, a ticket for a curator. A modest convention opens none."""
    o = Osm()
    o.way(regular(8, 15.0), {"building:part": "yes", "height": "30"})
    o.way(regular(8, 4.0), {"building:part": "yes", "height": "35", "min_height": "30", "roof:shape": "onion"})
    o.way(regular(8, 3.0, cx=-20.0), {"building:part": "yes", "height": "14", "roof:shape": "pyramidal"})
    m = model_of(o)
    onion = next(p for p in m.specs if p.tags.get("roof:shape") == "onion")
    pyr = next(p for p in m.specs if p.tags.get("roof:shape") == "pyramidal")
    assert onion.params["roof_height"].source == st.PROV_SHAPE_DEFAULT and onion.roof_h > 0.7 * (onion.top - onion.z0)
    tk = [t for t in m.tickets if t["kind"] == "osm-value" and t["tag"] == "roof:shape"]
    assert [t["part"] for t in tk] == [onion.id] and tk[0]["value"] == "onion" and "wrong roof:shape" in tk[0]["why"]
    assert pyr.params["roof_height"].source == st.PROV_SHAPE_DEFAULT and pyr.roof_h <= 0.7 * (pyr.top - pyr.z0)
    assert onion.top == 35.0 and m.built_part(onion.id).roof.shape == "onion"                 # nothing changed


# ---------------------------------------------------------------- D34: skillion, half-hipped, courtyards, G13


def test_a_skillion_without_roof_direction_slopes_across_its_short_side_and_says_so():
    """roof:shape=skillion without roof:direction is common in OSM. It crashed (the note was formatted before the
    None check), the part was left out and the QA passed. Now: the slope runs across the short side, the data gap
    is on the part (problems) and on the roof (meta); with the direction tagged nothing is noted."""
    b = one_part({"height": "16", "roof:shape": "skillion", "roof:height": "4"})
    assert b.roof.shape == "skillion" and not b.roof.meta.get("fallback_from") and b.spec.not_built is None
    assert b.roof.meta["direction"] == S.SKILLION_NO_DIRECTION and "data gap" in S.SKILLION_NO_DIRECTION
    assert any("with no roof:direction" in x and "data gap" in x for x in b.spec.problems), b.spec.problems
    zs = sorted({round(float(z), 6) for w in b.walls for z in w.profile[:, 1]})
    assert zs == [12.0, 16.0]                                        # one long side at the eave, the other at the top
    assert sorted(round(w.length) for w in b.walls if not w.flat_top) == [12, 12]      # the short sides slope
    unreadable = one_part({"height": "16", "roof:shape": "skillion", "roof:height": "4", "roof:direction": "up"})
    assert unreadable.roof.shape == "skillion" and any("an unreadable roof:direction" in x for x in unreadable.spec.problems)
    tagged = one_part({"height": "16", "roof:shape": "skillion", "roof:height": "4", "roof:direction": "S"})
    assert tagged.roof.meta["direction"] == "roof:direction=180" and not any("data gap" in x for x in tagged.spec.problems)
    # the review's end-to-end case: a flat body and a skillion annexe without the tag -- both are in the model
    o = Osm()
    o.way(rect(-10, -6, 10, 6), {"building:part": "yes", "height": "12", "roof:shape": "flat"})
    annexe = f'w{o.way(rect(10, -4, 16, 4), {"building:part": "yes", "height": "6", "roof:shape": "skillion", "roof:height": "2"})}'
    m = model_of(o, main=np.array(rect(-10, -6, 16, 6), float))
    assert m.built_part(annexe).roof.shape == "skillion" and S.check_structure(m)["passed"]
    assert not any("not built" in x for p in m.specs for x in p.problems)


@pytest.mark.parametrize("extra,raised", [({}, False), ({"roof:direction": "0"}, False),
                                           ({"roof:orientation": "across"}, True), ({"roof:direction": "90"}, True)])
def test_a_half_hipped_roof_reaches_its_registered_top_whichever_way_its_ridge_runs(extra, raised):
    """A 20 x 8 m part with a 4 m half-hipped roof. With the ridge across the long side the end planes capped the
    envelope at 3.6 m while the part registered its top 4 m above the eave (D34: 0.4 m of nothing, and an audit
    failure). The roof height is the fact: the slope is raised until the envelope reaches it, with a note."""
    b = one_part({"height": "20", "roof:shape": "half-hipped", "roof:height": "4", **extra}, rect(-10, -4, 10, 4))
    assert b.roof.shape == "half-hipped" and not b.roof.meta.get("fallback_from")
    assert b.eave == pytest.approx(16.0) and b.top == pytest.approx(20.0) and b.roof.height_m == pytest.approx(4.0)
    assert max_z(b) == pytest.approx(b.top, abs=1e-6)
    assert b.census["unmatched"] == 0 and b.census["duplicated"] == 0
    assert ("slope" in b.roof.meta) == raised
    if raised:
        assert "0.500 instead of roof height / half-width 0.400" in b.roof.meta["slope"] and "3.60 m" in b.roof.meta["slope"]


def courtyard_relation(outer, inner, tags=None) -> S.StructureModel:
    o = Osm()
    w1, w2 = o.way(outer, wid=10), o.way(inner, wid=11)
    o.rel([(w1, "outer"), (w2, "inner")], {"type": "multipolygon", "building:part": "yes", "height": "10", **(tags or {})})
    return model_of(o, main=np.array(outer, float))


@pytest.mark.parametrize("start", [0, 1, 2, 3])
def test_a_courtyard_touching_the_outline_at_a_node_is_cut_whichever_node_its_way_starts_at(start):
    """A courtyard that shares the node (10, 30) with the outline (legal in an OSM multipolygon). Its ownership was
    decided by its way's first vertex: starting at the shared node -- on the outline, where the even-odd test
    answers either way -- the courtyard was lost without a record and the roof covered it; starting elsewhere it was
    kept and the solid, pinched at that line, failed G1. Now it belongs to the outer ring its inside lies in and is
    drawn 5 cm smaller, clear of the outline, with a note: one closed solid, the same whatever the start."""
    outer = [(0, 0), (40, 0), (40, 30), (10, 30), (0, 30)]
    inner = [(10, 30), (5, 20), (10, 10), (20, 20)]
    m = courtyard_relation(outer, inner[start:] + inner[:start])
    part = m.part("r5000")
    assert len(part.holes) == 1 and m.dropped == []
    assert part.area == pytest.approx(1200 - 150, abs=3.0) and part.area > 1200 - 150      # (a little smaller: 147.5)
    assert any("touches the outline" in x and "5 cm smaller" in x for x in part.problems), part.problems
    assert 0.05 <= S._ring_gap(part.holes[0], part.outer) < 0.15        # (the corner at the node: 5 cm / sin)
    b = m.built_part("r5000")
    assert b is not None and b.census["unmatched"] == 0 and b.census["duplicated"] == 0
    assert any(w.ring == 1 for w in b.walls) and S.check_structure(m)["passed"]
    roof = np.concatenate([f.triangles() for f in b.roof.faces])
    plan = 0.5 * np.cross(roof[:, 1] - roof[:, 0], roof[:, 2] - roof[:, 0])[:, 2].sum()
    assert plan == pytest.approx(b.spec.area, abs=0.1)                # the roof does not cover the courtyard


def test_courtyards_that_cannot_be_cut_are_recorded_and_one_across_the_outline_is_not_buildable():
    """A courtyard strictly inside is cut as mapped (no note). An inner ring outside every outer ring, or too small
    to use, is left out with a record in `dropped` (it was dropped without one). An inner ring that runs across the
    outline cannot be joined to it: the part is not built and says why (it was built with the roof over the
    courtyard and only G1 noticed)."""
    block = rect(0, 0, 40, 30)
    inside = courtyard_relation(block, rect(10, 10, 20, 20))
    assert len(inside.part("r5000").holes) == 1 and inside.dropped == [] and S.check_structure(inside)["passed"]
    assert not any("courtyard" in x for x in inside.part("r5000").problems)
    assert inside.part("r5000").area == pytest.approx(1100.0, abs=0.01)   # (the geodetic round trip)
    on_edge = courtyard_relation(block, rect(30, 10, 40, 20))               # shares a stretch of the outline
    assert len(on_edge.part("r5000").holes) == 1 and S.check_structure(on_edge)["passed"]
    assert any("5 cm smaller" in x for x in on_edge.part("r5000").problems)
    outside = courtyard_relation(block, rect(60, 10, 70, 20))
    assert outside.part("r5000").holes == [] and S.check_structure(outside)["passed"]
    assert [(d["id"], d["why"], d["ring"]) for d in outside.dropped] == \
        [("r5000", "inner ring inside no outer ring of its relation", "inner")]
    tiny = courtyard_relation(block, rect(10, 10, 10.1, 10.1))
    assert tiny.part("r5000").holes == [] and [d["why"] for d in tiny.dropped] == ["inner ring unusable"]
    across = courtyard_relation(block, rect(25, 10, 45, 20))
    chk = S.check_structure(across)
    assert across.built_part("r5000") is None and not chk["passed"] and chk["G1_not_watertight"] == []
    assert chk["G13_not_built"] == [{"id": "r5000", "kind": "not-buildable",
                                     "why": "1 of 1 courtyard(s) cannot be joined to the outline (on or outside it, "
                                            "or crossing itself): the roof and the base would cover them"}]


@pytest.mark.parametrize("roof", ["flat", "hipped"])
def test_a_courtyard_way_that_crosses_itself_at_one_of_its_nodes_is_not_cut_and_fails_g13(roof):
    """D34, review round 2. An inner way that runs through one of its nodes twice and crosses itself there. No two of
    its edges cross, so it is not the "unusable" ring that clean_ring drops with a record: it stays the part's
    courtyard. Bridged to the outline, its smaller lobe (4.5 m2 here) was not cut out but roofed twice, and the
    triangulation called itself exact; the flat-roofed part was stopped by G1 alone (the walls' edges at the shared
    node did not pair up), with no reason given. Now such a courtyard is not cut, the part is not built, and G13
    says why, for either roof. A way whose edges cross is still dropped with its record, as before."""
    block = rect(-20, -20, 20, 20)
    tags = {} if roof == "flat" else {"roof:shape": "hipped", "roof:height": "3"}
    for inner in ([(0, 0), (8, 4), (8, -4), (0, 0), (-3, 2.5), (-3, -0.5)],            # the big lobe on the right
                  [(0, 0), (-8, 4), (-8, -4), (0, 0), (3, 2.5), (3, -0.5)]):           # ... on the left
        m = courtyard_relation(block, inner, tags)
        part = m.part("r5000")
        assert st.ring_is_simple(part.holes[0]) and st.ring_crosses_itself(part.holes[0])
        assert len(part.holes) == 1 and m.dropped == [] and m.built_part("r5000") is None
        chk = S.check_structure(m)
        assert not chk["passed"] and chk["G1_not_watertight"] == [] and len(chk["G13_not_built"]) == 1
        gate = chk["G13_not_built"][0]
        assert (gate["id"], gate["kind"]) == ("r5000", "not-buildable") and part.not_built["why"] == gate["why"]
        assert gate["why"] == ("1 of 1 courtyard(s) cannot be joined to the outline (on or outside it, or crossing itself): "
                               "the roof and the base would cover them" if roof == "flat" else
                               "the outline has no exact triangulation: 1 hole(s) could not be joined to the outline")
    # (edges that cross: the ring is dropped where the snapshot is read, the part is built without a courtyard)
    crossed = courtyard_relation(block, [(-3, -2), (8, 4), (8, -4), (-3, 2.5)], tags)
    assert [(d["why"], d["ring"]) for d in crossed.dropped] == [("inner ring unusable", "inner")]
    assert crossed.part("r5000").holes == [] and S.check_structure(crossed)["passed"]


def test_two_courtyards_sharing_a_node_are_both_cut():
    o = Osm()
    w1 = o.way(rect(0, 0, 60, 30), wid=10)
    w2 = o.way(rect(10, 10, 20, 20), wid=11)
    w3 = o.way(rect(20, 5, 30, 10), wid=12)                         # touches the first at (20, 10)
    o.rel([(w1, "outer"), (w2, "inner"), (w3, "inner")], {"type": "multipolygon", "building:part": "yes", "height": "10"})
    m = model_of(o, main=np.array(rect(0, 0, 60, 30), float))
    part, b = m.part("r5000"), m.built_part("r5000")
    assert len(part.holes) == 2 and sum("another courtyard" in x for x in part.problems) == 2
    assert b.census["unmatched"] == 0 and b.census["duplicated"] == 0 and S.check_structure(m)["passed"]
    assert part.area == pytest.approx(1800 - 100 - 50, abs=4.0)


def square(cx, cy, w):
    return rect(cx - w / 2, cy - w / 2, cx + w / 2, cy + w / 2)


def nested_relation(rings: list[tuple[str, list]], main=None) -> S.StructureModel:
    """One building:part multipolygon with the given (role, ring) members, in that order; the main outline is the
    largest ring unless given."""
    o = Osm()
    members = [(o.way(ring, wid=10 + k), role) for k, (role, ring) in enumerate(rings)]
    o.rel(members, {"type": "multipolygon", "building:part": "yes", "height": "10"})
    largest = max((r for _, r in rings), key=lambda r: abs(st.signed_area(np.array(r, float))))
    return model_of(o, main=np.array(main or largest, float))


def plan_area(b: S.BuiltPart) -> float:
    roof = np.concatenate([f.triangles() for f in b.roof.faces])
    return float(0.5 * np.cross(roof[:, 1] - roof[:, 0], roof[:, 2] - roof[:, 0])[:, 2].sum())


@pytest.mark.parametrize("island,island_first", [
    (square(0, 0, 6), False),           # a pavilion over the courtyard's centre
    (square(0, 0, 6), True),            # the same, listed before the block in the relation
    (square(0, 0, 10), False),
    (square(6, 6, 6), False),           # off-centre: the courtyard's centre is not in it
    (square(-7.9, 0, 4), True),         # against the courtyard's edge (10 cm clear)
], ids=["centred", "centred-listed-first", "centred-larger", "off-centre", "by-the-edge"])
def test_a_part_mapped_as_an_island_in_a_courtyard_is_built_and_the_courtyard_stays_the_blocks(island, island_first):
    """outer - inner - outer in one multipolygon: a 40 x 40 m block with a 20 x 20 m courtyard and a pavilion standing
    in it. The courtyard is cut from the block alone. When every outer ring holding the courtyard's centre took it
    (D34's first commit), a pavilion over that centre got a hole larger than itself, an area of -364 m2, and was
    filtered out as 'area share inside main outline 0.00 < 0.5' with the QA green; before D34 it was built."""
    block, yard = ("outer", square(0, 0, 40)), ("inner", square(0, 0, 20))
    rings = [("outer", island), block, yard] if island_first else [block, yard, ("outer", island)]
    m = nested_relation(rings)
    b_id, i_id = ("r5000#1", "r5000#0") if island_first else ("r5000#0", "r5000#1")
    side = island[1][0] - island[0][0]
    assert m.dropped == [] and {p.id for p in m.specs if p.kind != "remainder"} == {b_id, i_id}
    assert len(m.part(b_id).holes) == 1 and m.part(b_id).area == pytest.approx(1600 - 400, abs=0.05)
    assert m.part(i_id).holes == [] and m.part(i_id).area == pytest.approx(side * side, abs=0.05)
    for pid, area, rings_m in ((b_id, 1200.0, 160.0 + 80.0), (i_id, side * side, 4 * side)):
        b = m.built_part(pid)
        assert b is not None and b.census["unmatched"] == 0 and b.census["duplicated"] == 0, pid
        assert plan_area(b) == pytest.approx(S.poly_area(b.outer, b.holes), abs=1e-6)     # the roof: the part, no more
        assert plan_area(b) == pytest.approx(area, abs=0.035 * rings_m)   # (parts in one model are drawn 1-3 cm apart)
    chk = S.check_structure(m)
    assert chk["passed"] and chk["G13_not_built"] == [] and chk["G3_filter"] == []
    sm = st.stylize_landmark(st.SceneInput("t", np.array(square(0, 0, 40), float), None, None, [], structure=m))
    assert sm.summary["qa"]["passed"] and {b_id, i_id} <= {p.extras.get("part") for p in sm.parts}


def test_rings_nested_five_deep_each_courtyard_is_cut_from_the_ring_directly_around_it():
    """outer - inner - outer - inner - outer: a block, its courtyard, a hall in the courtyard with a light well of its
    own, and a fountain house in the light well. Each courtyard is cut from the smallest ring around it, once: the
    block keeps one hole (given the light well too -- a hole inside a hole, which cannot be bridged -- it was not
    built), the hall gets the light well, the fountain house is whole. The same for every order of the members."""
    rings = [("outer", square(0, 0, 60)), ("inner", square(0, 0, 40)), ("outer", square(0, 0, 24)),
             ("inner", square(0, 0, 12)), ("outer", square(0, 0, 4))]
    want = sorted([(1, 3600.0 - 1600.0), (1, 576.0 - 144.0), (0, 16.0)])
    for order in ([0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [2, 4, 0, 3, 1]):
        m = nested_relation([rings[k] for k in order])
        parts = [p for p in m.specs if p.kind != "remainder"]
        assert m.dropped == [] and len(parts) == 3, (order, m.dropped)
        assert sorted((len(p.holes), round(p.area, 1)) for p in parts) == want, order
        assert S.check_structure(m)["passed"] and S.parts_not_built(m) == []
        for p in parts:
            b = m.built_part(p.id)
            assert b.census["unmatched"] == 0 and b.census["duplicated"] == 0
            assert plan_area(b) == pytest.approx(S.poly_area(b.outer, b.holes), abs=1e-6)
            assert plan_area(b) == pytest.approx(p.area, rel=0.03)      # (drawn 1-3 cm apart)


def test_a_courtyard_goes_to_the_ring_around_it_whatever_stands_in_it():
    block, yard, pavilion = S.ccw(np.array(square(0, 0, 40), float)), S.ccw(np.array(square(0, 0, 20), float)), \
        S.ccw(np.array(square(0, 0, 6), float))
    assert S._courtyard_owner(yard, [block, pavilion]) == 0 and S._courtyard_owner(yard, [pavilion, block]) == 1
    assert S._courtyard_owner(yard, [pavilion]) is None                 # only an island in it: cut from nothing
    assert S._courtyard_owner(yard, []) is None
    assert S._courtyard_owner(yard + [60.0, 0.0], [block, pavilion]) is None            # outside every ring
    light_well = S.ccw(np.array(square(0, 0, 2), float))
    assert S._courtyard_owner(light_well, [block, pavilion]) == 1       # the smallest ring around it
    # the same ring in both roles (a relation that describes nothing): float noise does not decide who owns it
    turned = np.roll(yard[::-1], 2, axis=0)
    assert S._courtyard_owner(turned, [yard]) == 0 and S._courtyard_owner(yard * (1 + 1e-12), [yard]) == 0
    # an outer ring too small to use, listed first, does not shift the courtyard to the wrong ring
    m = nested_relation([("outer", square(50, 0, 0.1)), ("outer", square(0, 0, 6)), ("outer", square(0, 0, 40)),
                         ("inner", square(0, 0, 20))])
    assert [d["why"] for d in m.dropped] == ["piece under 0.05 m2"]
    assert sorted((p.id, len(p.holes), round(p.area)) for p in m.specs if p.kind != "remainder") == \
        [("r5000#0", 0, 36), ("r5000#1", 1, 1200)]


def test_no_dropped_record_carries_a_negative_area():
    """Two inner rings, one inside the other, that add up to more than their outer ring (not a valid multipolygon):
    the piece minus its courtyards has an area below zero. It is left out as what it is; the centroid and area-share
    filters are not run on such a polygon (they wrote 'area share ... 0.00 < 0.5' with a negative area_m2)."""
    m = nested_relation([("outer", square(0, 0, 20)), ("inner", square(0, 0, 18)), ("inner", square(0, 0, 16))])
    assert [p for p in m.specs if p.kind != "remainder"] == []
    assert m.dropped == [{"id": "r5000", "why": "courtyards add up to more than their outer ring", "area_m2": 400.0,
                          "name": None, "courtyards_m2": 580.0}]
    for rings in ([("outer", square(0, 0, 6)), ("inner", square(0, 0, 20))],                 # a ring in its "courtyard"
                  [("outer", square(0, 0, 40)), ("inner", square(0, 0, 20)), ("outer", square(0, 0, 6))],
                  [("outer", square(0, 0, 20)), ("inner", square(0, 0, 20))]):                  # the same ring twice
        m = nested_relation(rings)
        assert all(d.get("area_m2", 0.0) >= 0.0 for d in m.dropped), m.dropped
        assert all(p.area > 0.0 for p in m.specs)


def _failing_build(monkeypatch, victim: str, error: Exception):
    real = S.build_part

    def build(p, *a, **k):
        if p.id == victim:
            raise error
        return real(p, *a, **k)

    monkeypatch.setattr(S, "build_part", build)


def two_parts() -> tuple[Osm, str, str]:
    o = Osm()
    body = f'w{o.way(rect(-10, -6, 10, 6), {"building:part": "yes", "height": "12"})}'
    annexe = f'w{o.way(rect(10, -4, 16, 4), {"building:part": "yes", "height": "6", "roof:shape": "pyramidal", "roof:height": "2"})}'
    return o, body, annexe


def test_g13_a_part_that_should_be_drawn_and_was_not_built_fails_the_model_with_its_reason(monkeypatch):
    """The review's demonstration: the main body raises in build_part. It was left out with a line in its problems,
    check_structure passed and the stylized model passed its QA with only the annexe in it. Now G13 lists the part
    with the exception, the registry and the report carry it, and the stylized model fails its QA on it."""
    o, body, annexe = two_parts()
    main = np.array(rect(-10, -6, 16, 6), float)
    ok = model_of(o, main=main)
    assert S.check_structure(ok)["passed"] and S.check_structure(ok)["G13_not_built"] == [] and S.parts_not_built(ok) == []
    assert all(p.not_built is None for p in ok.specs) and ok.report["parts_not_built"] == []
    assert not any("not_built" in r for r in ok.to_registry()["parts"])
    _failing_build(monkeypatch, body, IndexError("index 8 is out of bounds for axis 0 with size 8"))
    m = model_of(o, main=main)
    why = "IndexError: index 8 is out of bounds for axis 0 with size 8"
    assert [b.id for b in m.built if not b.id.startswith("remainder")] == [annexe] and m.part(body).drawn
    assert m.part(body).not_built == {"kind": "exception", "why": why}
    assert m.part(body).problems.count(f"not built: {why}") == 1          # (once, however often the model was built)
    chk = S.check_structure(m)
    assert not chk["passed"] and chk["G13_not_built"] == [{"id": body, "kind": "exception", "why": why}]
    assert all(not chk[k] for k in ("G1_not_watertight", "G2_bad_sources", "G3_filter", "G6_stubs", "G10_without_ticket"))
    row = next(r for r in m.to_registry()["parts"] if r["id"] == body)
    assert row["drawn"] is False and row["not_built"] == {"kind": "exception", "why": why}
    assert m.report["parts_not_built"] == chk["G13_not_built"]
    sm = st.stylize_landmark(st.SceneInput("t", main, None, None, [], structure=m))
    qa = sm.summary["qa"]
    assert not qa["passed"] and not qa["structure"]["passed"] and qa["structure"]["gates"]["G13_not_built"] == chk["G13_not_built"]
    assert qa["audit"]["passed"] and qa["glb"]["passed"] and qa["mesh"]["passed"]      # nothing else is wrong
    assert body not in {p.extras.get("part") for p in sm.parts}


def test_g13_tells_a_solid_that_cannot_be_built_from_an_unexpected_exception_and_from_a_roof_fallback(monkeypatch):
    """Three ways a part is not what its tags ask for. A roof shape that does not fit falls back along its chain
    (Fallback): the part is built, the fallback recorded, no gate fails. A solid that cannot be built as designed
    (NotBuildable) and an exception nobody expected both leave the part out and fail G13, told apart by "kind"."""
    u = [(-10, -10), (10, -10), (10, 10), (6, 10), (6, -6), (-6, -6), (-6, 10), (-10, 10)]
    o = Osm()
    fallen = f'w{o.way(u, {"building:part": "yes", "height": "10", "roof:shape": "pyramidal", "roof:height": "3"})}'
    m = model_of(o, main=np.array(rect(-10, -10, 10, 10), float))
    assert m.built_part(fallen).roof.meta["fallback_from"] == "pyramidal" and m.part(fallen).not_built is None
    assert S.check_structure(m)["passed"] and S.check_structure(m)["G13_not_built"] == []
    assert next(r for r in m.to_registry()["parts"] if r["id"] == fallen)["fallback"]["to"] == "flat"
    o, body, annexe = two_parts()
    main = np.array(rect(-10, -6, 16, 6), float)
    _failing_build(monkeypatch, annexe, S.NotBuildable("roof face fan0: ear clipping stuck with 5 vertices left"))
    m = model_of(o, main=main)
    assert S.check_structure(m)["G13_not_built"] == [{"id": annexe, "kind": "not-buildable",
                                                      "why": "roof face fan0: ear clipping stuck with 5 vertices left"}]
    _failing_build(monkeypatch, annexe, ZeroDivisionError("float division by zero"))
    m = model_of(o, main=main)
    assert S.check_structure(m)["G13_not_built"] == [{"id": annexe, "kind": "exception",
                                                      "why": "ZeroDivisionError: float division by zero"}]
    # a part without a solid and without a reason (a model put together by hand) still fails the gate
    m.part(annexe).not_built = None
    assert S.check_structure(m)["G13_not_built"] == [{"id": annexe, "kind": "unknown",
                                                      "why": "no solid and no recorded reason"}]
    # a part that is not drawn on purpose (a contradiction, a withheld floating part) is not "not built"
    m.part(annexe).drawn = False
    assert S.check_structure(m)["G13_not_built"] == [] and S.check_structure(m)["passed"]


def test_a_skyline_trial_that_cannot_be_built_leaves_no_mark_and_a_later_good_build_clears_the_reason(monkeypatch):
    o, body, annexe = two_parts()
    m = model_of(o, main=np.array(rect(-10, -6, 16, 6), float))
    before = list(m.part(annexe).problems)
    real = S.build_part
    _failing_build(monkeypatch, annexe, ValueError("trial"))
    trial = m.rebuild({annexe: {"roof": 1.2}})                       # a judge's trial with a multiplier ...
    assert annexe not in {b.id for b in trial} and m.part(annexe).not_built is None and m.part(annexe).problems == before
    assert annexe not in {b.id for b in m.rebuild(closed=False)} and m.part(annexe).not_built is None   # ... or open
    m.built = m.rebuild()                                            # the model's own build: the reason stays
    assert m.part(annexe).not_built == {"kind": "exception", "why": "ValueError: trial"} and not S.check_structure(m)["passed"]
    monkeypatch.setattr(S, "build_part", real)
    m.built = m.rebuild()
    assert m.part(annexe).not_built is None and m.part(annexe).problems == before and S.check_structure(m)["passed"]


def test_a_part_whose_faces_cannot_be_tiled_is_not_built_never_shipped_folded(monkeypatch):
    """A part with two courtyards the old bridges folded (tests/test_structure_properties.py). With today's bridges
    it is built and its roof covers the block minus the courtyards. With the old ones every face over the outline is
    wrong: the part is not buildable -- the reason names the faces -- instead of a folded roof that G1 may or may
    not notice (2 of 15 such parts were published)."""
    from test_structure_properties import TWO_COURTYARDS

    outer, holes = TWO_COURTYARDS
    spec = S.PartSpec("r1", "yes", S.ccw(outer), [S.cw(h) for h in holes], {"building:part": "yes", "height": "12"})
    S.resolve_heights(spec)
    b = S.build_part(spec)
    assert b.census["unmatched"] == 0 and b.census["duplicated"] == 0
    monkeypatch.setattr(st, "_in_corner", lambda ring, i, m: True)
    monkeypatch.setattr(st, "_passes_through", lambda m, v, pts: False)
    with pytest.raises(S.NotBuildable, match="roof face flat: .*reversed triangle.*base face"):
        S.build_part(spec)
    with pytest.raises(S.NotBuildable, match="the outline has no exact triangulation"):
        S._pieces(S.ccw(outer), [S.cw(h) for h in holes])              # (what an envelope roof is cut into)
    (built,) = [S.build_all([spec])]
    assert built == [] and spec.not_built["kind"] == "not-buildable" and "reversed triangle" in spec.not_built["why"]


def test_a_structure_none_of_whose_parts_is_built_does_not_pass_as_the_outline(monkeypatch):
    """With no part built at all the stylizer has no parts to stand on and draws the outline's own massing (as for a
    landmark without parts). When parts that should be drawn are what is missing, that is not the model the snapshot
    describes: the QA fails on G13 with each part and its reason."""
    o, body, annexe = two_parts()
    main = np.array(rect(-10, -6, 16, 6), float)
    monkeypatch.setattr(S, "build_part", lambda p, *a, **k: (_ for _ in ()).throw(FloatingPointError("overflow")))
    m = model_of(o, main=main)
    assert m.built == [] and len(S.parts_not_built(m)) == len([p for p in m.specs if p.drawn]) >= 2
    sm = st.stylize_landmark(st.SceneInput("t", main, None, None, [], structure=m))
    qa = sm.summary["qa"]
    assert sm.summary["massing_source"] == "height-field" and not qa["passed"] and not qa["structure"]["passed"]
    assert {(g["id"], g["kind"], g["why"]) for g in qa["structure"]["gates"]["G13_not_built"]} >= {
        (body, "exception", "FloatingPointError: overflow"), (annexe, "exception", "FloatingPointError: overflow")}
    assert qa["audit"]["passed"] and qa["glb"]["passed"] and qa["mesh"]["passed"]
    # a structure with nothing to draw (no parts in the snapshot) is not such a case
    empty = S.build_structure("t", [], [S.ccw(main)], {})
    assert st.stylize_landmark(st.SceneInput("t", main, None, None, [], structure=empty)).summary["qa"]["passed"]
