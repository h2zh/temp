"""scripts/trim_osm_parts.py and the committed parts snapshots (eval/osm/<slug>.parts.osm, eval/landmarks.json)."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from crowd3d import structure as S
from crowd3d.geo import LocalENU

ROOT = Path(__file__).resolve().parent.parent
LAT, LON = 45.0, 9.0
ENU = LocalENU(LAT, LON)
META = 'version="1" changeset="77" user="someone" uid="42" timestamp="{t}"'


@pytest.fixture(scope="module")
def trim():
    spec = importlib.util.spec_from_file_location("trim_osm_parts", ROOT / "scripts" / "trim_osm_parts.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def full_snapshot() -> bytes:
    """A bbox download: the main way, a part way, a two-way part relation, a neighbour building and a tagged node."""
    nodes, ways, n = [], [], [0]

    def way(wid, ring, tags, t="2020-01-01T00:00:00Z"):
        ids = []
        for x, y in ring:
            n[0] += 1
            la, lo, _ = ENU.to_geodetic(x, y, 0.0)
            nodes.append(f'<node id="{n[0]}" {META.format(t=t)} lat="{la:.9f}" lon="{lo:.9f}"/>')
            ids.append(str(n[0]))
        nds = "".join(f'<nd ref="{i}"/>' for i in ids + [ids[0]])
        tg = "".join(f'<tag k="{k}" v="{v}"/>' for k, v in tags.items())
        ways.append(f'<way id="{wid}" {META.format(t=t)}>{nds}{tg}</way>')

    rect = lambda x0, y0, x1, y1: [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]     # noqa: E731
    way(11, rect(-10.5, -5.5, 10.5, 5.5), {"building": "yes", "height": "10"})
    way(10, rect(-10, -5, 0, 5), {"building:part": "yes", "height": "10", "roof:shape": "gabled"})
    way(20, rect(0, -5, 10, 5), {})
    way(30, rect(60, 60, 70, 70), {"building": "yes"}, t="2024-05-05T00:00:00Z")    # outside: the newest element
    nodes.append(f'<node id="999" {META.format(t="2021-01-01T00:00:00Z")} lat="45.001" lon="9.001">'
                 '<tag k="amenity" v="bench"/></node>')
    rel = (f'<relation id="5000" {META.format(t="2020-01-01T00:00:00Z")}><member type="way" ref="20" role="outer"/>'
           '<tag k="type" v="multipolygon"/><tag k="building:part" v="yes"/><tag k="height" v="6"/></relation>')
    return ('<?xml version="1.0" encoding="UTF-8"?>\n<osm version="0.6" generator="test" '
            'copyright="OpenStreetMap and contributors" license="http://opendatacommons.org/licenses/odbl/1-0/">\n'
            '<bounds minlat="44.99" minlon="8.99" maxlat="45.01" maxlon="9.01"/>\n'
            + "\n".join(nodes + ways + [rel]) + "\n</osm>\n").encode()


@pytest.fixture()
def outline(tmp_path):
    ring = [list(ENU.to_geodetic(x, y, 0.0)[:2])[::-1] for x, y in
            [(-10.5, -5.5), (10.5, -5.5), (10.5, 5.5), (-10.5, 5.5)]]
    gj = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {"osm_type": "way", "osm_id": 11},
                                                     "geometry": {"type": "Polygon", "coordinates": [ring + [ring[0]]]}}]}
    p = tmp_path / "x.geojson"
    p.write_text(json.dumps(gj))
    return p


def test_trim_keeps_the_structure_and_drops_the_rest(trim, outline):
    full = full_snapshot()
    data = trim.trim(full, "way", "11", "2026-10-22")
    a, b = (trim.structure_of(d, outline, LAT, LON, "x") for d in (full, data))
    assert trim.differences(a, b) == []
    assert sorted(a["triangles"]) == ["r5000", "w10"]
    text = data.decode()
    assert "user=" not in text and "uid=" not in text and "changeset=" not in text and "amenity" not in text
    assert 'id="30"' not in text and 'id="999"' not in text and "<bounds" in text and 'license="http' in text
    meta = S.parse_osm(data).meta
    assert meta["newest_element"] == S.parse_osm(full).meta["newest_element"] == "2024-05-05T00:00:00Z"
    assert meta["retrieved"] == "2026-10-22" and meta["extract"] == trim.EXTRACT
    assert set(S.parse_osm(full).meta) == {"file", "generator", "api_version", "newest_element", "licence"}


def test_a_lossy_extract_is_seen(trim, outline):
    full = full_snapshot()
    lossy = trim.trim(full, "way", "11", "2026-10-22").replace(b'<tag k="building:part" v="yes" />', b"", 1)
    diff = trim.differences(trim.structure_of(full, outline, LAT, LON, "x"),
                            trim.structure_of(lossy, outline, LAT, LON, "x"))
    assert diff and any(d.startswith("registry") for d in diff)
    with pytest.raises(ValueError):
        trim.trim(b'<?xml version="1.0"?><!DOCTYPE osm [<!ENTITY a "x">]><osm>&a;</osm>', "way", "1", "2026-10-22")


def test_main_writes_the_snapshot_and_the_catalog_block_or_nothing(trim, outline, tmp_path, monkeypatch):
    snaps, out = tmp_path / "snaps", tmp_path / "osm"
    snaps.mkdir()
    out.mkdir()
    (snaps / "x.osm").write_bytes(full_snapshot())
    catalog = tmp_path / "landmarks.json"
    entry = {"slug": "x", "lat": LAT, "lon": LON, "osm": {"file": str(outline)}}
    catalog.write_text(json.dumps([entry, {"slug": "y", "lat": 1.0, "lon": 2.0}]))
    args = ["--snapshots", str(snaps), "--retrieved", "2026-10-22", "--catalog", str(catalog), "--out", str(out)]
    real_trim = trim.trim
    monkeypatch.setattr(trim, "trim", lambda *a: real_trim(*a).replace(b'k="height" v="6"', b'k="height" v="7"'))
    assert trim.main(args) == 1
    assert not list(out.iterdir()) and "osm_parts" not in catalog.read_text()
    monkeypatch.setattr(trim, "trim", real_trim)
    assert trim.main(args) == 0
    block = json.loads(catalog.read_text())[0]["osm_parts"]
    data = (out / "x.parts.osm").read_bytes()
    assert block["sha256"] == hashlib.sha256(data).hexdigest() and block["file"] == "eval/osm/x.parts.osm"
    assert block["retrieved"] == "2026-10-22" and block["newest_element"] == "2024-05-05T00:00:00Z"
    assert block["licence"] == "ODbL-1.0" and "OpenStreetMap contributors" in block["attribution"]
    assert "osm_parts" not in json.loads(catalog.read_text())[1]


def test_committed_snapshots_match_the_catalog():
    """Every parts snapshot in eval/osm is in the catalog with its sha256, and its metadata is the full download's."""
    catalog = json.loads((ROOT / "eval" / "landmarks.json").read_text(encoding="utf-8"))
    blocks = {e["slug"]: e["osm_parts"] for e in catalog if e.get("osm_parts")}
    assert sorted(blocks) == sorted(p.name[:-len(".parts.osm")] for p in (ROOT / "eval" / "osm").glob("*.parts.osm"))
    assert len(blocks) == 13
    for slug, block in blocks.items():
        data = (ROOT / block["file"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == block["sha256"], slug
        meta = S.parse_osm(data).meta
        assert (meta["newest_element"], meta["retrieved"]) == (block["newest_element"], block["retrieved"]), slug
        assert block["licence"] == "ODbL-1.0" and "ODbL" in block["attribution"]
        assert b"user=" not in data and b"changeset=" not in data
