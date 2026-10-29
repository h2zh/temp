"""Maintainers: trim full OSM map snapshots to the landmark's parts and record them in the catalog.

    python -I scripts/trim_osm_parts.py --snapshots <dir with <slug>.osm> --retrieved 2026-10-22 [--slugs ...]

A snapshot is an OpenStreetMap API 0.6 map call (a bounding-box download around the landmark). Only what
crowd3d.structure.load_osm_structure reads is kept: the bounds, every building:part way and relation, the main
element named by the landmark's outline (eval/osm/<slug>.geojson), their member ways and those ways' nodes.
The user, uid and changeset attributes and node tags are dropped. The full snapshot's newest element timestamp,
the retrieval date and the extract rule go on the <osm> root, where parse_osm reads them back, so the snapshot
metadata stays that of the full download. Nothing is edited: kept elements keep their ids, versions,
timestamps, coordinates and tags.

A trimmed file is written to eval/osm/<slug>.parts.osm only when it builds exactly the structure the full file
builds (registry, report, dropped parts, main tags and every part's triangles); otherwise nothing is written.
eval/landmarks.json then gets an osm_parts block per landmark (file, sha256, source, retrieval date, newest
element, licence and the ODbL attribution).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crowd3d import structure as S  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
DROP_ATTRS = ("user", "uid", "changeset")
EXTRACT = ("building:part ways and relations, the main element, their member ways and nodes; "
           "user, uid, changeset and node tags dropped (scripts/trim_osm_parts.py)")
SOURCE = "OpenStreetMap API 0.6 map call (bounding box around the landmark)"
ATTRIBUTION = ("© OpenStreetMap contributors. Data available under the Open Database Licence (ODbL 1.0), "
               "https://www.openstreetmap.org/copyright")


def trim(data: bytes, main_type: str, main_id: str, retrieved: str) -> bytes:
    """The parts extract of one full snapshot (see the module docstring)."""
    meta = S.parse_osm(data).meta                      # refuses a DTD; the full file's newest element
    root = ET.fromstring(data)
    ways = {w.get("id"): w for w in root.iter("way")}
    keep_w: set[str] = set()
    keep_r: set[str] = set()
    for r in root.iter("relation"):
        if "building:part" in S._tags(r) or (main_type == "relation" and r.get("id") == main_id):
            keep_r.add(r.get("id"))
            keep_w |= {m.get("ref") for m in r.findall("member") if m.get("type") == "way"}
    for wid, w in ways.items():
        if "building:part" in S._tags(w) or (main_type == "way" and wid == main_id):
            keep_w.add(wid)
    keep_n = {nd.get("ref") for wid in keep_w if wid in ways for nd in ways[wid].findall("nd")}
    attrs = {**root.attrib, "newest_element": meta["newest_element"] or "", "retrieved": retrieved,
             "extract": EXTRACT}
    out = ET.Element("osm", attrs)
    out.text = root.text
    for el in root:
        keep = (el.tag == "bounds" or (el.tag == "node" and el.get("id") in keep_n)
                or (el.tag == "way" and el.get("id") in keep_w)
                or (el.tag == "relation" and el.get("id") in keep_r))
        if not keep:
            continue
        for a in DROP_ATTRS:
            el.attrib.pop(a, None)
        if el.tag == "node":
            for t in list(el):
                el.remove(t)
        out.append(el)
    if len(out):
        out[-1].tail = "\n"
    return ET.tostring(out, encoding="utf-8", xml_declaration=True) + b"\n"


def structure_of(data: bytes, outline: Path, lat: float, lon: float, slug: str) -> dict[str, Any]:
    """Everything load_osm_structure builds from a snapshot, JSON-comparable (the build time left out)."""
    m = S.load_osm_structure(data, outline, lat, lon, slug)
    return json.loads(json.dumps({
        "registry": m.to_registry(), "report": {k: v for k, v in m.report.items() if k != "build_s"},
        "dropped": m.dropped, "main_tags": m.main_tags,
        "triangles": {b.id: hashlib.sha256(b.triangles().tobytes()).hexdigest() for b in m.built},
    }, sort_keys=True, default=str))


def differences(full: dict[str, Any], trimmed: dict[str, Any]) -> list[str]:
    """The sections (and their keys) where two structure_of results differ. The trimmed snapshot's metadata may
    add its own keys (retrieved, extract) to sources.osm and nothing else."""
    osm = trimmed["registry"]["sources"]["osm"]
    trimmed["registry"]["sources"]["osm"] = {k: v for k, v in osm.items() if k not in ("retrieved", "extract")}
    out = []
    for section, a in full.items():
        b = trimmed[section]
        if a != b:
            keys = [k for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)] if isinstance(a, dict) else []
            out.append(f"{section} {keys}" if keys else section)
    return out


def catalog_block(slug: str, data: bytes, retrieved: str) -> dict[str, Any]:
    meta = S.parse_osm(data).meta
    return {"file": f"eval/osm/{slug}.parts.osm", "sha256": hashlib.sha256(data).hexdigest(), "source": SOURCE,
            "retrieved": retrieved, "newest_element": meta["newest_element"], "extract": EXTRACT,
            "licence": "ODbL-1.0", "attribution": ATTRIBUTION}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -I scripts/trim_osm_parts.py", description=__doc__.split("\n")[0])
    ap.add_argument("--snapshots", type=Path, required=True, help="directory of full snapshots <slug>.osm")
    ap.add_argument("--retrieved", required=True, help="date the snapshots were downloaded (YYYY-MM-DD)")
    ap.add_argument("--slugs", nargs="*", help="only these landmarks (default every catalog entry with a snapshot)")
    ap.add_argument("--catalog", type=Path, default=REPO_ROOT / "eval" / "landmarks.json")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "eval" / "osm")
    args = ap.parse_args(argv)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    todo = [e for e in catalog if (args.snapshots / f"{e['slug']}.osm").is_file()
            and (not args.slugs or e["slug"] in args.slugs)]
    if not todo:
        raise SystemExit(f"no catalog landmark has a snapshot in {args.snapshots}")
    results, failed = {}, []
    for e in todo:
        slug = e["slug"]
        outline = REPO_ROOT / e["osm"]["file"]
        props = json.loads(outline.read_text(encoding="utf-8"))["features"][0].get("properties", {})
        full = (args.snapshots / f"{slug}.osm").read_bytes()
        data = trim(full, str(props.get("osm_type", "")), str(props.get("osm_id", "")), args.retrieved)
        a = structure_of(full, outline, e["lat"], e["lon"], slug)
        diff = differences(a, structure_of(data, outline, e["lat"], e["lon"], slug))
        print(f"{slug:24s} parts built {a['report']['parts_built']:3d}  full {len(full) / 1024:6.0f} KB  "
              f"trimmed {len(data) / 1024:5.0f} KB  {'same structure' if not diff else 'DIFFERENT: ' + '; '.join(diff)}")
        if diff:
            failed.append(slug)
        results[slug] = data
    if failed:
        print(f"refused: the trimmed snapshot of {failed} builds another structure; nothing written", file=sys.stderr)
        return 1
    for e in todo:
        (args.out / f"{e['slug']}.parts.osm").write_bytes(results[e["slug"]])
        e["osm_parts"] = catalog_block(e["slug"], results[e["slug"]], args.retrieved)
    args.catalog.write_text(json.dumps(catalog, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {len(todo)} snapshots to {args.out} and their osm_parts blocks to {args.catalog}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
