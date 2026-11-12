"""The catalogue's landmarks, built from their committed OSM parts snapshots (eval/osm/<slug>.parts.osm).

Every clay model the product shows is made from one of these thirteen snapshots, and until D34 only the simplest
(the Baptistery, three parts) was ever built by a test. Each landmark here is loaded as the product loads it
(crowd3d.structure_pipeline.find_source / load_structure), stylized without photos (the OSM-parts clay massing: what a
model whose registration is refused gets) and checked:

- the photo-free structure gates pass (check_structure) and no part that should be drawn is missing (G13);
- every solid's faces are triangulated exactly, face their normals, have no degenerate triangle, and the solid's
  mesh tops out at its registered top;
- the stylized model passes its QA, its GLB's triangles agree with their vertex normals and none is degenerate;
- how many parts are built, which roofs, which fallbacks, how many nodes and triangles the GLB has: equal to
  tests/fixtures/catalogue_structure.json, recorded with this file (`python tests/test_catalogue_structure.py --record`
  rewrites it: do that only for a change that is meant to change a landmark, and say so in the commit).

The five small landmarks run in the fast suite (about three seconds); all thirteen are marked slow.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from crowd3d import structure as S  # noqa: E402
from crowd3d import structure_pipeline as SP  # noqa: E402
from crowd3d import stylize as st  # noqa: E402
from crowd3d.geo import LocalENU  # noqa: E402
from crowd3d.scene_io import load_footprint_rings  # noqa: E402

pytestmark = pytest.mark.structure

CATALOG = ROOT / "eval" / "landmarks.json"
RECORD = Path(__file__).resolve().parent / "fixtures" / "catalogue_structure.json"
ENTRIES = {e["slug"]: e for e in json.loads(CATALOG.read_text(encoding="utf-8")) if e.get("osm_parts")}
FAST = ("florence-baptistery", "india-gate", "arc-de-triomphe", "brandenburg-gate", "lincoln-memorial")


def build(slug: str) -> tuple[S.StructureModel, st.StylizedModel]:
    """The landmark's structure and its stylized model without photos, as the product's stylize step builds them."""
    e = ENTRIES[slug]
    source = SP.find_source(slug, e["lat"], e["lon"])
    model = SP.load_structure(source)
    footprint, holes = load_footprint_rings(source.outline, LocalENU(e["lat"], e["lon"]))
    return model, st.stylize_landmark(st.SceneInput(slug, footprint, None, None, [], holes=tuple(holes), structure=model))


def measure(model: S.StructureModel, stylized: st.StylizedModel) -> dict[str, Any]:
    """What is recorded per landmark (small, and stable against float noise: counts, and two lengths in centimetres)."""
    gates = S.check_structure(model)
    top = max(float(np.asarray(p.positions)[:, 2].max()) for p in stylized.parts)
    return {
        "parts": len(model.specs), "built": len(model.built),
        "not_drawn": sum(1 for p in model.specs if not p.drawn), "not_built": len(gates["G13_not_built"]),
        "withheld": gates["G10_withheld"], "dropped": len(model.dropped),
        "courtyards": sum(len(b.holes) for b in model.built),
        "roofs": dict(sorted(Counter(b.roof.shape for b in model.built).items())),
        "fallbacks": {b.id: f"{b.roof.meta['fallback_from']} -> {b.roof.shape}" for b in model.built
                      if b.roof.meta.get("fallback_from")},
        "solid_triangles": int(sum(len(b.triangles()) for b in model.built)),
        "glb": {"nodes": len(stylized.parts), "triangles": int(sum(len(p.indices) for p in stylized.parts)),
                "roof_area_m2": round(float(stylized.summary["roof_area_m2"])), "top_m": round(top, 2)},
    }


def mesh_problems(model: S.StructureModel, stylized: st.StylizedModel) -> list[str]:
    """Every way a mesh of this landmark is wrong (empty: none)."""
    out = []
    for b in model.built:
        tris = []
        for f in b.faces():
            t, inexact = f.checked_triangles()
            out += [f"{b.id}: {why}" for why in inexact]
            if not len(t):
                continue
            g = np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0])
            n = np.asarray(f.normal, float) / np.linalg.norm(f.normal)
            if (g @ n <= 0).any():
                out.append(f"{b.id}: {int((g @ n <= 0).sum())} triangle(s) of a {f.kind} face look away from its normal")
            if (np.linalg.norm(g, axis=1) < 1e-10).any():
                out.append(f"{b.id}: a degenerate triangle in a {f.kind} face")
            tris.append(t)
        tri = np.concatenate(tris)
        if not np.isfinite(tri).all():
            out.append(f"{b.id}: non-finite coordinates")
        if abs(float(tri[..., 2].max()) - b.top) > 1e-6:
            out.append(f"{b.id}: mesh top {tri[..., 2].max():.6f} m, registered top {b.top:.6f} m")
        if b.census.get("unmatched", 1) or b.census.get("duplicated", 1):
            out.append(f"{b.id}: not closed {b.census}")
    for p in stylized.parts:
        pos, idx, nrm = np.asarray(p.positions, float), np.asarray(p.indices).reshape(-1, 3), np.asarray(p.normals, float)
        tri = pos[idx]
        g = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        area = 0.5 * np.linalg.norm(g, axis=1)
        if (area < 1e-8).any():
            out.append(f"GLB node {p.name}: {int((area < 1e-8).sum())} degenerate triangle(s)")
        flipped = ((g * nrm[idx].mean(axis=1)).sum(axis=1) < -1e-9) & (area >= 1e-8)
        if flipped.any():
            out.append(f"GLB node {p.name}: {int(flipped.sum())} triangle(s) against their vertex normals")
    return out


def check(slug: str) -> None:
    model, stylized = build(slug)
    gates = S.check_structure(model)
    assert gates["passed"] and gates["G13_not_built"] == [] and gates["G1_not_watertight"] == [], gates
    assert all(p.not_built is None for p in model.specs) and model.report["parts_not_built"] == []
    assert not any(x.startswith("not built") for p in model.specs for x in p.problems)
    qa = stylized.summary["qa"]
    assert qa["passed"] and qa["structure"]["passed"] and qa["mesh"] == {"passed": True, "inexact_triangulations": []}, \
        SP.qa_failure_detail(qa)
    assert stylized.summary["massing_source"] == "osm-parts"
    assert {p.extras.get("part") for p in stylized.parts if p.extras.get("part")} == {b.id for b in model.built}
    assert mesh_problems(model, stylized) == []
    recorded = json.loads(RECORD.read_text(encoding="utf-8"))
    assert measure(model, stylized) == recorded[slug], slug


def test_the_record_lists_the_catalogue():
    recorded = json.loads(RECORD.read_text(encoding="utf-8"))
    assert sorted(recorded) == sorted(ENTRIES) and len(ENTRIES) == 13 and set(FAST) <= set(ENTRIES)
    assert sum(r["built"] for r in recorded.values()) == 824 and all(r["not_built"] == 0 for r in recorded.values())
    assert sum(r["courtyards"] for r in recorded.values()) > 0 and sum(len(r["fallbacks"]) for r in recorded.values()) == 7


@pytest.mark.parametrize("slug", FAST)
def test_a_small_catalogue_landmark_builds_as_recorded(slug):
    check(slug)


@pytest.mark.slow
@pytest.mark.parametrize("slug", sorted(ENTRIES))
def test_every_catalogue_landmark_builds_as_recorded(slug):
    check(slug)


if __name__ == "__main__":
    if sys.argv[1:] != ["--record"]:
        raise SystemExit(__doc__)
    record = {}
    for name in sorted(ENTRIES):
        m, sm = build(name)
        bad = mesh_problems(m, sm)
        if bad or not sm.summary["qa"]["passed"]:
            raise SystemExit(f"{name}: not recorded, it does not pass: {bad[:3]} {SP.qa_failure_detail(sm.summary['qa'])}")
        record[name] = measure(m, sm)
        print(name, json.dumps(record[name], ensure_ascii=False))
    RECORD.write_text(json.dumps(record, indent=1, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {RECORD}")
