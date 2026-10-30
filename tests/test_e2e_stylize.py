"""End to end through the HTTP API with real COLMAP: a published model gets its stylized GLB (structure in the product).

The synthetic box (crowd3d.synthetic: 20 x 14 x 12 m) is the landmark; a catalogue in the test's own folder lists an
OSM parts snapshot of it (one flat building:part plus the main building way) and its outline, under the landmark's
id. Two sides are uploaded in one request with the background worker on: both facade jobs reconstruct first, then
each published model's stylize job runs in its child process (crowd3d.structure_pipeline), and the landmark page's
models carry the GLB the server serves next to the PLY.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crowd3d import stylize as st
from crowd3d.api import STYLIZED_LABEL_ALIGNED, STYLIZED_LABEL_CLAY, create_app
from crowd3d.config import Settings
from crowd3d.geo import LocalENU
from crowd3d.records import JOB_DONE, JOB_STYLIZE
from crowd3d.synthetic import BuildingSpec, generate_dataset

pytestmark = pytest.mark.slow

LANDMARK_ID = "e2e-box"
SIZE = (640, 480)


def write_box_catalog(folder: Path, lat: float, lon: float, building: BuildingSpec) -> Path:
    """eval/landmarks.json-shaped catalogue with the box's OSM parts snapshot and outline (paths relative to folder,
    which plays the repository root)."""
    enu = LocalENU(lat, lon)
    hx, hy = building.width_e_m / 2, building.depth_n_m / 2
    ring = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    geo = [enu.to_geodetic(x, y, 0.0)[:2] for x, y in ring]
    nodes = "".join(f'<node id="{i + 1}" lat="{la:.9f}" lon="{lo:.9f}"/>' for i, (la, lo) in enumerate(geo))
    nds = "".join(f'<nd ref="{i}"/>' for i in (1, 2, 3, 4, 1))
    part = f'<tag k="building:part" v="yes"/><tag k="height" v="{building.height_m:g}"/><tag k="roof:shape" v="flat"/>'
    main = f'<tag k="building" v="yes"/><tag k="height" v="{building.height_m:g}"/>'
    osm = folder / "eval" / "osm" / f"{LANDMARK_ID}.parts.osm"
    osm.parent.mkdir(parents=True)
    osm.write_text(f'<?xml version="1.0"?><osm version="0.6" generator="test">{nodes}'
                   f'<way id="10">{nds}{part}</way><way id="11">{nds}{main}</way></osm>', encoding="utf-8")
    outline = osm.with_name(f"{LANDMARK_ID}.geojson")
    coords = [[lo, la] for la, lo in geo + geo[:1]]
    outline.write_text(json.dumps({"type": "FeatureCollection", "features": [{
        "type": "Feature", "properties": {"osm_type": "way", "osm_id": 11, "height": f"{building.height_m:g}"},
        "geometry": {"type": "Polygon", "coordinates": [coords]}}]}), encoding="utf-8")
    import hashlib

    catalog = folder / "eval" / "landmarks.json"
    catalog.write_text(json.dumps([{
        "slug": LANDMARK_ID, "name": "Synthetic box", "lat": lat, "lon": lon, "radius_m": 150,
        "osm": {"file": f"eval/osm/{outline.name}", "attribution": "test outline"},
        "osm_parts": {"file": f"eval/osm/{osm.name}", "sha256": hashlib.sha256(osm.read_bytes()).hexdigest(),
                      "attribution": "test parts"}}]), encoding="utf-8")
    return catalog


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e-stylize")
    building = BuildingSpec()
    manifest = generate_dataset(root / "photos", size=SIZE, sectors=[3, 4], photos_per_sector=4, include_bad=False,
                                seed=0, building=building)
    lm = manifest["landmark"]
    catalog = write_box_catalog(root / "repo", lm["lat"], lm["lon"], building)
    settings = Settings(data_dir=root / "data", reconstructor="colmap", inline_jobs=False, structure_catalog=catalog)
    app = create_app(settings)
    with TestClient(app) as client:
        r = client.post("/api/landmarks", json={"id": LANDMARK_ID, "name": lm["name"], "lat": lm["lat"],
                                                "lon": lm["lon"], "radius_m": 150})
        assert r.status_code == 201, r.text
        good = manifest["photos"]
        files = [("files", (e["file"], (root / "photos" / e["file"]).read_bytes(), "image/jpeg")) for e in good]
        r = client.post("/api/photos", data={"user_id": "alice", "landmark_id": LANDMARK_ID}, files=files)
        assert r.status_code == 200, r.text
        assert app.state.runner.drain(timeout=1200)
        detail = client.get(f"/api/landmarks/{LANDMARK_ID}").json()
        jobs = app.state.store.list_jobs(LANDMARK_ID)
        models = {m.id: m for m in app.state.store.list_models(LANDMARK_ID)}
        files = {m["id"]: {suffix: client.get(m["url"].replace(".ply", suffix)) for suffix in
                           (".glb", ".json", ".style.json")} for m in detail["models"]}
        yield settings, detail, jobs, models, files


def test_facades_reconstruct_first_then_each_published_model_is_stylized(run):
    settings, detail, jobs, models, _ = run
    recon = [j for j in jobs if j.kind != JOB_STYLIZE]
    styl = [j for j in jobs if j.kind == JOB_STYLIZE]
    assert {j.status for j in jobs} == {JOB_DONE}, [(j.kind, j.status, j.error) for j in jobs]
    assert len(detail["models"]) >= 2 and len(styl) == len([m for m in models.values() if m.meta["published"]])
    # a stylize job starts only once no reconstruction job is queued: every reconstruction ran first
    assert max(j.started_at for j in recon) < min(j.started_at for j in styl)
    assert {j.model_id for j in styl} == {m.id for m in models.values() if m.meta["published"]}
    for j in styl:
        assert j.error is None and (j.sector, sorted(j.photo_ids)) == (
            models[j.model_id].sector, sorted(next(r for r in recon if r.id == models[j.model_id].job_id).photo_ids))
    assert not any(settings.work_dir.iterdir())                       # the stylize work dirs are gone too


def test_every_shown_model_links_its_served_glb_with_the_registration_label(run):
    settings, detail, jobs, models, files = run
    for m in detail["models"]:
        styl = m["stylized"]
        assert styl["status"] == "ready" and styl["url"] == m["url"].replace(".ply", ".glb")
        assert styl["label"] == (STYLIZED_LABEL_ALIGNED if styl["registration"]["accepted"] else STYLIZED_LABEL_CLAY)
        got = files[m["id"]]
        assert got[".glb"].status_code == 200 and got[".glb"].content[:4] == b"glTF"
        assert got[".json"].status_code == 404 and got[".style.json"].status_code == 404   # not served
        rec = models[m["id"]].stylized
        assert rec["massing_source"] == "osm-parts" and rec["parts_built"] == 1
        assert rec["source"]["slug"] == LANDMARK_ID and rec["within_limits"]
        # the GLB, its summary beside the PLY (unserved) and the record say the same
        ply = settings.data_dir / models[m["id"]].ply_path
        summary = json.loads(ply.with_suffix(".style.json").read_text(encoding="utf-8"))
        assert summary["registration"]["accepted"] == styl["registration"]["accepted"]
        assert summary["structure_run"]["source"]["slug"] == LANDMARK_ID and summary["qa"]["passed"]
        gltf, _ = st.read_glb(got[".glb"].content)
        extras = gltf["scenes"][0]["extras"]
        assert "test parts" in json.dumps(extras.get("attribution")) or "test parts" in json.dumps(extras)
        if not styl["registration"]["accepted"]:
            assert summary["observed_walls"] == 0                     # all clay: nothing from the photos
