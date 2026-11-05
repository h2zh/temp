"""End to end through the HTTP API with real COLMAP: a published model gets its stylized GLB (structure in the product).

The synthetic box (crowd3d.synthetic: 20 x 14 x 12 m) is the landmark; a catalogue in the test's own folder lists an
OSM parts snapshot of it (one flat building:part plus the main building way) and its outline, under the landmark's
id. Two sides are uploaded in one request with the background worker on: both facade jobs reconstruct first, then
each published model's stylize job runs in its child process (crowd3d.structure_pipeline), and the landmark page's
models carry the GLB the server serves next to the PLY. The uploader agrees to the user agreement (D32), so every
accepted photo is shared, and then withdraws two: the public poses (D30) list the others, in the GLB's frame, and
neither they nor the GLB carry a store id.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from crowd3d import photoposes as pp
from crowd3d import stylize as st
from crowd3d.api import create_app
from crowd3d.config import Settings
from crowd3d.consent import deployment_salt
from crowd3d.geo import LocalENU
from crowd3d.records import JOB_DONE, JOB_STYLIZE, PHOTO_ACCEPTED, TERMS_CURRENT
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
        r = client.post("/api/photos", data={"user_id": "alice", "landmark_id": LANDMARK_ID, "terms": TERMS_CURRENT},
                        files=files)
        assert r.status_code == 200, r.text
        uploads = r.json()["results"]
        assert app.state.runner.drain(timeout=1200)
        # the uploader withdraws two photos (one per side)
        withdrawn = [uploads[0], uploads[-1]]
        for res in withdrawn:
            w = client.post(f"/api/photos/{res['photo_id']}/share_location",
                            json={"share": False, "token": res["manage_token"]})
            assert w.status_code == 200, w.text
        detail = client.get(f"/api/landmarks/{LANDMARK_ID}").json()
        jobs = app.state.store.list_jobs(LANDMARK_ID)
        models = {m.id: m for m in app.state.store.list_models(LANDMARK_ID)}
        files = {m["id"]: {suffix: client.get(m["url"].replace(".ply", suffix)) for suffix in
                           (".glb", ".json", ".style.json")} for m in detail["models"]}
        poses = client.get(f"/api/landmarks/{LANDMARK_ID}/poses")
        assert poses.status_code == 200, poses.text
        photos = app.state.store.list_photos(LANDMARK_ID)
        yield settings, detail, jobs, models, files, {"uploads": uploads, "withdrawn": withdrawn, "poses": poses,
                                                      "photos": photos}


def test_facades_reconstruct_first_then_each_published_model_is_stylized(run):
    settings, detail, jobs, models, _, _ = run
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


def test_every_shown_model_links_its_served_glb_with_the_registration(run):
    settings, detail, jobs, models, files, _ = run
    for m in detail["models"]:
        styl = m["stylized"]
        assert styl["status"] == "ready" and styl["url"] == m["url"].replace(".ply", ".glb")
        assert isinstance(styl["registration"]["accepted"], bool) and "label" not in styl   # the client words it (D32)
        got = files[m["id"]]
        assert got[".glb"].status_code == 200 and got[".glb"].content[:4] == b"glTF"
        assert got[".json"].status_code == 404 and got[".style.json"].status_code == 404   # not served
        rec = models[m["id"]].stylized
        assert styl["snapped"] is pp.snap_accepted(rec.get("snap"))    # the snap the job recorded, accepted or not
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
            assert summary["observed_walls"] == 0                     # refused: no wall counts as photographed


def test_the_public_poses_show_the_shared_photos_in_the_glb_frame_and_nothing_carries_a_store_id(run):
    """D30 with real COLMAP: the stylize record keeps the snap the GLB was built on (the same as its summary); the
    public poses list the photos still shared, each exactly public_row of its internal pose with that snap applied (sfm
    rows with the pitch, roll and field of view the model measured); the two withdrawn photos and every store id stay
    out of the poses, the served GLBs (their JSON chunk) and the detail."""
    settings, detail, jobs, models, files, extra = run
    salt = deployment_salt(settings.data_dir)
    published = [m for m in models.values() if m.meta["published"]]
    snaps = {}
    for m in published:
        rec = m.stylized
        summary = json.loads((settings.data_dir / m.ply_path).with_suffix(".style.json").read_text(encoding="utf-8"))
        assert "snap" in rec and rec["snap"] == summary["snap"]
        if pp.snap_accepted(rec["snap"]):
            snaps[m.id] = rec["snap"]
    photos = [p for p in extra["photos"] if p.status == PHOTO_ACCEPTED]
    withdrawn = {res["photo_id"] for res in extra["withdrawn"]}
    shared = [p for p in photos if p.id not in withdrawn]
    assert len(shared) == len(photos) - 2 and all(p.location_consent == TERMS_CURRENT for p in shared)
    internal = pp.photo_poses(shared, published, detail["lat"], detail["lon"], snaps=snaps,
                              aspects=pp.processed_aspects(shared, settings.data_dir))
    expected = sorted((r for p in internal if (r := pp.public_row(p, salt)) is not None), key=lambda r: r["id"])
    body = extra["poses"].json()
    assert body["photos"] == expected and len(expected) >= 4
    assert body["counts"]["shared"] == len(shared) and body["counts"]["photos"] == len(photos)
    sfm = [r for r in body["photos"] if r["src"] == "sfm"]
    assert sfm and all(r["pitch"] is not None and r["hfov"] is not None and r["aspect"] == 1.33 for r in sfm)
    pub = {res["photo_id"]: res["public_id"] for res in extra["uploads"]}
    assert {r["id"] for r in body["photos"]} <= {pub[p.id] for p in shared}
    assert not {pub[pid] for pid in withdrawn} & {r["id"] for r in body["photos"]}
    store_ids = {p.id for p in extra["photos"]}
    texts = {"poses": extra["poses"].text, "detail": json.dumps(detail)}
    for m in detail["models"]:
        gltf, _ = st.read_glb(files[m["id"]][".glb"].content)
        texts[m["id"]] = json.dumps(gltf)
    for name, text in texts.items():
        assert not any(pid in text for pid in store_ids), name
        assert not store_ids & set(re.findall(r"[0-9a-f]{32}", text)), name
        if name != "poses":
            assert not any(i in text for i in pub.values() if i), name        # nor a public id
    # nor a camera's exact position (D30 privacy item 9): no GLB JSON chunk prints any camera of any model, the two
    # withdrawn photos' included, to 0.1 m side by side in either order, before or after its model's snap
    cams = []
    for m in models.values():
        for pid, cam in (m.meta.get("cameras") or {}).items():
            cams.append((pid, *cam["enu"][:2]))
            if m.id in snaps:
                moved, _ = pp.apply_snap(snaps[m.id], cam["enu"], 0.0)
                cams.append((pid, *moved[:2]))
    assert withdrawn <= {pid for pid, _, _ in cams}
    for m in detail["models"]:
        nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?",
                                             re.sub(r"[0-9a-f]{8,}", " ", texts[m["id"]]))]
        pairs = list(zip(nums, nums[1:]))
        for pid, e, n in cams:
            assert not any(abs(a - e) <= 0.1 and abs(b - n) <= 0.1 or abs(a - n) <= 0.1 and abs(b - e) <= 0.1
                           for a, b in pairs), (m["id"], pid, e, n)
