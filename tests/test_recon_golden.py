"""RECON_OUTPUT_VERSION's golden test: what the product records for fixed uploads and fixed SfM output.

The synthetic facade photos go the product's way: crowd3d.ingest.ingest_upload (validation, the processed copy,
the EXIF fields of the PhotoRecord, focal_35mm included), JobRunner.schedule (coverage planning, the job's photo
order) and JobRunner.run_job (ReconPhoto inputs, ColmapReconstructor, the QA gate, the model JSON). COLMAP's own
result depends on thread timing (test_reconstruct.py), so its feature, matching and mapping calls are replaced by a
recorder that hands back the sparse model one real run made of these photos (fixtures/recon_golden/sparse); the
database the mapper would read (cameras with their focal priors) and the options of every call are recorded.
Everything is compared with fixtures/recon_golden/golden.json:
- the model JSON keeps every earlier key and value exactly; keys it gains (a purely additive change) need no bump,
  only a regeneration that pins them (CROWD3D_REGEN_GOLDEN=1, allowed while the earlier values are unchanged);
- the GeoAnchor's calibration is applied, not reconstructed: with a new Params.anchor_calibration_id its "sigma" and
  "calibration" blocks may change (each sigma must still be its raw core with the new floors in quadrature) and a
  regeneration pins them without a bump. Floors are refitted offline from the stored cores (docs/BENCHMARK.md,
  calibration) and rescores apply the current floors (GeoAnchor.recalibrated), so a refit never mixes calibrations;
- any other difference means the reconstruction output can change: bump crowd3d.reconstruct.RECON_OUTPUT_VERSION,
  then regenerate (refused while the version is unchanged, unless the uploads themselves changed).
Benchmark rescores rely on the version (scripts/benchmark.py rescore).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import crowd3d.ingest as ingest_module
from crowd3d import synthetic as syn
from crowd3d.anchor import floored_sigma
from crowd3d.config import Settings
from crowd3d.ingest import ingest_upload
from crowd3d.jobs import JobRunner
from crowd3d.reconstruct import RECON_OUTPUT_VERSION, get_reconstructor
from crowd3d.records import JOB_DONE, Landmark
from crowd3d.store import Store

FIXTURE = Path(__file__).parent / "fixtures" / "recon_golden"
REGEN = os.environ.get("CROWD3D_REGEN_GOLDEN") == "1"
LANDMARK = "golden"


def _canon(o: Any) -> Any:
    """JSON-equal form, floats to 6 decimals (immune to last-bit differences between linear algebra builds)."""
    if isinstance(o, float):
        return round(o, 6)
    if isinstance(o, dict):
        return {str(k): _canon(v) for k, v in o.items()}
    if isinstance(o, list | tuple):
        return [_canon(v) for v in o]
    return o


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def legacy_diff(old: Any, new: Any, path: str = "") -> list[str]:
    """Where new does not keep old: a dict may gain keys, every key and value old has must be equal (recursively,
    through lists of the same length)."""
    if isinstance(old, dict) and isinstance(new, dict):
        return [d for k, v in old.items() for d in (legacy_diff(v, new[k], f"{path}.{k}") if k in new
                                                    else [f"{path}.{k} (gone)"])]
    if isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        return [d for i, (a, b) in enumerate(zip(old, new)) for d in legacy_diff(a, b, f"{path}[{i}]")]
    return [] if old == new else [path or "(value)"]


def calibration_free(meta: dict[str, Any] | None, other: dict[str, Any] | None) -> dict[str, Any] | None:
    """meta without its anchor's "sigma" and "calibration" blocks when other's anchor has another calibration id (the
    module docstring: a recalibration keeps RECON_OUTPUT_VERSION)."""
    a, b = (meta or {}).get("anchor"), (other or {}).get("anchor")
    if not a or not b or (a.get("calibration") or {}).get("id") == (b.get("calibration") or {}).get("id"):
        return meta
    return {**meta, "anchor": {k: v for k, v in a.items() if k not in ("sigma", "calibration")}}


def test_legacy_diff_allows_added_keys_only():
    old = {"a": 1, "cams": {"p": {"enu": [1, 2], "flags": []}}, "qa": [{"x": 1}]}
    assert legacy_diff(old, {**old, "anchor": {"yaw": 1}}) == []
    assert legacy_diff(old, {**old, "cams": {"p": {"enu": [1, 2], "flags": [], "local": {"q": [0, 0, 0, 1]}}},
                             "qa": [{"x": 1, "y": 2}]}) == []
    assert legacy_diff(old, {**old, "a": 2}) == [".a"]
    assert legacy_diff(old, {k: v for k, v in old.items() if k != "a"}) == [".a (gone)"]
    assert legacy_diff(old, {**old, "cams": {"p": {"enu": [1, 2, 3], "flags": []}}}) == [".cams.p.enu"]
    assert legacy_diff(old, {**old, "qa": [{"x": 1}, {"x": 2}]}) == [".qa"]
    # a recalibrated anchor (new calibration id) may change its sigma block, nothing else
    a = {**old, "anchor": {"scale": 2.0, "sigma": {"yaw_deg": 3.0}, "calibration": {"id": "one"}}}
    b = {**old, "anchor": {"scale": 2.0, "sigma": {"yaw_deg": 5.0}, "calibration": {"id": "two"}}}
    assert legacy_diff(calibration_free(a, b), calibration_free(b, a)) == []
    same_id = {**b, "anchor": {**b["anchor"], "calibration": {"id": "one"}}}      # the same calibration: pinned
    assert legacy_diff(calibration_free(a, same_id), calibration_free(same_id, a)) == [".anchor.sigma.yaw_deg"]
    c = {**b, "anchor": {**b["anchor"], "scale": 2.5}}
    assert legacy_diff(calibration_free(a, c), calibration_free(c, a)) == [".anchor.scale"]


def _upload_and_reconstruct(root: Path, ds: Path, good: list[dict[str, Any]], lm: dict[str, Any],
                            monkeypatch) -> tuple[Settings, Store, Any]:
    """The product path: every photo through ingest_upload, then the job JobRunner.schedule plans, run inline. Photo
    ids are the file stems (ingest draws random ones), so the SfM image names match the fixture's sparse model."""
    ids = iter(Path(p["file"]).stem for p in good)
    monkeypatch.setattr(ingest_module, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex=next(ids))))
    settings = Settings(data_dir=root, reconstructor="colmap", inline_jobs=True, keep_work_dirs=True)
    store = Store(settings)
    landmark = store.create_landmark(Landmark(LANDMARK, "Golden", lm["lat"], lm["lon"], 150.0, 0.0))
    for p in good:
        out = ingest_upload(store, settings, filename=Path(p["file"]).name, data=(ds / p["file"]).read_bytes(),
                            user_id="u", landmark=landmark)
        assert out.accepted, (p["file"], out.reasons)
    runner = JobRunner(store, settings, get_reconstructor(settings.reconstructor, settings))
    (job,) = runner.schedule(LANDMARK)
    return settings, store, job


def test_reconstruction_output_matches_the_golden_record(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip("pycolmap")
    ds = tmp_path / "facade"
    manifest = syn.generate_dataset(ds, sectors=[2], photos_per_sector=4, size=(640, 480), include_bad=False, seed=3)
    good = [p for p in manifest["photos"] if p["kind"] == "good"]
    lm = manifest["landmark"]
    sparse = FIXTURE / "sparse"
    if REGEN and not (sparse / "images.bin").is_file():      # the SfM output itself: one real COLMAP run
        with monkeypatch.context() as real:
            settings, store, job = _upload_and_reconstruct(tmp_path / "real", ds, good, lm, real)
            store.close()
        assert job.status == JOB_DONE, job.error
        shutil.copytree(settings.work_dir / job.id / "sparse" / "0", sparse)

    calls: list[dict[str, Any]] = []

    def database(db_path: Path) -> dict[str, Any]:
        db = pycolmap.Database.open(db_path)
        try:
            return {"cameras": [{"id": c.camera_id, "model": str(c.model), "size": [c.width, c.height],
                                 "params": list(c.params), "prior_focal": c.has_prior_focal_length}
                                for c in db.read_all_cameras()],
                    "images": [[im.name, im.camera_id] for im in db.read_all_images()]}
        finally:
            db.close()

    def extract(db_path, images, **kw):
        calls.append({"call": "extract_features", "image_names": list(kw["image_names"]),
                      "camera_mode": str(kw["camera_mode"]), "device": str(kw["device"]),
                      "options": kw["extraction_options"].todict()})

    def match(name):
        def f(db_path, **kw):
            calls.append({"call": name, "device": str(kw["device"]),
                          "pairing": kw["pairing_options"].todict() if "pairing_options" in kw else None,
                          "verification": kw["verification_options"].todict()})
        return f

    def mapping(db_path, images, out, options):
        calls.append({"call": "incremental_mapping", "options": options.todict(), "database": database(db_path)})
        return {0: pycolmap.Reconstruction(str(sparse))}

    monkeypatch.setattr(pycolmap, "extract_features", extract)
    monkeypatch.setattr(pycolmap, "match_exhaustive", match("match_exhaustive"))
    monkeypatch.setattr(pycolmap, "match_spatial", match("match_spatial"))
    monkeypatch.setattr(pycolmap, "incremental_mapping", mapping)
    settings, store, job = _upload_and_reconstruct(tmp_path / "data", ds, good, lm, monkeypatch)
    try:
        assert job.status == JOB_DONE, job.error
        photos = store.list_photos(LANDMARK)
        (model,) = store.list_models(LANDMARK)
    finally:
        store.close()
    ply = settings.data_dir / model.ply_path
    meta = json.loads(ply.with_suffix(".json").read_text(encoding="utf-8"))   # the model JSON the job wrote
    for volatile in ("timings_s", "ply_path"):
        meta.pop(volatile)
    doc = _canon(json.loads(json.dumps({
        "recon_output_version": RECON_OUTPUT_VERSION,
        "inputs": [[Path(p["file"]).name, _sha(ds / p["file"])] for p in good],          # the uploads
        # what ingest made of them and the job the planner made of those
        "photos": [[p.id, p.status, p.lat, p.lon, p.heading_deg, p.gps_accuracy_m, p.focal_35mm, p.sector,
                    _sha(settings.data_dir / p.processed_path)] for p in photos],
        "job": [job.kind, job.sector, job.photo_ids],
        "meta": meta, "ply_sha256": _sha(ply),
        "sparse_sha256": {f.name: _sha(f) for f in sorted((settings.work_dir / job.id / "sparse" / "0").iterdir())},
        "colmap_calls": calls}, default=str)))      # (pycolmap enums by name)

    golden_path = FIXTURE / "golden.json"
    old = json.loads(golden_path.read_text(encoding="utf-8")) if golden_path.is_file() else None

    def changes(golden: dict[str, Any]) -> list[str]:
        """What differs beyond keys the model JSON gained (and a recalibrated anchor's sigma)."""
        return (sorted(k for k in set(doc) | set(golden) if k != "meta" and doc.get(k) != golden.get(k))
                + legacy_diff(calibration_free(golden.get("meta"), doc["meta"]),
                              calibration_free(doc["meta"], golden.get("meta")), "meta"))

    if REGEN:
        if (old is not None and old["recon_output_version"] == RECON_OUTPUT_VERSION and old["inputs"] == doc["inputs"]
                and changes(old)):
            pytest.fail(f"the reconstruction output changed ({changes(old)}): bump "
                        "crowd3d.reconstruct.RECON_OUTPUT_VERSION first")
        golden_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["recon_output_version"] == RECON_OUTPUT_VERSION, (
        "RECON_OUTPUT_VERSION changed: regenerate tests/fixtures/recon_golden/golden.json (CROWD3D_REGEN_GOLDEN=1)")
    assert doc["inputs"] == golden["inputs"], (
        "the synthetic facade photos changed (crowd3d.synthetic): regenerate golden.json (CROWD3D_REGEN_GOLDEN=1)")
    changed = changes(golden)
    assert not changed, (f"the reconstruction output changed ({changed}): bump crowd3d.reconstruct."
                         "RECON_OUTPUT_VERSION, then regenerate golden.json with CROWD3D_REGEN_GOLDEN=1")
    assert doc == golden, ("the model JSON gained keys and keeps every earlier value: no RECON_OUTPUT_VERSION bump "
                           "needed, regenerate golden.json with CROWD3D_REGEN_GOLDEN=1 to pin them")
    assert doc["meta"]["ok"] and doc["meta"]["published"] and doc["meta"]["georegistered"]   # a real model
    anchor = doc["meta"]["anchor"]
    assert anchor["level"] == "full"
    assert anchor["sigma"] == _canon(floored_sigma(anchor["raw"]["sigma_core"], anchor["sigma"]["up_m"],
                                                   anchor["calibration"]["floors"]))
    assert doc["meta"]["num_registered"] == len(good)
    assert all(p[6] is not None for p in doc["photos"])          # focal_35mm reaches the job (its focal prior)
