"""RECON_OUTPUT_VERSION's golden test: what a reconstruction job records for fixed SfM output.

COLMAP's own result depends on thread timing (test_reconstruct.py), so here its calls are replaced by a recorder
that hands back the sparse model one real run made of the synthetic facade (fixtures/recon_golden/sparse). The
options those calls get and everything after them (model choice, georegistration, camera checks, export, the
QA gate) are compared with fixtures/recon_golden/golden.json. A difference means the reconstruction output can
change: bump crowd3d.reconstruct.RECON_OUTPUT_VERSION, then regenerate with CROWD3D_REGEN_GOLDEN=1 (refused while
the version is unchanged, unless the inputs themselves changed: the synthetic photos and their EXIF fields).
Benchmark rescores rely on the version (scripts/benchmark.py rescore).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest

from crowd3d import synthetic as syn
from crowd3d.config import Settings
from crowd3d.qa import assess
from crowd3d.reconstruct import RECON_OUTPUT_VERSION, ColmapReconstructor, ReconPhoto
from crowd3d.records import JOB_FACE, PHOTO_ACCEPTED, Landmark, PhotoRecord

FIXTURE = Path(__file__).parent / "fixtures" / "recon_golden"
REGEN = os.environ.get("CROWD3D_REGEN_GOLDEN") == "1"


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


def test_reconstruction_output_matches_the_golden_record(tmp_path, monkeypatch):
    pycolmap = pytest.importorskip("pycolmap")
    ds = tmp_path / "facade"
    manifest = syn.generate_dataset(ds, sectors=[2], photos_per_sector=4, size=(640, 480), include_bad=False, seed=3)
    good = [p for p in manifest["photos"] if p["kind"] == "good"]
    photos = [ReconPhoto(Path(p["file"]).stem, ds / p["file"], p["exif_lat"], p["exif_lon"], p["exif_heading_deg"],
                         gps_accuracy_m=p["exif_gps_accuracy_m"]) for p in good]
    lm = manifest["landmark"]
    sparse = FIXTURE / "sparse"
    if REGEN and not (sparse / "images.bin").is_file():      # the SfM input itself: one real COLMAP run
        real = ColmapReconstructor(Settings(data_dir=tmp_path / "real")).reconstruct(
            photos, lm["lat"], lm["lon"], tmp_path / "real.ply", tmp_path / "real")
        assert real.ok, real.message
        shutil.copytree(tmp_path / "real" / "sparse" / "0", sparse)

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
    settings = Settings(data_dir=tmp_path / "data")
    result = ColmapReconstructor(settings).reconstruct(photos, lm["lat"], lm["lon"], tmp_path / "model.ply",
                                                       tmp_path / "work")
    assert result.ok, result.message
    # the QA gate as the job runner applies it (crowd3d.jobs), and the model JSON it writes
    landmark = Landmark("golden", "Golden", lm["lat"], lm["lon"], 150.0, 0.0)
    records = [PhotoRecord(p.photo_id, "golden", "u", PHOTO_ACCEPTED, 0.0, lat=p.lat, lon=p.lon,
                           heading_deg=p.heading_deg, gps_accuracy_m=p.gps_accuracy_m) for p in photos]
    report = assess(result, JOB_FACE, records, landmark, settings)
    meta = {**result.to_meta(), "published": report.passed, "qa": report.to_meta()}
    for volatile in ("timings_s", "ply_path"):
        meta.pop(volatile)
    doc = _canon(json.loads(json.dumps({
        "recon_output_version": RECON_OUTPUT_VERSION,
        "inputs": [[p.photo_id, p.lat, p.lon, p.heading_deg, p.gps_accuracy_m, _sha(p.image_path)] for p in photos],
        "meta": meta, "ply_sha256": _sha(tmp_path / "model.ply"),
        "sparse_sha256": {f.name: _sha(f) for f in sorted((tmp_path / "work" / "sparse" / "0").iterdir())},
        "colmap_calls": calls}, default=str)))      # (pycolmap enums by name)

    golden_path = FIXTURE / "golden.json"
    if REGEN:
        old = json.loads(golden_path.read_text(encoding="utf-8")) if golden_path.is_file() else None
        if (old is not None and old["recon_output_version"] == RECON_OUTPUT_VERSION and old["inputs"] == doc["inputs"]
                and old != doc):
            pytest.fail("the reconstruction output changed: bump crowd3d.reconstruct.RECON_OUTPUT_VERSION first")
        golden_path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    assert golden["recon_output_version"] == RECON_OUTPUT_VERSION, (
        "RECON_OUTPUT_VERSION changed: regenerate tests/fixtures/recon_golden/golden.json (CROWD3D_REGEN_GOLDEN=1)")
    assert doc["inputs"] == golden["inputs"], (
        "the synthetic facade photos changed (crowd3d.synthetic): regenerate golden.json (CROWD3D_REGEN_GOLDEN=1)")
    changed = sorted(k for k in set(doc) | set(golden) if doc.get(k) != golden.get(k))
    assert not changed, (f"the reconstruction output changed ({changed}): bump crowd3d.reconstruct."
                         "RECON_OUTPUT_VERSION, then regenerate golden.json with CROWD3D_REGEN_GOLDEN=1")
    assert doc["meta"]["ok"] and doc["meta"]["published"] and doc["meta"]["georegistered"]   # a real model
    assert doc["meta"]["num_registered"] == len(photos)
