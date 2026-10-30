"""Stage 2a of the landmark map (docs/ARCHITECTURE.md 3.4.2): would sector views cut from one map do what the face
models do? An offline replay of recorded benchmark runs, no SfM.

    python -I scripts/replay_map_views.py --testset commons-ten-v1 --result <recorded result JSON> \
        --reference-cache <dir> --out <out.json> [--md <out.md>] [--buildings ...]

For every full job a building run kept (its work dir's sparse model: the best component only), and every sector
that is face-ready among the job's photos:
- the view: the cameras of the photos a face job of that sector would take (coverage.select_face_photos of the full
  job's photos: the sector plus Settings.face_margin_deg), and the points with at least 2 observations from them
  (cut_view);
- georegistered twice: on its own cameras (as a face job is, "view"), and as part of the whole component (the full
  job's georegistration, "component");
- each judged as a face model would be: qa.assess(kind face), then the harness's outline check (wall points against
  the OSM outline) and reference check (gt_model_check against the outline-corrected reference);
- next to the face model of that sector the landmark page showed when the full job ran (latest published before it),
  scored the same way now.
The go/no-go rule for stage 2b: the views (either georegistration) publish at least as many facades, with no more
wrong-facade cameras shown (gt_gross_shown) and no larger reference position or rotation errors than the face models.
Data (photos, reference models) stays outside the repository; --out too.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import benchmark as bench  # noqa: E402
import eval_real_landmark as ev  # noqa: E402

from crowd3d import params  # noqa: E402
from crowd3d.config import Settings  # noqa: E402
from crowd3d.coverage import compute_coverage, select_face_photos  # noqa: E402
from crowd3d.geo import LocalENU  # noqa: E402
from crowd3d.qa import assess  # noqa: E402
from crowd3d.reconstruct import (  # noqa: E402
    MIN_REGISTERED,
    GeoregisterInputs,
    ReconPhoto,
    ReconstructionResult,
    apply_similarity,
    georegister_inputs,
    georegister_model,
    photo_id_from_image_name,
)
from crowd3d.records import JOB_FACE, JOB_FULL  # noqa: E402

log = logging.getLogger("replay_map_views")


def cut_view(image_photo_ids: list[str], tracks: list[list[int]], view_ids: set[str],
             min_observations: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """A sector view of one map: (mask of its images, mask of its points). image_photo_ids: the photo id of each posed
    image; tracks: per point, the indices (into image_photo_ids) of the images observing it. The view keeps the images
    whose photo is in view_ids and the points with at least min_observations observations from those images (two
    observations from one image count twice, as COLMAP's track length does)."""
    images = np.array([pid in view_ids for pid in image_photo_ids], dtype=bool)
    points = np.array([int(images[list(t)].sum()) >= min_observations if len(t) else False for t in tracks], dtype=bool)
    return images, points


def view_inputs(full: GeoregisterInputs, images: np.ndarray, xyz: np.ndarray) -> GeoregisterInputs | None:
    """georegister_inputs of a view: the component's cameras the image mask keeps, with the view's points; None below
    MIN_REGISTERED cameras. The focal-ratio sanity rule is applied to the view's cameras, as georegister_inputs does."""
    if images.sum() < MIN_REGISTERED:
        return None
    tun = params.current()
    focal = full.focal[images]
    sane = (focal >= tun.focal_ratio_min) & (focal <= tun.focal_ratio_max)
    return GeoregisterInputs(
        images=[im for im, keep in zip(full.images, images) if keep],
        photos=[p for p, keep in zip(full.photos, images) if keep],
        rots=full.rots[images], centers=full.centers[images], focal=focal, gps_en=full.gps_en[images],
        accuracy=full.accuracy[images], use=sane if sane.sum() >= MIN_REGISTERED else np.ones(len(focal), dtype=bool),
        xyz=xyz)


def judged(cameras: dict[str, dict[str, Any]], transform: np.ndarray, xyz_local: np.ndarray, errors: np.ndarray,
           photos: list, est_rots: dict[str, np.ndarray], lm, settings: Settings, outline, gt_poses, photo_key,
           georegistered: bool) -> dict[str, Any]:
    """A view (or a face model) judged as the product and the harness judge a face model: qa.assess, the outline
    check and the reference check."""
    points = apply_similarity(transform, xyz_local) if len(xyz_local) else np.zeros((0, 3))
    result = ReconstructionResult(
        ok=True, message="view", num_input=len(photos), num_registered=len(cameras), num_points=len(points),
        mean_reprojection_error_px=float(np.mean(errors)) if len(errors) else None, georegistered=georegistered,
        cameras={k: dict(v, flags=list(v.get("flags") or [])) for k, v in cameras.items()},
        points_median_en=[float(v) for v in np.median(points[:, :2], axis=0)] if len(points) else None)
    report = assess(result, JOB_FACE, photos, lm, settings)
    out: dict[str, Any] = {"published": report.passed, "qa_reasons": report.reasons, "registered": len(cameras),
                           "inputs": len(photos), "points": len(points)}
    out["outline"] = (ev.outline_check(points, outline).get("fit") or {}) if outline is not None else None
    out["gt"] = (ev.gt_model_check(result.cameras, est_rots, gt_poses, photo_key, report.passed)
                 if gt_poses is not None else None)
    return out


def face_model_scores(model, run_data: Path, outline, gt_poses, photo_key) -> dict[str, Any]:
    """The face model as scored now: its recorded cameras (flags as the job stored them), its PLY and kept SfM model."""
    meta = model.meta
    cams = meta.get("cameras") or {}
    rec = ev.load_job_model(run_data / "work" / model.job_id) if (run_data / "work" / model.job_id).is_dir() else None
    est_rots = {photo_id_from_image_name(im.name): np.asarray(im.cam_from_world().rotation.matrix())
                for im in rec.images.values() if im.has_pose} if rec is not None else {}
    ply = run_data / model.ply_path
    out: dict[str, Any] = {"published": bool(meta.get("published", True)), "registered": meta.get("num_registered"),
                           "inputs": model.num_photos, "points": meta.get("num_points"), "model_id": model.id}
    out["outline"] = ((ev.outline_check(ev.read_ply_points(ply), outline).get("fit") or {})
                      if outline is not None and ply.is_file() else None)
    out["gt"] = (ev.gt_model_check(cams, est_rots, gt_poses, photo_key, out["published"])
                 if gt_poses is not None else None)
    return out


def replay_run(run_dir: Path, outline_path: Path | None, gt_poses, all_full_jobs: bool = True) -> list[dict[str, Any]]:
    """Every (full job, face-ready sector) of one recorded building run: the view, georegistered alone and with its
    component, and the face model the page showed for that sector when the full job ran."""
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    data = ev.run_data_dir(run_dir, summary)
    lm, photos, models, jobs = ev.load_run_store(data)
    settings = params.settings_with(Settings(data_dir=data), (summary.get("effective_params") or {}).get("settings")
                                    or {})
    enu = LocalENU(lm.lat, lm.lon)
    outline = ev.load_footprint(outline_path, enu) if outline_path else None
    photo_key = {r["photo_id"]: r["key"] for r in summary["photos"] if r.get("photo_id")}
    by_id = {p.id: p for p in photos}
    fulls = sorted((j for j in jobs if j.kind == JOB_FULL and (data / "work" / j.id / "sparse").is_dir()),
                   key=lambda j: (j.created_at, j.id))
    if not all_full_jobs:
        fulls = fulls[-1:]
    rows = []
    for job in fulls:
        rec = ev.load_job_model(data / "work" / job.id)
        if rec is None:
            continue
        job_photos = [by_id[i] for i in job.photo_ids if i in by_id]
        recon = {p.id: ReconPhoto(p.id, Path(), p.lat, p.lon, p.heading_deg, gps_accuracy_m=p.gps_accuracy_m,
                                  focal_35mm=p.focal_35mm) for p in job_photos}
        by_name = {im.name: recon[pid] for im in rec.images.values()
                   if (pid := photo_id_from_image_name(im.name)) in recon}
        gps = {n: np.array(enu.to_enu(p.lat, p.lon)[:2]) for n, p in by_name.items()}
        full = georegister_inputs(rec, by_name, gps)
        if full is None:
            continue
        geo_full, cams_full = georegister_model(full, settings.camera_height_m)
        image_ids = [im.image_id for im in full.images]
        index = {iid: k for k, iid in enumerate(image_ids)}
        pids = [p.photo_id for p in full.photos]
        points = list(rec.points3D.values())
        tracks = [[index[e.image_id] for e in pt.track.elements if e.image_id in index] for pt in points]
        xyz = np.array([pt.xyz for pt in points]).reshape(-1, 3)
        errors = np.array([pt.error for pt in points])
        est_rots_all = {p: r for p, r in zip(pids, full.rots)}
        shown = ev.latest_published([m for m in models if m.created_at <= (job.finished_at or job.created_at)])
        for sector in compute_coverage(job_photos, settings).face_ready:
            want = select_face_photos(job_photos, sector, settings)
            view_ids = {p.id for p in want}
            img_mask, pt_mask = cut_view(pids, tracks, view_ids)
            row: dict[str, Any] = {"run": run_dir.name, "full_job": job.id, "full_photos": len(job.photo_ids),
                                   "full_registered": len(pids), "sector": sector, "view_inputs": len(want),
                                   "view_cameras": int(img_mask.sum()), "view_points": int(pt_mask.sum())}
            vin = view_inputs(full, img_mask, xyz[pt_mask])
            if vin is None:
                row["view"] = row["component"] = {"published": False, "qa_reasons": ["too few cameras"],
                                                  "registered": int(img_mask.sum()), "inputs": len(want), "gt": None,
                                                  "outline": None, "points": int(pt_mask.sum())}
            else:
                est_rots = {p.photo_id: est_rots_all[p.photo_id] for p in vin.photos}
                geo_v, cams_v = georegister_model(vin, settings.camera_height_m)
                row["view"] = judged(cams_v, geo_v.transform, xyz[pt_mask], errors[pt_mask], want, est_rots, lm,
                                     settings, outline, gt_poses, photo_key, geo_v.georegistered)
                cams_c = {p.photo_id: cams_full[p.photo_id] for p in vin.photos}
                row["component"] = judged(cams_c, geo_full.transform, xyz[pt_mask], errors[pt_mask], want, est_rots,
                                          lm, settings, outline, gt_poses, photo_key, geo_full.georegistered)
            face = shown.get((JOB_FACE, sector))
            row["face"] = face_model_scores(face, data, outline, gt_poses, photo_key) if face is not None else None
            rows.append(row)
    return rows


def _gt(entry: dict[str, Any] | None, key: str) -> Any:
    gt = (entry or {}).get("gt") or {}
    if key in ("position", "rotation"):
        return ((gt.get("position_error_m" if key == "position" else "rotation_error_deg")) or {}).get("median")
    return gt.get(key)


VARIANTS = ("face", "view", "component", "hybrid")


def pick(row: dict[str, Any], variant: str) -> dict[str, Any] | None:
    """The model a variant shows for a (full job, sector) pair; hybrid: the view (own georegistration) where it passes
    the quality gate, else the face model (stage 2b's fallback for sectors no accepted component covers)."""
    if variant == "hybrid":
        return row["view"] if row["view"]["published"] else row.get("face")
    return row.get(variant)


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Per variant (face models, views alone, views with their component, views with face models as the fallback):
    facades published, wrong-facade cameras shown, and the median over published models of their reference position /
    rotation error medians and of their outline rotation; over the (full job, sector) pairs given."""
    out = {}
    for variant in VARIANTS:
        pub = [e for r in rows if (e := pick(r, variant)) and e["published"]]

        def med(values):
            v = [float(x) for x in values if x is not None]
            return round(float(np.median(v)), 3) if v else None

        out[variant] = {
            "facades_published": len(pub),
            "gt_gross_shown": int(sum(_gt(e, "gross_shown") or 0 for e in pub)),
            "gt_cameras": int(sum(_gt(e, "n") or 0 for e in pub)),
            "gt_position_median_m": med(_gt(e, "position") for e in pub),
            "gt_rotation_median_deg": med(_gt(e, "rotation") for e in pub),
            "outline_rotation_abs_deg": med(abs(e["outline"]["rotation_deg"]) for e in pub
                                            if e.get("outline") and e["outline"].get("rotation_deg") is not None),
        }
    return out


def scopes(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """summarize_rows and the verdicts over every full job ("all") and over each run's last one ("final")."""
    out: dict[str, Any] = {}
    for scope, sel in (("all", rows), ("final", [r for r in rows if r.get("final")])):
        s = summarize_rows(sel)
        out[scope] = {**s, "pairs": len(sel), "verdict": {k: verdict(s, k) for k in VARIANTS[1:]}}
    return out


def verdict(s: dict[str, Any], variant: str) -> dict[str, Any]:
    """The stage 2b rule for one variant against the face models: at least as many facades, no more wrong-facade
    cameras shown, reference position and rotation errors not larger."""
    f, v = s["face"], s[variant]
    checks = {"facades": v["facades_published"] >= f["facades_published"],
              "gross_shown": v["gt_gross_shown"] <= f["gt_gross_shown"]}
    for k in ("gt_position_median_m", "gt_rotation_median_deg"):
        checks[k] = None if f[k] is None or v[k] is None else v[k] <= f[k]
    return {"checks": checks, "go": all(c is not False for c in checks.values())}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--testset", required=True)
    p.add_argument("--result", type=Path, required=True, help="a recorded run result whose artefacts to replay")
    p.add_argument("--reference-cache", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=bench.DEFAULT_DATA_ROOT)
    p.add_argument("--gt-root", type=Path, default=None)
    p.add_argument("--buildings", nargs="*")
    p.add_argument("--last-full-job-only", action="store_true", help="only each run's last full job")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--md", type=Path, default=None)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ts, ts_path, ts_sha = bench.load_testset(args.testset)
    tid = ts["testset_id"]
    src = bench.load_result(args.result)
    doc: dict[str, Any] = {"testset": tid, "result": str(args.result), "buildings": {}}
    for b in bench.select_buildings(ts, args.buildings):
        slug = b["slug"]
        if not (src["buildings"].get(slug) or {}).get("metrics"):
            continue
        integ = bench.verify_photos(b, args.data_root / tid / slug)
        gt, gt_integ = bench.resolve_gt(b, args.data_root / tid / slug, args.data_root, args.gt_root)
        spec = bench.building_spec(b, ts_path.parent, integ, gt, False)
        spec["reference_cache"] = bench.reference_cache_spec(args.reference_cache, ts_sha, slug, gt_integ)
        cb = ev.building_from_spec(spec)
        _, _, gt_poses = ev.building_reference(cb, {}, LocalENU(cb.lat, cb.lon))
        outline, _ = bench.building_outline(b, ts_path.parent)
        rows = []
        for run_dir in bench.repeat_dirs(src, slug):
            if (run_dir / "summary.json").is_file():
                rows += replay_run(run_dir, outline, gt_poses, all_full_jobs=not args.last_full_job_only)
        if not rows:
            continue
        last: dict[str, tuple[int, str]] = {}       # run -> its last (largest) full job
        for r in rows:
            if last.get(r["run"], (0, ""))[0] <= r["full_photos"]:
                last[r["run"]] = (r["full_photos"], r["full_job"])
        for r in rows:
            r["final"] = r["full_job"] == last[r["run"]][1]
        doc["buildings"][slug] = {"rows": rows, **scopes(rows)}
        log.info("%s: %d (full job, sector) pairs; final: %s", slug, len(rows),
                 json.dumps(doc["buildings"][slug]["final"]))
    doc["aggregate"] = scopes([r for b in doc["buildings"].values() for r in b["rows"]])
    args.out.write_text(json.dumps(doc, indent=1, default=ev._json_default), encoding="utf-8")
    if args.md:
        args.md.write_text(format_md(doc), encoding="utf-8")
    print(format_md(doc))
    return 0


def format_md(doc: dict[str, Any]) -> str:
    def cells(s: dict[str, Any]) -> str:
        return (f"{s['facades_published']} | {s['gt_gross_shown']} / {s['gt_cameras']} | {s['gt_position_median_m']} | "
                f"{s['gt_rotation_median_deg']} | {s['outline_rotation_abs_deg']}")

    lines = [f"# Sector views cut from one map vs face models ({doc['testset']})", "",
             "| building | pairs | variant | facades published | gross shown / gt cameras | gt position median m |"
             " gt rotation median deg | outline rotation deg |", "|---|---|---|---|---|---|---|---|"]
    for slug, b in [*doc["buildings"].items(), ("all buildings", doc["aggregate"])]:
        for scope in ("all", "final"):
            for variant in VARIANTS:
                lines.append(f"| {slug} | {scope} ({b[scope]['pairs']}) | {variant} | {cells(b[scope][variant])} |")
    lines.append("")
    for scope in ("all", "final"):
        verdicts = doc["aggregate"][scope]["verdict"].items()
        lines.append(f"Go / no-go for stage 2b ({scope} full jobs, views vs face models): " + ", ".join(
            f"{k}: {'go' if v['go'] else 'no-go'} {v['checks']}" for k, v in verdicts))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    sys.exit(main())
