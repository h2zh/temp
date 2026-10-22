#!/usr/bin/env python3
"""Build a stylized ("toy" / anime) landmark model with crowd3d.stylize and write a GLB + summary.

Inputs (one of):
  --scene DIR        a MegaDepth-X style scene (DIR/images, DIR/sparse) whose reference model is
                     georeferenced into the landmark ENU frame from the photos' real EXIF GPS and
                     compass, exactly as scripts/eval_real_landmark.py does (georeference_gt);
  --job-dir DIR      a Crowd3D COLMAP job's work dir (DIR/sparse/0, DIR/images) plus --meta, the
                     model's JSON (ReconstructionResult.to_meta(), with per-camera ENU positions):
                     the similarity to ENU is recovered from the camera centres.
The landmark (centre, OSM footprint, facts) comes from --catalog (eval/landmarks.json) by --landmark.

Outputs go to --out (refused inside the repository, except the git-ignored data/ directory: the
textures are derived from photos whose licences require attribution):
  <slug>.glb, <slug>.json (summary), ortho/<slug>-wall<k>-{raw,mask,stylized}.png, <slug>-topdown.png

Examples:
  python scripts/stylize_model.py --scene /path/realdata/baptistry/1 --landmark florence-baptistery --out /tmp/out
  python scripts/stylize_model.py --job-dir data/work/<job> --meta data/models/<lm>/<model>.json \
      --landmark florence-baptistery --out /tmp/out
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from crowd3d import stylize as st  # noqa: E402
from crowd3d.geo import LocalENU  # noqa: E402
from crowd3d.reconstruct import apply_similarity  # noqa: E402


def check_outside_repo(path: Path) -> None:
    p = path.resolve()
    if (p == REPO or REPO in p.parents) and not (p == REPO / "data" or (REPO / "data") in p.parents):
        raise SystemExit(f"refusing to write photo-derived outputs into the repository: {p}")


def catalog_entry(catalog: Path, slug: str) -> dict[str, Any]:
    for e in json.loads(catalog.read_text(encoding="utf-8")):
        if e.get("slug") == slug:
            return e
    raise SystemExit(f"landmark {slug!r} not in {catalog}")


def facts_from_entry(entry: dict[str, Any], footprint_file: Path) -> st.HeightFacts:
    props: dict[str, Any] = {}
    try:
        data = json.loads(footprint_file.read_text(encoding="utf-8"))
        for f in data.get("features", []):
            props.update(f.get("properties") or {})
    except (OSError, ValueError):
        pass

    def num(v):
        try:
            return float(str(v).split()[0]) if v not in (None, "") else None
        except ValueError:
            return None

    return st.HeightFacts(osm_height_m=num(props.get("height")), osm_levels=num(props.get("building:levels")),
                          wikidata_height_m=num(entry.get("wikidata_height_m")))


def camera_from_colmap(name: str, cam, cam_from_enu: np.ndarray, center_enu: np.ndarray, image_path: Path) -> st.PosedCamera:
    model = cam.model.name
    p = [float(v) for v in cam.params]
    if model == "PINHOLE":
        fx, fy, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], 0.0, 0.0
    elif model == "SIMPLE_PINHOLE":
        fx = fy = p[0]
        cx, cy, k1, k2 = p[1], p[2], 0.0, 0.0
    elif model == "SIMPLE_RADIAL":
        fx = fy = p[0]
        cx, cy, k1, k2 = p[1], p[2], p[3], 0.0
    elif model == "RADIAL":
        fx = fy = p[0]
        cx, cy, k1, k2 = p[1], p[2], p[3], p[4]
    elif model == "OPENCV":
        fx, fy, cx, cy, k1, k2 = p[0], p[1], p[2], p[3], p[4], p[5]   # tangential terms ignored (tiny on phones)
    else:
        raise ValueError(f"unsupported camera model {model}")
    return st.PosedCamera(name, int(cam.width), int(cam.height), fx, fy, cx, cy, cam_from_enu, center_enu, k1, k2,
                          image_path=image_path)


def load_megadepth_scene(scene: Path, entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[st.PosedCamera], dict]:
    import pycolmap

    import eval_real_landmark as ev

    gt = ev.load_ground_truth(scene)
    frame = ev.georeference_gt(gt, entry["lat"], entry["lon"], 1.6)
    rec = pycolmap.Reconstruction(str(scene / "sparse"))
    pts3d = list(rec.points3D.values())
    points = apply_similarity(frame.transform, np.array([p.xyz for p in pts3d]).reshape(-1, 3))
    colors = np.array([p.color for p in pts3d], dtype=np.uint8).reshape(-1, 3)
    r_enu = frame.rotation
    cams = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        if not im.has_pose:
            continue
        path = scene / "images" / im.name
        if not path.is_file():
            continue
        rot = im.cam_from_world().rotation.matrix() @ r_enu.T          # stored pixels' frame, ENU
        center = apply_similarity(frame.transform, np.asarray(im.projection_center())[None])[0]
        cams.append(camera_from_colmap(im.name, rec.cameras[im.camera_id], rot, center, path))
    report = {k: v for k, v in frame.report.items() if k != "per_photo"}
    return points, colors, cams, {"georeference": "GT reference model + real EXIF GPS/compass (eval_real_landmark)",
                                  "gt_frame": report}


def load_job(job_dir: Path, meta_path: Path) -> tuple[np.ndarray, np.ndarray, list[st.PosedCamera], dict]:
    import pycolmap

    from crowd3d.reconstruct import photo_id_from_image_name

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = meta.get("meta", meta)
    rec = pycolmap.Reconstruction(str(job_dir / "sparse" / "0"))
    src, dst = [], []
    for im in rec.images.values():
        cam = meta.get("cameras", {}).get(photo_id_from_image_name(im.name))
        if im.has_pose and cam and cam.get("enu") is not None:
            src.append(np.asarray(im.projection_center()))
            dst.append(np.asarray(cam["enu"], float))
    if len(src) < 3:
        raise SystemExit("the meta file has fewer than 3 camera positions for this reconstruction")
    import eval_real_landmark as ev

    scale, rot, trans = ev.umeyama(np.array(src), np.array(dst), with_scale=True)
    transform = np.eye(4)
    transform[:3, :3] = scale * rot
    transform[:3, 3] = trans
    resid = np.linalg.norm(apply_similarity(transform, np.array(src)) - np.array(dst), axis=1)
    pts3d = list(rec.points3D.values())
    points = apply_similarity(transform, np.array([p.xyz for p in pts3d]).reshape(-1, 3))
    colors = np.array([p.color for p in pts3d], dtype=np.uint8).reshape(-1, 3)
    cams = []
    for im in sorted(rec.images.values(), key=lambda i: i.name):
        path = job_dir / "images" / im.name
        if not im.has_pose or not path.exists():
            continue
        r = im.cam_from_world().rotation.matrix() @ rot.T
        center = apply_similarity(transform, np.asarray(im.projection_center())[None])[0]
        cams.append(camera_from_colmap(im.name, rec.cameras[im.camera_id], r, center, path))
    return points, colors, cams, {"georeference": "Crowd3D georegister (recovered from the meta's camera ENU)",
                                  "camera_fit_residual_max_m": round(float(resid.max()), 4)}


def save_debug(model: st.StylizedModel, out: Path, slug: str, footprint: np.ndarray) -> list[str]:
    """Orthophotos before / after stylization and a top-down sketch of the massing."""
    files = []
    od = out / "ortho"
    od.mkdir(parents=True, exist_ok=True)
    for wi, ortho in model.orthophotos.items():
        raw = ortho.rgb.copy()
        masked = raw.copy()
        masked[~ortho.mask] = (masked[~ortho.mask] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)
        cv2.imwrite(str(od / f"{slug}-wall{wi}-raw.png"), raw[..., ::-1])
        cv2.imwrite(str(od / f"{slug}-wall{wi}-mask.png"), masked[..., ::-1])
        files += [f"ortho/{slug}-wall{wi}-raw.png", f"ortho/{slug}-wall{wi}-mask.png"]
        if wi in model.facades:
            cv2.imwrite(str(od / f"{slug}-wall{wi}-stylized.png"), model.facades[wi].texture[..., ::-1])
            files.append(f"ortho/{slug}-wall{wi}-stylized.png")
    # top-down: footprint (grey), blocks (outline), observed walls (thick green), locked (dashed red)
    pts = np.vstack([footprint] + [p.polygon for p in model.massing.prisms])
    lo, hi = pts.min(0) - 5, pts.max(0) + 5
    s = 900 / max(hi - lo)
    size = (int((hi[1] - lo[1]) * s) + 1, int((hi[0] - lo[0]) * s) + 1)
    img = np.full((*size, 3), 255, np.uint8)

    def px(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        return np.stack([(xy[:, 0] - lo[0]) * s, (hi[1] - xy[:, 1]) * s], 1).astype(np.int32)

    cv2.polylines(img, [px(footprint)], True, (190, 190, 190), 1)
    for pr in model.massing.prisms:
        cv2.polylines(img, [px(pr.polygon)], True, (120, 120, 120), 1)
        if pr.apex is not None:
            cv2.circle(img, tuple(int(v) for v in px(pr.apex[:2])[0]), 4, (0, 0, 0), -1)
    for w in model.massing.walls:
        a, b = px(w.a)[0], px(w.b)[0]
        if w.index in model.facades:
            cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (60, 170, 60), 4)
        cv2.putText(img, str(w.index), tuple(int(v) for v in ((a + b) // 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200), 1)
    cv2.imwrite(str(out / f"{slug}-topdown.png"), img)
    files.append(f"{slug}-topdown.png")
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--scene", type=Path, help="MegaDepth-X style scene dir (images/, sparse/)")
    src.add_argument("--job-dir", type=Path, help="Crowd3D COLMAP job work dir (sparse/0, images/)")
    ap.add_argument("--meta", type=Path, help="with --job-dir: the model's JSON with per-camera ENU positions")
    ap.add_argument("--landmark", required=True, help="slug in the catalog")
    ap.add_argument("--catalog", type=Path, default=REPO / "eval" / "landmarks.json")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--no-snap", action="store_true", help="do not move the model onto the OSM footprint")
    ap.add_argument("--no-debug", action="store_true", help="skip orthophoto / top-down images")
    args = ap.parse_args(argv)
    check_outside_repo(args.out)
    if args.job_dir and not args.meta:
        ap.error("--job-dir needs --meta")

    t0 = time.perf_counter()
    entry = catalog_entry(args.catalog, args.landmark)
    enu = LocalENU(entry["lat"], entry["lon"])
    import eval_real_landmark as ev

    fp_file = REPO / entry["osm"]["file"]
    footprint = ev.load_footprint(fp_file, enu)
    if args.scene:
        points, colors, cams, source = load_megadepth_scene(args.scene, entry)
    else:
        points, colors, cams, source = load_job(args.job_dir, args.meta)
    load_s = time.perf_counter() - t0

    attribution = [entry["osm"].get("attribution", "© OpenStreetMap contributors (ODbL)"),
                   "Facade colours derived from Wikimedia Commons photos (per-photo licences; see the scene's licence table)"]
    scene = st.SceneInput(entry.get("name", args.landmark), footprint, points, colors, cams,
                          facts_from_entry(entry, fp_file), attribution, snap=not args.no_snap)
    model = st.stylize_landmark(scene)
    args.out.mkdir(parents=True, exist_ok=True)
    slug = args.landmark
    (args.out / f"{slug}.glb").write_bytes(model.glb)
    summary = dict(model.summary)
    summary["source"] = source
    summary["timings_s"] = {"load_and_georeference": round(load_s, 2), **summary["timings_s"]}
    summary["total_s"] = round(load_s + model.summary["total_s"], 2)
    if not args.no_debug:
        summary["debug_files"] = save_debug(model, args.out, slug, footprint)
    (args.out / f"{slug}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=_json_default),
                                           encoding="utf-8")
    s = model.summary
    print(f"{slug}: {s['observed_walls']}/{s['num_walls']} walls observed "
          f"({s['observed_wall_area_fraction']:.0%} of wall area, {s['observed_pixel_area_fraction']:.0%} pixels), "
          f"{s['triangles']} triangles, GLB {s['glb_bytes'] / 1024:.0f} KB, QA {'ok' if s['qa']['passed'] else 'FAILED'}, "
          f"{summary['total_s']:.1f} s -> {args.out / (slug + '.glb')}")
    return 0 if s["qa"]["passed"] else 1


def _json_default(o: Any) -> Any:
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    raise TypeError(type(o))


if __name__ == "__main__":
    raise SystemExit(main())
