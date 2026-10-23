"""Build a stylized ("toy" / anime) landmark model with crowd3d.stylize and write a GLB + summary.

Inputs (one of):
  --scene DIR        a MegaDepth-X style scene (DIR/images, DIR/sparse) whose reference model is
                     georeferenced into the landmark ENU frame from the photos' real EXIF GPS and
                     compass, exactly as scripts/eval_real_landmark.py does (georeference_gt);
  --job-dir DIR      a Crowd3D COLMAP job's work dir (DIR/sparse/0, DIR/images) plus --meta, the
                     model's JSON (ReconstructionResult.to_meta(), with per-camera ENU positions):
                     the similarity to ENU is recovered from the camera centres.
The landmark (centre, OSM footprint, facts) comes from --catalog (eval/landmarks.json) by --landmark.

The footprint keeps its courtyards (inner rings); a MultiPolygon uses its largest member. A catalog
entry without an OSM footprint is modelled from an outline traced in the points (or refused when there
are none). --snap-from reuses the footprint snap stored in an earlier summary (<slug>.json) or GLB, so a
republished model does not shift.

Outputs go to --out (refused inside the repository, except the git-ignored data/ directory: the
textures are derived from photos whose licences require attribution):
  <slug>.glb, <slug>.json (summary), ortho/<slug>-wall<k>-{raw,mask,stylized}.png (orthophoto, its
  unobserved part in magenta and its observed-fill holes (D12) in cyan, the stylized facade),
  <slug>-facades.jpg (before / after sheet of the most observed facades), <slug>-topdown.png
View the GLB with web/stylized.html?model=<url of the GLB>.

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

from crowd3d import stylize as st
from crowd3d.geo import LocalENU
from crowd3d.reconstruct import apply_similarity


def check_outside_repo(path: Path) -> None:
    p = path.resolve()
    if (p == REPO or REPO in p.parents) and not (p == REPO / "data" or (REPO / "data") in p.parents):
        raise SystemExit(f"refusing to write photo-derived outputs into the repository: {p}")


def catalog_entry(catalog: Path, slug: str) -> dict[str, Any]:
    for e in json.loads(catalog.read_text(encoding="utf-8")):
        if e.get("slug") == slug:
            return e
    raise SystemExit(f"landmark {slug!r} not in {catalog}")


def _num(v: Any) -> float | None:
    """An OSM / Wikidata height value: '12', '12 m', '12.5;14' (first value), None or junk -> None."""
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(str(v).replace(",", ".").split(";")[0].split()[0])
    except (ValueError, IndexError):
        return None
    return x if math.isfinite(x) and 0 < x < 1000 else None


def facts_from_entry(entry: dict[str, Any], footprint_file: Path | None) -> st.HeightFacts:
    props: dict[str, Any] = {}
    if footprint_file is not None:
        try:
            data = json.loads(footprint_file.read_text(encoding="utf-8"))
            for f in data.get("features", []):
                props.update(f.get("properties") or {})
        except (OSError, ValueError):
            pass
    return st.HeightFacts(osm_height_m=_num(props.get("height")), osm_levels=_num(props.get("building:levels")),
                          wikidata_height_m=_num(entry.get("wikidata_height_m")),
                          osm_roof_height_m=_num(props.get("roof:height")))


def load_footprint_rings(path: Path, enu: LocalENU) -> tuple[np.ndarray, list[np.ndarray]]:
    """(outer ring, courtyard rings) in ENU metres from a GeoJSON / Nominatim JSON file: the first
    Polygon, or the largest member of the first MultiPolygon. Unlike eval_real_landmark.load_footprint
    (outer ring only) it keeps the inner rings, so courtyards are not roofed over."""

    def find(obj):
        if isinstance(obj, dict):
            if obj.get("type") == "Polygon" and obj.get("coordinates"):
                return [obj["coordinates"]]
            if obj.get("type") == "MultiPolygon" and obj.get("coordinates"):
                return obj["coordinates"]
            for v in obj.values():
                got = find(v)
                if got:
                    return got
        elif isinstance(obj, list):
            for v in obj:
                got = find(v)
                if got:
                    return got
        return None

    polys = find(json.loads(path.read_text(encoding="utf-8")))
    if not polys:
        raise ValueError(f"no Polygon in {path}")

    def ring_enu(ring):
        en = np.array([enu.to_enu(lat, lon)[:2] for lon, lat in ring])
        return en[:-1] if len(en) > 1 and np.allclose(en[0], en[-1]) else en

    best = max(polys, key=lambda rings: abs(st.signed_area(ring_enu(rings[0]))) if rings and len(rings[0]) >= 3 else 0.0)
    return ring_enu(best[0]), [ring_enu(r) for r in best[1:] if len(r) >= 4]


def point_quality(rec, n: int) -> np.ndarray:
    """(n, 2) per point of `rec.points3D.values()` order: track length and an approximate triangulation
    angle (twice the largest angle between a viewing ray and the mean ray, degrees)."""
    pts = list(rec.points3D.values())
    pid, iid = [], []
    for k, p in enumerate(pts):
        for el in p.track.elements:
            pid.append(k)
            iid.append(el.image_id)
    pid, iid = np.asarray(pid, int), np.asarray(iid, int)
    xyz = np.array([p.xyz for p in pts]).reshape(-1, 3)
    ids = np.array(sorted(rec.images))
    centres = np.array([np.asarray(rec.images[i].projection_center()) for i in ids])
    ray = centres[np.searchsorted(ids, iid)] - xyz[pid]
    ray /= np.maximum(np.linalg.norm(ray, axis=1, keepdims=True), 1e-12)
    mean = np.zeros_like(xyz)
    np.add.at(mean, pid, ray)
    mean /= np.maximum(np.linalg.norm(mean, axis=1, keepdims=True), 1e-12)
    ang = np.degrees(np.arccos(np.clip((ray * mean[pid]).sum(axis=1), -1.0, 1.0)))
    amax = np.zeros(len(pts))
    np.maximum.at(amax, pid, ang)
    out = np.stack([np.bincount(pid, minlength=len(pts)), np.minimum(2.0 * amax, 180.0)], 1).astype(float)
    return out[:n]


def snap_from(path: Path) -> st.FootprintSnap:
    """The footprint snap of an earlier model version: its summary JSON ("snap") or its GLB (scene extras)."""
    if path.suffix.lower() == ".glb":
        gltf, _ = st.read_glb(path.read_bytes())
        meta = gltf["scenes"][0].get("extras", {}).get("snap")
    else:
        meta = json.loads(path.read_text(encoding="utf-8")).get("snap")
    if not meta or not meta.get("accepted"):
        raise SystemExit(f"{path} holds no accepted footprint snap")
    return st.FootprintSnap.from_meta(meta)


def camera_from_colmap(name: str, cam, cam_from_enu: np.ndarray, center_enu: np.ndarray, image_path: Path,
                       photo_id: str | None = None) -> st.PosedCamera:
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
                          image_path=image_path, photo_id=photo_id or name)


def load_megadepth_scene(scene: Path, entry: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[st.PosedCamera], dict, np.ndarray]:
    import eval_real_landmark as ev
    import pycolmap

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
                                  "gt_frame": report}, point_quality(rec, len(points))


def load_job(job_dir: Path, meta_path: Path) -> tuple[np.ndarray, np.ndarray, list[st.PosedCamera], dict, np.ndarray]:
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
        cams.append(camera_from_colmap(im.name, rec.cameras[im.camera_id], r, center, path,
                                       photo_id=photo_id_from_image_name(im.name)))
    return points, colors, cams, {"georeference": "Crowd3D georegister (recovered from the meta's camera ENU)",
                                  "camera_fit_residual_max_m": round(float(resid.max()), 4)}, point_quality(rec, len(points))


def snap_prior(meta_path: Path) -> st.SnapPrior:
    """The georegistration's yaw sigma and the cameras' own GPS from a model JSON, for the snap's yaw window and
    GPS check, and whether it is georegistered at all (STYLIZED_MODEL 3.2: a prior-only model colours nothing;
    a model JSON that does not say counts as prior only). Flagged cameras (they disagree with their own
    sensors) are left out; a model written before the sigma fields existed gives the default window and no
    GPS check."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = meta.get("meta", meta)
    cams = [c for c in (meta.get("cameras") or {}).values()
            if isinstance(c, dict) and not c.get("flags") and c.get("enu") is not None and c.get("gps_en") is not None]
    sigma, scale_sigma = meta.get("yaw_sigma_deg"), meta.get("scale_sigma")
    xy = np.array([c["enu"][:2] for c in cams], float).reshape(-1, 2) if cams else None
    gps = np.array([c["gps_en"] for c in cams], float).reshape(-1, 2) if cams else None
    return st.SnapPrior(float(sigma) if sigma is not None else None, xy, gps,
                        float(scale_sigma) if scale_sigma is not None else None,
                        georegistered=bool(meta.get("georegistered", False)))


def save_debug(model: st.StylizedModel, out: Path, slug: str, footprint: np.ndarray) -> list[str]:
    """Orthophotos before / after stylization and a top-down sketch of the massing."""
    files = []
    od = out / "ortho"
    od.mkdir(parents=True, exist_ok=True)
    tiles = []
    for wi, ortho in model.orthophotos.items():
        raw = ortho.rgb.copy()
        mask = model.facades[wi].mask if wi in model.facades else ortho.mask
        fill = model.facades[wi].fill if wi in model.facades else None
        masked = raw.copy()
        masked[~mask] = (masked[~mask] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)
        if fill is not None and fill.any():
            masked[fill] = (raw[fill] * 0.35 + np.array([0, 255, 255]) * 0.65).astype(np.uint8)
        cv2.imwrite(str(od / f"{slug}-wall{wi}-raw.png"), raw[..., ::-1])
        cv2.imwrite(str(od / f"{slug}-wall{wi}-mask.png"), masked[..., ::-1])
        files += [f"ortho/{slug}-wall{wi}-raw.png", f"ortho/{slug}-wall{wi}-mask.png"]
        if wi in model.facades:
            tex = model.facades[wi].texture
            cv2.imwrite(str(od / f"{slug}-wall{wi}-stylized.png"), tex[..., ::-1])
            files.append(f"ortho/{slug}-wall{wi}-stylized.png")
            tiles.append((int(mask.sum()), wi, masked, tex))
    if tiles:
        # before / after sheet of the most observed facades: orthophoto (magenta = not observed, cyan = observed-fill) over its stylization
        cols = []
        for _, wi, masked, tex in sorted(tiles, key=lambda t: -t[0])[:8]:
            s_ = 300 / masked.shape[0]
            size = (max(1, int(masked.shape[1] * s_)), 300)
            a = cv2.resize(masked, size, interpolation=cv2.INTER_AREA)
            b = cv2.resize(tex, size, interpolation=cv2.INTER_NEAREST)
            col = np.vstack([a, np.full((8, size[0], 3), 255, np.uint8), b])
            cv2.putText(col, f"wall {wi}", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)
            cols += [col, np.full((col.shape[0], 8, 3), 255, np.uint8)]
        cv2.imwrite(str(out / f"{slug}-facades.jpg"), np.hstack(cols)[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 88])
        files.append(f"{slug}-facades.jpg")
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
    ap.add_argument("--snap-from", type=Path, help="reuse the footprint snap of an earlier version (<slug>.json or .glb)")
    ap.add_argument("--no-debug", action="store_true", help="skip orthophoto / top-down images")
    args = ap.parse_args(argv)
    check_outside_repo(args.out)
    if args.job_dir and not args.meta:
        ap.error("--job-dir needs --meta")

    t0 = time.perf_counter()
    entry = catalog_entry(args.catalog, args.landmark)
    enu = LocalENU(entry["lat"], entry["lon"])
    osm = entry.get("osm") or {}
    fp_file = REPO / osm["file"] if osm.get("file") else None
    footprint, holes = None, []
    if fp_file is not None and fp_file.is_file():
        try:
            footprint, holes = load_footprint_rings(fp_file, enu)
        except ValueError as e:
            print(f"warning: {e}; the outline will be traced from the points", file=sys.stderr)
    else:
        print(f"warning: no OSM footprint for {args.landmark}; the outline will be traced from the points", file=sys.stderr)
    prior = None
    if args.scene:
        points, colors, cams, source, quality = load_megadepth_scene(args.scene, entry)
    else:
        points, colors, cams, source, quality = load_job(args.job_dir, args.meta)
        prior = snap_prior(args.meta)
    load_s = time.perf_counter() - t0

    attribution = ([osm.get("attribution", "© OpenStreetMap contributors (ODbL)")] if footprint is not None else []) + [
        "Facade colours derived from Wikimedia Commons photos (per-photo licences; see the scene's licence table)"]
    scene = st.SceneInput(entry.get("name", args.landmark), footprint, points, colors, cams,
                          facts_from_entry(entry, fp_file if footprint is not None else None), attribution,
                          snap=not args.no_snap, holes=tuple(holes),
                          snap_override=snap_from(args.snap_from) if args.snap_from else None, point_quality=quality,
                          snap_prior=prior)
    try:
        model = st.stylize_landmark(scene)
    except st.StylizeError as e:
        print(f"{args.landmark}: refused: {e}", file=sys.stderr)
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    slug = args.landmark
    (args.out / f"{slug}.glb").write_bytes(model.glb)
    summary = dict(model.summary)
    summary["source"] = source
    summary["timings_s"] = {"load_and_georeference": round(load_s, 2), **summary["timings_s"]}
    summary["total_s"] = round(load_s + model.summary["total_s"], 2)
    if not args.no_debug:
        outline = footprint if footprint is not None else model.massing.prisms[0].polygon
        summary["debug_files"] = save_debug(model, args.out, slug, outline)
    (args.out / f"{slug}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=_json_default),
                                           encoding="utf-8")
    s = model.summary
    print(f"{slug}: {s['observed_walls']}/{s['num_walls']} walls observed "
          f"({s['observed_wall_area_fraction']:.0%} of wall area, {s['observed_pixel_area_fraction']:.0%} pixels), "
          f"{s['observed_fill']['area_m2']:.1f} m2 observed-fill in {s['observed_fill']['holes']} holes, "
          f"roof {s['roof_observed_area_fraction']:.0%} observed, registration "
          f"{'ok' if s['registration']['accepted'] else 'REJECTED (clay only): ' + str(s['registration'].get('reason'))}, "
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
