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

Structure first: when the landmark has an OSM parts snapshot (the catalog's osm_parts block, eval/osm/<slug>.parts.osm;
--osm-parts FILE for another one) the model goes through crowd3d.structure_pipeline: the OSM building:parts are the
massing and, when the photos' registration is accepted, they are judged with the A+R sky mask, the SfM points and the
3-D lines (D15); D25 runs per --d25 (auto: only with two licensed sky masks, so not today; survey: measured, never
applied; off). The summary's "structure_run" says what ran and why. --no-structure keeps the footprint massing. The
photo cache (sky masks, line segments) is <out>/cache.

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
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from crowd3d import structure_pipeline as SP
from crowd3d import stylize as st
from crowd3d.geo import LocalENU
from crowd3d.reconstruct import apply_similarity
from crowd3d.scene_io import (  # noqa: F401  (re-exported: drivers and tests import them from here)
    _num,
    camera_from_colmap,
    facts_from_entry,
    load_footprint_rings,
    load_job,
    point_quality,
    point_tracks,
    snap_from,
    snap_prior,
)


def load_megadepth_scene(scene: Path, entry: dict[str, Any], with_tracks: bool = False) -> tuple:
    """(points, colors, cameras, source, point quality) of a MegaDepth-X scene in the landmark ENU frame; with_tracks:
    plus the points' observations (crowd3d.relief.PointTracks, for the facade relief's split-half check)."""
    import eval_real_landmark as ev
    import pycolmap

    gt = ev.load_ground_truth(scene)
    frame = ev.georeference_gt(gt, entry["lat"], entry["lon"], 1.6)
    rec = pycolmap.Reconstruction(str(scene / "sparse"))
    pts3d = list(rec.points3D.values())
    points = apply_similarity(frame.transform, np.array([p.xyz for p in pts3d]).reshape(-1, 3))
    colors = np.array([p.color for p in pts3d], dtype=np.uint8).reshape(-1, 3)
    r_enu = frame.rotation
    cams, cam_of = [], {}
    for iid, im in sorted(rec.images.items(), key=lambda kv: kv[1].name):
        if not im.has_pose:
            continue
        path = scene / "images" / im.name
        if not path.is_file():
            continue
        rot = im.cam_from_world().rotation.matrix() @ r_enu.T          # stored pixels' frame, ENU
        center = apply_similarity(frame.transform, np.asarray(im.projection_center())[None])[0]
        cam_of[iid] = len(cams)
        cams.append(camera_from_colmap(im.name, rec.cameras[im.camera_id], rot, center, path))
    report = {k: v for k, v in frame.report.items() if k != "per_photo"}
    out = (points, colors, cams, {"georeference": "GT reference model + real EXIF GPS/compass (eval_real_landmark)",
                                  "gt_frame": report}, point_quality(rec, len(points)))
    return out + (point_tracks(rec, cam_of),) if with_tracks else out


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
    ap.add_argument("--no-relief", action="store_true", help="do not measure / build facade relief (crowd3d.relief)")
    ap.add_argument("--no-structure", action="store_true",
                    help="footprint massing even when the landmark has an OSM parts snapshot")
    ap.add_argument("--osm-parts", type=Path, help="this OSM parts snapshot instead of the catalog's")
    ap.add_argument("--d25", choices=SP.D25_MODES, default="auto",
                    help="D25 frame corrections: auto (only with two licensed sky masks), survey (never applied), off")
    args = ap.parse_args(argv)
    import eval_real_landmark as ev

    ev.check_outside_repo(args.out, "--out")
    if args.job_dir and not args.meta:
        ap.error("--job-dir needs --meta")
    if args.no_structure and args.osm_parts:
        ap.error("--osm-parts is the structure's snapshot: not with --no-structure")

    t0 = time.perf_counter()
    entry = ev.catalog_entry(args.catalog, args.landmark)
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
        points, colors, cams, source, quality, tracks = load_megadepth_scene(args.scene, entry, with_tracks=True)
    else:
        points, colors, cams, source, quality, tracks = load_job(args.job_dir, args.meta, with_tracks=True)
        prior = snap_prior(args.meta)
    load_s = time.perf_counter() - t0

    attribution = ([osm.get("attribution", "© OpenStreetMap contributors (ODbL)")] if footprint is not None else []) + [
        "Facade colours derived from Wikimedia Commons photos (per-photo licences; see the scene's licence table)"]
    scene = st.SceneInput(entry.get("name", args.landmark), footprint, points, colors, cams,
                          facts_from_entry(entry, fp_file if footprint is not None else None), attribution,
                          snap=not args.no_snap, holes=tuple(holes),
                          snap_override=snap_from(args.snap_from) if args.snap_from else None, point_quality=quality,
                          snap_prior=prior, point_tracks=None if args.no_relief else tracks)
    structure = None
    if not args.no_structure:
        structure = SP.find_source(args.landmark, entry["lat"], entry["lon"], catalog=args.catalog,
                                   osm_parts=args.osm_parts, outline=fp_file if footprint is not None else None)
        if structure is None and args.osm_parts:
            ap.error("--osm-parts needs the landmark's OSM outline in the catalog")
    try:
        model, _ = SP.run(scene, structure, SP.PhotoCache(args.out / "cache"), SP.Limits(d25=args.d25),
                          st.DEFAULT_PARAMS if not args.no_relief else replace(st.DEFAULT_PARAMS, relief_detail=False))
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
    (args.out / f"{slug}.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False, default=ev._json_default),
                                           encoding="utf-8")
    st.write_osm_fix_report(summary, args.out, slug)          # D25's suggested OSM fixes (a report; OSM is never edited)
    s = model.summary
    print(f"{slug}: {s['observed_walls']}/{s['num_walls']} walls observed "
          f"({s['observed_wall_area_fraction']:.0%} of wall area, {s['observed_pixel_area_fraction']:.0%} pixels), "
          f"{s['observed_fill']['area_m2']:.1f} m2 observed-fill in {s['observed_fill']['holes']} holes, "
          f"roof {s['roof_observed_area_fraction']:.0%} observed, registration "
          f"{'ok' if s['registration']['accepted'] else 'REJECTED (clay only): ' + str(s['registration'].get('reason'))}, "
          f"{s['triangles']} triangles, GLB {s['glb_bytes'] / 1024:.0f} KB, QA {'ok' if s['qa']['passed'] else 'FAILED'}, "
          f"{summary['total_s']:.1f} s -> {args.out / (slug + '.glb')}")
    run = s.get("structure_run")
    if run:
        judge, over = run["judge"], run["limits"]["over"]
        print(f"{slug}: structure {run['parts_built']} OSM parts, judge {judge['status']}"
              f"{' (' + judge['reason'] + ')' if judge.get('reason') else ''}, "
              f"{sum(a['active'] for a in run['adoptions'])} photo-measured adoptions, D25 {run['d25']['mode']}: "
              f"{run['d25'].get('why') or len(run['d25']['records'])}, {run['total']['seconds']:.0f} s, "
              f"{run['total']['peak_rss_mb']:.0f} MB{', OVER ' + '; '.join(over) if over else ''}")
    return 0 if s["qa"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
