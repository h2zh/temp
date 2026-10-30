"""Loaders of a landmark's stylization inputs (moved from scripts/stylize_model.py, which re-exports them): a Crowd3D
job's kept SfM model and model JSON in the landmark ENU frame (load_job, snap_prior, model_anchor), COLMAP cameras
(camera_from_colmap), point quality and tracks (point_quality, point_tracks), the OSM outline with its courtyards
(load_footprint_rings), height facts (facts_from_entry) and an earlier version's footprint snap (snap_from)."""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from . import stylize as st
from .anchor import GeoAnchor, read_anchor
from .geo import LocalENU
from .reconstruct import apply_similarity, photo_id_from_image_name, umeyama


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


def point_tracks(rec, camera_of_image: dict[int, int]):
    """The points' 2-D observations (crowd3d.relief.PointTracks), points in rec.points3D.values() order."""
    from .relief import PointTracks
    return PointTracks.from_colmap(rec, camera_of_image)


ANCHOR_MAX_CAMERA_DIFF_M = 0.05   # load_job with an anchor: it must put every camera this close to the meta's ENU


def model_anchor(meta_path: Path) -> GeoAnchor | None:
    """The GeoAnchor of a model JSON with the current calibration floors (GeoAnchor.recalibrated); None for a model
    recorded before anchors existed."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    a = read_anchor(meta.get("meta", meta))
    return None if a is None else a.recalibrated()


def load_job(job_dir: Path | None, meta_path: Path, with_tracks: bool = False, anchor: GeoAnchor | None = None, *,
             sparse_dir: Path | None = None, image_path_of: Callable[[str], Path | None] | None = None) -> tuple:
    """(points, colours, cameras, source, point quality) of a Crowd3D job in the landmark ENU frame: the similarity
    to ENU from the meta's camera positions, or the anchor's transform (StyleParams.snap_use_anchor: the model's
    GeoAnchor, whose local frame is this job's sparse/0; it must reproduce the meta's camera positions within
    ANCHOR_MAX_CAMERA_DIFF_M); cameras the meta flags are left out (they colour nothing, and their observations are
    left out of the tracks too). with_tracks: plus the points' observations (crowd3d.relief.PointTracks).
    The job's files: job_dir/sparse/0 and job_dir/images/<image name> (a kept work dir), or sparse_dir and
    image_path_of(image name) (a saved map and the photos' processed copies; None: no photo for that image)."""
    import pycolmap

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = meta.get("meta", meta)
    rec = pycolmap.Reconstruction(str(sparse_dir if sparse_dir is not None else job_dir / "sparse" / "0"))
    src, dst = [], []
    for im in rec.images.values():
        cam = meta.get("cameras", {}).get(photo_id_from_image_name(im.name))
        if im.has_pose and cam and cam.get("enu") is not None:
            src.append(np.asarray(im.projection_center()))
            dst.append(np.asarray(cam["enu"], float))
    if len(src) < 3:
        raise SystemExit("the meta file has fewer than 3 camera positions for this reconstruction")
    if anchor is not None:
        transform = anchor.matrix()
        rot = anchor.rotation()
    else:
        scale, rot, trans = umeyama(np.array(src), np.array(dst), with_scale=True)
        transform = np.eye(4)
        transform[:3, :3] = scale * rot
        transform[:3, 3] = trans
    resid = np.linalg.norm(apply_similarity(transform, np.array(src)) - np.array(dst), axis=1)
    if anchor is not None and resid.max() > ANCHOR_MAX_CAMERA_DIFF_M:
        raise SystemExit(f"the GeoAnchor puts a camera {resid.max():.3f} m from the meta's ENU position: it does not "
                         "belong to this reconstruction")
    pts3d = list(rec.points3D.values())
    points = apply_similarity(transform, np.array([p.xyz for p in pts3d]).reshape(-1, 3))
    colors = np.array([p.color for p in pts3d], dtype=np.uint8).reshape(-1, 3)
    cams, cam_of, flagged = [], {}, []
    for iid, im in sorted(rec.images.items(), key=lambda kv: kv[1].name):
        path = image_path_of(im.name) if image_path_of is not None else job_dir / "images" / im.name
        if not im.has_pose or path is None or not Path(path).exists():
            continue
        pid = photo_id_from_image_name(im.name)
        if (meta.get("cameras", {}).get(pid) or {}).get("flags"):
            # withheld from users (disagrees with its own GPS / compass, outside the radius, degenerate focal):
            # its pose is not trusted, so it colours nothing (Frauenkirche SW#14: one 154 m off fed 3 walls)
            flagged.append(pid)
            continue
        r = im.cam_from_world().rotation.matrix() @ rot.T
        center = apply_similarity(transform, np.asarray(im.projection_center())[None])[0]
        cam_of[iid] = len(cams)
        cams.append(camera_from_colmap(im.name, rec.cameras[im.camera_id], r, center, path, photo_id=pid))
    how = ("Crowd3D georegister (the model's GeoAnchor)" if anchor is not None
           else "Crowd3D georegister (recovered from the meta's camera ENU)")
    out = (points, colors, cams, {"georeference": how,
                                  "camera_fit_residual_max_m": round(float(resid.max()), 4),
                                  "flagged_cameras_left_out": sorted(flagged)}, point_quality(rec, len(points)))
    return out + (point_tracks(rec, cam_of),) if with_tracks else out


def snap_prior(meta_path: Path, anchor: GeoAnchor | None = None) -> st.SnapPrior:
    """The georegistration's yaw sigma and the cameras' own GPS from a model JSON, for the snap's yaw window and
    GPS check, the unflagged cameras' 3-D centres for the tilt and visibility checks, and whether it is
    georegistered at all (STYLIZED_MODEL 3.2: a prior-only model colours nothing; a model JSON that does not
    say counts as prior only). Flagged cameras (they disagree with their own sensors) are left out; a model
    written before the sigma fields existed gives the default window (refused with
    StyleParams.snap_refuse_clipped_window) and no GPS check. anchor: the model's GeoAnchor (model_anchor), attached
    for StyleParams.snap_use_anchor."""
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta = meta.get("meta", meta)
    unflagged = [c for c in (meta.get("cameras") or {}).values()
                 if isinstance(c, dict) and not c.get("flags") and c.get("enu") is not None and len(c["enu"]) == 3]
    cams = [c for c in unflagged if c.get("gps_en") is not None]
    sigma, scale_sigma = meta.get("yaw_sigma_deg"), meta.get("scale_sigma")
    xy = np.array([c["enu"][:2] for c in cams], float).reshape(-1, 2) if cams else None
    gps = np.array([c["gps_en"] for c in cams], float).reshape(-1, 2) if cams else None
    xyz = np.array([c["enu"] for c in unflagged], float).reshape(-1, 3) if unflagged else None
    return st.SnapPrior(float(sigma) if sigma is not None else None, xy, gps,
                        float(scale_sigma) if scale_sigma is not None else None,
                        georegistered=bool(meta.get("georegistered", False)), camera_xyz=xyz, anchor=anchor)
