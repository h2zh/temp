"""Export a COLMAP model and its images as a plain dataset for Gaussian-splatting trainers.

Output: <out>/images/ and <out>/sparse/0/{cameras,images,points3D}.txt in the classic COLMAP text format
with PINHOLE cameras, which Brush, gsplat, nerfstudio, OpenSplat and Postshot all read. Input is anything
pycolmap reads: a Crowd3D job's work_dir/sparse/0 (kept with Settings.keep_work_dirs), a MegaDepth-X scene
component (COLMAP 4 binary with rigs/frames), or COLMAP GUI output. Distorted cameras are undistorted
first; images are optionally downscaled (intrinsics and 2-D observations scaled to match) and written
without EXIF, so no trainer rotates them a second time.

    python scripts/export_for_splatting.py --sparse SCENE/1/sparse --images SCENE/1/images --out splat-data
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

PINHOLE_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


def quat_wxyz(r: np.ndarray) -> np.ndarray:
    """Unit quaternion (w, x, y, z) of a rotation matrix (COLMAP's order), w >= 0."""
    m = np.asarray(r, dtype=float)
    t = np.trace(m)
    if t > 0:
        s = 2.0 * np.sqrt(t + 1.0)
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(m)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2.0 * np.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k])
        q = [0.0, 0.0, 0.0, 0.0]
        q[0] = (m[k, j] - m[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (m[j, i] + m[i, j]) / s
        q[1 + k] = (m[k, i] + m[i, k]) / s
    q = np.array(q)
    q /= np.linalg.norm(q)
    return q if q[0] >= 0 else -q


def _cam_from_world(image):
    pose = image.cam_from_world
    return pose() if callable(pose) else pose


def _model_name(camera) -> str:
    return str(camera.model).split(".")[-1]


def _pinhole(camera) -> tuple[float, float, float, float]:
    p = list(camera.params)
    if _model_name(camera) == "SIMPLE_PINHOLE":
        return p[0], p[0], p[1], p[2]
    return p[0], p[1], p[2], p[3]


def export(sparse: Path, images: Path, out: Path, max_side: int = 1600, max_images: int | None = None) -> dict:
    """Write the dataset; returns counts and the mean reprojection error (px) of the exported model."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse))
    with tempfile.TemporaryDirectory() as tmp:
        if any(_model_name(c) not in PINHOLE_MODELS for c in rec.cameras.values()):
            pycolmap.undistort_images(tmp, str(sparse), str(images))
            rec = pycolmap.Reconstruction(str(Path(tmp) / "sparse"))
            images = Path(tmp) / "images"
        return _write(rec, images, out, max_side, max_images)


def _write(rec, images: Path, out: Path, max_side: int, max_images: int | None) -> dict:
    img_ids = sorted(i for i, im in rec.images.items() if im.has_pose)
    if max_images is not None and len(img_ids) > max_images:
        # Keep the images that observe the most 3-D points: they carry the scene.
        img_ids = sorted(sorted(img_ids, key=lambda i: -rec.images[i].num_points3D)[:max_images])
    keep = set(img_ids)
    # Points observed by at least two kept images; observations of any other point are written as -1.
    tracks = {}
    for pid, pt in rec.points3D.items():
        track = [(el.image_id, el.point2D_idx) for el in pt.track.elements if el.image_id in keep]
        if len(track) >= 2:
            tracks[pid] = track
    if out.exists():
        shutil.rmtree(out)
    (out / "sparse" / "0").mkdir(parents=True)
    (out / "images").mkdir(parents=True)

    scale: dict[int, tuple[float, float]] = {}
    with open(out / "sparse/0/cameras.txt", "w") as f:
        f.write("# CAMERA_ID MODEL WIDTH HEIGHT fx fy cx cy\n")
        for cid, cam in sorted(rec.cameras.items()):
            s = min(1.0, max_side / max(cam.width, cam.height))
            w, h = max(1, round(cam.width * s)), max(1, round(cam.height * s))
            sx, sy = w / cam.width, h / cam.height
            scale[cid] = (sx, sy)
            fx, fy, cx, cy = _pinhole(cam)
            # COLMAP's image origin is the corner of the first pixel, so intrinsics scale linearly.
            f.write(f"{cid} PINHOLE {w} {h} {fx * sx:.6f} {fy * sy:.6f} {cx * sx:.6f} {cy * sy:.6f}\n")

    with open(out / "sparse/0/images.txt", "w") as f:
        f.write("# IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        for iid in img_ids:
            im = rec.images[iid]
            pose = _cam_from_world(im)
            q = quat_wxyz(pose.rotation.matrix())
            t = np.asarray(pose.translation, dtype=float)
            sx, sy = scale[im.camera_id]
            f.write(f"{iid} {' '.join(f'{v:.9f}' for v in q)} {' '.join(f'{v:.6f}' for v in t)} "
                    f"{im.camera_id} {im.name}\n")
            obs = []
            for p in im.points2D:
                pid = p.point3D_id if p.has_point3D() and p.point3D_id in tracks else -1
                obs.append(f"{p.xy[0] * sx:.3f} {p.xy[1] * sy:.3f} {pid}")
            f.write(" ".join(obs) + "\n")
            cam = rec.cameras[im.camera_id]
            with Image.open(images / im.name) as src:
                if src.size != (cam.width, cam.height):
                    raise ValueError(f"{im.name}: pixels are {src.size}, camera says {(cam.width, cam.height)}")
                dst = out / "images" / im.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                w, h = round(cam.width * sx), round(cam.height * sy)
                pic = src.convert("RGB")      # raw pixel order: COLMAP never applies EXIF orientation
                if (w, h) != src.size:
                    pic = pic.resize((w, h), Image.Resampling.LANCZOS)
                pic.save(dst, "JPEG", quality=95)   # no EXIF carried over

    with open(out / "sparse/0/points3D.txt", "w") as f:
        f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        for pid, track in sorted(tracks.items()):
            pt = rec.points3D[pid]
            x, y, z = pt.xyz
            r, g, b = (int(c) for c in pt.color)
            f.write(f"{pid} {x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {pt.error:.4f} "
                    + " ".join(f"{i} {k}" for i, k in track) + "\n")

    return {"images": len(img_ids), "points": len(tracks), "reprojection_error_px": _check(out)}


def _check(out: Path) -> float:
    """Re-read the export and measure how well its points reproject into its own (scaled) images."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(out / "sparse" / "0"))
    errs = []
    for im in rec.images.values():
        cam = rec.cameras[im.camera_id]
        pose = _cam_from_world(im)
        r, t = pose.rotation.matrix(), np.asarray(pose.translation)
        fx, fy, cx, cy = _pinhole(cam)
        for p in im.points2D:
            if not p.has_point3D():
                continue
            xc = r @ rec.points3D[p.point3D_id].xyz + t
            if xc[2] <= 0:
                continue
            uv = np.array([fx * xc[0] / xc[2] + cx, fy * xc[1] / xc[2] + cy])
            errs.append(float(np.linalg.norm(uv - p.xy)))
    return float(np.mean(errs)) if errs else float("nan")


NEXT_STEPS = """
Next: train a Gaussian splat on {out}
  Brush (macOS / Windows / Linux, any modern GPU): open Brush and load the folder {out}
  gsplat (NVIDIA + CUDA):  python examples/simple_trainer.py default --data_dir {out} --data_factor 1 --result_dir results
  nerfstudio (NVIDIA):     ns-train splatfacto colmap --data {out} --colmap-path sparse/0
  Postshot (Windows + NVIDIA): import the images together with the COLMAP poses in {out}/sparse/0
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sparse", type=Path, required=True, help="COLMAP model folder (binary or text)")
    ap.add_argument("--images", type=Path, required=True, help="folder the model's image names are relative to")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-side", type=int, default=1600, help="downscale longer image side to this (VRAM)")
    ap.add_argument("--max-images", type=int, default=None, help="keep only the N best-connected images")
    args = ap.parse_args(argv)
    stats = export(args.sparse, args.images, args.out, args.max_side, args.max_images)
    print(f"exported {stats['images']} images, {stats['points']} points, "
          f"reprojection error {stats['reprojection_error_px']:.2f} px -> {args.out}")
    print(NEXT_STEPS.format(out=args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
