"""Reusable benchmark: frozen test sets x pipeline configs -> committed results, comparisons, a leaderboard.

    python -I scripts/benchmark.py prepare --testset synthetic-v1           # materialise generated/MegaDepth-X sets
    python -I scripts/benchmark.py run --testset synthetic-v1 --config default --name default-1 \\
        [--buildings slug ...] [--repeat 2] [--time-cap 3600]
    python -I scripts/benchmark.py compare eval/results/synthetic-v1/default-1.json \
        eval/results/synthetic-v1/default-2.json [--noise <result with a noise block>]
    python -I scripts/benchmark.py report                                   # regenerates eval/results/LEADERBOARD.md
    python -I scripts/benchmark.py freeze-synthetic --testset-id synthetic-v2   # maintainers: a new set version
    python -I scripts/benchmark.py freeze-mdx --testset-id mdx-two-v2 --building florence-baptistery=<component dir> ...

Test sets: eval/testsets/<testset_id>/testset.json (metadata only; a released id is immutable). Photos are
materialised under --data-root/<testset_id>/<slug>/<photo file> (scripts/materialize_testset.py for Commons
sets; `prepare` for the generated synthetic set and MegaDepth-X photos), reference models under
--data-root/mdx/<slug>/<component>/sparse. Before a building runs, every listed file is hashed: "ok",
"changed" or "missing" is recorded; only ok photos are uploaded unless the set (or --on-changed use) says
otherwise, and inputs_sha256 fingerprints exactly what was used, so compare can tell runs on different
inputs apart.

Configs: eval/configs/<config_id>.json = reconstructor + Settings overrides + params.Params overrides +
harness options (scripts/eval_real_landmark.load_pipeline_config checks every name against the
whitelists). A run records the full effective parameter set.

Each building runs in its own process (scripts/eval_real_landmark.py --building-spec) under a hard time
cap; heavy artefacts (data dir, PLY, plan view, markdown, stylized GLBs) stay under --work-root and are
referenced by path. By default every published model is also stylized (crowd3d.stylize, as
scripts/stylize_model.py --job-dir does, with Settings.keep_work_dirs on): the style_* metrics, including
the registration gate's pass rate; --no-stylize skips it (style_* metrics then stay empty). The
committed result is eval/results/<testset_id>/<run_name>.json: provenance, config, effective parameters,
integrity, per-building metrics (eval_real_landmark.METRICS) and their aggregate. --repeat N runs every
building N times and stores the spread (run-to-run noise) that compare uses to flag only
changes beyond noise. --resume reuses a building run left by an interrupted invocation when its run_key.json
says it ran the same code (code_sha256 over CODE_PATHS), config, test set, inputs and options.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_real_landmark as ev  # noqa: E402

from crowd3d.exif import read_photo_meta  # noqa: E402

log = logging.getLogger("benchmark")

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "eval_real_landmark.py"
TESTSETS_DIR = REPO_ROOT / "eval" / "testsets"
CONFIGS_DIR = REPO_ROOT / "eval" / "configs"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
_SCRATCH = Path("/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad/ten")
DEFAULT_DATA_ROOT = Path(os.environ.get("CROWD3D_TESTSET_ROOT", _SCRATCH / "testsets"))
DEFAULT_WORK_ROOT = Path(os.environ.get("CROWD3D_BENCH_DIR", _SCRATCH / "bench"))
RESULT_SCHEMA_VERSION = 1
TESTSET_SCHEMA_VERSION = 1
DEFAULT_TIME_CAP_S = 3600.0
SOFT_CAP_FRACTION = 0.8          # the harness starts no new upload request after this share of the hard cap
STATUS_OK, STATUS_CHANGED, STATUS_MISSING = "ok", "changed", "missing"
SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")
USER_AGENT = "Crowd3D-prototype/0.1 (https://github.com/h2zh/temp; research evaluation)"
CODE_PATHS = ("crowd3d", "scripts", "eval/configs")    # what a building run executes or reads (code_sha256)
RUN_KEY_FILE = "run_key.json"


# ---------------------------------------------------------------- small helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def jpeg_exif_segment(data: bytes) -> bytes | None:
    """Payload of the first APP1 "Exif" segment of a JPEG (None without one)."""
    if data[:2] != b"\xff\xd8":
        return None
    k = 2
    while k + 4 <= len(data) and data[k] == 0xFF:
        marker = data[k + 1]
        if marker in (0xD9, 0xDA):          # end of image / start of scan: no more metadata
            return None
        length = int.from_bytes(data[k + 2:k + 4], "big")
        payload = data[k + 4:k + 2 + length]
        if marker == 0xE1 and payload.startswith(b"Exif\0\0"):
            return payload
        k += 2 + length
    return None


def canonical_sha256(obj: Any) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def _round(v: Any, nd: int = 4) -> Any:
    return round(float(v), nd) if isinstance(v, (int, float)) and not isinstance(v, bool) else v


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds")


def _repo_relative(path: Path) -> str:
    """path relative to the repository when inside it, else absolute."""
    return str(path.resolve().relative_to(REPO_ROOT)) if ev._inside(path, REPO_ROOT) else str(path.resolve())


def check_outside_repo(path: Path, what: str) -> None:
    ev.check_outside_repo(path, what)


# ---------------------------------------------------------------- test sets


def testset_path(ref: str | Path) -> Path:
    """A registered id ('synthetic-v1'), a testset.json path or its folder -> the testset.json path."""
    p = Path(ref)
    if p.is_file():
        return p
    if (p / "testset.json").is_file():
        return p / "testset.json"
    q = TESTSETS_DIR / str(ref) / "testset.json"
    if q.is_file():
        return q
    raise SystemExit(f"no test set {ref!r} (looked for {p} and {q})")


def load_testset(ref: str | Path) -> tuple[dict[str, Any], Path, str]:
    """(test set, testset.json path, sha256 of its bytes: the same definition as materialize_testset.py)."""
    path = testset_path(ref)
    raw = path.read_bytes()
    ts = json.loads(raw)
    if ts.get("schema_version") != TESTSET_SCHEMA_VERSION:
        raise SystemExit(f"{path}: schema_version {ts.get('schema_version')!r}, "
                         f"this tool reads {TESTSET_SCHEMA_VERSION}")
    return ts, path, sha256_bytes(raw)


def building_centre(b: dict[str, Any]) -> tuple[float, float]:
    c = b.get("centre") or b.get("center") or b
    return float(c["lat"]), float(c["lon"])


def building_radius(b: dict[str, Any]) -> float:
    return float(b.get("radius_m") or b.get("radius") or 150.0)


def building_outline(b: dict[str, Any], ts_dir: Path) -> tuple[Path | None, str | None]:
    o = b.get("osm_outline") or b.get("outline")
    if not o:
        return None, None
    return ts_dir / o["file"], o.get("sha256")


def photo_key(p: dict[str, Any]) -> str:
    return str(p.get("key") or p.get("file_name") or Path(p["file"]).name)


def photo_sha(p: dict[str, Any]) -> str | None:
    return p.get("sha256") or p.get("sha256_file")


def select_buildings(ts: dict[str, Any], slugs: list[str] | None) -> list[dict[str, Any]]:
    if not slugs:
        return list(ts["buildings"])
    known = {b["slug"]: b for b in ts["buildings"]}
    unknown = sorted(set(slugs) - set(known))
    if unknown:
        raise SystemExit(f"{ts['testset_id']} has no building(s) {unknown}; it has {sorted(known)}")
    return [known[s] for s in slugs]


def verify_photos(b: dict[str, Any], bdir: Path) -> dict[str, Any]:
    """Per listed photo: ok (sha256 matches), changed or missing; plus counts."""
    rows = []
    for p in b["photos"]:
        path = bdir / p["file"]
        want = photo_sha(p)
        if not path.is_file():
            status, got = STATUS_MISSING, None
        else:
            got = sha256_file(path)
            status = STATUS_OK if want is None or got == want else STATUS_CHANGED
        rows.append({"key": photo_key(p), "path": str(path), "status": status, "sha256": got, "sha256_expected": want})
    counts = Counter(r["status"] for r in rows)
    return {"photos": len(rows), **{s: counts.get(s, 0) for s in (STATUS_OK, STATUS_CHANGED, STATUS_MISSING)},
            "changed_keys": [r["key"] for r in rows if r["status"] == STATUS_CHANGED],
            "missing_keys": [r["key"] for r in rows if r["status"] == STATUS_MISSING], "rows": rows}


def _expected_sha(entry: Any) -> str | None:
    return entry.get("sha256") if isinstance(entry, dict) else entry


def gt_reference(b: dict[str, Any]) -> dict[str, Any] | None:
    return b.get("gt_reference") or b.get("gt")


def _mdx_image_name(entry: dict[str, Any]) -> str | None:
    """The exact images.bin name of a reference entry (may contain sub-folders): "name", else the part of
    name_in_megadepth_x after "<scene>/<component>/images/"."""
    if entry.get("name"):
        return str(entry["name"])
    full = entry.get("name_in_megadepth_x")
    return full.split("/images/", 1)[1] if full and "/images/" in full else full


def gt_image_maps(b: dict[str, Any], ref: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    """component -> photo key -> {"name": exact reference image name, "orientation": that image's EXIF orientation}.

    Sources: gt_reference.images (an entry's "component" routes it, default the primary component), each
    additional component's own "images", and per photo gt_image / gt_orientation (primary) or gt_additional
    ({component, name, orientation}, or a list of them).
    """
    primary = str(ref.get("component", ""))
    maps: dict[str, dict[str, dict[str, Any]]] = {primary: {}}

    def add(comp: str, key: str, entry: Any) -> None:
        entry = dict(entry) if isinstance(entry, dict) else {"name": entry}
        name = _mdx_image_name(entry)
        if name:
            maps.setdefault(comp, {}).setdefault(key, {"name": name, "orientation": entry.get("orientation")})

    for key, entry in (ref.get("images") or {}).items():
        add(str(entry.get("component", primary)) if isinstance(entry, dict) else primary, key, entry)
    for extra in ref.get("additional_components") or []:
        for key, entry in (extra.get("images") or {}).items():
            add(str(extra.get("component", "")), key, entry)
    for p in b["photos"]:
        key = photo_key(p)
        if p.get("gt_image"):
            add(primary, key, {"name": p["gt_image"], "orientation": p.get("gt_orientation")})
        extra = p.get("gt_additional")
        for e in extra if isinstance(extra, list) else [extra] if extra else []:
            add(str(e.get("component", "")), key, e)
    return maps


def resolve_gt(b: dict[str, Any], bdir: Path, data_root: Path, gt_root: Path | None) -> tuple[dict | None, dict]:
    """(reference spec for the harness or None, integrity of the reference files per component).

    A component whose sparse files are missing or changed is left out (and recorded); the others are used.
    """
    gen = b.get("generator")
    if gen is not None:
        manifest_path = bdir / "manifest.json"
        if not manifest_path.is_file():
            return None, {"status": STATUS_MISSING, "detail": "generator manifest.json missing"}
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return {"kind": "poses", "poses": ev.synthetic_gt_poses(manifest)}, {"status": STATUS_OK, "kind": "synthetic"}
    ref = gt_reference(b)
    if not ref:
        return None, {"status": "none"}
    maps = gt_image_maps(b, ref)
    components, integrity = [], {"components": {}}
    for comp_ref in [ref, *(ref.get("additional_components") or [])]:
        comp = str(comp_ref.get("component", ""))
        candidates = [bdir / "gt" / comp / "sparse", data_root / "mdx" / b["slug"] / comp / "sparse"]
        if gt_root is not None:
            candidates.insert(0, gt_root / b["slug"] / comp / "sparse")
        sparse = next((c for c in candidates if (c / "images.bin").is_file()), candidates[0])
        files = {}
        for name, want in (comp_ref.get("sparse_files") or {}).items():
            path = sparse / name
            if not path.is_file():
                files[name] = STATUS_MISSING
            else:
                files[name] = STATUS_OK if _expected_sha(want) in (None, sha256_file(path)) else STATUS_CHANGED
        if not files:
            files["images.bin"] = STATUS_OK if (sparse / "images.bin").is_file() else STATUS_MISSING
        status = (STATUS_MISSING if STATUS_MISSING in files.values() else
                  STATUS_CHANGED if STATUS_CHANGED in files.values() else STATUS_OK)
        integrity["components"][comp] = {"status": status, "files": files, "mapped_photos": len(maps.get(comp, {}))}
        if status == STATUS_OK and maps.get(comp):
            components.append({"component": comp, "sparse_dir": str(sparse), "images": maps[comp]})
    states = [c["status"] for c in integrity["components"].values()]
    integrity["status"] = (STATUS_OK if all(x == STATUS_OK for x in states) else
                           STATUS_MISSING if STATUS_MISSING in states else STATUS_CHANGED)
    integrity["files"] = {c: v["files"] for c, v in integrity["components"].items()}
    return ({"kind": "colmap", "components": components} if components else None), integrity


# ---------------------------------------------------------------- materialising generated / MegaDepth-X sets


def materialize_synthetic(b: dict[str, Any], bdir: Path) -> None:
    """Regenerate a crowd3d.synthetic building (deterministic for a seed) into bdir."""
    from crowd3d import synthetic

    gen = b["generator"]
    kwargs = dict(gen["kwargs"])
    if "size" in kwargs:
        kwargs["size"] = tuple(kwargs["size"])
    if "building" in kwargs:
        kwargs["building"] = synthetic.BuildingSpec(**kwargs["building"])
    lat, lon = building_centre(b)
    synthetic.generate_dataset(bdir, landmark_lat=lat, landmark_lon=lon, **kwargs)


def stream_extract_members(url: str, members: dict[str, Path]) -> list[str]:
    """Stream a .tar.gz once and write the wanted members (archive path -> target); returns those written."""
    written = []
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            target = members.get(member.name.lstrip("./"))
            if target is None or not member.isfile():
                continue
            src = tar.extractfile(member)
            if src is None:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            written.append(member.name)
    return written


def materialize_mdx(b: dict[str, Any], bdir: Path, local: Path | None) -> None:
    """Photos and reference model of a MegaDepth-X building: symlinks from a local extraction of its
    <scene>/<component> folder when given, else the listed archive members streamed from Hugging Face."""
    ref = gt_reference(b) or {}
    wanted: dict[str, Path] = {}
    for p in b["photos"]:
        src = p.get("source") or {}
        if src.get("kind") != "megadepth-x":
            continue
        target = bdir / p["file"]
        if target.is_file() and sha256_file(target) == photo_sha(p):
            continue
        if local is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.unlink(missing_ok=True)
            target.symlink_to((local / "images" / Path(src["member"]).name).resolve())
        else:
            wanted[src["member"]] = target
    comp = str(ref.get("component", ""))
    for name in ref.get("sparse_files") or {}:
        target = bdir / "gt" / comp / "sparse" / name
        if target.is_file():
            continue
        if local is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to((local / "sparse" / name).resolve())
        else:
            wanted[f"{ref['scene']}/{comp}/sparse/{name}"] = target
    if wanted:
        url = ref.get("archive_url") or next(p["source"]["archive_url"] for p in b["photos"] if p.get("source"))
        log.info("%s: streaming %d member(s) from %s", b["slug"], len(wanted), url)
        stream_extract_members(url, wanted)


def cmd_prepare(args: argparse.Namespace) -> int:
    ts, _, _ = load_testset(args.testset)
    check_outside_repo(args.data_root, "--data-root")
    local = dict(item.split("=", 1) for item in args.mdx_local or [])
    for b in select_buildings(ts, args.buildings):
        bdir = args.data_root / ts["testset_id"] / b["slug"]
        if b.get("generator"):
            if verify_photos(b, bdir)[STATUS_OK] < len(b["photos"]):
                materialize_synthetic(b, bdir)
        elif any((p.get("source") or {}).get("kind") == "megadepth-x" for p in b["photos"]):
            materialize_mdx(b, bdir, Path(local[b["slug"]]) if b["slug"] in local else None)
        else:
            print(f"{b['slug']}: Commons photos are materialised by scripts/materialize_testset.py "
                  f"{ts['testset_id']} --cache-dir {args.data_root} --gt")
            continue
        integ = verify_photos(b, bdir)
        _, gt_integ = resolve_gt(b, bdir, args.data_root, None)
        print(f"{b['slug']}: {integ[STATUS_OK]} ok, {integ[STATUS_CHANGED]} changed, {integ[STATUS_MISSING]} missing;"
              f" reference {gt_integ['status']} ({bdir})")
    return 0


# ---------------------------------------------------------------- run


def config_path(ref: str | Path) -> Path:
    p = Path(ref)
    if p.is_file():
        return p
    q = CONFIGS_DIR / f"{ref}.json"
    if q.is_file():
        return q
    raise SystemExit(f"no config {ref!r} (looked for {p} and {q})")


def git_state() -> dict[str, Any]:
    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True, check=False).stdout

    # The results ledger itself is not code: a run that rewrites earlier results does not make the tree dirty.
    ledger = "eval/results/"
    modified = [line[3:] for line in git("status", "--porcelain", "--untracked-files=no").splitlines()
                if line and not line[3:].startswith(ledger)]
    untracked = [line[3:] for line in git("status", "--porcelain").splitlines()
                 if line.startswith("??") and not line[3:].startswith(ledger)]
    diff = git("diff", "HEAD", "--", ".", f":(exclude){ledger}")
    return {"commit": git("rev-parse", "HEAD").strip() or None, "dirty": bool(modified), "modified": modified,
            "untracked": untracked, "diff_sha256": sha256_bytes(diff.encode()) if diff else None}


def code_fingerprint() -> str | None:
    """sha256 over the bytes of the files under CODE_PATHS (tracked or untracked, not ignored). Unlike commit +
    diff it stays the same when the same code gets committed, so --resume can rely on it."""
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z", "--", *CODE_PATHS], cwd=REPO_ROOT,
                         capture_output=True, check=False).stdout.decode()
    files = sorted({f for f in out.split("\0") if f and (REPO_ROOT / f).is_file()})
    return canonical_sha256({f: sha256_file(REPO_ROOT / f) for f in files}) if files else None


def run_key(spec: dict[str, Any], *, code_sha: str | None, testset_sha: str, cfg_sha: str, effective_sha: str,
            inputs_sha: str, stylize: bool, time_cap_s: float) -> dict[str, Any]:
    """What a building run depends on; --resume reuses a finished run only when its saved key is equal."""
    return {"code_sha256": code_sha, "harness_version": ev.HARNESS_VERSION,
            "metric_schema_version": ev.METRIC_SCHEMA_VERSION, "testset_sha256": testset_sha, "config_sha256": cfg_sha,
            "effective_sha256": effective_sha, "inputs_sha256": inputs_sha, "spec_sha256": canonical_sha256(spec),
            "stylize": stylize, "time_cap_s": time_cap_s}


def versions() -> dict[str, Any]:
    import cv2
    import PIL
    import pycolmap

    return {"python": platform.python_version(), "pycolmap": pycolmap.__version__, "opencv": cv2.__version__,
            "numpy": np.__version__, "pillow": PIL.__version__, "platform": platform.platform(),
            "cpu_count": os.cpu_count()}


def building_spec(b: dict[str, Any], ts_dir: Path, integrity: dict[str, Any], gt: dict | None,
                  use_changed: bool) -> dict[str, Any]:
    lat, lon = building_centre(b)
    outline, _ = building_outline(b, ts_dir)
    keep = {STATUS_OK, STATUS_CHANGED} if use_changed else {STATUS_OK}
    by_key = {photo_key(p): p for p in b["photos"]}
    photos = []
    for row in integrity["rows"]:
        if row["status"] not in keep:
            continue
        p = by_key[row["key"]]
        photos.append({"key": row["key"], "path": row["path"], "title": p.get("title"), "author": p.get("author"),
                       "licence": p.get("licence")})
    return {"slug": b["slug"], "name": b.get("name") or b["slug"], "lat": lat, "lon": lon,
            "radius_m": building_radius(b), "outline_path": str(outline) if outline else None, "photos": photos,
            "gt": gt, "upload_all": bool(b.get("upload_all", False))}


def inputs_fingerprint(integrity: dict[str, Any], use_changed: bool, outline_sha: str | None,
                       gt_integrity: dict[str, Any]) -> str:
    """sha256 over exactly what a building run consumed: used photos with their bytes' sha256, outline, reference."""
    keep = {STATUS_OK, STATUS_CHANGED} if use_changed else {STATUS_OK}
    used = sorted((r["key"], r["sha256"]) for r in integrity["rows"] if r["status"] in keep)
    return canonical_sha256({"photos": used, "outline": outline_sha, "gt": gt_integrity.get("files")})


def compact_details(summary: dict[str, Any]) -> dict[str, Any]:
    """The diagnostic part of a harness summary that is small enough to commit (no per-photo rows)."""
    models = []
    style_by_id = {r["model_id"]: r for r in ((summary.get("style") or {}).get("records") or [])}
    for m in summary["models"]:
        fit = (m.get("outline") or {}).get("fit") or {}
        gt = m.get("gt") or {}
        models.append({
            "kind": m["kind"], "sector": m["sector"], "request": m["request"], "photos": m["num_photos"],
            "registered": m["num_registered"], "points": m["num_points"], "published": m["published"],
            "shown": m["latest_published"], "qa_reasons": m["qa_reason_codes"], "yaw_source": m["yaw_source"],
            "georegistered": m["georegistered"], "flagged": m["flagged_cameras"],
            "outline_rotation_deg": fit.get("rotation_deg"), "outline_shift_m": fit.get("shift_m"),
            "outline_residual_m": fit.get("median_distance_after_m"), "outline_ambiguous": fit.get("ambiguous"),
            "gt_n": gt.get("n"), "gt_rotation_median_deg": (gt.get("rotation_error_deg") or {}).get("median"),
            "gt_position_median_m": (gt.get("position_error_m") or {}).get("median"), "gt_gross": gt.get("gross"),
            "gt_gross_shown": gt.get("gross_shown"), "seconds": m["job_seconds"],
        })
        if m["id"] in style_by_id:
            models[-1]["style"] = compact_style(style_by_id[m["id"]])
    f = summary["funnel"]
    return {
        "funnel": {k: f[k] for k in ("listed", "no_gps_in_file", "uploaded", "accepted", "rejected_by_reason",
                                     "with_heading", "magnetic_heading")},
        "sectors": {k: v["count"] for k, v in summary["sectors"].items() if v["count"]},
        "requests": len(summary["requests"]),
        "jobs": [{k: j[k] for k in ("kind", "sector", "request", "photos", "status", "error", "seconds")}
                 for j in summary["jobs"]],
        "models": models,
        "truncated": summary["truncated"],
        "style": {k: (summary.get("style") or {}).get(k) for k in ("best", "reference", "seconds", "note")}
        if summary.get("style") else None,
    }


def compact_style(r: dict[str, Any]) -> dict[str, Any]:
    """One stylized model's record without per-wall rows (gate, snap, walls, QA, reference check, diagnosis)."""
    if r.get("status") != "ok":
        return {"status": r.get("status"), "eligible": r.get("eligible")}
    snap = {k: v for k, v in r["snap"].items() if k != "meta"}
    walls = {k: v for k, v in r["walls"].items() if k != "coloured_walls"}
    ref = r.get("reference")
    return {"status": "ok", "eligible": r["eligible"], "gate": r["gate"], "snap": snap,
            "registration": r["registration"], "walls": walls,
            "observed_pixel_area_fraction": r["observed_pixel_area_fraction"],
            "roof_observed_area_fraction": r["roof_observed_area_fraction"], "fill": r["fill"], "qa": r["qa"],
            "glb_bytes": r["glb_bytes"], "glb_gz_bytes": r["glb_gz_bytes"], "triangles": r["triangles"],
            "time_s": r["time_s"], "glb": r["glb"], "snap_candidate": r.get("snap_candidate"),
            "reference": ref, "reference_component_verified": r.get("reference_component_verified"),
            "diagnosis": r.get("diagnosis")}


def run_building_process(spec: dict[str, Any], cfg_path: Path, out_dir: Path, time_cap_s: float,
                         stylize: bool = True, key: dict[str, Any] | None = None,
                         resume: bool = False) -> dict[str, Any]:
    """Run the harness on one building in a child process under a hard time cap; returns status + summary.
    With resume, a finished run in out_dir whose saved run key equals key is reused instead."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path, summary_path = out_dir / RUN_KEY_FILE, out_dir / "summary.json"
    if resume and key is not None and key_path.is_file() and summary_path.is_file():
        saved = json.loads(key_path.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            return {"status": "ok", "wall_s": saved["wall_s"], "log": str(out_dir / "harness.log"), "resumed": True,
                    "summary": json.loads(summary_path.read_text(encoding="utf-8"))}
    spec_path = out_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    for stale in ("summary.json", "progress.json", RUN_KEY_FILE):
        (out_dir / stale).unlink(missing_ok=True)
    cmd = [sys.executable, "-I", str(HARNESS), "--building-spec", str(spec_path), "--config", str(cfg_path),
           "--out", str(out_dir), "--overwrite", "--time-cap", str(round(SOFT_CAP_FRACTION * time_cap_s))]
    if stylize:
        cmd.append("--stylize")
    t0 = time.monotonic()
    with open(out_dir / "harness.log", "w", encoding="utf-8") as logf:
        try:
            proc = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT, timeout=time_cap_s, check=False)
            status = "ok" if proc.returncode == 0 else f"failed (exit {proc.returncode})"
        except subprocess.TimeoutExpired:
            status = "timeout"
    result: dict[str, Any] = {"status": status, "wall_s": round(time.monotonic() - t0, 1),
                              "log": str(out_dir / "harness.log")}
    if status == "ok" and summary_path.is_file():
        result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    elif (out_dir / "progress.json").is_file():
        result["progress"] = json.loads((out_dir / "progress.json").read_text(encoding="utf-8"))
    if status == "ok" and "summary" not in result:
        result["status"] = "failed (no summary)"
    if "summary" in result and key is not None:
        key_path.write_text(json.dumps({"key": key, "wall_s": result["wall_s"], "finished": utc_now()}, indent=1),
                            encoding="utf-8")
    return result


def aggregate(per_building: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate of each metric over buildings (Metric.aggregate of the non-missing values; "pooled" metrics are
    the ratio of their numerator's and denominator's sums)."""
    out: dict[str, Any] = {"buildings": len(per_building)}
    for m in ev.METRICS:
        if m.aggregate == "pooled":
            num, den = m.pooled
            pairs = [(float(v[num]), float(v[den])) for v in per_building.values()
                     if v.get(num) is not None and v.get(den) is not None]
            total = sum(d for _, d in pairs)
            out[m.name] = _round(sum(n for n, _ in pairs) / total) if total else None
            continue
        values = [float(v[m.name]) for v in per_building.values() if v.get(m.name) is not None]
        if not values:
            out[m.name] = None
        elif m.aggregate == "max":
            out[m.name] = _round(max(values))
        elif m.aggregate == "sum":
            out[m.name] = _round(sum(values))
        elif m.aggregate == "mean":
            out[m.name] = _round(float(np.mean(values)))
        else:
            out[m.name] = _round(float(np.median(values)))
    return out


def median_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Per metric, the median over repeated runs of the runs that have a value; None unless more than half do."""
    out = {}
    for m in ev.METRICS:
        present = [r[m.name] for r in runs if r.get(m.name) is not None]
        if 2 * len(present) <= len(runs):
            out[m.name] = None
            continue
        med = float(np.median(present))
        out[m.name] = int(med) if med.is_integer() and all(isinstance(v, int) for v in present) else _round(med)
    return out


def noise_from_repeats(repeats: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    """Run-to-run spread (max - min over repeats) per building and metric, its maximum over buildings, and the
    spread of the aggregate over the repeats."""
    per_building, presence = {}, {}
    for slug, runs in repeats.items():
        per_building[slug] = {}
        for m in ev.METRICS:
            values = [r[m.name] for r in runs if r.get(m.name) is not None]
            per_building[slug][m.name] = _round(max(values) - min(values)) if len(values) >= 2 else None
        presence[slug] = sorted(m.name for m in ev.METRICS if 0 < sum(r.get(m.name) is not None for r in runs)
                                < len(runs))
    per_metric = {m.name: max((v[m.name] for v in per_building.values() if v[m.name] is not None), default=None)
                  for m in ev.METRICS}
    k = min((len(r) for r in repeats.values()), default=0)
    aggs = [aggregate({slug: runs[i] for slug, runs in repeats.items()}) for i in range(k)]
    agg_spread = {}
    for m in ev.METRICS:
        values = [a[m.name] for a in aggs if a.get(m.name) is not None]
        agg_spread[m.name] = _round(max(values) - min(values)) if len(values) >= 2 else None
    return {"repeats": k, "buildings": sorted(repeats), "per_building": per_building, "per_metric": per_metric,
            "aggregate": agg_spread, "presence_varies": presence}


def cmd_run(args: argparse.Namespace) -> int:
    ts, ts_path, ts_sha = load_testset(args.testset)
    tid = ts["testset_id"]
    cfg_file = config_path(args.config)
    cfg = ev.load_pipeline_config(cfg_file)
    effective = ev.config_effective(cfg)
    out_json = (args.results_dir or RESULTS_DIR) / tid / f"{args.name}.json"
    if out_json.exists() and not args.overwrite:
        raise SystemExit(f"{out_json} exists; pick another --name or pass --overwrite")
    check_outside_repo(args.work_root, "--work-root")
    check_outside_repo(args.data_root, "--data-root")
    use_changed = (args.on_changed or ts.get("integrity_policy", "skip")) == "use"
    run_dir = args.work_root / tid / args.name
    started, t0 = utc_now(), time.monotonic()
    code_sha, cfg_sha, effective_sha = code_fingerprint(), sha256_file(cfg_file), canonical_sha256(effective)
    buildings: dict[str, dict[str, Any]] = {}
    repeats: dict[str, list[dict[str, Any]]] = {}
    integrity_all: dict[str, Any] = {}
    for b in select_buildings(ts, args.buildings):
        slug = b["slug"]
        bdir = args.data_root / tid / slug
        if b.get("generator") and verify_photos(b, bdir)[STATUS_MISSING]:
            log.info("%s: generating the synthetic photos into %s", slug, bdir)
            materialize_synthetic(b, bdir)
        integ = verify_photos(b, bdir)
        outline, outline_want = building_outline(b, ts_path.parent)
        outline_sha = sha256_file(outline) if outline and outline.is_file() else None
        gt, gt_integ = resolve_gt(b, bdir, args.data_root, args.gt_root)
        integrity_all[slug] = {
            **{k: integ[k] for k in ("photos", STATUS_OK, STATUS_CHANGED, STATUS_MISSING, "changed_keys",
                                     "missing_keys")},
            "outline": (STATUS_MISSING if outline_sha is None and outline else
                        STATUS_OK if outline_want in (None, outline_sha) else STATUS_CHANGED),
            "reference": {k: v for k, v in gt_integ.items() if k != "sparse_dir"},
            "photos_used": integ[STATUS_OK] + (integ[STATUS_CHANGED] if use_changed else 0),
        }
        spec = building_spec(b, ts_path.parent, integ, gt, use_changed)
        record: dict[str, Any] = {"integrity": integrity_all[slug],
                                  "inputs_sha256": inputs_fingerprint(integ, use_changed, outline_sha, gt_integ)}
        key = run_key(spec, code_sha=code_sha, testset_sha=ts_sha, cfg_sha=cfg_sha, effective_sha=effective_sha,
                      inputs_sha=record["inputs_sha256"], stylize=not args.no_stylize, time_cap_s=args.time_cap)
        if not spec["photos"]:
            record.update(status="skipped (no usable photos)", metrics=None)
            buildings[slug] = record
            continue
        runs = []
        for r in range(args.repeat):
            out_dir = run_dir / slug / (f"r{r + 1}" if args.repeat > 1 else "")
            log.info("%s: run %d/%d (%d photos, config %s) -> %s", slug, r + 1, args.repeat, len(spec["photos"]),
                     cfg["config_id"], out_dir)
            res = run_building_process(spec, cfg_file, out_dir, args.time_cap, stylize=not args.no_stylize,
                                       key=key, resume=args.resume)
            summary = res.pop("summary", None)
            if summary is not None:
                got = json.loads(json.dumps(summary["effective_params"]))
                want = json.loads(json.dumps({k: effective[k] for k in ("reconstructor", "settings", "params")}))
                if got != want:
                    raise SystemExit(f"{slug}: the harness ran with other parameters than the config: {got} != {want}")
                res["metrics"] = summary["metrics"]
                res["details"] = compact_details(summary)
                res["artefacts"] = {"summary_json": str(out_dir / "summary.json"),
                                    "summary_md": str(out_dir / "summary.md"),
                                    "topdown_png": summary["topdown_png"], "data_dir": summary["data_dir"]}
            runs.append(res)
            log.info("%s: %s in %.0f s%s", slug, res["status"], res["wall_s"],
                     " (resumed: the earlier run's result, same run key)" if res.get("resumed") else "")
        first = runs[0]
        record.update(status=first["status"], wall_s=first["wall_s"], metrics=first.get("metrics"),
                      details=first.get("details"), artefacts=first.get("artefacts"), log=first["log"])
        if any(x.get("resumed") for x in runs):
            record["resumed_runs"] = [k + 1 for k, x in enumerate(runs) if x.get("resumed")]
        ok_runs = [x["metrics"] for x in runs if x.get("metrics")]
        if args.repeat > 1 and ok_runs:
            # The recorded performance of a repeated run is the per-metric median; details/artefacts are run 1's.
            record.update(metrics=median_metrics(ok_runs), metrics_first=first.get("metrics"),
                          status="ok" if len(ok_runs) == len(runs) else f"{len(ok_runs)}/{len(runs)} runs ok")
        if "progress" in first:
            record["progress"] = {k: first["progress"].get(k) for k in ("requests_done", "requests", "elapsed_s")}
        if args.repeat > 1:
            record["repeats"] = [{"status": x["status"], "wall_s": x["wall_s"], "metrics": x.get("metrics")}
                                 for x in runs]
            if len(ok_runs) >= 2:
                repeats[slug] = ok_runs
        buildings[slug] = record
    per_building_ok = {s: r["metrics"] for s, r in buildings.items() if r.get("metrics")}
    code_sha_end = code_fingerprint()
    if code_sha_end != code_sha:
        log.warning("the code under %s changed during the run (code_sha256 %s -> %s)", CODE_PATHS, code_sha,
                    code_sha_end)
    materialised = args.data_root / tid / "integrity_report.json"     # scripts/materialize_testset.py writes it
    mat = json.loads(materialised.read_text(encoding="utf-8")) if materialised.is_file() else None
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "metric_schema_version": ev.METRIC_SCHEMA_VERSION,
        "run_name": args.name,
        "note": args.note,
        "testset": {"id": tid, "sha256": ts_sha, "path": _repo_relative(ts_path),
                    "buildings_selected": list(buildings), "on_changed": "use" if use_changed else "skip",
                    "integrity": integrity_all,
                    "materialize_report": {"path": str(materialised), **{k: mat.get(k) for k in (
                        "generated", "testset_sha256", "totals", "complete", "stopped")}} if mat else None},
        "config": {"id": cfg["config_id"], "sha256": cfg_sha,
                   "path": _repo_relative(cfg_file),
                   "overrides": {k: cfg[k] for k in ("reconstructor", "settings", "params", "harness")},
                   "effective": effective, "effective_sha256": effective_sha},
        "provenance": {"git": git_state(), "code_sha256": code_sha,
                       **({"code_sha256_end": code_sha_end} if code_sha_end != code_sha else {}),
                       "versions": versions(), "started": started,
                       "wall_s": round(time.monotonic() - t0, 1), "harness_version": ev.HARNESS_VERSION,
                       "time_cap_s": args.time_cap, "repeat": args.repeat, "stylize": not args.no_stylize,
                       "command": " ".join(sys.argv)},
        "buildings": buildings,
        "aggregate": aggregate(per_building_ok),
        "noise": noise_from_repeats(repeats) if repeats else None,
        "artefacts_dir": str(run_dir),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    print(format_run(result))
    return 0


def format_run(result: dict[str, Any]) -> str:
    lines = [f"{result['testset']['id']} / {result['config']['id']} / {result['run_name']}: "
             f"{sum(1 for b in result['buildings'].values() if b.get('metrics'))}/{len(result['buildings'])} "
             f"buildings ok, {result['provenance']['wall_s']} s"]
    for slug, b in result["buildings"].items():
        m = b.get("metrics") or {}
        lines.append(f"  {slug}: {b['status']}; accepted {m.get('photos_accepted')}/{m.get('photos_uploaded')}, "
                     f"published {m.get('models_published')}, QA rejected {m.get('models_qa_rejected')}, "
                     f"outline rot {m.get('outline_rotation_abs_deg')} deg / shift {m.get('outline_shift_m')} m, "
                     f"GT rot {m.get('gt_rotation_error_median_deg')} deg, {m.get('wall_time_s')} s; style gate "
                     f"{m.get('style_registration_passed')}/{m.get('style_models_eligible')} (right "
                     f"{m.get('style_registration_right')}, wrong {m.get('style_registration_wrong')}), walls {m.get('style_walls_coloured')}/"
                     f"{m.get('style_walls_total')} coloured, {m.get('style_time_s')} s")
    return "\n".join(lines)


# ---------------------------------------------------------------- compare


def load_result(path: Path) -> dict[str, Any]:
    r = json.loads(Path(path).read_text(encoding="utf-8"))
    if r.get("result_schema_version") != RESULT_SCHEMA_VERSION:
        raise SystemExit(f"{path}: result_schema_version {r.get('result_schema_version')!r}, "
                         f"this tool reads {RESULT_SCHEMA_VERSION}")
    return r


def noise_sources(results: list[dict[str, Any]], extra: list[Path], results_dir: Path) -> list[dict[str, Any]]:
    """Recorded noise blocks usable for these runs: same test set sha and metric schema. Looks into the two
    runs, explicit --noise files and every result of the same test set under results_dir."""
    tid, sha = results[0]["testset"]["id"], results[0]["testset"]["sha256"]
    schema = results[0]["metric_schema_version"]
    found, seen = [], set()
    candidates = list(results) + [load_result(p) for p in extra]
    for p in sorted((results_dir / tid).glob("*.json")) if (results_dir / tid).is_dir() else []:
        try:
            candidates.append(load_result(p))
        except (SystemExit, json.JSONDecodeError):
            continue
    for r in candidates:
        n = r.get("noise")
        key = (r.get("run_name"), r["provenance"].get("started"))
        if not n or key in seen or r["testset"]["sha256"] != sha or r["metric_schema_version"] != schema:
            continue
        seen.add(key)
        found.append({"run": r["run_name"], "config": r["config"]["id"], **n})
    return found


def judge(metric: ev.Metric, a: Any, b: Any, noise: float | None, factor: float,
          presence_varies: bool = False) -> dict[str, Any]:
    """Delta b - a of one metric with its direction, threshold and significance.

    A change counts as significant when |delta| exceeds max(min_delta, rel_floor x |a|, factor x noise).
    A metric that appears or disappears (a model published in one run only) is significant unless the
    recorded repeats show that it comes and goes anyway (presence_varies).
    """
    if a is None or b is None:
        verdict = "n/a" if a is None and b is None else ("appeared" if a is None else "disappeared")
        return {"a": a, "b": b, "delta": None, "verdict": verdict, "noise": noise, "threshold": None,
                "significant": verdict != "n/a" and not presence_varies}
    delta = float(b) - float(a)
    threshold = max(metric.min_delta, metric.rel_floor * abs(float(a)), factor * noise if noise is not None else 0.0)
    if delta == 0:
        verdict = "same"
    elif metric.direction == "none":
        verdict = "changed"
    else:
        verdict = "better" if (delta > 0) == (metric.direction == "higher") else "worse"
    # The tolerance keeps a delta equal to the recorded spread (up to float rounding) within the noise.
    significant = abs(delta) > threshold * (1 + 1e-9) + 1e-9
    return {"a": a, "b": b, "delta": _round(delta), "verdict": verdict, "significant": significant,
            "noise": noise, "threshold": _round(threshold)}


def compare_results(a: dict[str, Any], b: dict[str, Any], sources: list[dict[str, Any]], factor: float = 1.0,
                    force: bool = False) -> dict[str, Any]:
    """Per building and aggregate, metric by metric. Refuses (ValueError) different test sets or metric
    schemas unless force; buildings whose inputs differ are listed and left out of the aggregate."""
    problems = []
    if a["testset"]["sha256"] != b["testset"]["sha256"]:
        problems.append(f"test sets differ: {a['testset']['id']} {a['testset']['sha256'][:12]} vs "
                        f"{b['testset']['id']} {b['testset']['sha256'][:12]}")
    if a["metric_schema_version"] != b["metric_schema_version"]:
        problems.append(f"metric_schema_version differs: {a['metric_schema_version']} vs {b['metric_schema_version']}")
    if problems and not force:
        raise ValueError("; ".join(problems) + " (use --force to compare anyway)")
    ea, eb = a["config"]["effective"], b["config"]["effective"]
    param_diff = {}
    for section in ("reconstructor", "settings", "params", "harness"):
        va, vb = ea.get(section), eb.get(section)
        if isinstance(va, dict) or isinstance(vb, dict):
            for k in sorted(set(va or {}) | set(vb or {})):
                if (va or {}).get(k) != (vb or {}).get(k):
                    param_diff[f"{section}.{k}"] = [(va or {}).get(k), (vb or {}).get(k)]
        elif va != vb:
            param_diff[section] = [va, vb]

    def presence_for(slug: str | None, name: str) -> bool:
        return any(name in (s.get("presence_varies") or {}).get(b, []) for s in sources
                   for b in ([slug] if slug else (s.get("presence_varies") or {})))

    def noise_for(slug: str | None, name: str) -> float | None:
        vals = []
        for s in sources:
            v = (s["per_building"].get(slug) or {}).get(name) if slug else s["aggregate"].get(name)
            if v is None and slug is None:
                v = s["per_metric"].get(name)
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    common = [s for s in a["buildings"] if s in b["buildings"] and a["buildings"][s].get("metrics")
              and b["buildings"][s].get("metrics")]
    same_inputs = [s for s in common if a["buildings"][s]["inputs_sha256"] == b["buildings"][s]["inputs_sha256"]]
    per_building = {}
    for slug in common:
        ma, mb = a["buildings"][slug]["metrics"], b["buildings"][slug]["metrics"]
        per_building[slug] = {
            "inputs_identical": slug in same_inputs,
            "metrics": {m.name: judge(m, ma.get(m.name), mb.get(m.name), noise_for(slug, m.name), factor,
                                      presence_for(slug, m.name)) for m in ev.METRICS},
        }
    agg_a = aggregate({s: a["buildings"][s]["metrics"] for s in same_inputs})
    agg_b = aggregate({s: b["buildings"][s]["metrics"] for s in same_inputs})
    agg = {m.name: judge(m, agg_a[m.name], agg_b[m.name], noise_for(None, m.name), factor,
                         presence_for(None, m.name)) for m in ev.METRICS}
    rows = [j for pb in per_building.values() for j in pb["metrics"].values()] + list(agg.values())
    return {
        "a": {"run": a["run_name"], "config": a["config"]["id"], "started": a["provenance"]["started"]},
        "b": {"run": b["run_name"], "config": b["config"]["id"], "started": b["provenance"]["started"]},
        "testset": a["testset"]["id"], "problems": problems, "parameter_differences": param_diff,
        "effective_identical": a["config"]["effective_sha256"] == b["config"]["effective_sha256"],
        "noise_sources": [f"{s['run']} ({s['config']}, {s['repeats']} repeats)" for s in sources],
        "buildings_compared": common, "buildings_inputs_differ": [s for s in common if s not in same_inputs],
        "buildings_only_in_a": [s for s in a["buildings"] if s not in common],
        "buildings_only_in_b": [s for s in b["buildings"] if s not in common],
        "aggregate": agg, "per_building": per_building,
        "significant_worse": sum(r["significant"] and r["verdict"] == "worse" for r in rows),
        "significant_better": sum(r["significant"] and r["verdict"] == "better" for r in rows),
    }


def format_compare(c: dict[str, Any], show_all: bool = False) -> str:
    def fmt(v: Any, signed: bool = False) -> str:
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:+g}" if signed and v != 0 else f"{v:g}"
        return str(v)

    lines = [f"compare {c['testset']}: A = {c['a']['run']} ({c['a']['config']}, {c['a']['started']}) vs "
             f"B = {c['b']['run']} ({c['b']['config']}, {c['b']['started']})"]
    lines += [f"WARNING: {p}" for p in c["problems"]]
    if c["effective_identical"]:
        lines.append("Effective parameters identical: any change is run-to-run variation.")
    else:
        lines.append("Parameter differences (A -> B): " + "; ".join(
            f"{k}: {va} -> {vb}" for k, (va, vb) in c["parameter_differences"].items()))
    lines.append("Noise estimates from: " + (", ".join(c["noise_sources"]) or
                                             "none recorded (only each metric's minimum change applies)"))
    if c["buildings_inputs_differ"]:
        lines.append(f"Inputs differ (left out of the aggregate): {', '.join(c['buildings_inputs_differ'])}")
    for side in ("a", "b"):
        if c[f"buildings_only_in_{side}"]:
            only = ", ".join(c[f"buildings_only_in_{side}"])
            lines.append(f"Only in {side.upper()} (or failed in the other): {only}")

    def table(rows: dict[str, dict[str, Any]], everything: bool) -> list[str]:
        out = ["| metric | A | B | delta | verdict | noise | significant |", "|---|---|---|---|---|---|---|"]
        for name, j in rows.items():
            if not everything and j["verdict"] in ("same", "n/a"):
                continue
            flag = "**yes**" if j["significant"] else "no"
            out.append(f"| {name} | {fmt(j['a'])} | {fmt(j['b'])} | {fmt(j['delta'], True)} | {j['verdict']} | "
                       f"{fmt(j['noise'])} | {flag} |")
        return out if len(out) > 2 else ["(no metric changed)"]

    n = len(c["buildings_compared"]) - len(c["buildings_inputs_differ"])
    lines += ["", f"Aggregate over {n} building(s) with identical inputs:"] + table(c["aggregate"], True)
    for slug, pb in c["per_building"].items():
        lines += ["", f"{slug}{'' if pb['inputs_identical'] else ' (inputs differ)'}:"] + table(pb["metrics"], show_all)
    lines += ["", f"Significant changes: {c['significant_worse']} worse, {c['significant_better']} better."]
    return "\n".join(lines)


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = load_result(args.run_a), load_result(args.run_b)
    sources = noise_sources([a, b], args.noise or [], args.results_dir or RESULTS_DIR)
    try:
        c = compare_results(a, b, sources, args.noise_factor, args.force)
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    print(format_compare(c, args.all))
    if args.json:
        args.json.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    return 1 if args.fail_on_worse and c["significant_worse"] else 0


# ---------------------------------------------------------------- report

LEADERBOARD_COLUMNS = ("models_published", "models_qa_rejected", "facades_published", "full_model_published",
                       "registration_rate", "estimate_coverage", "outline_rotation_abs_deg", "outline_shift_m",
                       "gt_rotation_error_median_deg", "gt_gross_shown", "style_models_eligible",
                       "style_registration_pass_rate", "style_registration_right", "style_registration_wrong",
                       "style_walls_coloured", "style_qa_failed", "wall_time_s")
BUILDING_COLUMNS = ("photos_accepted", "models_published", "models_qa_rejected", "facades_published",
                    "full_model_published", "registration_rate", "estimate_coverage", "outline_rotation_abs_deg",
                    "outline_shift_m", "outline_residual_m", "gps_residual_median_m", "compass_error_median_deg",
                    "gt_rotation_error_median_deg", "gt_gross_cameras", "gt_gross_shown", "style_models_eligible",
                    "style_registration_passed", "style_registration_right", "style_registration_wrong",
                    "style_snap_refused_gps", "style_snap_refused_ambiguous", "style_snap_residual_before_m",
                    "style_walls_coloured", "style_walls_total", "style_qa_failed", "wall_time_s", "style_time_s")


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def build_leaderboard(results: list[dict[str, Any]]) -> str:
    unit = {m.name: m.unit for m in ev.METRICS}
    lines = [
        "# Crowd3D 基准排行榜（benchmark leaderboard）",
        "",
        "由 `python -I scripts/benchmark.py report` 根据 `eval/results/<testset>/*.json` 生成，请勿手工编辑。"
        "指标定义见文末；方向：↑ 越大越好，↓ 越小越好。",
        "",
        "## 全部运行",
        "",
        "| 测试集 | 运行 | 配置 | 开始时间 (UTC) | 提交 | 建筑 成功/总数 | "
        + " | ".join(f"{c} ({unit[c]})" for c in LEADERBOARD_COLUMNS) + " | 备注 |",
        "|" + "---|" * (7 + len(LEADERBOARD_COLUMNS)),
    ]
    results = sorted(results, key=lambda r: (r["testset"]["id"], r["provenance"]["started"], r["run_name"]))
    for r in results:
        g = r["provenance"]["git"]
        commit = (g.get("commit") or "?")[:7] + ("+dirty" if g.get("dirty") else "")
        ok = sum(1 for b in r["buildings"].values() if b.get("metrics"))
        lines.append(f"| {r['testset']['id']} | {r['run_name']} | {r['config']['id']} | "
                     f"{r['provenance']['started'][:16].replace('T', ' ')} | {commit} | {ok}/{len(r['buildings'])} | "
                     + " | ".join(_cell(r["aggregate"].get(c)) for c in LEADERBOARD_COLUMNS)
                     + f" | {(r.get('note') or '').replace('|', '/')} |")
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        newest[(r["testset"]["id"], r["config"]["id"])] = r
    lines += ["", "## 每个配置最新一次运行的分建筑结果", ""]
    for (tid, cid), r in sorted(newest.items()):
        lines += [f"### {tid} · {cid} · {r['run_name']}", "",
                  "| 建筑 | 状态 | " + " | ".join(BUILDING_COLUMNS) + " |",
                  "|" + "---|" * (2 + len(BUILDING_COLUMNS))]
        for slug, b in r["buildings"].items():
            m = b.get("metrics") or {}
            lines.append(f"| {slug} | {b['status']} | " + " | ".join(_cell(m.get(c)) for c in BUILDING_COLUMNS) + " |")
        lines.append("")
    noisy = [r for r in results if r.get("noise")]
    if noisy:
        lines += ["## 运行间噪声（同一配置重复运行的极差）", ""]
        for r in noisy:
            n = r["noise"]
            shown = {k: v for k, v in n["per_metric"].items() if v}
            where = ", ".join(n["buildings"])
            lines.append(f"- {r['testset']['id']} · {r['run_name']}（{n['repeats']} 次，建筑 {where}）："
                         + ("; ".join(f"{k} {_cell(v)}" for k, v in shown.items()) or "所有指标完全相同"))
        lines.append("")
    lines += ["## 指标定义", "", f"metric_schema_version = {ev.METRIC_SCHEMA_VERSION}", "",
              "| 指标 | 单位 | 方向 | 跨建筑汇总 | 说明 |", "|---|---|---|---|---|"]
    arrows = {"higher": "↑", "lower": "↓", "none": "-"}
    for m in ev.METRICS:
        text = m.description.replace("|", "&#124;")
        lines.append(f"| {m.name} | {m.unit} | {arrows[m.direction]} | {m.aggregate} | {text} |")
    return "\n".join(lines) + "\n"


def cmd_report(args: argparse.Namespace) -> int:
    results_dir = args.results_dir or RESULTS_DIR
    results = []
    for p in sorted(results_dir.glob("*/*.json")):
        try:
            results.append(load_result(p))
        except (SystemExit, json.JSONDecodeError) as e:
            log.warning("skipping %s: %s", p, e)
    out = args.out or results_dir / "LEADERBOARD.md"
    out.write_text(build_leaderboard(results), encoding="utf-8")
    print(f"wrote {out} ({len(results)} runs)")
    return 0


# ---------------------------------------------------------------- freezing new test set versions (maintainers)


def exif_summary(data: bytes) -> dict[str, Any]:
    m = read_photo_meta(data)
    return {"lat": m.lat, "lon": m.lon, "alt": m.alt, "gps_accuracy_m": m.gps_accuracy_m, "heading_deg": m.heading_deg,
            "heading_ref": m.heading_ref, "focal_mm": m.focal_mm, "focal_35mm": m.focal_35mm,
            "captured_at": m.captured_at, "make": m.make, "model": m.model, "orientation": m.orientation,
            "width": m.width, "height": m.height}


def _write_new_testset(testset_id: str, ts: dict[str, Any]) -> Path:
    path = TESTSETS_DIR / testset_id / "testset.json"
    if path.exists():
        raise SystemExit(f"{path} exists: a released test set is immutable, use a new id")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ts, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def box_outline_geojson(lat: float, lon: float, width_e_m: float, depth_n_m: float, name: str) -> dict[str, Any]:
    from crowd3d.geo import LocalENU

    enu = LocalENU(lat, lon)
    corners = [(-1, -1), (1, -1), (1, 1), (-1, 1), (-1, -1)]
    ring = []
    for sx, sy in corners:
        la, lo, _ = enu.to_geodetic(sx * width_e_m / 2, sy * depth_n_m / 2, 0.0)
        ring.append([round(lo, 9), round(la, 9)])
    return {"type": "FeatureCollection", "licence": "generated (crowd3d.synthetic BuildingSpec), no third-party data",
            "features": [{"type": "Feature", "properties": {"name": name, "width_e_m": width_e_m,
                                                            "depth_n_m": depth_n_m},
                          "geometry": {"type": "Polygon", "coordinates": [ring]}}]}


SYNTHETIC_BUILDINGS = (
    ("synthetic-ring", "Synthetic box, photographed from all 8 sides (plus one photo of each invalid kind)",
     {"photos_per_sector": 4, "num_users": 5, "size": [640, 480], "gps_noise_m": 2.0, "heading_noise_deg": 8.0,
      "include_bad": True, "seed": 0}),
    ("synthetic-two-sides", "Synthetic box, photographed from the E and SE sides only",
     {"sectors": [2, 3], "photos_per_sector": 6, "num_users": 3, "size": [640, 480], "gps_noise_m": 2.0,
      "heading_noise_deg": 8.0, "include_bad": False, "seed": 1}),
)


def cmd_freeze_synthetic(args: argparse.Namespace) -> int:
    from crowd3d.synthetic import DEMO_LAT, DEMO_LON, BuildingSpec

    tdir = TESTSETS_DIR / args.testset_id
    spec = BuildingSpec()
    outline = box_outline_geojson(DEMO_LAT, DEMO_LON, spec.width_e_m, spec.depth_n_m, "crowd3d.synthetic box")
    buildings = []
    for slug, name, kwargs in SYNTHETIC_BUILDINGS:
        b = {"slug": slug, "name": name, "centre": {"lat": DEMO_LAT, "lon": DEMO_LON}, "radius_m": 150.0,
             "generator": {"module": "crowd3d.synthetic", "function": "generate_dataset", "kwargs": kwargs},
             "upload_all": bool(kwargs.get("include_bad")), "photos": []}
        bdir = args.data_root / args.testset_id / slug
        shutil.rmtree(bdir, ignore_errors=True)
        materialize_synthetic(b, bdir)
        manifest = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
        for p in manifest["photos"]:
            data = (bdir / p["file"]).read_bytes()
            seg = jpeg_exif_segment(data)
            b["photos"].append({"key": p["file"], "title": p["file"], "file": p["file"], "sha256": sha256_bytes(data),
                                "exif_sha256": sha256_bytes(seg) if seg else None, "kind": p["kind"],
                                "expected_reason": p["expected_reason"], "author": p["user_id"],
                                "licence": "generated", "exif": exif_summary(data)})
        b["osm_outline"] = {"file": "outlines/synthetic-box.geojson", "licence": "generated, no third-party data"}
        buildings.append(b)
    outline_path = tdir / "outlines" / "synthetic-box.geojson"
    if (tdir / "testset.json").exists():
        raise SystemExit(f"{tdir / 'testset.json'} exists: a released test set is immutable, use a new id")
    outline_path.parent.mkdir(parents=True, exist_ok=True)
    outline_path.write_text(json.dumps(outline, indent=1) + "\n", encoding="utf-8")
    for b in buildings:
        b["osm_outline"]["sha256"] = sha256_file(outline_path)
    ts = {
        "schema_version": TESTSET_SCHEMA_VERSION, "testset_id": args.testset_id, "created": utc_now()[:10],
        "kind": "synthetic",
        "description": "Cheap, fully reproducible suite: procedurally rendered photos of the crowd3d.synthetic box "
                       "building with known poses (no download). For quick checks of parameter changes.",
        "licence": "Generated by crowd3d.synthetic; no third-party content.",
        "integrity_policy": "use",
        "materialise": "python -I scripts/benchmark.py prepare --testset " + args.testset_id,
        "schema_notes": {
            "buildings[].generator": "crowd3d.synthetic.generate_dataset kwargs; the photos are regenerated, and their "
                                     "sha256 checks that the generator (and its libraries) still render the same bytes",
            "buildings[].upload_all": "true: also upload the deliberately invalid photos (no GPS etc.) to test "
                                      "the validation funnel",
            "buildings[].photos[].sha256": "of the generated file", "integrity_policy": "use: changed files are still "
            "uploaded (and recorded), since a renderer change alters every file",
        },
        "buildings": buildings,
    }
    print(f"wrote {_write_new_testset(args.testset_id, ts)}")
    return 0


def commons_original_url(file_name: str) -> str:
    """upload.wikimedia.org URL of a Commons file (the md5 of the underscored name picks the folders)."""
    from urllib.parse import quote

    name = file_name.replace(" ", "_")
    h = hashlib.md5(name.encode("utf-8")).hexdigest()
    return f"https://upload.wikimedia.org/wikipedia/commons/{h[0]}/{h[:2]}/{quote(name)}"


def cmd_freeze_mdx(args: argparse.Namespace) -> int:
    """A test set of the real-EXIF-GPS photos of MegaDepth-X components (photos = the dataset's own files)."""
    import pyarrow.parquet as pq

    lic = pq.read_table(args.license_parquet, columns=["name_in_megadepth-x", "image_name", "license_short_name",
                                                       "subcat"]).to_pylist()
    lic_by = {r["name_in_megadepth-x"]: r for r in lic}
    authors: dict[str, tuple[str, str | None]] = {}
    for m in args.authors_from or []:
        for e in json.loads(Path(m).read_text(encoding="utf-8"))["photos"]:
            if e.get("author"):
                authors[e["file_name"]] = (e["author"], e.get("author_url"))
    client = None
    if args.page_cache:
        import fetch_commons_landmark as fc

        client = fc.PoliteClient(args.page_cache, offline=True)   # cached description pages only: no request

    def author_of(commons: str | None) -> tuple[str | None, str | None]:
        if not commons:
            return None, None
        if commons in authors:
            return authors[commons]
        if client is not None:
            try:
                resp = client.get(fc.page_url("File:" + commons), gzip_ok=True)
            except Exception:   # not in the cache (offline): unknown
                return None, None
            if resp.ok:
                page = fc.parse_file_page(resp.text())
                if page.author_url and "wikipedia.org/wiki/" in page.author_url:
                    return None, None       # the cell credits the depicted artwork's author, not the photographer
                # A Creator template renders the person's whole infobox: keep the name.
                name = (page.author or "").split(" Description ")[0].strip()[:80] or None
                return name, page.author_url
        return None, None
    tdir = TESTSETS_DIR / args.testset_id
    if (tdir / "testset.json").exists():
        raise SystemExit(f"{tdir / 'testset.json'} exists: a released test set is immutable, use a new id")
    buildings = []
    for item in args.building:
        slug, _, comp_dir = item.partition("=")
        comp_dir = Path(comp_dir)
        scene, comp = args.scene_names[slug], comp_dir.name
        entry = ev.catalog_entry(args.catalog, slug)
        url = ev.MEGADEPTH_X_URL.format(name=urllib.parse.quote(scene))
        osm_src = REPO_ROOT / entry["osm"]["file"]
        osm_dst = tdir / "outlines" / osm_src.name
        osm_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(osm_src, osm_dst)
        photos, gt_images = [], {}
        for img in sorted((comp_dir / "images").iterdir()):
            data = img.read_bytes()
            ex = exif_summary(data)
            if ex["lat"] is None or ex["lon"] is None:
                continue
            member = f"{scene}/{comp}/images/{img.name}"
            row = lic_by.get(member) or {}
            commons = row.get("image_name")
            seg = jpeg_exif_segment(data)
            author, author_url = author_of(commons)
            photos.append({
                "key": img.name, "title": f"File:{commons}" if commons else None, "file_name": commons,
                "file": f"photos/{img.name}", "sha256": sha256_bytes(data),
                "exif_sha256": sha256_bytes(seg) if seg else None,
                "source": {"kind": "megadepth-x", "archive_url": url, "member": member},
                "commons_page": (f"https://commons.wikimedia.org/wiki/File:{commons.replace(' ', '_')}"
                                 if commons else None),
                "original_url": commons_original_url(commons) if commons else None, "thumbnail_width": None,
                "author": author, "author_url": author_url, "licence": row.get("license_short_name"),
                "commons_subcategory": row.get("subcat"), "exif": ex,
            })
            gt_images[img.name] = {"name": img.name, "orientation": ex["orientation"]}
        sparse = comp_dir / "sparse"
        buildings.append({
            "slug": slug, "name": entry["name"], "centre": {"lat": entry["lat"], "lon": entry["lon"]},
            "centre_source": entry.get("centre", {}).get("source"), "radius_m": float(entry["radius_m"]),
            "osm_outline": {"file": f"outlines/{osm_src.name}", "sha256": sha256_file(osm_dst),
                            "licence": entry["osm"].get("license"), "attribution": entry["osm"].get("attribution"),
                            "osm_url": entry["osm"].get("url")},
            "gt_reference": {"source": "megadepth-x", "scene": scene, "component": comp, "archive_url": url,
                             "licence": "CC BY 4.0 (MegaDepth-X SfM data)",
                             "sparse_files": {n: {"sha256": sha256_file(sparse / n),
                                                  "bytes": (sparse / n).stat().st_size}
                                              for n in SPARSE_FILES if (sparse / n).is_file()},
                             "images": gt_images},
            "photos": photos,
        })
    ts = {
        "schema_version": TESTSET_SCHEMA_VERSION, "testset_id": args.testset_id, "created": utc_now()[:10],
        "kind": "megadepth-x",
        "description": args.description,
        "licence": "Photos: Wikimedia Commons, each under its own licence (per photo; MegaDepth-X license.parquet). "
                   "Reference SfM models: MegaDepth-X, CC BY 4.0. Outlines: OpenStreetMap contributors, ODbL 1.0. "
                   "Only metadata is committed; photos and anything derived from them stay outside the repository.",
        "integrity_policy": "skip",
        "materialise": f"python -I scripts/benchmark.py prepare --testset {args.testset_id} "
                       "[--mdx-local <slug>=<extracted component dir>]",
        "schema_notes": {
            "buildings[].photos[]": "only photos whose own EXIF carries GPS; file = path under "
                                    "<data-root>/<testset_id>/<slug>; sha256 = of that file (the MegaDepth-X image, "
                                    "~1000 px); exif_sha256 = of its EXIF APP1 payload (the dataset keeps GPS, "
                                    "compass and capture time but drops focal length and camera model); "
                                    "source.member = path in the archive; author/author_url = from the Commons "
                                    "description page where known, else null; original_url = the Commons original "
                                    "the dataset image was made from; thumbnail_width null (not a thumbnail)",
            "buildings[].gt_reference": "MegaDepth-X component model; images maps photo key -> reference image "
                                        "name and that image's EXIF orientation; materialised under "
                                        "<data-root>/<testset_id>/<slug>/gt/<component>/sparse",
        },
        "buildings": buildings,
    }
    print(f"wrote {_write_new_testset(args.testset_id, ts)}")
    return 0


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -I scripts/benchmark.py", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help=f"materialised test sets (default $CROWD3D_TESTSET_ROOT or {DEFAULT_DATA_ROOT})")
    common.add_argument("--results-dir", type=Path, default=None, help="default eval/results")

    p = sub.add_parser("prepare", parents=[common], help="materialise a synthetic or MegaDepth-X test set")
    p.add_argument("--testset", required=True)
    p.add_argument("--buildings", nargs="*")
    p.add_argument("--mdx-local", nargs="*", metavar="SLUG=DIR",
                   help="link MegaDepth-X photos and model from a local extraction of <scene>/<component>")

    p = sub.add_parser("run", parents=[common], help="run a test set with a pipeline config")
    p.add_argument("--testset", required=True)
    p.add_argument("--config", required=True, help="config id (eval/configs/<id>.json) or path")
    p.add_argument("--name", required=True, help="run name: eval/results/<testset>/<name>.json")
    p.add_argument("--buildings", nargs="*", help="subset of building slugs (default all)")
    p.add_argument("--repeat", type=int, default=1, help="runs per building; >= 2 records run-to-run noise")
    p.add_argument("--time-cap", type=float, default=DEFAULT_TIME_CAP_S, help="hard cap per building run, seconds")
    p.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT,
                   help=f"artefacts (default $CROWD3D_BENCH_DIR or {DEFAULT_WORK_ROOT})")
    p.add_argument("--gt-root", type=Path, default=None,
                   help="reference models under <gt-root>/<slug>/<component>/sparse")
    p.add_argument("--on-changed", choices=("skip", "use"), default=None,
                   help="photos whose sha256 changed: leave out (default) or upload anyway; default from the set")
    p.add_argument("--note", default=None, help="free text stored with the result")
    p.add_argument("--no-stylize", action="store_true",
                   help="do not stylize the published models (style_* metrics stay empty; work dirs are pruned)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing result of that name")
    p.add_argument("--resume", action="store_true",
                   help="reuse building runs in --work-root whose run_key.json matches (same code, config, test set,"
                        " inputs and options), e.g. after an interruption; the others run again")

    p = sub.add_parser("compare", help="compare two results metric by metric")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--noise", type=Path, nargs="*", help="extra results whose noise blocks to use")
    p.add_argument("--noise-factor", type=float, default=1.0, help="significant beyond this x the recorded spread")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--force", action="store_true", help="compare even across test sets or metric schemas")
    p.add_argument("--all", action="store_true", help="also list unchanged metrics per building")
    p.add_argument("--json", type=Path, default=None, help="also write the comparison as JSON")
    p.add_argument("--fail-on-worse", action="store_true", help="exit 1 when anything got significantly worse")

    p = sub.add_parser("report", help="regenerate eval/results/LEADERBOARD.md")
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)

    p = sub.add_parser("freeze-synthetic", parents=[common], help="maintainers: freeze a new synthetic test set")
    p.add_argument("--testset-id", required=True)

    p = sub.add_parser("freeze-mdx", parents=[common], help="maintainers: freeze a MegaDepth-X real-GPS test set")
    p.add_argument("--testset-id", required=True)
    p.add_argument("--building", nargs="+", required=True, metavar="SLUG=COMPONENT_DIR")
    p.add_argument("--scene", nargs="+", required=True, metavar="SLUG=SCENE", help="MegaDepth-X scene name per slug")
    p.add_argument("--catalog", type=Path, default=REPO_ROOT / "eval" / "landmarks.json")
    p.add_argument("--license-parquet", type=Path, required=True)
    p.add_argument("--authors-from", nargs="*", help="fetch_commons_landmark manifest.json files with authors")
    p.add_argument("--page-cache", type=Path, default=None,
                   help="fetch_commons_landmark HTTP cache: authors from cached description pages (offline)")
    p.add_argument("--description", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "freeze-mdx":
        args.scene_names = dict(item.split("=", 1) for item in args.scene)
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    handlers = {"prepare": cmd_prepare, "run": cmd_run, "compare": cmd_compare, "report": cmd_report,
                "freeze-synthetic": cmd_freeze_synthetic, "freeze-mdx": cmd_freeze_mdx}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
