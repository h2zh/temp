"""Reusable benchmark: frozen test sets x pipeline configs -> committed results, comparisons, a leaderboard.

    python -I scripts/benchmark.py prepare --testset synthetic-v1           # materialise generated/MegaDepth-X sets
    python -I scripts/benchmark.py run --testset synthetic-v1 --config default --name default-1 \\
        [--buildings slug ...] [--repeat 2] [--time-cap 3600] [--max-requests N|SLUG=N ...]
    python -I scripts/benchmark.py compare eval/results/synthetic-v1/default-1.json \\
        eval/results/synthetic-v1/default-2.json [--noise <result with a noise block>] \\
        [--criteria review-v1] [--exclude-group style ...]  # pass rules per split (eval/criteria)
    python -I scripts/benchmark.py posediff <result A> <result B>          # final shown camera poses, A vs B
    python -I scripts/benchmark.py report                                   # regenerates eval/results/LEADERBOARD.md
    python -I scripts/benchmark.py rescore --testset commons-ten-v1 --from eval/results/commons-ten-v1/v1-after-fixes.json \
        --name v1-poses [--restylize [--structure on|off] [--snap-gate on|off]] [--regeo]   # no new SfM
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
the registration gate's pass rate; where the landmark has an OSM parts snapshot the step goes through
crowd3d.structure_pipeline (OSM-parts massing, judged with the photos when their registration is accepted: the
struct_* metrics); --no-stylize skips it (style_* and struct_* metrics then stay empty). The
committed result is eval/results/<testset_id>/<run_name>.json: provenance, config, effective parameters,
integrity, per-building metrics (eval_real_landmark.METRICS) and their aggregate. --repeat N runs every
building N times and stores the spread (run-to-run noise) that compare uses to flag only
changes beyond noise. --resume reuses a building run left by an interrupted invocation when its run_key.json
says it ran the same code (code_sha256 over CODE_PATHS), config, test set, inputs and options. --max-requests caps a
building's upload requests at an earlier run's request prefix (its requests count in the result's details), so a
faster configuration is compared on the same uploads rather than on more of them (eval/configs/cache.json: the
landmark feature cache, whose hits and size each run reports in details.cache).

`rescore` computes the photo_pose_* metrics (crowd3d.photoposes: where each photo was taken and which way it
looked, against the outline-corrected reference) and the schema-7 scores (geo_*, wrong_face_*, cov_*, calib_*,
stale_overwrites, version_drift_*, and what the request log holds: latency, and CPU time, memory and time to pose
where it recorded them) from the artefacts a recorded run left (summary.json, the SQLite store, PLYs, kept work
dirs, the stylization's accepted snaps) without running SfM again, and writes a new result whose other metrics
are copied from the source run; --restylize first stylizes the run's kept models again with the current code
(style_* and struct_* metrics; --structure off: the footprint path, the structure-off control; --snap-gate on|off:
the task #73 snap gate flags, stylize.SNAP_GATE_73; with --regeo the replayed GeoAnchors stand in for the anchors
models recorded before them lack). It refuses artefacts of another crowd3d.reconstruct.RECON_OUTPUT_VERSION.

The outline-corrected reference of each building is built once and cached under --reference-cache (run and
rescore share it); a cached building never loads its reference models again.

A test set's dev/test split (eval/splits/<split_id>.json) is recorded with each result (testset.split); compare
aggregates each split apart and judges the rules of a criteria file (eval/criteria/<id>.json) on each. The
OSM parts snapshot of a building's landmark (eval/landmarks.json osm_parts) is part of inputs_sha256. A
building run is not started with less than 3 GiB free disk under --work-root.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import math
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
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_real_landmark as ev  # noqa: E402
import materialize_testset as mt  # noqa: E402

from crowd3d.anchor import FLOORED  # noqa: E402
from crowd3d.anchor import calibration as anchor_calibration  # noqa: E402
from crowd3d.exif import read_photo_meta  # noqa: E402
from crowd3d.photoposes import uncertainty_rule  # noqa: E402
from crowd3d.reconstruct import RECON_OUTPUT_VERSION  # noqa: E402

log = logging.getLogger("benchmark")

REPO_ROOT = Path(__file__).resolve().parent.parent
HARNESS = Path(__file__).resolve().parent / "eval_real_landmark.py"
TESTSETS_DIR = REPO_ROOT / "eval" / "testsets"
CONFIGS_DIR = REPO_ROOT / "eval" / "configs"
RESULTS_DIR = REPO_ROOT / "eval" / "results"
SPLITS_DIR = REPO_ROOT / "eval" / "splits"
CATALOG = REPO_ROOT / "eval" / "landmarks.json"
_SCRATCH = Path("/tmp/claude-0/-home-user-temp/04ee4750-a5e3-5a4d-b130-9dfde89f5621/scratchpad/ten")
DEFAULT_DATA_ROOT = Path(os.environ.get("CROWD3D_TESTSET_ROOT", _SCRATCH / "testsets"))
DEFAULT_WORK_ROOT = Path(os.environ.get("CROWD3D_BENCH_DIR", _SCRATCH / "bench"))
RESULT_SCHEMA_VERSION = 1
TESTSET_SCHEMA_VERSION = mt.SCHEMA_VERSION
DEFAULT_TIME_CAP_S = 3600.0
SOFT_CAP_FRACTION = 0.8          # the harness starts no new upload request after this share of the hard cap
STATUS_OK, STATUS_CHANGED, STATUS_MISSING = "ok", "changed", "missing"
SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")
USER_AGENT = "Crowd3D-prototype/0.1 (https://github.com/h2zh/temp; research evaluation)"
CODE_PATHS = ("crowd3d", "scripts", "eval/configs")    # what a building run executes or reads (code_sha256)
RUN_KEY_FILE = "run_key.json"
MIN_FREE_BYTES = 3 * 2**30       # a building run is not started with less free disk than this under its out dir
# RECON_OUTPUT_VERSION of results recorded before it existed, by commit: their crowd3d pipeline files (reconstruct,
# qa, coverage, jobs, params, config, ingest, validation, store, api, exif, geo, records) equal db06c3c's, and
# the dirty trees of 3bbb0dd changed none of them (git diff, checked when version 1 was defined)
LEGACY_RECON_OUTPUT_VERSION = {"0a49b220fc4ec727b1264a42d6f1841a3e1fc6d8": 1,   # v1-merged
                               "1d6055b1ff8eddd5642bd5a954ebd346e1edb1ef": 1,   # commons-ten-v1 v1-after-fixes
                               "3bbb0dd38b7aa4f147c64dd07c3d88f032f76d91": 1}   # synthetic, mdx v1-after-fixes


# ---------------------------------------------------------------- small helpers


sha256_bytes, sha256_file = mt.sha256_bytes, mt.sha256_file


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


# ---------------------------------------------------------------- test sets


def load_testset(ref: str | Path) -> tuple[dict[str, Any], Path, str]:
    """materialize_testset.load_testset (test set, testset.json path, sha256 of its bytes), failing as a CLI error."""
    try:
        return mt.load_testset(ref)
    except (FileNotFoundError, ValueError) as e:
        raise SystemExit(str(e)) from None


def load_split(ts: dict[str, Any], ts_sha: str, ref: str | None = None) -> dict[str, Any] | None:
    """The test set's dev/test split as a result records it (id, path, sha256, dev, test): the split file ref
    (a path or an eval/splits id), else the test set's one file in eval/splits (None when it has none). The file
    must name this version of the test set and put every building in exactly one of dev and test."""
    tid = ts["testset_id"]
    if ref is not None:
        paths = [Path(ref) if Path(ref).is_file() else SPLITS_DIR / f"{ref}.json"]
        if not paths[0].is_file():
            raise SystemExit(f"no split {ref!r} (looked for {ref} and {paths[0]})")
    else:
        paths = [p for p in sorted(SPLITS_DIR.glob("*.json"))
                 if json.loads(p.read_text(encoding="utf-8")).get("testset_id") == tid]
        if not paths:
            return None
        if len(paths) > 1:
            raise SystemExit(f"{tid} has several splits ({', '.join(p.stem for p in paths)}): name one with --split")
    path = paths[0]
    split = json.loads(path.read_text(encoding="utf-8"))
    if (split.get("testset_id"), split.get("testset_sha256")) != (tid, ts_sha):
        raise SystemExit(f"{path} splits {split.get('testset_id')} {str(split.get('testset_sha256'))[:12]}, "
                         f"not {tid} {ts_sha[:12]}")
    if sorted(split["dev"] + split["test"]) != sorted(b["slug"] for b in ts["buildings"]):
        raise SystemExit(f"{path}: every building of {tid} must be in exactly one of dev and test")
    return {"id": split["split_id"], "path": _repo_relative(path), "sha256": sha256_file(path),
            "dev": split["dev"], "test": split["test"]}


def osm_parts_sha(slug: str, catalog: Path = CATALOG) -> str | None:
    """sha256 of the landmark's OSM parts snapshot (the osm_parts block of eval/landmarks.json); None when it has
    none. A snapshot whose bytes are not the catalog's is refused."""
    entry = next((e for e in json.loads(catalog.read_text(encoding="utf-8")) if e["slug"] == slug), None)
    block = (entry or {}).get("osm_parts")
    if not block:
        return None
    path = REPO_ROOT / block["file"]
    got = sha256_file(path) if path.is_file() else None
    if got != block["sha256"]:
        raise SystemExit(f"{slug}: {path} is {'missing' if got is None else 'not the snapshot the catalog lists'}")
    return got


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
    ev.check_outside_repo(args.data_root, "--data-root")
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
            inputs_sha: str, stylize: bool, time_cap_s: float, max_requests: int | None = None) -> dict[str, Any]:
    """What a building run depends on; --resume reuses a finished run only when its saved key is equal."""
    return {"code_sha256": code_sha, "harness_version": ev.HARNESS_VERSION,
            "metric_schema_version": ev.METRIC_SCHEMA_VERSION, "testset_sha256": testset_sha, "config_sha256": cfg_sha,
            "effective_sha256": effective_sha, "inputs_sha256": inputs_sha, "spec_sha256": canonical_sha256(spec),
            "stylize": stylize, "time_cap_s": time_cap_s, "max_requests": max_requests}


def request_caps(tokens: list[str] | None, slugs: list[str]) -> dict[str, int | None]:
    """--max-requests N (every building) and/or SLUG=N (that building) -> building slug -> cap (None: no cap)."""
    caps: dict[str, int | None] = dict.fromkeys(slugs)
    for token in tokens or []:
        slug, _, n = token.rpartition("=")
        if not n.isdigit() or int(n) < 1 or (slug and slug not in caps):
            raise SystemExit(f"--max-requests {token!r}: expected N or SLUG=N with N >= 1 and SLUG one of {slugs}")
        for s in ([slug] if slug else slugs):
            if slug or caps[s] is None:
                caps[s] = int(n)
    return caps


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


def reference_cache_spec(cache_dir: Path, ts_sha: str, slug: str, gt_integrity: dict[str, Any]) -> dict[str, Any]:
    """Where the building's outline-corrected reference is cached (eval_real_landmark.building_reference) and its
    key: the test set's sha256 (photos, reference image maps and sparse file hashes, outlines) and REFERENCE_VERSION.
    The harness version is not in it, so a metric change keeps the cache (and the reference models can go)."""
    key = canonical_sha256({"testset_sha256": ts_sha, "slug": slug, "reference_version": ev.REFERENCE_VERSION})
    return {"path": str(cache_dir / f"{slug}.json"), "key": key,
            "about": {"slug": slug, "testset_sha256": ts_sha, "integrity": gt_integrity}}


def reference_integrity(gt_integrity: dict[str, Any], cache: dict[str, Any]) -> dict[str, Any]:
    """The reference integrity a run depends on: the one stored with the building's cached reference when the cache
    is valid (so deleting the reference models once cached keeps inputs_sha256), else gt_integrity."""
    path = Path(cache["path"])
    try:
        doc = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    except ValueError:
        doc = {}
    return doc["integrity"] if doc.get("key") == cache["key"] and "integrity" in doc else gt_integrity


def inputs_fingerprint(integrity: dict[str, Any], use_changed: bool, outline_sha: str | None,
                       gt_integrity: dict[str, Any], parts_sha: str | None = None) -> str:
    """sha256 over exactly what a building run consumed: used photos with their bytes' sha256, outline, reference,
    and the OSM parts snapshot when the landmark has one (without one the fingerprint is the one recorded before
    snapshots were part of it)."""
    keep = {STATUS_OK, STATUS_CHANGED} if use_changed else {STATUS_OK}
    used = sorted((r["key"], r["sha256"]) for r in integrity["rows"] if r["status"] in keep)
    return canonical_sha256({"photos": used, "outline": outline_sha, "gt": gt_integrity.get("files"),
                             **({"osm_parts": parts_sha} if parts_sha else {})})


def compact_details(summary: dict[str, Any]) -> dict[str, Any]:
    """The diagnostic part of a harness summary that is small enough to commit (no per-photo rows)."""
    models = []
    style_by_id = {r["model_id"]: r for r in ((summary.get("style") or {}).get("records") or [])}
    geo_by_id = {r["model_id"]: r for r in ((summary.get("geo") or {}).get("models") or [])}
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
        if m.get("cache_stats"):
            models[-1]["cache_stats"] = m["cache_stats"]
        if m["id"] in style_by_id:
            models[-1]["style"] = compact_style(style_by_id[m["id"]])
        models[-1].update({k: v for k, v in (geo_by_id.get(m["id"]) or {}).items() if k in ("geo", "coverage")})
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
        **{k: summary[k] for k in ("max_requests", "requests_planned", "cache") if summary.get(k) is not None},
        "style": {k: (summary.get("style") or {}).get(k) for k in ("best", "reference", "structure", "seconds", "note")}
        if summary.get("style") else None,
        "poses": (summary.get("poses") or {}).get("summary"),
        "coverage": (summary.get("geo") or {}).get("coverage"),
        **({"regeo": summary["geo"]["regeo"]} if "regeo" in (summary.get("geo") or {}) else {}),
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
            "diagnosis": r.get("diagnosis"), **({"structure": r["structure"]} if r.get("structure") else {})}


def run_building_process(spec: dict[str, Any], cfg_path: Path, out_dir: Path, time_cap_s: float,
                         stylize: bool = True, key: dict[str, Any] | None = None,
                         resume: bool = False, max_requests: int | None = None) -> dict[str, Any]:
    """Run the harness on one building in a child process under a hard time cap; returns status + summary.
    With resume, a finished run in out_dir whose saved run key equals key is reused instead."""
    out_dir.mkdir(parents=True, exist_ok=True)
    key_path, summary_path = out_dir / RUN_KEY_FILE, out_dir / "summary.json"
    if resume and key is not None and key_path.is_file() and summary_path.is_file():
        saved = json.loads(key_path.read_text(encoding="utf-8"))
        if saved.get("key") == key:
            return {"status": "ok", "wall_s": saved["wall_s"], "log": str(out_dir / "harness.log"), "resumed": True,
                    "summary": json.loads(summary_path.read_text(encoding="utf-8"))}
    free = shutil.disk_usage(out_dir).free
    if free < MIN_FREE_BYTES:
        raise SystemExit(f"{free / 2**30:.1f} GiB free under {out_dir}: a building run needs {MIN_FREE_BYTES / 2**30:g} "
                         "GiB free to start (free some space, then run again with --resume)")
    spec_path = out_dir / "spec.json"
    spec_path.write_text(json.dumps(spec, indent=1, ensure_ascii=False), encoding="utf-8")
    for stale in ("summary.json", "progress.json", RUN_KEY_FILE):
        (out_dir / stale).unlink(missing_ok=True)
    cmd = [sys.executable, "-I", str(HARNESS), "--building-spec", str(spec_path), "--config", str(cfg_path),
           "--out", str(out_dir), "--overwrite", "--time-cap", str(round(SOFT_CAP_FRACTION * time_cap_s))]
    if stylize:
        cmd.append("--stylize")
    if max_requests is not None:
        cmd += ["--max-requests", str(max_requests)]
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
    ev.check_outside_repo(args.work_root, "--work-root")
    ev.check_outside_repo(args.data_root, "--data-root")
    cache_dir = args.reference_cache or args.work_root / tid / "references"
    ev.check_outside_repo(cache_dir, "--reference-cache")
    use_changed = (args.on_changed or ts.get("integrity_policy", "skip")) == "use"
    split = load_split(ts, ts_sha, args.split)
    run_dir = args.work_root / tid / args.name
    started, t0 = utc_now(), time.monotonic()
    code_sha, cfg_sha, effective_sha = code_fingerprint(), sha256_file(cfg_file), canonical_sha256(effective)
    buildings: dict[str, dict[str, Any]] = {}
    repeats: dict[str, list[dict[str, Any]]] = {}
    integrity_all: dict[str, Any] = {}
    selected = select_buildings(ts, args.buildings)
    caps = request_caps(args.max_requests, [b["slug"] for b in selected])
    for b in selected:
        slug = b["slug"]
        bdir = args.data_root / tid / slug
        if b.get("generator") and verify_photos(b, bdir)[STATUS_MISSING]:
            log.info("%s: generating the synthetic photos into %s", slug, bdir)
            materialize_synthetic(b, bdir)
        integ = verify_photos(b, bdir)
        outline, outline_want = building_outline(b, ts_path.parent)
        outline_sha = sha256_file(outline) if outline and outline.is_file() else None
        gt, gt_integ = resolve_gt(b, bdir, args.data_root, args.gt_root)
        parts_sha = osm_parts_sha(slug)
        integrity_all[slug] = {
            **{k: integ[k] for k in ("photos", STATUS_OK, STATUS_CHANGED, STATUS_MISSING, "changed_keys",
                                     "missing_keys")},
            "outline": (STATUS_MISSING if outline_sha is None and outline else
                        STATUS_OK if outline_want in (None, outline_sha) else STATUS_CHANGED),
            "reference": {k: v for k, v in gt_integ.items() if k != "sparse_dir"},
            "photos_used": integ[STATUS_OK] + (integ[STATUS_CHANGED] if use_changed else 0),
            "osm_parts_sha256": parts_sha,
        }
        spec = building_spec(b, ts_path.parent, integ, gt, use_changed)
        spec["reference_cache"] = reference_cache_spec(cache_dir, ts_sha, slug, gt_integ)
        ref_integ = reference_integrity(gt_integ, spec["reference_cache"])
        record: dict[str, Any] = {"integrity": integrity_all[slug],
                                  "inputs_sha256": inputs_fingerprint(integ, use_changed, outline_sha, ref_integ,
                                                                      parts_sha)}
        key = run_key(spec, code_sha=code_sha, testset_sha=ts_sha, cfg_sha=cfg_sha, effective_sha=effective_sha,
                      inputs_sha=record["inputs_sha256"], stylize=not args.no_stylize, time_cap_s=args.time_cap,
                      max_requests=caps[slug])
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
                                       key=key, resume=args.resume, max_requests=caps[slug])
            summary = res.pop("summary", None)
            if summary is not None:
                got = json.loads(json.dumps(summary["effective_params"]))
                want = json.loads(json.dumps({k: effective[k] for k in ("reconstructor", "settings", "params")}))
                if got != want:
                    raise SystemExit(f"{slug}: the harness ran with other parameters than the config: {got} != {want}")
                res["metrics"] = summary["metrics"]
                res["details"] = compact_details(summary)
                res["calibration"] = [{"run": out_dir.name or "r1", **row}
                                      for row in (summary.get("geo") or {}).get("calibration") or []]
                res["artefacts"] = {"summary_json": str(out_dir / "summary.json"),
                                    "summary_md": str(out_dir / "summary.md"),
                                    "topdown_png": summary["topdown_png"], "data_dir": summary["data_dir"]}
            runs.append(res)
            log.info("%s: %s in %.0f s%s", slug, res["status"], res["wall_s"],
                     " (resumed: the earlier run's result, same run key)" if res.get("resumed") else "")
        first = runs[0]
        record.update(status=first["status"], wall_s=first["wall_s"], metrics=first.get("metrics"),
                      details=first.get("details"), artefacts=first.get("artefacts"), log=first["log"],
                      calibration=[row for x in runs for row in x.get("calibration") or []])
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
        "testset": {"id": tid, "sha256": ts_sha, "path": _repo_relative(ts_path), "split": split,
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
                       "recon_output_version": RECON_OUTPUT_VERSION,
                       "time_cap_s": args.time_cap, "repeat": args.repeat, "stylize": not args.no_stylize,
                       "max_requests": {s: n for s, n in caps.items() if n is not None} or None,
                       # the stylize step goes through crowd3d.structure_pipeline (OSM parts where there is a snapshot)
                       "structure": "on" if not args.no_stylize else None,
                       "photo_pose_uncertainty_rule": uncertainty_rule(), "command": " ".join(sys.argv)},
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
                     f"{m.get('style_registration_right')}, wrong {m.get('style_registration_wrong')}, refused although"
                     f" placed right {m.get('style_rejected_placed_right')}, not eligible but passed"
                     f" {m.get('style_gate_passed_not_eligible')}), walls {m.get('style_walls_coloured')}/"
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


def split_of(run: dict[str, Any], slug: str) -> str:
    """"test" when the result's recorded split (testset.split) holds the building out, else "dev" (a result recorded
    without a split is all dev)."""
    return "test" if slug in ((run["testset"].get("split") or {}).get("test") or ()) else "dev"


def compare_results(a: dict[str, Any], b: dict[str, Any], sources: list[dict[str, Any]], factor: float = 1.0,
                    force: bool = False, exclude_groups: tuple[str, ...] = ()) -> dict[str, Any]:
    """Per building, aggregate and per split (dev / test), metric by metric. Refuses (ValueError) different test
    sets or metric schemas, or a building that is dev in one result and test in the other, unless force; buildings
    whose inputs differ are listed and left out of the aggregates. exclude_groups: metric groups left out."""
    problems = []
    if a["testset"]["sha256"] != b["testset"]["sha256"]:
        problems.append(f"test sets differ: {a['testset']['id']} {a['testset']['sha256'][:12]} vs "
                        f"{b['testset']['id']} {b['testset']['sha256'][:12]}")
    if a["metric_schema_version"] != b["metric_schema_version"]:
        problems.append(f"metric_schema_version differs: {a['metric_schema_version']} vs {b['metric_schema_version']}")
    common = [s for s in a["buildings"] if s in b["buildings"] and a["buildings"][s].get("metrics")
              and b["buildings"][s].get("metrics")]
    moved = [s for s in common if split_of(a, s) != split_of(b, s)]
    if moved:
        problems.append("split differs: " + ", ".join(f"{s} is {split_of(a, s)} in A, {split_of(b, s)} in B"
                                                      for s in moved))
    if problems and not force:
        raise ValueError("; ".join(problems) + " (use --force to compare anyway)")
    shown = [m for m in ev.METRICS if m.group not in exclude_groups]
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

    def presence_for(slugs: list[str] | None, name: str) -> bool:
        return any(name in (s.get("presence_varies") or {}).get(x, []) for s in sources
                   for x in (slugs if slugs is not None else (s.get("presence_varies") or {})))

    def noise_for(slug: str | None, name: str) -> float | None:
        vals = []
        for s in sources:
            v = (s["per_building"].get(slug) or {}).get(name) if slug else s["aggregate"].get(name)
            if v is None and slug is None:
                v = s["per_metric"].get(name)
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    def noise_bound(slugs: list[str], name: str) -> float | None:
        # the noise of an aggregate over some of the buildings, bounded by their own spreads: their sum for a
        # summed metric, the largest otherwise (a median, mean or maximum moves by at most that much)
        vals = []
        for s in sources:
            spreads = [v for v in ((s["per_building"].get(x) or {}).get(name) for x in slugs) if v is not None]
            if spreads:
                vals.append(sum(spreads) if ev.METRIC_BY_NAME[name].aggregate == "sum" else max(spreads))
        return _round(max(vals)) if vals else None

    same_inputs = [s for s in common if a["buildings"][s]["inputs_sha256"] == b["buildings"][s]["inputs_sha256"]]
    def style_incomplete(run: dict[str, Any], slug: str) -> bool:
        return bool(run["buildings"][slug]["metrics"].get("style_incomplete"))

    def incomplete(m: ev.Metric, a_: Any, b_: Any) -> dict[str, Any]:
        # a stylization cut short by its time cap covers only some models: not a change of the metric
        return {"a": a_, "b": b_, "delta": None, "verdict": "incomplete", "noise": None, "threshold": None,
                "significant": False}

    per_building = {}
    for slug in common:
        ma, mb = a["buildings"][slug]["metrics"], b["buildings"][slug]["metrics"]
        cut = style_incomplete(a, slug) or style_incomplete(b, slug)
        per_building[slug] = {
            "inputs_identical": slug in same_inputs, "split": split_of(b, slug),
            "metrics": {m.name: (incomplete(m, ma.get(m.name), mb.get(m.name)) if cut and m.group == "style"
                                 and m.name != "style_incomplete" else
                                 judge(m, ma.get(m.name), mb.get(m.name), noise_for(slug, m.name), factor,
                                       presence_for([slug], m.name))) for m in shown},
        }

    def judged(slugs: list[str], whole: bool) -> dict[str, dict[str, Any]]:
        agg_a = aggregate({s: a["buildings"][s]["metrics"] for s in slugs})
        agg_b = aggregate({s: b["buildings"][s]["metrics"] for s in slugs})
        styled = [s for s in slugs if not style_incomplete(a, s) and not style_incomplete(b, s)]
        if len(styled) < len(slugs):      # style metrics aggregate over the buildings stylized in full in both
            sa = aggregate({s: a["buildings"][s]["metrics"] for s in styled})
            sb = aggregate({s: b["buildings"][s]["metrics"] for s in styled})
            for m in shown:
                if m.group == "style" and m.name != "style_incomplete":
                    agg_a[m.name], agg_b[m.name] = sa[m.name], sb[m.name]
        return {m.name: judge(m, agg_a[m.name], agg_b[m.name],
                              noise_for(None, m.name) if whole else noise_bound(slugs, m.name), factor,
                              presence_for(None if whole else slugs, m.name)) for m in shown}

    agg = judged(same_inputs, True)
    split_buildings = {label: [s for s in same_inputs if split_of(b, s) == label] for label in ("dev", "test")}
    split_buildings = {label: slugs for label, slugs in split_buildings.items() if slugs}
    splits = {label: agg if len(slugs) == len(same_inputs) else judged(slugs, False)
              for label, slugs in split_buildings.items()}
    rows = [j for pb in per_building.values() for j in pb["metrics"].values()] + list(agg.values())
    return {
        "a": {"run": a["run_name"], "config": a["config"]["id"], "started": a["provenance"]["started"]},
        "b": {"run": b["run_name"], "config": b["config"]["id"], "started": b["provenance"]["started"]},
        "testset": a["testset"]["id"], "problems": problems, "parameter_differences": param_diff,
        "effective_identical": a["config"]["effective_sha256"] == b["config"]["effective_sha256"],
        "noise_sources": [f"{s['run']} ({s['config']}, {s['repeats']} repeats)" for s in sources],
        "excluded_groups": sorted(exclude_groups),
        "buildings_compared": common, "buildings_inputs_differ": [s for s in common if s not in same_inputs],
        "buildings_only_in_a": [s for s in a["buildings"] if s not in common],
        "buildings_only_in_b": [s for s in b["buildings"] if s not in common],
        "aggregate": agg, "splits": splits, "split_buildings": split_buildings, "per_building": per_building,
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
    if c["excluded_groups"]:
        lines.append(f"Metric groups left out (--exclude-group): {', '.join(c['excluded_groups'])}")
    held_out = len(c["splits"]) > 1 or "test" in c["splits"]

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
    if len(c["splits"]) > 1:
        for label, rows in c["splits"].items():
            lines += ["", f"Split {label}: aggregate over {', '.join(c['split_buildings'][label])} (noise bounded by"
                          " their own spreads):"] + table(rows, True)
    for slug, pb in c["per_building"].items():
        tag = f" [{pb['split']}]" if held_out else ""
        lines += ["", f"{slug}{tag}{'' if pb['inputs_identical'] else ' (inputs differ)'}:"] + table(pb["metrics"],
                                                                                                    show_all)
    lines += ["", f"Significant changes: {c['significant_worse']} worse, {c['significant_better']} better."]
    if c.get("criteria"):
        lines += [""] + format_criteria(c["criteria"])
    return "\n".join(lines)


# ---------------------------------------------------------------- pass criteria (eval/criteria)

CRITERIA_DIR = REPO_ROOT / "eval" / "criteria"
CRITERIA_ROLES = ("primary", "guard", "calibration", "cost", "secondary", "info", "context")
REPORT_ROLES = ("secondary", "info", "context")      # shown with the comparison, never decide the verdict
RELATIVE = ("better", "not_worse", "not_worse_per_building", "max_ratio")   # B against A; the rest judge B alone


def _requirements(rule: dict[str, Any]) -> dict[str, Any]:
    """{split: requirement} of a rule: "require" names one for both splits, or one per split ({"dev": ..., "test": ...}).
    A rule of a report-only role has none ("report")."""
    req = rule.get("require", "report")
    if isinstance(req, dict) and req and set(req) <= {"dev", "test"}:
        return dict(req)
    return {"dev": req, "test": req}


def _kind(req: Any) -> str:
    return req if isinstance(req, str) else next((k for k in ("max_ratio", "interval", "max_share") if k in req), "?")


def load_criteria(ref: str | Path) -> dict[str, Any]:
    """A criteria file (a path or an eval/criteria id) with its repo path and sha256, every rule checked against
    METRICS: a known metric and role, a known requirement (interval only on a pooled metric; max_share of a metric)."""
    path = Path(ref) if Path(ref).is_file() else CRITERIA_DIR / f"{ref}.json"
    if not path.is_file():
        raise SystemExit(f"no criteria {ref!r} (looked for {ref} and {path})")
    doc = json.loads(path.read_text(encoding="utf-8"))
    for rule in doc["rules"]:
        m = ev.METRIC_BY_NAME.get(rule["metric"])
        bad = None if m else "unknown metric"
        if not bad and rule.get("role") not in CRITERIA_ROLES:
            bad = f"role {rule.get('role')!r} is not one of {', '.join(CRITERIA_ROLES)}"
        if not bad and (rule["role"] in REPORT_ROLES) != ("require" not in rule):
            bad = "a report-only role has no requirement, every other role needs one"
        for req in ([] if bad else _requirements(rule).values()):
            kind = _kind(req)
            if kind not in (*RELATIVE, "zero", "interval", "max_share", "report"):
                bad = f"unknown requirement {req!r}"
            elif kind == "interval" and m.aggregate != "pooled":
                bad = "an interval needs a pooled metric"
            elif kind == "max_share" and req.get("of") not in ev.METRIC_BY_NAME:
                bad = f"max_share of unknown metric {req.get('of')!r}"
        if bad:
            raise SystemExit(f"{path}: rule {rule.get('metric')!r}: {bad}")
    return {**doc, "path": _repo_relative(path), "sha256": sha256_file(path)}


def wilson_interval(k: float, n: float, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of the proportion k / n."""
    p, d = k / n, 1.0 + z * z / n
    centre, half = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def apply_criteria(crit: dict[str, Any], c: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Each rule of a criteria file on each split (dev / test) the comparison c of A with B (compare_results) has,
    as PASS, FAIL, INSUFFICIENT (a value or enough models missing), REPORT (report-only roles) or EXCLUDED (a group
    left out). Relative requirements read the split aggregates of the buildings with identical inputs in both runs
    (and, per building, those buildings); zero, interval and max_share read every building of B in the split (zero
    each repeat of it: a median over repeats would hide one repeat's violation). An interval passes when the pooled
    share reaches its lower end and its 95 % Wilson interval reaches the upper end.
    Verdict: FAIL when a rule fails, else INSUFFICIENT when one lacks data, else PASS."""
    mine: dict[str, dict[str, dict[str, Any]]] = {}
    repeats: dict[str, list[dict[str, Any]]] = {}
    for slug, rec in b["buildings"].items():
        if rec.get("metrics"):
            mine.setdefault(split_of(b, slug), {})[slug] = rec["metrics"]
            repeats[slug] = [r["metrics"] for r in rec.get("repeats") or [] if r.get("metrics")] or [rec["metrics"]]
    own = {label: aggregate(ms) for label, ms in mine.items()}
    rows = []
    for rule in crit["rules"]:
        m = ev.METRIC_BY_NAME[rule["metric"]]
        for label, req in _requirements(rule).items():
            if label not in c["splits"] and label not in mine:
                continue                                    # no building of this split in the comparison
            kind, j = _kind(req), (c["splits"].get(label) or {}).get(m.name)
            row = {"metric": m.name, "role": rule["role"], "split": label, "require": req, "a": j["a"] if j else None,
                   "b": j["b"] if j and kind in (*RELATIVE, "report") else (own.get(label) or {}).get(m.name)}
            if m.group in c["excluded_groups"]:
                row.update(outcome="EXCLUDED", detail="group left out")
            elif kind == "report":
                row.update(outcome="REPORT", detail=f"{j['verdict']}{' (significant)' if j['significant'] else ''}"
                           if j else "no building in both runs")
            elif kind in RELATIVE:
                row.update(_relative(kind, req, m, j, label, c))
            else:
                row.update(_absolute(kind, req, m, mine.get(label) or {}, own.get(label) or {}, repeats))
            rows.append(row)
    outcomes = {r["outcome"] for r in rows if r["outcome"] not in ("REPORT", "EXCLUDED")}
    verdict = "FAIL" if "FAIL" in outcomes else "INSUFFICIENT" if "INSUFFICIENT" in outcomes or not outcomes else "PASS"
    return {"id": crit["criteria_id"], "path": crit["path"], "sha256": crit["sha256"], "verdict": verdict,
            "splits": sorted({r["split"] for r in rows}), "rules": rows}


def _relative(kind: str, req: Any, m: ev.Metric, j: dict[str, Any] | None, label: str,
              c: dict[str, Any]) -> dict[str, str]:
    if j is None:
        return {"outcome": "INSUFFICIENT", "detail": "no building of this split in both runs with identical inputs"}
    if j["a"] is None or j["b"] is None:
        return {"outcome": "INSUFFICIENT", "detail": f"no value in {'A' if j['a'] is None else 'B'}"}
    change = f"{j['verdict']}, delta {j['delta']:+g} (threshold {j['threshold']:g})"
    if kind == "better":
        ok = j["verdict"] == "better" and j["significant"]
        return {"outcome": "PASS" if ok else "FAIL", "detail": change}
    if kind == "max_ratio":
        limit = req["max_ratio"] * float(j["a"])
        ok = float(j["b"]) <= limit or float(j["b"]) - float(j["a"]) <= m.min_delta
        return {"outcome": "PASS" if ok else "FAIL", "detail": f"B / A = {float(j['b']) / float(j['a']):.3g}"
                if float(j["a"]) else f"A = 0, B = {j['b']}"}
    worse = [] if not (j["verdict"] == "worse" and j["significant"]) else ["aggregate"]
    if kind == "not_worse_per_building":
        worse += [slug for slug, pb in c["per_building"].items() if pb["inputs_identical"] and pb["split"] == label
                  and pb["metrics"][m.name]["verdict"] == "worse" and pb["metrics"][m.name]["significant"]]
    return {"outcome": "FAIL" if worse else "PASS",
            "detail": f"significantly worse: {', '.join(worse)}" if worse else change}


def _absolute(kind: str, req: Any, m: ev.Metric, buildings: dict[str, dict[str, Any]], agg: dict[str, Any],
              repeats: dict[str, list[dict[str, Any]]]) -> dict[str, str]:
    if not buildings:
        return {"outcome": "INSUFFICIENT", "detail": "no building of this split in B"}
    if kind == "zero":
        def worst(slug: str, name: str) -> Any:     # the largest value over the building's repeats
            runs = repeats.get(slug) or [buildings[slug]]
            return max((r[name] for r in runs if r.get(name) is not None), default=None)

        def where(slug: str) -> str:
            n = len(repeats.get(slug) or ())
            return f" (largest of {n} repeats)" if n > 1 else ""

        bad = [f"{s} {worst(s, m.name):g}{where(s)}" for s in buildings if (worst(s, m.name) or 0) > 0]
        missing = [s for s, v in buildings.items() if v.get(m.name) is None]
        # a stylization its time cap cut short has not looked at every model
        cut = [s for s in buildings if m.group == "style" and (worst(s, "style_incomplete") or 0) > 0]
        if bad:
            return {"outcome": "FAIL", "detail": ", ".join(bad)}
        if missing or cut:
            return {"outcome": "INSUFFICIENT", "detail": "; ".join(
                x for x in (missing and f"no value: {', '.join(missing)}",
                            cut and f"stylization incomplete: {', '.join(cut)}") if x)}
        return {"outcome": "PASS", "detail": f"0 on {len(buildings)} building(s)"}
    num, den = m.pooled if kind == "interval" else (m.name, req["of"])
    k, n = agg.get(num), agg.get(den)
    if k is None or not n or n < req.get("min_n", 1):
        return {"outcome": "INSUFFICIENT", "detail": f"{n or 0:g} < {req.get('min_n', 1)} models"}
    if kind == "interval":
        # an overconfident sigma is the harmful side (the range shown is too small): the share itself must reach the
        # lower end; above the upper end it fails only when the whole Wilson interval does (a right 2-sigma range
        # holds 20 of 20 models 39 % of the time)
        lo, hi = wilson_interval(float(k), float(n))
        ok = k / n >= req["interval"][0] and lo <= req["interval"][1]
        return {"outcome": "PASS" if ok else "FAIL",
                "detail": f"{k:g}/{n:g} = {k / n:.3f}, 95% Wilson {lo:.3f}-{hi:.3f}"}
    return {"outcome": "PASS" if k / n <= req["max_share"] else "FAIL", "detail": f"{k:g}/{n:g} = {k / n:.3f}"}


def format_criteria(cr: dict[str, Any]) -> list[str]:
    def req_text(req: Any) -> str:
        kind = _kind(req)
        if kind == "max_ratio":
            return f"B <= {req['max_ratio']:g} x A"
        if kind == "interval":
            return f"in [{req['interval'][0]:g}, {req['interval'][1]:g}], n >= {req.get('min_n', 1)}"
        if kind == "max_share":
            return f"<= {req['max_share']:g} of {req['of']}, n >= {req.get('min_n', 1)}"
        return kind

    def cell(v: Any) -> str:
        return "-" if v is None else f"{v:g}" if isinstance(v, float) else str(v)

    lines = [f"Criteria {cr['id']} ({cr['path']}), splits {', '.join(cr['splits']) or 'none'}: {cr['verdict']}",
             "| metric | role | split | requirement | A | B | result | detail |", "|---|---|---|---|---|---|---|---|"]
    for r in cr["rules"]:
        lines.append(f"| {r['metric']} | {r['role']} | {r['split']} | {req_text(r['require'])} | {cell(r['a'])} | "
                     f"{cell(r['b'])} | {r['outcome']} | {r['detail']} |")
    return lines


def cmd_compare(args: argparse.Namespace) -> int:
    unknown = sorted(set(args.exclude_group or []) - {m.group for m in ev.METRICS})
    if unknown:
        print(f"refused: unknown metric group(s) {', '.join(unknown)}", file=sys.stderr)
        return 2
    crit = load_criteria(args.criteria) if args.criteria else None
    a, b = load_result(args.run_a), load_result(args.run_b)
    sources = noise_sources([a, b], args.noise or [], args.results_dir or RESULTS_DIR)
    try:
        c = compare_results(a, b, sources, args.noise_factor, args.force, tuple(args.exclude_group or ()))
    except ValueError as e:
        print(f"refused: {e}", file=sys.stderr)
        return 2
    if crit:
        c["criteria"] = apply_criteria(crit, c, b)
    print(format_compare(c, args.all))
    if args.json:
        args.json.write_text(json.dumps(c, indent=1, ensure_ascii=False), encoding="utf-8")
    if crit and c["criteria"]["verdict"] != "PASS":
        return 1
    return 1 if args.fail_on_worse and c["significant_worse"] else 0


# ---------------------------------------------------------------- report

LEADERBOARD_COLUMNS = ("models_published", "models_qa_rejected", "facades_published", "full_model_published",
                       "registration_rate", "estimate_coverage", "outline_rotation_abs_deg", "outline_shift_m",
                       "gt_rotation_error_median_deg", "gt_gross_shown", "style_models_eligible",
                       "style_registration_pass_rate", "style_registration_right", "style_registration_wrong",
                       "style_rejected_placed_right", "style_walls_coloured", "style_qa_failed", "wall_time_s")
BUILDING_COLUMNS = ("photos_accepted", "models_published", "models_qa_rejected", "facades_published",
                    "full_model_published", "registration_rate", "estimate_coverage", "outline_rotation_abs_deg",
                    "outline_shift_m", "outline_residual_m", "gps_residual_median_m", "compass_error_median_deg",
                    "gt_rotation_error_median_deg", "gt_gross_cameras", "gt_gross_shown", "style_models_eligible",
                    "style_registration_passed", "style_registration_right", "style_registration_wrong",
                    "style_rejected_placed_right", "style_gate_passed_not_eligible", "style_snap_refused_gps",
                    "style_snap_refused_ambiguous", "style_snap_residual_before_m", "style_ref_camera_median_worst_m",
                    "style_walls_coloured", "style_walls_total", "style_qa_failed", "wall_time_s", "style_time_s")
# the review section (schema >= 7, per run and split): the primary coverage metric, the guards, calibration and cost
# of eval/criteria/review-v1.json (docs/BENCHMARK.md 6.2)
REVIEW_COLUMNS = ("cov_local_supported_m_1p5", "cov_local_majority_m_1p5", "cov_geo_supported_m_5",
                  "cov_local_wrong_m_1p5", "cov_local_wrong_sum_m_1p5", "cov_geo_wrong_m_5", "cov_geo_wrong_sum_m_5",
                  "geo_camera_error_median_m", "wrong_face_models_published", "wrong_face_cameras_shown",
                  "photo_pose_correct_rate_testset", "calib_yaw_cover2", "calib_scale_cover2", "style_registration_wrong",
                  "style_coloured_walls_wrong", "style_qa_failed", "struct_incorrect", "stale_overwrites", "cpu_time_s",
                  "request_latency_p90_s", "peak_rss_jobs_mb", "disk_run_mb")
# the photo-pose section of the leaderboard: per run and building (docs/BENCHMARK.md "照片位姿")
POSE_COLUMNS = ("photo_pose_testset_photos", "photos_uploaded", "photo_pose_determined",
                "photo_pose_determined_rate_testset", "photo_pose_sfm", "photo_pose_exif", "photo_pose_gps_only",
                "photo_pose_gt_testset", "photo_pose_gt", "photo_pose_correct", "photo_pose_correct_rate",
                "photo_pose_correct_rate_testset", "photo_pose_correct_strict_rate", "photo_pose_wrong",
                "photo_pose_pass_rate_low_testset", "photo_pose_pass_rate_high_testset",
                "photo_pose_position_error_median_m", "photo_pose_heading_error_median_deg",
                "photo_pose_uncertainty_coverage", "photo_pose_sfm_uncertainty_coverage",
                "photo_pose_exif_uncertainty_coverage")


def _cell(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    return str(v)


def split_aggregates(r: dict[str, Any]) -> list[tuple[str, list[str], dict[str, Any]]]:
    """(split, its buildings, their aggregate) of a result: dev and test apart when buildings of both ran, else one
    row with every building and the recorded aggregate (labelled with the split of the buildings that ran)."""
    groups: dict[str, list[str]] = {}
    for slug in r["buildings"]:
        groups.setdefault(split_of(r, slug), []).append(slug)
    ran = [k for k in ("dev", "test") if any(r["buildings"][s].get("metrics") for s in groups.get(k, []))]
    if len(ran) < 2:
        return [(ran[0] if ran else "dev", list(r["buildings"]), r["aggregate"])]
    return [(k, groups[k], aggregate({s: r["buildings"][s]["metrics"] for s in groups[k]
                                      if r["buildings"][s].get("metrics")})) for k in ran]


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
        "划分：结果记录的 dev / test（`testset.split`，没记的全算 dev）；两个划分都有建筑跑过的结果分两行汇总。",
        "",
        "| 测试集 | 运行 | 配置 | 划分 | 开始时间 (UTC) | 提交 | 建筑 成功/总数 | "
        + " | ".join(f"{c} ({unit[c]})" for c in LEADERBOARD_COLUMNS) + " | 备注 |",
        "|" + "---|" * (8 + len(LEADERBOARD_COLUMNS)),
    ]
    results = sorted(results, key=lambda r: (r["testset"]["id"], r["provenance"]["started"], r["run_name"]))
    for r in results:
        g = r["provenance"]["git"]
        commit = (g.get("commit") or "?")[:7] + ("+dirty" if g.get("dirty") else "")
        for label, slugs, agg in split_aggregates(r):
            ok = sum(1 for s in slugs if r["buildings"][s].get("metrics"))
            lines.append(f"| {r['testset']['id']} | {r['run_name']} | {r['config']['id']} | {label} | "
                         f"{r['provenance']['started'][:16].replace('T', ' ')} | {commit} | {ok}/{len(slugs)} | "
                         + " | ".join(_cell(agg.get(c)) for c in LEADERBOARD_COLUMNS)
                         + f" | {(r.get('note') or '').replace('|', '/')} |")
    newest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        newest[(r["testset"]["id"], r["config"]["id"])] = r
    lines += ["", "## 每个配置最新一次运行的分建筑结果", ""]
    for (tid, cid), r in sorted(newest.items()):
        lines += [f"### {tid} · {cid} · {r['run_name']}", "",
                  "| 建筑 | 划分 | 状态 | " + " | ".join(BUILDING_COLUMNS) + " |",
                  "|" + "---|" * (3 + len(BUILDING_COLUMNS))]
        for slug, b in r["buildings"].items():
            m = b.get("metrics") or {}
            lines.append(f"| {slug} | {split_of(r, slug)} | {b['status']} | "
                         + " | ".join(_cell(m.get(c)) for c in BUILDING_COLUMNS) + " |")
        lines.append("")
    lines += review_leaderboard(results)
    lines += pose_leaderboard(results)
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


def review_leaderboard(results: list[dict[str, Any]]) -> list[str]:
    """The review section: every run of metric schema 7 or later, per split."""
    runs = [r for r in results if r["metric_schema_version"] >= 7]
    if not runs:
        return []
    lines = ["## 评审指标（指标第 7 版起，按划分汇总）", "",
             "外部评审采纳计划的通过标准（`eval/criteria/review-v1.json`，`compare --criteria review-v1`）看的数："
             "主要指标是建筑局部 1.5 m 的支持覆盖 `cov_local_supported_m_1p5`（负责人问题 1 的默认答复），硬守卫"
             "（`style_registration_wrong`、`style_coloured_walls_wrong`、`style_qa_failed`、`struct_incorrect`、"
             "`stale_overwrites` 等）必须为 0，代价看 CPU 时间、请求延迟、峰值内存和磁盘。dev / test 分开汇总；"
             "通过与否要用 `compare` 和基线比，这里只列数。", "",
             "| 测试集 | 运行 | 划分 | 建筑 成功/总数 | " + " | ".join(REVIEW_COLUMNS) + " |",
             "|" + "---|" * (4 + len(REVIEW_COLUMNS))]
    for r in runs:
        for label, slugs, agg in split_aggregates(r):
            ok = sum(1 for s in slugs if r["buildings"][s].get("metrics"))
            lines.append(f"| {r['testset']['id']} | {r['run_name']} | {label} | {ok}/{len(slugs)} | "
                         + " | ".join(_cell(agg.get(c)) for c in REVIEW_COLUMNS) + " |")
    return lines + [""]


def pose_leaderboard(results: list[dict[str, Any]]) -> list[str]:
    """The photo-pose section: every run that has photo_pose_* metrics, per building and pooled."""
    with_poses = [r for r in results if any((b.get("metrics") or {}).get("photo_pose_determined") is not None
                                            for b in r["buildings"].values())]
    if not with_poses:
        return []
    lines = ["## 照片位姿：每张照片的拍摄位置和朝向", "",
             f"确定 = 位置和朝向都有（已发布 SfM 模型的相机或 EXIF GPS + 罗盘）；被拒的上传、只有 GPS 没有罗盘"
             f"的照片都算失败。正确 = 和参考位姿相差不超过 {ev.POSE_RIGHT_M:g} 米且 {ev.POSE_RIGHT_DEG:g}°（严格："
             f"{ev.POSE_STRICT_M:g} 米 / {ev.POSE_STRICT_DEG:g}°），只在有参考位姿的照片上算。`*_testset` 的分母是测试集"
             "里该上传的全部照片：时间上限截掉、没上传的也算失败。有参考位姿的照片偏向 SfM 定位到的那些，所以正确率要和"
             "全部照片上的上下界一起看：下界 `pass_rate_low_testset` = 确定且对 / 全部（没参考的都算失败），上界"
             " `pass_rate_high_testset` = 确定且没被证明是错的 / 全部（没参考但确定了的算通过）。", ""]
    for r in with_poses:
        lines += [f"### {r['testset']['id']} · {r['run_name']}", "",
                  "| 建筑 | 状态 | " + " | ".join(POSE_COLUMNS) + " |", "|" + "---|" * (2 + len(POSE_COLUMNS))]
        for slug, b in r["buildings"].items():
            m = b.get("metrics") or {}
            lines.append(f"| {slug} | {b['status']} | " + " | ".join(_cell(m.get(c)) for c in POSE_COLUMNS) + " |")
        lines.append("| **合计** | | " + " | ".join(_cell(r["aggregate"].get(c)) for c in POSE_COLUMNS) + " |")
        lines.append("")
    return lines


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


# ---------------------------------------------------------------- posediff: the final shown poses of two runs


def repeat_dirs(result: dict[str, Any], slug: str) -> list[Path]:
    """A recorded result's run dirs of one building, one per repeat (<artefacts_dir>/<slug>/r1, r2, ... or the
    building dir itself)."""
    repeat = int(result["provenance"].get("repeat") or 1)
    return [Path(result["artefacts_dir"]) / slug / (f"r{k + 1}" if repeat > 1 else "") for k in range(repeat)]


def shown_poses(run_dir: Path) -> dict[str, np.ndarray]:
    """Photo key -> ENU camera centre the landmark page shows at the end of a recorded building run
    (crowd3d.api.camera_estimates of its store: one per photo, the unflagged cameras of published models)."""
    from crowd3d.api import camera_estimates

    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    _, _, models, _ = ev.load_run_store(ev.run_data_dir(run_dir, summary))
    key = {r["photo_id"]: r["key"] for r in summary["photos"] if r.get("photo_id")}
    return {key[pid]: np.asarray(cam["enu"], dtype=float) for pid, cam in camera_estimates(models).items()
            if pid in key and cam.get("enu") is not None}


def pose_difference(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, Any]:
    """How far B's shown camera centres lie from A's on the photos both show: as shown (both in georegistered ENU)
    and after the Sim3 of B onto A (eval_real_landmark.sim3_position_errors: refit once without gross outliers),
    which leaves the difference in shape. Median and 90th percentile in metres; None below 3 common photos."""
    common = sorted(set(a) & set(b))
    out: dict[str, Any] = {"photos_a": len(a), "photos_b": len(b), "common": len(common)}
    if len(common) < 3:
        return {**out, **dict.fromkeys(("shown_median_m", "shown_p90_m", "aligned_median_m", "aligned_p90_m",
                                        "scale"))}
    src, dst = np.array([b[k] for k in common]), np.array([a[k] for k in common])
    shown = np.linalg.norm(src - dst, axis=1)
    scale, aligned = ev.sim3_position_errors(src, dst)
    return {**out, "shown_median_m": _round(float(np.median(shown)), 3),
            "shown_p90_m": _round(float(np.percentile(shown, 90)), 3),
            "aligned_median_m": _round(float(np.median(aligned)), 3),
            "aligned_p90_m": _round(float(np.percentile(aligned, 90)), 3), "scale": _round(float(scale), 4)}


def posediff(a: dict[str, Any], b: dict[str, Any], slugs: list[str] | None = None) -> dict[str, Any]:
    """Per building both results ran: pose_difference of B's first run against A's, next to each result's own first
    against second repeat (the run-to-run spread of one config) when it has one."""
    out = {}
    for slug in slugs or [s for s in a["buildings"] if s in b["buildings"]]:
        if not ((a["buildings"].get(slug) or {}).get("metrics") and (b["buildings"].get(slug) or {}).get("metrics")):
            continue
        poses = {side: [shown_poses(d) for d in repeat_dirs(r, slug) if (d / "summary.json").is_file()][:2]
                 for side, r in (("a", a), ("b", b))}
        if not poses["a"] or not poses["b"]:
            out[slug] = {"status": "artefacts gone"}
            continue
        out[slug] = {"status": "ok", "a_vs_b": pose_difference(poses["a"][0], poses["b"][0]),
                     **{f"{side}_repeats": pose_difference(*p) if len(p) == 2 else None for side, p in poses.items()}}
    return out


def format_posediff(a: dict[str, Any], b: dict[str, Any], d: dict[str, Any]) -> str:
    def cells(x: dict[str, Any]) -> list[str]:
        if x["aligned_median_m"] is None:
            return [str(x["common"]), "-", "-", "-"]
        return [str(x["common"]), f"{x['shown_median_m']:g} / {x['shown_p90_m']:g}",
                f"{x['aligned_median_m']:g} / {x['aligned_p90_m']:g}", f"{x['scale']:g}"]

    lines = [f"posediff {a['testset']['id']}: A = {a['run_name']} vs B = {b['run_name']}: the final shown camera centres "
             "(metres, median / p90; aligned = after the Sim3 of B onto A)", "",
             "| building | pair | photos in both | shown | aligned | Sim3 scale |", "|---|---|---|---|---|---|"]
    for slug, r in d.items():
        if r["status"] != "ok":
            lines.append(f"| {slug} | - | {r['status']} | - | - | - |")
            continue
        for pair, x in (("A r1 vs B r1", r["a_vs_b"]), ("A r1 vs A r2", r["a_repeats"]),
                        ("B r1 vs B r2", r["b_repeats"])):
            if x is not None:
                lines.append("| " + " | ".join([slug, pair, *cells(x)]) + " |")
    return "\n".join(lines)


def cmd_posediff(args: argparse.Namespace) -> int:
    a, b = load_result(args.run_a), load_result(args.run_b)
    if a["testset"]["sha256"] != b["testset"]["sha256"]:
        print(f"refused: test sets differ: {a['testset']['id']} vs {b['testset']['id']}", file=sys.stderr)
        return 2
    d = posediff(a, b, args.buildings)
    print(format_posediff(a, b, d))
    if args.json:
        args.json.write_text(json.dumps(d, indent=1), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- calibration: the georegistration's sigmas


CALIB_TARGET = 0.954          # calibration --fit: the building-weighted 2-sigma coverage each fitted floor reaches
CALIB_FLOOR_OF = {"yaw": "yaw_deg", "scale": "log_scale", "xy": "xy_m", "tilt": "tilt_deg"}   # -> anchor.FLOORED
CALIB_FLOOR_STEP = {"yaw_deg": 0.01, "log_scale": 0.001, "xy_m": 0.01, "tilt_deg": 0.01}     # fitted floors round up
CALIB_COVER = (1.0, 2.0, 3.0)
assert set(CALIB_FLOOR_OF.values()) == set(FLOORED) == set(CALIB_FLOOR_STEP)


def calibration_rows(results: list[dict[str, Any]], split: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """building -> the calibration rows of its checked models (record["calibration"], eval_real_landmark.geo_scores)
    over the results and their repeats; a model another repeat already gave (the same "key": every camera in the same
    place, so the same anchor) counts once. split: only that split's buildings."""
    out: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        for slug, rec in r["buildings"].items():
            if split and split_of(r, slug) != split:
                continue
            rows = out.setdefault(slug, [])
            seen = {row["key"] for row in rows}
            for row in rec.get("calibration") or []:
                if row["key"] not in seen:
                    seen.add(row["key"])
                    rows.append(row)
    return {s: rows for s, rows in out.items() if rows}


def _component_sigma(c: dict[str, Any], comp: str, floors: dict[str, float] | None) -> float:
    """A row's sigma of one component: as scored, or its anchor core with floors in quadrature (legacy: as scored)."""
    if floors is None or c.get("core") is None:
        return float(c["sigma"])
    return math.hypot(float(c["core"]), floors[CALIB_FLOOR_OF[comp]])


def _error_size(e: Any) -> float:
    return float(np.hypot(*e)) if isinstance(e, list) else abs(float(e))


def calibration_table(rows: dict[str, list[dict[str, Any]]], floors: dict[str, float] | None = None
                      ) -> dict[str, Any]:
    """Per component (yaw, scale, xy, tilt): per building and building-weighted (each building one vote), the models
    checked, the error median (|e|; for the one-axis components also the signed median: the bias, model minus
    reference), the sigma median (sharpness), the median |z| (0.674 when right) and the shares within 1 / 2 / 3 sigma
    (0.683 / 0.954 / 0.997) and beyond 3 (eval_real_landmark.calib_z). floors: re-floor the anchors' cores instead of
    using the sigma each row was scored with."""
    table: dict[str, Any] = {}
    for comp in ev.CALIB_COMPONENTS:
        per: dict[str, dict[str, Any]] = {}
        for slug, rs in rows.items():
            cs = [row["comps"][comp] for row in rs if (row.get("comps") or {}).get(comp)]
            if not cs:
                continue
            sig = [_component_sigma(c, comp, floors) for c in cs]
            z = np.array([ev.calib_z(comp, c["e"], sg, c["ref"]) for c, sg in zip(cs, sig)], float)
            signed = [c["e"] for c in cs if not isinstance(c["e"], list)]
            per[slug] = {"n": len(cs), "error_median": round(float(np.median([_error_size(c["e"]) for c in cs])), 4),
                         "bias": round(float(np.median(signed)), 4) if signed else None,
                         "sigma_median": round(float(np.median(sig)), 4), "z_median": round(float(np.median(z)), 3),
                         **{f"cover{k:g}": round(float(np.mean(z <= k)), 3) for k in CALIB_COVER},
                         "beyond3": round(float(np.mean(z > 3.0)), 3), "z": z.tolist()}
        if not per:
            continue
        zs = np.array([v for p in per.values() for v in p["z"]])
        w = np.array([1.0 / p["n"] for p in per.values() for _ in p["z"]])
        order = np.argsort(zs)
        cum = np.cumsum(w[order])
        weighted = {"buildings": len(per), "n": int(len(zs)),
                    "z_median": round(float(zs[order][int(np.searchsorted(cum, 0.5 * cum[-1]))]), 3),
                    "sigma_median": round(float(np.median([p["sigma_median"] for p in per.values()])), 4),
                    **{k: round(float(np.mean([p[k] for p in per.values()])), 3)
                       for k in (*(f"cover{k:g}" for k in CALIB_COVER), "beyond3")}}
        pooled = {"n": int(len(zs)), "z_median": round(float(np.median(zs)), 3),
                  **{f"cover{k:g}": round(float(np.mean(zs <= k)), 3) for k in CALIB_COVER},
                  "beyond3": round(float(np.mean(zs > 3.0)), 3)}
        table[comp] = {"building_weighted": weighted, "pooled": pooled,
                       "buildings": {s: {k: v for k, v in p.items() if k != "z"} for s, p in per.items()}}
    return table


def fit_floors(rows: dict[str, list[dict[str, Any]]], target: float = CALIB_TARGET) -> dict[str, float | None]:
    """Per component the smallest floor (in quadrature with every anchor's core) whose building-weighted share within
    2 sigma reaches target, rounded up to CALIB_FLOOR_STEP. A model gets within 2 sigma once floor^2 >= (|e| / k)^2 -
    core^2 - ref^2 (k = 2; for xy and tilt the 2-D radius holding 0.954, 2.486), so those thresholds are the only
    candidates. None for a component no row has an anchor core for."""
    out: dict[str, float | None] = {}
    for comp, name in CALIB_FLOOR_OF.items():
        k = 2.0 if comp not in ev.CALIB_2D else math.sqrt(-2.0 * math.log(1.0 - math.erf(math.sqrt(2.0))))
        per = {s: cs for s, rs in rows.items()
               if (cs := [row["comps"][comp] for row in rs
                          if ((row.get("comps") or {}).get(comp) or {}).get("core") is not None])}
        if not per:
            out[name] = None
            continue
        cand = {0.0} | {math.sqrt(max(0.0, (_error_size(c["e"]) / k) ** 2 - c["core"] ** 2 - c["ref"] ** 2))
                        * (1 + 1e-9) for cs in per.values() for c in cs}

        def cover(f: float) -> float:
            return float(np.mean([np.mean([ev.calib_z(comp, c["e"], math.hypot(c["core"], f), c["ref"]) <= 2.0
                                           for c in cs]) for cs in per.values()]))

        best = next(f for f in sorted(cand) if cover(f) >= target)
        step = CALIB_FLOOR_STEP[name]
        out[name] = round(math.ceil(best / step - 1e-9) * step, 6)
    return out


def format_calibration(table: dict[str, Any], title: str) -> str:
    lines = [f"## {title}", "", "| component | building | n | |error| median | bias | sigma median | median |z| | "
             "within 1 sigma | within 2 | within 3 | beyond 3 |", "|---|---|---|---|---|---|---|---|---|---|---|"]

    def cell(v: Any) -> str:
        return "-" if v is None else f"{v:g}" if isinstance(v, float) else str(v)

    for comp, t in table.items():
        rows = [("building-weighted", t["building_weighted"]), ("pooled", t["pooled"]), *t["buildings"].items()]
        for name, r in rows:
            lines.append("| " + " | ".join([comp, name, cell(r.get("n")), cell(r.get("error_median")),
                                             cell(r.get("bias")), cell(r.get("sigma_median")), cell(r.get("z_median")),
                                             *(cell(r.get(f"cover{k:g}")) for k in CALIB_COVER),
                                             cell(r.get("beyond3"))]) + " |")
    lines += ["", "Ideal: median |z| 0.674; within 1 / 2 / 3 sigma 0.683 / 0.954 / 0.997; beyond 3 sigma 0.003. xy "
              "(east/north at the anchor's pivot) and tilt are 2-D: z is the normal deviate of the same tail."]
    return "\n".join(lines)


def cmd_calibration(args: argparse.Namespace) -> int:
    results = [load_result(p) for p in args.results]
    rows = calibration_rows(results, args.split)
    if not rows:
        raise SystemExit("no calibration rows in those results (record['calibration']: rescore them with this code)")
    floors = None
    if args.floors:
        floors = {k: float(v) for k, v in (item.split("=", 1) for item in args.floors.split(","))}
        if set(floors) != set(CALIB_FLOOR_STEP):
            raise SystemExit(f"--floors needs exactly {sorted(CALIB_FLOOR_STEP)}")
    doc: dict[str, Any] = {"results": [str(p) for p in args.results], "split": args.split,
                           "rows": {s: len(r) for s, r in rows.items()},
                           "as_scored": calibration_table(rows)}
    print(format_calibration(doc["as_scored"], "As scored"))
    if floors is not None:
        doc["floors"], doc["refloored"] = floors, calibration_table(rows, floors)
        print("\n" + format_calibration(doc["refloored"], f"Floors {floors}"))
    if args.fit:
        fitted = fit_floors(rows)
        doc["fitted_floors"] = fitted
        full = {k: (v if v is not None else 0.0) for k, v in fitted.items()}
        doc["fitted"] = calibration_table(rows, full)
        print(f"\nfitted floors (building-weighted within 2 sigma >= {CALIB_TARGET}): {fitted}")
        print("\n" + format_calibration(doc["fitted"], f"Fitted floors {full}"))
    if args.json:
        args.json.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return 0


# ---------------------------------------------------------------- rescore: metrics again from a recorded run


def build_reference(b: dict[str, Any], tid: str, ts_dir: Path, ts_sha: str, integ: dict[str, Any],
                    data_root: Path, gt_root: Path | None, cache_dir: Path) -> tuple[dict | None, dict | None, dict]:
    """The building's outline-corrected reference, its report (eval_real_landmark.building_reference, through the
    cache run uses too: reference_cache_spec; (None, report) without one) and the reference integrity a run's
    inputs_sha256 records (reference_integrity)."""
    gt, gt_integ = resolve_gt(b, data_root / tid / b["slug"], data_root, gt_root)
    spec = building_spec(b, ts_dir, integ, gt, False)
    spec["reference_cache"] = reference_cache_spec(cache_dir, ts_sha, b["slug"], gt_integ)
    ref_integ = reference_integrity(gt_integ, spec["reference_cache"])
    cb = ev.building_from_spec(spec)
    metas = {}
    for p in cb.photos:
        try:
            metas[p.key] = read_photo_meta(p.path.read_bytes())
        except Exception:  # noqa: BLE001  (an unreadable photo has no GPS for the reference fit)
            continue
    reference, report, _ = ev.building_reference(cb, metas, ev.LocalENU(cb.lat, cb.lon))
    return reference, report or {"message": "no reference available", "integrity": gt_integ}, ref_integ


UNMATCHED_INPUTS = "unmatched:"     # rescore: inputs_sha256 of artefacts whose inputs today's files do not reproduce


def source_recon_output_version(src: dict[str, Any]) -> int | None:
    """crowd3d.reconstruct.RECON_OUTPUT_VERSION of the artefacts a result scored (by commit for older results)."""
    prov = src.get("provenance") or {}
    if prov.get("recon_output_version") is not None:
        return prov["recon_output_version"]
    return LEGACY_RECON_OUTPUT_VERSION.get((prov.get("git") or {}).get("commit"))


def cmd_rescore(args: argparse.Namespace) -> int:
    ts, ts_path, ts_sha = load_testset(args.testset)
    tid = ts["testset_id"]
    src_path = Path(args.source)
    src = load_result(src_path)
    if src["testset"]["id"] != tid:
        raise SystemExit(f"{src_path} is a {src['testset']['id']} result, not {tid}")
    if src["testset"]["sha256"] != ts_sha:
        raise SystemExit(f"{src_path} ran on another version of {tid} (testset sha256 differs)")
    if (args.structure or args.snap_gate) and not args.restylize:
        raise SystemExit("--structure and --snap-gate choose how --restylize stylizes: pass --restylize")
    structure = (args.structure or "on") if args.restylize else None
    from crowd3d import stylize as st
    style_params = replace(st.DEFAULT_PARAMS, **st.SNAP_GATE_73[args.snap_gate]) if args.snap_gate else None
    snap_gate = {k: getattr(style_params or st.DEFAULT_PARAMS, k) for k in st.SNAP_GATE_73["on"]}
    recon = source_recon_output_version(src)
    if recon != RECON_OUTPUT_VERSION and not args.allow_recon_diff:
        raise SystemExit(f"{src_path} scored reconstruction output version {recon}, this code makes version "
                         f"{RECON_OUTPUT_VERSION} (crowd3d.reconstruct.RECON_OUTPUT_VERSION): run it again, or pass "
                         "--allow-recon-diff to score the old artefacts anyway")
    out_json = (args.results_dir or RESULTS_DIR) / tid / f"{args.name}.json"
    if out_json.exists() and not args.overwrite:
        raise SystemExit(f"{out_json} exists; pick another --name or pass --overwrite")
    ev.check_outside_repo(args.work_root, "--work-root")
    cache_dir = args.reference_cache or args.work_root / tid / "references"
    ev.check_outside_repo(cache_dir, "--reference-cache")
    run_root = Path(src["artefacts_dir"])
    restyle_root = args.work_root / tid / args.name
    split = load_split(ts, ts_sha, args.split)
    use_changed = src["testset"].get("on_changed") == "use"
    repeat = int(src["provenance"].get("repeat") or 1)
    started, t0 = utc_now(), time.monotonic()
    code_sha = code_fingerprint()
    # measured again from the artefacts; the rest (the pipeline's own outcome, wall and CPU time, ...) is copied
    rescored = {*ev.POSE_METRICS, *ev.GEO_METRICS, *ev.VERSION_METRICS, *ev.REQUEST_METRICS}
    keep = {name for name in ev.METRIC_BY_NAME if name not in rescored
            and not (args.restylize and name in ev.STYLE_METRICS)}
    buildings: dict[str, dict[str, Any]] = {}
    repeats: dict[str, list[dict[str, Any]]] = {}
    for b in select_buildings(ts, args.buildings):
        slug = b["slug"]
        srec = src["buildings"].get(slug)
        if srec and srec.get("metrics"):
            integ = verify_photos(b, args.data_root / tid / slug)
            reference, ref_report, ref_integ = build_reference(b, tid, ts_path.parent, ts_sha, integ, args.data_root,
                                                               args.gt_root, cache_dir)
            log.info("%s: reference %s verified cameras", slug, len(ev.reference_poses(reference)))
            # the photos the run would have uploaded without its time cap: the ones it never did fail too
            spec = building_spec(b, ts_path.parent, integ, None, (ts.get("integrity_policy", "skip")) == "use")
            cb = ev.building_from_spec(spec)
            planned = ev.planned_upload_keys(cb)
            reps = repeat_dirs(src, slug)
            runs, summaries, details, calib_rows, regeo = [], [], None, [], []
            for k, rep_dir in enumerate(reps):
                base = (srec.get("repeats") or [{}] * repeat)[k].get("metrics") if repeat > 1 else srec["metrics"]
                if not base or not (rep_dir / "summary.json").is_file():
                    continue
                style = None
                gr = ev.rescore_geo(rep_dir, cb, reference, regeo=args.regeo)
                if args.restylize:
                    out = restyle_root / slug / (rep_dir.name or "r1")
                    out.mkdir(parents=True, exist_ok=True)
                    style = ev.restylize_run(rep_dir, cb, (reference, ref_report), out / "style",
                                             structure=structure == "on", style_params=style_params,
                                             anchors=gr["anchors"])
                    (out / "style.json").write_text(json.dumps(style, indent=1, default=ev._json_default),
                                                    encoding="utf-8")
                pr = ev.rescore_poses(rep_dir, reference, planned, style=style)
                calib_rows += [{"run": rep_dir.name or "r1", **row} for row in gr["geo"]["calibration"]]
                if args.regeo:
                    regeo.append({"run": rep_dir.name or "r1", **gr["geo"]["regeo"]})
                if details is None:
                    summary = json.loads((rep_dir / "summary.json").read_text(encoding="utf-8"))
                    details = compact_details({**summary, "style": style if style is not None else summary.get("style"),
                                               "geo": gr["geo"]})
                    details.pop("regeo", None)          # every repeat's: record["regeo"]
                f = pr["funnel"]
                expected = f["listed"] if spec["upload_all"] else f["listed"] - f["no_gps_in_file"]
                if expected != len(planned):
                    log.warning("%s %s: the run planned %s uploads, the test set gives %s now", slug, rep_dir.name,
                                expected, len(planned))
                metrics = {name: base.get(name) for name in keep}
                metrics.update(pr["metrics"])
                metrics.update(gr["metrics"])
                if style is not None:
                    metrics.update(style["metrics"])
                runs.append(metrics)
                summaries.append({"run": rep_dir.name or "r1", "truncated": pr["truncated"],
                                  "testset_photos": len(planned), "run_planned_uploads": expected, **pr["summary"]})
                log.info("%s %s: determined %s/%s (test set %s), correct %s/%s", slug, rep_dir.name,
                         pr["metrics"]["photo_pose_determined"], pr["photos_uploaded"], len(planned),
                         pr["metrics"]["photo_pose_correct"], pr["metrics"]["photo_pose_gt"])
            record = json.loads(json.dumps(srec))
            # the inputs as a run of this code fingerprints them (OSM parts snapshot included); the source run's own
            # fingerprint is one of the two when the local photos, outline and reference are still what it used.
            # Otherwise the artefacts came from other inputs than today's: a marker no run's fingerprint equals, so
            # compare lists the building under "inputs differ"
            outline, _ = building_outline(b, ts_path.parent)
            outline_sha = sha256_file(outline) if outline and outline.is_file() else None
            parts_sha = osm_parts_sha(slug)
            fps = [inputs_fingerprint(integ, use_changed, outline_sha, ref_integ, sha) for sha in (None, parts_sha)]
            record["inputs_sha256"] = fps[1]
            record.pop("inputs_unmatched", None)
            if srec.get("inputs_sha256") not in fps:
                log.warning("%s: the local photos, outline or reference differ from what %s used", slug,
                            src["run_name"])
                record["inputs_unmatched"] = {"source": srec.get("inputs_sha256"), "local": fps[1]}
                record["inputs_sha256"] = UNMATCHED_INPUTS + canonical_sha256(record["inputs_unmatched"])
            record.setdefault("integrity", {})["osm_parts_sha256"] = parts_sha
            if details is not None:
                record["details"] = details
            record["calibration"] = calib_rows
            if args.regeo:
                record["regeo"] = regeo
                log.info("%s: regeo replayed %s, faithful %s of %s published models (largest ENU difference %s m)",
                         slug, sum(r["replayed"] for r in regeo), sum(r["faithful"] for r in regeo),
                         sum(r["published"] for r in regeo), max((r["max_diff_m"] or 0.0) for r in regeo) if regeo
                         else None)
            if not runs:
                record.update(status="skipped (the source run's artefacts are gone)", metrics=None)
            elif repeat > 1:
                record["metrics"] = median_metrics(runs)
                record["metrics_first"] = runs[0]
                for k, m in enumerate(runs):
                    if k < len(record.get("repeats") or []):
                        record["repeats"][k]["metrics"] = m
                if len(runs) >= 2:
                    repeats[slug] = runs
            else:
                record["metrics"] = runs[0]
            record["poses"] = {"reference": ref_report, "runs": summaries}
        else:
            record = {"status": "skipped (no run in the source result)", "metrics": None}
        buildings[slug] = record
    per_building_ok = {s: r["metrics"] for s, r in buildings.items() if r.get("metrics")}
    result = {
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "metric_schema_version": ev.METRIC_SCHEMA_VERSION,
        "run_name": args.name,
        "note": args.note,
        "testset": {**src["testset"], "split": split, "buildings_selected": list(buildings)},
        "config": src["config"],
        "provenance": {"git": git_state(), "code_sha256": code_sha, "versions": versions(), "started": started,
                       "wall_s": round(time.monotonic() - t0, 1), "harness_version": ev.HARNESS_VERSION,
                       "recon_output_version": recon, "repeat": repeat,
                       "stylize": src["provenance"].get("stylize"), "restylized": bool(args.restylize),
                       # --restylize: whether the stylize step went through crowd3d.structure_pipeline
                       "structure": structure if args.restylize else src["provenance"].get("structure"),
                       # --restylize: the task #73 snap gate flags it stylized with (stylize.SNAP_GATE_73)
                       **({"snap_gate": snap_gate} if args.restylize else {}),
                       # --regeo: the georegistration replayed on the kept SfM models; the floors it applied
                       "regeo": bool(args.regeo), "anchor_calibration": anchor_calibration(),
                       # the sp / sy rules the photo_pose_*uncertainty* metrics were measured with
                       "photo_pose_uncertainty_rule": uncertainty_rule(), "command": " ".join(sys.argv),
                       "rescored_from": {"run_name": src["run_name"], "path": _repo_relative(src_path),
                                         "sha256": sha256_file(src_path),
                                         "metric_schema_version": src["metric_schema_version"],
                                         "provenance": src["provenance"]}},
        "buildings": buildings,
        "aggregate": aggregate(per_building_ok),
        "noise": noise_from_repeats(repeats) if repeats else None,
        "artefacts_dir": str(run_root),
        **({"restyle_dir": str(restyle_root)} if args.restylize else {}),
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_json}")
    for slug, r in buildings.items():
        m = r.get("metrics") or {}
        print(f"  {slug}: {r['status']}; determined {m.get('photo_pose_determined')}/{m.get('photos_uploaded')}, "
              f"correct {m.get('photo_pose_correct')}/{m.get('photo_pose_gt')}, pass rate {m.get('photo_pose_pass_rate')}"
              f"; geo checked {m.get('geo_models_checked')}, camera error {m.get('geo_camera_error_median_m')} m, "
              f"supported {m.get('cov_local_supported_m_1p5')} m local / {m.get('cov_geo_supported_m_5')} m geo"
              + (f"; style passed {m.get('style_registration_passed')}, wrong {m.get('style_registration_wrong')}"
                 if args.restylize else ""))
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
    p.add_argument("--max-requests", nargs="+", default=None, metavar="N|SLUG=N",
                   help="upload only the first N requests of each building's arrival plan (SLUG=N: that building);"
                        " an earlier run's request prefix, so a faster pipeline sees the same uploads (in the run key)")
    p.add_argument("--no-stylize", action="store_true",
                   help="do not stylize the published models (style_* metrics stay empty; work dirs are pruned)")
    p.add_argument("--overwrite", action="store_true", help="replace an existing result of that name")
    p.add_argument("--resume", action="store_true",
                   help="reuse building runs in --work-root whose run_key.json matches (same code, config, test set,"
                        " inputs and options), e.g. after an interruption; the others run again")
    p.add_argument("--reference-cache", type=Path, default=None,
                   help="outline-corrected references per building (default <work-root>/<testset>/references)")
    p.add_argument("--split", default=None,
                   help="dev/test split recorded with the result: eval/splits id or path (default the test set's one)")

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
    p.add_argument("--criteria", default=None,
                   help="pass rules (eval/criteria id or path, e.g. review-v1): PASS / FAIL / INSUFFICIENT per rule and"
                        " split, and a verdict; exit 1 unless it is PASS")
    p.add_argument("--exclude-group", nargs="+", default=None, metavar="GROUP",
                   help="leave these metric groups out of the tables, the counts and the criteria (e.g. style when A"
                        " was stylized by older code)")

    p = sub.add_parser("rescore", parents=[common],
                       help="photo_pose_*, the schema-7 scores (and with --restylize style_*) from a recorded run's "
                            "artefacts (no new SfM) into a new result")
    p.add_argument("--testset", required=True)
    p.add_argument("--from", dest="source", required=True, help="the recorded result (eval/results/<testset>/<run>.json)")
    p.add_argument("--name", required=True, help="new run name: eval/results/<testset>/<name>.json")
    p.add_argument("--buildings", nargs="*", help="subset of building slugs (default all of the test set)")
    p.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT,
                   help=f"scratch for the reference cache and restyled models (default {DEFAULT_WORK_ROOT})")
    p.add_argument("--reference-cache", type=Path, default=None,
                   help="outline-corrected references per building (default <work-root>/<testset>/references)")
    p.add_argument("--restylize", action="store_true",
                   help="stylize the run's kept models again with the current code (style_* metrics; the stylized "
                        "models go under <work-root>/<testset>/<name>)")
    p.add_argument("--allow-recon-diff", action="store_true",
                   help="rescore artefacts of another RECON_OUTPUT_VERSION anyway")
    p.add_argument("--structure", choices=("on", "off"), default=None,
                   help="with --restylize: through crowd3d.structure_pipeline where the landmark has an OSM parts "
                        "snapshot (on, as a run stylizes; the default) or the footprint path (off)")
    p.add_argument("--snap-gate", choices=("on", "off"), default=None,
                   help="with --restylize: the task #73 snap gate flags (stylize.SNAP_GATE_73: the GeoAnchor's yaw "
                        "sigma, refusal of a clipped window or a missing sigma, the cameras' GPS median check) on or "
                        "off (default: stylize's defaults)")
    p.add_argument("--regeo", action="store_true",
                   help="replay every published model's georegistration on its kept SfM model (work/<job>/sparse/0)"
                        " with the product's code and the current GeoAnchor floors; calib_* takes those anchors (a"
                        " model the replay does not reproduce within 1 cm is left out), and so does --restylize with"
                        " the snap gate's snap_use_anchor")
    p.add_argument("--split", default=None,
                   help="dev/test split recorded with the result: eval/splits id or path (default the test set's one)")
    p.add_argument("--gt-root", type=Path, default=None,
                   help="reference models under <gt-root>/<slug>/<component>/sparse")
    p.add_argument("--note", default=None, help="free text stored with the result")
    p.add_argument("--overwrite", action="store_true", help="replace an existing result of that name")

    p = sub.add_parser("calibration", help="the georegistration's sigmas against the reference: coverage per "
                                           "component (record['calibration'] rows), optionally refloored or fitted")
    p.add_argument("results", type=Path, nargs="+", help="results (run or rescore) whose models to pool")
    p.add_argument("--split", default=None, help="only this split's buildings (dev, test)")
    p.add_argument("--floors", default=None, metavar="yaw_deg=..,log_scale=..,xy_m=..,tilt_deg=..",
                   help="put these floors in quadrature on the anchors' cores instead of the scored sigmas")
    p.add_argument("--fit", action="store_true",
                   help=f"fit each floor to a building-weighted 2-sigma coverage of {CALIB_TARGET} and show that table")
    p.add_argument("--json", type=Path, default=None, help="also write the tables (and fitted floors) as JSON")

    p = sub.add_parser("posediff", help="how far the final shown camera poses of two results differ, per building")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--buildings", nargs="*", help="subset of building slugs (default all both ran)")
    p.add_argument("--json", type=Path, default=None, help="also write the differences as JSON")

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
                "rescore": cmd_rescore, "posediff": cmd_posediff, "calibration": cmd_calibration,
                "freeze-synthetic": cmd_freeze_synthetic, "freeze-mdx": cmd_freeze_mdx}
    return handlers[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
