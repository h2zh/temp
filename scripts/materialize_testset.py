"""Frozen real-photo test sets: materialise them from testset.json (metadata in the repo, pixels never).

    # rebuild the photos of a released test set into a cache dir, verifying every sha256
    python scripts/materialize_testset.py commons-ten-v1 --cache-dir <dir outside the repo> [--buildings a b]
        [--gt] [--gt-dir <existing MegaDepth-X sparse root>] [--offline] [--refetch-pages]

    # building a new version (the maintainers' side; a released id is never rewritten):
    python scripts/materialize_testset.py collect --pool 225 --max-screen 600 -- <fetch_commons_landmark args>
    python scripts/materialize_testset.py freeze --testset-id <id> --buildings <slug> ... \
        --fetch-root <collect --out parent> --mdx-root <sparse models> --licence-parquet <MegaDepth-X> \
        --description "..." [--extra-json <top-level fields>]

A test set is eval/testsets/<testset_id>/testset.json; its "schema_notes" document every field. A released
test set is immutable: a different photo list, outline or GT reference is a new id (-v2). `freeze` refuses
to overwrite an existing testset.json.

Materialising downloads exactly the listed files with fetch_commons_landmark's PoliteClient (one request
at a time, >= 1 s apart, Retry-After honoured, every response cached in <cache-dir>/http-cache), rebuilds
each photo the way the collector built it and writes <cache-dir>/<testset_id>/integrity_report.json: per
photo "ok", "changed" (written anyway; says whether the thumbnail bytes or the EXIF block differ and
whether the pixels still look the same) or "missing". A benchmark can run on a partially changed set and
leave the changed photos out of comparisons.

How a photo is rebuilt (the code path of scripts/fetch_commons_landmark.py):
- pixels: the Commons thumbnail at the recorded standard width (or the original file when that was used);
- EXIF block: exif_source "original" -> the original's EXIF APP1 (Range read of its first 128 KiB);
  "commons_page_metadata" -> fetch_commons_landmark.exif_from_page over the MediaWiki EXIF table text
  stored in testset.json (no request; --refetch-pages re-reads the description pages instead);
- graft: patch_exif (Orientation 1 unless recorded otherwise, pixel sizes) + set_exif_segment.

Licensing: every photo keeps its Commons licence (recorded per photo). --cache-dir is refused inside the
repository (except its gitignored data/ folder); so are collect's --out/--cache-dir.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import logging
import os
import re
import shutil
import struct
import sys
import tarfile
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from PIL import Image

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import fetch_commons_landmark as fc  # noqa: E402

from crowd3d.coverage import sector_for_bearing, sector_label  # noqa: E402
from crowd3d.geo import local_bearing_distance  # noqa: E402
from crowd3d.validation import dhash_hex, hamming_hex  # noqa: E402

log = logging.getLogger("materialize_testset")

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTSETS_DIR = REPO_ROOT / "eval" / "testsets"
SCHEMA_VERSION = 1
NUM_SECTORS = 8
# Thumbnails re-rendered by Wikimedia (new encoder) change bytes, not content: a dHash this close means the
# same picture.
SAME_PIXELS_MAX_HAMMING = 4
# MediaWiki EXIF-table rows fetch_commons_landmark.exif_from_page reads: stored per photo so the EXIF
# block can be rebuilt without a request.
PAGE_EXIF_KEYS = ("exif-gpslatitude", "exif-gpslongitude", "exif-gpsaltitude", "exif-gpshpositioningerror",
                  "exif-gpsimgdirection", "exif-gpsimgdirectionref", "exif-focallength",
                  "exif-focallengthin35mmfilm", "exif-datetimeoriginal", "exif-make", "exif-model")
SPARSE_FILES = ("cameras.bin", "images.bin", "points3D.bin", "frames.bin", "rigs.bin")
MEGADEPTH_X_URL = "https://huggingface.co/datasets/y-u-a-n-l-i/MegaDepth-X/resolve/main/release/{scene}.tar.gz"
STATUS_OK, STATUS_CHANGED, STATUS_MISSING = "ok", "changed", "missing"

# Off-target = taken from the building or inside it (title or the file's own Commons categories). Same rule
# as the curator's MegaDepth-X classification (views from / interior unless labelled exterior).
VIEW_FROM_RE = re.compile(r"observation|deck|glass.floor|edgewalk|skywalk|from.the.(?:top|tower|dome|cupola|roof)|"
                          r"views?.from|panorama.from|aerial|on.top|rooftop", re.IGNORECASE)
INTERIOR_RE = re.compile(r"interior|inside|innen|intérieur|interno|\bnave\b|altar|ceiling|vault|organ|crypt|mosaic|"
                         r"stained.glass|treasury|pulpit|fresco", re.IGNORECASE)
EXTERIOR_RE = re.compile(r"exterior|outdoor|fa[cç]ade", re.IGNORECASE)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_dhash(data: bytes) -> str:
    with Image.open(io.BytesIO(data)) as im:
        return dhash_hex(im)


# ---------------------------------------------------------------- test set files


def testset_path(ref: str | Path) -> Path:
    """'commons-ten-v1' (a registered id), a testset.json path or its directory -> the testset.json path."""
    p = Path(ref)
    if p.is_file():
        return p
    if p.is_dir() and (p / "testset.json").is_file():
        return p / "testset.json"
    q = TESTSETS_DIR / str(ref) / "testset.json"
    if q.is_file():
        return q
    raise FileNotFoundError(f"no test set {ref!r} (looked for {p} and {q})")


def load_testset(ref: str | Path) -> tuple[dict[str, Any], Path, str]:
    """(test set, its testset.json path, sha256 of that file's bytes)."""
    path = testset_path(ref)
    raw = path.read_bytes()
    ts = json.loads(raw)
    if ts.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {ts.get('schema_version')!r}, this tool reads {SCHEMA_VERSION}")
    return ts, path, sha256_bytes(raw)


def outline_path(ts_path: Path, building: dict[str, Any]) -> Path:
    """The committed OSM outline of a building (paths in testset.json are relative to its directory)."""
    return ts_path.parent / building["osm_outline"]["file"]


def photos_dir(cache_dir: Path, testset_id: str, slug: str) -> Path:
    return cache_dir / testset_id / slug


# ---------------------------------------------------------------- MegaDepth-X reference models


def colmap_image_names(images_bin: Path) -> list[str]:
    """Image names in a COLMAP images.bin, read without loading the 2-D points (1.7 GB for big scenes)."""
    out = []
    with images_bin.open("rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            f.read(4 + 8 * 7 + 4)                 # image_id, qvec, tvec, camera_id
            name = bytearray()
            while (c := f.read(1)) not in (b"\0", b""):
                name += c
            (npts,) = struct.unpack("<Q", f.read(8))
            f.seek(npts * 24, os.SEEK_CUR)        # x, y (double) + point3D_id (int64)
            out.append(name.decode("utf-8"))
    return out


def sparse_manifest(sparse_dir: Path) -> dict[str, dict[str, Any]]:
    return {name: {"sha256": sha256_file(sparse_dir / name), "bytes": (sparse_dir / name).stat().st_size}
            for name in SPARSE_FILES if (sparse_dir / name).exists()}


def verify_sparse(sparse_dir: Path, expected: dict[str, dict[str, Any]]) -> dict[str, Any]:
    rows = {}
    for name, want in expected.items():
        p = sparse_dir / name
        if not p.exists():
            rows[name] = STATUS_MISSING
        elif p.stat().st_size != want["bytes"] or sha256_file(p) != want["sha256"]:
            rows[name] = STATUS_CHANGED
        else:
            rows[name] = STATUS_OK
    status = (STATUS_OK if all(v == STATUS_OK for v in rows.values()) else
              STATUS_MISSING if any(v == STATUS_MISSING for v in rows.values()) else STATUS_CHANGED)
    return {"status": status, "sparse_dir": str(sparse_dir), "files": rows}


def stream_extract_sparse(url: str, scene: str, components: list[str], dest: Path) -> list[Path]:
    """Stream a MegaDepth-X archive and keep only <scene>/<component>/sparse/* (the rest is never stored)."""
    want = {c: dest / c / "sparse" for c in components}
    written = []
    req = urllib.request.Request(url, headers={"User-Agent": fc.USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp, tarfile.open(fileobj=resp, mode="r|gz") as tar:
        for member in tar:
            parts = Path(member.name).parts      # "./<component>/sparse/<file>" (or "<scene>/<component>/...")
            if (len(parts) < 3 or parts[-2] != "sparse" or parts[-3] not in want or parts[-1] not in SPARSE_FILES
                    or not member.isfile()):
                continue
            target = want[parts[-3]] / parts[-1]
            target.parent.mkdir(parents=True, exist_ok=True)
            src = tar.extractfile(member)
            if src is None:
                continue
            with target.open("wb") as f:
                shutil.copyfileobj(src, f, 1 << 20)
            written.append(target)
    return written


def materialize_gt(building: dict[str, Any], cache_dir: Path, gt_dir: Path | None, offline: bool) -> dict[str, Any]:
    """Verify (and when absent, download) a building's reference sparse models."""
    ref = building.get("gt_reference")
    if not ref:
        return {"status": "none"}
    comps = [ref, *ref.get("additional_components", [])]
    root = (gt_dir / building["slug"]) if gt_dir is not None else (cache_dir / "mdx" / building["slug"])
    expected = {c["component"]: c["sparse_files"] for c in comps}
    result: dict[str, Any] = {"components": {k: verify_sparse(root / k / "sparse", v) for k, v in expected.items()}}
    missing = [k for k, v in result["components"].items() if v["status"] != STATUS_OK]
    if missing and gt_dir is None and not offline:
        log.info("%s: streaming %s for components %s", building["slug"], ref["archive_url"], missing)
        t0 = time.monotonic()
        stream_extract_sparse(ref["archive_url"], ref["scene"], missing, root)
        result["download_s"] = round(time.monotonic() - t0, 1)
        for k in missing:
            result["components"][k] = verify_sparse(root / k / "sparse", expected[k])
    states = [v["status"] for v in result["components"].values()]
    result["status"] = (STATUS_OK if all(s == STATUS_OK for s in states) else
                        STATUS_MISSING if STATUS_MISSING in states else STATUS_CHANGED)
    return result


# ---------------------------------------------------------------- rebuilding one photo


def exif_block(client: fc.PoliteClient, photo: dict[str, Any], refetch_pages: bool) -> bytes:
    """The EXIF APP1 payload before grafting, rebuilt as the collector built it."""
    if photo["exif_source"] == "commons_page_metadata":
        if refetch_pages:
            resp = client.get(photo["page_url"], gzip_ok=True)
            if not resp.ok:
                raise fc.Skip("page_missing", f"HTTP {resp.status}")
            page = fc.parse_file_page(resp.text())
        else:
            page = fc.FilePage(metadata=dict(photo["exif_page_table"]))
        app1 = fc.exif_from_page(page)
        if app1 is None:
            raise fc.Skip("page_gps_unparsed")
        return app1
    if photo["exif_source"] == "original":
        kind = fc.file_kind(photo["title"]) or "jpeg"
        scan, _ = fc.read_original_exif(client, photo["title"], kind)
        if scan.app1 is None:
            raise fc.Skip("no_exif")
        return scan.app1
    raise ValueError(f"unknown exif_source {photo['exif_source']!r}")


def rebuild_photo(client: fc.PoliteClient, photo: dict[str, Any], refetch_pages: bool = False) -> tuple[bytes, dict[str, Any]]:
    """(materialised bytes, check details). Raises fc.Skip when a needed download is refused or absent."""
    resp = client.get(photo["pixels_url"])
    if not resp.ok:
        raise fc.Skip("pixels_unavailable", f"HTTP {resp.status}")
    raw = resp.body
    check: dict[str, Any] = {"thumbnail_match": sha256_bytes(raw) == photo["pixels_sha256"]}
    if photo["thumbnail_width"] is None:          # the original file itself, EXIF and all
        data = raw
        scan = fc.scan_jpeg(raw) if raw[:2] == b"\xff\xd8" else fc.scan_png(raw)
        check["exif_match"] = scan.app1 is not None and sha256_bytes(scan.app1) == photo["exif_sha256"]
    else:
        if raw[:2] != b"\xff\xd8":
            raise fc.Skip("thumbnail_not_jpeg")
        app1 = exif_block(client, photo, refetch_pages)
        check["exif_match"] = sha256_bytes(app1) == photo["exif_sha256"]
        with Image.open(io.BytesIO(raw)) as im:
            w, h = im.size
        patched = fc.patch_exif(app1, width=w, height=h, orientation=1 if photo.get("orientation_reset", True) else None)
        data = fc.set_exif_segment(raw, patched)
    return data, check


def materialize(ref: str | Path, cache_dir: Path, *, buildings: Iterable[str] | None = None, offline: bool = False,
                refetch_pages: bool = False, gt: bool = False, gt_dir: Path | None = None,
                client: fc.PoliteClient | None = None, report_path: Path | None = None) -> dict[str, Any]:
    """Materialise (a subset of) a test set into cache_dir and return + write the integrity report.

    Photos already present with the recorded sha256 are not downloaded again. A photo whose rebuilt bytes
    differ is written anyway and reported as "changed"; one that cannot be downloaded is "missing". After
    a rate-limit stop no further requests are made and the remaining photos are "missing" (not_attempted).
    """
    ts, ts_path, ts_sha = load_testset(ref)
    tid = ts["testset_id"]
    fc.check_outside_repo(cache_dir, "--cache-dir")
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = client or fc.PoliteClient(cache_dir / "http-cache", offline=offline)
    wanted = set(buildings) if buildings else None
    unknown = (wanted or set()) - {b["slug"] for b in ts["buildings"]}
    if unknown:
        raise SystemExit(f"{tid} has no building(s) {sorted(unknown)}")
    t0 = time.monotonic()
    stopped: str | None = None
    report: dict[str, Any] = {
        "testset_id": tid, "testset_sha256": ts_sha, "testset_path": str(ts_path), "schema_version": SCHEMA_VERSION,
        "generated": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"), "cache_dir": str(cache_dir),
        "offline": offline, "refetch_pages": refetch_pages, "buildings": {},
    }
    for b in ts["buildings"]:
        if wanted is not None and b["slug"] not in wanted:
            continue
        bdir = photos_dir(cache_dir, tid, b["slug"])
        (bdir / "photos").mkdir(parents=True, exist_ok=True)
        rows = []
        for photo in b["photos"]:
            target = bdir / photo["file"]
            row: dict[str, Any] = {"key": photo.get("key") or photo["file_name"], "title": photo["title"],
                                   "file": str(target), "sha256_expected": photo["sha256"]}
            if target.exists() and sha256_file(target) == photo["sha256"]:
                rows.append({**row, "status": STATUS_OK, "sha256": photo["sha256"], "from": "materialised_before"})
                continue
            if stopped:
                rows.append({**row, "status": STATUS_MISSING, "detail": "not_attempted: " + stopped})
                continue
            try:
                data, check = rebuild_photo(client, photo, refetch_pages)
            except fc.RateLimited as exc:
                stopped = str(exc)
                rows.append({**row, "status": STATUS_MISSING, "detail": "rate_limited: " + stopped})
                continue
            except fc.Skip as sk:
                detail = sk.reason + (f" ({sk.detail})" if sk.detail else "")
                if offline:
                    detail += " [offline]"
                rows.append({**row, "status": STATUS_MISSING, "detail": detail})
                continue
            except (ValueError, OSError, struct.error) as exc:     # undecodable thumbnail or EXIF block
                rows.append({**row, "status": STATUS_MISSING, "detail": f"unbuildable: {exc}"})
                continue
            target.write_bytes(data)
            got = sha256_bytes(data)
            row.update(check, sha256=got)
            if got == photo["sha256"]:
                row["status"] = STATUS_OK
            else:
                row["status"] = STATUS_CHANGED
                try:
                    row["pixel_dhash_distance"] = hamming_hex(pixel_dhash(data), photo["dhash"])
                    row["pixels_equivalent"] = row["pixel_dhash_distance"] <= SAME_PIXELS_MAX_HAMMING
                except (OSError, ValueError) as exc:
                    row["pixels_equivalent"] = False
                    row["detail"] = f"undecodable: {exc}"
            rows.append(row)
        counts = Counter(r["status"] for r in rows)
        entry: dict[str, Any] = {"photos_dir": str(bdir), "photos": len(rows), **{s: counts.get(s, 0) for s in
                                 (STATUS_OK, STATUS_CHANGED, STATUS_MISSING)}, "rows": rows}
        if gt:
            entry["gt"] = materialize_gt(b, cache_dir, gt_dir, offline)
        report["buildings"][b["slug"]] = entry
        log.info("%s: %d ok, %d changed, %d missing", b["slug"], entry[STATUS_OK], entry[STATUS_CHANGED],
                 entry[STATUS_MISSING])
    totals = Counter()
    for e in report["buildings"].values():
        for s in ("photos", STATUS_OK, STATUS_CHANGED, STATUS_MISSING):
            totals[s] += e[s]
    report["totals"] = dict(totals)
    report["complete"] = totals["photos"] == totals[STATUS_OK]
    report["stopped"] = stopped
    http = client.report()
    http["retry_events"] = http["retry_events"][-50:]
    report["http"] = http
    report["elapsed_s"] = round(time.monotonic() - t0, 1)
    out = report_path or (cache_dir / tid / "integrity_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    report["report_path"] = str(out)
    return report


def usable_photos(report: dict[str, Any], slug: str, include_changed: bool = True) -> list[Path]:
    """Materialised photo paths of one building: ok ones, plus changed ones unless include_changed=False."""
    keep = {STATUS_OK, STATUS_CHANGED} if include_changed else {STATUS_OK}
    return [Path(r["file"]) for r in report["buildings"][slug]["rows"] if r["status"] in keep]


# ---------------------------------------------------------------- selection rule (collect)


def views_from_re(names: Iterable[str]) -> re.Pattern[str] | None:
    """'... from (the) <landmark>' (e.g. 'Washington Monument from the Lincoln Memorial'): taken AT the
    landmark, looking away. names: the landmark's Commons categories (with and without the parentheses)."""
    variants = {v for n in names for v in (n, re.sub(r"\s*\((.*?)\)", r" \1", n).strip())}
    if not variants:
        return None
    alt = "|".join(re.escape(v).replace(r"\ ", r"[\s_]+") for v in sorted(variants, key=len, reverse=True))
    return re.compile(rf"\bfrom[\s_]+(?:the[\s_]+)?(?:{alt})", re.IGNORECASE)


def is_off_target(title: str, categories: Iterable[str], views_from: re.Pattern[str] | None = None) -> bool:
    """True for photos taken from or inside the building, judged by the title and the file's categories."""
    text = " ".join([title, *categories])
    return bool(VIEW_FROM_RE.search(text) or (views_from is not None and views_from.search(text))
                or (INTERIOR_RE.search(text) and not EXTERIOR_RE.search(text)))


def screening_order(candidates: list[fc.Candidate]) -> list[fc.Candidate]:
    """Files named by several kinds of source first (as fetch_commons_landmark ranks them), then the
    single-source files round-robin over source kinds (titles, search, category), so a capped screen
    samples every source instead of exhausting the first one."""
    multi = [c for c in candidates if len(c.source_kinds) > 1]
    groups: dict[str, list[fc.Candidate]] = defaultdict(list)
    for c in candidates:
        if len(c.source_kinds) == 1:
            groups[next(iter(c.source_kinds))].append(c)
    kinds = [k for k in ("titles", "search", "category") if k in groups] + sorted(
        k for k in groups if k not in ("titles", "search", "category"))
    out = list(multi)
    for i in range(max((len(g) for g in groups.values()), default=0)):
        out.extend(groups[k][i] for k in kinds if i < len(groups[k]))
    return out


def diversity_order(rows: list[dict[str, Any]]) -> list[str]:
    """Titles in the order phase 2 should visit them when only N photos are kept.

    rows: {"title", "sector" (int or None), "distance_m", "author_key"}. Rule: within a compass sector,
    round-robin over authors (each author's photos closest first); then round-robin over sectors. So
    every sector contributes until it runs out, no single author dominates a sector, and nearer photos
    (more likely to show the building large) win ties. Rows without GPS come last. Deterministic.
    """
    by_sector: dict[int, list[dict[str, Any]]] = defaultdict(list)
    tail = []
    for r in rows:
        (by_sector[r["sector"]] if r.get("sector") is not None else tail).append(r)
    ranked: list[tuple[int, int, float, str]] = []
    for sector, items in by_sector.items():
        by_author: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in items:
            by_author[r.get("author_key") or r["title"]].append(r)
        in_sector = []
        for photos in by_author.values():
            photos.sort(key=lambda r: (r["distance_m"], r["title"]))
            in_sector.extend((k, r["distance_m"], r["title"]) for k, r in enumerate(photos))
        in_sector.sort()
        ranked.extend((k, sector, d, t) for k, (_, d, t) in enumerate(in_sector))
    ranked.sort()
    return [t for _, _, _, t in ranked] + sorted(r["title"] for r in tail)


def author_key(page: fc.FilePage) -> str | None:
    """The uploader/author identity used for diversity: the author's user page when linked, else the name."""
    if page.author_url:
        return page.author_url.split("?")[0].rstrip("/")
    return page.author.strip().lower() if page.author else None


def collect(argv: list[str], *, pool: int, max_screen: int, drop_off_target: bool = True,
            client: fc.PoliteClient | None = None) -> int:
    """Run fetch_commons_landmark with the test-set selection rule.

    Before the fetcher's own run, candidate pages are screened in screening_order until `pool` files pass
    the page screen (EXIF GPS inside the radius, not off-target) or `max_screen` were looked at; the passing
    files are handed to the fetcher in diversity_order, so its --max-photos cap keeps the most diverse ones
    (a dropped file is replaced by the next in that order). Off-target files are left out. All of it is
    recorded in screening_stats.json under "listing" -> "selection".
    """
    original_gather = fc.gather

    def gather(client: fc.PoliteClient, args: argparse.Namespace) -> tuple[list[fc.Candidate], dict[str, Any]]:
        candidates, report = original_gather(client, args)
        cfg = fc.Config(name=args.name, lat=args.lat, lon=args.lon, radius_m=args.radius,
                        keep_radius_m=args.keep_radius if args.keep_radius is not None else args.radius,
                        out=args.out)
        views_from = views_from_re(args.category)
        screened: list[fc.Candidate] = []
        passed: list[dict[str, Any]] = []
        off_target: list[str] = []
        skips: Counter[str] = Counter()
        for cand in screening_order(candidates):
            if len(screened) >= max_screen or len(passed) >= pool:
                break
            screened.append(cand)
            try:
                _, page = fc.screen_page(client, cand, cfg)
            except fc.Skip as sk:
                skips[sk.reason] += 1
                continue
            if drop_off_target and is_off_target(cand.title, page.categories, views_from):
                off_target.append(cand.title)
                continue
            gps = fc.page_gps(page)
            row: dict[str, Any] = {"title": cand.title, "sector": None, "distance_m": None,
                                   "author_key": author_key(page)}
            if gps is not None:
                bearing, dist = local_bearing_distance(args.lat, args.lon, *gps)
                row.update(sector=sector_for_bearing(bearing, NUM_SECTORS), distance_m=round(dist, 2))
            passed.append(row)
        order = diversity_order(passed)
        by_title = {c.title: c for c in screened}
        chosen = set(order)
        dropped = set(off_target)
        final = [by_title[t] for t in order] + [c for c in screened if c.title not in chosen | dropped]
        report["selection"] = {
            "rule": "screening_order + diversity_order (scripts/materialize_testset.py)",
            "pool": pool, "max_screen": max_screen, "listed": len(candidates), "screened": len(screened),
            "screened_by_source_kinds": dict(Counter("+".join(sorted(c.source_kinds)) for c in screened)),
            "page_screen_skips": dict(skips.most_common()), "page_passed": len(passed) + len(off_target),
            "off_target_dropped": off_target, "views_from_pattern": views_from.pattern if views_from else None,
            "visit_order": order,
        }
        return final, report

    fc.gather = gather
    try:
        return fc.main(argv, client=client)
    finally:
        fc.gather = original_gather


# ---------------------------------------------------------------- freezing a new test set (maintainers)

SCHEMA_NOTES: dict[str, Any] = {
    "paths": "osm_outline.file is relative to the directory of testset.json. photos[].file is relative to "
             "<cache-dir>/<testset_id>/<slug>/ after `python scripts/materialize_testset.py <testset_id> "
             "--cache-dir <cache-dir>`; the integrity report is <cache-dir>/<testset_id>/integrity_report.json.",
    "sha256": "photos[].sha256 is the materialised file (thumbnail with the EXIF block grafted in); "
              "pixels_sha256 the downloaded thumbnail (or original) bytes; exif_sha256 the EXIF APP1 payload "
              "(starting with Exif\\0\\0) before the graft patched Orientation and pixel sizes. With "
              "exif_source 'original' that is the original's own EXIF block; with 'commons_page_metadata' "
              "it is fetch_commons_landmark.exif_from_page(exif_page_table). testset sha256 = sha256 of the "
              "testset.json bytes; outline and sparse files carry their own sha256.",
    "exif": "photos[].exif is what crowd3d.exif.read_photo_meta reads from the materialised file (lat/lon "
            "rounded to 1e-7 deg, others to 1e-3); null = absent. heading_ref 'T' true / 'M' magnetic north.",
    "exif_page_table": "Extra field: the MediaWiki EXIF table rows (text as rendered) the EXIF block is "
                       "rebuilt from when exif_source is 'commons_page_metadata' (originals were not "
                       "reachable when the set was collected). Absent for exif_source 'original'.",
    "landmark_to_camera": "Extra field: bearing/distance from the building centre to the EXIF GPS, the "
                          "8-sector index/label it falls in (crowd3d.coverage) and heading_offset_deg "
                          "(0 = camera looks at the centre) when a heading exists.",
    "gt_reference": "Optional MegaDepth-X reference model: scene, component (the one the curator chose), "
                    "archive_url, sparse_files {name: {sha256, bytes}}, and images = the image-name mapping "
                    "{photo key: {name, name_in_megadepth_x, orientation}}. name is the image's exact name "
                    "inside the sparse model's images.bin: name_in_megadepth_x (license.parquet) after "
                    "'<scene>/<component>/images/' - '000123.jpg' in some scenes but "
                    "'commons/<subcat>/0/pictures/<file>' in others, so never reduce it to a basename. "
                    "orientation = EXIF Orientation of the MegaDepth-X image the model was built from (its "
                    "camera frame is the stored pixel grid; the materialised photos are upright), null when "
                    "absent or invalid. Only photos registered in the model are mapped. "
                    "additional_components (extra field): other components of the same scene, same layout "
                    "(Brandenburg Gate west face); their photos carry gt_additional instead of gt_image.",
    "key": "photos[].key = the Commons file name (unique per building); what results and integrity rows use.",
    "gt_image": "Extra per-photo fields: gt_image / gt_orientation / name_in_megadepth_x repeat the "
                "gt_reference.images entry of the primary component; gt_additional {component, name, "
                "orientation, name_in_megadepth_x} marks a photo registered in an additional component.",
    "dhash": "Extra field: 64-bit dHash (crowd3d.validation.dhash_hex) of the thumbnail pixels; a "
             "re-rendered thumbnail within 4 bits counts as the same picture in the integrity report.",
    "validation_at_freeze": "Extra field: crowd3d.validation.validate_photo verdict at freeze time (code at "
                            "the commit in 'created_with'). Informational: benchmarks re-run validation.",
    "stats": "Extra field per building: the collection funnel (listed, screened, EXIF/GPS counts) and "
             "summaries of the frozen photos (heading, focal, authors, sectors, GT overlap).",
    "selection": "Extra top-level field: how candidates were screened and which photos were kept.",
}


def _mdx_rows(parquet: Path, scenes: set[str]) -> dict[str, list[dict[str, str]]]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    t = pq.read_table(parquet, columns=["name_in_megadepth-x", "cat", "image_name", "license_short_name"])
    t = t.filter(pc.is_in(t["cat"], value_set=__import__("pyarrow").array(sorted(scenes))))
    out: dict[str, list[dict[str, str]]] = defaultdict(list)
    for r in t.to_pylist():
        out[r["cat"]].append(r)
    return out


def gt_reference(lm: dict[str, Any], local_root: Path, rows: list[dict[str, str]],
                 photos: dict[str, str]) -> dict[str, Any] | None:
    """MegaDepth-X reference for one building: sparse sha256s and photo key -> GT image (registered only).

    photos: title -> photo key. The image name inside the sparse model is name_in_megadepth-x after
    "<scene>/<component>/images/" ("000123.jpg" in some scenes, "commons/<subcat>/0/pictures/<file>" in
    others). Its EXIF Orientation comes from <local_root>/orientation_index.json when present.
    """
    mdx = lm.get("megadepth_x")
    if not mdx or not local_root.is_dir():
        return None
    scene, primary = mdx["scene"], str(mdx["component"])
    comps = sorted((p.parent.parent.name for p in local_root.glob("*/sparse/images.bin")),
                   key=lambda c: (c != primary, c))
    if primary not in comps:
        return None
    index_path = local_root / "orientation_index.json"
    orient = json.loads(index_path.read_text(encoding="utf-8"))["components"] if index_path.exists() else {}
    by_title: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        by_title[fc.normalize_title("File:" + r["image_name"])].append(r["name_in_megadepth-x"])
    blocks = []
    mapped: set[str] = set()
    for comp in comps:
        sparse = local_root / comp / "sparse"
        registered = set(colmap_image_names(sparse / "images.bin"))
        images: dict[str, dict[str, Any]] = {}
        for title in sorted(photos):
            for name in by_title.get(title, []):
                head, _, image = name.partition("/images/")
                if head.split("/")[-1] == comp and image in registered and title not in mapped:
                    o = (orient.get(comp) or {}).get(image)
                    images[photos[title]] = {"name": image, "name_in_megadepth_x": name,
                                             "orientation": o if o in range(1, 9) else None}
                    mapped.add(title)
        blocks.append({"component": comp, "registered_images": len(registered),
                       "sparse_files": sparse_manifest(sparse), "images": dict(sorted(images.items())),
                       "orientation_source": "orientation_index.json" if comp in orient else None})
    return {"dataset": "MegaDepth-X", "scene": scene, **blocks[0],
            "archive_url": MEGADEPTH_X_URL.format(scene=urllib.parse.quote(scene)), "archive_mb": mdx.get("archive_mb"),
            "licence": "CC BY 4.0 (reference models); photos keep their Commons licences",
            "local_dir_convention": "<gt-dir>/<slug>/<component>/sparse",
            "additional_components": blocks[1:], "photos_with_gt": len(mapped)}


def gt_fields(gt: dict[str, Any] | None, key: str) -> dict[str, Any]:
    """Per-photo GT fields: gt_image/gt_orientation for the primary component, gt_additional otherwise."""
    out: dict[str, Any] = {"gt_image": None, "gt_orientation": None, "name_in_megadepth_x": None}
    if not gt:
        return out
    if key in gt["images"]:
        e = gt["images"][key]
        return {"gt_image": e["name"], "gt_orientation": e["orientation"], "name_in_megadepth_x": e["name_in_megadepth_x"]}
    for block in gt.get("additional_components", []):
        if key in block["images"]:
            return {**out, "gt_additional": {"component": block["component"], **block["images"][key]}}
    return out


def freeze_photo(client: fc.PoliteClient, m: dict[str, Any], gt: dict[str, Any] | None) -> dict[str, Any]:
    """One testset.json photo entry from a fetcher manifest entry, re-derived from the HTTP cache and
    checked by rebuilding the materialised bytes."""
    pix = m["pixels"]
    title = m["title"]
    resp = client.get(fc.page_url(title), gzip_ok=True)
    if not resp.ok:
        raise SystemExit(f"{title}: description page not in the cache")
    page = fc.parse_file_page(resp.text())
    table = None
    if m["exif_source"] == "commons_page_metadata":
        table = {k: page.metadata[k] for k in PAGE_EXIF_KEYS if k in page.metadata}
        app1 = fc.exif_from_page(fc.FilePage(metadata=dict(table)))
    else:
        scan, _ = fc.read_original_exif(client, title, fc.file_kind(title) or "jpeg")
        app1 = scan.app1
    if app1 is None:
        raise SystemExit(f"{title}: EXIF block cannot be rebuilt")
    raw = client.get(pix["url"])
    if not raw.ok:
        raise SystemExit(f"{title}: pixels not in the cache ({pix['url']})")
    src = pix["source"]
    entry = {
        "key": m["file_name"], "title": title, "file_name": m["file_name"], "page_url": m["page_url"], "original_url": m["original_url"],
        "thumbnail_width": int(src.split("_", 1)[1]) if src.startswith("thumbnail_") else None,
        "pixels_url": pix["url"], "pixels_sha256": sha256_bytes(raw.body),
        "orientation_reset": bool(pix.get("orientation_reset", True)),
        "file": pix["file"], "sha256": pix["sha256"], "bytes": pix["bytes"], "width": pix["width"],
        "height": pix["height"], "dhash": pixel_dhash(raw.body),
        "exif_source": m["exif_source"], "exif_sha256": sha256_bytes(app1),
        **({"exif_page_table": table} if table is not None else {}),
        "exif": {k: m["exif"][k] for k in fc.SURVIVING_FIELDS},
        "original": {k: m["original"].get(k) for k in ("width", "height", "mime", "bytes")},
        "author": m["author"], "author_url": m["author_url"], "credit": m["credit"], "licence": m["licence"],
        "licence_url": next((lic.get("link") or lic.get("url") for lic in m["licences"]
                             if lic.get("link") or lic.get("url")), None),
        "landmark_to_camera": m["landmark_to_camera"],
        "sources": m["sources"], "in_category": m["in_category"],
        "validation_at_freeze": {"accepted": m["validation"]["accepted"], "reasons": m["validation"]["reasons"]},
        **gt_fields(gt, m["file_name"]),
    }
    data, check = rebuild_photo(client, entry)
    if sha256_bytes(data) != pix["sha256"] or not all(check.values()):
        raise SystemExit(f"{title}: rebuilding from the cache does not reproduce the collected file {check}")
    return entry


PAGE_SKIPS_NOT_EXAMINED = ("unsupported_format", "panorama", "too_small_original", "page_missing")


def building_stats(stats: dict[str, Any], cands: list[dict[str, Any]], photos: list[dict[str, Any]]) -> dict[str, Any]:
    """Collection funnel + summaries of the frozen photos (all counts, no rates: readers divide)."""
    sel = stats["listing"].get("selection", {})
    off = set(sel.get("off_target_dropped", []))
    # page-level fate of every screened candidate, by source kind
    fate: dict[str, Counter[str]] = defaultdict(Counter)
    for row in cands + [{"title": t, "sources": None, "outcome": "off_target"} for t in off]:
        # off-target files passed the page screen; their sources are not recorded by collect
        kinds = "+".join(sorted({s.split(":", 1)[0] for s in row["sources"]})) if row["sources"] else "off_target"
        o = row["outcome"]
        if o == "off_target":
            f = "gps_in_radius"
        elif o in PAGE_SKIPS_NOT_EXAMINED:
            f = "not_examined"
        elif o == "page_no_exif":
            f = "no_exif"
        elif o == "page_no_gps":
            f = "exif_no_gps"
        elif o == "gps_outside_radius" and row.get("detail"):
            f = "gps_outside_radius"
        else:
            f = "gps_in_radius"
        fate[kinds][f] += 1
    ex = [p["exif"] for p in photos]
    authors = Counter(p["author_url"] or p["author"] or "?" for p in photos)
    days = Counter((e["captured_at"] or "?")[:10] for e in ex)
    funnel = stats["funnel"]
    return {
        "listed": sel.get("listed"), "screened": sel.get("screened"),
        "screened_by_source_kinds": sel.get("screened_by_source_kinds"),
        "page_screen_skips": sel.get("page_screen_skips"), "page_passed": sel.get("page_passed"),
        "off_target_dropped": len(off),
        "fate_by_source_kinds": {k: dict(v) for k, v in sorted(fate.items())},
        "phase2_outcomes": funnel.get("outcomes"),
        "kept": len(photos),
        "with_heading": sum(e["heading_deg"] is not None for e in ex),
        "with_heading_true": sum(e["heading_deg"] is not None and e["heading_ref"] == "T" for e in ex),
        "with_heading_magnetic": sum(e["heading_deg"] is not None and e["heading_ref"] == "M" for e in ex),
        "with_focal": sum(e["focal_mm"] is not None or e["focal_35mm"] is not None for e in ex),
        "with_focal_35mm": sum(e["focal_35mm"] is not None for e in ex),
        "with_gps_accuracy": sum(e["gps_accuracy_m"] is not None for e in ex),
        "with_make_model": sum(bool(e["make"] or e["model"]) for e in ex),
        "distinct_authors": len(authors), "top_author_photos": authors.most_common(1)[0][1] if authors else 0,
        "distinct_cameras": len({(e["make"], e["model"]) for e in ex if e["make"] or e["model"]}),
        "capture_days": len(days), "top_day_photos": days.most_common(1)[0][1] if days else 0,
        "sectors": {sector_label(s, NUM_SECTORS): n for s, n in sorted(Counter(
            p["landmark_to_camera"]["sector"] for p in photos).items())},
        "distance_m_median": sorted(p["landmark_to_camera"]["distance_m"] for p in photos)[len(photos) // 2]
        if photos else None,
        "validator_accepted": sum(p["validation_at_freeze"]["accepted"] for p in photos),
        "validator_reasons": dict(Counter(r for p in photos for r in p["validation_at_freeze"]["reasons"])),
        "in_category": sum(p["in_category"] for p in photos),
        "from_mdx_titles": sum(any(s.startswith("titles:") for s in p["sources"]) for p in photos),
        "with_gt_pose": sum(p["gt_image"] is not None for p in photos),
        "with_gt_pose_additional_component": sum("gt_additional" in p for p in photos),
        "licences": dict(Counter(p["licence"] or "unknown" for p in photos).most_common()),
        "bytes": sum(p["bytes"] for p in photos),
        "collect_elapsed_s": stats.get("elapsed_s"),
        "collect_http": {k: stats["http"].get(k) for k in ("network_requests", "cache_hits", "bytes_downloaded",
                                                             "http_429", "http_5xx", "by_host")},
    }


def freeze(*, testset_id: str, slugs: list[str], landmarks: Path, fetch_root: Path, http_cache: Path,
           mdx_root: Path | None, licence_parquet: Path | None, description: str, extra: dict[str, Any],
           out_dir: Path | None = None) -> Path:
    """Write eval/testsets/<testset_id>/testset.json + osm/ from collector outputs. Never overwrites."""
    out_dir = out_dir or TESTSETS_DIR / testset_id
    target = out_dir / "testset.json"
    if target.exists():
        raise SystemExit(f"{target} exists: a released test set is immutable, freeze a new id instead")
    client = fc.PoliteClient(http_cache, offline=True)
    lms = {lm["slug"]: lm for lm in json.loads(landmarks.read_text(encoding="utf-8"))}
    scenes = {lms[s]["megadepth_x"]["scene"] for s in slugs if lms[s].get("megadepth_x")}
    rows = _mdx_rows(licence_parquet, scenes) if licence_parquet and scenes else {}
    (out_dir / "osm").mkdir(parents=True, exist_ok=True)
    buildings = []
    for slug in slugs:
        lm = lms[slug]
        src = fetch_root / slug
        manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
        stats = json.loads((src / "screening_stats.json").read_text(encoding="utf-8"))
        cands = json.loads((src / "candidates.json").read_text(encoding="utf-8"))
        keys = {m["title"]: m["file_name"] for m in manifest["photos"]}
        if len(set(keys.values())) != len(keys):
            raise SystemExit(f"{slug}: photo keys (file names) are not unique")
        gt = None
        if mdx_root is not None and lm.get("megadepth_x"):
            gt = gt_reference(lm, mdx_root / slug, rows.get(lm["megadepth_x"]["scene"], []), keys)
        photos = [freeze_photo(client, m, gt) for m in manifest["photos"]]
        outline_src = Path(lm["osm"]["file"])
        outline_src = outline_src if outline_src.is_absolute() else REPO_ROOT / outline_src
        outline_dst = out_dir / "osm" / f"{slug}.geojson"
        shutil.copyfile(outline_src, outline_dst)
        buildings.append({
            "slug": slug, "name": lm["name"], "name_zh": lm.get("name_zh"), "wikidata": lm.get("wikidata"),
            "centre": {"lat": lm["lat"], "lon": lm["lon"], "source": lm["centre"]["source"]},
            "radius_m": lm["radius_m"],
            "osm_outline": {"file": f"osm/{slug}.geojson", "sha256": sha256_file(outline_dst),
                            "osm_type": lm["osm"]["type"], "osm_id": lm["osm"]["id"], "url": lm["osm"]["url"],
                            "licence": lm["osm"]["license"], "attribution": lm["osm"]["attribution"]},
            "commons_categories": lm["commons_categories"],
            "shape_notes": lm.get("shape_notes"),
            "gt_reference": gt,
            "stats": building_stats(stats, cands, photos),
            "collect_params": {**{k: stats["params"][k] for k in ("radius", "category", "category_depth",
                                                                   "exclude_category", "max_photos", "exif_source")},
                               # input files by name only (they lived in the collector's scratch dir)
                               "titles": [Path(x).name for x in stats["params"]["titles"]],
                               "search_html": [Path(x).name for x in stats["params"]["search_html"]]},
            "photos": photos,
        })
    ts = {
        "schema_version": SCHEMA_VERSION, "testset_id": testset_id,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "description": description, **extra, "schema_notes": SCHEMA_NOTES, "buildings": buildings,
    }
    target.write_text(dump_testset(ts), encoding="utf-8")
    return target


def dump_testset(ts: dict[str, Any]) -> str:
    """testset.json text: indented, but one compact line per photo (a third smaller, one-line diffs)."""
    photos = {}
    shell = json.loads(json.dumps(ts))
    for i, b in enumerate(shell["buildings"]):
        photos[f"@@photos-{i}@@"] = b["photos"]
        b["photos"] = f"@@photos-{i}@@"
    text = json.dumps(shell, indent=1, ensure_ascii=False)
    for tag, items in photos.items():
        lines = ",\n".join("   " + json.dumps(p, ensure_ascii=False, separators=(",", ":")) for p in items)
        text = text.replace(json.dumps(tag), "[\n" + lines + "\n  ]" if items else "[]")
    return text + "\n"


# ---------------------------------------------------------------- CLI


def _materialize_cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/materialize_testset.py",
                                 description="Materialise a frozen test set and write its integrity report.")
    ap.add_argument("testset", help="test set id under eval/testsets/, or a testset.json path")
    ap.add_argument("--cache-dir", type=Path, required=True, help="downloads + materialised photos (outside the repo)")
    ap.add_argument("--buildings", nargs="*", default=None, help="only these slugs")
    ap.add_argument("--offline", action="store_true", help="no network: use only what the cache holds")
    ap.add_argument("--refetch-pages", action="store_true",
                    help="rebuild page-metadata EXIF from freshly fetched description pages (one request each)")
    ap.add_argument("--gt", action="store_true", help="also verify / download the MegaDepth-X reference models")
    ap.add_argument("--gt-dir", type=Path, default=None,
                    help="verify reference models in place under <gt-dir>/<slug>/<component>/sparse (no download)")
    ap.add_argument("--report", type=Path, default=None, help="integrity report path")
    ap.add_argument("--min-interval", type=float, default=fc.MIN_INTERVAL_S)
    ap.add_argument("--max-wait", type=float, default=900.0)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    fc.check_outside_repo(a.cache_dir, "--cache-dir")
    client = fc.PoliteClient(a.cache_dir / "http-cache", min_interval_s=a.min_interval, max_wait_s=a.max_wait,
                             offline=a.offline)
    rep = materialize(a.testset, a.cache_dir, buildings=a.buildings, offline=a.offline,
                      refetch_pages=a.refetch_pages, gt=a.gt or a.gt_dir is not None, gt_dir=a.gt_dir,
                      client=client, report_path=a.report)
    t = rep["totals"]
    print(f"{rep['testset_id']} (sha256 {rep['testset_sha256'][:12]}): {t.get('photos', 0)} photos, {t.get('ok', 0)} "
          f"ok, {t.get('changed', 0)} changed, {t.get('missing', 0)} missing; {rep['http']['network_requests']} "
          f"requests, {rep['http']['http_429']} x 429, {rep['elapsed_s']} s")
    for slug, b in rep["buildings"].items():
        gt = b.get("gt", {}).get("status", "-")
        print(f"  {slug:24s} ok {b['ok']:4d}  changed {b['changed']:3d}  missing {b['missing']:3d}  gt {gt}")
    print(f"wrote {rep['report_path']}")
    return 0 if rep["complete"] else 1


def _collect_cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/materialize_testset.py collect")
    ap.add_argument("--pool", type=int, default=225, help="stop screening once this many pass the page screen")
    ap.add_argument("--max-screen", type=int, default=600, help="screen at most this many candidate pages")
    ap.add_argument("--keep-off-target", action="store_true", help="do not drop interior / 'views from' files")
    ap.add_argument("fetch_args", nargs=argparse.REMAINDER, help="-- then scripts/fetch_commons_landmark.py args")
    a = ap.parse_args(argv)
    rest = a.fetch_args[1:] if a.fetch_args[:1] == ["--"] else a.fetch_args
    return collect(rest, pool=a.pool, max_screen=a.max_screen, drop_off_target=not a.keep_off_target)


def _freeze_cli(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python scripts/materialize_testset.py freeze")
    ap.add_argument("--testset-id", required=True)
    ap.add_argument("--buildings", nargs="+", required=True, help="slugs from --landmarks, in order")
    ap.add_argument("--landmarks", type=Path, default=REPO_ROOT / "eval" / "landmarks.json")
    ap.add_argument("--fetch-root", type=Path, required=True, help="collector output: <fetch-root>/<slug>/manifest.json")
    ap.add_argument("--http-cache", type=Path, default=None, help="collector HTTP cache (default <fetch-root>/http-cache)")
    ap.add_argument("--mdx-root", type=Path, default=None, help="MegaDepth-X sparse models: <mdx-root>/<slug>/<comp>/sparse")
    ap.add_argument("--licence-parquet", type=Path, default=None, help="MegaDepth-X license.parquet")
    ap.add_argument("--description", required=True)
    ap.add_argument("--extra-json", type=Path, default=None, help="extra top-level fields (licence note, selection, ...)")
    ap.add_argument("--out-dir", type=Path, default=None)
    a = ap.parse_args(argv)
    extra = json.loads(a.extra_json.read_text(encoding="utf-8")) if a.extra_json else {}
    path = freeze(testset_id=a.testset_id, slugs=a.buildings, landmarks=a.landmarks, fetch_root=a.fetch_root,
                  http_cache=a.http_cache or a.fetch_root / "http-cache", mdx_root=a.mdx_root,
                  licence_parquet=a.licence_parquet, description=a.description, extra=extra, out_dir=a.out_dir)
    print(f"wrote {path} (sha256 {sha256_file(path)})")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    commands = {"collect": _collect_cli, "freeze": _freeze_cli, "materialize": _materialize_cli}
    if argv and argv[0] in commands:
        return commands[argv[0]](argv[1:])
    return _materialize_cli(argv)


if __name__ == "__main__":
    raise SystemExit(main())
