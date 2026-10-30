"""A landmark's persistent map (docs/ARCHITECTURE.md 2.2): the feature and match cache (stage 1a), the saved
published models, and what a job continues of them (stage 1b, MapBase).

Every reconstruction job used to extract SIFT features from all of its photos and match all of its pairs again
(a photo was extracted 9x on average under the benchmark's session arrival, a pair verified 5.7x). FeatureCache keeps,
per landmark, exactly the rows COLMAP wrote into a job's database (data/maps/<landmark>/cache.sqlite3, outside the
served models dir), and the next job of the landmark seeds its own database with them before COLMAP runs:
- features: an image's keypoints and descriptors rows, keyed by the sha256 of its processed file (the pixels SIFT
  reads) and feature_version (pycolmap's version, the extraction options);
- pairs: a pair's matches and two-view geometry rows, keyed by both images' keys in the order COLMAP matched them
  (job_pairs), whether that order is the order of their image ids (rows store the pair in image-id order), the
  signature of both COLMAP cameras (camera_signature: verification reads them) and match_version.
COLMAP skips an image whose features are in the database and a pair whose matches and two-view geometry are
(pycolmap 4.2.1), so a job's database ends with the rows it would have computed itself: CPU SIFT, matching and the
seeded RANSAC verification are deterministic, an image's features do not depend on the other images of the job, nor a
pair's rows on the rest of the job (measured on real photos: docs/BENCHMARK.md). Rows are copied as raw SQLite rows,
byte for byte. Only what today's pairing picks is seeded (job_pairs), so a cached pair never adds a pair to a job.
Every connection to a job database is closed before COLMAP opens it again: pycolmap links its own SQLite, and two
SQLite libraries in one process must not hold the same WAL file open at once.
A pair whose two-view geometry carries cameras of its own (COLMAP stores the cameras it estimated for some
uncalibrated pairs under the job's camera ids) is not cached.

The cache is on with Params.feature_cache. Entries are keyed by content, so concurrent writers stay consistent
(INSERT OR IGNORE, WAL, a busy timeout); nothing is evicted (about 1.2 MB per photo).

Saved maps (save_map): every published model's relative reconstruction (its job's sparse/0) with map.json beside it
(model, target, image name -> photo id, feature version, the GeoAnchor placing it on the map, inputs), under
data/maps/<landmark>/models/<model id>/, written whole or not at all; the newest MAPS_KEPT per target stay. A later job
of the target continues the one its target shows (Params.map_continue: jobs.JobRunner._map_plan hands the reconstructor
a MapBase; map.json records mode, parent_model_id, n_scratch and continuations) and the stylize step reads them (F.1);
nothing under data/maps is served.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import sqlite3
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

CACHE_FILE = "cache.sqlite3"      # Settings.maps_dir / <landmark id> / CACHE_FILE
MODELS_DIR = "models"             # Settings.maps_dir / <landmark id> / MODELS_DIR / <model id> / {sparse, map.json}
MAP_SCHEMA = "crowd3d.map/1"
MAPS_KEPT = 2                     # saved maps kept per target (kind, sector)
SCHEMA = 1
# COLMAP's own columns (pycolmap 4.2.1), copied as they are; versions name the pycolmap release, so another schema
# never meets these rows.
_FEATURE_COLUMNS = {"keypoints": ("rows", "cols", "data"), "descriptors": ("type", "rows", "cols", "data")}
_PAIR_COLUMNS = {"matches": ("rows", "cols", "data"),
                 "two_view_geometries": ("rows", "cols", "data", "config", "F", "E", "H", "qvec", "tvec", "camera1",
                                         "camera2")}
_DDL = f"""
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS features (feat_key TEXT NOT NULL, version TEXT NOT NULL, created_at REAL NOT NULL,
    {", ".join(f"{t}_{c}" for t, cols in _FEATURE_COLUMNS.items() for c in cols)},
    PRIMARY KEY (feat_key, version));
CREATE TABLE IF NOT EXISTS pairs (key1 TEXT NOT NULL, key2 TEXT NOT NULL, version TEXT NOT NULL, sig1 TEXT NOT NULL,
    sig2 TEXT NOT NULL, ids_ascending INTEGER NOT NULL, created_at REAL NOT NULL,
    {", ".join(f"{t}_{c}" for t, cols in _PAIR_COLUMNS.items() for c in cols)},
    PRIMARY KEY (key1, key2, version, sig1, sig2, ids_ascending));
CREATE TABLE IF NOT EXISTS uses (at REAL NOT NULL, label TEXT, features_hit INTEGER, features INTEGER,
    pairs_hit INTEGER, pairs INTEGER, seconds REAL);
"""


@dataclass
class CacheStats:
    """One job's use of the cache: features and pairs found in it, of those the job needed."""

    features_hit: int = 0
    features: int = 0
    pairs_hit: int = 0
    pairs: int = 0

    def to_meta(self) -> dict[str, int]:
        return asdict(self)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _options(options: Any) -> dict[str, Any]:
    """An options object's values that change what COLMAP computes (threads and GPU choice do not; jobs run on CPU)."""
    skip = {"num_threads", "use_gpu", "gpu_index"}

    def clean(v: Any) -> Any:
        if isinstance(v, dict):
            return {k: clean(x) for k, x in v.items() if k not in skip}
        return v if isinstance(v, (int, float, str, bool, type(None))) else str(v)

    return clean(options.todict())


def _version(pycolmap, **parts: Any) -> str:
    doc = {"pycolmap": pycolmap.__version__, "device": "cpu", **parts}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:32]


def extraction_options(pycolmap, max_side_px: int):
    """The SIFT extraction options of a job (ColmapReconstructor): COLMAP's defaults at the processed copies' size."""
    options = pycolmap.FeatureExtractionOptions()
    options.max_image_size = max_side_px
    return options


def feature_version(pycolmap, extraction_options_) -> str:
    return _version(pycolmap, extraction=_options(extraction_options_))


def job_feature_version(max_side_px: int) -> str:
    """feature_version of the features a ColmapReconstructor job extracts from copies of this size."""
    import pycolmap

    return feature_version(pycolmap, extraction_options(pycolmap, max_side_px))


def match_version(pycolmap, verification_options, matching_options=None) -> str:
    """The version of a pair's rows; matching_options None: COLMAP's defaults (what reconstruct uses)."""
    matching = matching_options if matching_options is not None else pycolmap.FeatureMatchingOptions()
    return _version(pycolmap, matching=_options(matching), verification=_options(verification_options))


def camera_signatures(db_path: Path) -> dict[int, str]:
    """image id -> sha256 of its COLMAP camera row (model, size, parameters, prior flag): what verification reads."""
    with _sqlite(db_path) as db:
        rows = db.execute("SELECT i.image_id, c.model, c.width, c.height, c.params, c.prior_focal_length"
                          " FROM images i JOIN cameras c ON c.camera_id = i.camera_id").fetchall()
    return {r[0]: hashlib.sha256(repr(r[1:]).encode()).hexdigest()[:32] for r in rows}


def job_pairs(pycolmap, db_path: Path, spatial_options=None) -> list[tuple[int, int]]:
    """The image pairs COLMAP's matching of this database computes, each in the order it matches them: the order of
    the pair generator (exhaustive, or spatial with these options and the pose priors already written), whose first
    occurrence of a pair is the one matched (a later one finds it in the database)."""
    db = pycolmap.Database.open(db_path)
    try:
        if spatial_options is None:
            generator = pycolmap.ExhaustivePairGenerator(pycolmap.ExhaustivePairingOptions(), db)
        else:
            generator = pycolmap.SpatialPairGenerator(spatial_options, db)
        pairs = generator.all_pairs()
    finally:
        db.close()
    seen: set[tuple[int, int]] = set()
    out = []
    for a, b in pairs:
        a, b = int(a), int(b)
        if a != b and (min(a, b), max(a, b)) not in seen:
            seen.add((min(a, b), max(a, b)))
            out.append((a, b))
    return out


def pair_id(a: int, b: int) -> int:
    """COLMAP's pair id (pycolmap.image_pair_to_pair_id): the smaller image id first."""
    lo, hi = min(a, b), max(a, b)
    return 2147483647 * lo + hi


@contextmanager
def _sqlite(path: Path) -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(path, timeout=60.0)
    try:
        yield conn
    finally:
        conn.close()


class FeatureCache:
    """The feature and match cache of one landmark (module docstring). Each call opens its own connection."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _sqlite(self.path) as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(_DDL)
            conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema', ?)", (str(SCHEMA),))
            yield conn

    def seed_features(self, db_path: Path, keys: dict[int, str], version: str) -> int:
        """Write the cached features of the database's images (image id -> key) into it; returns how many."""
        cols = [f"{t}_{c}" for t, cc in _FEATURE_COLUMNS.items() for c in cc]
        with self._connect() as cache, _sqlite(db_path) as db:
            have = {r[0] for r in db.execute("SELECT image_id FROM keypoints")}
            hits = 0
            for image_id, key in keys.items():
                if image_id in have:
                    continue
                row = cache.execute(f"SELECT {', '.join(cols)} FROM features WHERE feat_key = ? AND version = ?",
                                    (key, version)).fetchone()
                if row is None:
                    continue
                values = dict(zip(cols, row))
                for table, tcols in _FEATURE_COLUMNS.items():
                    db.execute(f"INSERT INTO {table} (image_id, {', '.join(tcols)}) VALUES (?{', ?' * len(tcols)})",
                               (image_id, *(values[f"{table}_{c}"] for c in tcols)))
                hits += 1
            db.commit()
        return hits

    def seed_pairs(self, db_path: Path, pairs: list[tuple[int, int]], keys: dict[int, str], sigs: dict[int, str],
                   version: str) -> int:
        """Write the cached rows of these pairs (job_pairs order) into the database; returns how many."""
        cols = [f"{t}_{c}" for t, cc in _PAIR_COLUMNS.items() for c in cc]
        with self._connect() as cache, _sqlite(db_path) as db:
            have = {r[0] for r in db.execute("SELECT pair_id FROM two_view_geometries")}
            hits = 0
            for a, b in pairs:
                pid = pair_id(a, b)
                if pid in have or a not in keys or b not in keys:
                    continue
                row = cache.execute(
                    f"SELECT {', '.join(cols)} FROM pairs WHERE key1 = ? AND key2 = ? AND version = ? AND sig1 = ?"
                    " AND sig2 = ? AND ids_ascending = ?", (keys[a], keys[b], version, sigs[a], sigs[b], int(a < b))
                ).fetchone()
                if row is None:
                    continue
                values = dict(zip(cols, row))
                for table, tcols in _PAIR_COLUMNS.items():
                    db.execute(f"DELETE FROM {table} WHERE pair_id = ?", (pid,))
                    db.execute(f"INSERT INTO {table} (pair_id, {', '.join(tcols)}) VALUES (?{', ?' * len(tcols)})",
                               (pid, *(values[f"{table}_{c}"] for c in tcols)))
                hits += 1
            db.commit()
        return hits

    def harvest(self, db_path: Path, keys: dict[int, str], feature_version_: str, pairs: list[tuple[int, int]],
                sigs: dict[int, str], match_version_: str) -> tuple[int, int]:
        """Store what the database holds for these images and pairs (job_pairs order) and the cache lacks; returns
        (features, pairs) added."""
        now = time.time()
        fcols = [f"{t}_{c}" for t, cc in _FEATURE_COLUMNS.items() for c in cc]
        pcols = [f"{t}_{c}" for t, cc in _PAIR_COLUMNS.items() for c in cc]
        added_f = added_p = 0
        with self._connect() as cache, _sqlite(db_path) as db:
            for image_id, key in keys.items():
                if cache.execute("SELECT 1 FROM features WHERE feat_key = ? AND version = ?",
                                 (key, feature_version_)).fetchone():
                    continue
                kp = db.execute("SELECT rows, cols, data FROM keypoints WHERE image_id = ?", (image_id,)).fetchone()
                de = db.execute("SELECT type, rows, cols, data FROM descriptors WHERE image_id = ?",
                                (image_id,)).fetchone()
                if kp is None or de is None:
                    continue
                added_f += cache.execute(
                    f"INSERT OR IGNORE INTO features (feat_key, version, created_at, {', '.join(fcols)})"
                    f" VALUES (?, ?, ?{', ?' * len(fcols)})", (key, feature_version_, now, *kp, *de)).rowcount
            for a, b in pairs:
                if a not in keys or b not in keys:
                    continue
                pid, k = pair_id(a, b), (keys[a], keys[b], match_version_, sigs[a], sigs[b], int(a < b))
                if cache.execute("SELECT 1 FROM pairs WHERE key1 = ? AND key2 = ? AND version = ? AND sig1 = ?"
                                 " AND sig2 = ? AND ids_ascending = ?", k).fetchone():
                    continue
                m = db.execute("SELECT rows, cols, data FROM matches WHERE pair_id = ?", (pid,)).fetchone()
                g = db.execute(f"SELECT {', '.join(_PAIR_COLUMNS['two_view_geometries'])} FROM two_view_geometries"
                               " WHERE pair_id = ?", (pid,)).fetchone()
                if m is None or g is None or g[-2] is not None or g[-1] is not None:
                    continue
                added_p += cache.execute(
                    f"INSERT OR IGNORE INTO pairs (key1, key2, version, sig1, sig2, ids_ascending, created_at,"
                    f" {', '.join(pcols)}) VALUES (?, ?, ?, ?, ?, ?, ?{', ?' * len(pcols)})", (*k, now, *m, *g)
                ).rowcount
            cache.commit()
        return added_f, added_p

    def record_use(self, stats: CacheStats, seconds: float, label: str | None = None) -> None:
        with self._connect() as cache:
            cache.execute("INSERT INTO uses VALUES (?, ?, ?, ?, ?, ?, ?)",
                          (time.time(), label, stats.features_hit, stats.features, stats.pairs_hit, stats.pairs,
                           round(seconds, 3)))
            cache.commit()

    def uses(self) -> list[dict[str, Any]]:
        """Every job's CacheStats and seconds spent in the cache, oldest first."""
        if not self.path.is_file():
            return []
        with self._connect() as cache:
            cur = cache.execute("SELECT * FROM uses ORDER BY at")
            names = [d[0] for d in cur.description]
            return [dict(zip(names, r)) for r in cur.fetchall()]


class JobCache:
    """One job's use of a FeatureCache, in reconstruct's order: seed_features before extract_features, seed_pairs
    after the focal (and GPS) priors and before matching, harvest after matching (before mapping, so a job killed while
    mapping leaves a warm cache for its retry). A failing cache never fails the job: it logs and gets out of the way."""

    def __init__(self, cache: FeatureCache, pycolmap, db_path: Path, image_paths: dict[str, Path], extraction_options,
                 verification_options) -> None:
        self.cache, self.pycolmap, self.db_path = cache, pycolmap, Path(db_path)
        self.image_paths = image_paths
        self.feature_version = feature_version(pycolmap, extraction_options)
        self.match_version = match_version(pycolmap, verification_options)
        self.stats = CacheStats()
        self.seconds = 0.0
        self.keys: dict[int, str] = {}
        self.pairs: list[tuple[int, int]] = []
        self.sigs: dict[int, str] = {}
        self.ok = True

    def _run(self, what: str, step) -> None:
        if not self.ok:
            return
        t = time.monotonic()
        try:
            step()
        except (sqlite3.Error, OSError, RuntimeError) as e:
            self.ok = False
            log.warning("feature cache %s: %s failed (%s); this job goes on without it", self.cache.path, what, e)
        finally:
            self.seconds += time.monotonic() - t

    def seed_features(self) -> None:
        def step() -> None:
            with _sqlite(self.db_path) as db:
                ids = dict(db.execute("SELECT name, image_id FROM images").fetchall())
            self.keys = {ids[name]: file_sha256(path) for name, path in self.image_paths.items() if name in ids}
            self.stats.features = len(self.keys)
            self.stats.features_hit = self.cache.seed_features(self.db_path, self.keys, self.feature_version)
        self._run("seeding features", step)

    def seed_pairs(self, spatial_options=None) -> None:
        def step() -> None:
            self.pairs = job_pairs(self.pycolmap, self.db_path, spatial_options)
            self.sigs = camera_signatures(self.db_path)
            self.stats.pairs = len(self.pairs)
            self.stats.pairs_hit = self.cache.seed_pairs(self.db_path, self.pairs, self.keys, self.sigs,
                                                         self.match_version)
        self._run("seeding pairs", step)

    def harvest(self, label: str | None = None) -> None:
        """Store the job's new rows and record its use (CacheStats, seconds spent in the cache)."""
        self._run("harvest", lambda: self.cache.harvest(self.db_path, self.keys, self.feature_version, self.pairs,
                                                        self.sigs, self.match_version))
        self._run("recording its use", lambda: self.cache.record_use(self.stats, self.seconds, label))


# ---------------------------------------------------------------- saved maps


def save_map(root: Path, model_id: str, sparse_dir: Path, record: dict[str, Any]) -> Path:
    """Copy a published model's sparse dir and its map.json (record plus schema and image names) to
    root/<model_id>, through a temporary dir renamed into place, so a reader never sees half a map; returns the dir."""
    import pycolmap

    root.mkdir(parents=True, exist_ok=True)
    for stale in root.glob(".tmp-*"):             # left by a process that died while saving (one writer per landmark)
        shutil.rmtree(stale, ignore_errors=True)
    tmp = root / f".tmp-{model_id}-{uuid.uuid4().hex[:8]}"
    try:
        shutil.copytree(sparse_dir, tmp / "sparse")
        images = {im.name: Path(im.name).stem.split("_", 1)[-1]          # reconstruct.photo_id_from_image_name
                  for im in pycolmap.Reconstruction(str(tmp / "sparse")).images.values()}
        doc = {"schema": MAP_SCHEMA, "model_id": model_id, **record, "images": dict(sorted(images.items()))}
        (tmp / "map.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        final = root / model_id
        os.replace(tmp, final)
        return final
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_map(root: Path, model_id: str) -> dict[str, Any] | None:
    """The map.json of the saved map of this model under root, None when there is none (never saved, or pruned)."""
    path = root / model_id / "map.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        log.warning("saved map %s is unreadable: %s", path, e)
        return None


@dataclass
class MapBase:
    """The saved map a job continues from (stage 1b): the published model its target shows. n_scratch is the input
    count of its last from-scratch ancestor (itself, when it was built from scratch), continuations how many
    continuations lie between that ancestor and it."""

    model_id: str
    sparse_dir: Path
    images: dict[str, str]          # image name in its job -> photo id (registered photos only)
    n_scratch: int
    continuations: int


def saved_maps(root: Path) -> list[dict[str, Any]]:
    """The map.json of every saved map under root, oldest first (created_at, then model id)."""
    docs = []
    for d in sorted(root.iterdir()) if root.is_dir() else []:
        if d.name.startswith(".") or not (d / "map.json").is_file():
            continue
        docs.append(json.loads((d / "map.json").read_text(encoding="utf-8")))
    return sorted(docs, key=lambda m: (m.get("created_at") or 0.0, m["model_id"]))


def prune_maps(root: Path, kind: str, sector: int | None, keep: int = MAPS_KEPT) -> list[str]:
    """Delete all but the newest keep saved maps of the target (kind, sector); returns the deleted model ids."""
    mine = [m for m in saved_maps(root) if (m.get("kind"), m.get("sector")) == (kind, sector)]
    gone = [m["model_id"] for m in mine[:max(0, len(mine) - keep)]]
    for model_id in gone:
        shutil.rmtree(root / model_id, ignore_errors=True)
    return gone
