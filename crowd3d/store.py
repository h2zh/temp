"""SQLite persistence for landmarks, photos, contributors, jobs, models and the photos' visual localizations.

Thread-safe: the API thread and the job worker thread share one Store.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from typing import Any, TypeVar

from .config import Settings
from .geo import haversine_m, local_bearing_distance
from .records import (
    LOC_ACCEPTED,
    LOC_STATUSES,
    PHOTO_ACCEPTED,
    TERMS_V1,
    TERMS_VERSIONS,
    Contributor,
    Job,
    Landmark,
    LocRecord,
    ModelRecord,
    PhotoRecord,
)


log = logging.getLogger(__name__)

# The store's meta key recording the one-time D32 migration (Store._migrate_d32_terms).
MIGRATION_D32_TERMS_KEY = "migration.d32-terms"


class NotFound(KeyError):
    pass


class AlreadyExists(ValueError):
    pass


_SCHEMA = """
CREATE TABLE IF NOT EXISTS landmarks (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    lat REAL NOT NULL,
    lon REAL NOT NULL,
    radius_m REAL NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,
    landmark_id TEXT REFERENCES landmarks(id),
    user_id TEXT NOT NULL,
    status TEXT NOT NULL,
    uploaded_at REAL NOT NULL,
    reasons TEXT NOT NULL,
    warnings TEXT NOT NULL,
    lat REAL,
    lon REAL,
    alt REAL,
    heading_deg REAL,
    bearing_deg REAL,
    distance_m REAL,
    sector INTEGER,
    width INTEGER,
    height INTEGER,
    focal_35mm REAL,
    phash TEXT,
    original_path TEXT,
    processed_path TEXT,
    captured_at TEXT,
    gps_accuracy_m REAL,
    outline_distance_m REAL,
    subject_share REAL,
    heading_ref TEXT,
    location_consent TEXT,
    contributor_id TEXT
);
CREATE INDEX IF NOT EXISTS photos_landmark ON photos(landmark_id, status);
CREATE TABLE IF NOT EXISTS contributors (
    id TEXT PRIMARY KEY,
    nickname TEXT NOT NULL,
    token_hash TEXT NOT NULL UNIQUE,
    terms TEXT NOT NULL,
    terms_accepted_at REAL NOT NULL,
    registered_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    landmark_id TEXT NOT NULL REFERENCES landmarks(id),
    kind TEXT NOT NULL,
    sector INTEGER,
    status TEXT NOT NULL,
    photo_ids TEXT NOT NULL,
    created_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    error TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    model_id TEXT
);
CREATE INDEX IF NOT EXISTS jobs_landmark ON jobs(landmark_id);
CREATE INDEX IF NOT EXISTS jobs_status ON jobs(status);
CREATE TABLE IF NOT EXISTS models (
    id TEXT PRIMARY KEY,
    landmark_id TEXT NOT NULL REFERENCES landmarks(id),
    job_id TEXT NOT NULL REFERENCES jobs(id),
    kind TEXT NOT NULL,
    sector INTEGER,
    ply_path TEXT NOT NULL,
    num_photos INTEGER NOT NULL,
    meta TEXT NOT NULL,
    created_at REAL NOT NULL,
    stylized TEXT
);
CREATE INDEX IF NOT EXISTS models_landmark ON models(landmark_id);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS localizations (
    photo_id TEXT PRIMARY KEY REFERENCES photos(id),
    landmark_id TEXT NOT NULL REFERENCES landmarks(id),
    status TEXT NOT NULL,
    updated_at REAL NOT NULL,
    reason TEXT,
    maps_key TEXT,
    attempts INTEGER NOT NULL DEFAULT 0,
    model_id TEXT,
    pose TEXT,
    stats TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS localizations_landmark ON localizations(landmark_id);
"""

# Columns added after the first schema: databases created before get them on open.
_ADDED_COLUMNS = {
    # location_consent: the user agreement's version an accepted photo was uploaded under (D32); a row stored before
    # the column existed gets it from the one-time migration (Store._migrate_d32_terms); contributor_id: the
    # registration an upload was made under (D33), NULL on rows stored before (no data migration: they keep user_id)
    "photos": [("gps_accuracy_m", "REAL"), ("outline_distance_m", "REAL"), ("subject_share", "REAL"),
               ("heading_ref", "TEXT"), ("location_consent", "TEXT"), ("contributor_id", "TEXT")],
    "jobs": [("attempts", "INTEGER NOT NULL DEFAULT 0"), ("model_id", "TEXT")],
    "models": [("stylized", "TEXT")],
}

# Columns holding lists/dicts, stored as JSON text (SQL NULL, in a column added later, reads as None).
_JSON_COLUMNS = {"reasons", "warnings", "photo_ids", "meta", "stylized", "pose", "stats"}
_JOB_UPDATABLE = {"status", "started_at", "finished_at", "error", "attempts"}

R = TypeVar("R")


def _json_default(o: Any) -> Any:
    # Reconstructors may hand back numpy scalars/arrays in their metadata.
    if hasattr(o, "tolist"):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serializable")


def to_json(value: Any, **kwargs: Any) -> str:
    """json.dumps that also accepts numpy scalars/arrays; used for every JSON column."""
    return json.dumps(value, ensure_ascii=False, default=_json_default, **kwargs)


class Store:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        settings.ensure_dirs()
        self._lock = threading.RLock()
        # Autocommit; every public method is a single statement or a read, except the two consent withdrawals,
        # whose two statements run in one explicit transaction (_transaction), as does the D32 migration on open.
        self._conn = sqlite3.connect(settings.db_path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)
        for table, columns in _ADDED_COLUMNS.items():
            have = {r[1] for r in self._conn.execute(f"PRAGMA table_info({table})")}
            for name, decl in columns:
                if name not in have:
                    self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {decl}")
        self._migrate_d32_terms()

    def _migrate_d32_terms(self) -> None:
        """One-time migration (D32 addendum, owner 2026-11-05: the site is not live, every photo so far is a test
        photo): every accepted photo counts as agreed under the current user agreement (TERMS_V1), every rejected one
        as not shared; the share-location-v1/v2 consents, the photos without one and the withdrawn ones included.
        Recorded in meta MIGRATION_D32_TERMS_KEY ({"at", "photos_agreed": accepted photos changed, "rejected_cleared":
        other photos whose stale consent was cleared, "version"}) in the same transaction, so it runs once per store:
        a later withdrawal stays withdrawn across restarts. A fresh store records 0 and 0.
        Remove this migration before the site launches: a restored pre-D32 backup would otherwise be agreed
        silently. Nothing else relies on it: the benchmark harness counts every accepted upload of a run recorded
        before the agreement as agreed itself (eval_real_landmark.run_consent_assumed)."""
        args = {"accepted": PHOTO_ACCEPTED, "version": TERMS_V1}
        with self._transaction() as conn:
            if conn.execute("SELECT 1 FROM meta WHERE key = ?", (MIGRATION_D32_TERMS_KEY,)).fetchone():
                return
            agreed = conn.execute("UPDATE photos SET location_consent = :version WHERE status = :accepted"
                                  " AND location_consent IS NOT :version", args).rowcount
            cleared = conn.execute("UPDATE photos SET location_consent = NULL WHERE status IS NOT :accepted"
                                   " AND location_consent IS NOT NULL", args).rowcount
            record = {"at": time.time(), "photos_agreed": agreed, "rejected_cleared": cleared, "version": TERMS_V1}
            conn.execute("INSERT INTO meta (key, value) VALUES (?, ?)", (MIGRATION_D32_TERMS_KEY, to_json(record)))
        if agreed or cleared:
            log.warning("D32: %d accepted photos now count as agreed under %s, %d rejected photos' stale consent "
                        "cleared (owner, 2026-11-05: every photo so far is a test photo)", agreed, TERMS_V1, cleared)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # Landmarks
    def create_landmark(self, landmark: Landmark) -> Landmark:
        """Raises AlreadyExists on duplicate id."""
        self._insert("landmarks", landmark)
        return self.get_landmark(landmark.id)

    def get_landmark(self, landmark_id: str) -> Landmark:
        """Raises NotFound."""
        return self._get(Landmark, "landmarks", landmark_id)

    def list_landmarks(self) -> list[Landmark]:
        """Ordered by created_at, then id."""
        return self._select(Landmark, "SELECT * FROM landmarks ORDER BY created_at, id")

    def find_landmark_for_location(self, lat: float, lon: float) -> Landmark | None:
        """Nearest landmark whose radius_m contains the point, else None."""
        best: tuple[float, Landmark] | None = None
        for lm in self.list_landmarks():
            # The great-circle check rules out far-away landmarks, where a tangent-plane
            # distance is meaningless; the ENU distance is what validation measures.
            if haversine_m(lm.lat, lm.lon, lat, lon) > 2 * lm.radius_m:
                continue
            _, d = local_bearing_distance(lm.lat, lm.lon, lat, lon)
            if d <= lm.radius_m and (best is None or d < best[0]):
                best = (d, lm)
        return best[1] if best else None

    # Photos
    def add_photo(self, photo: PhotoRecord) -> PhotoRecord:
        self._insert("photos", photo)
        return self.get_photo(photo.id)

    def get_photo(self, photo_id: str) -> PhotoRecord:
        return self._get(PhotoRecord, "photos", photo_id)

    def list_photos(self, landmark_id: str, status: str | None = None) -> list[PhotoRecord]:
        """Ordered by uploaded_at, then id."""
        sql, args = "SELECT * FROM photos WHERE landmark_id = ?", [landmark_id]
        if status is not None:
            sql += " AND status = ?"
            args.append(status)
        return self._select(PhotoRecord, sql + " ORDER BY uploaded_at, id", args)

    def photo_hashes(self, landmark_id: str) -> list[str]:
        """dHashes of accepted photos at the landmark."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT phash FROM photos WHERE landmark_id = ? AND status = ? AND phash IS NOT NULL"
                " ORDER BY uploaded_at, id",
                (landmark_id, PHOTO_ACCEPTED),
            ).fetchall()
        return [r[0] for r in rows]

    def contributors(self, landmark_id: str) -> list[str]:
        """user_ids (nicknames) with an accepted photo at the landmark, by user_id. No per-person counts (D30):
        together with the public photo poses they would tell whose photos are whose. The registered contributors
        (add_contributor) are never listed: only the nicknames their accepted photos carry."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT user_id FROM photos WHERE landmark_id = ? AND status = ? ORDER BY user_id",
                (landmark_id, PHOTO_ACCEPTED),
            ).fetchall()
        return [r[0] for r in rows]

    # Contributors (D33): registered once, with the user agreement version they accepted; never served
    def add_contributor(self, contributor: Contributor) -> Contributor:
        """Raises AlreadyExists on a duplicate id or token hash."""
        self._insert("contributors", contributor)
        return self._get(Contributor, "contributors", contributor.id)

    def contributor_by_token_hash(self, token_hash: str) -> Contributor | None:
        rows = self._select(Contributor, "SELECT * FROM contributors WHERE token_hash = ?", [token_hash])
        return rows[0] if rows else None

    def accept_terms(self, contributor_id: str, version: str, at: float) -> Contributor:
        """The contributor accepts the user agreement version (D33: a new version asks them once). Raises NotFound."""
        with self._lock:
            cur = self._conn.execute("UPDATE contributors SET terms = ?, terms_accepted_at = ? WHERE id = ?",
                                     (version, at, contributor_id))
            if cur.rowcount == 0:
                raise NotFound(contributor_id)
        return self._get(Contributor, "contributors", contributor_id)

    @contextlib.contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """The statements inside commit together or not at all (an exception rolls them back)."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def withdraw_location_consent(self, photo_id: str) -> PhotoRecord:
        """The uploader withdraws the photo (D30, D32): location_consent NULL, which hides its public pose and its
        public photo at once (records.location_shared), and its visual localization, made only because it was shared,
        is deleted, both or neither. Raises NotFound."""
        with self._transaction() as conn:
            cur = conn.execute("UPDATE photos SET location_consent = NULL WHERE id = ?", (photo_id,))
            if cur.rowcount == 0:
                raise NotFound(photo_id)
            conn.execute("DELETE FROM localizations WHERE photo_id = ?", (photo_id,))
        return self.get_photo(photo_id)

    def location_consent_count(self) -> int:
        """Photos, at every landmark, with a consent on record (location_consent not NULL: uploaded under the user
        agreement and not withdrawn, D30, D32)."""
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM photos WHERE location_consent IS NOT NULL").fetchone()[0]

    def clear_location_consents(self) -> int:
        """Withdraw every photo, at every landmark (crowd3d.consent.forget_lost_salt: the manage tokens can no longer
        withdraw them), and delete every visual localization with them. Returns how many photos were shared."""
        with self._transaction() as conn:
            n = conn.execute("UPDATE photos SET location_consent = NULL WHERE location_consent IS NOT NULL").rowcount
            conn.execute("DELETE FROM localizations")
        return n

    # Visual localizations (crowd3d.localize, D30): one per photo, only for a photo whose location may be shown
    def put_localization(self, rec: LocRecord) -> bool:
        """Store (or replace) the photo's localization. Only while the photo is accepted, at rec.landmark_id, uploaded
        under the user agreement and not withdrawn (records.location_shared), checked in the same statement, so a
        photo withdrawn while the localizer ran leaves nothing behind. Returns whether it was stored. Raises
        ValueError on an unknown status, or an accepted record without a model and a pose."""
        if rec.status not in LOC_STATUSES:
            raise ValueError(f"unknown localization status {rec.status!r}")
        if rec.status == LOC_ACCEPTED and (not rec.model_id or not isinstance(rec.pose, dict)):
            raise ValueError("an accepted localization needs its model_id and pose")
        names = [f.name for f in dataclasses.fields(LocRecord)]
        values = [to_json(getattr(rec, n)) if n in _JSON_COLUMNS else getattr(rec, n) for n in names]
        consents = sorted(TERMS_VERSIONS)
        sql = (f"INSERT INTO localizations ({', '.join(names)}) SELECT {', '.join('?' * len(names))}"
               " WHERE EXISTS (SELECT 1 FROM photos WHERE id = ? AND landmark_id = ? AND status = ?"
               f" AND location_consent IN ({', '.join('?' * len(consents))}))"
               " ON CONFLICT(photo_id) DO UPDATE SET "
               + ", ".join(f"{n} = excluded.{n}" for n in names if n != "photo_id"))
        with self._lock:
            cur = self._conn.execute(sql, [*values, rec.photo_id, rec.landmark_id, PHOTO_ACCEPTED, *consents])
        return cur.rowcount > 0

    def localizations(self, landmark_id: str) -> dict[str, LocRecord]:
        """photo_id -> its localization, for the landmark's photos."""
        rows = self._select(LocRecord, "SELECT * FROM localizations WHERE landmark_id = ? ORDER BY photo_id",
                            [landmark_id])
        return {r.photo_id: r for r in rows}

    def get_localization(self, photo_id: str) -> LocRecord | None:
        rows = self._select(LocRecord, "SELECT * FROM localizations WHERE photo_id = ?", [photo_id])
        return rows[0] if rows else None

    # Deployment metadata (key -> text)
    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return None if row is None else row[0]

    def setdefault_meta(self, key: str, value: str) -> str:
        """Store value under key unless the key has one; returns the stored value."""
        with self._lock:
            self._conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES (?, ?)", (key, value))
            return self.get_meta(key)

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute("INSERT INTO meta (key, value) VALUES (?, ?) "
                               "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (key, value))

    # Jobs
    def create_job(self, job: Job) -> Job:
        self._insert("jobs", job)
        return self.get_job(job.id)

    def get_job(self, job_id: str) -> Job:
        return self._get(Job, "jobs", job_id)

    def update_job(self, job_id: str, **fields) -> Job:
        """Update status/started_at/finished_at/error/attempts; returns the new row."""
        unknown = set(fields) - _JOB_UPDATABLE
        if unknown:
            raise ValueError(f"cannot update job fields {sorted(unknown)}")
        if not fields:
            return self.get_job(job_id)
        assignments = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            sql = f"UPDATE jobs SET {assignments} WHERE id = ?"
            cur = self._conn.execute(sql, [*fields.values(), job_id])
            if cur.rowcount == 0:
                raise NotFound(job_id)
            return self.get_job(job_id)

    def list_jobs(self, landmark_id: str | None = None, statuses: tuple[str, ...] | None = None) -> list[Job]:
        """Ordered by created_at, then id."""
        where, args = [], []
        if landmark_id is not None:
            where.append("landmark_id = ?")
            args.append(landmark_id)
        if statuses is not None:
            if not statuses:
                return []
            where.append(f"status IN ({', '.join('?' * len(statuses))})")
            args.extend(statuses)
        sql = "SELECT * FROM jobs" + (" WHERE " + " AND ".join(where) if where else "")
        return self._select(Job, sql + " ORDER BY created_at, id", args)

    # Models
    def add_model(self, model: ModelRecord) -> ModelRecord:
        self._insert("models", model)
        return self._get(ModelRecord, "models", model.id)

    def get_model(self, model_id: str) -> ModelRecord:
        """Raises NotFound."""
        return self._get(ModelRecord, "models", model_id)

    def set_model_stylized(self, model_id: str, stylized: dict[str, Any] | None) -> ModelRecord:
        """Record (or clear) the model's stylized GLB; the only change a stored model ever gets."""
        with self._lock:
            cur = self._conn.execute("UPDATE models SET stylized = ? WHERE id = ?", (to_json(stylized), model_id))
            if cur.rowcount == 0:
                raise NotFound(model_id)
        return self.get_model(model_id)

    def list_models(self, landmark_id: str) -> list[ModelRecord]:
        """Ordered by created_at, then id."""
        return self._select(
            ModelRecord, "SELECT * FROM models WHERE landmark_id = ? ORDER BY created_at, id", [landmark_id]
        )

    def published_models(self, landmark_id: str) -> list[ModelRecord]:
        """list_models without the ones that failed the quality gate (meta["published"] is False)."""
        return [m for m in self.list_models(landmark_id) if m.meta.get("published", True)]

    def latest_models(self, landmark_id: str) -> list[ModelRecord]:
        """Newest published model per (kind, sector): full first, then faces by sector."""
        latest = {(m.kind, m.sector): m for m in self.published_models(landmark_id)}
        return sorted(latest.values(), key=lambda m: (m.sector is not None, m.sector or 0))

    # Row mapping
    def _insert(self, table: str, record: Any) -> None:
        names = [f.name for f in dataclasses.fields(record)]
        values = [to_json(getattr(record, n)) if n in _JSON_COLUMNS else getattr(record, n) for n in names]
        sql = f"INSERT INTO {table} ({', '.join(names)}) VALUES ({', '.join('?' * len(names))})"
        with self._lock:
            try:
                self._conn.execute(sql, values)
            except sqlite3.IntegrityError as e:
                if "UNIQUE" in str(e):
                    raise AlreadyExists(f"{table} {getattr(record, 'id', '')!r} already exists") from e
                raise

    def _get(self, cls: type[R], table: str, record_id: str) -> R:
        rows = self._select(cls, f"SELECT * FROM {table} WHERE id = ?", [record_id])
        if not rows:
            raise NotFound(record_id)
        return rows[0]

    def _select(self, cls: type[R], sql: str, args: list[Any] | tuple[Any, ...] = ()) -> list[R]:
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [cls(**{k: json.loads(r[k]) if k in _JSON_COLUMNS and r[k] is not None else r[k] for k in r.keys()})
                for r in rows]
