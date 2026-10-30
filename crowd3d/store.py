"""SQLite persistence for landmarks, photos, jobs and models.

Thread-safe: the API thread and the job worker thread share one Store.
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
import threading
from typing import Any, TypeVar

from .config import Settings
from .geo import haversine_m, local_bearing_distance
from .records import PHOTO_ACCEPTED, Job, Landmark, ModelRecord, PhotoRecord


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
    gps_accuracy_m REAL
);
CREATE INDEX IF NOT EXISTS photos_landmark ON photos(landmark_id, status);
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
"""

# Columns added after the first schema: databases created before get them on open.
_ADDED_COLUMNS = {
    "photos": [("gps_accuracy_m", "REAL")],
    "jobs": [("attempts", "INTEGER NOT NULL DEFAULT 0"), ("model_id", "TEXT")],
    "models": [("stylized", "TEXT")],
}

# Columns holding lists/dicts, stored as JSON text (SQL NULL, in a column added later, reads as None).
_JSON_COLUMNS = {"reasons", "warnings", "photo_ids", "meta", "stylized"}
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
        # Autocommit; every public method is a single statement or a read.
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

    def contributors(self, landmark_id: str) -> list[tuple[str, int]]:
        """(user_id, accepted photo count), most photos first, then user_id."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT user_id, COUNT(*) AS n FROM photos WHERE landmark_id = ? AND status = ?"
                " GROUP BY user_id ORDER BY n DESC, user_id",
                (landmark_id, PHOTO_ACCEPTED),
            ).fetchall()
        return [(r[0], r[1]) for r in rows]

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
