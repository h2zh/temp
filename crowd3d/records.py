"""Plain records shared by the store, the coverage planner, the job runner and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PHOTO_ACCEPTED = "accepted"
PHOTO_REJECTED = "rejected"

JOB_FACE = "face"
JOB_FULL = "full"

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"
ACTIVE_JOB_STATUSES = (JOB_QUEUED, JOB_RUNNING)


@dataclass
class Landmark:
    id: str
    name: str
    lat: float
    lon: float
    radius_m: float
    created_at: float


@dataclass
class PhotoRecord:
    id: str
    landmark_id: str | None
    user_id: str
    status: str                          # PHOTO_ACCEPTED | PHOTO_REJECTED
    uploaded_at: float
    reasons: list[str] = field(default_factory=list)   # rejection codes
    warnings: list[str] = field(default_factory=list)
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    heading_deg: float | None = None     # compass heading from EXIF
    bearing_deg: float | None = None     # bearing from the landmark centre to the camera
    distance_m: float | None = None
    sector: int | None = None
    width: int | None = None
    height: int | None = None
    focal_35mm: float | None = None
    phash: str | None = None             # 16 hex chars (64-bit dHash)
    original_path: str | None = None     # relative to Settings.data_dir
    processed_path: str | None = None    # relative to Settings.data_dir
    captured_at: str | None = None


@dataclass
class Job:
    id: str
    landmark_id: str
    kind: str                            # JOB_FACE | JOB_FULL
    sector: int | None                   # set for JOB_FACE only
    status: str
    photo_ids: list[str]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None


@dataclass
class ModelRecord:
    id: str
    landmark_id: str
    job_id: str
    kind: str
    sector: int | None
    ply_path: str                        # relative to Settings.data_dir
    num_photos: int
    meta: dict[str, Any]                 # ReconstructionResult.to_meta()
    created_at: float
