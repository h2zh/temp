"""Plain records shared by the store, the coverage planner, the job runner and the API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

PHOTO_ACCEPTED = "accepted"
PHOTO_REJECTED = "rejected"

# The user agreement an upload is accepted under (D32): agreeing is required to upload, and it makes the photo's
# rounded pose and its downscaled, metadata-free photo public. Each version names one agreement text, web/terms.html
# (data-terms-version on both language versions); a change of substance in either language needs a new version.
# Uploads without the current version are refused (crowd3d.api, crowd3d.ingest.ingest_upload). TERMS_VERSIONS: the
# versions whose consent covers what is shown now.
TERMS_V1 = "terms-v1"
TERMS_CURRENT = TERMS_V1
TERMS_VERSIONS = frozenset({TERMS_V1})

JOB_FACE = "face"
JOB_FULL = "full"
JOB_STYLIZE = "stylize"       # a published model's stylized GLB (Job.model_id); the coverage planner never plans one
JOB_LOCALIZE = "localize"     # visual localization of consented photos no published model shows (crowd3d.localize,
#                               D30): display only, never an input of a reconstruction; the coverage planner never
#                               plans one
RECONSTRUCTION_KINDS = (JOB_FACE, JOB_FULL)
# jobs that change no reconstruction: the worker runs them only while no reconstruction job is queued (jobs.JobRunner)
BACKGROUND_KINDS = (JOB_STYLIZE, JOB_LOCALIZE)

JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"
ACTIVE_JOB_STATUSES = (JOB_QUEUED, JOB_RUNNING)

# a photo's visual localization (LocRecord.status)
LOC_ACCEPTED = "accepted"     # localized: LocRecord.pose is its camera's pose in the frame of LocRecord.model_id
LOC_REJECTED = "rejected"     # tried against the landmark's maps, and no pose passed the gates (LocRecord.reason)
LOC_SKIPPED = "skipped"       # not tried (LocRecord.reason: no EXIF focal length, a flagged SfM camera, no map, ...)
LOC_STATUSES = (LOC_ACCEPTED, LOC_REJECTED, LOC_SKIPPED)


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
    gps_accuracy_m: float | None = None  # EXIF GPSHPositioningError, when the phone wrote one
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
    # the qualification (crowd3d.subject, D31) at upload: None when it did not run (no GPS, no landmark, or a
    # landmark without a bound building outline)
    outline_distance_m: float | None = None   # camera GPS to the building's OSM main outline (0 inside)
    subject_share: float | None = None         # estimated share of the frame the building fills, 0..1
    heading_ref: str | None = None       # EXIF GPSImgDirectionRef of heading_deg: "T" true north, "M" magnetic, None
    #                                      when there is no heading or no such tag (rows stored before it: None)
    location_consent: str | None = None  # the TERMS_VERSIONS text of the user agreement an accepted photo was
    #                                      uploaded under (D32); None = withdrawn (withdrawing sets it back to None) or
    #                                      rejected


def location_shared(photo: PhotoRecord) -> bool:
    """Whether other people may see the photo's rounded pose and its downscaled, metadata-free photo (D30, D32): it
    was accepted, uploaded under the user agreement, and not withdrawn. The one rule for every reader: the photo-pose
    builder, the localizer and the public endpoint."""
    return photo.status == PHOTO_ACCEPTED and photo.location_consent in TERMS_VERSIONS


@dataclass
class Job:
    id: str
    landmark_id: str
    kind: str                            # JOB_FACE | JOB_FULL | JOB_STYLIZE | JOB_LOCALIZE
    sector: int | None                   # set for JOB_FACE, and for JOB_STYLIZE of a facade model
    status: str
    photo_ids: list[str]
    created_at: float
    started_at: float | None = None
    finished_at: float | None = None
    error: str | None = None
    attempts: int = 0                    # times a worker started it (a run that kills the process counts)
    model_id: str | None = None          # JOB_STYLIZE: the model it stylizes


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
    # its stylized GLB, once a stylize job made one: glb_path (relative to Settings.data_dir), registration
    # {accepted, reason}, what the structure step did (JobRunner._stylize)
    stylized: dict[str, Any] | None = None


@dataclass
class LocRecord:
    """One photo's visual localization (crowd3d.localize, D30): internal, exact, keyed by store photo id; other people
    see only photoposes.public_row of the pose it gives. The store keeps one per photo, and only for a photo whose
    uploader agreed to share its location (location_shared; Store.put_localization refuses any other, and withdrawing
    the consent deletes it)."""

    photo_id: str
    landmark_id: str
    status: str                          # LOC_STATUSES
    updated_at: float
    reason: str | None = None            # rejected / skipped: why (crowd3d.localize reason codes)
    maps_key: str | None = None          # the candidate maps it was tried against (localize.maps_key): tried again
    #                                      only when the landmark's candidate maps change
    attempts: int = 0                    # times the localizer tried it (a skip is no attempt)
    model_id: str | None = None          # accepted: the published model whose saved map it was localized into
    # accepted: {enu, heading_deg, pitch_deg, roll_deg, hfov_deg, aspect}: the camera in the landmark ENU frame of
    # model_id's GeoAnchor, before any footprint snap (photoposes moves it with the model's accepted snap, as it moves
    # the model's own cameras), hfov from the EXIF focal length the pose was solved with, aspect of the processed copy
    pose: dict[str, Any] | None = None
    stats: dict[str, Any] = field(default_factory=dict)   # diagnostics: per map tried, inliers, the GPS check, ...
