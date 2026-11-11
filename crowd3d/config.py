"""Runtime settings. Every threshold the product logic depends on lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(os.environ.get("CROWD3D_DATA_DIR", "data"))


REPO_ROOT = Path(__file__).resolve().parent.parent


def _default_structure_catalog() -> Path | None:
    """The landmark catalogue with the OSM parts snapshots (eval/landmarks.json); $CROWD3D_STRUCTURE_CATALOG names
    another one, and an empty value turns the stylize step off (None)."""
    value = os.environ.get("CROWD3D_STRUCTURE_CATALOG")
    if value is None:
        return REPO_ROOT / "eval" / "landmarks.json"
    return Path(value) if value.strip() else None


def _default_keep_work_dirs() -> bool:
    """$CROWD3D_KEEP_WORK_DIRS = 1 / true / yes / on keeps every job's work dir (off without it): the switch an
    operator has for looking into a failed job without changing code (D34)."""
    return os.environ.get("CROWD3D_KEEP_WORK_DIRS", "").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    data_dir: Path = field(default_factory=_default_data_dir)

    # Coverage model: the ring around a landmark is split into equal sectors,
    # sector 0 is centred on north and indices increase clockwise.
    num_sectors: int = 8
    face_min_photos: int = 4        # accepted photos in a sector before its facade model is built
    face_rebuild_growth: int = 3    # rebuild a facade model once this many new photos arrive
    face_margin_deg: float = 10.0   # facade jobs also use photos this far into neighbouring sectors
    full_min_photos: int = 24       # accepted photos before the full model is built ...
    full_min_sectors: int = 6       # ... and this many sectors must be face-ready (all sides reachable)
    full_reachability_min_photos: int = 48   # with this many photos, sides still without any count as unreachable
    full_rebuild_growth: int = 8    # rebuild the full model once this many new photos arrive

    # Photo validation.
    default_radius_m: float = 150.0
    max_heading_offset_deg: float = 60.0   # compass may point at most this far away from the landmark
    min_image_side_px: int = 480
    blur_min_laplacian_var: float = 30.0
    duplicate_max_hamming: int = 4         # dHash distance at or below which a photo is a duplicate
    max_upload_bytes: int = 25 * 1024 * 1024
    # The pixel cap of everything that decodes at full cost (too_large above it): PNG and WebP (a 2 MB, 144 MP PNG
    # takes 1.2 GB of RAM), and a JPEG that is progressive, sends its components in separate scans or is lossless
    # (the decoder buffers the full image whatever scale is asked for: 9000 x 9000 took 290-525 MB). Only a
    # sequential single-scan JPEG is exempt, which really decodes at 1/8 scale (104 MB for 9000 x 9000): what phones
    # write, some of them at 108 MP (exif.jpeg_decodes_reduced; D34).
    max_image_pixels: int = 50_000_000
    # Uploads Pillow works on at once, in the whole process (ingest; D34): reading an upload's header and EXIF, and
    # decoding its pixels; the others wait. One decode just under the pixel cap peaks at about 0.9 GB for a WebP,
    # 0.5 GB for a PNG and 0.33 GB for a progressive JPEG (measured at 49 MP), so this bounds decoding at about
    # 2 GB whatever the number of requests in flight. It holds in a server only with malloc_mmap_threshold_bytes
    # below: measured on one with 8, 16 and 40 uploads at once, 0.6 GB for PNG and 1.6 GB for WebP at the peak,
    # 0.1 GB when idle again. (An upload waiting its turn holds its file's bytes, 25 MiB at most.)
    max_concurrent_decodes: int = 2
    # Over 0: the server keeps glibc's allocator in hand from its start (ingest.pin_malloc_threshold; D34), in two
    # ways. The mmap threshold is pinned at this value: a block of this size or more is a mapping of its own, given
    # back to the operating system the moment it is freed. Left alone, glibc raises the threshold to the size of
    # the first big block freed (up to 32 MiB) and from then on keeps Pillow's 16 MB image blocks in a heap per
    # worker thread, so a server's memory grew with the number of threads that had ever decoded, not with
    # max_concurrent_decodes: 8 uploads of a 49 MP PNG at once took 2.5 GB, 3.4 GB after two more rounds, and the
    # idle server kept them. And every decode slot ends by handing freed memory back (malloc_trim), for the smaller
    # blocks, which the threads' heaps keep as well: 16 and then 40 uploads at once of a 69 KB PNG with 64 MiB of
    # compressed text left an idle server at 2.3 GB. 0 leaves the allocator alone, as does a C library other than glibc;
    # with a MALLOC_MMAP_THRESHOLD_ in the environment (glibc's own switch) that threshold stays and only the
    # handing back is added.
    malloc_mmap_threshold_bytes: int = 4 * 1024 * 1024
    # The building must be the photo's subject (D31, crowd3d.subject; not_subject): the camera's GPS within this
    # distance of the OSM main outline ...
    subject_max_distance_m: float = 50.0
    # ... or the building estimated to fill at least this share of the frame (width or height; GPS, OSM parts and
    # the field of view, no compass). 1/3 rather than 1/2: checked against reference camera poses, the estimate runs
    # low, and 1/3 comes closest to the building really filling half the frame.
    subject_min_share: float = 0.33
    # The field of view of a photo whose FocalLengthIn35mmFilm is missing or outside the range: the median of the
    # known values in the Commons test set.
    subject_default_focal_35mm: float = 28.0
    subject_min_focal_35mm: float = 10.0
    subject_max_focal_35mm: float = 1200.0

    # Reconstruction.
    processing_max_side_px: int = 1600     # processed copies fed to SfM are downscaled to this
    reconstructor: str = field(default_factory=lambda: os.environ.get("CROWD3D_RECONSTRUCTOR", "colmap"))
    camera_height_m: float = 1.6           # assumed eye height; phone GPS altitude is too noisy to use
    inline_jobs: bool = False              # run reconstruction synchronously inside the request (tests)
    # keep each job's work dir: a reconstruction's COLMAP database and models (~1.5 MB per photo), a stylize job's
    # spec, child.log and output. Default: $CROWD3D_KEEP_WORK_DIRS (off without it)
    keep_work_dirs: bool = field(default_factory=_default_keep_work_dirs)
    max_job_attempts: int = 2              # a job the process died in this often is failed, not re-run

    # Structure in the product (crowd3d.structure_pipeline): each published model of a landmark with an OSM parts
    # snapshot gets a stylized GLB (clay OSM-parts massing, D29; photo evidence -- D15, D24, D25 -- only when the
    # photos' registration is accepted). A stylize job runs in a child process, only while no reconstruction job is
    # queued.
    stylize_published: bool = True
    structure_catalog: Path | None = field(default_factory=_default_structure_catalog)
    # landmark id -> catalogue slug, for a landmark whose id is not its slug (the catalogue's centre must still lie
    # within the landmark's radius)
    structure_bindings: dict[str, str] = field(default_factory=dict)
    stylize_timeout_s: float = 1200.0      # the child process is killed after this; the model keeps no GLB from it

    # Visual localization of consented photos no published model shows (D30, crowd3d.localize; jobs.JobRunner): PnP
    # into the landmark's saved maps, display only (source "loc" of the public photo poses), never an input of a
    # reconstruction. A localize job runs only while no reconstruction job is queued and stops between two photos as
    # soon as one is. Off: the D30 verification (eval/criteria/d30-v1.json, docs/BENCHMARK.md 6.12) failed its cost
    # rule on both commons-ten-v1 splits as written (localization CPU 0.267 and 0.665 of the run's, at most 0.25) with
    # every accuracy and privacy guard passing; switching it on is a new decision. Only the held-out (test) failure is
    # robust: the run CPU came from the D31 runs a day earlier, and on the day the localization CPU was measured the
    # same work took about 1.5 times the CPU, so dev is about 0.17-0.19 and test about 0.41-0.46 without that drift.
    # Off also withholds every loc pose stored before (jobs.shown_localizations): switching it off withdraws them
    # without a code change.
    localize_photos: bool = False
    localize_batch: int = 50               # photos one localize job tries at most; the rest are left to the next one

    # GeoAnchor calibrations (Params.anchor_calibration_id) that passed the held-out check (docs/BENCHMARK.md 7.8):
    # only for these does ModelInfo.geo say "calibrated" and the landmark page show "位置 ±X 米（1σ）". Empty (off)
    # until the final evaluation passes one; a refit (a new id) is off again until it passes too.
    calibrated_anchor_ids: tuple[str, ...] = ()

    # Quality gate before a model replaces the published one (qa.assess).
    qa_min_registered_fraction: float = 0.6       # facade models
    qa_full_min_registered_fraction: float = 0.7  # the full model: a fragment of one side is not "full"
    # D31: a facade model below the fraction still passes registration with at least this many registered photos when
    # its georegistration fitted GPS (ReconstructionResult.georegistered; every other check unchanged). Measured on the
    # F runs: such models got 78.4 % of poses right, F's published facades 74.5 %; 6-10 behave alike, higher only drops
    # good small models. Not for the full model.
    qa_face_min_registered: int = 6
    qa_min_points: int = 300
    qa_max_reprojection_error_px: float = 2.5
    qa_max_twin_fraction: float = 0.1      # cameras off both their GPS and compass (look-alike facades)
    qa_max_flagged_fraction: float = 0.2   # cameras with any implausibility flag
    qa_max_compass_error_deg: float = 25.0

    @property
    def db_path(self) -> Path:
        return self.data_dir / "crowd3d.sqlite3"

    @property
    def photos_dir(self) -> Path:
        return self.data_dir / "photos"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def work_dir(self) -> Path:
        return self.data_dir / "work"

    @property
    def maps_dir(self) -> Path:
        """Each landmark's persistent map (crowd3d.landmark_map), outside the served models dir."""
        return self.data_dir / "maps"

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.photos_dir, self.models_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
