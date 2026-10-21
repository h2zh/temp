"""Runtime settings. Every threshold the product logic depends on lives here."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_dir() -> Path:
    return Path(os.environ.get("CROWD3D_DATA_DIR", "data"))


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
    full_min_sectors: int = 6       # ... and this many sectors must be face-ready
    full_rebuild_growth: int = 8    # rebuild the full model once this many new photos arrive

    # Photo validation.
    default_radius_m: float = 150.0
    max_heading_offset_deg: float = 60.0   # compass may point at most this far away from the landmark
    min_image_side_px: int = 480
    blur_min_laplacian_var: float = 30.0
    duplicate_max_hamming: int = 4         # dHash distance at or below which a photo is a duplicate
    max_upload_bytes: int = 25 * 1024 * 1024

    # Reconstruction.
    processing_max_side_px: int = 1600     # processed copies fed to SfM are downscaled to this
    reconstructor: str = field(default_factory=lambda: os.environ.get("CROWD3D_RECONSTRUCTOR", "colmap"))
    camera_height_m: float = 1.6           # assumed eye height; phone GPS altitude is too noisy to use
    inline_jobs: bool = False              # run reconstruction synchronously inside the request (tests)

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

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.photos_dir, self.models_dir, self.work_dir):
            d.mkdir(parents=True, exist_ok=True)
