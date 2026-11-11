"""Decide whether an uploaded photo can contribute to a landmark's model."""

from __future__ import annotations

import io
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import params
from .config import Settings
from .coverage import sector_for_bearing
from .exif import ALLOWED_FORMATS, PhotoMeta, upright
from .geo import angle_diff_deg, local_bearing_distance
from .records import Landmark, PhotoRecord
from .subject import Subject, assess

# Rejection codes (all applicable codes are reported, not just the first one).
R_UNREADABLE = "unreadable_image"
R_NO_EXIF = "no_exif"                # no readable EXIF at all: a screenshot, an edited or messaging-app copy
R_NO_GPS = "no_gps"
R_NO_LANDMARK = "no_landmark"        # has GPS but no landmark covers that location
R_TOO_FAR = "too_far"                # outside the landmark's radius
R_FACING_AWAY = "facing_away"        # compass heading points away from the landmark
R_TOO_SMALL = "too_small"            # shorter side below Settings.min_image_side_px
R_TOO_BLURRY = "too_blurry"          # Laplacian variance below Settings.blur_min_laplacian_var
R_DUPLICATE = "duplicate"            # dHash within Settings.duplicate_max_hamming of an existing photo
R_NOT_SUBJECT = "not_subject"        # over Settings.subject_max_distance_m from the building's outline and the
                                     # building fills under Settings.subject_min_share of the frame (D31)

# Warning codes (photo is still accepted).
W_NO_HEADING = "no_heading"          # cannot check facing direction
W_MAGNETIC_HEADING = "magnetic_heading"  # GPSImgDirectionRef == "M"; declination ignored
W_SUBJECT_UNCHECKED = "subject_unchecked"  # no bound building outline: not_subject was not checked
W_HEADING_AWAY = "heading_away_unverified"  # Settings.facing_away_warns_other_refs: a heading not tagged true north
                                            # points away from the landmark; kept, the heading is not believed
                                            # (believed_heading)

# Sharpness and dHash are measured on a grayscale copy whose longer side is at most
# Params.analysis_max_side_px, so Settings.blur_min_laplacian_var means the same for any upload size.


@dataclass
class ValidationResult:
    accepted: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    bearing_deg: float | None = None     # landmark centre -> camera
    distance_m: float | None = None
    sector: int | None = None
    phash: str | None = None
    outline_distance_m: float | None = None   # the qualification (crowd3d.subject), when it ran
    subject_share: float | None = None


def laplacian_variance(gray: np.ndarray) -> float:
    """Sharpness score of a 2-D uint8/float image (variance of the Laplacian).

    4-neighbour kernel over interior pixels (same as cv2.Laplacian with ksize=1, minus the border).
    """
    g = np.asarray(gray, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {g.shape}")
    if min(g.shape) < 3:
        return 0.0
    lap = g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
    return float(lap.var())


def dhash_hex(image: Image.Image, hash_size: int = 8) -> str:
    """64-bit difference hash as 16 lowercase hex chars.

    Grayscale, resized to (hash_size + 1) x hash_size; a bit is 1 where a pixel is
    brighter than its left neighbour, row-major with the first bit most significant
    (the imagehash.dhash convention).
    """
    small = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.LANCZOS)
    px = np.asarray(small, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).ravel()
    value = int("".join("1" if b else "0" for b in bits), 2)
    return f"{value:0{(bits.size + 3) // 4}x}"


def hamming_hex(a: str, b: str) -> int:
    """Bit distance between two equal-length hex hashes."""
    if len(a) != len(b):
        raise ValueError(f"hash lengths differ: {len(a)} != {len(b)}")
    return (int(a, 16) ^ int(b, 16)).bit_count()


def _analysis_gray(image_bytes: bytes) -> Image.Image:
    """Upright grayscale copy with its longer side at most Params.analysis_max_side_px (never upscaled)."""
    max_side = params.current().analysis_max_side_px
    with Image.open(io.BytesIO(image_bytes), formats=ALLOWED_FORMATS) as img:
        longer = max(img.size)
        if longer > max_side:
            # A sequential JPEG decodes straight at 1/2, 1/4 or 1/8 scale, far cheaper than a full 12 MP decode
            # (any other image decodes in full: ingest caps those at Settings.max_image_pixels).
            s = max_side / longer
            img.draft("L", (math.ceil(img.width * s), math.ceil(img.height * s)))
        gray = upright(img).convert("L")
    longer = max(gray.size)
    if longer > max_side:
        s = max_side / longer
        size = (max(1, round(gray.width * s)), max(1, round(gray.height * s)))
        gray = gray.resize(size, Image.Resampling.LANCZOS)
    return gray


def believed_heading(photo: PhotoRecord) -> float | None:
    """A stored photo's compass heading as a reconstruction may use it (ReconPhoto.heading_deg): None when the photo
    has none, or when validation kept the photo but did not believe its heading (W_HEADING_AWAY: it points away from
    the landmark and is not tagged true north). Such a photo counts as one without a compass: no reading for the yaw
    prior or the camera checks, and the quality gate checks the direction of its estimate (qa.no_checked_compass)."""
    return None if W_HEADING_AWAY in (getattr(photo, "warnings", None) or ()) else photo.heading_deg


def validate_photo(
    image_bytes: bytes,
    meta: PhotoMeta | None,
    landmark: Landmark | None,
    existing_hashes: Iterable[str],
    settings: Settings,
    subject: Subject | None = None,
) -> ValidationResult:
    """Run every check and collect all failing codes.

    meta is None when the image could not be decoded (-> R_UNREADABLE only).
    If meta was read but the pixels cannot be decoded (e.g. a truncated JPEG),
    R_UNREADABLE is reported alongside the metadata checks.
    The facing check: the camera must look roughly toward the landmark, i.e.
    |angle_diff(heading, bearing(camera -> landmark centre))| <= max_heading_offset_deg. With
    Settings.facing_away_warns_other_refs only a heading tagged true north (heading_ref "T")
    rejects; any other one that points away gives the warning W_HEADING_AWAY.
    The sector is coverage.sector_for_bearing(bearing landmark -> camera).
    Bearing and distance come from geo.local_bearing_distance (ENU-consistent).
    Geometry fields are filled whenever GPS and a landmark are available, even
    for rejected photos. Without GPS no landmark checks run.
    Heading warnings depend on the metadata only.

    The qualification (D31, crowd3d.subject.assess) runs with GPS and a landmark:
    subject is the landmark's bound building (subject.load_subject); a photo that
    is over Settings.subject_max_distance_m from its outline and fills under
    Settings.subject_min_share of the frame is R_NOT_SUBJECT. Without a subject
    (no bound outline) the rule is skipped and the photo carries
    W_SUBJECT_UNCHECKED. The compass plays no part in it.

    Blur and dHash are measured on the EXIF-upright grayscale image downscaled
    (Lanczos, never upscaled) so its longer side is at most Params.analysis_max_side_px.
    Raw Laplacian variance of the same scene drops as resolution grows (edges
    spread over more pixels: about 4x from 800 px to 3200 px), so this
    normalisation is what lets one blur_min_laplacian_var threshold work for a
    640 px and a 12 MP upload alike.
    """
    if meta is None:
        return ValidationResult(accepted=False, reasons=[R_UNREADABLE])
    try:
        gray: Image.Image | None = _analysis_gray(image_bytes)
    except Exception:   # whatever a decoder raises on hostile or broken input, the photo is unreadable
        gray = None

    result = ValidationResult(accepted=False)
    reasons, warnings = result.reasons, result.warnings
    if gray is None:
        reasons.append(R_UNREADABLE)

    # Uploads must be camera originals with their EXIF and GPS (product rule).
    if not meta.has_exif:
        reasons.append(R_NO_EXIF)
    if not meta.has_gps:
        reasons.append(R_NO_GPS)
    elif landmark is None:
        reasons.append(R_NO_LANDMARK)
    else:
        bearing, distance = local_bearing_distance(landmark.lat, landmark.lon, meta.lat, meta.lon)
        result.bearing_deg, result.distance_m = bearing, distance
        result.sector = sector_for_bearing(bearing, settings.num_sectors)
        if distance > landmark.radius_m:
            reasons.append(R_TOO_FAR)
        if meta.heading_deg is not None:
            offset = angle_diff_deg(meta.heading_deg, bearing + 180.0)
            if abs(offset) > settings.max_heading_offset_deg:
                if settings.facing_away_warns_other_refs and (meta.heading_ref or "").strip().upper() != "T":
                    warnings.append(W_HEADING_AWAY)
                else:
                    reasons.append(R_FACING_AWAY)
        if subject is not None:
            q = assess(subject, meta, settings)
            result.outline_distance_m, result.subject_share = q.outline_distance_m, q.share
            if not q.qualifies:
                reasons.append(R_NOT_SUBJECT)

    if meta.heading_deg is None:
        warnings.append(W_NO_HEADING)
    elif meta.heading_ref == "M":
        warnings.append(W_MAGNETIC_HEADING)
    if meta.has_gps and landmark is not None and subject is None:
        warnings.append(W_SUBJECT_UNCHECKED)

    if min(meta.width, meta.height) < settings.min_image_side_px:
        reasons.append(R_TOO_SMALL)
    if gray is not None:
        if laplacian_variance(np.asarray(gray)) < settings.blur_min_laplacian_var:
            reasons.append(R_TOO_BLURRY)
        result.phash = dhash_hex(gray)
        # Rejected rows may carry no hash; skip them rather than fail the upload.
        if any(h and hamming_hex(result.phash, h) <= settings.duplicate_max_hamming for h in existing_hashes):
            reasons.append(R_DUPLICATE)

    result.accepted = not reasons
    return result
