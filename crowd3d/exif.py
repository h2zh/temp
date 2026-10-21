"""Read and write the EXIF fields the pipeline relies on.

Reading never raises on malformed EXIF: a missing or broken tag just becomes
None. Only an undecodable image raises UnreadableImage.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from fractions import Fraction

import piexif
from PIL import Image, UnidentifiedImageError


class UnreadableImage(ValueError):
    """The bytes are not an image Pillow can decode."""


@dataclass(frozen=True)
class PhotoMeta:
    width: int
    height: int
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    heading_deg: float | None = None     # GPSImgDirection: where the camera was pointing
    heading_ref: str | None = None       # "T" (true north) or "M" (magnetic north)
    focal_mm: float | None = None
    focal_35mm: float | None = None
    captured_at: str | None = None       # EXIF DateTimeOriginal, "YYYY:MM:DD HH:MM:SS"
    make: str | None = None
    model: str | None = None
    orientation: int | None = None

    @property
    def has_gps(self) -> bool:
        return self.lat is not None and self.lon is not None


def _rational_to_float(r) -> float | None:
    try:
        num, den = r
        return None if den == 0 else num / den
    except (TypeError, ValueError):
        return None


def _to_rational(value: float, max_den: int = 1_000_000) -> tuple[int, int]:
    f = Fraction(value).limit_denominator(max_den)
    return f.numerator, f.denominator


def _dms_to_deg(dms, ref) -> float | None:
    try:
        d, m, s = (_rational_to_float(x) for x in dms)
    except (TypeError, ValueError):
        return None
    if d is None or m is None or s is None:
        return None
    deg = d + m / 60.0 + s / 3600.0
    if isinstance(ref, bytes):
        ref = ref.decode(errors="ignore")
    if ref in ("S", "W"):
        deg = -deg
    return deg


def _deg_to_dms(value: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    value = abs(value)
    d = int(value)
    m_full = (value - d) * 60.0
    m = int(m_full)
    s = (m_full - m) * 60.0
    return (d, 1), (m, 1), (int(round(s * 10000)), 10000)


def _text(v) -> str | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.decode(errors="ignore")
    v = v.strip("\x00 ").strip()
    return v or None


def read_photo_meta(data: bytes) -> PhotoMeta:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        with Image.open(io.BytesIO(data)) as img:
            width, height = img.size
            exif_bytes = img.info.get("exif")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError) as exc:
        raise UnreadableImage(str(exc)) from exc

    if not exif_bytes:
        return PhotoMeta(width=width, height=height)
    try:
        ex = piexif.load(exif_bytes)
    except Exception:  # piexif raises a zoo of exception types on corrupt blobs
        return PhotoMeta(width=width, height=height)

    gps = ex.get("GPS") or {}
    exif_ifd = ex.get("Exif") or {}
    zeroth = ex.get("0th") or {}

    lat = lon = None
    if piexif.GPSIFD.GPSLatitude in gps and piexif.GPSIFD.GPSLongitude in gps:
        lat = _dms_to_deg(gps[piexif.GPSIFD.GPSLatitude], gps.get(piexif.GPSIFD.GPSLatitudeRef))
        lon = _dms_to_deg(gps[piexif.GPSIFD.GPSLongitude], gps.get(piexif.GPSIFD.GPSLongitudeRef))
        if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
            lat = lon = None

    alt = None
    if piexif.GPSIFD.GPSAltitude in gps:
        alt = _rational_to_float(gps[piexif.GPSIFD.GPSAltitude])
        if alt is not None and gps.get(piexif.GPSIFD.GPSAltitudeRef) == 1:
            alt = -alt

    heading = None
    if piexif.GPSIFD.GPSImgDirection in gps:
        heading = _rational_to_float(gps[piexif.GPSIFD.GPSImgDirection])
        if heading is not None:
            heading %= 360.0

    focal_mm = None
    if piexif.ExifIFD.FocalLength in exif_ifd:
        focal_mm = _rational_to_float(exif_ifd[piexif.ExifIFD.FocalLength])
    focal_35 = exif_ifd.get(piexif.ExifIFD.FocalLengthIn35mmFilm)
    focal_35 = float(focal_35) if isinstance(focal_35, int) and focal_35 > 0 else None

    orientation = zeroth.get(piexif.ImageIFD.Orientation)
    return PhotoMeta(
        width=width,
        height=height,
        lat=lat,
        lon=lon,
        alt=alt,
        heading_deg=heading,
        heading_ref=_text(gps.get(piexif.GPSIFD.GPSImgDirectionRef)),
        focal_mm=focal_mm,
        focal_35mm=focal_35,
        captured_at=_text(exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)),
        make=_text(zeroth.get(piexif.ImageIFD.Make)),
        model=_text(zeroth.get(piexif.ImageIFD.Model)),
        orientation=orientation if isinstance(orientation, int) else None,
    )


def build_exif_bytes(
    *,
    lat: float | None = None,
    lon: float | None = None,
    alt: float | None = None,
    heading_deg: float | None = None,
    heading_ref: str = "T",
    focal_mm: float | None = None,
    focal_35mm: float | None = None,
    captured_at: str | None = None,
    make: str | None = "Crowd3D",
    model: str | None = "SyntheticCam",
) -> bytes:
    """Serialise an EXIF blob for Image.save(..., exif=...)."""
    zeroth: dict = {}
    exif_ifd: dict = {}
    gps: dict = {}
    if make:
        zeroth[piexif.ImageIFD.Make] = make.encode()
    if model:
        zeroth[piexif.ImageIFD.Model] = model.encode()
    if lat is not None and lon is not None:
        gps[piexif.GPSIFD.GPSVersionID] = (2, 3, 0, 0)
        gps[piexif.GPSIFD.GPSLatitudeRef] = b"N" if lat >= 0 else b"S"
        gps[piexif.GPSIFD.GPSLatitude] = _deg_to_dms(lat)
        gps[piexif.GPSIFD.GPSLongitudeRef] = b"E" if lon >= 0 else b"W"
        gps[piexif.GPSIFD.GPSLongitude] = _deg_to_dms(lon)
    if alt is not None:
        gps[piexif.GPSIFD.GPSAltitudeRef] = 0 if alt >= 0 else 1
        gps[piexif.GPSIFD.GPSAltitude] = _to_rational(abs(alt), 1000)
    if heading_deg is not None:
        gps[piexif.GPSIFD.GPSImgDirectionRef] = heading_ref.encode()
        gps[piexif.GPSIFD.GPSImgDirection] = _to_rational(heading_deg % 360.0, 100)
    if focal_mm is not None:
        exif_ifd[piexif.ExifIFD.FocalLength] = _to_rational(focal_mm, 100)
    if focal_35mm is not None:
        exif_ifd[piexif.ExifIFD.FocalLengthIn35mmFilm] = int(round(focal_35mm))
    if captured_at:
        exif_ifd[piexif.ExifIFD.DateTimeOriginal] = captured_at.encode()
    return piexif.dump({"0th": zeroth, "Exif": exif_ifd, "GPS": gps, "1st": {}, "thumbnail": None})
