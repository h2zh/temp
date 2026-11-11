"""Read and write the EXIF fields the pipeline relies on.

Reading never raises on malformed EXIF: a missing or broken tag just becomes
None. Only an undecodable image raises UnreadableImage.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from fractions import Fraction

import piexif
from PIL import (  # noqa: F401 (registers formats)
    Image,
    JpegImagePlugin,
    PngImagePlugin,
    UnidentifiedImageError,
    WebPImagePlugin,
)

# Formats the pipeline decodes. Pillow would otherwise accept anything it knows, including EPS,
# which it renders by running Ghostscript on the uploaded PostScript.
ALLOWED_FORMATS = ("JPEG", "PNG", "WEBP")
# MPO: phone JPEGs with extra frames, which Pillow opens through its JPEG plugin. Not every JPEG is cheap to decode:
# only a sequential one with all its components in one scan decodes at 1/2..1/8 scale (draft) without buffers for
# the full image (PhotoMeta.decodes_reduced, jpeg_decodes_reduced).
JPEG_FORMATS = ("JPEG", "MPO")
# An EXIF text tag (camera make and model, the capture time, the heading's reference) is read up to this many
# characters: real ones are far shorter, and a crafted one can be megabytes (a PNG or WebP EXIF chunk has no size
# limit, and Pillow joins a JPEG's EXIF segments).
MAX_TEXT_CHARS = 64
_MAX_TEXT_BYTES = 1024      # of a text tag's bytes, what is decoded at all (leading padding and multi-byte characters)
_CAPTURE_TIME = re.compile(r"[0-9]{4}:[0-9]{2}:[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}")    # ASCII digits only
# JPEG segments a decoder skips by their length on the way to the first scan (APPn, COM, DQT, DHT, DRI, DAC, DNL)
_JPEG_SKIPPED = frozenset({*range(0xE0, 0xF0), 0xFE, 0xDB, 0xC4, 0xDD, 0xCC, 0xDC})
_JPEG_SEQUENTIAL = (0xC0, 0xC1)     # SOF0, SOF1: sequential DCT, Huffman coding
_JPEG_SOS = 0xDA
_JPEG_MAX_STEPS = 1000              # segments and fill bytes walked before the first scan (a camera's JPEG: dozens)


class UnreadableImage(ValueError):
    """The bytes are not an image Pillow can decode."""


@dataclass(frozen=True)
class PhotoMeta:
    width: int
    height: int
    lat: float | None = None
    lon: float | None = None
    alt: float | None = None
    gps_accuracy_m: float | None = None  # GPSHPositioningError: horizontal error the phone reported
    heading_deg: float | None = None     # GPSImgDirection: where the camera was pointing
    heading_ref: str | None = None       # "T" (true north) or "M" (magnetic north)
    focal_mm: float | None = None
    focal_35mm: float | None = None
    captured_at: str | None = None       # EXIF DateTimeOriginal as written (capture_time checks its form)
    make: str | None = None
    model: str | None = None
    orientation: int | None = None
    format: str | None = None            # Pillow format name, one of ALLOWED_FORMATS
    has_exif: bool = False               # a readable EXIF block was present at all
    decodes_reduced: bool = False        # a JPEG that decodes at 1/2..1/8 scale cheaply (jpeg_decodes_reduced)

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
    """An EXIF text tag: stripped, cut at MAX_TEXT_CHARS characters; None when missing, empty or not text. piexif
    returns a tag as the type its entry declares, so a text tag written as SHORT or RATIONAL arrives as an int or a
    tuple: a broken tag, not an error."""
    if isinstance(v, bytes):
        v = v[:_MAX_TEXT_BYTES].decode(errors="ignore")
    elif isinstance(v, str):
        v = v[:_MAX_TEXT_BYTES]
    else:
        return None
    v = v.strip("\x00 ").strip()[:MAX_TEXT_CHARS]
    return v or None


def capture_time(text: str | None) -> str | None:
    """text when it is exactly EXIF's date and time form, "YYYY:MM:DD HH:MM:SS" (19 ASCII characters), else None:
    the one form of a capture time the store keeps (ingest)."""
    return text if isinstance(text, str) and _CAPTURE_TIME.fullmatch(text) else None


def jpeg_decodes_reduced(data: bytes) -> bool:
    """Whether a JPEG's pixels can be decoded straight at 1/2..1/8 scale (Pillow's draft) without buffers for the full
    image: a sequential DCT image (SOF0 or SOF1) whose first scan holds every component. Any other JPEG costs memory
    by its pixel count whatever the scale asked for: a progressive one, and equally a sequential one that sends its
    components in separate scans (Pillow reports only the first as "progressive"), make libjpeg keep every coefficient
    of the full image (9000 x 9000: 290-525 MB against 104 MB), and a lossless one has no reduced decode. Read from
    the segments before the first scan as libjpeg reads them (each skipped by its length, 0xFF fill bytes between
    them); anything else there (junk, a second frame header, a marker without a length, more than _JPEG_MAX_STEPS
    segments and fill bytes) is answered False, the careful side: ingest then applies Settings.max_image_pixels."""
    if data[:2] != b"\xff\xd8":
        return False
    pos, end, components = 2, len(data), None
    for _ in range(_JPEG_MAX_STEPS):
        if pos + 4 > end or data[pos] != 0xFF:
            return False
        marker = data[pos + 1]
        if marker == 0xFF:          # a fill byte before the marker
            pos += 1
            continue
        length = int.from_bytes(data[pos + 2:pos + 4], "big")
        if length < 2:
            return False
        if marker == _JPEG_SOS:
            return components is not None and pos + 4 < end and data[pos + 4] == components
        if marker in _JPEG_SEQUENTIAL:
            if components is not None or length < 8 or pos + 9 >= end:
                return False
            components = data[pos + 9]      # after the length: precision (1), height (2), width (2), components (1)
        elif marker not in _JPEG_SKIPPED:
            return False
        pos += 2 + length
    return False


def read_photo_meta(data: bytes) -> PhotoMeta:
    try:
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as img:
            img.verify()
        with Image.open(io.BytesIO(data), formats=ALLOWED_FORMATS) as img:
            width, height = img.size
            fmt = img.format
            exif_bytes = img.info.get("exif")
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError) as exc:
        raise UnreadableImage(str(exc)) from exc

    reduced = fmt in JPEG_FORMATS and jpeg_decodes_reduced(data)
    if not exif_bytes:
        return PhotoMeta(width=width, height=height, format=fmt, decodes_reduced=reduced)
    try:
        ex = piexif.load(exif_bytes)
    except Exception:  # piexif raises a zoo of exception types on corrupt blobs
        return PhotoMeta(width=width, height=height, format=fmt, decodes_reduced=reduced)

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

    accuracy = None
    if piexif.GPSIFD.GPSHPositioningError in gps:
        accuracy = _rational_to_float(gps[piexif.GPSIFD.GPSHPositioningError])
        if accuracy is not None and not 0 < accuracy < 10_000:
            accuracy = None

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
        gps_accuracy_m=accuracy if lat is not None else None,
        heading_deg=heading,
        heading_ref=_text(gps.get(piexif.GPSIFD.GPSImgDirectionRef)),
        focal_mm=focal_mm,
        focal_35mm=focal_35,
        captured_at=_text(exif_ifd.get(piexif.ExifIFD.DateTimeOriginal)),
        make=_text(zeroth.get(piexif.ImageIFD.Make)),
        model=_text(zeroth.get(piexif.ImageIFD.Model)),
        orientation=orientation if isinstance(orientation, int) else None,
        format=fmt,
        has_exif=True,
        decodes_reduced=reduced,
    )


def build_exif_bytes(
    *,
    lat: float | None = None,
    lon: float | None = None,
    alt: float | None = None,
    gps_accuracy_m: float | None = None,
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
    if gps_accuracy_m is not None:
        gps[piexif.GPSIFD.GPSHPositioningError] = _to_rational(gps_accuracy_m, 100)
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
