import io
import math
import random
import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import piexif
import pytest
from PIL import Image, ImageDraw, ImageFilter

from crowd3d import validation as v
from crowd3d.config import Settings
from crowd3d.exif import build_exif_bytes, read_photo_meta
from crowd3d.geo import LocalENU, normalize_deg
from crowd3d.records import Landmark
from crowd3d.subject import Subject

LM = Landmark(id="tower", name="Tower", lat=31.2397, lon=121.4998, radius_m=150.0, created_at=0.0)


def _draw_scene(seed: int, size: tuple[int, int]) -> Image.Image:
    """Random coloured rectangles and ellipses in resolution-independent coordinates."""
    rng = random.Random(seed)
    w, h = size
    img = Image.new("RGB", size, (128, 128, 128))
    draw = ImageDraw.Draw(img)
    for _ in range(120):
        x0, y0 = rng.random(), rng.random()
        x1, y1 = x0 + 0.2 * rng.random(), y0 + 0.2 * rng.random()
        fill = tuple(rng.randrange(256) for _ in range(3))
        shape = draw.rectangle if rng.random() < 0.5 else draw.ellipse
        shape([x0 * w, y0 * h, x1 * w, y1 * h], fill=fill)
    return img


@lru_cache
def _scene(seed: int, size: tuple[int, int] = (640, 480)) -> Image.Image:
    """Sharp procedural photo, 2x supersampled so edges look like camera edges rather than aliased steps."""
    big = _draw_scene(seed, (size[0] * 2, size[1] * 2))
    return big.resize(size, Image.Resampling.BOX)


def _at(bearing: float, distance: float, heading_offset: float | None = 0.0, **extra) -> dict:
    """EXIF kwargs for a camera `distance` m from LM at `bearing`, looking `heading_offset` deg off the centre."""
    rad = math.radians(bearing)
    lat, lon, _ = LocalENU(LM.lat, LM.lon).to_geodetic(distance * math.sin(rad), distance * math.cos(rad))
    heading = None if heading_offset is None else normalize_deg(bearing + 180.0 + heading_offset)
    return dict(lat=lat, lon=lon, heading_deg=heading, **extra)


def _jpeg(img: Image.Image, quality: int = 90, **exif) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, exif=build_exif_bytes(**exif))
    return buf.getvalue()


def _check(data: bytes, landmark=LM, existing=(), settings: Settings | None = None) -> v.ValidationResult:
    return v.validate_photo(data, read_photo_meta(data), landmark, existing, settings or Settings())


# --- primitives -----------------------------------------------------------------------------------------


def test_laplacian_variance_known_values():
    assert v.laplacian_variance(np.full((20, 30), 77, np.uint8)) == 0.0
    checker = (np.indices((16, 16)).sum(axis=0) % 2 * 255).astype(np.uint8)
    # Every interior pixel's Laplacian is +-4 * 255.
    assert v.laplacian_variance(checker) == pytest.approx(1020.0**2)
    assert v.laplacian_variance(checker.astype(np.float32)) == pytest.approx(1020.0**2)
    assert v.laplacian_variance(np.zeros((2, 50))) == 0.0
    with pytest.raises(ValueError):
        v.laplacian_variance(np.zeros((8, 8, 3)))


def test_dhash_format_and_convention():
    ramp = Image.fromarray(np.tile(np.linspace(0, 255, 90), (80, 1)).astype(np.uint8))
    assert v.dhash_hex(ramp) == "f" * 16          # brighter than the left neighbour -> 1
    assert v.dhash_hex(ramp.transpose(Image.Transpose.FLIP_LEFT_RIGHT)) == "0" * 16
    assert v.dhash_hex(ramp, hash_size=4) == "ffff"
    h = v.dhash_hex(_scene(1))
    assert len(h) == 16 and h == h.lower() and int(h, 16) >= 0
    assert v.dhash_hex(_scene(1).convert("L")) == h


def test_hamming_hex():
    assert v.hamming_hex("0" * 16, "f" * 16) == 64
    assert v.hamming_hex("00ff", "00ff") == 0
    assert v.hamming_hex("0001", "0003") == 1
    with pytest.raises(ValueError):
        v.hamming_hex("00", "000")


# --- validate_photo -------------------------------------------------------------------------------------


def test_good_photo_is_accepted_with_geometry():
    r = _check(_jpeg(_scene(1), **_at(200.0, 40.0, heading_offset=10.0)))
    # LM has no bound building outline here: the subject rule (D31) is skipped, with a warning
    assert r.accepted and r.reasons == [] and r.warnings == [v.W_SUBJECT_UNCHECKED]
    assert r.outline_distance_m is r.subject_share is None
    assert r.bearing_deg == pytest.approx(200.0, abs=0.01)
    assert r.distance_m == pytest.approx(40.0, abs=0.01)
    assert r.sector == 4
    assert r.phash == v.dhash_hex(_scene(1))


@pytest.mark.parametrize(
    "bearing, sector", [(0.0, 0), (350.0, 0), (21.0, 0), (24.0, 1), (95.0, 2), (181.0, 4), (260.0, 6), (315.0, 7)]
)
def test_sector_is_the_side_the_photographer_stands_on(bearing, sector):
    r = _check(_jpeg(_scene(1), **_at(bearing, 60.0)))
    assert r.accepted
    assert r.bearing_deg == pytest.approx(bearing, abs=0.01)
    assert r.sector == sector


def test_sector_uses_settings_num_sectors():
    r = _check(_jpeg(_scene(1), **_at(100.0, 60.0)), settings=Settings(num_sectors=4))
    assert r.sector == 1


def test_unreadable_without_meta_reports_only_that():
    r = v.validate_photo(b"definitely not a jpeg", None, LM, ["0" * 16], Settings())
    assert not r.accepted
    assert r.reasons == [v.R_UNREADABLE] and r.warnings == []
    assert r.bearing_deg is r.distance_m is r.sector is r.phash is None


def test_truncated_pixels_are_unreadable_but_metadata_is_still_checked():
    data = _jpeg(_scene(1), **_at(30.0, 300.0))[:4000]
    meta = read_photo_meta(data)          # header and EXIF survive the truncation
    r = v.validate_photo(data, meta, LM, [], Settings())
    assert r.reasons == [v.R_UNREADABLE, v.R_TOO_FAR]
    assert r.phash is None
    assert r.distance_m == pytest.approx(300.0, abs=0.01)


def test_a_photo_without_any_exif_is_rejected_as_not_a_camera_original():
    # Screenshots, edited copies and photos sent through messaging apps lose their EXIF.
    buf = io.BytesIO()
    _scene(1).save(buf, "JPEG", quality=90)
    r = _check(buf.getvalue())
    assert r.reasons == [v.R_NO_EXIF, v.R_NO_GPS]
    assert not r.accepted


def test_no_gps_skips_landmark_checks():
    r = _check(_jpeg(_scene(1), heading_deg=123.0))
    assert r.reasons == [v.R_NO_GPS] and r.warnings == []
    assert r.bearing_deg is r.distance_m is r.sector is None
    assert r.phash is not None


def test_heading_warnings_do_not_depend_on_gps():
    r = _check(_jpeg(_scene(1)))
    assert r.reasons == [v.R_NO_GPS]
    assert r.warnings == [v.W_NO_HEADING]


def test_gps_without_landmark():
    r = _check(_jpeg(_scene(1), **_at(10.0, 30.0)), landmark=None)
    assert r.reasons == [v.R_NO_LANDMARK]
    assert r.bearing_deg is r.distance_m is r.sector is None


def test_too_far_uses_the_landmark_radius_and_keeps_geometry():
    data = _jpeg(_scene(1), **_at(80.0, 120.0))
    assert _check(data).accepted
    r = _check(data, landmark=replace(LM, radius_m=100.0))
    assert r.reasons == [v.R_TOO_FAR]
    assert r.distance_m == pytest.approx(120.0, abs=0.01)
    assert r.bearing_deg == pytest.approx(80.0, abs=0.01)
    assert r.sector == 2


@pytest.mark.parametrize("offset, ok", [(0.0, True), (59.5, True), (-59.5, True), (60.5, False), (-60.5, False), (180.0, False)])
def test_facing_threshold(offset, ok):
    r = _check(_jpeg(_scene(1), **_at(135.0, 50.0, heading_offset=offset)))
    assert r.accepted is ok
    assert r.reasons == ([] if ok else [v.R_FACING_AWAY])
    assert r.sector == 3 and r.bearing_deg == pytest.approx(135.0, abs=0.01)


@pytest.mark.parametrize("bearing, heading", [(190.0, 350.0), (170.0, 10.0)])
def test_facing_check_wraps_around_north(bearing, heading):
    # Camera -> landmark is 10 (resp. 350) deg; the compass reads 350 (resp. 10): 20 deg apart, not 340.
    data = _jpeg(_scene(1), **{**_at(bearing, 50.0), "heading_deg": heading})
    assert _check(data, settings=Settings(max_heading_offset_deg=25.0)).accepted
    assert _check(data, settings=Settings(max_heading_offset_deg=15.0)).reasons == [v.R_FACING_AWAY]


def test_missing_heading_is_a_warning_only():
    r = _check(_jpeg(_scene(1), **_at(300.0, 50.0, heading_offset=None)))
    assert r.accepted
    assert r.warnings == [v.W_NO_HEADING, v.W_SUBJECT_UNCHECKED]
    assert r.sector == 7


def test_magnetic_heading_warns_and_is_still_checked():
    r = _check(_jpeg(_scene(1), **_at(45.0, 50.0, heading_offset=5.0, heading_ref="M")))
    assert r.accepted and r.warnings == [v.W_MAGNETIC_HEADING, v.W_SUBJECT_UNCHECKED]
    r = _check(_jpeg(_scene(1), **_at(45.0, 50.0, heading_offset=170.0, heading_ref="M")))
    assert r.reasons == [v.R_FACING_AWAY] and r.warnings == [v.W_MAGNETIC_HEADING, v.W_SUBJECT_UNCHECKED]


def test_too_small_checks_the_shorter_side():
    s = Settings()
    portrait = _scene(2, (s.min_image_side_px, 900))
    assert _check(_jpeg(portrait, **_at(0.0, 40.0))).accepted
    r = _check(_jpeg(_scene(2, (900, s.min_image_side_px - 1)), **_at(0.0, 40.0)))
    assert r.reasons == [v.R_TOO_SMALL]


def test_blurry_photo_is_rejected():
    sharp = _scene(3)
    assert _check(_jpeg(sharp, **_at(0.0, 40.0))).accepted
    r = _check(_jpeg(sharp.filter(ImageFilter.GaussianBlur(3)), **_at(0.0, 40.0)))
    assert r.reasons == [v.R_TOO_BLURRY]
    assert r.sector == 0 and r.distance_m == pytest.approx(40.0, abs=0.01)


def test_blur_threshold_is_resolution_normalised():
    # The same slightly soft scene (lens blur) from a 2400 px and an 800 px camera.
    big = _draw_scene(4, (2400, 1800)).filter(ImageFilter.GaussianBlur(1.0))
    small = big.resize((800, 600), Image.Resampling.BOX)
    threshold = Settings().blur_min_laplacian_var
    native_big = v.laplacian_variance(np.asarray(big.convert("L")))
    native_small = v.laplacian_variance(np.asarray(small.convert("L")))
    assert native_big < threshold < native_small / 2     # raw scores depend heavily on resolution

    big_jpeg, small_jpeg = _jpeg(big, **_at(0.0, 40.0)), _jpeg(small, **_at(0.0, 40.0))
    analysed = v._analysis_gray(big_jpeg)
    assert analysed.size == (800, 600)
    assert v._analysis_gray(_jpeg(_scene(1))).size == (640, 480)    # never upscaled
    norm_big = v.laplacian_variance(np.asarray(analysed))
    norm_small = v.laplacian_variance(np.asarray(v._analysis_gray(small_jpeg)))
    assert norm_big == pytest.approx(norm_small, rel=0.25)
    assert _check(big_jpeg).accepted
    assert _check(small_jpeg).accepted

    # The same optical blur is rejected at either resolution.
    blurred_big = big.filter(ImageFilter.GaussianBlur(9))
    blurred_small = blurred_big.resize((800, 600), Image.Resampling.BOX)
    assert _check(_jpeg(blurred_big, **_at(0.0, 40.0))).reasons == [v.R_TOO_BLURRY]
    assert _check(_jpeg(blurred_small, **_at(0.0, 40.0))).reasons == [v.R_TOO_BLURRY]


def test_reencoded_copy_is_a_duplicate():
    original = _check(_jpeg(_scene(5, (800, 600)), quality=95, **_at(100.0, 40.0)))
    assert original.accepted
    others = [v.dhash_hex(_scene(seed)) for seed in (6, 7, 8)]
    assert all(v.hamming_hex(original.phash, h) > 10 for h in others)

    # Another user's copy: downscaled, heavier JPEG compression, different EXIF.
    copy = _scene(5, (800, 600)).resize((640, 480), Image.Resampling.BICUBIC)
    copy_jpeg = _jpeg(copy, quality=60, make="OtherPhone", **_at(110.0, 45.0))
    r = _check(copy_jpeg, existing=[None, *others, original.phash])
    assert r.reasons == [v.R_DUPLICATE]
    assert v.hamming_hex(r.phash, original.phash) <= Settings().duplicate_max_hamming
    assert r.sector == 2                     # geometry is still reported

    assert _check(_jpeg(_scene(6), **_at(100.0, 40.0)), existing=[original.phash]).accepted
    assert _check(copy_jpeg, existing=[]).accepted


def test_duplicate_threshold_is_inclusive():
    data = _jpeg(_scene(1), **_at(0.0, 40.0))
    h = int(_check(data).phash, 16)
    four_bits_off, five_bits_off = f"{h ^ 0b1111:016x}", f"{h ^ 0b11111:016x}"
    s = Settings(duplicate_max_hamming=4)
    assert _check(data, existing=[four_bits_off], settings=s).reasons == [v.R_DUPLICATE]
    assert _check(data, existing=[five_bits_off], settings=s).accepted


def test_exif_rotated_copy_hashes_like_the_upright_photo():
    upright = _scene(9)
    exif = piexif.load(build_exif_bytes(**_at(0.0, 40.0)))
    exif["0th"][piexif.ImageIFD.Orientation] = 6     # viewers rotate 90 deg clockwise to display
    buf = io.BytesIO()
    upright.transpose(Image.Transpose.ROTATE_90).save(buf, "JPEG", quality=90, exif=piexif.dump(exif))
    rotated = _check(buf.getvalue())
    assert rotated.accepted
    assert v.hamming_hex(rotated.phash, v.dhash_hex(upright)) <= 1


def test_all_applicable_reasons_are_collected():
    # No GPS, too small, blurry and already uploaded.
    tiny_blurry = _jpeg(_scene(10, (400, 300)).filter(ImageFilter.GaussianBlur(4)), heading_deg=10.0)
    first = _check(tiny_blurry)
    assert first.reasons == [v.R_NO_GPS, v.R_TOO_SMALL, v.R_TOO_BLURRY]
    again = _check(tiny_blurry, existing=[first.phash])
    assert again.reasons == [v.R_NO_GPS, v.R_TOO_SMALL, v.R_TOO_BLURRY, v.R_DUPLICATE]

    # Far away, facing away, too small, with magnetic heading; geometry still filled.
    r = _check(_jpeg(_scene(11, (400, 300)), **_at(225.0, 400.0, heading_offset=150.0, heading_ref="M")))
    assert r.reasons == [v.R_TOO_FAR, v.R_FACING_AWAY, v.R_TOO_SMALL]
    assert r.warnings == [v.W_MAGNETIC_HEADING, v.W_SUBJECT_UNCHECKED]
    assert r.distance_m == pytest.approx(400.0, abs=0.01)
    assert r.bearing_deg == pytest.approx(225.0, abs=0.01)
    assert r.sector == 5
    assert not r.accepted


# --- the qualification (D31): the building must be the photo's subject ------------------------------------------------


def _block(half: float = 10.0, height: float = 20.0) -> Subject:
    """A 20 x 20 m, 20 m tall block centred on LM (its catalogue frame)."""
    ring = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    pts = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (0.0, height)], np.float32)
    return Subject("block", LM.lat, LM.lon, (ring,), pts, "0" * 64)


def _check_subject(data: bytes, subject: Subject | None = None, settings: Settings | None = None, **kw):
    return v.validate_photo(data, read_photo_meta(data), kw.get("landmark", LM), (), settings or Settings(),
                            subject=_block() if subject is None else subject)


def test_a_photo_near_the_building_qualifies_and_carries_its_values():
    r = _check_subject(_jpeg(_scene(1), focal_35mm=28.0, **_at(180.0, 40.0)))       # 30 m from the outline
    assert r.accepted and r.reasons == [] and r.warnings == []
    assert r.outline_distance_m == pytest.approx(30.0, abs=0.01)
    assert 0.0 < r.subject_share <= 1.0


def test_a_far_photo_of_a_small_building_is_not_subject():
    r = _check_subject(_jpeg(_scene(1), focal_35mm=28.0, **_at(0.0, 140.0)))         # 130 m from a 20 m block
    assert r.reasons == [v.R_NOT_SUBJECT] and not r.accepted and r.warnings == []
    assert r.outline_distance_m == pytest.approx(130.0, abs=0.01) and r.subject_share < 0.33
    assert r.distance_m == pytest.approx(140.0, abs=0.01)                         # today's geometry still filled
    # a telephoto frames it: the same place qualifies on the share
    tele = _check_subject(_jpeg(_scene(1), focal_35mm=200.0, **_at(0.0, 140.0)))
    assert tele.accepted and tele.subject_share >= 0.33
    # the thresholds are Settings
    assert _check_subject(_jpeg(_scene(1), focal_35mm=28.0, **_at(0.0, 140.0)),
                          settings=Settings(subject_max_distance_m=130.5)).accepted
    assert _check_subject(_jpeg(_scene(1), focal_35mm=28.0, **_at(0.0, 140.0)),
                          settings=Settings(subject_min_share=r.subject_share)).accepted


def test_not_subject_is_reported_with_every_other_reason_and_ignores_the_compass():
    r = _check_subject(_jpeg(_scene(11, (400, 300)), focal_35mm=28.0,
                             **_at(225.0, 400.0, heading_offset=150.0, heading_ref="M")))
    assert r.reasons == [v.R_TOO_FAR, v.R_FACING_AWAY, v.R_NOT_SUBJECT, v.R_TOO_SMALL]
    assert r.warnings == [v.W_MAGNETIC_HEADING]
    # the compass decides facing_away only: pointing anywhere, the qualification is the same
    a = _check_subject(_jpeg(_scene(1), focal_35mm=28.0, **_at(90.0, 35.0, heading_offset=0.0)))
    b = _check_subject(_jpeg(_scene(1), focal_35mm=28.0, **_at(90.0, 35.0, heading_offset=100.0)))
    assert (a.outline_distance_m, a.subject_share) == (b.outline_distance_m, b.subject_share)
    assert a.reasons == [] and b.reasons == [v.R_FACING_AWAY]


def test_no_gps_or_no_landmark_runs_no_qualification_and_gives_no_subject_warning():
    r = _check_subject(_jpeg(_scene(1), heading_deg=10.0))
    assert r.reasons == [v.R_NO_GPS] and r.warnings == [] and r.outline_distance_m is None
    r = _check_subject(_jpeg(_scene(1), **_at(10.0, 30.0)), landmark=None)
    assert r.reasons == [v.R_NO_LANDMARK] and r.warnings == [] and r.subject_share is None


def test_the_web_text_names_the_distance_threshold():
    """web/strings.js says "离建筑超过 50 米" / "More than 50 m" for the reason not_subject (D32: both languages): the
    number must be Settings.subject_max_distance_m."""
    strings = (Path(__file__).resolve().parent.parent / "web" / "strings.js").read_text(encoding="utf-8")
    m = re.search(r"'api\.reason\.not_subject':\s*\{\s*zh:\s*'([^']*)',\s*en:\s*'([^']*)',?\s*\}", strings)
    assert m, "the not_subject entry of web/strings.js"
    limit = f"{Settings().subject_max_distance_m:g}"
    assert f"超过 {limit} 米" in m.group(1)
    assert f"More than {limit} m " in m.group(2)
