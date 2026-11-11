import io
import logging
import sqlite3
import threading
import time
from dataclasses import replace

import piexif
import pytest
from PIL import ExifTags, Image, ImageDraw

import hostile_images as hostile   # (pytest puts tests/ on the path)
from crowd3d import contributors
from crowd3d.config import Settings
from crowd3d.exif import UnreadableImage, read_photo_meta
from crowd3d.ingest import R_TOO_LARGE, ingest_upload, make_processed_copy
from crowd3d.records import PHOTO_ACCEPTED, PHOTO_REJECTED, TERMS_CURRENT, Landmark, location_shared
from crowd3d.store import Store
from crowd3d.synthetic import CAMERA_RADIUS_M
from crowd3d.validation import W_SUBJECT_UNCHECKED

LM = "demo"
BAD_KINDS = ["blurry", "far", "facing_away", "duplicate", "no_gps", "too_small"]


@pytest.fixture
def store(settings):
    s = Store(settings)
    yield s
    s.close()


@pytest.fixture
def landmark(store, synthetic_dataset):
    ds = synthetic_dataset
    return store.create_landmark(Landmark(id=LM, name="Demo", lat=ds.lat, lon=ds.lon, radius_m=150.0, created_at=0.0))


def register(store, nickname="u"):
    """A contributor registered the product's way (D33): the nickname and the current user agreement."""
    return contributors.register(store, nickname, TERMS_CURRENT)[0]


@pytest.fixture
def contributor(store):
    return register(store, "user1")


def ingest(store, settings, ds, entry, landmark, user_id="user1", data=None, contributor=None):
    """Upload as contributor, else as a contributor registered now under the nickname user_id."""
    data = ds.read(entry) if data is None else data
    contributor = contributor or register(store, user_id)
    return ingest_upload(store, settings, filename=entry["file"], data=data, contributor=contributor,
                         landmark=landmark)


def stored_files(settings):
    return sorted(p.relative_to(settings.data_dir).as_posix() for p in settings.photos_dir.rglob("*") if p.is_file())


def jpeg(size=(640, 480), mode="RGB", exif=None, fmt="JPEG", draw=None):
    img = Image.new(mode, size, "white" if mode != "P" else 0)
    if draw:
        draw(ImageDraw.Draw(img))
    buf = io.BytesIO()
    img.save(buf, fmt, **({"exif": exif} if exif is not None else {}))
    return buf.getvalue()


def test_accepted_photo_is_recorded_with_original_and_processed_copy(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    entry = ds.entries("good", sector=1)[0]
    data = ds.read(entry)
    out = ingest(store, settings, ds, entry, landmark, user_id="alice")

    # no bound building outline at the test landmark: not checked for the subject rule (D31)
    assert (out.accepted, out.reasons, out.warnings, out.filename) == (True, [], [W_SUBJECT_UNCHECKED], entry["file"])
    p = out.photo
    assert p.outline_distance_m is None and p.subject_share is None
    assert store.get_photo(p.id) == p
    assert (p.landmark_id, p.user_id, p.status, p.sector) == (LM, "alice", PHOTO_ACCEPTED, 1)
    assert p.lat == pytest.approx(entry["exif_lat"], abs=1e-7)
    assert p.lon == pytest.approx(entry["exif_lon"], abs=1e-7)
    assert p.heading_deg == pytest.approx(entry["exif_heading_deg"], abs=0.01)
    assert CAMERA_RADIUS_M[0] - 10 < p.distance_m < CAMERA_RADIUS_M[1] + 10 and p.bearing_deg is not None
    assert (p.width, p.height, p.focal_35mm) == (640, 480, 28.0)
    assert len(p.phash) == 16 and p.captured_at

    assert p.original_path == f"photos/{LM}/{p.id}.jpg"
    assert p.processed_path == f"photos/{LM}/{p.id}.proc.jpg"
    assert (settings.data_dir / p.original_path).read_bytes() == data
    assert stored_files(settings) == sorted([p.original_path, p.processed_path])


def test_an_upload_needs_a_contributor_who_accepted_the_current_agreement(store, settings, synthetic_dataset,
                                                                         landmark, contributor):
    """D33: the uploader is a registered contributor (crowd3d.contributors); the photo takes its user_id (the
    nickname) and contributor_id from the record, and an accepted photo the version the contributor accepted as its
    location_consent (shared), while a rejected photo is never shown, so it keeps none. A missing contributor is a
    TypeError, one whose accepted version is not the current one a ValueError, and nothing is stored."""
    ds = synthetic_dataset
    good = ds.entries("good", sector=0)
    shared = ingest(store, settings, ds, good[0], landmark, contributor=contributor)
    p = store.get_photo(shared.photo.id)
    assert shared.accepted and p.location_consent == contributor.terms == TERMS_CURRENT == "terms-v1"
    assert (p.user_id, p.contributor_id) == ("user1", contributor.id) and location_shared(p)
    rejected = ingest(store, settings, ds, ds.entry("blurry"), landmark, contributor=contributor)
    r = store.get_photo(rejected.photo.id)
    assert not rejected.accepted and r.location_consent is None
    assert (r.user_id, r.contributor_id) == ("user1", contributor.id)
    before, files = len(store.list_photos(LM)), stored_files(settings)
    with pytest.raises(TypeError, match="contributor"):
        ingest_upload(store, settings, filename="a.jpg", data=ds.read(good[1]), landmark=landmark)
    with pytest.raises(TypeError, match="user_id"):
        ingest_upload(store, settings, filename="a.jpg", data=ds.read(good[1]), user_id="u", landmark=landmark,
                      contributor=contributor)
    for other in ("", "share-location-v1", "share-location-v2", "terms-v0", "yes"):
        with pytest.raises(ValueError, match="needs a contributor who accepted the current user agreement"):
            ingest(store, settings, ds, good[1], landmark, contributor=replace(contributor, terms=other))
    assert len(store.list_photos(LM)) == before and stored_files(settings) == files


def _with_heading_ref(data, ref):
    exif = piexif.load(data)
    if ref is None:
        exif["GPS"].pop(piexif.GPSIFD.GPSImgDirectionRef, None)
    else:
        exif["GPS"][piexif.GPSIFD.GPSImgDirectionRef] = ref
    out = io.BytesIO()
    piexif.insert(piexif.dump(exif), data, out)
    return out.getvalue()


def test_heading_ref_is_stored_as_true_magnetic_or_none(store, settings, synthetic_dataset, landmark):
    """The compass heading's north (EXIF GPSImgDirectionRef) is kept for the heading's honest uncertainty (D30): "T"
    or "M"; no tag, or any other value, is None."""
    ds = synthetic_dataset
    good = ds.entries("good", sector=1)
    for entry, ref, stored in zip(good, (b"T", b"M", None, b"X"), ("T", "M", None, None)):
        out = ingest(store, settings, ds, entry, landmark, data=_with_heading_ref(ds.read(entry), ref))
        assert out.accepted, (ref, out.reasons)
        assert out.photo.heading_deg is not None and store.get_photo(out.photo.id).heading_ref == stored, ref


def test_processed_copy_keeps_only_the_exif_sfm_needs(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    out = ingest(store, settings, ds, ds.entries("good")[0], landmark)
    original = read_photo_meta(ds.read(ds.entries("good")[0]))
    proc_bytes = (settings.data_dir / out.photo.processed_path).read_bytes()
    proc = read_photo_meta(proc_bytes)

    assert Image.open(io.BytesIO(proc_bytes)).format == "JPEG"
    assert (proc.width, proc.height) == (640, 480)  # under processing_max_side_px: not resized
    assert (proc.focal_mm, proc.focal_35mm, proc.make, proc.model) == (
        original.focal_mm, original.focal_35mm, original.make, original.model
    )
    assert original.focal_mm and original.has_gps and original.heading_deg is not None
    assert not proc.has_gps and proc.heading_deg is None and proc.captured_at is None


def test_processed_copy_is_upright_and_downscaled(settings):
    # Stored landscape with a red block top-left; orientation 6 means "rotate 90 degrees clockwise to view".
    exif = Image.Exif()
    exif[ExifTags.Base.Orientation] = 6
    exif[ExifTags.Base.Make] = "Acme"
    exif[ExifTags.Base.Model] = "One"
    exif.get_ifd(ExifTags.IFD.Exif)[ExifTags.Base.FocalLengthIn35mmFilm] = 26
    data = jpeg((1000, 600), exif=exif, draw=lambda d: d.rectangle([0, 0, 199, 199], fill=(255, 0, 0)))
    meta = read_photo_meta(data)
    assert meta.orientation == 6

    out = make_processed_copy(data, meta, replace(settings, processing_max_side_px=500))
    img = Image.open(io.BytesIO(out))
    assert img.size == (300, 500) and img.mode == "RGB"
    r, g, b = img.getpixel((img.width - 20, 20))  # stored top-left is now top-right
    assert r > 200 and g < 60 and b < 60
    proc = read_photo_meta(out)
    assert proc.orientation is None  # pixels are upright now; a stale tag would rotate them twice
    assert (proc.focal_35mm, proc.make, proc.model) == (26.0, "Acme", "One")


@pytest.mark.parametrize("mode,fmt", [("RGBA", "PNG"), ("P", "PNG"), ("L", "JPEG"), ("CMYK", "JPEG")])
def test_processed_copy_converts_any_mode_to_rgb_jpeg(settings, mode, fmt):
    data = jpeg((700, 500), mode=mode, fmt=fmt)
    out = make_processed_copy(data, read_photo_meta(data), settings)
    img = Image.open(io.BytesIO(out))
    assert (img.format, img.mode, img.size) == ("JPEG", "RGB", (700, 500))
    meta = read_photo_meta(out)
    assert meta.make is None and meta.model is None and meta.focal_35mm is None


@pytest.mark.parametrize("kind", BAD_KINDS)
def test_bad_photo_is_rejected_with_its_reason_and_keeps_no_bytes(store, settings, synthetic_dataset, landmark, kind):
    ds = synthetic_dataset
    entry = ds.entry(kind)
    if kind == "duplicate":
        original = next(e for e in ds.entries("good") if e["file"] == entry["duplicate_of"])
        assert ingest(store, settings, ds, original, landmark).accepted
    before = stored_files(settings)

    out = ingest(store, settings, ds, entry, landmark)

    assert not out.accepted
    assert out.reasons == [entry["expected_reason"]]
    p = store.get_photo(out.photo.id)
    assert (p.status, p.reasons, p.landmark_id) == (PHOTO_REJECTED, [entry["expected_reason"]], LM)
    assert p.original_path is None and p.processed_path is None
    assert stored_files(settings) == before
    if kind == "no_gps":
        assert p.sector is None and p.bearing_deg is None
    else:
        # Geometry is filled even for rejected photos, for feedback in the UI.
        assert p.sector == entry["sector"] and p.distance_m is not None
    if kind == "far":
        assert p.distance_m == pytest.approx(500, abs=10)


def test_too_large_is_rejected_without_a_record(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    entry = ds.entries("good")[0]
    data = ds.read(entry)

    out = ingest(store, replace(settings, max_upload_bytes=len(data) - 1), ds, entry, landmark)
    assert (out.accepted, out.photo, out.reasons, out.warnings) == (False, None, [R_TOO_LARGE], [])
    assert store.list_photos(LM) == [] and stored_files(settings) == []

    assert ingest(store, replace(settings, max_upload_bytes=len(data)), ds, entry, landmark).accepted


def test_unreadable_bytes_are_recorded_as_rejected(store, settings):
    out = ingest_upload(store, settings, filename="notes.txt", data=b"not an image", contributor=register(store),
                        landmark=None)
    assert (out.accepted, out.reasons) == (False, ["unreadable_image"])
    p = store.get_photo(out.photo.id)
    assert (p.status, p.landmark_id, p.width, p.phash) == (PHOTO_REJECTED, None, None, None)


def test_decompression_bomb_is_unreadable_not_a_crash(store, settings, monkeypatch):
    data = jpeg((64, 64), fmt="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    out = ingest_upload(store, settings, filename="bomb.png", data=data, contributor=register(store),
                        landmark=None)
    assert out.reasons == ["unreadable_image"]


def test_landmark_is_found_from_gps_when_not_given(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    # A second landmark 1 km north must not capture the photo.
    store.create_landmark(Landmark(id="other", name="Other", lat=ds.lat + 0.009, lon=ds.lon, radius_m=150.0,
                                   created_at=1.0))
    out = ingest(store, settings, ds, ds.entries("good", sector=0)[0], None)
    assert out.accepted and out.photo.landmark_id == LM and out.photo.sector == 0


def test_gps_outside_every_landmark_is_no_landmark(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    out = ingest(store, settings, ds, ds.entry("far"), None)
    assert out.reasons == ["no_landmark"]
    assert out.photo.landmark_id is None and out.photo.sector is None


def test_client_filename_never_reaches_a_path(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    entry = ds.entries("good")[0]
    out = ingest_upload(store, settings, filename="../../../etc/passwd.jpg", data=ds.read(entry),
                        contributor=register(store), landmark=landmark)
    assert out.accepted and out.filename == "../../../etc/passwd.jpg"
    assert out.photo.original_path == f"photos/{LM}/{out.photo.id}.jpg"
    assert all(".." not in f and "passwd" not in f for f in stored_files(settings))


def test_same_photo_twice_is_a_duplicate(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    entry = ds.entries("good")[0]
    assert ingest(store, settings, ds, entry, landmark).accepted
    again = ingest(store, settings, ds, entry, landmark, user_id="someone-else")
    assert again.reasons == ["duplicate"]


def test_concurrent_uploads_of_one_photo_accept_exactly_one(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    entry = ds.entries("good")[0]
    data = ds.read(entry)
    outcomes = []
    barrier = threading.Barrier(4)

    def worker(i):
        barrier.wait()
        outcomes.append(ingest(store, settings, ds, entry, landmark, user_id=f"u{i}", data=data))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(o.accepted for o in outcomes) == [False, False, False, True]
    assert len(store.list_photos(LM, PHOTO_ACCEPTED)) == 1


# Hostile or unlucky uploads


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_huge_non_jpeg_is_rejected_before_any_pixel_is_decoded(store, settings, monkeypatch):
    # A flat 12000 x 12000 PNG compresses to ~2 MB but decodes to 1.2 GB of RAM, even when rejected later.
    from crowd3d import ingest as ingest_module

    buf = io.BytesIO()
    Image.new("L", (12000, 12000)).save(buf, "PNG")
    data = buf.getvalue()
    assert len(data) < settings.max_upload_bytes

    def no_decode(*args, **kwargs):
        raise AssertionError("pixels decoded")

    monkeypatch.setattr(ingest_module, "validate_photo", no_decode)
    monkeypatch.setattr(ingest_module, "make_processed_copy", no_decode)
    out = ingest_upload(store, settings, filename="flat.png", data=data, contributor=register(store),
                        landmark=None)
    assert (out.accepted, out.photo, out.reasons) == (False, None, [R_TOO_LARGE])


def test_postscript_is_never_handed_to_a_decoder(store, settings):
    # Pillow renders EPS by running Ghostscript on the uploaded PostScript.
    eps = (b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 800 600\n"
           b"newpath 0 0 moveto 800 600 lineto stroke\nshowpage\n%%EOF\n")
    with pytest.raises(UnreadableImage):
        read_photo_meta(eps)
    out = ingest_upload(store, settings, filename="x.eps", data=eps, contributor=register(store), landmark=None)
    assert out.reasons == ["unreadable_image"]


def test_a_photo_whose_processed_copy_fails_is_rejected_and_leaves_no_file(store, settings, synthetic_dataset,
                                                                           landmark, monkeypatch):
    # Crafted EXIF (Make and Model sharing one 40 kB string) made the re-encode raise "EXIF data is too
    # long": the request died with a 500 and the original, GPS and all, stayed on disk without a record.
    from crowd3d import ingest as ingest_module

    def explode(*args, **kwargs):
        raise ValueError("EXIF data is too long")

    monkeypatch.setattr(ingest_module, "make_processed_copy", explode)
    ds = synthetic_dataset
    out = ingest(store, settings, ds, ds.entries("good")[0], landmark)
    assert (out.accepted, out.reasons) == (False, ["unreadable_image"])
    assert store.get_photo(out.photo.id).status == PHOTO_REJECTED
    assert stored_files(settings) == []


def test_a_failed_write_leaves_neither_a_record_nor_an_orphan_file(store, settings, synthetic_dataset, landmark,
                                                                   monkeypatch):
    ds = synthetic_dataset
    real_add = store.add_photo

    def disk_full(photo):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(store, "add_photo", disk_full)
    with pytest.raises(OSError):
        ingest(store, settings, ds, ds.entries("good")[0], landmark)
    assert stored_files(settings) == []
    monkeypatch.setattr(store, "add_photo", real_add)
    assert ingest(store, settings, ds, ds.entries("good")[0], landmark).accepted   # not a "duplicate" of itself


def test_processed_copy_caps_exif_text(settings):
    meta = read_photo_meta(jpeg())
    long = replace(meta, make="M" * 40000, model="M" * 40000)
    out = make_processed_copy(jpeg(), long, settings)
    proc = read_photo_meta(out)
    assert len(proc.make) == len(proc.model) == 64


def test_a_bound_landmark_checks_the_subject_and_stores_its_values(store, settings, synthetic_dataset):
    """India Gate, bound by its id to the repo catalogue's OSM snapshot (D31): near the arch the photo qualifies, at
    250 m with a 28 mm frame it is not the subject. The values are stored with the photo, and the building is loaded
    once (the landmark's cache file)."""
    import json

    from crowd3d.config import REPO_ROOT
    from crowd3d.exif import build_exif_bytes
    from crowd3d.geo import LocalENU
    from crowd3d.validation import R_NOT_SUBJECT

    entry = next(e for e in json.loads((REPO_ROOT / "eval" / "landmarks.json").read_text()) if e["slug"] == "india-gate")
    lm = store.create_landmark(Landmark(id="india-gate", name="India Gate", lat=entry["lat"], lon=entry["lon"],
                                        radius_m=300.0, created_at=0.0))
    good = synthetic_dataset.entries("good")
    enu = LocalENU(entry["lat"], entry["lon"])

    def at(n: float, k: int) -> bytes:       # the pixels of the k-th good synthetic photo (no duplicates)
        lat, lon, _ = enu.to_geodetic(0.0, n)
        buf = io.BytesIO()
        Image.open(io.BytesIO(synthetic_dataset.read(good[k]))).save(
            buf, "JPEG", quality=92, exif=build_exif_bytes(lat=lat, lon=lon, focal_35mm=28.0))
        return buf.getvalue()

    near = ingest_upload(store, settings, filename="near.jpg", data=at(-30.0, 0), contributor=register(store),
                          landmark=lm)
    assert near.accepted and near.reasons == [] and "subject_unchecked" not in near.warnings
    assert near.photo.outline_distance_m < 50.0 and 0.0 < near.photo.subject_share <= 1.0
    assert store.get_photo(near.photo.id) == near.photo
    far = ingest_upload(store, settings, filename="far.jpg", data=at(-250.0, 4), contributor=register(store),
                         landmark=lm)
    assert not far.accepted and far.reasons == [R_NOT_SUBJECT]
    assert far.photo.outline_distance_m > 200.0 and far.photo.subject_share < settings.subject_min_share
    assert store.get_photo(far.photo.id).subject_share == far.photo.subject_share
    assert [p.name for p in (settings.maps_dir / "india-gate").glob("subject-*.npz")] == \
        [f"subject-{entry['osm_parts']['sha256'][:12]}.npz"]


# Crafted metadata and decode cost (D34)


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_only_the_exif_form_of_a_capture_time_is_stored(store, settings, synthetic_dataset, landmark, fmt):
    """photos.captured_at holds EXIF's own 19-character form or nothing. A 1,000,000-character DateTimeOriginal (every
    format carries one) was stored as sent, in accepted and rejected rows alike, and came back with every read of the
    photo table. The original file is kept as uploaded (D34: originals stay)."""
    ds = synthetic_dataset
    long = b"2024:05:06 07:08:09" + b"9" * (1_000_000 - 19)
    data = hostile.with_exif(fmt, hostile.exif_with(capture_time=long, lat=ds.lat + 0.0002, lon=ds.lon))
    assert len(data) > 1_000_000
    out = ingest_upload(store, settings, filename="long." + fmt.lower(), data=data, contributor=register(store),
                        landmark=None)
    assert out.accepted and out.photo.landmark_id == LM, out.reasons
    assert out.photo.captured_at is None and store.get_photo(out.photo.id).captured_at is None
    assert (settings.data_dir / out.photo.original_path).read_bytes() == data
    # a rejected row (no landmark there) keeps none of it either
    far = hostile.with_exif(fmt, hostile.exif_with(capture_time=long, lat=ds.lat + 1.0, lon=ds.lon))
    rejected = ingest_upload(store, settings, filename="far", data=far, contributor=register(store), landmark=None)
    assert rejected.reasons == ["no_landmark"] and store.get_photo(rejected.photo.id).captured_at is None
    # the form itself is stored, and one character more or less is not
    stored = {}
    for i, text in enumerate([b"2024:05:06 07:08:09", b"2024:05:06 07:08:090", b"2024-05-06 07:08:09", b"2024:05:06"]):
        d = hostile.with_exif(fmt, hostile.exif_with(capture_time=text, lat=ds.lat + 1.0, lon=ds.lon))
        o = ingest_upload(store, settings, filename=f"t{i}", data=d, contributor=register(store), landmark=None)
        stored[text] = store.get_photo(o.photo.id).captured_at
    assert stored == {b"2024:05:06 07:08:09": "2024:05:06 07:08:09", b"2024:05:06 07:08:090": None,
                      b"2024-05-06 07:08:09": None, b"2024:05:06": None}
    with sqlite3.connect(settings.db_path) as conn:
        assert conn.execute("SELECT max(length(captured_at)), count(captured_at) FROM photos").fetchone() == (19, 1)


def test_the_synthetic_photos_keep_their_capture_time(store, settings, synthetic_dataset, landmark):
    ds = synthetic_dataset
    entry = ds.entries("good")[0]
    out = ingest(store, settings, ds, entry, landmark)
    meta = read_photo_meta(ds.read(entry))
    assert out.accepted and out.photo.captured_at == meta.captured_at and len(meta.captured_at) == 19


@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_a_text_tag_written_as_a_number_does_not_fail_the_upload(store, settings, synthetic_dataset, landmark, fmt):
    """The review's finding: a camera make declared as SHORT made read_photo_meta raise, and the API answer
    server_error for the file. The tag is now missing and the photo is judged like any other."""
    ds = synthetic_dataset
    good = hostile.exif_with(capture_time=b"2024:05:06 07:08:09", lat=ds.lat + 0.0002, lon=ds.lon, make=b"ab")
    for i, tag in enumerate((piexif.ImageIFD.Make, piexif.ExifIFD.DateTimeOriginal)):
        img = hostile.textured(variant=i)                         # other pixels: not a duplicate of the first
        data = hostile.with_exif(fmt, hostile.retyped(good, tag), img)
        out = ingest_upload(store, settings, filename="typed", data=data, contributor=register(store), landmark=None)
        assert out.accepted and out.reasons == [], (tag, out.reasons)
        assert out.photo.captured_at == (None if tag == piexif.ExifIFD.DateTimeOriginal else "2024:05:06 07:08:09")


def test_whatever_reading_the_metadata_raises_the_upload_is_unreadable_not_an_error(store, settings, monkeypatch,
                                                                                    caplog):
    """Belt and braces for parsers of hostile input: an exception read_photo_meta does not expect is one warning line
    (no traceback: 50 files make 50 of them) and an unreadable_image row, not an exception out of ingest_upload."""
    from crowd3d import ingest as ingest_module

    def broken(data):
        raise AttributeError("'int' object has no attribute 'strip'")

    monkeypatch.setattr(ingest_module, "read_photo_meta", broken)
    caplog.set_level(logging.DEBUG)
    out = ingest_upload(store, settings, filename="odd.jpg", data=hostile.jpeg(), contributor=register(store),
                        landmark=None)
    assert (out.accepted, out.reasons) == (False, ["unreadable_image"])
    assert store.get_photo(out.photo.id).status == PHOTO_REJECTED
    (record,) = [r for r in caplog.records if r.name == "crowd3d.ingest"]
    assert record.levelno == logging.WARNING and record.exc_info is None and "odd.jpg" in record.getMessage()


def no_decode(*args, **kwargs):
    raise AssertionError("pixels decoded")


def test_the_pixel_cap_bounds_every_jpeg_that_does_not_decode_reduced(store, settings, synthetic_dataset, landmark,
                                                                      monkeypatch):
    """Settings.max_image_pixels: a progressive JPEG, and a sequential one with a scan per component, are too_large
    above it before any pixel is decoded, like a PNG or a WebP (JPEG as a whole used to be exempt: the decoder keeps
    the full image's coefficients for these, 290-525 MB for 9000 x 9000). At the cap they are decoded. A sequential
    single-scan JPEG, what cameras write, stays exempt: it decodes at 1/8 scale."""
    from crowd3d import ingest as ingest_module

    ds = synthetic_dataset
    exif = hostile.exif_with(lat=ds.lat + 0.0002, lon=ds.lon)
    over = replace(settings, max_image_pixels=640 * 480 - 1)
    at = replace(settings, max_image_pixels=640 * 480)
    sequential = hostile.with_exif("JPEG", exif)
    progressive = _resaved(sequential, progressive=True, exif=exif)
    scans = hostile.non_interleaved_jpeg()
    png, webp = hostile.with_exif("PNG", exif), hostile.with_exif("WEBP", exif)
    assert not read_photo_meta(progressive).decodes_reduced and read_photo_meta(progressive).has_gps

    def upload(data, s):
        return ingest_upload(store, s, filename="x", data=data, contributor=register(store), landmark=None)

    with monkeypatch.context() as m:
        m.setattr(ingest_module, "validate_photo", no_decode)
        m.setattr(ingest_module, "make_processed_copy", no_decode)
        for what, data in (("progressive", progressive), ("a scan per component", scans), ("png", png),
                           ("webp", webp)):
            out = upload(data, over)
            assert (out.accepted, out.photo, out.reasons) == (False, None, [R_TOO_LARGE]), what
    assert store.list_photos(LM) == [] and stored_files(settings) == []
    first = upload(sequential, over)                     # exempt: decoded and judged
    assert first.accepted, first.reasons
    for what, data in (("progressive", progressive), ("png", png), ("webp", webp)):
        out = upload(data, at)                           # at the cap: decoded and judged (the same pixels: duplicates)
        assert out.photo is not None and out.reasons == ["duplicate"], (what, out.reasons)
    out = upload(scans, at)
    assert out.photo is not None and R_TOO_LARGE not in out.reasons      # (flat, no EXIF: rejected for that)


def _resaved(data, **save):
    buf = io.BytesIO()
    with Image.open(io.BytesIO(data)) as im:
        im.save(buf, "JPEG", quality=92, **save)
    return buf.getvalue()


@pytest.mark.filterwarnings("ignore::PIL.Image.DecompressionBombWarning")
def test_a_64_megapixel_progressive_jpeg_is_too_large_at_the_default_cap(store, settings, monkeypatch):
    from crowd3d import ingest as ingest_module

    assert Settings().max_image_pixels == 50_000_000
    buf = io.BytesIO()
    Image.new("L", (8000, 8000)).save(buf, "JPEG", progressive=True)
    data = buf.getvalue()
    assert len(data) < 1_000_000 < settings.max_upload_bytes
    monkeypatch.setattr(ingest_module, "validate_photo", no_decode)
    monkeypatch.setattr(ingest_module, "make_processed_copy", no_decode)
    out = ingest_upload(store, settings, filename="flat.jpg", data=data, contributor=register(store), landmark=None)
    assert (out.accepted, out.photo, out.reasons) == (False, None, [R_TOO_LARGE])


def test_at_most_max_concurrent_decodes_uploads_decode_at_once(store, settings, synthetic_dataset, landmark,
                                                               monkeypatch):
    """Settings.max_concurrent_decodes is one limit for the whole process: uploads to different landmarks and uploads
    that match no landmark (which take no landmark lock) wait for the same slots, and both decodes of an upload (the
    analysis copy, the processed copy) run inside one slot. A decode that raises gives its slot back."""
    from crowd3d import ingest as ingest_module

    assert Settings().max_concurrent_decodes == 2
    ds = synthetic_dataset
    other = store.create_landmark(Landmark(id="other", name="Other", lat=ds.lat + 0.5, lon=ds.lon, radius_m=150.0,
                                           created_at=0.0))
    real_validate, real_processed = ingest_module.validate_photo, ingest_module.make_processed_copy
    guard = threading.Lock()
    state = {"now": 0, "most": 0, "validate": 0, "processed": 0, "outside": 0}

    def decoding(key, real):
        def run(*args, **kwargs):
            with guard:
                state[key] += 1
                state["now"] += 1
                state["most"] = max(state["most"], state["now"])
                if state["now"] > ingest_module._decode_slots._active:     # more decodes running than slots held
                    state["outside"] += 1
            try:
                time.sleep(0.05)
                return real(*args, **kwargs)
            finally:
                with guard:
                    state["now"] -= 1
        return run

    monkeypatch.setattr(ingest_module, "validate_photo", decoding("validate", real_validate))
    monkeypatch.setattr(ingest_module, "make_processed_copy", decoding("processed", real_processed))
    good = ds.entries("good")
    plain = hostile.jpeg()                               # no EXIF: matches no landmark
    people = [register(store, f"u{i}") for i in range(9)]

    def run(limit):
        state.update(now=0, most=0, validate=0, processed=0, outside=0)
        s = replace(settings, max_concurrent_decodes=limit)
        jobs = [(ds.read(good[i]), landmark if i % 2 else None) for i in range(4)]      # found by GPS or given
        jobs += [(plain, None)] * 3 + [(plain, other)] * 2
        outcomes = [None] * len(jobs)
        barrier = threading.Barrier(len(jobs))

        def worker(i):
            barrier.wait()
            outcomes[i] = ingest_upload(store, s, filename=f"f{i}", data=jobs[i][0], contributor=people[i],
                                        landmark=jobs[i][1])

        threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(len(jobs))]
        for t in threads:
            t.start()
        deadline = time.monotonic() + 60
        for t in threads:
            t.join(timeout=max(0.0, deadline - time.monotonic()))
        assert not any(t.is_alive() for t in threads), "uploads are still waiting for a decode slot"
        assert all(o is not None for o in outcomes)
        return dict(state), outcomes

    seen, outcomes = run(2)
    assert seen["most"] == 2 and seen["validate"] == 9 and seen["outside"] == 0, seen
    assert sum(o.accepted for o in outcomes) == 4 == seen["processed"]       # each made inside its upload's slot
    seen, _ = run(1)
    assert seen["most"] == 1 and seen["validate"] == 9, seen
    seen, _ = run(0)                                     # a limit below 1 is 1: uploads never wait for ever
    assert seen["most"] == 1
    seen, _ = run(50)
    assert 2 < seen["most"] <= 9, seen                   # without a limit to speak of they do overlap

    def boom(*args, **kwargs):
        raise RuntimeError("decoder crashed")

    monkeypatch.setattr(ingest_module, "validate_photo", boom)
    one = replace(settings, max_concurrent_decodes=1)
    raised = []

    def crashing():
        for _ in range(3):                               # the slot comes back each time
            try:
                ingest_upload(store, one, filename="x", data=plain, contributor=people[0], landmark=None)
            except RuntimeError as exc:
                raised.append(str(exc))

    t = threading.Thread(target=crashing, daemon=True)
    t.start()
    t.join(timeout=30)
    assert not t.is_alive(), "an upload waits for a slot a crashed decode kept"
    assert raised == ["decoder crashed"] * 3 and ingest_module._decode_slots._active == 0


# EXIF that costs the server memory or time (D34, review round 1; the reader itself: tests/test_geo_exif.py)


@pytest.mark.filterwarnings("ignore:Tag data larger than file:UserWarning")
@pytest.mark.parametrize("fmt", ["JPEG", "PNG", "WEBP"])
def test_an_upload_whose_exif_declares_billions_of_values_is_an_ordinary_rejection(store, settings, fmt):
    """The review's finding, as an upload: a small file whose EXIF declares 60,000,000 or 2**31 values for one tag
    took 2 GB of the server's memory and kept it. It is rejected like any photo without readable EXIF, in
    milliseconds, and leaves a rejected record and no file."""
    import tracemalloc

    contributor = register(store)
    for count in (60_000_000, 2 ** 31):
        for kind in (hostile.TIFF_BYTE, hostile.TIFF_SHORT, hostile.TIFF_LONG):
            blob = hostile.exif_block([(piexif.ImageIFD.Make, 2, 5, 8), (piexif.ImageIFD.Model, kind, count, 8)],
                                      b"Acme\0")
            data = hostile.with_exif(fmt, blob)
            tracemalloc.start()
            start = time.perf_counter()
            out = ingest_upload(store, settings, filename="bomb", data=data, contributor=contributor, landmark=None)
            seconds, peak = time.perf_counter() - start, tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            assert (out.accepted, out.reasons) == (False, ["no_exif", "no_gps"]), (kind, count, out.reasons)
            assert seconds < 2.0 and peak < 16 * 1024 * 1024, (kind, count, seconds, peak)
    assert stored_files(settings) == []


def test_an_exif_block_over_the_limit_is_unreadable_and_nothing_decodes_it(store, settings, monkeypatch):
    """An image whose EXIF block is over exif.MAX_EXIF_BYTES is unreadable: no decoder and no EXIF parser (Pillow's
    own runs when the pixels are turned upright) gets to see it."""
    from crowd3d import exif as exif_module
    from crowd3d import ingest as ingest_module
    from crowd3d import validation as validation_module

    decodes = []
    real_gray = validation_module._analysis_gray
    monkeypatch.setattr(validation_module, "_analysis_gray", lambda data: decodes.append(len(data)) or real_gray(data))
    monkeypatch.setattr(ingest_module, "make_processed_copy", no_decode)
    make = (piexif.ImageIFD.Make, 2, 5, 8)
    contributor = register(store)
    for extra, reasons, decoded in ((0, ["no_gps"], 1), (1, ["unreadable_image"], 0)):
        filler = b"Acme\0".ljust(exif_module.MAX_EXIF_BYTES - 6 - 8 - 18 + extra, b"\0")
        data = hostile.with_exif("PNG", hostile.exif_block([make], filler))
        del decodes[:]
        out = ingest_upload(store, settings, filename="big-exif.png", data=data, contributor=contributor, landmark=None)
        assert (out.accepted, out.reasons, out.photo.status) == (False, reasons, PHOTO_REJECTED), extra
        assert len(decodes) == decoded, (extra, decodes)
    assert stored_files(settings) == []


# The decode limit bounds a server's memory only with glibc's mmap threshold pinned (D34, review round 1)


class FakeLibc:
    """Stands for the C library: records mallopt's calls, and is glibc only when asked to be."""

    def __init__(self, answer=1, glibc=True):
        self.calls, self.answer = [], answer
        if glibc:
            self.gnu_get_libc_version = object()

    def mallopt(self, parameter, value):
        self.calls.append((parameter, value))
        return self.answer


def test_the_malloc_threshold_is_pinned_once_and_only_on_glibc(monkeypatch):
    """ingest.pin_malloc_threshold: mallopt(M_MMAP_THRESHOLD, bytes) once per process; nothing is touched for a
    threshold of 0, when the operator set MALLOC_MMAP_THRESHOLD_ (glibc has pinned that value already), or on another
    C library. (What it does to a server's memory: tests/test_api.py, on a real server.)"""
    from crowd3d import ingest as ingest_module

    def fresh(libc):
        monkeypatch.setattr(ingest_module, "_malloc_pinned_at", None)
        monkeypatch.setattr(ingest_module.ctypes, "CDLL", lambda name: libc)
        return libc

    monkeypatch.delenv("MALLOC_MMAP_THRESHOLD_", raising=False)
    assert Settings().malloc_mmap_threshold_bytes == 4 * 1024 * 1024
    libc = fresh(FakeLibc())
    assert ingest_module.pin_malloc_threshold(0) is False and ingest_module.pin_malloc_threshold(-1) is False
    assert libc.calls == []
    assert ingest_module.pin_malloc_threshold(4 * 1024 * 1024) is True
    assert libc.calls == [(-3, 4 * 1024 * 1024)]                # M_MMAP_THRESHOLD is -3 in <malloc.h>
    assert ingest_module.pin_malloc_threshold(1024) is True and ingest_module.pin_malloc_threshold(0) is True
    assert libc.calls == [(-3, 4 * 1024 * 1024)]                # once: the first value stays
    libc = fresh(FakeLibc())
    monkeypatch.setenv("MALLOC_MMAP_THRESHOLD_", "1048576")
    assert ingest_module.pin_malloc_threshold(4 * 1024 * 1024) is False and libc.calls == []
    monkeypatch.delenv("MALLOC_MMAP_THRESHOLD_")
    libc = fresh(FakeLibc(glibc=False))                          # musl, macOS: no gnu_get_libc_version
    assert ingest_module.pin_malloc_threshold(4 * 1024 * 1024) is False and libc.calls == []
    libc = fresh(FakeLibc(answer=0))                             # mallopt refused the value
    assert ingest_module.pin_malloc_threshold(4 * 1024 * 1024) is False and len(libc.calls) == 1
    assert ingest_module._malloc_pinned_at is None

    def no_libc(name):
        raise OSError("no C library to load")

    fresh(None)
    monkeypatch.setattr(ingest_module.ctypes, "CDLL", no_libc)
    assert ingest_module.pin_malloc_threshold(4 * 1024 * 1024) is False
