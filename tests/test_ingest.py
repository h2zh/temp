import io
import threading
from dataclasses import replace

import pytest
from PIL import ExifTags, Image, ImageDraw

from crowd3d.exif import UnreadableImage, read_photo_meta
from crowd3d.ingest import R_TOO_LARGE, ingest_upload, make_processed_copy
from crowd3d.records import PHOTO_ACCEPTED, PHOTO_REJECTED, Landmark
from crowd3d.store import Store
from crowd3d.synthetic import CAMERA_RADIUS_M

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


def ingest(store, settings, ds, entry, landmark, user_id="user1", data=None):
    data = ds.read(entry) if data is None else data
    return ingest_upload(store, settings, filename=entry["file"], data=data, user_id=user_id, landmark=landmark)


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

    assert (out.accepted, out.reasons, out.warnings, out.filename) == (True, [], [], entry["file"])
    p = out.photo
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
    out = ingest_upload(store, settings, filename="notes.txt", data=b"not an image", user_id="u", landmark=None)
    assert (out.accepted, out.reasons) == (False, ["unreadable_image"])
    p = store.get_photo(out.photo.id)
    assert (p.status, p.landmark_id, p.width, p.phash) == (PHOTO_REJECTED, None, None, None)


def test_decompression_bomb_is_unreadable_not_a_crash(store, settings, monkeypatch):
    data = jpeg((64, 64), fmt="PNG")
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    out = ingest_upload(store, settings, filename="bomb.png", data=data, user_id="u", landmark=None)
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
    out = ingest_upload(store, settings, filename="../../../etc/passwd.jpg", data=ds.read(entry), user_id="u",
                        landmark=landmark)
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
    out = ingest_upload(store, settings, filename="flat.png", data=data, user_id="u", landmark=None)
    assert (out.accepted, out.photo, out.reasons) == (False, None, [R_TOO_LARGE])


def test_postscript_is_never_handed_to_a_decoder(store, settings):
    # Pillow renders EPS by running Ghostscript on the uploaded PostScript.
    eps = (b"%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 800 600\n"
           b"newpath 0 0 moveto 800 600 lineto stroke\nshowpage\n%%EOF\n")
    with pytest.raises(UnreadableImage):
        read_photo_meta(eps)
    out = ingest_upload(store, settings, filename="x.eps", data=eps, user_id="u", landmark=None)
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
