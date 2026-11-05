"""Contributor registration (D33, crowd3d.contributors): the user agreement is accepted once, at registration; the
store keeps only a hash of the contributor token."""

import json
import logging
import re
import sqlite3
import time

import pytest

from crowd3d import contributors
from crowd3d.config import Settings
from crowd3d.records import PHOTO_ACCEPTED, TERMS_CURRENT, TERMS_V1, Contributor, PhotoRecord
from crowd3d.store import MIGRATION_D32_TERMS_KEY, NotFound, Store

BAD_NICKNAMES = ["", "   ", "u" * 65, "bell\x07", "two\nlines", "nul\x00byte", None, 5]
BAD_TERMS = [None, "", "  ", "terms-v0", "true", "on", f" {TERMS_CURRENT}", TERMS_CURRENT.upper()]


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def store(settings):
    s = Store(settings)
    yield s
    s.close()


def stored_contributors(settings) -> int:
    with sqlite3.connect(settings.db_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM contributors").fetchone()[0]


def test_a_token_is_64_hex_characters_and_every_registration_gets_its_own(store, settings):
    tokens = set()
    for n in range(100):
        c, token = contributors.register(store, f"u{n}", TERMS_CURRENT)
        assert re.fullmatch(contributors.TOKEN_PATTERN, token)
        assert c.token_hash == contributors.token_hash(token) != token and len(c.token_hash) == 64
        assert (c.nickname, c.terms, c.terms_accepted_at) == (f"u{n}", TERMS_CURRENT, c.registered_at)
        tokens.add(token)
    assert len(tokens) == 100 and stored_contributors(settings) == 100


def test_the_store_keeps_the_hash_never_the_token(store, settings):
    """Neither the database file nor its write-ahead log holds a token's bytes (as text or as the raw 32 bytes);
    the hash is there."""
    registered = [contributors.register(store, nick, TERMS_CURRENT) for nick in ("alice", "小明", "bob")]
    blobs = [p.read_bytes() for p in settings.db_path.parent.glob(settings.db_path.name + "*")]
    assert any(p.name.endswith("-wal") for p in settings.db_path.parent.glob(settings.db_path.name + "*"))
    for c, token in registered:
        for blob in blobs:
            assert token.encode() not in blob and bytes.fromhex(token) not in blob
        assert any(c.token_hash.encode() in blob for blob in blobs)
    with sqlite3.connect(settings.db_path) as conn:
        columns = [r[1] for r in conn.execute("PRAGMA table_info(contributors)")]
    assert columns == ["id", "nickname", "token_hash", "terms", "terms_accepted_at", "registered_at"]


def test_authenticate_finds_the_contributor_and_nothing_else(store):
    c, token = contributors.register(store, "alice", TERMS_CURRENT)
    assert contributors.authenticate(store, token) == c
    unknown = contributors.new_token()
    for bad in (None, "", " ", 123, b"\x00", [token], token.upper(), token[:-1], token + "0", token + "\n",
                f" {token}", "g" * 64, c.token_hash, c.id, unknown):
        assert contributors.authenticate(store, bad) is None, bad


@pytest.mark.parametrize("terms", BAD_TERMS)
def test_register_refuses_anything_but_the_current_agreement_before_storing(store, settings, terms):
    with pytest.raises(ValueError):
        contributors.register(store, "alice", terms)
    assert stored_contributors(settings) == 0


@pytest.mark.parametrize("nickname", BAD_NICKNAMES)
def test_register_refuses_a_bad_nickname_before_storing(store, settings, nickname):
    assert contributors.clean_nickname(nickname) is None
    with pytest.raises(ValueError):
        contributors.register(store, nickname, TERMS_CURRENT)
    assert stored_contributors(settings) == 0


def test_nicknames_are_stripped_any_script_and_not_unique(store, settings):
    assert contributors.clean_nickname("  小明  ") == "小明"
    assert contributors.clean_nickname("u" * 64) == "u" * 64
    a, token_a = contributors.register(store, "  小明  ", TERMS_CURRENT)
    b, token_b = contributors.register(store, "小明", TERMS_CURRENT)
    assert a.nickname == b.nickname == "小明" and a.id != b.id and token_a != token_b
    assert stored_contributors(settings) == 2


def test_accept_records_the_current_version_and_when(store, settings):
    c, token = contributors.register(store, "alice", TERMS_CURRENT)
    with sqlite3.connect(settings.db_path) as conn:
        conn.execute("UPDATE contributors SET terms = 'terms-v0', terms_accepted_at = 1.0 WHERE id = ?", (c.id,))
    old = contributors.authenticate(store, token)
    assert (old.terms, old.terms_accepted_at) == ("terms-v0", 1.0)
    for bad in BAD_TERMS:
        with pytest.raises(ValueError):
            contributors.accept(store, old, bad)
    assert contributors.authenticate(store, token) == old                  # a refused accept changes nothing
    before = time.time()
    new = contributors.accept(store, old, TERMS_CURRENT)
    assert new.terms == TERMS_CURRENT and new.terms_accepted_at >= before
    assert (new.id, new.nickname, new.token_hash, new.registered_at) == (c.id, c.nickname, c.token_hash,
                                                                          c.registered_at)
    assert contributors.authenticate(store, token) == new
    with pytest.raises(NotFound):
        store.accept_terms("nobody", TERMS_CURRENT, 1.0)


def test_nothing_is_logged(store, caplog):
    caplog.set_level(logging.DEBUG)
    c, token = contributors.register(store, "alice", TERMS_CURRENT)
    contributors.authenticate(store, token)
    contributors.accept(store, c, TERMS_CURRENT)
    assert token not in caplog.text and c.token_hash not in caplog.text


def test_a_database_from_before_d33_gains_the_table_and_keeps_its_photos(settings):
    """An old-schema database (no contributors table, no photos.contributor_id; tests/test_store.py's pattern): both
    are added on open, old rows keep their user_id with contributor_id NULL, and the D32 migration is unchanged."""
    settings.ensure_dirs()
    conn = sqlite3.connect(settings.db_path)
    conn.executescript("""
        CREATE TABLE landmarks (id TEXT PRIMARY KEY, name TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
                                radius_m REAL NOT NULL, created_at REAL NOT NULL);
        CREATE TABLE photos (id TEXT PRIMARY KEY, landmark_id TEXT REFERENCES landmarks(id), user_id TEXT NOT NULL,
            status TEXT NOT NULL, uploaded_at REAL NOT NULL, reasons TEXT NOT NULL, warnings TEXT NOT NULL,
            lat REAL, lon REAL, alt REAL, heading_deg REAL, bearing_deg REAL, distance_m REAL, sector INTEGER,
            width INTEGER, height INTEGER, focal_35mm REAL, phash TEXT, original_path TEXT, processed_path TEXT,
            captured_at TEXT, gps_accuracy_m REAL, outline_distance_m REAL, subject_share REAL, heading_ref TEXT,
            location_consent TEXT);
        INSERT INTO landmarks VALUES ('tower', 'Tower', 31.0, 121.0, 150.0, 1.0);
        INSERT INTO photos (id, landmark_id, user_id, status, uploaded_at, reasons, warnings)
            VALUES ('p', 'tower', '小明', 'accepted', 1.0, '[]', '[]');
        INSERT INTO photos (id, landmark_id, user_id, status, uploaded_at, reasons, warnings, location_consent)
            VALUES ('r', 'tower', 'bob', 'rejected', 2.0, '["too_far"]', '[]', 'share-location-v1');
    """)
    conn.close()
    s = Store(settings)
    try:
        p, r = s.get_photo("p"), s.get_photo("r")
        assert (p.user_id, p.contributor_id, p.location_consent) == ("小明", None, TERMS_V1)    # D32 migration
        assert (r.user_id, r.contributor_id, r.location_consent) == ("bob", None, None)
        assert json.loads(s.get_meta(MIGRATION_D32_TERMS_KEY))["photos_agreed"] == 1
        assert s.contributors("tower") == ["小明"]
        c, token = contributors.register(s, "carol", TERMS_CURRENT)
        assert contributors.authenticate(s, token) == c
        s.add_photo(PhotoRecord(id="n", landmark_id="tower", user_id=c.nickname, status=PHOTO_ACCEPTED,
                                uploaded_at=3.0, location_consent=c.terms, contributor_id=c.id))
        assert s.get_photo("n").contributor_id == c.id
        assert s.contributors("tower") == ["carol", "小明"]
    finally:
        s.close()
    s = Store(settings)                                                     # a second open changes nothing
    try:
        assert s.get_photo("p").contributor_id is None and s.get_photo("n").contributor_id == c.id
        assert contributors.authenticate(s, token) == c
    finally:
        s.close()


def test_a_contributor_record_holds_no_token(store):
    c, token = contributors.register(store, "alice", TERMS_CURRENT)
    assert isinstance(c, Contributor) and token not in repr(c) and token not in json.dumps(c.__dict__)
