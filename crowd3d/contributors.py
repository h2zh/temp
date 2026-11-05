"""Contributor registration (D33): the user agreement is accepted once, at registration, not with every upload.

A contributor registers with a nickname and by agreeing to the current user agreement (records.TERMS_CURRENT); the
store keeps the nickname, the version accepted and when, and only the SHA-256 of a secret contributor token, which is
handed out once and kept by the contributor's browser. Uploads carry the token (crowd3d.api: in the POST body only)
and take their nickname and their consent from the record. Not an account system: no password, no email, and
nicknames are not unique (as before D33). Nothing here logs: a token must never reach a log.
"""

from __future__ import annotations

import hashlib
import re
import secrets
import time
import unicodedata
import uuid      # its own import: tests/test_recon_golden.py monkeypatches crowd3d.ingest's uuid for the photo ids

from .records import TERMS_CURRENT, Contributor
from .store import Store

MAX_NICKNAME_CHARS = 64
TOKEN_PATTERN = r"^[0-9a-f]{64}$"
_TOKEN = re.compile(TOKEN_PATTERN)


def new_token() -> str:
    """A secret contributor token: 256 random bits, 64 lowercase hex characters."""
    return secrets.token_hex(32)


def token_hash(token: str) -> str:
    """What the store keeps of a token: its SHA-256 hex digest. A plain hash is enough for a 256-bit random secret
    (nothing to guess); deliberately not keyed by the public-id salt (consent.py), so a lost salt de-registers no one."""
    return hashlib.sha256(token.encode("ascii")).hexdigest()


def clean_nickname(nickname: object) -> str | None:
    """The nickname as stored (whitespace stripped), or None unless it is 1-MAX_NICKNAME_CHARS characters without
    control characters (the D32 user_id rule). Any script is fine."""
    if not isinstance(nickname, str):
        return None
    nickname = nickname.strip()
    if not 1 <= len(nickname) <= MAX_NICKNAME_CHARS or any(unicodedata.category(c) == "Cc" for c in nickname):
        return None
    return nickname


def register(store: Store, nickname: object, terms: object) -> tuple[Contributor, str]:
    """Register a contributor who agreed to the user agreement version terms; returns the record and the token, the
    only time the token exists outside the contributor's browser. Raises ValueError, before anything is stored,
    unless terms is the current version and the nickname is clean (clean_nickname)."""
    if terms != TERMS_CURRENT:
        raise ValueError("registering needs agreeing to the current user agreement")
    clean = clean_nickname(nickname)
    if clean is None:
        raise ValueError(f"a nickname is 1-{MAX_NICKNAME_CHARS} characters without control characters")
    token, now = new_token(), time.time()
    contributor = Contributor(id=uuid.uuid4().hex, nickname=clean, token_hash=token_hash(token), terms=TERMS_CURRENT,
                              terms_accepted_at=now, registered_at=now)
    return store.add_contributor(contributor), token


def authenticate(store: Store, token: object) -> Contributor | None:
    """The contributor the token belongs to; None for no token, a non-string, a malformed or an unknown one."""
    if not isinstance(token, str) or not _TOKEN.fullmatch(token):
        return None
    return store.contributor_by_token_hash(token_hash(token))


def accept(store: Store, contributor: Contributor, terms: object) -> Contributor:
    """The contributor accepts the user agreement version terms (a new version asks them once; no new nickname).
    Raises ValueError unless it is the current version."""
    if terms != TERMS_CURRENT:
        raise ValueError("only the current user agreement can be accepted")
    return store.accept_terms(contributor.id, TERMS_CURRENT, time.time())
