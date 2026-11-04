"""The deployment's secret for location sharing (D30): the public id another user sees for a shared photo, and the
uploader's token to withdraw the consent.

The secret salt lives in <data_dir>/public_id.salt (256 random bits, mode 0600, made on first use), outside the
served directories (models_dir, the web dir), or in $CROWD3D_PUBLIC_ID_SALT. It never leaves the server, so neither a
public id nor a manage token can be computed from a photo id without it. Losing or replacing the file changes every
public id and invalidates every manage token, so an empty file is an error, not a reason to make a new one.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import os
import secrets
import stat
from pathlib import Path

PUBLIC_ID_SALT_ENV = "CROWD3D_PUBLIC_ID_SALT"
PUBLIC_ID_SALT_FILE = "public_id.salt"
MIN_SALT_CHARS = 32
PUBLIC_ID_HEX = 12
MANAGE_TOKEN_PATTERN = r"^[0-9a-f]{64}$"

log = logging.getLogger(__name__)


def salt_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / PUBLIC_ID_SALT_FILE


def deployment_salt(data_dir: Path | str) -> str:
    """The secret this deployment keys public ids and manage tokens with: $CROWD3D_PUBLIC_ID_SALT (at least 32
    characters), else <data_dir>/public_id.salt, written on first use (64 hex digits, mode 0600). A salt file other
    users can read is narrowed to 0600."""
    env = os.environ.get(PUBLIC_ID_SALT_ENV, "").strip()
    if env:
        if len(env) < MIN_SALT_CHARS:
            raise ValueError(f"${PUBLIC_ID_SALT_ENV} must have at least {MIN_SALT_CHARS} characters")
        return env
    f = salt_path(data_dir)
    try:
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        mode = stat.S_IMODE(f.stat().st_mode)
        if mode & 0o077:
            log.warning("%s was readable by other users (mode %o): narrowed to 0600", f, mode)
            os.chmod(f, 0o600)
        value = f.read_text(encoding="ascii").strip()
        if len(value) < MIN_SALT_CHARS:
            raise RuntimeError(f"{f} holds no usable salt: restore it from a backup (a new one changes every public "
                               f"id and invalidates every manage token), or delete it to start over") from None
        return value
    value = secrets.token_hex(32)
    with os.fdopen(fd, "w", encoding="ascii") as h:
        h.write(value + "\n")
    return value


def _hmac(salt: str, message: str) -> str:
    if not salt:
        raise ValueError("needs the deployment's secret salt (deployment_salt)")
    return hmac.new(salt.encode(), message.encode(), hashlib.sha256).hexdigest()


def public_id(photo_id: str, salt: str) -> str:
    """The id other people see for a shared photo: HMAC-SHA256(salt, photo_id), first 12 hex digits."""
    return _hmac(salt, photo_id)[:PUBLIC_ID_HEX]


def manage_token(photo_id: str, salt: str) -> str:
    """The uploader's token to withdraw the photo's location consent: HMAC-SHA256(salt, "manage:" + photo_id), 64
    hex digits. Handed out once, in the upload's response."""
    return _hmac(salt, "manage:" + photo_id)


def check_manage_token(photo_id: str, token: str, salt: str) -> bool:
    return hmac.compare_digest(token.encode(), manage_token(photo_id, salt).encode())
