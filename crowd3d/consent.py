"""The deployment's secret for location sharing (D30): the public id another user sees for a shared photo, and the
uploader's token to withdraw the consent.

The secret salt lives in <data_dir>/public_id.salt (256 random bits, mode 0600, made on first use), outside the
served directories (models_dir, the web dir), or in $CROWD3D_PUBLIC_ID_SALT. It never leaves the server, so neither a
public id nor a manage token can be computed from a photo id without it.

A different salt changes every public id and invalidates every manage token: the photos would stay shared and nobody
could withdraw them. So the salt is tied to the store it keys (bound_salt): the first start records a check value,
HMAC(salt, "salt-check"), in the store's meta table, and a later start with another salt (an env salt set or changed,
the data dir restored on a host without it, the file deleted or replaced) refuses to start while any photo is shared.
An empty or short salt file is an error too, never a reason to make a new one. When the old salt is lost for good, the
operator runs `python -m crowd3d.consent forget-lost-salt`: it withdraws every consent and keys the store with the salt
in use now.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import logging
import os
import secrets
import stat
import sys
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import Settings
    from .store import Store

PUBLIC_ID_SALT_ENV = "CROWD3D_PUBLIC_ID_SALT"
PUBLIC_ID_SALT_FILE = "public_id.salt"
MIN_SALT_CHARS = 32
PUBLIC_ID_HEX = 12
MANAGE_TOKEN_PATTERN = r"^[0-9a-f]{64}$"
SALT_CHECK_KEY = "public_id_salt_check"     # the store's meta key for HMAC(salt, "salt-check")
FORGET_COMMAND = "python -m crowd3d.consent forget-lost-salt"

log = logging.getLogger(__name__)


def salt_path(data_dir: Path | str) -> Path:
    return Path(data_dir) / PUBLIC_ID_SALT_FILE


class SaltMismatch(RuntimeError):
    """The salt in use is not the one the store's location consents were given under (bound_salt)."""


def deployment_salt(data_dir: Path | str, *, create: bool = True) -> str:
    """The secret this deployment keys public ids and manage tokens with: $CROWD3D_PUBLIC_ID_SALT (at least 32
    characters), else <data_dir>/public_id.salt, written on first use (64 hex digits, mode 0600) when create, else
    FileNotFoundError. A salt file other users can read is narrowed to 0600. Use bound_salt to key a store."""
    env = os.environ.get(PUBLIC_ID_SALT_ENV, "").strip()
    if env:
        if len(env) < MIN_SALT_CHARS:
            raise ValueError(f"${PUBLIC_ID_SALT_ENV} must have at least {MIN_SALT_CHARS} characters")
        return env
    f = salt_path(data_dir)
    if not create and not f.exists():
        raise FileNotFoundError(f"{f} is missing and ${PUBLIC_ID_SALT_ENV} is not set")
    try:
        fd = os.open(f, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        mode = stat.S_IMODE(f.stat().st_mode)
        if mode & 0o077:
            log.warning("%s was readable by other users (mode %o): narrowed to 0600", f, mode)
            os.chmod(f, 0o600)
        value = f.read_text(encoding="ascii").strip()
        if len(value) < MIN_SALT_CHARS:
            raise RuntimeError(f"{f} holds no usable salt: restore it from a backup (another salt changes every "
                               f"public id and invalidates every manage token). If it is lost for good, "
                               f"`{FORGET_COMMAND}` withdraws every photo's location consent and starts a new "
                               f"one") from None
        return value
    value = secrets.token_hex(32)
    with os.fdopen(fd, "w", encoding="ascii") as h:
        h.write(value + "\n")
    return value


def salt_check(salt: str) -> str:
    """The value the store keeps to recognise its salt: HMAC-SHA256(salt, "salt-check"). It tells nothing about the
    salt, a public id or a manage token."""
    return _hmac(salt, "salt-check")


def _salt_source(data_dir: Path | str) -> str:
    return f"${PUBLIC_ID_SALT_ENV}" if os.environ.get(PUBLIC_ID_SALT_ENV, "").strip() else str(salt_path(data_dir))


def bound_salt(data_dir: Path | str, store: Store) -> str:
    """deployment_salt, checked against the store it keys (D30: a manage token must keep working as long as its
    photo is shared). The first start records salt_check in the store (a store with no check yet takes the salt in
    use: only a server that recorded its check hands out tokens). Later, a salt that does not match, or a
    missing salt file with no $CROWD3D_PUBLIC_ID_SALT, raises SaltMismatch while any photo is shared (no new salt
    file is written then); with no photo shared nothing can be lost, and the salt in use is adopted, with a
    warning."""
    recorded = store.get_meta(SALT_CHECK_KEY)
    try:
        salt = deployment_salt(data_dir, create=recorded is None)
    except FileNotFoundError:
        salt = None
    if recorded is None:
        recorded = store.setdefault_meta(SALT_CHECK_KEY, salt_check(salt))
    if salt is not None and hmac.compare_digest(recorded, salt_check(salt)):
        return salt
    shared = store.location_consent_count()
    if shared:
        used = "no salt (the file is missing)" if salt is None else f"the salt in {_salt_source(data_dir)}"
        photos = "1 photo" if shared == 1 else f"{shared} photos"
        raise SaltMismatch(
            f"refusing to start: {used} is not the salt the location consents of {photos} in {Path(data_dir)} "
            f"were given under, so every manage token handed out would be refused while those photos stay shared. "
            f"Restore {salt_path(data_dir)} from a backup, or set ${PUBLIC_ID_SALT_ENV} back to its old value. If "
            f"the old salt is lost for good, `{FORGET_COMMAND} --data-dir {data_dir}` withdraws those consents "
            f"(nobody could withdraw them any more) and keys the data dir with the salt in use now.")
    if salt is None:
        salt = deployment_salt(data_dir)
    log.warning("the public-id salt changed (%s); no photo's location is shared, so it is adopted",
                _salt_source(data_dir))
    store.set_meta(SALT_CHECK_KEY, salt_check(salt))
    return salt


def forget_lost_salt(settings: Settings) -> int:
    """The operator's way out when the salt the consents were given under is lost for good: withdraw every photo's
    location consent (its manage token can no longer do it, so it must not stay shared), replace an unusable salt
    file, and key the store with the salt in use now. Refuses while a server holds the data dir, or when the salt
    in use still matches. Returns how many consents were withdrawn."""
    from .api import data_dir_lock      # crowd3d.api imports this module
    from .store import Store

    if not settings.db_path.is_file():
        raise RuntimeError(f"{settings.data_dir} holds no Crowd3D database")
    with data_dir_lock(settings):
        store = Store(settings)
        try:
            recorded = store.get_meta(SALT_CHECK_KEY)
            try:
                salt = deployment_salt(settings.data_dir, create=False)
            except (FileNotFoundError, RuntimeError):
                salt = None
            if salt is not None and (recorded is None or hmac.compare_digest(recorded, salt_check(salt))):
                raise RuntimeError(f"the salt in {_salt_source(settings.data_dir)} is the one this data dir's "
                                   f"consents were given under: nothing to forget")
            withdrawn = store.clear_location_consents()
            if salt is None:          # no $CROWD3D_PUBLIC_ID_SALT, and the file is missing, empty or short
                salt_path(settings.data_dir).unlink(missing_ok=True)
                salt = deployment_salt(settings.data_dir)
            store.set_meta(SALT_CHECK_KEY, salt_check(salt))
        finally:
            store.close()
    log.warning("withdrew the location consent of %d photos: the salt they were given under is lost", withdrawn)
    return withdrawn


def _hmac(salt: str, message: str) -> str:
    if not salt:
        raise ValueError("needs the deployment's secret salt (deployment_salt)")
    return hmac.new(salt.encode(), message.encode(), hashlib.sha256).hexdigest()


def public_id(photo_id: str, salt: str) -> str:
    """The id other people see for a shared photo: HMAC-SHA256(salt, photo_id), first 12 hex digits."""
    return _hmac(salt, photo_id)[:PUBLIC_ID_HEX]


def manage_token(photo_id: str, salt: str) -> str:
    """The uploader's token to withdraw the photo's location consent: HMAC-SHA256(salt, "manage:" + photo_id), 64
    hex digits. Handed out in the upload's response, and again to the registration the photo was uploaded under
    whenever it asks (crowd3d.api OWN_PHOTOS_ROUTE, D34): the response may never have arrived."""
    return _hmac(salt, "manage:" + photo_id)


def check_manage_token(photo_id: str, token: str, salt: str) -> bool:
    return hmac.compare_digest(token.encode(), manage_token(photo_id, salt).encode())


def main(argv: list[str] | None = None) -> int:
    """CLI: python -m crowd3d.consent forget-lost-salt [--data-dir data]"""
    from .config import Settings

    ap = argparse.ArgumentParser(prog="python -m crowd3d.consent", description="Crowd3D location consent (D30)")
    sub = ap.add_subparsers(dest="command", required=True)
    forget = sub.add_parser("forget-lost-salt", help="the public-id salt is lost for good: withdraw every photo's "
                            "location consent and key the data dir with the salt in use now")
    forget.add_argument("--data-dir", type=Path, default=None, help="default: $CROWD3D_DATA_DIR or ./data")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    settings = Settings()
    if args.data_dir is not None:
        settings = replace(settings, data_dir=args.data_dir)
    try:
        withdrawn = forget_lost_salt(settings)
    except (RuntimeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    print(f"withdrew the location consent of {withdrawn} photos; {settings.data_dir} now uses the salt in "
          f"{_salt_source(settings.data_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
